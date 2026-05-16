"""R2 backup + 48 h preview window for Live Studio streams.

Lifecycle
---------
1. **During upload**: chunks land in the local temp file (handled by
   ``uploads.py``).
2. **During broadcast**: ffmpeg streams from the same temp file.
3. **Right after broadcast** (or on every chunk completion if we want
   true crash recovery): the orchestrator hands off the file to this
   module to upload to R2.
4. **Crash recovery on backend startup**: ``recover_pending_streams()``
   scans for ``status in ("streaming", "starting", "uploading")`` rows
   that have an R2 backup and re-spawns the worker pointed at the R2
   copy (downloads to a fresh temp file first).
5. **Auto-delete after 48 h**: ``run_expiry_sweep()`` is called from a
   scheduler (or admin endpoint). Any LiveStream whose
   ``backup_expires_at`` is in the past gets its R2 object deleted +
   ``backup_url`` cleared. The DB row itself stays forever for history.

Why "after broadcast" not "during"
-----------------------------------
True concurrent multipart-streaming-to-R2 alongside ffmpeg is doable
with S3 multipart APIs but adds substantial complexity (chunk
coordination, abort on failure, partial cleanup). For v1, the simpler
"upload the temp file right after broadcast completes" gives crash
recovery for everything EXCEPT the broadcast itself — if the backend
crashes mid-stream, you re-upload the source and retry. R2 saves the
post-broadcast preview window, not in-flight bytes.

The chunk-level concurrent backup is wired as a separate flag we can
flip in Phase 6.5 if needed.
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


logger = logging.getLogger("kaizer.live_studio.r2_backup")


# 48 h preview window per the user's spec. Override via env for
# testing (e.g. 0.0001 hr to immediately expire).
PREVIEW_HOURS = float(
    (os.environ.get("KAIZER_LIVE_STUDIO_PREVIEW_HOURS") or "48").strip()
)


def _key_for_stream(stream_id: int, user_id: int) -> str:
    """Stable R2 key. Forward slashes regardless of OS. Includes
    the user_id so admin can audit + scope per-user lifecycle rules
    in R2's bucket policy."""
    return f"live-studio/{user_id}/stream-{stream_id}.mp4"


def upload_for_preview(
    *, stream_id: int, user_id: int, local_path: str,
) -> Optional[dict]:
    """Upload the broadcast's source file to R2 with a 48 h preview
    expiry. Returns ``{url, key, expires_at}`` on success, None on
    soft failure (caller logs but doesn't crash the broadcast).

    Soft-fails because YouTube is the durable copy of the actual
    video — the R2 preview is best-effort.
    """
    if not local_path or not os.path.isfile(local_path):
        logger.warning(f"[live_studio/backup] local file gone, skip R2 upload: {local_path}")
        return None
    try:
        from pipeline_core.storage import get_storage_provider
    except ImportError:
        logger.warning("[live_studio/backup] storage module missing — R2 skipped")
        return None

    provider = get_storage_provider()
    key = _key_for_stream(stream_id, user_id)
    try:
        obj = provider.upload(local_path, key, content_type="video/mp4")
    except Exception as exc:
        logger.warning(f"[live_studio/backup] upload failed for stream={stream_id}: {exc}")
        return None

    expires_at = datetime.now(timezone.utc) + timedelta(hours=PREVIEW_HOURS)
    return {
        "url":        obj.url,
        "key":        key,
        "expires_at": expires_at,
    }


def delete_backup(stream_id: int, user_id: int, *, key: Optional[str] = None) -> bool:
    """Remove the R2 object. Safe to call multiple times (soft-fails
    on already-gone objects). Returns True on success."""
    try:
        from pipeline_core.storage import get_storage_provider
    except ImportError:
        return False
    provider = get_storage_provider()
    k = key or _key_for_stream(stream_id, user_id)
    try:
        provider.delete(k)
        return True
    except Exception as exc:
        logger.warning(f"[live_studio/backup] delete failed for {k}: {exc}")
        return False


def download_for_recovery(stream_id: int, user_id: int, dest_path: str,
                          *, key: Optional[str] = None) -> bool:
    """Pull the R2 preview back to a local temp file so a re-spawned
    worker can resume the broadcast. Returns True on success."""
    try:
        from pipeline_core.storage import get_storage_provider
    except ImportError:
        return False
    provider = get_storage_provider()
    k = key or _key_for_stream(stream_id, user_id)
    try:
        provider.download(k, dest_path)
        return os.path.isfile(dest_path) and os.path.getsize(dest_path) > 1000
    except Exception as exc:
        logger.warning(f"[live_studio/backup] download failed for {k}: {exc}")
        return False


# ─── Scheduled tasks ─────────────────────────────────────────────

def run_expiry_sweep() -> dict:
    """Find LiveStream rows whose backup has expired + delete the R2
    object + clear ``backup_url`` / ``backup_key``. Returns a stats
    dict for logging / admin display.

    Called by a daily scheduler (Kaizer's existing learning/scheduler
    or a simple thread launched from main.py startup).
    """
    from database import SessionLocal
    import models

    now = datetime.now(timezone.utc)
    sess = SessionLocal()
    deleted = 0
    failed = 0
    skipped = 0
    try:
        rows = (
            sess.query(models.LiveStream)
                .filter(
                    models.LiveStream.backup_url.isnot(None),
                    models.LiveStream.backup_expires_at.isnot(None),
                    models.LiveStream.backup_expires_at < now,
                )
                .all()
        )
        for row in rows:
            if not row.backup_key:
                # Should never happen; skip + clear to avoid loops.
                row.backup_url = None
                row.backup_expires_at = None
                sess.commit()
                skipped += 1
                continue
            ok = delete_backup(row.id, row.user_id, key=row.backup_key)
            if ok:
                row.backup_url = None
                row.backup_key = None
                row.backup_expires_at = None
                sess.commit()
                deleted += 1
            else:
                failed += 1
    finally:
        sess.close()
    logger.info(
        f"[live_studio/backup] sweep: deleted={deleted} failed={failed} skipped={skipped}"
    )
    return {"deleted": deleted, "failed": failed, "skipped": skipped}


def recover_pending_streams() -> dict:
    """Backend-startup hook: find streams stuck in non-terminal states
    and re-spawn workers for them. Called by main.py at boot.

    A stream qualifies for recovery when ALL of:
      - status in (uploading, starting, provisioning, streaming)
      - backup_url is set (we have an R2 copy to resume from)
      - backup_expires_at is in the future

    Without an R2 backup we can't recover — the temp file was lost
    when the process died, so we mark those rows ``failed`` with a
    clear error so the user can re-upload.
    """
    from database import SessionLocal
    from live_studio import orchestrator as live_orch
    from live_studio import uploads as live_uploads
    import models

    now = datetime.now(timezone.utc)
    sess = SessionLocal()
    recovered = 0
    abandoned = 0
    try:
        active = {"uploading", "starting", "provisioning", "streaming"}
        rows = (
            sess.query(models.LiveStream)
                .filter(models.LiveStream.status.in_(active))
                .all()
        )
        for row in rows:
            # Have an R2 backup AND it hasn't expired yet?
            has_backup = (
                row.backup_url and row.backup_key
                and row.backup_expires_at and row.backup_expires_at > now
            )
            if not has_backup:
                row.status = "failed"
                row.error  = ("backend restarted mid-broadcast; no R2 backup "
                              "available — please re-upload and retry")
                row.finished_at = now
                sess.commit()
                abandoned += 1
                continue

            # Download R2 copy to the canonical temp path.
            dest = live_uploads.upload_path_for(row.id)
            ok = download_for_recovery(row.id, row.user_id,
                                       dest_path=dest, key=row.backup_key)
            if not ok:
                row.status = "failed"
                row.error  = "R2 backup present but download failed; please retry"
                row.finished_at = now
                sess.commit()
                abandoned += 1
                continue

            # Reset to "starting" so the worker re-enters the normal
            # provision + push flow. The YT broadcast may still be
            # active — obtain_rtmp_target will mint a fresh one
            # (the abandoned one auto-stops after a few minutes of
            # silence).
            row.status   = "starting"
            row.message  = "recovering after backend restart"
            row.upload_done = True
            row.upload_path = dest
            row.upload_bytes = os.path.getsize(dest)
            sess.commit()
            live_orch.kick_off(row.id)
            recovered += 1
    finally:
        sess.close()
    if recovered or abandoned:
        logger.info(f"[live_studio/backup] recovery: spawned={recovered} abandoned={abandoned}")
    return {"recovered": recovered, "abandoned": abandoned}
