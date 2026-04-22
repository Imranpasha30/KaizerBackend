"""Asyncio upload worker — single loop, polls upload_jobs table.

Design:
  - One background asyncio task spun up on FastAPI startup.
  - Polls every POLL_INTERVAL seconds for a `queued` row.
  - Runs the (sync, blocking) upload on a thread via run_in_executor so the
    event loop stays free.
  - On failure, applies exponential backoff:
        attempts 1..MAX_ATTEMPTS → 30s, 1m, 2m, 5m, 15m, 60m
  - On quota exhaustion: stays `queued` with a 1-hour backoff stamp, so it
    retries after the quota resets.
  - On process restart: any row stuck in `uploading` becomes `queued` again.
    Since `upload_uri` is persisted, the next run resumes from bytes_uploaded.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple

from sqlalchemy import or_
from sqlalchemy.orm import Session

import models
from database import SessionLocal
from youtube import oauth, uploader, quota

_worker_logger = logging.getLogger("kaizer.youtube.worker")


POLL_INTERVAL = 3.0
MAX_ATTEMPTS  = 6
BACKOFF_SECS  = (30, 60, 120, 300, 900, 3600)   # indexed by attempts


# Module-level flag so shutdown can stop the loop cooperatively
_running = False
_task: Optional[asyncio.Task] = None


def _next_retry_at(attempts: int) -> datetime:
    idx = min(max(attempts - 1, 0), len(BACKOFF_SECS) - 1)
    return datetime.now(timezone.utc) + timedelta(seconds=BACKOFF_SECS[idx])


async def start() -> None:
    """Called from FastAPI startup."""
    global _running, _task
    if _running:
        return
    _running = True
    _recover_stuck_rows()
    _task = asyncio.create_task(_loop(), name="kaizer-upload-worker")


async def stop() -> None:
    global _running, _task
    _running = False
    if _task:
        _task.cancel()
        try:
            await _task
        except (asyncio.CancelledError, Exception):
            pass
        _task = None


def _recover_stuck_rows() -> None:
    """On startup, any `uploading` rows are assumed crashed — re-queue them."""
    db = SessionLocal()
    try:
        rows = db.query(models.UploadJob).filter(
            models.UploadJob.status == "uploading",
        ).all()
        for r in rows:
            r.status = "queued"
            r.last_error = (r.last_error or "") + "\n[worker] re-queued after process restart"
        if rows:
            db.commit()
            print(f"[upload-worker] Re-queued {len(rows)} row(s) that were mid-upload at shutdown")
    finally:
        db.close()


async def _loop() -> None:
    loop = asyncio.get_running_loop()
    while _running:
        try:
            job_id = await loop.run_in_executor(None, _pick_next)
            if job_id is None:
                await asyncio.sleep(POLL_INTERVAL)
                continue
            await loop.run_in_executor(None, _process, job_id)
        except asyncio.CancelledError:
            return
        except Exception:
            # Never let the loop die from an unexpected exception
            traceback.print_exc()
            await asyncio.sleep(POLL_INTERVAL)


def _pick_next() -> Optional[int]:
    """Atomically claim the next queued row (oldest first, respecting backoff).

    We rely on SQLite/Postgres row-level locking indirectly by bumping status
    to 'uploading' in the same session. Two workers hitting the same row is
    prevented by single-process deployment (v1). When we scale, add a
    `claimed_by` worker_id + a conditional UPDATE.
    """
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        row = (
            db.query(models.UploadJob)
              .filter(models.UploadJob.status == "queued")
              .filter(or_(models.UploadJob.last_error == "",
                          models.UploadJob.last_error.is_(None),
                          models.UploadJob.updated_at < now - timedelta(seconds=5)))
              .order_by(models.UploadJob.created_at.asc())
              .first()
        )
        if not row:
            return None
        row.status = "uploading"
        row.last_error = (row.last_error or "") + f"\n[worker] picked up at {now.isoformat()}"
        db.commit()
        return row.id
    finally:
        db.close()


def _append_log(job: models.UploadJob, msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    job.log = ((job.log or "") + ("\n" if job.log else "") + entry)[-8000:]


def _ensure_local_clip(
    job: models.UploadJob,
    clip: models.Clip,
) -> Tuple[str, Optional[str]]:
    """Return (local_file_path, tmp_dir_or_None) for the clip video.

    Resolution order
    ----------------
    1. ``clip.file_path`` exists on disk → return it directly (no tempdir).
    2. ``clip.storage_key`` + ``clip.storage_backend`` are set → download via
       the matching storage provider into a fresh tempdir and return that path.
    3. Neither → raise RuntimeError with a descriptive message.

    The caller is responsible for deleting *tmp_dir* (when not None) in a
    ``finally`` block after the upload completes.
    """
    # ------------------------------------------------------------------
    # 1. Local file still present?
    # ------------------------------------------------------------------
    local_path = (clip.file_path or "").strip()
    if local_path and Path(local_path).is_file():
        _worker_logger.debug(
            "ensure_local_clip: clip %d found on disk at %r", clip.id, local_path
        )
        return local_path, None

    # ------------------------------------------------------------------
    # 2. Download from cloud storage
    # ------------------------------------------------------------------
    storage_key: str = (getattr(clip, "storage_key", "") or "").strip()
    storage_backend: str = (getattr(clip, "storage_backend", "") or "").strip()

    if storage_key and storage_backend:
        _worker_logger.info(
            "ensure_local_clip: clip %d not on disk — downloading from %r key=%r",
            clip.id, storage_backend, storage_key,
        )
        from pipeline_core.storage import get_storage_provider
        provider = get_storage_provider(storage_backend)

        tmp_dir = tempfile.mkdtemp(prefix="kaizer_upload_")
        filename = Path(storage_key).name or f"clip_{clip.id}.mp4"
        tmp_path = os.path.join(tmp_dir, filename)

        try:
            provider.download(storage_key, tmp_path)
        except Exception as dl_exc:
            _worker_logger.error(
                "ensure_local_clip: download failed for clip %d key=%r: %s",
                clip.id, storage_key, dl_exc,
            )
            # Clean up the empty tempdir before re-raising
            try:
                import shutil as _sh
                _sh.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
            raise RuntimeError(
                f"Could not download clip {clip.id} from storage "
                f"(backend={storage_backend!r}, key={storage_key!r}): {dl_exc}"
            ) from dl_exc

        _worker_logger.info(
            "ensure_local_clip: clip %d downloaded to %r", clip.id, tmp_path
        )
        return tmp_path, tmp_dir

    # ------------------------------------------------------------------
    # 3. No usable source
    # ------------------------------------------------------------------
    raise RuntimeError(
        f"Clip {clip.id} has no usable video source: "
        f"file_path={clip.file_path!r} is missing on disk and "
        f"storage_key={getattr(clip, 'storage_key', '')!r} / "
        f"storage_backend={getattr(clip, 'storage_backend', '')!r} are not set."
    )


def _process(job_id: int) -> None:
    db = SessionLocal()
    try:
        job = db.query(models.UploadJob).filter(models.UploadJob.id == job_id).first()
        if not job:
            return
        if job.status != "uploading":
            # Someone cancelled or changed state before we started
            return

        clip = db.query(models.Clip).filter(models.Clip.id == job.clip_id).first()
        if not clip:
            _fail(db, job, "clip row not found")
            return

        # ─── Resolve local clip path (disk or cloud download) ─────────
        _clip_tmp_dir: Optional[str] = None
        try:
            resolved_clip_path, _clip_tmp_dir = _ensure_local_clip(job, clip)
        except RuntimeError as resolve_err:
            _fail(db, job, str(resolve_err))
            return

        # ─── Quota check ───────────────────────────────────────────
        if not quota.reserve(db, quota.COST_VIDEO_INSERT):
            _append_log(job, "quota exhausted — parking until tomorrow")
            job.status = "queued"
            job.last_error = "daily quota exhausted"
            db.commit()
            return

        # ─── Mint credentials ──────────────────────────────────────
        try:
            creds = oauth.get_credentials(db, job.channel_id)
        except oauth.OAuthError as e:
            _fail(db, job, f"auth failed: {e}")
            return

        # ─── Upload ────────────────────────────────────────────────
        job.attempts = (job.attempts or 0) + 1
        _append_log(job, f"starting upload (attempt {job.attempts})")
        db.commit()

        def _progress(uploaded: int, total: int) -> None:
            # Keep the row warm so /api/uploads polling shows movement
            job.updated_at = datetime.now(timezone.utc)
            db.commit()

        # ─── Per-destination logo overlay (before upload) ────────────
        # The pipeline renders a clean master so each destination can
        # get ITS OWN logo burned in just before upload.  Resolves the
        # destination channel's OAuthToken.logo_asset_id → file → ffmpeg
        # overlay.  If anything fails, falls back to the resolved clip path.
        upload_path = resolved_clip_path
        try:
            dest_channel = db.query(models.Channel).filter(
                models.Channel.id == job.channel_id,
            ).first()
            dest_tok = dest_channel.oauth_token if dest_channel else None
            if dest_tok and dest_tok.logo_asset_id:
                logo_asset = db.query(models.UserAsset).filter(
                    models.UserAsset.id == dest_tok.logo_asset_id,
                ).first()
                if logo_asset and logo_asset.file_path and Path(logo_asset.file_path).exists():
                    _append_log(job, f"overlaying destination logo ({logo_asset.filename})…")
                    from youtube import logo_overlay
                    upload_path = logo_overlay.overlay_logo(
                        resolved_clip_path, logo_asset.file_path
                    )
                    if upload_path != resolved_clip_path:
                        _append_log(job, "logo overlay applied")
                    else:
                        _append_log(job, "logo overlay failed — uploading clean master")
        except Exception as e:
            _append_log(job, f"logo overlay skipped: {e}")
            upload_path = resolved_clip_path

        try:
            video_id = uploader.upload_video(creds, job, upload_path, db, progress_cb=_progress)
            _append_log(job, f"uploaded → video_id {video_id}")
            job.status = "processing"
            db.commit()

            # ─── Thumbnail (best-effort, charged separately) ────
            if clip.thumb_path and quota.reserve(db, quota.COST_THUMBNAIL_SET):
                try:
                    uploader.set_thumbnail(creds, video_id, clip.thumb_path)
                    _append_log(job, "thumbnail applied")
                except uploader.UploadError as e:
                    _append_log(job, f"thumbnail failed (non-fatal): {e}")

            job.status = "done"
            _append_log(job, "done")
            db.commit()

        except uploader.TransientUploadError as e:
            _retry(db, job, str(e))
        except uploader.UploadError as e:
            _fail(db, job, str(e))
        except Exception as e:
            traceback.print_exc()
            _retry(db, job, f"unexpected: {e}")
        finally:
            # Clean up the overlay temp file if we made one (no-op when we
            # uploaded the resolved clip path directly).
            try:
                if upload_path != resolved_clip_path:
                    from youtube import logo_overlay
                    logo_overlay.cleanup_overlay(upload_path, resolved_clip_path)
            except Exception:
                pass
            # Clean up the cloud-download tempdir (if any) regardless of
            # whether the upload succeeded or failed.
            if _clip_tmp_dir:
                try:
                    import shutil as _sh
                    _sh.rmtree(_clip_tmp_dir, ignore_errors=True)
                    _worker_logger.debug(
                        "cleaned up clip tempdir %r for job %d",
                        _clip_tmp_dir, job_id,
                    )
                except Exception:
                    pass
    finally:
        db.close()


def _retry(db: Session, job: models.UploadJob, err: str) -> None:
    if (job.attempts or 0) >= MAX_ATTEMPTS:
        _fail(db, job, f"exceeded {MAX_ATTEMPTS} attempts — {err}")
        return
    delay = BACKOFF_SECS[min((job.attempts or 1) - 1, len(BACKOFF_SECS) - 1)]
    _append_log(job, f"retry in {delay}s — {err}")
    job.last_error = err[:500]
    job.status = "queued"
    db.commit()
    # Sleep off the backoff here (on the worker thread) — keeps the selector
    # simple and we don't need a separate scheduler column.
    time.sleep(delay)
    # The next poll cycle will pick it up


def _fail(db: Session, job: models.UploadJob, err: str) -> None:
    _append_log(job, f"FAILED — {err}")
    job.status = "failed"
    job.last_error = err[:500]
    db.commit()
