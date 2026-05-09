"""Upload worker — Redis Streams consumer, with DB-poll fallback.

Two job-distribution paths live here:

1. **Primary — Redis Streams + Consumer Group** (see redis_queue.py).
   The API XADDs a job_id to ``kaizer:uploads`` after creating the
   row; this worker XREADGROUPs the message and processes the job.
   Multiple worker pods in the same group load-balance naturally and
   crash recovery via XCLAIM is built-in. This is the enterprise path.

2. **Fallback — legacy DB poll on `upload_jobs.status='queued'`**.
   Used only when ``REDIS_URL`` is unset or Redis is unreachable —
   keeps single-host deployments running without a Redis dependency.
   Also runs on startup once to bootstrap any rows that were created
   before the Redis migration.

Both paths converge on ``_process(job_id)`` which holds the actual
upload + Postiz routing + retry policy.

Failure model:
  - On transient failure: ``_retry`` schedules a backoff and leaves
    the message UNACKED so XCLAIM picks it back up if this worker
    dies. Status flips back to 'queued' in DB.
  - On terminal failure (>= MAX_ATTEMPTS, auth error, missing source):
    ``_fail`` flips status to 'failed' AND XACKs the Redis message
    so we don't reprocess. A future Stage will route these to the
    DLQ stream (see ``redis_queue.send_to_dlq``).

Worker enable gate:
  Set ``KAIZER_WORKER_ENABLED=false`` in the env to make this module
  a no-op. Useful on the Railway pod that's running stale code while
  development happens locally.
"""
from __future__ import annotations

import asyncio
import logging
import os
import socket
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

# Redis Streams XREADGROUP block — long enough that an idle worker
# isn't busy-looping, short enough that shutdown is responsive.
REDIS_BLOCK_MS = 5_000

# How many concurrent consumer slots run inside one process. Each slot
# is its own asyncio task with its own Redis consumer name, so they
# don't fight over PEL ownership. Bound by the FFmpeg/network budget
# of the host — concurrency=4 means 4 simultaneous uploads in flight.
DEFAULT_CONCURRENCY = int(os.environ.get("KAIZER_WORKER_CONCURRENCY", "4"))


# Module-level lifecycle state. Multiple slot-tasks live under _tasks.
_running = False
_tasks: list = []

# Base consumer name. Each slot appends `-{i}` so consumers stay
# distinct in the same group.
_CONSUMER_BASE = f"{socket.gethostname()}-{os.getpid()}"
# Backwards-compat alias kept for code paths that reference the
# legacy single-consumer name (admin debug endpoints, etc.).
_CONSUMER_NAME = _CONSUMER_BASE

# Runtime decision on whether to use the Redis path. Computed once
# at startup (see `start()`) so the loop body doesn't keep probing.
_using_redis = False


def _worker_enabled() -> bool:
    return (os.environ.get("KAIZER_WORKER_ENABLED", "true").strip().lower()
            not in ("false", "0", "no", "off"))


# ── Per-stage timing logger ─────────────────────────────────────
# Emits a single line per job-lifecycle event so we can pipe to Loki/
# CloudWatch later. Format is k=v space-separated for cheap grep.
_metrics_logger = logging.getLogger("kaizer.worker.metrics")
_metrics_logger.setLevel(logging.INFO)
if not _metrics_logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[metrics] %(message)s"))
    _metrics_logger.addHandler(_h)
    _metrics_logger.propagate = False


def _emit_metric(stage: str, *, job_id: int, ms: float = 0.0,
                 consumer: str = "", **extra) -> None:
    parts = [f"stage={stage}", f"job_id={job_id}", f"ms={ms:.0f}"]
    if consumer:
        parts.append(f"consumer={consumer}")
    for k, v in extra.items():
        parts.append(f"{k}={v}")
    _metrics_logger.info(" ".join(parts))


def _next_retry_at(attempts: int) -> datetime:
    idx = min(max(attempts - 1, 0), len(BACKOFF_SECS) - 1)
    return datetime.now(timezone.utc) + timedelta(seconds=BACKOFF_SECS[idx])


async def start() -> None:
    """Called from FastAPI startup. Spawns N concurrent consumer slots."""
    global _running, _tasks, _using_redis
    if _running:
        return

    if not _worker_enabled():
        print("[upload-worker] DISABLED — KAIZER_WORKER_ENABLED is false. "
              "This pod will NOT process jobs.")
        return

    # Probe Redis once. If it's unreachable we transparently fall back
    # to the DB-poll path so single-host deployments still work.
    try:
        from redis_queue import is_enabled as _redis_enabled, ensure_group
        if _redis_enabled():
            ensure_group()
            _using_redis = True
        else:
            _using_redis = False
            print("[upload-worker] queue mode = DB-POLL (Redis disabled or unreachable)")
    except Exception as exc:
        _using_redis = False
        print(f"[upload-worker] queue mode = DB-POLL (Redis init failed: {exc})")

    _running = True
    _recover_stuck_rows()

    # One-shot: bring legacy DB-only `queued` rows into the stream so
    # the Redis path doesn't ignore them on first deploy.
    if _using_redis:
        try:
            from redis_queue import bootstrap_existing_queued_jobs
            db = SessionLocal()
            try:
                n = bootstrap_existing_queued_jobs(db)
                if n:
                    print(f"[upload-worker] bootstrap: enqueued {n} pre-existing job(s)")
            finally:
                db.close()
        except Exception as exc:
            print(f"[upload-worker] bootstrap failed (non-fatal): {exc}")

    # Spin up N consumer slots (only useful with Redis — DB-poll
    # serialises through SessionLocal claims anyway, so 1 slot in
    # that mode).
    if _using_redis:
        slots = max(1, DEFAULT_CONCURRENCY)
        print(f"[upload-worker] queue mode = REDIS  base={_CONSUMER_BASE}  slots={slots}")
        for i in range(slots):
            consumer = f"{_CONSUMER_BASE}-{i}"
            _tasks.append(asyncio.create_task(
                _slot_loop(consumer),
                name=f"kaizer-upload-worker-{i}",
            ))
    else:
        _tasks.append(asyncio.create_task(
            _slot_loop(_CONSUMER_BASE),
            name="kaizer-upload-worker-0",
        ))


async def stop() -> None:
    global _running, _tasks
    _running = False
    for t in list(_tasks):
        t.cancel()
    for t in list(_tasks):
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    _tasks = []


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


async def _slot_loop(consumer: str) -> None:
    """One consumer slot — pulls + processes its share of jobs.

    Multiple slots run concurrently in the same process and same
    consumer group. Redis Streams guarantees exactly-one-of-the-
    available-consumers receives each message, so slots don't fight.
    Each slot's PEL is attributed to its consumer name, so a slot
    crash only strands that slot's in-flight messages — XCLAIM
    recovers them onto another slot after idle_ms.
    """
    loop = asyncio.get_running_loop()

    if _using_redis:
        # First, claim any messages that were stranded in PEL by a
        # previous crashed consumer (this consumer's previous boot,
        # or a sibling slot that died). Walks every priority lane.
        try:
            from redis_queue import recover_pending
            for msg_id, job_id, priority in recover_pending(consumer, idle_ms=60_000):
                if not _running:
                    return
                await loop.run_in_executor(
                    None, _process_redis, consumer, msg_id, job_id, priority,
                )
        except Exception:
            traceback.print_exc()

    while _running:
        try:
            if _using_redis:
                msgs = await loop.run_in_executor(None, _redis_pull_batch, consumer)
                if not msgs:
                    continue
                for msg_id, job_id, priority in msgs:
                    if not _running:
                        return
                    await loop.run_in_executor(
                        None, _process_redis, consumer, msg_id, job_id, priority,
                    )
            else:
                # Legacy fallback path. With concurrency=1 in DB-poll
                # mode, the existing single-claim guard inside
                # _pick_next is enough.
                job_id = await loop.run_in_executor(None, _pick_next)
                if job_id is None:
                    await asyncio.sleep(POLL_INTERVAL)
                    continue
                await loop.run_in_executor(None, _process, job_id)
        except asyncio.CancelledError:
            return
        except Exception:
            traceback.print_exc()
            await asyncio.sleep(POLL_INTERVAL)


def _redis_pull_batch(consumer: str) -> list:
    """One XREADGROUP round across priority lanes. Returns a list of
    ``(msg_id, job_id, priority)``."""
    from redis_queue import consume_upload_jobs
    try:
        return list(consume_upload_jobs(
            consumer, count=4, block_ms=REDIS_BLOCK_MS,
        ))
    except Exception as exc:
        _worker_logger.warning("redis pull failed (will retry): %s", exc)
        time.sleep(1.0)
        return []


def _process_redis(consumer: str, msg_id: str, job_id: int, priority: str) -> None:
    """Wraps `_process` with XACK on terminal status + DLQ on failure.

    Carries ``priority`` end-to-end so each XACK / DLQ / re-enqueue
    targets the correct lane. A pro-tier job that retries should come
    back on ``hi``, not ``normal`` — the priority value is the source
    of truth for "which lane did this message live in?".

    Outcomes & PEL bookkeeping:
      - status='done' on entry  → ACK, no work.
      - status='failed'/'cancelled' on entry → ACK (terminal already).
      - status='queued' or 'uploading' on entry → claim (status →
        'uploading') and run _process.
      - After _process: terminal status determines ACK vs DLQ vs
        re-enqueue. See decision table below.
    """
    from redis_queue import (
        ack_upload_job, enqueue_upload_job, send_to_dlq,
    )
    # OTel root span for the whole job lifecycle. When OTel is off this
    # is a no-op; when on, the claim/process/ack child spans below
    # become children of this one and the trace shows the full timeline
    # in Tempo/Jaeger/Honeycomb.
    import tracing
    with tracing.span(
        "worker.process_job",
        **{
            "kaizer.job_id":   job_id,
            "kaizer.consumer": consumer,
            "kaizer.priority": priority,
            "kaizer.msg_id":   msg_id,
        },
    ) as job_span:
        t_claim_start = time.monotonic()

        # ── Phase A: Claim ─────────────────────────────────────────
        with tracing.span("worker.claim", **{"kaizer.job_id": job_id}):
            db = SessionLocal()
            try:
                row = db.query(models.UploadJob).filter(models.UploadJob.id == job_id).first()
                if not row:
                    _worker_logger.warning("redis: job_id=%s not in DB — acking and skipping", job_id)
                    job_span.set_attribute("kaizer.outcome", "row_missing")
                    ack_upload_job(msg_id, priority)
                    return
                if row.status == "done":
                    job_span.set_attribute("kaizer.outcome", "already_done")
                    ack_upload_job(msg_id, priority)
                    return
                if row.status in ("failed", "cancelled"):
                    job_span.set_attribute("kaizer.outcome", "already_terminal")
                    ack_upload_job(msg_id, priority)
                    return
                # Claim — flip status. Mirrors the old _pick_next semantics.
                row.status = "uploading"
                row.last_error = (row.last_error or "") + \
                    f"\n[worker] redis-claimed at {datetime.now(timezone.utc).isoformat()} by {consumer} (lane={priority})"
                db.commit()
            finally:
                db.close()

        _emit_metric("claim", job_id=job_id, ms=(time.monotonic() - t_claim_start) * 1000,
                     consumer=consumer, priority=priority)

        # ── Phase B: Process ──────────────────────────────────────
        with tracing.span("worker.process", **{"kaizer.job_id": job_id}) as proc_span:
            t_proc_start = time.monotonic()
            _process(job_id)
            proc_ms = (time.monotonic() - t_proc_start) * 1000
            proc_span.set_attribute("kaizer.proc_ms", proc_ms)

        # ── Phase C: Decide ACK / DLQ / re-enqueue ────────────────
        with tracing.span("worker.finalise", **{"kaizer.job_id": job_id}) as fin_span:
            db = SessionLocal()
            try:
                row = db.query(models.UploadJob).filter(models.UploadJob.id == job_id).first()
                if not row:
                    ack_upload_job(msg_id, priority)
                    _emit_metric("process_end", job_id=job_id, ms=proc_ms,
                                 consumer=consumer, priority=priority, outcome="row_missing")
                    fin_span.set_attribute("kaizer.outcome", "row_missing")
                    job_span.set_attribute("kaizer.outcome", "row_missing")
                    return
                outcome = row.status
                if outcome == "done":
                    ack_upload_job(msg_id, priority)
                elif outcome == "failed":
                    send_to_dlq(msg_id, job_id, priority,
                                reason=(row.last_error or "")[:500] or "no last_error recorded")
                elif outcome == "cancelled":
                    ack_upload_job(msg_id, priority)
                elif outcome == "queued":
                    ack_upload_job(msg_id, priority)
                    try:
                        enqueue_upload_job(job_id, priority=priority)
                    except Exception as exc:
                        _worker_logger.warning("re-enqueue failed for job %s: %s", job_id, exc)
                else:
                    outcome = f"unfinished:{outcome}"
                _emit_metric("process_end", job_id=job_id, ms=proc_ms,
                             consumer=consumer, priority=priority, outcome=outcome)
                fin_span.set_attribute("kaizer.outcome", outcome)
                job_span.set_attribute("kaizer.outcome", outcome)
            finally:
                db.close()


def _pick_next() -> Optional[int]:
    """Legacy DB-poll picker — used only when Redis is unavailable.

    Atomically claim the next queued row (oldest first, respecting backoff).
    Two workers hitting the same row is prevented in this fallback by
    single-process deployment; the Redis path uses consumer groups
    instead, which handle multi-worker correctly.
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
    print(f"[worker DBG] _process({job_id}) entered")
    db = SessionLocal()
    try:
        job = db.query(models.UploadJob).filter(models.UploadJob.id == job_id).first()
        if not job:
            print(f"[worker DBG] _process({job_id}): job missing")
            return
        if job.status != "uploading":
            print(f"[worker DBG] _process({job_id}): status={job.status} (not 'uploading') — skipping")
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

        # ─── Provider routing (admin-controlled, system-wide) ─────
        # Default = 'postiz': uploads go through the Postiz integration
        # for that destination's YouTube channel. Native Kaizer code
        # path (quota/auth/uploader below) only runs when admin has
        # flipped the toggle to 'kaizer' in Admin → Settings.
        from system_settings import get_upload_provider
        provider = get_upload_provider(db)
        if provider == "postiz":
            _process_via_postiz(db, job, clip, resolved_clip_path)
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
                # Resolve via shared helper — handles R2 download when the
                # logo isn't on this container's disk (Railway redeploy,
                # asset uploaded from a different host, etc.). Returns ""
                # when the asset has no bytes anywhere.
                from asset_resolver import materialize_asset_locally
                logo_local = materialize_asset_locally(logo_asset)
                if logo_local:
                    _append_log(job, f"overlaying destination logo ({logo_asset.filename})…")
                    from youtube import logo_overlay
                    upload_path = logo_overlay.overlay_logo(
                        resolved_clip_path, logo_local
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


def _process_via_postiz(
    db: Session,
    job: models.UploadJob,
    clip: models.Clip,
    resolved_clip_path: str,
) -> None:
    """Upload pathway when admin has set upload_provider='postiz'.

    Skips Kaizer's own quota + OAuth token + videos.insert dance —
    posts to Postiz, which holds its own per-platform OAuth grants
    and runs against its own Google Cloud project's quota.

    Channel-mapping rule: the destination Kaizer Channel's
    OAuthToken.google_channel_id is matched against Postiz
    integrations[i].identifier (Postiz exposes the YouTube channel
    id there for provider='youtube' rows). If no match is found,
    fail loud — the user has to connect that specific YT channel
    inside Postiz first.
    """
    from clients import postiz as postiz_client
    import os as _os
    _key_len = len((_os.environ.get("POSTIZ_API_KEY") or "").strip())
    _base    = _os.environ.get("POSTIZ_BASE_URL", "(default)")
    print(f"[postiz-worker DBG] is_enabled={postiz_client.is_enabled()}  "
          f"POSTIZ_API_KEY len={_key_len}  base={_base}")
    if not postiz_client.is_enabled():
        _fail(db, job,
              f"upload_provider=postiz but POSTIZ_API_KEY is empty in worker env "
              f"(len={_key_len} base={_base}). load_dotenv may have failed or "
              f"the .env was edited after uvicorn started.")
        return

    # 1) Resolve the destination + its YouTube channel id.
    dest_channel = db.query(models.Channel).filter(
        models.Channel.id == job.channel_id,
    ).first()
    if not dest_channel or not dest_channel.oauth_token:
        _fail(db, job, "destination profile not linked to YouTube")
        return
    yt_title = (dest_channel.oauth_token.google_channel_title or "").strip()

    # 2) Pull Postiz integrations and find the matching YT one.
    try:
        integrations = postiz_client.list_integrations()
    except postiz_client.PostizError as e:
        _retry(db, job, f"Postiz unreachable: {e}")
        return

    # Postiz Cloud's response shape:
    #   { id, name, identifier ('youtube'/'twitter'/...),
    #     profile (@handle-suffix), picture, disabled }
    # The `identifier` field is the PROVIDER, not the YouTube channel
    # id, so we match by `name` (channel display name like
    # "Kaizer 30") against the Kaizer OAuthToken.google_channel_title.
    yt_integrations = [
        i for i in integrations
        if (i.get("provider") or i.get("identifier") or "").lower() == "youtube"
        and not i.get("disabled")
    ]

    def _norm(s: str) -> str:
        return "".join(ch for ch in (s or "").lower() if ch.isalnum())

    target_norm = _norm(yt_title)
    match = next(
        (i for i in yt_integrations if _norm(i.get("name", "")) == target_norm),
        None,
    )
    # Fallback: handle (profile field) starts-with match — Postiz adds
    # a random suffix like "-y5p" to handles, so we strip it.
    if not match:
        for i in yt_integrations:
            handle = (i.get("profile") or "").lstrip("@").split("-", 1)[0]
            kz_handle = (dest_channel.oauth_token.channel_custom_url or "").lstrip("@")
            if handle and kz_handle and _norm(handle) == _norm(kz_handle):
                match = i
                break
    # Last resort: SOLO youtube integration on the Postiz account.
    if not match and len(yt_integrations) == 1:
        match = yt_integrations[0]

    if not match:
        names = [i.get("name", "?") for i in yt_integrations]
        _fail(
            db, job,
            f"No Postiz YouTube integration matches channel "
            f"{dest_channel.name!r} (yt_title={yt_title!r}). "
            f"Postiz has {len(yt_integrations)} youtube integration(s): "
            f"{names!r}. Match by exact name in Postiz UI or rename "
            f"the channel there to match.",
        )
        return

    # 3) Upload the local video bytes to Postiz first. Postiz Cloud
    #    references media by INTERNAL id, not URL — passing R2 URLs
    #    fails validation with "image.0.id should not be null". Use
    #    the resolved local path (already downloaded by
    #    _ensure_local_clip) so we don't depend on Postiz's
    #    network reaching R2.
    job.attempts = (job.attempts or 0) + 1
    _append_log(job, f"postiz: uploading video bytes "
                       f"({Path(resolved_clip_path).name})…")
    db.commit()
    try:
        upload = postiz_client.upload_file(resolved_clip_path,
                                           mime_type="video/mp4")
    except postiz_client.PostizAuthError as e:
        _fail(db, job, f"Postiz auth on upload: {e}")
        return
    except postiz_client.PostizError as e:
        _retry(db, job, f"Postiz upload error: {e}")
        return

    media_id   = upload.get("id")
    media_path = upload.get("path")
    if not media_id:
        _fail(db, job, f"Postiz upload returned no id: {upload!r}")
        return
    _append_log(job, f"postiz: upload OK id={media_id}")

    # 4) Compose the post + dispatch.
    _append_log(job, f"postiz: scheduling on integration {match.get('id','?')} "
                       f"({match.get('name','?')})")
    db.commit()
    try:
        result = postiz_client.schedule_post(
            integration_ids=[match["id"]],
            text=(job.title or "") + (("\n\n" + job.description) if job.description else ""),
            media_id=media_id,
            media_path=media_path,
            schedule_at_iso=(job.publish_at.isoformat()
                              if getattr(job, "publish_at", None) else None),
            type_="scheduled" if getattr(job, "publish_at", None) else "now",
            yt_title=(job.title or "").strip()[:100] or None,
            yt_privacy=(job.privacy_status or "public"),
        )
    except postiz_client.PostizAuthError as e:
        _fail(db, job, f"Postiz auth: {e}")
        return
    except postiz_client.PostizError as e:
        _retry(db, job, f"Postiz error: {e}")
        return

    job.status = "done"
    _append_log(job, f"postiz: scheduled OK ({result})")
    db.commit()


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
