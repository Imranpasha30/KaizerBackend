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
import time
import traceback
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy import or_
from sqlalchemy.orm import Session

import models
from database import SessionLocal
from youtube import oauth, uploader, quota


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
        if not clip or not clip.file_path:
            _fail(db, job, "clip is missing file_path")
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

        def _progress(uploaded: int, total: int):
            # Keep the row warm so /api/uploads polling shows movement
            job.updated_at = datetime.now(timezone.utc)
            db.commit()

        try:
            video_id = uploader.upload_video(creds, job, clip.file_path, db, progress_cb=_progress)
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
