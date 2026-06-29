"""WebSocket live progress (Wave 3).

Two channels, both backed by the shared-tailer hub
(``services/progress_hub.py``) so N viewers of the same job cost ONE
DB poll per tick instead of N HTTP polls:

* ``WS /ws/jobs/{job_id}?token=<jwt>`` — render pipeline progress.
  Frames: ``{"type":"job", "status", "log_offset", "log_lines": [...
  only NEW lines since the last frame], "error", "final"?}``. The
  first frame carries the full log so a late joiner renders instantly.

* ``WS /ws/uploads?token=<jwt>`` — the user's non-terminal publish
  jobs. Frames: ``{"type":"uploads", "jobs":[{id,status,publish_kind,
  upload_path,bytes_uploaded,youtube_video_id,last_error}]}``, sent
  only when something changed.

Auth mirrors ``auth.current_user`` (Bearer-in-query because browsers
can't set WS headers): valid JWT → that user; missing/invalid →
legacy-user fallback unless KAIZER_AUTH_REQUIRED is on (then 4401
close). Job channel additionally requires ownership (or admin).

The frontend treats a failed WS as a signal to fall back to the cached
``?since=`` HTTP polling — these frames deliberately carry the same
field names as ``GET /api/jobs/{id}/status/`` so both feeds are
interchangeable in the UI.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import AsyncIterator, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import auth
import models
from database import SessionLocal
from services.progress_hub import hub

log = logging.getLogger("kaizer.ws_progress")

router = APIRouter()

_JOB_TICK_S = 1.0
_UPLOADS_TICK_S = 2.0
_TERMINAL_JOB = {"done", "failed", "cancelled"}
_TERMINAL_UPLOAD = {"completed", "failed", "cancelled"}
_PING_EVERY_S = 30.0


# ─── Auth (WS variant of auth.current_user) ──────────────────────────


def _auth_required() -> bool:
    return os.getenv("KAIZER_AUTH_REQUIRED", "false").lower() in (
        "1", "true", "yes", "on",
    )


def _ws_user(db, token: str) -> Optional[models.User]:
    """Resolve the user for a WS connection. None → reject."""
    if token:
        payload = auth.decode_token(token)
        if payload:
            try:
                uid = int(payload.get("sub") or 0)
            except Exception:
                uid = 0
            if uid:
                u = db.query(models.User).filter(models.User.id == uid).first()
                if u is not None and u.is_active:
                    return u
        if _auth_required():
            return None
    if _auth_required():
        return None
    return auth.ensure_legacy_user(db)


# ─── Job progress tailer ─────────────────────────────────────────────


def _job_payload(job: models.Job, prev_offset: int) -> tuple[dict, int]:
    lines = (job.log or "").split("\n") if job.log else []
    offset = len(lines)
    payload = {
        "type": "job",
        "job_id": int(job.id),
        "status": job.status,
        "current_stage": getattr(job, "current_stage", None),
        "error": job.error,
        "log_offset": offset,
        "log_lines": lines[prev_offset:],
    }
    return payload, offset


def _job_tail_factory(job_id: int) -> AsyncIterator[dict]:
    async def _tail() -> AsyncIterator[dict]:
        prev_offset = 0
        prev_status = None
        first = True
        while True:
            def _read() -> Optional[tuple[dict, int, str]]:
                db = SessionLocal()
                try:
                    job = db.query(models.Job).filter(
                        models.Job.id == int(job_id)
                    ).first()
                    if job is None:
                        return None
                    payload, off = _job_payload(job, 0)
                    return payload, off, (job.status or "")
                finally:
                    db.close()

            row = await asyncio.to_thread(_read)
            if row is None:
                yield {"type": "error", "error": "job not found", "final": True}
                return
            full_payload, offset, status = row

            changed = first or offset != prev_offset or status != prev_status
            if changed:
                # Slice the delta for this frame; first frame ships the
                # full log so late joiners render the whole timeline.
                delta = dict(full_payload)
                if not first:
                    delta["log_lines"] = full_payload["log_lines"][prev_offset:]
                if status in _TERMINAL_JOB:
                    delta["final"] = True
                yield delta
                if status in _TERMINAL_JOB:
                    return
                prev_offset, prev_status, first = offset, status, False
            await asyncio.sleep(_JOB_TICK_S)

    return _tail()


@router.websocket("/ws/jobs/{job_id}")
async def ws_job_progress(websocket: WebSocket, job_id: int, token: str = ""):
    await websocket.accept()
    db = SessionLocal()
    try:
        user = _ws_user(db, token)
        if user is None:
            await websocket.close(code=4401)
            return
        job = db.query(models.Job).filter(models.Job.id == int(job_id)).first()
        if job is None or (
            job.user_id is not None
            and job.user_id != user.id
            and not bool(user.is_admin)
        ):
            await websocket.close(code=4404)
            return
    finally:
        db.close()

    q, detach = await hub.subscribe(
        f"job:{int(job_id)}", lambda: _job_tail_factory(int(job_id)),
    )
    try:
        while True:
            try:
                payload = await asyncio.wait_for(q.get(), timeout=_PING_EVERY_S)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
                continue
            await websocket.send_json(payload)
            if payload.get("final"):
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        log.debug("ws_progress: job socket closed uncleanly", exc_info=True)
    finally:
        await detach()
        try:
            await websocket.close()
        except Exception:
            pass


# ─── Uploads tailer (per-user) ───────────────────────────────────────


def _uploads_tail_factory(user_id: int) -> AsyncIterator[dict]:
    async def _tail() -> AsyncIterator[dict]:
        prev_fingerprint = None
        idle_terminal_ticks = 0
        while True:
            def _read() -> list[dict]:
                db = SessionLocal()
                try:
                    rows = (
                        db.query(models.UploadJobV2)
                        .filter(models.UploadJobV2.user_id == int(user_id))
                        .order_by(models.UploadJobV2.id.desc())
                        .limit(50)
                        .all()
                    )
                    return [{
                        "id": int(r.id),
                        # Job-wise Publishes UI: lets the detail page
                        # filter frames for its own publish task.
                        "publish_task_id": int(r.publish_task_id),
                        "status": r.status,
                        "publish_kind": r.publish_kind,
                        "upload_path": r.upload_path,
                        "bytes_uploaded": int(r.bytes_uploaded or 0),
                        "attempts": int(r.attempts or 0),
                        "youtube_video_id": r.youtube_video_id,
                        "last_error": (r.last_error or "")[:300],
                    } for r in rows]
                finally:
                    db.close()

            jobs = await asyncio.to_thread(_read)
            fingerprint = tuple(
                (j["id"], j["status"], j["bytes_uploaded"]) for j in jobs
            )
            if fingerprint != prev_fingerprint:
                prev_fingerprint = fingerprint
                yield {"type": "uploads", "jobs": jobs}
            # Keep tailing while anything is non-terminal; after
            # everything settles, idle out so the tailer doesn't poll
            # forever for a user who left the page open overnight.
            if jobs and all(j["status"] in _TERMINAL_UPLOAD for j in jobs):
                idle_terminal_ticks += 1
                if idle_terminal_ticks > 150:  # ~5 min of stable terminal
                    yield {"type": "uploads_idle", "final": True}
                    return
            else:
                idle_terminal_ticks = 0
            await asyncio.sleep(_UPLOADS_TICK_S)

    return _tail()


@router.websocket("/ws/uploads")
async def ws_uploads(websocket: WebSocket, token: str = ""):
    await websocket.accept()
    db = SessionLocal()
    try:
        user = _ws_user(db, token)
        if user is None:
            await websocket.close(code=4401)
            return
        user_id = int(user.id)
    finally:
        db.close()

    q, detach = await hub.subscribe(
        f"uploads:{user_id}", lambda: _uploads_tail_factory(user_id),
    )
    try:
        while True:
            try:
                payload = await asyncio.wait_for(q.get(), timeout=_PING_EVERY_S)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
                continue
            await websocket.send_json(payload)
            if payload.get("final"):
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        log.debug("ws_progress: uploads socket closed uncleanly", exc_info=True)
    finally:
        await detach()
        try:
            await websocket.close()
        except Exception:
            pass


__all__ = ["router"]
