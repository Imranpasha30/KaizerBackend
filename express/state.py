"""In-memory job state for Express Mode autopub jobs.

Mirrors the teammate's ``autopubJobs`` Map with ~6h TTL — short jobs
don't justify a DB table. Each job is owned by a single user; cross-
user reads return 404 so this is multi-tenant safe.

Status flow:
  queued -> running (with progressing ``step`` + ``progress``)
         -> done  (with ``results`` payload)
         -> failed (with ``error``)

Step labels match the teammate's so log diffs are useful side-by-side:
  starting, transcribe, plan, render-trim, render-shorts, upload,
  publish, done, failed
"""
from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def _db_session():
    """Get a fresh DB session. Imports are lazy so unit tests that
    don't have the DB ready can still exercise the in-memory path."""
    from database import SessionLocal
    return SessionLocal()


def _save_to_db(j: dict) -> None:
    """Write the in-memory job dict to the ExpressJob table. Idempotent
    (insert-or-update by id). Errors are logged but never raised — DB
    persistence is a "best effort" mirror of the in-memory truth."""
    try:
        import models
        log_list = j.get("log") or []
        results  = j.get("results") or {}
        title    = ""
        if isinstance(results, dict):
            title = (results.get("title") or "")[:255]
        sess = _db_session()
        try:
            row = sess.get(models.ExpressJob, j["id"])
            if row is None:
                row = models.ExpressJob(id=j["id"])
                sess.add(row)
            row.user_id      = int(j["user_id"])
            row.status       = (j.get("status") or "queued")[:16]
            row.mode         = (results.get("mode") if isinstance(results, dict) else None) or row.mode
            row.step         = (j.get("step") or "")[:32]
            row.progress     = int(j.get("progress", 0))
            row.message      = (j.get("message") or "")[:512]
            row.title        = title
            row.log_json     = json.dumps(log_list[-500:], ensure_ascii=False)
            row.results_json = json.dumps(results, ensure_ascii=False) if results else None
            row.error        = (j.get("error") or "")[:4000] if j.get("error") else None
            row.updated_at   = datetime.now(timezone.utc)
            if not row.created_at:
                row.created_at = datetime.fromtimestamp(j.get("created_at", time.time()), tz=timezone.utc)
            sess.commit()
        finally:
            sess.close()
    except Exception as exc:
        print(f"[express/state] DB persist skipped: {exc}")


# 6 hours — same as teammate's TTL. Long enough for the slowest
# publish (~30 min for a 10-min source with shorts + AI thumbnail),
# short enough that abandoned jobs don't leak memory forever.
_TTL_SEC = 6 * 60 * 60

_LOCK = threading.Lock()
_JOBS: dict[str, dict] = {}


def _now() -> float:
    return time.time()


def _purge_expired() -> None:
    """Remove jobs older than TTL. Called inline on every read/write —
    O(N) but N is tiny (max a few dozen concurrent jobs)."""
    now = _now()
    expired = [jid for jid, j in _JOBS.items()
               if now - j.get("created_at", now) > _TTL_SEC]
    for jid in expired:
        _JOBS.pop(jid, None)


def new_job(*, user_id: int) -> str:
    """Mint a fresh job id and put it in queued state.

    user_id is the owning Kaizer user — every other call cross-checks
    against this so user A can never read/modify user B's job.
    """
    jid = secrets.token_urlsafe(8)
    with _LOCK:
        _purge_expired()
        _JOBS[jid] = {
            "id":          jid,
            "user_id":     int(user_id),
            "status":      "queued",
            "step":        "starting",
            "progress":    0,
            "message":     "Queued",
            "created_at":  _now(),
            "updated_at":  _now(),
            "log":         [],
            "results":     None,
            "error":       None,
        }
        snapshot = dict(_JOBS[jid])
    _save_to_db(snapshot)
    return jid


def get(jid: str, *, user_id: int) -> Optional[dict]:
    """Return the job dict, or None if unknown/not yours.

    Lookup order: in-memory first (freshest), then DB (survives
    backend restart). Returns a copy so callers can't mutate state.
    """
    with _LOCK:
        _purge_expired()
        j = _JOBS.get(jid)
        if j:
            if int(j.get("user_id", -1)) != int(user_id):
                return None
            return dict(j)

    # Not in memory — try DB.
    try:
        import models
        sess = _db_session()
        try:
            row = sess.get(models.ExpressJob, jid)
            if not row:
                return None
            if int(row.user_id) != int(user_id):
                return None
            try:
                results = json.loads(row.results_json) if row.results_json else None
            except (ValueError, TypeError):
                results = None
            try:
                log = json.loads(row.log_json) if row.log_json else []
            except (ValueError, TypeError):
                log = []
            return {
                "id":         row.id,
                "user_id":    row.user_id,
                "status":     row.status,
                "step":       row.step or "",
                "progress":   int(row.progress or 0),
                "message":    row.message or "",
                "log":        log,
                "results":    results,
                "error":      row.error,
                "created_at": row.created_at.timestamp() if row.created_at else 0,
                "updated_at": row.updated_at.timestamp() if row.updated_at else 0,
            }
        finally:
            sess.close()
    except Exception as exc:
        print(f"[express/state] DB get fallback skipped: {exc}")
        return None


def list_for_user(user_id: int, *, limit: int = 50) -> list[dict]:
    """Return all jobs owned by ``user_id``, newest first.

    Source order:
      1. In-memory map (current uptime, freshest data).
      2. DB rows for any job ids not already in memory (history that
         survives a backend restart).

    Returns plain dicts — callers can't mutate live state. ``limit``
    caps the merged response.
    """
    with _LOCK:
        _purge_expired()
        mem = {
            j["id"]: dict(j) for j in _JOBS.values()
            if int(j.get("user_id", -1)) == int(user_id)
        }

    # Pull DB rows for the same user; in-memory entries win on
    # collision because they have the freshest log / progress data.
    db_rows: list[dict] = []
    try:
        import models
        sess = _db_session()
        try:
            q = (
                sess.query(models.ExpressJob)
                .filter(models.ExpressJob.user_id == int(user_id))
                .order_by(models.ExpressJob.created_at.desc())
                .limit(limit * 2)
            )
            for row in q.all():
                if row.id in mem:
                    continue
                try:
                    results = json.loads(row.results_json) if row.results_json else None
                except (ValueError, TypeError):
                    results = None
                try:
                    log = json.loads(row.log_json) if row.log_json else []
                except (ValueError, TypeError):
                    log = []
                db_rows.append({
                    "id":         row.id,
                    "user_id":    row.user_id,
                    "status":     row.status,
                    "step":       row.step or "",
                    "progress":   int(row.progress or 0),
                    "message":    row.message or "",
                    "log":        log,
                    "results":    results,
                    "error":      row.error,
                    "created_at": row.created_at.timestamp() if row.created_at else 0,
                    "updated_at": row.updated_at.timestamp() if row.updated_at else 0,
                })
        finally:
            sess.close()
    except Exception as exc:
        print(f"[express/state] DB list fallback skipped: {exc}")

    merged = list(mem.values()) + db_rows
    merged.sort(key=lambda j: j.get("created_at", 0), reverse=True)
    return merged[:limit]


def update(jid: str, **patch: Any) -> None:
    """Merge ``patch`` into the job. No-op if the job has expired."""
    snapshot: Optional[dict] = None
    with _LOCK:
        _purge_expired()
        j = _JOBS.get(jid)
        if not j:
            return
        j.update(patch)
        j["updated_at"] = _now()
        snapshot = dict(j)
    if snapshot:
        _save_to_db(snapshot)


def append_log(jid: str, line: str) -> None:
    """Tack one log line onto the job. Trimmed to 500 entries to keep
    memory bounded even on a misbehaving 30-minute pipeline."""
    if not line:
        return
    snapshot: Optional[dict] = None
    with _LOCK:
        j = _JOBS.get(jid)
        if not j:
            return
        log = j.setdefault("log", [])
        log.append(line)
        if len(log) > 500:
            del log[: len(log) - 500]
        j["updated_at"] = _now()
        snapshot = dict(j)
    if snapshot:
        _save_to_db(snapshot)


def set_step(jid: str, step: str, progress: int, message: str = "") -> None:
    """Convenience wrapper for the common progress update pattern.
    Matches the shape of the teammate's setAutopubStatus(...)."""
    update(jid, status="running", step=step, progress=int(progress), message=message)
    if message:
        append_log(jid, f"[{step}] {message}")


def mark_done(jid: str, results: dict) -> None:
    update(jid, status="done", step="done", progress=100,
           message="Published", results=results)
    append_log(jid, "[done] pipeline finished")


def mark_failed(jid: str, error: str) -> None:
    update(jid, status="failed", step="failed", message=error[:300],
           error=error[:2000])
    append_log(jid, f"[failed] {error[:300]}")
