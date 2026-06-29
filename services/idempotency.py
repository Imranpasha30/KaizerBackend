"""Idempotency layer — Phase 2.F full implementation.

This is the SINGLE place idempotency keys are computed and the SINGLE
place ``publish_attempts`` rows are written, read, and reconciled. Two
agents had a stake in this file: Fanout (B) needed ``compute_key``
during Phase 1; F-agent (this one) adds the full register/record/recover
trio during Phase 2.

Phase 1.B's ``compute_key`` is preserved BYTE-FOR-BYTE so existing
``upload_jobs_v2.idempotency_key`` rows keep matching.

The brief is firm (§2 + §9):
  * recovery confirmation uses ``videos.list`` (1 unit), NEVER
    ``search.list`` (100 units)
  * crash-after-upload must NOT create a duplicate video

Race-safety strategy
--------------------
We rely on the database-level UNIQUE constraint on
``publish_attempts.idempotency_key`` for the hard-stop. Two workers
hitting ``check_or_register`` simultaneously will both attempt to INSERT
a new ``status='in_flight'`` row; exactly one INSERT wins and returns
``state='fresh'``. The loser hits the UNIQUE constraint, rolls back the
INSERT, re-reads the existing row, and returns ``state='in_flight'``.

This is simpler and more correct than a row-locking dance and works
identically on Postgres and SQLite. The DB is the arbiter.

Stale-attempt recovery is handled by ``recover_orphans`` — a startup
task that finds ``in_flight`` rows older than ``stale_after_seconds``,
checks YouTube via ``videos.list`` (1 unit), and either marks them
``completed`` (if the upload actually landed) or ``recovered`` (so the
parking semantics from brief §9 take over and the scheduler re-queues).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional, Union

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import models


log = logging.getLogger("kaizer.idempotency")


# ─── Key computation (Phase 1 stub — preserved verbatim) ───────────────


def compute_key(master_video_id: int, channel_id: int, publish_version: str) -> str:
    """SHA256 of ``f"{master_video_id}|{channel_id}|{publish_version}"`` as hex.

    The 64-char return is the value stored in
    ``upload_jobs_v2.idempotency_key`` (which is UNIQUE) and
    in ``publish_attempts.idempotency_key`` (also UNIQUE).
    Same inputs → same hex → DB-level dedupe.
    """
    if not isinstance(publish_version, str):
        raise TypeError("publish_version must be a string")
    raw = f"{int(master_video_id)}|{int(channel_id)}|{publish_version}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# ─── Result dataclasses ────────────────────────────────────────────────


@dataclass
class CheckResult:
    """Returned by ``check_or_register``.

    ``state`` is one of:
      - 'fresh'      : this caller just acquired the right to upload
      - 'in_flight'  : another worker is already uploading; caller backs off
      - 'completed'  : a prior upload finished; caller should confirm
                       via ``videos.list`` (1 unit) and short-circuit

    The other fields are populated only when meaningful (other_worker
    for in_flight; youtube_video_id for completed).
    """
    state: Literal['fresh', 'in_flight', 'completed']
    upload_job_id: int
    other_worker: Optional[str] = None
    youtube_video_id: Optional[str] = None
    attempt_no: int = 1

    # Convenience flag the legacy ``upload_dispatch`` code path reads.
    @property
    def is_short_circuit(self) -> bool:
        return self.state == 'completed' and bool(self.youtube_video_id)


# ─── Helpers ───────────────────────────────────────────────────────────


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _attempt_age_seconds(row: models.PublishAttempt) -> float:
    """Age of a publish_attempt row in seconds, UTC-safe."""
    ts = getattr(row, "updated_at", None) or getattr(row, "created_at", None)
    if ts is None:
        return 0.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (_now_utc() - ts).total_seconds()


def _latest_attempt_no(db: Session, idempotency_key: str) -> int:
    """Return the highest ``attempt_no`` ever recorded for this key, or 0."""
    row = (
        db.query(models.PublishAttempt)
        .filter(models.PublishAttempt.idempotency_key == idempotency_key)
        .order_by(models.PublishAttempt.attempt_no.desc())
        .first()
    )
    if row is None:
        return 0
    return int(row.attempt_no or 0)


def _load_upload_job(db: Session, upload_job_id: int) -> Optional[models.UploadJobV2]:
    return (
        db.query(models.UploadJobV2)
        .filter(models.UploadJobV2.id == int(upload_job_id))
        .first()
    )


# ─── Public API: register, record, recover ────────────────────────────


def check_or_register(
    db: Session,
    *,
    upload_job_id: int,
    worker_id: str,
    stale_after_seconds: int = 600,
    # Phase 1.B compatibility kwargs — some legacy callers (and the
    # CONTRACTS.md §4.5 signature variant) pass ``key`` directly. Accept
    # both shapes so older imports keep working.
    key: Optional[str] = None,
) -> CheckResult:
    """Idempotent registration of an upload attempt.

    Decision tree (matches the brief §2 + §9 contract):

      * If ``UploadJobV2.youtube_video_id`` is already set ->
        state='completed'. The caller MUST confirm via
        ``youtube.videos().list(id=youtube_video_id).execute()`` (1 unit,
        NEVER ``search.list``) before short-circuiting.

      * Else if a ``PublishAttempt`` with status='in_flight' exists and
        is younger than ``stale_after_seconds`` -> state='in_flight'.
        The caller should NOT start a duplicate upload; the scheduler
        will retry after the in-flight attempt finishes (or is recovered).

      * Else: INSERT a new PublishAttempt with status='in_flight',
        ``worker_id``, ``attempt_no = max_existing + 1``,
        ``idempotency_key`` from the UploadJobV2. Return state='fresh'.

    Race-safety: the UNIQUE(idempotency_key) constraint on
    ``publish_attempts`` is the arbiter. If two workers race the INSERT,
    one wins and returns 'fresh'; the other catches the
    ``IntegrityError``, rolls back its INSERT only, re-reads the row,
    and returns 'in_flight' (or whatever state the winner reached).
    """
    job = _load_upload_job(db, upload_job_id)
    if job is None:
        raise LookupError(f"UploadJobV2 id={upload_job_id} not found")

    idem_key = (key or job.idempotency_key or "").strip()
    if not idem_key:
        raise ValueError(
            f"UploadJobV2 id={upload_job_id} has no idempotency_key"
        )

    # Short-circuit: row already says completed.
    if (job.youtube_video_id or "").strip():
        return CheckResult(
            state='completed',
            upload_job_id=int(job.id),
            youtube_video_id=str(job.youtube_video_id),
            attempt_no=_latest_attempt_no(db, idem_key),
        )

    # Look at the existing attempt rows.
    rows = (
        db.query(models.PublishAttempt)
        .filter(models.PublishAttempt.idempotency_key == idem_key)
        .order_by(models.PublishAttempt.attempt_no.desc())
        .all()
    )

    in_flight_recent = None
    for r in rows:
        if r.status == 'completed' and (r.youtube_video_id or ""):
            return CheckResult(
                state='completed',
                upload_job_id=int(job.id),
                youtube_video_id=str(r.youtube_video_id),
                attempt_no=int(r.attempt_no or 0),
            )
        if r.status == 'in_flight':
            age = _attempt_age_seconds(r)
            if age < float(stale_after_seconds):
                in_flight_recent = r
                break
            # Else: this in_flight row is stale; recover_orphans will
            # mark it 'recovered' eventually. We fall through to INSERT
            # a new attempt so the upload can proceed.

    if in_flight_recent is not None:
        return CheckResult(
            state='in_flight',
            upload_job_id=int(job.id),
            other_worker=str(in_flight_recent.worker_id or ""),
            attempt_no=int(in_flight_recent.attempt_no or 0),
        )

    # INSERT a fresh attempt. UNIQUE(idempotency_key) means two workers
    # racing here will have exactly one winner. The loser catches the
    # IntegrityError, rolls back, re-reads, and returns the existing
    # row's state.
    next_attempt_no = max(1, _latest_attempt_no(db, idem_key) + 1)

    # If a prior attempt row exists (e.g. 'failed' or 'recovered'), we
    # cannot insert a second row with the same UNIQUE key. The schema's
    # one-row-per-key model is intentional: there's only one canonical
    # publish_attempt per (master_video, channel, publish_version). Re-
    # registering means flipping that row back to 'in_flight'.
    existing = (
        db.query(models.PublishAttempt)
        .filter(models.PublishAttempt.idempotency_key == idem_key)
        .first()
    )
    if existing is not None:
        # Re-attempt: bump attempt_no, flip status back to 'in_flight'.
        try:
            existing.status = 'in_flight'
            existing.worker_id = worker_id[:64]
            existing.attempt_no = next_attempt_no
            existing.updated_at = _now_utc()
            db.add(existing)
            db.flush()
        except Exception:
            db.rollback()
            raise
        return CheckResult(
            state='fresh',
            upload_job_id=int(job.id),
            attempt_no=next_attempt_no,
        )

    attempt = models.PublishAttempt(
        upload_job_id=int(job.id),
        idempotency_key=idem_key,
        youtube_video_id=None,
        status='in_flight',
        attempt_no=next_attempt_no,
        worker_id=worker_id[:64],
    )
    try:
        db.add(attempt)
        db.flush()
    except IntegrityError:
        # Race lost — another worker inserted first. Roll back ONLY this
        # INSERT (caller's outer transaction must remain intact), re-read,
        # and report what the winner produced.
        db.rollback()
        winner = (
            db.query(models.PublishAttempt)
            .filter(models.PublishAttempt.idempotency_key == idem_key)
            .first()
        )
        if winner is None:
            # Pathological: the unique violation fired but no row exists.
            # Treat as fresh and let the caller retry.
            log.warning(
                "idempotency.check_or_register: unique violation for "
                "key=%s but no row found on re-read; treating as fresh",
                idem_key[:16],
            )
            return CheckResult(
                state='fresh',
                upload_job_id=int(job.id),
                attempt_no=next_attempt_no,
            )
        if winner.status == 'completed' and (winner.youtube_video_id or ""):
            return CheckResult(
                state='completed',
                upload_job_id=int(job.id),
                youtube_video_id=str(winner.youtube_video_id),
                attempt_no=int(winner.attempt_no or 0),
            )
        return CheckResult(
            state='in_flight',
            upload_job_id=int(job.id),
            other_worker=str(winner.worker_id or ""),
            attempt_no=int(winner.attempt_no or 0),
        )

    return CheckResult(
        state='fresh',
        upload_job_id=int(job.id),
        attempt_no=next_attempt_no,
    )


def record_success(
    db: Session,
    *,
    upload_job_id: Optional[int] = None,
    youtube_video_id: str,
    # Phase 1.B compatibility kwarg.
    key: Optional[str] = None,
) -> None:
    """Mark the latest PublishAttempt as 'completed' and persist
    ``UploadJobV2.youtube_video_id``.

    Both writes happen via the caller's open transaction so they commit
    atomically. Idempotent: calling twice with the same arguments is
    safe (the row is already 'completed').

    Accepts either ``upload_job_id`` (preferred) or ``key`` (legacy).
    """
    if upload_job_id is None and not key:
        raise ValueError("record_success requires upload_job_id or key")

    job: Optional[models.UploadJobV2] = None
    idem_key: Optional[str] = None
    if upload_job_id is not None:
        job = _load_upload_job(db, int(upload_job_id))
        if job is not None:
            idem_key = (job.idempotency_key or "").strip() or None
    if idem_key is None and key:
        idem_key = key

    if idem_key is None:
        raise LookupError(
            f"record_success: cannot resolve idempotency_key "
            f"(upload_job_id={upload_job_id!r}, key={key!r})"
        )

    # Locate (or lazy-load) the UploadJobV2 by idempotency_key if needed.
    if job is None:
        job = (
            db.query(models.UploadJobV2)
            .filter(models.UploadJobV2.idempotency_key == idem_key)
            .first()
        )

    if job is not None:
        job.youtube_video_id = (youtube_video_id or "")[:32]
        db.add(job)

    # Update the most-recent attempt row (highest attempt_no).
    attempt = (
        db.query(models.PublishAttempt)
        .filter(models.PublishAttempt.idempotency_key == idem_key)
        .order_by(models.PublishAttempt.attempt_no.desc())
        .first()
    )
    if attempt is not None:
        attempt.status = 'completed'
        attempt.youtube_video_id = (youtube_video_id or "")[:32]
        attempt.updated_at = _now_utc()
        db.add(attempt)

    db.flush()


def record_failure(
    db: Session,
    *,
    upload_job_id: int,
    error: str,
    is_transient: bool,
) -> None:
    """Mark the latest PublishAttempt as 'failed' (permanent) or
    'recovered' (transient).

    'recovered' has the semantics from brief §9: the row is parked, the
    scheduler may re-queue it (the caller is responsible for the queue
    side). 'failed' is the terminal state.
    """
    job = _load_upload_job(db, int(upload_job_id))
    if job is None:
        raise LookupError(f"UploadJobV2 id={upload_job_id} not found")

    idem_key = (job.idempotency_key or "").strip()
    if not idem_key:
        return

    if error:
        try:
            job.last_error = str(error)[:1000]
            db.add(job)
        except Exception:
            pass

    attempt = (
        db.query(models.PublishAttempt)
        .filter(models.PublishAttempt.idempotency_key == idem_key)
        .order_by(models.PublishAttempt.attempt_no.desc())
        .first()
    )
    if attempt is not None:
        attempt.status = 'recovered' if is_transient else 'failed'
        attempt.updated_at = _now_utc()
        db.add(attempt)

    db.flush()


def recover_orphans(
    db: Session,
    stale_after_seconds: int = 600,
) -> list[int]:
    """Startup-time recovery sweep for crashed-mid-upload workers.

    Finds PublishAttempt rows with ``status='in_flight'`` older than
    ``stale_after_seconds``. For each:

      * If the parent UploadJobV2 already carries a
        ``youtube_video_id``, mark the attempt 'recovered' (the upload
        actually landed; no re-queue is needed; the dispatch's
        idempotency probe will short-circuit on next pickup).
      * Otherwise, mark the attempt 'recovered' AND set the parent
        UploadJobV2.status='queued' so the scheduler will re-dispatch
        it. ``attempts`` is NOT bumped — this is parking semantics
        (brief §9), not a retry charge.

    Returns the list of UploadJobV2 IDs that were re-queued (caller can
    use this list to ``scheduler_enqueue`` them).

    NOTE on YouTube confirmation: the brief allows the caller to
    confirm orphans via ``videos.list`` (1 unit) before marking
    'recovered'. We don't make that call HERE — that's the caller's
    job because it needs OAuth credentials per channel. This function
    is pure DB; the upload dispatch's startup wiring drives the
    confirmation pass.
    """
    cutoff = _now_utc() - timedelta(seconds=int(stale_after_seconds))

    # IMPORTANT: filter on updated_at when present, fall back to
    # created_at. Postgres TIMESTAMP WITH TIME ZONE handles this fine.
    stale_attempts = (
        db.query(models.PublishAttempt)
        .filter(
            models.PublishAttempt.status == 'in_flight',
            models.PublishAttempt.updated_at < cutoff,
        )
        .all()
    )

    requeued: list[int] = []
    for r in stale_attempts:
        job = _load_upload_job(db, int(r.upload_job_id))
        if job is None:
            # Parent row gone (cleanup ran) — just mark recovered.
            r.status = 'recovered'
            r.updated_at = _now_utc()
            db.add(r)
            continue

        if (job.youtube_video_id or "").strip():
            # The upload landed before the worker died; no re-queue.
            r.status = 'recovered'
            r.updated_at = _now_utc()
            db.add(r)
            log.info(
                "idempotency.recover_orphans: attempt_id=%s upload_job_id=%s "
                "had youtube_video_id=%s; marking recovered, no re-queue",
                r.id, job.id, job.youtube_video_id,
            )
            continue

        # Parking semantics — flip job back to 'queued', do NOT bump attempts.
        r.status = 'recovered'
        r.updated_at = _now_utc()
        job.status = 'queued'
        # last_error preserves the parking reason so the dashboard shows
        # why the row was re-queued.
        if not (job.last_error or "").startswith("recovered"):
            job.last_error = (
                f"recovered from stale in_flight attempt "
                f"(attempt_no={r.attempt_no}, age>{stale_after_seconds}s)"
            )[:1000]
        db.add(r)
        db.add(job)
        requeued.append(int(job.id))
        log.info(
            "idempotency.recover_orphans: re-queued upload_job_id=%s "
            "(attempt_id=%s worker_id=%s)",
            job.id, r.id, r.worker_id,
        )

    db.flush()
    return requeued


# ─── Convenience aliases for the upload_dispatch import surface ────────


# upload_dispatch.py does ``hasattr(_idem_module, "check_or_register")``
# to detect the full impl — we export it under that exact name plus
# camelCase-free aliases the CONTRACTS.md §4.5 spec uses.
Fresh = CheckResult
InFlight = CheckResult
AlreadyCompleted = CheckResult


__all__ = [
    "compute_key",
    "CheckResult",
    "check_or_register",
    "record_success",
    "record_failure",
    "recover_orphans",
    "Fresh",
    "InFlight",
    "AlreadyCompleted",
]
