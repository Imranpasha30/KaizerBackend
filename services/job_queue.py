"""Durable publish queue — Postgres ``FOR UPDATE SKIP LOCKED`` contract.

Wave 1 of the enterprise push. The ``upload_jobs_v2`` row IS the queue
entry: every claim-relevant fact (status, priority, attempts, user,
backoff time, lease) lives on the row, so there is no second source of
truth to drift. This module is the single home for all queue SQL —
``services/publish_worker.py`` (the claim loop) and
``services/cron_runner.py`` (reaper / un-parker) call into here and
contain no queue SQL of their own.

Semantics carried over 1:1 from the in-memory scheduler it replaces
(``services/scheduler.py``, dormant when KAIZER_DURABLE_QUEUE=1):

* 4 priority ranks      critical=0 < high=1 < normal=2 < low=3
* aging                 one rank step per KAIZER_PRIORITY_AGING_MIN
                        minutes waited (floor at critical)
* per-user active cap   plan_tiers.slot_cap_active_uploads
* reserved capacity     KAIZER_RESERVED_CAPACITY_PCT — rank r may not
                        eat into the slots reserved for better ranks
* global concurrency    KAIZER_GLOBAL_ACTIVE_CAP — cluster-wide now
                        (the DB counts every worker's active jobs),
                        replacing the per-process net-token semaphore

New semantics (the reliability fixes):

* retry w/ backoff      status='queued' + next_attempt_at in the future
                        (30s → 2m → 8m → 30m, ±20% jitter, max 5 tries)
* lease + fencing       claimed_by / lease_expires_at; the reaper
                        requeues any active-status row whose lease
                        expired (covers crash mid-branding/uploading —
                        recovery is now runtime, not boot-only)
* un-parking            parked_quota rows return to the queue when the
                        YouTube quota window resets

Multi-process correctness: the claim is an atomic UPDATE guarded by
``FOR UPDATE SKIP LOCKED`` — N workers across M machines can claim
concurrently; each row has exactly one winner. The eligibility pass
runs on an unlocked snapshot, so concurrent claimers can transiently
overshoot *soft* limits (user cap / reserved floors) by at most
(workers-1) × batch; with small batches this is the standard, accepted
SKIP-LOCKED trade. Hard correctness (no double-claim, no lost job)
never depends on the snapshot.

SQLite fallback (tests / toy dev): no SKIP LOCKED, no window-function
eligibility — a simplified single-process claim keeps the same public
surface so the worker runs unchanged. The dev + prod DB is Postgres.
"""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Optional, Sequence

from sqlalchemy import text

from database import SessionLocal, engine

log = logging.getLogger("kaizer.job_queue")


# ─── Tunables (read per call so tests can flip env at runtime) ───────

PRIORITY_RANK = {"critical": 0, "high": 1, "normal": 2, "low": 3}
RANK_PRIORITY = {v: k for k, v in PRIORITY_RANK.items()}

#: statuses that occupy a slot (count against user + global caps) and
#: that the reaper watches for expired leases.
ACTIVE_STATUSES = ("claimed", "branding", "ready_to_upload", "uploading")

_DEFAULT_RESERVED_PCT = {"critical": 50, "high": 30, "normal": 15, "low": 5}


def max_attempts() -> int:
    try:
        return max(1, int(os.environ.get("KAIZER_MAX_UPLOAD_ATTEMPTS", "5")))
    except Exception:
        return 5


def lease_seconds() -> int:
    try:
        return max(30, int(os.environ.get("KAIZER_JOB_LEASE_SECONDS", "300")))
    except Exception:
        return 300


def global_active_cap() -> int:
    """Cluster-wide concurrent-upload budget. Replaces the old
    per-process KAIZER_SCHED_NET_TOKENS semaphore (and defaults to the
    same 16 so a flag flip changes nothing about throughput)."""
    try:
        return max(1, int(
            os.environ.get("KAIZER_GLOBAL_ACTIVE_CAP")
            or os.environ.get("KAIZER_SCHED_NET_TOKENS")
            or "16"
        ))
    except Exception:
        return 16


def aging_step_min() -> float:
    try:
        v = float(os.environ.get("KAIZER_PRIORITY_AGING_MIN", "5"))
        return v if v > 0 else 5.0
    except Exception:
        return 5.0


def reserved_pct() -> dict[str, int]:
    raw = os.environ.get("KAIZER_RESERVED_CAPACITY_PCT", "")
    if not raw.strip():
        return dict(_DEFAULT_RESERVED_PCT)
    try:
        parsed = json.loads(raw)
        out = dict(_DEFAULT_RESERVED_PCT)
        for k in out:
            if k in parsed:
                out[k] = max(0, min(100, int(parsed[k])))
        return out
    except Exception:
        return dict(_DEFAULT_RESERVED_PCT)


def default_user_cap() -> int:
    """Cap for rows whose user has no plan tier (legacy / NULL user_id).
    Matches the pro default so a flag flip never throttles anyone."""
    try:
        return max(1, int(os.environ.get("KAIZER_SLOT_CAP_PRO", "20")))
    except Exception:
        return 20


def backoff_seconds(attempts: int) -> int:
    """30s → 2m → 8m → 30m (cap), ±20% full jitter.

    ``attempts`` is the value AFTER the claim bump, i.e. 1 on the first
    failure. Crash/reap cycles also bump attempts (the claim does it),
    which is deliberate poison-pill protection: a job that keeps killing
    its worker runs out of attempts like any other failure.
    """
    base = min(1800, 30 * (4 ** max(0, int(attempts) - 1)))
    jitter = base * 0.2
    return max(5, int(base + random.uniform(-jitter, jitter)))


def _is_pg() -> bool:
    return engine.dialect.name == "postgresql"


# ─── Claim result row ────────────────────────────────────────────────


@dataclass(frozen=True)
class ClaimedJob:
    id: int
    user_id: Optional[int]
    publish_task_id: int
    priority_at_dispatch: str


# ─── Claim (the heart of the queue) ──────────────────────────────────

# Statement 1 — eligibility snapshot. Window functions express, in one
# round trip, exactly what the old in-memory scheduler computed per
# tick: aging-adjusted rank, the per-user admission window, and the
# reserved-capacity budget per rank.
_ELIGIBILITY_SQL_PG = """
WITH active AS (
    SELECT user_id, count(*) AS cnt
    FROM upload_jobs_v2
    WHERE status IN ('claimed','branding','ready_to_upload','uploading')
    GROUP BY user_id
),
totals AS (
    SELECT COALESCE(sum(cnt), 0) AS active_total FROM active
),
eligible AS (
    SELECT j.id,
           j.user_id,
           j.created_at,
           GREATEST(0,
               (CASE j.priority
                    WHEN 'critical' THEN 0
                    WHEN 'high'     THEN 1
                    WHEN 'normal'   THEN 2
                    ELSE 3 END)
               - FLOOR(EXTRACT(EPOCH FROM (now() - j.created_at))
                       / 60.0 / :aging_step_min)::int
           ) AS eff_rank,
           COALESCE(pt.slot_cap_active_uploads, :default_cap) AS user_cap,
           COALESCE(a.cnt, 0) AS user_active
    FROM upload_jobs_v2 j
    LEFT JOIN users u       ON u.id = j.user_id
    LEFT JOIN plan_tiers pt ON pt.id = u.plan_tier_id
    LEFT JOIN active a      ON a.user_id = j.user_id
    WHERE j.status = 'queued'
      AND j.next_attempt_at <= now()
      AND j.attempts < :max_attempts
),
ranked AS (
    SELECT e.*,
           row_number() OVER (PARTITION BY e.user_id
                              ORDER BY e.eff_rank, e.created_at, e.id) AS user_rn,
           row_number() OVER (ORDER BY e.eff_rank, e.created_at, e.id) AS global_rn
    FROM eligible e
)
SELECT r.id, r.eff_rank
FROM ranked r, totals t
WHERE r.user_rn <= GREATEST(0, r.user_cap - r.user_active)
  AND (t.active_total + r.global_rn)
      <= (:global_cap - CASE r.eff_rank
              WHEN 0 THEN 0
              WHEN 1 THEN (:global_cap * :pct_critical) / 100
              WHEN 2 THEN (:global_cap * (:pct_critical + :pct_high)) / 100
              ELSE        (:global_cap * (:pct_critical + :pct_high + :pct_normal)) / 100
          END)
ORDER BY r.eff_rank, r.created_at, r.id
LIMIT :batch
"""

# Statement 2 — atomic claim. Re-verifies status under the row lock, so
# snapshot staleness can only ever cause a *skip*, never a double-claim.
_CLAIM_SQL_PG = """
UPDATE upload_jobs_v2 j
SET status               = 'claimed',
    claimed_by           = :worker_id,
    lease_expires_at     = now() + make_interval(secs => :lease_seconds),
    attempts             = j.attempts + 1,
    dispatched_at        = now(),
    updated_at           = now()
WHERE j.id IN (
    SELECT id FROM upload_jobs_v2
    WHERE id = ANY(:candidate_ids)
      AND status = 'queued'
      AND next_attempt_at <= now()
    FOR UPDATE SKIP LOCKED
)
RETURNING j.id, j.user_id, j.publish_task_id
"""


def claim_jobs(worker_id: str, batch: int = 4) -> list[ClaimedJob]:
    """Claim up to ``batch`` dispatchable jobs for this worker.

    Returns the claimed rows (possibly empty). Never raises on an empty
    queue; DB errors propagate so the worker loop can log + retry.
    """
    batch = max(1, int(batch))
    db = SessionLocal()
    try:
        if not _is_pg():
            return _claim_jobs_sqlite(db, worker_id, batch)

        pct = reserved_pct()
        rows = db.execute(text(_ELIGIBILITY_SQL_PG), {
            "aging_step_min": aging_step_min(),
            "default_cap": default_user_cap(),
            "max_attempts": max_attempts(),
            "global_cap": global_active_cap(),
            "pct_critical": pct["critical"],
            "pct_high": pct["high"],
            "pct_normal": pct["normal"],
            "batch": batch,
        }).fetchall()
        if not rows:
            db.rollback()
            return []

        candidate_ids = [int(r[0]) for r in rows]
        eff_by_id = {int(r[0]): RANK_PRIORITY.get(int(r[1]), "normal") for r in rows}

        claimed = db.execute(text(_CLAIM_SQL_PG), {
            "worker_id": worker_id[:64],
            "lease_seconds": lease_seconds(),
            "candidate_ids": candidate_ids,
        }).fetchall()

        # Freeze the aging-adjusted effective priority for honest
        # per-tier accounting (matches old scheduler semantics). At most
        # 4 tiny UPDATEs per batch — one per distinct effective rank.
        by_eff: dict[str, list[int]] = {}
        for r in claimed:
            by_eff.setdefault(eff_by_id.get(int(r[0]), "normal"), []).append(int(r[0]))
        for eff, ids in by_eff.items():
            db.execute(text(
                "UPDATE upload_jobs_v2 SET priority_at_dispatch = :eff "
                "WHERE id = ANY(:ids)"
            ), {"eff": eff, "ids": ids})

        db.commit()
        return [
            ClaimedJob(
                id=int(r[0]),
                user_id=(int(r[1]) if r[1] is not None else None),
                publish_task_id=int(r[2]),
                priority_at_dispatch=eff_by_id.get(int(r[0]), "normal"),
            )
            for r in claimed
        ]
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        raise
    finally:
        db.close()


def _claim_jobs_sqlite(db, worker_id: str, batch: int) -> list[ClaimedJob]:
    """Simplified single-process claim for SQLite (tests only): base
    priority + FIFO, per-call atomic via the connection's write lock.
    No aging / reserved floors — Postgres is the real target."""
    rows = db.execute(text(
        "SELECT id, user_id, publish_task_id, priority FROM upload_jobs_v2 "
        "WHERE status = 'queued' "
        "  AND (next_attempt_at IS NULL OR next_attempt_at <= CURRENT_TIMESTAMP) "
        "  AND attempts < :max_attempts "
        "ORDER BY CASE priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1 "
        "         WHEN 'normal' THEN 2 ELSE 3 END, created_at, id "
        "LIMIT :batch"
    ), {"max_attempts": max_attempts(), "batch": batch}).fetchall()
    out: list[ClaimedJob] = []
    for r in rows:
        res = db.execute(text(
            "UPDATE upload_jobs_v2 SET status='claimed', claimed_by=:w, "
            "lease_expires_at = datetime('now', '+' || :lease || ' seconds'), "
            "attempts = attempts + 1, priority_at_dispatch = :prio, "
            "dispatched_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP "
            "WHERE id = :id AND status = 'queued'"
        ), {"w": worker_id[:64], "lease": lease_seconds(),
            "prio": str(r[3] or "normal"), "id": int(r[0])})
        if res.rowcount == 1:
            out.append(ClaimedJob(
                id=int(r[0]),
                user_id=(int(r[1]) if r[1] is not None else None),
                publish_task_id=int(r[2]),
                priority_at_dispatch=str(r[3] or "normal"),
            ))
    db.commit()
    return out


# ─── Lease renewal (worker heartbeat) ────────────────────────────────


def renew_leases(worker_id: str, job_ids: Sequence[int]) -> set[int]:
    """Extend the lease on every job this worker still owns.

    Returns the set of ids that were successfully renewed. Any id NOT
    in the returned set was reaped (lease expired and another actor
    took it back) — the worker should treat it as cancelled and stop
    pushing writes for it (the fenced writes would no-op anyway).
    """
    ids = [int(i) for i in job_ids]
    if not ids:
        return set()
    db = SessionLocal()
    try:
        if _is_pg():
            rows = db.execute(text(
                "UPDATE upload_jobs_v2 "
                "SET lease_expires_at = now() + make_interval(secs => :lease), "
                "    updated_at = now() "
                "WHERE id = ANY(:ids) AND claimed_by = :w "
                "  AND status IN ('claimed','branding','ready_to_upload','uploading') "
                "RETURNING id"
            ), {"lease": lease_seconds(), "ids": ids, "w": worker_id[:64]}).fetchall()
            renewed = {int(r[0]) for r in rows}
        else:
            renewed = set()
            for jid in ids:
                res = db.execute(text(
                    "UPDATE upload_jobs_v2 "
                    "SET lease_expires_at = datetime('now', '+' || :lease || ' seconds'), "
                    "    updated_at = CURRENT_TIMESTAMP "
                    "WHERE id = :id AND claimed_by = :w "
                    "  AND status IN ('claimed','branding','ready_to_upload','uploading')"
                ), {"lease": lease_seconds(), "id": jid, "w": worker_id[:64]})
                if res.rowcount == 1:
                    renewed.add(jid)
        db.commit()
        return renewed
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        log.exception("job_queue.renew_leases failed for worker=%s", worker_id)
        return set(int(i) for i in ids)  # fail open: don't self-cancel on a DB blip
    finally:
        db.close()


# ─── Retry / requeue ─────────────────────────────────────────────────


def requeue_for_retry(job_id: int, worker_id: Optional[str], reason: str) -> str:
    """Hand a job back to the queue with backoff. Fenced: when
    ``worker_id`` is given, only the claim holder can requeue (a zombie
    thread's late requeue silently no-ops).

    Returns 'requeued' | 'exhausted' | 'stale'.
      exhausted → attempts >= max; caller must terminal-fail + refund
                  (job_queue does not touch credits — layering).
      stale     → fence mismatch; another actor owns the row now.
    """
    db = SessionLocal()
    try:
        row = db.execute(text(
            "SELECT attempts, claimed_by FROM upload_jobs_v2 WHERE id = :id"
        ), {"id": int(job_id)}).fetchone()
        if row is None:
            return "stale"
        attempts = int(row[0] or 0)
        holder = row[1]
        if worker_id is not None and holder is not None and holder != worker_id[:64]:
            return "stale"
        if attempts >= max_attempts():
            return "exhausted"

        delay = backoff_seconds(attempts)
        fence = "AND (claimed_by = :w OR claimed_by IS NULL)" if worker_id else ""
        if _is_pg():
            res = db.execute(text(
                "UPDATE upload_jobs_v2 "
                "SET status='queued', claimed_by=NULL, lease_expires_at=NULL, "
                "    next_attempt_at = now() + make_interval(secs => :delay), "
                "    last_error = :err, updated_at = now() "
                f"WHERE id = :id {fence}"
            ), {"delay": delay, "err": (reason or "transient")[:1000],
                "id": int(job_id), **({"w": worker_id[:64]} if worker_id else {})})
        else:
            res = db.execute(text(
                "UPDATE upload_jobs_v2 "
                "SET status='queued', claimed_by=NULL, lease_expires_at=NULL, "
                "    next_attempt_at = datetime('now', '+' || :delay || ' seconds'), "
                "    last_error = :err, updated_at = CURRENT_TIMESTAMP "
                f"WHERE id = :id {fence}"
            ), {"delay": delay, "err": (reason or "transient")[:1000],
                "id": int(job_id), **({"w": worker_id[:64]} if worker_id else {})})
        db.commit()
        if res.rowcount == 1:
            log.info("job_queue: job=%d requeued (attempts=%d, retry in ~%ds): %s",
                     job_id, attempts, delay, (reason or "")[:200])
            return "requeued"
        return "stale"
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        log.exception("job_queue.requeue_for_retry failed for job=%d", job_id)
        return "stale"
    finally:
        db.close()


def requeue_for_stage_handoff(job_id: int, worker_id: Optional[str], reason: str) -> str:
    """Stage-split conveyor handoff: branding finished, so RELEASE the slot and
    re-queue the job for IMMEDIATE re-claim by its upload phase.

    This is NOT a retry — branding SUCCEEDED. Differences from
    :func:`requeue_for_retry`:
      * ``next_attempt_at = now()`` — no backoff; claim it right away.
      * ``attempts`` is DECREMENTED by 1 (floor 0) to NEUTRALISE the +1 the
        branding claim just made (claim_jobs bumps attempts on every claim).
        Net effect: the brand claim is "free", so the upload phase keeps the
        SAME retry budget a non-split (brand+upload-in-one-claim) job would
        have had — the split is invisible to max_attempts.
      * No max-attempts gate: we are only moving an already-branded row to its
        upload phase, not retrying a failure.

    Fenced like requeue_for_retry: only the current claim holder can hand off,
    so a zombie/abandoned thread's late handoff no-ops against a cleared or
    reassigned ``claimed_by``. Returns ``'requeued'`` | ``'stale'``.
    """
    db = SessionLocal()
    try:
        row = db.execute(text(
            "SELECT claimed_by FROM upload_jobs_v2 WHERE id = :id"
        ), {"id": int(job_id)}).fetchone()
        if row is None:
            return "stale"
        holder = row[0]
        if worker_id is not None and holder is not None and holder != worker_id[:64]:
            return "stale"

        fence = "AND (claimed_by = :w OR claimed_by IS NULL)" if worker_id else ""
        params = {"err": (reason or "stage-handoff")[:1000], "id": int(job_id)}
        if worker_id:
            params["w"] = worker_id[:64]
        if _is_pg():
            res = db.execute(text(
                "UPDATE upload_jobs_v2 "
                "SET status='queued', claimed_by=NULL, lease_expires_at=NULL, "
                "    next_attempt_at = now(), "
                "    attempts = GREATEST(attempts - 1, 0), "
                "    last_error = :err, updated_at = now() "
                f"WHERE id = :id {fence}"
            ), params)
        else:
            res = db.execute(text(
                "UPDATE upload_jobs_v2 "
                "SET status='queued', claimed_by=NULL, lease_expires_at=NULL, "
                "    next_attempt_at = CURRENT_TIMESTAMP, "
                "    attempts = MAX(attempts - 1, 0), "
                "    last_error = :err, updated_at = CURRENT_TIMESTAMP "
                f"WHERE id = :id {fence}"
            ), params)
        db.commit()
        if res.rowcount == 1:
            log.info("job_queue: job=%d brand→upload handoff (re-queued for immediate "
                     "upload claim): %s", job_id, (reason or "")[:200])
            return "requeued"
        return "stale"
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        log.exception("job_queue.requeue_for_stage_handoff failed for job=%d", job_id)
        return "stale"
    finally:
        db.close()


# ─── Reaper (lease expiry → back to queue) ───────────────────────────


def reap_expired() -> list[int]:
    """Return every active-status job whose lease expired to the queue.

    The ``lease_expires_at IS NULL`` arm sweeps legacy mid-state rows
    from before the durable-queue cutover (and anything that slipped
    through), permanently closing the "crashed mid-branding, abandoned
    forever" hole. Runs from the leader-elected cron every 60s.
    """
    db = SessionLocal()
    try:
        if _is_pg():
            rows = db.execute(text(
                "UPDATE upload_jobs_v2 "
                "SET status='queued', claimed_by=NULL, lease_expires_at=NULL, "
                "    next_attempt_at = now() + interval '30 seconds', "
                "    last_error = 'lease expired (worker ' || COALESCE(claimed_by, '?') "
                "                 || ' presumed dead)', "
                "    updated_at = now() "
                "WHERE status IN ('claimed','branding','ready_to_upload','uploading') "
                "  AND (lease_expires_at < now() "
                "       OR (lease_expires_at IS NULL "
                "           AND updated_at < now() - interval '15 minutes')) "
                "RETURNING id"
            )).fetchall()
        else:
            rows = db.execute(text(
                "SELECT id FROM upload_jobs_v2 "
                "WHERE status IN ('claimed','branding','ready_to_upload','uploading') "
                "  AND (lease_expires_at < CURRENT_TIMESTAMP "
                "       OR (lease_expires_at IS NULL "
                "           AND updated_at < datetime('now', '-15 minutes')))"
            )).fetchall()
            for r in rows:
                db.execute(text(
                    "UPDATE upload_jobs_v2 SET status='queued', claimed_by=NULL, "
                    "lease_expires_at=NULL, "
                    "next_attempt_at = datetime('now', '+30 seconds'), "
                    "last_error='lease expired (worker presumed dead)', "
                    "updated_at = CURRENT_TIMESTAMP WHERE id = :id"
                ), {"id": int(r[0])})
        db.commit()
        ids = [int(r[0]) for r in rows]
        if ids:
            log.warning("job_queue.reap_expired: requeued %d orphaned job(s): %s",
                        len(ids), ids[:20])
        return ids
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        log.exception("job_queue.reap_expired failed")
        return []
    finally:
        db.close()


# ─── Un-parking (quota window reset) ─────────────────────────────────


def unpark_batch(batch: int = 25) -> list[dict]:
    """Move up to ``batch`` parked_quota jobs back to 'queued'.

    Returns [{id, user_id, predicted_credit_cost, upload_path,
    publish_kind, predicted_quota_units}] so the caller (cron_runner)
    can RE-RESERVE credits — ``_park_quota`` refunded them at park
    time, so skipping this step would make the retry free.
    """
    _cols = ("id, user_id, predicted_credit_cost, upload_path, "
             "publish_kind, predicted_quota_units")
    db = SessionLocal()
    try:
        if _is_pg():
            rows = db.execute(text(
                "WITH unparked AS ("
                "    SELECT id FROM upload_jobs_v2 "
                "    WHERE status = 'parked_quota' "
                "    ORDER BY created_at "
                "    LIMIT :batch "
                "    FOR UPDATE SKIP LOCKED"
                ") "
                "UPDATE upload_jobs_v2 j "
                "SET status='queued', next_attempt_at=now(), claimed_by=NULL, "
                "    lease_expires_at=NULL, last_error=NULL, updated_at=now() "
                "FROM unparked u WHERE j.id = u.id "
                "RETURNING j.id, j.user_id, j.predicted_credit_cost, "
                "          j.upload_path, j.publish_kind, j.predicted_quota_units"
            ), {"batch": int(batch)}).fetchall()
        else:
            rows = db.execute(text(
                f"SELECT {_cols} FROM upload_jobs_v2 "
                "WHERE status = 'parked_quota' ORDER BY created_at LIMIT :batch"
            ), {"batch": int(batch)}).fetchall()
            for r in rows:
                db.execute(text(
                    "UPDATE upload_jobs_v2 SET status='queued', "
                    "next_attempt_at=CURRENT_TIMESTAMP, claimed_by=NULL, "
                    "lease_expires_at=NULL, last_error=NULL, "
                    "updated_at=CURRENT_TIMESTAMP WHERE id = :id"
                ), {"id": int(r[0])})
        db.commit()
        return [
            {"id": int(r[0]),
             "user_id": (int(r[1]) if r[1] is not None else None),
             "predicted_credit_cost": int(r[2] or 0),
             "upload_path": str(r[3] or "direct"),
             "publish_kind": str(r[4] or "video"),
             "predicted_quota_units": int(r[5] or 0)}
            for r in rows
        ]
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        log.exception("job_queue.unpark_batch failed")
        return []
    finally:
        db.close()


# ─── Exhausted-attempts sweep ────────────────────────────────────────


def list_exhausted(limit: int = 100) -> list[dict]:
    """Queued rows that ran out of attempts (the claim query skips
    them, so without this sweep they'd sit invisible forever).

    Read-only: returns [{id, user_id, predicted_credit_cost, last_error}]
    for the cron to terminal-fail WITH the refund (credits live a layer
    up — this module stays SQL-only).
    """
    db = SessionLocal()
    try:
        rows = db.execute(text(
            "SELECT id, user_id, predicted_credit_cost, last_error "
            "FROM upload_jobs_v2 "
            "WHERE status = 'queued' AND attempts >= :max_attempts "
            "ORDER BY id LIMIT :limit"
        ), {"max_attempts": max_attempts(), "limit": int(limit)}).fetchall()
        return [
            {"id": int(r[0]),
             "user_id": (int(r[1]) if r[1] is not None else None),
             "predicted_credit_cost": int(r[2] or 0),
             "last_error": (r[3] or "")}
            for r in rows
        ]
    finally:
        db.close()


# ─── Observability ───────────────────────────────────────────────────


def queue_depth_snapshot() -> dict:
    """Live queue stats for /metrics + /admin/upload-v2 (one query)."""
    db = SessionLocal()
    try:
        rows = db.execute(text(
            "SELECT status, COALESCE(priority, 'normal'), count(*) "
            "FROM upload_jobs_v2 "
            "WHERE status IN ('queued','claimed','branding',"
            "                 'ready_to_upload','uploading','parked_quota') "
            "GROUP BY status, COALESCE(priority, 'normal')"
        )).fetchall()
        snap: dict = {
            "queued_by_priority": {},
            "active_total": 0,
            "parked_quota": 0,
        }
        for status, prio, cnt in rows:
            cnt = int(cnt)
            if status == "queued":
                snap["queued_by_priority"][str(prio)] = (
                    snap["queued_by_priority"].get(str(prio), 0) + cnt
                )
            elif status == "parked_quota":
                snap["parked_quota"] += cnt
            else:
                snap["active_total"] += cnt
        snap["queued_total"] = sum(snap["queued_by_priority"].values())
        return snap
    except Exception:
        log.exception("job_queue.queue_depth_snapshot failed")
        return {"queued_by_priority": {}, "active_total": 0,
                "parked_quota": 0, "queued_total": 0}
    finally:
        db.close()


__all__ = [
    "ClaimedJob",
    "ACTIVE_STATUSES",
    "PRIORITY_RANK",
    "claim_jobs",
    "renew_leases",
    "requeue_for_retry",
    "reap_expired",
    "unpark_batch",
    "list_exhausted",
    "queue_depth_snapshot",
    "backoff_seconds",
    "max_attempts",
    "lease_seconds",
    "global_active_cap",
]
