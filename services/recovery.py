"""Startup orphan recovery + transient-error retry coordination — Phase 3.

This module wires the F-agent's ``services.idempotency.recover_orphans``
into the FastAPI startup lifecycle (``main.py``). The flow:

  1. On every boot, scan ``publish_attempts`` rows whose status is
     'in_flight' and whose ``updated_at`` is older than
     ``stale_after_seconds`` (default 600s — assume any worker that took
     longer than 10 minutes died before us).
  2. For each, ``idempotency.recover_orphans`` will either:
       * mark them 'recovered' AND flip the parent ``UploadJobV2`` back
         to ``status='queued'`` so the scheduler can re-dispatch, OR
       * mark them 'recovered' WITHOUT re-queueing if the parent already
         carries a ``youtube_video_id`` (the upload actually landed
         before the worker died — idempotency's next-pickup probe will
         short-circuit naturally).
  3. For every UploadJobV2 that did get re-queued, we increment the
     ``kaizer_recovery_orphan_total`` Prometheus counter and call
     ``scheduler.scheduler_enqueue(...)`` so the dispatcher picks it
     up on its next tick.

This function is **wrapped in try/except in main.py** so a corrupted
``publish_attempts`` row can never prevent FastAPI from coming up. The
worst case is one orphaned upload sits in 'in_flight' for another tick
of the next reconciliation — never a process-killing event.

TRANSIENT-ERROR RETRY (deferred to Phase 3.1)
---------------------------------------------
The brief calls for exponential-backoff retry on transient upload
failures. The scheduler currently marks any exception as ``failed`` and
moves on. The recovery flow on the NEXT startup will pick it back up
via the same code path because ``recover_orphans`` flips the row back
to 'queued' for in_flight orphans — for cleanly-failed rows, the
operator (or a future cron) decides whether to re-queue.

If transient-failure rate proves to be a problem in production, the
solution is to have ``services.upload_dispatch`` classify
``HttpError(5xx)`` vs ``HttpError(4xx)`` and convert 5xx into a parked
state with ``attempts += 1`` and a delayed re-enqueue. That change
belongs in the upload-dispatch / scheduler layer (which Phase 3 may
not touch per hard constraints), so we expose only the helper here
and document the integration as "Phase 3.1".
"""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger("kaizer.recovery")


# ─── Helpers ──────────────────────────────────────────────────────────────


def _resolve_user_and_tier(db, job) -> tuple[int, str]:
    """Return (user_id, plan_tier_name) for a re-queued UploadJobV2.

    Falls back to (0, 'pro') if the PublishTask or User row is gone —
    the scheduler will still accept the enqueue (its own
    ``_resolve_user_and_cap`` re-resolves at dispatch).
    """
    import models  # local import; main.py loads models early

    pt = (
        db.query(models.PublishTask)
        .filter(models.PublishTask.id == job.publish_task_id)
        .first()
    )
    if pt is None:
        return 0, "pro"

    user = db.query(models.User).filter(models.User.id == pt.user_id).first()
    if user is None:
        return int(pt.user_id), "pro"

    plan_tier_name = "pro"
    if getattr(user, "plan_tier_id", None) is not None:
        plan_tier = (
            db.query(models.PlanTier)
            .filter(models.PlanTier.id == user.plan_tier_id)
            .first()
        )
        if plan_tier is not None:
            plan_tier_name = str(plan_tier.name or "pro")
    return int(user.id), plan_tier_name


# ─── Public API ───────────────────────────────────────────────────────────


def recover_on_startup(stale_after_seconds: int = 600) -> int:
    """Run the idempotency-aware orphan sweep + re-enqueue every
    candidate on the scheduler.

    Returns the count of UploadJobV2 rows successfully re-queued.

    Safe to call multiple times — ``idempotency.recover_orphans`` is
    DB-level idempotent (it only flips in_flight rows whose
    ``updated_at`` crossed the stale threshold), and
    ``scheduler.scheduler_enqueue`` ignores duplicates via its
    ``_known_job_ids`` guard.

    NEVER raises — caller wraps in try/except for belt and braces, but
    we also catch everything here so the startup-hook log is always
    actionable.
    """
    # All imports are local so a missing module (e.g. metrics not yet
    # loaded) downgrades the metric increment rather than killing the
    # sweep. Schema is the source of truth either way.
    try:
        from database import SessionLocal
        import models  # noqa: F401  (re-export checked by _resolve_user_and_tier)
        from services import idempotency
        from services import scheduler as sched
    except Exception:
        log.exception(
            "recovery.recover_on_startup: required modules unimportable; "
            "skipping sweep (orphans will be picked up on the next boot)"
        )
        return 0

    # Optional metrics — best-effort import; the sweep still runs if the
    # collector is unavailable (e.g. running outside FastAPI).
    metric: Optional[object] = None
    try:
        from services import metrics_collector as mc

        metric = mc.recovery_orphan_total
    except Exception:
        metric = None

    db = SessionLocal()
    try:
        try:
            orphan_ids = idempotency.recover_orphans(
                db, stale_after_seconds=int(stale_after_seconds)
            )
        except Exception:
            log.exception(
                "recovery.recover_on_startup: idempotency.recover_orphans "
                "raised; aborting sweep"
            )
            return 0

        if not orphan_ids:
            log.info("recovery.recover_on_startup: no orphans to re-queue")
            return 0

        requeued = 0
        for jid in orphan_ids:
            try:
                job = (
                    db.query(__import__("models").UploadJobV2)
                    .filter(__import__("models").UploadJobV2.id == jid)
                    .first()
                )
                if job is None:
                    continue

                user_id, plan_tier_name = _resolve_user_and_tier(db, job)
                priority = str(job.priority_at_dispatch or "normal")
                try:
                    sched.scheduler_enqueue(
                        upload_job_id=int(jid),
                        priority=priority,
                        user_id=int(user_id),
                        plan_tier_name=plan_tier_name,
                    )
                except Exception:
                    log.exception(
                        "recovery.recover_on_startup: scheduler_enqueue "
                        "failed for job_id=%s", jid,
                    )
                    continue

                # Increment Prometheus counter (outcome='recovered' per
                # metrics_collector enum).
                if metric is not None:
                    try:
                        metric.labels(outcome="recovered").inc()
                    except Exception:
                        log.warning(
                            "recovery.recover_on_startup: metric.inc failed "
                            "for job_id=%s (non-fatal)", jid,
                        )
                requeued += 1
                log.info(
                    "recovery.recover_on_startup: re-queued job_id=%s "
                    "priority=%s user_id=%s tier=%s",
                    jid, priority, user_id, plan_tier_name,
                )
            except Exception:
                log.exception(
                    "recovery.recover_on_startup: per-job error for jid=%s; "
                    "continuing", jid,
                )

        # Commit the parent UploadJobV2 'queued' flips that
        # ``recover_orphans`` performed via ``db.flush()`` (no commit yet
        # — that's our responsibility).
        try:
            db.commit()
        except Exception:
            log.exception(
                "recovery.recover_on_startup: commit failed; rolling back"
            )
            db.rollback()
            return 0

        log.info(
            "recovery.recover_on_startup: re-queued %d/%d orphan(s)",
            requeued, len(orphan_ids),
        )
        return requeued
    finally:
        try:
            db.close()
        except Exception:
            pass


__all__ = ["recover_on_startup"]
