"""Background asyncio crons — Phase 3.

Three long-running asyncio tasks the FastAPI lifecycle owns:

  1. ``_branding_cleanup_loop`` (hourly) — calls
     ``services.branding.cleanup_expired()`` to delete R2 artifacts
     older than ``KAIZER_BRANDED_ARTIFACT_TTL_HOURS`` (default 24).

  2. ``_credits_allot_loop`` (daily) — iterates every ``User`` and
     idempotently calls ``services.credits.allot_monthly(...)``. The
     dedupe key inside ``allot_monthly`` is ``(user_id, calendar month
     UTC)`` so re-running the same day is a no-op.

  3. ``_burn_reconcile_loop`` (daily) — calls
     ``services.burn_log.reconcile_window(start=now-24h, end=now)`` so
     the predicted-vs-actual quota burn ledger fills in the actuals.

Each loop is shape-identical:

    while not _shutdown.is_set():
        try:
            <do work>
        except Exception:
            log.exception(...)
        try:
            await asyncio.wait_for(_shutdown.wait(), timeout=<interval>)
        except asyncio.TimeoutError:
            continue

So a clean shutdown wakes the wait IMMEDIATELY and the loop exits on
the next ``not _shutdown.is_set()`` check — no 1-hour-of-sleep waits.

``shutdown()`` sets the event and awaits up to 10s for every task to
drain. If a task is stuck mid-cleanup, it'll be cancelled at that
boundary; the next boot's startup-recovery sweep + the cron's own
``allot_monthly`` idempotency picks up cleanly.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

log = logging.getLogger("kaizer.crons")


# ─── Module-level state (single in-process instance) ──────────────────────


_shutdown: Optional[asyncio.Event] = None
_tasks: List[asyncio.Task] = []
_started: bool = False


# Tunables (intervals). All in seconds. Override in tests via direct
# attribute assignment — these are deliberately module-level so the
# Phase 3 exit test can shrink them to milliseconds for assertions.
BRANDING_CLEANUP_INTERVAL_SECONDS = 3600     # 1 h
CREDITS_ALLOT_INTERVAL_SECONDS = 86400        # 24 h
BURN_RECONCILE_INTERVAL_SECONDS = 86400       # 24 h

# How long ``shutdown()`` waits for tasks to drain after signalling.
_SHUTDOWN_GRACE_SECONDS = 10.0


# ─── Cron 1: branding cleanup ─────────────────────────────────────────────


async def _branding_cleanup_loop() -> None:
    """Hourly: delete expired branded artifacts from R2.

    The actual TTL is owned by ``branding.cleanup_expired`` which reads
    ``KAIZER_BRANDED_ARTIFACT_TTL_HOURS``. We just call it on a fixed
    cadence; the function is a no-op when STORAGE_BACKEND != 'r2'
    (local dev keeps everything around).
    """
    log.info(
        "crons: branding cleanup loop starting (every %d s)",
        BRANDING_CLEANUP_INTERVAL_SECONDS,
    )
    while _shutdown is not None and not _shutdown.is_set():
        try:
            # Lazy import — keeps the module loadable in tests/CI that
            # don't have the branding stack installed.
            from services import branding
            removed = branding.cleanup_expired()
            log.info(
                "crons: branding cleanup removed %d expired artifacts",
                int(removed or 0),
            )
        except Exception:
            log.exception("crons: branding cleanup failed; next tick will retry")

        try:
            await asyncio.wait_for(
                _shutdown.wait(), timeout=BRANDING_CLEANUP_INTERVAL_SECONDS
            )
            # Event fired -> exit loop cleanly.
            break
        except asyncio.TimeoutError:
            # Normal wakeup; loop again.
            continue
        except asyncio.CancelledError:
            log.info("crons: branding cleanup loop cancelled")
            raise
    log.info("crons: branding cleanup loop exiting")


# ─── Cron 2: monthly credit allotment ─────────────────────────────────────


async def _credits_allot_loop() -> None:
    """Daily: for every user, idempotently allot this month's credits.

    ``services.credits.allot_monthly(db, user_id=...)`` dedupes by
    ``(user_id, calendar-month UTC)``, so re-running the same day is
    a no-op. We sweep every user once per tick so a user who upgraded
    plan-tier mid-month doesn't have to wait for the next monthly cron.
    """
    log.info(
        "crons: credit allotment loop starting (every %d s)",
        CREDITS_ALLOT_INTERVAL_SECONDS,
    )
    while _shutdown is not None and not _shutdown.is_set():
        try:
            from database import SessionLocal
            from services import credits as credits_svc
            import models

            db = SessionLocal()
            seeded = 0
            skipped = 0
            try:
                # User has plan_tier_id NULL only when the backfill
                # migration didn't run; ``allot_monthly`` raises in that
                # case which we catch + count as a skip.
                users = db.query(models.User).all()
                for u in users:
                    try:
                        result = credits_svc.allot_monthly(db, user_id=int(u.id))
                        if result is not None:
                            seeded += 1
                        else:
                            skipped += 1
                    except credits_svc.CreditLedgerArgumentError as exc:
                        # User has no plan_tier_id, or the tier row is
                        # missing. Don't kill the whole sweep.
                        log.warning(
                            "crons: credits.allot_monthly skipped user_id=%s (%s)",
                            u.id, exc,
                        )
                        skipped += 1
                        continue
                    except Exception:
                        log.exception(
                            "crons: credits.allot_monthly raised for "
                            "user_id=%s; continuing", u.id,
                        )
                        skipped += 1
                        continue
                try:
                    db.commit()
                except Exception:
                    log.exception("crons: credits.allot_monthly commit failed")
                    db.rollback()
            finally:
                db.close()

            log.info(
                "crons: credits allotted=%d skipped=%d this tick",
                seeded, skipped,
            )
        except Exception:
            log.exception("crons: credit allotment failed; next tick will retry")

        try:
            await asyncio.wait_for(
                _shutdown.wait(), timeout=CREDITS_ALLOT_INTERVAL_SECONDS
            )
            break
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            log.info("crons: credit allotment loop cancelled")
            raise
    log.info("crons: credit allotment loop exiting")


# ─── Cron 3: burn log reconciliation ──────────────────────────────────────


async def _burn_reconcile_loop() -> None:
    """Daily: reconcile yesterday's QuotaBurnLog rows.

    The current ``burn_log.reconcile_window`` is a Phase 2 stub that
    fills ``reconciled_actual_cost = predicted_cost`` for successful
    rows (failed rows stay NULL until the production reconciliation
    against Google's reported usage lands). That's fine — the schema +
    timestamps are real; only the inner ``actual = ...`` line will
    change when we wire up Google's Quota Status API.
    """
    log.info(
        "crons: burn log reconcile loop starting (every %d s)",
        BURN_RECONCILE_INTERVAL_SECONDS,
    )
    while _shutdown is not None and not _shutdown.is_set():
        try:
            from database import SessionLocal
            from services import burn_log

            db = SessionLocal()
            try:
                end = datetime.now(timezone.utc)
                start = end - timedelta(hours=24)
                report = burn_log.reconcile_window(db, start=start, end=end)
                try:
                    db.commit()
                except Exception:
                    log.exception("crons: burn reconcile commit failed")
                    db.rollback()
                log.info(
                    "crons: burn reconciled rows_seen=%d rows_reconciled=%d "
                    "predicted=%d actual=%d delta=%d",
                    int(getattr(report, "rows_seen", 0)),
                    int(getattr(report, "rows_reconciled", 0)),
                    int(getattr(report, "total_predicted", 0)),
                    int(getattr(report, "total_actual", 0)),
                    int(getattr(report, "delta", 0)),
                )
            finally:
                db.close()
        except Exception:
            log.exception("crons: burn reconcile failed; next tick will retry")

        try:
            await asyncio.wait_for(
                _shutdown.wait(), timeout=BURN_RECONCILE_INTERVAL_SECONDS
            )
            break
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            log.info("crons: burn reconcile loop cancelled")
            raise
    log.info("crons: burn reconcile loop exiting")


# ─── Lifecycle ────────────────────────────────────────────────────────────


async def start() -> None:
    """Spawn the three background cron tasks. Idempotent — calling
    twice is a no-op + warning."""
    global _shutdown, _tasks, _started

    if _started:
        log.warning("crons.start: already started; ignoring")
        return

    _shutdown = asyncio.Event()
    _tasks = [
        asyncio.create_task(_branding_cleanup_loop(), name="kaizer-cron-branding"),
        asyncio.create_task(_credits_allot_loop(),    name="kaizer-cron-credits"),
        asyncio.create_task(_burn_reconcile_loop(),   name="kaizer-cron-burn"),
    ]
    _started = True
    log.info("crons.start: 3 background tasks launched")


async def shutdown() -> None:
    """Signal all tasks to exit, wait up to 10 s for them to drain.

    Any task that doesn't drain in time gets cancelled. Idempotent;
    a second call is a no-op.
    """
    global _started

    if not _started or _shutdown is None:
        return

    log.info("crons.shutdown: signalling; tasks=%d", len(_tasks))
    _shutdown.set()

    if _tasks:
        try:
            done, pending = await asyncio.wait(
                _tasks, timeout=_SHUTDOWN_GRACE_SECONDS
            )
            for t in pending:
                log.warning(
                    "crons.shutdown: task %r didn't drain in %.1fs; cancelling",
                    t.get_name() if hasattr(t, "get_name") else repr(t),
                    _SHUTDOWN_GRACE_SECONDS,
                )
                t.cancel()
            # Allow cancellations to settle without raising.
            if pending:
                try:
                    await asyncio.gather(*pending, return_exceptions=True)
                except Exception:
                    pass
        except Exception:
            log.exception("crons.shutdown: wait failed; cancelling all tasks")
            for t in _tasks:
                t.cancel()

    _tasks = []
    _started = False
    log.info("crons.shutdown: all background tasks stopped")


def snapshot() -> dict:
    """Observability snapshot: number of running cron tasks + their states."""
    return {
        "started": bool(_started),
        "task_count": len(_tasks),
        "tasks": [
            {
                "name": t.get_name() if hasattr(t, "get_name") else repr(t),
                "done": bool(t.done()),
                "cancelled": bool(t.cancelled()),
            }
            for t in _tasks
        ],
    }


__all__ = ["start", "shutdown", "snapshot"]
