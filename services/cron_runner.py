"""Leader-elected cron runner for the durable publish queue (Wave 1).

Supersedes the *lifecycle* of ``services/crons.py`` when
``KAIZER_DURABLE_QUEUE=1`` (the three original job bodies are
re-registered here unchanged in spirit). Adds the four reliability
crons the old system was missing:

* **lease_reaper** (60s) — returns crashed workers' jobs to the queue.
  Closes the "crashed mid-branding/uploading, abandoned forever" hole;
  recovery is now continuous at runtime, not boot-only.
* **quota_unpark** (15min + midnight America/Los_Angeles, YouTube's
  quota reset) — parked_quota jobs go back to 'queued' WITH credits
  re-reserved (park refunded them; without the re-reserve the retry
  would be free).
* **exhausted_sweep** (5min) — queued rows that ran out of attempts
  get terminal-failed with the idempotent refund (the claim query
  skips them, so without this they'd sit invisible forever).
* **counter_reconcile** (1h) — recomputes PublishTask counters from
  terminal job rows for tasks stuck in 'dispatched' (self-heals the
  rare zombie-completion window where no worker bumped the counter).

Leader election: ``pg_try_advisory_lock`` on a dedicated, pinned
connection. Session-scoped — the lock dies with the connection, so a
crashed leader is replaced within ~30s by any other worker. Exactly
ONE process across N workers × M machines runs the crons; every cron
body is idempotent, so a failover double-run is harmless. On SQLite
(tests) the election trivially always wins.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional

from sqlalchemy import text

from database import SessionLocal, engine

log = logging.getLogger("kaizer.cron_runner")

_LEADER_LOCK_NAME = "kaizer_cron_leader"
_TICK_SECONDS = 15.0
_RETRY_LEADERSHIP_SECONDS = 30.0


# ─── Cron job bodies (each: sync, idempotent, own session) ──────────


def run_reaper() -> None:
    """Requeue active-status jobs whose lease expired (worker died),
    then align the idempotency layer's own orphan sweep so a reaped
    job isn't blocked by its stale ``publish_attempts.in_flight`` row."""
    from services import job_queue

    reaped = job_queue.reap_expired()
    if not reaped:
        return
    try:
        from services import idempotency as idem
        db = SessionLocal()
        try:
            # Align staleness with 2× the lease so there's no dead zone
            # where the job is requeued but its attempt row still reads
            # in_flight (check_or_register would back off until 600s).
            idem.recover_orphans(
                db, stale_after_seconds=job_queue.lease_seconds() * 2,
            )
            db.commit()
        finally:
            db.close()
    except Exception:
        log.exception("cron_runner: idempotency orphan sweep failed "
                      "(reaped jobs will still retry; the in_flight row "
                      "ages out on its own)")


_last_unpark_midnight_key: Optional[str] = None


def _pacific_midnight_key(now_utc: datetime) -> str:
    """Date string of 'today' in America/Los_Angeles — YouTube's quota
    window resets at midnight Pacific. When this key changes, a new
    quota day began."""
    try:
        from zoneinfo import ZoneInfo
        return now_utc.astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    except Exception:
        # zoneinfo missing tzdata (slim Windows installs) — fall back
        # to UTC-8 fixed offset; off by an hour in DST, acceptable for
        # an un-parker that also runs opportunistically every 15 min.
        return (now_utc - timedelta(hours=8)).strftime("%Y-%m-%d")


def run_unpark() -> None:
    """Return parked_quota jobs to the queue when quota headroom exists.

    Two triggers, both handled here (the cron itself ticks every 15min):
      * a new Pacific quota day began since the last unpark → drain
      * opportunistic: remaining quota > KAIZER_UNPARK_HEADROOM_UNITS

    Every unparked job gets its credits RE-RESERVED (park refunded
    them). Insufficient balance → terminal fail WITHOUT refund (none
    held) via upload_dispatch.fail_terminal — whose refund dedupe sees
    the park-time refund row and correctly skips a second one.
    """
    from services import job_queue

    global _last_unpark_midnight_key
    now_key = _pacific_midnight_key(datetime.now(timezone.utc))
    new_day = (_last_unpark_midnight_key != now_key)

    if not new_day:
        # Opportunistic path — only when we KNOW there's headroom, so
        # unpark→reserve-fails→re-park churn can't happen.
        try:
            headroom = int(os.environ.get("KAIZER_UNPARK_HEADROOM_UNITS", "2000"))
            from youtube import quota_v2
            db = SessionLocal()
            try:
                snap = quota_v2.snapshot(db)
            finally:
                db.close()
            if int(snap.get("remaining", 0)) < headroom:
                return
        except Exception:
            log.exception("cron_runner: quota snapshot failed — skipping "
                          "opportunistic unpark this tick")
            return

    try:
        batch_size = max(1, int(os.environ.get("KAIZER_UNPARK_BATCH", "25")))
    except Exception:
        batch_size = 25

    total = 0
    while True:
        rows = job_queue.unpark_batch(batch_size)
        if not rows:
            break
        total += len(rows)
        _re_reserve_credits(rows)
        if len(rows) < batch_size:
            break
        if total >= 500:  # safety valve per tick
            break
    if total:
        log.info("cron_runner: un-parked %d quota-parked job(s) (%s)",
                 total, "new quota day" if new_day else "headroom")
    _last_unpark_midnight_key = now_key


def _re_reserve_credits(rows: list[dict]) -> None:
    """Re-reserve the predicted credit cost for each unparked job."""
    from services import credits as credits_svc
    from services import upload_dispatch

    db = SessionLocal()
    try:
        for r in rows:
            uid = r.get("user_id")
            cost = int(r.get("predicted_credit_cost") or 0)
            if uid is None or cost <= 0:
                continue
            path = r.get("upload_path") or "direct"
            try:
                credits_svc.reserve(
                    db,
                    user_id=int(uid),
                    cost=cost,
                    reason=("upload_direct" if path == "direct" else "upload_rtmp"),
                    upload_job_id=int(r["id"]),
                    path=path,
                    publish_kind=(r.get("publish_kind") or "video"),
                    predicted_quota_units=int(r.get("predicted_quota_units") or 0),
                )
                db.commit()
            except credits_svc.InsufficientCreditsError:
                db.rollback()
                # No credits held → fail WITHOUT refund. fail_terminal's
                # _refund_already sees the park-time refund row and skips.
                upload_dispatch.fail_terminal(
                    int(r["id"]), None,
                    "insufficient credits to resume after quota park",
                )
                log.warning(
                    "cron_runner: job=%d failed on unpark — user=%s has "
                    "insufficient credits", r["id"], uid,
                )
            except Exception:
                db.rollback()
                log.exception(
                    "cron_runner: credit re-reserve failed for job=%d "
                    "(job stays queued; dispatch-side accounting still "
                    "guards the upload)", r["id"],
                )
    finally:
        db.close()


def run_exhausted_sweep() -> None:
    """Terminal-fail queued rows whose attempts cap is spent."""
    from services import job_queue
    from services import upload_dispatch

    rows = job_queue.list_exhausted(limit=100)
    for r in rows:
        upload_dispatch.fail_terminal(
            int(r["id"]), None,
            f"retries exhausted ({job_queue.max_attempts()}): "
            f"{(r.get('last_error') or 'transient errors')[:500]}",
        )
    if rows:
        log.warning("cron_runner: terminal-failed %d retry-exhausted job(s)",
                    len(rows))


_COUNTER_RECONCILE_SQL = """
UPDATE publish_tasks pt
SET completed_count = sub.done,
    failed_count    = sub.failed,
    status = CASE
        WHEN sub.done + sub.failed >= pt.target_count THEN
            CASE WHEN sub.failed = 0 THEN 'completed'
                 WHEN sub.done = 0 THEN 'failed'
                 ELSE 'partial_failed' END
        ELSE pt.status
    END,
    updated_at = CURRENT_TIMESTAMP
FROM (
    SELECT publish_task_id,
           count(*) FILTER (WHERE status = 'completed') AS done,
           count(*) FILTER (WHERE status IN ('failed', 'cancelled')) AS failed
    FROM upload_jobs_v2
    GROUP BY publish_task_id
) sub
WHERE sub.publish_task_id = pt.id
  AND pt.status IN ('dispatched', 'fanning_out')
  AND pt.updated_at < CURRENT_TIMESTAMP - interval '30 minutes'
  AND (pt.completed_count <> sub.done OR pt.failed_count <> sub.failed)
"""


def run_counter_reconcile() -> None:
    """Self-heal PublishTask counters from the job rows (truth). Only
    touches stale non-terminal tasks whose counters drifted — e.g. the
    zombie-completion window where a video id was recorded but no
    worker lived to bump the counter."""
    if engine.dialect.name != "postgresql":
        return  # FILTER clause is PG; dev/prod are PG, tests skip.
    db = SessionLocal()
    try:
        res = db.execute(text(_COUNTER_RECONCILE_SQL))
        db.commit()
        if res.rowcount:
            log.info("cron_runner: reconciled counters on %d publish_task(s)",
                     res.rowcount)
    except Exception:
        db.rollback()
        log.exception("cron_runner: counter reconcile failed")
    finally:
        db.close()


def run_branding_cleanup() -> None:
    from services import branding
    removed = branding.cleanup_expired()
    log.info("cron_runner: branding cleanup removed %d artifact(s)",
             int(removed or 0))


def run_credits_allot() -> None:
    """Idempotent monthly allotment sweep (same body as crons.py's)."""
    from services import credits as credits_svc
    import models

    db = SessionLocal()
    seeded = skipped = 0
    try:
        for u in db.query(models.User).all():
            try:
                if credits_svc.allot_monthly(db, user_id=int(u.id)) is not None:
                    seeded += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1
                continue
        db.commit()
    except Exception:
        db.rollback()
        log.exception("cron_runner: credit allotment sweep failed")
    finally:
        db.close()
    log.info("cron_runner: credits allotted=%d skipped=%d", seeded, skipped)


def run_burn_reconcile() -> None:
    from services import burn_log
    db = SessionLocal()
    try:
        end = datetime.now(timezone.utc)
        burn_log.reconcile_window(db, start=end - timedelta(hours=24), end=end)
        db.commit()
    except Exception:
        db.rollback()
        log.exception("cron_runner: burn reconcile failed")
    finally:
        db.close()


def run_quota_sync() -> None:
    """Hourly: pull the REAL Google-assigned YouTube daily quota via
    the Service Usage API into system_settings, so every gate enforces
    the actual limit (and a granted increase propagates within an hour,
    no deploy). First fire happens within one leader tick of startup."""
    from services import quota_sync
    quota_sync.run_quota_sync()


# ─── Registry ────────────────────────────────────────────────────────

#: (name, interval_seconds, fn) — intervals are module attrs so tests
#: can shrink them (same convention as crons.py).
CRONS: list[tuple[str, int, Callable[[], None]]] = [
    ("lease_reaper",       60,    run_reaper),
    ("exhausted_sweep",    300,   run_exhausted_sweep),
    ("quota_unpark",       900,   run_unpark),
    ("quota_sync",         3600,  run_quota_sync),
    ("counter_reconcile",  3600,  run_counter_reconcile),
    ("branding_cleanup",   3600,  run_branding_cleanup),
    ("credits_allot",      86400, run_credits_allot),
    ("burn_reconcile",     86400, run_burn_reconcile),
]


# ─── Leader election + tick loop ─────────────────────────────────────


def _try_acquire_leadership(conn) -> bool:
    """Session-scoped advisory lock on a PINNED connection. The lock
    auto-releases when the connection dies — crashed leader is replaced
    on the next contender's retry tick."""
    if engine.dialect.name != "postgresql":
        return True  # SQLite tests: single process, always leader.
    try:
        return bool(conn.execute(
            text("SELECT pg_try_advisory_lock(hashtext(:name))"),
            {"name": _LEADER_LOCK_NAME},
        ).scalar())
    except Exception:
        log.exception("cron_runner: leadership probe failed")
        return False


async def cron_leader_loop(stop: asyncio.Event) -> None:
    """Race for leadership; while leader, run due crons every tick.
    Loses leadership only by losing the pinned connection (process
    death / network) — then any other worker takes over in ≤30s."""
    while not stop.is_set():
        conn = None
        try:
            conn = engine.connect()
            if _try_acquire_leadership(conn):
                log.info("cron_runner: acquired cron leadership")
                print("[cron_runner] this process is the cron leader")
                await _leader_ticks(stop, conn)
        except Exception:
            log.exception("cron_runner: leader loop error; retrying")
        finally:
            if conn is not None:
                try:
                    conn.close()  # releases the advisory lock
                except Exception:
                    pass
        try:
            await asyncio.wait_for(stop.wait(),
                                   timeout=_RETRY_LEADERSHIP_SECONDS)
        except asyncio.TimeoutError:
            pass


async def _leader_ticks(stop: asyncio.Event, conn) -> None:
    """Run due crons until stop, or until the pinned connection dies
    (which would silently hand leadership elsewhere — so we probe it
    each tick)."""
    last_run: dict[str, float] = {}
    while not stop.is_set():
        # Liveness probe on the pinned connection: if it died, our lock
        # is gone — step down immediately rather than double-run.
        try:
            conn.execute(text("SELECT 1"))
        except Exception:
            log.warning("cron_runner: pinned connection lost — stepping down")
            return

        now = time.monotonic()
        for name, interval, fn in CRONS:
            if now - last_run.get(name, 0.0) < interval:
                continue
            last_run[name] = now
            try:
                # Cron bodies are sync + DB-bound; keep the event loop
                # responsive by pushing them to a thread.
                await asyncio.to_thread(fn)
            except Exception:
                log.exception("cron_runner: cron %s failed (next tick retries)",
                              name)
        try:
            await asyncio.wait_for(stop.wait(), timeout=_TICK_SECONDS)
        except asyncio.TimeoutError:
            pass


# ─── Embedded-mode lifecycle (mirrors crons.py API) ──────────────────

_stop_event: Optional[asyncio.Event] = None
_leader_task: Optional[asyncio.Task] = None


async def start() -> None:
    global _stop_event, _leader_task
    if _leader_task is not None and not _leader_task.done():
        return
    _stop_event = asyncio.Event()
    _leader_task = asyncio.create_task(cron_leader_loop(_stop_event))


async def shutdown() -> None:
    global _stop_event, _leader_task
    if _stop_event is not None:
        _stop_event.set()
    if _leader_task is not None:
        try:
            await asyncio.wait_for(_leader_task, timeout=10)
        except Exception:
            _leader_task.cancel()
    _stop_event = None
    _leader_task = None


__all__ = [
    "start", "shutdown", "cron_leader_loop", "CRONS",
    "run_reaper", "run_unpark", "run_exhausted_sweep",
    "run_counter_reconcile", "run_branding_cleanup",
    "run_credits_allot", "run_burn_reconcile",
]
