"""Durable publish worker — claims UploadJobV2 rows from the Postgres
SKIP LOCKED queue (``services/job_queue.py``) and drives them through
``services/upload_dispatch.process``.

Runs in two modes with IDENTICAL behaviour:

* **Embedded** (Windows dev): ``main.py``'s startup hook calls
  :func:`start` when ``KAIZER_DURABLE_QUEUE=1`` — an asyncio task in
  the web process. The ProactorEventLoop the V4 ffmpeg subprocesses
  need is untouched (dispatch work runs in plain threads).
* **Standalone** (Linux/cloud): ``python -m services.publish_worker``
  × N processes on M machines. Scale upload throughput by adding
  processes — the SKIP LOCKED claim makes every job single-winner with
  zero coordination beyond Postgres.

Reliability contract (Wave 1 of the enterprise push):

* The worker NEVER decides job state from an exception — that's how
  the old scheduler turned transient 503s into permanent failures.
  ``upload_dispatch.process`` owns all persistence and reports an
  :class:`~services.upload_dispatch.Outcome`; the worker only maps
  outcomes to PublishTask counter bumps.
* Heartbeat renews leases every LEASE/5; a worker that dies simply
  stops renewing and the cron reaper hands its jobs back to the queue.
* A watchdog (`asyncio.wait_for`) abandons threads stuck past the
  dispatch deadline + grace; the cooperative deadline checks inside
  dispatch make this a rare backstop, and the claimed_by fence makes
  abandoned zombie threads harmless.
* Counter bumps are atomic SQL (``completed_count = completed_count
  + 1``) — no read-modify-write lost updates across workers.
"""
from __future__ import annotations

import asyncio
import logging
import os
import socket
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from sqlalchemy import text

from database import SessionLocal
from services import job_queue
from services import upload_dispatch

log = logging.getLogger("kaizer.publish_worker")

# ─── Identity + tunables ─────────────────────────────────────────────

WORKER_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"[:64]


def _concurrency() -> int:
    try:
        return max(1, int(os.environ.get("KAIZER_WORKER_CONCURRENCY", "4")))
    except Exception:
        return 4


def _claim_batch() -> int:
    try:
        return max(1, int(os.environ.get("KAIZER_WORKER_CLAIM_BATCH", "4")))
    except Exception:
        return 4


def _poll_interval_s() -> float:
    try:
        return max(0.25, float(os.environ.get("KAIZER_WORKER_POLL_SECONDS", "2")))
    except Exception:
        return 2.0


def _dispatch_timeout_s() -> int:
    """Overall per-job wall-clock budget (branding + upload). The
    cooperative deadline inside dispatch fires at this; the asyncio
    watchdog backstops at +60s for threads that stopped cooperating."""
    try:
        return max(60, int(os.environ.get("KAIZER_DISPATCH_TIMEOUT_SECONDS", "3600")))
    except Exception:
        return 3600


def _shutdown_grace_s() -> int:
    try:
        return max(0, int(os.environ.get("KAIZER_WORKER_SHUTDOWN_GRACE_SECONDS", "30")))
    except Exception:
        return 30


# ─── PublishTask counter finalization (atomic) ───────────────────────

_FINALIZE_SQL = """
UPDATE publish_tasks
SET completed_count = completed_count + :c,
    failed_count    = failed_count + :f,
    status = CASE
        WHEN (completed_count + :c) + (failed_count + :f) >= target_count THEN
            CASE WHEN (failed_count + :f) = 0 THEN 'completed'
                 WHEN (completed_count + :c) = 0 THEN 'failed'
                 ELSE 'partial_failed' END
        ELSE status
    END,
    updated_at = CURRENT_TIMESTAMP
WHERE id = :pt_id
"""


def _bump_publish_task(publish_task_id: int, outcome: upload_dispatch.Outcome) -> None:
    """Map a job outcome to the parent PublishTask's counters.

    COMPLETED → completed_count+1; FAILED → failed_count+1.
    RETRY / PARKED_QUOTA / SKIPPED bump NOTHING — the job is still in
    flight from the user's point of view. (The old scheduler counted a
    quota-park as 'completed', inflating success stats; deliberate fix.)
    """
    c = 1 if outcome == upload_dispatch.Outcome.COMPLETED else 0
    f = 1 if outcome == upload_dispatch.Outcome.FAILED else 0
    if not (c or f):
        return
    db = SessionLocal()
    try:
        db.execute(text(_FINALIZE_SQL), {
            "c": c, "f": f, "pt_id": int(publish_task_id),
        })
        db.commit()
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        log.exception(
            "publish_worker: counter bump failed for publish_task=%d "
            "(self-heals via cron_runner's counter reconciler)",
            publish_task_id,
        )
    finally:
        db.close()


# ─── Per-job runner ──────────────────────────────────────────────────


async def _run_one(
    executor: ThreadPoolExecutor,
    job: job_queue.ClaimedJob,
) -> None:
    """Run one claimed job: dispatch in a thread, watchdog on top,
    outcome → counter bump. Never raises."""
    loop = asyncio.get_running_loop()
    timeout = _dispatch_timeout_s()
    deadline = time.monotonic() + timeout
    started = time.monotonic()
    try:
        fut = loop.run_in_executor(
            executor, upload_dispatch.process, job.id, WORKER_ID, deadline,
        )
        try:
            outcome = await asyncio.wait_for(fut, timeout=timeout + 60)
        except asyncio.TimeoutError:
            # The thread is a zombie: it ignored the cooperative
            # deadline (most likely stuck in a non-timeout'd syscall).
            # Fence it out by handing the job back; its late writes
            # no-op against the cleared claimed_by.
            log.error(
                "publish_worker: WATCHDOG timeout job=%d after %.0fs — "
                "requeueing and abandoning the thread",
                job.id, time.monotonic() - started,
            )
            state = job_queue.requeue_for_retry(
                job.id, WORKER_ID, "dispatch watchdog timeout",
            )
            if state == "exhausted":
                upload_dispatch.fail_terminal(
                    job.id, WORKER_ID,
                    "retries exhausted: dispatch watchdog timeout",
                )
                _bump_publish_task(
                    job.publish_task_id, upload_dispatch.Outcome.FAILED,
                )
            return
        if not isinstance(outcome, upload_dispatch.Outcome):
            # Legacy-shaped return (None) — defensive only; the durable
            # call path always returns an Outcome.
            outcome = upload_dispatch.Outcome.COMPLETED
        log.info(
            "publish_worker: job=%d outcome=%s in %.1fs",
            job.id, outcome.value, time.monotonic() - started,
        )
        _bump_publish_task(job.publish_task_id, outcome)
    except Exception:
        # _run_one must never kill the claim loop. An exception that
        # reaches here is a worker bug (dispatch swallows everything in
        # durable mode) — requeue defensively so the job isn't lost.
        log.exception("publish_worker: _run_one crashed for job=%d", job.id)
        try:
            job_queue.requeue_for_retry(job.id, WORKER_ID, "worker internal error")
        except Exception:
            pass


# ─── Heartbeat (lease renewal) ───────────────────────────────────────


async def _heartbeat_loop(
    stop: asyncio.Event,
    in_flight: dict[int, asyncio.Task],
) -> None:
    interval = max(10, job_queue.lease_seconds() // 5)
    while not stop.is_set():
        try:
            ids = list(in_flight.keys())
            if ids:
                renewed = await asyncio.to_thread(
                    job_queue.renew_leases, WORKER_ID, ids,
                )
                lost = [i for i in ids if i not in renewed]
                if lost:
                    # Reaped from under us — the fenced writes inside
                    # dispatch will abort those threads at their next
                    # checkpoint; nothing to do here but log.
                    log.warning(
                        "publish_worker: lost lease on job(s) %s "
                        "(reaper handed them to another worker)", lost,
                    )
        except Exception:
            log.exception("publish_worker: heartbeat tick failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


# ─── Main worker loop ────────────────────────────────────────────────


async def run_worker(stop: asyncio.Event) -> None:
    """Claim → dispatch → finalize, until ``stop`` is set."""
    concurrency = _concurrency()
    # Oversized executor: watchdog-abandoned zombie threads keep their
    # pool slot until they unblock; the headroom keeps fresh jobs
    # flowing meanwhile.
    executor = ThreadPoolExecutor(
        max_workers=concurrency * 2,
        thread_name_prefix="publish_dispatch",
    )
    in_flight: dict[int, asyncio.Task] = {}
    heartbeat = asyncio.create_task(_heartbeat_loop(stop, in_flight))
    log.info(
        "publish_worker: started worker_id=%s concurrency=%d "
        "lease=%ds dispatch_timeout=%ds",
        WORKER_ID, concurrency, job_queue.lease_seconds(), _dispatch_timeout_s(),
    )
    print(f"[publish_worker] running (worker_id={WORKER_ID}, "
          f"concurrency={concurrency})")

    try:
        while not stop.is_set():
            try:
                free = concurrency - len(in_flight)
                if free > 0:
                    claimed = await asyncio.to_thread(
                        job_queue.claim_jobs, WORKER_ID,
                        min(free, _claim_batch()),
                    )
                    for job in claimed:
                        t = asyncio.create_task(_run_one(executor, job))
                        in_flight[job.id] = t
                        t.add_done_callback(
                            lambda _t, jid=job.id: in_flight.pop(jid, None)
                        )
                    if claimed:
                        # Immediately try to fill remaining slots before
                        # sleeping — burst drain.
                        continue
            except Exception:
                log.exception("publish_worker: claim tick failed")
            try:
                await asyncio.wait_for(stop.wait(), timeout=_poll_interval_s())
            except asyncio.TimeoutError:
                pass
    finally:
        # Graceful shutdown: stop claiming, give in-flight a grace
        # window, then hand survivors back explicitly (leases would
        # cover them anyway, but explicit requeue skips the wait).
        grace = _shutdown_grace_s()
        if in_flight:
            log.info(
                "publish_worker: shutdown — waiting up to %ds for %d job(s)",
                grace, len(in_flight),
            )
            try:
                await asyncio.wait_for(
                    asyncio.gather(*in_flight.values(), return_exceptions=True),
                    timeout=grace,
                )
            except asyncio.TimeoutError:
                survivors = list(in_flight.keys())
                log.warning(
                    "publish_worker: requeueing %d unfinished job(s) on "
                    "shutdown: %s", len(survivors), survivors,
                )
                for jid in survivors:
                    try:
                        await asyncio.to_thread(
                            job_queue.requeue_for_retry, jid, WORKER_ID,
                            "worker shutdown",
                        )
                    except Exception:
                        pass
        heartbeat.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        log.info("publish_worker: stopped worker_id=%s", WORKER_ID)


# ─── Embedded-mode lifecycle (mirrors services/scheduler API) ────────

_stop_event: Optional[asyncio.Event] = None
_worker_task: Optional[asyncio.Task] = None


async def start() -> None:
    """Start the embedded worker inside the web process's event loop.
    Called by main.py's startup hook when KAIZER_DURABLE_QUEUE=1."""
    global _stop_event, _worker_task
    if _worker_task is not None and not _worker_task.done():
        return
    _stop_event = asyncio.Event()
    _worker_task = asyncio.create_task(run_worker(_stop_event))


async def shutdown() -> None:
    global _stop_event, _worker_task
    if _stop_event is not None:
        _stop_event.set()
    if _worker_task is not None:
        try:
            await asyncio.wait_for(
                _worker_task, timeout=_shutdown_grace_s() + 15,
            )
        except Exception:
            _worker_task.cancel()
    _stop_event = None
    _worker_task = None


# ─── Standalone entry: python -m services.publish_worker ────────────


def _main() -> None:
    logging.basicConfig(
        level=os.environ.get("KAIZER_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if sys.platform == "win32":
        # Parity with the web process — and required if this worker
        # ever shells out to ffmpeg via asyncio.
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    async def main() -> None:
        stop = asyncio.Event()

        def _signal_stop(*_a) -> None:
            stop.set()

        try:
            import signal
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, _signal_stop)
                except (NotImplementedError, RuntimeError):
                    # Windows: no add_signal_handler — fall back.
                    signal.signal(sig, lambda *_a: stop.set())
        except Exception:
            pass

        # Standalone workers also race for cron leadership (reaper,
        # un-parker, branding cleanup, …) — exactly one wins.
        cron_task = None
        try:
            from services import cron_runner
            cron_task = asyncio.create_task(cron_runner.cron_leader_loop(stop))
        except Exception:
            log.exception("publish_worker: cron_runner unavailable — "
                          "running without cron leadership")

        await run_worker(stop)
        if cron_task is not None:
            stop.set()
            try:
                await asyncio.wait_for(cron_task, timeout=10)
            except Exception:
                cron_task.cancel()

    asyncio.run(main())


if __name__ == "__main__":
    _main()


__all__ = ["start", "shutdown", "run_worker", "WORKER_ID"]
