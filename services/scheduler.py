"""Scheduler service — Phase 1.C (Scheduler agent C).

Weighted-fair, priority-aware, in-process scheduler that the Fanout
agent (B) enqueues ``UploadJobV2`` rows onto.

PHASE 1 SCOPE
-------------
This is the **skeleton**. The dispatch loop is real:
  - priority precedence (critical > high > normal > low)
  - per-user slot cap (PlanTier.slot_cap_active_uploads)
  - global CPU + network token pools (Decision 9)
  - reserved-capacity floors per tier (env KAIZER_RESERVED_CAPACITY_PCT)
  - aging promotion (env KAIZER_PRIORITY_AGING_MIN)
  - orphan recovery on startup (queued rows from a previous process)

…but ``_run_job`` does NOT call YouTube. It walks the new state machine
synthetically:

    queued → branding → ready_to_upload → uploading → completed

with small ``asyncio.sleep`` calls between transitions so the exit
test can observe the progression. **Phase 2's Upload agent (E)
replaces the body of ``_run_job`` to call the real Branding Worker
then the real Upload Worker.** The outer slot-acquisition + status
transitions + PublishTask counter update + error handling MUST be
preserved.

PUBLIC API (exact symbols Fanout imports — see services/fanout.py
``_resolve_scheduler_enqueue``)
  - ``scheduler_enqueue(upload_job_id, priority, user_id, plan_tier_name)``
    synchronous, safe to call from a request handler.
  - ``start()``  / ``shutdown()`` — FastAPI lifecycle hooks.
  - ``snapshot()`` — observability.

QUEUEING (CONTRACTS.md §5)
--------------------------
A single in-process min-heap keyed by
``(priority_rank, -age_seconds, fifo_seq)``. The heap is rebuilt each
tick if aging promoted any items so the priority change is reflected
in the dispatch order. We do NOT mutate the persisted
``upload_jobs_v2.priority_at_dispatch`` until the job is actually
released for dispatch — aging affects the queue's effective ordering
only, not the per-tier accounting frozen on the row.

CANCELLATION
------------
``shutdown()`` sets an ``asyncio.Event`` and waits up to 10s for the
dispatcher task + every in-flight ``_run_job`` task to finish. The
slot manager's context-manager-based acquire guarantees tokens are
always released even on a hard cancel.

HARD CONSTRAINTS (from task spec)
---------------------------------
* Single in-process instance — no multi-worker (Phase 3).
* Dispatcher loop is cooperatively cancellable in ~1s.
* Every tuning knob (token totals, slot caps, aging step, reserved
  capacity) reads from env at start; never hard-coded.
* ``scheduler_enqueue`` is synchronous; internally schedules onto the
  event loop.
"""
from __future__ import annotations

import asyncio
import heapq
import itertools
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from database import SessionLocal
import models

from services.slot_manager import (
    SlotManager,
    cpu_tokens_from_env,
    net_tokens_from_env,
)
# Lazy import — upload_dispatch lands in Phase 2.E. While it doesn't exist
# yet (or fails to import for any reason), _run_job falls back to the
# Phase 1 skeleton state walk so the scheduler stays operational. The flip
# from skeleton → real upload happens automatically once the module is
# importable. See DECISIONS.md Decision 13.
try:
    from services import upload_dispatch as _upload_dispatch
except Exception as _exc:  # pragma: no cover — defensive
    _upload_dispatch = None
    log_init = logging.getLogger("kaizer.scheduler")
    log_init.warning(
        "scheduler: services.upload_dispatch not importable (%s); "
        "_run_job will use the Phase 1 skeleton state walk", _exc,
    )

log = logging.getLogger("kaizer.scheduler")


# ─── Constants ─────────────────────────────────────────────────────────────

# Lower rank dispatches first.
PRIORITY_RANK: dict[str, int] = {
    "critical": 0,
    "high": 1,
    "normal": 2,
    "low": 3,
}
RANK_TO_PRIORITY: dict[int, str] = {v: k for k, v in PRIORITY_RANK.items()}

# Reserved-capacity floor (% of net tokens) per tier. Phase 1 reads
# from env KAIZER_RESERVED_CAPACITY_PCT (JSON object). Defaults per
# brief §2 / Decision 3.
_DEFAULT_RESERVED_PCT = {"critical": 50, "high": 30, "normal": 15, "low": 5}

# Aging step in minutes (Decision 3 default).
_DEFAULT_AGING_STEP_MIN = 5

# Plan-tier slot cap fallback when the user has no plan_tier row.
_DEFAULT_SLOT_CAP = 5

# Default slot caps if PlanTier row missing (env fallback — Decision 3).
_SLOT_CAP_BY_TIER = {
    "free": int(os.environ.get("KAIZER_SLOT_CAP_FREE", "5")),
    "pro": int(os.environ.get("KAIZER_SLOT_CAP_PRO", "20")),
    "enterprise": int(os.environ.get("KAIZER_SLOT_CAP_ENTERPRISE", "100")),
}

# How long shutdown() waits for in-flight _run_job tasks to drain.
_SHUTDOWN_GRACE_SECONDS = 10.0

# Dispatcher loop tick — how often we look for a dispatchable job
# when the queue is non-empty but blocked on slots/tokens.
_DISPATCH_TICK_SECONDS = 0.1


def _load_reserved_pct() -> dict[str, int]:
    raw = os.environ.get("KAIZER_RESERVED_CAPACITY_PCT", "").strip()
    if not raw:
        return dict(_DEFAULT_RESERVED_PCT)
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("not a JSON object")
        out: dict[str, int] = {}
        for k in PRIORITY_RANK:
            out[k] = int(parsed.get(k, _DEFAULT_RESERVED_PCT[k]))
        total = sum(out.values())
        if total > 100:
            log.warning(
                "scheduler: KAIZER_RESERVED_CAPACITY_PCT sums to %d (>100); "
                "using defaults",
                total,
            )
            return dict(_DEFAULT_RESERVED_PCT)
        return out
    except Exception as exc:
        log.warning(
            "scheduler: KAIZER_RESERVED_CAPACITY_PCT=%r invalid (%s); using defaults",
            raw, exc,
        )
        return dict(_DEFAULT_RESERVED_PCT)


def _load_aging_step_min() -> int:
    raw = os.environ.get("KAIZER_PRIORITY_AGING_MIN")
    if not raw:
        return _DEFAULT_AGING_STEP_MIN
    try:
        n = int(raw)
        if n <= 0:
            raise ValueError("must be positive")
        return n
    except Exception:
        log.warning(
            "scheduler: KAIZER_PRIORITY_AGING_MIN=%r invalid; using %d",
            raw, _DEFAULT_AGING_STEP_MIN,
        )
        return _DEFAULT_AGING_STEP_MIN


# ─── Queue item ────────────────────────────────────────────────────────────


@dataclass(order=True)
class _Item:
    """One row in the scheduler's min-heap.

    ``sort_key`` is what the heap orders by; everything else is
    ``compare=False`` so equal sort keys don't try to compare ints
    against strings.
    """
    sort_key: tuple = field(compare=True)
    upload_job_id: int = field(compare=False, default=0)
    user_id: int = field(compare=False, default=0)
    plan_tier_name: str = field(compare=False, default="pro")
    priority: str = field(compare=False, default="normal")          # current (may be aged up)
    original_priority: str = field(compare=False, default="normal")  # frozen at enqueue
    enqueued_at: float = field(compare=False, default=0.0)
    fifo_seq: int = field(compare=False, default=0)


# ─── Module-level state ───────────────────────────────────────────────────
#
# Single in-process scheduler — module-level globals are fine for Phase
# 1 (no multi-worker). Phase 3 will hoist into a class instance when we
# move to gunicorn.


_loop: Optional[asyncio.AbstractEventLoop] = None
_slot_manager: Optional[SlotManager] = None
_heap: list[_Item] = []
_heap_lock = threading.Lock()  # protects _heap from sync enqueue() callers
_fifo_counter = itertools.count()
_known_job_ids: set[int] = set()  # idempotent re-enqueue guard

_dispatcher_task: Optional[asyncio.Task] = None
_in_flight: set[asyncio.Task] = set()
_shutdown_event: Optional[asyncio.Event] = None
_aging_step_min: int = _DEFAULT_AGING_STEP_MIN
_reserved_pct: dict[str, int] = dict(_DEFAULT_RESERVED_PCT)
_started: bool = False


# ─── Helpers ───────────────────────────────────────────────────────────────


def _make_sort_key(priority: str, age_seconds: float, fifo_seq: int) -> tuple:
    """Sort key tuple: (rank ascending, -age descending, fifo_seq ascending).

    The negative age means older items at the same rank sort before
    younger ones. fifo_seq is a strict-monotonic tiebreaker so
    deterministic insertion order is preserved when ages are equal.
    """
    rank = PRIORITY_RANK.get(priority, PRIORITY_RANK["normal"])
    return (rank, -age_seconds, fifo_seq)


def _bump_priority(current: str) -> str:
    """Aging: raise priority one tier (low → normal → high → critical)."""
    rank = PRIORITY_RANK.get(current, PRIORITY_RANK["normal"])
    if rank == 0:
        return current  # already critical
    return RANK_TO_PRIORITY[rank - 1]


def _slot_cap_for_tier(tier_name: str) -> int:
    """Plan-tier slot cap fallback when no DB row available."""
    return _SLOT_CAP_BY_TIER.get(tier_name, _DEFAULT_SLOT_CAP)


def _reserved_floor_for(priority: str, net_total: int) -> int:
    """How many net tokens are reserved for ``priority`` and tiers
    above it. Lower-priority items can use ``net_total - reserved`` at
    most.
    """
    pct = _reserved_pct.get(priority, _DEFAULT_RESERVED_PCT[priority])
    return max(0, (net_total * pct) // 100)


# ─── Aging ────────────────────────────────────────────────────────────────


def _age_jobs() -> None:
    """Re-evaluate every heap item's effective priority based on age.

    Called every dispatcher tick. Cheap: O(n) on a heap that's almost
    always tiny (max practical depth ≈ a few hundred items).
    """
    if not _heap:
        return
    step_seconds = _aging_step_min * 60
    now = time.monotonic()
    rebuild = False
    with _heap_lock:
        for item in _heap:
            age = now - item.enqueued_at
            # How many step-windows old? Bump by that many tiers,
            # capped at the original priority's rank (lowest rank=0).
            steps = int(age // step_seconds)
            if steps <= 0:
                continue
            target_rank = max(
                0,
                PRIORITY_RANK.get(item.original_priority, PRIORITY_RANK["normal"]) - steps,
            )
            new_priority = RANK_TO_PRIORITY[target_rank]
            if new_priority != item.priority:
                item.priority = new_priority
                item.sort_key = _make_sort_key(new_priority, age, item.fifo_seq)
                rebuild = True
        if rebuild:
            heapq.heapify(_heap)


# ─── Dispatch eligibility ─────────────────────────────────────────────────


def _peek_dispatchable_index() -> Optional[int]:
    """Find the index of the highest-priority heap item we are
    *currently allowed to dispatch* respecting reserved-capacity
    floors.

    Returns the heap index (NOT a pop) so the dispatcher can decide
    whether to run the job (and pop) or back off. None means nothing
    is dispatchable right now.
    """
    if not _slot_manager or not _heap:
        return None
    snap = _slot_manager.snapshot()
    net_total = snap["net_total"]
    net_free = snap["net_free"]
    # Pass over a sorted copy without disturbing the heap.
    sorted_indices = sorted(range(len(_heap)), key=lambda i: _heap[i].sort_key)
    for idx in sorted_indices:
        item = _heap[idx]
        # Free tokens must be enough that, after we take one, every
        # higher-priority tier's floor is still intact.
        rank = PRIORITY_RANK.get(item.priority, PRIORITY_RANK["normal"])
        higher_floor = 0
        for p, r in PRIORITY_RANK.items():
            if r < rank:
                higher_floor += _reserved_floor_for(p, net_total)
        # We need at least 1 free token AND we may not eat into the
        # higher tiers' floor.
        if net_free <= 0:
            continue
        # If taking this token would push net_in_use over (net_total -
        # higher_floor), reject — leave room for the higher tiers.
        if (net_total - net_free + 1) > (net_total - higher_floor):
            continue
        return idx
    return None


# ─── Public API ───────────────────────────────────────────────────────────


def scheduler_enqueue(
    upload_job_id: int,
    priority: str,
    user_id: int,
    plan_tier_name: str,
) -> None:
    """Synchronous enqueue. Safe to call from a request handler before
    the response is returned.

    Idempotent: if the same ``upload_job_id`` is enqueued twice (e.g.
    Fanout retried, or orphan recovery), the second call is a no-op.
    The DB row is the source of truth for whether the job is actually
    pending.
    """
    if priority not in PRIORITY_RANK:
        log.warning(
            "scheduler_enqueue: invalid priority=%r for job=%d; coercing to 'normal'",
            priority, upload_job_id,
        )
        priority = "normal"
    enqueued_at = time.monotonic()
    fifo_seq = next(_fifo_counter)
    item = _Item(
        sort_key=_make_sort_key(priority, 0.0, fifo_seq),
        upload_job_id=int(upload_job_id),
        user_id=int(user_id),
        plan_tier_name=str(plan_tier_name) if plan_tier_name else "pro",
        priority=priority,
        original_priority=priority,
        enqueued_at=enqueued_at,
        fifo_seq=fifo_seq,
    )
    with _heap_lock:
        if upload_job_id in _known_job_ids:
            log.debug(
                "scheduler_enqueue: job=%d already known; skipping duplicate enqueue",
                upload_job_id,
            )
            return
        heapq.heappush(_heap, item)
        _known_job_ids.add(upload_job_id)
    log.info(
        "scheduler_enqueue: job=%d priority=%s user=%d tier=%s queue_depth=%d",
        upload_job_id, priority, user_id, plan_tier_name, len(_heap),
    )


def snapshot() -> dict:
    """Observability snapshot for ``/admin`` and Phase 3 Prometheus."""
    with _heap_lock:
        items = list(_heap)
    queue_by_priority: dict[str, int] = {p: 0 for p in PRIORITY_RANK}
    oldest_enqueued_at: Optional[float] = None
    for item in items:
        queue_by_priority[item.priority] = queue_by_priority.get(item.priority, 0) + 1
        if oldest_enqueued_at is None or item.enqueued_at < oldest_enqueued_at:
            oldest_enqueued_at = item.enqueued_at
    oldest_age = (
        time.monotonic() - oldest_enqueued_at if oldest_enqueued_at is not None else None
    )
    return {
        "started": _started,
        "queue_depth": len(items),
        "queue_by_priority": queue_by_priority,
        "in_flight_count": len(_in_flight),
        "oldest_age_seconds": oldest_age,
        "aging_step_min": _aging_step_min,
        "reserved_capacity_pct": dict(_reserved_pct),
        "slot_manager": (
            _slot_manager.snapshot() if _slot_manager is not None else {}
        ),
    }


# ─── DB helpers ───────────────────────────────────────────────────────────


def _resolve_user_and_cap(db, publish_task_id: int) -> tuple[Optional[int], int, str]:
    """Resolve (user_id, slot_cap, plan_tier_name) from a PublishTask.

    Returns sensible fallbacks rather than raising; we want the
    dispatch to proceed even if the row is unusual, and the per-job
    error handler will log a useful traceback if something's broken.
    """
    pt = db.query(models.PublishTask).filter(
        models.PublishTask.id == publish_task_id
    ).first()
    if pt is None:
        return None, _DEFAULT_SLOT_CAP, "pro"
    user = db.query(models.User).filter(models.User.id == pt.user_id).first()
    if user is None:
        return pt.user_id, _DEFAULT_SLOT_CAP, "pro"
    plan_tier = None
    if getattr(user, "plan_tier_id", None) is not None:
        plan_tier = db.query(models.PlanTier).filter(
            models.PlanTier.id == user.plan_tier_id
        ).first()
    if plan_tier is not None:
        return (
            int(user.id),
            int(plan_tier.slot_cap_active_uploads or _DEFAULT_SLOT_CAP),
            str(plan_tier.name or "pro"),
        )
    return int(user.id), _DEFAULT_SLOT_CAP, "pro"


def _increment_publish_task_counter(
    db, publish_task_id: Optional[int], *, completed: bool = False, failed: bool = False,
) -> None:
    """Atomic-ish counter bump on the parent PublishTask.

    Phase 1 uses a plain UPDATE; F-agent Phase 2 swaps for the
    SQL-level ``completed_count = completed_count + 1`` row-locked
    variant. We also flip the PublishTask.status to a terminal state
    once ``completed_count + failed_count == target_count``.
    """
    if publish_task_id is None:
        return
    pt = db.query(models.PublishTask).filter(
        models.PublishTask.id == publish_task_id
    ).first()
    if pt is None:
        return
    if completed:
        pt.completed_count = int(pt.completed_count or 0) + 1
    if failed:
        pt.failed_count = int(pt.failed_count or 0) + 1
    done = int(pt.completed_count or 0) + int(pt.failed_count or 0)
    target = int(pt.target_count or 0)
    if target > 0 and done >= target:
        if pt.failed_count == 0:
            pt.status = "completed"
        elif pt.completed_count == 0:
            pt.status = "failed"
        else:
            pt.status = "partial_failed"
    db.add(pt)
    db.commit()


# ─── Job run (Phase 1 skeleton) ───────────────────────────────────────────


async def _run_job(item: _Item) -> None:
    """PHASE 1 SKELETON.

    Walks the new state machine without calling YouTube:

        queued → branding → ready_to_upload → uploading → completed

    Phase 2's Upload agent (E) replaces the body of this function with
    the real Branding Worker + Upload Worker calls. **The surrounding
    contract MUST be preserved**:

        1. Open a DB session with SessionLocal().
        2. Look up UploadJobV2, parent PublishTask, User, PlanTier.
        3. Acquire (user_slot, network_token) via slot_manager.acquire.
        4. Transition statuses + timestamps + commit per step.
        5. On any exception, mark the job 'failed' with last_error and
           bump the PublishTask failed_count.
        6. Always close the DB session in a finally.
    """
    assert _slot_manager is not None, "_run_job called before start()"
    db = SessionLocal()
    job = None
    publish_task_id = None
    try:
        job = db.query(models.UploadJobV2).filter(
            models.UploadJobV2.id == item.upload_job_id
        ).first()
        if job is None:
            log.warning(
                "scheduler._run_job: job_id=%d not found in DB; skipping",
                item.upload_job_id,
            )
            return
        if job.status not in ("queued",):
            # Already past the queued state (e.g. orphan recovery race).
            log.info(
                "scheduler._run_job: job_id=%d status=%r is not 'queued'; skipping",
                item.upload_job_id, job.status,
            )
            return
        publish_task_id = int(job.publish_task_id)
        user_id, cap, plan_tier_name = _resolve_user_and_cap(db, publish_task_id)
        if user_id is None:
            user_id = item.user_id
        # Trust the live DB-resolved cap over the in-flight item value
        # (a plan upgrade between enqueue + dispatch should be honoured).
        item.user_id = user_id
        item.plan_tier_name = plan_tier_name

        # Freeze priority_at_dispatch on the row now — aging may have
        # bumped item.priority but the books use the snapshot here.
        job.priority_at_dispatch = item.priority
        job.dispatched_at = datetime.utcnow()
        db.commit()

        async with _slot_manager.acquire(user_id, cap, "network"):
            if _upload_dispatch is not None:
                # ── Phase 2: real dispatch ──
                # upload_dispatch.process() is synchronous-callable (it
                # opens its own DB session, runs branding → idempotency
                # → quota.reserve → credits.reserve → Direct/RTMP upload
                # → status transitions). Run it in a worker thread so the
                # asyncio loop isn't blocked. It internally transitions
                # the row through branding → ready_to_upload → uploading
                # → completed | failed | parked_quota, so we re-read the
                # row after to learn the outcome and bump counters.
                await asyncio.to_thread(_upload_dispatch.process, item.upload_job_id)
                db.expire_all()
                final = db.query(models.UploadJobV2).filter(
                    models.UploadJobV2.id == item.upload_job_id
                ).first()
                final_status = final.status if final is not None else "failed"
                if final_status == "completed":
                    _increment_publish_task_counter(
                        db, publish_task_id, completed=True,
                    )
                    log.info(
                        "scheduler._run_job: job_id=%d completed via upload_dispatch",
                        item.upload_job_id,
                    )
                elif final_status == "failed":
                    _increment_publish_task_counter(
                        db, publish_task_id, failed=True,
                    )
                    log.info(
                        "scheduler._run_job: job_id=%d failed via upload_dispatch (status=%s)",
                        item.upload_job_id, final_status,
                    )
                elif final_status == "parked_quota":
                    # Quota exhaustion: do not bump completed/failed
                    # counters. Job stays parked; orphan recovery on the
                    # next IST/UTC midnight reset will re-pick it up.
                    log.info(
                        "scheduler._run_job: job_id=%d parked on quota",
                        item.upload_job_id,
                    )
                else:
                    # Defensive: unexpected final state.
                    log.warning(
                        "scheduler._run_job: job_id=%d unexpected final status=%r",
                        item.upload_job_id, final_status,
                    )
            else:
                # ── Phase 1 fallback: synthetic state walk ──
                # Kept for the rare case where upload_dispatch failed to
                # import at boot. Used by the Phase 1 exit test only.
                job.status = "branding"
                db.commit()
                await asyncio.sleep(0.05)
                job.status = "ready_to_upload"
                db.commit()
                await asyncio.sleep(0.05)
                job.status = "uploading"
                job.started_at = datetime.utcnow()
                db.commit()
                await asyncio.sleep(0.1)
                job.status = "completed"
                job.finished_at = datetime.utcnow()
                db.commit()
                _increment_publish_task_counter(
                    db, publish_task_id, completed=True,
                )
                log.info(
                    "scheduler._run_job: job_id=%d completed (Phase 1 skeleton fallback)",
                    item.upload_job_id,
                )

    except asyncio.CancelledError:
        log.warning(
            "scheduler._run_job: job_id=%d cancelled during shutdown",
            item.upload_job_id,
        )
        # Don't try to commit anything — let it stay in whatever
        # intermediate state it reached so orphan recovery can re-pick
        # it up. Re-raise so the task is marked cancelled.
        raise
    except Exception as exc:
        log.exception(
            "scheduler._run_job: job_id=%d failed: %s",
            item.upload_job_id, exc,
        )
        try:
            failed_job = db.query(models.UploadJobV2).filter(
                models.UploadJobV2.id == item.upload_job_id
            ).first()
            if failed_job is not None:
                failed_job.status = "failed"
                failed_job.last_error = str(exc)[:1000]
                failed_job.finished_at = datetime.utcnow()
                db.commit()
                _increment_publish_task_counter(
                    db, publish_task_id, failed=True,
                )
        except Exception:
            log.exception(
                "scheduler._run_job: secondary error while marking job_id=%d failed",
                item.upload_job_id,
            )
    finally:
        try:
            db.close()
        except Exception:
            pass


# ─── Dispatcher loop ──────────────────────────────────────────────────────


async def _dispatcher_loop() -> None:
    """Main loop. Polls the heap for a dispatchable item every tick.

    Cooperatively cancellable: each iteration is short, and we ``await
    asyncio.sleep(_DISPATCH_TICK_SECONDS)`` between work so shutdown's
    cancel takes effect within ~1 tick.
    """
    assert _shutdown_event is not None
    log.info("scheduler: dispatcher loop starting")
    while not _shutdown_event.is_set():
        try:
            _age_jobs()
            idx = _peek_dispatchable_index()
            if idx is None:
                await asyncio.sleep(_DISPATCH_TICK_SECONDS)
                continue
            # Pop the item at idx — heapq doesn't have a "pop at index"
            # so we do it manually and re-heapify.
            with _heap_lock:
                if idx >= len(_heap):
                    await asyncio.sleep(_DISPATCH_TICK_SECONDS)
                    continue
                item = _heap[idx]
                # Standard trick: swap with the last item then pop.
                _heap[idx] = _heap[-1]
                _heap.pop()
                if idx < len(_heap):
                    heapq.heapify(_heap)
                _known_job_ids.discard(item.upload_job_id)
            # Spawn the job task. We track it in _in_flight so shutdown
            # can await it.
            task = asyncio.create_task(_run_job(item))
            _in_flight.add(task)
            task.add_done_callback(_in_flight.discard)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("scheduler: dispatcher loop iteration failed; continuing")
            await asyncio.sleep(_DISPATCH_TICK_SECONDS)
    log.info("scheduler: dispatcher loop exiting (shutdown signalled)")


# ─── Lifecycle ────────────────────────────────────────────────────────────


async def _rehydrate_orphans() -> int:
    """Re-enqueue UploadJobV2 rows left in status='queued' by a prior
    process. Rows in branding/ready_to_upload/uploading from a previous
    crash are NOT re-enqueued in Phase 1 — that's idempotency-aware
    recovery owned by F-agent in Phase 3. We log a warning instead.
    """
    db = SessionLocal()
    rehydrated = 0
    try:
        queued = db.query(models.UploadJobV2).filter(
            models.UploadJobV2.status == "queued"
        ).all()
        for j in queued:
            # Resolve user + plan_tier via PublishTask → User → PlanTier.
            user_id, _cap, tier_name = _resolve_user_and_cap(db, int(j.publish_task_id))
            if user_id is None:
                user_id = 0
            scheduler_enqueue(
                upload_job_id=int(j.id),
                priority=j.priority_at_dispatch or "normal",
                user_id=int(user_id),
                plan_tier_name=tier_name,
            )
            rehydrated += 1

        # Non-terminal orphans from a prior crash (Phase 3 will own
        # the idempotent recovery — for now we just log).
        non_terminal = db.query(models.UploadJobV2).filter(
            models.UploadJobV2.status.in_(("branding", "ready_to_upload", "uploading"))
        ).count()
        if non_terminal:
            log.warning(
                "scheduler: %d UploadJobV2 rows are in non-terminal "
                "in-flight states from a prior process — NOT re-enqueueing "
                "(Phase 3 owns idempotent recovery)",
                non_terminal,
            )
    finally:
        db.close()
    return rehydrated


async def start() -> None:
    """Called from FastAPI startup. Idempotent."""
    global _loop, _slot_manager, _shutdown_event, _dispatcher_task
    global _aging_step_min, _reserved_pct, _started

    if _started:
        log.warning("scheduler.start: already started; ignoring")
        return

    _loop = asyncio.get_running_loop()
    _aging_step_min = _load_aging_step_min()
    _reserved_pct = _load_reserved_pct()

    cpu = cpu_tokens_from_env()
    net = net_tokens_from_env()
    _slot_manager = SlotManager(cpu_tokens=cpu, net_tokens=net)

    log.info(
        "scheduler.start: cpu_tokens=%d net_tokens=%d aging_step_min=%d "
        "reserved_pct=%s",
        cpu, net, _aging_step_min, _reserved_pct,
    )

    # Orphan recovery BEFORE launching the dispatcher so the rehydrated
    # items are in the queue when the loop's first tick runs.
    try:
        n = await _rehydrate_orphans()
        log.info("scheduler: rehydrated %d queued jobs", n)
    except Exception:
        log.exception("scheduler: orphan recovery failed; dispatcher will still start")

    _shutdown_event = asyncio.Event()
    _dispatcher_task = asyncio.create_task(_dispatcher_loop())
    _started = True
    log.info("scheduler.start: dispatcher task launched")


async def shutdown() -> None:
    """Called from FastAPI shutdown. Cancels the dispatcher loop and
    waits up to ``_SHUTDOWN_GRACE_SECONDS`` for in-flight jobs to
    finish.
    """
    global _started
    if not _started:
        return
    log.info(
        "scheduler.shutdown: signalling; in_flight=%d queue_depth=%d",
        len(_in_flight), len(_heap),
    )
    if _shutdown_event is not None:
        _shutdown_event.set()
    if _dispatcher_task is not None:
        try:
            await asyncio.wait_for(_dispatcher_task, timeout=2.0)
        except asyncio.TimeoutError:
            log.warning("scheduler.shutdown: dispatcher loop didn't exit in 2s; cancelling")
            _dispatcher_task.cancel()
            try:
                await _dispatcher_task
            except (asyncio.CancelledError, Exception):
                pass
    # Wait for in-flight job tasks to drain.
    if _in_flight:
        log.info(
            "scheduler.shutdown: waiting up to %.1fs for %d in-flight jobs",
            _SHUTDOWN_GRACE_SECONDS, len(_in_flight),
        )
        try:
            await asyncio.wait_for(
                asyncio.gather(*list(_in_flight), return_exceptions=True),
                timeout=_SHUTDOWN_GRACE_SECONDS,
            )
        except asyncio.TimeoutError:
            log.warning(
                "scheduler.shutdown: %d job task(s) did not finish in %.1fs; cancelling",
                len(_in_flight), _SHUTDOWN_GRACE_SECONDS,
            )
            for t in list(_in_flight):
                t.cancel()
    _started = False
    log.info("scheduler.shutdown: done")


__all__ = [
    "scheduler_enqueue",
    "start",
    "shutdown",
    "snapshot",
    "PRIORITY_RANK",
]
