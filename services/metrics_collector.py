"""Prometheus metrics collector — Phase 3.G (Observability agent).

This module is the **central metric registration** for the new
publish/upload pipeline. It defines the 11 named metrics required by
CONTRACTS §4 G-agent (10 from the brief table + ``recovery_orphan_total``
which the PHASE3 plan also requires) and exposes a single
``collect_snapshot(db)`` function that the ``/metrics`` router calls at
scrape time to refresh gauge values and counter deltas.

DESIGN — pull, not push
-----------------------
The Fanout / Scheduler / Branding / Upload / Burn services already exist
and we must NOT modify them (orchestrator constraint). So instead of
emitting metric increments from inside those services, we compute
everything at scrape time:

  * **Gauges** (``kaizer_slot_active``, ``kaizer_queue_depth``,
    ``kaizer_credit_balance``) are read from ``scheduler.snapshot()`` /
    DB and ``.set()`` directly.
  * **Counters** (``kaizer_publish_task_total``, ``kaizer_upload_job_total``,
    ``kaizer_quota_burn_*_total``, ``kaizer_upload_throughput_bytes``,
    ``kaizer_recovery_orphan_total``) must be monotonic, so we track the
    last-seen DB total in module state + ``.inc()`` by the positive
    delta on each scrape.
  * **Histograms**: the credit-balance histogram is observed once per
    user per scrape (cheap with a TTL cache); the critical-queue-latency
    histogram observes new (dispatched_at - created_at) measurements for
    jobs that crossed from queued → dispatched since the last scrape.

CACHING
-------
A per-user balance sweep over ``credit_ledger`` (last balance_after per
user) is expensive on large tables. We cache the resulting list for 30s
so a Prometheus scrape storm doesn't hammer the DB. The cache TTL is
intentionally less than the typical 15-60s Prometheus scrape interval,
so the first scrape after the TTL refreshes; intermediate scrapes serve
the cached value.

COUNTER MONOTONICITY
--------------------
Prometheus rejects counter decreases. After a process restart the
``Counter`` resets to 0 and the "last seen" DB total in ``_state`` also
resets to 0 — so the first scrape after restart re-inc()s from 0 up to
the current DB total, which Prometheus interprets as a single huge spike.
That's the correct behaviour (a counter reset is a known Prometheus
concept; ``rate()`` and ``increase()`` handle it).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from sqlalchemy import func
from sqlalchemy.orm import Session

import models

log = logging.getLogger("kaizer.metrics")


# ─── Registry (separate from default so we control what's exposed) ─────────

registry = CollectorRegistry()


# ─── Enumerations (match the contract §3.x CHECK constraints) ──────────────

_PRIORITY_LEVELS = ("critical", "high", "normal", "low")
_PUBLISH_TASK_STATUSES = (
    "queued", "fanning_out", "dispatched", "completed",
    "partial_failed", "failed", "cancelled",
)
_UPLOAD_JOB_STATUSES = (
    "queued", "branding", "ready_to_upload", "uploading", "uploaded",
    "completed", "failed", "cancelled", "parked_quota",
)
_UPLOAD_PATHS = ("direct", "rtmp")
_PUBLISH_KINDS = ("video", "short")
_PLAN_TIERS = ("free", "pro", "enterprise")
_BURN_OUTCOMES = ("success", "transient_error", "quota_exceeded", "permanent_error")


# ─── 11 named metrics (CONTRACTS §4 G-agent table) ─────────────────────────

publish_task_total = Counter(
    "kaizer_publish_task_total",
    "Total PublishTasks observed in a given terminal status (cumulative).",
    ["status"],
    registry=registry,
)

upload_job_total = Counter(
    "kaizer_upload_job_total",
    "Total UploadJobV2 rows observed by status/path/publish_kind/plan_tier.",
    ["status", "path", "publish_kind", "plan_tier"],
    registry=registry,
)

slot_active = Gauge(
    "kaizer_slot_active",
    "Active scheduler slots by token kind and plan tier.",
    ["token_kind", "plan_tier"],
    registry=registry,
)

queue_depth = Gauge(
    "kaizer_queue_depth",
    "Scheduler queue depth by priority.",
    ["priority"],
    registry=registry,
)

credit_balance = Histogram(
    "kaizer_credit_balance",
    "Per-user credit balance distribution (observed once per user per scrape).",
    ["plan_tier"],
    buckets=(0, 50, 100, 250, 500, 1000, 2000, 5000, 10000, 20000),
    registry=registry,
)

quota_burn_predicted_total = Counter(
    "kaizer_quota_burn_predicted_total",
    "Predicted quota units burned per operation (sum of QuotaBurnLog.predicted_cost).",
    ["operation"],
    registry=registry,
)

quota_burn_actual_total = Counter(
    "kaizer_quota_burn_actual_total",
    "Reconciled actual quota units burned per QuotaBurnLog.reconciled_actual_cost.",
    ["operation"],
    registry=registry,
)

quota_burn_delta = Counter(
    "kaizer_quota_burn_delta",
    "Cumulative |actual - predicted| delta per operation (always >= 0).",
    ["operation"],
    registry=registry,
)

upload_throughput_bytes = Counter(
    "kaizer_upload_throughput_bytes",
    "Total bytes uploaded successfully per path.",
    ["path"],
    registry=registry,
)

critical_queue_latency_seconds = Histogram(
    "kaizer_critical_queue_latency_seconds",
    "Time a critical-priority job waited in the queue before dispatch.",
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 300),
    registry=registry,
)

recovery_orphan_total = Counter(
    "kaizer_recovery_orphan_total",
    "Orphan publish_attempts recovered (outcome=recovered|completed_late|abandoned).",
    ["outcome"],
    registry=registry,
)


# ─── Wave 2 — durable DB queue (KAIZER_DURABLE_QUEUE=1) ────────────────────
# These mirror kaizer_queue_depth (in-memory scheduler heap) for the Wave 1
# path where the upload_jobs_v2 table itself IS the queue. Sourced from
# services.job_queue.queue_depth_snapshot() at scrape time.

queue_depth_db = Gauge(
    "kaizer_queue_depth_db",
    "Durable (Postgres-backed) queue depth by priority — upload_jobs_v2 rows in status='queued'.",
    ["priority"],
    registry=registry,
)

parked_quota_jobs = Gauge(
    "kaizer_parked_quota_jobs",
    "UploadJobV2 rows parked for YouTube quota (status='parked_quota').",
    registry=registry,
)

jobs_retrying = Gauge(
    "kaizer_jobs_retrying",
    "UploadJobV2 rows queued with next_attempt_at in the future (retry/backoff pressure).",
    registry=registry,
)


# ─── Last-seen DB totals for counter delta math ────────────────────────────
#
# Counters are monotonic. To map a re-queryable DB total into Counter.inc()
# we remember what we've already accounted for and only inc() by the
# positive delta. On process restart everything resets to 0 — Prometheus
# treats the next scrape's burst as a counter reset event and rate()
# handles it correctly.

_state_lock = threading.Lock()
_state: dict = {
    # publish_task: key=status -> count we've already inc()'d
    "publish_task": {s: 0 for s in _PUBLISH_TASK_STATUSES},
    # upload_job: key=(status,path,publish_kind,plan_tier) -> count
    "upload_job": {},
    # quota_burn: key=operation -> {predicted, actual, delta}
    "quota_burn": {},
    # upload_throughput: key=path -> bytes seen
    "upload_throughput": {p: 0 for p in _UPLOAD_PATHS},
    # recovery_orphan: key=outcome -> count
    "recovery_orphan": {"recovered": 0, "completed_late": 0, "abandoned": 0},
    # critical latency observation cursor: only observe jobs dispatched
    # AFTER this timestamp; updated at the end of each scrape.
    "critical_latency_cursor": None,  # set to now() on first scrape
    # credit-balance TTL cache: (expires_at_monotonic, [(plan_tier, balance), ...])
    "credit_balance_cache_until": 0.0,
    "credit_balance_cache_value": None,
}


# Credit-balance cache TTL — short enough to be fresh, long enough that
# a 15s scrape doesn't hammer the DB. 30s is a safe middle.
_CREDIT_BALANCE_TTL_SECONDS = 30.0


# ─── Snapshot collection ───────────────────────────────────────────────────


def _collect_publish_task_counts(db: Session) -> None:
    """SELECT status, COUNT(*) FROM publish_tasks GROUP BY status — for
    each status, .inc() the publish_task_total counter by the positive
    delta since last scrape."""
    try:
        rows = (
            db.query(models.PublishTask.status, func.count(models.PublishTask.id))
            .group_by(models.PublishTask.status)
            .all()
        )
        seen: dict[str, int] = {s: 0 for s in _PUBLISH_TASK_STATUSES}
        for status, count in rows:
            s = str(status or "").strip()
            if s in seen:
                seen[s] = int(count or 0)
        with _state_lock:
            for s, current in seen.items():
                last = int(_state["publish_task"].get(s, 0))
                delta = current - last
                if delta > 0:
                    publish_task_total.labels(status=s).inc(delta)
                # Only ratchet up — never down. If a status row is deleted
                # the counter just stops increasing for that label, which
                # matches Prometheus' "counter never decreases" semantics.
                if current > last:
                    _state["publish_task"][s] = current
    except Exception as exc:
        log.warning("metrics: _collect_publish_task_counts failed: %s", exc)


def _collect_upload_job_counts(db: Session) -> None:
    """SELECT status,upload_path,publish_kind, COUNT(*) FROM upload_jobs_v2
    GROUP BY ... — joined to publish_tasks → users → plan_tiers so we
    can label by plan_tier."""
    try:
        rows = (
            db.query(
                models.UploadJobV2.status,
                models.UploadJobV2.upload_path,
                models.UploadJobV2.publish_kind,
                models.PlanTier.name,
                func.count(models.UploadJobV2.id),
            )
            .join(models.PublishTask, models.UploadJobV2.publish_task_id == models.PublishTask.id)
            .join(models.User, models.PublishTask.user_id == models.User.id)
            .outerjoin(models.PlanTier, models.User.plan_tier_id == models.PlanTier.id)
            .group_by(
                models.UploadJobV2.status,
                models.UploadJobV2.upload_path,
                models.UploadJobV2.publish_kind,
                models.PlanTier.name,
            )
            .all()
        )
        with _state_lock:
            for status, path, kind, plan_tier, count in rows:
                key = (
                    str(status or ""),
                    str(path or ""),
                    str(kind or ""),
                    str(plan_tier or "unknown"),
                )
                last = int(_state["upload_job"].get(key, 0))
                current = int(count or 0)
                delta = current - last
                if delta > 0:
                    upload_job_total.labels(
                        status=key[0], path=key[1],
                        publish_kind=key[2], plan_tier=key[3],
                    ).inc(delta)
                if current > last:
                    _state["upload_job"][key] = current
    except Exception as exc:
        log.warning("metrics: _collect_upload_job_counts failed: %s", exc)


def _collect_quota_burn(db: Session) -> None:
    """Aggregate QuotaBurnLog by operation; inc predicted/actual/delta
    counters by positive deltas."""
    try:
        rows = (
            db.query(
                models.QuotaBurnLog.operation,
                func.coalesce(func.sum(models.QuotaBurnLog.predicted_cost), 0),
                func.coalesce(func.sum(models.QuotaBurnLog.reconciled_actual_cost), 0),
            )
            .group_by(models.QuotaBurnLog.operation)
            .all()
        )
        with _state_lock:
            for operation, predicted_total, actual_total in rows:
                op = str(operation or "unknown")
                last = _state["quota_burn"].get(
                    op, {"predicted": 0, "actual": 0, "delta": 0},
                )
                p_cur = int(predicted_total or 0)
                a_cur = int(actual_total or 0)
                # Absolute delta is cumulative |actual - predicted|. We
                # represent it as a Counter (always increases), so on each
                # scrape we ratchet up by the positive change in |delta|.
                d_cur = abs(a_cur - p_cur)
                if p_cur > last["predicted"]:
                    quota_burn_predicted_total.labels(operation=op).inc(
                        p_cur - last["predicted"]
                    )
                if a_cur > last["actual"]:
                    quota_burn_actual_total.labels(operation=op).inc(
                        a_cur - last["actual"]
                    )
                if d_cur > last["delta"]:
                    quota_burn_delta.labels(operation=op).inc(
                        d_cur - last["delta"]
                    )
                _state["quota_burn"][op] = {
                    "predicted": max(p_cur, last["predicted"]),
                    "actual": max(a_cur, last["actual"]),
                    "delta": max(d_cur, last["delta"]),
                }
    except Exception as exc:
        log.warning("metrics: _collect_quota_burn failed: %s", exc)


def _collect_upload_throughput(db: Session) -> None:
    """SUM(bytes_uploaded) WHERE status='completed' GROUP BY upload_path."""
    try:
        rows = (
            db.query(
                models.UploadJobV2.upload_path,
                func.coalesce(func.sum(models.UploadJobV2.bytes_uploaded), 0),
            )
            .filter(models.UploadJobV2.status == "completed")
            .group_by(models.UploadJobV2.upload_path)
            .all()
        )
        with _state_lock:
            for path, total_bytes in rows:
                p = str(path or "")
                if p not in _UPLOAD_PATHS:
                    continue
                last = int(_state["upload_throughput"].get(p, 0))
                cur = int(total_bytes or 0)
                if cur > last:
                    upload_throughput_bytes.labels(path=p).inc(cur - last)
                    _state["upload_throughput"][p] = cur
    except Exception as exc:
        log.warning("metrics: _collect_upload_throughput failed: %s", exc)


def _collect_recovery_orphans(db: Session) -> None:
    """Total publish_attempts rows in 'recovered' state. We bucket as a
    single outcome=recovered; the other outcomes (completed_late,
    abandoned) are reserved for the F-agent's future recovery taxonomy.
    """
    try:
        n_recovered = (
            db.query(func.count(models.PublishAttempt.id))
            .filter(models.PublishAttempt.status == "recovered")
            .scalar()
        ) or 0
        with _state_lock:
            last = int(_state["recovery_orphan"].get("recovered", 0))
            cur = int(n_recovered)
            if cur > last:
                recovery_orphan_total.labels(outcome="recovered").inc(cur - last)
                _state["recovery_orphan"]["recovered"] = cur
    except Exception as exc:
        log.warning("metrics: _collect_recovery_orphans failed: %s", exc)


def _collect_queue_depth() -> None:
    """Pull from scheduler.snapshot() — already in-process, no DB hit."""
    try:
        from services import scheduler as _scheduler
        snap = _scheduler.snapshot()
        per_prio = snap.get("queue_by_priority") or {}
        for p in _PRIORITY_LEVELS:
            queue_depth.labels(priority=p).set(int(per_prio.get(p, 0)))
    except Exception as exc:
        log.warning("metrics: _collect_queue_depth failed: %s", exc)
        for p in _PRIORITY_LEVELS:
            queue_depth.labels(priority=p).set(0)


def _collect_slot_active(db: Session) -> None:
    """Slot manager's `user_in_use` is keyed by user_id. We attribute each
    user's network-token usage to their plan tier so a Grafana panel can
    show 'enterprise users are hogging slots' at a glance.
    CPU tokens are global (not per-user); we record them under plan_tier='all'."""
    try:
        from services import scheduler as _scheduler
        snap = _scheduler.snapshot()
        slot = snap.get("slot_manager") or {}
        cpu_in_use = int(slot.get("cpu_in_use", 0))
        user_in_use = slot.get("user_in_use") or {}

        # Reset all per-tier gauges first; absent labels are interpreted
        # as 0 by Grafana on next-evaluation but explicit .set(0) is safer.
        for tier in _PLAN_TIERS:
            slot_active.labels(token_kind="network", plan_tier=tier).set(0)
        slot_active.labels(token_kind="cpu", plan_tier="all").set(cpu_in_use)

        if user_in_use:
            user_ids = [int(uid) for uid in user_in_use.keys()]
            # Single query: SELECT user.id, plan_tier.name FROM users
            #               LEFT JOIN plan_tiers
            #               WHERE user.id IN (...)
            rows = (
                db.query(models.User.id, models.PlanTier.name)
                .outerjoin(models.PlanTier, models.User.plan_tier_id == models.PlanTier.id)
                .filter(models.User.id.in_(user_ids))
                .all()
            )
            tier_by_user: dict[int, str] = {
                int(uid): (str(name) if name else "unknown") for uid, name in rows
            }
            tally: dict[str, int] = {t: 0 for t in _PLAN_TIERS}
            for uid_str, n in user_in_use.items():
                try:
                    uid = int(uid_str)
                except (TypeError, ValueError):
                    continue
                tier = tier_by_user.get(uid, "pro")  # default to pro if unknown
                if tier not in tally:
                    tally[tier] = 0
                tally[tier] += int(n or 0)
            for tier, n in tally.items():
                slot_active.labels(token_kind="network", plan_tier=tier).set(int(n))
    except Exception as exc:
        log.warning("metrics: _collect_slot_active failed: %s", exc)


def _collect_credit_balance(db: Session) -> None:
    """Observe each user's current credit balance into the histogram,
    labelled by plan_tier. TTL-cached: the per-user sweep is the most
    expensive query in the collector and a 30s cache is plenty for a
    15-60s Prometheus scrape interval."""
    try:
        now_mono = time.monotonic()
        with _state_lock:
            if (
                _state["credit_balance_cache_value"] is not None
                and now_mono < _state["credit_balance_cache_until"]
            ):
                cached = list(_state["credit_balance_cache_value"])
                # Observe the cached values into the histogram (the histogram
                # itself doesn't have a snapshot/reset; we just keep adding,
                # which is fine because histograms are cumulative).
                for plan_tier, balance in cached:
                    credit_balance.labels(plan_tier=plan_tier).observe(float(balance))
                return

        # Cache miss — compute fresh. "Current balance per user" is the
        # last credit_ledger row per user (denormalised `balance_after`).
        # We do a window-function-free portable query: SELECT user_id,
        # MAX(id) → join back.
        from sqlalchemy import select
        latest_ids_subq = (
            db.query(func.max(models.CreditLedger.id).label("max_id"))
            .group_by(models.CreditLedger.user_id)
            .subquery()
        )
        rows = (
            db.query(
                models.CreditLedger.user_id,
                models.CreditLedger.balance_after,
                models.PlanTier.name,
            )
            .join(latest_ids_subq, models.CreditLedger.id == latest_ids_subq.c.max_id)
            .join(models.User, models.CreditLedger.user_id == models.User.id)
            .outerjoin(models.PlanTier, models.User.plan_tier_id == models.PlanTier.id)
            .all()
        )
        observations: list[tuple[str, int]] = []
        for _user_id, balance_after, plan_name in rows:
            tier = str(plan_name) if plan_name else "pro"
            observations.append((tier, int(balance_after or 0)))

        for plan_tier, balance in observations:
            credit_balance.labels(plan_tier=plan_tier).observe(float(balance))

        with _state_lock:
            _state["credit_balance_cache_value"] = observations
            _state["credit_balance_cache_until"] = now_mono + _CREDIT_BALANCE_TTL_SECONDS
    except Exception as exc:
        log.warning("metrics: _collect_credit_balance failed: %s", exc)


def _collect_critical_queue_latency(db: Session) -> None:
    """Observe (dispatched_at - created_at) for upload_jobs_v2 that crossed
    from queued → dispatched at priority_at_dispatch='critical' since
    the last scrape. We don't worry about exact-once observation —
    Prometheus tolerates eventual consistency on histograms.

    NOTE: ``priority_at_dispatch`` is set by the scheduler at the moment
    the job is released (CONTRACTS §3.3) so this captures real queue
    latency at the dispatch decision.
    """
    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        with _state_lock:
            cursor = _state.get("critical_latency_cursor")
            if cursor is None:
                # First scrape: don't backfill historical data. Just set
                # the cursor and bail.
                _state["critical_latency_cursor"] = now
                return

        rows = (
            db.query(
                models.UploadJobV2.created_at,
                models.UploadJobV2.dispatched_at,
            )
            .filter(models.UploadJobV2.priority_at_dispatch == "critical")
            .filter(models.UploadJobV2.dispatched_at.isnot(None))
            .filter(models.UploadJobV2.dispatched_at > cursor)
            .filter(models.UploadJobV2.dispatched_at <= now)
            .all()
        )
        for created_at, dispatched_at in rows:
            if created_at is None or dispatched_at is None:
                continue
            try:
                latency = (dispatched_at - created_at).total_seconds()
            except Exception:
                continue
            if latency >= 0:
                critical_queue_latency_seconds.observe(latency)
        with _state_lock:
            _state["critical_latency_cursor"] = now
    except Exception as exc:
        log.warning("metrics: _collect_critical_queue_latency failed: %s", exc)


def _collect_durable_queue(db: Session) -> None:
    """Wave 2: durable-queue gauges from services.job_queue (Wave 1).

    Lazy import — tolerate the module being absent on deployments that
    predate the durable queue (gauges simply stay at their last value).
    """
    try:
        from services.job_queue import queue_depth_snapshot
    except ImportError:
        return
    try:
        snap = queue_depth_snapshot()
        per_prio = snap.get("queued_by_priority") or {}
        for p in _PRIORITY_LEVELS:
            queue_depth_db.labels(priority=p).set(int(per_prio.get(p, 0)))
        parked_quota_jobs.set(int(snap.get("parked_quota", 0)))
    except Exception as exc:
        log.warning("metrics: _collect_durable_queue snapshot failed: %s", exc)

    # Retry pressure: queued rows still waiting out a backoff window
    # (next_attempt_at in the future — claimable later, not now).
    # CURRENT_TIMESTAMP is portable across Postgres + SQLite.
    try:
        from sqlalchemy import text as _text
        n = db.execute(_text(
            "SELECT count(*) FROM upload_jobs_v2 "
            "WHERE status = 'queued' AND next_attempt_at > CURRENT_TIMESTAMP"
        )).scalar() or 0
        jobs_retrying.set(int(n))
    except Exception as exc:
        log.warning("metrics: _collect_durable_queue retry count failed: %s", exc)


def collect_snapshot(db: Session) -> None:
    """Refresh all gauge/counter values from the current DB +
    scheduler/branding snapshots. Called at the top of GET /metrics so
    the scrape always sees fresh data."""
    _collect_queue_depth()
    _collect_durable_queue(db)
    _collect_slot_active(db)
    _collect_publish_task_counts(db)
    _collect_upload_job_counts(db)
    _collect_quota_burn(db)
    _collect_upload_throughput(db)
    _collect_recovery_orphans(db)
    _collect_credit_balance(db)
    _collect_critical_queue_latency(db)


def render_latest() -> bytes:
    """Return the Prometheus text-exposition payload for the registry.
    Thin wrapper so the router doesn't import prometheus_client directly."""
    return generate_latest(registry)


__all__ = [
    "registry",
    "collect_snapshot",
    "render_latest",
    "CONTENT_TYPE_LATEST",
    # Metric handles (exported for tests):
    "publish_task_total",
    "upload_job_total",
    "slot_active",
    "queue_depth",
    "credit_balance",
    "quota_burn_predicted_total",
    "quota_burn_actual_total",
    "quota_burn_delta",
    "upload_throughput_bytes",
    "critical_queue_latency_seconds",
    "recovery_orphan_total",
    "queue_depth_db",
    "parked_quota_jobs",
    "jobs_retrying",
]
