"""Predicted-vs-actual quota burn ledger — Phase 2.F.

The brief (§2, §6, §7) is firm: never treat the published per-operation
unit cost (e.g. ``videos.insert = 1600u``) as truth. We predict at the
``reserve()`` gate; we then RECORD what actually happened — success vs
quota_exceeded vs other — into ``quota_burn_log``. Later, a
reconciliation pass against Google's reported usage fills
``reconciled_actual_cost`` and surfaces the predicted-vs-actual delta
on the admin dashboard.

This module is the F-agent's append-side of that ledger. It is
ORTHOGONAL to the existing ``learning/youtube_quota_log.py`` forensic
log (which writes ``youtube_api_calls`` rows; preserved per brief §0).
That forensic log captures every API call ever made; THIS ledger
captures the predicted-vs-actual *delta* the admin dashboard graphs.

Public surface (CONTRACTS.md §4.5):
  * ``log_predicted_and_actual`` — one INSERT per API call
  * ``reconcile_window``         — fills reconciled_actual_cost in [start, end]
  * ``snapshot``                 — last-N-minutes observability rollup

The reconciliation is a TODO-style STUB for Phase 2: we fill
``reconciled_actual_cost = predicted_cost`` for successful calls and
leave failed/quota-exceeded as NULL. The PRODUCTION reconciliation
that consumes Google's Quota Status API (or Cloud Console export)
lands in a future milestone — the schema + plumbing are ready for it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

import models


log = logging.getLogger("kaizer.burn_log")


_VALID_OUTCOMES = {
    "success", "transient_error", "quota_exceeded", "permanent_error",
}


# ─── Dataclasses ────────────────────────────────────────────────────────


@dataclass
class ReconciliationReport:
    """Returned by ``reconcile_window``. Surfaces totals + delta so the
    admin dashboard graph can plot one point per reconciliation pass.

    ``delta`` is ``total_actual - total_predicted``:
      * positive: we burned MORE than predicted (under-charged the gate)
      * negative: we burned LESS than predicted (over-charged the gate)
      * 0:        in the Phase 2 stub, success-only rows always reconcile
                  to predicted; failed rows stay NULL and don't sum in
    """
    window_start: datetime
    window_end: datetime
    rows_seen: int
    rows_reconciled: int
    total_predicted: int
    total_actual: int
    delta: int


# ─── Helpers ────────────────────────────────────────────────────────────


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_utc(when: Optional[datetime]) -> datetime:
    if when is None:
        return _now_utc()
    if when.tzinfo is None:
        return when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc)


# ─── Public API ─────────────────────────────────────────────────────────


def log_predicted_and_actual(
    db: Session,
    *,
    upload_job_id: Optional[int],
    operation: str,
    predicted_cost: int,
    http_status: Optional[int] = None,
    observed_outcome: str = "success",
) -> models.QuotaBurnLog:
    """One INSERT into ``quota_burn_log``. Cheap; safe to call after
    every wrapped ``log_youtube_call`` exits.

    ``observed_outcome`` must be one of:
      ``success`` | ``transient_error`` | ``quota_exceeded`` | ``permanent_error``

    ``was_quota_exceeded`` is derived from ``observed_outcome`` so the
    admin dashboard's "quota_exceeded today" panel is one indexed-WHERE
    away from accurate.
    """
    if observed_outcome not in _VALID_OUTCOMES:
        raise ValueError(
            f"observed_outcome {observed_outcome!r} not in {sorted(_VALID_OUTCOMES)}"
        )

    row = models.QuotaBurnLog(
        upload_job_id=upload_job_id,
        operation=str(operation)[:40],
        predicted_cost=int(predicted_cost or 0),
        observed_outcome=observed_outcome,
        http_status=(int(http_status) if http_status is not None else None),
        was_quota_exceeded=(observed_outcome == "quota_exceeded"),
        reconciled_actual_cost=None,
        reconciled_at=None,
    )
    db.add(row)
    try:
        db.flush()
    except Exception as exc:
        # Defensive: a burn_log INSERT failure must NOT cascade-kill the
        # upload pipeline. Log + swallow.
        log.warning(
            "burn_log.log_predicted_and_actual: INSERT failed (%s); "
            "operation=%s predicted_cost=%s",
            exc, operation, predicted_cost,
        )
        db.rollback()
        return row
    return row


def reconcile_window(
    db: Session,
    *,
    start: datetime,
    end: datetime,
) -> ReconciliationReport:
    """Reconciliation pass for the [start, end) window.

    STUB STATUS (Phase 2)
    ---------------------
    We don't have Google's Quota Status API wired up yet. This stub
    implements the SCHEMA-correct flow so the cron job is real:

      * Read all QuotaBurnLog rows in [start, end) WHERE reconciled_at IS NULL.
      * For each, set ``reconciled_actual_cost = predicted_cost`` IFF
        ``observed_outcome == 'success'``. Otherwise leave it NULL
        (failed/quota_exceeded calls didn't burn the full predicted amount).
      * Stamp ``reconciled_at = now()``.
      * Return aggregate totals + the implied delta (always 0 in this
        stub; the production reconciliation will surface real drift).

    PRODUCTION TODO
    ---------------
    Replace the per-row "predicted = actual" assumption with the
    Google-reported value. Two known data sources:

      1. Google Cloud Console quota dashboard export (CSV / API).
      2. The YouTube API's reported usage endpoint (if Google ships one).

    The schema + return type don't change when this lands — only the
    inner ``actual = ...`` line.
    """
    start = _ensure_utc(start)
    end = _ensure_utc(end)

    rows = (
        db.query(models.QuotaBurnLog)
        .filter(
            models.QuotaBurnLog.created_at >= start,
            models.QuotaBurnLog.created_at < end,
        )
        .all()
    )

    rows_seen = len(rows)
    rows_reconciled = 0
    total_predicted = 0
    total_actual = 0

    now = _now_utc()
    for r in rows:
        total_predicted += int(r.predicted_cost or 0)

        if r.reconciled_at is None:
            if r.observed_outcome == "success":
                r.reconciled_actual_cost = int(r.predicted_cost or 0)
                total_actual += int(r.reconciled_actual_cost or 0)
                rows_reconciled += 1
            else:
                # Leave reconciled_actual_cost = NULL so the dashboard
                # shows "unknown actual"; this is correct given we don't
                # have Google's feed yet for failed calls.
                rows_reconciled += 1
            r.reconciled_at = now
            db.add(r)
        else:
            # Already reconciled — just include in totals.
            total_actual += int(r.reconciled_actual_cost or 0)

    try:
        db.flush()
    except Exception as exc:
        log.warning("burn_log.reconcile_window: flush failed (%s)", exc)
        db.rollback()

    delta = total_actual - total_predicted
    return ReconciliationReport(
        window_start=start,
        window_end=end,
        rows_seen=rows_seen,
        rows_reconciled=rows_reconciled,
        total_predicted=total_predicted,
        total_actual=total_actual,
        delta=delta,
    )


def snapshot(db: Session, *, window_minutes: int = 60) -> dict:
    """Observability rollup: predicted-vs-actual totals over the last
    ``window_minutes``. Cheap aggregation suitable for a Prometheus
    Gauge backfill loop.

    Schema:
      {
        "window_minutes": 60,
        "rows_total":     <n>,
        "rows_reconciled":<n>,
        "predicted":      <units>,
        "actual":         <units>,
        "delta":          <actual - predicted>,
        "by_outcome":     {"success": n, "quota_exceeded": n, ...},
      }
    """
    now = _now_utc()
    start = now - timedelta(minutes=int(window_minutes))

    rows = (
        db.query(models.QuotaBurnLog)
        .filter(models.QuotaBurnLog.created_at >= start)
        .all()
    )

    rows_total = len(rows)
    rows_reconciled = sum(1 for r in rows if r.reconciled_at is not None)
    predicted = sum(int(r.predicted_cost or 0) for r in rows)
    actual = sum(int(r.reconciled_actual_cost or 0) for r in rows if r.reconciled_at is not None)
    by_outcome: dict[str, int] = {}
    for r in rows:
        by_outcome[r.observed_outcome] = by_outcome.get(r.observed_outcome, 0) + 1

    return {
        "window_minutes": int(window_minutes),
        "rows_total": rows_total,
        "rows_reconciled": rows_reconciled,
        "predicted": predicted,
        "actual": actual,
        "delta": actual - predicted,
        "by_outcome": by_outcome,
    }


__all__ = [
    "ReconciliationReport",
    "log_predicted_and_actual",
    "reconcile_window",
    "snapshot",
]
