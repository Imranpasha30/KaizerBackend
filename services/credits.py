"""Credit ledger — Phase 2.F full implementation (replaces Phase 1 stub).

Owns the append-only signed ledger (table ``credit_ledger``) plus the
five public entry points the rest of the system calls:

  * ``reserve``        – deduct N credits, row-locked race-safe
  * ``refund``         – credit N credits back, dedupe-by-upload_job_id
  * ``get_balance``    – read the user's current balance
  * ``allot_monthly``  – idempotent monthly allotment seeding
  * ``check_path_allowed`` – Decision 6 plan-tier gate

The Phase 1.B stub already shipped most of this surface; this file is
the F-agent's Phase 2 superset:

  * concurrent ``reserve`` is now race-safe via Postgres advisory locks
    (SQLite degrades to the previous in-process behaviour)
  * ``refund`` is dedupe-by-upload_job_id (the caller no longer has to
    pre-check)
  * monthly allotment is idempotent per (user_id, calendar-month UTC)
  * Decision 6 plan-tier helper lives here so Fanout + upload_dispatch
    can share one check

Public surface kept backward-compatible:
  * ``ReserveResult`` keeps the four fields the Phase 1 callers read:
    ``previous_balance``, ``cost``, ``new_balance``, ``ledger_id``.
    A new ``balance_after`` property aliases ``new_balance`` so callers
    written to CONTRACTS.md §4.5 (which spells it that way) also work.
  * ``InsufficientCreditsError``, ``CreditLedgerArgumentError`` unchanged.
  * ``reserve``, ``refund``, ``get_balance`` keyword signatures unchanged.

CONTRACTS.md §3.5 invariant (re-enforced in code, defense in depth):
when ``reason in ('upload_direct', 'upload_rtmp')`` both ``path`` and
``publish_kind`` MUST be populated; for all other reasons they MUST be
NULL.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

import models


log = logging.getLogger("kaizer.credits")


# ─── Exceptions ───────────────────────────────────────────────────────────


class InsufficientCreditsError(Exception):
    """Raised by ``reserve()`` when the user's balance < cost.

    The router maps this to HTTP 402 (Payment Required) with
    ``code='insufficient_credits'`` and exposes ``balance`` + ``needed``
    in the response body.
    """

    def __init__(self, user_id: int, balance: int, needed: int) -> None:
        self.user_id = user_id
        self.balance = balance
        self.needed = needed
        super().__init__(f"user {user_id} needs {needed} cr, has {balance}")


class CreditLedgerArgumentError(ValueError):
    """Wrong combination of (reason, path, publish_kind) per CONTRACTS.md §3.5."""


class PlanTierViolationError(Exception):
    """Raised by ``check_path_allowed`` when a user's plan_tier disallows
    the requested upload_path (Decision 6: Free is RTMP-only)."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(detail or code)


# ─── Result dataclass ─────────────────────────────────────────────────────


@dataclass
class ReserveResult:
    """Returned by ``reserve()``. Phase 1 callers read
    ``new_balance``/``ledger_id``; CONTRACTS.md §4.5 spec calls it
    ``balance_after`` — we expose both for compatibility.
    """
    previous_balance: int
    cost: int
    new_balance: int
    ledger_id: int

    @property
    def balance_after(self) -> int:
        return self.new_balance


# ─── Internals ────────────────────────────────────────────────────────────


_UPLOAD_REASONS = {"upload_direct", "upload_rtmp"}
_VALID_REASONS = {
    "monthly_allotment", "upload_direct", "upload_rtmp", "refund", "admin_adjustment",
}


def _validate_reason_combo(
    reason: str,
    path: Optional[str],
    publish_kind: Optional[str],
) -> None:
    if reason not in _VALID_REASONS:
        raise CreditLedgerArgumentError(
            f"reason {reason!r} not in {sorted(_VALID_REASONS)}"
        )
    if reason in _UPLOAD_REASONS:
        if path not in ("direct", "rtmp"):
            raise CreditLedgerArgumentError(
                f"reason={reason!r} requires path in ('direct','rtmp'); got {path!r}"
            )
        if publish_kind not in ("video", "short"):
            raise CreditLedgerArgumentError(
                f"reason={reason!r} requires publish_kind in ('video','short'); got {publish_kind!r}"
            )
    else:
        # CONTRACTS.md §3.5: path/publish_kind STAY NULL for non-upload reasons.
        if path is not None or publish_kind is not None:
            raise CreditLedgerArgumentError(
                f"reason={reason!r} must not carry path/publish_kind"
            )


def _is_postgres(db: Session) -> bool:
    """Detect the active SQLAlchemy dialect so we can branch on locking
    primitives."""
    try:
        return db.bind.dialect.name == "postgresql"  # type: ignore[union-attr]
    except Exception:
        return False


def _advisory_lock(db: Session, user_id: int) -> None:
    """Acquire a Postgres transactional advisory lock keyed by user_id.

    The lock is automatically released at COMMIT/ROLLBACK — no manual
    release needed. Two concurrent ``reserve()`` calls for the same user
    will be serialised at this point; concurrent calls for different
    users do NOT block each other (the lock key is the user_id).

    Falls back to a no-op on SQLite (advisory locks don't exist there).
    The dev DB is Postgres per Schema agent's report, so this is the
    production path.
    """
    if not _is_postgres(db):
        # SQLite dev mode — the in-process GIL plus single-writer model
        # makes contention rare. Document the dialect branch.
        return
    try:
        # pg_advisory_xact_lock is the BLOCKING variant; we WANT to block
        # so the second caller waits for the first to finish. Using the
        # blocking version avoids a retry loop here. The lock is released
        # automatically at commit/rollback.
        db.execute(text("SELECT pg_advisory_xact_lock(:key)"),
                   {"key": int(user_id)})
    except OperationalError as exc:
        # If the connection is in an aborted state, the caller's enclosing
        # transaction will be rolled back anyway — log and continue with
        # the simpler (race-prone but functional) path.
        log.warning(
            "credits._advisory_lock(user_id=%s): pg lock failed (%s); "
            "falling through to non-locked path",
            user_id, exc,
        )


def _current_balance(db: Session, user_id: int) -> int:
    """SELECT COALESCE(SUM(delta), 0) FROM credit_ledger WHERE user_id=…

    We read the SUM rather than the last ``balance_after`` so the
    function is correct even if a prior write was rolled back mid-flight.
    Phase 2's row-locked variant relies on this being canonical truth.
    """
    total = (
        db.query(func.coalesce(func.sum(models.CreditLedger.delta), 0))
        .filter(models.CreditLedger.user_id == user_id)
        .scalar()
    )
    return int(total or 0)


def _month_key_utc(when: Optional[datetime] = None) -> str:
    """``YYYY-MM`` for the calendar month in UTC."""
    when = when or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc).strftime("%Y-%m")


def _month_window_utc(when: Optional[datetime] = None) -> tuple[datetime, datetime]:
    """Return [month_start, next_month_start) as UTC-aware datetimes."""
    when = when or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    month_start = when.astimezone(timezone.utc).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0,
    )
    if month_start.month == 12:
        next_month_start = month_start.replace(year=month_start.year + 1, month=1)
    else:
        next_month_start = month_start.replace(month=month_start.month + 1)
    return month_start, next_month_start


# ─── Public API ───────────────────────────────────────────────────────────


def reserve(
    db: Session,
    *,
    user_id: int,
    cost: int,
    reason: str,
    upload_job_id: Optional[int] = None,
    path: Optional[str] = None,
    publish_kind: Optional[str] = None,
    predicted_quota_units: Optional[int] = None,
) -> ReserveResult:
    """Deduct ``cost`` credits from ``user_id``'s balance, atomically.

    Race-safety: a Postgres transactional advisory lock keyed by
    ``user_id`` serialises concurrent ``reserve()`` callers for the same
    user. The lock is held until the caller's transaction commits or
    rolls back. Different users run fully in parallel.

    Raises ``InsufficientCreditsError`` if the balance is < cost (and
    writes no row).
    """
    if cost <= 0:
        raise CreditLedgerArgumentError(f"cost must be positive; got {cost}")
    _validate_reason_combo(reason, path, publish_kind)

    # Serialise concurrent reservations on this user.
    _advisory_lock(db, user_id)

    previous_balance = _current_balance(db, user_id)
    if previous_balance < cost:
        # Caller's transaction will roll back — we wrote nothing yet.
        raise InsufficientCreditsError(user_id, previous_balance, cost)

    new_balance = previous_balance - cost
    row = models.CreditLedger(
        user_id=user_id,
        delta=-cost,
        reason=reason,
        upload_job_id=upload_job_id,
        path=path,
        publish_kind=publish_kind,
        predicted_quota_units=predicted_quota_units,
        balance_after=new_balance,
    )
    db.add(row)
    db.flush()  # populate row.id without committing — caller controls the tx

    return ReserveResult(
        previous_balance=previous_balance,
        cost=cost,
        new_balance=new_balance,
        ledger_id=int(row.id),
    )


def refund(
    db: Session,
    *,
    user_id: int,
    cost: int,
    upload_job_id: Optional[int],
    reason: str = "refund",
) -> None:
    """Inverse of ``reserve``: write a +cost row.

    Dedupe-by-upload_job_id (Phase 2 contract): if a row with
    ``reason='refund'`` already exists for the same ``upload_job_id``,
    this is a no-op + INFO log. The caller no longer has to pre-check.

    The dedupe ONLY applies when ``upload_job_id`` is non-None. A
    refund tied to no upload job (e.g. an admin-issued goodwill credit
    routed through this path) is always written.
    """
    if cost <= 0:
        raise CreditLedgerArgumentError(f"refund cost must be positive; got {cost}")
    if reason not in _VALID_REASONS:
        raise CreditLedgerArgumentError(
            f"refund reason {reason!r} not in {sorted(_VALID_REASONS)}"
        )

    # Serialise concurrent writes for this user so the balance is consistent.
    _advisory_lock(db, user_id)

    if upload_job_id is not None and reason == "refund":
        existing = (
            db.query(models.CreditLedger)
            .filter(
                models.CreditLedger.user_id == user_id,
                models.CreditLedger.upload_job_id == upload_job_id,
                models.CreditLedger.reason == "refund",
            )
            .first()
        )
        if existing is not None:
            log.info(
                "credits.refund: dedupe — refund for user_id=%s upload_job_id=%s "
                "already exists (ledger_id=%s); no-op",
                user_id, upload_job_id, existing.id,
            )
            return

    previous_balance = _current_balance(db, user_id)
    new_balance = previous_balance + cost
    row = models.CreditLedger(
        user_id=user_id,
        delta=cost,
        reason=reason,
        upload_job_id=upload_job_id,
        path=None,
        publish_kind=None,
        predicted_quota_units=None,
        balance_after=new_balance,
    )
    db.add(row)
    db.flush()


def get_balance(db: Session, user_id: int) -> int:
    """Return the user's current credit balance."""
    return _current_balance(db, user_id)


def allot_monthly(
    db: Session,
    *,
    user_id: int,
    when: Optional[datetime] = None,
) -> Optional[ReserveResult]:
    """Idempotent monthly credit allotment seeding.

    Looks up the user's plan_tier, computes the calendar month in UTC,
    and INSERTs one ``credit_ledger`` row with
    ``reason='monthly_allotment'`` and ``delta=plan_tier.monthly_credit_allotment``
    — UNLESS such a row already exists for this user in the current
    month, in which case this is a no-op + INFO log.

    Safe to call repeatedly from a cron job; the dedupe check guarantees
    at most one allotment per user per UTC calendar month.

    Returns the ReserveResult-shaped credit update on success, or None
    on no-op.
    """
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user is None:
        raise CreditLedgerArgumentError(f"user_id={user_id} not found")

    plan_tier_id = getattr(user, "plan_tier_id", None)
    if plan_tier_id is None:
        raise CreditLedgerArgumentError(
            f"user_id={user_id} has no plan_tier_id; cannot allot"
        )
    plan_tier = db.query(models.PlanTier).filter(
        models.PlanTier.id == plan_tier_id
    ).first()
    if plan_tier is None:
        raise CreditLedgerArgumentError(
            f"user_id={user_id} plan_tier_id={plan_tier_id} not found"
        )

    allotment = int(plan_tier.monthly_credit_allotment or 0)
    if allotment <= 0:
        # A tier configured with zero allotment is legal (e.g. an internal
        # comp tier); no-op silently.
        log.info(
            "credits.allot_monthly: user_id=%s plan_tier=%s allotment=0; no-op",
            user_id, plan_tier.name,
        )
        return None

    # Serialise: only one allotment per user can be in-flight at a time.
    _advisory_lock(db, user_id)

    month_start, next_month_start = _month_window_utc(when)
    existing = (
        db.query(models.CreditLedger)
        .filter(
            models.CreditLedger.user_id == user_id,
            models.CreditLedger.reason == "monthly_allotment",
            models.CreditLedger.created_at >= month_start,
            models.CreditLedger.created_at < next_month_start,
        )
        .first()
    )
    if existing is not None:
        log.info(
            "credits.allot_monthly: user_id=%s month=%s already allotted "
            "(ledger_id=%s); no-op",
            user_id, _month_key_utc(when), existing.id,
        )
        return None

    previous_balance = _current_balance(db, user_id)
    new_balance = previous_balance + allotment
    row = models.CreditLedger(
        user_id=user_id,
        delta=allotment,
        reason="monthly_allotment",
        upload_job_id=None,
        path=None,
        publish_kind=None,
        predicted_quota_units=None,
        balance_after=new_balance,
    )
    db.add(row)
    db.flush()

    log.info(
        "credits.allot_monthly: user_id=%s plan_tier=%s month=%s "
        "delta=+%s balance_after=%s ledger_id=%s",
        user_id, plan_tier.name, _month_key_utc(when),
        allotment, new_balance, row.id,
    )
    return ReserveResult(
        previous_balance=previous_balance,
        cost=-allotment,         # negative cost — semantic credit
        new_balance=new_balance,
        ledger_id=int(row.id),
    )


def check_path_allowed(user_or_plan_tier, upload_path: str) -> None:
    """Decision 6 enforcement: raise ``PlanTierViolationError`` if the
    user's plan_tier disallows the requested upload_path.

    Accepts either a ``User`` (with ``user.plan_tier`` resolved via the
    SQLAlchemy relationship), or a ``PlanTier`` directly. Defensive
    against either shape so callers don't need to reach through the ORM.
    """
    if upload_path not in ("direct", "rtmp"):
        raise CreditLedgerArgumentError(
            f"upload_path must be 'direct' or 'rtmp'; got {upload_path!r}"
        )

    # Resolve the PlanTier.
    plan_tier = None
    if isinstance(user_or_plan_tier, models.PlanTier):
        plan_tier = user_or_plan_tier
    else:
        plan_tier = getattr(user_or_plan_tier, "plan_tier", None)

    if plan_tier is None:
        # Defense-in-depth: refuse Direct when we cannot prove the tier
        # allows it. RTMP is universally allowed.
        if upload_path == "direct":
            raise PlanTierViolationError(
                "direct_path_requires_pro",
                "plan_tier unresolved; refusing direct path",
            )
        return

    if upload_path == "direct" and not bool(getattr(plan_tier, "direct_path_allowed", True)):
        raise PlanTierViolationError(
            "direct_path_requires_pro",
            f"plan_tier={getattr(plan_tier, 'name', '?')!r} disallows upload_path='direct'",
        )


__all__ = [
    "InsufficientCreditsError",
    "CreditLedgerArgumentError",
    "PlanTierViolationError",
    "ReserveResult",
    "reserve",
    "refund",
    "get_balance",
    "allot_monthly",
    "check_path_allowed",
]
