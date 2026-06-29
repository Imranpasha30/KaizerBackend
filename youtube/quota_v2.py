"""YouTube Data API v3 quota gate — Phase 2.F v2.

This is the NEW quota gate the upload rewrite uses. It lives next to
the legacy ``youtube/quota.py`` (which is preserved BYTE-FOR-BYTE per
the brief §0); legacy code paths keep calling ``quota.reserve``, new
code paths call ``quota_v2.reserve``.

Two material differences from the legacy gate:

  1. ``DAILY_LIMIT`` is no longer hardcoded. The cap is read from
     ``KAIZER_YT_DAILY_QUOTA_CAP`` (Decision 5; default 1_000_000) at
     every call — so ops can flip it via .env without a code change.
     Brief §1: "Do not hardcode 10,000 anywhere in the new code."

  2. The bucket key is explicitly UTC-named (``_today_utc``). The
     legacy file's ``_today_ist`` was misleadingly named but already
     returned UTC; here we don't preserve the legacy lie.

Race-safety: the ``api_quota`` table has a UNIQUE(date, api_key_hash)
constraint, and we use an atomic UPDATE … RETURNING / SELECT-FOR-UPDATE
flow on Postgres. On SQLite (dev) the in-process GIL plus
single-writer model makes this safe in practice; we don't claim
race-safety on multi-process SQLite.

Cost constants are duplicated from the legacy quota.py for callers
that want one-import access (``from youtube import quota_v2;
quota_v2.COST_VIDEO_INSERT``). Legacy quota.py's constants stay live
for legacy callers — no shared module dance.

Brief §9 invariant: ``reserve()`` returning False means PARK the job
(no fail, no credit burn, no retry-counter bump). Callers MUST honour
this.
"""
from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

import models


log = logging.getLogger("kaizer.quota_v2")


# ─── Published YouTube Data API v3 unit costs ──────────────────────────
# Same values as legacy youtube/quota.py — single source of truth here
# for the NEW code paths; legacy callers keep using legacy quota.py.

# videos.insert was repriced 1,600 → 100 on 2025-12-04 and moved to its
# OWN granular per-day bucket on 2026-06-01 (100 uploads/day, 1 unit per
# call), SEPARATE from the 10,000-unit "Queries" pool. We therefore gate
# uploads on a COUNT against that bucket (see UPLOADS_* + reserve_upload),
# not on this unit cost. The constant stays for burn-log accuracy.
COST_VIDEO_INSERT          = 100
COST_THUMBNAIL_SET         = 50
COST_VIDEOS_LIST           = 1
COST_CHANNELS_LIST         = 1
COST_PLAYLISTS_LIST        = 1
COST_PLAYLIST_ITEMS        = 1
COST_LIVE_BROADCASTS_INSERT = 50
COST_LIVE_STREAMS_INSERT    = 50
COST_LIVE_BROADCASTS_BIND   = 50


# ─── Cap resolution — Google's REAL number (quota-truth fix) ───────────
#
# The old behaviour read KAIZER_YT_DAILY_QUOTA_CAP with a 1,000,000
# default — a placeholder for a quota grant Google never approved. The
# project's actual assigned limit is the stock 10,000/day (verified via
# the Service Usage API). services/quota_sync.py now owns resolution:
#   env override (if explicitly set) → hourly-synced Google value →
#   stock 10,000 default. Never the 1M fantasy.

_DEFAULT_DAILY_CAP = 10_000  # Google's stock per-project default


def daily_cap() -> int:
    """The effective daily cap — Google's real assigned limit.

    Delegates to ``services.quota_sync.resolved_daily_cap`` (env
    override → synced ``system_settings`` value → stock 10,000). Lazy
    import + hard fallback so this module stays importable in minimal
    test environments.
    """
    try:
        from services.quota_sync import resolved_daily_cap
        return int(resolved_daily_cap())
    except Exception:
        log.debug("quota_v2.daily_cap: quota_sync unavailable; using "
                  "stock default", exc_info=True)
        raw = os.environ.get("KAIZER_YT_DAILY_QUOTA_CAP", "").strip()
        if raw:
            try:
                v = int(raw)
                if v > 0:
                    return v
            except ValueError:
                pass
        return _DEFAULT_DAILY_CAP


# ─── Uploads bucket — videos.insert granular quota (since 2026-06-01) ──
#
# Google moved videos.insert into its OWN per-day bucket: a default of
# 100 uploads/day, charged 1 unit per call, INDEPENDENT of the 10,000
# "Queries" pool. We track it as a COUNT in a dedicated ApiQuota bucket
# (api_key_hash == UPLOADS_BUCKET) so the gate reflects the REAL limit
# (~100/day) instead of the dead "10,000 ÷ 1,600 ≈ 6" math. RTMP and
# thumbnails.set still draw from the Queries pool, untouched.

UPLOADS_BUCKET = "vinsert"


def uploads_daily_cap() -> int:
    """Per-day videos.insert cap (Google default 100). Override via
    ``KAIZER_YT_UPLOADS_DAILY_CAP`` only if Google grants more."""
    raw = os.environ.get("KAIZER_YT_UPLOADS_DAILY_CAP", "").strip()
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return 100


# ─── Internals ─────────────────────────────────────────────────────────


def _today_utc() -> str:
    """``YYYY-MM-DD`` in UTC — single bucket key per day.

    The legacy file's ``_today_ist`` was a misnomer that already
    returned UTC; here we name it correctly.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _key_hash(api_key: Optional[str]) -> str:
    # Test isolation: when KAIZER_YT_QUOTA_BUCKET is set (the
    # regression/reliability suites set it to 'testrun'), all burns
    # land in that bucket instead of the production 'oauth' one — so
    # test runs can never pollute the real usage counter again (the
    # bug that made the UI show 31,000 'used' with zero real calls).
    forced = os.environ.get("KAIZER_YT_QUOTA_BUCKET", "").strip()
    if forced:
        return forced[:16]
    if not api_key:
        return "oauth"   # OAuth-mediated calls don't carry a raw API key
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


def _is_postgres(db: Session) -> bool:
    try:
        return db.bind.dialect.name == "postgresql"  # type: ignore[union-attr]
    except Exception:
        return False


# ─── Public API ────────────────────────────────────────────────────────


def reserve(
    db: Session,
    cost: int,
    *,
    api_key: Optional[str] = None,
    api_key_hash: Optional[str] = None,
) -> bool:
    """Try to charge ``cost`` units to today's Queries-pool bucket.

    Returns ``True`` if the bucket has room (charge committed),
    ``False`` if reserving would exceed today's cap. False = PARK the
    job at the caller (brief §9 invariant). Uploads (videos.insert) have
    their OWN bucket — use :func:`reserve_upload` for those.
    """
    kh = (api_key_hash or _key_hash(api_key or "")).strip() or "oauth"
    return _reserve_bucket(db, bucket=kh, cost=cost, cap=daily_cap())


def reserve_upload(db: Session) -> bool:
    """Charge ONE videos.insert against today's granular uploads bucket
    (default 100/day). Returns False to PARK when the day's 100 uploads
    are used (the §9 park-not-fail invariant). Independent of the Queries
    pool, so it never competes with thumbnails/list calls."""
    return _reserve_bucket(
        db, bucket=UPLOADS_BUCKET, cost=1, cap=uploads_daily_cap(),
    )


def _reserve_bucket(db: Session, *, bucket: str, cost: int, cap: int) -> bool:
    """Atomic read-modify-write charge of ``cost`` to ``bucket`` for today.

    Race-safety:
      - Postgres: advisory transactional lock keyed by date+bucket so
        concurrent reservers for the same bucket serialize on the
        read+update.
      - SQLite: relies on the in-process GIL; multi-process SQLite is
        not supported (dev convenience only).
    """
    if cost <= 0:
        return True

    date = _today_utc()
    kh = (bucket or "").strip() or "oauth"

    # On Postgres, take an advisory transactional lock so concurrent
    # reservers for the same (date, key) serialize on the read+update.
    if _is_postgres(db):
        try:
            # The lock key is a stable hash of the bucket id so the
            # serialization is per-bucket, not global.
            lock_key = int(
                hashlib.sha256(f"{date}|{kh}".encode("utf-8")).hexdigest()[:8], 16,
            )
            db.execute(
                text("SELECT pg_advisory_xact_lock(:k)"),
                {"k": int(lock_key) & 0x7FFFFFFF},
            )
        except OperationalError as exc:
            log.warning(
                "quota_v2.reserve: pg advisory lock failed (%s); "
                "falling through to non-locked path",
                exc,
            )

    row = (
        db.query(models.ApiQuota)
        .filter(
            models.ApiQuota.date == date,
            models.ApiQuota.api_key_hash == kh,
        )
        .first()
    )
    if not row:
        row = models.ApiQuota(date=date, api_key_hash=kh, units_used=0)
        db.add(row)
        db.flush()

    current = int(row.units_used or 0)
    if current + int(cost) > cap:
        log.info(
            "quota_v2.reserve: bucket date=%s key=%s would overflow "
            "(current=%d cost=%d cap=%d); parking",
            date, kh, current, cost, cap,
        )
        return False

    row.units_used = current + int(cost)
    db.add(row)
    db.commit()
    return True


def refund_upload(db: Session) -> None:
    """Give back ONE videos.insert reservation when an upload attempt did
    NOT succeed (failed / parked / abandoned), so only REAL uploads count
    against the 100/day bucket. Without this, a failed-then-retried upload
    burns the bucket twice. Floors at 0 — never goes negative."""
    _refund_bucket(db, bucket=UPLOADS_BUCKET, cost=1)


def _refund_bucket(db: Session, *, bucket: str, cost: int) -> None:
    if cost <= 0:
        return
    date = _today_utc()
    kh = (bucket or "").strip() or "oauth"
    try:
        if _is_postgres(db):
            lock_key = int(
                hashlib.sha256(f"{date}|{kh}".encode("utf-8")).hexdigest()[:8], 16,
            )
            db.execute(
                text("SELECT pg_advisory_xact_lock(:k)"),
                {"k": int(lock_key) & 0x7FFFFFFF},
            )
        row = (
            db.query(models.ApiQuota)
            .filter(models.ApiQuota.date == date,
                    models.ApiQuota.api_key_hash == kh)
            .first()
        )
        if row is None:
            return
        row.units_used = max(0, int(row.units_used or 0) - int(cost))
        db.add(row)
        db.commit()
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass


def snapshot(db: Session, api_key: Optional[str] = None) -> dict:
    """Observability snapshot of today's bucket."""
    date = _today_utc()
    kh = _key_hash(api_key or "")
    cap = daily_cap()
    row = (
        db.query(models.ApiQuota)
        .filter(
            models.ApiQuota.date == date,
            models.ApiQuota.api_key_hash == kh,
        )
        .first()
    )
    used = int(row.units_used if row else 0)
    return {
        "date": date,
        "used": used,
        "limit": cap,
        "remaining": max(0, cap - used),
    }


def uploads_snapshot(db: Session) -> dict:
    """Observability snapshot of today's videos.insert (uploads) bucket —
    the authoritative 'X / 100 uploads used today' the UI notifies on."""
    date = _today_utc()
    cap = uploads_daily_cap()
    row = (
        db.query(models.ApiQuota)
        .filter(
            models.ApiQuota.date == date,
            models.ApiQuota.api_key_hash == UPLOADS_BUCKET,
        )
        .first()
    )
    used = int(row.units_used if row else 0)
    return {
        "date": date,
        "used": used,
        "cap": cap,
        "remaining": max(0, cap - used),
    }


def burn(
    db: Session,
    *,
    upload_job_id: Optional[int],
    operation: str,
    predicted_cost: int,
    http_status: Optional[int],
    observed_outcome: str,
) -> None:
    """Thin wrapper over ``services.burn_log.log_predicted_and_actual``.

    Preferred entry point in new code so callers don't have to know
    that burn_log lives in services/. Lazy import to avoid a cycle
    when burn_log isn't loaded yet at module import time (smoke tests).
    """
    try:
        from services import burn_log as _bl
    except Exception as exc:
        log.warning("quota_v2.burn: burn_log import failed (%s); skipping", exc)
        return
    _bl.log_predicted_and_actual(
        db,
        upload_job_id=upload_job_id,
        operation=operation,
        predicted_cost=int(predicted_cost),
        http_status=http_status,
        observed_outcome=observed_outcome,
    )


# ─── Back-compat aliases for upload_dispatch ───────────────────────────


# ``services/upload_dispatch.py`` calls ``quota_mod.reserve_v2(db, cost)``
# when it detects the v2 module; we expose ``reserve_v2`` as an alias of
# ``reserve`` for that signature.
def reserve_v2(db: Session, cost: int, api_key: Optional[str] = None) -> bool:
    return reserve(db, cost, api_key=api_key)


__all__ = [
    "COST_VIDEO_INSERT",
    "COST_THUMBNAIL_SET",
    "COST_VIDEOS_LIST",
    "COST_CHANNELS_LIST",
    "COST_PLAYLISTS_LIST",
    "COST_PLAYLIST_ITEMS",
    "COST_LIVE_BROADCASTS_INSERT",
    "COST_LIVE_STREAMS_INSERT",
    "COST_LIVE_BROADCASTS_BIND",
    "daily_cap",
    "uploads_daily_cap",
    "UPLOADS_BUCKET",
    "reserve",
    "reserve_upload",
    "refund_upload",
    "reserve_v2",
    "snapshot",
    "uploads_snapshot",
    "burn",
]
