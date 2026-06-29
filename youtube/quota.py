"""YouTube Data API v3 quota tracking.

Per-day, per-API-key accounting persisted in `api_quota`. Decorator is opt-in;
call sites that care about quota wrap their call site. Defaults are Google's
published v3 costs — tune as you measure.
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

import models


DAILY_LIMIT = 10_000   # Google's default project quota (fallback only)


def _resolved_limit() -> int:
    """Quota-truth fix: prefer the live-synced Google cap (see
    services/quota_sync.py) over the hardcoded constant, so the legacy
    gate and the v2 gate enforce the SAME real number. Falls back to
    the stock 10,000 when quota_sync is unavailable."""
    try:
        from services.quota_sync import resolved_daily_cap
        return int(resolved_daily_cap())
    except Exception:
        return DAILY_LIMIT


def _today_ist() -> str:
    """YYYY-MM-DD in IST — matches Google's quota rollover at midnight PT.

    We use IST here because this stack is Telugu-news targeted; the exact
    timezone alignment doesn't matter as long as it's consistent with where
    the pipeline runs. The persisted date is just a bucket key.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _key_hash(api_key: str) -> str:
    # Test isolation — mirrors quota_v2: suites set
    # KAIZER_YT_QUOTA_BUCKET so fake burns never pollute the real
    # 'oauth' bucket the UI reports.
    forced = os.environ.get("KAIZER_YT_QUOTA_BUCKET", "").strip()
    if forced:
        return forced[:16]
    if not api_key:
        return "oauth"   # OAuth-mediated calls don't carry a raw API key
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


def reserve(db: Session, cost: int, api_key: Optional[str] = None) -> bool:
    """Try to charge `cost` units to today's bucket. Returns True if under
    the daily limit, False if reserving would exceed it."""
    if cost <= 0:
        return True
    date = _today_ist()
    kh = _key_hash(api_key or "")

    row = (
        db.query(models.ApiQuota)
          .filter(models.ApiQuota.date == date, models.ApiQuota.api_key_hash == kh)
          .first()
    )
    if not row:
        row = models.ApiQuota(date=date, api_key_hash=kh, units_used=0)
        db.add(row)
        db.flush()

    if (row.units_used or 0) + cost > _resolved_limit():
        return False

    row.units_used = (row.units_used or 0) + cost
    db.commit()
    return True


def snapshot(db: Session, api_key: Optional[str] = None) -> dict:
    date = _today_ist()
    kh = _key_hash(api_key or "")
    row = (
        db.query(models.ApiQuota)
          .filter(models.ApiQuota.date == date, models.ApiQuota.api_key_hash == kh)
          .first()
    )
    used = (row.units_used if row else 0)
    limit = _resolved_limit()
    return {
        "date": date,
        "used": used,
        "limit": limit,
        "remaining": max(0, limit - used),
    }


# Published costs for YouTube Data API v3 (rough — revisit if Google changes them)
# videos.insert: repriced 1,600 → 100 on 2025-12-04 (own 100/day bucket since 2026-06-01).
COST_VIDEO_INSERT   = 100
COST_THUMBNAIL_SET  = 50
COST_CHANNELS_LIST  = 1
COST_PLAYLISTS_LIST = 1
COST_PLAYLIST_ITEMS = 1
COST_VIDEOS_LIST    = 1
