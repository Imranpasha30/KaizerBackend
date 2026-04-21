"""Slot-picker — finds the next open publish slot on a channel.

Rules:
  - Start from max(now + 10min, last_scheduled_upload + spacing).
  - Skip quiet hours (campaign.quiet_hours_start..end, IST).
  - Respect daily_cap (count today's scheduled+published uploads).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

import models

IST_OFFSET = timedelta(hours=5, minutes=30)


def _to_ist(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone(IST_OFFSET))


def _in_quiet_hours(dt: datetime, start: int, end: int) -> bool:
    """start/end are IST hours [0,23]. start==end disables quiet hours."""
    if start == end:
        return False
    h = _to_ist(dt).hour
    if start < end:
        return start <= h < end
    # Wrap around midnight (e.g. 22 → 6)
    return h >= start or h < end


def _advance_past_quiet(dt: datetime, start: int, end: int) -> datetime:
    """If dt falls in quiet hours, move forward to the end of the quiet window."""
    if start == end:
        return dt
    while _in_quiet_hours(dt, start, end):
        dt = dt + timedelta(minutes=15)
    return dt


def count_scheduled_today(db: Session, channel_id: int) -> int:
    """Upload jobs scheduled or already done today (IST) on this channel."""
    now_ist = _to_ist(datetime.now(timezone.utc))
    day_start_ist = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
    day_start_utc = day_start_ist.astimezone(timezone.utc)
    day_end_utc   = day_start_utc + timedelta(days=1)

    return (
        db.query(models.UploadJob)
          .filter(models.UploadJob.channel_id == channel_id)
          .filter(models.UploadJob.status.in_(["queued", "uploading", "processing", "done"]))
          .filter(
              (models.UploadJob.publish_at.between(day_start_utc, day_end_utc))
              | (
                  (models.UploadJob.publish_at.is_(None))
                  & (models.UploadJob.created_at.between(day_start_utc, day_end_utc))
              )
          )
          .count()
    )


def next_slot(
    db: Session,
    channel_id: int,
    spacing_minutes: int = 120,
    quiet_start: int = 0,
    quiet_end:   int = 0,
    after: Optional[datetime] = None,
) -> datetime:
    """Return the next free publish time on this channel (UTC).

    Looks up the latest publish_at among queued/done jobs for this channel,
    adds `spacing_minutes`, then advances past any quiet-hour window.
    """
    now = datetime.now(timezone.utc)
    min_start = (after or now) + timedelta(minutes=10)

    latest = (
        db.query(models.UploadJob.publish_at)
          .filter(models.UploadJob.channel_id == channel_id)
          .filter(models.UploadJob.publish_at.isnot(None))
          .filter(models.UploadJob.status.in_(["queued", "uploading", "processing", "done"]))
          .order_by(models.UploadJob.publish_at.desc())
          .first()
    )
    latest_dt = latest[0] if latest and latest[0] else None
    if latest_dt and latest_dt.tzinfo is None:
        latest_dt = latest_dt.replace(tzinfo=timezone.utc)

    if latest_dt:
        candidate = max(min_start, latest_dt + timedelta(minutes=spacing_minutes))
    else:
        candidate = min_start

    return _advance_past_quiet(candidate, quiet_start, quiet_end)
