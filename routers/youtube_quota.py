"""YouTube Data API quota dashboard endpoint.

Surfaces the YouTubeApiCall log as a per-day summary so the V4 editor
can render a "4 / 6 uploads used today" widget without operators
having to dig into the admin Usage page.

Default daily cap is 10,000 quota units per Google Cloud project.
videos.insert is 1,600 units = ~6 uploads/day. The operator can request
a higher cap from Google; this dashboard makes the burn rate visible
so they know whether they need to.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, and_
from sqlalchemy.orm import Session

import auth
import models
from database import get_db


router = APIRouter(prefix="/api/youtube/quota", tags=["youtube-quota"])

# Default Google Cloud daily cap. Operator can request more from
# Google Cloud Console. When/if they do, set KAIZER_YT_DAILY_QUOTA_CAP
# in .env to override.
import os
DAILY_CAP = int(os.environ.get("KAIZER_YT_DAILY_QUOTA_CAP", "10000"))


def _day_window(day: datetime) -> tuple[datetime, datetime]:
    """[00:00, 24:00) UTC window for the given day."""
    start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


@router.get("/today")
def quota_today(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Per-operator quota snapshot for today. Returns:
      - used / cap / remaining (units)
      - uploads_used: count of videos.insert calls today
      - uploads_remaining: cap / 1600, floor-divided
      - by_operation: breakdown of {operation: units}
      - by_channel: breakdown of {channel_id: units}
      - last_5_uploads: most recent videos.insert rows for context
    """
    start, end = _day_window(datetime.now(timezone.utc))

    total_used_row = (
        db.query(func.coalesce(func.sum(models.YouTubeApiCall.quota_cost), 0))
          .filter(
              models.YouTubeApiCall.user_id == user.id,
              models.YouTubeApiCall.created_at >= start,
              models.YouTubeApiCall.created_at < end,
          )
          .first()
    )
    used = int(total_used_row[0] or 0)

    uploads_used_row = (
        db.query(func.count(models.YouTubeApiCall.id))
          .filter(
              models.YouTubeApiCall.user_id == user.id,
              models.YouTubeApiCall.operation == "videos.insert",
              models.YouTubeApiCall.created_at >= start,
              models.YouTubeApiCall.created_at < end,
          )
          .first()
    )
    uploads_used = int(uploads_used_row[0] or 0)

    by_op = (
        db.query(
              models.YouTubeApiCall.operation,
              func.sum(models.YouTubeApiCall.quota_cost),
              func.count(models.YouTubeApiCall.id),
          )
          .filter(
              models.YouTubeApiCall.user_id == user.id,
              models.YouTubeApiCall.created_at >= start,
              models.YouTubeApiCall.created_at < end,
          )
          .group_by(models.YouTubeApiCall.operation)
          .all()
    )

    by_channel = (
        db.query(
              models.YouTubeApiCall.channel_id,
              models.YouTubeApiCall.google_channel_id,
              func.sum(models.YouTubeApiCall.quota_cost),
              func.count(models.YouTubeApiCall.id),
          )
          .filter(
              models.YouTubeApiCall.user_id == user.id,
              models.YouTubeApiCall.created_at >= start,
              models.YouTubeApiCall.created_at < end,
          )
          .group_by(models.YouTubeApiCall.channel_id, models.YouTubeApiCall.google_channel_id)
          .all()
    )

    last_5 = (
        db.query(models.YouTubeApiCall)
          .filter(
              models.YouTubeApiCall.user_id == user.id,
              models.YouTubeApiCall.operation == "videos.insert",
              models.YouTubeApiCall.created_at >= start - timedelta(hours=24),
          )
          .order_by(models.YouTubeApiCall.created_at.desc())
          .limit(5)
          .all()
    )

    upload_cost = 1600
    return {
        "cap": DAILY_CAP,
        "used": used,
        "remaining": max(0, DAILY_CAP - used),
        "pct": (used / DAILY_CAP * 100) if DAILY_CAP > 0 else 0,
        "uploads_used": uploads_used,
        "uploads_remaining_estimate": max(0, (DAILY_CAP - used) // upload_cost),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "by_operation": [
            {"operation": op, "units": int(u or 0), "calls": int(c or 0)}
            for (op, u, c) in by_op
        ],
        "by_channel": [
            {
                "channel_id": cid,
                "google_channel_id": gcid or "",
                "units": int(u or 0),
                "calls": int(c or 0),
            }
            for (cid, gcid, u, c) in by_channel
        ],
        "last_uploads": [
            {
                "video_id": r.video_id,
                "upload_job_id": r.upload_job_id,
                "channel_id": r.channel_id,
                "publish_kind": r.publish_kind,
                "quota_cost": r.quota_cost,
                "success": r.success,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in last_5
        ],
    }
