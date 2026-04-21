"""Performance router — feedback loop read API."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from sqlalchemy.orm import Session

from database import get_db
import models
from analytics import calibrator, poller


router = APIRouter(prefix="/api/performance", tags=["performance"])


@router.get("/leaderboard")
def leaderboard(
    limit: int = Query(20, ge=1, le=100),
    channel_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    return calibrator.channel_leaderboard(db, limit=limit, channel_id=channel_id)


@router.get("/calibration")
def calibration(
    channel_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    return calibrator.calibration_report(db, channel_id=channel_id)


@router.get("/history/{upload_job_id}")
def history(upload_job_id: int, db: Session = Depends(get_db)):
    """Time series of samples for one upload."""
    rows = (
        db.query(models.ClipPerformance)
          .filter(models.ClipPerformance.upload_job_id == upload_job_id)
          .order_by(models.ClipPerformance.sampled_at.asc())
          .all()
    )
    return [{
        "sampled_at":          r.sampled_at.isoformat() if r.sampled_at else None,
        "hours_since_publish": round(r.hours_since_publish or 0, 2),
        "views":               r.views,
        "likes":               r.likes,
        "comments":            r.comments,
        "seo_score":           r.seo_score,
    } for r in rows]


@router.post("/poll")
def trigger_poll(background: BackgroundTasks):
    """Force a stats poll now instead of waiting for the next cron tick."""
    background.add_task(poller.poll_once)
    return {"triggered": True}
