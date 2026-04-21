"""Analyze ClipPerformance → derive score-vs-outcome insight.

Bucketizes SEO score and returns the mean/median view count per bucket.
This is what closes the loop: the enforcer's score is a prediction; the
calibrator tells you whether it predicts well.
"""
from __future__ import annotations

from statistics import mean, median
from typing import Dict, List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import func

import models


SCORE_BUCKETS = [(0, 59), (60, 69), (70, 79), (80, 89), (90, 100)]


def _bucket_of(score: int) -> str:
    for lo, hi in SCORE_BUCKETS:
        if lo <= score <= hi:
            return f"{lo}-{hi}"
    return "unknown"


def latest_perf_per_upload(db: Session, channel_id: Optional[int] = None) -> List[models.ClipPerformance]:
    """For each upload, keep the most recent ClipPerformance row (flattens history)."""
    sub = (
        db.query(
            models.ClipPerformance.upload_job_id.label("uid"),
            func.max(models.ClipPerformance.sampled_at).label("max_sampled"),
        )
        .group_by(models.ClipPerformance.upload_job_id)
        .subquery()
    )
    q = (
        db.query(models.ClipPerformance)
          .join(sub, (models.ClipPerformance.upload_job_id == sub.c.uid)
                   & (models.ClipPerformance.sampled_at   == sub.c.max_sampled))
    )
    if channel_id:
        q = q.filter(models.ClipPerformance.channel_id == channel_id)
    return q.all()


def calibration_report(db: Session, channel_id: Optional[int] = None) -> Dict:
    rows = latest_perf_per_upload(db, channel_id)
    by_bucket: Dict[str, List[int]] = {f"{lo}-{hi}": [] for lo, hi in SCORE_BUCKETS}
    for r in rows:
        by_bucket.setdefault(_bucket_of(r.seo_score), []).append(r.views)

    summary = []
    for key, views in by_bucket.items():
        if not views:
            summary.append({"bucket": key, "n": 0, "mean_views": 0, "median_views": 0})
            continue
        summary.append({
            "bucket":       key,
            "n":            len(views),
            "mean_views":   round(mean(views), 1),
            "median_views": round(median(views), 1),
            "max_views":    max(views),
        })

    return {
        "channel_id": channel_id,
        "total_samples": len(rows),
        "by_bucket": summary,
    }


def channel_leaderboard(db: Session, limit: int = 20, channel_id: Optional[int] = None) -> List[Dict]:
    """Top uploads by views (latest sample)."""
    rows = sorted(latest_perf_per_upload(db, channel_id), key=lambda r: r.views, reverse=True)[:limit]
    out = []
    for r in rows:
        job = db.query(models.UploadJob).filter(models.UploadJob.id == r.upload_job_id).first()
        out.append({
            "upload_job_id": r.upload_job_id,
            "video_id": r.video_id,
            "video_url": f"https://youtu.be/{r.video_id}" if r.video_id else "",
            "title": (job.title if job else "")[:120],
            "channel_id": r.channel_id,
            "views": r.views,
            "likes": r.likes,
            "comments": r.comments,
            "seo_score": r.seo_score,
            "hours_since_publish": round(r.hours_since_publish or 0, 1),
            "sampled_at": r.sampled_at.isoformat() if r.sampled_at else None,
        })
    return out
