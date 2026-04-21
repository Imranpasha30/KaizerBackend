"""Hourly stats poller — samples views/likes/comments for recent uploads."""
from __future__ import annotations

import json
import traceback
from datetime import datetime, timedelta, timezone
from typing import Iterable, List

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sqlalchemy.orm import Session

import models
from config import settings
from database import SessionLocal
from youtube import oauth as yt_oauth


SAMPLE_WINDOW_DAYS = 7
BATCH_SIZE = 50


def _extract_score(clip: models.Clip) -> int:
    if not clip or not clip.seo:
        return 0
    try:
        return int(json.loads(clip.seo).get("seo_score") or 0)
    except Exception:
        return 0


def _public_yt():
    if settings.yt_data_api_key:
        return build("youtube", "v3", developerKey=settings.yt_data_api_key,
                     cache_discovery=False)
    return None


def _authed_yt_for(db: Session, channel_id: int):
    try:
        creds = yt_oauth.get_credentials(db, channel_id)
    except yt_oauth.OAuthError:
        return None
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def _pick_pollable_uploads(db: Session) -> List[models.UploadJob]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=SAMPLE_WINDOW_DAYS)
    return (
        db.query(models.UploadJob)
          .filter(models.UploadJob.status.in_(["done", "processing"]))
          .filter(models.UploadJob.video_id != "")
          .filter(models.UploadJob.updated_at >= cutoff)
          .order_by(models.UploadJob.updated_at.desc())
          .limit(500)
          .all()
    )


def _chunks(seq: List, n: int) -> Iterable[List]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def poll_once() -> dict:
    """Single sweep — poll stats for every recent upload, write ClipPerformance rows."""
    db = SessionLocal()
    try:
        uploads = _pick_pollable_uploads(db)
        if not uploads:
            return {"sampled_at": datetime.now(timezone.utc).isoformat(),
                    "sampled":    0,
                    "note":       "no recent uploads"}

        public_client = _public_yt()
        sampled = 0

        # Group by channel when we need authed clients (no public API key).
        by_channel = {}
        for u in uploads:
            by_channel.setdefault(u.channel_id, []).append(u)

        for ch_id, rows in by_channel.items():
            yt = public_client or _authed_yt_for(db, ch_id)
            if not yt:
                continue  # cannot poll this batch

            video_ids = [r.video_id for r in rows if r.video_id]
            id_to_row = {r.video_id: r for r in rows}

            for batch in _chunks(video_ids, BATCH_SIZE):
                try:
                    resp = yt.videos().list(
                        part="statistics",
                        id=",".join(batch),
                        maxResults=BATCH_SIZE,
                    ).execute()
                except HttpError as e:
                    print(f"[analytics] videos.list failed: {e}")
                    continue

                for item in resp.get("items") or []:
                    vid = item.get("id")
                    row = id_to_row.get(vid)
                    if not row:
                        continue
                    stats = item.get("statistics") or {}
                    views = int(stats.get("viewCount") or 0)
                    likes = int(stats.get("likeCount") or 0)
                    comments = int(stats.get("commentCount") or 0)

                    hours_since = 0.0
                    if row.updated_at:
                        ref = row.updated_at if row.updated_at.tzinfo else row.updated_at.replace(tzinfo=timezone.utc)
                        hours_since = (datetime.now(timezone.utc) - ref).total_seconds() / 3600

                    perf = models.ClipPerformance(
                        upload_job_id=row.id,
                        clip_id=row.clip_id,
                        channel_id=row.channel_id,
                        video_id=vid,
                        views=views,
                        likes=likes,
                        comments=comments,
                        seo_score=_extract_score(row.clip),
                        hours_since_publish=hours_since,
                    )
                    db.add(perf)
                    sampled += 1
            db.commit()

        return {
            "sampled_at": datetime.now(timezone.utc).isoformat(),
            "sampled":    sampled,
            "uploads":    len(uploads),
        }
    except Exception:
        traceback.print_exc()
        return {"sampled_at": datetime.now(timezone.utc).isoformat(), "error": "poll failed"}
    finally:
        db.close()
