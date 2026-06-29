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


def _pick_pollable_uploads(db: Session, channel_id: int | None = None) -> List[models.UploadJob]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=SAMPLE_WINDOW_DAYS)
    q = (
        db.query(models.UploadJob)
          .filter(models.UploadJob.status.in_(["done", "processing"]))
          .filter(models.UploadJob.video_id != "")
          .filter(models.UploadJob.updated_at >= cutoff)
    )
    if channel_id is not None:
        q = q.filter(models.UploadJob.channel_id == int(channel_id))
    return (
        q.order_by(models.UploadJob.updated_at.desc())
         .limit(500)
         .all()
    )


def _pick_pollable_v2(db: Session, channel_id: int | None = None) -> List["models.UploadJobV2"]:
    """Recent COMPLETED V2 uploads that landed on YouTube — the live publish
    path. These never reach ClipPerformance (its FK is legacy-only), so we feed
    them straight into the training dataset via ``record_sample``."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=SAMPLE_WINDOW_DAYS)
    q = (
        db.query(models.UploadJobV2)
          .filter(models.UploadJobV2.status == "completed")
          .filter(models.UploadJobV2.youtube_video_id.isnot(None))
          .filter(models.UploadJobV2.youtube_video_id != "")
          .filter(models.UploadJobV2.updated_at >= cutoff)
    )
    if channel_id is not None:
        q = q.filter(models.UploadJobV2.channel_id == int(channel_id))
    return (
        q.order_by(models.UploadJobV2.updated_at.desc())
         .limit(500)
         .all()
    )


def _clip_for_v2(db: Session, job):
    """Resolve the Clip (which carries the SEO that shipped) for a V2 job.
    V2 jobs are parented by publish_task → master_video; the Clip is reached
    via ``MasterVideo.clip_id``. Falls back to ``job.clip_id`` (legacy bridge)
    when set. Returns None if unresolvable."""
    try:
        if getattr(job, "clip_id", None):
            c = db.query(models.Clip).filter(models.Clip.id == job.clip_id).first()
            if c:
                return c
        pt = db.query(models.PublishTask).filter(
            models.PublishTask.id == job.publish_task_id).first()
        if not pt:
            return None
        mv = db.query(models.MasterVideo).filter(
            models.MasterVideo.id == pt.master_video_id).first()
        if mv and mv.clip_id:
            return db.query(models.Clip).filter(models.Clip.id == mv.clip_id).first()
    except Exception:
        return None
    return None


def _poll_v2(db: Session, public_client, channel_id: int | None = None) -> int:
    """Sample COMPLETED V2 uploads → TrainingSample rows (no ClipPerformance,
    whose FK only accepts legacy ids). Mirrors the legacy sweep: group by
    channel, batch stats + opportunistic CTR, upsert one row per video.
    Best-effort throughout — never raises, commits per channel."""
    from types import SimpleNamespace

    rows = _pick_pollable_v2(db, channel_id=channel_id)
    if not rows:
        return 0

    by_channel: dict = {}
    for u in rows:
        by_channel.setdefault(u.channel_id, []).append(u)

    sampled = 0
    for ch_id, jobs in by_channel.items():
        yt = public_client or _authed_yt_for(db, ch_id)
        if not yt:
            continue
        id_to_job = {j.youtube_video_id: j for j in jobs if j.youtube_video_id}
        video_ids = list(id_to_job.keys())

        for batch in _chunks(video_ids, BATCH_SIZE):
            try:
                resp = yt.videos().list(
                    part="statistics", id=",".join(batch), maxResults=BATCH_SIZE,
                ).execute()
            except HttpError as e:
                print(f"[analytics] v2 videos.list failed: {e}")
                continue
            try:
                from analytics.ctr import fetch_video_ctr
                ctr_map = fetch_video_ctr(db, ch_id, batch)
            except Exception:
                ctr_map = {}

            for item in resp.get("items") or []:
                vid = item.get("id")
                job = id_to_job.get(vid)
                if not job:
                    continue
                stats = item.get("statistics") or {}
                views = int(stats.get("viewCount") or 0)
                likes = int(stats.get("likeCount") or 0)
                comments = int(stats.get("commentCount") or 0)
                ctr_val = (ctr_map.get(vid) or {}).get("ctr")

                ref = job.finished_at or job.updated_at or job.created_at
                hours_since = 0.0
                if ref:
                    ref = ref if ref.tzinfo else ref.replace(tzinfo=timezone.utc)
                    hours_since = (datetime.now(timezone.utc) - ref).total_seconds() / 3600

                clip = _clip_for_v2(db, job)
                shim = SimpleNamespace(video_id=vid, channel_id=job.channel_id, id=job.id)
                try:
                    from learning import dataset as _ds
                    if _ds.record_sample(
                        db, upload_job=shim, clip=clip,
                        views=views, likes=likes, comments=comments,
                        hours_since_publish=hours_since, ctr=ctr_val,
                    ) is not None:
                        sampled += 1
                except Exception:
                    pass
        try:
            db.commit()
        except Exception:
            db.rollback()
    return sampled


def _chunks(seq: List, n: int) -> Iterable[List]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def poll_once(channel_id: int | None = None) -> dict:
    """Single sweep — poll stats for every recent upload, write ClipPerformance rows.

    When ``channel_id`` is provided, only uploads for that style profile
    are sampled.  Powers the per-channel "Poll Now" button on the
    Performance page so the admin can refresh one channel's numbers
    without burning YouTube Data API quota on the others.
    """
    db = SessionLocal()
    try:
        uploads = _pick_pollable_uploads(db, channel_id=channel_id)
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

                # Opportunistic CTR (the "gold" label) — returns {} unless the
                # channel re-approved with the analytics scope. Best-effort.
                try:
                    from analytics.ctr import fetch_video_ctr
                    ctr_map = fetch_video_ctr(db, ch_id, batch)
                except Exception:
                    ctr_map = {}

                for item in resp.get("items") or []:
                    vid = item.get("id")
                    row = id_to_row.get(vid)
                    if not row:
                        continue
                    stats = item.get("statistics") or {}
                    views = int(stats.get("viewCount") or 0)
                    likes = int(stats.get("likeCount") or 0)
                    comments = int(stats.get("commentCount") or 0)
                    ctr_val = (ctr_map.get(vid) or {}).get("ctr")

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
                    # Snapshot a clean, training-ready row (features + this
                    # outcome). Best-effort — learning collection must never
                    # break the poll. Committed with the batch below.
                    try:
                        from learning import dataset as _ds
                        _ds.record_sample(
                            db, upload_job=row, clip=row.clip,
                            views=views, likes=likes, comments=comments,
                            hours_since_publish=hours_since, ctr=ctr_val,
                        )
                    except Exception:
                        pass
            db.commit()

        # Live publish path: COMPLETED V2 uploads feed the training dataset
        # directly (they never reach ClipPerformance — legacy-only FK).
        v2_sampled = _poll_v2(db, public_client, channel_id=channel_id)

        return {
            "sampled_at": datetime.now(timezone.utc).isoformat(),
            "sampled":    sampled,
            "v2_sampled": v2_sampled,
            "uploads":    len(uploads),
        }
    except Exception:
        traceback.print_exc()
        return {"sampled_at": datetime.now(timezone.utc).isoformat(), "error": "poll failed"}
    finally:
        db.close()
