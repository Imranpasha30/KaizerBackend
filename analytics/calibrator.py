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
    """Top uploads by views (latest sample).

    Includes the REAL YouTube channel name + thumbnail (from
    ``OAuthToken``) alongside the style-profile id, so the Performance
    page can group / display by "Kaizer 30" instead of the internal
    "Personal 3" template name.
    """
    rows = sorted(latest_perf_per_upload(db, channel_id), key=lambda r: r.views, reverse=True)[:limit]
    out = []
    for r in rows:
        job = db.query(models.UploadJob).filter(models.UploadJob.id == r.upload_job_id).first()
        ch  = job.channel if job else None
        tok = ch.oauth_token if ch else None
        # Thumbnail: prefer the clip's still frame (what the user saw
        # in our editor) — falls back to the OAuth channel avatar and
        # finally to nothing.  All three are storage-agnostic URLs.
        clip = job.clip if job else None
        thumb = ""
        if clip:
            thumb = (getattr(clip, "thumb_storage_url", "") or "")
            if not thumb and getattr(clip, "thumb_path", ""):
                thumb = f"/api/file/?path={clip.thumb_path}"
        out.append({
            "upload_job_id": r.upload_job_id,
            "video_id": r.video_id,
            "video_url": f"https://youtu.be/{r.video_id}" if r.video_id else "",
            "title": (job.title if job else "")[:120],
            "thumb_url": thumb,
            # Style-profile id (kept for existing client-side filters).
            "channel_id": r.channel_id,
            "style_profile_name": ch.name if ch else "",
            # Real YouTube channel — what the user actually sees on YT.
            "google_channel_id":  tok.google_channel_id if tok else "",
            "youtube_channel_title": tok.google_channel_title if tok else "",
            "channel_thumbnail_url": tok.channel_thumbnail_url if tok else "",
            "views": r.views,
            "likes": r.likes,
            "comments": r.comments,
            "engagement_rate": round(
                ((r.likes or 0) + (r.comments or 0)) / max(r.views or 1, 1) * 100, 2
            ),
            "seo_score": r.seo_score,
            "hours_since_publish": round(r.hours_since_publish or 0, 1),
            "sampled_at": r.sampled_at.isoformat() if r.sampled_at else None,
        })
    return out


def channel_summary(db: Session, user_id: Optional[int] = None) -> List[Dict]:
    """Per-YouTube-channel rollup for the Performance page header cards.

    Source of truth is the ``channel_videos`` cache (synced from
    YouTube Data API by the user clicking "Sync from YouTube" on each
    channel's card).  This means cards reflect the channel's ACTUAL
    YouTube performance — every public video on it — not just the
    ones Kaizer uploaded.  A channel that's only ever published
    through Postiz (where we don't get a video_id synchronously) still
    shows the correct totals once it's been synced.

    Cards are seeded from every connected OAuthToken so users see
    every channel they own immediately on page load.  Channels that
    haven't been synced yet appear with ``total_videos: 0`` so the
    UI can render a "Sync from YouTube" prompt instead of confusing
    zeros.

    When ``user_id`` is supplied the rollup is scoped to that user.

    Each card returns:
      - google_channel_id / youtube_channel_title / channel avatar
      - aggregate views, likes, comments across the channel's videos
      - engagement rate (likes+comments / views)
      - top clip (the channel's highest-viewed video, from the cache)
      - total_videos / videos_synced — drives the empty-state UI
      - synced_at — most-recent ``last_synced_at`` across the catalogue
      - style_profile_ids — the Kaizer Channels routing to this YT
        account, so the dropdown filter still works for the
        Kaizer-only sub-views (calibration histogram, leaderboard).
    """
    # ── Step 1: every connected YT account becomes a card seed ────
    # Each row is a real YouTube destination the user owns.  Picking
    # this up here (instead of inferring channels from ClipPerformance)
    # is what fixes the "I have 2 channels connected but only 1 shows
    # because the other only ever published through Postiz" bug.
    tok_q = (
        db.query(models.OAuthToken)
          .join(models.Channel, models.Channel.id == models.OAuthToken.channel_id)
          .filter(
              models.OAuthToken.google_channel_id != "",
              models.OAuthToken.refresh_token_enc != "",
          )
    )
    if user_id is not None:
        tok_q = tok_q.filter(models.Channel.user_id == user_id)
    tokens = tok_q.all()

    # Seed bucket per real YouTube channel id.  Multiple style
    # profiles routing to one YT account collapse here naturally.
    groups: Dict[str, Dict] = {}
    profile_ids_by_gcid: Dict[str, List[int]] = {}
    for t in tokens:
        gid = t.google_channel_id
        if not gid:
            continue
        if gid not in groups:
            groups[gid] = {
                "google_channel_id":     gid,
                "youtube_channel_title": t.google_channel_title or "",
                "channel_thumbnail_url": t.channel_thumbnail_url or "",
                "subscriber_count":      int(t.subscriber_count or 0),
                "video_count":           int(t.video_count or 0),
                # Aggregates populated from ``channel_videos`` below.
                # total_videos == 0 means "not synced yet" — the UI
                # renders a "Sync from YouTube" prompt instead of fake
                # zeros so users know they have an action to take.
                "total_videos":          0,
                "total_views":           0,
                "total_likes":           0,
                "total_comments":        0,
                "top_clip":              None,
                "synced_at":             None,
                # Carried in from ClipPerformance overlay, optional.
                "uploads_sampled":       0,
                "avg_seo_score":         0.0,
                "last_sampled_at":       None,
                "style_profile_ids":     [],
            }
            profile_ids_by_gcid[gid] = []
        if t.channel_id and t.channel_id not in profile_ids_by_gcid[gid]:
            profile_ids_by_gcid[gid].append(t.channel_id)

    # ── Step 2: aggregate the full YouTube catalogue per channel ──
    # ``ChannelVideo`` is hydrated by ``/performance/yt/{gcid}/sync``
    # straight from YouTube Data API — it covers every public video on
    # the channel, not just the ones Kaizer published.  That makes the
    # card numbers reflect real channel performance even when no
    # ClipPerformance row exists yet (e.g. Postiz-only uploads).
    video_q = db.query(models.ChannelVideo)
    if user_id is not None:
        video_q = video_q.filter(models.ChannelVideo.user_id == user_id)
    catalogue = video_q.all()

    for v in catalogue:
        gid = v.google_channel_id
        bucket = groups.get(gid)
        if bucket is None:
            # Stale row pointing at a disconnected token — skip.
            continue
        bucket["total_videos"]    += 1
        bucket["total_views"]     += int(v.view_count    or 0)
        bucket["total_likes"]     += int(v.like_count    or 0)
        bucket["total_comments"]  += int(v.comment_count or 0)
        if (v.last_synced_at and
            (bucket["synced_at"] is None or v.last_synced_at > bucket["synced_at"])):
            bucket["synced_at"] = v.last_synced_at
        # Top clip = highest-viewed video on this channel.
        if (bucket["top_clip"] is None
            or (v.view_count or 0) > (bucket["top_clip"]["views"] or 0)):
            bucket["top_clip"] = {
                "title":     (v.title or "")[:120],
                "thumb_url": v.thumbnail_url or "",
                "video_url": f"https://youtu.be/{v.video_id}" if v.video_id else "",
                "video_id":  v.video_id or "",
                "views":     int(v.view_count    or 0),
                "likes":     int(v.like_count    or 0),
                "comments":  int(v.comment_count or 0),
            }

    # ── Step 3: overlay Kaizer-only signals from ClipPerformance ──
    # Engagement on the full catalogue is already covered above; this
    # pass adds the SEO-score average (only meaningful for Kaizer-
    # published clips because YouTube doesn't expose our SEO score).
    perf_rows = latest_perf_per_upload(db, None)
    if user_id is not None and tokens:
        valid_pids = {t.channel_id for t in tokens if t.channel_id}
        perf_rows = [r for r in perf_rows if r.channel_id in valid_pids]

    perf_channel_ids = {r.channel_id for r in perf_rows if r.channel_id}
    perf_profiles = (db.query(models.Channel)
                       .filter(models.Channel.id.in_(perf_channel_ids))
                       .all()) if perf_channel_ids else []
    perf_profile_by_id = {c.id: c for c in perf_profiles}

    seo_sums: Dict[str, List[int]] = {}
    for r in perf_rows:
        profile = perf_profile_by_id.get(r.channel_id)
        tok = profile.oauth_token if profile else None
        if not tok or not tok.google_channel_id:
            continue
        gid = tok.google_channel_id
        bucket = groups.get(gid)
        if bucket is None:
            continue
        bucket["uploads_sampled"] += 1
        if r.channel_id and r.channel_id not in bucket["style_profile_ids"]:
            bucket["style_profile_ids"].append(r.channel_id)
        if (r.sampled_at and
            (bucket["last_sampled_at"] is None
             or r.sampled_at > bucket["last_sampled_at"])):
            bucket["last_sampled_at"] = r.sampled_at
        seo_sums.setdefault(gid, []).append(int(r.seo_score or 0))

    for gid, scores in seo_sums.items():
        if scores and gid in groups:
            groups[gid]["avg_seo_score"] = round(sum(scores) / len(scores), 1)

    # ── Step 4: ensure every seeded card has its full profile_ids list
    # (the dropdown filter falls back to this so style-profile-level
    # scoping still works for channels that haven't been sampled).
    for gid, bucket in groups.items():
        for pid in profile_ids_by_gcid.get(gid, []):
            if pid not in bucket["style_profile_ids"]:
                bucket["style_profile_ids"].append(pid)

    # Finalise — compute engagement rate, ISO timestamps, empty-state flag.
    out: List[Dict] = []
    for g in groups.values():
        views = max(g["total_views"], 1)
        g["engagement_rate"] = round(
            (g["total_likes"] + g["total_comments"]) / views * 100, 2
        )
        if g["synced_at"]:
            g["synced_at"] = g["synced_at"].isoformat()
        if g["last_sampled_at"]:
            g["last_sampled_at"] = g["last_sampled_at"].isoformat()
        # Empty-state signal so the frontend can render "Click Sync"
        # instead of showing fake-looking zero stats.
        g["needs_sync"] = (g["total_videos"] == 0)
        out.append(g)

    # Sort highest-traffic channel first — that's the one users care
    # about most on a glance.  Channels that need syncing fall to the
    # bottom so they don't dominate the layout.
    out.sort(
        key=lambda g: (not g["needs_sync"], g["total_views"]),
        reverse=True,
    )
    return out
