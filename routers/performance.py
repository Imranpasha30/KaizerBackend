"""Performance router — feedback loop read API."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session

import auth
from database import get_db
import models
from analytics import calibrator, channel_catalog, poller


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


@router.get("/channels")
def channels_summary(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Per-YouTube-channel summary cards for the Performance header.

    Returns one card per real YT channel the requesting user has
    connected (every OAuthToken with a refresh token), with
    aggregates drawn from the ``channel_videos`` cache.  Channels
    that haven't been synced yet still appear (with
    ``needs_sync: true`` and zeroed stats) so the UI can prompt the
    user to hit "Sync from YouTube".
    """
    return calibrator.channel_summary(db, user_id=user.id)


# ─── Phase 2: full-channel video catalogue ────────────────────────────


def _assert_owns_channel(db: Session, user: models.User,
                          google_channel_id: str) -> None:
    """Reject if the user has no OAuth token for ``google_channel_id``."""
    owned = (
        db.query(models.OAuthToken)
          .join(models.Channel, models.Channel.id == models.OAuthToken.channel_id)
          .filter(
              models.Channel.user_id == user.id,
              models.OAuthToken.google_channel_id == google_channel_id,
          )
          .first()
    )
    if not owned:
        raise HTTPException(
            status_code=404,
            detail=f"You don't have a connected YouTube channel matching "
                   f"{google_channel_id!r}.",
        )


@router.post("/yt/{google_channel_id}/sync")
def sync_yt_channel(
    google_channel_id: str,
    background: BackgroundTasks,
    max_videos: int = Query(200, ge=1, le=500),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Fetch the channel's uploads playlist from YouTube and cache it.

    Runs synchronously so the UI can show "synced N videos" right after
    the click — at 200 videos this is still <10 seconds on the wire.
    For massively larger channels the caller can lower max_videos
    or we can promote this to a background task later.
    """
    _assert_owns_channel(db, user, google_channel_id)
    try:
        result = channel_catalog.sync_channel_videos(
            db, user.id, google_channel_id, max_videos=max_videos
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return result


@router.get("/yt/{google_channel_id}/videos")
def list_yt_videos(
    google_channel_id: str,
    limit: int = Query(50, ge=1, le=500),
    offset: int = 0,
    q: Optional[str] = None,
    order_by: str = Query("views", regex=r"^(views|likes|comments|published)$"),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Paginated list of cached videos for one connected YT channel.

    ``q`` is a case-insensitive substring match on the title; useful
    for the explorer dropdown.  ``order_by`` controls the sort axis —
    default is by view count (descending) which is what users almost
    always want when browsing their channel's catalogue.
    """
    _assert_owns_channel(db, user, google_channel_id)
    base = (
        db.query(models.ChannelVideo)
          .filter(
              models.ChannelVideo.user_id == user.id,
              models.ChannelVideo.google_channel_id == google_channel_id,
          )
    )
    if q:
        base = base.filter(models.ChannelVideo.title.ilike(f"%{q.strip()}%"))
    order_col = {
        "views":     models.ChannelVideo.view_count,
        "likes":     models.ChannelVideo.like_count,
        "comments":  models.ChannelVideo.comment_count,
        "published": models.ChannelVideo.published_at,
    }[order_by]
    rows = (base
            .order_by(order_col.desc())
            .offset(offset)
            .limit(limit)
            .all())
    total = base.count()
    return {
        "google_channel_id": google_channel_id,
        "total":             total,
        "limit":             limit,
        "offset":            offset,
        "videos": [
            {
                "video_id":         r.video_id,
                "video_url":        f"https://youtu.be/{r.video_id}",
                "title":            r.title or "",
                "thumbnail_url":    r.thumbnail_url or "",
                "published_at":     r.published_at.isoformat() if r.published_at else None,
                "duration_seconds": int(r.duration_seconds or 0),
                "view_count":       int(r.view_count    or 0),
                "like_count":       int(r.like_count    or 0),
                "comment_count":    int(r.comment_count or 0),
                "engagement_rate":  round(
                    ((r.like_count or 0) + (r.comment_count or 0))
                    / max(r.view_count or 1, 1) * 100, 2
                ),
                "last_synced_at":   r.last_synced_at.isoformat() if r.last_synced_at else None,
            }
            for r in rows
        ],
    }


@router.get("/yt/{google_channel_id}/percentiles")
def channel_percentiles(
    google_channel_id: str,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """View / like / comment percentile distribution for one channel.

    Drives the "where does this clip rank?" bars on the compare card.
    """
    _assert_owns_channel(db, user, google_channel_id)
    return channel_catalog.channel_percentiles(db, user.id, google_channel_id)


# ─── Phase 3: single-video comparison ─────────────────────────────────


@router.get("/compare/video/{video_id}")
def compare_one_video(
    video_id: str,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """One video's stats + its rank in its channel + 3 nearby peers.

    The video must be in the cache — caller is expected to sync the
    channel first.  Used by the "compare this video" card on the
    Performance page.
    """
    try:
        return channel_catalog.compare_video(db, user.id, video_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ─── Phase 3b: one video, ranked across every connected channel ───


@router.get("/compare/video/{video_id}/across-channels")
def compare_video_across_channels(
    video_id: str,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Rank one video within EVERY connected channel's distribution.

    Answers "if I'd posted this video on channel B, where would it
    sit?".  The target must already be in the cache (synced from its
    home channel).  Channels that haven't been synced themselves
    appear in the response with ``needs_sync: true`` so the UI can
    prompt for that action rather than pretend the video would rank
    at the top.
    """
    try:
        return channel_catalog.compare_video_across_channels(
            db, user.id, video_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ─── Phase 4: cross-channel comparison ────────────────────────────────


@router.get("/compare/channels")
def compare_all_channels(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Side-by-side per-channel rollup: median views, cadence, etc.

    Returns one row per connected YT channel the user has synced
    catalogue data for.  Cards / bars on the Performance page render
    this row-by-row so the user can compare their own channels.
    """
    return channel_catalog.compare_channels(db, user.id)


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
def trigger_poll(
    background: BackgroundTasks,
    channel_id: Optional[int] = None,
):
    """Force a stats poll now instead of waiting for the next cron tick.

    When ``channel_id`` is supplied, only uploads for that style profile
    are sampled — lets the admin refresh one channel's numbers in
    isolation instead of burning quota on every connected channel.
    Omit the param to keep the original "poll everything" behaviour.
    """
    background.add_task(poller.poll_once, channel_id=channel_id)
    return {"triggered": True, "channel_id": channel_id}
