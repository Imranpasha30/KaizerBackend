"""Trending-topic radar router — competitor CRUD + topic list + refresh."""
from __future__ import annotations

import re
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from config import settings
from database import get_db
import models
import auth
from trending import radar


router = APIRouter(prefix="/api/trending", tags=["trending"])


# ─── YouTube channel-ID resolution ─────────────────────────────────────────

def _resolve_channel_id(raw: str) -> tuple[str, str]:
    """Accept a UC-id, @handle, or full channel URL; return (channel_id, handle).

    Uses YouTube Data API v3 `channels.list?forHandle=...` (1 quota unit) to
    convert a handle to a UC id so users don't have to dig through YouTube
    Studio settings to find it.
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("YouTube channel identifier is empty")

    # Already a UC-prefixed channel ID — return verbatim
    if re.fullmatch(r"UC[A-Za-z0-9_-]{20,30}", raw):
        return raw, ""

    # Full URL — pull the path segment (handle or /channel/UC…)
    url_m = re.match(
        r"^(?:https?://)?(?:www\.|m\.)?youtube\.com/(@[\w\-.]+|channel/UC[A-Za-z0-9_-]{20,30})/?",
        raw, re.IGNORECASE,
    )
    if url_m:
        seg = url_m.group(1)
        if seg.startswith("channel/"):
            return seg.split("/", 1)[1], ""
        raw = seg  # fall through with just the @handle

    # Normalize handle form
    if not raw.startswith("@"):
        raw = "@" + raw
    handle = raw

    # Resolve via YouTube Data API
    api_key = settings.yt_data_api_key
    if not api_key:
        raise ValueError(
            "Set YOUTUBE_DATA_API_KEY in .env to resolve @handles, or paste "
            "the UC… channel ID directly."
        )

    import httpx
    try:
        r = httpx.get(
            "https://www.googleapis.com/youtube/v3/channels",
            params={
                "part":       "id,snippet",
                "forHandle":  handle,
                "key":        api_key,
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError as e:
        raise ValueError(f"YouTube API lookup failed: {e}") from e

    items = data.get("items") or []
    if not items:
        raise ValueError(
            f"No YouTube channel found for handle '{handle}'. "
            "Double-check the spelling or paste the UC id directly."
        )
    return items[0]["id"], handle


# ─── Competitors CRUD ─────────────────────────────────────────────────────

class CompetitorIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    handle: str = ""
    # Accepts a UC id, a @handle, or a full youtube.com URL.  Resolved in the
    # endpoint handler so the validator doesn't need to hit the network.
    youtube_channel_id: str = Field(..., min_length=2, max_length=200)
    language: str = "te"
    active: bool = True


def _comp_to_dict(c: models.CompetitorChannel) -> dict:
    return {
        "id": c.id,
        "name": c.name,
        "handle": c.handle or "",
        "youtube_channel_id": c.youtube_channel_id,
        "language": c.language or "te",
        "active": bool(c.active),
        "created_at": c.created_at.isoformat() if c.created_at else None,
    }


@router.get("/competitors")
def list_competitors(db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    rows = (
        db.query(models.CompetitorChannel)
          .filter(models.CompetitorChannel.user_id == user.id)
          .order_by(models.CompetitorChannel.name)
          .all()
    )
    return [_comp_to_dict(c) for c in rows]


@router.post("/competitors", status_code=201)
def create_competitor(payload: CompetitorIn, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    # Resolve whatever the user pasted (UC id, @handle, or full URL) → UC id
    try:
        channel_id, resolved_handle = _resolve_channel_id(payload.youtube_channel_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if db.query(models.CompetitorChannel).filter(
        models.CompetitorChannel.youtube_channel_id == channel_id,
        models.CompetitorChannel.user_id == user.id,
    ).first():
        raise HTTPException(status_code=409, detail="This YouTube channel is already tracked")

    data = payload.model_dump()
    data["youtube_channel_id"] = channel_id
    # Don't overwrite an explicit handle the user typed
    if not data.get("handle") and resolved_handle:
        data["handle"] = resolved_handle
    row = models.CompetitorChannel(user_id=user.id, **data)
    db.add(row); db.commit(); db.refresh(row)
    return _comp_to_dict(row)


@router.post("/resolve-channel")
def resolve_channel_endpoint(body: dict):
    """Standalone resolver — useful for a 'Lookup' button on the form."""
    raw = (body or {}).get("input", "")
    try:
        channel_id, handle = _resolve_channel_id(raw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"youtube_channel_id": channel_id, "handle": handle}


# ─── Channel discovery — browse by category/region ────────────────────────

@router.get("/yt-categories")
def list_yt_categories(region: str = Query("IN", min_length=2, max_length=3)):
    """YouTube video categories for a region — powers the category picker."""
    if not settings.yt_data_api_key:
        raise HTTPException(status_code=503, detail="YOUTUBE_DATA_API_KEY not set")
    import httpx
    try:
        r = httpx.get(
            "https://www.googleapis.com/youtube/v3/videoCategories",
            params={"part": "snippet", "regionCode": region, "key": settings.yt_data_api_key},
            timeout=10,
        )
        r.raise_for_status()
        items = (r.json().get("items") or [])
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"YouTube API: {e}")
    # Filter out `assignable=false` categories — those can't be used for video search
    out = []
    for it in items:
        sn = it.get("snippet") or {}
        if not sn.get("assignable", True):
            continue
        out.append({"id": it.get("id", ""), "title": sn.get("title", "")})
    out.sort(key=lambda x: x["title"])
    return {"region": region, "categories": out}


@router.get("/suggest-channels")
def suggest_channels(
    region: str = Query("IN", min_length=2, max_length=3),
    category: Optional[str] = Query(None, description="YouTube videoCategoryId"),
    limit: int = Query(15, ge=1, le=40),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Return the top channels currently trending in a region + category.

    Uses YouTube's mostPopular chart to find the top videos, then groups by
    channel so the user can add high-signal channels as competitors with one
    click.  Skips channels already tracked by this user.
    """
    if not settings.yt_data_api_key:
        raise HTTPException(status_code=503, detail="YOUTUBE_DATA_API_KEY not set")
    import httpx
    params = {
        "part":       "snippet,statistics",
        "chart":      "mostPopular",
        "regionCode": region,
        "maxResults": 50,  # pull enough to find unique channels
        "key":        settings.yt_data_api_key,
    }
    if category:
        params["videoCategoryId"] = category
    try:
        r = httpx.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params=params, timeout=12,
        )
        r.raise_for_status()
        items = r.json().get("items") or []
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"YouTube API: {e}")

    # Group by channelId — keep first (most popular) hit per channel
    seen_ch = {}
    for it in items:
        sn = it.get("snippet") or {}
        ch_id = sn.get("channelId")
        if not ch_id or ch_id in seen_ch:
            continue
        seen_ch[ch_id] = {
            "youtube_channel_id":    ch_id,
            "name":                  sn.get("channelTitle", ""),
            "handle":                "",                    # fetched below
            "sample_video_title":    sn.get("title", ""),
            "sample_view_count":     int((it.get("statistics") or {}).get("viewCount") or 0),
            "published_at":          sn.get("publishedAt", ""),
        }

    # Already-tracked channels for this user — exclude from suggestions
    existing = {
        c.youtube_channel_id
        for c in db.query(models.CompetitorChannel)
                   .filter(models.CompetitorChannel.user_id == user.id)
                   .all()
    }
    results = [v for k, v in seen_ch.items() if k not in existing]

    # Enrich with subscriber count + handle via channels.list (batch of 50)
    ch_ids = [c["youtube_channel_id"] for c in results[:limit]]
    if ch_ids:
        try:
            r2 = httpx.get(
                "https://www.googleapis.com/youtube/v3/channels",
                params={
                    "part": "snippet,statistics",
                    "id":   ",".join(ch_ids),
                    "key":  settings.yt_data_api_key,
                },
                timeout=12,
            )
            r2.raise_for_status()
            by_id = {it["id"]: it for it in (r2.json().get("items") or [])}
        except httpx.HTTPError:
            by_id = {}
        for c in results[:limit]:
            meta = by_id.get(c["youtube_channel_id"]) or {}
            sn = meta.get("snippet") or {}
            st = meta.get("statistics") or {}
            c["handle"]         = sn.get("customUrl") or ""
            c["subscribers"]    = int(st.get("subscriberCount") or 0)
            c["total_videos"]   = int(st.get("videoCount") or 0)
            c["description"]    = (sn.get("description") or "")[:200]
            c["country"]        = sn.get("country") or ""
            c["thumbnail_url"]  = (
                (sn.get("thumbnails") or {}).get("default", {}).get("url", "")
            )

    # Sort by sample view count as a proxy for "trending-ness"
    results.sort(key=lambda c: c.get("sample_view_count", 0), reverse=True)
    return {
        "region":   region,
        "category": category or "",
        "results":  results[:limit],
    }


@router.get("/suggest-from-profiles")
def suggest_from_profiles(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Quick-add source: the user's own style profiles whose `handle` looks
    like a real YouTube handle AND that aren't already tracked as competitors.

    The 13 seeded profiles (TV9 Telugu, RTV, …) each ship with a handle, so
    this lets the user bulk-add them in one click.
    """
    profiles = (
        db.query(models.Channel)
          .filter(models.Channel.user_id == user.id)
          .all()
    )
    tracked_handles = set()
    tracked_ids     = set()
    for c in (
        db.query(models.CompetitorChannel)
          .filter(models.CompetitorChannel.user_id == user.id)
          .all()
    ):
        tracked_handles.add((c.handle or "").lower().lstrip("@"))
        tracked_ids.add(c.youtube_channel_id)

    out = []
    for p in profiles:
        handle = (p.handle or "").strip()
        if not handle or not handle.startswith("@"):
            continue
        if handle.lower().lstrip("@") in tracked_handles:
            continue
        out.append({
            "name":     p.name,
            "handle":   handle,
            "language": p.language or "te",
            # youtube_channel_id unknown here — resolved when the user clicks add
        })
    return {"results": out}


@router.delete("/competitors/{competitor_id}")
def delete_competitor(competitor_id: int, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    row = db.query(models.CompetitorChannel).filter(
        models.CompetitorChannel.id == competitor_id,
        models.CompetitorChannel.user_id == user.id,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Competitor not found")
    db.delete(row); db.commit()
    return {"deleted": competitor_id}


# ─── Topics ───────────────────────────────────────────────────────────────

@router.get("/topics")
def list_topics(
    urgency: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    unused_only: bool = False,
    since_hours: Optional[int] = Query(
        None, ge=1, le=24 * 365,
        description="Restrict to topics published in the last N hours.",
    ),
    since: Optional[str] = Query(
        None,
        description="ISO-8601 lower bound on published_at (takes precedence over since_hours).",
    ),
    until: Optional[str] = Query(
        None,
        description="ISO-8601 upper bound on published_at.",
    ),
    date_field: str = Query(
        "published_at",
        pattern="^(published_at|fetched_at)$",
        description="Which column the date filters apply to.",
    ),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    from datetime import datetime, timezone, timedelta
    # Scope to topics whose source competitor belongs to this user
    q = (
        db.query(models.TrendingTopic)
          .join(models.CompetitorChannel,
                models.CompetitorChannel.id == models.TrendingTopic.source_channel_id)
          .filter(models.CompetitorChannel.user_id == user.id)
    )
    if urgency:
        q = q.filter(models.TrendingTopic.urgency == urgency)
    if unused_only:
        q = q.filter(models.TrendingTopic.used_for_job_id.is_(None))

    col = (models.TrendingTopic.published_at
           if date_field == "published_at"
           else models.TrendingTopic.fetched_at)

    # Parse explicit ISO strings first (custom range). Silently tolerant of the
    # trailing-Z form the browser produces for datetime-local inputs.
    def _parse(s: str | None):
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None

    since_dt = _parse(since)
    until_dt = _parse(until)
    if since_dt is None and since_hours:
        since_dt = datetime.now(timezone.utc) - timedelta(hours=since_hours)

    # SQLite stores naive timestamps; strip tzinfo so comparisons don't crash
    if since_dt and since_dt.tzinfo:
        since_dt = since_dt.astimezone(timezone.utc).replace(tzinfo=None)
    if until_dt and until_dt.tzinfo:
        until_dt = until_dt.astimezone(timezone.utc).replace(tzinfo=None)

    if since_dt:
        q = q.filter(col >= since_dt)
    if until_dt:
        q = q.filter(col <= until_dt)

    rows = q.order_by(col.desc().nullslast()).limit(limit).all()

    source_map = {c.id: c for c in db.query(models.CompetitorChannel).all()}
    return [{
        "id":            r.id,
        "source":        (source_map.get(r.source_channel_id).name
                          if source_map.get(r.source_channel_id) else ""),
        "video_id":      r.video_id,
        "video_title":   r.video_title,
        "video_url":     r.video_url,
        "published_at":  r.published_at.isoformat() if r.published_at else None,
        "view_count":    r.view_count,
        "topic_summary": r.topic_summary,
        "keywords":      list(r.keywords or []),
        "urgency":       r.urgency,
        "used_for_job_id": r.used_for_job_id,
        "fetched_at":    r.fetched_at.isoformat() if r.fetched_at else None,
    } for r in rows]


@router.post("/refresh")
def refresh_now(
    background: BackgroundTasks,
    since_hours: Optional[int] = Query(
        None, ge=1, le=24 * 365,
        description="Only pull + summarize videos published within the last N hours.",
    ),
):
    """Force a radar sweep now instead of waiting for the scheduler.

    When `since_hours` is passed the sweep skips any competitor video older
    than that window — saves Gemini summarizer quota and keeps the DB focused.
    """
    background.add_task(radar.refresh_all, since_hours=since_hours)
    return {"triggered": True, "window_hours": since_hours or 0}


@router.post("/topics/{topic_id}/use")
def mark_used(topic_id: int, job_id: int, db: Session = Depends(get_db)):
    """Mark that a pipeline job was started from this topic (for bookkeeping)."""
    row = db.query(models.TrendingTopic).filter(models.TrendingTopic.id == topic_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Topic not found")
    row.used_for_job_id = job_id
    db.commit()
    return {"topic_id": topic_id, "used_for_job_id": job_id}


@router.delete("/topics/{topic_id}")
def delete_topic(topic_id: int, db: Session = Depends(get_db)):
    """Remove a single topic from the radar feed."""
    row = db.query(models.TrendingTopic).filter(models.TrendingTopic.id == topic_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Topic not found")
    db.delete(row); db.commit()
    return {"deleted": topic_id}
