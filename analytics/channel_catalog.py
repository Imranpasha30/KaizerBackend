"""Full-channel video catalogue sync + analytics primitives.

`analytics/poller.py` only knows about uploads Kaizer published.  This
module knows about every video on the user's connected YouTube channels
— pulled from the YouTube Data API uploads playlist, cached in the
``channel_videos`` table, and used to compute:

  * Percentiles for one channel (median, p75, p95 of view_count etc.)
    so a single video's stats can be rendered with rank context.
  * Cross-channel comparisons (median views per upload, engagement,
    posting cadence) for the Performance page's compare view.
  * The "pick any of my videos" dropdown — the dashboard explorer.

Cost model:
  * channels.list .................. 1 quota unit
  * playlistItems.list (50 per page) 1 unit per page
  * videos.list      (50 per page)   1 unit per page
  So a 200-video channel costs ~5 quota units to fully sync once.

The sync is idempotent: re-running upserts rows by
(user_id, google_channel_id, video_id).
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sqlalchemy.orm import Session

import models
from config import settings
from youtube import oauth as yt_oauth

logger = logging.getLogger("kaizer.analytics.channel_catalog")


PAGE_SIZE      = 50
DEFAULT_LIMIT  = 200      # cap a single sync to keep quota predictable


# ─── YouTube Data API helpers ─────────────────────────────────────────


def _yt_client_for(db: Session, channel_id: int):
    """Build a youtube-v3 client. Prefer the public API key (cheap, no
    OAuth burn); fall back to the channel's OAuth credentials so the
    sync works even if the admin hasn't set ``YOUTUBE_DATA_API_KEY``."""
    if settings.yt_data_api_key:
        return build("youtube", "v3", developerKey=settings.yt_data_api_key,
                     cache_discovery=False)
    creds = yt_oauth.get_credentials(db, channel_id)
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def _resolve_uploads_playlist(yt, google_channel_id: str) -> str:
    """``UC<channel_id>`` → uploads playlist id (usually ``UU<channel_id>``).
    Doing this lookup once instead of guessing avoids breakage if YouTube
    changes the convention.
    """
    resp = yt.channels().list(part="contentDetails",
                              id=google_channel_id).execute()
    items = resp.get("items") or []
    if not items:
        raise RuntimeError(f"YouTube returned no channel for id {google_channel_id}")
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def _parse_iso_duration(iso: str) -> int:
    """ISO-8601 PT#H#M#S → total seconds. Tiny inline parser — pulling
    isodate just for this would be overkill."""
    if not iso or not iso.startswith("PT"):
        return 0
    s = iso[2:]
    total = 0
    num = ""
    for ch in s:
        if ch.isdigit():
            num += ch
        elif ch == "H":
            total += int(num or 0) * 3600; num = ""
        elif ch == "M":
            total += int(num or 0) * 60;   num = ""
        elif ch == "S":
            total += int(num or 0);        num = ""
    return total


def _parse_published(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    try:
        # YT returns "2024-01-15T08:30:00Z"
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


# ─── Sync ─────────────────────────────────────────────────────────────


def sync_channel_videos(
    db: Session,
    user_id: int,
    google_channel_id: str,
    *,
    max_videos: int = DEFAULT_LIMIT,
) -> Dict:
    """Fetch the channel's uploads playlist and upsert into channel_videos.

    Picks the requesting user's first style profile that's connected
    to this YT account for the API client's credentials (when no shared
    API key is configured).  Returns a status dict the router can
    surface to the UI.
    """
    # Find one of THIS user's Channels whose oauth_token points at the
    # target YT channel — gives us auth for the API client + ownership
    # check so user A can't sync user B's connected channel.
    owner_channel = (
        db.query(models.Channel)
          .join(models.OAuthToken, models.OAuthToken.channel_id == models.Channel.id)
          .filter(
              models.Channel.user_id == user_id,
              models.OAuthToken.google_channel_id == google_channel_id,
              models.OAuthToken.refresh_token_enc != "",
          )
          .first()
    )
    if not owner_channel:
        raise RuntimeError(
            f"User {user_id} doesn't own a profile linked to YT channel "
            f"{google_channel_id!r}."
        )

    yt = _yt_client_for(db, owner_channel.id)

    try:
        uploads_pid = _resolve_uploads_playlist(yt, google_channel_id)
    except HttpError as exc:
        raise RuntimeError(f"YouTube channels.list failed: {exc}") from exc

    seen_video_ids: List[str] = []
    page_token: Optional[str] = None
    while len(seen_video_ids) < max_videos:
        try:
            pl = yt.playlistItems().list(
                part="contentDetails", playlistId=uploads_pid,
                maxResults=PAGE_SIZE, pageToken=page_token or None,
            ).execute()
        except HttpError as exc:
            raise RuntimeError(f"YouTube playlistItems.list failed: {exc}") from exc

        batch = [
            it["contentDetails"]["videoId"]
            for it in (pl.get("items") or [])
            if it.get("contentDetails", {}).get("videoId")
        ]
        seen_video_ids.extend(batch)
        page_token = pl.get("nextPageToken")
        if not page_token:
            break
    seen_video_ids = seen_video_ids[:max_videos]

    if not seen_video_ids:
        return {
            "google_channel_id": google_channel_id,
            "synced":            0,
            "new":               0,
            "updated":           0,
            "note":              "channel has no uploads",
        }

    # Walk videos.list in batches of 50 to get stats + duration + thumb.
    rows_by_id: Dict[str, Dict] = {}
    for i in range(0, len(seen_video_ids), PAGE_SIZE):
        chunk = seen_video_ids[i:i + PAGE_SIZE]
        try:
            v = yt.videos().list(
                part="snippet,statistics,contentDetails",
                id=",".join(chunk),
                maxResults=PAGE_SIZE,
            ).execute()
        except HttpError as exc:
            raise RuntimeError(f"YouTube videos.list failed: {exc}") from exc
        for item in (v.get("items") or []):
            vid = item.get("id") or ""
            if not vid:
                continue
            sn = item.get("snippet") or {}
            st = item.get("statistics") or {}
            cd = item.get("contentDetails") or {}
            thumbs = sn.get("thumbnails") or {}
            # Prefer medium/default — high is overkill for the picker.
            thumb_url = (
                (thumbs.get("medium") or {}).get("url")
                or (thumbs.get("default") or {}).get("url")
                or ""
            )
            rows_by_id[vid] = {
                "video_id":          vid,
                "title":             (sn.get("title") or "")[:500],
                "description_short": (sn.get("description") or "")[:500],
                "published_at":      _parse_published(sn.get("publishedAt") or ""),
                "duration_seconds":  _parse_iso_duration(cd.get("duration") or ""),
                "view_count":        int(st.get("viewCount") or 0),
                "like_count":        int(st.get("likeCount") or 0),
                "comment_count":     int(st.get("commentCount") or 0),
                "thumbnail_url":     thumb_url,
            }

    # Upsert one at a time — keeps the logic dialect-agnostic. Volumes
    # are small (~200 rows per sync), so the loop cost is negligible.
    new_count = 0
    updated   = 0
    now = datetime.now(timezone.utc)
    for vid, fields in rows_by_id.items():
        existing = (
            db.query(models.ChannelVideo)
              .filter(
                  models.ChannelVideo.user_id == user_id,
                  models.ChannelVideo.google_channel_id == google_channel_id,
                  models.ChannelVideo.video_id == vid,
              )
              .first()
        )
        if existing is None:
            row = models.ChannelVideo(
                user_id=user_id,
                google_channel_id=google_channel_id,
                **fields,
                last_synced_at=now,
            )
            db.add(row)
            new_count += 1
        else:
            for k, val in fields.items():
                setattr(existing, k, val)
            existing.last_synced_at = now
            updated += 1
    db.commit()

    return {
        "google_channel_id": google_channel_id,
        "synced":            new_count + updated,
        "new":               new_count,
        "updated":           updated,
    }


# ─── Analytics over the catalogue ──────────────────────────────────────


def _percentile(values: List[int], p: float) -> int:
    """Nearest-rank percentile.  ``p`` in [0, 1].  Returns 0 on empty
    input — saves the caller from a guard."""
    if not values:
        return 0
    s = sorted(values)
    k = max(0, min(len(s) - 1, math.ceil(p * len(s)) - 1))
    return int(s[k])


def channel_percentiles(db: Session, user_id: int, google_channel_id: str) -> Dict:
    """Per-channel distribution stats over the cached catalogue.

    Returns p25/p50/p75/p95 + max for view, like, and comment counts,
    plus the total upload count and engagement rate.  The Performance
    page renders a single video against these bars so the user knows
    "this clip is in the top 10%" at a glance.
    """
    rows = (
        db.query(models.ChannelVideo)
          .filter(
              models.ChannelVideo.user_id == user_id,
              models.ChannelVideo.google_channel_id == google_channel_id,
          )
          .all()
    )
    views    = [r.view_count    or 0 for r in rows]
    likes    = [r.like_count    or 0 for r in rows]
    comments = [r.comment_count or 0 for r in rows]

    total_views    = sum(views)
    total_likes    = sum(likes)
    total_comments = sum(comments)

    return {
        "google_channel_id": google_channel_id,
        "total_videos":      len(rows),
        "total_views":       total_views,
        "total_likes":       total_likes,
        "total_comments":    total_comments,
        "engagement_rate":   round(
            (total_likes + total_comments) / max(total_views, 1) * 100, 2
        ),
        "views":    {
            "p25": _percentile(views, 0.25),
            "p50": _percentile(views, 0.50),
            "p75": _percentile(views, 0.75),
            "p95": _percentile(views, 0.95),
            "max": max(views or [0]),
        },
        "likes":    {
            "p25": _percentile(likes, 0.25),
            "p50": _percentile(likes, 0.50),
            "p75": _percentile(likes, 0.75),
            "p95": _percentile(likes, 0.95),
            "max": max(likes or [0]),
        },
        "comments": {
            "p25": _percentile(comments, 0.25),
            "p50": _percentile(comments, 0.50),
            "p75": _percentile(comments, 0.75),
            "p95": _percentile(comments, 0.95),
            "max": max(comments or [0]),
        },
    }


def _rank(value: int, sorted_values: List[int]) -> float:
    """Return percentile rank (0..1) of ``value`` within sorted ``sorted_values``."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    # Count values strictly less + half the ties — standard percentile rank.
    less = sum(1 for v in sorted_values if v < value)
    equal = sum(1 for v in sorted_values if v == value)
    return round((less + equal / 2) / n, 3)


def compare_video(db: Session, user_id: int, video_id: str) -> Dict:
    """Return one video's stats + its rank in its channel + 3 peers.

    Peer selection: same channel, published within ±21 days of the
    target video, closest by published date.  Lets the user say
    "did this video over- or under-perform compared to what I posted
    around the same time?" — which is more useful than ranking against
    a video published 4 years ago.
    """
    target = (
        db.query(models.ChannelVideo)
          .filter(
              models.ChannelVideo.user_id == user_id,
              models.ChannelVideo.video_id == video_id,
          )
          .first()
    )
    if not target:
        raise RuntimeError(
            f"Video {video_id!r} not in cache — sync the channel first."
        )

    channel_rows = (
        db.query(models.ChannelVideo)
          .filter(
              models.ChannelVideo.user_id == user_id,
              models.ChannelVideo.google_channel_id == target.google_channel_id,
          )
          .all()
    )

    views_sorted    = sorted(r.view_count    or 0 for r in channel_rows)
    likes_sorted    = sorted(r.like_count    or 0 for r in channel_rows)
    comments_sorted = sorted(r.comment_count or 0 for r in channel_rows)

    # Pick peers: same channel, ±21 days, exclude target.
    peers: List[Tuple[int, models.ChannelVideo]] = []
    if target.published_at:
        for r in channel_rows:
            if r.id == target.id or not r.published_at:
                continue
            diff_days = abs((r.published_at - target.published_at).days)
            if diff_days <= 21:
                peers.append((diff_days, r))
    peers.sort(key=lambda t: t[0])
    peers = peers[:3]

    def _row_dict(r: models.ChannelVideo) -> Dict:
        return {
            "video_id":      r.video_id,
            "title":         r.title or "",
            "thumbnail_url": r.thumbnail_url or "",
            "published_at":  r.published_at.isoformat() if r.published_at else None,
            "view_count":    int(r.view_count    or 0),
            "like_count":    int(r.like_count    or 0),
            "comment_count": int(r.comment_count or 0),
            "duration_seconds": int(r.duration_seconds or 0),
        }

    return {
        "video": _row_dict(target),
        "channel": {
            "google_channel_id": target.google_channel_id,
            "total_videos":      len(channel_rows),
        },
        "ranks": {
            "views":    _rank(target.view_count    or 0, views_sorted),
            "likes":    _rank(target.like_count    or 0, likes_sorted),
            "comments": _rank(target.comment_count or 0, comments_sorted),
        },
        "channel_stats": channel_percentiles(db, user_id, target.google_channel_id),
        "peers": [_row_dict(p) for _, p in peers],
    }


def compare_video_across_channels(
    db: Session, user_id: int, video_id: str,
) -> Dict:
    """Where would this video rank in EVERY connected channel?

    Takes one video (must already be in the cache) and computes its
    percentile rank within each of the user's other connected channels'
    distributions.  Answers the question "if I'd posted this on
    channel B instead, where would it fall?" — useful when deciding
    whether a hit on one channel would have travelled.

    Returns one row per connected channel: its title + the video's
    rank within that channel's view / like / comment distributions
    plus the channel's median for context.
    """
    target = (
        db.query(models.ChannelVideo)
          .filter(
              models.ChannelVideo.user_id == user_id,
              models.ChannelVideo.video_id == video_id,
          )
          .first()
    )
    if not target:
        raise RuntimeError(
            f"Video {video_id!r} not in cache — sync the channel first."
        )

    # Every YT account this user owns (so we can label rows with the
    # channel title even when no catalogue rows are cached yet).
    tokens = (
        db.query(models.OAuthToken)
          .join(models.Channel, models.Channel.id == models.OAuthToken.channel_id)
          .filter(
              models.Channel.user_id == user_id,
              models.OAuthToken.google_channel_id != "",
              models.OAuthToken.refresh_token_enc != "",
          )
          .all()
    )

    # Bucket the user's catalogue by google_channel_id once so each
    # rank lookup is O(n) on its own channel, not O(N total).
    rows = (
        db.query(models.ChannelVideo)
          .filter(models.ChannelVideo.user_id == user_id)
          .all()
    )
    by_gcid: Dict[str, List[models.ChannelVideo]] = {}
    for r in rows:
        by_gcid.setdefault(r.google_channel_id, []).append(r)

    tv  = int(target.view_count    or 0)
    tl  = int(target.like_count    or 0)
    tc  = int(target.comment_count or 0)

    out: List[Dict] = []
    for tok in tokens:
        gid = tok.google_channel_id
        ch_rows = by_gcid.get(gid, [])
        # Honour "this is also the target's home channel" — keep it
        # in the list with ``is_home: True`` so the UI can label it
        # ("this video is on this channel") instead of pretending it
        # would rank somewhere else.
        is_home = (gid == target.google_channel_id)

        views    = sorted(r.view_count    or 0 for r in ch_rows)
        likes    = sorted(r.like_count    or 0 for r in ch_rows)
        comments = sorted(r.comment_count or 0 for r in ch_rows)

        out.append({
            "google_channel_id":     gid,
            "youtube_channel_title": tok.google_channel_title or "",
            "channel_thumbnail_url": tok.channel_thumbnail_url or "",
            "subscriber_count":      int(tok.subscriber_count or 0),
            "total_videos":          len(ch_rows),
            "is_home":               is_home,
            # null ranks when the channel hasn't been synced yet — the
            # UI shows a "Sync to compare" prompt for those rows
            # instead of pretending the video ranks at the top.
            "needs_sync":            not bool(ch_rows),
            "ranks": (
                {
                    "views":    _rank(tv, views),
                    "likes":    _rank(tl, likes),
                    "comments": _rank(tc, comments),
                } if ch_rows else
                {"views": None, "likes": None, "comments": None}
            ),
            "channel_medians": (
                {
                    "views":    _percentile(views,    0.50),
                    "likes":    _percentile(likes,    0.50),
                    "comments": _percentile(comments, 0.50),
                } if ch_rows else
                {"views": 0, "likes": 0, "comments": 0}
            ),
        })

    # Home channel first so the user sees ground truth, then the
    # most-trafficked channels for context.
    out.sort(
        key=lambda r: (not r["is_home"], -(r["channel_medians"]["views"] or 0))
    )

    return {
        "video": {
            "video_id":      target.video_id,
            "title":         target.title or "",
            "thumbnail_url": target.thumbnail_url or "",
            "view_count":    tv,
            "like_count":    tl,
            "comment_count": tc,
            "home_google_channel_id": target.google_channel_id,
        },
        "channels": out,
    }


def compare_channels(db: Session, user_id: int) -> List[Dict]:
    """Side-by-side comparison of EVERY connected YT channel.

    Returns one row per channel: total uploads, median views per
    upload, top view count, engagement rate, and posting cadence
    (uploads per week over the last 90 days).  The Performance page
    renders these as bars so the user can see "channel A posts twice
    as often but channel B is converting better."
    """
    rows = (
        db.query(models.ChannelVideo)
          .filter(models.ChannelVideo.user_id == user_id)
          .all()
    )

    # Bucket per google_channel_id
    by_gcid: Dict[str, List[models.ChannelVideo]] = {}
    for r in rows:
        by_gcid.setdefault(r.google_channel_id, []).append(r)

    # Look up YT channel titles via OAuthToken for display labels.
    gcids = list(by_gcid.keys())
    tokens = (
        db.query(models.OAuthToken)
          .join(models.Channel, models.Channel.id == models.OAuthToken.channel_id)
          .filter(
              models.Channel.user_id == user_id,
              models.OAuthToken.google_channel_id.in_(gcids),
          )
          .all()
    ) if gcids else []
    tok_by_gcid = {t.google_channel_id: t for t in tokens}

    out: List[Dict] = []
    now = datetime.now(timezone.utc)
    cutoff_90 = now - timedelta(days=90)

    for gcid, channel_rows in by_gcid.items():
        views    = [r.view_count    or 0 for r in channel_rows]
        likes    = [r.like_count    or 0 for r in channel_rows]
        comments = [r.comment_count or 0 for r in channel_rows]
        # Uploads in the trailing 90 days for cadence.
        recent = [r for r in channel_rows
                  if r.published_at and r.published_at >= cutoff_90]
        weeks  = 90 / 7.0
        cadence_per_week = round(len(recent) / weeks, 2) if weeks else 0.0

        tok = tok_by_gcid.get(gcid)
        total_views = sum(views)
        total_likes = sum(likes)
        total_comments = sum(comments)
        out.append({
            "google_channel_id":     gcid,
            "youtube_channel_title": (tok.google_channel_title if tok else ""),
            "channel_thumbnail_url": (tok.channel_thumbnail_url if tok else ""),
            "subscriber_count":      int(tok.subscriber_count if tok else 0),
            "total_videos":          len(channel_rows),
            "uploads_last_90d":      len(recent),
            "cadence_per_week":      cadence_per_week,
            "median_views":          _percentile(views, 0.50),
            "p95_views":             _percentile(views, 0.95),
            "max_views":             max(views or [0]),
            "total_views":           total_views,
            "engagement_rate":       round(
                (total_likes + total_comments) / max(total_views, 1) * 100, 2
            ),
        })

    out.sort(key=lambda c: c["total_views"], reverse=True)
    return out
