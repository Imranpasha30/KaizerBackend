"""Insights ingestion orchestrator — pull a channel's FULL upload history + enrich.

Composes the data-fetch layer into one ChannelSnapshot + its VideoMetric rows:

  Data API v3 (full history, NO cap)  → title/stats/duration/… per video
  + Analytics API (owner, if scope)   → retention, traffic source, subs, early velocity
  + Reporting API (owner, if scope)   → real thumbnail impressions + CTR
  + derive.py                         → reach-ratio, velocity, local timing, topic clusters
  → persist + 24h cache + quota log

The whole report spans the WHOLE history; only thumbnail CTR is bounded by the Reporting
API's 30-day backfill (each VideoMetric carries that data verbatim — the analyzer labels it).

``assemble_metrics`` (the merge+derive) is pure and unit-tested in
``scripts/test_insights_ingest.py``; ``run_ingestion`` is the live wrapper.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from insights import models as im
from insights import derive as D
from insights import maturity as M
from insights import analytics_api as AA
from insights import reporting_api as RA

log = logging.getLogger("kaizer.insights.ingest")

ANALYTICS_SCOPE = "https://www.googleapis.com/auth/yt-analytics.readonly"
PAGE = 50
CACHE_TTL_HOURS = 24

# Minimal country → IANA tz for publish-hour localization (top markets; UTC fallback).
_COUNTRY_TZ = {
    "IN": "Asia/Kolkata", "US": "America/New_York", "GB": "Europe/London",
    "PK": "Asia/Karachi", "BD": "Asia/Dhaka", "AE": "Asia/Dubai", "SG": "Asia/Singapore",
    "AU": "Australia/Sydney", "CA": "America/Toronto", "DE": "Europe/Berlin",
    "FR": "Europe/Paris", "BR": "America/Sao_Paulo", "NG": "Africa/Lagos",
    "ID": "Asia/Jakarta", "PH": "Asia/Manila", "JP": "Asia/Tokyo",
}


# ── tiny local parsers (keep the module self-contained) ──────────────────

def _parse_iso_duration(iso: str) -> int:
    if not iso or not iso.startswith("PT"):
        return 0
    total, num = 0, ""
    for ch in iso[2:]:
        if ch.isdigit():
            num += ch
        elif ch == "H":
            total += int(num or 0) * 3600; num = ""
        elif ch == "M":
            total += int(num or 0) * 60; num = ""
        elif ch == "S":
            total += int(num or 0); num = ""
    return total


def _parse_published(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return None


# ── pure assembly (merge fetched data + derive) — unit-tested ────────────

def assemble_metrics(videos: List[dict], *, core: dict, traffic: dict, early: dict,
                     ctr: dict, subscriber_count: int, tz_name: str,
                     now: Optional[datetime] = None) -> List[dict]:
    """Merge Data-API videos with Analytics/Reporting enrichment + derived signals into
    VideoMetric field dicts (one per video). Missing enrichment → those fields stay None
    (public mode). Pure: no I/O."""
    titles = [v.get("title", "") for v in videos]
    clusters = D.build_topic_clusters(titles)
    core = core or {}; traffic = traffic or {}; early = early or {}; ctr = ctr or {}
    rows: List[dict] = []
    for i, v in enumerate(videos):
        vid = v.get("video_id")
        if not vid:
            continue
        pub = _parse_published(v.get("published_at"))
        local_dt, dow, hour = D.publish_local(pub, tz_name)
        c, ew, ct = core.get(vid, {}), early.get(vid, {}), ctr.get(vid, {})
        first48 = ew.get("first_48h")
        if first48 is not None:
            vph = D.velocity_vph(first48, 48)
        else:
            vph = D.lifetime_vph(int(v.get("view_count") or 0), D.age_hours(pub, now=now) or 0)
        rows.append({
            "video_id": vid,
            "title": v.get("title", "") or "",
            "description_short": (v.get("description", "") or "")[:500],
            "tags": v.get("tags") or [],
            "published_at_utc": pub,
            "published_at_local": local_dt,
            "category_id": str(v.get("category_id", "") or ""),
            "thumbnail_url": v.get("thumbnail_url", "") or "",
            "duration_seconds": int(v.get("duration_seconds") or 0),
            "default_language": v.get("default_language", "") or "",
            "is_short": D.is_short(v.get("duration_seconds") or 0),
            "view_count": int(v.get("view_count") or 0),
            "like_count": int(v.get("like_count") or 0),
            "comment_count": int(v.get("comment_count") or 0),
            "impressions": ct.get("impressions"),
            "impressions_ctr": ct.get("ctr"),
            "avg_view_seconds": c.get("avg_view_seconds"),
            "avg_view_percentage": c.get("avg_view_pct"),
            "subscribers_gained": c.get("subs_gained"),
            "estimated_minutes_watched": c.get("minutes"),
            "views_first_24h": ew.get("first_24h"),
            "views_first_48h": ew.get("first_48h"),
            "views_first_7d": ew.get("first_7d"),
            "traffic_sources": traffic.get(vid),
            "views_per_hour_48h": vph,
            "reach_ratio": D.reach_ratio(int(v.get("view_count") or 0), subscriber_count),
            "publish_dow": dow,
            "publish_hour_local": hour,
            "topic_cluster": clusters[i] if i < len(clusters) else "general",
        })
    return rows


# ── live data-fetch helpers ──────────────────────────────────────────────

def _owner_channel(db, user_id: int, gcid: str):
    """The user's Channel linked (via OAuthToken) to this YouTube channel — for creds +
    ownership (user A can't ingest user B's channel)."""
    import models
    return (db.query(models.Channel)
            .join(models.OAuthToken, models.OAuthToken.channel_id == models.Channel.id)
            .filter(models.Channel.user_id == user_id,
                    models.OAuthToken.google_channel_id == gcid,
                    models.OAuthToken.refresh_token_enc != "")
            .first())


def _data_client(db, channel_id: int):
    from config import settings
    from googleapiclient.discovery import build
    if getattr(settings, "yt_data_api_key", None):
        return build("youtube", "v3", developerKey=settings.yt_data_api_key, cache_discovery=False)
    from youtube import oauth as yt_oauth
    return build("youtube", "v3", credentials=yt_oauth.get_credentials(db, channel_id),
                 cache_discovery=False)


def _fetch_channel(yt, gcid: str) -> dict:
    resp = yt.channels().list(part="snippet,statistics,contentDetails", id=gcid).execute()
    items = resp.get("items") or []
    if not items:
        raise RuntimeError(f"YouTube returned no channel for {gcid}")
    it = items[0]
    sn, st, cd = it.get("snippet", {}), it.get("statistics", {}), it.get("contentDetails", {})
    return {
        "title": sn.get("title", ""),
        "published_at": sn.get("publishedAt", ""),
        "subscriber_count": int(st.get("subscriberCount") or 0),
        "view_count": int(st.get("viewCount") or 0),
        "video_count": int(st.get("videoCount") or 0),
        "uploads": cd.get("relatedPlaylists", {}).get("uploads", ""),
    }


def _enumerate_video_ids(yt, uploads_pid: str, *, max_videos: Optional[int] = None) -> List[str]:
    """Full uploads playlist (NO cap by default — the report must span all history)."""
    ids, page = [], None
    while True:
        resp = yt.playlistItems().list(part="contentDetails", playlistId=uploads_pid,
                                       maxResults=PAGE, pageToken=page).execute()
        for it in (resp.get("items") or []):
            vid = (it.get("contentDetails") or {}).get("videoId")
            if vid:
                ids.append(vid)
        page = resp.get("nextPageToken")
        if not page or (max_videos and len(ids) >= max_videos):
            break
    return ids[:max_videos] if max_videos else ids


def _hydrate_videos(yt, ids: List[str]) -> List[dict]:
    out = []
    for i in range(0, len(ids), PAGE):
        batch = ids[i:i + PAGE]
        resp = yt.videos().list(part="snippet,statistics,contentDetails",
                                id=",".join(batch)).execute()
        for it in (resp.get("items") or []):
            sn, st, cd = it.get("snippet", {}), it.get("statistics", {}), it.get("contentDetails", {})
            thumbs = sn.get("thumbnails", {})
            thumb = (thumbs.get("maxres") or thumbs.get("high") or thumbs.get("medium")
                     or thumbs.get("default") or {}).get("url", "")
            out.append({
                "video_id": it.get("id"),
                "title": sn.get("title", ""),
                "description": sn.get("description", ""),
                "tags": sn.get("tags") or [],
                "published_at": sn.get("publishedAt", ""),
                "category_id": sn.get("categoryId", ""),
                "thumbnail_url": thumb,
                "duration_seconds": _parse_iso_duration(cd.get("duration", "")),
                "default_language": sn.get("defaultAudioLanguage") or sn.get("defaultLanguage") or "",
                "view_count": int(st.get("viewCount") or 0),
                "like_count": int(st.get("likeCount") or 0),
                "comment_count": int(st.get("commentCount") or 0),
            })
    return out


def _resolve_timezone(analytics_client) -> str:
    """Best-effort channel TZ from the Analytics top-country (YouTube has no TZ field).
    UTC on any shortfall."""
    if analytics_client is None:
        return "UTC"
    try:
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=365)
        resp = analytics_client.reports().query(
            ids="channel==MINE", startDate=start.isoformat(), endDate=end.isoformat(),
            metrics="views", dimensions="country", sort="-views", maxResults=1).execute()
        rows = AA._rows(resp)
        if rows:
            return _COUNTRY_TZ.get((rows[0].get("country") or "").upper(), "UTC")
    except Exception as exc:
        log.info("tz resolve failed: %s", str(exc)[:120])
    return "UTC"


def run_ingestion(db, user_id: int, google_channel_id: str, *, force: bool = False,
                  max_videos: Optional[int] = None,
                  max_early_window: Optional[int] = 300) -> im.ChannelSnapshot:
    """Pull + enrich + persist a ChannelSnapshot for a channel. Reuses a <24h snapshot
    unless ``force``. ``max_early_window`` caps the per-video early-velocity queries (the
    costliest Analytics calls) to the most-recent N videos; None = all."""
    gcid = google_channel_id

    if not force:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=CACHE_TTL_HOURS)
        recent = (db.query(im.ChannelSnapshot)
                  .filter(im.ChannelSnapshot.user_id == user_id,
                          im.ChannelSnapshot.google_channel_id == gcid,
                          im.ChannelSnapshot.status == "ok",
                          im.ChannelSnapshot.created_at >= cutoff)
                  .order_by(im.ChannelSnapshot.created_at.desc()).first())
        if recent:
            log.info("insights: reusing snapshot %s (<24h) for %s", recent.id, gcid)
            return recent

    owner = _owner_channel(db, user_id, gcid)
    if not owner:
        raise RuntimeError(f"User {user_id} doesn't own a profile linked to channel {gcid!r}.")

    import models
    token = (db.query(models.OAuthToken)
             .filter(models.OAuthToken.channel_id == owner.id).first())
    analytics_granted = bool(token and ANALYTICS_SCOPE in (getattr(token, "scopes", "") or ""))

    yt = _data_client(db, owner.id)
    ch = _fetch_channel(yt, gcid)
    if not ch["uploads"]:
        snap = im.ChannelSnapshot(user_id=user_id, channel_id=owner.id, google_channel_id=gcid,
                                  channel_title=ch["title"], access_mode="public",
                                  subscriber_count=ch["subscriber_count"], video_count=0,
                                  view_count=ch["view_count"], videos_ingested=0,
                                  status="ok", error="no uploads")
        db.add(snap); db.commit(); db.refresh(snap)
        return snap

    ids = _enumerate_video_ids(yt, ch["uploads"], max_videos=max_videos)
    videos = _hydrate_videos(yt, ids)
    n = len(videos)
    data_units = 1 + math.ceil(n / PAGE) + math.ceil(n / PAGE)  # channels + playlistItems + videos

    # Enrichment (owner + scope only).
    core = traffic = ctr = {}
    early: dict = {}
    tz_name = "UTC"
    analytics_calls = 0
    access_mode = "public"
    analytics_error = None          # captured reason when the Analytics API is unusable
    reporting_status = None         # reporting_ok | reporting_populating | reporting_unavailable
    if analytics_granted:
        access_mode = "full"
        end = datetime.now(timezone.utc).date().isoformat()
        ac = None
        try:
            ac = AA._client(db, owner.id)
        except Exception as exc:
            analytics_error = "analytics_unavailable"
            log.warning("insights: analytics client build failed for %s: %s", gcid, str(exc)[:160])
        # PROBE one cheap query so we surface the REAL reason (API disabled / forbidden) instead of
        # silently returning empty analytics — fetch_* swallow per-call errors and would otherwise
        # look like "no data" when the actual cause is the Analytics API being off in the GCP project.
        probe_vid = next((v["video_id"] for v in videos if v.get("video_id")), None)
        if ac is not None and probe_vid:
            try:
                ac.reports().query(ids="channel==MINE", startDate=end, endDate=end,
                                   metrics="views", dimensions="video",
                                   filters="video==" + probe_vid, maxResults=1).execute()
            except Exception as exc:
                msg = str(exc).lower()
                if "has not been used" in msg or "is disabled" in msg or "is not enabled" in msg:
                    analytics_error = "analytics_api_disabled"
                elif "permission" in msg or "insufficient" in msg or "forbidden" in msg or " 403" in msg:
                    analytics_error = "analytics_forbidden"
                else:
                    analytics_error = "analytics_unavailable"
                log.warning("insights: analytics probe failed for %s (%s): %s",
                            gcid, analytics_error, str(exc)[:200])
        if ac is not None and not analytics_error:
            try:
                tz_name = _resolve_timezone(ac); analytics_calls += 1
                # cover full history: oldest publish → today
                pubs = [p for p in (_parse_published(v.get("published_at")) for v in videos) if p]
                start = (min(pubs).date().isoformat() if pubs else
                         (datetime.now(timezone.utc) - timedelta(days=3650)).date().isoformat())
                vids = [v["video_id"] for v in videos if v.get("video_id")]
                core = AA.fetch_core(ac, vids, start=start, end=end); analytics_calls += math.ceil(n / AA._BATCH)
                traffic = AA.fetch_traffic(ac, vids, start=start, end=end); analytics_calls += math.ceil(n / AA._BATCH)
                # early-window per video (costliest) — most-recent N by publish date
                ew_videos = sorted([v for v in videos if v.get("published_at")],
                                   key=lambda v: v["published_at"], reverse=True)
                if max_early_window:
                    ew_videos = ew_videos[:max_early_window]
                for v in ew_videos:
                    early[v["video_id"]] = AA.fetch_early_window(ac, v["video_id"], v["published_at"])
                    analytics_calls += 1
            except Exception as exc:
                log.warning("insights analytics enrichment failed for %s: %s", gcid, str(exc)[:160])
            # Real thumbnail CTR (Reporting API v1, channel_reach_basic_a1; ~24–48h + 30-day backfill).
            # Create-if-missing so the clock starts at connect/first-run; null CTR until the first
            # report = "populating" (NOT 0, NOT an error).
            try:
                job_id = RA.ensure_job_for_channel(db, owner.id)
                if not job_id:
                    reporting_status = "reporting_unavailable"        # API off / scope missing
                else:
                    ctr = RA.fetch_thumbnail_ctr(db, owner.id) or {}
                    reporting_status = "reporting_ok" if ctr else "reporting_populating"
            except Exception as exc:
                reporting_status = "reporting_unavailable"
                log.info("insights reporting CTR unavailable for %s: %s", gcid, str(exc)[:120])

    age_days = None
    chpub = _parse_published(ch.get("published_at"))
    if chpub:
        age_days = max(0, (datetime.now(timezone.utc) - chpub).days)

    rows = assemble_metrics(videos, core=core, traffic=traffic, early=early, ctr=ctr,
                            subscriber_count=ch["subscriber_count"], tz_name=tz_name)

    snap = im.ChannelSnapshot(
        user_id=user_id, channel_id=owner.id, google_channel_id=gcid,
        channel_title=ch["title"], timezone=tz_name, access_mode=access_mode,
        subscriber_count=ch["subscriber_count"], video_count=ch["video_count"],
        view_count=ch["view_count"], videos_ingested=len(rows), channel_age_days=age_days,
        quota_predicted=data_units + analytics_calls, quota_actual=data_units + analytics_calls,
        status="ok",
        # The most relevant pipeline note for the report's CTR caveat (analytics issue wins;
        # else the reporting status when CTR isn't flowing yet).
        error=(analytics_error or (reporting_status if reporting_status not in (None, "reporting_ok") else None)),
    )
    db.add(snap); db.flush()
    for r in rows:
        db.add(im.VideoMetric(snapshot_id=snap.id, **r))
    db.commit(); db.refresh(snap)
    log.info("insights: snapshot %s for %s — %d videos, mode=%s, ~%d units",
             snap.id, gcid, len(rows), access_mode, data_units + analytics_calls)
    return snap


def classify_snapshot(snap: im.ChannelSnapshot) -> M.Maturity:
    """Maturity for a persisted snapshot (drives which analysis path runs)."""
    return M.classify(video_count=snap.video_count or 0,
                      channel_age_days=snap.channel_age_days or 0,
                      total_views=snap.view_count or 0,
                      analytics_granted=(snap.access_mode == "full"))
