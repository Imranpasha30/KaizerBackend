"""YouTube Analytics API v2 enrichment — retention, traffic source, subscribers, early velocity.

The owner-only "deep" signals that the Data API can't give. Uses ``reports.query`` (the
proven pattern from ``analytics/ctr.py``) but with the metrics/dimensions VERIFIED in
DISCOVERY.md — NOT thumbnail impressions/CTR (those live in the Reporting API; see
``reporting_api.py``). Every fetch degrades to ``{}`` on missing scope / error so the
analyzer falls back gracefully.

Pure parsing (``_rows``, ``normalize_traffic``, ``aggregate_traffic``, ``sum_early_window``)
is unit-tested in ``scripts/test_insights_reporting.py``'s sibling cases; the I/O wrappers
mirror ``ctr.py``.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import List, Optional

log = logging.getLogger("kaizer.insights.analytics")

ANALYTICS_SCOPE = "https://www.googleapis.com/auth/yt-analytics.readonly"
_BATCH = 200   # video ids per query (matches ctr.py)

# Core per-video metrics available via reports.query (verified — DISCOVERY.md §4).
CORE_METRICS = "views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,subscribersGained"

# insightTrafficSourceType enum → our normalized bucket. The operator cares specifically
# about "trapped in the SUBSCRIBER feed vs reached discovery", so those stay distinct.
# VERIFIED against the live API (the real insightTrafficSourceType enum values) — the docs/
# Studio labels ("Browse features", "Suggested videos") are NOT the API enum names. There is
# NO "BROWSE_FEATURES" enum; the home/other-pages bucket is YT_OTHER_PAGE. Shorts-feed traffic
# (SHORTS/SOUND_PAGE/VIDEO_REMIXES/SHORTS_CONTENT_LINKS) is its own discovery engine.
TRAFFIC_MAP = {
    "YT_SEARCH": "search",
    "SUBSCRIBER": "subscriber",
    "RELATED_VIDEO": "suggested",
    "END_SCREEN": "suggested",
    "SHORTS": "shorts",
    "SOUND_PAGE": "shorts",
    "VIDEO_REMIXES": "shorts",
    "SHORTS_CONTENT_LINKS": "shorts",
    "NOTIFICATION": "notifications",
    "EXT_URL": "external",
    "NO_LINK_OTHER": "external",
    "NO_LINK_EMBEDDED": "external",
    "YT_CHANNEL": "channel",
    "YT_OTHER_PAGE": "browse",          # closest API source to home / "Browse features"
    "PLAYLIST": "playlist",
    "YT_PLAYLIST_PAGE": "playlist",
    "HASHTAGS": "other",
    "IMMERSIVE_LIVE": "other",
    "ADVERTISING": "other",
    "ANNOTATION": "other",
    "PROMOTED": "other",
    "CAMPAIGN_CARD": "other",
}
TRAFFIC_BUCKETS = ("browse", "suggested", "search", "subscriber", "shorts",
                   "external", "notifications", "channel", "playlist", "other")
# Discovery = every source that reaches NEW (non-subscriber) viewers. Used to detect videos
# "trapped" in the subscriber feed regardless of which discovery engine a channel relies on.
DISCOVERY_BUCKETS = ("shorts", "suggested", "search", "browse", "external", "channel", "playlist")


def normalize_traffic(enum: str) -> str:
    return TRAFFIC_MAP.get((enum or "").strip().upper(), "other")


def _rows(resp: dict) -> List[dict]:
    """Turn a reports.query response (columnHeaders + rows) into a list of dicts."""
    headers = [h.get("name") for h in (resp.get("columnHeaders") or [])]
    out = []
    for row in (resp.get("rows") or []):
        out.append(dict(zip(headers, row)))
    return out


def aggregate_traffic(rows: List[dict]) -> dict:
    """Pure. rows = dicts with keys video / insightTrafficSourceType / views →
    ``{video_id: {bucket: views, ...}}`` summed into normalized buckets."""
    out: dict = {}
    for r in rows:
        vid = r.get("video")
        if not vid:
            continue
        bucket = normalize_traffic(r.get("insightTrafficSourceType"))
        try:
            v = int(float(r.get("views") or 0))
        except Exception:
            v = 0
        d = out.setdefault(str(vid), {})
        d[bucket] = d.get(bucket, 0) + v
    return out


def sum_early_window(day_rows: List[dict], published_date: str) -> dict:
    """Pure. day_rows = dicts with keys day / views for ONE video → views in the first
    ~day, ~2 days, ~7 days from publish. (Calendar-day proxy per DISCOVERY.md §6 — label
    it "first day(s)", not a literal rolling 24h.)"""
    try:
        pub = datetime.strptime(published_date[:10], "%Y-%m-%d").date()
    except Exception:
        return {"first_24h": None, "first_48h": None, "first_7d": None}
    w24 = pub + timedelta(days=1)
    w48 = pub + timedelta(days=2)
    w7 = pub + timedelta(days=7)
    s24 = s48 = s7 = 0
    for r in day_rows:
        try:
            d = datetime.strptime(str(r.get("day"))[:10], "%Y-%m-%d").date()
            v = int(float(r.get("views") or 0))
        except Exception:
            continue
        if d < pub:
            continue
        if d <= w24:
            s24 += v
        if d <= w48:
            s48 += v
        if d <= w7:
            s7 += v
    return {"first_24h": s24, "first_48h": s48, "first_7d": s7}


# ── I/O wrappers (degrade to {} on any failure) ──────────────────────────

def _chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def _client(db, channel_id: int):
    import models
    from youtube import oauth as yt_oauth
    token = (db.query(models.OAuthToken)
             .filter(models.OAuthToken.channel_id == int(channel_id)).first())
    if token is None or ANALYTICS_SCOPE not in (getattr(token, "scopes", "") or ""):
        return None
    creds = yt_oauth.get_credentials(db, int(channel_id))
    from googleapiclient.discovery import build
    return build("youtubeAnalytics", "v2", credentials=creds, cache_discovery=False)


def fetch_core(client, video_ids: List[str], *, start: str, end: str) -> dict:
    """{video_id: {views, minutes, avg_view_seconds, avg_view_pct, subs_gained}}."""
    out: dict = {}
    for batch in _chunks([v for v in video_ids if v], _BATCH):
        try:
            resp = client.reports().query(
                ids="channel==MINE", startDate=start, endDate=end,
                metrics=CORE_METRICS, dimensions="video",
                filters="video==" + ",".join(batch), maxResults=len(batch),
            ).execute()
        except Exception as exc:
            log.info("analytics core query failed: %s", str(exc)[:160]); continue
        for r in _rows(resp):
            vid = r.get("video")
            if not vid:
                continue
            out[str(vid)] = {
                "views": int(float(r.get("views") or 0)),
                "minutes": int(float(r.get("estimatedMinutesWatched") or 0)),
                "avg_view_seconds": float(r.get("averageViewDuration") or 0),
                "avg_view_pct": float(r.get("averageViewPercentage") or 0),
                "subs_gained": int(float(r.get("subscribersGained") or 0)),
            }
    return out


def fetch_traffic(client, video_ids: List[str], *, start: str, end: str) -> dict:
    """{video_id: {bucket: views}} via dimensions=video,insightTrafficSourceType."""
    out: dict = {}
    for batch in _chunks([v for v in video_ids if v], _BATCH):
        try:
            resp = client.reports().query(
                ids="channel==MINE", startDate=start, endDate=end,
                metrics="views", dimensions="video,insightTrafficSourceType",
                filters="video==" + ",".join(batch), maxResults=10 * len(batch),
            ).execute()
        except Exception as exc:
            log.info("analytics traffic query failed: %s", str(exc)[:160]); continue
        merged = aggregate_traffic(_rows(resp))
        out.update(merged)
    return out


def fetch_early_window(client, video_id: str, published_date: str, *, span_days: int = 8) -> dict:
    """Per-video early-window views (1 query/video). Calendar-day proxy."""
    try:
        pub = datetime.strptime(published_date[:10], "%Y-%m-%d").date()
    except Exception:
        return {"first_24h": None, "first_48h": None, "first_7d": None}
    end = min(pub + timedelta(days=span_days), date.today())
    try:
        resp = client.reports().query(
            ids="channel==MINE", startDate=pub.isoformat(), endDate=end.isoformat(),
            metrics="views", dimensions="day", filters="video==" + video_id, maxResults=span_days + 2,
        ).execute()
    except Exception as exc:
        log.info("analytics early-window query failed: %s", str(exc)[:160])
        return {"first_24h": None, "first_48h": None, "first_7d": None}
    return sum_early_window(_rows(resp), published_date)
