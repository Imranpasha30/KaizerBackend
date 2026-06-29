"""YouTube Analytics CTR fetch — the "gold signal" for the SEO feedback loop.

Views/likes come from the Data API (current scope, no re-consent). CTR +
impressions + average-view-duration come from the YouTube ANALYTICS API, which
needs the ``yt-analytics.readonly`` scope. We added that scope to the OAuth
request (youtube/oauth.py), so:

  * Existing tokens (no analytics scope) -> ``fetch_video_ctr`` returns {} and
    the feedback loop keeps using views. NOTHING breaks.
  * The moment an operator RE-APPROVES, their new token carries the scope and
    this module starts returning real CTR/impressions automatically — no code
    change, no redeploy. That is the "works the instant I re-approve" guarantee.

Every call is wrapped: missing scope, missing creds, API/quota error, or an
empty report all degrade to {} (never raises).
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

log = logging.getLogger("kaizer.analytics.ctr")

ANALYTICS_SCOPE = "https://www.googleapis.com/auth/yt-analytics.readonly"
_BATCH = 200  # video ids per Analytics query (filter list cap is generous)


def token_has_analytics_scope(token) -> bool:
    """True iff this channel's stored OAuth token was granted the analytics
    scope (i.e. the operator has re-approved). Cheap string check."""
    try:
        return ANALYTICS_SCOPE in (getattr(token, "scopes", "") or "")
    except Exception:
        return False


def _chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def fetch_video_ctr(db, channel_id: int, video_ids: list[str], *, days: int = 120) -> dict[str, dict[str, float]]:
    """Return ``{video_id: {ctr, impressions, views, avg_view_seconds,
    avg_view_pct}}`` from the YouTube Analytics API for the channel's videos.

    Returns ``{}`` on ANY shortfall (no analytics scope yet, no creds, API
    error, empty report) so callers transparently fall back to views.
    """
    video_ids = [v for v in (video_ids or []) if v]
    if not video_ids:
        return {}
    try:
        import models
        from youtube import oauth as yt_oauth
        token = (
            db.query(models.OAuthToken)
              .filter(models.OAuthToken.channel_id == int(channel_id))
              .first()
        )
        if token is None or not token_has_analytics_scope(token):
            return {}  # not re-approved yet -> views-only path
        creds = yt_oauth.get_credentials(db, int(channel_id))
    except Exception as exc:
        log.info("ctr: creds/scope unavailable for channel=%s (%s)", channel_id, str(exc)[:120])
        return {}

    try:
        from googleapiclient.discovery import build
        ya = build("youtubeAnalytics", "v2", credentials=creds, cache_discovery=False)
    except Exception as exc:
        log.warning("ctr: could not build youtubeAnalytics client: %s", str(exc)[:160])
        return {}

    start = (date.today() - timedelta(days=max(1, days))).isoformat()
    end = date.today().isoformat()
    out: dict[str, dict[str, float]] = {}
    for batch in _chunks(video_ids, _BATCH):
        try:
            resp = ya.reports().query(
                ids="channel==MINE",
                startDate=start,
                endDate=end,
                metrics="views,impressions,impressionClickThroughRate,"
                        "averageViewDuration,averageViewPercentage",
                dimensions="video",
                filters="video==" + ",".join(batch),
                maxResults=len(batch),
            ).execute()
        except Exception as exc:
            log.info("ctr: analytics query failed (channel=%s): %s", channel_id, str(exc)[:160])
            continue
        headers = [h.get("name") for h in (resp.get("columnHeaders") or [])]
        for row in (resp.get("rows") or []):
            rec = dict(zip(headers, row))
            vid = rec.get("video")
            if not vid:
                continue
            out[str(vid)] = {
                "views": float(rec.get("views") or 0),
                "impressions": float(rec.get("impressions") or 0),
                # API returns CTR as a percentage (e.g. 4.3) — normalize to 0..1.
                "ctr": float(rec.get("impressionClickThroughRate") or 0) / 100.0,
                "avg_view_seconds": float(rec.get("averageViewDuration") or 0),
                "avg_view_pct": float(rec.get("averageViewPercentage") or 0),
            }
    return out
