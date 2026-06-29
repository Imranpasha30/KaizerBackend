"""Per-channel "what works" profile — the engine of the SEO feedback loop.

The existing analytics poller (analytics/poller.py) already stores
``models.ClipPerformance`` rows (views/likes/comments + the SEO score) per
published video per channel. This module turns that raw history into a small
profile the per-channel SEO adapter (seo/per_channel.py) consumes:

    {ready, n_videos, winning_keywords, top_views_per_hour, ...}

"Winning keywords" = the terms shared by this channel's BEST-performing videos
(ranked by views-per-hour, so a 2-day-old hit isn't unfairly beaten by a
6-month-old one). Pure DB + TF-IDF (reuses seo.score_checker), no network, no
AI. Returns an empty/not-ready profile when the channel lacks enough history,
so the adapter safely falls back to the base SEO. This is also what the
channel-wise Insights tab reads ("what the system learnt + how it's going").
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

log = logging.getLogger("kaizer.seo.performance_profile")


def _views_per_hour(row) -> float:
    h = max(1.0, float(getattr(row, "hours_since_publish", 0) or 1.0))
    return float(getattr(row, "views", 0) or 0) / h


# Terms that recur across the channel but carry no SEO signal.
_KW_STOP = {
    "the", "a", "an", "of", "on", "in", "to", "for", "and", "or", "is", "are",
    "with", "at", "by", "video", "videos", "shorts", "short", "live", "new",
    "today", "latest", "full", "watch", "ft", "vs",
}


def _recurring_keywords(titles, *, top: int = 12, min_videos: int = 3,
                        min_df_frac: float = 0.12) -> list[str]:
    """The channel's EVERGREEN / RECURRING terms — words & phrases that appear
    across MANY of its videos (its brand + beat: "telugu news", "breaking",
    the channel name). This is DOCUMENT FREQUENCY, the OPPOSITE of TF-IDF:
    TF-IDF surfaces a story's distinctive one-off terms (a person/event), which
    for NEWS must NOT be carried onto a different new video. Recurring terms are
    the only ones safe to reuse across topics. Returns [] on sparse input."""
    titles = [(t or "").strip() for t in (titles or []) if (t or "").strip()]
    n = len(titles)
    if n < min_videos:
        return []
    from collections import Counter
    df: Counter = Counter()
    for t in titles:
        toks = [w for w in re.findall(r"[#\w]+", t.lower())
                if len(w) >= 3 and w not in _KW_STOP]
        terms = set(toks)
        for i in range(len(toks) - 1):           # bigrams (e.g. "telugu news")
            terms.add(f"{toks[i]} {toks[i + 1]}")
        for term in terms:
            df[term] += 1
    thresh = max(2, int(round(min_df_frac * n)))
    rec = [(term, c) for term, c in df.items() if c >= thresh]
    # Higher recurrence first; prefer multi-word phrases as tie-break (more
    # specific brand/beat signals than a bare unigram).
    rec.sort(key=lambda kv: (kv[1], kv[0].count(" ")), reverse=True)
    return [term for term, _ in rec[:top]]


def _winning_from_titles(titles, max_keywords: int) -> list[str]:
    """Winning keywords for a channel: prefer EVERGREEN recurring terms; fall
    back to TF-IDF only when history is too sparse to find recurrence."""
    rec = _recurring_keywords(titles, top=max_keywords)
    if len(rec) >= 3:
        return rec
    try:
        from seo.score_checker import _content_keywords
        return _content_keywords("\n".join(titles), top=max_keywords)
    except Exception:
        return rec


def _profile_from_catalog(
    db, channel_id: int, *, top_frac: float, min_videos: int, max_keywords: int,
) -> dict[str, Any] | None:
    """Winning keywords from the channel's REAL YouTube catalogue
    (``models.ChannelVideo``, synced by analytics/channel_catalog), ranked by
    view_count. Reads the CACHED catalogue only — no network here (the no-network
    contract of this module is preserved; the sync is triggered elsewhere).
    Returns None if the catalogue isn't populated for this channel."""
    try:
        import models
        ch = db.get(models.Channel, int(channel_id))
        tok = getattr(ch, "oauth_token", None) if ch else None
        gcid = (getattr(tok, "google_channel_id", "") or "") if tok else ""
        uid = getattr(ch, "user_id", None) if ch else None
        if not gcid or uid is None:
            return None
        rows = (
            db.query(models.ChannelVideo)
              .filter(models.ChannelVideo.user_id == uid,
                      models.ChannelVideo.google_channel_id == gcid)
              .all()
        )
        if len(rows) < min_videos:
            return None
        rows.sort(key=lambda r: int(getattr(r, "view_count", 0) or 0), reverse=True)
        n_top = max(1, int(round(len(rows) * top_frac)))
        top = rows[:n_top]
        titles = [(getattr(r, "title", "") or "").strip()
                  for r in top if (getattr(r, "title", "") or "").strip()]
        winning = _winning_from_titles(titles, max_keywords)
        return {
            "ready": True,
            "signal": "catalog_views",          # ranked by real YT view_count
            "n_videos": len(rows),
            "top_n": n_top,
            "winning_keywords": winning,
            "top_views_per_hour": 0.0,           # N/A for the catalogue path
            "median_views_per_hour": 0.0,
            "top_titles": titles[:5],
            "source": "youtube_catalog",
        }
    except Exception as exc:
        log.warning("performance_profile: catalog fallback failed for channel=%s: %s",
                    channel_id, exc)
        return None


def build_channel_profile(
    db,
    channel_id: int,
    *,
    top_frac: float = 0.4,
    min_videos: int = 4,
    max_keywords: int = 12,
) -> dict[str, Any]:
    """Build a channel's performance profile from ClipPerformance history.
    Never raises — returns ``{"ready": False, ...}`` on any shortfall."""
    try:
        import models
        rows = (
            db.query(models.ClipPerformance)
              .filter(models.ClipPerformance.channel_id == int(channel_id))
              .all()
        )
    except Exception as exc:
        log.warning("performance_profile: query failed for channel=%s: %s", channel_id, exc)
        return {"ready": False, "n_videos": 0, "winning_keywords": [], "reason": "query failed"}

    # Keep the LATEST sample per video (the poller appends over time).
    latest: dict[str, Any] = {}
    for r in rows:
        vid = getattr(r, "video_id", None)
        if not vid:
            continue
        prev = latest.get(vid)
        if prev is None or _views_per_hour(r) >= _views_per_hour(prev):
            latest[vid] = r
    vids = list(latest.values())

    if len(vids) < min_videos:
        # Kaizer-published history is sparse — but a monetized channel has
        # 100s of NATIVE YouTube videos that never went through Kaizer. Mine
        # the channel's REAL catalogue (channel_videos, synced from the YT
        # Data API) for winning keywords instead of giving up.
        cat = _profile_from_catalog(
            db, int(channel_id),
            top_frac=top_frac, min_videos=min_videos, max_keywords=max_keywords,
        )
        if cat is not None:
            return cat
        return {"ready": False, "n_videos": len(vids), "winning_keywords": [],
                "reason": f"need >= {min_videos} videos with performance data"}

    # Rank by the BEST available signal. CTR (click-through-rate) is the gold
    # standard but needs the analytics scope (operator re-approval); until then
    # fetch_video_ctr returns {} and we transparently rank by views-per-hour.
    # The moment a channel is re-approved, this auto-upgrades to CTR — no code
    # change needed.
    signal = "views_per_hour"
    ctr_map: dict = {}
    try:
        from analytics.ctr import fetch_video_ctr
        ctr_map = fetch_video_ctr(
            db, int(channel_id), [getattr(v, "video_id", None) for v in vids]
        )
    except Exception:
        ctr_map = {}
    have_ctr = sum(1 for v in vids if getattr(v, "video_id", None) in ctr_map)
    if ctr_map and have_ctr >= min_videos:
        signal = "ctr"
        vids.sort(
            key=lambda v: ctr_map.get(getattr(v, "video_id", None), {}).get("ctr", 0.0),
            reverse=True,
        )
    else:
        vids.sort(key=_views_per_hour, reverse=True)
    n_top = max(1, int(round(len(vids) * top_frac)))
    top = vids[:n_top]

    # Collect the title + tags of the TOP performers.
    texts: list[str] = []
    top_titles: list[str] = []
    for r in top:
        seo = {}
        try:
            import models
            clip = db.get(models.Clip, r.clip_id) if getattr(r, "clip_id", None) else None
            if clip and clip.seo:
                seo = json.loads(clip.seo)
        except Exception:
            seo = {}
        title = (seo.get("title") or "").strip()
        tags = seo.get("tags") or []
        if title:
            top_titles.append(title)
        texts.append(f"{title} {' '.join(str(t) for t in tags)}")

    # Evergreen recurring terms across the top performers' titles (not the
    # topical one-offs TF-IDF would surface). Falls back to TF-IDF when sparse.
    try:
        winning = _winning_from_titles(top_titles or [t for t in texts if t.strip()],
                                       max_keywords)
    except Exception as exc:
        log.warning("performance_profile: keyword extraction failed: %s", exc)
        winning = []

    return {
        "ready": True,
        "signal": signal,   # "ctr" once re-approved, else "views_per_hour"
        "n_videos": len(vids),
        "top_n": n_top,
        "winning_keywords": winning,
        "top_views_per_hour": round(_views_per_hour(top[0]), 2),
        "median_views_per_hour": round(
            sorted(_views_per_hour(v) for v in vids)[len(vids) // 2], 2),
        "top_titles": top_titles[:5],
    }
