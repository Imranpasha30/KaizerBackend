"""YouTube top-ranking benchmark fetcher.

Queries the YouTube Data API for the top ~5 videos currently ranking for a
news topic, in the target language and uploaded in the last 7 days.  Their
titles become "winning patterns" for Gemini to emulate — no guessing what
works, we measure it.

Uses YOUTUBE_DATA_API_KEY (public API key, no OAuth).  Result is cached for
a short window to avoid re-hitting quota for the same topic within a minute.
"""
from __future__ import annotations

import os
import time
import threading
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


_API_KEY = os.environ.get("YOUTUBE_DATA_API_KEY", "")
_CACHE_TTL = 120  # seconds
_cache: dict[str, tuple[float, list[dict]]] = {}
_lock = threading.Lock()


_LANG_REGION = {
    "te": ("in", "te"), "hi": ("in", "hi"), "ta": ("in", "ta"),
    "kn": ("in", "kn"), "ml": ("in", "ml"), "bn": ("in", "bn"),
    "mr": ("in", "mr"), "gu": ("in", "gu"), "en": ("us", "en"),
}


def fetch_top_videos(topic: str, *, lang: str = "te", max_results: int = 5) -> List[Dict[str, Any]]:
    """Return up to `max_results` top-viewed videos for the topic in the last
    week.  Shape per item:
        {"title": str, "channel": str, "views": int, "published_at": str}

    Returns [] on any error — SEO generation proceeds without this context.
    """
    if not _API_KEY or not topic or not topic.strip():
        return []

    topic_norm = topic.strip()[:150]
    cache_key = f"{lang}|{topic_norm}"
    now = time.time()

    with _lock:
        cached = _cache.get(cache_key)
        if cached and (now - cached[0]) < _CACHE_TTL:
            return cached[1]

    try:
        region, hl = _LANG_REGION.get(lang, ("us", "en"))
        svc = build("youtube", "v3", developerKey=_API_KEY, cache_discovery=False)

        published_after = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

        search = svc.search().list(
            q=topic_norm,
            part="id",
            type="video",
            order="viewCount",
            maxResults=max(max_results, 5),
            regionCode=region.upper(),
            relevanceLanguage=hl,
            publishedAfter=published_after,
        ).execute()

        video_ids = [item["id"]["videoId"] for item in (search.get("items") or []) if item.get("id", {}).get("videoId")]
        if not video_ids:
            with _lock:
                _cache[cache_key] = (now, [])
            return []

        details = svc.videos().list(
            part="snippet,statistics",
            id=",".join(video_ids[:max_results]),
            maxResults=max_results,
        ).execute()

        out: list[dict] = []
        for v in (details.get("items") or [])[:max_results]:
            s = v.get("snippet", {}) or {}
            st = v.get("statistics", {}) or {}
            try:
                views = int(st.get("viewCount", 0))
            except (ValueError, TypeError):
                views = 0
            out.append({
                "title":        s.get("title", ""),
                "channel":      s.get("channelTitle", ""),
                "views":        views,
                "published_at": s.get("publishedAt", ""),
            })
        out.sort(key=lambda x: x["views"], reverse=True)

        with _lock:
            _cache[cache_key] = (now, out)
        return out

    except HttpError as e:
        print(f"[yt-benchmark] HTTP error ({e.resp.status if e.resp else '?'}): {e}")
        return []
    except Exception as e:
        print(f"[yt-benchmark] failed: {e}")
        return []
