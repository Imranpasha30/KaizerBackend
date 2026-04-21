"""Google News RSS grounding — ported from the extension's `fetchNewsContext`.

Kept deliberately simple: synchronous feedparser call behind a strict timeout.
Used as best-effort context for SEO generation — empty list on any failure is
acceptable and the pipeline continues without grounding.
"""
from __future__ import annotations

import urllib.parse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Dict, List

import feedparser


_NEWS_TIMEOUT_S = 8.0   # bumped from 4s — cold Google News fetches can take 5-7s
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="kaizer-news")

# User-Agent matters: bare feedparser UA is blocked by Google News in some regions.
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Kaizer/1.0 News-Fetcher"


def _parse(url: str):
    return feedparser.parse(url, request_headers={"User-Agent": _UA, "Accept-Language": "te,en;q=0.5"})


def fetch_news_context(topic: str, lang: str = "te", country: str = "IN",
                       max_items: int = 5) -> List[Dict]:
    """Return ``[{title, source, link, published}]`` for the given topic.

    Hard timeout of 4 seconds. Empty list on any failure — the SEO pipeline
    never fails because the news fetch timed out.
    """
    if not topic or not topic.strip():
        return []

    q = urllib.parse.quote_plus(topic.strip()[:200])
    url = (
        f"https://news.google.com/rss/search"
        f"?q={q}&hl={lang}-{country}&gl={country}&ceid={country}:{lang}"
    )

    try:
        future = _executor.submit(_parse, url)
        feed = future.result(timeout=_NEWS_TIMEOUT_S)
    except (FuturesTimeout, Exception):
        return []

    items: List[Dict] = []
    for entry in (getattr(feed, "entries", None) or [])[:max_items]:
        title = (entry.get("title") or "").strip()
        if not title:
            continue
        source = ""
        src = entry.get("source")
        if isinstance(src, dict):
            source = (src.get("title") or "").strip()
        elif hasattr(src, "get"):
            source = (src.get("title") or "").strip()
        items.append({
            "title": title,
            "source": source or "Google News",
            "link": entry.get("link") or "",
            "published": entry.get("published") or "",
        })
    return items
