"""Trending-keyword fetcher — Google Autocomplete (direct API) + pytrends fallback.

Google changed the Trends API in 2026 and pytrends 4.9.x now returns 404 on
`trending_searches` / `related_queries`.  Direct HTTP to Google's public
Autocomplete endpoint is stable, free, rate-limit-tolerant, and returns
exactly what we want for SEO keyword injection: real-world related queries
for a given topic, in the target language.

Output shape (matches the old pytrends contract so callers don't care):
    {
        "related_queries":  [str, ...],   # autocomplete expansions
        "rising_queries":   [str, ...],   # "why / how / when" expansions
        "trending_now":     [str, ...],   # top autocomplete seeds
        "source":           "google-autocomplete" | "unavailable",
    }
"""
from __future__ import annotations

import json
from typing import List, Dict, Any

import requests


_LANG_TO_HL = {
    "te": "te", "hi": "hi", "ta": "ta", "kn": "kn", "ml": "ml",
    "bn": "bn", "mr": "mr", "gu": "gu", "en": "en",
}
_LANG_TO_GL = {
    "te": "IN", "hi": "IN", "ta": "IN", "kn": "IN", "ml": "IN",
    "bn": "IN", "mr": "IN", "gu": "IN", "en": "US",
}

# User-Agent matters — empty UA gets rate-limited fast
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Kaizer/2.0"


def _autocomplete(query: str, *, hl: str, gl: str, timeout: int = 6) -> List[str]:
    """Hit Google's public suggest API → list of related search strings.

    Endpoint returns a JSON array: [query, [suggestion, suggestion, ...], ...]
    """
    if not query or not query.strip():
        return []
    try:
        r = requests.get(
            "https://suggestqueries.google.com/complete/search",
            params={"client": "firefox", "q": query[:150], "hl": hl, "gl": gl},
            headers={"User-Agent": _UA, "Accept-Language": f"{hl},en;q=0.5"},
            timeout=timeout,
        )
        r.raise_for_status()
        data = json.loads(r.text)
        if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], list):
            return [str(s).strip() for s in data[1] if s and str(s).strip()]
    except Exception as e:
        print(f"[trends] autocomplete failed for '{query[:40]}': {e}")
    return []


def fetch_trending_keywords(
    topic: str,
    *,
    lang: str = "te",
    max_terms: int = 12,
    timeout: int = 6,
) -> Dict[str, Any]:
    """Return trending/related keywords for a topic.  Never raises."""
    empty = {
        "related_queries": [], "rising_queries": [], "trending_now": [],
        "source": "unavailable",
    }
    if not topic or not topic.strip():
        return empty

    hl = _LANG_TO_HL.get(lang, "en")
    gl = _LANG_TO_GL.get(lang, "US")

    # ── Related queries: autocomplete on the topic itself ──
    related = _autocomplete(topic, hl=hl, gl=gl, timeout=timeout)

    # ── Rising queries: autocomplete with question-word prefixes ──
    # These surface "how/why/when" related queries people are actually asking.
    rising: List[str] = []
    for prefix in ("why ", "how ", "what ", "latest "):
        rising.extend(_autocomplete(f"{prefix}{topic}", hl=hl, gl=gl, timeout=timeout))
        if len(rising) >= max_terms * 2:
            break

    # ── Trending "now": broad autocomplete — top-of-mind keywords in region ──
    # Use the topic's first token as a broader seed.
    seed = (topic.strip().split()[0] if topic.strip().split() else topic)[:80]
    trending = _autocomplete(seed, hl=hl, gl=gl, timeout=timeout)

    # Dedupe + trim
    def _dedupe(xs: List[str]) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for x in xs:
            k = x.lower().strip()
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(x)
        return out

    related_unique  = _dedupe(related)[:max_terms]
    rising_unique   = [r for r in _dedupe(rising) if r.lower() not in {q.lower() for q in related_unique}][:max_terms]
    trending_unique = [t for t in _dedupe(trending) if t.lower() not in {q.lower() for q in related_unique + rising_unique}][:max_terms]

    if not (related_unique or rising_unique or trending_unique):
        return empty

    return {
        "related_queries":  related_unique,
        "rising_queries":   rising_unique,
        "trending_now":     trending_unique,
        "source":            "google-autocomplete",
    }
