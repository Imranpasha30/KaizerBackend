"""Derived, channel-relative metrics — pure functions (no I/O, no API).

Computed at ingest from the raw per-video data and attached to each VideoMetric. These
turn absolute numbers into the comparable signals the analysis engine ranks on. Unit-
tested in ``scripts/test_insights_logic.py``.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import List, Optional, Tuple

# Classic Shorts are ≤60s. (YouTube now allows up to 180s, but for news channels the
# long-form is minutes and Shorts are sub-minute; 60 keeps false-positives low.) Tunable.
SHORT_MAX_SECONDS = 60

# Generic stopwords stripped before topic clustering. Deliberately small + domain-light so
# meaningful news terms survive; native-script (Telugu/Hindi/…) tokens pass through as-is.
_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "have", "has", "had", "are",
    "was", "were", "will", "your", "you", "his", "her", "she", "him", "they", "them",
    "but", "not", "all", "any", "can", "out", "who", "what", "when", "why", "how",
    "new", "live", "today", "watch", "video", "full", "latest", "update", "updates",
    "viral", "shorts", "short", "part", "episode",
}


def is_short(duration_seconds: int) -> bool:
    """Heuristic: a Short is a very short clip. No API flag exists for this."""
    try:
        return 0 < int(duration_seconds or 0) <= SHORT_MAX_SECONDS
    except Exception:
        return False


def reach_ratio(views: int, subscriber_count: int) -> float:
    """views ÷ subscriber base — >1 means the video reached well beyond subscribers
    (i.e. it escaped the sub feed into Browse/Suggested). The single clearest 'did the
    algorithm pick this up' signal when CTR isn't available."""
    subs = max(1, int(subscriber_count or 0))
    return round(float(views or 0) / subs, 4)


def velocity_vph(views_in_window: Optional[int], window_hours: float) -> Optional[float]:
    """Early velocity = views per hour over an early window (e.g. first 48h). None when
    the early-window data isn't available (public mode / no analytics)."""
    if views_in_window is None or window_hours <= 0:
        return None
    return round(float(views_in_window) / float(window_hours), 3)


def lifetime_vph(total_views: int, age_hours: float) -> Optional[float]:
    """Fallback velocity from lifetime views ÷ age — weaker than the early-window figure
    (it's diluted by the long tail) but available for every video."""
    if age_hours <= 0:
        return None
    return round(float(total_views or 0) / float(age_hours), 3)


def _to_aware_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def age_hours(published_utc: Optional[datetime], *, now: Optional[datetime] = None) -> Optional[float]:
    pub = _to_aware_utc(published_utc)
    if pub is None:
        return None
    ref = _to_aware_utc(now) or datetime.now(timezone.utc)
    secs = (ref - pub).total_seconds()
    return round(secs / 3600.0, 3) if secs > 0 else 0.0


def publish_local(published_utc: Optional[datetime], tz_name: str) -> Tuple[Optional[datetime], Optional[int], Optional[int]]:
    """Convert a UTC publish time to the channel's local TZ → (local_dt, dow, hour).
    dow: 0=Mon … 6=Sun. Falls back to UTC if the tz name is unknown."""
    pub = _to_aware_utc(published_utc)
    if pub is None:
        return None, None, None
    tz = timezone.utc
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name) if tz_name else timezone.utc
    except Exception:
        tz = timezone.utc
    local = pub.astimezone(tz)
    return local, local.weekday(), local.hour


def _tokens(title: str) -> List[str]:
    out = []
    for tok in re.split(r"[^0-9A-Za-z-￿]+", (title or "")):
        t = tok.strip().lower()
        if len(t) >= 4 and t not in _STOPWORDS and not t.isdigit():
            out.append(t)
    return out


def build_topic_clusters(titles: List[str]) -> List[str]:
    """Assign each title a topic-cluster label (one per title, same order). Deterministic:
    label = the title's token with the highest GLOBAL frequency across all titles, so
    titles sharing a common significant word land in the same cluster. 'general' when a
    title has no significant token. A lightweight, dependency-free clusterer for v1 — the
    LLM report can name clusters more naturally on top of these groupings."""
    toks_per = [_tokens(t) for t in titles]
    freq: dict[str, int] = {}
    for toks in toks_per:
        for t in set(toks):           # count each token once per title
            freq[t] = freq.get(t, 0) + 1
    labels: List[str] = []
    for toks in toks_per:
        best, best_f = "general", 0
        for t in toks:
            f = freq.get(t, 0)
            # prefer higher global frequency; tie-break alphabetically for determinism
            if f > best_f or (f == best_f and best != "general" and t < best):
                best, best_f = t, f
        # a token seen in only ONE title isn't a cluster — keep it but it'll be a singleton
        labels.append(best if best_f >= 1 else "general")
    return labels
