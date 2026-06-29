"""Channel-maturity auto-classification — the user never picks; we detect.

Decides which analysis path runs for a channel, from public size signals + whether the
owner granted the analytics scope. Thresholds + decision order are the ones locked in
``DISCOVERY.md`` §7 (verified against YouTube's data-stabilization guidance: CTR variance
settles ~50+ videos; seasonality needs ~6 months).

  DEEP    — enough owned history for full root-cause diagnostics
  BLEND   — some owned history; diagnose where the sample allows, labelled "growing confidence"
  STARTER — too little / no analytics access → prescribe a plan instead of diagnosing

Pure function, no I/O — unit-tested in ``scripts/test_insights_logic.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

DEEP = "deep"
BLEND = "blend"
STARTER = "starter"

# Thresholds (DISCOVERY.md §7). Kept as named constants so the freeze test pins them.
MIN_VIDEOS_BLEND = 10
MIN_AGE_DAYS_BLEND = 30
MIN_VIEWS_BLEND = 10_000
MIN_VIDEOS_DEEP = 100
MIN_AGE_DAYS_DEEP = 180
MIN_VIEWS_DEEP = 100_000


@dataclass(frozen=True)
class Maturity:
    mode: str            # deep | blend | starter
    reason: str          # human-readable why (shown in-product for transparency)
    analytics: bool      # was the analytics scope granted for this channel


def classify(*, video_count: int, channel_age_days: int, total_views: int,
             analytics_granted: bool) -> Maturity:
    """Return the analysis mode for a channel. Decision order matches DISCOVERY.md §7."""
    v = max(0, int(video_count or 0))
    age = max(0, int(channel_age_days or 0))
    views = max(0, int(total_views or 0))
    a = bool(analytics_granted)

    # Brand-new / no-access → prescribe, don't diagnose.
    if (not a or v < MIN_VIDEOS_BLEND) and age < MIN_AGE_DAYS_BLEND:
        why = ("no analytics access yet" if not a
               else f"only {v} videos and {age}d old")
        return Maturity(STARTER, f"Starter plan — {why}; not enough owned history to diagnose.", a)

    # Enough owned history to analyze.
    if a and (v >= MIN_VIDEOS_BLEND or views >= MIN_VIEWS_BLEND) and age >= MIN_AGE_DAYS_BLEND:
        if v >= MIN_VIDEOS_DEEP and views >= MIN_VIEWS_DEEP and age >= MIN_AGE_DAYS_DEEP:
            return Maturity(
                DEEP,
                f"Deep diagnostic — {v} videos, {views:,} views, {age}d of owned history.",
                a)
        return Maturity(
            BLEND,
            f"Early-stage (growing confidence) — {v} videos, {views:,} views, {age}d; "
            f"diagnosing where the sample is large enough.",
            a)

    # Has some history but no analytics scope (or below blend floor) → starter fallback.
    why = ("analytics not granted" if not a else f"{v} videos / {age}d below diagnostic floor")
    return Maturity(STARTER, f"Starter plan — {why}.", a)
