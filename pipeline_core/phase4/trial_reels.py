"""
kaizer.pipeline.phase4.trial_reels
===================================
Meta Trial Reels adapter + auto-promote orchestrator.

Meta shipped Trial Reels in Dec 2024: a Reel is published only to
non-followers, A/B tested for 24h, auto-promoted if engagement > X.
Scheduling was added in 2025. Eligibility: 1000+ followers.

The Graph API has a `is_trial_reel` boolean at media-container creation.
24h after publish, Meta exposes engagement metrics via the standard
/{ig-media-id}/insights endpoint. If shares/reach > threshold and
completion > threshold, call the promotion endpoint to flip the trial
to public.

Nobody in the SaaS category programs this. We can.

Phase 4 scope
-------------
  - OAuth token handling (shares with the existing Meta flow — not yet
    integrated in v1 Kaizer which is YouTube-first)
  - Trial publish path in the Instagram upload worker
  - 24h-post-publish cron that pulls insights, decides, and acts
  - Threshold configuration per-creator
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("kaizer.pipeline.phase4.trial_reels")


@dataclass
class TrialMetrics:
    media_id: str
    shares: int
    reach: int
    completion_pct: float
    saves: int
    elapsed_hours: float


@dataclass
class TrialDecision:
    media_id: str
    action: str          # 'promote' | 'keep_trial' | 'delete'
    reason: str
    evaluated_at: Optional[str] = None


def publish_as_trial(
    media_path: str,
    *,
    access_token: str,
    ig_user_id: str,
    caption: str = "",
) -> str:
    """Publish a Reel in Trial mode via Meta Graph API.

    Returns the created media_id. Phase 4 implementation required.
    """
    raise NotImplementedError(
        "trial_reels.publish_as_trial — Phase 4. "
        "See docs/PHASE4_ROADMAP.md § Trial Reels."
    )


def fetch_trial_metrics(media_id: str, *, access_token: str) -> TrialMetrics:
    raise NotImplementedError(
        "trial_reels.fetch_trial_metrics — Phase 4."
    )


def decide_promotion(
    metrics: TrialMetrics,
    *,
    shares_per_reach_threshold: float = 0.015,    # 1.5% from Buffer 2026
    completion_threshold: float = 50.0,
) -> TrialDecision:
    """Apply the auto-promote rule.

    Pure function — safe to use in tests. Phase 4 wires it into the cron.
    """
    shares_per_reach = (metrics.shares / max(1, metrics.reach))
    if (shares_per_reach >= shares_per_reach_threshold
            and metrics.completion_pct >= completion_threshold
            and metrics.elapsed_hours >= 24.0):
        return TrialDecision(
            media_id=metrics.media_id, action="promote",
            reason=(f"shares/reach {shares_per_reach:.3f} ≥ {shares_per_reach_threshold}, "
                    f"completion {metrics.completion_pct:.0f}% ≥ {completion_threshold}"),
        )
    if metrics.elapsed_hours < 24.0:
        return TrialDecision(
            media_id=metrics.media_id, action="keep_trial",
            reason=f"only {metrics.elapsed_hours:.1f}h elapsed (<24h evaluation window)",
        )
    return TrialDecision(
        media_id=metrics.media_id, action="keep_trial",
        reason=(f"engagement below promotion threshold (shares/reach "
                f"{shares_per_reach:.3f}, completion {metrics.completion_pct:.0f}%)"),
    )


def promote_trial(media_id: str, *, access_token: str) -> bool:
    raise NotImplementedError(
        "trial_reels.promote_trial — Phase 4."
    )
