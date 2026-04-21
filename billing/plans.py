"""Single source-of-truth for Kaizer's billing tiers.

The backend uses these limits for plan-gating middleware (check_clip_quota,
check_channel_limit, etc.).  The frontend fetches the same config from
`GET /api/billing/plans` so tier cards stay in sync automatically.

To roll out a pricing change, edit this file only — no other code touches
the numbers directly.
"""
from __future__ import annotations

from typing import Dict, Any


# Canonical plan keys.  `User.plan` stores one of these strings.
PLAN_FREE     = "free"
PLAN_CREATOR  = "creator"
PLAN_PRO      = "pro"
PLAN_AGENCY   = "agency"

DEFAULT_PLAN = PLAN_FREE


# Per-tier limits enforced server-side.  `None` / -1 means "unlimited".
PLANS: Dict[str, Dict[str, Any]] = {
    PLAN_FREE: {
        "key":             PLAN_FREE,
        "name":            "Starter",
        "price_monthly":   0,
        "price_yearly":    0,
        "currency":        "USD",
        "clips_per_month":  5,
        "yt_accounts":      1,
        "languages":        1,
        "brand_kits":       1,
        "veo_videos_monthly": 0,
        "campaigns_enabled": False,
        "advanced_seo":      False,     # trends + YT benchmark + retry loop
        "analytics_days":    7,
        "channel_groups":    False,
        "thumb_ab":          False,
        "team_seats":        1,
        "priority_support":  False,
        "byok":              False,
        "highlights": [
            "5 clips / month",
            "1 YouTube account",
            "1 language",
            "Basic AI SEO (no trends / YT benchmark)",
            "7-day analytics",
            "Community support",
        ],
    },
    PLAN_CREATOR: {
        "key":             PLAN_CREATOR,
        "name":            "Creator",
        "price_monthly":   19,
        "price_yearly":    190,         # ~17% off — 2 months free
        "currency":        "USD",
        "clips_per_month":  50,
        "yt_accounts":      3,
        "languages":        3,
        "brand_kits":       1,
        "veo_videos_monthly": 0,
        "campaigns_enabled": True,
        "advanced_seo":      True,
        "analytics_days":    30,
        "channel_groups":    True,
        "thumb_ab":          False,
        "team_seats":        1,
        "priority_support":  False,
        "byok":              False,
        "highlights": [
            "50 clips / month",
            "3 YouTube accounts",
            "3 languages",
            "Full AI SEO (Trends + YT top-5 benchmark)",
            "Basic campaigns + scheduling",
            "30-day analytics",
            "Email support",
        ],
    },
    PLAN_PRO: {
        "key":             PLAN_PRO,
        "name":            "Pro",
        "price_monthly":   49,
        "price_yearly":    490,
        "currency":        "USD",
        "clips_per_month":  200,
        "yt_accounts":      10,
        "languages":        9,          # all supported native languages
        "brand_kits":       -1,
        "veo_videos_monthly": 10,
        "campaigns_enabled": True,
        "advanced_seo":      True,
        "analytics_days":    180,
        "channel_groups":    True,
        "thumb_ab":          True,
        "team_seats":        1,
        "priority_support":  True,
        "byok":              False,
        "highlights": [
            "200 clips / month",
            "10 YouTube accounts",
            "All 9 languages",
            "Full power-ups (Grounding + Trends + YT benchmark + retry ≥95)",
            "Channel groups + campaigns + quiet hours",
            "Thumbnail A/B testing",
            "Veo 3 video generation (10 / month)",
            "180-day analytics + performance leaderboard",
            "Priority email support",
        ],
    },
    PLAN_AGENCY: {
        "key":             PLAN_AGENCY,
        "name":            "Agency",
        "price_monthly":   199,
        "price_yearly":    1990,
        "currency":        "USD",
        "clips_per_month":  -1,         # unlimited
        "yt_accounts":     -1,
        "languages":        9,
        "brand_kits":      -1,
        "veo_videos_monthly": -1,
        "campaigns_enabled": True,
        "advanced_seo":      True,
        "analytics_days":    -1,        # unlimited
        "channel_groups":    True,
        "thumb_ab":          True,
        "team_seats":        5,
        "priority_support":  True,
        "byok":              True,
        "highlights": [
            "Unlimited clips (BYOK optional for own YT quota)",
            "Unlimited YouTube accounts",
            "5 team seats",
            "Unlimited Veo 3 video generation",
            "Competitor radar + full trending",
            "Bulk operations + API access",
            "White-label branding option",
            "Unlimited analytics history",
            "Priority Slack + phone support",
        ],
    },
}

PLAN_ORDER = [PLAN_FREE, PLAN_CREATOR, PLAN_PRO, PLAN_AGENCY]


def get_plan(key: str) -> Dict[str, Any]:
    """Safe lookup — unknown keys fall back to Free to keep the app usable."""
    return PLANS.get((key or "").lower(), PLANS[DEFAULT_PLAN])


def all_plans() -> list[Dict[str, Any]]:
    return [PLANS[k] for k in PLAN_ORDER]
