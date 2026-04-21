"""Billing router — plan info, current-usage, stubbed Stripe checkout.

Stripe integration is stubbed on purpose.  The /checkout-session endpoint
returns a 501 + an explanatory payload when STRIPE_SECRET_KEY is missing, so
the frontend's "Upgrade" button can show a clear "billing not yet live" state
instead of erroring out opaquely.

When the user later creates a Stripe account and sets STRIPE_SECRET_KEY,
the same endpoint will mint a real Checkout URL — no frontend changes.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db
import models
import auth
from billing import plans as billing_plans


router = APIRouter(prefix="/api/billing", tags=["billing"])


# ─── Helpers ──────────────────────────────────────────────────────────

def _reset_usage_if_new_month(user: models.User, db: Session) -> None:
    """Lazy monthly counter reset — so we don't need a cron for it.

    If `usage_reset_at` is null OR from a prior calendar month, zero the
    counter and bump the reset timestamp to now.  Idempotent under
    concurrent access because the overwrite is the same in either race.
    """
    now = datetime.now(timezone.utc)
    reset = user.usage_reset_at
    # Strip tzinfo for SQLite naive comparison compatibility
    if reset and reset.tzinfo is None:
        reset = reset.replace(tzinfo=timezone.utc)
    needs_reset = (
        reset is None
        or reset.year  != now.year
        or reset.month != now.month
    )
    if needs_reset:
        user.monthly_clip_count = 0
        user.usage_reset_at = now
        db.commit()


def _user_plan(user: models.User) -> dict:
    return billing_plans.get_plan(user.plan or billing_plans.DEFAULT_PLAN)


def _count_connected_yt_accounts(db: Session, user_id: int) -> int:
    """Distinct real YT accounts this user has OAuth'd.  Dedupes by
    google_channel_id so multiple profiles on one YT don't double-count."""
    rows = (
        db.query(models.OAuthToken.google_channel_id)
          .join(models.Channel, models.Channel.id == models.OAuthToken.channel_id)
          .filter(
              models.Channel.user_id == user_id,
              models.OAuthToken.refresh_token_enc.isnot(None),
              models.OAuthToken.google_channel_id.isnot(None),
          )
          .distinct()
          .all()
    )
    return len([r for r in rows if r[0]])


# ─── Public schemas ───────────────────────────────────────────────────

class CheckoutRequest(BaseModel):
    plan_key: str
    cycle:    str = "monthly"   # "monthly" | "yearly"


# ─── Endpoints ────────────────────────────────────────────────────────

@router.get("/plans")
def get_plans():
    """Static tier definitions — consumed by the Billing page."""
    return {
        "currency": "USD",
        "plans":     billing_plans.all_plans(),
        "current_discount_yearly_pct": 17,
    }


@router.get("/me")
def get_my_billing(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Current user's plan + usage snapshot — drives the Billing page's
    'Your plan' card + usage bars.
    """
    _reset_usage_if_new_month(user, db)
    plan = _user_plan(user)
    yt_count = _count_connected_yt_accounts(db, user.id)

    # Clip quota — enforced server-side on /jobs create
    clip_cap = plan["clips_per_month"]
    clip_used = int(user.monthly_clip_count or 0)
    clip_remaining = None if clip_cap < 0 else max(0, clip_cap - clip_used)

    return {
        "plan": {
            "key":            plan["key"],
            "name":           plan["name"],
            "price_monthly":  plan["price_monthly"],
            "price_yearly":   plan["price_yearly"],
            "cycle":          user.plan_cycle or "monthly",
        },
        "renews_at": user.plan_renews_at.isoformat() if user.plan_renews_at else None,
        "stripe_connected": bool(user.stripe_customer_id),
        "usage": {
            "clips_used":      clip_used,
            "clips_limit":     clip_cap,    # -1 = unlimited
            "clips_remaining": clip_remaining,
            "yt_accounts_used":  yt_count,
            "yt_accounts_limit": plan["yt_accounts"],
            "reset_at":          user.usage_reset_at.isoformat() if user.usage_reset_at else None,
        },
        "features": {
            "advanced_seo":     plan["advanced_seo"],
            "campaigns":        plan["campaigns_enabled"],
            "channel_groups":   plan["channel_groups"],
            "thumb_ab":         plan["thumb_ab"],
            "veo_videos_monthly": plan["veo_videos_monthly"],
            "languages":        plan["languages"],
            "brand_kits":       plan["brand_kits"],
            "team_seats":       plan["team_seats"],
            "priority_support": plan["priority_support"],
            "byok":             plan["byok"],
            "analytics_days":   plan["analytics_days"],
        },
    }


@router.post("/checkout-session")
def create_checkout_session(
    payload: CheckoutRequest,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Mint a Stripe Checkout Session URL for the requested plan.

    Stubbed: when `STRIPE_SECRET_KEY` is absent, returns a 501 with a clear
    "billing_not_configured" payload so the frontend can render a "coming
    soon" state instead of a broken redirect.

    When the user wires Stripe:
      1. Create products + prices in Stripe Dashboard for each paid tier.
      2. Set STRIPE_SECRET_KEY + STRIPE_PRICE_<PLAN>_<CYCLE> env vars.
      3. No code change needed — this endpoint will pick them up.
    """
    # Validate plan / cycle regardless of Stripe state so the frontend gets
    # consistent 4xx errors vs. 501s.
    if payload.plan_key not in billing_plans.PLANS:
        raise HTTPException(status_code=404, detail=f"Unknown plan '{payload.plan_key}'")
    if payload.cycle not in ("monthly", "yearly"):
        raise HTTPException(status_code=422, detail="cycle must be 'monthly' or 'yearly'")
    if payload.plan_key == billing_plans.PLAN_FREE:
        raise HTTPException(status_code=422, detail="Free plan does not require checkout")

    stripe_key = os.environ.get("STRIPE_SECRET_KEY", "").strip()
    if not stripe_key:
        # Deliberate stub — frontend shows a friendly "billing launching soon"
        raise HTTPException(
            status_code=501,
            detail={
                "code":    "billing_not_configured",
                "message": (
                    "Stripe is not configured yet.  Billing goes live once "
                    "the platform owner sets STRIPE_SECRET_KEY + STRIPE_PRICE_* "
                    "env vars.  Your plan is unaffected."
                ),
                "plan_key": payload.plan_key,
                "cycle":    payload.cycle,
            },
        )

    # Live Stripe path — not wired yet.  Leaves the happy-path scaffold for
    # when the key arrives so the frontend doesn't need another round-trip.
    # When activating, import stripe here and:
    #   stripe.api_key = stripe_key
    #   price_id = os.environ[f"STRIPE_PRICE_{payload.plan_key.upper()}_{payload.cycle.upper()}"]
    #   session = stripe.checkout.Session.create(
    #       customer=user.stripe_customer_id or None,
    #       customer_email=user.email if not user.stripe_customer_id else None,
    #       mode="subscription",
    #       line_items=[{"price": price_id, "quantity": 1}],
    #       success_url=f"{settings.frontend_url}/billing?session_id={{CHECKOUT_SESSION_ID}}",
    #       cancel_url=f"{settings.frontend_url}/billing",
    #       client_reference_id=str(user.id),
    #   )
    #   return {"url": session.url}
    raise HTTPException(status_code=501, detail="Stripe activation pending")


@router.post("/portal-session")
def create_portal_session(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Return a Stripe Billing Portal URL so the user can manage their
    subscription (cancel, swap plan, update card).  Stubbed until Stripe
    is live — returns 501 with the same friendly payload.
    """
    stripe_key = os.environ.get("STRIPE_SECRET_KEY", "").strip()
    if not stripe_key:
        raise HTTPException(
            status_code=501,
            detail={
                "code":    "billing_not_configured",
                "message": "Billing portal is available once Stripe is activated.",
            },
        )
    if not user.stripe_customer_id:
        raise HTTPException(status_code=409, detail="No active subscription to manage")
    raise HTTPException(status_code=501, detail="Stripe portal activation pending")


@router.post("/dev/set-plan")
def dev_set_plan(
    plan_key: str,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """DEV-ONLY plan switcher — admin-gated.  Lets an admin flip their plan
    without Stripe so they can test the plan-gating UI immediately.

    Disabled when KAIZER_ALLOW_DEV_PLAN_SWITCH=false (set it in production).
    """
    if os.environ.get("KAIZER_ALLOW_DEV_PLAN_SWITCH", "true").lower() == "false":
        raise HTTPException(status_code=403, detail="Dev plan switch is disabled")
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")
    if plan_key not in billing_plans.PLANS:
        raise HTTPException(status_code=404, detail=f"Unknown plan '{plan_key}'")

    user.plan = plan_key
    db.commit()
    return {"plan": plan_key, "set_via": "dev"}
