"""Admin-only endpoints for cross-platform posting via Postiz.

All routes require ``is_admin=True`` so regular Kaizer users don't
see the cross-post option until we've completed full verification on
each platform (Twitter / IG / LinkedIn / TikTok / …).

Endpoints:
  GET  /api/postiz/status                 → is Postiz reachable + how
                                            many platforms connected
  GET  /api/postiz/integrations           → list connected platforms
                                            (Twitter / IG / LinkedIn / …)
  POST /api/postiz/schedule               → schedule a post across N
                                            integrations with a video URL

The frontend hides this UI behind a `user.is_admin` check too
(belt-and-suspenders so a deep-linker can't render a leaky modal),
but the backend gate here is the source of truth.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import auth
import models
from clients import postiz as postiz_client

logger = logging.getLogger("kaizer.routers.postiz")
router = APIRouter(prefix="/api/postiz", tags=["postiz"])


# ─── Models ──────────────────────────────────────────────────────────────────

class PostizSchedule(BaseModel):
    integration_ids: list[str]
    text: str = ""
    media_url: Optional[str] = None       # public URL (R2) Postiz fetches
    schedule_at_iso: Optional[str] = None # ISO-8601 UTC; None = now
    type: str = "now"                      # "draft" | "scheduled" | "now"


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/status")
def status(_user: models.User = Depends(auth.admin_required)) -> dict:
    """Cheap reachability probe — used by the admin UI to decide
    whether to render the Cross-post toggle. No data leak: gated by
    admin_required so non-admins can't even see if it's configured."""
    if not postiz_client.is_enabled():
        return {
            "enabled": False,
            "reason": "POSTIZ_API_KEY not set in backend env",
        }
    try:
        integrations = postiz_client.list_integrations()
    except postiz_client.PostizAuthError as e:
        return {"enabled": False, "reason": f"auth: {e}"}
    except postiz_client.PostizError as e:
        return {"enabled": False, "reason": f"unreachable: {e}"}
    return {
        "enabled": True,
        "integration_count": len(integrations),
        "providers": sorted({i.get("provider", "") for i in integrations
                              if i.get("provider")}),
    }


@router.get("/integrations")
def integrations(_user: models.User = Depends(auth.admin_required)) -> list[dict]:
    """List connected platforms Postiz can post to on this user's
    behalf. Only the fields the UI needs are returned — no tokens."""
    try:
        raw = postiz_client.list_integrations()
    except postiz_client.PostizAuthError as e:
        raise HTTPException(status_code=503, detail=f"Postiz auth: {e}")
    except postiz_client.PostizError as e:
        raise HTTPException(status_code=503, detail=f"Postiz unreachable: {e}")
    return [
        {
            "id":         i.get("id", ""),
            "name":       i.get("name", ""),
            "provider":   i.get("provider", ""),
            "picture":    i.get("picture", ""),
            "identifier": i.get("identifier", ""),
        }
        for i in raw
    ]


@router.post("/schedule")
def schedule(
    payload: PostizSchedule,
    _user: models.User = Depends(auth.admin_required),
) -> dict:
    """Schedule a post on the chosen Postiz integrations.

    ``media_url`` should be a publicly-reachable URL (R2 works); Postiz
    fetches the bytes server-side and uploads to each platform.
    """
    if not payload.integration_ids:
        raise HTTPException(status_code=400, detail="No integrations selected")
    try:
        result = postiz_client.schedule_post(
            integration_ids=payload.integration_ids,
            text=payload.text or "",
            media_url=payload.media_url,
            schedule_at_iso=payload.schedule_at_iso,
            type_=payload.type or "now",
        )
    except postiz_client.PostizAuthError as e:
        raise HTTPException(status_code=503, detail=f"Postiz auth: {e}")
    except postiz_client.PostizError as e:
        raise HTTPException(status_code=502, detail=f"Postiz scheduling failed: {e}")
    logger.info("postiz.schedule ok: integrations=%s", payload.integration_ids)
    return {"ok": True, "postiz_response": result}
