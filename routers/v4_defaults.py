"""V4 automation defaults — one-time setup so the user only has to
upload a video; everything else (frame layout, language, channels,
SEO style, auto-publish) flows from these settings.

End goal:
  raw video -> trim + image fetch + render + SEO + publish to YouTube,
  with one consent gate before the upload step. Manual edit is still
  available as a fallback when the user wants to tweak anything.
"""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import auth
import models
from database import get_db


router = APIRouter(prefix="/api/v4/defaults", tags=["v4-defaults"])


class V4Defaults(BaseModel):
    """User's saved auto-pipeline preferences.

    Every field is optional — missing keys fall back to system defaults
    so an old user record without these set still works."""
    # Wizard skip
    platform: str = "full_video_shorts_v4"
    frame_layout: str = "torn_card"       # torn_card | clean_card | split_frame | follow_bar
    language: str = "te"                  # ISO code

    # Short rendering defaults
    font_file: str = "NotoSansTelugu-Bold.ttf"
    text_color: str = "#FFFFFF"

    # Publishing — channels to fan out to + privacy
    channel_ids: list[int] = Field(default_factory=list)
    privacy: str = "public"               # public | unlisted | private

    # Automation
    auto_publish: bool = False            # publish straight after render
    require_consent: bool = True          # show "Confirm upload" modal first
    brand_suffix: str = ""                # appended to titles, e.g. " | KAIZER X"

    # Watermark — semi-transparent text/logo painted on every rendered
    # output. Mirrors the way real news channels stamp their bug onto
    # every frame so the clip stays attributable when re-shared. Empty
    # text = logo-only watermark.
    watermark_text:    str  = ""
    watermark_opacity: float = 0.35       # 0.0 = invisible, 1.0 = solid
    watermark_position: str = "top-right" # top-left | top-right | bottom-left | bottom-right


def _read(user: models.User) -> V4Defaults:
    """Best-effort parse — broken JSON returns the empty defaults so
    the editor isn't blocked on a one-off corrupted row."""
    if not user.v4_defaults:
        return V4Defaults()
    try:
        return V4Defaults.model_validate_json(user.v4_defaults)
    except Exception:
        return V4Defaults()


@router.get("")
def get_defaults(
    user: models.User = Depends(auth.current_user),
) -> dict:
    return json.loads(_read(user).model_dump_json())


@router.put("")
def put_defaults(
    payload: V4Defaults,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    user.v4_defaults = payload.model_dump_json()
    db.add(user); db.commit(); db.refresh(user)
    return {"ok": True, "defaults": json.loads(payload.model_dump_json())}


@router.get("/has")
def has_defaults(
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Cheap check for the NewJob wizard — when True it can offer a
    one-click 'Use my defaults' button that bypasses the per-step
    pickers entirely."""
    return {"has": bool(user.v4_defaults)}
