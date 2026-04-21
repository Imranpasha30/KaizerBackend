"""SEO router — per-clip GENERIC (channel-agnostic) SEO generation.

Flow (post Content + Brand Overlay refactor):
  - ONE Gemini-powered generation per clip, produces a GENERIC SEO JSON.
  - At publish time, `seo.composer` overlays each destination's brand onto
    the generic SEO to produce the actual uploaded title/description/tags.

The old per-destination multi-variant endpoint surface is preserved (it's
still used by the older PublishModal code paths) but now it all routes
through the same generic generator — `channel_ids` and per-destination
overrides are quietly ignored for generation.
"""
from __future__ import annotations

import json
import threading
import traceback
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import SessionLocal, get_db
import models
import auth
from seo import generator


router = APIRouter(prefix="/api", tags=["seo"])


# ─── Request schemas ──────────────────────────────────────────────────────────

class GenerateSEORequest(BaseModel):
    # Optional: learn writing RHYTHM from this channel's corpus + title formula.
    # Its branding never appears in the output (sanitizer + verifier enforce).
    style_source_id: Optional[int] = None
    force: bool = False
    include_news: bool = True
    include_trends: bool = True
    include_yt_benchmark: bool = True

    # Legacy fields — accepted but ignored post-refactor (keeps older frontend
    # builds from 422-ing while they're still deploying).
    channel_id:  Optional[int] = None
    channel_ids: Optional[List[int]] = None


class UpdateSEORequest(BaseModel):
    title: Optional[str] = Field(None, max_length=150)
    description: Optional[str] = None
    keywords: Optional[List[str]] = None
    hashtags: Optional[List[str]] = None
    hook: Optional[str] = None
    thumbnail_text: Optional[str] = None


# ─── In-process status tracker ────────────────────────────────────────────────

_generation_status: dict[int, str] = {}   # clip_id -> "idle" | "generating ..." | "done" | "error: ..."
_generation_lock = threading.Lock()


def _set_status(clip_id: int, status: str) -> None:
    with _generation_lock:
        _generation_status[clip_id] = status


def _get_status(clip_id: int) -> str:
    with _generation_lock:
        return _generation_status.get(clip_id, "idle")


def _run_generate(
    clip_id: int,
    include_news: bool,
    include_trends: bool,
    include_yt_benchmark: bool,
    style_source_id: Optional[int] = None,
) -> None:
    """Background worker — 1 generic SEO per clip with retry-until-target loop."""
    db = SessionLocal()
    try:
        clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
        if not clip:
            _set_status(clip_id, "error: clip not found")
            return

        style_source = None
        if style_source_id:
            style_source = (
                db.query(models.Channel)
                  .filter(models.Channel.id == style_source_id)
                  .first()
            )
            if not style_source:
                print(f"[seo] style_source {style_source_id} not found — using self-style")

        def _progress(stage: str, info) -> None:
            label = f"{stage}"
            if isinstance(info, str) and info:
                label = f"{stage}: {info}"
            elif isinstance(info, dict):
                label = f"{stage}: " + ", ".join(f"{k}={v}" for k, v in info.items())
            _set_status(clip_id, label)

        _set_status(clip_id, "generating: starting")
        generator.generate_seo_for_clip(
            clip,
            db=db,
            style_source=style_source,
            include_news=include_news,
            include_trends=include_trends,
            include_yt_benchmark=include_yt_benchmark,
            progress_cb=_progress,
        )
        _set_status(clip_id, "done")
    except Exception as e:
        traceback.print_exc()
        _set_status(clip_id, f"error: {str(e)[:200]}")
    finally:
        db.close()


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/clips/{clip_id}/seo/generate")
def generate_clip_seo(
    clip_id: int,
    payload: GenerateSEORequest,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Kick off GENERIC SEO generation for this clip.

    A single background call produces one channel-agnostic SEO that will be
    reused across every destination at publish time.  The optional
    `style_source_id` teaches Gemini a writing voice; no branding from
    that channel is ever emitted.
    """
    clip = (
        db.query(models.Clip)
          .join(models.Job, models.Clip.job_id == models.Job.id)
          .filter(models.Clip.id == clip_id, models.Job.user_id == user.id)
          .first()
    )
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    # Validate optional style source belongs to this user
    if payload.style_source_id is not None:
        src = (
            db.query(models.Channel)
              .filter(
                  models.Channel.id == payload.style_source_id,
                  models.Channel.user_id == user.id,
              )
              .first()
        )
        if not src:
            raise HTTPException(
                status_code=404,
                detail=f"Style-source channel {payload.style_source_id} not found",
            )

    # Refuse to regenerate if SEO already exists AND force flag not set.
    if not payload.force and clip.seo:
        try:
            existing = json.loads(clip.seo)
        except (ValueError, TypeError):
            existing = {}
        if isinstance(existing, dict) and existing.get("title"):
            raise HTTPException(
                status_code=409,
                detail="SEO already exists for this clip. Pass force=true to regenerate.",
            )

    if _get_status(clip_id).startswith("generating"):
        return {"status": "generating", "clip_id": clip_id}

    _set_status(clip_id, "generating: queued")
    threading.Thread(
        target=_run_generate,
        args=(
            clip_id,
            payload.include_news,
            payload.include_trends,
            payload.include_yt_benchmark,
            payload.style_source_id,
        ),
        daemon=True,
    ).start()

    return {
        "status": "generating",
        "clip_id": clip_id,
        "style_source_id": payload.style_source_id,
    }


@router.get("/clips/{clip_id}/seo/status")
def get_seo_status(clip_id: int, db: Session = Depends(get_db)):
    clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    status = _get_status(clip_id)
    seo = None
    if clip.seo:
        try:
            seo = json.loads(clip.seo)
        except (ValueError, TypeError):
            seo = None

    return {"clip_id": clip_id, "status": status, "seo": seo}


@router.put("/clips/{clip_id}/seo")
def update_clip_seo(
    clip_id: int,
    payload: UpdateSEORequest,
    db: Session = Depends(get_db),
):
    """Manual edit — merge user changes into the generic SEO JSON and
    recompute the deterministic verifier score."""
    clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    if not clip.seo:
        raise HTTPException(
            status_code=409,
            detail="No SEO on this clip yet — generate first",
        )

    try:
        seo = json.loads(clip.seo)
    except (ValueError, TypeError):
        raise HTTPException(status_code=500, detail="Existing SEO is corrupt — regenerate")

    updates = payload.model_dump(exclude_unset=True)
    for key, val in updates.items():
        seo[key] = val

    # Re-score with the deterministic verifier
    from seo import verifier as _verifier
    report = _verifier.verify(
        seo,
        clip_topic=(clip.text or ""),
        trend_keywords=seo.get("trending_keywords") or [],
        news_items=seo.get("news_context") or [],
    )
    seo["seo_score"]           = report["score"]
    seo["verifier_breakdown"]  = report["breakdown"]
    seo["verifier_reasons"]    = report["reasons"]
    seo["edited_by_user"]      = True
    seo["edited_at"]           = datetime.now(timezone.utc).isoformat()

    clip.seo = json.dumps(seo, ensure_ascii=False)
    db.commit()

    return {"clip_id": clip_id, "seo": seo}


@router.delete("/clips/{clip_id}/seo")
def clear_clip_seo(clip_id: int, db: Session = Depends(get_db)):
    clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    clip.seo = ""
    db.commit()
    _set_status(clip_id, "idle")
    return {"clip_id": clip_id, "cleared": True}


# ─── Compose-preview for PublishModal (pure function, no side effects) ────────

@router.get("/clips/{clip_id}/seo/compose-preview")
def preview_composed_seo(
    clip_id: int,
    channel_id: int,
    publish_kind: str = "video",
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Return the exact title/description/tags that would be uploaded to
    `channel_id` for this clip — generic SEO + brand overlay from that channel.

    Used by PublishModal to show the user a destination-specific preview
    BEFORE confirming publish.
    """
    clip = (
        db.query(models.Clip)
          .join(models.Job, models.Clip.job_id == models.Job.id)
          .filter(models.Clip.id == clip_id, models.Job.user_id == user.id)
          .first()
    )
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    if not clip.seo:
        raise HTTPException(status_code=409, detail="No SEO on this clip yet — generate first")

    dest = db.query(models.Channel).filter(
        models.Channel.id == channel_id, models.Channel.user_id == user.id,
    ).first()
    if not dest:
        raise HTTPException(status_code=404, detail="Destination channel not found")

    try:
        generic = json.loads(clip.seo)
    except (ValueError, TypeError):
        raise HTTPException(status_code=500, detail="Existing SEO is corrupt — regenerate")

    from seo.composer import compose, assert_no_foreign_brand
    composed = compose(generic, dest, publish_kind=publish_kind)

    # Safety audit — lists any rival brand that leaked despite sanitizer
    other_channels = (
        db.query(models.Channel)
          .filter(models.Channel.user_id == user.id, models.Channel.id != dest.id)
          .all()
    )
    warnings = assert_no_foreign_brand(composed, dest, other_channels)

    return {
        "clip_id":    clip_id,
        "channel_id": dest.id,
        "channel_name": dest.name,
        "composed":   composed,
        "leak_warnings": warnings,
    }
