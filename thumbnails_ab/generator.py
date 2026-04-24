"""Register thumbnail variants for an upload.

v1: variant 0 is the existing clip.thumb_path (what we already upload).
    variant 1 is the same image with a Gemini-picked hook text tagged for
    later overlay. For MVP we keep both pointing at the same base image —
    the swap test validates whether YT thumbnail changes drive views even
    with the same frame. v2 can add real different frames + overlays.
"""
from __future__ import annotations

import json
import os
from typing import List

import google.generativeai as genai
from sqlalchemy.orm import Session

import models
from config import settings
from learning.gemini_log import log_gemini_call


def _pick_hook_texts(clip: models.Clip) -> List[str]:
    """Two short hook-text candidates (<=35 chars each)."""
    default = [
        (clip.text or "BREAKING").strip()[:35],
        "WATCH NOW",
    ]
    if not settings.gemini_api_key:
        return default
    try:
        genai.configure(api_key=settings.gemini_api_key)
        seo = {}
        if clip.seo:
            try: seo = json.loads(clip.seo)
            except Exception: seo = {}
        topic = (seo.get("title") or clip.text or "").strip()[:200]
        if not topic:
            return default
        model = genai.GenerativeModel(
            os.environ.get("KAIZER_THUMB_MODEL", "gemini-2.5-flash"),
            system_instruction=(
                "You write ultra-short YouTube thumbnail overlays for Telugu news. "
                "Return STRICT JSON: {\"variants\":[\"…\", \"…\"]}. "
                "Each variant: max 5 words, max 35 chars, ALL CAPS, high-CTR. "
                "Variant 0 is urgent/dramatic. Variant 1 is curiosity/controversy. "
                "No emojis. No quotes. No punctuation at end."
            ),
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "required": ["variants"],
                    "properties": {
                        "variants": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                },
                "temperature": 1.1,
                "max_output_tokens": 200,
            },
        )
        _thumb_model = os.environ.get("KAIZER_THUMB_MODEL", "gemini-2.5-flash")
        with log_gemini_call(
            db=None, clip_id=getattr(clip, "id", None),
            job_id=getattr(clip, "job_id", None),
            model=_thumb_model, purpose="thumbnail",
        ) as _gcall:
            resp = model.generate_content(f"Topic: {topic}")
            _gcall.record(resp)
        data = json.loads((resp.text or "").strip())
        vs = [(v or "").strip()[:35] for v in (data.get("variants") or []) if v]
        if len(vs) >= 2:
            return vs[:2]
    except Exception as e:
        print(f"[thumb-ab] Gemini hook-text failed: {e}")
    return default


def register_variants(db: Session, upload_job_id: int, clip: models.Clip) -> List[models.ThumbnailVariant]:
    """Create variant-0 (primary) and variant-1 (alt) rows for this upload."""
    existing = (
        db.query(models.ThumbnailVariant)
          .filter(models.ThumbnailVariant.upload_job_id == upload_job_id)
          .all()
    )
    if existing:
        return existing

    hooks = _pick_hook_texts(clip)
    rows = []
    for i, hook in enumerate(hooks[:2]):
        v = models.ThumbnailVariant(
            upload_job_id=upload_job_id,
            variant_idx=i,
            image_path=clip.thumb_path or "",
            hook_text=hook,
            status="pending" if i == 0 else "alternate",
        )
        db.add(v)
        rows.append(v)
    db.commit()
    return rows
