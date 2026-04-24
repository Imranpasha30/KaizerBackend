"""Gemini-powered SEO translation + sibling upload fan-out.

ensure_translation(clip_id, lang, target_channel_id) does:
  1. Look up the clip's SEO; translate title/description/tags/hashtags
     into `lang` with Gemini; cache in ClipTranslation row.
  2. Create an UploadJob on target_channel_id using the translated metadata
     and the same clip.file_path.

Idempotent — second call on same (clip, lang) just reuses the cached
translation and skips upload creation if one is already queued/done.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict

import google.generativeai as genai
from sqlalchemy.orm import Session
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
)

import models
from config import settings
from learning.gemini_log import log_gemini_call


TRANSLATE_MODEL = os.environ.get("KAIZER_TRANSLATE_MODEL", "gemini-2.5-flash")


class TranslationError(Exception):
    pass


class TransientTranslationError(TranslationError):
    pass


_LANG_NAMES = {
    "te": "Telugu",
    "hi": "Hindi",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "en": "English",
}

_SCHEMA = {
    "type": "object",
    "required": ["title", "description", "keywords", "hashtags"],
    "properties": {
        "title":       {"type": "string"},
        "description": {"type": "string"},
        "keywords":    {"type": "array", "items": {"type": "string"}},
        "hashtags":    {"type": "array", "items": {"type": "string"}},
        "hook":        {"type": "string"},
    },
}


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(TransientTranslationError),
)
def _translate(seo: Dict, target_lang: str) -> Dict:
    if not settings.gemini_api_key:
        raise TranslationError("GEMINI_API_KEY not set")
    genai.configure(api_key=settings.gemini_api_key)
    lang_name = _LANG_NAMES.get(target_lang, target_lang.upper())

    title = (seo.get("title") or "").strip()
    desc  = (seo.get("description") or "").strip()
    tags  = seo.get("keywords") or []
    hashs = seo.get("hashtags") or []
    hook  = (seo.get("hook") or "").strip()

    src = json.dumps(
        {"title": title, "description": desc, "keywords": tags, "hashtags": hashs, "hook": hook},
        ensure_ascii=False,
    )

    try:
        model = genai.GenerativeModel(
            TRANSLATE_MODEL,
            system_instruction=(
                f"You are a professional YouTube localizer. Translate the given SEO JSON "
                f"from its source language into {lang_name}. Preserve proper nouns, brand "
                f"names, and #hashtag CamelCase conventions. Keep hashtags as hashtags. "
                f"Keep keyword count similar. Title must stay under 100 characters after "
                f"translation. Return STRICT JSON matching the schema."
            ),
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": _SCHEMA,
                "temperature": 0.4,
                "max_output_tokens": 4096,
            },
        )
        with log_gemini_call(
            db=None, model=TRANSLATE_MODEL, purpose="translation",
        ) as _gcall:
            resp = model.generate_content(src)
            _gcall.record(resp)
        text = (resp.text or "").strip()
        if not text:
            raise TransientTranslationError("empty response")
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise TransientTranslationError(f"JSON decode: {e}") from e
    except (TransientTranslationError, TranslationError):
        raise
    except Exception as e:
        msg = str(e).lower()
        if any(m in msg for m in ("503", "500", "502", "504", "rate", "deadline", "quota", "unavailable")):
            raise TransientTranslationError(f"transient: {e}") from e
        raise TranslationError(str(e)) from e


def ensure_translation(db: Session, clip_id: int, lang: str, target_channel_id: int) -> Dict:
    """Translate (cached) + create an UploadJob on the target channel."""
    clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
    if not clip or not clip.seo:
        raise TranslationError(f"Clip {clip_id} has no SEO to translate")
    if not clip.file_path:
        raise TranslationError(f"Clip {clip_id} has no rendered file")
    try:
        seo = json.loads(clip.seo)
    except Exception:
        raise TranslationError(f"Clip {clip_id} SEO is not valid JSON")

    target = db.query(models.Channel).filter(models.Channel.id == target_channel_id).first()
    if not target:
        raise TranslationError(f"Target channel {target_channel_id} not found")
    if not target.oauth_token or not target.oauth_token.refresh_token_enc:
        raise TranslationError(f"Target channel {target.name} is not connected to YouTube")

    # 1. Cache-first
    row = (
        db.query(models.ClipTranslation)
          .filter(models.ClipTranslation.clip_id == clip_id,
                  models.ClipTranslation.language == lang)
          .first()
    )
    if row and row.payload:
        payload = row.payload
    else:
        payload = _translate(seo, lang)
        if row:
            row.payload = payload
        else:
            row = models.ClipTranslation(clip_id=clip_id, language=lang, payload=payload)
            db.add(row)
        db.commit()

    # 2. Queue upload on the target channel — skip if one already exists.
    existing = db.query(models.UploadJob).filter(
        models.UploadJob.clip_id == clip_id,
        models.UploadJob.channel_id == target_channel_id,
        models.UploadJob.status.in_(["queued", "uploading", "processing", "done"]),
    ).first()
    if existing:
        return {"clip_id": clip_id, "language": lang,
                "upload_job_id": existing.id, "created": False, "payload": payload}

    title = (payload.get("title") or "")[:100]
    desc  = payload.get("description") or ""
    tags  = list(payload.get("keywords") or [])[:30]

    up = models.UploadJob(
        clip_id=clip_id,
        channel_id=target_channel_id,
        status="queued",
        privacy_status="private",
        title=title,
        description=desc,
        tags=tags,
        category_id="25",
    )
    db.add(up); db.commit(); db.refresh(up)
    return {"clip_id": clip_id, "language": lang,
            "upload_job_id": up.id, "created": True, "payload": payload}
