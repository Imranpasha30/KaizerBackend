"""Main SEO generation service: fetch news → call Gemini → enforce → save.

Called from a background thread. Writes the final enforced JSON to
`Clip.seo` on success. Retries transient errors with tenacity; terminal
errors propagate to the router which records them on the clip.
"""
from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone
from typing import Any, Dict

import google.generativeai as genai
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
)

import models
from seo import enforcer, news, prompts


GEMINI_MODEL = os.environ.get("KAIZER_SEO_MODEL", "gemini-2.5-flash")
# Fallback chain: tried in order when the primary returns quota/404 errors.
# Accuracy-first ordering that picks models with higher daily free quotas first
# so a single SEO run doesn't instantly burn the 20/day 2.5-flash limit.
_SEO_MODEL_CHAIN = [
    m.strip() for m in os.environ.get(
        "KAIZER_SEO_MODELS",
        "gemini-2.0-flash,gemini-2.0-flash-lite,gemini-2.5-flash-lite,gemini-2.5-flash",
    ).split(",") if m.strip()
]


# Response schema — Gemini's structured output feature enforces this, so the
# 6-strategy JSON-salvage from the extension is not needed here.
_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["title", "description", "keywords", "hashtags", "hook"],
    "properties": {
        "title":          {"type": "string"},
        "description":    {"type": "string"},
        "keywords":       {"type": "array", "items": {"type": "string"}},
        "hashtags":       {"type": "array", "items": {"type": "string"}},
        "hook":           {"type": "string"},
        "thumbnail_text": {"type": "string"},
        "metadata": {
            "type": "object",
            "properties": {
                "viral_score": {"type": "integer"},
                "sentiment":   {"type": "string"},
                "category":    {"type": "string"},
            },
        },
    },
}


class SEOGenerationError(Exception):
    """Terminal SEO generation error — propagates without retry."""


class TransientSEOError(SEOGenerationError):
    """Transient error (5xx, rate-limit) — tenacity retries this."""


class QuotaSEOError(SEOGenerationError):
    """Quota exhaustion (429) OR 404 model-not-found — caller falls through
    to the next model in the chain instead of retrying the same model.
    """


def _configure_gemini() -> None:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise SEOGenerationError("GEMINI_API_KEY is not set")
    genai.configure(api_key=key)


@retry(
    reraise=True,
    stop=stop_after_attempt(2),                              # retry one extra time, not three
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type(TransientSEOError),
)
def _try_one_model(model_name: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Single-model attempt. Raises QuotaSEOError on 429/404 so the caller
    can fall through to the next model in the chain instead of retrying.
    """
    try:
        model = genai.GenerativeModel(
            model_name,
            system_instruction=system_prompt,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": _RESPONSE_SCHEMA,
                "temperature": 0.9,
                "top_p": 0.95,
                "max_output_tokens": 4096,
            },
        )
        resp = model.generate_content(user_prompt)
        text = (resp.text or "").strip()
        if not text:
            raise TransientSEOError("Gemini returned empty response")
        data = json.loads(text)
        if not isinstance(data, dict):
            raise SEOGenerationError(f"Gemini returned non-object JSON: {type(data).__name__}")
        return data
    except json.JSONDecodeError as e:
        raise TransientSEOError(f"JSON decode failed: {e}") from e
    except (TransientSEOError, SEOGenerationError):
        raise
    except Exception as e:
        msg = str(e)
        low = msg.lower()
        # Quota / not-found → skip this model, caller will try the next one.
        if "429" in msg or "resourceexhausted" in low or "quota" in low \
                or "404" in msg or "not found" in low:
            raise QuotaSEOError(msg) from e
        # Real 5xx transient → tenacity retries in place
        if any(m in msg for m in ("500", "502", "503", "504")) or "unavailable" in low or "deadline" in low:
            raise TransientSEOError(f"transient: {msg}") from e
        raise SEOGenerationError(msg) from e


def _call_gemini(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Tries each model in the chain until one succeeds.  Primary is slowest
    to hit quota; fallbacks have larger free-tier pools.  Manual override:
    `KAIZER_SEO_MODELS=model_a,model_b,...` in .env.
    """
    chain = _SEO_MODEL_CHAIN or [GEMINI_MODEL]
    last_err: Exception | None = None
    for model_name in chain:
        try:
            return _try_one_model(model_name, system_prompt, user_prompt)
        except QuotaSEOError as e:
            print(f"[seo] {model_name}: quota/404 — trying next model. {str(e)[:120]}")
            last_err = e
            continue
        except SEOGenerationError as e:
            # Non-recoverable — stop here, no point trying fallbacks
            raise
    # Every model in the chain exhausted
    raise SEOGenerationError(
        f"All SEO models exhausted. Last error: {last_err}. "
        "Enable billing on your Google Cloud project or wait for daily reset."
    )


def _extract_topic(clip: models.Clip) -> str:
    """Best single phrase for Google News grounding."""
    try:
        meta = json.loads(clip.meta or "{}")
    except Exception:
        meta = {}
    for val in (
        (clip.text or "").strip(),
        (meta.get("summary_telugu") or "").strip(),
        (meta.get("summary") or "").strip(),
    ):
        if val:
            return val
    return ""


def generate_seo_for_clip(
    clip: models.Clip,
    channel: models.Channel,
    *,
    db,
    include_news: bool = True,
    include_corpus: bool = True,
) -> Dict[str, Any]:
    """End-to-end SEO generation for one clip. Persists to `clip.seo` on success."""
    _configure_gemini()

    # 1. News grounding (best-effort; never blocks the main call)
    news_items: list[dict] = []
    if include_news:
        try:
            topic = _extract_topic(clip)
            if topic:
                news_items = news.fetch_news_context(
                    topic, lang=channel.language or "te",
                )
        except Exception as e:
            print(f"[seo] news fetch failed (non-fatal): {e}")

    # 2. Channel corpus (Phase 7 — may be absent)
    corpus_payload = None
    if include_corpus and channel.corpus is not None:
        corpus_payload = channel.corpus.payload or None

    # 3. Prompts — include the publishing user's social links so Gemini can
    # embed them as a cross-promo footer in the description.
    socials: dict = {}
    try:
        # clip → job → user.  `clip.job` is a back-ref, `job.user` may be absent
        # in old records.  Look up defensively.
        owner_id = None
        if getattr(clip, "job", None) is not None:
            owner_id = getattr(clip.job, "user_id", None)
        if owner_id:
            u = db.query(models.User).filter(models.User.id == owner_id).first()
            if u and isinstance(u.socials, dict):
                socials = u.socials
    except Exception:
        socials = {}

    system_prompt = prompts.build_system_prompt(channel)
    user_prompt = prompts.build_user_prompt(
        clip=clip, channel=channel,
        news_items=news_items, corpus=corpus_payload,
        socials=socials,
    )

    # 4. Gemini call (with retries)
    raw = _call_gemini(system_prompt, user_prompt)

    # 5. Enforcement (30 tags, 100-char title, CamelCase hashtags, computed score)
    enforced = enforcer.enforce_quality(raw, channel)

    # 6. Attach bookkeeping
    enforced["channel_id"] = channel.id
    enforced["channel_name"] = channel.name
    enforced["generated_at"] = datetime.now(timezone.utc).isoformat()
    enforced["model"] = GEMINI_MODEL
    enforced["edited_by_user"] = False
    enforced["news_context"] = [
        {
            "title": n["title"],
            "source": n.get("source", ""),
            "link": n.get("link", ""),
        }
        for n in news_items
    ]

    # 7. Persist — store under this channel's key in seo_variants AND update
    # the legacy `clip.seo` field to this generation (for back-compat callers
    # that still read the "current" SEO).
    enforced_json = json.dumps(enforced, ensure_ascii=False)
    clip.seo = enforced_json
    try:
        variants = json.loads(clip.seo_variants or "{}")
        if not isinstance(variants, dict):
            variants = {}
    except (ValueError, TypeError):
        variants = {}
    variants[str(channel.id)] = enforced
    clip.seo_variants = json.dumps(variants, ensure_ascii=False)
    db.commit()
    try:
        db.refresh(clip)
    except Exception:
        pass

    return enforced
