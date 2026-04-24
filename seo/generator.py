"""Research → Generate → Verify → Retry pipeline for GENERIC (brand-agnostic) SEO.

Flow per clip:
  1. Research (parallel, best-effort):
     - Google News topical context       (seo.news)
     - Google Trends keywords             (seo.trends, pytrends)
     - YouTube top-5 videos for topic     (seo.yt_benchmark)
  2. Generate via Gemini:
     - System prompt: channel-agnostic, target ≥95 score
     - User prompt: clip facts + all research layers
     - Structured output (response_schema)
  3. Sanitize: strip any style_source brand leaks
  4. Verify: deterministic 0-100 score (seo.verifier)
  5. If score < target: feed verifier.reasons back into prompt, retry
     up to N times.  Track best attempt — always return the highest-scoring.

The output is stored on `clip.seo` as GENERIC SEO.  At publish time, the
composer overlays the destination channel's brand onto this generic base.
"""
from __future__ import annotations

import json
import os
import re
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List

import google.generativeai as genai
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
)

import models
from seo import news, prompts, trends, yt_benchmark, verifier, sanitizer
from learning.gemini_log import log_gemini_call


GEMINI_MODEL = os.environ.get("KAIZER_SEO_MODEL", "gemini-2.5-flash")
_SEO_MODEL_CHAIN = [
    m.strip() for m in os.environ.get(
        "KAIZER_SEO_MODELS",
        "gemini-2.5-flash,gemini-2.5-flash-lite",
    ).split(",") if m.strip()
]

# Retry loop targets
TARGET_SCORE     = int(os.environ.get("KAIZER_SEO_TARGET_SCORE", "95"))
MAX_RETRIES      = int(os.environ.get("KAIZER_SEO_MAX_RETRIES", "4"))   # 1 initial + up to 4 retries


_TRAILING_SUFFIX_RE = re.compile(r"\s*\|\s*[^|]+$")

def _strip_trailing_suffix(title: str) -> str:
    """Remove any " | Xxx" tail from the title.

    Generic SEO must NOT carry a channel suffix — branding is injected at
    publish time by the composer.  Even with firm prompting Gemini sometimes
    still emits one; this post-pass is the deterministic safety net.
    """
    if not title:
        return title
    return _TRAILING_SUFFIX_RE.sub("", title).strip(" |\t")


# ── Response schema (generic SEO — no channel_id, no footer baked in) ────────

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


# ── Error classes ────────────────────────────────────────────────────────────

class SEOGenerationError(Exception):
    """Terminal SEO generation error — propagates without retry."""

class TransientSEOError(SEOGenerationError):
    """Transient (5xx / rate-limit) — tenacity retries the same call."""

class QuotaSEOError(SEOGenerationError):
    """Quota / 404 — caller should try the next model in the chain."""


def _configure_gemini() -> None:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise SEOGenerationError("GEMINI_API_KEY is not set")
    genai.configure(api_key=key)


# ── Single-model attempt + model-chain fallback ──────────────────────────────

@retry(
    reraise=True,
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type(TransientSEOError),
)
def _try_one_model(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    *,
    db=None,
    user_id=None,
    job_id=None,
    clip_id=None,
) -> Dict[str, Any]:
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
        with log_gemini_call(
            db=db, user_id=user_id, job_id=job_id, clip_id=clip_id,
            model=model_name, purpose="seo",
        ) as _gcall:
            resp = model.generate_content(user_prompt)
            _gcall.record(resp)
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
        msg = str(e); low = msg.lower()
        if "429" in msg or "resourceexhausted" in low or "quota" in low \
                or "404" in msg or "not found" in low:
            raise QuotaSEOError(msg) from e
        if any(m in msg for m in ("500", "502", "503", "504")) or "unavailable" in low or "deadline" in low:
            raise TransientSEOError(f"transient: {msg}") from e
        raise SEOGenerationError(msg) from e


def _call_gemini(
    system_prompt: str,
    user_prompt: str,
    *,
    db=None,
    user_id=None,
    job_id=None,
    clip_id=None,
) -> tuple[Dict[str, Any], str]:
    chain = _SEO_MODEL_CHAIN or [GEMINI_MODEL]
    last_err: Exception | None = None
    for model_name in chain:
        try:
            raw = _try_one_model(
                model_name, system_prompt, user_prompt,
                db=db, user_id=user_id, job_id=job_id, clip_id=clip_id,
            )
            return raw, model_name
        except QuotaSEOError as e:
            print(f"[seo] {model_name}: quota/404 — next model. {str(e)[:120]}")
            last_err = e
            continue
        except SEOGenerationError:
            raise
    raise SEOGenerationError(
        f"All SEO models exhausted. Last error: {last_err}. "
        "Enable billing or wait for daily reset."
    )


# ── Topic extraction ─────────────────────────────────────────────────────────

def _extract_topic(clip: models.Clip) -> str:
    try:
        meta = json.loads(clip.meta or "{}")
    except Exception:
        meta = {}
    for val in (
        (clip.text or "").strip(),
        (meta.get("summary") or "").strip(),
        (meta.get("summary_telugu") or meta.get("summary_native") or "").strip(),
    ):
        if val:
            return val
    return ""


def _fallback_keyword_seed(topic: str, trend_kw: List[str]) -> List[str]:
    """Deterministic keyword backup so verifier's keyword dimension can pass
    even if Gemini forgets a term — used only in the final retry."""
    seed = [k.lower() for k in trend_kw if k]
    for tok in (topic or "").split():
        tok = tok.strip().lower()
        if len(tok) >= 3 and tok not in seed:
            seed.append(tok)
    return seed[:10]


# ── Public entry ─────────────────────────────────────────────────────────────

def generate_seo_for_clip(
    clip: models.Clip,
    *,
    db,
    style_source: models.Channel | None = None,
    include_news: bool = True,
    include_trends: bool = True,
    include_yt_benchmark: bool = True,
    language: str = "te",
    progress_cb=None,
) -> Dict[str, Any]:
    """End-to-end GENERIC SEO generation for one clip.

    - `language` — target language for SEO (defaults to clip's language via
      clip.job.language, overridden by explicit arg).
    - `style_source` — optional Channel whose corpus + title-formula-style
      voice teaches Gemini how to write.  Its branding is NOT injected.
    - `progress_cb(stage, info)` — optional callback for UI status updates.

    Persists the top-scoring attempt to `clip.seo` and returns the same dict.
    """
    _configure_gemini()

    def tick(stage: str, info: Any = None) -> None:
        if progress_cb:
            try: progress_cb(stage, info)
            except Exception: pass

    # Resolve language from clip's job if not overridden
    if clip.job and getattr(clip.job, "language", None):
        language = clip.job.language or language

    topic = _extract_topic(clip)
    tick("research", "news + trends + yt benchmark")

    # ── 1. Research phase (all best-effort, all independent) ──
    news_items: List[Dict[str, Any]] = []
    if include_news:
        try:
            if topic:
                news_items = news.fetch_news_context(topic, lang=language)
        except Exception as e:
            print(f"[seo] news fetch failed: {e}")

    trend_data: Dict[str, Any] = {
        "trending_now": [], "related_queries": [], "rising_queries": [],
        "source": "disabled",
    }
    if include_trends:
        try:
            trend_data = trends.fetch_trending_keywords(topic, lang=language)
        except Exception as e:
            print(f"[seo] trends fetch failed: {e}")

    # Combine all trend keyword surfaces for the verifier
    all_trend_keywords = list({
        *(trend_data.get("related_queries") or []),
        *(trend_data.get("rising_queries") or []),
        *(trend_data.get("trending_now") or []),
    })

    yt_top: List[Dict[str, Any]] = []
    if include_yt_benchmark:
        try:
            yt_top = yt_benchmark.fetch_top_videos(topic, lang=language, max_results=5)
        except Exception as e:
            print(f"[seo] yt benchmark failed: {e}")

    # Corpus (style voice) — from style_source if provided
    corpus_payload = None
    if style_source and style_source.corpus is not None:
        corpus_payload = style_source.corpus.payload or None

    # ── 2. Prompts (base, no retry feedback yet) ──
    system_prompt = prompts.build_system_prompt(
        language=language, style_source=style_source, target_score=TARGET_SCORE,
    )

    def _build_user(retry_feedback=None) -> str:
        return prompts.build_user_prompt(
            clip=clip, language=language,
            news_items=news_items,
            trends=trend_data,
            yt_top=yt_top,
            corpus=corpus_payload,
            style_source=style_source,
            retry_feedback=retry_feedback,
        )

    # ── 3-5. Generate → sanitize → verify → retry loop ──
    best: Dict[str, Any] | None = None
    best_score = -1
    best_report: Dict[str, Any] | None = None
    model_used = GEMINI_MODEL
    attempts_log: List[Dict[str, Any]] = []

    retry_feedback: List[str] = []
    total_rounds = MAX_RETRIES + 1   # 1 initial + MAX_RETRIES retries

    for attempt in range(1, total_rounds + 1):
        tick("generate", f"attempt {attempt}/{total_rounds}")
        user_prompt = _build_user(retry_feedback=retry_feedback if attempt > 1 else None)

        try:
            _job = getattr(clip, "job", None)
            raw, model_used = _call_gemini(
                system_prompt, user_prompt,
                db=db,
                user_id=getattr(_job, "user_id", None) if _job else None,
                job_id=getattr(clip, "job_id", None),
                clip_id=getattr(clip, "id", None),
            )
        except SEOGenerationError as e:
            # Hard failure — if we already have a best, ship it; else propagate
            if best:
                print(f"[seo] attempt {attempt} exhausted models; keeping best={best_score}")
                break
            raise

        # Deterministic safety net: strip any " | Suffix" Gemini tacked on
        # (it is told NOT to, but the fix is mechanical so we don't burn a
        # retry round on it).  Also strips before sanitizer so brand leaks
        # inside the suffix still get caught.
        raw["title"] = _strip_trailing_suffix(raw.get("title", ""))

        cleaned = sanitizer.sanitize(raw, style_source)
        report = verifier.verify(
            cleaned,
            clip_topic=topic,
            trend_keywords=all_trend_keywords,
            news_items=news_items,
        )
        attempts_log.append({
            "attempt": attempt,
            "score":   report["score"],
            "reasons": report["reasons"][:6],
        })
        print(f"[seo] attempt {attempt}: score={report['score']}, fails={len(report['reasons'])}")

        if report["score"] > best_score:
            best_score = report["score"]
            best = cleaned
            best_report = report

        if report["score"] >= TARGET_SCORE:
            tick("verified", {"score": report["score"], "attempt": attempt})
            break

        # Prepare retry feedback for next round
        retry_feedback = report["reasons"]

    if not best:
        raise SEOGenerationError("SEO generation produced no candidates")

    # ── 6. Attach bookkeeping + persist ──
    best["seo_score"]   = best_score
    best["verifier_breakdown"] = (best_report or {}).get("breakdown") or {}
    best["verifier_reasons"]   = (best_report or {}).get("reasons") or []
    best["generated_at"] = datetime.now(timezone.utc).isoformat()
    best["model"]        = model_used
    best["edited_by_user"] = False
    best["style_source_id"]   = style_source.id   if style_source else None
    best["style_source_name"] = style_source.name if style_source else None
    best["attempts_log"]      = attempts_log
    best["target_score"]      = TARGET_SCORE
    best["news_context"] = [
        {"title": n["title"], "source": n.get("source", ""), "link": n.get("link", "")}
        for n in news_items
    ]
    best["trending_keywords"] = all_trend_keywords[:15]
    best["yt_benchmark"] = [
        {"title": v["title"], "views": v["views"], "channel": v.get("channel", "")}
        for v in yt_top
    ]

    # Persist as the single canonical generic SEO on the clip.  The legacy
    # per-channel `seo_variants` field is left untouched (read-only legacy).
    clip.seo = json.dumps(best, ensure_ascii=False)
    db.commit()
    try: db.refresh(clip)
    except Exception: pass

    return best
