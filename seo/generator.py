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
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

from google import genai
from google.genai import types as genai_types
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


def _gemini_client() -> "genai.Client":
    """Return a fresh google.genai client.

    Auth precedence (matches the rest of the V4 pipeline so SEO bills
    to the same account as thumbnail / image generation):

      1. KAIZER_GCP_PROJECT set  → Vertex AI mode. Spends GCP credits
         (e.g. GenAI App Builder / Free Trial). The Vertex SA is loaded
         from KAIZER_VERTEX_CREDENTIALS explicitly so the SDK doesn't
         pick up the wrong default project from
         GOOGLE_APPLICATION_CREDENTIALS (the STT pipeline's SA).
      2. KAIZER_GEMINI_API_KEY / GEMINI_API_KEY set → AI Studio mode.
         This is the depleted-prepay path that triggered the
         "All SEO models exhausted: 429 RESOURCE_EXHAUSTED" loop;
         only used when the operator hasn't configured Vertex.

    Without either, SEO generation is impossible — raise a clear error
    that names BOTH paths so the operator knows which env var to set.
    """
    project = (os.environ.get("KAIZER_GCP_PROJECT") or "").strip()
    if project:
        location = (os.environ.get("KAIZER_GCP_LOCATION") or "us-central1").strip()
        vertex_creds_path = (os.environ.get("KAIZER_VERTEX_CREDENTIALS") or "").strip()
        explicit_creds = None
        if vertex_creds_path and os.path.isfile(vertex_creds_path):
            try:
                from google.oauth2 import service_account
                explicit_creds = service_account.Credentials.from_service_account_file(
                    vertex_creds_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
            except Exception as exc:
                print(f"[seo] failed to load Vertex SA from {vertex_creds_path}: {exc}", flush=True)
        return genai.Client(
            vertexai=True, project=project, location=location,
            credentials=explicit_creds,
        )

    key = (os.environ.get("KAIZER_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        raise SEOGenerationError(
            "No Gemini auth for SEO. Either:\n"
            "  - set KAIZER_GCP_PROJECT (recommended — uses GCP credits via Vertex AI), OR\n"
            "  - set GEMINI_API_KEY (AI Studio mode)."
        )
    return genai.Client(api_key=key)


# Backwards-compatibility alias for any caller that imports the old
# name. `_configure_gemini()` no longer mutates global state — the
# returned client is the one to use. Existing call sites that do
# `_configure_gemini(); model = genai.GenerativeModel(...)` need to
# be updated to `client = _gemini_client(); client.models.generate_content(...)`.
def _configure_gemini() -> "genai.Client":
    return _gemini_client()


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
        client = _gemini_client()
        cfg = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
            temperature=0.9,
            top_p=0.95,
            # Bumped from 4096 → 8192. Telugu/Hindi tokens are denser
            # than English (1 char ≈ 2-3 tokens), and the SEO payload
            # is title + long description + 30 keywords + 12 hashtags
            # + hook + ad copy. 4096 was getting clipped mid-string,
            # producing "Unterminated string" JSONDecodeError. 8192 is
            # the hard cap for gemini-2.0/2.5-flash; safe on all
            # models in our fallback chain.
            max_output_tokens=8192,
        )
        with log_gemini_call(
            db=db, user_id=user_id, job_id=job_id, clip_id=clip_id,
            model=model_name, purpose="seo",
        ) as _gcall:
            resp = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=cfg,
            )
            _gcall.record(resp)
        text = (resp.text or "").strip()
        if not text:
            raise TransientSEOError("Gemini returned empty response")

        # If Gemini hit max_output_tokens mid-stream, the JSON will be
        # truncated (unterminated string). Treat that as TRANSIENT so
        # the @retry decorator gives us another shot — sometimes the
        # second attempt comes in shorter and parses cleanly.
        try:
            _finish_reason = (
                resp.candidates[0].finish_reason.name
                if resp.candidates and resp.candidates[0].finish_reason else ""
            )
        except Exception:
            _finish_reason = ""
        if _finish_reason == "MAX_TOKENS":
            raise TransientSEOError(
                "Gemini hit max_output_tokens mid-stream (truncated JSON). "
                "Will retry — bump max_output_tokens further if this keeps happening."
            )
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


def _call_claude_seo(system_prompt: str, user_prompt: str,
                     model: str = "") -> tuple[Dict[str, Any], str]:
    """Anthropic writer path (user-selectable engine). Same prompts, same
    downstream sanitize/verify/retry — only the writer differs. Claude has
    no response-schema enforcement, so we demand raw JSON and parse
    leniently; any failure raises and the caller falls back to Gemini so
    engine choice can never break SEO."""
    from anthropic import Anthropic
    from pipeline_v4.trim_engine import _loads_lenient
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    model = model or os.environ.get("KAIZER_SEO_CLAUDE_MODEL",
                                    "claude-sonnet-4-6")
    client = Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model, max_tokens=8192,
        system=(system_prompt
                + "\n\nReturn ONLY the JSON object — no prose, no fences."),
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = (msg.content[0].text if msg.content else "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    raw = _loads_lenient(text)
    if not isinstance(raw, dict) or not raw.get("title"):
        raise RuntimeError("claude SEO returned no usable JSON object")
    return raw, model


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
    own_channel: models.Channel | None = None,
    include_news: bool = True,
    include_trends: bool = True,
    include_yt_benchmark: bool = True,
    language: str = "te",
    avoid_titles: list | None = None,
    angle_hint: str | None = None,
    engine_choice: str | None = None,
    persist: bool = True,
    progress_cb=None,
) -> Dict[str, Any]:
    """End-to-end GENERIC SEO generation for one clip.

    - `language` — target language for SEO (defaults to clip's language via
      clip.job.language, overridden by explicit arg).
    - `style_source` — optional Channel whose corpus + title-formula-style
      voice teaches Gemini how to write.  Its branding is NOT injected.
    - `own_channel` — the OWN publishing account whose MEASURED performance
      drives learned policy / dedupe / competitor intel. When None we fall
      back to `style_source` (legacy behaviour). Set this on a per-channel
      publish so the channel's own learning steers its SEO.
    - `avoid_titles` — sibling per-channel titles to differ from (joins the
      dedupe pool so cross-channel variants stay distinct).
    - `angle_hint` — a structural angle for THIS variant (per-channel).
    - `engine_choice` — 'gemini' | 'claude' override (else env / user pref).
    - `persist` — write the result to `clip.seo` + commit. False for a
      synthetic clip (delegation) where the caller owns persistence.
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

    # Learned policy (learning/seo_learning.py) — the channel's MEASURED
    # hook/script/length/keyword performance, refreshed by the analytics
    # poll. None until the channel has >=5 sampled videos; then every
    # generation is steered by what actually earned views/CTR there.
    learned_policy = None
    recent_titles: list = []
    explore_hook = None
    explored = False
    # The channel whose MEASURED YouTube performance drives learned policy /
    # dedupe / competitor intel: the OWN publishing account when known, else
    # the style_source (legacy). Its corpus/voice still comes from style_source.
    policy_channel = own_channel or style_source
    if policy_channel is not None and getattr(policy_channel, "id", None):
        try:
            from sqlalchemy.orm import object_session
            from learning.seo_learning import (latest_policy, pick_hook,
                                               merged_rows)
            # NOTE: use the MODULE-level datetime/timezone/timedelta — a local
            # `from datetime import ...` here would make `datetime` function-
            # local and crash the channel-less path (UnboundLocalError at the
            # bookkeeping `datetime.now()` when this block is skipped).
            _ls = object_session(policy_channel)
            if _ls is not None:
                learned_policy = latest_policy(_ls, policy_channel.id)
                # DEDUPE GUARD input: this channel's recent titles — a
                # candidate matching one becomes a retry failure (two
                # identical published titles happened; never again).
                _since = datetime.now(timezone.utc) - timedelta(days=60)
                recent_titles = [
                    (getattr(r, "seo_title", "") or "")
                    for r in merged_rows(_ls, policy_channel.id, _since)][:400]
        except Exception:
            learned_policy = None
        # EXPLORATION: mostly exploit the measured best hook, but at
        # KAIZER_SEO_EXPLORE_RATE (default 0.2) deliberately test a
        # non-favored form — published, measured, and fed back, so the
        # policy can never freeze on a local optimum.
        try:
            from learning.seo_learning import pick_hook as _ph
            _rate = float(os.environ.get("KAIZER_SEO_EXPLORE_RATE", "0.2")
                          or "0.2")
            explore_hook, explored = _ph(learned_policy, explore_rate=_rate)
        except Exception:
            explore_hook, explored = None, False

    # COMPETITOR INTELLIGENCE (opt-in per channel): rivals' best videos on
    # THIS video's topic — their tags to harvest, terms to cover, titles
    # to DIFFERENTIATE from (rival titles also join the dedupe list so we
    # can never ship a near-copy of a bigger channel's headline).
    competitor = None
    if (policy_channel is not None
            and bool(getattr(policy_channel, "use_competitor_intel", False))):
        try:
            from sqlalchemy.orm import object_session
            from learning.competitor_intel import topic_intel
            import json as _json
            _ls2 = object_session(policy_channel)
            if _ls2 is not None:
                _meta = {}
                try:
                    _meta = _json.loads(clip.meta or "{}")
                except (ValueError, TypeError):
                    pass
                _terms = ([str(x) for x in (_meta.get("key_people") or [])]
                          + [str(x) for x in (_meta.get("key_topics") or [])]
                          + [str(x) for x in (_meta.get("key_locations") or [])])
                competitor = topic_intel(
                    _ls2, getattr(policy_channel, "user_id", 0) or 0, _terms)
                if competitor and competitor.get("rival_titles"):
                    recent_titles = list(recent_titles) + \
                        competitor["rival_titles"]
        except Exception:
            competitor = None

    # Cross-channel distinctness: sibling per-channel titles join the dedupe
    # pool so is_duplicate_title rejects reused openings on other channels.
    if avoid_titles:
        recent_titles = list(recent_titles) + [str(t) for t in avoid_titles if t]

    # SEO WRITER ENGINE (user-selectable): 'gemini' (default) | 'claude'.
    # Env KAIZER_SEO_ENGINE forces globally; otherwise the publishing
    # user's saved preference decides. Claude failures fall back to the
    # Gemini chain per attempt — engine choice can never break SEO.
    # Explicit engine_choice (per-channel delegation passes the publishing
    # user's pick) > env force > the clip's user preference > gemini.
    seo_engine = (engine_choice or os.environ.get("KAIZER_SEO_ENGINE")
                  or "").strip().lower()
    if seo_engine not in ("gemini", "claude"):
        seo_engine = "gemini"
        try:
            from sqlalchemy.orm import object_session
            _jb = getattr(clip, "job", None)
            _uid = getattr(_jb, "user_id", None) if _jb else None
            _cs = object_session(clip)
            if _uid and _cs is not None:
                _usr = (_cs.query(models.User)
                        .filter(models.User.id == _uid).first())
                if _usr and (getattr(_usr, "seo_engine", "") or "") == "claude":
                    seo_engine = "claude"
        except Exception:
            seo_engine = "gemini"

    # TITLE SCRIPT POLICY (operator decision 2026-08): per-channel-LEARNABLE.
    # 'learned' (default) = follow the channel's measured best script when
    # confident, bilingual otherwise; or force via KAIZER_SEO_SCRIPT_POLICY=
    # bilingual|english|native.
    _sp_env = (os.environ.get("KAIZER_SEO_SCRIPT_POLICY", "learned")
               or "learned").strip().lower()
    if _sp_env in ("bilingual", "english", "native"):
        script_policy = _sp_env
    else:
        script_policy = ((learned_policy or {}).get("best_script")
                         if (learned_policy or {}).get("best_script")
                         in ("english", "native", "mixed") else None)
        script_policy = {"mixed": "bilingual", "english": "english",
                         "native": "native"}.get(script_policy, "bilingual")

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
            learned=learned_policy,
            explore_hook=(explore_hook if explored else None),
            script_policy=script_policy,
            competitor=competitor,
            avoid_titles=avoid_titles,
            angle_hint=angle_hint,
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
            raw = None
            if seo_engine == "claude":
                try:
                    raw, model_used = _call_claude_seo(
                        system_prompt, user_prompt)
                except Exception as _ce:
                    print(f"[seo] claude engine failed "
                          f"({str(_ce)[:100]}) — falling back to gemini",
                          flush=True)
                    raw = None
            if raw is None:
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
            script_policy=script_policy,
            competitor_terms=((competitor or {}).get("harvest_tags") or [])
                             + ((competitor or {}).get("cover_terms") or []),
        )
        # DEDUPE GUARD: a title matching one of this channel's recent
        # videos is a hard failure — dock it below target and hand the
        # retry loop a rewrite order (identical published titles shipped
        # once; the guard makes that structurally impossible).
        try:
            from learning.seo_learning import is_duplicate_title
            if recent_titles and is_duplicate_title(
                    cleaned.get("title", ""), recent_titles):
                report["score"] = min(report["score"], TARGET_SCORE - 20)
                report["reasons"].append(
                    "title DUPLICATES a recent video on this channel — "
                    "write a fresh headline with a different angle/wording")
        except Exception:
            pass
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

    # One-line visibility into which ADVANCED features engaged for THIS
    # generation (dev observability; no behaviour change, fail-soft).
    try:
        _lp = learned_policy or {}
        _co = competitor or {}
        print(
            "[seo:features] clip=%s engine=%s learned=%s%s competitor=%s%s "
            "script=%s explore=%s trends=%d news=%d dedupe_pool=%d "
            "attempts=%d score=%d model=%s" % (
                getattr(clip, "id", "?"), seo_engine,
                "YES" if learned_policy else "no",
                (f"(hooks={','.join(_lp.get('best_hooks') or []) or '-'},"
                 f"script={_lp.get('best_script') or '-'},"
                 f"kw={len(_lp.get('top_keywords') or [])},"
                 f"topics={len(_lp.get('top_topics') or [])},"
                 f"base={_lp.get('based_on', '?')})") if learned_policy else "",
                "YES" if competitor else "no",
                (f"(rivals={len(_co.get('rival_titles') or [])},"
                 f"cover={len(_co.get('cover_terms') or [])},"
                 f"tags={len(_co.get('harvest_tags') or [])})") if competitor else "",
                script_policy, (explore_hook if explored else "-"),
                len(all_trend_keywords or []), len(news_items or []),
                len(recent_titles or []), len(attempts_log),
                best_score, model_used,
            ), flush=True)
    except Exception:
        pass

    # ── 6. Attach bookkeeping + persist ──
    best["seo_score"]   = best_score
    if explored and explore_hook:
        # A/B ledger stamp — flows through clip.seo into TrainingSample.
        best["explored_hook"] = explore_hook
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

    # ── 6b. Tool-based SEO score (content relevance + keyword coverage +
    # title/tag quality). Stored alongside the verifier score so the editor
    # shows an objective "does this match the video / use real keywords" read.
    # Trends OFF here (we already gathered trending_keywords above); advisory.
    try:
        from seo.score_checker import score_seo as _score_seo
        _content = " ".join(x for x in [getattr(clip, "text", "") or "", topic or ""] if x)
        _sc = _score_seo(
            title=best.get("title", ""), description=best.get("description", ""),
            tags=best.get("tags") or best.get("keywords") or [],
            content_text=_content, language=language, use_trends=False,
        )
        best["tool_score"]       = _sc.get("score")
        best["tool_verdict"]     = _sc.get("verdict")
        best["tool_suggestions"] = _sc.get("suggestions", [])
        best["tool_dimensions"]  = _sc.get("dimensions", {})
        tick("scored", f"tool score {_sc.get('score')}/100 ({_sc.get('verdict')})")
    except Exception as _exc:
        tick("score", f"score-checker skipped: {_exc}")

    # Persist as the single canonical generic SEO on the clip.  The legacy
    # per-channel `seo_variants` field is left untouched (read-only legacy).
    # persist=False for a synthetic clip (delegation) — the caller owns the
    # returned dict and there is no real row to write.
    if persist:
        clip.seo = json.dumps(best, ensure_ascii=False)
        db.commit()
        try: db.refresh(clip)
        except Exception: pass

    return best
