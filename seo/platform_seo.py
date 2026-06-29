"""Per-platform social SEO — Instagram / Facebook captions + hashtags.

The operator chose **a separate AI call per platform** (quality over cost):
each social platform gets its own Gemini generation so the copy reads
*native* to that surface, not a reshaped YouTube title.

  * Instagram (Reels): a scroll-stopping caption (hook line + a few short
    lines + soft CTA) and up to 30 hashtags (broad + niche + language/region).
  * Facebook: a longer, story-style caption that gives context and ends on a
    question/CTA to drive comments, plus 5-8 hashtags (FB uses far fewer).

YouTube is deliberately NOT handled here — it keeps its proven
title+description+tags path (``seo_provider`` / ``seo.generator`` +
``seo.composer``). This module only shapes the *social* platforms.

Results are meant to be cached into the clip's SEO JSON under
``seo['platform_variants'][platform]`` (see ``services.platform_variants``)
so we spend at most ONE Gemini call per (clip, platform) actually used.

Every function fails CLOSED — on any SDK / key / parse error it returns an
empty-but-valid dict (never raises) so a publish is never blocked on social
SEO. The caller can fall back to deterministic shaping in the composer.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


DEFAULT_MODEL = os.environ.get("KAIZER_SEO_MODEL", "gemini-2.5-flash")

# Platforms this module knows how to shape. "youtube" is intentionally
# excluded — it uses the dedicated YouTube path. Keep lowercase.
SUPPORTED_PLATFORMS = ("instagram", "facebook")

# Hashtag ceilings per platform (Instagram allows 30; Facebook reads as spam
# past a handful, so we keep it tight).
_HASHTAG_CAP = {"instagram": 30, "facebook": 8}


def _empty_variant(platform: str, language: str) -> Dict[str, Any]:
    """A safe-default platform variant so callers never see None."""
    return {
        "platform": platform,
        "caption": "",
        "hashtags": [],
        "tags": [],
        "hook": "",
        "language": language,
        "model": "",
        "edited_by_user": False,
    }


def _strip_json_fence(raw: str) -> str:
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def _normalize_hashtag(s: str) -> str:
    """Mirror of seo.composer._normalize_hashtag so the hashtag shape is
    identical everywhere (CamelCase, single leading #)."""
    import re
    s = (s or "").strip().lstrip("#")
    if not s:
        return ""
    parts = re.split(r"[\s_\-./,!?:;]+", s)
    camel = "".join(p[:1].upper() + p[1:] for p in parts if p)
    return f"#{camel}" if camel else ""


def _dedupe_hashtags(items: List[str], cap: int) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for h in items:
        norm = _normalize_hashtag(h)
        if not norm:
            continue
        k = norm.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(norm)
        if len(out) >= cap:
            break
    return out


# ── Prompts ──────────────────────────────────────────────────────────────

def _system_prompt(platform: str, language: str) -> str:
    common = (
        f"Write everything in the SAME language as the source content "
        f"(language code: {language}). Use the native script for that "
        f"language. Hashtags may mix native-script and English tags. "
        f"NEVER name, @-mention, or imitate the @handle of any other "
        f"creator, channel, or news outlet. No surrounding prose — respond "
        f"with ONE strict JSON object only."
    )
    if platform == "instagram":
        return (
            "You are a senior Instagram Reels strategist for a regional "
            "(Indian / Telugu) news + entertainment brand. Write a "
            "scroll-stopping Reel caption that maximises saves, shares and "
            "follows. Rules: line 1 is a curiosity hook (no clickbait lies); "
            "then 2-4 short punchy lines; tasteful emojis; end with a soft "
            "CTA (follow / share / comment). Provide a SEPARATE list of "
            "20-30 hashtags mixing broad reach tags (#reels #viral #trending "
            "#explore), the specific topic, and language/region tags. Do NOT "
            "put a title. Do NOT dump all hashtags inside the caption text — "
            "return them in the hashtags array. " + common +
            ' JSON keys: {"caption": string, "hashtags": [string], '
            '"hook": string}.'
        )
    # facebook
    return (
        "You are a senior Facebook page editor for a regional (Indian / "
        "Telugu) news brand. Write a Facebook post caption: a longer, "
        "story-style lead of 2-4 sentences that gives real context and makes "
        "people stop scrolling, ending with a question or CTA that drives "
        "comments and shares. Facebook audiences dislike hashtag spam, so "
        "provide only 5-8 highly-relevant hashtags. Do NOT put a title. "
        + common +
        ' JSON keys: {"caption": string, "hashtags": [string], '
        '"hook": string}.'
    )


def _user_prompt(
    *,
    platform: str,
    language: str,
    title_native: str,
    title_english: str,
    summary: str,
    base_seo: Optional[Dict[str, Any]],
    corpus: Optional[dict],
) -> str:
    facts = [f"language: {language}", f"target platform: {platform}"]
    if title_native:
        facts.append(f"native-script headline: {title_native}")
    if title_english:
        facts.append(f"English headline: {title_english}")
    if summary:
        facts.append(f"summary: {summary}")
    # Feed the already-generated YouTube SEO as source material so the
    # social caption stays consistent with the headline framing.
    if base_seo:
        bt = (base_seo.get("title") or "").strip()
        bd = (base_seo.get("description") or "").strip()
        bk = [k for k in (base_seo.get("keywords") or []) if str(k).strip()]
        if bt:
            facts.append(f"YouTube title (reference, do not copy verbatim): {bt}")
        if bd:
            facts.append(f"YouTube description (reference): {bd[:600]}")
        if bk:
            facts.append("topic keywords: " + ", ".join(bk[:18]))
    corpus_block = ""
    if corpus and corpus.get("top_titles"):
        corpus_block = (
            "\n\n# Writing-voice corpus (emulate the RHYTHM/wording only, do "
            "NOT mention or name the reference channel):\n"
        )
        for t in (corpus.get("top_titles") or [])[:6]:
            corpus_block += f"- {t}\n"
        if corpus.get("power_words"):
            corpus_block += "Power words: " + ", ".join(corpus["power_words"][:10]) + "\n"
    return (
        f"Generate a native {platform} caption + hashtags for the news clip "
        f"below.\n\n" + "\n".join(facts) + corpus_block
    )


# ── Public entry ───────────────────────────────────────────────────────────

def generate_platform_seo(
    *,
    platform: str,
    language: str = "te",
    title_native: str = "",
    title_english: str = "",
    summary: str = "",
    base_seo: Optional[Dict[str, Any]] = None,
    style_source=None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate one platform's caption + hashtags via a dedicated Gemini call.

    Returns a platform-variant dict (see ``_empty_variant`` for the shape) on
    success, an empty-but-valid dict on any failure. Never raises.

    ``base_seo`` is the clip's already-generated YouTube SEO (clip.seo) — used
    as reference material so the social caption matches the headline framing.
    ``style_source`` (optional) is a competitor-style Channel whose learned
    ``corpus`` (top titles / power words) shapes the writing rhythm.
    """
    platform = (platform or "").strip().lower()
    if platform not in SUPPORTED_PLATFORMS:
        # youtube / unknown — this module doesn't own that path.
        return _empty_variant(platform or "unknown", language)

    try:
        from seo.generator import _gemini_client
        from google.genai import types as genai_types
    except Exception as exc:
        print(f"[platform-seo] SDK unavailable, skipping: {exc}", flush=True)
        return _empty_variant(platform, language)

    try:
        client = _gemini_client()
    except Exception as exc:
        print(f"[platform-seo] Gemini key missing, skipping: {exc}", flush=True)
        return _empty_variant(platform, language)

    corpus_payload = None
    if style_source is not None:
        try:
            corp = getattr(style_source, "corpus", None)
            corpus_payload = getattr(corp, "payload", None) if corp else None
        except Exception:
            corpus_payload = None

    sys_prompt = _system_prompt(platform, language)
    usr_prompt = _user_prompt(
        platform=platform, language=language,
        title_native=title_native, title_english=title_english,
        summary=summary, base_seo=base_seo, corpus=corpus_payload,
    )
    model = model_name or DEFAULT_MODEL

    try:
        resp = client.models.generate_content(
            model=model,
            contents=usr_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=sys_prompt,
                response_mime_type="application/json",
                temperature=0.8,
            ),
        )
    except Exception as exc:
        print(f"[platform-seo] Gemini call failed ({platform}): {exc}", flush=True)
        return _empty_variant(platform, language)

    raw = ""
    try:
        raw = (resp.text or "").strip()
    except Exception:
        raw = ""
    if not raw:
        return _empty_variant(platform, language)

    # Strip any leaked competitor name/handle when a style reference was used.
    if style_source is not None:
        try:
            from seo import sanitizer as _san
            raw = _san.sanitize(raw, style_source)
        except Exception:
            pass

    try:
        data = json.loads(_strip_json_fence(raw))
    except Exception as exc:
        print(f"[platform-seo] JSON parse failed ({platform}): {exc} | "
              f"raw[:200]={raw[:200]}", flush=True)
        return _empty_variant(platform, language)

    cap = _HASHTAG_CAP.get(platform, 12)
    out = _empty_variant(platform, language)
    out.update({
        "caption":  str(data.get("caption", "")).strip(),
        "hashtags": _dedupe_hashtags(
            [str(h) for h in (data.get("hashtags") or [])], cap),
        "hook":     str(data.get("hook", "")).strip(),
        "model":    model,
    })
    # Carry the base SEO's keyword list so search-style tags survive for any
    # platform field that wants them (kept separate from the # hashtags).
    if base_seo:
        out["tags"] = [str(k).strip() for k in (base_seo.get("keywords") or [])
                       if str(k).strip()][:30]

    # Advisory tool score — reuse the same checker as YouTube. Trends are
    # search-demand-biased (meaningless for IG/FB engagement) so keep them OFF;
    # the relevance + keyword-coverage dimensions are still useful signal.
    try:
        from seo.score_checker import score_seo as _score_seo
        _content = " ".join(x for x in [title_native, title_english, summary] if x)
        _sc = _score_seo(
            title=out["caption"][:120], description=out["caption"],
            tags=out["hashtags"], content_text=_content,
            language=language, use_trends=False,
        )
        out["tool_score"]       = _sc.get("score")
        out["tool_verdict"]     = _sc.get("verdict")
        out["tool_suggestions"] = _sc.get("suggestions", [])
    except Exception as _exc:
        print(f"[platform-seo] score-checker skipped ({platform}): {_exc}", flush=True)

    return out
