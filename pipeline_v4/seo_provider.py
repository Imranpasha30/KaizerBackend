"""V4 SEO provider — generates YouTube-grade title / description /
hashtags / keywords per short and per bulletin.

Reuses V2's SEO system prompt (``seo/prompts.py:build_system_prompt``)
and the same Gemini 2.5 Flash model so the editorial voice matches
what the team is already shipping on V2 jobs. V4 doesn't have a
``Clip`` row to feed into ``seo/generator.generate_seo_for_clip``, so
this module builds the user-prompt and parses Gemini's reply directly.

The returned dict has the same shape as ``clip.seo`` so the publish
flow (``routers/youtube_upload.py::_compose_metadata``) can consume it
without changes.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional


DEFAULT_MODEL = os.environ.get("KAIZER_SEO_MODEL", "gemini-2.5-flash")


@dataclass
class SeoInput:
    """One unit of content to SEO. Short or bulletin — same shape."""
    kind: str               # "short" | "bulletin"
    language: str           # ISO code
    title_native: str
    title_english: str = ""
    summary: str = ""
    body: str = ""          # extra context (joined headlines for bulletin)
    # Optional competitor style reference id — recorded on the output so
    # the editor can show which voice was used. The resolved Channel is
    # passed separately to generate_seo(style_source=...).
    style_source_id: Optional[int] = None
    # Optional facts for the UNIFIED (advanced-engine) path — feed the
    # competitor topic-match + richer grounding. All default empty so the
    # legacy Engine-B path is byte-identical when they're absent.
    key_people: Optional[list] = None
    key_topics: Optional[list] = None
    key_locations: Optional[list] = None
    sentiment: str = ""
    duration: float = 0.0
    # CLEAN content (transcript / story text) for grounding — kept separate
    # from `body`, which per-channel packs with steer instructions. The unified
    # engine grounds on this so per-channel SEO stays about the actual video.
    content: str = ""


# Master switch for the unified engine: when "1", generate_seo* delegate to
# the advanced seo.generator (learned policy + competitor intel + script
# policy + dedupe + 100-pt verifier + Gemini|Claude) via a synthetic clip,
# with the legacy in-file Gemini path kept as an automatic fallback. Read per
# call so a .env flip + restart is the full rollback.
def _unified_enabled() -> bool:
    return (os.environ.get("KAIZER_SEO_UNIFIED", "0") or "0").strip() == "1"


def _build_user_prompt(inp: SeoInput, corpus: Optional[dict] = None) -> str:
    facts = [
        f"language: {inp.language}",
        f"output kind: {inp.kind}",
    ]
    if inp.title_native:
        facts.append(f"native-script title: {inp.title_native}")
    if inp.title_english:
        facts.append(f"English title: {inp.title_english}")
    if inp.summary:
        facts.append(f"summary: {inp.summary}")
    if inp.body:
        facts.append(f"additional context:\n{inp.body[:1500]}")
    # Writing-voice corpus from the chosen style reference — emulate the
    # RHYTHM, never name the channel (the system prompt enforces this).
    corpus_block = ""
    if corpus and corpus.get("top_titles"):
        corpus_block = (
            "\n\n# Writing-voice corpus (emulate the RHYTHM/wording, do NOT "
            "mention or name the reference channel):\n"
        )
        for t in (corpus.get("top_titles") or [])[:8]:
            corpus_block += f"- {t}\n"
        if corpus.get("hook_patterns"):
            corpus_block += ("Common hooks: "
                             + " | ".join(corpus["hook_patterns"][:6]) + "\n")
        if corpus.get("power_words"):
            corpus_block += ("Power words: "
                             + ", ".join(corpus["power_words"][:10]) + "\n")
    return (
        "Generate channel-agnostic YouTube SEO for the news clip below.\n\n"
        + "\n".join(facts)
        + corpus_block
        + "\n\nRespond as ONE JSON object with keys: title, description, "
          "keywords (array of 28-30 strings), hashtags (array of 10-12 "
          "strings each starting with #), hook, thumbnail_text, "
          "metadata (object: sentiment, category, viral_score). "
          "No surrounding prose. Strict JSON."
    )


def _strip_json_fence(raw: str) -> str:
    s = (raw or "").strip()
    if s.startswith("```"):
        # ```json\n...\n```  or  ```\n...\n```
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()


def _loads_lenient_json(raw: str):
    """Parse model JSON tolerantly: strip fences, then strict → outermost
    ``{ ... }`` slice → trailing-comma repair. Repairs are conservative (only
    ever-invalid constructs), so valid content is never corrupted. Raises
    ``json.JSONDecodeError`` if none parse — a genuine missing-comma error is
    left to the caller's model-retry, the only safe fix for those. Mirrors
    trim_engine._loads_lenient so SEO JSON is as robust as the cut-plan JSON."""
    import re as _re
    s = _strip_json_fence(raw or "")
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = _re.search(r"\{.*\}", s, flags=_re.DOTALL)
    cand = m.group(0) if m else s
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        pass
    repaired = _re.sub(r",(\s*[}\]])", r"\1", cand)
    return json.loads(repaired)   # raises if still malformed


def _empty_seo(language: str) -> dict:
    """A safe-default SEO blob so downstream code never sees a None."""
    return {
        "title": "",
        "description": "",
        "keywords": [],
        "hashtags": [],
        "hook": "",
        "thumbnail_text": "",
        "metadata": {"sentiment": "", "category": "", "viral_score": 0},
        "language": language,
        "model": "",
        "edited_by_user": False,
    }


def generate_seo(
    inp: SeoInput,
    *,
    model_name: Optional[str] = None,
    style_source=None,
    db=None,
    own_channel=None,
    avoid_titles: Optional[list] = None,
    angle_hint: Optional[str] = None,
    engine_choice: Optional[str] = None,
) -> dict:
    """Single-shot SEO generation. Returns the V2-compatible dict on
    success, an empty-but-valid skeleton on failure. Never raises so
    the caller's render path is never blocked on SEO.

    ``style_source`` (optional) is a competitor-style ``Channel`` the
    user picked: its ``title_formula`` / ``desc_style`` shape the system
    prompt's writing-voice block and its learned ``corpus`` (top titles /
    hooks / power words) is injected as rhythm examples. Resolve it from
    the caller's DB session BEFORE calling (we read its ``.corpus``
    relationship here, so the session must still be open)."""
    # UNIFIED path (KAIZER_SEO_UNIFIED=1): delegate to the advanced engine so
    # even single-shot SEO gets learned policy + competitor intel + script
    # policy + the 100-pt verifier + Gemini|Claude. Falls back to the legacy
    # in-file Gemini path below on any failure — SEO is never blocked.
    if _unified_enabled() and db is not None:
        try:
            return generate_seo_from_input(
                inp, db=db, own_channel=own_channel, style_source=style_source,
                avoid_titles=avoid_titles, angle_hint=angle_hint,
                engine_choice=engine_choice)
        except Exception as exc:
            print(f"[v4/seo] unified engine failed, using legacy path: "
                  f"{str(exc)[:160]}", flush=True)
    try:
        from seo.generator import _gemini_client
        from seo.prompts import build_system_prompt
        from google.genai import types as genai_types
    except Exception as exc:
        print(f"[v4/seo] SDK unavailable, skipping: {exc}", flush=True)
        return _empty_seo(inp.language)

    try:
        client = _gemini_client()
    except Exception as exc:
        print(f"[v4/seo] Gemini key missing, skipping: {exc}", flush=True)
        return _empty_seo(inp.language)

    sys_prompt = build_system_prompt(
        language=inp.language, style_source=style_source,
    )
    corpus_payload = None
    if style_source is not None:
        try:
            corp = getattr(style_source, "corpus", None)
            corpus_payload = getattr(corp, "payload", None) if corp else None
        except Exception:
            corpus_payload = None
    user_prompt = _build_user_prompt(inp, corpus=corpus_payload)
    model = model_name or DEFAULT_MODEL

    try:
        resp = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=sys_prompt,
                response_mime_type="application/json",
                temperature=0.7,
            ),
        )
    except Exception as exc:
        print(f"[v4/seo] Gemini call failed: {exc}", flush=True)
        return _empty_seo(inp.language)

    raw = ""
    try:
        raw = (resp.text or "").strip()
    except Exception:
        raw = ""
    if not raw:
        return _empty_seo(inp.language)

    # Safety net: when a style reference was used, strip any leaked
    # competitor name/handle from the output (same sanitizer the rich
    # engine uses). Best-effort — never block on it.
    if style_source is not None and raw:
        try:
            from seo import sanitizer as _san
            raw = _san.sanitize(raw, style_source)
        except Exception:
            pass

    try:
        data = _loads_lenient_json(raw)
    except Exception as exc:
        print(f"[v4/seo] JSON parse failed: {exc} | raw[:200]={raw[:200]}",
              flush=True)
        return _empty_seo(inp.language)

    seo = _empty_seo(inp.language)
    seo.update({
        "title":          str(data.get("title", "")).strip(),
        "description":    str(data.get("description", "")).strip(),
        "keywords":       [str(k).strip() for k in (data.get("keywords") or [])
                           if str(k).strip()][:30],
        "hashtags":       [str(h).strip() for h in (data.get("hashtags") or [])
                           if str(h).strip()][:12],
        "hook":           str(data.get("hook", "")).strip(),
        "thumbnail_text": str(data.get("thumbnail_text", "")).strip(),
        "metadata":       dict(data.get("metadata") or {}),
        "model":          model,
    })
    # Inline hashtags into the description body. YouTube's feed
    # discovery picks up hashtags from the description text, not just
    # the structured tags field. The operator explicitly requested
    # "description must contain hashtags" (2026-06-09 follow-up).
    seo["description"] = inline_hashtags_into_description(
        description=seo["description"], hashtags=seo["hashtags"],
    )
    # Record which style reference (if any) shaped this copy, so the
    # editor/persistence can show "written in X's style" and re-pick it.
    seo["style_source_id"] = inp.style_source_id

    # Auto-score with the tool-based SEO checker (content relevance + keyword
    # coverage + title/tag quality). Stored on the SEO so the editor shows it
    # and weak SEO is visible. Trends is OFF here (it would rate-limit on bulk
    # generation) — it runs on-demand in the score-checker. Advisory: never
    # blocks generation; any failure leaves the SEO untouched.
    try:
        from seo.score_checker import score_seo as _score_seo
        _content = " ".join(x for x in [inp.title_native, inp.title_english,
                                        inp.summary, inp.body] if x)
        _sc = _score_seo(
            title=seo["title"], description=seo["description"],
            tags=seo.get("keywords") or [], content_text=_content,
            language=inp.language, use_trends=False,
        )
        seo["tool_score"]       = _sc.get("score")
        seo["tool_verdict"]     = _sc.get("verdict")
        seo["tool_suggestions"] = _sc.get("suggestions", [])
        seo["tool_dimensions"]  = _sc.get("dimensions", {})
    except Exception as _exc:
        print(f"[v4/seo] score-checker skipped: {_exc}", flush=True)
    return seo


def generate_seo_from_input(
    inp: SeoInput,
    *,
    db,
    own_channel=None,
    style_source=None,
    avoid_titles: Optional[list] = None,
    angle_hint: Optional[str] = None,
    engine_choice: Optional[str] = None,
) -> dict:
    """UNIFIED path: run the ADVANCED engine (seo.generator.generate_seo_for_clip)
    from a SeoInput, so render/publish SEO gets learned policy + competitor
    intel + script policy + dedupe + the 100-pt verifier + Gemini|Claude choice.

    Builds a DETACHED synthetic clip (never added to the session; persist=False)
    carrying the facts the engine mines. Returns the V4/publish dict shape.
    Raises on any failure so the caller can fall back to the legacy path."""
    from seo.generator import generate_seo_for_clip
    import models as _m

    # GROUND ON THE REAL CONTENT. Quick Publish puts the transcript in `body`
    # (transcript mode) or the user's blurb in `summary` (description mode), and
    # a raw upload's title is a GENERIC placeholder ('bulletin'). If we seed the
    # topic from that placeholder the engine writes SEO for the WRONG topic
    # (verified: a finance video got a "Jr NTR surgery" title). So derive the
    # topic + grounding from the actual content, not the placeholder title.
    _summary = (inp.summary or "").strip()
    # Prefer the CLEAN content (per-channel packs `body` with steer text) so the
    # grounding stays about the actual video, not the steer instructions.
    _clean = (inp.content or "").strip()
    _body = (_clean or inp.body or "").strip()
    _content = (_clean or _summary or _body).strip()
    _title = (inp.title_native or inp.title_english or "").strip()
    _generic = _title.lower() in (
        "", "bulletin", "short", "shorts", "clip", "video", "raw", "untitled")
    topic_seed = _title if not _generic else (_content[:160] or _title)

    meta = {
        # English-summary line (description mode) + native-summary line grounds
        # on the real transcript/content so the writer stays ON topic.
        "summary": _summary[:2500],
        "summary_native": (_body or (inp.title_native if not _generic else ""))[:3000],
        "key_people": list(inp.key_people or []),
        "key_topics": list(inp.key_topics or []),
        "key_locations": list(inp.key_locations or []),
        "text": (_content or _title)[:2500],
    }
    clip = _m.Clip()                       # transient, NOT added to the session
    clip.text = topic_seed
    clip.meta = json.dumps(meta, ensure_ascii=False)
    clip.sentiment = inp.sentiment or ""
    clip.duration = float(inp.duration or 0.0)

    rich = generate_seo_for_clip(
        clip, db=db, style_source=style_source, own_channel=own_channel,
        include_news=True, include_trends=True, include_yt_benchmark=True,
        language=inp.language, avoid_titles=avoid_titles, angle_hint=angle_hint,
        engine_choice=engine_choice, persist=False,
    )

    out = _empty_seo(inp.language)
    out.update({
        "title":          str(rich.get("title", "")).strip(),
        "description":    str(rich.get("description", "")).strip(),
        "keywords":       [str(k).strip() for k in (rich.get("keywords") or [])
                           if str(k).strip()][:30],
        "hashtags":       [str(h).strip() for h in (rich.get("hashtags") or [])
                           if str(h).strip()][:12],
        "hook":           str(rich.get("hook", "")).strip(),
        "thumbnail_text": str(rich.get("thumbnail_text", "")).strip(),
        "model":          rich.get("model") or DEFAULT_MODEL,
        "style_source_id": inp.style_source_id,
    })
    if not (out["title"]).strip():
        raise ValueError("unified engine returned an empty title")
    out["description"] = inline_hashtags_into_description(
        description=out["description"], hashtags=out["hashtags"])
    # Carry the advanced engine's REAL 100-pt verifier score + provenance.
    out["seo_score"] = rich.get("seo_score")
    out["tool_score"] = rich.get("seo_score")
    out["verifier_breakdown"] = rich.get("verifier_breakdown")
    out["verifier_reasons"] = rich.get("verifier_reasons")
    out["engine"] = "unified"
    return out


def generate_seo_to_score(
    inp: SeoInput,
    *,
    db=None,
    own_channel=None,
    style_source=None,
    avoid_titles: Optional[list] = None,
    angle_hint: Optional[str] = None,
    engine_choice: Optional[str] = None,
    target_score: int = 85,
    max_attempts: int = 4,
) -> dict:
    """Generate SEO and KEEP IMPROVING it until it scores ``target_score``+ (or
    ``max_attempts`` is reached). Each retry feeds the score-checker's own suggestions
    (``seo['tool_suggestions']``) back into the prompt so the next attempt fixes the exact
    weak dimensions (relevance / keyword coverage / title hook / tags). Reuses generate_seo's
    built-in tool score (no extra scoring call). Returns the BEST-scoring SEO dict with
    ``seo_score`` + ``seo_attempts`` set. Never raises — returns the best attempt so far.

    Cost note: up to ``max_attempts`` Gemini calls. 85 isn't always reachable (sparse/thin
    content yields low relevance); we then return the highest-scoring attempt rather than loop.
    """
    # UNIFIED path (KAIZER_SEO_UNIFIED=1): the advanced engine self-improves via
    # its own verifier retry loop, so one call replaces this score loop. Falls
    # back to the legacy loop below on any failure — a publish is never blocked.
    if _unified_enabled() and db is not None:
        try:
            return generate_seo_from_input(
                inp, db=db, own_channel=own_channel, style_source=style_source,
                avoid_titles=avoid_titles, angle_hint=angle_hint,
                engine_choice=engine_choice)
        except Exception as exc:
            print(f"[v4/seo] unified engine failed, using legacy path: "
                  f"{str(exc)[:160]}", flush=True)

    from dataclasses import replace as _dc_replace
    base_body = inp.body or ""
    best: Optional[dict] = None
    best_score = -1.0
    suggestions: list[str] = []
    attempts = 0
    for attempt in range(max(1, int(max_attempts))):
        attempts = attempt + 1
        cur = inp
        if attempt > 0 and suggestions:
            steer = (
                f"\n\nIMPROVE THIS SEO — the previous attempt scored {int(round(best_score))}"
                f"/100 (target {target_score}+). Fix these specifically, keeping it accurate — "
                f"TITLE in ENGLISH, description/tags in the target language:\n- "
                + "\n- ".join(str(s) for s in suggestions[:6])
            )
            cur = _dc_replace(inp, body=(base_body + steer)[:1800])
        seo = generate_seo(cur, style_source=style_source)
        if not (seo.get("title") or "").strip():
            continue   # empty (rate-limit / error) — retry, else fall through to best
        try:
            sc = float(seo.get("tool_score") or 0)
        except (TypeError, ValueError):
            sc = 0.0
        if sc > best_score:
            best_score, best = sc, seo
        if sc >= float(target_score):
            break
        suggestions = list(seo.get("tool_suggestions") or [])
    if best is None:
        best = generate_seo(inp, style_source=style_source)
        best_score = float(best.get("tool_score") or 0)
    best["seo_score"] = int(round(best_score)) if best_score >= 0 else 0
    best["seo_attempts"] = attempts
    return best


# ── Hashtag inlining: keep description + hashtags in sync ────────────

def inline_hashtags_into_description(*, description: str, hashtags: list[str]) -> str:
    """Return ``description`` with the hashtag set appended on a fresh
    line. Idempotent — if any of the hashtags already appear in the
    description (case-insensitive), we skip those tags and only add
    the missing ones. Used by every SEO write path so YouTube's feed
    discovery picks up our tags from the description body, not just
    the structured field.
    """
    desc = (description or "").strip()
    tags = [str(t).strip() for t in (hashtags or []) if str(t).strip()]
    if not tags:
        return desc
    # Normalise tags: ensure each starts with '#' so the join produces
    # a valid hashtag block. Some tag sources strip the #.
    normalised = []
    for t in tags:
        normalised.append(t if t.startswith("#") else f"#{t}")
    # De-dupe (case-insensitive) against what's already in the desc.
    haystack = desc.lower()
    missing = [t for t in normalised if t.lower() not in haystack]
    if not missing:
        return desc
    tail = " ".join(missing)
    if desc:
        return f"{desc}\n\n{tail}"
    return tail
