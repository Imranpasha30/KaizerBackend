"""Pass 1 — turn news context into a structured ScenePlan.

Owned by the planning team. The contract with Pass 2 is the ScenePlan
schema in schemas.py; as long as a planner returns one that satisfies
``ScenePlan.is_complete()`` Pass 2 doesn't care HOW it was produced.

This implementation calls Gemini 2.5 Pro with response_schema enforced.
Pro (not Flash) because the editorial reasoning here — "what should be
in this thumbnail and why" — is the highest-leverage decision in the
whole image pipeline. Worth the extra ~$0.0005 per call vs Flash.

Soft-fails to a deterministic plan built from raw inputs when:
  * the Vertex SDK can't be imported
  * Gemini auth fails
  * Gemini's response misses required fields
  * the response can't be parsed as JSON

The deterministic fallback is much weaker than a real planner output
but at least guarantees the prompter has something to work with.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

from .schemas import ScenePlan, SCENEPLAN_JSON_SCHEMA
from .templates import get_template
from .examples import examples_for_style


# Gemini 2.5 Pro for planning — high editorial leverage, low call volume
# (one per thumbnail) so the cost increase over Flash is rounding error.
DEFAULT_PLANNER_MODEL = os.environ.get(
    "KAIZER_THUMB_PLANNER_MODEL", "gemini-2.5-pro"
)


def _build_system_prompt(*, style: str, language: str, has_references: int) -> str:
    """Compose the planner system prompt.

    Two modes:
      * style == "auto" (default for one-button UX): Gemini picks the
        composition direction itself based on the news content. No
        per-style template hint is offered. Maximum creative freedom.
      * style == <specific>: legacy path. Operator picked a style
        explicitly; we surface its template as a hint (not rule).

    Identity preservation (when references attached) is enforced as a
    code-side lock in prompter.py, not relied on the LLM to honour.
    """
    is_auto = (style or "").strip().lower() in ("", "auto")

    # Few-shot examples — even in auto mode we show shape-of-output
    # references so Gemini knows the JSON cadence and detail level
    # we expect. Sample evenly across styles in auto mode so the
    # examples don't bias toward one composition.
    if is_auto:
        from .examples import EXAMPLES as _ALL_EXAMPLES
        shots = _ALL_EXAMPLES[:3]
    else:
        shots = examples_for_style(style, max_count=2)

    shots_block = ""
    if shots:
        parts = []
        for i, shot in enumerate(shots, start=1):
            parts.append(
                f"---\nExample {i} input news context:\n{shot['input']}\n\n"
                f"Example {i} ScenePlan (shape-of-output reference):\n"
                f"{json.dumps(shot['plan'], ensure_ascii=False, indent=2)}"
            )
        shots_block = "\n\n# Few-shot examples (shape only — copy the structure, not the content)\n\n" + "\n\n".join(parts)

    ref_note = ""
    if has_references > 0:
        ref_note = (
            f"\n\n# Reference photo lock\n"
            f"The operator supplied {has_references} reference photograph(s). "
            f"The primary_subject MUST be a description of the reference "
            f"person(s) with EXPLICIT gender / face / hair / age preserved. "
            f"Add to what_to_avoid: \"do not change gender, do not replace "
            f"with stock anchor, the output must be recognisably the same "
            f"individual\"."
        )

    # Style-direction block: only injected when the operator picked a
    # specific style. In auto mode we leave it out so Gemini decides
    # composition freely.
    style_block = ""
    if not is_auto:
        tpl = get_template(style)
        style_block = (
            f"\n\n# Operator-picked starting direction: {style}\n"
            f"The operator selected style '{style}' as a starting point:\n"
            f"  - focal rule:  {tpl['focal_rule']}\n"
            f"  - scene rule:  {tpl['scene_rule']}\n"
            f"  - mood rule:   {tpl['mood_rule']}\n"
            f"  - avoid rule:  {tpl['avoid_rule']}\n"
            f"Use this as starting direction. If the story calls for "
            f"something else, IGNORE the hint and design what's right "
            f"for the news."
        )

    return (
        "# You are an Art Director for an Indian regional news channel\n"
        "(Telugu, Hindi, Tamil, Kannada — TV9, NTV, ABN, Sakshi, ETV "
        "Bharat, Republic Bharat). You design YouTube-style clickbait "
        "thumbnails for those channels.\n\n"

        "# The ONE locked constraint\n"
        "**The output MUST look like an Indian news-channel YouTube "
        "thumbnail.** That means: dramatic, attention-grabbing, "
        "photographic, with Indian visual vocabulary (Indian Khaki "
        "police uniforms, Rupee ₹ bundles, Indian courts / Parliament, "
        "Indian newsroom LED walls, Indian street furniture). NOT "
        "American. NOT a calm newspaper press photo — this is YouTube "
        "clickbait energy. Tight framing, dramatic subject expressions, "
        "high contrast, one clear focal point per frame.\n\n"

        "# Your job\n"
        "You read the actual news story (transcript + SEO context) and "
        "produce a structured editorial whiteboard — a ScenePlan — that "
        "a downstream prompt engineer will execute. Every decision is "
        "yours: composition, colour palette, lighting, mood, focal "
        "subject, the shout text on the overlay. Make the thumbnail "
        "that will make a Telugu / Hindi viewer most eager to click.\n\n"

        "# Hard quality rules (these are NOT creative choices)\n"
        "1. NO generic 'a newsroom' / 'a news anchor' answers. Every "
        "field MUST reference something specific from the actual story.\n"
        "2. Tight, dramatic framing — close-up or medium-close, low or "
        "eye-level angle, strong focal point. NEVER a wide landscape "
        "press-photo composition (the bus-depot mistake).\n"
        "3. High contrast, saturated palette by default — even calm "
        "stories want YouTube punch.\n"
        "4. The frame should have ONE clear focal point the viewer's "
        "eye lands on within 200ms.\n\n"

        "# Story-specific overlay text\n"
        f"text_for_overlay must be 2-4 words in the native script of the "
        f"story's language ({language}). It must reflect the SPECIFIC "
        f"story (the actual incident, the number, the place, the quote) "
        f"— NEVER a generic shout like 'BIG BREAKING' or 'BREAKING NEWS'. "
        f"If the story is about a 100-crore scam: '100 కోట్ల కుంభకోణం'. "
        f"If a TSRTC loan limit: 'రూ.10 లక్షల రుణం!'. If KCR vs Revanth: "
        f"'KCR vs Revanth'. Get it specific to THIS story or the "
        f"thumbnail fails.\n\n"

        "# Text rendering note\n"
        "text_for_overlay is painted externally by Pillow using a real "
        "Noto Sans font — the image model NEVER renders text. Your plan "
        "informs WHAT we paint and WHERE the safe zone is. Leave that "
        "zone visually quieter in the image so the painted text reads.\n\n"

        "# Output format\n"
        "Output ONE JSON object matching the ScenePlan schema. No prose, "
        "no markdown fences, no preamble. Use the EXACT field names from "
        "the schema. Fill EVERY required field with a specific, story-"
        "grounded answer."
        f"{style_block}"
        f"{ref_note}{shots_block}"
    )


def _build_user_prompt(*,
                       title_native: str, title_english: str,
                       summary: str, seo_hook: str,
                       seo_thumbnail_text: str, seo_description: str,
                       transcript_excerpt: str) -> str:
    """Assemble the news-context user message. Front-loads the strongest
    signals (headlines + hook) so Gemini reads them first."""
    facts: list[str] = []
    if title_native:       facts.append(f"native-script headline: {title_native}")
    if title_english:      facts.append(f"English headline: {title_english}")
    if seo_hook:           facts.append(f"SEO hook: {seo_hook}")
    if seo_thumbnail_text: facts.append(
        f"SEO thumbnail_text (suggested shout text): {seo_thumbnail_text}"
    )
    if summary:            facts.append(f"summary: {summary[:600]}")
    if seo_description:    facts.append(
        f"SEO description (full editorial body):\n{seo_description[:2000]}"
    )
    if transcript_excerpt: facts.append(
        f"transcript (verbatim spoken content from the clip):\n"
        f"{transcript_excerpt[:2000]}"
    )
    return (
        "Produce a ScenePlan for the thumbnail of this news story.\n\n"
        + "\n\n".join(facts)
    )


def _fallback_plan(*,
                   title_native: str, title_english: str,
                   summary: str, language: str, style: str,
                   has_references: int) -> ScenePlan:
    """Deterministic last-resort plan built from raw inputs when Gemini
    fails entirely. Weaker than a real plan but unblocks the prompter."""
    tpl = get_template(style)
    headline = (title_native or title_english or "BREAKING").strip()
    return ScenePlan(
        primary_subject=f"Indian-context visual of: {headline[:120]}",
        scene_location="Specific Indian institutional setting drawn from the story",
        key_action=(summary or headline)[:140],
        supporting_props=[],
        lighting="cinematic Indian newsroom lighting matching the story tone",
        color_grade="muted news palette matching the story tone",
        framing="medium close-up, 16:9, slight low angle for authority",
        emotional_hook="invites the viewer to want to know what happened",
        what_to_avoid=tpl.get("avoid_rule", "no Western visual vocabulary"),
        safe_zone=tpl.get("safe_zone", "bottom 28% strap"),
        text_for_overlay=(seo_thumbnail_text_fallback(title_native, title_english) or ""),
        style=style,
        language=language,
        has_references=has_references,
    )


def seo_thumbnail_text_fallback(title_native: str, title_english: str) -> str:
    """Pick a short shout-text fragment from the headline as a last
    resort. The Pillow overlay step uses the canvas SEO's
    thumbnail_text → hook → title chain anyway; this is only used when
    the planner can't even reach Gemini."""
    src = (title_native or title_english or "").strip()
    if not src:
        return ""
    # Take the first 4 native-script words by whitespace split.
    words = src.split()
    return " ".join(words[:4])


def _plan_via_claude(*,
                     title_native: str, title_english: str,
                     summary: str, seo_hook: str,
                     seo_thumbnail_text: str, seo_description: str,
                     transcript_excerpt: str, language: str, style: str,
                     has_references: int) -> Optional[ScenePlan]:
    """Pass 1 via Claude Opus 4.7. Same system prompt, same output
    schema, different brain. Generally stronger at editorial
    specificity ("the TV9 studio with a curved LED wall showing a
    faded Hyderabad railway map") than Gemini Pro, which tends
    toward more generic answers.

    Costs ~10x Gemini Pro per call. Operator picks via toggle.

    Returns None on any failure so the caller can fall through to the
    Gemini path (don't double-charge the operator on a partial outage).
    """
    try:
        from anthropic import Anthropic
    except Exception as exc:
        print(f"[thumb-director/plan/claude] Anthropic SDK unavailable: {exc}",
              flush=True)
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("[thumb-director/plan/claude] ANTHROPIC_API_KEY not set", flush=True)
        return None

    sys_prompt = _build_system_prompt(
        style=style, language=language, has_references=has_references,
    )
    user_prompt = _build_user_prompt(
        title_native=title_native, title_english=title_english,
        summary=summary, seo_hook=seo_hook,
        seo_thumbnail_text=seo_thumbnail_text,
        seo_description=seo_description,
        transcript_excerpt=transcript_excerpt,
    )

    # Claude doesn't have native response_schema enforcement like
    # Gemini does, but it's reliable about following "Output ONE JSON
    # object" when told so. We strip code fences + regex-recover if
    # it decorates the output.
    from contextlib import nullcontext
    try:
        from learning.claude_log import log_anthropic_call as _log_anthropic
        _cm = _log_anthropic(db=None, model="claude-opus-4-7", purpose="thumbnail-plan")
    except Exception:
        _cm = nullcontext(None)

    try:
        client = Anthropic(api_key=api_key)
        with _cm as _acall:
            msg = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=4096,
                system=sys_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            if _acall is not None:
                _acall.record(msg)
    except Exception as exc:
        print(f"[thumb-director/plan/claude] Claude call failed: {exc}", flush=True)
        return None

    raw = ""
    try:
        raw = msg.content[0].text if msg.content else ""
    except Exception:
        raw = ""
    if not raw:
        print("[thumb-director/plan/claude] Claude returned empty body", flush=True)
        return None

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(),
                      flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            print(f"[thumb-director/plan/claude] non-JSON: {cleaned[:200]}",
                  flush=True)
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            print(f"[thumb-director/plan/claude] truncated JSON: {exc}",
                  flush=True)
            return None

    plan = ScenePlan(
        primary_subject=str(data.get("primary_subject", "")).strip(),
        scene_location=str(data.get("scene_location", "")).strip(),
        key_action=str(data.get("key_action", "")).strip(),
        supporting_props=[str(x).strip() for x in (data.get("supporting_props") or [])
                          if str(x).strip()],
        lighting=str(data.get("lighting", "")).strip(),
        color_grade=str(data.get("color_grade", "")).strip(),
        framing=str(data.get("framing", "")).strip(),
        emotional_hook=str(data.get("emotional_hook", "")).strip(),
        what_to_avoid=str(data.get("what_to_avoid", "")).strip(),
        safe_zone=str(data.get("safe_zone", "")).strip(),
        text_for_overlay=str(data.get("text_for_overlay", "")).strip(),
        style=style,
        language=language,
        has_references=has_references,
    )
    if not plan.is_complete():
        print("[thumb-director/plan/claude] incomplete plan returned", flush=True)
        return None
    print(f"[thumb-director/plan/claude] OK style={style} "
          f"subject={plan.primary_subject[:60]!r}", flush=True)
    return plan


def plan_scene(*,
               title_native: str = "",
               title_english: str = "",
               summary: str = "",
               seo_hook: str = "",
               seo_thumbnail_text: str = "",
               seo_description: str = "",
               transcript_excerpt: str = "",
               language: str = "te",
               style: str = "symbolic",
               has_references: int = 0,
               planner: str = "gemini") -> ScenePlan:
    """Pass 1 entry point — produce a ScenePlan from news context.

    ``planner`` picks the brain:
      * "gemini" (default, ~10x cheaper) — Gemini 2.5 Pro with
        ``response_schema`` enforcement.
      * "claude" — Claude Opus 4.7. Generally stronger editorial
        specificity, ~10x cost.

    Falls back to a deterministic plan built from raw inputs when the
    chosen LLM is unreachable so the downstream prompter never starves.
    """
    planner_choice = (planner or "gemini").strip().lower()

    # Claude path: try it first, fall through to deterministic
    # fallback on failure (do NOT silently route to Gemini — that
    # would double-charge an operator who explicitly picked Claude).
    if planner_choice == "claude":
        claude_plan = _plan_via_claude(
            title_native=title_native, title_english=title_english,
            summary=summary, seo_hook=seo_hook,
            seo_thumbnail_text=seo_thumbnail_text,
            seo_description=seo_description,
            transcript_excerpt=transcript_excerpt,
            language=language, style=style, has_references=has_references,
        )
        if claude_plan is not None:
            return claude_plan
        print("[thumb-director/plan] Claude failed, using deterministic fallback",
              flush=True)
        return _fallback_plan(
            title_native=title_native, title_english=title_english,
            summary=summary, language=language, style=style,
            has_references=has_references,
        )

    try:
        from seo.generator import _gemini_client
        from google.genai import types as genai_types
    except Exception as exc:
        print(f"[thumb-director/plan] Vertex SDK unavailable, using fallback: {exc}",
              flush=True)
        return _fallback_plan(
            title_native=title_native, title_english=title_english,
            summary=summary, language=language, style=style,
            has_references=has_references,
        )

    try:
        client = _gemini_client()
    except Exception as exc:
        print(f"[thumb-director/plan] Gemini auth failed: {exc}", flush=True)
        return _fallback_plan(
            title_native=title_native, title_english=title_english,
            summary=summary, language=language, style=style,
            has_references=has_references,
        )

    sys_prompt = _build_system_prompt(
        style=style, language=language, has_references=has_references,
    )
    user_prompt = _build_user_prompt(
        title_native=title_native, title_english=title_english,
        summary=summary, seo_hook=seo_hook,
        seo_thumbnail_text=seo_thumbnail_text,
        seo_description=seo_description,
        transcript_excerpt=transcript_excerpt,
    )

    try:
        resp = client.models.generate_content(
            model=DEFAULT_PLANNER_MODEL,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=sys_prompt,
                response_mime_type="application/json",
                response_schema=SCENEPLAN_JSON_SCHEMA,
                # Low temperature: editorial decisions should be repeatable,
                # not creative. Two runs against the same story should
                # produce two near-identical plans — that's a feature.
                temperature=0.25,
                max_output_tokens=4096,
            ),
        )
    except Exception as exc:
        print(f"[thumb-director/plan] Gemini call failed: {exc}", flush=True)
        return _fallback_plan(
            title_native=title_native, title_english=title_english,
            summary=summary, language=language, style=style,
            has_references=has_references,
        )

    raw = (resp.text or "").strip()
    if not raw:
        print(f"[thumb-director/plan] Gemini returned empty body", flush=True)
        return _fallback_plan(
            title_native=title_native, title_english=title_english,
            summary=summary, language=language, style=style,
            has_references=has_references,
        )

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw,
                      flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            print(f"[thumb-director/plan] Gemini returned non-JSON: "
                  f"{cleaned[:200]}", flush=True)
            return _fallback_plan(
                title_native=title_native, title_english=title_english,
                summary=summary, language=language, style=style,
                has_references=has_references,
            )
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            print(f"[thumb-director/plan] truncated JSON: {exc}", flush=True)
            return _fallback_plan(
                title_native=title_native, title_english=title_english,
                summary=summary, language=language, style=style,
                has_references=has_references,
            )

    plan = ScenePlan(
        primary_subject=str(data.get("primary_subject", "")).strip(),
        scene_location=str(data.get("scene_location", "")).strip(),
        key_action=str(data.get("key_action", "")).strip(),
        supporting_props=[str(x).strip() for x in (data.get("supporting_props") or [])
                          if str(x).strip()],
        lighting=str(data.get("lighting", "")).strip(),
        color_grade=str(data.get("color_grade", "")).strip(),
        framing=str(data.get("framing", "")).strip(),
        emotional_hook=str(data.get("emotional_hook", "")).strip(),
        what_to_avoid=str(data.get("what_to_avoid", "")).strip(),
        safe_zone=str(data.get("safe_zone", "")).strip(),
        text_for_overlay=str(data.get("text_for_overlay", "")).strip(),
        style=style,
        language=language,
        has_references=has_references,
    )

    if not plan.is_complete():
        # Gemini returned a plan but it's missing critical fields. Log
        # the gap and merge with the deterministic fallback so we ship
        # the best of both.
        print(f"[thumb-director/plan] incomplete plan from Gemini — "
              f"merging with fallback", flush=True)
        fallback = _fallback_plan(
            title_native=title_native, title_english=title_english,
            summary=summary, language=language, style=style,
            has_references=has_references,
        )
        if not plan.primary_subject: plan.primary_subject = fallback.primary_subject
        if not plan.scene_location:  plan.scene_location  = fallback.scene_location
        if not plan.key_action:      plan.key_action      = fallback.key_action
        if not plan.lighting:        plan.lighting        = fallback.lighting
        if not plan.color_grade:     plan.color_grade     = fallback.color_grade
        if not plan.framing:         plan.framing         = fallback.framing
        if not plan.emotional_hook:  plan.emotional_hook  = fallback.emotional_hook
        if not plan.what_to_avoid:   plan.what_to_avoid   = fallback.what_to_avoid
        if not plan.safe_zone:       plan.safe_zone       = fallback.safe_zone

    print(f"[thumb-director/plan] OK style={style} subject={plan.primary_subject[:60]!r}",
          flush=True)
    return plan


__all__ = ["plan_scene", "DEFAULT_PLANNER_MODEL"]
