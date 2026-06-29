"""Pass 2 — turn a ScenePlan into a Nano Banana image prompt.

Owned by the prompt-engineering team. Reads ScenePlans, writes prompts
optimized for the way image-generation models actually attend to text.

Prompt engineering rules baked into the system prompt:
  * Front-load the VISUAL (subject + location + action) — Nano Banana
    weights early tokens more heavily.
  * Concrete nouns + prepositional phrases beat adjectives — "a stack of
    ₹500 rupee bundles on a wooden government desk" beats "an Indian
    investigative scene".
  * End with style + technical constraints (16:9, no text, no logos)
    so the model has those instructions fresh when finalising.
  * For reference styles, the FIRST sentence of the output must be
    the verbatim identity-preservation block — image models pay
    disproportionate attention to opening tokens, and verbatim repeats
    survive paraphrasing-by-LLM better than restated directives.

Soft-fails to ScenePlan.to_prose() when:
  * Vertex SDK can't be imported
  * Gemini auth fails
  * Gemini returns an empty body

The prose fallback is shorter than a real prompter output but Nano
Banana still produces a usable image from it — better than nothing.
"""
from __future__ import annotations

import os
import re
from typing import Optional

from .schemas import ScenePlan
from .examples import examples_for_style


DEFAULT_PROMPTER_MODEL = os.environ.get(
    "KAIZER_THUMB_PROMPTER_MODEL", "gemini-2.5-pro"
)


# Hard-coded verbatim identity-preservation block — when the style needs
# a reference image, the first sentence of the output prompt MUST be
# this exact text. Burying it inside paraphrase is what was letting
# Nano Banana swap genders.

_IDENTITY_BLOCK_TEMPLATE = (
    "Use the attached reference photograph as the EXACT identity of "
    "the {role}. Preserve gender, age, skin tone, facial bone structure, "
    "hair (length, colour, style), beard / clean-shaven state, glasses "
    "or no glasses, and clothing style EXACTLY as shown — DO NOT change "
    "the gender, DO NOT replace with a stock {role}, the output must be "
    "recognisably the same individual as the reference."
)

_TWO_REF_IDENTITY_BLOCK = (
    "Two reference photographs are attached. The FIRST attached image is "
    "the LEFT subject. The SECOND attached image is the RIGHT subject. "
    "For EACH subject, preserve gender, age, skin tone, facial bone "
    "structure, hair (length, colour, style), beard / clean-shaven "
    "state, glasses or no glasses, and clothing style EXACTLY as shown "
    "in their respective reference. DO NOT swap the subjects. DO NOT "
    "make them look alike. DO NOT replace either with a generic stand-in."
)


def _identity_block_for(style: str, has_references: int) -> str:
    """Return the verbatim identity block that must open the prompt,
    or "" when the style doesn't use references."""
    if has_references >= 2 and style == "split_screen":
        return _TWO_REF_IDENTITY_BLOCK
    if has_references >= 1 and style in ("reporter_led", "subject_photo"):
        role = "reporter" if style == "reporter_led" else "subject"
        return _IDENTITY_BLOCK_TEMPLATE.format(role=role)
    return ""


def _build_system_prompt(*, style: str, has_references: int,
                          text_mode: str = "overlay") -> str:
    """Compose the prompter system instruction.

    ``text_mode`` flips between two pipelines:
      * "overlay"  — AI plate is text-less; Pillow paints the native-
                     script shout text after generation. Guarantees
                     perfect Telugu / Hindi typography but the text
                     looks like a sticker.
      * "ai"       — AI renders the text directly in the image with
                     dramatic typography (3D extrusion, glow, motion).
                     Looks more dramatic and integrated, but Indic
                     ligatures may garble (operator's call to take
                     this risk for the visual lift).
    """
    # Few-shot examples: the curated exemplars in examples.py were
    # written for the OVERLAY path ("Leave the safe zone visually
    # quieter, NO text rendered inside the image"). Showing those to
    # Gemini when text_mode=ai contradicts the new "render the text
    # dramatically inside the image" rule — the model would err toward
    # the concrete examples and produce text-less plates.
    #
    # Until we curate ai-mode exemplars, skip the shots block entirely
    # for text_mode=ai. The mechanical rules + the ScenePlan are
    # enough signal on their own.
    shots_block = ""
    if text_mode == "overlay":
        shots = examples_for_style(style, max_count=2)
        if shots:
            parts = []
            for i, shot in enumerate(shots, start=1):
                parts.append(
                    f"---\nExample {i} ScenePlan:\n"
                    f"{shot['plan']}\n\n"
                    f"Example {i} prompt (shape-of-output reference):\n"
                    f"{shot['prompt']}"
                )
            shots_block = "\n\n# Few-shot examples (shape only)\n\n" + "\n\n".join(parts)

    # Text-handling block: flips between asking the model to RENDER
    # the text dramatically vs to LEAVE A SAFE ZONE for an external
    # overlay. Everything else (locked constraint, mechanical rules,
    # output format) stays identical.
    if text_mode == "ai":
        text_rules = (
            "3. The prompt MUST instruct the image model to render the "
            "plan's text_for_overlay PROMINENTLY inside the image with "
            "dramatic broadcast-graphics typography — bold, large (30-"
            "45% of frame), 3D extrusion or heavy drop shadow, bright "
            "fill (yellow / white) with thick contrasting outline "
            "(black / red). The text must be the single most eye-"
            "catching element after the primary subject.\n"
            "4. Quote the text_for_overlay characters EXACTLY (inside "
            "double quotes in the prompt) so the model rasterises them "
            "verbatim. Add 'render the text as perfect, flawless "
            "typography matching the quoted characters exactly — no "
            "spelling drift, no missing ligatures, treat it as high-"
            "contrast vector graphic text'.\n"
        )
    else:
        text_rules = (
            "3. The prompt MUST include 'NO text rendered inside the "
            "image' — text is painted externally by Pillow.\n"
            "4. The prompt MUST include the plan's safe-zone description "
            "so the AI plate leaves that area visually quiet.\n"
        )

    return (
        "# You are a prompt engineer for image-generation models\n"
        "(Nano Banana / gpt-image-1 / Imagen). You receive a structured "
        "ScenePlan from an art director and turn it into a single image "
        "prompt.\n\n"

        "# The ONE locked constraint\n"
        "**The output must look like an Indian regional news-channel "
        "thumbnail.** Photographic, dramatic, Indian visual vocabulary. "
        "NOT generic stock, NOT American.\n\n"

        "# Hard mechanical rules (these are NOT creative choices)\n"
        "1. Front-load the VISUAL (subject + location + action) — image "
        "models weight early tokens more heavily.\n"
        "2. Use concrete nouns + prepositional phrases. Concrete beats "
        "abstract every time for image models.\n"
        f"{text_rules}"
        "5. The prompt MUST include the plan's what_to_avoid items as "
        "explicit negative 'NO ...' instructions.\n"
        "6. Length: 200-350 words. Single paragraph. ONE prompt string.\n\n"

        "# Your freedom\n"
        "Everything else — word choice, compositional details beyond what "
        "the plan specifies, lighting nuance, atmospheric phrasing — is "
        "your call. Make the prompt vivid and specific.\n\n"

        "# Output format\n"
        "Output ONE prompt string. No preamble, no JSON, no markdown "
        "fences. The prompt is sent to the image model verbatim, so don't "
        "include any meta-commentary."
        f"{shots_block}"
    )


def _scene_plan_to_user_message(plan: ScenePlan) -> str:
    """Serialise the ScenePlan into a tidy block that Gemini can read."""
    lines = [
        "Convert this ScenePlan into a Nano Banana image prompt.",
        "",
        f"primary_subject:   {plan.primary_subject}",
        f"scene_location:    {plan.scene_location}",
        f"key_action:        {plan.key_action}",
    ]
    if plan.supporting_props:
        lines.append("supporting_props:  - " +
                      "\n                    - ".join(plan.supporting_props))
    lines.extend([
        f"lighting:          {plan.lighting}",
        f"color_grade:       {plan.color_grade}",
        f"framing:           {plan.framing}",
        f"emotional_hook:    {plan.emotional_hook}",
        f"what_to_avoid:     {plan.what_to_avoid}",
        f"safe_zone:         {plan.safe_zone}",
    ])
    if plan.text_for_overlay:
        lines.append(
            f"text_for_overlay:  {plan.text_for_overlay} "
            f"(INFORMATIONAL — Pillow paints this externally; "
            f"do NOT render any text inside the image)"
        )
    return "\n".join(lines)


def _prompt_via_claude(*, plan: ScenePlan,
                       style: str, has_references: int,
                       text_mode: str = "overlay") -> Optional[str]:
    """Pass 2 via Claude Opus 4.7. Same system prompt + same user
    serialisation as Gemini, different brain. Returns the prompt string
    or None on any failure so the caller can fall through to the
    deterministic prose fallback.
    """
    try:
        from anthropic import Anthropic
    except Exception as exc:
        print(f"[thumb-director/prompt/claude] Anthropic SDK unavailable: {exc}",
              flush=True)
        return None
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("[thumb-director/prompt/claude] ANTHROPIC_API_KEY not set", flush=True)
        return None

    sys_prompt = _build_system_prompt(
        style=style, has_references=has_references, text_mode=text_mode,
    )
    user_msg = _scene_plan_to_user_message(plan)

    from contextlib import nullcontext
    try:
        from learning.claude_log import log_anthropic_call as _log_anthropic
        _cm = _log_anthropic(db=None, model="claude-opus-4-7", purpose="thumbnail-prompt")
    except Exception:
        _cm = nullcontext(None)

    try:
        client = Anthropic(api_key=api_key)
        with _cm as _acall:
            msg = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=2048,
                system=sys_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            if _acall is not None:
                _acall.record(msg)
    except Exception as exc:
        print(f"[thumb-director/prompt/claude] Claude call failed: {exc}", flush=True)
        return None

    text = ""
    try:
        text = (msg.content[0].text if msg.content else "").strip()
    except Exception:
        text = ""
    if not text:
        print("[thumb-director/prompt/claude] empty body", flush=True)
        return None

    # Strip accidental code fences / preambles, same hygiene the Gemini
    # path does.
    text = re.sub(r"^```(?:[a-z]*)?\s*|\s*```$", "", text,
                   flags=re.IGNORECASE | re.DOTALL).strip()
    text = re.sub(r"^(prompt|nano banana prompt)\s*[:\-]\s*", "", text,
                   flags=re.IGNORECASE).strip()

    # Enforce identity-block opener for reference styles. Belt-and-braces
    # the same way the Gemini path does.
    required_opener = _identity_block_for(style, has_references)
    if required_opener and required_opener[:80] not in text[:300]:
        text = f"{required_opener} {text}"

    print(f"[thumb-director/prompt/claude] OK style={style} len={len(text)}",
          flush=True)
    return text


def prompt_from_plan(*, plan: ScenePlan,
                      language: str = "te",
                      style: str = "symbolic",
                      has_references: int = 0,
                      prompter: str = "gemini",
                      text_mode: str = "overlay") -> str:
    """Pass 2 entry point — write a Nano Banana prompt from a ScenePlan.

    ``prompter`` picks the brain (gemini | claude).
    ``text_mode`` picks how the shout text gets onto the image:
      * "overlay" — prompt asks image model to leave a safe zone;
                    Pillow paints the text post-generation.
      * "ai"      — prompt instructs the image model to render the
                    text dramatically inside the image; Pillow skipped.

    Falls back to ``ScenePlan.to_prose()`` when the chosen LLM is
    unreachable so the caller always gets a usable prompt string.
    """
    prompter_choice = (prompter or "gemini").strip().lower()
    text_mode = (text_mode or "overlay").strip().lower()
    if text_mode not in ("ai", "overlay"):
        text_mode = "overlay"

    if prompter_choice == "claude":
        claude_prompt = _prompt_via_claude(
            plan=plan, style=style, has_references=has_references,
            text_mode=text_mode,
        )
        if claude_prompt:
            return claude_prompt
        print("[thumb-director/prompt] Claude failed, using prose fallback",
              flush=True)
        return plan.to_prose()

    try:
        from seo.generator import _gemini_client
        from google.genai import types as genai_types
    except Exception as exc:
        print(f"[thumb-director/prompt] Vertex SDK unavailable, using prose fallback: {exc}",
              flush=True)
        return plan.to_prose()

    try:
        client = _gemini_client()
    except Exception as exc:
        print(f"[thumb-director/prompt] Gemini auth failed: {exc}", flush=True)
        return plan.to_prose()

    sys_prompt = _build_system_prompt(
        style=style, has_references=has_references, text_mode=text_mode,
    )
    user_msg = _scene_plan_to_user_message(plan)

    try:
        resp = client.models.generate_content(
            model=DEFAULT_PROMPTER_MODEL,
            contents=user_msg,
            config=genai_types.GenerateContentConfig(
                system_instruction=sys_prompt,
                # Prompter is creative-ish — 0.5 lets it choose words
                # well without going off-script. Higher temps started
                # paraphrasing the verbatim identity block.
                temperature=0.5,
                max_output_tokens=4096,
            ),
        )
    except Exception as exc:
        print(f"[thumb-director/prompt] Gemini call failed: {exc}", flush=True)
        return plan.to_prose()

    text = (resp.text or "").strip()
    if not text:
        print(f"[thumb-director/prompt] Gemini returned empty body, using prose fallback",
              flush=True)
        return plan.to_prose()

    # Strip any accidental markdown decoration. Gemini sometimes wraps
    # in ``` or prefixes with "Prompt:" despite the system instruction.
    text = re.sub(r"^```(?:[a-z]*)?\s*|\s*```$", "", text,
                   flags=re.IGNORECASE | re.DOTALL).strip()
    text = re.sub(r"^(prompt|nano banana prompt)\s*[:\-]\s*", "", text,
                   flags=re.IGNORECASE).strip()

    # Enforce the identity-block opener for reference styles. If Gemini
    # paraphrased the block despite the system instruction, prepend the
    # verbatim version. Better belt-and-braces than the model drifting.
    required_opener = _identity_block_for(style, has_references)
    if required_opener and required_opener[:80] not in text[:300]:
        text = f"{required_opener} {text}"

    print(f"[thumb-director/prompt] OK style={style} len={len(text)} preview={text[:100]!r}",
          flush=True)
    return text


__all__ = ["prompt_from_plan", "DEFAULT_PROMPTER_MODEL"]
