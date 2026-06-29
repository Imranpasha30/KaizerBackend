"""Thumbnail Director — 2-stage orchestration for Nano Banana prompts.

The director replaces the legacy "Gemini reads SEO → writes one prompt"
single-pass approach with a proper art-director flow:

  Pass 1 (planner.py)    : news context → structured ``ScenePlan`` JSON
                            (Gemini 2.5 Pro OR Claude Opus 4.7)
  Pass 2 (prompter.py)   : ``ScenePlan`` → Nano Banana image prompt
                            (Gemini 2.5 Pro OR Claude Opus 4.7)

The brain behind each pass is operator-selectable via the ``engine``
preset (gemini / hybrid / claude) — see DIRECTOR_PRESETS below.

Splitting the work this way fixes the recurring failure mode where the
single-pass writer would skim the news, produce a generic "newsroom
anchor" prompt, and Nano Banana would deliver a forgettable thumbnail.
The plan IS the editorial decision — the prompt just executes it.

Public API (the only function callers should import):

    from pipeline_v4.thumbnail_director import build_thumbnail_prompt

    final_prompt = build_thumbnail_prompt(
        title_native="...",
        title_english="...",
        summary="...",
        seo_hook="...",
        seo_thumbnail_text="...",
        seo_description="...",
        transcript_excerpt="...",
        language="te",
        style="reporter_led",
        has_references=1,
    )

Module ownership (so teammates can iterate without conflicts):
  * schemas.py    — ScenePlan dataclass (stable contract; rarely changes)
  * templates.py  — per-style required-fields catalog (product team)
  * examples.py   — few-shot bank (content / curator team)
  * planner.py    — Gemini Pass 1 (planning team)
  * prompter.py   — Gemini Pass 2 (prompt-engineering team)

Adding a new style is a templates.py + examples.py edit only — no other
file changes needed. Adding a new few-shot example is examples.py only.
"""
from __future__ import annotations

from typing import Optional

from .schemas import ScenePlan
from .planner import plan_scene
from .prompter import prompt_from_plan


# ── Director engine presets ─────────────────────────────────────────
#
# Each preset maps to a (planner, prompter) pair so the operator picks
# one tier instead of two independent toggles.
#
# Pricing (per thumbnail, very approximate):
#   gemini  → 2× Gemini 2.5 Pro     ~ $0.002
#   hybrid  → Claude + Gemini       ~ $0.012
#   claude  → 2× Claude Opus 4.7    ~ $0.025
#
# Quality (in our internal testing):
#   gemini  — decent, tends generic on edge cases
#   hybrid  — best practical default: Claude's editorial specificity
#             without the full Claude-tier cost
#   claude  — marginal lift over hybrid on the prompter side; mainly
#             useful for A/B comparison

DIRECTOR_PRESETS: dict[str, dict] = {
    "gemini": {
        "planner":  "gemini",
        "prompter": "gemini",
        "label":    "Gemini (cheap, ~$0.002/thumbnail)",
    },
    "hybrid": {
        "planner":  "claude",
        "prompter": "gemini",
        "label":    "Hybrid: Claude planner + Gemini prompter (~$0.012)",
    },
    "claude": {
        "planner":  "claude",
        "prompter": "claude",
        "label":    "Claude (premium, ~$0.025/thumbnail)",
    },
}

DEFAULT_DIRECTOR_ENGINE = "gemini"


def _resolve_engine(engine: str) -> dict:
    """Map an engine preset key to its (planner, prompter) pair.
    Unknown engines fall back to the default so a stale frontend
    never starves the pipeline."""
    key = (engine or DEFAULT_DIRECTOR_ENGINE).strip().lower()
    return DIRECTOR_PRESETS.get(key) or DIRECTOR_PRESETS[DEFAULT_DIRECTOR_ENGINE]


def build_thumbnail_prompt(
    *,
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
    engine: str = DEFAULT_DIRECTOR_ENGINE,
    text_mode: str = "overlay",
) -> str:
    """End-to-end 2-stage thumbnail prompt generation.

    Pass 1 builds a structured ``ScenePlan`` from the news context.
    Pass 2 converts the plan into a Nano Banana image prompt.

    ``engine`` picks which brain runs each pass — see DIRECTOR_PRESETS.
    ``text_mode`` picks how the shout text lands on the image:
      * "overlay" — Pillow paints the text after generation
                    (Indic typography guaranteed correct).
      * "ai"      — image model renders the text directly
                    (more dramatic, but may garble Indic ligatures).

    Both stages soft-fail individually. If Pass 1 fails the prompter
    still produces a (weaker) prompt from raw context. If Pass 2 fails
    we serialise the plan into a prose paragraph as a last-resort
    fallback so the image step never starves.
    """
    pair = _resolve_engine(engine)
    plan = plan_scene(
        title_native=title_native,
        title_english=title_english,
        summary=summary,
        seo_hook=seo_hook,
        seo_thumbnail_text=seo_thumbnail_text,
        seo_description=seo_description,
        transcript_excerpt=transcript_excerpt,
        language=language,
        style=style,
        has_references=has_references,
        planner=pair["planner"],
    )
    return prompt_from_plan(
        plan=plan,
        language=language,
        style=style,
        has_references=has_references,
        prompter=pair["prompter"],
        text_mode=text_mode,
    )


__all__ = [
    "ScenePlan",
    "build_thumbnail_prompt",
    "plan_scene",
    "prompt_from_plan",
    "DIRECTOR_PRESETS",
    "DEFAULT_DIRECTOR_ENGINE",
]
