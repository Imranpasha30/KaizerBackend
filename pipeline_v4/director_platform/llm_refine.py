# Ported from kaizer-platform@d5fd482 server/pipeline_core/ai_director/llm_refine.py
# Changes from upstream: their shared llm.providers.resolve_chain is replaced
# with a direct provider loop over OUR clients (seo.generator._gemini_client
# for Gemini, anthropic.Anthropic for Claude). Prompt text, JSON contract,
# env var (KAIZER_DIRECTOR_LLM_CHAIN, default "gemini,claude") and the
# FAIL-OPEN contract are unchanged.
"""Layer 3 of the platform AI Director ("LLM"): reviews the formula layer's
candidate against the story's actual topic/script text and may override it —
e.g. a tragedy shouldn't get the "energetic" pack even if crowd noise made
the audio read hot. Never raises; every failure keeps the formula candidate."""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from pipeline_v4.director_platform.formula import FormulaCandidate
from pipeline_v4.director_platform.sensors import SensorReadings
from pipeline_v4.director_platform.style_vocab import PLATFORM_PACKS

logger = logging.getLogger("kaizer.pipeline_v4.director_platform.llm_refine")

_DEFAULT_CHAIN = "gemini,claude"
_MAX_STORY_CHARS = 4_000
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class RefinedDirection:
    """Layer-3 output: the final mood/style_pack decision after LLM review."""

    mood: str
    style_pack: str
    reason: str
    overridden: bool
    provider_used: Optional[str]


def _resolve_chain_name(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    return (
        os.environ.get("KAIZER_DIRECTOR_LLM_CHAIN", "") or ""
    ).strip() or _DEFAULT_CHAIN


def _build_prompt(
    candidate: FormulaCandidate, sensors: SensorReadings, story_text: str,
) -> str:
    style_names = ", ".join(sorted(PLATFORM_PACKS.keys()))
    return f"""You are the final review step of an automated video-direction
system for a news video pipeline. A deterministic rule engine ("formula
layer") already measured the clip's audio/visual signals and proposed a
candidate treatment. Your job is to catch cases where the RAW SIGNALS are
misleading given what the story is actually ABOUT — e.g. a tragedy or
somber story should never get an "energetic"/"vibrant" treatment even if
crowd noise or a siren made the audio read loud.

STORY TEXT (topic/script — read this for tone and subject matter):
{story_text.strip()[:_MAX_STORY_CHARS]}

MEASURED SIGNALS (layer 1 — sensors):
  audio_rms_mean        = {sensors.audio_rms_mean:.4f}
  audio_rms_peak        = {sensors.audio_rms_peak:.4f}
  audio_energy_variance = {sensors.audio_energy_variance:.4f}
  integrated_lufs       = {sensors.integrated_lufs}
  speech_pace_wps       = {sensors.speech_pace_wps:.2f}
  scene_change_rate/min = {sensors.scene_change_rate_per_min:.1f}
  avg_brightness        = {sensors.avg_brightness:.2f}
  avg_saturation        = {sensors.avg_saturation:.2f}

FORMULA LAYER'S CANDIDATE (layer 2 — deterministic rule table):
  mood        = {candidate.mood}
  style_pack  = {candidate.style_pack}
  rule fired  = {candidate.rule_id}
  rule reason = {candidate.reason}

Valid style_pack values: {style_names}

Decide whether to CONFIRM the formula candidate or OVERRIDE it based on
the story's actual subject matter and tone. Respond with ONLY a JSON
object, no markdown fences, no prose outside the JSON:

{{"override": true|false, "mood": "<one word mood tag>", "style_pack": "<one of the valid values above>", "reason": "<one sentence, specific to this story>"}}
"""


def _extract_json(text: str) -> Optional[dict]:
    match = _JSON_BLOCK_RE.search(text or "")
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# ── Provider loop (replaces upstream llm.providers.resolve_chain) ──────────

# One client for the whole per-story loop. Module-level (not per call): a
# fresh Vertex client per story pays SA-credential load N times, and in
# google-genai 2.4.0 an unreferenced temporary Client can be GC'd
# mid-request (the exact bug director.py:1152-1157 documents).
_GEMINI_CLIENT = None


def _gemini() :
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        from seo.generator import _gemini_client
        _GEMINI_CLIENT = _gemini_client()
    return _GEMINI_CLIENT


def _call_gemini_text(prompt: str) -> str:
    from google.genai import types as genai_types
    client = _gemini()
    model = os.environ.get("KAIZER_V4_DIRECTOR_MODEL", "gemini-2.5-flash")
    # 2.5 thinking models spend max_output_tokens on THINKING first — the
    # upstream 400-token cap truncated every reply to empty on Vertex
    # (this repo already diagnosed the identical bug at director.py:292).
    # 2048 gives thinking headroom; the reply JSON itself is ~50 tokens.
    _tries = max(1, int(os.environ.get(
        "KAIZER_V4_DIRECTOR_RETRIES", "3") or 3))
    resp = None
    for _at in range(_tries):
        try:
            from pipeline_v4.api_pace import pace
            pace()  # Vertex quota is PER-MINUTE; space request starts
            resp = client.models.generate_content(
                model=model, contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.2, max_output_tokens=2048,
                    response_mime_type="application/json"),
            )
            break
        except Exception as _rexc:
            if "429" in str(_rexc) and _at < _tries - 1:
                import time as _time
                _wait = 30 * (_at + 1)
                logger.warning(
                    "director_platform: Gemini 429 per-minute quota — "
                    "retry %d/%d in %ds", _at + 2, _tries, _wait)
                _time.sleep(_wait)
                continue
            raise
    return (resp.text or "").strip()


def _call_claude_text(prompt: str) -> str:
    from anthropic import Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    model = os.environ.get("KAIZER_SEO_CLAUDE_MODEL", "claude-sonnet-4-6")
    client = Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model, max_tokens=400, temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    return (msg.content[0].text if msg.content else "").strip()


_PROVIDERS = {"gemini": _call_gemini_text, "claude": _call_claude_text}


def _resolve_chain(chain: str, prompt: str) -> tuple:
    """First provider in the comma chain that returns text wins.
    Raises RuntimeError when every provider fails (caller fails open)."""
    last: Optional[Exception] = None
    for name in [c.strip().lower() for c in chain.split(",") if c.strip()]:
        fn = _PROVIDERS.get(name)
        if fn is None:
            continue
        try:
            text = fn(prompt)
            if text:
                return text, name
        except Exception as exc:  # noqa: BLE001 — chain semantics
            last = exc
            continue
    raise RuntimeError(f"all providers failed (last: {last})")


# ── Public API ──────────────────────────────────────────────────────────────

def refine_with_llm(
    candidate: FormulaCandidate,
    sensors: SensorReadings,
    story_text: Optional[str],
    *,
    provider_chain: Optional[str] = None,
) -> RefinedDirection:
    """Ask an LLM to confirm or override *candidate* given *story_text*.
    Never raises — FAIL-OPEN on every failure mode."""
    if not story_text or not story_text.strip():
        return RefinedDirection(
            mood=candidate.mood,
            style_pack=candidate.style_pack,
            reason="No story text provided; formula candidate kept as-is.",
            overridden=False,
            provider_used=None,
        )

    prompt = _build_prompt(candidate, sensors, story_text)
    chain = _resolve_chain_name(provider_chain)

    try:
        text, provider_used = _resolve_chain(chain, prompt)
    except Exception as exc:
        logger.warning(
            "director_platform.llm_refine: LLM unavailable/failed (%s); "
            "keeping formula candidate %s/%s",
            exc, candidate.mood, candidate.style_pack,
        )
        return RefinedDirection(
            mood=candidate.mood,
            style_pack=candidate.style_pack,
            reason=f"LLM refinement unavailable ({exc}); formula candidate kept.",
            overridden=False,
            provider_used=None,
        )

    parsed = _extract_json(text)
    if not parsed:
        return RefinedDirection(
            mood=candidate.mood,
            style_pack=candidate.style_pack,
            reason="LLM response unparseable; formula candidate kept.",
            overridden=False,
            provider_used=provider_used,
        )

    should_override = bool(parsed.get("override", False))
    llm_reason = str(parsed.get("reason") or "").strip() or "LLM confirmed formula candidate."
    # Mood is a free-text tag (the prompt never enumerates valid moods) —
    # fold case so "Somber" == "somber". The adapter keys treatment off the
    # VALIDATED style_pack, so mood is decorative trail data, like upstream.
    llm_mood = str(parsed.get("mood") or "").strip().lower() or candidate.mood
    llm_style_pack = str(parsed.get("style_pack") or "").strip()

    if not should_override:
        return RefinedDirection(
            mood=candidate.mood,
            style_pack=candidate.style_pack,
            reason=llm_reason,
            overridden=False,
            provider_used=provider_used,
        )

    if llm_style_pack not in PLATFORM_PACKS:
        return RefinedDirection(
            mood=candidate.mood,
            style_pack=candidate.style_pack,
            reason=(
                f"LLM requested override to unknown style_pack "
                f"{llm_style_pack!r}; keeping formula candidate. "
                f"LLM reason: {llm_reason}"
            ),
            overridden=False,
            provider_used=provider_used,
        )

    return RefinedDirection(
        mood=llm_mood,
        style_pack=llm_style_pack,
        reason=llm_reason,
        overridden=True,
        provider_used=provider_used,
    )
