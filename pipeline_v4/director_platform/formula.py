# Ported from kaizer-platform@d5fd482 server/pipeline_core/ai_director/formula.py
# Changes from upstream: import path only (sensors now local). The 5-rule
# ladder, thresholds and mood→pack table are verbatim.
"""Layer 2 of the platform AI Director ("formula"): a deterministic rule
table mapping ``SensorReadings`` to a candidate direction — a mood tag plus
a platform style-pack name (see style_vocab.PLATFORM_PACKS)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from pipeline_v4.director_platform.sensors import SensorReadings

# ── Tunable thresholds (upstream values; RMS of [-1,1] float samples) ──
HIGH_ENERGY_RMS: float = 0.08
LOW_ENERGY_RMS: float = 0.02
FAST_PACE_WPS: float = 2.6
SLOW_PACE_WPS: float = 1.6
HIGH_CUT_RATE_PER_MIN: float = 20.0
DARK_BRIGHTNESS: float = 0.35
BRIGHT_BRIGHTNESS: float = 0.55

MOOD_TAGS: tuple = (
    "energetic", "urgent", "somber", "cinematic", "neutral",
)


@dataclass(frozen=True)
class FormulaCandidate:
    """Layer-2 output: candidate direction + the rule that produced it."""

    mood: str
    style_pack: str
    rule_id: str
    reason: str


# ── Rule functions (first match wins) ───────────────────────────────────────

def _rule_energetic(s: SensorReadings) -> Optional[FormulaCandidate]:
    if s.audio_rms_mean >= HIGH_ENERGY_RMS and s.speech_pace_wps >= FAST_PACE_WPS:
        return FormulaCandidate(
            mood="energetic",
            style_pack="vibrant",
            rule_id="energetic_fast_pace",
            reason=(
                f"high audio energy (rms_mean={s.audio_rms_mean:.4f} >= "
                f"{HIGH_ENERGY_RMS}) + fast speech pace "
                f"({s.speech_pace_wps:.2f} wps >= {FAST_PACE_WPS}) → "
                "punchy zoom-punch transitions + saturated grade"
            ),
        )
    return None


def _rule_urgent_cuts(s: SensorReadings) -> Optional[FormulaCandidate]:
    if s.scene_change_rate_per_min >= HIGH_CUT_RATE_PER_MIN:
        return FormulaCandidate(
            mood="urgent",
            style_pack="news_flash",
            rule_id="high_cut_rate",
            reason=(
                f"high scene-change rate "
                f"({s.scene_change_rate_per_min:.1f}/min >= "
                f"{HIGH_CUT_RATE_PER_MIN}) → source is already fast-cut; "
                "match it with whip-pan urgency + red-warm grade"
            ),
        )
    return None


def _rule_somber(s: SensorReadings) -> Optional[FormulaCandidate]:
    if s.audio_rms_mean <= LOW_ENERGY_RMS and s.speech_pace_wps <= SLOW_PACE_WPS:
        return FormulaCandidate(
            mood="somber",
            style_pack="calm",
            rule_id="low_energy_slow_pace",
            reason=(
                f"low audio energy (rms_mean={s.audio_rms_mean:.4f} <= "
                f"{LOW_ENERGY_RMS}) + slow speech pace "
                f"({s.speech_pace_wps:.2f} wps <= {SLOW_PACE_WPS}) → "
                "slow dissolve transitions + desaturated cool-blue grade"
            ),
        )
    return None


def _rule_cinematic(s: SensorReadings) -> Optional[FormulaCandidate]:
    if (
        DARK_BRIGHTNESS < s.avg_brightness < BRIGHT_BRIGHTNESS
        and LOW_ENERGY_RMS < s.audio_rms_mean < HIGH_ENERGY_RMS
    ):
        return FormulaCandidate(
            mood="cinematic",
            style_pack="cinematic",
            rule_id="balanced_midtone",
            reason=(
                f"mid-range brightness ({s.avg_brightness:.2f}) + mid-range "
                f"audio energy ({s.audio_rms_mean:.4f}) → warm cinematic "
                "grade with gentle Ken Burns motion"
            ),
        )
    return None


def _rule_default(s: SensorReadings) -> FormulaCandidate:
    return FormulaCandidate(
        mood="neutral",
        style_pack="minimal",
        rule_id="default_fallback",
        reason="No rule matched the measured signals; defaulting to a clean, unobtrusive treatment.",
    )


_RULE_LADDER: list = [
    _rule_energetic,
    _rule_urgent_cuts,
    _rule_somber,
    _rule_cinematic,
]


# ── Public API ──────────────────────────────────────────────────────────────

def apply_formula(sensors: SensorReadings) -> FormulaCandidate:
    """Run the rule ladder; first match wins, default fallback otherwise."""
    for rule in _RULE_LADDER:
        candidate = rule(sensors)
        if candidate is not None:
            return candidate
    return _rule_default(sensors)
