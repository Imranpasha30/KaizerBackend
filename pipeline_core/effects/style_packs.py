"""
kaizer.pipeline.effects.style_packs
======================================
Coherent preset bundles ("style packs") for the beta rendering pipeline.

Each StylePack specifies a coherent set of transition, colour grade, camera
motion, and text/caption animation choices. The five built-in packs cover the
most common creator aesthetics:

  minimal      → clean cuts, no grade, typewriter captions
  cinematic    → warm grade, Ken Burns, cinematic fades
  news_flash   → red/warm, whip-pan, typewriter urgency
  vibrant      → saturated, zoom-punch, word-pop captions
  calm         → cool-blue, slow dissolve, sliding captions

Usage
-----
    from pipeline_core.effects.style_packs import get_style_pack, list_style_packs

    pack = get_style_pack('cinematic')
    print(pack.color_preset, pack.transition)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "StylePack",
    "STYLE_PACKS",
    "list_style_packs",
    "get_style_pack",
]


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class StylePack:
    """A coherent bundle of visual-effects settings for one aesthetic.

    Attributes
    ----------
    name : str
        Machine identifier (used as dict key).
    label : str
        User-facing display name.
    description : str
        Short description for UI tooltips.
    transition : str
        Transition name from SUPPORTED_TRANSITIONS.
    transition_duration_s : float
        Duration of each clip-to-clip transition (seconds).
    color_preset : str
        Preset name from COLOR_PRESETS.
    motion : str | None
        Motion name from SUPPORTED_MOTIONS, or None to skip.
    motion_intensity : float
        Fractional zoom intensity passed to MotionSpec.
    text_animation : str
        Animation name used for hook/headline text.
    caption_animation : str
        Animation name used for spoken-word captions.
    """

    name: str
    label: str
    description: str
    transition: str
    transition_duration_s: float = 0.5
    color_preset: str = "none"
    motion: Optional[str] = None
    motion_intensity: float = 0.08
    text_animation: str = "bounce_in"
    caption_animation: str = "karaoke"


# ── Built-in packs ────────────────────────────────────────────────────────────

STYLE_PACKS: dict[str, StylePack] = {
    "minimal": StylePack(
        name="minimal",
        label="Minimal",
        description=(
            "Clean cuts, no colour grade. "
            "Good when the content carries itself."
        ),
        transition="fade",
        transition_duration_s=0.3,
        color_preset="none",
        motion=None,
        motion_intensity=0.0,
        text_animation="slide_up",
        caption_animation="typewriter",
    ),
    "cinematic": StylePack(
        name="cinematic",
        label="Cinematic",
        description=(
            "Warm colour grade, gentle Ken Burns, "
            "cinematic fade transitions."
        ),
        transition="fade",
        transition_duration_s=0.6,
        color_preset="cinematic_warm",
        motion="ken_burns_in",
        motion_intensity=0.06,
        text_animation="bounce_in",
        caption_animation="word_pop",
    ),
    "news_flash": StylePack(
        name="news_flash",
        label="News Flash",
        description=(
            "Urgent red/warm push, whip-pan transitions, "
            "typewriter captions."
        ),
        transition="whip_pan",
        transition_duration_s=0.35,
        color_preset="news_red",
        motion="zoom_focus",
        motion_intensity=0.10,
        text_animation="typewriter",
        caption_animation="typewriter",
    ),
    "vibrant": StylePack(
        name="vibrant",
        label="Vibrant",
        description=(
            "High-saturation punchy colour, zoom-punch cuts, "
            "word-pop captions."
        ),
        transition="zoom_punch",
        transition_duration_s=0.4,
        color_preset="vibrant",
        motion="ken_burns_out",
        motion_intensity=0.10,
        text_animation="word_pop",
        caption_animation="word_pop",
    ),
    "calm": StylePack(
        name="calm",
        label="Calm",
        description=(
            "Cool-blue grade, slow dissolve transitions, "
            "sliding captions."
        ),
        transition="dissolve",
        transition_duration_s=0.8,
        color_preset="cool_blue",
        motion="parallax_still",
        motion_intensity=0.04,
        text_animation="slide_up",
        caption_animation="slide_up",
    ),
}


# ── Public API ────────────────────────────────────────────────────────────────

def list_style_packs() -> list[StylePack]:
    """Return all built-in StylePack instances."""
    return list(STYLE_PACKS.values())


def get_style_pack(name: str) -> StylePack:
    """Return the StylePack for *name*.

    Raises
    ------
    ValueError
        If *name* is not a key in STYLE_PACKS.
    """
    if name not in STYLE_PACKS:
        raise ValueError(
            f"Unknown style pack {name!r}. "
            f"Valid: {sorted(STYLE_PACKS.keys())}"
        )
    return STYLE_PACKS[name]
