"""
kaizer.pipeline.phase4.vertical_packs
======================================
Per-niche narrative + scoring tuning.

Indian news (Telugu, Hindi, Tamil) is the v1 launch niche; Phase 4
expands to: podcast, gaming, finance, fitness, beauty, parenting. Each
pack is a drop-in override containing:

  prompt_overrides   : {narrative_prompt: str, taxonomy: list[str], weights: dict}
  scoring_weights    : per-mode hook / importance / completion adjustments
  hook_opener_words  : list[str] scored in narrative.py hook heuristics
  completion_hints   : list[regex] considered during detect_completion
  cta_templates      : {soft_follow, related_video, next_part, url_overlay}
  default_aspect     : '9:16' | '1:1' | '4:5' | '16:9'
  platform_weights   : {youtube_short, instagram_reel, tiktok}

Packs live in resources/vertical_packs/{niche}.yaml (Phase 4 adds the
YAML loader). For v1, in-code defaults suffice.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("kaizer.pipeline.phase4.vertical_packs")


@dataclass
class VerticalPack:
    niche: str
    prompt_overrides: dict = field(default_factory=dict)
    scoring_weights: dict = field(default_factory=dict)
    hook_opener_words: list[str] = field(default_factory=list)
    completion_hints: list[str] = field(default_factory=list)
    cta_templates: dict = field(default_factory=dict)
    default_aspect: str = "9:16"
    platform_weights: dict = field(default_factory=dict)


# v1 ships the 'news' pack inline for launch content. Other packs are stubs.
_BUILTIN_PACKS: dict[str, VerticalPack] = {
    "news": VerticalPack(
        niche="news",
        hook_opener_words=[
            "breaking", "just in", "exclusive", "shocking", "update",
            "confirmed", "revealed", "huge", "developing",
        ],
        cta_templates={
            "soft_follow": "Follow for breaking updates",
            "related_video": "Full story on my channel →",
            "next_part": "Part 2 continues the story ↗",
            "url_overlay": "Full report ↗",
        },
        default_aspect="9:16",
        platform_weights={"youtube_short": 1.0, "instagram_reel": 1.0, "tiktok": 0.8},
    ),
}


def load_pack(niche: str) -> Optional[VerticalPack]:
    """Return the VerticalPack for `niche` or None if unknown.

    Phase 4 will also load from resources/vertical_packs/<niche>.yaml.
    """
    pack = _BUILTIN_PACKS.get(niche.lower())
    if pack is None:
        logger.info("vertical_packs.load_pack: no pack for niche=%r (returning None)", niche)
    return pack


def list_available() -> list[str]:
    return sorted(_BUILTIN_PACKS.keys())


def apply_pack_to_narrative(pack: VerticalPack, candidate_meta: dict) -> dict:
    """Transform a ClipCandidate.meta dict using the pack's weights/hints.

    Phase 4 enhancements: inject pack.prompt_overrides into the Gemini
    narrative call, score opener-word presence, etc. For v1 stub: returns
    the meta unchanged plus {'vertical_pack': pack.niche}.
    """
    return {**candidate_meta, "vertical_pack": pack.niche}
