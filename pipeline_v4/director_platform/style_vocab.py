# Ported from kaizer-platform@d5fd482 server/pipeline_core/effects/style_packs.py:81-160
# Frozen copy of the platform's 5 style packs as plain dicts. Used for the
# LLM prompt's valid-values list + the decision trail; the ADAPTER maps a
# chosen pack onto OUR V4 vocabulary (see adapter.MOOD_TO_DIRECTIVE) — these
# field values are the platform's semantics, not V4 registry ids.
from __future__ import annotations

PLATFORM_PACKS: dict = {
    "minimal": {
        "label": "Minimal",
        "description": "Clean cuts, no colour grade. Good when the content carries itself.",
        "transition": "fade",
        "color_preset": "none",
        "motion": None,
        "caption_animation": "typewriter",
    },
    "cinematic": {
        "label": "Cinematic",
        "description": "Warm colour grade, gentle Ken Burns, cinematic fade transitions.",
        "transition": "fade",
        "color_preset": "cinematic_warm",
        "motion": "ken_burns_in",
        "caption_animation": "word_pop",
    },
    "news_flash": {
        "label": "News Flash",
        "description": "Urgent red/warm push, whip-pan transitions, typewriter captions.",
        "transition": "whip_pan",
        "color_preset": "news_red",
        "motion": "zoom_focus",
        "caption_animation": "typewriter",
    },
    "vibrant": {
        "label": "Vibrant",
        "description": "High-saturation punchy colour, zoom-punch cuts, word-pop captions.",
        "transition": "zoom_punch",
        "color_preset": "vibrant",
        "motion": "ken_burns_out",
        "caption_animation": "word_pop",
    },
    "calm": {
        "label": "Calm",
        "description": "Cool-blue grade, slow dissolve transitions, sliding captions.",
        "transition": "dissolve",
        "color_preset": "cool_blue",
        "motion": "parallax_still",
        "caption_animation": "slide_up",
    },
}
