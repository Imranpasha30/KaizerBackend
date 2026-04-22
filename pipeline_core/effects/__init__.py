"""
kaizer.pipeline.effects
=======================
Beta-mode rendering pipeline: transitions, text animations, color grading,
camera motion, and style packs.

Re-exports all public names from the sub-modules.
"""

from .transitions import (  # noqa: F401
    TransitionSpec,
    SUPPORTED_TRANSITIONS,
    apply_transition,
)
from .text_animations import (  # noqa: F401
    TextAnimationSpec,
    SUPPORTED_ANIMATIONS,
    render_animation_frames,
    apply_text_animation,
)
from .color_grade import (  # noqa: F401
    COLOR_PRESETS,
    apply_color_grade,
)
from .motion import (  # noqa: F401
    MotionSpec,
    SUPPORTED_MOTIONS,
    apply_motion,
)
from .style_packs import (  # noqa: F401
    StylePack,
    STYLE_PACKS,
    list_style_packs,
    get_style_pack,
)

__all__ = [
    # transitions
    "TransitionSpec",
    "SUPPORTED_TRANSITIONS",
    "apply_transition",
    # text_animations
    "TextAnimationSpec",
    "SUPPORTED_ANIMATIONS",
    "render_animation_frames",
    "apply_text_animation",
    # color_grade
    "COLOR_PRESETS",
    "apply_color_grade",
    # motion
    "MotionSpec",
    "SUPPORTED_MOTIONS",
    "apply_motion",
    # style_packs
    "StylePack",
    "STYLE_PACKS",
    "list_style_packs",
    "get_style_pack",
]
