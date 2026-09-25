# Ported from kaizer-platform@d5fd482 server/pipeline_core/podcast/__init__.py.
# Changes from upstream: none (verbatim copy; origin header added).
"""Podcast mode (Phase 1, single-cam MVP).

Pure planning functions (cutlist / punch-in / promo) + an ffmpeg renderer,
consuming the word-level transcript already produced by the STT layer
(``pipeline_v2.models.Word`` / ``WordLevelTranscript``).

The renderer is imported lazily (it pulls ffmpeg/edl_builder) so the pure
planners stay import-light and test-friendly.
"""

from __future__ import annotations

from pipeline_core.podcast.cutlist import (
    CutlistConfig,
    CutlistResult,
    WordT,
    build_keep_ranges,
)
from pipeline_core.podcast.punchin import (
    PunchIn,
    PunchInConfig,
    build_punch_in_plan,
)
from pipeline_core.podcast.promo import (
    PromoConfig,
    PromoPlan,
    PromoSegment,
    build_promo_plan,
)

__all__ = [
    "CutlistConfig",
    "CutlistResult",
    "WordT",
    "build_keep_ranges",
    "PunchIn",
    "PunchInConfig",
    "build_punch_in_plan",
    "PromoConfig",
    "PromoPlan",
    "PromoSegment",
    "build_promo_plan",
]
