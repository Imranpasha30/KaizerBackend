"""
kaizer.pipeline.live_director.analyzers
=========================================
Per-camera signal analyzers for the Autonomous Live Director.

Each analyzer is an independent asyncio task that pulls from a
``CameraRingBuffer``, computes one or more ``SignalFrame`` fields,
and publishes partial ``SignalFrame`` objects to the ``SignalBus``.

Memory contract
---------------
- No heavyweight ML libraries imported at module level.
- All model loading deferred to the first ``analyze()`` invocation.
- Raw frames downsampled by ``ANALYZER_FRAME_DOWNSAMPLE`` before any
  computation; downsampled arrays are computed, consumed, and discarded
  within a single tick — never stored as instance state.
- ``ANALYZER_FRAME_DOWNSAMPLE`` default 0.333 maps 1080×1920 → ~360×640.
"""
from __future__ import annotations

# ── Config knob ───────────────────────────────────────────────────────────────
ANALYZER_FRAME_DOWNSAMPLE: float = 0.333
"""Scale factor applied to every video frame before analysis.

Default 0.333 maps the native 1080×1920 ring-buffer frame to ~360×640,
keeping per-tick CPU/memory cost bounded.  Lower values save more memory
at the cost of detection accuracy.
"""

# ── Re-exports ─────────────────────────────────────────────────────────────────
from .base import Analyzer, AnalyzerConfig
from .audio import AudioAnalyzer
from .face import FaceAnalyzer
from .motion import MotionAnalyzer
from .scene import SceneAnalyzer
from .reaction import ReactionAnalyzer
from .beat import BeatAnalyzer

__all__ = [
    "ANALYZER_FRAME_DOWNSAMPLE",
    "Analyzer",
    "AnalyzerConfig",
    "AudioAnalyzer",
    "FaceAnalyzer",
    "MotionAnalyzer",
    "SceneAnalyzer",
    "ReactionAnalyzer",
    "BeatAnalyzer",
]
