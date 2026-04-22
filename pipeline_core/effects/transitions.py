"""
kaizer.pipeline.effects.transitions
=====================================
Clip-to-clip transition renderer using FFmpeg's xfade filter.

Supported transitions:
  fade        → xfade:fade
  slide_left  → xfade:slideleft
  slide_up    → xfade:slideup
  zoom_punch  → xfade:zoomin
  whip_pan    → xfade:smoothleft
  dissolve    → xfade:dissolve

Usage
-----
    from pipeline_core.effects.transitions import TransitionSpec, apply_transition

    spec = TransitionSpec(name='fade', duration_s=0.5, params={})
    out = apply_transition(left, right, transition=spec, output_path='/tmp/out.mp4')
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field

from pipeline_core.pipeline import FFMPEG_BIN, ENCODE_ARGS_SHORT_FORM

logger = logging.getLogger("kaizer.pipeline.editor_pro.transitions")

# ── Public constants ──────────────────────────────────────────────────────────

SUPPORTED_TRANSITIONS: tuple[str, ...] = (
    "fade",
    "slide_left",
    "slide_up",
    "zoom_punch",
    "whip_pan",
    "dissolve",
)

# Map from our name → FFmpeg xfade transition name
_XFADE_MAP: dict[str, str] = {
    "fade":       "fade",
    "slide_left": "slideleft",
    "slide_up":   "slideup",
    "zoom_punch": "zoomin",
    "whip_pan":   "smoothleft",
    "dissolve":   "dissolve",
}


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class TransitionSpec:
    """Specification for a single clip-to-clip transition.

    Attributes
    ----------
    name : str
        One of SUPPORTED_TRANSITIONS.
    duration_s : float
        Duration of the overlap (blend) window in seconds (0.25 – 1.0 typical).
    params : dict
        Transition-specific overrides (direction, intensity, easing).
        Currently reserved for future extension.
    """

    name: str
    duration_s: float = 0.5
    params: dict = field(default_factory=dict)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _probe_duration(video_path: str) -> float:
    """Return duration of *video_path* in seconds. Returns 0.0 on failure."""
    import json
    try:
        from pipeline_core.qa import FFPROBE_BIN as _fp
    except Exception:
        import shutil as _sh
        _fp = _sh.which("ffprobe") or "ffprobe"

    cmd = [
        _fp, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            return float(data.get("format", {}).get("duration", 0.0))
    except Exception as exc:
        logger.warning("transitions: ffprobe duration probe failed for %s: %s", video_path, exc)
    return 0.0


# ── Public API ────────────────────────────────────────────────────────────────

def apply_transition(
    left_video: str,
    right_video: str,
    *,
    transition: TransitionSpec,
    output_path: str,
) -> str:
    """Blend two videos using FFmpeg's xfade filter.

    The transition occupies the tail of *left_video* and the head of
    *right_video*. The overlap offset is computed as:
        offset = duration(left_video) - transition.duration_s

    Parameters
    ----------
    left_video : str
        Absolute path to the first (incoming) clip.
    right_video : str
        Absolute path to the second (outgoing) clip.
    transition : TransitionSpec
        Transition specification.
    output_path : str
        Absolute path for the blended output MP4.

    Returns
    -------
    str
        *output_path* on success.

    Raises
    ------
    ValueError
        If ``transition.name`` is not in SUPPORTED_TRANSITIONS.
    RuntimeError
        If FFmpeg fails (non-zero exit).  Callers should treat this as a
        warning and fall back.
    """
    if transition.name not in SUPPORTED_TRANSITIONS:
        raise ValueError(
            f"Unknown transition {transition.name!r}. "
            f"Valid: {SUPPORTED_TRANSITIONS}"
        )

    # Normalise paths
    left_video  = left_video.replace("\\", "/")
    right_video = right_video.replace("\\", "/")
    output_path = output_path.replace("\\", "/")

    xfade_mode = _XFADE_MAP[transition.name]
    dur = max(0.05, float(transition.duration_s))

    # Probe left duration to compute offset
    left_dur = _probe_duration(left_video)
    if left_dur <= 0.0:
        left_dur = dur + 0.1  # fallback: place transition immediately

    offset = max(0.0, left_dur - dur)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Build filter_complex:
    #   [0:v][1:v]xfade=transition=...:duration=...:offset=...[vout]
    #   [0:a][1:a]acrossfade=d=...[aout]
    filter_complex = (
        f"[0:v][1:v]xfade=transition={xfade_mode}:"
        f"duration={dur:.4f}:offset={offset:.4f}[vout];"
        f"[0:a][1:a]acrossfade=d={dur:.4f}[aout]"
    )

    cmd: list[str] = [
        FFMPEG_BIN, "-y",
        "-i", left_video,
        "-i", right_video,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
    ] + ENCODE_ARGS_SHORT_FORM + [output_path]

    logger.debug("apply_transition(%s): %s", transition.name, " ".join(cmd))

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
        raise RuntimeError(
            f"apply_transition FFmpeg failed (rc={proc.returncode}) "
            f"transition={transition.name!r}:\n{stderr_tail}"
        )

    logger.info(
        "apply_transition: %s → %s [%s, %.2fs]",
        os.path.basename(left_video),
        os.path.basename(right_video),
        transition.name,
        dur,
    )
    return output_path
