"""
kaizer.pipeline.effects.color_grade
=====================================
Cinematic colour-correction presets using FFmpeg's eq + colorbalance + curves
filters.

Supported presets
-----------------
  none            → stream-copy (no re-encode)
  cinematic_warm  → warm-toned, increased contrast + saturation
  cool_blue       → desaturated cool-blue grade
  vintage         → warm, desaturated, lifted gamma
  news_red        → punchy red-warm news look
  vibrant         → highly saturated, boosted contrast

Usage
-----
    from pipeline_core.effects.color_grade import apply_color_grade

    out = apply_color_grade('/tmp/clip.mp4', preset='cinematic_warm',
                            output_path='/tmp/graded.mp4')
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Optional

from pipeline_core.pipeline import FFMPEG_BIN, ENCODE_ARGS_SHORT_FORM

logger = logging.getLogger("kaizer.pipeline.editor_pro.color_grade")

# ── Public constants ──────────────────────────────────────────────────────────

COLOR_PRESETS: dict[str, Optional[dict]] = {
    "none": None,
    "cinematic_warm": {
        "temperature": 0.08,
        "tint": -0.02,
        "saturation": 1.08,
        "contrast": 1.12,
        "vibrance": 0.1,
    },
    "cool_blue": {
        "temperature": -0.10,
        "tint": 0.03,
        "saturation": 0.95,
        "contrast": 1.05,
    },
    "vintage": {
        "temperature": 0.05,
        "tint": 0.02,
        "saturation": 0.75,
        "contrast": 0.92,
        "gamma": 1.05,
    },
    "news_red": {
        "temperature": 0.12,
        "tint": -0.05,
        "saturation": 1.2,
        "contrast": 1.15,
        "channel_boost": {"r": 0.05, "g": 0, "b": -0.03},
    },
    "vibrant": {
        "saturation": 1.3,
        "contrast": 1.1,
        "vibrance": 0.2,
    },
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_filter_chain(params: dict) -> list[str]:
    """Convert a params dict into an ordered list of FFmpeg filter strings.

    Returns a list of individual filter expressions to be joined with commas
    in the -vf value.
    """
    filters: list[str] = []

    # ── eq filter (brightness/contrast/saturation/gamma) ─────────────────────
    eq_parts: list[str] = []

    contrast = params.get("contrast")
    if contrast is not None:
        # FFmpeg eq contrast range: 0.0–2.0 (1.0 = neutral)
        eq_parts.append(f"contrast={float(contrast):.4f}")

    saturation = params.get("saturation")
    if saturation is not None:
        # FFmpeg eq saturation range: 0.0–3.0 (1.0 = neutral)
        eq_parts.append(f"saturation={float(saturation):.4f}")

    gamma = params.get("gamma")
    if gamma is not None:
        # FFmpeg eq gamma: 0.1–10.0 (1.0 = neutral)
        eq_parts.append(f"gamma={float(gamma):.4f}")

    # temperature simulated via eq gamma_r/gamma_b
    temperature = params.get("temperature", 0.0)
    if abs(temperature) > 0.001:
        # Positive temperature: warm (boost red, reduce blue)
        gr = max(0.5, min(2.0, 1.0 + temperature * 0.5))
        gb = max(0.5, min(2.0, 1.0 - temperature * 0.5))
        eq_parts.append(f"gamma_r={gr:.4f}")
        eq_parts.append(f"gamma_b={gb:.4f}")

    if eq_parts:
        filters.append("eq=" + ":".join(eq_parts))

    # ── colorbalance filter (channel boosts + tint + temperature) ────────────
    cb_parts: list[str] = []

    tint = params.get("tint", 0.0)
    if abs(tint) > 0.001:
        # Positive tint → green cast in midtones
        cb_parts.append(f"gm={tint:.4f}")

    channel_boost = params.get("channel_boost", {})
    r_boost = float(channel_boost.get("r", 0.0))
    g_boost = float(channel_boost.get("g", 0.0))
    b_boost = float(channel_boost.get("b", 0.0))

    if abs(r_boost) > 0.001:
        cb_parts.append(f"rm={r_boost:.4f}")
    if abs(g_boost) > 0.001:
        cb_parts.append(f"gm={g_boost:.4f}")
    if abs(b_boost) > 0.001:
        cb_parts.append(f"bm={b_boost:.4f}")

    if cb_parts:
        filters.append("colorbalance=" + ":".join(cb_parts))

    # ── curves filter (vibrance approximation) ────────────────────────────────
    vibrance = params.get("vibrance", 0.0)
    if abs(vibrance) > 0.01:
        # Use curves preset for mild contrast boost as vibrance approximation
        if vibrance > 0:
            filters.append("curves=preset=increase_contrast")
        else:
            filters.append("curves=preset=linear_contrast")

    return filters


# ── Public API ────────────────────────────────────────────────────────────────

def apply_color_grade(
    video_path: str,
    *,
    preset: str,
    output_path: str,
    custom_params: Optional[dict] = None,
) -> str:
    """Apply an FFmpeg colour-correction chain to *video_path*.

    Parameters
    ----------
    video_path : str
        Absolute path to the source video.
    preset : str
        One of the keys in COLOR_PRESETS.
    output_path : str
        Absolute path for the colour-graded output MP4.
    custom_params : dict | None
        If provided, individual keys override the preset's defaults.

    Returns
    -------
    str
        *output_path* on success.

    Raises
    ------
    ValueError
        If *preset* is not in COLOR_PRESETS.
    RuntimeError
        If FFmpeg fails.
    """
    if preset not in COLOR_PRESETS:
        raise ValueError(
            f"Unknown color preset {preset!r}. "
            f"Valid presets: {sorted(COLOR_PRESETS.keys())}"
        )

    video_path  = video_path.replace("\\", "/")
    output_path = output_path.replace("\\", "/")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 'none' preset → stream-copy (fast remux, no re-encode)
    if preset == "none":
        cmd: list[str] = [
            FFMPEG_BIN, "-y",
            "-i", video_path,
            "-c", "copy",
            output_path,
        ]
        logger.debug("apply_color_grade(none): stream-copy → %s", output_path)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
            raise RuntimeError(
                f"apply_color_grade stream-copy failed (rc={proc.returncode}):\n{stderr_tail}"
            )
        logger.info("apply_color_grade: none (copy) → %s", os.path.basename(output_path))
        return output_path

    # Merge preset params with any custom overrides
    base_params: dict = dict(COLOR_PRESETS[preset] or {})
    if custom_params:
        base_params.update(custom_params)

    filter_parts = _build_filter_chain(base_params)
    vf_string = ",".join(filter_parts) if filter_parts else "null"

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", video_path,
        "-vf", vf_string,
    ] + ENCODE_ARGS_SHORT_FORM + [output_path]

    logger.debug(
        "apply_color_grade(%s): vf=%r → %s", preset, vf_string, output_path
    )

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
        raise RuntimeError(
            f"apply_color_grade FFmpeg failed (rc={proc.returncode}) "
            f"preset={preset!r}:\n{stderr_tail}"
        )

    logger.info(
        "apply_color_grade: %s → %s", preset, os.path.basename(output_path)
    )
    return output_path
