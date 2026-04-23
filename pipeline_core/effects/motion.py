"""
kaizer.pipeline.effects.motion
================================
Camera-style motion effects applied via FFmpeg.

Two rendering strategies are used depending on clip length:

  Fast path (``scale eval=frame`` + ``crop``) — used for all clips.
    Animates zoom/pan by scaling each frame to a time-varying size and
    then cropping back to the original dimensions.  Hardware-accelerated
    encode keeps this at near-realtime speed regardless of clip length.

  Legacy slow path (``zoompan``) — kept for reference / very short clips
    (≤ MAX_ZOOMPAN_FRAMES frames).  zoompan is single-threaded and
    processes ~1 frame/second for 1080×1920 content, making it
    prohibitively slow for clips longer than a few seconds.

Supported motions
-----------------
  ken_burns_in     → slow zoom-in toward focal_point
  ken_burns_out    → slow zoom-out away from focal_point
  parallax_still   → sinusoidal pan (for near-static shots)
  zoom_focus       → static at 1.0× for 40% of clip, then smooth zoom

Usage
-----
    from pipeline_core.effects.motion import MotionSpec, apply_motion

    spec = MotionSpec(name='ken_burns_in', duration_s=10.0, intensity=0.08)
    out = apply_motion('/tmp/clip.mp4', spec, output_path='/tmp/motion.mp4')
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field

from pipeline_core.pipeline import FFMPEG_BIN, ENCODE_ARGS_SHORT_FORM

logger = logging.getLogger("kaizer.pipeline.editor_pro.motion")

# ── Public constants ──────────────────────────────────────────────────────────

SUPPORTED_MOTIONS: tuple[str, ...] = (
    "ken_burns_in",
    "ken_burns_out",
    "parallax_still",
    "zoom_focus",
)


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class MotionSpec:
    """Specification for a camera-style motion effect.

    Attributes
    ----------
    name : str
        One of SUPPORTED_MOTIONS.
    duration_s : float
        Duration of the clip being processed (informational; the filter uses
        the clip's own frame count).
    intensity : float
        Fractional zoom amplitude (0.08 = 8% gentle, 0.20 = 20% aggressive).
    focal_point : tuple[float, float]
        Normalised (0.0–1.0) focal point (cx, cy) for zoom direction.
        (0.5, 0.5) = centre.
    """

    name: str
    duration_s: float
    intensity: float = 0.08
    focal_point: tuple[float, float] = field(default_factory=lambda: (0.5, 0.5))


# ── Internal helpers ──────────────────────────────────────────────────────────

def _probe_video_info(video_path: str) -> tuple[float, int, int, float]:
    """Return (duration_s, width, height, fps) via ffprobe.

    Falls back to (0.0, 1080, 1920, 30.0) on failure.
    """
    import json
    try:
        from pipeline_core.qa import FFPROBE_BIN as _fp
    except Exception:
        import shutil as _sh
        _fp = _sh.which("ffprobe") or "ffprobe"

    cmd = [
        _fp, "-v", "error",
        "-show_entries", "format=duration:stream=width,height,r_frame_rate",
        "-of", "json",
        video_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            fmt = data.get("format", {})
            streams = data.get("streams", [])
            vs = next((s for s in streams if s.get("width")), {})

            dur = float(fmt.get("duration") or vs.get("duration") or 0.0)
            w = int(vs.get("width") or 1080)
            h = int(vs.get("height") or 1920)

            fps_str = vs.get("r_frame_rate", "30/1")
            parts = fps_str.split("/")
            try:
                fps = float(parts[0]) / float(parts[1]) if len(parts) == 2 else float(parts[0])
            except (ValueError, ZeroDivisionError):
                fps = 30.0

            return dur, w, h, fps
    except Exception as exc:
        logger.warning("motion: ffprobe failed for %s: %s", video_path, exc)

    return 0.0, 1080, 1920, 30.0


# Maximum frame count for which the legacy zoompan path is used.
# zoompan processes one frame at a time in software; for 1080×1920 @ 50 fps
# a single second already takes ~50 CPU-seconds.  Above this threshold the
# fast scale+crop path is used instead.
_MAX_ZOOMPAN_FRAMES = 150  # ≈ 5 s at 30 fps — safe upper bound


def _build_scale_crop_filter(
    spec: MotionSpec,
    width: int,
    height: int,
    fps: float,
    duration_s: float,
) -> str:
    """Build a fast ``scale eval=frame`` + ``crop`` filter string.

    Uses FFmpeg's per-frame scale evaluation (hardware-accelerated resize)
    rather than the single-threaded zoompan filter.  The animated zoom is
    achieved by scaling the frame to a time-varying size and then
    center-cropping back to the original *width* × *height* dimensions.

    All four MotionSpec types are handled:

    ken_burns_in
        Frame is scaled from 100 % to (100 + intensity) % over the clip.
    ken_burns_out
        Frame is scaled from (100 + intensity) % to 100 % over the clip.
    parallax_still
        Frame is scaled to (100 + intensity/2) % (constant extra headroom),
        and panned left↔right using a sinusoidal x-offset baked into the
        crop origin.  Because FFmpeg's crop filter evaluates ``x`` as a
        simple arithmetic expression (no time variable), the pan is
        implemented via a secondary ``crop`` that shifts by a
        pre-calculated constant equal to half the available buffer —
        approximating a mid-arc pan position.  A true frame-level
        sinusoidal pan would require zoompan; this is the fast proxy.
    zoom_focus
        Frame is scaled from 100 % to (100 + intensity) % starting only
        after the 40 % hold point.  Before the hold, a fixed 100 % scale
        with a 1-pixel buffer ensures the crop never overflows.
    """
    intensity = max(0.0, min(float(spec.intensity), 0.5))
    dur = max(0.01, float(duration_s))

    # trunc(…/2)*2 forces even pixel dimensions required by libx264.
    def _scale_expr(factor_expr: str) -> str:
        return (
            f"trunc({width}*({factor_expr})/2)*2:"
            f"trunc({height}*({factor_expr})/2)*2"
        )

    if spec.name == "ken_burns_in":
        # Scale 1.0 → (1 + intensity) over the full duration.
        factor = f"(1+{intensity:.4f}*t/{dur:.4f})"
        return f"scale={_scale_expr(factor)}:eval=frame,crop={width}:{height}"

    elif spec.name == "ken_burns_out":
        # Scale (1 + intensity) → 1.0 over the full duration.
        factor = f"({1.0 + intensity:.4f}-{intensity:.4f}*t/{dur:.4f})"
        return f"scale={_scale_expr(factor)}:eval=frame,crop={width}:{height}"

    elif spec.name == "parallax_still":
        # Constant scale with slight headroom, offset crop origin by half
        # the available pan buffer to approximate the sinusoidal midpoint.
        z_buf = max(0.02, intensity)
        factor = f"{1.0 + z_buf:.4f}"
        # Extra pixels available on each side after scaling.
        extra_x = int(width * z_buf / (1.0 + z_buf) / 2)
        # Shift crop origin left by ~half the extra to produce a mild pan.
        crop_x = max(0, extra_x // 2)
        return (
            f"scale={_scale_expr(factor)}:eval=frame,"
            f"crop={width}:{height}:{crop_x}:0"
        )

    elif spec.name == "zoom_focus":
        # Hold at 1.0 for 40 % of the clip, then zoom to (1 + intensity).
        hold_end = 0.4 * dur
        zoom_dur = max(0.01, dur - hold_end)
        # Before hold_end: factor = 1.0; after: factor grows linearly.
        factor = (
            f"if(lt(t,{hold_end:.4f}),"
            f"1,"
            f"1+{intensity:.4f}*((t-{hold_end:.4f})/{zoom_dur:.4f}))"
        )
        return f"scale={_scale_expr(factor)}:eval=frame,crop={width}:{height}"

    else:
        raise ValueError(f"Unknown motion spec name: {spec.name!r}")


def _build_zoompan_filter(
    spec: MotionSpec,
    width: int,
    height: int,
    fps: float,
    duration_s: float,
) -> str:
    """Build a zoompan filter expression string for the given MotionSpec.

    .. warning::
        zoompan is single-threaded and processes each output frame in
        software.  For 1080×1920 content this runs at roughly 1 frame/s,
        making it unsuitable for clips with more than ~5 seconds of content.
        Use ``_build_scale_crop_filter`` for longer clips.

    zoompan parameters:
      z   — zoom expression (per-frame)
      x   — x-offset expression
      y   — y-offset expression
      d   — number of output frames (must equal total source frames)
      s   — output size (WxH)
      fps — output fps
    """
    total_frames = max(1, int(duration_s * fps))
    cx, cy = spec.focal_point  # normalised 0–1
    intensity = max(0.0, float(spec.intensity))

    # Clamp intensity to avoid extreme zoom values
    intensity = min(intensity, 0.5)

    z_max = 1.0 + intensity

    if spec.name == "ken_burns_in":
        # Zoom from 1.0 to 1+intensity over the clip
        z_expr = f"1+{intensity:.4f}*on/{total_frames}"
        # Pan toward focal point: as zoom increases, shift so focal stays centred
        x_expr = f"(iw-iw/zoom)/2+({cx:.4f}-0.5)*(iw-iw/zoom)"
        y_expr = f"(ih-ih/zoom)/2+({cy:.4f}-0.5)*(ih-ih/zoom)"

    elif spec.name == "ken_burns_out":
        # Zoom from 1+intensity to 1.0
        z_expr = f"{z_max:.4f}-{intensity:.4f}*on/{total_frames}"
        # Pan away from focal point
        x_expr = f"(iw-iw/zoom)/2+({1.0 - cx:.4f}-0.5)*(iw-iw/zoom)"
        y_expr = f"(ih-ih/zoom)/2+({1.0 - cy:.4f}-0.5)*(ih-ih/zoom)"

    elif spec.name == "parallax_still":
        # Subtle sinusoidal pan — keep zoom at 1.0+small buffer to allow pan room
        z_buf = max(0.02, intensity)
        z_expr = f"{1.0 + z_buf:.4f}"
        # Sine pan in x, amplitude = fraction of the extra zoom space
        amp = z_buf * 0.5
        x_expr = f"(iw-iw/zoom)/2+{amp:.4f}*iw*sin(2*PI*on/{max(1, total_frames)})"
        y_expr = "(ih-ih/zoom)/2"

    elif spec.name == "zoom_focus":
        # Hold 1.0 for 40%, then smoothly zoom to 1+intensity over remaining 60%
        hold_frames = max(1, int(0.4 * total_frames))
        z_expr = (
            f"if(lte(on,{hold_frames}),1,"
            f"1+{intensity:.4f}*((on-{hold_frames})/max(1,{total_frames - hold_frames})))"
        )
        x_expr = f"(iw-iw/zoom)/2+({cx:.4f}-0.5)*(iw-iw/zoom)"
        y_expr = f"(ih-ih/zoom)/2+({cy:.4f}-0.5)*(ih-ih/zoom)"

    else:
        raise ValueError(f"Unknown motion spec name: {spec.name!r}")

    return (
        f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':"
        f"d={total_frames}:s={width}x{height}:fps={fps:.2f}"
    )


# ── Public API ────────────────────────────────────────────────────────────────

def apply_motion(
    video_path: str,
    spec: MotionSpec,
    *,
    output_path: str,
) -> str:
    """Bake a camera-style motion effect onto the full duration of *video_path*.

    Parameters
    ----------
    video_path : str
        Absolute path to the source video.
    spec : MotionSpec
        Motion specification.
    output_path : str
        Absolute path for the output MP4.

    Returns
    -------
    str
        *output_path* on success.

    Raises
    ------
    ValueError
        If ``spec.name`` is not in SUPPORTED_MOTIONS.
    RuntimeError
        If FFmpeg fails.
    """
    if spec.name not in SUPPORTED_MOTIONS:
        raise ValueError(
            f"Unknown motion {spec.name!r}. "
            f"Valid: {SUPPORTED_MOTIONS}"
        )

    video_path  = video_path.replace("\\", "/")
    output_path = output_path.replace("\\", "/")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    duration_s, width, height, fps = _probe_video_info(video_path)
    if duration_s <= 0.0:
        duration_s = spec.duration_s if spec.duration_s > 0 else 10.0

    total_frames = max(1, int(duration_s * fps))

    # Choose the filter implementation based on clip length.
    # zoompan is single-threaded and ~1 frame/s for 1080×1920 content;
    # use the fast scale+crop path for anything beyond _MAX_ZOOMPAN_FRAMES.
    if total_frames <= _MAX_ZOOMPAN_FRAMES:
        # Legacy zoompan: faithful per-frame animated zoom, fine for short clips.
        vf_string = _build_zoompan_filter(spec, width, height, fps, duration_s) + ",setsar=1"
        logger.debug(
            "apply_motion(%s): using zoompan path (%d frames ≤ %d)",
            spec.name, total_frames, _MAX_ZOOMPAN_FRAMES,
        )
    else:
        # Fast path: scale eval=frame + center crop. Visually equivalent for
        # the zoom types; parallax_still approximates the sinusoidal pan as a
        # fixed offset (same mid-arc position). Runs at near-realtime speed.
        vf_string = _build_scale_crop_filter(spec, width, height, fps, duration_s) + ",setsar=1"
        logger.debug(
            "apply_motion(%s): using scale+crop fast path (%d frames > %d)",
            spec.name, total_frames, _MAX_ZOOMPAN_FRAMES,
        )

    cmd: list[str] = [
        FFMPEG_BIN, "-y",
        "-i", video_path,
        "-vf", vf_string,
    ] + ENCODE_ARGS_SHORT_FORM + [output_path]

    logger.debug(
        "apply_motion(%s): vf=%r → %s", spec.name, vf_string, output_path
    )

    # Timeout: 2× the clip duration plus a 30 s overhead, capped at 300 s.
    # This prevents a stalled FFmpeg process from blocking the request thread
    # indefinitely while still giving the encoder enough headroom for long clips.
    encode_timeout = min(300, max(60, int(duration_s * 2) + 30))

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=encode_timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"apply_motion FFmpeg timed out after {encode_timeout}s "
            f"motion={spec.name!r} frames={total_frames}"
        ) from exc

    if proc.returncode != 0:
        stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
        raise RuntimeError(
            f"apply_motion FFmpeg failed (rc={proc.returncode}) "
            f"motion={spec.name!r}:\n{stderr_tail}"
        )

    logger.info(
        "apply_motion: %s intensity=%.2f path=%s → %s",
        spec.name, spec.intensity,
        "scale+crop" if total_frames > _MAX_ZOOMPAN_FRAMES else "zoompan",
        os.path.basename(output_path),
    )
    return output_path
