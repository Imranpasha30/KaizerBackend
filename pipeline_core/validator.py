"""
kaizer.pipeline.validator
=========================
FFprobe-based INPUT gate for the Kaizer News video pipeline.

Usage
-----
    from pipeline_core.validator import validate_input, ValidationResult

    result = validate_input("/path/to/video.mp4")
    if not result.ok:
        # Hard failure — reject the upload
        raise ValueError(result.errors)
    if result.warnings:
        logger.warning("Input warnings: %s", result.warnings)

ValidationResult fields
-----------------------
  ok        : bool          — False means at least one hard error; reject input.
  errors    : list[str]     — Hard failures (file missing, bad codec, corruption …).
  warnings  : list[str]     — Soft issues (missing audio stream, borderline fps …).
  meta      : dict          — Parsed ffprobe output:
                              {
                                "duration_s": float,
                                "width": int, "height": int,
                                "fps": float,
                                "video_codec": str,
                                "audio_codec": str | None,
                                "bitrate_kbps": float,
                                "probe_score": int,
                                "has_video": bool,
                                "has_audio": bool,
                                "container": str,
                              }

Validation rules (hard errors unless stated otherwise)
------------------------------------------------------
  1. File exists and is readable on disk.
  2. ffprobe succeeds and produces parseable JSON output.
  3. probe_score >= 50 (corruption gate).
  4. At least one video stream present.
  5. Video codec in ALLOWED_VIDEO_CODECS.
  6. 1.0 s < duration <= max_duration_s.
  7. Resolution <= max_resolution (both width and height).
  8. 1 <= fps <= 120.
  9. At least one audio stream (WARNING only — some news clips are silent).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger("kaizer.pipeline.validator")

# ── Allowed video codecs (lowercase, as reported by ffprobe codec_name) ──────
ALLOWED_VIDEO_CODECS: frozenset[str] = frozenset({
    "h264",
    "hevc",
    "av1",
    "vp9",
    "mpeg4",
    "prores",
    "dnxhd",
})


# ── Locate the ffprobe binary, mirroring _find_binary() in pipeline.py ───────

def _find_ffprobe() -> str:
    """Find the ffprobe binary.

    Search order:
      1. PATH (via shutil.which).
      2. The same directory as the ffmpeg binary from pipeline.py, so
         side-by-side installs (e.g. a bundled ffmpeg folder) are covered.
      3. Common Unix/Linux/Railway prefix directories.
      4. Fallback to the bare name "ffprobe" — will produce a clear error
         at call time if not found.
    """
    import shutil as _sh

    # 1. PATH first
    p = _sh.which("ffprobe")
    if p:
        return p

    # 2. Try the directory that pipeline.py resolved for ffmpeg.
    #    Import lazily to avoid circular imports and startup side-effects.
    try:
        from pipeline_core.pipeline import FFMPEG_BIN as _ffmpeg  # type: ignore
        sibling = os.path.join(os.path.dirname(os.path.abspath(_ffmpeg)), "ffprobe")
        if os.path.isfile(sibling) and os.access(sibling, os.X_OK):
            return sibling
        # Windows: try with .exe extension
        sibling_exe = sibling + ".exe"
        if os.path.isfile(sibling_exe) and os.access(sibling_exe, os.X_OK):
            return sibling_exe
    except Exception:
        pass

    # 3. Common Unix/Linux paths
    for prefix in [
        "/usr/bin",
        "/usr/local/bin",
        "/nix/var/nix/profiles/default/bin",
        "/run/current-system/sw/bin",
    ]:
        candidate = os.path.join(prefix, "ffprobe")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    # 4. Fallback
    return "ffprobe"


FFPROBE_BIN: str = _find_ffprobe()


# ── Public dataclass ──────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Result of a single validate_input() call.

    Attributes
    ----------
    ok : bool
        True if no hard errors were found and the file is safe to process.
    errors : list[str]
        Hard failures.  A non-empty list always sets ok=False.
    warnings : list[str]
        Soft issues that are logged but do not block processing.
    meta : dict
        Parsed ffprobe metadata.  See module docstring for keys.
    """

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _run_ffprobe(path: str) -> dict:
    """Run ffprobe on *path* and return the parsed JSON dict.

    Raises
    ------
    subprocess.CalledProcessError
        If ffprobe exits with a non-zero return code.
    ValueError
        If stdout cannot be parsed as JSON.
    """
    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    logger.debug("Running ffprobe: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        stderr_tail = result.stderr.strip().splitlines()[-5:]
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr="\n".join(stderr_tail),
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ffprobe output is not valid JSON: {exc}") from exc


def _parse_fps(r_frame_rate: str) -> float:
    """Parse ffprobe r_frame_rate string (e.g. '30000/1001') into float."""
    parts = r_frame_rate.split("/")
    try:
        if len(parts) == 2:
            num, den = float(parts[0]), float(parts[1])
            return num / den if den != 0.0 else 0.0
        return float(parts[0])
    except (ValueError, ZeroDivisionError):
        return 0.0


# ── Public API ────────────────────────────────────────────────────────────────

def validate_input(
    path: str,
    *,
    max_duration_s: float = 7200.0,
    max_resolution: tuple[int, int] = (3840, 2160),
) -> ValidationResult:
    """Validate a video file before it enters the Kaizer pipeline.

    Parameters
    ----------
    path : str
        Absolute (or resolvable) path to the input video file.
    max_duration_s : float
        Upper bound on video duration in seconds.  Default: 7200 (2 hours).
    max_resolution : tuple[int, int]
        Maximum (width, height) in pixels.  Both dimensions must be within
        bounds.  Default: (3840, 2160) — 4K UHD.

    Returns
    -------
    ValidationResult
        .ok is True only when there are zero hard errors.

    Notes
    -----
    - Audio-stream absence is a WARNING, not an error, because some news clips
      are intentionally silent.
    - This function never raises; all failure paths are captured into
      ValidationResult.errors.
    """
    errors: list[str] = []
    warnings: list[str] = []
    meta: dict = {}

    # ── Rule 1: File exists and is readable ──────────────────────────────────
    if not os.path.exists(path):
        errors.append(f"File not found: {path!r}")
        return ValidationResult(ok=False, errors=errors, warnings=warnings, meta=meta)

    if not os.path.isfile(path):
        errors.append(f"Path is not a regular file: {path!r}")
        return ValidationResult(ok=False, errors=errors, warnings=warnings, meta=meta)

    if not os.access(path, os.R_OK):
        errors.append(f"File is not readable (permission denied): {path!r}")
        return ValidationResult(ok=False, errors=errors, warnings=warnings, meta=meta)

    # ── Rule 2: ffprobe must succeed ─────────────────────────────────────────
    try:
        probe = _run_ffprobe(path)
    except subprocess.CalledProcessError as exc:
        errors.append(
            f"ffprobe failed (exit {exc.returncode}): {exc.stderr or '(no stderr)'}"
        )
        return ValidationResult(ok=False, errors=errors, warnings=warnings, meta=meta)
    except ValueError as exc:
        errors.append(str(exc))
        return ValidationResult(ok=False, errors=errors, warnings=warnings, meta=meta)
    except FileNotFoundError:
        errors.append(
            f"ffprobe binary not found at {FFPROBE_BIN!r}. "
            "Ensure ffmpeg/ffprobe is installed and on PATH."
        )
        return ValidationResult(ok=False, errors=errors, warnings=warnings, meta=meta)
    except subprocess.TimeoutExpired:
        errors.append("ffprobe timed out after 60 s — file may be corrupt or extremely large.")
        return ValidationResult(ok=False, errors=errors, warnings=warnings, meta=meta)

    fmt = probe.get("format", {})
    streams = probe.get("streams", [])

    # ── Rule 3: probe_score >= 50 ────────────────────────────────────────────
    probe_score = int(fmt.get("probe_score", 0))
    if probe_score < 50:
        errors.append(
            f"File appears corrupt or unrecognised (ffprobe probe_score={probe_score} < 50)."
        )
        # Continue collecting metadata even for corrupt files so callers get context.

    # ── Separate video and audio streams ─────────────────────────────────────
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    # ── Rule 4: at least one video stream ────────────────────────────────────
    has_video = len(video_streams) > 0
    if not has_video:
        errors.append("No video stream found in the file.")

    # ── Rule 9: at least one audio stream (warning only) ─────────────────────
    has_audio = len(audio_streams) > 0
    if not has_audio:
        warnings.append(
            "No audio stream found. The clip will be rendered without audio. "
            "This is acceptable only for intentionally silent news clips."
        )

    # ── Extract video stream metadata ─────────────────────────────────────────
    video_codec: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0

    if video_streams:
        vs = video_streams[0]
        video_codec = (vs.get("codec_name") or "").lower()
        width = int(vs.get("width") or 0)
        height = int(vs.get("height") or 0)
        fps = _parse_fps(vs.get("r_frame_rate") or vs.get("avg_frame_rate") or "0/1")

    # ── Extract audio stream metadata ─────────────────────────────────────────
    audio_codec: str | None = None
    if audio_streams:
        audio_codec = (audio_streams[0].get("codec_name") or "").lower() or None

    # ── Duration (from format section; fallback to video stream) ─────────────
    duration_s: float = 0.0
    try:
        duration_s = float(fmt.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration_s = 0.0

    if duration_s == 0.0 and video_streams:
        try:
            duration_s = float(video_streams[0].get("duration") or 0.0)
        except (TypeError, ValueError):
            duration_s = 0.0

    # ── Bitrate (bits/s → kbps) ───────────────────────────────────────────────
    bitrate_kbps: float = 0.0
    try:
        bitrate_bps = float(fmt.get("bit_rate") or 0.0)
        bitrate_kbps = bitrate_bps / 1000.0
    except (TypeError, ValueError):
        bitrate_kbps = 0.0

    # ── Container name ────────────────────────────────────────────────────────
    container: str = (fmt.get("format_name") or "").split(",")[0].strip()

    # ── Populate meta dict ────────────────────────────────────────────────────
    meta = {
        "duration_s": duration_s,
        "width": width,
        "height": height,
        "fps": fps,
        "video_codec": video_codec,
        "audio_codec": audio_codec,
        "bitrate_kbps": bitrate_kbps,
        "probe_score": probe_score,
        "has_video": has_video,
        "has_audio": has_audio,
        "container": container,
    }

    # ── Rule 5: video codec in allowed set ───────────────────────────────────
    if has_video and video_codec and video_codec not in ALLOWED_VIDEO_CODECS:
        errors.append(
            f"Unsupported video codec {video_codec!r}. "
            f"Allowed: {sorted(ALLOWED_VIDEO_CODECS)}."
        )
    elif has_video and not video_codec:
        warnings.append("Could not determine video codec name from ffprobe output.")

    # ── Rule 6: duration sanity ───────────────────────────────────────────────
    if duration_s <= 1.0:
        errors.append(
            f"Video duration {duration_s:.2f}s is too short (must be > 1.0 s)."
        )
    elif duration_s > max_duration_s:
        errors.append(
            f"Video duration {duration_s:.1f}s exceeds maximum allowed "
            f"{max_duration_s:.0f}s."
        )

    # ── Rule 7: resolution bounds ─────────────────────────────────────────────
    max_w, max_h = max_resolution
    if has_video:
        if width > max_w:
            errors.append(
                f"Video width {width}px exceeds maximum {max_w}px."
            )
        if height > max_h:
            errors.append(
                f"Video height {height}px exceeds maximum {max_h}px."
            )
        if width == 0 or height == 0:
            warnings.append(
                f"Could not determine video resolution from ffprobe output "
                f"(width={width}, height={height})."
            )

    # ── Rule 8: framerate sanity ──────────────────────────────────────────────
    if has_video:
        if fps == 0.0:
            warnings.append("Could not determine framerate from ffprobe output.")
        elif fps < 1.0 or fps > 120.0:
            errors.append(
                f"Framerate {fps:.2f} fps is outside the sane range [1, 120]."
            )

    ok = len(errors) == 0
    if ok:
        logger.info(
            "Input validation passed: %s  [%dx%d, %.2ffps, %.1fs, codec=%s]",
            path, width, height, fps, duration_s, video_codec,
        )
    else:
        logger.warning(
            "Input validation FAILED: %s  errors=%s warnings=%s",
            path, errors, warnings,
        )

    return ValidationResult(ok=ok, errors=errors, warnings=warnings, meta=meta)
