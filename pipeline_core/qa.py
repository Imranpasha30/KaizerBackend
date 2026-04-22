"""
kaizer.pipeline.qa
==================
FFprobe + FFmpeg-based OUTPUT gate for the Kaizer News video pipeline.

Run AFTER each compose_* function writes its output file to verify the
rendered clip meets platform specifications before it is imported into the DB
and surfaced to the user.

Usage
-----
    from pipeline_core.qa import validate_output, QAResult

    result = validate_output(
        "/path/to/output.mp4",
        platform="youtube_short",
        expected_duration_s=45.0,
    )
    if not result.ok:
        raise PipelineQAError(result.errors)

QAResult fields
---------------
  ok           : bool        — False means at least one hard error.
  errors       : list[str]   — Hard failures (wrong pixel fmt, extreme loudness …).
  warnings     : list[str]   — Soft issues (logged, clip still accepted).
  measurements : dict        — All measured values:
                               {
                                 "duration_s": float,
                                 "width": int, "height": int,
                                 "vbitrate_kbps": float,
                                 "abitrate_kbps": float,
                                 "lufs": float | None,
                                 "peak_db": float | None,
                                 "black_frame_pct": float | None,
                                 "color_space": str | None,
                                 "pix_fmt": str | None,
                                 "fps": float,
                               }

Platform spec matrix (PLATFORM_SPECS constant)
----------------------------------------------
  Key              max_dur  min_dur  aspect   min_vbr   max_vbr   lufs_target  min_res
  youtube_short    180s     3s       9:16     4 Mbps    20 Mbps   -14 ±1 LUFS  1080×1920
  instagram_reel   180s     3s       9:16     5 Mbps    10 Mbps   -14 ±1 LUFS  1080×1920
  tiktok           180s     3s       9:16     2 Mbps    10 Mbps   -14 ±1 LUFS  1080×1920
  youtube_long     None     15s      16:9     4 Mbps    50 Mbps   -14 ±1 LUFS  ≥1280×720

Checks
------
  1.  Duration within platform min/max.
  2.  If expected_duration_s is given, actual must be within ±0.5 s.
  3.  Aspect ratio (width/height ratio) matches platform.
  4.  Video bitrate: warning if outside range, error if > 2× max.
  5.  Audio LUFS within -14 ±1 (measured with loudnorm print_format=json).
  6.  True peak ≤ -1.0 dBTP (error if > -0.5 dBTP).
  7.  Black-frame detection: warn if > 5%, error if > 15%.
  8.  Color space must be bt709 (warning if not).
  9.  Pixel format must be yuv420p (hard error if not).
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("kaizer.pipeline.qa")

# ── ffprobe / ffmpeg binary resolution ───────────────────────────────────────

def _find_binary(name: str) -> str:
    """Find *name* binary: PATH first, then pipeline.py's FFMPEG_BIN sibling,
    then common Unix/Linux/Railway prefix directories."""
    import shutil as _sh

    p = _sh.which(name)
    if p:
        return p

    try:
        from pipeline_core.pipeline import FFMPEG_BIN as _ffmpeg  # type: ignore
        sibling = os.path.join(os.path.dirname(os.path.abspath(_ffmpeg)), name)
        if os.path.isfile(sibling) and os.access(sibling, os.X_OK):
            return sibling
        sibling_exe = sibling + ".exe"
        if os.path.isfile(sibling_exe) and os.access(sibling_exe, os.X_OK):
            return sibling_exe
    except Exception:
        pass

    for prefix in [
        "/usr/bin",
        "/usr/local/bin",
        "/nix/var/nix/profiles/default/bin",
        "/run/current-system/sw/bin",
    ]:
        candidate = os.path.join(prefix, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    return name  # fallback — will fail at call time with a clear error


FFPROBE_BIN: str = _find_binary("ffprobe")
FFMPEG_BIN: str = _find_binary("ffmpeg")


# ── Platform spec matrix ──────────────────────────────────────────────────────

# Aspect ratios stored as (width_ratio, height_ratio) for exact comparison.
# Bitrates stored in Kbps (1 Mbps = 1000 Kbps).
PLATFORM_SPECS: dict[str, dict] = {
    "youtube_short": {
        "label":          "YouTube Short",
        "max_dur_s":      180.0,
        "min_dur_s":      3.0,
        "aspect":         (9, 16),      # 9:16 portrait
        "min_vbitrate_kbps": 4_000,
        "max_vbitrate_kbps": 20_000,
        "target_lufs":    -14.0,
        "lufs_tolerance": 1.0,
        "min_width":      1080,
        "min_height":     1920,
        "max_width":      1080,
        "max_height":     1920,
    },
    "instagram_reel": {
        "label":          "Instagram Reel",
        "max_dur_s":      180.0,
        "min_dur_s":      3.0,
        "aspect":         (9, 16),
        "min_vbitrate_kbps": 5_000,
        "max_vbitrate_kbps": 10_000,
        "target_lufs":    -14.0,
        "lufs_tolerance": 1.0,
        "min_width":      1080,
        "min_height":     1920,
        "max_width":      1080,
        "max_height":     1920,
    },
    "tiktok": {
        "label":          "TikTok",
        "max_dur_s":      180.0,
        "min_dur_s":      3.0,
        "aspect":         (9, 16),
        "min_vbitrate_kbps": 2_000,
        "max_vbitrate_kbps": 10_000,
        "target_lufs":    -14.0,
        "lufs_tolerance": 1.0,
        "min_width":      1080,
        "min_height":     1920,
        "max_width":      1080,
        "max_height":     1920,
    },
    "youtube_long": {
        "label":          "YouTube Long",
        "max_dur_s":      None,         # no upper limit
        "min_dur_s":      15.0,
        "aspect":         (16, 9),      # 16:9 landscape
        "min_vbitrate_kbps": 4_000,
        "max_vbitrate_kbps": 50_000,
        "target_lufs":    -14.0,
        "lufs_tolerance": 1.0,
        "min_width":      1280,
        "min_height":     720,
        "max_width":      None,         # any width
        "max_height":     None,
    },
}

# Aspect ratio tolerance: allow ±1% deviation from the ideal ratio.
_ASPECT_TOLERANCE = 0.01

# Black frame thresholds (fraction of total frames)
_BLACKFRAME_WARN_PCT = 5.0
_BLACKFRAME_ERROR_PCT = 15.0

# True peak thresholds (dBTP)
_PEAK_ERROR_THRESHOLD = -0.5   # error if true peak exceeds this
_PEAK_WARN_THRESHOLD  = -1.0   # warn if true peak is between -1.0 and -0.5


# ── Public dataclass ──────────────────────────────────────────────────────────

@dataclass
class QAResult:
    """Result of a validate_output() call.

    Attributes
    ----------
    ok : bool
        True only when there are zero hard errors.
    errors : list[str]
        Hard failures — the clip should NOT be published as-is.
    warnings : list[str]
        Soft issues — logged but the clip can still be imported.
    measurements : dict
        Raw measured values (see module docstring for keys).
    """

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    measurements: dict = field(default_factory=dict)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _run_ffprobe(path: str) -> dict:
    """Run ffprobe on *path* and return the parsed JSON."""
    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd,
            output=result.stdout,
            stderr=result.stderr.strip(),
        )
    return json.loads(result.stdout)


def _parse_fps(r_frame_rate: str) -> float:
    """Parse '30000/1001' → 29.97."""
    parts = r_frame_rate.split("/")
    try:
        if len(parts) == 2:
            num, den = float(parts[0]), float(parts[1])
            return num / den if den != 0.0 else 0.0
        return float(parts[0])
    except (ValueError, ZeroDivisionError):
        return 0.0


def _measure_loudness(path: str) -> tuple[float | None, float | None]:
    """Measure integrated loudness (LUFS) and true peak (dBTP) via FFmpeg.

    Uses the loudnorm filter in analysis mode (print_format=json, null muxer).
    The filter writes a JSON block to stderr.

    Returns
    -------
    (integrated_lufs, true_peak_db)
        Both are None if measurement fails (e.g. no audio stream).
    """
    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-i", path,
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
        "-f", "null",
        "-",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # loudnorm requires a full decode pass
        )
    except subprocess.TimeoutExpired:
        logger.warning("Loudness measurement timed out for %s", path)
        return None, None

    stderr = result.stderr

    # loudnorm emits the JSON block to stderr between { and } lines.
    # Extract the last JSON object in stderr (the summary block).
    json_match = re.search(r"\{[^{}]*\}", stderr, re.DOTALL)
    if not json_match:
        logger.warning(
            "Could not find loudnorm JSON output in ffmpeg stderr for %s", path
        )
        return None, None

    try:
        ld = json.loads(json_match.group())
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse loudnorm JSON for %s: %s", path, exc)
        return None, None

    # Keys emitted by loudnorm print_format=json (analysis pass):
    #   input_i, input_tp, input_lra, input_thresh, …
    try:
        lufs = float(ld.get("input_i") or ld.get("output_i") or 0.0)
    except (TypeError, ValueError):
        lufs = None

    try:
        peak = float(ld.get("input_tp") or ld.get("output_tp") or 0.0)
    except (TypeError, ValueError):
        peak = None

    return lufs, peak


def _count_black_frames(path: str, total_frames: int) -> float | None:
    """Return the percentage of frames detected as black (0–100).

    Uses FFmpeg's blackframe filter.
    Returns None if detection fails.
    """
    if total_frames <= 0:
        return None

    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-i", path,
        "-vf", "blackframe=amount=98:thresh=32",
        "-f", "null",
        "-",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Black-frame detection timed out for %s", path)
        return None

    # blackframe logs one line per black frame to stderr:
    #   [blackframe @ …] frame:N pblack:M pts:P t:T type:I last_keyframe:K
    stderr = result.stderr
    black_count = len(re.findall(r"\[blackframe ", stderr))
    return (black_count / total_frames) * 100.0


# ── Public API ────────────────────────────────────────────────────────────────

def validate_output(
    path: str,
    *,
    platform: str,
    expected_duration_s: Optional[float] = None,
) -> QAResult:
    """Validate a rendered output video against platform specifications.

    Parameters
    ----------
    path : str
        Absolute path to the rendered output file.
    platform : str
        One of: 'youtube_short', 'instagram_reel', 'tiktok', 'youtube_long'.
    expected_duration_s : float | None
        If provided, the actual clip duration must be within ±0.5 s of this
        value.  Useful for asserting the trim worked correctly.

    Returns
    -------
    QAResult
        .ok is True only when there are zero hard errors.
    """
    errors: list[str] = []
    warnings: list[str] = []
    measurements: dict = {}

    # ── Resolve platform spec ─────────────────────────────────────────────────
    if platform not in PLATFORM_SPECS:
        raise ValueError(
            f"Unknown platform: {platform!r}. "
            f"Valid platforms: {sorted(PLATFORM_SPECS.keys())}."
        )
    spec = PLATFORM_SPECS[platform]

    # ── File existence ────────────────────────────────────────────────────────
    if not os.path.exists(path):
        errors.append(f"Output file not found: {path!r}")
        return QAResult(ok=False, errors=errors, warnings=warnings, measurements=measurements)

    if os.path.getsize(path) == 0:
        errors.append(f"Output file is empty (0 bytes): {path!r}")
        return QAResult(ok=False, errors=errors, warnings=warnings, measurements=measurements)

    # ── ffprobe pass ──────────────────────────────────────────────────────────
    try:
        probe = _run_ffprobe(path)
    except Exception as exc:
        errors.append(f"ffprobe failed on output file: {exc}")
        return QAResult(ok=False, errors=errors, warnings=warnings, measurements=measurements)

    fmt = probe.get("format", {})
    streams = probe.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    vs = video_streams[0] if video_streams else {}
    as_ = audio_streams[0] if audio_streams else {}

    # Duration
    duration_s: float = 0.0
    try:
        duration_s = float(fmt.get("duration") or vs.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration_s = 0.0

    # Resolution
    width: int = int(vs.get("width") or 0)
    height: int = int(vs.get("height") or 0)

    # Framecount (for black-frame pct calculation)
    fps: float = _parse_fps(vs.get("r_frame_rate") or vs.get("avg_frame_rate") or "0/1")
    total_frames: int = int(duration_s * fps) if (duration_s > 0 and fps > 0) else 0

    # Bitrates
    try:
        total_bitrate_kbps = float(fmt.get("bit_rate") or 0.0) / 1000.0
    except (TypeError, ValueError):
        total_bitrate_kbps = 0.0

    # Per-stream bitrates from tags / side_data (best-effort)
    try:
        vbitrate_kbps = float(vs.get("bit_rate") or 0.0) / 1000.0
    except (TypeError, ValueError):
        vbitrate_kbps = 0.0

    try:
        abitrate_kbps = float(as_.get("bit_rate") or 0.0) / 1000.0
    except (TypeError, ValueError):
        abitrate_kbps = 0.0

    # When individual stream bitrates are not available, fall back to container
    # total (common for mp4/mov containers that store bitrate only in moov atom).
    if vbitrate_kbps == 0.0 and total_bitrate_kbps > 0.0:
        vbitrate_kbps = total_bitrate_kbps - abitrate_kbps

    # Color space & pixel format
    color_space: str | None = vs.get("color_space") or vs.get("colorspace") or None
    pix_fmt: str | None = vs.get("pix_fmt") or None

    # ── Measurements snapshot (before derived checks) ─────────────────────────
    measurements = {
        "duration_s": duration_s,
        "width": width,
        "height": height,
        "fps": fps,
        "vbitrate_kbps": vbitrate_kbps,
        "abitrate_kbps": abitrate_kbps,
        "lufs": None,
        "peak_db": None,
        "black_frame_pct": None,
        "color_space": color_space,
        "pix_fmt": pix_fmt,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # PLATFORM CHECKS
    # ══════════════════════════════════════════════════════════════════════════

    # ── Check 1: Duration within platform min/max ─────────────────────────────
    min_dur: float = spec.get("min_dur_s", 0.0)
    max_dur: Optional[float] = spec.get("max_dur_s")

    if duration_s < min_dur:
        errors.append(
            f"Output duration {duration_s:.2f}s is below platform minimum {min_dur:.0f}s "
            f"for {platform!r}."
        )
    if max_dur is not None and duration_s > max_dur:
        errors.append(
            f"Output duration {duration_s:.2f}s exceeds platform maximum {max_dur:.0f}s "
            f"for {platform!r}."
        )

    # ── Check 2: Expected duration ────────────────────────────────────────────
    if expected_duration_s is not None:
        delta = abs(duration_s - expected_duration_s)
        if delta > 0.5:
            errors.append(
                f"Actual duration {duration_s:.2f}s deviates from expected "
                f"{expected_duration_s:.2f}s by {delta:.2f}s (tolerance: ±0.5s)."
            )

    # ── Check 3: Aspect ratio ─────────────────────────────────────────────────
    aspect = spec.get("aspect")
    if aspect and width > 0 and height > 0:
        expected_ratio = aspect[0] / aspect[1]
        actual_ratio = width / height
        if abs(actual_ratio - expected_ratio) / expected_ratio > _ASPECT_TOLERANCE:
            errors.append(
                f"Aspect ratio mismatch for {platform!r}: "
                f"expected {aspect[0]}:{aspect[1]} ({expected_ratio:.4f}), "
                f"got {width}×{height} ({actual_ratio:.4f})."
            )

    # ── Check 4: Video bitrate ────────────────────────────────────────────────
    min_vbr: float = spec.get("min_vbitrate_kbps", 0.0)
    max_vbr: Optional[float] = spec.get("max_vbitrate_kbps")

    if vbitrate_kbps > 0.0:
        if vbitrate_kbps < min_vbr:
            warnings.append(
                f"Video bitrate {vbitrate_kbps:.0f} kbps is below platform minimum "
                f"{min_vbr:.0f} kbps for {platform!r}."
            )
        if max_vbr is not None:
            if vbitrate_kbps > 2.0 * max_vbr:
                errors.append(
                    f"Video bitrate {vbitrate_kbps:.0f} kbps exceeds 2× platform maximum "
                    f"({max_vbr:.0f} kbps) for {platform!r}. "
                    "The file is dangerously over-bitrated."
                )
            elif vbitrate_kbps > max_vbr:
                warnings.append(
                    f"Video bitrate {vbitrate_kbps:.0f} kbps exceeds platform maximum "
                    f"{max_vbr:.0f} kbps for {platform!r}."
                )
    else:
        warnings.append(
            "Could not determine video bitrate from ffprobe — bitrate checks skipped."
        )

    # ── Check 9: Pixel format (hard error) ───────────────────────────────────
    if pix_fmt and pix_fmt != "yuv420p":
        errors.append(
            f"Pixel format is {pix_fmt!r}; must be 'yuv420p' for broad platform "
            "compatibility. Re-encode with -pix_fmt yuv420p."
        )
    elif not pix_fmt:
        warnings.append(
            "Could not determine pixel format from ffprobe — pix_fmt check skipped."
        )

    # ── Check 8: Color space (warning) ───────────────────────────────────────
    if color_space and color_space not in ("bt709", "bt709-2"):
        warnings.append(
            f"Color space is {color_space!r}; expected 'bt709'. "
            "Videos with incorrect color space may look washed-out on some platforms."
        )
    elif not color_space:
        warnings.append(
            "Could not determine color space from ffprobe — colorspace check skipped."
        )

    # ── Resolution for youtube_long (≥1280×720) ───────────────────────────────
    min_w: Optional[int] = spec.get("min_width")
    min_h: Optional[int] = spec.get("min_height")
    max_w: Optional[int] = spec.get("max_width")
    max_h: Optional[int] = spec.get("max_height")

    if width > 0 and height > 0:
        if min_w and width < min_w:
            errors.append(
                f"Video width {width}px is below platform minimum {min_w}px for {platform!r}."
            )
        if min_h and height < min_h:
            errors.append(
                f"Video height {height}px is below platform minimum {min_h}px for {platform!r}."
            )
        if max_w and width > max_w:
            errors.append(
                f"Video width {width}px exceeds platform maximum {max_w}px for {platform!r}."
            )
        if max_h and height > max_h:
            errors.append(
                f"Video height {height}px exceeds platform maximum {max_h}px for {platform!r}."
            )

    # ══════════════════════════════════════════════════════════════════════════
    # AUDIO CHECKS (require a full decode pass — only run when audio is present)
    # ══════════════════════════════════════════════════════════════════════════

    if audio_streams:
        # ── Check 5 & 6: Loudness and true peak ──────────────────────────────
        target_lufs: float = spec.get("target_lufs", -14.0)
        lufs_tol: float = spec.get("lufs_tolerance", 1.0)

        lufs, peak_db = _measure_loudness(path)
        measurements["lufs"] = lufs
        measurements["peak_db"] = peak_db

        if lufs is not None:
            lufs_lower = target_lufs - lufs_tol
            lufs_upper = target_lufs + lufs_tol
            if not (lufs_lower <= lufs <= lufs_upper):
                # Treat as a warning (loudnorm should have fixed it; if it
                # failed, we still want to surface the clip rather than block it).
                warnings.append(
                    f"Integrated loudness {lufs:.1f} LUFS is outside target range "
                    f"[{lufs_lower:.0f}, {lufs_upper:.0f}] LUFS for {platform!r}."
                )
        else:
            warnings.append("Integrated loudness measurement failed — LUFS check skipped.")

        if peak_db is not None:
            if peak_db > _PEAK_ERROR_THRESHOLD:
                errors.append(
                    f"True peak {peak_db:.1f} dBTP exceeds -0.5 dBTP limit. "
                    "This will cause clipping on platform re-encodes."
                )
            elif peak_db > _PEAK_WARN_THRESHOLD:
                warnings.append(
                    f"True peak {peak_db:.1f} dBTP is between {_PEAK_WARN_THRESHOLD} and "
                    f"{_PEAK_ERROR_THRESHOLD} dBTP (target: ≤ -1.0 dBTP)."
                )
        else:
            warnings.append("True peak measurement failed — peak check skipped.")
    else:
        warnings.append(
            "No audio stream in output file — loudness and true peak checks skipped."
        )

    # ── Check 7: Black-frame detection ───────────────────────────────────────
    if total_frames > 0:
        black_pct = _count_black_frames(path, total_frames)
        measurements["black_frame_pct"] = black_pct
        if black_pct is not None:
            if black_pct > _BLACKFRAME_ERROR_PCT:
                errors.append(
                    f"Black-frame percentage {black_pct:.1f}% exceeds {_BLACKFRAME_ERROR_PCT:.0f}% "
                    "threshold. The render may have failed to composite correctly."
                )
            elif black_pct > _BLACKFRAME_WARN_PCT:
                warnings.append(
                    f"Black-frame percentage {black_pct:.1f}% exceeds {_BLACKFRAME_WARN_PCT:.0f}% "
                    "warning threshold. Check for unintended black sections."
                )
    else:
        warnings.append(
            "Could not determine frame count — black-frame detection skipped."
        )

    ok = len(errors) == 0
    level = logging.INFO if ok else logging.WARNING
    logger.log(
        level,
        "Output QA %s: %s  platform=%s dur=%.2fs res=%dx%d vbr=%.0fkbps "
        "lufs=%s peak=%s black_pct=%s errors=%s warnings=%s",
        "PASSED" if ok else "FAILED",
        path,
        platform,
        duration_s,
        width, height,
        vbitrate_kbps,
        f"{measurements.get('lufs'):.1f}" if measurements.get("lufs") is not None else "N/A",
        f"{measurements.get('peak_db'):.1f}" if measurements.get("peak_db") is not None else "N/A",
        f"{measurements.get('black_frame_pct'):.1f}%" if measurements.get("black_frame_pct") is not None else "N/A",
        errors,
        warnings,
    )

    return QAResult(ok=ok, errors=errors, warnings=warnings, measurements=measurements)
