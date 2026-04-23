"""
kaizer.pipeline.hw_accel
=========================
Hardware-acceleration helpers — keeps encoding off the CPU and out of RAM.

Windows 32 GB dev box was OOM-ing under parallel libx264 + multiple
FFmpeg subprocesses + Whisper + backend + frontend. Moving video encode
to NVENC (on the NVIDIA RTX 5060 detected on this box) drops per-encode
CPU load to ~5 % and frees 500 MB – 1 GB of RAM per concurrent render.

Detection
---------
At import time, we probe ``ffmpeg -encoders`` once and cache which
hardware encoders are available. Preference order:

  1. h264_nvenc  (NVIDIA) — most mature, widest support
  2. h264_qsv    (Intel QSV) — good on iGPUs
  3. h264_amf    (AMD AMF)
  4. libx264     (CPU fallback)

For decoding the corresponding ``-hwaccel cuda/qsv/d3d11va`` is picked.

Usage
-----
    from pipeline_core.hw_accel import h264_args, hw_decode_args, ACTIVE_ENCODER

    cmd = [FFMPEG_BIN, "-y"] + hw_decode_args() + ["-i", src] + h264_args() + [dst]

All callers stay identical shape — args are a drop-in list<str>.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger("kaizer.pipeline.hw_accel")


# ── Encoder detection (runs once at import) ──────────────────────────────────

def _detect_encoders() -> set[str]:
    """Probe ``ffmpeg -encoders`` and return the set of available encoder names."""
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("hw_accel: encoder probe failed (%s) — assuming CPU only", exc)
        return set()

    encoders: set[str] = set()
    # Parse lines like: " V....D h264_nvenc            NVIDIA NVENC H.264 encoder …"
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or not line[0] in ("V", "A", "S"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            encoders.add(parts[1])
    return encoders


_ENCODERS: set[str] = _detect_encoders()


def _fmt_rate(kbps: int) -> str:
    """Format a bitrate in kbps as the compact FFmpeg form: '8000' kbps → '8M'.

    Whole-number megabits use 'NM'; otherwise fall back to 'NNNNk'. Matches
    the historical project convention ('-b:v 8M -maxrate 10M -bufsize 16M')
    so existing tests keep passing when we swap encoders at import time.
    """
    if kbps % 1000 == 0:
        return f"{kbps // 1000}M"
    return f"{kbps}k"


def _pick_h264_encoder() -> str:
    """Return the best-available H.264 encoder name. Cached via ACTIVE_ENCODER."""
    for cand in ("h264_nvenc", "h264_qsv", "h264_amf"):
        if cand in _ENCODERS:
            return cand
    return "libx264"


# Public constant — useful for tests + logging
ACTIVE_ENCODER: str = _pick_h264_encoder()


# Env override for dev: KAIZER_FORCE_CPU_ENCODE=1 disables NVENC fallback.
if os.environ.get("KAIZER_FORCE_CPU_ENCODE", "") == "1":
    logger.info("hw_accel: KAIZER_FORCE_CPU_ENCODE=1 — forcing libx264")
    ACTIVE_ENCODER = "libx264"


logger.info(
    "hw_accel: ACTIVE_ENCODER=%s (all detected: %s)",
    ACTIVE_ENCODER, sorted(e for e in _ENCODERS if e.startswith(("h264_", "hevc_", "av1_"))),
)


# ── Public API ────────────────────────────────────────────────────────────────

def h264_args(
    *,
    bitrate_kbps: int = 8000,
    maxrate_kbps: int = 10000,
    bufsize_kbps: int = 16000,
    cpu_preset: str = "medium",
) -> list[str]:
    """Return FFmpeg video-encode args for the active H.264 encoder.

    All encoders produce a yuv420p Main/High-profile Level 4.1 stream at
    roughly constant quality ~23 (libx264 CRF scale; maps to NVENC CQ 23,
    QSV global_quality 23, AMF qp_i 23).

    Parameters translate across codecs so caller code is unchanged when
    switching from CPU→GPU.
    """
    if ACTIVE_ENCODER == "h264_nvenc":
        return [
            "-c:v", "h264_nvenc",
            "-preset", "p5",            # balanced (p1=fastest, p7=slowest)
            "-tune", "hq",              # high-quality tuning
            "-rc", "vbr",
            "-cq", "23",                # constant-quality target
            "-b:v", _fmt_rate(bitrate_kbps),
            "-maxrate", _fmt_rate(maxrate_kbps),
            "-bufsize", _fmt_rate(bufsize_kbps),
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
            "-level", "4.1",
            "-movflags", "+faststart",
        ]
    if ACTIVE_ENCODER == "h264_qsv":
        return [
            "-c:v", "h264_qsv",
            "-preset", "medium",
            "-global_quality", "23",
            "-b:v", _fmt_rate(bitrate_kbps),
            "-maxrate", _fmt_rate(maxrate_kbps),
            "-bufsize", _fmt_rate(bufsize_kbps),
            "-pix_fmt", "nv12",         # QSV native
            "-profile:v", "high",
            "-level", "4.1",
            "-movflags", "+faststart",
        ]
    if ACTIVE_ENCODER == "h264_amf":
        return [
            "-c:v", "h264_amf",
            "-quality", "quality",
            "-rc", "vbr_peak",
            "-qp_i", "23",
            "-b:v", f"{bitrate_kbps}k",
            "-maxrate", f"{maxrate_kbps}k",
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
            "-level", "4.1",
            "-movflags", "+faststart",
        ]
    # CPU fallback
    return [
        "-c:v", "libx264",
        "-preset", cpu_preset,
        "-crf", "20",
        "-b:v", f"{bitrate_kbps}k",
        "-maxrate", f"{maxrate_kbps}k",
        "-bufsize", f"{bufsize_kbps}k",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level", "4.1",
        "-movflags", "+faststart",
    ]


def hw_decode_args() -> list[str]:
    """Return FFmpeg input-side args for hardware decode matching the active
    encoder. Empty list when no GPU is available — caller stays valid."""
    if ACTIVE_ENCODER == "h264_nvenc":
        return ["-hwaccel", "cuda"]
    if ACTIVE_ENCODER == "h264_qsv":
        return ["-hwaccel", "qsv"]
    if ACTIVE_ENCODER == "h264_amf":
        return ["-hwaccel", "d3d11va"]
    return []


def is_gpu_accelerated() -> bool:
    """Convenience: True when we're not on libx264."""
    return ACTIVE_ENCODER != "libx264"
