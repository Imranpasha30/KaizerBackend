"""Express Mode hardware-encoder detection.

Self-contained — does NOT touch ``pipeline_core/hw_accel.py`` (the main
Kaizer pipeline's detector) so changes here only affect Express Mode.

Detects whether NVENC (NVIDIA's hardware H.264 encoder) is available
on the local ffmpeg build. If yes, Express Mode renders 5-10× faster
than CPU libx264. If no, falls back to libx264 cleanly — no crash,
identical output bytes, just slower.

Detection is cached for the process lifetime — ``ffmpeg -encoders``
takes ~50 ms but we don't want to pay it per cut.
"""
from __future__ import annotations

import os
import subprocess
import threading
from typing import Optional


# Result cache + lock so concurrent first calls don't race the probe.
_PROBE_LOCK = threading.Lock()
_CACHED_ENCODER: Optional[str] = None
_CACHED_PRESET:  Optional[str] = None


def _ffmpeg_bin() -> str:
    return os.environ.get("FFMPEG_BIN", "ffmpeg")


def _probe_encoders() -> str:
    """Run ``ffmpeg -encoders`` and look for ``h264_nvenc``.

    Returns the encoder name to use: ``h264_nvenc`` when NVENC is
    compiled in AND the runtime actually has an NVIDIA driver, else
    ``libx264``. We also do a tiny dry-run encode to confirm the
    driver loads — ``ffmpeg -encoders`` lists nvenc even on machines
    without drivers, which would fail at runtime with a confusing
    error inside the user's job.
    """
    bin_ = _ffmpeg_bin()

    # Step 1: is the encoder compiled in?
    try:
        proc = subprocess.run(
            [bin_, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[express/hw_accel] ffmpeg -encoders failed: {exc} -- using libx264")
        return "libx264"

    if "h264_nvenc" not in (proc.stdout or ""):
        print("[express/hw_accel] no h264_nvenc in build -- using libx264")
        return "libx264"

    # Step 2: dry-run encode to confirm the GPU driver is actually
    # present. ``lavfi=color`` generates 4 frames of a tiny solid color
    # → encode → /dev/null. Total takes <500 ms on a working setup,
    # fails fast on a no-driver box.
    try:
        proc = subprocess.run(
            [bin_, "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
             "-c:v", "h264_nvenc", "-preset", "p1", "-frames:v", "4",
             "-f", "null", "-"],
            capture_output=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[express/hw_accel] nvenc dry-run failed: {exc} -- using libx264")
        return "libx264"

    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")[-200:]
        print(f"[express/hw_accel] nvenc dry-run error -- using libx264. stderr: {err}")
        return "libx264"

    print("[express/hw_accel] h264_nvenc available -- using GPU encoding")
    return "h264_nvenc"


def video_encoder() -> str:
    """Return the active video encoder for Express Mode. Cached after
    first call. Override at startup via ``KAIZER_EXPRESS_ENCODER`` env
    (set to ``libx264`` to force CPU, or ``h264_nvenc`` to force GPU)."""
    global _CACHED_ENCODER
    if _CACHED_ENCODER:
        return _CACHED_ENCODER
    with _PROBE_LOCK:
        if _CACHED_ENCODER:
            return _CACHED_ENCODER
        override = os.environ.get("KAIZER_EXPRESS_ENCODER", "").strip()
        if override in ("libx264", "h264_nvenc"):
            print(f"[express/hw_accel] using env override: {override}")
            _CACHED_ENCODER = override
        else:
            _CACHED_ENCODER = _probe_encoders()
    return _CACHED_ENCODER


def encoder_args() -> list[str]:
    """Return the ``-c:v <enc> -preset <p> -cq <n>`` triplet that goes
    into the ffmpeg command. Picks NVENC-appropriate flags when GPU
    is active (`-cq` quality + `-preset p4`) and libx264 flags otherwise
    (`-crf 20` + `-preset medium`). Output quality is comparable; NVENC
    runs 5-10× faster on a modern NVIDIA GPU."""
    enc = video_encoder()
    if enc == "h264_nvenc":
        return [
            "-c:v",     "h264_nvenc",
            "-preset",  "p4",        # p1 fastest, p7 slowest. p4 is balanced.
            "-tune",    "hq",
            "-rc",      "vbr",
            "-cq",      "23",        # quality target, comparable to crf=20 libx264
            "-b:v",     "0",         # let -cq drive bitrate
            "-pix_fmt", "yuv420p",
        ]
    # CPU fallback — matches the teammate's settings exactly.
    return [
        "-c:v",     "libx264",
        "-preset",  "medium",
        "-crf",     "20",
        "-pix_fmt", "yuv420p",
    ]


def active_encoder_label() -> str:
    """One-word label for the log / UI: ``nvenc`` or ``libx264``."""
    return "nvenc" if video_encoder() == "h264_nvenc" else "libx264"
