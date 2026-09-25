# Ported-support for kaizer-platform@d5fd482 server/pipeline_core/ai_director/
# This module REPLACES their pipeline_core sensor dependencies
# (speaker_separation._extract_wav, qa._measure_loudness, shot_detect.detect_shots,
# effects.transitions._probe_duration, librosa loading) with self-contained
# equivalents built on OUR pipeline_v4 helpers — so the platform director has
# zero librosa / pipeline_core coupling.
from __future__ import annotations

import json
import os
import subprocess
import wave
from typing import Optional

import numpy as np


def resolve_ffmpeg() -> str:
    return os.environ.get("KAIZER_FFMPEG_BIN", "ffmpeg")


def resolve_ffprobe() -> str:
    return os.environ.get("KAIZER_FFPROBE_BIN", "ffprobe")


def probe_duration(path: str, *, timeout: int = 60) -> float:
    """Container duration in seconds; 0.0 on any failure (same contract as
    their effects.transitions._probe_duration)."""
    cmd = [resolve_ffprobe(), "-v", "quiet", "-print_format", "json",
           "-show_format", path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, errors="replace")
        data = json.loads(proc.stdout or "{}")
        return max(0.0, float((data.get("format") or {}).get("duration") or 0.0))
    except Exception:
        return 0.0


def extract_wav(video_path: str, tmp_dir: str, ffmpeg_bin: str | None = None,
                *, sr: int = 16_000, timeout: int = 300) -> str:
    """Extract mono 16-bit PCM wav (replaces their
    speaker_separation._extract_wav). Raises on ffmpeg failure — the caller
    (sensors) wraps every sub-measurement fail-soft."""
    out = os.path.join(tmp_dir, "director_audio.wav")
    cmd = [ffmpeg_bin or resolve_ffmpeg(), "-y", "-v", "error",
           "-i", video_path, "-vn", "-ac", "1", "-ar", str(sr),
           "-c:a", "pcm_s16le", out]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, errors="replace")
    if proc.returncode != 0 or not os.path.isfile(out):
        raise RuntimeError(f"wav extract failed: {(proc.stderr or '')[-200:]}")
    return out


def load_wav_mono(path: str) -> tuple[np.ndarray, int]:
    """Load OUR own 16-bit PCM wav into float32 samples in [-1, 1]
    (replaces librosa.load — same value range, so the platform formula's
    RMS thresholds keep their meaning)."""
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return y, sr


def measure_integrated_lufs(path: str) -> Optional[float]:
    """Integrated LUFS via OUR pass-1 loudnorm measurement
    (pipeline_v4.audio_conform.measure_loudness → input_i). None on failure
    (replaces their qa._measure_loudness)."""
    try:
        from pipeline_v4.audio_conform import measure_loudness
        data = measure_loudness(path)
        if isinstance(data, dict) and data.get("input_i") is not None:
            return float(data["input_i"])
    except Exception:
        pass
    return None


def detect_scene_cuts(path: str, *, threshold: float = 0.4) -> list[float]:
    """Scene-change timestamps — shim to OUR downscaled scdet sensor
    (pipeline_v4.director.sense_scene_cuts). [] on failure (replaces their
    shot_detect.detect_shots)."""
    try:
        from pipeline_v4.director import sense_scene_cuts
        return sense_scene_cuts(path, threshold=threshold)
    except Exception:
        return []
