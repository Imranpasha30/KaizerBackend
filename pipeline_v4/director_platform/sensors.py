# Ported from kaizer-platform@d5fd482 server/pipeline_core/ai_director/sensors.py
# Changes from upstream: imports rewired to local deps.py (no pipeline_core, no
# librosa — audio loads via stdlib wave over our own 16k mono PCM extract).
# Signal semantics, dataclass shape and the never-raise contract are unchanged.
"""Layer 1 of the platform AI Director ("sensors"): real signal extraction
from a story's source video + transcript. Measures only — no judgement."""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

from pipeline_v4.director_platform.deps import (
    detect_scene_cuts, extract_wav, load_wav_mono, probe_duration,
    resolve_ffmpeg,
)

logger = logging.getLogger("kaizer.pipeline_v4.director_platform.sensors")

_TARGET_SR: int = 16_000
_RMS_FRAME_MS: float = 50.0
_SIGNALSTATS_SAMPLE_FPS: float = 1.0

_YAVG_RE = re.compile(r"lavfi\.signalstats\.YAVG=([\-0-9.]+)")
_SATAVG_RE = re.compile(r"lavfi\.signalstats\.SATAVG=([\-0-9.]+)")

# A word-timing entry can be an ASR word object (attrs) or a plain dict
# ({'start','end'} or V4's {'w','s','e'} word shape).
WordLike = Union[object, dict]


@dataclass(frozen=True)
class SensorReadings:
    """Layer-1 output: raw, measured signals for one clip."""

    duration_s: float
    audio_rms_mean: float
    audio_rms_peak: float
    audio_energy_variance: float
    integrated_lufs: Optional[float]
    speech_pace_wps: float
    scene_change_rate_per_min: float
    scene_change_count: int
    avg_brightness: float   # 0..1 normalised luma
    avg_saturation: float   # 0..1 normalised saturation
    warnings: list = field(default_factory=list)


# ── Internal helpers ────────────────────────────────────────────────────────

def _speech_pace_wps(words: Optional[Sequence[WordLike]]) -> float:
    """Words-per-second from transcript word timings. Accepts objects with
    .start/.end, dicts with start/end, or V4's {'w','s','e'} shape. Falls
    back to a 0.35 s/word nominal span on degenerate input."""
    if not words:
        return 0.0

    starts: list = []
    ends: list = []
    for w in words:
        if isinstance(w, dict):
            start = w.get("start", w.get("s"))
            end = w.get("end", w.get("e"))
        else:
            start = getattr(w, "start", None)
            end = getattr(w, "end", None)
        if start is None:
            continue
        starts.append(float(start))
        ends.append(float(end) if end is not None else float(start))

    if not starts:
        return 0.0

    n = len(starts)
    span = max(ends) - min(starts)
    if span <= 0.0:
        span = n * 0.35
    return n / span if span > 0.0 else 0.0


def _extract_color_stats(
    video_path: str, ffmpeg_bin: str, *, sample_fps: float = _SIGNALSTATS_SAMPLE_FPS,
) -> tuple:
    """(avg_brightness_0_1, avg_saturation_0_1, warnings) via FFmpeg
    signalstats. Neutral (0.5, 0.5) on any failure."""
    warns: list = []
    # fps-first drops to ~1 frame/s before the expensive stats, and the
    # 180-line downscale keeps signalstats itself cheap; on this product's
    # real 12-40 min masters the undecimated form ran into the 300 s
    # timeout and silently neutralised brightness/saturation (0.5, 0.5) —
    # which spuriously fires the formula's cinematic rule on long jobs.
    cmd = [
        ffmpeg_bin, "-hide_banner",
        "-i", video_path,
        "-vf", f"fps={sample_fps},scale=-2:180,signalstats,metadata=print",
        "-an", "-sn",
        "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=300, errors="replace")
    except subprocess.TimeoutExpired:
        return 0.5, 0.5, ["signalstats: ffmpeg timed out; using neutral defaults"]
    except Exception as exc:
        return 0.5, 0.5, [f"signalstats: ffmpeg failed to run: {exc}"]

    output = (proc.stdout or "") + (proc.stderr or "")
    yavg = [float(m) for m in _YAVG_RE.findall(output)]
    satavg = [float(m) for m in _SATAVG_RE.findall(output)]

    if not yavg:
        warns.append("signalstats: no YAVG samples parsed; using neutral defaults")
        return 0.5, 0.5, warns

    brightness = (sum(yavg) / len(yavg)) / 255.0
    saturation = (sum(satavg) / len(satavg)) / 255.0 if satavg else 0.5
    return (
        max(0.0, min(1.0, brightness)),
        max(0.0, min(1.0, saturation)),
        warns,
    )


def _extract_audio_energy(video_path: str, ffmpeg_bin: str) -> tuple:
    """(rms_mean, rms_peak, rms_variance, warnings) from a real RMS contour
    of the source audio (50 ms frames)."""
    warns: list = []
    tmp_dir = tempfile.mkdtemp(prefix="kaizer_dirplat_sensors_")
    try:
        wav_path = extract_wav(video_path, tmp_dir, ffmpeg_bin, sr=_TARGET_SR)
        y, sr = load_wav_mono(wav_path)
        if len(y) == 0:
            warns.append("Extracted audio was empty; RMS metrics default to 0.")
            return 0.0, 0.0, 0.0, warns

        frame_len = max(1, int(sr * _RMS_FRAME_MS / 1000.0))
        n_frames = max(1, len(y) // frame_len)
        rms_vals = np.array([
            float(np.sqrt(np.mean(
                y[i * frame_len:(i + 1) * frame_len].astype(np.float64) ** 2
            )) + 1e-12)
            for i in range(n_frames)
        ])
        return (
            float(np.mean(rms_vals)),
            float(np.max(rms_vals)),
            float(np.var(rms_vals)),
            warns,
        )
    except Exception as exc:
        warns.append(f"Audio RMS extraction failed: {exc}")
        return 0.0, 0.0, 0.0, warns
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Public API ──────────────────────────────────────────────────────────────

def extract_sensors(
    video_path: str,
    *,
    words: Optional[Sequence[WordLike]] = None,
    scene_change_threshold: float = 0.4,
) -> SensorReadings:
    """Run every Layer-1 sensor and return a populated ``SensorReadings``.
    Never raises: individual sensor failures degrade to neutral defaults
    recorded in ``.warnings``."""
    video_path = video_path.replace("\\", "/")
    warnings_list: list = []

    duration_s = probe_duration(video_path)
    if duration_s <= 0.0:
        warnings_list.append(
            "Could not probe duration; rate-based metrics default to 0."
        )

    ffmpeg_bin = resolve_ffmpeg()
    from pipeline_v4.director_platform.deps import measure_integrated_lufs

    # The four media measurements are INDEPENDENT ffmpeg passes over the
    # same source — run them concurrently so a long master costs max(),
    # not sum() (the V4 engine parallelises its sensors the same way,
    # director.py:1082). Upstream ran them sequentially; readings are
    # identical either way.
    def _cuts():
        return detect_scene_cuts(video_path,
                                 threshold=scene_change_threshold)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4,
                            thread_name_prefix="dirplat-sense") as pool:
        f_cuts = pool.submit(_cuts)
        f_rms = pool.submit(_extract_audio_energy, video_path, ffmpeg_bin)
        f_lufs = pool.submit(measure_integrated_lufs, video_path)
        f_color = pool.submit(_extract_color_stats, video_path, ffmpeg_bin)

        # Scene changes — our downscaled scdet sensor.
        try:
            boundaries = f_cuts.result()
        except Exception as exc:
            boundaries = []
            warnings_list.append(f"scene-cut detect failed: {exc}")
        scene_change_count = len(boundaries)
        scene_change_rate = (
            scene_change_count / (duration_s / 60.0)
            if duration_s > 0.0 else 0.0
        )

        # Audio RMS contour (never raises — returns its warnings).
        rms_mean, rms_peak, rms_var, rms_warnings = f_rms.result()
        warnings_list.extend(rms_warnings)

        # Integrated loudness — our pass-1 loudnorm measurement.
        integrated_lufs = None
        try:
            integrated_lufs = f_lufs.result()
        except Exception as exc:
            warnings_list.append(f"Loudness measurement failed: {exc}")

        # Colour / brightness (never raises — returns its warnings).
        avg_brightness, avg_saturation, color_warnings = f_color.result()
        warnings_list.extend(color_warnings)

    # Speech pace from transcript timing.
    speech_pace = _speech_pace_wps(words)

    return SensorReadings(
        duration_s=duration_s,
        audio_rms_mean=rms_mean,
        audio_rms_peak=rms_peak,
        audio_energy_variance=rms_var,
        integrated_lufs=integrated_lufs,
        speech_pace_wps=speech_pace,
        scene_change_rate_per_min=scene_change_rate,
        scene_change_count=scene_change_count,
        avg_brightness=avg_brightness,
        avg_saturation=avg_saturation,
        warnings=warnings_list,
    )
