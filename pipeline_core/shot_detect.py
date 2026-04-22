"""
kaizer.pipeline.shot_detect
============================
Shot boundary detection for the Kaizer News video pipeline.

Two complementary detection modes:
  ``"scdet"``     — FFmpeg scene-change detection filter (default; fast; no ML).
  ``"cutpoint"``  — ONNX Bi-LSTM model trained on 5-dim audio features
                    (requires ``cut_point.onnx`` on disk).

Usage
-----
    from pipeline_core.shot_detect import detect_shots, get_shot_ranges, ShotBoundary

    # Fast mode — uses FFmpeg scdet filter
    boundaries = detect_shots("/path/to/video.mp4", method="scdet", threshold=0.4)

    # ML mode — uses audio features + ONNX Bi-LSTM
    boundaries = detect_shots("/path/to/video.mp4", method="cutpoint", threshold=0.5)

    # Convert to (start, end) shot ranges
    ranges = get_shot_ranges(boundaries, total_duration=120.0)

ShotBoundary fields
-------------------
  t          : float  — Timestamp in seconds from the start of the video.
  confidence : float  — Normalised confidence score in [0.0, 1.0].
  method     : str    — ``"scdet"`` or ``"cutpoint"``.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("kaizer.pipeline.shot_detect")

# ── Model path ────────────────────────────────────────────────────────────────

_BASE_DIR: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_CUTPOINT_ONNX: str = os.path.join(_BASE_DIR, "models", "cut_point.onnx")

# Audio feature extraction parameters (must match training in 06_train_cutpoint.py)
_FRAME_HOP_S: float = 0.1    # 100 ms hop → 10 frames/second
_SEQ_LEN: int = 300           # window length fed to ONNX model (~30 s of audio)
_INPUT_DIM: int = 5           # rms_energy, spectral_flux, zcr, silence_flag, energy_delta
_SILENCE_THRESHOLD: float = 0.05   # RMS below this → silence_flag=1


# ── Public dataclass ───────────────────────────────────────────────────────────

@dataclass
class ShotBoundary:
    """A single detected shot boundary.

    Attributes
    ----------
    t : float
        Position in seconds from the start of the video.
    confidence : float
        Normalised confidence in [0.0, 1.0].
    method : str
        Either ``"scdet"`` or ``"cutpoint"``.
    """

    t: float
    confidence: float
    method: str


# ── FFmpeg scdet helpers ───────────────────────────────────────────────────────

def _get_ffmpeg() -> str:
    """Return the FFmpeg binary path."""
    try:
        from pipeline_core.pipeline import FFMPEG_BIN  # type: ignore
        return FFMPEG_BIN
    except Exception:
        import shutil as _sh
        return _sh.which("ffmpeg") or "ffmpeg"


def _scdet_detect(video_path: str, threshold: float) -> list[ShotBoundary]:
    """Run FFmpeg scdet filter and parse shot boundaries from stderr.

    Parameters
    ----------
    video_path : str
        Path to the input video.
    threshold : float
        Normalised threshold in [0, 1] — converted to the scdet scale [0, 100].

    Returns
    -------
    list[ShotBoundary]
        Detected boundaries sorted by timestamp.
    """
    ffmpeg = _get_ffmpeg()
    scdet_threshold = max(0.0, min(100.0, threshold * 100.0))

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-i", video_path,
        "-vf", f"scdet=threshold={scdet_threshold:.1f}",
        "-f", "null",
        "-",
    ]
    logger.debug("shot_detect scdet: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        logger.warning("shot_detect: FFmpeg scdet timed out for %s", video_path)
        return []
    except FileNotFoundError:
        logger.warning("shot_detect: FFmpeg binary not found at %s", ffmpeg)
        return []

    # Parse stderr lines like:
    #   [scdet @ 0x...] lavfi.scd.score: 45.23 lavfi.scd.mafd: 12.1 pts:1234 pts_time:41.1
    boundaries: list[ShotBoundary] = []
    pattern = re.compile(
        r"lavfi\.scd\.score:\s*([0-9.]+).*?pts_time:\s*([0-9.]+)", re.IGNORECASE
    )
    for line in proc.stderr.splitlines():
        if "scdet" not in line.lower() and "scd.score" not in line.lower():
            continue
        m = pattern.search(line)
        if m:
            score_raw = float(m.group(1))
            pts_time = float(m.group(2))
            confidence = min(1.0, score_raw / 100.0)
            boundaries.append(
                ShotBoundary(t=pts_time, confidence=confidence, method="scdet")
            )

    boundaries.sort(key=lambda b: b.t)
    logger.info(
        "shot_detect: scdet found %d boundaries in %s", len(boundaries), video_path
    )
    return boundaries


# ── Audio feature extraction helpers ──────────────────────────────────────────

def _extract_audio_features(video_path: str) -> Optional[np.ndarray]:
    """Extract 5-dim per-frame audio features for the cutpoint model.

    Feature vector per 100 ms frame:
      [0] rms_energy       — root mean square of the frame
      [1] spectral_flux    — sum of positive spectral difference from prev frame
      [2] zero_crossing_rate
      [3] silence_flag     — 1 if rms < 0.05, else 0
      [4] energy_delta     — first difference of rms energy

    Returns
    -------
    np.ndarray of shape (n_frames, 5) or None if extraction fails.
    """
    try:
        import librosa  # type: ignore
    except ImportError:
        logger.warning("shot_detect: librosa not available; cutpoint mode unavailable")
        return None

    try:
        import tempfile, shutil
        from pipeline_core.pipeline import FFMPEG_BIN  # type: ignore

        ffmpeg = FFMPEG_BIN
    except Exception:
        import shutil as _sh
        ffmpeg = _sh.which("ffmpeg") or "ffmpeg"

    # Extract 16 kHz mono WAV to a temp file
    tmp_wav: Optional[str] = None
    try:
        fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="kaizer_shot_")
        os.close(fd)
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-y", tmp_wav,
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
        if proc.returncode != 0:
            logger.warning(
                "shot_detect: audio extraction failed (exit %d)", proc.returncode
            )
            return None

        y, sr = librosa.load(tmp_wav, sr=16000, mono=True)
    except Exception as exc:
        logger.warning("shot_detect: audio load failed: %s", exc)
        return None
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except Exception:
                pass

    if len(y) == 0:
        logger.warning("shot_detect: empty audio array for %s", video_path)
        return None

    # Frame length and hop in samples
    hop_length = int(sr * _FRAME_HOP_S)
    frame_length = hop_length * 2  # 200 ms window for each 100 ms frame

    # RMS energy
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Spectral flux: positive part of the magnitude spectrum difference
    stft = np.abs(librosa.stft(y, hop_length=hop_length, n_fft=frame_length))
    flux = np.maximum(0, np.diff(stft, axis=1, prepend=stft[:, :1]))
    spectral_flux = flux.sum(axis=0)

    # Trim / pad to match rms length
    n = len(rms)
    spectral_flux = spectral_flux[:n] if len(spectral_flux) >= n else np.pad(
        spectral_flux, (0, n - len(spectral_flux))
    )

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length,
                                              hop_length=hop_length)[0]
    zcr = zcr[:n] if len(zcr) >= n else np.pad(zcr, (0, n - len(zcr)))

    # Normalise to [0, 1]
    def _norm(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            return (arr - mn) / (mx - mn)
        return np.zeros_like(arr)

    rms_n = _norm(rms).astype(np.float32)
    flux_n = _norm(spectral_flux).astype(np.float32)
    zcr_n = _norm(zcr).astype(np.float32)
    silence = (rms < _SILENCE_THRESHOLD).astype(np.float32)
    energy_delta = np.diff(rms_n, prepend=rms_n[:1]).astype(np.float32)

    features = np.stack([rms_n, flux_n, zcr_n, silence, energy_delta], axis=1)
    return features  # shape: (n_frames, 5)


def _cutpoint_detect(video_path: str, threshold: float) -> list[ShotBoundary]:
    """Use the ONNX Bi-LSTM model to detect shot boundaries from audio features.

    Parameters
    ----------
    video_path : str
        Path to the input video.
    threshold : float
        Per-frame probability threshold for declaring a boundary.

    Returns
    -------
    list[ShotBoundary]
        Detected boundaries sorted by timestamp.
    """
    if not os.path.isfile(_CUTPOINT_ONNX):
        logger.warning(
            "shot_detect: cut_point.onnx not found at %s; skipping cutpoint detection",
            _CUTPOINT_ONNX,
        )
        return []

    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        logger.warning("shot_detect: onnxruntime not available; cutpoint mode disabled")
        return []

    features = _extract_audio_features(video_path)
    if features is None or features.shape[0] == 0:
        logger.warning("shot_detect: could not extract audio features from %s", video_path)
        return []

    n_frames = features.shape[0]
    logger.info("shot_detect: cutpoint — %d audio frames extracted", n_frames)

    # Build overlapping windows of shape (_SEQ_LEN, 5)
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3  # suppress ONNX verbose output
    try:
        sess = ort.InferenceSession(_CUTPOINT_ONNX, sess_options=sess_options)
    except Exception as exc:
        logger.warning("shot_detect: ONNX session creation failed: %s", exc)
        return []

    input_name = sess.get_inputs()[0].name
    # Aggregate per-frame probabilities across all windows
    probs = np.zeros(n_frames, dtype=np.float32)
    counts = np.zeros(n_frames, dtype=np.int32)

    step = _SEQ_LEN // 2  # 50% overlap
    for start in range(0, n_frames, step):
        end = start + _SEQ_LEN
        chunk = features[start:end]
        if len(chunk) < _SEQ_LEN:
            # Pad the last window
            pad = np.zeros((_SEQ_LEN - len(chunk), _INPUT_DIM), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
        chunk_input = chunk[np.newaxis, :, :]  # (1, SEQ_LEN, 5)

        try:
            outputs = sess.run(None, {input_name: chunk_input})
        except Exception as exc:
            logger.warning("shot_detect: ONNX inference failed at frame %d: %s", start, exc)
            continue

        # Output is (1, SEQ_LEN) or (1, SEQ_LEN, 1) probabilities
        out = np.array(outputs[0]).squeeze()
        if out.ndim == 0:
            out = out.reshape(1)
        out = out.flatten()
        real_len = min(len(out), n_frames - start, _SEQ_LEN)
        probs[start:start + real_len] += out[:real_len]
        counts[start:start + real_len] += 1

    # Average overlapping predictions
    valid = counts > 0
    probs[valid] /= counts[valid]

    # Detect peaks above threshold with minimum distance of 10 frames (1 s)
    boundaries: list[ShotBoundary] = []
    min_dist_frames = 10
    last_boundary_frame = -min_dist_frames

    for i, p in enumerate(probs):
        if p >= threshold and (i - last_boundary_frame) >= min_dist_frames:
            t = i * _FRAME_HOP_S
            boundaries.append(
                ShotBoundary(t=t, confidence=float(p), method="cutpoint")
            )
            last_boundary_frame = i

    boundaries.sort(key=lambda b: b.t)
    logger.info(
        "shot_detect: cutpoint found %d boundaries in %s", len(boundaries), video_path
    )
    return boundaries


# ── Public API ─────────────────────────────────────────────────────────────────

def detect_shots(
    video_path: str,
    *,
    method: str = "scdet",
    threshold: float = 0.4,
) -> list[ShotBoundary]:
    """Detect shot boundaries in a video file.

    Parameters
    ----------
    video_path : str
        Absolute path to the input video.
    method : str
        ``"scdet"`` (default) — FFmpeg scene-change detection (fast, no ML).
        ``"cutpoint"``        — ONNX Bi-LSTM audio-feature model.
    threshold : float
        Detection sensitivity in [0, 1].  Lower = more sensitive.

    Returns
    -------
    list[ShotBoundary]
        Shot boundaries sorted by ascending timestamp.  Empty list if none
        found or if the video cannot be processed.
    """
    if not os.path.isfile(video_path):
        logger.warning("shot_detect: file not found: %s", video_path)
        return []

    method = method.lower().strip()

    if method == "scdet":
        return _scdet_detect(video_path, threshold)
    elif method == "cutpoint":
        return _cutpoint_detect(video_path, threshold)
    else:
        logger.warning(
            "shot_detect: unknown method %r; defaulting to 'scdet'", method
        )
        return _scdet_detect(video_path, threshold)


def get_shot_ranges(
    boundaries: list[ShotBoundary],
    total_duration: float,
) -> list[tuple[float, float]]:
    """Convert a sorted list of shot boundaries into (start, end) ranges.

    Parameters
    ----------
    boundaries : list[ShotBoundary]
        Shot boundaries as returned by :func:`detect_shots` (must be sorted
        by ``t`` ascending).
    total_duration : float
        Total video duration in seconds (used to close the final shot).

    Returns
    -------
    list[tuple[float, float]]
        One ``(start, end)`` tuple per shot, sorted by start time.
    """
    if not boundaries:
        return [(0.0, total_duration)]

    times = [b.t for b in boundaries]
    # Ensure we start from 0
    if times[0] > 0.0:
        times = [0.0] + times

    ranges: list[tuple[float, float]] = []
    for i in range(len(times) - 1):
        ranges.append((times[i], times[i + 1]))
    ranges.append((times[-1], total_duration))

    return ranges
