"""
kaizer.pipeline.loop_score
===========================
Loop-quality scorer for Instagram Reels variants.

Reels' algorithm silently rewards seamless-looping clips — a watched loop
counts as additional watch-time, compounding the primary distribution
signal. A clip whose last frame gracefully matches its first frame lifts
reach materially compared with a clip that hard-cuts at the seam.

This module scores how loopable a given MP4 is and suggests concrete
fixes when the seam is weak.

Usage
-----
    from pipeline_core.loop_score import score_loop, LoopScore

    score = score_loop("/path/to/clip.mp4")
    if score.overall < 60:
        for fix in score.suggestions:
            print("suggested fix:", fix)

LoopScore fields
----------------
    overall              : float     — 0-100 composite
    visual_phash_distance : int      — lower is better (0 = identical seam)
    audio_xcorr           : float    — |correlation| between 0.5s audio tails
    motion_continuity     : float    — 0-1; 1 = perfectly smooth seam
    suggestions           : list[str]

Composite formula (documented in code):
    visual  = max(0, 1 - phash_distance / 30) * 100
    motion  = motion_continuity * 100
    audio   = audio_xcorr * 100
    overall = 0.5 * visual + 0.3 * motion + 0.2 * audio
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("kaizer.pipeline.loop_score")


# ── Thresholds (module-level so tests can monkey-patch) ──────────────────────
_PHASH_BAD_DISTANCE = 20        # above this → bad seam → suggest crossfade
_AUDIO_XCORR_BAD = 0.3          # below this → bad audio seam → trim tail
_MOTION_BAD = 0.4               # below this → bad motion seam → crossfade


# ── Public dataclass ──────────────────────────────────────────────────────────

@dataclass
class LoopScore:
    """Result of a single score_loop() call."""
    overall: float
    visual_phash_distance: int
    audio_xcorr: float
    motion_continuity: float
    suggestions: list[str] = field(default_factory=list)


# ── pHash implementation (DCT-based, 64-bit) ─────────────────────────────────

def _phash_64(img_bgr: np.ndarray) -> int:
    """Compute a 64-bit perceptual hash of a BGR frame.

    Algorithm (Marr 2011, common pHash):
      1. Convert to greyscale.
      2. Resize to 32x32 (fixed) using INTER_AREA.
      3. 2-D DCT.
      4. Take the top-left 8x8 DCT coefficients (low-frequency energy).
      5. Exclude the DC term (position [0,0]) when computing the mean.
      6. Hash bit = 1 where coefficient > mean, else 0.
      7. Pack 64 bits into a Python int, MSB first.
    """
    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    block = dct[:8, :8].copy()
    # Exclude DC term from mean — it swamps the signal
    flat = np.concatenate(([block[0, 0]], block.flatten()[1:]))
    mean = float(flat[1:].mean())  # mean over AC terms only
    bits = (block > mean).flatten().astype(np.uint8)
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _hamming(a: int, b: int) -> int:
    """Population count of XOR — Hamming distance for 64-bit ints."""
    return bin(a ^ b).count("1")


# ── Audio extraction + xcorr ─────────────────────────────────────────────────

def _extract_audio_tails(
    video_path: str,
    *,
    head_s: float = 0.5,
    tail_s: float = 0.5,
    sr: int = 16000,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract the first head_s and last tail_s of audio as 1-D float32 arrays
    at `sr` Hz. Returns (head_samples, tail_samples) or (None, None) if the
    file has no audio or extraction fails.
    """
    # Probe duration
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", video_path],
            capture_output=True, text=True, timeout=30,
        )
        dur = float((probe.stdout or "0").strip())
    except Exception as exc:
        logger.warning("ffprobe duration failed for %s: %s", video_path, exc)
        return None, None

    if dur <= (head_s + tail_s):
        # Too short for distinct head/tail — fold the clip onto itself
        logger.info(
            "Video %s (%.2fs) shorter than head+tail (%.2fs); using full clip twice",
            video_path, dur, head_s + tail_s,
        )
        tails_start = 0.0
        head_len = tail_len = max(0.1, dur / 2)
    else:
        tails_start = max(0.0, dur - tail_s)
        head_len = head_s
        tail_len = tail_s

    import librosa  # lazy import — heavy

    def _load_window(offset: float, duration: float) -> Optional[np.ndarray]:
        try:
            y, _ = librosa.load(video_path, sr=sr, mono=True,
                                offset=offset, duration=duration)
            return y.astype(np.float32, copy=False) if y.size else None
        except Exception as exc:
            logger.warning("librosa failed on %s @ %.2fs: %s", video_path, offset, exc)
            return None

    head = _load_window(0.0, head_len)
    tail = _load_window(tails_start, tail_len)
    return head, tail


def _audio_xcorr(head: np.ndarray, tail: np.ndarray) -> float:
    """Absolute normalized cross-correlation between head and tail audio.

    Returns a value in [0, 1]. Resamples the shorter array to match the
    longer, then uses np.corrcoef.
    """
    if head is None or tail is None or head.size == 0 or tail.size == 0:
        return 0.0

    # Trim to same length
    n = min(head.size, tail.size)
    if n < 16:
        return 0.0
    h = head[:n]
    t = tail[:n]

    # Zero-variance → undefined correlation; treat as 0 (no signal)
    if h.std() < 1e-6 or t.std() < 1e-6:
        return 0.0

    with np.errstate(invalid="ignore"):
        r = np.corrcoef(h, t)[0, 1]
    if np.isnan(r):
        return 0.0
    return float(min(1.0, abs(r)))


# ── Motion continuity ────────────────────────────────────────────────────────

def _motion_continuity(last_frame_bgr: np.ndarray, first_frame_bgr: np.ndarray) -> float:
    """1 - normalized mean absolute difference between seam frames (greyscale).

    Returns a value in [0, 1]. 1 = frames identical, 0 = frames maximally
    different.
    """
    last_g = cv2.cvtColor(last_frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    first_g = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Resize to match — handle any res drift
    if last_g.shape != first_g.shape:
        first_g = cv2.resize(first_g, (last_g.shape[1], last_g.shape[0]))
    abs_diff = np.mean(np.abs(last_g - first_g)) / 255.0
    return float(max(0.0, 1.0 - abs_diff))


# ── Frame extraction ──────────────────────────────────────────────────────────

def _extract_end_frames(
    video_path: str,
    *,
    frames_each_end: int = 6,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Pull the first N and last N frames from video_path as BGR arrays."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("cv2.VideoCapture failed to open %s", video_path)
        return [], []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return [], []

    n = max(1, min(frames_each_end, total // 2))

    first_frames: list[np.ndarray] = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok and frame is not None:
            first_frames.append(frame)

    last_frames: list[np.ndarray] = []
    for i in range(total - n, total):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, i))
        ok, frame = cap.read()
        if ok and frame is not None:
            last_frames.append(frame)

    cap.release()
    return first_frames, last_frames


# ── Public API ────────────────────────────────────────────────────────────────

def score_loop(
    video_path: str,
    *,
    frames_each_end: int = 6,
) -> LoopScore:
    """Compute a loop-quality score for `video_path`.

    Parameters
    ----------
    video_path : str
        Absolute path to a readable video file.
    frames_each_end : int
        How many frames on each end to average for the visual and motion seams.

    Returns
    -------
    LoopScore
        Always returned, even for missing audio / unreadable frames (those
        components degrade to 0 with suggestions logged).

    Notes
    -----
    Never raises — all failure paths degrade gracefully so the caller can
    still decide whether to proceed with a variant render.
    """
    if not os.path.exists(video_path):
        logger.warning("loop_score: path not found %s", video_path)
        return LoopScore(
            overall=0.0, visual_phash_distance=64,
            audio_xcorr=0.0, motion_continuity=0.0,
            suggestions=["video_path does not exist"],
        )

    # ── Visual: pHash distance at seam ────────────────────────────────────────
    first_frames, last_frames = _extract_end_frames(video_path, frames_each_end=frames_each_end)
    if not first_frames or not last_frames:
        logger.warning("loop_score: could not extract seam frames from %s", video_path)
        visual_distance = 64
        motion_cont = 0.0
    else:
        first_hashes = [_phash_64(f) for f in first_frames]
        last_hashes = [_phash_64(f) for f in last_frames]
        # Cross-compare: average Hamming distance across all first-last pairs
        dists = [_hamming(a, b) for a in last_hashes for b in first_hashes]
        visual_distance = int(sum(dists) / max(1, len(dists)))
        motion_cont = _motion_continuity(last_frames[-1], first_frames[0])

    # ── Audio xcorr ──────────────────────────────────────────────────────────
    head, tail = _extract_audio_tails(video_path)
    audio_xcorr = _audio_xcorr(head, tail) if head is not None and tail is not None else 0.0

    # ── Composite ────────────────────────────────────────────────────────────
    visual_component = max(0.0, 1.0 - visual_distance / 30.0) * 100.0
    motion_component = motion_cont * 100.0
    audio_component = audio_xcorr * 100.0
    overall = 0.5 * visual_component + 0.3 * motion_component + 0.2 * audio_component

    # ── Suggestions ──────────────────────────────────────────────────────────
    suggestions: list[str] = []
    if visual_distance > _PHASH_BAD_DISTANCE:
        suggestions.append(
            f"Freeze or crossfade last {frames_each_end} frames to match first frame "
            f"(visual seam Hamming distance {visual_distance} > {_PHASH_BAD_DISTANCE})"
        )
    if audio_xcorr < _AUDIO_XCORR_BAD:
        suggestions.append(
            f"Trim audio tail to beat-boundary of first sample "
            f"(audio_xcorr {audio_xcorr:.2f} < {_AUDIO_XCORR_BAD})"
        )
    if motion_cont < _MOTION_BAD:
        suggestions.append(
            f"Add 150ms crossfade at seam "
            f"(motion continuity {motion_cont:.2f} < {_MOTION_BAD})"
        )

    logger.info(
        "loop_score: %s overall=%.1f phash=%d xcorr=%.2f motion=%.2f suggestions=%d",
        video_path, overall, visual_distance, audio_xcorr, motion_cont, len(suggestions),
    )

    return LoopScore(
        overall=round(overall, 2),
        visual_phash_distance=visual_distance,
        audio_xcorr=audio_xcorr,
        motion_continuity=motion_cont,
        suggestions=suggestions,
    )
