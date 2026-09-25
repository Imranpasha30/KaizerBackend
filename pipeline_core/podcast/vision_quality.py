# Ported from kaizer-platform@d5fd482 server/pipeline_core/podcast/vision_quality.py.
# Changes from upstream: none (verbatim copy; origin header added).
"""Vision quality gate (Phase B) — cuts out-of-focus / bad-exposure footage.

The mechanical cutlist (cutlist.py) and the LLM editorial pass (editorial.py)
both decide what to cut from the AUDIO/transcript side. Neither looks at the
picture. A span can be perfectly good audio and still be unusable footage —
someone knocks the camera out of focus, a hand covers the lens, the exposure
blows out pointing at a window. This module finds those spans by sampling
frames with OpenCV (already a dependency — see face_track.py) and reports
them in the SAME (start_s, end_s) source-time coordinate system as
cutlist.CutlistResult.keep_ranges, so the caller can carve them out with
``subtract_spans`` before rendering.

Method (honest, no ML): Laplacian-variance sharpness (a standard, well-known
focus-quality proxy — a sharp image has strong high-frequency edges, a
blurry one doesn't) + mean luminance for exposure. Sampled at a low fixed
rate (default 2fps) — blur/exposure problems are not sub-second events, so
per-frame precision buys nothing but cost.

Honesty: this catches OPTICAL defects (blur, blown-out/under-exposed),
NOT bad framing, bad composition, or "nothing interesting is happening" —
those require actual scene understanding this module does not attempt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

logger = logging.getLogger("pipeline_core.podcast.vision_quality")


@dataclass(frozen=True)
class QualityConfig:
    sample_fps: float = 2.0
    blur_threshold: float = 60.0     # Laplacian variance below this = blurry
    dark_threshold: float = 25.0     # mean luminance (0-255) below this = too dark
    bright_threshold: float = 235.0  # mean luminance above this = blown out/overexposed
    min_bad_run_s: float = 0.6       # ignore blips shorter than this (avoid flicker cuts)
    pad_s: float = 0.15              # pad each bad span so the cut lands clear of the defect


@dataclass(frozen=True)
class QualitySpan:
    start_s: float
    end_s: float
    category: str   # "blurry" | "too_dark" | "overexposed"
    reason: str


# ── Frame sampling (I/O — not unit tested at this layer) ──────────────────


def _sample_metrics(video_path: str, sample_fps: float) -> tuple[list[tuple[float, float, float]], float]:
    """Sample the video at ~``sample_fps`` and return ``(samples, interval_s)``
    where each sample is ``(t, blur_variance, brightness)``.

    Uses ``cap.grab()`` (cheap, no decode) for skipped frames and only
    ``cap.retrieve()`` (decode) on the sampled ones — the same skip-cheaply
    pattern as a full per-frame face-tracking pass would need, but far
    lighter since blur/exposure only need ~2 samples/sec.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cv2.VideoCapture could not open {video_path!r}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, round(fps / sample_fps))
        interval_s = stride / fps

        samples: list[tuple[float, float, float]] = []
        idx = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % stride == 0:
                ok2, frame = cap.retrieve()
                if ok2:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    brightness = float(gray.mean())
                    samples.append((idx / fps, blur_var, brightness))
            idx += 1
        return samples, interval_s
    finally:
        cap.release()


# ── Pure classification / coalescing (unit tested) ─────────────────────────


def _category(blur_var: float, brightness: float, cfg: QualityConfig) -> str | None:
    if blur_var < cfg.blur_threshold:
        return "blurry"
    if brightness < cfg.dark_threshold:
        return "too_dark"
    if brightness > cfg.bright_threshold:
        return "overexposed"
    return None


def _classify_and_coalesce(
    samples: Sequence[tuple[float, float, float]],
    cfg: QualityConfig,
    sample_interval_s: float,
) -> list[QualitySpan]:
    """Walk time-ordered samples, coalesce consecutive same-category bad
    samples into runs, drop runs shorter than ``min_bad_run_s``, pad the
    survivors, then merge any spans padding brought into contact."""
    runs: list[QualitySpan] = []
    run_cat: str | None = None
    run_start: float | None = None
    run_end: float | None = None

    def _flush() -> None:
        if run_cat is None or run_start is None or run_end is None:
            return
        dur = (run_end - run_start) + sample_interval_s
        if dur >= cfg.min_bad_run_s:
            runs.append(QualitySpan(
                start_s=max(0.0, run_start - cfg.pad_s),
                end_s=run_end + sample_interval_s + cfg.pad_s,
                category=run_cat,
                reason=f"{run_cat} for {dur:.1f}s",
            ))

    for t, blur_var, brightness in samples:
        cat = _category(blur_var, brightness, cfg)
        if cat is not None and cat == run_cat:
            run_end = t
            continue
        _flush()
        if cat is not None:
            run_cat, run_start, run_end = cat, t, t
        else:
            run_cat = run_start = run_end = None
    _flush()

    runs.sort(key=lambda r: r.start_s)
    merged: list[QualitySpan] = []
    for r in runs:
        if merged and r.start_s <= merged[-1].end_s:
            prev = merged[-1]
            merged[-1] = QualitySpan(
                start_s=prev.start_s, end_s=max(prev.end_s, r.end_s),
                category=prev.category, reason=f"{prev.reason}; {r.reason}",
            )
        else:
            merged.append(r)
    return merged


def subtract_spans(
    keep_ranges: Sequence[tuple[float, float]],
    bad_spans: Sequence[QualitySpan],
) -> tuple[tuple[float, float], ...]:
    """Carve ``bad_spans`` (SOURCE time) out of ``keep_ranges`` (SOURCE time),
    splitting a kept range in two when a bad span falls in its interior.

    Pure interval subtraction — no ffmpeg, no I/O — so this stays cheap to
    call even when there are no bad spans (returns the input unchanged).
    """
    if not bad_spans:
        return tuple(keep_ranges)
    bads = sorted(((b.start_s, b.end_s) for b in bad_spans), key=lambda b: b[0])
    out: list[tuple[float, float]] = []
    for r_s, r_e in keep_ranges:
        pieces = [(r_s, r_e)]
        for b_s, b_e in bads:
            next_pieces: list[tuple[float, float]] = []
            for p_s, p_e in pieces:
                if b_e <= p_s or b_s >= p_e:
                    next_pieces.append((p_s, p_e))
                    continue
                if b_s > p_s:
                    next_pieces.append((p_s, b_s))
                if b_e < p_e:
                    next_pieces.append((b_e, p_e))
            pieces = next_pieces
        out.extend(p for p in pieces if p[1] - p[0] > 1e-6)
    return tuple(out)


# ── Public API ───────────────────────────────────────────────────────────


def analyze_video_quality(video_path: str, cfg: QualityConfig | None = None) -> list[QualitySpan]:
    """Sample ``video_path`` and return bad-quality spans in source time.

    Returns an empty list (never raises for "no problems found") — callers
    should still let genuine I/O errors (unreadable file) propagate so a
    broken video doesn't silently render as if it were fine.
    """
    cfg = cfg or QualityConfig()
    samples, interval_s = _sample_metrics(video_path, cfg.sample_fps)
    if not samples:
        return []
    spans = _classify_and_coalesce(samples, cfg, interval_s)
    if spans:
        logger.info("vision quality: %d bad span(s) in %s", len(spans), video_path)
    return spans


__all__ = ["QualityConfig", "QualitySpan", "analyze_video_quality", "subtract_spans"]
