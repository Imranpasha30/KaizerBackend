# Ported from kaizer-platform@d5fd482 server/pipeline_core/podcast/camera_plan.py.
# Changes from upstream: pipeline_core.face_track._find_cascade (module not present in this
# tree) inlined as a local helper; logic unchanged.
"""Podcast camera plan — virtual multi-cam reframing (Phase C).

The "director grammar" ask: on a single-camera two-shot (two people sitting
in one frame — the common single-cam podcast setup), reframe to whoever is
speaking and switch a little BEFORE they start (the visual leads the ear,
the same J-cut instinct a human editor uses), instead of leaving the wide
shot static the whole episode.

Pipeline
--------
1. Sample the source video, run Haar face detection (reuses face_track.py's
   cascade), and cluster detections into up to ``max_faces`` spatial SLOTS
   by x-position — purely spatial, no identity.
2. For each slot, track MOUTH-REGION MOTION ENERGY (frame-to-frame pixel
   difference in the lower third of the face box) — a lightweight, well
   established proxy for "this person is talking" that needs no extra model.
3. Correlate each slot's motion energy against the diarization speaker-turns
   already in the transcript (``word.speaker``) to map speaker-label ->
   slot. If the correlation is inconclusive (thin margin, or fewer than 2
   confidently-mapped speakers), we do NOT guess — the plan comes back
   disengaged and the caller renders the original framing throughout.
4. Turn the mapped speaker-turn sequence into reframe windows in
   EDITED-timeline coordinates, switching ``anticipate_s`` before the next
   speaker's first kept word (bounded so windows never overlap).

Honesty / scope (v1, consistent with face_track.py's stated v1 tradeoffs)
--------------------------------------------------------------------------
* Reframe windows share ONE crop size — only x/y position pans between
  windows. This keeps the ffmpeg filter graph a constant-size crop, so the
  existing fixed-size scale/pad/punch-in/caption chain downstream (render.py)
  needs no changes to cope with a time-varying frame size.
* Speaker-to-face mapping is a MOTION-ENERGY HEURISTIC, not lip-sync or face
  recognition. It is deliberately conservative: no signal -> no reframing.
* Handles exactly the "two guests, one camera" case the product asks for.
  True N-camera switching (separate camera files, real crossfade
  transitions) is a distinct, larger feature and out of scope here.
* Applies to the MAIN EDIT only for now; promo variants and the Remotion
  renderer do not yet consume a camera plan (logged, not silently dropped).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from pipeline_core.podcast.cutlist import WordT

logger = logging.getLogger("pipeline_core.podcast.camera_plan")


@dataclass(frozen=True)
class CameraPlanConfig:
    sample_fps: float = 2.0
    max_faces: int = 2
    anticipate_s: float = 0.45       # switch to the next speaker this long before their first word
    min_turn_s: float = 1.5          # ignore turns shorter than this (avoid flicker-switching)
    zoom_fraction: float = 0.58      # reframe crop width as a fraction of source frame width
    min_confidence_ratio: float = 1.25  # winning slot's energy must beat the runner-up by this much
    mouth_roi_size: int = 32         # mouth ROI is resized to this NxN before diffing


@dataclass(frozen=True)
class ReframeWindow:
    start_s: float              # edited-timeline seconds
    end_s: float
    x: int                      # SOURCE-pixel crop top-left (size is uniform — see CameraPlan)
    y: int
    speaker: Optional[int]


@dataclass(frozen=True)
class CameraPlan:
    windows: tuple[ReframeWindow, ...]
    crop_w: int
    crop_h: int
    default_x: int               # resting/wide position used outside every window
    default_y: int
    engaged: bool                 # False = render the original framing throughout
    reason: str                   # transparency: why engaged/not (mirrors editorial.py's audit)


# ── Timeline mapping (mirrors punchin.py's private helper; duplicated on
#    purpose so the two independently-tested planners stay decoupled) ──────


def _edited_time_of(src_t: float, keep_ranges: Sequence[tuple[float, float]]) -> Optional[float]:
    acc = 0.0
    for s, e in keep_ranges:
        if s <= src_t <= e:
            return acc + (src_t - s)
        acc += e - s
    return None


# ── Geometry ────────────────────────────────────────────────────────────


def _reframe_box_size(
    frame_w: int, frame_h: int, out_aspect: tuple[int, int], zoom_fraction: float
) -> tuple[int, int]:
    aw, ah = out_aspect
    ratio = aw / ah
    w = int(frame_w * zoom_fraction)
    h = int(w / ratio)
    if h > frame_h:
        h = frame_h
        w = int(h * ratio)
    w -= w % 2
    h -= h % 2
    return max(2, w), max(2, h)


def _empty_plan(frame_w: int, frame_h: int, out_aspect: tuple[int, int], reason: str) -> CameraPlan:
    # Full-frame fit (no zoom) as the disengaged resting box.
    crop_w, crop_h = _reframe_box_size(frame_w, frame_h, out_aspect, 1.0)
    x = (frame_w - crop_w) // 2
    y = (frame_h - crop_h) // 2
    logger.info("camera plan disengaged: %s", reason)
    return CameraPlan(windows=(), crop_w=crop_w, crop_h=crop_h,
                       default_x=x, default_y=y, engaged=False, reason=reason)


def _probe_dims(video_path: str) -> tuple[int, int]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cv2.VideoCapture could not open {video_path!r}")
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if w == 0 or h == 0:
        raise ValueError(f"Could not determine frame dimensions from {video_path!r}")
    return w, h


# ── Face-slot detection + mouth-motion sampling (I/O — not unit tested) ────


def _find_cascade() -> str:
    """Locate the Haar frontal-face cascade XML packaged with OpenCV.

    Inlined from kaizer-platform@d5fd482 server/pipeline_core/face_track.py
    (this tree does not carry face_track.py); logic unchanged apart from the
    lazy ``cv2`` import (keeps the pure planners import-light).
    """
    import os

    import cv2

    # cv2.data.haarcascades is the standard path in OpenCV ≥ 3.x
    cascade_dir = getattr(cv2, "data", None)
    if cascade_dir is not None:
        p = os.path.join(cascade_dir.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.isfile(p):
            return p
    # Fallback: try common installation paths
    for candidate in [
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
    ]:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not locate haarcascade_frontalface_default.xml.  "
        "Ensure OpenCV was installed with its data files."
    )


def _match_or_create_slot(
    slot_boxes: list[list[float]], fx: float, fy: float, fw: float, fh: float, *, max_slots: int,
) -> int:
    """Assign a detected face to the nearest existing slot by x-centroid
    distance, or start a new slot while under ``max_slots``. Purely
    spatial clustering — no identity, no face recognition."""
    cx = fx + fw / 2
    if slot_boxes:
        dists = [abs(sb[0] - cx) for sb in slot_boxes]
        best = int(np.argmin(dists))
        if dists[best] < fw * 1.5 or len(slot_boxes) >= max_slots:
            return best
    slot_boxes.append([cx, fy + fh / 2, fw, fh, 0.0])
    return len(slot_boxes) - 1


def _detect_slots_and_motion(
    video_path: str, cfg: CameraPlanConfig,
) -> tuple[int, int, list[dict], list[tuple[float, dict[int, float]]]]:
    """Single pass over the source video: detect up to ``cfg.max_faces`` per
    sampled frame, cluster into stable slots, and compute mouth-region
    motion energy per slot per sample.

    Returns ``(frame_w, frame_h, slots, motion_samples)`` where each slot is
    ``{"cx": float, "cy": float, "w": float, "h": float}`` (running average
    over all detections assigned to it) and ``motion_samples`` is
    ``[(t, {slot_idx: energy})]`` in increasing time order.
    """
    import cv2

    cascade = cv2.CascadeClassifier(_find_cascade())
    if cascade.empty():
        raise RuntimeError("OpenCV CascadeClassifier failed to load the Haar cascade")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cv2.VideoCapture could not open {video_path!r}")
    try:
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, round(fps / cfg.sample_fps))

        slot_boxes: list[list[float]] = []   # [cx, cy, w, h, n] running average
        prev_mouth: dict[int, np.ndarray] = {}
        motion_samples: list[tuple[float, dict[int, float]]] = []

        idx = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % stride == 0:
                ok2, frame = cap.retrieve()
                if ok2:
                    t = idx / fps
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                    )
                    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:cfg.max_faces]

                    frame_energy: dict[int, float] = {}
                    for (fx, fy, fw, fh) in faces:
                        slot_idx = _match_or_create_slot(
                            slot_boxes, fx, fy, fw, fh, max_slots=cfg.max_faces)
                        sb = slot_boxes[slot_idx]
                        n = sb[4]
                        sb[0] = (sb[0] * n + (fx + fw / 2)) / (n + 1)
                        sb[1] = (sb[1] * n + (fy + fh / 2)) / (n + 1)
                        sb[2] = (sb[2] * n + fw) / (n + 1)
                        sb[3] = (sb[3] * n + fh) / (n + 1)
                        sb[4] = n + 1

                        my0, my1 = fy + int(fh * 0.65), min(frame_h, fy + fh)
                        mx0, mx1 = fx, min(frame_w, fx + fw)
                        if my1 > my0 and mx1 > mx0:
                            roi = cv2.resize(
                                gray[my0:my1, mx0:mx1],
                                (cfg.mouth_roi_size, cfg.mouth_roi_size))
                            prev = prev_mouth.get(slot_idx)
                            if prev is not None:
                                frame_energy[slot_idx] = float(
                                    np.abs(roi.astype(np.int16) - prev.astype(np.int16)).mean())
                            prev_mouth[slot_idx] = roi
                    if frame_energy:
                        motion_samples.append((t, frame_energy))
            idx += 1
    finally:
        cap.release()

    slots = [{"cx": sb[0], "cy": sb[1], "w": sb[2], "h": sb[3]} for sb in slot_boxes]
    return frame_w, frame_h, slots, motion_samples


# ── Pure logic (unit tested) ────────────────────────────────────────────


def _assign_slots_to_speakers(
    motion_samples: Sequence[tuple[float, dict[int, float]]],
    words: Sequence[WordT],
    cfg: CameraPlanConfig,
) -> Optional[dict[int, int]]:
    """Correlate each slot's motion energy with each speaker's active
    windows (word spans, padded 0.25s). Returns ``{speaker: slot_idx}``, or
    None if fewer than 2 speakers get a confident (non-ambiguous) slot, or
    two speakers would resolve to the SAME slot."""
    if not motion_samples:
        return None

    spans = sorted(
        ((float(w.s) - 0.25, float(w.e) + 0.25, w.speaker)
         for w in words if w.speaker is not None),
        key=lambda s: s[0],
    )
    if not spans:
        return None

    energy: dict[int, dict[int, float]] = {}
    j = 0
    for t, per_slot in motion_samples:
        while j < len(spans) and spans[j][1] < t:
            j += 1
        spk = None
        k = j
        while k < len(spans) and spans[k][0] <= t:
            if spans[k][0] <= t <= spans[k][1]:
                spk = spans[k][2]
                break
            k += 1
        if spk is None:
            continue
        bucket = energy.setdefault(spk, {})
        for slot_idx, e in per_slot.items():
            bucket[slot_idx] = bucket.get(slot_idx, 0.0) + e

    if len(energy) < 2:
        return None

    assignment: dict[int, int] = {}
    for spk, by_slot in energy.items():
        if not by_slot:
            continue
        ranked = sorted(by_slot.items(), key=lambda kv: kv[1], reverse=True)
        top_slot, top_energy = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        if runner_up > 0 and top_energy / runner_up < cfg.min_confidence_ratio:
            continue
        assignment[spk] = top_slot

    if len(assignment) < 2 or len(set(assignment.values())) != len(assignment):
        return None
    return assignment


def _speaker_turns(
    words: Sequence[WordT], keep_ranges: Sequence[tuple[float, float]], cfg: CameraPlanConfig,
) -> list[dict]:
    """Group KEPT words into contiguous same-speaker turns, in edited-timeline
    coordinates, dropping turns shorter than ``min_turn_s``."""
    turns: list[dict] = []
    sentinel = object()
    cur_speaker = sentinel
    cur_start_et: Optional[float] = None
    cur_end_et: Optional[float] = None

    def _flush() -> None:
        if cur_speaker is not sentinel and cur_start_et is not None:
            turns.append({"speaker": cur_speaker, "start_et": cur_start_et, "end_et": cur_end_et})

    for w in sorted(words, key=lambda w: float(w.s)):
        et_start = _edited_time_of(float(w.s), keep_ranges)
        if et_start is None:
            continue
        if w.speaker != cur_speaker:
            _flush()
            cur_speaker, cur_start_et = w.speaker, et_start
        et_end = _edited_time_of(float(w.e), keep_ranges)
        cur_end_et = et_end if et_end is not None else et_start
    _flush()

    return [t for t in turns if (t["end_et"] - t["start_et"]) >= cfg.min_turn_s]


def _anticipated_cut_windows(
    turns: Sequence[dict], anticipate_s: float,
) -> list[tuple[float, float, object]]:
    """Turn a speaker-turn sequence (edited-timeline dicts with
    ``start_et``/``end_et``/``speaker``, e.g. ``_speaker_turns``'s output)
    into non-overlapping ``(start_s, end_s, speaker)`` windows, switching
    ``anticipate_s`` seconds BEFORE the next turn's start rather than
    exactly at the speaker boundary — the visual leads the ear, the J-cut
    instinct a human editor uses (see module docstring).

    Extracted from ``build_camera_plan``'s window-building loop so the
    live-studio real multi-cam auto-director
    (``live_studio.switch_plan.build_switch_plan``) can call the identical
    anticipate-and-clamp arithmetic instead of re-deriving it — only what a
    "window" gets rendered AS (a reframe crop position vs. a physical
    camera index) differs between the two callers, not the timing logic
    itself. Pure and total; no I/O.
    """
    windows: list[tuple[float, float, object]] = []
    for i, turn in enumerate(turns):
        start = max(0.0, turn["start_et"] - anticipate_s)
        if windows and start < windows[-1][1]:
            start = windows[-1][1]
        end = turn["end_et"]
        if i + 1 < len(turns):
            end = max(turn["end_et"], turns[i + 1]["start_et"] - anticipate_s)
        if end <= start:
            continue
        windows.append((start, end, turn["speaker"]))
    return windows


# ── Public API ───────────────────────────────────────────────────────────


def build_camera_plan(
    words: Sequence[WordT],
    keep_ranges: Sequence[tuple[float, float]],
    video_path: str,
    cfg: Optional[CameraPlanConfig] = None,
    *,
    out_aspect: tuple[int, int] = (16, 9),
) -> CameraPlan:
    """Build a virtual multi-cam reframe plan, or an honest no-op.

    Never raises for "couldn't find a confident plan" — only for genuine I/O
    failures (video unreadable). The caller decides how to treat a disengaged
    plan (render.py simply renders the original framing).
    """
    cfg = cfg or CameraPlanConfig()

    if not any(w.speaker is not None for w in words):
        fw, fh = _probe_dims(video_path)
        return _empty_plan(fw, fh, out_aspect, "no diarization (speaker labels absent)")

    frame_w, frame_h, slots, motion_samples = _detect_slots_and_motion(video_path, cfg)
    if len(slots) < 2:
        return _empty_plan(frame_w, frame_h, out_aspect,
                            f"only {len(slots)} face slot(s) detected; need 2 for reframing")

    speaker_to_slot = _assign_slots_to_speakers(motion_samples, words, cfg)
    if speaker_to_slot is None:
        return _empty_plan(frame_w, frame_h, out_aspect,
                            "speaker-to-face correlation was inconclusive")

    turns = [t for t in _speaker_turns(words, keep_ranges, cfg) if t["speaker"] in speaker_to_slot]
    if not turns:
        return _empty_plan(frame_w, frame_h, out_aspect,
                            "no speaker turn met the minimum duration")

    crop_w, crop_h = _reframe_box_size(frame_w, frame_h, out_aspect, cfg.zoom_fraction)
    default_x = (frame_w - crop_w) // 2
    default_y = (frame_h - crop_h) // 2

    def _pos_for_slot(slot_idx: int) -> tuple[int, int]:
        s = slots[slot_idx]
        x = int(max(0, min(s["cx"] - crop_w / 2, frame_w - crop_w)))
        y = int(max(0, min(s["cy"] - crop_h / 2, frame_h - crop_h)))
        return x, y

    raw_windows = _anticipated_cut_windows(turns, cfg.anticipate_s)
    windows: list[ReframeWindow] = []
    for start, end, speaker in raw_windows:
        x, y = _pos_for_slot(speaker_to_slot[speaker])
        windows.append(ReframeWindow(start_s=start, end_s=end, x=x, y=y, speaker=speaker))

    if not windows:
        return _empty_plan(frame_w, frame_h, out_aspect, "no valid reframe windows after clamping")

    plan = CameraPlan(
        windows=tuple(windows), crop_w=crop_w, crop_h=crop_h,
        default_x=default_x, default_y=default_y, engaged=True,
        reason=f"{len(windows)} speaker switch(es) mapped across 2 face slots",
    )
    logger.info("camera plan engaged: %s", plan.reason)
    return plan


__all__ = ["CameraPlanConfig", "ReframeWindow", "CameraPlan", "build_camera_plan"]
