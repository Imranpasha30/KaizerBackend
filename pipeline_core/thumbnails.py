"""
kaizer.pipeline.thumbnails
===========================
Smart thumbnail candidate generator for the Kaizer News video pipeline.

Generates up to 3 candidate thumbnails per video clip, each representing a
different visual strategy, so the user (or a downstream ranker) can pick the
best one.  Closes the OpusClip 73-upvote per-clip thumbnail selection gap.

Usage
-----
    from pipeline_core.thumbnails import generate_thumbnails, ThumbnailCandidate

    candidates = generate_thumbnails(
        "/path/to/clip.mp4",
        title="Hyderabad Metro Expansion Approved",
        output_dir="/tmp/thumbs",
        target_size=(1080, 1920),
        candidates=3,
    )
    best = candidates[0]   # sorted by score descending
    print(best.path, best.kind, best.score)

ThumbnailCandidate fields
--------------------------
  path             : str    — Absolute path to the generated PNG file.
  kind             : str    — 'face_lock' | 'quote_card' | 'punch_frame'
  score            : float  — 0.0–1.0 internal quality heuristic.
  source_frame_t   : float  — Seconds into the video where the frame was taken.
  meta             : dict   — Kind-specific data:
                              face_lock  → face_bbox, face_size_norm, centerness
                              quote_card → histogram_entropy
                              punch_frame→ motion_energy

Candidate strategies
---------------------
  face_lock   — Frame with the largest, most centred face detected by OpenCV
                Haar cascade.  Close-cropped to target_size with a subtle
                vignette.
  quote_card  — Frame with highest visual diversity (3-D histogram entropy).
                Title text overlaid in lower third using captions.render_caption.
  punch_frame — Frame with highest inter-frame motion energy (sum-of-absolute
                differences between consecutive grey frames).

All 3 PNGs are written to *output_dir* as ``thumb_{kind}_{index}.png`` where
*index* is the candidate rank (0-based, sorted by score descending).

Temporary frames (extracted via ffmpeg at 1 fps) are always cleaned up in a
``finally:`` block regardless of errors.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from pipeline_core.captions import render_caption

logger = logging.getLogger("kaizer.pipeline.thumbnails")

# ── Constants ─────────────────────────────────────────────────────────────────

# Haar cascade for frontal face detection (bundled with opencv-python-headless)
_FACE_CASCADE_PATH: str = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Histogram bin count per channel for visual-diversity scoring
_HIST_BINS: int = 8

# Caption font size for quote-card thumbnails
_QUOTE_FONT_SIZE: int = 56

# Lower-third band: text starts at this fraction of the frame height from top
_LOWER_THIRD_START: float = 0.62

# Vignette strength for face-lock (0 = none, 1 = full black edges)
_VIGNETTE_STRENGTH: float = 0.5

# ffmpeg binary resolution (mirrors pipeline.py / qa.py pattern)

def _find_ffmpeg() -> str:
    """Find the ffmpeg binary (PATH first, then pipeline.py's FFMPEG_BIN, then
    common Unix paths)."""
    import shutil as _sh

    p = _sh.which("ffmpeg")
    if p:
        return p

    try:
        from pipeline_core.pipeline import FFMPEG_BIN as _ff  # type: ignore
        if os.path.isfile(_ff) and os.access(_ff, os.X_OK):
            return _ff
    except Exception:
        pass

    for prefix in [
        "/usr/bin",
        "/usr/local/bin",
        "/nix/var/nix/profiles/default/bin",
        "/run/current-system/sw/bin",
    ]:
        candidate = os.path.join(prefix, "ffmpeg")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    return "ffmpeg"


FFMPEG_BIN: str = _find_ffmpeg()


# ── Public dataclass ──────────────────────────────────────────────────────────

@dataclass
class ThumbnailCandidate:
    """A single thumbnail candidate.

    Attributes
    ----------
    path : str
        Absolute path to the generated PNG file.
    kind : str
        Strategy used: ``'face_lock'``, ``'quote_card'``, or ``'punch_frame'``.
    score : float
        Internal quality heuristic in [0.0, 1.0].  Higher is better.
    source_frame_t : float
        Timestamp (seconds) of the source video frame.
    meta : dict
        Kind-specific metadata (face bbox, motion energy, etc.).
    """

    path: str
    kind: str
    score: float
    source_frame_t: float
    meta: dict = field(default_factory=dict)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_frames(video_path: str, out_dir: str) -> list[tuple[float, str]]:
    """Extract 1 frame per second from *video_path* into *out_dir*.

    Returns
    -------
    list of (timestamp_seconds, absolute_frame_path) sorted by timestamp.

    Raises
    ------
    RuntimeError
        If ffmpeg fails or produces no frames.
    """
    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-loglevel", "error",
        "-i", video_path,
        "-vf", "fps=1",
        "-q:v", "2",
        os.path.join(out_dir, "frame_%04d.jpg"),
    ]
    logger.debug("Extracting frames: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffmpeg timed out extracting frames from {video_path!r}"
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"ffmpeg binary not found at {FFMPEG_BIN!r}. "
            "Ensure ffmpeg is installed and on PATH."
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}) extracting frames from "
            f"{video_path!r}: {result.stderr.strip()[-500:]}"
        )

    # Collect frame files; frame_%04d.jpg starts at 1
    frames: list[tuple[float, str]] = []
    for entry in sorted(os.listdir(out_dir)):
        if not entry.startswith("frame_") or not entry.endswith(".jpg"):
            continue
        # frame_0001.jpg → t=0.5s (midpoint of the 1-s window), frame_0002.jpg → t=1.5s …
        try:
            idx = int(entry[len("frame_"):-4])
        except ValueError:
            continue
        t = float(idx) - 0.5  # frame N covers second [N-1, N); midpoint = N-0.5
        abs_path = os.path.join(out_dir, entry)
        frames.append((t, abs_path))

    if not frames:
        raise RuntimeError(
            f"ffmpeg ran but produced no frame files in {out_dir!r} "
            f"for video {video_path!r}."
        )

    logger.debug("Extracted %d frames from %s", len(frames), video_path)
    return frames


def _load_bgr(path: str) -> Optional[np.ndarray]:
    """Load *path* as a BGR numpy array, or None on error."""
    frame = cv2.imread(path)
    if frame is None:
        logger.warning("cv2.imread returned None for %s", path)
    return frame


def _resize_crop(frame_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize *frame_bgr* to exactly (*target_w*, *target_h*) using
    centre-crop to preserve the source aspect ratio."""
    src_h, src_w = frame_bgr.shape[:2]
    src_ratio = src_w / src_h
    tgt_ratio = target_w / target_h

    if src_ratio > tgt_ratio:
        # Source is wider — match height, then crop width
        new_h = target_h
        new_w = int(src_w * target_h / src_h)
    else:
        # Source is taller — match width, then crop height
        new_w = target_w
        new_h = int(src_h * target_w / src_w)

    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Centre crop
    cx = (new_w - target_w) // 2
    cy = (new_h - target_h) // 2
    cropped = resized[cy: cy + target_h, cx: cx + target_w]

    # Guard: if rounding left us short, pad with black
    if cropped.shape[:2] != (target_h, target_w):
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        ch = min(cropped.shape[0], target_h)
        cw = min(cropped.shape[1], target_w)
        padded[:ch, :cw] = cropped[:ch, :cw]
        return padded

    return cropped


def _apply_vignette(frame_bgr: np.ndarray, strength: float = _VIGNETTE_STRENGTH) -> np.ndarray:
    """Apply a radial vignette to *frame_bgr* (in-place copy).

    *strength* = 0 → no vignette; 1 → edges fully black.
    """
    h, w = frame_bgr.shape[:2]
    # Build a Gaussian kernel that's bright in the centre, dark at edges
    sigma_x = w * 0.55
    sigma_y = h * 0.55
    x = np.linspace(-w / 2, w / 2, w)
    y = np.linspace(-h / 2, h / 2, h)
    xv, yv = np.meshgrid(x, y)
    kernel = np.exp(-(xv ** 2 / (2 * sigma_x ** 2) + yv ** 2 / (2 * sigma_y ** 2)))
    # Blend: 1 = original, 0 = black (according to strength)
    vignette = 1.0 - strength * (1.0 - kernel)
    vignette = np.clip(vignette, 0, 1).astype(np.float32)
    out = frame_bgr.astype(np.float32)
    for c in range(3):
        out[:, :, c] *= vignette
    return np.clip(out, 0, 255).astype(np.uint8)


def _histogram_entropy(frame_bgr: np.ndarray) -> float:
    """Compute Shannon entropy of the 3-D colour histogram of *frame_bgr*.

    Higher entropy → more visually diverse / interesting frame.
    """
    hist = cv2.calcHist(
        [frame_bgr],
        [0, 1, 2],
        None,
        [_HIST_BINS, _HIST_BINS, _HIST_BINS],
        [0, 256, 0, 256, 0, 256],
    )
    hist_flat = hist.flatten().astype(np.float64)
    total = hist_flat.sum()
    if total <= 0:
        return 0.0
    p = hist_flat / total
    # Manual Shannon entropy (avoids scipy dependency for this computation,
    # but scipy.stats.entropy is also available)
    p_nonzero = p[p > 0]
    entropy = float(-np.sum(p_nonzero * np.log(p_nonzero)))
    return entropy


def _motion_energy(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    """Return the normalised sum-of-absolute-differences between two grey frames."""
    diff = cv2.absdiff(gray_a, gray_b)
    return float(diff.sum()) / max(1, gray_a.size)


def _detect_faces(
    frame_bgr: np.ndarray,
    cascade: cv2.CascadeClassifier,
) -> list[tuple[int, int, int, int]]:
    """Detect faces in *frame_bgr* and return a list of (x, y, w, h) bounding boxes."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return []
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def _face_score(
    faces: list[tuple[int, int, int, int]],
    frame_w: int,
    frame_h: int,
) -> tuple[float, tuple[int, int, int, int]]:
    """Score faces and return (best_score, best_face_bbox).

    Score = face_area_norm × centerness, where centerness is 1 when the face
    centre is exactly at the frame centre and 0 at the corners.
    """
    best_score = 0.0
    best_face = (0, 0, 0, 0)
    frame_cx = frame_w / 2
    frame_cy = frame_h / 2
    max_dist = math.hypot(frame_cx, frame_cy)

    for x, y, w, h in faces:
        face_cx = x + w / 2
        face_cy = y + h / 2
        dist = math.hypot(face_cx - frame_cx, face_cy - frame_cy)
        centerness = 1.0 - (dist / max_dist) if max_dist > 0 else 1.0
        area_norm = (w * h) / (frame_w * frame_h)
        score = area_norm * centerness
        if score > best_score:
            best_score = score
            best_face = (x, y, w, h)

    return best_score, best_face


def _crop_around_face(
    frame_bgr: np.ndarray,
    face_bbox: tuple[int, int, int, int],
    target_w: int,
    target_h: int,
) -> np.ndarray:
    """Centre-crop *frame_bgr* around *face_bbox* to (target_w, target_h).

    Expands the crop region from the face centre outward while staying within
    frame bounds, then falls back to _resize_crop if needed.
    """
    fh, fw = frame_bgr.shape[:2]
    fx, fy, fow, foh = face_bbox  # face origin and size

    face_cx = fx + fow // 2
    face_cy = fy + foh // 2

    # Desired crop region centred on the face
    half_w = target_w // 2
    half_h = target_h // 2

    x1 = max(0, face_cx - half_w)
    y1 = max(0, face_cy - half_h)
    x2 = x1 + target_w
    y2 = y1 + target_h

    # Clamp to frame bounds
    if x2 > fw:
        x1 = max(0, fw - target_w)
        x2 = fw
    if y2 > fh:
        y1 = max(0, fh - target_h)
        y2 = fh

    crop = frame_bgr[y1:y2, x1:x2]

    # If crop is still smaller than target (frame smaller than target_size),
    # resize-crop the full frame
    if crop.shape[0] < target_h or crop.shape[1] < target_w:
        return _resize_crop(frame_bgr, target_w, target_h)

    return crop


def _overlay_caption(
    frame_bgr: np.ndarray,
    title: str,
    target_w: int,
    target_h: int,
) -> np.ndarray:
    """Overlay the title caption in the lower-third of *frame_bgr*.

    Returns a BGR numpy array of the same size as the input.
    """
    caption_max_w = int(target_w * 0.85)

    caption_result = render_caption(
        title,
        max_width=caption_max_w,
        font_size=_QUOTE_FONT_SIZE,
        color="#FFFFFF",
        stroke_color="#000000",
        stroke_width=3,
        bg_color="#00000099",
        bg_padding=20,
        bg_radius=14,
        align="center",
    )

    # Convert BGR frame to PIL RGBA
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    base = Image.fromarray(frame_rgb).convert("RGBA")

    cap_img = caption_result.image  # already RGBA

    # Position: horizontally centred, lower-third start
    cap_x = (target_w - cap_img.width) // 2
    lower_third_y = int(target_h * _LOWER_THIRD_START)
    cap_y = min(lower_third_y, target_h - cap_img.height - 10)

    # Composite caption onto base
    base.alpha_composite(cap_img, dest=(cap_x, cap_y))

    # Convert back to BGR
    result_rgb = np.array(base.convert("RGB"))
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)


def _save_thumbnail(
    frame_bgr: np.ndarray,
    output_path: str,
    target_size: tuple[int, int],
) -> None:
    """Resize-crop *frame_bgr* to *target_size* and save as PNG to *output_path*."""
    tw, th = target_size
    final = _resize_crop(frame_bgr, tw, th)
    cv2.imwrite(output_path, final)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_thumbnails(
    video_path: str,
    *,
    title: str,
    output_dir: str,
    target_size: tuple[int, int] = (1080, 1920),
    candidates: int = 3,
) -> list[ThumbnailCandidate]:
    """Generate up to *candidates* thumbnail candidates from *video_path*.

    Parameters
    ----------
    video_path : str
        Absolute path to the input video file.
    title : str
        Title text to overlay on the quote-card thumbnail.
    output_dir : str
        Directory where the generated PNG files are written.  Must exist or
        be creatable by this function.
    target_size : (width, height)
        Target pixel dimensions for the output thumbnails.  Default: (1080, 1920).
    candidates : int
        Number of thumbnails to generate (max 3; each kind is generated at most
        once).  Default: 3.

    Returns
    -------
    list[ThumbnailCandidate]
        Up to *candidates* candidates sorted by *score* descending.  The list
        may be shorter than *candidates* if the video is too short or frames
        cannot be decoded.

    Raises
    ------
    FileNotFoundError
        If *video_path* does not exist.
    RuntimeError
        If ffmpeg fails to extract any frames.

    Notes
    -----
    - Temporary frames are always deleted in a ``finally:`` block.
    - Output PNGs are written to *output_dir* as ``thumb_{kind}_{rank}.png``.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path!r}")

    os.makedirs(output_dir, exist_ok=True)

    target_w, target_h = target_size
    tmp_dir = tempfile.mkdtemp(prefix="kaizer_frames_")

    try:
        # ── Step 1: Extract frames (1 fps) ───────────────────────────────────
        frames = _extract_frames(video_path, tmp_dir)
        n_frames = len(frames)
        logger.info("Generating thumbnails for %s (%d frames)", video_path, n_frames)

        # Load frames as BGR arrays (skip unreadable ones)
        loaded: list[tuple[float, np.ndarray]] = []
        for t, fpath in frames:
            bgr = _load_bgr(fpath)
            if bgr is not None:
                loaded.append((t, bgr))

        if not loaded:
            raise RuntimeError(
                f"Could not load any frames from {video_path!r} — "
                "all cv2.imread() calls returned None."
            )

        # Always attempt all 3 kinds; slicing to `candidates` happens AFTER
        # sorting by score so the user always sees the best subset, not the
        # first-N-computed subset. This guarantees `candidates=1` returns the
        # single best candidate across all strategies, not just face_lock.
        results: list[ThumbnailCandidate] = []
        requested_kinds = ["face_lock", "quote_card", "punch_frame"]

        # ── Step 2: Face-lock candidate ───────────────────────────────────────
        if "face_lock" in requested_kinds:
            try:
                cascade = cv2.CascadeClassifier(_FACE_CASCADE_PATH)
                if cascade.empty():
                    raise RuntimeError(
                        f"Haar cascade empty — could not load {_FACE_CASCADE_PATH!r}"
                    )

                best_face_score = -1.0
                best_face_t = loaded[0][0]
                best_face_frame: Optional[np.ndarray] = None
                best_face_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
                best_centerness = 0.0

                for t, bgr in loaded:
                    fh, fw = bgr.shape[:2]
                    faces = _detect_faces(bgr, cascade)
                    if not faces:
                        continue
                    score, bbox = _face_score(faces, fw, fh)
                    if score > best_face_score:
                        best_face_score = score
                        best_face_t = t
                        best_face_frame = bgr.copy()
                        best_face_bbox = bbox
                        fx, fy, bw, bh = bbox
                        best_centerness = 1.0 - (
                            math.hypot(fx + bw / 2 - fw / 2, fy + bh / 2 - fh / 2)
                            / math.hypot(fw / 2, fh / 2)
                        )

                if best_face_frame is not None and best_face_score > 0:
                    face_cropped = _crop_around_face(
                        best_face_frame, best_face_bbox, target_w, target_h
                    )
                    vignetted = _apply_vignette(face_cropped)
                    out_path = os.path.join(output_dir, "thumb_face_lock_0.png").replace("\\", "/")
                    _save_thumbnail(vignetted, out_path, target_size)

                    fh_src, fw_src = best_face_frame.shape[:2]
                    fx, fy, fw_bb, fh_bb = best_face_bbox
                    normalised_score = min(1.0, best_face_score * 10)

                    results.append(
                        ThumbnailCandidate(
                            path=out_path,
                            kind="face_lock",
                            score=float(normalised_score),
                            source_frame_t=best_face_t,
                            meta={
                                "face_bbox": list(best_face_bbox),
                                "face_size_norm": float(fw_bb * fh_bb) / float(fw_src * fh_src),
                                "centerness": float(best_centerness),
                            },
                        )
                    )
                    logger.info(
                        "face_lock: t=%.1fs score=%.3f bbox=%s",
                        best_face_t, normalised_score, best_face_bbox,
                    )
                else:
                    # Fallback: no face detected in any frame. Emit a "face_lock"
                    # candidate using a rule-of-thirds centre crop of the
                    # first frame, with a low score so it sorts last. This
                    # preserves the UX contract that the user always gets 3
                    # candidates to pick from even on faceless footage.
                    logger.info(
                        "face_lock: no faces detected in %d frames — "
                        "emitting centre-crop fallback", n_frames,
                    )
                    fallback_t, fallback_bgr = loaded[0]
                    fallback_crop = _resize_crop(fallback_bgr, target_w, target_h)
                    fallback_vignette = _apply_vignette(fallback_crop)
                    out_path = os.path.join(
                        output_dir, "thumb_face_lock_0.png"
                    ).replace("\\", "/")
                    _save_thumbnail(fallback_vignette, out_path, target_size)
                    results.append(
                        ThumbnailCandidate(
                            path=out_path,
                            kind="face_lock",
                            score=0.05,  # low — user will usually pick another
                            source_frame_t=fallback_t,
                            meta={
                                "face_bbox": None,
                                "face_size_norm": 0.0,
                                "centerness": 0.0,
                                "fallback": "no_face_detected_center_crop",
                            },
                        )
                    )
            except Exception as exc:
                logger.warning("face_lock candidate failed: %s", exc)

        # ── Step 3: Quote-card candidate ──────────────────────────────────────
        if "quote_card" in requested_kinds:
            try:
                best_entropy = -1.0
                best_entropy_t = loaded[0][0]
                best_entropy_frame: Optional[np.ndarray] = None

                for t, bgr in loaded:
                    ent = _histogram_entropy(bgr)
                    if ent > best_entropy:
                        best_entropy = ent
                        best_entropy_t = t
                        best_entropy_frame = bgr.copy()

                if best_entropy_frame is not None:
                    # Resize frame to target size first
                    base_resized = _resize_crop(best_entropy_frame, target_w, target_h)
                    # Overlay caption
                    captioned = _overlay_caption(base_resized, title, target_w, target_h)
                    out_path = os.path.join(output_dir, "thumb_quote_card_0.png").replace("\\", "/")
                    cv2.imwrite(out_path, captioned)

                    # Normalise entropy to 0-1 range (max possible for 8^3 bins ≈ ln(512) ≈ 6.24)
                    max_entropy = math.log(_HIST_BINS ** 3)
                    norm_score = min(1.0, best_entropy / max(max_entropy, 1e-6))

                    results.append(
                        ThumbnailCandidate(
                            path=out_path,
                            kind="quote_card",
                            score=float(norm_score),
                            source_frame_t=best_entropy_t,
                            meta={"histogram_entropy": float(best_entropy)},
                        )
                    )
                    logger.info(
                        "quote_card: t=%.1fs entropy=%.3f score=%.3f",
                        best_entropy_t, best_entropy, norm_score,
                    )
            except Exception as exc:
                logger.warning("quote_card candidate failed: %s", exc)

        # ── Step 4: Punch-frame candidate ─────────────────────────────────────
        if "punch_frame" in requested_kinds:
            try:
                if len(loaded) < 2:
                    logger.info("punch_frame: fewer than 2 frames — skipping")
                else:
                    best_motion = -1.0
                    best_motion_t = loaded[1][0]
                    best_motion_frame: Optional[np.ndarray] = None

                    prev_gray = cv2.cvtColor(loaded[0][1], cv2.COLOR_BGR2GRAY)

                    for t, bgr in loaded[1:]:
                        curr_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                        energy = _motion_energy(prev_gray, curr_gray)
                        if energy > best_motion:
                            best_motion = energy
                            best_motion_t = t
                            best_motion_frame = bgr.copy()
                        prev_gray = curr_gray

                    if best_motion_frame is not None:
                        out_path = os.path.join(
                            output_dir, "thumb_punch_frame_0.png"
                        ).replace("\\", "/")
                        _save_thumbnail(best_motion_frame, out_path, target_size)

                        # Normalise: 40 SAD/pixel ≈ near-scene-cut level → 1.0
                        norm_score = min(1.0, best_motion / 40.0)

                        results.append(
                            ThumbnailCandidate(
                                path=out_path,
                                kind="punch_frame",
                                score=float(norm_score),
                                source_frame_t=best_motion_t,
                                meta={"motion_energy": float(best_motion)},
                            )
                        )
                        logger.info(
                            "punch_frame: t=%.1fs motion_energy=%.3f score=%.3f",
                            best_motion_t, best_motion, norm_score,
                        )
            except Exception as exc:
                logger.warning("punch_frame candidate failed: %s", exc)

    finally:
        # ── Cleanup temp frames ───────────────────────────────────────────────
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.debug("Cleaned up temp dir %s", tmp_dir)
        except Exception as exc:
            logger.warning("Could not clean up temp dir %s: %s", tmp_dir, exc)

    # Sort by score descending, return up to `candidates` items
    results.sort(key=lambda c: c.score, reverse=True)
    final = results[:candidates]

    # Rename output files to reflect final rank
    for rank, candidate in enumerate(final):
        kind = candidate.kind
        new_path = os.path.join(output_dir, f"thumb_{kind}_{rank}.png").replace("\\", "/")
        if candidate.path != new_path and os.path.isfile(candidate.path):
            try:
                os.replace(candidate.path, new_path)
                candidate.path = new_path
            except OSError as exc:
                logger.warning(
                    "Could not rename %s → %s: %s", candidate.path, new_path, exc
                )

    logger.info(
        "generate_thumbnails: %d/%d candidates generated for %s",
        len(final), candidates, video_path,
    )
    return final
