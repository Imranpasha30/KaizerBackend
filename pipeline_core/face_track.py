"""
kaizer.pipeline.face_track
===========================
Face + upper-body tracking with Kalman-filter smoothing for the Kaizer News
video pipeline.

Computes per-frame crop bounding boxes that follow a subject through a video.
The OpenCV Haar face cascade alone suffers from missed detections during
gestures and rapid motion; this module backs it with a 6-state Kalman filter
that predicts box position between detections and applies temporal smoothing
to kill frame-to-frame jitter.

Usage
-----
    from pipeline_core.face_track import compute_crop_trajectory, apply_crop

    result = compute_crop_trajectory(
        "/path/to/source.mp4",
        target_aspect="9:16",
        smooth_window=15,
    )
    if result.warnings:
        logger.warning("Tracking warnings: %s", result.warnings)

    out = apply_crop(
        "/path/to/source.mp4",
        result.trajectory,
        output_path="/path/to/cropped.mp4",
        target_size=(1080, 1920),
    )

TrackingResult fields
---------------------
  trajectory  : list[CropBox]  — One CropBox per source frame.
  detections  : int            — Frames where a face was found by the cascade.
  predictions : int            — Frames where the Kalman filter predicted position.
  warnings    : list[str]      — Non-fatal issues.

CropBox fields
--------------
  x, y : int  — Top-left corner of the crop rectangle in source pixels.
  w, h : int  — Width and height of the crop rectangle in source pixels.

apply_crop (v1 limitation)
--------------------------
  v1 uses a single representative crop box (the median of the trajectory)
  applied uniformly to every frame of the output video.  True per-frame
  variable-crop via extract-crop-reencode is planned for v2.

  TODO (v2): Implement per-frame variable-crop by:
    1. Extracting all frames to individual images.
    2. Cropping each image according to trajectory[frame_idx].
    3. Re-assembling + muxing audio with FFmpeg.
  This is deferred because the FFmpeg crop= filter does not natively support
  a per-frame changing crop box without scripting or zmq/sendcmd, which adds
  significant complexity and latency.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("kaizer.pipeline.face_track")

# ── Haar cascade path (ships with OpenCV) ─────────────────────────────────────

def _find_cascade() -> str:
    """Locate the Haar frontal-face cascade XML packaged with OpenCV."""
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


_CASCADE_PATH: str = _find_cascade()

# ── Aspect-ratio map ──────────────────────────────────────────────────────────

_ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "9:16": (9, 16),
    "1:1":  (1, 1),
    "4:5":  (4, 5),
    "16:9": (16, 9),
}

# ── Kalman filter constants (6-state: x, y, w, h, vx, vy) ────────────────────
# State vector: [cx, cy, w, h, vx, vy]  (cx, cy = centroid of detected box)
# Measurement:  [cx, cy, w, h]

_KALMAN_STATES = 6
_KALMAN_MEASUREMENTS = 4

# ── Public dataclasses ────────────────────────────────────────────────────────

@dataclass
class CropBox:
    """A crop rectangle in source-video pixel coordinates.

    Attributes
    ----------
    x : int   Top-left column.
    y : int   Top-left row.
    w : int   Width in pixels.
    h : int   Height in pixels.
    """

    x: int
    y: int
    w: int
    h: int


@dataclass
class TrackingResult:
    """Result returned by compute_crop_trajectory().

    Attributes
    ----------
    trajectory : list[CropBox]
        One CropBox per frame of the source video, in order.  Always the same
        length as the video's total frame count.
    detections : int
        Number of frames where the Haar cascade found at least one face.
    predictions : int
        Number of frames where the Kalman filter supplied the position (no face
        was detected by the cascade in that frame).
    warnings : list[str]
        Non-fatal issues collected during tracking.
    """

    trajectory: list[CropBox]
    detections: int
    predictions: int
    warnings: list[str] = field(default_factory=list)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_kalman() -> cv2.KalmanFilter:
    """Construct and initialise a 6-state, 4-measurement Kalman filter.

    State vector  : [cx, cy, w, h, vx, vy]
    Measurement   : [cx, cy, w, h]

    Transition model assumes constant velocity: cx += vx * dt, cy += vy * dt.
    dt is set to 1.0 (one frame); scale if you need true-time dynamics.

    Process noise  : small diagonal (1e-3) — trust the motion model.
    Measurement noise : larger diagonal (1e-1) — detector is noisy; let Kalman
                        smooth it.
    """
    kf = cv2.KalmanFilter(_KALMAN_STATES, _KALMAN_MEASUREMENTS)

    dt = 1.0  # one-frame timestep

    # State transition matrix F (6×6)
    kf.transitionMatrix = np.array([
        [1, 0, 0, 0, dt, 0],   # cx' = cx + vx*dt
        [0, 1, 0, 0, 0,  dt],  # cy' = cy + vy*dt
        [0, 0, 1, 0, 0,  0],   # w   = w  (size assumed constant)
        [0, 0, 0, 1, 0,  0],   # h   = h
        [0, 0, 0, 0, 1,  0],   # vx  = vx
        [0, 0, 0, 0, 0,  1],   # vy  = vy
    ], dtype=np.float32)

    # Measurement matrix H (4×6) — we observe cx, cy, w, h only
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
    ], dtype=np.float32)

    # Process noise covariance Q (6×6) — small = trust model
    kf.processNoiseCov = np.eye(_KALMAN_STATES, dtype=np.float32) * 1e-3

    # Measurement noise covariance R (4×4) — larger = detector is noisy
    kf.measurementNoiseCov = np.eye(_KALMAN_MEASUREMENTS, dtype=np.float32) * 1e-1

    # Initial state covariance P (6×6)
    kf.errorCovPost = np.eye(_KALMAN_STATES, dtype=np.float32) * 1.0

    return kf


def _clamp_box(cx: float, cy: float, w: float, h: float,
               frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Clamp centroid + size to frame bounds; return top-left x, y, w, h."""
    w = max(1.0, w)
    h = max(1.0, h)
    # Ensure the box fits inside the frame
    w = min(w, frame_w)
    h = min(h, frame_h)
    x = int(cx - w / 2.0)
    y = int(cy - h / 2.0)
    x = max(0, min(x, frame_w - int(w)))
    y = max(0, min(y, frame_h - int(h)))
    return x, y, int(w), int(h)


def _box_to_target_aspect(
    cx: float,
    cy: float,
    detect_w: float,
    detect_h: float,
    frame_w: int,
    frame_h: int,
    aspect: tuple[int, int],
) -> CropBox:
    """Convert a detected (cx, cy, w, h) into a target-aspect crop box.

    The crop box is centred on (cx, cy) and sized to include the subject with
    padding, constrained by the frame bounds.

    Strategy:
      1. Start from the *detect_h* (which already includes upper-body extension).
      2. Compute the target-aspect width from that height; if it exceeds the
         frame, re-derive from the max frame width instead.
      3. Expand the box by 10% in both dimensions as head-room padding.
      4. Clamp to frame bounds.
    """
    aspect_w, aspect_h = aspect
    ratio = aspect_w / aspect_h  # width/height target ratio

    # Use detect_h as the initial height; add 10% padding
    box_h = detect_h * 1.10
    box_w = box_h * ratio

    # If box_w exceeds frame width, re-derive from frame width
    if box_w > frame_w:
        box_w = float(frame_w)
        box_h = box_w / ratio

    # If box_h exceeds frame height, clamp
    if box_h > frame_h:
        box_h = float(frame_h)
        box_w = box_h * ratio
        box_w = min(box_w, float(frame_w))

    x, y, w, h = _clamp_box(cx, cy, box_w, box_h, frame_w, frame_h)
    return CropBox(x=x, y=y, w=w, h=h)


def _smooth_trajectory(
    trajectory: list[CropBox],
    window: int,
    frame_w: int,
    frame_h: int,
) -> list[CropBox]:
    """Apply a moving-average temporal smoothing to kill per-frame jitter.

    Parameters
    ----------
    trajectory : list[CropBox]
        Raw trajectory with one entry per frame.
    window : int
        Number of frames in the smoothing kernel.  Must be ≥ 1.
    frame_w, frame_h : int
        Source video dimensions for clamping.

    Returns
    -------
    list[CropBox]
        Smoothed trajectory of the same length.
    """
    if window <= 1 or len(trajectory) == 0:
        return trajectory

    n = len(trajectory)
    xs = np.array([b.x for b in trajectory], dtype=np.float32)
    ys = np.array([b.y for b in trajectory], dtype=np.float32)
    ws = np.array([b.w for b in trajectory], dtype=np.float32)
    hs = np.array([b.h for b in trajectory], dtype=np.float32)

    half = window // 2

    def _ma(arr: np.ndarray) -> np.ndarray:
        out = np.empty_like(arr)
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            out[i] = arr[lo:hi].mean()
        return out

    xs_s = _ma(xs)
    ys_s = _ma(ys)
    ws_s = _ma(ws)
    hs_s = _ma(hs)

    smoothed: list[CropBox] = []
    for i in range(n):
        cx = xs_s[i] + ws_s[i] / 2.0
        cy = ys_s[i] + hs_s[i] / 2.0
        x, y, w, h = _clamp_box(cx, cy, float(ws_s[i]), float(hs_s[i]), frame_w, frame_h)
        smoothed.append(CropBox(x=x, y=y, w=w, h=h))

    return smoothed


def _median_box(trajectory: list[CropBox]) -> CropBox:
    """Return the median CropBox across the trajectory (component-wise median).

    Used by apply_crop v1 to pick a single representative box for the whole
    video.
    """
    xs = [b.x for b in trajectory]
    ys = [b.y for b in trajectory]
    ws = [b.w for b in trajectory]
    hs = [b.h for b in trajectory]
    return CropBox(
        x=int(np.median(xs)),
        y=int(np.median(ys)),
        w=int(np.median(ws)),
        h=int(np.median(hs)),
    )


def _centered_fallback_box(
    frame_w: int,
    frame_h: int,
    aspect: tuple[int, int],
) -> CropBox:
    """Return a centred crop at the target aspect when no faces are found."""
    aw, ah = aspect
    ratio = aw / ah
    # Fit within frame
    if frame_w / frame_h >= ratio:
        # Frame is wider than target — constrain by height
        h = frame_h
        w = int(h * ratio)
    else:
        w = frame_w
        h = int(w / ratio)
    x = (frame_w - w) // 2
    y = (frame_h - h) // 2
    return CropBox(x=max(0, x), y=max(0, y), w=min(w, frame_w), h=min(h, frame_h))


def _get_ffmpeg() -> str:
    """Return the FFmpeg binary path."""
    try:
        from pipeline_core.pipeline import FFMPEG_BIN  # type: ignore
        return FFMPEG_BIN
    except Exception:
        import shutil as _sh
        p = _sh.which("ffmpeg")
        return p or "ffmpeg"


# ── Public API ────────────────────────────────────────────────────────────────

def compute_crop_trajectory(
    video_path: str,
    *,
    target_aspect: str = "9:16",
    smooth_window: int = 15,
) -> TrackingResult:
    """Analyse the video frame-by-frame and return a per-frame crop trajectory.

    Algorithm
    ---------
    1. For every frame, run the OpenCV Haar frontal-face cascade.
    2. When a face is found, extend its bounding box downward by the face height
       to capture the upper body (then clamp to frame bounds).
    3. Feed the centroid + size of the body box into a 6-state Kalman filter
       (state: x, y, w, h, vx, vy) as a measurement update.
    4. When no face is detected, use the Kalman filter's prediction step to
       extrapolate the box position.
    5. Convert each predicted/updated centroid + size into a target-aspect crop
       box with subject-centred padding.
    6. Apply a temporal moving-average smoother of `smooth_window` frames to
       kill per-frame jitter.

    Graceful failure: if no faces are detected in the entire video, a centred
    crop at the target aspect is returned for every frame, and a warning is
    logged.

    Parameters
    ----------
    video_path : str
        Absolute path to the source video.
    target_aspect : str
        Desired output aspect ratio: '9:16' (default), '1:1', or '4:5'.
    smooth_window : int
        Number of frames for the temporal smoothing kernel (default 15).

    Returns
    -------
    TrackingResult
        .trajectory has one CropBox per frame.
        .detections counts frames with face found.
        .predictions counts frames using Kalman prediction.
        .warnings lists non-fatal issues.

    Raises
    ------
    ValueError
        If the file does not exist, cannot be opened by VideoCapture, or the
        target_aspect string is not recognised.
    """
    if not os.path.exists(video_path):
        raise ValueError(f"Source video not found: {video_path!r}")

    if target_aspect not in _ASPECT_RATIOS:
        raise ValueError(
            f"Unknown target_aspect {target_aspect!r}. "
            f"Supported: {sorted(_ASPECT_RATIOS.keys())}."
        )
    aspect = _ASPECT_RATIOS[target_aspect]

    warnings: list[str] = []

    # Load cascade
    try:
        cascade = cv2.CascadeClassifier(_CASCADE_PATH)
        if cascade.empty():
            raise RuntimeError(
                f"OpenCV CascadeClassifier could not load {_CASCADE_PATH!r}."
            )
    except Exception as exc:
        raise ValueError(f"Failed to load Haar cascade: {exc}") from exc

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cv2.VideoCapture could not open {video_path!r}.")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_w == 0 or frame_h == 0:
        cap.release()
        raise ValueError(
            f"Could not determine frame dimensions from {video_path!r} "
            f"(got {frame_w}×{frame_h})."
        )

    logger.info(
        "Face tracking: %s  %dx%d  %d frames  target_aspect=%s",
        video_path, frame_w, frame_h, total_frames, target_aspect,
    )

    # Initialise Kalman filter
    kf = _make_kalman()

    # Pre-allocate trajectory (will fill per-frame)
    raw_trajectory: list[CropBox] = []

    detection_count = 0
    prediction_count = 0
    kalman_initialised = False

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Haar cascade face detection
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) > 0:
            # Pick the largest face by area
            areas = [w * h for (x, y, w, h) in faces]
            best = int(np.argmax(areas))
            fx, fy, fw, fh = [int(v) for v in faces[best]]

            # Extend to upper-body: double the face height downward, clamped
            body_h = min(fh * 2, frame_h - fy)
            body_y = fy
            body_x = fx
            body_w = fw

            # Centroid of the body box
            cx = body_x + body_w / 2.0
            cy = body_y + body_h / 2.0

            measurement = np.array([cx, cy, float(body_w), float(body_h)], dtype=np.float32)

            if not kalman_initialised:
                # Seed the Kalman state with the first measurement
                kf.statePre = np.array(
                    [cx, cy, float(body_w), float(body_h), 0.0, 0.0],
                    dtype=np.float32,
                ).reshape((_KALMAN_STATES, 1))
                kf.statePost = kf.statePre.copy()
                kalman_initialised = True

            kf.predict()
            kf.correct(measurement.reshape((_KALMAN_MEASUREMENTS, 1)))

            state = kf.statePost.flatten()
            crop = _box_to_target_aspect(
                state[0], state[1], state[2], state[3],
                frame_w, frame_h, aspect,
            )
            detection_count += 1

        else:
            # No face — use Kalman prediction
            if kalman_initialised:
                pred = kf.predict()
                state = pred.flatten()
                crop = _box_to_target_aspect(
                    state[0], state[1], state[2], state[3],
                    frame_w, frame_h, aspect,
                )
                prediction_count += 1
            else:
                # Kalman not yet initialised (no face seen so far)
                crop = _centered_fallback_box(frame_w, frame_h, aspect)
                prediction_count += 1

        raw_trajectory.append(crop)
        frame_idx += 1

    cap.release()

    if detection_count == 0:
        warnings.append(
            f"No faces detected in any frame of {video_path!r}. "
            "Using centred fallback crop for the entire video."
        )
        fallback = _centered_fallback_box(frame_w, frame_h, aspect)
        raw_trajectory = [fallback] * len(raw_trajectory)

    logger.info(
        "Tracking complete: %d frames, %d detections, %d predictions",
        len(raw_trajectory), detection_count, prediction_count,
    )

    # Temporal smoothing
    smoothed = _smooth_trajectory(raw_trajectory, smooth_window, frame_w, frame_h)

    return TrackingResult(
        trajectory=smoothed,
        detections=detection_count,
        predictions=prediction_count,
        warnings=warnings,
    )


def apply_crop(
    video_path: str,
    trajectory: list[CropBox],
    *,
    output_path: str,
    target_size: tuple[int, int] = (1080, 1920),
) -> str:
    """Render a cropped-and-resized version of the video.

    v1 Implementation (single representative box)
    -----------------------------------------------
    For simplicity and robustness, v1 computes the MEDIAN crop box across the
    entire trajectory and applies that single static crop to every frame.  This
    avoids the complexity of per-frame variable-crop (see TODO below) while
    still placing the subject correctly for the majority of the video.

    TODO (v2 — per-frame variable crop):
        Implement true per-frame crop by:
          1. Extracting all frames to individual image files.
          2. Applying trajectory[frame_idx] crop + resize to each image.
          3. Re-encoding the image sequence back to video and muxing with the
             original audio track.
        The FFmpeg crop= filter does not support frame-varying parameters
        without zmq/sendcmd scripting, so the extract→process→reassemble
        approach is the practical v2 path.

    Parameters
    ----------
    video_path : str
        Absolute path to the source video.
    trajectory : list[CropBox]
        Per-frame crop boxes as returned by compute_crop_trajectory().
    output_path : str
        Destination path for the cropped output.
    target_size : tuple[int, int]
        (width, height) of the output video in pixels.  Default: 1080×1920.

    Returns
    -------
    str
        Absolute path to the output file (same as output_path).

    Raises
    ------
    ValueError
        If trajectory is empty or the source file does not exist.
    """
    if not os.path.exists(video_path):
        raise ValueError(f"Source video not found: {video_path!r}")
    if not trajectory:
        raise ValueError("trajectory is empty; cannot apply crop.")

    out_w, out_h = target_size

    # Compute the representative (median) crop box — v1 simplification.
    median = _median_box(trajectory)

    logger.info(
        "apply_crop (v1): median box x=%d y=%d w=%d h=%d → resize to %dx%d",
        median.x, median.y, median.w, median.h, out_w, out_h,
    )

    ffmpeg = _get_ffmpeg()

    # Build FFmpeg crop + scale filter chain.
    # crop=w:h:x:y then scale to target output size.
    vf = (
        f"crop={median.w}:{median.h}:{median.x}:{median.y},"
        f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
        f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2"
    )

    cmd = [
        ffmpeg, "-y",
        "-i", video_path,
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",   # preserve source audio unchanged
        output_path,
    ]

    logger.debug("Running apply_crop: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(
            f"FFmpeg crop render failed for {video_path!r}: {proc.stderr.strip()}"
        )

    logger.info("apply_crop complete → %s", output_path)
    return output_path
