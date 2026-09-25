"""Auto focal-point detection for pool/canvas images (face-aware framing).

``focal_point(path) -> (offset_x_pct, offset_y_pct)`` for the CanvasImage
cover-crop fields: 50/50 = centered (the legacy behaviour), values are
clamped 15..85 so the crop never slams to an edge (gentle framing — the
editor's 9-point focal grid can always override). Detection = cv2 Haar
(frontal + profile), the same proven stack as pipeline_core/thumbnails.py
— deliberately re-implemented here instead of imported: thumbnails.py
pays numpy/cv2 at module import and is video-frame oriented, while this
module must stay import-safe inside uvicorn workers and on GPU-less hosts.

Fail-soft contract: ANY problem (cv2 missing, unreadable file, no face)
returns (50.0, 50.0) — a face-less image simply keeps today's centered
crop, and detection can never block a render.

Kill switch: KAIZER_V4_FACE_FOCUS=0 (default ON).

Cache-safety note (the reason callers stamp results conditionally): the
offsets are ALREADY part of v1_bridge's per-story cache hash, so writing a
non-default value correctly forks the cache, while leaving the field
absent keeps every pre-existing canvas hash-identical (100% cache hits).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

_DEFAULT = (50.0, 50.0)
# Never pin the crop to an image edge: a slightly-off face reads better
# than a hard edge-ride, and Haar false positives stay low-impact.
_CLAMP_LO, _CLAMP_HI = 15.0, 85.0
_MAX_SIDE = 640          # downscale before detect — Haar cost is O(pixels)
_CACHE_CAP = 4096        # naive guard: clear() when exceeded (long daemons)
_cache: dict = {}        # (str(path), mtime) -> (ox, oy); mtime key means a
                         # replaced pool image re-detects automatically
_lock = threading.Lock()
_cascades = None         # lazy: None = not loaded, False = load failed,
                         # else [frontal, profile]


def enabled() -> bool:
    """KAIZER_V4_FACE_FOCUS kill switch — default ON."""
    return (os.environ.get("KAIZER_V4_FACE_FOCUS", "1") or "1").strip().lower() \
        not in ("0", "false", "off", "no")


def _load_cascades():
    """Lazy, thread-safe, once. cv2 is imported HERE (never at module top)
    so uvicorn request handlers and cv2-less hosts never pay/crash on
    import of this module."""
    global _cascades
    if _cascades is not None:
        return _cascades
    with _lock:
        if _cascades is not None:
            return _cascades
        try:
            import cv2
            frontal = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            profile = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_profileface.xml")
            _cascades = (False if (frontal.empty() or profile.empty())
                         else [frontal, profile])
        except Exception:
            _cascades = False
    return _cascades


def _detect(gray) -> list:
    """Face boxes [(x, y, w, h), ...] on a grayscale frame. Frontal first
    (thumbnails.py's proven params); only when it finds nothing, try the
    profile cascade — which is ONE-SIDED, so run it on the mirror too and
    map the mirrored x back (x' = W - x - w)."""
    import cv2
    cas = _load_cascades()
    if not cas:
        return []
    frontal, profile = cas
    kw = dict(scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
              flags=cv2.CASCADE_SCALE_IMAGE)
    faces = [tuple(int(v) for v in f)
             for f in frontal.detectMultiScale(gray, **kw)]
    if faces:
        return faces
    faces = [tuple(int(v) for v in f)
             for f in profile.detectMultiScale(gray, **kw)]
    w_img = int(gray.shape[1])
    for (x, y, fw, fh) in profile.detectMultiScale(cv2.flip(gray, 1), **kw):
        faces.append((w_img - int(x) - int(fw), int(y), int(fw), int(fh)))
    return faces


def _group_focal(faces, w: int, h: int) -> tuple[float, float]:
    """Face boxes -> clamped (offset_x_pct, offset_y_pct).

    Faces within 60% of the biggest face's area form the "subject group";
    the focal point is their area-weighted mean center — a single face is
    its own center, a group shot lands between the heads (the same
    largest/most-centred spirit as thumbnails.py's _face_score). Tiny
    background faces are excluded so a crowd behind the anchor can't drag
    the crop away."""
    boxes = [(float(x), float(y), float(fw), float(fh))
             for (x, y, fw, fh) in (faces or []) if fw > 0 and fh > 0]
    if not boxes or w <= 0 or h <= 0:
        return _DEFAULT
    max_area = max(fw * fh for (_x, _y, fw, fh) in boxes)
    grp = [b for b in boxes if b[2] * b[3] >= 0.6 * max_area]
    tot = sum(fw * fh for (_x, _y, fw, fh) in grp)
    cx = sum((x + fw / 2.0) * (fw * fh) for (x, y, fw, fh) in grp) / tot
    cy = sum((y + fh / 2.0) * (fw * fh) for (x, y, fw, fh) in grp) / tot
    ox = round(min(_CLAMP_HI, max(_CLAMP_LO, cx / w * 100.0)), 1)
    oy = round(min(_CLAMP_HI, max(_CLAMP_LO, cy / h * 100.0)), 1)
    return (ox, oy)


def face_boxes_pct(image_path) -> list:
    """Face boxes as ``[(x_pct, y_pct, w_pct, h_pct), …]`` of the frame —
    ``[]`` on any failure / no faces / kill switch. Same detection stack
    and fail-soft contract as ``focal_point``; used by the layout
    panel-safety pass (keep OTS picture boxes off people's faces).
    Cached per (path, mtime)."""
    if not enabled():
        return []
    try:
        p = Path(image_path)
        key = ("boxes", str(p), p.stat().st_mtime)
    except OSError:
        return []
    hit = _cache.get(key)
    if hit is not None:
        return hit
    res: list = []
    try:
        import cv2
        img = cv2.imread(str(p))
        if img is not None:
            ih, iw = img.shape[:2]
            m = max(ih, iw)
            if m > _MAX_SIDE:
                sc = _MAX_SIDE / float(m)
                img = cv2.resize(img, (max(1, int(iw * sc)), max(1, int(ih * sc))),
                                 interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            gh, gw = gray.shape[:2]
            if gw > 0 and gh > 0:
                res = [
                    (round(x / gw * 100.0, 2), round(y / gh * 100.0, 2),
                     round(fw / gw * 100.0, 2), round(fh / gh * 100.0, 2))
                    for (x, y, fw, fh) in _detect(gray)
                    if fw > 0 and fh > 0
                ]
    except Exception:
        res = []
    with _lock:
        if len(_cache) > _CACHE_CAP:
            _cache.clear()
        _cache[key] = res
    return res


def focal_point(image_path) -> tuple[float, float]:
    """(offset_x_pct, offset_y_pct) for ``image_path`` — (50.0, 50.0) on
    any failure or when no face is found. Pure read-only on the file, so
    it is safe from both the orchestrator subprocess and uvicorn request
    handlers; results are cached per (path, mtime)."""
    if not enabled():
        return _DEFAULT
    try:
        p = Path(image_path)
        key = (str(p), p.stat().st_mtime)
    except OSError:
        return _DEFAULT
    hit = _cache.get(key)
    if hit is not None:
        return hit
    res = _DEFAULT
    try:
        import cv2
        img = cv2.imread(str(p))
        if img is not None:
            ih, iw = img.shape[:2]
            m = max(ih, iw)
            if m > _MAX_SIDE:
                # INTER_AREA downscale: Haar speed; the focal point is a
                # percentage, so working at reduced scale loses nothing.
                sc = _MAX_SIDE / float(m)
                img = cv2.resize(img, (max(1, int(iw * sc)), max(1, int(ih * sc))),
                                 interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)     # low-contrast news photos
            faces = _detect(gray)
            if faces:
                gh, gw = gray.shape[:2]
                res = _group_focal(faces, gw, gh)
    except Exception:
        res = _DEFAULT
    with _lock:
        if len(_cache) > _CACHE_CAP:
            _cache.clear()
        _cache[key] = res
    return res
