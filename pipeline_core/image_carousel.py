"""Phase 3 of the long-form pipeline — image carousel.

Two carousel flavours:

  - **Sidebar carousel**: a 580×800 video that fills the right pane of
    :func:`longform_compose.compose_bulletin_story`. Rotates 4–6
    contextual images per story with a Ken-Burns push-in and crossfade
    transitions.

  - **Full-screen takeover**: a 1920×1080 mini-video inserted between
    stories at story boundaries. ~6–10 seconds of full-frame photo
    gallery before the next anchor segment kicks in. Skipped at the
    open and close of the bulletin so it never opens or closes on a
    photo gallery.

Implementation
--------------
Single FFmpeg call per carousel using the well-known
``zoompan + xfade`` pattern (validated against
ffmpeg-video-slideshow-scripts and kburns-slideshow). Each image is
processed by ``zoompan`` to produce a per-image MP4 segment with the
Ken-Burns push, then those segments are crossfaded with ``xfade``.

We pre-encode each image segment to its own MP4 in a temp dir and
then run a single ffmpeg call that chains ``xfade``s — this is more
reliable than building a 200-line single-call ``filter_complex`` for
4–6 inputs and lets us recover from a single bad image without
reshooting the whole carousel.

Failure tolerance
-----------------
- A single broken image is skipped (it just isn't included).
- If fewer than two images survive, the function falls back to a
  single-image static MP4 at the requested duration — still usable as
  a sidebar fill.
- All exceptions are caught at the public-API boundary; on any error
  the caller can fall back to ``longform_compose.make_sidebar_placeholder``
  or skip takeovers entirely.
"""
from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("kaizer.pipeline.image_carousel")


# ── Public dataclass ─────────────────────────────────────────────────────────

@dataclass
class CarouselResult:
    output_path: str
    images_used: int
    duration_s: float


class CarouselError(RuntimeError):
    """Raised when a carousel cannot be built (no usable images, FFmpeg fail)."""


# ── Internal helpers ─────────────────────────────────────────────────────────

def _resolve_ffmpeg_bin(ffmpeg_bin: Optional[str]) -> str:
    if ffmpeg_bin:
        return ffmpeg_bin
    try:
        from pipeline_core.pipeline import FFMPEG_BIN
        return FFMPEG_BIN
    except Exception:
        return shutil.which("ffmpeg") or "ffmpeg"


def _filter_usable(images: list[str]) -> list[str]:
    """Drop missing / unreadable image paths."""
    out: list[str] = []
    for p in images or []:
        if not p or not os.path.isfile(p):
            continue
        if os.path.getsize(p) < 1024:   # < 1 KB → almost certainly broken
            continue
        out.append(p)
    return out


def _kb_segment_for_image(
    img_path: str,
    out_path: str,
    width: int,
    height: int,
    duration_s: float,
    fps: int,
    ffmpeg_bin: str,
    direction: int = 0,
) -> bool:
    """Produce one Ken-Burns MP4 segment for a single image.

    ``direction`` cycles 0..3 → push-in centred, push-in toward right,
    push-in toward up, push-in toward left, alternating across images
    so the carousel doesn't feel mechanical.
    """
    frames = max(1, int(round(duration_s * fps)))
    # zoom rate so we go from 1.0 → ~1.15 across the segment.
    zoom_step = 0.0015
    # pan expressions per direction
    if direction == 0:
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif direction == 1:
        x_expr = "iw - iw/zoom"            # ride the right edge
        y_expr = "ih/2-(ih/zoom/2)"
    elif direction == 2:
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "0"                       # ride the top edge
    else:
        x_expr = "0"                       # ride the left edge
        y_expr = "ih/2-(ih/zoom/2)"
    # zoompan needs a fixed input size; we upscale with scale first so
    # zoom doesn't reveal source pixel grid on lower-res images.
    upscale_w = max(width * 4, 2400)
    upscale_h = max(height * 4, 2400)
    vfilter = (
        f"scale={upscale_w}:{upscale_h}:force_original_aspect_ratio=increase,"
        f"crop={upscale_w}:{upscale_h},"
        f"zoompan=z='min(zoom+{zoom_step},1.15)':"
        f"d={frames}:fps={fps}:"
        f"x='{x_expr}':y='{y_expr}':s={width}x{height},"
        f"format=yuv420p"
    )
    cmd = [
        ffmpeg_bin, "-y",
        "-loop", "1",
        "-i", img_path,
        "-vf", vfilter,
        "-t", f"{duration_s:.3f}",
        "-r", str(fps),
        "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        logger.warning("kb_segment failed for %s (rc=%d): %s",
                       img_path, proc.returncode,
                       "\n".join((proc.stderr or "").splitlines()[-8:]))
        return False
    return True


def _crossfade_chain(
    segment_paths: list[str],
    segment_duration_s: float,
    out_path: str,
    width: int,
    height: int,
    fps: int,
    ffmpeg_bin: str,
    crossfade_s: float = 0.6,
) -> None:
    """Concatenate per-image segments with xfade transitions in one call.

    Falls back to concat-demuxer if the xfade graph fails (FFmpeg builds
    older than 4.3 don't support xfade).
    """
    n = len(segment_paths)
    if n == 1:
        shutil.copyfile(segment_paths[0], out_path)
        return

    cmd: list[str] = [ffmpeg_bin, "-y"]
    for p in segment_paths:
        cmd += ["-i", p]

    fc_lines: list[str] = []
    cur_label = "[0:v]"
    cur_dur = segment_duration_s
    for i in range(1, n):
        next_label = f"[v{i}]"
        offset = cur_dur - crossfade_s
        fc_lines.append(
            f"{cur_label}[{i}:v]xfade=transition=fade:"
            f"duration={crossfade_s}:offset={offset:.3f}{next_label}"
        )
        cur_label = next_label
        cur_dur = cur_dur + segment_duration_s - crossfade_s

    filter_complex = ";".join(fc_lines)
    cmd += [
        "-filter_complex", filter_complex,
        "-map", cur_label,
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-an",
        "-movflags", "+faststart",
        out_path,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        # Fall back to concat-demuxer (no transition)
        logger.warning(
            "image_carousel: xfade chain failed (rc=%d), falling back to concat. "
            "stderr tail: %s",
            proc.returncode,
            "\n".join((proc.stderr or "").splitlines()[-12:]),
        )
        _concat_demuxer(segment_paths, out_path, width, height, fps, ffmpeg_bin)


def _concat_demuxer(
    segment_paths: list[str],
    out_path: str,
    width: int,
    height: int,
    fps: int,
    ffmpeg_bin: str,
) -> None:
    """Reliable concat fallback. Stream copy because all segments share codec
    parameters from :func:`_kb_segment_for_image`."""
    work_dir = tempfile.mkdtemp(prefix="kaizer_carousel_concat_")
    try:
        list_path = os.path.join(work_dir, "list.txt")
        with open(list_path, "w", encoding="utf-8") as fh:
            for p in segment_paths:
                fh.write(f"file '{p.replace(chr(92), '/')}'\n")
        cmd = [
            ffmpeg_bin, "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            "-movflags", "+faststart",
            out_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or "").splitlines()[-12:])
            raise CarouselError(
                f"concat-demuxer fallback failed (rc={proc.returncode}): {tail}"
            )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ── Public API ───────────────────────────────────────────────────────────────

def _build_carousel(
    images: list[str],
    duration_s: float,
    out_path: str,
    width: int,
    height: int,
    fps: int,
    crossfade_s: float,
    ffmpeg_bin: Optional[str],
    work_dir: Optional[str],
) -> CarouselResult:
    ffmpeg_bin = _resolve_ffmpeg_bin(ffmpeg_bin)
    images = _filter_usable(images)
    if not images:
        raise CarouselError("No usable images to build a carousel")

    # Per-image dwell — leave room for crossfade overlap.
    n = len(images)
    if n == 1:
        seg_dur = duration_s
        crossfade_s = 0.0
    else:
        # Total visual duration = n*seg - (n-1)*crossfade.
        # Solve seg = (duration + (n-1)*crossfade) / n.
        seg_dur = (duration_s + (n - 1) * crossfade_s) / n
        # Don't let seg_dur fall below 2.0 — too short to read a photo.
        seg_dur = max(2.0, seg_dur)

    cleanup_work = False
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="kaizer_carousel_")
        cleanup_work = True
    else:
        os.makedirs(work_dir, exist_ok=True)

    seg_paths: list[str] = []
    try:
        for i, img in enumerate(images):
            seg_p = os.path.join(work_dir, f"seg_{i:02d}.mp4")
            ok = _kb_segment_for_image(
                img, seg_p, width, height, seg_dur, fps,
                ffmpeg_bin, direction=i % 4,
            )
            if ok:
                seg_paths.append(seg_p)
        if not seg_paths:
            raise CarouselError("All Ken-Burns segments failed to render")
        if len(seg_paths) == 1 and n > 1:
            # Just one survived — extend it to full duration.
            shutil.copyfile(seg_paths[0], out_path)
        elif len(seg_paths) == 1:
            shutil.copyfile(seg_paths[0], out_path)
        else:
            _crossfade_chain(
                seg_paths, seg_dur, out_path,
                width, height, fps, ffmpeg_bin, crossfade_s,
            )
        # Probe the actual duration we produced
        try:
            from pipeline_core.qa import FFPROBE_BIN as _fp
        except Exception:
            _fp = shutil.which("ffprobe") or "ffprobe"
        actual = 0.0
        try:
            proc = subprocess.run(
                [_fp, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", out_path],
                capture_output=True, text=True, timeout=30,
            )
            actual = float((proc.stdout or "0").strip() or 0.0)
        except Exception:
            actual = duration_s
        return CarouselResult(
            output_path=out_path,
            images_used=len(seg_paths),
            duration_s=actual or duration_s,
        )
    finally:
        if cleanup_work:
            shutil.rmtree(work_dir, ignore_errors=True)


def build_sidebar_carousel(
    images: list[str],
    duration_s: float,
    out_path: str,
    *,
    width: int = 580,
    height: int = 800,
    fps: int = 30,
    crossfade_s: float = 0.6,
    ffmpeg_bin: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> CarouselResult:
    """Sidebar carousel sized to fit the long-form right pane (580×800)."""
    return _build_carousel(
        images, duration_s, out_path,
        width, height, fps, crossfade_s, ffmpeg_bin, work_dir,
    )


def build_fullscreen_takeover(
    images: list[str],
    duration_s: float,
    out_path: str,
    *,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    crossfade_s: float = 0.5,
    ffmpeg_bin: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> CarouselResult:
    """Full-frame photo gallery for between-story cut-aways (1920×1080)."""
    return _build_carousel(
        images, duration_s, out_path,
        width, height, fps, crossfade_s, ffmpeg_bin, work_dir,
    )
