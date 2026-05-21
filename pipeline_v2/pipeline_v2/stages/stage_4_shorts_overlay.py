"""Item 117 Phase 4 -- per-short overlay application.

Each ``short_NN_raw.mp4`` from Phase 2 gets its visual decoration
applied in a SINGLE ffmpeg pass:

  - 9:16 crop + scale to 1080x1920 (standard vertical short canvas)
  - Hook text + lower-third overlays + gradient (if a graph builder
    is supplied)
  - ``-c:a copy`` -- audio byte-identical through this step

Symmetric with ``stage_4_bulletin_overlay`` but with shorts-specific
defaults (vertical canvas, hook line at top). The same architectural
discipline applies: -c:a copy is enforced and verified via sha256
comparison of the input and output audio bitstreams.

For each short the caller supplies:

  - The Phase 2 raw path (``short_NN_raw.mp4``)
  - An overlay graph (default helper available) consuming ``[0:v]``
    and producing ``[out]``
  - Optional extra inputs (overlay PNGs etc.)

This module purposely does NOT iterate over a list of shorts -- the
caller is expected to call ``apply_short_overlays`` per file. That
keeps the failure surface per-short (one short blowing up doesn't
take the whole batch down) and lets the orchestrator parallelise
across shorts if it wants.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger("pipeline_v2.stage_4_shorts_overlay")


# Default shorts canvas (vertical 9:16 1080x1920).
DEFAULT_SHORT_WIDTH: int = 1080
DEFAULT_SHORT_HEIGHT: int = 1920

# Same encoder defaults as Phase 2 / 3.
NVENC_VIDEO_ARGS: tuple[str, ...] = (
    "-c:v", "h264_nvenc",
    "-preset", "p4",
    "-tune", "hq",
    "-rc", "vbr",
    "-cq", "23",
)
LIBX264_VIDEO_ARGS: tuple[str, ...] = (
    "-c:v", "libx264",
    "-preset", "medium",
    "-crf", "23",
    "-pix_fmt", "yuv420p",
)


class ShortsOverlayError(RuntimeError):
    """ffmpeg failure or audio-bit-identity violation."""


@dataclass(frozen=True)
class ShortOverlayResult:
    output_path: str
    wall_seconds: float
    audio_bit_identical: bool
    video_duration_s: float
    audio_duration_s: float
    av_delta_ms: float


def _ffprobe_streams(path: str, ffprobe_bin: str) -> dict:
    try:
        r = subprocess.run(
            [ffprobe_bin, "-v", "error", "-show_streams",
             "-of", "json", path],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return {}
        data = json.loads(r.stdout)
        out: dict = {}
        for s in data.get("streams", []):
            t = s.get("codec_type")
            if not t:
                continue
            out[t] = {
                "duration": float(s.get("duration", 0) or 0),
                "nb_frames": int(s.get("nb_frames", 0) or 0),
            }
        return out
    except Exception as exc:
        logger.warning("shorts_overlay: ffprobe failed for %r: %s", path, exc)
        return {}


def _audio_sha256(path: str, ffmpeg_bin: str) -> Optional[str]:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as tf:
        tmp_path = tf.name
    try:
        r = subprocess.run(
            [ffmpeg_bin, "-y", "-i", path, "-vn", "-c:a", "copy", tmp_path],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            return None
        h = hashlib.sha256()
        with open(tmp_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def build_default_short_graph(
    *,
    width: int = DEFAULT_SHORT_WIDTH,
    height: int = DEFAULT_SHORT_HEIGHT,
    hook_text: Optional[str] = None,
    lt_png_path: Optional[str] = None,
    font_path: Optional[str] = None,
) -> tuple[str, str, list[str]]:
    """Build a default per-short filter graph.

    The base operation is: crop the source video to 9:16 ratio
    (centred) then scale to the target canvas. Hook text (top) and
    lower-third (bottom) overlays are appended when supplied.

    Returns ``(filter_complex, out_label, extra_input_paths)``.
    """
    extras: list[str] = []
    next_idx = 1

    parts: list[str] = []
    # Crop to 9:16 centred. The source video may be any aspect ratio
    # (1080p horizontal, 4K, etc.); crop math: keep min(h, w*9/16)
    # height + match width to maintain ratio.
    parts.append(
        f"[0:v]crop='min(iw\\,ih*9/16)':'min(ih\\,iw*16/9)',"
        f"scale={width}:{height},setsar=1[cropped]"
    )
    cur_label = "cropped"

    # Hook text (top of canvas, big bold).
    if hook_text:
        # Escape ' for ffmpeg.
        safe = (hook_text or "").replace("'", "\\'").replace(":", "\\:")
        font_arg = f":fontfile='{font_path}'" if font_path else ""
        next_label = "with_hook"
        parts.append(
            f"[{cur_label}]drawtext=text='{safe}'{font_arg}:"
            f"x=(w-text_w)/2:y=80:fontsize=72:fontcolor=white:"
            f"borderw=4:bordercolor=black[{next_label}]"
        )
        cur_label = next_label

    # Lower-third PNG overlay.
    if lt_png_path:
        extras.append(lt_png_path)
        lt_idx = next_idx
        next_idx += 1
        next_label = "with_lt"
        parts.append(
            f"[{cur_label}][{lt_idx}:v]overlay="
            f"x=(W-w)/2:y=H-h-120:format=auto[{next_label}]"
        )
        cur_label = next_label

    # No overlays: just the crop+scale.
    return ";".join(parts), cur_label, extras


def apply_short_overlays(
    short_raw_path: str,
    output_path: str,
    *,
    video_filter_complex: str,
    video_out_label: str,
    extra_input_paths: Sequence[str] = (),
    ffmpeg_bin: Optional[str] = None,
    ffprobe_bin: Optional[str] = None,
    use_nvenc: Optional[bool] = None,
    timeout_s: float = 1800.0,
    verify_audio_bit_identity: bool = True,
    progress_cb: Optional[callable] = None,
) -> ShortOverlayResult:
    """Apply caller-built video filter to short_NN_raw.mp4 with
    ``-c:a copy``. Symmetric with apply_bulletin_overlays."""
    if ffmpeg_bin is None:
        try:
            from pipeline_core.pipeline import FFMPEG_BIN as _FF
            ffmpeg_bin = _FF or "ffmpeg"
        except Exception:
            ffmpeg_bin = "ffmpeg"
    if ffprobe_bin is None:
        try:
            from pipeline_core.qa import FFPROBE_BIN as _FP
            ffprobe_bin = _FP or "ffprobe"
        except Exception:
            import shutil as _sh
            ffprobe_bin = _sh.which("ffprobe") or "ffprobe"

    if use_nvenc is None:
        try:
            from pipeline_v2.utils.hw_accel import detect_encoder
            use_nvenc = (detect_encoder() == "h264_nvenc")
        except Exception:
            use_nvenc = False

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    audio_sha_in: Optional[str] = None
    if verify_audio_bit_identity:
        audio_sha_in = _audio_sha256(short_raw_path, ffmpeg_bin)

    video_args = list(NVENC_VIDEO_ARGS) if use_nvenc else list(LIBX264_VIDEO_ARGS)

    cmd: list[str] = [ffmpeg_bin, "-y", "-i", short_raw_path]
    for extra in extra_input_paths:
        cmd += ["-loop", "1", "-i", extra]
    cmd += [
        "-filter_complex", video_filter_complex,
        "-map", f"[{video_out_label}]",
        "-map", "0:a",
        *video_args,
        "-c:a", "copy",   # <- THE INVARIANT
        "-movflags", "+faststart",
        output_path,
    ]

    if progress_cb:
        progress_cb(
            f"Stage 6/7 short overlay: "
            f"applying graph ({len(extra_input_paths)} extras)"
        )

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise ShortsOverlayError(
            f"shorts overlay ffmpeg timed out after {timeout_s:.0f}s"
        ) from exc
    wall = time.time() - t0

    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").splitlines()[-20:])
        raise ShortsOverlayError(
            f"shorts overlay ffmpeg rc={proc.returncode}: {tail}"
        )

    streams = _ffprobe_streams(output_path, ffprobe_bin)
    v_dur = float(streams.get("video", {}).get("duration", 0))
    a_dur = float(streams.get("audio", {}).get("duration", 0))
    av_delta_ms = (a_dur - v_dur) * 1000.0

    audio_bit_identical = True
    if verify_audio_bit_identity and audio_sha_in is not None:
        audio_sha_out = _audio_sha256(output_path, ffmpeg_bin)
        audio_bit_identical = (audio_sha_out == audio_sha_in)
        if not audio_bit_identical:
            raise ShortsOverlayError(
                f"audio bit-identity violated: input sha {audio_sha_in[:16]}... "
                f"!= output sha {(audio_sha_out or 'None')[:16]}..."
            )

    if progress_cb:
        progress_cb(
            f"Stage 6/7 short overlay done in {wall:.1f}s: "
            f"v={v_dur:.2f}s a={a_dur:.2f}s a-v={av_delta_ms:+.2f}ms"
        )

    return ShortOverlayResult(
        output_path=output_path,
        wall_seconds=wall,
        audio_bit_identical=audio_bit_identical,
        video_duration_s=v_dur,
        audio_duration_s=a_dur,
        av_delta_ms=av_delta_ms,
    )
