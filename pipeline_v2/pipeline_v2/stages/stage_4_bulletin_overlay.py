"""Item 117 Phase 3 -- bulletin overlay application.

Takes ``bulletin_raw.mp4`` from Phase 2 and applies the visual
decoration (sidebar / lower-third / ticker / channel bug) in a
SINGLE ffmpeg pass. The architectural invariant is uncompromising:

  ``-c:a copy`` -- audio is byte-identical through this step.

The diagnostic phase verified this works: applying a filter_complex
overlay graph to bulletin_raw.mp4 with -c:a copy produced an output
file whose audio sha256 matched the input bit-for-bit, while video
re-encoded with the overlays baked in. The 4a -> 4b boundary thus
guarantees lip-sync is preserved past the slice/concat stage.

API shape

  This module is intentionally narrow: it ENFORCES the discipline
  (single ffmpeg call, -c:a copy, audio bit-identity verification)
  but does NOT prescribe the visual graph. Callers supply the
  video filter_complex string + the final video out-label.

  ``build_default_overlay_graph()`` is a helper that produces a
  reasonable default graph (lower-third PNG + ticker PNG + channel
  bug PNG with time-conditioned ``enable=between(t, A, B)`` overlays
  for per-story lower-thirds). Callers with more complex needs
  (sidebar carousel video, marquee scrolling, PiP insets) can build
  their own filter_complex and pass it directly.

Per the V1 lineage, all overlay PNGs (lower-third, ticker, bug) are
pre-rendered by ``pipeline_core.longform_compose.render_lower_third``
etc. before this step runs.
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

logger = logging.getLogger("pipeline_v2.stage_4_bulletin_overlay")


# NVENC re-encode args -- video changes (overlays baked) but spec
# matches Stage 0's mezzanine for round-trip cleanliness.
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


class BulletinOverlayError(RuntimeError):
    """ffmpeg invocation failed, OR audio bit-identity was violated
    (would indicate -c:a copy got dropped or the muxer remuxed
    audio packets -- a real architectural regression)."""


@dataclass(frozen=True)
class StoryOverlaySpec:
    """One story's timing in the bulletin timeline + its lower-third
    PNG path.

    ``start_s`` and ``end_s`` are in bulletin-output time (NOT source
    time). ``lt_png_path`` is the pre-rendered lower-third image.
    ``lt_native_width`` is what ``render_lower_third`` returned for
    this story -- used to choose static-overlay vs scrolling-marquee
    behaviour for long headlines.
    """

    start_s: float
    end_s: float
    lt_png_path: str
    lt_native_width: int


@dataclass(frozen=True)
class BulletinOverlayResult:
    """Outcome of one apply_bulletin_overlays() call."""

    output_path: str
    wall_seconds: float
    audio_bit_identical: bool
    video_duration_s: float
    audio_duration_s: float
    av_delta_ms: float


def _ffprobe_streams(path: str, ffprobe_bin: str) -> dict:
    """Same shape as the raw_extract helper -- duplicated to keep the
    overlay module dependency-free of stage_4_raw_extract.py."""
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
        logger.warning("bulletin_overlay: ffprobe failed for %r: %s", path, exc)
        return {}


def _audio_sha256(path: str, ffmpeg_bin: str) -> Optional[str]:
    """SHA-256 of the AAC audio bitstream. Returns None if extraction
    fails. Used to verify -c:a copy preserved every byte."""
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


def build_default_overlay_graph(
    stories: Sequence[StoryOverlaySpec],
    ticker_png_path: Optional[str] = None,
    channel_bug_png_path: Optional[str] = None,
    *,
    width: int = 1920,
    height: int = 1080,
    ticker_h: int = 60,
    lt_h: int = 200,
    ticker_speed_px_s: float = 200.0,
    bug_pad_x: int = 30,
    bug_pad_y: int = 30,
) -> tuple[str, str, list[str]]:
    """Build a default overlay filter_complex graph.

    Returns ``(filter_complex_str, out_label, extra_input_paths)``.
    The caller passes ``extra_input_paths`` as additional ``-i``
    arguments AFTER the bulletin_raw input (so the bulletin is
    input 0; PNG inputs follow in order).

    Time-conditioned per-story lower-thirds:
      - LT_k is drawn with ``enable=between(t, story_k_start,
        story_k_end)`` so each story shows ITS OWN lower-third.

    Continuous ticker + bug:
      - Ticker scrolls horizontally at ``ticker_speed_px_s``.
      - Bug sits in the top-right with ``bug_pad_*`` margins.
    """
    extra_inputs: list[str] = []
    # Order of -i after the main bulletin (which is [0:v]):
    #   ticker (if present)      -> [N:v]
    #   bug (if present)         -> [N+1:v]
    #   each story's LT          -> [N+2:v] ... etc.
    next_idx = 1
    ticker_input_idx: Optional[int] = None
    bug_input_idx: Optional[int] = None
    lt_input_idxs: list[int] = []

    if ticker_png_path:
        extra_inputs.append(ticker_png_path)
        ticker_input_idx = next_idx
        next_idx += 1
    if channel_bug_png_path:
        extra_inputs.append(channel_bug_png_path)
        bug_input_idx = next_idx
        next_idx += 1
    for s in stories:
        extra_inputs.append(s.lt_png_path)
        lt_input_idxs.append(next_idx)
        next_idx += 1

    # Build the chain. Start with [0:v]; each overlay step writes to
    # a new label; the final label is returned.
    parts: list[str] = []
    cur_label = "0:v"

    # Per-story lower-thirds (time-conditioned).
    ticker_y = height - ticker_h
    lt_y = height - lt_h - ticker_h
    for k, (spec, idx) in enumerate(zip(stories, lt_input_idxs)):
        next_label = f"lt_{k:02d}"
        # Long headlines (LT PNG wider than canvas) get a marquee
        # scroll. Short headlines slide-in then lock at x=0.
        if spec.lt_native_width > width:
            x_expr = (
                f"if(lt(t-{spec.start_s:.6f}\\,0.4)\\,-w+w*(t-{spec.start_s:.6f})/0.4\\,"
                f"if(lt(t-{spec.start_s:.6f}\\,2.0)\\,0\\,"
                f"max(W-w\\,-((t-{spec.start_s:.6f}-2.0)*60))))"
            )
        else:
            x_expr = (
                f"if(lt(t-{spec.start_s:.6f}\\,0.4)\\,"
                f"-w+w*(t-{spec.start_s:.6f})/0.4\\,0)"
            )
        parts.append(
            f"[{cur_label}][{idx}:v]overlay="
            f"x='{x_expr}':y={lt_y}:"
            f"enable='between(t\\,{spec.start_s:.6f}\\,{spec.end_s:.6f})':"
            f"format=auto[{next_label}]"
        )
        cur_label = next_label

    # Continuous ticker (one moving image across the full bulletin).
    if ticker_input_idx is not None:
        next_label = "with_ticker"
        parts.append(
            f"[{cur_label}][{ticker_input_idx}:v]overlay="
            f"x='W-mod(t*{ticker_speed_px_s:.1f}\\,w+W)':"
            f"y={ticker_y}:format=auto[{next_label}]"
        )
        cur_label = next_label

    # Channel bug (static, top-right).
    if bug_input_idx is not None:
        next_label = "with_bug"
        parts.append(
            f"[{cur_label}][{bug_input_idx}:v]overlay="
            f"x=W-w-{bug_pad_x}:y={bug_pad_y}:format=auto[{next_label}]"
        )
        cur_label = next_label

    if not parts:
        # No overlays requested -- pass-through. The caller will
        # still get a re-encode but the video is unchanged.
        parts.append(f"[0:v]copy[out]")
        cur_label = "out"

    return ";".join(parts), cur_label, extra_inputs


def apply_bulletin_overlays(
    bulletin_raw_path: str,
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
) -> BulletinOverlayResult:
    """Apply a pre-built overlay filter graph to bulletin_raw, with
    ``-c:a copy``. Verifies (when requested) that the output audio
    stream is byte-identical to the input.

    Parameters
    ----------
    bulletin_raw_path
        Phase 2's output ``bulletin_raw.mp4``.
    output_path
        Where to write ``bulletin_with_overlays.mp4``.
    video_filter_complex
        The video graph string. Must consume ``[0:v]`` (bulletin) and
        any ``[N:v]`` references for ``extra_input_paths`` in order.
        Must produce a final ``[{video_out_label}]`` for ``-map``.
    video_out_label
        Label name (no brackets) of the final video stream in the
        filter graph.
    extra_input_paths
        Additional ``-i`` files (PNGs etc.) after the bulletin input.
        Order matters: ``[1:v]`` is the first extra, ``[2:v]`` the
        second, etc.
    use_nvenc
        See ``stage_4_raw_extract``.
    verify_audio_bit_identity
        When ``True`` (default) computes SHA-256 of the input AND
        output audio bitstreams; raises if they differ. Set to
        ``False`` only when you intend the audio to change (you
        almost certainly don't at this step).

    Raises
    ------
    BulletinOverlayError
        ffmpeg failure OR audio bit-identity violation.
    """
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

    # Snapshot input audio sha BEFORE we run ffmpeg, so the check is
    # truly comparing input vs output bytes.
    audio_sha_in: Optional[str] = None
    if verify_audio_bit_identity:
        audio_sha_in = _audio_sha256(bulletin_raw_path, ffmpeg_bin)

    video_args = list(NVENC_VIDEO_ARGS) if use_nvenc else list(LIBX264_VIDEO_ARGS)

    cmd: list[str] = [ffmpeg_bin, "-y", "-i", bulletin_raw_path]
    for extra in extra_input_paths:
        # Stills (PNG/JPG) need -loop 1 to feed the filter graph
        # indefinitely; ffmpeg auto-loops when the filter graph runs
        # past the still's EOF, but we set it explicitly for clarity.
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
            f"Stage 6/7 overlay: applying graph "
            f"({len(extra_input_paths)} extra inputs, "
            f"{len(video_filter_complex)} filter chars)"
        )

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise BulletinOverlayError(
            f"overlay ffmpeg timed out after {timeout_s:.0f}s"
        ) from exc
    wall = time.time() - t0

    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").splitlines()[-20:])
        raise BulletinOverlayError(
            f"overlay ffmpeg rc={proc.returncode}: {tail}"
        )

    # Probe + verify.
    streams = _ffprobe_streams(output_path, ffprobe_bin)
    v_dur = float(streams.get("video", {}).get("duration", 0))
    a_dur = float(streams.get("audio", {}).get("duration", 0))
    av_delta_ms = (a_dur - v_dur) * 1000.0

    audio_bit_identical = True
    if verify_audio_bit_identity and audio_sha_in is not None:
        audio_sha_out = _audio_sha256(output_path, ffmpeg_bin)
        audio_bit_identical = (audio_sha_out == audio_sha_in)
        if not audio_bit_identical:
            raise BulletinOverlayError(
                f"audio bit-identity violated: input sha {audio_sha_in[:16]}... "
                f"!= output sha {(audio_sha_out or 'None')[:16]}... "
                f"(-c:a copy was meant to byte-preserve audio)"
            )

    if progress_cb:
        progress_cb(
            f"Stage 6/7 overlay done in {wall:.1f}s: v={v_dur:.2f}s "
            f"a={a_dur:.2f}s a-v={av_delta_ms:+.2f}ms "
            f"audio_bit_identical={audio_bit_identical}"
        )

    return BulletinOverlayResult(
        output_path=output_path,
        wall_seconds=wall,
        audio_bit_identical=audio_bit_identical,
        video_duration_s=v_dur,
        audio_duration_s=a_dur,
        av_delta_ms=av_delta_ms,
    )
