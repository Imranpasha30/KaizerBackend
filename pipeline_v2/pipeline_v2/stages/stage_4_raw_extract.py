"""Item 117 Phase 2 -- Raw timeline extraction.

ONE ffmpeg invocation, multi-output. Decodes the mezzanine once and
emits ``bulletin_raw.mp4`` plus zero or more ``short_NN_raw.mp4``
files. Audio + video for each output come from the same decode
timestamps, so per-clip lip-sync is locked at the architecture
level (no -695ms drift class that the cut-then-recombine chain
suffered).

See ``pipeline_v2.render.edl_builder`` for the pure filter_complex
builder. This module owns the ffmpeg invocation, NVENC args, and
post-extract a-v duration verification.

Production verification (diagnostic phase, Job 51 mezzanine):
  - 28-bulletin + 8-shorts in one call: 145s wall, A/V delta -0.01ms
  - Cross-video (HEVC 1080p 25fps test.mp4): bulletin -0.01ms,
    shorts 0.00ms across the board.

This module deliberately does NOT touch the editor_meta.json shape,
the overlay step, or the orchestrator wiring. Phase 3/4/5 handle
those.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from pipeline_v2.render.edl_builder import (
    DEFAULT_SNAP_GRID_S,
    EDL,
    OutputSpec,
    build_extraction_edl,
)

logger = logging.getLogger("pipeline_v2.stage_4_raw_extract")


# Production-default NVENC args. Mirrored from Stage 0's transcode
# (mezzanine spec) so re-encoded outputs round-trip cleanly.
NVENC_VIDEO_ARGS: tuple[str, ...] = (
    "-c:v", "h264_nvenc",
    "-preset", "p4",
    "-tune", "hq",
    "-rc", "vbr",
    "-cq", "23",
)
# libx264 fallback for boxes without NVENC.
LIBX264_VIDEO_ARGS: tuple[str, ...] = (
    "-c:v", "libx264",
    "-preset", "medium",
    "-crf", "23",
    "-pix_fmt", "yuv420p",
)
AUDIO_ARGS: tuple[str, ...] = (
    "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
)

# Post-extract verification tolerance. The diagnostic test produced
# <=0.67ms across all outputs; 5ms gives generous headroom for
# encoder edge cases without masking a real regression.
POST_EXTRACT_AV_TOLERANCE_MS: float = 5.0


class RawExtractError(RuntimeError):
    """ffmpeg invocation failed, OR a post-extract duration check
    exceeded the tolerance. The orchestrator catches this and lets
    Inngest retry (transient) or marks the job failed (terminal --
    handled by the caller's classifier)."""


@dataclass(frozen=True)
class ExtractedOutput:
    """Result for one of the raw timeline files."""

    role: str
    index: int
    out_path: str
    expected_duration_s: float
    video_duration_s: float
    audio_duration_s: float
    nb_frames: int
    av_delta_ms: float


@dataclass(frozen=True)
class RawExtractResult:
    """Result of one ``extract_raw_timeline`` invocation."""

    outputs: tuple[ExtractedOutput, ...]
    wall_seconds: float
    dropped_cuts: tuple = field(default_factory=tuple)

    @property
    def bulletin(self) -> Optional[ExtractedOutput]:
        for o in self.outputs:
            if o.role == "bulletin":
                return o
        return None

    @property
    def shorts(self) -> tuple[ExtractedOutput, ...]:
        return tuple(o for o in self.outputs if o.role == "short")

    @property
    def bulletin_stories(self) -> tuple[ExtractedOutput, ...]:
        """per_story mode: one ExtractedOutput per bulletin cut."""
        return tuple(o for o in self.outputs if o.role == "bulletin_story")


def _ffprobe_streams(path: str, ffprobe_bin: str) -> dict:
    """Return ``{codec_type: {"duration": float, "nb_frames": int}}``.
    Empty dict on probe failure (caller handles)."""
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
        logger.warning(
            "raw_extract: ffprobe failed for %r: %s", path, exc,
        )
        return {}


def _output_path_for(
    spec: OutputSpec,
    *,
    shorts_dir: Path,
    bulletin_dir: Path,
) -> Path:
    """Production filenames + directories:
      - ``bulletin_raw.mp4`` (concat mode) lands in ``bulletin_dir``
      - ``raw_clip_NN.mp4`` (per_story mode) lands in ``bulletin_dir``
        -- matches V1 / current cut step output names so the existing
        compose chain can consume them unchanged.
      - ``short_NN_raw.mp4`` lands in ``shorts_dir`` (V2 keeps shorts
        and bulletin in separate dirs to avoid V1 cache-key collision;
        see ``cut_raw_bulletin_stories`` docstring).

    NOTE on item 117 per_story integration: V1's compose chain expects
    raw_clip_NN.mp4 in the bulletin_dir. We honour that layout so the
    new extract is a drop-in replacement for the legacy
    cut_clips_frame_aligned step at the file-system level.
    """
    if spec.role == "bulletin":
        return bulletin_dir / "bulletin_raw.mp4"
    if spec.role == "bulletin_story":
        return bulletin_dir / f"raw_clip_{spec.index:02d}.mp4"
    return shorts_dir / f"short_{spec.index:02d}_raw.mp4"


def _build_ffmpeg_cmd(
    mezzanine_path: str,
    edl: EDL,
    shorts_dir: Path,
    bulletin_dir: Path,
    *,
    ffmpeg_bin: str,
    use_nvenc: bool,
) -> list[str]:
    """Assemble the multi-output ffmpeg cmd."""
    video_args = list(NVENC_VIDEO_ARGS) if use_nvenc else list(LIBX264_VIDEO_ARGS)
    audio_args = list(AUDIO_ARGS)

    cmd: list[str] = [
        ffmpeg_bin, "-y",
        "-i", mezzanine_path,
        "-filter_complex", edl.filter_complex,
    ]
    for spec in edl.outputs:
        out_path = _output_path_for(
            spec, shorts_dir=shorts_dir, bulletin_dir=bulletin_dir,
        )
        cmd += [
            "-map", f"[{spec.v_label}]",
            "-map", f"[{spec.a_label}]",
            *video_args,
            *audio_args,
            "-movflags", "+faststart",
            str(out_path),
        ]
    return cmd


def extract_raw_timeline(
    mezzanine_path: str,
    bulletin_cuts: Sequence[tuple[float, float]],
    shorts_cuts: Sequence[tuple[float, float]],
    out_dir: str,
    *,
    snap_grid_s: float = DEFAULT_SNAP_GRID_S,
    bulletin_mode: str = "concat",
    bulletin_out_dir: Optional[str] = None,
    ffmpeg_bin: Optional[str] = None,
    ffprobe_bin: Optional[str] = None,
    use_nvenc: Optional[bool] = None,
    timeout_s: float = 1800.0,
    progress_cb: Optional[callable] = None,
) -> RawExtractResult:
    """Run the single-pass multi-output extraction.

    Parameters
    ----------
    mezzanine_path
        Path to Stage 0's CFR 30fps mezzanine.
    bulletin_cuts
        List of ``(start_s, end_s)`` ranges in source-time. Must be in
        playback order (the EDL builder preserves order in the concat).
    shorts_cuts
        List of ``(start_s, end_s)`` ranges, one per desired short.
    out_dir
        Shorts output directory. ``short_NN_raw.mp4`` files land
        here. Created if missing.
    bulletin_out_dir
        Optional separate directory for bulletin outputs
        (``bulletin_raw.mp4`` or ``raw_clip_NN.mp4``). When ``None``
        bulletin outputs share ``out_dir``. In production V2 layout
        (item 117 phase 5 integration) callers should pass the
        bulletin_dir to avoid the V1 cache-key collision documented
        in ``cut_raw_bulletin_stories``.
    bulletin_mode
        ``"concat"`` -> single ``bulletin_raw.mp4`` covering all
        kept ranges. ``"per_story"`` -> one ``raw_clip_NN.mp4`` per
        bulletin cut (drop-in for the legacy cut step; the existing
        compose chain runs over these unchanged).
    snap_grid_s
        Frame grid for boundary snapping. Defaults to 1/30 s.
    ffmpeg_bin / ffprobe_bin
        Optional paths. When ``None`` falls back to V1's
        ``pipeline_core.pipeline.FFMPEG_BIN`` if importable, else
        bare ``ffmpeg`` / ``ffprobe``.
    use_nvenc
        ``True`` -> NVENC, ``False`` -> libx264, ``None`` ->
        autodetect via the optional ``hw_accel`` helper.
    timeout_s
        Hard wall-time cap for the ffmpeg invocation.
    progress_cb
        Optional ``str -> None`` callback for sub-phase logging.

    Returns
    -------
    RawExtractResult
        Per-output durations + post-extract a-v delta. All outputs
        whose ``|av_delta_ms| > POST_EXTRACT_AV_TOLERANCE_MS`` raise
        ``RawExtractError`` BEFORE returning.

    Raises
    ------
    RawExtractError
        ffmpeg returned non-zero OR any output failed the post-
        extract a-v delta check OR mezzanine probe failed.
    ValueError
        Caller-side mistake: empty plan, etc. (propagated from EDL
        builder).
    """
    shorts_dir_p = Path(out_dir)
    shorts_dir_p.mkdir(parents=True, exist_ok=True)
    bulletin_dir_p = Path(bulletin_out_dir) if bulletin_out_dir else shorts_dir_p
    bulletin_dir_p.mkdir(parents=True, exist_ok=True)

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

    edl = build_extraction_edl(
        bulletin_cuts, shorts_cuts,
        snap_grid_s=snap_grid_s,
        bulletin_mode=bulletin_mode,
    )

    if progress_cb:
        roles = ", ".join(
            f"{o.role}[{o.index}]" for o in edl.outputs
        )
        progress_cb(
            f"Stage 5-6/7 raw-extract: {len(edl.outputs)} outputs "
            f"({roles}); filter graph {len(edl.filter_complex)} chars"
        )

    cmd = _build_ffmpeg_cmd(
        mezzanine_path, edl, shorts_dir_p, bulletin_dir_p,
        ffmpeg_bin=ffmpeg_bin, use_nvenc=bool(use_nvenc),
    )

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RawExtractError(
            f"raw-extract ffmpeg timed out after {timeout_s:.0f}s "
            f"({len(edl.outputs)} outputs requested)"
        ) from exc
    wall = time.time() - t0

    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").splitlines()[-20:])
        raise RawExtractError(
            f"raw-extract ffmpeg rc={proc.returncode}: {tail}"
        )

    # Probe + verify every output.
    results: list[ExtractedOutput] = []
    failures: list[str] = []
    for spec in edl.outputs:
        out_path = _output_path_for(
            spec, shorts_dir=shorts_dir_p, bulletin_dir=bulletin_dir_p,
        )
        if not out_path.is_file():
            failures.append(
                f"missing output {out_path.name} for {spec.role}[{spec.index}]"
            )
            continue
        streams = _ffprobe_streams(str(out_path), ffprobe_bin)
        v = streams.get("video", {})
        a = streams.get("audio", {})
        v_dur = float(v.get("duration", 0))
        a_dur = float(a.get("duration", 0))
        nb_frames = int(v.get("nb_frames", 0))
        delta_ms = (a_dur - v_dur) * 1000.0
        results.append(ExtractedOutput(
            role=spec.role, index=spec.index,
            out_path=str(out_path),
            expected_duration_s=spec.duration_s,
            video_duration_s=v_dur,
            audio_duration_s=a_dur,
            nb_frames=nb_frames,
            av_delta_ms=delta_ms,
        ))
        if abs(delta_ms) > POST_EXTRACT_AV_TOLERANCE_MS:
            failures.append(
                f"{spec.role}[{spec.index}] {out_path.name}: "
                f"a-v delta {delta_ms:+.2f}ms exceeds "
                f"{POST_EXTRACT_AV_TOLERANCE_MS}ms tolerance"
            )

    if failures:
        raise RawExtractError(
            "raw-extract verification failed: " + "; ".join(failures)
        )

    if progress_cb:
        bul = next((r for r in results if r.role == "bulletin"), None)
        if bul:
            progress_cb(
                f"Stage 5-6/7 raw-extract: bulletin={bul.video_duration_s:.2f}s "
                f"({bul.nb_frames} fr) a-v={bul.av_delta_ms:+.2f}ms"
            )
        shorts_n = sum(1 for r in results if r.role == "short")
        if shorts_n:
            max_av = max(
                (abs(r.av_delta_ms) for r in results if r.role == "short"),
                default=0.0,
            )
            progress_cb(
                f"Stage 5-6/7 raw-extract: {shorts_n} shorts, "
                f"max |a-v|={max_av:.2f}ms"
            )

    return RawExtractResult(
        outputs=tuple(results),
        wall_seconds=wall,
        dropped_cuts=edl.dropped,
    )
