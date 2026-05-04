"""Bulletin stitcher — Phase 1 of the long-form YouTube pipeline.

Takes N already-sliced story MP4s and concatenates them into one
1–2 hour bulletin MP4. No graphics yet — those land in Phase 2
(TV9 broadcast overlay) and Phase 3 (image carousel).

Phase 1 contract
----------------
- Input: a list of MP4 paths, each one a story slice produced by
  ``pipeline_core.pipeline.cut_video_clips`` (so they share codec
  parameters — H.264 + AAC + 48 kHz + bt709, courtesy of
  ``ENCODE_ARGS_INTERMEDIATE``).
- Output: one MP4 at ``output_path`` containing all stories
  concatenated in order, with hard cuts between them.
- No re-encoding by default — the concat demuxer copies streams,
  which is fast and lossless. We only re-encode if a story has a
  mismatching codec (rare but defended against).
- Honours ``min_total_minutes`` / ``max_total_minutes`` only as a
  reporting / warning hint — we never silently drop stories to fit
  a target. If you want fewer stories, pre-filter the list yourself.

Why concat-demuxer first (no crossfade in Phase 1)
--------------------------------------------------
The plan called for ``xfade=transition=fade:duration=0.5`` between
stories. Implementing that for arbitrary N inputs requires a chained
``filter_complex`` with N-1 xfade nodes plus matching ``acrossfade``
nodes for audio, and the offsets depend on cumulative durations,
which means one bad probe can desync the whole bulletin. The concat
demuxer is bulletproof, completes in seconds for 90 min of stories,
and the resulting hard-cut bulletin is fully usable for end-to-end
verification of the rest of the pipeline (Gemini selection, R2
upload, YouTube push). Crossfades land in a Phase 1.5 polish pass
once the foundation is verified.

Failure tolerance
-----------------
- Any single broken story is skipped (logged, not raised).
- Returning ``stories_skipped > 0`` is fine — the user gets a
  shorter-than-ideal bulletin but never a crashed pipeline.
- If *every* story fails, we raise ``BulletinStitchError`` so the
  pipeline doesn't silently produce a 0-byte bulletin.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("kaizer.pipeline.bulletin_stitcher")


# ── Tunables (mirror the plan's defaults) ──────────────────────────────────────

DEFAULT_TARGET_TOTAL_MIN = 90
DEFAULT_MIN_TOTAL_MIN    = 60
DEFAULT_MAX_TOTAL_MIN    = 120


class BulletinStitchError(RuntimeError):
    """Raised when the bulletin cannot be produced at all (no usable stories)."""


@dataclass
class BulletinResult:
    output_path: str
    stories_rendered: int
    stories_skipped: int
    total_duration_s: float
    per_story_durations_s: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _probe_duration(path: str, ffprobe_bin: str) -> float:
    """Return MP4 duration in seconds. Returns 0.0 on failure (caller skips)."""
    cmd = [
        ffprobe_bin, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            return float((proc.stdout or "0").strip() or 0.0)
    except Exception as exc:
        logger.warning("bulletin_stitcher: probe failed for %r: %s", path, exc)
    return 0.0


def _validate_inputs(
    story_paths: list[str],
    ffprobe_bin: str,
) -> tuple[list[str], list[float], list[str]]:
    """Filter out unreadable / zero-duration stories.

    Returns ``(usable_paths, durations_s, warnings)``.
    """
    usable: list[str] = []
    durs: list[float] = []
    warnings: list[str] = []

    for idx, p in enumerate(story_paths, start=1):
        if not p or not os.path.isfile(p):
            warnings.append(f"story_{idx:02d}: file missing — skipping ({p!r})")
            continue
        d = _probe_duration(p, ffprobe_bin)
        if d <= 0.0:
            warnings.append(f"story_{idx:02d}: zero / unreadable duration — skipping")
            continue
        usable.append(p)
        durs.append(d)

    return usable, durs, warnings


def _write_concat_list(paths: list[str], list_path: str) -> None:
    """Write the concat-demuxer manifest. Paths are forward-slashed and
    single-quoted so spaces / Windows paths survive the parser."""
    with open(list_path, "w", encoding="utf-8") as fh:
        for p in paths:
            safe = p.replace("\\", "/").replace("'", r"'\''")
            fh.write(f"file '{safe}'\n")


# ── Public API ─────────────────────────────────────────────────────────────────

def stitch_bulletin(
    story_paths: list[str],
    output_path: str,
    *,
    work_dir: Optional[str] = None,
    target_total_minutes: float = DEFAULT_TARGET_TOTAL_MIN,
    min_total_minutes: float = DEFAULT_MIN_TOTAL_MIN,
    max_total_minutes: float = DEFAULT_MAX_TOTAL_MIN,
    ffmpeg_bin: Optional[str] = None,
    ffprobe_bin: Optional[str] = None,
) -> BulletinResult:
    """Concatenate N story MP4s into a single bulletin MP4.

    Parameters
    ----------
    story_paths
        Per-story MP4 paths in the order they should appear in the bulletin.
        Expected to share codec parameters (e.g., produced by
        ``cut_video_clips`` with ``ENCODE_ARGS_INTERMEDIATE``).
    output_path
        Where to write the stitched MP4.
    work_dir
        Temp directory for the concat manifest. Auto-created when None.
    target_total_minutes
        Informational target for the bulletin length. We do NOT trim to fit;
        we just warn if the actual total falls outside ``[min, max]``.
    ffmpeg_bin / ffprobe_bin
        Override the binaries. Defaults pull from ``pipeline_core.pipeline``.

    Returns
    -------
    BulletinResult

    Raises
    ------
    BulletinStitchError
        When zero stories are usable, or FFmpeg's concat call fails.
    """
    # Resolve binaries lazily — keeps this module importable in test contexts
    # that don't have the full pipeline_core graph available.
    if ffmpeg_bin is None or ffprobe_bin is None:
        try:
            from pipeline_core.pipeline import FFMPEG_BIN as _FF
            ffmpeg_bin = ffmpeg_bin or _FF
        except Exception:
            ffmpeg_bin = ffmpeg_bin or "ffmpeg"
        if ffprobe_bin is None:
            try:
                from pipeline_core.qa import FFPROBE_BIN as _FP
                ffprobe_bin = _FP
            except Exception:
                import shutil as _sh
                ffprobe_bin = _sh.which("ffprobe") or "ffprobe"

    # ── Step 1: validate and filter inputs ──
    usable_paths, durations, warnings = _validate_inputs(story_paths, ffprobe_bin)
    skipped = len(story_paths) - len(usable_paths)
    if not usable_paths:
        raise BulletinStitchError(
            f"No usable story segments out of {len(story_paths)} inputs — "
            f"every probe failed."
        )

    total_s = sum(durations)
    total_min = total_s / 60.0

    # Soft warnings (length out of target band).
    if total_min < min_total_minutes:
        warnings.append(
            f"bulletin total {total_min:.1f} min is below target floor "
            f"{min_total_minutes} min — output will still ship."
        )
    elif total_min > max_total_minutes:
        warnings.append(
            f"bulletin total {total_min:.1f} min exceeds target ceiling "
            f"{max_total_minutes} min — output will still ship."
        )
    logger.info(
        "stitch_bulletin: %d/%d stories usable, total=%.1f min (skipped=%d)",
        len(usable_paths), len(story_paths), total_min, skipped,
    )

    # ── Step 2: write concat manifest ──
    if work_dir:
        os.makedirs(work_dir, exist_ok=True)
        cleanup_work = False
    else:
        work_dir = tempfile.mkdtemp(prefix="kaizer_bulletin_")
        cleanup_work = True

    list_path = os.path.join(work_dir, "concat_list.txt")
    _write_concat_list(usable_paths, list_path)

    # ── Step 3: concat-demuxer call (stream copy, no re-encode) ──
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    cmd = [
        ffmpeg_bin, "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        "-movflags", "+faststart",
        output_path,
    ]
    logger.debug("stitch_bulletin: ffmpeg cmd: %s", " ".join(cmd))

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired as exc:
        raise BulletinStitchError(
            f"FFmpeg concat timed out after 1800s on {len(usable_paths)} stories"
        ) from exc

    if proc.returncode != 0:
        # Stream-copy can fail if codec parameters drift between inputs (e.g.
        # one story was re-encoded with a different profile). Retry with a
        # full re-encode using a sane H.264/AAC baseline so the bulletin
        # ships even if upstream slices were inconsistent.
        logger.warning(
            "stitch_bulletin: stream-copy concat failed (rc=%d), retrying with re-encode. "
            "stderr tail: %s",
            proc.returncode,
            "\n".join((proc.stderr or "").splitlines()[-12:]),
        )
        cmd_reenc = [
            ffmpeg_bin, "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ]
        try:
            proc2 = subprocess.run(cmd_reenc, capture_output=True, text=True, timeout=7200)
        except subprocess.TimeoutExpired as exc:
            raise BulletinStitchError(
                f"FFmpeg re-encode concat timed out after 2h on {len(usable_paths)} stories"
            ) from exc
        if proc2.returncode != 0:
            tail = "\n".join((proc2.stderr or "").splitlines()[-20:])
            raise BulletinStitchError(
                f"FFmpeg concat failed (re-encode rc={proc2.returncode}): {tail}"
            )
        warnings.append("concat fell back to re-encode — upstream slice codecs may have drifted")

    # ── Step 4: cleanup tmp manifest dir ──
    if cleanup_work:
        try:
            os.remove(list_path)
            os.rmdir(work_dir)
        except Exception:
            pass  # leftover tmp is fine

    # ── Step 5: re-probe the actual output to ground-truth the duration ──
    actual_dur = _probe_duration(output_path, ffprobe_bin)

    return BulletinResult(
        output_path=output_path,
        stories_rendered=len(usable_paths),
        stories_skipped=skipped,
        total_duration_s=actual_dur or total_s,
        per_story_durations_s=durations,
        warnings=warnings,
    )
