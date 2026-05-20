"""Item 108 -- crossfade-capable bulletin stitcher (V2 only).

The V1 stitcher in ``pipeline_core/bulletin_stitcher.py`` uses
ffmpeg's concat demuxer with ``-c copy`` (no re-encode, hard cuts,
fast). It works well but cannot insert any transition between
stories -- every splice is a hard cut, and the resulting audio
shows clicks / ambient spikes at the splice point because the
amplitude can be non-zero on either side of the cut.

This module adds a re-encoding path that performs a short audio
crossfade (default 80 ms) and a tiny video crossfade (default 40 ms,
~2 frames at 50fps) at every splice. Audible spikes are smoothed
into a tiny dip-and-recover; the video shows a barely-perceptible
1-2 frame blend.

Used only by the V2 renderer when ``transition_style`` resolves to
``smart_cut`` (80 ms) or ``crossfade`` (500 ms). All other catalog
entries still fall back to the V1 concat path via
``transitions.resolve_for_render``.

Filter graph layout
-------------------
For N inputs A, B, C, ... with durations d[0], d[1], d[2], ... and
overlap O, the chained xfade + acrossfade filter graph is::

    [0:v][1:v]xfade=duration=O:offset=d[0]-O[v01];
    [v01][2:v]xfade=duration=O:offset=(d[0]+d[1]-2O)[v012];
    ...
    [0:a][1:a]acrossfade=d=O[a01];
    [a01][2:a]acrossfade=d=O[a012];
    ...

The k-th xfade's ``offset`` is::

    sum(d[0..k]) - (k+1) * O

Total output duration is ``sum(d) - (N-1) * O`` -- the A/V invariant
the renderer's existing guardrail checks against.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("pipeline_v2.bulletin_crossfade_stitcher")


# Defaults match the iteration-2 spec from the user's brief:
#   smart_cut -> 80 ms audio, 40 ms video
#   crossfade -> 500 ms audio + video
DEFAULT_AUDIO_OVERLAP_S: float = 0.08
DEFAULT_VIDEO_OVERLAP_S: float = 0.04


class BulletinCrossfadeError(RuntimeError):
    """Raised when the crossfade stitcher cannot produce output."""


@dataclass
class BulletinCrossfadeResult:
    output_path: str
    stories_rendered: int
    stories_skipped: int
    total_duration_s: float
    audio_overlap_s: float
    video_overlap_s: float
    per_story_durations_s: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# --- Pure helpers (no I/O, fully unit-testable) ----------------------


def compute_xfade_offsets(
    durations: list[float],
    overlap_s: float,
) -> list[float]:
    """The offset value for each of the (N-1) chained xfade filters.

    For the k-th xfade (0-indexed) the offset, measured against the
    output of the previous chained xfade, is::

        sum(durations[0..k]) - (k+1) * overlap_s

    Raises ``ValueError`` for fewer than 2 durations, non-positive
    overlap, or an overlap that exceeds any individual segment's
    duration (would make the offset negative).
    """
    if len(durations) < 2:
        raise ValueError(
            "compute_xfade_offsets needs at least 2 segments; got "
            f"{len(durations)}."
        )
    if overlap_s <= 0:
        raise ValueError(
            f"overlap_s must be positive; got {overlap_s}."
        )
    for i, d in enumerate(durations):
        if d <= overlap_s:
            raise ValueError(
                f"segment[{i}] duration {d:.3f}s is <= overlap_s "
                f"{overlap_s:.3f}s -- the crossfade would consume the "
                f"entire segment. Skip the segment upstream or use a "
                f"smaller overlap."
            )
    offsets: list[float] = []
    cumulative = 0.0
    for k in range(len(durations) - 1):
        cumulative += durations[k]
        offsets.append(cumulative - (k + 1) * overlap_s)
    return offsets


def compute_total_duration(
    durations: list[float],
    overlap_s: float,
) -> float:
    """A/V invariant: ``total = sum(durations) - (N-1) * overlap_s``.

    Returns ``sum(durations)`` when ``len(durations) <= 1`` (no
    splices to subtract).
    """
    if len(durations) <= 1:
        return float(sum(durations))
    return float(sum(durations)) - (len(durations) - 1) * float(overlap_s)


def build_crossfade_filter_graph(
    durations: list[float],
    audio_overlap_s: float = DEFAULT_AUDIO_OVERLAP_S,
    video_overlap_s: float = DEFAULT_VIDEO_OVERLAP_S,
    transition: str = "fade",
) -> tuple[str, str, str]:
    """Construct the ``-filter_complex`` argument for N inputs.

    Returns ``(filter_str, video_out_label, audio_out_label)``.

    The labels are passed to ``-map`` so ffmpeg knows which streams
    to write to the output. Caller is responsible for ordering the
    ``-i`` arguments to match ``durations`` (input k corresponds to
    ``durations[k]``).

    For N == 1, the filter graph is empty (caller should bypass the
    crossfade pipeline entirely -- single-segment "bulletin" has no
    splices to crossfade).
    """
    n = len(durations)
    if n < 2:
        return ("", "0:v", "0:a")

    v_offsets = compute_xfade_offsets(durations, video_overlap_s)
    a_offsets = compute_xfade_offsets(durations, audio_overlap_s)

    parts: list[str] = []
    # Video chain. The first node takes [0:v][1:v]; subsequent take
    # the previous output label + the next [k:v] input.
    prev_v = "0:v"
    for k in range(n - 1):
        out_v = f"v{k+1:03d}"
        parts.append(
            f"[{prev_v}][{k+1}:v]xfade="
            f"transition={transition}:"
            f"duration={video_overlap_s}:"
            f"offset={v_offsets[k]:.6f}"
            f"[{out_v}]"
        )
        prev_v = out_v
    # Audio chain (acrossfade has no transition param).
    prev_a = "0:a"
    for k in range(n - 1):
        out_a = f"a{k+1:03d}"
        parts.append(
            f"[{prev_a}][{k+1}:a]acrossfade=d={audio_overlap_s}[{out_a}]"
        )
        prev_a = out_a

    filter_str = ";".join(parts)
    return filter_str, prev_v, prev_a


# --- ffmpeg wrapper (impure) -----------------------------------------


def _probe_duration(path: str, ffprobe_bin: str) -> float:
    """Return MP4 duration in seconds. 0.0 on failure (caller skips)."""
    try:
        r = subprocess.run(
            [
                ffprobe_bin, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode == 0:
            return float((r.stdout or "0").strip() or 0.0)
    except Exception as exc:
        logger.warning(
            "bulletin_crossfade_stitcher: probe failed for %r: %s",
            path, exc,
        )
    return 0.0


def stitch_bulletin_with_crossfade(
    story_paths: list[str],
    output_path: str,
    *,
    audio_overlap_s: float = DEFAULT_AUDIO_OVERLAP_S,
    video_overlap_s: float = DEFAULT_VIDEO_OVERLAP_S,
    transition: str = "fade",
    work_dir: Optional[str] = None,
    ffmpeg_bin: Optional[str] = None,
    ffprobe_bin: Optional[str] = None,
) -> BulletinCrossfadeResult:
    """Crossfade-stitched bulletin. Replaces V1 concat for V2 only.

    Bypass on N <= 1 (single input copied verbatim). Falls back to a
    V1-style concat (no overlap) if any segment is shorter than
    ``max(audio_overlap_s, video_overlap_s)``. The fallback is logged
    as a warning so the operator sees the regression.
    """
    if ffmpeg_bin is None:
        try:
            from pipeline_core.pipeline import FFMPEG_BIN as _FF
            ffmpeg_bin = _FF
        except Exception:
            ffmpeg_bin = "ffmpeg"
    if ffprobe_bin is None:
        try:
            from pipeline_core.qa import FFPROBE_BIN as _FP
            ffprobe_bin = _FP
        except Exception:
            import shutil as _sh
            ffprobe_bin = _sh.which("ffprobe") or "ffprobe"

    # Validate + probe durations.
    usable: list[str] = []
    durs: list[float] = []
    warnings: list[str] = []
    for idx, p in enumerate(story_paths, start=1):
        if not p or not os.path.isfile(p):
            warnings.append(f"story_{idx:02d}: file missing -- skipping ({p!r})")
            continue
        d = _probe_duration(p, ffprobe_bin)
        if d <= 0.0:
            warnings.append(f"story_{idx:02d}: zero / unreadable duration -- skipping")
            continue
        usable.append(p)
        durs.append(d)

    if not usable:
        raise BulletinCrossfadeError(
            f"No usable story segments out of {len(story_paths)} inputs -- "
            f"every probe failed."
        )

    skipped = len(story_paths) - len(usable)
    n = len(usable)

    # Trivial cases -- bypass the crossfade machinery.
    if n == 1:
        # Copy the single input verbatim.
        warnings.append("Only one segment; bypassing crossfade (stream copy).")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        r = subprocess.run(
            [ffmpeg_bin, "-y", "-i", usable[0], "-c", "copy",
             "-movflags", "+faststart", output_path],
            capture_output=True, text=True, timeout=1800,
        )
        if r.returncode != 0:
            raise BulletinCrossfadeError(
                f"ffmpeg copy failed (rc={r.returncode}): "
                f"{(r.stderr or '')[-400:]}"
            )
        actual = _probe_duration(output_path, ffprobe_bin)
        return BulletinCrossfadeResult(
            output_path=output_path,
            stories_rendered=n,
            stories_skipped=skipped,
            total_duration_s=actual or durs[0],
            audio_overlap_s=audio_overlap_s,
            video_overlap_s=video_overlap_s,
            per_story_durations_s=durs,
            warnings=warnings,
        )

    max_overlap = max(audio_overlap_s, video_overlap_s)
    if any(d <= max_overlap for d in durs):
        warnings.append(
            "At least one segment is shorter than the crossfade "
            "overlap window; falling back to V1 hard-cut concat for "
            "this bulletin."
        )
        from pipeline_core.bulletin_stitcher import stitch_bulletin
        v1_res = stitch_bulletin(
            usable, output_path,
            work_dir=work_dir,
            ffmpeg_bin=ffmpeg_bin,
            ffprobe_bin=ffprobe_bin,
        )
        return BulletinCrossfadeResult(
            output_path=v1_res.output_path,
            stories_rendered=v1_res.stories_rendered,
            stories_skipped=v1_res.stories_skipped + skipped,
            total_duration_s=v1_res.total_duration_s,
            audio_overlap_s=0.0,   # fallback: no overlap applied
            video_overlap_s=0.0,
            per_story_durations_s=v1_res.per_story_durations_s,
            warnings=warnings + v1_res.warnings,
        )

    # Build filter graph.
    filter_str, v_label, a_label = build_crossfade_filter_graph(
        durs,
        audio_overlap_s=audio_overlap_s,
        video_overlap_s=video_overlap_s,
        transition=transition,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    cmd: list[str] = [ffmpeg_bin, "-y"]
    for p in usable:
        cmd.extend(["-i", p])
    cmd.extend([
        "-filter_complex", filter_str,
        "-map", f"[{v_label}]", "-map", f"[{a_label}]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-r", "30", "-fps_mode", "cfr",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        output_path,
    ])

    logger.info(
        "bulletin_crossfade_stitcher: %d segments, audio_overlap=%.3fs, "
        "video_overlap=%.3fs, expected_total=%.2fs",
        n, audio_overlap_s, video_overlap_s,
        compute_total_duration(durs, audio_overlap_s),
    )

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired as exc:
        raise BulletinCrossfadeError(
            f"ffmpeg crossfade timed out after 2h on {n} segments"
        ) from exc

    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").splitlines()[-20:])
        raise BulletinCrossfadeError(
            f"ffmpeg crossfade failed (rc={proc.returncode}): {tail}"
        )

    actual = _probe_duration(output_path, ffprobe_bin)
    expected = compute_total_duration(durs, audio_overlap_s)
    return BulletinCrossfadeResult(
        output_path=output_path,
        stories_rendered=n,
        stories_skipped=skipped,
        total_duration_s=actual or expected,
        audio_overlap_s=audio_overlap_s,
        video_overlap_s=video_overlap_s,
        per_story_durations_s=durs,
        warnings=warnings,
    )
