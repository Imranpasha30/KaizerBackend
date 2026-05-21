"""Item 108 / 111 -- audio-crossfade bulletin stitcher (V2 only).

The V1 stitcher in ``pipeline_core/bulletin_stitcher.py`` uses
ffmpeg's concat demuxer with ``-c copy`` (no re-encode, hard cuts,
fast). It works well but produces audible audio clicks / ambient
spikes at every splice because the amplitude can be non-zero on
either side of the cut.

This module adds a short audio crossfade (default 80 ms) at every
splice. VIDEO IS CONCAT'D WITH HARD CUTS -- no video xfade. The
reason is item 111's diagnosis (Job 46, 2026-05-21): ffmpeg's
``xfade`` filter does not chain reliably for 20+ video transitions
with cumulative offsets in the hundreds of seconds. The video
stream silently collapsed to ~one segment's worth of frames while
the audio chain (acrossfade) produced the correct cumulative
duration. We split the two concerns into separate ffmpeg passes:

  Pass 1: concat-demuxer for VIDEO only (``-c copy -an``). Lossless,
          fast (no re-encode). Inputs must share codec parameters
          (which they do: every composed_story_NN.mp4 comes out of
          Stage 4's compose step with the same encoder settings).
          Output duration = sum(per-segment video durations).
  Pass 2: filter_complex acrossfade chain for AUDIO only
          (``-vn``). 25-segment chains are fine -- the audio chain
          was never broken; only xfade-on-video failed.
          Output duration = sum(per-segment audio durations) - (N-1)
          * audio_overlap_s.
  Pass 3: mux video + audio (``-c copy -shortest``). The
          ``-shortest`` flag trims to the shorter of the two
          streams, which guarantees the file has matching A/V end
          points (without it, the natural ~1s drift between per-
          segment video and audio durations would leave video
          extending past the last audio sample).

A 1-2 frame video hard cut at every 80ms audio crossfade boundary
is visually imperceptible because the eye doesn't perceive frame-
level discontinuity at 30fps. The audio crossfade is the part
that matters for Gemini's "ambient spike at splice point"
finding (item 108).

Used only by the V2 renderer when ``transition_style`` resolves to
``smart_cut`` (80 ms audio) or ``crossfade`` (500 ms audio).
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("pipeline_v2.bulletin_crossfade_stitcher")


# Defaults match the iteration-2 / item-108 spec:
#   smart_cut -> 80 ms audio crossfade
#   crossfade -> 500 ms audio crossfade
# VIDEO is now hard-cut (item 111). The ``video_overlap_s``
# argument is retained for API compatibility but is no longer used.
DEFAULT_AUDIO_OVERLAP_S: float = 0.08
DEFAULT_VIDEO_OVERLAP_S: float = 0.0     # hard cut (item 111)


class BulletinCrossfadeError(RuntimeError):
    """Raised when the crossfade stitcher cannot produce output."""


@dataclass
class BulletinCrossfadeResult:
    output_path: str
    stories_rendered: int
    stories_skipped: int
    total_duration_s: float
    audio_overlap_s: float
    video_overlap_s: float       # always 0.0 since item 111
    per_story_durations_s: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# --- Pure helpers (no I/O, fully unit-testable) ----------------------


def compute_xfade_offsets(
    durations: list[float],
    overlap_s: float,
) -> list[float]:
    """The offset value for each of the (N-1) chained acrossfade
    filters.

    For the k-th acrossfade (0-indexed) the offset, measured against
    the output of the previous chained acrossfade, is::

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
    """A/V invariant for the AUDIO chain: ``total = sum(durations) -
    (N-1) * overlap_s``. Video uses hard concat so its total is just
    ``sum(durations)`` -- callers needing the video total can pass
    ``overlap_s=0``.

    Returns ``sum(durations)`` when ``len(durations) <= 1`` (no
    splices to subtract).
    """
    if len(durations) <= 1:
        return float(sum(durations))
    return float(sum(durations)) - (len(durations) - 1) * float(overlap_s)


def build_audio_acrossfade_graph(
    durations: list[float],
    audio_overlap_s: float = DEFAULT_AUDIO_OVERLAP_S,
) -> tuple[str, str]:
    """Construct the ``-filter_complex`` argument for the AUDIO chain.

    Returns ``(filter_str, audio_out_label)``. Item 111: this
    replaces the previous ``build_crossfade_filter_graph`` which
    also produced a video xfade chain. The video chain is gone --
    video is now hard-concat via the concat demuxer in pass 1.

    Caller is responsible for ordering the ``-i`` arguments to match
    ``durations`` (input k corresponds to ``durations[k]``).

    For N == 1, the filter graph is empty (caller should bypass).
    """
    n = len(durations)
    if n < 2:
        return ("", "0:a")
    # compute_xfade_offsets validates inputs + raises on degenerate
    # cases. The acrossfade filter doesn't itself accept an offset
    # parameter (it crossfades the END of input 1 with the START of
    # input 2), so we don't pass offsets through to the filter graph
    # at all -- but we still call compute_xfade_offsets so the
    # "segment shorter than overlap" guard fires.
    _ = compute_xfade_offsets(durations, audio_overlap_s)
    parts: list[str] = []
    prev_a = "0:a"
    for k in range(n - 1):
        out_a = f"a{k+1:03d}"
        parts.append(
            f"[{prev_a}][{k+1}:a]acrossfade=d={audio_overlap_s}[{out_a}]"
        )
        prev_a = out_a
    return ";".join(parts), prev_a


# Backward-compat shim. Old callers (and a couple of tests) used
# ``build_crossfade_filter_graph`` which returned (filter, v_label,
# a_label). After item 111 the video chain is gone, so the v_label
# is fixed to "0:v" (single video input -- but this function is no
# longer used by the production path; only kept for tests that
# explicitly assert the OLD behaviour, which we'll prune).
def build_crossfade_filter_graph(
    durations: list[float],
    audio_overlap_s: float = DEFAULT_AUDIO_OVERLAP_S,
    video_overlap_s: float = 0.0,   # ignored since item 111
    transition: str = "fade",       # ignored since item 111
) -> tuple[str, str, str]:
    """DEPRECATED (item 111). Use ``build_audio_acrossfade_graph``.

    Returns ``(audio_filter_str, "0:v", audio_out_label)`` for
    callers that still want the 3-tuple shape. The video chain is
    empty -- video flows through the 3-pass stitcher's concat path
    in pass 1, not through a filter_complex video chain.
    """
    filter_str, a_label = build_audio_acrossfade_graph(
        durations, audio_overlap_s=audio_overlap_s,
    )
    return filter_str, "0:v", a_label


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


def _write_concat_manifest(paths: list[str], manifest_path: str) -> None:
    """Write the concat-demuxer manifest. Same forward-slash + quoting
    convention as ``pipeline_core.bulletin_stitcher._write_concat_list``.
    """
    with open(manifest_path, "w", encoding="utf-8") as fh:
        for p in paths:
            abs_p = os.path.abspath(p).replace("\\", "/").replace("'", r"'\''")
            fh.write(f"file '{abs_p}'\n")


def stitch_bulletin_with_crossfade(
    story_paths: list[str],
    output_path: str,
    *,
    audio_overlap_s: float = DEFAULT_AUDIO_OVERLAP_S,
    video_overlap_s: float = DEFAULT_VIDEO_OVERLAP_S,
    transition: str = "fade",   # ignored since item 111
    work_dir: Optional[str] = None,
    ffmpeg_bin: Optional[str] = None,
    ffprobe_bin: Optional[str] = None,
) -> BulletinCrossfadeResult:
    """Item 111: 3-pass audio-crossfade stitcher.

    Pass 1: concat-demuxer for video only (-c:v copy -an). Lossless,
            no re-encode. Inputs must share video codec parameters
            (which they do for composed_story_NN.mp4 outputs).
    Pass 2: filter_complex acrossfade chain for audio only (-vn).
            The audio chain handles 25+ segments reliably (the bug
            from Job 46 was in the VIDEO xfade chain, not audio).
    Pass 3: mux video + audio with -c copy -shortest. The -shortest
            flag handles the natural ~1s drift between per-segment
            video and audio durations by trimming to the shorter
            stream -- the end-frame trim (item 109) handles any
            remaining slack.

    Trivial-case bypass:
      - N == 0: raise (no usable inputs)
      - N == 1: stream-copy the single input verbatim
      - Any segment shorter than audio_overlap_s: fall back to V1
        hard-cut concat for the whole bulletin (the acrossfade
        guard would otherwise reject the chain)

    The ``video_overlap_s`` and ``transition`` kwargs are retained
    for API compatibility but are ignored. Item 111 hard-cuts video
    at every splice; only audio is crossfaded.
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

    # Resolve / create work_dir for intermediates.
    if work_dir:
        os.makedirs(work_dir, exist_ok=True)
        cleanup_work = False
    else:
        work_dir = tempfile.mkdtemp(prefix="kaizer_crossfade_")
        cleanup_work = True

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    # ---- N == 1: bypass (stream copy) ----
    if n == 1:
        warnings.append("Only one segment; bypassing crossfade (stream copy).")
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
        if cleanup_work:
            try:
                os.rmdir(work_dir)
            except Exception:
                pass
        return BulletinCrossfadeResult(
            output_path=output_path,
            stories_rendered=n,
            stories_skipped=skipped,
            total_duration_s=actual or durs[0],
            audio_overlap_s=audio_overlap_s,
            video_overlap_s=0.0,
            per_story_durations_s=durs,
            warnings=warnings,
        )

    # ---- Any segment too short for the audio overlap -> V1 fallback ----
    if any(d <= audio_overlap_s for d in durs):
        warnings.append(
            "At least one segment is shorter than the audio overlap "
            "window; falling back to V1 hard-cut concat for this "
            "bulletin."
        )
        from pipeline_core.bulletin_stitcher import stitch_bulletin
        v1_res = stitch_bulletin(
            usable, output_path,
            work_dir=work_dir,
            ffmpeg_bin=ffmpeg_bin,
            ffprobe_bin=ffprobe_bin,
        )
        if cleanup_work:
            try:
                os.rmdir(work_dir)
            except Exception:
                pass
        return BulletinCrossfadeResult(
            output_path=v1_res.output_path,
            stories_rendered=v1_res.stories_rendered,
            stories_skipped=v1_res.stories_skipped + skipped,
            total_duration_s=v1_res.total_duration_s,
            audio_overlap_s=0.0,
            video_overlap_s=0.0,
            per_story_durations_s=v1_res.per_story_durations_s,
            warnings=warnings + v1_res.warnings,
        )

    # ---- Pass 1: VIDEO concat (no re-encode) ----
    manifest_path = os.path.join(work_dir, "concat_video.txt")
    _write_concat_manifest(usable, manifest_path)
    video_only_path = os.path.join(work_dir, "_pass1_video.mp4")
    cmd1 = [
        ffmpeg_bin, "-y",
        "-f", "concat", "-safe", "0",
        "-i", manifest_path,
        "-c:v", "copy", "-an",          # video only, no re-encode
        "-movflags", "+faststart",
        video_only_path,
    ]
    logger.info(
        "bulletin_crossfade_stitcher pass 1/3 (video concat, %d segments)", n,
    )
    try:
        proc1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired as exc:
        raise BulletinCrossfadeError(
            f"pass 1 (video concat) timed out after 1800s on {n} segments"
        ) from exc
    if proc1.returncode != 0:
        tail = "\n".join((proc1.stderr or "").splitlines()[-20:])
        raise BulletinCrossfadeError(
            f"pass 1 (video concat) failed (rc={proc1.returncode}): {tail}"
        )

    # ---- Pass 2: AUDIO acrossfade chain ----
    audio_filter, audio_label = build_audio_acrossfade_graph(
        durs, audio_overlap_s=audio_overlap_s,
    )
    audio_only_path = os.path.join(work_dir, "_pass2_audio.m4a")
    cmd2: list[str] = [ffmpeg_bin, "-y"]
    for p in usable:
        cmd2.extend(["-i", p])
    cmd2.extend([
        "-filter_complex", audio_filter,
        "-map", f"[{audio_label}]",
        "-vn",                          # audio only
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        audio_only_path,
    ])
    logger.info(
        "bulletin_crossfade_stitcher pass 2/3 (audio acrossfade, "
        "audio_overlap=%.3fs, expected_total=%.2fs)",
        audio_overlap_s,
        compute_total_duration(durs, audio_overlap_s),
    )
    try:
        proc2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired as exc:
        raise BulletinCrossfadeError(
            f"pass 2 (audio acrossfade) timed out after 1h on {n} segments"
        ) from exc
    if proc2.returncode != 0:
        tail = "\n".join((proc2.stderr or "").splitlines()[-20:])
        raise BulletinCrossfadeError(
            f"pass 2 (audio acrossfade) failed (rc={proc2.returncode}): {tail}"
        )

    # ---- Pass 3: mux video + audio with -shortest ----
    # Item 115: re-encode AUDIO (not -c copy) so -shortest can trim
    # sample-accurately to the video EOF. With -c:a copy the demuxer
    # could not split an AAC packet at video EOF, leaving up to one
    # AAC frame (~21ms) of audio extending past the last video frame
    # -- the residual lip-sync drift that survived item 112's cut
    # fix. -c:v copy still gives lossless video.
    cmd3 = [
        ffmpeg_bin, "-y",
        "-i", video_only_path,
        "-i", audio_only_path,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-shortest",
        "-movflags", "+faststart",
        output_path,
    ]
    logger.info(
        "bulletin_crossfade_stitcher pass 3/3 (mux v+a, -shortest)"
    )
    try:
        proc3 = subprocess.run(cmd3, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired as exc:
        raise BulletinCrossfadeError(
            f"pass 3 (mux) timed out after 1800s"
        ) from exc
    if proc3.returncode != 0:
        tail = "\n".join((proc3.stderr or "").splitlines()[-20:])
        raise BulletinCrossfadeError(
            f"pass 3 (mux) failed (rc={proc3.returncode}): {tail}"
        )

    # ---- Cleanup intermediates ----
    for p in (video_only_path, audio_only_path, manifest_path):
        try:
            os.remove(p)
        except Exception:
            pass
    if cleanup_work:
        try:
            os.rmdir(work_dir)
        except Exception:
            pass

    actual = _probe_duration(output_path, ffprobe_bin)
    expected = compute_total_duration(durs, audio_overlap_s)
    return BulletinCrossfadeResult(
        output_path=output_path,
        stories_rendered=n,
        stories_skipped=skipped,
        total_duration_s=actual or expected,
        audio_overlap_s=audio_overlap_s,
        video_overlap_s=0.0,   # always 0 since item 111
        per_story_durations_s=durs,
        warnings=warnings,
    )
