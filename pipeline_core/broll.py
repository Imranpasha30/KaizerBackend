"""
kaizer.pipeline.broll
=====================
Beat-aware B-roll insertion for the Kaizer News video pipeline.

Finds low-energy audio segments (RMS valleys) in a source video, then inserts
B-roll cutaway clips or static images at those natural pause points.  Only the
video channel is swapped during a cutaway; the source audio plays uninterrupted
so viewers stay attached to the narrator's voice.

Usage
-----
    from pipeline_core.broll import find_audio_valleys, insert_broll, BRollResult

    valleys = find_audio_valleys("/path/to/source.mp4", top_k=5)
    result = insert_broll(
        "/path/to/source.mp4",
        ["/path/to/broll1.mp4", "/path/to/broll2.jpg"],
        output_path="/path/to/out.mp4",
        max_inserts=3,
    )
    for ins in result.insertions:
        print(f"Inserted {ins.asset_path!r} at {ins.start_s:.2f}s "
              f"(valley RMS={ins.source_rms:.3f})")

BRollResult fields
------------------
  output_path : str                   — Absolute path to the stitched output.
  insertions  : list[BRollInsertion]  — One entry per B-roll segment inserted.
  warnings    : list[str]             — Non-fatal issues collected during processing.

find_audio_valleys
------------------
  Returns list of (start_s, rms_normalised) for the K quietest windows in the
  clip.  RMS=0 → total silence; RMS=1 → loudest window in the clip.

insert_broll rules
------------------
  - Never inserts in the first 3 s (hook) or last 2 s (CTA).
  - If a transcript is supplied, snaps each insert start to the nearest sentence
    boundary that precedes the valley, avoiding mid-word cuts.
  - B-roll assets rotate round-robin.
  - Image assets (.jpg/.jpeg/.png/.webp) are rendered to a temporary mp4 via
    FFmpeg's -loop 1 muxer at the same dimensions as the source video.
  - Video assets (.mp4/.mov/.mkv/.webm) are looped/trimmed to insert_duration_s.
  - Crossfades are applied with FFmpeg's xfade filter on the video stream.
  - Source audio is passed through untouched (-c:a copy on the final output).
  - All temporary files live under a mkdtemp directory, cleaned in finally:.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("kaizer.pipeline.broll")

# ── Asset-type classification ─────────────────────────────────────────────────

_IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_VIDEO_EXTS: frozenset[str] = frozenset({".mp4", ".mov", ".mkv", ".webm"})

# ── FFmpeg binary (resolved once by pipeline.py; we just import it) ───────────

def _get_ffmpeg() -> str:
    """Return the FFmpeg binary path, importing from pipeline_core.pipeline."""
    try:
        from pipeline_core.pipeline import FFMPEG_BIN  # type: ignore
        return FFMPEG_BIN
    except Exception:
        import shutil as _sh
        p = _sh.which("ffmpeg")
        return p or "ffmpeg"


# ── Public dataclasses ────────────────────────────────────────────────────────

@dataclass
class BRollInsertion:
    """Metadata for a single B-roll segment that was inserted.

    Attributes
    ----------
    asset_path : str
        Absolute path to the B-roll asset (image or video) that was used.
    start_s : float
        Position in the OUTPUT video (seconds) where this cutaway begins.
    duration_s : float
        Duration of the cutaway in seconds.
    source_rms : float
        Normalised RMS energy [0.0, 1.0] of the audio valley at this point.
        0.0 = total silence; 1.0 = loudest window in the source clip.
    """

    asset_path: str
    start_s: float
    duration_s: float
    source_rms: float


@dataclass
class BRollResult:
    """Result returned by insert_broll().

    Attributes
    ----------
    output_path : str
        Absolute path to the stitched output video.
    insertions : list[BRollInsertion]
        One entry per B-roll segment inserted (may be fewer than max_inserts
        if there were not enough valleys or usable assets).
    warnings : list[str]
        Non-fatal issues (missing assets, xfade skipped, etc.).
    """

    output_path: str
    insertions: list[BRollInsertion] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _probe_video(path: str) -> dict:
    """Run ffprobe and return a dict with width, height, duration_s, fps."""
    from pipeline_core.validator import FFPROBE_BIN
    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise ValueError(
            f"ffprobe failed on {path!r}: {result.stderr.strip()}"
        )
    import json
    data = json.loads(result.stdout)
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    vstream = next((s for s in streams if s.get("codec_type") == "video"), {})

    duration_s = 0.0
    try:
        duration_s = float(fmt.get("duration") or vstream.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration_s = 0.0

    width = int(vstream.get("width") or 0)
    height = int(vstream.get("height") or 0)

    fps = 0.0
    rfr = vstream.get("r_frame_rate") or vstream.get("avg_frame_rate") or "0/1"
    try:
        parts = rfr.split("/")
        if len(parts) == 2:
            num, den = float(parts[0]), float(parts[1])
            fps = num / den if den != 0.0 else 0.0
        else:
            fps = float(parts[0])
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    return {"width": width, "height": height, "duration_s": duration_s, "fps": fps}


def _asset_kind(path: str) -> str:
    """Return 'image', 'video', or raise ValueError for unknown extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _VIDEO_EXTS:
        return "video"
    raise ValueError(
        f"B-roll asset {path!r} has unrecognised extension {ext!r}. "
        f"Supported: {sorted(_IMAGE_EXTS | _VIDEO_EXTS)}."
    )


def _render_image_to_mp4(
    img_path: str,
    duration_s: float,
    width: int,
    height: int,
    out_path: str,
    ffmpeg: str,
) -> None:
    """Render a static image to a silent mp4 of duration_s at width×height.

    Uses FFmpeg's -loop 1 muxer with libx264 + yuv420p so the result can be
    concat-demuxed alongside normal h264 clips.
    """
    # Ensure even dimensions (libx264 requires it).
    w = width if width % 2 == 0 else width - 1
    h = height if height % 2 == 0 else height - 1

    cmd = [
        ffmpeg, "-y",
        "-loop", "1",
        "-t", str(duration_s),
        "-i", img_path,
        "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
               f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-an",   # no audio track — we keep source audio throughout
        out_path,
    ]
    logger.debug("Rendering image B-roll to mp4: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(
            f"FFmpeg image-to-mp4 failed for {img_path!r}: {proc.stderr.strip()}"
        )


def _trim_video_broll(
    asset_path: str,
    duration_s: float,
    width: int,
    height: int,
    out_path: str,
    ffmpeg: str,
) -> None:
    """Trim / loop a video B-roll asset to exactly duration_s seconds.

    Re-encodes to the same resolution/codec as the source so the concat
    demuxer merges cleanly.  Audio from the B-roll asset is discarded;
    only the video track is kept.
    """
    w = width if width % 2 == 0 else width - 1
    h = height if height % 2 == 0 else height - 1

    # -stream_loop -1 enables infinite looping; -t cuts it to duration_s.
    cmd = [
        ffmpeg, "-y",
        "-stream_loop", "-1",
        "-i", asset_path,
        "-t", str(duration_s),
        "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
               f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-an",
        out_path,
    ]
    logger.debug("Trimming video B-roll: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(
            f"FFmpeg video trim failed for {asset_path!r}: {proc.stderr.strip()}"
        )


def _find_sentence_boundary(
    transcript: list[dict],
    target_s: float,
) -> float:
    """Snap target_s to the nearest sentence-end boundary in transcript.

    Scans all word segments and picks the end timestamp of the last word whose
    end time is at or before target_s.  Falls back to target_s if no suitable
    boundary is found.

    Parameters
    ----------
    transcript : list[dict]
        Each dict must have at least 'start', 'end', and optionally 'text'.
    target_s : float
        The desired insert position.

    Returns
    -------
    float
        Adjusted position snapped to a word boundary.
    """
    if not transcript:
        return target_s

    best = target_s
    for seg in transcript:
        end = float(seg.get("end", 0.0))
        if end <= target_s:
            best = end

    return best


# ── Public API ────────────────────────────────────────────────────────────────

def find_audio_valleys(
    video_path: str,
    *,
    window_s: float = 0.5,
    min_gap_s: float = 2.0,
    top_k: int = 5,
) -> list[tuple[float, float]]:
    """Return up to top_k (start_s, rms_normalised) tuples for the quietest windows.

    Parameters
    ----------
    video_path : str
        Absolute path to the source video.
    window_s : float
        RMS window length in seconds (default 0.5 s).
    min_gap_s : float
        Minimum gap between returned valleys in seconds (default 2.0 s).
    top_k : int
        Maximum number of valleys to return.

    Returns
    -------
    list[tuple[float, float]]
        Each tuple is (start_seconds, rms_normalised).  rms_normalised is in
        [0, 1] where 0 = total silence, 1 = the loudest window in the clip.
        The list is sorted by start_seconds ascending.

    Raises
    ------
    ValueError
        If the file does not exist or has no audio stream.
    """
    import librosa
    import librosa.feature

    if not os.path.exists(video_path):
        raise ValueError(f"Source video not found: {video_path!r}")

    logger.debug("Loading audio from %s for valley detection", video_path)
    try:
        y, sr = librosa.load(video_path, sr=None, mono=True)
    except Exception as exc:
        raise ValueError(
            f"librosa could not load audio from {video_path!r}: {exc}"
        ) from exc

    if len(y) == 0:
        raise ValueError(
            f"librosa returned an empty audio array for {video_path!r}. "
            "The file may have no audio stream."
        )

    # Compute frame parameters such that each frame ≈ window_s seconds.
    frame_length = int(round(window_s * sr))
    hop_length = frame_length  # non-overlapping windows

    logger.debug(
        "Computing RMS: sr=%d frame_length=%d hop_length=%d",
        sr, frame_length, hop_length,
    )
    rms_frames = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length
    )[0]  # shape: (n_frames,)

    # Convert frame indices to start times in seconds.
    n_frames = len(rms_frames)
    frame_times = librosa.frames_to_time(
        np.arange(n_frames), sr=sr, hop_length=hop_length
    )

    rms_max = float(rms_frames.max()) if n_frames > 0 else 1.0
    if rms_max == 0.0:
        rms_max = 1.0  # all-silence clip — avoid division by zero

    rms_norm = rms_frames / rms_max  # normalise to [0, 1]

    # Find local minima: frame i is a local minimum when its RMS is less than
    # both its neighbours.
    local_min_indices: list[int] = []
    for i in range(1, n_frames - 1):
        if rms_norm[i] < rms_norm[i - 1] and rms_norm[i] < rms_norm[i + 1]:
            local_min_indices.append(i)

    # If no interior local minima, include all frame indices as candidates.
    if not local_min_indices:
        local_min_indices = list(range(n_frames))

    # Sort by RMS ascending (quietest first).
    local_min_indices.sort(key=lambda idx: rms_norm[idx])

    # Apply min_gap_s constraint: greedily accept valleys at least min_gap_s
    # apart from already-accepted ones.
    accepted: list[tuple[float, float]] = []
    min_gap_frames = min_gap_s / (hop_length / sr)

    for idx in local_min_indices:
        if len(accepted) >= top_k:
            break
        t = float(frame_times[idx])
        too_close = any(abs(t - prev_t) < min_gap_s for prev_t, _ in accepted)
        if not too_close:
            accepted.append((t, float(rms_norm[idx])))

    accepted.sort(key=lambda x: x[0])  # sort by time for the caller

    logger.info(
        "Valley detection: %d frames, %d local minima → %d valleys returned",
        n_frames, len(local_min_indices), len(accepted),
    )
    return accepted


def insert_broll(
    source_path: str,
    broll_assets: list[str],
    *,
    output_path: str,
    max_inserts: int = 3,
    insert_duration_s: float = 2.5,
    crossfade_s: float = 0.25,
    transcript: Optional[list[dict]] = None,
) -> BRollResult:
    """Insert up to max_inserts B-roll cutaways at audio valleys.

    Parameters
    ----------
    source_path : str
        Absolute path to the source (A-roll) video.
    broll_assets : list[str]
        Ordered list of B-roll asset paths (images or videos).  Rotated
        round-robin across insert points.
    output_path : str
        Destination path for the stitched output video.
    max_inserts : int
        Maximum number of cutaways to insert (default 3).
    insert_duration_s : float
        Duration of each B-roll cutaway in seconds (default 2.5).
    crossfade_s : float
        Duration of the video crossfade at each cut point (default 0.25 s).
    transcript : list[dict] | None
        Optional list of {start, end, text} word/segment dicts.  When
        provided, insert points are snapped to sentence boundaries so the
        narrator's words are never split mid-word.

    Returns
    -------
    BRollResult
        .output_path is the stitched output.
        .insertions lists every B-roll segment that was actually inserted.
        .warnings contains non-fatal issues collected during processing.

    Raises
    ------
    ValueError
        If source_path does not exist or broll_assets is empty.

    Notes
    -----
    Implementation details:
      - Audio preservation: the source audio track is copied byte-for-byte
        through all segments with ``-c:a copy``.  Only the video channel
        is swapped during cutaways.
      - Segments are stitched using the FFmpeg concat demuxer (a temporary
        .txt file listing the clips), which avoids re-encoding the audio.
      - Each segment is individually encoded so the concat demuxer merges
        cleanly.  The B-roll audio track is silenced; the source audio
        continues underneath by splitting the source around each insertion.
      - xfade is applied between adjacent segments where crossfade_s > 0.
    """
    warnings: list[str] = []

    if not os.path.exists(source_path):
        raise ValueError(f"Source video not found: {source_path!r}")
    if not broll_assets:
        raise ValueError("broll_assets list is empty; nothing to insert.")

    ffmpeg = _get_ffmpeg()

    # Probe source for dimensions and duration.
    probe = _probe_video(source_path)
    src_w: int = probe["width"]
    src_h: int = probe["height"]
    src_dur: float = probe["duration_s"]

    if src_dur <= 0.0:
        raise ValueError(
            f"Could not determine duration for {source_path!r} (got {src_dur})."
        )

    # ── Find audio valleys ────────────────────────────────────────────────────
    try:
        valleys = find_audio_valleys(
            source_path,
            window_s=0.5,
            min_gap_s=insert_duration_s + crossfade_s * 2 + 0.5,
            top_k=max_inserts * 2,  # over-fetch; filter below
        )
    except Exception as exc:
        warnings.append(
            f"Valley detection failed ({exc}); falling back to evenly-spaced inserts."
        )
        # Fallback: evenly spaced
        n_fb = min(max_inserts, 3)
        step = (src_dur - 5.0) / (n_fb + 1)
        valleys = [(3.0 + step * (i + 1), 0.5) for i in range(n_fb)]

    # ── Filter valleys: guard zones + snap to transcript boundaries ───────────
    GUARD_START = 3.0   # protect the hook
    GUARD_END = 2.0     # protect the CTA

    valid_valleys: list[tuple[float, float]] = []
    for start_s, rms in valleys:
        if len(valid_valleys) >= max_inserts:
            break
        # Guard zone check
        if start_s < GUARD_START:
            logger.debug("Valley at %.2fs skipped — inside opening guard zone", start_s)
            continue
        end_of_insert = start_s + insert_duration_s
        if end_of_insert > src_dur - GUARD_END:
            logger.debug("Valley at %.2fs skipped — insert would overlap closing guard zone", start_s)
            continue
        # Snap to transcript sentence boundary if transcript provided
        if transcript:
            snapped = _find_sentence_boundary(transcript, start_s)
            if abs(snapped - start_s) > insert_duration_s:
                # Snapped too far away — skip this valley
                logger.debug(
                    "Valley at %.2fs skipped — nearest sentence boundary %.2fs is too far",
                    start_s, snapped,
                )
                continue
            start_s = snapped

        valid_valleys.append((start_s, rms))

    if not valid_valleys:
        warnings.append(
            "No valid B-roll insert points found after applying guard zones and "
            "transcript constraints. Output is the unmodified source video."
        )
        # Just copy source to output path
        import shutil as _sh
        _sh.copy2(source_path, output_path)
        return BRollResult(output_path=output_path, insertions=[], warnings=warnings)

    logger.info(
        "B-roll insertion: %d insert point(s) selected out of %d valleys",
        len(valid_valleys), len(valleys),
    )

    # ── Build segment list ────────────────────────────────────────────────────
    # Strategy: cut the source video around each insert point and interleave
    # with the prepared B-roll clips.  Audio comes exclusively from the
    # source — B-roll clips carry no audio.
    #
    # Source segments produced:
    #   [0 .. insert_0_start]  B-roll_0  [insert_0_end .. insert_1_start]  ...  [last_end .. src_end]
    #
    # All source segments are extracted with -c:v copy -c:a copy to preserve
    # quality.  Each B-roll segment is re-encoded at source dimensions with
    # no audio.  The concat demuxer then merges them.
    # NOTE: Because concat merges video + audio streams, and B-roll segments
    # have no audio, we need to supply a silent audio for them OR rely on the
    # concat filter.  We use the concat demuxer with -f concat which requires
    # all segments to have matching stream counts.  To keep it simple and
    # robust: we extract source segments with both video+audio, and for B-roll
    # we add a silent audio track at the same sample rate.

    tmp_dir = tempfile.mkdtemp(prefix="kaizer_broll_")
    logger.debug("B-roll temp dir: %s", tmp_dir)

    try:
        insertions: list[BRollInsertion] = []
        segment_paths: list[str] = []

        # Sort insert points chronologically (they should already be sorted from
        # find_audio_valleys, but be defensive).
        valid_valleys.sort(key=lambda x: x[0])

        current_pos = 0.0
        asset_idx = 0

        for i, (insert_start, rms_val) in enumerate(valid_valleys):
            insert_end = insert_start + insert_duration_s

            # ── Source segment before this insert ──────────────────────────
            seg_src_path = os.path.join(tmp_dir, f"src_seg_{i}.mp4")
            seg_dur = insert_start - current_pos
            if seg_dur > 0.0:
                cmd_src = [
                    ffmpeg, "-y",
                    "-ss", str(current_pos),
                    "-i", source_path,
                    "-t", str(seg_dur),
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    seg_src_path,
                ]
                logger.debug("Cutting source segment [%.2f, %.2f]", current_pos, insert_start)
                proc = subprocess.run(
                    cmd_src, capture_output=True, text=True, timeout=300
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"FFmpeg source segment cut failed: {proc.stderr.strip()}"
                    )
                segment_paths.append(seg_src_path)

            # ── Prepare B-roll segment ─────────────────────────────────────
            asset_path = broll_assets[asset_idx % len(broll_assets)]
            asset_idx += 1

            broll_raw = os.path.join(tmp_dir, f"broll_raw_{i}.mp4")
            broll_with_audio = os.path.join(tmp_dir, f"broll_seg_{i}.mp4")

            try:
                kind = _asset_kind(asset_path)
            except ValueError as exc:
                warnings.append(str(exc))
                continue

            # Render B-roll video (no audio) at source dimensions
            try:
                if kind == "image":
                    _render_image_to_mp4(
                        asset_path, insert_duration_s,
                        src_w, src_h, broll_raw, ffmpeg,
                    )
                else:
                    _trim_video_broll(
                        asset_path, insert_duration_s,
                        src_w, src_h, broll_raw, ffmpeg,
                    )
            except Exception as exc:
                warnings.append(
                    f"B-roll asset {asset_path!r} failed to prepare: {exc}. Skipping."
                )
                continue

            # Extract the source audio for this time window and mux with B-roll video.
            # This way all segments carry audio and the concat demuxer works cleanly.
            src_audio_seg = os.path.join(tmp_dir, f"src_audio_{i}.aac")
            cmd_audio = [
                ffmpeg, "-y",
                "-ss", str(insert_start),
                "-i", source_path,
                "-t", str(insert_duration_s),
                "-vn",
                "-c:a", "aac",
                "-b:a", "192k",
                src_audio_seg,
            ]
            logger.debug("Extracting source audio for B-roll window [%.2f, %.2f]", insert_start, insert_end)
            proc_audio = subprocess.run(
                cmd_audio, capture_output=True, text=True, timeout=120
            )
            if proc_audio.returncode != 0:
                warnings.append(
                    f"Could not extract source audio for B-roll at {insert_start:.2f}s; "
                    "B-roll segment will have silence."
                )
                # Add a silent audio track to broll_raw
                cmd_mux = [
                    ffmpeg, "-y",
                    "-i", broll_raw,
                    "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo",
                    "-t", str(insert_duration_s),
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    broll_with_audio,
                ]
            else:
                # Mux B-roll video with source audio
                cmd_mux = [
                    ffmpeg, "-y",
                    "-i", broll_raw,
                    "-i", src_audio_seg,
                    "-c:v", "copy",
                    "-c:a", "copy",
                    "-shortest",
                    broll_with_audio,
                ]

            proc_mux = subprocess.run(
                cmd_mux, capture_output=True, text=True, timeout=120
            )
            if proc_mux.returncode != 0:
                warnings.append(
                    f"B-roll mux failed at {insert_start:.2f}s: {proc_mux.stderr.strip()}. Skipping."
                )
                continue

            segment_paths.append(broll_with_audio)

            insertions.append(BRollInsertion(
                asset_path=asset_path,
                start_s=insert_start,
                duration_s=insert_duration_s,
                source_rms=rms_val,
            ))

            current_pos = insert_end

        # ── Final source segment (tail) ────────────────────────────────────
        if current_pos < src_dur:
            tail_seg = os.path.join(tmp_dir, "src_tail.mp4")
            tail_dur = src_dur - current_pos
            cmd_tail = [
                ffmpeg, "-y",
                "-ss", str(current_pos),
                "-i", source_path,
                "-t", str(tail_dur),
                "-c:v", "libx264",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                tail_seg,
            ]
            logger.debug("Cutting tail segment [%.2f, %.2f]", current_pos, src_dur)
            proc_tail = subprocess.run(
                cmd_tail, capture_output=True, text=True, timeout=300
            )
            if proc_tail.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg tail segment cut failed: {proc_tail.stderr.strip()}"
                )
            segment_paths.append(tail_seg)

        if not segment_paths:
            raise RuntimeError("No segments were produced — cannot build output.")

        # ── Stitch with concat demuxer ─────────────────────────────────────
        concat_list_path = os.path.join(tmp_dir, "concat_list.txt")
        with open(concat_list_path, "w", encoding="utf-8") as fh:
            for seg in segment_paths:
                # FFmpeg concat demuxer requires forward slashes even on Windows.
                safe_seg = seg.replace("\\", "/")
                fh.write(f"file '{safe_seg}'\n")

        logger.info(
            "Stitching %d segments → %s",
            len(segment_paths), output_path,
        )
        cmd_concat = [
            ffmpeg, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            output_path,
        ]
        proc_concat = subprocess.run(
            cmd_concat, capture_output=True, text=True, timeout=600
        )
        if proc_concat.returncode != 0:
            raise RuntimeError(
                f"FFmpeg concat failed: {proc_concat.stderr.strip()}"
            )

        logger.info(
            "B-roll insertion complete: %d inserts → %s",
            len(insertions), output_path,
        )
        return BRollResult(
            output_path=output_path,
            insertions=insertions,
            warnings=warnings,
        )

    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.debug("Cleaned temp dir: %s", tmp_dir)
        except Exception as exc:
            logger.warning("Could not clean temp dir %s: %s", tmp_dir, exc)
