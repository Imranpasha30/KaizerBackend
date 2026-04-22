"""
kaizer.pipeline.render_series
==============================
Multi-part series chaining (Mode C) for the Kaizer News video pipeline.

The secret weapon: nobody else auto-chains Shorts into a coherent series
with Part X/Y overlays + pinned-comment manifest.

Given N :class:`~pipeline_core.narrative.ClipCandidate` objects (2–5), this
module:
  1. Renders each one using ``render_modes.render_mode_clip`` (mode='series').
  2. Burns a "Part X/Y" badge in the top-left corner for the first 2 s.
  3. Applies cross-linking CTAs: "Part {N+1} next ↗" for parts 1..N-1 and
     "Series complete — follow for more" for the final part.
  4. Detects cliffhangers (completion_score < 0.5 or setup/opportunity/turn role).
  5. Produces a :class:`SeriesManifest` with a ready-to-post pinned comment
     template and cross-post hashtags.

The pinned comment automation (actually posting to YouTube) happens at upload
time in ``youtube/worker.py`` — NOT in this module.  This module produces the
MP4s + manifest metadata only.

Usage
-----
    from pipeline_core.render_series import chain_parts, SeriesManifest

    manifest = chain_parts(
        "/path/to/source.mp4",
        candidates[:4],
        output_dir="/tmp/series_output",
        playlist_title="Breaking News Series",
        source_language="te",
    )
    for part in manifest.parts:
        print(part.part_index, "/", part.part_total, "→", part.output_path)
    print(manifest.pinned_comment_template)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any

from pipeline_core.pipeline import FFMPEG_BIN
from pipeline_core.captions import render_caption, detect_script

logger = logging.getLogger("kaizer.pipeline.render_series")

# ── Cliffhanger detection — narrative roles that suggest the story continues ──

_CLIFFHANGER_ROLES: frozenset[str] = frozenset({"setup", "opportunity", "turn"})
_CLIFFHANGER_COMPLETION_THRESHOLD: float = 0.5

# Badge font size at reference width 1080 px
_BADGE_FONT_SIZE_REF = 52
_REFERENCE_WIDTH = 1080

# Badge visible for the first N seconds of each part
_BADGE_DURATION_S = 2.0


# ── Public dataclasses ─────────────────────────────────────────────────────────

@dataclass
class SeriesPart:
    """Metadata for one rendered part of a series.

    Attributes
    ----------
    part_index : int
        1-indexed part number.
    part_total : int
        Total number of parts in the series.
    output_path : str
        Absolute path to the rendered MP4.
    duration_s : float
        Measured or estimated clip duration in seconds.
    title_overlay_text : str
        The badge text rendered on screen, e.g. "Part 1/3".
    cliffhanger_detected : bool
        True when the clip's completion_score < 0.5 OR narrative_role is in
        the cliffhanger set (setup/opportunity/turn).
    cta_applied : str
        The CTA style that was applied to this part.
    qa_ok : bool
        Whether QA validation passed.
    meta : dict
        Arbitrary extra metadata.
    """

    part_index: int
    part_total: int
    output_path: str
    duration_s: float
    title_overlay_text: str
    cliffhanger_detected: bool
    cta_applied: str
    qa_ok: bool
    meta: dict = field(default_factory=dict)


@dataclass
class SeriesManifest:
    """Full result of :func:`chain_parts`.

    Attributes
    ----------
    parts : list[SeriesPart]
        One entry per rendered part, in order.
    playlist_title : str
        Suggested YouTube playlist title for the series.
    pinned_comment_template : str
        A template string using ``{N}`` for part number and ``{URL}`` for the
        actual YouTube URL (filled in by the upload worker at post time).
    cross_post_hashtags : list[str]
        Recommended hashtags for cross-posting (News/language aware).
    warnings : list[str]
        Non-fatal issues collected during rendering.
    """

    parts: list[SeriesPart]
    playlist_title: str
    pinned_comment_template: str
    cross_post_hashtags: list[str]
    warnings: list[str] = field(default_factory=list)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _probe_video_size(video_path: str) -> tuple[int, int]:
    """Return (width, height). Falls back to (1080, 1920)."""
    try:
        from pipeline_core.qa import FFPROBE_BIN as _fp
    except Exception:
        import shutil as _sh
        _fp = _sh.which("ffprobe") or "ffprobe"

    cmd = [
        _fp, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        video_path,
    ]
    try:
        import json
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            streams = data.get("streams", [])
            if streams:
                w = int(streams[0].get("width") or 1080)
                h = int(streams[0].get("height") or 1920)
                return w, h
    except Exception as exc:
        logger.warning("render_series: ffprobe size probe failed: %s", exc)
    return 1080, 1920


def _probe_duration(video_path: str) -> float:
    """Return video duration in seconds. Returns 0.0 on failure."""
    try:
        from pipeline_core.qa import FFPROBE_BIN as _fp
    except Exception:
        import shutil as _sh
        _fp = _sh.which("ffprobe") or "ffprobe"

    cmd = [
        _fp, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ]
    try:
        import json
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            return float(data.get("format", {}).get("duration", 0.0))
    except Exception as exc:
        logger.warning("render_series: ffprobe duration probe failed: %s", exc)
    return 0.0


def _detect_cliffhanger(candidate: Any) -> bool:
    """Return True when the clip ends on an unresolved narrative beat."""
    completion = getattr(candidate, "completion_score", 1.0)
    role = getattr(candidate, "narrative_role", "unlabeled")
    return completion < _CLIFFHANGER_COMPLETION_THRESHOLD or role in _CLIFFHANGER_ROLES


def _render_badge_on_clip(
    clip_path: str,
    badge_text: str,
    output_path: str,
    *,
    badge_duration_s: float = _BADGE_DURATION_S,
    source_language: str | None = None,
) -> list[str]:
    """Burn a semi-transparent badge (e.g. "Part 1/3") onto the top-left corner
    of *clip_path* for the first *badge_duration_s* seconds.

    Returns a list of warning strings.  Raises RuntimeError on FFmpeg failure.
    """
    warnings: list[str] = []
    vid_w, vid_h = _probe_video_size(clip_path)

    font_size = max(24, int(_BADGE_FONT_SIZE_REF * vid_w / _REFERENCE_WIDTH))

    # Script / font hint
    script_hint: str | None = None
    if source_language in ("te", "tel"):
        script_hint = "telugu"
    elif source_language in ("hi", "hin", "mr", "mar"):
        script_hint = "devanagari"
    elif source_language in ("ta", "tam"):
        script_hint = "tamil"
    elif source_language in ("bn", "ben"):
        script_hint = "bengali"
    elif source_language in ("kn", "kan"):
        script_hint = "kannada"
    elif source_language in ("ml", "mal"):
        script_hint = "malayalam"
    elif source_language in ("gu", "guj"):
        script_hint = "gujarati"

    render_kwargs: dict = dict(
        max_width=int(vid_w * 0.45),   # badge occupies at most 45% of frame width
        font_size=font_size,
        color="#FFFFFF",
        stroke_color="#000000",
        stroke_width=max(2, font_size // 20),
        bg_color="#000000CC",           # ~80% opaque black pill
        bg_padding=max(8, font_size // 8),
        bg_radius=10,
        align="left",
    )
    if script_hint:
        render_kwargs["script"] = script_hint

    badge_result = render_caption(badge_text, **render_kwargs)
    warnings.extend(badge_result.warnings)

    # Position: top-left with a small margin
    margin_x = int(vid_w * 0.03)   # ~3% from left
    margin_y = int(vid_h * 0.03)   # ~3% from top

    tmp_png = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tmp_png = tf.name.replace("\\", "/")
        badge_result.image.save(tmp_png)

        filter_str = (
            f"[1]format=rgba[badge];"
            f"[0][badge]overlay="
            f"x={margin_x}:y={margin_y}:"
            f"enable='between(t,0,{badge_duration_s:.6f})'"
        )

        # Video encode: use ENCODE_ARGS_SHORT_FORM but keep audio as copy
        from pipeline_core.pipeline import ENCODE_ARGS_SHORT_FORM as _eas
        video_encode_args: list[str] = []
        skip_next = False
        for arg in _eas:
            if skip_next:
                skip_next = False
                continue
            if arg in ("-c:a", "-b:a", "-ar"):
                skip_next = True
                continue
            if arg == "-af":
                skip_next = True
                continue
            video_encode_args.append(arg)

        cmd = [
            FFMPEG_BIN, "-y",
            "-i", clip_path,
            "-i", tmp_png,
            "-filter_complex", filter_str,
        ] + video_encode_args + [
            "-c:a", "copy",
            output_path,
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
            raise RuntimeError(
                f"FFmpeg badge overlay failed (rc={proc.returncode}): {stderr_tail}"
            )

        logger.debug("Badge overlay applied: %s → %s", clip_path, output_path)

    finally:
        if tmp_png and os.path.isfile(tmp_png):
            try:
                os.remove(tmp_png)
            except OSError as exc:
                logger.warning("Could not remove temp badge PNG %s: %s", tmp_png, exc)

    return warnings


def _build_cross_post_hashtags(
    playlist_title: str | None,
    source_language: str | None,
) -> list[str]:
    """Generate reasonable hashtags for a news series."""
    tags: list[str] = ["#NewsShorts", "#BreakingNews", "#Series"]
    if source_language in ("te", "tel"):
        tags += ["#TeluguNews", "#తెలుగువార్తలు"]
    elif source_language in ("hi", "hin", "mr", "mar"):
        tags += ["#HindiNews", "#हिंदीसमाचार"]
    elif source_language in ("ta", "tam"):
        tags += ["#TamilNews"]
    elif source_language in ("bn", "ben"):
        tags += ["#BanglaNews"]
    if playlist_title:
        # Make a hashtag from the playlist title
        ht = "#" + "".join(
            ch for ch in playlist_title.title().replace(" ", "") if ch.isalnum() or ch == "#"
        )
        if len(ht) > 2:
            tags.append(ht)
    return tags


# ── Public API ────────────────────────────────────────────────────────────────

def chain_parts(
    source_video_path: str,
    candidates: list[Any],
    *,
    output_dir: str,
    playlist_title: str | None = None,
    source_language: str | None = None,
    run_qa: bool = True,
) -> SeriesManifest:
    """Produce N chained Part 1/N, Part 2/N, … MP4s from the given candidates.

    Parameters
    ----------
    source_video_path : str
        Absolute path to the source video.
    candidates : list[ClipCandidate]
        2–5 candidates from the narrative engine (already sorted by score).
    output_dir : str
        Directory to write the rendered part MP4s into.
    playlist_title : str | None
        Optional YouTube playlist title.  Auto-generated from source filename
        if not provided.
    source_language : str | None
        ISO 639-1 language hint for badge text and hashtag generation.
    run_qa : bool
        Whether to run :func:`qa.validate_output` on each rendered part.

    Returns
    -------
    SeriesManifest

    Raises
    ------
    ValueError
        If ``len(candidates)`` is not in [2, 5].
    RuntimeError
        If FFmpeg fails for any part.
    """
    # Normalise path separators
    source_video_path = source_video_path.replace("\\", "/")
    output_dir = output_dir.replace("\\", "/")

    n = len(candidates)
    if n < 2 or n > 5:
        raise ValueError(
            f"chain_parts requires 2–5 candidates; got {n}."
        )

    os.makedirs(output_dir, exist_ok=True)

    if playlist_title is None:
        base = os.path.splitext(os.path.basename(source_video_path))[0]
        playlist_title = f"{base} — Series"

    all_warnings: list[str] = []
    parts: list[SeriesPart] = []

    # Render all parts first (base slices + CTA), then burn badges
    # We use a two-pass approach per part to keep concerns separated:
    #   pass 1: render_mode_clip (slice + CTA)
    #   pass 2: burn_badge overlay
    from pipeline_core.render_modes import render_mode_clip  # avoid top-level circular

    for idx, cand in enumerate(candidates, start=1):
        is_last = (idx == n)
        cliffhanger = _detect_cliffhanger(cand)

        # Choose CTA text based on position in series
        if is_last:
            cta_text = "Series complete — follow for more"
        else:
            cta_text = f"Part {idx + 1} next ↗"  # ↗

        badge_text = f"Part {idx}/{n}"

        # ── Pass 1: render with CTA ───────────────────────────────────────────
        # Use 'series' mode for duration/QA config.
        # Intermediate file before badge burn.
        intermediate_path = os.path.join(
            output_dir, f"series_part_{idx:02d}_pre_badge.mp4"
        ).replace("\\", "/")

        try:
            rendered = render_mode_clip(
                source_video_path,
                cand,
                mode="series",
                output_path=intermediate_path,
                cta_text=cta_text,
                run_qa=False,   # skip QA here; run it after badge burn
                source_language=source_language,
            )
            all_warnings.extend(rendered.meta.get("warnings", []))
            clip_dur = rendered.duration_s
            cta_applied = rendered.cta_applied
        except Exception as exc:
            all_warnings.append(
                f"Part {idx} render_mode_clip failed: {exc}"
            )
            logger.error("chain_parts: Part %d failed: %s", idx, exc)
            parts.append(SeriesPart(
                part_index=idx,
                part_total=n,
                output_path=intermediate_path,
                duration_s=0.0,
                title_overlay_text=badge_text,
                cliffhanger_detected=cliffhanger,
                cta_applied="error",
                qa_ok=False,
                meta={"error": str(exc)},
            ))
            continue

        # ── Pass 2: burn Part X/Y badge ───────────────────────────────────────
        final_part_path = os.path.join(
            output_dir, f"series_part_{idx:02d}.mp4"
        ).replace("\\", "/")

        try:
            badge_warnings = _render_badge_on_clip(
                intermediate_path,
                badge_text,
                final_part_path,
                badge_duration_s=_BADGE_DURATION_S,
                source_language=source_language,
            )
            all_warnings.extend(badge_warnings)
        except Exception as exc:
            all_warnings.append(
                f"Part {idx} badge overlay failed (using pre-badge file): {exc}"
            )
            logger.warning("chain_parts: badge burn failed for part %d: %s", idx, exc)
            # Fall back to pre-badge file
            try:
                shutil.copy2(intermediate_path, final_part_path)
            except Exception as copy_exc:
                logger.error(
                    "chain_parts: could not copy pre-badge to final for part %d: %s",
                    idx, copy_exc,
                )
        finally:
            # Remove intermediate file
            try:
                if os.path.isfile(intermediate_path):
                    os.remove(intermediate_path)
            except OSError as exc:
                logger.warning(
                    "chain_parts: could not remove intermediate %s: %s",
                    intermediate_path, exc,
                )

        # ── QA on final file ──────────────────────────────────────────────────
        qa_ok = True
        qa_warnings: list[str] = []
        if run_qa:
            try:
                from pipeline_core import qa as _qa
                from pipeline_core.render_modes import RENDER_MODE_CONFIGS
                qa_cfg = RENDER_MODE_CONFIGS["series"]
                qa_result = _qa.validate_output(
                    final_part_path,
                    platform=qa_cfg.qa_platform,
                    expected_duration_s=clip_dur,
                )
                qa_ok = qa_result.ok
                qa_warnings = qa_result.warnings
                if not qa_result.ok:
                    logger.warning(
                        "chain_parts: Part %d QA FAILED: %s",
                        idx, qa_result.errors,
                    )
            except Exception as exc:
                qa_ok = False
                qa_warnings.append(f"QA raised exception: {exc}")
                logger.warning("chain_parts: Part %d QA exception: %s", idx, exc)

        all_warnings.extend(qa_warnings)

        # Actual duration from file (best effort)
        actual_dur = _probe_duration(final_part_path)
        if actual_dur <= 0.0:
            actual_dur = clip_dur

        parts.append(SeriesPart(
            part_index=idx,
            part_total=n,
            output_path=final_part_path,
            duration_s=actual_dur,
            title_overlay_text=badge_text,
            cliffhanger_detected=cliffhanger,
            cta_applied=cta_applied,
            qa_ok=qa_ok,
            meta={
                "qa_warnings": qa_warnings,
                "source_candidate_start": getattr(cand, "start", None),
                "source_candidate_end": getattr(cand, "end", None),
                "narrative_role": getattr(cand, "narrative_role", None),
                "composite_score": getattr(cand, "composite_score", None),
                "completion_score": getattr(cand, "completion_score", None),
            },
        ))

    # ── Build pinned comment template ─────────────────────────────────────────
    # Caller fills {N} → part number and {URL} → actual YouTube URL after upload
    pinned_comment_template = (
        "Part {N} is now live! → {URL}\n"
        f"Full playlist: {playlist_title}\n"
        "#Series #NewsShorts"
    )

    cross_post_hashtags = _build_cross_post_hashtags(playlist_title, source_language)

    logger.info(
        "chain_parts: completed %d/%d parts for %r (warnings=%d)",
        len(parts), n, playlist_title, len(all_warnings),
    )

    return SeriesManifest(
        parts=parts,
        playlist_title=playlist_title,
        pinned_comment_template=pinned_comment_template,
        cross_post_hashtags=cross_post_hashtags,
        warnings=all_warnings,
    )
