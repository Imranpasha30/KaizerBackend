"""
kaizer.pipeline.render_modes
=============================
Per-mode render dispatch for the Kaizer News video pipeline.

Given a source video + ClipCandidate + mode, renders a mode-appropriate MP4
using the Narrative Engine's candidates as clip boundaries.

The six modes are:
  standalone      — self-contained clip, soft-follow CTA
  trailer         — hook-heavy teaser, related-video CTA
  series          — long-form part of a series, next-part CTA
  promo           — very short promo clip, URL overlay CTA
  highlight       — pure highlight, no CTA
  full_narrative  — full narrative arc up to 3 min, soft-follow CTA

Usage
-----
    from pipeline_core.render_modes import render_mode_clip, RENDER_MODE_CONFIGS

    clip = render_mode_clip(
        "/path/to/source.mp4",
        candidate,                  # ClipCandidate from narrative.py
        mode="standalone",
        output_path="/path/to/output.mp4",
        run_qa=True,
    )
    print(clip.qa_ok, clip.duration_s)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any

from pipeline_core.pipeline import FFMPEG_BIN, ENCODE_ARGS_SHORT_FORM
from pipeline_core.validator import validate_input
from pipeline_core import cta_overlay as _cta_mod
from pipeline_core.cta_overlay import apply_cta

logger = logging.getLogger("kaizer.pipeline.render_modes")


# ── Mode configuration registry ───────────────────────────────────────────────

@dataclass
class RenderModeConfig:
    """Static configuration for one render mode.

    Attributes
    ----------
    target_dur_s : float
        Ideal clip duration in seconds (informational; clamp uses min/max).
    min_dur_s : float
        Minimum allowed clip duration (clips shorter than this are padded by
        keeping the source boundary and accepting a shorter output).
    max_dur_s : float
        Maximum allowed clip duration (clips longer are truncated at this limit).
    cta_style : str
        Which :func:`cta_overlay.apply_cta` style to apply.
    cta_text_default : str
        Fallback CTA text when caller does not supply one.
    importance_weight : float
        Mirrors the narrative weight used in composite scoring (documentation).
    hook_weight : float
        Mirrors the hook weight (documentation).
    completion_weight : float
        Mirrors the completion weight (documentation).
    qa_platform : str
        Platform string passed to :func:`qa.validate_output`.
    """

    target_dur_s: float
    min_dur_s: float
    max_dur_s: float
    cta_style: str
    cta_text_default: str
    importance_weight: float
    hook_weight: float
    completion_weight: float
    qa_platform: str


# Registry — the SIX render modes
RENDER_MODE_CONFIGS: dict[str, RenderModeConfig] = {
    "standalone": RenderModeConfig(
        target_dur_s=50,
        min_dur_s=45,
        max_dur_s=60,
        cta_style="soft_follow",
        cta_text_default="Follow for more",
        importance_weight=0.4,
        hook_weight=0.3,
        completion_weight=0.3,
        qa_platform="youtube_short",
    ),
    "trailer": RenderModeConfig(
        target_dur_s=40,
        min_dur_s=30,
        max_dur_s=50,
        cta_style="related_video",
        cta_text_default="Watch full video on my channel →",
        importance_weight=0.25,
        hook_weight=0.5,
        completion_weight=0.25,
        qa_platform="youtube_short",
    ),
    "series": RenderModeConfig(
        target_dur_s=70,
        min_dur_s=45,
        max_dur_s=90,
        cta_style="next_part",
        cta_text_default="Part continues →",
        importance_weight=0.4,
        hook_weight=0.3,
        completion_weight=0.3,
        qa_platform="youtube_short",
    ),
    "promo": RenderModeConfig(
        target_dur_s=20,
        min_dur_s=15,
        max_dur_s=25,
        cta_style="url_overlay",
        cta_text_default="Watch full video ↗",
        importance_weight=0.25,
        hook_weight=0.5,
        completion_weight=0.25,
        qa_platform="youtube_short",
    ),
    "highlight": RenderModeConfig(
        target_dur_s=45,
        min_dur_s=30,
        max_dur_s=60,
        cta_style="none",
        cta_text_default="",
        importance_weight=0.4,
        hook_weight=0.3,
        completion_weight=0.3,
        qa_platform="youtube_short",
    ),
    "full_narrative": RenderModeConfig(
        target_dur_s=120,
        min_dur_s=60,
        max_dur_s=180,
        cta_style="soft_follow",
        cta_text_default="Follow for more",
        importance_weight=0.4,
        hook_weight=0.3,
        completion_weight=0.3,
        qa_platform="youtube_short",
    ),
    # bulletin = the long-form 1–2 hr stitched output for YouTube Full.
    # Per-story values (target/min/max) describe each individual story
    # that the bulletin is composed of; the bulletin total is governed
    # by ``bulletin_stitcher.DEFAULT_TARGET_TOTAL_MIN`` etc. CTAs are
    # disabled per-story because the bulletin has its own outro CTA in
    # a later phase.
    "bulletin": RenderModeConfig(
        target_dur_s=90,
        min_dur_s=30,
        max_dur_s=240,
        cta_style="none",
        cta_text_default="",
        importance_weight=0.5,
        hook_weight=0.2,
        completion_weight=0.3,
        qa_platform="youtube_full",
    ),
}


# ── Public dataclass ──────────────────────────────────────────────────────────

@dataclass
class RenderedClip:
    """Result of a single :func:`render_mode_clip` call.

    Attributes
    ----------
    mode : str
        The render mode used (e.g. 'standalone').
    output_path : str
        Absolute path to the rendered MP4.
    duration_s : float
        Measured or estimated clip duration in seconds.
    clip_candidate : object
        The source :class:`~pipeline_core.narrative.ClipCandidate`.
    cta_applied : str
        The CTA style that was actually applied.
    qa_ok : bool
        Whether :func:`qa.validate_output` passed (True when run_qa=False too).
    qa_warnings : list[str]
        Warnings from QA (empty when run_qa=False).
    meta : dict
        Additional metadata (timings, composite_score, warnings, etc.).
    """

    mode: str
    output_path: str
    duration_s: float
    clip_candidate: Any
    cta_applied: str
    qa_ok: bool
    qa_warnings: list[str]
    meta: dict = field(default_factory=dict)


# ── Internal helpers ──────────────────────────────────────────────────────────

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
        logger.warning("render_modes: ffprobe duration probe failed: %s", exc)
    return 0.0


def _ffmpeg_slice(
    source_path: str,
    start_s: float,
    duration_s: float,
    output_path: str,
) -> None:
    """Slice *source_path* from *start_s* for *duration_s* seconds into
    *output_path*, re-encoding with ENCODE_ARGS_SHORT_FORM.

    Raises
    ------
    RuntimeError
        If FFmpeg exits with a non-zero return code.
    """
    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", f"{start_s:.6f}",
        "-i", source_path,
        "-t", f"{duration_s:.6f}",
    ] + ENCODE_ARGS_SHORT_FORM + [output_path]

    logger.debug("FFmpeg slice: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
        raise RuntimeError(
            f"FFmpeg slice failed (rc={proc.returncode}): {stderr_tail}"
        )


# ── Public API ────────────────────────────────────────────────────────────────

def render_mode_clip(
    source_video_path: str,
    candidate: Any,
    *,
    mode: str,
    output_path: str,
    cta_text: str | None = None,
    sub_text: str | None = None,
    run_qa: bool = True,
    source_language: str | None = None,
) -> RenderedClip:
    """Render one clip in the given mode from a single :class:`ClipCandidate`.

    Parameters
    ----------
    source_video_path : str
        Absolute path to the source video.
    candidate : ClipCandidate
        The narrative candidate (supplies .start, .end, .composite_score, etc.).
    mode : str
        One of the six mode keys in :data:`RENDER_MODE_CONFIGS`.
    output_path : str
        Absolute path for the output MP4.
    cta_text : str | None
        Overrides the mode's ``cta_text_default``.
    sub_text : str | None
        Optional second CTA line (URL, etc.).
    run_qa : bool
        If True, runs :func:`qa.validate_output` and populates qa_ok/qa_warnings.
    source_language : str | None
        ISO 639-1 language hint for CTA localisation.

    Returns
    -------
    RenderedClip

    Raises
    ------
    ValueError
        If *mode* is not in RENDER_MODE_CONFIGS or input validation fails hard.
    RuntimeError
        If FFmpeg fails.
    """
    # Normalise path separators
    source_video_path = source_video_path.replace("\\", "/")
    output_path = output_path.replace("\\", "/")

    # ── Step 1: Validate mode ─────────────────────────────────────────────────
    if mode not in RENDER_MODE_CONFIGS:
        raise ValueError(
            f"Unknown render mode {mode!r}. "
            f"Valid modes: {sorted(RENDER_MODE_CONFIGS.keys())}."
        )
    cfg = RENDER_MODE_CONFIGS[mode]

    all_warnings: list[str] = []

    # ── Step 2: Validate source input ─────────────────────────────────────────
    val_result = validate_input(source_video_path)
    if not val_result.ok:
        raise ValueError(
            f"Source video failed input validation: {val_result.errors}"
        )
    if val_result.warnings:
        logger.warning(
            "render_mode_clip: input warnings for %s: %s",
            source_video_path, val_result.warnings,
        )
        all_warnings.extend(val_result.warnings)

    # ── Step 3: Compute + clamp clip duration ─────────────────────────────────
    raw_dur = float(candidate.end) - float(candidate.start)

    # Hard upper cap: clips longer than max_dur_s are truncated.
    # Soft lower bound: clips shorter than min_dur_s emit a warning but are NOT
    # artificially extended — the candidate's natural span is respected.  Forcing
    # extension would push start+dur past the candidate's narrative boundary and
    # produce content the engine did not select.
    clamped_dur = min(cfg.max_dur_s, raw_dur)

    if raw_dur > cfg.max_dur_s:
        all_warnings.append(
            f"Clip duration {raw_dur:.2f}s exceeds max_dur_s={cfg.max_dur_s}s "
            f"for mode {mode!r}; truncating to {clamped_dur:.2f}s."
        )
    elif raw_dur < cfg.min_dur_s:
        all_warnings.append(
            f"Clip duration {raw_dur:.2f}s is below min_dur_s={cfg.min_dur_s}s "
            f"for mode {mode!r} (using natural candidate span)."
        )

    clip_start = float(candidate.start)

    # Cap clamped_dur to the available source material from clip_start onward.
    # This prevents requesting more frames than the source contains, which would
    # produce an over-long output or FFmpeg warnings.
    source_dur = val_result.meta.get("duration_s", 0.0)
    if source_dur > 0.0:
        available = max(0.0, source_dur - clip_start)
        if clamped_dur > available:
            all_warnings.append(
                f"Clamped duration {clamped_dur:.2f}s exceeds available source "
                f"({available:.2f}s from t={clip_start:.2f}s); trimming to available."
            )
            clamped_dur = available

    # ── Step 4: FFmpeg-slice to temp file ─────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="kaizer_render_")
    try:
        slice_path = os.path.join(tmp_dir, "slice.mp4").replace("\\", "/")

        _ffmpeg_slice(
            source_video_path,
            clip_start,
            clamped_dur,
            slice_path,
        )
        logger.debug(
            "Slice: mode=%s start=%.2f dur=%.2f → %s",
            mode, clip_start, clamped_dur, slice_path,
        )

        # ── Step 5: Apply CTA (or just copy if style is 'none') ───────────────
        effective_cta_text = cta_text or cfg.cta_text_default

        cta_result = apply_cta(
            slice_path,
            cta_style=cfg.cta_style,
            output_path=output_path,
            text=effective_cta_text if cfg.cta_style != "none" else None,
            sub_text=sub_text,
            platform=cfg.qa_platform,
            cta_duration_s=3.0,
            source_language=source_language,
        )
        all_warnings.extend(cta_result.warnings)

    finally:
        # ── Step 7: Clean up temp files ───────────────────────────────────────
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as exc:
            logger.warning("Could not remove temp dir %s: %s", tmp_dir, exc)

    # ── Step 6: Run QA ────────────────────────────────────────────────────────
    qa_ok = True
    qa_warnings: list[str] = []
    qa_errors: list[str] = []

    if run_qa:
        try:
            from pipeline_core import qa as _qa  # type: ignore
            qa_result = _qa.validate_output(
                output_path,
                platform=cfg.qa_platform,
                expected_duration_s=clamped_dur,
            )
            qa_ok = qa_result.ok
            qa_warnings = qa_result.warnings
            qa_errors = qa_result.errors
            if not qa_result.ok:
                logger.warning(
                    "render_mode_clip QA FAILED for %s: %s",
                    output_path, qa_result.errors,
                )
            else:
                logger.info(
                    "render_mode_clip QA passed: mode=%s path=%s",
                    mode, output_path,
                )
        except Exception as exc:
            qa_ok = False
            qa_warnings.append(f"QA check raised an exception: {exc}")
            logger.warning("render_mode_clip: QA exception: %s", exc)

    # ── Step 8: Return RenderedClip ───────────────────────────────────────────
    return RenderedClip(
        mode=mode,
        output_path=output_path,
        duration_s=clamped_dur,
        clip_candidate=candidate,
        cta_applied=cfg.cta_style,
        qa_ok=qa_ok,
        qa_warnings=qa_warnings,
        meta={
            "clip_start_s": clip_start,
            "clip_end_s": clip_start + clamped_dur,
            "raw_duration_s": raw_dur,
            "clamped_duration_s": clamped_dur,
            "composite_score": getattr(candidate, "composite_score", None),
            "narrative_role": getattr(candidate, "narrative_role", None),
            "hook_score": getattr(candidate, "hook_score", None),
            "completion_score": getattr(candidate, "completion_score", None),
            "importance_score": getattr(candidate, "importance_score", None),
            "cta_start_s": cta_result.cta_start_s,
            "cta_duration_s": cta_result.cta_duration_s,
            "warnings": all_warnings,
            "qa_errors": qa_errors,
            "source_language": source_language,
        },
    )


def render_mode_from_narrative(
    source_video_path: str,
    narrative: Any,
    *,
    mode: str,
    output_dir: str,
    target_clips: int | None = None,
    run_qa: bool = True,
    source_language: str | None = None,
) -> list[RenderedClip]:
    """Render the top-K candidates from a :class:`NarrativeResult` in *mode*.

    Parameters
    ----------
    source_video_path : str
        Absolute path to the source video.
    narrative : NarrativeResult
        Output from :func:`narrative.extract_narrative_clips`.
    mode : str
        One of the six mode keys in :data:`RENDER_MODE_CONFIGS`.
    output_dir : str
        Directory to write rendered MP4s into.
    target_clips : int | None
        How many candidates to render.  None = use all available candidates.
    run_qa : bool
        Passed through to :func:`render_mode_clip`.
    source_language : str | None
        ISO 639-1 language hint.

    Returns
    -------
    list[RenderedClip]
        Empty list when ``narrative.candidates`` is empty.
    """
    # Normalise path separators
    source_video_path = source_video_path.replace("\\", "/")
    output_dir = output_dir.replace("\\", "/")

    if mode not in RENDER_MODE_CONFIGS:
        raise ValueError(
            f"Unknown render mode {mode!r}. "
            f"Valid modes: {sorted(RENDER_MODE_CONFIGS.keys())}."
        )

    candidates = getattr(narrative, "candidates", [])
    if not candidates:
        logger.warning(
            "render_mode_from_narrative: narrative has no candidates; returning []."
        )
        return []

    os.makedirs(output_dir, exist_ok=True)

    # ── full_narrative: produces exactly 1 RenderedClip ──────────────────────
    if mode == "full_narrative":
        # Use the single best candidate (index 0, sorted by composite_score desc)
        best = candidates[0]
        out_path = os.path.join(output_dir, "full_narrative_01.mp4").replace("\\", "/")
        clip = render_mode_clip(
            source_video_path,
            best,
            mode=mode,
            output_path=out_path,
            run_qa=run_qa,
            source_language=source_language,
        )
        return [clip]

    # ── series: delegate to render_series.chain_parts ────────────────────────
    if mode == "series":
        from pipeline_core import render_series as _rs  # type: ignore
        n = target_clips if target_clips is not None else len(candidates)
        series_candidates = candidates[:max(2, n)]
        manifest = _rs.chain_parts(
            source_video_path,
            series_candidates,
            output_dir=output_dir,
            run_qa=run_qa,
            source_language=source_language,
        )
        # Convert SeriesPart list to RenderedClip list for API consistency
        rendered: list[RenderedClip] = []
        for part in manifest.parts:
            rc = RenderedClip(
                mode="series",
                output_path=part.output_path,
                duration_s=part.duration_s,
                clip_candidate=candidates[part.part_index - 1]
                if part.part_index - 1 < len(candidates)
                else None,
                cta_applied=part.cta_applied,
                qa_ok=part.qa_ok,
                qa_warnings=[],
                meta={
                    "part_index": part.part_index,
                    "part_total": part.part_total,
                    "title_overlay_text": part.title_overlay_text,
                    "cliffhanger_detected": part.cliffhanger_detected,
                    "series_warnings": manifest.warnings,
                    "playlist_title": manifest.playlist_title,
                },
            )
            rendered.append(rc)
        return rendered

    # ── standalone / trailer / promo / highlight: iterate candidates ──────────
    k = target_clips if target_clips is not None else len(candidates)
    selected = candidates[:k]

    rendered_clips: list[RenderedClip] = []
    for idx, cand in enumerate(selected, start=1):
        out_path = os.path.join(
            output_dir, f"{mode}_{idx:02d}.mp4"
        ).replace("\\", "/")
        try:
            clip = render_mode_clip(
                source_video_path,
                cand,
                mode=mode,
                output_path=out_path,
                run_qa=run_qa,
                source_language=source_language,
            )
            rendered_clips.append(clip)
        except Exception as exc:
            logger.error(
                "render_mode_from_narrative: failed rendering candidate %d "
                "(mode=%s): %s",
                idx, mode, exc,
            )
            # Record a stub RenderedClip so the caller knows what happened
            rendered_clips.append(
                RenderedClip(
                    mode=mode,
                    output_path=out_path,
                    duration_s=0.0,
                    clip_candidate=cand,
                    cta_applied="error",
                    qa_ok=False,
                    qa_warnings=[str(exc)],
                    meta={"error": str(exc)},
                )
            )

    return rendered_clips
