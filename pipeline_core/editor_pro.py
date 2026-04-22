"""
kaizer.pipeline.editor_pro
============================
Beta-mode rendering orchestrator.

Applies a coherent StylePack effect chain (colour → motion → text → captions)
to a master clip, producing a polished "beta" MP4. The original clip is never
mutated.

Effect-application order
------------------------
  1. Colour grade (apply_color_grade)
  2. Camera motion (apply_motion)
  3. Hook text animation (apply_text_animation) — if hook_text is provided
  4. Caption word animation (apply_text_animation with karaoke/pack.caption_animation)
     — if caption_word_timings is provided
  5. Final QA gate (qa.validate_output)

Any individual stage failure is caught, recorded as a warning, and the pipeline
falls back to the previous-stage output. The beta_path is **always** populated
when render_beta returns.

Usage
-----
    from pipeline_core.editor_pro import render_beta, render_both_versions

    result = render_beta(
        '/path/to/master.mp4',
        style_pack='cinematic',
        hook_text='Markets crash as inflation surges',
        output_dir='/path/to/output',
        platform='youtube_short',
    )
    print(result.beta_path, result.effects_applied)
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("kaizer.pipeline.editor_pro")

# ── Module-level effect imports so they can be patched by tests ───────────────
from pipeline_core.effects.style_packs import get_style_pack as _get_style_pack
from pipeline_core.effects.color_grade import apply_color_grade
from pipeline_core.effects.motion import apply_motion, MotionSpec
from pipeline_core.effects.text_animations import apply_text_animation, TextAnimationSpec


# ── Public dataclass ──────────────────────────────────────────────────────────

@dataclass
class BetaRenderResult:
    """Result of a render_beta() call.

    Attributes
    ----------
    current_path : str
        The original (un-beta) master clip path (unchanged from input).
    beta_path : str
        The beta-rendered MP4 with all style-pack effects applied.
    style_pack : str
        Name of the style pack used.
    effects_applied : list[str]
        Ordered list of effect identifiers that were successfully applied,
        e.g. ['color_grade:cinematic_warm', 'motion:ken_burns_in', ...].
    render_time_s : float
        Wall-clock seconds taken to produce beta_path.
    qa_ok : bool
        Whether the final QA check passed (always True when run_qa=False).
    warnings : list[str]
        Any warnings collected during the render chain.
    """

    current_path: str
    beta_path: str
    style_pack: str
    effects_applied: list[str] = field(default_factory=list)
    render_time_s: float = 0.0
    qa_ok: bool = True
    warnings: list[str] = field(default_factory=list)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _probe_duration(video_path: str) -> float:
    """Return video duration in seconds via ffprobe. Returns 0.0 on failure."""
    import json
    import subprocess as _sp
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
        proc = _sp.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            return float(data.get("format", {}).get("duration", 0.0))
    except Exception as exc:
        logger.warning("editor_pro: ffprobe duration probe failed: %s", exc)
    return 0.0


def _safe_copy(src: str, dst: str) -> None:
    """Copy *src* to *dst*, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    shutil.copy2(src, dst)


def _safe_effect(
    fn,
    *args,
    effect_label: str,
    warnings: list[str],
    fallback_path: str,
    **kwargs,
) -> tuple[str, bool]:
    """Call *fn(*args, **kwargs)*, catching any exception.

    Returns
    -------
    (output_path, success)
        If the call succeeds, returns (result_path, True).
        If it fails, records the error as a warning and returns
        (fallback_path, False).
    """
    try:
        result = fn(*args, **kwargs)
        return result, True
    except Exception as exc:
        msg = f"{effect_label} failed: {exc}"
        logger.warning("editor_pro: %s", msg)
        warnings.append(msg)
        return fallback_path, False


# ── Public API ────────────────────────────────────────────────────────────────

def render_beta(
    master_clip_path: str,
    *,
    style_pack: str = "cinematic",
    hook_text: Optional[str] = None,
    caption_word_timings: Optional[list] = None,
    output_dir: str,
    platform: str = "youtube_short",
    run_qa: bool = True,
) -> BetaRenderResult:
    """Apply a style-pack's effect chain to a master clip, producing the beta MP4.

    Parameters
    ----------
    master_clip_path : str
        Absolute path to the source master clip.
    style_pack : str
        Name of the style pack to apply (one of the keys in STYLE_PACKS).
    hook_text : str | None
        Optional headline/hook text overlaid via text_animation at start_s=0.2.
    caption_word_timings : list | None
        [{'word': str, 'start': float}, ...] for karaoke-style caption overlay.
    output_dir : str
        Directory to write the beta MP4 into.
    platform : str
        Platform identifier for QA checks.
    run_qa : bool
        If True, run qa.validate_output on the final beta_path.

    Returns
    -------
    BetaRenderResult
        Always populated — even on partial failure (warnings record issues).
    """
    t0 = time.monotonic()
    master_clip_path = master_clip_path.replace("\\", "/")
    output_dir = output_dir.replace("\\", "/")
    os.makedirs(output_dir, exist_ok=True)

    warnings_list: list[str] = []
    effects_applied: list[str] = []

    # Resolve the style pack
    try:
        pack = _get_style_pack(style_pack)
    except ValueError as exc:
        # Unknown pack — record and fall back to a pass-through copy
        warnings_list.append(str(exc))
        logger.warning("render_beta: %s", exc)
        beta_path = os.path.join(output_dir, "beta_output.mp4").replace("\\", "/")
        _safe_copy(master_clip_path, beta_path)
        return BetaRenderResult(
            current_path=master_clip_path,
            beta_path=beta_path,
            style_pack=style_pack,
            effects_applied=[],
            render_time_s=time.monotonic() - t0,
            qa_ok=False,
            warnings=warnings_list,
        )

    tmp_dir = tempfile.mkdtemp(prefix="kaizer_beta_")
    beta_path: str = ""

    try:
        # Working path — threads through each stage
        current = master_clip_path
        stage_idx = 0

        def _tmp(suffix: str) -> str:
            nonlocal stage_idx
            stage_idx += 1
            return os.path.join(
                tmp_dir, f"stage_{stage_idx:02d}_{suffix}.mp4"
            ).replace("\\", "/")

        # ── Stage 1: Colour grade ─────────────────────────────────────────────
        if pack.color_preset and pack.color_preset != "none":
            stage_out = _tmp("color")
            current, ok = _safe_effect(
                apply_color_grade,
                current,
                effect_label=f"color_grade:{pack.color_preset}",
                warnings=warnings_list,
                fallback_path=current,
                preset=pack.color_preset,
                output_path=stage_out,
            )
            if ok:
                effects_applied.append(f"color_grade:{pack.color_preset}")

        # ── Stage 2: Camera motion ────────────────────────────────────────────
        if pack.motion:
            clip_dur = _probe_duration(current)
            motion_spec = MotionSpec(
                name=pack.motion,
                duration_s=clip_dur,
                intensity=pack.motion_intensity,
            )
            stage_out = _tmp("motion")
            current, ok = _safe_effect(
                apply_motion,
                current,
                motion_spec,
                effect_label=f"motion:{pack.motion}",
                warnings=warnings_list,
                fallback_path=current,
                output_path=stage_out,
            )
            if ok:
                effects_applied.append(f"motion:{pack.motion}")

        # ── Stage 3: Hook/headline text animation ─────────────────────────────
        if hook_text:
            clip_dur = _probe_duration(current)
            text_dur = min(3.0, max(1.0, clip_dur * 0.15))
            text_spec = TextAnimationSpec(
                name=pack.text_animation,
                text=hook_text,
                duration_s=text_dur,
                start_s=0.2,
                font_size=72,
                color="#FFFFFF",
                position=(80, 1600),
                platform=platform,
            )
            stage_out = _tmp("hooktext")
            current, ok = _safe_effect(
                apply_text_animation,
                current,
                text_spec,
                effect_label=f"text_animation:{pack.text_animation}",
                warnings=warnings_list,
                fallback_path=current,
                output_path=stage_out,
            )
            if ok:
                effects_applied.append(f"text_animation:{pack.text_animation}")

        # ── Stage 4: Caption word animation ──────────────────────────────────
        if caption_word_timings:
            full_text = " ".join(
                wt.get("word", "") for wt in caption_word_timings
            ).strip()
            if full_text:
                clip_dur = _probe_duration(current)
                cap_dur = min(clip_dur - 0.5, max(2.0, clip_dur * 0.6))
                cap_spec = TextAnimationSpec(
                    name=pack.caption_animation,
                    text=full_text,
                    duration_s=cap_dur,
                    start_s=0.5,
                    font_size=52,
                    color="#FFFFFF",
                    position=(80, 1700),
                    platform=platform,
                    params={"word_timings": caption_word_timings},
                )
                stage_out = _tmp("captions")
                current, ok = _safe_effect(
                    apply_text_animation,
                    current,
                    cap_spec,
                    effect_label=f"caption_animation:{pack.caption_animation}",
                    warnings=warnings_list,
                    fallback_path=current,
                    output_path=stage_out,
                )
                if ok:
                    effects_applied.append(f"caption_animation:{pack.caption_animation}")

        # ── Finalise: copy working result to beta_path ────────────────────────
        base_name = os.path.splitext(os.path.basename(master_clip_path))[0]
        beta_path = os.path.join(
            output_dir, f"{base_name}_beta_{style_pack}.mp4"
        ).replace("\\", "/")

        _safe_copy(current, beta_path)

        # ── Stage 5: QA ───────────────────────────────────────────────────────
        qa_ok = True
        if run_qa:
            try:
                from pipeline_core import qa as _qa
                qa_result = _qa.validate_output(beta_path, platform=platform)
                qa_ok = qa_result.ok
                if qa_result.warnings:
                    warnings_list.extend(qa_result.warnings)
                if not qa_result.ok:
                    warnings_list.extend(
                        [f"QA error: {e}" for e in qa_result.errors]
                    )
                    logger.warning(
                        "render_beta: QA FAILED for %s: %s",
                        beta_path, qa_result.errors,
                    )
            except Exception as exc:
                msg = f"QA check raised: {exc}"
                warnings_list.append(msg)
                logger.warning("render_beta: %s", msg)
                qa_ok = False

    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as exc:
            logger.warning("render_beta: could not remove tmp dir: %s", exc)

    render_time = time.monotonic() - t0
    logger.info(
        "render_beta: pack=%s effects=%s qa_ok=%s duration=%.2fs → %s",
        style_pack, effects_applied, qa_ok, render_time,
        os.path.basename(beta_path),
    )

    return BetaRenderResult(
        current_path=master_clip_path,
        beta_path=beta_path,
        style_pack=style_pack,
        effects_applied=effects_applied,
        render_time_s=render_time,
        qa_ok=qa_ok,
        warnings=warnings_list,
    )


def render_both_versions(
    master_clip_path: str,
    *,
    style_pack: str = "cinematic",
    hook_text: Optional[str] = None,
    output_dir: str,
    platform: str = "youtube_short",
) -> BetaRenderResult:
    """Convenience wrapper: always returns both current_path (= input) and beta_path.

    Suitable for side-by-side comparison UI. QA is always run.

    Parameters
    ----------
    master_clip_path : str
        Absolute path to the source master clip.
    style_pack : str
        Style pack to apply for the beta version.
    hook_text : str | None
        Optional headline text for hook text overlay.
    output_dir : str
        Directory to write the beta MP4 into.
    platform : str
        Platform identifier for QA checks.

    Returns
    -------
    BetaRenderResult
        .current_path is the unmodified input path.
        .beta_path is the new styled output.
    """
    return render_beta(
        master_clip_path,
        style_pack=style_pack,
        hook_text=hook_text,
        output_dir=output_dir,
        platform=platform,
        run_qa=True,
    )
