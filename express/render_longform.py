"""renderLongformTrim — Python port of teammate's same-named function.

Builds a filter_complex graph that:
  1. Trims the source into N segments per Claude's keep plan
  2. Applies the chosen color grade to each segment
  3. (Cinematic mode) Ken Burns zoom + film grain + xfade transitions
  4. Concatenates the segments (hard cuts OR cinematic xfade)
  5. Optionally overlays a small channel logo top-right
  6. Encodes via libx264 CRF 20 (matches teammate)

Same wire shapes as the JS version — feeding the same ``keep=[{start,
end,reason}, ...]`` plus the same ``colorGrade`` + ``cinematicEdit``
flags reproduces the same output bit-for-bit-similar.

Difference vs teammate:
  - Uses Python subprocess instead of Node's child_process.spawn
  - Hard 20-minute timeout via subprocess.run timeout
  - Pulls ``FFMPEG_BIN`` from Kaizer's pipeline_core.hw_accel so the
    NVENC-capable build wins automatically; falls back to "ffmpeg".
"""
from __future__ import annotations

import os
import subprocess
from typing import Optional

from . import color_grade, hw_accel


# Teammate's render uses libx264 CRF 20 for editorial control. We
# match that for visual A/B testing. NVENC could speed up the encode
# 5-10×; we keep CPU x264 for now so the grade chain renders identically.
_CRF = "20"
_PRESET = "medium"


class RenderError(RuntimeError):
    """ffmpeg long-form render failed or timed out."""


def _ffmpeg_bin() -> str:
    """Resolve the ffmpeg binary. Honours ``FFMPEG_BIN`` env so the
    Kaizer hw-accel detector and this module agree on which binary
    they're calling."""
    return os.environ.get("FFMPEG_BIN", "ffmpeg")


def render_longform_trim(
    *,
    input_path: str,
    output_path: str,
    keep: list[dict],
    logo_path: Optional[str] = None,
    color_grade_preset: str = "subtle",
    cinematic_edit: bool = False,
    timeout_s: int = 20 * 60,
) -> str:
    """Concatenate the ``keep`` segments of ``input_path`` into
    ``output_path`` with the requested grade.

    Parameters
    ----------
    keep : list of {start, end, reason}. Must be sorted+merged
           (planner does this already). At least 1 entry.
    logo_path : if provided + exists, overlaid 55px wide at top-right
                with 20px margin (matches teammate's scale=55:-1).
    color_grade_preset : one of color_grade.VALID_PRESETS.
    cinematic_edit : enables Ken Burns + xfade + acrossfade + grain.

    Returns
    -------
    ``output_path`` on success. Raises RenderError on failure.
    """
    if not keep:
        raise RenderError("No segments to keep — nothing to render.")
    if not os.path.isfile(input_path):
        raise RenderError(f"input not found: {input_path}")

    FADE_S = 0.080         # 80ms audio fade in/out per segment (declick)
    XFADE_DUR = 0.5        # cinematic mode crossfade duration
    KEN_BURNS_ZOOM = 1.04  # subtle 1.00 → 1.04 zoom over the segment

    grade = color_grade.grade_chain(color_grade_preset)

    parts: list[str] = []
    seg_durs: list[float] = []

    # ── Per-segment trim → grade → (Ken Burns + grain in cinematic) ─
    for i, seg in enumerate(keep):
        start = max(0.0, float(seg.get("start") or 0.0))
        end   = max(start + 0.05, float(seg.get("end") or 0.0))
        dur   = max(0.05, end - start)
        seg_durs.append(dur)

        v_steps: list[str] = [
            f"trim=start={start:.3f}:end={end:.3f}",
            "setpts=PTS-STARTPTS",
        ]
        if grade:
            v_steps.append(grade)

        if cinematic_edit:
            # Assume source fps=30 for zoompan; downstream xfade math
            # doesn't depend on this so a wrong assumption only
            # affects zoompan smoothness, not duration.
            frames = max(2, round(dur * 30))
            v_steps.append(
                "scale=2400:1350:force_original_aspect_ratio=increase,"
                "crop=2400:1350,"
                f"zoompan=z='min(zoom+0.0008,{KEN_BURNS_ZOOM})':d={frames}:s=1920x1080:fps=30"
            )
            v_steps.append("noise=alls=8:allf=t")

        parts.append(f"[0:v]{','.join(v_steps)}[v{i}]")
        parts.append(
            f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS,"
            f"afade=t=in:st=0:d={FADE_S},afade=t=out:st={max(0, dur - FADE_S):.3f}:d={FADE_S}[a{i}]"
        )

    # ── Concat or xfade ────────────────────────────────────────────
    if cinematic_edit and len(keep) > 1:
        # Cinematic: chained xfade + acrossfade.
        last_v = "v0"
        last_a = "a0"
        cum = seg_durs[0]
        for i in range(1, len(keep)):
            offset = max(0.0, cum - XFADE_DUR)
            new_v = f"vx{i}"
            new_a = f"ax{i}"
            parts.append(
                f"[{last_v}][v{i}]xfade=transition=fade:"
                f"duration={XFADE_DUR}:offset={offset:.3f}[{new_v}]"
            )
            parts.append(
                f"[{last_a}][a{i}]acrossfade=d={XFADE_DUR}:c1=tri:c2=tri[{new_a}]"
            )
            cum = cum + seg_durs[i] - XFADE_DUR
            last_v = new_v
            last_a = new_a
        parts.append(f"[{last_v}]copy[vcat]")
        parts.append(f"[{last_a}]acopy[acat]")
    else:
        # Default: hard concat. Each segment already has audio fades.
        concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(keep)))
        parts.append(f"{concat_inputs}concat=n={len(keep)}:v=1:a=1[vcat][acat]")

    # ── Optional logo overlay ──────────────────────────────────────
    has_logo = bool(logo_path and os.path.isfile(logo_path))
    out_v = "vcat"
    if has_logo:
        parts.append("[1:v]scale=55:-1[lg]")
        parts.append("[vcat][lg]overlay=W-w-20:20[vlogo]")
        out_v = "vlogo"

    filter_complex = ";".join(parts)

    # ── Spawn ffmpeg ───────────────────────────────────────────────
    args: list[str] = [_ffmpeg_bin(), "-i", input_path]
    if has_logo:
        args += ["-i", logo_path]
    args += [
        "-filter_complex", filter_complex,
        "-map", f"[{out_v}]",
        "-map", "[acat]",
        # Picks NVENC when an NVIDIA GPU is present, libx264 otherwise.
        *hw_accel.encoder_args(),
        "-c:a", "aac",
        "-b:a", "160k",
        "-ar", "48000",
        "-movflags", "+faststart",
        "-y", output_path,
    ]
    print(f"[render_longform] encoder={hw_accel.active_encoder_label()} segments={len(keep)} grade={color_grade_preset}")

    try:
        proc = subprocess.run(args, capture_output=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise RenderError(f"render timed out after {timeout_s}s") from exc

    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace")[-700:]
        raise RenderError(f"ffmpeg longform render failed: {stderr}")

    if not os.path.isfile(output_path) or os.path.getsize(output_path) < 1000:
        raise RenderError("ffmpeg exited 0 but the output file is empty/missing")

    return output_path
