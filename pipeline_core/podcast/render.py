# Ported from kaizer-platform@d5fd482 server/pipeline_core/podcast/render.py.
# Changes from upstream: build_extraction_edl import rewired from the _v2path sys.path
# bootstrap (pipeline_v2 removed on this side) to the local edl_builder module; logic unchanged.
"""Podcast ffmpeg renderer (Phase 1, single-cam).

Consumes the pure plans (keep_ranges + punch-ins + promo) and executes
ffmpeg to produce:

  * ``podcast_edit.mp4``  — the main cut (silence/filler/stutter removed),
    punch-ins applied, word-pop captions burned, loudness normalised.
  * ``promo_169.mp4``     — 16:9 promo/trailer.
  * ``promo_916.mp4``     — 9:16 vertical promo (center-crop).

Design decisions (honesty, per the task):

  * EXTRACTION reuses ``pipeline_v2.render.edl_builder.build_extraction_edl``
    to build the single-decode trim+concat graph for the keep_ranges. We do
    NOT fork it — we call it, take its per-segment trim labels, and layer
    punch-in / caption / loudnorm filters on top in the SAME ffmpeg call.

  * PUNCH-INS are applied as **crop+scale** on the affected concatenated
    output (robust, deterministic — chosen over ``zoompan`` which is
    frame-index fiddly and can jitter on VFR). A punch-in is a static
    zoomed crop for its window; "snap" vs "slow_push" differ only in the
    metadata we record (Phase 1 renders both as a hold — animated easing is
    a Phase 2 refinement). This is stated plainly rather than pretended.

  * CAPTIONS use ffmpeg **drawtext** (word-pop: one word shown at a time,
    timed via ``enable='between(t,a,b)'``). We do NOT use
    ``pipeline_core.captions`` here because that renders PIL images (great
    for Indic shaping, but overlaying hundreds of per-word PNGs is heavy);
    drawtext is the honest Phase-1 path. Indic word-pop via the PIL caption
    engine is a documented Phase-2 upgrade. drawtext needs a font file; we
    resolve one from the existing fonts dir.

  * LOUDNESS: single-pass ``loudnorm`` to -14 LUFS / -1.0 dBTP (rule 18).

If ffmpeg/ffprobe are unavailable the module still imports; only the
execution functions raise.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from pipeline_core.podcast.edl_builder import build_extraction_edl
from pipeline_core.podcast.camera_plan import CameraPlan
from pipeline_core.podcast.punchin import PunchIn

logger = logging.getLogger("pipeline_core.podcast.render")

# Loudness (rule 18).
_TARGET_LUFS = -14.0
_TRUE_PEAK = -1.0
_LRA = 11.0

# drawtext font resolution — reuse the resources/fonts dir.
_HERE = os.path.dirname(os.path.abspath(__file__))
_FONTS_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "resources", "fonts"))
_DRAWTEXT_FONT_CANDIDATES = (
    "NotoSans-Bold.ttf", "Roboto-Bold.ttf", "Oswald-Bold.ttf", "NotoSans-Regular.ttf",
)


def _resolve_font() -> str | None:
    for name in _DRAWTEXT_FONT_CANDIDATES:
        p = os.path.join(_FONTS_DIR, name)
        if os.path.isfile(p):
            return p.replace("\\", "/")
    return None


def _ffpath(p: str) -> str:
    """Escape a path for use inside an ffmpeg filter option value.

    On Windows, drawtext fontfile / the drive-colon must be escaped.
    """
    q = p.replace("\\", "/")
    # Escape the drive colon (C: -> C\:) for filtergraph parsing.
    q = q.replace(":", "\\:")
    return q


def _drawtext_escape(text: str) -> str:
    """Escape caption text for drawtext (single-quote wrapped)."""
    return (
        text.replace("\\", "\\\\")
        .replace("'", "’")   # curly apostrophe avoids quote-breaking
        .replace(":", "\\:")
        .replace("%", "\\%")
    )


# ── Encoder args (lazy import so tests without hw_accel deps still load) ──


def _video_encode_args() -> list[str]:
    try:
        from pipeline_core.hw_accel import h264_args  # type: ignore
        return h264_args()
    except Exception:
        return [
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        ]


# ── Filter builders ─────────────────────────────────────────────────────


def _write_filtergraph_script(filter_complex: str, out_path: str, name: str) -> str:
    """Write the filtergraph to a file next to the output; return its path.

    Windows' CreateProcess caps the WHOLE command line at 32,767 chars; the
    per-word drawtext caption chain blows past that at ~150 words, so the
    graph is never passed inline — it goes to a file consumed via ffmpeg's
    ``-filter_complex_script``. The file is KEPT on render failure for
    debugging (``_run`` deletes it only after a successful exit).
    """
    script = Path(out_path).resolve().parent / name
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(filter_complex, encoding="utf-8")
    return str(script)


def _caption_drawtext_chain(
    in_label: str,
    out_label: str,
    captions: Sequence[dict],
    font: str | None,
    *,
    frame_h: int = 1080,
) -> str:
    """Build a chain of drawtext filters — one word shown at a time.

    ``captions`` entries are ``{w, start_s, end_s}`` in the OUTPUT timeline.
    Returns a filtergraph fragment ``[in]drawtext=...,drawtext=...[out]``.
    If there are no captions, returns a null passthrough.
    """
    if not captions:
        return f"[{in_label}]null[{out_label}]"

    font_opt = f"fontfile='{_ffpath(font)}':" if font else ""
    y_expr = f"h-(h/6)"  # lower third-ish
    dts: list[str] = []
    for c in captions:
        word = _drawtext_escape(str(c.get("w", "")).strip())
        if not word:
            continue
        a = float(c["start_s"])
        b = float(c["end_s"])
        if b <= a:
            b = a + 0.15
        dts.append(
            f"drawtext={font_opt}"
            f"text='{word}':"
            f"fontcolor=white:fontsize=54:borderw=4:bordercolor=black@0.9:"
            f"x=(w-text_w)/2:y={y_expr}:"
            f"enable='between(t,{a:.3f},{b:.3f})'"
        )
    if not dts:
        return f"[{in_label}]null[{out_label}]"
    return f"[{in_label}]" + ",".join(dts) + f"[{out_label}]"


def _punchin_crop_chain(
    in_label: str,
    out_label: str,
    punch_ins: Sequence[PunchIn],
    *,
    out_w: int,
    out_h: int,
) -> str:
    """Apply punch-ins as a time-varying center ``crop`` + fixed ``scale``.

    ``zoompan`` cannot key on wall-clock ``t`` (its vocabulary is frame-counter
    based: ``in``/``on``), so we use ``crop`` — whose w/h/x/y expressions DO
    expose ``t`` — to crop a tighter centered region during each punch window,
    then ``scale`` that crop back up to a FIXED output size (``out_w`` x
    ``out_h``). A tighter crop scaled up == a digital punch-in (rule 6).

    Zoom Z during a window: crop to ``iw/Z x ih/Z`` centered; outside all
    windows Z=1 (full frame). The zoom is a piecewise-constant
    ``if(between(t,...))`` expression — a HOLD for the window. Both "snap" and
    "slow_push" render as a hold in Phase 1; animated easing is a Phase-2
    refinement, and the per-window mode is preserved in the sidecar plan for
    the future Remotion renderer. Crop dims are rounded to even numbers so the
    yuv420p encoder accepts them.
    """
    if not punch_ins:
        return f"[{in_label}]null[{out_label}]"

    # Piecewise zoom over time. Base 1.0 (full frame). Commas escaped for the
    # filtergraph parser.
    zexpr = "1.0"
    for p in punch_ins:
        zexpr = (
            f"if(between(t\\,{p.start_s:.3f}\\,{p.end_s:.3f})\\,{p.zoom:.4f}\\,{zexpr})"
        )
    # Even-rounded centered crop, then scale to the fixed output size.
    return (
        f"[{in_label}]"
        f"crop=w='floor(iw/({zexpr})/2)*2':h='floor(ih/({zexpr})/2)*2':"
        f"x='(iw-floor(iw/({zexpr})/2)*2)/2':y='(ih-floor(ih/({zexpr})/2)*2)/2',"
        f"scale={out_w}:{out_h}:eval=frame,setsar=1"
        f"[{out_label}]"
    )


def _reframe_crop_chain(in_label: str, out_label: str, plan: CameraPlan) -> str:
    """Crop to a FIXED-size (``plan.crop_w`` x ``plan.crop_h``) window whose
    POSITION pans between speaker slots over time — a virtual multi-cam
    "reframe" built from one source track (see camera_plan.py). Only x/y
    move; the output size never changes, so the fixed-size scale/pad/
    punch-in/caption chain that runs after this needs no changes to cope
    with a time-varying frame size.

    Applied on the RAW concatenated (source-resolution) video, BEFORE the
    normalize-to-output-size step, so window x/y stay in source-pixel space
    — the same space camera_plan.py computed them in.
    """
    if not plan.windows:
        return f"[{in_label}]null[{out_label}]"
    xexpr = str(plan.default_x)
    yexpr = str(plan.default_y)
    for w in plan.windows:
        xexpr = f"if(between(t\\,{w.start_s:.3f}\\,{w.end_s:.3f})\\,{w.x}\\,{xexpr})"
        yexpr = f"if(between(t\\,{w.start_s:.3f}\\,{w.end_s:.3f})\\,{w.y}\\,{yexpr})"
    return (
        f"[{in_label}]crop=w={plan.crop_w}:h={plan.crop_h}:x='{xexpr}':y='{yexpr}'"
        f"[{out_label}]"
    )


@dataclass(frozen=True)
class RenderRequest:
    source_path: str
    keep_ranges: tuple[tuple[float, float], ...]
    punch_ins: tuple[PunchIn, ...] = ()
    captions: tuple[dict, ...] = ()
    camera_plan: Optional[CameraPlan] = None
    out_path: str = "podcast_edit.mp4"
    width: int = 1920
    height: int = 1080
    loudnorm: bool = True


def _build_main_command(req: RenderRequest, font: str | None) -> tuple[list[str], float]:
    """Assemble the ffmpeg command for the main edit. Returns (cmd, expected_dur)."""
    edl = build_extraction_edl(bulletin_cuts=req.keep_ranges)
    out_spec = edl.outputs[0]  # concat bulletin output
    parts = [edl.filter_complex]

    cur_v = out_spec.v_label
    cur_a = out_spec.a_label

    # Camera plan (virtual multi-cam reframe, Phase C) — applied on the RAW
    # concatenated source-resolution video, before the fixed-size normalize
    # step below, so its crop coordinates stay in source-pixel space.
    if req.camera_plan is not None and req.camera_plan.engaged and req.camera_plan.windows:
        parts.append(_reframe_crop_chain(cur_v, "rf", req.camera_plan))
        cur_v = "rf"

    # Normalize the concatenated video to a fixed WxH (letterbox-pad, no
    # distortion) so the punch-in crop+scale has a stable target and captions
    # sit consistently. This also guarantees even dims for yuv420p.
    parts.append(
        f"[{cur_v}]scale={req.width}:{req.height}:force_original_aspect_ratio=decrease,"
        f"pad={req.width}:{req.height}:(ow-iw)/2:(oh-ih)/2,setsar=1[nv]"
    )
    cur_v = "nv"

    # Punch-in (time-varying crop + scale back to the fixed frame).
    if req.punch_ins:
        parts.append(_punchin_crop_chain(
            cur_v, "pv", req.punch_ins, out_w=req.width, out_h=req.height,
        ))
        cur_v = "pv"

    # Captions (drawtext word-pop).
    if req.captions:
        parts.append(_caption_drawtext_chain(cur_v, "cv", req.captions, font, frame_h=req.height))
        cur_v = "cv"

    # Loudness normalize.
    if req.loudnorm:
        parts.append(
            f"[{cur_a}]loudnorm=I={_TARGET_LUFS}:TP={_TRUE_PEAK}:LRA={_LRA}[na]"
        )
        cur_a = "na"

    filter_complex = ";".join(parts)
    # Graph goes to a FILE (-filter_complex_script), never inline: per-word
    # drawtext captions exceed Windows' 32,767-char command-line limit.
    script_path = _write_filtergraph_script(
        filter_complex, req.out_path, "filtergraph_edit.txt")
    cmd = [
        "ffmpeg", "-y", "-i", req.source_path,
        "-filter_complex_script", script_path,
        "-map", f"[{cur_v}]", "-map", f"[{cur_a}]",
        *_video_encode_args(),
        "-c:a", "aac", "-b:a", "192k",
        req.out_path,
    ]
    return cmd, out_spec.duration_s


def _build_promo_command(
    source_path: str,
    keep_ranges: Sequence[tuple[float, float]],
    captions: Sequence[dict],
    out_path: str,
    *,
    vertical: bool,
    font: str | None,
    loudnorm: bool = True,
) -> tuple[list[str], float]:
    """Assemble the ffmpeg command for a promo variant (16:9 or 9:16)."""
    if not keep_ranges:
        raise ValueError("promo keep_ranges empty; nothing to render")
    edl = build_extraction_edl(bulletin_cuts=tuple(keep_ranges))
    out_spec = edl.outputs[0]
    parts = [edl.filter_complex]
    cur_v = out_spec.v_label
    cur_a = out_spec.a_label

    if vertical:
        # center-crop to 9:16 then scale to 1080x1920.
        parts.append(
            f"[{cur_v}]crop=w='min(iw,ih*9/16)':h=ih:x='(iw-min(iw,ih*9/16))/2':y=0,"
            f"scale=1080:1920,setsar=1[vv]"
        )
        cur_v = "vv"
        frame_h = 1920
    else:
        parts.append(f"[{cur_v}]scale=1920:1080:force_original_aspect_ratio=decrease,"
                     f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[vv]")
        cur_v = "vv"
        frame_h = 1080

    if captions:
        parts.append(_caption_drawtext_chain(cur_v, "cv", captions, font, frame_h=frame_h))
        cur_v = "cv"

    if loudnorm:
        parts.append(f"[{cur_a}]loudnorm=I={_TARGET_LUFS}:TP={_TRUE_PEAK}:LRA={_LRA}[na]")
        cur_a = "na"

    # Graph via file for the same Windows command-line-limit reason as the
    # main edit (promo captions are per-word drawtext too).
    script_path = _write_filtergraph_script(
        ";".join(parts), out_path, f"filtergraph_{Path(out_path).stem}.txt")
    cmd = [
        "ffmpeg", "-y", "-i", source_path,
        "-filter_complex_script", script_path,
        "-map", f"[{cur_v}]", "-map", f"[{cur_a}]",
        *_video_encode_args(),
        "-c:a", "aac", "-b:a", "192k",
        out_path,
    ]
    return cmd, out_spec.duration_s


# ── Execution ────────────────────────────────────────────────────────────


def _run(cmd: list[str], *, timeout: int = 900) -> None:
    logger.info("podcast render: %s", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        # NOTE: any -filter_complex_script file is deliberately KEPT on
        # failure so the graph can be inspected/replayed by hand.
        raise RuntimeError(
            "ffmpeg failed (exit %d):\n%s"
            % (proc.returncode, proc.stderr.decode(errors="replace")[-4000:])
        )
    # Success: the filtergraph script (if this command used one) is spent.
    try:
        idx = cmd.index("-filter_complex_script")
        os.unlink(cmd[idx + 1])
    except (ValueError, IndexError, OSError):
        pass


def ffprobe_duration(path: str) -> float:
    """Return the container duration in seconds via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", path],
        capture_output=True, timeout=60,
    )
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {out.stderr.decode(errors='replace')}")
    data = json.loads(out.stdout.decode())
    return float(data["format"]["duration"])


def ffprobe_video_info(path: str) -> dict:
    """Return ``{width, height, fps, duration}`` for the first video stream.

    Used only by the remotion branch to populate the EDL ``source`` block.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate:format=duration",
         "-of", "json", path],
        capture_output=True, timeout=60,
    )
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {out.stderr.decode(errors='replace')}")
    data = json.loads(out.stdout.decode())
    stream = (data.get("streams") or [{}])[0]
    num, _, den = str(stream.get("r_frame_rate", "30/1")).partition("/")
    try:
        fps = float(num) / float(den) if den and float(den) else float(num)
    except (ValueError, ZeroDivisionError):
        fps = 30.0
    return {
        "width": int(stream.get("width", 1920)),
        "height": int(stream.get("height", 1080)),
        "fps": fps,
        "duration": float(data.get("format", {}).get("duration", 0.0)) or None,
    }


@dataclass(frozen=True)
class RenderResult:
    edit_path: str
    promo_169_path: str | None
    promo_916_path: str | None
    edit_expected_s: float
    renderer_used: str = "ffmpeg"
    theme: str | None = None


def _select_renderer(renderer: str | None) -> str:
    """Resolve the renderer: explicit arg > env > default 'ffmpeg'.

    ``KAIZER_PODCAST_RENDERER=remotion`` opts in globally; the ``renderer``
    argument overrides the env. Unknown values fall back to 'ffmpeg'.
    """
    choice = (renderer or os.environ.get("KAIZER_PODCAST_RENDERER") or "ffmpeg").lower()
    return choice if choice in ("ffmpeg", "remotion") else "ffmpeg"


def _render_podcast_remotion(
    *,
    source_path: str,
    out: Path,
    edit_keep_ranges: Sequence[tuple[float, float]],
    punch_ins: Sequence[PunchIn],
    edit_captions: Sequence[dict],
    promo_keep_ranges: Sequence[tuple[float, float]],
    promo_captions: Sequence[dict],
    language: str | None = None,
    name_plate: str | None = None,
    promo_end_card_sec: float = 0.0,
) -> RenderResult:
    """Render via the Remotion node CLI. Raises RemotionUnavailable so the
    caller can fall back to ffmpeg. Animated punch-ins + word-pop captions are
    the upgrade over the ffmpeg held-zoom path.

    ``language`` selects the Remotion theme (te->'telugu', else 'kaizerDark')
    and the localized promo end-card CTA."""
    from pipeline_core.podcast import remotion_bridge as rb
    from pipeline_core.podcast.theme import (
        end_card_cta_for_language, theme_for_language,
    )

    # Fail fast (before any ffprobe work) if the toolchain is missing, so the
    # caller falls back to ffmpeg cleanly.
    if not rb.remotion_available():
        raise rb.RemotionUnavailable("remotion toolchain not available")

    info = ffprobe_video_info(source_path)
    edit_path = str(out / "podcast_edit.mp4")
    edl = rb.build_edl(
        source_path=os.path.abspath(source_path),
        source_width=info["width"], source_height=info["height"],
        source_fps=info["fps"], source_duration=info["duration"],
        keep_ranges=edit_keep_ranges, punch_ins=punch_ins, captions=edit_captions,
        promo_keep_ranges=promo_keep_ranges, promo_captions=promo_captions,
        promo_end_card_sec=promo_end_card_sec,
        theme=theme_for_language(language),
        language=language,
        name_plate=name_plate,
        promo_end_card_text=(
            end_card_cta_for_language(language) if promo_keep_ranges else None
        ),
    )
    summary = rb.render_with_remotion(
        composition=rb.COMPOSITION_EDIT, edl=edl, out_path=edit_path,
    )
    logger.info("podcast remotion edit done: %s", summary)

    promo_169 = promo_916 = None
    if promo_keep_ranges:
        promo_169 = str(out / "promo_169.mp4")
        rb.render_with_remotion(
            composition=rb.COMPOSITION_PROMO_169, edl=edl, out_path=promo_169,
        )
        promo_916 = str(out / "promo_916.mp4")
        rb.render_with_remotion(
            composition=rb.COMPOSITION_PROMO_916, edl=edl, out_path=promo_916,
        )

    expected = sum(e - s for s, e in edit_keep_ranges)
    return RenderResult(
        edit_path=edit_path,
        promo_169_path=promo_169,
        promo_916_path=promo_916,
        edit_expected_s=expected,
        renderer_used="remotion",
        theme=theme_for_language(language),
    )


def render_podcast(
    *,
    source_path: str,
    out_dir: str,
    edit_keep_ranges: Sequence[tuple[float, float]],
    punch_ins: Sequence[PunchIn] = (),
    edit_captions: Sequence[dict] = (),
    camera_plan: Optional[CameraPlan] = None,
    promo_keep_ranges: Sequence[tuple[float, float]] = (),
    promo_captions: Sequence[dict] = (),
    loudnorm: bool = True,
    renderer: str = "ffmpeg",
    language: str | None = None,
    name_plate: str | None = None,
    promo_end_card_sec: float = 0.0,
) -> RenderResult:
    """Render the main edit + both promo variants.

    Any promo variant is skipped (returns None) if ``promo_keep_ranges`` is
    empty. All ffmpeg calls raise ``RuntimeError`` on failure.

    ``renderer`` selects the backend ('ffmpeg' default | 'remotion' opt-in;
    overridable by env ``KAIZER_PODCAST_RENDERER``). The remotion path animates
    punch-ins and word-pop captions; if the node toolchain is unavailable it
    falls back to ffmpeg (logged honestly). ffmpeg behaviour is unchanged.

    ``camera_plan`` (Phase C virtual multi-cam reframe) applies to the MAIN
    EDIT on the ffmpeg path only. It is not yet wired into promo rendering or
    the Remotion path — an engaged plan there is logged, not silently dropped.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if _select_renderer(renderer) == "remotion":
        if camera_plan is not None and camera_plan.engaged:
            logger.warning(
                "podcast: camera plan is engaged (%s) but the remotion "
                "renderer does not consume it yet; rendering without reframe",
                camera_plan.reason,
            )
        from pipeline_core.podcast.remotion_bridge import RemotionUnavailable
        try:
            return _render_podcast_remotion(
                source_path=source_path, out=out,
                edit_keep_ranges=edit_keep_ranges, punch_ins=punch_ins,
                edit_captions=edit_captions,
                promo_keep_ranges=promo_keep_ranges, promo_captions=promo_captions,
                language=language, name_plate=name_plate,
                promo_end_card_sec=promo_end_card_sec,
            )
        except RemotionUnavailable as exc:
            logger.warning(
                "podcast: remotion renderer unavailable (%s); falling back to ffmpeg",
                exc,
            )
            # fall through to the ffmpeg path below.
    font = _resolve_font()
    if font is None:
        logger.warning("podcast render: no drawtext font found in %s; captions disabled", _FONTS_DIR)

    edit_path = str(out / "podcast_edit.mp4")
    req = RenderRequest(
        source_path=source_path,
        keep_ranges=tuple(edit_keep_ranges),
        punch_ins=tuple(punch_ins),
        captions=tuple(edit_captions) if font else (),
        camera_plan=camera_plan,
        out_path=edit_path,
        loudnorm=loudnorm,
    )
    cmd, expected = _build_main_command(req, font)
    _run(cmd)

    promo_169 = promo_916 = None
    if promo_keep_ranges:
        promo_169 = str(out / "promo_169.mp4")
        c1, _ = _build_promo_command(
            source_path, promo_keep_ranges, promo_captions if font else (),
            promo_169, vertical=False, font=font, loudnorm=loudnorm,
        )
        _run(c1)

        promo_916 = str(out / "promo_916.mp4")
        c2, _ = _build_promo_command(
            source_path, promo_keep_ranges, promo_captions if font else (),
            promo_916, vertical=True, font=font, loudnorm=loudnorm,
        )
        _run(c2)

    from pipeline_core.podcast.theme import theme_for_language
    return RenderResult(
        edit_path=edit_path,
        promo_169_path=promo_169,
        promo_916_path=promo_916,
        edit_expected_s=expected,
        renderer_used="ffmpeg",
        theme=theme_for_language(language),
    )


__all__ = [
    "RenderRequest",
    "RenderResult",
    "render_podcast",
    "ffprobe_duration",
]
