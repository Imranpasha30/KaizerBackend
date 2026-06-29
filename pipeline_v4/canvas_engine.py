"""V4 Step 2 — Canvas compositor.

Takes a Canvas (see canvas_schema.py) plus a trimmed video and
produces the final mp4. Single ffmpeg pass:

  - Build a background layer of canvas.layout (color + size)
  - Scale + place the trimmed video on the canvas
  - For each story: layer its images at the right t_start/t_end with
    ``enable='between(t,X,Y)'`` so they appear only during that
    story's segment
  - For each story: layer its text panels (lower-third, headline,
    ticker) the same way
  - Place brand logo if any
  - -c:a copy → audio passes through untouched, no drift possible

This entire step is replayable: edit canvas.json, re-run, get a new
mp4 in 5-15s. Step 1's trimmed video is sunk cost.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from pipeline_v4.encoder import video_encoder_args as _enc_args
from pipeline_v4.encoder import video_decoder_args as _dec_args
from pipeline_v4.ffmpeg_exec import run_ffmpeg as _run_ffmpeg

from pipeline_v4.canvas_schema import Canvas, CanvasImage, CanvasStory, CanvasTextBlock
from pipeline_v4.text_renderer import render_text_panel_png


def _pct(value_pct: float, total: int) -> int:
    return int(round((value_pct / 100.0) * total))


# ─── Background-video resolution ─────────────────────────────────────
# bg_video_path on the canvas is a logical reference; the renderer needs
# an absolute file path. Two forms are recognised:
#   "sample:<filename>"  → the bundled demo videos shipped with the
#                          frontend (public/video/<filename>).
#   "asset:<asset_id>"   → a UserAsset row the operator uploaded earlier.
# Anything else is treated as a literal path (also accepted for ad-hoc
# scripts / tests). When the file can't be located the renderer falls
# back to the flat bg_color silently — never crashes the render.

# Frontend public dir lives next to KaizerBackend in dev. In production
# KAIZER_V4_BG_SAMPLES_DIR can point at wherever the samples were copied
# to inside the backend image / container.
_DEFAULT_SAMPLES_DIR = _BACKEND_ROOT.parent / "kaizerFrontned" / "public" / "video"


def bg_video_samples_dir() -> Path:
    return Path(os.environ.get("KAIZER_V4_BG_SAMPLES_DIR", str(_DEFAULT_SAMPLES_DIR)))


def _resolve_bg_video_path(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("sample:"):
        name = raw[len("sample:"):]
        p = bg_video_samples_dir() / name
        return str(p) if p.is_file() else None
    if raw.startswith("asset:"):
        try:
            asset_id = int(raw[len("asset:"):])
        except ValueError:
            return None
        try:
            from database import SessionLocal
            import models as _m
            with SessionLocal() as db:
                a = db.query(_m.UserAsset).filter(_m.UserAsset.id == asset_id).first()
                if a and a.file_path and Path(a.file_path).is_file():
                    return a.file_path
        except Exception as exc:
            print(f"[v4/render] bg asset lookup failed: {exc}", flush=True)
        return None
    # Literal path
    return raw if Path(raw).is_file() else None


def _probe_has_audio(path: str, ffmpeg_bin: str) -> bool:
    """Cheap ffprobe to see whether a media file carries an audio stream.
    Used to decide whether the renderer can amix bg audio at all — if
    not, we skip the volume mix even when the user dialed it up, so
    ffmpeg doesn't 1) error on a missing [N:a] label or 2) silently
    write a video with no sound."""
    if not path:
        return False
    # Resolve sibling ffprobe next to ffmpeg.
    bin_dir = Path(ffmpeg_bin).parent
    candidates = [
        str(bin_dir / ("ffprobe.exe" if os.name == "nt" else "ffprobe")),
        "ffprobe",
    ]
    for probe in candidates:
        try:
            r = subprocess.run(
                [probe, "-v", "error", "-select_streams", "a",
                 "-show_entries", "stream=index", "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=10,
            )
            return bool(r.stdout.strip())
        except FileNotFoundError:
            continue
        except Exception as exc:
            print(f"[v4/render] ffprobe failed: {exc}", flush=True)
            return False
    return False


def _default_text_geom(
    kind: str,
    canvas_w: int,
    canvas_h: int,
) -> dict:
    """Reasonable defaults for each text-block kind so most canvases
    don't need to set per-block positions. The operator can override
    by setting x_pct / y_pct / w_pct on the block itself."""
    if kind == "lower_third":
        # Red strap, full-width, sits in 79-90% Y band — the BREAKING
        # headline. Bold white text on red, news-channel style.
        # Y/H tuned to butt up against the video+image row (which now
        # takes 79% of canvas height in both bulletin and short).
        return {
            "x_px": _pct(0, canvas_w),
            "y_px": _pct(79, canvas_h),
            "w_px": _pct(100, canvas_w),
            "h_px": _pct(11, canvas_h),
            "font_size_px": _pct(5.0, canvas_h),
            "fg":  "#FFFFFF",
            "bg":  "#C10000",
            "align": "left",
            "bold": True,
        }
    if kind == "ticker":
        # Yellow ticker, full-width, bottom 10%. Sits directly below
        # the red strap with no gap.
        return {
            "x_px": _pct(0, canvas_w),
            "y_px": _pct(90, canvas_h),
            "w_px": _pct(100, canvas_w),
            "h_px": _pct(10, canvas_h),
            "font_size_px": _pct(3.5, canvas_h),
            "fg":  "#000000",
            "bg":  "#FFD400",
            "align": "left",
            "bold": True,
        }
    if kind == "headline":
        return {
            "x_px": _pct(2, canvas_w),
            "y_px": _pct(4, canvas_h),
            "w_px": _pct(60, canvas_w),
            "h_px": _pct(10, canvas_h),
            "font_size_px": _pct(5, canvas_h),
            "fg":  "#FFFFFF",
            "bg":  None,
            "align": "left",
            "bold": True,
        }
    if kind == "watermark":
        return {
            "x_px": _pct(88, canvas_w),
            "y_px": _pct(92, canvas_h),
            "w_px": _pct(11, canvas_w),
            "h_px": _pct(5, canvas_h),
            "font_size_px": _pct(2.5, canvas_h),
            "fg":  "#FFFFFFCC",
            "bg":  None,
            "align": "right",
            "bold": False,
        }
    # "custom" or unknown
    return {
        "x_px": _pct(50, canvas_w),
        "y_px": _pct(50, canvas_h),
        "w_px": _pct(50, canvas_w),
        "h_px": _pct(10, canvas_h),
        "font_size_px": _pct(3, canvas_h),
        "fg":  "#FFFFFF",
        "bg":  None,
        "align": "center",
        "bold": True,
    }


def _resolve_block_geom(block: CanvasTextBlock, canvas_w: int, canvas_h: int) -> dict:
    """Apply user-set overrides on top of the kind-defaults."""
    g = _default_text_geom(block.kind, canvas_w, canvas_h)
    if block.x_pct is not None:    g["x_px"] = _pct(block.x_pct, canvas_w)
    if block.y_pct is not None:    g["y_px"] = _pct(block.y_pct, canvas_h)
    if block.w_pct is not None:    g["w_px"] = _pct(block.w_pct, canvas_w)
    if block.font_size_pct is not None:
        g["font_size_px"] = _pct(block.font_size_pct, canvas_h)
    if block.fg_color: g["fg"] = block.fg_color
    if block.bg_color is not None: g["bg"] = block.bg_color or None
    return g


def _ffescape(p: str) -> str:
    """Escape a path for use INSIDE a filter_complex expression. ffmpeg
    treats ':' specially inside filtergraph option lists; we double-
    escape it. Also handle Windows backslashes."""
    return p.replace("\\", "/").replace(":", "\\:")


def _build_text_pngs(
    *,
    canvas: Canvas,
    out_dir: Path,
) -> list[dict]:
    """Render every text-block PNG up front so we can pass them as
    ffmpeg inputs. Returns a list of {png_path, story_video_t_start,
    story_video_t_end, abs_t_start, abs_t_end, x_px, y_px, w_px, h_px}.

    Each text block's absolute time = story.video_t_start + block.t_start.
    """
    out: list[dict] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    counter = 0
    for st in canvas.stories:
        for block in st.text_blocks:
            geom = _resolve_block_geom(block, canvas.layout.width, canvas.layout.height)
            png_path = out_dir / f"text_s{st.story_index:02d}_{counter:02d}.png"
            counter += 1
            render_text_panel_png(
                text=block.text,
                out_path=str(png_path),
                width=max(8, geom["w_px"]),
                height=max(8, geom["h_px"]),
                font_size=max(8, geom["font_size_px"]),
                fg_color=geom["fg"],
                bg_color=geom["bg"],
                bold=geom["bold"],
                align=geom["align"],
                valign="middle",
                padding_px=int(geom["h_px"] * 0.10),
            )
            abs_t_start = st.video_t_start + (block.t_start or 0.0)
            if block.t_end is None:
                abs_t_end = st.video_t_end
            else:
                abs_t_end = st.video_t_start + block.t_end
            out.append({
                "png_path": str(png_path),
                "x_px": geom["x_px"],
                "y_px": geom["y_px"],
                "abs_t_start": abs_t_start,
                "abs_t_end": abs_t_end,
            })
    return out


def _resolve_image_path(filename: str, pool_dir: Path) -> Optional[str]:
    """Return an absolute path to the image, or None if not found."""
    p = pool_dir / filename
    if p.is_file():
        return str(p)
    # Allow operator to drop arbitrary absolute paths through
    if os.path.isabs(filename) and os.path.isfile(filename):
        return filename
    return None


def _enumerate_image_overlays(
    *,
    canvas: Canvas,
    pool_dir: Path,
) -> list[dict]:
    """Return ordered list of image overlay entries:
       {abs_path, abs_t_start, abs_t_end, x_px, y_px, w_px, h_px}.
    Skips images whose files can't be resolved."""
    out: list[dict] = []
    layout = canvas.layout
    x_px = _pct(layout.picture_x_pct, layout.width)
    y_px = _pct(layout.picture_y_pct, layout.height)
    w_px = _pct(layout.picture_w_pct, layout.width)
    h_px = _pct(layout.picture_h_pct, layout.height)
    for st in canvas.stories:
        for img in st.images:
            abs_path = _resolve_image_path(img.src, pool_dir)
            if not abs_path:
                continue
            abs_t_start = st.video_t_start + (img.t_start or 0.0)
            abs_t_end = st.video_t_start + (img.t_end or 0.0)
            if abs_t_end <= abs_t_start + 0.05:
                continue
            # Clamp effect duration so the in + out windows fit inside the
            # visible window. If the user set effect_duration=1.0 on a
            # 1.2-s image, force it down so we don't end up with an
            # always-faded image that never reaches full opacity.
            ed_raw = float(getattr(img, "effect_duration", 0.4) or 0.0)
            window = abs_t_end - abs_t_start
            ed = max(0.0, min(ed_raw, window / 2.0))
            out.append({
                "abs_path": abs_path,
                "abs_t_start": abs_t_start,
                "abs_t_end": abs_t_end,
                "x_px": x_px, "y_px": y_px,
                "w_px": w_px, "h_px": h_px,
                "effect": getattr(img, "effect", "cut") or "cut",
                "effect_duration": ed,
                "fit": getattr(img, "fit", "cover") or "cover",
                "offset_x_pct": float(getattr(img, "offset_x_pct", 50.0) or 0.0),
                "offset_y_pct": float(getattr(img, "offset_y_pct", 50.0) or 0.0),
            })
    return out


def render_canvas(
    *,
    canvas: Canvas,
    output_dir: str,
    pool_dir: Optional[str] = None,
    ffmpeg_bin: str = "ffmpeg",
) -> str:
    """Composite ``canvas`` onto ``canvas.trimmed_video_path`` and
    write the result to ``output_dir/<canvas.output_filename>``.

    Returns the absolute path to the rendered video.

    KAIZER_CLEAN_MASTER (Decision 1, see docs/upload-rewrite/DECISIONS.md):
    when the env var is ``"1"``, the brand-logo overlay block below
    is SKIPPED entirely. This produces a clean MasterVideo with no
    logo baked in — the Phase 2 Branding Worker (services/branding)
    then does the only logo overlay pass downstream. Default is
    ``"0"`` (Decision 12) so existing renders keep their current
    behaviour until ops flip the flag at Phase 3 cutover.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = Path(pool_dir or out_dir / "_pool")
    pool_path.mkdir(parents=True, exist_ok=True)

    text_pngs_dir = out_dir / "_text_pngs"
    text_overlays = _build_text_pngs(canvas=canvas, out_dir=text_pngs_dir)
    image_overlays = _enumerate_image_overlays(canvas=canvas, pool_dir=pool_path)

    # ─── Resolve optional background video ─────────────────────────
    # Replaces the flat bg_color with a looping mp4 (e.g. abstract news
    # studio b-roll) so the bulletin reads as a real broadcast, not a
    # PowerPoint over black. Falls back silently to color when the field
    # is missing or the file can't be found.
    bg_video_abs: Optional[str] = _resolve_bg_video_path(getattr(canvas.layout, "bg_video_path", None))
    bg_video_volume = float(getattr(canvas.layout, "bg_video_volume", 0.0) or 0.0)

    # ─── Build the ffmpeg command ───────────────────────────────────
    layout = canvas.layout
    # GPU decode of the trimmed video when NVENC is active. Input
    # option — precedes the FIRST -i only (PNG/text inputs keep their
    # own decoders). Input INDICES are unaffected by input options, so
    # the bg_in_idx math below stays valid.
    inputs: list[str] = [*_dec_args(), "-i", canvas.trimmed_video_path]
    # Each text overlay adds one input
    for o in text_overlays:
        inputs += ["-i", o["png_path"]]
    # Each image overlay adds one input
    for o in image_overlays:
        inputs += ["-loop", "1", "-i", o["abs_path"]]

    # filter_complex chain
    chain: list[str] = []
    # 1) Background layer — looping bg video OR flat color, picked at
    #    render time based on layout.bg_video_path. Both produce a label
    #    named [bg] that the rest of the chain composites onto.
    if bg_video_abs:
        # Add bg as a fresh input AFTER all overlays so we don't have to
        # renumber image/text overlay input indices. `-stream_loop -1`
        # repeats the file for the lifetime of the output; `-shortest`
        # downstream cuts when the trimmed video ends.
        bg_in_idx = 1 + len(text_overlays) + len(image_overlays)
        inputs += ["-stream_loop", "-1", "-i", bg_video_abs]
        chain.append(
            f"[{bg_in_idx}:v]scale={layout.width}:{layout.height}:"
            f"force_original_aspect_ratio=increase,"
            f"crop={layout.width}:{layout.height},setsar=1,fps=30[bg]"
        )
    else:
        bg_in_idx = None
        chain.append(
            f"color=c={layout.bg_color}:s={layout.width}x{layout.height}:r=30[bg]"
        )
    # 2) Scale + place the trimmed video on the canvas
    vid_w = _pct(layout.video_w_pct, layout.width)
    vid_h = _pct(layout.video_h_pct, layout.height)
    vid_x = _pct(layout.video_x_pct, layout.width)
    vid_y = _pct(layout.video_y_pct, layout.height)
    # Cover-crop, not letterbox: fills the video panel completely so
    # no black bars show inside it when the source AR doesn't match.
    # Matches V1's longform_compose approach (increase + crop).
    chain.append(
        f"[0:v]scale={vid_w}:{vid_h}:force_original_aspect_ratio=increase,"
        f"crop={vid_w}:{vid_h},setsar=1[vsrc]"
    )
    chain.append(f"[bg][vsrc]overlay=x={vid_x}:y={vid_y}:shortest=1[layer0]")
    layer = "layer0"
    layer_n = 1

    # 3) Image overlays — each scaled then overlaid with timed visibility
    in_idx = 1
    for o in text_overlays:
        # text PNGs are pre-sized; just overlay
        next_layer = f"layer{layer_n}"
        enable = f"between(t,{o['abs_t_start']:.3f},{o['abs_t_end']:.3f})"
        chain.append(
            f"[{layer}][{in_idx}:v]overlay=x={o['x_px']}:y={o['y_px']}:"
            f"enable='{enable}'[{next_layer}]"
        )
        layer = next_layer
        layer_n += 1
        in_idx += 1

    for o in image_overlays:
        # Scale image to the picture-panel size, then apply the chosen
        # transition effect, then overlay with timed visibility.
        scaled_label = f"img{layer_n}"
        ts = o["abs_t_start"]
        te = o["abs_t_end"]
        ed = o["effect_duration"]
        effect = o["effect"]
        # Base scale chain — every effect starts from a panel-sized RGBA frame.
        # fit=cover : scale-to-fill then crop with the user-chosen focal
        #             offset (50/50 = center, 0/0 = top-left, etc.).
        # fit=contain: scale-to-fit and pad with bg so nothing is cropped.
        if o["fit"] == "contain":
            scale_filter = (
                f"[{in_idx}:v]scale={o['w_px']}:{o['h_px']}:"
                f"force_original_aspect_ratio=decrease,"
                f"pad={o['w_px']}:{o['h_px']}:(ow-iw)/2:(oh-ih)/2:color={layout.bg_color},"
                f"setsar=1"
            )
        else:
            ox = max(0.0, min(1.0, o["offset_x_pct"] / 100.0))
            oy = max(0.0, min(1.0, o["offset_y_pct"] / 100.0))
            scale_filter = (
                f"[{in_idx}:v]scale={o['w_px']}:{o['h_px']}:"
                f"force_original_aspect_ratio=increase,"
                f"crop={o['w_px']}:{o['h_px']}:(in_w-{o['w_px']})*{ox:.3f}:(in_h-{o['h_px']})*{oy:.3f},"
                f"setsar=1"
            )
        # Fade is implemented on the IMAGE (alpha fade) so it blends
        # against whatever layer is underneath — that gives a true
        # crossfade when two images overlap in time, not just a hard cut.
        if effect == "fade" and ed > 0.0:
            scale_filter += (
                f",format=yuva420p,"
                f"fade=t=in:st={ts:.3f}:d={ed:.3f}:alpha=1,"
                f"fade=t=out:st={te - ed:.3f}:d={ed:.3f}:alpha=1"
            )
        chain.append(f"{scale_filter}[{scaled_label}]")

        next_layer = f"layer{layer_n}"
        # Slide effects animate the overlay's x position; everything else
        # holds x at the panel anchor.
        x_expr = str(o["x_px"])
        if effect in ("slide_left", "slide_right") and ed > 0.0:
            # slide_left  enters from the RIGHT off-canvas, exits to the LEFT
            # slide_right enters from the LEFT  off-canvas, exits to the RIGHT
            #   - in window  [ts, ts+ed]      : animate to anchor
            #   - hold       [ts+ed, te-ed]   : park at anchor (x_px)
            #   - out window [te-ed, te]      : animate off again
            x_px = o["x_px"]
            if effect == "slide_left":
                enter_from = "W"            # off-canvas right
                exit_to    = "-w"           # off-canvas left
            else:
                enter_from = "-w"           # off-canvas left
                exit_to    = "W"            # off-canvas right
            x_expr = (
                f"'if(lt(t,{ts + ed:.3f}),"
                f"{enter_from}+({x_px}-({enter_from}))*((t-{ts:.3f})/{ed:.3f}),"
                f"if(lt(t,{te - ed:.3f}),"
                f"{x_px},"
                f"{x_px}+(({exit_to})-{x_px})*((t-{te - ed:.3f})/{ed:.3f})"
                f"))'"
            )
        enable = f"between(t,{ts:.3f},{te:.3f})"
        chain.append(
            f"[{layer}][{scaled_label}]overlay="
            f"x={x_expr}:y={o['y_px']}:enable='{enable}'[{next_layer}]"
        )
        layer = next_layer
        layer_n += 1
        in_idx += 1

    # 4) Brand logo (optional)
    #
    # KAIZER_CLEAN_MASTER (Decision 1 in DECISIONS.md, gated by Decision 12):
    # When the env var is "1", skip the entire logo overlay block — no
    # ``-i logo`` input, no scale+overlay chain entries. The new Phase 2
    # Branding Worker (services/branding) owns the only logo overlay
    # pass; baking the logo here too would double-stamp the artifact.
    # Default is "0" so existing pipelines keep producing bit-identical
    # output until ops flip the flag at Phase 3 cutover. See
    # docs/upload-rewrite/DECISIONS.md Decision 1.
    _clean_master = os.environ.get("KAIZER_CLEAN_MASTER", "0").strip() == "1"
    if _clean_master:
        import logging as _lg
        _lg.getLogger("kaizer.pipeline_v4").info(
            "canvas_engine.render_canvas: KAIZER_CLEAN_MASTER=1 — skipping "
            "brand_logo overlay (Decision 1; branding worker handles it)"
        )
    elif layout.brand_logo_path and os.path.isfile(layout.brand_logo_path):
        inputs += ["-i", layout.brand_logo_path]
        logo_w_px = _pct(layout.brand_logo_w_pct, layout.width)
        logo_x_px = _pct(layout.brand_logo_x_pct, layout.width)
        logo_y_px = _pct(layout.brand_logo_y_pct, layout.height)
        chain.append(f"[{in_idx}:v]scale={logo_w_px}:-1[logo]")
        next_layer = f"layer{layer_n}"
        chain.append(
            f"[{layer}][logo]overlay=x={logo_x_px}:y={logo_y_px}[{next_layer}]"
        )
        layer = next_layer
        layer_n += 1
        in_idx += 1

    # Final tag
    chain.append(f"[{layer}]copy[vout]")

    filter_complex = ";".join(chain)
    output_path = str(out_dir / canvas.output_filename)

    # ─── Audio handling ────────────────────────────────────────────
    # Default: passthrough the trimmed video's audio (zero drift, fast).
    # When a bg video is set AND the user asked for non-zero volume on
    # it, build a separate amix filter that combines the two tracks.
    # That forces re-encoding (`-c:a aac`) because amix is a filter, not
    # a copy op.
    audio_filter = ""
    audio_map = "0:a?"
    audio_codec = ["-c:a", "copy"]
    if bg_in_idx is not None and bg_video_volume > 0.0 and _probe_has_audio(bg_video_abs, ffmpeg_bin):
        audio_filter = (
            f"[0:a]volume=1.0[a0];"
            f"[{bg_in_idx}:a]volume={bg_video_volume:.3f}[abg];"
            f"[a0][abg]amix=inputs=2:duration=first:dropout_transition=0[aout]"
        )
        audio_map = "[aout]"
        audio_codec = ["-c:a", "aac", "-b:a", "192k"]
    if audio_filter:
        filter_complex = filter_complex + ";" + audio_filter

    # When the user asked for an intro reel, write the main composite
    # to an inner temp file and concat with the intro afterwards. Single
    # output_path is overwritten so downstream code (clip rows, editor)
    # doesn't care about the two-step path.
    intro_seconds = float(getattr(canvas.layout, "bg_intro_seconds", 0.0) or 0.0)
    use_intro = bg_video_abs and intro_seconds > 0.0
    inner_path = str(out_dir / ("__inner_" + canvas.output_filename)) if use_intro else output_path

    cmd = [ffmpeg_bin, "-y", "-v", "error"]
    cmd += inputs
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", audio_map,
        *_enc_args(crf=20, preset_hint="medium"),
        "-pix_fmt", "yuv420p",
    ] + audio_codec + [
        "-shortest",
        "-movflags", "+faststart",
        inner_path,
    ]

    print(f"[v4/step2] compositing {len(text_overlays)} text + {len(image_overlays)} image overlays -> {canvas.output_filename}", flush=True)
    try:
        # Centralised runner: retry + NVENC→libx264 fallback + stderr
        # tail logging. Raises RuntimeError on final failure.
        _run_ffmpeg(cmd, timeout=60 * 20, log_label="canvas_render")
    except RuntimeError as exc:
        # Persist the filter graph alongside the failure for debugging
        try:
            (out_dir / "_failed_filter.txt").write_text(filter_complex, encoding="utf-8")
        except OSError:
            pass
        raise RuntimeError(f"canvas render failed: {exc}") from exc

    if use_intro:
        intro_path = str(out_dir / "_intro.mp4")
        try:
            _render_intro_clip(
                bg_video_path=bg_video_abs,
                duration_s=intro_seconds,
                width=layout.width, height=layout.height,
                out_path=intro_path,
                ffmpeg_bin=ffmpeg_bin,
            )
            _concat_intro_with_main(
                intro_path=intro_path,
                main_path=inner_path,
                out_path=output_path,
                ffmpeg_bin=ffmpeg_bin,
                intro_duration_s=intro_seconds,
                crossfade_s=0.8,
            )
        except Exception as exc:
            # Intro failures shouldn't take down the whole render — fall
            # back to the inner result so the operator still gets a
            # bulletin out the door.
            print(f"[v4/step2] intro stage failed ({exc}); falling back to main render", flush=True)
            try:
                if Path(inner_path).exists():
                    Path(inner_path).replace(output_path)
            except OSError:
                pass
        else:
            # Clean up the intermediates on success.
            for p in (inner_path, intro_path):
                try: Path(p).unlink(missing_ok=True)
                except OSError: pass

    print(f"[v4/step2]   done -> {output_path}", flush=True)
    return output_path


def _render_intro_clip(*, bg_video_path: str, duration_s: float,
                        width: int, height: int, out_path: str,
                        ffmpeg_bin: str) -> None:
    """Render the first ``duration_s`` of the bg video, scaled to canvas
    size, at full audio volume. Becomes the leader the operator sees
    before the bulletin layout kicks in."""
    has_audio = _probe_has_audio(bg_video_path, ffmpeg_bin)
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1,fps=30"
    )
    cmd = [
        ffmpeg_bin, "-y", "-v", "error",
        "-stream_loop", "-1", "-t", f"{duration_s:.3f}", "-i", bg_video_path,
        "-vf", vf,
        *_enc_args(crf=20, preset_hint="medium"),
        "-pix_fmt", "yuv420p",
    ]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        # Fabricate a silent track so concat with the main pass (which
        # always has audio) doesn't desync. lavfi anullsrc must come AS
        # AN INPUT, not a filter on the existing input.
        cmd[cmd.index("-stream_loop"):cmd.index(bg_video_path)+1] = (
            ["-stream_loop", "-1", "-t", f"{duration_s:.3f}", "-i", bg_video_path,
             "-f", "lavfi", "-t", f"{duration_s:.3f}", "-i",
             "anullsrc=channel_layout=stereo:sample_rate=44100"]
        )
        cmd += ["-map", "0:v", "-map", "1:a",
                "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-movflags", "+faststart", out_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60 * 5)
    if proc.returncode != 0:
        raise RuntimeError(f"intro render failed: {proc.stderr[-800:]}")


def _concat_intro_with_main(*, intro_path: str, main_path: str,
                             out_path: str, ffmpeg_bin: str,
                             intro_duration_s: float = 0.0,
                             crossfade_s: float = 0.8) -> None:
    """Join intro + main with a smooth crossfade (xfade for video,
    acrossfade for audio) so the hand-off doesn't look like a hard cut.
    The transition starts ``crossfade_s`` seconds before the intro ends,
    so the intro's tail dissolves into the bulletin's head. When the
    intro is too short for the transition to fit, falls back to a hard
    concat — better a hard cut than ffmpeg erroring on negative offset."""
    use_xfade = crossfade_s > 0.05 and intro_duration_s > crossfade_s + 0.1
    if use_xfade:
        offset = max(0.0, intro_duration_s - crossfade_s)
        filter_complex = (
            f"[0:v][1:v]xfade=transition=fade:"
            f"duration={crossfade_s:.3f}:offset={offset:.3f}[v];"
            f"[0:a][1:a]acrossfade=d={crossfade_s:.3f}:c1=tri:c2=tri[a]"
        )
    else:
        filter_complex = "[0:v:0][0:a:0][1:v:0][1:a:0]concat=n=2:v=1:a=1[v][a]"
    cmd = [
        ffmpeg_bin, "-y", "-v", "error",
        "-i", intro_path, "-i", main_path,
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "[a]",
        *_enc_args(crf=20, preset_hint="medium"),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60 * 20)
    if proc.returncode != 0:
        raise RuntimeError(f"concat failed: {proc.stderr[-800:]}")
