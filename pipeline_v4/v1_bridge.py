"""V4 -> V1 layout bridge.

V4's Step 1 produces a trimmed bulletin video and per-story short clips
with frame-accurate cuts and audio passthrough (zero lipsync drift).

This module hands those clips off to V1's proven layout composers in
``pipeline_core``:

  * ``compose_bulletin_story`` — per-story 1920x1080 broadcast frame
    with main video left, sidebar right, lower-third slide-in animation,
    and a horizontally-scrolling news ticker. Stitched via
    ``stitch_bulletin``.
  * ``compose_clip`` (torn_card), ``compose_clip_clean_card``,
    ``compose_split_frame``, ``compose_follow_bar`` — the four 1080x1920
    shorts layouts the team already maintains.

Why a bridge instead of a rewrite: V1's compose functions are pure
(input paths -> output path, no DB / globals). Calling them directly
keeps the V4 pipeline DRY — we get V1's exact animations, fonts, and
visual polish without re-deriving any of the filtergraph math.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Bump when the per-story compose filter graph changes in a way that
# would produce different pixels from the same canvas inputs. Treated
# as a string ingredient in the hash so old cached clips invalidate
# automatically the next time render_bulletin runs.
_PER_STORY_RENDERER_VERSION = "v4-2026-06-06-no-ticker"


def _per_story_cache_hash(*,
                          story,
                          ticker_path: str,  # kept in signature for back-compat; ignored now
                          sidebar_path: str,
                          layout,
                          channel_bug_path: str,
                          watermark_path: str,
                          watermark_position: str,
                          font_path: str,
                          bg_video_abs: Optional[str],
                          bg_video_volume: float,
                          language_code: str) -> str:
    """Deterministic short hash of every input that influences the
    rendered story clip. If anything in here changes, the cached
    composed_story_NN.mp4 is stale and must be re-rendered."""
    layout_keys = ("width", "height", "bg_color",
                   "video_x_pct", "video_y_pct", "video_w_pct", "video_h_pct",
                   "picture_x_pct", "picture_y_pct", "picture_w_pct", "picture_h_pct",
                   "brand_logo_path",
                   "brand_logo_x_pct", "brand_logo_y_pct", "brand_logo_w_pct",
                   "bg_video_path", "bg_video_volume", "bg_intro_seconds")
    layout_blob = {}
    if layout is not None:
        for k in layout_keys:
            v = getattr(layout, k, None)
            if isinstance(v, float):
                v = round(v, 4)
            layout_blob[k] = v
    # _file_fingerprint: include path + size + mtime so a same-named
    # ticker.png with different content invalidates the cache.
    def _fp(p: Optional[str]) -> dict:
        if not p:
            return {"p": None}
        try:
            st = os.stat(p)
            return {"p": str(p), "s": st.st_size, "m": int(st.st_mtime)}
        except OSError:
            return {"p": str(p), "s": 0, "m": 0}
    blob = {
        "renderer": _PER_STORY_RENDERER_VERSION,
        "language": language_code,
        "story": {
            "title_native":  getattr(story, "title_native", "") or "",
            "title_english": getattr(story, "title_english", "") or "",
            "summary":       getattr(story, "summary", "") or "",
            "video_t_start": round(float(getattr(story, "video_t_start", 0.0)), 3),
            "video_t_end":   round(float(getattr(story, "video_t_end", 0.0)), 3),
            "story_index":   int(getattr(story, "story_index", 0) or 0),
            "total_stories": int(getattr(story, "total_stories", 0) or 0),
        },
        # NOTE: ticker_path intentionally excluded — the ticker is now
        # overlaid AFTER stitch, so per-story clips no longer depend
        # on the headline list. Title edits only invalidate the
        # ticker-overlay pass, not the per-story cache.
        "sidebar":  _fp(sidebar_path),
        "bug":      _fp(channel_bug_path),
        "watermark": _fp(watermark_path),
        "watermark_position": watermark_position or "",
        "font":     _fp(font_path),
        "bg_video": _fp(bg_video_abs),
        "bg_video_volume": round(float(bg_video_volume or 0.0), 3),
        "layout":   layout_blob,
    }
    raw = json.dumps(blob, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _read_cached_hash(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""


def _write_cached_hash(path: str, value: str) -> None:
    try:
        Path(path).write_text(value, encoding="utf-8")
    except OSError as exc:
        print(f"[v4/v1_bridge] cache hash write failed: {exc}", flush=True)


# ── Bg-video helpers ─────────────────────────────────────────────────
# These let v1_bridge honour CanvasLayout.bg_video_path / bg_intro_seconds
# on the initial render too — not only on editor re-render. Without this
# the operator's choice in the new-job wizard would be silently dropped
# until they hit Re-render in the editor.

def _resolve_layout_bg_video(layout) -> Optional[str]:
    if layout is None:
        return None
    ref = (getattr(layout, "bg_video_path", None) or "").strip()
    if not ref:
        return None
    try:
        from pipeline_v4.canvas_engine import _resolve_bg_video_path
        return _resolve_bg_video_path(ref)
    except Exception as exc:
        print(f"[v4/v1_bridge] bg resolver failed: {exc}", flush=True)
        return None


def _render_bg_intro_clip(*, bg_video_path: str, duration_s: float,
                           width: int, height: int, out_path: str) -> None:
    """Render the first ``duration_s`` of the bg video, scaled to canvas
    size, with full audio. Prepended to the bulletin as the cold-open."""
    ffmpeg = _ffmpeg_bin()
    # ffprobe for audio presence — silent intro if the source has no
    # audio so the concat still succeeds.
    has_audio = _file_has_audio(bg_video_path)
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1,fps=30"
    )
    if has_audio:
        cmd = [
            ffmpeg, "-y", "-v", "error",
            "-stream_loop", "-1", "-t", f"{duration_s:.3f}", "-i", bg_video_path,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-r", "30", "-fps_mode", "cfr",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart", out_path,
        ]
    else:
        cmd = [
            ffmpeg, "-y", "-v", "error",
            "-stream_loop", "-1", "-t", f"{duration_s:.3f}", "-i", bg_video_path,
            "-f", "lavfi", "-t", f"{duration_s:.3f}", "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-vf", vf,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-r", "30", "-fps_mode", "cfr",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart", out_path,
        ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60 * 5)
    if proc.returncode != 0:
        raise RuntimeError(f"intro render failed: {proc.stderr[-800:]}")


def _concat_clips(intro_path: str, main_path: str, out_path: str,
                   intro_duration_s: float = 0.0,
                   crossfade_s: float = 0.8) -> None:
    """Join intro + main with a smooth crossfade so the hand-off doesn't
    look like a hard cut. ``intro_duration_s`` is the intro's runtime; the
    transition starts at ``intro_duration_s - crossfade_s`` so the intro's
    tail dissolves into the bulletin's head over ``crossfade_s`` seconds.
    Audio uses ``acrossfade`` for the matching volume blend.

    When ``crossfade_s <= 0`` or ``intro_duration_s`` is too short to fit
    the crossfade, falls back to a hard concat so we don't ever produce
    a black-out or a frozen frame."""
    ffmpeg = _ffmpeg_bin()
    # Need enough intro headroom for the crossfade — otherwise xfade's
    # offset goes negative and ffmpeg errors. Hard concat in that case.
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
        ffmpeg, "-y", "-v", "error",
        "-i", intro_path, "-i", main_path,
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-r", "30", "-fps_mode", "cfr",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60 * 20)
    if proc.returncode != 0:
        raise RuntimeError(f"concat failed: {proc.stderr[-800:]}")


def _overlay_ticker_post_stitch(*, bulletin_path: str, ticker_png_path: str,
                                  out_path: str,
                                  canvas_w: int, canvas_h: int,
                                  ticker_y: int,
                                  ticker_speed_px_s: float = 200.0) -> None:
    """Overlay a scrolling ticker onto the already-stitched bulletin.

    Moving the ticker out of the per-story composer means a title edit
    invalidates only the ticker.png + this one post-stitch pass, not
    every per-story compose. Cost: one re-encoding pass over the full
    bulletin instead of N concat-copy concat ops. Net win on incremental
    edits scales with N.
    """
    ffmpeg = _ffmpeg_bin()
    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-i", bulletin_path,
        "-loop", "1", "-i", ticker_png_path,
        "-filter_complex",
        f"[0:v][1:v]overlay="
        f"x='W-mod(t*{ticker_speed_px_s:.1f}\\,w+W)':y={ticker_y}:"
        f"format=auto:shortest=1[v]",
        "-map", "[v]",
        "-map", "0:a?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-r", "30", "-fps_mode", "cfr",
        "-c:a", "copy",
        "-movflags", "+faststart",
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60 * 20)
    if proc.returncode != 0:
        raise RuntimeError(f"ticker overlay failed: {proc.stderr[-800:]}")


def _file_has_audio(path: str) -> bool:
    """ffprobe quick check — returns False if the file has no audio
    streams (e.g. silent stock footage). Falls back to False on any
    probe error so we don't break the render."""
    if not path:
        return False
    ff = _ffmpeg_bin()
    candidates = [
        str(Path(ff).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")),
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
        except Exception:
            return False
    return False


# ── Defaults ──────────────────────────────────────────────────────────

SHORTS_PRESET = {"width": 1080, "height": 1920}
DEFAULT_SHORTS_LAYOUT = "torn_card"
SUPPORTED_SHORTS_LAYOUTS = ("torn_card", "clean_card", "split_frame", "follow_bar")

# ── V4 bulletin geometry (1920x1080, framed look) ─────────────────────
# V1's compose_bulletin_story has video and sidebar butting up against
# each other with no top margin. V4 wants visible spacing:
#   * 50 px top margin (mirrors the bottom gap above the lower-third)
#   * 30 px side margins
#   * 30 px gap between the video tile and the sidebar tile
#   * 3 px white border around both tiles
# Lower-third + ticker geometry stays identical to V1 so the animation
# expressions translate directly.
V4_W            = 1920
V4_H            = 1080
V4_TOP_MARGIN   = 50
V4_SIDE_MARGIN  = 30
V4_INNER_GAP    = 30
V4_TILE_BORDER  = 3
V4_TILE_OUTER_H = 800                            # combined border + content height
V4_TILE_INNER_H = V4_TILE_OUTER_H - 2 * V4_TILE_BORDER
V4_MAIN_OUTER_W = 1260
V4_SIDE_OUTER_W = 570
V4_MAIN_INNER_W = V4_MAIN_OUTER_W - 2 * V4_TILE_BORDER
V4_SIDE_INNER_W = V4_SIDE_OUTER_W - 2 * V4_TILE_BORDER
V4_MAIN_X       = V4_SIDE_MARGIN                                  # 30
V4_SIDE_X       = V4_SIDE_MARGIN + V4_MAIN_OUTER_W + V4_INNER_GAP # 30+1260+30=1320
V4_TILE_Y       = V4_TOP_MARGIN                                   # 50
V4_LT_H         = 140
V4_TICKER_H     = 50
V4_LT_Y         = V4_H - V4_LT_H - V4_TICKER_H                    # 890
V4_TICKER_Y     = V4_H - V4_TICKER_H                              # 1030


@dataclass
class BulletinRenderInputs:
    """Everything the V1 bulletin composer needs for one V4 job."""
    trimmed_bulletin_path: str
    stories: list                # list of trim_engine.TrimmedStory
    output_path: str             # final stitched bulletin.mp4
    work_dir: Path
    language: str = "te"
    brand_logo: Optional[str] = None
    sidebar_images: Optional[list[str]] = None   # one per story; None -> placeholder
    channel_name: str = ""                       # empty -> logo-only bug (no fallback text)
    layout: Optional[object] = None              # CanvasLayout drag-to-move overrides
    watermark_text: str = ""                     # semi-transparent overlay; empty -> logo only
    watermark_opacity: float = 0.35
    watermark_position: str = "top-right"        # top-left | top-right | bottom-left | bottom-right


@dataclass
class ShortRenderInputs:
    """Everything one V1 short composer call needs. Defaults mirror V1's
    Editor.jsx — populate only what the user changed."""
    trimmed_short_path: str
    title_text: str
    output_path: str
    work_dir: Path
    layout: str = DEFAULT_SHORTS_LAYOUT
    language: str = "te"
    image_path: Optional[str] = None
    brand_logo: Optional[str] = None
    thumbnail_path: Optional[str] = None             # split_frame only

    # V1-parity editor knobs (applied when set; otherwise the V1 default
    # path inside the composer kicks in).
    font_file: Optional[str] = None                  # TTF basename, e.g. "NotoSansTelugu-Bold.ttf"
    font_size: Optional[int] = None
    text_color: Optional[str] = None                 # hex
    section_pct: Optional[dict] = None               # {"video": .., "text": .., "image": ..}
    card_style: Optional[dict] = None                # torn-card knobs
    follow_params: Optional[dict] = None             # follow_bar layout
    watermark_text: str = ""
    watermark_opacity: float = 0.35
    watermark_position: str = "top-right"


# ── Helpers ───────────────────────────────────────────────────────────

def _lang_cfg(language: str):
    """Return the V1 LanguageConfig for the given ISO code, falling back
    to Telugu if unknown — V1 callers expect a non-None config."""
    from languages import LANGUAGES
    return LANGUAGES.get(language) or LANGUAGES["te"]


def _ffmpeg_bin() -> str:
    """Resolve ffmpeg binary via V1's own discovery (so both paths
    pick up the same one — important on Windows where PATH may not
    match the venv)."""
    try:
        from pipeline_core.pipeline import FFMPEG_BIN  # type: ignore
        if FFMPEG_BIN:
            return FFMPEG_BIN
    except Exception:
        pass
    return shutil.which("ffmpeg") or "ffmpeg"


def _slice_video(
    *,
    source_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
) -> str:
    """Cut [start_sec, end_sec) out of ``source_path`` into ``output_path``.

    Uses re-encode (-c:v libx264, AAC audio) so the resulting clip is
    safe to feed into V1's composers. Concat-style stream copy isn't
    reliable across container boundaries when the source has B-frames
    inside the cut range.
    """
    dur = max(0.05, float(end_sec) - float(start_sec))
    cmd = [
        _ffmpeg_bin(), "-y", "-v", "error",
        "-ss", f"{float(start_sec):.3f}",
        "-i", source_path,
        "-t",  f"{dur:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        output_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(
            f"_slice_video failed: {proc.stderr[-800:]}"
        )
    return output_path


def _resolve_sidebar(
    *,
    work_dir: Path,
    story_index: int,
    pool_image_path: Optional[str],
) -> str:
    """Return a sidebar PNG path (real image or generated placeholder)."""
    from pipeline_core.longform_compose import make_sidebar_placeholder
    out = work_dir / f"_sidebar_{story_index:02d}.png"
    return make_sidebar_placeholder(pool_image_path, str(out))


# ── Watermark + logo-only channel-bug helpers ─────────────────────────

def _render_logo_only_bug(
    *,
    logo_path: Optional[str],
    out_path: str,
    width: int = 200,
    height: int = 70,
) -> str:
    """Channel bug with NO baked-in text — just the logo on a rounded
    translucent plate. Replaces V1's ``render_channel_bug`` which falls
    back to literal "KAIZER NEWS" text when no channel_name is given,
    which then leaks across users.
    """
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    try:
        d.rounded_rectangle([0, 0, width - 1, height - 1],
                            radius=10, fill=(0, 0, 0, 170))
    except AttributeError:
        d.rectangle([0, 0, width - 1, height - 1], fill=(0, 0, 0, 170))
    if logo_path and os.path.isfile(logo_path):
        try:
            with Image.open(logo_path) as logo:
                logo = logo.convert("RGBA")
                target_h = height - 16
                ratio = target_h / max(1, logo.height)
                lw = int(logo.width * ratio)
                logo = logo.resize((lw, target_h), Image.LANCZOS)
                x = (width - lw) // 2
                img.paste(logo, (x, 8), logo)
        except Exception:
            pass
    img.save(out_path, "PNG")
    return out_path


def _render_watermark_png(
    *,
    text: str,
    logo_path: Optional[str],
    canvas_w: int,
    canvas_h: int,
    opacity: float,
    out_path: str,
    font_path: Optional[str] = None,
) -> str:
    """Build a transparent PNG carrying the user's watermark text +
    optional channel logo. Sized to a corner of the canvas so the
    ffmpeg overlay can drop it in without per-letter rendering.

    The text and logo are drawn at ``opacity`` (0-1). 0.35 default is
    the same translucency news channels use so the bug is visible
    without dominating the frame."""
    from PIL import Image, ImageDraw, ImageFont
    # ~15% of canvas width, scales nicely on both 1920x1080 and 1080x1920.
    plate_w = max(200, int(canvas_w * 0.18))
    plate_h = max(60, int(canvas_h * 0.06))
    img = Image.new("RGBA", (plate_w, plate_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    alpha = max(0, min(255, int(round(opacity * 255))))
    text_x = 12
    if logo_path and os.path.isfile(logo_path):
        try:
            with Image.open(logo_path) as logo:
                logo = logo.convert("RGBA")
                target_h = plate_h - 16
                ratio = target_h / max(1, logo.height)
                lw = int(logo.width * ratio)
                logo = logo.resize((lw, target_h), Image.LANCZOS)
                # Apply opacity to the logo alpha channel.
                if alpha < 255:
                    r, g, b, a = logo.split()
                    a = a.point(lambda v: int(v * (alpha / 255.0)))
                    logo = Image.merge("RGBA", (r, g, b, a))
                img.paste(logo, (8, 8), logo)
                text_x = 8 + lw + 10
        except Exception:
            pass

    if text:
        try:
            font_size = max(18, int(plate_h * 0.45))
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        # White text with the requested opacity, plus a 1-px shadow for
        # contrast on bright frames.
        d.text((text_x + 1, plate_h // 2 - 12 + 1), text[:30],
               font=font, fill=(0, 0, 0, alpha))
        d.text((text_x, plate_h // 2 - 12), text[:30],
               font=font, fill=(255, 255, 255, alpha))

    img.save(out_path, "PNG")
    return out_path


def _watermark_overlay_xy(position: str, canvas_w: int, canvas_h: int,
                          plate_w_expr: str = "w", plate_h_expr: str = "h",
                          margin: int = 40) -> tuple[str, str]:
    """Map a friendly corner name to ffmpeg overlay x/y expressions
    using runtime W/H (canvas) and w/h (overlay) refs."""
    p = (position or "top-right").lower()
    if p == "top-left":
        return f"{margin}", f"{margin}"
    if p == "bottom-left":
        return f"{margin}", f"H-h-{margin}"
    if p == "bottom-right":
        return f"W-w-{margin}", f"H-h-{margin}"
    return f"W-w-{margin}", f"{margin}"   # default top-right


# ── V4 bulletin story composer ────────────────────────────────────────

def _compose_v4_bulletin_story(
    *,
    story_clip_path: str,
    story_meta,                  # pipeline_core.longform_compose.StoryMeta
    out_path: str,
    sidebar_path: str,
    ticker_path: str,
    channel_bug_path: Optional[str],
    font_path: Optional[str],
    sidebar_is_video: bool = False,
    ticker_speed_px_s: float = 200.0,
    work_dir: str,
    layout=None,                 # optional CanvasLayout — drag-to-move overrides
    watermark_path: str = "",    # optional translucent overlay PNG
    watermark_position: str = "top-right",
    bg_video_path: Optional[str] = None,   # studio bg looped behind canvas
    bg_video_volume: float = 0.0,          # 0..1 mixed against story audio
    apply_ticker: bool = True,             # bake scrolling ticker into clip
) -> str:
    """V4-flavoured bulletin story composer.

    Mirrors V1's :func:`compose_bulletin_story` (same lower-third PNG,
    same ticker, same channel bug, same slide-in / marquee animation)
    but with the V4 framed-layout geometry: top margin, side margins,
    inner gap between video and sidebar, and a 3 px white border around
    each tile."""
    from pipeline_core.longform_compose import render_lower_third

    os.makedirs(work_dir, exist_ok=True)
    lt_path = os.path.join(work_dir, f"_lt_{story_meta.story_index:02d}.png")
    _, lt_w = render_lower_third(story_meta, font_path, lt_path)

    # Resolve geometry. When the editor's drag-to-move/resize has
    # written custom pcts into ``layout``, prefer those; otherwise
    # fall back to the V4 defaults so legacy canvases still render.
    canvas_w = V4_W
    canvas_h = V4_H
    if layout is not None:
        canvas_w = getattr(layout, "width",  None)  or V4_W
        canvas_h = getattr(layout, "height", None) or V4_H

    def _pct_to_px(pct: float, base: int) -> int:
        return max(0, int(round((pct / 100.0) * base)))

    if layout is not None and (layout.video_w_pct and layout.video_h_pct
                               and layout.picture_w_pct and layout.picture_h_pct):
        main_outer_w = _pct_to_px(layout.video_w_pct,   canvas_w)
        side_outer_w = _pct_to_px(layout.picture_w_pct, canvas_w)
        tile_outer_h = _pct_to_px(max(layout.video_h_pct, layout.picture_h_pct), canvas_h)
        main_x       = _pct_to_px(layout.video_x_pct,   canvas_w)
        side_x       = _pct_to_px(layout.picture_x_pct, canvas_w)
        tile_y       = _pct_to_px(layout.video_y_pct,   canvas_h)
    else:
        main_outer_w = V4_MAIN_OUTER_W
        side_outer_w = V4_SIDE_OUTER_W
        tile_outer_h = V4_TILE_OUTER_H
        main_x       = V4_MAIN_X
        side_x       = V4_SIDE_X
        tile_y       = V4_TILE_Y

    main_inner_w = max(1, main_outer_w - 2 * V4_TILE_BORDER)
    side_inner_w = max(1, side_outer_w - 2 * V4_TILE_BORDER)
    tile_inner_h = max(1, tile_outer_h - 2 * V4_TILE_BORDER)
    lt_y     = canvas_h - V4_LT_H - V4_TICKER_H
    ticker_y = canvas_h - V4_TICKER_H

    ffmpeg = _ffmpeg_bin()
    cmd: list[str] = [ffmpeg, "-y", "-v", "error", "-i", story_clip_path]
    if sidebar_is_video:
        cmd += ["-i", sidebar_path]
    else:
        cmd += ["-loop", "1", "-i", sidebar_path]
    cmd += ["-loop", "1", "-i", lt_path]
    # Ticker can be skipped per-story when the stitch-level overlay
    # path is in use — this is what lets title edits stay incremental
    # (changing a title invalidates only the ticker pass, not every
    # per-story compose).
    if apply_ticker:
        cmd += ["-loop", "1", "-i", ticker_path]
    has_bug = bool(channel_bug_path and os.path.isfile(channel_bug_path))
    if has_bug:
        cmd += ["-loop", "1", "-i", channel_bug_path]
    has_wm = bool(watermark_path and os.path.isfile(watermark_path))
    if has_wm:
        cmd += ["-loop", "1", "-i", watermark_path]
    # Studio bg video — last input so the existing per-feature input
    # indices stay stable. `-stream_loop -1` keeps the bg playing for
    # the entire story clip length; -shortest downstream truncates to
    # the story audio length.
    has_bg = bool(bg_video_path and os.path.isfile(bg_video_path))
    if has_bg:
        cmd += ["-stream_loop", "-1", "-i", bg_video_path]

    # Compute input indices dynamically — ticker may or may not exist.
    # Layout: 0=story, 1=sidebar, 2=lt, [ticker?], [bug?], [wm?], [bg?]
    _next_idx = 3
    ticker_in_idx = _next_idx if apply_ticker else None
    if apply_ticker: _next_idx += 1
    bug_in_idx = _next_idx if has_bug else None
    if has_bug: _next_idx += 1
    wm_in_idx = _next_idx if has_wm else None
    if has_wm: _next_idx += 1
    bg_input_idx = _next_idx if has_bg else None
    if has_bg: _next_idx += 1

    # Lower-third animation — identical branch to V1's so the slide-in /
    # marquee feels the same.
    if lt_w > canvas_w:
        lt_x_expr = (
            f"if(lt(t\\,0.4)\\,-w+w*t/0.4\\,"
            f"if(lt(t\\,2.0)\\,0\\,"
            f"max(W-w\\,-((t-2.0)*60))))"
        )
    else:
        lt_x_expr = "if(lt(t\\,0.4)\\,-w+w*t/0.4\\,0)"

    border_colour = "white"
    if layout is not None and getattr(layout, "bg_color", None):
        # bg_color is the canvas backdrop; the white border is a fixed
        # tile frame. Keep it white but allow override via a future
        # ``tile_border_color`` field without breaking existing canvases.
        border_colour = getattr(layout, "tile_border_color", None) or "white"

    fc = [
        f"[0:v]scale={main_inner_w}:{tile_inner_h}:"
        f"force_original_aspect_ratio=increase,"
        f"crop={main_inner_w}:{tile_inner_h},setsar=1,"
        f"pad={main_outer_w}:{tile_outer_h}:"
        f"{V4_TILE_BORDER}:{V4_TILE_BORDER}:color={border_colour}[main_v]",

        f"[1:v]scale={side_inner_w}:{tile_inner_h}:"
        f"force_original_aspect_ratio=increase,"
        f"crop={side_inner_w}:{tile_inner_h},setsar=1,"
        f"pad={side_outer_w}:{tile_outer_h}:"
        f"{V4_TILE_BORDER}:{V4_TILE_BORDER}:color={border_colour}[side_v]",
    ]
    # Background layer: looping bg video when present, else flat colour.
    if has_bg:
        fc.append(
            f"[{bg_input_idx}:v]scale={canvas_w}:{canvas_h}:"
            f"force_original_aspect_ratio=increase,"
            f"crop={canvas_w}:{canvas_h},setsar=1,fps=30[bg]"
        )
    else:
        fc.append(
            f"color=c={getattr(layout, 'bg_color', None) or 'black'}:"
            f"s={canvas_w}x{canvas_h}:r=30[bg]"
        )
    fc += [
        f"[bg][main_v]overlay=x={main_x}:y={tile_y}:shortest=1[stage_m]",
        f"[stage_m][side_v]overlay=x={side_x}:y={tile_y}[stage_top]",

        f"[stage_top][2:v]overlay=x='{lt_x_expr}':y={lt_y}:format=auto[stage_lt]",
    ]
    cursor = "stage_lt"
    if apply_ticker and ticker_in_idx is not None:
        fc.append(
            f"[{cursor}][{ticker_in_idx}:v]overlay="
            f"x='W-mod(t*{ticker_speed_px_s:.1f}\\,w+W)':"
            f"y={ticker_y}:format=auto[stage_ticker]"
        )
        cursor = "stage_ticker"
    if has_bug:
        fc.append(
            f"[{cursor}][{bug_in_idx}:v]overlay="
            f"x=W-w-{V4_SIDE_MARGIN}:y={tile_y}:format=auto[stage_bug]"
        )
        cursor = "stage_bug"
    last_stage = cursor

    if has_wm:
        wx, wy = _watermark_overlay_xy(watermark_position, canvas_w, canvas_h)
        fc.append(
            f"[{last_stage}][{wm_in_idx}:v]overlay="
            f"x={wx}:y={wy}:format=auto[outv]"
        )
        out_label = "outv"
    else:
        out_label = last_stage

    # Audio handling. Default: story audio passthrough (re-encoded as
    # aac — the cmd tail already does that). When the user dialled a
    # non-zero bg volume AND the bg file has audio, append an amix into
    # the filter chain. Otherwise the story audio plays alone (bg is
    # video-only here regardless of file content).
    audio_map = "0:a?"
    if has_bg and bg_video_volume > 0.0:
        fc.append(
            f"[0:a]volume=1.0[a0];"
            f"[{bg_input_idx}:a]volume={bg_video_volume:.3f}[abg];"
            f"[a0][abg]amix=inputs=2:duration=first:dropout_transition=0[aout]"
        )
        audio_map = "[aout]"
    filter_complex = ";".join(fc)
    cmd += [
        "-filter_complex", filter_complex,
        "-map", f"[{out_label}]",
        "-map", audio_map,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        # Same lipsync-safe flags V1 uses.
        "-r", "30", "-fps_mode", "cfr", "-async", "1",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-shortest", "-movflags", "+faststart",
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").splitlines()[-20:])
        raise RuntimeError(
            f"_compose_v4_bulletin_story failed (rc={proc.returncode}): {tail}"
        )
    return out_path


# ── Bulletin rendering ────────────────────────────────────────────────

def render_bulletin(inputs: BulletinRenderInputs) -> str:
    """Produce the final 1920x1080 bulletin.mp4 using V1's per-story
    composer plus V1's stitcher. Returns the output path on success;
    raises on failure (caller logs + reports)."""
    from pipeline_core.longform_compose import (
        StoryMeta, render_ticker, render_channel_bug,
    )
    from pipeline_core.bulletin_stitcher import stitch_bulletin

    bdir = Path(inputs.work_dir) / "_bulletin"
    bdir.mkdir(parents=True, exist_ok=True)
    lang_cfg = _lang_cfg(inputs.language)

    # Studio background — resolve once and reuse across every story
    # composer call so all clips share the same bg + intro.
    _bg_abs = _resolve_layout_bg_video(getattr(inputs, "layout", None))
    _bg_vol = float(getattr(inputs.layout, "bg_video_volume", 0.0) or 0.0) if _bg_abs else 0.0
    _intro_sec = float(getattr(inputs.layout, "bg_intro_seconds", 0.0) or 0.0) if _bg_abs else 0.0

    # 1) Ticker — one wide PNG built from every story's headline, used
    #    across all stories so the scroll is coherent.
    headlines = [
        (s.title_native or s.title_english or "").strip()
        for s in inputs.stories
    ]
    headlines = [h for h in headlines if h] or ([inputs.channel_name] if inputs.channel_name else ["NEWS"])
    ticker_path = str(bdir / "ticker.png")
    render_ticker(headlines, lang_cfg.code, lang_cfg.font_primary, ticker_path)

    # 2) Channel bug — DISABLED at render time.
    # The per-channel logo + watermark is now applied at UPLOAD time
    # (and at download time) by pipeline_v4.watermark.stamp_for_channel,
    # so the same clean render can ship to multiple channels each with
    # its own brand. Baking a bug in here would either:
    #   - paint the wrong brand on a multi-destination render, or
    #   - leak an empty translucent plate (visible as a small dark dot
    #     in the corner) when no logo / channel name is configured.
    # Setting bug_path to "" makes _compose_v4_bulletin_story skip the
    # overlay entirely — its has_bug check goes False.
    bug_path = ""

    # 3) Watermark plate — also disabled at render time. Same per-
    # channel stamping logic in pipeline_v4.watermark handles this at
    # upload/download. Keeping render output clean means a single file
    # can ship to multiple channels each with its own brand.
    watermark_path = ""
    if False and ((inputs.watermark_text and inputs.watermark_text.strip()) or inputs.brand_logo):
        watermark_path = str(bdir / "watermark.png")
        _render_watermark_png(
            text=inputs.watermark_text or "",
            logo_path=inputs.brand_logo,
            canvas_w=1920, canvas_h=1080,
            opacity=max(0.05, min(1.0, inputs.watermark_opacity)),
            out_path=watermark_path,
            font_path=lang_cfg.font_primary,
        )

    # 3) Per-story compose: slice the bulletin into the story's range
    #    and call V1's composer. Each story's render-affecting inputs
    #    are hashed and compared against a sidecar file from the
    #    previous render — when nothing meaningful changed we reuse the
    #    cached composed_story_NN.mp4 instead of paying for another
    #    ffmpeg pass. Typical wins: image swaps (one story dirty),
    #    trim-point tweaks (one story dirty), per-slot edits.
    composed_paths: list[str] = []
    pool = inputs.sidebar_images or []
    cache_hits = 0
    cache_misses = 0
    for i, s in enumerate(inputs.stories):
        sidebar_img = pool[i] if i < len(pool) else None
        sidebar_path = _resolve_sidebar(
            work_dir=bdir, story_index=i, pool_image_path=sidebar_img,
        )
        story_meta = StoryMeta(
            title=(s.title_native or s.title_english or "").strip() or "KAIZER NEWS",
            kicker="BREAKING",
            language=lang_cfg.code,
            story_index=i,
            total_stories=len(inputs.stories),
        )
        composed = str(bdir / f"composed_story_{i:02d}.mp4")
        hash_file = str(bdir / f"composed_story_{i:02d}.hash")

        # Stash story_index/total_stories on the story object so the
        # hash sees them — most TrimmedStory shapes already carry these
        # but SimpleNamespace stand-ins from the editor route may not.
        try:
            if getattr(s, "story_index", None) is None:
                setattr(s, "story_index", i)
            if getattr(s, "total_stories", None) is None:
                setattr(s, "total_stories", len(inputs.stories))
        except Exception:
            pass

        new_hash = _per_story_cache_hash(
            story=s,
            ticker_path=ticker_path,
            sidebar_path=sidebar_path,
            layout=getattr(inputs, "layout", None),
            channel_bug_path=bug_path,
            watermark_path=watermark_path,
            watermark_position=inputs.watermark_position,
            font_path=lang_cfg.font_primary,
            bg_video_abs=_bg_abs,
            bg_video_volume=_bg_vol,
            language_code=lang_cfg.code,
        )
        old_hash = _read_cached_hash(hash_file)
        if old_hash == new_hash and os.path.isfile(composed):
            print(f"[v4/v1_bridge] story {i+1}/{len(inputs.stories)} cache hit "
                  f"({new_hash}) — skipping recompose", flush=True)
            composed_paths.append(composed)
            cache_hits += 1
            continue
        cache_misses += 1
        raw_slice = str(bdir / f"raw_story_{i:02d}.mp4")
        _slice_video(
            source_path=inputs.trimmed_bulletin_path,
            start_sec=s.video_t_start,
            end_sec=s.video_t_end,
            output_path=raw_slice,
        )
        _compose_v4_bulletin_story(
            story_clip_path=raw_slice,
            story_meta=story_meta,
            out_path=composed,
            sidebar_path=sidebar_path,
            ticker_path=ticker_path,
            channel_bug_path=bug_path,
            font_path=lang_cfg.font_primary,
            sidebar_is_video=False,
            work_dir=str(bdir),
            layout=getattr(inputs, "layout", None),
            watermark_path=watermark_path,
            watermark_position=inputs.watermark_position,
            bg_video_path=_bg_abs,
            bg_video_volume=_bg_vol,
            # Ticker is overlaid AFTER stitch instead of per-story so
            # title-only edits keep the per-story cache warm.
            apply_ticker=False,
        )
        _write_cached_hash(hash_file, new_hash)
        composed_paths.append(composed)
    print(f"[v4/v1_bridge] per-story cache: {cache_hits} hit / {cache_misses} miss",
          flush=True)

    if not composed_paths:
        raise RuntimeError("render_bulletin: no stories composed")

    # 4) Stitch (no ticker yet) — V1's bulletin_stitcher handles codec
    #    mismatches and uses -c copy for speed.
    no_ticker_stitched = str(bdir / "_stitched_no_ticker.mp4")
    stitch_bulletin(
        composed_paths,
        no_ticker_stitched,
        work_dir=str(bdir),
    )

    # 5) Overlay the scrolling ticker once across the full bulletin.
    #    Doing this here (instead of inside every per-story compose)
    #    means a title edit only invalidates THIS pass; per-story
    #    composes stay cached. Cost: one re-encoding pass over the
    #    stitched file; win: skips N per-story re-encodes.
    canvas_w_l = getattr(getattr(inputs, "layout", None), "width", V4_W) or V4_W
    canvas_h_l = getattr(getattr(inputs, "layout", None), "height", V4_H) or V4_H
    ticker_y_l = canvas_h_l - V4_TICKER_H
    # When an intro reel is configured, the ticker-overlaid file is the
    # "inner" file we'll prepend to; otherwise it's the final output.
    stitched_path = inputs.output_path
    if _intro_sec > 0 and _bg_abs:
        stitched_path = str(Path(inputs.output_path).with_name(
            "_inner_" + Path(inputs.output_path).name))
    _overlay_ticker_post_stitch(
        bulletin_path=no_ticker_stitched,
        ticker_png_path=ticker_path,
        out_path=stitched_path,
        canvas_w=canvas_w_l,
        canvas_h=canvas_h_l,
        ticker_y=ticker_y_l,
    )
    try: Path(no_ticker_stitched).unlink(missing_ok=True)
    except OSError: pass

    if _intro_sec > 0 and _bg_abs:
        intro_path = str(Path(inputs.output_path).with_name("_intro.mp4"))
        try:
            _render_bg_intro_clip(
                bg_video_path=_bg_abs,
                duration_s=_intro_sec,
                width=getattr(inputs.layout, "width", 1920) or 1920,
                height=getattr(inputs.layout, "height", 1080) or 1080,
                out_path=intro_path,
            )
            _concat_clips(
                intro_path, stitched_path, inputs.output_path,
                intro_duration_s=_intro_sec,
                crossfade_s=0.8,
            )
        except Exception as exc:
            print(f"[v4/v1_bridge] intro stage failed ({exc}); falling back to main render", flush=True)
            try:
                Path(stitched_path).replace(inputs.output_path)
            except OSError:
                pass
        else:
            for p in (intro_path, stitched_path):
                try: Path(p).unlink(missing_ok=True)
                except OSError: pass

    return inputs.output_path


# ── Shorts rendering ──────────────────────────────────────────────────

def render_short(inputs: ShortRenderInputs) -> str:
    """Compose one V1-style short (torn_card / clean_card / split_frame
    / follow_bar). Returns the output path on success; raises on failure."""
    layout = (inputs.layout or DEFAULT_SHORTS_LAYOUT).lower()
    if layout not in SUPPORTED_SHORTS_LAYOUTS:
        layout = DEFAULT_SHORTS_LAYOUT
    lang_cfg = _lang_cfg(inputs.language)
    preset = dict(SHORTS_PRESET)
    # Caller-supplied font_file wins; otherwise pick the right script's
    # font from the language config.
    font_basename = (
        inputs.font_file
        or os.path.basename(lang_cfg.font_primary)
        or "NotoSansTelugu-Bold.ttf"
    )

    if layout == "torn_card":
        from pipeline_core.pipeline import compose_clip
        compose_clip(
            inputs.trimmed_short_path,
            inputs.image_path,
            inputs.title_text or "KAIZER NEWS",
            inputs.output_path,
            preset,
            font_size=inputs.font_size,
            text_color=inputs.text_color,
            font_file=font_basename,
            section_pct=inputs.section_pct,
            card_style=inputs.card_style,
            platform="youtube_short",
        )
    elif layout == "clean_card":
        from pipeline_core.pipeline import compose_clip_clean_card
        compose_clip_clean_card(
            inputs.trimmed_short_path,
            inputs.image_path,
            inputs.title_text or "KAIZER NEWS",
            inputs.output_path,
            preset,
            font_size=inputs.font_size,
            text_color=inputs.text_color,
            font_file=font_basename,
            platform="youtube_short",
        )
    elif layout == "split_frame":
        from pipeline_core.pipeline import compose_split_frame
        thumb = inputs.thumbnail_path or inputs.image_path
        if not thumb:
            raise RuntimeError("split_frame requires a thumbnail/image")
        compose_split_frame(
            inputs.trimmed_short_path,
            thumb,
            inputs.output_path,
            preset,
            video_logo=inputs.brand_logo,
            platform="youtube_short",
        )
    elif layout == "follow_bar":
        from pipeline_core.pipeline import compose_follow_bar
        fp = inputs.follow_params or {}
        compose_follow_bar(
            inputs.trimmed_short_path,
            inputs.output_path,
            preset,
            title_text=inputs.title_text or "",
            font_file=font_basename,
            text_color=inputs.text_color or fp.get("text_color", "#ffff00"),
            text_size=inputs.font_size or 60,
            bg_color=fp.get("bg_color", "#1a0a2e"),
            follow_text=fp.get("follow_text", "FOLLOW KAIZER NEWS TELUGU"),
            follow_text_color=fp.get("follow_text_color", "#ffffff"),
            video_logo=inputs.brand_logo,
            platform="youtube_short",
        )

    # Watermark post-pass — V1's short composers don't expose an
    # overlay slot, so we re-encode once to drop the user's translucent
    # text+logo bug onto every frame. Skipped when neither watermark
    # text nor a brand logo is configured.
    wm_text = getattr(inputs, "watermark_text", "") or ""
    wm_op   = float(getattr(inputs, "watermark_opacity", 0.35) or 0.35)
    wm_pos  = getattr(inputs, "watermark_position", "top-right") or "top-right"
    if (wm_text.strip() or inputs.brand_logo) and os.path.isfile(inputs.output_path):
        wm_path = str(Path(inputs.work_dir) / "_wm_short.png")
        try:
            _render_watermark_png(
                text=wm_text, logo_path=inputs.brand_logo,
                canvas_w=preset["width"], canvas_h=preset["height"],
                opacity=max(0.05, min(1.0, wm_op)),
                out_path=wm_path,
                font_path=lang_cfg.font_primary,
            )
            wx, wy = _watermark_overlay_xy(wm_pos, preset["width"], preset["height"])
            stamped = inputs.output_path + ".wm.mp4"
            cmd = [
                _ffmpeg_bin(), "-y", "-v", "error",
                "-i", inputs.output_path,
                "-loop", "1", "-i", wm_path,
                "-filter_complex",
                f"[0:v][1:v]overlay=x={wx}:y={wy}:format=auto[outv]",
                "-map", "[outv]", "-map", "0:a?",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-c:a", "copy",
                "-shortest", "-movflags", "+faststart",
                stamped,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if r.returncode == 0 and os.path.isfile(stamped):
                os.replace(stamped, inputs.output_path)
        except Exception as exc:
            print(f"[v4/wm] short watermark post-pass failed: {exc}", flush=True)

    return inputs.output_path
