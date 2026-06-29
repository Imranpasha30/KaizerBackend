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

from pipeline_v4.encoder import video_encoder_args as _enc_args
from pipeline_v4.encoder import video_decoder_args as _dec_args
from pipeline_v4.ffmpeg_exec import run_ffmpeg as _run_ffmpeg


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
                          language_code: str,
                          images=None,
                          pool_dir: Optional[Path] = None) -> str:
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
    def _ia(obj, name, default=None):
        return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)
    def _carousel_blob() -> list:
        out = []
        for img in (images or []):
            src = _ia(img, "src", "") or ""
            try:
                ts = round(float(_ia(img, "t_start", 0.0) or 0.0), 3)
                te = round(float(_ia(img, "t_end", 0.0) or 0.0), 3)
                ed = round(float(_ia(img, "effect_duration", 0.4) or 0.0), 3)
                ox = round(float(_ia(img, "offset_x_pct", 50.0) or 50.0), 1)
                oy = round(float(_ia(img, "offset_y_pct", 50.0) or 50.0), 1)
            except (TypeError, ValueError):
                ts = te = ed = 0.0; ox = oy = 50.0
            # Content fingerprint via the resolved pool file (so a same-named
            # overwrite still busts); falls back to {"p": None} if unresolved
            # — the raw src + timing below still change the hash on edits.
            fp = _fp(_resolve_pool_image(src, pool_dir)) if pool_dir else {"p": None}
            out.append({
                "src": src, "ts": ts, "te": te, "ed": ed,
                "effect": _ia(img, "effect", "") or "",
                "fit": _ia(img, "fit", "") or "", "ox": ox, "oy": oy, "fp": fp,
            })
        return out
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
        # The image carousel — hashed DIRECTLY (not just via the sidebar
        # file's mtime) so replacing/reframing/retiming ANY image in the
        # story deterministically invalidates this story's composed clip.
        # Each entry carries the src + content fingerprint + timing + effect
        # + fit + focal offset; ``_carousel_segments`` resolves src -> file
        # for the fingerprint. Excludes the absolute "path" (machine-specific).
        "carousel": _carousel_blob(),
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
            *_enc_args(crf=20, preset_hint="medium"),
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
            *_enc_args(crf=20, preset_hint="medium"),
            "-pix_fmt", "yuv420p",
            "-r", "30", "-fps_mode", "cfr",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart", out_path,
        ]
    _run_ffmpeg(cmd, timeout=60 * 5, log_label="bulletin_intro_render")


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
        *_enc_args(crf=20, preset_hint="medium"),
        "-pix_fmt", "yuv420p",
        "-r", "30", "-fps_mode", "cfr",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        out_path,
    ]
    _run_ffmpeg(cmd, timeout=60 * 20, log_label="bulletin_intro_concat")


def _overlay_ticker_and_concat_intro(
    *,
    intro_path: str,
    main_noticker_path: str,
    ticker_png_path: str,
    out_path: str,
    ticker_y: int,
    intro_duration_s: float,
    crossfade_s: float = 0.8,
    ticker_speed_px_s: float = 200.0,
) -> None:
    """Combined ticker-overlay + intro-concat in ONE ffmpeg pass (Wave 4
    item G). Replaces the legacy sequence of TWO full re-encodes of the
    bulletin (ticker overlay pass, then intro xfade pass) when both are
    needed.

    Filtergraph:
      - [1:v] (stitched main, no ticker) gets the scrolling ticker PNG
        overlaid exactly as ``_overlay_ticker_post_stitch`` did — the
        overlay's ``t`` is the MAIN input's own timeline (starts at 0),
        so the scroll position matches the legacy two-pass output, and
        the ticker never appears over the intro.
      - [0:v] (intro) xfades into the tickered main with the same
        duration/offset math as ``_concat_clips``.

    Lip-sync: the audio path is the EXACT acrossfade the legacy
    ``_concat_clips`` pass already used — no new audio operation is
    introduced; one whole video re-encode generation is removed. When
    the intro is too short for the crossfade, falls back to the same
    hard-concat the legacy path used.
    """
    ffmpeg = _ffmpeg_bin()
    ticker_chain = (
        f"[1:v][2:v]overlay="
        f"x='W-mod(t*{ticker_speed_px_s:.1f}\\,w+W)':y={ticker_y}:"
        f"format=auto:shortest=1,format=yuv420p[mt]"
    )
    use_xfade = crossfade_s > 0.05 and intro_duration_s > crossfade_s + 0.1
    if use_xfade:
        offset = max(0.0, intro_duration_s - crossfade_s)
        filter_complex = (
            ticker_chain + ";"
            f"[0:v][mt]xfade=transition=fade:"
            f"duration={crossfade_s:.3f}:offset={offset:.3f}[v];"
            f"[0:a][1:a]acrossfade=d={crossfade_s:.3f}:c1=tri:c2=tri[a]"
        )
    else:
        filter_complex = (
            ticker_chain + ";"
            f"[0:v:0][0:a:0][mt][1:a:0]concat=n=2:v=1:a=1[v][a]"
        )
    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-i", intro_path,
        "-i", main_noticker_path,
        "-loop", "1", "-i", ticker_png_path,
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "[a]",
        *_enc_args(crf=20, preset_hint="medium"),
        "-pix_fmt", "yuv420p",
        "-r", "30", "-fps_mode", "cfr",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        out_path,
    ]
    _run_ffmpeg(cmd, timeout=60 * 20, log_label="ticker_intro_combined")


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
        *_enc_args(crf=20, preset_hint="medium"),
        "-pix_fmt", "yuv420p",
        "-r", "30", "-fps_mode", "cfr",
        "-c:a", "copy",
        "-movflags", "+faststart",
        out_path,
    ]
    _run_ffmpeg(cmd, timeout=60 * 20, log_label="ticker_overlay")


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
    # Ticker overrides pulled from the canvas's ticker text-block (the
    # editor's live ticker controls). None -> broadcast defaults so the
    # initial render and untouched jobs stay byte-identical.
    ticker_speed_s: Optional[float] = None       # seconds per full scroll loop (lower = faster)
    ticker_bg_color: Optional[str] = None        # hex bar color, e.g. "#FFD400"


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
    # Custom-template per-slot media (custom:<id> only): {slot_key: local path} for the
    # NON-main slots the user filled; main_media_slot names the slot that gets the
    # AI-trimmed clip (trimmed_short_path). Other slots are used as-is.
    template_media: Optional[dict] = None
    main_media_slot: str = ""
    # Crash-guard: the output form this render expects ("short" for 9:16, "full" for
    # 16:9). When a custom template is used, _render_custom_short verifies the template's
    # own aspect matches and fails clean otherwise (a landscape template can't render a
    # short). "" = skip the check (non-custom layouts).
    expected_kind: str = ""
    # Real story content fed to a custom template's slots (the offline filler maps these
    # onto headline/hook/subtitle/ticker/etc. so no author placeholder leaks). All optional;
    # when headline is None the filler falls back to title_text, keeping built-in layouts
    # (which only read title_text) byte-identical.
    headline: Optional[str] = None
    headline_alt: Optional[str] = None
    subtitle: Optional[str] = None
    body: Optional[str] = None
    ticker: Optional[str] = None
    kicker: Optional[str] = None
    cta: Optional[str] = None
    # PER-JOB HTML OVERRIDE (custom:<id> only): the operator visually edited the template for
    # THIS job in the inline builder (moved/resized/recolored/retyped) and saved the result.
    # When set, _render_custom_short renders this HTML VERBATIM (text/images already baked in)
    # and only composites the video clip into the video slot — the PARENT template is never
    # touched. Empty/None -> normal path (parent template + offline filler).
    template_html_override: Optional[str] = None
    support_images: Optional[list] = None      # ordered ABSOLUTE supporting-image paths
    stories: Optional[list] = None             # full-form: [{headline, headline_alt, body, ...}]
    # Per-slot TEXT overrides from the custom-template editor: {slot_key: text}. These
    # REPLACE the filler/AI text for that slot (operator's manual edit). Empty -> filler.
    template_text_overrides: Optional[dict] = None


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
        # GPU decode when NVENC is active (input options precede -i).
        *_dec_args(),
        "-ss", f"{float(start_sec):.3f}",
        "-i", source_path,
        "-t",  f"{dur:.3f}",
        *_enc_args(crf=20, preset_hint="veryfast"),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        output_path,
    ]
    # retry=2 so the runner can cascade cuda -> CPU-decode -> full-CPU before
    # giving up (GPU input-seek can decode 0 frames on some sources/positions).
    _run_ffmpeg(cmd, timeout=600, log_label="slice_video", retry=2)
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


# ─── Sidebar image carousel (the per-story picture timeline) ──────────
#
# The editor lets the user put MULTIPLE images on a story's sidebar, each
# with its own start/duration and a fade transition. The old render
# showed just ONE static image per story (sidebar_images[i]). These
# helpers turn a story's ``images[]`` into a real timed carousel video
# that the per-story composer drops in via its ``sidebar_is_video`` path
# — so replace / reorder / per-image timing / fade all actually appear.


def _resolve_pool_image(src: str, pool_dir: Path) -> Optional[str]:
    """Resolve a canvas image ``src`` (a bare pool filename or an absolute
    path) to a real file on disk, or None."""
    src = (src or "").strip()
    if not src:
        return None
    p = pool_dir / src
    if p.is_file():
        return str(p)
    if os.path.isabs(src) and os.path.isfile(src):
        return src
    return None


def _carousel_segments(images, pool_dir: Path) -> list[dict]:
    """Resolve a story's canvas images into renderable carousel segments
    (story-local time windows). Drops missing files and zero-length
    windows. Accepts dicts OR pydantic image objects."""
    def _attr(obj, name, default=None):
        return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)

    segs: list[dict] = []
    for img in (images or []):
        path = _resolve_pool_image(_attr(img, "src", "") or "", pool_dir)
        if not path:
            continue
        try:
            ts = max(0.0, float(_attr(img, "t_start", 0.0) or 0.0))
            te = float(_attr(img, "t_end", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if te <= ts + 0.05:
            continue
        win = te - ts
        try:
            ed = float(_attr(img, "effect_duration", 0.4) or 0.0)
        except (TypeError, ValueError):
            ed = 0.4
        ed = max(0.0, min(ed, win / 2.0))
        effect = _attr(img, "effect", "fade") or "fade"
        try:
            stt = os.stat(path)
            fp = f"{stt.st_size}:{int(stt.st_mtime)}"
        except OSError:
            fp = "0:0"
        try:
            fit = _attr(img, "fit", "cover") or "cover"
            ox = round(float(_attr(img, "offset_x_pct", 50.0) or 50.0), 1)
            oy = round(float(_attr(img, "offset_y_pct", 50.0) or 50.0), 1)
        except (TypeError, ValueError):
            fit, ox, oy = "cover", 50.0, 50.0
        segs.append({
            "path": path, "ts": ts, "te": te, "ed": ed,
            "fade": (effect == "fade"), "fp": fp,
            "src": _attr(img, "src", "") or "",
            # Full visual spec so the cache key reflects EVERY per-image edit
            # (replace, framing, fit, effect type) — not just the few fields
            # the renderer happened to use. Missing these let an edited image
            # reuse a stale composed clip → the old image reappeared.
            "effect": effect, "fit": fit, "ox": ox, "oy": oy,
        })
    return segs


def _render_sidebar_carousel(
    *, segs: list[dict], out_path: str, story_duration: float,
    sidebar_w: int = V4_SIDE_INNER_W, sidebar_h: int = V4_TILE_INNER_H,
    bg_color: str = "black",
) -> Optional[str]:
    """Render the story's image carousel as a sidebar-sized silent mp4 —
    each image cover-fitted into the panel and shown in its
    ``[t_start, t_end]`` window with a fade in/out. Returns ``out_path`` on
    success, or None to fall back to a static sidebar."""
    if not segs:
        return None
    dur = max(float(story_duration or 0.0), max(s["te"] for s in segs)) + 0.05
    ffmpeg = _ffmpeg_bin()
    inputs = ["-f", "lavfi", "-t", f"{dur:.3f}",
              "-i", f"color=c={bg_color or 'black'}:s={sidebar_w}x{sidebar_h}:r=25"]
    for s in segs:
        inputs += ["-loop", "1", "-i", s["path"]]
    chain: list[str] = []
    last = "[0:v]"
    for idx, s in enumerate(segs, start=1):
        scale = (
            f"[{idx}:v]scale={sidebar_w}:{sidebar_h}:force_original_aspect_ratio=increase,"
            f"crop={sidebar_w}:{sidebar_h},setsar=1"
        )
        if s["fade"] and s["ed"] > 0.0:
            scale += (
                f",format=yuva420p,"
                f"fade=t=in:st={s['ts']:.3f}:d={s['ed']:.3f}:alpha=1,"
                f"fade=t=out:st={s['te'] - s['ed']:.3f}:d={s['ed']:.3f}:alpha=1"
            )
        chain.append(f"{scale}[im{idx}]")
        nxt = f"[v{idx}]"
        chain.append(
            f"{last}[im{idx}]overlay=enable='between(t,{s['ts']:.3f},{s['te']:.3f})'{nxt}"
        )
        last = nxt
    cmd = [
        ffmpeg, "-y", "-v", "error", *inputs,
        "-filter_complex", ";".join(chain),
        "-map", last, "-t", f"{dur:.3f}",
        *_enc_args(crf=20, preset_hint="medium"),
        "-pix_fmt", "yuv420p", "-an", out_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            print(f"[v4/v1_bridge] sidebar carousel ffmpeg failed: "
                  f"{(r.stderr or '')[-400:]}", flush=True)
            return None
    except Exception as exc:
        print(f"[v4/v1_bridge] sidebar carousel error: {exc}", flush=True)
        return None
    return out_path if (os.path.isfile(out_path) and os.path.getsize(out_path) > 0) else None


def _ensure_sidebar_carousel(
    *, images, pool_dir: Path, out_path: str, story_duration: float,
    bg_color: str = "black",
) -> Optional[str]:
    """Render the carousel only when its content changed (guarded by a
    ``.sig`` sidecar) so the per-story render cache still works. Returns
    the carousel mp4 path, or None when there are ≤1 images (caller uses
    the cheaper static sidebar)."""
    segs = _carousel_segments(images, pool_dir)
    if len(segs) <= 1:          # one image → static sidebar is enough
        return None
    sig = json.dumps(
        {"segs": [{"s": s["src"], "ts": round(s["ts"], 3), "te": round(s["te"], 3),
                   "ed": round(s["ed"], 3), "fd": s["fade"], "fp": s["fp"],
                   "ef": s.get("effect"), "fit": s.get("fit"),
                   "ox": s.get("ox"), "oy": s.get("oy")} for s in segs],
         "dur": round(float(story_duration or 0.0), 3), "bg": bg_color or "black"},
        sort_keys=True,
    )
    sig_file = out_path + ".sig"
    if os.path.isfile(out_path) and os.path.isfile(sig_file):
        try:
            if Path(sig_file).read_text(encoding="utf-8") == sig:
                return out_path          # unchanged — reuse
        except OSError:
            pass
    res = _render_sidebar_carousel(
        segs=segs, out_path=out_path, story_duration=story_duration, bg_color=bg_color,
    )
    if res:
        try:
            Path(sig_file).write_text(sig, encoding="utf-8")
        except OSError:
            pass
    return res


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
    back to literal "KAIZER X" text when no channel_name is given,
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
    # GPU decode of the story clip when NVENC is active. Input option —
    # placed before the FIRST -i so it only applies to the video input
    # (the looped PNG inputs that follow keep their own decoders).
    cmd: list[str] = [ffmpeg, "-y", "-v", "error",
                      *_dec_args(), "-i", story_clip_path]
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
        *_enc_args(crf=20, preset_hint="medium"),
        "-pix_fmt", "yuv420p",
        # Same lipsync-safe flags V1 uses.
        "-r", "30", "-fps_mode", "cfr", "-async", "1",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-shortest", "-movflags", "+faststart",
        out_path,
    ]
    _run_ffmpeg(
        cmd, timeout=900,
        log_label=f"compose_story_{story_meta.story_index:02d}",
    )
    return out_path


# ── Bulletin rendering ────────────────────────────────────────────────

def extract_ticker_overrides(stories) -> tuple[Optional[float], Optional[str]]:
    """Pull the ticker scroll-speed (seconds/loop) + bar color from the
    first ticker text-block that sets them. The editor applies one global
    pair across every ticker block, so the first set value wins. Returns
    (None, None) when the canvas has no ticker overrides (default render)."""
    speed_s = None
    color = None
    for s in (stories or []):
        for b in (getattr(s, "text_blocks", None) or []):
            if getattr(b, "kind", None) != "ticker":
                continue
            if speed_s is None and getattr(b, "ticker_speed", None):
                try:
                    speed_s = float(b.ticker_speed)
                except (TypeError, ValueError):
                    pass
            if color is None and getattr(b, "ticker_color", None):
                color = str(b.ticker_color)
        if speed_s is not None and color is not None:
            break
    return speed_s, color


def render_bulletin(inputs: BulletinRenderInputs) -> str:
    """Produce the final 1920x1080 bulletin.mp4 using V1's per-story
    composer plus V1's stitcher. Returns the output path on success;
    raises on failure (caller logs + reports)."""
    from pipeline_core.longform_compose import (
        StoryMeta, render_ticker, render_channel_bug, estimate_ticker_width,
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
    render_ticker(headlines, lang_cfg.code, lang_cfg.font_primary, ticker_path,
                  bg_color=getattr(inputs, "ticker_bg_color", None))

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
    pool = inputs.sidebar_images or []
    n_stories = len(inputs.stories)

    def _compose_one_story(i: int, s) -> tuple[int, str, bool]:
        """Compose story ``i`` (or reuse its cache hit). Returns
        ``(index, composed_path, cache_hit)``. Every artefact this task
        touches is keyed by the story index (sidebar PNG, lower-third
        PNG, raw slice, composed mp4, hash sidecar) so tasks are safe
        to run concurrently."""
        # Sidebar: prefer the story's OWN image carousel (timed, faded)
        # so every per-image edit shows. Falls back to a single static
        # image when the story has ≤1 image — preferring the story's own
        # first image over the legacy alphabetical pool[i] so a replace of
        # the first slot still takes effect.
        story_images = getattr(s, "images", None)
        pool_dir = (
            Path(pool[0]).parent if pool
            else Path(inputs.work_dir) / "_pool"
        )
        story_dur = (float(getattr(s, "video_t_end", 0.0) or 0.0)
                     - float(getattr(s, "video_t_start", 0.0) or 0.0))
        bg_col = getattr(getattr(inputs, "layout", None), "bg_color", None) or "black"
        carousel = _ensure_sidebar_carousel(
            images=story_images, pool_dir=pool_dir,
            out_path=str(bdir / f"_sidebar_carousel_{i:02d}.mp4"),
            story_duration=story_dur, bg_color=bg_col,
        )
        if carousel:
            sidebar_path = carousel
            sidebar_is_video = True
        else:
            first_src = None
            if story_images:
                fa = story_images[0]
                first_src = (fa.get("src") if isinstance(fa, dict)
                             else getattr(fa, "src", None))
            sidebar_img = (
                _resolve_pool_image(first_src or "", pool_dir)
                or (pool[i] if i < len(pool) else None)
            )
            sidebar_path = _resolve_sidebar(
                work_dir=bdir, story_index=i, pool_image_path=sidebar_img,
            )
            sidebar_is_video = False
        story_meta = StoryMeta(
            title=(s.title_native or s.title_english or "").strip() or "KAIZER X",
            kicker="BREAKING",
            language=lang_cfg.code,
            story_index=i,
            total_stories=n_stories,
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
                setattr(s, "total_stories", n_stories)
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
            images=story_images,
            pool_dir=pool_dir,
        )
        old_hash = _read_cached_hash(hash_file)
        if old_hash == new_hash and os.path.isfile(composed):
            print(f"[v4/v1_bridge] story {i+1}/{n_stories} cache hit "
                  f"({new_hash}) — skipping recompose", flush=True)
            return (i, composed, True)
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
            sidebar_is_video=sidebar_is_video,
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
        return (i, composed, False)

    # Bounded parallel per-story compose (Wave 4 item E). Cache hits
    # short-circuit inside each task exactly as before; the stitch
    # below consumes results strictly in story order. A single story
    # failure still fails the whole bulletin (same contract as the old
    # sequential loop — the orchestrator catches and reports it).
    try:
        _story_workers = max(1, int(os.environ.get(
            "KAIZER_V4_RENDER_CONCURRENCY", "3") or "3"))
    except ValueError:
        _story_workers = 3
    results: list[tuple[int, str, bool]] = []
    if n_stories > 1 and _story_workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        first_exc: Optional[BaseException] = None
        with ThreadPoolExecutor(
            max_workers=min(_story_workers, n_stories),
            thread_name_prefix="v4-story",
        ) as story_pool:
            futures = [
                story_pool.submit(_compose_one_story, i, s)
                for i, s in enumerate(inputs.stories)
            ]
            for fut in futures:   # submit order == story order
                try:
                    results.append(fut.result())
                except BaseException as exc:  # noqa: BLE001 — re-raised below
                    if first_exc is None:
                        first_exc = exc
        if first_exc is not None:
            raise first_exc
    else:
        for i, s in enumerate(inputs.stories):
            results.append(_compose_one_story(i, s))

    results.sort(key=lambda r: r[0])
    composed_paths: list[str] = [r[1] for r in results]
    cache_hits = sum(1 for r in results if r[2])
    cache_misses = len(results) - cache_hits
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

    # 5) Ticker + (optional) intro. The scrolling ticker is overlaid
    #    post-stitch (instead of inside every per-story compose) so a
    #    title edit only invalidates this pass; per-story composes stay
    #    cached.
    #
    #    Wave 4 item G: when BOTH the ticker overlay AND the intro reel
    #    are needed, they now run as ONE ffmpeg pass
    #    (_overlay_ticker_and_concat_intro) instead of two full
    #    re-encodes of the bulletin. The audio path through that pass is
    #    the identical acrossfade the legacy intro-concat already used —
    #    no new audio handling. If the combined pass fails for any
    #    reason we fall back to the legacy two-pass sequence, and from
    #    there to a ticker-only output, so robustness is strictly >= the
    #    old code.
    canvas_w_l = getattr(getattr(inputs, "layout", None), "width", V4_W) or V4_W
    canvas_h_l = getattr(getattr(inputs, "layout", None), "height", V4_H) or V4_H
    ticker_y_l = canvas_h_l - V4_TICKER_H

    # Editor ticker SPEED: convert seconds-per-loop into the px/s the
    # overlay x-expression uses. One loop scrolls (ticker_width + canvas_w)
    # px, so px/s = that distance / seconds. None -> the 200 px/s default.
    ticker_speed_px_s = 200.0
    _tspeed_s = getattr(inputs, "ticker_speed_s", None)
    if _tspeed_s and _tspeed_s > 0:
        _tw = estimate_ticker_width(headlines)
        # Clamp to a readable band so a hand-edited sub-second canvas value can't produce a
        # thousands-of-px/s blur (the UI slider stays well inside this range).
        ticker_speed_px_s = max(20.0, min(1000.0, (_tw + (canvas_w_l or V4_W)) / float(_tspeed_s)))

    want_intro = _intro_sec > 0 and _bg_abs
    intro_path = str(Path(inputs.output_path).with_name("_intro.mp4"))

    if want_intro:
        try:
            _render_bg_intro_clip(
                bg_video_path=_bg_abs,
                duration_s=_intro_sec,
                width=canvas_w_l or 1920,
                height=canvas_h_l or 1080,
                out_path=intro_path,
            )
        except Exception as exc:
            print(f"[v4/v1_bridge] intro render failed ({exc}); "
                  f"continuing without intro", flush=True)
            want_intro = False

    combined_done = False
    if want_intro:
        try:
            _overlay_ticker_and_concat_intro(
                intro_path=intro_path,
                main_noticker_path=no_ticker_stitched,
                ticker_png_path=ticker_path,
                out_path=inputs.output_path,
                ticker_y=ticker_y_l,
                intro_duration_s=_intro_sec,
                crossfade_s=0.8,
                ticker_speed_px_s=ticker_speed_px_s,
            )
            combined_done = True
        except Exception as exc:
            print(f"[v4/v1_bridge] combined ticker+intro pass failed ({exc}); "
                  f"falling back to legacy two-pass", flush=True)

    if not combined_done:
        if want_intro:
            # Legacy two-pass: ticker overlay to an inner file, then
            # intro concat. Same behaviour as pre-Wave-4.
            inner_path = str(Path(inputs.output_path).with_name(
                "_inner_" + Path(inputs.output_path).name))
            _overlay_ticker_post_stitch(
                bulletin_path=no_ticker_stitched,
                ticker_png_path=ticker_path,
                out_path=inner_path,
                canvas_w=canvas_w_l,
                canvas_h=canvas_h_l,
                ticker_y=ticker_y_l,
                ticker_speed_px_s=ticker_speed_px_s,
            )
            try:
                _concat_clips(
                    intro_path, inner_path, inputs.output_path,
                    intro_duration_s=_intro_sec,
                    crossfade_s=0.8,
                )
            except Exception as exc:
                print(f"[v4/v1_bridge] intro stage failed ({exc}); "
                      f"falling back to main render", flush=True)
                try:
                    Path(inner_path).replace(inputs.output_path)
                except OSError:
                    pass
            else:
                try: Path(inner_path).unlink(missing_ok=True)
                except OSError: pass
        else:
            _overlay_ticker_post_stitch(
                bulletin_path=no_ticker_stitched,
                ticker_png_path=ticker_path,
                out_path=inputs.output_path,
                canvas_w=canvas_w_l,
                canvas_h=canvas_h_l,
                ticker_y=ticker_y_l,
                ticker_speed_px_s=ticker_speed_px_s,
            )

    for _p in (no_ticker_stitched, intro_path):
        try: Path(_p).unlink(missing_ok=True)
        except OSError: pass

    return inputs.output_path


# ── Shorts rendering ──────────────────────────────────────────────────

def _render_custom_short(inputs: "ShortRenderInputs", layout_key: str) -> dict:
    """Render a developer-uploaded custom template (layout ``custom:<id>``).

    Logo / watermark placement: if the template *marks a location* (a ``data-kaizer=
    "logo"`` / ``"watermark"`` slot) we inject there — ONCE — and tell the caller to
    skip the default corner stamp; if the template does NOT mark one, the caller's
    post-pass injects it at the default corner. Returns flags so render_short knows
    which case applied: ``{"logo_in_slot": bool, "watermark_in_slot": bool}``."""
    try:
        tid = int(layout_key.split(":", 1)[1])
    except Exception:
        raise RuntimeError(f"bad custom template key: {layout_key!r}")
    from database import SessionLocal
    import models
    from services import custom_templates as ct

    db = SessionLocal()
    try:
        t = db.get(models.CustomTemplate, tid)
        if not t or not t.dir_path or not os.path.isdir(t.dir_path):
            raise RuntimeError(f"custom template {tid} not available")

        # ── PER-JOB HTML OVERRIDE ──────────────────────────────────────────────────
        # The operator visually edited THIS job's template in the inline builder (moved /
        # resized / recoloured / retyped) and saved the result. Render that HTML VERBATIM
        # (text + images already baked into the DOM) and composite ONLY the video clip — the
        # parent template's own files are never touched. Logo/watermark slots are left empty
        # so the per-channel publish stamp still drops each channel's own brand at the marked
        # spot. Self-contained early return so the normal filler path stays byte-identical.
        _ov_html = (getattr(inputs, "template_html_override", "") or "").strip()
        if _ov_html:
            try:
                _norm_ov, contract = ct.normalize_and_discover(_ov_html)
            except Exception:
                contract = ct.discover(_ov_html); _norm_ov = _ov_html
            _html_to_render = _norm_ov or _ov_html

            _exp = (getattr(inputs, "expected_kind", "") or "").strip().lower()
            if _exp in ("short", "full"):
                _tk = ct.aspect_kind(contract.canvas_w, contract.canvas_h)
                if _tk != _exp:
                    raise RuntimeError(
                        f"edited template for this job is a {_tk}-form design ({contract.canvas_w}"
                        f"x{contract.canvas_h}) but this render needs {_exp}-form; refusing.")

            has_logo_slot = any(s.kind == "logo" for s in contract.slots)
            has_wm_slot = any(s.kind == "text" and s.key == "watermark" for s in contract.slots)
            wm_text = getattr(inputs, "watermark_text", "") or ""
            main_slot = (inputs.main_media_slot or "").strip()
            if not main_slot:
                vs = contract.video_slots
                main_slot = vs[0].key if vs else "video"

            # Write the override into the bundle ROOT (so its relative asset URLs resolve)
            # as a SEPARATE temp entry — never overwrite the parent template's entry file.
            import tempfile as _tf
            _fd, _ovpath = _tf.mkstemp(dir=t.dir_path, suffix=".joboverride.html")
            with os.fdopen(_fd, "w", encoding="utf-8") as _fh:
                _fh.write(_html_to_render)
            try:
                ov_bundle = ct.Bundle(root_dir=t.dir_path,
                                      entry_rel=os.path.basename(_ovpath), files=[])
                # texts stay EMPTY (the override HTML already carries the operator's words —
                # clear_unfilled=False keeps them VERBATIM). MEDIA slots, however, fill exactly
                # like the non-override path so images/video/background never go blank in override
                # mode: the offline filler's positional supporting images + any per-slot uploaded
                # media (template_media) + the image_path fallback; the main clip drives video.
                _exp_kind2 = "full" if _exp == "full" else "short"
                _supp = [p for p in (getattr(inputs, "support_images", None)
                                     or ([inputs.image_path] if inputs.image_path else [])) if p]
                try:
                    _content = ct.ContentBundle(
                        headline=(inputs.headline or inputs.title_text or ""),
                        images=_supp, stories=list(getattr(inputs, "stories", None) or []))
                    _imgs = dict(ct.build_slot_fill(contract, _content, kind=_exp_kind2).images)
                except Exception:
                    _imgs = {}
                    for _i, _s in enumerate(contract.image_slots):
                        if _i < len(_supp):
                            _imgs[_s.key] = _supp[_i]
                _videos = {main_slot: inputs.trimmed_short_path}
                _kindmap = {s.key: s.kind for s in contract.slots}
                for _slot, _path in (inputs.template_media or {}).items():
                    if not _path or _slot == main_slot:
                        continue
                    _k = _kindmap.get(_slot)
                    if _k in ("video", "background"):
                        _videos[_slot] = _path
                    elif _k == "image":
                        _imgs[_slot] = _path
                if inputs.image_path:
                    for _s in contract.image_slots:
                        _imgs.setdefault(_s.key, inputs.image_path)
                req = ct.RenderRequest(
                    videos=_videos,
                    texts={}, images=_imgs, logo_path=None, brand={}, fps=30,
                    main_slot=main_slot, intro_path=None, literal=True,
                    # scrolling ticker: text comes from the edited HTML's ticker slot (stashed
                    # during the still capture); speed/colour/font fall back to slot styling.
                    ticker_speed_s=getattr(inputs, "ticker_speed_s", None),
                    ticker_bg_color=getattr(inputs, "ticker_bg_color", None),
                    ticker_font_px=getattr(inputs, "ticker_font_px", None),
                    ticker_lang=(getattr(inputs, "language", None) or None),
                )
                work = str(Path(inputs.output_path).parent / f"_ctmpl_{tid}_job")
                _report = ct.render_template(ov_bundle, contract, req,
                                             work_dir=work, out_path=inputs.output_path)
            finally:
                try:
                    os.remove(_ovpath)
                except Exception:
                    pass

            try:
                _br = (_report or {}).get("brand_rects") or {}
                if _br.get("logo") or _br.get("watermark"):
                    import json as _json
                    with open(inputs.output_path + ".slots.json", "w", encoding="utf-8") as _fh:
                        _json.dump({"canvas": [contract.canvas_w, contract.canvas_h],
                                    "logo": _br.get("logo"), "watermark": _br.get("watermark")}, _fh)
            except Exception:
                pass
            try:
                t.use_count = (t.use_count or 0) + 1
                db.commit()
            except Exception:
                db.rollback()
            return {
                "logo_in_slot": bool(has_logo_slot and inputs.brand_logo),
                "watermark_in_slot": bool(has_wm_slot and wm_text.strip()),
            }
        # ── end per-job HTML override ──────────────────────────────────────────────

        bundle = ct.Bundle(root_dir=t.dir_path, entry_rel=t.entry_rel or "index.html", files=[])
        with open(bundle.entry_path, encoding="utf-8", errors="replace") as fh:
            _raw_html = fh.read()
        # Tolerant normalization: infer canvas + inject synthetic data-kaizer markers for
        # templates that used id/class/semantic-tags/other attrs (or no markers). Write the
        # normalized entry back (idempotent) so the renderer fills inferred slots too. This
        # also covers templates uploaded before inference existed. Never fail the render.
        try:
            _norm_html, contract = ct.normalize_and_discover(_raw_html)
            if _norm_html and _norm_html != _raw_html:
                try:
                    import tempfile as _tf
                    _d = os.path.dirname(bundle.entry_path) or "."
                    _fd, _tmp = _tf.mkstemp(dir=_d, suffix=".tmp")
                    with os.fdopen(_fd, "w", encoding="utf-8") as _fh:
                        _fh.write(_norm_html)
                    os.replace(_tmp, bundle.entry_path)   # atomic: no torn read by a concurrent render
                except Exception:
                    pass
        except Exception:
            contract = ct.discover(_raw_html)

        # CRASH-GUARD (render-time backstop): the template's output form is its canvas
        # aspect. If this render expects a short but the template is full-form (or vice-
        # versa), fail clean with a clear message instead of producing a wrong-aspect
        # master that breaks the pipeline downstream. job-create already rejects this, so
        # reaching here means a stale/forged path — refuse rather than crash.
        _exp = (getattr(inputs, "expected_kind", "") or "").strip().lower()
        if _exp in ("short", "full"):
            _tk = ct.aspect_kind(contract.canvas_w, contract.canvas_h)
            if _tk != _exp:
                raise RuntimeError(
                    f"custom template {tid} is a {_tk}-form template ({contract.canvas_w}x"
                    f"{contract.canvas_h}) but this render needs a {_exp}-form template; "
                    f"refusing to render a mismatched aspect.")

        # Identify whether the template marks its own logo / watermark location.
        has_logo_slot = any(s.kind == "logo" for s in contract.slots)
        has_wm_slot = any(s.kind == "text" and s.key == "watermark" for s in contract.slots)
        wm_text = getattr(inputs, "watermark_text", "") or ""

        # Build the REAL content for this template's slots via the OFFLINE filler, instead
        # of pouring the single title into a fixed set of keys. This is what stops author
        # placeholders ("Your headline goes here", a "...lower third" ticker, "Supporting
        # image") from leaking — every text slot maps to real story content (or is cleared
        # by the renderer), and image slots fill positionally from the supporting images.
        _exp_kind = "full" if _exp == "full" else "short"
        content = ct.ContentBundle(
            headline=(inputs.headline or inputs.title_text or ""),
            headline_alt=(inputs.headline_alt or ""),
            subtitle=(inputs.subtitle or ""),
            body=(inputs.body or ""),
            kicker=(inputs.kicker or ""),
            cta=(inputs.cta or ""),
            ticker=(inputs.ticker or ""),
            channel_name=(inputs.title_text or ""),
            watermark=(wm_text if has_wm_slot else ""),
            images=[p for p in (inputs.support_images
                                or ([inputs.image_path] if inputs.image_path else [])) if p],
            logo_path=(inputs.brand_logo if has_logo_slot else None),
            text_color=(inputs.text_color or None),
            stories=list(inputs.stories or []),
        )
        # AI-generate-to-fit: shorten any slot whose mapped text overflows its capacity
        # (Gemini, cached, gated by KAIZER_TEMPLATE_AI_FIT) so a small box reads well; the
        # renderer's auto-fit still guarantees a pixel-exact fit. Falls back to offline.
        try:
            texts, sf = ct.fit_texts(contract, content, kind=_exp_kind, db=db)
        except Exception:
            sf = ct.build_slot_fill(contract, content, kind=_exp_kind)
            texts = dict(sf.texts)
        brand = dict(sf.brand)
        # Editor per-slot TEXT overrides win over the filler/AI text (operator typed it).
        # Only apply to real text slots; an empty override clears the slot (renderer hides it).
        _ov = inputs.template_text_overrides or {}
        if _ov:
            _text_keys = {s.key for s in contract.text_slots}
            for _k, _v in _ov.items():
                if _k in _text_keys:
                    _vs = ("" if _v is None else str(_v)).strip()
                    if _vs:
                        texts[_k] = _vs
                    else:
                        texts.pop(_k, None)   # explicit blank -> clear (renderer hides it)

        # Per-slot media: the MAIN slot gets the AI-trimmed clip; the other slots get
        # the user's chosen media (resolved to local paths by the orchestrator). Falls
        # back to filling all video slots with the trimmed clip when nothing was chosen.
        kindmap = {s.key: s.kind for s in contract.slots}
        main_slot = (inputs.main_media_slot or "").strip()
        if not main_slot:
            vs = contract.video_slots
            main_slot = vs[0].key if vs else "video"
        per_slot = inputs.template_media or {}
        videos = {main_slot: inputs.trimmed_short_path}
        images = dict(sf.images)   # filler's positional supporting images
        intro_path = None
        for slot, path in per_slot.items():
            if not path or slot == main_slot:
                continue
            k = kindmap.get(slot)
            if k in ("video", "background"):
                videos[slot] = path
            elif k == "image":
                images[slot] = path        # user-chosen per-slot media wins over the filler
            elif k == "intro":
                intro_path = path
        # Backward-compat: the editor's single chosen image fills any image slot still empty.
        if inputs.image_path:
            for s in contract.image_slots:
                images.setdefault(s.key, inputs.image_path)

        # Logo + watermark are CLEAN-MASTER: leave their slots empty here and let the
        # per-channel publish stamp (pipeline_v4.watermark.stamp_for_channel) place each
        # channel's OWN logo/watermark — AT the template's marked slot when it has one
        # (rects persisted below), else the channel's default corner. Matches the operator's
        # rule "template says where -> there; else default" + keeps per-channel branding.
        req = ct.RenderRequest(
            videos=videos,
            texts=texts,
            images=images,
            logo_path=None,
            brand=brand, fps=30,
            main_slot=main_slot, intro_path=intro_path,
            # scrolling ticker: the resolved ticker text (the still capture hides it, then a
            # marquee strip is composited at the ticker slot). speed/colour/font are optional.
            ticker_text=(texts.get("ticker") or texts.get("marquee") or (inputs.ticker or None)),
            ticker_speed_s=getattr(inputs, "ticker_speed_s", None),
            ticker_bg_color=getattr(inputs, "ticker_bg_color", None),
            ticker_font_px=getattr(inputs, "ticker_font_px", None),
            ticker_lang=(getattr(inputs, "language", None) or None),
        )
        work = str(Path(inputs.output_path).parent / f"_ctmpl_{tid}")
        _report = ct.render_template(bundle, contract, req, work_dir=work, out_path=inputs.output_path)
        # Persist the logo/watermark slot rects (canvas px) next to the master so the
        # publish stamp can position each channel's logo/watermark at the marked spot.
        try:
            _br = (_report or {}).get("brand_rects") or {}
            if _br.get("logo") or _br.get("watermark"):
                import json as _json
                with open(inputs.output_path + ".slots.json", "w", encoding="utf-8") as _fh:
                    _json.dump({"canvas": [contract.canvas_w, contract.canvas_h],
                                "logo": _br.get("logo"), "watermark": _br.get("watermark")}, _fh)
        except Exception:
            pass
        try:
            t.use_count = (t.use_count or 0) + 1
            db.commit()
        except Exception:
            db.rollback()
        return {
            "logo_in_slot": bool(has_logo_slot and inputs.brand_logo),
            "watermark_in_slot": bool(has_wm_slot and wm_text.strip()),
        }
    finally:
        db.close()


def render_short(inputs: ShortRenderInputs) -> str:
    """Compose one V1-style short (torn_card / clean_card / split_frame
    / follow_bar). Returns the output path on success; raises on failure."""
    raw_layout = (inputs.layout or DEFAULT_SHORTS_LAYOUT)
    _is_custom = str(raw_layout).lower().startswith("custom:")
    layout = str(raw_layout) if _is_custom else str(raw_layout).lower()
    if not _is_custom and layout not in SUPPORTED_SHORTS_LAYOUTS:
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

    # KAIZER_CLEAN_MASTER (Decision 1): when "1", do not pass the brand
    # logo into the V1 short composers — the Phase 2 Branding Worker
    # owns the only logo overlay pass. Read the env at call time so
    # tests + ops can flip without a reload. Default "0" (Decision 12).
    # See docs/upload-rewrite/DECISIONS.md Decision 1.
    _clean_master = os.environ.get("KAIZER_CLEAN_MASTER", "0").strip() == "1"
    _short_logo = None if _clean_master else inputs.brand_logo

    _custom_brand = {}
    if _is_custom:
        _custom_brand = _render_custom_short(inputs, layout) or {}
    elif layout == "torn_card":
        from pipeline_core.pipeline import compose_clip
        compose_clip(
            inputs.trimmed_short_path,
            inputs.image_path,
            inputs.title_text or "KAIZER X",
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
            inputs.title_text or "KAIZER X",
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
            video_logo=_short_logo,
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
            follow_text=fp.get("follow_text", "FOLLOW KAIZER X TELUGU"),
            follow_text_color=fp.get("follow_text_color", "#ffffff"),
            video_logo=_short_logo,
            platform="youtube_short",
        )

    # Watermark post-pass — V1's short composers don't expose an
    # overlay slot, so we re-encode once to drop the user's translucent
    # text+logo bug onto every frame. Skipped when neither watermark
    # text nor a brand logo is configured.
    #
    # KAIZER_CLEAN_MASTER (Decision 1): when "1", we also suppress the
    # logo from this watermark post-pass — the Phase 2 Branding Worker
    # handles it downstream. The user-text watermark itself is also
    # gated by `inputs.brand_logo` here, so when the logo is dropped
    # and the text is empty, the whole post-pass is a no-op.
    wm_text = getattr(inputs, "watermark_text", "") or ""
    wm_op   = float(getattr(inputs, "watermark_opacity", 0.35) or 0.35)
    wm_pos  = getattr(inputs, "watermark_position", "top-right") or "top-right"
    _wm_logo = None if _clean_master else inputs.brand_logo
    # Custom template already placed the logo/watermark at its own marked location →
    # don't also stamp the default corner (avoid a double logo / double watermark).
    if _custom_brand.get("logo_in_slot"):
        _wm_logo = None
    if _custom_brand.get("watermark_in_slot"):
        wm_text = ""
    if (wm_text.strip() or _wm_logo) and os.path.isfile(inputs.output_path):
        # Keyed by output stem: shorts now render in parallel (Wave 4
        # item C) and a shared "_wm_short.png" would race.
        wm_path = str(Path(inputs.work_dir)
                      / f"_wm_{Path(inputs.output_path).stem}.png")
        try:
            _render_watermark_png(
                text=wm_text, logo_path=_wm_logo,
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
                *_enc_args(crf=20, preset_hint="veryfast"),
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
