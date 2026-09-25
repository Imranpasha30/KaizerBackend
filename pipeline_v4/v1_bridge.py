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
import re
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


# ── Full-form effects (operator: "effects not only for the trailer") ─
# User-chosen per job: KAIZER_V4_EFFECTS_MODE = off/None (legacy, byte-
# identical) | auto (tasteful broadcast polish) | rich (the job's
# content-type style-pack look on every story). Applied to the story's
# SOURCE video before framing, so tile + full-screen gaps match.

_NEWS_POLISH_VF = ("eq=contrast=1.06:saturation=1.08,"
                   "unsharp=5:5:0.5:5:5:0.0,vignette=PI/6")


def _effects_vf_from_env(mode_override: Optional[str] = None) -> str:
    """Resolve the job's effects mode to a linear -vf fragment ('' =
    legacy untouched render). ``mode_override`` (from the canvas, editor
    re-render path) wins over the env. Fail-soft: never raises."""
    mode = ((mode_override or "").strip().lower()
            or (os.environ.get("KAIZER_V4_EFFECTS_MODE") or "").strip().lower())
    if mode in ("", "off", "none", "legacy"):
        return ""
    if mode == "rich":
        try:
            from pipeline_v4.trailer_styles import get_style
            ct = (os.environ.get("KAIZER_V4_CONTENT_TYPE") or "").strip().lower()
            p = get_style(ct if ct and ct != "auto" else "news")
            chain = p.grade + ("," + p.extra_vf if p.extra_vf else "")
            if ";" in chain:            # graph chains can't prefix a branch
                chain = p.grade
            return chain + ",vignette=PI/5.5"
        except Exception:
            return _NEWS_POLISH_VF
    return _NEWS_POLISH_VF              # "auto" and anything unknown


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
                          pool_dir: Optional[Path] = None,
                          gap_windows: tuple = (),
                          name_straps: tuple = (),
                          bleep_spans: tuple = (),
                          effects_vf: str = "") -> str:
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
        # Eased lower-third entrance — conditional ingredient (only when
        # set) so every pre-motion canvas keeps its exact old hash.
        if getattr(layout, "lt_ease", None):
            layout_blob["lt_ease"] = layout.lt_ease
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
                # Explicit None check: a user-set 0 (framing grid "face at
                # top/left") is a REAL value — `or 50.0` aliased it to center.
                _oxv = _ia(img, "offset_x_pct", None)
                _oyv = _ia(img, "offset_y_pct", None)
                ox = 50.0 if _oxv is None else round(float(_oxv), 1)
                oy = 50.0 if _oyv is None else round(float(_oyv), 1)
            except (TypeError, ValueError):
                ts = te = ed = 0.0; ox = oy = 50.0
            # Content fingerprint via the resolved pool file (so a same-named
            # overwrite still busts); falls back to {"p": None} if unresolved
            # — the raw src + timing below still change the hash on edits.
            fp = _fp(_resolve_pool_image(src, pool_dir)) if pool_dir else {"p": None}
            entry = {
                "src": src, "ts": ts, "te": te, "ed": ed,
                "effect": _ia(img, "effect", "") or "",
                "fit": _ia(img, "fit", "") or "", "ox": ox, "oy": oy, "fp": fp,
            }
            # Phase-1 engine fields — hashed ONLY when non-default so every
            # pre-engine canvas keeps its exact old hash (100% cache hits →
            # byte-identical re-renders). Rule: any NEW render-affecting
            # CanvasImage field gets a conditional entry here + a test.
            spot = _ia(img, "spotlight", None)
            if spot and spot != "off":
                entry["spot"] = spot
            out.append(entry)
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
    # Gap fallback participates in the render ONLY when gaps exist, so the
    # ingredient is added conditionally: gapless (i.e. every pre-engine)
    # canvas keeps its exact old hash → 100% cache hits, byte-identical.
    if gap_windows:
        blob["gapfb"] = [[round(float(a), 3), round(float(b), 3)]
                         for a, b in gap_windows]
    # Name-strap (polish A) — same conditional rule: only stories with the
    # strap on hash its (label, window) list; everything else is untouched.
    if name_straps:
        blob["strap"] = [[str(t), round(float(a), 3), round(float(b), 3)]
                         for (t, a, b) in name_straps]
    # Bleep censor — the composed clip's audio is sliced from the (bleeped)
    # trimmed master, so the spans overlapping THIS story are render inputs.
    # Conditional: unbleeped jobs keep their exact old hash.
    if bleep_spans:
        blob["bleep"] = [[round(float(a), 3), round(float(b), 3)]
                         for (a, b) in bleep_spans]
    # Full-form effects — conditional: legacy/off jobs keep their exact
    # old hash; the RESOLVED chain is hashed so a pack/polish tweak
    # deterministically re-renders.
    if effects_vf:
        blob["fx"] = effects_vf
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
SUPPORTED_SHORTS_LAYOUTS = ("torn_card", "clean_card", "split_frame",
                            "follow_bar", "dual_video")

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

# Full-screen spotlight pop transition. A spotlight is a FULL-SCREEN opaque
# image overlaid on top of the always-present FRAMED base layout (main tile +
# sidebar). An alpha cross-dissolve here blends the fullscreen image against
# that DIFFERENT framed layout underneath → a double-exposure "ghost/flicker"
# at every appear/disappear (operator-reported). 0.0 = clean hard cut
# (broadcast-standard for a full-screen graphic pop; guaranteed ghost-free).
# A future structure-preserving transition (push/wipe or fullscreen-anchor
# dissolve) can reintroduce softness without ghosting.
V4_SPOTLIGHT_FADE_S = 0.0


def _flag_on(name: str, default: str = "1") -> bool:
    """Env kill-switch helper: '0'/'off'/'false' disables (default ON) —
    the same convention as KAIZER_V4_LIVE_LAYOUTS in the orchestrator."""
    return ((os.environ.get(name, default) or default)
            .strip().lower() not in ("0", "off", "false"))


# Overlay ENTRANCE anims by catalog family. Lower-anchored furniture rises
# from below; top-anchored chips drop from above; full-frame treatments
# (countdown/frame_hud = "cover") and rotated stamps keep the legacy alpha
# fade — a slide reads wrong for them.
_OVERLAY_ANIM_BY_FAMILY = {
    "banner": "slide_up", "headline_tag": "slide_up",
    "attribution": "slide_up", "poll_bar": "slide_up",
    "social_card": "slide_up", "info_panel": "slide_up",
    "bug": "slide_down", "locator": "slide_down", "progress": "slide_down",
    "score_bug": "slide_down", "weather_chip": "slide_down",
    "market_chip": "slide_down",
}


def _overlay_anim(item_id: str) -> str:
    """Entrance for one Director-overlay id via its catalog family.
    Fail-soft: unknown id/family or any import error → legacy fade."""
    try:
        from pipeline_v4.overlays import REGISTRY
        row = next((r for r in REGISTRY if r["id"] == item_id), None)
        return _OVERLAY_ANIM_BY_FAMILY.get(row["family"], "fade") if row else "fade"
    except Exception:
        return "fade"


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
    # Story-to-story transition at the stitch. None = legacy hard cut;
    # a safe xfade name ("fade", "wipeleft", …) crossfades video+audio
    # between stories. Falls back to the hard-cut stitcher on failure.
    story_transition: Optional[str] = None
    # Full-form effects mode from the canvas ("auto"/"rich"/"off").
    # None/"" = fall back to KAIZER_V4_EFFECTS_MODE env (runner path);
    # editor re-renders pass the canvas value so no process-global env
    # is needed in uvicorn.
    effects_mode: Optional[str] = None
    # AI Director per-story decisions {story_index: StoryDirective} —
    # per-story mood pack + garnish fx + per-joint transitions. None =
    # no Director (legacy / gate off). Overrides effects_mode per story.
    directives: Optional[dict] = None


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
    # dual_video layout only: the SECOND video (top = trimmed_short_path,
    # bottom = this). Typically the audio-first reference b-roll or a
    # second speaker cam. Missing/unreadable → torn_card fallback.
    second_video_path: Optional[str] = None
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
    # Phase 2 — EFFECTS grade for the short: a resolved LINEAR -vf chain
    # (the AI Director's per-story mood grade + garnish, or the effects-mode
    # base grade) applied as one post-composite pass so the short matches
    # its bulletin story's look. "" = no grade (render stays byte-identical).
    effects_vf: str = ""


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


def _focal_crop(w: int, h: int, ox=50.0, oy=50.0) -> str:
    """Cover-crop honoring the canvas focal point (offset_x/y_pct — set by
    the 9-point focal-grid editor or face_focus auto-detection).

    50/50 emits the bare legacy ``crop=w:h`` (ffmpeg's default crop x/y IS
    the center) so every pre-focal graph stays CHARACTER-identical — the
    byte-compat guarantee the golden snapshots in test_layout_live /
    test_carousel_gaps pin. x is inherently in range: ox∈[0,1] and
    ``scale=...:force_original_aspect_ratio=increase`` guarantees
    in_w>=w, so x∈[0,in_w-w] (ffmpeg's vf_crop clamps anyway). ``.3f`` is
    lossless for editor values (the grid writes 0/50/100; carousel segs
    round to 1 decimal)."""
    try:
        ox = max(0.0, min(100.0, float(ox)))
        oy = max(0.0, min(100.0, float(oy)))
    except (TypeError, ValueError):
        ox = oy = 50.0
    if ox == 50.0 and oy == 50.0:
        return f"crop={w}:{h}"
    return (f"crop={w}:{h}:(in_w-{w})*{ox / 100.0:.3f}:"
            f"(in_h-{h})*{oy / 100.0:.3f}")


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
    try:
        _run_ffmpeg(cmd, timeout=600, log_label="slice_video", retry=2)
    except RuntimeError:
        # LAST RESORT (job 600): input-seek slicing died on every rung
        # while a full-file decode of the same source was clean — retry
        # with ACCURATE output-seek (-ss after -i) on pure CPU. Decodes
        # from the file start (slow for late stories) but is immune to
        # seek-position and GPU-session pathologies.
        print("[v4/v1_bridge] slice_video: input-seek exhausted — "
              "final accurate-seek CPU attempt", flush=True)
        slow = [
            _ffmpeg_bin(), "-y", "-v", "error",
            "-i", source_path,
            "-ss", f"{float(start_sec):.3f}",
            "-t", f"{dur:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart",
            output_path,
        ]
        _run_ffmpeg(slow, timeout=1800, log_label="slice_video_slow", retry=0)
    return output_path


def _resolve_sidebar(
    *,
    work_dir: Path,
    story_index: int,
    pool_image_path: Optional[str],
    offset_x_pct: float = 50.0,
    offset_y_pct: float = 50.0,
) -> str:
    """Return a sidebar PNG path (real image or generated placeholder).

    Idempotence guard (cache fix): regenerating the PNG on every call
    changed its mtime, and the per-story hash fingerprints the sidebar
    file — so every STATIC-sidebar story re-rendered on every identical
    run (carousel stories were immune via the carousel .sig). Regenerate
    only when the source image actually changed."""
    from pipeline_core.longform_compose import make_sidebar_placeholder
    try:
        offset_x_pct = float(offset_x_pct)
        offset_y_pct = float(offset_y_pct)
    except (TypeError, ValueError):
        offset_x_pct = offset_y_pct = 50.0
    out = work_dir / f"_sidebar_{story_index:02d}.png"
    try:
        _st = (os.stat(pool_image_path)
               if (pool_image_path and os.path.isfile(pool_image_path))
               else None)
        sig = (f"{pool_image_path or ''}|{_st.st_size if _st else 0}"
               f"|{int(_st.st_mtime) if _st else 0}")
    except OSError:
        sig = f"{pool_image_path or ''}|0|0"
    # Focal framing forks the .sig ONLY when non-default: the default sig
    # string stays byte-identical (no cold PNG regen fleet-wide), while a
    # reframe regenerates the PNG. Without this the story HASH would bust
    # (ox/oy are hashed) but the stale centered _sidebar_NN.png — same
    # source file, same mtime — would silently be reused.
    if offset_x_pct != 50.0 or offset_y_pct != 50.0:
        sig += f"|fx{offset_x_pct:.1f},{offset_y_pct:.1f}"
    sig_file = str(out) + ".sig"
    if out.is_file():
        try:
            if Path(sig_file).read_text(encoding="utf-8") == sig:
                return str(out)          # unchanged source — reuse
        except OSError:
            pass
    res = make_sidebar_placeholder(pool_image_path, str(out),
                                   offset_x_pct=offset_x_pct,
                                   offset_y_pct=offset_y_pct)
    try:
        Path(sig_file).write_text(sig, encoding="utf-8")
    except OSError:
        pass
    return res


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
        # Video cutaways (media_kind="video") render via the dedicated
        # video-spotlight path — never as a static sidebar segment.
        if (_attr(img, "media_kind", "image") or "image") == "video":
            continue
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
            # None check (not `or`): 0 is a legitimate edge framing value.
            _oxv = _attr(img, "offset_x_pct", None)
            _oyv = _attr(img, "offset_y_pct", None)
            ox = 50.0 if _oxv is None else round(float(_oxv), 1)
            oy = 50.0 if _oyv is None else round(float(_oyv), 1)
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
    # COALESCE runs of the SAME image (job 608: the timing engine split one
    # image into 25 back-to-back 4s windows — the carousel then built a
    # 25-LAYER filtergraph compositing the identical picture 25×, frame by
    # frame, which alone made a 4-story bulletin take ~50min). Merging
    # contiguous same-src/same-look windows into ONE is visually IDENTICAL
    # (same image over the same span) but collapses the layer count, so the
    # render is dramatically faster. Sorted by start so adjacency is real.
    segs.sort(key=lambda s: s["ts"])
    merged: list[dict] = []
    for s in segs:
        if merged:
            p = merged[-1]
            same_look = (p["path"] == s["path"] and p["effect"] == s["effect"]
                         and p["fit"] == s["fit"] and p["ox"] == s["ox"]
                         and p["oy"] == s["oy"])
            # contiguous or overlapping (small gap tolerated: same picture,
            # so bridging a sub-0.5s seam changes nothing visible)
            if same_look and s["ts"] <= p["te"] + 0.5:
                p["te"] = max(p["te"], s["te"])
                continue
        merged.append(dict(s))
    return merged


def _render_sidebar_carousel(
    *, segs: list[dict], out_path: str, story_duration: float,
    sidebar_w: int = V4_SIDE_INNER_W, sidebar_h: int = V4_TILE_INNER_H,
    bg_color: str = "black",
    base_image: Optional[str] = None,
    base_ox: float = 50.0, base_oy: float = 50.0,
) -> Optional[str]:
    """Render the story's image carousel as a sidebar-sized silent mp4 —
    each image cover-fitted into the panel and shown in its
    ``[t_start, t_end]`` window with a fade in/out. Returns ``out_path`` on
    success, or None to fall back to a static sidebar.

    ``base_image``: full-duration bottom plate under the timed windows —
    designed-layout panels must NEVER show bare background between image
    windows (operator-reported: an OTS box rendered as a black plate for
    most of the story). None keeps the legacy graph byte-identical."""
    if not segs:
        return None

    # BELT (job 625): an undecodable "image" (an HTML bot-wall saved as
    # .jpg, a truncated download, a corrupt upload) makes this ffmpeg hang
    # until its 300s timeout. PIL-verify every input first; bad frames are
    # dropped so the carousel degrades gracefully (or falls back to the
    # static sidebar) instead of stalling the whole compose.
    def _decodable(p) -> bool:
        try:
            from PIL import Image as _PILImage
            with _PILImage.open(p) as _im:
                _im.verify()
            return True
        except Exception as _bad:
            print(f"[v4/v1_bridge] carousel input not decodable ({_bad}) "
                  f"-- dropped {p}", flush=True)
            return False

    segs = [s for s in segs if _decodable(s["path"])]
    if not segs:
        return None
    if base_image and not _decodable(base_image):
        base_image = None

    dur = max(float(story_duration or 0.0), max(s["te"] for s in segs)) + 0.05
    ffmpeg = _ffmpeg_bin()
    inputs = ["-f", "lavfi", "-t", f"{dur:.3f}",
              "-i", f"color=c={bg_color or 'black'}:s={sidebar_w}x{sidebar_h}:r=25"]
    for s in segs:
        if s.get("effect") == "zoom_in":
            # zoompan synthesizes the whole window from ONE still frame
            # (no -loop): looping would multiply the generated frames.
            inputs += ["-i", s["path"]]
        else:
            inputs += ["-loop", "1", "-i", s["path"]]
    if base_image:
        inputs += ["-loop", "1", "-i", base_image]
    chain: list[str] = []
    last = "[0:v]"
    if base_image:
        _bidx = len(segs) + 1
        chain.append(
            f"[{_bidx}:v]scale={sidebar_w}:{sidebar_h}:force_original_aspect_ratio=increase,"
            f"{_focal_crop(sidebar_w, sidebar_h, base_ox, base_oy)},setsar=1[imbase]")
        chain.append(f"{last}[imbase]overlay[vbase]")
        last = "[vbase]"
    for idx, s in enumerate(segs, start=1):
        if s.get("effect") == "zoom_in":
            # Polish C motion: gentle Ken-Burns push-in (≈1.5%/s, capped
            # at 1.15x) so a long dwell never sits static. Upsample 2x
            # first — zoompan on a panel-sized still jitters. setpts
            # shifts the synthesized clip to the window start; the
            # overlay's enable clips it to [ts, te].
            win = max(0.1, s["te"] - s["ts"])
            frames = max(2, int(round(win * 25)))
            # Focal cover-crop (face-aware framing / 9-point grid): the
            # segs already carry ox/oy — they used to feed ONLY the cache
            # key while the chain center-cropped (the beheaded-face bug).
            # zoompan's own centered pan stays: it zooms ≤1.15x around the
            # already-focal-cropped frame.
            scale = (
                f"[{idx}:v]scale={sidebar_w * 2}:{sidebar_h * 2}:"
                f"force_original_aspect_ratio=increase,"
                f"{_focal_crop(sidebar_w * 2, sidebar_h * 2, s.get('ox', 50.0), s.get('oy', 50.0))},"
                f"zoompan=z='min(1+0.0006*on,1.15)':"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={frames}:s={sidebar_w}x{sidebar_h}:fps=25,"
                f"setsar=1,setpts=PTS+{s['ts']:.3f}/TB"
            )
        else:
            scale = (
                f"[{idx}:v]scale={sidebar_w}:{sidebar_h}:force_original_aspect_ratio=increase,"
                f"{_focal_crop(sidebar_w, sidebar_h, s.get('ox', 50.0), s.get('oy', 50.0))},setsar=1"
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


def _mirror_rect_pct(rect):
    """Mirror an (x, y, w, h) pct rect across the vertical centreline."""
    x, y, w, h = rect
    return (round(100.0 - x - w, 3), y, w, h)


def _face_boxes_for_probe(src_path: str, t_abs: float, probe_path: str) -> Optional[list]:
    """Grab ONE frame of ``src_path`` at ``t_abs`` into ``probe_path`` and
    return its face boxes in frame-pct — None when the frame can't be
    read (distinct from [] = frame fine, no faces). Split out so tests
    can stub the expensive ffmpeg+cv2 part."""
    try:
        if not (os.path.isfile(probe_path) and os.path.getsize(probe_path) > 0):
            r = subprocess.run(
                [_ffmpeg_bin(), "-y", "-v", "error", "-ss", f"{max(0.0, t_abs):.3f}",
                 "-i", src_path, "-frames:v", "1", "-q:v", "4", probe_path],
                capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                return None
        if not (os.path.isfile(probe_path) and os.path.getsize(probe_path) > 0):
            return None
        from pipeline_v4 import face_focus
        return face_focus.face_boxes_pct(probe_path)
    except Exception:
        return None


def _panel_face_safety(sgeom, *, src_path: str, t0: float, t1: float,
                       work_dir: str, story_index: int, gaps=()):
    """Picture-panel guard (operator-reported on job 598: the box sat ON
    the co-host, and with sparse image windows it rendered as a black
    plate over the speaker). Two situations put the panel over PEOPLE:

    * a truly FLOATING panel (layout video is full-frame — OTS family):
      MIRROR the box to the cleaner side, or STRIP it when both sides
      carry faces (a two-shot);
    * a framed layout whose GAP windows dominate the story (every
      image is a timed spotlight → the composer shows full-screen video
      almost throughout, with the panel layered above it): STRIP-only —
      mirroring could collide with the video tile in the framed moments.

    Samples the story's own frames (gap midpoints when gaps dominate),
    detects faces via face_focus. Returns ``(sgeom, action)``, action ∈
    {"", "flip", "veto"}. Fail-soft: any problem returns the input
    unchanged. Kill switch: KAIZER_V4_PANEL_SAFETY (default ON).
    Cache-safe by construction: corrections mutate the rects serialized
    into the |slay: fingerprint, so exactly the corrected stories
    re-render."""
    try:
        if sgeom is None or not _flag_on("KAIZER_V4_PANEL_SAFETY"):
            return sgeom, ""
        rect = sgeom.picture or ((sgeom.pips[0].x, sgeom.pips[0].y,
                                  sgeom.pips[0].w, sgeom.pips[0].h)
                                 if sgeom.pips else None)
        if not rect:
            return sgeom, ""
        vx, vy, vw, vh = sgeom.video
        span = max(0.0, t1 - t0)
        floating = (vx <= 2.0 and vy <= 2.0 and vw >= 96.0 and vh >= 96.0)
        gap_cov = sum(max(0.0, b - a) for (a, b) in (gaps or ()))
        gap_heavy = bool(span > 0 and gaps and gap_cov / span >= 0.5)
        if not (floating or gap_heavy):
            return sgeom, ""
        if not (src_path and os.path.isfile(src_path)):
            return sgeom, ""
        if floating:
            probes = ([t0 + span * 0.3, t0 + span * 0.7] if span >= 2.0
                      else [t0 + span * 0.5])
        else:
            # sample where the panel actually floats: the longest gaps
            _gs = sorted((gaps or ()), key=lambda g: g[1] - g[0],
                         reverse=True)[:2]
            probes = [t0 + (a + b) / 2.0 for (a, b) in _gs]
        mirrored = _mirror_rect_pct(rect)

        def _overlap(face, r) -> bool:
            fx, fy, fw, fh = face
            rx, ry, rw, rh = r
            ix = max(0.0, min(fx + fw, rx + rw) - max(fx, rx))
            iy = max(0.0, min(fy + fh, ry + rh) - max(fy, ry))
            fa = fw * fh
            return fa > 0 and (ix * iy) / fa >= 0.35

        hit_panel = hit_mirror = False
        saw_frame = False
        for tt in probes:
            probe = os.path.join(
                work_dir, f"_face_probe_{story_index:02d}_{int(tt * 10)}.jpg")
            boxes = _face_boxes_for_probe(src_path, tt, probe)
            if boxes is None:
                continue
            saw_frame = True
            for b in boxes:
                if _overlap(b, rect):
                    hit_panel = True
                if _overlap(b, mirrored):
                    hit_mirror = True
        if not saw_frame or not hit_panel:
            return sgeom, ""
        import dataclasses
        if gap_heavy and not floating:
            # framed layout, panel above fullscreen gaps: strip only.
            return dataclasses.replace(sgeom, picture=None, picture_kind="",
                                       pips=()), "veto"
        if not hit_mirror:
            if sgeom.picture:
                return dataclasses.replace(sgeom, picture=mirrored), "flip"
            from pipeline_v4.layout_library import PipInset
            p0 = sgeom.pips[0]
            mp = _mirror_rect_pct((p0.x, p0.y, p0.w, p0.h))
            return dataclasses.replace(
                sgeom, pips=(PipInset(mp[0], mp[1], mp[2], mp[3],
                                      getattr(p0, "label", "")),)), "flip"
        # Both sides carry faces (a two-shot) — no clean side exists:
        # strip the panel; better an honest anchor shot than a box on a
        # face. Spotlights and PiP moments still play over it.
        return dataclasses.replace(sgeom, picture=None, picture_kind="",
                                   pips=()), "veto"
    except Exception:
        return sgeom, ""


def _gap_windows(
    images, pool_dir: Path, story_duration: float,
    *, min_gap: float = 0.5, cap: int = 12,
) -> list[tuple[float, float]]:
    """Complement of the story's image windows over [0, story_duration] —
    the stretches where NO image is scheduled, so the renderer should cut
    the main video full-screen ("back to the anchor").

    Deliberately returns [] when the story has no resolvable image
    windows at all: legacy no-image stories keep their placeholder-
    sidebar look (and their cache hash) unchanged — the gap fallback
    only activates for canvases where the timing engine actually placed
    sparse windows. Sliver gaps < ``min_gap`` are skipped (a sub-second
    video flash reads as a glitch); at most ``cap`` gaps are returned
    (longest kept) to bound the ffmpeg enable-expression length."""
    dur = float(story_duration or 0.0)
    if dur <= 0.0:
        return []
    segs = _carousel_segments(images, pool_dir)
    if not segs:
        return []
    ivals = sorted((s["ts"], min(s["te"], dur)) for s in segs)
    merged: list[list[float]] = []
    for a, b in ivals:
        if merged and a <= merged[-1][1] + 0.01:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for a, b in merged:
        if a - cursor >= min_gap:
            gaps.append((round(cursor, 3), round(a, 3)))
        cursor = max(cursor, b)
    if dur - cursor >= min_gap:
        gaps.append((round(cursor, 3), round(dur, 3)))
    if len(gaps) > cap:
        gaps = sorted(sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)[:cap])
    return gaps


def _spotlight_windows(
    images, pool_dir: Path, story_duration: float,
    *, kind: str = "fullscreen", cap: int = 2,
) -> list[tuple[str, float, float, float, float]]:
    """[(abs_image_path, t_start, t_end, ox, oy), …] for images the canvas
    marks ``spotlight=kind`` — the moments that pop full-screen (or PiP).
    The canvas is the source of truth (auto-selection was materialized at
    build/resync time), so the render just honors it. ox/oy = the image's
    focal point (offset_x/y_pct — face-aware framing / 9-point grid); the
    graph builders also accept legacy 3-tuples (50/50 implied) so existing
    callers/tests keep working. Capped to bound ffmpeg input count;
    earliest windows win."""
    dur = float(story_duration or 0.0)
    if dur <= 0.0:
        return []
    def _attr(obj, name, default=None):
        return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)
    out: list[tuple[str, float, float, float, float]] = []
    for img in (images or []):
        # Video cutaways are handled by _video_spotlight_windows, not here.
        if (_attr(img, "media_kind", "image") or "image") == "video":
            continue
        if (_attr(img, "spotlight", None) or "") != kind:
            continue
        path = _resolve_pool_image(_attr(img, "src", "") or "", pool_dir)
        if not path:
            continue
        try:
            ts = max(0.0, float(_attr(img, "t_start", 0.0) or 0.0))
            te = min(dur, float(_attr(img, "t_end", 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
        if te <= ts + 0.05:
            continue
        # Same rounding as _carousel_segments so the graph and the cache
        # key (ox/oy hash unconditionally in _carousel_blob) agree.
        try:
            # None check (not `or`): 0 is a legitimate edge framing value.
            _fxv = _attr(img, "offset_x_pct", None)
            _fyv = _attr(img, "offset_y_pct", None)
            fx = 50.0 if _fxv is None else round(float(_fxv), 1)
            fy = 50.0 if _fyv is None else round(float(_fyv), 1)
        except (TypeError, ValueError):
            fx = fy = 50.0
        out.append((path, round(ts, 3), round(te, 3), fx, fy))
    out.sort(key=lambda x: x[1])
    return out[:max(0, cap)]


def _video_spotlight_windows(
    images, pool_dir: Path, story_duration: float, *, cap: int = 2,
) -> list[tuple[str, float, float, str, float]]:
    """[(video_path, t_start, t_end, audio_mode, trim_start), …] for canvas
    entries marked media_kind='video' + spotlight='fullscreen' — full-screen
    B-roll cutaways. Video files only; earliest windows win, capped to bound
    the ffmpeg input count."""
    dur = float(story_duration or 0.0)
    if dur <= 0.0:
        return []
    def _attr(obj, name, default=None):
        return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)
    out: list[tuple[str, float, float, str, float]] = []
    for img in (images or []):
        if (_attr(img, "media_kind", "image") or "image") != "video":
            continue
        if (_attr(img, "spotlight", None) or "") != "fullscreen":
            continue
        path = _resolve_pool_image(_attr(img, "src", "") or "", pool_dir)
        if not path:
            continue
        try:
            ts = max(0.0, float(_attr(img, "t_start", 0.0) or 0.0))
            te = min(dur, float(_attr(img, "t_end", 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
        if te <= ts + 0.05:
            continue
        am = str(_attr(img, "audio_mode", "mute") or "mute").strip().lower()
        if am not in ("mute", "duck", "auto"):
            am = "mute"
        try:
            tr = max(0.0, float(_attr(img, "video_trim_start", 0.0) or 0.0))
        except (TypeError, ValueError):
            tr = 0.0
        out.append((path, round(ts, 3), round(te, 3), am, round(tr, 3)))
    out.sort(key=lambda x: x[1])
    return out[:max(0, cap)]


def _ensure_sidebar_carousel(
    *, images, pool_dir: Path, out_path: str, story_duration: float,
    bg_color: str = "black",
    sidebar_w: Optional[int] = None,     # per-story layout picture rect —
    sidebar_h: Optional[int] = None,     # None = the classic sidebar panel
    base_image: Optional[str] = None,    # full-duration plate under the windows
    base_ox: float = 50.0, base_oy: float = 50.0,
) -> Optional[str]:
    """Render the carousel only when its content changed (guarded by a
    ``.sig`` sidecar) so the per-story render cache still works. Returns
    the carousel mp4 path, or None when the cheaper static sidebar is
    visually identical (no images, or ONE image covering the whole
    story). A single image with a NARROW window must go through the
    carousel path — the static sidebar would ignore its t_start/t_end
    and pin it on screen for the whole story."""
    segs = _carousel_segments(images, pool_dir)
    if not segs:
        return None
    # WinError 206 guard (job 592): every segment adds an ffmpeg input
    # plus two filter stages, and the Windows CreateProcess command line
    # tops out at ~32k chars — a story with dozens of images built an
    # unlaunchable command. Cap the carousel: keep the longest-visible
    # windows (earliest wins ties), render chronologically, and log
    # exactly what was dropped.
    try:
        _max_segs = max(4, int(os.environ.get(
            "KAIZER_V4_CAROUSEL_MAX_SEGS", "24") or "24"))
    except ValueError:
        _max_segs = 24
    _n_all = len(segs)
    _capped = _n_all > _max_segs
    if _capped:
        _rank = sorted(range(_n_all),
                       key=lambda k: (-(segs[k]["te"] - segs[k]["ts"]),
                                      segs[k]["ts"], k))
        _drop = [segs[k] for k in sorted(_rank[_max_segs:])]
        segs = [segs[k] for k in sorted(_rank[:_max_segs])]
        print(f"[v4/v1_bridge] sidebar carousel capped {_max_segs}/{_n_all} "
              f"segments (Windows command-length guard) — dropped: "
              + ", ".join(f"{os.path.basename(s['path'])}"
                          f"[{s['ts']:.1f}-{s['te']:.1f}s]"
                          for s in _drop[:8])
              + (f" (+{len(_drop) - 8} more)" if len(_drop) > 8 else ""),
              flush=True)
    if len(segs) == 1:
        s0 = segs[0]
        covers_story = (
            s0["ts"] <= 0.05
            and s0["te"] >= float(story_duration or 0.0) - 0.05
        )
        if covers_story:        # one full-story image → static sidebar is enough
            return None
    # Base plate is pointless when the timed windows already tile the
    # whole story — dropping it keeps those carousels' graphs and .sigs
    # exactly as before (no cold re-renders).
    if base_image:
        _cov = 0.0
        _edge = 0.0
        for s0 in sorted(segs, key=lambda s: s["ts"]):
            a = max(s0["ts"], _edge)
            if a - _edge > 0.3:
                break
            _edge = max(_edge, s0["te"])
        else:
            _cov = _edge
        if _cov >= float(story_duration or 0.0) - 0.3:
            base_image = None
    _sig_payload = {
        "segs": [{"s": s["src"], "ts": round(s["ts"], 3), "te": round(s["te"], 3),
                  "ed": round(s["ed"], 3), "fd": s["fade"], "fp": s["fp"],
                  "ef": s.get("effect"), "fit": s.get("fit"),
                  "ox": s.get("ox"), "oy": s.get("oy")} for s in segs],
        "dur": round(float(story_duration or 0.0), 3), "bg": bg_color or "black",
    }
    # Per-story layout dims — keyed ONLY when non-default so every
    # pre-layout carousel keeps its exact .sig (no cold re-renders).
    if sidebar_w or sidebar_h:
        _sig_payload["dims"] = [int(sidebar_w or 0), int(sidebar_h or 0)]
    # Cap fingerprint — keyed ONLY when the cap actually fired, so every
    # un-capped story keeps its exact legacy .sig (no cold re-renders);
    # changing KAIZER_V4_CAROUSEL_MAX_SEGS re-renders capped stories only.
    if _capped:
        _sig_payload["cap"] = [_max_segs, _n_all]
    # Base-plate fingerprint — conditional, same discipline as "cap".
    if base_image:
        try:
            _bstt = os.stat(base_image)
            _bfp = f"{_bstt.st_size}:{int(_bstt.st_mtime)}"
        except OSError:
            _bfp = "0:0"
        _sig_payload["pbase"] = [os.path.basename(base_image), _bfp,
                                 round(float(base_ox), 1), round(float(base_oy), 1)]
    sig = json.dumps(_sig_payload, sort_keys=True)
    sig_file = out_path + ".sig"
    if os.path.isfile(out_path) and os.path.isfile(sig_file):
        try:
            if Path(sig_file).read_text(encoding="utf-8") == sig:
                return out_path          # unchanged — reuse
        except OSError:
            pass
    res = _render_sidebar_carousel(
        segs=segs, out_path=out_path, story_duration=story_duration, bg_color=bg_color,
        sidebar_w=(int(sidebar_w) if sidebar_w else V4_SIDE_INNER_W),
        sidebar_h=(int(sidebar_h) if sidebar_h else V4_TILE_INNER_H),
        base_image=base_image, base_ox=base_ox, base_oy=base_oy,
    )
    if res:
        try:
            Path(sig_file).write_text(sig, encoding="utf-8")
        except OSError:
            pass
    return res


# ── Category sound design (Director on): stings + ticks + ducked bed ─

# Sting per mood pack — covers EVERY pack in trailer_styles.STYLES
# (tests/test_director_av.py pins the coverage + that every value is a
# real sound_library key). Families: urgency → sting_breaking; upbeat /
# celebratory → sting_positive; serious / dark → sting_negative;
# suspense / cliffhanger → sting_question; editorial-neutral packs keep
# the classic news sting. An explicit Director "sting" pick wins over
# this map (see _sting_key_for).
_STING_BY_PACK = {
    # news family / neutral-informative
    "news": "sting_news", "politics": "sting_news",
    "tech": "sting_news", "business": "sting_news",
    "finance": "sting_news", "health": "sting_news",
    "education": "sting_news", "history": "sting_news",
    "documentary": "sting_news",
    # urgency
    "breaking_news": "sting_breaking",
    # upbeat / celebratory
    "sports": "sting_positive", "cricket": "sting_positive",
    "action": "sting_positive", "comedy": "sting_positive",
    "festival": "sting_positive", "kids": "sting_positive",
    "music": "sting_positive", "gaming": "sting_positive",
    "travel": "sting_positive", "food": "sting_positive",
    "romance": "sting_positive", "celebrity": "sting_positive",
    "motivational": "sting_positive", "devotional": "sting_positive",
    "mythology": "sting_positive",
    # serious / dark
    "crime": "sting_negative", "horror": "sting_negative",
    "weather": "sting_negative", "drama": "sting_negative",
    # suspense / cliffhanger
    "thriller": "sting_question", "movie_review": "sting_question",
}


def _sting_key_for(d) -> str:
    """Sting id opening one story: the Director's explicit per-story
    ``sting`` pick wins WHEN it is a real library key (fail-soft: a
    stray id falls back rather than silencing the open); else the
    story's mood pack maps through _STING_BY_PACK; else the news sting.
    Pure — unit-tested."""
    try:
        from pipeline_v4.sound_library import LIBRARY
    except Exception:
        LIBRARY = {}
    k = str(getattr(d, "sting", "") or "").strip().lower()
    if k and k in LIBRARY:
        return k
    mood = str(getattr(d, "mood", "") or "").strip().lower() or "news"
    return _STING_BY_PACK.get(mood, "sting_news")


def _ui_key_for(d) -> str:
    """UI tick id under each Director graphic: the directive's
    ``ui_sound`` pick wins when it is a real library key; else the
    classic pop. Pure — unit-tested."""
    try:
        from pipeline_v4.sound_library import LIBRARY
    except Exception:
        LIBRARY = {}
    k = str(getattr(d, "ui_sound", "") or "").strip().lower()
    return k if (k and k in LIBRARY) else "ui_pop"


def _impact_key() -> str:
    """The emphasis-beat impact: impact_punch when the library has it,
    else the closest impact-family key (never invents an id — a rename
    in sound_library degrades to a sibling impact, not silence)."""
    try:
        from pipeline_v4.sound_library import LIBRARY
    except Exception:
        return "impact_punch"
    if "impact_punch" in LIBRARY:
        return "impact_punch"
    for k in sorted(LIBRARY):
        if k.startswith("impact"):
            return k
    return "impact_punch"


def _bed_wanted(directives_list) -> bool:
    """Majority vote across the stories' Director ``bed_on`` choices
    (missing/None attribute = True — the bed is default-on). The bed
    underscores the WHOLE stitched mix, so one story can't veto it;
    only a dominant no-bed vote skips it. Ties keep the bed. Pure."""
    votes: list[bool] = []
    for d in (directives_list or []):
        if d is None:
            continue
        v = getattr(d, "bed_on", True)
        votes.append(True if v is None else bool(v))   # None = unset = on
    if not votes:
        return True
    return votes.count(False) <= votes.count(True)


def _apply_sound_design(final_path: str, *, directives: dict, stories,
                        story_starts: list, stitched_dur: float,
                        work_dir: str) -> None:
    """CATEGORY SOUND DESIGN on the finished bulletin: a Director-picked
    (or mood-mapped) sting opens each story, a UI tick lands under each
    Director graphic, a whoosh rides every mid-story layout switch, an
    impact hits each emphasis beat, and the dominant mood's bed plays
    DUCKED under speech (sidechaincompress) — the TV mix; the stories'
    bed_on majority vote can switch the bed off. One audio-only pass
    (-c:v copy); ANY failure leaves the file untouched. Gate:
    KAIZER_V4_SOUND_DESIGN (default on when the Director ran).

    NOTE: this pass runs POST-STITCH on the FINAL file on every render —
    it is never part of a per-story compose, so nothing here participates
    in the per-story cache hash (no |tag: needed): a changed Director
    sound decision applies on the next render without re-rendering any
    story."""
    from pipeline_v4.sound_library import ensure_sound
    from pipeline_v4.trailer import synth_bed
    from pipeline_v4.trailer_styles import STYLES

    ffmpeg = _ffmpeg_bin()
    final_dur = _probe_clip_duration(final_path) or stitched_dur
    intro_off = max(0.0, final_dur - stitched_dur)
    events: list[tuple[str, float, float]] = []      # (wav, at, gain)
    moods: list[str] = []
    _dlist: list = []                                # for the bed_on vote
    for k, s in enumerate(stories or []):
        idx = int(getattr(s, "story_index", k) or k)
        d = directives.get(idx)
        if d is None or k >= len(story_starts):
            continue
        _dlist.append(d)
        at = story_starts[k] + intro_off
        moods.append(getattr(d, "mood", "") or "news")
        # Sting: the Director's explicit per-story pick wins; else the
        # story's mood pack maps through _STING_BY_PACK (all packs covered).
        sting = ensure_sound(_sting_key_for(d), Path(work_dir))
        if sting:
            events.append((sting, max(0.0, at - 0.1), 0.55))
        # UI tick under each Director graphic — directive ui_sound wins.
        for o in (getattr(d, "overlays", None) or [])[:3]:
            tick = ensure_sound(_ui_key_for(d), Path(work_dir))
            if tick:
                events.append((tick, at + float(o.get("t", 0.8) or 0.8), 0.4))
        # WHOOSH at every mid-story LAYOUT MOMENT start — the same
        # canvas-wins / Director-fills precedence + sanitizer the moment
        # RENDER path uses, so the swoosh lands where the screen actually
        # moves. (Cutaway spans aren't re-derived here — a moment the
        # render dropped for a cutaway costs at most one stray whoosh;
        # fail-soft, never worth re-resolving the image pool for.)
        _sdur = max(0.5, float(getattr(s, "video_t_end", 0.0) or 0.0)
                    - float(getattr(s, "video_t_start", 0.0) or 0.0))
        _mom_src = (getattr(s, "layout_moments", None)
                    or getattr(d, "layout_moments", None) or ())
        _mom_wins = [(float(_mts), float(_mte)) for (_g, _mts, _mte, _mtr)
                     in _sanitize_layout_moments(_mom_src, _sdur)]
        for (_mts, _mte) in _mom_wins:
            wh = ensure_sound("whoosh_air", Path(work_dir))
            if wh:                       # fail-soft: missing key → no event
                events.append((wh, at + _mts, 0.5))
        # IMPACT on each Director emphasis beat — the audio half of the
        # punch-in; the same sanitizer (moments blocked) keeps the hit
        # aligned with where the zoom actually renders. Gated by the SAME
        # kill switch as the visual punch — audio without the zoom would
        # read as a stray thump. Known gap (accepted, like the whoosh
        # note above): spot/pip/cutaway spans aren't available post-
        # stitch, so an image-rich story can rarely sound a hit where
        # the render dropped its zoom window.
        if _flag_on("KAIZER_V4_EMPHASIS_PUNCH"):
            for (_ea, _eb) in _sanitize_emphasis_windows(
                    getattr(d, "emphasis", []) or [], _sdur, blocked=_mom_wins):
                imp = ensure_sound(_impact_key(), Path(work_dir))
                if imp:
                    events.append((imp, at + _ea, 0.5))
    if not events:
        return
    events = events[:24]     # keep the mix sane + the ffmpeg cmd bounded
    # dominant mood's bed, ducked under the speech track — unless the
    # stories' Director bed_on majority vote switched the underscore off.
    bed_path = None
    bed_gain = 0.12
    if _bed_wanted(_dlist):
        try:
            dom = max(set(moods), key=moods.count) if moods else "news"
            pack = STYLES.get(dom)
            if pack is not None and pack.bed:
                bed_path = synth_bed(Path(work_dir), pack, stitched_dur)
                bed_gain = float(pack.bed_gain) * 0.7
        except Exception:
            bed_path = None
    else:
        print("[v4/v1_bridge] sound design: bed off (Director bed_on vote)",
              flush=True)
    cmd = [ffmpeg, "-y", "-v", "error", "-i", final_path]
    for p, _t, _g in events:
        cmd += ["-i", p]
    if bed_path:
        cmd += ["-i", bed_path]
    parts, mix_ins = [], "[0:a]"
    for k, (_p, t, g) in enumerate(events, start=1):
        ms = int(round(t * 1000))
        parts.append(f"[{k}:a]adelay={ms}|{ms},volume={g:.2f}[e{k}]")
        mix_ins += f"[e{k}]"
    n_mix = len(events) + 1
    if bed_path:
        bidx = len(events) + 1
        ms = int(round(intro_off * 1000))
        parts.append(f"[{bidx}:a]adelay={ms}|{ms},volume={bed_gain:.2f}[bd0]")
        parts.append("[bd0][0:a]sidechaincompress="
                     "threshold=0.05:ratio=10:attack=25:release=400[bd]")
        mix_ins += "[bd]"
        n_mix += 1
    parts.append(f"{mix_ins}amix=inputs={n_mix}:"
                 f"duration=first:dropout_transition=0:normalize=0[aout]")
    tmp_out = str(Path(final_path).with_name("_snddsg_" + Path(final_path).name))
    cmd += ["-filter_complex", ";".join(parts),
            "-map", "0:v", "-map", "[aout]", "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart", tmp_out]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=900)
        os.replace(tmp_out, final_path)
        print(f"[v4/v1_bridge] sound design: {len(events)} events"
              f"{' + ducked bed' if bed_path else ''} mixed", flush=True)
    except Exception as exc:
        print(f"[v4/v1_bridge] sound design failed (kept original): {exc}",
              flush=True)
        try:
            os.remove(tmp_out)
        except OSError:
            pass


def _edge_clamp_windows(wins, story_dur, *, edge: float = 0.75,
                        min_len: float = 1.0, with_path: bool = False):
    """Keep full-screen windows (gap/spotlight/PiP) away from story
    edges and kill sliver windows. Fixes the operator-reported 'layout
    blinks at transitions': a crossfade blending a FULL-SCREEN story
    boundary against the next story's FRAMED layout makes the frame
    ghost in/out; and sub-second fullscreen windows read as blinking.
    With the guard, every joint blends two identical framed layouts."""
    out = []
    for w in (wins or ()):
        if with_path:
            # Positional unpack (not `pth, a, b = w`): spotlight/PiP windows
            # may carry a focal-point tail (path, ts, te, ox, oy) — clamp
            # only the times and pass any extra slots through untouched.
            pth, a, b = w[0], float(w[1]), float(w[2])
        else:
            a, b = w
        a = max(float(a), edge)
        b = min(float(b), float(story_dur) - edge)
        if b - a >= min_len:
            out.append((pth, round(a, 3), round(b, 3), *tuple(w[3:]))
                       if with_path else (round(a, 3), round(b, 3)))
    return out


def _sanitize_emphasis_windows(times, story_dur, *, blocked=(), cap=2,
                               edge: float = 0.75,
                               punch_dur: float = 0.45) -> list[tuple[float, float]]:
    """Director ``emphasis`` timestamps → render-ready punch-in windows
    ``[(ts, te), …]`` (each a ~0.45s hard-cut 1.06x zoom of the main
    source). Rules: the window is CLAMPED >= ``edge`` off both story
    edges (a full-canvas pop touching a stitch joint ghosts against the
    crossfade — same reason as ``_edge_clamp_windows``); windows
    overlapping any ``blocked`` (start, end) span (layout moments,
    cutaways, spotlights, PiPs — full-screen states that own their
    stretch) or an earlier accepted punch are DROPPED; at most ``cap``
    survive. Stories too short to host a clamped punch return []. Pure
    (no ffmpeg, no I/O) — unit-tested in tests/test_director_av.py."""
    try:
        dur = float(story_dur or 0.0)
    except (TypeError, ValueError):
        return []
    if dur < 2 * edge + punch_dur:
        return []
    blk: list[tuple[float, float]] = []
    for w in (blocked or ()):
        try:
            blk.append((float(w[0]), float(w[1])))
        except (TypeError, ValueError, IndexError):
            continue
    out: list[tuple[float, float]] = []
    for t in (times or []):
        if len(out) >= cap:
            break
        try:
            ts = float(t)
        except (TypeError, ValueError):
            continue
        ts = max(edge, min(ts, dur - edge - punch_dur))
        te = ts + punch_dur
        if any(b > ts and a < te for (a, b) in blk):
            continue                    # a full-screen state owns that window
        if any(b > ts and a < te for (a, b) in out):
            continue                    # two beats too close — first wins
        out.append((round(ts, 3), round(te, 3)))
    return out


def _emphasis_fingerprint(windows) -> str:
    """Conditional ``|emph:`` cache tag for the punch-in windows —
    appended to the effects ingredient ONLY when windows survived
    sanitize, so every emphasis-free story keeps its exact legacy hash
    (the CACHE RULE: conditional |tag:, never a renderer-version bump)."""
    if not windows:
        return ""
    return "|emph:" + ",".join(f"{a:.2f}-{b:.2f}" for (a, b) in windows)


def _sanitize_layout_moments(moments, story_dur: float,
                             cutaways: tuple = ()) -> list[tuple]:
    """Validate mid-story layout windows into render-ready specs
    ``[(StoryGeometry, ts, te, transition), …]``. A moment must resolve
    to a VIDEO-CENTRIC designed layout (no picture surface — image
    moments use the per-image spotlight instead), sit >=0.75s off the
    story edges, run >=2.5s, not overlap an earlier moment, and not
    intersect a reference-video cutaway (explicit content wins). Max 2
    per story. Anything invalid degrades to nothing — never to a
    broken frame."""
    def _a(obj, name, default=None):
        return (obj.get(name, default) if isinstance(obj, dict)
                else getattr(obj, name, default))

    try:
        from pipeline_v4.layout_library import story_geometry
    except Exception:
        return []
    edge, min_len = 0.75, 2.5
    # PREMIUM pacing: the moment budget scales with the story — a long
    # single-story video is a presentation ARC, not a static frame
    # (~one switch every ~25s, min 2, max 8).
    cap = max(2, min(8, int(float(story_dur or 0.0) // 25)))
    cut_wins = [(float(a), float(b)) for (_p, a, b, _am, _tr)
                in (cutaways or ())]
    out: list[tuple] = []
    ordered = sorted((moments or ()), key=lambda m: float(_a(m, "t", 0.0) or 0.0))
    for m in ordered:
        if len(out) >= cap:
            break
        key = str(_a(m, "layout", "") or "").strip().lower()
        geom = story_geometry(key) if key else None
        if geom is None or geom.picture or geom.pips:
            continue
        try:
            ts = max(edge, float(_a(m, "t", 0.0) or 0.0))
            te = min(float(story_dur) - edge,
                     ts + max(0.0, float(_a(m, "dur", 0.0) or 0.0)))
        except (TypeError, ValueError):
            continue
        if te - ts < min_len:
            continue
        if any(b > ts and a < te for (a, b) in cut_wins):
            continue                      # cutaway owns that window
        if out and ts < out[-1][2] + 0.5:
            continue                      # overlaps/crowds the previous one
        tr = str(_a(m, "transition", "push") or "push").strip().lower()
        out.append((geom, round(ts, 3), round(te, 3),
                    tr if tr in ("push", "cut") else "push"))
    return out


# ── Director-window helpers (category graphics + karaoke captions) ───

def _story_words_sidecar(work_dir, story_index: int) -> list:
    """Story-relative word timestamps from the job's story_words.json
    sidecar ({"<story_index>": [{w,s,e}, ...]}). [] on any failure."""
    try:
        p = Path(work_dir) / "story_words.json"
        if not p.is_file():
            return []
        data = json.loads(p.read_text(encoding="utf-8"))
        words = data.get(str(int(story_index))) or []
        return words if isinstance(words, list) else []
    except Exception:
        return []


def _karaoke_params(variant_id: str) -> tuple[str, tuple]:
    """style + hilite color for a karaoke typography variant id."""
    try:
        from pipeline_v4.typography import HILITE, VARIANTS
        row = next((v for v in VARIANTS
                    if v["id"] == (variant_id or "").strip().lower()), None)
        kw = (row or {}).get("kwargs", {})
        return kw.get("style", "hilite"), tuple(kw.get("hilite", HILITE))
    except Exception:
        return "hilite", (255, 214, 0, 255)


# ── Name-strap (polish A) ─────────────────────────────────────────────

NAME_STRAP_H = 44   # fixed height so the overlay y is deterministic


def _render_name_strap_png(
    *,
    text: str,
    font_path: Optional[str],
    out_path: str,
    max_w: int = 700,
) -> Optional[str]:
    """Small broadcast strap naming the on-screen image's subject —
    dark translucent plate, red accent bar, white label text (PIL, same
    pattern as ``_render_logo_only_bug``; the language font renders
    Telugu/Hindi correctly). Width follows the measured text, capped at
    ``max_w`` with an ellipsis. Returns out_path, or None when the text
    is empty / rendering fails (strap is a nicety, never a blocker)."""
    label = " ".join((text or "").split())
    if not label:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont
        try:
            font = (ImageFont.truetype(font_path, 24)
                    if font_path and os.path.isfile(font_path)
                    else ImageFont.load_default())
        except Exception:
            font = ImageFont.load_default()
        probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))

        def _w(t: str) -> int:
            box = probe.textbbox((0, 0), t, font=font)
            return box[2] - box[0]

        pad_x, bar_w = 14, 6
        while label and _w(label + "…") > (max_w - 2 * pad_x - bar_w) and len(label) > 4:
            label = label[:-2].rstrip()
            if _w(label + "…") <= (max_w - 2 * pad_x - bar_w):
                label += "…"
                break
        tw = _w(label)
        w = min(max_w, tw + 2 * pad_x + bar_w)
        img = Image.new("RGBA", (w, NAME_STRAP_H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, w - 1, NAME_STRAP_H - 1], fill=(10, 10, 12, 200))
        d.rectangle([0, 0, bar_w - 1, NAME_STRAP_H - 1], fill=(193, 18, 18, 255))
        box = probe.textbbox((0, 0), label, font=font)
        ty = (NAME_STRAP_H - (box[3] - box[1])) // 2 - box[1]
        d.text((bar_w + pad_x, ty), label, font=font, fill=(255, 255, 255, 255))
        img.save(out_path, "PNG")
        return out_path
    except Exception as exc:
        print(f"[v4/v1_bridge] name-strap render failed ({exc}) — skipping",
              flush=True)
        return None


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


# ── Story-to-story transitions (native motion, Unit D) ───────────────

# xfade transitions vetted for broadcast use — anything else falls back
# to a hard cut rather than risking a garish wipe on a paid render.
SAFE_STORY_TRANSITIONS = {
    "fade", "fadeblack", "wipeleft", "wiperight",
    "slideleft", "slideright", "circleopen",
}


def _probe_clip_duration(path: str) -> Optional[float]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=60).stdout.strip()
        d = float(out)
        return d if d > 0 else None
    except Exception:
        return None


def build_xfade_stitch_graph(
    durations: list[float], *, transition="fade", fade_d: float = 0.5,
) -> tuple[str, str, str]:
    """Pure builder for the N-clip xfade + acrossfade stitch graph.
    Returns ``(filter_complex, video_label, audio_label)``. Offsets are
    cumulative: each transition starts ``fade_d`` before the end of the
    material stitched so far (standard xfade arithmetic).

    ``transition`` is a single name OR a per-joint list (the Director's
    per-story choices — joint i uses transition[i-1]); names resolve via
    xfade_arg so the 20 custom expressions work here too."""
    from pipeline_v4.trailer_styles import xfade_arg
    n = len(durations)
    if n < 2:
        raise ValueError("xfade stitch needs >= 2 clips")
    trans_list = (list(transition) if isinstance(transition, (list, tuple))
                  else [str(transition)])
    if not trans_list:
        trans_list = ["fade"]
    fc: list[str] = []
    v_cur, a_cur = "[0:v]", "[0:a]"
    total = durations[0]
    for i in range(1, n):
        off = max(0.0, total - fade_d)
        v_out = f"[v{i}]" if i < n - 1 else "[vout]"
        a_out = f"[a{i}]" if i < n - 1 else "[aout]"
        t_name = trans_list[(i - 1) % len(trans_list)]
        fc.append(
            f"{v_cur}[{i}:v]xfade={xfade_arg(t_name)}:"
            f"duration={fade_d:.3f}:offset={off:.3f}{v_out}"
        )
        fc.append(
            f"{a_cur}[{i}:a]acrossfade=d={fade_d:.3f}{a_out}"
        )
        v_cur, a_cur = v_out, a_out
        total = off + fade_d + (durations[i] - fade_d)
    return ";".join(fc), "vout", "aout"


def _stitch_with_transitions(
    paths: list[str], out_path: str, *, transition, fade_d: float = 0.5,
) -> Optional[str]:
    """Crossfade-stitch the composed story clips (one re-encode pass).
    Returns out_path on success, None on ANY failure — the caller then
    uses the legacy hard-cut stitcher, so robustness is >= the old path.
    ``transition``: single safe name (canvas) or a per-joint list from
    the Director (validated against the full transition catalog)."""
    try:
        if isinstance(transition, (list, tuple)):
            from pipeline_v4.trailer_styles import TRANSITION_CATALOG
            if len(paths) < 2 or not transition or any(
                    t not in TRANSITION_CATALOG for t in transition):
                return None
        elif len(paths) < 2 or transition not in SAFE_STORY_TRANSITIONS:
            return None
        durations = [_probe_clip_duration(p) for p in paths]
        if any(d is None for d in durations):
            return None
        # A transition can never consume more than a third of the
        # shortest neighbouring clip.
        d_eff = max(0.2, min(fade_d, min(durations) / 3.0))
        fc, v_lbl, a_lbl = build_xfade_stitch_graph(
            durations, transition=transition, fade_d=d_eff)
        cmd = [_ffmpeg_bin(), "-y", "-v", "error"]
        for p in paths:
            cmd += ["-i", p]
        cmd += [
            "-filter_complex", fc,
            "-map", f"[{v_lbl}]", "-map", f"[{a_lbl}]",
            *_enc_args(crf=20, preset_hint="medium"),
            "-pix_fmt", "yuv420p",
            "-r", "30", "-fps_mode", "cfr",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart",
            out_path,
        ]
        _run_ffmpeg(cmd, timeout=1800, log_label="stitch_xfade")
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            print(f"[v4/v1_bridge] stitched {len(paths)} stories with "
                  f"'{transition}' transitions ({d_eff:.2f}s)", flush=True)
            return out_path
        return None
    except Exception as exc:
        print(f"[v4/v1_bridge] transition stitch failed ({exc}) — "
              f"hard-cut fallback", flush=True)
        return None


# ── V4 bulletin story composer ────────────────────────────────────────

# FRAMESYNC HEARTBEAT for sparse video-derived branches (gap / emphasis /
# layout-moment). A branch that goes SILENT for a long stretch — trim
# emits nothing before its window; a window-only select emits nothing
# after its last window — starves the downstream overlay's framesync,
# which then BUFFERS every main frame until the branch speaks again:
# silent_seconds × fps × ~3.1MB/frame. On 30-90s news stories that dam is
# a survivable spike; on a 27-min podcast's 3-8min stories it is 8-20GB
# per compose and killed job 610 even single-threaded (bisect-proven:
# primer-only emphasis buffered 4.4GB in 15s; with this heartbeat 0.6GB
# flat). Passing one frame every 25 keeps framesync fed (~0.5s worst-case
# wait) while heavy per-branch filters (blur/scale) still skip ~96% of
# out-of-window frames — the perf goal the old trim= had. The overlay's
# enable= window gates visibility, so the trickle frames NEVER appear in
# the output: pixels are identical. n=0 passes (mod(0,25)=0), so this is
# a superset of the old eq(n,0) primer.
_FS_HEARTBEAT = "not(mod(n,25))"


def _build_story_filter_graph(
    *,
    canvas_w: int, canvas_h: int,
    main_inner_w: int, side_inner_w: int, tile_inner_h: int,
    main_outer_w: int, side_outer_w: int, tile_outer_h: int,
    main_x: int, side_x: int, tile_y: int,
    lt_y: int, ticker_y: int,
    border_colour: str, bg_color: str,
    lt_x_expr: str, ticker_speed_px_s: float,
    has_bg: bool, bg_input_idx,
    apply_ticker: bool, ticker_in_idx,
    has_bug: bool, bug_in_idx,
    has_wm: bool, wm_in_idx, watermark_xy: tuple[str, str] = ("", ""),
    bg_video_volume: float = 0.0,
    gap_windows: tuple = (),
    spotlight: tuple = (),
    pip: tuple = (),
    straps: tuple = (),
    story_fx_vf: str = "",
    dwin: tuple = (),          # Director windows: (idx, ts, te, x_expr, y_expr[, anim])
    video_spotlight: tuple = (),  # (idx, ts, te, audio_mode, trim_start) B-roll cutaways
    # LIVE LAYOUTS (per-story designed layout). Legacy defaults keep the
    # graph CHARACTER-IDENTICAL for every pre-layout canvas (byte-compat).
    main_outer_h=None,         # None → tile_outer_h (legacy shared height)
    side_outer_h=None,         # None → tile_outer_h
    side_y=None,               # None → tile_y (legacy shared row)
    main_border: int = V4_TILE_BORDER,   # 0 = borderless full-bleed video
    has_side: bool = True,               # False = no picture surface at all
    bg_mode: str = "legacy",   # legacy | blur_self | image
    bg_image_idx=None,         # looped-image input idx for bg_mode="image"
    moments: tuple = (),       # [(StoryGeometry, ts, te, "push"|"cut"), …]
    side_x_expr: str = "",     # TILE ENTRANCE MOTION: picture-tile overlay
    side_y_expr: str = "",     # x/y time exprs ('' = legacy constant x/y)
    emphasis: tuple = (),      # [(ts, te), …] Director emphasis punch-ins
) -> tuple[str, str, str]:
    """Build the story compose filter graph. Returns
    ``(filter_complex, out_label, audio_map)``.

    Extracted from ``_compose_v4_bulletin_story`` so tests can assert the
    no-gap graph stays CHARACTER-IDENTICAL to the legacy one (the
    byte-compat guarantee for every pre-engine canvas), and so the gap /
    spotlight branches have one auditable home.

    ``gap_windows`` — [(start, end), …] story-relative stretches with no
    scheduled image: the main video pops FULL-SCREEN there ("cut back to
    the anchor"), via a split + enable-gated overlay inserted BEFORE the
    lower-third/ticker/bug/watermark overlays so text always stays on
    top. Empty ⇒ the graph is exactly the legacy graph.

    ``spotlight`` — [(input_idx, start, end), …] full-screen image pops
    for the story's key moments: each is an extra looped-image input,
    cover-scaled to the canvas with a 0.3s alpha fade in/out, overlaid
    after the gap stage and before the text overlays (spec 3.13: the
    strap/ticker stay in their safe strip ON TOP of the spotlight).

    ``pip`` — [(input_idx, start, end), …] picture-in-picture (spec 3.8:
    show BOTH, big + small corner): for each window the main video pops
    FULL-SCREEN (its time range joins the gap enable) and the image sits
    as a 320x180 white-padded inset at the top-right (compose_pip_story
    geometry), below the channel bug. Insets draw after spotlight,
    before text.

    ``emphasis`` — [(start, end), …] Director emphasis PUNCH-INS: a
    ~1.06x zoomed copy of the (already graded) main source shown
    full-canvas for its window with a HARD CUT in/out — the broadcast
    punch on a key word. Same extra-split-consumer + opaque-cover +
    enable idiom as the spotlight; empty ⇒ the graph is exactly the
    legacy graph (byte-compat)."""
    # The main video fills the frame during image gaps AND during PiP
    # windows (PiP = big video + small image). Positional indexing — pip
    # entries may be legacy 3-tuples or focal 5-tuples (idx, ts, te, ox, oy).
    fs_windows = tuple(gap_windows) + tuple((w[1], w[2]) for w in pip)
    fs_windows = tuple(sorted(fs_windows))
    has_gaps = bool(fs_windows)
    # Per-tile geometry: legacy shares one row/height across both tiles;
    # a per-story designed layout gives each tile its own rect. Defaults
    # resolve to the legacy values so the emitted strings are identical.
    _mo_h = main_outer_h if main_outer_h is not None else tile_outer_h
    _mi_h = max(1, _mo_h - 2 * main_border)
    _so_h = side_outer_h if side_outer_h is not None else tile_outer_h
    _si_h = max(1, _so_h - 2 * V4_TILE_BORDER)
    _sd_y = side_y if side_y is not None else tile_y
    # Full-form effects prefix on the SOURCE video (before framing) so the
    # tile and the full-screen gap fallback carry the same look. Empty →
    # the graph stays CHARACTER-IDENTICAL to legacy (byte-compat guard).
    _fxp = (story_fx_vf + ",") if story_fx_vf else ""
    fc: list[str] = []
    # Mid-story layout moments: each needs its own copy of the source —
    # one for the moment's video tile, plus one for the blurred-echo
    # backdrop when the moment layout is NOT full-bleed.
    _mom_specs = []
    for (mg, m_ts, m_te, m_tr) in (moments or ()):
        _mom_specs.append({"geom": mg, "ts": float(m_ts), "te": float(m_te),
                           "tr": (m_tr or "push"),
                           "full": bool(getattr(mg, "video_borderless", False))})
    _mom_consumers = sum(1 + (0 if m["full"] else 1) for m in _mom_specs)
    has_emph = bool(emphasis)
    _n_split = (1 + (1 if has_gaps else 0)
                + (1 if bg_mode == "blur_self" else 0) + _mom_consumers
                + (1 if has_emph else 0))
    if _n_split > 1:
        # [0:v] feeds the framed tile + the full-screen fallback and/or
        # the blurred-echo backdrop and/or the moment branches and/or the
        # emphasis punch-in (graded once, before the split, so every
        # consumer carries the same look).
        _labels = ("[v_tile]" + ("[v_full]" if has_gaps else "")
                   + ("[v_bgsrc]" if bg_mode == "blur_self" else ""))
        for _mk, _m in enumerate(_mom_specs):
            _labels += f"[v_mom{_mk}]"
            if not _m["full"]:
                _labels += f"[v_mombg{_mk}]"
        if has_emph:
            # appended LAST so gap/blur/moment label order — and every
            # emphasis-free graph string — stays character-identical.
            _labels += "[v_emph]"
        fc.append(f"[0:v]{_fxp}split={_n_split}{_labels}")
        main_src = "[v_tile]"
        _fxp = ""            # applied once, before the split
    else:
        main_src = "[0:v]"
    _main_tail = (f",pad={main_outer_w}:{_mo_h}:"
                  f"{main_border}:{main_border}:color={border_colour}"
                  if main_border > 0 else "")
    fc.append(
        f"{main_src}{_fxp}scale={main_inner_w}:{_mi_h}:"
        f"force_original_aspect_ratio=increase,"
        f"crop={main_inner_w}:{_mi_h},setsar=1"
        f"{_main_tail}[main_v]")
    if has_side:
        fc.append(
            f"[1:v]scale={side_inner_w}:{_si_h}:"
            f"force_original_aspect_ratio=increase,"
            f"crop={side_inner_w}:{_si_h},setsar=1,"
            f"pad={side_outer_w}:{_so_h}:"
            f"{V4_TILE_BORDER}:{V4_TILE_BORDER}:color={border_colour}[side_v]")
    # Background layer: the story's own blurred echo (per-story layout),
    # a story-image wash, a looping bg video, or the flat colour.
    if bg_mode == "blur_self":
        fc.append(
            f"[v_bgsrc]scale={canvas_w}:{canvas_h}:"
            f"force_original_aspect_ratio=increase,"
            f"crop={canvas_w}:{canvas_h},setsar=1,"
            f"boxblur=luma_radius=24:luma_power=2,"
            f"eq=brightness=-0.08:saturation=1.06[bg]"
        )
    elif bg_mode == "image" and bg_image_idx is not None:
        fc.append(
            f"[{bg_image_idx}:v]scale={canvas_w}:{canvas_h}:"
            f"force_original_aspect_ratio=increase,"
            f"crop={canvas_w}:{canvas_h},setsar=1,fps=30,"
            f"boxblur=luma_radius=18:luma_power=2,"
            f"eq=brightness=-0.10:saturation=1.05[bg]"
        )
    elif bg_mode == "plate" and bg_image_idx is not None:
        # THEME backdrop plate: the design itself — clean, no blur.
        fc.append(
            f"[{bg_image_idx}:v]scale={canvas_w}:{canvas_h}:"
            f"force_original_aspect_ratio=increase,"
            f"crop={canvas_w}:{canvas_h},setsar=1,fps=30[bg]"
        )
    elif has_bg:
        fc.append(
            f"[{bg_input_idx}:v]scale={canvas_w}:{canvas_h}:"
            f"force_original_aspect_ratio=increase,"
            f"crop={canvas_w}:{canvas_h},setsar=1,fps=30[bg]"
        )
    else:
        fc.append(
            f"color=c={bg_color}:"
            f"s={canvas_w}x{canvas_h}:r=30[bg]"
        )
    fc.append(f"[bg][main_v]overlay=x={main_x}:y={tile_y}:shortest=1[stage_m]")
    if has_side:
        if side_x_expr or side_y_expr:
            # TILE ENTRANCE MOTION: the picture tile slides in from its
            # nearest screen edge at story start (0.5s cubic ease-out —
            # the lower-third / layout-moment-push idiom). Only ever
            # non-empty for NEW-feature stories (per-story layout or
            # themed canvas), so every legacy graph stays
            # CHARACTER-IDENTICAL (test_carousel_gaps snapshot).
            _sx = f"'{side_x_expr}'" if side_x_expr else str(side_x)
            _sy = f"'{side_y_expr}'" if side_y_expr else str(_sd_y)
            fc.append(f"[stage_m][side_v]overlay=x={_sx}:y={_sy}[stage_top]")
        else:
            fc.append(f"[stage_m][side_v]overlay=x={side_x}:y={_sd_y}[stage_top]")
        cursor = "stage_top"
    else:
        cursor = "stage_m"
    if has_gaps:
        # Full-screen anchor video during image gaps (and PiP windows).
        # Sum of between() = boolean OR (the carousel renderer uses the
        # same idiom). ``select`` drops every frame OUTSIDE the windows at
        # the branch head (PTS preserved, so the enable= windows match
        # exactly): without it this branch carries the WHOLE story through
        # scale, and ffmpeg 8's queued scheduler buffers it — job 600's
        # 131s story OOM'd a single compose with 25GB RAM free.
        enable = "+".join(
            f"between(t,{a:.3f},{b:.3f})" for a, b in fs_windows
        )
        fc += [
            # Heartbeat keeps framesync fed between gap windows (see
            # _FS_HEARTBEAT; supersedes the old eq(n,0) primer).
            f"[v_full]select='{_FS_HEARTBEAT}+{enable}',scale={canvas_w}:{canvas_h}:"
            f"force_original_aspect_ratio=increase,"
            f"crop={canvas_w}:{canvas_h},setsar=1[vfs]",
            f"[{cursor}][vfs]overlay=x=0:y=0:enable='{enable}'[stage_gap]",
        ]
        cursor = "stage_gap"
    for k, _sw in enumerate(spotlight):
        # Legacy 3-tuples (idx, ts, te) or focal 5-tuples (…, ox, oy):
        # 50/50 emits the bare legacy crop via _focal_crop, so 3-tuple and
        # default-offset graphs stay CHARACTER-identical (byte-compat).
        sp_idx, sp_ts, sp_te = _sw[0], float(_sw[1]), float(_sw[2])
        sp_ox, sp_oy = ((float(_sw[3]), float(_sw[4])) if len(_sw) > 4
                        else (50.0, 50.0))
        fade = V4_SPOTLIGHT_FADE_S
        if fade > 0.0:
            # Legacy alpha cross-dissolve (ghosts against the framed base —
            # kept only if a >0 fade is explicitly configured).
            fc += [
                f"[{sp_idx}:v]scale={canvas_w}:{canvas_h}:"
                f"force_original_aspect_ratio=increase,"
                f"{_focal_crop(canvas_w, canvas_h, sp_ox, sp_oy)},setsar=1,format=yuva420p,"
                f"fade=t=in:st={sp_ts:.3f}:d={fade:.3f}:alpha=1,"
                f"fade=t=out:st={max(sp_ts, sp_te - fade):.3f}:d={fade:.3f}:alpha=1[sp{k}]",
                f"[{cursor}][sp{k}]overlay=x=0:y=0:"
                f"enable='between(t,{sp_ts:.3f},{sp_te:.3f})'[stage_sp{k}]",
            ]
        else:
            # Clean hard cut: the full-screen image is fully opaque for its
            # whole window, so the framed base can never bleed through. No
            # alpha channel needed (yuv420p is cheaper than yuva420p).
            fc += [
                f"[{sp_idx}:v]scale={canvas_w}:{canvas_h}:"
                f"force_original_aspect_ratio=increase,"
                f"{_focal_crop(canvas_w, canvas_h, sp_ox, sp_oy)},setsar=1,format=yuv420p[sp{k}]",
                f"[{cursor}][sp{k}]overlay=x=0:y=0:"
                f"enable='between(t,{sp_ts:.3f},{sp_te:.3f})'[stage_sp{k}]",
            ]
        cursor = f"stage_sp{k}"
    for k, _m in enumerate(_mom_specs):
        # MID-STORY LAYOUT MOMENT: the screen switches to a different
        # designed arrangement for [ts,te] — the "live direction" feel.
        # The branch is an OPAQUE full-canvas unit (video full-bleed, or
        # a framed card floating on the clip's blurred echo) overlaid on
        # the base stage; narration audio is untouched. "push" slides the
        # whole unit in from the right and back out (0.4s cubic — the
        # same eased-entrance idiom as the lower-third; no alpha
        # dissolve, so no double-exposure ghosting); "cut" is the
        # broadcast hard switch.
        m_ts, m_te, mg = _m["ts"], _m["te"], _m["geom"]
        # Heartbeat-windowed select (see _FS_HEARTBEAT). The old
        # trim={ts}:{te} was a FRAMESYNC DAM: the branch emitted NOTHING
        # until decode reached ts, so the moment overlay buffered every
        # main frame from t=0..ts (~8GB for a t=90 moment on a long
        # story) — the job-610 OOM that killed long-source bulletins even
        # single-threaded. The heartbeat keeps framesync fed before AND
        # after the window; the between() window still carries the full
        # frame rate, so heavy filters (blur) pay only for the moment's
        # own seconds — the perf goal the trim had. select keeps original
        # PTS (the old setpts=PTS-STARTPTS+ts was an identity re-stamp)
        # and the enable window below hides every out-of-window frame.
        _mtrim = f"select='{_FS_HEARTBEAT}+between(t,{m_ts:.3f},{m_te:.3f})',"
        if _m["full"]:
            fc.append(
                f"[v_mom{k}]{_mtrim}scale={canvas_w}:{canvas_h}:"
                f"force_original_aspect_ratio=increase,"
                f"crop={canvas_w}:{canvas_h},setsar=1,format=yuv420p[mom{k}]")
        else:
            _mvx, _mvy, _mvw, _mvh = mg.video
            _mo_w = max(2, int(round(_mvw / 100.0 * canvas_w)))
            _mo_hh = max(2, int(round(_mvh / 100.0 * canvas_h)))
            _mi_w = max(1, _mo_w - 2 * V4_TILE_BORDER)
            _mi_hh = max(1, _mo_hh - 2 * V4_TILE_BORDER)
            _mx = int(round(_mvx / 100.0 * canvas_w))
            _my = int(round(_mvy / 100.0 * canvas_h))
            fc += [
                f"[v_mombg{k}]{_mtrim}scale={canvas_w}:{canvas_h}:"
                f"force_original_aspect_ratio=increase,"
                f"crop={canvas_w}:{canvas_h},setsar=1,"
                f"boxblur=luma_radius=24:luma_power=2,"
                f"eq=brightness=-0.08:saturation=1.06[mombg{k}]",
                f"[v_mom{k}]{_mtrim}scale={_mi_w}:{_mi_hh}:"
                f"force_original_aspect_ratio=increase,"
                f"crop={_mi_w}:{_mi_hh},setsar=1,"
                f"pad={_mo_w}:{_mo_hh}:{V4_TILE_BORDER}:{V4_TILE_BORDER}:"
                f"color={border_colour}[momtile{k}]",
                f"[mombg{k}][momtile{k}]overlay=x={_mx}:y={_my},"
                f"format=yuv420p[mom{k}]",
            ]
        if _m["tr"] == "push" and (m_te - m_ts) > 1.0:
            _d = 0.4
            _mx_expr = (
                f"if(lt(t\\,{m_ts + _d:.3f})\\,"
                f"{canvas_w}*pow(1-(t-{m_ts:.3f})/{_d}\\,3)\\,"
                f"if(gt(t\\,{m_te - _d:.3f})\\,"
                f"{canvas_w}*(1-pow(1-(t-{m_te - _d:.3f})/{_d}\\,3))\\,0))"
            )
        else:
            _mx_expr = "0"
        fc.append(
            f"[{cursor}][mom{k}]overlay=x='{_mx_expr}':y=0:"
            f"enable='between(t,{m_ts:.3f},{m_te:.3f})'[stage_mom{k}]")
        cursor = f"stage_mom{k}"
    for k, (vs_idx, vs_ts, vs_te, vs_am, vs_tr) in enumerate(video_spotlight):
        # Full-screen reference-video (B-roll) CUTAWAY. Opaque hard cut
        # (same clean-switch principle as the image spotlight) — the framed
        # base is fully covered for [ts,te], no ghosting. setpts shifts the
        # looped clip so it plays from its head at story-time ts; the
        # upstream -stream_loop -1 prevents any -shortest truncation.
        # Broadcast furniture (pip/strap/captions/lower-third/ticker/bug)
        # is drawn AFTER this, so it stays visible over the clip.
        fc += [
            f"[{vs_idx}:v]scale={canvas_w}:{canvas_h}:"
            f"force_original_aspect_ratio=increase,"
            f"crop={canvas_w}:{canvas_h},setsar=1,fps=30,"
            f"setpts=PTS-STARTPTS+{vs_ts:.3f}/TB,format=yuv420p[vsp{k}]",
            f"[{cursor}][vsp{k}]overlay=x=0:y=0:"
            f"enable='between(t,{vs_ts:.3f},{vs_te:.3f})'[stage_vsp{k}]",
        ]
        cursor = f"stage_vsp{k}"
    for k, _pw in enumerate(pip):
        # compose_pip_story geometry: 320x180 content + 12px white pad,
        # top-right, below the channel-bug strip. A 16:9 inset from a
        # portrait source is the classic beheaded-face case — the focal
        # crop (5-tuple tail; 3-tuples imply centered) keeps the face in.
        pp_idx, pp_ts, pp_te = _pw[0], float(_pw[1]), float(_pw[2])
        pp_ox, pp_oy = ((float(_pw[3]), float(_pw[4])) if len(_pw) > 4
                        else (50.0, 50.0))
        fc += [
            f"[{pp_idx}:v]scale=320:180:"
            f"force_original_aspect_ratio=increase,"
            f"{_focal_crop(320, 180, pp_ox, pp_oy)},setsar=1,"
            f"pad=344:204:12:12:color=white[pip{k}]",
            f"[{cursor}][pip{k}]overlay="
            f"x=W-w-{V4_SIDE_MARGIN}:y={tile_y + 90}:"
            f"enable='between(t,{pp_ts:.3f},{pp_te:.3f})'[stage_pip{k}]",
        ]
        cursor = f"stage_pip{k}"
    if has_emph:
        # EMPHASIS PUNCH-IN: a ~1.06x zoomed copy of the main source pops
        # full-canvas for ~0.45s on each Director emphasis beat. HARD CUT
        # (the spotlight idiom: fully opaque cover, no alpha fade — so the
        # framed base can never bleed through / ghost). Drawn BEFORE the
        # straps/graphics/LT/ticker so broadcast text stays on top. The
        # windows were sanitized upstream to never overlap a moment /
        # cutaway / spotlight, so their position among those stages is
        # inert; sum of between() = boolean OR (the gap idiom).
        _ez_w = max(2, int(round(canvas_w * 1.06)))
        _ez_h = max(2, int(round(canvas_h * 1.06)))
        _e_en = "+".join(f"between(t,{a:.3f},{b:.3f})" for (a, b) in emphasis)
        # Same select-at-the-head memory guard as the gap branch: the
        # punch-in copy only ever shows ~0.45s windows — never let the
        # whole story queue through the 1.06x scale.
        fc += [
            f"[v_emph]select='{_FS_HEARTBEAT}+{_e_en}',scale={_ez_w}:{_ez_h}:"
            f"force_original_aspect_ratio=increase,"
            f"crop={canvas_w}:{canvas_h},setsar=1,format=yuv420p[epch]",
            f"[{cursor}][epch]overlay=x=0:y=0:enable='{_e_en}'[stage_emph]",
        ]
        cursor = "stage_emph"
    for k, (ns_idx, ns_ts, ns_te) in enumerate(straps):
        # Name-strap (polish A): pre-rendered PNG at final size, sitting
        # just above the lower-third inside the safe zone, visible for
        # its image's window. Drawn before the LT so the LT slide-in
        # animation stays on top of everything.
        fc.append(
            f"[{cursor}][{ns_idx}:v]overlay="
            f"x={main_x}:y={lt_y - NAME_STRAP_H - 8}:"
            f"enable='between(t,{ns_ts:.3f},{ns_te:.3f})'[stage_ns{k}]"
        )
        cursor = f"stage_ns{k}"
    for k, _dw in enumerate(dwin):
        # DIRECTOR windows: category graphics (full-canvas positioned
        # PNGs at 0:0) and karaoke caption states (element PNGs with
        # x/y exprs). Entrance per catalog family (6th tuple slot):
        # slide_up/slide_down ride an overlay-y TIME EXPRESSION (the
        # lower-third / layout-moment-push cubic ease-out — no alpha,
        # no ghosting); everything else keeps the legacy 0.25s alpha
        # fade (captions always do). Before the LT so broadcast
        # furniture stays on top.
        dw_idx, dw_ts, dw_te, dw_x, dw_y = _dw[:5]
        dw_anim = _dw[5] if len(_dw) > 5 else "fade"
        _ad = 0.45
        if dw_anim in ("slide_up", "slide_down") and (dw_te - dw_ts) > _ad + 0.15:
            _off = canvas_h if dw_anim == "slide_up" else -canvas_h
            _y_expr = (f"if(lt(t\\,{dw_ts + _ad:.3f})\\,"
                       f"{_off}*pow(1-(t-{dw_ts:.3f})/{_ad}\\,3)\\,0)")
            fc += [
                f"[{dw_idx}:v]format=rgba[dw{k}]",
                f"[{cursor}][dw{k}]overlay=x={dw_x}:y='{_y_expr}':"
                f"enable='between(t,{dw_ts:.3f},{dw_te:.3f})'[stage_dw{k}]",
            ]
        else:
            fade = 0.25
            fc += [
                f"[{dw_idx}:v]format=rgba,"
                f"fade=t=in:st={dw_ts:.3f}:d={fade:.3f}:alpha=1[dw{k}]",
                f"[{cursor}][dw{k}]overlay=x={dw_x}:y={dw_y}:"
                f"enable='between(t,{dw_ts:.3f},{dw_te:.3f})'[stage_dw{k}]",
            ]
        cursor = f"stage_dw{k}"
    fc.append(
        f"[{cursor}][2:v]overlay=x='{lt_x_expr}':y={lt_y}:format=auto[stage_lt]"
    )
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
        wx, wy = watermark_xy
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
    # Reference-video cutaways whose audio should play (audio_mode="duck").
    _duck = [(idx, ts, te) for (idx, ts, te, am, tr) in (video_spotlight or ())
             if am == "duck"]
    audio_map = "0:a?"
    if _duck or (has_bg and bg_video_volume > 0.0):
        _abits: list[str] = []
        _mix: list[str] = []
        if _duck:
            # Anchor ducks to 25% whenever a cutaway clip's audio is playing.
            _betw = "+".join(f"between(t,{ts:.3f},{te:.3f})" for (_i, ts, te) in _duck)
            _abits.append(f"[0:a]volume='if(gt({_betw},0),0.25,1.0)':eval=frame[a0]")
        else:
            _abits.append("[0:a]volume=1.0[a0]")
        _mix.append("[a0]")
        if has_bg and bg_video_volume > 0.0:
            _abits.append(f"[{bg_input_idx}:a]volume={bg_video_volume:.3f}[abg]")
            _mix.append("[abg]")
        for k, (idx, ts, te) in enumerate(_duck):
            # Clip audio gated to its window (silent outside, full inside).
            _abits.append(
                f"[{idx}:a]volume='if(between(t,{ts:.3f},{te:.3f}),1.0,0.0)':"
                f"eval=frame[avc{k}]")
            _mix.append(f"[avc{k}]")
        fc.append(";".join(_abits) + ";" + "".join(_mix)
                  + f"amix=inputs={len(_mix)}:duration=first:"
                  f"dropout_transition=0:normalize=0[aout]")
        audio_map = "[aout]"
    return ";".join(fc), out_label, audio_map


def _adaptive_filter_threads(base_ft: int, story_dur: float,
                             *, long_s: float = 300.0,
                             mid_s: float = 150.0) -> int:
    """Filtergraph thread count scaled DOWN for long stories.

    ffmpeg's threaded filtergraph scheduler queues frames PER thread, so a
    branch-heavy story graph's peak RAM grows with the thread count. A news
    story (30-90s) fits fine at the full cap; a podcast story (3-8 min) at the
    same cap OOMs when two compose 2-wide (job 609: 2 × 8-min × 5 threads =
    ~16GB), forcing the SLOW single-thread rescue. Dropping long stories to a
    lower count lets them fit in RAM 2-wide and compose at full speed. Thread
    count is a SCHEDULING knob only — the encoded frames are byte-identical
    regardless — so no effect/quality/feature is touched. The OOM rung still
    backstops anything heavier than this proactive cap.

    Caps relaxed (long 2→3, mid 3→4) after the framesync-heartbeat fix
    (9d44cb2): the multi-GB peaks were the DAM, not thread queues — dam-free
    long-story composes measured 1.6-3.3GB at 2 threads (job 610 re-render),
    so an extra thread costs a few hundred MB of queues and buys real compose
    speed; 2-wide worst case stays far under free RAM."""
    try:
        d = float(story_dur or 0.0)
    except (TypeError, ValueError):
        d = 0.0
    if d >= long_s:
        return max(1, min(base_ft, 3))
    if d >= mid_s:
        return max(1, min(base_ft, 4))
    return max(1, base_ft)


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
    gap_windows: tuple = (),               # image-less stretches → video full-screen
    spotlight_windows: tuple = (),         # [(image_path, ts, te[, ox, oy])] full-screen pops
    pip_windows: tuple = (),               # [(image_path, ts, te[, ox, oy])] video-big + image-inset
    name_strap_windows: tuple = (),        # [(strap_png_path, ts, te)] subject straps
    story_fx_vf: str = "",                 # full-form effects chain ('' = legacy)
    directive_windows: tuple = (),         # [(png, ts, te, kind[, anim])] kind: full|caption
    video_spotlight_windows: tuple = (),   # [(video_path, ts, te, audio_mode, trim_start)] B-roll cutaways
    story_layout=None,                     # layout_library.StoryGeometry — per-story designed layout
    bg_image_path: Optional[str] = None,   # story-image backdrop for story_layout.bg == "bg_image"
    layout_moments: tuple = (),            # sanitized [(StoryGeometry, ts, te, transition)]
    emphasis_windows: tuple = (),          # sanitized [(ts, te)] Director punch-ins
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

    # LIVE LAYOUTS: a per-story designed layout wins over both the
    # canvas pcts and the V4 defaults — each tile gets its OWN rect
    # (legacy shares one row/height). The bg can come from the layout:
    # the job's studio set, a story-image wash, or the clip's own
    # blurred echo. Fail-soft on missing assets (wash → echo).
    _sl = story_layout
    _main_border = V4_TILE_BORDER
    _main_outer_h = _side_outer_h = _side_y = None      # legacy: shared row
    _has_side = True
    _sl_bg = "inherit"
    if _sl is not None:
        vx, vy, vw, vh = _sl.video
        main_x = _pct_to_px(vx, canvas_w)
        tile_y = _pct_to_px(vy, canvas_h)
        main_outer_w = _pct_to_px(vw, canvas_w)
        _main_outer_h = _pct_to_px(vh, canvas_h)
        if getattr(_sl, "video_borderless", False):
            _main_border = 0
        _pic = _sl.picture or ((_sl.pips[0].x, _sl.pips[0].y,
                                _sl.pips[0].w, _sl.pips[0].h)
                               if _sl.pips else None)
        if _pic:
            side_x = _pct_to_px(_pic[0], canvas_w)
            _side_y = _pct_to_px(_pic[1], canvas_h)
            side_outer_w = _pct_to_px(_pic[2], canvas_w)
            _side_outer_h = _pct_to_px(_pic[3], canvas_h)
        else:
            _has_side = False
        _sl_bg = getattr(_sl, "bg", "inherit") or "inherit"
        if _sl_bg == "bg_video" and not (bg_video_path
                                         and os.path.isfile(bg_video_path)):
            _sl_bg = "blur_self"    # layout wants a set the job doesn't have
        if _sl_bg == "bg_image" and not (bg_image_path
                                         and os.path.isfile(bg_image_path)):
            _sl_bg = "blur_self"
        if _sl_bg in ("blur_self", "bg_image"):
            # this story's backdrop comes from the layout — drop the job's
            # studio bg input entirely (video AND its audio mix).
            bg_video_path = None
            bg_video_volume = 0.0

    # THEME PACKS: the theme's backdrop plate skins this story whenever
    # nothing more specific owns the backdrop — the job's studio video
    # wins, a story-layout bg (blurred echo / image wash) wins, the
    # theme plate comes next, flat colour last. Fail-soft everywhere.
    _theme = None
    _theme_plate = None
    _theme_key = str(getattr(layout, "theme", "") or "").strip().lower()
    if _theme_key:
        try:
            from pipeline_v4.theme_packs import get_theme, ensure_backdrop
            _theme = get_theme(_theme_key)
            if (_theme is not None
                    and not (_sl is not None and _sl_bg in ("blur_self", "bg_image"))
                    and not (bg_video_path and os.path.isfile(bg_video_path))):
                _theme_plate = ensure_backdrop(_theme.key, work_dir)
        except Exception:
            _theme = _theme_plate = None

    main_inner_w = max(1, main_outer_w - 2 * _main_border)
    side_inner_w = max(1, side_outer_w - 2 * V4_TILE_BORDER)
    tile_inner_h = max(1, tile_outer_h - 2 * V4_TILE_BORDER)
    lt_y     = canvas_h - V4_LT_H - V4_TICKER_H
    ticker_y = canvas_h - V4_TICKER_H

    # TILE ENTRANCE MOTION (KAIZER_V4_LAYOUT_MOTION, default on): the
    # picture tile slides in from its NEAREST screen edge over 0.5s
    # (cubic ease-out) at story start. Gated to NEW-feature stories
    # ONLY — a per-story designed layout or a themed canvas — so every
    # legacy canvas keeps its constant x/y overlay string (byte-compat).
    _side_x_expr = _side_y_expr = ""
    if (_has_side and (_sl is not None or _theme is not None)
            and _flag_on("KAIZER_V4_LAYOUT_MOTION")):
        _tm_h = _side_outer_h if _side_outer_h is not None else tile_outer_h
        _tm_y = _side_y if _side_y is not None else tile_y
        _tm_T = 0.5
        _edge = min([(side_x, "left"),
                     (canvas_w - (side_x + side_outer_w), "right"),
                     (_tm_y, "top"),
                     (canvas_h - (_tm_y + _tm_h), "bottom")],
                    key=lambda e: e[0])[1]
        if _edge == "left":
            _side_x_expr = (f"if(lt(t\\,{_tm_T})\\,{side_x}-"
                            f"{side_x + side_outer_w}*pow(1-t/{_tm_T}\\,3)\\,{side_x})")
        elif _edge == "right":
            _side_x_expr = (f"if(lt(t\\,{_tm_T})\\,{side_x}+"
                            f"{canvas_w - side_x}*pow(1-t/{_tm_T}\\,3)\\,{side_x})")
        elif _edge == "top":
            _side_y_expr = (f"if(lt(t\\,{_tm_T})\\,{_tm_y}-"
                            f"{_tm_y + _tm_h}*pow(1-t/{_tm_T}\\,3)\\,{_tm_y})")
        else:
            _side_y_expr = (f"if(lt(t\\,{_tm_T})\\,{_tm_y}+"
                            f"{canvas_h - _tm_y}*pow(1-t/{_tm_T}\\,3)\\,{_tm_y})")

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
    # Story-image backdrop (per-story layout bg — only ever present for
    # NEW layout stories, so legacy input indices stay untouched).
    _has_bgimg = bool(_sl is not None and _sl_bg == "bg_image"
                      and bg_image_path)
    _bgimg_idx = None
    if _has_bgimg:
        cmd += ["-loop", "1", "-i", bg_image_path]
        _bgimg_idx = _next_idx
        _next_idx += 1
    # Theme backdrop plate — only ever present for themed jobs, so
    # legacy input indices stay untouched (same discipline as above).
    _has_plate = bool(_theme_plate and not _has_bgimg)
    if _has_plate:
        cmd += ["-loop", "1", "-i", _theme_plate]
        _bgimg_idx = _next_idx
        _next_idx += 1
    # Spotlight / PiP images — appended LAST so every existing input
    # index stays stable (same discipline as the bg input). Windows are
    # legacy 3-tuples (path, ts, te) — tests/older callers — or focal
    # 5-tuples (…, ox, oy); the focal tail rides through to the graph
    # where 50/50 degrades to the exact legacy crop string.
    spot_inputs: list[tuple[int, float, float, float, float]] = []
    for _sw in (spotlight_windows or ()):
        cmd += ["-loop", "1", "-i", _sw[0]]
        _sp_ox, _sp_oy = ((float(_sw[3]), float(_sw[4])) if len(_sw) > 4
                          else (50.0, 50.0))
        spot_inputs.append((_next_idx, float(_sw[1]), float(_sw[2]),
                            _sp_ox, _sp_oy))
        _next_idx += 1
    pip_inputs: list[tuple[int, float, float, float, float]] = []
    for _pw in (pip_windows or ()):
        cmd += ["-loop", "1", "-i", _pw[0]]
        _pp_ox, _pp_oy = ((float(_pw[3]), float(_pw[4])) if len(_pw) > 4
                          else (50.0, 50.0))
        pip_inputs.append((_next_idx, float(_pw[1]), float(_pw[2]),
                           _pp_ox, _pp_oy))
        _next_idx += 1
    strap_inputs: list[tuple[int, float, float]] = []
    for (_ns_path, _ns_ts, _ns_te) in (name_strap_windows or ()):
        cmd += ["-loop", "1", "-i", _ns_path]
        strap_inputs.append((_next_idx, float(_ns_ts), float(_ns_te)))
        _next_idx += 1
    # Director windows — appended LAST (index discipline). Full-canvas
    # graphics overlay at 0:0; caption states center above the LT strip.
    dwin_inputs: list[tuple[int, float, float, str, str, str]] = []
    _cap_y = f"{lt_y}-h-14"
    for _dwin in (directive_windows or ()):
        _dw_path, _dw_ts, _dw_te, _dw_kind = _dwin[:4]
        _dw_anim = _dwin[4] if len(_dwin) > 4 else "fade"
        cmd += ["-loop", "1", "-i", _dw_path]
        if _dw_kind == "caption":
            _x, _y, _dw_anim = "'(W-w)/2'", _cap_y, "fade"
        else:
            _x, _y = "0", "0"
        dwin_inputs.append((_next_idx, float(_dw_ts), float(_dw_te),
                            _x, _y, _dw_anim))
        _next_idx += 1
    # Reference-video cutaways — full-screen B-roll played during their
    # window. Looped (-stream_loop -1) so a clip shorter than the story
    # never triggers -shortest truncation; the enable window gates
    # visibility. Appended LAST (index discipline). Audio stays unmapped
    # here (mute) — the "duck" mix is a later, separate pass.
    vspot_inputs: list[tuple[int, float, float, str, float]] = []
    for (_vs_path, _vs_ts, _vs_te, _vs_am, _vs_tr) in (video_spotlight_windows or ()):
        # Resolve the audio decision NOW: "duck"/"auto" only actually duck
        # when the clip HAS an audio stream — else fall back to "mute"
        # (keeps the anchor narration and avoids mapping a missing [idx:a]).
        _am = str(_vs_am or "mute").strip().lower()
        if _am in ("duck", "auto"):
            _am = "duck" if _file_has_audio(_vs_path) else "mute"
        cmd += ["-stream_loop", "-1", "-i", _vs_path]
        vspot_inputs.append((_next_idx, float(_vs_ts), float(_vs_te),
                             _am, float(_vs_tr)))
        _next_idx += 1

    # Lower-third animation. Legacy = linear slide-in (pre-motion
    # canvases render byte-identically). NEW canvases carry
    # layout.lt_ease="out_cubic" → a decelerating broadcast entrance
    # (closed-form cubic ease-out baked into the ffmpeg x-expression:
    # x = -w + w*(1-(1-t/T)^3) — the native motion toolkit's out_cubic).
    _lt_eased = getattr(layout, "lt_ease", None) == "out_cubic"
    _enter = ("(-w+w*(1-pow(1-t/0.4\\,3)))" if _lt_eased
              else "-w+w*t/0.4")
    if lt_w > canvas_w:
        lt_x_expr = (
            f"if(lt(t\\,0.4)\\,{_enter}\\,"
            f"if(lt(t\\,2.0)\\,0\\,"
            f"max(W-w\\,-((t-2.0)*60))))"
        )
    else:
        lt_x_expr = f"if(lt(t\\,0.4)\\,{_enter}\\,0)"

    border_colour = "white"
    if layout is not None and getattr(layout, "bg_color", None):
        # bg_color is the canvas backdrop; the white border is a fixed
        # tile frame. Keep it white but allow override via a future
        # ``tile_border_color`` field without breaking existing canvases.
        border_colour = getattr(layout, "tile_border_color", None) or "white"
    # Theme frame colour — an explicit editor tile_border_color wins.
    if _theme is not None and not (layout is not None
                                   and getattr(layout, "tile_border_color", None)):
        border_colour = _theme.tile_border

    filter_complex, out_label, audio_map = _build_story_filter_graph(
        canvas_w=canvas_w, canvas_h=canvas_h,
        main_inner_w=main_inner_w, side_inner_w=side_inner_w,
        tile_inner_h=tile_inner_h,
        main_outer_w=main_outer_w, side_outer_w=side_outer_w,
        tile_outer_h=tile_outer_h,
        main_x=main_x, side_x=side_x, tile_y=tile_y,
        lt_y=lt_y, ticker_y=ticker_y,
        border_colour=border_colour,
        bg_color=getattr(layout, "bg_color", None) or "black",
        lt_x_expr=lt_x_expr, ticker_speed_px_s=ticker_speed_px_s,
        has_bg=has_bg, bg_input_idx=bg_input_idx,
        apply_ticker=apply_ticker, ticker_in_idx=ticker_in_idx,
        has_bug=has_bug, bug_in_idx=bug_in_idx,
        has_wm=has_wm, wm_in_idx=wm_in_idx,
        watermark_xy=(_watermark_overlay_xy(watermark_position, canvas_w, canvas_h)
                      if has_wm else ("", "")),
        bg_video_volume=bg_video_volume,
        gap_windows=tuple(gap_windows or ()),
        spotlight=tuple(spot_inputs),
        pip=tuple(pip_inputs),
        straps=tuple(strap_inputs),
        story_fx_vf=story_fx_vf,
        dwin=tuple(dwin_inputs),
        video_spotlight=tuple(vspot_inputs),
        main_outer_h=_main_outer_h,
        side_outer_h=_side_outer_h,
        side_y=_side_y,
        main_border=_main_border,
        has_side=_has_side,
        bg_mode=("blur_self" if (_sl is not None and _sl_bg == "blur_self")
                 else ("image" if _has_bgimg
                       else ("plate" if _has_plate else "legacy"))),
        bg_image_idx=_bgimg_idx,
        moments=tuple(layout_moments or ()),
        side_x_expr=_side_x_expr,
        side_y_expr=_side_y_expr,
        emphasis=tuple(emphasis_windows or ()),
    )
    # PROACTIVE filter-thread cap (the render-time regression fix): ffmpeg 8
    # defaults filter_complex_threads to the logical-core count (~20 here),
    # and its threaded scheduler queues frames PER thread — 3 parallel
    # composes × ~20 threads buffered these branch-heavy story graphs into
    # an OOM, which then fell back to the SLOW single-thread rescue (job
    # 603: a 75-min bulletin). A moderate cap bounds that memory while
    # keeping near-full speed (overlay chains gain little past ~6 threads),
    # so heavy stories compose fast INSTEAD of OOM→1-thread. Tunable via
    # KAIZER_V4_FILTER_THREADS; the OOM rung still drops to 1 if a graph
    # exceeds even this.
    try:
        _base_ft = max(1, int(os.environ.get("KAIZER_V4_FILTER_THREADS", "5") or "5"))
    except ValueError:
        _base_ft = 5
    # ADAPTIVE to story length. The threaded scheduler queues frames PER thread,
    # so a LONG story (a podcast split into 3-8 min segments — vs a 30-90s news
    # story) buffers far more and OOMs at the fixed cap, dropping to the SLOW
    # single-thread rescue (job 609: two 8-min stories × 5 threads = 16GB, > free
    # RAM). Scaling threads DOWN with duration lets long stories fit in RAM 2-wide
    # and compose at full speed INSTEAD of OOM→1-thread retry — the single biggest
    # long-video time sink. Short stories keep the full cap (unchanged/fast). Pure
    # scheduling: byte-identical output, every effect preserved. The OOM rung still
    # backstops anything heavier. Thresholds tunable (KAIZER_V4_FT_LONG_S/_MID_S).
    _fthreads = _base_ft
    try:
        _fthreads = _adaptive_filter_threads(
            _base_ft, _probe_clip_duration(story_clip_path) or 0.0,
            long_s=float(os.environ.get("KAIZER_V4_FT_LONG_S", "300") or "300"),
            mid_s=float(os.environ.get("KAIZER_V4_FT_MID_S", "150") or "150"))
    except Exception:
        pass
    cmd += [
        "-filter_complex_threads", str(_fthreads),
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

    # Applied bleep spans (absolute trimmed-timeline) — hashed per story so
    # a changed censor set re-renders exactly the affected clips.
    from pipeline_v4.bleep import load_report_spans as _load_bleep_spans
    _bleep_all = _load_bleep_spans(inputs.work_dir)

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
    # Ticker colourway: an explicit editor/job override wins; else the
    # THEME's colourway; else the classic default.
    _tk_col = getattr(inputs, "ticker_bg_color", None)
    if not _tk_col:
        _th_k = str(getattr(getattr(inputs, "layout", None), "theme", "")
                    or "").strip().lower()
        if _th_k:
            try:
                from pipeline_v4.theme_packs import get_theme as _gt
                _tk_col = (_gt(_th_k).ticker_bg if _gt(_th_k) else None)
            except Exception:
                _tk_col = None
    render_ticker(headlines, lang_cfg.code, lang_cfg.font_primary, ticker_path,
                  bg_color=_tk_col)

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
        # Bleep spans overlapping THIS story's window (absolute timeline).
        _s0 = float(getattr(s, "video_t_start", 0.0) or 0.0)
        _s1 = float(getattr(s, "video_t_end", 0.0) or 0.0)
        story_bleeps = tuple(
            (a, b) for (a, b) in _bleep_all if b > _s0 and a < _s1
        )
        # Director directive for THIS story — looked up once here; feeds
        # the layout choice below AND the fx/overlays/captions later.
        _dv = (getattr(inputs, "directives", None) or {}).get(
            int(getattr(s, "story_index", i) or i))
        # LIVE LAYOUTS: this story's designed layout. The PERSISTED
        # canvas value wins (an editor edit); the live Director plan
        # fills the first render (same precedence as overlays). Unknown /
        # not-yet-expressible keys resolve to None → the job's default
        # geometry (fail-soft: a stray key can never block a render).
        _slay_key = str(getattr(s, "layout_key", "") or "").strip().lower()
        if not _slay_key and _dv is not None:
            _slay_key = str(getattr(_dv, "layout", "") or "").strip().lower()
        _sgeom = None
        if _slay_key:
            try:
                from pipeline_v4.layout_library import story_geometry
                _sgeom = story_geometry(_slay_key)
            except Exception:
                _sgeom = None
            if _sgeom is None:
                print(f"[v4/v1_bridge] story {i}: layout {_slay_key!r} not "
                      f"renderable — using the default layout", flush=True)
        # Image-less stretches of this story → main video full-screen
        # ("cut back to the anchor"). [] for every gapless/legacy canvas.
        story_gaps = _gap_windows(story_images, pool_dir, story_dur)
        # PANEL FACE-SAFETY (operator-reported on job 598): the picture
        # box must never sit on a face — mirror a floating (OTS) box to
        # the cleaner side, or strip the panel when both sides carry
        # faces / when it rides above gap-fullscreen video. Runs BEFORE
        # the |slay: fingerprint so corrected stories re-render
        # deterministically.
        if _sgeom is not None:
            _sgeom, _pfs_action = _panel_face_safety(
                _sgeom,
                src_path=str(getattr(inputs, "trimmed_video_path", "") or ""),
                t0=float(getattr(s, "video_t_start", 0.0) or 0.0),
                t1=float(getattr(s, "video_t_end", 0.0) or 0.0),
                work_dir=str(bdir), story_index=i,
                gaps=tuple(story_gaps or ()))
            if _pfs_action:
                print(f"[v4/v1_bridge] story {i}: picture panel "
                      f"{_pfs_action} (face-safety)", flush=True)
        if _sgeom is not None and not (_sgeom.picture or _sgeom.pips):
            # the layout IS a full-screen video design — a gap fallback
            # would composite the identical picture; skip the extra split
            story_gaps = ()
        # Key-moment images the canvas marks spotlight="fullscreen" pop
        # over the whole frame for their window (text stays on top);
        # spotlight="pip" shows BOTH — video full-screen + image inset.
        story_spots = _spotlight_windows(story_images, pool_dir, story_dur)
        story_pips = _spotlight_windows(story_images, pool_dir, story_dur, kind="pip")
        # Reference-video (B-roll) cutaways — full-screen video for [ts,te].
        story_videos = _video_spotlight_windows(story_images, pool_dir, story_dur)
        # STORY-EDGE GUARD (operator-reported): when a crossfade stitch
        # will blend this story's boundaries, a full-screen state (gap /
        # spotlight / PiP) touching the first/last moments makes the
        # framed layout "disappear and reappear" mid-transition — the
        # dissolve blends full-frame video against the framed tile. Keep
        # every full-screen window >=0.75s away from both story edges so
        # joints always blend two IDENTICAL layout geometries. Runs
        # BEFORE the hash, so affected stories re-render deterministically.
        if (getattr(inputs, "story_transition", None)
                or getattr(inputs, "directives", None)):
            story_gaps = _edge_clamp_windows(story_gaps, story_dur)
            story_spots = _edge_clamp_windows(story_spots, story_dur,
                                              with_path=True)
            story_pips = _edge_clamp_windows(story_pips, story_dur,
                                             with_path=True)
            # Video cutaways carry (path, ts, te, audio_mode, trim_start) —
            # clamp their window off the story edges too (same anti-ghost
            # reason as spotlights) and drop sub-1s slivers.
            _ve = 0.75
            story_videos = [
                (p, round(max(a, _ve), 3), round(min(b, story_dur - _ve), 3), am, tr)
                for (p, a, b, am, tr) in story_videos
                if (min(b, story_dur - _ve) - max(a, _ve)) >= 1.0
            ]
        # Mid-story layout moments: persisted canvas value wins, live
        # Director plan fills the first render (overlays precedence);
        # sanitized against THIS story's final cutaway windows so
        # explicit reference footage always wins the timeline.
        _smom_src = (getattr(s, "layout_moments", None)
                     or (getattr(_dv, "layout_moments", None)
                         if _dv is not None else None) or ())
        _smoments = _sanitize_layout_moments(
            _smom_src, story_dur, cutaways=tuple(story_videos))
        # EMPHASIS PUNCH-IN (the directive field was planned + sanitized
        # by the Director but never rendered until now): each surviving
        # timestamp becomes a ~0.45s hard-cut 1.06x zoom of the main
        # source. Read STRAIGHT from the directive — emphasis is NOT
        # canvas-persisted, so there is no persisted-canvas-wins step
        # here (unlike overlays/layout_moments). Windows that would fight
        # a full-screen state (moment/cutaway/spotlight/PiP) are dropped.
        # Kill switch: KAIZER_V4_EMPHASIS_PUNCH (default on) gates the
        # windows AND (via the conditional fingerprint below) the hash,
        # so toggling it re-renders exactly the punched stories.
        _epunch: list[tuple[float, float]] = []
        if _dv is not None and _flag_on("KAIZER_V4_EMPHASIS_PUNCH"):
            _eblk = ([(m[1], m[2]) for m in _smoments]
                     + [(w[1], w[2]) for w in story_spots]
                     + [(w[1], w[2]) for w in story_pips]
                     + [(w[1], w[2]) for w in story_videos])
            _epunch = _sanitize_emphasis_windows(
                getattr(_dv, "emphasis", []) or [], story_dur, blocked=_eblk)
        # Name-strap (polish A): (label, ts, te) per labeled image when
        # the story opted in. Hash on the SPECS; PNGs render lazily only
        # on a cache miss.
        strap_specs: list[tuple[str, float, float]] = []
        if getattr(s, "name_strap", False):
            def _ia_(obj, name, default=None):
                return (obj.get(name, default) if isinstance(obj, dict)
                        else getattr(obj, name, default))
            for img in (story_images or []):
                lab = (_ia_(img, "label", "") or "").strip()
                if not lab:
                    continue
                # A bare FILENAME is not a subject line — no strap.
                # (Web-chain images fall back to their file stem as the
                # label; painting "story00_news_01_c8e05072" on screen
                # was operator-visible on job 598.)
                if (re.match(r"(?i)^story\d+_", lab)
                        or re.search(r"(?i)_[0-9a-f]{6,}$", lab)
                        or ("_" in lab and " " not in lab)):
                    continue
                try:
                    ns_ts = max(0.0, float(_ia_(img, "t_start", 0.0) or 0.0))
                    ns_te = min(story_dur, float(_ia_(img, "t_end", 0.0) or 0.0))
                except (TypeError, ValueError):
                    continue
                if ns_te <= ns_ts + 0.05:
                    continue
                strap_specs.append((lab[:120], round(ns_ts, 3), round(ns_te, 3)))
            strap_specs = strap_specs[:8]   # bound the ffmpeg input count
        # Per-story layout: render the carousel at the layout's picture
        # rect (side tile OR inset) so images are composed for the shape
        # they'll occupy — not cover-cropped from the classic tall panel.
        _car_w = _car_h = None
        if _sgeom is not None:
            _prect = _sgeom.picture or ((_sgeom.pips[0].x, _sgeom.pips[0].y,
                                         _sgeom.pips[0].w, _sgeom.pips[0].h)
                                        if _sgeom.pips else None)
            if _prect:
                _lay_base = getattr(inputs, "layout", None)
                _cw_full = getattr(_lay_base, "width", None) or V4_W
                _ch_full = getattr(_lay_base, "height", None) or V4_H
                _car_w = max(64, int(round(_prect[2] / 100.0 * _cw_full))
                             - 2 * V4_TILE_BORDER)
                _car_h = max(64, int(round(_prect[3] / 100.0 * _ch_full))
                             - 2 * V4_TILE_BORDER)
        # First image + its focal offsets — used by BOTH the carousel's
        # base plate (designed-layout panels must never show black
        # between image windows) and the static-sidebar fallback below.
        first_src = None
        if story_images:
            fa = story_images[0]
            first_src = (fa.get("src") if isinstance(fa, dict)
                         else getattr(fa, "src", None))
        _first_img = _resolve_pool_image(first_src or "", pool_dir)
        # Focal framing rides along ONLY when the panel shows the
        # story's own first image — the legacy pool[i] fallback has no
        # canvas entry, so its offsets would belong to a DIFFERENT
        # picture. Defaults keep the PNG byte-identical (cache-safe).
        _fx = _fy = 50.0
        if _first_img and story_images:
            _fa0 = story_images[0]
            try:
                _fxv = (_fa0.get("offset_x_pct") if isinstance(_fa0, dict)
                        else getattr(_fa0, "offset_x_pct", None))
                _fyv = (_fa0.get("offset_y_pct") if isinstance(_fa0, dict)
                        else getattr(_fa0, "offset_y_pct", None))
                # None check (not `or`): 0 is a legitimate edge framing.
                _fx = 50.0 if _fxv is None else float(_fxv)
                _fy = 50.0 if _fyv is None else float(_fyv)
            except (TypeError, ValueError):
                _fx = _fy = 50.0
        # Base plate only for DESIGNED-layout panels (classic sidebar
        # keeps its exact legacy graph): the story's first image, else
        # the pool fallback — never bare background.
        _car_base = None
        _car_bx = _car_by = 50.0
        if _car_w and _flag_on("KAIZER_V4_PANEL_BASE"):
            _car_base = _first_img or (pool[i] if i < len(pool) else None)
            if _car_base and _car_base == _first_img:
                _car_bx, _car_by = _fx, _fy
        carousel = _ensure_sidebar_carousel(
            images=story_images, pool_dir=pool_dir,
            out_path=str(bdir / f"_sidebar_carousel_{i:02d}.mp4"),
            story_duration=story_dur, bg_color=bg_col,
            sidebar_w=_car_w, sidebar_h=_car_h,
            base_image=_car_base, base_ox=_car_bx, base_oy=_car_by,
        )
        if carousel:
            sidebar_path = carousel
            sidebar_is_video = True
        else:
            sidebar_img = (
                _first_img
                or (pool[i] if i < len(pool) else None)
            )
            sidebar_path = _resolve_sidebar(
                work_dir=bdir, story_index=i, pool_image_path=sidebar_img,
                offset_x_pct=_fx, offset_y_pct=_fy,
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

        _fx_env = _effects_vf_from_env(getattr(inputs, "effects_mode", None))
        # Director override: this story's OWN mood pack + garnish fx
        # (multi-style per video). Hash gets the resolved chain, so any
        # Director decision change deterministically re-renders the story.
        # (_dv itself was looked up once, before the layout resolution.)
        _dv_fp = ""
        if _dv is not None:
            try:
                from pipeline_v4.director import story_fx_chain
                _fx_env = story_fx_chain(_dv, base_chain=_fx_env)
            except Exception:
                pass
        # Overlay ENTRANCE anims: switch + per-id plan decided HERE
        # (pre-hash) so the |oanim marker and the rendered dwin specs
        # can never drift. KAIZER_V4_OVERLAY_ANIM=0 → all-fade → the
        # graph AND the hash are exactly legacy.
        _oanim_on = _flag_on("KAIZER_V4_OVERLAY_ANIM")
        # overlays + captions are render inputs too — fingerprint them into
        # the conditional effects ingredient so a changed graphic (edited
        # TEXT / timing / which id) or Director decision re-renders the story
        # deterministically. Prefer the PERSISTED canvas overlays (which carry
        # the editor's text) over the live Director plan.
        try:
            _co_fp = getattr(s, "overlays", None)
            if _co_fp:
                _ov_blob = [{"id": getattr(o, "id", ""), "t": getattr(o, "t", 0),
                             "dur": getattr(o, "dur", None),
                             "fields": dict(getattr(o, "fields", {}) or {})}
                            for o in _co_fp]
            elif _dv is not None:
                _ov_blob = _dv.overlays
            else:
                _ov_blob = None
            if _ov_blob is not None:
                _cap_fp = (_dv.captions if _dv is not None else "none") or "none"
                _dv_fp = ("|dov:" + json.dumps(_ov_blob, sort_keys=True, default=str)
                          + "|cap:" + _cap_fp)
                # Conditional cache marker — appended ONLY when at least
                # one graphic will actually animate, so every fade-only /
                # switch-off render keeps its exact legacy hash.
                if _oanim_on:
                    _oanims = [_overlay_anim(str((_o.get("id", "")
                                                  if isinstance(_o, dict)
                                                  else getattr(_o, "id", "")) or ""))
                               for _o in list(_ov_blob)[:3]]
                    if any(_a != "fade" for _a in _oanims):
                        _dv_fp += "|oanim:" + ",".join(_oanims)
        except Exception:
            _dv_fp = ""
        # Reference-video cutaway fingerprint — folded into the hash ONLY
        # when videos exist, so legacy canvases keep their exact hash (100%
        # cache hits, byte-identical). File size+mtime → replacing a clip
        # re-renders just that story.
        _vid_fp = ""
        if story_videos:
            _vparts = []
            for (_vp, _va, _vb, _vam, _vtr) in story_videos:
                try:
                    _vst = os.stat(_vp)
                    _vfp = f"{_vst.st_size}:{int(_vst.st_mtime)}"
                except OSError:
                    _vfp = "0:0"
                _vparts.append(f"{os.path.basename(_vp)}:{_vfp}:"
                               f"{_va:.3f}:{_vb:.3f}:{_vam}:{_vtr:.3f}")
            _vid_fp = "|vid:" + ";".join(_vparts)
        # Per-story layout fingerprint — folded into the hash ONLY when a
        # layout key resolved, so every legacy canvas keeps its exact hash
        # (100% cache hits, byte-identical). Geometry + bg mode + backdrop
        # file identity all bust the cache deterministically.
        _bg_img_path = None
        if _sgeom is not None and _sgeom.bg == "bg_image":
            _bg_first = None
            if story_images:
                _fa0 = story_images[0]
                _bg_first = (_fa0.get("src") if isinstance(_fa0, dict)
                             else getattr(_fa0, "src", None))
            _bg_img_path = (_resolve_pool_image(_bg_first or "", pool_dir)
                            or (pool[i] if i < len(pool) else None))
        _slay_fp = ""
        if _sgeom is not None:
            _bgfp = ""
            if _bg_img_path:
                try:
                    _bst = os.stat(_bg_img_path)
                    _bgfp = (f"{os.path.basename(_bg_img_path)}:"
                             f"{_bst.st_size}:{int(_bst.st_mtime)}")
                except OSError:
                    _bgfp = "0:0"
            _slay_fp = "|slay:" + json.dumps(
                [_sgeom.key, _sgeom.bg, int(_sgeom.video_borderless),
                 list(_sgeom.video), list(_sgeom.picture or ()),
                 [[p.x, p.y, p.w, p.h] for p in _sgeom.pips], _bgfp],
                sort_keys=False)
        # Layout-moment fingerprint — conditional, same discipline: only
        # stories that carry moments hash them; legacy hashes untouched.
        if _smoments:
            _slay_fp += "|smom:" + json.dumps(
                [[g.key, ts, te, tr] for (g, ts, te, tr) in _smoments],
                sort_keys=False)
        # Theme fingerprint — conditional: only themed jobs hash it (key +
        # painter version cover plate, frame colour and ticker colourway).
        _th_fp_key = str(getattr(getattr(inputs, "layout", None), "theme", "")
                         or "").strip().lower()
        if _th_fp_key:
            try:
                from pipeline_v4.theme_packs import THEME_VERSION as _TV
            except Exception:
                _TV = "t?"
            _slay_fp += f"|theme:{_th_fp_key}:{_TV}"
        # Tile-entrance motion fingerprint — conditional (same discipline
        # as |slay/|smom/|theme): only NEW-feature stories that will
        # slide carry it; legacy hashes never move, and toggling
        # KAIZER_V4_LAYOUT_MOTION re-renders exactly those stories.
        if (_flag_on("KAIZER_V4_LAYOUT_MOTION")
                and (_sgeom is not None or bool(_th_fp_key))
                and (_sgeom is None or bool(_sgeom.picture or _sgeom.pips))):
            _slay_fp += "|tmot:v1"
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
            gap_windows=tuple(story_gaps),
            name_straps=tuple(strap_specs),
            bleep_spans=story_bleeps,
            # |emph: rides the effects ingredient CONDITIONALLY (only when
            # punch windows survived sanitize) — same discipline as
            # |slay/|smom/|vid, so emphasis-free stories keep their hash.
            effects_vf=(_fx_env + _dv_fp + _vid_fp + _slay_fp
                        + _emphasis_fingerprint(_epunch)),
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
        # Cache miss → materialize the strap PNGs now (specs already hashed).
        strap_windows: list[tuple[str, float, float]] = []
        for k, (ns_lab, ns_ts, ns_te) in enumerate(strap_specs):
            ns_png = str(bdir / f"_strap_{i:02d}_{k:02d}.png")
            if _render_name_strap_png(text=ns_lab,
                                      font_path=lang_cfg.font_primary,
                                      out_path=ns_png):
                strap_windows.append((ns_png, ns_ts, ns_te))
        # Director windows: category GRAPHICS (positioned full-canvas
        # PNGs) + karaoke CAPTION states from the word timestamps. Every
        # element fail-soft — a missing graphic never blocks a story.
        dwin_specs: list[tuple[str, float, float, str, str]] = []
        # Broadcast graphics (overlays / HUD): prefer the PERSISTED canvas
        # overlays (editor-editable text/timing/which-id) and fall back to the
        # live Director plan for legacy canvases that carry none. Captions still
        # come from the Director directive below.
        _ovl_list = []
        _canvas_ovl = getattr(s, "overlays", None)
        if _canvas_ovl:
            for _co in _canvas_ovl:
                _ovl_list.append({"id": getattr(_co, "id", ""),
                                  "t": getattr(_co, "t", 0.8),
                                  "dur": getattr(_co, "dur", None),
                                  "fields": dict(getattr(_co, "fields", {}) or {})})
        elif _dv is not None:
            _ovl_list = [{"id": _o.get("id", ""), "t": _o.get("t", 0.8),
                          "dur": None, "fields": {}}
                         for _o in (_dv.overlays or [])]
        if _ovl_list or _dv is not None:
            _sdur = max(0.5, float(getattr(s, "video_t_end", 0.0))
                        - float(getattr(s, "video_t_start", 0.0)))
            _lay = getattr(inputs, "layout", None)
            _cw = getattr(_lay, "width", None) or V4_W
            _ch = getattr(_lay, "height", None) or V4_H
            try:
                from pipeline_v4.overlays import render_overlay_positioned
                for _oi, _o in enumerate(_ovl_list[:3]):
                    if not _o.get("id"):
                        continue
                    _png = str(bdir / f"_dov_{i:02d}_{_oi}.png")
                    if render_overlay_positioned(
                            _o.get("id", ""), _png, canvas_w=_cw,
                            canvas_h=_ch, font_path=lang_cfg.font_primary,
                            overrides=_o.get("fields")):
                        _a = max(0.0, min(float(_o.get("t", 0.8) or 0.8),
                                          _sdur - 1.0))
                        _odur = float(_o.get("dur") or 4.5)
                        _anim = (_overlay_anim(str(_o.get("id", "") or ""))
                                 if _oanim_on else "fade")
                        dwin_specs.append((_png, round(_a, 3),
                                           round(min(_sdur, _a + _odur), 3),
                                           "full", _anim))
            except Exception as _ox:
                print(f"[v4/v1_bridge] director graphics skipped "
                      f"(story {i}): {_ox}", flush=True)
            if _dv is not None and (getattr(_dv, "captions", "none") or "none") != "none":
                try:
                    _words = _story_words_sidecar(inputs.work_dir, i)
                    if _words:
                        from pipeline_v4 import typography as _ty
                        _st_k, _hi_k = _karaoke_params(_dv.captions)
                        _wins = _ty.karaoke_captions(
                            _words, out_dir=str(bdir),
                            canvas_w=min(int(_cw) - 80, 1400),
                            font_path=lang_cfg.font_primary,
                            prefix=f"dcap{i:02d}", style=_st_k, hilite=_hi_k)
                        if len(_wins) > 60:
                            print(f"[v4/v1_bridge] captions capped 60/"
                                  f"{len(_wins)} (story {i})", flush=True)
                        for (_p, _a, _b) in _wins[:60]:
                            dwin_specs.append((_p, float(_a),
                                               min(float(_b), _sdur),
                                               "caption", "fade"))
                except Exception as _cx:
                    print(f"[v4/v1_bridge] captions skipped (story {i}): "
                          f"{_cx}", flush=True)
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
            gap_windows=tuple(story_gaps),
            spotlight_windows=tuple(story_spots),
            pip_windows=tuple(story_pips),
            name_strap_windows=tuple(strap_windows),
            story_fx_vf=_fx_env,
            directive_windows=tuple(dwin_specs),
            video_spotlight_windows=tuple(story_videos),
            story_layout=_sgeom,
            bg_image_path=_bg_img_path,
            layout_moments=tuple(_smoments),
            emphasis_windows=tuple(_epunch),
        )
        _write_cached_hash(hash_file, new_hash)
        # Progress marker for the job UI's live substeps (parseV4Log) —
        # mirrors the cache-hit line above so every story logs exactly one
        # "story K/N" completion either way.
        print(f"[v4/v1_bridge] story {i+1}/{n_stories} composed",
              flush=True)
        return (i, composed, False)

    # Bounded parallel per-story compose (Wave 4 item E). Cache hits
    # short-circuit inside each task exactly as before; the stitch
    # below consumes results strictly in story order. A single story
    # failure still fails the whole bulletin (same contract as the old
    # sequential loop — the orchestrator catches and reports it).
    # Default 2 — restored now that the moment cap (KAIZER_V4_MAX_MOMENTS=3)
    # halves each story's filtergraph weight. The earlier OOM (job 606,
    # concurrency 2) was two MOMENT-RICH graphs (up to 8 moments = 16
    # branches each) colliding; with the cap each graph is ~half that, so
    # TWO capped stories fit the budget together — reclaiming the ~2x speed
    # that sequential (job 607: 25 stories × ~2min = 49min bulletin) gave
    # up for reliability. The per-graph thread cap + OOM rung + NVENC→CPU
    # fallback still protect a stray heavy story. Pure scheduling; identical
    # output. Tune with KAIZER_V4_RENDER_CONCURRENCY (1 = the safe fallback
    # if a job ever regresses; 3 on a bigger box).
    try:
        _story_workers = max(1, int(os.environ.get(
            "KAIZER_V4_RENDER_CONCURRENCY", "2") or "2"))
    except ValueError:
        _story_workers = 2
    def _is_oom(exc: BaseException) -> bool:
        m = str(exc)
        return ("Cannot allocate memory" in m or "4294967284" in m
                or "error code: -12" in m)

    results: list[tuple[int, str, bool]] = []
    if n_stories > 1 and _story_workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        first_exc: Optional[BaseException] = None
        oom_failed: list[int] = []
        with ThreadPoolExecutor(
            max_workers=min(_story_workers, n_stories),
            thread_name_prefix="v4-story",
        ) as story_pool:
            futures = [
                story_pool.submit(_compose_one_story, i, s)
                for i, s in enumerate(inputs.stories)
            ]
            for i, fut in enumerate(futures):   # submit order == story order
                try:
                    results.append(fut.result())
                except BaseException as exc:  # noqa: BLE001 — re-raised below
                    if _is_oom(exc):
                        # Concurrent composes exhausted RAM (job 600: an
                        # 11-story job × 3 workers OOM'd every story, and
                        # each story's OWN retry also died because its
                        # neighbours were still holding the memory).
                        # Rescue AFTER the pool drains, serially.
                        oom_failed.append(i)
                    elif first_exc is None:
                        first_exc = exc
        if first_exc is not None:
            raise first_exc
        if oom_failed:
            print(f"[v4/v1_bridge] {len(oom_failed)} story compose(s) hit "
                  f"out-of-memory under {_story_workers} parallel workers — "
                  f"rescuing serially: {oom_failed}", flush=True)
            for i in oom_failed:
                results.append(_compose_one_story(i, inputs.stories[i]))
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

    # 4) Stitch (no ticker yet). NEW canvases carry story_transition
    #    ("fade" by default) → crossfade stitch; legacy/None → V1's
    #    concat-demuxer hard cuts (-c copy, byte-identical to before).
    #    Any transition-stitch failure falls back to the hard cut.
    no_ticker_stitched = str(bdir / "_stitched_no_ticker.mp4")
    _trans = (getattr(inputs, "story_transition", None) or "").strip().lower()
    # Director wiring: per-joint transitions — the INCOMING story's own
    # transition_in choice, one per joint. Falls back to the single
    # canvas transition, then hard cuts.
    _dirs = getattr(inputs, "directives", None) or {}
    _joint_list = None
    if _dirs and len(composed_paths) > 1:
        _joint_list = [
            (getattr(_dirs.get(i), "transition_in", "") or _trans or "fade")
            for i in range(1, len(composed_paths))
        ]
    _stitched = None
    if (_joint_list or _trans) and len(composed_paths) > 1:
        _stitched = _stitch_with_transitions(
            composed_paths, no_ticker_stitched,
            transition=(_joint_list or _trans))
    if _stitched is None:
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

    # CATEGORY SOUND DESIGN (Director on): stings + graphic ticks +
    # ducked mood bed, mixed BEFORE the loudness conform so the final
    # still lands exactly at -14 LUFS. Fail-soft: original kept.
    if _dirs and (os.environ.get("KAIZER_V4_SOUND_DESIGN") or "1"
                  ).strip().lower() not in ("0", "off"):
        try:
            _cd = [(_probe_clip_duration(p) or 0.0) for p in composed_paths]
            _fd = 0.5 if _stitched is not None else 0.0
            _starts, _acc = [], 0.0
            for _k, _d in enumerate(_cd):
                _starts.append(_acc)
                _acc += _d - (_fd if _k < len(_cd) - 1 else 0.0)
            _apply_sound_design(inputs.output_path, directives=_dirs,
                                stories=inputs.stories,
                                story_starts=_starts, stitched_dur=_acc,
                                work_dir=str(bdir))
        except Exception as _sx:
            print(f"[v4/v1_bridge] sound design skipped: {_sx}", flush=True)

    # Broadcast loudness conform (-14 LUFS two-pass, audio-only, fail-soft).
    # Runs on the FINAL file so every caller (orchestrator, editor
    # re-render, recompose) ships normalized audio.
    from pipeline_v4.audio_conform import conform_loudness
    conform_loudness(inputs.output_path)

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
        if _ov_html and (getattr(t, "format", None) or "html") == "svg":
            # A stale per-job HTML override (authored against a PREVIOUS
            # HTML template) can never apply to an SVG template — rendering
            # it verbatim would silently ignore the selected template.
            print(f"[v4/custom] ignoring stale HTML override — template "
                  f"{tid} is an SVG layout", flush=True)
            _ov_html = ""
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
                    effects_vf=getattr(inputs, "effects_vf", "") or "",
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
            effects_vf=getattr(inputs, "effects_vf", "") or "",
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


def _render_dual_video_short(inputs: "ShortRenderInputs",
                             font_path: Optional[str]) -> bool:
    """dual_video shorts layout — TWO videos stacked (spec: "add another
    functionality of both videos"):

        ┌──────────────────┐
        │ VIDEO A  (910px) │  the AI-trimmed main clip (audio master)
        ├──────────────────┤
        │ TITLE BAND 100px │  headline strap (language font)
        ├──────────────────┤
        │ VIDEO B  (910px) │  second cam / reference b-roll (muted, looped)
        └──────────────────┘

    Returns True on success, False on any failure (caller falls back to
    torn_card so a short never dies on a missing second video)."""
    try:
        second = inputs.second_video_path
        if not (second and os.path.isfile(second)):
            return False
        W, H, BAND = 1080, 1920, 100
        half = (H - BAND) // 2                    # 910
        work = Path(inputs.work_dir)
        work.mkdir(parents=True, exist_ok=True)

        # Title band PNG — PIL with the language font (Telugu/Hindi safe).
        band_png = str(work / f"_dualband_{Path(inputs.output_path).stem}.png")
        title = " ".join((inputs.title_text or "").split())[:120]
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new("RGBA", (W, BAND), (10, 10, 12, 255))
            d = ImageDraw.Draw(img)
            d.rectangle([0, 0, W - 1, 5], fill=(193, 18, 18, 255))
            d.rectangle([0, BAND - 6, W - 1, BAND - 1], fill=(193, 18, 18, 255))
            try:
                font = (ImageFont.truetype(font_path, 40)
                        if font_path and os.path.isfile(font_path)
                        else ImageFont.load_default())
            except Exception:
                font = ImageFont.load_default()
            while title:
                box = d.textbbox((0, 0), title, font=font)
                if box[2] - box[0] <= W - 60:
                    break
                title = title[:-4].rstrip() + "…"
            box = d.textbbox((0, 0), title, font=font)
            d.text(((W - (box[2] - box[0])) // 2,
                    (BAND - (box[3] - box[1])) // 2 - box[1]),
                   title, font=font, fill=(255, 255, 255, 255))
            img.save(band_png, "PNG")
        except Exception as exc:
            print(f"[v4/dual] title band failed ({exc})", flush=True)
            return False

        fc = (
            f"[0:v]scale={W}:{half}:force_original_aspect_ratio=increase,"
            f"crop={W}:{half},setsar=1[va];"
            f"[1:v]scale={W}:{half}:force_original_aspect_ratio=increase,"
            f"crop={W}:{half},setsar=1[vb];"
            f"color=c=black:s={W}x{H}:r=30[bg];"
            f"[bg][va]overlay=x=0:y=0:shortest=1[s1];"
            f"[s1][vb]overlay=x=0:y={half + BAND}[s2];"
            f"[s2][2:v]overlay=x=0:y={half}[outv]"
        )
        cmd = [
            _ffmpeg_bin(), "-y", "-v", "error",
            "-i", inputs.trimmed_short_path,
            "-stream_loop", "-1", "-i", second,      # B loops to cover A
            "-loop", "1", "-i", band_png,
            "-filter_complex", fc,
            "-map", "[outv]", "-map", "0:a?",        # audio = main clip only
            *_enc_args(crf=20, preset_hint="medium"),
            "-pix_fmt", "yuv420p",
            "-r", "30", "-fps_mode", "cfr",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-shortest", "-movflags", "+faststart",
            inputs.output_path,
        ]
        _run_ffmpeg(cmd, timeout=900, log_label="dual_video_short")
        ok = (os.path.isfile(inputs.output_path)
              and os.path.getsize(inputs.output_path) > 0)
        if ok:
            print(f"[v4/dual] dual-video short composed "
                  f"({Path(second).name} under the main clip)", flush=True)
        return ok
    except Exception as exc:
        print(f"[v4/dual] failed ({exc}) — torn_card fallback", flush=True)
        return False


def _custom_short_fx_handled() -> bool:
    """True when the custom-template engine consumes per-clip effects
    (RenderRequest has an ``effects_vf`` field — _render_custom_short
    threads inputs.effects_vf into it). Then render_short must NOT also
    run its whole-frame grade post-pass on a custom short: the design
    PNG (headline/ticker plate) must stay crisp and the footage must not
    be graded twice. A capability probe (not a flag) so the guard flips
    with the engine itself; False → legacy behaviour (post-pass grades
    the whole frame, exactly as before the per-clip path existed)."""
    try:
        import dataclasses as _dc
        from services import custom_templates as _ct
        return "effects_vf" in {f.name for f in _dc.fields(_ct.RenderRequest)}
    except Exception:
        return False


def render_short(inputs: ShortRenderInputs) -> str:
    """Compose one V1-style short (torn_card / clean_card / split_frame
    / follow_bar / dual_video). Returns the output path on success;
    raises on failure."""
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

    # dual_video needs a second source; degrade to torn_card when absent
    # or when the native compose fails — a short never dies on layout.
    if layout == "dual_video":
        if _render_dual_video_short(inputs, lang_cfg.font_primary):
            layout = "__done__"
        else:
            layout = "torn_card"

    _custom_brand = {}
    if layout == "__done__":
        pass
    elif _is_custom:
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

    # Effects grade post-pass (Phase 2) — apply the AI Director's per-story
    # mood grade (+ garnish fx) to the finished short so it matches its
    # bulletin story's look (crime short = cold grade, etc.). Runs BEFORE
    # the watermark so the brand stamp stays crisp/un-graded. Empty
    # effects_vf -> skipped (byte-identical to legacy). Fail-soft: a bad
    # chain leaves the ungraded short untouched.
    _efx_vf = (getattr(inputs, "effects_vf", "") or "").strip().strip(",")
    # CUSTOM SHORTS double-grade guard: when the layout is custom: AND the
    # engine's per-clip effects path is active (inputs.effects_vf was
    # threaded into the RenderRequest above), the footage inside each video
    # slot is ALREADY graded under the design overlay — grading the whole
    # finished frame here would soften the design PNG and grade the footage
    # twice. Built-in shorts keep the post-pass exactly as today.
    if _efx_vf and _is_custom and _custom_short_fx_handled():
        print("[v4/fx] custom short: per-clip grade rendered inside the "
              "template engine — skipping the whole-frame post-pass "
              "(design stays crisp, footage graded once)", flush=True)
        _efx_vf = ""
    if _efx_vf and os.path.isfile(inputs.output_path):
        graded = inputs.output_path + ".fx.mp4"
        try:
            cmd = [
                _ffmpeg_bin(), "-y", "-v", "error",
                "-i", inputs.output_path,
                "-vf", _efx_vf,
                *_enc_args(crf=20, preset_hint="veryfast"),
                "-pix_fmt", "yuv420p",
                "-c:a", "copy",
                "-movflags", "+faststart",
                graded,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if r.returncode == 0 and os.path.isfile(graded) and os.path.getsize(graded) > 0:
                os.replace(graded, inputs.output_path)
            else:
                print(f"[v4/fx] short effects pass failed (kept original): "
                      f"{(r.stderr or '')[-200:]}", flush=True)
                try:
                    os.remove(graded)
                except OSError:
                    pass
        except Exception as exc:
            print(f"[v4/fx] short effects pass error (kept original): {exc}", flush=True)

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

    # Broadcast loudness conform (-14 LUFS two-pass, audio-only, fail-soft) —
    # same finalization the bulletin gets.
    from pipeline_v4.audio_conform import conform_loudness
    conform_loudness(inputs.output_path)

    return inputs.output_path
