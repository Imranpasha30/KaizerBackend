"""Phase 2 of the long-form pipeline — TV9-style broadcast layout.

Composes one bulletin story (1920×1080) with:
  - Main video on the left ~67 % (1280×800)
  - Sidebar on the right ~30 % (580×800) — Phase 3 fills this with a
    Ken-Burns image carousel; Phase 2 ships a static placeholder.
  - Lower-third with kicker chip + headline (slide-in from left)
  - News ticker scrolling continuously across the bottom
  - Channel bug (logo) in the top-right corner

Reuses the existing Playwright HTML→PNG infrastructure (defined at
``pipeline.py:_playwright_render_title``) so Telugu / Devanagari /
Tamil shape correctly via HarfBuzz. Falls back to PIL/FreeType when
Playwright is unavailable so the bulletin still ships graphics on
boxes that don't have headless Chromium installed (e.g. CI).

Phase 2 contract
----------------
- ONE FFmpeg invocation per story (no Python frame loops).
- All overlays are static PNGs; animation is driven by FFmpeg overlay
  ``x``/``y`` expressions (so we never write per-frame PNGs).
- The lower-third PNG and channel-bug PNG are re-rendered per story.
  The ticker PNG is built once per bulletin and passed in by the
  caller — the same PNG drives every story's ticker so the marquee
  reads coherently across cuts.
- All renderers return a path or raise; helpers swallow Playwright
  errors and fall back to PIL transparently.

Phase 4 (PiP) extension
-----------------------
``compose_pip_story()`` takes everything ``compose_bulletin_story()``
takes plus a ``pip_clip_path`` and inserts a 320×180 video inset in
the top-right of the main video area, anchored under the channel
bug. ``pick_pip_source()`` chooses between (a) the next story's
first 8 s as B-roll, (b) the sidebar carousel re-rendered at
320×180, or (c) None (skipped).
"""
from __future__ import annotations

import base64
import html as _html
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("kaizer.pipeline.longform_compose")


# ── Public dataclasses ────────────────────────────────────────────────────────

@dataclass
class StoryMeta:
    """Per-story metadata used to drive the broadcast graphics."""
    title: str                        # full headline → lower-third + ticker
    kicker: str = "BREAKING"          # short red chip (e.g. POLITICS, BREAKING)
    language: str = "te"              # ISO-639-1 — drives font + script
    story_index: int = 0
    total_stories: int = 1
    importance: int = 5               # 1-10; >=8 swaps kicker to "BREAKING"


# ── Geometry constants (1920×1080 reference) ─────────────────────────────────

_W = 1920
_H = 1080
_MAIN_W   = 1280
_MAIN_H   = 800
_SIDE_W   = 580
_SIDE_H   = 800
_LT_H     = 140        # lower-third height
_TICKER_H = 50
_BUG_W    = 200
_BUG_H    = 70


# ── Rendering helpers — Playwright primary, PIL fallback ─────────────────────

def _try_playwright_html_to_png(
    html: str,
    out_path: str,
    width: int,
    height: int,
    timeout_ms: int = 8000,
) -> bool:
    """Render an HTML page to a transparent PNG. Returns True on success."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": width, "height": height})
            page.set_content(html)
            try:
                page.wait_for_function("window._done === true", timeout=timeout_ms)
            except Exception:
                # Page didn't signal — render whatever's painted.
                pass
            page.screenshot(path=out_path, omit_background=True)
            browser.close()
        return True
    except Exception as exc:
        logger.warning("longform_compose: Playwright render failed: %s", exc)
        return False


def _font_data_uri(font_path: Optional[str]) -> tuple[str, str]:
    """Return ``(font_face_css, font_family_name)`` from a TTF path."""
    if not font_path or not os.path.isfile(font_path):
        return ("", '"Inter","Helvetica Neue",Arial,sans-serif')
    try:
        with open(font_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        face = (
            "@font-face{font-family:'KaizerFont';"
            f"src:url('data:font/truetype;base64,{b64}');}}"
        )
        return (face, "'KaizerFont',sans-serif")
    except Exception as exc:
        logger.warning("longform_compose: font embed failed (%s): %s", font_path, exc)
        return ("", '"Inter","Helvetica Neue",Arial,sans-serif')


def _pil_fallback_lower_third(
    meta: StoryMeta, font_path: Optional[str], out_path: str,
) -> str:
    """Minimal PIL lower-third when Playwright isn't available.

    Renders a flat red bar with white kicker + white headline. Slide-in
    is still handled by FFmpeg's overlay ``x`` expression downstream.
    """
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGBA", (_W, _LT_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # Red gradient bar with darker bottom edge
    d.rectangle([0, 20, _W, _LT_H - 10], fill=(192, 24, 36, 235))
    d.rectangle([0, _LT_H - 10, _W, _LT_H], fill=(120, 12, 20, 235))
    # Kicker chip
    chip_w = 220
    d.rectangle([20, 35, 20 + chip_w, _LT_H - 25], fill=(255, 255, 255, 240))
    try:
        kicker_font = ImageFont.truetype(font_path, 36) if font_path else ImageFont.load_default()
        head_font = ImageFont.truetype(font_path, 56) if font_path else ImageFont.load_default()
    except Exception:
        kicker_font = ImageFont.load_default()
        head_font = ImageFont.load_default()
    d.text((20 + 18, 35 + 14), (meta.kicker or "NEWS")[:14],
           font=kicker_font, fill=(192, 24, 36, 255))
    headline = (meta.title or "").strip()[:90]
    d.text((20 + chip_w + 24, 35 + 12), headline, font=head_font, fill=(255, 255, 255, 255))
    img.save(out_path, "PNG")
    return out_path


def render_lower_third(
    meta: StoryMeta,
    font_path: Optional[str],
    out_path: str,
) -> str:
    """Render a single static lower-third PNG (full-width 1920×140)."""
    face, family = _font_data_uri(font_path)
    kicker = _html.escape((meta.kicker or "NEWS")[:14])
    headline = _html.escape((meta.title or "").strip()[:120])
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><style>
{face}
*{{margin:0;padding:0;box-sizing:border-box;}}
html,body{{width:{_W}px;height:{_LT_H}px;background:transparent;font-family:{family};}}
.bar{{position:absolute;left:0;right:0;top:20px;bottom:0;
  background:linear-gradient(180deg,#e02538 0%,#c01524 75%,#7a0a14 100%);
  box-shadow:0 -4px 14px rgba(0,0,0,0.35);
  border-top:3px solid #ffd640;}}
.row{{position:absolute;left:0;right:0;top:38px;height:{_LT_H-58}px;
  display:flex;align-items:center;padding:0 28px;}}
.kicker{{background:#ffffff;color:#c01524;font-weight:900;
  font-size:30px;letter-spacing:1.5px;padding:8px 20px;
  border-radius:4px;margin-right:24px;text-transform:uppercase;
  text-shadow:none;}}
.headline{{color:#ffffff;font-weight:800;font-size:48px;
  text-shadow:2px 2px 0 rgba(0,0,0,0.45);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  max-width:{_W-360}px;line-height:1.05;}}
</style></head><body>
<div class="bar"></div>
<div class="row">
  <div class="kicker">{kicker}</div>
  <div class="headline">{headline}</div>
</div>
<script>document.fonts.ready.then(function(){{window._done=true;}});</script>
</body></html>"""
    if _try_playwright_html_to_png(html, out_path, _W, _LT_H):
        return out_path
    # Fall through to PIL
    return _pil_fallback_lower_third(meta, font_path, out_path)


def render_ticker(
    headlines: list[str],
    lang: str,
    font_path: Optional[str],
    out_path: str,
    *,
    height: int = _TICKER_H,
    pad_chars: int = 6,
) -> str:
    """Render the scrolling-ticker source PNG.

    Concatenates all headlines separated by a star. The output PNG is
    intentionally wide (a few thousand pixels) so FFmpeg's overlay can
    scroll it as one strip via an ``x`` expression. Caller drives the
    scroll speed and looping.
    """
    sep = "  " + "★" + "  "
    text = sep.join(h.strip() for h in headlines if h and h.strip())
    if not text:
        text = "KAIZER NEWS"
    # Estimate width: ~24 px per character at 36 px font (rough; Playwright
    # will lay out the real width, we just need the canvas big enough).
    est_width = max(2400, 24 * (len(text) + pad_chars * 2))
    face, family = _font_data_uri(font_path)
    safe_text = _html.escape(text)
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><style>
{face}
*{{margin:0;padding:0;box-sizing:border-box;}}
html,body{{width:{est_width}px;height:{height}px;background:transparent;
  font-family:{family};}}
.strip{{position:absolute;left:0;right:0;top:0;bottom:0;
  background:linear-gradient(180deg,#0a1530 0%,#05091e 100%);
  border-top:2px solid #ffd640;border-bottom:2px solid #ffd640;
  display:flex;align-items:center;padding:0 24px;
  color:#ffffff;font-weight:700;font-size:28px;
  white-space:nowrap;letter-spacing:0.4px;}}
</style></head><body>
<div class="strip">{safe_text}</div>
<script>document.fonts.ready.then(function(){{window._done=true;}});</script>
</body></html>"""
    if _try_playwright_html_to_png(html, out_path, est_width, height):
        return out_path
    # PIL fallback — single line of text on a navy bar.
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGBA", (est_width, height), (10, 21, 48, 235))
    try:
        f = ImageFont.truetype(font_path, 28) if font_path else ImageFont.load_default()
    except Exception:
        f = ImageFont.load_default()
    d = ImageDraw.Draw(img)
    d.text((24, 8), text, font=f, fill=(255, 255, 255, 255))
    img.save(out_path, "PNG")
    return out_path


def render_channel_bug(
    channel_name: str,
    logo_path: Optional[str],
    out_path: str,
    *,
    width: int = _BUG_W,
    height: int = _BUG_H,
) -> str:
    """Render a static channel bug — logo + name with rounded background."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # Rounded translucent black plate
    try:
        d.rounded_rectangle([0, 0, width - 1, height - 1],
                            radius=10, fill=(0, 0, 0, 170))
    except AttributeError:
        d.rectangle([0, 0, width - 1, height - 1], fill=(0, 0, 0, 170))
    # Logo on the left if provided
    text_x = 14
    if logo_path and os.path.isfile(logo_path):
        try:
            with Image.open(logo_path) as logo:
                logo = logo.convert("RGBA")
                # Fit to ~50px tall, preserve aspect
                ratio = (height - 16) / max(1, logo.height)
                lw = int(logo.width * ratio)
                logo = logo.resize((lw, height - 16), Image.LANCZOS)
                img.paste(logo, (10, 8), logo)
                text_x = 10 + lw + 10
        except Exception as exc:
            logger.warning("channel_bug: logo load failed (%s): %s", logo_path, exc)
    # Channel name to the right of logo
    try:
        f = ImageFont.truetype(
            os.path.join(os.path.dirname(__file__), "..", "resources", "fonts",
                         "Inter-Bold.ttf"), 22)
    except Exception:
        f = ImageFont.load_default()
    name = (channel_name or "KAIZER NEWS")[:20]
    d.text((text_x, height // 2 - 12), name, font=f, fill=(255, 255, 255, 240))
    img.save(out_path, "PNG")
    return out_path


# ── Sidebar placeholder (Phase 3 replaces this with a real carousel) ─────────

def make_sidebar_placeholder(
    image_path: Optional[str],
    out_path: str,
    *,
    width: int = _SIDE_W,
    height: int = _SIDE_H,
) -> str:
    """Build a static sidebar placeholder PNG. Either the supplied image
    cropped to fit or a flat dark blue panel."""
    from PIL import Image
    img = Image.new("RGB", (width, height), (8, 18, 40))
    if image_path and os.path.isfile(image_path):
        try:
            with Image.open(image_path) as src:
                src = src.convert("RGB")
                # Cover-fit: scale up + center-crop to fill the panel.
                src_ratio = src.width / max(1, src.height)
                tgt_ratio = width / height
                if src_ratio > tgt_ratio:
                    # Source is wider — scale by height
                    nh = height
                    nw = int(src.width * (height / src.height))
                else:
                    nw = width
                    nh = int(src.height * (width / src.width))
                src = src.resize((nw, nh), Image.LANCZOS)
                left = (nw - width) // 2
                top = (nh - height) // 2
                src = src.crop((left, top, left + width, top + height))
                img.paste(src, (0, 0))
        except Exception as exc:
            logger.warning("sidebar_placeholder: image load failed (%s): %s",
                           image_path, exc)
    img.save(out_path, "PNG")
    return out_path


# ── Compositor ───────────────────────────────────────────────────────────────

def _resolve_ffmpeg_bin(ffmpeg_bin: Optional[str]) -> str:
    if ffmpeg_bin:
        return ffmpeg_bin
    try:
        from pipeline_core.pipeline import FFMPEG_BIN
        return FFMPEG_BIN
    except Exception:
        import shutil
        return shutil.which("ffmpeg") or "ffmpeg"


def compose_bulletin_story(
    story_clip_path: str,
    story_meta: StoryMeta,
    out_path: str,
    *,
    sidebar_path: str,
    ticker_path: str,
    channel_bug_path: Optional[str] = None,
    font_path: Optional[str] = None,
    sidebar_is_video: bool = False,
    ticker_speed_px_s: float = 200.0,
    width: int = _W,
    height: int = _H,
    ffmpeg_bin: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> str:
    """Compose one TV9-style story segment.

    Inputs:
      - 0:v / 0:a → main story video
      - 1:v       → sidebar (video if ``sidebar_is_video`` else looped PNG)
      - 2:v       → lower-third PNG (rendered fresh per story)
      - 3:v       → ticker PNG (wide; same one for all stories)
      - 4:v       → channel-bug PNG (optional; overlay only when supplied)

    Returns the output path on success. Raises ``RuntimeError`` if FFmpeg
    fails — the caller decides whether to fall back to bare slice.
    """
    ffmpeg_bin = _resolve_ffmpeg_bin(ffmpeg_bin)

    # Render the per-story lower-third
    work_dir = work_dir or os.path.dirname(os.path.abspath(out_path))
    os.makedirs(work_dir, exist_ok=True)
    lt_path = os.path.join(work_dir, f"_lt_{story_meta.story_index:02d}.png")
    render_lower_third(story_meta, font_path, lt_path)

    # Build the FFmpeg command
    cmd: list[str] = [ffmpeg_bin, "-y", "-i", story_clip_path]
    if sidebar_is_video:
        cmd += ["-i", sidebar_path]
    else:
        cmd += ["-loop", "1", "-i", sidebar_path]
    cmd += ["-loop", "1", "-i", lt_path]
    cmd += ["-loop", "1", "-i", ticker_path]
    has_bug = bool(channel_bug_path and os.path.isfile(channel_bug_path))
    if has_bug:
        cmd += ["-loop", "1", "-i", channel_bug_path]

    # Filter graph
    main_w, main_h = _MAIN_W, _MAIN_H
    side_w, side_h = _SIDE_W, _SIDE_H
    lt_h = _LT_H
    bug_pad_x, bug_pad_y = 30, 30
    lt_y = height - lt_h - _TICKER_H        # lower-third sits above the ticker
    ticker_y = height - _TICKER_H

    fc_lines = [
        f"[0:v]scale={main_w}:{main_h}:force_original_aspect_ratio=increase,"
        f"crop={main_w}:{main_h},setsar=1[main_v]",
        f"[1:v]scale={side_w}:{side_h}:force_original_aspect_ratio=increase,"
        f"crop={side_w}:{side_h},setsar=1[side_v]",
        f"[main_v][side_v]hstack=inputs=2[top_row]",
        f"[top_row]pad={width}:{height}:30:0:black[stage_bg]",
        f"[stage_bg][2:v]overlay="
        f"x='if(lt(t,0.4),-w+w*t/0.4,0)':"
        f"y={lt_y}:format=auto[stage_lt]",
        f"[stage_lt][3:v]overlay="
        f"x='W-mod(t*{ticker_speed_px_s:.1f}\\,w+W)':"
        f"y={ticker_y}:format=auto[stage_ticker]",
    ]
    if has_bug:
        fc_lines.append(
            f"[stage_ticker][4:v]overlay="
            f"x=W-w-{bug_pad_x}:y={bug_pad_y}:format=auto[outv]"
        )
        out_label = "outv"
    else:
        out_label = "stage_ticker"

    filter_complex = ";".join(fc_lines)

    cmd += [
        "-filter_complex", filter_complex,
        "-map", f"[{out_label}]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-shortest",
        "-movflags", "+faststart",
        out_path,
    ]

    logger.debug("compose_bulletin_story: cmd=%s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").splitlines()[-20:])
        raise RuntimeError(
            f"FFmpeg compose_bulletin_story failed (rc={proc.returncode}): {tail}"
        )

    # Best-effort cleanup of the per-story lower-third PNG
    try:
        os.remove(lt_path)
    except Exception:
        pass

    return out_path


# ── Phase 4 — PiP / video-in-video inset ─────────────────────────────────────

def pick_pip_source(
    clips: list[dict],
    current_index: int,
    *,
    min_seconds: float = 4.0,
    max_lookahead: int = 1,
) -> Optional[tuple[str, float, float]]:
    """Select an inset source for the PiP at *current_index*.

    Strategy: take the next story's first 8 seconds as B-roll. Returns
    ``(raw_path, start_s, dur_s)`` or ``None`` if no suitable source.
    """
    n = len(clips)
    if n == 0:
        return None
    for offset in range(1, max_lookahead + 2):
        nxt = current_index + offset
        if nxt >= n:
            break
        c = clips[nxt]
        raw = c.get("raw_path") or ""
        dur = float(c.get("duration_sec") or 0.0)
        if raw and os.path.isfile(raw) and dur >= min_seconds:
            return (raw, 0.0, min(8.0, dur))
    return None


def compose_pip_story(
    story_clip_path: str,
    story_meta: StoryMeta,
    out_path: str,
    *,
    pip_clip_path: str,
    pip_start_s: float = 0.0,
    pip_duration_s: float = 8.0,
    sidebar_path: str,
    ticker_path: str,
    channel_bug_path: Optional[str] = None,
    font_path: Optional[str] = None,
    sidebar_is_video: bool = False,
    ticker_speed_px_s: float = 200.0,
    width: int = _W,
    height: int = _H,
    ffmpeg_bin: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> str:
    """Same as :func:`compose_bulletin_story` but with a 320×180 PiP inset
    in the top-right corner of the main video area (under the channel bug).

    The inset is enabled only for ``[0, pip_duration_s)`` so it disappears
    automatically partway through long stories.
    """
    ffmpeg_bin = _resolve_ffmpeg_bin(ffmpeg_bin)
    work_dir = work_dir or os.path.dirname(os.path.abspath(out_path))
    os.makedirs(work_dir, exist_ok=True)

    lt_path = os.path.join(work_dir, f"_lt_{story_meta.story_index:02d}.png")
    render_lower_third(story_meta, font_path, lt_path)

    cmd: list[str] = [ffmpeg_bin, "-y", "-i", story_clip_path]
    # PiP source — clip the requested window so we only re-encode what's needed
    cmd += [
        "-ss", f"{pip_start_s:.3f}",
        "-t", f"{pip_duration_s:.3f}",
        "-i", pip_clip_path,
    ]
    if sidebar_is_video:
        cmd += ["-i", sidebar_path]
    else:
        cmd += ["-loop", "1", "-i", sidebar_path]
    cmd += ["-loop", "1", "-i", lt_path]
    cmd += ["-loop", "1", "-i", ticker_path]
    has_bug = bool(channel_bug_path and os.path.isfile(channel_bug_path))
    if has_bug:
        cmd += ["-loop", "1", "-i", channel_bug_path]

    pip_w, pip_h = 320, 180
    pip_x = _MAIN_W - pip_w - 30
    pip_y = _BUG_H + 50           # below the channel bug
    main_w, main_h = _MAIN_W, _MAIN_H
    side_w, side_h = _SIDE_W, _SIDE_H
    lt_h = _LT_H
    lt_y = height - lt_h - _TICKER_H
    ticker_y = height - _TICKER_H

    fc_lines = [
        f"[0:v]scale={main_w}:{main_h}:force_original_aspect_ratio=increase,"
        f"crop={main_w}:{main_h},setsar=1[main_v]",
        f"[1:v]scale={pip_w}:{pip_h}:force_original_aspect_ratio=increase,"
        f"crop={pip_w}:{pip_h},setsar=1,"
        f"pad={pip_w + 12}:{pip_h + 12}:6:6:white,setsar=1[pip_v]",
        f"[main_v][pip_v]overlay=x={pip_x}:y={pip_y}:"
        f"enable='lt(t,{pip_duration_s:.3f})'[main_with_pip]",
        f"[2:v]scale={side_w}:{side_h}:force_original_aspect_ratio=increase,"
        f"crop={side_w}:{side_h},setsar=1[side_v]",
        f"[main_with_pip][side_v]hstack=inputs=2[top_row]",
        f"[top_row]pad={width}:{height}:30:0:black[stage_bg]",
        f"[stage_bg][3:v]overlay="
        f"x='if(lt(t,0.4),-w+w*t/0.4,0)':"
        f"y={lt_y}:format=auto[stage_lt]",
        f"[stage_lt][4:v]overlay="
        f"x='W-mod(t*{ticker_speed_px_s:.1f}\\,w+W)':"
        f"y={ticker_y}:format=auto[stage_ticker]",
    ]
    if has_bug:
        fc_lines.append(
            f"[stage_ticker][5:v]overlay="
            f"x=W-w-30:y=30:format=auto[outv]"
        )
        out_label = "outv"
    else:
        out_label = "stage_ticker"

    filter_complex = ";".join(fc_lines)
    cmd += [
        "-filter_complex", filter_complex,
        "-map", f"[{out_label}]",
        "-map", "0:a?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-shortest",
        "-movflags", "+faststart",
        out_path,
    ]

    logger.debug("compose_pip_story: cmd=%s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").splitlines()[-20:])
        raise RuntimeError(
            f"FFmpeg compose_pip_story failed (rc={proc.returncode}): {tail}"
        )

    try:
        os.remove(lt_path)
    except Exception:
        pass

    return out_path
