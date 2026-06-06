"""PIL-based text panel renderer.

Produces a transparent PNG for each text overlay. The PNG is then
fed to ffmpeg's overlay filter with timed visibility — keeping the
canvas pipeline language-agnostic (any text PIL can draw, we can
overlay).

Telugu / Hindi / Bengali / Tamil / Kannada all use the same renderer;
the right font is auto-picked from ``resources/fonts/`` based on the
unicode block of the input text. Same approach V1 uses.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_FONTS_DIR = _BACKEND_ROOT / "resources" / "fonts"


# Unicode-block → (regular-font-filename, bold-font-filename)
# Order matters: pick the first block that the text contains.
_LANG_FONTS: list[tuple[str, str, str]] = [
    # (regex matching a code-point range, regular ttf, bold ttf)
    (r"[ఀ-౿]", "NotoSansTelugu-Regular.ttf",   "NotoSansTelugu-Bold.ttf"),
    (r"[ऀ-ॿ]", "NotoSansDevanagari-Regular.ttf","NotoSansDevanagari-Bold.ttf"),
    (r"[ঀ-৿]", "NotoSansBengali-Regular.ttf",   "NotoSansBengali-Bold.ttf"),
    (r"[஀-௿]", "NotoSansTamil-Regular.ttf",     "NotoSansTamil-Bold.ttf"),
    (r"[ಀ-೿]", "NotoSansKannada-Regular.ttf",   "NotoSansKannada-Bold.ttf"),
]
_DEFAULT_FONT_REGULAR = "NotoSans-Regular.ttf"
_DEFAULT_FONT_BOLD    = "NotoSans-Bold.ttf"


def _pick_font_path(text: str, bold: bool) -> str:
    """Return the absolute path to the right font file for this text."""
    for pattern, reg, bld in _LANG_FONTS:
        if re.search(pattern, text):
            return str(_FONTS_DIR / (bld if bold else reg))
    return str(_FONTS_DIR / (_DEFAULT_FONT_BOLD if bold else _DEFAULT_FONT_REGULAR))


def _hex_to_rgba(hex_color: Optional[str], default_alpha: int = 255) -> tuple[int, int, int, int]:
    """'#RRGGBB' or '#RRGGBBAA' → (R, G, B, A)."""
    if not hex_color:
        return (0, 0, 0, 0)
    s = hex_color.strip().lstrip("#")
    if len(s) == 6:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), default_alpha)
    if len(s) == 8:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), int(s[6:8], 16))
    return (255, 255, 255, default_alpha)


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    """Greedy word-wrap so the text fits in ``max_w`` pixels.

    Falls back to character-wrap when a single 'word' is wider than
    max_w (long Devanagari conjuncts, URLs, etc.).
    """
    if not text:
        return []
    words = text.split(" ")
    lines: list[str] = []
    cur = ""
    for w in words:
        candidate = (cur + " " + w).strip()
        try:
            bbox = font.getbbox(candidate)
            width = bbox[2] - bbox[0]
        except Exception:
            width = len(candidate) * font.size // 2
        if width <= max_w:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            # Word alone too wide → char-wrap it
            if font.getbbox(w)[2] - font.getbbox(w)[0] > max_w:
                chunk = ""
                for ch in w:
                    nchunk = chunk + ch
                    if font.getbbox(nchunk)[2] - font.getbbox(nchunk)[0] > max_w and chunk:
                        lines.append(chunk)
                        chunk = ch
                    else:
                        chunk = nchunk
                cur = chunk
            else:
                cur = w
    if cur:
        lines.append(cur)
    return lines


def render_text_panel_png(
    *,
    text: str,
    out_path: str,
    width: int,
    height: int,
    font_size: int,
    fg_color: str = "#FFFFFF",
    bg_color: Optional[str] = None,
    bold: bool = True,
    padding_px: int = 24,
    align: str = "left",
    valign: str = "middle",
) -> str:
    """Render ``text`` into a PNG at ``out_path``. Returns the path.

    Transparent background when ``bg_color`` is None — perfect for
    overlays. Use a solid color for full-bleed panels like a
    lower-third strap or ticker.
    """
    font_path = _pick_font_path(text, bold=bold)
    try:
        font = ImageFont.truetype(font_path, size=font_size)
    except OSError:
        # Final fallback — Pillow's built-in default; doesn't shape
        # Telugu but at least produces SOMETHING so the pipeline
        # doesn't crash.
        font = ImageFont.load_default()

    img = Image.new("RGBA", (width, height), _hex_to_rgba(bg_color, default_alpha=0 if bg_color is None else 255))
    draw = ImageDraw.Draw(img)

    inner_w = max(1, width - 2 * padding_px)
    lines = _wrap_text(text, font, inner_w)
    line_h = int(font_size * 1.25)
    total_h = max(line_h, line_h * len(lines))

    if valign == "top":
        y0 = padding_px
    elif valign == "bottom":
        y0 = height - padding_px - total_h
    else:  # middle
        y0 = (height - total_h) // 2

    for i, ln in enumerate(lines):
        try:
            bbox = font.getbbox(ln)
            line_w = bbox[2] - bbox[0]
        except Exception:
            line_w = len(ln) * font_size // 2
        if align == "center":
            x = (width - line_w) // 2
        elif align == "right":
            x = width - padding_px - line_w
        else:
            x = padding_px
        draw.text(
            (x, y0 + i * line_h),
            ln,
            font=font,
            fill=_hex_to_rgba(fg_color, default_alpha=255),
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, "PNG", optimize=True)
    return out_path
