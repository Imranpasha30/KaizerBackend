"""Telugu / Indic title PNG renderer.

Two strategies, in order of preference, matching the teammate's
``renderTeluguTitlePNGViaRsvg`` → ``renderTeluguTitlePNG`` fallback:

1. ``rsvg-convert`` (librsvg). Best quality on any platform because
   librsvg uses Pango + harfbuzz for proper Indic shaping (conjuncts,
   vowel marks, ZWJ/ZWNJ handling). The teammate's preferred path.
2. **Pillow (PIL) fallback.** Works without any extra install — uses
   FreeType directly. Shaping is decent for most modern Telugu fonts
   but doesn't run Pango's full layout engine, so very complex
   conjunct stacks may render less cleanly. Good enough for 95% of
   news-headline copy.

Both paths support the same bomb-word highlighting: words wrapped in
``*asterisks*`` are rendered in yellow (#fde047) while everything
else is rendered in white. The thick black stroke (paint-order =
stroke fill) gives legibility on any panel color.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


# ── Font + binary discovery ────────────────────────────────────────

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_TELUGU_FONT = _BACKEND_DIR / "resources" / "fonts" / "NotoSansTelugu-Bold.ttf"


def telugu_font_path() -> Optional[str]:
    """Locate the bundled Noto Sans Telugu Bold. Returns absolute
    path or None when missing (callers degrade to system default)."""
    if _DEFAULT_TELUGU_FONT.is_file():
        return str(_DEFAULT_TELUGU_FONT)
    return None


def rsvg_binary() -> Optional[str]:
    """Find ``rsvg-convert`` — honour an explicit env override, then
    PATH, then a few well-known Windows locations. Returns the path
    or None if not available."""
    override = os.environ.get("KAIZER_RSVG_PATH", "").strip()
    if override and os.path.isfile(override):
        return override
    found = shutil.which("rsvg-convert")
    if found:
        return found
    # Common Windows install paths we'll probe before giving up.
    candidates = [
        r"C:\Program Files\GIMP 2\bin\rsvg-convert.exe",
        r"C:\msys64\mingw64\bin\rsvg-convert.exe",
        r"C:\ProgramData\chocolatey\bin\rsvg-convert.exe",
        r"C:\tools\librsvg\bin\rsvg-convert.exe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


# ── XML escaping ───────────────────────────────────────────────────

_XML_TRANS = {
    "&": "&amp;",  "<": "&lt;",  ">": "&gt;",  '"': "&quot;",  "'": "&#39;",
}


def _xml_escape(s: str) -> str:
    out = []
    for ch in (s or ""):
        out.append(_XML_TRANS.get(ch, ch))
    return "".join(out)


# ── Bomb-word segment parser (shared by both paths) ────────────────

_BOMB_RE = re.compile(r"\*([^*]+)\*")


def _parse_segments(line: str) -> list[dict]:
    """Split a line into ``[{emph: bool, text: str}, ...]`` parts.

    Direct port of teammate's parseSegments. Same logic: ``*word*``
    markers become emphasis segments, and adjacent non-whitespace
    boundaries get a synthesised space so words don't run together
    across the color boundary.
    """
    parts: list[dict] = []
    cursor = 0
    for m in _BOMB_RE.finditer(line):
        if m.start() > cursor:
            parts.append({"emph": False, "text": line[cursor:m.start()]})
        parts.append({"emph": True, "text": m.group(1)})
        cursor = m.end()
    if cursor < len(line):
        parts.append({"emph": False, "text": line[cursor:]})
    if not parts:
        parts.append({"emph": False, "text": line})

    # Insert space at adjacency points where neither side has whitespace.
    for i in range(len(parts) - 1):
        left  = parts[i]["text"]
        right = parts[i + 1]["text"]
        if left and right and not left[-1].isspace() and not right[0].isspace():
            parts[i] = {**parts[i], "text": left + " "}
    return parts


# ── Strategy 1: rsvg-convert ───────────────────────────────────────

def _render_via_rsvg(
    *,
    text: str,
    output_path: str,
    font_size: int,
    font_family: str,
    font_weight: str,
    fill_color: str,
    emphasis_color: str,
    stroke_color: str,
    stroke_width: int,
    timeout_s: int = 30,
) -> Optional[dict]:
    """Generate an SVG, hand it to ``rsvg-convert -f png``.
    Matches the teammate's SVG output character-for-character so
    rendering is identical when both have librsvg installed.

    Returns ``{path, height}`` on success, None otherwise."""
    bin_path = rsvg_binary()
    if not bin_path:
        return None

    lines = str(text).split("\n")
    line_height = round(font_size * 1.18)
    padding = max(20, stroke_width * 2)
    svg_w = 1080
    svg_h = padding * 2 + line_height * len(lines)

    text_elements: list[str] = []
    for i, line in enumerate(lines):
        y = padding + font_size + i * line_height
        segs = _parse_segments(line)
        tspans = "".join(
            f'<tspan xml:space="preserve" '
            f'fill="{p["emph"] and emphasis_color or fill_color}">'
            f'{_xml_escape(p["text"])}</tspan>'
            for p in segs
        )
        text_elements.append(
            f'<text x="{svg_w // 2}" y="{y}" '
            'xml:space="preserve" '
            f'font-family="{_xml_escape(font_family)}" '
            f'font-weight="{font_weight}" '
            f'font-size="{font_size}" '
            'text-anchor="middle" '
            f'stroke="{stroke_color}" stroke-width="{stroke_width}" stroke-linejoin="round" '
            f'paint-order="stroke fill">{tspans}</text>'
        )

    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
        f'viewBox="0 0 {svg_w} {svg_h}">\n  '
        + "\n  ".join(text_elements)
        + "\n</svg>"
    )

    svg_path = output_path + ".svg"
    try:
        Path(svg_path).write_text(svg, encoding="utf-8")
    except OSError as exc:
        print(f"[telugu_title] svg write failed: {exc}")
        return None

    try:
        proc = subprocess.run(
            [bin_path, "-f", "png", "-o", output_path, svg_path],
            capture_output=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        try: Path(svg_path).unlink(missing_ok=True)
        except OSError: pass
        return None
    finally:
        try: Path(svg_path).unlink(missing_ok=True)
        except OSError: pass

    if proc.returncode != 0:
        print(f"[telugu_title] rsvg-convert failed: "
              f"{(proc.stderr or b'').decode('utf-8', errors='replace')[-200:]}")
        return None

    if not os.path.isfile(output_path):
        return None
    return {"path": output_path, "height": svg_h}


# ── Strategy 2: Pillow fallback ────────────────────────────────────

def _render_via_pillow(
    *,
    text: str,
    output_path: str,
    font_size: int,
    font_path: Optional[str],
    fill_color: tuple,
    emphasis_color: tuple,
    stroke_color: tuple,
    stroke_width: int,
) -> Optional[dict]:
    """Draw the title via Pillow. Handles bomb-word coloring by
    rendering each segment as its own draw call, advancing x manually.
    Telugu shaping quality is FreeType-only (no Pango/harfbuzz), but
    on modern Noto Sans Telugu Bold it's close enough for headlines."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("[telugu_title] Pillow not installed — `pip install Pillow`")
        return None

    fp = font_path or telugu_font_path()
    if not fp or not os.path.isfile(fp):
        print(f"[telugu_title] font missing at {fp}")
        return None

    try:
        font = ImageFont.truetype(fp, font_size)
    except OSError as exc:
        print(f"[telugu_title] couldn't load font: {exc}")
        return None

    lines = str(text).split("\n")
    line_height = round(font_size * 1.18)
    padding = max(20, stroke_width * 2)
    svg_w = 1080
    svg_h = padding * 2 + line_height * len(lines)

    img = Image.new("RGBA", (svg_w, svg_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for i, line in enumerate(lines):
        y = padding + i * line_height
        segs = _parse_segments(line)
        # Measure each segment to compute total width for centering.
        seg_widths: list[int] = []
        for p in segs:
            try:
                bbox = draw.textbbox((0, 0), p["text"], font=font)
                seg_widths.append(bbox[2] - bbox[0])
            except (TypeError, ValueError):
                seg_widths.append(len(p["text"]) * (font_size // 2))
        total_w = sum(seg_widths)
        x = (svg_w - total_w) // 2

        for p, w in zip(segs, seg_widths):
            color = emphasis_color if p["emph"] else fill_color
            draw.text(
                (x, y), p["text"], font=font, fill=color,
                stroke_width=stroke_width, stroke_fill=stroke_color,
            )
            x += w

    try:
        img.save(output_path, "PNG")
    except OSError as exc:
        print(f"[telugu_title] PNG save failed: {exc}")
        return None
    return {"path": output_path, "height": svg_h}


# ── Hex → RGBA helper ──────────────────────────────────────────────

def _hex_to_rgba(spec: str) -> tuple:
    """Accept ``#rrggbb`` / ``#rrggbbaa`` / a named color from a tiny
    set; return ``(r, g, b, a)``. Pillow needs tuples, not hex
    strings, when ``stroke_fill`` is involved."""
    spec = (spec or "").strip().lower()
    named = {"white": "#ffffff", "black": "#000000", "yellow": "#ffff00"}
    if spec in named:
        spec = named[spec]
    if spec.startswith("#"):
        s = spec[1:]
        if len(s) == 6:
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), 255)
        if len(s) == 8:
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16),
                    int(s[6:8], 16))
    # Default: opaque white
    return (255, 255, 255, 255)


# ── Public entry point ─────────────────────────────────────────────

def render_title_png(
    *,
    text: str,
    output_path: str,
    font_size: int = 82,
    font_family: str = "Noto Sans Telugu, NotoSansTelugu, Kohinoor Telugu, Telugu Sangam MN, sans-serif",
    font_weight: str = "bold",
    fill_color: str = "#ffffff",
    emphasis_color: str = "#fde047",
    stroke_color: str = "#000000",
    stroke_width: int = 9,
) -> Optional[dict]:
    """Render the title PNG. Returns ``{path, height}`` or None.

    Tries rsvg-convert first, then falls back to Pillow. Both paths
    produce a transparent-background PNG sized at canvas-width (1080)
    so the news layout filter can overlay it at the panel top
    without scaling.
    """
    if not text or not str(text).strip():
        return None

    res = _render_via_rsvg(
        text=text, output_path=output_path,
        font_size=font_size, font_family=font_family, font_weight=font_weight,
        fill_color=fill_color, emphasis_color=emphasis_color,
        stroke_color=stroke_color, stroke_width=stroke_width,
    )
    if res:
        return res

    # Fallback. Pillow can't easily emulate the SVG paint-order tricks,
    # so the stroke is slightly thinner to compensate for FreeType's
    # different stroke rendering.
    return _render_via_pillow(
        text=text.replace("*", "") if not _BOMB_RE.search(text) else text,
        output_path=output_path,
        font_size=font_size,
        font_path=None,
        fill_color=_hex_to_rgba(fill_color),
        emphasis_color=_hex_to_rgba(emphasis_color),
        stroke_color=_hex_to_rgba(stroke_color),
        stroke_width=max(2, stroke_width - 3),
    )
