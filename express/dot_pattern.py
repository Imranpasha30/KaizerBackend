"""Dot-pattern texture overlay generator.

Port of teammate's ``ensureDotPatternPNG``. Generates a single 1080×960
PNG (transparent + tiny white dots @ 15% opacity, on a 32px grid)
that the news_panel filter overlays on top of the solid panel color
so the panel doesn't look flat / dull.

Strategy:
  1. Build the SVG with a tiled <pattern> element.
  2. Rasterize via rsvg-convert if available.
  3. Fallback: rasterize via Pillow by drawing dots directly.

The output is cached at ``BACKEND/output/_express/dot_pattern.png`` —
idempotent, only re-generated if missing or zero-bytes.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from . import telugu_title    # for rsvg_binary() reuse


_BACKEND_DIR = Path(__file__).resolve().parent.parent
_CACHE_DIR = _BACKEND_DIR / "output" / "_express"
_OUT_PATH = _CACHE_DIR / "dot_pattern.png"


_W, _H = 1080, 960
_DOT_R = 3
_SPACING = 32
_OPACITY = 0.15


def _svg() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_W}" height="{_H}" '
        f'viewBox="0 0 {_W} {_H}">\n'
        '  <defs>\n'
        f'    <pattern id="dots" x="0" y="0" width="{_SPACING}" height="{_SPACING}" '
        'patternUnits="userSpaceOnUse">\n'
        f'      <circle cx="{_SPACING // 2}" cy="{_SPACING // 2}" r="{_DOT_R}" '
        f'fill="white" fill-opacity="{_OPACITY}" />\n'
        '    </pattern>\n'
        '  </defs>\n'
        f'  <rect width="{_W}" height="{_H}" fill="url(#dots)" />\n'
        '</svg>'
    )


def _render_via_rsvg(svg_path: str, out_path: str) -> bool:
    bin_path = telugu_title.rsvg_binary()
    if not bin_path:
        return False
    try:
        proc = subprocess.run(
            [bin_path, "-f", "png", "-o", out_path, svg_path],
            capture_output=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and os.path.isfile(out_path)


def _render_via_pillow(out_path: str) -> bool:
    """Direct Pillow rasterizer — draws dots on a transparent canvas
    without going through SVG. Works on any Python install that has
    Pillow."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False
    img = Image.new("RGBA", (_W, _H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    alpha = max(1, int(_OPACITY * 255))
    color = (255, 255, 255, alpha)
    cx0 = _SPACING // 2
    cy0 = _SPACING // 2
    for y in range(cy0, _H, _SPACING):
        for x in range(cx0, _W, _SPACING):
            draw.ellipse(
                [x - _DOT_R, y - _DOT_R, x + _DOT_R, y + _DOT_R],
                fill=color,
            )
    try:
        img.save(out_path, "PNG")
    except OSError:
        return False
    return True


def ensure_dot_pattern() -> Optional[str]:
    """Idempotently generate (or re-use) the dot-pattern PNG.
    Returns the absolute path on success, or None if rasterization
    failed via both paths (very rare — Pillow ships with Kaizer's
    requirements)."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_str = str(_OUT_PATH)

    if _OUT_PATH.is_file() and _OUT_PATH.stat().st_size > 1000:
        return out_str

    svg_path = out_str + ".svg"
    try:
        Path(svg_path).write_text(_svg(), encoding="utf-8")
    except OSError as exc:
        print(f"[dot_pattern] svg write failed: {exc}")
        svg_path = ""

    ok = False
    if svg_path:
        ok = _render_via_rsvg(svg_path, out_str)
    if not ok:
        ok = _render_via_pillow(out_str)

    if svg_path:
        try: Path(svg_path).unlink(missing_ok=True)
        except OSError: pass

    if ok and _OUT_PATH.is_file():
        return out_str
    print("[dot_pattern] rasterize failed via both rsvg + Pillow")
    return None
