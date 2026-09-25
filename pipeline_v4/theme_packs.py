"""THEME PACKS — the visual skin of the BUILT-IN render path.

Operator's model (2026-07-13): "layout is the placeholder, template is
the theme." Layouts decide WHERE things sit (layout_library + the live
per-story/per-moment engine); a ThemePack decides how it all LOOKS —
the backdrop plate under the tiles, the tile frame colour, the ticker
colourway and the accent. Themes never move placements, so every layout
operation and push animation runs unchanged inside them.

The four packs mirror the first-party premium templates (ids 36-39) so
the same design language exists both as a fixed template AND as a theme
for the dynamic path. Backdrops are painted NATIVELY with PIL —
deterministic (fixed star seed, no timestamps) so the per-story cache
fingerprint stays stable across runs.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Bump when a painter changes — backdrop files are cached by key+version
# and the render fingerprint includes the version (never stale plates).
THEME_VERSION = "t1"

W, H = 1920, 1080


@dataclass(frozen=True)
class ThemePack:
    key: str
    label: str
    used_for: str
    tile_border: str      # ffmpeg colour for the tile frames
    ticker_bg: str        # ticker strip colourway (hex)
    accent: str           # brand accent (hex) — straps / future typography
    bg_color: str         # flat fallback if the plate can't render


THEMES: dict[str, ThemePack] = {t.key: t for t in (
    ThemePack("obsidian", "Obsidian Editorial",
              "Dark newspaper-luxury: near-black paper, hairline gold",
              tile_border="#c9a227", ticker_bg="#c9a227",
              accent="#e6c96a", bg_color="#0b0b0d"),
    ThemePack("ivory", "Ivory Light",
              "Bright airy premium: warm ivory, coral accents",
              tile_border="#ffffff", ticker_bg="#e8604c",
              accent="#e8604c", bg_color="#f7f4ee"),
    ThemePack("nightline", "Nightline Horizon",
              "Dusk gradient with a glowing horizon — prime-time",
              tile_border="#a9bcff", ticker_bg="#6f8cff",
              accent="#ffb454", bg_color="#0a0c26"),
    ThemePack("verde", "Studio Verde",
              "Deep emerald cinema with mint light",
              tile_border="#3ddc97", ticker_bg="#3ddc97",
              accent="#ffd98e", bg_color="#062019"),
)}


def get_theme(key: str) -> Optional[ThemePack]:
    return THEMES.get((key or "").strip().lower())


# ── backdrop painters (pure PIL, deterministic) ──────────────────────

def _vgrad(size, stops):
    """Vertical gradient: stops = [(pos0..1, (r,g,b)), …] sorted."""
    from PIL import Image
    w, h = size
    strip = Image.new("RGB", (1, h))
    px = strip.load()
    for y in range(h):
        p = y / max(1, h - 1)
        lo = stops[0]
        hi = stops[-1]
        for i in range(len(stops) - 1):
            if stops[i][0] <= p <= stops[i + 1][0]:
                lo, hi = stops[i], stops[i + 1]
                break
        span = max(1e-6, hi[0] - lo[0])
        f = (p - lo[0]) / span
        px[0, y] = tuple(int(lo[1][c] + (hi[1][c] - lo[1][c]) * f)
                         for c in range(3))
    return strip.resize((w, h))


def _radial(size, center, radius, color, peak=1.0):
    """Soft radial glow as an L-mask scaled alpha layer."""
    from PIL import Image, ImageDraw, ImageFilter
    w, h = size
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    cx, cy = center
    d.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
              fill=int(255 * peak))
    m = m.filter(ImageFilter.GaussianBlur(radius * 0.55))
    layer = Image.new("RGB", (w, h), color)
    return layer, m


def _paint_obsidian(img):
    from PIL import Image, ImageDraw
    base = _vgrad(img.size, [(0.0, (35, 33, 38)), (0.46, (19, 19, 22)),
                             (1.0, (10, 10, 12))])
    img.paste(base)
    layer, m = _radial(img.size, (W // 2, -220), 900, (46, 42, 52), 0.8)
    img.paste(layer, (0, 0), m)
    d = ImageDraw.Draw(img, "RGBA")
    for x in range(0, W, 118):                      # faint column rules
        d.line([(x, 0), (x, H)], fill=(255, 255, 255, 10))
    d.rectangle([0, 0, W, 5], fill=(201, 162, 39, 190))   # gold top rule
    d.line([(64, 128), (W - 64, 128)], fill=(255, 255, 255, 24))


def _paint_ivory(img):
    from PIL import Image, ImageDraw
    base = _vgrad(img.size, [(0.0, (247, 244, 238)), (1.0, (238, 232, 222))])
    img.paste(base)
    layer, m = _radial(img.size, (int(W * 0.82), 0), 820, (253, 240, 228), 0.9)
    img.paste(layer, (0, 0), m)
    layer, m = _radial(img.size, (0, H), 760, (231, 224, 212), 0.8)
    img.paste(layer, (0, 0), m)
    d = ImageDraw.Draw(img, "RGBA")
    for x in range(0, W, 24):
        d.line([(x, 0), (x, H)], fill=(29, 27, 24, 4))
    for y in range(0, H, 24):
        d.line([(0, y), (W, y)], fill=(29, 27, 24, 4))


def _paint_nightline(img):
    from PIL import Image, ImageDraw, ImageFilter
    base = _vgrad(img.size, [(0.0, (10, 12, 38)), (0.46, (23, 26, 58)),
                             (0.74, (10, 11, 32)), (1.0, (7, 8, 26))])
    img.paste(base)
    rng = random.Random(42)                          # deterministic stars
    d = ImageDraw.Draw(img, "RGBA")
    for _ in range(220):
        x, y = rng.randrange(W), rng.randrange(int(H * 0.58))
        a = rng.randrange(40, 130)
        d.point((x, y), fill=(255, 255, 255, a))
    hy = int(H * 0.62)                               # glowing horizon
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.line([(0, hy), (W, hy)], fill=(111, 140, 255, 230), width=3)
    glow = glow.filter(ImageFilter.GaussianBlur(26))
    img.paste(Image.new("RGB", img.size, (111, 140, 255)), (0, 0),
              glow.split()[3])
    d.line([(0, hy), (W, hy)], fill=(169, 188, 255, 220), width=2)


def _paint_verde(img):
    from PIL import ImageDraw
    base = _vgrad(img.size, [(0.0, (12, 59, 46)), (1.0, (6, 32, 25))])
    img.paste(base)
    layer, m = _radial(img.size, (int(W * 0.18), 0), 900, (20, 82, 63), 0.85)
    img.paste(layer, (0, 0), m)
    layer, m = _radial(img.size, (W, H), 820, (10, 53, 39), 0.85)
    img.paste(layer, (0, 0), m)
    d = ImageDraw.Draw(img, "RGBA")
    for i in range(-H, W + H, 46):                   # soft diagonal mesh
        d.line([(i, 0), (i + H, H)], fill=(143, 240, 198, 6))
        d.line([(i + H, 0), (i, H)], fill=(143, 240, 198, 4))


_PAINTERS = {"obsidian": _paint_obsidian, "ivory": _paint_ivory,
             "nightline": _paint_nightline, "verde": _paint_verde}


def ensure_backdrop(key: str, work_dir) -> Optional[str]:
    """The theme's full-canvas backdrop plate (1920x1080 PNG), painted
    once per work_dir and reused (deterministic content; filename carries
    THEME_VERSION so painter changes can never serve a stale plate).
    None for unknown themes or on any failure — the composer falls back
    to the pack's flat bg colour (fail-soft, never blocks a render)."""
    t = get_theme(key)
    if not t:
        return None
    try:
        out = Path(work_dir) / f"_theme_{t.key}_{THEME_VERSION}.png"
        if out.is_file() and out.stat().st_size > 0:
            return str(out)
        from PIL import Image
        img = Image.new("RGB", (W, H), t.bg_color)
        _PAINTERS[t.key](img)
        img.save(str(out), "PNG")
        return str(out)
    except Exception as exc:
        print(f"[v4/theme] backdrop {key!r} failed (soft): {exc}", flush=True)
        return None
