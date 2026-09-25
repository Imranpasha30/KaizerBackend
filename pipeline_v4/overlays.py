"""Overlay generator engine — the spec's ~190 graphics from 12 generators.

Every broadcast graphic (banners, bugs, stamps, HUD frames, locators,
progress chips, countdowns, info panels…) is produced by a parameterized
PIL generator, so "breaking_news_banner", "developing_story_banner" and
every CTA banner are ONE function with different rows in the REGISTRY —
never 190 hand-drawn assets. All text goes through PIL with the language
font (Indic-safe; ffmpeg drawtext is never used for text).

Output contract: every generator returns the path to a transparent PNG
sized for the given canvas; the compose layer overlays it with
enable-windows exactly like straps/spotlights. Fail-soft: None on any
error — a missing graphic never kills a render.

The REGISTRY at the bottom is the machine-readable catalog the Admin
"Editing Features" tab lists and previews.
"""
from __future__ import annotations

import os
from typing import Optional

# Broadcast palette
RED = (193, 18, 18, 255)
AMBER = (230, 140, 10, 255)
DARK = (10, 10, 14, 230)
WHITE = (245, 245, 247, 255)
GOLD = (212, 175, 55, 255)


def _font(font_path: Optional[str], size: int):
    """Scalable font, always: caller's font, else the bundled NotoSans,
    else PIL default. load_default() ignores ``size`` (tiny bitmap font),
    so falling back to it silently shrinks every graphic — the bundled
    fonts are the real default."""
    from PIL import ImageFont
    candidates = [font_path] if font_path else []
    res = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "resources", "fonts")
    candidates += [os.path.join(res, "NotoSans-Bold.ttf"),
                   os.path.join(res, "NotoSansTelugu-Bold.ttf")]
    for c in candidates:
        try:
            if c and os.path.isfile(c):
                return ImageFont.truetype(c, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)   # PIL >= 10
    except TypeError:
        return ImageFont.load_default()


def _text_size(draw, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1], box[1]


def _fail_soft(fn):
    def wrapped(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as exc:
            print(f"[v4/overlays] {fn.__name__} failed (soft): {exc}", flush=True)
            return None
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped


# ── 1) BANNER — full-width strip (breaking / developing / alert / CTA) ──

@_fail_soft
def banner(*, text: str, out_path: str, canvas_w: int = 1920,
           color: tuple = RED, sub: str = "", font_path: str = None) -> Optional[str]:
    from PIL import Image, ImageDraw
    H = 110 if not sub else 150
    img = Image.new("RGBA", (canvas_w, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, canvas_w, H], fill=(color[0], color[1], color[2], 235))
    d.rectangle([0, H - 6, canvas_w, H], fill=(255, 255, 255, 90))
    f = _font(font_path, 56)
    t = " ".join((text or "").upper().split())[:80]
    tw, th, ty = _text_size(d, t, f)
    d.text((40, (H - (24 if sub else 0) - th) // 2 - ty), t, font=f, fill=WHITE)
    if sub:
        fs = _font(font_path, 26)
        d.text((42, H - 40), sub[:120], font=fs, fill=(255, 255, 255, 220))
    img.save(out_path, "PNG")
    return out_path


# ── 2) BUG — small corner chip (LIVE / location / replay / handle) ──

@_fail_soft
def bug(*, text: str, out_path: str, dot: bool = False,
        color: tuple = DARK, font_path: str = None) -> Optional[str]:
    from PIL import Image, ImageDraw
    f = _font(font_path, 30)
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    tw, th, ty = _text_size(probe, text[:40], f)
    pad, dot_w = 16, (34 if dot else 0)
    W, H = tw + 2 * pad + dot_w, 52
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=8, fill=color)
    x = pad
    if dot:
        d.ellipse([x, H // 2 - 8, x + 16, H // 2 + 8], fill=RED)
        x += dot_w
    d.text((x, (H - th) // 2 - ty), text[:40], font=f, fill=WHITE)
    img.save(out_path, "PNG")
    return out_path


# ── 3) STAMP — rotated seal (EXCLUSIVE / CONFIDENTIAL / TOP SECRET) ──

@_fail_soft
def stamp(*, text: str, out_path: str, color: tuple = RED, angle: float = -12.0,
          font_path: str = None) -> Optional[str]:
    from PIL import Image, ImageDraw
    f = _font(font_path, 64)
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    t = " ".join((text or "").upper().split())[:24]
    tw, th, ty = _text_size(probe, t, f)
    pad = 26
    img = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([3, 3, img.width - 4, img.height - 4],
                outline=(color[0], color[1], color[2], 240), width=6)
    d.text((pad, pad - ty), t, font=f, fill=(color[0], color[1], color[2], 240))
    img = img.rotate(angle, expand=True, resample=Image.BICUBIC)
    img.save(out_path, "PNG")
    return out_path


# ── 4) LOCATOR / dateline + SOURCE attribution (small context chips) ──

@_fail_soft
def locator(*, text: str, out_path: str, live: bool = False,
            font_path: str = None) -> Optional[str]:
    return bug(text=(("LIVE  •  " if live else "") + text.upper()),
               out_path=out_path, dot=live, font_path=font_path)


@_fail_soft
def source_attribution(*, text: str, out_path: str,
                       font_path: str = None) -> Optional[str]:
    from PIL import Image, ImageDraw
    f = _font(font_path, 24)
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    tw, th, ty = _text_size(probe, text[:60], f)
    img = Image.new("RGBA", (tw + 24, 38), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, img.height], fill=(0, 0, 0, 150))
    d.text((12, (38 - th) // 2 - ty), text[:60], font=f, fill=(230, 230, 232, 230))
    img.save(out_path, "PNG")
    return out_path


# ── 5) PROGRESS — "Story 2 of 5" with a fill bar ─────────────────────

@_fail_soft
def progress(*, current: int, total: int, out_path: str, label: str = "Story",
             font_path: str = None) -> Optional[str]:
    from PIL import Image, ImageDraw
    W, H = 340, 64
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=10, fill=DARK)
    f = _font(font_path, 28)
    t = f"{label} {current} of {total}"
    tw, th, ty = _text_size(d, t, f)
    d.text((16, 8 - ty + 2), t, font=f, fill=WHITE)
    frac = max(0.0, min(1.0, current / max(1, total)))
    d.rounded_rectangle([16, H - 18, W - 16, H - 10], radius=4, fill=(255, 255, 255, 60))
    d.rounded_rectangle([16, H - 18, 16 + int((W - 32) * frac), H - 10],
                        radius=4, fill=RED)
    img.save(out_path, "PNG")
    return out_path


# ── 6) COUNTDOWN — big digits card ("3 STORIES" interstitials too) ──

@_fail_soft
def countdown(*, value: str, out_path: str, sub: str = "",
              canvas_w: int = 1920, canvas_h: int = 1080,
              font_path: str = None) -> Optional[str]:
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (canvas_w, canvas_h), (6, 6, 8, 255))
    d = ImageDraw.Draw(img)
    f = _font(font_path, int(canvas_h * 0.3))
    tw, th, ty = _text_size(d, str(value), f)
    d.text(((canvas_w - tw) // 2, (canvas_h - th) // 2 - ty - (40 if sub else 0)),
           str(value), font=f, fill=WHITE)
    if sub:
        fs = _font(font_path, 48)
        sw, sh, sy = _text_size(d, sub.upper(), fs)
        d.text(((canvas_w - sw) // 2, (canvas_h + th) // 2 + 10 - sy),
               sub.upper(), font=fs, fill=RED)
    img.save(out_path, "PNG")
    return out_path


# ── 7) FRAME / HUD — the "camera recording look" (full-canvas frame) ──

@_fail_soft
def frame_hud(*, style: str, out_path: str, canvas_w: int = 1920,
              canvas_h: int = 1080, label: str = "",
              font_path: str = None) -> Optional[str]:
    """Transparent full-canvas frame: 'viewfinder' (REC dot, corner
    brackets, timecode, battery), 'cctv' (CAM label, timestamp,
    scanlines), 'phone' (bezel), 'conference' (name tag + mute)."""
    from PIL import Image, ImageDraw
    W, H = canvas_w, canvas_h
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f = _font(font_path, 34)
    s = (style or "").strip().lower()
    if s == "viewfinder":
        L, T = 60, 50
        for (x0, y0, dx, dy) in ((L, T, 1, 1), (W - L, T, -1, 1),
                                 (L, H - T, 1, -1), (W - L, H - T, -1, -1)):
            d.line([x0, y0, x0 + dx * 70, y0], fill=WHITE, width=6)
            d.line([x0, y0, x0, y0 + dy * 70], fill=WHITE, width=6)
        d.ellipse([L + 20, T + 40, L + 44, T + 64], fill=RED)
        d.text((L + 56, T + 38), "REC", font=f, fill=WHITE)
        d.text((W - 300, T + 38), label or "00:00:14:08", font=f, fill=WHITE)
        d.rectangle([W - 130, H - T - 40, W - 70, H - T - 12], outline=WHITE, width=4)
        d.rectangle([W - 126, H - T - 36, W - 96, H - T - 16], fill=WHITE)
        d.rectangle([W - 70, H - T - 32, W - 62, H - T - 20], fill=WHITE)
    elif s == "cctv":
        for y in range(0, H, 6):
            d.line([0, y, W, y], fill=(0, 0, 0, 46), width=2)
        d.text((40, 30), label or "CAM-04", font=f, fill=(220, 255, 220, 235))
        d.text((40, H - 70), "2026-07-06  21:58:11", font=f, fill=(220, 255, 220, 235))
        d.ellipse([W - 90, 34, W - 66, 58], fill=RED)
    elif s == "phone":
        r, bw = 90, 26
        d.rounded_rectangle([0, 0, W - 1, H - 1], radius=r,
                            outline=(12, 12, 14, 255), width=bw)
        notch_w = int(W * 0.32)
        d.rounded_rectangle([(W - notch_w) // 2, 6, (W + notch_w) // 2, 40],
                            radius=16, fill=(12, 12, 14, 255))
    elif s == "conference":
        d.rounded_rectangle([30, H - 100, 30 + 420, H - 40], radius=8, fill=DARK)
        d.text((48, H - 92), label or "Guest Speaker", font=f, fill=WHITE)
        d.ellipse([30 + 380, H - 88, 30 + 412, H - 56], fill=(60, 60, 66, 255))
        d.line([30 + 388, H - 64, 30 + 404, H - 80], fill=RED, width=4)
        d.rectangle([0, 0, W - 1, H - 1], outline=(70, 70, 80, 200), width=3)
    elif s == "drone":
        fs = _font(font_path, 26)
        g = (140, 255, 170, 230)
        for cx, cy in ((60, 60), (W - 60, 60), (60, H - 60), (W - 60, H - 60)):
            d.line([cx - 26, cy, cx + 26, cy], fill=g, width=3)
            d.line([cx, cy - 26, cx, cy + 26], fill=g, width=3)
        d.text((100, 40), label or "DRONE 01", font=f, fill=g)
        d.text((100, H - 84), "ALT 120m   SPD 38km/h   GPS LOCK", font=fs, fill=g)
        d.ellipse([W // 2 - 40, H // 2 - 40, W // 2 + 40, H // 2 + 40],
                  outline=g, width=3)
    elif s == "bodycam":
        fs = _font(font_path, 26)
        d.rectangle([0, 0, W - 1, H - 1], outline=(200, 200, 210, 160), width=4)
        d.rectangle([30, 26, 360, 76], fill=DARK)
        d.ellipse([44, 40, 66, 62], fill=RED)
        d.text((80, 34), label or "BODYCAM 07", font=f, fill=WHITE)
        d.text((30, H - 66), "2026-07-06 23:41:07   UNIT-12", font=fs,
               fill=(230, 230, 235, 235))
    elif s == "dashcam":
        fs = _font(font_path, 26)
        d.rectangle([0, 0, W - 1, H - 1], outline=(220, 220, 230, 120), width=3)
        d.rectangle([W - 460, H - 76, W - 26, H - 26], fill=DARK)
        d.text((W - 444, H - 68), "62 km/h  NH-44  23:41", font=fs, fill=WHITE)
        d.text((30, 30), label or "DASHCAM", font=f, fill=(255, 220, 90, 240))
    else:
        raise ValueError(f"unknown frame style {style!r}")
    img.save(out_path, "PNG")
    return out_path


# ── 8) INFO PANEL — side box (fact check / stats / callout) ──────────

@_fail_soft
def info_panel(*, title: str, lines: list, out_path: str, w: int = 520,
               font_path: str = None) -> Optional[str]:
    from PIL import Image, ImageDraw
    ft, fl = _font(font_path, 36), _font(font_path, 28)
    H = 70 + 44 * min(len(lines), 8) + 20
    img = Image.new("RGBA", (w, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w - 1, H - 1], radius=10, fill=(10, 10, 14, 215))
    d.rectangle([0, 0, 8, H], fill=RED)
    d.text((26, 16), title[:36].upper(), font=ft, fill=WHITE)
    for i, ln in enumerate(lines[:8]):
        d.text((26, 70 + 44 * i), f"• {str(ln)[:44]}", font=fl,
               fill=(225, 225, 230, 235))
    img.save(out_path, "PNG")
    return out_path


# ── REGISTRY — the machine-readable catalog (admin tab reads this) ──

REGISTRY: list[dict] = [
    # family, id, label, used_for, generator kwargs (for the preview)
    {"family": "banner", "id": "breaking_news_banner", "label": "Breaking News Banner",
     "used_for": "Breaking stories — full-width red strip",
     "gen": "banner", "kwargs": {"text": "BREAKING NEWS", "sub": "Major development in ongoing story"}},
    {"family": "banner", "id": "developing_story_banner", "label": "Developing Story Banner",
     "used_for": "Ongoing coverage — amber strip",
     "gen": "banner", "kwargs": {"text": "DEVELOPING STORY", "color": AMBER}},
    {"family": "banner", "id": "alert_strip", "label": "Alert Strip",
     "used_for": "Warnings / emergencies",
     "gen": "banner", "kwargs": {"text": "WEATHER ALERT", "color": AMBER, "sub": "Stay tuned for updates"}},
    {"family": "banner", "id": "cta_banner", "label": "CTA Banner (subscribe/watch/join…)",
     "used_for": "Any call-to-action — one generator, many texts",
     "gen": "banner", "kwargs": {"text": "SUBSCRIBE FOR MORE", "color": (20, 90, 200, 255)}},
    {"family": "bug", "id": "live_bug", "label": "LIVE Bug",
     "used_for": "Live coverage chip with red dot",
     "gen": "bug", "kwargs": {"text": "LIVE", "dot": True}},
    {"family": "bug", "id": "location_bug", "label": "Location Bug",
     "used_for": "Where the footage is from",
     "gen": "bug", "kwargs": {"text": "NEW DELHI"}},
    {"family": "bug", "id": "replay_bug", "label": "Replay Bug",
     "used_for": "Sports replays",
     "gen": "bug", "kwargs": {"text": "REPLAY"}},
    {"family": "bug", "id": "social_handle_bug", "label": "Social Handle Bug",
     "used_for": "Channel handle on screen",
     "gen": "bug", "kwargs": {"text": "@kaizerx"}},
    {"family": "stamp", "id": "exclusive_stamp", "label": "EXCLUSIVE Stamp",
     "used_for": "Exclusive stories — rotated seal",
     "gen": "stamp", "kwargs": {"text": "EXCLUSIVE"}},
    {"family": "stamp", "id": "confidential_stamp", "label": "CONFIDENTIAL Stamp",
     "used_for": "Investigations / documents",
     "gen": "stamp", "kwargs": {"text": "CONFIDENTIAL"}},
    {"family": "stamp", "id": "top_secret_stamp", "label": "TOP SECRET Stamp",
     "used_for": "Investigations",
     "gen": "stamp", "kwargs": {"text": "TOP SECRET"}},
    {"family": "locator", "id": "dateline_locator", "label": "Dateline / Locator",
     "used_for": "LIVE • CITY chip, top corner",
     "gen": "locator", "kwargs": {"text": "Hyderabad", "live": True}},
    {"family": "attribution", "id": "source_attribution", "label": "Source Attribution",
     "used_for": "'Source: Reuters' / 'File footage' credit",
     "gen": "source_attribution", "kwargs": {"text": "Source: Reuters"}},
    {"family": "progress", "id": "story_progress", "label": "Story Progress",
     "used_for": "'Story 2 of 5' with fill bar",
     "gen": "progress", "kwargs": {"current": 2, "total": 5}},
    {"family": "countdown", "id": "countdown_card", "label": "Countdown Card",
     "used_for": "Trailer interstitials / event countdowns",
     "gen": "countdown", "kwargs": {"value": "3", "sub": "stories tonight"}},
    {"family": "frame_hud", "id": "viewfinder_hud", "label": "Camcorder Viewfinder HUD",
     "used_for": "Found footage / caught-on-tape look (REC + timecode + battery)",
     "gen": "frame_hud", "kwargs": {"style": "viewfinder"}},
    {"family": "frame_hud", "id": "cctv_hud", "label": "CCTV / Security Feed",
     "used_for": "Crime / investigation footage (CAM label + timestamp + scanlines)",
     "gen": "frame_hud", "kwargs": {"style": "cctv", "label": "CAM-04"}},
    {"family": "frame_hud", "id": "phone_frame", "label": "Phone Frame",
     "used_for": "Social/viral video shown inside a phone bezel",
     "gen": "frame_hud", "kwargs": {"style": "phone"}},
    {"family": "frame_hud", "id": "conference_frame", "label": "Video-Conference UI",
     "used_for": "Remote guests / interviews (name tag + mute icon)",
     "gen": "frame_hud", "kwargs": {"style": "conference", "label": "Dr. Rao — Expert"}},
    {"family": "info_panel", "id": "fact_box", "label": "Info / Fact Box",
     "used_for": "Side panel with key facts, fact-checks, stats",
     "gen": "info_panel", "kwargs": {"title": "Key facts",
                                     "lines": ["3 districts affected", "Rescue teams deployed", "Helpline 108"]}},
]

GENERATORS = {
    "banner": banner, "bug": bug, "stamp": stamp, "locator": locator,
    "source_attribution": source_attribution, "progress": progress,
    "countdown": countdown, "frame_hud": frame_hud, "info_panel": info_panel,
}


def render_registry_item(item_id: str, out_path: str,
                         font_path: str = None,
                         overrides: dict = None) -> Optional[str]:
    """Render one catalog row (used by the admin features preview and by the
    per-story compositor). ``overrides`` lets a caller change this instance's
    text (e.g. {"text": "MY HEADLINE", "sub": "..."}) — only KNOWN kwargs are
    overridden and only with a non-empty value, so a stale/foreign field can
    never break the generator; anything unset falls back to the REGISTRY default."""
    row = next((r for r in REGISTRY if r["id"] == item_id), None)
    if not row:
        return None
    kwargs = dict(row["kwargs"])
    if overrides:
        for _k, _v in overrides.items():
            if _k in kwargs and _v not in (None, ""):
                kwargs[_k] = _v
    kwargs["out_path"] = out_path
    kwargs.setdefault("font_path", font_path)
    return GENERATORS[row["gen"]](**kwargs)


def overlay_text_fields(item_id: str) -> dict:
    """The editable TEXT kwargs of an overlay + their current defaults, so the
    editor can render one input per field and seed it. Only str-valued kwargs
    are exposed (numeric/bool/tuple kwargs like colours, flags and angles are
    not user-text and stay at their defaults). {} for an unknown id."""
    row = next((r for r in REGISTRY if r["id"] == item_id), None)
    if not row:
        return {}
    return {k: v for k, v in row["kwargs"].items() if isinstance(v, str)}


# ═══ UG.4 EXPANSION — 6 more generators + preset tables → 190+ rows ═══

GREEN = (30, 160, 70, 255)
BLUE = (25, 95, 210, 255)
CYAN = (0, 170, 200, 255)
PURPLE = (120, 45, 200, 255)


@_fail_soft
def score_bug(*, team_a: str, score_a: str, team_b: str, score_b: str,
              out_path: str, accent: tuple = GREEN,
              font_path: str = None) -> Optional[str]:
    """Sports score strip (cricket/football broadcast corner bug)."""
    from PIL import Image, ImageDraw
    f, fsc = _font(font_path, 34), _font(font_path, 38)
    W, H = 620, 78
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=10, fill=DARK)
    d.rectangle([0, 0, 10, H], fill=accent)
    d.text((26, 20), team_a[:10], font=f, fill=WHITE)
    d.text((196, 16), score_a[:9], font=fsc, fill=accent)
    d.text((330, 20), team_b[:10], font=f, fill=WHITE)
    d.text((500, 16), score_b[:9], font=fsc, fill=accent)
    img.save(out_path, "PNG")
    return out_path


@_fail_soft
def weather_chip(*, city: str, temp: str, cond: str = "", out_path: str,
                 font_path: str = None) -> Optional[str]:
    """Weather chip: city + temperature + condition, with a sun/cloud glyph."""
    from PIL import Image, ImageDraw
    f, ft = _font(font_path, 32), _font(font_path, 44)
    W, H = 460, 86
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=12, fill=(10, 24, 44, 225))
    d.ellipse([20, 18, 68, 66], fill=(255, 190, 40, 255))          # sun
    d.ellipse([44, 36, 96, 70], fill=(235, 238, 244, 255))         # cloud
    d.text((116, 10), city[:16], font=f, fill=WHITE)
    d.text((116, 42), cond[:20], font=_font(font_path, 24),
           fill=(180, 195, 215, 235))
    d.text((W - 130, 18), temp[:6], font=ft, fill=(255, 214, 0, 255))
    img.save(out_path, "PNG")
    return out_path


@_fail_soft
def market_chip(*, name: str, value: str, change: str, out_path: str,
                font_path: str = None) -> Optional[str]:
    """Market index chip — green/red by the change sign."""
    from PIL import Image, ImageDraw
    up = not change.strip().startswith("-")
    col = GREEN if up else (200, 40, 40, 255)
    f, fv = _font(font_path, 30), _font(font_path, 34)
    W, H = 500, 74
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=10, fill=DARK)
    d.text((22, 20), name[:12], font=f, fill=WHITE)
    d.text((210, 18), value[:12], font=fv, fill=WHITE)
    arr = [(408, 46), (428, 22), (448, 46)] if up else [(408, 26), (428, 50), (448, 26)]
    d.polygon(arr, fill=col)
    d.text((300 if len(value) <= 9 else 320, 20), "", font=f, fill=col)
    d.text((W - 130, 44), change[:8], font=_font(font_path, 24), fill=col)
    img.save(out_path, "PNG")
    return out_path


@_fail_soft
def poll_bar(*, question: str, yes_pct: int, out_path: str,
             yes_label: str = "YES", no_label: str = "NO",
             font_path: str = None) -> Optional[str]:
    """Two-way poll result bar (elections, debates, audience votes)."""
    from PIL import Image, ImageDraw
    yes = max(0, min(100, int(yes_pct)))
    f, fq = _font(font_path, 28), _font(font_path, 30)
    W, H = 760, 130
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=12, fill=DARK)
    d.text((24, 12), question[:44], font=fq, fill=WHITE)
    bx, by, bw, bh = 24, 62, W - 48, 42
    split = bx + int(bw * yes / 100)
    d.rounded_rectangle([bx, by, split, by + bh], radius=8, fill=GREEN)
    d.rounded_rectangle([split, by, bx + bw, by + bh], radius=8,
                        fill=(170, 40, 40, 255))
    d.text((bx + 12, by + 6), f"{yes_label} {yes}%", font=f, fill=WHITE)
    tw = d.textbbox((0, 0), f"{no_label} {100 - yes}%", font=f)[2]
    d.text((bx + bw - tw - 12, by + 6), f"{no_label} {100 - yes}%",
           font=f, fill=WHITE)
    img.save(out_path, "PNG")
    return out_path


@_fail_soft
def social_card(*, handle: str, text: str, out_path: str,
                platform: str = "X", font_path: str = None) -> Optional[str]:
    """Social-post quote card (viral tweet/post shown on screen)."""
    from PIL import Image, ImageDraw
    f, fh = _font(font_path, 30), _font(font_path, 28)
    W, H = 900, 210
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=16,
                        fill=(250, 250, 252, 242))
    d.ellipse([22, 20, 74, 72], fill=(90, 120, 200, 255))
    d.text((34, 26), (handle or "?")[1:2].upper() or "?",
           font=_font(font_path, 30), fill=(255, 255, 255, 255))
    d.text((90, 22), handle[:24], font=fh, fill=(20, 20, 26, 255))
    d.text((90, 54), platform[:12], font=_font(font_path, 20),
           fill=(110, 115, 125, 255))
    t = " ".join((text or "").split())[:150]
    mid = len(t) // 2
    cut = t.rfind(" ", 0, mid + 16)
    lines = [t] if len(t) < 56 else [t[:cut], t[cut + 1:]]
    for i, ln in enumerate(lines[:3]):
        d.text((26, 96 + i * 40), ln[:60], font=f, fill=(25, 25, 32, 255))
    img.save(out_path, "PNG")
    return out_path


@_fail_soft
def headline_tag(*, text: str, out_path: str, color: tuple = RED,
                 font_path: str = None) -> Optional[str]:
    """Kicker tag — small colored chip above headlines (EXCLUSIVE /
    GROUND REPORT / FACT CHECK…)."""
    from PIL import Image, ImageDraw
    f = _font(font_path, 30)
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    tw = probe.textbbox((0, 0), text.upper(), font=f)[2]
    W, H = tw + 44, 54
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.polygon([(0, 0), (W - 16, 0), (W - 1, H // 2), (W - 16, H - 1), (0, H - 1)],
              fill=color)
    d.text((16, 8), text.upper()[:30], font=f, fill=WHITE)
    img.save(out_path, "PNG")
    return out_path


GENERATORS.update({
    "score_bug": score_bug, "weather_chip": weather_chip,
    "market_chip": market_chip, "poll_bar": poll_bar,
    "social_card": social_card, "headline_tag": headline_tag,
})


def _expansion_rows() -> list[dict]:
    """Preset tables → catalog rows. Every row is a real, distinct,
    genuinely-useful broadcast graphic (not filler): the spec's ~190
    assets from 15 generators."""
    rows: list[dict] = []

    def add(family, rid, lbl, used, gen, **kwargs):
        # NB: *kwargs* are the GENERATOR'S kwargs (may legitimately
        # contain 'label' for progress/frame_hud) — hence lbl/used names.
        rows.append({"family": family, "id": rid, "label": lbl,
                     "used_for": used, "gen": gen, "kwargs": kwargs})

    # banners — styles × editorial texts (20)
    for rid, text, color, sub in [
        ("banner_flash_update", "FLASH UPDATE", RED, "Just in"),
        ("banner_big_breaking", "BIG BREAKING", RED, "Watch till the end"),
        ("banner_exclusive", "EXCLUSIVE", (10, 10, 14, 245), "Only on this channel"),
        ("banner_ground_report", "GROUND REPORT", AMBER, "From the spot"),
        ("banner_fact_check", "FACT CHECK", BLUE, "Claim vs truth"),
        ("banner_election", "ELECTION 2026", PURPLE, "Full coverage"),
        ("banner_sports_final", "MATCH DAY", GREEN, "Live scores inside"),
        ("banner_market_live", "MARKETS LIVE", BLUE, "Sensex · Nifty"),
        ("banner_weather_warn", "RED ALERT", RED, "Heavy rain warning"),
        ("banner_top_story", "TOP STORY", (10, 10, 14, 245), "Tonight's lead"),
        ("banner_first_visuals", "FIRST VISUALS", RED, "Exclusive footage"),
        ("banner_public_interest", "PUBLIC NOTICE", AMBER, "Important information"),
        ("banner_crime_watch", "CRIME WATCH", (60, 8, 8, 245), "Investigation report"),
        ("banner_tech_launch", "JUST LAUNCHED", CYAN, "Tech desk"),
        ("banner_health_advisory", "HEALTH ADVISORY", (0, 140, 120, 255), "Do's and don'ts"),
        ("banner_good_news", "GOOD NEWS", GREEN, "A story to smile about"),
        ("banner_cta_comment", "COMMENT YOUR VIEW", BLUE, "Join the debate"),
        ("banner_cta_share", "SHARE THIS STORY", PURPLE, "Spread the word"),
        ("banner_cta_bell", "TAP THE BELL", RED, "Never miss an update"),
        ("banner_coming_up", "COMING UP", (10, 10, 14, 245), "After the break"),
    ]:
        add("banner", rid, text.title() + " Banner", f"{sub}", "banner",
            text=text, color=color, sub=sub)

    # bugs — statuses + cities + misc (22)
    for rid, text, dot in [
        ("bug_live_updates", "LIVE UPDATES", True), ("bug_file", "FILE", False),
        ("bug_archive", "ARCHIVE", False), ("bug_viral", "VIRAL", True),
        ("bug_exclusive", "EXCLUSIVE", True), ("bug_4k", "4K", False),
        ("bug_alert", "ALERT", True), ("bug_sting_op", "STING OP", True),
        ("bug_court", "COURT", False), ("bug_parliament", "PARLIAMENT", False),
        ("bug_stock", "MARKETS", False), ("bug_weather", "WEATHER", False),
        ("bug_traffic", "TRAFFIC", True), ("bug_sports_desk", "SPORTS DESK", False),
        ("bug_flashback", "FLASHBACK", False), ("bug_developing", "DEVELOPING", True),
    ]:
        add("bug", rid, f"{text} bug", "Corner status chip", "bug",
            text=text, dot=dot)
    for city in ("DELHI", "MUMBAI", "CHENNAI", "BENGALURU", "VIZAG", "WARANGAL"):
        add("bug", f"bug_city_{city.lower()}", f"{city} bug",
            "Where the footage is from", "bug", text=city)

    # stamps (17)
    for rid, text, color, angle in [
        ("stamp_breaking", "BREAKING", RED, -12.0),
        ("stamp_verified", "VERIFIED", GREEN, -8.0),
        ("stamp_fake", "FAKE", RED, 10.0),
        ("stamp_busted", "BUSTED", RED, -14.0),
        ("stamp_myth", "MYTH", AMBER, 8.0),
        ("stamp_fact", "FACT", GREEN, -8.0),
        ("stamp_leaked", "LEAKED", (60, 8, 8, 255), -10.0),
        ("stamp_urgent", "URGENT", RED, -6.0),
        ("stamp_classified", "CLASSIFIED", (10, 10, 14, 255), -12.0),
        ("stamp_approved", "APPROVED", GREEN, -8.0),
        ("stamp_rejected", "REJECTED", RED, 8.0),
        ("stamp_censored", "CENSORED", (10, 10, 14, 255), 0.0),
        ("stamp_wanted", "WANTED", (120, 30, 20, 255), -10.0),
        ("stamp_solved", "CASE SOLVED", GREEN, -8.0),
        ("stamp_evidence", "EVIDENCE", AMBER, -12.0),
        ("stamp_first", "FIRST ON THIS CHANNEL", BLUE, -6.0),
        ("stamp_alert100", "100% REAL", GREEN, -10.0),
    ]:
        add("stamp", rid, f"{text} stamp", "Rotated rubber-stamp seal",
            "stamp", text=text, color=color, angle=angle)

    # locators (12)
    for city in ("Delhi", "Mumbai", "Chennai", "Bengaluru", "Vizag", "Amaravati"):
        add("locator", f"loc_{city.lower()}_live", f"LIVE · {city}",
            "Live dateline chip", "locator", text=city, live=True)
        add("locator", f"loc_{city.lower()}", f"{city} dateline",
            "Plain dateline chip", "locator", text=city, live=False)

    # attributions (12)
    for src in ("Reuters", "ANI", "PTI", "AP", "File footage", "Viral video",
                "CCTV footage", "Social media", "Screen grab", "Handout",
                "Agency", "Archives"):
        add("attribution", f"attr_{src.split()[0].lower()}",
            f"Source: {src}", "Footage credit line", "source_attribution",
            text=f"Source: {src}")

    # progress chips (8)
    for cur, tot, lab in [(1, 5, "Story"), (3, 5, "Story"), (5, 5, "Story"),
                          (1, 3, "Part"), (2, 3, "Part"),
                          (1, 10, "Top"), (5, 10, "Top"), (10, 10, "Top")]:
        add("progress", f"prog_{lab.lower()}_{cur}_{tot}",
            f"{lab} {cur} of {tot}", "Sequence position chip",
            "progress", current=cur, total=tot, label=lab)

    # countdowns (8)
    for val, sub in [("3", "stories tonight"), ("5", "big updates"),
                     ("10", "seconds to verdict"), ("2", "days to elections"),
                     ("1", "question remains"), ("7", "facts you missed"),
                     ("NEXT", "the main story"), ("TONIGHT", "9 PM")]:
        add("countdown", f"cd_{val.lower()}", f"Countdown — {val}",
            "Interstitial number card", "countdown", value=val, sub=sub)

    # HUD frames (9)
    for rid, style, label in [
        ("hud_cctv_cam01", "cctv", "CAM-01"), ("hud_cctv_cam07", "cctv", "CAM-07"),
        ("hud_cctv_gate", "cctv", "MAIN GATE"),
        ("hud_viewfinder_tape", "viewfinder", "00:12:44:19"),
        ("hud_conference_reporter", "conference", "Field Reporter"),
        ("hud_conference_lawyer", "conference", "Adv. Meena — Legal Expert"),
        ("hud_drone", "drone", "DRONE 01"),
        ("hud_bodycam", "bodycam", "BODYCAM 07"),
        ("hud_dashcam", "dashcam", "DASHCAM"),
    ]:
        add("frame_hud", rid, label, "Full-frame footage treatment",
            "frame_hud", style=style, label=label)

    # info panels (12)
    for rid, title, lines in [
        ("panel_key_facts", "Key facts", ["3 districts affected", "Rescue teams deployed", "Helpline 108"]),
        ("panel_timeline", "Timeline", ["6 AM — first report", "9 AM — teams reach", "2 PM — operation over"]),
        ("panel_what_we_know", "What we know", ["Cause under probe", "No casualties", "Traffic diverted"]),
        ("panel_what_next", "What happens next", ["Hearing on Friday", "Report in 2 weeks"]),
        ("panel_numbers", "By the numbers", ["₹300 Cr project", "12 villages", "3 years delay"]),
        ("panel_dos", "Do's", ["Stay indoors", "Keep phones charged", "Follow advisories"]),
        ("panel_donts", "Don'ts", ["Avoid rumours", "Don't share unverified clips"]),
        ("panel_scheme", "Scheme highlights", ["Free for BPL families", "Apply before Aug 15"]),
        ("panel_verdict", "Court said", ["Bail granted", "Next hearing Aug 2"]),
        ("panel_scorecard", "Scorecard", ["IND 287/4 (48 ov)", "AUS 260 all out"]),
        ("panel_symptoms", "Watch for", ["High fever", "Fatigue", "Consult a doctor"]),
        ("panel_checklist", "Checklist", ["Aadhaar", "Photo", "Address proof"]),
    ]:
        add("info_panel", rid, title, "Side fact box", "info_panel",
            title=title, lines=lines)

    # score bugs (8)
    for rid, a, sa, b, sb, acc in [
        ("score_cricket_t20", "IND", "187/5", "AUS", "142/7", GREEN),
        ("score_cricket_test", "IND", "402/6d", "ENG", "218/9", GREEN),
        ("score_cricket_chase", "CSK", "165/3", "MI", "164/8", AMBER),
        ("score_football", "IND", "2", "QAT", "1", BLUE),
        ("score_kabaddi", "TEL", "38", "TN", "34", AMBER),
        ("score_hockey", "IND", "3", "PAK", "2", GREEN),
        ("score_badminton", "SINDHU", "21-18", "YAMAG", "19-21", PURPLE),
        ("score_election", "NDA", "162", "INDIA", "148", RED),
    ]:
        add("score_bug", rid, f"{a} vs {b}", "Score strip", "score_bug",
            team_a=a, score_a=sa, team_b=b, score_b=sb, accent=acc)

    # weather chips (8)
    for city, temp, cond in [("Hyderabad", "34°C", "Partly cloudy"),
                             ("Vijayawada", "38°C", "Heat wave"),
                             ("Vizag", "29°C", "Heavy rain"),
                             ("Delhi", "41°C", "Severe heat"),
                             ("Mumbai", "31°C", "Monsoon showers"),
                             ("Chennai", "36°C", "Humid"),
                             ("Bengaluru", "26°C", "Pleasant"),
                             ("Warangal", "35°C", "Thunderstorms")]:
        add("weather_chip", f"wx_{city.lower()}", f"{city} weather",
            "City weather chip", "weather_chip", city=city, temp=temp,
            cond=cond)

    # market chips (8)
    for rid, name, val, chg in [
        ("mkt_sensex_up", "SENSEX", "81,250", "+420"),
        ("mkt_sensex_down", "SENSEX", "79,980", "-610"),
        ("mkt_nifty_up", "NIFTY 50", "24,720", "+130"),
        ("mkt_nifty_down", "NIFTY 50", "24,310", "-180"),
        ("mkt_gold", "GOLD 10g", "₹74,300", "+250"),
        ("mkt_usdinr", "USD/INR", "83.42", "-0.12"),
        ("mkt_crude", "CRUDE", "$82.4", "+1.3"),
        ("mkt_btc", "BITCOIN", "$96,400", "+2,100"),
    ]:
        add("market_chip", rid, f"{name} {chg}", "Market ticker chip",
            "market_chip", name=name, value=val, change=chg)

    # poll bars (6)
    for rid, q, y in [
        ("poll_govt", "Do you support the new policy?", 58),
        ("poll_close", "Was the verdict fair?", 51),
        ("poll_no", "Should the minister resign?", 44),
        ("poll_yes", "Will India win the series?", 78),
        ("poll_split", "Is fuel pricing justified?", 50),
        ("poll_low", "Do you trust exit polls?", 32),
    ]:
        add("poll_bar", rid, q, "Two-way audience poll", "poll_bar",
            question=q, yes_pct=y)

    # social cards (6)
    for rid, h, t in [
        ("soc_min", "@MinisterOffice", "The project will be completed before the deadline. No compromise on quality."),
        ("soc_actor", "@Actor_Official", "Thank you for all the love. The trailer drops tomorrow at 9 AM!"),
        ("soc_police", "@CityPolice", "Traffic diverted at NH-44 due to waterlogging. Please use alternate routes."),
        ("soc_viral", "@user_2026", "Never seen rain like this in 20 years. Stay safe everyone."),
        ("soc_sports", "@BCCI", "Squad announced for the T20 series. Two debutants named."),
        ("soc_startup", "@FounderDesk", "We are hiring 500 engineers across Hyderabad and Bengaluru."),
    ]:
        add("social_card", rid, f"Post by {h}", "Viral post on screen",
            "social_card", handle=h, text=t)

    # headline kicker tags (14)
    for rid, text, color in [
        ("tag_exclusive", "EXCLUSIVE", RED), ("tag_ground", "GROUND REPORT", AMBER),
        ("tag_factcheck", "FACT CHECK", BLUE), ("tag_bigstory", "BIG STORY", RED),
        ("tag_explainer", "EXPLAINER", CYAN), ("tag_interview", "INTERVIEW", PURPLE),
        ("tag_opinion", "OPINION", (90, 90, 100, 255)),
        ("tag_investigation", "INVESTIGATION", (60, 8, 8, 255)),
        ("tag_positive", "GOOD NEWS", GREEN), ("tag_alert", "ALERT", RED),
        ("tag_sports", "SPORTS", GREEN), ("tag_business", "BUSINESS", BLUE),
        ("tag_tech", "TECH", CYAN), ("tag_entertainment", "ENTERTAINMENT", PURPLE),
    ]:
        add("headline_tag", rid, f"{text} kicker", "Chip above headlines",
            "headline_tag", text=text, color=color)

    return rows


REGISTRY.extend(_expansion_rows())


# ── Full-canvas positioned render (compose-pipeline consumption) ─────
# The Director picks overlay ids with timestamps; the bulletin compose
# overlays FULL-CANVAS transparent PNGs at 0:0 with enable-windows
# (same mechanism as name-straps). Each family has a broadcast position.

OVERLAY_CANVAS_POS = {
    # family: (x_rule, y_frac)  — x_rule: float frac | "full_width" | "cover" | None=center
    "banner": ("full_width", 0.70),
    "bug": (0.80, 0.06),
    "stamp": (None, 0.24),
    "locator": (0.02, 0.055),
    "attribution": (0.02, 0.76),
    "progress": (0.35, 0.045),
    "countdown": ("cover", None),
    "frame_hud": ("cover", None),
    "info_panel": (0.58, 0.16),
    "score_bug": (0.02, 0.05),
    "weather_chip": (0.02, 0.05),
    "market_chip": (0.02, 0.05),
    "poll_bar": (0.16, 0.56),
    "social_card": (0.07, 0.22),
    "headline_tag": (0.03, 0.62),
}


@_fail_soft
def render_overlay_positioned(item_id: str, out_path: str, *,
                              canvas_w: int = 1920, canvas_h: int = 1080,
                              font_path: str = None,
                              overrides: dict = None) -> Optional[str]:
    """One catalog row rendered onto a TRANSPARENT full-canvas PNG at its
    broadcast position — ready for a 0:0 enable-window overlay. ``overrides``
    changes this instance's text (see ``render_registry_item``)."""
    from PIL import Image
    row = next((r for r in REGISTRY if r["id"] == item_id), None)
    if not row:
        return None
    el_path = out_path + ".el.png"
    if not render_registry_item(item_id, el_path, font_path=font_path,
                                overrides=overrides):
        return None
    el = Image.open(el_path).convert("RGBA")
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    rule = OVERLAY_CANVAS_POS.get(row["family"], (None, None))
    if rule[0] == "cover":
        el = el.resize((canvas_w, canvas_h))
        canvas.alpha_composite(el)
    elif rule[0] == "full_width":
        r = canvas_w / el.width
        el = el.resize((canvas_w, max(1, int(el.height * r))))
        canvas.alpha_composite(el, (0, int(canvas_h * rule[1])))
    else:
        # native element scale relative to a 1920-wide reference canvas
        r = canvas_w / 1920.0
        el = el.resize((max(1, int(el.width * r)), max(1, int(el.height * r))))
        x = (int(canvas_w * rule[0]) if rule[0] is not None
             else (canvas_w - el.width) // 2)
        y = (int(canvas_h * rule[1]) if rule[1] is not None
             else (canvas_h - el.height) // 2)
        canvas.alpha_composite(el, (max(0, min(x, canvas_w - el.width)),
                                    max(0, min(y, canvas_h - el.height))))
    canvas.save(out_path, "PNG")
    try:
        os.remove(el_path)
    except OSError:
        pass
    return out_path
