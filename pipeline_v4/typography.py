"""Typography engine — animated text built on the word timestamps.

Three launch effects (the spec's §8 backbone), all PIL-rendered
(Indic-safe — ffmpeg drawtext is never used for text) and returned as
``[(png_path, t_start, t_end), …]`` overlay windows that the existing
enable-gated compose consumes exactly like name-straps:

  * karaoke_captions — the CapCut/Descript signature: caption lines with
    the CURRENT word highlighted, driven by TrimmedStory.words. One PNG
    per word-state; windows tile the speech exactly.
  * typewriter — text appears character by character (investigations,
    documents).
  * headline_crash — the headline slams in with a physics spring from
    the native motion toolkit (breaking news, trailer cards).

Fail-soft: any error returns [] — text animation is a garnish, never a
render blocker.
"""
from __future__ import annotations

import os
from typing import Optional

from pipeline_v4.overlays import _font  # shared scalable-font resolver

WHITE = (255, 255, 255, 255)
HILITE = (255, 214, 0, 255)      # karaoke current-word yellow
DARKBOX = (8, 8, 12, 190)
RED = (193, 18, 18, 255)


def _fail_soft_list(fn):
    def wrapped(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as exc:
            print(f"[v4/typography] {fn.__name__} failed (soft): {exc}", flush=True)
            return []
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped


# ── Styled text renderer — the polish layer ──────────────────────────
# Flat PIL text reads as "basic" (operator). Broadcast-grade text =
# heavy dark stroke + soft drop shadow + optional vertical gradient
# fill. One renderer, used by karaoke, headlines and flashes.

def _styled_text_img(text: str, font, *, fill=WHITE, stroke: int = 3,
                     stroke_fill=(10, 10, 14, 255), shadow: bool = True,
                     shadow_off=(3, 5), shadow_blur: int = 4,
                     gradient=None, weight: int = 0):
    """Render text as a padded RGBA image: dark stroke, blurred drop
    shadow, optional (top_color, bottom_color) vertical gradient fill.
    ``weight`` fakes a heavier font by dilating the SHAPED glyph bitmap
    with a same-colour inner stroke — Indic-safe (only Bold Notos are
    bundled, and raster ops after shaping never break conjuncts/matras).
    weight=0 is arithmetically identical to the pre-weight renderer, so
    every existing caller stays pixel-identical.
    Returns the PIL Image (caller composites)."""
    from PIL import Image, ImageDraw, ImageFilter
    total = stroke + weight
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    box = probe.textbbox((0, 0), text, font=font, stroke_width=total)
    tw, th = box[2] - box[0], box[3] - box[1]
    pad = total + shadow_blur + max(abs(shadow_off[0]), abs(shadow_off[1])) + 2
    W, H = tw + 2 * pad, th + 2 * pad
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ox, oy = pad - box[0], pad - box[1]
    if shadow:
        sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(sh).text((ox + shadow_off[0], oy + shadow_off[1]),
                                text, font=font, fill=(0, 0, 0, 190),
                                stroke_width=total,
                                stroke_fill=(0, 0, 0, 190))
        img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(shadow_blur)))
    d = ImageDraw.Draw(img)
    d.text((ox, oy), text, font=font, fill=fill,
           stroke_width=total, stroke_fill=stroke_fill)
    if weight:
        # inner self-coloured stroke: dilates the SHAPED glyph bitmap
        # (Indic-safe heavy weight) — the outer `stroke` px stay dark.
        d.text((ox, oy), text, font=font, fill=fill,
               stroke_width=weight, stroke_fill=fill)
    if gradient:
        top, bottom = gradient
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).text((ox, oy), text, font=font, fill=255)
        grad = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        gd = ImageDraw.Draw(grad)
        for y in range(H):
            f = y / max(1, H - 1)
            gd.line([(0, y), (W, y)], fill=tuple(
                int(top[c] + (bottom[c] - top[c]) * f) for c in range(3)
            ) + (255,))
        img.paste(grad, (0, 0), mask)
    return img


def _glow_text_img(text: str, font, color):
    """Gradient-filled word with a soft same-colour halo (Indic-safe:
    the halo is a blur of the shaped glyph's alpha channel)."""
    from PIL import Image, ImageFilter
    grad = (tuple(min(255, c + 70) for c in color[:3]), tuple(color[:3]))
    core = _styled_text_img(text, font, fill=color, stroke=2,
                            stroke_fill=(10, 10, 14, 200), shadow=False,
                            gradient=grad)
    pad = 10
    W, H = core.width + 2 * pad, core.height + 2 * pad
    a = Image.new("L", (W, H), 0)
    a.paste(core.getchannel("A"), (pad, pad))
    tint = Image.new("RGBA", (W, H), tuple(color[:3]) + (0,))
    tint.putalpha(a.point(lambda v: int(v * 0.85)))
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    img.alpha_composite(tint.filter(ImageFilter.GaussianBlur(7)))
    img.alpha_composite(core, (pad, pad))
    return img


# ── Karaoke captions ────────────────────────────────────────────────

def group_caption_lines(words: list[dict], *, max_words: int = 4,
                        max_chars: int = 26, max_gap: float = 1.0) -> list[list[dict]]:
    """Split a story-relative word array into caption lines: at most
    ``max_words``/``max_chars`` per line, and never bridging a silence
    longer than ``max_gap`` (a caption hanging over dead air looks
    broken). Pure function."""
    lines: list[list[dict]] = []
    cur: list[dict] = []
    cur_chars = 0
    for w in (words or []):
        tok = str(w.get("w", "")).strip()
        if not tok:
            continue
        try:
            ws, we = float(w["s"]), float(w["e"])
        except (KeyError, TypeError, ValueError):
            continue
        gap_break = cur and (ws - float(cur[-1]["e"])) > max_gap
        full = cur and (len(cur) >= max_words
                        or cur_chars + 1 + len(tok) > max_chars)
        if gap_break or full:
            lines.append(cur)
            cur, cur_chars = [], 0
        cur.append({"w": tok, "s": ws, "e": we})
        cur_chars += (1 if cur_chars else 0) + len(tok)
    if cur:
        lines.append(cur)
    return lines


@_fail_soft_list
def karaoke_captions(
    words: list[dict], *, out_dir, canvas_w: int = 1080,
    font_path: Optional[str] = None, font_size: int = 58,
    prefix: str = "kc", style: str = "hilite", hilite: tuple = HILITE,
    hilite2: Optional[tuple] = None,
) -> list[tuple[str, float, float]]:
    """One PNG per word-state: the full caption line with the CURRENT
    word emphasized; windows are the word's exact [s, e] (last word of
    a line holds until the line ends). Sized for a 9:16 short by
    default. ``style``: hilite (colored word), box (colored pill behind
    it), underline (bar below it), pop (bold outlined + lifted),
    kinetic (word pops up boxed in accent), chips (every word in its
    own plate, spoken one accent-filled), glow (gradient word + soft
    halo), flip (accent alternates hilite/hilite2 per word), impact
    (heavy bar-less broadcast promo)."""
    from PIL import Image, ImageDraw
    os.makedirs(out_dir, exist_ok=True)
    f = _font(font_path, font_size)
    style = (style or "hilite").lower()
    out: list[tuple[str, float, float]] = []
    k = 0
    f_big = _font(font_path, int(font_size * 1.14))
    f_pop = _font(font_path, int(font_size * 1.30))
    f_pop2 = _font(font_path, int(font_size * 1.45))   # kinetic blink
    f_imp = _font(font_path, int(font_size * 1.08))    # impact base
    # CYAN default matches karaoke_flip_duo's registry row, so the
    # bridge's style+hilite-only call site renders flip correctly
    # without passing hilite2 (v1_bridge stays untouched).
    hilite2 = tuple(hilite2) if hilite2 else CYAN

    def _grad(c):
        return (tuple(min(255, x + 60) for x in c[:3]), tuple(c[:3]))

    def _base_img(tok):
        if style == "impact":       # heavy weight + thin dark outline
            return _styled_text_img(tok, f_imp, stroke=2, weight=2)
        return _styled_text_img(tok, f)

    def _active_img(tok: str, fnt, color=None):
        c = tuple(color or hilite)
        if style in ("box", "kinetic", "chips"):   # dark text on a bright chip
            return _styled_text_img(tok, fnt, fill=(12, 12, 14, 255),
                                    stroke=0, shadow=False)
        if style == "impact":
            return _styled_text_img(tok, fnt, fill=c, stroke=2, weight=2)
        if style == "glow":
            return _glow_text_img(tok, fnt, c)
        return _styled_text_img(tok, fnt, fill=c, gradient=_grad(c))

    def _frame(toks, base_imgs, i, fnt_active, canvas_w_):
        """One caption state: styled word images composed on a pill bar,
        the ACTIVE word physically larger (real pop, not a recolor)."""
        imgs = list(base_imgs)
        c_active = hilite2 if (style == "flip" and i % 2) else hilite
        imgs[i] = _active_img(toks[i], fnt_active, c_active)
        space_w = (max(14, font_size // 2) if style == "chips"
                   else max(10, font_size // 3))   # chips need gutters
        total_w = sum(im.width for im in imgs) + space_w * (len(imgs) - 1)
        W = min(canvas_w_, total_w + 44)
        H = max(im.height for im in imgs) + 24
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        if style not in ("chips", "impact"):   # chips = per-word plates,
            d.rounded_rectangle([0, 0, W - 1, H - 1],   # impact floats bar-less
                                radius=16, fill=DARKBOX)
            d.rounded_rectangle([0, 0, W - 1, H - 1], radius=16,
                                outline=(255, 255, 255, 26), width=1)
        x = max(6, (W - total_w) // 2)
        for j, im in enumerate(imgs):
            y = (H - im.height) // 2
            if style == "chips":
                d.rounded_rectangle(
                    [x - 8, y + 6, x + im.width + 8, y + im.height - 2],
                    radius=12,
                    fill=(tuple(hilite) if j == i else (16, 16, 22, 215)))
            if j == i and style in ("box", "kinetic"):
                d.rounded_rectangle(
                    [x - 6, y + 8, x + im.width + 6, y + im.height - 4],
                    radius=10, fill=hilite)
            if j == i and style == "underline":
                d.rounded_rectangle(
                    [x + 2, y + im.height - 8, x + im.width - 2,
                     y + im.height - 2], radius=3, fill=hilite)
            img.alpha_composite(im, (x, y))
            x += im.width + space_w
        return img

    for line in group_caption_lines(words):
        toks = [w["w"] for w in line]
        base_imgs = [_base_img(t) for t in toks]
        line_end = line[-1]["e"]
        for i, cur in enumerate(line):
            a = round(cur["s"], 3)
            t_end = line[i + 1]["s"] if i + 1 < len(line) else line_end
            b = round(max(t_end, cur["s"] + 0.05), 3)
            # temporal POP: an oversized first blink, then the settled state
            if style in ("pop", "kinetic") and (b - a) >= 0.2:
                pop_img = _frame(toks, base_imgs, i,
                                 f_pop2 if style == "kinetic" else f_pop,
                                 canvas_w)
                pp = os.path.join(out_dir, f"_{prefix}_{k:04d}.png")
                pop_img.save(pp, "PNG")
                out.append((pp, a, round(a + 0.09, 3)))
                k += 1
                a = round(a + 0.09, 3)
            # kinetic settles scaled-up; impact's spoken word runs bigger too
            img = _frame(toks, base_imgs, i,
                         f_pop if style in ("kinetic", "impact") else f_big,
                         canvas_w)
            p = os.path.join(out_dir, f"_{prefix}_{k:04d}.png")
            img.save(p, "PNG")
            out.append((p, a, b))
            k += 1
    return out


# ── Typewriter ──────────────────────────────────────────────────────

@_fail_soft_list
def typewriter(
    text: str, *, out_dir, t_start: float = 0.0, char_per_sec: float = 16.0,
    hold: float = 1.5, font_path: Optional[str] = None, font_size: int = 52,
    prefix: str = "tw", color: tuple = (210, 255, 210, 255),
    bg: tuple = (4, 4, 6, 200), cursor: bool = True,
) -> list[tuple[str, float, float]]:
    """Character-by-character reveal (optionally with a block cursor);
    the finished line holds for ``hold`` seconds."""
    from PIL import Image, ImageDraw
    os.makedirs(out_dir, exist_ok=True)
    t = " ".join((text or "").split())[:90]
    if not t:
        return []
    f = _font(font_path, font_size)
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    full_w = probe.textbbox((0, 0), t + "█", font=f)[2]
    W, H = full_w + 40, int(font_size * 1.8)
    out: list[tuple[str, float, float]] = []
    step = 1.0 / max(1.0, char_per_sec)
    for i in range(1, len(t) + 1):
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W - 1, H - 1], fill=bg)
        shown = t[:i] + ("█" if (cursor and i < len(t)) else "")
        d.text((20, (H - font_size) // 2 - font_size // 8), shown,
               font=f, fill=color)
        p = os.path.join(out_dir, f"_{prefix}_{i:04d}.png")
        img.save(p, "PNG")
        a = t_start + (i - 1) * step
        b = t_start + i * step if i < len(t) else t_start + i * step + hold
        out.append((p, round(a, 3), round(b, 3)))
    return out


# ── Headline crash (spring slam-in) ─────────────────────────────────

@_fail_soft_list
def headline_crash(
    text: str, *, out_dir, t_start: float = 0.0, hold: float = 2.5,
    fps: int = 30, frames: int = 14, canvas_w: int = 1920,
    font_path: Optional[str] = None, font_size: int = 96,
    color: tuple = RED, prefix: str = "hc", motion: str = "crash",
) -> list[tuple[str, float, float]]:
    """The headline enters with physics. ``motion``: crash (spring scale
    slam, the signature), bounce (spring up from below), slide_up /
    slide_down / whip_left / whip_right (eased offsets), fade_scale
    (grow-in), spin_in (rotate settle). Frames at ``fps``, then the
    settled frame holds."""
    from PIL import Image, ImageDraw
    from pipeline_v4.motion import spring
    os.makedirs(out_dir, exist_ok=True)
    t = " ".join((text or "").upper().split())[:60]
    if not t:
        return []
    f = _font(font_path, font_size)
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    box = probe.textbbox((0, 0), t, font=f)
    tw, th = box[2] - box[0], box[3] - box[1]
    pad = 36
    base_w, base_h = min(canvas_w, tw + 2 * pad), th + 2 * pad
    motion = (motion or "crash").lower()

    def _card(alpha: float) -> Image.Image:
        """Designed title card: soft outer shadow, vertical gradient,
        light top edge, accent underline, stroked+shadowed text — not a
        flat rectangle (operator: 'typography is so basic')."""
        from PIL import ImageFilter
        M = 14                                       # shadow margin
        img = Image.new("RGBA", (base_w, base_h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        r0 = [M, M, base_w - M - 1, base_h - M - 1]
        # outer soft shadow
        sh = Image.new("RGBA", (base_w, base_h), (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle(
            [M + 4, M + 7, base_w - M + 3, base_h - M + 6],
            radius=12, fill=(0, 0, 0, 150))
        img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(7)))
        # vertical gradient card (color → 45% darker)
        dark = tuple(int(c * 0.55) for c in color[:3])
        card_h = r0[3] - r0[1]
        for yy in range(card_h):
            fr = yy / max(1, card_h - 1)
            row = tuple(int(color[c] + (dark[c] - color[c]) * fr)
                        for c in range(3))
            d.line([(r0[0], r0[1] + yy), (r0[2], r0[1] + yy)],
                   fill=row + (235,))
        d.rounded_rectangle(r0, radius=10, outline=(255, 255, 255, 60),
                            width=2)
        d.line([(r0[0] + 8, r0[1] + 2), (r0[2] - 8, r0[1] + 2)],
               fill=(255, 255, 255, 110), width=2)
        # accent underline bar
        d.rounded_rectangle([r0[0] + 22, r0[3] - 14, r0[0] + 132, r0[3] - 8],
                            radius=3, fill=(255, 255, 255, 200))
        txt = _styled_text_img(t, f, fill=(255, 255, 255, 255), stroke=3,
                               stroke_fill=(0, 0, 0, 160), shadow=True)
        img.alpha_composite(txt, ((base_w - txt.width) // 2,
                                  (base_h - txt.height) // 2 - 4))
        if alpha < 0.999:
            a_ch = img.getchannel("A").point(lambda v: int(v * alpha))
            img.putalpha(a_ch)
        return img

    def _frame(i: int) -> Image.Image:
        # progress drivers: spring (overshoot) and out-cubic ease
        s = spring(i + 1, fps, stiffness=210.0, damping=11.0)
        e = 1.0 - pow(max(0.0, 1.0 - (i + 1) / max(1, frames - 2)), 3)
        alpha = min(1.0, (i + 1) / 4.0)
        card = _card(alpha)
        cw, chh = int(base_w * 1.6), int(base_h * 2.2)
        canvas = Image.new("RGBA", (cw, chh), (0, 0, 0, 0))
        cx, cy = (cw - base_w) // 2, (chh - base_h) // 2
        if motion == "crash":
            scale = 1.55 - 0.55 * s
            w2, h2 = max(1, int(base_w * scale)), max(1, int(base_h * scale))
            card = card.resize((w2, h2), Image.LANCZOS)
            canvas.paste(card, ((cw - w2) // 2, (chh - h2) // 2), card)
        elif motion == "bounce":
            canvas.paste(card, (cx, cy + int((1.0 - s) * base_h * 1.2)), card)
        elif motion == "slide_up":
            canvas.paste(card, (cx, cy + int((1.0 - e) * 110)), card)
        elif motion == "slide_down":
            canvas.paste(card, (cx, cy - int((1.0 - e) * 110)), card)
        elif motion == "whip_left":
            canvas.paste(card, (cx + int((1.0 - e) * 260), cy), card)
        elif motion == "whip_right":
            canvas.paste(card, (cx - int((1.0 - e) * 260), cy), card)
        elif motion == "fade_scale":
            scale = 0.80 + 0.20 * e
            w2, h2 = max(1, int(base_w * scale)), max(1, int(base_h * scale))
            card = card.resize((w2, h2), Image.LANCZOS)
            canvas.paste(card, ((cw - w2) // 2, (chh - h2) // 2), card)
        elif motion == "spin_in":
            card = card.rotate((1.0 - e) * 22, expand=True,
                               resample=Image.BICUBIC)
            canvas.paste(card, ((cw - card.width) // 2,
                                (chh - card.height) // 2), card)
        else:
            canvas.paste(card, (cx, cy), card)
        return canvas

    out: list[tuple[str, float, float]] = []
    step = 1.0 / fps
    for i in range(frames):
        img = _frame(i)
        p = os.path.join(out_dir, f"_{prefix}_{i:03d}.png")
        img.save(p, "PNG")
        a = t_start + i * step
        b = a + step if i < frames - 1 else a + step + hold
        out.append((p, round(a, 3), round(b, 3)))
    return out


# ── Static / few-frame text cards (spec §8 vocabulary) ──────────────

GOLD = (212, 175, 55, 255)
CYAN = (0, 200, 220, 255)
GREEN = (60, 200, 90, 255)
ORANGE = (240, 130, 30, 255)


def _text_card(draw_fn, w: int, h: int, out_path: str) -> Optional[str]:
    """Tiny helper: build a transparent card, hand (img, draw) to
    draw_fn, save. Fail-soft to None."""
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw_fn(img, ImageDraw.Draw(img))
        img.save(out_path, "PNG")
        return out_path
    except Exception as exc:
        print(f"[v4/typography] card failed (soft): {exc}", flush=True)
        return None


@_fail_soft_list
def counter(value: int, *, out_dir, label: str = "", prefix_symbol: str = "",
            suffix: str = "", frames: int = 16, fps: int = 30,
            t_start: float = 0.0, hold: float = 2.0,
            font_path: Optional[str] = None, color: tuple = WHITE,
            prefix: str = "cnt") -> list[tuple[str, float, float]]:
    """Count-up number (money/percent/plain): eased 0→value over
    ``frames``, then holds. The stats moment every explainer needs."""
    from PIL import Image, ImageDraw
    os.makedirs(out_dir, exist_ok=True)
    fbig, fsmall = _font(font_path, 120), _font(font_path, 40)
    W, H = 760, 260
    out = []
    step = 1.0 / fps
    for i in range(frames):
        e = 1.0 - pow(1.0 - (i + 1) / frames, 3)
        shown = f"{prefix_symbol}{int(round(value * e)):,}{suffix}"
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, W - 1, H - 1], radius=18, fill=DARKBOX)
        tw = d.textbbox((0, 0), shown, font=fbig)[2]
        d.text(((W - tw) // 2, 30), shown, font=fbig, fill=color)
        if label:
            lw = d.textbbox((0, 0), label, font=fsmall)[2]
            d.text(((W - lw) // 2, 178), label, font=fsmall,
                   fill=(210, 210, 215, 235))
        p = os.path.join(out_dir, f"_{prefix}_{i:03d}.png")
        img.save(p, "PNG")
        a = t_start + i * step
        b = a + step if i < frames - 1 else a + step + hold
        out.append((p, round(a, 3), round(b, 3)))
    return out


def quote_card(text: str, *, out_path, author: str = "",
               style: str = "news", font_path: Optional[str] = None) -> Optional[str]:
    """Big quotation card (news / dark / minimal)."""
    f, fa = _font(font_path, 54), _font(font_path, 34)
    W, H = 1100, 360
    bg = {"news": (12, 12, 16, 235), "dark": (5, 5, 8, 245),
          "minimal": (255, 255, 255, 235)}.get(style, (12, 12, 16, 235))
    fg = (20, 20, 24, 255) if style == "minimal" else WHITE
    accent = RED if style == "news" else (GOLD if style == "dark" else (120, 120, 130, 255))

    def draw(img, d):
        d.rounded_rectangle([0, 0, W - 1, H - 1], radius=16, fill=bg)
        d.rectangle([0, 0, 10, H], fill=accent)
        d.text((44, 24), "“", font=_font(font_path, 120), fill=accent)
        t = " ".join((text or "").split())[:120]
        # naive 2-line wrap
        mid = len(t) // 2
        cut = t.rfind(" ", 0, mid + 12)
        lines = [t] if len(t) < 46 else [t[:cut], t[cut + 1:]]
        for i, ln in enumerate(lines[:2]):
            d.text((120, 70 + i * 78), ln, font=f, fill=fg)
        if author:
            d.text((120, H - 70), f"— {author[:40]}", font=fa, fill=accent)
    return _text_card(draw, W, H, out_path)


def name_tag(name: str, *, out_path, role: str = "",
             style: str = "news", font_path: Optional[str] = None) -> Optional[str]:
    """Speaker name tag (news bar / clean pill / badge)."""
    f, fr = _font(font_path, 46), _font(font_path, 30)
    W, H = 640, 130

    def draw(img, d):
        if style == "badge":
            d.rounded_rectangle([0, 0, W - 1, H - 1], radius=64, fill=DARKBOX)
            d.ellipse([12, 15, 12 + 100, 115], fill=RED)
            d.text((56, 38), (name or "?")[:1].upper(), font=f, fill=WHITE)
            d.text((136, 22), name[:24], font=f, fill=WHITE)
            if role:
                d.text((136, 78), role[:34], font=fr, fill=(200, 200, 205, 230))
        elif style == "clean":
            d.rounded_rectangle([0, 0, W - 1, H - 1], radius=10,
                                fill=(250, 250, 252, 235))
            d.text((26, 18), name[:26], font=f, fill=(15, 15, 20, 255))
            if role:
                d.text((26, 76), role[:40], font=fr, fill=(90, 90, 100, 255))
        else:      # news
            d.rectangle([0, 0, W - 1, H - 1], fill=DARKBOX)
            d.rectangle([0, 0, 8, H], fill=RED)
            d.text((30, 16), name[:26], font=f, fill=WHITE)
            if role:
                d.rectangle([0, H - 44, W, H], fill=RED)
                d.text((30, H - 40), role[:44], font=fr, fill=WHITE)
    return _text_card(draw, W, H, out_path)


@_fail_soft_list
def location_reveal(city: str, *, out_dir, pin_color: tuple = RED,
                    font_path: Optional[str] = None,
                    prefix: str = "loc") -> list[tuple[str, float, float]]:
    """Map-pin + city name typed out — dateline moments."""
    from PIL import Image, ImageDraw
    os.makedirs(out_dir, exist_ok=True)
    wins = typewriter(city, out_dir=out_dir, char_per_sec=20, hold=1.6,
                      font_path=font_path, font_size=48, prefix=f"{prefix}t",
                      color=WHITE, bg=(0, 0, 0, 0), cursor=False)
    out = []
    for k, (p, a, b) in enumerate(wins):
        base = Image.open(p).convert("RGBA")
        W, H = base.width + 96, max(base.height, 96)
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, W - 1, H - 1], radius=12, fill=DARKBOX)
        # pin: circle + triangle
        d.ellipse([24, 14, 64, 54], fill=pin_color)
        d.polygon([(30, 46), (58, 46), (44, 80)], fill=pin_color)
        d.ellipse([38, 28, 50, 40], fill=WHITE)
        img.alpha_composite(base, (88, (H - base.height) // 2))
        p2 = os.path.join(out_dir, f"_{prefix}_{k:04d}.png")
        img.save(p2, "PNG")
        out.append((p2, a, b))
    return out


def cta_card(kind: str, *, out_path, font_path: Optional[str] = None) -> Optional[str]:
    """Call-to-action chip: subscribe / follow / watch_next / like_share."""
    f = _font(font_path, 44)
    txt = {"subscribe": "SUBSCRIBE", "follow": "FOLLOW US",
           "watch_next": "WATCH NEXT ▶", "like_share": "LIKE · SHARE"}.get(kind, kind.upper())
    bg = {"subscribe": (200, 30, 30, 240), "follow": (30, 90, 200, 240),
          "watch_next": (10, 10, 14, 230), "like_share": (120, 40, 200, 240)}.get(kind, DARKBOX)
    W, H = 460, 96

    def draw(img, d):
        d.rounded_rectangle([0, 0, W - 1, H - 1], radius=48, fill=bg)
        tw = d.textbbox((0, 0), txt, font=f)[2]
        d.text(((W - tw) // 2, (H - 44) // 2 - 6), txt, font=f, fill=WHITE)
    return _text_card(draw, W, H, out_path)


def big_number(value: str, *, out_path, label: str = "",
               accent: tuple = RED, font_path: Optional[str] = None) -> Optional[str]:
    """Giant standalone stat (₹300Cr, 5X, 90%…) with a label bar."""
    fbig, fl = _font(font_path, 160), _font(font_path, 42)
    W, H = 900, 340

    def draw(img, d):
        tw = d.textbbox((0, 0), value, font=fbig)[2]
        d.text(((W - tw) // 2, 10), value, font=fbig, fill=WHITE,
               stroke_width=4, stroke_fill=accent)
        if label:
            lw = d.textbbox((0, 0), label, font=fl)[2]
            d.rounded_rectangle([(W - lw) // 2 - 20, 232,
                                 (W + lw) // 2 + 20, 300], radius=10,
                                fill=accent)
            d.text(((W - lw) // 2, 242), label, font=fl, fill=WHITE)
    return _text_card(draw, W, H, out_path)


@_fail_soft_list
def list_reveal(items: list, *, out_dir, style: str = "top5",
                per_item: float = 0.9, t_start: float = 0.0,
                hold: float = 2.0, font_path: Optional[str] = None,
                prefix: str = "lst") -> list[tuple[str, float, float]]:
    """Items appear one by one (top-5 countdowns, checklists)."""
    from PIL import Image, ImageDraw
    os.makedirs(out_dir, exist_ok=True)
    f = _font(font_path, 44)
    items = [str(x)[:42] for x in (items or [])][:6]
    if not items:
        return []
    W, row = 820, 76
    H = 40 + row * len(items)
    out = []
    for shown in range(1, len(items) + 1):
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, W - 1, H - 1], radius=16, fill=DARKBOX)
        for i in range(shown):
            y = 24 + i * row
            if style == "checklist":
                d.rounded_rectangle([24, y + 4, 60, y + 40], radius=8,
                                    fill=GREEN)
                d.line([(32, y + 22), (42, y + 32), (54, y + 12)],
                       fill=WHITE, width=5)
                d.text((80, y), items[i], font=f, fill=WHITE)
            else:
                d.text((28, y), f"{i + 1}.", font=f, fill=HILITE)
                d.text((92, y), items[i], font=f, fill=WHITE)
        p = os.path.join(out_dir, f"_{prefix}_{shown:02d}.png")
        img.save(p, "PNG")
        a = t_start + (shown - 1) * per_item
        b = a + per_item if shown < len(items) else a + per_item + hold
        out.append((p, round(a, 3), round(b, 3)))
    return out


def chip(text: str, *, out_path, kind: str = "timestamp",
         font_path: Optional[str] = None) -> Optional[str]:
    """Small info chip: timestamp / hashtag / breaking / live_now."""
    f = _font(font_path, 34)
    bg = {"breaking": (200, 20, 20, 245), "live_now": (200, 20, 20, 245),
          "hashtag": (30, 90, 200, 235)}.get(kind, DARKBOX)
    label = {"breaking": "BREAKING", "live_now": "● LIVE NOW"}.get(kind, text)
    W, H = 380, 66

    def draw(img, d):
        d.rounded_rectangle([0, 0, W - 1, H - 1], radius=33, fill=bg)
        tw = d.textbbox((0, 0), label, font=f)[2]
        d.text((max(18, (W - tw) // 2), 14), label[:24], font=f, fill=WHITE)
    return _text_card(draw, W, H, out_path)


@_fail_soft_list
def word_flash(word: str, *, out_dir, color: tuple = WHITE,
               t_start: float = 0.0, dur: float = 0.5,
               font_path: Optional[str] = None,
               prefix: str = "wf") -> list[tuple[str, float, float]]:
    """One giant word flashed full-width (hype cuts, emphasis)."""
    from PIL import Image, ImageDraw
    os.makedirs(out_dir, exist_ok=True)
    f = _font(font_path, 170)
    t = (word or "").upper()[:14]
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    box = probe.textbbox((0, 0), t, font=f)
    W, H = box[2] + 80, box[3] - box[1] + 80
    alphas = (0.55, 1.0, 0.75)
    out = []
    step = dur / len(alphas)
    for i, al in enumerate(alphas):
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.text((40 - box[0], 40 - box[1]), t, font=f,
               fill=(color[0], color[1], color[2], int(255 * al)),
               stroke_width=6, stroke_fill=(0, 0, 0, int(200 * al)))
        p = os.path.join(out_dir, f"_{prefix}_{i}.png")
        img.save(p, "PNG")
        out.append((p, round(t_start + i * step, 3),
                    round(t_start + (i + 1) * step, 3)))
    return out


def split_headline(line1: str, line2: str, *, out_path,
                   accent: tuple = RED, font_path: Optional[str] = None) -> Optional[str]:
    """Two-deck headline: colored kicker + white main line."""
    f1, f2 = _font(font_path, 44), _font(font_path, 72)
    W, H = 1100, 200

    def draw(img, d):
        w1 = d.textbbox((0, 0), line1.upper(), font=f1)[2]
        d.rectangle([0, 0, w1 + 48, 66], fill=accent)
        d.text((24, 8), line1.upper()[:36], font=f1, fill=WHITE)
        d.rectangle([0, 66, W, H], fill=DARKBOX)
        d.text((24, 84), line2[:40], font=f2, fill=WHITE)
    return _text_card(draw, W, H, out_path)


def subtitle_line(text: str, *, out_path, style: str = "classic",
                  font_path: Optional[str] = None) -> Optional[str]:
    """Plain bottom subtitle (classic shadow / boxed)."""
    f = _font(font_path, 46)
    W, H = 1000, 92

    def draw(img, d):
        t = " ".join((text or "").split())[:60]
        tw = d.textbbox((0, 0), t, font=f)[2]
        if style == "boxed":
            d.rounded_rectangle([(W - tw) // 2 - 24, 6,
                                 (W + tw) // 2 + 24, H - 6],
                                radius=10, fill=(0, 0, 0, 200))
        d.text(((W - tw) // 2, 18), t, font=f, fill=WHITE,
               stroke_width=0 if style == "boxed" else 3,
               stroke_fill=(0, 0, 0, 230))
    return _text_card(draw, W, H, out_path)


# ── VARIANT REGISTRY — the ~60 text effects the admin tab lists ─────
# Each row: engine + kwargs. render_variant() renders a representative
# still for previews; the engines themselves return timed PNG windows
# for the compose pipeline.

_KARAOKE_COLORS = [("yellow", HILITE), ("red", (235, 60, 50, 255)),
                   ("cyan", CYAN), ("green", GREEN), ("orange", ORANGE)]
_CRASH_MOTIONS = ["crash", "bounce", "slide_up", "slide_down",
                  "whip_left", "whip_right", "fade_scale", "spin_in"]

VARIANTS: list[dict] = []
for _st in ("hilite", "box", "underline", "pop"):
    for _cn, _cv in _KARAOKE_COLORS:
        VARIANTS.append({
            "id": f"karaoke_{_st}_{_cn}",
            "label": f"Karaoke captions — {_st} ({_cn})",
            "used_for": "Word-by-word speech-synced captions (shorts/reels)",
            "engine": "karaoke", "kwargs": {"style": _st, "hilite": _cv}})
# ── Premium karaoke set — engine="karaoke" ⇒ auto-enters the Director
#    caption vocabulary (director._vocab) and the admin catalog.
VARIANTS += [
    {"id": "karaoke_kinetic_yellow",
     "label": "Karaoke — kinetic word pop (yellow chip)",
     "used_for": "High-energy shorts: the spoken word pops up boxed in accent",
     "engine": "karaoke", "kwargs": {"style": "kinetic", "hilite": HILITE}},
    {"id": "karaoke_chips_red",
     "label": "Karaoke — boxed reveal (word chips, red)",
     "used_for": "Every word in a filled chip; the spoken chip lights up",
     "engine": "karaoke", "kwargs": {"style": "chips",
                                     "hilite": (235, 60, 50, 255)}},
    {"id": "karaoke_glow_gold",
     "label": "Karaoke — gradient glow (gold)",
     "used_for": "Premium look: spoken word gradient-filled with a soft halo",
     "engine": "karaoke", "kwargs": {"style": "glow", "hilite": GOLD}},
    {"id": "karaoke_flip_duo",
     "label": "Karaoke — accent flip (yellow / cyan)",
     "used_for": "Rhythmic reels: the accent colour alternates per word",
     "engine": "karaoke", "kwargs": {"style": "flip", "hilite": HILITE,
                                     "hilite2": CYAN}},
    {"id": "karaoke_impact_promo",
     "label": "Karaoke — big impact (broadcast promo)",
     "used_for": "Heavy-weight bar-less captions with a thin outline",
     "engine": "karaoke", "kwargs": {"style": "impact", "hilite": HILITE}},
]
for _m in _CRASH_MOTIONS:
    VARIANTS.append({
        "id": f"headline_{_m}",
        "label": f"Headline — {_m.replace('_', ' ')}",
        "used_for": "Breaking headlines / trailer cards with physics entrances",
        "engine": "crash", "kwargs": {"motion": _m}})
VARIANTS += [
    {"id": "typewriter_classic", "label": "Typewriter (terminal green)",
     "used_for": "Investigations, case files", "engine": "typewriter",
     "kwargs": {}},
    {"id": "typewriter_white", "label": "Typewriter (white card)",
     "used_for": "Documents, quotes", "engine": "typewriter",
     "kwargs": {"color": (20, 20, 24, 255), "bg": (245, 245, 246, 235)}},
    {"id": "typewriter_plain", "label": "Typewriter (no cursor)",
     "used_for": "Clean captions typed in", "engine": "typewriter",
     "kwargs": {"cursor": False, "color": WHITE}},
    {"id": "typewriter_fast", "label": "Typewriter (rapid)",
     "used_for": "Fast data readouts", "engine": "typewriter",
     "kwargs": {"char_per_sec": 34.0}},
    {"id": "typewriter_red", "label": "Typewriter (alert red)",
     "used_for": "Warnings, alerts", "engine": "typewriter",
     "kwargs": {"color": (255, 90, 80, 255)}},
    {"id": "counter_plain", "label": "Count-up number",
     "used_for": "Any rising stat", "engine": "counter",
     "kwargs": {"value": 5200, "label": "cases resolved"}},
    {"id": "counter_money", "label": "Count-up money",
     "used_for": "Budgets, scams, box office", "engine": "counter",
     "kwargs": {"value": 300, "prefix_symbol": "₹", "suffix": " Cr",
                "label": "total cost", "color": GOLD}},
    {"id": "counter_percent", "label": "Count-up percent",
     "used_for": "Growth, polls, shares", "engine": "counter",
     "kwargs": {"value": 92, "suffix": "%", "label": "of voters",
                "color": GREEN}},
    {"id": "quote_news", "label": "Quote card — newsroom",
     "used_for": "Statements, reactions", "engine": "quote",
     "kwargs": {"style": "news"}},
    {"id": "quote_dark", "label": "Quote card — dark gold",
     "used_for": "Dramatic statements", "engine": "quote",
     "kwargs": {"style": "dark"}},
    {"id": "quote_minimal", "label": "Quote card — minimal light",
     "used_for": "Lifestyle, positive quotes", "engine": "quote",
     "kwargs": {"style": "minimal"}},
    {"id": "nametag_news", "label": "Name tag — news bar",
     "used_for": "Speaker identification", "engine": "nametag",
     "kwargs": {"style": "news"}},
    {"id": "nametag_clean", "label": "Name tag — clean light",
     "used_for": "Interviews, podcasts", "engine": "nametag",
     "kwargs": {"style": "clean"}},
    {"id": "nametag_badge", "label": "Name tag — avatar badge",
     "used_for": "Panel speakers", "engine": "nametag",
     "kwargs": {"style": "badge"}},
    {"id": "location_pin_red", "label": "Location reveal — red pin",
     "used_for": "Datelines, on-the-ground reports", "engine": "location",
     "kwargs": {"pin_color": RED}},
    {"id": "location_pin_gold", "label": "Location reveal — gold pin",
     "used_for": "Travel, heritage stories", "engine": "location",
     "kwargs": {"pin_color": GOLD}},
    {"id": "cta_subscribe", "label": "CTA — Subscribe",
     "used_for": "End cards, mid-roll CTAs", "engine": "cta",
     "kwargs": {"kind": "subscribe"}},
    {"id": "cta_follow", "label": "CTA — Follow us",
     "used_for": "Social handles", "engine": "cta",
     "kwargs": {"kind": "follow"}},
    {"id": "cta_watch_next", "label": "CTA — Watch next",
     "used_for": "End screens", "engine": "cta",
     "kwargs": {"kind": "watch_next"}},
    {"id": "cta_like_share", "label": "CTA — Like & share",
     "used_for": "Engagement prompts", "engine": "cta",
     "kwargs": {"kind": "like_share"}},
    {"id": "bignum_red", "label": "Giant stat — red",
     "used_for": "Shock numbers", "engine": "bignum",
     "kwargs": {"value": "5X", "label": "faster", "accent": RED}},
    {"id": "bignum_gold", "label": "Giant stat — gold",
     "used_for": "Money numbers", "engine": "bignum",
     "kwargs": {"value": "₹300Cr", "label": "scam size", "accent": GOLD}},
    {"id": "bignum_clean", "label": "Giant stat — clean",
     "used_for": "Neutral stats", "engine": "bignum",
     "kwargs": {"value": "90%", "label": "approval",
                "accent": (90, 90, 100, 255)}},
    {"id": "list_top5", "label": "Top-5 list reveal",
     "used_for": "Countdowns, rankings", "engine": "list",
     "kwargs": {"style": "top5",
                "items": ["Hyderabad", "Vijayawada", "Warangal",
                          "Guntur", "Tirupati"]}},
    {"id": "list_checklist", "label": "Checklist reveal",
     "used_for": "How-tos, requirements", "engine": "list",
     "kwargs": {"style": "checklist",
                "items": ["Aadhaar card", "Passport photo",
                          "Address proof"]}},
    {"id": "chip_timestamp", "label": "Timestamp chip",
     "used_for": "When it happened", "engine": "chip",
     "kwargs": {"kind": "timestamp", "text": "06 JUL · 11:40 PM"}},
    {"id": "chip_hashtag", "label": "Hashtag chip",
     "used_for": "Trending tags on screen", "engine": "chip",
     "kwargs": {"kind": "hashtag", "text": "#Elections2026"}},
    {"id": "chip_breaking", "label": "BREAKING chip",
     "used_for": "Urgent flags", "engine": "chip",
     "kwargs": {"kind": "breaking", "text": "BREAKING"}},
    {"id": "chip_live", "label": "LIVE NOW chip",
     "used_for": "Live coverage flags", "engine": "chip",
     "kwargs": {"kind": "live_now", "text": "LIVE"}},
    {"id": "flash_white", "label": "Word flash — white",
     "used_for": "Hype single words", "engine": "flash",
     "kwargs": {"color": WHITE, "word": "EXPOSED"}},
    {"id": "flash_red", "label": "Word flash — red",
     "used_for": "Shock words", "engine": "flash",
     "kwargs": {"color": (235, 50, 40, 255), "word": "SCAM"}},
    {"id": "flash_yellow", "label": "Word flash — yellow",
     "used_for": "Highlight words", "engine": "flash",
     "kwargs": {"color": HILITE, "word": "WOW"}},
    {"id": "split_red", "label": "Two-deck headline — red kicker",
     "used_for": "Kicker + main headline", "engine": "split",
     "kwargs": {"accent": RED, "line1": "Breaking",
                "line2": "Cabinet reshuffle tonight"}},
    {"id": "split_gold", "label": "Two-deck headline — gold kicker",
     "used_for": "Feature stories", "engine": "split",
     "kwargs": {"accent": GOLD, "line1": "Exclusive",
                "line2": "Inside the deal"}},
    {"id": "subtitle_classic", "label": "Subtitle — classic shadow",
     "used_for": "Translations, quiet captions", "engine": "subtitle",
     "kwargs": {"style": "classic", "text": "మాట్లాడుతున్నారు…"}},
    {"id": "subtitle_boxed", "label": "Subtitle — boxed",
     "used_for": "High-contrast captions", "engine": "subtitle",
     "kwargs": {"style": "boxed", "text": "అధికారిక ప్రకటన"}},
]


def render_variant(variant_id: str, out_dir) -> Optional[str]:
    """Render ONE representative still of a catalog variant (admin
    preview). Timed engines render a mid-animation state."""
    import os as _os
    row = next((v for v in VARIANTS if v["id"] == variant_id), None)
    if not row:
        return None
    _os.makedirs(out_dir, exist_ok=True)
    kw = dict(row["kwargs"])
    eng = row["engine"]
    dst = _os.path.join(out_dir, f"_var_{variant_id}.png")
    try:
        if eng == "karaoke":
            words = [{"w": "మోదీ", "s": 0.0, "e": 0.4},
                     {"w": "గారి", "s": 0.5, "e": 0.9},
                     {"w": "ప్రకటన", "s": 1.0, "e": 1.4}]
            wins = karaoke_captions(words, out_dir=out_dir,
                                    prefix=f"v_{variant_id}", **kw)
            return wins[1][0] if len(wins) > 1 else (wins[0][0] if wins else None)
        if eng == "crash":
            wins = headline_crash("BREAKING NEWS", out_dir=out_dir, frames=10,
                                  prefix=f"v_{variant_id}", **kw)
            return wins[-1][0] if wins else None
        if eng == "typewriter":
            wins = typewriter("CASE FILE #42", out_dir=out_dir,
                              prefix=f"v_{variant_id}", **kw)
            return wins[len(wins) // 2][0] if wins else None
        if eng == "counter":
            wins = counter(out_dir=out_dir, prefix=f"v_{variant_id}", **kw)
            return wins[-1][0] if wins else None
        if eng == "quote":
            return quote_card("నిజం ఎప్పటికీ దాచలేరు", author="Ravi Kumar",
                              out_path=dst, **kw)
        if eng == "nametag":
            return name_tag("Dr. Priya Sharma", role="Political Analyst",
                            out_path=dst, **kw)
        if eng == "location":
            wins = location_reveal("HYDERABAD", out_dir=out_dir,
                                   prefix=f"v_{variant_id}", **kw)
            return wins[-1][0] if wins else None
        if eng == "cta":
            return cta_card(out_path=dst, **kw)
        if eng == "bignum":
            return big_number(out_path=dst, **kw)
        if eng == "list":
            items = kw.pop("items")
            wins = list_reveal(items, out_dir=out_dir,
                               prefix=f"v_{variant_id}", **kw)
            return wins[-1][0] if wins else None
        if eng == "chip":
            text = kw.pop("text", "")
            return chip(text, out_path=dst, **kw)
        if eng == "flash":
            word = kw.pop("word")
            wins = word_flash(word, out_dir=out_dir,
                              prefix=f"v_{variant_id}", **kw)
            return wins[1][0] if len(wins) > 1 else (wins[0][0] if wins else None)
        if eng == "split":
            l1, l2 = kw.pop("line1"), kw.pop("line2")
            return split_headline(l1, l2, out_path=dst, **kw)
        if eng == "subtitle":
            text = kw.pop("text")
            return subtitle_line(text, out_path=dst, **kw)
    except Exception as exc:
        print(f"[v4/typography] render_variant({variant_id}) failed: {exc}",
              flush=True)
    return None


# Catalog for the Admin Features tab — one row per variant.
CATALOG = [{"id": v["id"], "label": v["label"], "used_for": v["used_for"]}
           for v in VARIANTS]
