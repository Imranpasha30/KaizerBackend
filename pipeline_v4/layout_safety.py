"""Layout safety — media never covers the text (Phase-1 engine, spec 3.13).

Today the bulletin only avoids text-over-media collisions because the
default percentages happen to miss each other; the editor's drag-to-move
can violate that silently. This module makes the rule explicit:

  * ``text_rects``  — where the renderer actually burns text, mirroring
    ``_compose_v4_bulletin_story``'s geometry: the lower-third strip is
    treated as FULL-WIDTH (long Telugu headlines marquee across the whole
    canvas), the ticker as the bottom strip, plus any positioned
    ``CanvasTextBlock`` overrides.
  * ``media_rects`` — where the video tile and picture panel sit
    (layout pcts, falling back to the V4 constants exactly like the
    composer does).
  * ``check``       — violations list; [] = safe.
  * ``clamp_media`` — pull offending media rects up out of the text
    strip (used at canvas build so generated layouts are safe by
    construction).
  * ``safe_rect``   — the media-allowed band; grid/PiP planners take it
    as input so they CANNOT generate a violating cell.

Enforcement points: canvas build (clamp), the KAIZER_PRECOMPOSE_QC
advisory gate (fail-loud log), and the editor PUT (warn — the operator
may still force it).
"""
from __future__ import annotations

from typing import Optional

# Mirrors v1_bridge's fixed text geometry (heights are absolute px on
# any canvas; the composer computes lt_y = H - 140 - 50, ticker_y = H - 50).
LT_H = 140
TICKER_H = 50
DEFAULT_PAD = 8

Rect = tuple[float, float, float, float]   # (x, y, w, h)


def _layout_geom(layout) -> tuple[int, int]:
    w = int(getattr(layout, "width", None) or 1920)
    h = int(getattr(layout, "height", None) or 1080)
    return w, h


def text_rects(layout, stories=None) -> list[tuple[str, Rect]]:
    """[(name, rect), …] of every text region the renderer draws."""
    w, h = _layout_geom(layout)
    rects: list[tuple[str, Rect]] = [
        # Marquee-conservative: the LT can scroll across the full width.
        ("lower_third", (0.0, float(h - LT_H - TICKER_H), float(w), float(LT_H))),
        ("ticker", (0.0, float(h - TICKER_H), float(w), float(TICKER_H))),
    ]
    for s in (stories or []):
        for b in (getattr(s, "text_blocks", None) or []):
            bx = getattr(b, "x_pct", None)
            by = getattr(b, "y_pct", None)
            if bx is None or by is None:
                continue    # default-position block → covered by the strips
            bw = getattr(b, "w_pct", None) or 100.0
            fs = getattr(b, "font_size_pct", None) or 5.0
            rects.append((
                f"text:{getattr(b, 'kind', 'custom')}",
                (w * bx / 100.0, h * by / 100.0, w * bw / 100.0, h * fs / 100.0 * 1.4),
            ))
    return rects


def media_rects(layout) -> list[tuple[str, Rect]]:
    """[(name, rect), …] for the video tile + picture panel — the same
    pct-or-V4-defaults decision _compose_v4_bulletin_story makes."""
    from pipeline_v4.v1_bridge import (
        V4_MAIN_OUTER_W, V4_SIDE_OUTER_W, V4_TILE_OUTER_H,
        V4_MAIN_X, V4_SIDE_X, V4_TILE_Y,
    )
    w, h = _layout_geom(layout)
    use_pct = (layout is not None
               and getattr(layout, "video_w_pct", 0) and getattr(layout, "video_h_pct", 0)
               and getattr(layout, "picture_w_pct", 0) and getattr(layout, "picture_h_pct", 0))
    if use_pct:
        # Faithful to the composer: BOTH tiles share video_y_pct as their
        # top and max(video_h, picture_h) as their outer height.
        tile_h = max(layout.video_h_pct, layout.picture_h_pct) / 100.0 * h
        return [
            ("video", (layout.video_x_pct / 100.0 * w, layout.video_y_pct / 100.0 * h,
                       layout.video_w_pct / 100.0 * w, tile_h)),
            ("picture", (layout.picture_x_pct / 100.0 * w, layout.video_y_pct / 100.0 * h,
                         layout.picture_w_pct / 100.0 * w, tile_h)),
        ]
    return [
        ("video", (float(V4_MAIN_X), float(V4_TILE_Y),
                   float(V4_MAIN_OUTER_W), float(V4_TILE_OUTER_H))),
        ("picture", (float(V4_SIDE_X), float(V4_TILE_Y),
                     float(V4_SIDE_OUTER_W), float(V4_TILE_OUTER_H))),
    ]


def _intersects(a: Rect, b: Rect, pad: float = 0.0) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw + pad <= bx or bx + bw + pad <= ax
                or ay + ah + pad <= by or by + bh + pad <= ay)


def _is_full_bleed(rect: Rect, w: int, h: int) -> bool:
    """A media rect covering (nearly) the whole canvas is a deliberate
    full-screen backdrop — the audio-first fullscreen layout, the shorts
    look, the schema default. Text draws ON TOP of it with opaque
    strap/ticker strips, so it is exempt from the overlap rule (same
    reason the spotlight is exempt by construction)."""
    _x, _y, rw, rh = rect
    return rw >= 0.97 * w and rh >= 0.97 * h


def check(layout, stories=None, *, pad: float = DEFAULT_PAD) -> list[str]:
    """Violations ("media X overlaps text Y"); [] = layout is safe.
    Full-bleed backdrops are exempt (see ``_is_full_bleed``) — the rule
    targets FRAMED tiles dipping into the text band, where the strap
    visually crops the tile."""
    out: list[str] = []
    w, h = _layout_geom(layout)
    texts = text_rects(layout, stories)
    for m_name, m in media_rects(layout):
        if _is_full_bleed(m, w, h):
            continue
        for t_name, t in texts:
            if _intersects(m, t, pad):
                out.append(
                    f"{m_name} tile (y {m[1]:.0f}-{m[1] + m[3]:.0f}) overlaps "
                    f"{t_name} strip (y {t[1]:.0f}-{t[1] + t[3]:.0f}) — media must "
                    f"stay >= {pad}px above the text"
                )
    return out


def safe_rect(layout, *, pad: float = DEFAULT_PAD,
              margin_x: float = 30.0, margin_top: float = 50.0) -> Rect:
    """The band media may occupy: above the lower-third strip with the
    standard side/top margins. Grid/PiP planners take this as input so a
    violating cell can never be generated."""
    w, h = _layout_geom(layout)
    text_top = h - LT_H - TICKER_H
    return (margin_x, margin_top,
            max(1.0, w - 2 * margin_x),
            max(1.0, text_top - pad - margin_top))


def clamp_media(layout, *, pad: float = DEFAULT_PAD) -> bool:
    """Shrink the layout's tile heights so FRAMED media stays out of the
    text strip. Mutates ``layout`` (pct fields) in place; returns True
    when a change was made. No-ops: layouts already safe, the V4
    constants fallback (safe by construction: 50+800=850 < 890), and
    full-bleed backdrops (deliberate — see ``_is_full_bleed``)."""
    if layout is None:
        return False
    if not (getattr(layout, "video_w_pct", 0) and getattr(layout, "video_h_pct", 0)
            and getattr(layout, "picture_w_pct", 0) and getattr(layout, "picture_h_pct", 0)):
        return False
    w, h = _layout_geom(layout)
    text_top = h - LT_H - TICKER_H
    changed = False
    for w_attr, y_attr, h_attr in (("video_w_pct", "video_y_pct", "video_h_pct"),
                                   ("picture_w_pct", "video_y_pct", "picture_h_pct")):
        w_px = float(getattr(layout, w_attr, 0.0) or 0.0) / 100.0 * w
        y_px = float(getattr(layout, y_attr, 0.0) or 0.0) / 100.0 * h
        h_px = float(getattr(layout, h_attr, 0.0) or 0.0) / 100.0 * h
        if _is_full_bleed((0.0, y_px, w_px, h_px), w, h):
            continue
        # 0.6px slack absorbs the pct→px→pct rounding so the clamped
        # bottom can never re-cross the pad line.
        overshoot = (y_px + h_px) - (text_top - pad) + 0.6
        if overshoot > 0.6 and h_px > overshoot:
            setattr(layout, h_attr, round((h_px - overshoot) / h * 100.0, 3))
            changed = True
    return changed
