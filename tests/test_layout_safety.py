"""Unit 10 — layout safety: media never covers the headline/ticker text."""
from __future__ import annotations

from pipeline_v4 import layout_safety as ls
from pipeline_v4.canvas_schema import CanvasLayout, CanvasStory, CanvasTextBlock


def _framed(**over):
    """The stock framed-bulletin geometry in pcts (tiles 50..850 on 1080)."""
    base = dict(video_x_pct=1.5625, video_y_pct=4.63, video_w_pct=65.625,
                video_h_pct=74.07, picture_x_pct=68.75, picture_y_pct=4.63,
                picture_w_pct=29.6875, picture_h_pct=74.07)
    base.update(over)
    return CanvasLayout(**base)


def test_framed_bulletin_layout_is_safe():
    """Tiles ending at ~850 sit clear of the LT strip at 890."""
    assert ls.check(_framed()) == []


def test_audio_fullscreen_layout_is_exempt():
    """The audio-first fullscreen layout (video AND picture 100%x100%)
    is a deliberate full-bleed backdrop — text draws on top with opaque
    strips → exempt."""
    layout = CanvasLayout(
        video_x_pct=0.0, video_y_pct=0.0, video_w_pct=100.0, video_h_pct=100.0,
        picture_x_pct=0.0, picture_y_pct=0.0, picture_w_pct=100.0,
        picture_h_pct=100.0,
    )
    assert ls.check(layout) == []


def test_constants_fallback_layout_is_safe():
    """A layout with zeroed pcts falls back to the V4 constants — safe."""
    layout = CanvasLayout(video_w_pct=0, video_h_pct=0,
                          picture_w_pct=0, picture_h_pct=0)
    assert ls.check(layout) == []


def test_tile_dragged_into_ticker_is_flagged():
    # tiles stretched down to y ≈ 1026 → inside the LT strip
    viol = ls.check(_framed(video_h_pct=90.0, picture_h_pct=90.0))
    assert viol and any("lower_third" in v for v in viol)


def test_positioned_text_block_participates():
    """A custom-positioned text block in the middle of the canvas makes
    an overlapping framed tile a violation."""
    story = CanvasStory(
        story_index=0, video_t_start=0.0, video_t_end=10.0,
        text_blocks=[CanvasTextBlock(kind="custom", text="hi",
                                     x_pct=5.0, y_pct=40.0, w_pct=50.0,
                                     font_size_pct=5.0)],
    )
    viol = ls.check(_framed(), [story])
    assert viol and any("text:custom" in v for v in viol)


def test_clamp_pulls_media_out_of_text_strip():
    layout = _framed(video_h_pct=90.0, picture_h_pct=90.0)
    assert ls.check(layout)                # unsafe before
    assert ls.clamp_media(layout) is True
    assert ls.check(layout) == []          # safe after
    # bottom edge: y + h ≤ 890 - 8
    bottom = 4.63 / 100 * 1080 + layout.video_h_pct / 100 * 1080
    assert bottom <= 1080 - ls.LT_H - ls.TICKER_H - ls.DEFAULT_PAD + 0.5


def test_clamp_noop_on_safe_default_and_fullbleed_layouts():
    assert ls.clamp_media(_framed()) is False       # already safe
    assert ls.clamp_media(CanvasLayout()) is False  # full-bleed backdrop
    consts = CanvasLayout(video_w_pct=0, video_h_pct=0,
                          picture_w_pct=0, picture_h_pct=0)
    assert ls.clamp_media(consts) is False
    assert ls.clamp_media(None) is False


def test_safe_rect_sits_above_text():
    x, y, w, h = ls.safe_rect(CanvasLayout())
    assert y + h <= 1080 - ls.LT_H - ls.TICKER_H - ls.DEFAULT_PAD + 0.5
    assert x >= 0 and w > 0 and h > 0
    # grid planner accepts it directly
    from pipeline_v4 import grid_engine as ge
    pages = ge.plan_grid(4, duration=10.0,
                         safe_rect=(int(x), int(y), int(w), int(h)))
    for (cx, cy, cw, ch) in pages[0].cells:
        assert cy + ch <= y + h + 0.5


def test_telugu_long_headline_marquee_is_fullwidth():
    """The LT strip is full-width by design — a tile parked at the far
    right is still flagged when it dips into the strip (marquee text
    scrolls across the whole canvas)."""
    layout = CanvasLayout(video_x_pct=1.5, video_y_pct=5.0,
                          video_w_pct=20.0, video_h_pct=40.0,
                          picture_x_pct=75.0, picture_y_pct=5.0,
                          picture_w_pct=24.0, picture_h_pct=85.0)
    viol = ls.check(layout)
    assert viol and any("picture" in v for v in viol)
