"""Unit 11 — polish A (name-strap) + polish C motion (Ken-Burns zoompan)."""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from pipeline_v4 import v1_bridge as vb


# ─── name-strap PNG ─────────────────────────────────────────────────


def test_strap_png_renders_fixed_height(tmp_path):
    out = tmp_path / "strap.png"
    assert vb._render_name_strap_png(text="CM at press meet",
                                     font_path=None, out_path=str(out))
    from PIL import Image
    with Image.open(out) as im:
        assert im.height == vb.NAME_STRAP_H
        assert 0 < im.width <= 700
        assert im.mode == "RGBA"


def test_strap_png_telugu_and_ellipsis(tmp_path):
    from PIL import Image
    te = tmp_path / "te.png"
    assert vb._render_name_strap_png(text="మోదీ పార్లమెంట్ ప్రసంగం",
                                     font_path=None, out_path=str(te))
    long = tmp_path / "long.png"
    assert vb._render_name_strap_png(text="word " * 60,
                                     font_path=None, out_path=str(long))
    with Image.open(long) as im:
        assert im.width <= 700


def test_strap_png_empty_or_blank_returns_none(tmp_path):
    assert vb._render_name_strap_png(text="", font_path=None,
                                     out_path=str(tmp_path / "x.png")) is None
    assert vb._render_name_strap_png(text="   ", font_path=None,
                                     out_path=str(tmp_path / "y.png")) is None


# ─── graph: strap overlay position + order ──────────────────────────


def _minimal():
    return dict(
        canvas_w=1920, canvas_h=1080,
        main_inner_w=1254, side_inner_w=564, tile_inner_h=794,
        main_outer_w=1260, side_outer_w=570, tile_outer_h=800,
        main_x=30, side_x=1320, tile_y=50, lt_y=890, ticker_y=1030,
        border_colour="white", bg_color="black",
        lt_x_expr="x", ticker_speed_px_s=200.0,
        has_bg=False, bg_input_idx=None, apply_ticker=False, ticker_in_idx=None,
        has_bug=False, bug_in_idx=None, has_wm=False, wm_in_idx=None,
        watermark_xy=("", ""), bg_video_volume=0.0,
    )


def test_strap_graph_position_and_order():
    fcs, out_label, _ = vb._build_story_filter_graph(
        **_minimal(),
        pip=((3, 2.0, 5.0),),
        straps=((4, 2.0, 5.0),),
    )
    # strap sits just above the LT strip, inside the safe zone
    assert f"y={890 - vb.NAME_STRAP_H - 8}" in fcs
    # drawn after PiP, before the lower-third
    assert fcs.index("[stage_pip0]") < fcs.index("[stage_ns0]") < fcs.index("[stage_lt]")
    assert "enable='between(t,2.000,5.000)'" in fcs
    assert out_label == "stage_lt"


def test_no_strap_keeps_legacy_graph():
    a = vb._build_story_filter_graph(**_minimal(), gap_windows=())
    b = vb._build_story_filter_graph(**_minimal(), gap_windows=(), straps=())
    assert a == b


def test_strap_hash_ingredient_conditional():
    story = SimpleNamespace(title_native="t", title_english="t", summary="s",
                            video_t_start=0.0, video_t_end=10.0,
                            story_index=0, total_stories=1)
    def _h(straps):
        return vb._per_story_cache_hash(
            story=story, ticker_path="", sidebar_path="", layout=None,
            channel_bug_path="", watermark_path="", watermark_position="",
            font_path="", bg_video_abs=None, bg_video_volume=0.0,
            language_code="te", images=None, pool_dir=None,
            name_straps=straps,
        )
    assert _h(()) == _h(())
    assert _h(()) != _h((("Modi PC", 0.0, 4.0),))
    assert _h((("Modi PC", 0.0, 4.0),)) != _h((("CM meet", 0.0, 4.0),))


# ─── Ken-Burns zoompan in the sidebar carousel ──────────────────────


@pytest.mark.slow
def test_zoom_in_carousel_renders_motion(tmp_path):
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("ffmpeg not available")
    from PIL import Image
    # A still with a bright center square — under zoompan the square GROWS,
    # so early vs late frames differ (a static render would be identical).
    img = Image.new("RGB", (1280, 720), (10, 10, 10))
    for x in range(560, 720):
        for y in range(300, 420):
            img.putpixel((x, y), (250, 250, 250))
    src = tmp_path / "a.jpg"
    img.save(src, "JPEG", quality=92)

    segs = vb._carousel_segments(
        [{"src": "a.jpg", "t_start": 0.0, "t_end": 8.0, "effect": "zoom_in"}],
        tmp_path,
    )
    assert segs and segs[0]["effect"] == "zoom_in"
    out = tmp_path / "car.mp4"
    res = vb._render_sidebar_carousel(
        segs=segs, out_path=str(out), story_duration=8.0,
        sidebar_w=564, sidebar_h=794,
    )
    assert res and out.stat().st_size > 0

    def frame(t):
        f = tmp_path / f"f{t:.0f}.png"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}",
                        "-i", str(out), "-frames:v", "1", str(f)],
                       check=True, capture_output=True)
        with Image.open(f) as im:
            return list(im.convert("L").resize((32, 32)).getdata())

    early, late = frame(0.5), frame(7.0)
    diff = sum(abs(a - b) for a, b in zip(early, late)) / len(early)
    assert diff > 1.0, "zoom_in produced a static image (no Ken-Burns motion)"
