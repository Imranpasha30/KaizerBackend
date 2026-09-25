"""Unit 5 — image gaps → main video full-screen.

Three layers:
  1. `_gap_windows` pure math (complement of image windows).
  2. Filter-graph SNAPSHOT: with no gaps, `_build_story_filter_graph`
     must emit a CHARACTER-IDENTICAL graph to the legacy inline code —
     the byte-compat guarantee for every pre-engine canvas.
  3. A slow ffmpeg smoke: during a gap the anchor video fills the frame;
     during an image window the framed layout shows.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline_v4 import v1_bridge as vb


# ─── 1) _gap_windows pure math ──────────────────────────────────────


@pytest.fixture
def pool(tmp_path):
    d = tmp_path / "_pool"
    d.mkdir()
    for name in ("a.jpg", "b.jpg"):
        (d / name).write_bytes(b"\xff\xd8\xff" + b"0" * 100)   # enough for stat()
    return d


def _img(src, ts, te):
    return {"src": src, "t_start": ts, "t_end": te}


def test_gaps_contiguous_coverage_yields_none(pool):
    imgs = [_img("a.jpg", 0.0, 4.0), _img("b.jpg", 4.0, 10.0)]
    assert vb._gap_windows(imgs, pool, 10.0) == []


def test_gaps_hole_and_tail(pool):
    imgs = [_img("a.jpg", 3.0, 7.0)]
    assert vb._gap_windows(imgs, pool, 10.0) == [(0.0, 3.0), (7.0, 10.0)]


def test_gaps_empty_images_stay_legacy(pool):
    """No resolvable windows → NO gap fallback: legacy no-image stories
    keep their placeholder-sidebar look and their cache hash."""
    assert vb._gap_windows([], pool, 10.0) == []
    assert vb._gap_windows([_img("missing.jpg", 0, 4)], pool, 10.0) == []


def test_gaps_sliver_skipped_and_overlaps_merged(pool):
    imgs = [_img("a.jpg", 0.0, 4.0), _img("b.jpg", 4.3, 10.0)]   # 0.3s sliver
    assert vb._gap_windows(imgs, pool, 10.0) == []
    imgs = [_img("a.jpg", 0.0, 5.0), _img("b.jpg", 3.0, 6.0)]    # overlap → one block
    assert vb._gap_windows(imgs, pool, 10.0) == [(6.0, 10.0)]


def test_gaps_capped_to_longest(pool):
    imgs = [_img("a.jpg", i * 2.0, i * 2.0 + 1.0) for i in range(20)]  # 19 1s gaps
    gaps = vb._gap_windows(imgs, pool, 40.0, cap=5)
    assert len(gaps) == 5
    assert gaps == sorted(gaps)


# ─── 2) Filter-graph snapshot (byte-compat guarantee) ───────────────


def _legacy_graph(*, canvas_w, canvas_h, main_inner_w, side_inner_w,
                  tile_inner_h, main_outer_w, side_outer_w, tile_outer_h,
                  main_x, side_x, tile_y, lt_y, ticker_y, border_colour,
                  bg_color, lt_x_expr, ticker_speed_px_s,
                  has_bg, bg_input_idx, apply_ticker, ticker_in_idx,
                  has_bug, bug_in_idx, has_wm, wm_in_idx,
                  watermark_xy, bg_video_volume):
    """Verbatim replica of the pre-Unit-5 inline construction in
    _compose_v4_bulletin_story — the frozen reference the refactor must
    reproduce character-for-character when no gaps exist."""
    fc = [
        f"[0:v]scale={main_inner_w}:{tile_inner_h}:"
        f"force_original_aspect_ratio=increase,"
        f"crop={main_inner_w}:{tile_inner_h},setsar=1,"
        f"pad={main_outer_w}:{tile_outer_h}:"
        f"{vb.V4_TILE_BORDER}:{vb.V4_TILE_BORDER}:color={border_colour}[main_v]",

        f"[1:v]scale={side_inner_w}:{tile_inner_h}:"
        f"force_original_aspect_ratio=increase,"
        f"crop={side_inner_w}:{tile_inner_h},setsar=1,"
        f"pad={side_outer_w}:{tile_outer_h}:"
        f"{vb.V4_TILE_BORDER}:{vb.V4_TILE_BORDER}:color={border_colour}[side_v]",
    ]
    if has_bg:
        fc.append(
            f"[{bg_input_idx}:v]scale={canvas_w}:{canvas_h}:"
            f"force_original_aspect_ratio=increase,"
            f"crop={canvas_w}:{canvas_h},setsar=1,fps=30[bg]"
        )
    else:
        fc.append(
            f"color=c={bg_color}:"
            f"s={canvas_w}x{canvas_h}:r=30[bg]"
        )
    fc += [
        f"[bg][main_v]overlay=x={main_x}:y={tile_y}:shortest=1[stage_m]",
        f"[stage_m][side_v]overlay=x={side_x}:y={tile_y}[stage_top]",

        f"[stage_top][2:v]overlay=x='{lt_x_expr}':y={lt_y}:format=auto[stage_lt]",
    ]
    cursor = "stage_lt"
    if apply_ticker and ticker_in_idx is not None:
        fc.append(
            f"[{cursor}][{ticker_in_idx}:v]overlay="
            f"x='W-mod(t*{ticker_speed_px_s:.1f}\\,w+W)':"
            f"y={ticker_y}:format=auto[stage_ticker]"
        )
        cursor = "stage_ticker"
    if has_bug:
        fc.append(
            f"[{cursor}][{bug_in_idx}:v]overlay="
            f"x=W-w-{vb.V4_SIDE_MARGIN}:y={tile_y}:format=auto[stage_bug]"
        )
        cursor = "stage_bug"
    last_stage = cursor
    if has_wm:
        wx, wy = watermark_xy
        fc.append(
            f"[{last_stage}][{wm_in_idx}:v]overlay="
            f"x={wx}:y={wy}:format=auto[outv]"
        )
        out_label = "outv"
    else:
        out_label = last_stage
    audio_map = "0:a?"
    if has_bg and bg_video_volume > 0.0:
        fc.append(
            f"[0:a]volume=1.0[a0];"
            f"[{bg_input_idx}:a]volume={bg_video_volume:.3f}[abg];"
            f"[a0][abg]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
        )
        audio_map = "[aout]"
    return ";".join(fc), out_label, audio_map


_MINIMAL = dict(
    canvas_w=1920, canvas_h=1080,
    main_inner_w=1254, side_inner_w=564, tile_inner_h=794,
    main_outer_w=1260, side_outer_w=570, tile_outer_h=800,
    main_x=30, side_x=1320, tile_y=50, lt_y=890, ticker_y=1030,
    border_colour="white", bg_color="black",
    lt_x_expr="if(lt(t\\,0.4)\\,-w+w*t/0.4\\,0)", ticker_speed_px_s=200.0,
    has_bg=False, bg_input_idx=None, apply_ticker=False, ticker_in_idx=None,
    has_bug=False, bug_in_idx=None, has_wm=False, wm_in_idx=None,
    watermark_xy=("", ""), bg_video_volume=0.0,
)

_FULL = dict(
    _MINIMAL,
    has_bg=True, bg_input_idx=6, apply_ticker=True, ticker_in_idx=3,
    has_bug=True, bug_in_idx=4, has_wm=True, wm_in_idx=5,
    watermark_xy=("W-w-24", "24"), bg_video_volume=0.35,
)


def test_no_gap_graph_is_character_identical_minimal():
    assert vb._build_story_filter_graph(**_MINIMAL, gap_windows=()) \
        == _legacy_graph(**_MINIMAL)


def test_no_gap_graph_is_character_identical_full():
    assert vb._build_story_filter_graph(**_FULL, gap_windows=()) \
        == _legacy_graph(**_FULL)


def test_gap_graph_shape():
    fcs, out_label, audio_map = vb._build_story_filter_graph(
        **_MINIMAL, gap_windows=((0.0, 3.0), (7.0, 10.0)))
    assert fcs.startswith("[0:v]split=2[v_tile][v_full];[v_tile]scale=")
    assert "between(t,0.000,3.000)+between(t,7.000,10.000)" in fcs
    # gap overlay sits BEFORE the lower-third → text stays on top
    assert fcs.index("[stage_gap]") < fcs.index("[stage_lt]")
    assert "[stage_gap][2:v]overlay" in fcs
    assert out_label == "stage_lt"
    assert audio_map == "0:a?"


def test_gap_hash_ingredient_conditional(tmp_path):
    """gapfb participates in the story hash ONLY when gaps exist."""
    story = SimpleNamespace(title_native="t", title_english="t", summary="s",
                            video_t_start=0.0, video_t_end=10.0,
                            story_index=0, total_stories=1)
    def _h(gaps):
        return vb._per_story_cache_hash(
            story=story, ticker_path="", sidebar_path="", layout=None,
            channel_bug_path="", watermark_path="", watermark_position="",
            font_path="", bg_video_abs=None, bg_video_volume=0.0,
            language_code="te", images=None, pool_dir=None,
            gap_windows=gaps,
        )
    assert _h(()) == _h(())                       # deterministic
    assert _h(()) != _h(((0.0, 3.0),))            # gaps invalidate
    assert _h(((0.0, 3.0),)) == _h(((0.0, 3.0),))


# ─── 3) Carousel input cap (WinError 206 guard, job 592) ────────────
# Every segment adds one ffmpeg -i plus two filter stages; a story with
# dozens of images overflowed the ~32k Windows CreateProcess line. The
# cap keeps the longest-visible windows, renders chronologically, and
# fingerprints itself into the .sig ONLY when it fired.


def _capture_carousel_render(monkeypatch):
    captured = {}

    def _fake_render(*, segs, out_path, **_kw):
        captured["segs"] = segs
        Path(out_path).write_bytes(b"x")
        return out_path

    monkeypatch.setattr(vb, "_render_sidebar_carousel", _fake_render)
    return captured


def test_carousel_cap_keeps_longest_chronological(pool, tmp_path, monkeypatch):
    captured = _capture_carousel_render(monkeypatch)
    monkeypatch.delenv("KAIZER_V4_CAROUSEL_MAX_SEGS", raising=False)
    # 30 disjoint windows; window k lasts (1 + k*0.1)s → k=0..5 are the
    # six shortest and must be the ones dropped.
    imgs = [_img("a.jpg", k * 10.0, k * 10.0 + 1.0 + k * 0.1)
            for k in range(30)]
    out = tmp_path / "car.mp4"
    res = vb._ensure_sidebar_carousel(
        images=imgs, pool_dir=pool, out_path=str(out), story_duration=300.0)
    assert res == str(out)
    segs = captured["segs"]
    assert len(segs) == 24
    assert [s["ts"] for s in segs] == sorted(s["ts"] for s in segs)
    assert min(s["te"] - s["ts"] for s in segs) > 1.55   # shortest six gone
    import json
    sig = json.loads(Path(str(out) + ".sig").read_text(encoding="utf-8"))
    assert sig["cap"] == [24, 30]
    assert len(sig["segs"]) == 24


def test_carousel_cap_env_override_and_tie_break(pool, tmp_path, monkeypatch):
    captured = _capture_carousel_render(monkeypatch)
    monkeypatch.setenv("KAIZER_V4_CAROUSEL_MAX_SEGS", "5")
    # equal-duration windows → earliest win the tie
    imgs = [_img("a.jpg", k * 5.0, k * 5.0 + 2.0) for k in range(8)]
    out = tmp_path / "car.mp4"
    vb._ensure_sidebar_carousel(
        images=imgs, pool_dir=pool, out_path=str(out), story_duration=60.0)
    assert [s["ts"] for s in captured["segs"]] == [0.0, 5.0, 10.0, 15.0, 20.0]
    import json
    sig = json.loads(Path(str(out) + ".sig").read_text(encoding="utf-8"))
    assert sig["cap"] == [5, 8]


def test_carousel_under_cap_keeps_legacy_sig(pool, tmp_path, monkeypatch):
    captured = _capture_carousel_render(monkeypatch)
    monkeypatch.delenv("KAIZER_V4_CAROUSEL_MAX_SEGS", raising=False)
    imgs = [_img("a.jpg", k * 5.0, k * 5.0 + 3.0) for k in range(10)]
    out = tmp_path / "car.mp4"
    vb._ensure_sidebar_carousel(
        images=imgs, pool_dir=pool, out_path=str(out), story_duration=60.0)
    assert len(captured["segs"]) == 10           # untouched
    import json
    sig = json.loads(Path(str(out) + ".sig").read_text(encoding="utf-8"))
    assert "cap" not in sig                      # legacy .sig, no re-render


# ─── 4) ffmpeg smoke — anchor fills the frame during a gap ──────────


def _px(video, t):
    """Mean RGB of the 16x16 top-left corner at time t."""
    out = video.parent / f"probe_{t:.1f}.png"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(video),
         "-frames:v", "1", str(out)],
        check=True, capture_output=True,
    )
    from PIL import Image
    with Image.open(out) as im:
        im = im.convert("RGB").crop((0, 0, 16, 16))
        px = list(im.getdata())
    n = len(px)
    return tuple(sum(c[i] for c in px) / n for i in range(3))


@pytest.mark.slow
def test_gap_shows_video_fullscreen(tmp_path):
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("ffmpeg not available")

    # 10s RED story clip with silent audio — red at the canvas corner can
    # only come from the full-screen gap fallback (the framed layout puts
    # black bg + white border there).
    src = tmp_path / "story.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=c=red:s=640x360:r=30:d=10",
         "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
         "-t", "10", "-c:v", "libx264", "-preset", "ultrafast",
         "-c:a", "aac", "-shortest", str(src)],
        check=True, capture_output=True,
    )
    sidebar = tmp_path / "side.png"
    from PIL import Image
    Image.new("RGB", (640, 800), (0, 0, 255)).save(sidebar)

    from pipeline_core.longform_compose import StoryMeta
    meta = StoryMeta(title="test story", kicker="BREAKING", language="en",
                     story_index=0, total_stories=1)
    out = tmp_path / "composed.mp4"
    vb._compose_v4_bulletin_story(
        story_clip_path=str(src),
        story_meta=meta,
        out_path=str(out),
        sidebar_path=str(sidebar),
        ticker_path="",
        channel_bug_path=None,
        font_path=None,
        sidebar_is_video=False,
        work_dir=str(tmp_path),
        apply_ticker=False,
        gap_windows=((0.0, 3.0), (7.0, 10.0)),
    )
    assert out.is_file() and out.stat().st_size > 0

    in_gap = _px(out, 1.5)        # gap → full-screen red video
    in_image = _px(out, 5.0)      # image window → framed layout, black corner
    assert in_gap[0] > 150 and in_gap[1] < 90 and in_gap[2] < 90
    assert sum(in_image) < 120    # near-black canvas corner
