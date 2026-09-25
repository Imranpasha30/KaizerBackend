"""Unit 7 — spotlight: key-moment images pop full-screen (text on top)."""
from __future__ import annotations

import subprocess

import pytest

from pipeline_v4 import image_timing as it
from pipeline_v4 import v1_bridge as vb


# ─── auto-selection (canvas-build time) ─────────────────────────────


def _entry(idx, ts, te, imp):
    return {"pool_index": idx, "t_start": ts, "t_end": te,
            "confidence": 0.9, "importance": imp}


def test_auto_spotlight_picks_top_important(monkeypatch):
    monkeypatch.delenv("KAIZER_V4_SPOTLIGHT", raising=False)
    plan = [_entry(0, 0, 4, 0.95), _entry(1, 5, 9, 0.5),
            _entry(2, 10, 14, 0.85), _entry(0, 15, 19, 0.9)]
    out = it.auto_spotlight(plan, min_dwell_s=2.5, cap=2)
    spot = [(e["pool_index"], e["t_start"]) for e in out if e.get("spotlight")]
    assert spot == [(0, 0), (0, 15)]           # two highest importance
    assert all(e.get("spotlight") == "fullscreen" for e in out if e.get("spotlight"))


def test_auto_spotlight_respects_dwell_and_threshold(monkeypatch):
    monkeypatch.delenv("KAIZER_V4_SPOTLIGHT", raising=False)
    plan = [_entry(0, 0, 1.0, 0.99),    # too short
            _entry(1, 5, 9, 0.79)]      # below threshold
    out = it.auto_spotlight(plan, min_dwell_s=2.5)
    assert not any(e.get("spotlight") for e in out)


def test_auto_spotlight_env_off(monkeypatch):
    monkeypatch.setenv("KAIZER_V4_SPOTLIGHT", "0")
    plan = [_entry(0, 0, 4, 0.95)]
    out = it.auto_spotlight(plan)
    assert not any(e.get("spotlight") for e in out)


# ─── render-side window collection ──────────────────────────────────


@pytest.fixture
def pool(tmp_path):
    d = tmp_path / "_pool"
    d.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (d / name).write_bytes(b"\xff\xd8\xff" + b"0" * 64)
    return d


def test_spotlight_windows_filter_clamp_cap(pool):
    imgs = [
        {"src": "a.jpg", "t_start": 2.0, "t_end": 6.0, "spotlight": "fullscreen"},
        {"src": "b.jpg", "t_start": 0.0, "t_end": 4.0},                      # not marked
        {"src": "c.jpg", "t_start": 8.0, "t_end": 14.0, "spotlight": "fullscreen"},  # clamped
        {"src": "missing.jpg", "t_start": 0, "t_end": 3, "spotlight": "fullscreen"},
        {"src": "a.jpg", "t_start": 1.0, "t_end": 3.0, "spotlight": "pip"},  # other kind
    ]
    wins = vb._spotlight_windows(imgs, pool, 10.0)
    assert [(w[1], w[2]) for w in wins] == [(2.0, 6.0), (8.0, 10.0)]
    assert wins[0][0].endswith("a.jpg")
    # cap
    many = [{"src": "a.jpg", "t_start": i * 3.0, "t_end": i * 3.0 + 2.9,
             "spotlight": "fullscreen"} for i in range(4)]
    assert len(vb._spotlight_windows(many, pool, 20.0, cap=2)) == 2


# ─── graph shape ────────────────────────────────────────────────────


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


def test_spotlight_graph_order_and_fades():
    fcs, out_label, _ = vb._build_story_filter_graph(
        **_minimal(),
        gap_windows=((0.0, 2.0),),
        spotlight=((3, 4.0, 8.0),),
    )
    # order: gap stage → spotlight stage → lower-third
    assert fcs.index("[stage_gap]") < fcs.index("[stage_sp0]") < fcs.index("[stage_lt]")
    assert "[3:v]scale=1920:1080" in fcs
    # V4_SPOTLIGHT_FADE_S=0.0 (ghost fix): clean opaque hard cut — no
    # alpha fade against a different base layout, plain yuv420p.
    assert "fade=t=in" not in fcs and "fade=t=out" not in fcs
    assert "setsar=1,format=yuv420p[sp0]" in fcs
    assert "enable='between(t,4.000,8.000)'" in fcs
    assert out_label == "stage_lt"


def test_no_spotlight_keeps_legacy_graph():
    base = vb._build_story_filter_graph(**_minimal(), gap_windows=())
    with_default = vb._build_story_filter_graph(**_minimal(), gap_windows=(),
                                                spotlight=(), pip=())
    assert base == with_default


# ─── Unit 8: picture-in-picture ─────────────────────────────────────


def test_pip_graph_video_fullscreen_plus_inset():
    fcs, out_label, _ = vb._build_story_filter_graph(
        **_minimal(), gap_windows=(), pip=((3, 4.0, 7.0),))
    # PiP window forces the video full-screen even with no gaps…
    assert fcs.startswith("[0:v]split=2[v_tile][v_full]")
    assert "enable='between(t,4.000,7.000)'" in fcs
    # …and adds the white-padded 320x180 inset below the bug, before text.
    assert "[3:v]scale=320:180" in fcs
    assert "pad=344:204:12:12:color=white[pip0]" in fcs
    assert f"overlay=x=W-w-{vb.V4_SIDE_MARGIN}:y=140" in fcs
    assert fcs.index("[stage_pip0]") < fcs.index("[stage_lt]")
    assert out_label == "stage_lt"


def test_pip_windows_join_gap_enable():
    fcs, _, _ = vb._build_story_filter_graph(
        **_minimal(), gap_windows=((0.0, 2.0),), pip=((3, 4.0, 7.0),))
    # fullscreen enable covers BOTH the gap and the PiP window (sorted)
    assert "enable='between(t,0.000,2.000)+between(t,4.000,7.000)'" in fcs


def test_pip_window_collection(pool):
    imgs = [
        {"src": "a.jpg", "t_start": 2.0, "t_end": 6.0, "spotlight": "pip"},
        {"src": "b.jpg", "t_start": 7.0, "t_end": 9.0, "spotlight": "fullscreen"},
    ]
    pips = vb._spotlight_windows(imgs, pool, 10.0, kind="pip")
    assert [(w[1], w[2]) for w in pips] == [(2.0, 6.0)]


@pytest.mark.slow
def test_pip_renders_video_big_image_small(tmp_path):
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("ffmpeg not available")
    src = tmp_path / "story.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=c=red:s=640x360:r=30:d=10",
         "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
         "-t", "10", "-c:v", "libx264", "-preset", "ultrafast",
         "-c:a", "aac", "-shortest", str(src)],
        check=True, capture_output=True,
    )
    from PIL import Image
    side = tmp_path / "side.png"
    Image.new("RGB", (640, 800), (0, 0, 255)).save(side)
    pip_img = tmp_path / "pip.png"
    Image.new("RGB", (640, 360), (0, 200, 0)).save(pip_img)

    from pipeline_core.longform_compose import StoryMeta
    meta = StoryMeta(title="t", kicker="BREAKING", language="en",
                     story_index=0, total_stories=1)
    out = tmp_path / "composed.mp4"
    vb._compose_v4_bulletin_story(
        story_clip_path=str(src), story_meta=meta, out_path=str(out),
        sidebar_path=str(side), ticker_path="", channel_bug_path=None,
        font_path=None, sidebar_is_video=False, work_dir=str(tmp_path),
        apply_ticker=False,
        pip_windows=((str(pip_img), 4.0, 7.0),),
    )
    assert out.is_file() and out.stat().st_size > 0

    frame = tmp_path / "f.png"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "5.50",
                    "-i", str(out), "-frames:v", "1", str(frame)],
                   check=True, capture_output=True)
    with Image.open(frame) as im:
        im = im.convert("RGB")
        corner = im.crop((0, 0, 16, 16))
        # inset content box: x 1558..1878, y 152..332 (pad 12 inside 344x204
        # plate at x=1546, y=140)
        inset = im.crop((1650, 200, 1666, 216))
        c = [sum(p[i] for p in corner.getdata()) / 256 for i in range(3)]
        n = [sum(p[i] for p in inset.getdata()) / 256 for i in range(3)]
    assert c[0] > 150 and c[1] < 90          # full-screen red video
    assert n[1] > 120 and n[0] < 90          # green inset


# ─── ffmpeg smoke ───────────────────────────────────────────────────


@pytest.mark.slow
def test_spotlight_renders_fullscreen_image(tmp_path):
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("ffmpeg not available")
    src = tmp_path / "story.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=c=red:s=640x360:r=30:d=10",
         "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
         "-t", "10", "-c:v", "libx264", "-preset", "ultrafast",
         "-c:a", "aac", "-shortest", str(src)],
        check=True, capture_output=True,
    )
    from PIL import Image
    side = tmp_path / "side.png"
    Image.new("RGB", (640, 800), (0, 0, 255)).save(side)
    spot = tmp_path / "spot.png"
    Image.new("RGB", (1280, 720), (0, 200, 0)).save(spot)

    from pipeline_core.longform_compose import StoryMeta
    meta = StoryMeta(title="t", kicker="BREAKING", language="en",
                     story_index=0, total_stories=1)
    out = tmp_path / "composed.mp4"
    vb._compose_v4_bulletin_story(
        story_clip_path=str(src), story_meta=meta, out_path=str(out),
        sidebar_path=str(side), ticker_path="", channel_bug_path=None,
        font_path=None, sidebar_is_video=False, work_dir=str(tmp_path),
        apply_ticker=False,
        spotlight_windows=((str(spot), 4.0, 7.0),),
    )
    assert out.is_file() and out.stat().st_size > 0

    def px(t):
        f = tmp_path / f"f{t:.0f}.png"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}",
                        "-i", str(out), "-frames:v", "1", str(f)],
                       check=True, capture_output=True)
        with Image.open(f) as im:
            im = im.convert("RGB").crop((0, 0, 16, 16))
            data = list(im.getdata())
        return tuple(sum(c[i] for c in data) / len(data) for i in range(3))

    normal = px(1.0)     # framed layout — near-black canvas corner
    lit = px(5.5)        # spotlight — green fills the frame
    assert sum(normal) < 120
    assert lit[1] > 120 and lit[0] < 90 and lit[2] < 90
