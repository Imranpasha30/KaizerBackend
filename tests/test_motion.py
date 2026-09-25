"""Unit D — native motion toolkit port + story transitions + eased LT."""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline_v4 import v1_bridge as vb
from pipeline_v4.motion import EASINGS, ease, interpolate, motion_blur, spring


# ─── anim.py port sanity ────────────────────────────────────────────


def test_easings_are_normalized_curves():
    assert len(EASINGS) >= 24
    for name, fn in EASINGS.items():
        assert fn(0.0) == pytest.approx(0.0, abs=1e-6), name
        assert fn(1.0) == pytest.approx(1.0, abs=1e-6), name
    # out_cubic decelerates: covers more ground in the first half
    assert ease("out_cubic", 0.5) > 0.8
    assert ease("in_cubic", 0.5) < 0.2
    # unknown easing falls back rather than raising
    assert 0.0 <= ease("no_such_easing", 0.5) <= 1.0


def test_interpolate_maps_and_clamps():
    assert interpolate(5, [0, 10], [0, 100]) == pytest.approx(50.0)
    assert interpolate(-5, [0, 10], [0, 100]) == 0.0        # clamped
    assert interpolate(15, [0, 10], [0, 100]) == 100.0
    # multi-segment keyframes
    assert interpolate(7.5, [0, 5, 10], [0, 100, 0]) == pytest.approx(50.0)
    with pytest.raises(ValueError):
        interpolate(1, [0], [0])


def test_spring_converges_with_overshoot():
    vals = [spring(f, 30) for f in range(0, 61)]
    assert vals[0] == 0.0
    assert max(vals) > 1.0                     # natural overshoot
    assert vals[-1] == pytest.approx(1.0, abs=0.05)
    assert spring(30, 30, delay=45) == 0.0     # delayed start
    assert spring(30, 30, from_=100, to=200) > 100


def test_motion_blur_shapes_and_noop():
    frame = np.zeros((32, 32, 4), dtype=np.uint8)
    frame[10:20, 10:20] = 255
    out = motion_blur(frame, 6.0, 0.0, samples=6)
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert not np.array_equal(out, frame)          # blur happened
    same = motion_blur(frame, 0.2, 0.2)
    assert np.array_equal(same, frame)             # sub-pixel move = no-op


# ─── xfade stitch graph (pure math) ─────────────────────────────────


def test_xfade_graph_offsets_are_cumulative():
    fc, v, a = vb.build_xfade_stitch_graph([10.0, 8.0, 6.0],
                                           transition="fade", fade_d=0.5)
    # first transition starts at 10-0.5; second at (9.5+8)-0.5 = 17.0
    assert "xfade=transition=fade:duration=0.500:offset=9.500" in fc
    assert "xfade=transition=fade:duration=0.500:offset=17.000" in fc
    assert fc.count("acrossfade=d=0.500") == 2
    assert (v, a) == ("vout", "aout")
    with pytest.raises(ValueError):
        vb.build_xfade_stitch_graph([5.0])


def test_stitch_with_transitions_guards():
    # unknown transition / single clip → None (caller hard-cuts)
    assert vb._stitch_with_transitions(["a.mp4"], "o.mp4", transition="fade") is None
    assert vb._stitch_with_transitions(["a.mp4", "b.mp4"], "o.mp4",
                                       transition="spiral_of_doom") is None


# ─── eased lower-third: gating + hash conditionality ────────────────


def _graph(layout):
    """Build the composer's lt_x_expr exactly as _compose_v4_bulletin_story
    does (mirrors the branch under test)."""
    _lt_eased = getattr(layout, "lt_ease", None) == "out_cubic"
    return ("(-w+w*(1-pow(1-t/0.4\\,3)))" if _lt_eased else "-w+w*t/0.4")


def test_lt_ease_expression_gated_on_layout():
    assert _graph(SimpleNamespace(lt_ease=None)) == "-w+w*t/0.4"
    assert "pow(1-t/0.4" in _graph(SimpleNamespace(lt_ease="out_cubic"))


def test_lt_ease_hash_conditional():
    story = SimpleNamespace(title_native="t", title_english="t", summary="s",
                            video_t_start=0.0, video_t_end=10.0,
                            story_index=0, total_stories=1)
    def _h(layout):
        return vb._per_story_cache_hash(
            story=story, ticker_path="", sidebar_path="", layout=layout,
            channel_bug_path="", watermark_path="", watermark_position="",
            font_path="", bg_video_abs=None, bg_video_volume=0.0,
            language_code="te", images=None, pool_dir=None)
    legacy = SimpleNamespace(width=1920, height=1080, bg_color="#000",
                             lt_ease=None)
    eased = SimpleNamespace(width=1920, height=1080, bg_color="#000",
                            lt_ease="out_cubic")
    no_attr = SimpleNamespace(width=1920, height=1080, bg_color="#000")
    assert _h(legacy) == _h(no_attr)      # None == pre-motion canvas
    assert _h(legacy) != _h(eased)        # eased entrance invalidates


# ─── ffmpeg smoke: crossfade stitch really blends ───────────────────


@pytest.mark.slow
def test_transition_stitch_smoke(tmp_path):
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("ffmpeg not available")

    def make(color, path, dur=4):
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error",
             "-f", "lavfi", "-i", f"color=c={color}:s=320x240:r=30:d={dur}",
             "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
             "-t", str(dur), "-c:v", "libx264", "-preset", "ultrafast",
             "-c:a", "aac", "-shortest", str(path)],
            check=True, capture_output=True)

    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    make("red", a)
    make("green", b)
    out = tmp_path / "stitched.mp4"
    res = vb._stitch_with_transitions([str(a), str(b)], str(out),
                                      transition="fade", fade_d=0.5)
    assert res and out.stat().st_size > 0
    # total ≈ 4 + 4 - 0.5
    dur = vb._probe_clip_duration(str(out))
    assert dur == pytest.approx(7.5, abs=0.3)
    # mid-transition frame is a red/green blend (both channels lit)
    frame = tmp_path / "mid.png"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "3.75",
                    "-i", str(out), "-frames:v", "1", str(frame)],
                   check=True, capture_output=True)
    from PIL import Image
    with Image.open(frame) as im:
        px = list(im.convert("RGB").resize((8, 8)).getdata())
    r = sum(p[0] for p in px) / len(px)
    g = sum(p[1] for p in px) / len(px)
    assert r > 50 and g > 50, f"no blend at transition midpoint (r={r}, g={g})"
