"""Face-aware image framing — pipeline_v4.face_focus + the focal cover-crop.

Contract under test:
  1. face_focus fail-soft guarantees: env kill switch (no cv2 touched),
     unreadable/missing file, blank face-less image → (50.0, 50.0), never
     an exception; results cached per (path, mtime).
  2. Focal arithmetic + the 15..85 clamp (pure _group_focal math AND a
     monkeypatched detector through focal_point — a synthetic "real"
     detection is too flaky for CI).
  3. The cv2 Haar cascades actually load in this venv.
  4. Graph emission: focal crop coords appear ONLY for non-50 offsets.
     Default offsets / legacy 3-tuples must emit the bare legacy
     ``crop=W:H`` — CHARACTER-identical to the goldens pinned by
     tests/test_layout_live.py and tests/test_carousel_gaps.py.
  5. Static sidebar: 50/50 keeps the PNG BYTE-identical ((n)//2 center)
     and the legacy ``.sig`` string; non-default appends ``|fx…``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from pipeline_v4 import face_focus as ff
from pipeline_v4 import v1_bridge as vb


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    """Isolate the per-file result cache between tests."""
    monkeypatch.setattr(ff, "_cache", {})
    yield


# ─── 1) fail-soft guarantees ─────────────────────────────────────────


def test_env_off_returns_default_without_cv2(monkeypatch, tmp_path):
    monkeypatch.setenv("KAIZER_V4_FACE_FOCUS", "0")
    monkeypatch.setattr(ff, "_cascades", None)
    img = tmp_path / "x.jpg"
    img.write_bytes(b"\xff\xd8\xff" + b"0" * 64)
    assert ff.focal_point(img) == (50.0, 50.0)
    # kill switch short-circuits BEFORE any cv2 work — cascades untouched
    assert ff._cascades is None


def test_missing_file_returns_default():
    assert ff.focal_point(Path("Z:/definitely/missing.jpg")) == (50.0, 50.0)


def test_unreadable_garbage_returns_default_and_caches(tmp_path, monkeypatch):
    monkeypatch.delenv("KAIZER_V4_FACE_FOCUS", raising=False)
    p = tmp_path / "garbage.jpg"
    p.write_bytes(b"not an image at all")
    assert ff.focal_point(p) == (50.0, 50.0)
    key = (str(p), p.stat().st_mtime)
    assert ff._cache.get(key) == (50.0, 50.0)   # negative result cached too


def test_blank_image_returns_default(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from PIL import Image
    monkeypatch.delenv("KAIZER_V4_FACE_FOCUS", raising=False)
    p = tmp_path / "blank.png"
    Image.new("RGB", (400, 300), (128, 128, 128)).save(p)
    assert ff.focal_point(p) == (50.0, 50.0)
    key = (str(p), p.stat().st_mtime)
    assert key in ff._cache


def test_cascades_load_in_this_venv():
    pytest.importorskip("cv2")
    cas = ff._load_cascades()
    assert cas and len(cas) == 2      # frontal + profile both present


# ─── 2) focal arithmetic + clamp ─────────────────────────────────────


def test_group_focal_single_face_center():
    # face box (80,80,40,40) in 400x300 → center (100,100) → 25% / 33.3%
    assert ff._group_focal([(80, 80, 40, 40)], 400, 300) == (25.0, 33.3)


def test_group_focal_clamps_to_15_85():
    assert ff._group_focal([(0, 0, 20, 20)], 400, 300) == (15.0, 15.0)
    assert ff._group_focal([(370, 270, 30, 30)], 400, 300) == (85.0, 85.0)


def test_group_focal_group_shot_lands_between_heads():
    ox, oy = ff._group_focal([(40, 90, 20, 20), (140, 90, 20, 20)], 200, 200)
    assert (ox, oy) == (50.0, 50.0)


def test_group_focal_excludes_tiny_background_faces():
    # 20x20 face is 4% of the 100x100 subject's area — below the 60%
    # group threshold, so it must not drag the focal point right.
    ox, _oy = ff._group_focal([(0, 0, 100, 100), (300, 0, 20, 20)], 400, 300)
    assert ox == 15.0                 # subject center 12.5% → clamped


def test_group_focal_degenerate_inputs_default():
    assert ff._group_focal([], 400, 300) == (50.0, 50.0)
    assert ff._group_focal([(0, 0, 0, 0)], 400, 300) == (50.0, 50.0)
    assert ff._group_focal([(0, 0, 10, 10)], 0, 0) == (50.0, 50.0)


def test_focal_point_monkeypatched_detector_clamps(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from PIL import Image
    monkeypatch.delenv("KAIZER_V4_FACE_FOCUS", raising=False)
    p = tmp_path / "face.png"
    Image.new("RGB", (200, 100), (10, 10, 10)).save(p)
    # deterministic "detection" at the extreme top-left → clamp to 15/15
    monkeypatch.setattr(ff, "_detect", lambda gray: [(0, 0, 10, 10)])
    assert ff.focal_point(p) == (15.0, 15.0)


# ─── 3) _focal_crop helper — default emits the bare legacy string ───


def test_focal_crop_default_is_bare_legacy():
    assert vb._focal_crop(564, 794) == "crop=564:794"
    assert vb._focal_crop(564, 794, 50.0, 50.0) == "crop=564:794"
    assert vb._focal_crop(564, 794, "50", "50.0") == "crop=564:794"


def test_focal_crop_non_default_and_clamp_and_failsoft():
    assert vb._focal_crop(564, 794, 0.0, 50.0) == \
        "crop=564:794:(in_w-564)*0.000:(in_h-794)*0.500"
    # out-of-range values clamp to the valid 0..100 window
    assert vb._focal_crop(100, 100, 150, -3) == \
        "crop=100:100:(in_w-100)*1.000:(in_h-100)*0.000"
    # garbage degrades to the bare legacy string, never raises
    assert vb._focal_crop(10, 10, "junk", None) == "crop=10:10"


# ─── 4) graph emission — spotlight / PiP / carousel chain ───────────


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


def test_spotlight_default_offsets_keep_graph_character_identical():
    legacy = vb._build_story_filter_graph(
        **_minimal(), spotlight=((3, 4.0, 8.0),))
    with_default = vb._build_story_filter_graph(
        **_minimal(), spotlight=((3, 4.0, 8.0, 50.0, 50.0),))
    assert legacy == with_default
    assert "crop=1920:1080,setsar=1,format=yuv420p[sp0]" in legacy[0]


def test_spotlight_focal_crop_emitted_only_for_non_default():
    fcs, _, _ = vb._build_story_filter_graph(
        **_minimal(), spotlight=((3, 4.0, 8.0, 30.0, 15.0),))
    assert "crop=1920:1080:(in_w-1920)*0.300:(in_h-1080)*0.150" in fcs
    assert "crop=1920:1080,setsar=1,format=yuv420p[sp0]" not in fcs


def test_pip_default_offsets_keep_graph_character_identical():
    legacy = vb._build_story_filter_graph(**_minimal(), pip=((3, 4.0, 7.0),))
    with_default = vb._build_story_filter_graph(
        **_minimal(), pip=((3, 4.0, 7.0, 50.0, 50.0),))
    assert legacy == with_default
    assert "crop=320:180,setsar=1" in legacy[0]


def test_pip_focal_crop_emitted_for_non_default():
    fcs, _, _ = vb._build_story_filter_graph(
        **_minimal(), pip=((3, 4.0, 7.0, 100.0, 0.0),))
    assert "crop=320:180:(in_w-320)*1.000:(in_h-180)*0.000" in fcs
    # the PiP window still joins the full-screen enable exactly as before
    assert "enable='between(t,4.000,7.000)'" in fcs


def _seg(path, ts, te, effect, ox, oy):
    return {"path": path, "ts": ts, "te": te, "ed": 0.4,
            "fade": effect == "fade", "fp": "1:1",
            "src": os.path.basename(path), "effect": effect,
            "fit": "cover", "ox": ox, "oy": oy}


def _capture_carousel_fc(monkeypatch, segs, tmp_path):
    """Run _render_sidebar_carousel with subprocess stubbed out and return
    the -filter_complex string it would have executed."""
    captured = {}

    class _R:
        returncode = 1
        stderr = "stubbed"

    def _fake_run(cmd, **_kw):
        captured["cmd"] = cmd
        return _R()

    monkeypatch.setattr(vb.subprocess, "run", _fake_run)
    vb._render_sidebar_carousel(segs=segs, out_path=str(tmp_path / "o.mp4"),
                                story_duration=9.0)
    cmd = captured["cmd"]
    return cmd[cmd.index("-filter_complex") + 1]


def test_sidebar_carousel_chain_honors_focal(tmp_path, monkeypatch):
    segs = [_seg(str(tmp_path / "a.jpg"), 0.0, 4.0, "fade", 0.0, 50.0),
            _seg(str(tmp_path / "b.jpg"), 4.0, 9.0, "zoom_in", 100.0, 0.0)]
    fc = _capture_carousel_fc(monkeypatch, segs, tmp_path)
    # fade seg (panel 564x794): x pinned to the source's left edge
    assert "crop=564:794:(in_w-564)*0.000:(in_h-794)*0.500" in fc
    # zoom seg crops at 2x with its own focal point before zoompan
    assert "crop=1128:1588:(in_w-1128)*1.000:(in_h-1588)*0.000" in fc


def test_sidebar_carousel_default_offsets_keep_legacy_chain(tmp_path, monkeypatch):
    segs = [_seg(str(tmp_path / "a.jpg"), 0.0, 4.0, "fade", 50.0, 50.0),
            _seg(str(tmp_path / "b.jpg"), 4.0, 9.0, "zoom_in", 50.0, 50.0)]
    fc = _capture_carousel_fc(monkeypatch, segs, tmp_path)
    assert "crop=564:794,setsar=1" in fc          # bare legacy crop
    assert "crop=1128:1588,zoompan" in fc
    assert "crop=564:794:(" not in fc and "crop=1128:1588:(" not in fc


# ─── 5) window plumbing — offsets survive collection + edge clamp ────


def test_spotlight_windows_carry_offsets(tmp_path):
    pool = tmp_path / "_pool"
    pool.mkdir()
    (pool / "a.jpg").write_bytes(b"\xff\xd8\xff" + b"0" * 64)
    imgs = [{"src": "a.jpg", "t_start": 1.0, "t_end": 5.0,
             "spotlight": "fullscreen",
             "offset_x_pct": 20.0, "offset_y_pct": 80.0}]
    wins = vb._spotlight_windows(imgs, pool, 10.0)
    assert wins == [(str(pool / "a.jpg"), 1.0, 5.0, 20.0, 80.0)]
    # no explicit offsets → the centered default rides the tuple
    wins2 = vb._spotlight_windows(
        [{"src": "a.jpg", "t_start": 1.0, "t_end": 5.0, "spotlight": "pip"}],
        pool, 10.0, kind="pip")
    assert wins2 == [(str(pool / "a.jpg"), 1.0, 5.0, 50.0, 50.0)]


def test_edge_clamp_preserves_focal_tail():
    wins = vb._edge_clamp_windows([("p.jpg", 0.0, 5.0, 30.0, 15.0)], 10.0,
                                  with_path=True)
    assert wins == [("p.jpg", 0.75, 5.0, 30.0, 15.0)]
    # legacy 3-tuples still clamp to the identical shape
    wins3 = vb._edge_clamp_windows([("p.jpg", 0.0, 5.0)], 10.0, with_path=True)
    assert wins3 == [("p.jpg", 0.75, 5.0)]


# ─── 6) static sidebar — byte identity + conditional .sig ────────────


def _gradient_src(tmp_path):
    """Horizontal-gradient source: a wrong crop x is guaranteed to change
    pixel bytes (a flat color would hide the bug). Odd dims on purpose —
    (n)//2 vs round(n*0.5) differ there, which is exactly the byte-compat
    trap the 50/50 branch guards."""
    from PIL import Image
    g = Image.linear_gradient("L").transpose(Image.TRANSPOSE).resize((1601, 901))
    src = tmp_path / "src.png"
    Image.merge("RGB", (g, g, g)).save(src)
    return src


def test_make_sidebar_placeholder_default_bytes_identical(tmp_path):
    from pipeline_core.longform_compose import make_sidebar_placeholder
    src = _gradient_src(tmp_path)
    a, b, c = (tmp_path / n for n in ("a.png", "b.png", "c.png"))
    make_sidebar_placeholder(str(src), str(a))
    make_sidebar_placeholder(str(src), str(b),
                             offset_x_pct=50.0, offset_y_pct=50.0)
    assert a.read_bytes() == b.read_bytes()       # explicit 50/50 == legacy
    make_sidebar_placeholder(str(src), str(c),
                             offset_x_pct=0.0, offset_y_pct=0.0)
    assert c.read_bytes() != a.read_bytes()       # reframe actually moves


def test_resolve_sidebar_sig_conditional(tmp_path):
    from PIL import Image
    src = tmp_path / "p.jpg"
    Image.new("RGB", (800, 600), (0, 80, 160)).save(src)
    out = vb._resolve_sidebar(work_dir=tmp_path, story_index=0,
                              pool_image_path=str(src))
    st = os.stat(src)
    sig = Path(out + ".sig").read_text(encoding="utf-8")
    # default sig string is EXACTLY the legacy one — no cold regen fleet-wide
    assert sig == f"{src}|{st.st_size}|{int(st.st_mtime)}"
    out2 = vb._resolve_sidebar(work_dir=tmp_path, story_index=1,
                               pool_image_path=str(src),
                               offset_x_pct=20.0, offset_y_pct=80.0)
    sig2 = Path(out2 + ".sig").read_text(encoding="utf-8")
    assert sig2 == f"{src}|{st.st_size}|{int(st.st_mtime)}|fx20.0,80.0"
