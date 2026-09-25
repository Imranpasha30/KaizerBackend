"""Panel face-safety + carousel base plate (operator-reported, job 598:
an OTS picture box rendered as a BLACK plate over the co-host's face).

Offline: ffmpeg/cv2 paths are stubbed. Covers: rect mirroring, the
base-plate ffmpeg chain (and its absence keeping the legacy graph),
conditional .sig fingerprinting, coverage-skip, and the face-safety
decision matrix — flip/veto on floating (OTS) panels, strip-only on
framed layouts riding above gap-fullscreen video, no-ops, kill switch.
"""
import dataclasses
import json
import os
from pathlib import Path

import pytest

from pipeline_v4 import v1_bridge as vb
from pipeline_v4.layout_library import story_geometry


def _seg(path, ts, te, **kw):
    d = {"path": path, "ts": ts, "te": te, "ed": 0.4, "fade": True,
         "fp": "1:1", "src": os.path.basename(path), "effect": "fade",
         "fit": "cover", "ox": 50.0, "oy": 50.0}
    d.update(kw)
    return d


def _img_file(tmp_path, name="a.jpg"):
    p = tmp_path / name
    p.write_bytes(b"x")
    return str(p)


# ── mirroring ──────────────────────────────────────────────────────

def test_mirror_rect():
    x, y, w, h = vb._mirror_rect_pct((70.888, 16.578, 24.698, 58.863))
    assert (round(x, 3), y, w, h) == (4.414, 16.578, 24.698, 58.863)
    back = vb._mirror_rect_pct((x, y, w, h))     # involution
    assert round(back[0], 3) == 70.888


# ── base plate in the carousel graph ───────────────────────────────

def _run_carousel(tmp_path, monkeypatch, **kw):
    captured = {}

    def fake_run(cmd, **_kw):
        captured["cmd"] = cmd

        class R:
            returncode = 0
            stderr = ""
            stdout = ""
        out = Path(kw.get("out_path") or (tmp_path / "o.mp4"))
        out.write_bytes(b"v")
        return R()

    monkeypatch.setattr(vb, "_enc_args", lambda **_k: ["-c:v", "libx264"])
    monkeypatch.setattr(vb.subprocess, "run", fake_run)
    vb._render_sidebar_carousel(**kw)
    return " ".join(str(c) for c in captured.get("cmd", []))


def test_base_plate_added_to_chain(tmp_path, monkeypatch):
    img = _img_file(tmp_path)
    base = _img_file(tmp_path, "base.jpg")
    cmd = _run_carousel(
        tmp_path, monkeypatch,
        segs=[_seg(img, 2.4, 8.0)], out_path=str(tmp_path / "o.mp4"),
        story_duration=45.0, base_image=base)
    assert "[imbase]" in cmd
    assert base in cmd
    # base composes BEFORE the first timed window
    assert cmd.index("[imbase]") < cmd.index("between(t,2.400,8.000)")


def test_no_base_keeps_legacy_chain(tmp_path, monkeypatch):
    img = _img_file(tmp_path)
    cmd = _run_carousel(
        tmp_path, monkeypatch,
        segs=[_seg(img, 2.4, 8.0)], out_path=str(tmp_path / "o.mp4"),
        story_duration=45.0)
    assert "[imbase]" not in cmd and "[vbase]" not in cmd


# ── .sig discipline ────────────────────────────────────────────────

def test_sig_gains_pbase_only_with_base(tmp_path, monkeypatch):
    img = _img_file(tmp_path)
    base = _img_file(tmp_path, "base.jpg")
    monkeypatch.setattr(vb, "_render_sidebar_carousel",
                        lambda **kw: (Path(kw["out_path"]).write_bytes(b"v")
                                      or kw["out_path"]))
    images = [{"src": os.path.basename(img), "t_start": 2.4, "t_end": 8.0}]
    out1 = str(tmp_path / "c1.mp4")
    vb._ensure_sidebar_carousel(images=images, pool_dir=tmp_path,
                                out_path=out1, story_duration=45.0)
    sig1 = Path(out1 + ".sig").read_text(encoding="utf-8")
    assert "pbase" not in sig1
    out2 = str(tmp_path / "c2.mp4")
    vb._ensure_sidebar_carousel(images=images, pool_dir=tmp_path,
                                out_path=out2, story_duration=45.0,
                                base_image=base, base_ox=30.0, base_oy=40.0)
    sig2 = json.loads(Path(out2 + ".sig").read_text(encoding="utf-8"))
    assert sig2["pbase"][0] == "base.jpg"
    assert sig2["pbase"][2:] == [30.0, 40.0]


def test_base_dropped_when_windows_cover_story(tmp_path, monkeypatch):
    # Two DISTINCT images tile the whole story (0-20, 20-45). Distinct
    # sources stay separate segments (contiguous SAME-image windows would
    # coalesce — see test_base_dropped_when_coalesced_cover). The windows
    # leave no gap, so the base plate is pointless and must be dropped — it
    # could otherwise show as a black box behind a face (job 598).
    img_a = _img_file(tmp_path, "a.jpg")
    img_b = _img_file(tmp_path, "b.jpg")
    base = _img_file(tmp_path, "base.jpg")
    seen = {}

    def fake_render(**kw):
        seen.update(kw)
        Path(kw["out_path"]).write_bytes(b"v")
        return kw["out_path"]

    monkeypatch.setattr(vb, "_render_sidebar_carousel", fake_render)
    images = [{"src": os.path.basename(img_a), "t_start": 0.0, "t_end": 20.0},
              {"src": os.path.basename(img_b), "t_start": 20.0, "t_end": 45.0}]
    out = str(tmp_path / "c3.mp4")
    vb._ensure_sidebar_carousel(images=images, pool_dir=tmp_path,
                                out_path=out, story_duration=45.0,
                                base_image=base)
    assert seen["base_image"] is None
    assert "pbase" not in Path(out + ".sig").read_text(encoding="utf-8")


def test_base_dropped_when_coalesced_cover(tmp_path, monkeypatch):
    # Contiguous SAME-image windows covering the story coalesce (ca89d3f)
    # into ONE full-story segment → the cheaper static sidebar is used
    # (return None), no carousel render at all. Also panel-safe: a single
    # full-panel image leaves no gap, so no base plate / black box is ever
    # composited behind a face.
    img = _img_file(tmp_path)
    base = _img_file(tmp_path, "base.jpg")
    called = {"n": 0}

    def fake_render(**kw):
        called["n"] += 1
        Path(kw["out_path"]).write_bytes(b"v")
        return kw["out_path"]

    monkeypatch.setattr(vb, "_render_sidebar_carousel", fake_render)
    images = [{"src": os.path.basename(img), "t_start": 0.0, "t_end": 20.0},
              {"src": os.path.basename(img), "t_start": 20.0, "t_end": 45.0}]
    out = str(tmp_path / "c3b.mp4")
    res = vb._ensure_sidebar_carousel(images=images, pool_dir=tmp_path,
                                      out_path=out, story_duration=45.0,
                                      base_image=base)
    assert res is None            # coalesced to one full-story image → static sidebar
    assert called["n"] == 0       # no carousel render, so no base plate either


# ── face-safety decision matrix ────────────────────────────────────

FRAMED = story_geometry("news_ots_right")   # framed: video tile + side panel
pytestmark_geom = pytest.mark.skipif(FRAMED is None, reason="catalog changed")
FLOATING = (dataclasses.replace(FRAMED, video=(0.0, 0.0, 100.0, 100.0))
            if FRAMED is not None else None)
GAPS_HEAVY = ((0.0, 2.4), (8.08, 9.92), (15.77, 45.0))   # job 598's shape


def _panel_rect(g):
    return g.picture or (g.pips[0].x, g.pips[0].y, g.pips[0].w, g.pips[0].h)


def _src(tmp_path):
    p = tmp_path / "src.mp4"
    p.write_bytes(b"v")
    return str(p)


def _face_on(rect):
    return (rect[0] + rect[2] * 0.3, rect[1] + rect[3] * 0.3, 12.0, 20.0)


@pytestmark_geom
def test_floating_flips_to_clean_side(tmp_path, monkeypatch):
    rect = _panel_rect(FLOATING)
    monkeypatch.setattr(vb, "_face_boxes_for_probe",
                        lambda *a, **k: [_face_on(rect)])
    g2, action = vb._panel_face_safety(
        FLOATING, src_path=_src(tmp_path), t0=0.0, t1=45.0,
        work_dir=str(tmp_path), story_index=0)
    assert action == "flip"
    assert round(_panel_rect(g2)[0], 3) == round(100.0 - rect[0] - rect[2], 3)


@pytestmark_geom
def test_floating_vetoes_two_shot(tmp_path, monkeypatch):
    rect = _panel_rect(FLOATING)
    mir = vb._mirror_rect_pct(rect)
    monkeypatch.setattr(vb, "_face_boxes_for_probe",
                        lambda *a, **k: [_face_on(rect), _face_on(mir)])
    g2, action = vb._panel_face_safety(
        FLOATING, src_path=_src(tmp_path), t0=0.0, t1=45.0,
        work_dir=str(tmp_path), story_index=0)
    assert action == "veto"
    assert g2.picture is None and g2.pips == ()


@pytestmark_geom
def test_gap_heavy_framed_strips_panel(tmp_path, monkeypatch):
    """Job 598's exact shape: framed layout, every image a timed
    spotlight → gaps dominate → the panel rides above fullscreen video
    ON the co-host — strip it (never mirror: the framed moments would
    collide with the video tile)."""
    rect = _panel_rect(FRAMED)
    monkeypatch.setattr(vb, "_face_boxes_for_probe",
                        lambda *a, **k: [_face_on(rect)])
    g2, action = vb._panel_face_safety(
        FRAMED, src_path=_src(tmp_path), t0=0.0, t1=45.0,
        work_dir=str(tmp_path), story_index=0, gaps=GAPS_HEAVY)
    assert action == "veto"
    assert g2.picture is None and g2.pips == ()


@pytestmark_geom
def test_framed_without_gaps_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(vb, "_face_boxes_for_probe",
                        lambda *a, **k: [_face_on(_panel_rect(FRAMED))])
    g2, action = vb._panel_face_safety(
        FRAMED, src_path=_src(tmp_path), t0=0.0, t1=45.0,
        work_dir=str(tmp_path), story_index=0)
    assert action == "" and g2 is FRAMED


@pytestmark_geom
def test_clean_frames_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(vb, "_face_boxes_for_probe", lambda *a, **k: [])
    g2, action = vb._panel_face_safety(
        FLOATING, src_path=_src(tmp_path), t0=0.0, t1=45.0,
        work_dir=str(tmp_path), story_index=0)
    assert action == "" and g2 is FLOATING


@pytestmark_geom
def test_unreadable_frames_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(vb, "_face_boxes_for_probe", lambda *a, **k: None)
    g2, action = vb._panel_face_safety(
        FLOATING, src_path=_src(tmp_path), t0=0.0, t1=45.0,
        work_dir=str(tmp_path), story_index=0)
    assert action == "" and g2 is FLOATING


@pytestmark_geom
def test_kill_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIZER_V4_PANEL_SAFETY", "0")
    monkeypatch.setattr(vb, "_face_boxes_for_probe",
                        lambda *a, **k: [_face_on(_panel_rect(FLOATING))])
    g2, action = vb._panel_face_safety(
        FLOATING, src_path=_src(tmp_path), t0=0.0, t1=45.0,
        work_dir=str(tmp_path), story_index=0)
    assert action == "" and g2 is FLOATING


def test_face_boxes_pct_fail_soft(tmp_path, monkeypatch):
    from pipeline_v4 import face_focus
    monkeypatch.setenv("KAIZER_V4_FACE_FOCUS", "0")
    assert face_focus.face_boxes_pct(str(tmp_path / "nope.jpg")) == []
    monkeypatch.setenv("KAIZER_V4_FACE_FOCUS", "1")
    assert face_focus.face_boxes_pct(str(tmp_path / "missing.jpg")) == []
