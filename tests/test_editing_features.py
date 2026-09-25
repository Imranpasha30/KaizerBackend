"""UF.1 — Admin Editing Features catalog + real-clip dummy previews."""
from __future__ import annotations

import subprocess

import pytest
from PIL import Image

from routers import editing_features as ef


def _has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


@pytest.fixture()
def source_clip(tmp_path):
    """A synthesized stand-in for the operator's real footage."""
    clip = tmp_path / "real_clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc2=s=640x360:d=8:r=25",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         str(clip)], check=True, capture_output=True, timeout=120)
    return clip


@pytest.fixture()
def preview_env(tmp_path, source_clip, monkeypatch):
    """Point the preview engine at an isolated cache + the fixture clip."""
    monkeypatch.setenv("KAIZER_OUTPUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("KAIZER_FEATURE_PREVIEW_CLIP", str(source_clip))
    return tmp_path


@pytest.fixture()
def clipless_env(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIZER_OUTPUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("KAIZER_FEATURE_PREVIEW_CLIP", "")
    monkeypatch.setattr(ef, "_CLIP_ROOTS", ())
    return tmp_path


def test_catalog_covers_every_registry():
    cat = ef.catalog(None)  # route fn is directly callable; auth arg unused
    sections = {s["key"]: s for s in cat["sections"]}
    # every engine registry must be represented, with real counts
    assert len(sections["style_packs"]["items"]) >= 30
    assert len(sections["transitions"]["items"]) >= 50
    assert len(sections["color_grades"]["items"]) >= 20
    assert len(sections["frame_fx"]["items"]) >= 14
    assert len(sections["overlays"]["items"]) >= 18
    assert len(sections["typography"]["items"]) >= 3
    assert len(sections["sound"]["items"]) >= 8
    assert len(sections["layouts"]["items"]) >= 12
    assert len(sections["taxonomy"]["items"]) >= 80
    for s in cat["sections"]:
        for it in s["items"]:
            assert it["id"] and it["label"] and "used_for" in it
    assert cat["totals"]["transitions"] == len(sections["transitions"]["items"])


def test_layout_png_all_keys(tmp_path):
    keys = ["bulletin", "audio_fullscreen", "torn_card", "clean_card",
            "split_frame", "follow_bar", "dual_video", "grid_2", "grid_3",
            "grid_4", "pip_corner", "spotlight", "trailer", "custom_html"]
    for k in keys:
        out = tmp_path / f"{k}.png"
        assert ef._layout_png(k, out), k
        assert out.is_file() and out.stat().st_size > 0, k
    assert not ef._layout_png("no_such_layout", tmp_path / "x.png")


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")
def test_base_extraction_from_real_clip(preview_env):
    base = ef._ensure_base()
    assert base is not None
    for k in ("base", "a", "b", "frame"):
        assert base[k].is_file() and base[k].stat().st_size > 0, k
    img = Image.open(base["frame"])
    assert img.size == (ef._FR_W, ef._FR_H)
    # cached second call returns the same fingerprint
    assert ef._ensure_base()["fp"] == base["fp"]


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")
def test_grade_fx_pack_previews_are_real_clip_video(preview_env):
    from pipeline_v4.color_grades import get_grade_vf
    from pipeline_v4.frame_fx import get_fx_vf
    r = ef._chain_preview(get_grade_vf("cinematic_teal_orange"), "tealo", "grade")
    assert r["media"] == "video" and r["url"].endswith(".mp4")
    r2 = ef._chain_preview(get_fx_vf("mirror"), "mirror", "fx")  # graph chain
    assert r2["media"] == "video"


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")
def test_clipless_fallback_still_previews(clipless_env):
    from pipeline_v4.color_grades import get_grade_vf
    assert ef._ensure_base() is None
    r = ef._chain_preview(get_grade_vf("noir"), "noir", "grade")
    assert r["media"] == "image" and r["url"].endswith(".png")


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")
def test_style_pack_demo_is_mini_trailer_with_audio(preview_env):
    """A pack preview must demo the WHOLE identity (look+cut+sound+card),
    not just the color: video AND audio streams, full demo length."""
    import json
    from pipeline_v4.trailer_styles import STYLES
    base = ef._ensure_base()
    out = ef._cache_dir() / "packdemo_test.mp4"
    assert ef._pack_demo_mp4(STYLES["news"], out, base)
    pr = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type:format=duration", "-of", "json", str(out)],
        capture_output=True, text=True, timeout=30)
    j = json.loads(pr.stdout)
    kinds = sorted(s["codec_type"] for s in j["streams"])
    assert kinds == ["audio", "video"]
    assert float(j["format"]["duration"]) > 3.0  # card not chopped


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")
def test_transition_preview_on_real_clip(preview_env):
    base = ef._ensure_base()
    out = ef._cache_dir() / "t.mp4"
    assert ef._transition_mp4("circleopen", out, base)
    assert out.stat().st_size > 0


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")
def test_overlay_composited_onto_real_frame(preview_env):
    from pipeline_v4.overlays import render_registry_item
    base = ef._ensure_base()
    el = ef._cache_dir() / "el.png"
    assert render_registry_item("breaking_news_banner", str(el))
    out = ef._cache_dir() / "composited.png"
    assert ef._composite_on_frame(el, "banner", out, base)
    img = Image.open(out)
    assert img.size == (ef._FR_W, ef._FR_H)  # in-context, frame-sized


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")
def test_sound_preview(preview_env, tmp_path):
    p = ef._sound_wav("hit@horror", tmp_path)
    assert p
    b = ef._sound_wav("bleep", tmp_path)
    assert b and b.endswith(".wav")


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")
def test_typography_preview_on_real_frame(preview_env):
    cache = ef._cache_dir()
    out = cache / "h.png"
    assert ef._typography_png("headline_crash", cache, out)
    img = Image.open(out)
    assert img.size == (ef._FR_W, ef._FR_H)
    assert not ef._typography_png("nope", cache, cache / "n.png")
