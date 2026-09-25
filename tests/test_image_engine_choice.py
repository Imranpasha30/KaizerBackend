"""Engine-choice ecosystem for AI story images (gemini | openai).

Offline: every network-touching function is monkeypatched. Covers the
routing contract end to end — render_image dispatch, the shared two-step
prompt-writer feeding BOTH engines, the OpenAI renderer's size/crop
behaviour, and image_provider's upgraded _generate_via_openai (two-step +
name-tag label, replacing the old garish direct-template path).
"""
import io
import os
from pathlib import Path

import pytest

from pipeline_v4 import image_ai, image_provider


def _png_bytes(w=64, h=64, color=(200, 30, 30)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "PNG")
    return buf.getvalue()


# ── render_image dispatch ──────────────────────────────────────────

def test_render_image_routes_to_gemini_by_default(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(image_ai, "generate_image",
                        lambda **kw: calls.append(("gemini", kw)) or str(tmp_path / "g.jpg"))
    monkeypatch.setattr(image_ai, "_generate_image_openai",
                        lambda **kw: calls.append(("openai", kw)) or str(tmp_path / "o.jpg"))
    out = image_ai.render_image(prompt="P", out_path=str(tmp_path / "x.jpg"))
    assert calls[0][0] == "gemini" and out.endswith("g.jpg")
    # unknown engine value never breaks a render — falls back to gemini
    image_ai.render_image(prompt="P", out_path=str(tmp_path / "x.jpg"), engine="banana")
    assert calls[1][0] == "gemini"


def test_render_image_routes_to_openai(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(image_ai, "generate_image",
                        lambda **kw: calls.append("gemini") or None)
    monkeypatch.setattr(image_ai, "_generate_image_openai",
                        lambda **kw: calls.append("openai") or str(tmp_path / "o.jpg"))
    out = image_ai.render_image(prompt="P", out_path=str(tmp_path / "x.jpg"),
                                engine="OpenAI ")  # case/space tolerant
    assert calls == ["openai"] and out.endswith("o.jpg")


# ── the two-step contract: SAME prompt writer, chosen renderer ─────

def test_make_image_for_story_passes_engine(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(image_ai, "write_image_prompt", lambda **kw: "REFINED-PROMPT")

    def fake_render(*, prompt, out_path, width, height, engine):
        seen.update(prompt=prompt, engine=engine)
        Path(out_path).write_bytes(b"x")
        return out_path

    monkeypatch.setattr(image_ai, "render_image", fake_render)
    saved, final_prompt = image_ai.make_image_for_story(
        title_native="వార్త", summary="S", out_path=str(tmp_path / "a.jpg"),
        engine="openai")
    assert saved and final_prompt == "REFINED-PROMPT"
    assert seen["engine"] == "openai" and seen["prompt"] == "REFINED-PROMPT"


def test_make_images_for_story_passes_engine(monkeypatch, tmp_path):
    engines = []
    monkeypatch.setattr(image_ai, "plan_story_beats", lambda **kw: [
        {"prompt": "beat one", "label": "L1"},
        {"prompt": "beat two", "label": "L2"},
    ])

    def fake_render(*, prompt, out_path, width, height, engine):
        engines.append(engine)
        Path(out_path).write_bytes(b"x")
        return out_path

    monkeypatch.setattr(image_ai, "render_image", fake_render)
    triples = image_ai.make_images_for_story(
        title_native="t", out_dir=str(tmp_path), engine="openai")
    assert engines == ["openai", "openai"]
    assert [lab for _, _, lab in triples] == ["L1", "L2"]


# ── OpenAI renderer: orientation-mapped canvas + exact cover-crop ──

def test_openai_renderer_landscape(monkeypatch, tmp_path):
    import express.ai_image as ai_image_mod
    captured = {}

    def fake_raw(*, api_key, prompt, size, quality, timeout_s):
        captured.update(size=size, quality=quality, prompt=prompt)
        return _png_bytes(160, 90)

    monkeypatch.setattr(ai_image_mod, "_generate", fake_raw)
    out = str(tmp_path / "l.jpg")
    saved = image_ai._generate_image_openai(prompt="documentary scene",
                                            out_path=out, width=640, height=360)
    assert saved == out
    assert captured["size"] == "1536x1024"
    assert "No text" in captured["prompt"]          # B-roll suffix, not BIG-TV style
    from PIL import Image
    with Image.open(out) as img:
        assert (img.width, img.height) == (640, 360)


def test_openai_renderer_portrait_size(monkeypatch, tmp_path):
    import express.ai_image as ai_image_mod
    captured = {}

    def fake_raw(*, api_key, prompt, size, quality, timeout_s):
        captured["size"] = size
        return _png_bytes(90, 160)

    monkeypatch.setattr(ai_image_mod, "_generate", fake_raw)
    saved = image_ai._generate_image_openai(prompt="p", out_path=str(tmp_path / "p.jpg"),
                                            width=360, height=640)
    assert saved and captured["size"] == "1024x1536"


def test_openai_renderer_missing_key_is_clear(monkeypatch, tmp_path):
    import express.ai_image as ai_image_mod

    def fake_raw(**kw):
        raise ai_image_mod.AIImageError("OPENAI_API_KEY missing")

    monkeypatch.setattr(ai_image_mod, "_generate", fake_raw)
    saved = image_ai._generate_image_openai(prompt="p", out_path=str(tmp_path / "x.jpg"))
    assert saved is None
    assert "OPENAI_API_KEY missing" in image_ai.generate_image.last_error


# ── image_provider: openai story path is two-step + labelled now ──

def test_generate_via_openai_two_step_with_label(monkeypatch, tmp_path):
    seen = {}

    def fake_make(**kw):
        seen.update(kw)
        p = Path(kw["out_path"])
        p.write_bytes(b"x")
        return str(p), "Cinematic 16:9 news B-roll photograph about: parliament at dusk."

    monkeypatch.setattr(image_ai, "make_image_for_story", fake_make)
    fname, label = image_provider._generate_via_openai(
        story_index=0, title_native="శాసనసభ", title_english="Assembly",
        summary="Session summary.", pool_dir=tmp_path,
        language="te", transcript_text="the assembly met in Hyderabad today")
    assert fname and fname.endswith(".jpg")
    assert seen["engine"] == "openai"
    assert "Spoken transcript:" in seen["summary"]   # transcript grounding kept
    assert label  # name-tag contract for image↔speech sync


def test_generate_via_openai_no_headline_skips(tmp_path):
    fname, label = image_provider._generate_via_openai(
        story_index=0, title_native="", title_english="",
        summary="s", pool_dir=tmp_path)
    assert fname is None and label == ""


# ── assets routes: explicit pick wins, env default, auto→gemini ───

def test_resolve_engine_precedence(monkeypatch):
    from routers.assets import _resolve_engine
    monkeypatch.delenv("KAIZER_V4_IMAGE_PROVIDER", raising=False)
    assert _resolve_engine("") == "gemini"
    assert _resolve_engine("openai") == "openai"
    assert _resolve_engine("auto") == "gemini"      # direct-generate: auto = gemini
    monkeypatch.setenv("KAIZER_V4_IMAGE_PROVIDER", "openai")
    assert _resolve_engine("") == "openai"
    assert _resolve_engine("gemini") == "gemini"    # explicit pick beats env
