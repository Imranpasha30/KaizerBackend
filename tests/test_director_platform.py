"""Dual AI Director — the ported platform engine (pipeline_v4/director_platform).

Pins: (1) the 5-rule formula ladder fires exactly as upstream; (2) the adapter
maps every platform mood onto StoryDirectives whose ids are ALWAYS inside the
V4 vocabulary; (3) neutral == the untouched category baseline; (4) user pins
constrain the adapter exactly like the V4 engine; (5) fail-soft everywhere —
LLM failures keep the formula candidate, adapter errors return {} so the
orchestrator falls back to the V4 engine. All offline (sensors monkeypatched)."""
from types import SimpleNamespace

import pytest

from pipeline_v4.director_platform import adapter as ad
from pipeline_v4.director_platform.formula import (
    FormulaCandidate, apply_formula,
)
from pipeline_v4.director_platform.llm_refine import refine_with_llm
from pipeline_v4.director_platform.sensors import SensorReadings


def _readings(**over) -> SensorReadings:
    base = dict(duration_s=30.0, audio_rms_mean=0.05, audio_rms_peak=0.1,
                audio_energy_variance=0.001, integrated_lufs=-14.0,
                speech_pace_wps=2.0, scene_change_rate_per_min=5.0,
                scene_change_count=3, avg_brightness=0.7, avg_saturation=0.5)
    base.update(over)
    return SensorReadings(**base)


# ── (1) formula rule ladder ─────────────────────────────────────────

def test_formula_energetic():
    c = apply_formula(_readings(audio_rms_mean=0.09, speech_pace_wps=3.0))
    assert (c.mood, c.style_pack) == ("energetic", "vibrant")


def test_formula_urgent_cuts():
    c = apply_formula(_readings(scene_change_rate_per_min=25.0))
    assert (c.mood, c.style_pack) == ("urgent", "news_flash")


def test_formula_somber():
    c = apply_formula(_readings(audio_rms_mean=0.01, speech_pace_wps=1.0))
    assert (c.mood, c.style_pack) == ("somber", "calm")


def test_formula_cinematic():
    c = apply_formula(_readings(avg_brightness=0.45, audio_rms_mean=0.05))
    assert (c.mood, c.style_pack) == ("cinematic", "cinematic")


def test_formula_default_neutral():
    c = apply_formula(_readings())  # mid energy, bright frame → no rule
    assert (c.mood, c.style_pack) == ("neutral", "minimal")
    assert c.rule_id == "default_fallback"


# ── (2)-(4) adapter → StoryDirective mapping ────────────────────────

def _story(idx, dur=30.0, title="టెస్ట్ కథ"):
    # Field names match the REAL canvas_schema.CanvasStory (title_native /
    # title_english / summary) — a generic `title=` here once masked an
    # adapter bug where story text never reached the LLM.
    return SimpleNamespace(story_index=idx, video_t_start=0.0,
                           video_t_end=dur, images=[],
                           title_native=title, title_english="Test story",
                           summary="")


def _words(n, span):
    """n words spread over span seconds → pace = n/span wps."""
    step = span / max(1, n)
    return [{"start": i * step, "end": i * step + step * 0.8}
            for i in range(n)]


@pytest.fixture()
def offline(monkeypatch):
    """No ffmpeg, no LLM: canned global sensors + formula-only mode."""
    monkeypatch.setenv("KAIZER_V4_PLATFORM_LLM", "0")

    def _fake(path, *, words=None, scene_change_threshold=0.4):
        return _readings()
    monkeypatch.setattr(ad, "extract_sensors", _fake)
    return monkeypatch


def _vocab_now(picks=None):
    from pipeline_v4.director import _vocab
    return _vocab(picks)


def test_adapter_energetic_maps_into_vocab(offline):
    offline.setattr(ad, "extract_sensors",
                    lambda p, **k: _readings(audio_rms_mean=0.09))
    plan = ad.plan_direction_platform(
        stories=[_story(0)], words_by_story={0: _words(9, 3.0)},  # 3 wps
        category="news", source_path="unused.mp4")
    d = plan[0]
    v = _vocab_now()
    assert d.transition_in in v["transitions"]
    assert all(f in v["fx"] for f in d.fx)
    assert d.transition_in == "zoom_punch_out"      # the energetic override
    assert "vibrance_pop" in d.fx
    assert d.why.startswith("platform:energetic_fast_pace")


def test_adapter_somber_silences_bed(offline):
    offline.setattr(ad, "extract_sensors",
                    lambda p, **k: _readings(audio_rms_mean=0.01))
    plan = ad.plan_direction_platform(
        stories=[_story(0)], words_by_story={0: _words(2, 2.0)},  # 1 wps
        category="news", source_path="unused.mp4")
    d = plan[0]
    assert d.bed_on is False
    v = _vocab_now()
    if d.grade:                                     # grade only if registry has it
        assert d.grade in v["grades"]


def test_adapter_neutral_is_pure_baseline(offline):
    from pipeline_v4.director import formula_plan
    stories = [_story(0)]
    plan = ad.plan_direction_platform(
        stories=stories, words_by_story={0: _words(4, 2.0)},  # 2 wps → neutral
        category="news", source_path="unused.mp4")
    base = formula_plan(stories, "news", None)[0]
    d = plan[0]
    assert (d.transition_in, d.fx, d.grade, d.bed_on) == \
        (base.transition_in, base.fx, base.grade, base.bed_on)


def test_adapter_accepts_string_story_keys(offline):
    """The story_words.json sidecar carries STRING keys ("0") — the adapter
    must still find the words (int-only lookup was the job-610 class bug)."""
    offline.setattr(ad, "extract_sensors",
                    lambda p, **k: _readings(audio_rms_mean=0.09))
    plan = ad.plan_direction_platform(
        stories=[_story(0)], words_by_story={"0": _words(9, 3.0)},
        category="news", source_path="unused.mp4")
    # pace found via the string key → energetic override applied
    assert plan[0].transition_in == "zoom_punch_out"


def test_adapter_user_pins_win_over_mood(offline):
    offline.setattr(ad, "extract_sensors",
                    lambda p, **k: _readings(audio_rms_mean=0.09))
    picks = {"transitions": ["fade"]}
    plan = ad.plan_direction_platform(
        stories=[_story(0)], words_by_story={0: _words(9, 3.0)},
        category="news", source_path="unused.mp4", user_picks=picks)
    # energetic wants zoom_punch_out, but the user pinned "fade" — the pin wins.
    assert plan[0].transition_in == "fade"


def test_story_text_reads_real_canvas_fields():
    """CanvasStory carries title_native/title_english/summary — the probe
    must surface them (the LLM judges tone from this text)."""
    s = SimpleNamespace(title_native="ఘోర ప్రమాదం", title_english="Tragedy",
                        summary="A somber story.")
    text = ad._story_text(s)
    assert "ఘోర ప్రమాదం" in text and "Tragedy" in text and "somber" in text


def test_adapter_llm_override_keys_on_pack_not_mood(offline):
    """An LLM override's mood is FREE TEXT ('tragic'); the validated
    style_pack ('calm') must drive the treatment — vendor semantics."""
    from pipeline_v4.director_platform.llm_refine import RefinedDirection
    offline.setenv("KAIZER_V4_PLATFORM_LLM", "1")
    offline.setattr(ad, "extract_sensors",
                    lambda p, **k: _readings(audio_rms_mean=0.09))
    offline.setattr(ad, "refine_with_llm",
                    lambda c, s, t: RefinedDirection(
                        mood="tragic", style_pack="calm",
                        reason="tragedy; use calm treatment",
                        overridden=True, provider_used="gemini"))
    plan = ad.plan_direction_platform(
        stories=[_story(0)], words_by_story={0: _words(9, 3.0)},
        category="news", source_path="unused.mp4")
    d = plan[0]
    # calm pack applied despite the out-of-vocab mood tag
    assert d.transition_in == "dissolve"
    assert d.bed_on is False
    assert "llm[gemini]" in d.why


def test_adapter_variety_no_adjacent_repeat(offline):
    """Same mood on every story must NOT mean the same transition on every
    story — the V4 anti-monotony invariant applies to platform plans too."""
    offline.setattr(ad, "extract_sensors",
                    lambda p, **k: _readings(audio_rms_mean=0.09))
    stories = [_story(0), _story(1), _story(2)]
    wbs = {i: _words(9, 3.0) for i in range(3)}
    plan = ad.plan_direction_platform(
        stories=stories, words_by_story=wbs,
        category="news", source_path="unused.mp4")
    assert plan[0].transition_in != plan[1].transition_in
    assert plan[1].transition_in != plan[2].transition_in


def test_adapter_writes_platform_trace(offline, tmp_path):
    """A platform plan persists director_trace.json (mode='platform') next
    to the source — the job UI's Director-decisions view reads it, and it
    must overwrite a stale v4 trace when a retry switches engines."""
    import json
    src = tmp_path / "master.mp4"
    src.write_bytes(b"\x00")
    (tmp_path / "director_trace.json").write_text(
        '{"mode": "llm"}', encoding="utf-8")     # stale v4 trace
    plan = ad.plan_direction_platform(
        stories=[_story(0)], words_by_story={0: _words(4, 2.0)},
        category="news", source_path=str(src))
    assert plan
    trace = json.loads((tmp_path / "director_trace.json")
                       .read_text(encoding="utf-8"))
    assert trace["mode"] == "platform"
    assert "0" in trace["decisions"]
    assert trace["stories"]["0"]["rule_id"]


# ── (5) fail-soft contracts ─────────────────────────────────────────

def test_adapter_empty_stories_returns_empty(offline):
    assert ad.plan_direction_platform(
        stories=[], words_by_story={}, category="news",
        source_path="x.mp4") == {}


def test_adapter_total_failure_returns_empty(monkeypatch):
    monkeypatch.setenv("KAIZER_V4_PLATFORM_LLM", "0")

    def _boom(path, **k):
        raise RuntimeError("sensor explosion")
    monkeypatch.setattr(ad, "extract_sensors", _boom)
    out = ad.plan_direction_platform(
        stories=[_story(0)], words_by_story={}, category="news",
        source_path="x.mp4")
    assert out == {}                                # orchestrator falls back to v4


def test_llm_refine_fails_open_without_story_text():
    c = FormulaCandidate(mood="energetic", style_pack="vibrant",
                         rule_id="r", reason="x")
    r = refine_with_llm(c, _readings(), None)
    assert (r.mood, r.style_pack, r.overridden) == ("energetic", "vibrant", False)


def test_llm_refine_fails_open_on_provider_error(monkeypatch):
    from pipeline_v4.director_platform import llm_refine as lr

    def _boom(prompt):
        raise RuntimeError("no key")
    monkeypatch.setattr(lr, "_PROVIDERS", {"gemini": _boom, "claude": _boom})
    c = FormulaCandidate(mood="somber", style_pack="calm",
                         rule_id="r", reason="x")
    r = refine_with_llm(c, _readings(), "a sad story about loss")
    assert (r.mood, r.style_pack, r.overridden) == ("somber", "calm", False)
