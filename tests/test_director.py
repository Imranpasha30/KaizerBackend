"""UG.7 — AI Director: formulas, sanitizer, fail-soft, chain building."""
from __future__ import annotations

from types import SimpleNamespace

from pipeline_v4 import director as dr


def _story(idx, a=0.0, b=30.0, title="t", summary="s"):
    return SimpleNamespace(story_index=idx, video_t_start=a, video_t_end=b,
                           title_native=title, summary=summary)


STORIES = [_story(0), _story(1, 30.0, 55.0), _story(2, 55.0, 90.0)]


def test_formulas_reference_real_registry_ids():
    """Every formula id must exist in the real registries — a formula
    with a typo would silently break the baseline layer."""
    v = dr._vocab()
    for key, f in dr.FORMULAS.items():
        assert f.pack in v["packs"], (key, f.pack)
        assert f.transition in v["transitions"], (key, f.transition)
        for fx in f.fx:
            assert fx in v["fx"], (key, fx)
        for o in f.overlays:
            assert o in v["overlays"], (key, o)
        assert f.captions in v["captions"], (key, f.captions)
    assert dr.formula_for("MURDER_HOMICIDE").pack == "crime"   # alias route
    # unknown text routes through the taxonomy default (news), never crashes
    assert dr.formula_for("nonsense").pack == "news"


def test_sanitizer_enforces_vocabulary_and_clamps():
    raw = {"stories": [
        {"index": 0, "mood": "comedy", "transition_in": "glitch_slices",
         "fx": ["vignette_soft", "slow_motion", "made_up_fx"],
         "overlays": [{"id": "tag_investigation", "t": 999.0},
                      {"id": "not_an_overlay", "t": 1.0}],
         "captions": "karaoke_box_yellow", "emphasis": [5.0, 999.0]},
        {"index": 1, "mood": "hallucinated_pack",
         "transition_in": "warp_speed", "captions": "invented"},
        {"index": 99, "mood": "comedy"},        # unknown story → dropped
        "garbage",
    ]}
    plan = dr.sanitize_plan(raw, stories=STORIES, category="crime")
    assert set(plan) == {0, 1, 2}
    d0 = plan[0]
    assert d0.mood == "comedy" and d0.transition_in == "glitch_slices"
    assert d0.fx == ["vignette_soft"]            # banned + fake dropped
    assert d0.overlays == [{"id": "tag_investigation", "t": 28.0}]  # clamped
    assert d0.captions == "karaoke_box_yellow"
    assert d0.emphasis == [5.0, 30.0]
    # invalid ids → formula fallbacks (crime)
    d1 = plan[1]
    assert d1.mood == "crime" and d1.transition_in == "glitch_slices"
    assert d1.captions == "none"
    # skipped story 2 → full formula row
    d2 = plan[2]
    assert d2.mood == "crime" and d2.overlays[0]["id"] == "tag_investigation"


def test_formula_plan_covers_all_stories():
    plan = dr.formula_plan(STORIES, "sports")
    assert set(plan) == {0, 1, 2}
    assert all(d.mood == "sports" for d in plan.values())
    assert all(d.transition_in == "whip_right" for d in plan.values())


def test_plan_direction_fail_soft(monkeypatch):
    """LLM dead → formula baseline, never an exception."""
    import seo.generator as gen
    def _boom():
        raise RuntimeError("no llm in tests")
    monkeypatch.setattr(gen, "_gemini_client", _boom)
    plan = dr.plan_direction(stories=STORIES, words_by_story={},
                             category="weather", source_path="")
    assert set(plan) == {0, 1, 2}
    assert all(d.mood == "weather" for d in plan.values())


def test_story_fx_chain_is_linear_and_real():
    d = dr.StoryDirective(mood="crime", fx=["vignette_soft", "film_grain"])
    chain = dr.story_fx_chain(d)
    assert chain and ";" not in chain
    assert "vignette" in chain and "noise" in chain
    # unknown mood + no fx → base chain passthrough
    d2 = dr.StoryDirective(mood="not_a_pack")
    assert dr.story_fx_chain(d2, base_chain="eq=contrast=1.1") == "eq=contrast=1.1"


def test_gate_env(monkeypatch):
    monkeypatch.delenv("KAIZER_V4_DIRECTOR", raising=False)
    assert dr.directives_enabled() is False
    monkeypatch.setenv("KAIZER_V4_DIRECTOR", "1")
    assert dr.directives_enabled() is True


def test_stitch_graph_accepts_per_joint_list():
    """Director per-joint transitions flow through the bulletin stitch
    builder — including the 20 custom expressions."""
    from pipeline_v4.v1_bridge import build_xfade_stitch_graph
    fc, v, a = build_xfade_stitch_graph(
        [10.0, 8.0, 12.0], transition=["glitch_slices", "clock_sweep"],
        fade_d=0.5)
    assert "transition=custom:expr=" in fc          # customs resolved
    assert fc.count("xfade=") == 2                  # one per joint
    # single-name legacy call is unchanged in shape
    fc2, _, _ = build_xfade_stitch_graph([10.0, 8.0], transition="fade")
    assert "xfade=transition=fade:" in fc2


def test_tone_slice_and_sanitize(tmp_path):
    """Tone sensor plumbing: real 16k mono slice + schema clamping."""
    import subprocess
    src = tmp_path / "story.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "sine=frequency=300:duration=8",
         "-f", "lavfi", "-i", "color=c=black:s=320x180:d=8:r=25",
         "-map", "1:v", "-map", "0:a", "-c:v", "libx264", "-preset",
         "ultrafast", "-c:a", "aac", "-shortest", str(src)],
        check=True, capture_output=True, timeout=120)
    wav = dr._slice_story_audio(str(src), start=1.0, end=6.0,
                                out_dir=str(tmp_path), idx=0)
    assert wav
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=sample_rate,channels", "-of", "csv=p=0", wav],
        capture_output=True, text=True, timeout=30).stdout.strip()
    assert probe.startswith("16000,1")
    assert dr._slice_story_audio(str(src), start=1.0, end=1.2,
                                 out_dir=str(tmp_path), idx=1) is None
    # sanitizer clamps + rejects
    ok = dr._san_tone({"tone": "URGENT", "arousal": 7, "valence": -9,
                       "note": "x" * 200})
    assert ok == {"tone": "urgent", "arousal": 1.0, "valence": -1.0,
                  "note": "x" * 80}
    assert dr._san_tone({"tone": "vibing"}) is None
    assert dr._san_tone("junk") is None


def test_tone_sense_fail_soft(monkeypatch, tmp_path):
    """No Gemini → {} and the Director proceeds from words alone."""
    import seo.generator as gen
    monkeypatch.setattr(gen, "_gemini_client",
                        lambda: (_ for _ in ()).throw(RuntimeError("no llm")))
    assert dr.sense_tone("missing.mp4", STORIES) == {}
    monkeypatch.setenv("KAIZER_V4_TONE_SENSE", "0")
    assert dr.sense_tone("missing.mp4", STORIES) == {}
    monkeypatch.delenv("KAIZER_V4_TONE_SENSE", raising=False)
    monkeypatch.setenv("KAIZER_V4_SER", "0")
    assert dr.sense_ser({0: "x.wav"}) == {}
