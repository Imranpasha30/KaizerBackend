"""Unit C — edit profiles, deterministic silence tightening, type detection."""
from __future__ import annotations

import json

import pytest

from pipeline_v4 import content_type as ct
from pipeline_v4 import edit_profiles as ep
from pipeline_v4.trim_engine import tighten_spans_to_speech


# ─── Profiles / prompt generation ───────────────────────────────────


def test_news_profile_is_byte_identical_to_legacy_prompt():
    """THE compat lock: every existing job plans under 'news', and the
    news prompt must be exactly the string it always was."""
    from pipeline_v4.prompts import KEEP_CUT_SYSTEM
    assert ep.keep_cut_system_for("news") == KEEP_CUT_SYSTEM


def test_all_profiles_produce_valid_planner_prompts():
    for key, p in ep.PROFILES.items():
        s = ep.keep_cut_system_for(key)
        assert p.persona.split(".")[0] in s
        # the schema contract every planner parser depends on
        assert '"kept_spans"' in s and '"removed_sec_total"' in s
        assert "TARGET_MIN_SEC" in s
    # unknown key → news (never crash a job on a typo)
    assert ep.keep_cut_system_for("nonsense") == ep.keep_cut_system_for("news")


def test_active_profile_key_env(monkeypatch):
    monkeypatch.delenv("KAIZER_V4_EDIT_PROFILE", raising=False)
    assert ep.active_profile_key() == "news"
    monkeypatch.setenv("KAIZER_V4_EDIT_PROFILE", "podcast")
    assert ep.active_profile_key() == "podcast"
    monkeypatch.setenv("KAIZER_V4_EDIT_PROFILE", "garbage")
    assert ep.active_profile_key() == "news"


# ─── Deterministic silence tightening ───────────────────────────────


def _w(s, e):
    return {"w": "x", "s": s, "e": e}


def test_tighten_splits_at_internal_silence():
    """A kept span containing a 3s dead-air hole gets split in two."""
    words = [_w(1.0, 1.4), _w(1.5, 2.0),          # speech run A
             _w(5.0, 5.5), _w(5.6, 6.2)]           # speech run B (3s gap)
    stories = [{"kept_spans": [{"start_sec": 0.5, "end_sec": 7.0, "reason": "r"}]}]
    out = tighten_spans_to_speech(stories, words, max_gap=1.2, pad=0.25)
    spans = out[0]["kept_spans"]
    assert len(spans) == 2
    assert spans[0]["start_sec"] == pytest.approx(0.75, abs=0.01)   # 1.0 - pad
    assert spans[0]["end_sec"] == pytest.approx(2.25, abs=0.01)     # 2.0 + pad
    assert spans[1]["start_sec"] == pytest.approx(4.75, abs=0.01)
    assert spans[1]["end_sec"] == pytest.approx(6.45, abs=0.01)


def test_tighten_trims_edges_to_speech():
    """Leading/trailing dead air inside a span is shaved to pad."""
    words = [_w(3.0, 3.5), _w(3.6, 4.0)]
    stories = [{"kept_spans": [{"start_sec": 0.0, "end_sec": 9.0, "reason": ""}]}]
    spans = tighten_spans_to_speech(stories, words, max_gap=1.2, pad=0.25)[0]["kept_spans"]
    assert len(spans) == 1
    assert spans[0]["start_sec"] == pytest.approx(2.75, abs=0.01)
    assert spans[0]["end_sec"] == pytest.approx(4.25, abs=0.01)


def test_tighten_keeps_wordless_spans_verbatim():
    """The planner may keep intentional non-speech footage — untouched."""
    stories = [{"kept_spans": [{"start_sec": 10.0, "end_sec": 14.0, "reason": "broll"}]}]
    spans = tighten_spans_to_speech(stories, [_w(1, 2)], max_gap=1.2, pad=0.25)[0]["kept_spans"]
    assert spans == [{"start_sec": 10.0, "end_sec": 14.0, "reason": "broll"}]


def test_tighten_never_leaves_a_story_empty():
    """Fragments all below min_span → the originals are kept (never
    silently delete a story)."""
    words = [_w(1.0, 1.1)]
    stories = [{"kept_spans": [{"start_sec": 0.9, "end_sec": 1.2, "reason": ""}]}]
    out = tighten_spans_to_speech(stories, words, max_gap=1.2, pad=0.0, min_span=0.4)
    assert out[0]["kept_spans"] == stories[0]["kept_spans"]


def test_tighten_respects_gap_threshold():
    """Gaps smaller than max_gap never split (breathing room stays)."""
    words = [_w(1.0, 1.5), _w(2.4, 3.0)]           # 0.9s gap
    stories = [{"kept_spans": [{"start_sec": 0.5, "end_sec": 3.5, "reason": ""}]}]
    spans = tighten_spans_to_speech(stories, words, max_gap=1.2, pad=0.25)[0]["kept_spans"]
    assert len(spans) == 1


# ─── Content-type resolution ────────────────────────────────────────


_WORDS = [{"w": f"word{i}", "s": i * 0.4, "e": i * 0.4 + 0.3} for i in range(60)]


def test_explicit_pick_wins_no_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIZER_V4_CONTENT_TYPE", "podcast")
    monkeypatch.setattr(ct, "_classify_llm",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM must not run")))
    key = ct.resolve_and_record(words=_WORDS, duration=600, out_dir=tmp_path)
    assert key == "podcast"
    rec = json.loads((tmp_path / ct.SIDECAR_NAME).read_text(encoding="utf-8"))
    assert rec["source"] == "user" and rec["needs_confirmation"] is False
    import os
    assert os.environ["KAIZER_V4_EDIT_PROFILE"] == "podcast"


def test_confident_detection_used(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIZER_V4_CONTENT_TYPE", "auto")
    monkeypatch.setattr(ct, "_classify_llm",
                        lambda words, duration, language="": {
                            "type": "interview", "confidence": 0.87, "reasons": "Q&A"})
    key = ct.resolve_and_record(words=_WORDS, duration=1200, out_dir=tmp_path)
    assert key == "interview"
    rec = json.loads((tmp_path / ct.SIDECAR_NAME).read_text(encoding="utf-8"))
    assert rec["source"] == "detected" and rec["confidence"] == 0.87


def test_unsure_falls_back_to_news_and_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIZER_V4_CONTENT_TYPE", "auto")
    monkeypatch.setattr(ct, "_classify_llm",
                        lambda words, duration, language="": {
                            "type": "vlog", "confidence": 0.35, "reasons": "unclear"})
    key = ct.resolve_and_record(words=_WORDS, duration=300, out_dir=tmp_path)
    assert key == "news"                          # legacy behavior when unsure
    rec = json.loads((tmp_path / ct.SIDECAR_NAME).read_text(encoding="utf-8"))
    assert rec["needs_confirmation"] is True      # the "ask the user" hook
    assert rec["detected_type"] == "vlog"


def test_llm_failure_falls_back_to_news(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIZER_V4_CONTENT_TYPE", "auto")
    monkeypatch.setattr(ct, "_classify_llm", lambda *a, **k: None)
    assert ct.resolve_and_record(words=_WORDS, duration=300, out_dir=tmp_path) == "news"
