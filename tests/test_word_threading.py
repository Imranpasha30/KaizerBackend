"""Unit 3 — per-word timestamps threaded onto stories.

The trimmed video is the KEEP spans concatenated; a Deepgram word at
SOURCE time ws inside a span whose output copy starts at span_out_start
lands at  span_out_start + (ws - span.start)  in the output, minus the
story's own output start to make it story-relative. Words in cut regions
are dropped; straddlers clamp to the span edge.
"""
from __future__ import annotations

import contextlib
import json

import pytest

from pipeline_v4 import trim_engine
from pipeline_v4 import orchestrator as orch


def _w(word, s, e):
    return {"i": 0, "w": word, "s": float(s), "e": float(e)}


# ─── run_step1 end-to-end remap (mocked I/O — the real _flatten runs) ───


@pytest.fixture
def step1(tmp_path, monkeypatch):
    """Run trim_engine.run_step1 with synthetic words + a scripted planner
    and every external effect (ffmpeg, Deepgram, NVENC gate, belts) stubbed.
    Returns a runner fn: (words, planner_stories, duration) -> TrimResult."""
    def _run(words, planner_stories, duration):
        # These tests target the remap math: pin the content type (no LLM
        # classification) and disable the Unit-C silence tightening so the
        # scripted planner spans reach _flatten verbatim.
        monkeypatch.setenv("KAIZER_V4_CONTENT_TYPE", "news")
        monkeypatch.setenv("KAIZER_V4_TIGHTEN_SILENCE", "0")
        monkeypatch.setattr(trim_engine, "_extract_audio_mp3",
                            lambda src, dst, **k: dst)
        monkeypatch.setattr(trim_engine, "_deepgram_words",
                            lambda path, language="multi": (words, duration))
        monkeypatch.setattr(trim_engine, "_select_planner",
                            lambda: ("test", lambda **k: (planner_stories, 0.0)))
        monkeypatch.setattr(trim_engine, "_atomic_trim_concat",
                            lambda **k: None)
        monkeypatch.setattr(trim_engine, "_encode_gate",
                            lambda: contextlib.nullcontext())
        monkeypatch.setattr(trim_engine, "_belt", lambda *a, **k: None)
        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake")
        return trim_engine.run_step1(
            source_video=str(src), output_dir=str(tmp_path / "out"),
            language="te",
        )
    return _run


def test_words_remap_across_multiple_spans(step1):
    """Story = two non-contiguous KEEP spans [10,15] + [20,25].
    Output timeline: span1 → [0,5), span2 → [5,10)."""
    words = [
        _w("early", 2.0, 2.5),      # before any span → dropped
        _w("alpha", 11.0, 11.5),    # span1: 11-10 = story-rel 1.0
        _w("cutme", 17.0, 17.5),    # in the cut hole → dropped
        _w("beta", 21.0, 21.4),     # span2: 5 + (21-20) = story-rel 6.0
        _w("gamma", 24.5, 25.5),    # straddles span2 end → clamped to 10.0
    ]
    plan = [{
        "title_native": "t", "title_english": "t", "summary": "s",
        "kept_spans": [
            {"start_sec": 10.0, "end_sec": 15.0},
            {"start_sec": 20.0, "end_sec": 25.0},
        ],
    }]
    res = step1(words, plan, 30.0)
    st = res.stories[0]
    got = {w["w"]: (w["s"], w["e"]) for w in st.words}
    assert set(got) == {"alpha", "beta", "gamma"}
    assert got["alpha"] == (1.0, 1.5)
    assert got["beta"] == (6.0, 6.4)
    assert got["gamma"] == (9.5, 10.0)   # end clamped to span edge
    # transcript_text still built from the same walk
    assert st.transcript_text == "alpha beta gamma"


def test_words_are_story_relative_for_second_story(step1):
    """Two stories: story 2's words start at 0 relative to ITS start."""
    words = [_w("one", 1.0, 1.5), _w("two", 31.0, 31.5)]
    plan = [
        {"title_native": "a", "title_english": "a", "summary": "",
         "kept_spans": [{"start_sec": 0.0, "end_sec": 10.0}]},
        {"title_native": "b", "title_english": "b", "summary": "",
         "kept_spans": [{"start_sec": 30.0, "end_sec": 40.0}]},
    ]
    res = step1(words, plan, 60.0)
    s1, s2 = res.stories
    assert s1.words == [{"w": "one", "s": 1.0, "e": 1.5}]
    # story 2 starts at output 10.0; word at source 31 → output 11 → rel 1.0
    assert s2.words == [{"w": "two", "s": 1.0, "e": 1.5}]


def test_full_source_fallback_words_identity(step1):
    """Planner keeps nothing (twice) → full-source fallback story carries
    the whole word list with identity times."""
    words = [_w("hello", 0.5, 1.0), _w("world", 2.0, 2.6)]
    res = step1(words, [], 12.0)
    st = res.stories[0]
    assert st.words == [{"w": "hello", "s": 0.5, "e": 1.0},
                        {"w": "world", "s": 2.0, "e": 2.6}]
    assert st.video_t_end == 12.0


# ─── _identity_words (source-preserved / audio-first paths) ─────────


def test_identity_words_window_and_clamp():
    words = [_w("pre", 1.0, 2.0), _w("in", 6.0, 6.5),
             _w("edge", 9.5, 10.5), _w("post", 12.0, 13.0)]
    out = orch._identity_words(words, 5.0, 10.0)
    got = {w["w"]: (w["s"], w["e"]) for w in out}
    assert set(got) == {"in", "edge"}
    assert got["in"] == (1.0, 1.5)
    assert got["edge"] == (4.5, 5.0)     # clamped to the window end


def test_identity_words_empty_and_garbage():
    assert orch._identity_words([], 0, 10) == []
    assert orch._identity_words([{"w": "x", "s": "bad", "e": None}], 0, 10) == []


# ─── story_words.json sidecar ───────────────────────────────────────


def test_sidecar_roundtrip(tmp_path):
    stories = [
        trim_engine.TrimmedStory(
            story_index=0, title_native="", title_english="", summary="",
            video_t_start=0.0, video_t_end=5.0,
            words=[{"w": "మోదీ", "s": 1.42, "e": 1.81}],
        ),
        trim_engine.TrimmedStory(
            story_index=1, title_native="", title_english="", summary="",
            video_t_start=5.0, video_t_end=9.0,
            words=[{"w": "వరదలు", "s": 0.3, "e": 0.9}],
        ),
    ]
    tr = trim_engine.TrimResult(
        trimmed_path="x.mp4", trimmed_duration_sec=9.0, stories=stories,
        source_duration_sec=9.0, removed_sec_total=0.0,
    )
    orch._write_story_words_sidecar(tmp_path, tr, language="te")
    data = json.loads((tmp_path / "story_words.json").read_text(encoding="utf-8"))
    assert data["schema"] == 1
    assert data["language"] == "te"
    assert data["stories"]["0"] == [{"w": "మోదీ", "s": 1.42, "e": 1.81}]
    assert data["stories"]["1"][0]["w"] == "వరదలు"


def test_sidecar_write_fail_soft(tmp_path):
    """A bogus target dir must not raise — the sidecar is a convenience."""
    tr = trim_engine.TrimResult(
        trimmed_path="x", trimmed_duration_sec=0, stories=[],
        source_duration_sec=0, removed_sec_total=0,
    )
    orch._write_story_words_sidecar(tmp_path / "does" / "not" / "exist", tr)
