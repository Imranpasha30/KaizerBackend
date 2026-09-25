"""Unit 4 — the image↔speech timing engine.

Covers the sanitizer (never trust raw model JSON), the fail-soft entry
point, schema backward-compat, and the cache-hash regression rule
(default new fields must NOT change any pre-engine hash).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from pipeline_v4 import image_timing as it
from pipeline_v4.canvas_schema import CanvasImage


def _words(*triples):
    return [{"w": w, "s": float(s), "e": float(e)} for (w, s, e) in triples]


W = _words(("మోదీ", 1.0, 1.4), ("పార్లమెంట్", 1.5, 2.1), ("వరదలు", 6.0, 6.6),
            ("హైదరాబాద్", 7.0, 7.8), ("rescue", 8.0, 8.4))


def _san(entries, **kw):
    args = dict(duration=20.0, pool_count=3,
                pool_labels=["Modi parliament", "Hyderabad floods", "rescue boats"],
                words=W, min_dwell_s=2.5, max_dwell_s=8.0, min_conf=0.55)
    args.update(kw)
    return it.sanitize_timing_plan(entries, **args)


# ─── Sanitizer ──────────────────────────────────────────────────────


def test_sanitize_drops_invalid_entries():
    plan = _san([
        {"pool_index": 99, "t_start": 0, "t_end": 5},          # bad index
        {"pool_index": "x", "t_start": 0, "t_end": 5},          # non-numeric
        {"pool_index": 0, "t_start": 5, "t_end": 5.01},         # zero-length
        "garbage",                                               # not a dict
        {"pool_index": 0, "t_start": 0.0, "t_end": 4.0,
         "confidence": 0.9, "anchor_word_indexes": [0, 1]},      # the one good row
    ])
    assert len(plan) == 1
    assert plan[0]["pool_index"] == 0


def test_sanitize_returns_none_when_nothing_survives():
    assert _san([]) is None
    assert _san([{"pool_index": 42, "t_start": 0, "t_end": 4}]) is None
    assert it.sanitize_timing_plan(
        [{"pool_index": 0, "t_start": 0, "t_end": 4}],
        duration=0.0, pool_count=1) is None


def test_sanitize_clamps_and_snaps_edges():
    plan = _san([{"pool_index": 0, "t_start": 0.2, "t_end": 19.9,
                  "confidence": 0.9, "anchor_word_indexes": [0]}])
    # 0.2 snaps to 0.0; 19.9 snaps to duration then truncates to max dwell
    assert plan[0]["t_start"] == 0.0
    assert plan[0]["t_end"] == 8.0     # max dwell truncation


def test_sanitize_resolves_overlaps_by_trimming_later():
    plan = _san([
        {"pool_index": 0, "t_start": 1.0, "t_end": 6.0,
         "confidence": 0.9, "anchor_word_indexes": [0]},
        {"pool_index": 1, "t_start": 4.0, "t_end": 10.0,
         "confidence": 0.9, "anchor_word_indexes": [3]},
    ])
    assert plan[0]["t_end"] == 6.0
    assert plan[1]["t_start"] == 6.0   # later window trimmed forward
    assert plan[1]["t_end"] == 10.0


def test_sanitize_dwell_extend_and_drop():
    # 1.0s window extends to min dwell (2.5s) when space is free
    plan = _san([{"pool_index": 0, "t_start": 1.0, "t_end": 2.0,
                  "confidence": 0.9, "anchor_word_indexes": [0]}])
    assert plan[0]["t_end"] == pytest.approx(3.5)
    # but drops when the next window blocks reaching min dwell
    plan = _san([
        {"pool_index": 0, "t_start": 1.0, "t_end": 2.0,
         "confidence": 0.9, "anchor_word_indexes": [0]},
        {"pool_index": 1, "t_start": 2.2, "t_end": 6.2,
         "confidence": 0.9, "anchor_word_indexes": [2]},
    ])
    assert [e["pool_index"] for e in plan] == [1]


def test_sanitize_pinned_windows_are_routed_around():
    plan = _san([
        {"pool_index": 0, "t_start": 0.5, "t_end": 7.0,     # overlaps pin tail
         "confidence": 0.9, "anchor_word_indexes": [0]},
        {"pool_index": 1, "t_start": 5.0, "t_end": 9.0,     # head clipped to pin end
         "confidence": 0.9, "anchor_word_indexes": [4]},    # word at 8.0-8.4
    ], pinned=((4.0, 8.0),))
    # window 1: tail clipped at 4.0; window 2: head clipped to 8.0
    assert plan[0]["t_end"] == 4.0
    assert plan[1]["t_start"] == 8.0


def test_sanitize_pinned_drops_fully_covered_and_stale_anchor():
    """A window fully inside a pin dies; a clipped window whose anchor no
    longer falls inside it fails the confidence gate (a mistimed image is
    worse than none)."""
    plan = _san([
        {"pool_index": 0, "t_start": 4.5, "t_end": 7.5,     # fully inside pin
         "confidence": 0.9, "anchor_word_indexes": [2]},
        {"pool_index": 1, "t_start": 5.0, "t_end": 9.0,     # clipped to [8,9+];
         "confidence": 0.9, "anchor_word_indexes": [3]},    # anchor at 7.4 → outside
    ], pinned=((4.0, 8.0),))
    assert plan is None


def test_sanitize_confidence_gate_and_anchor_verification():
    # Anchors outside the window → confidence halved → below gate → gap.
    plan = _san([{"pool_index": 0, "t_start": 10.0, "t_end": 14.0,
                  "confidence": 0.8, "anchor_word_indexes": [0]}])   # word at 1.0s
    assert plan is None
    # Same entry with a matching anchor survives.
    plan = _san([{"pool_index": 1, "t_start": 5.5, "t_end": 9.0,
                  "confidence": 0.8, "anchor_word_indexes": [2, 3]}])
    assert plan and plan[0]["matched_text"] == "వరదలు హైదరాబాద్"


def test_sanitize_entity_consistency_same_label_same_image():
    plan = _san([
        {"pool_index": 0, "t_start": 0.0, "t_end": 4.0,
         "confidence": 0.9, "anchor_word_indexes": [0]},
        {"pool_index": 2, "t_start": 6.0, "t_end": 10.0,
         "confidence": 0.9, "anchor_word_indexes": [4]},
    ], pool_labels=["Modi PC", "floods", "modi  pc"])   # 0 and 2 = same entity
    assert plan[1]["pool_index"] == 0


# ─── Entry point (fail-soft) ────────────────────────────────────────


def _decide(monkeypatch, raw_response, **kw):
    monkeypatch.setattr(it, "_call_model", lambda s, u: raw_response)
    args = dict(title_native="టెస్ట్", title_english="test", summary="s",
                duration=20.0, words=W,
                pool=[{"label": "Modi parliament", "kind": "ai"},
                      {"label": "Hyderabad floods", "kind": "photo"}],
                language="te")
    args.update(kw)
    return it.decide_story_timings(**args)


def test_decide_valid_response(monkeypatch):
    raw = json.dumps({"images": [
        {"pool_index": 0, "t_start": 0.2, "t_end": 4.0, "confidence": 0.9,
         "importance": 0.85, "anchor_word_indexes": [0, 1], "reason": "modi spoken"},
        {"pool_index": 1, "t_start": 5.8, "t_end": 9.0, "confidence": 0.8,
         "importance": 0.4, "anchor_word_indexes": [2, 3]},
    ]})
    plan = _decide(monkeypatch, raw)
    assert len(plan) == 2
    assert plan[0]["t_start"] == 0.0          # 0.2 is within the 0.3s snap
    assert plan[0]["importance"] == 0.85
    assert plan[1]["matched_text"]


def test_decide_gaps_allowed(monkeypatch):
    """A sparse plan (one 4s window in a 20s story) is returned as-is —
    the rest of the story is a gap (video full-screen)."""
    raw = json.dumps({"images": [
        {"pool_index": 0, "t_start": 6.0, "t_end": 10.0, "confidence": 0.9,
         "anchor_word_indexes": [2]},
    ]})
    plan = _decide(monkeypatch, raw)
    assert len(plan) == 1
    assert plan[0]["t_start"] == 6.0


def test_decide_garbage_returns_none(monkeypatch):
    assert _decide(monkeypatch, "totally not json {{{") is None
    assert _decide(monkeypatch, json.dumps({"images": []})) is None


def test_decide_model_exception_returns_none(monkeypatch):
    def _boom(s, u):
        raise RuntimeError("api down")
    monkeypatch.setattr(it, "_call_model", _boom)
    assert it.decide_story_timings(
        duration=20.0, words=W, pool=[{"label": "x", "kind": "ai"}]) is None


def test_decide_disabled_or_no_words(monkeypatch):
    monkeypatch.setenv("KAIZER_V4_IMAGE_TIMING", "0")
    assert _decide(monkeypatch, "{}") is None
    monkeypatch.delenv("KAIZER_V4_IMAGE_TIMING", raising=False)
    assert _decide(monkeypatch, "{}", words=[]) is None


def test_model_spec_parsing(monkeypatch):
    monkeypatch.delenv("KAIZER_V4_IMAGE_TIMING_MODEL", raising=False)
    assert it._model_spec() == ("gemini", "gemini-2.5-flash")
    monkeypatch.setenv("KAIZER_V4_IMAGE_TIMING_MODEL", "claude:claude-opus-4-7")
    assert it._model_spec() == ("claude", "claude-opus-4-7")
    monkeypatch.setenv("KAIZER_V4_IMAGE_TIMING_MODEL", "banana:x")
    assert it._model_spec() == ("gemini", "gemini-2.5-flash")


# ─── Schema backward-compat + cache-hash regression ─────────────────


def test_old_canvas_image_validates_with_defaults():
    """A pre-engine CanvasImage dict (no new fields) must validate and
    get pure defaults — old canvas.json files keep working."""
    img = CanvasImage.model_validate({
        "src": "news_01.jpg", "t_start": 0.0, "t_end": 4.0,
    })
    assert img.timing_mode == "content"
    assert img.confidence is None
    assert img.importance is None
    assert img.spotlight is None
    assert img.matched_text is None


def _hash_for(images):
    from pipeline_v4.v1_bridge import _per_story_cache_hash
    story = SimpleNamespace(title_native="t", title_english="t", summary="s",
                            video_t_start=0.0, video_t_end=10.0,
                            story_index=0, total_stories=1)
    return _per_story_cache_hash(
        story=story, ticker_path="", sidebar_path="", layout=None,
        channel_bug_path="", watermark_path="", watermark_position="",
        font_path="", bg_video_abs=None, bg_video_volume=0.0,
        language_code="te", images=images, pool_dir=None,
    )


def test_cache_hash_unchanged_by_default_new_fields():
    """THE regression guard: adding the Phase-1 fields with default
    values must produce the EXACT hash an old canvas produced — else
    every cached story on LIVE would re-render after promote."""
    old_shape = [{"src": "a.jpg", "t_start": 0.0, "t_end": 4.0,
                  "effect": "fade", "effect_duration": 0.4,
                  "fit": "cover", "offset_x_pct": 50.0, "offset_y_pct": 50.0}]
    new_defaults = [dict(old_shape[0],
                         timing_mode="content", confidence=None,
                         importance=None, spotlight=None, matched_text=None)]
    assert _hash_for(old_shape) == _hash_for(new_defaults)


def test_cache_hash_changes_when_spotlight_set():
    base = [{"src": "a.jpg", "t_start": 0.0, "t_end": 4.0,
             "effect": "fade", "effect_duration": 0.4,
             "fit": "cover", "offset_x_pct": 50.0, "offset_y_pct": 50.0}]
    spot = [dict(base[0], spotlight="fullscreen")]
    off = [dict(base[0], spotlight="off")]
    assert _hash_for(base) != _hash_for(spot)
    assert _hash_for(base) == _hash_for(off)   # "off" renders identically
