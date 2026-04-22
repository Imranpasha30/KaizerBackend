"""
tests/test_clip_boundaries.py
==============================
Pytest coverage for pipeline_core.clip_boundaries.

All tests are pure-Python / fast — no subprocess, no GPU, no network.
Covers snap_boundaries() and detect_completion() exhaustively.
"""
from __future__ import annotations

import pytest
from dataclasses import fields

from pipeline_core.clip_boundaries import snap_boundaries, detect_completion, SnapResult
from pipeline_core.asr import Sentence, Word


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_sentence(text: str, start: float = 0.0, end: float = 1.0) -> Sentence:
    """Build a minimal Sentence for testing."""
    return Sentence(
        text=text,
        start=start,
        end=end,
        words=[Word(text=text, start=start, end=end)],
    )


# ── snap_boundaries: dataclass shape ──────────────────────────────────────────

def test_snap_returns_dataclass():
    """snap_boundaries must return a SnapResult with all expected fields."""
    result = snap_boundaries(10.0, 20.0, shots=[], sentences=None, valleys=None)
    assert isinstance(result, SnapResult), f"Expected SnapResult, got {type(result)}"
    field_names = {f.name for f in fields(result)}
    for expected in ("start", "end", "start_sources", "end_sources", "adjusted_s", "warnings"):
        assert expected in field_names, f"Missing field {expected!r} in SnapResult"


# ── snap_boundaries: signal snapping ──────────────────────────────────────────

def test_snap_uses_nearest_shot_within_window():
    """A shot boundary within the search window snaps the proposed edge."""
    result = snap_boundaries(
        10.0, 20.0,
        shots=[10.8],        # 0.8 s from proposed_start=10.0 — within default 2s window
        sentences=None,
        valleys=None,
    )
    assert abs(result.start - 10.8) < 1e-6, (
        f"Expected start snapped to 10.8, got {result.start}"
    )
    assert "shot" in result.start_sources, (
        f"Expected 'shot' in start_sources, got {result.start_sources}"
    )


def test_snap_uses_nearest_sentence_within_window():
    """A sentence start within the search window snaps the start edge."""
    sent = _make_sentence("Hello world.", start=9.5, end=11.0)
    result = snap_boundaries(
        10.0, 20.0,
        shots=[],
        sentences=[sent],
        valleys=None,
    )
    assert abs(result.start - 9.5) < 1e-6, (
        f"Expected start snapped to 9.5 (sentence.start), got {result.start}"
    )
    assert "sentence" in result.start_sources, (
        f"Expected 'sentence' in start_sources, got {result.start_sources}"
    )


def test_snap_uses_nearest_valley_within_window():
    """An audio valley within the search window snaps the end edge."""
    result = snap_boundaries(
        10.0, 20.0,
        shots=[],
        sentences=None,
        valleys=[19.5],   # 0.5 s from proposed_end=20.0 — within default 2s window
    )
    assert abs(result.end - 19.5) < 1e-6, (
        f"Expected end snapped to 19.5 (valley), got {result.end}"
    )
    assert "valley" in result.end_sources, (
        f"Expected 'valley' in end_sources, got {result.end_sources}"
    )


def test_snap_start_preference_sentence_over_shot_over_valley():
    """For the start edge, sentence (priority 0) beats shot (1) beats valley (2)."""
    sent = _make_sentence("Starting sentence.", start=10.3, end=11.0)
    result = snap_boundaries(
        10.0, 30.0,
        shots=[10.5],         # also within window
        sentences=[sent],     # sentence.start=10.3, closer but higher priority
        valleys=[10.1],       # also within window
    )
    # Sentence wins regardless of absolute proximity because priority=0
    assert abs(result.start - 10.3) < 1e-6, (
        f"Expected sentence start 10.3 to win, got {result.start}"
    )
    assert "sentence" in result.start_sources


def test_snap_end_preference_punctuation_sentence_over_valley_over_shot():
    """For the end edge: terminal-punct sentence (prio 0) > valley (1) > shot (2)."""
    sent_punct = _make_sentence("Done here.", start=18.0, end=20.2)
    result = snap_boundaries(
        10.0, 20.0,
        shots=[20.5],
        sentences=[sent_punct],   # ends with '.' → prio 0
        valleys=[19.8],
    )
    # Terminal-punct sentence end wins
    assert abs(result.end - 20.2) < 1e-6, (
        f"Expected end snapped to 20.2 (punct sentence), got {result.end}"
    )
    assert "sentence" in result.end_sources


def test_snap_leaves_edge_unchanged_when_no_signal_in_window():
    """If no signal is within the search window, the proposed edge is kept unchanged."""
    result = snap_boundaries(
        10.0, 20.0,
        shots=[50.0],          # far away
        sentences=None,
        valleys=None,
        search_window_s=2.0,
    )
    assert abs(result.start - 10.0) < 1e-6, (
        f"Expected start unchanged at 10.0, got {result.start}"
    )
    assert abs(result.end - 20.0) < 1e-6, (
        f"Expected end unchanged at 20.0, got {result.end}"
    )
    # No signal → no sources, and a warning should be issued
    assert result.start_sources == [], f"Expected empty start_sources, got {result.start_sources}"
    assert result.end_sources == [], f"Expected empty end_sources, got {result.end_sources}"
    assert len(result.warnings) >= 2, (
        f"Expected at least 2 warnings for no-signal edges, got {result.warnings}"
    )


# ── snap_boundaries: duration clamping ────────────────────────────────────────

def test_snap_clamps_below_min_duration():
    """When snapping produces a clip shorter than min_clip_duration_s, end is expanded."""
    # Proposed 10.0→12.0 = 2 s; min is 5 s → end should be pushed to 15.0
    result = snap_boundaries(
        10.0, 12.0,
        shots=[],
        sentences=None,
        valleys=None,
        min_clip_duration_s=5.0,
    )
    duration = result.end - result.start
    assert duration >= 5.0, (
        f"Expected duration >= 5s after clamping, got {duration:.2f}s"
    )
    # A warning must mention the clamping
    has_min_warning = any("minimum" in w.lower() for w in result.warnings)
    assert has_min_warning, f"Expected min-duration warning, got warnings={result.warnings}"


def test_snap_clamps_above_max_duration():
    """When snapping produces a clip longer than max_clip_duration_s, end is shrunk."""
    # Proposed 0.0→1000.0 = 1000 s; max is 179 s → end should shrink
    result = snap_boundaries(
        0.0, 1000.0,
        shots=[],
        sentences=None,
        valleys=None,
        max_clip_duration_s=179.0,
    )
    duration = result.end - result.start
    assert duration <= 179.0, (
        f"Expected duration <= 179s after clamping, got {duration:.2f}s"
    )
    has_max_warning = any("maximum" in w.lower() for w in result.warnings)
    assert has_max_warning, f"Expected max-duration warning, got warnings={result.warnings}"


# ── snap_boundaries: source tracking ──────────────────────────────────────────

def test_snap_records_start_sources():
    """start_sources must be a non-empty list when a signal was used."""
    result = snap_boundaries(
        10.0, 30.0,
        shots=[10.3],
        sentences=None,
        valleys=None,
    )
    assert isinstance(result.start_sources, list), "start_sources must be a list"
    assert len(result.start_sources) > 0, (
        f"Expected at least one start source, got {result.start_sources}"
    )


def test_snap_records_end_sources():
    """end_sources must be a non-empty list when a signal was used."""
    result = snap_boundaries(
        10.0, 30.0,
        shots=[29.7],
        sentences=None,
        valleys=None,
    )
    assert isinstance(result.end_sources, list), "end_sources must be a list"
    assert len(result.end_sources) > 0, (
        f"Expected at least one end source, got {result.end_sources}"
    )


def test_snap_adjusted_s_nonzero_when_edges_moved():
    """adjusted_s must be non-zero when at least one edge was snapped."""
    result = snap_boundaries(
        10.0, 20.0,
        shots=[10.8],   # start will snap from 10.0 → 10.8
        sentences=None,
        valleys=None,
    )
    assert result.adjusted_s > 0.0, (
        f"Expected adjusted_s > 0 after snapping, got {result.adjusted_s}"
    )


# ── detect_completion ──────────────────────────────────────────────────────────

def test_completion_period_counts_as_complete():
    """A sentence ending with '.' should be detected as complete."""
    sent = _make_sentence("The event started at noon.")
    is_complete, reasons = detect_completion(sent)
    assert is_complete, (
        f"Expected complete=True for period-terminated sentence, reasons={reasons}"
    )
    assert "terminal_punctuation" in reasons


def test_completion_question_mark_counts_as_complete():
    """A sentence ending with '?' should be detected as complete."""
    sent = _make_sentence("Did they arrive on time?")
    is_complete, reasons = detect_completion(sent)
    assert is_complete, (
        f"Expected complete=True for question-mark sentence, reasons={reasons}"
    )
    assert "terminal_punctuation" in reasons


def test_completion_dangling_but_incomplete():
    """A fragment ending with a dangling connective should be incomplete.

    The text ends without terminal punctuation AND the stripped last word
    is a known dangling marker, failing multiple heuristics.
    """
    # Ends with "but" — no terminal punct, has dangling marker
    sent = _make_sentence("He wanted to leave but")
    is_complete, reasons = detect_completion(sent)
    # terminal_punctuation fails + no_dangling_marker fails → must be incomplete
    assert not is_complete, (
        f"Expected incomplete for trailing connective, reasons={reasons}"
    )


def test_completion_devanagari_purna_viram_complete():
    """A sentence ending with the Devanagari danda '।' is complete."""
    sent = _make_sentence("यह एक वाक्य है।")   # "This is a sentence."
    is_complete, reasons = detect_completion(sent)
    assert is_complete, (
        f"Expected complete=True for Devanagari purna viram, reasons={reasons}"
    )
    assert "terminal_punctuation" in reasons


def test_completion_short_fragment_incomplete():
    """A fragment starting with a dangling pronoun AND ending without punctuation
    is incomplete — at least two heuristics fail.

    The implementation requires ≥2 heuristics to pass for is_complete=True.
    A sentence starting with "he" (unresolved pronoun) that also lacks terminal
    punctuation AND ends with a connective fails at least two heuristics.
    """
    # "He said but" → no terminal punct + dangling marker "but" → 2 heuristics fail
    sent = _make_sentence("He said but")
    is_complete, reasons = detect_completion(sent)
    # terminal_punctuation fails AND no_dangling_marker fails → only ≤2 pass
    assert not is_complete, (
        f"Expected incomplete for pronoun-leading dangling fragment, reasons={reasons}"
    )
