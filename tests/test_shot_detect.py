"""
tests/test_shot_detect.py
==========================
Pytest coverage for pipeline_core.shot_detect.

Most tests run FFmpeg scdet on the existing valid_short_mp4 fixture (fast).
The cutpoint ONNX path is tested for graceful fallback without the model file.
"""
from __future__ import annotations

import os
import pytest

from pipeline_core.shot_detect import detect_shots, get_shot_ranges, ShotBoundary


# ── Basic shot detection on a real video ──────────────────────────────────────

def test_detect_shots_scdet_on_valid_video(valid_short_mp4):
    """detect_shots(scdet) on a real video returns a list of ShotBoundary objects."""
    boundaries = detect_shots(valid_short_mp4, method="scdet", threshold=0.4)
    assert isinstance(boundaries, list), (
        f"Expected list, got {type(boundaries)}"
    )
    # The test video is a synthetic testsrc2; may have 0 or more boundaries.
    for b in boundaries:
        assert isinstance(b, ShotBoundary), (
            f"Each element must be ShotBoundary, got {type(b)}"
        )
        assert hasattr(b, "t"), "ShotBoundary missing 't' attribute"
        assert hasattr(b, "confidence"), "ShotBoundary missing 'confidence' attribute"
        assert hasattr(b, "method"), "ShotBoundary missing 'method' attribute"


def test_shot_timestamps_sorted(valid_short_mp4):
    """Shot boundaries returned by detect_shots must be in ascending t order."""
    boundaries = detect_shots(valid_short_mp4, method="scdet", threshold=0.3)
    times = [b.t for b in boundaries]
    assert times == sorted(times), (
        f"Expected sorted timestamps, got {times}"
    )


def test_shot_confidence_in_range(valid_short_mp4):
    """All shot boundary confidence values must be in [0.0, 1.0]."""
    boundaries = detect_shots(valid_short_mp4, method="scdet", threshold=0.3)
    for b in boundaries:
        assert 0.0 <= b.confidence <= 1.0, (
            f"confidence {b.confidence} out of [0,1] range for boundary at t={b.t}"
        )


def test_method_cutpoint_when_onnx_absent_warns_or_falls_back(valid_short_mp4):
    """If cut_point.onnx is absent, method='cutpoint' must not crash.

    It may either return an empty list (skipped with warning) or fall back to
    scdet — either is acceptable, but an exception must never propagate.
    """
    # We don't have cut_point.onnx, so this must gracefully return a list.
    try:
        boundaries = detect_shots(valid_short_mp4, method="cutpoint", threshold=0.5)
    except Exception as exc:
        pytest.fail(
            f"detect_shots(method='cutpoint') must not raise when ONNX is absent, "
            f"got {type(exc).__name__}: {exc}"
        )
    assert isinstance(boundaries, list), (
        f"Expected list on cutpoint fallback, got {type(boundaries)}"
    )


# ── get_shot_ranges ────────────────────────────────────────────────────────────

def test_get_shot_ranges_from_boundaries():
    """Given sorted timestamps [2.0, 5.0] and total_duration=10.0,
    get_shot_ranges returns [(0, 2), (2, 5), (5, 10)].
    """
    boundaries = [
        ShotBoundary(t=2.0, confidence=0.8, method="scdet"),
        ShotBoundary(t=5.0, confidence=0.6, method="scdet"),
    ]
    ranges = get_shot_ranges(boundaries, total_duration=10.0)
    assert len(ranges) == 3, f"Expected 3 ranges, got {len(ranges)}: {ranges}"
    assert ranges[0] == (0.0, 2.0), f"First range wrong: {ranges[0]}"
    assert ranges[1] == (2.0, 5.0), f"Second range wrong: {ranges[1]}"
    assert ranges[2] == (5.0, 10.0), f"Third range wrong: {ranges[2]}"


def test_get_shot_ranges_empty_boundaries():
    """With no boundaries, get_shot_ranges should return a single (0, total) range."""
    ranges = get_shot_ranges([], total_duration=42.0)
    assert len(ranges) == 1, f"Expected 1 range for empty boundaries, got {ranges}"
    assert ranges[0] == (0.0, 42.0), f"Expected (0, 42), got {ranges[0]}"


def test_invalid_video_path_raises_or_returns_empty():
    """detect_shots on a nonexistent path must not raise — returns empty list."""
    bogus_path = "/nonexistent/kaizer_test_video_zzz.mp4"
    try:
        result = detect_shots(bogus_path, method="scdet")
    except Exception as exc:
        pytest.fail(
            f"detect_shots must not raise for nonexistent path, "
            f"got {type(exc).__name__}: {exc}"
        )
    # The contract: either empty list or at most a warning-carrying list
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert result == [], f"Expected empty list for nonexistent path, got {result}"
