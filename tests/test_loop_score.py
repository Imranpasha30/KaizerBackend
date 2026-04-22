"""
tests/test_loop_score.py
=========================
pytest coverage for pipeline_core/loop_score.py

Fast tests mock cv2.VideoCapture, _phash_64, _audio_xcorr, _motion_continuity,
_extract_end_frames, and _extract_audio_tails so they never hit real disk I/O
or decode any frames.

Slow test (gated by @pytest.mark.slow) uses the real valid_short_mp4 fixture.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------
from pipeline_core.loop_score import (
    LoopScore,
    score_loop,
    _PHASH_BAD_DISTANCE,
    _AUDIO_XCORR_BAD,
    _MOTION_BAD,
)


# ---------------------------------------------------------------------------
# Helper: minimal blank frame
# ---------------------------------------------------------------------------

def _blank_frame(h: int = 10, w: int = 10) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _blank_frames(n: int = 6) -> list[np.ndarray]:
    return [_blank_frame() for _ in range(n)]


# ===========================================================================
# 1. test_loopscore_dataclass_shape
# ===========================================================================

def test_loopscore_dataclass_shape():
    """LoopScore must be a dataclass with the five documented fields + correct types."""
    assert dataclasses.is_dataclass(LoopScore), "LoopScore must be a dataclass"

    field_names = {f.name for f in dataclasses.fields(LoopScore)}
    expected = {"overall", "visual_phash_distance", "audio_xcorr", "motion_continuity", "suggestions"}
    assert expected <= field_names, (
        f"Missing fields: {expected - field_names}. Got: {field_names}"
    )

    # Instantiate and check runtime types
    ls = LoopScore(overall=50.0, visual_phash_distance=10, audio_xcorr=0.5,
                   motion_continuity=0.8, suggestions=["fix something"])

    assert isinstance(ls.overall, float), f"overall must be float, got {type(ls.overall)}"
    assert isinstance(ls.visual_phash_distance, int), (
        f"visual_phash_distance must be int, got {type(ls.visual_phash_distance)}"
    )
    assert isinstance(ls.audio_xcorr, float), f"audio_xcorr must be float, got {type(ls.audio_xcorr)}"
    assert isinstance(ls.motion_continuity, float), (
        f"motion_continuity must be float, got {type(ls.motion_continuity)}"
    )
    assert isinstance(ls.suggestions, list), f"suggestions must be list, got {type(ls.suggestions)}"


# ===========================================================================
# 2. test_missing_video_returns_zero_score_with_suggestion
# ===========================================================================

def test_missing_video_returns_zero_score_with_suggestion():
    """A path that does not exist must return overall=0.0 and a suggestion about
    'does not exist' without raising any exception."""
    result = score_loop("/nonexistent/path/kaizer_test_does_not_exist_xyz.mp4")

    assert isinstance(result, LoopScore), f"Must return LoopScore, got {type(result)}"
    assert result.overall == 0.0, f"Expected overall=0.0 for missing path, got {result.overall}"
    assert result.visual_phash_distance == 64, (
        f"Expected phash_distance=64 (worst) for missing path, got {result.visual_phash_distance}"
    )

    assert result.suggestions, "Expected at least one suggestion for missing path"
    combined = " ".join(result.suggestions).lower()
    assert "does not exist" in combined, (
        f"Suggestion must mention 'does not exist'. Got: {result.suggestions}"
    )


# ===========================================================================
# 3. test_perfect_visual_seam_drives_high_visual_component
# ===========================================================================

def test_perfect_visual_seam_drives_high_visual_component(mocker, tmp_path):
    """When _phash_64 returns the same int for every frame → phash_distance=0
    → visual_component=100.  With audio=0 and motion=0 the overall should be
    exactly 0.5*100 + 0.3*0 + 0.2*0 = 50.0 (if motion is also perfect we get 80)."""

    # Create a minimal placeholder file so os.path.exists passes
    fake_mp4 = str(tmp_path / "fake.mp4")
    open(fake_mp4, "wb").close()

    # Six identical blank frames for each end
    frames = _blank_frames(6)

    mocker.patch(
        "pipeline_core.loop_score._extract_end_frames",
        return_value=(frames, frames),
    )
    # All frames identical → hamming distance = 0
    mocker.patch("pipeline_core.loop_score._phash_64", return_value=0)
    # Kill audio path
    mocker.patch("pipeline_core.loop_score._extract_audio_tails", return_value=(None, None))
    # motion_continuity = 0 → motion_component = 0
    mocker.patch("pipeline_core.loop_score._motion_continuity", return_value=0.0)

    result = score_loop(fake_mp4)

    # visual = max(0, 1 - 0/30)*100 = 100
    # overall = 0.5*100 + 0.3*0 + 0.2*0 = 50
    assert result.visual_phash_distance == 0, (
        f"Identical hashes must yield phash_distance=0, got {result.visual_phash_distance}"
    )
    assert abs(result.overall - 50.0) < 1e-6, (
        f"Expected overall=50.0 (pure visual=100, motion=0, audio=0), got {result.overall}"
    )


# ===========================================================================
# 4. test_bad_visual_seam_fires_suggestion
# ===========================================================================

def test_bad_visual_seam_fires_suggestion(mocker, tmp_path):
    """When _phash_64 returns hashes with Hamming distance > _PHASH_BAD_DISTANCE,
    the suggestion list must mention 'Freeze or crossfade'."""

    fake_mp4 = str(tmp_path / "fake.mp4")
    open(fake_mp4, "wb").close()

    frames = _blank_frames(6)
    mocker.patch(
        "pipeline_core.loop_score._extract_end_frames",
        return_value=(frames, frames),
    )
    # Alternating 0/full-mask so every pair has Hamming distance 64 (>> _PHASH_BAD_DISTANCE=20)
    call_count = {"n": 0}
    def _alt_hash(img):
        call_count["n"] += 1
        return 0 if call_count["n"] % 2 == 1 else (2**64 - 1)

    mocker.patch("pipeline_core.loop_score._phash_64", side_effect=_alt_hash)
    mocker.patch("pipeline_core.loop_score._extract_audio_tails", return_value=(None, None))
    mocker.patch("pipeline_core.loop_score._motion_continuity", return_value=0.8)

    result = score_loop(fake_mp4)

    assert result.visual_phash_distance > _PHASH_BAD_DISTANCE, (
        f"Expected phash_distance > {_PHASH_BAD_DISTANCE}, got {result.visual_phash_distance}"
    )
    combined = " ".join(result.suggestions)
    assert "Freeze or crossfade" in combined, (
        f"Expected 'Freeze or crossfade' suggestion for bad visual seam. Got: {result.suggestions}"
    )


# ===========================================================================
# 5. test_low_audio_xcorr_fires_suggestion
# ===========================================================================

def test_low_audio_xcorr_fires_suggestion(mocker, tmp_path):
    """When _audio_xcorr returns 0.1 (< _AUDIO_XCORR_BAD=0.3), the suggestion
    must mention 'Trim audio tail'."""

    fake_mp4 = str(tmp_path / "fake.mp4")
    open(fake_mp4, "wb").close()

    frames = _blank_frames(6)
    mocker.patch(
        "pipeline_core.loop_score._extract_end_frames",
        return_value=(frames, frames),
    )
    mocker.patch("pipeline_core.loop_score._phash_64", return_value=0)

    # Inject real-looking audio arrays so _audio_xcorr is actually called
    dummy_audio = np.zeros(8000, dtype=np.float32)
    mocker.patch(
        "pipeline_core.loop_score._extract_audio_tails",
        return_value=(dummy_audio, dummy_audio),
    )
    # Override the xcorr computation to return a bad value
    mocker.patch("pipeline_core.loop_score._audio_xcorr", return_value=0.1)
    mocker.patch("pipeline_core.loop_score._motion_continuity", return_value=0.8)

    result = score_loop(fake_mp4)

    assert result.audio_xcorr < _AUDIO_XCORR_BAD, (
        f"Expected audio_xcorr < {_AUDIO_XCORR_BAD}, got {result.audio_xcorr}"
    )
    combined = " ".join(result.suggestions)
    assert "Trim audio tail" in combined, (
        f"Expected 'Trim audio tail' suggestion. Got: {result.suggestions}"
    )


# ===========================================================================
# 6. test_low_motion_fires_suggestion
# ===========================================================================

def test_low_motion_fires_suggestion(mocker, tmp_path):
    """When _motion_continuity returns 0.2 (< _MOTION_BAD=0.4), the suggestion
    must mention 'crossfade'."""

    fake_mp4 = str(tmp_path / "fake.mp4")
    open(fake_mp4, "wb").close()

    frames = _blank_frames(6)
    mocker.patch(
        "pipeline_core.loop_score._extract_end_frames",
        return_value=(frames, frames),
    )
    mocker.patch("pipeline_core.loop_score._phash_64", return_value=0)
    mocker.patch("pipeline_core.loop_score._extract_audio_tails", return_value=(None, None))
    mocker.patch("pipeline_core.loop_score._motion_continuity", return_value=0.2)

    result = score_loop(fake_mp4)

    assert result.motion_continuity < _MOTION_BAD, (
        f"Expected motion_continuity < {_MOTION_BAD}, got {result.motion_continuity}"
    )
    combined = " ".join(result.suggestions)
    assert "crossfade" in combined.lower(), (
        f"Expected 'crossfade' in low-motion suggestion. Got: {result.suggestions}"
    )


# ===========================================================================
# 7. test_overall_in_0_100_range
# ===========================================================================

def test_overall_in_0_100_range(mocker, tmp_path):
    """overall must be clipped to [0, 100] in both degenerate cases."""

    # ── Case A: all zeros → overall = 0 ──────────────────────────────────────
    fake_mp4 = str(tmp_path / "fake_zeros.mp4")
    open(fake_mp4, "wb").close()

    mocker.patch(
        "pipeline_core.loop_score._extract_end_frames",
        return_value=(_blank_frames(), _blank_frames()),
    )
    # phash_distance=64 → visual_component = max(0, 1-64/30)*100 = 0 (negative clamped)
    mocker.patch("pipeline_core.loop_score._phash_64", return_value=0)
    # Force distance=64 by patching _hamming indirectly: give every hash pair distance 64
    # We can force phash distance explicitly by having first/last hashes differ maximally
    # Simpler: patch _extract_end_frames to return empty lists (worst case path in module)
    mocker.patch(
        "pipeline_core.loop_score._extract_end_frames",
        return_value=([], []),
    )
    mocker.patch("pipeline_core.loop_score._extract_audio_tails", return_value=(None, None))

    result_zero = score_loop(fake_mp4)
    assert 0.0 <= result_zero.overall <= 100.0, (
        f"overall must be in [0,100], got {result_zero.overall}"
    )
    assert result_zero.overall == 0.0, (
        f"All-zeros path (empty frames, no audio) must produce overall=0.0, got {result_zero.overall}"
    )

    # ── Case B: perfect seam → overall near 100 ───────────────────────────────
    fake_mp4_b = str(tmp_path / "fake_perfect.mp4")
    open(fake_mp4_b, "wb").close()

    mocker.patch(
        "pipeline_core.loop_score._extract_end_frames",
        return_value=(_blank_frames(), _blank_frames()),
    )
    mocker.patch("pipeline_core.loop_score._phash_64", return_value=0)    # distance=0
    dummy_audio = np.ones(8000, dtype=np.float32)
    mocker.patch(
        "pipeline_core.loop_score._extract_audio_tails",
        return_value=(dummy_audio, dummy_audio),
    )
    mocker.patch("pipeline_core.loop_score._audio_xcorr", return_value=1.0)
    mocker.patch("pipeline_core.loop_score._motion_continuity", return_value=1.0)

    result_perfect = score_loop(fake_mp4_b)
    assert result_perfect.overall == 100.0, (
        f"Perfect seam (visual=100, motion=100, audio=100) must yield overall=100.0, "
        f"got {result_perfect.overall}"
    )


# ===========================================================================
# 8. test_composite_formula_weights
# ===========================================================================

def test_composite_formula_weights(mocker, tmp_path):
    """Composite: 0.5*visual + 0.3*motion + 0.2*audio.
    With visual=100, motion=0, audio=0 → overall must be exactly 50.0."""

    fake_mp4 = str(tmp_path / "fake_weights.mp4")
    open(fake_mp4, "wb").close()

    frames = _blank_frames(6)
    mocker.patch(
        "pipeline_core.loop_score._extract_end_frames",
        return_value=(frames, frames),
    )
    # Same hash for all → distance = 0 → visual_component = 100
    mocker.patch("pipeline_core.loop_score._phash_64", return_value=0)
    # No audio → audio_component = 0
    mocker.patch("pipeline_core.loop_score._extract_audio_tails", return_value=(None, None))
    # motion = 0 → motion_component = 0
    mocker.patch("pipeline_core.loop_score._motion_continuity", return_value=0.0)

    result = score_loop(fake_mp4)

    # 0.5*100 + 0.3*0 + 0.2*0 = 50.0
    expected = 50.0
    assert abs(result.overall - expected) < 1e-6, (
        f"Weight test failed: visual=100, motion=0, audio=0 → expected overall=50.0, "
        f"got {result.overall}"
    )


# ===========================================================================
# 9. test_audio_xcorr_value_range
# ===========================================================================

def test_audio_xcorr_value_range(mocker, tmp_path):
    """audio_xcorr returned in LoopScore must be in [0, 1]."""

    fake_mp4 = str(tmp_path / "fake_xcorr.mp4")
    open(fake_mp4, "wb").close()

    frames = _blank_frames(6)
    mocker.patch(
        "pipeline_core.loop_score._extract_end_frames",
        return_value=(frames, frames),
    )
    mocker.patch("pipeline_core.loop_score._phash_64", return_value=0)

    # Inject a controlled xcorr value
    dummy_audio = np.random.rand(8000).astype(np.float32)
    mocker.patch(
        "pipeline_core.loop_score._extract_audio_tails",
        return_value=(dummy_audio, dummy_audio),
    )
    mocker.patch("pipeline_core.loop_score._audio_xcorr", return_value=0.65)
    mocker.patch("pipeline_core.loop_score._motion_continuity", return_value=0.5)

    result = score_loop(fake_mp4)

    assert 0.0 <= result.audio_xcorr <= 1.0, (
        f"audio_xcorr must be in [0, 1], got {result.audio_xcorr}"
    )
    assert result.audio_xcorr == pytest.approx(0.65, abs=1e-6), (
        f"audio_xcorr must reflect the mocked value 0.65, got {result.audio_xcorr}"
    )


# ===========================================================================
# 10. test_motion_continuity_value_range
# ===========================================================================

def test_motion_continuity_value_range(mocker, tmp_path):
    """motion_continuity returned in LoopScore must be in [0, 1]."""

    fake_mp4 = str(tmp_path / "fake_motion.mp4")
    open(fake_mp4, "wb").close()

    frames = _blank_frames(6)
    mocker.patch(
        "pipeline_core.loop_score._extract_end_frames",
        return_value=(frames, frames),
    )
    mocker.patch("pipeline_core.loop_score._phash_64", return_value=0)
    mocker.patch("pipeline_core.loop_score._extract_audio_tails", return_value=(None, None))
    mocker.patch("pipeline_core.loop_score._motion_continuity", return_value=0.75)

    result = score_loop(fake_mp4)

    assert 0.0 <= result.motion_continuity <= 1.0, (
        f"motion_continuity must be in [0, 1], got {result.motion_continuity}"
    )
    assert result.motion_continuity == pytest.approx(0.75, abs=1e-6), (
        f"motion_continuity must reflect the mocked value 0.75, got {result.motion_continuity}"
    )


# ===========================================================================
# 11. Slow — real file integration
# ===========================================================================

@pytest.mark.slow
def test_real_loop_score_on_valid_short_mp4(valid_short_mp4):
    """Run score_loop on the real 15s fixture and verify the LoopScore shape."""
    result = score_loop(valid_short_mp4)

    assert isinstance(result, LoopScore), (
        f"score_loop must return LoopScore, got {type(result)}"
    )
    assert isinstance(result.overall, float), "overall must be float"
    assert 0.0 <= result.overall <= 100.0, (
        f"overall must be in [0, 100], got {result.overall}"
    )
    assert isinstance(result.visual_phash_distance, int), "visual_phash_distance must be int"
    assert 0 <= result.visual_phash_distance <= 64, (
        f"visual_phash_distance must be in [0, 64], got {result.visual_phash_distance}"
    )
    assert 0.0 <= result.audio_xcorr <= 1.0, (
        f"audio_xcorr must be in [0, 1], got {result.audio_xcorr}"
    )
    assert 0.0 <= result.motion_continuity <= 1.0, (
        f"motion_continuity must be in [0, 1], got {result.motion_continuity}"
    )
    assert isinstance(result.suggestions, list), "suggestions must be a list"
