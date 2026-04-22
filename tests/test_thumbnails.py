"""
test_thumbnails.py — Tests for pipeline_core.thumbnails

TDD-style, written against the spec before the Builder ships the module.
When thumbnails.py does not exist, all tests are skipped with "blocked on Builder".

Spec under test:
  @dataclass
  class ThumbnailCandidate:
      path: str             # absolute path to PNG
      kind: str             # 'face_lock' | 'quote_card' | 'punch_frame'
      score: float          # 0.0-1.0
      source_frame_t: float
      meta: dict

  generate_thumbnails(
      video_path, *, title, output_dir,
      target_size=(1080, 1920), candidates=3
  ) -> list[ThumbnailCandidate]

Uses the `valid_short_mp4` fixture (15 s, 1080x1920) from conftest.py.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Import guard — collected but skipped until Builder ships
# ---------------------------------------------------------------------------
try:
    from pipeline_core.thumbnails import generate_thumbnails, ThumbnailCandidate
    _THUMBNAILS_AVAILABLE = True
except ImportError:
    _THUMBNAILS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _THUMBNAILS_AVAILABLE,
    reason="pipeline_core.thumbnails not yet implemented (blocked on Builder)",
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_EXPECTED_KINDS = {"face_lock", "quote_card", "punch_frame"}
_DEFAULT_TARGET_SIZE = (1080, 1920)
_TEST_TITLE = "Breaking: Kaizer AI News Pipeline"


# ---------------------------------------------------------------------------
# Session-scoped output dir fixture (separate from conftest tmp_video_dir
# so thumbnail PNGs don't clutter video fixtures)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def thumb_output_dir(tmp_path_factory):
    """Dedicated temp dir for generated thumbnail PNGs."""
    return str(tmp_path_factory.mktemp("kaizer_thumbs_"))


@pytest.fixture(scope="session")
def default_thumbnails(valid_short_mp4, thumb_output_dir):
    """
    Run generate_thumbnails once with default settings and cache the result.
    Depends on ffmpeg being available (same skip mechanism as conftest).
    """
    return generate_thumbnails(
        valid_short_mp4,
        title=_TEST_TITLE,
        output_dir=thumb_output_dir,
        target_size=_DEFAULT_TARGET_SIZE,
        candidates=3,
    )


# ===========================================================================
# 1. test_generate_returns_three_candidates_by_default
# ===========================================================================

def test_generate_returns_three_candidates_by_default(default_thumbnails):
    """generate_thumbnails with candidates=3 must return exactly 3 items."""
    assert len(default_thumbnails) == 3, (
        f"Expected 3 candidates, got {len(default_thumbnails)}"
    )


# ===========================================================================
# 2. test_each_candidate_has_unique_kind
# ===========================================================================

def test_each_candidate_has_unique_kind(default_thumbnails):
    """The three candidates must cover all three kinds exactly once."""
    kinds = {c.kind for c in default_thumbnails}
    assert kinds == _EXPECTED_KINDS, (
        f"Expected kinds {_EXPECTED_KINDS}, got {kinds}"
    )


# ===========================================================================
# 3. test_candidates_sorted_by_score_descending
# ===========================================================================

def test_candidates_sorted_by_score_descending(default_thumbnails):
    """
    Candidates must be returned in descending score order
    (best thumbnail first).
    """
    scores = [c.score for c in default_thumbnails]
    assert scores == sorted(scores, reverse=True), (
        f"Candidates not sorted by score desc: scores={scores}"
    )


# ===========================================================================
# 4. test_all_candidate_paths_exist_on_disk
# ===========================================================================

def test_all_candidate_paths_exist_on_disk(default_thumbnails):
    """Every candidate.path must point to a file that exists on disk."""
    for candidate in default_thumbnails:
        assert os.path.isfile(candidate.path), (
            f"Candidate path does not exist on disk: {candidate.path!r} "
            f"(kind={candidate.kind!r})"
        )


# ===========================================================================
# 5. test_all_candidates_match_target_size
# ===========================================================================

def test_all_candidates_match_target_size(default_thumbnails):
    """
    Each PNG must have exactly the requested target_size dimensions when
    opened with Pillow.
    """
    from PIL import Image

    for candidate in default_thumbnails:
        with Image.open(candidate.path) as img:
            assert img.size == _DEFAULT_TARGET_SIZE, (
                f"Candidate {candidate.kind!r} has size {img.size}, "
                f"expected {_DEFAULT_TARGET_SIZE}"
            )


# ===========================================================================
# 6. test_quote_card_contains_title_text_ish
# ===========================================================================

def test_quote_card_contains_title_text_ish(default_thumbnails):
    """
    Pixel-level proxy: the central 20x20 patch of the quote_card thumbnail
    must differ from the same patch in the face_lock thumbnail, confirming
    that title text was overlaid on the quote_card.
    """
    from PIL import Image

    quote_card = next(c for c in default_thumbnails if c.kind == "quote_card")
    face_lock = next(c for c in default_thumbnails if c.kind == "face_lock")

    w, h = _DEFAULT_TARGET_SIZE
    cx, cy = w // 2, h // 2
    patch_box = (cx - 10, cy - 10, cx + 10, cy + 10)  # 20x20 box

    with Image.open(quote_card.path) as qc_img:
        qc_patch = qc_img.crop(patch_box).tobytes()

    with Image.open(face_lock.path) as fl_img:
        fl_patch = fl_img.crop(patch_box).tobytes()

    assert qc_patch != fl_patch, (
        "quote_card and face_lock share identical center patches — "
        "title text was likely not rendered onto quote_card"
    )


# ===========================================================================
# 7. test_source_frame_t_within_video_duration
# ===========================================================================

def test_source_frame_t_within_video_duration(default_thumbnails):
    """
    Each candidate's source_frame_t must be a non-negative float within
    [0, video_duration]. We use the known fixture duration of 15 s.
    """
    video_duration = 15.0  # valid_short_mp4 is 15 seconds

    for candidate in default_thumbnails:
        assert isinstance(candidate.source_frame_t, (int, float)), (
            f"source_frame_t must be numeric, got {type(candidate.source_frame_t)}"
        )
        assert 0.0 <= candidate.source_frame_t <= video_duration, (
            f"source_frame_t={candidate.source_frame_t} is outside [0, {video_duration}] "
            f"for kind={candidate.kind!r}"
        )


# ===========================================================================
# 8. test_output_dir_is_created_if_missing
# ===========================================================================

def test_output_dir_is_created_if_missing(valid_short_mp4, tmp_path_factory):
    """
    Passing a non-existent output_dir must not raise — the function must
    create the directory and write PNG files into it.
    """
    # Build a path that definitely doesn't exist yet
    base = str(tmp_path_factory.mktemp("kaizer_thumb_autocreate_"))
    non_existent_dir = os.path.join(base, "auto_created_subdir")
    assert not os.path.exists(non_existent_dir), "Pre-condition: dir must not exist"

    candidates = generate_thumbnails(
        valid_short_mp4,
        title=_TEST_TITLE,
        output_dir=non_existent_dir,
        target_size=_DEFAULT_TARGET_SIZE,
        candidates=3,
    )

    assert os.path.isdir(non_existent_dir), (
        f"output_dir was not created automatically: {non_existent_dir!r}"
    )
    assert len(candidates) > 0, "No candidates returned after auto-creating output_dir"
    for c in candidates:
        assert os.path.isfile(c.path), f"Candidate PNG missing: {c.path!r}"


# ===========================================================================
# 9. test_candidates_parameter_reduces_output
# ===========================================================================

def test_candidates_parameter_reduces_output(valid_short_mp4, tmp_path_factory):
    """
    Calling generate_thumbnails with candidates=1 must return exactly 1
    ThumbnailCandidate (the highest-scoring one).
    """
    out_dir = str(tmp_path_factory.mktemp("kaizer_thumb_single_"))
    result = generate_thumbnails(
        valid_short_mp4,
        title=_TEST_TITLE,
        output_dir=out_dir,
        target_size=_DEFAULT_TARGET_SIZE,
        candidates=1,
    )

    assert len(result) == 1, (
        f"candidates=1 must return exactly 1 item, got {len(result)}"
    )
    assert result[0].kind in _EXPECTED_KINDS, (
        f"Single candidate kind {result[0].kind!r} not in {_EXPECTED_KINDS}"
    )


# ===========================================================================
# 10. test_invalid_video_path_raises
# ===========================================================================

def test_invalid_video_path_raises(tmp_path_factory):
    """
    Passing a path that doesn't exist must raise FileNotFoundError (or a
    subclass / equivalent exception that indicates the file is missing).
    """
    out_dir = str(tmp_path_factory.mktemp("kaizer_thumb_invalid_"))
    nonexistent = "/nonexistent/path/does_not_exist_kaizer_thumb_test.mp4"

    with pytest.raises((FileNotFoundError, OSError, ValueError, RuntimeError)):
        generate_thumbnails(
            nonexistent,
            title=_TEST_TITLE,
            output_dir=out_dir,
        )


# ===========================================================================
# 11. test_candidate_score_in_valid_range  (bonus — still counted)
# ===========================================================================

def test_candidate_score_in_valid_range(default_thumbnails):
    """Every candidate.score must be a float in [0.0, 1.0]."""
    for candidate in default_thumbnails:
        assert isinstance(candidate.score, (int, float)), (
            f"score must be numeric, got {type(candidate.score)}"
        )
        assert 0.0 <= candidate.score <= 1.0, (
            f"score {candidate.score} out of [0.0, 1.0] for kind={candidate.kind!r}"
        )


# ===========================================================================
# 12. test_candidate_meta_is_dict  (bonus — still counted)
# ===========================================================================

def test_candidate_meta_is_dict(default_thumbnails):
    """Every candidate.meta must be a dict (may be empty)."""
    for candidate in default_thumbnails:
        assert isinstance(candidate.meta, dict), (
            f"meta must be dict, got {type(candidate.meta)} for kind={candidate.kind!r}"
        )
