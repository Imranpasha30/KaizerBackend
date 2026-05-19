"""Unit tests for Step 5.4 clean_transcript reconstruction helpers.

These are PURE functions -- no SDK mocks needed. The Step 5.2 tests
already cover the LLM call path; this file isolates the deterministic
post-processing layer that runs AFTER Gemini returns and BEFORE the
output reaches the orchestrator.

Invariants under test:

  1. ``words[clean_idx] is original_words[source_word_map[clean_idx]]``
     (zero coordinate drift -- Word objects passed through verbatim).
  2. ``clip_boundaries[cut.index]`` always points to non-skipped clean
     indices, with boundaries snapped inward if a cut's edge lands on
     a skipped word.
  3. Validation: overlapping skips, out-of-bounds indices, and empty
     retake_audit raise ValueError -- never silent fallbacks.
  4. A cut fully covered by skipped_segments is dropped (warning
     logged) but the StageTwoOutput.full_video_cuts list still
     contains it for telemetry.
"""

from __future__ import annotations

import logging

import pytest

from pipeline_v2.models import (
    CleanTranscript,
    FullVideoCut,
    SkippedSegment,
    Stage1Output,
    Stage2Output,
    StageTwoOutput,
    Word,
    WordLevelTranscript,
)
from pipeline_v2.stages.stage_2_continuity import (
    assemble_stage_two_output,
    build_clean_transcript,
)


# ====================================================================== #
# Fixtures                                                                #
# ====================================================================== #


def _w(text: str, s: float, e: float) -> Word:
    return Word(w=text, s=s, e=e)


def _make_words(n: int) -> list[Word]:
    """n sequential words 'w0'..'w{n-1}' each 0.5s long with 0.05s gap."""
    return [
        _w(f"w{i}", round(i * 0.55, 3), round(i * 0.55 + 0.5, 3))
        for i in range(n)
    ]


def _cut(index: int, start: int, end: int, importance: int = 5) -> FullVideoCut:
    """FullVideoCut with start/end IDX -> derive start/end sec from helper words."""
    words = _make_words(end + 1)
    return FullVideoCut(
        index=index,
        start_word_idx=start,
        end_word_idx=end,
        start_sec=words[start].s,
        end_sec=words[end].e,
        importance=importance,
    )


def _skip(start: int, end: int, category: str = "hesitation") -> SkippedSegment:
    words = _make_words(end + 1)
    return SkippedSegment(
        start_word_idx=start,
        end_word_idx=end,
        start_sec=words[start].s,
        end_sec=words[end].e,
        category=category,
        reason="test",
    )


# ====================================================================== #
# build_clean_transcript: happy path + edge cases                         #
# ====================================================================== #


class TestBuildCleanTranscriptHappyPath:
    def test_no_skips_passes_through_words_unchanged(self):
        words = _make_words(10)
        out = build_clean_transcript(words, [], [_cut(0, 0, 9)])
        assert len(out.words) == 10
        # source_word_map is identity 0..9
        assert out.source_word_map == list(range(10))
        # Words are the same objects (verbatim pass-through)
        for i in range(10):
            assert out.words[i] is words[i]
        # Single cut spans 0-9 in both spaces
        assert out.clip_boundaries == {0: (0, 9)}

    def test_single_middle_skip_removes_words(self):
        words = _make_words(10)
        # Skip indices 3-5; cut covers 0-9
        out = build_clean_transcript(
            words, [_skip(3, 5)], [_cut(0, 0, 9)],
        )
        assert len(out.words) == 7
        # original indices remaining: 0,1,2,6,7,8,9
        assert out.source_word_map == [0, 1, 2, 6, 7, 8, 9]
        # The clip still spans the full kept range (0->6 in clean space)
        assert out.clip_boundaries == {0: (0, 6)}

    def test_multiple_disjoint_skips(self):
        words = _make_words(20)
        skips = [_skip(2, 4), _skip(10, 12), _skip(17, 18)]
        cuts = [_cut(0, 0, 19)]
        out = build_clean_transcript(words, skips, cuts)
        # 20 - 3 - 3 - 2 = 12 surviving
        assert len(out.words) == 12
        assert out.source_word_map == [
            0, 1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 19,
        ]
        # Cut clip_boundaries should snap to first/last surviving
        assert out.clip_boundaries == {0: (0, 11)}

    def test_skip_at_start(self):
        words = _make_words(10)
        out = build_clean_transcript(
            words, [_skip(0, 2)], [_cut(0, 0, 9)],
        )
        # 0-2 dropped; 3-9 kept = 7 clean words
        assert out.source_word_map == [3, 4, 5, 6, 7, 8, 9]
        # Cut started at orig=0 (skipped); snaps to first non-skipped
        # at orig=3 -> clean_idx=0
        assert out.clip_boundaries == {0: (0, 6)}

    def test_skip_at_end(self):
        words = _make_words(10)
        out = build_clean_transcript(
            words, [_skip(7, 9)], [_cut(0, 0, 9)],
        )
        assert out.source_word_map == [0, 1, 2, 3, 4, 5, 6]
        # end_word_idx=9 is skipped; snaps inward to orig=6 -> clean_idx=6
        assert out.clip_boundaries == {0: (0, 6)}

    def test_cut_boundary_lands_on_skipped_word_snaps_inward(self):
        # Cut spans 3-12; skip covers 3-5 (cut start). first_clean
        # should snap forward to orig=6 -> clean=3 (since 0,1,2 are
        # also kept before the skip).
        words = _make_words(15)
        out = build_clean_transcript(
            words, [_skip(3, 5)], [_cut(0, 3, 12)],
        )
        # source_word_map: [0,1,2,6,7,8,9,10,11,12,13,14]
        assert out.source_word_map == [
            0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14,
        ]
        # Cut starts at orig=3 (skipped) -> snap to orig=6 -> clean=3
        # Cut ends at orig=12 -> clean=9
        assert out.clip_boundaries == {0: (3, 9)}

    def test_cut_boundary_skipped_at_both_ends(self):
        # Cut spans 5-10; skips at 5-6 and 10
        words = _make_words(15)
        skips = [_skip(5, 6), _skip(10, 10)]
        out = build_clean_transcript(words, skips, [_cut(0, 5, 10)])
        # First non-skipped >= 5: orig=7
        # Last non-skipped <= 10: orig=9
        # source_word_map = [0,1,2,3,4,7,8,9,11,12,13,14]
        # orig 7 -> clean 5; orig 9 -> clean 7
        assert out.clip_boundaries == {0: (5, 7)}

    def test_multiple_cuts(self):
        # Two distinct cuts with a skip between them
        words = _make_words(20)
        skips = [_skip(8, 10)]
        cuts = [_cut(0, 0, 7), _cut(1, 11, 19)]
        out = build_clean_transcript(words, skips, cuts)
        # source_word_map = [0..7, 11..19]
        assert out.source_word_map == [0, 1, 2, 3, 4, 5, 6, 7,
                                       11, 12, 13, 14, 15, 16, 17, 18, 19]
        assert out.clip_boundaries == {0: (0, 7), 1: (8, 16)}


# ====================================================================== #
# build_clean_transcript: degenerate cases                                #
# ====================================================================== #


class TestBuildCleanTranscriptEdgeCases:
    def test_cut_entirely_inside_skipped_span_is_dropped(self, caplog):
        # Cut 0-4 entirely covered by skip 0-9
        words = _make_words(10)
        with caplog.at_level(logging.WARNING, logger="pipeline_v2.stage_2"):
            out = build_clean_transcript(
                words, [_skip(0, 9)], [_cut(0, 0, 4)],
            )
        # All 10 words skipped -> no clean words
        assert out.words == []
        assert out.source_word_map == []
        # Cut dropped from clip_boundaries
        assert out.clip_boundaries == {}
        # Warning surfaced (not silent)
        assert any(
            "entirely covered" in rec.message
            for rec in caplog.records
        )

    def test_fully_skipped_word_array_logs_empty_warning(self, caplog):
        # Q1 (Step 5.5): 100%-skipped recording is legitimate but must
        # surface a warning so operators see the signal in logs --
        # downstream stages still get an empty CleanTranscript.
        words = _make_words(8)
        with caplog.at_level(logging.WARNING, logger="pipeline_v2.stage_2"):
            out = build_clean_transcript(
                words, [_skip(0, 7)], [],
            )
        assert out.words == []
        assert out.source_word_map == []
        assert any(
            "empty word array" in rec.message
            for rec in caplog.records
        )

    def test_empty_cuts_empty_skips(self):
        words = _make_words(5)
        out = build_clean_transcript(words, [], [])
        assert len(out.words) == 5
        assert out.source_word_map == [0, 1, 2, 3, 4]
        assert out.clip_boundaries == {}

    def test_word_object_identity_preserved(self):
        # Stage 1 may pass Word objects that downstream stages cache
        # by id() -- the helper must NOT clone them.
        words = _make_words(5)
        out = build_clean_transcript(words, [_skip(2, 2)], [_cut(0, 0, 4)])
        # words[0,1] -> clean[0,1]; words[3,4] -> clean[2,3]
        assert out.words[0] is words[0]
        assert out.words[1] is words[1]
        assert out.words[2] is words[3]
        assert out.words[3] is words[4]

    def test_returns_valid_clean_transcript_instance(self):
        words = _make_words(5)
        out = build_clean_transcript(words, [], [])
        assert isinstance(out, CleanTranscript)


# ====================================================================== #
# build_clean_transcript: validation errors                               #
# ====================================================================== #


class TestBuildCleanTranscriptValidation:
    def test_overlapping_skipped_segments_raise(self):
        words = _make_words(10)
        skips = [_skip(2, 5), _skip(4, 7)]
        with pytest.raises(ValueError, match="Overlapping"):
            build_clean_transcript(words, skips, [])

    def test_skip_out_of_bounds_high_raises(self):
        words = _make_words(5)
        skips = [_skip(2, 10)]
        with pytest.raises(ValueError, match="out of bounds"):
            build_clean_transcript(words, skips, [])

    def test_skip_out_of_bounds_negative_raises(self):
        # Pydantic accepts negative ints unless we enforce; the helper
        # rejects via its own bounds check.
        words = _make_words(5)
        # Build SkippedSegment with negative idx directly (bypassing
        # convenience helper which derives sec from idx).
        bad = SkippedSegment(
            start_word_idx=-1, end_word_idx=2,
            start_sec=0.0, end_sec=1.0,
            category="hesitation", reason="x",
        )
        with pytest.raises(ValueError, match="out of bounds"):
            build_clean_transcript(words, [bad], [])

    def test_cut_out_of_bounds_raises(self):
        words = _make_words(5)
        cut = FullVideoCut(
            index=0, start_word_idx=0, end_word_idx=99,
            start_sec=0.0, end_sec=99.0, importance=5,
        )
        with pytest.raises(ValueError, match="FullVideoCut out of bounds"):
            build_clean_transcript(words, [], [cut])


# ====================================================================== #
# assemble_stage_two_output                                               #
# ====================================================================== #


def _make_stage1(n: int) -> Stage1Output:
    words = _make_words(n)
    return Stage1Output(
        transcript=WordLevelTranscript(
            words=words,
            duration_sec=words[-1].e if words else 0.0,
            detected_languages=["en"],
            provider="test",
        ),
        stt_provider="test",
        stt_audio_duration_sec=words[-1].e if words else 0.0,
        stt_wall_seconds=1.0,
        stt_cost_usd=0.001,
        stt_word_count=n,
        stt_language_detected="en",
        stt_request_id="test-req-0",
    )


class TestAssembleStageTwoOutput:
    def test_happy_path_returns_stage_two_output(self):
        stage1 = _make_stage1(10)
        decisions = Stage2Output(
            full_video_cuts=[_cut(0, 0, 9, importance=7)],
            skipped_segments=[_skip(3, 4)],
            retake_audit="Skipped 1 hesitation at ~1.65-2.75s.",
        )
        out = assemble_stage_two_output(stage1, decisions)
        assert isinstance(out, StageTwoOutput)
        assert out.retake_audit.startswith("Skipped 1 hesitation")
        # full_video_cuts and skipped_segments pass through unchanged
        assert out.full_video_cuts == decisions.full_video_cuts
        assert out.skipped_segments == decisions.skipped_segments
        # clean_transcript reconstructed
        assert len(out.clean_transcript.words) == 8
        assert out.clean_transcript.clip_boundaries == {0: (0, 7)}

    def test_empty_retake_audit_raises(self):
        stage1 = _make_stage1(5)
        decisions = Stage2Output(
            full_video_cuts=[_cut(0, 0, 4)],
            skipped_segments=[],
            retake_audit="",
        )
        with pytest.raises(ValueError, match="empty/missing retake_audit"):
            assemble_stage_two_output(stage1, decisions)

    def test_none_retake_audit_raises(self):
        stage1 = _make_stage1(5)
        decisions = Stage2Output(
            full_video_cuts=[_cut(0, 0, 4)],
            skipped_segments=[],
            retake_audit=None,
        )
        with pytest.raises(ValueError, match="empty/missing retake_audit"):
            assemble_stage_two_output(stage1, decisions)

    def test_whitespace_only_retake_audit_raises(self):
        stage1 = _make_stage1(5)
        decisions = Stage2Output(
            full_video_cuts=[_cut(0, 0, 4)],
            skipped_segments=[],
            retake_audit="   \n  \t  ",
        )
        with pytest.raises(ValueError, match="empty/missing retake_audit"):
            assemble_stage_two_output(stage1, decisions)


# ====================================================================== #
# Round-trip determinism                                                  #
# ====================================================================== #


class TestRoundTripDeterminism:
    def test_source_word_map_recovers_original_word(self):
        # The critical invariant: anyone holding a clean_idx can recover
        # the original Word (and therefore the original timestamp) via
        # source_word_map. No timestamp drift, no off-by-one.
        words = _make_words(30)
        skips = [_skip(3, 5), _skip(15, 17), _skip(25, 28)]
        out = build_clean_transcript(words, skips, [_cut(0, 0, 29)])
        for clean_idx, w in enumerate(out.words):
            orig_idx = out.source_word_map[clean_idx]
            assert words[orig_idx] is w
            # Timestamps recoverable bit-for-bit
            assert words[orig_idx].s == w.s
            assert words[orig_idx].e == w.e

    def test_same_inputs_produce_same_output_twice(self):
        words = _make_words(20)
        skips = [_skip(5, 7)]
        cuts = [_cut(0, 0, 19)]
        a = build_clean_transcript(words, skips, cuts)
        b = build_clean_transcript(words, skips, cuts)
        assert a.source_word_map == b.source_word_map
        assert a.clip_boundaries == b.clip_boundaries
        # Word equality is by value (Pydantic) -- but we want object
        # identity for the surviving words too.
        for x, y in zip(a.words, b.words):
            assert x is y
