"""Item 108 -- crossfade-capable bulletin stitcher tests.

Covers the spec from iteration-2 Issue 2:
  1. smart_cut overlap durations are 80 ms audio / 40 ms video
  2. A/V invariant: total = sum(durations) - (N-1) * overlap_s
  3. Filter graph for 2 + 3 segment chains has the right labels
     and offsets (catches arithmetic regressions without spinning
     up ffmpeg)
  4. Defensive: rejects single-segment input, rejects overlap that
     would consume an entire segment
  5. Catalog wiring: SMART_CUT.implemented == True with 0.08s
     duration; CROSSFADE.implemented == True with 0.5s duration
  6. overlap_for_render maps catalog names to (audio, video)
     overlap pairs and falls back to smart_cut for non-implemented
     entries
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_PIPELINE_V2_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PIPELINE_V2_ROOT))
sys.path.insert(0, str(_PIPELINE_V2_ROOT.parent))


# ---- transitions.py wiring (catalog + overlap helper) -----------------


class TestTransitionsCatalogImplementedFlags:

    def test_smart_cut_implemented_at_80ms(self):
        """SMART_CUT is the default and item-108 ship gate. Catalog
        duration_s == 0.08 (80 ms audio crossfade)."""
        from pipeline_v2.transitions import SMART_CUT
        assert SMART_CUT.implemented is True
        assert SMART_CUT.duration_s == 0.08
        assert SMART_CUT.name == "smart_cut"

    def test_crossfade_implemented_at_500ms(self):
        """CROSSFADE was the second item-108 ship; 500 ms variant
        of the same ffmpeg pipeline."""
        from pipeline_v2.transitions import CROSSFADE
        assert CROSSFADE.implemented is True
        assert CROSSFADE.duration_s == 0.5

    def test_resolve_for_render_smart_cut_stays_smart_cut(self):
        """smart_cut is implemented -- resolver returns it."""
        from pipeline_v2.transitions import resolve_for_render, SMART_CUT
        assert resolve_for_render("smart_cut") is SMART_CUT

    def test_resolve_for_render_crossfade_stays_crossfade(self):
        """crossfade is implemented as of item 108 -- resolver
        returns crossfade (NOT smart_cut)."""
        from pipeline_v2.transitions import resolve_for_render, CROSSFADE
        assert resolve_for_render("crossfade") is CROSSFADE

    def test_resolve_for_render_non_implemented_falls_back(self):
        """The other five catalog entries (fade_to_black, dip_to_white,
        slide_left, wipe_right, dissolve) are NOT implemented and
        must fall back to smart_cut."""
        from pipeline_v2.transitions import resolve_for_render
        for name in ("fade_to_black", "dip_to_white", "slide_left",
                     "wipe_right", "dissolve"):
            assert resolve_for_render(name).name == "smart_cut"


class TestOverlapForRender:

    def test_smart_cut_overlap_is_80ms_audio_40ms_video(self):
        """The headline spec: smart_cut = 80 ms audio crossfade +
        40 ms video crossfade (~2 frames at 50fps)."""
        from pipeline_v2.transitions import overlap_for_render
        audio, video = overlap_for_render("smart_cut")
        assert audio == 0.08
        assert video == 0.04

    def test_crossfade_overlap_is_500ms_both(self):
        """The crossfade catalog entry uses the same value for
        audio + video (no separate video frame budget -- 500 ms is
        long enough to also be the visible blend window)."""
        from pipeline_v2.transitions import overlap_for_render
        audio, video = overlap_for_render("crossfade")
        assert audio == 0.5
        assert video == 0.5

    def test_unknown_and_non_implemented_resolve_to_smart_cut_overlap(self):
        """Non-implemented catalog entries fall through to smart_cut
        via resolve_for_render -- so their overlap pair is
        smart_cut's (0.08, 0.04)."""
        from pipeline_v2.transitions import overlap_for_render
        for name in ("fade_to_black", "dissolve", "xyz", "", None):
            assert overlap_for_render(name) == (0.08, 0.04)


# ---- bulletin_crossfade_stitcher pure helpers -------------------------


class TestComputeXfadeOffsets:

    def test_two_segments(self):
        """For N=2: offset[0] = d[0] - O. The transition starts
        ``overlap_s`` before the end of the first segment."""
        from pipeline_v2.bulletin_crossfade_stitcher import compute_xfade_offsets
        durs = [10.0, 12.0]
        assert compute_xfade_offsets(durs, 0.08) == [9.92]
        assert compute_xfade_offsets(durs, 0.5) == [9.5]

    def test_three_segments_cumulative_offsets(self):
        """N=3: offset[1] = d[0] + d[1] - 2O. Cumulative arithmetic
        is correct -- each chained xfade subtracts ONE more O."""
        from pipeline_v2.bulletin_crossfade_stitcher import compute_xfade_offsets
        durs = [10.0, 12.0, 8.0]
        offsets = compute_xfade_offsets(durs, 0.08)
        assert offsets == [9.92, 21.84]

    def test_four_segments_general_formula(self):
        """offset[k] = sum(d[0..k]) - (k+1) * O."""
        from pipeline_v2.bulletin_crossfade_stitcher import compute_xfade_offsets
        durs = [5.0, 6.0, 7.0, 8.0]
        offsets = compute_xfade_offsets(durs, 0.1)
        assert offsets == [4.9, 10.8, 17.7]
        # Verify: each offset should be sum so far minus (k+1)*O.
        running = 0.0
        for k, d in enumerate(durs[:-1]):
            running += d
            expected = running - (k + 1) * 0.1
            assert abs(offsets[k] - expected) < 1e-9

    def test_single_segment_raises(self):
        """N=1 has no splice -- caller must bypass the crossfade
        path entirely."""
        import pytest as _pytest
        from pipeline_v2.bulletin_crossfade_stitcher import compute_xfade_offsets
        with _pytest.raises(ValueError, match="at least 2 segments"):
            compute_xfade_offsets([5.0], 0.08)
        with _pytest.raises(ValueError, match="at least 2 segments"):
            compute_xfade_offsets([], 0.08)

    def test_overlap_exceeding_segment_duration_raises(self):
        """A segment shorter than the overlap window would have the
        crossfade consume its entire payload. Refuse rather than
        produce a broken bulletin."""
        import pytest as _pytest
        from pipeline_v2.bulletin_crossfade_stitcher import compute_xfade_offsets
        # 0.05s segment, 0.08s overlap -> reject.
        with _pytest.raises(ValueError, match="consume the entire"):
            compute_xfade_offsets([10.0, 0.05], 0.08)
        # Boundary: overlap == segment dur -> still rejected
        # (predicate is ``<= overlap_s``, leaves no audio).
        with _pytest.raises(ValueError, match="consume the entire"):
            compute_xfade_offsets([10.0, 0.08], 0.08)


class TestAVInvariantTotal:
    """Item 108 spec: total bulletin duration = sum(per-segment
    durations) - (N-1) * audio_overlap_s. This is the same A/V
    invariant the renderer's item-102 guardrail checks against."""

    def test_av_invariant_two_segments(self):
        from pipeline_v2.bulletin_crossfade_stitcher import compute_total_duration
        # 10 + 12 = 22; subtract one overlap of 0.08 = 21.92
        assert compute_total_duration([10.0, 12.0], 0.08) == 21.92

    def test_av_invariant_three_segments(self):
        from pipeline_v2.bulletin_crossfade_stitcher import compute_total_duration
        # 10 + 12 + 8 = 30; subtract two overlaps of 0.08 = 29.84
        assert compute_total_duration([10.0, 12.0, 8.0], 0.08) == 29.84

    def test_av_invariant_single_segment(self):
        """1 segment -> no splice subtraction; total is just the
        segment's own duration."""
        from pipeline_v2.bulletin_crossfade_stitcher import compute_total_duration
        assert compute_total_duration([12.5], 0.08) == 12.5

    def test_av_invariant_empty_input(self):
        from pipeline_v2.bulletin_crossfade_stitcher import compute_total_duration
        assert compute_total_duration([], 0.08) == 0.0

    def test_av_invariant_with_500ms_overlap(self):
        """The crossfade entry uses 500 ms. Verify the formula
        scales correctly."""
        from pipeline_v2.bulletin_crossfade_stitcher import compute_total_duration
        # 4 segments of 10s; 3 overlaps of 0.5s -> 40 - 1.5 = 38.5
        assert compute_total_duration([10.0] * 4, 0.5) == 38.5


# ---- filter_complex graph construction --------------------------------


class TestBuildCrossfadeFilterGraph:

    def test_two_segment_graph_has_one_xfade_and_one_acrossfade(self):
        from pipeline_v2.bulletin_crossfade_stitcher import (
            build_crossfade_filter_graph,
        )
        filter_str, v_label, a_label = build_crossfade_filter_graph(
            [10.0, 12.0], audio_overlap_s=0.08, video_overlap_s=0.04,
        )
        # Final labels.
        assert v_label == "v001"
        assert a_label == "a001"
        # Two filter nodes: one xfade, one acrossfade.
        nodes = filter_str.split(";")
        assert len(nodes) == 2
        # Video xfade first, then audio.
        assert nodes[0].startswith("[0:v][1:v]xfade=")
        assert "duration=0.04" in nodes[0]
        assert "offset=9.96" in nodes[0]   # 10.0 - 0.04
        assert nodes[0].endswith("[v001]")
        assert nodes[1].startswith("[0:a][1:a]acrossfade=")
        assert "d=0.08" in nodes[1]
        assert nodes[1].endswith("[a001]")

    def test_three_segment_graph_chains_labels_correctly(self):
        """The k-th xfade reads from the (k-1)th output, not from
        the original input. Catches the off-by-one regression where
        every node references [0:v]."""
        from pipeline_v2.bulletin_crossfade_stitcher import (
            build_crossfade_filter_graph,
        )
        filter_str, v_label, a_label = build_crossfade_filter_graph(
            [10.0, 12.0, 8.0],
            audio_overlap_s=0.08, video_overlap_s=0.04,
        )
        assert v_label == "v002"
        assert a_label == "a002"
        nodes = filter_str.split(";")
        assert len(nodes) == 4   # 2 xfade + 2 acrossfade
        # Second video xfade reads from v001, third video input,
        # produces v002.
        assert "[v001][2:v]xfade=" in nodes[1]
        assert nodes[1].endswith("[v002]")
        assert "offset=21.92" in nodes[1]   # 10 + 12 - 2*0.04
        # Second audio acrossfade reads from a001.
        assert "[a001][2:a]acrossfade=" in nodes[3]
        assert nodes[3].endswith("[a002]")

    def test_single_input_returns_passthrough_labels(self):
        """N=1 is a bypass case (caller copies the single input
        verbatim). The graph string is empty; labels point at the
        raw input streams."""
        from pipeline_v2.bulletin_crossfade_stitcher import (
            build_crossfade_filter_graph,
        )
        filter_str, v_label, a_label = build_crossfade_filter_graph([10.0])
        assert filter_str == ""
        assert v_label == "0:v"
        assert a_label == "0:a"


# ---- Stage 4 wiring smoke (no ffmpeg) ---------------------------------


class TestStage4DispatchToCrossfade:
    """Item 108 wires Stage4Render to call the crossfade stitcher
    when ``transition_style`` has a nonzero overlap. These tests
    don't run ffmpeg -- they verify the dispatch logic returns the
    right overlap pair so the renderer chooses the right path."""

    def test_default_transition_style_dispatches_to_crossfade(self):
        """Stage4Render's default transition_style='smart_cut' must
        map to nonzero overlap so the crossfade stitcher is chosen."""
        from pipeline_v2.transitions import overlap_for_render
        audio, video = overlap_for_render("smart_cut")
        assert audio > 0
        assert video > 0
