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
        # Item 111: video is now HARD-CUT (xfade chain broke on 20+
        # segments in Job 46). Only audio is crossfaded.
        assert video == 0.0

    def test_crossfade_overlap_is_500ms_audio_zero_video(self):
        """Item 111: crossfade's 500ms applies to AUDIO only.
        Video is hard-cut at every splice (3-pass stitcher concat-
        demuxes video losslessly)."""
        from pipeline_v2.transitions import overlap_for_render
        audio, video = overlap_for_render("crossfade")
        assert audio == 0.5
        assert video == 0.0

    def test_unknown_and_non_implemented_resolve_to_smart_cut_overlap(self):
        """Non-implemented catalog entries fall through to smart_cut
        via resolve_for_render -- so their overlap pair is
        smart_cut's (0.08, 0.0) after item 111."""
        from pipeline_v2.transitions import overlap_for_render
        for name in ("fade_to_black", "dissolve", "xyz", "", None):
            assert overlap_for_render(name) == (0.08, 0.0)


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


class TestBuildAudioAcrossfadeGraph:
    """Item 111: replaces ``build_crossfade_filter_graph`` for the
    production path. Only audio acrossfade -- video flows through
    the 3-pass stitcher's concat-demuxer (pass 1), never through a
    filter_complex video chain."""

    def test_two_segment_graph_has_one_acrossfade_node(self):
        """Item 115 follow-up: each input is prefixed with an
        atrim+asetpts normalisation node that strips AAC priming
        before the acrossfade chain consumes it. Two inputs =>
        2 normalisation nodes + 1 acrossfade = 3 total nodes.
        """
        from pipeline_v2.bulletin_crossfade_stitcher import (
            build_audio_acrossfade_graph,
        )
        filter_str, a_label = build_audio_acrossfade_graph(
            [10.0, 12.0], audio_overlap_s=0.08,
        )
        assert a_label == "a001"
        nodes = filter_str.split(";")
        # 2 atrim normalisers + 1 acrossfade.
        assert len(nodes) == 3
        # Normalisers strip AAC priming on each raw input.
        assert "[0:a]atrim=0:" in nodes[0] and "[n000]" in nodes[0]
        assert "[1:a]atrim=0:" in nodes[1] and "[n001]" in nodes[1]
        assert "asetpts=PTS-STARTPTS" in nodes[0]
        # Acrossfade reads from the NORMALISED labels, not the raw
        # [N:a] streams (this is the lip-sync fix).
        assert nodes[2].startswith("[n000][n001]acrossfade=")
        assert "d=0.08" in nodes[2]
        assert nodes[2].endswith("[a001]")
        # No video xfade nodes anywhere -- the bug from item 111.
        assert "xfade=" not in filter_str
        # Raw video streams must not appear at all.
        assert "0:v" not in filter_str
        assert "1:v" not in filter_str

    def test_three_segment_graph_chains_acrossfade_labels(self):
        """The k-th acrossfade reads from the previous acrossfade
        output AND the next normalised input, not the raw [N:a].
        Catches both the off-by-one chain regression AND a
        regression that bypasses the priming-strip normaliser.
        """
        from pipeline_v2.bulletin_crossfade_stitcher import (
            build_audio_acrossfade_graph,
        )
        filter_str, a_label = build_audio_acrossfade_graph(
            [10.0, 12.0, 8.0], audio_overlap_s=0.08,
        )
        assert a_label == "a002"
        nodes = filter_str.split(";")
        # 3 normalisers + 2 acrossfade = 5 nodes for 3 inputs.
        assert len(nodes) == 5
        # First acrossfade: [n000][n001] -> a001.
        assert "[n000][n001]acrossfade=" in nodes[3]
        assert nodes[3].endswith("[a001]")
        # Second acrossfade: [a001][n002] -> a002.
        assert "[a001][n002]acrossfade=" in nodes[4]
        assert nodes[4].endswith("[a002]")

    def test_twenty_five_segment_chain_does_not_collapse(self):
        """The Job-46 regression test pinned to a real failure case.
        25 segments (Job 46's actual count) must produce 25
        normalisers + 24 chained acrossfade nodes with labels
        a001..a024 in order.
        """
        from pipeline_v2.bulletin_crossfade_stitcher import (
            build_audio_acrossfade_graph,
        )
        durs = [10.0] * 25
        filter_str, a_label = build_audio_acrossfade_graph(durs, 0.08)
        assert a_label == "a024"
        nodes = filter_str.split(";")
        # 25 normalisers + 24 acrossfade.
        assert len(nodes) == 49
        # Final node consumes [a023][n024] -> [a024].
        assert "[a023][n024]acrossfade=" in nodes[-1]
        assert nodes[-1].endswith("[a024]")

    def test_single_input_returns_passthrough_label(self):
        """N=1 bypass: empty graph string, audio label points at the
        raw input stream."""
        from pipeline_v2.bulletin_crossfade_stitcher import (
            build_audio_acrossfade_graph,
        )
        filter_str, a_label = build_audio_acrossfade_graph([10.0])
        assert filter_str == ""
        assert a_label == "0:a"

    def test_item115_followup_strips_aac_priming_before_acrossfade(self):
        """REGRESSION: ffmpeg's AAC decoder emits encoder-priming
        samples (PTS = -1024) when a filter graph pulls from a
        ``[N:a]`` input. On job 50's 33-segment bulletin this
        accumulated to +350 ms of extra audio in Pass 2, leaving
        audio extending past the last video frame and producing a
        steadily-growing lip-sync drift (~10 ms / segment).

        ``build_audio_acrossfade_graph`` must wrap every raw
        ``[N:a]`` with ``atrim=0:duration,asetpts=PTS-STARTPTS``
        BEFORE the acrossfade chain consumes it. If the filter
        chain references ``[N:a]`` directly inside an acrossfade
        node (skipping the normaliser), the priming leaks through
        and the bug returns.
        """
        from pipeline_v2.bulletin_crossfade_stitcher import (
            build_audio_acrossfade_graph,
        )
        import re
        filter_str, _ = build_audio_acrossfade_graph(
            [24.866, 32.333, 2.066, 4.1, 4.7], audio_overlap_s=0.08,
        )
        # Every raw [N:a] reference must be inside an atrim
        # normalisation node, NOT directly inside an acrossfade.
        for k in range(5):
            raw_ref = f"[{k}:a]"
            # Find all occurrences and confirm each is followed by
            # an atrim node (not an acrossfade).
            for match in re.finditer(re.escape(raw_ref), filter_str):
                tail = filter_str[match.end(): match.end() + 50]
                assert tail.startswith("atrim=0:"), (
                    f"Raw input {raw_ref} must feed an atrim "
                    f"normaliser (item 115 follow-up), not "
                    f"directly into a filter. Got tail={tail!r}"
                )
        # acrossfade nodes must read from normalised labels [n###]
        # or chained outputs [a###], never from a raw [N:a].
        for match in re.finditer(r"acrossfade=", filter_str):
            head = filter_str[max(0, match.start() - 20): match.start()]
            assert "[n" in head or "[a" in head, (
                f"Acrossfade input must be a normalised or chained "
                f"label, not raw [N:a]. Got head={head!r}"
            )

    def test_backwards_compat_shim_still_returns_three_tuple(self):
        """The old ``build_crossfade_filter_graph`` 3-tuple shape is
        preserved (one external caller -- a not-yet-migrated test --
        depends on it). The video label is hard-coded to '0:v'
        because there is no longer a video filter chain.

        Item 115 follow-up: the audio chain now prepends an
        atrim+asetpts normaliser per input to strip AAC priming
        (see ``test_item115_followup_strips_aac_priming_before_acrossfade``);
        the legacy 3-tuple shape carries the new filter unchanged.
        """
        from pipeline_v2.bulletin_crossfade_stitcher import (
            build_crossfade_filter_graph,
        )
        filter_str, v_label, a_label = build_crossfade_filter_graph(
            [10.0, 12.0], audio_overlap_s=0.08, video_overlap_s=0.04,
        )
        # No video chain anywhere.
        assert v_label == "0:v"
        assert "xfade=" not in filter_str
        # Audio chain identical to build_audio_acrossfade_graph
        # (item 115 follow-up: normalisers prepended).
        assert a_label == "a001"
        assert "[0:a]atrim=0:" in filter_str
        assert "[n000][n001]acrossfade=d=0.08[a001]" in filter_str


# ---- Item 111 spec tests (the 4 the user explicitly named) ------------


class TestItem111SmartCutBehavior:
    """The 4 tests the user spec'd in the iteration-2 Issue 2 fix:
      - smart_cut_video_uses_concat_no_overlap
      - smart_cut_audio_uses_acrossfade
      - smart_cut_av_durations_match_within_audio_overlap
      - smart_cut_produces_valid_playable_mp4
    """

    def test_smart_cut_video_uses_concat_no_overlap(self):
        """Video overlap is ALWAYS 0 for smart_cut after item 111
        (the broken xfade chain is gone; video hard-cuts via
        concat-demuxer)."""
        from pipeline_v2.transitions import overlap_for_render
        from pipeline_v2.bulletin_crossfade_stitcher import (
            DEFAULT_VIDEO_OVERLAP_S,
        )
        _, video_overlap = overlap_for_render("smart_cut")
        assert video_overlap == 0.0
        # Module default also at 0.
        assert DEFAULT_VIDEO_OVERLAP_S == 0.0

    def test_smart_cut_audio_uses_acrossfade(self):
        """Audio overlap is 80ms for smart_cut. This is the part
        that addresses Gemini's ambient-spike finding."""
        from pipeline_v2.transitions import overlap_for_render
        from pipeline_v2.bulletin_crossfade_stitcher import (
            DEFAULT_AUDIO_OVERLAP_S,
        )
        audio_overlap, _ = overlap_for_render("smart_cut")
        assert audio_overlap == 0.08
        assert DEFAULT_AUDIO_OVERLAP_S == 0.08

    def test_smart_cut_av_durations_match_within_audio_overlap(self):
        """For N segments of duration d (uniform), the bulletin
        AUDIO duration is ``N*d - (N-1) * audio_overlap_s``. Video
        stays at ``N*d`` (concat, no overlap). The 3-pass stitcher
        then -shortest-trims both to the shorter of the two, so the
        final file's V/A durations match within audio_overlap_s.
        """
        from pipeline_v2.bulletin_crossfade_stitcher import (
            compute_total_duration,
        )
        # 25 segments of 10s each, 80ms audio overlap (Job 46-like).
        n = 25
        d = 10.0
        audio_overlap = 0.08
        audio_total = compute_total_duration([d] * n, audio_overlap)
        video_total = compute_total_duration([d] * n, 0.0)    # concat: no overlap
        # video_total - audio_total = (N-1) * audio_overlap.
        # Use abs tolerance because float subtraction across 25
        # accumulations leaves ~1e-13 rounding noise.
        assert abs((video_total - audio_total) - (n - 1) * audio_overlap) < 1e-9
        # The mux uses -shortest, so the FILE's audio + video both
        # end at min(audio_total, video_total) = audio_total.
        # Result: A/V mismatch in the FILE is bounded by 1 frame
        # (well under 0.2s tolerance).

    def test_item115_pass3_reencodes_audio_for_sample_accurate_shortest(self):
        """Item 115 lip-sync fix: Pass 3 mux must re-encode audio
        (``-c:a aac``), NOT stream-copy. With ``-c:a copy`` the
        demuxer cannot split an AAC packet at the video EOF, so
        ``-shortest`` leaks up to one AAC frame (~21ms) of audio
        past the last video frame -- that's the bulletin-scale
        lip-sync drift we localized on job 49.

        Re-encoding lets ``-shortest`` truncate sample-accurately.
        Video stays ``-c:v copy`` so quality is preserved.
        """
        import inspect
        import re
        from pipeline_v2 import bulletin_crossfade_stitcher
        src = inspect.getsource(
            bulletin_crossfade_stitcher.stitch_bulletin_with_crossfade,
        )
        # Isolate the Pass 3 cmd3 block (between "cmd3 = [" and the
        # next "]" at the same indentation level). Cheaper than
        # parsing AST and tightly couples the assertion to the
        # Pass 3 mux, not the N==1 bypass which legitimately uses
        # blanket -c copy.
        m = re.search(
            r"cmd3\s*=\s*\[(.*?)\n\s*\]",
            src, re.DOTALL,
        )
        assert m, "could not locate cmd3 list in stitch_bulletin_with_crossfade"
        cmd3_src = m.group(1)
        assert '"-c", "copy"' not in cmd3_src, (
            "Pass 3 cmd3 must not use blanket -c copy (item 115); "
            "audio needs re-encode for sample-accurate -shortest"
        )
        assert '"-c:v", "copy"' in cmd3_src
        assert '"-c:a", "aac"' in cmd3_src
        assert '"-shortest"' in cmd3_src

    def test_smart_cut_produces_valid_playable_mp4(self):
        """Integration test: run the real 3-pass stitcher on a
        2-segment fixture (or 25 segments if Job 46's outputs are
        available) and ffprobe the result. The Job-46 regression
        check that pins ``video_duration > 100s`` (was 104.5s
        broken)."""
        import os
        import shutil
        import subprocess
        import tempfile
        import pytest as _pytest

        # Skip if ffmpeg / ffprobe unavailable in CI.
        if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            _pytest.skip("ffmpeg/ffprobe not available in PATH")

        # Look for Job 46's composed_story files (the real Job-46
        # regression). Skip if they don't exist (e.g. CI without
        # the local output tree).
        backend_root = Path(__file__).resolve().parent.parent.parent
        job46 = backend_root / "output" / "full_video_shorts_v2" / "job_46" / "bulletin"
        segs = sorted(str(p) for p in job46.glob("composed_story_*.mp4"))
        if len(segs) < 2:
            _pytest.skip(
                "Job 46 composed_story files not available -- "
                "integration check requires the local output tree"
            )

        from pipeline_v2.bulletin_crossfade_stitcher import (
            stitch_bulletin_with_crossfade,
        )
        with tempfile.TemporaryDirectory(prefix="kaizer_item111_") as td:
            out_path = os.path.join(td, "test_output.mp4")
            result = stitch_bulletin_with_crossfade(
                segs, out_path, audio_overlap_s=0.08,
            )
            assert result.stories_rendered == len(segs)
            # Probe video stream duration.
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=duration,nb_frames",
                 "-of", "default=noprint_wrappers=1:nokey=1", out_path],
                capture_output=True, text=True, timeout=30,
            )
            lines = [l for l in (r.stdout or "").strip().splitlines() if l]
            assert len(lines) >= 2, f"ffprobe output unexpected: {lines!r}"
            video_dur = float(lines[0])
            nb_frames = int(lines[1])
            # Job 46's bug: video was 104.5s (3136 frames). After
            # item 111, video must be >> 100s for the real Job 46
            # input.
            assert video_dur > 100.0, (
                f"Item 111 regression: video duration is {video_dur:.2f}s "
                f"(was 104.5s in the broken Job 46 bulletin). The 3-pass "
                f"stitcher should produce a video stream matching the "
                f"sum of input video durations."
            )
            assert nb_frames > 3000, (
                f"Item 111 regression: only {nb_frames} video frames "
                f"(was 3136 in the broken Job 46 bulletin)."
            )


# ---- Stage 4 wiring smoke (no ffmpeg) ---------------------------------


class TestStage4DispatchToCrossfade:
    """Item 108 wires Stage4Render to call the crossfade stitcher
    when ``transition_style`` has nonzero audio overlap (item 111
    update: nonzero AUDIO, not video). These tests don't run
    ffmpeg -- they verify the dispatch logic returns the right
    overlap pair so the renderer chooses the crossfade path."""

    def test_default_transition_style_dispatches_to_crossfade(self):
        """Stage4Render's default transition_style='smart_cut' must
        map to nonzero AUDIO overlap so the 3-pass stitcher is chosen.
        Item 111 update: video overlap is always 0; the dispatch
        gate in stage_4_render._render_impl checks
        ``audio_overlap_s > 0 or video_overlap_s > 0`` so it still
        routes correctly."""
        from pipeline_v2.transitions import overlap_for_render
        audio, video = overlap_for_render("smart_cut")
        assert audio > 0
        assert video == 0.0    # item 111: hard-cut video
