"""Item 117 Phase 1 -- EDL builder unit tests.

Pure-function tests: no ffmpeg invocation, no file I/O. Covers:

  - filter_complex node structure (1-cut bulletin bypasses concat;
    N-cut bulletin emits the concat node)
  - output map labels match documented shape ([bv_out],[ba_out],
    [svNN],[saNN])
  - zero-length cuts are dropped, surfaced in EDL.dropped
  - both-empty plan raises ValueError
  - boundary snapping rounds to 1/30s grid
  - shorts plan can be empty
  - frame-aligned vs non-aligned cuts both produce valid output
  - large 28+8 graph matches the diagnostic test's shape
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_PIPELINE_V2_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PIPELINE_V2_ROOT))


# ---------------------------------------------------------------------
# Snap helper
# ---------------------------------------------------------------------


def test_snap_to_grid_frame_aligned_is_idempotent():
    from pipeline_v2.render.edl_builder import snap_to_grid
    # 1/30 s grid: t = 0.5 -> 15 frames * (1/30) = 0.5
    assert snap_to_grid(0.5) == pytest.approx(0.5, abs=1e-9)
    # 27.1 == 813/30 exactly
    assert snap_to_grid(27.1) == pytest.approx(27.1, abs=1e-9)


def test_snap_to_grid_rounds_to_nearest_frame():
    from pipeline_v2.render.edl_builder import snap_to_grid
    # 1.617 s -> nearest frame is 1.6 (48 frames) or 1.6333 (49 frames)?
    # 1.617 / (1/30) = 48.51 -> round = 49 -> 49/30 = 1.633333...
    snapped = snap_to_grid(1.617)
    assert abs(snapped - 49.0 / 30.0) < 1e-9


# ---------------------------------------------------------------------
# build_extraction_edl basic shapes
# ---------------------------------------------------------------------


def test_build_edl_raises_when_both_inputs_empty():
    from pipeline_v2.render.edl_builder import build_extraction_edl
    with pytest.raises(ValueError, match="nothing to extract"):
        build_extraction_edl([], [])


def test_build_edl_single_cut_bulletin_bypasses_concat():
    """N=1 bulletin: concat=n=1 is wasteful + occasionally erroneous
    on older ffmpeg builds. The single trim's labels become the
    bulletin output's labels directly."""
    from pipeline_v2.render.edl_builder import build_extraction_edl
    edl = build_extraction_edl([(1.6, 26.467)], [])
    assert "concat=" not in edl.filter_complex
    assert len(edl.outputs) == 1
    out = edl.outputs[0]
    assert out.role == "bulletin"
    assert out.v_label == "bv00"
    assert out.a_label == "ba00"
    # Snapped: 1.6 already aligned; 26.467 -> 794/30 = 26.4666...
    assert out.duration_s == pytest.approx(26.4666 - 1.6, abs=0.001)


def test_build_edl_multi_cut_bulletin_emits_concat():
    """N>1 bulletin: trim+atrim per cut, then concat node with
    v=1:a=0 + v=0:a=1 (split) -> bv_out / ba_out."""
    from pipeline_v2.render.edl_builder import build_extraction_edl
    edl = build_extraction_edl(
        [(1.6, 26.467), (27.1, 59.433), (60.4, 62.467)], [],
    )
    fc = edl.filter_complex
    # 3 trim + 3 atrim + 1 concat-v + 1 concat-a = 8 nodes
    assert fc.count(";") == 7   # 7 separators between 8 nodes
    assert "concat=n=3:v=1:a=0[bv_out]" in fc
    assert "concat=n=3:v=0:a=1[ba_out]" in fc
    assert edl.outputs[0].v_label == "bv_out"
    assert edl.outputs[0].a_label == "ba_out"
    assert len(edl.outputs[0].source_cuts) == 3


def test_build_edl_shorts_only_no_bulletin():
    """Empty bulletin + N shorts: only short outputs in the spec list,
    no bv_*/ba_* nodes in the filter_complex."""
    from pipeline_v2.render.edl_builder import build_extraction_edl
    edl = build_extraction_edl(
        [],
        [(10.0, 30.0), (50.0, 70.0)],
    )
    assert "bv" not in edl.filter_complex
    assert "ba" not in edl.filter_complex
    assert len(edl.outputs) == 2
    assert edl.outputs[0].role == "short"
    assert edl.outputs[0].index == 1
    assert edl.outputs[0].v_label == "sv00"
    assert edl.outputs[1].index == 2
    assert edl.outputs[1].v_label == "sv01"


def test_build_edl_bulletin_and_shorts_both_present():
    """The full production shape: bulletin output first, then shorts
    in input order. Labels do not collide."""
    from pipeline_v2.render.edl_builder import build_extraction_edl
    edl = build_extraction_edl(
        [(1.6, 26.467), (27.1, 59.433)],
        [(10.0, 30.0), (50.0, 70.0)],
    )
    # Bulletin output first.
    assert edl.outputs[0].role == "bulletin"
    # Shorts follow, 1-indexed.
    assert edl.outputs[1].role == "short"
    assert edl.outputs[1].index == 1
    assert edl.outputs[2].role == "short"
    assert edl.outputs[2].index == 2
    # All map labels are distinct.
    labels = {o.v_label for o in edl.outputs} | {o.a_label for o in edl.outputs}
    assert len(labels) == 2 * len(edl.outputs)


# ---------------------------------------------------------------------
# Drop / edge cases
# ---------------------------------------------------------------------


def test_build_edl_drops_zero_length_cuts_and_surfaces_them():
    """A cut whose snapped duration is <= 0 is dropped from the output
    set AND surfaced in EDL.dropped so the caller can log a warning."""
    from pipeline_v2.render.edl_builder import build_extraction_edl
    # (5.0, 5.0) snaps to (5.0, 5.0) -> zero duration.
    # (5.0, 4.0) snaps to (5.0, 4.0) -> negative duration.
    edl = build_extraction_edl(
        [(1.6, 26.467), (5.0, 5.0), (5.0, 4.0)], [],
    )
    # Only the valid cut survived; bulletin shows N=1.
    assert len(edl.outputs[0].source_cuts) == 1
    # Both invalid cuts surfaced in dropped.
    assert len(edl.dropped) == 2
    reasons = {d.reason for d in edl.dropped}
    assert reasons == {"non-positive duration after snap"}


def test_build_edl_all_zero_length_raises():
    """All cuts degenerate -> raise (caller must not invoke ffmpeg)."""
    from pipeline_v2.render.edl_builder import build_extraction_edl
    with pytest.raises(ValueError, match="non-positive duration"):
        build_extraction_edl([(5.0, 5.0), (10.0, 10.0)], [])


def test_build_edl_handles_non_frame_aligned_starts():
    """Non-frame-aligned starts (e.g. 1.617s) must snap cleanly. The
    diagnostic test verified this works at ffmpeg-level; here we
    just confirm the EDL emits valid timestamps."""
    from pipeline_v2.render.edl_builder import build_extraction_edl
    edl = build_extraction_edl([(1.617, 12.717)], [])
    # 1.617*30 = 48.51 -> 49 -> 49/30 = 1.633333...
    # 12.717*30 = 381.51 -> 382 -> 382/30 = 12.733333...
    assert "trim=start=1.633333:end=12.733333" in edl.filter_complex


def test_build_edl_filter_complex_uses_microsecond_precision():
    """Every timestamp in the filter graph must format with 6 decimal
    places so float-format truncation can't shift frame inclusion
    (the bug item 116 closed with the .6f bump)."""
    from pipeline_v2.render.edl_builder import build_extraction_edl
    edl = build_extraction_edl([(1.6, 26.467)], [(10.0, 30.0)])
    # Spot-check: 6 decimals on every numeric timestamp.
    import re
    numbers = re.findall(r"=(\d+\.\d+)[:,\]]", edl.filter_complex)
    for n in numbers:
        # Each number should have at least 6 decimals (we pad with zeros).
        decimals = n.split(".", 1)[1]
        assert len(decimals) >= 6, (
            f"timestamp {n!r} has fewer than 6 decimals -- precision "
            f"truncation could re-introduce item 116's bug"
        )


# ---------------------------------------------------------------------
# Production-scale shape (Job 51's 28+8 plan)
# ---------------------------------------------------------------------


def test_build_edl_production_scale_28_bulletin_plus_8_shorts():
    """The exact shape the diagnostic test verified at -0.01ms A/V
    delta. Catches any regression in graph construction at scale."""
    from pipeline_v2.render.edl_builder import build_extraction_edl
    # 28 bulletin cuts (synthetic but realistic durations).
    bulletin = [(i * 20.0 + 1.0, i * 20.0 + 18.0) for i in range(28)]
    # 8 shorts.
    shorts = [(i * 60.0 + 5.0, i * 60.0 + 25.0) for i in range(8)]
    edl = build_extraction_edl(bulletin, shorts)
    # Expected node count: 28 v-trim + 28 a-trim + 2 concat (v+a)
    # + 8 v-trim + 8 a-trim = 2*28 + 2 + 2*8 = 74 nodes.
    node_count = edl.filter_complex.count(";") + 1
    assert node_count == 74, f"expected 74 nodes, got {node_count}"
    # 1 bulletin + 8 short outputs.
    assert len(edl.outputs) == 9
    assert edl.outputs[0].role == "bulletin"
    assert sum(1 for o in edl.outputs if o.role == "short") == 8


# ---------------------------------------------------------------------
# Map argument helper (consumed by stage_4_raw_extract)
# ---------------------------------------------------------------------


def test_build_edl_output_specs_carry_duration_for_post_verify():
    """Each OutputSpec.duration_s is the EXPECTED output duration
    (sum of snapped cut durations). Phase 4a's caller asserts the
    ffprobe-measured duration against this within tolerance."""
    from pipeline_v2.render.edl_builder import build_extraction_edl
    edl = build_extraction_edl(
        [(1.6, 26.467), (27.1, 59.433)],   # 24.866 + 32.333 = 57.2
        [(10.0, 30.0)],                     # 20.0
    )
    bul = next(o for o in edl.outputs if o.role == "bulletin")
    short = next(o for o in edl.outputs if o.role == "short")
    assert bul.duration_s == pytest.approx(57.2, abs=0.001)
    assert short.duration_s == pytest.approx(20.0, abs=0.001)
