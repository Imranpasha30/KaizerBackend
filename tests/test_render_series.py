"""
tests/test_render_series.py
============================
pytest coverage for pipeline_core/render_series.py

chain_parts() calls render_mode_clip() (imported locally inside the function)
and _render_badge_on_clip() (module-level helper).  Fast tests patch both so
no real ffmpeg I/O happens.  One slow integration test runs real ffmpeg.
"""
from __future__ import annotations

import os
import subprocess
import json
import shutil
from dataclasses import fields
from unittest.mock import MagicMock, patch

import pytest

from pipeline_core.narrative import ClipCandidate, NarrativeResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(
    start: float = 0.0,
    end: float = 8.0,
    narrative_role: str = "climax",
    hook_score: float = 0.7,
    completion_score: float = 0.8,
    importance_score: float = 0.9,
    composite_score: float = 0.8,
) -> ClipCandidate:
    return ClipCandidate(
        start=start,
        end=end,
        duration=end - start,
        narrative_role=narrative_role,
        hook_score=hook_score,
        completion_score=completion_score,
        importance_score=importance_score,
        composite_score=composite_score,
        transcript_slice="Test transcript slice.",
        start_sources=["shot"],
        end_sources=["sentence"],
        meta={},
    )


def _make_candidates(n: int, window: float = 8.0, gap: float = 2.0) -> list[ClipCandidate]:
    """Generate n non-overlapping candidates."""
    result = []
    t = 0.0
    for i in range(n):
        result.append(_make_candidate(start=t, end=t + window))
        t += window + gap
    return result


def _make_rendered_clip(output_path: str, cta_applied: str = "next_part", duration_s: float = 8.0):
    """Return a mock RenderedClip."""
    from pipeline_core.render_modes import RenderedClip
    return RenderedClip(
        mode="series",
        output_path=output_path,
        duration_s=duration_s,
        clip_candidate=None,
        cta_applied=cta_applied,
        qa_ok=True,
        qa_warnings=[],
        meta={"warnings": []},
    )


def _patch_chain_parts_internals(tmp_path, n: int, cta_styles: list[str] | None = None):
    """
    Return a context-manager tuple that stubs render_mode_clip and
    _render_badge_on_clip for chain_parts unit tests.

    render_mode_clip is imported locally inside chain_parts as:
        from pipeline_core.render_modes import render_mode_clip
    So the correct patch target is pipeline_core.render_modes.render_mode_clip.

    _render_badge_on_clip is a module-level function, so it is patched at
    pipeline_core.render_series._render_badge_on_clip.
    """
    if cta_styles is None:
        cta_styles = ["next_part"] * (n - 1) + ["soft_follow"]

    call_count = {"i": 0}

    def fake_render_mode_clip(source, cand, *, mode, output_path, **kwargs):
        i = call_count["i"]
        # Write a stub file so downstream checks don't error
        with open(output_path, "wb") as f:
            f.write(b"stub")
        result = _make_rendered_clip(
            output_path,
            cta_applied=cta_styles[i] if i < len(cta_styles) else "soft_follow",
        )
        call_count["i"] += 1
        return result

    def fake_render_badge(clip_path, badge_text, output_path, **kwargs):
        # Copy the stub file to simulate badge burn
        shutil.copy2(clip_path, output_path)
        return []

    return (
        patch(
            "pipeline_core.render_modes.render_mode_clip",
            side_effect=fake_render_mode_clip,
        ),
        patch(
            "pipeline_core.render_series._render_badge_on_clip",
            side_effect=fake_render_badge,
        ),
        patch(
            "pipeline_core.render_series._probe_duration",
            return_value=8.0,
        ),
    )


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestChainPartsValidation:
    def test_chain_parts_requires_at_least_2_candidates(self, tmp_path):
        """1 candidate → ValueError."""
        from pipeline_core.render_series import chain_parts

        fake_video = str(tmp_path / "v.mp4")
        (tmp_path / "v.mp4").write_bytes(b"fake")

        with pytest.raises(ValueError, match="2"):
            chain_parts(
                fake_video,
                [_make_candidate()],
                output_dir=str(tmp_path),
            )

    def test_chain_parts_max_5_candidates(self, tmp_path):
        """6 candidates → ValueError."""
        from pipeline_core.render_series import chain_parts

        fake_video = str(tmp_path / "v.mp4")
        (tmp_path / "v.mp4").write_bytes(b"fake")

        candidates = _make_candidates(6)
        with pytest.raises(ValueError, match="5"):
            chain_parts(
                fake_video,
                candidates,
                output_dir=str(tmp_path),
            )


# ---------------------------------------------------------------------------
# Manifest shape tests
# ---------------------------------------------------------------------------

class TestSeriesManifestShape:
    def test_chain_parts_returns_series_manifest_dataclass(self, tmp_path):
        """chain_parts returns a SeriesManifest dataclass."""
        from pipeline_core.render_series import chain_parts, SeriesManifest

        fake_video = str(tmp_path / "v.mp4")
        (tmp_path / "v.mp4").write_bytes(b"fake")

        candidates = _make_candidates(2)
        patches = _patch_chain_parts_internals(tmp_path, 2)

        with patches[0], patches[1], patches[2]:
            result = chain_parts(
                fake_video,
                candidates,
                output_dir=str(tmp_path),
                run_qa=False,
            )

        assert isinstance(result, SeriesManifest), (
            f"Expected SeriesManifest, got {type(result)}"
        )

    def test_part_index_is_one_based(self, tmp_path):
        """First part has part_index=1, last has part_index=N."""
        from pipeline_core.render_series import chain_parts

        fake_video = str(tmp_path / "v.mp4")
        (tmp_path / "v.mp4").write_bytes(b"fake")

        n = 3
        candidates = _make_candidates(n)
        patches = _patch_chain_parts_internals(tmp_path, n)

        with patches[0], patches[1], patches[2]:
            result = chain_parts(
                fake_video,
                candidates,
                output_dir=str(tmp_path),
                run_qa=False,
            )

        assert len(result.parts) == n, f"Expected {n} parts, got {len(result.parts)}"
        assert result.parts[0].part_index == 1, (
            f"First part_index should be 1, got {result.parts[0].part_index}"
        )
        assert result.parts[-1].part_index == n, (
            f"Last part_index should be {n}, got {result.parts[-1].part_index}"
        )
        for part in result.parts:
            assert part.part_total == n, (
                f"part_total should be {n}, got {part.part_total}"
            )

    def test_pinned_comment_template_contains_placeholder(self, tmp_path):
        """The manifest's pinned_comment_template must contain {N} or {URL} or both."""
        from pipeline_core.render_series import chain_parts

        fake_video = str(tmp_path / "v.mp4")
        (tmp_path / "v.mp4").write_bytes(b"fake")

        candidates = _make_candidates(2)
        patches = _patch_chain_parts_internals(tmp_path, 2)

        with patches[0], patches[1], patches[2]:
            result = chain_parts(
                fake_video,
                candidates,
                output_dir=str(tmp_path),
                run_qa=False,
            )

        template = result.pinned_comment_template
        assert isinstance(template, str), "pinned_comment_template must be a str"
        has_placeholder = "{N}" in template or "{URL}" in template or "{n}" in template
        assert has_placeholder, (
            f"pinned_comment_template must contain {{N}} or {{URL}}, got: {template!r}"
        )


# ---------------------------------------------------------------------------
# Cliffhanger detection tests
# ---------------------------------------------------------------------------

class TestCliffhangerDetection:
    def test_cliffhanger_detected_on_low_completion(self, tmp_path):
        """Candidate with completion_score=0.2 → part.cliffhanger_detected == True."""
        from pipeline_core.render_series import chain_parts

        fake_video = str(tmp_path / "v.mp4")
        (tmp_path / "v.mp4").write_bytes(b"fake")

        c1 = _make_candidate(start=0.0, end=8.0, completion_score=0.2)
        c2 = _make_candidate(start=10.0, end=18.0, completion_score=0.9)

        patches = _patch_chain_parts_internals(tmp_path, 2)
        with patches[0], patches[1], patches[2]:
            result = chain_parts(
                fake_video,
                [c1, c2],
                output_dir=str(tmp_path),
                run_qa=False,
            )

        assert result.parts[0].cliffhanger_detected is True, (
            "completion_score=0.2 should trigger cliffhanger_detected=True"
        )

    def test_cliffhanger_detected_on_setup_role(self, tmp_path):
        """narrative_role='setup' → cliffhanger=True even with high completion_score."""
        from pipeline_core.render_series import chain_parts

        fake_video = str(tmp_path / "v.mp4")
        (tmp_path / "v.mp4").write_bytes(b"fake")

        c1 = _make_candidate(start=0.0, end=8.0,
                              narrative_role="setup", completion_score=0.9)
        c2 = _make_candidate(start=10.0, end=18.0,
                              narrative_role="climax", completion_score=0.9)

        patches = _patch_chain_parts_internals(tmp_path, 2)
        with patches[0], patches[1], patches[2]:
            result = chain_parts(
                fake_video,
                [c1, c2],
                output_dir=str(tmp_path),
                run_qa=False,
            )

        assert result.parts[0].cliffhanger_detected is True, (
            "narrative_role='setup' should trigger cliffhanger_detected=True"
        )


# ---------------------------------------------------------------------------
# CTA assignment tests
# ---------------------------------------------------------------------------

class TestSeriesCTAAssignment:
    def test_last_part_cta_is_follow_not_next(self, tmp_path):
        """The final part's cta_applied should not be 'next_part'."""
        from pipeline_core.render_series import chain_parts

        fake_video = str(tmp_path / "v.mp4")
        (tmp_path / "v.mp4").write_bytes(b"fake")

        n = 3
        candidates = _make_candidates(n)

        # Provide explicit CTA styles: first two get next_part, last gets soft_follow
        patches = _patch_chain_parts_internals(
            tmp_path, n,
            cta_styles=["next_part", "next_part", "soft_follow"],
        )
        with patches[0], patches[1], patches[2]:
            result = chain_parts(
                fake_video,
                candidates,
                output_dir=str(tmp_path),
                run_qa=False,
            )

        last_cta = result.parts[-1].cta_applied
        assert last_cta != "next_part", (
            f"Last part must not use 'next_part' CTA, got: {last_cta!r}"
        )


# ---------------------------------------------------------------------------
# Slow integration test
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_chain_three_parts_end_to_end(valid_long_mp4, tmp_path):
    """3 synthetic candidates → 3 MP4 output files, each with Part X/3 overlay."""
    from pipeline_core.render_series import chain_parts

    candidates = [
        _make_candidate(start=0.0, end=8.0),
        _make_candidate(start=10.0, end=18.0),
        _make_candidate(start=20.0, end=28.0),
    ]

    result = chain_parts(
        valid_long_mp4,
        candidates,
        output_dir=str(tmp_path),
        playlist_title="Test Series",
        run_qa=False,
    )

    assert len(result.parts) == 3, f"Expected 3 parts, got {len(result.parts)}"

    ffprobe = shutil.which("ffprobe") or "ffprobe"

    for part in result.parts:
        assert os.path.isfile(part.output_path), (
            f"Part {part.part_index} output file missing: {part.output_path}"
        )
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", part.output_path],
            capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode == 0, (
            f"ffprobe failed on part {part.part_index}: {proc.stderr}"
        )
        data = json.loads(proc.stdout)
        dur = float(data["format"]["duration"])
        assert dur > 0, f"Part {part.part_index} has zero duration"
