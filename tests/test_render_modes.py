"""
tests/test_render_modes.py
==========================
pytest coverage for pipeline_core/render_modes.py

Fast tests mock all ffmpeg and narrative calls.
Slow integration tests use real ffmpeg via valid_long_mp4 fixture.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import fields
from unittest.mock import MagicMock, patch, call

import pytest

from pipeline_core.narrative import ClipCandidate, NarrativeResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(
    start: float = 2.0,
    end: float = 12.0,
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
        transcript_slice="Test transcript.",
        start_sources=["shot"],
        end_sources=["sentence"],
        meta={},
    )


def _make_narrative(
    candidates=None,
    source_duration: float = 30.0,
    language: str = "en",
) -> NarrativeResult:
    if candidates is None:
        candidates = [_make_candidate()]
    return NarrativeResult(
        candidates=candidates,
        source_duration=source_duration,
        language=language,
    )


def _make_validation_ok(duration_s: float = 30.0):
    """Return a mock ValidationResult that passes.

    meta must be a real dict so that code like
    ``val_result.meta.get("duration_s", 0.0)`` returns a float, not a MagicMock.
    """
    m = MagicMock()
    m.ok = True
    m.errors = []
    m.warnings = []
    m.meta = {
        "duration_s": duration_s,
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "video_codec": "h264",
        "audio_codec": "aac",
        "bitrate_kbps": 8000.0,
        "probe_score": 100,
        "has_video": True,
        "has_audio": True,
        "container": "mp4",
    }
    return m


def _make_cta_result(output_path: str):
    """Return a mock CTAResult."""
    from pipeline_core.cta_overlay import CTAResult
    return CTAResult(
        output_path=output_path,
        cta_style="soft_follow",
        cta_start_s=7.0,
        cta_duration_s=3.0,
        warnings=[],
    )


# ---------------------------------------------------------------------------
# Config / constant tests
# ---------------------------------------------------------------------------

class TestRenderModeConfigs:
    EXPECTED_MODES = {"standalone", "trailer", "series", "promo", "highlight", "full_narrative"}

    def test_all_six_modes_in_config(self):
        """RENDER_MODE_CONFIGS must contain exactly the 6 documented modes."""
        from pipeline_core.render_modes import RENDER_MODE_CONFIGS

        assert set(RENDER_MODE_CONFIGS.keys()) == self.EXPECTED_MODES, (
            f"Modes mismatch: {set(RENDER_MODE_CONFIGS.keys())} vs {self.EXPECTED_MODES}"
        )

    def test_each_mode_config_has_required_fields(self):
        """Each RenderModeConfig entry must have min_dur_s, max_dur_s, and hook_weight."""
        from pipeline_core.render_modes import RENDER_MODE_CONFIGS, RenderModeConfig

        required_field_names = {"min_dur_s", "max_dur_s", "hook_weight"}
        for mode_name, cfg in RENDER_MODE_CONFIGS.items():
            assert isinstance(cfg, RenderModeConfig), (
                f"Config for {mode_name!r} is not a RenderModeConfig"
            )
            cfg_fields = {f.name for f in fields(cfg)}
            missing = required_field_names - cfg_fields
            assert not missing, (
                f"RenderModeConfig for {mode_name!r} is missing fields: {missing}"
            )

    def test_trailer_weights_hook_highest(self):
        """trailer.hook_weight > standalone.hook_weight (trailers live on hooks)."""
        from pipeline_core.render_modes import RENDER_MODE_CONFIGS

        trailer_w = RENDER_MODE_CONFIGS["trailer"].hook_weight
        standalone_w = RENDER_MODE_CONFIGS["standalone"].hook_weight
        assert trailer_w > standalone_w, (
            f"trailer.hook_weight ({trailer_w}) should be > standalone ({standalone_w})"
        )

    def test_full_narrative_max_180s(self):
        """full_narrative.max_dur_s == 180."""
        from pipeline_core.render_modes import RENDER_MODE_CONFIGS

        assert RENDER_MODE_CONFIGS["full_narrative"].max_dur_s == 180.0, (
            f"full_narrative.max_dur_s = {RENDER_MODE_CONFIGS['full_narrative'].max_dur_s}"
        )

    def test_promo_max_25s(self):
        """promo.max_dur_s == 25."""
        from pipeline_core.render_modes import RENDER_MODE_CONFIGS

        assert RENDER_MODE_CONFIGS["promo"].max_dur_s == 25.0, (
            f"promo.max_dur_s = {RENDER_MODE_CONFIGS['promo'].max_dur_s}"
        )


# ---------------------------------------------------------------------------
# render_mode_clip unit tests
# ---------------------------------------------------------------------------

class TestRenderModeClip:
    def test_invalid_mode_raises(self, tmp_path):
        """render_mode_clip(mode='potato') must raise ValueError."""
        from pipeline_core.render_modes import render_mode_clip

        fake_video = str(tmp_path / "v.mp4")
        (tmp_path / "v.mp4").write_bytes(b"fake")

        candidate = _make_candidate()

        with pytest.raises(ValueError, match="potato"):
            render_mode_clip(
                fake_video,
                candidate,
                mode="potato",
                output_path=str(tmp_path / "out.mp4"),
            )

    def test_render_mode_clip_returns_rendered_clip(self, tmp_path):
        """render_mode_clip with a valid mode returns a RenderedClip dataclass."""
        from pipeline_core.render_modes import render_mode_clip, RenderedClip

        fake_video = str(tmp_path / "v.mp4")
        (tmp_path / "v.mp4").write_bytes(b"fake")

        candidate = _make_candidate(start=2.0, end=12.0)
        out_path = str(tmp_path / "out.mp4")

        with (
            patch("pipeline_core.render_modes.validate_input",
                  return_value=_make_validation_ok()),
            patch("pipeline_core.render_modes._ffmpeg_slice", return_value=None),
            patch("pipeline_core.render_modes.apply_cta",
                  return_value=_make_cta_result(out_path)),
        ):
            result = render_mode_clip(
                fake_video,
                candidate,
                mode="standalone",
                output_path=out_path,
                run_qa=False,
            )

        assert isinstance(result, RenderedClip)
        assert result.mode == "standalone"

    def test_candidate_duration_clamped_to_mode_range(self, tmp_path):
        """A 200 s candidate in mode='promo' (max=25 s) → result.duration_s ≤ 25.5."""
        from pipeline_core.render_modes import render_mode_clip, RenderedClip

        fake_video = str(tmp_path / "v.mp4")
        (tmp_path / "v.mp4").write_bytes(b"fake")

        # Candidate spanning 200 s
        candidate = _make_candidate(start=0.0, end=200.0)
        out_path = str(tmp_path / "out.mp4")

        with (
            patch("pipeline_core.render_modes.validate_input",
                  return_value=_make_validation_ok()),
            patch("pipeline_core.render_modes._ffmpeg_slice", return_value=None),
            patch("pipeline_core.render_modes.apply_cta",
                  return_value=_make_cta_result(out_path)),
        ):
            result = render_mode_clip(
                fake_video,
                candidate,
                mode="promo",
                output_path=out_path,
                run_qa=False,
            )

        assert result.duration_s <= 25.0 + 0.5, (
            f"Promo clip duration {result.duration_s} exceeds 25.5 s (max=25 s)"
        )


# ---------------------------------------------------------------------------
# render_mode_from_narrative unit tests
# ---------------------------------------------------------------------------

class TestRenderModeFromNarrative:
    def test_empty_narrative_returns_empty_list(self, tmp_path):
        """Narrative with zero candidates → empty list returned."""
        from pipeline_core.render_modes import render_mode_from_narrative

        narrative = _make_narrative(candidates=[])
        result = render_mode_from_narrative(
            str(tmp_path / "v.mp4"),
            narrative,
            mode="standalone",
            output_dir=str(tmp_path),
        )

        assert result == []

    def test_full_narrative_returns_at_most_one_clip(self, tmp_path):
        """full_narrative mode returns at most 1 RenderedClip even with 5 candidates."""
        from pipeline_core.render_modes import render_mode_from_narrative

        candidates = [
            _make_candidate(start=float(i * 20), end=float(i * 20 + 15))
            for i in range(5)
        ]
        narrative = _make_narrative(candidates=candidates, source_duration=120.0)

        fake_video = str(tmp_path / "v.mp4")
        (tmp_path / "v.mp4").write_bytes(b"fake")

        out_path = str(tmp_path / "full_narrative_01.mp4")

        with (
            patch("pipeline_core.render_modes.validate_input",
                  return_value=_make_validation_ok()),
            patch("pipeline_core.render_modes._ffmpeg_slice", return_value=None),
            patch("pipeline_core.render_modes.apply_cta",
                  return_value=_make_cta_result(out_path)),
        ):
            result = render_mode_from_narrative(
                fake_video,
                narrative,
                mode="full_narrative",
                output_dir=str(tmp_path),
            )

        assert len(result) <= 1, f"full_narrative returned {len(result)} clips (max 1)"

    def test_series_mode_delegates_to_chain_parts(self, tmp_path):
        """mode='series' calls render_series.chain_parts exactly once."""
        from pipeline_core.render_modes import render_mode_from_narrative

        candidates = [
            _make_candidate(start=float(i * 10), end=float(i * 10 + 8))
            for i in range(3)
        ]
        narrative = _make_narrative(candidates=candidates, source_duration=50.0)

        fake_video = str(tmp_path / "v.mp4")
        (tmp_path / "v.mp4").write_bytes(b"fake")

        # Build mock SeriesManifest parts
        mock_part = MagicMock()
        mock_part.output_path = str(tmp_path / "p1.mp4")
        mock_part.duration_s = 8.0
        mock_part.part_index = 1
        mock_part.part_total = 3
        mock_part.cta_applied = "next_part"
        mock_part.qa_ok = True
        mock_part.title_overlay_text = "Part 1/3"
        mock_part.cliffhanger_detected = False

        mock_manifest = MagicMock()
        mock_manifest.parts = [mock_part]
        mock_manifest.warnings = []
        mock_manifest.playlist_title = "Test Series"

        # render_series is imported locally inside render_mode_from_narrative as:
        #   from pipeline_core import render_series as _rs
        # So patch at the source module's chain_parts directly.
        with patch("pipeline_core.render_series.chain_parts",
                   return_value=mock_manifest) as mock_chain:

            render_mode_from_narrative(
                fake_video,
                narrative,
                mode="series",
                output_dir=str(tmp_path),
            )

        mock_chain.assert_called_once()

    def test_render_mode_from_narrative_returns_list(self, tmp_path):
        """render_mode_from_narrative always returns a list."""
        from pipeline_core.render_modes import render_mode_from_narrative

        narrative = _make_narrative(candidates=[_make_candidate()])
        fake_video = str(tmp_path / "v.mp4")
        (tmp_path / "v.mp4").write_bytes(b"fake")
        out_path = str(tmp_path / "standalone_01.mp4")

        with (
            patch("pipeline_core.render_modes.validate_input",
                  return_value=_make_validation_ok()),
            patch("pipeline_core.render_modes._ffmpeg_slice", return_value=None),
            patch("pipeline_core.render_modes.apply_cta",
                  return_value=_make_cta_result(out_path)),
        ):
            result = render_mode_from_narrative(
                fake_video,
                narrative,
                mode="standalone",
                output_dir=str(tmp_path),
            )

        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Slow integration tests  (real ffmpeg)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_render_standalone_end_to_end(valid_long_mp4, tmp_path):
    """Render a 10 s standalone clip from a 30 s source, verify output exists."""
    from pipeline_core.render_modes import render_mode_clip

    candidate = _make_candidate(start=2.0, end=12.0)
    output = str(tmp_path / "standalone_out.mp4")

    result = render_mode_clip(
        valid_long_mp4,
        candidate,
        mode="standalone",
        output_path=output,
        run_qa=False,  # avoid platform checks on 16:9 test source
        cta_text=None,
    )

    assert os.path.isfile(result.output_path), "Standalone output file missing"

    import shutil
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "json", result.output_path],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, f"ffprobe failed: {proc.stderr}"
    data = json.loads(proc.stdout)
    dur = float(data["format"]["duration"])
    assert abs(dur - 10.0) <= 1.5, f"Standalone duration {dur} not near 10 s"


@pytest.mark.slow
def test_render_trailer_end_to_end(valid_long_mp4, tmp_path):
    """Render a 10 s trailer clip from a 30 s source, verify output exists."""
    from pipeline_core.render_modes import render_mode_clip

    candidate = _make_candidate(start=2.0, end=12.0)
    output = str(tmp_path / "trailer_out.mp4")

    result = render_mode_clip(
        valid_long_mp4,
        candidate,
        mode="trailer",
        output_path=output,
        run_qa=False,
    )

    assert os.path.isfile(result.output_path), "Trailer output file missing"

    import shutil
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "json", result.output_path],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    dur = float(data["format"]["duration"])
    assert dur > 0, "Trailer output has zero duration"
