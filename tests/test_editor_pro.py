"""
tests/test_editor_pro.py
========================
pytest coverage for pipeline_core/effects/* and pipeline_core/editor_pro.py.

Fast tests (no ffmpeg I/O) cover:
  - All dataclass shapes (TransitionSpec, TextAnimationSpec, MotionSpec, StylePack)
  - All SUPPORTED_* constant lengths and values
  - Error raises (ValueError on bad names)
  - list_style_packs / get_style_pack contract
  - render_animation_frames correctness (frame count, type) for all 5 animations
  - render_beta orchestration (mocked ffmpeg) — chain ordering, result shape
  - render_beta with 'minimal' pack skips color + motion stages
  - render_beta effects_applied ordered list
  - render_beta on stage failure records warning but still produces beta_path
  - render_both_versions returns both paths populated
  - apply_color_grade 'none' → copy (mock subprocess)
  - apply_color_grade 'invalid' → ValueError
  - apply_transition invalid → ValueError
  - apply_motion invalid → ValueError

Slow integration tests (real ffmpeg, marked @pytest.mark.slow):
  - render_beta('minimal') on valid_short_mp4: beta_path exists + QA passes
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from dataclasses import fields
from unittest.mock import MagicMock, patch, call
from typing import Any

import pytest
from PIL import Image


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_fake_proc(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    """Return a mock subprocess.CompletedProcess."""
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = stderr
    return m


def _ffprobe_duration_json(duration: float) -> str:
    import json
    return json.dumps({"format": {"duration": str(duration)}})


# ══════════════════════════════════════════════════════════════════════════════
# 1. Dataclass shape tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTransitionSpecDataclass:
    def test_fields_exist(self):
        from pipeline_core.effects.transitions import TransitionSpec
        field_names = {f.name for f in fields(TransitionSpec)}
        assert field_names >= {"name", "duration_s", "params"}

    def test_defaults(self):
        from pipeline_core.effects.transitions import TransitionSpec
        spec = TransitionSpec(name="fade")
        assert spec.duration_s == 0.5
        assert spec.params == {}

    def test_custom_values(self):
        from pipeline_core.effects.transitions import TransitionSpec
        spec = TransitionSpec(name="slide_left", duration_s=0.3, params={"easing": "ease_in"})
        assert spec.name == "slide_left"
        assert spec.duration_s == 0.3
        assert spec.params["easing"] == "ease_in"


class TestTextAnimationSpecDataclass:
    def test_fields_exist(self):
        from pipeline_core.effects.text_animations import TextAnimationSpec
        field_names = {f.name for f in fields(TextAnimationSpec)}
        assert field_names >= {
            "name", "text", "duration_s", "start_s",
            "font_size", "color", "position", "platform",
        }

    def test_defaults(self):
        from pipeline_core.effects.text_animations import TextAnimationSpec
        spec = TextAnimationSpec(
            name="bounce_in", text="Hello", duration_s=1.5, start_s=0.0,
            font_size=72, color="#FFFFFF", position=(100, 1600),
        )
        assert spec.platform == "youtube_short"
        assert spec.params == {}


class TestMotionSpecDataclass:
    def test_fields_exist(self):
        from pipeline_core.effects.motion import MotionSpec
        field_names = {f.name for f in fields(MotionSpec)}
        assert field_names >= {"name", "duration_s", "intensity", "focal_point"}

    def test_defaults(self):
        from pipeline_core.effects.motion import MotionSpec
        spec = MotionSpec(name="ken_burns_in", duration_s=10.0)
        assert spec.intensity == 0.08
        assert spec.focal_point == (0.5, 0.5)


class TestStylePackDataclass:
    def test_fields_exist(self):
        from pipeline_core.effects.style_packs import StylePack
        field_names = {f.name for f in fields(StylePack)}
        assert field_names >= {
            "name", "label", "description",
            "transition", "transition_duration_s",
            "color_preset", "motion", "motion_intensity",
            "text_animation", "caption_animation",
        }

    def test_optional_motion(self):
        from pipeline_core.effects.style_packs import StylePack
        pack = StylePack(
            name="test", label="Test", description="test",
            transition="fade", color_preset="none", motion=None,
        )
        assert pack.motion is None


class TestBetaRenderResultDataclass:
    def test_fields_exist(self):
        from pipeline_core.editor_pro import BetaRenderResult
        field_names = {f.name for f in fields(BetaRenderResult)}
        assert field_names >= {
            "current_path", "beta_path", "style_pack",
            "effects_applied", "render_time_s", "qa_ok", "warnings",
        }

    def test_defaults(self):
        from pipeline_core.editor_pro import BetaRenderResult
        r = BetaRenderResult(
            current_path="/a.mp4",
            beta_path="/b.mp4",
            style_pack="cinematic",
        )
        assert r.effects_applied == []
        assert r.warnings == []
        assert r.qa_ok is True
        assert r.render_time_s == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 2. SUPPORTED_* constant tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSupportedConstants:
    def test_supported_transitions_count(self):
        from pipeline_core.effects.transitions import SUPPORTED_TRANSITIONS
        assert len(SUPPORTED_TRANSITIONS) == 6

    def test_supported_transitions_values(self):
        from pipeline_core.effects.transitions import SUPPORTED_TRANSITIONS
        expected = {"fade", "slide_left", "slide_up", "zoom_punch", "whip_pan", "dissolve"}
        assert set(SUPPORTED_TRANSITIONS) == expected

    def test_supported_animations_count(self):
        from pipeline_core.effects.text_animations import SUPPORTED_ANIMATIONS
        assert len(SUPPORTED_ANIMATIONS) == 5

    def test_supported_animations_values(self):
        from pipeline_core.effects.text_animations import SUPPORTED_ANIMATIONS
        expected = {"typewriter", "word_pop", "bounce_in", "slide_up", "karaoke"}
        assert set(SUPPORTED_ANIMATIONS) == expected

    def test_supported_motions_count(self):
        from pipeline_core.effects.motion import SUPPORTED_MOTIONS
        assert len(SUPPORTED_MOTIONS) == 4

    def test_supported_motions_values(self):
        from pipeline_core.effects.motion import SUPPORTED_MOTIONS
        expected = {"ken_burns_in", "ken_burns_out", "parallax_still", "zoom_focus"}
        assert set(SUPPORTED_MOTIONS) == expected


# ══════════════════════════════════════════════════════════════════════════════
# 3. ValueError on bad names
# ══════════════════════════════════════════════════════════════════════════════

class TestValueErrors:
    def test_apply_transition_invalid_name(self, tmp_path):
        from pipeline_core.effects.transitions import TransitionSpec, apply_transition
        spec = TransitionSpec(name="warp_speed")
        with pytest.raises(ValueError, match="warp_speed"):
            apply_transition(
                str(tmp_path / "a.mp4"),
                str(tmp_path / "b.mp4"),
                transition=spec,
                output_path=str(tmp_path / "out.mp4"),
            )

    def test_apply_color_grade_invalid_preset(self, tmp_path):
        from pipeline_core.effects.color_grade import apply_color_grade
        with pytest.raises(ValueError, match="not_a_preset"):
            apply_color_grade(
                str(tmp_path / "a.mp4"),
                preset="not_a_preset",
                output_path=str(tmp_path / "out.mp4"),
            )

    def test_apply_motion_invalid_name(self, tmp_path):
        from pipeline_core.effects.motion import MotionSpec, apply_motion
        spec = MotionSpec(name="warp_zoom", duration_s=5.0)
        with pytest.raises(ValueError, match="warp_zoom"):
            apply_motion(
                str(tmp_path / "a.mp4"),
                spec,
                output_path=str(tmp_path / "out.mp4"),
            )

    def test_render_animation_frames_invalid_name(self):
        from pipeline_core.effects.text_animations import (
            TextAnimationSpec, render_animation_frames,
        )
        spec = TextAnimationSpec(
            name="flying_text", text="hi", duration_s=1.0, start_s=0.0,
            font_size=48, color="#FFF", position=(100, 100),
        )
        with pytest.raises(ValueError, match="flying_text"):
            render_animation_frames(spec)

    def test_get_style_pack_invalid_name(self):
        from pipeline_core.effects.style_packs import get_style_pack
        with pytest.raises(ValueError, match="xray_vision"):
            get_style_pack("xray_vision")


# ══════════════════════════════════════════════════════════════════════════════
# 4. StylePack registry tests
# ══════════════════════════════════════════════════════════════════════════════

class TestStylePacks:
    def test_list_style_packs_returns_five(self):
        from pipeline_core.effects.style_packs import list_style_packs
        packs = list_style_packs()
        assert len(packs) == 5

    def test_all_pack_names_resolvable(self):
        from pipeline_core.effects.style_packs import list_style_packs, get_style_pack
        for pack in list_style_packs():
            resolved = get_style_pack(pack.name)
            assert resolved.name == pack.name

    def test_all_packs_have_required_attributes(self):
        from pipeline_core.effects.style_packs import list_style_packs
        from pipeline_core.effects.transitions import SUPPORTED_TRANSITIONS
        from pipeline_core.effects.text_animations import SUPPORTED_ANIMATIONS
        from pipeline_core.effects.color_grade import COLOR_PRESETS
        from pipeline_core.effects.motion import SUPPORTED_MOTIONS

        for pack in list_style_packs():
            assert pack.transition in SUPPORTED_TRANSITIONS, (
                f"Pack {pack.name!r}: transition {pack.transition!r} not in SUPPORTED_TRANSITIONS"
            )
            assert pack.color_preset in COLOR_PRESETS, (
                f"Pack {pack.name!r}: color_preset {pack.color_preset!r} not in COLOR_PRESETS"
            )
            if pack.motion is not None:
                assert pack.motion in SUPPORTED_MOTIONS, (
                    f"Pack {pack.name!r}: motion {pack.motion!r} not in SUPPORTED_MOTIONS"
                )
            assert pack.text_animation in SUPPORTED_ANIMATIONS, (
                f"Pack {pack.name!r}: text_animation {pack.text_animation!r} not in SUPPORTED_ANIMATIONS"
            )
            assert pack.caption_animation in SUPPORTED_ANIMATIONS, (
                f"Pack {pack.name!r}: caption_animation {pack.caption_animation!r} not in SUPPORTED_ANIMATIONS"
            )
            assert isinstance(pack.transition_duration_s, float)
            assert pack.transition_duration_s > 0

    def test_expected_pack_names_present(self):
        from pipeline_core.effects.style_packs import STYLE_PACKS
        assert set(STYLE_PACKS.keys()) == {"minimal", "cinematic", "news_flash", "vibrant", "calm"}

    def test_minimal_pack_has_no_motion(self):
        from pipeline_core.effects.style_packs import get_style_pack
        pack = get_style_pack("minimal")
        assert pack.motion is None
        assert pack.color_preset == "none"

    def test_cinematic_pack_configuration(self):
        from pipeline_core.effects.style_packs import get_style_pack
        pack = get_style_pack("cinematic")
        assert pack.color_preset == "cinematic_warm"
        assert pack.motion == "ken_burns_in"
        assert pack.transition == "fade"

    def test_news_flash_pack_configuration(self):
        from pipeline_core.effects.style_packs import get_style_pack
        pack = get_style_pack("news_flash")
        assert pack.color_preset == "news_red"
        assert pack.transition == "whip_pan"

    def test_pack_labels_non_empty(self):
        from pipeline_core.effects.style_packs import list_style_packs
        for pack in list_style_packs():
            assert pack.label, f"Pack {pack.name!r} has empty label"
            assert pack.description, f"Pack {pack.name!r} has empty description"


# ══════════════════════════════════════════════════════════════════════════════
# 5. render_animation_frames tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRenderAnimationFrames:
    """Test frame count and type for all 5 animations (mocked render_caption)."""

    def _make_spec(self, name: str, dur: float = 1.5) -> Any:
        from pipeline_core.effects.text_animations import TextAnimationSpec
        return TextAnimationSpec(
            name=name,
            text="Breaking News Today",
            duration_s=dur,
            start_s=0.0,
            font_size=48,
            color="#FFFFFF",
            position=(100, 1400),
        )

    def _mock_render_caption(self):
        """Return a mock CaptionResult with a small white RGBA image."""
        img = Image.new("RGBA", (400, 60), (255, 255, 255, 200))
        m = MagicMock()
        m.image = img
        m.width = 400
        m.height = 60
        m.warnings = []
        return m

    def _run(self, name: str, dur: float = 1.5, fps: int = 30) -> list:
        from pipeline_core.effects.text_animations import render_animation_frames
        spec = self._make_spec(name, dur)
        with patch(
            "pipeline_core.effects.text_animations.render_caption",
            return_value=self._mock_render_caption(),
        ):
            frames = render_animation_frames(spec, canvas_size=(1080, 1920), fps=fps)
        return frames

    def test_typewriter_frame_count(self):
        frames = self._run("typewriter", dur=1.0, fps=30)
        assert len(frames) == math.ceil(1.0 * 30)

    def test_word_pop_frame_count(self):
        frames = self._run("word_pop", dur=2.0, fps=30)
        assert len(frames) == math.ceil(2.0 * 30)

    def test_bounce_in_frame_count(self):
        frames = self._run("bounce_in", dur=1.5, fps=30)
        assert len(frames) == math.ceil(1.5 * 30)

    def test_slide_up_frame_count(self):
        frames = self._run("slide_up", dur=0.9, fps=24)
        assert len(frames) == math.ceil(0.9 * 24)

    def test_karaoke_frame_count(self):
        from pipeline_core.effects.text_animations import TextAnimationSpec, render_animation_frames
        spec = TextAnimationSpec(
            name="karaoke",
            text="hello world test",
            duration_s=2.0,
            start_s=0.0,
            font_size=48,
            color="#FFFFFF",
            position=(100, 1400),
            params={"word_timings": [
                {"word": "hello", "start": 0.0},
                {"word": "world", "start": 0.5},
                {"word": "test",  "start": 1.0},
            ]},
        )
        with patch(
            "pipeline_core.effects.text_animations.render_caption",
            return_value=self._mock_render_caption(),
        ):
            frames = render_animation_frames(spec, canvas_size=(1080, 1920), fps=30)
        assert len(frames) == math.ceil(2.0 * 30)

    def test_all_frames_are_rgba_images(self):
        for anim in ("typewriter", "word_pop", "bounce_in", "slide_up"):
            frames = self._run(anim, dur=0.5, fps=30)
            for f in frames:
                assert isinstance(f, Image.Image)
                assert f.mode == "RGBA"

    def test_frames_have_correct_canvas_size(self):
        frames = self._run("bounce_in", dur=0.5, fps=30)
        for f in frames:
            assert f.size == (1080, 1920)

    def test_non_integer_duration_ceiling(self):
        frames = self._run("typewriter", dur=1.333, fps=30)
        expected = math.ceil(1.333 * 30)
        assert len(frames) == expected


# ══════════════════════════════════════════════════════════════════════════════
# 6. apply_color_grade 'none' mock test
# ══════════════════════════════════════════════════════════════════════════════

class TestColorGradeMocked:
    def test_none_preset_calls_copy(self, tmp_path):
        """apply_color_grade('none') uses -c copy, not re-encode."""
        from pipeline_core.effects.color_grade import apply_color_grade

        src = tmp_path / "input.mp4"
        src.write_bytes(b"fake")
        out = tmp_path / "out.mp4"

        with patch("pipeline_core.effects.color_grade.subprocess.run") as mock_run:
            mock_run.return_value = _make_fake_proc(returncode=0)
            apply_color_grade(str(src), preset="none", output_path=str(out))

        call_args = mock_run.call_args[0][0]
        assert "-c" in call_args
        copy_idx = call_args.index("-c")
        assert call_args[copy_idx + 1] == "copy"

    def test_cinematic_warm_includes_eq_filter(self, tmp_path):
        """apply_color_grade('cinematic_warm') passes -vf with eq filter."""
        from pipeline_core.effects.color_grade import apply_color_grade

        src = tmp_path / "input.mp4"
        src.write_bytes(b"fake")
        out = tmp_path / "out.mp4"

        with patch("pipeline_core.effects.color_grade.subprocess.run") as mock_run:
            mock_run.return_value = _make_fake_proc(returncode=0)
            apply_color_grade(str(src), preset="cinematic_warm", output_path=str(out))

        call_args = mock_run.call_args[0][0]
        assert "-vf" in call_args
        vf_idx = call_args.index("-vf")
        vf_val = call_args[vf_idx + 1]
        assert "eq=" in vf_val

    def test_none_preset_output_path_returned(self, tmp_path):
        from pipeline_core.effects.color_grade import apply_color_grade

        src = tmp_path / "input.mp4"
        src.write_bytes(b"fake")
        out = str(tmp_path / "graded.mp4")

        with patch("pipeline_core.effects.color_grade.subprocess.run") as mock_run:
            mock_run.return_value = _make_fake_proc(returncode=0)
            result = apply_color_grade(str(src), preset="none", output_path=out)

        assert result == out.replace("\\", "/")


# ══════════════════════════════════════════════════════════════════════════════
# 7. render_beta orchestration tests (all ffmpeg calls mocked)
# ══════════════════════════════════════════════════════════════════════════════

class TestRenderBetaMocked:
    """All FFmpeg / file I/O is mocked; tests chain ordering and result shape."""

    def _setup_mock_effects(self, mocker, tmp_dir: str):
        """Patch all effect functions to be no-ops that return their output_path."""

        def _fake_effect(input_path, *args, output_path: str, **kwargs) -> str:
            # Create a real (empty) file so _safe_copy works
            open(output_path, "wb").close()
            return output_path

        def _fake_motion_effect(input_path, spec, *, output_path: str, **kwargs) -> str:
            open(output_path, "wb").close()
            return output_path

        mocker.patch(
            "pipeline_core.editor_pro.apply_color_grade",
            side_effect=_fake_effect,
        )
        mocker.patch(
            "pipeline_core.editor_pro.apply_motion",
            side_effect=_fake_motion_effect,
        )
        mocker.patch(
            "pipeline_core.editor_pro.apply_text_animation",
            side_effect=_fake_effect,
        )
        # Patch QA to always pass
        mocker.patch(
            "pipeline_core.editor_pro._qa",
            create=True,
        )

    def test_render_beta_returns_betarenderresult(self, mocker, tmp_path):
        from pipeline_core.editor_pro import render_beta, BetaRenderResult

        src = tmp_path / "master.mp4"
        src.write_bytes(b"fake_video")
        out_dir = str(tmp_path / "output")

        self._setup_mock_effects(mocker, out_dir)
        mocker.patch("pipeline_core.editor_pro._probe_duration", return_value=15.0)
        mocker.patch("pipeline_core.editor_pro.run_qa", create=True)

        # Patch QA import inside render_beta
        qa_mock = MagicMock()
        qa_result = MagicMock()
        qa_result.ok = True
        qa_result.warnings = []
        qa_result.errors = []
        qa_mock.validate_output.return_value = qa_result
        mocker.patch.dict("sys.modules", {"pipeline_core.qa": qa_mock})

        result = render_beta(str(src), style_pack="cinematic", output_dir=out_dir, run_qa=False)

        assert isinstance(result, BetaRenderResult)
        assert result.current_path == str(src).replace("\\", "/")
        assert result.beta_path.endswith(".mp4")
        assert result.style_pack == "cinematic"
        assert isinstance(result.effects_applied, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.render_time_s, float)
        assert result.render_time_s >= 0.0

    def test_render_beta_cinematic_applies_color_and_motion(self, mocker, tmp_path):
        from pipeline_core.editor_pro import render_beta

        src = tmp_path / "master.mp4"
        src.write_bytes(b"fake_video")
        out_dir = str(tmp_path / "output")

        self._setup_mock_effects(mocker, out_dir)
        mocker.patch("pipeline_core.editor_pro._probe_duration", return_value=15.0)

        result = render_beta(str(src), style_pack="cinematic", output_dir=out_dir, run_qa=False)

        # cinematic pack has color_preset='cinematic_warm' and motion='ken_burns_in'
        assert any("color_grade" in e for e in result.effects_applied)
        assert any("motion" in e for e in result.effects_applied)

    def test_render_beta_minimal_skips_color_and_motion(self, mocker, tmp_path):
        from pipeline_core.editor_pro import render_beta

        src = tmp_path / "master.mp4"
        src.write_bytes(b"fake_video")
        out_dir = str(tmp_path / "output")

        self._setup_mock_effects(mocker, out_dir)
        mocker.patch("pipeline_core.editor_pro._probe_duration", return_value=15.0)

        result = render_beta(str(src), style_pack="minimal", output_dir=out_dir, run_qa=False)

        # minimal pack has color_preset='none' and motion=None → both skipped
        assert not any("color_grade" in e for e in result.effects_applied)
        assert not any("motion" in e for e in result.effects_applied)

    def test_render_beta_effects_applied_ordered(self, mocker, tmp_path):
        from pipeline_core.editor_pro import render_beta

        src = tmp_path / "master.mp4"
        src.write_bytes(b"fake_video")
        out_dir = str(tmp_path / "output")

        self._setup_mock_effects(mocker, out_dir)
        mocker.patch("pipeline_core.editor_pro._probe_duration", return_value=15.0)

        result = render_beta(
            str(src),
            style_pack="cinematic",
            hook_text="Breaking News",
            output_dir=out_dir,
            run_qa=False,
        )

        # Check ordering: color always before motion before text_animation
        positions = {effect.split(":")[0]: i for i, effect in enumerate(result.effects_applied)}
        if "color_grade" in positions and "motion" in positions:
            assert positions["color_grade"] < positions["motion"]
        if "motion" in positions and "text_animation" in positions:
            assert positions["motion"] < positions["text_animation"]

    def test_render_beta_with_hook_text_applies_text_animation(self, mocker, tmp_path):
        from pipeline_core.editor_pro import render_beta

        src = tmp_path / "master.mp4"
        src.write_bytes(b"fake_video")
        out_dir = str(tmp_path / "output")

        self._setup_mock_effects(mocker, out_dir)
        mocker.patch("pipeline_core.editor_pro._probe_duration", return_value=15.0)

        result = render_beta(
            str(src),
            style_pack="cinematic",
            hook_text="Markets Crash",
            output_dir=out_dir,
            run_qa=False,
        )
        assert any("text_animation" in e for e in result.effects_applied)

    def test_render_beta_with_caption_timings_applies_caption_animation(self, mocker, tmp_path):
        from pipeline_core.editor_pro import render_beta

        src = tmp_path / "master.mp4"
        src.write_bytes(b"fake_video")
        out_dir = str(tmp_path / "output")

        self._setup_mock_effects(mocker, out_dir)
        mocker.patch("pipeline_core.editor_pro._probe_duration", return_value=15.0)

        timings = [
            {"word": "hello", "start": 0.0},
            {"word": "world", "start": 0.5},
        ]
        result = render_beta(
            str(src),
            style_pack="cinematic",
            caption_word_timings=timings,
            output_dir=out_dir,
            run_qa=False,
        )
        assert any("caption_animation" in e for e in result.effects_applied)

    def test_render_beta_stage_failure_records_warning_and_produces_output(self, mocker, tmp_path):
        from pipeline_core.editor_pro import render_beta

        src = tmp_path / "master.mp4"
        src.write_bytes(b"fake_video")
        out_dir = str(tmp_path / "output")

        # Make color_grade raise
        mocker.patch(
            "pipeline_core.editor_pro.apply_color_grade",
            side_effect=RuntimeError("Simulated FFmpeg failure"),
        )

        def _ok_motion(input_path, spec, *, output_path: str, **kwargs) -> str:
            open(output_path, "wb").close()
            return output_path

        mocker.patch("pipeline_core.editor_pro.apply_motion", side_effect=_ok_motion)
        mocker.patch("pipeline_core.editor_pro.apply_text_animation",
                     side_effect=lambda i, s, *, output_path, **kw: (
                         open(output_path, "wb").close() or output_path
                     ))
        mocker.patch("pipeline_core.editor_pro._probe_duration", return_value=15.0)

        result = render_beta(
            str(src), style_pack="cinematic", output_dir=out_dir, run_qa=False
        )

        # Should still produce a beta_path
        assert result.beta_path.endswith(".mp4")
        # Warning should mention the failure
        assert any("color_grade" in w or "Simulated" in w for w in result.warnings)
        # color_grade should NOT be in effects_applied
        assert not any("color_grade" in e for e in result.effects_applied)

    def test_render_beta_unknown_pack_records_warning(self, mocker, tmp_path):
        from pipeline_core.editor_pro import render_beta

        src = tmp_path / "master.mp4"
        src.write_bytes(b"fake_video")
        out_dir = str(tmp_path / "output")

        mocker.patch("pipeline_core.editor_pro._probe_duration", return_value=15.0)

        result = render_beta(str(src), style_pack="does_not_exist", output_dir=out_dir, run_qa=False)

        assert result.beta_path.endswith(".mp4")
        assert len(result.warnings) > 0
        assert result.effects_applied == []

    def test_render_both_versions_returns_both_paths(self, mocker, tmp_path):
        from pipeline_core.editor_pro import render_both_versions

        src = tmp_path / "master.mp4"
        src.write_bytes(b"fake_video")
        out_dir = str(tmp_path / "output")

        self._setup_mock_effects(mocker, out_dir)
        mocker.patch("pipeline_core.editor_pro._probe_duration", return_value=15.0)

        # QA mock
        qa_mock = MagicMock()
        qa_result = MagicMock()
        qa_result.ok = True
        qa_result.warnings = []
        qa_result.errors = []
        qa_mock.validate_output.return_value = qa_result
        mocker.patch.dict("sys.modules", {"pipeline_core.qa": qa_mock})

        result = render_both_versions(str(src), style_pack="minimal", output_dir=out_dir)

        assert result.current_path
        assert result.beta_path
        assert result.current_path == str(src).replace("\\", "/")
        assert result.beta_path.endswith(".mp4")

    def test_render_beta_beta_path_always_exists_on_disk(self, mocker, tmp_path):
        from pipeline_core.editor_pro import render_beta

        src = tmp_path / "master.mp4"
        src.write_bytes(b"fake_video")
        out_dir = str(tmp_path / "output")

        # All effects fail
        mocker.patch(
            "pipeline_core.editor_pro.apply_color_grade",
            side_effect=RuntimeError("fail"),
        )
        mocker.patch(
            "pipeline_core.editor_pro.apply_motion",
            side_effect=RuntimeError("fail"),
        )
        mocker.patch(
            "pipeline_core.editor_pro.apply_text_animation",
            side_effect=RuntimeError("fail"),
        )
        mocker.patch("pipeline_core.editor_pro._probe_duration", return_value=15.0)

        result = render_beta(
            str(src), style_pack="cinematic", hook_text="Test", output_dir=out_dir, run_qa=False
        )
        assert os.path.exists(result.beta_path), "beta_path must exist even when all stages fail"


# ══════════════════════════════════════════════════════════════════════════════
# 8. COLOR_PRESETS content tests
# ══════════════════════════════════════════════════════════════════════════════

class TestColorPresets:
    def test_none_preset_is_none(self):
        from pipeline_core.effects.color_grade import COLOR_PRESETS
        assert COLOR_PRESETS["none"] is None

    def test_all_expected_presets_present(self):
        from pipeline_core.effects.color_grade import COLOR_PRESETS
        expected = {"none", "cinematic_warm", "cool_blue", "vintage", "news_red", "vibrant"}
        assert set(COLOR_PRESETS.keys()) == expected

    def test_cinematic_warm_has_saturation_and_contrast(self):
        from pipeline_core.effects.color_grade import COLOR_PRESETS
        p = COLOR_PRESETS["cinematic_warm"]
        assert p is not None
        assert "saturation" in p
        assert "contrast" in p

    def test_news_red_has_channel_boost(self):
        from pipeline_core.effects.color_grade import COLOR_PRESETS
        p = COLOR_PRESETS["news_red"]
        assert p is not None
        assert "channel_boost" in p
        assert "r" in p["channel_boost"]


# ══════════════════════════════════════════════════════════════════════════════
# 9. Slow integration tests
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestRenderBetaIntegration:
    """Real ffmpeg I/O — skipped unless ffmpeg is available."""

    def test_render_beta_minimal_produces_valid_file(self, valid_short_mp4, tmp_path):
        """render_beta with 'minimal' pack on a real MP4 → beta_path exists + QA passes."""
        from pipeline_core.editor_pro import render_beta

        out_dir = str(tmp_path / "beta_output")

        result = render_beta(
            valid_short_mp4,
            style_pack="minimal",
            output_dir=out_dir,
            platform="youtube_short",
            run_qa=True,
        )

        # beta_path must exist
        assert os.path.exists(result.beta_path), (
            f"beta_path {result.beta_path!r} does not exist on disk"
        )

        # Must be a non-empty file
        assert os.path.getsize(result.beta_path) > 0

        # current_path must equal the input
        assert result.current_path == valid_short_mp4.replace("\\", "/")

        # For minimal pack: no color or motion effects
        assert not any("color_grade" in e for e in result.effects_applied)
        assert not any("motion" in e for e in result.effects_applied)

        # QA should pass (minimal pack = stream copy)
        assert result.qa_ok, (
            f"QA failed for minimal beta render. Warnings: {result.warnings}"
        )
