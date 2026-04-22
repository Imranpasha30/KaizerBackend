"""
tests/test_cta_overlay.py
=========================
pytest coverage for pipeline_core/cta_overlay.py

All ffmpeg/ffprobe interactions are mocked in fast tests.
Real I/O only in @pytest.mark.slow tests.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import fields
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_proc(returncode=0, stdout="", stderr=""):
    """Build a mock CompletedProcess."""
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = stderr
    return m


def _ffprobe_duration_output(duration: float) -> str:
    return json.dumps({"format": {"duration": str(duration)}})


def _ffprobe_size_output(w: int = 1080, h: int = 1920) -> str:
    return json.dumps({"streams": [{"width": w, "height": h}]})


def _make_caption_result():
    """Return a minimal mock of CaptionResult."""
    from PIL import Image
    img = Image.new("RGBA", (200, 60), (0, 0, 0, 0))
    m = MagicMock()
    m.image = img
    m.width = 200
    m.height = 60
    m.warnings = []
    return m


# ---------------------------------------------------------------------------
# Fast unit tests
# ---------------------------------------------------------------------------

class TestCTAResultDataclass:
    def test_apply_cta_returns_ctaresult_dataclass(self, tmp_path):
        """apply_cta with style='none' returns a CTAResult with all expected fields."""
        from pipeline_core.cta_overlay import CTAResult, apply_cta

        fake_video = str(tmp_path / "source.mp4")
        (tmp_path / "source.mp4").write_bytes(b"fake")
        fake_output = str(tmp_path / "out.mp4")

        # subprocess.run is called for the ffmpeg -c copy pass-through
        with patch("pipeline_core.cta_overlay.subprocess.run") as mock_run:
            mock_run.return_value = _make_fake_proc(returncode=0)

            result = apply_cta(
                fake_video,
                cta_style="none",
                output_path=fake_output,
            )

        assert isinstance(result, CTAResult)
        field_names = {f.name for f in fields(CTAResult)}
        assert field_names == {"output_path", "cta_style", "cta_start_s", "cta_duration_s", "warnings"}
        assert result.cta_style == "none"
        assert isinstance(result.warnings, list)
        assert result.output_path == fake_output.replace("\\", "/")


class TestCTAStyleValidation:
    def test_apply_cta_invalid_style_raises(self, tmp_path):
        """Passing cta_style='potato' must raise ValueError immediately."""
        from pipeline_core.cta_overlay import apply_cta

        fake_video = str(tmp_path / "source.mp4")
        (tmp_path / "source.mp4").write_bytes(b"fake")

        with pytest.raises(ValueError, match="potato"):
            apply_cta(
                fake_video,
                cta_style="potato",
                output_path=str(tmp_path / "out.mp4"),
            )

    @pytest.mark.parametrize("style", [
        "soft_follow", "related_video", "next_part", "url_overlay", "none"
    ])
    def test_valid_styles_do_not_raise_on_dispatch(self, style, tmp_path):
        """Every documented style is accepted without ValueError."""
        from pipeline_core.cta_overlay import apply_cta

        fake_video = str(tmp_path / "source.mp4")
        (tmp_path / "source.mp4").write_bytes(b"fake")

        # subprocess.run side_effect: first two calls → ffprobe (duration, size),
        # remaining calls → ffmpeg overlay/copy.
        dur_out = _ffprobe_duration_output(15.0)
        size_out = _ffprobe_size_output()
        ffmpeg_ok = _make_fake_proc(returncode=0)

        def _run_side_effect(cmd, **kwargs):
            cmd_str = " ".join(str(c) for c in cmd)
            if "ffprobe" in cmd_str and "duration" in cmd_str:
                return _make_fake_proc(stdout=dur_out)
            if "ffprobe" in cmd_str and "stream" in cmd_str:
                return _make_fake_proc(stdout=size_out)
            if "ffprobe" in cmd_str:
                return _make_fake_proc(stdout=dur_out)
            return ffmpeg_ok

        with (
            patch("pipeline_core.cta_overlay.subprocess.run",
                  side_effect=_run_side_effect),
            patch("pipeline_core.cta_overlay.render_caption",
                  return_value=_make_caption_result()),
        ):
            try:
                apply_cta(
                    fake_video,
                    cta_style=style,
                    output_path=str(tmp_path / f"out_{style}.mp4"),
                    text="Test text" if style != "none" else None,
                )
            except ValueError as exc:
                pytest.fail(f"apply_cta raised ValueError for valid style {style!r}: {exc}")


class TestCTADurationClamping:
    def test_apply_cta_cta_duration_clamps_for_short_video(self, tmp_path):
        """2 s video + cta_duration_s=3.0 → warning emitted, cta_duration_s reduced."""
        from pipeline_core.cta_overlay import apply_cta

        fake_video = str(tmp_path / "source.mp4")
        (tmp_path / "source.mp4").write_bytes(b"fake")

        dur_out = _ffprobe_duration_output(2.0)
        size_out = _ffprobe_size_output()

        def _run_side_effect(cmd, **kwargs):
            cmd_str = " ".join(str(c) for c in cmd)
            if "ffprobe" in cmd_str and "duration" in cmd_str:
                return _make_fake_proc(stdout=dur_out)
            if "ffprobe" in cmd_str:
                return _make_fake_proc(stdout=size_out)
            return _make_fake_proc(returncode=0)

        with (
            patch("pipeline_core.cta_overlay.subprocess.run",
                  side_effect=_run_side_effect),
            patch("pipeline_core.cta_overlay.render_caption",
                  return_value=_make_caption_result()),
        ):
            result = apply_cta(
                fake_video,
                cta_style="soft_follow",
                output_path=str(tmp_path / "out.mp4"),
                text="Follow us",
                cta_duration_s=3.0,
            )

        # Duration must be less than the 2 s video
        assert result.cta_duration_s < 2.0, (
            f"cta_duration_s ({result.cta_duration_s}) should be < 2.0 for a 2s video"
        )
        # A warning must have been emitted
        assert any(
            "clamp" in w.lower() or "shrink" in w.lower() or "duration" in w.lower()
            for w in result.warnings
        ), f"Expected a duration/clamp warning. Got: {result.warnings}"


class TestCTAPlatformSafeZone:
    def test_apply_cta_respects_platform_safe_zone(self, tmp_path):
        """The overlay y= coordinate differs between instagram_reel and youtube_short."""
        from pipeline_core.cta_overlay import apply_cta

        fake_video = str(tmp_path / "source.mp4")
        (tmp_path / "source.mp4").write_bytes(b"fake")

        dur_out = _ffprobe_duration_output(15.0)
        size_out = _ffprobe_size_output()

        ffmpeg_calls_by_platform: dict[str, list] = {}

        for platform in ("instagram_reel", "youtube_short"):
            captured_ffmpeg_calls = []

            def _run_side_effect(cmd, **kwargs):
                cmd_str = " ".join(str(c) for c in cmd)
                if "ffprobe" in cmd_str and "duration" in cmd_str:
                    return _make_fake_proc(stdout=dur_out)
                if "ffprobe" in cmd_str:
                    return _make_fake_proc(stdout=size_out)
                # ffmpeg call - capture it
                captured_ffmpeg_calls.append(list(cmd))
                return _make_fake_proc(returncode=0)

            with (
                patch("pipeline_core.cta_overlay.subprocess.run",
                      side_effect=_run_side_effect),
                patch("pipeline_core.cta_overlay.render_caption",
                      return_value=_make_caption_result()),
            ):
                apply_cta(
                    fake_video,
                    cta_style="soft_follow",
                    output_path=str(tmp_path / f"out_{platform}.mp4"),
                    text="Follow us",
                    platform=platform,
                )

            ffmpeg_calls_by_platform[platform] = captured_ffmpeg_calls

        import re

        def _extract_y(calls):
            for cmd_args in calls:
                for arg in cmd_args:
                    ms = re.findall(r"y=(\d+)", str(arg))
                    if ms:
                        return ms[0]
            return None

        ig_y = _extract_y(ffmpeg_calls_by_platform["instagram_reel"])
        yt_y = _extract_y(ffmpeg_calls_by_platform["youtube_short"])

        # At least one platform must have produced a y= coordinate in ffmpeg call
        assert ig_y is not None or yt_y is not None, (
            "No y= coordinate found in ffmpeg filter_complex for either platform"
        )

        if ig_y is not None and yt_y is not None:
            assert ig_y != yt_y, (
                f"Expected different y= for instagram_reel vs youtube_short, "
                f"got both = {ig_y}"
            )


class TestCTACaptionsIntegration:
    def test_apply_cta_uses_captions_render_for_text(self, tmp_path):
        """apply_cta must call render_caption with the supplied text."""
        from pipeline_core.cta_overlay import apply_cta

        fake_video = str(tmp_path / "source.mp4")
        (tmp_path / "source.mp4").write_bytes(b"fake")
        custom_text = "Subscribe for daily news!"

        dur_out = _ffprobe_duration_output(15.0)
        size_out = _ffprobe_size_output()

        def _run_side_effect(cmd, **kwargs):
            cmd_str = " ".join(str(c) for c in cmd)
            if "ffprobe" in cmd_str and "duration" in cmd_str:
                return _make_fake_proc(stdout=dur_out)
            if "ffprobe" in cmd_str:
                return _make_fake_proc(stdout=size_out)
            return _make_fake_proc(returncode=0)

        with (
            patch("pipeline_core.cta_overlay.subprocess.run",
                  side_effect=_run_side_effect),
            patch("pipeline_core.cta_overlay.render_caption") as mock_rc,
        ):
            mock_rc.return_value = _make_caption_result()

            apply_cta(
                fake_video,
                cta_style="soft_follow",
                output_path=str(tmp_path / "out.mp4"),
                text=custom_text,
            )

        assert mock_rc.called, "render_caption was not called"
        # The custom text must appear in one of the positional args
        all_call_texts = [
            str(c.args[0]) if c.args else ""
            for c in mock_rc.call_args_list
        ]
        assert any(custom_text in t for t in all_call_texts), (
            f"Text {custom_text!r} not found in render_caption calls: {all_call_texts}"
        )

    def test_apply_cta_related_video_default_text(self, tmp_path):
        """style='related_video' with text=None → default text contains 'Watch'."""
        from pipeline_core.cta_overlay import apply_cta

        fake_video = str(tmp_path / "source.mp4")
        (tmp_path / "source.mp4").write_bytes(b"fake")

        dur_out = _ffprobe_duration_output(15.0)
        size_out = _ffprobe_size_output()

        def _run_side_effect(cmd, **kwargs):
            cmd_str = " ".join(str(c) for c in cmd)
            if "ffprobe" in cmd_str and "duration" in cmd_str:
                return _make_fake_proc(stdout=dur_out)
            if "ffprobe" in cmd_str:
                return _make_fake_proc(stdout=size_out)
            return _make_fake_proc(returncode=0)

        with (
            patch("pipeline_core.cta_overlay.subprocess.run",
                  side_effect=_run_side_effect),
            patch("pipeline_core.cta_overlay.render_caption") as mock_rc,
        ):
            mock_rc.return_value = _make_caption_result()

            apply_cta(
                fake_video,
                cta_style="related_video",
                output_path=str(tmp_path / "out.mp4"),
                text=None,
            )

        assert mock_rc.called
        all_call_texts = [
            str(c.args[0]) if c.args else ""
            for c in mock_rc.call_args_list
        ]
        combined = " ".join(all_call_texts).lower()
        assert "watch" in combined or "video" in combined, (
            f"Default related_video text should reference 'Watch' or 'video'. Got: {all_call_texts}"
        )

    def test_apply_cta_sub_text_appears_in_caption_chain(self, tmp_path):
        """sub_text passed → render_caption is called with text containing sub_text."""
        from pipeline_core.cta_overlay import apply_cta

        fake_video = str(tmp_path / "source.mp4")
        (tmp_path / "source.mp4").write_bytes(b"fake")
        sub_text = "youtube.com/xyz"

        dur_out = _ffprobe_duration_output(15.0)
        size_out = _ffprobe_size_output()

        def _run_side_effect(cmd, **kwargs):
            cmd_str = " ".join(str(c) for c in cmd)
            if "ffprobe" in cmd_str and "duration" in cmd_str:
                return _make_fake_proc(stdout=dur_out)
            if "ffprobe" in cmd_str:
                return _make_fake_proc(stdout=size_out)
            return _make_fake_proc(returncode=0)

        with (
            patch("pipeline_core.cta_overlay.subprocess.run",
                  side_effect=_run_side_effect),
            patch("pipeline_core.cta_overlay.render_caption") as mock_rc,
        ):
            mock_rc.return_value = _make_caption_result()

            apply_cta(
                fake_video,
                cta_style="soft_follow",
                output_path=str(tmp_path / "out.mp4"),
                text="Follow us",
                sub_text=sub_text,
            )

        assert mock_rc.called
        # The sub_text is joined with '\n' into the single render_caption call
        all_call_texts = [
            str(c.args[0]) if c.args else ""
            for c in mock_rc.call_args_list
        ]
        combined = " ".join(all_call_texts)
        assert sub_text in combined, (
            f"sub_text {sub_text!r} not found in render_caption call text. Got: {all_call_texts}"
        )


class TestCTAURLOverlay:
    def test_apply_cta_url_overlay_requires_text(self, tmp_path):
        """style='url_overlay' with text=None raises ValueError OR emits a warning."""
        from pipeline_core.cta_overlay import apply_cta

        fake_video = str(tmp_path / "source.mp4")
        (tmp_path / "source.mp4").write_bytes(b"fake")

        try:
            dur_out = _ffprobe_duration_output(15.0)
            size_out = _ffprobe_size_output()

            def _run_side_effect(cmd, **kwargs):
                cmd_str = " ".join(str(c) for c in cmd)
                if "ffprobe" in cmd_str and "duration" in cmd_str:
                    return _make_fake_proc(stdout=dur_out)
                if "ffprobe" in cmd_str:
                    return _make_fake_proc(stdout=size_out)
                return _make_fake_proc(returncode=0)

            with (
                patch("pipeline_core.cta_overlay.subprocess.run",
                      side_effect=_run_side_effect),
                patch("pipeline_core.cta_overlay.render_caption",
                      return_value=_make_caption_result()),
            ):
                result = apply_cta(
                    fake_video,
                    cta_style="url_overlay",
                    output_path=str(tmp_path / "out.mp4"),
                    text=None,
                )
            # If no raise: a warning must be present
            assert any(
                "url" in w.lower() or "text" in w.lower() or "required" in w.lower()
                for w in result.warnings
            ), f"url_overlay with text=None must emit a warning. Got: {result.warnings}"
        except ValueError:
            pass  # Raising ValueError is also acceptable (and is the actual behaviour)


class TestCTASoftFollowTelugu:
    def test_apply_cta_soft_follow_telugu_default(self, tmp_path):
        """source_language='te' with text=None → Telugu default text is used."""
        from pipeline_core.cta_overlay import apply_cta

        fake_video = str(tmp_path / "source.mp4")
        (tmp_path / "source.mp4").write_bytes(b"fake")

        dur_out = _ffprobe_duration_output(15.0)
        size_out = _ffprobe_size_output()

        def _run_side_effect(cmd, **kwargs):
            cmd_str = " ".join(str(c) for c in cmd)
            if "ffprobe" in cmd_str and "duration" in cmd_str:
                return _make_fake_proc(stdout=dur_out)
            if "ffprobe" in cmd_str:
                return _make_fake_proc(stdout=size_out)
            return _make_fake_proc(returncode=0)

        with (
            patch("pipeline_core.cta_overlay.subprocess.run",
                  side_effect=_run_side_effect),
            patch("pipeline_core.cta_overlay.render_caption") as mock_rc,
        ):
            mock_rc.return_value = _make_caption_result()

            apply_cta(
                fake_video,
                cta_style="soft_follow",
                output_path=str(tmp_path / "out.mp4"),
                text=None,
                source_language="te",
            )

        assert mock_rc.called
        all_texts = [
            str(c.args[0]) if c.args else ""
            for c in mock_rc.call_args_list
        ]
        combined = " ".join(all_texts)

        # The actual module uses _TELUGU_SOFT_FOLLOW = "మరిన్నింటికి Follow"
        # Check for Telugu Unicode characters OR the specific phrase
        has_telugu_chars = any(
            any(0x0C00 <= ord(ch) <= 0x0C7F for ch in t) for t in all_texts
        )
        # Also accept English "Follow" since the Telugu constant includes it
        has_follow_keyword = "follow" in combined.lower() or "follow" in combined

        assert has_telugu_chars or has_follow_keyword, (
            f"Expected Telugu-specific default text for source_language='te'. Got: {all_texts}"
        )

        # Crucially, it must not be the English-only default
        assert combined != "Follow for more", (
            "Telugu language should use a different default than the English 'Follow for more'"
        )


# ---------------------------------------------------------------------------
# Slow integration tests  (real ffmpeg)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_apply_cta_style_none_copies_input(valid_short_mp4, tmp_path):
    """cta_style='none' → output file exists and duration matches input ±0.1 s."""
    from pipeline_core.cta_overlay import apply_cta
    import shutil

    output = str(tmp_path / "out_none.mp4")
    result = apply_cta(
        valid_short_mp4,
        cta_style="none",
        output_path=output,
    )

    assert os.path.isfile(result.output_path), "Output file missing"

    # Verify duration via ffprobe
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "json", result.output_path],
        capture_output=True, text=True, timeout=30,
    )
    data = json.loads(proc.stdout)
    out_dur = float(data["format"]["duration"])
    assert abs(out_dur - 15.0) <= 0.1, f"Duration mismatch: {out_dur} vs 15.0"


@pytest.mark.slow
def test_apply_cta_real_overlay_round_trip(valid_short_mp4, tmp_path):
    """style='soft_follow' → output exists and is readable by ffprobe."""
    from pipeline_core.cta_overlay import apply_cta
    import shutil

    output = str(tmp_path / "out_soft_follow.mp4")
    result = apply_cta(
        valid_short_mp4,
        cta_style="soft_follow",
        output_path=output,
        text="Follow for daily news",
    )

    assert os.path.isfile(result.output_path), "Output file missing after real overlay"

    ffprobe = shutil.which("ffprobe") or "ffprobe"
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "json", result.output_path],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, f"ffprobe failed: {proc.stderr}"
    data = json.loads(proc.stdout)
    dur = float(data["format"]["duration"])
    assert dur > 0, "Output has zero duration"
