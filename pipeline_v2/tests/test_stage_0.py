"""Tests for Stage 0 (ingest).

Two tiers:

1. **Unit tests** (always run): mock ``subprocess.run`` to assert
   encoder-detection logic, ffprobe parsing, and the Pydantic
   Stage0Output contract. No ffmpeg needed.

2. **Integration tests** (skipped if ffmpeg isn't on PATH): generate a
   short synthetic clip with ffmpeg, run the real ``run_stage_0``
   coroutine, and assert outputs exist with the right shape. Also
   includes the timestamp-preservation click test (audio tone burst at
   t=2.000s in the source -> mezzanine has it within ~33ms).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline_v2.models import Stage0Output
from pipeline_v2.stages.stage_0_ingest import run_stage_0
from pipeline_v2.utils import ffmpeg_runner, ffprobe


# ====================================================================== #
# Unit tests                                                             #
# ====================================================================== #


class TestListEncoders:
    """``list_encoders()`` parses ffmpeg -encoders output."""

    SAMPLE = (
        "Encoders:\n"
        " V..... = Video\n"
        " A..... = Audio\n"
        " ------\n"
        " V....D libx264              libx264 H.264\n"
        " V....D h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)\n"
        " V....D hevc_nvenc           NVIDIA NVENC hevc encoder (codec hevc)\n"
        " A....D aac                  AAC (Advanced Audio Coding)\n"
        " A....D libmp3lame           libmp3lame MP3 (MPEG audio layer 3)\n"
    )

    def test_parses_table_correctly(self):
        with patch("subprocess.run") as srun:
            srun.return_value = MagicMock(returncode=0, stdout=self.SAMPLE, stderr="")
            encoders = ffmpeg_runner.list_encoders()
        assert "libx264" in encoders
        assert "h264_nvenc" in encoders
        assert "hevc_nvenc" in encoders
        assert "aac" in encoders

    def test_ffmpeg_missing_raises_filenotfound(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError, match="ffmpeg is not on PATH"):
                ffmpeg_runner.list_encoders()

    def test_nonzero_exit_raises_runtime(self):
        with patch("subprocess.run") as srun:
            srun.return_value = MagicMock(returncode=1, stdout="", stderr="boom")
            with pytest.raises(RuntimeError, match="ffmpeg -encoders failed"):
                ffmpeg_runner.list_encoders()


class TestNvencRuntimeOk:
    """``nvenc_runtime_ok()`` actually runs an NVENC probe."""

    def test_returns_true_on_zero_exit(self):
        with patch("subprocess.run") as srun:
            srun.return_value = MagicMock(returncode=0, stdout="", stderr="")
            assert ffmpeg_runner.nvenc_runtime_ok() is True

    def test_returns_false_on_nonzero_exit(self):
        # This is the real-world case: NVENC compiled in but no NVIDIA
        # driver -- ffmpeg exits with rc=234 / -22 invalid argument.
        with patch("subprocess.run") as srun:
            srun.return_value = MagicMock(returncode=234, stdout="", stderr="Invalid argument")
            assert ffmpeg_runner.nvenc_runtime_ok() is False

    def test_returns_false_on_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=15)):
            assert ffmpeg_runner.nvenc_runtime_ok() is False

    def test_returns_false_on_missing_ffmpeg(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert ffmpeg_runner.nvenc_runtime_ok() is False


class TestDetectEncoder:
    """``detect_encoder()`` combines the two probes."""

    def test_nvenc_compiled_and_runtime_ok_returns_nvenc(self):
        with patch.object(ffmpeg_runner, "list_encoders",
                          return_value={"libx264", "h264_nvenc", "aac"}), \
             patch.object(ffmpeg_runner, "nvenc_runtime_ok", return_value=True):
            assert ffmpeg_runner.detect_encoder() == "h264_nvenc"

    def test_nvenc_compiled_but_runtime_fails_falls_back(self):
        # This is the user's local-dev case: ffmpeg has NVENC support
        # but the machine has no NVIDIA GPU, so the probe fails.
        with patch.object(ffmpeg_runner, "list_encoders",
                          return_value={"libx264", "h264_nvenc"}), \
             patch.object(ffmpeg_runner, "nvenc_runtime_ok", return_value=False):
            assert ffmpeg_runner.detect_encoder() == "libx264"

    def test_nvenc_not_compiled_falls_back(self):
        # ffmpeg build without NVENC support -- some distro packages.
        with patch.object(ffmpeg_runner, "list_encoders",
                          return_value={"libx264", "aac"}), \
             patch.object(ffmpeg_runner, "nvenc_runtime_ok", return_value=True) as runtime:
            # Should NOT even call the runtime probe when nvenc isn't listed.
            assert ffmpeg_runner.detect_encoder() == "libx264"
            runtime.assert_not_called()


class TestProbeVideo:
    """``ffprobe.probe()`` parses ffprobe JSON."""

    CFR_SAMPLE = {
        "format": {"duration": "120.5"},
        "streams": [
            {
                "codec_type": "video", "codec_name": "h264",
                "avg_frame_rate": "30/1", "r_frame_rate": "30/1",
                "width": 1920, "height": 1080,
            },
            {
                "codec_type": "audio", "codec_name": "aac",
                "sample_rate": "48000", "channels": 2,
            },
        ],
    }

    VFR_SAMPLE = {
        "format": {"duration": "30.0"},
        "streams": [
            {
                "codec_type": "video", "codec_name": "h264",
                "avg_frame_rate": "29950/1000",      # ~29.95 actual
                "r_frame_rate": "30000/1001",        # 29.97 nominal
                "width": 1280, "height": 720,
            }
        ],
    }

    def test_cfr_sample_parses(self):
        p = ffprobe._build_probe(self.CFR_SAMPLE)
        assert p.video_codec == "h264"
        assert p.audio_codec == "aac"
        assert p.fps == 30.0
        assert p.nominal_fps == 30.0
        assert p.width == 1920 and p.height == 1080
        assert p.sample_rate == 48000
        assert p.channels == 2
        assert p.duration_sec == pytest.approx(120.5)
        assert p.nb_streams == 2
        assert p.is_vfr is False

    def test_vfr_sample_detected(self):
        p = ffprobe._build_probe(self.VFR_SAMPLE)
        assert p.is_vfr is True
        # avg ≈ 29.95, nominal ≈ 29.97; well above 0.01 tolerance.

    def test_missing_audio_stream_handled(self):
        sample = {
            "format": {"duration": "5"},
            "streams": [{"codec_type": "video", "codec_name": "h264",
                         "avg_frame_rate": "30/1", "r_frame_rate": "30/1",
                         "width": 640, "height": 480}],
        }
        p = ffprobe._build_probe(sample)
        assert p.audio_codec is None
        assert p.sample_rate is None
        assert p.channels is None
        assert p.is_vfr is False

    def test_zero_denominator_fps_returns_none(self):
        sample = {"format": {"duration": "5"},
                  "streams": [{"codec_type": "video", "codec_name": "h264",
                               "avg_frame_rate": "0/0", "r_frame_rate": "0/0",
                               "width": 640, "height": 480}]}
        p = ffprobe._build_probe(sample)
        assert p.fps is None
        assert p.nominal_fps is None

    def test_probe_invokes_ffprobe_with_json_args(self):
        sample_json = json.dumps(self.CFR_SAMPLE)
        with patch("subprocess.run") as srun:
            srun.return_value = MagicMock(returncode=0, stdout=sample_json, stderr="")
            p = ffprobe.probe("/some/path.mp4")
        args = srun.call_args.args[0]
        assert args[0] == "ffprobe"
        assert "json" in args
        assert "/some/path.mp4" in args
        assert p.duration_sec == 120.5

    def test_ffprobe_missing_raises(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError, match="ffprobe is not on PATH"):
                ffprobe.probe("/x")

    def test_ffprobe_nonzero_raises(self):
        with patch("subprocess.run") as srun:
            srun.return_value = MagicMock(returncode=1, stdout="", stderr="bad input")
            with pytest.raises(RuntimeError, match="ffprobe failed"):
                ffprobe.probe("/x")


class TestStage0OutputModel:
    def test_construct_minimal(self):
        s = Stage0Output(
            mezzanine_path="/tmp/m.mp4",
            audio_path="/tmp/a.mp3",
            duration_sec=120.5,
            encoder_used="libx264",
            width=1920, height=1080,
            source_was_vfr=False,
            transcode_seconds=12.3,
            audio_extract_seconds=4.5,
            wall_seconds=12.4,
        )
        assert s.encoder_used == "libx264"

    def test_rejects_invalid_encoder_literal(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Stage0Output(
                mezzanine_path="x", audio_path="y", duration_sec=1.0,
                encoder_used="qsv",                # not in the Literal set
                source_was_vfr=False,
                transcode_seconds=0, audio_extract_seconds=0, wall_seconds=0,
            )


# ====================================================================== #
# Integration tests (skipped if ffmpeg isn't on PATH)                    #
# ====================================================================== #


HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
skip_if_no_ffmpeg = pytest.mark.skipif(
    not HAS_FFMPEG,
    reason="ffmpeg / ffprobe not on PATH; skipping integration tests",
)


def _ffmpeg_make_fixture(out_path: Path, *, duration_s: float = 5.0,
                         click_at_s: float = 2.0) -> None:
    """Synthesise a test clip with a 100ms audio tone burst at click_at_s.

    Black video at 30fps + silence punctuated by a 1kHz sine wave for
    100ms. Used as both input and 'click' reference for timestamp tests.
    """
    # We build the audio as: silence -> tone -> silence using concat.
    # Easier than mixing with adelay (which has tricky channel-layout
    # behaviour across ffmpeg versions).
    pre = click_at_s
    burst = 0.1
    post = max(0.0, duration_s - pre - burst)
    filter_complex = (
        f"color=c=black:s=320x240:r=30:d={duration_s}[v];"
        f"aevalsrc=0:d={pre}[a1];"
        f"sine=frequency=1000:duration={burst}[a2];"
        f"aevalsrc=0:d={post}[a3];"
        f"[a1][a2][a3]concat=n=3:v=0:a=1[a]"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-y", "-v", "error",
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac", "-ar", "48000", "-ac", "1",
        "-t", str(duration_s),
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"fixture ffmpeg failed: {proc.stderr[-500:]}")


def _detect_click_time(audio_path: Path) -> float:
    """Return the start time of the first non-silent segment, in seconds.

    Uses ffmpeg's silencedetect filter. With a clean tone burst above
    -30 dBFS and ~100ms duration, the first 'silence_end' announcement
    marks where the burst started.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(audio_path),
        "-af", "silencedetect=noise=-40dB:d=0.05",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    # silencedetect logs to stderr. Look for "silence_end: <t>".
    for line in proc.stderr.splitlines():
        if "silence_end:" in line:
            # e.g. "[silencedetect @ ...] silence_end: 2.00012 | silence_duration: 2.0"
            for token in line.split():
                try:
                    if token.endswith("|"):
                        token = token[:-1]
                    t = float(token)
                    if 0.5 < t < 100:    # plausible click time window
                        return t
                except ValueError:
                    continue
    raise AssertionError(
        f"no silence_end found in audio {audio_path}; "
        f"ffmpeg stderr was:\n{proc.stderr[-1000:]}"
    )


@skip_if_no_ffmpeg
class TestStage0Integration:
    """Real ffmpeg, real outputs, real timestamps."""

    @pytest.mark.asyncio
    async def test_round_trip_produces_outputs(self, tmp_path):
        src = tmp_path / "src.mp4"
        _ffmpeg_make_fixture(src, duration_s=3.0)

        out_dir = tmp_path / "out"
        result = await run_stage_0(str(src), str(out_dir))

        assert isinstance(result, Stage0Output)
        assert Path(result.mezzanine_path).is_file()
        assert Path(result.audio_path).is_file()
        # Mezzanine duration should match source within encoder rounding.
        p = ffprobe.probe(result.mezzanine_path)
        assert abs(p.duration_sec - 3.0) < 0.3
        # Encoder choice must be a known literal.
        assert result.encoder_used in ("h264_nvenc", "libx264")

    @pytest.mark.asyncio
    async def test_mezzanine_is_30fps_cfr(self, tmp_path):
        src = tmp_path / "src.mp4"
        _ffmpeg_make_fixture(src, duration_s=2.0)

        out_dir = tmp_path / "out"
        result = await run_stage_0(str(src), str(out_dir))

        p = ffprobe.probe(result.mezzanine_path)
        # avg ≈ nominal (CFR), and both near 30.
        assert p.fps is not None and 29.5 < p.fps < 30.5
        assert p.nominal_fps == pytest.approx(30.0, rel=0.02)
        assert p.is_vfr is False

    @pytest.mark.asyncio
    async def test_audio_is_mp3(self, tmp_path):
        src = tmp_path / "src.mp4"
        _ffmpeg_make_fixture(src, duration_s=2.0)

        out_dir = tmp_path / "out"
        result = await run_stage_0(str(src), str(out_dir))

        p = ffprobe.probe(result.audio_path)
        assert p.audio_codec == "mp3"
        assert p.sample_rate == 48000

    @pytest.mark.asyncio
    async def test_click_at_2s_preserved_within_33ms(self, tmp_path):
        # The plan's smoke test: insert an audible click at a known
        # source time, verify the mezzanine has it within ±33ms (one
        # frame at 30fps).
        click_t = 2.0
        src = tmp_path / "click_src.mp4"
        _ffmpeg_make_fixture(src, duration_s=5.0, click_at_s=click_t)

        # Sanity-check the fixture itself first.
        src_click_t = _detect_click_time(src)
        assert abs(src_click_t - click_t) < 0.05, (
            f"fixture click at {src_click_t:.4f}s, expected {click_t}s"
        )

        out_dir = tmp_path / "out"
        result = await run_stage_0(str(src), str(out_dir))

        mezz_click_t = _detect_click_time(Path(result.mezzanine_path))
        drift = abs(mezz_click_t - click_t)
        assert drift < 0.033, (
            f"click drift {drift*1000:.1f}ms exceeds 33ms tolerance "
            f"(fixture={src_click_t:.4f}s, mezzanine={mezz_click_t:.4f}s)"
        )
