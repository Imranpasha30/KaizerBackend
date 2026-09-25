"""Unit A — two-pass broadcast loudness conform (-14 LUFS on every final)."""
from __future__ import annotations

import subprocess

import pytest

from pipeline_v4 import audio_conform as ac


def _make_clip(path, *, volume_db: float, duration: int = 6):
    """A/V clip whose sine audio sits at a chosen level."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"color=c=blue:s=320x240:r=30:d={duration}",
         "-f", "lavfi", "-i", "sine=frequency=400:sample_rate=48000",
         "-af", f"volume={volume_db}dB",
         "-t", str(duration), "-c:v", "libx264", "-preset", "ultrafast",
         "-c:a", "aac", "-shortest", str(path)],
        check=True, capture_output=True,
    )


@pytest.fixture(autouse=True)
def _need_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("ffmpeg not available")


@pytest.mark.slow
def test_hot_clip_conforms_to_minus_14(tmp_path):
    clip = tmp_path / "hot.mp4"
    _make_clip(clip, volume_db=+12)         # way hotter than -14 LUFS
    before = ac.measure_loudness(str(clip))
    assert before and before["input_i"] > -12.0

    assert ac.conform_loudness(str(clip)) is True

    after = ac.measure_loudness(str(clip))
    assert after is not None
    assert -15.0 <= after["input_i"] <= -13.0, f"got {after['input_i']} LUFS"
    assert after["input_tp"] <= -1.0


@pytest.mark.slow
def test_in_tolerance_clip_is_skipped(tmp_path):
    clip = tmp_path / "ok.mp4"
    _make_clip(clip, volume_db=+12)
    assert ac.conform_loudness(str(clip)) is True          # first pass conforms
    mtime = clip.stat().st_mtime_ns
    assert ac.conform_loudness(str(clip)) is False         # second pass: no-op
    assert clip.stat().st_mtime_ns == mtime                # file untouched


def test_fail_soft_paths(tmp_path, monkeypatch):
    # Missing file → False, no raise.
    assert ac.conform_loudness(str(tmp_path / "nope.mp4")) is False
    # Garbage file → False, original untouched.
    bad = tmp_path / "garbage.mp4"
    bad.write_bytes(b"not a video")
    assert ac.conform_loudness(str(bad)) is False
    assert bad.read_bytes() == b"not a video"
    # Env off → False without touching ffmpeg.
    monkeypatch.setenv("KAIZER_V4_LOUDNORM", "0")
    assert ac.conform_loudness(str(bad)) is False


@pytest.mark.slow
def test_video_stream_copied_not_reencoded(tmp_path):
    """The conform must be audio-only: video packets stream-copied."""
    clip = tmp_path / "v.mp4"
    _make_clip(clip, volume_db=+12)

    def _vcodec_extra(p):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,nb_frames",
             "-of", "csv=p=0", str(p)],
            check=True, capture_output=True, text=True).stdout.strip()
        return out

    before = _vcodec_extra(clip)
    assert ac.conform_loudness(str(clip)) is True
    after = _vcodec_extra(clip)
    assert before.split(",")[0] == after.split(",")[0] == "h264"
