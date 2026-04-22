"""
conftest.py — shared pytest fixtures for Phase 1 pipeline tests.

All video fixtures are generated at session scope into a single temp directory
so that the expensive FFmpeg synthesis runs only once per test session.
Each fixture returns a str path.

FFmpeg availability is checked once; if unavailable the fixtures skip.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ffmpeg(*args: str, output_path: str) -> str:
    """Run ffmpeg with the given args, writing to output_path.

    Raises pytest.skip if ffmpeg is not callable.
    Raises subprocess.CalledProcessError if the command fails.
    Returns output_path on success.
    """
    cmd = ["ffmpeg", "-y", *args, output_path]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError:
        pytest.skip("ffmpeg not available on PATH")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed (exit {exc.returncode}):\n"
            f"  cmd : {' '.join(cmd)}\n"
            f"  stderr: {exc.stderr.decode(errors='replace')}"
        ) from exc
    return output_path


# ---------------------------------------------------------------------------
# Session-scoped temp directory
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tmp_video_dir():
    """A single temporary directory that lives for the whole test session."""
    with tempfile.TemporaryDirectory(prefix="kaizer_test_") as d:
        yield d


# ---------------------------------------------------------------------------
# Video fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def valid_short_mp4(tmp_video_dir):
    """
    15 s, 1080x1920, H.264, 30 fps, AAC audio, ~8 Mbps, bt709, yuv420p.
    Suitable for youtube_short / instagram_reel / tiktok QA checks.
    """
    out = os.path.join(tmp_video_dir, "valid_short.mp4")
    return _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=duration=15:size=1080x1920:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=15",
        "-c:v", "libx264",
        "-preset", "fast",
        "-b:v", "8M",
        "-maxrate", "10M",
        "-bufsize", "16M",
        "-pix_fmt", "yuv420p",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-movflags", "+faststart",
        "-t", "15",
        output_path=out,
    )


@pytest.fixture(scope="session")
def valid_long_mp4(tmp_video_dir):
    """
    30 s, 1920x1080, H.264, 30 fps, AAC, bt709, yuv420p — 16:9 for youtube_long.
    """
    out = os.path.join(tmp_video_dir, "valid_long.mp4")
    return _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=duration=30:size=1920x1080:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=30",
        "-c:v", "libx264",
        "-preset", "fast",
        "-b:v", "8M",
        "-maxrate", "10M",
        "-bufsize", "16M",
        "-pix_fmt", "yuv420p",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-movflags", "+faststart",
        "-t", "30",
        output_path=out,
    )


@pytest.fixture(scope="session")
def oversized_mp4(tmp_video_dir):
    """
    5 s, 3840x2160 (4K) — triggers resolution-too-large check.
    """
    out = os.path.join(tmp_video_dir, "oversized.mp4")
    return _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=duration=5:size=3840x2160:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-t", "5",
        output_path=out,
    )


@pytest.fixture(scope="session")
def mid_duration_mp4(tmp_video_dir):
    """
    10 s, 1080x1920 — for validator boundary tests.
    """
    out = os.path.join(tmp_video_dir, "mid_duration.mp4")
    return _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=duration=10:size=1080x1920:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-t", "10",
        output_path=out,
    )


@pytest.fixture(scope="session")
def overly_long_mp4(tmp_video_dir):
    """
    200 s, 1080x1920 — exceeds 3-minute platform max for short-form platforms.
    We use -shortest + loop tricks to keep encoding fast by looping 10s content.
    """
    out = os.path.join(tmp_video_dir, "overly_long.mp4")
    return _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=duration=200:size=1080x1920:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=200",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-t", "200",
        output_path=out,
    )


@pytest.fixture(scope="session")
def no_audio_mp4(tmp_video_dir):
    """
    15 s, 1080x1920, H.264, no audio stream — should produce a warning not an error.
    """
    out = os.path.join(tmp_video_dir, "no_audio.mp4")
    return _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=duration=15:size=1080x1920:rate=30",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-an",
        "-t", "15",
        output_path=out,
    )


@pytest.fixture(scope="session")
def corrupt_mp4(tmp_video_dir):
    """
    A file with .mp4 extension but garbage/text contents — not a valid video.
    """
    out = os.path.join(tmp_video_dir, "corrupt.mp4")
    with open(out, "wb") as f:
        f.write(
            b"This is not a valid MP4 file.\n"
            b"It contains only garbage data to simulate a corrupt upload.\n"
            b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f" * 64
        )
    return out


@pytest.fixture(scope="session")
def missing_path():
    """
    A path that doesn't exist on disk.
    """
    return "/nonexistent/path/does_not_exist_kaizer_test.mp4"


# ---------------------------------------------------------------------------
# Additional QA-specific fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def wrong_aspect_mp4(tmp_video_dir):
    """
    15 s, 1920x1080 (16:9) — wrong aspect ratio for short-form platforms (expects 9:16).
    """
    out = os.path.join(tmp_video_dir, "wrong_aspect.mp4")
    return _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=duration=15:size=1920x1080:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=15",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-color_primaries", "bt709",
        "-c:a", "aac",
        "-b:a", "128k",
        "-t", "15",
        output_path=out,
    )


@pytest.fixture(scope="session")
def wrong_pixfmt_mp4(tmp_video_dir):
    """
    10 s, 1080x1920, yuv422p — wrong pixel format (QA expects yuv420p).
    """
    out = os.path.join(tmp_video_dir, "wrong_pixfmt.mp4")
    return _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=duration=10:size=1080x1920:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv422p",
        "-c:a", "aac",
        "-t", "10",
        output_path=out,
    )


@pytest.fixture(scope="session")
def silent_mp4(tmp_video_dir):
    """
    10 s, 1080x1920, with a near-silent audio track (anullsrc at very low volume).
    Used to test loudness-too-quiet warnings.
    """
    out = os.path.join(tmp_video_dir, "silent.mp4")
    return _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=duration=10:size=1080x1920:rate=30",
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-t", "10",
        output_path=out,
    )


@pytest.fixture(scope="session")
def loud_mp4(tmp_video_dir):
    """
    10 s, 1080x1920, audio at approx -3 LUFS (very loud sine wave).
    Used to test loudness-too-loud warnings.
    """
    out = os.path.join(tmp_video_dir, "loud.mp4")
    return _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=duration=10:size=1080x1920:rate=30",
        # High-amplitude sine to push LUFS close to -3
        "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=10",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        # volume filter to boost loudness significantly above -14 LUFS
        "-af", "volume=20dB",
        "-t", "10",
        output_path=out,
    )


@pytest.fixture(scope="session")
def short_duration_mp4(tmp_video_dir):
    """
    1 s clip — below platform minimum duration (3 s for short-form).
    """
    out = os.path.join(tmp_video_dir, "short_duration.mp4")
    return _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=duration=1:size=1080x1920:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-t", "1",
        output_path=out,
    )
