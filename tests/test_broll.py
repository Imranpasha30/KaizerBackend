"""
tests/test_broll.py — Phase 2B TDD coverage for pipeline_core/broll.py.

The module under test does NOT exist yet; every test is expected to fail
with ImportError until the Builder ships the code.  pytest.mark.xfail is NOT
used here — we want visible FAILURES so the Builder has clear signal.  Each
test is independent and self-contained.

Synthesised B-roll assets:
  - broll_jpg   : 1-frame JPEG produced via PIL (no FFmpeg required)
  - broll_mp4   : 3-second black 640×360 MP4 produced via FFmpeg lavfi
  - silent_source: 4-second silent 1080×1920 MP4 (< 6 s after padding guard
                    = no room for insertions)
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Local fixtures — kept in this file, not conftest.py
# ---------------------------------------------------------------------------

def _run_ffmpeg(*args: str, output_path: str) -> str:
    """Thin wrapper around ffmpeg; skips the test if ffmpeg is unavailable."""
    cmd = ["ffmpeg", "-y", *args, output_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    except FileNotFoundError:
        pytest.skip("ffmpeg not available")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed (exit {exc.returncode}):\n"
            f"  cmd   : {' '.join(cmd)}\n"
            f"  stderr: {exc.stderr.decode(errors='replace')}"
        ) from exc
    return output_path


@pytest.fixture(scope="module")
def _broll_tmpdir():
    with tempfile.TemporaryDirectory(prefix="kaizer_broll_test_") as d:
        yield d


@pytest.fixture(scope="module")
def broll_jpg(_broll_tmpdir):
    """Single-frame JPEG (1280×720, solid red) created via PIL."""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")
    p = os.path.join(_broll_tmpdir, "broll_frame.jpg")
    img = Image.new("RGB", (1280, 720), color=(200, 50, 50))
    img.save(p, format="JPEG", quality=90)
    return p


@pytest.fixture(scope="module")
def broll_mp4(_broll_tmpdir):
    """3-second black 640×360 H.264 MP4 produced via FFmpeg lavfi."""
    p = os.path.join(_broll_tmpdir, "broll_clip.mp4")
    return _run_ffmpeg(
        "-f", "lavfi", "-i", "color=black:size=640x360:rate=30",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac", "-b:a", "96k",
        "-t", "3",
        output_path=p,
    )


@pytest.fixture(scope="module")
def short_source_mp4(_broll_tmpdir):
    """4-second 1080×1920 MP4 — shorter than the 6 s minimum threshold
    (first 3 s guard + last 2 s guard > 4 s).
    """
    p = os.path.join(_broll_tmpdir, "short_source.mp4")
    return _run_ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=duration=4:size=1080x1920:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-t", "4",
        output_path=p,
    )


@pytest.fixture(scope="module")
def silent_source_mp4(_broll_tmpdir):
    """15-second 1080×1920 near-silent MP4 (anullsrc)."""
    p = os.path.join(_broll_tmpdir, "silent_source.mp4")
    return _run_ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=duration=15:size=1080x1920:rate=30",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k",
        "-t", "15",
        output_path=p,
    )


# ---------------------------------------------------------------------------
# Helper: lazy-import the module under test
# ---------------------------------------------------------------------------

def _import_broll():
    """Import pipeline_core.broll; raises ImportError if not yet written."""
    import importlib
    return importlib.import_module("pipeline_core.broll")


# ---------------------------------------------------------------------------
# find_audio_valleys — 5 tests
# ---------------------------------------------------------------------------

class TestFindAudioValleys:
    def test_find_audio_valleys_returns_list(self, valid_short_mp4):
        """On a valid 15s clip, function returns a list of (float, float) tuples."""
        broll = _import_broll()
        result = broll.find_audio_valleys(valid_short_mp4)
        assert isinstance(result, list), "Expected list"
        for item in result:
            assert isinstance(item, tuple) and len(item) == 2, (
                f"Each entry must be a 2-tuple, got {item!r}"
            )

    def test_find_audio_valleys_respects_top_k(self, valid_short_mp4):
        """Passing top_k=3 returns at most 3 valleys."""
        broll = _import_broll()
        result = broll.find_audio_valleys(valid_short_mp4, top_k=3)
        assert len(result) <= 3, f"Expected ≤3 entries, got {len(result)}"

    def test_find_audio_valleys_min_gap_enforced(self, valid_short_mp4):
        """Consecutive valley timestamps are at least min_gap_s apart."""
        broll = _import_broll()
        min_gap = 2.0
        valleys = broll.find_audio_valleys(valid_short_mp4, min_gap_s=min_gap, top_k=10)
        timestamps = sorted(t for t, _ in valleys)
        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i - 1]
            assert gap >= min_gap - 1e-6, (
                f"Consecutive valleys too close: {timestamps[i-1]:.3f}s and "
                f"{timestamps[i]:.3f}s (gap={gap:.3f}s < min_gap={min_gap}s)"
            )

    def test_find_audio_valleys_values_in_range(self, valid_short_mp4):
        """Each (t, rms): 0 ≤ t ≤ video_duration, 0 ≤ rms ≤ 1."""
        broll = _import_broll()
        valleys = broll.find_audio_valleys(valid_short_mp4, top_k=5)
        # Get video duration via ffprobe
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", valid_short_mp4],
                capture_output=True, text=True, timeout=15,
            )
            duration = float(r.stdout.strip())
        except Exception:
            duration = 15.0  # fallback from fixture spec

        for t, rms in valleys:
            assert 0.0 <= t <= duration + 0.1, f"Valley time {t:.3f} out of range [0, {duration:.1f}]"
            assert 0.0 <= rms <= 1.0, f"Valley RMS {rms} out of range [0, 1]"

    def test_find_audio_valleys_on_silent_video_returns_valleys(
        self, silent_source_mp4
    ):
        """Silent video must not crash; it may return an empty list or valid valleys."""
        broll = _import_broll()
        result = broll.find_audio_valleys(silent_source_mp4, top_k=5)
        # Must return a list — empty is acceptable for silent content
        assert isinstance(result, list), "Should return a list even for silent video"
        # Any returned items must still be valid 2-tuples of floats
        for item in result:
            assert isinstance(item, tuple) and len(item) == 2, (
                f"Invalid valley format: {item!r}"
            )


# ---------------------------------------------------------------------------
# insert_broll — 7 tests
# ---------------------------------------------------------------------------

class TestInsertBroll:
    def _get_video_duration(self, path: str) -> float:
        """Use ffprobe to measure actual video duration in seconds."""
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30,
        )
        return float(r.stdout.strip())

    def test_insert_broll_writes_output_file(
        self, valid_short_mp4, broll_mp4, tmp_path
    ):
        """After insert_broll completes the output_path must exist on disk."""
        broll = _import_broll()
        out = str(tmp_path / "out_insert.mp4")
        result = broll.insert_broll(
            valid_short_mp4,
            [broll_mp4],
            output_path=out,
            max_inserts=1,
        )
        assert os.path.exists(result.output_path), (
            f"output_path {result.output_path!r} was not created"
        )
        assert result.output_path == out, (
            "BRollResult.output_path must equal the requested output_path"
        )

    @pytest.mark.slow
    def test_insert_broll_preserves_duration_approximately(
        self, valid_short_mp4, broll_mp4, tmp_path
    ):
        """Output duration matches source duration ±0.5s (B-roll replaces video track;
        audio is untouched so total runtime is preserved)."""
        broll = _import_broll()
        out = str(tmp_path / "out_duration.mp4")
        broll.insert_broll(
            valid_short_mp4,
            [broll_mp4],
            output_path=out,
            max_inserts=2,
        )
        src_dur = self._get_video_duration(valid_short_mp4)
        out_dur = self._get_video_duration(out)
        assert abs(out_dur - src_dur) <= 0.5, (
            f"Duration mismatch: source={src_dur:.3f}s, output={out_dur:.3f}s "
            f"(delta={abs(out_dur-src_dur):.3f}s > 0.5s)"
        )

    def test_insert_broll_no_insert_in_first_3s(
        self, valid_short_mp4, broll_mp4, tmp_path
    ):
        """No B-roll insertion should start before t=3.0s."""
        broll = _import_broll()
        out = str(tmp_path / "out_no_early.mp4")
        result = broll.insert_broll(
            valid_short_mp4,
            [broll_mp4],
            output_path=out,
            max_inserts=5,
        )
        for ins in result.insertions:
            assert ins.start_s >= 3.0, (
                f"Insertion starts at {ins.start_s:.3f}s — violates 3s head guard"
            )

    def test_insert_broll_no_insert_in_last_2s(
        self, valid_short_mp4, broll_mp4, tmp_path
    ):
        """No B-roll insertion should end within the last 2s of the source clip."""
        broll = _import_broll()
        out = str(tmp_path / "out_no_late.mp4")
        result = broll.insert_broll(
            valid_short_mp4,
            [broll_mp4],
            output_path=out,
            max_inserts=5,
        )
        src_dur = self._get_video_duration(valid_short_mp4)
        for ins in result.insertions:
            end_s = ins.start_s + ins.duration_s
            assert end_s <= src_dur - 2.0 + 1e-6, (
                f"Insertion ends at {end_s:.3f}s but source ends at {src_dur:.3f}s "
                f"(tail guard violated by {end_s - (src_dur - 2.0):.3f}s)"
            )

    def test_insert_broll_max_inserts_respected(
        self, valid_short_mp4, broll_mp4, tmp_path
    ):
        """max_inserts=2 must produce ≤2 BRollInsertion entries."""
        broll = _import_broll()
        out = str(tmp_path / "out_max2.mp4")
        result = broll.insert_broll(
            valid_short_mp4,
            [broll_mp4],
            output_path=out,
            max_inserts=2,
        )
        assert len(result.insertions) <= 2, (
            f"max_inserts=2 but got {len(result.insertions)} insertions"
        )

    def test_insert_broll_returns_empty_insertions_on_short_clip(
        self, short_source_mp4, broll_mp4, tmp_path
    ):
        """A 4-second source has no room for insertions (3s head + 2s tail = 5s guard
        > 4s total).  insertions must be empty; no exception raised."""
        broll = _import_broll()
        out = str(tmp_path / "out_short_source.mp4")
        result = broll.insert_broll(
            short_source_mp4,
            [broll_mp4],
            output_path=out,
            max_inserts=3,
        )
        assert isinstance(result.insertions, list), "insertions must be a list"
        assert len(result.insertions) == 0, (
            f"Expected 0 insertions for a 4s source, got {len(result.insertions)}"
        )

    def test_insert_broll_with_image_asset(
        self, valid_short_mp4, broll_jpg, tmp_path
    ):
        """Passing a JPEG path as a B-roll asset must not raise; output file written."""
        broll = _import_broll()
        out = str(tmp_path / "out_image_asset.mp4")
        result = broll.insert_broll(
            valid_short_mp4,
            [broll_jpg],
            output_path=out,
            max_inserts=1,
        )
        assert os.path.exists(result.output_path), (
            f"Output not created when image asset provided: {result.output_path!r}"
        )

    # ── Structural / dataclass integrity ────────────────────────────────────

    def test_broll_result_dataclass_fields(
        self, valid_short_mp4, broll_mp4, tmp_path
    ):
        """BRollResult must expose .output_path, .insertions, and .warnings."""
        broll = _import_broll()
        out = str(tmp_path / "out_struct.mp4")
        result = broll.insert_broll(
            valid_short_mp4,
            [broll_mp4],
            output_path=out,
            max_inserts=1,
        )
        assert hasattr(result, "output_path")
        assert hasattr(result, "insertions")
        assert hasattr(result, "warnings")
        assert isinstance(result.warnings, list), ".warnings must be a list"

    def test_broll_insertion_dataclass_fields(
        self, valid_short_mp4, broll_mp4, tmp_path
    ):
        """Each BRollInsertion must have asset_path, start_s, duration_s, source_rms."""
        broll = _import_broll()
        out = str(tmp_path / "out_ins_struct.mp4")
        result = broll.insert_broll(
            valid_short_mp4,
            [broll_mp4],
            output_path=out,
            max_inserts=3,
        )
        # Only validate structure if there are any insertions
        for ins in result.insertions:
            assert hasattr(ins, "asset_path"), "BRollInsertion missing asset_path"
            assert hasattr(ins, "start_s"),    "BRollInsertion missing start_s"
            assert hasattr(ins, "duration_s"), "BRollInsertion missing duration_s"
            assert hasattr(ins, "source_rms"), "BRollInsertion missing source_rms"
            assert isinstance(ins.start_s, (int, float)), "start_s must be numeric"
            assert isinstance(ins.duration_s, (int, float)), "duration_s must be numeric"

    def test_insert_broll_round_robin_rotation(
        self, valid_long_mp4, broll_mp4, broll_jpg, tmp_path
    ):
        """With two assets and max_inserts=4 the assets rotate round-robin
        (asset_path alternates between the two supplied assets)."""
        broll = _import_broll()
        out = str(tmp_path / "out_rr.mp4")
        result = broll.insert_broll(
            valid_long_mp4,
            [broll_mp4, broll_jpg],
            output_path=out,
            max_inserts=4,
        )
        if len(result.insertions) >= 2:
            paths = [ins.asset_path for ins in result.insertions]
            # Consecutive insertions should not all use the same asset when two provided
            pairs = [(paths[i], paths[i + 1]) for i in range(len(paths) - 1)]
            all_same = all(a == b for a, b in pairs)
            assert not all_same, (
                f"All insertions use the same asset — round-robin not applied: {paths}"
            )
