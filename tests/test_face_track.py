"""
tests/test_face_track.py — Phase 2B TDD coverage for pipeline_core/face_track.py.

The module under test does NOT exist yet; every test is expected to fail with
ImportError until the Builder ships the code.

Fixtures are local to this file.  `valid_short_mp4` (15s 1080x1920 30fps) is
reused from conftest.py.

Notes on tolerance:
  - "≈ total frames": ffprobe reports frame count; we allow ±2 for any
    off-by-one between seek and decode at boundaries.
  - "within smoothing kernel": detections+predictions may differ from
    trajectory length by up to smooth_window frames.
"""
from __future__ import annotations

import math
import os
import subprocess
import statistics
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------

def _run_ffmpeg(*args: str, output_path: str) -> str:
    cmd = ["ffmpeg", "-y", *args, output_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except FileNotFoundError:
        pytest.skip("ffmpeg not available")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed:\n  cmd: {' '.join(cmd)}\n"
            f"  stderr: {exc.stderr.decode(errors='replace')}"
        ) from exc
    return output_path


@pytest.fixture(scope="module")
def _facetrack_tmpdir():
    with tempfile.TemporaryDirectory(prefix="kaizer_facetrack_test_") as d:
        yield d


@pytest.fixture(scope="module")
def no_face_mp4(_facetrack_tmpdir):
    """8-second 1920x1080 solid-grey video — no visual content the Haar cascade
    could mistake for a face.  A uniform colour frame produces zero detections
    reliably, which triggers the 'no face' fallback branch and warning in
    compute_crop_trajectory."""
    p = os.path.join(_facetrack_tmpdir, "no_face.mp4")
    return _run_ffmpeg(
        "-f", "lavfi", "-i", "color=c=gray:size=1920x1080:rate=30:duration=8",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-an",
        "-t", "8",
        output_path=p,
    )


@pytest.fixture(scope="module")
def output_mp4(_facetrack_tmpdir):
    """Reusable path for apply_crop output."""
    return os.path.join(_facetrack_tmpdir, "cropped_output.mp4")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_face_track():
    import importlib
    return importlib.import_module("pipeline_core.face_track")


def _probe_frames(video_path: str) -> int:
    """Return total frame count via ffprobe -count_packets for speed."""
    r = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0",
            video_path,
        ],
        capture_output=True, text=True, timeout=30,
    )
    out = r.stdout.strip()
    if not out:
        # Fallback: derive from duration * fps
        r2 = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries",
             "format=duration", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=15,
        )
        r3 = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=15,
        )
        dur = float(r2.stdout.strip() or 0)
        fps_str = r3.stdout.strip() or "30/1"
        num, _, den = fps_str.partition("/")
        fps = float(num) / float(den or 1)
        return int(dur * fps)
    return int(out)


def _probe_dimensions(video_path: str) -> tuple[int, int]:
    """Return (width, height) of the first video stream."""
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path],
        capture_output=True, text=True, timeout=15,
    )
    w_str, _, h_str = r.stdout.strip().partition(",")
    return int(w_str), int(h_str)


# ---------------------------------------------------------------------------
# compute_crop_trajectory — 9 tests
# ---------------------------------------------------------------------------

class TestComputeCropTrajectory:
    def test_compute_trajectory_returns_tracking_result(self, valid_short_mp4):
        """Function must return a TrackingResult dataclass with required attributes."""
        ft = _import_face_track()
        result = ft.compute_crop_trajectory(valid_short_mp4)
        assert hasattr(result, "trajectory"),  "Missing .trajectory"
        assert hasattr(result, "detections"),  "Missing .detections"
        assert hasattr(result, "predictions"), "Missing .predictions"
        assert hasattr(result, "warnings"),    "Missing .warnings"
        assert isinstance(result.trajectory, list)
        assert isinstance(result.detections, int)
        assert isinstance(result.predictions, int)
        assert isinstance(result.warnings, list)

    @pytest.mark.slow
    def test_trajectory_length_matches_frame_count(self, valid_short_mp4):
        """len(trajectory) must be within ±2 of total video frame count."""
        ft = _import_face_track()
        result = ft.compute_crop_trajectory(valid_short_mp4)
        total_frames = _probe_frames(valid_short_mp4)
        diff = abs(len(result.trajectory) - total_frames)
        assert diff <= 2, (
            f"trajectory length {len(result.trajectory)} differs from "
            f"frame count {total_frames} by {diff} (tolerance: 2)"
        )

    def test_trajectory_boxes_have_integer_coords(self, valid_short_mp4):
        """Every CropBox must have integer x, y, w, h."""
        ft = _import_face_track()
        result = ft.compute_crop_trajectory(valid_short_mp4)
        assert len(result.trajectory) > 0, "trajectory must be non-empty"
        for i, box in enumerate(result.trajectory):
            for attr in ("x", "y", "w", "h"):
                val = getattr(box, attr)
                assert isinstance(val, int), (
                    f"trajectory[{i}].{attr} = {val!r} is not int "
                    f"(type={type(val).__name__})"
                )

    def test_trajectory_boxes_within_frame_bounds(self, valid_short_mp4):
        """Every crop box must stay within the source video frame dimensions."""
        ft = _import_face_track()
        result = ft.compute_crop_trajectory(valid_short_mp4)
        frame_w, frame_h = _probe_dimensions(valid_short_mp4)
        for i, box in enumerate(result.trajectory):
            assert box.x >= 0,               f"[{i}] x={box.x} < 0"
            assert box.y >= 0,               f"[{i}] y={box.y} < 0"
            assert box.w > 0,                f"[{i}] w={box.w} ≤ 0"
            assert box.h > 0,                f"[{i}] h={box.h} ≤ 0"
            assert box.x + box.w <= frame_w, (
                f"[{i}] x+w={box.x+box.w} > frame_w={frame_w}"
            )
            assert box.y + box.h <= frame_h, (
                f"[{i}] y+h={box.y+box.h} > frame_h={frame_h}"
            )

    def test_target_aspect_9_16_box_ratio(self, valid_short_mp4):
        """Median (w/h) of 9:16 trajectory boxes must be close to 9/16 = 0.5625 ±0.05."""
        ft = _import_face_track()
        result = ft.compute_crop_trajectory(valid_short_mp4, target_aspect="9:16")
        ratios = [box.w / box.h for box in result.trajectory if box.h > 0]
        assert ratios, "No trajectory boxes to inspect"
        median_ratio = statistics.median(ratios)
        expected = 9 / 16
        assert abs(median_ratio - expected) <= 0.05, (
            f"Median w/h ratio {median_ratio:.4f} deviates from {expected:.4f} "
            f"by more than 0.05"
        )

    def test_target_aspect_1_1_box_ratio(self, valid_short_mp4):
        """Median (w/h) of 1:1 trajectory boxes must be ≈1.0 ±0.05."""
        ft = _import_face_track()
        result = ft.compute_crop_trajectory(valid_short_mp4, target_aspect="1:1")
        ratios = [box.w / box.h for box in result.trajectory if box.h > 0]
        assert ratios, "No trajectory boxes to inspect"
        median_ratio = statistics.median(ratios)
        assert abs(median_ratio - 1.0) <= 0.05, (
            f"Median w/h ratio {median_ratio:.4f} deviates from 1.0 by more than 0.05"
        )

    def test_detections_and_predictions_sum_to_trajectory_length_approx(
        self, valid_short_mp4
    ):
        """detections + predictions must be within ±smooth_window of trajectory length."""
        smooth_window = 15  # matches the default in the spec
        ft = _import_face_track()
        result = ft.compute_crop_trajectory(valid_short_mp4, smooth_window=smooth_window)
        total = result.detections + result.predictions
        traj_len = len(result.trajectory)
        diff = abs(total - traj_len)
        assert diff <= smooth_window, (
            f"detections({result.detections}) + predictions({result.predictions}) = {total}, "
            f"trajectory length = {traj_len}; difference {diff} > smooth_window={smooth_window}"
        )

    def test_no_face_in_video_still_returns_trajectory(self, no_face_mp4):
        """A video with no face must not crash; should return a centred fallback
        trajectory and a non-empty warnings list."""
        ft = _import_face_track()
        result = ft.compute_crop_trajectory(no_face_mp4, target_aspect="9:16")
        assert isinstance(result.trajectory, list), "Must return a trajectory list"
        assert len(result.trajectory) > 0, "Trajectory must not be empty for a valid video"
        assert len(result.warnings) > 0, (
            "At least one warning expected when no face detected "
            "(e.g. 'no face detected, using centre crop')"
        )

    def test_invalid_aspect_raises(self, valid_short_mp4):
        """Passing an unrecognisable aspect string must raise ValueError."""
        ft = _import_face_track()
        with pytest.raises(ValueError):
            ft.compute_crop_trajectory(valid_short_mp4, target_aspect="potato")


# ---------------------------------------------------------------------------
# apply_crop — 1 test (v1 implementation)
# ---------------------------------------------------------------------------

class TestApplyCrop:
    def test_apply_crop_writes_output_file(self, valid_short_mp4, output_mp4):
        """apply_crop must produce a non-empty MP4 file at output_path."""
        ft = _import_face_track()
        # First build a trajectory from the source video
        result = ft.compute_crop_trajectory(valid_short_mp4, target_aspect="9:16")
        assert result.trajectory, "Need a non-empty trajectory to test apply_crop"

        # Ensure we use a fresh path in a writable temp directory
        import tempfile
        with tempfile.TemporaryDirectory(prefix="kaizer_apply_crop_") as td:
            out = os.path.join(td, "cropped.mp4")
            returned_path = ft.apply_crop(
                valid_short_mp4,
                result.trajectory,
                output_path=out,
                target_size=(1080, 1920),
            )
            assert os.path.exists(returned_path), (
                f"apply_crop returned {returned_path!r} but file does not exist"
            )
            assert os.path.getsize(returned_path) > 0, (
                "apply_crop produced an empty file"
            )
