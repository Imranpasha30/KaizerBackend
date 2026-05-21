"""Item 117 Phase 2 -- raw timeline extraction tests.

Mostly cmd-shape unit tests (mock subprocess), plus one optional
live-ffmpeg integration test that skips when the test fixture
mezzanine is absent.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_HERE = Path(__file__).resolve()
_PIPELINE_V2_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PIPELINE_V2_ROOT))


def _stub_subprocess_run(*, rc: int = 0, stderr: str = ""):
    """Returns a callable that, when assigned to subprocess.run,
    creates empty output files at the requested paths and returns
    a fake CompletedProcess."""

    def _fake(cmd, *args, **kwargs):
        # Find the output paths: every argument after the last ``-map``
        # block that is NOT a flag is a potential output. Simpler: look
        # for ``.mp4`` / ``.aac`` filename tokens.
        for tok in cmd:
            if isinstance(tok, str) and tok.endswith(".mp4"):
                if "-i" in cmd and cmd.index("-i") + 1 < len(cmd) and cmd[cmd.index("-i") + 1] == tok:
                    continue  # this is the input path
                Path(tok).parent.mkdir(parents=True, exist_ok=True)
                Path(tok).write_bytes(b"x" * 200_000)

        class _Result:
            def __init__(self):
                self.returncode = rc
                self.stdout = ""
                self.stderr = stderr
        return _Result()

    return _fake


# ---------------------------------------------------------------------
# Command-shape tests (no real ffmpeg)
# ---------------------------------------------------------------------


def test_extract_raw_timeline_invokes_ffmpeg_with_multi_output(
    tmp_path, monkeypatch,
):
    """The cmd must contain exactly one -i (single decode) and N
    -map blocks (one per output)."""
    from pipeline_v2.stages import stage_4_raw_extract as rx

    captured = {}

    def _fake_run(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        for tok in cmd:
            if isinstance(tok, str) and tok.endswith(".mp4") and tok != cmd[cmd.index("-i") + 1]:
                Path(tok).parent.mkdir(parents=True, exist_ok=True)
                Path(tok).write_bytes(b"x" * 200_000)
        class _R:
            returncode = 0; stdout = ""; stderr = ""
        return _R()

    monkeypatch.setattr(subprocess, "run", _fake_run)

    def _fake_streams(path, ffprobe_bin):
        return {
            "video": {"duration": 20.0, "nb_frames": 600},
            "audio": {"duration": 20.0, "nb_frames": 940},
        }
    monkeypatch.setattr(rx, "_ffprobe_streams", _fake_streams)

    rx.extract_raw_timeline(
        mezzanine_path="/fake/mezz.mp4",
        bulletin_cuts=[(1.6, 21.6)],
        shorts_cuts=[(10.0, 30.0), (50.0, 70.0)],
        out_dir=str(tmp_path),
        use_nvenc=True,
    )
    cmd = captured["cmd"]
    # Single -i (one input decoded once).
    assert cmd.count("-i") == 1
    # One filter_complex.
    assert cmd.count("-filter_complex") == 1
    # N outputs (1 bulletin + 2 shorts = 3) -> 6 -map flags
    # (one per stream, audio + video).
    assert cmd.count("-map") == 6


def test_extract_raw_timeline_nvenc_args_present_when_requested(
    tmp_path, monkeypatch,
):
    """use_nvenc=True must put NVENC encoder args in the cmd."""
    from pipeline_v2.stages import stage_4_raw_extract as rx

    captured = {}

    def _fake_run(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        for tok in cmd:
            if isinstance(tok, str) and tok.endswith(".mp4") and tok != cmd[cmd.index("-i") + 1]:
                Path(tok).parent.mkdir(parents=True, exist_ok=True)
                Path(tok).write_bytes(b"x" * 200_000)
        class _R: returncode = 0; stdout = ""; stderr = ""
        return _R()
    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(rx, "_ffprobe_streams", lambda p, b: {
        "video": {"duration": 5.0, "nb_frames": 150},
        "audio": {"duration": 5.0, "nb_frames": 235},
    })
    rx.extract_raw_timeline(
        "/fake/m.mp4", [(0, 5)], [], str(tmp_path), use_nvenc=True,
    )
    cmd_str = " ".join(captured["cmd"])
    assert "h264_nvenc" in cmd_str
    assert "libx264" not in cmd_str


def test_extract_raw_timeline_libx264_fallback(tmp_path, monkeypatch):
    """use_nvenc=False -> libx264 args, no NVENC tokens."""
    from pipeline_v2.stages import stage_4_raw_extract as rx
    captured = {}

    def _fake_run(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        for tok in cmd:
            if isinstance(tok, str) and tok.endswith(".mp4") and tok != cmd[cmd.index("-i") + 1]:
                Path(tok).parent.mkdir(parents=True, exist_ok=True)
                Path(tok).write_bytes(b"x" * 200_000)
        class _R: returncode = 0; stdout = ""; stderr = ""
        return _R()
    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(rx, "_ffprobe_streams", lambda p, b: {
        "video": {"duration": 5.0, "nb_frames": 150},
        "audio": {"duration": 5.0, "nb_frames": 235},
    })
    rx.extract_raw_timeline(
        "/fake/m.mp4", [(0, 5)], [], str(tmp_path), use_nvenc=False,
    )
    cmd_str = " ".join(captured["cmd"])
    assert "libx264" in cmd_str
    assert "h264_nvenc" not in cmd_str


def test_extract_raw_timeline_writes_bulletin_and_short_filenames(
    tmp_path, monkeypatch,
):
    """Output filenames must be bulletin_raw.mp4 + short_NN_raw.mp4."""
    from pipeline_v2.stages import stage_4_raw_extract as rx
    monkeypatch.setattr(subprocess, "run", _stub_subprocess_run())
    monkeypatch.setattr(rx, "_ffprobe_streams", lambda p, b: {
        "video": {"duration": 5.0, "nb_frames": 150},
        "audio": {"duration": 5.0, "nb_frames": 235},
    })
    result = rx.extract_raw_timeline(
        "/fake/m.mp4",
        [(0, 5)],
        [(10, 15), (20, 25), (30, 35)],
        str(tmp_path),
        use_nvenc=True,
    )
    paths = sorted(p.name for p in tmp_path.iterdir())
    assert "bulletin_raw.mp4" in paths
    assert "short_01_raw.mp4" in paths
    assert "short_02_raw.mp4" in paths
    assert "short_03_raw.mp4" in paths


# ---------------------------------------------------------------------
# Verification / failure handling
# ---------------------------------------------------------------------


def test_extract_raw_timeline_raises_on_av_drift_exceeding_tolerance(
    tmp_path, monkeypatch,
):
    """If any output has |a-v delta| > 5ms, raise RawExtractError
    BEFORE returning. This is the architectural guarantee."""
    from pipeline_v2.stages import stage_4_raw_extract as rx

    monkeypatch.setattr(subprocess, "run", _stub_subprocess_run())

    # Bulletin clean, short 1 drifts +10ms (over tolerance).
    def _probe(path, ffprobe_bin):
        if "short_01" in path:
            return {
                "video": {"duration": 20.0, "nb_frames": 600},
                "audio": {"duration": 20.010, "nb_frames": 940},
            }
        return {
            "video": {"duration": 20.0, "nb_frames": 600},
            "audio": {"duration": 20.0, "nb_frames": 940},
        }
    monkeypatch.setattr(rx, "_ffprobe_streams", _probe)

    with pytest.raises(rx.RawExtractError, match="exceeds"):
        rx.extract_raw_timeline(
            "/fake/m.mp4",
            [(0, 20)],
            [(30, 50)],
            str(tmp_path),
            use_nvenc=True,
        )


def test_extract_raw_timeline_propagates_ffmpeg_failure(
    tmp_path, monkeypatch,
):
    """ffmpeg rc != 0 -> RawExtractError with stderr tail."""
    from pipeline_v2.stages import stage_4_raw_extract as rx
    monkeypatch.setattr(
        subprocess, "run",
        _stub_subprocess_run(rc=1, stderr="Some ffmpeg error here"),
    )
    monkeypatch.setattr(rx, "_ffprobe_streams", lambda p, b: {})
    with pytest.raises(rx.RawExtractError, match="rc=1"):
        rx.extract_raw_timeline(
            "/fake/m.mp4", [(0, 5)], [], str(tmp_path), use_nvenc=True,
        )


def test_extract_raw_timeline_progress_cb_invoked(tmp_path, monkeypatch):
    """progress_cb (when supplied) is called at sub-phase boundaries."""
    from pipeline_v2.stages import stage_4_raw_extract as rx
    monkeypatch.setattr(subprocess, "run", _stub_subprocess_run())
    monkeypatch.setattr(rx, "_ffprobe_streams", lambda p, b: {
        "video": {"duration": 5.0, "nb_frames": 150},
        "audio": {"duration": 5.0, "nb_frames": 235},
    })
    msgs: list[str] = []
    rx.extract_raw_timeline(
        "/fake/m.mp4", [(0, 5)], [(10, 15)],
        str(tmp_path), use_nvenc=True,
        progress_cb=msgs.append,
    )
    # At least two messages: pre-call (graph size) + post-call (bulletin + shorts).
    assert any("raw-extract" in m and "outputs" in m for m in msgs)
    assert any("bulletin=" in m for m in msgs)


def test_extract_raw_timeline_returns_result_with_per_output_data(
    tmp_path, monkeypatch,
):
    """The RawExtractResult exposes per-output a-v delta etc. for
    the orchestrator's log + ledger."""
    from pipeline_v2.stages import stage_4_raw_extract as rx
    monkeypatch.setattr(subprocess, "run", _stub_subprocess_run())
    monkeypatch.setattr(rx, "_ffprobe_streams", lambda p, b: {
        "video": {"duration": 10.0, "nb_frames": 300},
        "audio": {"duration": 10.001, "nb_frames": 470},   # +1ms drift, OK
    })
    result = rx.extract_raw_timeline(
        "/fake/m.mp4",
        [(0, 10)],
        [(20, 30)],
        str(tmp_path),
        use_nvenc=True,
    )
    assert result.bulletin is not None
    assert result.bulletin.role == "bulletin"
    assert result.bulletin.video_duration_s == 10.0
    assert abs(result.bulletin.av_delta_ms - 1.0) < 0.01
    assert len(result.shorts) == 1
    assert result.shorts[0].index == 1
