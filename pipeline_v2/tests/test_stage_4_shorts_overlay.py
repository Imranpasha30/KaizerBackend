"""Item 117 Phase 4 -- per-short overlay tests.

Mirrors test_stage_4_bulletin_overlay.py but for shorts (vertical
1080x1920 canvas, optional hook + LT).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_PIPELINE_V2_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PIPELINE_V2_ROOT))


def _make_stub_run(rc: int = 0, stderr: str = ""):
    _rc = rc
    _stderr = stderr
    def _fake(cmd, *args, **kwargs):
        out = cmd[-1]
        if isinstance(out, str) and (out.endswith(".mp4") or out.endswith(".aac")):
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(b"x" * 200_000)
        class _R:
            returncode = _rc
            stdout = ""
            stderr = _stderr
        return _R()
    return _fake


# ---------------------------------------------------------------------
# Default short graph builder
# ---------------------------------------------------------------------


def test_default_short_graph_crops_to_9_16_and_scales():
    """Base operation: crop to 9:16 centred + scale to 1080x1920."""
    from pipeline_v2.stages.stage_4_shorts_overlay import (
        build_default_short_graph,
    )
    fc, out_label, extras = build_default_short_graph()
    # Crop expression must reference iw/ih and 9/16 (or 16/9).
    assert "crop=" in fc
    assert "9/16" in fc or "16/9" in fc
    # Scale must hit the standard short canvas.
    assert "scale=1080:1920" in fc
    assert out_label != ""
    # No extras when neither hook nor LT supplied.
    assert extras == []


def test_default_short_graph_with_hook_text():
    """Hook text is drawn near the top of the canvas with bold style."""
    from pipeline_v2.stages.stage_4_shorts_overlay import (
        build_default_short_graph,
    )
    fc, out_label, extras = build_default_short_graph(
        hook_text="Hook line for short",
    )
    assert "drawtext=" in fc
    assert "Hook line for short" in fc
    # Hook sits near the top.
    assert "y=80" in fc


def test_default_short_graph_with_lt_png(tmp_path):
    """LT PNG becomes an extra input and is overlaid bottom-centred."""
    from pipeline_v2.stages.stage_4_shorts_overlay import (
        build_default_short_graph,
    )
    lt = tmp_path / "lt.png"; lt.write_bytes(b"x")
    fc, out_label, extras = build_default_short_graph(
        lt_png_path=str(lt),
    )
    assert "[1:v]overlay=" in fc   # first extra after [0:v]
    assert "H-h-120" in fc          # bottom-anchored
    assert extras == [str(lt)]


# ---------------------------------------------------------------------
# apply_short_overlays invariants
# ---------------------------------------------------------------------


def test_apply_short_overlays_uses_c_a_copy(tmp_path, monkeypatch):
    """Same architectural invariant as bulletin overlay: -c:a copy."""
    from pipeline_v2.stages import stage_4_shorts_overlay as so
    captured = {}

    def _fake_run(cmd, *args, **kwargs):
        captured.setdefault("calls", []).append(list(cmd))
        out = cmd[-1]
        if isinstance(out, str):
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(b"x" * 200_000)
        class _R: returncode = 0; stdout = ""; stderr = ""
        return _R()
    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(so, "_ffprobe_streams", lambda p, b: {
        "video": {"duration": 18.0, "nb_frames": 540},
        "audio": {"duration": 18.0, "nb_frames": 845},
    })
    monkeypatch.setattr(so, "_audio_sha256", lambda p, b: "same_sha")

    inp = tmp_path / "short_01_raw.mp4"; inp.write_bytes(b"x" * 200_000)
    out = tmp_path / "short_01_final.mp4"
    so.apply_short_overlays(
        str(inp), str(out),
        video_filter_complex="[0:v]crop=iw:ih,setsar=1[out]",
        video_out_label="out",
        use_nvenc=False,
    )
    overlay_cmd = next(c for c in captured["calls"] if c[-1] == str(out))
    assert "-c:a" in overlay_cmd
    assert overlay_cmd[overlay_cmd.index("-c:a") + 1] == "copy"
    cmd_str = " ".join(overlay_cmd)
    assert "-c:a aac" not in cmd_str


def test_apply_short_overlays_raises_on_sha_mismatch(tmp_path, monkeypatch):
    from pipeline_v2.stages import stage_4_shorts_overlay as so
    monkeypatch.setattr(subprocess, "run", _make_stub_run())
    monkeypatch.setattr(so, "_ffprobe_streams", lambda p, b: {
        "video": {"duration": 18.0, "nb_frames": 540},
        "audio": {"duration": 18.0, "nb_frames": 845},
    })
    calls = {"n": 0}
    def _fake_sha(p, b):
        calls["n"] += 1
        return "input_sha" if calls["n"] == 1 else "output_different"
    monkeypatch.setattr(so, "_audio_sha256", _fake_sha)

    inp = tmp_path / "in.mp4"; inp.write_bytes(b"x" * 200_000)
    out = tmp_path / "out.mp4"
    with pytest.raises(so.ShortsOverlayError, match="bit-identity"):
        so.apply_short_overlays(
            str(inp), str(out),
            video_filter_complex="[0:v]copy[out]",
            video_out_label="out",
            use_nvenc=False,
        )


def test_apply_short_overlays_propagates_ffmpeg_failure(tmp_path, monkeypatch):
    from pipeline_v2.stages import stage_4_shorts_overlay as so
    monkeypatch.setattr(
        subprocess, "run",
        _make_stub_run(rc=1, stderr="ffmpeg short error"),
    )
    monkeypatch.setattr(so, "_ffprobe_streams", lambda p, b: {})
    monkeypatch.setattr(so, "_audio_sha256", lambda p, b: "sha")

    inp = tmp_path / "in.mp4"; inp.write_bytes(b"x" * 200_000)
    with pytest.raises(so.ShortsOverlayError, match="rc=1"):
        so.apply_short_overlays(
            str(inp), str(tmp_path / "out.mp4"),
            video_filter_complex="[0:v]copy[out]",
            video_out_label="out",
            use_nvenc=False,
        )


def test_apply_short_overlays_returns_result_with_av_delta(
    tmp_path, monkeypatch,
):
    from pipeline_v2.stages import stage_4_shorts_overlay as so
    monkeypatch.setattr(subprocess, "run", _make_stub_run())
    monkeypatch.setattr(so, "_ffprobe_streams", lambda p, b: {
        "video": {"duration": 18.0, "nb_frames": 540},
        "audio": {"duration": 18.0005, "nb_frames": 845},
    })
    monkeypatch.setattr(so, "_audio_sha256", lambda p, b: "same")

    inp = tmp_path / "in.mp4"; inp.write_bytes(b"x" * 200_000)
    res = so.apply_short_overlays(
        str(inp), str(tmp_path / "out.mp4"),
        video_filter_complex="[0:v]copy[out]",
        video_out_label="out",
        use_nvenc=False,
    )
    assert res.audio_bit_identical is True
    assert abs(res.av_delta_ms - 0.5) < 0.01
    assert res.video_duration_s == 18.0


def test_apply_short_overlays_progress_cb(tmp_path, monkeypatch):
    from pipeline_v2.stages import stage_4_shorts_overlay as so
    monkeypatch.setattr(subprocess, "run", _make_stub_run())
    monkeypatch.setattr(so, "_ffprobe_streams", lambda p, b: {
        "video": {"duration": 18.0, "nb_frames": 540},
        "audio": {"duration": 18.0, "nb_frames": 845},
    })
    monkeypatch.setattr(so, "_audio_sha256", lambda p, b: "same")

    msgs: list[str] = []
    inp = tmp_path / "in.mp4"; inp.write_bytes(b"x" * 200_000)
    so.apply_short_overlays(
        str(inp), str(tmp_path / "out.mp4"),
        video_filter_complex="[0:v]copy[out]",
        video_out_label="out",
        progress_cb=msgs.append,
        use_nvenc=False,
    )
    assert any("short overlay" in m and "applying" in m for m in msgs)
    assert any("done" in m for m in msgs)
