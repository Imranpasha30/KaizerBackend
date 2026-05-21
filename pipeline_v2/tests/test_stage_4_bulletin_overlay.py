"""Item 117 Phase 3 -- bulletin overlay tests.

Covers:
  - cmd shape: single -i bulletin + N extras, -c:a copy enforced
  - default overlay graph builder for typical lower-third + ticker + bug
  - time-conditioned per-story lower-thirds use ``enable=between(...)``
  - audio bit-identity verification raises when violated
  - progress_cb hook invoked
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_HERE = Path(__file__).resolve()
_PIPELINE_V2_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PIPELINE_V2_ROOT))


def _make_stub_run(rc: int = 0, stderr: str = ""):
    def _fake(cmd, *args, **kwargs):
        # Write an output file at the last positional arg so downstream
        # probe + sha extraction can proceed.
        out = cmd[-1]
        if isinstance(out, str) and (out.endswith(".mp4") or out.endswith(".aac")):
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(b"x" * 200_000)
        class _R:
            def __init__(self):
                self.returncode = rc; self.stdout = ""; self.stderr = stderr
        return _R()
    return _fake


# ---------------------------------------------------------------------
# Default overlay graph builder
# ---------------------------------------------------------------------


def test_default_graph_with_no_overlays_passes_through(tmp_path):
    """With no LT / ticker / bug, the default graph is a no-op copy.
    Used when the caller still wants a re-encode pass for some reason."""
    from pipeline_v2.stages.stage_4_bulletin_overlay import (
        build_default_overlay_graph,
    )
    fc, out_label, extras = build_default_overlay_graph([], None, None)
    assert "copy" in fc.lower()
    assert out_label == "out"
    assert extras == []


def test_default_graph_with_two_stories_uses_enable_between(tmp_path):
    """Per-story lower-thirds must be time-conditioned via
    ``enable=between(t, story_start, story_end)``."""
    from pipeline_v2.stages.stage_4_bulletin_overlay import (
        StoryOverlaySpec, build_default_overlay_graph,
    )
    lt0 = tmp_path / "lt_0.png"; lt0.write_bytes(b"x")
    lt1 = tmp_path / "lt_1.png"; lt1.write_bytes(b"x")
    stories = [
        StoryOverlaySpec(start_s=0.0,  end_s=24.867, lt_png_path=str(lt0), lt_native_width=1200),
        StoryOverlaySpec(start_s=24.867, end_s=57.2,  lt_png_path=str(lt1), lt_native_width=2400),
    ]
    fc, out_label, extras = build_default_overlay_graph(stories, None, None)
    # Two LT enable conditions present.
    assert fc.count("enable='between(t") == 2
    # Story 0: between(t, 0.000000, 24.867000) -- escaping uses \,
    assert "between(t\\,0.000000\\,24.867000" in fc
    assert "between(t\\,24.867000\\,57.200000" in fc
    # Story 0 has short LT (1200 < 1920) -> slide-in only.
    # Story 1 has wide LT (2400 > 1920) -> marquee scroll.
    # Sanity: the marquee variant has the W-w/max expression.
    assert "max(W-w" in fc, "long LT must scroll left"
    # Two extras (one PNG per story).
    assert len(extras) == 2
    assert str(lt0) in extras and str(lt1) in extras


def test_default_graph_with_ticker_and_bug(tmp_path):
    """Ticker scrolls left at speed; bug sits at top-right."""
    from pipeline_v2.stages.stage_4_bulletin_overlay import (
        build_default_overlay_graph,
    )
    tk = tmp_path / "ticker.png"; tk.write_bytes(b"x")
    bg = tmp_path / "bug.png"; bg.write_bytes(b"x")
    fc, out_label, extras = build_default_overlay_graph(
        [], str(tk), str(bg),
    )
    # Ticker uses the W-mod(t*speed,w+W) trick.
    assert "W-mod(t*" in fc
    # Bug uses W-w-PAD coords.
    assert "x=W-w-30:y=30" in fc
    assert extras == [str(tk), str(bg)]


# ---------------------------------------------------------------------
# apply_bulletin_overlays cmd shape + invariants
# ---------------------------------------------------------------------


def test_apply_overlays_uses_c_a_copy(tmp_path, monkeypatch):
    """The cmd must contain ``-c:a copy`` -- this is the
    architectural invariant. A future regression that switches to
    ``-c:a aac`` here would silently re-encode audio and re-introduce
    the AAC priming drift bug."""
    from pipeline_v2.stages import stage_4_bulletin_overlay as bo
    captured = {}

    def _fake_run(cmd, *args, **kwargs):
        captured.setdefault("calls", []).append(list(cmd))
        out = cmd[-1]
        if isinstance(out, str):
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(b"x" * 200_000)
        class _R:
            returncode = 0; stdout = ""; stderr = ""
        return _R()
    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(bo, "_ffprobe_streams", lambda p, b: {
        "video": {"duration": 100.0, "nb_frames": 3000},
        "audio": {"duration": 100.0, "nb_frames": 4690},
    })
    # Force audio sha verification to pass: return the same fake sha
    # for both calls (input + output).
    monkeypatch.setattr(bo, "_audio_sha256", lambda p, b: "deadbeef")

    inp = tmp_path / "bulletin_raw.mp4"; inp.write_bytes(b"x" * 200_000)
    out = tmp_path / "bulletin_with_overlays.mp4"
    bo.apply_bulletin_overlays(
        str(inp), str(out),
        video_filter_complex="[0:v]copy[out]",
        video_out_label="out",
        use_nvenc=False,
    )
    # Find the ffmpeg cmd that wrote out.
    overlay_cmd = next(
        c for c in captured["calls"]
        if c[-1] == str(out)
    )
    assert "-c:a" in overlay_cmd
    assert overlay_cmd[overlay_cmd.index("-c:a") + 1] == "copy"
    # And -c:a aac MUST NOT be present (the regression we guard against).
    cmd_str = " ".join(overlay_cmd)
    assert "-c:a aac" not in cmd_str


def test_apply_overlays_raises_on_audio_sha_mismatch(tmp_path, monkeypatch):
    """If the output audio sha != input, raise (this would mean -c:a
    copy was dropped or the muxer remuxed audio packets)."""
    from pipeline_v2.stages import stage_4_bulletin_overlay as bo
    monkeypatch.setattr(subprocess, "run", _make_stub_run())
    monkeypatch.setattr(bo, "_ffprobe_streams", lambda p, b: {
        "video": {"duration": 100.0, "nb_frames": 3000},
        "audio": {"duration": 100.0, "nb_frames": 4690},
    })
    # Different sha for input vs output -> regression.
    calls = {"n": 0}
    def _fake_sha(path, b):
        calls["n"] += 1
        return "input_sha" if calls["n"] == 1 else "output_sha_different"
    monkeypatch.setattr(bo, "_audio_sha256", _fake_sha)

    inp = tmp_path / "bulletin_raw.mp4"; inp.write_bytes(b"x" * 200_000)
    out = tmp_path / "bulletin_with_overlays.mp4"
    with pytest.raises(bo.BulletinOverlayError, match="bit-identity"):
        bo.apply_bulletin_overlays(
            str(inp), str(out),
            video_filter_complex="[0:v]copy[out]",
            video_out_label="out",
            use_nvenc=False,
        )


def test_apply_overlays_extras_become_indexed_inputs(tmp_path, monkeypatch):
    """extra_input_paths should be added to the cmd as ``-loop 1 -i``
    in order, so the filter graph can reference them as [1:v], [2:v]
    etc."""
    from pipeline_v2.stages import stage_4_bulletin_overlay as bo
    captured = {}
    def _fake_run(cmd, *args, **kwargs):
        captured.setdefault("calls", []).append(list(cmd))
        out = cmd[-1]
        if isinstance(out, str):
            Path(out).write_bytes(b"x" * 200_000)
        class _R: returncode = 0; stdout = ""; stderr = ""
        return _R()
    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(bo, "_ffprobe_streams", lambda p, b: {
        "video": {"duration": 10.0, "nb_frames": 300},
        "audio": {"duration": 10.0, "nb_frames": 470},
    })

    inp = tmp_path / "bulletin_raw.mp4"; inp.write_bytes(b"x" * 200_000)
    lt0 = tmp_path / "lt_0.png"; lt0.write_bytes(b"x")
    tk  = tmp_path / "tk.png";   tk.write_bytes(b"x")
    out = tmp_path / "out.mp4"
    bo.apply_bulletin_overlays(
        str(inp), str(out),
        video_filter_complex="[0:v][1:v][2:v]overlay[out]",
        video_out_label="out",
        extra_input_paths=[str(lt0), str(tk)],
        verify_audio_bit_identity=False,   # skip sha noise in this test
        use_nvenc=False,
    )
    overlay_cmd = next(c for c in captured["calls"] if c[-1] == str(out))
    # Three -i: bulletin + 2 extras.
    assert overlay_cmd.count("-i") == 3
    # Each extra has -loop 1 before its -i.
    for extra in (str(lt0), str(tk)):
        idx = overlay_cmd.index(extra)
        assert overlay_cmd[idx - 1] == "-i"
        assert overlay_cmd[idx - 2] == "1"
        assert overlay_cmd[idx - 3] == "-loop"


def test_apply_overlays_propagates_ffmpeg_failure(tmp_path, monkeypatch):
    """ffmpeg rc != 0 -> BulletinOverlayError."""
    from pipeline_v2.stages import stage_4_bulletin_overlay as bo
    monkeypatch.setattr(
        subprocess, "run",
        _make_stub_run(rc=1, stderr="some ffmpeg error"),
    )
    monkeypatch.setattr(bo, "_ffprobe_streams", lambda p, b: {})
    monkeypatch.setattr(bo, "_audio_sha256", lambda p, b: "sha")

    inp = tmp_path / "in.mp4"; inp.write_bytes(b"x" * 200_000)
    out = tmp_path / "out.mp4"
    with pytest.raises(bo.BulletinOverlayError, match="rc=1"):
        bo.apply_bulletin_overlays(
            str(inp), str(out),
            video_filter_complex="[0:v]copy[out]",
            video_out_label="out",
            use_nvenc=False,
        )


def test_apply_overlays_progress_cb_invoked(tmp_path, monkeypatch):
    from pipeline_v2.stages import stage_4_bulletin_overlay as bo
    monkeypatch.setattr(subprocess, "run", _make_stub_run())
    monkeypatch.setattr(bo, "_ffprobe_streams", lambda p, b: {
        "video": {"duration": 10.0, "nb_frames": 300},
        "audio": {"duration": 10.0, "nb_frames": 470},
    })
    monkeypatch.setattr(bo, "_audio_sha256", lambda p, b: "same")

    msgs: list[str] = []
    inp = tmp_path / "in.mp4"; inp.write_bytes(b"x" * 200_000)
    bo.apply_bulletin_overlays(
        str(inp), str(tmp_path / "out.mp4"),
        video_filter_complex="[0:v]copy[out]",
        video_out_label="out",
        progress_cb=msgs.append,
        use_nvenc=False,
    )
    assert any("overlay" in m and "applying" in m for m in msgs)
    assert any("done" in m and "audio_bit_identical" in m for m in msgs)
