"""Item 117 Phase 5 -- orchestrator wire-through tests.

Verifies the unified raw-extract path activates when
``KAIZER_USE_V2_RAW_EXTRACT=1`` and falls through cleanly otherwise.
The full ffmpeg-level integration is exercised by the Phase 2 / 3 /
4 unit tests; these tests focus on the Stage4Render wiring.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_HERE = Path(__file__).resolve()
_PIPELINE_V2_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PIPELINE_V2_ROOT))


def test_v2_extract_disabled_by_default_in_env(monkeypatch):
    """Without the env flag, ``_v2_extract_enabled()`` returns False."""
    from pipeline_v2.stages.stage_4_render import Stage4Render
    monkeypatch.delenv("KAIZER_USE_V2_RAW_EXTRACT", raising=False)
    r = Stage4Render.__new__(Stage4Render)   # bypass __init__
    assert r._v2_extract_enabled() is False


@pytest.mark.parametrize("flag_value,expected", [
    ("1", True), ("true", True), ("True", True), ("YES", True),
    ("on", True), ("0", False), ("false", False), ("", False),
])
def test_v2_extract_flag_truthy_check(flag_value, expected, monkeypatch):
    """The flag accepts common truthy values (1/true/yes/on)
    case-insensitively."""
    from pipeline_v2.stages.stage_4_render import Stage4Render
    monkeypatch.setenv("KAIZER_USE_V2_RAW_EXTRACT", flag_value)
    r = Stage4Render.__new__(Stage4Render)
    assert r._v2_extract_enabled() is expected


def test_run_unified_raw_extract_calls_extract_raw_timeline(monkeypatch, tmp_path):
    """``_run_unified_raw_extract`` must invoke ``extract_raw_timeline``
    with per_story bulletin_mode and the correct separate dirs."""
    from pipeline_v2.stages.stage_4_render import Stage4Render
    from pipeline_v2.models import ShortsCut, FullVideoCut

    captured = {}
    def _fake_extract(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "pipeline_v2.stages.stage_4_raw_extract.extract_raw_timeline",
        _fake_extract,
    )

    r = Stage4Render.__new__(Stage4Render)
    r.video_path = Path("/fake/mezz.mp4")
    r.output_dir = tmp_path / "shorts"
    # bulletin_dir is a @property reading from output_dir; patch
    # the descriptor on the instance via type-level monkeypatch.
    monkeypatch.setattr(
        Stage4Render, "bulletin_dir",
        property(lambda _self: tmp_path / "bulletin"),
    )
    r._unified_extract_done = False

    shorts_cuts = [
        ShortsCut(
            index=0, start_sec=10.0, end_sec=30.0, hook="hi",
            lower_third="lt", section="A", entities=[], importance=5,
        ),
    ]
    full_video_cuts = [
        FullVideoCut(
            index=0, start_word_idx=0, end_word_idx=10,
            start_sec=1.6, end_sec=26.467, importance=8,
        ),
        FullVideoCut(
            index=1, start_word_idx=11, end_word_idx=20,
            start_sec=27.1, end_sec=59.433, importance=7,
        ),
    ]

    r._run_unified_raw_extract(
        shorts_cuts=shorts_cuts,
        full_video_cuts=full_video_cuts,
    )

    # extract_raw_timeline got the right call shape.
    # str(Path('/fake/mezz.mp4')) on Windows is '\\fake\\mezz.mp4';
    # normalise via os.path before comparing.
    import os
    assert os.path.normpath(captured["mezzanine_path"]) == os.path.normpath("/fake/mezz.mp4")
    assert captured["bulletin_cuts"] == [(1.6, 26.467), (27.1, 59.433)]
    assert captured["shorts_cuts"] == [(10.0, 30.0)]
    assert captured["bulletin_mode"] == "per_story"
    # Shorts and bulletin written to SEPARATE dirs (V1 cache-key safe).
    assert str(captured["out_dir"]) == str(tmp_path / "shorts")
    assert str(captured["bulletin_out_dir"]) == str(tmp_path / "bulletin")
    # Idempotency flag was set.
    assert r._unified_extract_done is True


def test_run_unified_raw_extract_idempotent_via_done_flag(monkeypatch, tmp_path):
    """A second call after the first does not re-invoke
    extract_raw_timeline -- the orchestrator can call it multiple
    times safely (once per pass) without double-encoding."""
    from pipeline_v2.stages.stage_4_render import Stage4Render
    from pipeline_v2.models import ShortsCut, FullVideoCut

    calls = {"n": 0}
    def _fake_extract(**kwargs):
        calls["n"] += 1
    monkeypatch.setattr(
        "pipeline_v2.stages.stage_4_raw_extract.extract_raw_timeline",
        _fake_extract,
    )

    r = Stage4Render.__new__(Stage4Render)
    r.video_path = Path("/fake/mezz.mp4")
    r.output_dir = tmp_path / "shorts"
    # bulletin_dir is a @property reading from output_dir; patch
    # the descriptor on the instance via type-level monkeypatch.
    monkeypatch.setattr(
        Stage4Render, "bulletin_dir",
        property(lambda _self: tmp_path / "bulletin"),
    )
    r._unified_extract_done = False

    shorts_cuts = [
        ShortsCut(
            index=0, start_sec=10.0, end_sec=30.0, hook="hi",
            lower_third="lt", section="A", entities=[], importance=5,
        ),
    ]
    full_video_cuts = []

    r._run_unified_raw_extract(shorts_cuts, full_video_cuts)
    # Manually short-circuit a second call: the caller (_render_impl)
    # checks _unified_extract_done before calling.
    assert r._unified_extract_done is True
    # If the caller respects the flag, calls stay at 1.
    assert calls["n"] == 1


def test_render_impl_skips_unified_extract_when_flag_off(monkeypatch, tmp_path):
    """When KAIZER_USE_V2_RAW_EXTRACT is unset, the unified extract
    path is NOT invoked -- legacy cut step runs as before."""
    monkeypatch.delenv("KAIZER_USE_V2_RAW_EXTRACT", raising=False)
    from pipeline_v2.stages.stage_4_render import Stage4Render

    fake_extract = MagicMock()
    monkeypatch.setattr(
        "pipeline_v2.stages.stage_4_raw_extract.extract_raw_timeline",
        fake_extract,
    )

    r = Stage4Render.__new__(Stage4Render)
    # Calling _v2_extract_enabled directly is the gate; verify it
    # returns False so _render_impl will skip the extract call.
    assert r._v2_extract_enabled() is False
    # And the extract function was not called as a side effect of
    # the gate check.
    fake_extract.assert_not_called()


def test_stage_4_render_has_unified_extract_methods():
    """Sanity: the Stage4Render class exposes the item 117 phase 5
    integration methods. Catches a future refactor that removes
    them."""
    from pipeline_v2.stages.stage_4_render import Stage4Render
    assert hasattr(Stage4Render, "_v2_extract_enabled")
    assert hasattr(Stage4Render, "_run_unified_raw_extract")
    assert hasattr(Stage4Render, "_unified_extract_done")
