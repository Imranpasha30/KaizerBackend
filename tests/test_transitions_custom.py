"""UG.2 — advanced custom transitions: every expression renders on host."""
from __future__ import annotations

import subprocess

import pytest

from pipeline_v4.trailer_styles import (EXTRA_TRANSITIONS,
                                        TRANSITION_CATALOG, xfade_arg)


def _has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def test_catalog_totals_and_arg_builder():
    assert len(EXTRA_TRANSITIONS) == 20
    assert len(TRANSITION_CATALOG) == 76
    for n in EXTRA_TRANSITIONS:
        assert n in TRANSITION_CATALOG
        assert xfade_arg(n).startswith("transition=custom:expr='")
    assert xfade_arg("circleopen") == "transition=circleopen"
    assert xfade_arg("no_such_wipe") == "transition=fade"  # fail-soft


def test_packs_actually_use_the_new_transitions():
    """Wired, not shelf-ware: pack transition sets reference the customs
    and every referenced name exists in the catalog."""
    from pipeline_v4.trailer_styles import STYLES
    used = set()
    for p in STYLES.values():
        for t in p.transitions:
            assert t in TRANSITION_CATALOG, f"{p.key} references unknown {t!r}"
            used.add(t)
    assert len(used & set(EXTRA_TRANSITIONS)) >= 18


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")
@pytest.mark.parametrize("name", sorted(EXTRA_TRANSITIONS))
def test_every_custom_transition_renders(name, tmp_path):
    """Render a real xfade through the custom expression — validates the
    expr against the host ffmpeg (same guard as grades/frame FX)."""
    out = tmp_path / f"{name}.mp4"
    fc = f"[0:v][1:v]xfade={xfade_arg(name)}:duration=0.6:offset=0.3[v]"
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "lavfi", "-i", "smptebars=s=320x180:d=1.0:r=25",
           "-f", "lavfi", "-i", "testsrc2=s=320x180:d=1.0:r=25",
           "-filter_complex", fc, "-map", "[v]",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
           str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"{name} failed: {proc.stderr[-400:]}"
    assert out.is_file() and out.stat().st_size > 0
