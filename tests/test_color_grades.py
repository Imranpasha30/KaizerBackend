"""UA.1 — color-grade registry: every chain must actually run on this host.

This is the test class that catches "option does not exist" ffmpeg bugs
(the colorbalance ms=/hs= incident) before a paid render does: each of
the 23 grades renders a real frame through its exact chain.
"""
from __future__ import annotations

import subprocess

import pytest

from pipeline_v4 import color_grades as cg


def test_registry_shape():
    assert len(cg.GRADES) == 23
    for key, g in cg.GRADES.items():
        assert g.key == key and g.label and g.used_for
        assert g.chain and " " not in g.chain.split(",")[0].split("=")[0]
    cat = cg.grade_catalog()
    assert len(cat) == 23
    assert all(c["source"] in ("procedural", "lut") for c in cat)


def test_unknown_key_falls_back_to_neutral():
    assert cg.get_grade_vf("no_such_look") == cg.GRADES["newsroom_neutral"].chain
    assert cg.get_grade_vf("") == cg.GRADES["newsroom_neutral"].chain


def test_lut_dropin_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(cg, "_luts_dir", lambda: tmp_path)
    (tmp_path / "monochrome.cube").write_text("LUT_3D_SIZE 2\n", encoding="utf-8")
    vf = cg.get_grade_vf("monochrome")
    assert vf.startswith("lut3d=file=")
    # others untouched
    assert cg.get_grade_vf("newsroom_neutral") == cg.GRADES["newsroom_neutral"].chain


@pytest.mark.parametrize("key", sorted(cg.GRADES))
def test_every_grade_chain_renders(key, tmp_path):
    """Render one real frame through the exact chain — validates every
    filter option against the host ffmpeg."""
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("ffmpeg not available")
    out = tmp_path / f"{key}.png"
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc2=s=320x180:d=0.1:r=25",
         "-vf", cg.GRADES[key].chain,
         "-frames:v", "1", str(out)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"{key} chain failed: {proc.stderr[-300:]}"
    assert out.is_file() and out.stat().st_size > 0
