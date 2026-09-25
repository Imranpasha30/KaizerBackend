"""UE.1 — frame FX registry: every fragment must run on this host."""
from __future__ import annotations

import subprocess

import pytest

from pipeline_v4 import frame_fx as fx


def test_registry_shape_and_unknown():
    assert len(fx.FX) >= 14
    for key, f in fx.FX.items():
        assert f.key == key and f.label and f.used_for
        frag = f.build(**f.defaults)
        assert frag  # some fragments are bare filter names (e.g. negate)
    assert fx.get_fx_vf("no_such_effect") == ""
    cat = fx.fx_catalog()
    assert len(cat) == len(fx.FX)


def test_param_overrides():
    assert "alls=20" in fx.get_fx_vf("film_grain", strength=20)
    assert "sigma=3.0" in fx.get_fx_vf("gaussian_blur", sigma=3)
    assert "setpts=4.0000*PTS" in fx.get_fx_vf("slow_motion", factor=0.25)


@pytest.mark.parametrize("key", sorted(fx.FX))
def test_every_fx_renders(key, tmp_path):
    """Render a real frame through the exact fragment — validates every
    filter option against the host ffmpeg (same guard as the grades)."""
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("ffmpeg not available")
    frag = fx.get_fx_vf(key)
    out = tmp_path / f"{key}.png"
    graph_mode = ";" in frag                      # e.g. mirror uses a graph
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "lavfi", "-i", "testsrc2=s=640x360:d=0.5:r=25"]
    if graph_mode:
        cmd += ["-filter_complex", frag]
    else:
        cmd += ["-vf", frag]
    cmd += ["-frames:v", "1", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"{key} failed: {proc.stderr[-300:]}"
    assert out.is_file() and out.stat().st_size > 0
