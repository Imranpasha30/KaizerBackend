"""Full-form effects mode (operator: 'effects not only for the trailer').

Proves the three safety properties:
  1. off/legacy → resolved chain '' → graph CHARACTER-IDENTICAL + hash
     unchanged (old jobs keep 100% cache hits, byte-identical renders).
  2. auto/rich resolve to real vetted chains that render on this host.
  3. the resolved chain is a conditional hash ingredient.
"""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from pipeline_v4 import v1_bridge as vb

_MINIMAL = dict(
    canvas_w=1920, canvas_h=1080,
    main_inner_w=1254, side_inner_w=564, tile_inner_h=794,
    main_outer_w=1260, side_outer_w=570, tile_outer_h=800,
    main_x=30, side_x=1320, tile_y=50, lt_y=890, ticker_y=1030,
    border_colour="white", bg_color="black",
    lt_x_expr="if(lt(t\\,0.4)\\,-w+w*t/0.4\\,0)", ticker_speed_px_s=200.0,
    has_bg=False, bg_input_idx=None, apply_ticker=False, ticker_in_idx=None,
    has_bug=False, bug_in_idx=None, has_wm=False, wm_in_idx=None,
    watermark_xy=("", ""), bg_video_volume=0.0,
)


def test_mode_resolution(monkeypatch):
    monkeypatch.delenv("KAIZER_V4_EFFECTS_MODE", raising=False)
    assert vb._effects_vf_from_env() == ""                    # legacy
    for off in ("off", "none", "legacy", ""):
        monkeypatch.setenv("KAIZER_V4_EFFECTS_MODE", off)
        assert vb._effects_vf_from_env() == ""
    monkeypatch.setenv("KAIZER_V4_EFFECTS_MODE", "auto")
    assert vb._effects_vf_from_env() == vb._NEWS_POLISH_VF
    monkeypatch.setenv("KAIZER_V4_EFFECTS_MODE", "rich")
    monkeypatch.setenv("KAIZER_V4_CONTENT_TYPE", "crime")
    rich = vb._effects_vf_from_env()
    assert "vignette" in rich and rich != vb._NEWS_POLISH_VF
    assert ";" not in rich                                    # linear chain
    # canvas override beats env (editor re-render path)
    assert vb._effects_vf_from_env("off") == ""
    monkeypatch.setenv("KAIZER_V4_EFFECTS_MODE", "off")
    assert vb._effects_vf_from_env("auto") == vb._NEWS_POLISH_VF


def test_graph_identity_when_off_and_prefix_when_on():
    plain = vb._build_story_filter_graph(**_MINIMAL, gap_windows=())
    off = vb._build_story_filter_graph(**_MINIMAL, gap_windows=(),
                                       story_fx_vf="")
    assert off == plain                                       # byte-compat
    on, _, _ = vb._build_story_filter_graph(
        **_MINIMAL, gap_windows=(), story_fx_vf=vb._NEWS_POLISH_VF)
    assert on.startswith(f"[0:v]{vb._NEWS_POLISH_VF},scale=")
    # with gaps the fx applies ONCE, before the split (tile + fullscreen match)
    gap, _, _ = vb._build_story_filter_graph(
        **_MINIMAL, gap_windows=((0.0, 2.0),),
        story_fx_vf=vb._NEWS_POLISH_VF)
    assert gap.startswith(f"[0:v]{vb._NEWS_POLISH_VF},split=2")
    assert gap.count(vb._NEWS_POLISH_VF) == 1


def test_hash_ingredient_conditional():
    story = SimpleNamespace(title_native="t", title_english="t", summary="s",
                            video_t_start=0.0, video_t_end=10.0,
                            story_index=0, total_stories=1)

    def _h(fx):
        return vb._per_story_cache_hash(
            story=story, ticker_path="", sidebar_path="", layout=None,
            channel_bug_path="", watermark_path="", watermark_position="",
            font_path="", bg_video_abs=None, bg_video_volume=0.0,
            language_code="te", images=None, pool_dir=None,
            effects_vf=fx)

    assert _h("") == _h("")                       # deterministic
    legacy_default = vb._per_story_cache_hash(
        story=story, ticker_path="", sidebar_path="", layout=None,
        channel_bug_path="", watermark_path="", watermark_position="",
        font_path="", bg_video_abs=None, bg_video_volume=0.0,
        language_code="te", images=None, pool_dir=None)
    assert _h("") == legacy_default               # off == pre-effects hash
    assert _h(vb._NEWS_POLISH_VF) != _h("")       # on forks the cache


@pytest.mark.parametrize("mode,ct", [("auto", ""), ("rich", "crime"),
                                     ("rich", "comedy"), ("rich", "horror")])
def test_resolved_chains_render_on_host(mode, ct, tmp_path, monkeypatch):
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("ffmpeg not available")
    monkeypatch.setenv("KAIZER_V4_EFFECTS_MODE", mode)
    if ct:
        monkeypatch.setenv("KAIZER_V4_CONTENT_TYPE", ct)
    chain = vb._effects_vf_from_env()
    assert chain
    out = tmp_path / f"{mode}_{ct or 'x'}.png"
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc2=s=640x360:d=0.5:r=25",
         "-vf", chain, "-frames:v", "1", str(out)],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"{mode}/{ct}: {proc.stderr[-300:]}"
    assert out.stat().st_size > 0
