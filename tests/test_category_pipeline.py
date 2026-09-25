"""Category edit pipeline: Director graphics + captions IN the compose."""
from __future__ import annotations

import json

from PIL import Image

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


def test_dwin_graph_step_shape_and_order():
    """Director windows overlay with alpha fade + enable, BEFORE the LT
    (broadcast furniture stays on top); empty dwin = identical graph."""
    plain, _, _ = vb._build_story_filter_graph(**_MINIMAL)
    empty, _, _ = vb._build_story_filter_graph(**_MINIMAL, dwin=())
    assert plain == empty
    fc, out, _ = vb._build_story_filter_graph(
        **_MINIMAL,
        dwin=((3, 1.0, 5.0, "0", "0"),
              (4, 2.0, 2.6, "'(W-w)/2'", "890-h-14")))
    assert "fade=t=in:st=1.000:d=0.250:alpha=1[dw0]" in fc
    assert "overlay=x=0:y=0:enable='between(t,1.000,5.000)'[stage_dw0]" in fc
    assert "overlay=x='(W-w)/2':y=890-h-14:" in fc
    assert fc.index("[stage_dw1]") < fc.index("[stage_lt]")
    assert out == "stage_lt"


def test_positioned_overlay_full_canvas(tmp_path):
    from pipeline_v4.overlays import render_overlay_positioned
    p = render_overlay_positioned("breaking_news_banner",
                                  str(tmp_path / "b.png"),
                                  canvas_w=1280, canvas_h=720)
    im = Image.open(p)
    assert im.size == (1280, 720)
    # banner sits at its broadcast position (70% down), transparent above
    assert im.getpixel((640, 60))[3] == 0
    assert im.getpixel((640, int(720 * 0.72)))[3] > 150
    assert render_overlay_positioned("nope", str(tmp_path / "x.png")) is None


def test_story_words_sidecar_and_karaoke_params(tmp_path):
    (tmp_path / "story_words.json").write_text(json.dumps(
        {"0": [{"w": "మోదీ", "s": 0.1, "e": 0.5}], "1": []}),
        encoding="utf-8")
    assert vb._story_words_sidecar(tmp_path, 0)[0]["w"] == "మోదీ"
    assert vb._story_words_sidecar(tmp_path, 1) == []
    assert vb._story_words_sidecar(tmp_path, 7) == []
    assert vb._story_words_sidecar(tmp_path / "nope", 0) == []
    st, hi = vb._karaoke_params("karaoke_box_red")
    assert st == "box" and hi[0] > 150
    assert vb._karaoke_params("unknown") == ("hilite", (255, 214, 0, 255))


def test_directive_fingerprint_forks_hash():
    """A changed Director overlay/caption decision re-renders the story;
    no directive keeps the exact pre-Director hash."""
    from types import SimpleNamespace
    story = SimpleNamespace(title_native="t", title_english="t", summary="s",
                            video_t_start=0.0, video_t_end=10.0,
                            story_index=0, total_stories=1)

    def _h(fx):
        return vb._per_story_cache_hash(
            story=story, ticker_path="", sidebar_path="", layout=None,
            channel_bug_path="", watermark_path="", watermark_position="",
            font_path="", bg_video_abs=None, bg_video_volume=0.0,
            language_code="te", images=None, pool_dir=None, effects_vf=fx)

    base = _h("")
    with_ov = _h('|dov:[{"id": "tag_bigstory", "t": 1.0}]|cap:none')
    with_cap = _h('|dov:[]|cap:karaoke_box_yellow')
    assert len({base, with_ov, with_cap}) == 3


def test_sound_design_pass(tmp_path):
    """Stings + ticks + ducked bed mixed onto a finished bulletin —
    duration unchanged, audio present, original kept on failure."""
    import subprocess
    from types import SimpleNamespace
    from pipeline_v4.director import StoryDirective
    final = tmp_path / "bulletin.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=25:d=12",
         "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000",
         "-t", "12", "-c:v", "libx264", "-preset", "ultrafast",
         "-c:a", "aac", "-shortest", str(final)],
        check=True, capture_output=True, timeout=120)
    stories = [SimpleNamespace(story_index=0), SimpleNamespace(story_index=1)]
    dirs = {0: StoryDirective(mood="crime",
                              overlays=[{"id": "tag_investigation", "t": 1.0}]),
            1: StoryDirective(mood="sports")}
    vb._apply_sound_design(str(final), directives=dirs, stories=stories,
                           story_starts=[0.0, 6.0], stitched_dur=12.0,
                           work_dir=str(tmp_path))
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type:format=duration", "-of", "json", str(final)],
        capture_output=True, text=True, timeout=30)
    j = json.loads(probe.stdout)
    kinds = sorted(s["codec_type"] for s in j["streams"])
    assert kinds == ["audio", "video"]
    assert 11.0 <= float(j["format"]["duration"]) <= 13.0
    # no directives → untouched (no exception, no temp litter)
    vb._apply_sound_design(str(final), directives={}, stories=stories,
                           story_starts=[0.0], stitched_dur=12.0,
                           work_dir=str(tmp_path))


def test_edge_clamp_kills_layout_blink():
    """Transition boundary guard: fullscreen windows never touch story
    edges (the crossfade would ghost the framed layout in/out) and
    sub-second windows (perceived blinking) are dropped."""
    # gap touching BOTH edges of a 20s story → pulled inside
    assert vb._edge_clamp_windows([(0.0, 20.0)], 20.0) == [(0.75, 19.25)]
    # window at the very end → clamped away from the joint
    assert vb._edge_clamp_windows([(17.9, 20.0)], 20.0) == [(17.9, 19.25)]
    # sliver windows → dropped entirely (no blink)
    assert vb._edge_clamp_windows([(5.0, 5.8)], 20.0) == []
    assert vb._edge_clamp_windows([(19.4, 20.0)], 20.0) == []
    # mid-story windows unchanged (rounding aside)
    assert vb._edge_clamp_windows([(4.0, 9.0)], 20.0) == [(4.0, 9.0)]
    # path-carrying windows (spotlight/pip) keep their path
    assert vb._edge_clamp_windows([("x.png", 0.0, 3.0)], 20.0,
                                  with_path=True) == [("x.png", 0.75, 3.0)]
    assert vb._edge_clamp_windows([], 20.0) == []
