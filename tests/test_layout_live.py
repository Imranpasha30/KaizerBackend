"""Live-layout engine tests.

Unit 1 — resolution layer (story_geometry / RENDERABLE_V2 / allowed_for).
Unit 2 — the composer's per-story layout graph: the legacy graph must stay
CHARACTER-IDENTICAL (byte-compat: every pre-layout canvas re-renders from
cache), and the new layout branches (per-tile rects, borderless, no-side,
blurred-echo / image backdrops) must take the documented shapes."""
from pipeline_v4 import layout_library as ll
from pipeline_v4.v1_bridge import _build_story_filter_graph

_COMMON = dict(canvas_w=1920, canvas_h=1080, main_inner_w=1254,
               side_inner_w=564, tile_inner_h=794, main_outer_w=1260,
               side_outer_w=570, tile_outer_h=800, main_x=30, side_x=1320,
               tile_y=50, lt_y=890, ticker_y=1030, border_colour="white",
               bg_color="black", lt_x_expr="0", ticker_speed_px_s=200.0,
               has_bg=False, bg_input_idx=None, apply_ticker=False,
               ticker_in_idx=None, has_bug=False, bug_in_idx=None,
               has_wm=False, wm_in_idx=None)

# Captured from the pre-layout implementation (2026-07-13) — the legacy
# contract. If these ever need to change, that is a renderer-version
# event, not a test update.
_GOLDEN_PLAIN = (
    "[0:v]scale=1254:794:force_original_aspect_ratio=increase,"
    "crop=1254:794,setsar=1,pad=1260:800:3:3:color=white[main_v];"
    "[1:v]scale=564:794:force_original_aspect_ratio=increase,"
    "crop=564:794,setsar=1,pad=570:800:3:3:color=white[side_v];"
    "color=c=black:s=1920x1080:r=30[bg];"
    "[bg][main_v]overlay=x=30:y=50:shortest=1[stage_m];"
    "[stage_m][side_v]overlay=x=1320:y=50[stage_top];"
    "[stage_top][2:v]overlay=x='0':y=890:format=auto[stage_lt]",
    "stage_lt", "0:a?")
_GOLDEN_GAPS = (
    "[0:v]eq=contrast=1.06,split=2[v_tile][v_full];"
    "[v_tile]scale=1254:794:force_original_aspect_ratio=increase,"
    "crop=1254:794,setsar=1,pad=1260:800:3:3:color=white[main_v];"
    "[1:v]scale=564:794:force_original_aspect_ratio=increase,"
    "crop=564:794,setsar=1,pad=570:800:3:3:color=white[side_v];"
    "color=c=black:s=1920x1080:r=30[bg];"
    "[bg][main_v]overlay=x=30:y=50:shortest=1[stage_m];"
    "[stage_m][side_v]overlay=x=1320:y=50[stage_top];"
    # 2026-07-29 (job 600): "select=" heads the gap branch so frames
    # OUTSIDE the windows are dropped at the source — without it the
    # branch carried the whole story through scale and ffmpeg 8's queued
    # scheduler OOM'd a single 131s compose with 25GB RAM free. This is
    # NOT a cache-invalidating change: the composed hash's ingredients
    # (gapfb windows) are untouched, and the frames INSIDE the enable
    # windows are pixel-identical — cached old outputs remain correct.
    "[v_full]select='not(mod(n,25))+between(t,1.000,2.500)',"
    "scale=1920:1080:force_original_aspect_ratio=increase,"
    "crop=1920:1080,setsar=1[vfs];"
    "[stage_top][vfs]overlay=x=0:y=0:enable='between(t,1.000,2.500)'"
    "[stage_gap];"
    "[stage_gap][2:v]overlay=x='0':y=890:format=auto[stage_lt]",
    "stage_lt", "0:a?")


def test_legacy_graph_byte_identical_plain():
    assert _build_story_filter_graph(**_COMMON) == _GOLDEN_PLAIN


def test_legacy_graph_byte_identical_with_gaps_and_fx():
    got = _build_story_filter_graph(**_COMMON, gap_windows=((1.0, 2.5),),
                                    story_fx_vf="eq=contrast=1.06")
    assert got == _GOLDEN_GAPS


def test_layout_graph_borderless_fullscreen_no_side():
    fc, out, _ = _build_story_filter_graph(
        **{**_COMMON, "main_inner_w": 1920, "main_outer_w": 1920,
           "main_x": 0, "tile_y": 0},
        main_outer_h=912, main_border=0, has_side=False)
    assert "pad=" not in fc.split(";")[0]        # borderless main tile
    assert "[side_v]" not in fc                  # no picture surface
    assert "scale=1920:912:" in fc
    assert "[stage_m][2:v]overlay" in fc         # LT composites onto stage_m


def test_layout_graph_side_tile_own_rect():
    fc, _, _ = _build_story_filter_graph(
        **_COMMON, side_outer_h=432, side_y=86)
    assert "scale=564:426:" in fc                # 432 - 2*3 border
    assert "pad=570:432:3:3" in fc
    assert "overlay=x=1320:y=86[stage_top]" in fc
    assert "overlay=x=30:y=50:shortest=1" in fc  # main row untouched


def test_layout_graph_blur_self_backdrop():
    fc, _, _ = _build_story_filter_graph(
        **_COMMON, bg_mode="blur_self", story_fx_vf="eq=contrast=1.06")
    assert "[0:v]eq=contrast=1.06,split=2[v_tile][v_bgsrc]" in fc
    assert "boxblur=luma_radius=24" in fc
    assert "color=c=black" not in fc             # echo replaces flat colour


def test_layout_graph_blur_self_with_gaps_splits_three():
    fc, _, _ = _build_story_filter_graph(
        **_COMMON, bg_mode="blur_self", gap_windows=((1.0, 2.0),))
    assert "split=3[v_tile][v_full][v_bgsrc]" in fc


def test_layout_graph_image_backdrop():
    fc, _, _ = _build_story_filter_graph(
        **_COMMON, bg_mode="image", bg_image_idx=7)
    assert "[7:v]scale=1920:1080:" in fc
    assert "fps=30,boxblur=luma_radius=18" in fc
    assert "color=c=black" not in fc


# ── Unit 3: mid-story layout moments ─────────────────────────────────
from pipeline_v4.v1_bridge import _sanitize_layout_moments


def _mom(layout, t, dur, transition="push"):
    return {"layout": layout, "t": t, "dur": dur, "transition": transition}


def test_moment_sanitizer_rules():
    dur = 30.0
    # valid fullscreen moment, clamped off the story edges
    out = _sanitize_layout_moments(
        [_mom("news_fullscreen_anchor", 0.2, 6.0)], dur)
    assert len(out) == 1
    g, ts, te, tr = out[0]
    assert g.key == "news_fullscreen_anchor" and ts == 0.75 and tr == "push"
    # too short, picture layouts, unknown keys → dropped
    assert _sanitize_layout_moments([_mom("news_fullscreen_anchor", 5, 1.0)], dur) == []
    assert _sanitize_layout_moments([_mom("news_bulletin_right", 5, 6.0)], dur) == []
    assert _sanitize_layout_moments([_mom("nope", 5, 6.0)], dur) == []
    # overlap: first (by time) wins; cap at 2
    out = _sanitize_layout_moments(
        [_mom("news_fullscreen_anchor", 4, 6.0),
         _mom("sp_documentary", 6, 6.0),
         _mom("news_breaking_full", 14, 4.0),
         _mom("sp_documentary", 20, 4.0)], dur)
    assert [m[0].key for m in out] == ["news_fullscreen_anchor",
                                       "news_breaking_full"]
    # a reference-video cutaway owns its window
    out = _sanitize_layout_moments(
        [_mom("news_fullscreen_anchor", 4, 6.0)], dur,
        cutaways=(("clip.mp4", 5.0, 8.0, "mute", 0.0),))
    assert out == []


def test_moment_graph_fullscreen_push():
    dur = 30.0
    (g, ts, te, tr), = _sanitize_layout_moments(
        [_mom("news_fullscreen_anchor", 5, 6.0)], dur)
    fc, _, _ = _build_story_filter_graph(**_COMMON, moments=((g, ts, te, tr),))
    assert "split=2[v_tile][v_mom0]" in fc
    # HEARTBEAT select (not trim): trim={ts}:{te} emitted nothing until decode
    # reached ts, so the moment overlay's framesync buffered every main frame
    # from 0..ts (~8GB for a t=90 moment) — the job-610 long-source OOM. The
    # not(mod(n,25)) trickle keeps framesync fed before AND after the window
    # (bisect-proven: primer-only still dammed 4.4GB/15s; heartbeat 0.6GB).
    assert ("[v_mom0]select='not(mod(n,25))+between(t,5.000,11.000)',"
            "scale=1920:1080:force_original_aspect_ratio=increase,"
            "crop=1920:1080,setsar=1,format=yuv420p[mom0]") in fc
    assert "enable='between(t,5.000,11.000)'" in fc
    assert "pow(1-(t-5.000)/0.4\\,3)" in fc          # eased push in
    assert "1-pow(1-(t-10.600)/0.4\\,3)" in fc       # eased push out


def test_moment_graph_card_over_echo_cut():
    dur = 30.0
    (g, ts, te, tr), = _sanitize_layout_moments(
        [_mom("studio_bg_center", 5, 6.0, transition="cut")], dur)
    assert not g.video_borderless        # framed card, needs the echo bg
    fc, _, _ = _build_story_filter_graph(**_COMMON, moments=((g, ts, te, tr),))
    assert "split=3[v_tile][v_mom0][v_mombg0]" in fc
    # heartbeat select — branch still pays blur only inside its own window
    assert "[v_mombg0]select='not(mod(n,25))+between(t,5.000,11.000)'," in fc
    assert "boxblur=luma_radius=24" in fc
    assert "[mombg0][momtile0]overlay=x=288:y=86,format=yuv420p[mom0]" in fc
    assert "overlay=x='0':y=0:enable='between(t,5.000,11.000)'" in fc


def test_moment_stage_sits_between_spotlight_and_cutaway():
    dur = 30.0
    (g, ts, te, tr), = _sanitize_layout_moments(
        [_mom("news_fullscreen_anchor", 12, 4.0)], dur)
    fc, _, _ = _build_story_filter_graph(
        **_COMMON, spotlight=((3, 4.0, 8.0),), moments=((g, ts, te, tr),),
        video_spotlight=((4, 20.0, 24.0, "mute", 0.0),))
    assert (fc.index("[stage_sp0]") < fc.index("[stage_mom0]")
            < fc.index("[stage_vsp0]") < fc.index("[stage_lt]"))


def test_no_moments_keeps_graph_byte_identical():
    assert _build_story_filter_graph(**_COMMON, moments=()) == _GOLDEN_PLAIN


def test_renderable_v2_is_substantial_and_16x9_only():
    assert len(ll.RENDERABLE_V2) >= 40
    for k in ll.RENDERABLE_V2:
        assert ll.LAYOUTS[k].aspect == "16:9"


def test_classic_bulletin_resolves_with_picture():
    g = ll.story_geometry("news_bulletin_right")
    assert g is not None
    assert g.picture is not None and g.picture_kind == "carousel"
    assert g.video == (1.5, 4.5, 64, 74)
    assert not g.video_borderless and g.bg == "inherit"


def test_fullscreen_anchor_is_borderless_no_picture():
    g = ll.story_geometry("news_fullscreen_anchor")
    assert g is not None
    assert g.picture is None and g.video_borderless


def test_ots_box_becomes_picture_tile():
    g = ll.story_geometry("news_ots_right")
    assert g is not None
    assert g.picture == (62, 8, 32, 40) and g.picture_kind == "image"


def test_panel_rail_precedent_split_70_30():
    # to_canvas_pcts already ships panel->picture for lib: picks; the
    # per-story resolver keeps that precedent.
    g = ll.story_geometry("news_split_70_30")
    assert g is not None and g.picture_kind == "panel"


def test_unexpressible_layouts_resolve_none():
    for k in ("news_lbar_right", "news_big_graphic", "news_map_focus",
              "news_triple_box", "sp_festival", "sp_sports_board",
              "bg_gradient_card", "studio_bg_carousel", "virtual_set_desk",
              "pod_screen_share"):
        assert ll.story_geometry(k) is None, k


def test_wrong_aspect_resolves_none():
    assert ll.story_geometry("short_torn_card") is None
    assert ll.story_geometry("sq_bulletin") is None
    assert ll.story_geometry("") is None
    assert ll.story_geometry("no_such_layout") is None


def test_pip_variants_resolve_except_circles_and_doubles():
    g = ll.story_geometry("pip_tr_small")
    assert g is not None
    assert len(g.pips) == 1 and g.video_borderless and g.picture is None
    # one picture surface today: doubles + circles stay out (no half-designs)
    assert ll.story_geometry("pip_double") is None
    assert ll.story_geometry("bg_pip_stack") is None
    assert ll.story_geometry("pip_circle_tr") is None
    assert ll.story_geometry("pip_circle_bl") is None


def test_studio_bg_modes():
    assert ll.story_geometry("studio_bg_classic").bg == "bg_video"
    assert ll.story_geometry("photo_wall").bg == "bg_image"
    assert ll.story_geometry("bg_blur_echo").bg == "blur_self"


def test_allowed_for_filters_by_assets():
    rich = ll.allowed_for(has_images=True)
    lean = ll.allowed_for(has_images=False)
    assert set(lean) <= set(rich)
    assert "news_bulletin_right" in rich and "news_bulletin_right" not in lean
    assert "news_fullscreen_anchor" in lean
    assert "photo_wall" not in lean          # image wash needs an image
    assert "bg_blur_echo" in lean            # echoes the clip itself
    assert "pip_tr_small" not in lean        # inset shows a story image


def test_legacy_renderable_untouched_and_subset_of_v2():
    assert ll.RENDERABLE == ("news_bulletin_right", "news_bulletin_left",
                             "news_split_50", "news_split_60_40",
                             "news_split_70_30", "news_ots_right",
                             "news_ots_left")
    assert set(ll.RENDERABLE) <= set(ll.RENDERABLE_V2)


def test_geometry_rects_inside_canvas():
    for k in ll.RENDERABLE_V2:
        g = ll.story_geometry(k)
        for rect in [g.video] + ([g.picture] if g.picture else []) \
                + [(p.x, p.y, p.w, p.h) for p in g.pips]:
            x, y, w, h = rect
            assert 0 <= x <= 100 and 0 <= y <= 100, (k, rect)
            assert 0 < w <= 100 and 0 < h <= 100, (k, rect)
            assert x + w <= 100.01 and y + h <= 100.01, (k, rect)


# ── Unit 4: overlay entrance anims + tile entrance motion ────────────
# KAIZER_V4_OVERLAY_ANIM / KAIZER_V4_LAYOUT_MOTION (both default ON,
# but only ever activated for stories that carry the new features —
# every legacy graph/hash must stay byte-identical).
from types import SimpleNamespace

import pipeline_v4.v1_bridge as vb

# The frozen legacy Director-window pair (format+alpha-fade, constant y).
_LEGACY_DWIN = (
    "[7:v]format=rgba,"
    "fade=t=in:st=1.000:d=0.250:alpha=1[dw0];"
    "[stage_top][dw0]overlay=x=0:y=0:"
    "enable='between(t,1.000,5.000)'[stage_dw0]"
)


def test_dwin_fade_and_legacy_tuples_stay_byte_identical():
    # a 5-tuple (pre-anim caller) and an explicit "fade" 6-tuple must
    # both emit the exact legacy pair — the byte-compat guarantee.
    for win in ((7, 1.0, 5.0, "0", "0"),
                (7, 1.0, 5.0, "0", "0", "fade")):
        fc, _, _ = _build_story_filter_graph(**_COMMON, dwin=(win,))
        assert _LEGACY_DWIN in fc
        assert fc.count("pow(") == 0


def test_dwin_slide_up_rides_y_time_expression():
    fc, _, _ = _build_story_filter_graph(
        **_COMMON, dwin=((7, 1.0, 5.0, "0", "0", "slide_up"),))
    assert "[7:v]format=rgba[dw0]" in fc          # no alpha fade → no ghosting
    assert "fade=t=in" not in fc
    assert ("[stage_top][dw0]overlay=x=0:"
            "y='if(lt(t\\,1.450)\\,1080*pow(1-(t-1.000)/0.45\\,3)\\,0)':"
            "enable='between(t,1.000,5.000)'[stage_dw0]") in fc


def test_dwin_slide_down_drops_from_above():
    fc, _, _ = _build_story_filter_graph(
        **_COMMON, dwin=((7, 2.0, 6.0, "0", "0", "slide_down"),))
    assert ("y='if(lt(t\\,2.450)\\,-1080*pow(1-(t-2.000)/0.45\\,3)\\,0)'"
            in fc)


def test_dwin_short_window_falls_back_to_fade():
    # a window shorter than anim+settle would still be mid-flight when it
    # disappears — keep the legacy fade there.
    fc, _, _ = _build_story_filter_graph(
        **_COMMON, dwin=((7, 1.0, 1.5, "0", "0", "slide_up"),))
    assert "fade=t=in:st=1.000:d=0.250:alpha=1[dw0]" in fc
    assert "pow(" not in fc


def test_side_exprs_empty_keeps_graph_byte_identical():
    assert _build_story_filter_graph(
        **_COMMON, side_x_expr="", side_y_expr="") == _GOLDEN_PLAIN


def test_side_expr_nonempty_is_quoted_into_overlay():
    fc, _, _ = _build_story_filter_graph(
        **_COMMON, side_x_expr="if(lt(t\\,0.5)\\,X\\,1320)")
    assert ("[stage_m][side_v]overlay="
            "x='if(lt(t\\,0.5)\\,X\\,1320)':y=50[stage_top]") in fc
    fc, _, _ = _build_story_filter_graph(
        **_COMMON, side_y_expr="if(lt(t\\,0.5)\\,Y\\,50)")
    assert ("[stage_m][side_v]overlay="
            "x=1320:y='if(lt(t\\,0.5)\\,Y\\,50)'[stage_top]") in fc


def test_overlay_anim_family_mapping_fail_soft():
    assert vb._overlay_anim("breaking_news_banner") == "slide_up"   # banner
    assert vb._overlay_anim("tag_exclusive") == "slide_up"          # headline_tag
    assert vb._overlay_anim("live_bug") == "slide_down"             # bug
    assert vb._overlay_anim("story_progress") == "slide_down"       # progress
    assert vb._overlay_anim("exclusive_stamp") == "fade"            # stamp keeps fade
    assert vb._overlay_anim("countdown_card") == "fade"             # cover treatment
    assert vb._overlay_anim("viewfinder_hud") == "fade"             # cover treatment
    assert vb._overlay_anim("no_such_overlay") == "fade"            # unknown id
    assert vb._overlay_anim("") == "fade"


def test_flag_on_kill_switch_convention(monkeypatch):
    monkeypatch.delenv("KAIZER_V4_OVERLAY_ANIM", raising=False)
    assert vb._flag_on("KAIZER_V4_OVERLAY_ANIM")            # default ON
    for off in ("0", "off", "FALSE", " Off "):
        monkeypatch.setenv("KAIZER_V4_OVERLAY_ANIM", off)
        assert not vb._flag_on("KAIZER_V4_OVERLAY_ANIM")
    monkeypatch.setenv("KAIZER_V4_OVERLAY_ANIM", "1")
    assert vb._flag_on("KAIZER_V4_OVERLAY_ANIM")


def _compose_filter_graph(monkeypatch, tmp_path, **kw):
    """Run _compose_v4_bulletin_story with ffmpeg + the LT renderer
    stubbed out and return the filter_complex it would have launched —
    the production path (geometry, gates, tuple plumbing) minus the
    encode."""
    captured = {}
    monkeypatch.setattr(vb, "_run_ffmpeg",
                        lambda cmd, **_k: captured.update(cmd=cmd))
    monkeypatch.setattr(vb, "_ffmpeg_bin", lambda: "ffmpeg")
    monkeypatch.setattr(vb, "_enc_args", lambda *a, **k: [])
    monkeypatch.setattr(vb, "_dec_args", lambda *a, **k: [])
    import pipeline_core.longform_compose as lfc
    monkeypatch.setattr(lfc, "render_lower_third",
                        lambda meta, font, path: (path, 800))
    vb._compose_v4_bulletin_story(
        story_clip_path="story.mp4",
        story_meta=SimpleNamespace(story_index=0),
        out_path=str(tmp_path / "out.mp4"),
        sidebar_path="side.png", ticker_path="",
        channel_bug_path=None, font_path=None,
        work_dir=str(tmp_path), apply_ticker=False, **kw)
    cmd = captured["cmd"]
    return cmd[cmd.index("-filter_complex") + 1]


def test_tile_motion_right_column_slides_from_nearest_edge(monkeypatch, tmp_path):
    monkeypatch.delenv("KAIZER_V4_LAYOUT_MOTION", raising=False)
    g = ll.StoryGeometry(key="t_right", video=(2, 10, 60, 70),
                         picture=(70, 10, 28, 45), picture_kind="image")
    fc = _compose_filter_graph(monkeypatch, tmp_path, story_layout=g)
    # picture rect hugs the right edge (38px) → slides in from the right,
    # easing into its resting x=1344 (0.5s cubic ease-out), constant after.
    assert ("[stage_m][side_v]overlay="
            "x='if(lt(t\\,0.5)\\,1344+576*pow(1-t/0.5\\,3)\\,1344)':"
            "y=108[stage_top]") in fc


def test_tile_motion_only_for_layout_or_theme_stories(monkeypatch, tmp_path):
    monkeypatch.delenv("KAIZER_V4_LAYOUT_MOTION", raising=False)
    # legacy story (no per-story layout, no theme): constant x/y, no motion
    fc = _compose_filter_graph(monkeypatch, tmp_path)
    assert "pow(1-t/0.5" not in fc
    assert "[stage_m][side_v]overlay=x=1320:y=50[stage_top]" in fc


def test_tile_motion_kill_switch_restores_legacy(monkeypatch, tmp_path):
    monkeypatch.setenv("KAIZER_V4_LAYOUT_MOTION", "0")
    g = ll.StoryGeometry(key="t_right", video=(2, 10, 60, 70),
                         picture=(70, 10, 28, 45), picture_kind="image")
    fc = _compose_filter_graph(monkeypatch, tmp_path, story_layout=g)
    assert "pow(1-t/0.5" not in fc
    assert "overlay=x=1344:y=108[stage_top]" in fc


def test_directive_window_anim_slot_flows_to_graph(monkeypatch, tmp_path):
    fc = _compose_filter_graph(
        monkeypatch, tmp_path,
        directive_windows=(("g.png", 1.0, 5.0, "full", "slide_up"),
                           ("c.png", 2.0, 8.0, "caption", "slide_up"),
                           ("h.png", 3.0, 7.0, "full")))
    # graphic slides; captions are FORCED to fade; a legacy 4-tuple
    # (older caller / cached spec) keeps the fade too.
    assert "[3:v]format=rgba[dw0]" in fc
    assert "y='if(lt(t\\,1.450)\\,1080*pow(1-(t-1.000)/0.45\\,3)\\,0)'" in fc
    assert "[4:v]format=rgba,fade=t=in:st=2.000:d=0.250:alpha=1[dw1]" in fc
    assert "[5:v]format=rgba,fade=t=in:st=3.000:d=0.250:alpha=1[dw2]" in fc
