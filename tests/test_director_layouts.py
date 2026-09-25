"""LIVE LAYOUTS Unit 4 — the Director plans per-story layouts + moments.

Covers: the layouts vocabulary, sanitize_plan's per-story asset
awareness (image layouts only for stories that HAVE images), moment
validation, and the formula fallback's deterministic screen rotation."""
from types import SimpleNamespace

from pipeline_v4 import director as dr
from pipeline_v4 import layout_library as ll


def _story(idx, dur=30.0, images=None):
    return SimpleNamespace(story_index=idx, video_t_start=0.0,
                           video_t_end=float(dur),
                           images=list(images or []))


def _plan_row(idx, **kw):
    row = {"index": idx, "mood": "news", "transition_in": "fade",
           "fx": [], "overlays": [], "captions": "none", "emphasis": []}
    row.update(kw)
    return row


def test_vocab_layouts_bucket_matches_renderable_v2():
    v = dr._vocab()
    assert v["layouts"] == set(ll.RENDERABLE_V2)
    picked = dr._vocab({"layouts": ["news_ots_right", "not_a_layout"]})
    assert picked["layouts"] == {"news_ots_right"}


def test_sanitize_keeps_valid_layout_for_image_story():
    stories = [_story(0, images=[{"src": "a.png"}])]
    plan = dr.sanitize_plan(
        {"stories": [_plan_row(0, layout="news_ots_right")]},
        stories=stories, category="news")
    assert plan[0].layout == "news_ots_right"


def test_sanitize_drops_image_layout_on_imageless_story():
    stories = [_story(0, images=[])]
    plan = dr.sanitize_plan(
        {"stories": [_plan_row(0, layout="news_ots_right")]},
        stories=stories, category="news")
    assert plan[0].layout == ""          # needs a picture the story lacks
    plan2 = dr.sanitize_plan(
        {"stories": [_plan_row(0, layout="news_fullscreen_anchor")]},
        stories=stories, category="news")
    assert plan2[0].layout == "news_fullscreen_anchor"


def test_sanitize_drops_unknown_layout():
    stories = [_story(0, images=[{"src": "a.png"}])]
    plan = dr.sanitize_plan(
        {"stories": [_plan_row(0, layout="totally_made_up")]},
        stories=stories, category="news")
    assert plan[0].layout == ""


def test_sanitize_moments_rules():
    stories = [_story(0, dur=30.0, images=[{"src": "a.png"}]),
               _story(1, dur=6.0)]
    plan = dr.sanitize_plan(
        {"stories": [
            _plan_row(0, layout_moments=[
                {"layout": "news_fullscreen_anchor", "t": 8, "dur": 5},
                {"layout": "news_bulletin_right", "t": 14, "dur": 5},  # picture layout → out
                {"layout": "news_fullscreen_anchor", "t": 20, "dur": 1.0},  # too short
            ]),
            _plan_row(1, layout_moments=[
                {"layout": "news_fullscreen_anchor", "t": 2, "dur": 3}]),
        ]},
        stories=stories, category="news")
    m0 = plan[0].layout_moments
    assert len(m0) == 1
    assert m0[0]["layout"] == "news_fullscreen_anchor"
    assert m0[0]["transition"] == "push"
    assert plan[1].layout_moments == []          # story too short (<8s)


def test_formula_plan_rotates_layouts_by_assets():
    stories = [_story(i, images=([{"src": "a.png"}] if i % 2 == 0 else []))
               for i in range(6)]
    plan = dr.formula_plan(stories, "news")
    lays = [plan[i].layout for i in range(6)]
    assert len(set(lays)) >= 2               # varies across stories
    for i, lay in enumerate(lays):
        if not lay:
            continue
        g = ll.story_geometry(lay)
        assert g is not None
        if i % 2 == 1:                       # image-less stories
            assert not (g.picture or g.pips or g.bg == "bg_image")
    assert plan[0].layout_moments == []      # deterministic path: no moments


# ── Creative vocabulary: grades / stings / ui sounds / bed toggle ─────

def test_vocab_grade_and_sound_buckets_exist_and_are_nonempty():
    from pipeline_v4.color_grades import GRADES
    from pipeline_v4.sound_library import LIBRARY
    v = dr._vocab()
    assert v["grades"] == set(GRADES) and v["grades"]
    assert v["stings"] and all(k.startswith("sting_") for k in v["stings"])
    assert v["ui_sounds"] and all(k.startswith("ui_") for k in v["ui_sounds"])
    assert v["stings"] <= set(LIBRARY) and v["ui_sounds"] <= set(LIBRARY)
    # user override constrains the bucket; stale ids can't leak in
    picked = dr._vocab({"grades": ["monochrome", "not_a_grade"]})
    assert picked["grades"] == {"monochrome"}


def test_sanitize_keeps_valid_grade_sting_ui_and_drops_junk():
    stories = [_story(0)]
    plan = dr.sanitize_plan(
        {"stories": [_plan_row(0, grade="bleach_bypass", sting="sting_news",
                               ui_sound="ui_tick", bed_on=False)]},
        stories=stories, category="news")
    d = plan[0]
    assert d.grade == "bleach_bypass"
    assert d.sting == "sting_news"
    assert d.ui_sound == "ui_tick"
    assert d.bed_on is False
    # junk ids (and a non-sting sound in the sting slot) drop to defaults;
    # a non-boolean bed_on keeps the bed
    plan2 = dr.sanitize_plan(
        {"stories": [_plan_row(0, grade="lut_of_doom", sting="impact_deep",
                               ui_sound="whoosh_air", bed_on="nope")]},
        stories=stories, category="news")
    d2 = plan2[0]
    assert d2.grade == ""
    assert d2.sting == ""
    assert d2.ui_sound == ""
    assert d2.bed_on is True


def test_sanitize_grade_user_pick_snaps_invalid_llm_choice():
    stories = [_story(0)]
    plan = dr.sanitize_plan(
        {"stories": [_plan_row(0, grade="totally_fake")]},
        stories=stories, category="news",
        user_picks={"grades": ["monochrome"]})
    assert plan[0].grade == "monochrome"     # snap to THEIR pin, not ""
    plan2 = dr.sanitize_plan(
        {"stories": [_plan_row(0, sting="not_a_sting")]},
        stories=stories, category="news",
        user_picks={"stings": ["sting_positive"]})
    assert plan2[0].sting == "sting_positive"


def test_parse_style_directives_maps_grade_and_sound_keys():
    picks, cat = dr.parse_style_directives(
        {"color_grades": ["monochrome"], "sound": ["sting_news"]})
    assert picks == {"grades": ["monochrome"], "stings": ["sting_news"]}
    assert cat is None
    picks2, _ = dr.parse_style_directives({"grades": "vibrant_pop",
                                           "sounds": ["sting_breaking"]})
    assert picks2 == {"grades": ["vibrant_pop"],
                      "stings": ["sting_breaking"]}


def test_story_fx_chain_uses_override_grade_when_set_pack_grade_when_not():
    from pipeline_v4.color_grades import get_grade_vf
    from pipeline_v4.trailer_styles import STYLES
    pack = STYLES["news"]
    # no override → the pack's own grade opens the chain (byte-compat)
    d_plain = dr.StoryDirective(mood="news")
    chain_plain = dr.story_fx_chain(d_plain)
    assert chain_plain.startswith(pack.grade)
    # override → the grade fragment REPLACES pack.grade, extra_vf kept
    d_ovr = dr.StoryDirective(mood="news", grade="monochrome")
    chain_ovr = dr.story_fx_chain(d_ovr)
    frag = get_grade_vf("monochrome")
    assert chain_ovr.startswith(frag)
    assert not chain_ovr.startswith(pack.grade)
    if pack.extra_vf and ";" not in frag + "," + pack.extra_vf:
        assert pack.extra_vf in chain_ovr
    # garnish fx still composed after the grade
    d_fx = dr.StoryDirective(mood="news", grade="monochrome",
                             fx=["vignette_soft"])
    chain_fx = dr.story_fx_chain(d_fx)
    assert chain_fx.startswith(frag) and len(chain_fx) > len(chain_ovr)


def test_formula_plan_look_sound_defaults_and_user_pins():
    plan = dr.formula_plan([_story(0)], "news")
    d = plan[0]
    assert d.grade == "" and d.sting == "" and d.ui_sound == ""
    assert d.bed_on is True
    plan2 = dr.formula_plan([_story(0)], "news",
                            user_picks={"grades": ["vibrant_pop"],
                                        "stings": ["sting_positive"],
                                        "ui_sounds": ["ui_pop"]})
    d2 = plan2[0]
    assert d2.grade == "vibrant_pop"
    assert d2.sting == "sting_positive"
    assert d2.ui_sound == "ui_pop"
    assert d2.bed_on is True
