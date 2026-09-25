"""Director A/V rendering tests — emphasis punch-in + sound design.

Unit 1 — _sanitize_emphasis_windows: pure punch-window rules (clamp off
story edges, ~0.45s duration, cap 2, blocked-window and self-overlap
skips, garbage tolerance).
Unit 2 — the per-story filter graph: the 1.06x punch branch (extra split
consumer + enable windows) appears ONLY when emphasis windows exist —
the emphasis-free graph must stay CHARACTER-IDENTICAL to the legacy one
(the byte-compat goldens in test_layout_live / test_carousel_gaps keep
pinning that; here we pin the inverse).
Unit 3 — |emph: conditional cache fingerprint (the CACHE RULE: a |tag:
appended only when active, never a renderer-version bump).
Unit 4 — sound-design selection helpers: Director sting / ui_sound /
bed_on consumption, impact-key resolution, and _STING_BY_PACK coverage
of every trailer_styles pack.
"""
from types import SimpleNamespace

from pipeline_v4.v1_bridge import (
    _STING_BY_PACK,
    _bed_wanted,
    _build_story_filter_graph,
    _custom_short_fx_handled,
    _emphasis_fingerprint,
    _flag_on,
    _impact_key,
    _sanitize_emphasis_windows,
    _sting_key_for,
    _ui_key_for,
)

_COMMON = dict(canvas_w=1920, canvas_h=1080, main_inner_w=1254,
               side_inner_w=564, tile_inner_h=794, main_outer_w=1260,
               side_outer_w=570, tile_outer_h=800, main_x=30, side_x=1320,
               tile_y=50, lt_y=890, ticker_y=1030, border_colour="white",
               bg_color="black", lt_x_expr="0", ticker_speed_px_s=200.0,
               has_bg=False, bg_input_idx=None, apply_ticker=False,
               ticker_in_idx=None, has_bug=False, bug_in_idx=None,
               has_wm=False, wm_in_idx=None)


# ── Unit 1: punch-window sanitize (pure) ─────────────────────────────

def test_emphasis_basic_window():
    assert _sanitize_emphasis_windows([5.0], 30.0) == [(5.0, 5.45)]


def test_emphasis_clamps_off_story_edges():
    # early beat clamps to the 0.75s edge guard
    assert _sanitize_emphasis_windows([0.1], 30.0) == [(0.75, 1.2)]
    # late beat clamps so the whole window ends >=0.75s before the edge
    (ts, te), = _sanitize_emphasis_windows([29.9], 30.0)
    assert te <= 30.0 - 0.75 + 1e-9
    assert abs((te - ts) - 0.45) < 1e-9


def test_emphasis_cap_two():
    wins = _sanitize_emphasis_windows([5.0, 10.0, 15.0, 20.0], 60.0)
    assert wins == [(5.0, 5.45), (10.0, 10.45)]


def test_emphasis_blocked_window_skipped():
    # a full-screen state (moment/cutaway/spotlight/PiP) owns [4, 6]
    assert _sanitize_emphasis_windows([5.0], 30.0, blocked=[(4.0, 6.0)]) == []
    # a beat clear of the block survives
    assert _sanitize_emphasis_windows(
        [5.0, 10.0], 30.0, blocked=[(4.0, 6.0)]) == [(10.0, 10.45)]


def test_emphasis_self_overlap_first_wins():
    wins = _sanitize_emphasis_windows([5.0, 5.2], 30.0)
    assert wins == [(5.0, 5.45)]


def test_emphasis_short_story_and_garbage():
    assert _sanitize_emphasis_windows([0.5], 1.5) == []      # no room to punch
    assert _sanitize_emphasis_windows(["x", None], 30.0) == []
    assert _sanitize_emphasis_windows([], 30.0) == []
    assert _sanitize_emphasis_windows([5.0], None) == []
    # blocked entries with garbage shapes are tolerated
    assert _sanitize_emphasis_windows(
        [5.0], 30.0, blocked=[("a", "b"), (None,), ()]) == [(5.0, 5.45)]


def test_emphasis_kill_switch_flag(monkeypatch):
    # the render path gates on _flag_on("KAIZER_V4_EMPHASIS_PUNCH") — default ON
    monkeypatch.delenv("KAIZER_V4_EMPHASIS_PUNCH", raising=False)
    assert _flag_on("KAIZER_V4_EMPHASIS_PUNCH") is True
    monkeypatch.setenv("KAIZER_V4_EMPHASIS_PUNCH", "0")
    assert _flag_on("KAIZER_V4_EMPHASIS_PUNCH") is False


# ── Unit 2: filter-graph assertions ──────────────────────────────────

def test_graph_no_emphasis_has_no_punch_branch():
    fc, _, _ = _build_story_filter_graph(**_COMMON)
    assert "v_emph" not in fc
    assert "split=" not in fc          # single consumer → no split at all
    # explicit empty tuple is byte-identical to the default
    assert _build_story_filter_graph(**_COMMON, emphasis=()) == \
        _build_story_filter_graph(**_COMMON)


def test_graph_emphasis_punch_branch():
    fc, out, amap = _build_story_filter_graph(
        **_COMMON, emphasis=((5.0, 5.45),))
    # extra split consumer (the spotlight idiom)
    assert "[0:v]split=2[v_tile][v_emph];" in fc
    # 1.06x cover: 1920*1.06→2035, 1080*1.06→1145, cropped back to canvas.
    # select= heads the branch (job 600 memory guard): frames outside the
    # punch windows are dropped at the source — output-identical, and the
    # composed hash's ingredients are untouched (cache-safe). The heartbeat
    # (not eq(n,0)) keeps framesync fed AFTER the window too — a primer-only
    # branch dammed 4.4GB/15s of main frames on long stories (job 610).
    assert ("[v_emph]select='not(mod(n,25))+between(t,5.000,5.450)',"
            "scale=2035:1145:force_original_aspect_ratio=increase,"
            "crop=1920:1080,setsar=1,format=yuv420p[epch]") in fc
    # hard cut: enable-gated opaque overlay, NO alpha fade on the branch
    assert "[epch]overlay=x=0:y=0:enable='between(t,5.000,5.450)'" in fc
    assert "fade=" not in fc
    # broadcast furniture still on top: LT composites after the punch
    assert "[stage_emph][2:v]overlay" in fc
    assert out == "stage_lt" and amap == "0:a?"


def test_graph_emphasis_two_windows_or_enable():
    fc, _, _ = _build_story_filter_graph(
        **_COMMON, emphasis=((5.0, 5.45), (12.0, 12.45)))
    assert ("enable='between(t,5.000,5.450)+between(t,12.000,12.450)'"
            in fc)


def test_graph_emphasis_stacks_with_gaps_and_fx():
    fc, _, _ = _build_story_filter_graph(
        **_COMMON, gap_windows=((1.0, 2.5),), story_fx_vf="eq=contrast=1.06",
        emphasis=((5.0, 5.45),))
    # grade applied ONCE before the split; emphasis label appended LAST
    assert "[0:v]eq=contrast=1.06,split=3[v_tile][v_full][v_emph];" in fc
    # gap fallback unchanged, punch drawn after it
    assert "[stage_gap][epch]" in fc or "[epch]overlay" in fc
    # the gap-only golden (test_layout_live) is unaffected: same call
    # without emphasis still splits to exactly 2
    fc2, _, _ = _build_story_filter_graph(
        **_COMMON, gap_windows=((1.0, 2.5),), story_fx_vf="eq=contrast=1.06")
    assert "split=2[v_tile][v_full];" in fc2 and "v_emph" not in fc2


# ── Unit 3: |emph: conditional cache fingerprint ─────────────────────

def test_emph_fingerprint_conditional():
    assert _emphasis_fingerprint([]) == ""          # legacy hash untouched
    assert _emphasis_fingerprint(()) == ""
    assert _emphasis_fingerprint([(5.0, 5.45)]) == "|emph:5.00-5.45"
    assert (_emphasis_fingerprint([(0.75, 1.2), (12.0, 12.45)])
            == "|emph:0.75-1.20,12.00-12.45")


# ── Unit 4: sound-design selection ───────────────────────────────────

def test_sting_explicit_directive_pick_wins():
    d = SimpleNamespace(sting="sting_breaking", mood="comedy")
    assert _sting_key_for(d) == "sting_breaking"


def test_sting_falls_back_to_mood_pack():
    assert _sting_key_for(SimpleNamespace(mood="crime")) == "sting_negative"
    assert _sting_key_for(SimpleNamespace(sting="", mood="comedy")) == \
        "sting_positive"
    # a stray non-library id must not silence the open — pack map wins
    assert _sting_key_for(SimpleNamespace(sting="not_a_key",
                                          mood="breaking_news")) == \
        "sting_breaking"
    # unknown mood → the classic news sting
    assert _sting_key_for(SimpleNamespace(mood="martian")) == "sting_news"
    assert _sting_key_for(SimpleNamespace()) == "sting_news"


def test_ui_sound_selection():
    assert _ui_key_for(SimpleNamespace()) == "ui_pop"
    assert _ui_key_for(SimpleNamespace(ui_sound="")) == "ui_pop"
    assert _ui_key_for(SimpleNamespace(ui_sound="ui_click")) == "ui_click"
    assert _ui_key_for(SimpleNamespace(ui_sound="bogus")) == "ui_pop"


def test_impact_key_is_a_real_impact():
    from pipeline_v4.sound_library import LIBRARY
    k = _impact_key()
    assert k == "impact_punch"                      # present today
    assert k in LIBRARY and LIBRARY[k].family == "impacts"


def test_bed_majority_vote():
    on = SimpleNamespace(bed_on=True)
    off = SimpleNamespace(bed_on=False)
    unset = SimpleNamespace()                       # missing attr = on
    nonev = SimpleNamespace(bed_on=None)            # None = unset = on
    assert _bed_wanted([]) is True                  # no directives → default
    assert _bed_wanted([unset, unset]) is True
    assert _bed_wanted([on, off]) is True           # tie keeps the bed
    assert _bed_wanted([off, off, on]) is False     # dominant no-bed vote
    assert _bed_wanted([off, nonev, unset]) is True
    assert _bed_wanted([None, off]) is False        # Nones don't vote


def test_sting_by_pack_covers_all_styles():
    from pipeline_v4.sound_library import LIBRARY
    from pipeline_v4.trailer_styles import STYLES
    missing = set(STYLES) - set(_STING_BY_PACK)
    assert not missing, f"packs without a sting mapping: {sorted(missing)}"
    bad = {v for v in _STING_BY_PACK.values() if v not in LIBRARY}
    assert not bad, f"mapped stings missing from the sound library: {bad}"


def test_custom_short_fx_capability_probe():
    # never raises, and reflects the engine's RenderRequest surface —
    # the per-clip effects field is threaded on this branch, so the
    # whole-frame post-pass must step aside for custom shorts.
    assert _custom_short_fx_handled() is True
