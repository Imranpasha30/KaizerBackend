"""Director enterprise upgrade — the pure/deterministic parts.

Offline: pacing sensor math, variety enforcement, the supervising-editor
quality review, and the vision sensor's gates. The LLM calls themselves
stay fail-soft behind plan_direction's existing formula fallback.
"""
import pytest

from pipeline_v4 import director as dr
from pipeline_v4.director import StoryDirective


# ── sense_pacing ───────────────────────────────────────────────────

def _words(ts):
    return [{"w": f"w{i}", "s": t} for i, t in enumerate(ts)]


def test_pacing_wpm_pauses_and_hot_moments():
    # 30 words over ~60s with a fat pause at 20→23s
    ts = [i * 0.7 for i in range(28)] + [23.0, 24.0]
    out = dr.sense_pacing({"0": _words(ts)}, {0: 60.0})
    assert 0 in out
    p = out[0]
    assert p["wpm"] == 30  # 30 words / 60s
    assert any(abs(a - 18.9) < 0.2 and g > 3.0 for a, g in p["pauses"])
    assert any(abs(h - 23.0) < 0.01 for h in p["hot_moments"])


def test_pacing_ignores_sparse_or_short():
    assert dr.sense_pacing({"0": _words([1, 2, 3])}, {0: 60.0}) == {}
    assert dr.sense_pacing({"0": _words([0.5 * i for i in range(20)])},
                           {0: 3.0}) == {}
    assert dr.sense_pacing({}, {}) == {}
    assert dr.sense_pacing({"bad": None}, {0: 10.0}) == {}


# ── _enforce_variety ───────────────────────────────────────────────

VOCAB = {"transitions": {"fade", "whip_left", "glitch_slices"}}


def _plan(transitions):
    return {i: StoryDirective(mood="news", transition_in=t)
            for i, t in enumerate(transitions)}


def test_variety_rotates_adjacent_repeats():
    plan = _plan(["fade", "fade", "fade"])
    fixed = dr._enforce_variety(plan, VOCAB)
    assert fixed >= 1
    ts = [plan[i].transition_in for i in range(3)]
    # the contract: NO adjacent pair may repeat afterwards
    assert all(a != b for a, b in zip(ts, ts[1:]))
    assert ts[0] == "fade"           # the first story is never rewritten


def test_variety_leaves_varied_plans_alone():
    plan = _plan(["fade", "whip_left", "fade"])
    assert dr._enforce_variety(plan, VOCAB) == 0


def test_variety_noop_on_tiny_vocab_or_plan():
    assert dr._enforce_variety(_plan(["fade", "fade"]),
                               {"transitions": {"fade"}}) == 0
    assert dr._enforce_variety(_plan(["fade"]), VOCAB) == 0


# ── _plan_quality_issues ───────────────────────────────────────────

def test_review_flags_static_long_story():
    plan = {0: StoryDirective(mood="news")}
    issues = dr._plan_quality_issues(plan, {0: 90.0})
    assert any("ZERO layout_moments" in i for i in issues)


def test_review_flags_monotone_and_missing_punch():
    plan = {i: StoryDirective(mood="news") for i in range(3)}
    words = {"0": _words([1, 2, 3])}
    issues = dr._plan_quality_issues(plan, {i: 20.0 for i in range(3)}, words)
    assert any("monotone" in i for i in issues)
    assert any("no emphasis" in i for i in issues)
    assert any("no overlays" in i for i in issues)


def test_review_quiet_on_a_good_plan():
    plan = {
        0: StoryDirective(mood="news", emphasis=[3.0],
                          overlays=[{"id": "x", "t": 1.0}],
                          layout_moments=[{"layout": "a", "t": 20.0,
                                          "dur": 5.0, "transition": "push"}]),
        1: StoryDirective(mood="crime", emphasis=[2.0]),
    }
    assert dr._plan_quality_issues(plan, {0: 60.0, 1: 30.0},
                                   {"0": _words([1, 2])}) == []


# ── sense_frames gates ─────────────────────────────────────────────

def test_vision_kill_switch(monkeypatch, tmp_path):
    monkeypatch.setenv("KAIZER_V4_VISION_SENSE", "0")
    src = tmp_path / "v.mp4"
    src.write_bytes(b"x")
    assert dr.sense_frames(str(src), []) == {}


def test_vision_missing_source(monkeypatch, tmp_path):
    monkeypatch.setenv("KAIZER_V4_VISION_SENSE", "1")
    assert dr.sense_frames(str(tmp_path / "nope.mp4"), []) == {}


# ── sensor decode cost (job 610: both media sensors burned their FULL
#    240s timeout on a 27-min source and returned NOTHING) ──────────

class _R:
    def __init__(self, stderr=""):
        self.stderr = stderr


def test_scene_cuts_decodes_downscaled_no_audio(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return _R("frame ... pts_time:12.50 ...\n... pts_time:99.10 ...")

    monkeypatch.setattr(dr.subprocess, "run", fake_run)
    out = dr.sense_scene_cuts("src.mp4")
    assert out == [12.5, 99.1]
    joined = " ".join(seen["cmd"])
    assert "scale=320:-2" in joined     # downscaled decode — same cuts, ~10x faster
    assert "-an" in seen["cmd"]         # no audio decode on the video pass


def test_scene_cuts_salvages_partial_on_timeout(monkeypatch):
    def fake_run(cmd, **kw):
        raise dr.subprocess.TimeoutExpired(
            cmd, 240, stderr="... pts_time:5.00 ... pts_time:44.25 ...")

    monkeypatch.setattr(dr.subprocess, "run", fake_run)
    # Old behaviour: timeout → [] (4 minutes burned, zero signal).
    assert dr.sense_scene_cuts("src.mp4") == [5.0, 44.25]


def test_energy_is_audio_only_and_salvages_timeout(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return _R("[Parsed_ebur128] t: 1.0   M: -20.0 \n"
                  "[Parsed_ebur128] t: 2.0   M: -18.0 \n"
                  "[Parsed_ebur128] t: 3.0   M: -50.0 \n"
                  "[Parsed_ebur128] t: 5.0   M: -50.0 \n"
                  "[Parsed_ebur128] t: 6.0   M: -19.0 ")

    monkeypatch.setattr(dr.subprocess, "run", fake_run)
    out = dr.sense_energy_peaks("src.mp4")
    assert "-vn" in seen["cmd"]         # ebur128 needs NO video decode
    assert out["peaks"]                  # loudest moments extracted
    assert out["silences"] == [[3.0, 5.0]]

    def fake_timeout(cmd, **kw):
        raise dr.subprocess.TimeoutExpired(
            cmd, 240, stderr="[Parsed_ebur128] t: 1.0   M: -20.0 \n"
                             "[Parsed_ebur128] t: 2.0   M: -18.0 \n"
                             "[Parsed_ebur128] t: 3.0   M: -17.0 \n"
                             "[Parsed_ebur128] t: 4.0   M: -16.0 \n"
                             "[Parsed_ebur128] t: 5.0   M: -15.0 \n"
                             "[Parsed_ebur128] t: 6.0   M: -14.0 \n"
                             "[Parsed_ebur128] t: 7.0   M: -13.0 \n"
                             "[Parsed_ebur128] t: 8.0   M: -12.0 ")

    monkeypatch.setattr(dr.subprocess, "run", fake_timeout)
    out2 = dr.sense_energy_peaks("src.mp4")
    assert out2.get("peaks")            # partial readings salvaged, not {}


# ── plan token budget (big-bulletin hardening) ─────────────────────

def test_plan_token_budget_small_bulletins_unchanged(monkeypatch):
    # <=10 stories keep the original 8192 → typical jobs render identically.
    monkeypatch.delenv("KAIZER_V4_DIRECTOR_MAX_TOKENS", raising=False)
    assert dr._plan_token_budget(1) == 8192
    assert dr._plan_token_budget(4) == 8192
    assert dr._plan_token_budget(10) == 8192


def test_plan_token_budget_scales_for_big_bulletins(monkeypatch):
    # A 25-story plan overflowed a fixed 8192 → tail stories dropped to the
    # formula baseline. Now the ceiling grows with the story count.
    monkeypatch.delenv("KAIZER_V4_DIRECTOR_MAX_TOKENS", raising=False)
    assert dr._plan_token_budget(25) == 25 * 500 + 3072
    assert dr._plan_token_budget(25) > 8192
    assert dr._plan_token_budget(1000) == 60000        # capped below 65536


def test_plan_token_budget_env_override_and_bad_input(monkeypatch):
    monkeypatch.setenv("KAIZER_V4_DIRECTOR_MAX_TOKENS", "20000")
    assert dr._plan_token_budget(4) == 20000
    monkeypatch.setenv("KAIZER_V4_DIRECTOR_MAX_TOKENS", "garbage")
    assert dr._plan_token_budget(4) == 8192            # unparseable → ignored
    monkeypatch.delenv("KAIZER_V4_DIRECTOR_MAX_TOKENS", raising=False)
    assert dr._plan_token_budget(None) == 8192         # bad count → floor
    assert dr._plan_token_budget("x") == 8192


def test_review_plan_always_returns_tuple(monkeypatch):
    """Regression: _review_plan must ALWAYS return (plan, adopted). The old
    `if not revised: return plan` bare-dict return unpacked wrong at the
    caller (`plan, _adopted = _review_plan(...)`), throwing and dropping the
    WHOLE Director to the formula baseline exactly when a revision came back
    empty."""
    from google.genai import types as gt
    from pipeline_v4.trim_engine import _loads_lenient

    class _Resp:
        text = "{}"

    class _Models:
        def generate_content(self, **kw):
            return _Resp()

    class _Client:
        models = _Models()

    # Force the "revision produced nothing usable" branch.
    monkeypatch.setattr(dr, "sanitize_plan", lambda **kw: {})
    plan = {0: StoryDirective(mood="news")}
    out = dr._review_plan(
        client=_Client(), genai_types=gt, loads_lenient=_loads_lenient,
        model="gemini-2.5-flash", user_prompt="u", plan=plan,
        issues=["something"], stories=[], category="news",
        user_picks=None, durs={0: 10.0}, words_by_story={})
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0] is plan and out[1] is False
