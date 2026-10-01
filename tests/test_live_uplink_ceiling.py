"""A file must fit what the LINK can upload, not only what YouTube accepts.

The measurement behind this: 0.71-0.72 Mbps up, 14.3 Mbps down, taken twice on
two different network stacks, and independently corroborated by YouTube itself
reporting 577 Kbps arriving during the broadcast. On that link a 4.3 Mbps file
cannot be delivered -- but it is far under YouTube's 7500 kbps ceiling for 720p,
so the checker returned "none", it streamed as-is, and YouTube reported
videoIngestionStarved for two hours while every signal on our side read healthy.

The tests that matter here are the WIRING ones. A ceiling the checker accepts
but no caller passes changes nothing at all, and that is the exact shape of
defect this codebase produces over and over: built, tested, and reached by
nothing.
"""
import ast
import inspect
import os
import pathlib

import pytest

from kaizer_live import checker


def _verdict(height, kbps, link_kbps):
    """Run the real evaluate() over a probe shaped like ffprobe's output."""
    info = {
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "width": int(height * 16 / 9),
             "height": height, "pix_fmt": "yuv420p", "avg_frame_rate": "30/1",
             "r_frame_rate": "30/1", "bit_rate": str(kbps * 1000)},
            {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000",
             "channels": 2},
        ],
        "format": {"format_name": "mov,mp4,m4a", "duration": "600.0",
                   "bit_rate": str(kbps * 1000)},
    }
    return checker.evaluate(info, [0.0, 2.0, 4.0], True, link_kbps=link_kbps)


def test_a_file_youtube_accepts_can_still_be_too_big_for_the_link():
    """The case that actually happened: 4.3 Mbps at 720p, a 430 kbps budget."""
    v = _verdict(720, 4300, link_kbps=430)
    assert v.action == "reencode", (
        "a 4.3 Mbps file was accepted for a link that can carry 0.43 Mbps")
    assert any("upload" in r for r in v.reasons), v.reasons


def test_the_same_file_is_fine_on_a_link_that_can_carry_it():
    """The ceiling must not punish a deployment with real upstream."""
    v = _verdict(720, 4300, link_kbps=20000)
    assert not any("upload" in r for r in v.reasons), v.reasons


def test_an_unmeasured_link_changes_nothing():
    """0 means unmeasured. An unmeasured deployment must never be throttled by
    a guess -- that would be worse than the bug."""
    v = _verdict(720, 4300, link_kbps=0)
    assert not any("upload" in r for r in v.reasons), v.reasons
    assert v.action in ("none", "remux"), v.action


def test_youtubes_own_ceiling_still_applies():
    """The link ceiling is an addition, not a replacement."""
    v = _verdict(720, 40000, link_kbps=0)
    assert v.action == "reencode"
    assert any("far above" in r for r in v.reasons), v.reasons


# ─────────── the wiring: a ceiling nobody passes changes nothing ───────────

def test_every_caller_passes_the_link_ceiling():
    """THE test. check() growing a parameter is worthless unless the call sites
    hand it one -- and there are exactly three."""
    from kaizer_live import api as live_api, encode as live_encode
    import live_integration

    for mod in (live_api, live_encode, live_integration):
        src = inspect.getsource(mod)
        tree = ast.parse(src)
        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "check"
                 and isinstance(n.func.value, ast.Name)
                 and n.func.value.id == "checker"]
        assert calls, f"{mod.__name__}: no checker.check() call found any more"
        for c in calls:
            kws = {k.arg for k in c.keywords}
            assert "link_kbps" in kws, (
                f"{mod.__name__} calls checker.check() without link_kbps -- the "
                f"ceiling is present in the code and reached by nothing")


def test_check_forwards_the_ceiling_to_evaluate():
    """check() accepting it and dropping it would be just as silent."""
    src = inspect.getsource(checker.check)
    assert "link_kbps=link_kbps" in src, (
        "check() takes link_kbps but does not pass it to evaluate()")


def test_the_repair_targets_the_link_not_just_the_setting():
    """Re-encoding to 3.0 Mbps for a 0.43 Mbps link produces a different file
    that starves exactly the same way."""
    from kaizer_live import encode as live_encode

    src = inspect.getsource(live_encode)
    assert "min(_target, _link / 1000.0)" in src, (
        "the repair does not clamp to what the link can carry")


# ─────────── the setting itself ───────────

def test_deliverable_kbps_is_zero_until_the_uplink_is_measured(monkeypatch):
    from kaizer_live import config

    monkeypatch.delenv("KAIZER_LIVE_UPLINK_MBPS", raising=False)
    s = config.Settings()
    assert s.deliverable_kbps == 0, "an unmeasured link must impose no ceiling"


def test_deliverable_kbps_keeps_headroom(monkeypatch):
    """A stream holds its rate for hours against everything else on the link,
    so the budget is a share of the measurement, not all of it."""
    from kaizer_live import config

    monkeypatch.setenv("KAIZER_LIVE_UPLINK_MBPS", "0.72")
    monkeypatch.setenv("KAIZER_LIVE_UPLINK_HEADROOM", "0.6")
    s = config.Settings()
    assert s.deliverable_kbps == 432, s.deliverable_kbps


def test_the_measured_link_refuses_the_file_that_actually_starved(monkeypatch):
    """End to end on the real numbers: 0.72 Mbps measured, the 4.3 Mbps file
    from broadcast 75-0."""
    from kaizer_live import config

    monkeypatch.setenv("KAIZER_LIVE_UPLINK_MBPS", "0.72")
    s = config.Settings()
    v = _verdict(720, 4300, link_kbps=s.deliverable_kbps)
    assert v.action == "reencode"
    assert any("upload" in r for r in v.reasons), v.reasons


# ─────────── audio must not eat a tight link ───────────

@pytest.mark.parametrize("mbps,want_audio,want_video", [
    (4.5,   "128k", "4372k"),    # generous link: unchanged
    (3.0,   "128k", "2872k"),    # unchanged
    (1.5,   "128k", "1372k"),    # the boundary, still unchanged
    (0.432, "64k",  "368k"),     # this machine's measured budget
])
def test_audio_scales_with_the_budget(mbps, want_audio, want_video):
    """At a flat 128 kbps, a 432 kbps link spends a third of itself on audio and
    leaves video 304. 64 kbps AAC is transparent for speech, which is what these
    broadcasts are, and video keeps the difference."""
    from kaizer_live import ffcmd

    cmd = ffcmd.normalize("ffmpeg", "in.mp4", "out.mp4", mbps=mbps)
    assert cmd[cmd.index("-b:a") + 1] == want_audio
    assert cmd[cmd.index("-b:v") + 1] == want_video


def test_a_generous_link_is_completely_unaffected():
    """Every deployment that is not bandwidth-bound must see no change at all."""
    from kaizer_live import ffcmd

    cmd = ffcmd.normalize("ffmpeg", "in.mp4", "out.mp4")     # the 4.5 default
    assert cmd[cmd.index("-b:a") + 1] == "128k"
    assert cmd[cmd.index("-b:v") + 1] == "4372k"
