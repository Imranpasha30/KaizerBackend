"""A broadcast that is still starting must not be drawn as broken.

The founder went live, saw the Live Map's YouTube node red -- "1 channel(s) are
being pushed to but YouTube does not report them live" -- and reported the
stream as failed. It had not failed. YouTube flipped it to `live` a few minutes
later and it ran healthily at 2965 Kbps.

`not_live` only ever meant "our one confirmation check has not come back live".
A channel one second old read identically to one that had been pushing into a
void for an hour, and the panel called both broken, in red.

A false alarm is the expensive kind of wrong: it teaches an operator that red
means nothing, so the next red -- a real one -- gets ignored too.
"""
import time

from live_map import CONFIRM_GRACE_S, _youtube_node


def _chan(youtube, joined_ago=5.0, health_state="live", **extra):
    c = {"state": "on", "youtube": youtube, "joined_at": time.time() - joined_ago,
         "health": {"state": health_state}}
    c.update(extra)
    return c


def _live(*channels):
    return [{"video_id": "1-0", "channels": list(channels)}]


def test_a_channel_seconds_into_startup_is_not_broken():
    """The exact case: pushing, YouTube has not flipped it live yet."""
    n = _youtube_node(_live(_chan("not_live", joined_ago=5)))
    assert n["state"] != "bad", f"a 5-second-old broadcast was called broken: {n}"
    assert "starting" in n["detail"].lower()


def test_ours_normally_confirm_well_inside_the_grace():
    """74-0 in production: channel_started 15:23:43, channel_confirmed 15:24:01.
    Eighteen seconds. The grace has to be comfortably wider than that."""
    assert CONFIRM_GRACE_S > 18 * 3, CONFIRM_GRACE_S


def test_a_channel_past_the_grace_IS_broken():
    """The fault this check exists for must still fire."""
    n = _youtube_node(_live(_chan("not_live", joined_ago=CONFIRM_GRACE_S + 30)))
    assert n["state"] == "bad", n
    assert "still does not" in n["detail"]


def test_a_confirmed_channel_is_ok():
    n = _youtube_node(_live(_chan("live")))
    assert n["state"] == "ok", n


def test_starving_is_still_bad_immediately():
    """Starvation is not a startup condition -- it is measured against a live
    source, so it never needs a grace period."""
    n = _youtube_node(_live(_chan("live", health_state="starved",
                                  joined_ago=5)))
    n2 = _youtube_node([{"video_id": "1-0", "channels": [
        {"state": "on", "youtube": "live", "joined_at": time.time() - 5,
         "health": {"state": "starved", "source_mbps": "4.4", "kbps": "600"}}]}])
    assert n2["state"] == "bad", n2
    assert "starved" in n2["detail"] or "cannot be fed" in n2["detail"]


def test_a_channel_with_no_joined_at_is_judged_not_excused():
    """Missing timing must not become a way to never report a fault."""
    c = {"state": "on", "youtube": "not_live", "health": {"state": "live"}}
    n = _youtube_node([{"video_id": "1-0", "channels": [c]}])
    assert n["state"] == "bad", "an unknown start time must not excuse a channel"


def test_starting_and_broken_are_counted_separately():
    """The operator needs to see which is which, not one merged number."""
    n = _youtube_node(_live(
        _chan("not_live", joined_ago=5),                       # starting
        _chan("not_live", joined_ago=CONFIRM_GRACE_S + 60),    # genuinely stuck
        _chan("live"),                                         # fine
    ))
    m = {x["label"]: x["value"] for x in n["metrics"]}
    assert m["starting"] == 1, m
    assert m["not live"] == 1, m
    assert m["confirmed live"] == 1, m
    assert n["state"] == "bad", "one genuinely stuck channel still makes it bad"
