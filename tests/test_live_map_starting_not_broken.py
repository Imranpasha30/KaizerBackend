"""A broadcast that is still starting must not be drawn as broken.

The founder went live, saw the YouTube node red -- "1 channel(s) are being
pushed to but YouTube does not report them live" -- and reported the stream as
failed. It had not failed. YouTube flipped it to live a few minutes later and it
ran healthily at 2965 Kbps.

Two separate defects produced that:

  * the map could not tell "not live YET" from "not live", and
  * the engine GAVE UP confirming after a few checks inside the first minute,
    recording the same value YouTube's own "complete" produces -- and never
    looked again, so the panel kept saying it for the rest of the run.

The engine now records four distinct things, and the map draws them differently:

    live          confirmed                                 -> ok
    pending       still inside the first checks             -> starting (amber)
    unconfirmed   we stopped asking; re-asked on a backoff  -> starting (amber)
    not_live      YouTube itself said complete/revoked      -> fault (red)

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


def _node(*channels):
    return _youtube_node([{"video_id": "1-0", "channels": list(channels)}])


def test_a_channel_seconds_into_startup_is_not_broken():
    """The exact case: pushing, YouTube has not flipped it live yet."""
    n = _node(_chan("pending", joined_ago=5))
    assert n["state"] != "bad", f"a 5-second-old broadcast was called broken: {n}"
    assert "not confirmed live yet" in n["detail"]


def test_a_channel_we_gave_up_on_is_not_broken_either():
    """`unconfirmed` means WE stopped asking. It says nothing about YouTube, and
    this is what kept 49-0 showing as broken while it was live and healthy."""
    n = _node(_chan("unconfirmed", joined_ago=CONFIRM_GRACE_S + 600))
    assert n["state"] != "bad", n
    assert "re-checking" in n["detail"] or "re-check" in n["detail"]


def test_ours_normally_confirm_well_inside_the_grace():
    """74-0 in production: channel_started 15:23:43, channel_confirmed 15:24:01.
    Eighteen seconds. The grace has to be comfortably wider than that."""
    assert CONFIRM_GRACE_S > 18 * 3, CONFIRM_GRACE_S


def test_a_pending_channel_past_the_grace_IS_broken():
    """Still 'pending' long after it started is a genuine fault."""
    n = _node(_chan("pending", joined_ago=CONFIRM_GRACE_S + 30))
    assert n["state"] == "bad", n
    assert "does not report them live" in n["detail"]


def test_youtubes_own_terminal_answer_is_always_a_fault():
    """`not_live` now only comes from complete/revoked, so it needs no grace."""
    n = _node(_chan("not_live", joined_ago=2))
    assert n["state"] == "bad", n


def test_the_fault_detail_names_the_audio_cause():
    """A silent file is the most common way a broadcast never goes live, and the
    operator should not have to remember that."""
    n = _node(_chan("not_live"))
    assert "audio" in n["detail"].lower(), n["detail"]


def test_a_confirmed_channel_is_ok():
    assert _node(_chan("live"))["state"] == "ok"


def test_starving_is_bad_immediately():
    """Starvation is measured against a live source, so it is never a startup
    condition and never gets a grace period."""
    n = _youtube_node([{"video_id": "1-0", "channels": [
        {"state": "on", "youtube": "live", "joined_at": time.time() - 5,
         "health": {"state": "starved", "source_mbps": "4.4", "kbps": "600"}}]}])
    assert n["state"] == "bad", n
    assert "cannot be fed" in n["detail"]


def test_a_pending_channel_with_no_joined_at_is_judged_not_excused():
    """Missing timing must not become a way to never report a fault."""
    n = _youtube_node([{"video_id": "1-0", "channels": [
        {"state": "on", "youtube": "pending", "health": {"state": "live"}}]}])
    assert n["state"] == "bad", "an unknown start time must not excuse a channel"


def test_starting_and_broken_are_counted_separately():
    """The operator needs to see which is which, not one merged number."""
    n = _node(
        _chan("pending", joined_ago=5),                      # starting
        _chan("unconfirmed", joined_ago=900),                # starting (re-checking)
        _chan("not_live", joined_ago=900),                   # genuinely over
        _chan("live"),                                       # fine
    )
    m = {x["label"]: x["value"] for x in n["metrics"]}
    assert m["starting"] == 2, m
    assert m["not live"] == 1, m
    assert m["confirmed live"] == 1, m
    assert n["state"] == "bad", "one genuinely dead channel still makes it bad"
