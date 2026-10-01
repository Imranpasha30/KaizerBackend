"""Stop must stop the thing that is actually running.

FOUND BY THE OPERATOR, ON DEV. Pressing Stop in Live Studio turned the row to
"cancelled" and the broadcast kept streaming. Pressing Stop in the admin panel
worked. Two buttons, the same word, and only one of them stopped anything.

Two quite different things can own a live broadcast: the classic path (a thread
in the API process, signalled through an in-memory `threading.Event`) and the
engine (relays in a SEPARATE process, which can only be stopped by asking the
LiveService). `cancel_stream` did the first only, so for an engine broadcast the
signal went to a worker that was never registered and returned False.

It is worse than a button that does nothing:

  * the row is terminal, so nothing will ever retry ending the broadcast;
  * the engine still holds `channel_busy`, so that channel's NEXT broadcast is
    refused with "already live on another video" and no way to find which;
  * the broadcast stays open on YouTube holding that channel's reused key;
  * the 50 units reserved to close it are never released.

The tests below are about behaviour, not wording, because the wording is what
was already correct.
"""
from __future__ import annotations

import inspect
import sys
import time
import uuid

import pytest

redis = pytest.importorskip("redis")

# Which copy of the engine does THIS deployment use? Asked of the application
# rather than hard-coded: pinning a path would make these tests import DEV's
# engine when run on LIVE -- passing, and proving nothing about the code that
# machine actually runs.
import live_integration as _li  # noqa: E402

STACK = _li.resolve_stack_dir()
if STACK and STACK not in sys.path:
    sys.path.insert(0, STACK)

import live_integration as li  # noqa: E402
import routers.live_studio as ls  # noqa: E402


# ── the shape of the fix ─────────────────────────────────────────────

def test_cancel_asks_the_engine_to_stop():
    src = inspect.getsource(ls.cancel_stream)
    assert "stop_channel" in src, "the user's Stop still does not reach the engine"
    assert "get_live_service" in src


def test_the_engine_is_stopped_before_the_row_is_marked():
    """Ending a broadcast writes the row too, and skips one already marked
    terminal. Marking first is what made the UI say "cancelled" about a stream
    that was still going out."""
    src = inspect.getsource(ls.cancel_stream)
    assert src.index("stop_channel") < src.index('row.status = "canceled"')


def test_it_refuses_rather_than_reporting_a_stop_that_did_not_happen():
    """The one outcome worth failing a request over: answering "cancelled" while
    the video is still reaching the customer's audience."""
    src = inspect.getsource(ls.cancel_stream)
    assert "status_code=502" in src
    assert "still" in src and "streaming" in src
    # And the row must be left non-terminal so a retry and the sweeper can act.
    i = src.index("status_code=502")
    assert 'row.status = "canceled"' not in src[:i], \
        "the row was marked cancelled before the failure was known"


def test_a_broadcast_not_on_the_engine_still_cancels_normally():
    """A classic broadcast, or one that never started, raises LiveError from the
    engine. That is the ordinary case and must not become an error for the
    customer."""
    src = inspect.getsource(ls.cancel_stream)
    assert "except LiveError:" in src
    # The classic signal must still be sent.
    assert "live_orch.request_cancel(stream_id)" in src


def test_cancelling_during_a_repair_does_not_then_go_live():
    """The same defect one step earlier. Repairing a long video is minutes, and
    prepare_and_start holds the row on a background thread for all of them;
    without re-reading it, Stop marked the row cancelled and the broadcast
    started anyway -- spending quota on a broadcast nobody asked for."""
    src = inspect.getsource(li.prepare_and_start)
    assert "db.refresh(row)" in src
    i = src.index("db.refresh(row)")
    j = src.index("start_through_engine")
    assert i < j, "the row is re-read after the broadcast has already started"
    assert 'row.status in ("canceled", "failed", "done")' in src


# ── it actually stops a running channel ──────────────────────────────

def test_stop_channel_releases_the_channel_and_the_reserve():
    """What the user's Stop now reaches. Ending a channel must free three
    things, or the customer's NEXT broadcast on that channel is refused:
    the worker slot, the `channel_busy` claim, and the open-broadcast count
    that holds 50 units in reserve.
    """
    import json

    from kaizer_live.config import Settings
    from kaizer_live.providers import ConnectedProvider
    from kaizer_live.service import LiveService

    url = "redis://127.0.0.1:6380/0"
    try:
        r = redis.Redis.from_url(url, decode_responses=True)
        r.ping()
    except Exception as exc:
        pytest.skip(f"the live stack's Redis is not running: {exc}")

    pfx = f"klstop{uuid.uuid4().hex[:8]}"
    s = Settings(prefix=pfx, redis_url=url,
                 fernet_key="Jl2xRPQXiZ0wGxLZ6r3LJ0GjCq4sPeC1hZqmHkXwFhs=")
    ended = []
    svc = LiveService(r, s, ConnectedProvider(
        lambda cid, meta: {"ingest_url": "rtmps://x/live2", "stream_key": "k",
                           "broadcast_id": "b" + cid, "watch_url": ""},
        lambda cid, target: ended.append(cid),
        lambda cid, bid: "live"))
    try:
        # A worker with room, so go_live is admitted.
        r.hset(svc.k.worker("w1"), mapping={"id": "w1", "slots": 10, "updated": time.time()})
        r.expire(svc.k.worker("w1"), 60)

        from kaizer_live.service import ChannelRequest
        svc.go_live("48-0", "7", "/tmp/x.mp4", [ChannelRequest("10", "t")])
        assert r.get(svc.k.channel_busy("10")) == "48-0", "the channel was not claimed"
        assert svc.ledger.reserve() == 50, "no close credit was reserved"

        svc.stop_channel("48-0", "10", reason="canceled by user")

        assert ended == ["10"], "the YouTube broadcast was never ended"
        assert r.get(svc.k.channel_busy("10")) is None, \
            "the channel is still claimed: its next broadcast would be refused"
        assert svc.ledger.reserve() == 0, "the 50-unit close reserve was stranded"
        assert int(r.get(svc.k.worker_reserved("w1")) or 0) == 0, \
            "the worker slot was not given back"
        spec = json.loads(r.hget(svc.k.outputs("48-0"), "10"))
        assert spec["state"] == "ended"
    finally:
        for k in r.scan_iter(f"{pfx}:*"):
            r.delete(k)


def test_the_admin_stop_and_the_user_stop_reach_the_same_code():
    """They diverged once and only one of them worked. Both must end up at the
    service: the admin stops a whole video, the customer stops their own
    channel on it."""
    from routers import admin_live_map  # noqa: F401  (import proves it loads)
    import main
    paths = {getattr(r, "path", "") for r in main.app.routes}
    assert "/api/admin/live/videos/{vid}/stop" in paths
    assert "/api/live-studio/streams/{stream_id}/cancel" in paths
    # The engine's own per-channel stop, used by the Live Studio panel.
    assert "/api/live/videos/{vid}/channels/{cid}" in paths


# ── the same defect, in every cleanup path that fires on a restart ───
#
# The Stop button was the visible instance. An audit found four more, and two
# of them ran on EVERY BACKEND RESTART -- which is how the operator's broadcast
# came to be marked failed while it was still on air.
#
# They are worse than the Stop bug, because `cancel_stream` returns early on a
# terminal status: a row wrongly marked failed DISARMS the customer's own Stop
# button, leaving the admin panel as the only way out.

def test_the_startup_sweeper_leaves_engine_broadcasts_running():
    """Its premise -- "the backend restarted, so the ffmpeg is gone" -- is true
    of the classic path and FALSE of the engine, whose relays run in their own
    process precisely so a redeploy does not touch them. Surviving a restart is
    the designed behaviour, not an orphan."""
    import main
    src = inspect.getsource(main._live_studio_orphan_sweeper)
    assert "engine_is_carrying" in src, "the sweeper still buries without asking"
    assert "UPDATE live_streams" not in src, \
        "still a blanket SQL update: it cannot skip the engine's rows"
    # Built from two adjacent f-strings, so the sentence is never contiguous
    # in the source. Assert on the parts that carry the meaning.
    assert "live engine broadcast" in src and "kept" in src, \
        "what it spared must be reported, or nobody can tell it worked"


def test_the_r2_recovery_never_touches_an_engine_row():
    """Three separate defects lived here. Two marked the row failed; the third
    re-spawned the CLASSIC worker for a row the engine owned, which mints a
    SECOND YouTube broadcast on a channel that already has one live."""
    from live_studio import r2_backup
    src = inspect.getsource(r2_backup.recover_pending_streams)
    assert "engine_is_carrying" in src
    i = src.index("_carrying(row)")
    for marker in ('row.status = "failed"', 'row.status   = "starting"'):
        assert marker in src, marker
        assert i < src.index(marker), f"the engine guard runs after {marker!r}"
    assert "live_orch.kick_off" in src
    assert i < src.index("live_orch.kick_off"), \
        "an engine row could still be handed to the classic worker"


def test_the_false_auto_stop_comment_is_gone():
    """It claimed an abandoned broadcast "auto-stops after a few minutes of
    silence". enable_auto_stop is deliberately False -- loop micro-gaps made
    YouTube close broadcasts hours early -- so transition(complete) is the only
    thing that ever ends one. A comment that wrong is how the bug got written."""
    from live_studio import r2_backup
    src = inspect.getsource(r2_backup.recover_pending_streams)
    assert "auto-stops after a few minutes" not in src or "used to claim" in src


def test_a_second_click_during_a_repair_cannot_start_a_second_broadcast():
    """`preparing` is set for every engine start and a repair takes minutes --
    by far this endpoint's widest window. Without it in the guard, a second
    click spawned a second prepare thread, whose add_channel then raised 409
    and wrote the row failed while the first was live."""
    src = inspect.getsource(ls.start_stream)
    guard = src.split("already started")[0]
    assert '"preparing"' in guard
    assert '"canceled"' in guard, "a cancelled broadcast could be started again"


def test_a_start_error_is_checked_against_the_engine_before_failing():
    """start_through_engine is not atomic -- it tries add_channel and falls back
    to go_live -- so things can throw AFTER the engine has taken the channel."""
    src = inspect.getsource(li.prepare_and_start)
    assert "engine_owns_row(row)" in src
    # There are TWO `except Exception as exc:` blocks here -- the repair
    # failing and the start failing. Only the second asks the engine, so
    # select it by that rather than by position: [1] was the repair block,
    # which correctly writes `failed` and knows nothing about the engine.
    blocks = [b for b in src.split("except Exception as exc:") if "engine_owns_row" in b]
    assert len(blocks) == 1, f"expected one start-failure block, got {len(blocks)}"
    after = blocks[0]
    assert 'row.status = "streaming"' in after, \
        "a live channel is still written off as failed"
    assert after.index("engine_owns_row") < after.index('row.status = "failed"'), \
        "the row is written failed before the engine is asked"
    assert after.index('row.status = "streaming"') < after.index('row.status = "failed"')


@pytest.mark.parametrize("state,carried", [
    ("on", True), ("queued", True), ("stopping", True), ("end_pending", True),
    ("unknown", True),                 # could not ask -> leave it alone
    ("ended", False), ("failed", False), ("", False),
])
def test_what_counts_as_still_carried(state, carried, monkeypatch):
    """The safe direction is always to leave a possibly-live broadcast running
    rather than abandon a definitely-live one, so "unknown" counts as carried."""
    monkeypatch.setattr(li, "engine_owns_row", lambda _row: state)
    assert li.engine_is_carrying(object()) is carried


def test_revoking_credentials_under_a_live_broadcast_is_refused():
    """transition(complete) authenticates as the channel, and it is the ONLY
    thing that closes a broadcast. Revoke mid-broadcast and it can never be
    closed: the reused stream key is held for ever and the 50-unit close
    reserve comes back every day, with nothing in the UI to show it."""
    from routers import youtube_oauth
    src = inspect.getsource(youtube_oauth.disconnect)
    assert "live_broadcasts_on_channel" in src
    assert "409" in src
    assert src.index("live_broadcasts_on_channel") < src.index("oauth.revoke"), \
        "the token is revoked before anyone checks"


def test_deleting_a_channel_under_a_live_broadcast_is_refused():
    """It asked this about upload jobs and never about broadcasts. A foreign key
    happens to refuse the delete today, which is luck, not design."""
    from routers import channels
    src = inspect.getsource(channels.delete_channel)
    assert "live_broadcasts_on_channel" in src
    assert "Stop them in Live Studio first" in src


def test_live_broadcasts_on_channel_counts_only_unfinished_ones():
    """A finished broadcast must not block a disconnect for ever."""
    src = inspect.getsource(li.live_broadcasts_on_channel)
    assert '("done", "failed", "canceled")' in src
    assert "~" in src, "the filter must EXCLUDE terminal rows, not include them"
