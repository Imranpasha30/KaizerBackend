"""The live stack's process separation, and the two bugs that had to be re-fixed.

The point of splitting live streaming into its own processes is that stopping
one must not stop a stream. These tests defend the mechanics of that, because
every one of them is invisible when it is wrong: the stream simply ends, or
never ends, and nothing logs a reason.

Real Redis, because all of this is Redis semantics -- an expiring lock, an
atomic decrement, a list hand-off. A fake would be testing the fake.
"""
from __future__ import annotations

import os
import subprocess
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

from kaizer_live import counters, encode  # noqa: E402
from kaizer_live.config import Settings  # noqa: E402
from kaizer_live.control import beat_key, lock_key, run_sweeper, sweeper_identity, sweeper_status  # noqa: E402
from kaizer_live.credit import CreditLedger  # noqa: E402
from kaizer_live.providers import ConnectedProvider, ProviderError  # noqa: E402
from kaizer_live.service import ChannelRequest, LiveService  # noqa: E402
from kaizer_live.worker import RelayProc, Worker, _detached  # noqa: E402

REDIS_URL = os.getenv("KAIZER_LIVE_REDIS", "redis://127.0.0.1:6380/0")
# A Fernet key for the tests only; nothing here reaches YouTube or a real key.
TEST_FERNET = "Jl2xRPQXiZ0wGxLZ6r3LJ0GjCq4sPeC1hZqmHkXwFhs="


@pytest.fixture
def r():
    try:
        c = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        c.ping()
    except Exception as exc:
        pytest.skip(f"the live stack's Redis is not running at {REDIS_URL}: {exc}")
    return c


@pytest.fixture
def prefix(r):
    p = f"kltest{uuid.uuid4().hex[:8]}"
    yield p
    for key in r.scan_iter(f"{p}:*"):
        r.delete(key)


@pytest.fixture
def settings(prefix):
    return Settings(prefix=prefix, redis_url=REDIS_URL, fernet_key=TEST_FERNET,
                    daily_credit=10000)


def _service(r, settings, *, start=None, end=None, check=None):
    return LiveService(r, settings, ConnectedProvider(
        start or (lambda cid, meta: {"ingest_url": "rtmps://x/live2", "stream_key": "k-" + cid,
                                     "broadcast_id": "b-" + cid, "watch_url": "https://y/" + cid}),
        end or (lambda cid, target: None),
        check or (lambda cid, bid: "live")))


# ══ 1 · the sweeper's identity ═══════════════════════════════════════

def test_two_sweepers_cannot_both_hold_the_lock(r, prefix):
    """The old identity was f"{time.time()}-{id(service)}".

    id() is a memory address. Two processes really can compute the same string,
    and the holder check is `r.get(lock) == me` -- so both would believe they
    held it. Two sweepers means every finished broadcast is ended twice: the
    50-unit close spent twice, and the second transition failing against an
    already-complete broadcast.
    """
    a, b = sweeper_identity(), sweeper_identity()
    assert a != b, "two identities collided"
    assert not a.replace("-", "").isdigit(), "an identity must not be a bare timestamp"

    lk = lock_key(prefix)
    assert r.set(lk, a, nx=True, px=5000) is True
    assert r.set(lk, b, nx=True, px=5000) is None, "the second sweeper took the lock"
    assert r.get(lk) == a
    assert r.get(lk) != b, "the second sweeper would believe it holds the lock"


def test_the_identity_names_the_host_and_pid():
    """So an operator reading "who is sweeping" in the admin panel can go and
    look at that process, rather than being shown a number."""
    me = sweeper_identity()
    import socket
    assert socket.gethostname() in me
    assert str(os.getpid()) in me


def test_a_clean_exit_hands_the_lock_over_at_once(r, settings, prefix):
    """Otherwise a redeploy leaves the queue unattended for a full lock TTL."""
    import threading
    svc = _service(r, settings)
    stop = threading.Event()
    me = sweeper_identity()
    t = threading.Thread(target=run_sweeper,
                         args=(svc,), kwargs={"interval": 0.2, "stop": stop, "identity": me},
                         daemon=True)
    t.start()
    # Wait for the BEAT, not the lock. run_sweeper takes the lock, then runs a
    # full sweep against Redis, and only then writes its first beat -- so
    # polling on the lock and asserting on the beat is a race that the rest of
    # the suite wins often enough to fail here consistently.
    for _ in range(100):
        if r.get(lock_key(prefix)) == me and sweeper_status(r, prefix)["alive"]:
            break
        time.sleep(0.05)
    assert r.get(lock_key(prefix)) == me, "the sweeper never took its lock"
    assert sweeper_status(r, prefix)["alive"] is True, "the sweeper never beat"
    stop.set()
    t.join(timeout=5)
    assert r.get(lock_key(prefix)) is None, "the lock was left behind for a whole TTL"


def test_a_missing_sweeper_is_reported_not_assumed(r, prefix):
    """The quietest serious fault in the system: nothing errors, broadcasts go
    live and simply never end, and the credit held back to end them is never
    released. It has to be visible or it is invisible."""
    st = sweeper_status(r, prefix)
    assert st["alive"] is False
    assert "not be confirmed or ended" in st["detail"]


# ══ 2 · the reserve must not be stranded ═════════════════════════════

def test_an_abandoned_end_releases_its_reserve(r, settings, prefix):
    """Every open broadcast has 50 units held back so it can always be ended.

    sweep() retries a failed end five times and then stops. Before this fix the
    row stayed at end_pending for ever and mark_closed was never reached, so the
    broadcast stayed COUNTED as open and its 50 units stayed reserved. Two
    hundred of those and the whole 10,000-unit day is reserved for broadcasts
    that no longer exist: nothing can go live, and nothing reports an error,
    because the reserve is doing exactly what it was written to do.
    """
    svc = _service(r, settings, end=lambda cid, t: (_ for _ in ()).throw(RuntimeError("YouTube said no")))
    ledger = svc.ledger

    # One open broadcast: 50 units reserved.
    ledger.mark_opened("7")
    assert ledger.reserve() == 50

    import json
    r.hset(svc.k.video("v1"), mapping={"video_id": "v1", "user_id": "7", "worker": "w1",
                                       "state": "live", "source": "x", "started_at": time.time()})
    r.sadd(svc.k.live_videos(), "v1")
    r.hset(svc.k.outputs("v1"), "9", json.dumps({
        "channel_id": "9", "state": "end_pending", "end_attempts": 5,
        "broadcast_id": "b-9", "error": "transition failed", "credit": 103}))

    out = svc.sweep()
    assert out.get("abandoned_ends") == 1, out

    spec = json.loads(r.hget(svc.k.outputs("v1"), "9"))
    assert spec["reserve_released"] is True
    assert "gave up" in spec["error"] and "YouTube Studio" in spec["error"], \
        "the operator has to be told where to finish it by hand"
    assert ledger.reserve() == 0, "50 units are stranded for a broadcast that no longer exists"


def test_a_retry_that_still_has_attempts_left_keeps_its_reserve(r, settings, prefix):
    """The release is for abandonment only. Releasing while retries remain would
    let a start be admitted that the reserve was holding room for."""
    import json
    svc = _service(r, settings, end=lambda cid, t: (_ for _ in ()).throw(RuntimeError("not yet")))
    svc.ledger.mark_opened("7")
    r.hset(svc.k.video("v2"), mapping={"video_id": "v2", "user_id": "7", "worker": "w1",
                                       "state": "live", "source": "x", "started_at": time.time()})
    r.sadd(svc.k.live_videos(), "v2")
    r.hset(svc.k.outputs("v2"), "9", json.dumps({
        "channel_id": "9", "state": "end_pending", "end_attempts": 1,
        "broadcast_id": "b-9", "credit": 103}))
    out = svc.sweep()
    assert out.get("retried_ends") == 1 and out.get("abandoned_ends") == 0
    assert svc.ledger.reserve() == 50, "the reserve was released while retries remained"


# ══ 3 · relays outlive the worker ════════════════════════════════════

def test_relays_are_spawned_detached():
    """A Ctrl-C in the worker's console, or a redeploy, must not reach them."""
    kw = _detached()
    if os.name == "nt":
        # DETACHED_PROCESS (no console, so no Ctrl-C) | CREATE_NEW_PROCESS_GROUP
        # (outside the worker's group, so a group signal misses it).
        assert kw["creationflags"] & 0x00000008
        assert kw["creationflags"] & 0x00000200
    else:
        assert kw["start_new_session"] is True


def test_shutdown_leaves_every_relay_running(r, settings, monkeypatch):
    """The whole reason for the split. Stopping the worker must not take a live
    stream off air."""
    w = Worker("t-worker", settings=settings, r=r)
    stopped = []
    fake = RelayProc("v1", instance="tok", addr="127.0.0.1:1", pid=999, adopted=True)
    monkeypatch.setattr(fake, "stop", lambda *a, **k: stopped.append("v1"))
    w.relays["v1"] = fake
    w.shutdown()
    assert stopped == [], "shutdown stopped a relay; a redeploy would end the broadcast"
    assert w.relays == {}, "the worker must stop supervising even though it leaves them running"


def test_adoption_refuses_a_relay_that_is_not_ours(monkeypatch):
    """A pid is a weak claim: pids get reused, and a relay recorded minutes ago
    may be a different process now. The relay echoes an instance token the
    worker generated, so adoption is exact."""
    seen = {}
    monkeypatch.setattr(RelayProc, "stats", lambda self: {"instance": "SOMEONE-ELSES"})
    monkeypatch.setattr(RelayProc, "retire", lambda self: seen.setdefault("retired", True))
    rp, note = RelayProc.adopt("v1", {"control_addr": "127.0.0.1:1", "instance": "OURS", "relay_pid": "5"})
    assert rp is None, "adopted a relay belonging to a different deployment"
    assert seen.get("retired") is True, "an unadoptable relay must be asked to shut down"
    assert "different relay" in note


def test_adoption_accepts_our_own_relay(monkeypatch):
    monkeypatch.setattr(RelayProc, "stats", lambda self: {"instance": "OURS", "outputs": {}})
    rp, note = RelayProc.adopt("v1", {"control_addr": "127.0.0.1:1", "instance": "OURS", "relay_pid": "5"})
    assert rp is not None and rp.adopted is True and rp.pid == 5
    assert "adopted" in note


def test_an_unreachable_relay_is_not_replaced_immediately(monkeypatch):
    """Two relays publishing to one stream key is worse than a gap. If the old
    one cannot be reached it may still be streaming, so the worker waits out its
    lease instead of starting a second."""
    monkeypatch.setattr(RelayProc, "stats", lambda self: None)
    rp, note = RelayProc.adopt("v1", {"control_addr": "127.0.0.1:1", "instance": "OURS", "relay_pid": "5"})
    assert rp is None
    assert "did not answer" in note


def test_an_adopted_relay_is_not_declared_dead_on_one_missed_poll():
    """A relay replaced too eagerly would be a second publisher on the same key."""
    rp = RelayProc("v1", instance="t", addr="127.0.0.1:1", pid=1, adopted=True)
    assert rp.alive() is True
    for _ in range(RelayProc.ADOPTED_DEAD_AFTER - 1):
        rp._stat_failures += 1
        assert rp.alive() is True, "declared dead after a single hiccup"
    rp._stat_failures += 1
    assert rp.alive() is False


def test_a_lost_adopted_relay_does_not_look_like_a_finished_video():
    """returncode 0 means "the video reached its duration", and the worker acts
    on it by ending every broadcast. An adopted relay we lost track of must
    never report that."""
    rp = RelayProc("v1", instance="t", addr="127.0.0.1:1", pid=1, adopted=True)
    rp._stat_failures = RelayProc.ADOPTED_DEAD_AFTER
    assert rp.returncode() == -1, "a lost relay reported a clean finish"


def test_the_relay_is_given_a_lease_and_a_token(settings):
    """Both halves of "survives a redeploy but cannot be orphaned for ever"."""
    # Built without __init__ deliberately: _relay_cmd needs only the settings,
    # and a real Worker would open a Redis connection this test does not want.
    w = Worker.__new__(Worker)
    w.s, w.id = settings, "t-worker"
    cmd = w._relay_cmd({"source": "/tmp/x.mp4", "duration_s": 0, "loop": "1",
                        "started_at": time.time()}, "TOK123")
    assert "--instance" in cmd and "TOK123" in cmd
    assert "--orphan-timeout" in cmd
    lease = float(cmd[cmd.index("--orphan-timeout") + 1].rstrip("s"))
    assert lease >= 60, "the lease must comfortably exceed a worker redeploy"
    assert "--listen" in cmd and cmd[cmd.index("--listen") + 1].startswith("127.0.0.1"), \
        "the control API carries stream keys and must never leave the box"


# ══ 4 · the encode queue ═════════════════════════════════════════════

def test_a_job_is_never_lost_between_popped_and_started(r, settings, prefix):
    """Taken onto the worker's own list atomically, so a crash in the hand-off
    leaves the job recoverable rather than leaving a customer waiting for ever."""
    encode.enqueue(r, prefix, "j1", source="x.mp4", action="reencode")
    taken = r.blmove(encode.queue_key(prefix), encode.working_key(prefix, "e1"), 1, "LEFT", "RIGHT")
    assert taken == "j1"
    assert r.lrange(encode.working_key(prefix, "e1"), 0, -1) == ["j1"]
    assert encode.queue_depth(r, prefix) == 0

    w = encode.EncodeWorker("e1", settings=settings, r=r)
    assert w.requeue_abandoned() == 1
    assert encode.queue_depth(r, prefix) == 1, "a job stranded by a crash vanished"


def test_the_encode_worker_runs_below_normal_priority():
    """A relay only copies bytes, but it must do so ON TIME or every viewer on
    every channel sees a stutter. An encode has no deadline. Where they share a
    box the scheduler must prefer the relay."""
    import inspect
    src = inspect.getsource(encode._lower_priority)
    if os.name == "nt":
        assert "BELOW_NORMAL_PRIORITY_CLASS" in src
    else:
        assert "os.nice" in src


def test_a_counter_typo_fails_loudly(r, prefix):
    """A miscounted metric that silently stays at zero is this codebase's
    recurring defect wearing a different hat."""
    counters.bump(r, prefix, "uploads_checked")
    assert counters.today(r, prefix)["uploads_checked"] == 1
    with pytest.raises(AssertionError):
        counters.bump(r, prefix, "uplods_checked")


def test_every_counter_the_map_reads_is_zero_filled(r, prefix):
    """A missing key and a real zero mean the same thing to a reader; a panel
    with gaps in it invites the wrong question."""
    t = counters.today(r, prefix)
    assert set(t) == set(counters.NAMES)
    assert all(v == 0 for v in t.values())


# ══ 5 · the engine never depends on the application ══════════════════

def test_the_engine_imports_nothing_from_kaizerbackend():
    """What makes the worker and the encode worker deployable on another
    machine. The dependency points one way: the application hands the engine
    three callables, and the engine knows nothing else about it."""
    import pathlib
    import re
    pkg = pathlib.Path(STACK) / "kaizer_live"
    host = {"models", "database", "auth", "crypto", "main", "routers", "live_integration", "config_app"}
    for f in pkg.glob("*.py"):
        text = f.read_text(encoding="utf-8")
        for mod in re.findall(r"^\s*(?:from|import)\s+([\w.]+)", text, re.M):
            assert mod.split(".")[0] not in host, f"{f.name} imports the application: {mod}"

# ══ 6 · the checker gate ═════════════════════════════════════════════

def test_a_single_keyframe_is_judged_by_the_looping_duration():
    """These files stream with -stream_loop -1, so the only keyframe comes round
    once per pass and a viewer joining mid-pass waits the WHOLE DURATION for a
    picture. That wait is the keyframe gap and answers to the same 4-second
    limit as any other.

    The rule used to read `len(keyframe_times) == 1 and duration_s > 8`, so an
    8.000-second file with one keyframe was passed as stream-ready -- an
    eight-second wait, twice the limit, called ready.
    """
    from kaizer_live.checker import evaluate
    probe = {"format": {"format_name": "mov,mp4", "duration": "8.0"},
             "streams": [{"codec_type": "video", "codec_name": "h264", "width": 1920,
                          "height": 1080, "pix_fmt": "yuv420p",
                          "avg_frame_rate": "30/1", "r_frame_rate": "30/1"},
                         {"codec_type": "audio", "codec_name": "aac",
                          "sample_rate": "48000", "channels": 2}]}
    v = evaluate(probe, [0.0], moov_first=True)
    assert not v.stream_ready, "an 8 s file with one keyframe was called stream-ready"
    assert v.action == "reencode"
    assert "one keyframe" in " ".join(v.reasons)

    # Short enough that the loop restarts inside the limit: genuinely fine.
    probe["format"]["duration"] = "3.0"
    assert evaluate(probe, [0.0], moov_first=True).stream_ready


def test_no_keyframe_at_all_is_refused():
    from kaizer_live.checker import evaluate
    probe = {"format": {"format_name": "mov,mp4", "duration": "60"},
             "streams": [{"codec_type": "video", "codec_name": "h264", "width": 1920,
                          "height": 1080, "pix_fmt": "yuv420p",
                          "avg_frame_rate": "30/1", "r_frame_rate": "30/1"}]}
    v = evaluate(probe, [], moov_first=True)
    assert not v.stream_ready and "no keyframe" in " ".join(v.reasons)


def test_widely_spaced_keyframes_are_still_refused():
    """The ordinary case: most encoders put keyframes on scene changes, not on a
    two-second grid."""
    from kaizer_live.checker import evaluate
    probe = {"format": {"format_name": "mov,mp4", "duration": "60"},
             "streams": [{"codec_type": "video", "codec_name": "h264", "width": 1920,
                          "height": 1080, "pix_fmt": "yuv420p",
                          "avg_frame_rate": "30/1", "r_frame_rate": "30/1"},
                         {"codec_type": "audio", "codec_name": "aac",
                          "sample_rate": "48000", "channels": 2}]}
    v = evaluate(probe, [0.0, 6.0, 12.0], moov_first=True)
    assert not v.stream_ready and v.action == "reencode"
    assert v.info["max_keyframe_gap_s"] == 6.0
    # Two seconds apart is what YouTube recommends and must pass.
    assert evaluate(probe, [0.0, 2.0, 4.0], moov_first=True).stream_ready


# ══ 7 · the ledger must charge what Google actually bills ════════════

def test_minting_a_stream_costs_49_more_than_reusing_one():
    """Measured on the first real production broadcast (stream 126, channel
    1904). The audit log recorded exactly four calls:

        liveBroadcasts.insert  50
        liveStreams.insert     50   <- minted: the channel had no stored stream
        liveBroadcasts.bind    50
        liveBroadcasts.list     1   <- the single confirmation
                              ---
                              151   billed by Google

    The ledger charged 103, because start_cost took Costs.start_cost's
    `reuse_key=True` default unconditionally -- charging for looking a stream up
    when one had to be created.

    It matters because the daily limit is enforced against the LEDGER. Under-count
    and the engine keeps admitting starts after Google's quota is gone, and the
    failure arrives as a 403 with the panel still showing credit remaining. Every
    channel mints exactly once, and there are about ninety of them.
    """
    from kaizer_live.config import Costs
    c = Costs()
    assert c.start_cost(reuse_key=False, check="confirm") == 152
    assert c.start_cost(reuse_key=True, check="confirm") == 103
    assert c.start_cost(reuse_key=False) - c.start_cost(reuse_key=True) == 49
    # 152 against the 151 Google billed: the unused second prepaid confirmation.
    # Erring high is the only safe direction here.
    assert c.start_cost(reuse_key=False, check="confirm") >= 50 + 50 + 50 + 1


def test_the_ledger_charges_the_mint_price_when_told_to(r, settings, prefix):
    from kaizer_live.credit import CreditLedger
    led = CreditLedger(r, settings)
    ok, why = led.admit("7", reuse_key=False)
    assert ok and led.used() == 152, f"charged {led.used()}, expected 152 — {why}"
    assert "minted" in why, "the reason should say why it cost more"

    ok, _ = led.admit("7", reuse_key=True)
    assert ok and led.used() == 152 + 103


def test_an_unknown_answer_is_charged_as_a_mint(r, settings, prefix):
    """The safe direction. Over-reserving delays one start; under-reserving
    spends quota the project does not have."""
    import inspect
    from kaizer_live.service import LiveService
    src = inspect.getsource(LiveService._attach)
    assert 'meta.get("reuse_key", False)' in src, \
        "a missing hint must read as 'mint', not as 'reuse'"


def test_the_application_supplies_the_fact_from_the_channel_row():
    """The engine has no database, so it cannot know whether a stream exists to
    reuse. The application reads Channel.yt_stream_id and passes it in."""
    import inspect

    import live_integration as _li
    src = inspect.getsource(_li.start_through_engine)
    assert 'yt_stream_id' in src
    assert '{"reuse_key": _reuse}' in src, "the fact never reaches the engine"


# ══ 8 · the upload failing to keep up must be SAID ═══════════════════

def test_an_output_that_cannot_keep_up_is_called_starved():
    """A real broadcast ran half an hour with YouTube reporting
    `videoIngestionStarved` -- "viewers will experience buffering" -- while
    every signal here read healthy: relay state live, zero reconnects, zero
    restarts, bytes advancing. The file was 3.6 Mbps and 577 Kbps was arriving.

    Nothing noticed because health asked "are bytes growing?", and a blocked
    socket still drains a trickle. Growing and KEEPING UP are different things,
    and only the second one means a watchable stream. It needs its own state
    because the remedy is different: encode smaller or get more upstream, not
    restart anything.
    """
    import inspect
    from kaizer_live.worker import Worker
    src = inspect.getsource(Worker._sync_outputs)
    assert '"starved"' in src, "an output that cannot keep up is still called live"
    assert "src_mbps" in src, "nothing measures what the source is producing"
    assert "STARVED_BELOW" in src and "STARVED_AFTER_S" in src
    # A burst must not be reported as a fault.
    assert Worker.STARVED_AFTER_S >= 10, "too eager: RTMP is bursty"
    assert 0.5 <= Worker.STARVED_BELOW < 1.0


def test_the_source_rate_is_published_so_the_number_means_something():
    """"340 kbps" alone says nothing. "340 kbps of a 3.6 Mbps video" is the
    whole diagnosis, and it is what tells an operator to re-encode."""
    import inspect
    from kaizer_live.worker import Worker
    assert '"source_mbps"' in inspect.getsource(Worker._sync_outputs)


def test_the_encode_bitrate_is_a_setting_not_a_constant():
    """It was hard-coded at 4.5 Mbps, which produced a file a third too big for
    the link that had to carry it. The ceiling is the UPLOAD, not the picture."""
    from kaizer_live.config import Settings
    s = Settings(fernet_key="x")
    assert hasattr(s, "encode_mbps")
    assert 1.0 <= s.encode_mbps <= 4.5
    import inspect
    import re
    from kaizer_live import encode
    src = inspect.getsource(encode.EncodeWorker.process)
    # The setting must reach the encoder. The expression moved into a variable
    # when the repair began clamping to what the LINK can carry, so assert the
    # intent -- settings in, no constant -- rather than one spelling of it.
    assert "self.s.encode_mbps" in src, "the encode bitrate no longer comes from settings"
    assert "mbps=" in src, "normalize() is called without an explicit bitrate"
    assert not re.search(r"mbps\s*=\s*\d", src), "a literal bitrate crept back in"
    # and it must not exceed what the link can deliver
    assert "deliverable_kbps" in src, (
        "the repair does not consider what the connection can upload")


def test_the_map_treats_starved_as_broken():
    """It reads healthy from every angle except the only one that matters."""
    from live_map import _BAD
    assert "starved" in _BAD


def test_sync_outputs_actually_runs(r, settings, prefix, monkeypatch):
    """EXECUTED, not read.

    The starvation check was inserted above the bitrate it compares against and
    raised UnboundLocalError on the worker's first reconcile -- in production.
    Every test around it read `inspect.getsource(...)` and passed, because the
    source contained all the right words in the wrong order.

    So this drives the real method with a stub relay and real Redis, twice, and
    asserts the health it writes. An ordering error cannot survive it.
    """
    import time as _t

    from kaizer_live.worker import Worker

    w = Worker.__new__(Worker)
    w.s, w.id, w.r, w.k = settings, "t", r, __import__(
        "kaizer_live.keys", fromlist=["K"]).K(prefix)
    w._init_trackers()          # the real one, so a new tracker cannot be missed
    w.vault = None

    class Relay:
        pid = 1234
        def __init__(self, sent, read):
            self.sent, self.read = sent, read
        def stats(self):
            return {"reader": {"state": "live", "pid": 99, "restarts": 0,
                               "out_time_s": 10, "in_bytes": self.read, "last_error": ""},
                    "outputs": {"10": {"state": "live", "gen": 1, "bytes": self.sent,
                                       "reconnects": 0, "dropped": 0, "last_error": "",
                                       "connected_at": _t.time()}}}
        def api(self, *a, **k):
            raise AssertionError("must not touch the relay's control API here")

    outs = {"10": {"run": True, "dst_enc": "x", "gen": 1}}

    # First pass establishes the baselines; nothing can be judged yet.
    w._sync_outputs("v1", Relay(0, 0), outs)
    h = r.hgetall(w.k.health("v1", "10"))
    assert h["state"] == "live", h
    assert "kbps" in h and "source_mbps" in h

    # Second pass, a window later: source produced 4 Mbps, only 0.4 Mbps sent.
    base = w.rate[("v1", "10")]
    w.rate[("v1", "10")] = (base[0], base[1] - 10, base[2])
    sb = w.src_rate["v1"]
    w.src_rate["v1"] = (sb[0], sb[1] - 10, sb[2])
    w.behind_since[("v1", "10")] = _t.time() - (Worker.STARVED_AFTER_S + 5)
    w._sync_outputs("v1", Relay(500_000, 5_000_000), outs)

    h = r.hgetall(w.k.health("v1", "10"))
    assert float(h["source_mbps"]) > 3, f"source rate not measured: {h}"
    assert h["state"] == "starved", (
        f"0.4 Mbps sent against a 4 Mbps source is not being reported as starved: {h}")

def test_nothing_getting_through_at_all_is_starved(r, settings, prefix):
    """The production case, and the one my first guard excluded.

        state: live   sending: 0.0 kbps   source needs: 4.42 Mbps

    `kbps > 0` was meant to avoid judging an output before its first window
    closed, and it also threw away the worst case. Nothing getting through is
    not "unmeasured" -- it is as starved as an output can be. `stalled` does not
    catch it either: that needs byte growth to STOP, and a blocked socket keeps
    draining 64 KB at a time.
    """
    import time as _t

    from kaizer_live.keys import K
    from kaizer_live.worker import Worker

    w = Worker.__new__(Worker)
    w.s, w.id, w.r, w.k = settings, "t", r, K(prefix)
    w._init_trackers()
    w.vault = None

    class Relay:
        pid = 1
        def __init__(self, sent, read): self.sent, self.read = sent, read
        def stats(self):
            return {"reader": {"state": "live", "pid": 9, "restarts": 0, "out_time_s": 10,
                               "in_bytes": self.read, "last_error": ""},
                    "outputs": {"10": {"state": "live", "gen": 1, "bytes": self.sent,
                                       "reconnects": 0, "dropped": 700, "last_error": "",
                                       "connected_at": _t.time()}}}
        def api(self, *a, **k): raise AssertionError("no control calls here")

    outs = {"10": {"run": True, "dst_enc": "x", "gen": 1}}
    w._sync_outputs("v9", Relay(1000, 0), outs)          # baselines

    # A window later: the source produced 5 MB, the output sent NOTHING.
    for d in (w.rate, w.src_rate):
        for kk, v in list(d.items()):
            d[kk] = (v[0], v[1] - 10, v[2])
    w._sync_outputs("v9", Relay(1000, 5_000_000), outs)  # closes the window at 0 kbps
    assert ("v9", "10") in w.rate_measured, "the window never closed"
    assert float(r.hgetall(w.k.health("v9", "10"))["kbps"]) == 0.0

    # Sustained, so it is not a burst.
    w.behind_since[("v9", "10")] = _t.time() - (Worker.STARVED_AFTER_S + 5)
    for d in (w.rate, w.src_rate):
        for kk, v in list(d.items()):
            d[kk] = (v[0], v[1] - 10, v[2])
    w._sync_outputs("v9", Relay(1000, 10_000_000), outs)

    h = r.hgetall(w.k.health("v9", "10"))
    assert h["state"] == "starved", f"0 kbps against a live source is not starved: {h}"
    assert float(h["source_mbps"]) > 3
