"""The admin Live Map: honest states, real numbers, and no secrets.

This endpoint gathers everything about every live stream into one object, which
makes it the single worst place in the system to be careless with a stream key
or a presigned URL. test_the_map_never_leaks_a_secret is the one that matters
most here, and it is written against a service whose destinations are REAL
sealed values rather than placeholders, so it would actually catch a leak.

The second theme is honesty. A panel that shows green because a component is
running, while the relationship between two components is broken, teaches people
to stop looking at it. So the interesting assertions are the red ones: no
sweeper, no worker, an eviction policy that will silently drop a live video.
"""
from __future__ import annotations

import json
import os
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

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

import models  # noqa: E402
from database import Base  # noqa: E402
from kaizer_live.config import Settings  # noqa: E402
from kaizer_live.providers import ConnectedProvider  # noqa: E402
from kaizer_live.service import LiveService  # noqa: E402
from live_map import SECRET_FIELDS, _scrub, _worst, build_map  # noqa: E402

REDIS_URL = os.getenv("KAIZER_LIVE_REDIS", "redis://127.0.0.1:6380/0")
TEST_FERNET = "Jl2xRPQXiZ0wGxLZ6r3LJ0GjCq4sPeC1hZqmHkXwFhs="

# A destination shaped exactly like a real one: the ingest, and a key that looks
# like a YouTube key. If any of this reaches the browser the test fails.
REAL_KEY = "abcd-efgh-ijkl-mnop-qrst"
REAL_DST = f"rtmps://a.rtmps.youtube.com/live2/{REAL_KEY}"
PRESIGNED = "https://r2.example.com/ready/48-0.mp4?X-Amz-Signature=deadbeefcafe&X-Amz-Expires=604800"


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
    p = f"klmap{uuid.uuid4().hex[:8]}"
    yield p
    for key in r.scan_iter(f"{p}:*"):
        r.delete(key)


@pytest.fixture
def db():
    eng = create_engine("sqlite://", future=True, connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    s = sessionmaker(bind=eng, autoflush=False)()
    s.add(models.User(id=1, email="rita@test", name="Rita", is_active=True))
    s.add(models.User(id=2, email="sam@test", name="Sam", is_active=True))
    s.add(models.Channel(id=10, user_id=1, name="Cyber Sphere"))
    s.add(models.Channel(id=11, user_id=1, name="Telugu Today"))
    s.add(models.Channel(id=12, user_id=2, name="Sam's Channel"))
    s.commit()
    yield s
    s.close()


@pytest.fixture
def svc(r, prefix):
    s = Settings(prefix=prefix, redis_url=REDIS_URL, fernet_key=TEST_FERNET, daily_credit=10000)
    return LiveService(r, s, ConnectedProvider(
        lambda cid, meta: {"ingest_url": "rtmps://a.rtmps.youtube.com/live2",
                           "stream_key": REAL_KEY, "broadcast_id": "b" + cid, "watch_url": "https://y/" + cid},
        lambda cid, target: None,
        lambda cid, bid: "live"))


def _go_live(svc, r, vid, uid, channels, *, youtube="live", health="live", kbps=4500):
    """A live video with real sealed destinations, as the engine writes it."""
    r.hset(svc.k.video(vid), mapping={
        "video_id": vid, "user_id": str(uid), "worker": "w1", "state": "live",
        "source": PRESIGNED, "duration_s": 3600, "loop": "1",
        "started_at": time.time(), "finished_at": ""})
    r.sadd(svc.k.live_videos(), vid)
    for cid in channels:
        r.hset(svc.k.outputs(vid), str(cid), json.dumps({
            "channel_id": str(cid), "state": "on", "run": True, "gen": 1,
            "dst_enc": svc.vault.seal(REAL_DST),
            "dst_masked": "rtmps://a.rtmps.youtube.com/live2/abcd…",
            "broadcast_id": f"b{cid}", "watch_url": f"https://youtube.com/watch?v=b{cid}",
            "credit": 103, "youtube": youtube, "checks": 1,
            "meta": {"title": "x", "video_id": vid, "user_id": str(uid)},
        }))
        r.hset(svc.k.health(vid, str(cid)), mapping={
            "state": health, "bytes": 100000, "kbps": kbps, "reconnects": 0,
            "started": time.time(), "updated": time.time()})
    r.hset(svc.k.health_reader(vid), mapping={"state": "live", "pid": 123, "out_time_s": 42})


# ══ the one that matters most ════════════════════════════════════════

def test_the_map_never_leaks_a_secret(svc, r, db, prefix):
    """Everything about every live stream, in one object. So: nothing in it may
    be a stream key, a destination, or a presigned URL with credentials in its
    query string."""
    _go_live(svc, r, "48-0", 1, [10, 11])
    # ensure_ascii=False, or the masked destination's ellipsis comes back as
    # … and the "the mask survived" assertion below silently never matches.
    blob = json.dumps(build_map(svc, db), ensure_ascii=False)

    assert REAL_KEY not in blob, "a stream key reached the admin panel"
    assert REAL_DST not in blob, "a full destination URL reached the admin panel"
    assert "X-Amz-Signature" not in blob, "the presigned source URL leaked"
    assert PRESIGNED not in blob
    assert "dst_enc" not in blob, "the sealed destination was included"
    assert TEST_FERNET not in blob
    # The MASKED form is deliberately kept: recognising which key a channel is
    # using is genuinely useful when it misbehaves, and four characters grant
    # nothing.
    assert "abcd…" in blob


def test_scrub_removes_every_named_field_at_any_depth():
    """Applied to the whole payload rather than per call site, so a field added
    to a node next year is covered without anyone remembering to think."""
    payload = {"a": {"stream_key": "x", "keep": 1},
               "b": [{"source": "s", "ok": 2}, {"dst_enc": "e"}],
               "url": "gone"}
    out = _scrub(payload)
    assert out == {"a": {"keep": 1}, "b": [{"ok": 2}, {}]}
    assert "stream_key" in SECRET_FIELDS and "source" in SECRET_FIELDS


# ══ honest states ════════════════════════════════════════════════════

def test_a_missing_sweeper_paints_the_control_node_red(svc, db):
    """The quietest serious fault: broadcasts go live and then never end, and
    nothing errors. If the map showed this as fine it would be worse than not
    having the node."""
    m = build_map(svc, db)
    assert m["nodes"]["control"]["state"] == "bad"
    assert "No sweeper is running" in m["nodes"]["control"]["detail"]
    assert "kaizer_live.control" in m["nodes"]["control"]["detail"], \
        "the panel must say what to start, not only that something is wrong"


def test_no_worker_is_red_and_says_why(svc, db):
    m = build_map(svc, db)
    w = m["nodes"]["workers"]
    assert w["state"] == "bad"
    assert "nothing can go live" in w["detail"]


def test_a_node_is_as_bad_as_its_worst_part():
    """Showing "ok" because most channels are fine is how a panel teaches people
    to ignore it."""
    assert _worst("ok", "ok", "bad") == "bad"
    assert _worst("ok", "warn") == "warn"
    assert _worst("ok", "idle") == "idle"
    assert _worst("ok", "ok") == "ok"


def test_the_redis_node_objects_to_an_eviction_policy(svc, db, monkeypatch):
    """An evicted key is a channel that goes silently off air: the worker's next
    reconcile sees nothing wanted, stops the relay, and the broadcast is left
    open on YouTube with nothing feeding it."""
    real_info = svc.r.info

    def fake_info(*a, **k):
        d = dict(real_info(*a, **k))
        d["maxmemory_policy"] = "allkeys-lru"
        return d

    monkeypatch.setattr(svc.r, "info", fake_info)
    n = build_map(svc, db)["nodes"]["redis"]
    assert n["state"] == "bad"
    assert "noeviction" in n["detail"]


def test_the_live_deployment_passes_its_own_redis_check(svc, db):
    """The stack's docker-compose sets noeviction and AOF; if that ever drifts,
    this fails on the real instance rather than in a comment."""
    n = build_map(svc, db)["nodes"]["redis"]
    assert n["state"] in ("ok", "warn"), n["detail"]
    assert any(m["label"] == "policy" and m["value"] == "noeviction" for m in n["metrics"])


def test_a_channel_pushed_to_but_not_live_on_youtube_is_red(svc, r, db):
    """Two healthy components and a broken relationship: bytes are flowing and
    YouTube does not agree there is a broadcast. This is the exact failure a
    list of components cannot show."""
    _go_live(svc, r, "48-0", 1, [10], youtube="not_live")
    m = build_map(svc, r and db)
    assert m["nodes"]["youtube"]["state"] == "bad"
    assert m["channels"][0]["state"] == "bad"


# ══ the questions the founder asked ══════════════════════════════════

def test_it_answers_how_many_users_and_channels_are_live(svc, r, db):
    _go_live(svc, r, "48-0", 1, [10, 11])
    _go_live(svc, r, "49-0", 2, [12])
    u = build_map(svc, db)["usage"]
    assert u["users_live_now"] == 2
    assert u["videos_live_now"] == 2
    assert u["channels_live_now"] == 3
    names = {x["name"] for x in u["live_users"]}
    assert names == {"Rita", "Sam"}, "the panel should read as people, not user ids"


def test_it_answers_one_video_to_how_many_channels(svc, r, db):
    _go_live(svc, r, "48-0", 1, [10, 11])
    _go_live(svc, r, "49-0", 2, [12])
    u = build_map(svc, db)["usage"]
    assert u["widest_fanout"] == 2
    assert u["widest_fanout_video"] == "48-0"
    per = {v["video_id"]: v for v in u["per_video"]}
    assert per["48-0"]["channels_on"] == 2 and per["49-0"]["channels_on"] == 1
    assert per["48-0"]["confirmed_live"] == 2


def test_channel_rows_carry_the_measured_bitrate(svc, r, db):
    """Measured by the worker over a 5 s window, not derived from admin polls —
    two people with the panel open would otherwise consume each other's
    baseline and both read nonsense."""
    _go_live(svc, r, "48-0", 1, [10], kbps=4321.5)
    rows = build_map(svc, db)["channels"]
    assert rows[0]["kbps"] == pytest.approx(4321.5)
    assert rows[0]["channel"] == "Cyber Sphere"


def test_worst_first_so_the_broken_row_is_at_the_top(svc, r, db):
    _go_live(svc, r, "48-0", 1, [10])
    _go_live(svc, r, "49-0", 1, [11], youtube="not_live")
    rows = build_map(svc, db)["channels"]
    assert rows[0]["state"] == "bad", "a healthy row was listed above a broken one"


# ══ the wiring ═══════════════════════════════════════════════════════

def test_every_wire_joins_two_real_nodes(svc, db):
    """A wire to a node that does not exist draws nothing and hides a stage."""
    m = build_map(svc, db)
    for w in m["wires"]:
        assert w["from"] in m["nodes"], w
        assert w["to"] in m["nodes"], w
        assert w["state"] in ("ok", "warn", "bad", "idle"), w


def test_the_nine_stages_are_all_present(svc, db):
    """Uploads, checker, encode, Redis, control, workers, relays, YouTube,
    credit. A stage that quietly stops being reported is a stage nobody
    notices has broken."""
    assert set(build_map(svc, db)["nodes"]) == {
        "uploads", "checker", "encode", "redis", "control", "workers",
        "relays", "youtube", "credit"}


def test_credit_reports_the_engine_s_real_arithmetic(svc, db):
    n = build_map(svc, db)["nodes"]["credit"]
    got = {m["label"]: m["value"] for m in n["metrics"]}
    assert got["per broadcast"] == 153
    assert got["limit"] == 10000
    assert got["left today"] == 65


# ══ the endpoint ═════════════════════════════════════════════════════

def test_the_map_endpoint_is_admin_only():
    from main import app
    route = next((r for r in app.routes if getattr(r, "path", "") == "/api/admin/live/map"), None)
    assert route is not None, "the Live Map endpoint is not mounted"
    deps = str(getattr(route, "dependant", ""))
    import inspect
    from routers import admin_live_map as m
    src = inspect.getsource(m.live_map)
    assert "admin_required" in src, "the Live Map is not behind an admin check"


def test_the_endpoint_explains_a_disabled_engine_rather_than_404ing():
    """The tab exists and the admin is allowed to see it; the engine simply is
    not running here. 503 with the reason is what lets someone fix it."""
    import inspect
    from routers import admin_live_map as m
    src = inspect.getsource(m._service)
    assert "503" in src
    assert "KAIZER_LIVE_ENGINE=v2" in src and "KAIZER_LIVE_REDIS" in src
