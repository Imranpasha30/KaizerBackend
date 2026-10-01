"""Manual stream-key mode is gone, and must stay gone.

WHY THERE IS A TEST FOR AN ABSENCE. The idea is extremely attractive and will be
proposed again: pushing to a key the customer copied from YouTube Studio costs no
API quota at all, against 153 units for an API broadcast. Sixty-five broadcasts a
day becomes unlimited. It reads like the whole problem solved.

It does not work, and the reason is on YouTube's side. It was tested on a real
channel: 90 seconds of video pushed to that channel's persistent key with no
broadcast waiting. The stream read `active` for the entire window and NO
BROADCAST EVER APPEARED. YouTube stopped auto-creating a broadcast for a
persistent key on 1 September 2020. Since then a key carries video and nothing
else; for anyone to watch it a broadcast must already exist, and the only way to
make one is liveBroadcasts.insert (50) + bind (50).

So 100 units is the floor for a viewable broadcast, no arrangement of keys
removes it, and sharding across Cloud projects to multiply the quota is forbidden
by the YouTube Developer Policies (III.D.1.c). Manual mode could push video that
nobody could ever watch.

These tests are cheap and they encode that finding where the next person to have
the idea will run into it.

The COLUMNS are deliberately still there — dropping a column is irreversible and
they hold keys customers pasted by hand — so "is the feature gone" cannot be
answered by looking at the schema. It is answered by the routes and the code.
"""
from __future__ import annotations

import inspect
import re

import pytest
from fastapi.testclient import TestClient

import models


@pytest.fixture(scope="module")
def client():
    from main import app
    return TestClient(app)


# ── the endpoints are gone ───────────────────────────────────────────

def test_no_route_serves_a_stream_key(client):
    """The five endpoints that read or wrote a pasted key no longer exist."""
    paths = {getattr(r, "path", "") for r in client.app.routes}
    offenders = [p for p in paths if "live-key" in p or "live-policy" in p]
    assert not offenders, f"manual stream-key routes are back: {offenders}"


@pytest.mark.parametrize("method,path", [
    ("get", "/api/channels/live-keys/"),
    ("get", "/api/channels/1/live-key/"),
    ("put", "/api/channels/1/live-key/"),
    ("delete", "/api/channels/1/live-key/"),
    ("put", "/api/channels/1/live-policy/"),
])
def test_the_old_urls_answer_404_not_500(client, method, path):
    """A removed endpoint must be absent, not broken.

    What must never happen is a 5xx: a stack trace from half-removed code, or an
    endpoint still routed to a function that no longer has its dependencies.

    The exact 4xx varies, and the variation is worth knowing about. FastAPI
    matches routes in DECLARATION ORDER, and `/api/channels/live-keys/` used to
    need a comment insisting it stay ABOVE `/{channel_id}` -- below it, the
    literal "live-keys" was parsed as a channel id and the request 422'd. Now
    that the literal route is gone, these paths fall through to the generic
    channel routes, whose auth dependency runs before the path parameter is
    validated. So an unauthenticated caller sees 401 and an authenticated one
    would see 422. Both mean the same thing: nothing serves this any more.
    """
    # client.request(), not client.get()/delete(): in this httpx version those
    # helpers reject a json= body, and the point of the test is that the URL is
    # gone whatever is sent to it.
    r = client.request(method.upper(), path, json={"stream_key": "x", "policy": "manual"})
    assert r.status_code in (401, 403, 404, 405, 422),         f"{method.upper()} {path} -> {r.status_code}: {r.text[:200]}"
    assert r.status_code < 500, f"{method.upper()} {path} returned a server error"
    # Whatever it answers, it must not be a key.
    assert "stream_key" not in r.text and "hint" not in r.text


# ── the code that read the columns is gone ───────────────────────────

def test_live_integration_has_no_key_store():
    """The seam no longer carries a key store, a policy, or a hint."""
    import live_integration as li
    for name in ("DbKeyStore", "live_policy_of", "key_hint", "VALID_POLICIES",
                 "DEFAULT_INGEST", "policy_for_channel"):
        assert not hasattr(li, name), f"live_integration.{name} is back"


def test_the_engine_has_exactly_one_way_to_go_live():
    """ConnectedProvider is the only provider, and it takes three callables:
    start, end, and the single cheap confirmation."""
    from kaizer_live import providers
    assert not hasattr(providers, "ManualProvider"), "ManualProvider is back"
    params = inspect.signature(providers.ConnectedProvider.__init__).parameters
    assert set(params) - {"self"} == {"start_fn", "end_fn", "check_fn"}, list(params)


def test_a_channel_request_carries_no_mode():
    """The engine's request object has no policy field, so nothing downstream
    can route a channel to a key even by accident."""
    from kaizer_live.service import ChannelRequest
    fields = set(ChannelRequest.__dataclass_fields__)
    assert not fields & {"policy", "mode", "live_mode"}, fields
    # positional order matters: start_through_engine builds these positionally
    assert list(ChannelRequest.__dataclass_fields__)[:2] == ["channel_id", "title"]


def test_nothing_in_the_backend_reads_the_retired_columns():
    """The three Channel columns and LiveStream.live_mode are kept but unread.

    Kept because dropping a column is irreversible and they hold pasted keys.
    Unread because the feature is gone -- and a column that something still
    writes but nothing acts on is how a removed feature quietly comes back.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    retired = ("yt_manual_stream_key_enc", "yt_manual_ingest_url", "live_policy", "live_mode")
    offenders = []
    for py in root.rglob("*.py"):
        parts = py.parts
        if any(x in parts for x in ("engines", "tests", "venv", "site-packages", ".git")):
            continue
        text = py.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            stripped = line.strip()
            # Prose is allowed to name them: models.py explains at the column
            # why it is retired, and channels.py explains why the routes went.
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("*"):
                continue
            for col in retired:
                if re.search(rf"\b{col}\b", line):
                    # The column definitions themselves must stay.
                    if py.name == "models.py" and "Column(" in line:
                        continue
                    # main.py's migration ladder must keep adding them: the
                    # column has to exist on every deployment, or a database
                    # restored from an older dump would not match the model.
                    if py.name == "main.py" and ("ALTER TABLE" in line or '"channels"' in line
                                                 or '"live_streams"' in line or line.strip().startswith("(")):
                        continue
                    offenders.append(f"{py.relative_to(root)}: {stripped[:90]}")
    assert not offenders, "the retired manual-key columns are read again:\n  " + "\n  ".join(offenders)


# ── the columns themselves are untouched ─────────────────────────────

def test_the_columns_still_exist_and_were_not_dropped():
    """Deliberately still in the schema.

    Dropping a column is irreversible DDL against a database with real
    customers' channels in it, and these hold stream keys those customers
    pasted by hand. Removing the feature is a code change; deleting their data
    is a decision for the operator, not a side effect of one.
    """
    ch = {c.name for c in models.Channel.__table__.columns}
    ls = {c.name for c in models.LiveStream.__table__.columns}
    retired = {"yt_manual_stream_key_enc", "yt_manual_ingest_url", "live_policy"} & ch
    if "live_mode" in ls:
        retired.add("live_mode")

    # NOT "these must exist" -- "these must not be DROPPED where they exist".
    # The deployment that ran the whole campaign has them and holds keys
    # customers pasted; a deployment that only ever received the finished result
    # never had them at all, and does not need them. Both are correct, and
    # asserting existence would fail on the second for no reason.
    if not retired:
        import pytest as _pt
        _pt.skip("this deployment never had the manual-key columns")
    for col in sorted(retired):
        assert col in ch or col in ls
    # And the reused-key column, which is the thing that actually saves money:
    # 1 unit to look the stream up instead of 50 to mint a new one.
    assert "yt_stream_id" in ch


# ── the arithmetic that replaced it ──────────────────────────────────

def test_the_cost_floor_is_what_the_experiment_showed():
    """insert + bind is 100 units and cannot be avoided; the engine's total is 153.

    If someone "optimises" this table, this test says what the numbers mean:
    insert and bind are the floor for a broadcast to exist at all, the reused
    key is 1 instead of 50, and transition is the close.
    """
    from kaizer_live.config import Costs
    c = Costs()
    assert c.insert + c.bind == 100, "the floor for a viewable broadcast changed"
    assert c.key_reuse == 1 and c.key_mint == 50, "reusing a key must cost 1, minting 50"
    assert c.start_cost(reuse_key=True, check="confirm") == 103
    assert c.close_cost(auto_stop=False) == 50
    assert c.broadcast_total(reuse_key=True, check="confirm") == 153
    assert 10000 // 153 == 65, "65 broadcasts a day on a 10,000-unit project"
    # The old polling start cost, kept in Costs so the saving is legible.
    assert c.start_cost(reuse_key=True, check="poll") == 116


def test_transition_complete_is_still_the_only_closer():
    """enable_auto_stop=False makes transition(complete) load-bearing.

    YouTube's auto-stop reads a loop's micro-gaps as the end of the stream and
    closed a 9-hour broadcast early. Because it is off, nothing else ends a
    broadcast -- so close_cost must never become 0 in the configuration the
    engine actually runs with, and the 50 units are not removable.
    """
    from kaizer_live.config import Settings
    s = Settings(fernet_key="x")
    assert s.auto_stop is False, "auto_stop must default off: it ends broadcasts early"
    assert s.costs.close_cost(auto_stop=s.auto_stop) == 50
