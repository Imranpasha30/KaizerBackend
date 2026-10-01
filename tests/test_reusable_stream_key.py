"""One persistent YouTube stream per channel, reused for every broadcast.

WHY THIS EXISTS. Every broadcast used to call liveStreams.insert (50 units) for
a throwaway key, and never delete it -- _safe_delete_stream runs only on the
bind-error path -- so each SUCCESSFUL broadcast abandoned a dead stream resource
on the customer's channel, for ever.

The root cause was one flag: `isReusable: False`. A non-reusable YouTube stream
cannot be bound to a second broadcast, so no amount of storing keys would have
helped until that flipped.

test_transition_complete_is_still_called is a REGRESSION GUARD, not a feature
test. Removing that call looks safe -- YouTube's docs say enableAutoStop makes it
redundant -- but orchestrator.py passes enable_auto_stop=False on purpose,
because looping a copy-stream makes micro-gaps at each loop boundary that
YouTube's auto-stop reads as the end of the broadcast (operator: "set 9h, closed
before time"). That transition is the ONLY closer for a Live Studio broadcast,
and with a reusable stream an un-closed broadcast blocks the channel's next one.
"""
import inspect
import pathlib

import pytest

import models
from youtube import rtmp_provider


# ── fakes ────────────────────────────────────────────────────────────

class _Call:
    def __init__(self, result): self._r = result
    def execute(self): return self._r


class _Broadcasts:
    def __init__(self, log): self.log = log
    def insert(self, **kw):
        self.log.append("liveBroadcasts.insert")
        return _Call({"id": "BC-1"})
    def bind(self, **kw):
        self.log.append("liveBroadcasts.bind")
        return _Call({"id": "BC-1"})


class _Streams:
    def __init__(self, log, existing): self.log, self.existing = log, existing
    def list(self, **kw):
        self.log.append("liveStreams.list")
        return _Call({"items": self.existing})
    def insert(self, **kw):
        self.log.append("liveStreams.insert")
        return _Call({
            "id": "ST-NEW",
            "cdn": {"ingestionInfo": {
                "ingestionAddress": "rtmp://x/live2",
                "rtmpsIngestionAddress": "rtmps://x/live2",
                "streamName": "new-key",
            }},
        })


class _YT:
    def __init__(self, log, existing):
        self._b, self._s = _Broadcasts(log), _Streams(log, existing)
    def liveBroadcasts(self): return self._b
    def liveStreams(self): return self._s


class _Job:
    id = 7; user_id = 1; clip_id = None; channel_id = 42; publish_kind = "live"


class _Channel:
    """Only what rtmp_provider reads off a channel."""
    def __init__(self, stream_id=None):
        self.id = 42
        self.yt_stream_id = stream_id
        self.oauth_token = None


SAVED_STREAM = [{
    "id": "ST-SAVED",
    "cdn": {"ingestionInfo": {
        "ingestionAddress": "rtmp://x/live2",
        "rtmpsIngestionAddress": "rtmps://x/live2",
        "streamName": "saved-key",
    }},
}]


@pytest.fixture
def provision(monkeypatch):
    """Return run(channel, existing) -> (result, call_log)."""
    def _run(channel, existing):
        log = []
        monkeypatch.setattr(rtmp_provider, "_yt", lambda creds: _YT(log, existing))
        monkeypatch.setattr(rtmp_provider, "_gcid_from_channel", lambda c: "UC-x")
        out = rtmp_provider.obtain_rtmp_target(
            creds=None, job=_Job(), channel=channel, title="t")
        return out, log
    return _run


# ── the flag that makes any of it possible ───────────────────────────

def test_streams_are_created_reusable():
    """A non-reusable stream cannot be bound to a second broadcast, so this
    flag is load-bearing -- with it False the whole feature is inert."""
    src = inspect.getsource(rtmp_provider)
    assert '"isReusable": True' in src
    assert '"isReusable": False' not in src


# ── reuse ────────────────────────────────────────────────────────────

def test_a_saved_stream_is_reused_and_nothing_is_minted(provision):
    out, log = provision(_Channel("ST-SAVED"), SAVED_STREAM)
    assert "liveStreams.list" in log
    assert "liveStreams.insert" not in log, "minted a stream despite having one"
    assert out["stream_id"] == "ST-SAVED"
    assert out["stream_key"] == "saved-key"


def test_reuse_costs_one_unit_instead_of_fifty(provision):
    """liveStreams.list is 1 unit; liveStreams.insert is 50."""
    _, reused = provision(_Channel("ST-SAVED"), SAVED_STREAM)
    _, minted = provision(_Channel(None), [])
    assert reused.count("liveStreams.list") == 1
    assert minted.count("liveStreams.insert") == 1


def test_rtmps_is_preferred_over_rtmp_on_the_reuse_path(provision):
    """The push should stay encrypted whichever path supplied the ingest."""
    out, _ = provision(_Channel("ST-SAVED"), SAVED_STREAM)
    assert out["ingest_url"].startswith("rtmps://")


# ── first time, and self-healing ─────────────────────────────────────

def test_a_channel_with_no_stream_mints_one(provision):
    out, log = provision(_Channel(None), [])
    assert "liveStreams.insert" in log
    assert "liveStreams.list" not in log, "looked up a stream that cannot exist"
    assert out["stream_id"] == "ST-NEW"


def test_a_stream_deleted_on_youtube_is_replaced(provision):
    """The customer can delete the stream in Studio. An empty list result must
    fall through to minting rather than failing the broadcast."""
    out, log = provision(_Channel("ST-GONE"), [])
    assert "liveStreams.list" in log and "liveStreams.insert" in log
    assert out["stream_id"] == "ST-NEW"


def test_a_failing_lookup_does_not_fail_the_broadcast(provision, monkeypatch):
    """A transient API error on the lookup must degrade to minting, not raise."""
    log = []

    class _Boom(_Streams):
        def list(self, **kw):
            log.append("liveStreams.list")
            raise RuntimeError("transient")

    class _YTBoom(_YT):
        def __init__(self): self._b, self._s = _Broadcasts(log), _Boom(log, [])

    monkeypatch.setattr(rtmp_provider, "_yt", lambda creds: _YTBoom())
    monkeypatch.setattr(rtmp_provider, "_gcid_from_channel", lambda c: "UC-x")
    out = rtmp_provider.obtain_rtmp_target(
        creds=None, job=_Job(), channel=_Channel("ST-SAVED"), title="t")
    assert out["stream_id"] == "ST-NEW"


# ── the deletion that would destroy the saved key ────────────────────

def test_cleanup_only_deletes_a_stream_this_call_created():
    """_safe_delete_stream on the bind-error path must be gated. Ungated it
    would delete the channel's persistent stream -- and could cut the ingest
    out from under a concurrent broadcast on the same channel."""
    src = inspect.getsource(rtmp_provider.obtain_rtmp_target)
    assert "if created_stream:" in src, "the stream delete is not gated"
    i = src.index("if created_stream:")
    assert "_safe_delete_stream" in src[i:i + 200]


# ── channels without the attribute at all ────────────────────────────

def test_a_channel_shim_without_the_field_still_works(provision):
    """rtmp_agent_v2 passes a _ChannelShim exposing only .oauth_token, so the
    lookup must use getattr with a default rather than attribute access."""
    class _Shim:
        oauth_token = None
    out, log = provision(_Shim(), [])
    assert out["stream_id"] == "ST-NEW"
    assert "liveStreams.list" not in log


def test_a_none_channel_still_works(provision):
    out, _ = provision(None, [])
    assert out["stream_id"] == "ST-NEW"


# ── REGRESSION GUARD: the call I nearly removed ──────────────────────

def test_transition_complete_is_still_called():
    """Do NOT remove this. YouTube's docs say enableAutoStop makes the
    transition redundant, which makes removal look safe -- but Live Studio
    passes enable_auto_stop=False because loop micro-gaps make auto-stop end
    the broadcast early. This transition is the only thing that closes a Live
    Studio broadcast, and an un-closed broadcast holds the channel's reusable
    stream and blocks the next broadcast."""
    src = inspect.getsource(rtmp_provider.finalize_broadcast)
    assert "transition(" in src and "complete" in src


def test_live_studio_still_disables_youtube_auto_stop():
    """The companion half of the guard above: if this ever becomes True, the
    transition really would be redundant -- and the loop micro-gap bug that
    enable_auto_stop=False fixed would come straight back."""
    src = pathlib.Path(
        rtmp_provider.__file__).parent.parent.joinpath(
        "live_studio", "orchestrator.py").read_text(encoding="utf-8")
    assert "enable_auto_stop=False" in src


# ── schema ───────────────────────────────────────────────────────────

def test_channel_has_somewhere_to_keep_the_stream():
    assert hasattr(models.Channel, "yt_stream_id")


def test_the_column_is_in_the_startup_migration_ladder():
    """create_all only creates missing TABLES -- it never ALTERs one. A column
    added to the model but not to _adds exists in Python and in NEITHER
    database, which is exactly how live_streams ended up missing
    apply_branding and 500ing every broadcast."""
    main_py = pathlib.Path(rtmp_provider.__file__).parent.parent / "main.py"
    src = main_py.read_text(encoding="utf-8")
    assert '"channels", "yt_stream_id"' in src
    assert "ALTER TABLE channels ADD COLUMN yt_stream_id" in src


def test_the_orchestrator_saves_the_id_back_to_the_channel():
    """Without this the channel never gets an id, the reuse branch never
    fires, and the whole feature is a silent no-op."""
    src = pathlib.Path(
        rtmp_provider.__file__).parent.parent.joinpath(
        "live_studio", "orchestrator.py").read_text(encoding="utf-8")
    assert "channel.yt_stream_id = _sid" in src


def test_the_connected_key_is_never_persisted():
    """Two kinds of key, two different rules — and this is the CONNECTED one.

    The reusable stream YouTube mints for us is identified by `yt_stream_id`
    alone; the key itself is fetched from liveStreams.list on every broadcast,
    so no standing credential sits at rest and a key regenerated in Studio is
    picked up automatically.

    The MANUAL key a customer pastes is a different thing: we cannot re-derive
    it from anything, so it must be stored — which is why it is Fernet
    ciphertext in a column whose name ends `_enc`. What must never exist is a
    PLAINTEXT stream-key column.
    """
    cols = {c.name for c in models.Channel.__table__.columns}
    assert "yt_stream_id" in cols, "the connected path needs somewhere for the id"

    key_cols = {c for c in cols if "stream_key" in c}
    # On a deployment that carried the manual-key experiment this is the one
    # column that ever held a stream key, and it is retired. On one that only
    # received the finished result there is none at all. What must be true of
    # BOTH is that no OTHER column holds one.
    assert key_cols <= {"yt_manual_stream_key_enc"}, (
        f"unexpected stream-key column(s) on channels: {sorted(key_cols)}")
    assert all(c.endswith("_enc") for c in key_cols), (
        "a stream key column that is not ciphertext")


def test_the_manual_key_column_is_text_not_a_bounded_string():
    """Fernet ciphertext runs roughly twice the plaintext length. A String(n)
    would truncate it silently and the key would fail to decrypt later, which
    reads as 'the customer pasted a bad key' rather than 'we corrupted it'."""
    # Only meaningful where the column exists. A deployment that carried the
    # manual-key experiment has it and it must stay Text -- a bounded String
    # would truncate a Fernet token into ciphertext that will not open. One
    # that only received the finished result never had it at all.
    if not hasattr(models.Channel, "yt_manual_stream_key_enc"):
        import pytest as _pt
        _pt.skip("this deployment never had the manual-key column")
    col = models.Channel.__table__.c.yt_manual_stream_key_enc
    assert col.type.__class__.__name__.upper() == "TEXT"
