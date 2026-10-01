"""A broadcast's delivery record must actually be written.

`_record_delivery` is wrapped in `except Exception: pass`, deliberately -- a
finished broadcast must never be recorded as failed because its statistics could
not be read. The cost of that is that EVERY mistake inside it is silent: a wrong
attribute, a missing function, a key built on the wrong prefix, and
`delivery_json` just stays NULL on every row forever while the feature looks
present.

The first version of this helper called `_engine()` and `target.video_id`,
neither of which exists, and `Target("", "", ...)` means the sweep would have
passed an empty video id even if it had. Nothing would have complained.

So these tests run the REAL function against REAL Redis keys and assert the
record lands. Reading the source would pass on code that cannot run.
"""
import json

import pytest

import live_integration


class _FakeRedis:
    """Just enough Redis: hgetall over a dict of hashes."""

    def __init__(self, hashes):
        self._h = hashes

    def hgetall(self, key):
        return dict(self._h.get(key, {}))


class _Keys:
    def __init__(self, prefix):
        self.p = prefix

    def health_reader(self, vid):
        return f"{self.p}:health:{vid}:reader"

    def health(self, vid, cid):
        return f"{self.p}:health:{vid}:{cid}"


class _Svc:
    def __init__(self, r, prefix):
        self.r, self.k = r, _Keys(prefix)


class _Row:
    def __init__(self, batch_id="75", slot=0):
        self.batch_id, self.video_slot = batch_id, slot
        self.delivery_json = None


def _reach_record_delivery():
    """Pull the REAL nested _record_delivery out of the provider's closure.

    make_connected_provider() returns ConnectedProvider(start_fn, end_fn,
    check_fn), and _record_delivery is a sibling in the same closure -- so
    end_fn's cells hold it. This is the only way to RUN the real function, and
    running it is the whole point: its `except Exception: pass` means a static
    read of the source passes happily on code that cannot execute.
    """
    prov = live_integration.make_connected_provider(lambda: None)
    end_fn = prov._end
    for cell in end_fn.__closure__ or ():
        try:
            val = cell.cell_contents
        except ValueError:
            continue
        if callable(val) and getattr(val, "__name__", "") == "_record_delivery":
            return val
    raise AssertionError("_record_delivery is not in the provider's closure")


def test_it_writes_the_record_for_a_healthy_broadcast(monkeypatch):
    """The real function, real key names, a delivery that kept up."""
    rec_fn = _reach_record_delivery()
    hashes = {
        "kl:health:75-0:reader": {"in_bytes": "1000000"},
        "kl:health:75-0:1904": {"bytes": "1000000", "dropped": "0",
                                "reconnects": "0", "kbps": "4200",
                                "source_mbps": "4.2"},
    }
    monkeypatch.setattr(live_integration, "get_live_service",
                        lambda: _Svc(_FakeRedis(hashes), "kl"))
    row = _Row("75", 0)
    rec_fn(row, "1904")
    assert row.delivery_json, (
        "nothing was written -- and the except:pass means the runtime would "
        "never have told us")
    rec = json.loads(row.delivery_json)
    assert rec["delivered_pct"] == 100.0
    assert rec["engine_video_id"] == "75-0"
    assert rec["source_bytes"] == 1_000_000


def test_it_writes_the_shortfall_for_a_starved_broadcast(monkeypatch):
    """The case the record exists for: the frames that never arrived."""
    rec_fn = _reach_record_delivery()
    hashes = {
        "kl:health:75-0:reader": {"in_bytes": "1000000"},
        "kl:health:75-0:1904": {"bytes": "350000", "dropped": "719",
                                "reconnects": "2", "kbps": "577",
                                "source_mbps": "4.42"},
    }
    monkeypatch.setattr(live_integration, "get_live_service",
                        lambda: _Svc(_FakeRedis(hashes), "kl"))
    row = _Row("75", 0)
    rec_fn(row, "1904")
    rec = json.loads(row.delivery_json)
    assert rec["delivered_pct"] == 35.0
    assert rec["dropped"] == 719
    assert rec["last_kbps"] == 577
    assert rec["source_mbps"] == 4.42


def test_a_missing_reading_leaves_null_rather_than_a_false_zero(monkeypatch):
    """No counters at all must not be recorded as a 0% delivery."""
    rec_fn = _reach_record_delivery()
    monkeypatch.setattr(live_integration, "get_live_service",
                        lambda: _Svc(_FakeRedis({}), "kl"))
    row = _Row("75", 0)
    rec_fn(row, "1904")
    assert row.delivery_json is None, (
        "with no counters the row must stay NULL; 0% is a different claim")


def test_it_reads_its_own_prefix_only(monkeypatch):
    """DEV and LIVE share a Redis. Reading the other deployment's counters and
    stamping them on this row would be worse than recording nothing."""
    rec_fn = _reach_record_delivery()
    hashes = {   # only the DEV prefix is populated
        "kldev:health:75-0:reader": {"in_bytes": "1000000"},
        "kldev:health:75-0:1904": {"bytes": "1000000"},
    }
    monkeypatch.setattr(live_integration, "get_live_service",
                        lambda: _Svc(_FakeRedis(hashes), "kl"))
    row = _Row("75", 0)
    rec_fn(row, "1904")
    assert row.delivery_json is None, "it read across the prefix boundary"


def test_a_broken_service_does_not_break_a_finished_broadcast(monkeypatch):
    """The reason for except:pass, kept as a test so the tradeoff is explicit."""
    rec_fn = _reach_record_delivery()

    def boom():
        raise RuntimeError("redis is down")

    monkeypatch.setattr(live_integration, "get_live_service", boom)
    row = _Row("75", 0)
    rec_fn(row, "1904")          # must not raise
    assert row.delivery_json is None


def test_the_helper_exists_and_is_called_where_the_row_is_finalized():
    """The record is written at the one moment both the counters and the row
    exist. If the call moves away from there, it writes nothing."""
    import inspect

    src = inspect.getsource(live_integration)
    fin = src.index('row.status = "done"')
    tail = src[fin:fin + 400]
    assert "_record_delivery(row, channel_id)" in tail, (
        "the delivery record is no longer taken where the row is finalized")


def test_every_name_the_helper_touches_exists():
    """The silent-by-design except means a missing name is invisible at runtime.

    The first version called _engine() and target.video_id -- neither existed,
    and nothing would ever have said so.
    """
    import ast
    import inspect

    src = inspect.getsource(live_integration)
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_record_delivery"), None)
    assert fn is not None, "_record_delivery is gone"

    # module-level names it relies on
    for name in ("engine_video_id", "get_live_service"):
        assert hasattr(live_integration, name), f"{name} is not in live_integration"

    # attributes it reads off the service, proven against the real class
    from kaizer_live.service import LiveService
    from kaizer_live.keys import K as _RealKeys
    import inspect as _i
    init = _i.getsource(LiveService.__init__)
    assert "self.r" in init and "self.k" in init, (
        "LiveService no longer exposes .r/.k; the helper reads both")
    for meth in ("health_reader", "health"):
        assert hasattr(_RealKeys, meth), f"K.{meth} is gone"

    # and it must NOT reach for the things that never existed
    called = {n.func.id for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "_engine" not in called, "_engine() does not exist anywhere"
    attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    assert "redis_and_keys" not in attrs, "redis_and_keys() does not exist anywhere"
    assert "video_id" not in attrs or "row" in {
        getattr(n.value, "id", "") for n in ast.walk(fn)
        if isinstance(n, ast.Attribute) and n.attr == "video_id"}, (
        "the video id must come from the row; the sweep's Target carries none")


def test_the_target_the_sweep_builds_carries_no_video_id():
    """Why the row is the source of the video id, kept as a test so a future
    change back to target.video_id fails here instead of silently recording
    nothing."""
    from kaizer_live.providers import Target

    t = Target("", "", "bid", "url")
    assert not hasattr(t, "video_id"), (
        "Target gained a video_id; if the sweep now fills it, this helper may "
        "use it -- but verify the SWEEP passes it, not just that the field exists")


def test_engine_video_id_matches_what_the_logs_show():
    """The record keys on the same id the founder reads in the logs ("75-0")."""
    assert live_integration.engine_video_id("75", 0) == "75-0"
    assert live_integration.engine_video_id("74", 0) == "74-0"


@pytest.mark.parametrize("produced,delivered,pct", [
    (1_000_000, 1_000_000, 100.0),
    (1_000_000, 350_000, 35.0),        # the starved shape
    (1_000_000, 0, 0.0),
])
def test_the_ratio_is_the_delivered_share(produced, delivered, pct):
    """The arithmetic the record exists to preserve."""
    assert round(delivered / produced * 100, 1) == pct


def test_no_source_reading_gives_null_not_zero():
    """0% would read as "nothing got through", which is a different claim from
    "we do not know"."""
    import ast
    import inspect

    src = inspect.getsource(live_integration)
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_record_delivery")
    body = ast.unparse(fn)
    assert "if produced > 0 else None" in body, (
        "with no source reading the ratio must be None, never 0")
