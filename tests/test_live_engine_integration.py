"""Where kaizer_live is joined to KaizerBackend.

The engine itself has its own suite (tests/kaizer_live, 27 tests against real
Redis, ffmpeg and MediaMTX). These cover the seam: the settings this
application hands it, the audit column its API calls are written to, the
duration cap the presigned source imposes, and the rule that the engine must
never be able to take the API down.

The load-bearing one is test_dev_cannot_share_live_redis_keys. DEV and LIVE run
against the same Redis server. If they shared a key prefix, a DEV sweep would
end a LIVE customer's broadcast — so the prefix is derived from the database
name when the environment variable is absent, and forgetting it on DEV cannot
reach LIVE.
"""
from __future__ import annotations

import os

import pytest

import live_integration as li

STACK = r"E:\kaizer-dev\kaizer-live-stack"
import models


# ── the prefix that keeps DEV away from LIVE ─────────────────────────

def test_dev_cannot_share_live_redis_keys():
    """Forgetting the variable on DEV must not reach LIVE.

    Tested on the rule itself rather than through the environment. Importing the
    application's config runs load_dotenv(), which puts .env's values BACK into
    os.environ -- so monkeypatch.delenv cannot express "pretend this is unset",
    and a test written that way passes or fails according to which module
    happened to be imported first.
    """
    assert li.derive_prefix(None, "postgresql://u:p@localhost/kaizer_dev") == "kldev"
    assert li.derive_prefix("", "postgresql://u:p@localhost/kaizer_dev") == "kldev"
    assert li.derive_prefix(None, "postgres://x/KAIZER_DEV") == "kldev", "must be case-insensitive"
    assert li.derive_prefix(None, "postgresql://u:p@localhost/kaizer") == "kl"
    # The dangerous default is the other way round: an unknown database must not
    # silently answer "kl" because someone typo'd the dev name. It does, and
    # that is the residual risk the explicit variable exists to remove.
    assert li.derive_prefix(None, None) == "kl"


def test_an_explicit_prefix_wins():
    assert li.derive_prefix("kltest", "postgresql://u:p@localhost/kaizer_dev") == "kltest"
    assert li.derive_prefix("  kltest  ", None) == "kltest"


def test_the_prefix_is_stable_across_calls(monkeypatch):
    """build_settings() must give the same answer every time it is called.

    It used to not: the prefix was read before the application's config was
    imported, and that import's load_dotenv() then repopulated the environment,
    so the first call and the second disagreed. A prefix that changes between
    calls is a DEV process that can address LIVE keys, once, unreproducibly.
    """
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/kaizer_dev")
    seen = {li.build_settings().prefix for _ in range(3)}
    assert len(seen) == 1, f"build_settings() returned {seen} across three calls"


def test_settings_take_the_apps_own_secrets(monkeypatch):
    monkeypatch.delenv("KAIZER_LIVE_FERNET_KEY", raising=False)
    s = li.build_settings()
    # Destinations are encrypted in Redis with the key this deployment already
    # manages; a second secret would be a second thing to lose.
    assert s.fernet_key, "no Fernet key resolved from the application config"
    assert s.daily_credit == 10000


# ── the engine may never take the API down ───────────────────────────

def test_the_engine_is_off_unless_asked_for(monkeypatch):
    monkeypatch.delenv("KAIZER_LIVE_ENGINE", raising=False)
    li._SERVICE = None
    assert li.get_live_service() is None


def test_a_broken_engine_returns_none_rather_than_raising(monkeypatch):
    """Redis unreachable, ffmpeg missing, no Fernet key — all of it must end
    as None and a printed reason, never an exception that stops the app from
    starting and takes every existing customer offline with it."""
    monkeypatch.setenv("KAIZER_LIVE_ENGINE", "v2")
    monkeypatch.setenv("KAIZER_LIVE_REDIS", "redis://127.0.0.1:6/0")  # nothing there
    li._SERVICE = None
    assert li.get_live_service() is None


# ── the audit column that was silently dropping rows ─────────────────

def test_a_live_stream_is_audited_against_live_streams():
    """youtube_api_calls.upload_job_id is an FK to upload_jobs. Live Studio was
    writing LiveStream ids into it; once those outgrew max(upload_jobs.id)
    every insert violated the constraint and log_youtube_call swallowed it, so
    the quota dashboard lost every live row from 2026-06-30."""
    from youtube import rtmp_provider as rp
    ls = models.LiveStream(); ls.id = 500
    assert rp._job_ref(ls) == {"live_stream_id": 500}


def test_an_upload_job_still_goes_where_it_always_did():
    from youtube import rtmp_provider as rp
    uj = models.UploadJob(); uj.id = 7
    assert rp._job_ref(uj) == {"upload_job_id": 7}


def test_a_job_with_no_id_is_written_to_neither():
    from youtube import rtmp_provider as rp
    assert rp._job_ref(object()) == {}


def test_the_audit_table_has_somewhere_to_put_it():
    cols = {c.name for c in models.YouTubeApiCall.__table__.columns}
    assert "live_stream_id" in cols


def test_the_audit_column_is_in_the_migration_ladder():
    import pathlib
    src = (pathlib.Path(li.__file__).parent / "main.py").read_text(encoding="utf-8")
    assert '"youtube_api_calls", "live_stream_id"' in src


# ── the presigned source cannot outlive its signature ────────────────

def test_a_broadcast_cannot_outlive_its_signed_url():
    """SigV4 caps a presigned URL at seven days, and the reader re-opens its
    source on every loop pass. A longer broadcast would simply stop, with the
    worker holding a URL it cannot renew."""
    hours, note = li.clamp_live_hours(24 * 30)
    assert hours == li.MAX_LIVE_HOURS
    assert "7 days" in note, "the cap must explain itself"
    assert li.MAX_LIVE_HOURS * 3600 < li.PRESIGN_TTL_S, "no margin before expiry"


@pytest.mark.parametrize("given,expected", [(1, 1.0), (24, 24.0), (144, 144.0)])
def test_ordinary_durations_pass_through(given, expected):
    assert li.clamp_live_hours(given) == (expected, "")


@pytest.mark.parametrize("bad", [0, -5, None, "", "abc"])
def test_a_missing_duration_becomes_one_hour(bad):
    assert li.clamp_live_hours(bad)[0] == 1.0


# ── there is one way to go live ──────────────────────────────────────

def test_the_provider_supplies_all_three_callables():
    """start, end, and check. The third is the one that used to be missing.

    Without check_fn the engine's confirmation silently does nothing:
    ConnectedProvider.check returns None for ever, every broadcast sits at
    youtube="unchecked", and "confirmed live" is a column that can never fill
    in. It is also what replaced polling the start path every two seconds for
    thirty seconds -- 15 units a broadcast down to 1, with 2 prepaid.
    """
    import inspect
    src = inspect.getsource(li.make_connected_provider)
    assert "def start_fn" in src and "def end_fn" in src and "def check_fn" in src
    assert "ConnectedProvider(start_fn, end_fn, check_fn)" in src


def test_every_youtube_call_from_the_live_path_has_a_timeout():
    """A hung Google connection would otherwise hold an HTTP request open for
    the transport's 120-second default, with the start credit already spent and
    the channel's slot already claimed. 120 s is right for the uploader, where
    minutes are legitimate, so the live path passes its own."""
    import inspect
    assert li._yt_timeout() <= 15, "the live path's YouTube timeout is too long to be useful"
    src = inspect.getsource(li.make_connected_provider)
    assert src.count("timeout_s=_yt_timeout()") >= 3, \
        "start, end and check must each pass the short timeout"


def test_the_connected_provider_keeps_auto_stop_off():
    """Loop micro-gaps make YouTube's auto-stop end a broadcast early — an
    operator reported a 9-hour stream closing before time. Because auto-stop
    is off, transition(complete) is the only thing that ends a broadcast, and
    with a reused key an un-closed broadcast blocks the channel's next one."""
    import inspect
    src = inspect.getsource(li.make_connected_provider)
    assert "enable_auto_stop=False" in src
    assert "enable_auto_stop=True" not in src


def test_the_provider_persists_the_reusable_stream():
    import inspect
    src = inspect.getsource(li.make_connected_provider)
    assert "ch.yt_stream_id = sid" in src, "the reused key is never saved"


# ── nothing unchecked reaches the live path ──────────────────────────

def test_a_file_that_is_not_stream_ready_is_refused(monkeypatch):
    """Starting a broadcast spends quota BEFORE any video moves, so a file
    YouTube will reject costs real units and puts a broken stream on a
    customer's channel. It is a hard stop, not a warning."""
    monkeypatch.setattr(li, "check_source",
                        lambda src: {"stream_ready": False, "action": "reencode",
                                     "reasons": ["video is VP9, YouTube Live needs H.264"]})
    with pytest.raises(li.NotStreamReady) as e:
        li.assert_stream_ready("whatever.webm")
    assert "reencode" in str(e.value) and "VP9" in str(e.value)


def test_a_stream_ready_file_passes(monkeypatch):
    monkeypatch.setattr(li, "check_source",
                        lambda src: {"stream_ready": True, "action": "none", "reasons": []})
    assert li.assert_stream_ready("fine.mp4")["stream_ready"] is True


def test_a_repair_happens_in_the_encode_queue_and_nowhere_else():
    """A re-encode is minutes at full tilt, so WHERE it runs matters.

    In the API process it competes with every request that process serves. Beside
    the relays it competes with streams -- which need very little CPU but need it
    ON TIME, or every viewer on every channel sees a stutter. So the repair is a
    queued job for a separate process that can be put on a separate machine.

    There must also be exactly ONE way to do it. There were two: this path, and
    an ensure_stream_ready() that downloaded, re-encoded and re-uploaded an R2
    object in whatever process called it -- and which nothing called. Two
    implementations of "how to repair a video for live" drift, and the drift is
    discovered on air.
    """
    import inspect
    assert not hasattr(li, "ensure_stream_ready"), "the second repair path is back"
    assert not hasattr(li, "normalize_for_live"), "the second repair path is back"

    src = inspect.getsource(li.prepare_source)
    assert "_encode.enqueue" in src, "the repair is not queued"
    # Asked of the CODE, not of the text. A substring check here matched this
    # function's own comment explaining that ffmpeg does not run in it -- the
    # same false positive this codebase keeps producing whenever a guard greps
    # for a word that the explanation of the rule inevitably contains.
    import ast as _ast
    import textwrap
    _tree = _ast.parse(textwrap.dedent(src))
    _spawns = [n.func.attr for n in _ast.walk(_tree)
               if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Attribute)
               and n.func.attr in ("run", "Popen", "call", "check_call", "check_output", "system")]
    assert not _spawns, f"prepare_source starts a process again: {_spawns}"
    # It waits on another process, so it must not be called from a request.
    assert "background thread" in inspect.getdoc(li.prepare_source)

    # And the queue must hold no credentials: its job carries presigned URLs.
    from kaizer_live import encode
    up = inspect.getsource(encode.EncodeWorker._upload)
    assert "presigned" in up.lower()
    assert "boto" not in up and "get_storage_provider" not in up


# ── the live path is gated, not merely gateable ──────────────────────

def test_go_live_is_wrapped_with_the_stream_ready_check(monkeypatch):
    """assert_stream_ready existing is not the same as it being CALLED.

    Requires Redis and ffmpeg, so it skips where they are absent rather than
    failing — but where they exist it proves the engine's go_live really is
    the wrapped one, not the vendored method.
    """
    monkeypatch.setenv("KAIZER_LIVE_ENGINE", "v2")
    li._SERVICE = None
    svc = li.get_live_service()
    if svc is None:
        pytest.skip("live engine could not start here (Redis/ffmpeg absent)")

    assert svc.go_live.__name__ == "_checked_go_live", "go_live is not gated"

    monkeypatch.setattr(li, "check_source",
                        lambda src: {"stream_ready": False, "action": "reencode",
                                     "reasons": ["video is VP9"]})
    with pytest.raises(li.NotStreamReady):
        svc.go_live("v1", "u1", "bad.webm", [])
    li._SERVICE = None


# ── the worker is a separate process, on a machine of its own ────────

def test_the_worker_needs_nothing_from_this_application():
    """What makes it deployable elsewhere.

    It used to be started through a live_worker.py wrapper that called the
    application's build_settings(), because `python -m kaizer_live.worker` read
    Settings straight from the environment and refused to start without
    KAIZER_LIVE_FERNET_KEY. That wrapper imported KaizerBackend, so the worker
    could only ever run on a machine with the whole application on it -- which
    is precisely what the split exists to stop being true.
    """
    import pathlib
    import re
    pkg = pathlib.Path(STACK) / "kaizer_live"
    host = {"models", "database", "auth", "crypto", "main", "routers", "live_integration"}
    for f in ("worker.py", "encode.py", "config.py"):
        text = (pkg / f).read_text(encoding="utf-8")
        for mod in re.findall(r"^\s*(?:from|import)\s+([\w.]+)", text, re.M):
            assert mod.split(".")[0] not in host, f"{f} imports the application: {mod}"


def test_the_worker_resolves_the_same_key_the_api_seals_with():
    """The API seals each stream destination with the application's Fernet key
    and the worker opens it. Two different keys means a worker that comes up
    healthy, reconciles happily, and fails on EVERY output -- so the engine's
    KAIZER_LIVE_FERNET_KEY falls back to the application's KAIZER_ENCRYPTION_KEY
    rather than being a second secret to keep in step."""
    import inspect
    from kaizer_live.config import load_env_file
    src = inspect.getsource(load_env_file)
    assert "KAIZER_ENCRYPTION_KEY" in src and "KAIZER_LIVE_FERNET_KEY" in src
    # An explicit engine key must still win: a deployment may want them separate.
    assert 'if not (os.getenv("KAIZER_LIVE_FERNET_KEY") or "").strip():' in src


def test_env_files_are_followed_and_never_override_the_shell():
    """On this box the stack's .env holds nothing but a pointer at the
    application's, so the key exists in ONE place. The first version of the
    loader read that file, set the pointer, and returned without following it --
    so the worker started with no key at all."""
    import inspect
    from kaizer_live.config import load_env_file
    src = inspect.getsource(load_env_file)
    assert "KAIZER_LIVE_ENV_FILE" in src
    assert "if k and k not in os.environ:" in src, \
        "a file must never override a value already in the environment"
    assert "for _ in range(5)" in src, "two files pointing at each other must not spin"


def test_the_worker_refuses_to_start_without_a_runnable_relay():
    """Spawning the relay is the worker's entire job. Discovering the binary is
    missing at the first broadcast is discovering it in front of an audience."""
    import inspect
    from kaizer_live import worker as w
    src = inspect.getsource(w.main)
    assert "is not a runnable file" in src
    assert "go build" in src, "the message must say how to produce it"


def test_the_vault_refuses_to_run_without_a_key():
    """It encrypts stream destinations at rest. Starting without a key would
    mean writing customers' stream keys into Redis in clear."""
    from kaizer_live.vault import Vault
    with pytest.raises(RuntimeError) as exc:
        Vault("")
    assert "KAIZER_LIVE_FERNET_KEY" in str(exc.value)
    assert "must not be stored in clear" in str(exc.value)


def test_the_procfile_starts_the_three_live_processes():
    """One per failure domain: the sweeper (calls YouTube), the worker (moves
    bytes) and the encode queue (heavy, no deadline, no credentials)."""
    import pathlib
    proc = (pathlib.Path(li.__file__).parent / "Procfile").read_text(encoding="utf-8")
    assert "python -m kaizer_live.control" in proc
    assert "python -m kaizer_live.worker" in proc
    assert "python -m kaizer_live.encode" in proc
    assert "python -m live_worker" not in proc, \
        "the wrapper is retired: it imported the application, which pinned the worker to this box"


# ── no mode reaches the engine any more ──────────────────────────────

def test_start_through_engine_builds_a_request_with_no_mode():
    """Every channel is an API broadcast on its reused key, so nothing carries a
    mode from the row into the engine. This guards the positional build: the
    old ChannelRequest took the policy SECOND, and leaving that argument in
    place while the dataclass changed would have silently passed a policy string
    as the broadcast's title."""
    import inspect
    src = inspect.getsource(li.start_through_engine)
    assert "policy_for_channel" not in src
    assert 'getattr(row, "live_mode"' not in src
    from kaizer_live.service import ChannelRequest
    order = list(ChannelRequest.__dataclass_fields__)
    assert order[:5] == ["channel_id", "title", "description", "privacy", "thumbnail"], order


def test_the_retired_live_mode_column_is_still_migrated():
    """Kept, not dropped, and still added by the ladder.

    Nothing reads it -- the mode it recorded no longer exists -- but the column
    must still be created on every deployment. create_all only makes MISSING
    TABLES, never ALTERs an existing one, so a column that is in the model and
    not in the ladder is a column that does not exist on any database that
    already had the table.
    """
    import pathlib
    cols = {c.name for c in models.LiveStream.__table__.columns}
    if "live_mode" not in cols:
        # A deployment that only ever received the finished result never had
        # this column: it was added and retired inside one campaign. Nothing
        # reads it either way, so its absence is correct, not a gap.
        import pytest as _pt
        _pt.skip("this deployment never had LiveStream.live_mode")
    src = (pathlib.Path(li.__file__).parent / "main.py").read_text(encoding="utf-8")
    assert '"live_streams", "live_mode"' in src


def test_the_engine_video_id_is_safe_in_a_url_and_a_redis_key():
    """It becomes an RTMP path segment and a Redis key, and is truncated to 64
    chars as LiveStream.batch_id. Anything needing escaping would break one of
    the three."""
    vid = li.engine_video_id("C3Fp/vNq!s Ug", 3)
    assert vid == "C3FpvNqsUg-3"
    assert all(c.isalnum() or c in "-_" for c in vid)


# ── start_stream must not quietly fall back ──────────────────────────

def test_start_stream_never_falls_back_to_the_quota_path():
    """If the engine refuses, the broadcast fails with the reason.

    Falling back to the classic path would spend quota to start a broadcast the
    operator was told had failed, and they would never know it had happened.

    The failure is recorded by prepare_and_start rather than inline, because the
    file is checked before anything goes live and repairing one is minutes of
    re-encoding -- far longer than an HTTP request should be held open. So the
    row goes to "preparing", a thread does the work, and the reason lands on the
    row either way. What must never appear in EITHER is a second attempt down
    the old path.
    """
    import inspect
    import routers.live_studio as ls
    src = inspect.getsource(ls.start_stream)
    assert "NO SILENT FALLBACK" in src
    assert "prepare_and_start" in src, "the engine path no longer starts the work"

    hand_off = inspect.getsource(li.prepare_and_start)
    assert "start_through_engine" in hand_off
    assert 'row.status = "failed"' in hand_off and "live engine refused" in hand_off, \
        "a refusal must be written to the row, not only raised into a dead thread"
    # A thread has nobody to raise to, so the reason must also reach the log.
    assert "flush=True" in hand_off


def test_the_engine_path_demands_a_finished_upload():
    """The classic pusher can start at 5 MB because it reads forward. The
    engine probes the file first, and ffprobe on a half-written MP4 whose
    index is at the end tells us nothing."""
    import inspect
    import routers.live_studio as ls
    src = inspect.getsource(ls.start_stream)
    assert "row.upload_done" in src and "checks the file before it starts" in src
