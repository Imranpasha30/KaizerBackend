"""tests/test_avatar_providers.py — the ported avatar provider stack.

Covers (per the port task):
  - registry resolution order: explicit name > AVATAR_PROVIDER env > smart
    default (avatar_studio when its checkout exists, else heygen)
  - get_provider(unknown) -> AvatarProviderError
  - HeyGenProvider.available(): False without env, True with
    AVATAR_HEYGEN_ENABLED + HEYGEN_API_KEY (env only — the real API is
    NEVER called)
  - AvatarStudioProvider.available(): False with a clear reason when the
    studio dir is missing; studio_dir() resolution (env wins, sibling
    fallback = the KaizerBackend repo-root sibling — the port change)
  - GenerationRequest / GenerationResult dataclass contract
  - router-level: GPU gate serializes local renders / exempts heygen,
    plus cheap TestClient checks of /status and /providers

No real generation, no HeyGen API calls, no GPU work — providers are
either exercised through env/filesystem-only paths or replaced by fakes.
"""
from __future__ import annotations

import dataclasses
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import avatar
from avatar import (
    AvatarProviderError,
    GenerationRequest,
    GenerationResult,
    default_provider,
    gender_counts,
    get_provider,
    provider_names,
)
from avatar.base import VoiceInfo
from avatar.heygen_provider import HeyGenProvider
from avatar.studio import AvatarStudioProvider, studio_dir


# ───────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_registry_and_env(monkeypatch):
    """Deterministic env + a fresh provider-instance cache per test.

    AvatarStudioProvider snapshots its root at __init__, so a cached
    instance from another test (or the app) would leak stale paths.
    """
    for var in ("AVATAR_PROVIDER", "AVATAR_HEYGEN_ENABLED",
                "HEYGEN_API_KEY", "AVATAR_STUDIO_DIR",
                "AVATAR_VAST_INSTANCE"):
        monkeypatch.delenv(var, raising=False)
    avatar._instances.clear()
    yield
    avatar._instances.clear()


@pytest.fixture
def studio_checkout(tmp_path, monkeypatch):
    """A fake avatar-studio checkout (generate.py present) wired via env."""
    root = tmp_path / "avatar-studio"
    root.mkdir()
    (root / "generate.py").write_text("# stub\n", encoding="utf-8")
    monkeypatch.setenv("AVATAR_STUDIO_DIR", str(root))
    return root


@pytest.fixture
def no_studio(tmp_path, monkeypatch):
    """AVATAR_STUDIO_DIR pointing at an empty dir — studio unavailable."""
    root = tmp_path / "empty"
    root.mkdir()
    monkeypatch.setenv("AVATAR_STUDIO_DIR", str(root))
    return root


# ───────────────────────────────────────────────────────────────────
# Registry resolution
# ───────────────────────────────────────────────────────────────────

def test_provider_names_contract():
    assert provider_names() == ["avatar_studio", "echomimic", "heygen"]


def test_explicit_name_beats_env(monkeypatch, studio_checkout):
    monkeypatch.setenv("AVATAR_PROVIDER", "avatar_studio")
    assert get_provider("heygen").name == "heygen"


def test_env_beats_smart_default(monkeypatch, studio_checkout):
    # smart default would say avatar_studio (checkout exists) — env wins
    monkeypatch.setenv("AVATAR_PROVIDER", "heygen")
    assert get_provider().name == "heygen"


def test_env_name_is_normalized(monkeypatch):
    monkeypatch.setenv("AVATAR_PROVIDER", "  HeyGen  ")
    assert get_provider().name == "heygen"


def test_smart_default_prefers_studio_when_checkout_exists(studio_checkout):
    assert default_provider() == "avatar_studio"
    assert get_provider().name == "avatar_studio"


def test_smart_default_falls_back_to_heygen_without_studio(no_studio):
    assert default_provider() == "heygen"
    assert get_provider().name == "heygen"


def test_get_provider_unknown_raises():
    with pytest.raises(AvatarProviderError) as exc:
        get_provider("definitely_not_a_provider")
    msg = str(exc.value)
    assert "unknown avatar provider" in msg
    # message must name the valid choices so the caller can self-serve
    for name in provider_names():
        assert name in msg


def test_get_provider_caches_instances(no_studio):
    assert get_provider("heygen") is get_provider("heygen")


# ───────────────────────────────────────────────────────────────────
# HeyGenProvider.available() — env only, never the real API
# ───────────────────────────────────────────────────────────────────

def test_heygen_unavailable_without_env():
    ready, reason = HeyGenProvider().available()
    assert ready is False
    assert "disabled" in reason
    assert "AVATAR_HEYGEN_ENABLED" in reason


def test_heygen_unavailable_without_api_key(monkeypatch):
    monkeypatch.setenv("AVATAR_HEYGEN_ENABLED", "true")
    ready, reason = HeyGenProvider().available()
    assert ready is False
    assert "HEYGEN_API_KEY" in reason


def test_heygen_available_with_flag_and_key(monkeypatch):
    monkeypatch.setenv("AVATAR_HEYGEN_ENABLED", "true")
    monkeypatch.setenv("HEYGEN_API_KEY", "test-key-not-real")
    ready, reason = HeyGenProvider().available()
    assert ready is True
    assert reason == ""


# ───────────────────────────────────────────────────────────────────
# HeyGenProvider catalog — AvatarProviderError contract (never a raw
# heygen.client error), preview_url passthrough. Real API never called.
# ───────────────────────────────────────────────────────────────────

def test_heygen_catalog_raises_provider_error_when_disabled():
    """Disabled/keyless heygen must raise AvatarProviderError from the
    catalog methods — NOT the raw HeyGenAuthError (a RuntimeError the
    router doesn't catch, which used to 500 the catalog endpoints)."""
    import avatar.heygen_provider as hp
    p = HeyGenProvider()
    for method in (p.list_avatars, p.list_voices):
        with pytest.raises(AvatarProviderError) as exc:
            method()
        assert "disabled" in str(exc.value)
        assert not isinstance(exc.value, hp.heygen_client.HeyGenError)


def test_heygen_catalog_wraps_client_errors(monkeypatch):
    """With env configured, raw heygen.client failures are re-raised as
    AvatarProviderError so the router's 502 mapping applies."""
    import avatar.heygen_provider as hp
    monkeypatch.setenv("AVATAR_HEYGEN_ENABLED", "true")
    monkeypatch.setenv("HEYGEN_API_KEY", "test-key-not-real")

    def boom(**kw):
        raise hp.heygen_client.HeyGenError("api down")

    monkeypatch.setattr(hp.heygen_client, "list_avatars", boom)
    monkeypatch.setattr(hp.heygen_client, "list_voices", boom)
    with pytest.raises(AvatarProviderError, match="api down"):
        HeyGenProvider().list_avatars()
    with pytest.raises(AvatarProviderError, match="api down"):
        HeyGenProvider().list_voices()


def test_heygen_list_voices_populates_preview_url(monkeypatch):
    import avatar.heygen_provider as hp
    monkeypatch.setenv("AVATAR_HEYGEN_ENABLED", "true")
    monkeypatch.setenv("HEYGEN_API_KEY", "test-key-not-real")
    monkeypatch.setattr(hp.heygen_client, "list_voices", lambda **kw: [
        {"voice_id": "v1", "language": "Telugu", "name": "A",
         "gender": "female", "preview_audio": "https://cdn.example/v1.mp3"},
        {"voice_id": "v2", "language": "Hindi", "preview_audio": None},
        {"voice_id": "v3", "language": "English"},
    ])
    voices = HeyGenProvider().list_voices()
    assert [v.preview_url for v in voices] == [
        "https://cdn.example/v1.mp3", "", ""]


# ───────────────────────────────────────────────────────────────────
# AvatarStudioProvider.available() + studio_dir() resolution
# ───────────────────────────────────────────────────────────────────

def test_studio_unavailable_when_dir_missing(tmp_path):
    missing = tmp_path / "nowhere"
    ready, reason = AvatarStudioProvider(root=missing).available()
    assert ready is False
    assert "avatar-studio not found" in reason
    assert str(missing) in reason
    assert "AVATAR_STUDIO_DIR" in reason      # self-diagnosing fix hint


def test_studio_unavailable_when_venv_missing(studio_checkout):
    ready, reason = AvatarStudioProvider(root=studio_checkout).available()
    assert ready is False
    assert "venv missing" in reason


def test_studio_dir_env_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("AVATAR_STUDIO_DIR", str(tmp_path))
    assert studio_dir() == tmp_path.resolve()


def test_studio_dir_fallback_prefers_in_repo_engines():
    """The canonical non-env home is INSIDE the backend
    (<KaizerBackend>/engines/avatar-studio — one folder holds everything;
    gitignored + excluded from promote), then the legacy repo-sibling,
    then ~/avatar-studio."""
    import avatar.studio as studio_mod
    backend_root = Path(studio_mod.__file__).resolve().parents[1]
    assert backend_root.name == "KaizerBackend"
    engines = backend_root / "engines" / "avatar-studio"
    sibling = backend_root.parent / "avatar-studio"
    home = Path.home() / "avatar-studio"
    result = studio_dir()          # env cleared by the autouse fixture
    assert result in (engines, sibling, home)
    # with the real checkout installed under engines/, it MUST win
    if (engines / "generate.py").exists():
        assert result == engines


# ───────────────────────────────────────────────────────────────────
# Dataclass contract
# ───────────────────────────────────────────────────────────────────

def test_generation_request_contract(tmp_path):
    req = GenerationRequest(
        script="hello", avatar_id="avatar1", voice_id="te_f_studio",
        out_dir=tmp_path,
    )
    assert (req.width, req.height) == (1080, 1920)   # vertical by default
    assert req.language == "te"
    assert req.engine == ""
    assert req.meta == {}
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.script = "mutated"
    # meta default_factory must not be shared across instances
    req.meta["k"] = "v"
    assert GenerationRequest(
        script="x", avatar_id="a", voice_id="v", out_dir=tmp_path).meta == {}


def test_generation_result_contract(tmp_path):
    res = GenerationResult(
        video_path=tmp_path / "clip.mp4", duration_s=12.5, provider="avatar_studio",
    )
    assert res.engine == ""
    assert res.thumbnail_path is None
    assert res.meta == {}
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.duration_s = 0.0


def test_voiceinfo_preview_url_default_and_positional_compat():
    """preview_url was appended LAST with default "" — pre-existing
    positional construction (id, language, name, gender) must be
    unaffected, and the frozen contract must hold."""
    v = VoiceInfo(id="x", language="te")
    assert v.preview_url == ""
    v2 = VoiceInfo(id="y", language="te", name="n", gender="f",
                   preview_url="https://cdn.example/p.mp3")
    assert v2.preview_url == "https://cdn.example/p.mp3"
    v3 = VoiceInfo("z", "hi", "name z", "m")     # positional, pre-field order
    assert (v3.name, v3.gender, v3.preview_url) == ("name z", "m", "")
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.preview_url = "mutated"


def test_gender_counts_buckets_untagged_separately():
    voices = [
        VoiceInfo(id="te_f_a", language="te", gender="f"),
        VoiceInfo(id="te_f_b", language="te", gender="f"),
        VoiceInfo(id="te_m_a", language="te", gender="m"),
        VoiceInfo(id="anchor_f1", language="te", gender=""),
        VoiceInfo(id="weird", language="te", gender="x"),
    ]
    assert gender_counts(voices) == {"f": 2, "m": 1, "": 2}


# ───────────────────────────────────────────────────────────────────
# Router — GPU gate + cheap endpoint checks
# ───────────────────────────────────────────────────────────────────

import routers.avatar as ravatar  # noqa: E402


class _FakeSession:
    """DB stand-in — the fake renders below fail before any DB write."""
    def close(self) -> None:
        pass


@pytest.fixture
def _stub_runner(monkeypatch, tmp_path):
    """Stub the lazy ``from runner import OUTPUT_ROOT`` inside the worker."""
    monkeypatch.setitem(
        sys.modules, "runner", types.SimpleNamespace(OUTPUT_ROOT=tmp_path))


def test_router_gpu_gate_contract():
    # one slot, SHARED via services.gpu_gate (so avatar renders serialize
    # against every other ad-hoc GPU render, not just among themselves),
    # heygen (remote API) exempt
    from services.gpu_gate import GPU_GATE as shared_gate
    assert ravatar.GPU_GATE is shared_gate
    assert isinstance(ravatar.GPU_GATE, threading.Semaphore)
    assert ravatar.GPU_GATE._value == 1
    assert not hasattr(ravatar, "_GPU_GATE")   # module-local gate removed
    assert "heygen" in ravatar._GPU_EXEMPT_PROVIDERS
    assert "avatar_studio" not in ravatar._GPU_EXEMPT_PROVIDERS
    assert "echomimic" not in ravatar._GPU_EXEMPT_PROVIDERS


def _fake_render(key: str) -> threading.Thread:
    t = threading.Thread(
        target=ravatar._render_and_persist,
        kwargs=dict(
            key=key, user_id=1, provider_name=None, script="s",
            avatar_id="avatar1", voice_id="v1",
            platform="youtube_short", language="te",
        ),
        daemon=True,
    )
    t.start()
    return t


def test_gpu_gate_serializes_local_renders(monkeypatch, _stub_runner):
    """Two concurrent local renders must run one at a time, never together."""
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    class FakeLocal:
        name = "avatar_studio"          # NOT exempt -> must be gated

        def available(self):
            return True, ""

        def generate(self, request, on_progress=None):
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.15)
            with state_lock:
                active -= 1
            raise AvatarProviderError("test stop before DB write")

    fake = FakeLocal()
    monkeypatch.setattr(ravatar, "get_provider", lambda name=None: fake)
    monkeypatch.setattr(ravatar, "SessionLocal", _FakeSession)

    threads = [_fake_render(f"direct:gate-serial-{i}") for i in range(2)]
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive()

    assert max_active == 1, "second local render overlapped the first"
    for i in range(2):
        st = ravatar._get_status(f"direct:gate-serial-{i}")
        assert st["state"] == "error"
        assert "test stop" in st["error"]
    assert ravatar.GPU_GATE._value == 1     # gate fully released after both


def test_gpu_gate_second_render_waits_while_gate_held(monkeypatch, _stub_runner):
    """A local render arriving while the GPU is busy WAITS (does not run)."""
    class FakeLocal:
        name = "avatar_studio"

        def available(self):
            return True, ""

        def generate(self, request, on_progress=None):
            raise AvatarProviderError("test stop before DB write")

    monkeypatch.setattr(ravatar, "get_provider", lambda name=None: FakeLocal())
    monkeypatch.setattr(ravatar, "SessionLocal", _FakeSession)

    key = "direct:gate-wait"
    ravatar.GPU_GATE.acquire()              # simulate a render in flight
    try:
        t = _fake_render(key)
        time.sleep(0.3)
        assert t.is_alive(), "render ran while the GPU gate was held"
        st = ravatar._get_status(key)
        assert st["state"] == "rendering"
        assert "Waiting for the GPU render slot" in st.get("message", "")
    finally:
        ravatar.GPU_GATE.release()
    t.join(timeout=10)
    assert not t.is_alive()
    assert ravatar._get_status(key)["state"] == "error"
    assert ravatar.GPU_GATE._value == 1


def test_gpu_gate_exempts_heygen(monkeypatch, _stub_runner):
    """A heygen (remote API) render proceeds even while the gate is held."""
    class FakeRemote:
        name = "heygen"

        def available(self):
            return True, ""

        def generate(self, request, on_progress=None):
            raise AvatarProviderError("test stop before DB write")

    monkeypatch.setattr(ravatar, "get_provider", lambda name=None: FakeRemote())
    monkeypatch.setattr(ravatar, "SessionLocal", _FakeSession)

    key = "direct:gate-exempt"
    ravatar.GPU_GATE.acquire()
    try:
        t = _fake_render(key)
        t.join(timeout=5)
        assert not t.is_alive(), "heygen render was wrongly gated on the GPU"
        assert ravatar._get_status(key)["state"] == "error"
    finally:
        ravatar.GPU_GATE.release()
    assert ravatar.GPU_GATE._value == 1


# ── TestClient endpoint checks (no DB, no generation) ─────────────

@pytest.fixture
def client(no_studio):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import auth

    app = FastAPI()
    app.include_router(ravatar.router)
    app.dependency_overrides[auth.current_user] = lambda: SimpleNamespace(id=1)
    with TestClient(app) as c:
        yield c


def test_status_endpoint_idle_then_tracked(client):
    r = client.get("/api/avatar/status/never-started")
    assert r.status_code == 200
    assert r.json()["state"] == "idle"

    ravatar._set_status("direct:api-test", state="queued", progress=0)
    r = client.get("/api/avatar/status/direct:api-test")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "queued"
    assert "updated_at" in body


def test_providers_endpoint_reports_availability(client):
    # env fully cleared + AVATAR_STUDIO_DIR at an empty dir (no_studio):
    # every engine is unavailable, and the smart default is heygen.
    r = client.get("/api/avatar/providers")
    assert r.status_code == 200
    body = r.json()
    by_name = {p["name"]: p for p in body["providers"]}
    assert set(by_name) == {"avatar_studio", "echomimic", "heygen"}
    for p in by_name.values():
        assert p["available"] is False
        assert p["reason"]                    # self-diagnosing, never blank
    assert "avatar-studio not found" in by_name["avatar_studio"]["reason"]
    assert "disabled" in by_name["heygen"]["reason"]
    assert body["default"] == "heygen"


def test_providers_endpoint_survives_misconfigured_default(client, monkeypatch):
    """A default provider that fails to resolve must not 500 /providers —
    the rows still list, default goes None with the error surfaced."""
    real = ravatar.get_provider

    def flaky(name=None):
        if name is None:
            raise AvatarProviderError("default provider broken")
        return real(name)

    monkeypatch.setattr(ravatar, "get_provider", flaky)
    r = client.get("/api/avatar/providers")
    assert r.status_code == 200
    body = r.json()
    assert body["default"] is None
    assert "default provider broken" in body["default_error"]
    assert {p["name"] for p in body["providers"]} == set(provider_names())


def test_generate_rejects_unsafe_ids(client):
    """avatar_id/voice_id reach filesystem paths + CLI args — traversal
    and shell-ish strings must 422 before any thread spawns."""
    for bad in ("../../etc/passwd", "a/b", "a;rm -rf .", "x" * 65, ""):
        r = client.post("/api/avatar/generate",
                        json={"script": "hello", "avatar_id": bad})
        assert r.status_code == 422, f"avatar_id {bad!r} was accepted"
        r = client.post("/api/avatar/generate",
                        json={"script": "hello", "voice_id": bad})
        assert r.status_code == 422, f"voice_id {bad!r} was accepted"
    # the safe helper accepts the shipped defaults
    assert ravatar._require_safe_id("avatar1", "avatar_id") == "avatar1"
    assert ravatar._require_safe_id("anchor_f1", "voice_id") == "anchor_f1"
    assert ravatar._require_safe_id("te_f.studio-2", "voice_id")


def test_status_privacy_hides_other_users_entries(client):
    """Entries carry the creator's user_id; another caller gets 404 (the
    client fixture authenticates as user 1). Untagged entries stay open
    for back-compat with pre-fix keys."""
    ravatar._set_status("direct:privacy-other", state="queued", user_id=999)
    ravatar._set_status("direct:privacy-mine", state="queued", user_id=1)
    assert client.get(
        "/api/avatar/status/direct:privacy-other").status_code == 404
    r = client.get("/api/avatar/status/direct:privacy-mine")
    assert r.status_code == 200
    assert r.json()["state"] == "queued"


def test_status_privacy_admin_exempt(no_studio):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import auth

    app = FastAPI()
    app.include_router(ravatar.router)
    app.dependency_overrides[auth.current_user] = (
        lambda: SimpleNamespace(id=42, is_admin=True))
    ravatar._set_status("direct:privacy-admin", state="queued", user_id=999)
    with TestClient(app) as c:
        r = c.get("/api/avatar/status/direct:privacy-admin")
    assert r.status_code == 200


def test_set_status_evicts_stale_entries():
    """Every write evicts entries idle for >24h (process-hygiene cap)."""
    ravatar._set_status("direct:stale", state="done")
    with ravatar._lock:
        ravatar._status["direct:stale"]["updated_at"] = (
            time.time() - 25 * 3600)
    ravatar._set_status("direct:fresh", state="queued")
    with ravatar._lock:
        assert "direct:stale" not in ravatar._status
        assert "direct:fresh" in ravatar._status


def test_voice_sample_redirects_to_preview_url(client, monkeypatch):
    """A remote provider voice with a hosted preview redirects to it;
    a voice without one still 404s."""
    class FakeRemote:
        name = "heygen"

        def available(self):
            return True, ""

        def list_voices(self):
            return [
                VoiceInfo(id="v1", language="te",
                          preview_url="https://cdn.example/v1.mp3"),
                VoiceInfo(id="v2", language="te"),
            ]

    monkeypatch.setattr(ravatar, "get_provider", lambda name=None: FakeRemote())
    r = client.get("/api/avatar/voices/v1/sample", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "https://cdn.example/v1.mp3"
    r = client.get("/api/avatar/voices/v2/sample", follow_redirects=False)
    assert r.status_code == 404
