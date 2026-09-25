"""Desktop-mode tests — desktop_entry, runner spawn/planner helpers,
config ENV_PATH relocation, and the routers/desktop_local surface.

Deliberately does NOT import main (reload-importing the whole app per
env-combination is too heavy); the `python -c "import main"` smoke runs
with and without KAIZER_DESKTOP=1 are part of the task's manual battery.
"""
from __future__ import annotations

import importlib
import os
import socket
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import desktop_entry
import runner
from routers import desktop_local

ALL_KEYS = ("GEMINI_API_KEY", "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY", "DEEPGRAM_API_KEY",
            "KAIZER_V4_TRIM_PLANNER")


# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """Blank slate: no AI keys in the process env, KAIZER_ENV_DIR → tmp.
    Restores EVERYTHING after the test (including keys the endpoint under
    test wrote straight into os.environ)."""
    snapshot = {k: os.environ.get(k) for k in ALL_KEYS}
    for k in ALL_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("KAIZER_ENV_DIR", str(tmp_path))
    monkeypatch.delenv("KAIZER_DESKTOP", raising=False)
    yield tmp_path
    for k, v in snapshot.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture
def client(clean_env):
    app = FastAPI()
    app.include_router(desktop_local.router)
    return TestClient(app)


# ─── desktop_entry helpers ───────────────────────────────────────────────────

def test_desktop_mode_predicate(monkeypatch):
    monkeypatch.delenv("KAIZER_DESKTOP", raising=False)
    assert desktop_entry.desktop_mode() is False
    monkeypatch.setenv("KAIZER_DESKTOP", "1")
    assert desktop_entry.desktop_mode() is True
    monkeypatch.setenv("KAIZER_DESKTOP", "0")
    assert desktop_entry.desktop_mode() is False
    monkeypatch.setenv("KAIZER_DESKTOP", " 1 ")
    assert desktop_entry.desktop_mode() is True


def test_database_url_forward_slashes(tmp_path):
    url = desktop_entry._database_url(tmp_path)
    assert url.startswith("sqlite:///")
    assert "\\" not in url
    assert url.endswith("/kaizer.db")
    # absolute path preserved (drive letter on Windows)
    assert tmp_path.resolve().as_posix() in url


def test_serve_parser_defaults():
    args = desktop_entry.build_parser().parse_args(["serve"])
    assert args.command == "serve"
    assert args.port == 8765
    assert args.data_dir  # a non-empty default exists for dev runs


def test_serve_parser_explicit():
    args = desktop_entry.build_parser().parse_args(
        ["serve", "--port", "9123", "--data-dir", "C:/tmp/kx"])
    assert args.port == 9123
    assert args.data_dir == "C:/tmp/kx"


def test_render_command_forwards_argv_verbatim():
    rest = ["--job-id", "5", "--source", "a.mp4",
            "--output-dir", "d", "--language", "te"]
    command, forwarded = desktop_entry.split_command(["render"] + rest)
    assert command == "render"
    assert forwarded == rest  # verbatim — nothing consumed or reordered
    # non-render argv passes through untouched to argparse
    command, forwarded = desktop_entry.split_command(["serve", "--port", "9"])
    assert command is None
    assert forwarded == ["serve", "--port", "9"]


def test_apply_desktop_env(tmp_path):
    # _apply_desktop_env writes os.environ DIRECTLY (that's its job), so
    # monkeypatch can't track it — snapshot/restore by hand or KAIZER_DESKTOP
    # leaks into every later test in this process (main would import in
    # desktop shape and e.g. the admin-router tests would 404).
    names = ("KAIZER_DESKTOP", "DATABASE_URL",
             "KAIZER_OUTPUT_ROOT", "KAIZER_ENV_DIR",
             "KAIZER_MEDIA_ROOT", "KAIZER_JWT_SECRET")
    saved = {n: os.environ.get(n) for n in names}
    try:
        for n in names:
            os.environ.pop(n, None)
        data_dir = tmp_path / "kx data"
        desktop_entry._apply_desktop_env(data_dir)
        assert os.environ["KAIZER_DESKTOP"] == "1"
        assert os.environ["DATABASE_URL"] == \
            "sqlite:///" + data_dir.resolve().as_posix() + "/kaizer.db"
        assert os.environ["KAIZER_OUTPUT_ROOT"] == str(data_dir / "output")
        assert os.environ["KAIZER_ENV_DIR"] == str(data_dir)
        assert os.environ["KAIZER_MEDIA_ROOT"] == str(data_dir / "media")
        assert (data_dir / "output").is_dir()
        assert (data_dir / "media").is_dir()
        # JWT secret provisioned + persisted (identity survives restarts)
        secret = os.environ["KAIZER_JWT_SECRET"]
        assert secret
        assert f"KAIZER_JWT_SECRET={secret}" in \
            (data_dir / ".env").read_text(encoding="utf-8")
    finally:
        for n, v in saved.items():
            if v is None:
                os.environ.pop(n, None)
            else:
                os.environ[n] = v


def test_port_in_use_probe():
    # Occupied port → True
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        assert desktop_entry._port_in_use(port) is True
    finally:
        srv.close()
    # Freshly freed / unused port → False
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    free_port = probe.getsockname()[1]
    probe.close()
    assert desktop_entry._port_in_use(free_port) is False


def test_apply_frozen_paths_noop_when_not_frozen(monkeypatch):
    """Dev checkout: PATH and PLAYWRIGHT_BROWSERS_PATH stay untouched."""
    monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
    before = os.environ.get("PATH", "")
    assert not getattr(sys, "frozen", False)
    desktop_entry._apply_frozen_paths()
    assert os.environ.get("PATH", "") == before
    assert "PLAYWRIGHT_BROWSERS_PATH" not in os.environ


def test_apply_frozen_paths_with_resources(monkeypatch, tmp_path):
    monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
    (tmp_path / "bin").mkdir()
    (tmp_path / "ms-playwright").mkdir()
    before = os.environ.get("PATH", "")
    monkeypatch.setenv("PATH", before)  # register restore
    desktop_entry._apply_frozen_paths(resources=tmp_path)
    path = os.environ["PATH"]
    assert path.split(os.pathsep)[0] == str(tmp_path / "bin")  # PREPENDED
    assert os.environ["PLAYWRIGHT_BROWSERS_PATH"] == str(tmp_path / "ms-playwright")
    # idempotent — a second call must not duplicate the PATH entry
    desktop_entry._apply_frozen_paths(resources=tmp_path)
    assert os.environ["PATH"].split(os.pathsep).count(str(tmp_path / "bin")) == 1
    monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)


# ─── runner: frozen spawn argv contract ──────────────────────────────────────

def test_frozen_spawn_argv_matches_orchestrator_contract():
    frozen = runner.build_v4_spawn_cmd(
        7, "src.mp4", "out_dir", "te", "logo.png", frozen=True)
    dev = runner.build_v4_spawn_cmd(
        7, "src.mp4", "out_dir", "te", "logo.png", frozen=False)
    # Shapes
    assert frozen[:2] == [sys.executable, "render"]
    assert dev[1:3] == ["-m", "pipeline_v4.orchestrator"]
    # THE contract: identical flag tails (string-compare)
    assert frozen[2:] == dev[3:]
    assert frozen[2:] == [
        "--job-id", "7",
        "--source", "src.mp4",
        "--output-dir", "out_dir",
        "--language", "te",
        "--brand-logo", "logo.png",
    ]
    # Every flag used is one pipeline_v4.orchestrator's argparse accepts
    flags = [a for a in frozen[2:] if a.startswith("--")]
    assert set(flags) <= set(desktop_entry.ORCHESTRATOR_ARGV_CONTRACT)


def test_spawn_argv_no_logo_and_language_default():
    frozen = runner.build_v4_spawn_cmd(3, "v.mp4", "o", "", "", frozen=True)
    assert "--brand-logo" not in frozen
    # empty language falls back to "te" exactly like the old inline code
    assert frozen[frozen.index("--language") + 1] == "te"


# ─── runner: desktop trim-planner default ────────────────────────────────────

def test_trim_planner_saas_default_stays_claude(monkeypatch):
    monkeypatch.delenv("KAIZER_DESKTOP", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert runner._resolve_trim_planner("claude") == "claude"
    assert runner._resolve_trim_planner("") == "claude"
    assert runner._resolve_trim_planner("garbage") == "claude"
    assert runner._resolve_trim_planner("gemini") == "gemini"


def test_trim_planner_desktop_without_anthropic_defaults_gemini(monkeypatch):
    monkeypatch.setenv("KAIZER_DESKTOP", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert runner._resolve_trim_planner("claude") == "gemini"
    assert runner._resolve_trim_planner("") == "gemini"
    assert runner._resolve_trim_planner(None) == "gemini"
    assert runner._resolve_trim_planner("gemini") == "gemini"


def test_trim_planner_desktop_with_anthropic_keeps_claude(monkeypatch):
    monkeypatch.setenv("KAIZER_DESKTOP", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert runner._resolve_trim_planner("claude") == "claude"


# ─── config: ENV_PATH honors KAIZER_ENV_DIR ──────────────────────────────────

def test_config_env_path_relocation(monkeypatch, tmp_path):
    import config
    monkeypatch.setenv("KAIZER_ENV_DIR", str(tmp_path))
    importlib.reload(config)
    try:
        assert config.ENV_PATH == tmp_path / ".env"
    finally:
        monkeypatch.delenv("KAIZER_ENV_DIR")
        importlib.reload(config)
    # Unset → byte-identical to the historical behavior
    assert config.ENV_PATH == config.BASE_DIR / ".env"


# ─── routers/desktop_local: keys surface ─────────────────────────────────────

def test_keys_list_empty_masked(client):
    r = client.get("/api/desktop-local/keys")
    assert r.status_code == 200
    body = r.json()
    names = [k["name"] for k in body["keys"]]
    assert names == list(ALL_KEYS)
    for k in body["keys"]:
        assert k["set"] is False
        assert k["masked"] == ""


def test_post_key_writes_env_file_and_process_env(client, clean_env):
    r = client.post("/api/desktop-local/keys",
                    json={"name": "GEMINI_API_KEY", "value": "AIzaSyTEST1234"})
    assert r.status_code == 200
    assert r.json() == {"name": "GEMINI_API_KEY", "set": True, "masked": "****1234"}
    # os.environ — running process + future render children see it
    assert os.environ["GEMINI_API_KEY"] == "AIzaSyTEST1234"
    # userData .env — survives restart
    env_file = clean_env / ".env"
    assert env_file.exists()
    assert "GEMINI_API_KEY=AIzaSyTEST1234" in env_file.read_text(encoding="utf-8")
    # GET now shows masked last-4, never the full value
    listed = client.get("/api/desktop-local/keys").json()["keys"]
    gem = next(k for k in listed if k["name"] == "GEMINI_API_KEY")
    assert gem["set"] is True and gem["masked"] == "****1234"
    assert "AIzaSyTEST1234" not in str(listed)


def test_post_key_update_replaces_line(client, clean_env):
    client.post("/api/desktop-local/keys",
                json={"name": "DEEPGRAM_API_KEY", "value": "dg_old_0001"})
    client.post("/api/desktop-local/keys",
                json={"name": "DEEPGRAM_API_KEY", "value": "dg_new_0002"})
    text = (clean_env / ".env").read_text(encoding="utf-8")
    assert text.count("DEEPGRAM_API_KEY=") == 1
    assert "dg_new_0002" in text and "dg_old_0001" not in text
    assert os.environ["DEEPGRAM_API_KEY"] == "dg_new_0002"


def test_post_key_empty_value_clears(client, clean_env):
    client.post("/api/desktop-local/keys",
                json={"name": "OPENAI_API_KEY", "value": "sk-abcd"})
    client.post("/api/desktop-local/keys",
                json={"name": "OPENAI_API_KEY", "value": ""})
    assert "OPENAI_API_KEY" not in os.environ
    assert "OPENAI_API_KEY=" not in (clean_env / ".env").read_text(encoding="utf-8")


def test_post_key_preserves_other_env_lines(client, clean_env):
    (clean_env / ".env").write_text("KAIZER_ENCRYPTION_KEY=abc123\n",
                                    encoding="utf-8")
    client.post("/api/desktop-local/keys",
                json={"name": "GEMINI_API_KEY", "value": "gk_xyz9"})
    text = (clean_env / ".env").read_text(encoding="utf-8")
    assert "KAIZER_ENCRYPTION_KEY=abc123" in text
    assert "GEMINI_API_KEY=gk_xyz9" in text


def test_post_key_rejects_non_whitelisted_name(client, clean_env):
    r = client.post("/api/desktop-local/keys",
                    json={"name": "DATABASE_URL", "value": "sqlite:///evil.db"})
    assert r.status_code == 400
    # plain-language error, and nothing written anywhere
    assert "Supported keys" in r.json()["detail"]
    assert not (clean_env / ".env").exists()
    assert os.environ.get("DATABASE_URL") != "sqlite:///evil.db"


# ─── routers/desktop_local: preflight ────────────────────────────────────────

def test_preflight_no_keys_honest_reasons(client):
    body = client.get("/api/desktop-local/preflight").json()
    f = body["features"]
    assert body["ready"] is False
    assert f["render"]["ready"] is False and "Gemini" in f["render"]["reason"]
    assert f["seo"]["ready"] is False and "Gemini" in f["seo"]["reason"]
    assert f["image_generation"]["ready"] is False
    assert "OpenAI" in f["image_generation"]["reason"]
    assert f["transcription"]["ready"] is False
    assert "Deepgram" in f["transcription"]["reason"]
    # Claude trim planner honestly unavailable + Gemini recommended
    assert f["trim_planner"]["claude_available"] is False
    assert f["trim_planner"]["recommended"] == "gemini"
    assert "Anthropic" in f["trim_planner"]["reason"]


def test_preflight_gemini_only(client):
    client.post("/api/desktop-local/keys",
                json={"name": "GEMINI_API_KEY", "value": "gk_1234"})
    f = client.get("/api/desktop-local/preflight").json()["features"]
    assert f["render"]["ready"] is True and f["render"]["reason"] == ""
    assert f["seo"]["ready"] is True
    assert f["image_generation"]["ready"] is True   # gemini counts
    assert f["transcription"]["ready"] is False     # still needs deepgram
    assert f["trim_planner"]["recommended"] == "gemini"


def test_preflight_openai_satisfies_image_gen(client):
    client.post("/api/desktop-local/keys",
                json={"name": "OPENAI_API_KEY", "value": "sk-9999"})
    f = client.get("/api/desktop-local/preflight").json()["features"]
    assert f["image_generation"]["ready"] is True
    assert f["render"]["ready"] is False  # openai does NOT unlock render


def test_preflight_anthropic_enables_claude(client):
    client.post("/api/desktop-local/keys",
                json={"name": "ANTHROPIC_API_KEY", "value": "sk-ant-01"})
    f = client.get("/api/desktop-local/preflight").json()["features"]
    assert f["trim_planner"]["claude_available"] is True
    assert f["trim_planner"]["recommended"] == "claude"
    assert f["trim_planner"]["reason"] == ""


# ─── routers/desktop_local: value hygiene (.env injection) ───────────────────

def test_post_key_newline_injection_sanitized(client, clean_env):
    """A newline inside a value would smuggle a 2nd NAME=value line into
    the .env, bypassing the whitelist. \\r/\\n are stripped before writing."""
    r = client.post(
        "/api/desktop-local/keys",
        json={"name": "GEMINI_API_KEY",
              "value": "goodpart\nDATABASE_URL=sqlite:///evil.db"})
    assert r.status_code == 200
    lines = (clean_env / ".env").read_text(encoding="utf-8").splitlines()
    # ONE line, and it's ours — the injected name never becomes a line
    assert len(lines) == 1
    assert lines[0].startswith("GEMINI_API_KEY=")
    assert not any(l.startswith("DATABASE_URL=") for l in lines)
    assert "\n" not in os.environ["GEMINI_API_KEY"]
    # \r\n variant too
    client.post("/api/desktop-local/keys",
                json={"name": "GEMINI_API_KEY",
                      "value": "abc\r\nOPENAI_API_KEY=stolen"})
    lines = (clean_env / ".env").read_text(encoding="utf-8").splitlines()
    assert not any(l.startswith("OPENAI_API_KEY=") for l in lines)


def test_post_key_length_capped(client, clean_env):
    r = client.post("/api/desktop-local/keys",
                    json={"name": "GEMINI_API_KEY", "value": "x" * 5000})
    assert r.status_code == 422  # pydantic max_length=4096 rejects it
    assert not (clean_env / ".env").exists()


# ─── routers/desktop_local: KAIZER_V4_TRIM_PLANNER plain setting ─────────────

def test_trim_planner_setting_roundtrip(client, clean_env):
    r = client.post("/api/desktop-local/keys",
                    json={"name": "KAIZER_V4_TRIM_PLANNER", "value": "gemini"})
    assert r.status_code == 200
    assert r.json()["set"] is True
    assert r.json()["value"] == "gemini"
    assert os.environ["KAIZER_V4_TRIM_PLANNER"] == "gemini"
    assert "KAIZER_V4_TRIM_PLANNER=gemini" in \
        (clean_env / ".env").read_text(encoding="utf-8")
    # GET returns it PLAIN (a setting, not a secret) — never masked
    listed = client.get("/api/desktop-local/keys").json()["keys"]
    row = next(k for k in listed if k["name"] == "KAIZER_V4_TRIM_PLANNER")
    assert row["set"] is True
    assert row["secret"] is False
    assert row["value"] == "gemini"
    assert row["masked"] == "gemini"          # not "****mini"
    assert set(row["choices"]) == {"gemini", "claude"}
    # "Claude" (any case) accepted + normalized
    r2 = client.post("/api/desktop-local/keys",
                     json={"name": "KAIZER_V4_TRIM_PLANNER", "value": "Claude"})
    assert r2.status_code == 200 and r2.json()["value"] == "claude"
    # empty clears it
    client.post("/api/desktop-local/keys",
                json={"name": "KAIZER_V4_TRIM_PLANNER", "value": ""})
    assert "KAIZER_V4_TRIM_PLANNER" not in os.environ


def test_trim_planner_setting_rejects_unknown_value(client, clean_env):
    r = client.post("/api/desktop-local/keys",
                    json={"name": "KAIZER_V4_TRIM_PLANNER", "value": "gpt-5"})
    assert r.status_code == 400
    # plain-language error naming the real choices; nothing saved
    assert "gemini" in r.json()["detail"] and "claude" in r.json()["detail"]
    assert "KAIZER_V4_TRIM_PLANNER" not in os.environ
    assert not (clean_env / ".env").exists()


# ─── runner: saved desktop setting drives the default ────────────────────────

def test_trim_planner_desktop_honors_saved_setting(monkeypatch):
    monkeypatch.setenv("KAIZER_DESKTOP", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("KAIZER_V4_TRIM_PLANNER", "gemini")
    # no valid per-job pick → the saved setting wins
    assert runner._resolve_trim_planner("") == "gemini"
    assert runner._resolve_trim_planner(None) == "gemini"
    assert runner._resolve_trim_planner("garbage") == "gemini"
    # an explicit per-job pick still beats the saved setting
    assert runner._resolve_trim_planner("claude") == "claude"
    monkeypatch.setenv("KAIZER_V4_TRIM_PLANNER", "claude")
    assert runner._resolve_trim_planner("") == "claude"
    # saved "claude" with no Anthropic key still downgrades safely
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert runner._resolve_trim_planner("") == "gemini"


def test_trim_planner_saas_ignores_env_setting(monkeypatch):
    monkeypatch.delenv("KAIZER_DESKTOP", raising=False)
    monkeypatch.setenv("KAIZER_V4_TRIM_PLANNER", "gemini")
    assert runner._resolve_trim_planner("") == "claude"  # SaaS byte-identical
    monkeypatch.delenv("KAIZER_V4_TRIM_PLANNER")


# ─── desktop_entry: JWT secret persistence (identity survives restarts) ──────

def test_jwt_secret_provisioned_once_and_reused(tmp_path):
    saved = os.environ.get("KAIZER_JWT_SECRET")
    try:
        os.environ.pop("KAIZER_JWT_SECRET", None)
        data_dir = tmp_path / "kx"
        data_dir.mkdir()
        # first boot: generates + persists into the userData .env
        desktop_entry._provision_jwt_secret(data_dir)
        first = os.environ["KAIZER_JWT_SECRET"]
        assert first
        env_text = (data_dir / ".env").read_text(encoding="utf-8")
        assert f"KAIZER_JWT_SECRET={first}" in env_text
        # "second boot": fresh process env, same data dir → SAME secret,
        # still exactly one line (no duplicate appends)
        os.environ.pop("KAIZER_JWT_SECRET", None)
        desktop_entry._provision_jwt_secret(data_dir)
        assert os.environ["KAIZER_JWT_SECRET"] == first
        assert (data_dir / ".env").read_text(
            encoding="utf-8").count("KAIZER_JWT_SECRET=") == 1
        # explicit env always wins — file untouched
        os.environ["KAIZER_JWT_SECRET"] = "operator-pinned"
        desktop_entry._provision_jwt_secret(data_dir)
        assert os.environ["KAIZER_JWT_SECRET"] == "operator-pinned"
    finally:
        if saved is None:
            os.environ.pop("KAIZER_JWT_SECRET", None)
        else:
            os.environ["KAIZER_JWT_SECRET"] = saved


def test_jwt_secret_preserves_other_env_lines(tmp_path):
    saved = os.environ.get("KAIZER_JWT_SECRET")
    try:
        os.environ.pop("KAIZER_JWT_SECRET", None)
        (tmp_path / ".env").write_text("GEMINI_API_KEY=gk_1\n",
                                       encoding="utf-8")
        desktop_entry._provision_jwt_secret(tmp_path)
        text = (tmp_path / ".env").read_text(encoding="utf-8")
        assert "GEMINI_API_KEY=gk_1" in text
        assert "KAIZER_JWT_SECRET=" in text
    finally:
        if saved is None:
            os.environ.pop("KAIZER_JWT_SECRET", None)
        else:
            os.environ["KAIZER_JWT_SECRET"] = saved


# ─── desktop_entry: frozen resources layout + parent watchdog ────────────────

def test_resources_path_matches_shipped_layout(monkeypatch, tmp_path):
    """Shipped: <resources>/python/kaizer_backend/kaizer_backend.exe —
    resources/ is parents[2] of the exe (bin/ and ms-playwright/ live
    directly under resources/)."""
    exe = tmp_path / "resources" / "python" / "kaizer_backend" / "kaizer_backend.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    assert desktop_entry._resources_path() == (tmp_path / "resources").resolve()


def test_serve_parser_parent_pid():
    args = desktop_entry.build_parser().parse_args(
        ["serve", "--parent-pid", "4321"])
    assert args.parent_pid == 4321
    # absent → 0 → watchdog is a no-op
    args = desktop_entry.build_parser().parse_args(["serve"])
    assert args.parent_pid == 0


def test_parent_watchdog_noop_without_pid():
    import threading
    before = {t.name for t in threading.enumerate()}
    desktop_entry._start_parent_watchdog(0)
    desktop_entry._start_parent_watchdog(-5)
    desktop_entry._start_parent_watchdog(None)
    after = {t.name for t in threading.enumerate()}
    assert "kaizer-parent-watchdog" not in (after - before)


# ─── SPA fallback (desktop "/" mount) ────────────────────────────────────────

@pytest.fixture
def spa_client(tmp_path):
    """A FastAPI app shaped like desktop main.py: /api routes first, then
    the SPA-aware "/" mount (desktop_local.SPAStaticFiles)."""
    spa = tmp_path / "spa_dist"
    spa.mkdir()
    (spa / "index.html").write_text("<html>KAIZER-SPA-SHELL</html>",
                                    encoding="utf-8")
    (spa / "real.txt").write_text("real-static-file", encoding="utf-8")
    app = FastAPI()

    @app.get("/api/health/")
    def health():
        return {"status": "ok"}

    app.mount("/", desktop_local.SPAStaticFiles(directory=str(spa),
                                                html=True), name="spa")
    return TestClient(app)


def test_spa_fallback_serves_index_for_client_routes(spa_client):
    for route in ("/app", "/jobs/5", "/settings/deep/link"):
        r = spa_client.get(route)
        assert r.status_code == 200, route
        assert "KAIZER-SPA-SHELL" in r.text, route


def test_spa_root_and_real_files_still_served(spa_client):
    assert "KAIZER-SPA-SHELL" in spa_client.get("/").text  # html=True index
    assert spa_client.get("/real.txt").text == "real-static-file"
    assert spa_client.get("/api/health/").json() == {"status": "ok"}


def test_spa_fallback_never_swallows_api_or_media_404(spa_client):
    for route in ("/api/nope", "/api/jobs/999/nothing", "/media/gone.mp4"):
        r = spa_client.get(route)
        assert r.status_code == 404, route
        assert "KAIZER-SPA-SHELL" not in r.text, route
        assert r.headers["content-type"].startswith("application/json"), route
