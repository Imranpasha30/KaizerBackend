"""Tests for pipeline_v2/scripts/preflight_v2_launch.py.

Mocked env-only smoke tests -- no live DB or network. The actual DB
check uses the existing engine which is mocked at import.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


_PREFLIGHT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "preflight_v2_launch.py"


def _import_preflight():
    """Load the script as a module under a stable name."""
    spec = importlib.util.spec_from_file_location(
        "preflight_v2_launch", _PREFLIGHT_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["preflight_v2_launch"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def pre():
    return _import_preflight()


@pytest.fixture
def all_env_set(monkeypatch):
    """Seed every required env var with a fake value so the env step
    passes without us having to repeat the list in every test."""
    for name in (
        "KAIZER_V2_ENABLED", "INNGEST_EVENT_KEY", "INNGEST_SIGNING_KEY",
        "DEEPGRAM_API_KEY", "GEMINI_API_KEY", "KAIZER_STT_DEFAULT_PROVIDER",
    ):
        monkeypatch.setenv(name, "fake-value")
    monkeypatch.setenv("KAIZER_V2_ENABLED", "1")
    monkeypatch.delenv("INNGEST_DEV", raising=False)
    monkeypatch.setenv("KAIZER_PREFLIGHT_SKIP_NETWORK", "1")
    yield


# ── check_env_vars ───────────────────────────────────────────────────


class TestCheckEnvVars:
    def test_all_set(self, pre, all_env_set):
        ok, lines = pre.check_env_vars()
        assert ok is True
        assert all("FAIL" not in ln for ln in lines if "PASS" in ln or "WARN" in ln)

    def test_one_missing(self, pre, monkeypatch, all_env_set):
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        ok, lines = pre.check_env_vars()
        assert ok is False
        assert any("FAIL" in ln and "DEEPGRAM_API_KEY" in ln for ln in lines)

    def test_empty_string_counts_as_missing(self, pre, monkeypatch, all_env_set):
        monkeypatch.setenv("GEMINI_API_KEY", "")
        ok, lines = pre.check_env_vars()
        assert ok is False
        assert any("FAIL" in ln and "GEMINI_API_KEY" in ln for ln in lines)

    def test_optional_var_missing_is_warn_not_fail(self, pre, monkeypatch, all_env_set):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        ok, lines = pre.check_env_vars()
        # Required vars still pass -> overall ok is True; just a WARN line.
        assert ok is True
        assert any("WARN" in ln and "OPENAI_API_KEY" in ln for ln in lines)


# ── check_inngest_dev_off ────────────────────────────────────────────


class TestCheckInngestDevOff:
    def test_unset_passes(self, pre, monkeypatch):
        monkeypatch.delenv("INNGEST_DEV", raising=False)
        ok, line = pre.check_inngest_dev_off()
        assert ok is True

    def test_zero_passes(self, pre, monkeypatch):
        monkeypatch.setenv("INNGEST_DEV", "0")
        ok, _ = pre.check_inngest_dev_off()
        assert ok is True

    def test_false_passes(self, pre, monkeypatch):
        monkeypatch.setenv("INNGEST_DEV", "false")
        ok, _ = pre.check_inngest_dev_off()
        assert ok is True

    def test_one_fails(self, pre, monkeypatch):
        monkeypatch.setenv("INNGEST_DEV", "1")
        ok, line = pre.check_inngest_dev_off()
        assert ok is False
        assert "FAIL" in line

    def test_truthy_string_fails(self, pre, monkeypatch):
        monkeypatch.setenv("INNGEST_DEV", "true")
        ok, _ = pre.check_inngest_dev_off()
        assert ok is False


# ── check_v2_enabled ─────────────────────────────────────────────────


class TestCheckV2Enabled:
    def test_one_passes(self, pre, monkeypatch):
        monkeypatch.setenv("KAIZER_V2_ENABLED", "1")
        ok, _ = pre.check_v2_enabled()
        assert ok is True

    def test_zero_fails(self, pre, monkeypatch):
        monkeypatch.setenv("KAIZER_V2_ENABLED", "0")
        ok, line = pre.check_v2_enabled()
        assert ok is False
        assert "FAIL" in line

    def test_unset_fails(self, pre, monkeypatch):
        # Unset = falls through to "" which the check treats as falsy
        # so V2 traffic would NOT be enabled. Preflight surfaces it.
        monkeypatch.delenv("KAIZER_V2_ENABLED", raising=False)
        ok, _ = pre.check_v2_enabled()
        assert ok is False


# ── check_inngest_cloud_reachable ────────────────────────────────────


class TestCheckInngestCloudReachable:
    def test_skip_when_env_set(self, pre, monkeypatch):
        monkeypatch.setenv("KAIZER_PREFLIGHT_SKIP_NETWORK", "1")
        ok, line = pre.check_inngest_cloud_reachable()
        assert ok is True
        assert "SKIP" in line

    def test_connect_called_when_skip_not_set(self, pre, monkeypatch):
        monkeypatch.delenv("KAIZER_PREFLIGHT_SKIP_NETWORK", raising=False)

        called = {"count": 0}
        class _FakeSock:
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def fake_create_connection(addr, timeout):
            called["count"] += 1
            assert addr == ("api.inngest.com", 443)
            return _FakeSock()

        with patch.object(pre.socket, "create_connection", fake_create_connection):
            ok, line = pre.check_inngest_cloud_reachable()
        assert ok is True
        assert called["count"] == 1
        assert "PASS" in line

    def test_connect_failure_reports_fail(self, pre, monkeypatch):
        monkeypatch.delenv("KAIZER_PREFLIGHT_SKIP_NETWORK", raising=False)

        def boom(*a, **k):
            raise OSError("no route to host")

        with patch.object(pre.socket, "create_connection", boom):
            ok, line = pre.check_inngest_cloud_reachable()
        assert ok is False
        assert "FAIL" in line


# ── run_all + main ───────────────────────────────────────────────────


class TestRunAll:
    def test_happy_path(self, pre, all_env_set, monkeypatch):
        # Stub the DB check so it doesn't touch a real engine.
        monkeypatch.setattr(pre, "check_database", lambda: (True, "  PASS  Database: stub"))
        ok, report = pre.run_all()
        assert ok is True
        assert "ALL CHECKS PASSED" in report
        # Every check section labelled
        for section in (
            "[1/5] Required env vars",
            "[2/5] Inngest dev mode",
            "[3/5] V2 feature flag",
            "[4/5] Database",
            "[5/5] Inngest Cloud reachability",
        ):
            assert section in report

    def test_env_missing_yields_overall_fail(self, pre, all_env_set, monkeypatch):
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        monkeypatch.setattr(pre, "check_database", lambda: (True, "  PASS  Database: stub"))
        ok, report = pre.run_all()
        assert ok is False
        assert "ONE OR MORE CHECKS FAILED" in report

    def test_db_failure_yields_overall_fail(self, pre, all_env_set, monkeypatch):
        monkeypatch.setattr(
            pre, "check_database",
            lambda: (False, "  FAIL  Database: simulated outage"),
        )
        ok, report = pre.run_all()
        assert ok is False

    def test_main_exit_zero_on_success(self, pre, all_env_set, monkeypatch, capsys):
        monkeypatch.setattr(pre, "check_database", lambda: (True, "  PASS  Database: stub"))
        code = pre.main()
        out = capsys.readouterr().out
        assert code == 0
        assert "ALL CHECKS PASSED" in out

    def test_main_exit_one_on_failure(self, pre, all_env_set, monkeypatch, capsys):
        monkeypatch.delenv("INNGEST_EVENT_KEY", raising=False)
        monkeypatch.setattr(pre, "check_database", lambda: (True, "  PASS  Database: stub"))
        code = pre.main()
        out = capsys.readouterr().out
        assert code == 1
        assert "ONE OR MORE CHECKS FAILED" in out
