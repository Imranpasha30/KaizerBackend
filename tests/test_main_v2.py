"""Backend tests for V2 UI integration (Step 11).

This file accumulates tests across Step 11 sub-steps:
  11.1 -- PLATFORMS entry + KAIZER_V2_ENABLED filter  (THIS COMMIT)
  11.2 -- /api/v2/stt/providers endpoint
  11.4 -- runner.run_pipeline V2 branch firing Inngest event
  11.6 -- error message tests (only if backend-side helpers added)

Frontend tests are intentionally NOT in this file (D-11.15: defer to
playwright smoke test in backlog item 54). Pure backend assertions
covering the contract V1 sees + the V2 surface we add.

These tests import main.py directly. main.py has heavy startup side
effects (DB engine, env vars, FastAPI app); we import it lazily
inside fixtures so collection of unrelated test files isn't slowed.
"""

from __future__ import annotations

import importlib
import os
import sys
from unittest.mock import MagicMock

import pytest


# ====================================================================== #
# Module reload fixture (env-var changes need main.py state refreshed)   #
# ====================================================================== #


@pytest.fixture
def main_module(monkeypatch):
    """Import (or reimport) main.py with the current env. monkeypatch
    teardown restores env vars, but we also explicitly clear the
    cached module so subsequent tests see fresh state.
    """
    # KAIZER_V2_ENABLED is read inside _v2_enabled() at request time
    # (not at module load), so we don't need to reimport main.py for
    # env changes -- they take effect immediately.
    if "main" in sys.modules:
        return sys.modules["main"]
    return importlib.import_module("main")


# ====================================================================== #
# Step 11.1: PLATFORMS dict shape                                         #
# ====================================================================== #


class TestPlatformsDictShape:
    def test_full_video_shorts_v2_entry_present(self, main_module):
        assert "full_video_shorts_v2" in main_module.PLATFORMS

    def test_v2_entry_has_expected_keys(self, main_module):
        entry = main_module.PLATFORMS["full_video_shorts_v2"]
        assert entry["label"] == "Full Video + Shorts (V2 Beta)"
        assert entry["width"] == 1080
        assert entry["height"] == 1920

    def test_v1_entries_unchanged(self, main_module):
        # Regression guard: the 4 V1 entries must remain byte-identical
        # so the V1 UI path is unaffected.
        platforms = main_module.PLATFORMS
        assert platforms["instagram_reel"] == {
            "label": "Instagram Reel", "width": 1080, "height": 1920,
        }
        assert platforms["youtube_short"] == {
            "label": "YouTube Short", "width": 1080, "height": 1920,
        }
        assert platforms["youtube_full"] == {
            "label": "YouTube Full", "width": 1920, "height": 1080,
        }
        # The compound platform retains its markers (THIS is what
        # create_job reads to fan out; modifying it would break V1).
        compound = platforms["youtube_full_plus_shorts"]
        assert compound["compound"] is True
        assert compound["expands_to"] == ["youtube_full", "youtube_short"]

    def test_total_entry_count(self, main_module):
        # 4 pre-existing + 1 new V2 entry = 5 total.
        assert len(main_module.PLATFORMS) == 5


# ====================================================================== #
# Step 11.1: CRITICAL correctness — V2 has NO compound markers           #
# ====================================================================== #


class TestV2NoCompoundMarkers:
    """The most important correctness assertion in Step 11.

    Per the user's locked flag #2: if the V2 platform entry had
    ``compound: True`` or ``expands_to: [...]``, create_job would
    fan out into TWO sibling Job rows. The Inngest worker would
    pick up only one event (idempotency key = ``f"job-{job_id}"``),
    so the OTHER Job row would hang in 'pending' forever -- the user
    would see a stuck job in their dashboard with no error, no
    progress, no cancel-from-Inngest path.

    These tests pin that V2's entry has NO compound markers. A
    future refactor accidentally adding them fails this suite
    immediately, before the bug ships to production.
    """

    def test_v2_entry_has_no_compound_key(self, main_module):
        entry = main_module.PLATFORMS["full_video_shorts_v2"]
        assert "compound" not in entry, (
            "V2 entry MUST NOT have a 'compound' key. See test "
            "docstring -- compound markers cause create_job to "
            "fan out into sibling Job rows; the Inngest worker "
            "only picks up one, leaving the other stuck in pending."
        )

    def test_v2_entry_has_no_expands_to_key(self, main_module):
        entry = main_module.PLATFORMS["full_video_shorts_v2"]
        assert "expands_to" not in entry, (
            "V2 entry MUST NOT have an 'expands_to' key. See "
            "test_v2_entry_has_no_compound_key docstring for the "
            "failure mode this prevents."
        )

    def test_only_v1_compound_platform_has_markers(self, main_module):
        # Inverted check: ONLY youtube_full_plus_shorts (the V1
        # compound platform) is allowed to have these markers.
        # Every other entry must NOT have them.
        compound_entries = [
            k for k, v in main_module.PLATFORMS.items()
            if v.get("compound") or v.get("expands_to")
        ]
        assert compound_entries == ["youtube_full_plus_shorts"], (
            f"Only the V1 compound platform is allowed to have "
            f"compound/expands_to markers. Found markers on: "
            f"{compound_entries}"
        )


# ====================================================================== #
# Step 11.1: KAIZER_V2_ENABLED feature flag                               #
# ====================================================================== #


class TestV2EnabledFeatureFlag:
    def test_v2_enabled_default_true(self, monkeypatch, main_module):
        # D-11.12: default ON for Beta.
        monkeypatch.delenv("KAIZER_V2_ENABLED", raising=False)
        assert main_module._v2_enabled() is True

    @pytest.mark.parametrize("truthy", ["1", "true", "True", "TRUE", "yes", "on"])
    def test_v2_enabled_truthy_values(self, monkeypatch, main_module, truthy):
        monkeypatch.setenv("KAIZER_V2_ENABLED", truthy)
        assert main_module._v2_enabled() is True

    @pytest.mark.parametrize("falsy", ["0", "false", "False", "no", "off", ""])
    def test_v2_enabled_falsy_values(self, monkeypatch, main_module, falsy):
        monkeypatch.setenv("KAIZER_V2_ENABLED", falsy)
        assert main_module._v2_enabled() is False


# ====================================================================== #
# Step 11.1: /api/platforms/ endpoint includes/excludes V2 per flag      #
# ====================================================================== #


class TestPlatformsEndpointFilter:
    """The endpoint function itself (get_platforms) is sync + has no
    auth dependency; we call it directly to verify the filter logic
    without needing a TestClient.
    """

    def test_endpoint_includes_v2_when_flag_on(
        self, monkeypatch, main_module,
    ):
        monkeypatch.setenv("KAIZER_V2_ENABLED", "1")
        result = main_module.get_platforms()
        assert "full_video_shorts_v2" in result
        # All 4 V1 entries also present
        assert "instagram_reel" in result
        assert "youtube_short" in result
        assert "youtube_full" in result
        assert "youtube_full_plus_shorts" in result

    def test_endpoint_excludes_v2_when_flag_off(
        self, monkeypatch, main_module,
    ):
        monkeypatch.setenv("KAIZER_V2_ENABLED", "0")
        result = main_module.get_platforms()
        assert "full_video_shorts_v2" not in result
        # V1 entries unaffected
        assert "instagram_reel" in result
        assert "youtube_short" in result
        assert "youtube_full" in result
        assert "youtube_full_plus_shorts" in result
        # Exactly 4 entries when V2 is off
        assert len(result) == 4

    def test_endpoint_default_includes_v2(
        self, monkeypatch, main_module,
    ):
        monkeypatch.delenv("KAIZER_V2_ENABLED", raising=False)
        result = main_module.get_platforms()
        assert "full_video_shorts_v2" in result
        assert len(result) == 5

    def test_endpoint_does_not_mutate_PLATFORMS(
        self, monkeypatch, main_module,
    ):
        # The filter returns a NEW dict when V2 is disabled; mutating
        # the response must NOT affect the module-level PLATFORMS.
        monkeypatch.setenv("KAIZER_V2_ENABLED", "0")
        result = main_module.get_platforms()
        result["test_key"] = {"label": "should not persist"}
        # PLATFORMS still has the original 5 keys
        assert len(main_module.PLATFORMS) == 5
        assert "test_key" not in main_module.PLATFORMS


# ====================================================================== #
# Step 11.2: /api/v2/stt/providers endpoint                               #
# ====================================================================== #


class TestV2STTProvidersEndpoint:
    def test_returns_all_three_providers(self, monkeypatch, main_module):
        # Clear all API key env vars to verify all 3 still surface
        # (with configured=false).
        for k in ("GROQ_API_KEY", "DEEPGRAM_API_KEY", "ASSEMBLYAI_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        result = main_module.get_v2_stt_providers()
        assert len(result) == 3
        ids = {p["id"] for p in result}
        assert ids == {"whisper-groq", "deepgram", "assemblyai"}

    def test_response_shape(self, monkeypatch, main_module):
        for k in ("GROQ_API_KEY", "DEEPGRAM_API_KEY", "ASSEMBLYAI_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        result = main_module.get_v2_stt_providers()
        expected_keys = {
            "id", "display_name", "tier",
            "cost_per_min_usd", "configured", "description",
            # Step 12.5: per-provider warnings list (item 59).
            "warnings",
        }
        for provider in result:
            assert set(provider.keys()) == expected_keys

    def test_no_internal_api_key_env_leaked(self, monkeypatch, main_module):
        # Internal _api_key_env field MUST NOT leak to the response.
        for k in ("GROQ_API_KEY", "DEEPGRAM_API_KEY", "ASSEMBLYAI_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        result = main_module.get_v2_stt_providers()
        for provider in result:
            assert "_api_key_env" not in provider

    def test_configured_false_when_api_key_unset(
        self, monkeypatch, main_module,
    ):
        for k in ("GROQ_API_KEY", "DEEPGRAM_API_KEY", "ASSEMBLYAI_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        result = main_module.get_v2_stt_providers()
        for provider in result:
            assert provider["configured"] is False, (
                f"{provider['id']} should be configured=False when "
                f"its API key env is unset"
            )

    def test_configured_true_when_api_key_set(
        self, monkeypatch, main_module,
    ):
        monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
        monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test-key")
        monkeypatch.setenv("ASSEMBLYAI_API_KEY", "aai-test-key")
        result = main_module.get_v2_stt_providers()
        by_id = {p["id"]: p for p in result}
        assert by_id["whisper-groq"]["configured"] is True
        assert by_id["deepgram"]["configured"] is True
        assert by_id["assemblyai"]["configured"] is True

    def test_configured_per_provider_independent(
        self, monkeypatch, main_module,
    ):
        # User-required test: when DEEPGRAM_API_KEY is unset, deepgram
        # returns configured=false; when set, configured=true. Other
        # providers' configured state is independent.
        monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
        result = main_module.get_v2_stt_providers()
        by_id = {p["id"]: p for p in result}
        assert by_id["whisper-groq"]["configured"] is True
        assert by_id["deepgram"]["configured"] is False
        assert by_id["assemblyai"]["configured"] is False

        # Now flip just deepgram on
        monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test-key")
        result = main_module.get_v2_stt_providers()
        by_id = {p["id"]: p for p in result}
        assert by_id["whisper-groq"]["configured"] is True
        assert by_id["deepgram"]["configured"] is True
        assert by_id["assemblyai"]["configured"] is False

    def test_empty_string_api_key_is_unconfigured(
        self, monkeypatch, main_module,
    ):
        # Edge case: API key env var present but empty string.
        # Operators sometimes "unset" by setting to "" in their env
        # file. Must NOT count as configured.
        monkeypatch.setenv("DEEPGRAM_API_KEY", "")
        monkeypatch.setenv("GROQ_API_KEY", "   ")    # whitespace only
        monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
        result = main_module.get_v2_stt_providers()
        by_id = {p["id"]: p for p in result}
        assert by_id["whisper-groq"]["configured"] is False
        assert by_id["deepgram"]["configured"] is False

    def test_tier_values(self, monkeypatch, main_module):
        for k in ("GROQ_API_KEY", "DEEPGRAM_API_KEY", "ASSEMBLYAI_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        result = main_module.get_v2_stt_providers()
        by_id = {p["id"]: p for p in result}
        assert by_id["whisper-groq"]["tier"] == "free"
        assert by_id["deepgram"]["tier"] == "premium"
        assert by_id["assemblyai"]["tier"] == "mid"

    def test_cost_per_min_values_match_locked_pricing(
        self, monkeypatch, main_module,
    ):
        # Pin pricing to Step 6 research-locked values (re-check
        # quarterly per backlog).
        for k in ("GROQ_API_KEY", "DEEPGRAM_API_KEY", "ASSEMBLYAI_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        result = main_module.get_v2_stt_providers()
        by_id = {p["id"]: p for p in result}
        assert by_id["whisper-groq"]["cost_per_min_usd"] == 0.0
        assert by_id["deepgram"]["cost_per_min_usd"] == 0.0097
        assert by_id["assemblyai"]["cost_per_min_usd"] == 0.0070

    # ---- Step 12.5 / backlog 59: warnings field ---------------------

    def test_v2_stt_providers_includes_warnings_for_whisper_groq(
        self, monkeypatch, main_module,
    ):
        # The whisper-groq entry surfaces the empirical
        # Indian-language timestamp issue (Step 12.2a Path 2
        # investigation, backlog 57). The frontend wizard reads
        # this list and shows it as a warning hint when the user
        # selects whisper-groq.
        for k in ("GROQ_API_KEY", "DEEPGRAM_API_KEY", "ASSEMBLYAI_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        result = main_module.get_v2_stt_providers()
        by_id = {p["id"]: p for p in result}
        groq_warnings = by_id["whisper-groq"]["warnings"]
        assert isinstance(groq_warnings, list)
        assert len(groq_warnings) >= 1, (
            "whisper-groq must have at least one warning string "
            "documenting the Indian-language timestamp issue."
        )
        # The warning text MUST mention 'Telugu' or 'Hindi' (or
        # both) so the UI rendering reads naturally.
        combined = " ".join(groq_warnings).lower()
        assert "telugu" in combined or "hindi" in combined, (
            f"whisper-groq warning must reference Telugu/Hindi; "
            f"got: {groq_warnings!r}"
        )

    def test_v2_stt_providers_other_providers_have_empty_warnings(
        self, monkeypatch, main_module,
    ):
        # Deepgram + AssemblyAI have no known issues today and
        # must return ``warnings: []`` (NOT missing the key).
        for k in ("GROQ_API_KEY", "DEEPGRAM_API_KEY", "ASSEMBLYAI_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        result = main_module.get_v2_stt_providers()
        by_id = {p["id"]: p for p in result}
        for provider_id in ("deepgram", "assemblyai"):
            warnings = by_id[provider_id]["warnings"]
            assert warnings == [], (
                f"{provider_id} should have empty warnings list "
                f"(no known issues); got {warnings!r}"
            )

    def test_display_names_human_readable(self, monkeypatch, main_module):
        # Pin display names so UI design doesn't silently drift.
        for k in ("GROQ_API_KEY", "DEEPGRAM_API_KEY", "ASSEMBLYAI_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        result = main_module.get_v2_stt_providers()
        by_id = {p["id"]: p for p in result}
        assert by_id["whisper-groq"]["display_name"] == "Whisper (Groq)"
        assert by_id["deepgram"]["display_name"] == "Deepgram Nova-3"
        assert by_id["assemblyai"]["display_name"] == "AssemblyAI Universal-2"

    def test_descriptions_present_and_nonempty(
        self, monkeypatch, main_module,
    ):
        for k in ("GROQ_API_KEY", "DEEPGRAM_API_KEY", "ASSEMBLYAI_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        result = main_module.get_v2_stt_providers()
        for provider in result:
            assert provider["description"]
            assert len(provider["description"]) > 20  # non-trivial copy


# ====================================================================== #
# Step 11.4: runner.run_pipeline V2 branch                                #
# ====================================================================== #


class TestRunPipelineV2Branch:
    """When platform == "full_video_shorts_v2", runner.run_pipeline
    MUST fire an Inngest event and return WITHOUT spawning the V1
    subprocess. The V1 4 platforms must still hit the subprocess path
    (regression guard).
    """

    def _import_runner(self):
        # Lazy import to avoid pulling DB engine into module collection.
        import importlib
        if "runner" in sys.modules:
            return sys.modules["runner"]
        return importlib.import_module("runner")

    def test_v2_branch_calls_inngest_no_subprocess(self, monkeypatch):
        runner = self._import_runner()
        # Stub the V2 dispatcher to capture its call args.
        captured: dict = {}

        def _capture(**kw):
            captured.update(kw)
        monkeypatch.setattr(runner, "_dispatch_v2_inngest_event", _capture)

        # Stub the subprocess path -- it must NEVER be called for V2.
        subprocess_called = [False]
        original_popen = runner.subprocess.Popen

        def _popen_should_not_be_called(*args, **kw):
            subprocess_called[0] = True
            raise AssertionError(
                "V2 platform must NOT spawn the V1 subprocess; "
                "Inngest worker takes over."
            )
        monkeypatch.setattr(runner.subprocess, "Popen", _popen_should_not_be_called)

        runner.run_pipeline(
            job_id=42,
            video_path="/abs/v.mp4",
            platform="full_video_shorts_v2",
            frame="torn_card",
            language="te",
            stt_provider="deepgram",
            db_session_factory=lambda: None,
        )

        assert captured["job_id"] == 42
        assert captured["video_path"] == "/abs/v.mp4"
        assert captured["platform"] == "full_video_shorts_v2"
        assert captured["stt_provider"] == "deepgram"
        # Verify subprocess was NOT touched
        assert subprocess_called[0] is False

    def test_v1_platforms_still_spawn_subprocess(self, monkeypatch):
        # Regression guard: the 4 pre-existing platforms must still
        # use the subprocess path. We don't run the whole subprocess
        # flow -- just verify the V2 branch doesn't accidentally
        # catch a V1 platform.
        runner = self._import_runner()
        v2_called = [False]

        def _v2_should_not_be_called(**kw):
            v2_called[0] = True
            raise AssertionError(
                "V1 platforms must NOT route through the V2 Inngest path."
            )
        monkeypatch.setattr(
            runner, "_dispatch_v2_inngest_event", _v2_should_not_be_called,
        )

        # The V1 path proceeds to subprocess spawn; we can't easily
        # mock the whole flow without ffmpeg + DB. Instead we just
        # verify the V2 dispatcher is NOT called for V1 platforms.
        # The actual subprocess invocation has integration coverage
        # elsewhere.
        #
        # To exercise just the early-return check, we use a fake
        # threading.Thread that does nothing.
        class _FakeThread:
            def __init__(self, *a, **kw): pass
            def start(self): pass
        monkeypatch.setattr(runner.threading, "Thread", _FakeThread)

        for v1_platform in (
            "instagram_reel",
            "youtube_short",
            "youtube_full",
            "youtube_full_plus_shorts",
        ):
            runner.run_pipeline(
                job_id=1,
                video_path="/abs/v.mp4",
                platform=v1_platform,
                frame="torn_card",
                language="te",
                db_session_factory=lambda: None,
            )

        # V2 dispatcher was never called for any of the 4 V1 platforms
        assert v2_called[0] is False

    def test_v2_dispatcher_sends_event_with_idempotency_key(self, monkeypatch):
        # Exercise the _dispatch_v2_inngest_event helper directly.
        runner = self._import_runner()

        # Mock the Inngest client's send_sync to capture the event.
        captured_events: list = []

        class _FakeClient:
            def send_sync(self, *, events):
                captured_events.append(events)

        # Stub the inngest_client.get_client() lookup
        import types
        fake_module = types.ModuleType("pipeline_v2.inngest_client")
        fake_module.get_client = lambda: _FakeClient()
        monkeypatch.setitem(
            sys.modules, "pipeline_v2.inngest_client", fake_module,
        )

        # Stub the DB write
        fake_session = MagicMock()
        fake_session.query.return_value.filter.return_value.update.return_value = 1
        db_factory = lambda: fake_session

        runner._dispatch_v2_inngest_event(
            job_id=99,
            video_path="/abs/v.mp4",
            language="te",
            platform="full_video_shorts_v2",
            frame="torn_card",
            stt_provider="whisper-groq",
            db_session_factory=db_factory,
        )

        assert len(captured_events) == 1
        event = captured_events[0]
        # Event has the expected name + data shape
        assert event.name == "video/v2/uploaded"
        # Idempotency key (D-10.10)
        assert event.id == "job-99"
        # Event data carries the required fields
        assert event.data["job_id"] == 99
        assert event.data["video_path"] == "/abs/v.mp4"
        assert event.data["platform"] == "full_video_shorts_v2"
        assert event.data["stt_provider"] == "whisper-groq"
        assert event.data["language"] == "te"
        assert event.data["frame_layout"] == "torn_card"
        # Preset shape
        assert event.data["preset"]["width"] == 1080
        assert event.data["preset"]["height"] == 1920


# ====================================================================== #
# Step 12.2b: V2 Inngest serve mount                                      #
# ====================================================================== #


class TestV2InngestServeMount:
    """The Inngest Dev Server (and prod Inngest Cloud) discover the V2
    function by polling ``/api/inngest`` on the host FastAPI app.
    ``main.py`` mounts that handler via ``register_v2_inngest`` at the
    very end of module load, guarded by ``KAIZER_V2_ENABLED``.

    These tests use route-inspection (``app.routes``) so they don't
    require an HTTP server -- the registration happens at module
    import time, so reloading ``main`` under different env vars is
    the right shape for the test.
    """

    @staticmethod
    def _reload_main_fresh():
        # Drop both main AND the inngest_app idempotency guard so the
        # mount happens fresh on the next import. Without resetting
        # the guard, a prior test's import would block the mount.
        if "main" in sys.modules:
            del sys.modules["main"]
        # Lazy import so a missing pipeline_v2 path doesn't trip
        # collection; register_v2_inngest's module is normally added
        # to sys.path by main.py's V2 block. Direct import works here
        # because conftest already arranged the path.
        try:
            from pipeline_v2.inngest_app import _reset_for_tests
            _reset_for_tests()
        except ImportError:
            # If pipeline_v2 isn't on sys.path yet, the next main
            # import will add it and the guard starts False naturally.
            pass
        return importlib.import_module("main")

    def test_inngest_app_module_imports_cleanly(self):
        # Smoke import: the module loads + register_v2_inngest is
        # callable. Catches dependency-missing regressions before
        # the route-inspection tests run.
        from pipeline_v2.inngest_app import register_v2_inngest
        assert callable(register_v2_inngest)

    def test_v2_enabled_mounts_inngest_route(self, monkeypatch):
        # With KAIZER_V2_ENABLED=1, main.py module-load mounts the
        # ``/api/inngest`` serve endpoint via inngest.fast_api.serve.
        # The Inngest SDK registers three HTTP methods on that path
        # (GET for function discovery, POST for invocation, PUT for
        # sync) -- we assert at least one of them appears.
        monkeypatch.setenv("KAIZER_V2_ENABLED", "1")
        m = self._reload_main_fresh()
        inngest_paths = [
            r.path for r in m.app.routes
            if hasattr(r, "path") and r.path == "/api/inngest"
        ]
        assert len(inngest_paths) >= 1, (
            f"expected /api/inngest mounted when V2 enabled, "
            f"got {len(inngest_paths)} matching routes"
        )
        # Inngest SDK registers exactly 3 HTTP methods on the path.
        methods = set()
        for r in m.app.routes:
            if hasattr(r, "path") and r.path == "/api/inngest":
                methods.update(getattr(r, "methods", set()) or set())
        assert {"GET", "POST", "PUT"}.issubset(methods), (
            f"expected GET+POST+PUT on /api/inngest, got {methods}"
        )

    def test_v2_disabled_does_not_mount_inngest_route(self, monkeypatch):
        # With KAIZER_V2_ENABLED=0, the V2 block is skipped and the
        # /api/inngest route does NOT mount. The 4 V1 platforms ship
        # with a byte-identical route table to pre-V2.
        monkeypatch.setenv("KAIZER_V2_ENABLED", "0")
        m = self._reload_main_fresh()
        inngest_paths = [
            r.path for r in m.app.routes
            if hasattr(r, "path") and "/inngest" in r.path.lower()
        ]
        assert len(inngest_paths) == 0, (
            f"expected NO inngest routes when V2 disabled, "
            f"got {inngest_paths}"
        )

    def test_process_video_v2_signature_compatible_with_inngest_sdk(self):
        """Inngest SDK 0.5.18 calls ``handler(ctx)`` -- a SINGLE
        positional argument. The handler must accept exactly one
        parameter; ``step`` is reached via ``ctx.step``.

        Pre-Step-12.2b the orchestrator was declared
        ``async def process_video_v2(ctx, step)`` -- which Inngest
        invoked as ``handler(ctx)`` and failed with
        ``TypeError: process_video_v2() missing 1 required positional
        argument: 'step'``. That bug cost 4 failed E2E runs to
        surface (none of which exercise this signature at unit-test
        time today). This test locks the contract so future Inngest
        SDK bumps that change the calling convention fail HERE
        instead of 76 seconds into a real Inngest dev run.

        See backlog item 72 for the signature evolution.
        """
        import inspect
        from pipeline_v2.orchestrator import process_video_v2

        # The @inngest.create_function decorator wraps the original
        # async def into an inngest.Function object. The underlying
        # handler is stored on ``_handler``.
        underlying = getattr(process_video_v2, "_handler", None)
        assert underlying is not None, (
            "process_video_v2._handler missing -- the @create_function "
            "decorator's API changed; update this test to find the "
            "new attribute name."
        )

        params = list(inspect.signature(underlying).parameters)
        assert params == ["ctx"], (
            f"process_video_v2 must accept exactly one positional "
            f"argument 'ctx' (Inngest SDK 0.5.18 calling convention); "
            f"got parameters={params}. Backlog item 72."
        )
        # Defensive: 'step' must NOT reappear in the signature --
        # that was the failure mode that took 4 E2E runs to diagnose.
        assert "step" not in params, (
            "process_video_v2 must not have a 'step' parameter; "
            "access step via ctx.step. See backlog item 72."
        )

    def test_orchestrator_terminal_catch_does_not_intercept_baseexception(
        self,
    ):
        """V2 orchestrator's outer terminal-failure try/except MUST catch
        ``Exception``, not ``BaseException``. Inngest SDK 0.5.18 uses
        ``BaseException``-subclassed flow-control exceptions
        (``ResponseInterrupt``, ``SkipInterrupt``, ``NestedStepInterrupt``)
        that are raised after every step.run() yield to signal step
        completion. These MUST propagate to the SDK executor uncaught.
        Catching ``BaseException`` intercepts them and falsely marks the
        Job as failed even though Inngest sees the run as completed --
        leading to the "Inngest dashboard COMPLETED but DB shows
        Job.status='failed'" symptom diagnosed in Step 12.2b run #5.

        AST-level check (read the source text + grep) rather than a
        runtime check because the actual ResponseInterrupt flow is hard
        to mock cleanly. Intent: "you shouldn't catch BaseException
        in the orchestrator's try-wrapper, period."

        See backlog item 74.
        """
        from pathlib import Path
        orch_path = (
            Path(__file__).resolve().parent.parent
            / "pipeline_v2" / "pipeline_v2" / "orchestrator.py"
        )
        assert orch_path.is_file(), f"orchestrator.py not found at {orch_path}"
        src = orch_path.read_text(encoding="utf-8")

        # Locate the process_video_v2 function body. We only want to
        # assert about the wrapper INSIDE that function -- other
        # helpers in the file may legitimately catch BaseException.
        marker = "async def process_video_v2"
        start = src.find(marker)
        assert start >= 0, "process_video_v2 not found in orchestrator.py"
        # Take a slice covering the function body (next function or EOF).
        next_def = src.find("\nasync def ", start + len(marker))
        if next_def == -1:
            next_def = src.find("\ndef ", start + len(marker))
        if next_def == -1:
            next_def = len(src)
        body = src[start:next_def]

        # The wrapper must NOT catch BaseException, and MUST catch
        # Exception. Both checks together lock the contract.
        assert "except BaseException" not in body, (
            "process_video_v2 must not catch BaseException -- Inngest "
            "SDK flow-control exceptions (ResponseInterrupt, etc.) "
            "subclass BaseException and must propagate uncaught. "
            "See backlog item 74."
        )
        assert "except Exception" in body, (
            "process_video_v2 must have a terminal `except Exception` "
            "wrapper that calls _mark_job_failed on real failures. "
            "See backlog item 74."
        )

    def test_inngest_sdk_flow_control_exceptions_subclass_baseexception(
        self,
    ):
        """Document the SDK invariant the orchestrator's
        ``except Exception`` choice depends on. If Inngest ever changes
        these to subclass ``Exception`` instead, this test fails and
        flags the orchestrator's catch as eligible for re-evaluation
        (the current `except Exception` would still be correct but
        the BaseException-avoidance rationale would no longer apply).

        See backlog item 74.
        """
        from inngest._internal.step_lib.base import (
            ResponseInterrupt,
            SkipInterrupt,
            NestedStepInterrupt,
        )
        for cls in (ResponseInterrupt, SkipInterrupt, NestedStepInterrupt):
            assert issubclass(cls, BaseException), (
                f"{cls.__name__} should subclass BaseException"
            )
            assert not issubclass(cls, Exception), (
                f"{cls.__name__} unexpectedly subclasses Exception. "
                f"This means Inngest changed the contract; re-evaluate "
                f"backlog item 74's empirical assumption."
            )



