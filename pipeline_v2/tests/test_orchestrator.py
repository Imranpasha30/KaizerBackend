"""Unit tests for the V2 Inngest orchestrator (Step 10).

This file accumulates tests across Step 10 sub-steps:
  10.1 -- skeleton: imports, stage name constants, client singleton,
          Function registration  (THIS COMMIT)
  10.2 -- Stage 0-2.5 wiring + retry policy + PermanentSTTError
  10.3 -- Stage 3 fanout + Stage 4 render + subprocess registration
  10.4 -- DB writes + cancel check + cost ledger
  10.5 -- end-to-end mocked test

Inngest 0.5.18 doesn't ship a heavy test harness; we exercise the
function shape directly + call the handler coroutines as plain async
functions for unit-level coverage. The E2E test in 10.5 verifies
sequence + state-passing under a mock Inngest ``Step``.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest

from pipeline_v2 import orchestrator
from pipeline_v2.orchestrator import (
    ALL_STAGES,
    FINALIZE,
    STAGE_0_INGEST,
    STAGE_1_TRANSCRIBE,
    STAGE_2_5_ENTITIES,
    STAGE_2_CONTINUITY,
    STAGE_3_FANOUT,
    STAGE_4_RENDER,
    process_video_v2,
)
from pipeline_v2.inngest_client import (
    _is_dev_mode,
    get_client,
    reset_client_for_tests,
)


# ====================================================================== #
# Stage name constants                                                    #
# ====================================================================== #


class TestStageNameConstants:
    """Stage names MUST match Job.current_stage values per D-10.7. If
    the constants drift from the DB migration's expected values, the
    UI's progress lookup breaks silently.
    """

    def test_all_stages_tuple_has_7_entries(self):
        assert len(ALL_STAGES) == 7

    def test_all_stages_order(self):
        # Order locked by D-10.2: 6 stage steps + 1 finalize step
        assert ALL_STAGES == (
            "stage_0_ingest",
            "stage_1_transcribe",
            "stage_2_continuity",
            "stage_2_5_entities",
            "stage_3_fanout",
            "stage_4_render",
            "finalize",
        )

    def test_constants_match_migration_values(self):
        # docs/MIGRATIONS.md Phase 13 lists these exact strings.
        # If the constants here change, the migration values must
        # change too -- pin via this test.
        expected = {
            "stage_0_ingest", "stage_1_transcribe",
            "stage_2_continuity", "stage_2_5_entities",
            "stage_3_fanout", "stage_4_render", "finalize",
        }
        assert set(ALL_STAGES) == expected

    def test_constants_max_length_under_varchar_40(self):
        # The DB column is VARCHAR(40). Longest constant must fit.
        for name in ALL_STAGES:
            assert len(name) <= 40, f"{name!r} exceeds VARCHAR(40)"


# ====================================================================== #
# Inngest client singleton                                                #
# ====================================================================== #


class TestInngestClient:
    def setup_method(self):
        reset_client_for_tests()

    def teardown_method(self):
        reset_client_for_tests()

    def test_get_client_returns_inngest_instance(self):
        client = get_client()
        # 0.5.18 Inngest class path
        assert type(client).__name__ == "Inngest"

    def test_get_client_caches_singleton(self):
        a = get_client()
        b = get_client()
        assert a is b

    def test_reset_client_drops_singleton(self):
        a = get_client()
        reset_client_for_tests()
        b = get_client()
        assert a is not b

    def test_dev_mode_default_when_no_keys(self, monkeypatch):
        monkeypatch.delenv("INNGEST_DEV", raising=False)
        monkeypatch.delenv("INNGEST_EVENT_KEY", raising=False)
        monkeypatch.delenv("INNGEST_SIGNING_KEY", raising=False)
        assert _is_dev_mode() is True

    def test_dev_mode_explicit_via_env(self, monkeypatch):
        monkeypatch.setenv("INNGEST_DEV", "1")
        # Keys set: dev flag wins
        monkeypatch.setenv("INNGEST_EVENT_KEY", "evt")
        monkeypatch.setenv("INNGEST_SIGNING_KEY", "sig")
        assert _is_dev_mode() is True

    def test_dev_mode_false_when_both_keys_set(self, monkeypatch):
        monkeypatch.delenv("INNGEST_DEV", raising=False)
        monkeypatch.setenv("INNGEST_EVENT_KEY", "evt")
        monkeypatch.setenv("INNGEST_SIGNING_KEY", "sig")
        assert _is_dev_mode() is False

    def test_app_id_defaults_to_kaizer_v2(self, monkeypatch):
        monkeypatch.delenv("INNGEST_APP_ID", raising=False)
        reset_client_for_tests()
        client = get_client()
        # Inngest stores app_id on the instance
        assert client.app_id == "kaizer-v2"

    def test_app_id_overridable_via_env(self, monkeypatch):
        monkeypatch.setenv("INNGEST_APP_ID", "kaizer-staging")
        reset_client_for_tests()
        client = get_client()
        assert client.app_id == "kaizer-staging"


# ====================================================================== #
# process_video_v2 Function registration                                  #
# ====================================================================== #


class TestProcessVideoV2Registration:
    def test_is_inngest_function(self):
        # @client.create_function returns a Function instance
        assert type(process_video_v2).__name__ == "Function"

    def test_has_expected_fn_id(self):
        # Inngest 0.5.18 prefixes the fn_id with the client's app_id
        # so the public id is "<app_id>-<fn_id>". With INNGEST_APP_ID
        # defaulting to "kaizer-v2", the final id is
        # "kaizer-v2-process-video-v2".
        assert process_video_v2.id == "kaizer-v2-process-video-v2"


# ====================================================================== #
# Remaining skeleton handlers (Step 9.3 - 9.4 wire these)                 #
# ====================================================================== #


# (Step 10.4 wired finalize; the obsolete skeleton passthrough class
# was removed once every handler had real bodies. Finalize tests live
# in TestFinalizeHandler below.)


# ====================================================================== #
# DB migration parity (locked by D-10.7)                                  #
# ====================================================================== #


class TestDbMigrationParity:
    """The Job.current_stage column added in main.py:_migrate_schema()
    must accept every value the orchestrator writes. We can't easily
    run the migration in this test, but we can pin:
      - the constants are all <= VARCHAR(40)
      - the model exposes the column
    """

    def test_job_model_has_current_stage_column(self):
        # Import V1's models (KaizerBackend on sys.path via conftest)
        import models as v1_models
        cols = {c.name for c in v1_models.Job.__table__.columns}
        assert "current_stage" in cols

    def test_current_stage_column_is_varchar_40_nullable(self):
        import models as v1_models
        col = v1_models.Job.__table__.columns["current_stage"]
        # SQLAlchemy String(40) -> length 40
        assert col.type.length == 40
        assert col.nullable is True


# ====================================================================== #
# NonRetriableError import (for D-10.3 PermanentSTTError pattern)         #
# ====================================================================== #


class TestNonRetriableErrorAvailable:
    """D-10.3 refinement: PermanentSTTError -> NonRetriableError so
    Inngest skips the retry burn. Verify the SDK exposes the class.
    """

    def test_non_retriable_error_importable(self):
        from inngest import NonRetriableError
        # Must be a subclass of Exception
        assert issubclass(NonRetriableError, Exception)


# ====================================================================== #
# Step 10.2: envelope helpers                                             #
# ====================================================================== #


class TestEnvelopeInit:
    """_envelope_init builds the per-job state envelope from incoming
    Inngest event data. Tests pin the shape so downstream stages can
    rely on every key being present.
    """

    def test_minimal_event_data(self):
        env = orchestrator._envelope_init({
            "job_id": 7,
            "video_path": "/abs/v.mp4",
            "language": "te",
            "platform": "full_video_shorts_v2",
            "frame_layout": "torn_card",
            "preset": {"width": 1080, "height": 1920},
        })
        assert env["job_id"] == 7
        assert env["video_path"] == "/abs/v.mp4"
        assert env["language"] == "te"
        assert env["platform"] == "full_video_shorts_v2"
        assert env["frame_layout"] == "torn_card"
        assert env["preset"] == {"width": 1080, "height": 1920}
        # All per-stage slots start None
        for k in ("stage_0", "stage_1", "stage_2", "stage_2_5",
                  "stage_3", "stage_4"):
            assert env[k] is None
        assert env["stage_costs"] == {}

    def test_missing_optional_fields_get_defaults(self):
        env = orchestrator._envelope_init({"job_id": 1})
        assert env["job_id"] == 1
        assert env["video_path"] == ""
        assert env["language"] == "te"
        assert env["platform"] == "full_video_shorts_v2"
        assert env["frame_layout"] == "torn_card"
        assert env["preset"] == {}
        assert env["user_id"] is None
        assert env["out_dir"] == ""

    def test_job_id_coerced_to_int(self):
        # Inngest event data is JSON-serialized; ints may arrive as
        # strings. The init coerces.
        env = orchestrator._envelope_init({"job_id": "42"})
        assert env["job_id"] == 42
        assert isinstance(env["job_id"], int)


# ====================================================================== #
# Step 10.2: cancel + current_stage helpers                               #
# ====================================================================== #


class TestCheckCancelled:
    """``_check_cancelled`` reads ``Job.cancel_requested`` and raises
    Inngest ``NonRetriableError`` if the flag is set.
    """

    def test_zero_job_id_skips_check(self):
        # Synthetic test job_ids -- helper should no-op without DB.
        orchestrator._check_cancelled(0)
        orchestrator._check_cancelled(-1)
        orchestrator._check_cancelled(None)

    def test_raises_when_cancel_flag_set(self, monkeypatch):
        from inngest import NonRetriableError
        # Fake the DB session + query path.
        fake_job = MagicMock()
        fake_job.cancel_requested = True
        fake_session = MagicMock()
        fake_session.query.return_value.filter.return_value.first.return_value = fake_job

        def _fake_open():
            return fake_session
        monkeypatch.setattr(orchestrator, "_open_db_session", _fake_open)

        with pytest.raises(NonRetriableError, match="cancelled: job 99"):
            orchestrator._check_cancelled(99)
        fake_session.close.assert_called_once()

    def test_passes_when_cancel_flag_unset(self, monkeypatch):
        fake_job = MagicMock()
        fake_job.cancel_requested = False
        fake_session = MagicMock()
        fake_session.query.return_value.filter.return_value.first.return_value = fake_job
        monkeypatch.setattr(
            orchestrator, "_open_db_session", lambda: fake_session,
        )
        # Should not raise
        orchestrator._check_cancelled(99)

    def test_db_failure_is_non_fatal(self, monkeypatch):
        # DB read failure -> log warning + continue (best-effort UX).
        def _bad_open():
            raise RuntimeError("DB connection lost")
        monkeypatch.setattr(orchestrator, "_open_db_session", _bad_open)
        # Should NOT raise -- transient DB error doesn't kill the job
        orchestrator._check_cancelled(99)


class TestWriteCurrentStage:
    """Synchronous fire-and-forget Job.current_stage write."""

    def test_zero_job_id_skips(self):
        orchestrator._write_current_stage(0, "stage_0_ingest")
        orchestrator._write_current_stage(-1, "stage_0_ingest")
        orchestrator._write_current_stage(None, "stage_0_ingest")

    def test_writes_stage_name(self, monkeypatch):
        fake_session = MagicMock()
        monkeypatch.setattr(
            orchestrator, "_open_db_session", lambda: fake_session,
        )
        orchestrator._write_current_stage(7, "stage_2_continuity")
        # Verify the UPDATE was issued
        fake_session.query.return_value.filter.return_value.update.assert_called_once()
        call_args = fake_session.query.return_value.filter.return_value.update.call_args
        assert call_args[0][0]["current_stage"] == "stage_2_continuity"
        fake_session.commit.assert_called_once()
        fake_session.close.assert_called_once()

    def test_db_failure_is_non_fatal(self, monkeypatch):
        def _bad_open():
            raise RuntimeError("DB down")
        monkeypatch.setattr(orchestrator, "_open_db_session", _bad_open)
        # Should NOT raise
        orchestrator._write_current_stage(7, "stage_0_ingest")


# ====================================================================== #
# Step 10.2: Stage 0-2.5 handler wiring (mocked stage calls)              #
# ====================================================================== #


def _stub_stage_helpers(monkeypatch):
    """Bypass _check_cancelled + _write_current_stage so handlers
    don't touch the DB during unit tests.
    """
    monkeypatch.setattr(orchestrator, "_check_cancelled", lambda jid: None)
    monkeypatch.setattr(
        orchestrator, "_write_current_stage", lambda jid, s: None,
    )


class TestStage0Handler:
    """Wires V2.run_stage_0; result lands in envelope['stage_0']."""

    @pytest.mark.asyncio
    async def test_calls_run_stage_0_and_stores_result(self, monkeypatch, tmp_path):
        _stub_stage_helpers(monkeypatch)

        from pipeline_v2.models import Stage0Output
        fake_out = Stage0Output(
            mezzanine_path=str(tmp_path / "mez.mp4"),
            audio_path=str(tmp_path / "audio.mp3"),
            duration_sec=60.0,
            encoder_used="libx264",
            source_was_vfr=False,
            transcode_seconds=5.0,
            audio_extract_seconds=2.0,
            wall_seconds=7.0,
        )

        async def _fake_run_stage_0(src, out, **kw):
            return fake_out
        monkeypatch.setattr(orchestrator, "run_stage_0", _fake_run_stage_0)

        env = orchestrator._envelope_init({
            "job_id": 1, "video_path": "/v.mp4",
            "platform": "full_video_shorts_v2",
            "out_dir": str(tmp_path),
        })
        out = await orchestrator._stage_0_ingest_handler(env)

        assert out["stage_0"]["mezzanine_path"] == fake_out.mezzanine_path
        assert out["stage_0"]["audio_path"] == fake_out.audio_path
        assert out["stage_costs"]["stage_0_ingest"] == 0.0

    @pytest.mark.asyncio
    async def test_default_out_dir_uses_platform_and_job_id(
        self, monkeypatch, tmp_path,
    ):
        _stub_stage_helpers(monkeypatch)
        captured_args = {}

        async def _fake_run_stage_0(src, out, **kw):
            captured_args["out"] = out
            from pipeline_v2.models import Stage0Output
            return Stage0Output(
                mezzanine_path=str(tmp_path / "mez.mp4"),
                audio_path=str(tmp_path / "a.mp3"),
                duration_sec=10.0,
                encoder_used="libx264",
                source_was_vfr=False,
                transcode_seconds=1.0,
                audio_extract_seconds=1.0,
                wall_seconds=2.0,
            )
        monkeypatch.setattr(orchestrator, "run_stage_0", _fake_run_stage_0)

        env = orchestrator._envelope_init({
            "job_id": 42, "video_path": "/v.mp4",
            "platform": "full_video_shorts_v2",
            # out_dir intentionally empty
        })
        out = await orchestrator._stage_0_ingest_handler(env)
        # Default out_dir uses platform + job_id
        assert "full_video_shorts_v2" in captured_args["out"]
        assert "job_42" in captured_args["out"]
        # Envelope's out_dir is updated for downstream stages
        assert out["out_dir"] == captured_args["out"]


class TestStage1Handler:
    """Wires V2.run_stage_1; PermanentSTTError translates to NonRetriableError."""

    @pytest.mark.asyncio
    async def test_calls_run_stage_1_with_env_provider(self, monkeypatch, tmp_path):
        _stub_stage_helpers(monkeypatch)
        monkeypatch.setenv("KAIZER_STT_PROVIDER", "deepgram")

        from pipeline_v2.models import (
            Stage1Output, WordLevelTranscript, Word,
        )
        fake_transcript = WordLevelTranscript(
            words=[Word(w="hello", s=0.0, e=0.5)],
            duration_sec=0.5,
            detected_languages=["te"],
            provider="deepgram",
        )
        fake_stage_1 = Stage1Output(
            transcript=fake_transcript,
            stt_provider="deepgram",
            stt_audio_duration_sec=0.5,
            stt_wall_seconds=1.0,
            stt_cost_usd=0.005,
            stt_word_count=1,
            stt_language_detected="te",
            stt_request_id="req-1",
        )
        captured_args = {}

        async def _fake_run_stage_1(audio_path, *, provider, **kw):
            captured_args["audio"] = audio_path
            captured_args["provider"] = provider
            return fake_stage_1
        monkeypatch.setattr(orchestrator, "run_stage_1", _fake_run_stage_1)

        env = orchestrator._envelope_init({"job_id": 1})
        env["stage_0"] = {
            "mezzanine_path": str(tmp_path / "mez.mp4"),
            "audio_path": str(tmp_path / "a.mp3"),
            "duration_sec": 0.5,
            "encoder_used": "libx264",
            "source_was_vfr": False,
            "transcode_seconds": 1.0,
            "audio_extract_seconds": 1.0,
            "wall_seconds": 2.0,
        }
        out = await orchestrator._stage_1_transcribe_handler(env)

        assert captured_args["audio"] == env["stage_0"]["audio_path"]
        assert captured_args["provider"] == "deepgram"
        assert out["stage_1"]["stt_cost_usd"] == 0.005
        assert out["stage_costs"]["stage_1_transcribe"] == 0.005

    @pytest.mark.asyncio
    async def test_permanent_stt_error_translates_to_non_retriable(
        self, monkeypatch, tmp_path,
    ):
        # D-10.3 refinement: PermanentSTTError -> NonRetriableError.
        _stub_stage_helpers(monkeypatch)
        from pipeline_v2.stages.stt import PermanentSTTError
        from inngest import NonRetriableError

        async def _raises_permanent(audio_path, **kw):
            raise PermanentSTTError("empty_file: audio file is 0 bytes")
        monkeypatch.setattr(orchestrator, "run_stage_1", _raises_permanent)

        env = orchestrator._envelope_init({"job_id": 1})
        env["stage_0"] = {
            "mezzanine_path": str(tmp_path / "m.mp4"),
            "audio_path": str(tmp_path / "a.mp3"),
            "duration_sec": 0.0, "encoder_used": "libx264",
            "source_was_vfr": False, "transcode_seconds": 1.0,
            "audio_extract_seconds": 1.0, "wall_seconds": 2.0,
        }
        with pytest.raises(NonRetriableError, match=r"permanent: empty_file"):
            await orchestrator._stage_1_transcribe_handler(env)

    @pytest.mark.asyncio
    async def test_transient_error_does_not_translate(
        self, monkeypatch, tmp_path,
    ):
        # Non-permanent errors pass through unchanged for Inngest retry.
        _stub_stage_helpers(monkeypatch)
        from inngest import NonRetriableError

        async def _raises_transient(audio_path, **kw):
            raise ConnectionError("network blip")
        monkeypatch.setattr(orchestrator, "run_stage_1", _raises_transient)

        env = orchestrator._envelope_init({"job_id": 1})
        env["stage_0"] = {
            "mezzanine_path": str(tmp_path / "m.mp4"),
            "audio_path": str(tmp_path / "a.mp3"),
            "duration_sec": 0.0, "encoder_used": "libx264",
            "source_was_vfr": False, "transcode_seconds": 1.0,
            "audio_extract_seconds": 1.0, "wall_seconds": 2.0,
        }
        # Transient ConnectionError propagates AS-IS (not as NonRetriableError)
        with pytest.raises(ConnectionError):
            await orchestrator._stage_1_transcribe_handler(env)


class TestStage2Handler:
    @pytest.mark.asyncio
    async def test_calls_continuity_editor(self, monkeypatch, tmp_path):
        _stub_stage_helpers(monkeypatch)

        from pipeline_v2.models import (
            CleanTranscript, FullVideoCut, SkippedSegment, Stage1Output,
            Stage2Output, StageTwoOutput, Word, WordLevelTranscript,
        )

        stage_1 = Stage1Output(
            transcript=WordLevelTranscript(
                words=[Word(w="x", s=0.0, e=0.3)],
                duration_sec=0.3,
                detected_languages=["te"],
                provider="deepgram",
            ),
            stt_provider="deepgram",
            stt_audio_duration_sec=0.3,
            stt_wall_seconds=1.0,
            stt_cost_usd=0.001,
            stt_word_count=1,
            stt_language_detected="te",
            stt_request_id="r",
        )

        fake_decisions = Stage2Output(
            full_video_cuts=[FullVideoCut(
                index=0, start_word_idx=0, end_word_idx=0,
                start_sec=0.0, end_sec=0.3, importance=5,
            )],
            skipped_segments=[],
            retake_audit="none",
        )
        fake_full = StageTwoOutput(
            full_video_cuts=fake_decisions.full_video_cuts,
            skipped_segments=[],
            clean_transcript=CleanTranscript(
                words=[Word(w="x", s=0.0, e=0.3)],
                clip_boundaries={0: (0, 0)},
                source_word_map=[0],
            ),
            retake_audit="none",
        )

        class _FakeEditor:
            async def transcribe_to_decisions(self, s1):
                return fake_decisions
        monkeypatch.setattr(
            orchestrator, "Stage2ContinuityEditor", _FakeEditor,
        )
        monkeypatch.setattr(
            orchestrator, "assemble_stage_two_output",
            lambda s1, dec: fake_full,
        )

        env = orchestrator._envelope_init({"job_id": 1})
        env["stage_1"] = stage_1.model_dump()
        out = await orchestrator._stage_2_continuity_handler(env)

        assert out["stage_2"]["retake_audit"] == "none"
        assert out["stage_costs"]["stage_2_continuity"] == 0.0


class TestStage25Handler:
    @pytest.mark.asyncio
    async def test_calls_canonicalizer(self, monkeypatch):
        _stub_stage_helpers(monkeypatch)

        from pipeline_v2.models import (
            CleanTranscript, Entity, FullVideoCut, Stage2_5Output,
            SkippedSegment, StageTwoOutput, Word,
        )
        clean = CleanTranscript(
            words=[Word(w="x", s=0.0, e=0.3)],
            clip_boundaries={0: (0, 0)},
            source_word_map=[0],
        )
        stage_2 = StageTwoOutput(
            full_video_cuts=[FullVideoCut(
                index=0, start_word_idx=0, end_word_idx=0,
                start_sec=0.0, end_sec=0.3, importance=5,
            )],
            skipped_segments=[],
            clean_transcript=clean,
            retake_audit="none",
        )
        fake_25 = Stage2_5Output(entities=[
            Entity(
                canonical_name="X", native_name="X",
                first_mention_word_idx=0, type="PERSON", mentions=[0],
            ),
        ])

        class _FakeCanonicalizer:
            async def classify(self, ct):
                return fake_25
        monkeypatch.setattr(
            orchestrator, "Stage2_5EntityCanonicalizer", _FakeCanonicalizer,
        )

        env = orchestrator._envelope_init({"job_id": 1})
        env["stage_2"] = stage_2.model_dump()
        out = await orchestrator._stage_2_5_entities_handler(env)

        assert len(out["stage_2_5"]["entities"]) == 1
        assert out["stage_2_5"]["entities"][0]["canonical_name"] == "X"
        assert out["stage_costs"]["stage_2_5_entities"] == 0.0


# ====================================================================== #
# Step 10.2: cancel-check fires inside step handler                       #
# ====================================================================== #


class TestCancelCheckInsideHandler:
    @pytest.mark.asyncio
    async def test_cancel_check_fires_in_stage_0(self, monkeypatch):
        from inngest import NonRetriableError

        # Configure the cancel check to raise (simulating an active
        # Job.cancel_requested=True at step start).
        def _raise_cancel(jid):
            raise NonRetriableError(f"cancelled: job {jid}")
        monkeypatch.setattr(orchestrator, "_check_cancelled", _raise_cancel)
        monkeypatch.setattr(
            orchestrator, "_write_current_stage", lambda jid, s: None,
        )

        env = orchestrator._envelope_init({"job_id": 99})
        with pytest.raises(NonRetriableError, match=r"cancelled: job 99"):
            await orchestrator._stage_0_ingest_handler(env)


# ====================================================================== #
# Step 10.2: current_stage is written at the start of every handler       #
# ====================================================================== #


class TestCurrentStageWrittenPerHandler:
    @pytest.mark.asyncio
    async def test_each_handler_writes_its_stage_name(self, monkeypatch, tmp_path):
        # Capture every (job_id, stage_name) call to _write_current_stage.
        writes: list[tuple[int, str]] = []
        monkeypatch.setattr(orchestrator, "_check_cancelled", lambda jid: None)
        monkeypatch.setattr(
            orchestrator, "_write_current_stage",
            lambda jid, s: writes.append((jid, s)),
        )

        # Stub all stage calls so handlers don't blow up
        from pipeline_v2.models import (
            CleanTranscript, FullVideoCut, Stage0Output, Stage1Output,
            Stage2Output, Stage2_5Output, StageTwoOutput, Word,
            WordLevelTranscript,
        )

        stage_0 = Stage0Output(
            mezzanine_path=str(tmp_path / "m.mp4"),
            audio_path=str(tmp_path / "a.mp3"),
            duration_sec=0.3, encoder_used="libx264",
            source_was_vfr=False, transcode_seconds=1.0,
            audio_extract_seconds=1.0, wall_seconds=2.0,
        )

        async def _fake_run_stage_0(src, out, **kw):
            return stage_0
        monkeypatch.setattr(orchestrator, "run_stage_0", _fake_run_stage_0)

        stage_1 = Stage1Output(
            transcript=WordLevelTranscript(
                words=[Word(w="x", s=0.0, e=0.3)],
                duration_sec=0.3, detected_languages=["te"],
                provider="deepgram",
            ),
            stt_provider="deepgram", stt_audio_duration_sec=0.3,
            stt_wall_seconds=1.0, stt_cost_usd=0.001,
            stt_word_count=1, stt_language_detected="te",
            stt_request_id="r",
        )

        async def _fake_run_stage_1(audio_path, **kw):
            return stage_1
        monkeypatch.setattr(orchestrator, "run_stage_1", _fake_run_stage_1)

        clean = CleanTranscript(
            words=[Word(w="x", s=0.0, e=0.3)],
            clip_boundaries={0: (0, 0)},
            source_word_map=[0],
        )
        full = StageTwoOutput(
            full_video_cuts=[FullVideoCut(
                index=0, start_word_idx=0, end_word_idx=0,
                start_sec=0.0, end_sec=0.3, importance=5,
            )],
            skipped_segments=[], clean_transcript=clean,
            retake_audit="none",
        )

        class _FakeEditor:
            async def transcribe_to_decisions(self, s1):
                return Stage2Output(
                    full_video_cuts=full.full_video_cuts,
                    skipped_segments=[], retake_audit="none",
                )
        monkeypatch.setattr(
            orchestrator, "Stage2ContinuityEditor", _FakeEditor,
        )
        monkeypatch.setattr(
            orchestrator, "assemble_stage_two_output",
            lambda s1, dec: full,
        )

        class _FakeCanonicalizer:
            async def classify(self, ct):
                return Stage2_5Output(entities=[])
        monkeypatch.setattr(
            orchestrator, "Stage2_5EntityCanonicalizer", _FakeCanonicalizer,
        )

        env = orchestrator._envelope_init({"job_id": 33})
        env = await orchestrator._stage_0_ingest_handler(env)
        env = await orchestrator._stage_1_transcribe_handler(env)
        env = await orchestrator._stage_2_continuity_handler(env)
        env = await orchestrator._stage_2_5_entities_handler(env)

        assert writes == [
            (33, "stage_0_ingest"),
            (33, "stage_1_transcribe"),
            (33, "stage_2_continuity"),
            (33, "stage_2_5_entities"),
        ]


# ====================================================================== #
# Step 10.2: PermanentSTTError surface (provider-side)                    #
# ====================================================================== #


class TestPermanentSTTErrorSurface:
    """Verify PermanentSTTError is exported + raised by providers for
    the documented permanent conditions.
    """

    def test_exception_importable(self):
        from pipeline_v2.stages.stt import PermanentSTTError
        assert issubclass(PermanentSTTError, Exception)

    def test_deepgram_raises_on_zero_byte_file(self, tmp_path):
        from pipeline_v2.stages.stt import PermanentSTTError
        from pipeline_v2.stages.stt.deepgram import DeepgramNova3Provider
        zero = tmp_path / "zero.mp3"
        zero.write_bytes(b"")
        provider = DeepgramNova3Provider()
        with pytest.raises(PermanentSTTError, match=r"empty_file"):
            provider._validate_file(str(zero))

    def test_assemblyai_raises_on_zero_byte_file(self, tmp_path):
        from pipeline_v2.stages.stt import PermanentSTTError
        from pipeline_v2.stages.stt.assemblyai import (
            AssemblyAIUniversal2Provider,
        )
        zero = tmp_path / "zero.mp3"
        zero.write_bytes(b"")
        provider = AssemblyAIUniversal2Provider()
        with pytest.raises(PermanentSTTError, match=r"empty_file"):
            provider._validate_file(str(zero))

    def test_whisper_groq_raises_on_zero_byte_file(self, tmp_path):
        from pipeline_v2.stages.stt import PermanentSTTError
        from pipeline_v2.stages.stt.whisper_groq import WhisperGroqProvider
        zero = tmp_path / "zero.mp3"
        zero.write_bytes(b"")
        provider = WhisperGroqProvider()
        with pytest.raises(PermanentSTTError, match=r"empty_file"):
            provider._validate_file(str(zero))


# ====================================================================== #
# Step 10.3: PermanentRenderError surface + classification               #
# ====================================================================== #


class TestPermanentRenderErrorClassification:
    """Stage 4's _classify_render_error helper maps exception types
    and message patterns to slug strings (or None for transient).
    """

    def test_ffmpeg_not_found_classified(self):
        from pipeline_v2.stages.stage_4_render import _classify_render_error
        slug = _classify_render_error(
            FileNotFoundError("[Errno 2] No such file: ffmpeg")
        )
        assert slug is not None
        assert slug.startswith("ffmpeg_not_found:")

    def test_ffprobe_not_found_classified(self):
        # ffprobe failure is the same architectural failure as ffmpeg
        # not found -- they ship together; if one's missing, both are.
        from pipeline_v2.stages.stage_4_render import _classify_render_error
        slug = _classify_render_error(
            FileNotFoundError("ffprobe: command not found")
        )
        assert slug is not None
        assert slug.startswith("ffmpeg_not_found:")

    def test_enospc_disk_full_classified(self):
        from pipeline_v2.stages.stage_4_render import _classify_render_error
        # errno 28 = ENOSPC on Linux/macOS
        exc = OSError(28, "No space left on device", "/output/x.mp4")
        slug = _classify_render_error(exc)
        assert slug is not None
        assert slug.startswith("disk_full:")

    def test_windows_disk_full_errno_classified(self):
        from pipeline_v2.stages.stage_4_render import _classify_render_error
        # Windows ERROR_DISK_FULL = 112
        exc = OSError(112, "There is not enough space on the disk")
        slug = _classify_render_error(exc)
        assert slug is not None
        assert slug.startswith("disk_full:")

    def test_disk_full_message_pattern_classified(self):
        # Some upstream code wraps OSError without preserving errno.
        # Message-based fallback catches these.
        from pipeline_v2.stages.stage_4_render import _classify_render_error
        slug = _classify_render_error(
            OSError("write failed: no space left on device")
        )
        assert slug is not None
        assert slug.startswith("disk_full:")

    def test_source_corrupt_invalid_data_classified(self):
        from pipeline_v2.stages.stage_4_render import _classify_render_error
        slug = _classify_render_error(
            RuntimeError("ffprobe returned: invalid data found")
        )
        assert slug is not None
        assert slug.startswith("source_video_corrupt:")

    def test_source_corrupt_moov_atom_classified(self):
        from pipeline_v2.stages.stage_4_render import _classify_render_error
        slug = _classify_render_error(
            RuntimeError("ffprobe stderr: moov atom not found")
        )
        assert slug is not None
        assert slug.startswith("source_video_corrupt:")

    def test_encoder_unavailable_classified(self):
        from pipeline_v2.stages.stage_4_render import _classify_render_error
        slug = _classify_render_error(
            RuntimeError("no h264 encoder available on this system")
        )
        assert slug is not None
        assert slug.startswith("encoder_unavailable_no_fallback:")

    # ---- DOES NOT raise: transient conditions ----

    def test_per_clip_compose_failure_NOT_permanent(self):
        # D-9.7 50% guardrail handles per-clip failures internally.
        # Even if it bubbles up as RuntimeError, the message doesn't
        # match the permanent slugs.
        from pipeline_v2.stages.stage_4_render import _classify_render_error
        exc = RuntimeError(
            "Stage 4 compose_shorts: 3/5 clips failed (60% > 50% threshold)"
        )
        assert _classify_render_error(exc) is None

    def test_r2_upload_failure_NOT_permanent(self):
        from pipeline_v2.stages.stage_4_render import _classify_render_error
        exc = ConnectionError("R2 upload failed: connection reset")
        assert _classify_render_error(exc) is None

    def test_gpu_oom_NOT_permanent(self):
        # GPU OOM is transient -- worker can retry with cleaner state
        from pipeline_v2.stages.stage_4_render import _classify_render_error
        exc = RuntimeError("CUDA out of memory on device 0")
        assert _classify_render_error(exc) is None

    def test_permission_error_NOT_permanent(self):
        # Filesystem permission could be a transient mount issue
        from pipeline_v2.stages.stage_4_render import _classify_render_error
        exc = PermissionError(13, "Permission denied", "/output/x.mp4")
        assert _classify_render_error(exc) is None

    def test_generic_runtime_NOT_permanent(self):
        from pipeline_v2.stages.stage_4_render import _classify_render_error
        assert _classify_render_error(RuntimeError("unknown error")) is None

    def test_value_error_NOT_permanent(self):
        from pipeline_v2.stages.stage_4_render import _classify_render_error
        assert _classify_render_error(ValueError("bad value")) is None


# ====================================================================== #
# Step 10.3: Stage4Render.render() wraps + raises PermanentRenderError    #
# ====================================================================== #


class TestStage4RenderTopLevelClassifies:
    """render() catches the 4 documented conditions and re-raises as
    PermanentRenderError. Other exceptions propagate unchanged.
    """

    def _renderer(self, tmp_path):
        from pipeline_v2.stages.stage_4_render import Stage4Render
        return Stage4Render(
            output_dir=tmp_path / "out",
            video_path=tmp_path / "v.mp4",
            preset={
                "label": "X", "width": 1080, "height": 1920,
                "min_dur": 15, "max_dur": 60, "ideal_dur": 45,
                "vertical": True,
            },
        )

    def test_ffmpeg_not_found_becomes_permanent(self, tmp_path):
        from pipeline_v2.stages.stage_4_render import PermanentRenderError
        from unittest.mock import patch

        renderer = self._renderer(tmp_path)
        # Stub _render_impl to raise FileNotFoundError for ffmpeg
        with patch.object(
            renderer, "_render_impl",
            side_effect=FileNotFoundError("ffmpeg not on PATH"),
        ):
            with pytest.raises(PermanentRenderError, match=r"ffmpeg_not_found"):
                renderer.render(MagicMock(), timestamp="20260519_120000")

    def test_disk_full_becomes_permanent(self, tmp_path):
        from pipeline_v2.stages.stage_4_render import PermanentRenderError
        from unittest.mock import patch

        renderer = self._renderer(tmp_path)
        with patch.object(
            renderer, "_render_impl",
            side_effect=OSError(28, "No space left on device"),
        ):
            with pytest.raises(PermanentRenderError, match=r"disk_full"):
                renderer.render(MagicMock(), timestamp="20260519_120000")

    def test_source_corrupt_becomes_permanent(self, tmp_path):
        from pipeline_v2.stages.stage_4_render import PermanentRenderError
        from unittest.mock import patch

        renderer = self._renderer(tmp_path)
        with patch.object(
            renderer, "_render_impl",
            side_effect=RuntimeError("ffprobe: moov atom not found"),
        ):
            with pytest.raises(PermanentRenderError, match=r"source_video_corrupt"):
                renderer.render(MagicMock(), timestamp="20260519_120000")

    def test_encoder_unavailable_becomes_permanent(self, tmp_path):
        from pipeline_v2.stages.stage_4_render import PermanentRenderError
        from unittest.mock import patch

        renderer = self._renderer(tmp_path)
        with patch.object(
            renderer, "_render_impl",
            side_effect=RuntimeError("no h264 encoder available"),
        ):
            with pytest.raises(
                PermanentRenderError, match=r"encoder_unavailable",
            ):
                renderer.render(MagicMock(), timestamp="20260519_120000")

    def test_per_clip_guardrail_NOT_permanent(self, tmp_path):
        # The 50%-failure guardrail (D-9.7) raises RuntimeError with
        # a percentage in the message -- NOT a permanent slug. Must
        # propagate as-is for Inngest retry.
        from pipeline_v2.stages.stage_4_render import PermanentRenderError
        from unittest.mock import patch

        renderer = self._renderer(tmp_path)
        guardrail_exc = RuntimeError(
            "Stage 4 compose_shorts: 4/5 clips failed (80% > 50%)"
        )
        with patch.object(
            renderer, "_render_impl", side_effect=guardrail_exc,
        ):
            with pytest.raises(RuntimeError) as excinfo:
                renderer.render(MagicMock(), timestamp="20260519_120000")
            # NOT a PermanentRenderError
            assert not isinstance(excinfo.value, PermanentRenderError)

    def test_r2_upload_failure_NOT_permanent(self, tmp_path):
        from pipeline_v2.stages.stage_4_render import PermanentRenderError
        from unittest.mock import patch

        renderer = self._renderer(tmp_path)
        with patch.object(
            renderer, "_render_impl",
            side_effect=ConnectionError("R2 upload timed out"),
        ):
            with pytest.raises(ConnectionError):
                renderer.render(MagicMock(), timestamp="20260519_120000")


# ====================================================================== #
# Step 10.3: Stage 3 fanout handler wiring                                #
# ====================================================================== #


class TestStage3FanoutHandler:
    @pytest.mark.asyncio
    async def test_calls_stage3_fanout_and_stores_result(
        self, monkeypatch, tmp_path,
    ):
        _stub_stage_helpers(monkeypatch)

        from pipeline_v2.models import (
            CleanTranscript, Entity, FullVideoCut, ImagePlan,
            Metadata, ShortsCut, Stage2_5Output, StageTwoOutput, Word,
        )
        from pipeline_v2.stages.stage_3_fanout import Stage3Output

        # Build the inputs Stage 3 needs
        clean = CleanTranscript(
            words=[Word(w="x", s=0.0, e=0.5)],
            clip_boundaries={0: (0, 0)},
            source_word_map=[0],
        )
        stage_2 = StageTwoOutput(
            full_video_cuts=[FullVideoCut(
                index=0, start_word_idx=0, end_word_idx=0,
                start_sec=0.0, end_sec=0.5, importance=5,
            )],
            skipped_segments=[],
            clean_transcript=clean,
            retake_audit="none",
        )
        stage_2_5 = Stage2_5Output(entities=[
            Entity(
                canonical_name="X", native_name="X",
                first_mention_word_idx=0, type="PERSON", mentions=[0],
            ),
        ])

        # Build the Stage 3 mock output
        fake_shorts = [
            ShortsCut(
                index=0, start_sec=0.0, end_sec=20.0,
                hook="A hook", importance=7,
            ),
        ]
        fake_metadata = Metadata(
            video_type="SOLO", language="te-en", total_speakers=1,
            overall_summary="S", overall_summary_native="S",
            shorts_headline_native="H",
            bulletin_marquee_points=["A", "B", "C"],
            image_search_queries=["q"],
            key_people=["X"], key_people_native=["X"],
            key_topics=["t"], key_locations=[],
        )
        fake_image_plan = ImagePlan(entries=[])
        fake_result = Stage3Output(
            shorts_cuts=fake_shorts,
            metadata=fake_metadata,
            image_plan=fake_image_plan,
        )

        class _FakeFanOut:
            async def run(self, *, clean, full_video_cuts, entities):
                return fake_result

        monkeypatch.setattr(orchestrator, "Stage3FanOut", _FakeFanOut)

        env = orchestrator._envelope_init({"job_id": 1})
        env["stage_2"] = stage_2.model_dump()
        env["stage_2_5"] = stage_2_5.model_dump()
        out = await orchestrator._stage_3_fanout_handler(env)

        assert out["stage_3"]["metadata"]["video_type"] == "SOLO"
        assert len(out["stage_3"]["shorts_cuts"]) == 1
        assert out["stage_3"]["image_plan"]["entries"] == []
        assert out["stage_costs"]["stage_3_fanout"] == 0.0


# ====================================================================== #
# Step 10.3: Stage 4 render handler wiring                                #
# ====================================================================== #


class TestStage4RenderHandler:
    def _envelope_with_upstream(self, tmp_path, job_id: int = 1) -> dict:
        """Build a fully-populated envelope as if stages 0-3 ran."""
        from pipeline_v2.models import (
            CleanTranscript, Entity, FullVideoCut, ImagePlan,
            Metadata, ShortsCut, Stage2_5Output, StageTwoOutput, Word,
        )
        clean = CleanTranscript(
            words=[Word(w="x", s=0.0, e=0.5)],
            clip_boundaries={0: (0, 0)},
            source_word_map=[0],
        )
        stage_2 = StageTwoOutput(
            full_video_cuts=[FullVideoCut(
                index=0, start_word_idx=0, end_word_idx=0,
                start_sec=0.0, end_sec=0.5, importance=5,
            )],
            skipped_segments=[],
            clean_transcript=clean,
            retake_audit="none",
        )
        stage_2_5 = Stage2_5Output(entities=[
            Entity(
                canonical_name="X", native_name="X",
                first_mention_word_idx=0, type="PERSON", mentions=[0],
            ),
        ])
        shorts = [ShortsCut(
            index=0, start_sec=0.0, end_sec=20.0,
            hook="A hook", importance=7,
        )]
        metadata = Metadata(
            video_type="SOLO", language="te-en", total_speakers=1,
            overall_summary="S", overall_summary_native="S",
            shorts_headline_native="H", bulletin_marquee_points=["A"],
            image_search_queries=[], key_people=[], key_people_native=[],
            key_topics=[], key_locations=[],
        )
        image_plan = ImagePlan(entries=[])

        env = orchestrator._envelope_init({
            "job_id": job_id,
            "video_path": str(tmp_path / "v.mp4"),
            "platform": "full_video_shorts_v2",
            "frame_layout": "torn_card",
            "preset": {
                "label": "X", "width": 1080, "height": 1920,
                "min_dur": 15, "max_dur": 60, "ideal_dur": 45,
                "vertical": True,
            },
            "out_dir": str(tmp_path / "out"),
        })
        env["stage_2"] = stage_2.model_dump()
        env["stage_2_5"] = stage_2_5.model_dump()
        env["stage_3"] = {
            "shorts_cuts": [s.model_dump() for s in shorts],
            "metadata":    metadata.model_dump(),
            "image_plan":  image_plan.model_dump(),
        }
        return env

    @pytest.mark.asyncio
    async def test_calls_stage_4_render_and_stores_result(
        self, monkeypatch, tmp_path,
    ):
        _stub_stage_helpers(monkeypatch)

        # Mock Stage4Render so we don't touch real FFmpeg
        from pipeline_v2.stages.stage_4_render import RenderResult

        fake_result = RenderResult(
            shorts_editor_meta_path=str(tmp_path / "shorts.json"),
            bulletin_editor_meta_path=str(tmp_path / "bulletin.json"),
            composed_shorts=[{"clip_path": "/abs/clip_01.mp4"}],
            bulletin={
                "bulletin_path":     str(tmp_path / "b.mp4"),
                "overlay_path":      str(tmp_path / "b.mp4"),
                "overlay_applied":   False,
                "duration_s":        60.0,
                "stories_rendered":  1,
                "stories_skipped":   0,
                "warnings":          [],
            },
        )

        class _FakeRenderer:
            def __init__(self, **kw):
                self.last_kw = kw
            def render(self, job_output, timestamp, **kw):
                return fake_result

        monkeypatch.setattr(orchestrator, "Stage4Render", _FakeRenderer)
        # Block _ACTIVE_PROCS bridge in this test (avoids needing
        # V1's runner module on sys.path during unit tests)
        monkeypatch.setattr(
            orchestrator, "_register_stage_4_with_active_procs",
            lambda jid: None,
        )
        monkeypatch.setattr(
            orchestrator, "_deregister_stage_4_from_active_procs",
            lambda jid: None,
        )

        env = self._envelope_with_upstream(tmp_path)
        out = await orchestrator._stage_4_render_handler(env)

        assert out["stage_4"]["shorts_editor_meta_path"].endswith("shorts.json")
        assert out["stage_4"]["composed_shorts_count"] == 1
        assert out["stage_4"]["bulletin"]["stories_rendered"] == 1
        assert out["stage_costs"]["stage_4_render"] == 0.0

    @pytest.mark.asyncio
    async def test_permanent_render_error_translates_to_non_retriable(
        self, monkeypatch, tmp_path,
    ):
        _stub_stage_helpers(monkeypatch)
        from pipeline_v2.stages.stage_4_render import PermanentRenderError
        from inngest import NonRetriableError

        class _FakeRenderer:
            def __init__(self, **kw):
                pass
            def render(self, job_output, timestamp, **kw):
                raise PermanentRenderError("ffmpeg_not_found: not on PATH")

        monkeypatch.setattr(orchestrator, "Stage4Render", _FakeRenderer)
        monkeypatch.setattr(
            orchestrator, "_register_stage_4_with_active_procs",
            lambda jid: None,
        )
        monkeypatch.setattr(
            orchestrator, "_deregister_stage_4_from_active_procs",
            lambda jid: None,
        )

        env = self._envelope_with_upstream(tmp_path)
        with pytest.raises(NonRetriableError, match=r"permanent render: ffmpeg_not_found"):
            await orchestrator._stage_4_render_handler(env)

    @pytest.mark.asyncio
    async def test_transient_error_does_not_translate(
        self, monkeypatch, tmp_path,
    ):
        _stub_stage_helpers(monkeypatch)
        from inngest import NonRetriableError

        class _FakeRenderer:
            def __init__(self, **kw):
                pass
            def render(self, job_output, timestamp, **kw):
                raise ConnectionError("network blip")

        monkeypatch.setattr(orchestrator, "Stage4Render", _FakeRenderer)
        monkeypatch.setattr(
            orchestrator, "_register_stage_4_with_active_procs",
            lambda jid: None,
        )
        monkeypatch.setattr(
            orchestrator, "_deregister_stage_4_from_active_procs",
            lambda jid: None,
        )

        env = self._envelope_with_upstream(tmp_path)
        with pytest.raises(ConnectionError):
            await orchestrator._stage_4_render_handler(env)

    @pytest.mark.asyncio
    async def test_active_procs_register_deregister_called(
        self, monkeypatch, tmp_path,
    ):
        # D-10.9 part 2: Stage 4 wraps render() in
        # register/finally-deregister so the V1 cancel-job endpoint
        # can SIGKILL FFmpeg descendants mid-render.
        _stub_stage_helpers(monkeypatch)
        from pipeline_v2.stages.stage_4_render import RenderResult

        fake_result = RenderResult(
            shorts_editor_meta_path=None,
            bulletin_editor_meta_path=None,
            composed_shorts=[],
            bulletin=None,
        )

        class _FakeRenderer:
            def __init__(self, **kw):
                pass
            def render(self, job_output, timestamp, **kw):
                return fake_result

        monkeypatch.setattr(orchestrator, "Stage4Render", _FakeRenderer)
        register_calls: list[int] = []
        deregister_calls: list[int] = []
        monkeypatch.setattr(
            orchestrator, "_register_stage_4_with_active_procs",
            lambda jid: register_calls.append(jid),
        )
        monkeypatch.setattr(
            orchestrator, "_deregister_stage_4_from_active_procs",
            lambda jid: deregister_calls.append(jid),
        )

        env = self._envelope_with_upstream(tmp_path, job_id=77)
        await orchestrator._stage_4_render_handler(env)

        assert register_calls == [77]
        assert deregister_calls == [77]

    @pytest.mark.asyncio
    async def test_active_procs_deregister_runs_on_exception(
        self, monkeypatch, tmp_path,
    ):
        # The deregister MUST run even when render raises -- otherwise
        # cancel-job sees a stale registration for the next job.
        _stub_stage_helpers(monkeypatch)

        class _FakeRenderer:
            def __init__(self, **kw):
                pass
            def render(self, job_output, timestamp, **kw):
                raise RuntimeError("render kaboom")

        monkeypatch.setattr(orchestrator, "Stage4Render", _FakeRenderer)
        deregister_calls: list[int] = []
        monkeypatch.setattr(
            orchestrator, "_register_stage_4_with_active_procs",
            lambda jid: None,
        )
        monkeypatch.setattr(
            orchestrator, "_deregister_stage_4_from_active_procs",
            lambda jid: deregister_calls.append(jid),
        )

        env = self._envelope_with_upstream(tmp_path, job_id=88)
        with pytest.raises(RuntimeError):
            await orchestrator._stage_4_render_handler(env)
        # Deregister still ran even though render threw
        assert deregister_calls == [88]


# ====================================================================== #
# Step 10.3: _V2WorkerProxy mimics Popen interface for V1 cancel_job      #
# ====================================================================== #


class TestV2WorkerProxy:
    """Proxy MUST satisfy V1 cancel_job's expectations:
      - .pid attribute (int)
      - .poll() returns None while "running"
      - .terminate() / .kill() are NO-OPs (don't kill the worker!)
      - .wait(timeout) raises TimeoutExpired
    """

    def test_pid_attribute(self):
        proxy = orchestrator._V2WorkerProxy(pid=12345)
        assert proxy.pid == 12345

    def test_poll_returns_none(self):
        proxy = orchestrator._V2WorkerProxy(pid=12345)
        assert proxy.poll() is None

    def test_terminate_is_noop(self):
        # Must NOT signal the worker process
        proxy = orchestrator._V2WorkerProxy(pid=12345)
        # Should return cleanly, not raise
        assert proxy.terminate() is None

    def test_kill_is_noop(self):
        proxy = orchestrator._V2WorkerProxy(pid=12345)
        assert proxy.kill() is None

    def test_wait_raises_timeout(self):
        import subprocess
        proxy = orchestrator._V2WorkerProxy(pid=12345)
        with pytest.raises(subprocess.TimeoutExpired):
            proxy.wait(timeout=0)

    def test_wait_timeout_param_handled(self):
        import subprocess
        proxy = orchestrator._V2WorkerProxy(pid=12345)
        with pytest.raises(subprocess.TimeoutExpired):
            proxy.wait(timeout=5)


class TestActiveProcsBridge:
    """_register / _deregister bridge functions are defensive: failure
    to find runner module logs a warning + returns/no-ops.
    """

    def test_register_skips_zero_job_id(self):
        assert orchestrator._register_stage_4_with_active_procs(0) is None
        assert orchestrator._register_stage_4_with_active_procs(-1) is None
        assert orchestrator._register_stage_4_with_active_procs(None) is None

    def test_deregister_skips_zero_job_id(self):
        # No-op, no raise
        orchestrator._deregister_stage_4_from_active_procs(0)
        orchestrator._deregister_stage_4_from_active_procs(-1)
        orchestrator._deregister_stage_4_from_active_procs(None)

    def test_register_returns_proxy_on_success(self, monkeypatch):
        # Fake the runner module to confirm register/deregister wires
        # against V1's _ACTIVE_PROCS API.
        fake_runner = MagicMock()
        registered: list[tuple[int, object]] = []
        deregistered: list[int] = []

        def _fake_register_proc(jid, proc):
            registered.append((jid, proc))

        def _fake_deregister_proc(jid):
            deregistered.append(jid)

        fake_runner._register_proc = _fake_register_proc
        fake_runner._deregister_proc = _fake_deregister_proc

        monkeypatch.setitem(sys.modules, "runner", fake_runner)

        proxy = orchestrator._register_stage_4_with_active_procs(99)
        assert proxy is not None
        assert isinstance(proxy, orchestrator._V2WorkerProxy)
        assert len(registered) == 1
        assert registered[0][0] == 99

        orchestrator._deregister_stage_4_from_active_procs(99)
        assert deregistered == [99]


# ====================================================================== #
# Step 10.4: finalize handler                                             #
# ====================================================================== #


def _make_finalize_envelope(tmp_path, job_id: int = 1) -> dict:
    """Build a complete envelope as if stages 0-4 ran."""
    env = orchestrator._envelope_init({
        "job_id": job_id,
        "video_path": str(tmp_path / "v.mp4"),
        "platform": "full_video_shorts_v2",
        "out_dir": str(tmp_path / "out"),
    })
    # Mark stage 0 ran so the passthrough-skip heuristic from
    # earlier dev isn't triggered.
    env["stage_0"] = {"mezzanine_path": "/x", "audio_path": "/y",
                      "duration_sec": 0.5, "encoder_used": "libx264",
                      "source_was_vfr": False,
                      "transcode_seconds": 1.0,
                      "audio_extract_seconds": 1.0, "wall_seconds": 2.0}
    env["stage_4"] = {
        "shorts_editor_meta_path":   str(tmp_path / "out" / "editor_meta.json"),
        "bulletin_editor_meta_path": str(tmp_path / "out" / "bulletin" / "editor_meta.json"),
        "composed_shorts_count":     3,
        "bulletin": {
            "bulletin_path":    str(tmp_path / "b.mp4"),
            "overlay_path":     str(tmp_path / "b.mp4"),
            "overlay_applied":  False,
            "duration_s":       60.5,
            "stories_rendered": 1,
            "stories_skipped":  0,
            "warnings":         [],
        },
    }
    env["stage_costs"] = {
        "stage_0_ingest":      0.0,
        "stage_1_transcribe":  0.005,
        "stage_2_continuity":  0.0,
        "stage_2_5_entities":  0.0,
        "stage_3_fanout":      0.0,
        "stage_4_render":      0.0,
    }
    return env


class TestFinalizeHandlerHappyPath:
    @pytest.mark.asyncio
    async def test_writes_job_status_done_and_clears_current_stage(
        self, monkeypatch, tmp_path, caplog,
    ):
        _stub_stage_helpers(monkeypatch)
        # Stub DB session: capture the update payload
        captured_updates: list[dict] = []
        fake_session = MagicMock()

        def _capture_update(payload, **kw):
            captured_updates.append(payload)
            return 1
        fake_session.query.return_value.filter.return_value.update = _capture_update
        # The Clip count call returns 0 (no clips imported in this test)
        fake_session.query.return_value.filter.return_value.count.return_value = 0
        # The Job lookup for _import_clips returns None (skip import)
        fake_session.query.return_value.filter.return_value.first.return_value = None
        monkeypatch.setattr(
            orchestrator, "_open_db_session", lambda: fake_session,
        )
        # Block runner import (V1 module not always loaded in tests)
        # by injecting a stub module.
        import types
        fake_runner = types.SimpleNamespace(
            _import_clips=lambda job, db, meta_override=None: None,
        )
        monkeypatch.setitem(sys.modules, "runner", fake_runner)

        env = _make_finalize_envelope(tmp_path, job_id=1)
        out = await orchestrator._finalize_handler(env)

        # Status=done + current_stage cleared
        assert len(captured_updates) >= 1
        status_update = captured_updates[0]
        assert status_update["status"] == "done"
        assert status_update["current_stage"] is None
        assert "finished_at" in status_update
        assert status_update["output_dir"] == env["out_dir"]
        # Result envelope
        assert out["finalize"]["status"] == "done"

    @pytest.mark.asyncio
    async def test_cost_ledger_logged_at_finalize(
        self, monkeypatch, tmp_path, caplog,
    ):
        import logging as pylog
        _stub_stage_helpers(monkeypatch)
        fake_session = MagicMock()
        fake_session.query.return_value.filter.return_value.first.return_value = None
        fake_session.query.return_value.filter.return_value.count.return_value = 0
        monkeypatch.setattr(
            orchestrator, "_open_db_session", lambda: fake_session,
        )
        import types
        monkeypatch.setitem(sys.modules, "runner", types.SimpleNamespace(
            _import_clips=lambda j, d, meta_override=None: None,
        ))

        env = _make_finalize_envelope(tmp_path, job_id=7)
        env["stage_costs"] = {
            "stage_1_transcribe": 0.012,
            "stage_2_continuity": 0.003,
            "stage_3_fanout":     0.001,
        }
        with caplog.at_level(pylog.INFO, logger="pipeline_v2.orchestrator"):
            out = await orchestrator._finalize_handler(env)

        # The ledger log line is structured (extra= kwargs). Find it.
        ledger_records = [
            r for r in caplog.records
            if r.getMessage() == "v2_cost_ledger"
        ]
        assert len(ledger_records) == 1
        record = ledger_records[0]
        assert record.job_id == 7
        # Total = sum of all stage_costs values
        assert record.total_usd == round(0.012 + 0.003 + 0.001, 4)
        # Plus per-stage breakdown
        assert record.stage_costs["stage_1_transcribe"] == 0.012
        # And the final envelope echoes the total
        assert out["finalize"]["total_cost_usd"] == round(0.016, 4)

    @pytest.mark.asyncio
    async def test_db_failure_at_finalize_is_non_fatal(
        self, monkeypatch, tmp_path,
    ):
        # Per the handler docstring: actual work IS done; DB write
        # failure logs a warning but doesn't raise (would otherwise
        # trigger Inngest retry of the whole pipeline).
        _stub_stage_helpers(monkeypatch)

        def _bad_open():
            raise RuntimeError("DB down")
        monkeypatch.setattr(orchestrator, "_open_db_session", _bad_open)
        # Also block runner module so _import_clips doesn't hit the
        # broken session.
        import types
        monkeypatch.setitem(sys.modules, "runner", types.SimpleNamespace(
            _import_clips=lambda j, d, meta_override=None: None,
        ))

        env = _make_finalize_envelope(tmp_path, job_id=1)
        # Should NOT raise
        out = await orchestrator._finalize_handler(env)
        assert out["finalize"]["status"] == "done"

    @pytest.mark.asyncio
    async def test_import_clips_failure_is_non_fatal(
        self, monkeypatch, tmp_path,
    ):
        # _import_clips failure (e.g. editor_meta.json missing) logs
        # warning + continues. The pipeline IS done.
        _stub_stage_helpers(monkeypatch)

        fake_session = MagicMock()
        fake_session.query.return_value.filter.return_value.first.return_value = MagicMock()
        fake_session.query.return_value.filter.return_value.count.return_value = 0
        monkeypatch.setattr(
            orchestrator, "_open_db_session", lambda: fake_session,
        )

        def _failing_import(job, db, meta_override=None):
            raise FileNotFoundError("editor_meta.json missing")

        import types
        monkeypatch.setitem(sys.modules, "runner", types.SimpleNamespace(
            _import_clips=_failing_import,
        ))

        env = _make_finalize_envelope(tmp_path, job_id=1)
        out = await orchestrator._finalize_handler(env)
        assert out["finalize"]["status"] == "done"
        assert out["finalize"]["imported_clip_count"] == 0


# ====================================================================== #
# Step 10.4: _mark_job_failed (outer try/except path, D-10.15)            #
# ====================================================================== #


class TestMarkJobFailed:
    def test_skips_zero_job_id(self, monkeypatch):
        # Should NOT raise + NOT open a session
        opened = []
        monkeypatch.setattr(
            orchestrator, "_open_db_session",
            lambda: opened.append(1) or MagicMock(),
        )
        orchestrator._mark_job_failed(0, RuntimeError("x"))
        orchestrator._mark_job_failed(-1, RuntimeError("x"))
        orchestrator._mark_job_failed(None, RuntimeError("x"))
        assert opened == []

    def test_writes_status_failed_and_error_text(self, monkeypatch):
        captured: list[dict] = []
        fake_session = MagicMock()

        def _capture(payload, **kw):
            captured.append(payload)
            return 1
        fake_session.query.return_value.filter.return_value.update = _capture
        monkeypatch.setattr(
            orchestrator, "_open_db_session", lambda: fake_session,
        )

        exc = RuntimeError("kaboom")
        try:
            raise exc
        except RuntimeError:
            orchestrator._mark_job_failed(42, exc)

        assert len(captured) == 1
        payload = captured[0]
        assert payload["status"] == "failed"
        assert payload["current_stage"] is None
        assert "RuntimeError" in payload["error"]
        assert "kaboom" in payload["error"]
        assert "finished_at" in payload
        fake_session.commit.assert_called_once()

    def test_truncates_long_traceback(self, monkeypatch):
        captured: list[dict] = []
        fake_session = MagicMock()
        fake_session.query.return_value.filter.return_value.update = (
            lambda payload, **kw: captured.append(payload) or 1
        )
        monkeypatch.setattr(
            orchestrator, "_open_db_session", lambda: fake_session,
        )

        # Synthetic very-long message
        exc = RuntimeError("X" * 8000)
        try:
            raise exc
        except RuntimeError:
            orchestrator._mark_job_failed(1, exc)

        assert len(captured) == 1
        error_text = captured[0]["error"]
        # Per the 4KB budget
        assert len(error_text) <= 4000

    def test_db_failure_is_non_fatal(self, monkeypatch):
        # Outer try/except already raised; mark_job_failed's own
        # DB failure must not raise (it would mask the original).
        def _bad_open():
            raise RuntimeError("DB also down")
        monkeypatch.setattr(orchestrator, "_open_db_session", _bad_open)

        # Should NOT raise -- bad_open's error logged + swallowed.
        orchestrator._mark_job_failed(7, ValueError("original"))


# ====================================================================== #
# Step 10.4: outer try/except wired into process_video_v2                 #
# ====================================================================== #


class TestProcessVideoV2OuterExceptionPath:
    """D-10.15: top-level try/except wraps the full step sequence.
    On unhandled exception (including NonRetriableError after final
    retry), Job.status='failed' + Job.error are written, then the
    exception is re-raised so Inngest sees the failure.

    We can't easily test the actual Inngest function decoration; the
    underlying coroutine logic is what matters. We test
    _mark_job_failed directly (above) + verify that the
    process_video_v2 function exists with the expected decorator.
    """

    def test_function_remains_registered(self):
        # After Step 10.4 wiring, the Inngest Function registration
        # MUST still be intact (no decorator regression).
        from pipeline_v2.orchestrator import process_video_v2
        assert type(process_video_v2).__name__ == "Function"
        assert process_video_v2.id == "kaizer-v2-process-video-v2"

    def test_mark_job_failed_is_callable(self):
        # Pin the helper exists and has the expected signature.
        import inspect
        sig = inspect.signature(orchestrator._mark_job_failed)
        params = list(sig.parameters)
        assert params == ["job_id", "exc"]
