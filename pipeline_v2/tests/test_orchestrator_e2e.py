"""End-to-end mocked test for the V2 Inngest orchestrator (Step 10.5).

This file is the high-value gate: it drives the FULL 7-step sequence
with every stage mocked and asserts:

  1. Stage handlers are called in the correct order
  2. State (the envelope) threads through every step intact
  3. Cancellation between steps short-circuits the sequence
  4. PermanentSTTError -> NonRetriableError translation in context
  5. PermanentRenderError -> NonRetriableError translation in context
  6. Cost ledger accumulation across stages + emission at finalize
  7. current_stage is written before each handler call (D-10.7)
  8. _ACTIVE_PROCS register/deregister wraps Stage 4 (D-10.9)

Why a dedicated file: Inngest 0.5.18 doesn't ship a heavy test
harness for ``Function`` objects. Calling ``process_video_v2``
directly requires constructing a real Inngest ``Context`` + ``Step``
which couples to SDK internals. Instead we exercise the SAME step
handler functions in the SAME order the Function's body invokes
them -- the orchestration topology lives in the function body
itself, so a parallel sequence here is a faithful E2E proxy.

If Inngest's Python SDK adds a public test harness in a future
release, replace this file's manual chaining with the official
harness.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pipeline_v2 import orchestrator
from pipeline_v2.orchestrator import (
    FINALIZE,
    STAGE_0_INGEST,
    STAGE_1_TRANSCRIBE,
    STAGE_2_5_ENTITIES,
    STAGE_2_CONTINUITY,
    STAGE_3_FANOUT,
    STAGE_4_RENDER,
)


# ====================================================================== #
# Reusable stage-output fixtures                                          #
# ====================================================================== #


def _stage_0_out(tmp_path: Path):
    from pipeline_v2.models import Stage0Output
    return Stage0Output(
        mezzanine_path=str(tmp_path / "mez.mp4"),
        audio_path=str(tmp_path / "a.mp3"),
        duration_sec=60.0, encoder_used="libx264",
        source_was_vfr=False,
        transcode_seconds=5.0,
        audio_extract_seconds=2.0, wall_seconds=7.0,
    )


def _stage_1_out():
    from pipeline_v2.models import Stage1Output, Word, WordLevelTranscript
    return Stage1Output(
        transcript=WordLevelTranscript(
            words=[Word(w="hello", s=0.0, e=0.5)],
            duration_sec=60.0,
            detected_languages=["te"],
            provider="deepgram",
        ),
        stt_provider="deepgram",
        stt_audio_duration_sec=60.0,
        stt_wall_seconds=8.0,
        stt_cost_usd=0.0097,
        stt_word_count=1,
        stt_language_detected="te",
        stt_request_id="req-e2e-1",
    )


def _stage_2_output_models():
    from pipeline_v2.models import (
        CleanTranscript, FullVideoCut, Stage2Output, StageTwoOutput, Word,
    )
    decisions = Stage2Output(
        full_video_cuts=[FullVideoCut(
            index=0, start_word_idx=0, end_word_idx=0,
            start_sec=0.0, end_sec=0.5, importance=8,
        )],
        skipped_segments=[],
        retake_audit="None skipped.",
    )
    full = StageTwoOutput(
        full_video_cuts=decisions.full_video_cuts,
        skipped_segments=[],
        clean_transcript=CleanTranscript(
            words=[Word(w="hello", s=0.0, e=0.5)],
            clip_boundaries={0: (0, 0)},
            source_word_map=[0],
        ),
        retake_audit="None skipped.",
    )
    return decisions, full


def _stage_2_5_out():
    from pipeline_v2.models import Entity, Stage2_5Output
    return Stage2_5Output(entities=[
        Entity(
            canonical_name="Modi", native_name="మోదీ",
            first_mention_word_idx=0, type="PERSON", mentions=[0],
        ),
    ])


def _stage_3_out():
    from pipeline_v2.models import (
        ImagePlan, ImagePlanEntry, Metadata, ShortsCut,
    )
    from pipeline_v2.stages.stage_3_fanout import Stage3Output
    shorts = [
        ShortsCut(
            index=0, start_sec=0.0, end_sec=20.0,
            hook="A hook", importance=7,
        ),
        ShortsCut(
            index=1, start_sec=30.0, end_sec=50.0,
            hook="B hook", importance=6,
        ),
        ShortsCut(
            index=2, start_sec=60.0, end_sec=80.0,
            hook="C hook", importance=8,
        ),
    ]
    metadata = Metadata(
        video_type="SOLO",
        language="te-en",
        total_speakers=1,
        overall_summary="Bandi case summary.",
        overall_summary_native="బండి కేసు సమీక్ష.",
        shorts_headline_native="బండి భగీరథ్ కేసులో మలుపులు",
        bulletin_marquee_points=["A", "B", "C"],
        image_search_queries=["Bandi case"],
        key_people=["Modi"],
        key_people_native=["మోదీ"],
        key_topics=["court"],
        key_locations=["Hyderabad"],
    )
    image_plan = ImagePlan(entries=[
        ImagePlanEntry(
            entity_name="Modi", entity_name_native="మోదీ",
            description="Modi at podium",
            clip_index=0, show_at_sec=5.0, duration_sec=4.0,
        ),
    ])
    return Stage3Output(
        shorts_cuts=shorts,
        metadata=metadata,
        image_plan=image_plan,
    )


def _stage_4_out(tmp_path: Path):
    from pipeline_v2.stages.stage_4_render import RenderResult
    return RenderResult(
        shorts_editor_meta_path=str(tmp_path / "out" / "editor_meta.json"),
        bulletin_editor_meta_path=str(tmp_path / "out" / "bulletin" / "editor_meta.json"),
        composed_shorts=[
            {"clip_path": str(tmp_path / "out" / "clip_01.mp4")},
            {"clip_path": str(tmp_path / "out" / "clip_02.mp4")},
            {"clip_path": str(tmp_path / "out" / "clip_03.mp4")},
        ],
        bulletin={
            "bulletin_path":     str(tmp_path / "out" / "bulletin" / "bulletin.mp4"),
            "overlay_path":      str(tmp_path / "out" / "bulletin" / "bulletin_with_overlays.mp4"),
            "overlay_applied":   True,
            "duration_s":        60.5,
            "stories_rendered":  1,
            "stories_skipped":   0,
            "warnings":          [],
        },
    )


# ====================================================================== #
# Patch every stage to deterministic mocks                                #
# ====================================================================== #


def _install_stage_mocks(monkeypatch, tmp_path: Path, *, call_order: list):
    """Replace every stage call so the orchestrator runs end-to-end
    without touching real LLMs, FFmpeg, or DB. ``call_order`` is a
    list that gets appended-to in stage order so tests can assert
    "stages ran in this sequence".
    """
    # Stage 0
    async def _fake_run_stage_0(src, out, **kw):
        call_order.append(STAGE_0_INGEST)
        return _stage_0_out(tmp_path)
    monkeypatch.setattr(orchestrator, "run_stage_0", _fake_run_stage_0)

    # Stage 1
    async def _fake_run_stage_1(audio_path, **kw):
        call_order.append(STAGE_1_TRANSCRIBE)
        return _stage_1_out()
    monkeypatch.setattr(orchestrator, "run_stage_1", _fake_run_stage_1)

    # Stage 2: editor + assembler
    decisions, full = _stage_2_output_models()

    class _FakeEditor:
        async def transcribe_to_decisions(self, s1):
            call_order.append(STAGE_2_CONTINUITY)
            return decisions
    monkeypatch.setattr(
        orchestrator, "Stage2ContinuityEditor", _FakeEditor,
    )
    monkeypatch.setattr(
        orchestrator, "assemble_stage_two_output",
        lambda s1, dec: full,
    )

    # Stage 2.5
    class _FakeCanonicalizer:
        async def classify(self, ct):
            call_order.append(STAGE_2_5_ENTITIES)
            return _stage_2_5_out()
    monkeypatch.setattr(
        orchestrator, "Stage2_5EntityCanonicalizer", _FakeCanonicalizer,
    )

    # Stage 3 fanout
    class _FakeFanOut:
        async def run(self, *, clean, full_video_cuts, entities):
            call_order.append(STAGE_3_FANOUT)
            return _stage_3_out()
    monkeypatch.setattr(orchestrator, "Stage3FanOut", _FakeFanOut)

    # Stage 4 render
    class _FakeRenderer:
        def __init__(self, **kw):
            self.kw = kw
        def render(self, job_output, timestamp, **kw):
            call_order.append(STAGE_4_RENDER)
            return _stage_4_out(tmp_path)
    monkeypatch.setattr(orchestrator, "Stage4Render", _FakeRenderer)

    # _ACTIVE_PROCS bridge (D-10.9): track register/deregister
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

    # Finalize: DB stubs
    fake_session = MagicMock()
    fake_session.query.return_value.filter.return_value.first.return_value = None
    fake_session.query.return_value.filter.return_value.count.return_value = 3
    monkeypatch.setattr(
        orchestrator, "_open_db_session", lambda: fake_session,
    )
    monkeypatch.setitem(sys.modules, "runner", types.SimpleNamespace(
        _import_clips=lambda j, d, meta_override=None: None,
    ))

    # Cancel check + current_stage write: track every (job_id, stage)
    cancel_calls: list[int] = []
    stage_writes: list[tuple[int, str]] = []
    monkeypatch.setattr(
        orchestrator, "_check_cancelled",
        lambda jid: cancel_calls.append(jid),
    )
    monkeypatch.setattr(
        orchestrator, "_write_current_stage",
        lambda jid, s: stage_writes.append((jid, s)),
    )

    return {
        "call_order": call_order,
        "register_calls": register_calls,
        "deregister_calls": deregister_calls,
        "cancel_calls": cancel_calls,
        "stage_writes": stage_writes,
        "db_session": fake_session,
    }


async def _drive_all_stages(envelope: dict) -> dict:
    """Drive every stage handler in process_video_v2's order. Returns
    the final envelope.
    """
    envelope = await orchestrator._stage_0_ingest_handler(envelope)
    envelope = await orchestrator._stage_1_transcribe_handler(envelope)
    envelope = await orchestrator._stage_2_continuity_handler(envelope)
    envelope = await orchestrator._stage_2_5_entities_handler(envelope)
    envelope = await orchestrator._stage_3_fanout_handler(envelope)
    envelope = await orchestrator._stage_4_render_handler(envelope)
    envelope = await orchestrator._finalize_handler(envelope)
    return envelope


# ====================================================================== #
# E2E happy path                                                          #
# ====================================================================== #


class TestE2EOrchestratorHappyPath:
    @pytest.mark.asyncio
    async def test_full_pipeline_seven_stages_in_order(
        self, monkeypatch, tmp_path,
    ):
        call_order: list[str] = []
        ctx = _install_stage_mocks(monkeypatch, tmp_path, call_order=call_order)

        env = orchestrator._envelope_init({
            "job_id":       100,
            "video_path":   str(tmp_path / "v.mp4"),
            "language":     "te-en",
            "platform":     "full_video_shorts_v2",
            "frame_layout": "torn_card",
            "preset": {
                "label": "X", "width": 1080, "height": 1920,
                "min_dur": 15, "max_dur": 60, "ideal_dur": 45,
                "vertical": True,
            },
            "out_dir": str(tmp_path / "out"),
        })
        out = await _drive_all_stages(env)

        # All 6 stage bodies ran (finalize doesn't append to
        # call_order because it has no stage call -- it just writes DB)
        assert ctx["call_order"] == [
            STAGE_0_INGEST,
            STAGE_1_TRANSCRIBE,
            STAGE_2_CONTINUITY,
            STAGE_2_5_ENTITIES,
            STAGE_3_FANOUT,
            STAGE_4_RENDER,
        ]
        # Final envelope has every stage output populated
        assert out["stage_0"]["audio_path"].endswith("a.mp3")
        assert out["stage_1"]["stt_provider"] == "deepgram"
        assert out["stage_2"]["retake_audit"] == "None skipped."
        assert len(out["stage_2_5"]["entities"]) == 1
        assert len(out["stage_3"]["shorts_cuts"]) == 3
        assert out["stage_4"]["composed_shorts_count"] == 3
        assert out["finalize"]["status"] == "done"

    @pytest.mark.asyncio
    async def test_current_stage_written_for_every_step(
        self, monkeypatch, tmp_path,
    ):
        call_order: list[str] = []
        ctx = _install_stage_mocks(monkeypatch, tmp_path, call_order=call_order)

        env = orchestrator._envelope_init({"job_id": 42})
        await _drive_all_stages(env)

        # current_stage written for ALL 7 stages in correct order
        written_stages = [s for (jid, s) in ctx["stage_writes"]]
        assert written_stages == [
            STAGE_0_INGEST,
            STAGE_1_TRANSCRIBE,
            STAGE_2_CONTINUITY,
            STAGE_2_5_ENTITIES,
            STAGE_3_FANOUT,
            STAGE_4_RENDER,
            FINALIZE,
        ]
        # Every write keyed by the correct job_id
        assert all(jid == 42 for jid, _ in ctx["stage_writes"])

    @pytest.mark.asyncio
    async def test_cancel_check_fires_at_every_step_start(
        self, monkeypatch, tmp_path,
    ):
        call_order: list[str] = []
        ctx = _install_stage_mocks(monkeypatch, tmp_path, call_order=call_order)

        env = orchestrator._envelope_init({"job_id": 77})
        await _drive_all_stages(env)

        # 7 cancel checks (one per step start)
        assert len(ctx["cancel_calls"]) == 7
        assert all(jid == 77 for jid in ctx["cancel_calls"])

    @pytest.mark.asyncio
    async def test_active_procs_bridge_brackets_stage_4(
        self, monkeypatch, tmp_path,
    ):
        # D-10.9 part 2: register on Stage 4 entry, deregister on exit
        call_order: list[str] = []
        ctx = _install_stage_mocks(monkeypatch, tmp_path, call_order=call_order)

        env = orchestrator._envelope_init({"job_id": 5})
        await _drive_all_stages(env)

        assert ctx["register_calls"] == [5]
        assert ctx["deregister_calls"] == [5]

    @pytest.mark.asyncio
    async def test_cost_ledger_accumulates_through_pipeline(
        self, monkeypatch, tmp_path,
    ):
        call_order: list[str] = []
        ctx = _install_stage_mocks(monkeypatch, tmp_path, call_order=call_order)

        env = orchestrator._envelope_init({"job_id": 1})
        out = await _drive_all_stages(env)

        # Stage 1 (Deepgram fake) contributes 0.0097; everything else 0.0
        costs = out["stage_costs"]
        assert STAGE_0_INGEST in costs
        assert STAGE_1_TRANSCRIBE in costs
        assert costs[STAGE_1_TRANSCRIBE] == 0.0097
        # Cost ledger total computed correctly at finalize
        assert out["finalize"]["total_cost_usd"] == round(0.0097, 4)

    @pytest.mark.asyncio
    async def test_cost_ledger_logged_at_finalize(
        self, monkeypatch, tmp_path, caplog,
    ):
        call_order: list[str] = []
        _install_stage_mocks(monkeypatch, tmp_path, call_order=call_order)

        env = orchestrator._envelope_init({"job_id": 9})
        with caplog.at_level(logging.INFO, logger="pipeline_v2.orchestrator"):
            await _drive_all_stages(env)

        ledger_records = [
            r for r in caplog.records if r.getMessage() == "v2_cost_ledger"
        ]
        assert len(ledger_records) == 1
        record = ledger_records[0]
        assert record.job_id == 9
        # Total = Stage 1's stt_cost_usd
        assert record.total_usd == round(0.0097, 4)


# ====================================================================== #
# E2E cancellation: NonRetriableError between stages                      #
# ====================================================================== #


class TestE2ECancellation:
    @pytest.mark.asyncio
    async def test_cancel_between_stage_2_and_stage_2_5_short_circuits(
        self, monkeypatch, tmp_path,
    ):
        from inngest import NonRetriableError
        call_order: list[str] = []
        _install_stage_mocks(monkeypatch, tmp_path, call_order=call_order)

        # Override cancel check to raise after Stage 2 completes
        # (i.e. it raises on the 4th call -- Stage 2.5's start).
        check_count = [0]

        def _cancel_after_stage_2(jid):
            check_count[0] += 1
            if check_count[0] == 4:    # Stage 2.5's check
                raise NonRetriableError(f"cancelled: job {jid}")
        monkeypatch.setattr(
            orchestrator, "_check_cancelled", _cancel_after_stage_2,
        )

        env = orchestrator._envelope_init({"job_id": 11})
        env = await orchestrator._stage_0_ingest_handler(env)
        env = await orchestrator._stage_1_transcribe_handler(env)
        env = await orchestrator._stage_2_continuity_handler(env)
        with pytest.raises(NonRetriableError, match="cancelled"):
            await orchestrator._stage_2_5_entities_handler(env)

        # Stage 0, 1, 2 ran; Stage 2.5 didn't reach the canonicalizer
        assert call_order == [
            STAGE_0_INGEST,
            STAGE_1_TRANSCRIBE,
            STAGE_2_CONTINUITY,
        ]


# ====================================================================== #
# E2E PermanentSTTError -> NonRetriableError in full pipeline context     #
# ====================================================================== #


class TestE2EPermanentSTTError:
    @pytest.mark.asyncio
    async def test_stage_1_permanent_failure_aborts_pipeline(
        self, monkeypatch, tmp_path,
    ):
        from inngest import NonRetriableError
        from pipeline_v2.stages.stt import PermanentSTTError

        call_order: list[str] = []
        _install_stage_mocks(monkeypatch, tmp_path, call_order=call_order)

        async def _empty_file_failure(audio_path, **kw):
            call_order.append(STAGE_1_TRANSCRIBE)
            raise PermanentSTTError("empty_file: audio is 0 bytes")
        monkeypatch.setattr(orchestrator, "run_stage_1", _empty_file_failure)

        env = orchestrator._envelope_init({"job_id": 1})
        env = await orchestrator._stage_0_ingest_handler(env)
        with pytest.raises(NonRetriableError, match=r"permanent: empty_file"):
            await orchestrator._stage_1_transcribe_handler(env)

        # Pipeline aborted at Stage 1; Stages 2-4 never ran.
        assert STAGE_2_CONTINUITY not in call_order
        assert STAGE_4_RENDER not in call_order


# ====================================================================== #
# E2E PermanentRenderError -> NonRetriableError in full pipeline context  #
# ====================================================================== #


class TestE2EPermanentRenderError:
    @pytest.mark.asyncio
    async def test_stage_4_permanent_failure_aborts_pipeline(
        self, monkeypatch, tmp_path,
    ):
        from inngest import NonRetriableError
        from pipeline_v2.stages.stage_4_render import PermanentRenderError

        call_order: list[str] = []
        _install_stage_mocks(monkeypatch, tmp_path, call_order=call_order)

        class _RaisingRenderer:
            def __init__(self, **kw):
                pass
            def render(self, job_output, timestamp, **kw):
                call_order.append(STAGE_4_RENDER)
                raise PermanentRenderError("disk_full: no space left")
        monkeypatch.setattr(orchestrator, "Stage4Render", _RaisingRenderer)

        env = orchestrator._envelope_init({"job_id": 1})
        env = await orchestrator._stage_0_ingest_handler(env)
        env = await orchestrator._stage_1_transcribe_handler(env)
        env = await orchestrator._stage_2_continuity_handler(env)
        env = await orchestrator._stage_2_5_entities_handler(env)
        env = await orchestrator._stage_3_fanout_handler(env)
        with pytest.raises(NonRetriableError, match=r"permanent render: disk_full"):
            await orchestrator._stage_4_render_handler(env)

        # Finalize never ran because Stage 4 aborted
        assert FINALIZE not in call_order
