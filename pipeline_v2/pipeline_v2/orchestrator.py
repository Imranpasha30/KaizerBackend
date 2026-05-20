"""Inngest orchestrator for V2 pipeline.

ONE top-level function: ``process_video_v2``, triggered by
``video/v2/uploaded`` events. Runs the 7-step sequence end-to-end:

  step_0  stage_0_ingest         FFprobe + NVENC mezzanine
  step_1  stage_1_transcribe     Deepgram / fallback (per env STT_PROVIDER)
  step_2  stage_2_continuity     Gemini 2.5 Pro
  step_3  stage_2_5_entities     Gemini 2.5 Flash
  step_4  stage_3_fanout         3a + 3b + 3c via asyncio.gather
  step_5  stage_4_render         raw cut + compose + bulletin + editor_meta
  step_6  finalize               Job.status='done', _import_clips, cost log

THIS FILE (Step 10.1) defines the function shape with EMPTY step
bodies that pass Pydantic-serialized state through. The actual
stage calls land in Steps 10.2-10.4. The decision matrix is locked
at D-10.1..D-10.15.

Hybrid state passing (D-10.14): each step's INTERNAL handler takes a
Pydantic input + returns a Pydantic output. The handler then
``.model_dump()``s for the dict that Inngest persists. The next
step deserializes via the matching Pydantic model.

Per-step retries (D-10.3): set via ``retries=N`` at function-level
for the outer Inngest retry; each step's internal corrective-retry
layer (Stages 2/2.5/3) handles validation errors before the outer
layer ever sees them.

  Stage 0/1: 3 retries (cheap + flaky)
  Stage 2/2.5/3-fanout: 2 retries (in-step corrective retry already
                                   covers validation)
  Stage 4: 1 retry (expensive; compose_deps cache makes single retry
                    semi-idempotent)
  Finalize: 3 retries (DB transient errors are essentially free)

The function-level ``retries`` keyword sets the DEFAULT for all steps;
per-step overrides happen by raising ``NonRetriableError`` for
permanent failures (PermanentSTTError pattern, D-10.3 refinement).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from inngest import (
    Context,
    NonRetriableError,
    TriggerEvent,
)

from pipeline_v2.inngest_client import get_client
from pipeline_v2.models import (
    Entity,
    ImagePlan,
    JobOutput,
    Metadata,
    ShortsCut,
    Stage0Output,
    Stage1Output,
    Stage2Output,
    Stage2_5Output,
    StageTwoOutput,
)
from pipeline_v2.stages.stage_0_ingest import run_stage_0
from pipeline_v2.stages.stage_2_continuity import (
    Stage2ContinuityEditor,
    assemble_stage_two_output,
)
from pipeline_v2.stages.stage_2_5_entity_canonicalizer import (
    Stage2_5EntityCanonicalizer,
)
from pipeline_v2.stages.stage_3_fanout import Stage3FanOut, Stage3Output
from pipeline_v2.stages.stage_4_render import (
    PermanentRenderError,
    RenderResult,
    Stage4Render,
)
from pipeline_v2.stages.stt import PermanentSTTError, run_stage_1

# Add KaizerBackend/ to sys.path so V1's models + main can be imported
# for the synchronous Job.current_stage write + cancel-flag read.
_KAIZER_BACKEND = Path(__file__).resolve().parents[2]
if str(_KAIZER_BACKEND) not in sys.path:
    sys.path.insert(0, str(_KAIZER_BACKEND))

logger = logging.getLogger("pipeline_v2.orchestrator")


# Default STT provider per D-10.11 (env-var-only for Beta).
DEFAULT_STT_PROVIDER = "deepgram"


# ====================================================================== #
# Step name constants (must match Job.current_stage values per D-10.7)   #
# ====================================================================== #

STAGE_0_INGEST = "stage_0_ingest"
STAGE_1_TRANSCRIBE = "stage_1_transcribe"
STAGE_2_CONTINUITY = "stage_2_continuity"
STAGE_2_5_ENTITIES = "stage_2_5_entities"
STAGE_3_FANOUT = "stage_3_fanout"
STAGE_4_RENDER = "stage_4_render"
FINALIZE = "finalize"

ALL_STAGES: tuple[str, ...] = (
    STAGE_0_INGEST, STAGE_1_TRANSCRIBE, STAGE_2_CONTINUITY,
    STAGE_2_5_ENTITIES, STAGE_3_FANOUT, STAGE_4_RENDER, FINALIZE,
)


# ====================================================================== #
# Helpers shared across step handlers                                     #
# ====================================================================== #


def _open_db_session():
    """Return a SQLAlchemy session for the V1 DB.

    Lazy import: pulled inside the helper so importing orchestrator.py
    in test contexts that don't have the DB doesn't blow up. The V1
    main.py defines the engine + SessionLocal.
    """
    from main import SessionLocal   # V1 module on sys.path
    return SessionLocal()


def _check_cancelled(job_id: int) -> None:
    """Raise NonRetriableError if the user has cancelled this job.

    Called at the START of every step handler (D-10.9 part 3). The
    HTTP cancel endpoint sets ``Job.cancel_requested = True``; this
    check catches it between Inngest steps so we don't burn a 5-min
    Stage 4 render on a cancelled job.

    Defensive: if the DB read fails (transient connection error), log
    a warning and continue -- the cancel-check is best-effort UX, NOT
    data integrity. The HTTP endpoint's SIGTERM on subprocess is the
    hard guarantee.
    """
    if job_id is None or job_id <= 0:
        # Synthetic test job_id or missing field -- skip the check.
        return
    try:
        from models import Job
        session = _open_db_session()
        try:
            job = session.query(Job).filter(Job.id == job_id).first()
            if job is not None and bool(job.cancel_requested):
                raise NonRetriableError(
                    f"cancelled: job {job_id} cancel_requested=True"
                )
        finally:
            session.close()
    except NonRetriableError:
        raise
    except Exception as exc:
        logger.warning(
            "stage cancel-check failed for job %s (continuing): %s",
            job_id, exc,
        )


def _write_current_stage(job_id: int, stage_name: str) -> None:
    """Synchronous fire-and-forget update of ``Job.current_stage``.

    Per D-10.7: the UI surfaces this for V2 job progress on long
    renders. Failure of this write is NON-FATAL -- current_stage is
    UX, not data integrity. Log the warning and continue.
    """
    if job_id is None or job_id <= 0:
        return
    try:
        from models import Job
        session = _open_db_session()
        try:
            session.query(Job).filter(Job.id == job_id).update(
                {"current_stage": stage_name},
                synchronize_session=False,
            )
            session.commit()
        finally:
            session.close()
    except Exception as exc:
        logger.warning(
            "stage current_stage write failed for job %s "
            "(continuing): %s", job_id, exc,
        )


# Per-job in-memory clock so progress lines can show elapsed-since-start.
# Survives only within one worker process; on Inngest retry the elapsed
# resets, which is fine -- the UI already shows wall-clock from Job.started_at.
_PROGRESS_START_TS: dict[int, float] = {}

# Hard cap on Job.log size for V2 jobs so a runaway loop can't fill
# the disk. ~50 KB = ~500 progress lines; well past any healthy run.
_MAX_LOG_BYTES = 50 * 1024


def _append_progress_log(job_id: int, message: str) -> None:
    """Append one progress line to ``Job.log`` so the V1-style UI log
    panel surfaces V2 progress in real time. Backlog item 88 (live
    progress UI parity with V1).

    Format mirrors V1's pipeline output: ``[HH:MM:SS] STEP: <text>``
    so the existing ``_estimate_progress`` heuristic (counts ``STEP``
    markers) gives V2 jobs a non-zero progress bar too. Write is
    synchronous fire-and-forget; failures log a warning and continue
    (this is UX, not data integrity). Caps the log at ``_MAX_LOG_BYTES``
    to prevent runaway growth on retry storms.
    """
    if job_id is None or job_id <= 0:
        return
    try:
        from datetime import datetime
        from models import Job
        session = _open_db_session()
        try:
            job = session.query(Job).filter(Job.id == job_id).first()
            if job is None:
                return
            ts = datetime.now().strftime("%H:%M:%S")
            # Show elapsed-since-handler-start in seconds so the
            # operator sees "Stage 4 started 240s ago" without
            # mental math.
            import time as _time
            start = _PROGRESS_START_TS.setdefault(job_id, _time.time())
            elapsed_s = int(_time.time() - start)
            line = f"[{ts}] STEP (+{elapsed_s}s): {message}\n"
            existing = job.log or ""
            new_log = existing + line
            if len(new_log) > _MAX_LOG_BYTES:
                # Drop the oldest ~1/3 so we keep the recent half +
                # don't blow the cap. Cheap O(N) slice; runs at most
                # once per ~500 progress lines.
                new_log = new_log[-int(_MAX_LOG_BYTES * 2 // 3):]
            job.log = new_log
            session.commit()
        finally:
            session.close()
    except Exception as exc:
        logger.warning(
            "_append_progress_log failed for job %s (continuing): %s",
            job_id, exc,
        )


def _envelope_init(event_data: dict) -> dict:
    """Build the initial step envelope from incoming Inngest event data.

    The envelope shape (passed between handlers; D-10.14 hybrid
    Pydantic/dict pattern):

      {
        "job_id":          int,
        "video_path":      str,
        "language":        str,
        "platform":        str,
        "frame_layout":    str,
        "preset":          dict,
        "user_id":         int | None,
        "out_dir":         str | None,  # caller-supplied, may be empty
        # per-stage Pydantic outputs (dicts; missing until that stage runs)
        "stage_0":         dict | None,
        "stage_1":         dict | None,
        "stage_2":         dict | None,
        "stage_2_5":       dict | None,
        "stage_3":         dict | None,
        "stage_4":         dict | None,
        # cost ledger (D-10.12; logged at finalize)
        "stage_costs":     dict[str, float],   # per-stage USD breakdown
      }
    """
    return {
        "job_id":       int(event_data.get("job_id", 0)),
        "video_path":   event_data.get("video_path", ""),
        "language":     event_data.get("language", "te"),
        "platform":     event_data.get("platform", "full_video_shorts_v2"),
        "frame_layout": event_data.get("frame_layout", "torn_card"),
        "preset":       event_data.get("preset", {}),
        "user_id":      event_data.get("user_id"),
        "out_dir":      event_data.get("out_dir", ""),
        "stage_0":      None,
        "stage_1":      None,
        "stage_2":      None,
        "stage_2_5":    None,
        "stage_3":      None,
        "stage_4":      None,
        "stage_costs":  {},
    }


# ====================================================================== #
# Step handlers (skeletons; bodies filled in Steps 10.2-10.4)            #
# ====================================================================== #


async def _stage_0_ingest_handler(envelope: dict) -> dict:
    """Stage 0: FFprobe + NVENC mezzanine extraction.

    Reads ``video_path`` + ``out_dir`` from the envelope, runs
    ``run_stage_0`` which produces the mezzanine.mp4 + source.mp3 +
    Stage0Output telemetry. Result is stored in envelope["stage_0"].
    """
    job_id = envelope["job_id"]
    _check_cancelled(job_id)
    _write_current_stage(job_id, STAGE_0_INGEST)
    _append_progress_log(job_id, "Stage 0/7 ingest started (ffprobe + NVENC mezzanine)")

    video_path = envelope["video_path"]
    out_dir = envelope.get("out_dir") or ""
    if not out_dir:
        # Default out_dir if caller didn't supply one. Per V1 pattern:
        # output/<platform>/<job_id>/
        out_dir = str(
            _KAIZER_BACKEND / "output" / envelope["platform"]
            / f"job_{job_id}"
        )

    stage_0: Stage0Output = await run_stage_0(video_path, out_dir)

    envelope["out_dir"] = out_dir   # propagate to downstream stages
    envelope["stage_0"] = stage_0.model_dump()
    # Stage 0 has no LLM cost; ffmpeg/ffprobe is free compute.
    envelope["stage_costs"][STAGE_0_INGEST] = 0.0
    _append_progress_log(
        job_id,
        f"Stage 0/7 ingest done (mezzanine + audio extracted)",
    )
    return envelope


async def _stage_1_transcribe_handler(envelope: dict) -> dict:
    """Stage 1: STT dispatcher (Deepgram default).

    D-10.3 refinement: when an STT provider raises ``PermanentSTTError``,
    the handler converts it to Inngest's ``NonRetriableError`` so the
    retry burn is skipped and the job routes straight to the dead-letter
    path. Per D-10.11 the provider name comes from ``KAIZER_STT_PROVIDER``
    env var (default ``"deepgram"``).
    """
    job_id = envelope["job_id"]
    _check_cancelled(job_id)
    _write_current_stage(job_id, STAGE_1_TRANSCRIBE)

    # Stage 0 produced source.mp3 in out_dir; pass it to Stage 1.
    stage_0 = Stage0Output.model_validate(envelope["stage_0"])
    audio_path = stage_0.audio_path
    provider = os.environ.get(
        "KAIZER_STT_PROVIDER", DEFAULT_STT_PROVIDER,
    ).strip() or DEFAULT_STT_PROVIDER
    _append_progress_log(
        job_id, f"Stage 1/7 transcription started ({provider})",
    )

    try:
        stage_1: Stage1Output = await run_stage_1(
            audio_path,
            provider=provider,
            language_hint=envelope.get("language") or None,
            out_dir=envelope["out_dir"],
        )
    except PermanentSTTError as exc:
        # D-10.3: translate to Inngest non-retriable + record on Job.
        # The outer try/except in process_video_v2 (D-10.15) records the
        # final Job.error / status; here we just surface a clear cause.
        logger.warning(
            "stage_1: PermanentSTTError (no Inngest retry): %s", exc,
        )
        _append_progress_log(job_id, f"Stage 1/7 FAILED (permanent): {exc}")
        raise NonRetriableError(f"permanent: {exc}") from exc

    envelope["stage_1"] = stage_1.model_dump()
    envelope["stage_costs"][STAGE_1_TRANSCRIBE] = float(stage_1.stt_cost_usd)
    word_count = len(getattr(stage_1, "words", None) or [])
    _append_progress_log(
        job_id,
        f"Stage 1/7 done ({word_count} words, ${stage_1.stt_cost_usd:.4f})",
    )
    return envelope


async def _stage_2_continuity_handler(envelope: dict) -> dict:
    """Stage 2: Gemini 2.5 Pro continuity editor.

    Builds the full StageTwoOutput (LLM decisions +
    deterministically-reconstructed clean_transcript via
    assemble_stage_two_output). Result is stored in envelope["stage_2"].
    """
    job_id = envelope["job_id"]
    _check_cancelled(job_id)
    _write_current_stage(job_id, STAGE_2_CONTINUITY)
    _append_progress_log(
        job_id, "Stage 2/7 continuity editor started (Gemini 2.5 Pro)",
    )

    stage_1 = Stage1Output.model_validate(envelope["stage_1"])
    editor = Stage2ContinuityEditor()
    decisions: Stage2Output = await editor.transcribe_to_decisions(stage_1)
    full: StageTwoOutput = assemble_stage_two_output(stage_1, decisions)

    envelope["stage_2"] = full.model_dump()
    # Gemini cost tracking is best-effort -- pull token counts off the
    # response only if available. For Beta we log $0.0 here; Step 12
    # can plumb usage_metadata through Stage 2's return.
    envelope["stage_costs"][STAGE_2_CONTINUITY] = 0.0
    cuts_n = len(getattr(full, "full_video_cuts", None) or [])
    skip_n = len(getattr(decisions, "skipped_segments", None) or [])
    _append_progress_log(
        job_id,
        f"Stage 2/7 done ({cuts_n} cuts, {skip_n} skipped segments)",
    )
    return envelope


async def _stage_2_5_entities_handler(envelope: dict) -> dict:
    """Stage 2.5: Gemini 2.5 Flash entity canonicalizer.

    Reads Stage 2's clean_transcript, runs canonicalization, stores
    Stage2_5Output in envelope["stage_2_5"].
    """
    job_id = envelope["job_id"]
    _check_cancelled(job_id)
    _write_current_stage(job_id, STAGE_2_5_ENTITIES)
    _append_progress_log(
        job_id, "Stage 3/7 entity canonicalization started (Gemini Flash)",
    )

    stage_2 = StageTwoOutput.model_validate(envelope["stage_2"])
    canonicalizer = Stage2_5EntityCanonicalizer()
    stage_2_5: Stage2_5Output = await canonicalizer.classify(
        stage_2.clean_transcript,
    )

    envelope["stage_2_5"] = stage_2_5.model_dump()
    envelope["stage_costs"][STAGE_2_5_ENTITIES] = 0.0
    ent_n = len(getattr(stage_2_5, "entities", None) or [])
    _append_progress_log(
        job_id, f"Stage 3/7 done ({ent_n} canonical entities)",
    )
    return envelope


async def _stage_3_fanout_handler(envelope: dict) -> dict:
    """Stage 3 fan-out: 3a (shorts) + 3b (metadata) + 3c (image_plan)
    via Stage3FanOut helper (asyncio.gather under the hood).

    Reads Stage 2's clean_transcript + full_video_cuts and Stage 2.5's
    canonical entities; runs all three Stage 3 sub-stages in parallel
    via ``asyncio.gather`` inside this single Inngest step (saves 3
    control-plane round-trips per D-7.4).
    """
    job_id = envelope["job_id"]
    _check_cancelled(job_id)
    _write_current_stage(job_id, STAGE_3_FANOUT)
    _append_progress_log(
        job_id,
        "Stage 4/7 parallel fan-out started "
        "(shorts + metadata + image plan via Gemini Flash)",
    )

    stage_2 = StageTwoOutput.model_validate(envelope["stage_2"])
    stage_2_5 = Stage2_5Output.model_validate(envelope["stage_2_5"])

    fanout = Stage3FanOut()
    result: Stage3Output = await fanout.run(
        clean=stage_2.clean_transcript,
        full_video_cuts=stage_2.full_video_cuts,
        entities=list(stage_2_5.entities),
    )

    # Stage3Output is a frozen dataclass; serialize each sub-field as
    # Pydantic dump (they ARE Pydantic, the dataclass is just the
    # carrier per Step 7.5).
    envelope["stage_3"] = {
        "shorts_cuts": [sc.model_dump() for sc in result.shorts_cuts],
        "metadata":    result.metadata.model_dump(),
        "image_plan":  result.image_plan.model_dump(),
    }
    # Gemini cost accounting deferred to Step 12.5; log 0.0 for Beta.
    envelope["stage_costs"][STAGE_3_FANOUT] = 0.0
    shorts_n = len(result.shorts_cuts)
    img_n = len(getattr(result.image_plan, "entries", None) or [])
    _append_progress_log(
        job_id,
        f"Stage 4/7 done ({shorts_n} shorts planned, "
        f"{img_n} image-plan entries, metadata ready)",
    )
    return envelope


# ---- V1 _ACTIVE_PROCS bridge for Stage 4 (D-10.9 part 2) ------------


class _V2WorkerProxy:
    """Popen-shaped proxy registered in V1's ``runner._ACTIVE_PROCS``
    so the existing HTTP cancel-job endpoint can SIGKILL the
    Stage 4 worker's FFmpeg descendants on user cancellation.

    Why a proxy instead of a real Popen: Stage 4 doesn't have a
    single FFmpeg subprocess -- V1's compose helpers spawn many
    short-lived FFmpegs through ``subprocess.run``. The proxy
    exposes the same surface ``runner.cancel_job`` expects (.pid,
    .poll, .terminate, .kill, .wait), but:

      * ``terminate`` / ``kill`` are NO-OPS so the worker itself
        is never SIGTERMed (it handles other Inngest steps).
      * ``wait`` raises ``TimeoutExpired`` so ``cancel_job``'s
        "still alive after 5s -> walk descendants + SIGKILL"
        branch fires, which kills all FFmpeg children of the
        worker (the desired behavior).

    Assumes ONE Inngest worker per V2 job (concurrency=1). If
    workers ever handle multiple V2 jobs concurrently, this proxy
    needs to filter descendants by job_id (out of Beta scope;
    flagged as backlog item 52).
    """

    def __init__(self, pid: int):
        self.pid = pid

    def poll(self):
        # Return None to indicate "still running" for the duration
        # of Stage 4. cancel_job() reads this to decide whether to
        # walk descendants.
        return None

    def terminate(self) -> None:
        # NO-OP: never SIGTERM the worker process.
        pass

    def kill(self) -> None:
        # NO-OP: never SIGKILL the worker process.
        pass

    def wait(self, timeout=None):
        # cancel_job calls this with timeout=5 after terminate.
        # Raise TimeoutExpired so the descendant-walk branch fires.
        import subprocess as _sp
        raise _sp.TimeoutExpired(["v2_worker_proxy"], timeout or 0)


def _register_stage_4_with_active_procs(job_id: int) -> Optional[Any]:
    """Register the current worker process as the cancel target for
    this job_id in V1's ``runner._ACTIVE_PROCS``. Returns the proxy
    instance so the caller can deregister in the finally block.

    Defensive: if V1's runner module can't be imported (test env
    without KaizerBackend on sys.path), log a warning and return
    None -- Stage 4 still runs, just without the cancel bridge.
    """
    if job_id is None or job_id <= 0:
        return None
    try:
        import runner       # V1 module on sys.path
        proxy = _V2WorkerProxy(pid=os.getpid())
        runner._register_proc(job_id, proxy)
        logger.info(
            "stage_4: registered worker pid=%d with _ACTIVE_PROCS for "
            "job %d (proxy enables V1 cancel-job tree-kill)",
            proxy.pid, job_id,
        )
        return proxy
    except Exception as exc:
        logger.warning(
            "stage_4: _ACTIVE_PROCS registration failed for job %d "
            "(continuing; mid-Stage-4 cancel will not SIGKILL "
            "FFmpegs): %s", job_id, exc,
        )
        return None


def _deregister_stage_4_from_active_procs(job_id: int) -> None:
    """Counterpart to _register. Defensive: failure logs a warning."""
    if job_id is None or job_id <= 0:
        return
    try:
        import runner
        runner._deregister_proc(job_id)
    except Exception as exc:
        logger.warning(
            "stage_4: _ACTIVE_PROCS deregister failed for job %d "
            "(non-fatal): %s", job_id, exc,
        )


async def _stage_4_render_handler(envelope: dict) -> dict:
    """Stage 4: raw cut + compose + bulletin + editor_meta. Wires
    Stage4Render.render() against the upstream JobOutput.

    Registers the worker process with V1's _ACTIVE_PROCS at entry +
    deregisters at exit, so the existing HTTP cancel-job endpoint can
    SIGKILL the FFmpeg descendants on user cancellation (D-10.9
    two-layer cancel: cooperative check between Inngest steps +
    in-Stage-4 SIGKILL via cancel_job's descendant walk).

    PermanentRenderError handling (Step 10.3 extension to D-10.3):
    catches ffmpeg_not_found / disk_full / source_video_corrupt /
    encoder_unavailable_no_fallback and re-raises as Inngest
    ``NonRetriableError`` so the 12-minute retry burn is avoided on
    systemic failures.
    """
    job_id = envelope["job_id"]
    _check_cancelled(job_id)
    _write_current_stage(job_id, STAGE_4_RENDER)
    _append_progress_log(
        job_id,
        "Stage 5/7 raw shorts cut + Stage 6/7 render started "
        "(this is the longest stage; expect 5-15 min on long bulletins)",
    )

    # Reconstruct JobOutput from the per-stage envelope dicts. The
    # Stage 4 render() takes a JobOutput at its public surface.
    stage_2 = StageTwoOutput.model_validate(envelope["stage_2"])
    stage_2_5 = Stage2_5Output.model_validate(envelope["stage_2_5"])
    stage_3 = envelope["stage_3"]
    shorts_cuts = [ShortsCut.model_validate(d) for d in stage_3["shorts_cuts"]]
    metadata = Metadata.model_validate(stage_3["metadata"])
    image_plan = ImagePlan.model_validate(stage_3["image_plan"])

    job_output = JobOutput(
        stage_two=stage_2,
        canonical_entities=list(stage_2_5.entities),
        shorts_cuts=shorts_cuts,
        metadata=metadata,
        image_plan=image_plan,
    )

    # Output directory: same convention as Stage 0 (output/<platform>/job_<id>/).
    output_dir = envelope.get("out_dir") or str(
        _KAIZER_BACKEND / "output" / envelope["platform"]
        / f"job_{job_id}"
    )
    renderer = Stage4Render(
        output_dir=output_dir,
        video_path=envelope["video_path"],
        preset=envelope["preset"],
        frame_layout=envelope.get("frame_layout", "torn_card"),
        platform=envelope["platform"],
    )

    # Register the worker with _ACTIVE_PROCS BEFORE render starts.
    # Use try/finally so a render exception still deregisters.
    _register_stage_4_with_active_procs(job_id)
    try:
        timestamp = envelope.get("timestamp", "")
        if not timestamp:
            import time as _time
            timestamp = _time.strftime("%Y%m%d_%H%M%S")

        # Stage 4 cancel_check (Step 12.3 Test 2 fix, backlog item
        # 76). Stage 4's render takes ~5 min; without an internal
        # cancel-check the user's cancel sits idle until finalize's
        # cooperative check fires (potentially 5 min later). The
        # callback below re-queries Job.cancel_requested at every
        # sub-phase boundary inside _render_impl and raises
        # NonRetriableError -- the same shape any other Inngest
        # step boundary cancel produces. Propagation:
        #   _render_impl     -> Stage 4 render() classifier
        #   render()         -> _stage_4_render_handler's caller
        #   process_video_v2 -> except Exception -> _mark_job_failed
        #   Inngest          -> run.status=FAILED with cancel slug
        def _stage_4_cancel_check() -> None:
            _check_cancelled(job_id)

        # Progress callback (item 88): Stage4Render.render() invokes
        # this at every sub-phase boundary so the UI log panel sees
        # cut/compose/bulletin progression in real time.
        def _stage_4_progress(msg: str) -> None:
            _append_progress_log(job_id, msg)

        # asyncio.to_thread (item 88 production fix): Stage4Render.render()
        # invokes ffmpeg via blocking subprocess.run all the way down.
        # On Windows uvicorn's --workers flag is a no-op (no fork()), so
        # a single worker process holds the asyncio event loop hostage
        # for the entire 5-15 min render -- /api/jobs/N/status polls
        # time out, /api/inngest invocations queue, the UI freezes.
        # Off-thread it: the event loop stays free, the SDK keeps
        # servicing inbound POST /api/inngest from inngest dev, and the
        # frontend's 2s status poll sees live current_stage + Job.log
        # updates throughout the render. progress_cb + cancel_check
        # are both called from the worker thread; both helpers open
        # fresh SessionLocal()s per call so SQLAlchemy thread-safety
        # is satisfied.
        import asyncio as _asyncio
        result: RenderResult = await _asyncio.to_thread(
            renderer.render,
            job_output,
            timestamp=timestamp,
            cancel_check=_stage_4_cancel_check,
            progress_cb=_stage_4_progress,
        )
    except PermanentRenderError as exc:
        logger.warning(
            "stage_4: PermanentRenderError (no Inngest retry): %s", exc,
        )
        _append_progress_log(
            job_id, f"Stage 5-6/7 FAILED (permanent): {exc}",
        )
        raise NonRetriableError(f"permanent render: {exc}") from exc
    finally:
        _deregister_stage_4_from_active_procs(job_id)

    envelope["stage_4"] = {
        "shorts_editor_meta_path":   result.shorts_editor_meta_path,
        "bulletin_editor_meta_path": result.bulletin_editor_meta_path,
        "composed_shorts_count":     len(result.composed_shorts),
        "bulletin":                  result.bulletin,
    }
    # Stage 4 has no LLM cost; render compute is "free" in the cost
    # ledger sense (it's the cost of the worker, not API).
    envelope["stage_costs"][STAGE_4_RENDER] = 0.0
    _append_progress_log(
        job_id,
        f"Stage 5-6/7 render done "
        f"({len(result.composed_shorts)} shorts + bulletin)",
    )
    return envelope


async def _finalize_handler(envelope: dict) -> dict:
    """Finalize: write Job.status='done' + _import_clips + cost ledger log.

    Per D-10.4 / D-10.7 / D-10.12:
      * Clear ``Job.current_stage`` (set to NULL — pipeline done,
        no current step).
      * Update ``Job.status = 'done'`` + ``Job.finished_at = now``.
      * Set ``Job.output_dir`` so V1's runner.py can find clips.
      * Run V1's ``_import_clips`` against the editor_meta.json Stage 4
        wrote (the Step 8 adapter ensures it's V1-shape-compatible).
      * Log the cost ledger via structured ``logger.info`` so the log
        aggregator can compute per-job spend.

    All DB writes are best-effort: if the DB is transiently down at
    finalize time, the Inngest function still returns success and
    the next runner restart's progress recheck (or the cancel
    endpoint) will see the inconsistent state. We avoid raising
    because the actual work IS done -- editor_meta.json + clips +
    bulletin are on disk; failing here would have Inngest retry the
    whole pipeline, which would be wasteful.
    """
    job_id = envelope["job_id"]
    _check_cancelled(job_id)
    _write_current_stage(job_id, FINALIZE)
    _append_progress_log(
        job_id, "Stage 7/7 finalize started (importing clips + cost ledger)",
    )

    stage_4 = envelope.get("stage_4") or {}
    output_dir = envelope.get("out_dir", "")

    # Cost ledger (D-10.12): log once at finalize so the aggregator
    # has the full per-stage breakdown + total.
    stage_costs = envelope.get("stage_costs", {})
    total_cost = sum(float(v) for v in stage_costs.values() if v is not None)
    logger.info(
        "v2_cost_ledger",
        extra={
            "job_id":      job_id,
            "platform":    envelope.get("platform"),
            "stage_costs": stage_costs,
            "total_usd":   round(total_cost, 4),
            # Surface a few stage-4-specific numbers ops want at a glance:
            "shorts_count": stage_4.get("composed_shorts_count", 0),
            "bulletin_duration_s": (
                (stage_4.get("bulletin") or {}).get("duration_s", 0.0)
            ),
        },
    )

    # DB writes: status=done + finished_at + output_dir + clear current_stage
    try:
        from datetime import datetime, timezone
        from models import Job
        session = _open_db_session()
        try:
            updates = {
                "status":        "done",
                "finished_at":   datetime.now(timezone.utc),
                "current_stage": None,    # D-10.7: clear progress at finalize
            }
            if output_dir:
                updates["output_dir"] = output_dir
            session.query(Job).filter(Job.id == job_id).update(
                updates, synchronize_session=False,
            )
            session.commit()
        finally:
            session.close()
    except Exception as exc:
        logger.warning(
            "finalize: DB status='done' update failed for job %s "
            "(non-fatal -- the pipeline IS done; runner restart will "
            "recheck): %s", job_id, exc,
        )

    # _import_clips: reuse V1's editor_meta.json reader to create
    # Clip rows in the DB. The Stage 8 adapter wrote a V1-compatible
    # editor_meta.json; this importer doesn't need V2 awareness.
    #
    # V2's compound platform produces TWO editor_metas (shorts pass +
    # bulletin pass). The original Step 10 finalize only imported the
    # shorts side, leaving the rendered bulletin orphaned on disk with
    # no Clip row. Backlog item 91: import BOTH passes so the bulletin
    # appears in JobDetail's clips grid alongside the shorts.
    imported_clip_count = 0
    try:
        import runner       # V1 module on sys.path
        from models import Job, Clip
        session = _open_db_session()
        try:
            job = session.query(Job).filter(Job.id == job_id).first()
            if job is not None:
                # Pass 1 of 2: shorts editor_meta -> N short Clip rows
                shorts_meta = stage_4.get("shorts_editor_meta_path")
                if shorts_meta:
                    try:
                        runner._import_clips(
                            job, session, meta_override=shorts_meta,
                        )
                    except Exception as e_short:
                        logger.warning(
                            "finalize: shorts _import_clips failed for "
                            "job %s (continuing to bulletin): %s",
                            job_id, e_short,
                        )

                # Pass 2 of 2: bulletin editor_meta -> 1 bulletin Clip row
                bulletin_meta = stage_4.get("bulletin_editor_meta_path")
                if bulletin_meta:
                    try:
                        runner._import_clips(
                            job, session, meta_override=bulletin_meta,
                        )
                    except Exception as e_bull:
                        logger.warning(
                            "finalize: bulletin _import_clips failed for "
                            "job %s (continuing): %s",
                            job_id, e_bull,
                        )

                # Count after both passes.
                imported_clip_count = (
                    session.query(Clip).filter(Clip.job_id == job_id).count()
                )
        finally:
            session.close()
    except Exception as exc:
        logger.warning(
            "finalize: _import_clips failed for job %s "
            "(non-fatal -- editor_meta.json IS on disk): %s",
            job_id, exc,
        )

    envelope["finalize"] = {
        "status":              "done",
        "total_cost_usd":      round(total_cost, 4),
        "imported_clip_count": imported_clip_count,
    }
    envelope["stage_costs"][FINALIZE] = 0.0
    logger.info(
        "finalize: job_id=%s status=done shorts=%d clips_imported=%d "
        "total_cost_usd=$%.4f",
        job_id, stage_4.get("composed_shorts_count", 0),
        imported_clip_count, total_cost,
    )
    _append_progress_log(
        job_id,
        f"Stage 7/7 finalize done -- pipeline complete "
        f"({imported_clip_count} clips, ${total_cost:.2f})",
    )
    # Drop the per-job clock so a future re-submission with the same
    # job_id starts fresh (not strictly needed because job_ids are
    # monotonic, but harmless and tidies up memory).
    _PROGRESS_START_TS.pop(job_id, None)
    return envelope


# ====================================================================== #
# Top-level Inngest function (D-10.1 + D-10.13)                          #
# ====================================================================== #


# Default retries=2 at the function level (D-10.3 base; per-step
# overrides use NonRetriableError to escape early).
_inngest = get_client()


async def _on_v2_failure(ctx: Context) -> None:
    """Inngest ``on_failure`` hook (backlog item 89).

    Fires after ``process_video_v2`` exhausts all retries. This is the
    canonical place to flip ``Job.status='failed'`` because:

      1. The SDK's flow-control BaseException sentinels exit the main
         function body cleanly on retry without my ``except Exception``
         block ever firing -- so the main body's terminal-failure path
         is unreliable on Windows + Selector-loop NotImplementedError-
         style failures (the case that surfaced job 37's empty
         ``Job.error`` + lingering ``status='running'``).
      2. ``on_failure`` runs as a SEPARATE Inngest invocation AFTER
         the main function has terminated. Inngest passes the
         ``inngest/function.failed`` event whose data carries the
         original triggering event + a structured error dict.

    Defensive: swallows ALL exceptions (we are the LAST line of
    defense; if THIS fails the user will see status='running' forever,
    but at least the worker doesn't crash and other jobs continue).
    """
    try:
        evt = getattr(ctx, "event", None)
        if evt is None:
            return
        data = getattr(evt, "data", None) or {}
        # Inngest's failure event nests the original event under "event"
        # and the error under "error". Both fields are dicts.
        orig_event = data.get("event") or {}
        orig_data = (orig_event.get("data") if isinstance(orig_event, dict) else {}) or {}
        job_id = orig_data.get("job_id")
        # Fall back to top-level data["job_id"] if the failure event
        # shape is flat (some SDK versions don't nest).
        if job_id is None:
            job_id = data.get("job_id")
        if job_id is None:
            logger.warning(
                "_on_v2_failure: could not extract job_id from event data; "
                "keys=%s",
                list(data.keys())[:10] if isinstance(data, dict) else None,
            )
            return
        error_dict = data.get("error") or {}
        # Also surface in the progress log so the user sees a clear
        # terminal line in the UI panel.
        _append_progress_log(
            int(job_id),
            f"PIPELINE FAILED: "
            f"{(error_dict.get('name') if isinstance(error_dict, dict) else None) or 'Error'}"
            f": "
            f"{(error_dict.get('message') if isinstance(error_dict, dict) else None) or '(no message)'}",
        )
        _mark_job_failed_from_event(int(job_id), error_dict)
    except Exception as exc:
        logger.error(
            "_on_v2_failure last-resort hook crashed (Job.status may not "
            "flip to failed): %s", exc,
        )


@_inngest.create_function(
    fn_id="process-video-v2",
    trigger=TriggerEvent(event="video/v2/uploaded"),
    retries=2,
    on_failure=_on_v2_failure,
)
async def process_video_v2(ctx: Context) -> dict:
    """Top-level orchestrator: 7 sequential Inngest steps.

    Inngest SDK 0.5.18 calls the handler with a single ``Context``
    argument; ``step`` is reached via ``ctx.step``. (Older versions
    passed ``step`` as a second positional arg -- see backlog item
    72 for the signature evolution.)

    Event data contract (sent by runner.py's V2 branch):

      {
        "job_id":      int,         # DB Job.id (Integer per D-10.8)
        "video_path":  str,         # absolute path on the worker
        "language":    str,         # ISO 639-1 (te|hi|en|...)
        "platform":    str,         # "full_video_shorts_v2"
        "frame_layout": str,        # "torn_card" | "split_frame" | ...
        "preset":      dict,        # render preset (caller-supplied)
        "user_id":     int | None,
      }

    Idempotency key (set by the dispatcher's send call, D-10.10):
      ``id=f"job-{job_id}"`` -- duplicate sends within Inngest's
      24h window are deduplicated.

    Outer try/except (D-10.15): on terminal failure the dispatcher's
    on_failure handler (separate Inngest function, future step)
    writes Job.status='failed' + Job.error. This top-level coroutine
    re-raises so Inngest sees the failure for its retry policy + UI.
    """
    event_data = ctx.event.data
    job_id = event_data.get("job_id")
    # SDK 0.5.18: step accessor lives on Context; bind locally so
    # the existing ``step.run(...)`` call sites below stay unchanged.
    step = ctx.step
    logger.info(
        "process_video_v2 starting: job_id=%s video_path=%s",
        job_id, event_data.get("video_path"),
    )

    # Build initial envelope; each step takes + returns it (D-10.14).
    envelope = _envelope_init(event_data)

    # ---- D-10.15: outer try/except for terminal-failure DB write ---
    # On any unhandled exception (including NonRetriableError after
    # final retry), write Job.status='failed' + Job.error before
    # re-raising. Inngest's UI sees the raised exception; our DB
    # captures the human-readable summary the Kaizer UI surfaces.
    try:
        # ---- Step sequence (each step is durable + checkpointed) ----
        # Inngest 0.5.18 NOTE: Step.run() has NO per-step retries kwarg.
        # Per-step retry counts cannot be set in this SDK version. The
        # function-level retries=2 above applies to ALL steps. The
        # D-10.3 PermanentSTTError + PermanentRenderError patterns +
        # NonRetriableError still give us early-exit on permanent
        # failures (no retry burn). Backlog item 50: when Inngest
        # Python SDK adds per-step retries, plumb the D-10.3 table.
        envelope = await step.run(
            STAGE_0_INGEST, _stage_0_ingest_handler, envelope,
        )
        envelope = await step.run(
            STAGE_1_TRANSCRIBE, _stage_1_transcribe_handler, envelope,
        )
        envelope = await step.run(
            STAGE_2_CONTINUITY, _stage_2_continuity_handler, envelope,
        )
        envelope = await step.run(
            STAGE_2_5_ENTITIES, _stage_2_5_entities_handler, envelope,
        )
        envelope = await step.run(
            STAGE_3_FANOUT, _stage_3_fanout_handler, envelope,
        )
        envelope = await step.run(
            STAGE_4_RENDER, _stage_4_render_handler, envelope,
        )
        envelope = await step.run(
            FINALIZE, _finalize_handler, envelope,
        )
    # IMPORTANT: catch Exception, not BaseException.
    # Inngest SDK 0.5.18 uses BaseException-subclassed flow-control
    # exceptions (ResponseInterrupt, SkipInterrupt, NestedStepInterrupt)
    # which are raised after every step.run() yield to signal step
    # completion. These MUST propagate to the SDK executor uncaught.
    # Catching BaseException here would intercept them, falsely marking
    # the Job as failed even though Inngest sees the run as completed.
    # See backlog item 74 for the empirical discovery (Step 12.2b run #5).
    except Exception as exc:
        # Terminal failure path. Write Job.status='failed' + Job.error
        # then RE-RAISE so Inngest's retry policy + UI see the failure.
        _append_progress_log(
            job_id, f"PIPELINE FAILED: {type(exc).__name__}: {exc}",
        )
        _mark_job_failed(job_id, exc)
        raise

    logger.info(
        "process_video_v2 done: job_id=%s", job_id,
    )
    return envelope


def _format_error_text(exc: BaseException) -> str:
    """Build the Job.error text for a terminal-failure exception.

    Inngest's ``StepError`` is a userland-error WRAPPER -- its public
    ``__str__`` returns just ``.message`` which the original exception
    may have left empty (e.g. ``NotImplementedError()`` from
    asyncio on Windows + Selector loop). The ``.name`` + ``.stack``
    properties carry the useful diagnostic info. Unpack them when
    present; fall back to the regular format otherwise. Capped at
    4 KB so the TEXT column stays sane.
    """
    import traceback
    max_total = 4000
    # Inngest StepError -- pull richer fields
    if (type(exc).__module__.startswith("inngest")
            and type(exc).__name__ in {"StepError", "Error"}):
        name = getattr(exc, "name", None) or "UnknownStepError"
        msg = getattr(exc, "message", None)
        if msg is None:
            msg = str(exc)
        stack = getattr(exc, "stack", None) or ""
        prefix = f"{name}: {msg}\n"
        tb = stack
    else:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        prefix = f"{type(exc).__name__}: {exc}\n"
    if len(prefix) + len(tb) > max_total:
        allowed_tb = max_total - len(prefix) - len("... [truncated]\n")
        if allowed_tb > 0:
            tb = "... [truncated]\n" + tb[-allowed_tb:]
        else:
            tb = ""
    return (prefix + tb)[:max_total]


def _mark_job_failed(job_id: int, exc: BaseException) -> None:
    """Per D-10.15: synchronous DB write on terminal failure.

    Sets:
      ``Job.status = 'failed'``
      ``Job.error  = "<exc_class>: <message>\\n<traceback truncated>"``
      ``Job.finished_at = now()``
      ``Job.current_stage = None``    (clear progress on failure too)

    For ``inngest.StepError`` (the SDK's retry-exhausted wrapper), the
    helper unpacks the underlying userland error's ``name`` /
    ``message`` / ``stack`` so the user-facing ``Job.error`` is
    meaningful instead of an empty ``StepError:`` line. See backlog
    item 89.

    Best-effort: DB write failures here log a warning and continue
    (re-raise above takes care of Inngest's path).
    """
    if job_id is None or job_id <= 0:
        return
    try:
        from datetime import datetime, timezone
        from models import Job
        error_text = _format_error_text(exc)

        session = _open_db_session()
        try:
            session.query(Job).filter(Job.id == job_id).update(
                {
                    "status":        "failed",
                    "error":         error_text,
                    "finished_at":   datetime.now(timezone.utc),
                    "current_stage": None,
                },
                synchronize_session=False,
            )
            session.commit()
            logger.error(
                "v2 terminal failure: job_id=%s exc_class=%s msg=%s",
                job_id, type(exc).__name__, str(exc)[:200],
            )
        finally:
            session.close()
    except Exception as db_exc:
        logger.warning(
            "_mark_job_failed: DB write failed for job %s (non-fatal "
            "-- Inngest will still surface the original failure): %s",
            job_id, db_exc,
        )


def _mark_job_failed_from_event(job_id: int, error_dict: dict) -> None:
    """Inngest ``on_failure`` hook variant: takes the failure payload
    Inngest's ``inngest/function.failed`` event carries (a dict with
    ``name`` / ``message`` / ``stack``) instead of a live exception.

    Backlog item 89: even when my ``except Exception`` inside
    process_video_v2 doesn't fire (because the SDK's flow-control
    BaseException exits cleanly without reaching my catch), Inngest
    still calls ``on_failure`` after retries exhaust. This is the
    LAST-RESORT hook that guarantees ``Job.status='failed'`` regardless
    of what happens inside the function body.
    """
    if job_id is None or job_id <= 0:
        return
    name = (error_dict.get("name") or "UnknownError") if isinstance(error_dict, dict) else "UnknownError"
    msg = (error_dict.get("message") or "") if isinstance(error_dict, dict) else ""
    stack = (error_dict.get("stack") or "") if isinstance(error_dict, dict) else ""
    error_text = (f"{name}: {msg}\n{stack}")[:4000]
    try:
        from datetime import datetime, timezone
        from models import Job
        session = _open_db_session()
        try:
            session.query(Job).filter(Job.id == job_id).update(
                {
                    "status":        "failed",
                    "error":         error_text,
                    "finished_at":   datetime.now(timezone.utc),
                    "current_stage": None,
                },
                synchronize_session=False,
            )
            session.commit()
            logger.error(
                "v2 on_failure marked job failed: job_id=%s name=%s",
                job_id, name,
            )
        finally:
            session.close()
    except Exception as db_exc:
        logger.warning(
            "_mark_job_failed_from_event: DB write failed for job %s "
            "(non-fatal): %s", job_id, db_exc,
        )
