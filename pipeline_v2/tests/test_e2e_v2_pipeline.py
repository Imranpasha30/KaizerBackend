"""V2 pipeline end-to-end tests against real APIs (Step 12).

This file accumulates four sub-tests across Step 12 sub-steps:

  12.2a -- Direct-orchestrator-drive E2E (real APIs, stubbed Inngest)
  12.2b -- Same E2E via Inngest Dev Server (real Inngest + real APIs)
  12.3  -- Mid-render cancellation (real cancel_job, psutil orphan check)
  12.4  -- Idempotency fixture comparison (Event(id=...) shape pin)

ALL tests in this module are marked ``@pytest.mark.integration`` and
SKIP unless ``KAIZER_RUN_INTEGRATION_TESTS=1`` is set. This matches
the convention from ``test_stage_4_render_integration.py``.

Required env (when integration tests are enabled):

  KAIZER_RUN_INTEGRATION_TESTS=1     unlock the integration mark
  GEMINI_API_KEY=...                 Stage 2 / 2.5 / 3 (all Gemini)
  GROQ_API_KEY=...                   Stage 1 STT (default whisper-groq)
                                     OR DEEPGRAM_API_KEY for deepgram path
  KAIZER_TEST_VIDEO=/abs/path        override default fallback below
  KAIZER_STT_PROVIDER=whisper-groq   per D-12.3 (free tier first)

Default fixture video path: ``e:/kaizer new data training/videos/test.mp4``
(the canonical Bandi Bhagirath fixture used since Step 5.0). Override
via ``KAIZER_TEST_VIDEO``.

For Step 12.2b only: ``inngest`` CLI must be on PATH and ``inngest dev``
must be running on http://localhost:8288. The test_inngest_e2e class
auto-detects this via ``shutil.which("inngest")`` and skips when
missing -- so a dev box without Inngest installed runs everything
except 12.2b unaffected.

Cost budget (D-12.9): $5 total for Step 12, $2 single-run warning. The
real-APIs path is expensive; do NOT enable these tests in CI without
explicit cost-budget approval.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest


# ====================================================================== #
# Module-level skip + env detection                                       #
# ====================================================================== #

RUN_INTEGRATION = os.environ.get("KAIZER_RUN_INTEGRATION_TESTS", "").strip() == "1"

# Default fixture path: matches the canonical project layout
# (e:/kaizer new data training/videos/test.mp4 -- the Bandi Bhagirath
# 12-minute Telugu source mined for prompts since Step 5.0).
# .parents[N] resolution from this file:
#   [0] tests/
#   [1] pipeline_v2/
#   [2] KaizerBackend/
#   [3] kaizer/
#   [4] <parent repo root> -- where videos/test.mp4 lives
_DEFAULT_TEST_VIDEO = (
    Path(__file__).resolve().parents[4] / "videos" / "test.mp4"
)
TEST_VIDEO_PATH = Path(
    os.environ.get("KAIZER_TEST_VIDEO", "").strip()
    or str(_DEFAULT_TEST_VIDEO)
)


pytestmark = pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason=(
        "V2 E2E integration tests skipped. Set "
        "KAIZER_RUN_INTEGRATION_TESTS=1 (+ GEMINI_API_KEY + "
        "GROQ_API_KEY) to enable. See module docstring."
    ),
)


# ====================================================================== #
# Shared fixtures + env-check helpers                                     #
# ====================================================================== #


@pytest.fixture(scope="module")
def test_video_path() -> Path:
    """Resolve + validate the source video. Skip the whole module
    if it can't be found (avoids surprising late failures during
    Stage 0 mezzanine extraction).
    """
    if not TEST_VIDEO_PATH.is_file():
        pytest.skip(
            f"Test video not found at {TEST_VIDEO_PATH}. "
            f"Set KAIZER_TEST_VIDEO=/abs/path/test.mp4 or place the "
            f"file at the default location."
        )
    return TEST_VIDEO_PATH


@pytest.fixture
def required_api_keys() -> dict[str, str]:
    """Verify the required API keys are present. Skip the whole
    test class if any are missing.
    """
    keys = {
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", "").strip(),
        # Either Groq (default whisper-groq) OR Deepgram must be set.
        # Stage 1 selects via KAIZER_STT_PROVIDER (D-12.3 default
        # whisper-groq).
    }
    stt_provider = os.environ.get(
        "KAIZER_STT_PROVIDER", "whisper-groq",
    ).strip()
    if stt_provider == "whisper-groq":
        keys["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "").strip()
    elif stt_provider == "deepgram":
        keys["DEEPGRAM_API_KEY"] = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    elif stt_provider == "assemblyai":
        keys["ASSEMBLYAI_API_KEY"] = os.environ.get("ASSEMBLYAI_API_KEY", "").strip()

    missing = [k for k, v in keys.items() if not v]
    if missing:
        pytest.skip(
            f"Required API keys missing: {missing}. "
            f"Add to .env or export before running."
        )
    return keys


@pytest.fixture(scope="module")
def diag_dir() -> Path:
    """Create a fresh diag subdirectory for this test run + return it.

    Naming: ``<YYYYMMDD_HHMMSS>__<mode>`` per fixtures/step12_diag/README.md.
    The mode suffix is added by individual tests via subdirectory.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(__file__).resolve().parent / "fixtures" / "step12_diag" / ts
    root.mkdir(parents=True, exist_ok=True)
    return root


def _inngest_cli_available() -> bool:
    """True if the inngest CLI binary is on PATH."""
    return shutil.which("inngest") is not None


def _inngest_dev_server_listening(port: int = 8288) -> bool:
    """True if a process is listening on the Inngest Dev Server's
    default port. We don't ping the actual /health endpoint here
    because that adds a network call to test collection; a TCP
    connect probe is enough.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            s.connect(("127.0.0.1", port))
            return True
    except OSError:
        return False


# ====================================================================== #
# Step 12.1: setup sanity tests                                           #
# ====================================================================== #
#
# These tests verify the INFRASTRUCTURE the real E2E tests rely on.
# They DO NOT invoke real APIs or run the pipeline. Their job is to
# catch fixture-path / env / import failures before the expensive
# tests run.


class TestE2EInfrastructure:
    """Sanity checks that the test infrastructure resolves correctly.

    These run only when KAIZER_RUN_INTEGRATION_TESTS=1 -- but they
    don't burn API budget, so they're a cheap pre-flight to confirm
    fixture paths + env vars + V2 import surface are all reachable
    before kicking off 12.2a.
    """

    def test_test_video_exists(self, test_video_path: Path):
        assert test_video_path.is_file()
        # Sanity: video file should be larger than 100KB
        assert test_video_path.stat().st_size > 100_000

    def test_diag_dir_created(self, diag_dir: Path):
        assert diag_dir.is_dir()
        # The dir name carries the timestamp so subsequent runs don't collide
        assert diag_dir.name.count("_") >= 1

    def test_v2_orchestrator_importable(self):
        # Confirm the import surface that 12.2a / 12.2b will use.
        from pipeline_v2.orchestrator import (
            process_video_v2,
            _stage_0_ingest_handler,
            _stage_1_transcribe_handler,
            _stage_2_continuity_handler,
            _stage_2_5_entities_handler,
            _stage_3_fanout_handler,
            _stage_4_render_handler,
            _finalize_handler,
            _envelope_init,
        )
        assert process_video_v2 is not None

    def test_v2_render_importable(self):
        from pipeline_v2.stages.stage_4_render import (
            Stage4Render, PermanentRenderError,
        )
        assert Stage4Render is not None

    def test_psutil_available_for_cancellation_test(self):
        """D-12.10 pushback: the cancellation sub-test (Step 12.3)
        uses psutil to walk the worker's descendants + verify no
        orphan FFmpeg processes survive cancel_job. Without psutil,
        the highest-value assertion in that test is unavailable.
        """
        import psutil
        # Smoke check: querying the current process's children works.
        # We don't assert any specific count here -- just that the
        # API surface exists + returns a list.
        children = psutil.Process(os.getpid()).children(recursive=True)
        assert isinstance(children, list)

    def test_required_api_keys_present(self, required_api_keys: dict):
        # required_api_keys fixture already SKIPs on missing; this
        # test exists so a missing-key failure is attributed to a
        # specific test name in pytest output rather than buried in
        # the first real E2E test's setup.
        assert all(required_api_keys.values())


# ====================================================================== #
# Placeholder hooks for Step 12.2a / 12.2b / 12.3 / 12.4                  #
# ====================================================================== #
#
# The real E2E test classes land in subsequent commits. This block
# pins their class names + skip patterns so the test runner has
# something to enumerate at collection time, and so a future
# maintainer running pytest --collect-only sees the planned shape.


class TestE2EDirectDrive_12_2a:
    """Step 12.2a -- direct orchestrator drive (Inngest stubbed).

    Drives the 7 V2 step handlers in the exact order
    ``process_video_v2`` invokes them, but with the Inngest dispatch
    stubbed (D-12.6). Real Deepgram/Groq + real Gemini + real FFmpeg
    + real test.mp4. Validates the 7 D-12.4 acceptance checks +
    captures diagnostic artifacts to ``fixtures/step12_diag/<ts>/direct/``.

    Cost per run: ~$0.70 (Gemini Pro at Stage 2 dominates). Time:
    ~5-10 min on a midrange CPU/GPU.
    """

    @pytest.fixture
    def in_memory_db(self, monkeypatch):
        """In-memory SQLite session + a Job row that finalize updates.

        Per D-12.7: real DB schema (Base.metadata.create_all) so
        Stage 4 + finalize exercise the actual column types, but
        the DB itself is :memory: so the test doesn't pollute the
        dev sqlite file.
        """
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        # Import V1's models (puts KaizerBackend on sys.path).
        from pipeline_v2 import orchestrator as _orch   # noqa: F401
        import models as v1_models

        engine = create_engine("sqlite:///:memory:")
        v1_models.Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine)

        session = SessionLocal()
        try:
            job = v1_models.Job(
                id=1, platform="full_video_shorts_v2",
                video_name="test.mp4",
                status="running",
                cancel_requested=False,
            )
            session.add(job)
            session.commit()
        finally:
            session.close()

        # Monkeypatch orchestrator._open_db_session to return a fresh
        # session each time (the real helper opens-+-closes per call).
        from pipeline_v2 import orchestrator as _orch
        monkeypatch.setattr(_orch, "_open_db_session", SessionLocal)

        return {"engine": engine, "SessionLocal": SessionLocal}

    @pytest.mark.asyncio
    async def test_full_pipeline_real_apis(
        self,
        test_video_path: Path,
        required_api_keys: dict,
        in_memory_db: dict,
        diag_dir: Path,
        tmp_path: Path,
        monkeypatch,
    ):
        from pipeline_v2 import orchestrator

        # Per-test diag subdir
        run_dir = diag_dir.parent / f"{diag_dir.name}__direct"
        run_dir.mkdir(parents=True, exist_ok=True)

        # D-12.6: Inngest dispatch stubbed for 12.2a.
        dispatch_calls: list[dict] = []
        monkeypatch.setattr(
            orchestrator, "_check_cancelled", lambda jid: None,
        )

        # Output dir lives under tmp_path so each test run is isolated.
        output_dir = tmp_path / "out"
        output_dir.mkdir(parents=True, exist_ok=True)

        env = orchestrator._envelope_init({
            "job_id":       1,
            "video_path":   str(test_video_path),
            "language":     "te",
            "platform":     "full_video_shorts_v2",
            "frame_layout": "torn_card",
            "preset": {
                "label":  "Full Video + Shorts (V2 Beta)",
                "width":  1080, "height": 1920,
                "min_dur": 15, "max_dur": 60, "ideal_dur": 45,
                "vertical": True,
            },
            "out_dir": str(output_dir),
        })

        run_started = time.perf_counter()
        checks: dict[str, str] = {}

        # ---- Stage 0 ----
        env = await orchestrator._stage_0_ingest_handler(env)
        stage_0 = env["stage_0"]
        (run_dir / "stage_0_output.json").write_text(
            json.dumps(stage_0, indent=2), encoding="utf-8",
        )
        mezz = Path(stage_0["mezzanine_path"])
        checks["stage_0_mezzanine"] = (
            "PASS" if mezz.is_file() and mezz.stat().st_size > 100_000 else "FAIL"
        )
        assert checks["stage_0_mezzanine"] == "PASS", (
            f"stage_0 mezzanine missing or <100KB: {mezz}"
        )

        # ---- Stage 1 ----
        env = await orchestrator._stage_1_transcribe_handler(env)
        stage_1 = env["stage_1"]
        # Trim word array before dumping (full transcript can be ~250KB).
        # Keep the structure but trim the word list to 20 + length-marker.
        stage_1_dump = json.loads(json.dumps(stage_1))
        word_count = len(stage_1_dump["transcript"]["words"])
        stage_1_dump["transcript"]["_word_count"] = word_count
        stage_1_dump["transcript"]["words"] = (
            stage_1_dump["transcript"]["words"][:20]
        )
        (run_dir / "stage_1_transcript.json").write_text(
            json.dumps(stage_1_dump, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        checks["stage_1_word_array"] = (
            "PASS" if word_count >= 100 else "FAIL"
        )
        assert checks["stage_1_word_array"] == "PASS", (
            f"stage_1 only produced {word_count} words; "
            f"expected >=100 for a 12-min video"
        )

        # ---- Stage 2 ----
        env = await orchestrator._stage_2_continuity_handler(env)
        stage_2 = env["stage_2"]
        # Trim clean_transcript words for the diag dump
        s2_dump = json.loads(json.dumps(stage_2))
        s2_dump["clean_transcript"]["_word_count"] = len(
            s2_dump["clean_transcript"]["words"]
        )
        s2_dump["clean_transcript"]["words"] = (
            s2_dump["clean_transcript"]["words"][:20]
        )
        (run_dir / "stage_2_decisions.json").write_text(
            json.dumps(s2_dump, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        checks["stage_2_decisions"] = (
            "PASS" if stage_2["full_video_cuts"]
                  and stage_2["clean_transcript"]["words"] else "FAIL"
        )
        assert checks["stage_2_decisions"] == "PASS"

        # ---- Stage 2.5 ----
        env = await orchestrator._stage_2_5_entities_handler(env)
        stage_2_5 = env["stage_2_5"]
        (run_dir / "stage_2_5_entities.json").write_text(
            json.dumps(stage_2_5, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        n_entities = len(stage_2_5["entities"])
        valid_types = {"PERSON", "ORG", "PLACE", "EVENT", "OTHER"}
        types_ok = all(e["type"] in valid_types for e in stage_2_5["entities"])
        checks["stage_2_5_entities"] = (
            "PASS" if 1 <= n_entities <= 6 and types_ok else "FAIL"
        )
        assert checks["stage_2_5_entities"] == "PASS", (
            f"entities={n_entities} types_ok={types_ok}"
        )

        # ---- Stage 3 fan-out ----
        env = await orchestrator._stage_3_fanout_handler(env)
        stage_3 = env["stage_3"]
        (run_dir / "stage_3_output.json").write_text(
            json.dumps(stage_3, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        n_shorts = len(stage_3["shorts_cuts"])
        durations_ok = all(
            15.0 <= (s["end_sec"] - s["start_sec"]) <= 60.0
            for s in stage_3["shorts_cuts"]
        )
        checks["stage_3_fanout"] = (
            "PASS" if 3 <= n_shorts <= 10 and durations_ok else "FAIL"
        )
        assert checks["stage_3_fanout"] == "PASS", (
            f"shorts={n_shorts} (need 3-10) durations_ok={durations_ok}"
        )

        # ---- Stage 4 render ----
        env = await orchestrator._stage_4_render_handler(env)
        stage_4 = env["stage_4"]
        (run_dir / "stage_4_result.json").write_text(
            json.dumps(stage_4, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # Verify editor_meta.json files + a sample clip + bulletin
        shorts_meta_path = Path(stage_4["shorts_editor_meta_path"])
        bulletin_meta_path = Path(stage_4["bulletin_editor_meta_path"])
        bulletin_mp4_path = Path(stage_4["bulletin"]["bulletin_path"])
        shorts_meta_ok = (
            shorts_meta_path.is_file()
            and shorts_meta_path.stat().st_size > 1_000
        )
        bulletin_meta_ok = (
            bulletin_meta_path.is_file()
            and bulletin_meta_path.stat().st_size > 100
        )
        bulletin_mp4_ok = (
            bulletin_mp4_path.is_file()
            and bulletin_mp4_path.stat().st_size > 100_000
        )
        checks["stage_4_render"] = (
            "PASS" if shorts_meta_ok and bulletin_meta_ok and bulletin_mp4_ok
            else "FAIL"
        )
        assert checks["stage_4_render"] == "PASS", (
            f"shorts_meta_ok={shorts_meta_ok} "
            f"bulletin_meta_ok={bulletin_meta_ok} "
            f"bulletin_mp4_ok={bulletin_mp4_ok}"
        )
        # Capture editor_meta.json contents for the diag artifact
        if shorts_meta_path.is_file():
            (run_dir / "editor_meta_shorts.json").write_text(
                shorts_meta_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        if bulletin_meta_path.is_file():
            (run_dir / "editor_meta_bulletin.json").write_text(
                bulletin_meta_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        # Capture bulletin meta (size + duration; the .mp4 itself stays
        # outside the diag dir -- too large to commit and noisy).
        bulletin_size = bulletin_mp4_path.stat().st_size if bulletin_mp4_ok else 0
        (run_dir / "bulletin_meta.json").write_text(
            json.dumps({
                "path":         str(bulletin_mp4_path),
                "size_bytes":   bulletin_size,
                "size_mb":      round(bulletin_size / (1024 * 1024), 2),
                "duration_s":   stage_4["bulletin"]["duration_s"],
                "stories":      stage_4["bulletin"]["stories_rendered"],
                "shorts_count": stage_4["composed_shorts_count"],
            }, indent=2),
            encoding="utf-8",
        )

        # ---- Finalize ----
        env = await orchestrator._finalize_handler(env)
        finalize = env["finalize"]
        (run_dir / "stage_costs.json").write_text(
            json.dumps(env["stage_costs"], indent=2),
            encoding="utf-8",
        )
        total_cost = finalize["total_cost_usd"]
        # Validate Job row was updated to status='done' in the DB
        from models import Job
        sess = in_memory_db["SessionLocal"]()
        try:
            job = sess.query(Job).filter(Job.id == 1).first()
            db_status = job.status
            db_current_stage = job.current_stage
        finally:
            sess.close()
        finalize_ok = (
            finalize["status"] == "done"
            and db_status == "done"
            and db_current_stage is None
        )
        checks["finalize_db"] = "PASS" if finalize_ok else "FAIL"
        assert checks["finalize_db"] == "PASS", (
            f"finalize={finalize['status']} db_status={db_status} "
            f"db_current_stage={db_current_stage}"
        )

        # ---- Cost-budget warning ----
        if total_cost > 2.0:
            print(
                f"\n!!! Cost ledger warning: this run cost ${total_cost:.4f} "
                f"-- exceeds D-12.9 single-run budget of $2.00. Investigate."
            )

        # ---- Write the manifest (Step 12 PASS certification record) ----
        wall_seconds = time.perf_counter() - run_started
        manifest = {
            "timestamp":     datetime.now().strftime("%Y%m%d_%H%M%S"),
            "mode":          "direct",
            "test_video":    str(test_video_path),
            "stt_provider":  os.environ.get("KAIZER_STT_PROVIDER", "whisper-groq"),
            "wall_seconds":  round(wall_seconds, 2),
            "cost_usd":      {
                **{k: round(float(v), 4) for k, v in env["stage_costs"].items()},
                "TOTAL":     round(total_cost, 4),
            },
            "checks":        checks,
            "outputs": {
                "shorts_editor_meta": str(shorts_meta_path),
                "bulletin_mp4":       str(bulletin_mp4_path),
                "bulletin_size_mb":   round(bulletin_size / (1024 * 1024), 2),
            },
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n[Step 12.2a PASS] manifest -> {run_dir / 'manifest.json'}")


@pytest.mark.skipif(
    not _inngest_cli_available(),
    reason=(
        "Inngest CLI not on PATH. Install per "
        "tests/fixtures/step12_diag/README.md -> 'Inngest Dev "
        "Server install (for 12.2b)' section, then re-run."
    ),
)
@pytest.mark.skipif(
    not _inngest_dev_server_listening(),
    reason=(
        "Inngest Dev Server not listening on localhost:8288. "
        "Start it with: ``inngest dev`` before running."
    ),
)
class TestE2EInngestDevServer_12_2b:
    """Step 12.2b -- real Inngest Dev Server event delivery.
    Same E2E scenario as 12.2a but via the production event path.
    Lands in 12.2b commit.
    """

    def test_placeholder(self):
        pytest.skip("12.2b body lands after 12.2a STOP+review")


class TestE2ECancellation_12_3:
    """Step 12.3 -- real mid-render cancellation (D-12.10 pushback).
    Asyncio task running render, real cancel_job() trigger, psutil
    orphan-FFmpeg check. Lands in 12.3 commit.
    """

    def test_placeholder(self):
        pytest.skip("12.3 body lands after 12.2b STOP+review")


class TestE2EIdempotency_12_4:
    """Step 12.4 -- idempotency fixture comparison. Verify the
    runner.py V2 dispatcher builds Event(id=f"job-{job_id}") on
    repeat calls. Lands in 12.4 commit.
    """

    def test_placeholder(self):
        pytest.skip("12.4 body lands after 12.3 STOP+review")
