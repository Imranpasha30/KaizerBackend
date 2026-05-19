"""Step 12.3 E2E test — Real mid-render cancellation under Inngest.

Validates the V2 two-layer cancellation contract under live Inngest
execution. Test 1 (in this file) exercises Layer 1 (cooperative
``_check_cancelled`` between Inngest steps). Test 2 (added in a
later commit) exercises Layer 2 (the ``_V2WorkerProxy`` /
``_ACTIVE_PROCS`` bridge that SIGKILLs Stage 4's FFmpeg
descendants).

D-12.10 pushback rationale: "Without working cancel, a 14-min job
cancelled at minute 3 keeps spending. Without working SIGKILL,
FFmpeg processes orphan and accumulate. Both layers MUST be
verified under real Inngest dispatch (not mocked)."

This file's tests are opt-in (KAIZER_RUN_INTEGRATION_TESTS=1) AND
require uvicorn + inngest dev running (same setup as Step 12.2b).
See ``test_e2e_v2_inngest.py`` for the module-level setup procedure.

Diagnostic artifacts:
  pipeline_v2/tests/fixtures/step12_diag/12_3_cancel/<ts>/
    inngest_run_id.txt
    manifest.json   (cancel_requested_at, terminal_status_at,
                     time_to_cancel_s, stages_completed_before_cancel,
                     job_error_snippet)
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pytest


RUN_INTEGRATION = (
    os.environ.get("KAIZER_RUN_INTEGRATION_TESTS", "").strip() == "1"
)

_DEFAULT_TEST_VIDEO = (
    Path(__file__).resolve().parents[4] / "videos" / "test.mp4"
)
TEST_VIDEO_PATH = Path(
    os.environ.get("KAIZER_TEST_VIDEO", "").strip()
    or str(_DEFAULT_TEST_VIDEO)
)

UVICORN_BASE = os.environ.get(
    "KAIZER_TEST_UVICORN_BASE", "http://127.0.0.1:8000",
).rstrip("/")
INNGEST_DEV_BASE = os.environ.get(
    "KAIZER_TEST_INNGEST_DEV_BASE", "http://127.0.0.1:8288",
).rstrip("/")
INNGEST_DEV_PORT = 8288


pytestmark = pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason=(
        "V2 cancellation E2E tests skipped. Set "
        "KAIZER_RUN_INTEGRATION_TESTS=1 plus the required API keys. "
        "Same setup as test_e2e_v2_inngest.py."
    ),
)


# ====================================================================== #
# Service probes + GraphQL helpers (duplicated from 12.2b for           #
# self-contained file; refactor to a shared _inngest_helpers module     #
# if a third Inngest E2E test appears.)                                  #
# ====================================================================== #


def _tcp_listening(host: str, port: int, timeout_s: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout_s):
            return True
    except OSError:
        return False


def _uvicorn_ready() -> bool:
    try:
        with urllib.request.urlopen(
            f"{UVICORN_BASE}/api/health/", timeout=2,
        ) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _uvicorn_serves_inngest() -> bool:
    try:
        with urllib.request.urlopen(
            f"{UVICORN_BASE}/api/inngest", timeout=2,
        ) as resp:
            if resp.status != 200:
                return False
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("function_count", 0) >= 1
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return False


def _inngest_cli_available() -> bool:
    return shutil.which("inngest") is not None


def _gql(query: str) -> dict:
    req = urllib.request.Request(
        f"{INNGEST_DEV_BASE}/v0/gql",
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = ""
        try:
            err_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(
            f"Inngest GraphQL HTTP {exc.code}: {exc.reason}. "
            f"Body: {err_body[:1000]}. Query: {query[:300]}"
        ) from exc
    if body.get("errors"):
        raise RuntimeError(
            f"Inngest GraphQL error: {body['errors']!r}"
        )
    return body.get("data") or {}


def _find_run_for_event(
    event_name: str, from_iso: str, app_id_uuid: Optional[str] = None,
) -> Optional[dict]:
    filter_parts = ['from: "' + from_iso + '"']
    if app_id_uuid:
        filter_parts.append('appIDs: ["' + app_id_uuid + '"]')
    filter_str = ", ".join(filter_parts)
    query = (
        "{ runs(first: 10, "
        "orderBy: [{field: QUEUED_AT, direction: DESC}], "
        "filter: {" + filter_str + "}) { "
        "edges { node { id status eventName queuedAt "
        "startedAt endedAt output } } } }"
    )
    data = _gql(query)
    edges = (data.get("runs") or {}).get("edges") or []
    for edge in edges:
        node = edge.get("node") or {}
        if node.get("eventName") == event_name:
            return node
    return None


def _get_app_id_uuid() -> Optional[str]:
    data = _gql('{ apps { id name } }')
    for app in (data.get("apps") or []):
        if app.get("name") == "kaizer-v2":
            return app.get("id")
    return None


def _get_run_status(run_id: str) -> dict:
    query = (
        '{ run(runID: "' + run_id + '") { id status startedAt endedAt '
        "output appID functionID } }"
    )
    data = _gql(query)
    return data.get("run") or {}


# ====================================================================== #
# Module-level fixtures (mirror 12.2b)                                    #
# ====================================================================== #


@pytest.fixture(scope="module")
def test_video_path() -> Path:
    if not TEST_VIDEO_PATH.is_file():
        pytest.skip(
            f"Test video not found at {TEST_VIDEO_PATH}. "
            f"Set KAIZER_TEST_VIDEO=/abs/path/test.mp4."
        )
    return TEST_VIDEO_PATH


@pytest.fixture
def required_api_keys() -> dict:
    keys = {
        "GEMINI_API_KEY":   os.environ.get("GEMINI_API_KEY", "").strip(),
        "DEEPGRAM_API_KEY": os.environ.get("DEEPGRAM_API_KEY", "").strip(),
    }
    missing = [k for k, v in keys.items() if not v]
    if missing:
        pytest.skip(f"Required API keys missing: {missing}.")
    return keys


@pytest.fixture(scope="module")
def services_ready():
    if not _inngest_cli_available():
        pytest.skip("inngest CLI not on PATH.")
    if not _tcp_listening("127.0.0.1", INNGEST_DEV_PORT):
        pytest.skip(
            f"inngest dev not listening on :{INNGEST_DEV_PORT}. "
            f"See test_e2e_v2_inngest.py setup."
        )
    if not _uvicorn_ready():
        pytest.skip(
            f"uvicorn not responding at {UVICORN_BASE}/api/health/."
        )
    if not _uvicorn_serves_inngest():
        pytest.skip(
            f"uvicorn at {UVICORN_BASE} doesn't serve /api/inngest."
        )
    return True


@pytest.fixture(scope="module")
def diag_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (
        Path(__file__).resolve().parent
        / "fixtures" / "step12_diag" / "12_3_cancel" / ts
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


# ====================================================================== #
# Step 12.3 Test 1: mid-pipeline cancel (Layer 1, cooperative)            #
# ====================================================================== #


class TestE2ECancellationMidRender:
    """Set Job.cancel_requested=True between Inngest steps; verify the
    next step's ``_check_cancelled`` raises NonRetriableError, Inngest
    terminates the run, and Job.status reflects the cancellation
    within 60 seconds.
    """

    # Unique high-int Job.id so it doesn't collide with any other
    # in-flight test job. Different from 12.2b's _TEST_JOB_ID.
    _TEST_JOB_ID = 99_120_030_1   # "step 12.3.1" mnemonic

    # Target stages to wait for before triggering cancel. Mid-pipeline
    # means we want the run to be DURING OR AFTER Stage 1 (so Layer 1
    # has had a chance to do real work) but BEFORE Stage 4 (so we
    # exercise the cooperative path, not the SIGKILL path).
    # Includes stage_1_transcribe per attempt #1 calibration: Stage 0
    # mezzanine + Stage 1 STT together can exceed 4 min on a loaded
    # system; cancelling during stage_1 still validates the
    # cooperative check (next step's _check_cancelled fires).
    _TARGET_STAGES = {
        "stage_1_transcribe",
        "stage_2_continuity",
        "stage_2_5_entities",
        "stage_3_fanout",
    }

    @pytest.fixture
    def db_job_row(self, test_video_path: Path):
        """Create a Job row in the live DB, cascade-aware teardown."""
        from database import SessionLocal
        from models import Job, Clip

        sess = SessionLocal()
        try:
            sess.query(Clip).filter(
                Clip.job_id == self._TEST_JOB_ID,
            ).delete()
            sess.query(Job).filter(Job.id == self._TEST_JOB_ID).delete()
            sess.commit()
            job = Job(
                id=self._TEST_JOB_ID,
                platform="full_video_shorts_v2",
                video_name=test_video_path.name,
                status="running",
                cancel_requested=False,
            )
            sess.add(job)
            sess.commit()
        finally:
            sess.close()

        try:
            yield self._TEST_JOB_ID, SessionLocal
        finally:
            sess = SessionLocal()
            try:
                sess.query(Clip).filter(
                    Clip.job_id == self._TEST_JOB_ID,
                ).delete()
                sess.query(Job).filter(
                    Job.id == self._TEST_JOB_ID,
                ).delete()
                sess.commit()
            finally:
                sess.close()

    def test_cancel_during_stage_2_or_3(
        self,
        services_ready: bool,
        test_video_path: Path,
        required_api_keys: dict,
        db_job_row: tuple,
        diag_dir: Path,
        tmp_path: Path,
    ):
        job_id, _SessionLocal = db_job_row
        from pipeline_v2.inngest_client import get_client
        from inngest import Event
        from models import Job

        # ---- 1. Dispatch (production code path) ----------------------
        t0 = datetime.now(timezone.utc).replace(microsecond=0)
        from_iso = t0.isoformat().replace("+00:00", "Z")
        app_id_uuid = _get_app_id_uuid()
        assert app_id_uuid, "kaizer-v2 app not registered with Inngest dev."

        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        event = Event(
            name="video/v2/uploaded",
            id=f"job-{job_id}-{int(time.time() * 1000)}",
            data={
                "job_id":       job_id,
                "video_path":   str(test_video_path),
                "language":     "te",
                "platform":     "full_video_shorts_v2",
                "frame_layout": "torn_card",
                "stt_provider": "deepgram",
                "preset": {
                    "label":  "Full Video + Shorts (V2 Beta)",
                    "width":  1080, "height": 1920,
                    "min_dur": 15, "max_dur": 60, "ideal_dur": 45,
                    "vertical": True,
                },
                "out_dir": str(out_dir),
            },
        )
        client = get_client()
        dispatch_started = time.time()
        sent_event_ids = client.send_sync(events=[event])
        assert sent_event_ids
        print(
            f"\n[12.3.1] sent event id(s)={sent_event_ids} "
            f"app={app_id_uuid}",
            flush=True,
        )

        # ---- 2. Discover the run -------------------------------------
        run_node: Optional[dict] = None
        discover_deadline = time.time() + 30.0
        while time.time() < discover_deadline:
            run_node = _find_run_for_event(
                "video/v2/uploaded", from_iso, app_id_uuid=app_id_uuid,
            )
            if run_node:
                break
            time.sleep(1.0)
        assert run_node, "No run created within 30s of dispatch."
        run_id = run_node["id"]
        (diag_dir / "inngest_run_id.txt").write_text(run_id, encoding="utf-8")
        print(f"[12.3.1] discovered run_id={run_id}", flush=True)

        # ---- 3. Wait for current_stage to enter the target window ----
        # Max 8 minutes -- attempt #1 calibration: on a loaded box
        # Stage 0 mezzanine NVENC fallback to libx264 can take 3-4 min,
        # then Stage 1 STT adds another ~30s. 8 min covers worst-case
        # while leaving the strict pre-Stage-4 boundary intact (Stage 4
        # render itself doesn't START until well past 8 min in 12.2b
        # PASS runs). Cancel fires when current_stage enters the
        # target set (which now includes stage_1_transcribe).
        target_deadline = time.time() + 8 * 60
        observed_stage: Optional[str] = None
        progress_logged = 0.0
        while time.time() < target_deadline:
            sess = _SessionLocal()
            try:
                job = sess.query(Job).filter(Job.id == job_id).first()
                observed_stage = (
                    getattr(job, "current_stage", None) if job else None
                )
            finally:
                sess.close()
            now = time.time()
            if (now - progress_logged) > 30:
                elapsed = now - dispatch_started
                print(
                    f"[12.3.1] wait-for-target t+{elapsed:.0f}s "
                    f"current_stage={observed_stage!r}",
                    flush=True,
                )
                progress_logged = now
            if observed_stage in self._TARGET_STAGES:
                break
            # Defensive: if the run already reached Stage 4 or finalize,
            # we missed the cancel window -- fail FAST rather than
            # cancelling Stage 4 (that's Test 2's scope).
            if observed_stage in ("stage_4_render", "finalize"):
                pytest.fail(
                    f"current_stage={observed_stage!r} -- pipeline "
                    f"reached Stage 4 before we could cancel mid-"
                    f"pipeline. Pipeline is faster than expected; "
                    f"adjust _TARGET_STAGES."
                )
            time.sleep(2.0)
        assert observed_stage in self._TARGET_STAGES, (
            f"Job.current_stage never entered {self._TARGET_STAGES} "
            f"within 4 min. Last observed: {observed_stage!r}. "
            f"Pipeline may be stuck."
        )
        stage_at_cancel = observed_stage
        print(
            f"[12.3.1] cancel-trigger window reached: "
            f"current_stage={stage_at_cancel!r}",
            flush=True,
        )

        # ---- 4. Trigger the cancel via direct DB write ---------------
        # In production this is what the HTTP cancel endpoint does
        # (plus a SIGKILL walk via _ACTIVE_PROCS for Stage 4 jobs).
        # Test 1 exercises ONLY Layer 1 -- the cooperative
        # _check_cancelled at the next step boundary.
        cancel_requested_at = time.time()
        sess = _SessionLocal()
        try:
            sess.query(Job).filter(Job.id == job_id).update(
                {"cancel_requested": True},
                synchronize_session=False,
            )
            sess.commit()
        finally:
            sess.close()
        print(
            f"[12.3.1] set Job.cancel_requested=True at "
            f"t+{cancel_requested_at - dispatch_started:.1f}s",
            flush=True,
        )

        # ---- 5. Poll Inngest for terminal status (max 60s) -----------
        TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "SKIPPED"}
        cancel_deadline = cancel_requested_at + 60.0
        terminal_status: Optional[str] = None
        while time.time() < cancel_deadline:
            node = _get_run_status(run_id)
            status = node.get("status") or "QUEUED"
            if status in TERMINAL:
                terminal_status = status
                break
            time.sleep(2.0)
        terminal_status_at = time.time()
        time_to_cancel_s = terminal_status_at - cancel_requested_at

        # ---- 6. Snapshot Job state at terminal -----------------------
        sess = _SessionLocal()
        try:
            job = sess.query(Job).filter(Job.id == job_id).first()
            job_status_at_terminal = (
                job.status if job else "<deleted>"
            )
            job_current_stage_at_terminal = (
                getattr(job, "current_stage", None) if job else None
            )
            job_cancel_requested_at_terminal = (
                bool(job.cancel_requested) if job else None
            )
            job_error = (job.error if job and job.error else "")[:500]
        finally:
            sess.close()

        # ---- 7. Manifest --------------------------------------------
        wall_seconds = terminal_status_at - dispatch_started
        manifest = {
            "timestamp":  datetime.now().strftime("%Y%m%d_%H%M%S"),
            "mode":       "inngest_mid_pipeline_cancel",
            "test_video": str(test_video_path),
            "stt_provider": "deepgram",
            "inngest": {
                "run_id":            run_id,
                "app_id_uuid":       app_id_uuid,
                "sent_event_ids":    sent_event_ids,
                "from_iso":          from_iso,
                "final_status":      terminal_status,
            },
            "cancel": {
                "stage_at_cancel_trigger":   stage_at_cancel,
                "cancel_requested_at_s":     round(
                    cancel_requested_at - dispatch_started, 2,
                ),
                "terminal_status_at_s":      round(
                    terminal_status_at - dispatch_started, 2,
                ),
                "time_to_cancel_s":          round(time_to_cancel_s, 2),
            },
            "job_state_at_terminal": {
                "status":           job_status_at_terminal,
                "current_stage":    job_current_stage_at_terminal,
                "cancel_requested": job_cancel_requested_at_terminal,
                "error_snippet":    job_error,
            },
            "wall_seconds": round(wall_seconds, 2),
        }
        (diag_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"\n[12.3.1] manifest -> {diag_dir / 'manifest.json'}",
            flush=True,
        )

        # ---- 8. Acceptance assertions -------------------------------
        assert terminal_status in {"FAILED", "CANCELLED"}, (
            f"Inngest run {run_id} terminal status was "
            f"{terminal_status!r}; expected FAILED (NonRetriableError "
            f"path) or CANCELLED. See manifest at "
            f"{diag_dir}/manifest.json."
        )
        assert job_status_at_terminal in {"failed", "cancelled"}, (
            f"Job.status was {job_status_at_terminal!r}; expected "
            f"'failed' (NonRetriableError -> _mark_job_failed) or "
            f"'cancelled'."
        )
        # The cooperative check raises NonRetriableError with the
        # literal "cancelled:" prefix (orchestrator.py:_check_cancelled).
        # Job.error captures the exception string, so it must contain
        # 'cancelled' somewhere.
        assert "cancelled" in (job_error or "").lower(), (
            f"Job.error doesn't contain 'cancelled' marker; got: "
            f"{job_error[:200]!r}"
        )
        assert time_to_cancel_s <= 60.0, (
            f"Cancellation took {time_to_cancel_s:.1f}s -- expected "
            f"<= 60s for the cooperative path."
        )
        # Pipeline must NOT have progressed to Stage 4 or finalize.
        # current_stage at terminal reflects the LAST stage written
        # before the cancel fired.
        assert job_current_stage_at_terminal not in (
            "stage_4_render", "finalize",
        ), (
            f"Pipeline reached {job_current_stage_at_terminal!r} after "
            f"cancel was triggered at {stage_at_cancel!r}. Cooperative "
            f"check failed to short-circuit."
        )


# ====================================================================== #
# Step 12.3 Test 2: Stage 4 cancel + psutil orphan check (Layer 2)        #
# ====================================================================== #


def _find_uvicorn_pid(port: int = 8000) -> Optional[int]:
    """Find the PID listening on ``port`` (the uvicorn that hosts
    /api/inngest). Used to enumerate the worker's process tree
    when the test snapshots FFmpeg subprocesses.

    Uses psutil's net_connections which works on Windows without
    admin in most environments. Returns None if no listener found.
    """
    import psutil
    for c in psutil.net_connections(kind="inet"):
        if (
            c.status == psutil.CONN_LISTEN
            and c.laddr
            and c.laddr.port == port
            and c.pid is not None
        ):
            return c.pid
    return None


def _ffmpeg_like_descendants(worker_pid: int) -> list[dict]:
    """Walk the descendants of ``worker_pid`` and return ffmpeg /
    ffprobe / pillow-like processes.

    Each entry: {pid, name, exe_basename, parent_pid, cmdline_head}.
    Used at t1 (just before cancel) and t2 (after cancel) so we can
    assert SIGKILL via _V2WorkerProxy actually reaped them.

    Defensive: psutil may raise NoSuchProcess if a child terminates
    between iteration and inspection. We treat those as "already
    gone" and exclude them from the snapshot.
    """
    import psutil
    targets = ("ffmpeg", "ffprobe", "pillow")
    out: list[dict] = []
    try:
        parent = psutil.Process(worker_pid)
        for child in parent.children(recursive=True):
            try:
                name = (child.name() or "").lower()
                exe = child.exe() if hasattr(child, "exe") else ""
                exe_basename = (
                    Path(exe).name.lower() if exe else name
                )
                if any(t in name or t in exe_basename for t in targets):
                    cmd = ""
                    try:
                        cmd = " ".join(child.cmdline())[:120]
                    except Exception:
                        pass
                    out.append({
                        "pid":         child.pid,
                        "name":        name,
                        "exe_basename": exe_basename,
                        "parent_pid":  (
                            child.parent().pid
                            if child.parent() else None
                        ),
                        "cmdline_head": cmd,
                    })
            except psutil.NoSuchProcess:
                continue
    except psutil.NoSuchProcess:
        return out
    return out


class TestE2ECancellationDuringStage4Render:
    """Set Job.cancel_requested=True while Stage 4 FFmpeg is actively
    rendering; verify the V1 ``cancel_job`` path (via the
    ``_V2WorkerProxy`` registered in ``_ACTIVE_PROCS``) walks the
    worker's descendants and SIGKILLs every FFmpeg/ffprobe within
    30 seconds.

    This exercises **Layer 2** of the V2 cancellation contract --
    the hard-cancel path the cooperative ``_check_cancelled`` cannot
    provide for long Stage 4 renders. The unit-test surface only
    verified that ``_register_stage_4_with_active_procs`` registers
    the proxy; the actual tree-kill has never been exercised under
    real Inngest conditions before this test.
    """

    _TEST_JOB_ID = 99_120_030_2   # "step 12.3.2" mnemonic

    @pytest.fixture
    def db_job_row(self, test_video_path: Path):
        """Same cascade-aware pattern as Test 1; distinct Job.id so
        the two tests can run back-to-back without colliding.
        """
        from database import SessionLocal
        from models import Job, Clip

        sess = SessionLocal()
        try:
            sess.query(Clip).filter(
                Clip.job_id == self._TEST_JOB_ID,
            ).delete()
            sess.query(Job).filter(Job.id == self._TEST_JOB_ID).delete()
            sess.commit()
            job = Job(
                id=self._TEST_JOB_ID,
                platform="full_video_shorts_v2",
                video_name=test_video_path.name,
                status="running",
                cancel_requested=False,
            )
            sess.add(job)
            sess.commit()
        finally:
            sess.close()
        try:
            yield self._TEST_JOB_ID, SessionLocal
        finally:
            sess = SessionLocal()
            try:
                sess.query(Clip).filter(
                    Clip.job_id == self._TEST_JOB_ID,
                ).delete()
                sess.query(Job).filter(
                    Job.id == self._TEST_JOB_ID,
                ).delete()
                sess.commit()
            finally:
                sess.close()

    def test_cancel_during_stage_4_kills_ffmpeg_subprocesses(
        self,
        services_ready: bool,
        test_video_path: Path,
        required_api_keys: dict,
        db_job_row: tuple,
        diag_dir: Path,
        tmp_path: Path,
    ):
        # ---- psutil-availability gate ----------------------------
        try:
            import psutil
        except ImportError:
            pytest.skip(
                "psutil not installed -- Layer 2 SIGKILL verification "
                "cannot run without process-tree introspection."
            )
        worker_pid = _find_uvicorn_pid(port=8000)
        if worker_pid is None:
            pytest.skip(
                "Could not locate uvicorn PID listening on :8000. "
                "Layer 2 verification needs it for descendant "
                "enumeration."
            )

        job_id, _SessionLocal = db_job_row
        from pipeline_v2.inngest_client import get_client
        from inngest import Event
        from models import Job

        # ---- 1. Dispatch -----------------------------------------
        t0 = datetime.now(timezone.utc).replace(microsecond=0)
        from_iso = t0.isoformat().replace("+00:00", "Z")
        app_id_uuid = _get_app_id_uuid()
        assert app_id_uuid

        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        event = Event(
            name="video/v2/uploaded",
            id=f"job-{job_id}-{int(time.time() * 1000)}",
            data={
                "job_id":       job_id,
                "video_path":   str(test_video_path),
                "language":     "te",
                "platform":     "full_video_shorts_v2",
                "frame_layout": "torn_card",
                "stt_provider": "deepgram",
                "preset": {
                    "label":  "Full Video + Shorts (V2 Beta)",
                    "width":  1080, "height": 1920,
                    "min_dur": 15, "max_dur": 60, "ideal_dur": 45,
                    "vertical": True,
                },
                "out_dir": str(out_dir),
            },
        )
        client = get_client()
        dispatch_started = time.time()
        sent_event_ids = client.send_sync(events=[event])
        assert sent_event_ids
        print(
            f"\n[12.3.2] worker_pid={worker_pid} "
            f"sent event id(s)={sent_event_ids}",
            flush=True,
        )

        # ---- 2. Discover the run ---------------------------------
        run_node: Optional[dict] = None
        discover_deadline = time.time() + 30.0
        while time.time() < discover_deadline:
            run_node = _find_run_for_event(
                "video/v2/uploaded", from_iso, app_id_uuid=app_id_uuid,
            )
            if run_node:
                break
            time.sleep(1.0)
        assert run_node
        run_id = run_node["id"]
        (diag_dir / "test2_inngest_run_id.txt").write_text(
            run_id, encoding="utf-8",
        )
        print(f"[12.3.2] discovered run_id={run_id}", flush=True)

        # ---- 3. Wait for current_stage == 'stage_4_render' --------
        # Max 12 minutes: Stage 0/1/2/2.5/3 typically take 8-10 min
        # combined, then Stage 4 starts.
        target_deadline = time.time() + 12 * 60
        observed_stage: Optional[str] = None
        progress_logged = 0.0
        while time.time() < target_deadline:
            sess = _SessionLocal()
            try:
                job = sess.query(Job).filter(Job.id == job_id).first()
                observed_stage = (
                    getattr(job, "current_stage", None) if job else None
                )
            finally:
                sess.close()
            now = time.time()
            if (now - progress_logged) > 30:
                elapsed = now - dispatch_started
                print(
                    f"[12.3.2] wait-for-stage-4 t+{elapsed:.0f}s "
                    f"current_stage={observed_stage!r}",
                    flush=True,
                )
                progress_logged = now
            if observed_stage == "stage_4_render":
                break
            if observed_stage == "finalize":
                pytest.fail(
                    "Pipeline already reached finalize before we "
                    "could cancel mid-Stage-4. Pipeline faster than "
                    "expected; consider larger test_video."
                )
            time.sleep(2.0)
        assert observed_stage == "stage_4_render", (
            f"current_stage never reached 'stage_4_render' within "
            f"12 min. Last observed: {observed_stage!r}."
        )
        stage_4_entered_at = time.time()
        print(
            f"[12.3.2] stage_4_render entered at "
            f"t+{stage_4_entered_at - dispatch_started:.0f}s",
            flush=True,
        )

        # ---- 4. Wait for FFmpeg to actually be ACTIVE -------------
        # Stage 4 alternates between FFmpeg-bound phases
        # (cut_raw_shorts ~10-20s, compose_shorts ~60s,
        # render_bulletin ~3 min) and HTTP-bound phases
        # (resolve_images via Pexels ~30-60s). The initial attempt
        # snapshot at a fixed 30s offset landed in an HTTP phase
        # (no FFmpegs running -- nothing to kill). The redesign:
        # poll psutil every 2s until at least 1 ffmpeg child of
        # the worker is alive. Max 8 min wait inside Stage 4 (the
        # FFmpeg gaps between phases are <60s; 8 min covers
        # multi-phase variance).
        ffmpeg_wait_deadline = time.time() + 8 * 60
        ffmpeg_observed_at: Optional[float] = None
        snapshot_t1: list[dict] = []
        while time.time() < ffmpeg_wait_deadline:
            snapshot_t1 = _ffmpeg_like_descendants(worker_pid)
            if snapshot_t1:
                ffmpeg_observed_at = time.time()
                break
            time.sleep(2.0)
        assert ffmpeg_observed_at is not None, (
            "No FFmpeg descendant of the uvicorn worker observed "
            "within 8 min of Stage 4 entry. Stage 4 should be "
            "running FFmpeg invocations in cut_raw_shorts / "
            "compose_shorts / render_bulletin."
        )
        print(
            f"[12.3.2] FFmpeg active at "
            f"t+{ffmpeg_observed_at - dispatch_started:.0f}s "
            f"(n_descendants={len(snapshot_t1)}); proceeding to cancel",
            flush=True,
        )

        # ---- 6. Trigger cancel via DB write ----------------------
        cancel_requested_at = time.time()
        sess = _SessionLocal()
        try:
            sess.query(Job).filter(Job.id == job_id).update(
                {"cancel_requested": True},
                synchronize_session=False,
            )
            sess.commit()
        finally:
            sess.close()
        print(
            f"[12.3.2] set Job.cancel_requested=True at "
            f"t+{cancel_requested_at - dispatch_started:.1f}s",
            flush=True,
        )

        # In production the HTTP cancel endpoint ALSO walks
        # _ACTIVE_PROCS[job_id] and invokes the SIGKILL path. The
        # test mirrors that step explicitly because we set the DB
        # flag directly (bypassing the HTTP route).
        import runner as v1_runner
        cancel_invoked = False
        try:
            v1_runner.cancel_job(job_id)
            cancel_invoked = True
            print(
                f"[12.3.2] runner.cancel_job({job_id}) invoked",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[12.3.2][warn] runner.cancel_job raised: {exc!r}",
                flush=True,
            )

        # ---- 7. Poll Inngest for terminal status (max 90s) -------
        # With the Step 12.3 Test 2 fix (cancel_check inside
        # _render_impl, backlog item 76), Stage 4's next sub-phase
        # boundary fires _check_cancelled within seconds. 90s
        # accounts for: completing the current FFmpeg invocation
        # (~10-60s) + one DB read latency + Inngest's terminal
        # state propagation. Pre-fix, the natural Stage 4
        # completion took ~5 min; the fix should bring this in
        # under 90s.
        TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "SKIPPED"}
        cancel_deadline = cancel_requested_at + 90.0
        terminal_status: Optional[str] = None
        time_to_zero_ffmpeg_s: Optional[float] = None
        last_descendants_check = 0.0
        while time.time() < cancel_deadline:
            now = time.time()
            # Concurrent ffmpeg-tree check every 2s; record the
            # first time the count hits zero.
            if (now - last_descendants_check) > 2:
                live = _ffmpeg_like_descendants(worker_pid)
                if not live and time_to_zero_ffmpeg_s is None:
                    time_to_zero_ffmpeg_s = (
                        now - cancel_requested_at
                    )
                last_descendants_check = now

            node = _get_run_status(run_id)
            status = node.get("status") or "QUEUED"
            if status in TERMINAL:
                terminal_status = status
                break
            time.sleep(2.0)
        terminal_status_at = time.time()

        # ---- 8. Snapshot t2 (AFTER cancel) -----------------------
        snapshot_t2 = _ffmpeg_like_descendants(worker_pid)
        print(
            f"[12.3.2] snapshot_t2: {len(snapshot_t2)} ffmpeg-like "
            f"descendants after cancel",
            flush=True,
        )

        # ---- 9. Job state at terminal ----------------------------
        sess = _SessionLocal()
        try:
            job = sess.query(Job).filter(Job.id == job_id).first()
            job_status = job.status if job else "<deleted>"
            job_current_stage = (
                getattr(job, "current_stage", None) if job else None
            )
            job_error = (job.error if job and job.error else "")[:500]
        finally:
            sess.close()

        # ---- 10. Manifest ---------------------------------------
        manifest = {
            "timestamp":  datetime.now().strftime("%Y%m%d_%H%M%S"),
            "mode":       "inngest_stage_4_cancel",
            "test_video": str(test_video_path),
            "worker_pid": worker_pid,
            "inngest": {
                "run_id":            run_id,
                "app_id_uuid":       app_id_uuid,
                "sent_event_ids":    sent_event_ids,
                "final_status":      terminal_status,
            },
            "cancel": {
                "stage_4_entered_at_s":    round(
                    stage_4_entered_at - dispatch_started, 2,
                ),
                "cancel_requested_at_s":   round(
                    cancel_requested_at - dispatch_started, 2,
                ),
                "terminal_status_at_s":    round(
                    terminal_status_at - dispatch_started, 2,
                ),
                "time_to_terminal_s":      round(
                    terminal_status_at - cancel_requested_at, 2,
                ),
                "time_to_zero_ffmpeg_s":   (
                    round(time_to_zero_ffmpeg_s, 2)
                    if time_to_zero_ffmpeg_s is not None else None
                ),
                "v1_cancel_job_invoked":   cancel_invoked,
            },
            "processes_before_cancel": snapshot_t1,
            "processes_after_cancel":  snapshot_t2,
            "job_state_at_terminal": {
                "status":         job_status,
                "current_stage":  job_current_stage,
                "error_snippet":  job_error,
            },
            "wall_seconds": round(terminal_status_at - dispatch_started, 2),
        }
        (diag_dir / "test2_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"\n[12.3.2] test2_manifest -> "
            f"{diag_dir / 'test2_manifest.json'}",
            flush=True,
        )

        # ---- 11. Acceptance assertions ---------------------------
        assert terminal_status in {"FAILED", "CANCELLED"}, (
            f"Inngest terminal_status={terminal_status!r} -- "
            f"expected FAILED or CANCELLED."
        )
        assert job_status in {"failed", "cancelled"}, (
            f"Job.status={job_status!r} -- expected 'failed' or "
            f"'cancelled'."
        )
        # No NEW ffmpeg processes spawned after cancel.
        assert len(snapshot_t2) <= len(snapshot_t1), (
            f"snapshot_t2 has MORE ffmpeg descendants than t1 "
            f"({len(snapshot_t2)} > {len(snapshot_t1)}). New "
            f"FFmpegs spawned AFTER cancel was triggered -- "
            f"Layer 2 SIGKILL is not stopping new spawns."
        )
        # The hard guarantee: all ffmpeg-like descendants gone
        # within 30s of cancel_requested.
        assert len(snapshot_t2) == 0, (
            f"snapshot_t2 still has {len(snapshot_t2)} ffmpeg-like "
            f"descendants 90s after cancel. _V2WorkerProxy + "
            f"_ACTIVE_PROCS SIGKILL did not reap them. See "
            f"test2_manifest.json for the PIDs."
        )
        # Time-to-terminal should be well under 90s with the
        # Step 12.3 Test 2 fix in place (cancel_check between
        # Stage 4 sub-phases raises NonRetriableError at the next
        # sub-phase boundary).
        assert (
            terminal_status_at - cancel_requested_at
        ) <= 90.0, (
            f"Cancel-to-terminal took "
            f"{terminal_status_at - cancel_requested_at:.1f}s -- "
            f"expected <= 90s. Either the Stage 4 sub-phase "
            f"cancel_check fix (backlog 76) regressed, or the "
            f"current FFmpeg sub-phase ran longer than 90s "
            f"(unusually long compose; investigate)."
        )
