"""Step 12.4 E2E test — Inngest idempotency dedup (D-10.10).

Validates the V2 D-10.10 idempotency contract end-to-end: sending
the SAME event (same ``Event.id``) twice within the 24h idempotency
window creates exactly ONE run, not two. This is the behavior
``runner.py``'s V2 dispatcher relies on so a user re-submitting the
same job doesn't double-process.

12.2b run #3 accidentally observed this behavior when its test
fixture reused a constant ``Event.id`` across attempts and Inngest
silently deduped the second send. Step 12.4 formalises that finding
as a deliberate assertion against the production contract.

Cost-conscious test design (D-12.9 budget):
  - We only need to OBSERVE "1 run, not 2" within ~15s of the second
    send. Once observed, the contract is validated.
  - The single run is left to start executing briefly (~30s) to
    confirm it's actually processing (not stalled).
  - Then we trigger cancel via Job.cancel_requested + wait for
    terminal. This caps cost at ~$0.30-0.50 (partial pipeline)
    instead of ~$0.50 (full pipeline).
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
        "V2 idempotency E2E test skipped. Set "
        "KAIZER_RUN_INTEGRATION_TESTS=1 plus the required API keys. "
        "Same setup as test_e2e_v2_inngest.py."
    ),
)


# ====================================================================== #
# Helpers (duplicated from 12.3 for self-contained file)                  #
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


def _list_runs_for_event(
    event_name: str, from_iso: str, app_id_uuid: Optional[str] = None,
) -> list[dict]:
    """Like _find_run_for_event in 12.3, but returns ALL matching
    runs (not just the first). 12.4 specifically wants to assert
    'exactly 1 run', so we need the full list.
    """
    filter_parts = ['from: "' + from_iso + '"']
    if app_id_uuid:
        filter_parts.append('appIDs: ["' + app_id_uuid + '"]')
    filter_str = ", ".join(filter_parts)
    query = (
        "{ runs(first: 50, "
        "orderBy: [{field: QUEUED_AT, direction: DESC}], "
        "filter: {" + filter_str + "}) { "
        "edges { node { id status eventName queuedAt "
        "startedAt endedAt } } } }"
    )
    data = _gql(query)
    edges = (data.get("runs") or {}).get("edges") or []
    return [
        edge["node"] for edge in edges
        if (edge.get("node") or {}).get("eventName") == event_name
    ]


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
# Fixtures                                                                #
# ====================================================================== #


@pytest.fixture(scope="module")
def test_video_path() -> Path:
    if not TEST_VIDEO_PATH.is_file():
        pytest.skip(f"Test video not found at {TEST_VIDEO_PATH}.")
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
        pytest.skip(f"inngest dev not listening on :{INNGEST_DEV_PORT}.")
    if not _uvicorn_ready():
        pytest.skip(f"uvicorn not responding at {UVICORN_BASE}.")
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
        / "fixtures" / "step12_diag" / "12_4_idempotency" / ts
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


# ====================================================================== #
# Step 12.4 test                                                          #
# ====================================================================== #


class TestE2EIdempotencyDedup_12_4:
    """Send the same ``Event`` (same ``id``) twice; verify exactly
    ONE run is created.

    Mirrors the accidental dedup observation from Step 12.2b run #3
    where re-running the test (which used a constant Event.id)
    silently suppressed the second send. This test makes that
    behavior a formal assertion of the D-10.10 contract.
    """

    _TEST_JOB_ID = 99_120_040_0  # "step 12.4" mnemonic

    @pytest.fixture
    def db_job_row(self, test_video_path: Path):
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

    def test_same_event_id_creates_one_run(
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

        # ---- 1. Build the event with a FIXED idempotency key ---------
        # The whole point: do NOT add a timestamp suffix. Both sends
        # use the exact same Event.id; Inngest's 24h idempotency
        # window must dedupe the second one.
        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        event = Event(
            name="video/v2/uploaded",
            id=f"job-{job_id}",   # constant -- D-10.10 dedup key
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

        # ---- 2. Time anchor + app id ---------------------------------
        t0 = datetime.now(timezone.utc).replace(microsecond=0)
        from_iso = t0.isoformat().replace("+00:00", "Z")
        app_id_uuid = _get_app_id_uuid()
        assert app_id_uuid, "kaizer-v2 app not registered with Inngest."

        client = get_client()

        # ---- 3. Send #1 ----------------------------------------------
        send_1_at = time.time()
        send_1_result = client.send_sync(events=[event])
        print(
            f"\n[12.4] send #1 -> {send_1_result} "
            f"at t+{send_1_at - t0.timestamp():.2f}s",
            flush=True,
        )
        assert send_1_result, "send #1 returned empty event ID list"

        # ---- 4. Wait 5s --------------------------------------------
        # Long enough for Inngest to accept + index the first event,
        # short enough that we're still well inside the idempotency
        # window.
        time.sleep(5.0)

        # ---- 5. Send #2 (SAME event, SAME id) ------------------------
        send_2_at = time.time()
        send_2_result = client.send_sync(events=[event])
        print(
            f"[12.4] send #2 -> {send_2_result} "
            f"at t+{send_2_at - t0.timestamp():.2f}s",
            flush=True,
        )
        assert send_2_result, "send #2 returned empty event ID list"

        # ---- 6. Wait for Inngest to process both -------------------
        # Both sends are accepted by Inngest's ingest API
        # immediately. The dedup happens at run-creation time, not
        # at event-receive time. We wait 10s to give the run table
        # time to settle to its final state.
        time.sleep(10.0)

        # ---- 7. Query runs and ASSERT exactly 1 --------------------
        runs = _list_runs_for_event(
            "video/v2/uploaded", from_iso, app_id_uuid=app_id_uuid,
        )
        print(
            f"[12.4] runs query returned {len(runs)} run(s) for this "
            f"event-name + app + from={from_iso}",
            flush=True,
        )
        for r in runs:
            print(
                f"  run: id={r['id']} status={r['status']} "
                f"queuedAt={r.get('queuedAt')}",
                flush=True,
            )

        (diag_dir / "event_send_results.json").write_text(
            json.dumps({
                "send_1_result":  send_1_result,
                "send_2_result":  send_2_result,
                "send_1_at_iso":  datetime.fromtimestamp(
                    send_1_at, tz=timezone.utc,
                ).isoformat(),
                "send_2_at_iso":  datetime.fromtimestamp(
                    send_2_at, tz=timezone.utc,
                ).isoformat(),
                "time_between_sends_s": round(send_2_at - send_1_at, 3),
            }, indent=2),
            encoding="utf-8",
        )
        (diag_dir / "inngest_runs_observed.json").write_text(
            json.dumps(runs, indent=2),
            encoding="utf-8",
        )

        # ---- 8. The acceptance check (D-10.10) -----------------------
        assert len(runs) == 1, (
            f"D-10.10 idempotency dedup violated: expected EXACTLY 1 "
            f"run for the duplicate-Event.id sends, got {len(runs)}. "
            f"Runs: {[r['id'] for r in runs]!r}. See "
            f"{diag_dir}/inngest_runs_observed.json."
        )

        run_node = runs[0]
        run_id = run_node["id"]
        (diag_dir / "inngest_run_record.json").write_text(
            json.dumps(run_node, indent=2),
            encoding="utf-8",
        )
        print(
            f"[12.4] D-10.10 dedup verified: 1 run created "
            f"(run_id={run_id})",
            flush=True,
        )

        # ---- 9. Let the run execute briefly to confirm not stalled --
        # Wait ~30s. Run should be RUNNING (executing real work)
        # before we cancel it for cost control.
        time.sleep(30.0)
        running_check = _get_run_status(run_id)
        running_status = running_check.get("status")
        print(
            f"[12.4] post-30s status check: {running_status}",
            flush=True,
        )
        assert running_status in (
            "RUNNING", "QUEUED", "COMPLETED",
        ), (
            f"Run status after 30s was {running_status!r}; expected "
            f"RUNNING/QUEUED/COMPLETED. The dedup'd run may have "
            f"failed for unrelated reasons -- check diag artifacts."
        )

        # ---- 10. Cancel for cost control -----------------------------
        # Per D-12.9 budget guard: once we've proven dedup, kill
        # the run. We're already at ~$5.65 / $6.65 projected for
        # all of Step 12; saving the rest of this pipeline's cost
        # (~$0.40) keeps us comfortably under the ceiling.
        cancel_requested_at = time.time()
        if running_status not in ("COMPLETED",):
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
                f"[12.4] set Job.cancel_requested=True (cost guard)",
                flush=True,
            )
            # Also trigger V1 cancel_job to short-circuit any
            # in-flight FFmpeg from Stage 4 (shouldn't be Stage 4
            # this fast but defensive).
            try:
                import runner as v1_runner
                v1_runner.cancel_job(job_id)
            except Exception:
                pass

        # ---- 11. Poll for terminal status (max 90s) ------------------
        TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "SKIPPED"}
        terminal_deadline = time.time() + 90.0
        terminal_status: Optional[str] = None
        while time.time() < terminal_deadline:
            node = _get_run_status(run_id)
            status = node.get("status") or "QUEUED"
            if status in TERMINAL:
                terminal_status = status
                break
            time.sleep(2.0)
        terminal_status_at = time.time()

        # ---- 12. Snapshot Job state ---------------------------------
        sess = _SessionLocal()
        try:
            job = sess.query(Job).filter(Job.id == job_id).first()
            job_status = job.status if job else "<deleted>"
            job_current_stage = (
                getattr(job, "current_stage", None) if job else None
            )
            job_error_head = (
                (job.error if job and job.error else "")[:300]
            )
        finally:
            sess.close()

        # ---- 13. Manifest --------------------------------------------
        manifest = {
            "timestamp":  datetime.now().strftime("%Y%m%d_%H%M%S"),
            "mode":       "inngest_idempotency_dedup",
            "test_video": str(test_video_path),
            "inngest": {
                "app_id_uuid":     app_id_uuid,
                "from_iso":        from_iso,
                "run_id":          run_id,
                "final_status":    terminal_status,
            },
            "idempotency": {
                "event_id_used":         f"job-{job_id}",
                "send_1_result":         send_1_result,
                "send_2_result":         send_2_result,
                "time_between_sends_s":  round(send_2_at - send_1_at, 3),
                "runs_observed":         len(runs),
                "dedup_verified":        len(runs) == 1,
            },
            "execution": {
                "post_30s_status":             running_status,
                "cancel_requested_at_s":       round(
                    cancel_requested_at - send_1_at, 2,
                ),
                "terminal_status_at_s":        round(
                    terminal_status_at - send_1_at, 2,
                ),
                "wall_seconds":                round(
                    terminal_status_at - send_1_at, 2,
                ),
            },
            "job_state_at_terminal": {
                "status":          job_status,
                "current_stage":   job_current_stage,
                "error_snippet":   job_error_head,
            },
        }
        (diag_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"\n[12.4] manifest -> {diag_dir / 'manifest.json'}",
            flush=True,
        )

        # ---- 14. Final assertion (already done above; verifying
        #          additional sanity) -------------------------------
        # Terminal status should be FAILED (cancelled) or COMPLETED
        # if the dedup'd run happened to finish in the 30s window.
        assert terminal_status in {
            "FAILED", "CANCELLED", "COMPLETED",
        }, (
            f"Run {run_id} terminal status was {terminal_status!r}; "
            f"expected one of FAILED/CANCELLED/COMPLETED. See "
            f"{diag_dir}/manifest.json."
        )
