"""Step 12.2b E2E test — Inngest Dev Server dispatch path.

Validates the V2 pipeline executes end-to-end when driven by REAL
Inngest event delivery (vs Step 12.2a which directly drives the
handler functions in-process). This catches a class of bugs that
12.2a cannot:

  * Pydantic state ACTUALLY round-trips through Inngest's event store
    at every step boundary (Inngest serializes each step's return
    to JSON between yields)
  * Function-level retries=2 applies in practice
  * NonRetriableError actually short-circuits the retry policy
  * Idempotency key actually dedupes (covered in Step 12.4)
  * The webhook contract (POST /api/inngest with fnId+stepId) is
    correctly serviced by the mounted inngest.fast_api.serve handler

Setup requirements (skips the test if missing):

  * KAIZER_RUN_INTEGRATION_TESTS=1                env-var opt-in
  * GEMINI_API_KEY + DEEPGRAM_API_KEY (+ R2_*)    same as 12.2a
  * ``inngest`` CLI on PATH                       inngest dev binary
  * ``inngest dev`` listening on :8288            dashboard + executor
  * uvicorn running on :8000 with KAIZER_V2_ENABLED=1
    serving main.py's /api/inngest handler. Manual launch:

        KAIZER_V2_ENABLED=1 python -m uvicorn main:app \\
            --host 127.0.0.1 --port 8000

The test does NOT spawn either of those servers itself -- they're
expected to be running already (see step 12.2b setup verification
procedure). The fixture probes both before running.

Diagnostic artifact dir (per run):
  pipeline_v2/tests/fixtures/step12_diag/12_2b_inngest/<YYYYMMDD_HHMMSS>/
    inngest_run_id.txt
    manifest.json
    envelope_final.json
    failure_trace.json   (only if status != COMPLETED)
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

# Local-host endpoints. The test ASSUMES these are reachable -- it
# probes them in a module-level fixture and SKIPs if not.
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
        "V2 Inngest E2E integration tests skipped. Set "
        "KAIZER_RUN_INTEGRATION_TESTS=1 plus the required API keys "
        "to enable. See module docstring."
    ),
)


# ====================================================================== #
# Local-service probes (uvicorn + inngest dev must be running)            #
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


# ====================================================================== #
# Inngest GraphQL helpers                                                 #
# ====================================================================== #


def _gql(query: str) -> dict:
    """POST a GraphQL query to the Inngest dev server.

    Returns the parsed response body. Raises RuntimeError on
    transport errors or GraphQL-level errors -- this is the test's
    only path to inspect run state, so silent failure here would
    mask real bugs.

    HTTP 4xx responses (e.g. 422 on GraphQL parse error) carry the
    actual error body but urllib raises HTTPError before we'd see
    it. We catch + re-raise with the body included so the failure
    log shows the actual GraphQL error message, not just the HTTP
    status code.
    """
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
    """Find the most recent run for ``event_name`` queued at >= from_iso.

    Returns a dict {id, status, eventName, queuedAt, startedAt,
    endedAt, output} when one exists; None when no matching run yet.
    """
    filter_parts = ['from: "' + from_iso + '"']
    if app_id_uuid:
        filter_parts.append('appIDs: ["' + app_id_uuid + '"]')
    filter_str = ", ".join(filter_parts)
    # Built via concatenation rather than f-string so the literal
    # GraphQL braces don't have to be escaped as {{/}} -- that
    # escaping previously produced a 6-opens-vs-9-closes mismatch
    # that the Inngest dev parser 422'd on.
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
    """The kaizer-v2 app's UUID per the dashboard. Used to scope
    the runs filter to our app only.
    """
    data = _gql(
        '{ apps { id name } }'
    )
    for app in (data.get("apps") or []):
        if app.get("name") == "kaizer-v2":
            return app.get("id")
    return None


def _get_run_status(run_id: str) -> dict:
    """Fetch the current state of a specific run via Query.run."""
    query = (
        '{ run(runID: "' + run_id + '") { id status startedAt endedAt '
        "output appID functionID } }"
    )
    data = _gql(query)
    return data.get("run") or {}


def _get_run_trace(run_id: str) -> dict:
    """Fetch the full step-by-step trace of a run. Used for the
    failure_trace.json diag artifact when the run doesn't complete.
    """
    query = (
        '{ run(runID: "' + run_id + '") { id status trace { '
        "spanID name status startedAt endedAt attempts "
        "outputID outputCount } } }"
    )
    try:
        data = _gql(query)
    except RuntimeError:
        # Trace shape can vary across Inngest dev versions; if the
        # query 400s, we still have status from the basic poll.
        return {}
    return data.get("run") or {}


# ====================================================================== #
# Module-level fixtures                                                   #
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
        pytest.skip(
            f"Required API keys missing: {missing}."
        )
    return keys


@pytest.fixture(scope="module")
def services_ready():
    """Bail with a clear skip reason if the two local services
    aren't both up. The 12.2b setup-verification procedure (see
    module docstring) covers how to start them.
    """
    if not _inngest_cli_available():
        pytest.skip("inngest CLI not on PATH. Install per setup guide.")
    if not _tcp_listening("127.0.0.1", INNGEST_DEV_PORT):
        pytest.skip(
            f"inngest dev not listening on :{INNGEST_DEV_PORT}. "
            f"Run: inngest dev -u {UVICORN_BASE}/api/inngest"
        )
    if not _uvicorn_ready():
        pytest.skip(
            f"uvicorn not responding at {UVICORN_BASE}/api/health/. "
            f"Run: KAIZER_V2_ENABLED=1 python -m uvicorn main:app "
            f"--host 127.0.0.1 --port 8000"
        )
    if not _uvicorn_serves_inngest():
        pytest.skip(
            f"uvicorn at {UVICORN_BASE} doesn't serve /api/inngest "
            f"with function_count >= 1. Confirm KAIZER_V2_ENABLED=1."
        )
    return True


@pytest.fixture(scope="module")
def diag_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (
        Path(__file__).resolve().parent
        / "fixtures" / "step12_diag" / "12_2b_inngest" / ts
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


# ====================================================================== #
# Step 12.2b E2E test                                                     #
# ====================================================================== #


class TestE2EInngestDispatch_12_2b:
    """Send a real ``video/v2/uploaded`` event via the Inngest client
    (production code path), poll the Inngest dev server's GraphQL
    until the run reaches a terminal status, then verify the 7
    acceptance checks against the captured function output.
    """

    # Pick a Job.id unlikely to collide with anything else in the
    # live DB. The test creates+cleans up this row in the production
    # SQLite (or whatever DATABASE_URL points to).
    _TEST_JOB_ID = 99_120_022_0  # "step 12.2b" mnemonic

    @pytest.fixture
    def db_job_row(self, test_video_path: Path):
        """Create a Job row in the live DB, yield (job_id, SessionLocal),
        delete the row + any child Clip rows in teardown.

        Cascade note: the V2 finalize step imports Clip rows referencing
        Job.id via the ``clips_job_id_fkey`` FK constraint. PostgreSQL
        enforces RESTRICT-by-default, so the Job row can't be deleted
        until all referencing Clips are gone. We delete Clips first,
        then the Job. (V1 production code paths use ON DELETE CASCADE
        at the V1 cleanup endpoints; tests do it explicitly.)
        """
        # Lazy imports: KaizerBackend on sys.path via conftest.
        from database import SessionLocal
        from models import Job, Clip

        # Wipe any stale rows from a prior aborted run (Clips before Job
        # to honour the FK), then insert the fresh Job.
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
            # Teardown: same FK-aware order -- Clips first, then Job.
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

    def test_real_event_dispatch_runs_to_completion(
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

        # ---- 1. Time anchor for runs filter -------------------------
        # `runs.filter.from` is REQUIRED. Capture a timestamp just
        # before send() so the poll only sees runs from this test.
        t0 = datetime.now(timezone.utc).replace(microsecond=0)
        from_iso = t0.isoformat().replace("+00:00", "Z")

        # ---- 2. Find the kaizer-v2 app UUID -------------------------
        app_id_uuid = _get_app_id_uuid()
        assert app_id_uuid, (
            "kaizer-v2 app not registered with Inngest dev. "
            "uvicorn + inngest dev sync may have failed."
        )

        # ---- 3. Dispatch event (production code path) ---------------
        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Idempotency key per D-10.10. In production, runner.py uses
        # the DB-assigned auto-increment Job.id which is unique per
        # submission. The test fixture reuses ``_TEST_JOB_ID`` across
        # repeated runs, so we suffix a millisecond timestamp to keep
        # each test attempt's idempotency key unique. Without this,
        # re-running the test within the 24h idempotency window gets
        # silently deduped (see Step 12.2b re-run #3 root cause).
        # The 12.4 idempotency test deliberately uses a constant id
        # to validate the dedup-as-feature contract.
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
        # Use time.time() (Unix epoch) for all timing/elapsed
        # calculations in this test. The polling loop below reads
        # time.time() and computes elapsed deltas; mixing in
        # time.perf_counter() (monotonic clock, returns seconds since
        # boot on POSIX or seconds since high-precision start on
        # Windows) produces nonsense subtractions
        # (1779167418s deltas in earlier runs).
        dispatch_started = time.time()
        # SDK 0.5.18 exposes both Inngest.send (async, returns
        # list[str]) and Inngest.send_sync (sync, same return).
        # This test body is plain sync, so we use send_sync --
        # matching runner.py's V2 dispatcher pattern. See backlog
        # item 70 for the pick-by-caller-context guidance.
        sent_event_ids = client.send_sync(events=[event])
        assert sent_event_ids, "Inngest.send_sync returned no event IDs"
        print(
            f"\n[12.2b] sent event id(s)={sent_event_ids}, "
            f"app_id_uuid={app_id_uuid}",
            flush=True,
        )

        # ---- 4. Discover the run for this event ---------------------
        # Defensive: if runs() returns 0 results for 30s, dispatch
        # broke -- bail fast with the diagnostic snapshot.
        run_node: Optional[dict] = None
        discover_deadline = time.time() + 30.0
        while time.time() < discover_deadline:
            run_node = _find_run_for_event(
                "video/v2/uploaded", from_iso, app_id_uuid=app_id_uuid,
            )
            if run_node:
                break
            time.sleep(1.0)
        if not run_node:
            (diag_dir / "dispatch_failure.json").write_text(
                json.dumps({
                    "event": "no_run_discovered_in_30s",
                    "sent_event_ids":  sent_event_ids,
                    "app_id_uuid":     app_id_uuid,
                    "from_iso":        from_iso,
                }, indent=2),
                encoding="utf-8",
            )
            pytest.fail(
                "Inngest accepted the event but no run was created "
                "within 30s. Check inngest dev logs + uvicorn logs "
                "for sync errors."
            )
        run_id = run_node["id"]
        (diag_dir / "inngest_run_id.txt").write_text(run_id, encoding="utf-8")
        print(
            f"[12.2b] discovered run_id={run_id} status={run_node['status']}",
            flush=True,
        )

        # ---- 5. Poll to terminal status -----------------------------
        TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "SKIPPED"}
        timeout_s = 25 * 60  # 25 min
        poll_interval_s = 5.0
        progress_interval_s = 30.0
        deadline = time.time() + timeout_s
        last_status = run_node["status"]
        last_progress_log = 0.0
        first_running_at: Optional[float] = None
        queued_warned = False
        terminal_node: Optional[dict] = None

        while time.time() < deadline:
            node = _get_run_status(run_id)
            status = node.get("status") or "QUEUED"
            now = time.time()

            if status == "RUNNING" and first_running_at is None:
                first_running_at = now

            if (
                status == "QUEUED"
                and not queued_warned
                and (now - dispatch_started) > 120
            ):
                queued_warned = True
                print(
                    "[12.2b][warn] run still QUEUED after 2 min; "
                    "inngest dev may have hung waiting for the SDK "
                    "to respond. Continuing to poll.",
                    flush=True,
                )

            if status != last_status or (
                now - last_progress_log > progress_interval_s
            ):
                elapsed = now - dispatch_started
                print(
                    f"[12.2b] poll t+{elapsed:.0f}s status={status}",
                    flush=True,
                )
                last_status = status
                last_progress_log = now

            if status in TERMINAL:
                terminal_node = node
                break
            time.sleep(poll_interval_s)

        wall_seconds = time.time() - dispatch_started
        dispatch_latency_s: Optional[float] = None
        if first_running_at is not None:
            dispatch_latency_s = first_running_at - dispatch_started

        if terminal_node is None:
            # 25 min elapsed without terminal status -- capture trace
            # + bail. Diag prep mirrors the failure path below.
            trace = _get_run_trace(run_id)
            (diag_dir / "failure_trace.json").write_text(
                json.dumps({
                    "event":       "timeout_25min",
                    "last_status": last_status,
                    "trace":       trace,
                }, indent=2),
                encoding="utf-8",
            )
            pytest.fail(
                f"Run {run_id} did not reach terminal status within "
                f"25 minutes. Last status: {last_status}."
            )

        # ---- 6. Capture diag + run acceptance checks ----------------
        final_status = terminal_node["status"]
        output_raw = terminal_node.get("output")

        envelope_dict: Optional[dict] = None
        if output_raw:
            try:
                # Inngest's `output` is the function's return value
                # serialized to JSON. The orchestrator returns the
                # envelope dict at the end of process_video_v2.
                envelope_dict = (
                    output_raw if isinstance(output_raw, dict)
                    else json.loads(output_raw)
                )
            except (json.JSONDecodeError, TypeError) as exc:
                envelope_dict = {
                    "_parse_error":  str(exc),
                    "_raw":          str(output_raw)[:2000],
                }

        if envelope_dict is not None:
            (diag_dir / "envelope_final.json").write_text(
                json.dumps(envelope_dict, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        if final_status != "COMPLETED":
            # Capture trace for diagnosis. Don't pytest.fail yet --
            # write the manifest first so the operator has the full
            # picture before raising.
            trace = _get_run_trace(run_id)
            (diag_dir / "failure_trace.json").write_text(
                json.dumps({
                    "status":         final_status,
                    "trace":          trace,
                    "terminal_node":  terminal_node,
                }, indent=2),
                encoding="utf-8",
            )

        # ---- 7. Acceptance checks (mirror 12.2a) --------------------
        checks: dict = {}
        if envelope_dict and isinstance(envelope_dict, dict):
            stage_0 = envelope_dict.get("stage_0") or {}
            stage_1 = envelope_dict.get("stage_1") or {}
            stage_2 = envelope_dict.get("stage_2") or {}
            stage_2_5 = envelope_dict.get("stage_2_5") or {}
            stage_3 = envelope_dict.get("stage_3") or {}
            stage_4 = envelope_dict.get("stage_4") or {}
            finalize = envelope_dict.get("finalize") or {}

            mezz_path = stage_0.get("mezzanine_path", "")
            mezz_ok = bool(mezz_path) and Path(mezz_path).is_file() and \
                Path(mezz_path).stat().st_size > 100_000
            checks["stage_0_mezzanine"] = "PASS" if mezz_ok else "FAIL"

            word_count = len(
                (stage_1.get("transcript") or {}).get("words") or []
            )
            checks["stage_1_word_array"] = (
                "PASS" if word_count >= 100 else "FAIL"
            )
            checks["stage_2_decisions"] = (
                "PASS" if stage_2.get("full_video_cuts") else "FAIL"
            )

            entities = stage_2_5.get("entities") or []
            valid_types = {"PERSON", "ORG", "PLACE", "EVENT", "OTHER"}
            checks["stage_2_5_entities"] = (
                "PASS"
                if 1 <= len(entities) <= 6
                and all(e.get("type") in valid_types for e in entities)
                else "FAIL"
            )

            shorts_cuts = stage_3.get("shorts_cuts") or []
            durations_ok = all(
                15.0 <= (s.get("end_sec", 0) - s.get("start_sec", 0)) <= 60.0
                for s in shorts_cuts
            )
            checks["stage_3_fanout"] = (
                "PASS" if 3 <= len(shorts_cuts) <= 10 and durations_ok
                else "FAIL"
            )

            bulletin_path = (
                stage_4.get("bulletin") or {}
            ).get("bulletin_path", "")
            checks["stage_4_render"] = (
                "PASS"
                if (
                    bulletin_path
                    and Path(bulletin_path).is_file()
                    and Path(bulletin_path).stat().st_size > 100_000
                ) else "FAIL"
            )

            # finalize_db: cross-reference the LIVE DB Job row
            from models import Job
            sess = _SessionLocal()
            try:
                job = sess.query(Job).filter(Job.id == job_id).first()
                db_status = (job.status if job else "")
                db_current_stage = (
                    getattr(job, "current_stage", None) if job else "?"
                )
            finally:
                sess.close()
            checks["finalize_db"] = (
                "PASS"
                if (
                    finalize.get("status") == "done"
                    and db_status == "done"
                    and db_current_stage is None
                ) else "FAIL"
            )
        else:
            for k in (
                "stage_0_mezzanine", "stage_1_word_array",
                "stage_2_decisions", "stage_2_5_entities",
                "stage_3_fanout",   "stage_4_render", "finalize_db",
            ):
                checks[k] = "FAIL"

        # ---- 8. Manifest --------------------------------------------
        stage_costs = (envelope_dict or {}).get("stage_costs") or {}
        total_cost = sum(float(v) for v in stage_costs.values() if v is not None)
        manifest = {
            "timestamp":  datetime.now().strftime("%Y%m%d_%H%M%S"),
            "mode":       "inngest",
            "test_video": str(test_video_path),
            "stt_provider": "deepgram",
            "inngest": {
                "run_id":            run_id,
                "app_id_uuid":       app_id_uuid,
                "sent_event_ids":    sent_event_ids,
                "from_iso":          from_iso,
                "final_status":      final_status,
                "dispatch_latency_s": (
                    round(dispatch_latency_s, 2)
                    if dispatch_latency_s is not None else None
                ),
            },
            "wall_seconds": round(wall_seconds, 2),
            "cost_usd": {
                **{k: round(float(v), 4) for k, v in stage_costs.items()},
                "TOTAL": round(total_cost, 4),
            },
            "checks": checks,
        }
        (diag_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"\n[12.2b] manifest -> {diag_dir / 'manifest.json'}",
            flush=True,
        )

        # ---- 9. Final assertion ------------------------------------
        assert final_status == "COMPLETED", (
            f"Inngest run {run_id} ended with status={final_status}; "
            f"expected COMPLETED. See {diag_dir}/failure_trace.json."
        )
        failed_checks = [k for k, v in checks.items() if v != "PASS"]
        assert not failed_checks, (
            f"Acceptance check(s) failed: {failed_checks}. "
            f"Manifest at {diag_dir}/manifest.json."
        )
