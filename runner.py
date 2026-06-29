import os
import sys
import json
import time
import threading
import subprocess
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

BASE_DIR        = Path(__file__).parent
PIPELINE_SCRIPT = BASE_DIR / "pipeline_core" / "pipeline.py"
# Use /tmp so clips survive the session but don't fill the deploy image
OUTPUT_ROOT     = Path(os.getenv("KAIZER_OUTPUT_ROOT", "/tmp/kaizer_output"))

# ─── Encoder concurrency cap ─────────────────────────────────────────
# How many pipeline subprocesses may run at once. Tuned for one
# mid-range NVIDIA GPU (1–2 NVENC engines): more than 2 simultaneous
# NVENC encodes thrash the encoder block and total throughput drops.
# On a CPU-only deploy (Railway / containers without CUDA) the OS
# scheduler shares CPU fairly, so set KAIZER_PIPELINE_CONCURRENCY=999
# (or any high number) to behave like before.
_PIPELINE_CONCURRENCY = max(1, int(os.getenv("KAIZER_PIPELINE_CONCURRENCY", "2")))
_PIPELINE_SEMAPHORE = threading.BoundedSemaphore(_PIPELINE_CONCURRENCY)


# ─── Wave 4 item I: fair render queue + disk/temp hygiene ────────────
# All of it is flag-gated / default-safe:
#   - KAIZER_FAIR_RENDER_QUEUE unset/0 → _acquire_render_slot is a bare
#     semaphore acquire — today's EXACT FIFO behaviour.
#   - KAIZER_FAIR_RENDER_QUEUE=1 → per-user round-robin ordering via
#     services.render_queue claims, with one slot reserved for
#     lane='interactive' when KAIZER_PIPELINE_CONCURRENCY >= 2.
#   - Disk guard (KAIZER_MIN_FREE_DISK_GB, default 10) refuses to start
#     a render on a nearly-full disk and fails the job with a clear
#     error instead of letting ffmpeg die midway.
#   - Startup once-only: render_queue.ensure_schema() (idempotent
#     ALTERs) + sweep of orphaned kaizer_* temp dirs older than 24h.

def _fair_queue_enabled() -> bool:
    return (os.getenv("KAIZER_FAIR_RENDER_QUEUE", "0") or "0").strip() == "1"


_STARTUP_LOCK = threading.Lock()
_STARTUP_DONE = False


def _startup_once() -> None:
    """One-time runner-side init, executed lazily on the first
    run_pipeline call (so importing runner never touches the DB).
    Every step is best-effort — failures never block a render."""
    global _STARTUP_DONE
    with _STARTUP_LOCK:
        if _STARTUP_DONE:
            return
        _STARTUP_DONE = True
    try:
        from services import render_queue
        render_queue.ensure_schema()
    except Exception as exc:
        print(f"[runner] render-queue schema init skipped: {exc}")
    try:
        from services import render_queue
        swept = render_queue.sweep_orphaned_tempdirs(max_age_hours=24)
        if swept:
            print(f"[runner] swept {swept} orphaned kaizer_* temp dir(s)")
    except Exception as exc:
        print(f"[runner] temp-dir sweep skipped: {exc}")


# Jobs whose worker threads are waiting for a fair-queue turn in THIS
# process (job_id -> lane). Scoping claims to live waiters means a
# zombie 'pending' row from a crashed process can never deadlock new
# work. Guarded by _FAIR_LOCK, as is _BATCH_RUNNING.
_FAIR_LOCK = threading.Lock()
_WAITING: dict[int, str] = {}
_BATCH_RUNNING = 0


def _worker_id() -> str:
    import socket
    try:
        host = socket.gethostname()
    except Exception:
        host = "host"
    return f"{host}-pid{os.getpid()}"[:64]


def _acquire_render_slot(job_id: int, lane: str = "batch") -> None:
    """Block until this job may start.

    Flag off (default): plain semaphore acquire — byte-identical to the
    pre-Wave-4 ordering.

    Flag on: poll services.render_queue.claim_next_render with
    only_job_id=<this job> — the claim succeeds only when this job is
    the per-user round-robin head among the jobs THIS process is
    waiting on AND its owner is under the per-user running cap. Batch
    jobs additionally cap at (concurrency - 1) running slots when
    concurrency >= 2, reserving one slot for lane='interactive'.
    Interactive jobs order among themselves (lane filter) so they can
    use the reserved slot without queueing behind the batch backlog.
    """
    global _BATCH_RUNNING
    lane = (lane or "batch").strip().lower()
    if not _fair_queue_enabled():
        _PIPELINE_SEMAPHORE.acquire()
        return

    from services import render_queue
    wid = _worker_id()
    batch_cap = (_PIPELINE_CONCURRENCY - 1
                 if _PIPELINE_CONCURRENCY >= 2 else _PIPELINE_CONCURRENCY)
    with _FAIR_LOCK:
        _WAITING[job_id] = lane
    try:
        while True:
            if lane != "interactive":
                with _FAIR_LOCK:
                    batch_full = _BATCH_RUNNING >= batch_cap
                if batch_full:
                    time.sleep(2.0)
                    continue
            with _FAIR_LOCK:
                cands = list(_WAITING.keys())
            try:
                got = render_queue.claim_next_render(
                    wid,
                    lane_filter=("interactive" if lane == "interactive" else None),
                    candidate_ids=cands,
                    only_job_id=job_id,
                )
            except Exception as exc:
                # Fail open: queue infrastructure must never wedge a
                # render — degrade to plain FIFO for this job.
                print(f"[runner] fair-queue claim failed ({exc}); "
                      f"falling back to FIFO for job {job_id}")
                got = job_id
            if got == job_id:
                break
            time.sleep(2.0)
    finally:
        with _FAIR_LOCK:
            _WAITING.pop(job_id, None)
    if lane != "interactive":
        with _FAIR_LOCK:
            _BATCH_RUNNING += 1
    _PIPELINE_SEMAPHORE.acquire()


def _release_render_slot(job_id: int, lane: str = "batch") -> None:
    """Counterpart of _acquire_render_slot — always release the
    semaphore; under the fair queue also drop the batch counter and
    clear the DB claim (best-effort; the lease expiry covers crashes)."""
    global _BATCH_RUNNING
    lane = (lane or "batch").strip().lower()
    try:
        _PIPELINE_SEMAPHORE.release()
    except ValueError:
        pass
    if _fair_queue_enabled():
        if lane != "interactive":
            with _FAIR_LOCK:
                _BATCH_RUNNING = max(0, _BATCH_RUNNING - 1)
        try:
            from services import render_queue
            render_queue.release_claim(job_id)
        except Exception:
            pass


def _disk_guard_or_fail(job_id: int, db_session_factory) -> bool:
    """Refuse to start a render when the output volume is nearly full
    (KAIZER_MIN_FREE_DISK_GB, default 10). Returns True when the render
    may proceed; on refusal marks the job failed with a clear error and
    returns False. A failed disk probe NEVER blocks (fail-open)."""
    try:
        from services import render_queue
        ok, msg = render_queue.disk_guard_ok(OUTPUT_ROOT)
    except Exception as exc:
        print(f"[runner] disk guard skipped: {exc}")
        return True
    if ok:
        return True
    print(f"[runner] disk guard BLOCKED job {job_id}: {msg}")
    try:
        from datetime import datetime as _dt, timezone as _tz
        from models import Job as _Job
        db = db_session_factory()
        try:
            j = db.query(_Job).filter(_Job.id == job_id).first()
            if j:
                j.status = "failed"
                j.error = msg
                j.finished_at = _dt.now(_tz.utc)
                db.commit()
        finally:
            db.close()
    except Exception as exc:
        print(f"[runner] disk-guard status write failed: {exc}")
    return False


# ─── User-initiated cancellation ─────────────────────────────────────
# Maps job_id → currently-running subprocess.Popen so the cancel API
# endpoint can find the process to terminate. Populated when the
# subprocess spawns, cleaned up when it exits. Thread-safe.
_ACTIVE_PROCS_LOCK = threading.Lock()
_ACTIVE_PROCS: dict[int, subprocess.Popen] = {}


# ── V2 Inngest event dispatcher — REMOVED 2026-06-17 ─────────────────
# The Inngest-orchestrated render pipeline v2 was retired; _dispatch_v2_inngest_event
# and the pipeline_v2 package are gone. run_pipeline normalises any stale
# v2/v3 platform onto V4 (the single render path).


def _register_proc(job_id: int, proc: subprocess.Popen) -> None:
    with _ACTIVE_PROCS_LOCK:
        _ACTIVE_PROCS[job_id] = proc


def _deregister_proc(job_id: int) -> None:
    with _ACTIVE_PROCS_LOCK:
        _ACTIVE_PROCS.pop(job_id, None)


def cancel_job(job_id: int) -> dict:
    """Kill the pipeline subprocess + its entire descendant tree for ``job_id``.

    Returns a small status dict the HTTP endpoint can serialise. Idempotent
    and safe to call on jobs that have already finished or never started.

    Implementation notes:
      - SIGTERM first via Popen.terminate() to give ffmpeg a chance to
        flush partial files. Wait 5 s.
      - If still alive, walk the process tree via psutil and SIGKILL
        every descendant (ffmpeg writers, helper python procs, etc.).
        Without this, killing only the parent leaves ffmpeg orphaned and
        the next render's GPU/disk locks stay held.
      - The job row's status / cancel_requested / finished_at are
        updated by the HTTP endpoint (it has the DB session). This
        function is process-control only.
    """
    with _ACTIVE_PROCS_LOCK:
        proc = _ACTIVE_PROCS.get(job_id)

    if proc is None:
        return {"job_id": job_id, "found_running": False, "killed_pids": []}

    if proc.poll() is not None:
        # Already exited on its own before we got the cancel.
        _deregister_proc(job_id)
        return {"job_id": job_id, "found_running": False, "killed_pids": []}

    killed: list[int] = []

    # Walk the process tree BEFORE terminating the parent so we don't
    # lose references after the parent reaps its children.
    descendants: list = []
    try:
        import psutil  # type: ignore
        try:
            parent = psutil.Process(proc.pid)
            descendants = parent.children(recursive=True)
        except psutil.NoSuchProcess:
            descendants = []
    except ImportError:
        # psutil is in requirements but guard anyway — without it we
        # still get the immediate child via Popen.kill().
        descendants = []

    # Graceful first.
    try:
        proc.terminate()
    except Exception:
        pass

    try:
        proc.wait(timeout=5.0)
        killed.append(proc.pid)
    except subprocess.TimeoutExpired:
        # SIGTERM ignored — escalate to SIGKILL on the whole tree.
        for child in descendants:
            try:
                child.kill()
                killed.append(child.pid)
            except Exception:
                pass
        try:
            proc.kill()
            killed.append(proc.pid)
        except Exception:
            pass
    except Exception:
        # wait() can raise on Windows when the handle is gone — still
        # try to kill descendants in case any are alive.
        for child in descendants:
            try:
                child.kill()
                killed.append(child.pid)
            except Exception:
                pass

    # Some ffmpeg children may have survived SIGTERM. Final sweep.
    for child in descendants:
        try:
            if child.is_running():
                child.kill()
                killed.append(child.pid)
        except Exception:
            pass

    _deregister_proc(job_id)
    return {"job_id": job_id, "found_running": True, "killed_pids": killed}


def run_pipeline(job_id: int, video_path: str, platform: str, frame: str,
                 db_session_factory, language: str = "te",
                 default_image: str = "",
                 default_logo: str = "",
                 bulletin_images: Optional[list] = None,
                 stt_provider: str = "",
                 transition_style: str = "smart_cut",
                 stage_2_provider: str = "gemini",
                 v4_bg_video_path: Optional[str] = None,
                 v4_bg_video_volume: float = 0.0,
                 v4_bg_intro_seconds: float = 0.0,
                 # V4 only: which model decides Step 1's KEEP/CUT plan.
                 # "claude" (default, Opus 4.7) or "gemini" (2.5 Flash on
                 # Vertex). Picked per-job in the new-job wizard so the
                 # operator can A/B quality. Forwarded to the orchestrator
                 # subprocess via KAIZER_V4_TRIM_PLANNER env var. Ignored
                 # by non-V4 platforms.
                 v4_trim_planner: str = "claude",
                 # V4 only: which provider generates story images.
                 # "auto" (default — V1 multi-source chain),
                 # "gemini" (Nano Banana), or "openai" (gpt-image-1).
                 # Forwarded via KAIZER_V4_IMAGE_PROVIDER env. Ignored
                 # by non-V4 platforms.
                 v4_image_provider: str = "auto",
                 # V4 only: operator-supplied bulletin description.
                 # When non-empty the orchestrator preserves the source
                 # video AS-IS (no Claude KEEP/CUT) and uses this text
                 # as the bulletin SEO description. Any language.
                 # Forwarded via KAIZER_V4_PREDEFINED_DESCRIPTION env.
                 v4_predefined_description: str = "",
                 # V4 only: which outputs to render.
                 #   "both"        (default) — full video (bulletin) + shorts
                 #   "full-only"   — render the bulletin, skip ALL shorts work
                 #   "shorts-only" — render shorts, skip the bulletin/full video
                 # Forwarded via KAIZER_V4_OUTPUT_FORMAT env. Ignored by
                 # non-V4 platforms. The orchestrator also persists the choice
                 # into canvas.json so editor re-renders honour it.
                 v4_output_format: str = "both",
                 # V4 Stage 2: defer the up-front compose (edit-first; export on demand).
                 # Forwarded as KAIZER_V4_DEFER_RENDER. Default off.
                 v4_defer_render: bool = False,
                 # V4 only: shorts-per-job ceiling (default 8). Forwarded as
                 # KAIZER_V4_MAX_SHORTS; the orchestrator caps the candidate list.
                 v4_max_shorts: int = 8,
                 # "Full form video" (16:9) custom template, e.g. "custom:<id>".
                 # Forwarded via KAIZER_V4_FULLFORM_LAYOUT; the orchestrator renders
                 # the full video through the custom engine instead of the bulletin.
                 fullform_layout: str = ""):
    """Launch pipeline as subprocess, stream stdout into Job.log.

    - `default_image` (non-empty absolute path) → the pipeline uses this
      image for every clip instead of fetching stock photos.
    - `default_logo` (non-empty absolute path) → the pipeline overlays this
      logo on every clip video.  Empty / missing = NO logo overlay.
    - `bulletin_images` (list of absolute paths) → user-selected images
      that the bulletin's per-story carousel will cycle through instead
      of calling OpenAI gpt-image-1. Passed to the subprocess via the
      ``KAIZER_BULLETIN_IMAGES`` env var (pipe-separated).
    - `stt_provider` (V2 only) → STT provider key from the new
      "Choose STT" wizard step. Ignored for the 4 V1 platforms;
      flows into the Inngest event data for ``full_video_shorts_v2``.

    Step 11.4 V2 branch (per Step 10 D-10.1, D-10.10):
      When ``platform == "full_video_shorts_v2"``, fire an Inngest
      ``video/v2/uploaded`` event with idempotency key
      ``f"job-{job_id}"`` and return immediately. NO subprocess
      spawn -- the Inngest worker picks up the event + runs
      ``process_video_v2``. The function-level retry policy + cancel
      bridge are wired in pipeline_v2/orchestrator.py.

      The 4 V1 platforms fall through to the existing subprocess
      path unchanged.
    """
    # V4 is the SINGLE render path (2026-06-17). The frontend already remaps
    # every platform tile to full_video_shorts_v4 + a v4_output_format (see
    # NewJob.LEGACY_TO_V4_PRESET); this is the defensive backstop for direct-API
    # / stale submissions so NOTHING reaches the retired v2/v3 dispatch or the
    # legacy V1 CLI fall-through below. Derive the output format from the
    # original tile unless the caller already set a non-default one.
    _LEGACY_TO_V4_FORMAT = {
        "instagram_reel": "shorts-only", "youtube_short": "shorts-only",
        "facebook_reel": "shorts-only", "youtube_full": "full-only",
        "youtube_full_plus_shorts": "both",
        "full_video_shorts_v2": "both", "full_video_shorts_v3": "both",
    }
    if platform != "full_video_shorts_v4" and platform in _LEGACY_TO_V4_FORMAT:
        if not (v4_output_format or "").strip() or v4_output_format == "both":
            v4_output_format = _LEGACY_TO_V4_FORMAT[platform]
        platform = "full_video_shorts_v4"

    if platform == "full_video_shorts_v4":
        # V4: trim+canvas architecture (single atomic trim, then atomic
        # canvas composite — zero lipsync drift, editable canvas.json).
        # Spawn as a detached Python subprocess; orchestrator updates
        # the Job row's status and log.
        out_dir = OUTPUT_ROOT / "full_video_shorts_v4" / f"job_{job_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Bound local render-scratch growth: keep only the most recent N
        # per-job dirs (finished masters already live in storage / R2). Never
        # touches DB-referenced media dirs. Best-effort — never blocks a render.
        try:
            from services import render_queue as _rq
            swept = _rq.sweep_old_render_outputs(OUTPUT_ROOT)
            if swept:
                print(f"[runner.v4] swept {swept} old render-output dir(s)", flush=True)
        except Exception as _exc:
            print(f"[runner.v4] render-output sweep skipped: {_exc}", flush=True)
        venv_python = str(BASE_DIR.parent / "venv" / "Scripts" / "python.exe")
        if not Path(venv_python).exists():
            venv_python = sys.executable
        cmd = [
            venv_python, "-m", "pipeline_v4.orchestrator",
            "--job-id", str(job_id),
            "--source", video_path,
            "--output-dir", str(out_dir),
            "--language", language or "te",
        ]
        if default_logo:
            cmd += ["--brand-logo", default_logo]
        log_out = open(out_dir / "stdout.log", "w", encoding="utf-8")
        log_err = open(out_dir / "stderr.log", "w", encoding="utf-8")
        # CRITICAL on Windows: the subprocess inherits cp1252 for its
        # stdout encoding by default, which crashes on any non-Latin-1
        # character (em-dash, right-arrow, Telugu text, etc.). Force
        # utf-8 so every print() inside pipeline_v4 just works.
        _env = {
            **os.environ,
            "PYTHONIOENCODING":  "utf-8",
            "PYTHONUNBUFFERED":  "1",
        }
        if bulletin_images:
            _env["KAIZER_BULLETIN_IMAGES"] = "|".join(p for p in bulletin_images if p)
        # Studio background video the user picked in the new-job wizard.
        # The orchestrator reads these env vars and stamps them onto the
        # initial canvas.json so the first render uses them.
        if v4_bg_video_path:
            _env["KAIZER_V4_BG_VIDEO_PATH"] = v4_bg_video_path
            _env["KAIZER_V4_BG_VIDEO_VOLUME"] = f"{max(0.0, min(1.0, v4_bg_video_volume or 0.0)):.3f}"
            _env["KAIZER_V4_BG_INTRO_SECONDS"] = f"{max(0.0, min(30.0, v4_bg_intro_seconds or 0.0)):.3f}"
        # KEEP/CUT planner choice — forwarded to trim_engine._select_planner.
        # Validate here so a stale frontend can't pass garbage and silently
        # downgrade quality (the planner's own fallback covers env typos).
        _planner = (v4_trim_planner or "claude").strip().lower()
        if _planner not in {"claude", "gemini"}:
            _planner = "claude"
        _env["KAIZER_V4_TRIM_PLANNER"] = _planner
        # Image-provider pick. Same validation pattern.
        _imgp = (v4_image_provider or "auto").strip().lower()
        if _imgp not in {"auto", "gemini", "openai"}:
            _imgp = "auto"
        _env["KAIZER_V4_IMAGE_PROVIDER"] = _imgp
        # Output-format pick (full video / shorts / both). Same validate-here
        # pattern so a stale frontend can't pass garbage. The orchestrator
        # falls back to canvas.output_format, then "both", if this is absent.
        _ofmt = (v4_output_format or "both").strip().lower()
        if _ofmt not in {"both", "full-only", "shorts-only"}:
            _ofmt = "both"
        _env["KAIZER_V4_OUTPUT_FORMAT"] = _ofmt
        # V4 Stage 2: per-job defer toggle (a globally-set KAIZER_V4_DEFER_RENDER is already
        # inherited via {**os.environ}). When set, the orchestrator skips the up-front compose.
        if v4_defer_render:
            _env["KAIZER_V4_DEFER_RENDER"] = "1"
        # Short template the operator picked (frame_layout). Without this the
        # orchestrator always hardcoded torn_card for shorts regardless of the
        # selection (the reported bug). Validate against the supported V1
        # shorts layouts so a stale frontend can't pass garbage.
        _slay = (frame or "torn_card").strip().lower()
        # Also allow a developer-uploaded template ("custom:<id>").
        _slay_custom = _slay.startswith("custom:") and _slay.split(":", 1)[1].isdigit()
        if _slay not in {"torn_card", "clean_card", "split_frame", "follow_bar"} and not _slay_custom:
            _slay = "torn_card"
        _env["KAIZER_V4_SHORT_LAYOUT"] = _slay
        # "Full form video" (16:9) custom template — only forwarded when it's a valid
        # custom key; otherwise the orchestrator renders the built-in bulletin.
        _ffl = (fullform_layout or "").strip().lower()
        if _ffl.startswith("custom:") and _ffl.split(":", 1)[1].isdigit():
            _env["KAIZER_V4_FULLFORM_LAYOUT"] = _ffl
        # Shorts-per-job ceiling (default 8; operator can opt into more).
        try:
            _ms = int(v4_max_shorts or 8)
        except (TypeError, ValueError):
            _ms = 8
        _env["KAIZER_V4_MAX_SHORTS"] = str(max(1, min(50, _ms)))

        # Predefined description. Encoded as base64-utf8 because env
        # vars on Windows can't reliably carry newlines + non-ASCII
        # (Telugu / Hindi). Orchestrator decodes via _decode_predef.
        _pre = (v4_predefined_description or "").strip()
        if _pre:
            import base64 as _b64
            _env["KAIZER_V4_PREDEFINED_DESCRIPTION"] = _b64.b64encode(
                _pre.encode("utf-8")
            ).decode("ascii")
        # Stash output_dir on the Job row up-front so the V4 editor can
        # find canvas.json even before the subprocess writes it. Also
        # stamp started_at so the JobDetail "elapsed - live" badge has a
        # reference time. The orchestrator updates finished_at on exit.
        try:
            from datetime import datetime as _dt, timezone as _tz
            from models import Job as _JobModel
            _db = db_session_factory()
            _j = _db.query(_JobModel).filter(_JobModel.id == job_id).first()
            if _j:
                _j.output_dir = str(out_dir)
                if _j.started_at is None:
                    _j.started_at = _dt.now(_tz.utc)
                _db.commit()
            _db.close()
        except Exception as exc:
            print(f"[runner.v4] could not stash output_dir on job: {exc}")
        def _v4_worker():
            _startup_once()
            _acquire_render_slot(job_id)
            try:
                if not _disk_guard_or_fail(job_id, db_session_factory):
                    log_out.close(); log_err.close()
                    return
                proc = subprocess.Popen(
                    cmd, cwd=str(BASE_DIR),
                    stdout=log_out, stderr=log_err,
                    env=_env,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                )
                _register_proc(job_id, proc)
                print(f"[runner.v4] subprocess spawned pid={proc.pid} job_id={job_id} -> {out_dir}", flush=True)
                try:
                    rc = proc.wait()
                    print(f"[runner.v4] job_id={job_id} subprocess exited rc={rc}", flush=True)
                finally:
                    _deregister_proc(job_id)
                    log_out.close(); log_err.close()
            finally:
                _release_render_slot(job_id)
        threading.Thread(target=_v4_worker, daemon=True).start()
        return   # V4 worker queued; subprocess runs when semaphore is available

    def _run():
        from models import Job, Clip

        db = db_session_factory()
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            job.status = "running"
            from datetime import datetime as _dt, timezone as _tz
            job.started_at = _dt.now(_tz.utc)
            db.commit()

            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

            # ── Pass list ─────────────────────────────────────────────
            # Single-platform: one entry, runs unchanged.
            # Compound ("youtube_full_plus_shorts"): TWO passes that
            # SHARE one Gemini analysis to avoid the duplicate call:
            #   1. Shorts FIRST — youtube_short has the stricter
            #      duration window (15–60 s). The clips it gets back
            #      are guaranteed to also fit the bulletin (which
            #      stitches arbitrary-length clips end to end).
            #   2. Bulletin SECOND — picks up the analysis from pass
            #      1 via KAIZER_REUSE_ANALYSIS_FROM and skips Gemini
            #      entirely. Saves ~$0.50 + ~30 s per compound job.
            # Both passes land their clips on the SAME Job row via
            # _import_clips, so the UI shows one job with mixed-
            # aspect clips.
            # Compound: ONE subprocess that runs run_compound_pipeline()
            # internally — ONE Gemini call, both passes back-to-back
            # in-process. The new pipeline writes the full + short outputs
            # under one job directory and emits [kaizer:meta] markers for
            # both, so the existing marker-streaming loop below picks
            # them up unchanged.
            if platform == "youtube_full_plus_shorts":
                passes = [("compound", "youtube_full_plus_shorts", frame or "torn_card")]
            else:
                passes = [("single", platform, frame)]

            # Shared state across all passes — log accumulates so the
            # user sees both passes' output in one job.log, and meta
            # paths get collected for the import phase at the end.
            # ``shared_analysis_path`` is set by pass 1 of a compound
            # job and consumed by pass 2 (via env var) — see below.
            log_lines: list[str] = []
            captured_meta_paths: list[str] = []
            shared_analysis_path: str = ""
            # Bulletin emits a manifest of OpenAI-generated per-story
            # images via "[kaizer:generated_images] <path>". We collect
            # these manifests and, after the pipeline succeeds, copy
            # the images into the user's UserAsset table under a
            # single ``folder_path="generated"`` folder.  The job
            # association lives on each row's ``tags`` (`"job:<id>"`)
            # so the Assets UI can display a badge without creating
            # an ever-growing tree of per-job folders.
            captured_generated_manifests: list[str] = []
            failed_pass: str = ""
            last_returncode: int = 0

            def _flush_log_to_db(lines: list[str]) -> None:
                """Persist current log_lines to job.log."""
                _db = db_session_factory()
                try:
                    _j = _db.query(Job).filter(Job.id == job_id).first()
                    if _j:
                        _j.log = "\n".join(lines)
                        _db.commit()
                finally:
                    _db.close()

            for pass_label, pass_platform, pass_frame in passes:
                # Banner so the user can grep the job log by pass.
                if len(passes) > 1:
                    log_lines.append("")
                    log_lines.append(f"════════════════════════════════════════════════")
                    log_lines.append(f"  PASS {pass_label.upper()}  "
                                     f"(--platform {pass_platform} --frame {pass_frame})")
                    log_lines.append(f"════════════════════════════════════════════════")
                    _flush_log_to_db(log_lines)

                if pass_label == "compound":
                    # Dedicated "Full Video + Shorts" pipeline.
                    # run_compound_pipeline() handles both passes itself —
                    # one Gemini call, two in-process renders. --platform
                    # / --render-mode are NOT passed; the compound function
                    # picks the right preset for each pass internally.
                    cmd = [
                        sys.executable, "-u", str(PIPELINE_SCRIPT),
                        video_path,
                        "--compound",
                        "--frame",    pass_frame,
                        "--language", language,
                    ]
                    if default_image:
                        cmd += ["--default-image", default_image]
                        # If the operator opted into the brand-image short-
                        # circuit (Phase D toggle), flip the matching CLI flag
                        # so run_compound_pipeline tells the shorts pass to
                        # use that image and skip generation.
                        cmd += ["--use-default-brand-image"]
                else:
                    cmd = [
                        sys.executable, "-u", str(PIPELINE_SCRIPT),
                        video_path,
                        "--platform", pass_platform,
                        "--frame",    pass_frame,
                        "--language", language,
                    ]
                    if default_image:
                        cmd += ["--default-image", default_image]

                # Build env. If pass 1 of a compound job captured an
                # analysis path, hand it to pass 2 via env var so the
                # pipeline skips the Gemini call entirely.
                _pass_env = {
                    **os.environ,
                    "KAIZER_OUTPUT_ROOT": str(OUTPUT_ROOT),
                    # Per-job logo path resolved from channel.logo_asset —
                    # pipeline reads this and overlays on every clip.  Empty
                    # = no logo overlay (deliberate SaaS default).
                    "KAIZER_DEFAULT_LOGO": default_logo or "",
                    # Job + user ids — passed through so the OpenAI image
                    # call wrapper (learning/openai_log.py) can stamp each
                    # logged call with the owning job/user.  Without these
                    # the admin Usage dashboard can't drill from cost →
                    # who/what burned it.
                    "KAIZER_JOB_ID":  str(job_id),
                    "KAIZER_USER_ID": str(job.user_id or 0),
                    "PYTHONUNBUFFERED":    "1",
                    "PYTHONIOENCODING":    "utf-8",
                }
                if shared_analysis_path:
                    _pass_env["KAIZER_REUSE_ANALYSIS_FROM"] = shared_analysis_path
                # Pre-selected bulletin images from the user — pipeline
                # cycles through these instead of calling OpenAI per
                # story. Pipe-separated so Windows paths (which contain
                # ':') don't break the parsing.
                if bulletin_images:
                    _pass_env["KAIZER_BULLETIN_IMAGES"] = "|".join(
                        p for p in bulletin_images if p
                    )

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    # On Windows `text=True` defaults to cp1252 which crashes as
                    # soon as the child emits Telugu / Hindi / any non-ASCII byte.
                    # Match the child's PYTHONIOENCODING=utf-8 explicitly, and
                    # replace any rogue bytes instead of raising.
                    encoding="utf-8",
                    errors="replace",
                    env=_pass_env,
                    cwd=str(BASE_DIR),
                )

                # Register so the cancel_job() helper (called by the
                # POST /api/jobs/<id>/cancel/ endpoint from a different
                # thread) can find this Popen and tree-kill it.
                _register_proc(job_id, process)

                # The COMPOUND subprocess emits TWO [kaizer:meta] markers
                # (one for the bulletin pass, one for the shorts pass);
                # the single-platform subprocess emits ONE. Collecting
                # ALL of them per-subprocess (instead of overwriting and
                # keeping only the last) lets the import step pick up
                # both meta files and the UI ends up with mixed-aspect
                # clips from both passes.
                pass_meta_paths: list[str] = []
                for line in process.stdout:
                    stripped = line.rstrip()
                    log_lines.append(stripped)
                    # Sniff for the machine-parseable markers the pipeline emits:
                    #   [kaizer:meta] <abs_path>              → editor_meta.json for clip import
                    #   [kaizer:analysis] <abs_path>          → cached Gemini analysis
                    #                                          for reuse on the next pass
                    #   [kaizer:generated_images] <abs_path>  → manifest of OpenAI-
                    #                                          generated bulletin images
                    if stripped.startswith("[kaizer:meta] "):
                        _mp = stripped[len("[kaizer:meta] "):].strip()
                        # The bulletin branch of pipeline.py prints a
                        # RELATIVE path (output\youtube_full\…) while the
                        # shorts branch prints an ABSOLUTE one. Resolve
                        # relatives against the subprocess's cwd (BASE_DIR)
                        # so _import_clips's `Path(_meta).exists()` check
                        # passes for both.
                        if _mp and not os.path.isabs(_mp):
                            _mp = str(Path(BASE_DIR) / _mp)
                        if _mp and _mp not in pass_meta_paths:
                            pass_meta_paths.append(_mp)
                    elif stripped.startswith("[kaizer:analysis] "):
                        # First pass only — keeps the earliest one if the
                        # pipeline emits more than one (it shouldn't, but
                        # we guard anyway).
                        if not shared_analysis_path:
                            shared_analysis_path = stripped[len("[kaizer:analysis] "):].strip()
                    elif stripped.startswith("[kaizer:generated_images] "):
                        _gp = stripped[len("[kaizer:generated_images] "):].strip()
                        if _gp:
                            captured_generated_manifests.append(_gp)
                    _flush_log_to_db(log_lines)
                process.wait()
                last_returncode = process.returncode
                # Deregister regardless of exit reason — the cancel
                # endpoint only cares about LIVE processes.
                _deregister_proc(job_id)

                # If the cancel endpoint killed this subprocess, treat
                # the run as user-cancelled rather than failed. The
                # returncode will be non-zero (SIGTERM / SIGKILL exits
                # negative on POSIX, 1 on Windows), but the cause is
                # user action, not a real pipeline error.
                _cancel_db = db_session_factory()
                try:
                    _cj = _cancel_db.query(Job).filter(Job.id == job_id).first()
                    _was_cancelled = bool(_cj and _cj.cancel_requested)
                finally:
                    _cancel_db.close()
                if _was_cancelled:
                    failed_pass = "cancelled"
                    break

                if process.returncode != 0:
                    failed_pass = pass_label
                    break

                # Extend (not replace) so a compound subprocess contributes
                # both passes' metas to the import phase.
                captured_meta_paths.extend(pass_meta_paths)

            # ── Import + final status ─────────────────────────────────
            db2 = db_session_factory()
            j = db2.query(Job).filter(Job.id == job_id).first()
            j.log = "\n".join(log_lines)

            if not failed_pass and captured_meta_paths:
                # Order: shorts first, bulletin (youtube_full) LAST so the
                # bulletin's parent dir wins as job.output_dir. This is
                # critical for the editor's "Images" tab which loads
                # <output_dir>/bulletin/ -- if the shorts dir wins the
                # tab shows "No bulletin images on disk yet." even when
                # they exist in the youtube_full sibling folder.
                _ordered_metas = sorted(
                    captured_meta_paths,
                    key=lambda p: 1 if "youtube_full" in p else 0,
                )
                try:
                    for _meta in _ordered_metas:
                        _import_clips(j, db2, meta_override=Path(_meta))
                    if not j.clips:
                        j.status = "failed"
                        j.error = ("Pipeline finished but 0 clips were imported. "
                                   "Check editor_meta.json and the runner log.")
                    else:
                        j.status = "done"
                        j.error = ""
                except Exception as import_err:
                    j.status = "failed"
                    j.error = f"Clip import failed: {import_err}"
                # Best-effort: import bulletin's OpenAI-generated images
                # into the user's Assets folder so they can browse what
                # was created for this job. Never fail the job over
                # asset import failure — it's a UX nicety.
                if captured_generated_manifests:
                    try:
                        _import_generated_images(
                            j, db2, captured_generated_manifests,
                            source_video_path=video_path,
                        )
                    except Exception as _ai_exc:
                        print(f"[runner] generated-image asset import skipped: {_ai_exc}")
            elif failed_pass == "cancelled":
                # User stopped the job. Don't surface it as a failure —
                # it's a deliberate action. Import any clips that DID
                # finish before the kill so partial work isn't lost.
                j.status = "cancelled"
                if captured_meta_paths:
                    try:
                        for _meta in captured_meta_paths:
                            _import_clips(j, db2, meta_override=Path(_meta))
                    except Exception as _e:
                        print(f"[runner] partial import after cancel: {_e}")
                j.error = "Cancelled by user."
            elif failed_pass:
                j.status = "failed"
                # If pass A succeeded and pass B failed, try to import
                # pass A's clips anyway so the user has partial output.
                if captured_meta_paths:
                    try:
                        for _meta in captured_meta_paths:
                            _import_clips(j, db2, meta_override=Path(_meta))
                    except Exception as _e:
                        print(f"[runner] partial import after {failed_pass} pass failure: {_e}")
                j.error = (f"Pass {failed_pass!r} failed (returncode={last_returncode}). "
                           f"Last log lines:\n" + "\n".join(log_lines[-20:]))
            else:
                j.status = "failed"
                j.error = "\n".join(log_lines[-20:])

            # Stamp completion wall-clock time on every terminal state.
            from datetime import datetime as _dt, timezone as _tz
            j.finished_at = _dt.now(_tz.utc)
            db2.commit()
            db2.close()

            # Auto-enqueue any campaigns attached to this job (Phase A).
            if not failed_pass and last_returncode == 0:
                try:
                    from campaigns import orchestrator as _campaigns_orch
                    _campaigns_orch.auto_enqueue_async(job_id)
                except Exception as e:
                    print(f"[runner] campaign auto-enqueue skipped: {e}")

        except Exception as e:
            db3 = db_session_factory()
            j = db3.query(Job).filter(Job.id == job_id).first()
            if j:
                from datetime import datetime as _dt, timezone as _tz
                j.status = "failed"
                j.error = str(e)
                j.finished_at = _dt.now(_tz.utc)
                db3.commit()
            db3.close()
        finally:
            # ─── Cleanup ────────────────────────────────────────────────
            # Railway's container has a small ephemeral disk cap. In R2
            # mode (STORAGE_BACKEND=r2) every clip + thumb + image is
            # already in R2 by this point — the local disk copies are
            # dead weight and would fill the container.
            #
            # In local-storage mode (STORAGE_BACKEND=local — dev path)
            # the local files ARE the storage; deleting them breaks
            # playback in the frontend. So we SKIP the output-dir
            # cleanup in local mode. Only the source-video upload from
            # MEDIA_ROOT/uploads/ is dropped either way (we don't need
            # the raw input after rendering, regardless of backend).
            try:
                import shutil as _shutil
                # 1. Source video that create_job dropped in MEDIA_ROOT/uploads/
                if video_path and os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                    except OSError as _e:
                        print(f"[runner] cleanup: failed to remove source {video_path!r}: {_e}")
                # 2. Pipeline output dir — render outputs are NEVER auto-
                # deleted by default. The operator prunes old renders manually
                # (explicit request, 2026-06-16). Auto-delete only happens when
                # BOTH (a) an external backend owns the bytes
                # (STORAGE_BACKEND != local, e.g. R2 on an ephemeral container)
                # AND (b) the operator explicitly opts in with
                # KAIZER_DELETE_RENDER_OUTPUT=1. Otherwise every render stays on
                # disk so the canvas editor / publish flow can always find it.
                _backend = (os.environ.get("STORAGE_BACKEND", "local") or "").strip().lower()
                _allow_output_delete = (
                    os.environ.get("KAIZER_DELETE_RENDER_OUTPUT", "0").strip().lower()
                    in ("1", "true", "yes", "on")
                )
                if _backend != "local" and _allow_output_delete:
                    try:
                        db_clean = db_session_factory()
                        j_clean = db_clean.query(Job).filter(Job.id == job_id).first()
                        out_dir = j_clean.output_dir if j_clean else ""
                        db_clean.close()
                        if out_dir and os.path.isdir(out_dir):
                            # Don't delete OUTPUT_ROOT itself, only the per-job subfolder
                            if os.path.abspath(out_dir) != os.path.abspath(str(OUTPUT_ROOT)):
                                _shutil.rmtree(out_dir, ignore_errors=True)
                                print(f"[runner] cleanup: dropped pipeline output {out_dir!r}")
                    except Exception as _e:
                        print(f"[runner] cleanup: output_dir removal warning: {_e}")
                else:
                    print("[runner] cleanup: keeping render output on disk "
                          "(auto-delete disabled — prune manually)")
            except Exception as _cleanup_e:
                # Cleanup failures are never fatal — container will get
                # wiped on next redeploy worst case.
                print(f"[runner] cleanup warning: {_cleanup_e}")
            db.close()

    def _run_throttled():
        """Wrap _run so concurrent pipeline jobs respect the encoder cap.

        Without this every uploaded job would race straight into NVENC,
        thrash the GPU's 1–2 encoder engines, and slow each other down.
        We mark the row 'queued' first so the frontend shows the wait
        state, then block on the semaphore until a slot frees up, then
        run the actual pipeline subprocess via _run.
        """
        _startup_once()
        try:
            db_pre = db_session_factory()
            from models import Job as _Job
            j_pre = db_pre.query(_Job).filter(_Job.id == job_id).first()
            if j_pre and j_pre.status not in ("running", "done", "failed"):
                j_pre.status = "queued"
                db_pre.commit()
            db_pre.close()
        except Exception as _e:
            # Status update is best-effort; don't block the run on it.
            print(f"[runner] queued-status update skipped: {_e}")

        _acquire_render_slot(job_id)
        try:
            if not _disk_guard_or_fail(job_id, db_session_factory):
                return
            _run()
        finally:
            _release_render_slot(job_id)

    threading.Thread(target=_run_throttled, daemon=True).start()


def _import_generated_images(job, db, manifest_paths: list[str],
                              source_video_path: str = "") -> int:
    """Import bulletin-generated OpenAI images into the user's Assets table.

    Reads each manifest JSON (written by pipeline.py's bulletin block,
    signaled via ``[kaizer:generated_images] <path>`` on stdout). For
    each entry, uploads the image bytes into the configured storage
    backend (R2 in prod, local disk in dev) under
    ``user_assets/{user_id}/generated/job_{job_id}/{filename}`` and
    creates a UserAsset row with ``folder_path="generated"`` and a
    ``"job:<id>"`` tag so the Assets UI can render a job badge
    without spawning a fresh subfolder per job.

    When ``source_video_path`` points at the job's original source file,
    we stamp every imported asset with its content fingerprint
    (``UserAsset.source_video_hash``).  Re-uploads of the same source
    can then look up these rows and offer "reuse instead of regenerate"
    without burning gpt-image-1 quota again.

    Returns the count of assets created. Failure to import any single
    image is logged but does not abort — this is a UX nicety, never
    fatal to the job.
    """
    from models import UserAsset
    from pipeline_core.storage import get_storage_provider

    try:
        storage = get_storage_provider()
    except Exception as exc:
        print(f"[runner] generated-images: storage provider unavailable ({exc}); skipping")
        return 0

    # Compute the source-video fingerprint ONCE for this whole import
    # pass (same hash function gemini_cache uses, so the value matches
    # what the cache lookups produce).  Empty string when the source
    # file isn't readable — assets still get imported, just without
    # the link back to a re-uploadable source.
    src_hash = ""
    if source_video_path and os.path.isfile(source_video_path):
        try:
            from gemini_cache import hash_file_prefix
            src_hash = hash_file_prefix(source_video_path)
        except Exception as exc:
            print(f"[runner] generated-images: hash source video failed ({exc}); "
                  f"assets will be created WITHOUT source_video_hash")

    user_id = job.user_id
    job_id  = job.id
    # Single flat folder for every generated image across every job.
    # The job association is preserved on each row's ``tags`` list
    # (``"job:<id>"``) so the UI can group/filter without an
    # ever-expanding folder tree.
    folder  = "generated"
    created = 0

    for mp in manifest_paths:
        mp_path = Path(mp)
        if not mp_path.is_file():
            print(f"[runner] generated-images: manifest {mp!r} not found; skipping")
            continue
        try:
            entries = json.loads(mp_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[runner] generated-images: manifest {mp!r} unreadable ({exc})")
            continue

        for entry in entries:
            local = entry.get("path", "")
            fname = entry.get("filename", "") or (Path(local).name if local else "")
            if not local or not Path(local).is_file() or not fname:
                continue
            # Skip if a row with the same (user_id, folder_path, filename) already
            # exists — idempotency for re-runs / retries on the same job.
            existing = (db.query(UserAsset)
                          .filter(UserAsset.user_id == user_id,
                                  UserAsset.folder_path == folder,
                                  UserAsset.filename == fname)
                          .first())
            if existing:
                continue

            # Unique key per story image so two stories' news_01.jpg
            # files don't collide. The pipeline writes them under
            # bulletin_dir/story_NN_assets/images/, but we lose that
            # structure once they're flattened into UserAssets — so we
            # mint a fresh key by joining the job id + a numeric suffix.
            safe_key = f"user_assets/{user_id}/generated/job_{job_id}/{Path(local).parent.name}_{fname}"
            try:
                obj = storage.upload(local, safe_key, content_type="image/jpeg")
                stored_url     = obj.url
                stored_key     = obj.key
                stored_backend = storage.name
            except Exception as exc:
                print(f"[runner] generated-images: upload failed for {local!r}: {exc}")
                stored_url = ""
                stored_key = ""
                stored_backend = ""

            try:
                row = UserAsset(
                    user_id=user_id,
                    filename=fname,
                    file_path=str(Path(local).resolve()),
                    folder_path=folder,
                    tags=[f"job:{job_id}", "generated", "bulletin"],
                    storage_url=stored_url,
                    storage_key=stored_key,
                    storage_backend=stored_backend,
                    source_video_hash=src_hash,
                )
                db.add(row)
                db.commit()
                created += 1
            except Exception as exc:
                db.rollback()
                print(f"[runner] generated-images: DB insert failed for {fname!r}: {exc}")

    if created:
        print(f"[runner] generated-images: created {created} UserAsset row(s) "
              f"under folder_path={folder!r}")
    return created


def _import_clips(job, db, meta_override: Path | None = None):
    """Find editor_meta.json, read it as UTF-8, create Clip rows.

    Priority for locating the metadata file:
      1. Explicit `meta_override` (set by the runner from the `[kaizer:meta]`
         marker line emitted by pipeline.py — canonical source).
      2. `job.output_dir` if it points directly at a directory containing the file.
      3. rglob from `job.output_dir`, then rglob from the global OUTPUT_ROOT.

    Exceptions propagate to the caller so the runner can mark the job failed
    with a useful message instead of silently ending up with zero clips.
    """
    from models import Clip

    meta_path: Path | None = None

    if meta_override and Path(meta_override).exists():
        meta_path = Path(meta_override)
    else:
        search_root = Path(job.output_dir) if job.output_dir else OUTPUT_ROOT
        # Direct-hit: the dir itself contains editor_meta.json
        direct = search_root / "editor_meta.json"
        if direct.exists():
            meta_path = direct
        else:
            for p in search_root.rglob("editor_meta.json"):
                meta_path = p
                break
        # Last-resort rglob over OUTPUT_ROOT used to silently pick up the
        # FIRST editor_meta.json it found anywhere -- which on a busy box
        # is almost always a different job. That caused V3 jobs (which
        # write editor_meta.json now, item 120) to occasionally import
        # clips from an unrelated V2 job after a search_root miss.
        # Scope the fallback to a per-job-id substring so we can only
        # accidentally match this job's directory.
        if not meta_path or not meta_path.exists():
            job_id_token = f"job_{job.id}"
            for p in OUTPUT_ROOT.rglob("editor_meta.json"):
                if job_id_token in str(p):
                    meta_path = p
                    break

    if not meta_path or not meta_path.exists():
        raise FileNotFoundError(
            f"editor_meta.json not found (searched override={meta_override!r}, "
            f"job.output_dir={job.output_dir!r}, OUTPUT_ROOT={OUTPUT_ROOT!s})"
        )

    # Must be utf-8 — editor_meta.json now contains native-script fields
    # (Telugu, Devanagari, Tamil, …).  Windows default cp1252 will crash.
    raw = meta_path.read_text(encoding="utf-8")
    data = json.loads(raw)

    # Lazy R2 import — pipeline-emitted images get mirrored to R2 so they
    # survive container restarts AND don't break when the user opens the
    # clip from a different machine. Each image lands at a clip-specific
    # key (clips/{job_id}/{clip_index}/<filename>) so the mapping is
    # permanent — even if the user later changes their default ad asset
    # or deletes the source file, the clip's image_storage_url still
    # resolves to the exact image used at render time.
    try:
        from pipeline_core.storage import get_storage_provider
        # Honours STORAGE_BACKEND — local writes to ``output/``, prod
        # ships to R2. Variable name kept as `_r2` for blame-friendliness
        # with the existing call sites; the actual backend is whatever
        # STORAGE_BACKEND resolves to.
        _r2 = get_storage_provider()
    except Exception as exc:
        print(f"[runner] storage provider unavailable, images will only live "
              f"on local disk: {exc}")
        _r2 = None

    def _r2_upload(local_path: str, key: str, ct: str) -> str:
        """Upload to the configured storage with a permanent clip-
        specific key. Returns URL or empty string on failure / no
        backend / no file."""
        if not _r2 or not local_path or not Path(local_path).exists():
            return ""
        try:
            return _r2.upload(local_path, key, content_type=ct).url
        except Exception as exc:
            print(f"[runner] R2 upload failed for {local_path!r}: {exc}")
            return ""

    clips_data = data.get("clips", [])
    imported = 0
    skipped  = 0
    for i, c in enumerate(clips_data):
        clip_path = c.get("clip_path", "")
        # Drop clips whose mp4 file is missing on disk — avoids broken cards
        # in the UI that 404 on every thumb + video fetch.
        if not clip_path or not Path(clip_path).exists():
            print(f"[runner] skipping clip {i}: file missing ({clip_path!r})")
            skipped += 1
            continue

        thumb_path = c.get("thumb_path", "") or ""
        image_path = c.get("image_path", "") or ""

        # Pipeline already populates storage_* for the video clip. We add
        # mirrors for the thumbnail + the editorial image so all three
        # bytes-on-disk artefacts of this clip live in R2.
        thumb_storage_url = _r2_upload(
            thumb_path,
            f"clips/{job.id}/{i:02d}/{Path(thumb_path).name}" if thumb_path else "",
            "image/jpeg",
        )
        image_storage_url = ""
        if image_path:
            ext = Path(image_path).suffix.lower()
            ct = {".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}.get(ext, "image/jpeg")
            image_storage_url = _r2_upload(
                image_path,
                f"clips/{job.id}/{i:02d}/{Path(image_path).name}",
                ct,
            )

        clip = Clip(
            job_id=job.id,
            clip_index=i,
            filename=Path(clip_path).name,
            file_path=clip_path,
            thumb_path=thumb_path,
            image_path=image_path,
            duration=float(c.get("duration", 0)),
            frame_type=c.get("frame_type", ""),
            text=c.get("text", ""),
            sentiment=c.get("sentiment", ""),
            entities=json.dumps(c.get("entities", [])),
            card_params=json.dumps(c.get("card_params", {})),
            section_pct=json.dumps(c.get("section_pct", {})),
            follow_params=json.dumps(c.get("follow_params", {})),
            meta=json.dumps(c),
            # Phase 5 storage fields — empty strings when backend is local
            storage_url=c.get("storage_url", ""),
            storage_key=c.get("storage_key", ""),
            storage_backend=c.get("storage_backend", ""),
            # Image mirrors — permanent clip-specific R2 keys so the
            # mapping survives default-asset rotation and container
            # rebuilds.
            thumb_storage_url=thumb_storage_url,
            image_storage_url=image_storage_url,
        )
        db.add(clip)
        imported += 1

    job.output_dir = str(meta_path.parent)
    db.commit()
    print(f"[runner] imported {imported} clip(s), skipped {skipped} for job {job.id}")
