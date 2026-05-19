import os
import sys
import json
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


# ─── User-initiated cancellation ─────────────────────────────────────
# Maps job_id → currently-running subprocess.Popen so the cancel API
# endpoint can find the process to terminate. Populated when the
# subprocess spawns, cleaned up when it exits. Thread-safe.
_ACTIVE_PROCS_LOCK = threading.Lock()
_ACTIVE_PROCS: dict[int, subprocess.Popen] = {}


# ── Step 11.4: V2 Inngest event dispatcher ───────────────────────────
def _dispatch_v2_inngest_event(
    *,
    job_id: int,
    video_path: str,
    language: str,
    platform: str,
    frame: str,
    stt_provider: str,
    db_session_factory,
) -> None:
    """Fire ``video/v2/uploaded`` so the Inngest V2 worker picks up.

    Idempotency key per Step 10 D-10.10: ``f"job-{job_id}"`` --
    duplicate sends within Inngest's window are deduplicated.

    Lazy import of pipeline_v2.inngest_client (i) avoids a
    module-load circular if V2 wiring evolves and (ii) means
    legacy V1-only runners that never call this path don't pay the
    inngest SDK import cost.

    Marks Job.status='running' synchronously so the UI shows the
    job leaving 'pending' immediately, even before the Inngest
    worker picks up the event. (The worker writes Job.current_stage
    + final status as it progresses.)
    """
    # Lazy imports to keep V1-only call sites unaffected.
    import sys as _sys
    import os as _os
    _pipeline_v2_dir = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)),
        "pipeline_v2",
    )
    if _pipeline_v2_dir not in _sys.path:
        _sys.path.insert(0, _pipeline_v2_dir)
    from inngest import Event
    from pipeline_v2.inngest_client import get_client

    # Mark the job as running + reset cancel flag in DB so the UI's
    # status badge flips off "pending" right away.
    try:
        from models import Job
        from datetime import datetime as _dt, timezone as _tz
        db = db_session_factory()
        try:
            db.query(Job).filter(Job.id == job_id).update(
                {
                    "status":           "running",
                    "started_at":       _dt.now(_tz.utc),
                    "cancel_requested": False,
                },
                synchronize_session=False,
            )
            db.commit()
        finally:
            db.close()
    except Exception as _db_exc:
        print(
            f"[runner.v2] Job.status='running' DB write failed for "
            f"job_id={job_id} (non-fatal; Inngest worker will keep "
            f"writing status as it progresses): {_db_exc}"
        )

    # Build the event payload (D-10.13 / Step 10 event contract).
    event_data = {
        "job_id":       int(job_id),
        "video_path":   str(video_path),
        "language":     language or "te",
        "platform":     platform,
        "frame_layout": frame or "torn_card",
        "stt_provider": stt_provider or "",
        # preset is caller-supplied via Stage 4; we ship the PLATFORMS
        # entry shape from main.py for the V2 platform. Looking it up
        # here avoids pulling main into runner's import graph (would
        # be circular). Stage 4's defaults handle missing fields.
        "preset": {
            "label": "Full Video + Shorts (V2 Beta)",
            "width":  1080, "height": 1920,
            "min_dur": 15, "max_dur": 60, "ideal_dur": 45,
            "vertical": True,
        },
    }

    client = get_client()
    client.send_sync(events=Event(
        name="video/v2/uploaded",
        data=event_data,
        id=f"job-{job_id}",   # D-10.10: idempotency key
    ))
    print(
        f"[runner.v2] Inngest event sent: name=video/v2/uploaded "
        f"job_id={job_id} stt_provider={stt_provider!r}"
    )


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
                 stt_provider: str = ""):
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
    if platform == "full_video_shorts_v2":
        _dispatch_v2_inngest_event(
            job_id=job_id,
            video_path=video_path,
            language=language,
            platform=platform,
            frame=frame,
            stt_provider=stt_provider,
            db_session_factory=db_session_factory,
        )
        return   # V2 worker takes over; no subprocess spawn

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
                try:
                    for _meta in captured_meta_paths:
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
                # 2. Pipeline output dir — only when an external backend
                # owns the bytes. ``STORAGE_BACKEND=local`` keeps them.
                _backend = (os.environ.get("STORAGE_BACKEND", "local") or "").strip().lower()
                if _backend != "local":
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
                    print(f"[runner] cleanup: STORAGE_BACKEND=local — keeping output dir on disk")
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

        with _PIPELINE_SEMAPHORE:
            _run()

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
        if not meta_path or not meta_path.exists():
            for p in OUTPUT_ROOT.rglob("editor_meta.json"):
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
