"""V3 orchestrator — V1 pipeline driven by Claude analysis.

Architecture (user-specified 2026-05-23):
  V3 = V1 + (Deepgram + Claude) replacing Gemini for the analysis step.
  EVERYTHING about V1's render path stays: sidebar carousel, lower-third,
  image card, ticker, channel bug, bulletin_stitcher (concat-demuxer,
  drift-free).

Steps:
  1. Extract a lightweight mp3 from the source video (mono 128k, just
     for Deepgram).
  2. Run Deepgram nova-3 multilingual word-level STT.
  3. Run Claude Sonnet 4.6 with the V1 compound-analysis prompt;
     Claude produces the EXACT same JSON shape V1's Gemini would.
  4. Save the analysis JSON to disk.
  5. Set ``KAIZER_REUSE_ANALYSIS_FROM=<analysis.json>`` env var.
  6. Spawn V1's ``pipeline_core/pipeline.py`` subprocess in
     ``--compound`` mode. V1 reads the env var, SKIPS Gemini, uses our
     JSON, and renders the full bulletin + shorts with all overlays.
  7. V1 emits ``[kaizer:meta] <path>`` to stdout when editor_meta.json
     is ready. We forward that to OUR stdout so the parent runner can
     parse it and call _import_clips.
  8. Update Job.status in DB at each milestone so the UI's live log
     shows progress.

CLI:
    python -m pipeline_v3.pipeline_v3.orchestrator \\
        --job-id N --source-video PATH --output-dir DIR \\
        --language te --frame torn_card

The output-dir parameter is informational; V1 picks its own output
location and our parent (runner.py V3 branch) reads job.output_dir
from the DB after V1 finishes (V1 sets it via the [kaizer:meta]
marker pickup in runner._import_clips).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("pipeline_v3.orchestrator")

# __file__ = .../KaizerBackend/pipeline_v3/pipeline_v3/orchestrator.py
# parent.parent.parent = KaizerBackend (root where database.py + pipeline_core live)
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_BACKEND_ROOT))

V1_PIPELINE_SCRIPT = _BACKEND_ROOT / "pipeline_core" / "pipeline.py"
assert V1_PIPELINE_SCRIPT.exists(), f"V1 pipeline.py missing at {V1_PIPELINE_SCRIPT}"


def _setup_logging(job_id: int, output_dir: str) -> None:
    log_path = os.path.join(output_dir, "v3_pipeline.log")
    os.makedirs(output_dir, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == fh.baseFilename for h in root.handlers):
        root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    if not any(isinstance(h, logging.StreamHandler) and h.stream is sys.stdout for h in root.handlers):
        root.addHandler(sh)


def update_job_status(
    job_id: int,
    status: Optional[str] = None,
    output_dir: Optional[str] = None,
    log_append: Optional[str] = None,
) -> None:
    """Best-effort Job row update. Safe to call from subprocess context."""
    try:
        from database import SessionLocal
        import models
        db = SessionLocal()
        try:
            job = db.query(models.Job).filter(models.Job.id == job_id).first()
            if job is None:
                return
            if status:
                job.status = status
                if status == "running" and not job.started_at:
                    job.started_at = datetime.now(timezone.utc)
                if status in ("done", "failed", "cancelled"):
                    job.finished_at = datetime.now(timezone.utc)
            if output_dir:
                job.output_dir = output_dir
            if log_append:
                job.log = (job.log or "") + log_append
            db.commit()
        finally:
            db.close()
    except Exception as exc:
        logger.warning("update_job_status failed: %s", exc)


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def extract_audio_mp3(source_video_path: str, output_dir: str) -> str:
    """Lightweight audio extract for Deepgram (mono 128k mp3).

    Idempotent. V1 will extract its OWN audio later for its own use;
    this one is just so Deepgram can listen.
    """
    mp3 = os.path.join(output_dir, "v3_source.mp3")
    if os.path.exists(mp3) and os.path.getsize(mp3) > 100_000:
        logger.info("audio_extract: cached at %s", mp3)
        return mp3
    cmd = [
        "ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
        "-i", source_video_path,
        "-vn", "-c:a", "libmp3lame", "-ar", "48000", "-ac", "1", "-b:a", "128k",
        mp3,
    ]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"audio extract failed: {r.stderr[-500:]}")
    logger.info("audio_extract: %s done in %.1fs", mp3, time.time() - t0)
    return mp3


def run_v1_subprocess(
    *,
    job_id: int,
    source_video_path: str,
    analysis_json_path: str,
    language: str,
    frame: str,
    output_dir: str,
) -> int:
    """Spawn V1's pipeline.py in --compound mode with KAIZER_REUSE_ANALYSIS_FROM
    set so V1 skips Gemini and uses our Claude-produced analysis.

    Forwards V1's stdout to our stdout so the parent runner's marker-
    streaming loop (if any) can pick up [kaizer:meta] / [kaizer:analysis]
    lines. Returns the V1 subprocess return code.
    """
    env = os.environ.copy()
    # V1's reuse-analysis hook does ``os.path.isfile(_reuse_path)``. Pass
    # an ABSOLUTE path so V1's check passes regardless of subprocess cwd.
    env["KAIZER_REUSE_ANALYSIS_FROM"] = os.path.abspath(analysis_json_path)
    # CRITICAL: PYTHONUNBUFFERED=1 forces stdout/stderr unbuffered in V1.
    # Without this, V1's print() output sits in a Windows pipe buffer
    # and never reaches our reader -- V1 LOOKS hung for hours but is
    # actually waiting on a flush. Track 3A research documented this
    # Windows-subprocess-buffering issue weeks ago (Job 53 timeout).
    # The -u flag on the venv launcher does NOT propagate through to
    # the actual python interpreter; PYTHONUNBUFSERED env does.
    env["PYTHONUNBUFFERED"] = "1"
    # V1 reads OUTPUT_ROOT from env or its CLI; we let it default to the
    # same env-controlled OUTPUT_ROOT runner.py uses. V1 picks
    # OUTPUT_ROOT/<platform>/job_<id>/ internally.
    cmd = [
        sys.executable, "-u", str(V1_PIPELINE_SCRIPT),
        source_video_path,
        "--compound",
        "--frame", frame,
        "--language", language,
    ]
    logger.info("v1 subprocess: %s", " ".join(cmd))
    logger.info("v1 subprocess env: KAIZER_REUSE_ANALYSIS_FROM=%s", env["KAIZER_REUSE_ANALYSIS_FROM"])

    # File-based output capture (avoids Windows pipe-drain deadlock).
    # V1 writes stdout+stderr directly to a file with line buffering.
    # We poll the file in a background thread for [kaizer:meta] markers
    # and milestone updates so the UI sees progress in near-realtime.
    v1_log_path = os.path.join(output_dir, "v1_subprocess.log")
    # buffering=1 = line-buffered text mode -- V1's print() with
    # PYTHONUNBUFFERED=1 will flush each line immediately.
    v1_log_file = open(v1_log_path, "w", encoding="utf-8", buffering=1)
    captured_meta_paths: list[str] = []
    seen_lines: set[int] = set()

    import threading
    stop_event = threading.Event()

    def _tail_v1_log():
        """Poll v1_log every 1.5s, parse for [kaizer:meta] / Step lines."""
        last_size = 0
        while not stop_event.is_set():
            try:
                if not os.path.exists(v1_log_path):
                    stop_event.wait(1.5); continue
                cur_size = os.path.getsize(v1_log_path)
                if cur_size > last_size:
                    with open(v1_log_path, "r", encoding="utf-8", errors="replace") as rf:
                        rf.seek(last_size)
                        new_data = rf.read()
                    last_size = cur_size
                    for ln in new_data.split("\n"):
                        ln = ln.rstrip()
                        if not ln:
                            continue
                        h = hash(ln)
                        if h in seen_lines:
                            continue
                        seen_lines.add(h)
                        if ln.startswith("[kaizer:meta]"):
                            _mp = ln[len("[kaizer:meta] "):].strip()
                            if _mp and not os.path.isabs(_mp):
                                _mp = str((_BACKEND_ROOT / _mp).resolve())
                            if _mp and _mp not in captured_meta_paths:
                                captured_meta_paths.append(_mp)
                            update_job_status(
                                job_id, log_append=f"\n[{_ts()}] V1: {ln}",
                            )
                        elif "Cutting clip" in ln:
                            update_job_status(
                                job_id, log_append=f"\n[{_ts()}] V1: {ln.strip()}",
                            )
                        elif "PASS 1/2" in ln or "PASS 2/2" in ln:
                            update_job_status(
                                job_id, log_append=f"\n[{_ts()}] V1: {ln.strip()}",
                            )
                        elif ln.lstrip().startswith("[") and "/6]" in ln:
                            # V1's "[N/6]" step markers
                            update_job_status(
                                job_id, log_append=f"\n[{_ts()}] V1 {ln.strip()}",
                            )
            except Exception as exc:
                logger.warning("v1 log tail error: %s", exc)
            stop_event.wait(1.5)

    tailer = threading.Thread(target=_tail_v1_log, daemon=True)
    tailer.start()

    proc = subprocess.Popen(
        cmd, env=env, cwd=str(_BACKEND_ROOT),
        stdout=v1_log_file, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    try:
        rc = proc.wait()
    finally:
        # Stop tail + final scan for markers we may have missed (race)
        stop_event.set()
        tailer.join(timeout=5)
        v1_log_file.close()
        # Final pass over the file (catches the last few lines incl. [kaizer:meta]
        # that arrived between last poll and exit).
        try:
            with open(v1_log_path, "r", encoding="utf-8", errors="replace") as rf:
                for ln in rf:
                    ln = ln.rstrip()
                    if ln.startswith("[kaizer:meta]"):
                        _mp = ln[len("[kaizer:meta] "):].strip()
                        if _mp and not os.path.isabs(_mp):
                            _mp = str((_BACKEND_ROOT / _mp).resolve())
                        if _mp and _mp not in captured_meta_paths:
                            captured_meta_paths.append(_mp)
        except Exception:
            pass
    return rc, captured_meta_paths


def run_pipeline(
    job_id: int,
    source_video_path: str,
    output_dir: str,
    language: str = "te",
    frame: str = "torn_card",
    stage_2_provider: str = "claude",
) -> dict:
    """V3 = Deepgram + {Claude|Gemini} -> V1 analysis JSON -> spawn V1 pipeline.py."""
    _setup_logging(job_id, output_dir)
    update_job_status(job_id, "running", output_dir=output_dir,
                      log_append=f"\n[{_ts()}] V3 pipeline started (Deepgram + {stage_2_provider} -> V1 render)")

    def _log_ui(msg: str) -> None:
        update_job_status(job_id, log_append=f"\n[{_ts()}] {msg}")

    try:
        logger.info("v3 pipeline: job_id=%d source=%s", job_id, source_video_path)
        _log_ui(f"source: {os.path.basename(source_video_path)}")

        # Stage 1: extract audio
        _log_ui("Stage 1/4: Extracting audio (mp3 128k mono) for Deepgram...")
        t0 = time.time()
        audio_mp3 = extract_audio_mp3(source_video_path, output_dir)
        _log_ui(f"Stage 1/4 DONE in {time.time()-t0:.1f}s  -> v3_source.mp3")

        # Stage 2: Deepgram + (Claude|Gemini) analysis
        from pipeline_v3.pipeline_v3.word_editor import produce_v1_analysis
        provider_label = "Claude sonnet-4-6" if stage_2_provider == "claude" else "Gemini 2.5 Pro"
        _log_ui(f"Stage 2/4: Deepgram nova-3 word-level STT + {provider_label} compound analysis...")
        t1 = time.time()
        preset = {"width": 1080, "height": 1920, "min_dur": 30, "max_clips": 6}
        result = produce_v1_analysis(
            audio_mp3_path=audio_mp3,
            language=language,
            preset=preset,
            provider=stage_2_provider,
        )
        _log_ui(
            f"Stage 2/4 DONE in {time.time()-t1:.1f}s  -> "
            f"clips={len(result.analysis.get('clips', []))} "
            f"full_video_cuts={len(result.analysis.get('full_video_cuts', []))} "
            f"shorts_cuts={len(result.analysis.get('shorts_cuts', []))} "
            f"image_plan={len(result.analysis.get('image_plan', []))} "
            f"skipped_segments={len(result.analysis.get('skipped_segments', []))} "
            f"(${result.llm_cost_usd:.4f})"
        )

        # Stage 3: save analysis.json
        analysis_path = os.path.join(output_dir, "v3_analysis.json")
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(result.analysis, f, ensure_ascii=False, indent=2)
        logger.info("v3 analysis saved: %s", analysis_path)
        _log_ui(f"Stage 3/4: v3_analysis.json saved -> {os.path.basename(analysis_path)}")

        # Stage 4: invoke V1's pipeline.py with our analysis pre-loaded
        _log_ui("Stage 4/4: Invoking V1 pipeline.py (--compound) with KAIZER_REUSE_ANALYSIS_FROM=v3_analysis.json")
        t2 = time.time()
        rc, captured_meta_paths = run_v1_subprocess(
            job_id=job_id,
            source_video_path=source_video_path,
            analysis_json_path=analysis_path,
            language=language,
            frame=frame,
            output_dir=output_dir,
        )
        if rc != 0:
            raise RuntimeError(f"V1 pipeline.py exited rc={rc}; see stdout above")

        # V1 has done everything: cut, compose with full overlays, stitch,
        # emitted editor_meta.json, printed [kaizer:meta] markers. Our parent
        # runner.py redirected our stdout to a log file, so it can't parse
        # those markers itself -- we call _import_clips here directly using
        # the meta paths we captured while streaming V1's stdout.
        _log_ui(f"Stage 4/4 DONE in {time.time()-t2:.1f}s  -> V1 finished cleanly (captured {len(captured_meta_paths)} meta path(s))")

        # Import clips from each meta path we captured. _import_clips also
        # sets job.output_dir to the meta's parent directory.
        if captured_meta_paths:
            try:
                import runner as _runner  # type: ignore
                from database import SessionLocal as _SessionLocal
                import models as _models
                _db = _SessionLocal()
                try:
                    _job = _db.query(_models.Job).filter(_models.Job.id == job_id).first()
                    n_imported = 0
                    for _mp in captured_meta_paths:
                        if not os.path.exists(_mp):
                            logger.warning("captured meta path does not exist: %s", _mp)
                            continue
                        try:
                            _runner._import_clips(_job, _db, meta_override=Path(_mp))
                            n_imported = len(_job.clips)
                            logger.info("_import_clips OK from %s -> %d clips", _mp, n_imported)
                        except Exception as _ie:
                            logger.error("_import_clips failed for %s: %s", _mp, _ie)
                    _log_ui(f"Clip import: {n_imported} clips imported")
                finally:
                    _db.close()
            except Exception as _ic_exc:
                logger.error("_import_clips wrapper failed: %s", _ic_exc)
                _log_ui(f"WARNING: clip import failed -- {_ic_exc}")
        else:
            _log_ui("WARNING: V1 emitted no [kaizer:meta] markers; clips will not appear in editor")

        update_job_status(
            job_id, "done",
            log_append=f"\n[{_ts()}] V3 DONE (total {time.time()-t0:.1f}s)",
        )

        return {
            "ok": True,
            "job_id": job_id,
            "analysis_path": analysis_path,
            "n_words_in": result.n_words_in,
            "audio_duration_sec": result.audio_duration_sec,
            "llm_cost_usd": result.llm_cost_usd,
            "llm_wall_sec": result.llm_wall_sec,
            "v1_rc": rc,
        }

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("v3 pipeline FAILED: %s\n%s", exc, tb)
        update_job_status(
            job_id, "failed",
            log_append=f"\n[{_ts()}] FAILED: {exc}\n{tb[-800:]}",
        )
        return {"ok": False, "job_id": job_id, "error": str(exc), "traceback": tb}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job-id", type=int, required=True)
    p.add_argument("--source-video", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--language", default="te")
    p.add_argument("--frame", default="torn_card")
    p.add_argument("--stage-2-provider", default="claude", choices=["claude", "gemini"],
                   help="LLM for the editorial analysis step. claude=Sonnet 4.6 T=0; gemini=2.5 Pro T=0.2.")
    args = p.parse_args()

    result = run_pipeline(
        job_id=args.job_id,
        source_video_path=args.source_video,
        output_dir=args.output_dir,
        language=args.language,
        frame=args.frame,
        stage_2_provider=args.stage_2_provider,
    )
    print(json.dumps(result, indent=2, default=str), flush=True)
    sys.exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
