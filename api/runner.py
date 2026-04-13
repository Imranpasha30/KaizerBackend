"""
Pipeline runner — launches 11_api_pipeline.py as a subprocess,
captures stdout line-by-line, updates Job model in real-time.
"""
import os
import sys
import glob
import json
import time
import threading
import subprocess
from django.conf import settings

# in-memory per-job log cache for fast SSE-style polling
_live_logs: dict[str, list[str]] = {}
_lock = threading.Lock()


def append_log(job_id: str, line: str):
    with _lock:
        _live_logs.setdefault(str(job_id), []).append(line)
    # persist to DB (non-critical failure OK)
    try:
        from api.models import Job
        Job.objects.filter(id=job_id).update(
            log="\n".join(_live_logs.get(str(job_id), []))
        )
    except Exception:
        pass


def get_live_log(job_id: str) -> list[str]:
    with _lock:
        return list(_live_logs.get(str(job_id), []))


def clear_live_log(job_id: str):
    with _lock:
        _live_logs.pop(str(job_id), None)


def _find_meta(platform: str, after_time: float) -> str | None:
    """Find the editor_meta.json created after `after_time`."""
    output_root = settings.PIPELINE_OUTPUT_ROOT
    # Try platform-specific first
    for pattern in [
        os.path.join(output_root, platform, "*", "editor_meta.json"),
        os.path.join(output_root, "**", "editor_meta.json"),
    ]:
        files = [
            f for f in glob.glob(pattern, recursive=True)
            if os.path.getctime(f) >= after_time - 10
        ]
        if files:
            return max(files, key=os.path.getctime)
    return None


def _estimate_progress(line: str) -> int | None:
    """Rough progress estimate from log line."""
    markers = {
        "Gemini":   10, "Gemini":    15,
        "[1/":      10, "[2/":       30,
        "[3/":      50, "[4/":       65,
        "[5/":      80, "[6/":       95,
        "Editor":   98, "done":     100,
    }
    for key, pct in markers.items():
        if key.lower() in line.lower():
            return pct
    return None


def _run(job_id: str, video_path: str, platform: str, frame_layout: str):
    from api.models import Job, Clip

    job_id = str(job_id)
    start_time = time.time()

    try:
        Job.objects.filter(id=job_id).update(status="running", progress_pct=5)
        append_log(job_id, "▶ Pipeline started")

        python_exe = sys.executable
        script = settings.PIPELINE_SCRIPT

        cmd = [python_exe, script, video_path,
               "--platform", platform, "--frame", frame_layout]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip()
            if not line:
                continue
            append_log(job_id, line)
            pct = _estimate_progress(line)
            if pct:
                Job.objects.filter(id=job_id).update(progress_pct=pct)

        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"Pipeline exited with code {proc.returncode}")

        # ── Find output meta ──────────────────────────────────
        meta_path = _find_meta(platform, start_time)
        if not meta_path or not os.path.exists(meta_path):
            raise RuntimeError("No editor_meta.json found after pipeline")

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        job = Job.objects.get(id=job_id)
        job.meta_path  = meta_path
        job.output_dir = os.path.dirname(meta_path)
        job.status     = "done"
        job.progress_pct = 100
        job.save()

        # ── Create Clip records ───────────────────────────────
        Clip.objects.filter(job=job).delete()
        for i, c in enumerate(meta.get("clips", [])):
            Clip.objects.create(
                job          = job,
                index        = i,
                clip_path    = c.get("clip_path", ""),
                raw_path     = c.get("raw_path", ""),
                thumb_path   = c.get("thumb_path", ""),
                image_path   = c.get("image_path", ""),
                text         = c.get("text", ""),
                frame_type   = c.get("frame_type", "torn_card"),
                card_params  = c.get("card_params", {}),
                section_pct  = c.get("section_pct", {}),
                follow_params= c.get("follow_params", {}),
                split_params = c.get("split_params", {}),
                preset       = meta.get("preset", {}),
            )

        append_log(job_id, f"✓ Done — {len(meta.get('clips', []))} clips ready")

    except Exception as exc:
        import traceback
        err = traceback.format_exc()
        append_log(job_id, f"✗ ERROR: {exc}")
        Job.objects.filter(id=job_id).update(
            status="failed", error=str(exc),
            log="\n".join(get_live_log(job_id))
        )


def start_pipeline(job_id, video_path, platform, frame_layout):
    """Spawn background thread to run pipeline."""
    t = threading.Thread(
        target=_run,
        args=(str(job_id), video_path, platform, frame_layout),
        daemon=True,
        name=f"pipeline-{job_id}",
    )
    t.start()
    return t
