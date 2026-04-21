import os
import sys
import json
import threading
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR        = Path(__file__).parent
PIPELINE_SCRIPT = BASE_DIR / "pipeline_core" / "pipeline.py"
# Use /tmp so clips survive the session but don't fill the deploy image
OUTPUT_ROOT     = Path(os.getenv("KAIZER_OUTPUT_ROOT", "/tmp/kaizer_output"))


def run_pipeline(job_id: int, video_path: str, platform: str, frame: str,
                 db_session_factory, language: str = "te",
                 default_image: str = ""):
    """Launch pipeline as subprocess, stream stdout into Job.log.

    When `default_image` is a non-empty absolute path to an existing file,
    the pipeline uses it for every clip instead of fetching stock photos.
    """

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

            # pipeline.py uses positional 'video', --platform, --frame, --language.
            # -u forces unbuffered stdout so log lines stream through the pipe
            # in real time instead of sitting in Python's 8KB block buffer.
            cmd = [
                sys.executable, "-u", str(PIPELINE_SCRIPT),
                video_path,
                "--platform", platform,
                "--frame", frame,
                "--language", language,
            ]
            if default_image:
                cmd += ["--default-image", default_image]

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
                env={
                    **os.environ,
                    "KAIZER_OUTPUT_ROOT": str(OUTPUT_ROOT),
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONIOENCODING": "utf-8",
                },
                cwd=str(BASE_DIR),
            )

            log_lines = []
            captured_meta_path: str = ""
            for line in process.stdout:
                stripped = line.rstrip()
                log_lines.append(stripped)
                # Sniff for the machine-parseable marker line the pipeline emits
                # right after writing editor_meta.json.  Canonical source for
                # import — removes rglob-picks-a-stale-file ambiguity.
                if stripped.startswith("[kaizer:meta] "):
                    captured_meta_path = stripped[len("[kaizer:meta] "):].strip()
                # Flush every line for live log visibility (testing mode).
                # Swap back to `% 5 == 0` for less DB load in production.
                db2 = db_session_factory()
                j = db2.query(Job).filter(Job.id == job_id).first()
                if j:
                    j.log = "\n".join(log_lines)
                    db2.commit()
                db2.close()

            process.wait()

            db2 = db_session_factory()
            j = db2.query(Job).filter(Job.id == job_id).first()
            j.log = "\n".join(log_lines)

            if process.returncode == 0:
                try:
                    meta_override = Path(captured_meta_path) if captured_meta_path else None
                    _import_clips(j, db2, meta_override=meta_override)
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
            else:
                j.status = "failed"
                j.error = "\n".join(log_lines[-20:])   # last 20 lines as error summary

            # Stamp completion wall-clock time on every terminal state.
            from datetime import datetime as _dt, timezone as _tz
            j.finished_at = _dt.now(_tz.utc)
            db2.commit()
            db2.close()

            # Auto-enqueue any campaigns attached to this job (Phase A).
            if process.returncode == 0:
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
            db.close()

    threading.Thread(target=_run, daemon=True).start()


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
        clip = Clip(
            job_id=job.id,
            clip_index=i,
            filename=Path(clip_path).name,
            file_path=clip_path,
            thumb_path=c.get("thumb_path", ""),
            image_path=c.get("image_path", ""),
            duration=float(c.get("duration", 0)),
            frame_type=c.get("frame_type", ""),
            text=c.get("text", ""),
            sentiment=c.get("sentiment", ""),
            entities=json.dumps(c.get("entities", [])),
            card_params=json.dumps(c.get("card_params", {})),
            section_pct=json.dumps(c.get("section_pct", {})),
            follow_params=json.dumps(c.get("follow_params", {})),
            meta=json.dumps(c),
        )
        db.add(clip)
        imported += 1

    job.output_dir = str(meta_path.parent)
    db.commit()
    print(f"[runner] imported {imported} clip(s), skipped {skipped} for job {job.id}")
