import os
import sys
import json
import threading
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
PIPELINE_SCRIPT = BASE_DIR / "pipeline_core" / "pipeline.py"
OUTPUT_ROOT = Path(os.getenv("KAIZER_OUTPUT_ROOT", str(BASE_DIR / "output")))


def run_pipeline(job_id: int, video_path: str, platform: str, frame: str, db_session_factory):
    """Launch pipeline as subprocess, stream stdout into Job.log."""

    def _run():
        from models import Job, Clip

        db = db_session_factory()
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            job.status = "running"
            db.commit()

            cmd = [
                sys.executable, str(PIPELINE_SCRIPT),
                "--video", video_path,
                "--platform", platform,
                "--frame", frame,
                "--output-root", str(OUTPUT_ROOT),
                "--yes",
            ]

            import subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ},
            )

            log_lines = []
            for line in process.stdout:
                log_lines.append(line.rstrip())
                if len(log_lines) % 5 == 0:
                    db2 = db_session_factory()
                    j = db2.query(Job).filter(Job.id == job_id).first()
                    j.log = "\n".join(log_lines)
                    db2.commit()
                    db2.close()

            process.wait()

            db2 = db_session_factory()
            j = db2.query(Job).filter(Job.id == job_id).first()
            j.log = "\n".join(log_lines)

            if process.returncode == 0:
                j.status = "done"
                _import_clips(j, db2, db_session_factory)
            else:
                j.status = "failed"

            db2.commit()
            db2.close()

        except Exception as e:
            db3 = db_session_factory()
            j = db3.query(Job).filter(Job.id == job_id).first()
            if j:
                j.status = "failed"
                j.log = (j.log or "") + f"\n\nError: {e}"
                db3.commit()
            db3.close()
        finally:
            db.close()

    threading.Thread(target=_run, daemon=True).start()


def _import_clips(job, db, db_session_factory):
    """Find editor_meta.json and create Clip records."""
    from models import Clip

    meta_path = None
    search_root = Path(job.output_dir) if job.output_dir else OUTPUT_ROOT
    for p in search_root.rglob("editor_meta.json"):
        meta_path = p
        break

    if not meta_path or not meta_path.exists():
        return

    try:
        clips_data = json.loads(meta_path.read_text()).get("clips", [])
        for i, c in enumerate(clips_data):
            clip = Clip(
                job_id=job.id,
                clip_index=i,
                filename=Path(c.get("clip_path", "")).name,
                file_path=c.get("clip_path", ""),
                duration=float(c.get("duration", 0)),
                sentiment=c.get("sentiment", ""),
                entities=json.dumps(c.get("entities", [])),
                meta=json.dumps(c),
            )
            db.add(clip)
        job.output_dir = str(meta_path.parent)
    except Exception as e:
        print(f"[runner] clip import error: {e}")
