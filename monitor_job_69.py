"""Poll job 69 until it terminates. Print state changes."""
import sys, os, time
from pathlib import Path
HERE = Path(__file__).parent
os.chdir(HERE)
sys.path.insert(0, str(HERE))
from dotenv import load_dotenv
load_dotenv()
from database import SessionLocal
import models

JOB_ID = 69
TIMEOUT_SEC = 60 * 90  # 90-min ceiling
t_start = time.time()
last_status = None
last_log_tail = ""
while True:
    db = SessionLocal()
    try:
        job = db.query(models.Job).filter(models.Job.id == JOB_ID).first()
    finally:
        db.close()
    if job is None:
        print(f"[{time.strftime('%H:%M:%S')}] job_id={JOB_ID} not found")
        time.sleep(10)
        continue
    elapsed = time.time() - t_start
    if job.status != last_status:
        print(f"[{time.strftime('%H:%M:%S')}] STATUS CHANGE: {last_status} -> {job.status} (elapsed {elapsed:.0f}s)")
        last_status = job.status
    log = (job.log or "")
    if log and log[-200:] != last_log_tail:
        last_log_tail = log[-200:]
    if job.status in ("done", "failed", "cancelled"):
        print(f"[{time.strftime('%H:%M:%S')}] FINAL status={job.status} after {elapsed:.0f}s")
        print(f"output_dir={job.output_dir!r}")
        if job.log:
            print(f"--- last 1500 chars of log ---")
            print(job.log[-1500:])
        break
    if elapsed > TIMEOUT_SEC:
        print(f"[{time.strftime('%H:%M:%S')}] TIMEOUT after {elapsed:.0f}s; status={job.status}")
        break
    time.sleep(20)
