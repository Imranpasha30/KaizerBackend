"""Monitor job 70 + record peak RAM usage. Exits on done/failed/cancelled."""
import sys, os, time, subprocess, json
from pathlib import Path
HERE = Path(__file__).parent
os.chdir(HERE)
sys.path.insert(0, str(HERE))
from dotenv import load_dotenv
load_dotenv()
from database import SessionLocal
import models

JOB_ID = 70
TIMEOUT_SEC = 60 * 30
t_start = time.time()
last_status = None
peak_used_gb = 0.0
peak_used_t = None
samples = []
PS_MEM = "powershell.exe -NoProfile -Command \"$o=Get-CimInstance Win32_OperatingSystem; '{0:0.00}|{1:0.00}' -f ($o.TotalVisibleMemorySize/1MB), ((($o.TotalVisibleMemorySize-$o.FreePhysicalMemory)/1MB))\""

while True:
    db = SessionLocal()
    try:
        job = db.query(models.Job).filter(models.Job.id == JOB_ID).first()
    finally:
        db.close()
    elapsed = time.time() - t_start
    try:
        out = subprocess.check_output(PS_MEM, shell=True, text=True, timeout=5).strip()
        total_gb, used_gb = (float(x) for x in out.split("|"))
        if used_gb > peak_used_gb:
            peak_used_gb = used_gb
            peak_used_t = elapsed
    except Exception:
        used_gb = -1
    if job.status != last_status:
        print(f"[{time.strftime('%H:%M:%S')} | +{elapsed:5.0f}s | RAM_now={used_gb:.1f}G | RAM_peak={peak_used_gb:.1f}G@{peak_used_t}s] STATUS: {last_status} -> {job.status}", flush=True)
        last_status = job.status
    else:
        # log line every minute even with no status change
        if int(elapsed) % 60 < 21:
            print(f"[{time.strftime('%H:%M:%S')} | +{elapsed:5.0f}s | RAM_now={used_gb:.1f}G | RAM_peak={peak_used_gb:.1f}G] status={job.status}", flush=True)
    if job.status in ("done", "failed", "cancelled"):
        total = time.time() - t_start
        print(f"\n=== JOB 70 FINAL ===", flush=True)
        print(f"status: {job.status}", flush=True)
        print(f"total wall: {total:.1f}s ({total/60:.2f} min)", flush=True)
        print(f"peak RAM:   {peak_used_gb:.2f} GB at t+{peak_used_t}s", flush=True)
        print(f"output_dir: {job.output_dir!r}", flush=True)
        if job.log:
            print(f"\n--- last 2500 chars of job.log ---", flush=True)
            print(job.log[-2500:], flush=True)
        break
    if elapsed > TIMEOUT_SEC:
        print(f"\n=== TIMEOUT at {elapsed:.0f}s ===", flush=True)
        print(f"last status: {job.status}", flush=True)
        print(f"peak RAM:    {peak_used_gb:.2f} GB at t+{peak_used_t}s", flush=True)
        break
    time.sleep(20)
