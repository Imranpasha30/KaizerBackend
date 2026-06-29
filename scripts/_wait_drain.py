"""Block until the publish queue drains, then exit (re-invokes the agent).

'Active' = jobs still in a moving state. When that hits 0, the current
publish run is finished (only completed/failed/cancelled/parked remain)
and it's safe to restart for the pipeline rebuild. Caps at ~2h so it can
never hang forever.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from sqlalchemy import text
from database import SessionLocal

ACTIVE = ("queued", "claimed", "branding", "ready_to_upload", "uploading", "running")
POLL_S = 20
MAX_POLLS = 360  # ~2h ceiling

last = None
for i in range(MAX_POLLS):
    db = SessionLocal()
    try:
        rows = db.execute(text(
            "SELECT status, count(*) FROM upload_jobs_v2 WHERE status = ANY(:s) "
            "GROUP BY status"), {"s": list(ACTIVE)}).fetchall()
    finally:
        db.close()
    n = sum(int(r[1]) for r in rows)
    detail = dict((r[0], int(r[1])) for r in rows)
    if n == 0:
        print(f"PUBLISH_DRAINED after ~{i * POLL_S}s — safe to rebuild")
        sys.exit(0)
    if detail != last:
        print(f"[poll {i}] active={n} {detail}", flush=True)
        last = detail
    time.sleep(POLL_S)
print("WATCHER_TIMEOUT after ~2h — active jobs still present; not draining")
sys.exit(0)
