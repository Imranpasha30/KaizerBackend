"""Measure isolated status-endpoint latency + response size to see
whether the 2s micro-cache is doing its job (cache-hit should be ms)."""
import statistics
import sys
import time

import httpx

sys.path.insert(0, ".")
import warnings

warnings.filterwarnings("ignore")
import auth  # noqa: E402
import models  # noqa: E402
from database import SessionLocal  # noqa: E402

db = SessionLocal()
u = db.query(models.User).filter(models.User.is_active == True).order_by(models.User.id).first()  # noqa: E712
tok = auth.issue_token(u)
db.close()

H = {"Authorization": f"Bearer {tok}"}
BASE = "http://localhost:8000"

with httpx.Client() as c:
    jobs = c.get(f"{BASE}/api/jobs/", params={"limit": 1}, headers=H).json()
    jid = jobs[0]["id"] if isinstance(jobs, list) else jobs["jobs"][0]["id"]

    # Full-log request (since=0)
    t0 = time.perf_counter()
    r = c.get(f"{BASE}/api/jobs/{jid}/status/", headers=H)
    full_ms = (time.perf_counter() - t0) * 1000
    body = r.json()
    off = body.get("log_offset", len(body.get("log_lines", [])))
    print(f"job={jid} full status: {full_ms:.1f}ms, "
          f"{len(r.content)} bytes, log_lines={len(body.get('log_lines', []))}, "
          f"log_offset={off}")

    # 20 incremental polls (since=offset) — should be cache hits + tiny
    lat = []
    for _ in range(20):
        t0 = time.perf_counter()
        r = c.get(f"{BASE}/api/jobs/{jid}/status/",
                  params={"since": off}, headers=H)
        lat.append((time.perf_counter() - t0) * 1000)
    print(f"incremental polls: p50={statistics.median(lat):.1f}ms "
          f"min={min(lat):.1f}ms max={max(lat):.1f}ms "
          f"size={len(r.content)} bytes")

    # 20 jobs-list calls
    lat2 = []
    for _ in range(20):
        t0 = time.perf_counter()
        r = c.get(f"{BASE}/api/jobs/", params={"limit": 20}, headers=H)
        lat2.append((time.perf_counter() - t0) * 1000)
    print(f"jobs list (limit=20): p50={statistics.median(lat2):.1f}ms "
          f"min={min(lat2):.1f}ms max={max(lat2):.1f}ms "
          f"size={len(r.content)} bytes")
