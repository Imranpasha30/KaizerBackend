"""Tail a file and append new lines to Job.log every few seconds.

Used so an out-of-band Stage 3 resume (scripts/resume_stage3.py)
streams progress into the V4 editor UI, which reads from Job.log in
the DB. Without this the UI only sees the original orchestrator
crash log and the user has no signal that the rerun is alive.

Exits when the tailed process ends + one final flush.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from database import SessionLocal
import models


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", type=int, required=True)
    ap.add_argument("--file", required=True, help="file to tail")
    ap.add_argument("--poll-sec", type=float, default=4.0)
    ap.add_argument("--idle-exit-sec", type=float, default=60.0,
                    help="exit after this many seconds with no new bytes")
    args = ap.parse_args()

    src = Path(args.file)
    pos = 0
    last_change = time.time()

    while True:
        if not src.is_file():
            time.sleep(args.poll_sec)
            continue
        try:
            with src.open("rb") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
        except OSError:
            time.sleep(args.poll_sec)
            continue

        if chunk:
            text = chunk.decode("utf-8", errors="replace")
            db = SessionLocal()
            try:
                j = db.query(models.Job).filter(models.Job.id == args.job_id).first()
                if j is not None:
                    j.log = (j.log or "") + text
                    db.commit()
            finally:
                db.close()
            last_change = time.time()

        # Exit when the resume script has been silent long enough that
        # we're confident it terminated. We don't watch the PID because
        # the resume runs in a separate background task harness.
        if time.time() - last_change > args.idle_exit_sec:
            # Flush a closing marker so the UI gets a tidy ending.
            db = SessionLocal()
            try:
                j = db.query(models.Job).filter(models.Job.id == args.job_id).first()
                if j is not None and not (j.log or "").rstrip().endswith("[tailer] done"):
                    j.log = (j.log or "") + "\n[tailer] done\n"
                    db.commit()
            finally:
                db.close()
            return 0

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())
