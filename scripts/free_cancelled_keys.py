"""Free idempotency keys held by CANCELLED jobs so a genuine re-publish
works, and close orphan 'fanning_out' tasks (0 jobs) left by the phantom
bug. Completed jobs are LEFT INTACT (a successful publish should still
dedupe).

FKs verified: publish_attempts → CASCADE (auto-deleted), credit_ledger →
SET NULL, quota_burn_log → SET NULL. So deleting a cancelled job is safe
and non-destructive to the audit ledger.

  python scripts/free_cancelled_keys.py            # DRY RUN
  python scripts/free_cancelled_keys.py --apply
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from sqlalchemy import text
from database import SessionLocal

APPLY = "--apply" in sys.argv

ORPHAN_TASK_STATES = ("fanning_out", "dispatched", "queued", "running", "active")
ACTIVE_JOB = ("queued", "parked_quota", "claimed", "branding", "uploading", "running")


def main() -> int:
    db = SessionLocal()
    try:
        print(f"{'APPLY' if APPLY else 'DRY-RUN'} mode\n")

        cancelled = db.execute(text(
            "SELECT id, publish_task_id, idempotency_key, status "
            "FROM upload_jobs_v2 WHERE status='cancelled' ORDER BY id")).fetchall()
        print(f"Cancelled jobs holding keys: {len(cancelled)}")
        for j in cancelled:
            print(f"  job={j[0]} task={j[1]} key={j[2][:16]}…")

        if APPLY and cancelled:
            ids = [int(j[0]) for j in cancelled]
            # publish_attempts cascade; ledger/burn SET NULL — so a plain
            # delete frees both upload_jobs_v2 AND publish_attempts keys.
            db.execute(text("DELETE FROM upload_jobs_v2 WHERE id = ANY(:ids)"),
                       {"ids": ids})
            db.commit()
            print(f"\nDeleted {len(ids)} cancelled jobs (keys freed; "
                  f"stale publish_attempts cascade-deleted).")

        # Close orphan tasks (no active jobs) — the phantom fanning_out rows.
        tasks = db.execute(text(
            "SELECT id, status FROM publish_tasks WHERE status = ANY(:ns)"),
            {"ns": list(ORPHAN_TASK_STATES)}).fetchall()
        orphans = []
        for t in tasks:
            active = db.execute(text(
                "SELECT count(*) FROM upload_jobs_v2 "
                "WHERE publish_task_id=:t AND status = ANY(:a)"),
                {"t": int(t[0]), "a": list(ACTIVE_JOB)}).scalar()
            if not active:
                orphans.append(t[0])
        print(f"\nOrphan tasks to close: {len(orphans)} -> {orphans}")
        if APPLY and orphans:
            db.execute(text(
                "UPDATE publish_tasks SET status='cancelled', "
                "updated_at=CURRENT_TIMESTAMP WHERE id = ANY(:ids)"),
                {"ids": [int(x) for x in orphans]})
            db.commit()

        # after
        print("\n=== AFTER: job statuses ===")
        for r in db.execute(text(
            "SELECT status,count(*) FROM upload_jobs_v2 GROUP BY status "
            "ORDER BY 2 DESC")).fetchall():
            print(f"  {r[0]:<12} {r[1]}")
        print("=== AFTER: task statuses ===")
        for r in db.execute(text(
            "SELECT status,count(*) FROM publish_tasks GROUP BY status "
            "ORDER BY 2 DESC")).fetchall():
            print(f"  {r[0]:<14} {r[1]}")
        if not APPLY:
            print("\n(DRY RUN — re-run with --apply.)")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
