"""STOP ALL UPLOADING — clean slate for a fresh test.

Cancels every non-terminal upload job (queued / parked_quota) using the
SAME semantics as the app's cancel endpoint (status flip + idempotent
refund + parent failed_count bump), then marks any orphan / non-terminal
PublishTask (no active jobs left, e.g. the 'fanning_out' phantoms) as
cancelled. Terminal rows (completed / already-cancelled) are left alone.

Run from KaizerBackend/ with the venv python:
  python scripts/stop_all_uploads.py            # DRY RUN (prints plan)
  python scripts/stop_all_uploads.py --apply    # actually stop
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
import models
from services import credits as credits_svc

APPLY = "--apply" in sys.argv

CANCEL_JOB_SQL = (
    "UPDATE upload_jobs_v2 "
    "SET status='cancelled', claimed_by=NULL, lease_expires_at=NULL, "
    "    finished_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP "
    "WHERE id = :jid AND status IN ('queued','parked_quota')"
)
BUMP_FAILED_SQL = """
UPDATE publish_tasks
SET failed_count = failed_count + 1,
    status = CASE
        WHEN completed_count + (failed_count + 1) >= target_count THEN
            CASE WHEN (failed_count + 1) = 0 THEN 'completed'
                 WHEN completed_count = 0 THEN 'failed'
                 ELSE 'partial_failed' END
        ELSE status
    END,
    updated_at = CURRENT_TIMESTAMP
WHERE id = :tid
"""

# PublishTask states that are NOT terminal — if one has no active jobs
# left, it's an orphan we cancel.
NON_TERMINAL_TASK = ("fanning_out", "dispatched", "queued", "running", "active")
# Job states that mean "still alive / could fire".
ACTIVE_JOB = ("queued", "parked_quota", "claimed", "branding", "uploading", "running")


def _refund_exists(db, user_id, job_id) -> bool:
    return (
        db.query(models.CreditLedger)
        .filter(
            models.CreditLedger.user_id == user_id,
            models.CreditLedger.upload_job_id == job_id,
            models.CreditLedger.reason == "refund",
        )
        .first()
        is not None
    )


def main() -> int:
    db = SessionLocal()
    try:
        print(f"{'APPLY' if APPLY else 'DRY-RUN'} mode\n")

        # ---- 1. Non-terminal jobs (queued / parked_quota) ----
        jobs = db.execute(text(
            "SELECT id, publish_task_id, user_id, status, "
            "       predicted_credit_cost, upload_path "
            "FROM upload_jobs_v2 WHERE status IN ('queued','parked_quota') "
            "ORDER BY id")).fetchall()
        print(f"Jobs to cancel (queued/parked_quota): {len(jobs)}")
        cancelled_jobs = 0
        refunded = 0
        for j in jobs:
            jid, tid, uid, st, cost, path = j
            cost = int(cost or 0)
            print(f"  job {jid} task {tid} status={st!r} path={path!r} cost={cost}")
            if not APPLY:
                continue
            res = db.execute(text(CANCEL_JOB_SQL), {"jid": int(jid)})
            if not res.rowcount:
                continue
            cancelled_jobs += 1
            if cost > 0 and uid is not None and not _refund_exists(db, int(uid), int(jid)):
                credits_svc.refund(db, user_id=int(uid), cost=cost, upload_job_id=int(jid))
                refunded += 1
            if tid is not None:
                db.execute(text(BUMP_FAILED_SQL), {"tid": int(tid)})
            db.commit()

        # ---- 2. Orphan / non-terminal PublishTasks with no active jobs ----
        tasks = db.execute(text(
            "SELECT id, status, target_count, completed_count, failed_count "
            "FROM publish_tasks WHERE status IN :ns"
        ).bindparams(__import__("sqlalchemy").bindparam("ns", expanding=True)),
            {"ns": list(NON_TERMINAL_TASK)}).fetchall()
        orphans = []
        for t in tasks:
            tid = t[0]
            active = db.execute(text(
                "SELECT count(*) FROM upload_jobs_v2 "
                "WHERE publish_task_id = :tid AND status IN :a"
            ).bindparams(__import__("sqlalchemy").bindparam("a", expanding=True)),
                {"tid": int(tid), "a": list(ACTIVE_JOB)}).scalar()
            if not active:
                orphans.append((tid, t[1]))
        print(f"\nOrphan/non-terminal tasks with NO active jobs: {len(orphans)}")
        for tid, st in orphans:
            print(f"  task {tid} status={st!r} -> cancelled")
            if APPLY:
                db.execute(text(
                    "UPDATE publish_tasks SET status='cancelled', "
                    "updated_at=CURRENT_TIMESTAMP WHERE id=:tid"), {"tid": int(tid)})
        if APPLY:
            db.commit()

        # ---- 3. After snapshot ----
        print("\n=== AFTER: job status counts ===")
        for r in db.execute(text(
            "SELECT status, count(*) FROM upload_jobs_v2 GROUP BY status "
            "ORDER BY 2 DESC")).fetchall():
            print(f"  {r[0]:<14} {r[1]}")
        print("=== AFTER: task status counts ===")
        for r in db.execute(text(
            "SELECT status, count(*) FROM publish_tasks GROUP BY status "
            "ORDER BY 2 DESC")).fetchall():
            print(f"  {r[0]:<14} {r[1]}")

        if APPLY:
            print(f"\nDONE — cancelled {cancelled_jobs} jobs "
                  f"({refunded} refunded), {len(orphans)} orphan tasks closed.")
        else:
            print("\n(DRY RUN — re-run with --apply to actually stop.)")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
