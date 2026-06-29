"""Wave 6 — live-loop E2E proof.

Inserts a REAL queued UploadJobV2 (synthetic fixtures, no credits, no
monkeypatching) into the live dev DB and watches the RUNNING backend's
embedded publish worker claim it: status must leave 'queued' within
~20s and carry the live worker's fence (claimed_by). Then cleans up —
the worker's subsequent fenced writes no-op gracefully on the deleted
row (that's the StaleClaim path doing its job).

Run from KaizerBackend/ while the stack is up:
  python scripts/check_live_worker.py
"""
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from sqlalchemy import text  # noqa: E402

import models  # noqa: E402
from database import SessionLocal  # noqa: E402

NONCE = uuid.uuid4().hex[:8]


def main() -> int:
    db = SessionLocal()
    created: dict = {}
    try:
        tier = db.query(models.PlanTier).filter(
            models.PlanTier.name == "pro").first()
        user = models.User(email=f"live_{NONCE}@test.kaizer",
                           name=f"live-{NONCE}", is_active=True,
                           plan_tier_id=(tier.id if tier else None))
        db.add(user); db.flush()
        ch = models.Channel(user_id=user.id, name=f"live-ch-{NONCE}")
        db.add(ch); db.flush()
        tok = models.OAuthToken(channel_id=ch.id, google_channel_id=f"UCLIVE{NONCE}")
        db.add(tok); db.flush()
        jrow = models.Job(user_id=user.id, status="done",
                          platform="full_video_shorts_v4",
                          video_name=f"live_{NONCE}.mp4",
                          output_dir=f"/tmp/live_{NONCE}")
        db.add(jrow); db.flush()
        mv = models.MasterVideo(source_upload_id=jrow.id,
                                r2_key=f"test/live_{NONCE}.mp4",
                                duration_seconds=5.0, bytes=1, width=1920,
                                height=1080, status="ready",
                                pipeline_version="v4_clean", clean_master=True)
        db.add(mv); db.flush()
        pt = models.PublishTask(user_id=user.id, master_video_id=mv.id,
                                priority="critical", status="dispatched",
                                target_count=1, completed_count=0,
                                failed_count=0)
        db.add(pt); db.flush()
        job = models.UploadJobV2(
            publish_task_id=pt.id, channel_id=ch.id, oauth_token_id=tok.id,
            upload_path="rtmp", publish_kind="video", status="queued",
            attempts=0, idempotency_key=uuid.uuid4().hex + uuid.uuid4().hex,
            publish_version=f"live{NONCE}", predicted_quota_units=200,
            predicted_credit_cost=0, user_id=user.id, priority="critical",
        )
        db.add(job); db.commit()
        created = {"user": user.id, "ch": ch.id, "tok": tok.id,
                   "jrow": jrow.id, "mv": mv.id, "pt": pt.id, "job": job.id}
        print(f"inserted queued job id={job.id} (critical priority)")

        # The claim → branding-fail (fake R2 key) → requeue round-trip
        # can complete in <1s, leaving status back at 'queued' between
        # polls — so ALSO watch attempts / last_error / next_attempt_at,
        # which only the live worker can have touched.
        deadline = time.time() + 30
        seen = None
        while time.time() < deadline:
            row = db.execute(text(
                "SELECT status, claimed_by, attempts, last_error, "
                "       next_attempt_at > CURRENT_TIMESTAMP AS backing_off "
                "FROM upload_jobs_v2 WHERE id = :id"
            ), {"id": job.id}).fetchone()
            if row and (row[0] != "queued" or int(row[2] or 0) > 0):
                seen = row
                break
            time.sleep(0.5)

        if seen:
            print(f"LIVE WORKER PROCESSED IT: status={seen[0]!r} "
                  f"claimed_by={seen[1]!r} attempts={seen[2]} "
                  f"backing_off={seen[4]} last_error={(seen[3] or '')[:120]!r}")
            ok = True
        else:
            print("FAIL: untouched after 30s — live worker not claiming")
            ok = False
        return 0 if ok else 1
    finally:
        # Cleanup — worker writes after this no-op against the fence.
        uid = created.get("user")
        if uid:
            for stmt, p in [
                ("DELETE FROM quota_burn_log WHERE upload_job_id = :j",
                 {"j": created["job"]}),
                ("DELETE FROM publish_attempts WHERE upload_job_id = :j",
                 {"j": created["job"]}),
                ("DELETE FROM credit_ledger WHERE user_id = :u", {"u": uid}),
                ("DELETE FROM upload_jobs_v2 WHERE id = :j", {"j": created["job"]}),
                ("DELETE FROM publish_tasks WHERE id = :p", {"p": created["pt"]}),
                ("DELETE FROM master_videos WHERE id = :m", {"m": created["mv"]}),
                ("DELETE FROM jobs WHERE id = :jr", {"jr": created["jrow"]}),
                ("DELETE FROM oauth_tokens WHERE id = :t", {"t": created["tok"]}),
                ("DELETE FROM channels WHERE id = :c", {"c": created["ch"]}),
                ("DELETE FROM users WHERE id = :u", {"u": uid}),
            ]:
                try:
                    db.execute(text(stmt), p)
                    db.commit()
                except Exception:
                    db.rollback()
            print("fixtures cleaned up")
        db.close()


if __name__ == "__main__":
    sys.exit(main())
