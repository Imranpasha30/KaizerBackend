"""Phase 1.C scheduler smoke test (one-shot dev script).

Creates a synthetic User → PublishTask → UploadJobV2 directly via
SQLAlchemy (bypasses the HTTP layer because the auth flow is gnarly
and we just want to prove the scheduler dispatches), calls
``scheduler_enqueue``, then waits up to 5s for the job to progress
to ``completed``.

Asserts:
  - UploadJobV2.status == 'completed'
  - PublishTask.completed_count == 1
  - snapshot() reports queue empty and all slots returned

Run::

    cd kaizer/KaizerBackend
    python scripts/test_scheduler_smoke.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

# Make ``import models`` / ``import database`` work from the scripts dir.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# Silence noisy module-level logs unless KAIZER_SMOKE_VERBOSE=1.
import logging
logging.basicConfig(
    level=logging.INFO if os.environ.get("KAIZER_SMOKE_VERBOSE") else logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from database import SessionLocal, engine, Base
import models
from services import scheduler


SMOKE_EMAIL = "scheduler_smoke@kaizer.test"


def _ensure_plan_tier(db) -> models.PlanTier:
    pt = db.query(models.PlanTier).filter(models.PlanTier.name == "pro").first()
    if pt is None:
        pt = models.PlanTier(
            name="pro",
            monthly_credit_allotment=2000,
            slot_cap_active_uploads=20,
            direct_path_allowed=True,
            max_channels=-1,
            max_publishes_per_day=-1,
        )
        db.add(pt)
        db.commit()
        db.refresh(pt)
    return pt


def _ensure_user(db, plan_tier_id: int) -> models.User:
    user = db.query(models.User).filter(models.User.email == SMOKE_EMAIL).first()
    if user is None:
        user = models.User(
            email=SMOKE_EMAIL,
            name="Scheduler Smoke",
            plan="pro",
            plan_tier_id=plan_tier_id,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        if user.plan_tier_id != plan_tier_id:
            user.plan_tier_id = plan_tier_id
            db.commit()
    return user


def _create_master_video(db, user: models.User) -> models.MasterVideo:
    # Need a parent Job row (source_upload_id is NOT NULL + FK to jobs.id).
    job = models.Job(
        user_id=user.id,
        status="done",
        platform="youtube_full",
        video_name="scheduler_smoke.mp4",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    master = models.MasterVideo(
        source_upload_id=job.id,
        r2_key=f"smoke/master/{job.id}.mp4",
        duration_seconds=10.0,
        bytes=1024,
        width=1920,
        height=1080,
        status="ready",
        pipeline_version="v4_clean",
        clean_master=True,
    )
    db.add(master)
    db.commit()
    db.refresh(master)
    return master


def _ensure_channel(db, user: models.User) -> models.Channel:
    name = f"smoke-channel-{user.id}"
    ch = (
        db.query(models.Channel)
        .filter(models.Channel.user_id == user.id, models.Channel.name == name)
        .first()
    )
    if ch is None:
        ch = models.Channel(user_id=user.id, name=name, language="te")
        db.add(ch)
        db.commit()
        db.refresh(ch)
    return ch


def _create_publish_task_and_job(
    db, user: models.User, master: models.MasterVideo, channel: models.Channel,
) -> tuple[models.PublishTask, models.UploadJobV2]:
    pt = models.PublishTask(
        user_id=user.id,
        master_video_id=master.id,
        priority="normal",
        status="dispatched",
        target_count=1,
        completed_count=0,
        failed_count=0,
    )
    db.add(pt)
    db.commit()
    db.refresh(pt)

    # Unique idempotency_key per run so re-running doesn't collide.
    suffix = f"{int(time.time() * 1000)}"
    job = models.UploadJobV2(
        publish_task_id=pt.id,
        channel_id=channel.id,
        upload_path="rtmp",
        publish_kind="short",
        status="queued",
        attempts=0,
        idempotency_key=f"smoke-{suffix}"[:64],
        publish_version=f"smk:smk:{suffix[:8]}"[:40],
        predicted_quota_units=150,
        predicted_credit_cost=2,
        priority_at_dispatch="normal",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return pt, job


async def _main() -> int:
    # Ensure tables exist (relies on production migrations already
    # having been applied; this is belt + braces for fresh dev DBs).
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        plan_tier = _ensure_plan_tier(db)
        user = _ensure_user(db, plan_tier.id)
        channel = _ensure_channel(db, user)
        master = _create_master_video(db, user)
        pt, job = _create_publish_task_and_job(db, user, master, channel)
        job_id = job.id
        pt_id = pt.id
        user_id = user.id
        tier_name = plan_tier.name
    finally:
        db.close()

    print(f"[smoke] created PublishTask id={pt_id} UploadJobV2 id={job_id}")

    # Start the scheduler in this event loop.
    await scheduler.start()

    print(f"[smoke] scheduler.snapshot() (initial): {scheduler.snapshot()}")

    scheduler.scheduler_enqueue(
        upload_job_id=job_id,
        priority="normal",
        user_id=user_id,
        plan_tier_name=tier_name,
    )

    # Wait up to 5s for the job to reach 'completed'.
    deadline = time.monotonic() + 5.0
    final_status = None
    while time.monotonic() < deadline:
        await asyncio.sleep(0.1)
        db = SessionLocal()
        try:
            row = db.query(models.UploadJobV2).filter(
                models.UploadJobV2.id == job_id
            ).first()
            final_status = row.status if row else None
        finally:
            db.close()
        if final_status in ("completed", "failed"):
            break

    # Re-read the final state.
    db = SessionLocal()
    try:
        job_row = db.query(models.UploadJobV2).filter(
            models.UploadJobV2.id == job_id
        ).first()
        pt_row = db.query(models.PublishTask).filter(
            models.PublishTask.id == pt_id
        ).first()
        print(f"[smoke] final UploadJobV2.status = {job_row.status if job_row else None}")
        print(f"[smoke] final PublishTask.completed_count = {pt_row.completed_count if pt_row else None}")
        print(f"[smoke] final PublishTask.status = {pt_row.status if pt_row else None}")
    finally:
        db.close()

    snap = scheduler.snapshot()
    print(f"[smoke] scheduler.snapshot() (final): {snap}")

    # Shut down cleanly.
    await scheduler.shutdown()

    # Assertions.
    failures: list[str] = []
    if job_row is None or job_row.status != "completed":
        failures.append(
            f"expected UploadJobV2.status='completed', got {job_row.status if job_row else None!r}"
        )
    if pt_row is None or pt_row.completed_count != 1:
        failures.append(
            f"expected PublishTask.completed_count=1, got "
            f"{pt_row.completed_count if pt_row else None!r}"
        )
    if snap["queue_depth"] != 0:
        failures.append(f"expected queue_depth=0, got {snap['queue_depth']}")
    if snap["in_flight_count"] != 0:
        failures.append(f"expected in_flight_count=0, got {snap['in_flight_count']}")
    sm = snap.get("slot_manager", {})
    if sm.get("net_in_use", -1) != 0:
        failures.append(f"expected slot_manager.net_in_use=0, got {sm.get('net_in_use')!r}")

    if failures:
        print("[smoke] FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    rc = asyncio.run(_main())
    sys.exit(rc)
