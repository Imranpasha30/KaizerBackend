"""Wave 6 â€” Durable queue reliability suite.

Proves the Wave-1 guarantees against the LIVE dev Postgres with
synthetic fixtures and monkeypatched upload paths (zero YouTube calls,
zero R2 calls). Each scenario creates its own rows tagged with a run
nonce and cleans them up afterwards.

Scenarios:
  1. transient-error      â†’ outcome RETRY, status queued, future
                            next_attempt_at, NO refund, progress kept
  2. retries-exhausted    â†’ outcome FAILED, refunded exactly once
  3. kill-worker (reaper) â†’ claimed job w/ expired lease returns to
                            queued; second worker can claim it
  4. park â†’ unpark        â†’ credits refunded at park, RE-reserved at
                            unpark (ledger net zero), job queued again
  5. per-user cap + fence â†’ claim honors plan-tier active cap; a stale
                            worker's writes no-op against the fence
  6. permanent-error      â†’ outcome FAILED, refunded exactly once

Run from KaizerBackend/:  python scripts/test_durable_queue.py
Exit code 0 = all pass.
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Quota-bucket isolation: burn the 'testrun' bucket so fake-upload
# reservations never pollute the production 'oauth' counter the UI
# quota chip reports (see youtube/quota_v2._key_hash).
os.environ.setdefault("KAIZER_YT_QUOTA_BUCKET", "testrun")
# Deterministic cap for suites: the REAL resolved cap is Google's 10k,
# which the shared testrun bucket exceeds after a few runs in one day.
# Env override is resolution priority #1 and never touches production
# (the bucket isolation above keeps the real counter clean).
os.environ.setdefault("KAIZER_YT_DAILY_QUOTA_CAP", "1000000")

# Windows consoles default to cp1252 â€” keep the suite's arrows printable.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from sqlalchemy import text  # noqa: E402

import models  # noqa: E402
from database import SessionLocal, engine  # noqa: E402

NONCE = uuid.uuid4().hex[:8]
PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


# â”€â”€â”€ Fixtures â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def make_fixtures(db, *, tier_name: str = "pro") -> dict:
    """User + channel + oauth token + job + master + publish task."""
    tier = db.query(models.PlanTier).filter(
        models.PlanTier.name == tier_name).first()
    user = models.User(
        email=f"dq_{NONCE}_{uuid.uuid4().hex[:6]}@test.kaizer",
        name=f"dq-test-{NONCE}",
        is_active=True,
        plan_tier_id=(tier.id if tier else None),
    )
    db.add(user)
    db.flush()

    channel = models.Channel(user_id=user.id, name=f"dq-ch-{NONCE}-{uuid.uuid4().hex[:4]}")
    db.add(channel)
    db.flush()

    tok = models.OAuthToken(channel_id=channel.id, google_channel_id=f"UC{NONCE}")
    db.add(tok)
    db.flush()

    job_row = models.Job(
        user_id=user.id, status="done", platform="full_video_shorts_v4",
        video_name=f"dq_{NONCE}.mp4", output_dir=f"/tmp/dq_{NONCE}",
    )
    db.add(job_row)
    db.flush()

    master = models.MasterVideo(
        source_upload_id=job_row.id, r2_key=f"test/dq_{NONCE}.mp4",
        duration_seconds=10.0, bytes=1000, width=1920, height=1080,
        status="ready", pipeline_version="v4_clean", clean_master=True,
    )
    db.add(master)
    db.flush()

    pt = models.PublishTask(
        user_id=user.id, master_video_id=master.id, priority="normal",
        status="dispatched", target_count=1,
        completed_count=0, failed_count=0,
    )
    db.add(pt)
    db.flush()
    db.commit()
    return {"user": user, "channel": channel, "token": tok,
            "job_row": job_row, "master": master, "pt": pt,
            "tier": tier}


def make_upload_job(db, fx, *, credits_reserved: int = 16,
                    status: str = "queued") -> models.UploadJobV2:
    j = models.UploadJobV2(
        publish_task_id=fx["pt"].id,
        channel_id=fx["channel"].id,
        oauth_token_id=fx["token"].id,
        upload_path="direct",
        publish_kind="video",
        status=status,
        attempts=0,
        idempotency_key=uuid.uuid4().hex + uuid.uuid4().hex,  # 64 hex
        publish_version=f"dq{NONCE}",
        predicted_quota_units=1650,
        predicted_credit_cost=credits_reserved,
        branded_artifact_r2_key=f"branded/test/dq_{NONCE}.mp4",
        user_id=fx["user"].id,
        priority="normal",
    )
    db.add(j)
    db.flush()
    if credits_reserved:
        # Seed a balance + the reserve row the fanout would have made.
        from services import credits as credits_svc
        db.execute(text(
            "INSERT INTO credit_ledger (user_id, delta, reason, balance_after, created_at) "
            "VALUES (:u, 1000, 'admin_adjustment', 1000, CURRENT_TIMESTAMP)"
        ), {"u": fx["user"].id})
        credits_svc.reserve(
            db, user_id=fx["user"].id, cost=credits_reserved,
            reason="upload_direct", upload_job_id=j.id, path="direct",
            publish_kind="video", predicted_quota_units=1650,
        )
    db.commit()
    return j


def cleanup(db, fx) -> None:
    uid = fx["user"].id
    for stmt, params in [
        ("DELETE FROM quota_burn_log WHERE upload_job_id IN "
         "(SELECT id FROM upload_jobs_v2 WHERE user_id = :u)", {"u": uid}),
        ("DELETE FROM publish_attempts WHERE upload_job_id IN "
         "(SELECT id FROM upload_jobs_v2 WHERE user_id = :u)", {"u": uid}),
        ("DELETE FROM credit_ledger WHERE user_id = :u", {"u": uid}),
        ("DELETE FROM upload_jobs_v2 WHERE user_id = :u", {"u": uid}),
        ("DELETE FROM publish_tasks WHERE user_id = :u", {"u": uid}),
        ("DELETE FROM master_videos WHERE id = :m", {"m": fx["master"].id}),
        ("DELETE FROM jobs WHERE id = :j", {"j": fx["job_row"].id}),
        ("DELETE FROM oauth_tokens WHERE id = :t", {"t": fx["token"].id}),
        ("DELETE FROM channels WHERE id = :c", {"c": fx["channel"].id}),
        ("DELETE FROM users WHERE id = :u", {"u": uid}),
    ]:
        try:
            db.execute(text(stmt), params)
        except Exception:
            db.rollback()
    db.commit()


def refund_rows(db, user_id: int, job_id: int) -> int:
    return db.execute(text(
        "SELECT count(*) FROM credit_ledger WHERE user_id = :u "
        "AND upload_job_id = :j AND reason = 'refund'"
    ), {"u": user_id, "j": job_id}).scalar()


# â”€â”€â”€ Monkeypatch helpers (no YouTube / no R2) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class _FakeCreds:
    token = "fake"


def patch_dispatch(monkey: dict, *, upload_behavior) -> None:
    """Patch upload_dispatch's external surfaces in-place.
    upload_behavior(job) -> dict result, or raises."""
    from services import upload_dispatch as ud

    monkey["_orig"] = {
        "get_credentials": ud._yt_oauth.get_credentials,
        "_download": ud._download_branded_artifact,
        "_quota": ud._quota_reserve,
        # Direct (videos.insert 100/day) bucket gate — patched so the suite
        # never reserves against the operator's REAL daily bucket. Default
        # fixtures use upload_path='direct', which gates on _quota_reserve_upload
        # (NOT _quota_reserve); without this the park scenario can't trigger and
        # success scenarios silently burn real quota.
        "_quota_upload": ud._quota_reserve_upload,
        "_direct": ud._do_direct_upload,
        "_idem": ud._idempotency_short_circuit,
    }
    ud._yt_oauth.get_credentials = lambda db, ch_id: _FakeCreds()
    ud._download_branded_artifact = lambda key, d: os.path.join(d, "x.mp4")

    def _fake_download(key, dest_dir):
        p = os.path.join(dest_dir, "_branded.mp4")
        with open(p, "wb") as f:
            f.write(b"0" * 1024)
        return p
    ud._download_branded_artifact = _fake_download
    ud._quota_reserve = lambda db, cost: True
    ud._quota_reserve_upload = lambda db: True
    ud._idempotency_short_circuit = lambda db, job, creds, gcid: False

    def _fake_direct(db, job, creds, branded_local, **kw):
        return upload_behavior(job)
    ud._do_direct_upload = _fake_direct


def unpatch_dispatch(monkey: dict) -> None:
    from services import upload_dispatch as ud
    o = monkey.get("_orig") or {}
    if not o:
        return
    ud._yt_oauth.get_credentials = o["get_credentials"]
    ud._download_branded_artifact = o["_download"]
    ud._quota_reserve = o["_quota"]
    ud._quota_reserve_upload = o["_quota_upload"]
    ud._do_direct_upload = o["_direct"]
    ud._idempotency_short_circuit = o["_idem"]


# â”€â”€â”€ Scenarios â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def scenario_transient_then_success() -> None:
    print("\n[1] transient error â†’ RETRY â†’ eventual success, credits intact")
    from services import job_queue, upload_dispatch as ud
    from youtube import uploader_v2

    db = SessionLocal()
    fx = make_fixtures(db)
    job = make_upload_job(db, fx)
    worker = f"dqtest:{NONCE}:w1"
    monkey: dict = {}
    try:
        claimed = job_queue.claim_jobs(worker, batch=1)
        check("claimed the queued job", any(c.id == job.id for c in claimed),
              f"claimed={claimed}")

        calls = {"n": 0}

        def behavior(j):
            calls["n"] += 1
            if calls["n"] == 1:
                raise uploader_v2.TransientUploadError("synthetic 503")
            return {"video_id": f"vid{NONCE}", "bytes_uploaded": 1024}

        patch_dispatch(monkey, upload_behavior=behavior)

        out1 = ud.process(job.id, worker, time.monotonic() + 300)
        check("first attempt â†’ RETRY", out1 == ud.Outcome.RETRY, f"got {out1}")

        db.expire_all()
        row = db.query(models.UploadJobV2).get(job.id)
        check("status back to queued", row.status == "queued", row.status)
        check("next_attempt_at in the future",
              row.next_attempt_at is not None
              and row.next_attempt_at.replace(tzinfo=None)
              > datetime.utcnow() - timedelta(seconds=1))
        check("NO refund on transient",
              refund_rows(db, fx["user"].id, job.id) == 0)
        check("attempts == 1 (claim bump only)", int(row.attempts) == 1,
              f"attempts={row.attempts}")

        # Make it claimable immediately + retry â†’ success.
        db.execute(text(
            "UPDATE upload_jobs_v2 SET next_attempt_at = CURRENT_TIMESTAMP "
            "WHERE id = :id"), {"id": job.id})
        db.commit()
        claimed2 = job_queue.claim_jobs(worker, batch=1)
        check("re-claimed after backoff", any(c.id == job.id for c in claimed2))
        out2 = ud.process(job.id, worker, time.monotonic() + 300)
        check("second attempt â†’ COMPLETED", out2 == ud.Outcome.COMPLETED,
              f"got {out2}")
        db.expire_all()
        row = db.query(models.UploadJobV2).get(job.id)
        check("youtube_video_id recorded",
              (row.youtube_video_id or "") == f"vid{NONCE}",
              row.youtube_video_id)
        check("still no refund after success",
              refund_rows(db, fx["user"].id, job.id) == 0)
    finally:
        unpatch_dispatch(monkey)
        cleanup(db, fx)
        db.close()


def scenario_exhausted() -> None:
    print("\n[2] retries exhausted â†’ FAILED + refunded exactly once")
    from services import job_queue, upload_dispatch as ud
    from youtube import uploader_v2

    db = SessionLocal()
    fx = make_fixtures(db)
    job = make_upload_job(db, fx)
    worker = f"dqtest:{NONCE}:w2"
    monkey: dict = {}
    try:
        patch_dispatch(monkey, upload_behavior=lambda j: (_ for _ in ()).throw(
            uploader_v2.TransientUploadError("always 503")))

        outcomes = []
        for _ in range(job_queue.max_attempts() + 1):
            db.execute(text(
                "UPDATE upload_jobs_v2 SET next_attempt_at = CURRENT_TIMESTAMP "
                "WHERE id = :id AND status = 'queued'"), {"id": job.id})
            db.commit()
            claimed = job_queue.claim_jobs(worker, batch=1)
            if not any(c.id == job.id for c in claimed):
                break
            outcomes.append(ud.process(job.id, worker, time.monotonic() + 300))

        check("eventually FAILED",
              outcomes and outcomes[-1] == ud.Outcome.FAILED,
              f"outcomes={[str(o) for o in outcomes]}")
        db.expire_all()
        row = db.query(models.UploadJobV2).get(job.id)
        check("status failed", row.status == "failed", row.status)
        check("refunded exactly once",
              refund_rows(db, fx["user"].id, job.id) == 1,
              f"refunds={refund_rows(db, fx['user'].id, job.id)}")
    finally:
        unpatch_dispatch(monkey)
        cleanup(db, fx)
        db.close()


def scenario_reaper() -> None:
    print("\n[3] kill-worker â†’ lease expires â†’ reaper requeues â†’ reclaimable")
    from services import job_queue

    db = SessionLocal()
    fx = make_fixtures(db)
    job = make_upload_job(db, fx, credits_reserved=0)
    w1 = f"dqtest:{NONCE}:dead"
    w2 = f"dqtest:{NONCE}:alive"
    try:
        claimed = job_queue.claim_jobs(w1, batch=1)
        check("worker1 claimed", any(c.id == job.id for c in claimed))
        # Simulate worker death: expire the lease.
        db.execute(text(
            "UPDATE upload_jobs_v2 SET lease_expires_at = "
            "CURRENT_TIMESTAMP - interval '5 minutes' WHERE id = :id"
        ), {"id": job.id})
        db.commit()
        reaped = job_queue.reap_expired()
        check("reaper requeued it", job.id in reaped, f"reaped={reaped}")
        db.expire_all()
        row = db.query(models.UploadJobV2).get(job.id)
        check("status queued after reap", row.status == "queued", row.status)
        check("claim cleared", row.claimed_by is None, row.claimed_by)
        # 30s delay before reclaim â€” fast-forward.
        db.execute(text(
            "UPDATE upload_jobs_v2 SET next_attempt_at = CURRENT_TIMESTAMP "
            "WHERE id = :id"), {"id": job.id})
        db.commit()
        claimed2 = job_queue.claim_jobs(w2, batch=1)
        check("worker2 reclaimed", any(c.id == job.id for c in claimed2))
        db.expire_all()
        row = db.query(models.UploadJobV2).get(job.id)
        check("fence now worker2", row.claimed_by == w2, row.claimed_by)
        check("attempts counted both claims", int(row.attempts) == 2,
              f"attempts={row.attempts}")
    finally:
        cleanup(db, fx)
        db.close()


def scenario_park_unpark() -> None:
    print("\n[4] quota park â†’ refund â†’ unpark â†’ credits re-reserved")
    from services import job_queue, upload_dispatch as ud, cron_runner

    db = SessionLocal()
    fx = make_fixtures(db)
    job = make_upload_job(db, fx)
    worker = f"dqtest:{NONCE}:w4"
    monkey: dict = {}
    try:
        claimed = job_queue.claim_jobs(worker, batch=1)
        check("claimed", any(c.id == job.id for c in claimed))

        patch_dispatch(monkey, upload_behavior=lambda j: {"video_id": "x"})
        from services import upload_dispatch as _ud
        _orig_quota = _ud._quota_reserve
        _orig_quota_upload = _ud._quota_reserve_upload
        # The default fixture is upload_path='direct', which gates on
        # _quota_reserve_upload — patch BOTH so the park path triggers
        # regardless of which gate the dispatcher consults.
        _ud._quota_reserve = lambda db_, cost: False          # rtmp/non-direct gate says NO
        _ud._quota_reserve_upload = lambda db_: False          # direct (videos.insert) gate says NO
        try:
            out = ud.process(job.id, worker, time.monotonic() + 300)
        finally:
            _ud._quota_reserve = _orig_quota
            _ud._quota_reserve_upload = _orig_quota_upload
        check("outcome PARKED_QUOTA", out == ud.Outcome.PARKED_QUOTA, str(out))
        db.expire_all()
        row = db.query(models.UploadJobV2).get(job.id)
        check("status parked_quota", row.status == "parked_quota", row.status)
        check("refunded at park",
              refund_rows(db, fx["user"].id, job.id) == 1)

        balance_before = db.execute(text(
            "SELECT balance_after FROM credit_ledger WHERE user_id = :u "
            "ORDER BY id DESC LIMIT 1"), {"u": fx["user"].id}).scalar()

        rows = job_queue.unpark_batch(50)
        mine = [r for r in rows if r["id"] == job.id]
        check("unparked", bool(mine), f"unparked={rows}")
        if mine:
            cron_runner._re_reserve_credits(mine)
        db.expire_all()
        row = db.query(models.UploadJobV2).get(job.id)
        check("queued after unpark", row.status == "queued", row.status)
        balance_after = db.execute(text(
            "SELECT balance_after FROM credit_ledger WHERE user_id = :u "
            "ORDER BY id DESC LIMIT 1"), {"u": fx["user"].id}).scalar()
        check("credits re-reserved on unpark (balance dropped by cost)",
              int(balance_after) == int(balance_before) - 16,
              f"{balance_before} -> {balance_after}")
    finally:
        unpatch_dispatch(monkey)
        cleanup(db, fx)
        db.close()


def scenario_cap_and_fence() -> None:
    print("\n[5] per-user active cap enforced + stale-worker fence")
    from services import job_queue, upload_dispatch as ud

    db = SessionLocal()
    # Free tier caps active uploads at 5 â€” make 7 jobs, expect â‰¤5 claims.
    fx = make_fixtures(db, tier_name="free")
    jobs = [make_upload_job(db, fx, credits_reserved=0) for _ in range(7)]
    worker = f"dqtest:{NONCE}:w5"
    try:
        cap = int(fx["tier"].slot_cap_active_uploads) if fx["tier"] else 5
        claimed = job_queue.claim_jobs(worker, batch=10)
        mine = [c for c in claimed if c.id in {j.id for j in jobs}]
        check(f"claims capped at tier cap ({cap})", len(mine) <= cap,
              f"claimed {len(mine)}")
        check("claimed more than zero", len(mine) > 0)

        # Fence: a stale worker can't flip a job it doesn't own.
        target = mine[0].id
        ok = ud._guarded_job_update(
            db, target, "someone:else:zzz", {"status": "failed"},
        )
        check("stale worker write fenced out", ok is False, str(ok))
        db.expire_all()
        row = db.query(models.UploadJobV2).get(target)
        check("row untouched by stale write", row.status == "claimed",
              row.status)
        # Rightful owner CAN write.
        ok2 = ud._guarded_job_update(db, target, worker, {"status": "claimed"})
        check("owner write passes fence", ok2 is True, str(ok2))
    finally:
        cleanup(db, fx)
        db.close()


def scenario_permanent() -> None:
    print("\n[6] permanent 4xx â†’ FAILED + single refund")
    from services import job_queue, upload_dispatch as ud
    from youtube import uploader_v2

    db = SessionLocal()
    fx = make_fixtures(db)
    job = make_upload_job(db, fx)
    worker = f"dqtest:{NONCE}:w6"
    monkey: dict = {}
    try:
        job_queue.claim_jobs(worker, batch=1)
        patch_dispatch(monkey, upload_behavior=lambda j: (_ for _ in ()).throw(
            uploader_v2.UploadError("invalid video metadata (400)")))
        out = ud.process(job.id, worker, time.monotonic() + 300)
        check("outcome FAILED", out == ud.Outcome.FAILED, str(out))
        db.expire_all()
        row = db.query(models.UploadJobV2).get(job.id)
        check("status failed", row.status == "failed", row.status)
        check("refunded exactly once",
              refund_rows(db, fx["user"].id, job.id) == 1)
        # Idempotent: a second fail_terminal must not double-refund.
        ud.fail_terminal(job.id, None, "double-fail probe")
        check("second terminal-fail did NOT double-refund",
              refund_rows(db, fx["user"].id, job.id) == 1)
    finally:
        unpatch_dispatch(monkey)
        cleanup(db, fx)
        db.close()


def main() -> int:
    if engine.dialect.name != "postgresql":
        print("This suite requires the Postgres dev DB.")
        return 2
    print(f"Durable-queue reliability suite (nonce={NONCE})")
    for fn in (scenario_transient_then_success, scenario_exhausted,
               scenario_reaper, scenario_park_unpark,
               scenario_cap_and_fence, scenario_permanent):
        try:
            fn()
        except Exception as exc:
            FAIL.append(fn.__name__)
            print(f"  FAIL  {fn.__name__} crashed: {exc!r}")
            import traceback
            traceback.print_exc()
    print(f"\n{'='*60}\n{len(PASS)} checks passed, {len(FAIL)} failed")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
