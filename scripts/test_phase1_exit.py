"""Phase 1 Integration Exit-Test (KAIZER_UPLOAD_REWRITE_BRIEF.md §6).

Exercises 4 scenarios end-to-end against the real dev Postgres DB:

A. 100-channel fanout → 100 UploadJobs → scheduler drains them.
B. Per-user slot cap respected (max concurrent never exceeds plan cap).
C. Priority precedence — critical jumps queued low-priority jobs.
D. Legacy + flag-gated routes still behave (no 500s).

All scenarios use direct SQLAlchemy + service calls (no HTTP for A/B/C)
because the auth gate makes HTTP-from-script gnarly; the goal is to
validate the publish/scheduler internals which is what Phase 1 is about.
Scenario D uses HTTP to prove the auth + feature-flag plumbing didn't
regress.

Run::

    cd kaizer/KaizerBackend
    python scripts/test_phase1_exit.py

Exit code 0 on all-pass, 1 on any failure.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# Make ``import models`` / ``import database`` work from the scripts dir.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

logging.basicConfig(
    level=logging.INFO if os.environ.get("KAIZER_EXIT_VERBOSE") else logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from database import SessionLocal  # noqa: E402
import models  # noqa: E402
from services import fanout as fanout_svc  # noqa: E402
from services import scheduler  # noqa: E402


# ─── Constants ──────────────────────────────────────────────────────────────

EMAIL_PREFIX = "phase1exit_test_"
EMAIL_DOMAIN = "@kaizer.test"
HTTP_BASE = "http://127.0.0.1:8000"
HTTP_TIMEOUT = 5.0


# ─── Result tracking ────────────────────────────────────────────────────────


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    reason: str = ""
    metrics: dict = field(default_factory=dict)


# ─── Test data factories ────────────────────────────────────────────────────


def _unique_email(scenario_tag: str) -> str:
    return f"{EMAIL_PREFIX}{scenario_tag}_{uuid.uuid4().hex[:12]}{EMAIL_DOMAIN}"


def _ensure_pro_tier(db) -> models.PlanTier:
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


def _ensure_test_tiny_tier(db) -> models.PlanTier:
    """Synthetic plan tier for Scenario B (small slot cap, RTMP-only).

    Free's max_channels=1 + max_publishes_per_day=1 would block a 50-channel
    fanout request; this tier lifts those caps but keeps slot_cap=5 +
    direct_path_allowed=False so the scenario can exercise the cap +
    Decision-6 enforcement together.
    """
    pt = db.query(models.PlanTier).filter(models.PlanTier.name == "test_tiny").first()
    if pt is None:
        pt = models.PlanTier(
            name="test_tiny",
            monthly_credit_allotment=10_000,    # plenty for the scenario
            slot_cap_active_uploads=5,
            direct_path_allowed=False,           # RTMP-only
            max_channels=-1,                     # bypass Free's 1-channel limit
            max_publishes_per_day=-1,            # bypass Free's daily cap
        )
        db.add(pt)
        db.commit()
        db.refresh(pt)
    return pt


def _create_user(db, scenario_tag: str, plan_tier: models.PlanTier) -> models.User:
    email = _unique_email(scenario_tag)
    user = models.User(
        email=email,
        name=f"Phase 1 Exit Test ({scenario_tag})",
        plan="pro",
        plan_tier_id=plan_tier.id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _create_master_video(db, user: models.User, tag: str) -> models.MasterVideo:
    job = models.Job(
        user_id=user.id,
        status="done",
        platform="youtube_full",
        video_name=f"phase1exit_{tag}.mp4",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    master = models.MasterVideo(
        source_upload_id=job.id,
        r2_key=f"phase1exit/master/{job.id}.mp4",
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


def _create_channels_and_tokens(
    db, user: models.User, n: int, tag: str,
) -> list[models.Channel]:
    """Create N channels (each with its own OAuthToken stub).

    OAuthToken.channel_id is UNIQUE so one token per channel — the
    Fanout service requires the token row to exist for every target.
    """
    channels: list[models.Channel] = []
    for i in range(n):
        ch = models.Channel(
            user_id=user.id,
            name=f"phase1exit-{tag}-ch{i:03d}-{uuid.uuid4().hex[:6]}",
            language="te",
        )
        db.add(ch)
    db.commit()
    # Re-fetch the channels we just created (by user_id + the name prefix
    # used above), sorted by id so test asserts are deterministic.
    channels = (
        db.query(models.Channel)
        .filter(
            models.Channel.user_id == user.id,
            models.Channel.name.like(f"phase1exit-{tag}-%"),
        )
        .order_by(models.Channel.id.asc())
        .all()
    )
    assert len(channels) == n, (
        f"created {n} channels but reloaded only {len(channels)}"
    )
    for ch in channels:
        tok = models.OAuthToken(
            channel_id=ch.id,
            google_channel_id=f"GCID-{ch.id}",
            google_channel_title=ch.name,
            refresh_token_enc="",
            access_token_enc="",
            scopes="",
        )
        db.add(tok)
    db.commit()
    return channels


def _seed_credits(db, user: models.User, amount: int) -> None:
    """Seed credits via admin_adjustment ledger row (CONTRACTS §3.5)."""
    if amount <= 0:
        return
    row = models.CreditLedger(
        user_id=user.id,
        delta=amount,
        reason="admin_adjustment",
        upload_job_id=None,
        path=None,
        publish_kind=None,
        predicted_quota_units=None,
        balance_after=amount,    # this user has no prior balance
    )
    db.add(row)
    db.commit()


# ─── Cleanup ────────────────────────────────────────────────────────────────


def _cleanup_user(db, user_id: int) -> None:
    """Delete every row created by/for the given test user.

    Order matters because of FK constraints:
      credit_ledger → upload_jobs_v2 → publish_attempts → publish_tasks →
      oauth_tokens → channels → master_videos → jobs → users.
    """
    if user_id <= 0:
        return

    # PublishTasks owned by user → UploadJobV2 → PublishAttempts.
    pt_ids = [
        r.id for r in db.query(models.PublishTask).filter(
            models.PublishTask.user_id == user_id
        ).all()
    ]
    if pt_ids:
        uj_ids = [
            r.id for r in db.query(models.UploadJobV2).filter(
                models.UploadJobV2.publish_task_id.in_(pt_ids)
            ).all()
        ]
        if uj_ids:
            db.query(models.PublishAttempt).filter(
                models.PublishAttempt.upload_job_id.in_(uj_ids)
            ).delete(synchronize_session=False)
            db.query(models.CreditLedger).filter(
                models.CreditLedger.upload_job_id.in_(uj_ids)
            ).delete(synchronize_session=False)
            db.query(models.UploadJobV2).filter(
                models.UploadJobV2.id.in_(uj_ids)
            ).delete(synchronize_session=False)
        db.query(models.PublishTask).filter(
            models.PublishTask.id.in_(pt_ids)
        ).delete(synchronize_session=False)

    # Remaining credit_ledger rows for this user (admin_adjustment + refunds).
    db.query(models.CreditLedger).filter(
        models.CreditLedger.user_id == user_id
    ).delete(synchronize_session=False)

    # MasterVideos that point at Jobs owned by this user.
    job_ids = [
        r.id for r in db.query(models.Job).filter(
            models.Job.user_id == user_id
        ).all()
    ]
    if job_ids:
        db.query(models.MasterVideo).filter(
            models.MasterVideo.source_upload_id.in_(job_ids)
        ).delete(synchronize_session=False)

    # OAuthTokens for this user's channels.
    ch_ids = [
        r.id for r in db.query(models.Channel).filter(
            models.Channel.user_id == user_id
        ).all()
    ]
    if ch_ids:
        db.query(models.OAuthToken).filter(
            models.OAuthToken.channel_id.in_(ch_ids)
        ).delete(synchronize_session=False)
        db.query(models.Channel).filter(
            models.Channel.id.in_(ch_ids)
        ).delete(synchronize_session=False)

    if job_ids:
        db.query(models.Job).filter(
            models.Job.id.in_(job_ids)
        ).delete(synchronize_session=False)

    db.query(models.User).filter(models.User.id == user_id).delete(
        synchronize_session=False
    )
    db.commit()


def _purge_stale_test_users() -> None:
    """Idempotency belt-and-braces: wipe any leftover test users from
    prior failed runs before starting."""
    db = SessionLocal()
    try:
        users = db.query(models.User).filter(
            models.User.email.like(f"{EMAIL_PREFIX}%{EMAIL_DOMAIN}")
        ).all()
        for u in users:
            print(f"[setup] purging stale test user id={u.id} email={u.email}")
            _cleanup_user(db, u.id)
    finally:
        db.close()


# ─── Helpers ────────────────────────────────────────────────────────────────


def _build_request(
    master_video_id: int,
    channels: list[models.Channel],
    *,
    upload_path: str,
    publish_kind: str,
    priority: str,
    tag: str,
) -> fanout_svc.PublishTaskRequest:
    targets: list[fanout_svc.FanoutTarget] = []
    for i, ch in enumerate(channels):
        # Distinct version inputs per target so idempotency keys differ
        # even if the same MasterVideo + channel is replayed in a future run.
        # Channel id alone already varies; we just add some headroom.
        targets.append(
            fanout_svc.FanoutTarget(
                channel_id=ch.id,
                upload_path=upload_path,
                publish_kind=publish_kind,
                brand_profile_id=None,
                thumbnail_source=(
                    "pipeline_generated" if publish_kind == "video" else None
                ),
                thumbnail_r2_key=None,
                scheduled_at=None,
                brand_profile_version=f"brand-{tag}",
                seo_version=f"seo-{tag}-{i:04d}",
                metadata_version=f"meta-{tag}-{i:04d}",
            )
        )
    return fanout_svc.PublishTaskRequest(
        master_video_id=master_video_id,
        targets=targets,
        priority=priority,
    )


def _balance(db, user_id: int) -> int:
    from sqlalchemy import func
    val = (
        db.query(func.coalesce(func.sum(models.CreditLedger.delta), 0))
        .filter(models.CreditLedger.user_id == user_id)
        .scalar()
    )
    return int(val or 0)


def _http_probe(method: str, path: str, body: dict | None = None) -> int:
    """Returns the HTTP status code (or -1 on transport error)."""
    url = f"{HTTP_BASE}{path}"
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"[probe] {method} {path} transport error: {exc!r}")
        return -1


# ─── Scenarios ──────────────────────────────────────────────────────────────


async def scenario_a_fanout_100() -> ScenarioResult:
    """Scenario A — 100-channel fanout."""
    name = "A: 100-channel fanout"
    user_id = 0
    metrics: dict = {}
    try:
        # ── Setup ──────────────────────────────────────────────────────────
        db = SessionLocal()
        try:
            tier = _ensure_pro_tier(db)
            user = _create_user(db, "scenA", tier)
            user_id = user.id
            channels = _create_channels_and_tokens(db, user, 100, "scenA")
            master = _create_master_video(db, user, "scenA")
            _seed_credits(db, user, 100 * 16)  # 1600 cr → exactly drained
            request = _build_request(
                master.id, channels,
                upload_path="direct", publish_kind="video",
                priority="normal", tag="scenA",
            )
            # Reload user to attach plan_tier relationship.
            user_loaded = db.query(models.User).filter(
                models.User.id == user.id
            ).first()
            result = fanout_svc.create_publish_task(db, user_loaded, request)
            db.commit()
            pt_id = result.publish_task_id
            job_ids = list(result.upload_job_ids)
            balance_after_fanout = _balance(db, user_id)
        finally:
            db.close()

        # ── Asserts: fanout shape ──────────────────────────────────────────
        if len(job_ids) != 100:
            return ScenarioResult(
                name, False,
                f"expected 100 upload_job_ids, got {len(job_ids)}",
                metrics,
            )
        db = SessionLocal()
        try:
            pt_row = db.query(models.PublishTask).filter(
                models.PublishTask.id == pt_id
            ).first()
            if pt_row is None or int(pt_row.target_count) != 100:
                return ScenarioResult(
                    name, False,
                    f"PublishTask.target_count != 100 "
                    f"(got {pt_row.target_count if pt_row else None})",
                    metrics,
                )
            queued = db.query(models.UploadJobV2).filter(
                models.UploadJobV2.publish_task_id == pt_id
            ).count()
            if queued != 100:
                return ScenarioResult(
                    name, False,
                    f"expected 100 UploadJobV2 rows, got {queued}",
                    metrics,
                )
        finally:
            db.close()
        if balance_after_fanout != 0:
            return ScenarioResult(
                name, False,
                f"expected balance=0 after deduction, got {balance_after_fanout}",
                metrics,
            )

        # ── Start scheduler + drain ────────────────────────────────────────
        await scheduler.start()
        peak_net = 0
        peak_user = 0
        t_start = time.monotonic()
        for jid in job_ids:
            scheduler.scheduler_enqueue(
                upload_job_id=jid,
                priority="normal",
                user_id=user_id,
                plan_tier_name="pro",
            )

        deadline = t_start + 60.0
        completed = 0
        while time.monotonic() < deadline:
            snap = scheduler.snapshot()
            sm = snap.get("slot_manager", {})
            peak_net = max(peak_net, int(sm.get("net_in_use", 0) or 0))
            peak_user = max(
                peak_user,
                int(sm.get("user_in_use", {}).get(user_id, 0) or 0),
            )
            await asyncio.sleep(0.25)
            db = SessionLocal()
            try:
                completed = db.query(models.UploadJobV2).filter(
                    models.UploadJobV2.publish_task_id == pt_id,
                    models.UploadJobV2.status == "completed",
                ).count()
            finally:
                db.close()
            if completed >= 100:
                break
        wall_seconds = time.monotonic() - t_start

        # Final snapshot (must be after all jobs settle).
        # Tiny grace tick so the slot-release async-with's `finally`
        # paths get to run after the last commit.
        await asyncio.sleep(0.5)
        snap_final = scheduler.snapshot()
        sm_final = snap_final.get("slot_manager", {})

        await scheduler.shutdown()

        # ── Final asserts ─────────────────────────────────────────────────
        db = SessionLocal()
        try:
            completed = db.query(models.UploadJobV2).filter(
                models.UploadJobV2.publish_task_id == pt_id,
                models.UploadJobV2.status == "completed",
            ).count()
            failed = db.query(models.UploadJobV2).filter(
                models.UploadJobV2.publish_task_id == pt_id,
                models.UploadJobV2.status == "failed",
            ).count()
            pt_row = db.query(models.PublishTask).filter(
                models.PublishTask.id == pt_id
            ).first()
        finally:
            db.close()

        throughput = (100 / wall_seconds) if wall_seconds > 0 else float("inf")
        metrics.update({
            "wall_seconds": round(wall_seconds, 3),
            "throughput_jobs_per_sec": round(throughput, 2),
            "peak_net_in_use": peak_net,
            "peak_user_in_use": peak_user,
            "balance_after_fanout": balance_after_fanout,
            "snap_final_queue_depth": snap_final.get("queue_depth"),
            "snap_final_in_flight": snap_final.get("in_flight_count"),
            "snap_final_net_in_use": sm_final.get("net_in_use"),
            "snap_final_user_in_use": sm_final.get("user_in_use", {}).get(user_id, 0),
            "completed_count": completed,
            "failed_count": failed,
        })

        failures: list[str] = []
        if completed != 100:
            failures.append(f"expected 100 completed jobs, got {completed}")
        if failed != 0:
            failures.append(f"expected 0 failed jobs, got {failed}")
        if pt_row is None or int(pt_row.completed_count) != 100:
            failures.append(
                f"PublishTask.completed_count != 100 "
                f"(got {pt_row.completed_count if pt_row else None})"
            )
        if pt_row is None or int(pt_row.failed_count) != 0:
            failures.append(
                f"PublishTask.failed_count != 0 "
                f"(got {pt_row.failed_count if pt_row else None})"
            )
        if pt_row is None or pt_row.status != "completed":
            failures.append(
                f"PublishTask.status != 'completed' "
                f"(got {pt_row.status if pt_row else None!r})"
            )
        if int(snap_final.get("queue_depth", -1)) != 0:
            failures.append(f"queue_depth != 0 (got {snap_final.get('queue_depth')})")
        if int(snap_final.get("in_flight_count", -1)) != 0:
            failures.append(
                f"in_flight_count != 0 (got {snap_final.get('in_flight_count')})"
            )
        if int(sm_final.get("net_in_use", -1) or 0) != 0:
            failures.append(
                f"slot_manager.net_in_use != 0 "
                f"(got {sm_final.get('net_in_use')})"
            )
        if int(sm_final.get("cpu_in_use", -1) or 0) != 0:
            failures.append(
                f"slot_manager.cpu_in_use != 0 "
                f"(got {sm_final.get('cpu_in_use')})"
            )
        final_user_in_use = int(sm_final.get("user_in_use", {}).get(user_id, 0) or 0)
        if final_user_in_use != 0:
            failures.append(
                f"slot_manager.user_in_use[{user_id}] != 0 "
                f"(got {final_user_in_use})"
            )

        if failures:
            return ScenarioResult(name, False, "; ".join(failures), metrics)
        return ScenarioResult(
            name, True,
            f"100 jobs completed in {wall_seconds:.2f}s "
            f"({throughput:.1f} jobs/sec); peak_net={peak_net} peak_user={peak_user}",
            metrics,
        )
    finally:
        if user_id > 0:
            db = SessionLocal()
            try:
                _cleanup_user(db, user_id)
            finally:
                db.close()


async def scenario_b_user_cap() -> ScenarioResult:
    """Scenario B — Per-user slot cap respected (RTMP, slot_cap=5)."""
    name = "B: per-user slot cap"
    user_id = 0
    metrics: dict = {}
    try:
        # ── Setup ──────────────────────────────────────────────────────────
        db = SessionLocal()
        try:
            tier = _ensure_test_tiny_tier(db)  # slot_cap=5, RTMP-only
            user = _create_user(db, "scenB", tier)
            user_id = user.id
            n_channels = 200  # >> cap so the cap is hit
            channels = _create_channels_and_tokens(db, user, n_channels, "scenB")
            master = _create_master_video(db, user, "scenB")
            _seed_credits(db, user, n_channels * 2)  # 400 cr for 200 RTMP jobs
            request = _build_request(
                master.id, channels,
                upload_path="rtmp", publish_kind="short",
                priority="normal", tag="scenB",
            )
            user_loaded = db.query(models.User).filter(
                models.User.id == user.id
            ).first()
            result = fanout_svc.create_publish_task(db, user_loaded, request)
            db.commit()
            pt_id = result.publish_task_id
            job_ids = list(result.upload_job_ids)
        finally:
            db.close()

        if len(job_ids) != n_channels:
            return ScenarioResult(
                name, False,
                f"expected {n_channels} upload_job_ids, got {len(job_ids)}",
                metrics,
            )

        # ── Drain + sample ────────────────────────────────────────────────
        await scheduler.start()
        max_user_in_use = 0
        cap_violations = 0
        sample_count = 0

        t_start = time.monotonic()
        for jid in job_ids:
            scheduler.scheduler_enqueue(
                upload_job_id=jid,
                priority="normal",
                user_id=user_id,
                plan_tier_name="test_tiny",
            )

        deadline = t_start + 120.0
        completed = 0
        while time.monotonic() < deadline:
            snap = scheduler.snapshot()
            sm = snap.get("slot_manager", {})
            cur_user_in_use = int(sm.get("user_in_use", {}).get(user_id, 0) or 0)
            max_user_in_use = max(max_user_in_use, cur_user_in_use)
            if cur_user_in_use > 5:
                cap_violations += 1
            sample_count += 1
            await asyncio.sleep(0.02)
            if sample_count % 25 == 0:
                db = SessionLocal()
                try:
                    completed = db.query(models.UploadJobV2).filter(
                        models.UploadJobV2.publish_task_id == pt_id,
                        models.UploadJobV2.status == "completed",
                    ).count()
                finally:
                    db.close()
                if completed >= n_channels:
                    break
        wall_seconds = time.monotonic() - t_start

        await asyncio.sleep(0.5)
        snap_final = scheduler.snapshot()
        await scheduler.shutdown()

        db = SessionLocal()
        try:
            completed = db.query(models.UploadJobV2).filter(
                models.UploadJobV2.publish_task_id == pt_id,
                models.UploadJobV2.status == "completed",
            ).count()
            failed = db.query(models.UploadJobV2).filter(
                models.UploadJobV2.publish_task_id == pt_id,
                models.UploadJobV2.status == "failed",
            ).count()
        finally:
            db.close()

        metrics.update({
            "wall_seconds": round(wall_seconds, 3),
            "samples_taken": sample_count,
            "max_user_in_use": max_user_in_use,
            "cap_violation_samples": cap_violations,
            "completed_count": completed,
            "failed_count": failed,
            "n_channels": n_channels,
        })

        failures: list[str] = []
        if completed != n_channels:
            failures.append(
                f"expected {n_channels} completed jobs, got {completed}"
            )
        if failed != 0:
            failures.append(f"expected 0 failed jobs, got {failed}")
        if max_user_in_use > 5:
            failures.append(
                f"slot cap violated — max user_in_use={max_user_in_use} > 5 "
                f"({cap_violations} sample(s) above cap)"
            )
        if max_user_in_use < 2:
            # Should at least hit some concurrency with 200 jobs queued.
            failures.append(
                f"suspiciously low max_user_in_use={max_user_in_use} "
                f"(expected to approach 5 with {n_channels} queued)"
            )
        if int(snap_final.get("queue_depth", -1)) != 0:
            failures.append(f"queue_depth != 0 (got {snap_final.get('queue_depth')})")

        if failures:
            return ScenarioResult(name, False, "; ".join(failures), metrics)
        return ScenarioResult(
            name, True,
            f"max user_in_use={max_user_in_use} (cap=5) over {sample_count} "
            f"samples; {completed}/{n_channels} jobs in {wall_seconds:.2f}s",
            metrics,
        )
    finally:
        if user_id > 0:
            db = SessionLocal()
            try:
                _cleanup_user(db, user_id)
            finally:
                db.close()


async def scenario_c_priority() -> ScenarioResult:
    """Scenario C — Priority precedence (critical jumps queued low)."""
    name = "C: priority precedence"
    user_id = 0
    metrics: dict = {}
    try:
        db = SessionLocal()
        try:
            tier = _ensure_pro_tier(db)
            user = _create_user(db, "scenC", tier)
            user_id = user.id
            # 30 low + 5 critical = 35 channels
            channels = _create_channels_and_tokens(db, user, 35, "scenC")
            master = _create_master_video(db, user, "scenC")
            _seed_credits(db, user, 35 * 2)  # 70 cr for 35 RTMP jobs
            low_channels = channels[:30]
            crit_channels = channels[30:]
            low_request = _build_request(
                master.id, low_channels,
                upload_path="rtmp", publish_kind="short",
                priority="low", tag="scenC-low",
            )
            crit_request = _build_request(
                master.id, crit_channels,
                upload_path="rtmp", publish_kind="short",
                priority="critical", tag="scenC-crit",
            )
            user_loaded = db.query(models.User).filter(
                models.User.id == user.id
            ).first()
            low_result = fanout_svc.create_publish_task(db, user_loaded, low_request)
            db.commit()
            crit_result = fanout_svc.create_publish_task(db, user_loaded, crit_request)
            db.commit()
            low_ids = set(low_result.upload_job_ids)
            crit_ids = set(crit_result.upload_job_ids)
        finally:
            db.close()

        await scheduler.start()

        # Submit Round 1 (low) immediately.
        for jid in sorted(low_ids):
            scheduler.scheduler_enqueue(
                upload_job_id=jid,
                priority="low",
                user_id=user_id,
                plan_tier_name="pro",
            )
        # Round 2 (critical) within 100ms.
        await asyncio.sleep(0.05)
        for jid in sorted(crit_ids):
            scheduler.scheduler_enqueue(
                upload_job_id=jid,
                priority="critical",
                user_id=user_id,
                plan_tier_name="pro",
            )

        # Poll every 100ms; record finish_time per job as it completes.
        finish_order: list[tuple[int, str, float]] = []  # (id, tier, finish_secs_from_start)
        seen: set[int] = set()
        t_start = time.monotonic()
        deadline = t_start + 60.0
        target_total = len(low_ids) + len(crit_ids)  # 30 + 5 = 35
        while time.monotonic() < deadline and len(seen) < target_total:
            db = SessionLocal()
            try:
                rows = db.query(models.UploadJobV2).filter(
                    models.UploadJobV2.publish_task_id.in_(
                        [low_result.publish_task_id, crit_result.publish_task_id]
                    ),
                    models.UploadJobV2.status == "completed",
                ).all()
            finally:
                db.close()
            now = time.monotonic()
            for r in rows:
                if r.id in seen:
                    continue
                seen.add(int(r.id))
                tier = "critical" if r.id in crit_ids else "low"
                finish_order.append((int(r.id), tier, now - t_start))
            if len(seen) >= target_total:
                break
            await asyncio.sleep(0.1)

        await asyncio.sleep(0.5)
        await scheduler.shutdown()

        # ── Assert ─────────────────────────────────────────────────────────
        # Sort by finish time and find the *rank* of the last critical job
        # and the count of low jobs that finished after it.
        finish_order.sort(key=lambda t: t[2])

        crit_finished_positions = [
            i for i, (_id, tier, _t) in enumerate(finish_order) if tier == "critical"
        ]
        low_finished_positions = [
            i for i, (_id, tier, _t) in enumerate(finish_order) if tier == "low"
        ]

        # We want: critical jobs were dispatched BEFORE at least 15 of the
        # low (queued) jobs. Concretely — at the moment the last critical
        # finishes, fewer than 15 low jobs have finished AFTER it. So we
        # require: count of low jobs whose finish-rank is AFTER the LAST
        # critical's finish-rank must be >= 15.
        if not crit_finished_positions:
            metrics["finish_order"] = finish_order
            return ScenarioResult(
                name, False,
                "no critical jobs finished — likely scheduler/dispatch failure",
                metrics,
            )
        last_crit_rank = crit_finished_positions[-1]
        first_crit_rank = crit_finished_positions[0]
        low_after_last_crit = sum(
            1 for pos in low_finished_positions if pos > last_crit_rank
        )

        # Build a 35-row table for the report.
        order_table = [
            {"rank": i + 1, "upload_job_id": jid, "tier": tier,
             "finish_seconds": round(t, 3)}
            for i, (jid, tier, t) in enumerate(finish_order)
        ]
        metrics.update({
            "finish_order": order_table,
            "first_critical_finish_rank": first_crit_rank + 1,
            "last_critical_finish_rank": last_crit_rank + 1,
            "low_after_last_critical": low_after_last_crit,
            "total_finished": len(finish_order),
        })

        failures: list[str] = []
        if len(finish_order) != 35:
            failures.append(
                f"expected 35 finishes recorded, got {len(finish_order)}"
            )
        if low_after_last_crit < 15:
            failures.append(
                f"only {low_after_last_crit} low jobs finished AFTER the last "
                f"critical — expected >= 15 (critical jumped queued low jobs?)"
            )

        if failures:
            return ScenarioResult(name, False, "; ".join(failures), metrics)
        return ScenarioResult(
            name, True,
            f"all 5 critical jobs finished by rank {last_crit_rank + 1}; "
            f"{low_after_last_crit} low jobs still after them",
            metrics,
        )
    finally:
        if user_id > 0:
            db = SessionLocal()
            try:
                _cleanup_user(db, user_id)
            finally:
                db.close()


async def scenario_d_legacy() -> ScenarioResult:
    """Scenario D — Legacy + flag-gated routes still alive."""
    name = "D: legacy path intact"
    metrics: dict = {}

    # 1. Legacy POST without auth → 401.
    code_legacy = _http_probe(
        "POST", "/api/clips/1/publish",
        body={"channel_ids": []},
    )
    # 2. New POST without auth (and/or with flag off) → 401 or 503, not 500.
    code_new = _http_probe(
        "POST", "/api/publish-tasks",
        body={
            "master_video_id": 1,
            "targets": [],
            "priority": "normal",
        },
    )
    # 3. Health endpoint → non-500.
    code_health = _http_probe("GET", "/api/health/")

    metrics.update({
        "legacy_clips_publish_unauth": code_legacy,
        "new_publish_tasks_unauth": code_new,
        "health": code_health,
    })

    failures: list[str] = []
    if code_legacy not in (401,):
        failures.append(
            f"legacy POST /api/clips/1/publish: expected 401, got {code_legacy}"
        )
    if code_new not in (401, 503):
        failures.append(
            f"new POST /api/publish-tasks: expected 401 or 503, got {code_new}"
        )
    if code_health < 0 or code_health >= 500:
        failures.append(
            f"GET /api/health/: expected non-5xx, got {code_health}"
        )

    if failures:
        return ScenarioResult(name, False, "; ".join(failures), metrics)
    return ScenarioResult(
        name, True,
        f"legacy={code_legacy} new={code_new} health={code_health}",
        metrics,
    )


# ─── Entry point ────────────────────────────────────────────────────────────


async def _main() -> int:
    print("=" * 78)
    print("Phase 1 Integration Exit-Test")
    print("Brief sec 6: 100 channels -> 100 UploadJobs -> scheduler respects caps + priority")
    print("=" * 78)

    _purge_stale_test_users()

    results: list[ScenarioResult] = []
    for runner in (
        scenario_a_fanout_100,
        scenario_b_user_cap,
        scenario_c_priority,
        scenario_d_legacy,
    ):
        print(f"\n--- {runner.__name__} ---")
        try:
            r = await runner()
        except Exception as exc:
            r = ScenarioResult(runner.__name__, False, f"exception: {exc!r}")
            import traceback
            traceback.print_exc()
        verdict = "PASS" if r.passed else "FAIL"
        print(f"[{verdict}] {r.name}: {r.reason}")
        if r.metrics:
            # Print metrics compact (skip per-row finish_order for terseness).
            terse = {k: v for k, v in r.metrics.items() if k != "finish_order"}
            print(f"      metrics: {terse}")
        results.append(r)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for r in results:
        print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}")
    n_pass = sum(1 for r in results if r.passed)
    n_total = len(results)
    print(f"\n{n_pass}/{n_total} scenarios passed.")

    # Dump JSON snapshot of all results to a side file so the report
    # generator (or a human) can pick up the raw numbers easily.
    snapshot_path = _BACKEND_DIR / "scripts" / "_phase1_exit_last_run.json"
    try:
        payload = [
            {
                "name": r.name,
                "passed": r.passed,
                "reason": r.reason,
                "metrics": r.metrics,
            }
            for r in results
        ]
        snapshot_path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\n(raw results JSON: {snapshot_path})")
    except Exception:
        pass

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    rc = asyncio.run(_main())
    sys.exit(rc)
