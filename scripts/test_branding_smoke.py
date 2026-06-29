"""Phase 2.D Branding smoke test.

End-to-end exercise of ``services.branding.process_upload_job`` against
the real dev Postgres + R2 (whichever STORAGE_BACKEND points at).

Scenario:
    1. Create a test user (Pro tier, has credits).
    2. Create a Channel + OAuthToken for the user.
    3. Generate a tiny test mp4 with ffmpeg (2s red rectangle) and
       upload it to R2 as the MasterVideo bytes.
    4. Generate a tiny test logo png with ffmpeg, upload to R2 as a
       UserAsset; attach it to the OAuthToken (precedence #1).
    5. Set Channel.watermark_text/opacity/position.
    6. Create a PublishTask + UploadJobV2 row.
    7. Call branding.process_upload_job(upload_job_id).
    8. Assert: returns expected cache_key, DB row has the key set,
       status is ready_to_upload, R2 key exists + non-empty.
    9. Call process_upload_job AGAIN; assert it's a cache hit (no new
       ffmpeg invocation, snapshot 'hits' bumped by 1).
   10. Cleanup: delete R2 keys + DB rows + temp files.

Run from KaizerBackend::

    python scripts/test_branding_smoke.py

Exit code 0 on all-pass, 1 on any failure. Re-runnable; cleanup is
idempotent.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# Make ``import models`` / ``import database`` work from the scripts dir.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

logging.basicConfig(
    level=logging.INFO if os.environ.get("KAIZER_BRANDING_VERBOSE") else logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from database import SessionLocal  # noqa: E402
import models  # noqa: E402
from services import branding  # noqa: E402
from services import brand_resolver  # noqa: E402

EMAIL_PREFIX = "branding_smoke_"
EMAIL_DOMAIN = "@kaizer.test"


# ─── Result tracking ───────────────────────────────────────────────────────


@dataclass
class AssertResult:
    name: str
    passed: bool
    detail: str = ""
    seconds: float = 0.0


_results: list[AssertResult] = []


def _assert(name: str, ok: bool, detail: str = "", t0: float | None = None) -> bool:
    seconds = (time.monotonic() - t0) if t0 is not None else 0.0
    _results.append(AssertResult(name=name, passed=bool(ok), detail=detail, seconds=seconds))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}  ({seconds*1000:.1f} ms)  {detail}")
    return bool(ok)


# ─── ffmpeg helpers ────────────────────────────────────────────────────────


def _ffmpeg_bin() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


def _make_test_mp4(out_path: str) -> None:
    """Generate a 2-second 320x180 red mp4 via ffmpeg. ~5-20 KB."""
    cmd = [
        _ffmpeg_bin(), "-y", "-v", "error",
        "-f", "lavfi", "-i", "color=c=red:s=320x180:d=2",
        "-c:v", "libx264", "-crf", "30", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to generate test mp4: {r.stderr[-300:]}")


def _make_test_logo_png(out_path: str) -> None:
    """Generate a 64x64 blue square PNG via ffmpeg."""
    cmd = [
        _ffmpeg_bin(), "-y", "-v", "error",
        "-f", "lavfi", "-i", "color=c=blue:s=64x64:d=1",
        "-frames:v", "1",
        out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to generate test logo: {r.stderr[-300:]}")


# ─── DB / R2 setup ─────────────────────────────────────────────────────────


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


def _provider():
    from pipeline_core.storage import get_storage_provider
    return get_storage_provider()


@dataclass
class TestArtifacts:
    user_id: int = 0
    channel_id: int = 0
    oauth_token_id: int = 0
    user_asset_id: int = 0
    job_id: int = 0
    master_video_id: int = 0
    publish_task_id: int = 0
    upload_job_id: int = 0
    master_r2_key: str = ""
    logo_r2_key: str = ""
    branded_cache_key: str = ""
    work_dir: str = ""
    extra_r2_keys: list[str] = field(default_factory=list)


def _setup(db, art: TestArtifacts) -> None:
    plan = _ensure_pro_tier(db)
    tag = uuid.uuid4().hex[:12]
    art.work_dir = tempfile.mkdtemp(prefix="kaizer_branding_smoke_")

    # ── 1. user ──
    user = models.User(
        email=f"{EMAIL_PREFIX}{tag}{EMAIL_DOMAIN}",
        name="Branding Smoke Test",
        plan="pro",
        plan_tier_id=plan.id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    art.user_id = user.id

    # ── 2. channel ──
    channel = models.Channel(
        user_id=user.id,
        name=f"branding-smoke-{tag}",
        language="te",
        watermark_text="@kaizer.test",
        watermark_opacity=0.35,
        watermark_position="lower-center",
    )
    db.add(channel)
    db.commit()
    db.refresh(channel)
    art.channel_id = channel.id

    # ── 3. OAuth token ──
    tok = models.OAuthToken(
        channel_id=channel.id,
        google_channel_id=f"GCID-{channel.id}",
        google_channel_title=channel.name,
        refresh_token_enc="",
        access_token_enc="",
        scopes="",
    )
    db.add(tok)
    db.commit()
    db.refresh(tok)
    art.oauth_token_id = tok.id

    # ── 4. logo PNG + UserAsset + R2 upload ──
    logo_local = os.path.join(art.work_dir, "logo.png")
    _make_test_logo_png(logo_local)
    logo_r2_key = f"branding_smoke/{tag}/logo.png"
    _provider().upload(logo_local, logo_r2_key, content_type="image/png")
    art.logo_r2_key = logo_r2_key

    user_asset = models.UserAsset(
        user_id=user.id,
        filename="logo.png",
        file_path=logo_local,                     # local copy survives the test for materialize fallback
        kind="logo",
        mime="image/png",
        size_bytes=os.path.getsize(logo_local),
        width=64,
        height=64,
        storage_key=logo_r2_key,
        storage_backend=os.environ.get("STORAGE_BACKEND", "local").lower(),
        storage_url=f"key:{logo_r2_key}",
    )
    db.add(user_asset)
    db.commit()
    db.refresh(user_asset)
    art.user_asset_id = user_asset.id

    # Attach to the OAuthToken (precedence #1).
    tok.logo_asset_id = user_asset.id
    db.add(tok)
    db.commit()

    # ── 5. test master mp4 + R2 upload + MasterVideo row ──
    master_local = os.path.join(art.work_dir, "master.mp4")
    _make_test_mp4(master_local)
    master_r2_key = f"branding_smoke/{tag}/master.mp4"
    _provider().upload(master_local, master_r2_key, content_type="video/mp4")
    art.master_r2_key = master_r2_key

    # Owning Job row (FK: master_videos.source_upload_id → jobs.id).
    job = models.Job(
        user_id=user.id,
        status="done",
        platform="youtube_full",
        video_name=f"branding_smoke_{tag}.mp4",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    art.job_id = job.id

    master = models.MasterVideo(
        source_upload_id=job.id,
        r2_key=master_r2_key,
        duration_seconds=2.0,
        bytes=os.path.getsize(master_local),
        width=320,
        height=180,
        status="ready",
        pipeline_version="v4_clean",
        clean_master=True,
    )
    db.add(master)
    db.commit()
    db.refresh(master)
    art.master_video_id = master.id

    # ── 6. PublishTask + UploadJobV2 ──
    pt = models.PublishTask(
        user_id=user.id,
        master_video_id=master.id,
        priority="normal",
        status="queued",
        target_count=1,
        completed_count=0,
        failed_count=0,
    )
    db.add(pt)
    db.commit()
    db.refresh(pt)
    art.publish_task_id = pt.id

    idem = f"smoke-{tag}-{uuid.uuid4().hex[:16]}"
    pub_v = f"smoke:v1:{tag[:6]}"
    job_v2 = models.UploadJobV2(
        publish_task_id=pt.id,
        channel_id=channel.id,
        oauth_token_id=tok.id,
        upload_path="direct",
        publish_kind="video",
        thumbnail_source="pipeline_generated",
        status="queued",
        attempts=0,
        idempotency_key=idem,
        publish_version=pub_v,
        predicted_quota_units=1650,
        predicted_credit_cost=16,
        bytes_uploaded=0,
    )
    db.add(job_v2)
    db.commit()
    db.refresh(job_v2)
    art.upload_job_id = job_v2.id


# ─── Cleanup ───────────────────────────────────────────────────────────────


def _cleanup(db, art: TestArtifacts) -> None:
    # ── R2 keys ──
    provider = _provider()
    for k in [art.master_r2_key, art.logo_r2_key, art.branded_cache_key, *art.extra_r2_keys]:
        if not k:
            continue
        try:
            provider.delete(k)
        except Exception as exc:
            print(f"[cleanup] R2 delete failed for {k!r}: {exc}")

    # ── DB rows (FK-safe order) ──
    try:
        if art.upload_job_id:
            db.query(models.UploadJobV2).filter(
                models.UploadJobV2.id == art.upload_job_id
            ).delete(synchronize_session=False)
        if art.publish_task_id:
            db.query(models.PublishTask).filter(
                models.PublishTask.id == art.publish_task_id
            ).delete(synchronize_session=False)
        if art.master_video_id:
            db.query(models.MasterVideo).filter(
                models.MasterVideo.id == art.master_video_id
            ).delete(synchronize_session=False)
        if art.job_id:
            db.query(models.Job).filter(
                models.Job.id == art.job_id
            ).delete(synchronize_session=False)
        if art.user_asset_id:
            db.query(models.UserAsset).filter(
                models.UserAsset.id == art.user_asset_id
            ).delete(synchronize_session=False)
        if art.oauth_token_id:
            db.query(models.OAuthToken).filter(
                models.OAuthToken.id == art.oauth_token_id
            ).delete(synchronize_session=False)
        if art.channel_id:
            db.query(models.Channel).filter(
                models.Channel.id == art.channel_id
            ).delete(synchronize_session=False)
        if art.user_id:
            # Also clear any credit ledger rows (none expected, belt+braces).
            db.query(models.CreditLedger).filter(
                models.CreditLedger.user_id == art.user_id
            ).delete(synchronize_session=False)
            db.query(models.User).filter(
                models.User.id == art.user_id
            ).delete(synchronize_session=False)
        db.commit()
    except Exception as exc:
        print(f"[cleanup] DB delete failed: {exc}")
        db.rollback()

    # ── Temp work dir ──
    if art.work_dir and os.path.isdir(art.work_dir):
        try:
            shutil.rmtree(art.work_dir, ignore_errors=True)
        except Exception:
            pass


def _purge_stale() -> None:
    """Belt-and-braces: wipe any leftover smoke-test users from prior
    failed runs."""
    db = SessionLocal()
    try:
        users = db.query(models.User).filter(
            models.User.email.like(f"{EMAIL_PREFIX}%{EMAIL_DOMAIN}")
        ).all()
        for u in users:
            art = TestArtifacts(user_id=u.id)
            # Collect descendants
            chs = db.query(models.Channel).filter(
                models.Channel.user_id == u.id
            ).all()
            for ch in chs:
                art.channel_id = ch.id  # ok to overwrite — _cleanup handles one at a time
                tok = db.query(models.OAuthToken).filter(
                    models.OAuthToken.channel_id == ch.id
                ).first()
                if tok:
                    art.oauth_token_id = tok.id
                pts = db.query(models.PublishTask).filter(
                    models.PublishTask.user_id == u.id
                ).all()
                for pt in pts:
                    art.publish_task_id = pt.id
                    jbs = db.query(models.UploadJobV2).filter(
                        models.UploadJobV2.publish_task_id == pt.id
                    ).all()
                    for jb in jbs:
                        art.upload_job_id = jb.id
                        _cleanup(db, art)
                        art.upload_job_id = 0
                    art.publish_task_id = 0
            mvs = db.query(models.MasterVideo).join(
                models.Job, models.MasterVideo.source_upload_id == models.Job.id
            ).filter(models.Job.user_id == u.id).all()
            for mv in mvs:
                art.master_video_id = mv.id
                _cleanup(db, art)
                art.master_video_id = 0
            jobs = db.query(models.Job).filter(models.Job.user_id == u.id).all()
            for j in jobs:
                art.job_id = j.id
                _cleanup(db, art)
                art.job_id = 0
            assets = db.query(models.UserAsset).filter(
                models.UserAsset.user_id == u.id
            ).all()
            for a in assets:
                art.user_asset_id = a.id
                _cleanup(db, art)
                art.user_asset_id = 0
            for ch in chs:
                art.channel_id = ch.id
                _cleanup(db, art)
                art.channel_id = 0
            _cleanup(db, art)
    finally:
        db.close()


# ─── Concurrent miss-storm test ────────────────────────────────────────────
#
# Regression coverage for the Phase 2 exit-test bug: when N concurrent jobs
# share the same brand_profile_version, naive cache-check + ffmpeg + R2
# write all race each other (4 misses + 0 hits observed instead of 1 + 3).
# The fix is a per-cache-key advisory lock that serializes the miss path so
# the loser threads observe the cache hit under the lock.


@dataclass
class _ConcurrentArtifacts:
    """Setup for the concurrent test — 4 UploadJobV2 rows sharing one
    brand-profile-version (one user, one channel, one master, one logo)."""
    user_id: int = 0
    channel_id: int = 0
    oauth_token_id: int = 0
    user_asset_id: int = 0
    job_id: int = 0
    master_video_id: int = 0
    publish_task_id: int = 0
    upload_job_ids: list[int] = field(default_factory=list)
    master_r2_key: str = ""
    logo_r2_key: str = ""
    branded_cache_key: str = ""
    work_dir: str = ""


def _setup_concurrent(db, art: _ConcurrentArtifacts, n_jobs: int = 4) -> None:
    plan = _ensure_pro_tier(db)
    tag = uuid.uuid4().hex[:12]
    art.work_dir = tempfile.mkdtemp(prefix="kaizer_branding_concurrent_")

    user = models.User(
        email=f"{EMAIL_PREFIX}conc_{tag}{EMAIL_DOMAIN}",
        name="Branding Concurrent Test",
        plan="pro",
        plan_tier_id=plan.id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    art.user_id = user.id

    channel = models.Channel(
        user_id=user.id,
        name=f"branding-concurrent-{tag}",
        language="te",
        watermark_text="@kaizer.test.conc",
        watermark_opacity=0.35,
        watermark_position="lower-center",
    )
    db.add(channel)
    db.commit()
    db.refresh(channel)
    art.channel_id = channel.id

    tok = models.OAuthToken(
        channel_id=channel.id,
        google_channel_id=f"GCID-{channel.id}",
        google_channel_title=channel.name,
        refresh_token_enc="",
        access_token_enc="",
        scopes="",
    )
    db.add(tok)
    db.commit()
    db.refresh(tok)
    art.oauth_token_id = tok.id

    logo_local = os.path.join(art.work_dir, "logo.png")
    _make_test_logo_png(logo_local)
    logo_r2_key = f"branding_smoke/conc_{tag}/logo.png"
    _provider().upload(logo_local, logo_r2_key, content_type="image/png")
    art.logo_r2_key = logo_r2_key

    user_asset = models.UserAsset(
        user_id=user.id,
        filename="logo.png",
        file_path=logo_local,
        kind="logo",
        mime="image/png",
        size_bytes=os.path.getsize(logo_local),
        width=64,
        height=64,
        storage_key=logo_r2_key,
        storage_backend=os.environ.get("STORAGE_BACKEND", "local").lower(),
        storage_url=f"key:{logo_r2_key}",
    )
    db.add(user_asset)
    db.commit()
    db.refresh(user_asset)
    art.user_asset_id = user_asset.id
    tok.logo_asset_id = user_asset.id
    db.add(tok)
    db.commit()

    master_local = os.path.join(art.work_dir, "master.mp4")
    _make_test_mp4(master_local)
    master_r2_key = f"branding_smoke/conc_{tag}/master.mp4"
    _provider().upload(master_local, master_r2_key, content_type="video/mp4")
    art.master_r2_key = master_r2_key

    job = models.Job(
        user_id=user.id,
        status="done",
        platform="youtube_full",
        video_name=f"branding_concurrent_{tag}.mp4",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    art.job_id = job.id

    master = models.MasterVideo(
        source_upload_id=job.id,
        r2_key=master_r2_key,
        duration_seconds=2.0,
        bytes=os.path.getsize(master_local),
        width=320,
        height=180,
        status="ready",
        pipeline_version="v4_clean",
        clean_master=True,
    )
    db.add(master)
    db.commit()
    db.refresh(master)
    art.master_video_id = master.id

    pt = models.PublishTask(
        user_id=user.id,
        master_video_id=master.id,
        priority="normal",
        status="queued",
        target_count=n_jobs,
        completed_count=0,
        failed_count=0,
    )
    db.add(pt)
    db.commit()
    db.refresh(pt)
    art.publish_task_id = pt.id

    for i in range(n_jobs):
        idem = f"smoke-conc-{tag}-{i:02d}-{uuid.uuid4().hex[:8]}"
        pub_v = f"smoke:conc:v1:{tag[:6]}:{i:02d}"
        job_v2 = models.UploadJobV2(
            publish_task_id=pt.id,
            channel_id=channel.id,
            oauth_token_id=tok.id,
            upload_path="direct",
            publish_kind="video",
            thumbnail_source="pipeline_generated",
            status="queued",
            attempts=0,
            idempotency_key=idem,
            publish_version=pub_v,
            predicted_quota_units=1650,
            predicted_credit_cost=16,
            bytes_uploaded=0,
        )
        db.add(job_v2)
        db.commit()
        db.refresh(job_v2)
        art.upload_job_ids.append(job_v2.id)


def _cleanup_concurrent(db, art: _ConcurrentArtifacts) -> None:
    provider = _provider()
    for k in [art.master_r2_key, art.logo_r2_key, art.branded_cache_key]:
        if not k:
            continue
        try:
            provider.delete(k)
        except Exception as exc:
            print(f"[cleanup_conc] R2 delete failed for {k!r}: {exc}")

    try:
        if art.upload_job_ids:
            db.query(models.UploadJobV2).filter(
                models.UploadJobV2.id.in_(art.upload_job_ids)
            ).delete(synchronize_session=False)
        if art.publish_task_id:
            db.query(models.PublishTask).filter(
                models.PublishTask.id == art.publish_task_id
            ).delete(synchronize_session=False)
        if art.master_video_id:
            db.query(models.MasterVideo).filter(
                models.MasterVideo.id == art.master_video_id
            ).delete(synchronize_session=False)
        if art.job_id:
            db.query(models.Job).filter(
                models.Job.id == art.job_id
            ).delete(synchronize_session=False)
        if art.user_asset_id:
            db.query(models.UserAsset).filter(
                models.UserAsset.id == art.user_asset_id
            ).delete(synchronize_session=False)
        if art.oauth_token_id:
            db.query(models.OAuthToken).filter(
                models.OAuthToken.id == art.oauth_token_id
            ).delete(synchronize_session=False)
        if art.channel_id:
            db.query(models.Channel).filter(
                models.Channel.id == art.channel_id
            ).delete(synchronize_session=False)
        if art.user_id:
            db.query(models.CreditLedger).filter(
                models.CreditLedger.user_id == art.user_id
            ).delete(synchronize_session=False)
            db.query(models.User).filter(
                models.User.id == art.user_id
            ).delete(synchronize_session=False)
        db.commit()
    except Exception as exc:
        print(f"[cleanup_conc] DB delete failed: {exc}")
        db.rollback()

    if art.work_dir and os.path.isdir(art.work_dir):
        try:
            shutil.rmtree(art.work_dir, ignore_errors=True)
        except Exception:
            pass


def _run_concurrent_test() -> bool:
    """4 concurrent threads call process_upload_job on 4 different
    UploadJobV2 rows that all resolve to the SAME brand_profile_version
    → same cache key. Expect exactly 1 miss + 3 hits and exactly 1
    ffmpeg invocation."""
    print("\n── Concurrent miss-storm regression test ──")
    db = SessionLocal()
    art = _ConcurrentArtifacts()
    n_jobs = 4
    ok = False
    try:
        t0 = time.monotonic()
        try:
            _setup_concurrent(db, art, n_jobs=n_jobs)
        except Exception as exc:
            _assert("concurrent_setup", False, f"raised: {exc!r}", t0=t0)
            raise
        _assert(
            "concurrent_setup", True,
            f"user_id={art.user_id} upload_job_ids={art.upload_job_ids}",
            t0=t0,
        )

        # Pre-clear the cache key from R2 in case a previous failed run
        # left an artifact behind (the cache key is deterministic per
        # master_video_id + brand_profile_version, but our master_video
        # id is fresh so collisions are astronomically unlikely).
        provider = _provider()
        resolved = brand_resolver.resolve_brand_profile(db, art.channel_id)
        expected_key = branding.brand_artifact_cache_key(
            art.master_video_id, resolved.version,
        )
        try:
            provider.delete(expected_key)
        except Exception:
            pass
        art.branded_cache_key = expected_key

        snap_before = branding.snapshot()
        misses_before = int(snap_before.get("misses", 0))
        hits_before = int(snap_before.get("hits", 0))
        ffmpeg_before = int(snap_before.get("ffmpeg_invocations", 0))

        # Close the setup session before spawning workers — each worker
        # opens its own SessionLocal() inside process_upload_job and we
        # do NOT want shared mutable session state across threads.
        db.close()

        # Start barrier so all 4 threads hit the cache check window
        # simultaneously — maximises the probability of catching the
        # race in the (broken) pre-fix code path.
        barrier = threading.Barrier(n_jobs)

        def _worker(job_id: int) -> tuple[int, str, float, Exception | None]:
            barrier.wait()
            t_start = time.monotonic()
            try:
                key = branding.process_upload_job(job_id)
                return (job_id, key, time.monotonic() - t_start, None)
            except Exception as exc:
                return (job_id, "", time.monotonic() - t_start, exc)

        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(_worker, jid) for jid in art.upload_job_ids]
            results = [f.result(timeout=180) for f in futures]
        wall = time.monotonic() - t0

        # Check all returned the same key, no exceptions.
        errs = [(jid, exc) for (jid, _k, _d, exc) in results if exc is not None]
        if errs:
            _assert(
                "concurrent_no_exceptions", False,
                f"errors: {errs!r}", t0=t0,
            )
            return False
        _assert(
            "concurrent_no_exceptions", True,
            f"{n_jobs} workers returned cleanly in {wall:.2f}s",
        )

        returned_keys = {k for (_jid, k, _d, _e) in results}
        _assert(
            "concurrent_all_same_cache_key",
            len(returned_keys) == 1 and expected_key in returned_keys,
            f"returned_keys={returned_keys!r} expected={expected_key!r}",
        )

        # Snapshot delta — THE key assertion: 1 miss + 3 hits.
        snap_after = branding.snapshot()
        misses_delta = int(snap_after.get("misses", 0)) - misses_before
        hits_delta = int(snap_after.get("hits", 0)) - hits_before
        ffmpeg_delta = int(snap_after.get("ffmpeg_invocations", 0)) - ffmpeg_before

        _assert(
            "concurrent_exactly_one_miss",
            misses_delta == 1,
            f"misses_delta={misses_delta} (expected 1; pre-fix bug observed 4)",
        )
        _assert(
            "concurrent_exactly_three_hits",
            hits_delta == (n_jobs - 1),
            f"hits_delta={hits_delta} (expected {n_jobs - 1}; pre-fix bug observed 0)",
        )
        _assert(
            "concurrent_exactly_one_ffmpeg_invocation",
            ffmpeg_delta == 1,
            f"ffmpeg_delta={ffmpeg_delta} (expected 1 — only one worker should "
            f"run the overlay pass)",
        )

        # The R2 key was written exactly once. We can't directly count
        # writes through the provider, but we can prove ONE artifact
        # exists and is non-empty (an idempotent re-write would also
        # satisfy this, but the ffmpeg_invocations==1 assertion above is
        # the tighter "wrote once" signal).
        head_ok = False
        size = 0
        try:
            head_ok = bool(provider.exists(expected_key))
            tmp_dl = os.path.join(art.work_dir, "_verify_conc.mp4")
            provider.download(expected_key, tmp_dl)
            size = os.path.getsize(tmp_dl) if os.path.isfile(tmp_dl) else 0
        except Exception as exc:
            _assert(
                "concurrent_artifact_in_r2", False,
                f"download failed: {exc!r}",
            )
            raise
        _assert(
            "concurrent_artifact_in_r2",
            head_ok and size > 0,
            f"head_ok={head_ok} size={size}",
        )

        # All 4 UploadJobV2 rows should now have the cache key persisted
        # and status='ready_to_upload'.
        db2 = SessionLocal()
        try:
            rows = db2.query(models.UploadJobV2).filter(
                models.UploadJobV2.id.in_(art.upload_job_ids)
            ).all()
            persisted_keys = {r.branded_artifact_r2_key for r in rows}
            statuses = {r.status for r in rows}
        finally:
            db2.close()
        _assert(
            "concurrent_all_rows_have_persisted_cache_key",
            persisted_keys == {expected_key},
            f"persisted_keys={persisted_keys!r}",
        )
        _assert(
            "concurrent_all_rows_status_ready_to_upload",
            statuses == {"ready_to_upload"},
            f"statuses={statuses!r}",
        )

        print(
            f"\n  [concurrent stats] wall={wall:.2f}s "
            f"misses={misses_delta} hits={hits_delta} ffmpeg={ffmpeg_delta} "
            f"per-worker durations={[round(d, 2) for (_j,_k,d,_e) in results]}"
        )
        ok = True
    finally:
        db_c = SessionLocal()
        try:
            _cleanup_concurrent(db_c, art)
        except Exception as exc:
            print(f"[cleanup_conc] non-fatal: {exc}")
        finally:
            db_c.close()
    return ok


# ─── Main ──────────────────────────────────────────────────────────────────


def main() -> int:
    print("\n── Phase 2.D Branding smoke test ──")
    print(f"STORAGE_BACKEND={os.environ.get('STORAGE_BACKEND', 'local')!r} "
          f"R2_BUCKET={os.environ.get('R2_BUCKET', '<unset>')!r}")
    print(f"ffmpeg={_ffmpeg_bin()!r}\n")

    print("[setup] purging stale smoke-test users from prior runs")
    _purge_stale()

    db = SessionLocal()
    art = TestArtifacts()
    overall_ok = False
    try:
        # ── Setup ──
        t0 = time.monotonic()
        try:
            _setup(db, art)
        except Exception as exc:
            _assert("setup", False, f"setup raised: {exc!r}", t0=t0)
            raise
        _assert("setup", True, f"user_id={art.user_id} upload_job_id={art.upload_job_id}", t0=t0)

        # ── 1st call (cache miss expected) ──
        snap_before = branding.snapshot()
        t0 = time.monotonic()
        cache_key_1 = ""
        try:
            cache_key_1 = branding.process_upload_job(art.upload_job_id)
        except Exception as exc:
            _assert("process_upload_job_1st_call", False, f"raised: {exc!r}", t0=t0)
            raise
        art.branded_cache_key = cache_key_1
        _assert(
            "process_upload_job_1st_call_returns_key",
            bool(cache_key_1) and cache_key_1.startswith("branded/"),
            f"got {cache_key_1!r}", t0=t0,
        )

        # Validate the formula directly.
        resolved = brand_resolver.resolve_brand_profile(db, art.channel_id)
        expected = branding.brand_artifact_cache_key(
            art.master_video_id, resolved.version,
        )
        _assert(
            "cache_key_matches_formula",
            cache_key_1 == expected,
            f"expected={expected!r} got={cache_key_1!r}",
        )
        _assert(
            "brand_resolver_picked_oauth_token_logo",
            resolved.logo_asset_id == art.user_asset_id,
            f"resolved.logo_asset_id={resolved.logo_asset_id} expected={art.user_asset_id}",
        )

        # Re-read the row to assert persisted state.
        db.expire_all()
        job_row = db.query(models.UploadJobV2).filter(
            models.UploadJobV2.id == art.upload_job_id
        ).first()
        _assert(
            "upload_job_branded_artifact_r2_key_set",
            job_row is not None and job_row.branded_artifact_r2_key == cache_key_1,
            f"row.branded_artifact_r2_key={getattr(job_row, 'branded_artifact_r2_key', None)!r}",
        )
        _assert(
            "upload_job_status_ready_to_upload",
            job_row is not None and job_row.status == "ready_to_upload",
            f"row.status={getattr(job_row, 'status', None)!r}",
        )

        # Verify the R2 key actually exists + non-empty (head + bytes check).
        provider = _provider()
        head_ok = False
        size = 0
        try:
            head_ok = bool(provider.exists(cache_key_1))
            tmp_dl = os.path.join(art.work_dir, "_verify.mp4")
            provider.download(cache_key_1, tmp_dl)
            size = os.path.getsize(tmp_dl) if os.path.isfile(tmp_dl) else 0
        except Exception as exc:
            _assert("branded_artifact_in_r2", False, f"download failed: {exc!r}")
            raise
        _assert(
            "branded_artifact_in_r2",
            head_ok and size > 0,
            f"head_ok={head_ok} size={size}",
        )

        snap_after_miss = branding.snapshot()
        misses_delta = snap_after_miss["misses"] - snap_before.get("misses", 0)
        hits_delta_first = snap_after_miss["hits"] - snap_before.get("hits", 0)
        _assert(
            "first_call_was_cache_miss",
            misses_delta == 1 and hits_delta_first == 0,
            f"miss_delta={misses_delta} hits_delta={hits_delta_first}",
        )

        # ── 2nd call (cache hit expected) ──
        t0 = time.monotonic()
        try:
            cache_key_2 = branding.process_upload_job(art.upload_job_id)
        except Exception as exc:
            _assert("process_upload_job_2nd_call", False, f"raised: {exc!r}", t0=t0)
            raise
        _assert(
            "2nd_call_same_key",
            cache_key_2 == cache_key_1,
            f"second={cache_key_2!r} first={cache_key_1!r}", t0=t0,
        )

        snap_after_hit = branding.snapshot()
        hits_delta = snap_after_hit["hits"] - snap_after_miss["hits"]
        misses_delta_second = snap_after_hit["misses"] - snap_after_miss["misses"]
        ffmpeg_delta = snap_after_hit["ffmpeg_invocations"] - snap_after_miss["ffmpeg_invocations"]
        _assert(
            "2nd_call_was_cache_hit",
            hits_delta == 1 and misses_delta_second == 0 and ffmpeg_delta == 0,
            f"hits_delta={hits_delta} misses_delta={misses_delta_second} "
            f"ffmpeg_delta={ffmpeg_delta}",
        )

        # ── Concurrent miss-storm regression ──
        # Cleanup the sequential test before spawning the concurrent one
        # so the two scenarios don't share rows / cached artifacts.
        try:
            _cleanup(db, art)
        except Exception as exc:
            print(f"[cleanup] sequential non-fatal: {exc}")
        try:
            db.close()
        except Exception:
            pass

        try:
            _run_concurrent_test()
        except Exception as exc:
            print(f"[concurrent] crashed: {exc!r}")
            import traceback
            traceback.print_exc()

        # ── Summary ──
        total = len(_results)
        passes = sum(1 for r in _results if r.passed)
        final_snap = branding.snapshot()
        print(f"\n── Result: {passes}/{total} assertions PASS ──")
        print(f"Total ffmpeg invocations across test: {final_snap['ffmpeg_invocations']}")
        print(f"ffmpeg duration last: {final_snap['last_ffmpeg_duration_seconds']}s")

        overall_ok = (passes == total)
    finally:
        # Sequential cleanup already happened mid-flow; this is belt-and-braces.
        try:
            _cleanup(db, art)
        except Exception as exc:
            print(f"[cleanup] non-fatal: {exc}")
        try:
            db.close()
        except Exception:
            pass

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
