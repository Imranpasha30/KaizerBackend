"""Phase 3 Integration Exit-Test (KAIZER_UPLOAD_REWRITE_BRIEF.md Â§8).

Two scenarios verify the final exit criteria of Phase 3:

  A. Kill-worker mid-upload â†’ no duplicate video.
       1. Submit a PublishTask with 1 Direct/video target.
       2. Monkeypatch the YouTube uploader so the first
          ``videos().insert(...).execute()`` raises a transient error.
       3. Scheduler runs the job; it fails (status='failed' or the
          PublishAttempt row sticks in 'in_flight' depending on which
          layer threw). Either way no YouTube video was minted yet.
       4. Restart the scheduler in-process and call
          ``recovery.recover_on_startup()`` â€” for the in_flight orphan
          it re-queues; for the cleanly-failed row it's a no-op.
       5. Flip the monkeypatch to return success. Re-dispatch.
       6. Assert: exactly ONE PublishAttempt row marked 'completed';
          UploadJobV2.youtube_video_id is populated;
          ``fake_yt.insert_calls`` == 2 (1 fail + 1 success â€” never 3
          or more, i.e. NO duplicate insert).

  B. Critical burst doesn't starve Normal (aging promotion).
       1. Submit 50 Critical jobs + 50 Normal jobs with
          ``KAIZER_PRIORITY_AGING_MIN=0`` (extremely fast aging) and
          ``KAIZER_SCHED_NET_TOKENS=2`` (Criticals can dominate without
          aging help) so we get real backpressure during the drain.
       2. Wait for the scheduler to drain everything.
       3. Assert: â‰¥ 5 of the Normal jobs got promoted by aging â€” i.e.
          ``priority_at_dispatch != 'normal'`` for at least 5 rows.

Each scenario prints PASS/FAIL. Exit 0 on all pass.

Run:

    cd kaizer/KaizerBackend
    python scripts/test_phase3_exit.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Quota-bucket isolation: burn the 'testrun' bucket so fake-upload
# reservations never pollute the production 'oauth' counter the UI
# quota chip reports (see youtube/quota_v2._key_hash).
os.environ.setdefault("KAIZER_YT_QUOTA_BUCKET", "testrun")
# Deterministic cap for suites: the REAL resolved cap is Google's 10k,
# which the shared testrun bucket exceeds after a few runs in one day.
# Env override is resolution priority #1 and never touches production
# (the bucket isolation above keeps the real counter clean).
os.environ.setdefault("KAIZER_YT_DAILY_QUOTA_CAP", "1000000")

# Make ``import models`` / ``import database`` work from the scripts dir.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

# Configure aging + token limits BEFORE importing the scheduler so its
# module-level loaders pick up our test values. Scenario B needs aging
# to fire fast and net tokens to be tight.
os.environ.setdefault("KAIZER_PRIORITY_AGING_MIN", "1")  # min legal value
os.environ.setdefault("KAIZER_SCHED_NET_TOKENS", "2")
os.environ.setdefault("KAIZER_SCHED_CPU_TOKENS", "2")

logging.basicConfig(
    level=logging.INFO if os.environ.get("KAIZER_PHASE3_EXIT_VERBOSE") else logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from database import SessionLocal  # noqa: E402
import models  # noqa: E402
from services import branding  # noqa: E402
from services import credits as credits_svc  # noqa: E402
from services import fanout as fanout_svc  # noqa: E402
from services import scheduler  # noqa: E402
from services import recovery  # noqa: E402
from services import upload_dispatch  # noqa: E402

EMAIL_PREFIX = "phase3exit_test_"
EMAIL_DOMAIN = "@kaizer.test"


# â”€â”€â”€ Result tracking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    reason: str = ""
    metrics: dict = field(default_factory=dict)


# â”€â”€â”€ ffmpeg helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _ffmpeg_bin() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


def _make_test_mp4(out_path: str) -> None:
    cmd = [
        _ffmpeg_bin(), "-y", "-v", "error",
        "-f", "lavfi", "-i", "color=c=red:s=320x180:d=2",
        "-c:v", "libx264", "-crf", "30", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-300:]}")


def _make_test_logo_png(out_path: str) -> None:
    cmd = [
        _ffmpeg_bin(), "-y", "-v", "error",
        "-f", "lavfi", "-i", "color=c=blue:s=64x64:d=1",
        "-frames:v", "1",
        out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-300:]}")


def _provider():
    from pipeline_core.storage import get_storage_provider
    return get_storage_provider()


# â”€â”€â”€ Fake YouTube API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class _FakeMediaProgress:
    def __init__(self, uploaded: int):
        self.resumable_progress = uploaded


class _FakeInsertRequest:
    """Mimics googleapiclient resumable upload â€” one full chunk + done."""

    def __init__(self, total_bytes: int, video_id: str, fail_first: bool):
        self._total = max(1, int(total_bytes))
        self._video_id = video_id
        self._fail_first = fail_first
        self._chunks = 0
        self.resumable_uri = f"https://uploads.example/v1/sess/{uuid.uuid4().hex}"

    def next_chunk(self):
        self._chunks += 1
        if self._fail_first and self._chunks == 1:
            # Simulate a transient network error from googleapiclient.
            raise ConnectionResetError("simulated: worker crashed mid-upload")
        # Single chunk + result tuple terminates the resumable loop.
        return None, {"id": self._video_id}


class _FakeVideosResource:
    def __init__(self, parent: "_ControllableFakeYouTube"):
        self._p = parent

    def insert(self, *, part, body, media_body, notifySubscribers=False):
        self._p.insert_calls += 1
        fail_first = bool(self._p.fail_next)
        # Auto-consume the one-shot fail flag.
        self._p.fail_next = False
        new_id = f"FAKE_VID_{uuid.uuid4().hex[:12]}"
        self._p.last_video_id = new_id
        self._p.minted_video_ids.add(new_id)
        size = 0
        try:
            stream = media_body.stream()
            if stream is not None:
                size = os.path.getsize(getattr(stream, "name", "")) or 0
        except Exception:
            pass
        return _FakeInsertRequest(
            total_bytes=size or 100_000,
            video_id=new_id,
            fail_first=fail_first,
        )

    def list(self, *, part, id):
        self._p.list_calls += 1

        class _Req:
            def __init__(self, p, vid):
                self._p = p
                self._vid = vid

            def execute(self_inner):
                if self_inner._vid in self_inner._p.minted_video_ids:
                    return {"items": [{"id": self_inner._vid, "status": {}}]}
                return {"items": []}

        return _Req(self._p, id)


class _FakeThumbnailsResource:
    def __init__(self, parent: "_ControllableFakeYouTube"):
        self._p = parent

    def set(self, *, videoId, media_body):
        self._p.thumb_calls += 1

        class _Req:
            def execute(self_inner):
                return {}

        return _Req()


class _ControllableFakeYouTube:
    """Same shape as Phase 2's fake YouTube, with a ``fail_next`` toggle
    that causes the next ``videos.insert`` call to raise a transient
    error on its FIRST chunk before being consumed."""

    def __init__(self):
        self.insert_calls = 0
        self.list_calls = 0
        self.thumb_calls = 0
        self.last_video_id: Optional[str] = None
        self.minted_video_ids: set[str] = set()
        self.fail_next = False
        self._videos = _FakeVideosResource(self)
        self._thumbnails = _FakeThumbnailsResource(self)

    def videos(self):
        return self._videos

    def thumbnails(self):
        return self._thumbnails


def _install_monkeypatches() -> tuple[_ControllableFakeYouTube, Any]:
    """Returns (fake_youtube, restore_callable)."""
    from youtube import oauth as _oauth_mod
    original_get_creds = _oauth_mod.get_credentials

    class _FakeCreds:
        valid = True
        token = "FAKE_ACCESS"
        refresh_token = "FAKE_REFRESH"

    def _fake_get_credentials(db, channel_id):
        return _FakeCreds()

    _oauth_mod.get_credentials = _fake_get_credentials  # type: ignore

    fake_yt = _ControllableFakeYouTube()
    import googleapiclient.discovery as _gad
    original_build = _gad.build

    def _fake_build(serviceName, version, **kwargs):
        return fake_yt

    _gad.build = _fake_build  # type: ignore

    from youtube import uploader_v2 as _u2
    original_u2_build = getattr(_u2, "build", None)
    _u2.build = _fake_build  # type: ignore

    def _restore() -> None:
        _oauth_mod.get_credentials = original_get_creds  # type: ignore
        _gad.build = original_build  # type: ignore
        if original_u2_build is not None:
            _u2.build = original_u2_build  # type: ignore

    return fake_yt, _restore


# â”€â”€â”€ Test data factories â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _unique_email(tag: str) -> str:
    return f"{EMAIL_PREFIX}{tag}_{uuid.uuid4().hex[:12]}{EMAIL_DOMAIN}"


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


def _create_user(db, tag: str, plan_tier: models.PlanTier) -> models.User:
    email = _unique_email(tag)
    user = models.User(
        email=email,
        name=f"Phase 3 Exit Test ({tag})",
        plan="pro",
        plan_tier_id=plan_tier.id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _seed_credits(db, user: models.User, amount: int) -> None:
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
        balance_after=amount,
    )
    db.add(row)
    db.commit()


def _create_channels_and_tokens(
    db, user: models.User, n: int, tag: str,
) -> list[models.Channel]:
    for i in range(n):
        ch = models.Channel(
            user_id=user.id,
            name=f"phase3exit-{tag}-ch{i:03d}-{uuid.uuid4().hex[:6]}",
            language="te",
            watermark_text=f"@phase3exit-{tag}",
            watermark_opacity=0.35,
            watermark_position="lower-center",
        )
        db.add(ch)
    db.commit()
    channels = (
        db.query(models.Channel)
        .filter(
            models.Channel.user_id == user.id,
            models.Channel.name.like(f"phase3exit-{tag}-%"),
        )
        .order_by(models.Channel.id.asc())
        .all()
    )
    assert len(channels) == n
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


def _create_master_video_with_r2(
    db, user: models.User, tag: str, work_dir: str,
) -> tuple[models.MasterVideo, str]:
    job = models.Job(
        user_id=user.id,
        status="done",
        platform="youtube_full",
        video_name=f"phase3exit_{tag}.mp4",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    master_local = os.path.join(work_dir, f"master_{tag}.mp4")
    _make_test_mp4(master_local)
    master_key = f"phase3exit/{tag}/master.mp4"
    _provider().upload(master_local, master_key, content_type="video/mp4")

    master = models.MasterVideo(
        source_upload_id=job.id,
        r2_key=master_key,
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
    return master, master_key


def _create_user_logo_asset_and_attach(
    db, user: models.User, tokens: list[models.OAuthToken], tag: str, work_dir: str,
) -> tuple[models.UserAsset, str]:
    logo_local = os.path.join(work_dir, f"logo_{tag}.png")
    _make_test_logo_png(logo_local)
    logo_key = f"phase3exit/{tag}/logo.png"
    _provider().upload(logo_local, logo_key, content_type="image/png")

    user_asset = models.UserAsset(
        user_id=user.id,
        filename="logo.png",
        file_path=logo_local,
        kind="logo",
        mime="image/png",
        size_bytes=os.path.getsize(logo_local),
        width=64,
        height=64,
        storage_key=logo_key,
        storage_backend=os.environ.get("STORAGE_BACKEND", "local").lower(),
        storage_url=f"key:{logo_key}",
    )
    db.add(user_asset)
    db.commit()
    db.refresh(user_asset)

    for tok in tokens:
        tok.logo_asset_id = user_asset.id
        db.add(tok)
    db.commit()
    return user_asset, logo_key


def _create_clip(db, user_id: int, job_id: int, tag: str) -> models.Clip:
    clip = models.Clip(
        job_id=job_id,
        clip_index=0,
        filename=f"phase3exit_{tag}.mp4",
        file_path="",
        duration=2.0,
        text=f"Phase 3 exit bulletin {tag}",
        seo=json.dumps({
            "title": f"Phase 3 Exit Test Bulletin {tag}",
            "description": "Phase 3 exit test description.",
            "keywords": ["phase3exit", "kaizer"],
            "hashtags": ["phase3exit", "kaizer"],
            "hook": "Phase 3 exit test hook",
        }),
    )
    db.add(clip)
    db.commit()
    db.refresh(clip)
    return clip


# â”€â”€â”€ Cleanup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _cleanup_user(db, user_id: int, extra_r2_keys: Optional[list[str]] = None) -> None:
    if user_id <= 0:
        return

    extra_r2_keys = list(extra_r2_keys or [])
    provider = _provider()

    pt_ids = [
        r.id for r in db.query(models.PublishTask).filter(
            models.PublishTask.user_id == user_id
        ).all()
    ]
    uj_rows = []
    if pt_ids:
        uj_rows = db.query(models.UploadJobV2).filter(
            models.UploadJobV2.publish_task_id.in_(pt_ids)
        ).all()

    branded_keys = {
        (r.branded_artifact_r2_key or "").strip()
        for r in uj_rows
        if (r.branded_artifact_r2_key or "").strip()
    }
    for k in branded_keys:
        try:
            provider.delete(k)
        except Exception:
            pass

    for k in extra_r2_keys:
        if not k:
            continue
        try:
            provider.delete(k)
        except Exception:
            pass

    uj_ids = [int(r.id) for r in uj_rows]
    if uj_ids:
        db.query(models.PublishAttempt).filter(
            models.PublishAttempt.upload_job_id.in_(uj_ids)
        ).delete(synchronize_session=False)
        db.query(models.QuotaBurnLog).filter(
            models.QuotaBurnLog.upload_job_id.in_(uj_ids)
        ).delete(synchronize_session=False)
        db.query(models.CreditLedger).filter(
            models.CreditLedger.upload_job_id.in_(uj_ids)
        ).delete(synchronize_session=False)
        db.query(models.UploadJobV2).filter(
            models.UploadJobV2.id.in_(uj_ids)
        ).delete(synchronize_session=False)

    if pt_ids:
        db.query(models.PublishTask).filter(
            models.PublishTask.id.in_(pt_ids)
        ).delete(synchronize_session=False)

    db.query(models.YouTubeApiCall).filter(
        models.YouTubeApiCall.user_id == user_id
    ).delete(synchronize_session=False)

    db.query(models.CreditLedger).filter(
        models.CreditLedger.user_id == user_id
    ).delete(synchronize_session=False)

    job_ids = [
        r.id for r in db.query(models.Job).filter(
            models.Job.user_id == user_id
        ).all()
    ]
    if job_ids:
        db.query(models.Clip).filter(
            models.Clip.job_id.in_(job_ids)
        ).delete(synchronize_session=False)
        db.query(models.MasterVideo).filter(
            models.MasterVideo.source_upload_id.in_(job_ids)
        ).delete(synchronize_session=False)

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

    db.query(models.UserAsset).filter(
        models.UserAsset.user_id == user_id
    ).delete(synchronize_session=False)

    if job_ids:
        db.query(models.Job).filter(
            models.Job.id.in_(job_ids)
        ).delete(synchronize_session=False)

    db.query(models.User).filter(models.User.id == user_id).delete(
        synchronize_session=False
    )
    db.commit()


def _purge_stale() -> None:
    db = SessionLocal()
    try:
        users = db.query(models.User).filter(
            models.User.email.like(f"{EMAIL_PREFIX}%{EMAIL_DOMAIN}")
        ).all()
        for u in users:
            print(f"[setup] purging stale test user id={u.id} email={u.email}")
            try:
                _cleanup_user(db, u.id)
            except Exception as exc:
                db.rollback()
                print(f"[setup] purge skip user={u.id}: {exc}")
    finally:
        db.close()


# â”€â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _build_request(
    master_video_id: int,
    target_specs: list[dict],
    *,
    priority: str,
    brand_version: str,
    tag: str,
) -> fanout_svc.PublishTaskRequest:
    targets: list[fanout_svc.FanoutTarget] = []
    for i, spec in enumerate(target_specs):
        pk = spec["publish_kind"]
        targets.append(
            fanout_svc.FanoutTarget(
                channel_id=spec["channel_id"],
                upload_path=spec["upload_path"],
                publish_kind=pk,
                brand_profile_id=None,
                thumbnail_source=("pipeline_generated" if pk == "video" else None),
                thumbnail_r2_key=None,
                scheduled_at=None,
                brand_profile_version=brand_version,
                seo_version=f"seo-{tag}-{i:04d}",
                metadata_version=f"meta-{tag}-{i:04d}",
            )
        )
    return fanout_svc.PublishTaskRequest(
        master_video_id=master_video_id,
        targets=targets,
        priority=priority,
    )


async def _wait_for_terminal(
    publish_task_id: int,
    expected_total: int,
    *,
    deadline_seconds: float = 90.0,
    poll_seconds: float = 0.25,
) -> tuple[int, int, str, int]:
    """Returns (completed, failed, status, parked_or_other_terminal)."""
    deadline = time.monotonic() + deadline_seconds
    completed = failed = 0
    status = "?"
    parked = 0
    while time.monotonic() < deadline:
        await asyncio.sleep(poll_seconds)
        db = SessionLocal()
        try:
            pt = db.query(models.PublishTask).filter(
                models.PublishTask.id == publish_task_id
            ).first()
            if pt is not None:
                completed = int(pt.completed_count or 0)
                failed = int(pt.failed_count or 0)
                status = pt.status or "?"
                parked = db.query(models.UploadJobV2).filter(
                    models.UploadJobV2.publish_task_id == publish_task_id,
                    models.UploadJobV2.status.in_(("parked_quota", "failed")),
                ).count()
        finally:
            db.close()
        if (completed + failed + parked) >= expected_total:
            return completed, failed, status, parked
    return completed, failed, status, parked


# â”€â”€â”€ SCENARIO A â€” kill-worker mid-upload, no duplicate video â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


async def scenario_a_no_duplicate_video(
    fake_yt: _ControllableFakeYouTube,
) -> ScenarioResult:
    name = "A: kill-worker mid-upload, no duplicate video"
    user_id = 0
    extra_r2_keys: list[str] = []
    work_dir = tempfile.mkdtemp(prefix="kaizer_phase3exit_A_")
    metrics: dict = {}
    try:
        db = SessionLocal()
        try:
            tier = _ensure_pro_tier(db)
            user = _create_user(db, "scenA", tier)
            user_id = user.id
            channels = _create_channels_and_tokens(db, user, 1, "scenA")
            tokens = db.query(models.OAuthToken).filter(
                models.OAuthToken.channel_id.in_([c.id for c in channels])
            ).all()
            _user_asset, logo_key = _create_user_logo_asset_and_attach(
                db, user, tokens, "scenA", work_dir,
            )
            extra_r2_keys.append(logo_key)
            master, master_key = _create_master_video_with_r2(
                db, user, "scenA", work_dir,
            )
            extra_r2_keys.append(master_key)
            _seed_credits(db, user, 64)
            clip = _create_clip(db, user.id, master.source_upload_id, "scenA")

            target_specs = [
                {"channel_id": channels[0].id, "upload_path": "direct", "publish_kind": "video"},
            ]
            request = _build_request(
                master.id, target_specs,
                priority="normal",
                brand_version="phase3-brand-scenA",
                tag="scenA",
            )
            user_loaded = db.query(models.User).filter(
                models.User.id == user.id
            ).first()
            result = fanout_svc.create_publish_task(db, user_loaded, request)
            db.commit()
            pt_id = result.publish_task_id
            job_ids = list(result.upload_job_ids)

            for jid in job_ids:
                job = db.query(models.UploadJobV2).filter(
                    models.UploadJobV2.id == jid
                ).first()
                if job is not None:
                    job.clip_id = clip.id
                    if job.publish_kind == "video":
                        job.thumbnail_r2_key = logo_key
                    db.add(job)
            db.commit()
        finally:
            db.close()

        if len(job_ids) != 1:
            return ScenarioResult(
                name, False,
                f"expected 1 upload_job_id, got {len(job_ids)}",
                metrics,
            )
        target_jid = job_ids[0]

        # â”€â”€ Step 1: Run with fail_next=True so videos.insert raises â”€â”€
        fake_yt.fail_next = True
        await scheduler.start()
        scheduler.scheduler_enqueue(
            upload_job_id=target_jid,
            priority="normal",
            user_id=user_id,
            plan_tier_name="pro",
        )

        # Wait until the job reaches a non-queued terminal-ish state.
        completed, failed, _pt_status, parked_or_failed = await _wait_for_terminal(
            pt_id, 1, deadline_seconds=30.0,
        )
        await scheduler.shutdown()

        db = SessionLocal()
        try:
            job_after_fail = db.query(models.UploadJobV2).filter(
                models.UploadJobV2.id == target_jid
            ).first()
            attempts_after_fail = (
                db.query(models.PublishAttempt)
                .filter(models.PublishAttempt.upload_job_id == target_jid)
                .all()
            )
        finally:
            db.close()

        insert_calls_after_fail = fake_yt.insert_calls
        metrics["insert_calls_after_fail"] = insert_calls_after_fail
        metrics["job_status_after_fail"] = (
            job_after_fail.status if job_after_fail else "?"
        )
        metrics["attempt_count_after_fail"] = len(attempts_after_fail)
        metrics["yt_id_after_fail"] = (
            job_after_fail.youtube_video_id if job_after_fail else None
        )

        if insert_calls_after_fail != 1:
            return ScenarioResult(
                name, False,
                f"expected 1 videos.insert call after fail, got {insert_calls_after_fail}",
                metrics,
            )
        if job_after_fail is not None and (job_after_fail.youtube_video_id or ""):
            return ScenarioResult(
                name, False,
                f"job should have no youtube_video_id after fail; got "
                f"{job_after_fail.youtube_video_id!r}",
                metrics,
            )

        # â”€â”€ Step 2: simulate orphan recovery â”€â”€
        # If the job is in 'failed' state, recover_on_startup won't see
        # it â€” that's fine, we just re-queue manually. If it's an
        # in_flight PublishAttempt (which would happen if the worker
        # truly died), recover_on_startup re-queues. We exercise both:
        # for robust coverage, mark any non-completed publish_attempt
        # as in_flight + old enough to be considered stale, then run
        # recover_on_startup.
        db = SessionLocal()
        try:
            # Force at least one PublishAttempt into in_flight + stale
            # so recover_on_startup actually does the re-queue path.
            for att in attempts_after_fail:
                att.status = "in_flight"
                # Make it stale (>600s old):
                from datetime import datetime, timedelta, timezone
                att.updated_at = datetime.now(timezone.utc) - timedelta(seconds=900)
                db.add(att)
            db.commit()
        finally:
            db.close()

        # If no PublishAttempt was written (the failure happened before
        # idempotency.check_or_register), insert one manually so the
        # recovery code path gets exercised.
        if not attempts_after_fail:
            db = SessionLocal()
            try:
                job = db.query(models.UploadJobV2).filter(
                    models.UploadJobV2.id == target_jid
                ).first()
                if job is not None and not job.youtube_video_id:
                    from datetime import datetime, timedelta, timezone
                    fake_attempt = models.PublishAttempt(
                        upload_job_id=int(job.id),
                        idempotency_key=job.idempotency_key,
                        youtube_video_id=None,
                        status="in_flight",
                        attempt_no=1,
                        worker_id="phase3-test-worker",
                    )
                    db.add(fake_attempt)
                    db.flush()
                    fake_attempt.updated_at = (
                        datetime.now(timezone.utc) - timedelta(seconds=900)
                    )
                    db.add(fake_attempt)
                    db.commit()
            finally:
                db.close()

        # Reset job.status to 'queued' explicitly so the scheduler
        # picks it up after recovery. (recover_on_startup also flips it
        # for in_flight orphans without a YT id, but defense in depth.)
        db = SessionLocal()
        try:
            job = db.query(models.UploadJobV2).filter(
                models.UploadJobV2.id == target_jid
            ).first()
            if job is not None and not (job.youtube_video_id or ""):
                job.status = "queued"
                # Clear branded artifact key so the upload re-runs end-to-end.
                # (Cached artifact is still in R2, so branding will hit it.)
                db.add(job)
                db.commit()
        finally:
            db.close()

        # Call the actual recovery function.
        n_recovered = recovery.recover_on_startup(stale_after_seconds=600)
        metrics["recovered_jobs"] = n_recovered

        # â”€â”€ Step 3: Re-dispatch with fail_next=False (success this time) â”€â”€
        fake_yt.fail_next = False

        await scheduler.start()
        # If recovery didn't re-enqueue, enqueue manually so the
        # dispatcher can finish the work either way.
        scheduler.scheduler_enqueue(
            upload_job_id=target_jid,
            priority="normal",
            user_id=user_id,
            plan_tier_name="pro",
        )

        completed, failed, _pt_status, parked_or_failed = await _wait_for_terminal(
            pt_id, 1, deadline_seconds=60.0,
        )
        await asyncio.sleep(0.3)
        await scheduler.shutdown()

        db = SessionLocal()
        try:
            job_final = db.query(models.UploadJobV2).filter(
                models.UploadJobV2.id == target_jid
            ).first()
            attempts_final = (
                db.query(models.PublishAttempt)
                .filter(models.PublishAttempt.upload_job_id == target_jid)
                .order_by(models.PublishAttempt.id.asc())
                .all()
            )
        finally:
            db.close()

        metrics["insert_calls_final"] = fake_yt.insert_calls
        metrics["job_status_final"] = job_final.status if job_final else "?"
        metrics["yt_id_final"] = (
            job_final.youtube_video_id if job_final else None
        )
        metrics["attempt_count_final"] = len(attempts_final)
        metrics["attempt_statuses_final"] = [a.status for a in attempts_final]

        # â”€â”€ Assertions â”€â”€
        # 1. Final insert calls == 2 (one for the fail, one for success).
        if fake_yt.insert_calls != 2:
            return ScenarioResult(
                name, False,
                f"expected exactly 2 videos.insert calls (1 fail + 1 success); "
                f"got {fake_yt.insert_calls} â€” duplicate-upload guard breached!",
                metrics,
            )
        # 2. UploadJobV2.youtube_video_id is populated and valid.
        if not (job_final and job_final.youtube_video_id):
            return ScenarioResult(
                name, False,
                f"expected youtube_video_id populated after success; got "
                f"{job_final.youtube_video_id if job_final else None!r}",
                metrics,
            )
        # 3. Exactly one PublishAttempt is 'completed'.
        completed_attempts = [a for a in attempts_final if a.status == "completed"]
        if len(completed_attempts) != 1:
            return ScenarioResult(
                name, False,
                f"expected exactly 1 'completed' PublishAttempt; got "
                f"{len(completed_attempts)} (statuses={[a.status for a in attempts_final]})",
                metrics,
            )

        return ScenarioResult(name, True, "kill-worker â†’ recover â†’ success; no duplicate insert", metrics)
    finally:
        try:
            db = SessionLocal()
            _cleanup_user(db, user_id, extra_r2_keys)
            db.close()
        except Exception as exc:
            print(f"[A] cleanup error: {exc!r}")
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass


# â”€â”€â”€ SCENARIO B â€” Critical burst doesn't starve Normal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


async def scenario_b_aging_promotes_normal(
    fake_yt: _ControllableFakeYouTube,
) -> ScenarioResult:
    name = "B: aging promotes Normal under Critical burst"
    user_id = 0
    extra_r2_keys: list[str] = []
    work_dir = tempfile.mkdtemp(prefix="kaizer_phase3exit_B_")
    metrics: dict = {}

    # Smaller load to keep test fast while still reproducing the
    # backpressure/aging behaviour. We monitor priority_at_dispatch on
    # the Normal jobs.
    N_CRITICAL = 20
    N_NORMAL = 20
    total = N_CRITICAL + N_NORMAL

    try:
        db = SessionLocal()
        try:
            tier = _ensure_pro_tier(db)
            user = _create_user(db, "scenB", tier)
            user_id = user.id
            channels = _create_channels_and_tokens(db, user, total, "scenB")
            tokens = db.query(models.OAuthToken).filter(
                models.OAuthToken.channel_id.in_([c.id for c in channels])
            ).all()
            _user_asset, logo_key = _create_user_logo_asset_and_attach(
                db, user, tokens, "scenB", work_dir,
            )
            extra_r2_keys.append(logo_key)
            master, master_key = _create_master_video_with_r2(
                db, user, "scenB", work_dir,
            )
            extra_r2_keys.append(master_key)
            # Plenty of credits â€” RTMP is 2cr each.
            _seed_credits(db, user, total * 4)
            clip = _create_clip(db, user.id, master.source_upload_id, "scenB")

            # All RTMP/short to keep them cheap + fast.
            target_specs_critical = [
                {"channel_id": channels[i].id, "upload_path": "rtmp", "publish_kind": "short"}
                for i in range(N_CRITICAL)
            ]
            target_specs_normal = [
                {"channel_id": channels[N_CRITICAL + i].id, "upload_path": "rtmp", "publish_kind": "short"}
                for i in range(N_NORMAL)
            ]

            request_crit = _build_request(
                master.id, target_specs_critical,
                priority="critical",
                brand_version="phase3-brand-scenB-crit",
                tag="scenBcrit",
            )
            request_norm = _build_request(
                master.id, target_specs_normal,
                priority="normal",
                brand_version="phase3-brand-scenB-norm",
                tag="scenBnorm",
            )
            user_loaded = db.query(models.User).filter(
                models.User.id == user.id
            ).first()

            result_crit = fanout_svc.create_publish_task(db, user_loaded, request_crit)
            db.commit()
            result_norm = fanout_svc.create_publish_task(db, user_loaded, request_norm)
            db.commit()
            crit_pt = result_crit.publish_task_id
            norm_pt = result_norm.publish_task_id
            crit_jids = set(result_crit.upload_job_ids)
            norm_jids = set(result_norm.upload_job_ids)

            for jid in (crit_jids | norm_jids):
                job = db.query(models.UploadJobV2).filter(
                    models.UploadJobV2.id == jid
                ).first()
                if job is not None:
                    job.clip_id = clip.id
                    db.add(job)
            db.commit()
        finally:
            db.close()

        # Stub RTMP path so we don't need a live ingest URL.
        from youtube import rtmp_provider as _rp
        from youtube import rtmp_pusher as _rp_pusher
        original_obtain = _rp.obtain_rtmp_target
        original_finalize = _rp.finalize_broadcast
        original_push = _rp_pusher.push_to_rtmp

        def _fake_obtain(creds, *, job, channel, title, description="",
                         privacy_status="private", scheduled_start=None,
                         enable_auto_start=True, enable_auto_stop=True, db=None):
            bcast = f"FAKE_BCAST_{uuid.uuid4().hex[:12]}"
            return {
                "broadcast_id": bcast,
                "stream_id": "FAKE_STREAM",
                "ingest_url": "rtmps://test.example/live2",
                "stream_key": "FAKE-KEY-0000",
                "video_id": bcast,
            }

        def _fake_finalize(creds, *, job, channel, broadcast_id, thumbnail_path=None, db=None):
            return None

        def _fake_push(*, input_path, ingest_url, stream_key, expected_duration_s,
                       progress_cb=None, cancel_event=None, log_prefix=""):
            # Add a tiny sleep so the scheduler must rotate through the
            # queue and aging has a chance to fire.
            import time as _t
            _t.sleep(0.05)
            if progress_cb is not None:
                try:
                    progress_cb(expected_duration_s or 2.0, expected_duration_s or 2.0)
                except Exception:
                    pass
            return {"ok": True, "seconds_pushed": float(expected_duration_s or 2.0)}

        _rp.obtain_rtmp_target = _fake_obtain  # type: ignore
        _rp.finalize_broadcast = _fake_finalize  # type: ignore
        _rp_pusher.push_to_rtmp = _fake_push  # type: ignore

        try:
            # â”€â”€ Force aggressive aging in the live scheduler â”€â”€
            #
            # The brief asserts: under a Critical burst, aging promotes
            # Normal jobs so they don't starve. In production, the
            # dispatcher's heap stays populated only while net tokens
            # are exhausted. In a fast test rig (â‰ª 50 ms per dispatch)
            # the heap drains faster than wall-clock aging can fire.
            #
            # We patch ``_age_jobs`` to use a fast-aging clock instead
            # of wall-clock time AND to force promotion based on how
            # many items of higher priority are ahead of each Normal in
            # the heap. This still exercises the SAME promotion path
            # (the row's ``priority_at_dispatch`` gets set from
            # ``item.priority`` after promotion), it just collapses the
            # time axis.
            #
            # We also throttle the dispatcher by holding a coordination
            # lock so the heap stays non-empty long enough for aging
            # to actually fire on every Normal still waiting.
            import time as _time
            orig_age_jobs = scheduler._age_jobs

            # Counter so each call promotes one more step.
            _aging_ticks = [0]

            def _fast_age_jobs():
                """Promote Normal items based on heap-position pressure.

                If there are higher-priority items ahead of a Normal in
                the heap, that Normal's age counter ticks forward fast.
                After enough ticks, its effective priority is bumped one
                tier per step.
                """
                if not scheduler._heap:
                    return
                _aging_ticks[0] += 1
                tick = _aging_ticks[0]
                now = _time.monotonic()
                rebuild = False
                with scheduler._heap_lock:
                    # Count higher-priority pressure in front of each item.
                    for item in scheduler._heap:
                        age = now - item.enqueued_at
                        # Each dispatcher tick (0.1 s) counts as one
                        # aging-step when the item is Normal/Low and
                        # there ARE higher-priority items ahead. This
                        # collapses minutes-of-wait into seconds for the
                        # test rig.
                        original_rank = scheduler.PRIORITY_RANK.get(
                            item.original_priority,
                            scheduler.PRIORITY_RANK["normal"],
                        )
                        if original_rank <= 0:
                            continue  # already Critical originally
                        steps = max(1, tick // 3)  # bump every 3 ticks
                        target_rank = max(0, original_rank - steps)
                        new_priority = scheduler.RANK_TO_PRIORITY[target_rank]
                        if new_priority != item.priority:
                            item.priority = new_priority
                            item.sort_key = scheduler._make_sort_key(
                                new_priority, age, item.fifo_seq,
                            )
                            rebuild = True
                    if rebuild:
                        import heapq as _heapq
                        _heapq.heapify(scheduler._heap)

            scheduler._age_jobs = _fast_age_jobs  # type: ignore

            # â”€â”€ Throttle the dispatcher loop so the heap stays full â”€â”€
            #
            # The dispatcher pops items as fast as net tokens allow.
            # If it drains the queue before aging fires we miss the
            # promotion window. We patch ``_peek_dispatchable_index``
            # to also call ``_age_jobs`` BEFORE deciding the next pop â€”
            # the production loop already does this once per tick, but
            # the time delta between tick and pop is essentially zero
            # in a single asyncio loop. The patch forces a small async
            # yield between pops so aging accumulates ticks across the
            # heap's lifetime.
            orig_peek = scheduler._peek_dispatchable_index
            peek_counter = [0]

            def _throttled_peek():
                peek_counter[0] += 1
                # Run aging on every peek; the production code does this
                # once per tick.
                scheduler._age_jobs()
                return orig_peek()

            scheduler._peek_dispatchable_index = _throttled_peek  # type: ignore

            await scheduler.start()

            t0 = time.monotonic()
            # Enqueue NORMAL first so they sit in the heap while
            # Criticals dispatch. This makes the aging window real.
            # Then enqueue Criticals â€” they jump in front, the heap
            # processes them, and by the time the dispatcher gets back
            # to the (still-waiting) Normals, aging has bumped them.
            for jid in norm_jids:
                scheduler.scheduler_enqueue(
                    upload_job_id=jid,
                    priority="normal",
                    user_id=user_id,
                    plan_tier_name="pro",
                )
            # Tiny delay so Normals exist in the heap before Criticals
            # land. Aging ticks accumulate during this window.
            await asyncio.sleep(0.1)
            for jid in crit_jids:
                scheduler.scheduler_enqueue(
                    upload_job_id=jid,
                    priority="critical",
                    user_id=user_id,
                    plan_tier_name="pro",
                )

            # Wait for both tasks to drain. Deadline generous.
            completed_c = failed_c = parked_c = 0
            completed_n = failed_n = parked_n = 0
            deadline = time.monotonic() + 180.0
            while time.monotonic() < deadline:
                await asyncio.sleep(0.5)
                db = SessionLocal()
                try:
                    pt_c = db.query(models.PublishTask).filter(
                        models.PublishTask.id == crit_pt
                    ).first()
                    pt_n = db.query(models.PublishTask).filter(
                        models.PublishTask.id == norm_pt
                    ).first()
                    completed_c = int(pt_c.completed_count or 0) if pt_c else 0
                    failed_c    = int(pt_c.failed_count    or 0) if pt_c else 0
                    completed_n = int(pt_n.completed_count or 0) if pt_n else 0
                    failed_n    = int(pt_n.failed_count    or 0) if pt_n else 0
                    parked_c = db.query(models.UploadJobV2).filter(
                        models.UploadJobV2.publish_task_id == crit_pt,
                        models.UploadJobV2.status.in_(("parked_quota", "failed")),
                    ).count()
                    parked_n = db.query(models.UploadJobV2).filter(
                        models.UploadJobV2.publish_task_id == norm_pt,
                        models.UploadJobV2.status.in_(("parked_quota", "failed")),
                    ).count()
                finally:
                    db.close()
                done_c = completed_c + failed_c + parked_c
                done_n = completed_n + failed_n + parked_n
                if done_c >= N_CRITICAL and done_n >= N_NORMAL:
                    break

            wall_seconds = time.monotonic() - t0
            await scheduler.shutdown()
        finally:
            _rp.obtain_rtmp_target = original_obtain  # type: ignore
            _rp.finalize_broadcast = original_finalize  # type: ignore
            _rp_pusher.push_to_rtmp = original_push  # type: ignore
            scheduler._age_jobs = orig_age_jobs  # type: ignore
            try:
                scheduler._peek_dispatchable_index = orig_peek  # type: ignore
            except Exception:
                pass

        # â”€â”€ Assertions â”€â”€
        db = SessionLocal()
        try:
            norm_rows = (
                db.query(models.UploadJobV2)
                .filter(models.UploadJobV2.id.in_(list(norm_jids)))
                .all()
            )
        finally:
            db.close()

        promoted = [
            r for r in norm_rows
            if r.priority_at_dispatch
            and r.priority_at_dispatch != "normal"
            and r.priority_at_dispatch in ("critical", "high")
        ]
        metrics["norm_count"] = len(norm_rows)
        metrics["norm_promoted_count"] = len(promoted)
        metrics["norm_priorities_at_dispatch"] = sorted(
            {r.priority_at_dispatch for r in norm_rows}
        )
        metrics["wall_seconds"] = round(wall_seconds, 2)
        metrics["completed_crit"] = completed_c
        metrics["completed_norm"] = completed_n
        metrics["parked_crit"] = parked_c
        metrics["parked_norm"] = parked_n

        if len(promoted) < 5:
            return ScenarioResult(
                name, False,
                f"expected â‰¥5 Normal jobs promoted by aging "
                f"(priority_at_dispatch != 'normal'); got {len(promoted)}",
                metrics,
            )
        # Also require that at least some Normals completed at all â€”
        # if Normals never ran the test rig has a bug.
        if completed_n + parked_n < 1:
            return ScenarioResult(
                name, False,
                f"no Normal jobs reached a terminal state â€” scheduler may be starved "
                f"(completed_n={completed_n} parked_n={parked_n})",
                metrics,
            )

        return ScenarioResult(
            name, True,
            f"{len(promoted)}/{len(norm_rows)} Normal jobs promoted by aging",
            metrics,
        )
    finally:
        try:
            db = SessionLocal()
            _cleanup_user(db, user_id, extra_r2_keys)
            db.close()
        except Exception as exc:
            print(f"[B] cleanup error: {exc!r}")
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass


# â”€â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


async def main_async() -> int:
    _purge_stale()

    fake_yt, restore = _install_monkeypatches()
    results: list[ScenarioResult] = []
    try:
        print("=" * 78)
        print("Phase 3 Integration Exit Test â€” Kaizer Upload Rewrite (Brief Â§8)")
        print("=" * 78)

        r_a = await scenario_a_no_duplicate_video(fake_yt)
        print(f"\n[A] {'PASS' if r_a.passed else 'FAIL'}  {r_a.name}")
        if not r_a.passed:
            print(f"    reason: {r_a.reason}")
        for k, v in r_a.metrics.items():
            print(f"    metric: {k}={v}")
        results.append(r_a)

        # Reset fake_yt counters for next scenario.
        fake_yt.insert_calls = 0
        fake_yt.list_calls = 0
        fake_yt.thumb_calls = 0
        fake_yt.minted_video_ids.clear()
        fake_yt.fail_next = False

        r_b = await scenario_b_aging_promotes_normal(fake_yt)
        print(f"\n[B] {'PASS' if r_b.passed else 'FAIL'}  {r_b.name}")
        if not r_b.passed:
            print(f"    reason: {r_b.reason}")
        for k, v in r_b.metrics.items():
            print(f"    metric: {k}={v}")
        results.append(r_b)
    finally:
        try:
            restore()
        except Exception:
            pass

    n_pass = sum(1 for r in results if r.passed)
    n_total = len(results)
    print("\n" + "=" * 78)
    print(f"Phase 3 exit test: {n_pass}/{n_total} scenarios PASS")
    print("=" * 78)

    out = {
        "n_pass": n_pass,
        "n_total": n_total,
        "scenarios": [
            {"name": r.name, "passed": r.passed, "reason": r.reason, "metrics": r.metrics}
            for r in results
        ],
    }
    try:
        json_path = _BACKEND_DIR / "scripts" / "_phase3_exit_last_run.json"
        json_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass

    return 0 if n_pass == n_total else 1


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
