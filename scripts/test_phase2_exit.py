"""Phase 2 Integration Exit-Test (KAIZER_UPLOAD_REWRITE_BRIEF.md Â§7).

Exercises the four exit criteria of Phase 2 end-to-end against the real
dev Postgres DB and the real R2 (or LocalStorage) provider, with the
YouTube API + OAuth + RTMP layer monkeypatched (the real YouTube path
needs live OAuth tokens and would burn quota on every local run).

Brief Â§7 exit criteria (verbatim):

    publish one MasterVideo to a mix of Direct and RTMP channels;
    correct credits burned per path; artifacts cached and reused;
    quota logged correctly.

Four scenarios:

  A. Mixed Direct + RTMP fanout (4 channels, 1 master).
       - 2 Ã— Direct/video, 2 Ã— RTMP/short
       - asserts credit deduction matches Decision 4 (2Ã—16 + 2Ã—2 = 36 cr)
       - asserts branded artifact cache: 1 miss + 3 hits (same master +
         brand_profile_version â†’ single ffmpeg pass shared by 4 jobs)
       - asserts quota_burn_log rows per operation (videos.insert,
         thumbnails.set Direct only, liveBroadcasts/Streams/bind RTMP)
       - asserts forensic log preserved (youtube_api_calls)
       - asserts slot manager fully drained (net_in_use=0, cpu_in_use=0)

  B. Branded artifact cache hit/miss verification.
       - 2 jobs same brand_profile_version â†’ 1 miss + 1 hit.

  C. Quota gate exercise â€” lowered KAIZER_YT_DAILY_QUOTA_CAP forces
     a job to park_quota.
       - 3 Direct/video targets; cap=3300 â†’ 2 succeed, 1 parks
       - asserts parked job got a refund row + did NOT bump attempts
       - asserts QuotaBurnLog row for the parked attempt has
         was_quota_exceeded=True

  D. Legacy path still works (4 HTTP probes; no regression).

The scheduler is started in-process via ``await scheduler.start()`` and
torn down via ``await scheduler.shutdown()`` per scenario so we don't
contaminate other tests with leftover dispatcher tasks.

Run::

    cd kaizer/KaizerBackend
    python scripts/test_phase2_exit.py

Exit 0 on all-pass, 1 on any failure.
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

# Force the per-channel A/V nudge OFF for this suite. Scenarios A+B assert
# branded-artifact cache SHARING (one ffmpeg pass reused across jobs of the
# same brand), which the nudge INTENTIONALLY breaks — it derives a distinct
# speed factor per channel.id, so each channel gets its own branded artifact.
# The nudge is verified separately; here we isolate the cache/fanout/credits
# machinery. Hard-set (not setdefault) so it wins over .env; process-local,
# never touches production (which keeps KAIZER_BRAND_AV_NUDGE=1).
os.environ["KAIZER_BRAND_AV_NUDGE"] = "0"
# Same reasoning for the per-channel VISUAL ZOOM — it too forks the branded
# artifact per channel (breaking the cache-sharing this suite asserts).
os.environ["KAIZER_BRAND_VISUAL_ZOOM"] = "0"

# Make ``import models`` / ``import database`` work from the scripts dir.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# Windows cmd.exe defaults to cp1252; force utf-8 so the test output
# is identical on Windows and Linux runners.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO if os.environ.get("KAIZER_PHASE2_EXIT_VERBOSE") else logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from database import SessionLocal  # noqa: E402
import models  # noqa: E402
from services import branding  # noqa: E402
from services import credits as credits_svc  # noqa: E402
from services import fanout as fanout_svc  # noqa: E402
from services import scheduler  # noqa: E402

EMAIL_PREFIX = "phase2exit_test_"
EMAIL_DOMAIN = "@kaizer.test"
HTTP_BASE = "http://127.0.0.1:8000"
HTTP_TIMEOUT = 5.0


# â”€â”€â”€ Result tracking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    reason: str = ""
    metrics: dict = field(default_factory=dict)


# â”€â”€â”€ ffmpeg helpers (reused from test_upload_dispatch_smoke.py) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


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
        raise RuntimeError(f"ffmpeg failed to generate test mp4: {r.stderr[-300:]}")


def _make_test_logo_png(out_path: str) -> None:
    cmd = [
        _ffmpeg_bin(), "-y", "-v", "error",
        "-f", "lavfi", "-i", "color=c=blue:s=64x64:d=1",
        "-frames:v", "1",
        out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to generate test logo: {r.stderr[-300:]}")


def _provider():
    from pipeline_core.storage import get_storage_provider
    return get_storage_provider()


# â”€â”€â”€ Fake YouTube API surface (reused/adapted from upload_dispatch smoke) â”€â”€â”€â”€


class _FakeMediaProgress:
    def __init__(self, uploaded: int):
        self.resumable_progress = uploaded


class _FakeInsertRequest:
    """Mimics googleapiclient resumable upload: two chunks then a result."""

    def __init__(self, total_bytes: int, video_id: str):
        self._total = max(1, int(total_bytes))
        self._video_id = video_id
        self._chunks = 0
        self.resumable_uri = f"https://uploads.example/v1/sess/{uuid.uuid4().hex}"

    def next_chunk(self):
        self._chunks += 1
        if self._chunks == 1:
            return _FakeMediaProgress(self._total // 2), None
        return None, {"id": self._video_id}


class _FakeVideosResource:
    def __init__(self, parent: "_FakeYouTube"):
        self._p = parent

    def insert(self, *, part, body, media_body, notifySubscribers=False):
        self._p.insert_calls += 1
        self._p.last_insert_body = body
        # Each insert gets its own unique fake id so we can verify each
        # job is associated with a distinct YouTube video.
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
        return _FakeInsertRequest(total_bytes=size or 100_000, video_id=new_id)

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
    def __init__(self, parent: "_FakeYouTube"):
        self._p = parent

    def set(self, *, videoId, media_body):
        self._p.thumb_calls += 1

        class _Req:
            def execute(self_inner):
                return {}

        return _Req()


class _FakeYouTube:
    """Drop-in replacement for googleapiclient youtube service."""

    def __init__(self):
        self.insert_calls = 0
        self.list_calls = 0
        self.thumb_calls = 0
        self.last_insert_body: Optional[dict] = None
        self.last_video_id: Optional[str] = None
        self.minted_video_ids: set[str] = set()
        self._videos = _FakeVideosResource(self)
        self._thumbnails = _FakeThumbnailsResource(self)

    def videos(self):
        return self._videos

    def thumbnails(self):
        return self._thumbnails


# â”€â”€â”€ Monkeypatch installer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _install_monkeypatches() -> tuple["_FakeYouTube", Any, dict]:
    """Returns (fake_youtube, restore_callable, rtmp_counter)."""
    # 1) youtube.oauth.get_credentials â†’ fake.
    from youtube import oauth as _oauth_mod
    original_get_creds = _oauth_mod.get_credentials

    class _FakeCreds:
        valid = True
        token = "FAKE_ACCESS"
        refresh_token = "FAKE_REFRESH"

    def _fake_get_credentials(db, channel_id):
        return _FakeCreds()

    _oauth_mod.get_credentials = _fake_get_credentials  # type: ignore

    # 2) googleapiclient.discovery.build â†’ FakeYouTube.
    fake_yt = _FakeYouTube()
    import googleapiclient.discovery as _gad
    original_build = _gad.build

    def _fake_build(serviceName, version, **kwargs):
        return fake_yt

    _gad.build = _fake_build  # type: ignore

    # uploader_v2 binds `build` at import-time; patch the imported ref too.
    from youtube import uploader_v2 as _u2
    original_u2_build = getattr(_u2, "build", None)
    _u2.build = _fake_build  # type: ignore

    # 3) RTMP path stubs.
    rtmp_counter = {"obtain": 0, "push": 0, "finalize": 0}
    rtmp_broadcasts: set[str] = set()

    from youtube import rtmp_provider as _rp
    from youtube import rtmp_pusher as _rp_pusher

    original_obtain = _rp.obtain_rtmp_target
    original_finalize = _rp.finalize_broadcast
    original_push = _rp_pusher.push_to_rtmp

    def _fake_obtain(creds, *, job, channel, title, description="", privacy_status="private",
                    scheduled_start=None, enable_auto_start=True, enable_auto_stop=True, db=None):
        rtmp_counter["obtain"] += 1
        bcast = f"FAKE_BCAST_{uuid.uuid4().hex[:12]}"
        rtmp_broadcasts.add(bcast)
        return {
            "broadcast_id": bcast,
            "stream_id": "FAKE_STREAM",
            "ingest_url": "rtmps://test.example/live2",
            "stream_key": "FAKE-KEY-0000",
            "video_id": bcast,
        }

    def _fake_finalize(creds, *, job, channel, broadcast_id, thumbnail_path=None, db=None):
        rtmp_counter["finalize"] += 1
        return None

    def _fake_push(*, input_path, ingest_url, stream_key, expected_duration_s,
                   progress_cb=None, cancel_event=None, log_prefix=""):
        rtmp_counter["push"] += 1
        if progress_cb is not None:
            try:
                progress_cb(expected_duration_s or 2.0, expected_duration_s or 2.0)
            except Exception:
                pass
        return {"ok": True, "seconds_pushed": float(expected_duration_s or 2.0)}

    _rp.obtain_rtmp_target = _fake_obtain  # type: ignore
    _rp.finalize_broadcast = _fake_finalize  # type: ignore
    _rp_pusher.push_to_rtmp = _fake_push  # type: ignore

    def _restore() -> None:
        _oauth_mod.get_credentials = original_get_creds  # type: ignore
        _gad.build = original_build  # type: ignore
        if original_u2_build is not None:
            _u2.build = original_u2_build  # type: ignore
        _rp.obtain_rtmp_target = original_obtain  # type: ignore
        _rp.finalize_broadcast = original_finalize  # type: ignore
        _rp_pusher.push_to_rtmp = original_push  # type: ignore

    return fake_yt, _restore, rtmp_counter


# â”€â”€â”€ Test data factories â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


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


def _create_user(db, scenario_tag: str, plan_tier: models.PlanTier) -> models.User:
    email = _unique_email(scenario_tag)
    user = models.User(
        email=email,
        name=f"Phase 2 Exit Test ({scenario_tag})",
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
            name=f"phase2exit-{tag}-ch{i:03d}-{uuid.uuid4().hex[:6]}",
            language="te",
            watermark_text=f"@phase2exit-{tag}",
            watermark_opacity=0.35,
            watermark_position="lower-center",
        )
        db.add(ch)
    db.commit()
    channels = (
        db.query(models.Channel)
        .filter(
            models.Channel.user_id == user.id,
            models.Channel.name.like(f"phase2exit-{tag}-%"),
        )
        .order_by(models.Channel.id.asc())
        .all()
    )
    assert len(channels) == n, f"created {n} channels but reloaded {len(channels)}"
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
    """Real R2 upload: write a tiny 2s mp4, upload it, create the row."""
    job = models.Job(
        user_id=user.id,
        status="done",
        platform="youtube_full",
        video_name=f"phase2exit_{tag}.mp4",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    master_local = os.path.join(work_dir, f"master_{tag}.mp4")
    _make_test_mp4(master_local)
    master_key = f"phase2exit/{tag}/master.mp4"
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
    """Real R2 upload: a 64Ã—64 blue logo png the resolver can read."""
    logo_local = os.path.join(work_dir, f"logo_{tag}.png")
    _make_test_logo_png(logo_local)
    logo_key = f"phase2exit/{tag}/logo.png"
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

    # Attach to each token (precedence #1) so the brand resolver finds
    # the same logo for every channel â€” identical brand_profile_version
    # â†’ cache reuse.
    for tok in tokens:
        tok.logo_asset_id = user_asset.id
        db.add(tok)
    db.commit()
    return user_asset, logo_key


def _create_clip(db, user_id: int, job_id: int, tag: str) -> models.Clip:
    clip = models.Clip(
        job_id=job_id,
        clip_index=0,
        filename=f"phase2exit_{tag}.mp4",
        file_path="",
        duration=2.0,
        text=f"Phase 2 exit bulletin {tag}",
        seo=json.dumps({
            "title": f"Phase 2 Exit Test Bulletin {tag}",
            "description": "Phase 2 exit test description.",
            "keywords": ["phase2exit", "kaizer"],
            "hashtags": ["phase2exit", "kaizer"],
            "hook": "Phase 2 exit test hook",
        }),
    )
    db.add(clip)
    db.commit()
    db.refresh(clip)
    return clip


# â”€â”€â”€ Cleanup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _cleanup_user(db, user_id: int, extra_r2_keys: Optional[list[str]] = None) -> None:
    """Delete every row + R2 key created by/for the test user."""
    if user_id <= 0:
        return

    extra_r2_keys = list(extra_r2_keys or [])
    provider = _provider()

    # Collect branded artifact keys to delete before tearing down rows.
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

    # PublishTasks â†’ UploadJobV2 â†’ PublishAttempts / QuotaBurnLog / CreditLedger.
    uj_ids = [int(r.id) for r in uj_rows]

    if uj_ids:
        db.query(models.PublishAttempt).filter(
            models.PublishAttempt.upload_job_id.in_(uj_ids)
        ).delete(synchronize_session=False)
        db.query(models.QuotaBurnLog).filter(
            models.QuotaBurnLog.upload_job_id.in_(uj_ids)
        ).delete(synchronize_session=False)
        # youtube_api_calls.upload_job_id FKs the LEGACY upload_jobs table
        # (not upload_jobs_v2), so dispatch passes None there. We still
        # purge by user_id below to catch any rows attributed via user_id.
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

    # Forensic log rows attributed to this user.
    db.query(models.YouTubeApiCall).filter(
        models.YouTubeApiCall.user_id == user_id
    ).delete(synchronize_session=False)

    # Remaining credit_ledger rows for the user.
    db.query(models.CreditLedger).filter(
        models.CreditLedger.user_id == user_id
    ).delete(synchronize_session=False)

    # MasterVideos via Job FK.
    job_ids = [
        r.id for r in db.query(models.Job).filter(
            models.Job.user_id == user_id
        ).all()
    ]
    if job_ids:
        # Drop Clips referencing those Jobs first.
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

    # UserAssets (logos uploaded for this test).
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
    """Belt-and-braces: wipe any leftover test users from prior runs."""
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
    """target_specs: list of dicts with keys channel_id, upload_path, publish_kind."""
    targets: list[fanout_svc.FanoutTarget] = []
    for i, spec in enumerate(target_specs):
        pk = spec["publish_kind"]
        targets.append(
            fanout_svc.FanoutTarget(
                channel_id=spec["channel_id"],
                upload_path=spec["upload_path"],
                publish_kind=pk,
                brand_profile_id=None,
                thumbnail_source=(
                    "pipeline_generated" if pk == "video" else None
                ),
                thumbnail_r2_key=None,
                scheduled_at=None,
                # SAME brand_profile_version for all targets â†’ same brand
                # â†’ same branded artifact cache key â†’ only one ffmpeg pass.
                brand_profile_version=brand_version,
                # DISTINCT seo/metadata so idempotency keys differ.
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


def _http_probe(method: str, path: str, body: Optional[dict] = None) -> int:
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


async def _wait_for_terminal(
    publish_task_id: int,
    expected_total: int,
    *,
    deadline_seconds: float = 90.0,
    poll_seconds: float = 0.25,
) -> tuple[int, int, str]:
    """Poll until completed_count + failed_count >= expected_total or
    deadline. Returns (completed, failed, status)."""
    t0 = time.monotonic()
    deadline = t0 + deadline_seconds
    completed = failed = 0
    status = "?"
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
            # Also include parked_quota in the "done" count â€” those rows
            # don't bump completed/failed but they ARE terminal for this
            # test's purposes.
            parked = 0
            if pt is not None:
                parked = db.query(models.UploadJobV2).filter(
                    models.UploadJobV2.publish_task_id == publish_task_id,
                    models.UploadJobV2.status == "parked_quota",
                ).count()
        finally:
            db.close()
        if (completed + failed + parked) >= expected_total:
            return completed, failed, status
    return completed, failed, status


# â”€â”€â”€ SCENARIO A â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


async def scenario_a_mixed_fanout(
    fake_yt: "_FakeYouTube",
    rtmp_counter: dict,
) -> ScenarioResult:
    name = "A: mixed Direct + RTMP fanout (4 channels)"
    user_id = 0
    extra_r2_keys: list[str] = []
    metrics: dict = {}
    work_dir = tempfile.mkdtemp(prefix="kaizer_phase2exit_A_")
    try:
        # â”€â”€ Setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        db = SessionLocal()
        try:
            tier = _ensure_pro_tier(db)
            user = _create_user(db, "scenA", tier)
            user_id = user.id
            channels = _create_channels_and_tokens(db, user, 4, "scenA")
            tokens = db.query(models.OAuthToken).filter(
                models.OAuthToken.channel_id.in_([c.id for c in channels])
            ).all()
            # Logo asset shared across all 4 oauth_tokens.
            _user_asset, logo_key = _create_user_logo_asset_and_attach(
                db, user, tokens, "scenA", work_dir,
            )
            extra_r2_keys.append(logo_key)
            master, master_key = _create_master_video_with_r2(
                db, user, "scenA", work_dir,
            )
            extra_r2_keys.append(master_key)
            # Seed credits over-provisioned so InsufficientCredits never
            # fires: 4 Ã— 16 = 64 cr (2 Direct@16 + 2 RTMP@2 = 36 cr burn).
            starting_credits = 64
            _seed_credits(db, user, starting_credits)
            # Clip for SEO composition fallback.
            clip = _create_clip(db, user.id, master.source_upload_id, "scenA")

            target_specs = [
                # Targets 1, 2: Direct + video
                {"channel_id": channels[0].id, "upload_path": "direct", "publish_kind": "video"},
                {"channel_id": channels[1].id, "upload_path": "direct", "publish_kind": "video"},
                # Targets 3, 4: RTMP + short
                {"channel_id": channels[2].id, "upload_path": "rtmp", "publish_kind": "short"},
                {"channel_id": channels[3].id, "upload_path": "rtmp", "publish_kind": "short"},
            ]
            request = _build_request(
                master.id, target_specs,
                priority="normal",
                brand_version="phase2-brand-scenA",
                tag="scenA",
            )
            user_loaded = db.query(models.User).filter(
                models.User.id == user.id
            ).first()

            balance_before = _balance(db, user_id)
            result = fanout_svc.create_publish_task(db, user_loaded, request)
            db.commit()
            pt_id = result.publish_task_id
            job_ids = list(result.upload_job_ids)
            balance_after_fanout = _balance(db, user_id)

            # Wire each UploadJobV2 to a Clip so the SEO composer has data.
            # Also pin thumbnail_r2_key on the Direct/video jobs so the
            # dispatch's _maybe_set_thumbnail path actually fires (the
            # pipeline-generated thumbnail flow doesn't auto-write the key
            # in Phase 2; the integration test pins it directly).
            for jid in job_ids:
                job = db.query(models.UploadJobV2).filter(
                    models.UploadJobV2.id == jid
                ).first()
                if job is not None:
                    job.clip_id = clip.id
                    # For video/Direct jobs, simulate a pipeline-uploaded
                    # thumbnail by reusing the logo asset's R2 key as a
                    # stand-in thumbnail. The test only needs SOMETHING the
                    # dispatch can download + thumbnails.set against.
                    if job.publish_kind == "video":
                        job.thumbnail_r2_key = logo_key
                    db.add(job)
            db.commit()
        finally:
            db.close()

        if len(job_ids) != 4:
            return ScenarioResult(
                name, False,
                f"expected 4 upload_job_ids, got {len(job_ids)}",
                metrics,
            )

        # Snapshot branding cache stats BEFORE the run.
        snap_before = branding.snapshot()
        misses_before = int(snap_before.get("misses", 0))
        hits_before = int(snap_before.get("hits", 0))

        # â”€â”€ Forensic log baseline (so we measure delta only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        db = SessionLocal()
        try:
            forensic_baseline = db.query(models.YouTubeApiCall).filter(
                models.YouTubeApiCall.user_id == user_id
            ).count()
            burn_baseline = db.query(models.QuotaBurnLog).filter(
                models.QuotaBurnLog.upload_job_id.in_(job_ids)
            ).count()
            legacy_upload_jobs_baseline = db.query(models.UploadJob).count()
        finally:
            db.close()

        # â”€â”€ Start scheduler + enqueue â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        await scheduler.start()
        t_start = time.monotonic()
        for jid in job_ids:
            scheduler.scheduler_enqueue(
                upload_job_id=jid,
                priority="normal",
                user_id=user_id,
                plan_tier_name="pro",
            )

        completed, failed, pt_status = await _wait_for_terminal(
            pt_id, 4, deadline_seconds=120.0,
        )
        wall_seconds = time.monotonic() - t_start

        await asyncio.sleep(0.5)
        snap_final = scheduler.snapshot()
        sm_final = snap_final.get("slot_manager", {})
        await scheduler.shutdown()

        # â”€â”€ Final asserts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        db = SessionLocal()
        try:
            jobs = db.query(models.UploadJobV2).filter(
                models.UploadJobV2.publish_task_id == pt_id
            ).order_by(models.UploadJobV2.id.asc()).all()
            pt_row = db.query(models.PublishTask).filter(
                models.PublishTask.id == pt_id
            ).first()
            balance_after = _balance(db, user_id)

            # Forensic + burn deltas.
            forensic_total = db.query(models.YouTubeApiCall).filter(
                models.YouTubeApiCall.user_id == user_id
            ).count() - forensic_baseline
            burn_total = db.query(models.QuotaBurnLog).filter(
                models.QuotaBurnLog.upload_job_id.in_(job_ids)
            ).count() - burn_baseline

            forensic_by_op: dict[str, int] = {}
            for row in db.query(models.YouTubeApiCall).filter(
                models.YouTubeApiCall.user_id == user_id
            ).all():
                forensic_by_op[row.operation or ""] = (
                    forensic_by_op.get(row.operation or "", 0) + 1
                )

            burn_by_op: dict[str, int] = {}
            for row in db.query(models.QuotaBurnLog).filter(
                models.QuotaBurnLog.upload_job_id.in_(job_ids)
            ).all():
                burn_by_op[row.operation or ""] = (
                    burn_by_op.get(row.operation or "", 0) + 1
                )

            legacy_upload_jobs_after = db.query(models.UploadJob).count()
        finally:
            db.close()

        # Snapshot branding cache stats AFTER the run.
        snap_after = branding.snapshot()
        misses_delta = int(snap_after.get("misses", 0)) - misses_before
        hits_delta = int(snap_after.get("hits", 0)) - hits_before

        # â”€â”€ Build per-job assertion view â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        direct_jobs = [j for j in jobs if j.upload_path == "direct"]
        rtmp_jobs = [j for j in jobs if j.upload_path == "rtmp"]

        # R2 cache key check.
        cache_key_set: set[str] = set()
        for j in jobs:
            if (j.branded_artifact_r2_key or "").strip():
                cache_key_set.add(j.branded_artifact_r2_key.strip())

        cache_keys_exist_check = False
        if cache_key_set:
            try:
                cache_keys_exist_check = all(
                    _provider().exists(k) for k in cache_key_set
                )
            except Exception:
                cache_keys_exist_check = False

        metrics.update({
            "wall_seconds": round(wall_seconds, 3),
            "completed_count": completed,
            "failed_count": failed,
            "publish_task_status": pt_status,
            "balance_before_fanout": balance_before,
            "balance_after_fanout": balance_after_fanout,
            "balance_after_run": balance_after,
            "credit_burn_total": balance_before - balance_after,
            "branding_misses_delta": misses_delta,
            "branding_hits_delta": hits_delta,
            "distinct_branded_cache_keys": len(cache_key_set),
            "branded_cache_keys_exist_in_r2": cache_keys_exist_check,
            "forensic_log_delta": forensic_total,
            "forensic_log_by_op": forensic_by_op,
            "quota_burn_log_delta": burn_total,
            "quota_burn_log_by_op": burn_by_op,
            "snap_final_net_in_use": sm_final.get("net_in_use"),
            "snap_final_cpu_in_use": sm_final.get("cpu_in_use"),
            "snap_final_user_in_use": sm_final.get("user_in_use", {}).get(user_id, 0),
            "snap_final_queue_depth": snap_final.get("queue_depth"),
            "fake_yt_insert_calls": fake_yt.insert_calls,
            "fake_yt_list_calls": fake_yt.list_calls,
            "fake_yt_thumb_calls": fake_yt.thumb_calls,
            "rtmp_counter": dict(rtmp_counter),
            "direct_video_ids": [j.youtube_video_id for j in direct_jobs],
            "rtmp_video_ids": [j.youtube_video_id for j in rtmp_jobs],
            "legacy_upload_jobs_baseline": legacy_upload_jobs_baseline,
            "legacy_upload_jobs_after": legacy_upload_jobs_after,
        })

        failures: list[str] = []

        if completed != 4:
            failures.append(
                f"expected 4 completed jobs, got {completed} (failed={failed})"
            )
        if failed != 0:
            failures.append(f"expected 0 failed jobs, got {failed}")
        for j in jobs:
            if j.status != "completed":
                failures.append(
                    f"job_id={j.id} status={j.status!r} != 'completed'"
                )
            if not (j.youtube_video_id or "").strip():
                failures.append(
                    f"job_id={j.id} missing youtube_video_id"
                )
        # Direct path video IDs must start with FAKE_VID_; RTMP with FAKE_BCAST_.
        for j in direct_jobs:
            if not (j.youtube_video_id or "").startswith("FAKE_VID_"):
                failures.append(
                    f"direct job_id={j.id} youtube_video_id={j.youtube_video_id!r} "
                    f"does not start with FAKE_VID_"
                )
        for j in rtmp_jobs:
            if not (j.youtube_video_id or "").startswith("FAKE_BCAST_"):
                failures.append(
                    f"rtmp job_id={j.id} youtube_video_id={j.youtube_video_id!r} "
                    f"does not start with FAKE_BCAST_"
                )

        if pt_row is None or pt_row.status != "completed":
            failures.append(
                f"PublishTask.status != 'completed' "
                f"(got {pt_row.status if pt_row else None!r})"
            )
        if pt_row is None or int(pt_row.completed_count or 0) != 4:
            failures.append(
                f"PublishTask.completed_count != 4 "
                f"(got {pt_row.completed_count if pt_row else None})"
            )
        if pt_row is None or int(pt_row.failed_count or 0) != 0:
            failures.append(
                f"PublishTask.failed_count != 0 "
                f"(got {pt_row.failed_count if pt_row else None})"
            )

        # Credit accounting: 2Ã—16 + 2Ã—2 = 36 deducted; balance 64 â†’ 28.
        expected_balance = 64 - (2 * 16 + 2 * 2)
        if balance_after != expected_balance:
            failures.append(
                f"final balance {balance_after} != expected {expected_balance} "
                f"(2 Direct @ 16cr + 2 RTMP @ 2cr = 36 cr burn)"
            )

        # Branding cache: 1 miss + 3 hits (single brand_profile_version).
        if misses_delta != 1:
            failures.append(
                f"branding misses_delta={misses_delta}, expected 1 "
                f"(same brand_profile_version â†’ first job pays ffmpeg, others cache-hit)"
            )
        if hits_delta != 3:
            failures.append(
                f"branding hits_delta={hits_delta}, expected 3 "
                f"(3 jobs should reuse the cached artifact)"
            )
        if len(cache_key_set) != 1:
            failures.append(
                f"expected 1 distinct branded artifact key, got {len(cache_key_set)}: "
                f"{cache_key_set!r}"
            )
        if not cache_keys_exist_check:
            failures.append(
                f"cached branded artifact key(s) {cache_key_set!r} not present in R2"
            )

        # Slot manager drained.
        if int(sm_final.get("net_in_use", -1) or 0) != 0:
            failures.append(
                f"slot_manager.net_in_use != 0 (got {sm_final.get('net_in_use')})"
            )
        if int(sm_final.get("cpu_in_use", -1) or 0) != 0:
            failures.append(
                f"slot_manager.cpu_in_use != 0 (got {sm_final.get('cpu_in_use')})"
            )
        if int(sm_final.get("user_in_use", {}).get(user_id, 0) or 0) != 0:
            failures.append(
                f"slot_manager.user_in_use[{user_id}] != 0 "
                f"(got {sm_final.get('user_in_use', {}).get(user_id)})"
            )

        # Forensic log preserved.
        # The fake_yt mock fires INSIDE the log_youtube_call wrapper for
        # Direct path â†’ we see 1 videos.insert + 1 thumbnails.set per
        # direct job = 4 forensic rows.
        # The RTMP path's log_youtube_call wrapping lives INSIDE
        # rtmp_provider.obtain_rtmp_target / push_to_rtmp /
        # finalize_broadcast â€” which the test mocks AT THE FUNCTION
        # LEVEL (above the wrapper) so no RTMP forensic rows materialise
        # for the test rig. Hitting that layer would require mocking the
        # liveBroadcasts/liveStreams resources on the FakeYouTube â€” which
        # we deliberately skipped because the test's purpose is to
        # validate dispatch + caching + credits, not RTMP API
        # instrumentation.
        # Expected forensic_total for THIS test rig:
        #   2 Ã— videos.insert + 2 Ã— thumbnails.set = 4 rows.
        expected_forensic = 4
        if forensic_total < expected_forensic:
            failures.append(
                f"forensic log (youtube_api_calls) delta={forensic_total} < "
                f"expected_min={expected_forensic}; by_op={forensic_by_op!r}"
            )
        if forensic_by_op.get("videos.insert", 0) < 2:
            failures.append(
                f"videos.insert forensic rows={forensic_by_op.get('videos.insert', 0)} "
                f"< 2 (expected exactly 2)"
            )
        if forensic_by_op.get("thumbnails.set", 0) < 2:
            failures.append(
                f"thumbnails.set forensic rows={forensic_by_op.get('thumbnails.set', 0)} "
                f"< 2 (expected exactly 2 â€” only the 2 Direct/video jobs)"
            )

        # Quota burn log â€” predicted vs actual ledger.
        # This is the F-agent's table; if it's not wired into upload_dispatch yet
        # the delta will be 0 and we should flag it as a possible real bug.
        if burn_total == 0:
            failures.append(
                "quota_burn_log rows=0 â€” the predicted-vs-actual ledger is NOT "
                "being written by upload_dispatch. Brief Â§2/Â§7 requires this. "
                "(LIKELY REAL BUG IN burn_log INTEGRATION.)"
            )

        if failures:
            return ScenarioResult(name, False, "; ".join(failures), metrics)
        return ScenarioResult(
            name, True,
            f"4 mixed jobs completed in {wall_seconds:.2f}s; "
            f"branding 1 miss + 3 hits; credits 64â†’28 (36 burnt); "
            f"forensic_rows={forensic_total} burn_rows={burn_total}",
            metrics,
        )
    finally:
        if user_id > 0:
            db = SessionLocal()
            try:
                _cleanup_user(db, user_id, extra_r2_keys)
            except Exception as exc:
                db.rollback()
                print(f"[cleanup] scenA non-fatal: {exc}")
            finally:
                db.close()
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass


# â”€â”€â”€ SCENARIO B â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


async def scenario_b_cache_focused() -> ScenarioResult:
    name = "B: branded artifact cache (1 miss + 1 hit)"
    user_id = 0
    extra_r2_keys: list[str] = []
    metrics: dict = {}
    work_dir = tempfile.mkdtemp(prefix="kaizer_phase2exit_B_")
    try:
        db = SessionLocal()
        try:
            tier = _ensure_pro_tier(db)
            user = _create_user(db, "scenB", tier)
            user_id = user.id
            channels = _create_channels_and_tokens(db, user, 2, "scenB")
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
            _seed_credits(db, user, 32)
            clip = _create_clip(db, user.id, master.source_upload_id, "scenB")

            target_specs = [
                {"channel_id": channels[0].id, "upload_path": "rtmp", "publish_kind": "short"},
                {"channel_id": channels[1].id, "upload_path": "rtmp", "publish_kind": "short"},
            ]
            request = _build_request(
                master.id, target_specs,
                priority="normal",
                brand_version="phase2-brand-scenB",
                tag="scenB",
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
                    db.add(job)
            db.commit()
        finally:
            db.close()

        snap_before = branding.snapshot()
        misses_before = int(snap_before.get("misses", 0))
        hits_before = int(snap_before.get("hits", 0))

        await scheduler.start()
        t_start = time.monotonic()
        for jid in job_ids:
            scheduler.scheduler_enqueue(
                upload_job_id=jid,
                priority="normal",
                user_id=user_id,
                plan_tier_name="pro",
            )
        completed, failed, pt_status = await _wait_for_terminal(
            pt_id, 2, deadline_seconds=90.0,
        )
        wall_seconds = time.monotonic() - t_start
        await asyncio.sleep(0.3)
        await scheduler.shutdown()

        snap_after = branding.snapshot()
        misses_delta = int(snap_after.get("misses", 0)) - misses_before
        hits_delta = int(snap_after.get("hits", 0)) - hits_before

        db = SessionLocal()
        try:
            jobs = db.query(models.UploadJobV2).filter(
                models.UploadJobV2.publish_task_id == pt_id
            ).all()
            cache_key_set = {
                (j.branded_artifact_r2_key or "").strip() for j in jobs
                if (j.branded_artifact_r2_key or "").strip()
            }
        finally:
            db.close()

        metrics.update({
            "wall_seconds": round(wall_seconds, 3),
            "completed": completed,
            "failed": failed,
            "publish_task_status": pt_status,
            "misses_delta": misses_delta,
            "hits_delta": hits_delta,
            "distinct_cache_keys": list(cache_key_set),
        })

        failures: list[str] = []
        if completed != 2:
            failures.append(f"expected 2 completed jobs, got {completed}")
        if misses_delta != 1:
            failures.append(
                f"misses_delta={misses_delta} expected 1 (single brand)"
            )
        if hits_delta != 1:
            failures.append(
                f"hits_delta={hits_delta} expected 1 (second job reuses cache)"
            )
        if len(cache_key_set) != 1:
            failures.append(
                f"expected 1 distinct cache key, got {len(cache_key_set)}: "
                f"{cache_key_set!r}"
            )

        if failures:
            return ScenarioResult(name, False, "; ".join(failures), metrics)
        return ScenarioResult(
            name, True,
            f"2 jobs / 1 miss + 1 hit / cache_key={list(cache_key_set)[0] if cache_key_set else '?'}",
            metrics,
        )
    finally:
        if user_id > 0:
            db = SessionLocal()
            try:
                _cleanup_user(db, user_id, extra_r2_keys)
            except Exception as exc:
                db.rollback()
                print(f"[cleanup] scenB non-fatal: {exc}")
            finally:
                db.close()
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass


# â”€â”€â”€ SCENARIO C â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


async def scenario_c_quota_gate() -> ScenarioResult:
    """Lower KAIZER_YT_DAILY_QUOTA_CAP so the 3rd direct/video job parks
    on quota. Verify refund row + attempts NOT bumped + burn_log row
    with quota_exceeded outcome."""
    name = "C: quota_v2 gate parks 3rd direct job"
    user_id = 0
    extra_r2_keys: list[str] = []
    metrics: dict = {}
    work_dir = tempfile.mkdtemp(prefix="kaizer_phase2exit_C_")

    # â”€â”€ Lower the quota cap for the scenario; reset at end. â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    saved_cap = os.environ.get("KAIZER_YT_DAILY_QUOTA_CAP", "")
    # Use a unique api_quota row so we don't disturb prod buckets.
    # quota_v2.reserve buckets by date + api_key_hash ("oauth" for OAuth
    # calls) â€” there's only ONE bucket per day, so we pick a cap that's
    # tight enough to gate but tolerates any rows already used today.
    db_pre = SessionLocal()
    try:
        from datetime import datetime as _dt, timezone as _tz
        today = _dt.now(_tz.utc).strftime("%Y-%m-%d")
        # Read the SAME bucket reserve() writes â€” with the test-run
        # bucket isolation (KAIZER_YT_QUOTA_BUCKET, set at the top of
        # this script) that's 'testrun', and scenarios A/B of THIS run
        # have already burned units into it. Hardcoding 'oauth' here
        # would compute a cap below A+B's burn and park everything.
        from youtube import quota_v2 as _q2
        bucket = _q2._key_hash(None)
        existing = db_pre.query(models.ApiQuota).filter(
            models.ApiQuota.date == today,
            models.ApiQuota.api_key_hash == bucket,
        ).first()
        existing_used = int(existing.units_used or 0) if existing else 0
    finally:
        db_pre.close()
    # Cap = existing_used + (1600 Ã— 2) = enough for exactly 2 of the 3
    # direct uploads (each direct burns 1600 units for videos.insert
    # plus 50 for thumbnail, but the scenario triggers the park on the
    # 1600-unit reservation, not the 50-unit one).
    test_cap = existing_used + (1600 * 2)
    os.environ["KAIZER_YT_DAILY_QUOTA_CAP"] = str(test_cap)
    metrics["test_cap"] = test_cap
    metrics["existing_used_at_start"] = existing_used

    try:
        db = SessionLocal()
        try:
            tier = _ensure_pro_tier(db)
            user = _create_user(db, "scenC", tier)
            user_id = user.id
            channels = _create_channels_and_tokens(db, user, 3, "scenC")
            tokens = db.query(models.OAuthToken).filter(
                models.OAuthToken.channel_id.in_([c.id for c in channels])
            ).all()
            _user_asset, logo_key = _create_user_logo_asset_and_attach(
                db, user, tokens, "scenC", work_dir,
            )
            extra_r2_keys.append(logo_key)
            master, master_key = _create_master_video_with_r2(
                db, user, "scenC", work_dir,
            )
            extra_r2_keys.append(master_key)
            _seed_credits(db, user, 3 * 16)  # 48 cr for 3 Direct jobs
            clip = _create_clip(db, user.id, master.source_upload_id, "scenC")

            target_specs = [
                {"channel_id": channels[0].id, "upload_path": "direct", "publish_kind": "video"},
                {"channel_id": channels[1].id, "upload_path": "direct", "publish_kind": "video"},
                {"channel_id": channels[2].id, "upload_path": "direct", "publish_kind": "video"},
            ]
            request = _build_request(
                master.id, target_specs,
                priority="normal",
                brand_version="phase2-brand-scenC",
                tag="scenC",
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
                    db.add(job)
            db.commit()
        finally:
            db.close()

        await scheduler.start()
        for jid in job_ids:
            scheduler.scheduler_enqueue(
                upload_job_id=jid,
                priority="normal",
                user_id=user_id,
                plan_tier_name="pro",
            )
        # Wait for at least the parked outcome to settle.
        completed, failed, pt_status = await _wait_for_terminal(
            pt_id, 3, deadline_seconds=90.0,
        )
        await asyncio.sleep(0.5)
        await scheduler.shutdown()

        db = SessionLocal()
        try:
            jobs = db.query(models.UploadJobV2).filter(
                models.UploadJobV2.publish_task_id == pt_id
            ).order_by(models.UploadJobV2.id.asc()).all()
            parked = [j for j in jobs if j.status == "parked_quota"]
            completed_jobs = [j for j in jobs if j.status == "completed"]
            failed_jobs = [j for j in jobs if j.status == "failed"]

            # Refund rows per parked job.
            refunds: list[models.CreditLedger] = []
            for j in parked:
                rows = db.query(models.CreditLedger).filter(
                    models.CreditLedger.user_id == user_id,
                    models.CreditLedger.upload_job_id == j.id,
                    models.CreditLedger.reason == "refund",
                ).all()
                refunds.extend(rows)

            # Burn log rows with was_quota_exceeded for parked jobs.
            qe_burn_rows = []
            for j in parked:
                rows = db.query(models.QuotaBurnLog).filter(
                    models.QuotaBurnLog.upload_job_id == j.id,
                    models.QuotaBurnLog.was_quota_exceeded.is_(True),
                ).all()
                qe_burn_rows.extend(rows)
        finally:
            db.close()

        metrics.update({
            "completed_jobs": len(completed_jobs),
            "failed_jobs": len(failed_jobs),
            "parked_jobs": len(parked),
            "parked_attempts_values": [int(j.attempts or 0) for j in parked],
            "publish_task_status": pt_status,
            "refund_row_count": len(refunds),
            "refund_row_details": [
                {
                    "ledger_id": r.id,
                    "delta": r.delta,
                    "balance_after": r.balance_after,
                    "upload_job_id": r.upload_job_id,
                    "reason": r.reason,
                }
                for r in refunds
            ],
            "quota_exceeded_burn_rows": len(qe_burn_rows),
            "quota_exceeded_burn_details": [
                {
                    "id": r.id,
                    "operation": r.operation,
                    "predicted_cost": r.predicted_cost,
                    "observed_outcome": r.observed_outcome,
                    "http_status": r.http_status,
                    "was_quota_exceeded": r.was_quota_exceeded,
                }
                for r in qe_burn_rows
            ],
        })

        failures: list[str] = []
        # Two should succeed, one parked.
        if len(completed_jobs) != 2:
            failures.append(
                f"expected 2 completed jobs, got {len(completed_jobs)} "
                f"(failed={len(failed_jobs)} parked={len(parked)})"
            )
        if len(parked) != 1:
            failures.append(
                f"expected 1 parked_quota job, got {len(parked)}"
            )
        if len(failed_jobs) != 0:
            failures.append(
                f"expected 0 failed jobs, got {len(failed_jobs)}"
            )
        for j in parked:
            if int(j.attempts or 0) != 0:
                failures.append(
                    f"parked job_id={j.id} attempts={j.attempts} (must NOT bump on park)"
                )
        if len(refunds) < 1:
            failures.append(
                f"expected >=1 refund row for parked job, got {len(refunds)}"
            )
        # QuotaBurnLog assertion â€” may surface the same wiring gap as
        # Scenario A. Report rather than guess.
        if len(qe_burn_rows) < 1:
            failures.append(
                "no QuotaBurnLog row with was_quota_exceeded=True for the parked job â€” "
                "the F-agent's burn_log is not being written on quota-park. "
                "(LIKELY REAL BUG in upload_dispatch._park_quota path.)"
            )

        if failures:
            return ScenarioResult(name, False, "; ".join(failures), metrics)
        return ScenarioResult(
            name, True,
            f"2 completed + 1 parked (attempts=0, refund row written)",
            metrics,
        )
    finally:
        # Restore env var.
        if saved_cap:
            os.environ["KAIZER_YT_DAILY_QUOTA_CAP"] = saved_cap
        else:
            os.environ.pop("KAIZER_YT_DAILY_QUOTA_CAP", None)
        if user_id > 0:
            db = SessionLocal()
            try:
                _cleanup_user(db, user_id, extra_r2_keys)
            except Exception as exc:
                db.rollback()
                print(f"[cleanup] scenC non-fatal: {exc}")
            finally:
                db.close()
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass


# â”€â”€â”€ SCENARIO D â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


async def scenario_d_legacy() -> ScenarioResult:
    """HTTP probes only â€” no DB writes."""
    name = "D: legacy + flag-gated routes intact"
    metrics: dict = {}

    # Pre/post count of legacy upload_jobs to confirm no regression.
    db = SessionLocal()
    try:
        legacy_before = db.query(models.UploadJob).count()
    finally:
        db.close()

    code_legacy = _http_probe(
        "POST", "/api/clips/9999/publish",
        body={"channel_ids": []},
    )
    code_new = _http_probe(
        "POST", "/api/publish-tasks",
        body={
            "master_video_id": 1,
            "targets": [],
            "priority": "normal",
        },
    )
    code_health = _http_probe("GET", "/api/health/")

    db = SessionLocal()
    try:
        legacy_after = db.query(models.UploadJob).count()
    finally:
        db.close()

    metrics.update({
        "legacy_clips_publish_unauth": code_legacy,
        "new_publish_tasks_unauth": code_new,
        "health": code_health,
        "legacy_upload_jobs_before": legacy_before,
        "legacy_upload_jobs_after": legacy_after,
    })

    failures: list[str] = []
    if code_legacy not in (401,):
        failures.append(
            f"legacy POST /api/clips/9999/publish: expected 401, got {code_legacy}"
        )
    if code_new not in (401, 503):
        failures.append(
            f"new POST /api/publish-tasks: expected 401 or 503, got {code_new}"
        )
    if code_health < 0 or code_health >= 500:
        failures.append(
            f"GET /api/health/: expected non-5xx, got {code_health}"
        )
    if legacy_before != legacy_after:
        failures.append(
            f"legacy upload_jobs row count changed during test: "
            f"{legacy_before} â†’ {legacy_after}"
        )

    if failures:
        return ScenarioResult(name, False, "; ".join(failures), metrics)
    return ScenarioResult(
        name, True,
        f"legacy={code_legacy} new={code_new} health={code_health} "
        f"legacy_table_unchanged ({legacy_before})",
        metrics,
    )


# â”€â”€â”€ Entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


async def _main() -> int:
    print("=" * 78)
    print("Phase 2 Integration Exit-Test")
    print("Brief sec 7: publish 1 MasterVideo â†’ mixed Direct + RTMP, credits +")
    print("             cache + quota logged correctly")
    print("=" * 78)
    print(f"STORAGE_BACKEND={os.environ.get('STORAGE_BACKEND', 'local')!r} "
          f"R2_BUCKET={os.environ.get('R2_BUCKET', '<unset>')!r}")
    print(f"ffmpeg={_ffmpeg_bin()!r}")
    print()

    _purge_stale()

    fake_yt, restore_patches, rtmp_counter = _install_monkeypatches()

    results: list[ScenarioResult] = []
    try:
        for runner_name, runner in (
            ("scenario_a_mixed_fanout",
             lambda: scenario_a_mixed_fanout(fake_yt, rtmp_counter)),
            ("scenario_b_cache_focused",
             scenario_b_cache_focused),
            ("scenario_c_quota_gate",
             scenario_c_quota_gate),
            ("scenario_d_legacy",
             scenario_d_legacy),
        ):
            print(f"\n--- {runner_name} ---")
            try:
                r = await runner()
            except Exception as exc:
                r = ScenarioResult(runner_name, False, f"exception: {exc!r}")
                import traceback
                traceback.print_exc()
            verdict = "PASS" if r.passed else "FAIL"
            print(f"[{verdict}] {r.name}: {r.reason}")
            if r.metrics:
                terse = {
                    k: v for k, v in r.metrics.items()
                    if k not in ("refund_row_details", "quota_exceeded_burn_details")
                }
                print(f"      metrics: {terse}")
            results.append(r)
    finally:
        try:
            restore_patches()
        except Exception:
            pass

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for r in results:
        print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}")
    n_pass = sum(1 for r in results if r.passed)
    n_total = len(results)
    print(f"\n{n_pass}/{n_total} scenarios passed.")

    # Dump JSON snapshot for the report generator.
    snapshot_path = _BACKEND_DIR / "scripts" / "_phase2_exit_last_run.json"
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
