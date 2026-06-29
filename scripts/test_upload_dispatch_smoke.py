"""Phase 2.E Upload dispatch smoke test.

End-to-end exercise of ``services.upload_dispatch.process(...)``
against the real dev Postgres + R2, with the YouTube API surface
heavily monkeypatched (the real one is too expensive to hit on every
local run, and we'd need live OAuth tokens anyway).

Three scenarios are run in sequence:

  1. Direct path, publish_kind='video': videos.insert + thumbnails.set
     - Asserts: status='completed', youtube_video_id matches the
       mocked id, bytes_uploaded > 0, branding ran once (cache miss),
       one YouTubeApiCall row for videos.insert is written, the user's
       credit balance dropped by 16 (Direct Decision 4).

  2. RTMP path, publish_kind='video':
     - Asserts: status='completed', youtube_video_id matches the
       mocked broadcast id, credits dropped by 2 (RTMP Decision 4).

  3. Idempotency variant: process() called a SECOND time on the
     completed Direct job.
     - Asserts: no NEW YouTubeApiCall row for videos.insert; status
       stays 'completed'; credit balance is unchanged. (The 1u
       videos.list probe IS expected — it's exactly the cheap
       confirmation the brief mandates.)

Monkeypatch surface (so the F-agent doesn't break it later):
  - youtube.oauth.get_credentials                  → returns fake Credentials
  - googleapiclient.discovery.build                → returns FakeYouTube
    (used by both uploader_v2._yt and rtmp_provider._yt)
  - youtube.rtmp_provider.obtain_rtmp_target       → returns fake target dict
  - youtube.rtmp_pusher.push_to_rtmp               → no-op success
  - youtube.rtmp_provider.finalize_broadcast       → no-op

Cleanup uses a recognizable email prefix so we can purge stale rows
from prior failed runs without touching production data.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Make ``import models`` / ``import database`` work from the scripts dir.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# Windows cmd.exe defaults to cp1252 which can't print the box-drawing
# chars below; force stdout to UTF-8 so the script runs identically on
# Windows and Linux.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO if os.environ.get("KAIZER_UPLOAD_DISPATCH_VERBOSE") else logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from database import SessionLocal  # noqa: E402
import models  # noqa: E402
from services import branding  # noqa: E402
from services import credits as credits_svc  # noqa: E402

EMAIL_PREFIX = "upload_dispatch_smoke_"
EMAIL_DOMAIN = "@kaizer.test"


# ─── Assertion tracking ─────────────────────────────────────────────


@dataclass
class AssertResult:
    name: str
    passed: bool
    detail: str = ""
    seconds: float = 0.0


_results: list[AssertResult] = []


def _assert(name: str, ok: bool, detail: str = "", t0: Optional[float] = None) -> bool:
    seconds = (time.monotonic() - t0) if t0 is not None else 0.0
    _results.append(AssertResult(name=name, passed=bool(ok), detail=detail, seconds=seconds))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}  ({seconds*1000:.1f} ms)  {detail}")
    return bool(ok)


# ─── ffmpeg helpers ─────────────────────────────────────────────────


def _ffmpeg_bin() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


def _make_test_mp4(out_path: str) -> None:
    """2-second 320x180 red mp4 — ~5-20 KB."""
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


def _make_test_thumb_jpg(out_path: str) -> None:
    cmd = [
        _ffmpeg_bin(), "-y", "-v", "error",
        "-f", "lavfi", "-i", "color=c=green:s=320x180:d=1",
        "-frames:v", "1",
        out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to generate test thumb: {r.stderr[-300:]}")


# ─── Fake YouTube API surface ───────────────────────────────────────


class _FakeMediaProgress:
    def __init__(self, uploaded: int):
        self.resumable_progress = uploaded


class _FakeInsertRequest:
    """Mimics googleapiclient's resumable-upload request: two chunks
    then a final response dict."""

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
        # Snapshot body for assertions.
        self._p.last_insert_body = body
        size = 0
        try:
            stream = media_body.stream()
            if stream is not None:
                size = os.path.getsize(getattr(stream, "name", "")) or 0
        except Exception:
            pass
        return _FakeInsertRequest(total_bytes=size or 100_000, video_id=self._p.fake_video_id)

    def list(self, *, part, id):
        self._p.list_calls += 1

        class _Req:
            def __init__(self, p, vid):
                self._p = p
                self._vid = vid

            def execute(self_inner):
                # Return one item iff fake_video_id matches the prior
                # success — i.e. the idempotency probe finds it.
                if self_inner._vid == self_inner._p.fake_video_id:
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
    """Drop-in replacement for googleapiclient's youtube service."""

    def __init__(self, fake_video_id: str):
        self.fake_video_id = fake_video_id
        self.insert_calls = 0
        self.list_calls = 0
        self.thumb_calls = 0
        self.last_insert_body: Optional[dict] = None
        self._videos = _FakeVideosResource(self)
        self._thumbnails = _FakeThumbnailsResource(self)

    def videos(self):
        return self._videos

    def thumbnails(self):
        return self._thumbnails

    # liveBroadcasts/liveStreams unused — rtmp_provider is monkeypatched
    # at a higher level (no real client is constructed for RTMP path).


# ─── Test artifacts ─────────────────────────────────────────────────


@dataclass
class TestArtifacts:
    user_id: int = 0
    channel_id: int = 0
    oauth_token_id: int = 0
    user_asset_id: int = 0
    job_row_id: int = 0
    master_video_id: int = 0
    clip_id: int = 0
    publish_task_id: int = 0
    direct_upload_job_id: int = 0
    rtmp_upload_job_id: int = 0
    master_r2_key: str = ""
    logo_r2_key: str = ""
    thumb_r2_key: str = ""
    branded_cache_keys: list[str] = field(default_factory=list)
    work_dir: str = ""


def _provider():
    from pipeline_core.storage import get_storage_provider
    return get_storage_provider()


def _ensure_pro_tier(db):
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


def _seed_user_balance(db, user_id: int, credits_amount: int) -> None:
    """Seed the credit_ledger so reserve() / refund() bookkeeping has
    a starting balance to work against."""
    row = models.CreditLedger(
        user_id=user_id,
        delta=credits_amount,
        reason="admin_adjustment",
        upload_job_id=None,
        path=None,
        publish_kind=None,
        predicted_quota_units=None,
        balance_after=credits_amount,
    )
    db.add(row)
    db.commit()


def _setup(db, art: TestArtifacts) -> None:
    plan = _ensure_pro_tier(db)
    tag = uuid.uuid4().hex[:12]
    art.work_dir = tempfile.mkdtemp(prefix="kaizer_upload_dispatch_smoke_")

    # ── 1. User ──
    user = models.User(
        email=f"{EMAIL_PREFIX}{tag}{EMAIL_DOMAIN}",
        name="Upload Dispatch Smoke",
        plan="pro",
        plan_tier_id=plan.id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    art.user_id = user.id
    _seed_user_balance(db, user.id, 500)  # plenty of credits for both jobs

    # ── 2. Channel ──
    channel = models.Channel(
        user_id=user.id,
        name=f"upload-dispatch-smoke-{tag}",
        language="te",
        watermark_text="@kaizer.test",
        watermark_opacity=0.35,
        watermark_position="lower-center",
    )
    db.add(channel)
    db.commit()
    db.refresh(channel)
    art.channel_id = channel.id

    # ── 3. OAuth token (refresh_token_enc is empty — we monkeypatch get_credentials) ──
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

    # ── 4. Logo + UserAsset + R2 upload (precedence #1: oauth_token) ──
    logo_local = os.path.join(art.work_dir, "logo.png")
    _make_test_logo_png(logo_local)
    logo_key = f"upload_dispatch_smoke/{tag}/logo.png"
    _provider().upload(logo_local, logo_key, content_type="image/png")
    art.logo_r2_key = logo_key

    user_asset = models.UserAsset(
        user_id=user.id,
        filename="logo.png",
        file_path=logo_local,
        kind="logo",
        mime="image/png",
        size_bytes=os.path.getsize(logo_local),
        width=64, height=64,
        storage_key=logo_key,
        storage_backend=os.environ.get("STORAGE_BACKEND", "local").lower(),
        storage_url=f"key:{logo_key}",
    )
    db.add(user_asset)
    db.commit()
    db.refresh(user_asset)
    art.user_asset_id = user_asset.id

    tok.logo_asset_id = user_asset.id
    db.add(tok)
    db.commit()

    # ── 5. Thumbnail asset (for Direct/Video to invoke thumbnails.set) ──
    thumb_local = os.path.join(art.work_dir, "thumb.jpg")
    _make_test_thumb_jpg(thumb_local)
    thumb_key = f"upload_dispatch_smoke/{tag}/thumb.jpg"
    _provider().upload(thumb_local, thumb_key, content_type="image/jpeg")
    art.thumb_r2_key = thumb_key

    # ── 6. Master mp4 + R2 upload + MasterVideo row ──
    master_local = os.path.join(art.work_dir, "master.mp4")
    _make_test_mp4(master_local)
    master_key = f"upload_dispatch_smoke/{tag}/master.mp4"
    _provider().upload(master_local, master_key, content_type="video/mp4")
    art.master_r2_key = master_key

    job_row = models.Job(
        user_id=user.id,
        status="done",
        platform="youtube_full",
        video_name=f"upload_dispatch_smoke_{tag}.mp4",
    )
    db.add(job_row)
    db.commit()
    db.refresh(job_row)
    art.job_row_id = job_row.id

    master = models.MasterVideo(
        source_upload_id=job_row.id,
        r2_key=master_key,
        duration_seconds=2.0,
        bytes=os.path.getsize(master_local),
        width=320, height=180,
        status="ready",
        pipeline_version="v4_clean",
        clean_master=True,
    )
    db.add(master)
    db.commit()
    db.refresh(master)
    art.master_video_id = master.id

    # ── 7. Clip (legacy bridge) — carries SEO JSON so the composer can
    # synthesize title/description/tags. UploadJobV2 does NOT persist
    # those columns directly; the composer recomputes per dispatch.
    import json as _json
    clip = models.Clip(
        job_id=job_row.id,
        clip_index=0,
        filename=f"smoke_{tag}.mp4",
        file_path="",
        duration=2.0,
        text=f"Smoke Test Bulletin {tag}",
        seo=_json.dumps({
            "title": f"Smoke Test Bulletin {tag}",
            "description": "Smoke test description body.",
            "keywords": ["smoke", "kaizer"],
            "hashtags": ["smoke", "kaizer"],
            "hook": "Smoke test hook",
        }),
    )
    db.add(clip)
    db.commit()
    db.refresh(clip)
    # Stash on the artifacts struct so cleanup can drop it.
    art.clip_id = clip.id  # type: ignore[attr-defined]

    # ── 8. PublishTask + two UploadJobV2 rows (Direct + RTMP) ──
    pt = models.PublishTask(
        user_id=user.id,
        master_video_id=master.id,
        priority="normal",
        status="dispatched",
        target_count=2,
        completed_count=0,
        failed_count=0,
    )
    db.add(pt)
    db.commit()
    db.refresh(pt)
    art.publish_task_id = pt.id

    direct_idem = f"smoke-direct-{tag}-{uuid.uuid4().hex[:16]}"
    direct = models.UploadJobV2(
        publish_task_id=pt.id,
        clip_id=clip.id,
        channel_id=channel.id,
        oauth_token_id=tok.id,
        upload_path="direct",
        publish_kind="video",
        thumbnail_source="user_uploaded",
        thumbnail_r2_key=thumb_key,
        status="queued",
        attempts=0,
        idempotency_key=direct_idem,
        publish_version=f"smoke:v1:{tag[:6]}",
        predicted_quota_units=1650,
        predicted_credit_cost=16,
        bytes_uploaded=0,
    )
    db.add(direct)
    db.commit()
    db.refresh(direct)
    art.direct_upload_job_id = direct.id

    rtmp_idem = f"smoke-rtmp-{tag}-{uuid.uuid4().hex[:16]}"
    rtmp = models.UploadJobV2(
        publish_task_id=pt.id,
        clip_id=clip.id,
        channel_id=channel.id,
        oauth_token_id=tok.id,
        upload_path="rtmp",
        publish_kind="video",
        thumbnail_source="user_uploaded",
        thumbnail_r2_key=thumb_key,
        status="queued",
        attempts=0,
        idempotency_key=rtmp_idem,
        publish_version=f"smoke:v2:{tag[:6]}",
        predicted_quota_units=200,
        predicted_credit_cost=2,
        bytes_uploaded=0,
    )
    db.add(rtmp)
    db.commit()
    db.refresh(rtmp)
    art.rtmp_upload_job_id = rtmp.id


# ─── Cleanup ────────────────────────────────────────────────────────


def _cleanup(db, art: TestArtifacts) -> None:
    provider = _provider()
    for k in (
        [art.master_r2_key, art.logo_r2_key, art.thumb_r2_key]
        + list(art.branded_cache_keys)
    ):
        if not k:
            continue
        try:
            provider.delete(k)
        except Exception as exc:
            print(f"[cleanup] R2 delete failed for {k!r}: {exc}")

    try:
        # FK-safe order.
        if art.user_id:
            db.query(models.CreditLedger).filter(
                models.CreditLedger.user_id == art.user_id
            ).delete(synchronize_session=False)
        for uid in (art.direct_upload_job_id, art.rtmp_upload_job_id):
            if uid:
                db.query(models.YouTubeApiCall).filter(
                    models.YouTubeApiCall.upload_job_id == uid
                ).delete(synchronize_session=False)
                db.query(models.PublishAttempt).filter(
                    models.PublishAttempt.upload_job_id == uid
                ).delete(synchronize_session=False)
                # F-agent's predicted-vs-actual ledger (Phase 2.F).
                # quota_burn_log.upload_job_id FKs upload_jobs_v2 with
                # ondelete='SET NULL'; cleaning these rows keeps the
                # table tidy across re-runs.
                db.query(models.QuotaBurnLog).filter(
                    models.QuotaBurnLog.upload_job_id == uid
                ).delete(synchronize_session=False)
                db.query(models.UploadJobV2).filter(
                    models.UploadJobV2.id == uid
                ).delete(synchronize_session=False)
        if art.publish_task_id:
            db.query(models.PublishTask).filter(
                models.PublishTask.id == art.publish_task_id
            ).delete(synchronize_session=False)
        if art.clip_id:
            db.query(models.Clip).filter(
                models.Clip.id == art.clip_id
            ).delete(synchronize_session=False)
        if art.master_video_id:
            db.query(models.MasterVideo).filter(
                models.MasterVideo.id == art.master_video_id
            ).delete(synchronize_session=False)
        if art.job_row_id:
            db.query(models.Job).filter(
                models.Job.id == art.job_row_id
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
            db.query(models.User).filter(
                models.User.id == art.user_id
            ).delete(synchronize_session=False)
        db.commit()
    except Exception as exc:
        print(f"[cleanup] DB delete failed: {exc}")
        db.rollback()

    if art.work_dir and os.path.isdir(art.work_dir):
        try:
            shutil.rmtree(art.work_dir, ignore_errors=True)
        except Exception:
            pass


def _purge_stale() -> None:
    """Belt-and-braces — wipe any leftover smoke users from earlier runs."""
    db = SessionLocal()
    try:
        users = db.query(models.User).filter(
            models.User.email.like(f"{EMAIL_PREFIX}%{EMAIL_DOMAIN}")
        ).all()
        for u in users:
            try:
                # Cascade by hand — the order matters because UploadJobV2
                # rows are FK'd from credit_ledger and youtube_api_calls.
                pt_ids = [
                    p.id for p in db.query(models.PublishTask).filter(
                        models.PublishTask.user_id == u.id
                    ).all()
                ]
                uj_ids = [
                    j.id for j in db.query(models.UploadJobV2).filter(
                        models.UploadJobV2.publish_task_id.in_(pt_ids or [-1])
                    ).all()
                ]
                if uj_ids:
                    db.query(models.YouTubeApiCall).filter(
                        models.YouTubeApiCall.upload_job_id.in_(uj_ids)
                    ).delete(synchronize_session=False)
                    db.query(models.PublishAttempt).filter(
                        models.PublishAttempt.upload_job_id.in_(uj_ids)
                    ).delete(synchronize_session=False)
                    db.query(models.QuotaBurnLog).filter(
                        models.QuotaBurnLog.upload_job_id.in_(uj_ids)
                    ).delete(synchronize_session=False)
                db.query(models.CreditLedger).filter(
                    models.CreditLedger.user_id == u.id
                ).delete(synchronize_session=False)
                if uj_ids:
                    db.query(models.UploadJobV2).filter(
                        models.UploadJobV2.id.in_(uj_ids)
                    ).delete(synchronize_session=False)
                if pt_ids:
                    db.query(models.PublishTask).filter(
                        models.PublishTask.id.in_(pt_ids)
                    ).delete(synchronize_session=False)
                mvs = db.query(models.MasterVideo).join(
                    models.Job, models.MasterVideo.source_upload_id == models.Job.id
                ).filter(models.Job.user_id == u.id).all()
                for mv in mvs:
                    db.delete(mv)
                jobs = db.query(models.Job).filter(models.Job.user_id == u.id).all()
                for j in jobs:
                    db.delete(j)
                assets = db.query(models.UserAsset).filter(
                    models.UserAsset.user_id == u.id
                ).all()
                for a in assets:
                    db.delete(a)
                chs = db.query(models.Channel).filter(
                    models.Channel.user_id == u.id
                ).all()
                for ch in chs:
                    toks = db.query(models.OAuthToken).filter(
                        models.OAuthToken.channel_id == ch.id
                    ).all()
                    for t in toks:
                        db.delete(t)
                    db.delete(ch)
                db.delete(u)
                db.commit()
            except Exception as exc:
                db.rollback()
                print(f"[purge_stale] skipping user={u.id}: {exc}")
    finally:
        db.close()


# ─── Monkeypatches ──────────────────────────────────────────────────


def _install_monkeypatches(fake_video_id: str, fake_broadcast_id: str) -> tuple:
    """Returns (fake_youtube, restore_callable, rtmp_call_counter)."""
    # 1) youtube.oauth.get_credentials — return a fake Credentials.
    from youtube import oauth as _oauth_mod
    original_get_creds = _oauth_mod.get_credentials

    class _FakeCreds:
        valid = True
        token = "FAKE_ACCESS"
        refresh_token = "FAKE_REFRESH"

    def _fake_get_credentials(db, channel_id):
        return _FakeCreds()

    _oauth_mod.get_credentials = _fake_get_credentials  # type: ignore

    # 2) googleapiclient.discovery.build — return our FakeYouTube.
    fake_yt = _FakeYouTube(fake_video_id=fake_video_id)
    import googleapiclient.discovery as _gad
    original_build = _gad.build

    def _fake_build(serviceName, version, **kwargs):
        return fake_yt

    _gad.build = _fake_build  # type: ignore

    # Patch the imported reference inside uploader_v2 too (already
    # imports `from googleapiclient.discovery import build`).
    from youtube import uploader_v2 as _u2
    original_u2_build = getattr(_u2, "build", None)
    _u2.build = _fake_build  # type: ignore

    # 3) RTMP path stubs — call counter so we can assert the mint
    # was invoked.
    rtmp_counter = {"obtain": 0, "push": 0, "finalize": 0}

    from youtube import rtmp_provider as _rp
    from youtube import rtmp_pusher as _rp_pusher

    original_obtain = _rp.obtain_rtmp_target
    original_finalize = _rp.finalize_broadcast
    original_push = _rp_pusher.push_to_rtmp

    def _fake_obtain(creds, *, job, channel, title, description="", privacy_status="private",
                    scheduled_start=None, enable_auto_start=True, enable_auto_stop=True, db=None):
        rtmp_counter["obtain"] += 1
        return {
            "broadcast_id": fake_broadcast_id,
            "stream_id": "FAKE_STREAM",
            "ingest_url": "rtmps://test.example/live2",
            "stream_key": "FAKE-KEY-0000",
            "video_id": fake_broadcast_id,
        }

    def _fake_finalize(creds, *, job, channel, broadcast_id, thumbnail_path=None, db=None):
        rtmp_counter["finalize"] += 1
        return None

    def _fake_push(*, input_path, ingest_url, stream_key, expected_duration_s,
                   progress_cb=None, cancel_event=None, log_prefix=""):
        rtmp_counter["push"] += 1
        if progress_cb is not None:
            try:
                progress_cb(expected_duration_s, expected_duration_s)
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


# ─── Assertion helpers ──────────────────────────────────────────────


def _count_burn_log_rows(
    db, upload_job_id: int, operation: Optional[str] = None,
) -> int:
    """Count predicted-vs-actual ledger rows (quota_burn_log) for an
    UploadJobV2.

    Unlike ``youtube_api_calls.upload_job_id`` (which FKs the LEGACY
    table — see _count_api_calls), ``quota_burn_log.upload_job_id`` FKs
    ``upload_jobs_v2.id`` directly (models.py:1747), so we filter on
    the real v2 id here.
    """
    q = db.query(models.QuotaBurnLog).filter(
        models.QuotaBurnLog.upload_job_id == upload_job_id,
    )
    if operation is not None:
        q = q.filter(models.QuotaBurnLog.operation == operation)
    return q.count()


def _count_api_calls(
    db, upload_job_id: int, operation: str,
    *, user_id: Optional[int] = None, video_id: Optional[str] = None,
) -> int:
    """Count forensic-log rows for an operation.

    ``youtube_api_calls.upload_job_id`` FK's the LEGACY ``upload_jobs``
    table, not ``upload_jobs_v2`` (models.py:1115). The dispatch passes
    ``upload_job_id=None`` to ``log_youtube_call`` so the row writes
    succeed; we identify by user_id + operation (+ optional video_id)
    instead. The forensic log itself is still preserved (one row per
    real API call) — only the attribution column is NULL until the
    A-agent migrates the FK.
    """
    q = db.query(models.YouTubeApiCall).filter(
        models.YouTubeApiCall.operation == operation,
    )
    if user_id is not None:
        q = q.filter(models.YouTubeApiCall.user_id == user_id)
    if video_id is not None:
        q = q.filter(models.YouTubeApiCall.video_id == video_id)
    return q.count()


# ─── Main ───────────────────────────────────────────────────────────


def main() -> int:
    print("\n── Phase 2.E Upload dispatch smoke test ──")
    print(f"STORAGE_BACKEND={os.environ.get('STORAGE_BACKEND', 'local')!r} "
          f"R2_BUCKET={os.environ.get('R2_BUCKET', '<unset>')!r}")
    print(f"ffmpeg={_ffmpeg_bin()!r}\n")

    print("[setup] purging stale smoke users")
    _purge_stale()

    fake_video_id = "FAKE_VIDEO_ID_123"
    fake_broadcast_id = "FAKE_BCAST_456"
    fake_yt, restore_patches, rtmp_counter = _install_monkeypatches(
        fake_video_id, fake_broadcast_id,
    )

    db = SessionLocal()
    art = TestArtifacts()
    overall_ok = False
    try:
        try:
            _setup(db, art)
        except Exception as exc:
            _assert("setup", False, f"raised: {exc!r}")
            raise
        _assert(
            "setup", True,
            f"user_id={art.user_id} direct_job={art.direct_upload_job_id} "
            f"rtmp_job={art.rtmp_upload_job_id}",
        )

        # Re-fetch fresh from DB.
        balance_before_all = credits_svc.get_balance(db, art.user_id)
        _assert(
            "starting_credit_balance_500",
            balance_before_all == 500,
            f"got {balance_before_all}",
        )

        # ── Scenario 1: Direct path, publish_kind=video ────────────
        from services import upload_dispatch

        # Simulate Fanout's pre-deduction (production Fanout deducts at
        # PublishTask creation; upload_dispatch only refunds on error).
        # Without this, dispatch's success path is a no-op on credits,
        # which is correct behaviour — we just need to see the deduction
        # reflected in the balance for the e2e assertion.
        credits_svc.reserve(
            db, user_id=art.user_id, cost=16, reason="upload_direct",
            upload_job_id=art.direct_upload_job_id,
            path="direct", publish_kind="video",
            predicted_quota_units=1650,
        )
        db.commit()

        snap_before = branding.snapshot()
        t0 = time.monotonic()
        try:
            upload_dispatch.process(art.direct_upload_job_id)
        except Exception as exc:
            _assert("direct_process_no_raise", False, f"raised: {exc!r}", t0=t0)
            raise
        _assert("direct_process_no_raise", True, "process() returned cleanly", t0=t0)

        db.expire_all()
        direct_row = db.query(models.UploadJobV2).filter(
            models.UploadJobV2.id == art.direct_upload_job_id
        ).first()
        _assert(
            "direct_status_completed",
            direct_row is not None and direct_row.status == "completed",
            f"status={getattr(direct_row, 'status', None)!r}",
        )
        _assert(
            "direct_youtube_video_id_matches",
            direct_row is not None and direct_row.youtube_video_id == fake_video_id,
            f"youtube_video_id={getattr(direct_row, 'youtube_video_id', None)!r}",
        )
        _assert(
            "direct_bytes_uploaded_positive",
            direct_row is not None and (direct_row.bytes_uploaded or 0) > 0,
            f"bytes_uploaded={getattr(direct_row, 'bytes_uploaded', None)}",
        )
        _assert(
            "direct_branded_artifact_set",
            direct_row is not None and bool(direct_row.branded_artifact_r2_key),
            f"branded_artifact_r2_key={getattr(direct_row, 'branded_artifact_r2_key', None)!r}",
        )
        if direct_row and direct_row.branded_artifact_r2_key:
            art.branded_cache_keys.append(direct_row.branded_artifact_r2_key)

        snap_after_direct = branding.snapshot()
        misses_delta = snap_after_direct["misses"] - snap_before.get("misses", 0)
        _assert(
            "direct_branding_cache_miss_once",
            misses_delta == 1,
            f"misses_delta={misses_delta}",
        )

        videos_insert_rows = _count_api_calls(
            db, art.direct_upload_job_id, "videos.insert",
            user_id=art.user_id, video_id=fake_video_id,
        )
        _assert(
            "direct_one_videos_insert_log_row",
            videos_insert_rows == 1,
            f"videos.insert rows={videos_insert_rows}",
        )

        # videos.insert quota event AND a thumbnails.set row (publish_kind=video).
        thumb_rows = _count_api_calls(
            db, art.direct_upload_job_id, "thumbnails.set",
            user_id=art.user_id, video_id=fake_video_id,
        )
        _assert(
            "direct_thumbnails_set_called",
            thumb_rows >= 1 and fake_yt.thumb_calls >= 1,
            f"thumb_rows={thumb_rows} fake_yt.thumb_calls={fake_yt.thumb_calls}",
        )

        # ── F-agent burn_log assertions (Direct variant) ──
        # Brief §2 + §7 mandate predicted-vs-actual rows for every
        # YouTube API call. Direct/video = 1 videos.insert + 1
        # thumbnails.set burn row, all success.
        burn_videos_insert = _count_burn_log_rows(
            db, art.direct_upload_job_id, "videos.insert",
        )
        _assert(
            "direct_burn_log_videos_insert_row",
            burn_videos_insert == 1,
            f"burn_log videos.insert rows={burn_videos_insert} (expected 1)",
        )
        burn_thumbnails_set = _count_burn_log_rows(
            db, art.direct_upload_job_id, "thumbnails.set",
        )
        _assert(
            "direct_burn_log_thumbnails_set_row",
            burn_thumbnails_set == 1,
            f"burn_log thumbnails.set rows={burn_thumbnails_set} (expected 1)",
        )

        balance_after_direct = credits_svc.get_balance(db, art.user_id)
        _assert(
            "direct_credit_balance_dropped_by_16",
            balance_after_direct == balance_before_all - 16,
            f"before={balance_before_all} after={balance_after_direct} (Decision 4)",
        )

        # ── Scenario 2: RTMP path, publish_kind=video ─────────────
        # Simulate Fanout's pre-deduction for the RTMP target (2 cr).
        credits_svc.reserve(
            db, user_id=art.user_id, cost=2, reason="upload_rtmp",
            upload_job_id=art.rtmp_upload_job_id,
            path="rtmp", publish_kind="video",
            predicted_quota_units=200,
        )
        db.commit()

        snap_before_rtmp = branding.snapshot()
        t0 = time.monotonic()
        try:
            upload_dispatch.process(art.rtmp_upload_job_id)
        except Exception as exc:
            _assert("rtmp_process_no_raise", False, f"raised: {exc!r}", t0=t0)
            raise
        _assert("rtmp_process_no_raise", True, "process() returned cleanly", t0=t0)

        db.expire_all()
        rtmp_row = db.query(models.UploadJobV2).filter(
            models.UploadJobV2.id == art.rtmp_upload_job_id
        ).first()
        _assert(
            "rtmp_status_completed",
            rtmp_row is not None and rtmp_row.status == "completed",
            f"status={getattr(rtmp_row, 'status', None)!r}",
        )
        _assert(
            "rtmp_youtube_video_id_matches_broadcast",
            rtmp_row is not None and rtmp_row.youtube_video_id == fake_broadcast_id,
            f"youtube_video_id={getattr(rtmp_row, 'youtube_video_id', None)!r}",
        )
        if rtmp_row and rtmp_row.branded_artifact_r2_key:
            art.branded_cache_keys.append(rtmp_row.branded_artifact_r2_key)

        _assert(
            "rtmp_obtain_target_called",
            rtmp_counter["obtain"] == 1,
            f"rtmp_counter={rtmp_counter}",
        )
        _assert(
            "rtmp_push_called",
            rtmp_counter["push"] == 1,
            f"rtmp_counter={rtmp_counter}",
        )

        # ── F-agent burn_log assertions (RTMP variant) ──
        # The RTMP triple-call (mint broadcast + stream + bind) writes
        # 3 separate burn_log rows — one per logical operation — so the
        # predicted-vs-actual delta stays attributable per-operation.
        # Plus 1 thumbnails.set row for the Full Video case.
        burn_lb_insert = _count_burn_log_rows(
            db, art.rtmp_upload_job_id, "liveBroadcasts.insert",
        )
        burn_ls_insert = _count_burn_log_rows(
            db, art.rtmp_upload_job_id, "liveStreams.insert",
        )
        burn_lb_bind = _count_burn_log_rows(
            db, art.rtmp_upload_job_id, "liveBroadcasts.bind",
        )
        _assert(
            "rtmp_burn_log_triple_rows",
            burn_lb_insert == 1 and burn_ls_insert == 1 and burn_lb_bind == 1,
            f"liveBroadcasts.insert={burn_lb_insert} liveStreams.insert={burn_ls_insert} "
            f"liveBroadcasts.bind={burn_lb_bind} (each expected 1)",
        )
        burn_rtmp_thumb = _count_burn_log_rows(
            db, art.rtmp_upload_job_id, "thumbnails.set",
        )
        _assert(
            "rtmp_burn_log_thumbnails_set_row",
            burn_rtmp_thumb == 1,
            f"burn_log thumbnails.set rows={burn_rtmp_thumb} (expected 1)",
        )

        balance_after_rtmp = credits_svc.get_balance(db, art.user_id)
        _assert(
            "rtmp_credit_balance_dropped_by_2",
            balance_after_rtmp == balance_after_direct - 2,
            f"before_rtmp={balance_after_direct} after_rtmp={balance_after_rtmp} (Decision 4)",
        )

        # ── Scenario 3: Idempotency short-circuit ─────────────────
        insert_rows_before_2nd = _count_api_calls(
            db, art.direct_upload_job_id, "videos.insert",
            user_id=art.user_id, video_id=fake_video_id,
        )
        list_rows_before_2nd = _count_api_calls(
            db, art.direct_upload_job_id, "videos.list",
            user_id=art.user_id, video_id=fake_video_id,
        )
        thumb_rows_before_2nd = _count_api_calls(
            db, art.direct_upload_job_id, "thumbnails.set",
            user_id=art.user_id, video_id=fake_video_id,
        )
        balance_before_2nd = credits_svc.get_balance(db, art.user_id)

        try:
            upload_dispatch.process(art.direct_upload_job_id)
        except Exception as exc:
            _assert("idempotency_process_no_raise", False, f"raised: {exc!r}")
            raise
        _assert("idempotency_process_no_raise", True, "process() returned cleanly")

        db.expire_all()
        direct_row2 = db.query(models.UploadJobV2).filter(
            models.UploadJobV2.id == art.direct_upload_job_id
        ).first()
        _assert(
            "idempotency_status_still_completed",
            direct_row2 is not None and direct_row2.status == "completed",
            f"status={getattr(direct_row2, 'status', None)!r}",
        )

        # NB: after-counts filter on the same (user_id, video_id) keys as
        # the before-counts so pre-existing rows from real production
        # uploads (which live in the same youtube_api_calls table) don't
        # confound the diff. The brief assertion is "no NEW row for this
        # user's fake video on the 2nd process() call".
        insert_rows_after = _count_api_calls(
            db, art.direct_upload_job_id, "videos.insert",
            user_id=art.user_id, video_id=fake_video_id,
        )
        _assert(
            "idempotency_no_new_videos_insert",
            insert_rows_after == insert_rows_before_2nd,
            f"before={insert_rows_before_2nd} after={insert_rows_after}",
        )

        list_rows_after = _count_api_calls(
            db, art.direct_upload_job_id, "videos.list",
            user_id=art.user_id, video_id=fake_video_id,
        )
        _assert(
            "idempotency_videos_list_called_not_search",
            list_rows_after > list_rows_before_2nd,
            f"before={list_rows_before_2nd} after={list_rows_after} (1u confirmation)",
        )

        search_rows = _count_api_calls(
            db, art.direct_upload_job_id, "search.list",
            user_id=art.user_id, video_id=fake_video_id,
        )
        _assert(
            "idempotency_no_search_list_ever",
            search_rows == 0,
            f"search.list rows={search_rows} (must be zero — brief §9)",
        )

        thumb_rows_after = _count_api_calls(
            db, art.direct_upload_job_id, "thumbnails.set",
            user_id=art.user_id, video_id=fake_video_id,
        )
        _assert(
            "idempotency_no_new_thumbnails_set",
            thumb_rows_after == thumb_rows_before_2nd,
            f"before={thumb_rows_before_2nd} after={thumb_rows_after}",
        )

        balance_after_2nd = credits_svc.get_balance(db, art.user_id)
        _assert(
            "idempotency_no_credit_change",
            balance_after_2nd == balance_before_2nd,
            f"before={balance_before_2nd} after={balance_after_2nd}",
        )

        # ── Summary ───────────────────────────────────────────────
        total = len(_results)
        passes = sum(1 for r in _results if r.passed)
        print(f"\n── Result: {passes}/{total} assertions PASS ──")
        print(f"YouTube videos.insert calls (fake): {fake_yt.insert_calls}")
        print(f"YouTube videos.list   calls (fake): {fake_yt.list_calls}")
        print(f"YouTube thumbnails.set calls (fake): {fake_yt.thumb_calls}")
        print(f"RTMP counters: {rtmp_counter}\n")

        overall_ok = (passes == total)
    finally:
        try:
            restore_patches()
        except Exception:
            pass
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
