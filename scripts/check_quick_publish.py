"""Quick Publish end-to-end smoke (no real YouTube calls).

Exercises against the LIVE backend over HTTP:
  1. ffmpeg lavfi → tiny 2s test mp4 (with audio)
  2. POST /api/clips/raw-upload/        → Job + Clip
  3. GET  quick-state                   → wizard resume shape
  4. POST quick-seo mode=manual         → persisted SEO
  5. POST quick-thumbnail/upload        → meta.publish_thumbnail_key set
  6. POST /api/clips/{id}/publish (fixture channel) → asserts the FIXED
     MasterVideo synthesis: clean_master=True, r2_key == clip.storage_key,
     and the UploadJobV2 carries thumbnail_source='user_uploaded' with
     our key. The live durable worker then claims it and fails safely at
     OAuth minting (fixture token has no refresh token) — we assert it
     reaches a terminal/retrying state WITHOUT touching YouTube.
  7. Cleanup of every fixture row.

Run from KaizerBackend/ with the stack up:
  python scripts/check_quick_publish.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import httpx  # noqa: E402
from sqlalchemy import text  # noqa: E402

import auth  # noqa: E402
import models  # noqa: E402
from database import SessionLocal  # noqa: E402

BASE = "http://localhost:8000"
NONCE = uuid.uuid4().hex[:8]
PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name
          + (f"  [{detail}]" if detail and not cond else ""))


def main() -> int:
    db = SessionLocal()
    created: dict = {}
    tmp = tempfile.mkdtemp(prefix="kaizer_qp_smoke_")
    try:
        # ── Fixtures: user (pro) + channel + oauth token ─────────────
        tier = db.query(models.PlanTier).filter(
            models.PlanTier.name == "pro").first()
        user = models.User(email=f"qp_{NONCE}@test.kaizer",
                           name=f"qp-{NONCE}", is_active=True,
                           plan_tier_id=(tier.id if tier else None))
        db.add(user); db.flush()
        ch = models.Channel(user_id=user.id, name=f"qp-ch-{NONCE}",
                            upload_provider="native_rtmp")
        db.add(ch); db.flush()
        tok = models.OAuthToken(channel_id=ch.id,
                                google_channel_id=f"UCQP{NONCE}",
                                # Non-empty so the publish route's
                                # "linked to YouTube" gate passes; the
                                # worker's credential decrypt still
                                # fails SAFELY (OAuthTokenMissing →
                                # terminal fail + refund, no YT call).
                                refresh_token_enc="smoke-fake-token")
        db.add(tok); db.flush()
        # Seed credits so fanout's reserve passes.
        db.execute(text(
            "INSERT INTO credit_ledger (user_id, delta, reason, "
            "balance_after, created_at) VALUES (:u, 100, "
            "'admin_adjustment', 100, CURRENT_TIMESTAMP)"), {"u": user.id})
        db.commit()
        created.update(user_id=user.id, ch_id=ch.id, tok_id=tok.id)
        jwt = auth.issue_token(user)
        H = {"Authorization": f"Bearer {jwt}"}
        print(f"fixtures ready (user={user.id}, channel={ch.id})")

        # ── 1) tiny test video ───────────────────────────────────────
        clip_path = os.path.join(tmp, "test.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi",
             "-i", "testsrc=size=1280x720:rate=30:duration=2",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
             "-c:v", "libx264", "-preset", "ultrafast",
             "-c:a", "aac", "-shortest", clip_path],
            capture_output=True, timeout=120, check=True)
        thumb_path = os.path.join(tmp, "thumb.jpg")
        subprocess.run(
            ["ffmpeg", "-y", "-i", clip_path, "-frames:v", "1",
             "-q:v", "3", thumb_path],
            capture_output=True, timeout=60, check=True)

        with httpx.Client(timeout=180) as c:
            # ── 2) raw-upload ────────────────────────────────────────
            with open(clip_path, "rb") as f:
                r = c.post(f"{BASE}/api/clips/raw-upload/", headers=H,
                           files={"video": ("test.mp4", f, "video/mp4")},
                           data={"title": f"QP smoke {NONCE}",
                                 "platform": "youtube_full",
                                 "language": "te"})
            check("raw-upload 200", r.status_code == 200, r.text[:200])
            up = r.json()
            clip_id = int(up["clip_id"])
            created["clip_id"] = clip_id
            created["job_id"] = int(up["job_id"])
            print(f"  uploaded clip_id={clip_id} job_id={up['job_id']} "
                  f"duration={up.get('duration')}")

            # ── 3) quick-state ───────────────────────────────────────
            r = c.get(f"{BASE}/api/clips/{clip_id}/quick-state", headers=H)
            check("quick-state 200", r.status_code == 200, r.text[:200])
            st = r.json()
            check("state has no SEO yet", st.get("seo") is None)
            check("state platform/language",
                  st.get("platform") == "youtube_full"
                  and st.get("language") == "te", str(st)[:200])

            # ── 4) quick-seo manual ──────────────────────────────────
            r = c.post(f"{BASE}/api/clips/{clip_id}/quick-seo", headers=H,
                       json={"mode": "manual", "language": "te",
                             "title": f"Smoke Title {NONCE}",
                             "description": "Manual description body",
                             "tags": ["news", "smoke"],
                             "hashtags": ["Breaking", "#Kaizer"]})
            check("quick-seo manual 200", r.status_code == 200, r.text[:300])
            seo = r.json().get("seo") or {}
            check("seo persisted shape",
                  seo.get("title", "").startswith("Smoke Title")
                  and seo.get("hashtags") == ["#Breaking", "#Kaizer"]
                  and seo.get("edited_by_user") is True, str(seo)[:200])

            # ── 5) thumbnail upload ──────────────────────────────────
            with open(thumb_path, "rb") as f:
                r = c.post(
                    f"{BASE}/api/clips/{clip_id}/quick-thumbnail/upload",
                    headers=H,
                    files={"image": ("thumb.jpg", f, "image/jpeg")})
            check("quick-thumbnail upload 200", r.status_code == 200,
                  r.text[:300])
            thumb_key = (r.json() or {}).get("key", "")
            check("thumbnail key returned", bool(thumb_key))

            # ── 6) publish through the FIXED bridge ──────────────────
            r = c.post(f"{BASE}/api/clips/{clip_id}/publish", headers=H,
                       json={"channel_ids": [str(ch.id)],
                             "privacy_status": "private",
                             "publish_kind": "video",
                             "use_seo": True})
            check("publish 200 via v2 bridge", r.status_code == 200,
                  r.text[:400])
            pub = r.json() if r.status_code == 200 else {}
            ptid = pub.get("publish_task_id") or pub.get("v2_publish_task_id")
            check("publish_task_id in response", bool(ptid), str(pub)[:300])
            created["ptid"] = ptid

        # ── Assertions on synthesized rows ───────────────────────────
        db.expire_all()
        clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
        master = (db.query(models.MasterVideo)
                  .filter(models.MasterVideo.source_upload_id == clip.job_id)
                  .first())
        check("MasterVideo synthesised", master is not None)
        if master:
            check("clean_master=True (branding WILL apply logo+watermark)",
                  bool(master.clean_master),
                  f"clean_master={master.clean_master}")
            check("r2_key is the REAL storage key (not placeholder)",
                  (master.r2_key or "") == (clip.storage_key or "")
                  and not (master.r2_key or "").startswith("legacy/"),
                  f"r2_key={master.r2_key!r} storage_key={clip.storage_key!r}")
            check("pipeline_version=quick_publish",
                  master.pipeline_version == "quick_publish",
                  master.pipeline_version)

        ujob = (db.query(models.UploadJobV2)
                .filter(models.UploadJobV2.publish_task_id == ptid)
                .first()) if ptid else None
        check("UploadJobV2 created", ujob is not None)
        if ujob:
            check("thumbnail passthrough (user_uploaded + our key)",
                  ujob.thumbnail_source == "user_uploaded"
                  and (ujob.thumbnail_r2_key or "") == thumb_key,
                  f"src={ujob.thumbnail_source!r} key={ujob.thumbnail_r2_key!r}")

            # ── 7) live worker picks it up; fails safely at OAuth ────
            deadline = time.time() + 90
            final = None
            while time.time() < deadline:
                db.expire_all()
                row = db.execute(text(
                    "SELECT status, attempts, last_error FROM upload_jobs_v2 "
                    "WHERE id = :id"), {"id": ujob.id}).fetchone()
                if row and (row[0] in ("failed", "completed", "parked_quota")
                            or int(row[1] or 0) > 0):
                    final = row
                    break
                time.sleep(2)
            check("live worker processed it (no YouTube call possible)",
                  final is not None,
                  "untouched after 90s")
            if final:
                print(f"  worker outcome: status={final[0]!r} "
                      f"attempts={final[1]} "
                      f"error={(final[2] or '')[:120]!r}")
                check("never completed against real YouTube",
                      final[0] != "completed", final[0])

        print(f"\n{'='*60}\n{len(PASS)} checks passed, {len(FAIL)} failed")
        return 0 if not FAIL else 1
    finally:
        # ── Cleanup all fixtures ─────────────────────────────────────
        try:
            uid = created.get("user_id")
            if uid:
                for stmt, p in [
                    ("DELETE FROM quota_burn_log WHERE upload_job_id IN "
                     "(SELECT id FROM upload_jobs_v2 WHERE user_id=:u)", {"u": uid}),
                    ("DELETE FROM publish_attempts WHERE upload_job_id IN "
                     "(SELECT id FROM upload_jobs_v2 WHERE user_id=:u)", {"u": uid}),
                    ("DELETE FROM credit_ledger WHERE user_id=:u", {"u": uid}),
                    ("DELETE FROM upload_jobs_v2 WHERE user_id=:u", {"u": uid}),
                    ("DELETE FROM publish_tasks WHERE user_id=:u", {"u": uid}),
                    ("DELETE FROM master_videos WHERE source_upload_id=:j",
                     {"j": created.get("job_id", -1)}),
                    ("DELETE FROM clips WHERE id=:c",
                     {"c": created.get("clip_id", -1)}),
                    ("DELETE FROM jobs WHERE id=:j",
                     {"j": created.get("job_id", -1)}),
                    ("DELETE FROM oauth_tokens WHERE id=:t",
                     {"t": created.get("tok_id", -1)}),
                    ("DELETE FROM channels WHERE id=:c",
                     {"c": created.get("ch_id", -1)}),
                    ("DELETE FROM users WHERE id=:u", {"u": uid}),
                ]:
                    try:
                        db.execute(text(stmt), p)
                        db.commit()
                    except Exception:
                        db.rollback()
                print("fixtures cleaned up")
        finally:
            db.close()
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
