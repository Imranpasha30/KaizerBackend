"""Focused proof of the brand->upload conveyor (KAIZER_SPLIT_STAGES).

Reuses test_durable_queue's fixtures/patches. A FRESH (unbranded) job must:
  split=1 -> first process() returns HANDOFF (no upload), requeues to 'queued'
             with the brand artifact set and attempts neutralized; the re-claim
             then returns COMPLETED with EXACTLY ONE upload.
  split=0 -> a single process() returns COMPLETED (brand+upload in one call).
Cleans up its rows. Zero real YouTube/R2 calls (all patched).
"""
import os
import sys
import time
import uuid

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BACKEND)
sys.path.insert(0, os.path.join(_BACKEND, "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from dotenv import load_dotenv
load_dotenv(os.path.join(_BACKEND, ".env"))

from sqlalchemy import text
from database import SessionLocal
import models
from services import job_queue, upload_dispatch as ud
import test_durable_queue as T

results = []
def check(name, cond, extra=""):
    results.append(bool(cond))
    print(("  PASS" if cond else "  FAIL"), name, ("" if cond else f" {extra}"))

_upload_calls = {"n": 0}

def _fake_branding(job_id):
    """Mimic services.branding.process_upload_job: persist key + flip status."""
    db = SessionLocal()
    try:
        key = f"branded/handoff/{T.NONCE}.mp4"
        db.execute(text(
            "UPDATE upload_jobs_v2 SET branded_artifact_r2_key=:k, "
            "status='ready_to_upload', updated_at=now() WHERE id=:id"
        ), {"k": key, "id": int(job_id)})
        db.commit()
        return key
    finally:
        db.close()

def _manual_claim(jid, worker):
    """Claim THIS job deterministically (mimics claim_jobs' attempts bump),
    avoiding claim_jobs picking unrelated dev rows."""
    db = SessionLocal()
    try:
        db.execute(text(
            "UPDATE upload_jobs_v2 SET status='claimed', claimed_by=:w, "
            "attempts=attempts+1, lease_expires_at=now() + interval '300 seconds', "
            "updated_at=now() WHERE id=:id"
        ), {"w": worker[:64], "id": int(jid)})
        db.commit()
    finally:
        db.close()

def _make_unbranded(db, fx):
    j = models.UploadJobV2(
        publish_task_id=fx["pt"].id, channel_id=fx["channel"].id,
        oauth_token_id=fx["token"].id, upload_path="direct", publish_kind="video",
        status="queued", attempts=0,
        idempotency_key=uuid.uuid4().hex + uuid.uuid4().hex,
        publish_version=f"hv{T.NONCE}", predicted_quota_units=1650,
        predicted_credit_cost=16, branded_artifact_r2_key="",   # <-- NOT branded yet
        user_id=fx["user"].id, priority="normal",
    )
    db.add(j); db.flush(); jid = j.id; db.commit()
    return jid

def _row(jid):
    db = SessionLocal()
    try:
        return db.query(models.UploadJobV2).get(jid)
    finally:
        db.close()

# ── SPLIT ON: fresh job -> HANDOFF, then COMPLETED ──────────────────
print("\n[A] split=1: fresh job -> brand HANDOFF -> upload COMPLETED")
os.environ["KAIZER_SPLIT_STAGES"] = "1"
db = SessionLocal(); fx = T.make_fixtures(db); monkey = {}
_orig_brand = ud.branding.process_upload_job
try:
    _upload_calls["n"] = 0
    jid = _make_unbranded(db, fx)
    T.patch_dispatch(monkey, upload_behavior=lambda j: (_upload_calls.__setitem__("n", _upload_calls["n"] + 1) or {"video_id": "vid_handoff"}))
    ud.branding.process_upload_job = _fake_branding
    worker = f"hv:{T.NONCE}:w1"
    _manual_claim(jid, worker)
    att1 = _row(jid).attempts
    out1 = ud.process(jid, worker, time.monotonic() + 300)
    check("first call returns HANDOFF", out1 == ud.Outcome.HANDOFF, str(out1))
    r = _row(jid)
    check("job requeued to 'queued'", r.status == "queued", r.status)
    check("brand artifact persisted", bool((r.branded_artifact_r2_key or "").strip()), r.branded_artifact_r2_key)
    check("claim cleared (slot released)", r.claimed_by is None, str(r.claimed_by))
    check("attempts neutralized (brand claim free)", r.attempts == att1 - 1, f"{att1}->{r.attempts}")
    check("no upload on brand call", _upload_calls["n"] == 0, str(_upload_calls["n"]))
    # Upload phase: re-claim + process again
    _manual_claim(jid, worker)
    out2 = ud.process(jid, worker, time.monotonic() + 300)
    check("second call returns COMPLETED", out2 == ud.Outcome.COMPLETED, str(out2))
    r = _row(jid)
    check("status completed", r.status == "completed", r.status)
    check("exactly one upload total", _upload_calls["n"] == 1, str(_upload_calls["n"]))
    check("video id recorded", (r.youtube_video_id or "") == "vid_handoff", r.youtube_video_id)
finally:
    T.unpatch_dispatch(monkey)
    ud.branding.process_upload_job = _orig_brand
    T.cleanup(db, fx); db.close()

# ── SPLIT OFF: fresh job -> single COMPLETED (no handoff) ───────────
print("\n[B] split=0: fresh job -> single COMPLETED (no handoff)")
os.environ["KAIZER_SPLIT_STAGES"] = "0"
db = SessionLocal(); fx = T.make_fixtures(db); monkey = {}
_orig_brand = ud.branding.process_upload_job
try:
    _upload_calls["n"] = 0
    jid = _make_unbranded(db, fx)
    T.patch_dispatch(monkey, upload_behavior=lambda j: (_upload_calls.__setitem__("n", _upload_calls["n"] + 1) or {"video_id": "vid_serial"}))
    ud.branding.process_upload_job = _fake_branding
    worker = f"hv:{T.NONCE}:w2"
    _manual_claim(jid, worker)
    out = ud.process(jid, worker, time.monotonic() + 300)
    check("single call returns COMPLETED (not HANDOFF)", out == ud.Outcome.COMPLETED, str(out))
    r = _row(jid)
    check("status completed", r.status == "completed", r.status)
    check("exactly one upload in one call", _upload_calls["n"] == 1, str(_upload_calls["n"]))
finally:
    T.unpatch_dispatch(monkey)
    ud.branding.process_upload_job = _orig_brand
    T.cleanup(db, fx); db.close()

print(f"\n{'='*52}\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
