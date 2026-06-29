"""Prove per-channel SEO applies at publish: _compose_metadata must use a
MARKED per-channel variant for the channel, and IGNORE an unmarked one
(so shared-mode / legacy clips are unchanged). Restores the clip after."""
import os, sys, json
_B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _B)
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from dotenv import load_dotenv
load_dotenv(os.path.join(_B, ".env"))
from database import SessionLocal
import models
from services import upload_dispatch as ud

db = SessionLocal()
try:
    # find a completed job whose clip resolves AND has its own seo (so the
    # "unmarked -> falls back to shared" branch is meaningful).
    job = clip = None
    for j in (db.query(models.UploadJobV2)
                .filter(models.UploadJobV2.status == "completed")
                .order_by(models.UploadJobV2.id.desc()).limit(40).all()):
        c = ud._clip_for_job(db, j)
        if c is not None and c.seo and "title" in (c.seo or ""):
            job, clip = j, c
            break
    if not job:
        print("no suitable completed job/clip found"); sys.exit(2)
    print(f"job #{job.id} channel={job.channel_id} clip={clip.id}")
    orig = clip.seo_variants
    cid = job.channel_id
    ok = True
    try:
        # 1) MARKED per-channel variant -> must be used
        clip.seo_variants = json.dumps({str(cid): {
            "title": "PERCHAN_TEST_TITLE_XYZ", "description": "d", "tags": ["x"],
            "_per_channel": True}})
        db.add(clip); db.commit(); db.expire(clip)
        t, *_ = ud._compose_metadata(db, db.query(models.UploadJobV2).get(job.id))
        print("  with marker  -> title:", t[:50])
        if not t.startswith("PERCHAN_TEST_TITLE_XYZ"):
            print("  FAIL: marked variant NOT used"); ok = False
        else:
            print("  PASS: marked per-channel variant used")

        # 2) UNMARKED variant -> must be IGNORED (falls back to shared clip.seo)
        clip.seo_variants = json.dumps({str(cid): {
            "title": "LEGACY_SHOULD_BE_IGNORED", "description": "d"}})
        db.add(clip); db.commit(); db.expire(clip)
        t2, *_ = ud._compose_metadata(db, db.query(models.UploadJobV2).get(job.id))
        print("  no marker    -> title:", t2[:50])
        if t2.startswith("LEGACY_SHOULD_BE_IGNORED"):
            print("  FAIL: unmarked variant wrongly used"); ok = False
        else:
            print("  PASS: unmarked variant ignored (shared SEO used)")
    finally:
        clip.seo_variants = orig
        db.add(clip); db.commit()
        print("  restored original seo_variants")
    print("\nRESULT:", "PASS" if ok else "FAIL")
finally:
    db.close()
