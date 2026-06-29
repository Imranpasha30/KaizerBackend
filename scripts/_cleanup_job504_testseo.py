"""Targeted cleanup: the FIRST auto-SEO test run only restored one clip, so the
other clips of job #504 kept TEST-generated per-channel variants for the two
test channels (1730, 1863). Remove just those marked entries so the job reverts
to its pre-test state (shared SEO; operator can re-apply per-channel any time)."""
import os, sys, json
_B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _B)
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from dotenv import load_dotenv
load_dotenv(os.path.join(_B, ".env"))
from database import SessionLocal
import models

JOB_ID = 504
TEST_CHANNELS = {"1730", "1863"}

db = SessionLocal()
try:
    clips = db.query(models.Clip).filter(models.Clip.job_id == JOB_ID).all()
    removed_total = 0
    for c in clips:
        if not c.seo_variants:
            continue
        try:
            v = json.loads(c.seo_variants)
        except Exception:
            continue
        if not isinstance(v, dict):
            continue
        removed = []
        for cid in list(TEST_CHANNELS):
            ent = v.get(cid)
            if isinstance(ent, dict) and ent.get("_per_channel"):
                del v[cid]
                removed.append(cid)
        if removed:
            c.seo_variants = json.dumps(v)
            db.add(c)
            removed_total += len(removed)
            print(f"clip {c.id}: removed per-channel variants for {removed}")
    db.commit()
    print(f"\nDONE: removed {removed_total} test-generated per-channel variant(s) from job {JOB_ID}")
finally:
    db.close()
