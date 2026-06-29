import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import SessionLocal
import models

db = SessionLocal()
try:
    chs = {c.id: c.name for c in db.query(models.Channel).filter(models.Channel.user_id == 2).all()}
    print("=== clips with persisted _per_channel variants (recent 25) ===")
    clips = (db.query(models.Clip)
               .filter(models.Clip.seo_variants.isnot(None))
               .order_by(models.Clip.id.desc()).limit(25).all())
    any_marked = False
    for c in clips:
        try:
            v = json.loads(c.seo_variants or "{}")
        except Exception:
            v = {}
        marked = {k: vv for k, vv in v.items() if isinstance(vv, dict) and vv.get("_per_channel")}
        if not marked:
            continue
        any_marked = True
        job = db.query(models.Job).filter(models.Job.id == c.job_id).first()
        tgt = getattr(job, "target_channel_ids", None) if job else None
        print(f"\nCLIP {c.id} job={c.job_id} job.target={tgt}")
        for k, vv in marked.items():
            print(f"   ch {k} ({chs.get(int(k),'?')}): {(vv.get('title') or '')[:60]}")
    if not any_marked:
        print("  (NONE) — no clip has an applied _per_channel variant. "
              "=> distinct SEO was previewed but never APPLIED/persisted.")
finally:
    db.close()
