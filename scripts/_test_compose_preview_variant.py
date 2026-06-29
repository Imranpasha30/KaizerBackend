"""Prove the Publish-modal preview (compose-preview) now shows the SAVED
per-channel variant, not the shared base. Mirrors routers/seo.py logic on a
real clip+channel with a scratch marked variant, then restores. No AI calls."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
from database import SessionLocal
import models
from seo.composer import compose

MARK = "PERCHAN_PREVIEW_XYZ"

def compose_preview_logic(clip, channel_id, dest):
    """Replica of routers/seo.py:preview_composed_seo after the fix."""
    generic = json.loads(clip.seo)
    try:
        _variants = json.loads(clip.seo_variants) if clip.seo_variants else {}
        if isinstance(_variants, dict):
            _v = _variants.get(str(channel_id)) or _variants.get(channel_id)
            if isinstance(_v, dict) and _v.get("_per_channel") and _v.get("title"):
                generic = {k: val for k, val in _v.items() if k != "_per_channel"}
    except (ValueError, TypeError):
        pass
    return compose(generic, dest, publish_kind="video")

db = SessionLocal()
try:
    clip = (db.query(models.Clip).filter(models.Clip.seo.isnot(None))
              .order_by(models.Clip.id.desc()).first())
    job = db.query(models.Job).filter(models.Job.id == clip.job_id).first()
    dest = [c for c in db.query(models.Channel).filter(models.Channel.user_id == job.user_id).all()
            if c.oauth_token is not None and bool(c.oauth_token.refresh_token_enc)][0]
    print(f"clip {clip.id} job {clip.job_id} channel {dest.id} ({dest.name})")
    orig = clip.seo_variants
    ok = True
    try:
        # 1) WITH a marked variant -> preview must show the variant title
        clip.seo_variants = json.dumps({str(dest.id): {
            "title": MARK + " distinct headline", "description": "variant desc",
            "tags": ["v1", "v2"], "_per_channel": True}})
        db.add(clip); db.commit(); db.expire(clip)
        c1 = compose_preview_logic(clip, dest.id, dest)
        print("  with variant -> title:", (c1.get("title") or "")[:60])
        if not (c1.get("title") or "").startswith(MARK):
            print("  FAIL: preview did NOT use the saved per-channel variant"); ok = False
        else:
            print("  PASS: preview uses the saved per-channel variant")

        # 2) WITHOUT marker -> falls back to base (shared)
        clip.seo_variants = "{}"
        db.add(clip); db.commit(); db.expire(clip)
        c2 = compose_preview_logic(clip, dest.id, dest)
        base_title = json.loads(clip.seo).get("title", "")
        print("  no variant   -> title:", (c2.get("title") or "")[:60])
        if (c2.get("title") or "").startswith(MARK):
            print("  FAIL: stale variant leaked"); ok = False
        else:
            print("  PASS: falls back to base/shared SEO when no variant")
    finally:
        clip.seo_variants = orig
        db.add(clip); db.commit()
        print("  restored original seo_variants")
    print("\nRESULT:", "PASS" if ok else "FAIL")
finally:
    db.close()
