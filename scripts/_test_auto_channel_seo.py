"""Prove auto-per-channel SEO writes DISTINCT, marked variants for a job's
chosen channels — the up-front dedup bypass. Sets up a scratch scenario on a
real job+clip (2 connected channels, empty variants), runs the auto generator,
asserts both channels got a distinct _per_channel title, then restores."""
import os, sys, json
_B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _B)
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from dotenv import load_dotenv
load_dotenv(os.path.join(_B, ".env"))
from database import SessionLocal
import models
from seo.auto_channel_seo import auto_generate_channel_seo_for_job

db = SessionLocal()
try:
    # find a clip with base SEO whose job's user owns >=2 connected channels
    job = clip = None
    chans = []
    for c in (db.query(models.Clip)
                .filter(models.Clip.seo.isnot(None))
                .order_by(models.Clip.id.desc()).limit(200).all()):
        if not (c.seo and "title" in c.seo):
            continue
        j = db.query(models.Job).filter(models.Job.id == c.job_id).first()
        if not j:
            continue
        connected = [ch for ch in db.query(models.Channel)
                       .filter(models.Channel.user_id == j.user_id).all()
                     if ch.oauth_token is not None and bool(ch.oauth_token.refresh_token_enc)]
        if len(connected) >= 2:
            job, clip, chans = j, c, connected[:2]
            break
    if not job:
        print("no clip+job with a 2+ connected-channel owner found"); sys.exit(2)

    print(f"job #{job.id} user={job.user_id} clip={clip.id}")
    print(f"channels: {[(c.id, c.name) for c in chans]}")
    orig_targets = job.target_channel_ids
    # The auto-fn processes EVERY clip of the job — snapshot ALL their variants
    # so the test restores the job exactly (no real-data pollution).
    all_clips = db.query(models.Clip).filter(models.Clip.job_id == job.id).all()
    orig_all = {c.id: c.seo_variants for c in all_clips}
    ok = True
    try:
        # scratch setup: target exactly these 2 channels, no marked variants yet
        job.target_channel_ids = json.dumps([chans[0].id, chans[1].id])
        clip.seo_variants = "{}"
        db.add(job); db.add(clip); db.commit()

        res = auto_generate_channel_seo_for_job(job.id, language=(job.language or "te") if hasattr(job, "language") else "te",
                                                log=lambda m: print("  ", m))
        print("  result:", res)

        db.expire(clip)
        v = json.loads(clip.seo_variants or "{}")
        t0 = (v.get(str(chans[0].id)) or {}).get("title", "")
        t1 = (v.get(str(chans[1].id)) or {}).get("title", "")
        m0 = (v.get(str(chans[0].id)) or {}).get("_per_channel")
        m1 = (v.get(str(chans[1].id)) or {}).get("_per_channel")
        s0 = (v.get(str(chans[0].id)) or {}).get("seo_score")
        s1 = (v.get(str(chans[1].id)) or {}).get("seo_score")
        print(f"  ch{chans[0].id}: marked={m0} score={s0} title={t0[:60]}")
        print(f"  ch{chans[1].id}: marked={m1} score={s1} title={t1[:60]}")

        # NEAR-duplicate guard: YouTube flags videos whose titles only differ in
        # the tail. Require a different opening + low word overlap, not just !=.
        def _words(s):
            import re as _re
            return set(w for w in _re.split(r"\s+", (s or "").lower()) if len(w) > 1)
        w0, w1 = _words(t0), _words(t1)
        jacc = (len(w0 & w1) / len(w0 | w1)) if (w0 | w1) else 1.0
        same_prefix = (t0[:18].strip() == t1[:18].strip())
        if not (m0 and m1):
            print("  FAIL: a channel variant is not marked _per_channel"); ok = False
        elif not (t0 and t1):
            print("  FAIL: a channel title is empty"); ok = False
        elif t0 == t1:
            print("  FAIL: titles are identical (no dedup differentiation)"); ok = False
        elif same_prefix or jacc >= 0.8:
            print(f"  FAIL: titles too SIMILAR (prefix_match={same_prefix} word_overlap={jacc:.2f}) "
                  f"-> YouTube would still see duplicates"); ok = False
        else:
            print(f"  PASS: DISTINCT per-channel titles (word_overlap={jacc:.2f})")

        # idempotency: a second run must NOT change anything (no clobber, no re-bill)
        before = clip.seo_variants
        res2 = auto_generate_channel_seo_for_job(job.id, log=lambda m: None)
        db.expire(clip)
        if clip.seo_variants == before and (res2.get("variants_written", 0) == 0):
            print("  PASS: idempotent re-run wrote nothing")
        else:
            print(f"  FAIL: re-run changed variants ({res2})"); ok = False
    finally:
        for c in db.query(models.Clip).filter(models.Clip.job_id == job.id).all():
            if c.id in orig_all:
                c.seo_variants = orig_all[c.id]
                db.add(c)
        job.target_channel_ids = orig_targets
        db.add(job); db.commit()
        print("  restored original target_channel_ids + seo_variants (all clips)")
    print("\nRESULT:", "PASS" if ok else "FAIL")
finally:
    db.close()
