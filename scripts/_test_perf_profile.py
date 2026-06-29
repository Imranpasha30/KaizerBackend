import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
from database import SessionLocal
import models
from seo.performance_profile import build_channel_profile

db = SessionLocal()
try:
    total = db.query(models.ClipPerformance).count()
    print(f"ClipPerformance rows in DB: {total}")
    # which channels have performance data
    from sqlalchemy import func
    by_ch = (db.query(models.ClipPerformance.channel_id, func.count())
               .group_by(models.ClipPerformance.channel_id).all())
    print("channels with perf data:", by_ch[:15])
    names = {c.id: c.name for c in db.query(models.Channel).all()}
    # build profile for the top few channels (or a sample)
    targets = [cid for cid, _ in by_ch[:5]] or [17, 23]
    for cid in targets:
        p = build_channel_profile(db, cid)
        print(f"\nchannel #{cid} ({names.get(cid)!r}): ready={p['ready']} n_videos={p['n_videos']}")
        if p["ready"]:
            print(f"   winning_keywords: {p['winning_keywords']}")
            print(f"   top vph={p.get('top_views_per_hour')} median vph={p.get('median_views_per_hour')}")
            print(f"   top_titles: {p.get('top_titles')}")
        else:
            print(f"   reason: {p.get('reason')}")
finally:
    db.close()
