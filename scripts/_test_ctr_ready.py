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
from analytics.ctr import fetch_video_ctr, token_has_analytics_scope, ANALYTICS_SCOPE

db = SessionLocal()
try:
    # show which channels' tokens already carry the analytics scope (none until re-approve)
    toks = db.query(models.OAuthToken).all()
    have = [t.channel_id for t in toks if token_has_analytics_scope(t)]
    print(f"tokens with analytics scope (re-approved): {have or 'NONE yet (expected)'}")
    print(f"ANALYTICS_SCOPE = {ANALYTICS_SCOPE}")

    vids = [r.video_id for r in db.query(models.ClipPerformance).filter(
        models.ClipPerformance.channel_id == 17).all() if r.video_id][:5]
    print(f"\nfetch_video_ctr(channel=17, {len(vids)} videos) — must return {{}} gracefully (no scope):")
    res = fetch_video_ctr(db, 17, vids)
    print(f"  result: {res!r}")
    print("  PASS: graceful (no crash, empty until re-approval)" if res == {} else f"  got data: {res}")
finally:
    db.close()
