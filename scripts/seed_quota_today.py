"""Rebase today's quota buckets to the REAL numbers, so the new
count-based model starts tracking from the truth.

Why: under the dead 1,600-unit model, 6 FAILED videos.insert attempts
charged ~9,600 phantom units to the shared 'oauth' (Queries) bucket,
leaving it falsely at ~9,900/10,000. Under the new model:

  * uploads bucket ('vinsert'): seed = today's SUCCESSFUL videos.insert
    count (the only thing that actually consumes the 100/day bucket).
  * Queries bucket ('oauth'): drop the phantom upload charges — rebase
    to today's real non-upload usage (≈ list/thumbnail/RTMP calls). The
    failed-upload reservations are gone under the new model, so the pool
    should reflect only genuine Queries-pool work.

Read-only by default. Pass --apply to write.
  python scripts/seed_quota_today.py            # DRY RUN
  python scripts/seed_quota_today.py --apply
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from database import SessionLocal
import models
from youtube import quota_v2

APPLY = "--apply" in sys.argv


def main() -> int:
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        today = start.strftime("%Y-%m-%d")
        print(f"{'APPLY' if APPLY else 'DRY-RUN'}  UTC today={today}\n")

        # --- Real successful uploads today (the only true bucket consumer) ---
        ok_uploads = db.execute(text(
            "SELECT count(*) FROM youtube_api_calls "
            "WHERE operation='videos.insert' AND success=true "
            "AND created_at >= :s AND created_at < :e"),
            {"s": start, "e": end}).scalar() or 0
        ok_uploads = int(ok_uploads)

        # --- Real non-upload Queries-pool usage today (successful calls) ---
        real_queries = db.execute(text(
            "SELECT coalesce(sum(quota_cost),0) FROM youtube_api_calls "
            "WHERE operation <> 'videos.insert' AND success=true "
            "AND created_at >= :s AND created_at < :e"),
            {"s": start, "e": end}).scalar() or 0
        real_queries = int(real_queries)

        cap_uploads = quota_v2.uploads_daily_cap()
        cap_queries = quota_v2.daily_cap()

        # current bucket states
        before = {r[0]: int(r[1]) for r in db.execute(text(
            "SELECT api_key_hash, units_used FROM api_quota WHERE date=:d"),
            {"d": today}).fetchall()}
        print("Current buckets:", before or "(none)")
        print(f"Real successful uploads today      = {ok_uploads}  (cap {cap_uploads})")
        print(f"Real non-upload Queries usage today = {real_queries}  (cap {cap_queries})")
        print()
        print(f"Plan:")
        print(f"  vinsert bucket  -> units_used = {ok_uploads}   (0/{cap_uploads} clean start)")
        print(f"  oauth   bucket  -> units_used = {real_queries}  (drop phantom failed-upload units)")

        if not APPLY:
            print("\n(DRY RUN — re-run with --apply to write.)")
            return 0

        for bucket, val in (("vinsert", ok_uploads), ("oauth", real_queries)):
            row = (db.query(models.ApiQuota)
                     .filter(models.ApiQuota.date == today,
                             models.ApiQuota.api_key_hash == bucket).first())
            if row is None:
                if val > 0:
                    db.add(models.ApiQuota(date=today, api_key_hash=bucket,
                                           units_used=val))
            else:
                row.units_used = val
                db.add(row)
        db.commit()

        after = {r[0]: int(r[1]) for r in db.execute(text(
            "SELECT api_key_hash, units_used FROM api_quota WHERE date=:d"),
            {"d": today}).fetchall()}
        print("\nAFTER buckets:", after)
        print(f"\nDONE — uploads today {ok_uploads}/{cap_uploads}, "
              f"Queries pool {real_queries}/{cap_queries}.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
