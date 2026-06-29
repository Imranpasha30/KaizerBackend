"""One-time rebase of today's quota 'used' counter to REALITY.

The api_quota 'oauth' bucket accumulated fake burns from regression
suites (phase2/phase3/durable) that used mock uploaders — the UI chip
showed 31,000 'used' while the forensic log of REAL YouTube API calls
showed zero. Suites now burn an isolated 'testrun' bucket
(KAIZER_YT_QUOTA_BUCKET), so this drift can't recur; this script fixes
the historical residue once.

Truth source: ``youtube_api_calls`` (the forensic log written around
every REAL googleapiclient call). We recompute today's units from it
and rebase the 'oauth' bucket to match.

Run from KaizerBackend/:  python scripts/rebase_quota_used.py
Add --dry-run to preview without writing.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from sqlalchemy import text  # noqa: E402

from database import SessionLocal, engine  # noqa: E402


def main() -> int:
    dry = "--dry-run" in sys.argv
    db = SessionLocal()
    try:
        # Real units burned today, from the forensic log. Column probe:
        # the cost column name differs across schema generations.
        real_used = None
        for costcol in ("quota_cost", "cost", "units"):
            try:
                real_used = int(db.execute(text(
                    f"SELECT COALESCE(SUM({costcol}), 0) "
                    f"FROM youtube_api_calls "
                    f"WHERE CAST(created_at AS DATE) = CAST(now() AS DATE)"
                )).scalar() or 0)
                print(f"real usage today (youtube_api_calls.{costcol}): "
                      f"{real_used} units")
                break
            except Exception:
                db.rollback()
                continue
        if real_used is None:
            print("ERROR: couldn't read the forensic log — aborting, "
                  "nothing changed")
            return 1

        rows = db.execute(text(
            "SELECT api_key_hash, units_used FROM api_quota "
            "WHERE date = to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD')"
            if engine.dialect.name == "postgresql" else
            "SELECT api_key_hash, units_used FROM api_quota "
            "WHERE date = strftime('%Y-%m-%d', 'now')"
        )).fetchall()
        print("today's api_quota buckets BEFORE:",
              {r[0]: int(r[1] or 0) for r in rows} or "(none)")

        if dry:
            print(f"[dry-run] would set oauth bucket -> {real_used}")
            return 0

        res = db.execute(text(
            "UPDATE api_quota SET units_used = :u "
            "WHERE api_key_hash = 'oauth' "
            "AND date = " + (
                "to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD')"
                if engine.dialect.name == "postgresql"
                else "strftime('%Y-%m-%d', 'now')"
            )
        ), {"u": real_used})
        db.commit()
        print(f"oauth bucket rebased -> {real_used} "
              f"({res.rowcount} row(s) updated)")

        rows = db.execute(text(
            "SELECT api_key_hash, units_used FROM api_quota "
            "WHERE date = to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD')"
            if engine.dialect.name == "postgresql" else
            "SELECT api_key_hash, units_used FROM api_quota "
            "WHERE date = strftime('%Y-%m-%d', 'now')"
        )).fetchall()
        print("today's api_quota buckets AFTER:",
              {r[0]: int(r[1] or 0) for r in rows} or "(none)")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
