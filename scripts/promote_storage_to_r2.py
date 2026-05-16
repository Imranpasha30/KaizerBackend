"""CLI: promote every local-stored Clip / UserAsset to R2.

Usage::

    # Dry-run (default) — shows what WOULD happen, writes nothing.
    python scripts/promote_storage_to_r2.py

    # Live run — uploads files + updates DB rows.
    python scripts/promote_storage_to_r2.py --live

    # Live + verbose per-row progress to stdout.
    python scripts/promote_storage_to_r2.py --live --verbose

The same logic backs the admin button
(``POST /api/admin/storage/promote-to-r2``). The CLI is intended for
ops/launch-day use; the admin button is for in-product use.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make the parent package importable when invoked directly.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from dotenv import load_dotenv
load_dotenv(override=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true",
                        help="Actually upload + update DB. Default is dry-run.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-row progress to stdout.")
    parser.add_argument("--json", action="store_true",
                        help="Emit the JSON report at the end (always on with --live).")
    args = parser.parse_args()

    from database import SessionLocal
    from storage_migration import promote_local_to_r2

    def progress(msg: str) -> None:
        if args.verbose:
            print(msg)

    db = SessionLocal()
    try:
        report = promote_local_to_r2(db, dry_run=not args.live, progress_cb=progress)
    finally:
        db.close()

    if args.json or args.live:
        print(json.dumps(report, indent=2))
    else:
        # Compact human-readable summary
        t = report["totals"]
        print()
        print(f"  Mode:      {'LIVE' if args.live else 'DRY-RUN'}")
        print(f"  Scanned:   {t['scanned']}")
        print(f"  Migrated:  {t['migrated']}  (would migrate)" if not args.live
              else f"  Migrated:  {t['migrated']}")
        print(f"  Skipped:   {t['skipped']}")
        print(f"  Failed:    {t['failed']}")
        for name, tr in report["tables"].items():
            print(f"    • {name:12s}: {tr['migrated']:3d} migrated, "
                  f"{tr['failed']:3d} failed")

    return 1 if report["totals"]["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
