"""Verify the per-clip MasterVideo fix landed and separates outputs."""
import sys
sys.path.insert(0, ".")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")

from sqlalchemy import inspect, text
from database import SessionLocal, engine

ok = True


def chk(label, cond, detail=""):
    global ok
    ok = ok and cond
    print(("  PASS  " if cond else "  FAIL  ") + label + ("" if cond else f"  [{detail}]"))


insp = inspect(engine)
cols = {c["name"] for c in insp.get_columns("master_videos")}
chk("master_videos.clip_id column exists", "clip_id" in cols)

if engine.dialect.name == "postgresql":
    with engine.connect() as conn:
        uq = conn.execute(text(
            "SELECT conname FROM pg_constraint "
            "WHERE conrelid = 'master_videos'::regclass AND contype = 'u'"
        )).fetchall()
    chk("per-job UNIQUE constraint on source_upload_id is dropped",
        not any("source_upload_id" in (c[0] or "") for c in uq),
        f"constraints={[c[0] for c in uq]}")

db = SessionLocal()
try:
    # Backfill check: the 4 repaired masters now carry their clip_id.
    rows = db.execute(text(
        "SELECT id, clip_id, r2_key FROM master_videos "
        "WHERE r2_key LIKE 'legacy/clip/%' ORDER BY id")).fetchall()
    print("\n  repaired masters after migration:")
    all_backfilled = True
    for mid, clip_id, key in rows:
        import re
        m = re.search(r"clip/(\d+)/", key)
        expected = int(m.group(1)) if m else None
        good = clip_id == expected
        all_backfilled = all_backfilled and good
        print(f"    master {mid}: clip_id={clip_id} key={key} "
              f"{'OK' if good else f'EXPECTED {expected}'}")
    chk("all legacy masters backfilled with correct clip_id", all_backfilled)

    # The collapse is gone: job 305 had clips 742 (long) + 743 (short).
    # 742 → its master; 743 → must NOT resolve to 742's master.
    long_master = db.execute(text(
        "SELECT id, r2_key FROM master_videos WHERE clip_id = 742")).fetchone()
    short_master = db.execute(text(
        "SELECT id, r2_key FROM master_videos WHERE clip_id = 743")).fetchone()
    if long_master:
        print(f"\n  clip 742 (long)  -> master {long_master[0]} "
              f"({long_master[1]})")
    if short_master:
        print(f"  clip 743 (short) -> master {short_master[0]} "
              f"({short_master[1]})")
    else:
        print("  clip 743 (short) -> no master yet (a fresh publish will "
              "create its OWN master with short_01.mp4 — correct)")
    # Key assertion: 743 does not point at 742's file.
    chk("short (743) will NOT reuse the long video's master/file",
        short_master is None
        or (short_master[0] != (long_master[0] if long_master else None)
            and "clip/742/" not in (short_master[1] or "")))
finally:
    db.close()

print("\nPER-CLIP MASTER FIX OK" if ok else "\nFIX INCOMPLETE")
sys.exit(0 if ok else 1)
