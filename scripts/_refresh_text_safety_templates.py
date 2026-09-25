"""One-off (DEV ONLY): push the text-safety hardened premium_templates sources into the
already-stored template bundles IN PLACE — the "Aurora pattern": same ids, overwrite the
bundle entry html (+ .kaizer_src.html sidecar), refresh contract_json, regen preview.png
(the picker plate is cached per id, so a render change NEEDS a fresh preview).

Reuses the router's _apply_html so a refreshed bundle goes through the exact upload
pipeline (sanitize -> normalize/infer -> contract -> preview). Per-template try/rollback;
a failing template never blocks the rest. Ids 44-47 (new shorts) are already capped and
36 (obsidian full) already bounded its headline — deliberately untouched.
"""
import os
import sys

os.environ.setdefault("PYTHONUTF8", "1")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../KaizerBackend
os.chdir(HERE)                       # load_dotenv() reads KaizerBackend/.env (DEV DB)
sys.path.insert(0, HERE)

import database                      # noqa: E402
from database import SessionLocal    # noqa: E402
import models                        # noqa: E402
from routers.custom_templates import _apply_html   # noqa: E402

# HARD safety: this script must only ever run against the DEV database.
assert "kaizer_dev" in (database.DATABASE_URL or ""), (
    f"refusing to run: DATABASE_URL is not the DEV db ({database.DATABASE_URL!r})")

SRC_DIR = os.path.join(os.path.dirname(HERE), "premium_templates")
# stored template id -> hardened source file (mapping verified via each bundle's <title>)
MAP = {
    30: "aurora_glass_full.html",
    31: "cinema_prime_full.html",
    32: "market_pulse_full.html",
    33: "neon_reel_short.html",
    34: "minimal_light_short.html",
    35: "sport_arena_short.html",
    37: "ivory_light_full.html",
    38: "nightline_horizon_full.html",
    39: "studio_verde_full.html",
}


def sig(cj):
    slots = (cj or {}).get("slots") or []
    return sorted((s.get("kind"), s.get("name") or s.get("key") or "") for s in slots)


def main() -> int:
    db = SessionLocal()
    ok = fail = 0
    try:
        for tid, fname in sorted(MAP.items()):
            path = os.path.join(SRC_DIR, fname)
            try:
                row = db.query(models.CustomTemplate).get(tid)
                if row is None:
                    print(f"tmpl {tid}: NOT IN DB, skip"); continue
                with open(path, encoding="utf-8") as fh:
                    html = fh.read()
                old_sig = sig(row.contract_json if isinstance(row.contract_json, dict) else {})
                contract = _apply_html(row, html, db)   # sanitize+normalize+contract+preview
                new_sig = sig(row.contract_json if isinstance(row.contract_json, dict) else {})
                entry = os.path.join(row.dir_path or "", row.entry_rel or "index.html")
                with open(entry, encoding="utf-8") as fh:
                    head = fh.read(40)
                dt = head.lstrip().lower().startswith("<!doctype")
                drift = "" if old_sig == new_sig else "  !! SLOT SET CHANGED"
                print(f"tmpl {tid}: OK ({fname}) status={row.status} "
                      f"doctype_kept={dt} preview={bool(row.preview_path)}{drift}")
                ok += 1
            except Exception as exc:
                db.rollback()
                print(f"tmpl {tid}: ERROR {exc!r}")
                fail += 1
    finally:
        db.close()
    print(f"done — {ok} refreshed, {fail} failed")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
