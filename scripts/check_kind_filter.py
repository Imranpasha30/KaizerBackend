"""Verify GET /api/channels/?kind=accounts|styles returns disjoint sets
and that accounts never leak into styles."""
import sys
sys.path.insert(0, ".")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")

import httpx
import auth
import models
from database import SessionLocal

db = SessionLocal()
u = (db.query(models.User)
     .filter(models.User.email == "imranpasha.ahmed@gmail.com")
     .first()) or db.query(models.User).filter(models.User.is_active == True).first()  # noqa: E712
tok = auth.issue_token(u)
db.close()
H = {"Authorization": f"Bearer {tok}"}
BASE = "http://localhost:8000"

ok = True
with httpx.Client(timeout=30) as c:
    full = c.get(f"{BASE}/api/channels/", headers=H).json()
    accts = c.get(f"{BASE}/api/channels/", params={"kind": "accounts"}, headers=H).json()
    styles = c.get(f"{BASE}/api/channels/", params={"kind": "styles"}, headers=H).json()

acct_ids = {r["id"] for r in accts}
style_ids = {r["id"] for r in styles}
print(f"user {u.id} ({u.email}): full={len(full)} accounts={len(accts)} styles={len(styles)}")

# 1. accounts are all connected
a_all_connected = all(r.get("connected") and r.get("kind") == "account" for r in accts)
print(("  PASS" if a_all_connected else "  FAIL") + "  accounts tab: every row connected + kind=account")
ok = ok and a_all_connected

# 2. styles are all NOT connected
s_none_connected = all((not r.get("connected")) and r.get("kind") == "style" for r in styles)
print(("  PASS" if s_none_connected else "  FAIL") + "  styles tab: no row connected + kind=style")
ok = ok and s_none_connected

# 3. disjoint
disjoint = acct_ids.isdisjoint(style_ids)
print(("  PASS" if disjoint else "  FAIL") + "  accounts and styles are disjoint (no leak)")
ok = ok and disjoint

# 4. partition covers the full list
partition = (acct_ids | style_ids) == {r["id"] for r in full}
print(("  PASS" if partition else "  FAIL") + "  accounts + styles == full list")
ok = ok and partition

# Show the account names (should be the real channels, deduped per card by the UI)
print("  accounts:", sorted(r["name"] for r in accts))

print("\nKIND FILTER OK" if ok else "\nKIND FILTER FAILED")
sys.exit(0 if ok else 1)
