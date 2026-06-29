"""Verify account-level branding fallback: a connected profile with no
brand of its own now inherits its real account's branding from a
sibling connection (same google_channel_id)."""
import sys
sys.path.insert(0, ".")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")

from sqlalchemy import text
from database import SessionLocal
from services import brand_resolver

ok = True


def chk(label, cond, detail=""):
    global ok
    ok = ok and cond
    print(("  PASS  " if cond else "  FAIL  ") + label + ("" if cond else f"  [{detail}]"))


db = SessionLocal()
try:
    # Find every group of >1 connected channel sharing a google_channel_id.
    groups = db.execute(text("""
        SELECT tok.google_channel_id, ch.user_id,
               array_agg(ch.id ORDER BY ch.id) AS chan_ids,
               array_agg(COALESCE(ch.name,'') ORDER BY ch.id) AS names
        FROM channels ch
        JOIN oauth_tokens tok ON tok.channel_id = ch.id
        WHERE tok.refresh_token_enc IS NOT NULL AND tok.refresh_token_enc <> ''
        GROUP BY tok.google_channel_id, ch.user_id
        HAVING count(*) > 1
    """)).fetchall()

    if not groups:
        print("  (no duplicate connections found — nothing to assert)")
    tested = 0
    for gcid, uid, chan_ids, names in groups:
        print(f"\n  account {gcid} (user {uid}) profiles {list(chan_ids)} {list(names)}")
        # Which of these has branding vs not?
        branded = []
        for cid in chan_ids:
            b = brand_resolver._resolve_for_channel(db, cid)
            empty = brand_resolver._brand_is_empty(b)
            print(f"    channel {cid}: own brand "
                  f"{'EMPTY' if empty else f'logo={b.logo_asset_id} wm={b.watermark_text!r}'}")
            if not empty:
                branded.append(cid)
        if not branded:
            print("    (no profile in this account has branding — skip)")
            continue
        # Assert: the EMPTY profiles now resolve to a non-empty brand via fallback.
        for cid in chan_ids:
            own = brand_resolver._resolve_for_channel(db, cid)
            if not brand_resolver._brand_is_empty(own):
                continue  # already branded
            resolved = brand_resolver.resolve_brand_profile(db, cid)
            inherited = not brand_resolver._brand_is_empty(resolved)
            chk(f"channel {cid} (no own brand) inherits account branding "
                f"(logo={resolved.logo_asset_id})", inherited,
                "still empty after fallback")
            tested += 1
    if tested == 0 and groups:
        print("\n  (all duplicate profiles already had their own branding — "
              "fallback not exercised, but no regression)")
finally:
    db.close()

print("\nACCOUNT BRANDING FALLBACK OK" if ok else "\nFALLBACK FAILED")
sys.exit(0 if ok else 1)
