"""Audit: connected YouTube ACCOUNTS vs style PROFILES, and where each
account's branding (logo / watermark / social links) actually lives.

Mental model the user is asserting:
  * ACCOUNT  = real YouTube account (OAuth-connected) = publish target;
               OWNS the logo + watermark + social links.
  * PROFILE  = "style reference" (Channel row) = SEO helper only.

This prints, per user, every account + its branding so we can see if
each account has its own logo/watermark/socials, and where they sit.
"""
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


def asset_summary(db, asset_id):
    if not asset_id:
        return "—"
    a = db.execute(text(
        "SELECT id, kind, filename, storage_url, storage_key, file_path "
        "FROM user_assets WHERE id = :i"), {"i": int(asset_id)}).fetchone()
    if not a:
        return f"#{asset_id} (MISSING ASSET ROW)"
    loc = a[3] or a[4] or a[5] or "no-bytes"
    return f"#{a[0]} {a[2] or ''} [{loc}]"


db = SessionLocal()
try:
    # Every Channel (style profile) joined to its 1:1 OAuthToken (account).
    rows = db.execute(text("""
        SELECT ch.id AS chan_id, ch.user_id, ch.name AS profile_name,
               ch.handle, ch.logo_asset_id AS profile_logo,
               ch.watermark_text, ch.watermark_opacity, ch.watermark_position,
               ch.socials, ch.upload_provider,
               tok.id AS token_id, tok.google_channel_title,
               tok.google_channel_id, tok.logo_asset_id AS account_logo,
               tok.upload_provider AS token_provider,
               (tok.refresh_token_enc IS NOT NULL) AS linked
        FROM channels ch
        LEFT JOIN oauth_tokens tok ON tok.channel_id = ch.id
        ORDER BY ch.user_id, ch.id
    """)).fetchall()

    by_user = {}
    for r in rows:
        by_user.setdefault(r[1], []).append(r)

    for uid, chans in by_user.items():
        u = db.execute(text("SELECT email, name FROM users WHERE id = :u"),
                       {"u": uid}).fetchone()
        print("\n" + "=" * 78)
        print(f"USER {uid}  {u[1] if u else ''} <{u[0] if u else ''}>  "
              f"— {len(chans)} profile(s)/account(s)")
        print("=" * 78)
        for r in chans:
            (chan_id, _uid, pname, handle, plogo, wm_text, wm_op, wm_pos,
             socials, ch_prov, tok_id, gtitle, gcid, alogo, tok_prov,
             linked) = r
            # Only audit ACTUALLY-CONNECTED accounts (skip the seeded
            # style-reference profiles that were never linked to YouTube).
            if not (tok_id and linked):
                continue
            eff_logo = alogo or plogo
            print(f"\n  ▸ PROFILE '{pname}' (channel id {chan_id}, handle {handle})")
            if tok_id:
                print(f"      ACCOUNT (OAuth): '{gtitle}'  {gcid}  "
                      f"{'LINKED' if linked else 'NOT LINKED'}")
            else:
                print(f"      ACCOUNT (OAuth): — none linked —")
            print(f"      logo on ACCOUNT (oauth_token.logo_asset_id): "
                  f"{asset_summary(db, alogo)}")
            print(f"      logo on PROFILE (channel.logo_asset_id):     "
                  f"{asset_summary(db, plogo)}")
            print(f"      => EFFECTIVE logo branding uses:             "
                  f"{asset_summary(db, eff_logo)}")
            wm_repr = repr(wm_text or "")
            print(f"      watermark: text={wm_repr} "
                  f"opacity={wm_op} position={wm_pos or '-'}")
            soc = (socials if isinstance(socials, str) else str(socials)) or ""
            print(f"      social links: {soc[:120] if soc and soc != 'None' else '— none —'}")
            print(f"      upload route: profile={ch_prov or 'auto'} "
                  f"account={tok_prov or 'auto'}")
            verdict = []
            if not eff_logo:
                verdict.append("NO LOGO")
            if not (wm_text or "").strip():
                verdict.append("NO WATERMARK")
            if not soc or soc in ("None", "{}", ""):
                verdict.append("NO SOCIALS")
            print(f"      BRANDING: {'fully set' if not verdict else ' / '.join(verdict)}")
finally:
    db.close()
