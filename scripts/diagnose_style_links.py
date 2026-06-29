"""Show, for a user, what lives under SEO Settings (style references)
and ANY link they still have to publish accounts (oauth tokens,
profile_destinations, shared google_channel_id)."""
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
import models

EMAIL = sys.argv[1] if len(sys.argv) > 1 else "imranpasha.ahmed@gmail.com"

db = SessionLocal()
try:
    u = db.query(models.User).filter(models.User.email == EMAIL).first()
    if not u:
        print(f"user {EMAIL} not found"); sys.exit(1)
    print(f"USER {u.id} <{u.email}>\n")

    chans = (db.query(models.Channel)
             .filter(models.Channel.user_id == u.id)
             .order_by(models.Channel.id).all())

    print("=== ACCOUNTS (connected, publish targets) ===")
    acct_gcids = set()
    for c in chans:
        tok = c.oauth_token
        if tok and tok.refresh_token_enc:
            acct_gcids.add(tok.google_channel_id)
            print(f"  #{c.id} '{c.name}' -> {tok.google_channel_title} "
                  f"({tok.google_channel_id})")

    print("\n=== SEO SETTINGS (style references = kind:styles) ===")
    for c in chans:
        tok = c.oauth_token
        is_account = bool(tok and tok.refresh_token_enc)
        if is_account:
            continue
        # Links that would tie this style ref to publish:
        stale_tok = bool(tok)  # an oauth_token row with NO refresh token
        tok_gcid = (tok.google_channel_id if tok else "") or ""
        dests = db.query(models.ProfileDestination).filter(
            models.ProfileDestination.profile_id == c.id).all()
        dest_gcids = [d.google_channel_id for d in dests]
        links = []
        if stale_tok:
            links.append(f"stale oauth_token row (gcid={tok_gcid or 'none'})")
        if dests:
            overlap = [g for g in dest_gcids if g in acct_gcids]
            links.append(f"{len(dests)} profile_destination(s)"
                         + (f" — {len(overlap)} point at YOUR accounts" if overlap else ""))
        has_corpus = db.query(models.ChannelCorpus).filter(
            models.ChannelCorpus.channel_id == c.id).count() > 0
        tag = "  <-- HAS PUBLISH LINKS" if links else ""
        print(f"  #{c.id} '{c.name}' corpus={'yes' if has_corpus else 'no'}"
              f"{tag}")
        for l in links:
            print(f"        link: {l}")
finally:
    db.close()
