"""Show — and optionally delete — EMPTY leftover style-reference rows
(kind:styles) that carry no real SEO data and no link to anything.

A style ref is EMPTY (safe to remove) when it has:
  * no usable oauth_token (not a connected account)
  * no learned corpus
  * no profile_destinations
  * no competitor identity: blank handle AND blank/default title_formula
A row failing ANY of these is KEPT (it's a real reference or has data).

Also prints each connected ACCOUNT's branding completeness so you can
see which still need a watermark / socials before publishing.

  python scripts/clean_empty_style_refs.py [email]            # dry-run
  python scripts/clean_empty_style_refs.py [email] --apply    # delete
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
import models

args = [a for a in sys.argv[1:] if not a.startswith("--")]
EMAIL = args[0] if args else "imranpasha.ahmed@gmail.com"
APPLY = "--apply" in sys.argv


def _socials_set(c):
    s = getattr(c, "socials", None)
    return bool(s) and s not in ({}, "{}", "")


db = SessionLocal()
try:
    u = db.query(models.User).filter(models.User.email == EMAIL).first()
    if not u:
        print(f"user {EMAIL} not found"); sys.exit(1)
    print(f"USER {u.id} <{u.email}>  ({'APPLY' if APPLY else 'DRY-RUN'})\n")

    chans = (db.query(models.Channel)
             .filter(models.Channel.user_id == u.id)
             .order_by(models.Channel.id).all())

    print("=== ACCOUNTS — branding completeness (for publishing) ===")
    for c in chans:
        tok = c.oauth_token
        if not (tok and tok.refresh_token_enc):
            continue
        logo = (getattr(tok, "logo_asset_id", None) or c.logo_asset_id)
        wm = (getattr(c, "watermark_text", "") or "").strip()
        missing = []
        if not logo: missing.append("logo")
        if not wm: missing.append("watermark")
        if not _socials_set(c): missing.append("socials")
        state = "all set" if not missing else "MISSING: " + ", ".join(missing)
        print(f"  '{c.name}' -> {tok.google_channel_title}: {state}")

    print("\n=== SEO SETTINGS — empty leftovers vs real references ===")
    to_delete = []
    for c in chans:
        tok = c.oauth_token
        if tok and tok.refresh_token_enc:
            continue  # it's an account
        has_corpus = db.query(models.ChannelCorpus).filter(
            models.ChannelCorpus.channel_id == c.id).count() > 0
        n_dests = db.query(models.ProfileDestination).filter(
            models.ProfileDestination.profile_id == c.id).count()
        handle = (c.handle or "").strip()
        tf = (c.title_formula or "").strip()
        gcid = (getattr(tok, "google_channel_id", "") if tok else "") or ""
        # The connect flow auto-fills title_formula as
        # "English Hook (తెలుగు అనువాదం) | <name>" — a name echo, NOT a
        # real competitor style config. Treat it as no config.
        tf_is_default = (not tf) or tf.startswith("English Hook")
        is_empty = (not has_corpus and n_dests == 0 and not handle
                    and not gcid and tf_is_default)
        if is_empty:
            to_delete.append(c)
            print(f"  EMPTY  #{c.id} '{c.name}' -> delete")
        else:
            keeps = []
            if has_corpus: keeps.append("corpus")
            if handle: keeps.append(f"handle {handle}")
            if tf: keeps.append("title_formula")
            if gcid: keeps.append(f"yt {gcid}")
            if n_dests: keeps.append(f"{n_dests} dests")
            print(f"  KEEP   #{c.id} '{c.name}' ({', '.join(keeps)})")

    print(f"\n{len(to_delete)} empty style-reference row(s) "
          f"{'deleted' if APPLY else 'would be deleted'}.")
    if APPLY and to_delete:
        for c in to_delete:
            if c.oauth_token is not None:
                db.delete(c.oauth_token)
            db.delete(c)
        db.commit()
        print("done.")
    elif to_delete:
        print("Re-run with --apply to delete them.")
finally:
    db.close()
