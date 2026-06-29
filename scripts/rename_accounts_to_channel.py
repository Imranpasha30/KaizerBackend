"""Rename connected YouTube accounts from their transient "Personal N"
placeholder to their REAL channel title, and strip the auto-stamped
default style fields ("English Hook … | Personal N" / "Neutral") that the
one-click connect left on them.

Background: a connected account is a `channels` row WITH a usable
oauth_token. The one-click connect created it as "Personal N" with
default style fields BEFORE OAuth, then never renamed it — so accounts
looked like style profiles. New connects now self-heal (youtube/oauth.py
exchange_code); this backfills the ones already connected.

Only style references (rows WITHOUT a usable token) are left untouched.
Renames are skipped when the real title is already taken by a DIFFERENT
row of the same user (uniqueness is (user_id, name)).

  python scripts/rename_accounts_to_channel.py [email]            # dry-run
  python scripts/rename_accounts_to_channel.py [email] --apply    # write
  python scripts/rename_accounts_to_channel.py --all --apply      # all users
"""
import sys
sys.path.insert(0, ".")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")

import models
from database import SessionLocal

args = [a for a in sys.argv[1:] if not a.startswith("--")]
APPLY = "--apply" in sys.argv
ALL_USERS = "--all" in sys.argv
EMAIL = args[0] if args else "imranpasha.ahmed@gmail.com"


def _usable(tok) -> bool:
    return bool(tok and (tok.refresh_token_enc or "").strip())


db = SessionLocal()
try:
    if ALL_USERS:
        users = db.query(models.User).all()
    else:
        u = db.query(models.User).filter(models.User.email == EMAIL).first()
        if not u:
            print(f"user {EMAIL} not found"); sys.exit(1)
        users = [u]

    print(f"{'APPLY' if APPLY else 'DRY-RUN'} — {len(users)} user(s)\n")
    total_renamed = total_cleared = total_skipped = 0

    for u in users:
        chans = (db.query(models.Channel)
                 .filter(models.Channel.user_id == u.id)
                 .order_by(models.Channel.id).all())
        names = {c.name for c in chans}
        accts = [c for c in chans if _usable(c.oauth_token)]
        if not accts:
            continue
        print(f"USER {u.id} <{u.email}>")
        for c in accts:
            title = (c.oauth_token.google_channel_title or "").strip()
            actions = []
            # 1. Rename Personal N -> real channel title (if free).
            if title and c.name != title:
                if title in names and title != c.name:
                    actions.append(f"SKIP rename (name '{title}' already used)")
                    total_skipped += 1
                else:
                    actions.append(f"rename '{c.name}' -> '{title}'")
                    if APPLY:
                        names.discard(c.name)
                        c.name = title
                        names.add(title)
                    total_renamed += 1
            # 2. Strip phantom default style fields off the account.
            tf = (getattr(c, "title_formula", "") or "")
            ds = (getattr(c, "desc_style", "") or "")
            cleared = []
            if tf.startswith("English Hook"):
                cleared.append("title_formula")
                if APPLY:
                    c.title_formula = ""
            if ds == "Neutral":
                cleared.append("desc_style")
                if APPLY:
                    c.desc_style = ""
            if cleared:
                actions.append("clear " + ", ".join(cleared))
                total_cleared += 1
            tag = "; ".join(actions) if actions else "already clean"
            print(f"  #{c.id} -> {title or '(no title)'}: {tag}")
        print()

    if APPLY:
        db.commit()
        print(f"DONE. renamed={total_renamed} cleared={total_cleared} "
              f"skipped={total_skipped}")
    else:
        print(f"DRY-RUN. would rename={total_renamed} clear={total_cleared} "
              f"skip={total_skipped}\nRe-run with --apply to write.")
finally:
    db.close()
