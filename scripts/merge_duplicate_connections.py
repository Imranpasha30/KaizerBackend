"""Collapse duplicate YouTube-account connections.

Background: connecting the same real channel more than once created
multiple `channels` profiles + `oauth_tokens` for ONE google_channel_id
(e.g. Auto Wala = Personal 9 + Personal 10), each with its own
branding. The brand resolver already inherits branding account-wide, so
this script is pure tidy-up: it keeps ONE canonical connection per real
channel and folds the duplicates into it.

For each (user_id, google_channel_id) group with >1 connected profile:
  * canonical = the profile WITH branding (logo/watermark/socials);
    tie-break: most ProfileDestinations, then lowest id.
  * copy the winning branding onto the canonical (logo on channel+token,
    watermark/socials on channel).
  * repoint upload_jobs_v2.channel_id, upload_jobs.channel_id, and
    profile_destinations.profile_id from each duplicate -> canonical
    (dedup destinations), delete the duplicate's oauth_token + channel.

Run from KaizerBackend/:
  python scripts/merge_duplicate_connections.py            # dry-run (default)
  python scripts/merge_duplicate_connections.py --apply    # perform it
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")

from sqlalchemy import text

import models
from database import SessionLocal


def _has_branding(db, ch) -> bool:
    tok = ch.oauth_token
    logo = (getattr(tok, "logo_asset_id", None) if tok else None) or ch.logo_asset_id
    wm = (getattr(ch, "watermark_text", "") or "").strip()
    soc = getattr(ch, "socials", None)
    soc_has = bool(soc) and soc not in ({}, "{}", "")
    return bool(logo or wm or soc_has)


def _dest_count(db, ch_id) -> int:
    return db.query(models.ProfileDestination).filter(
        models.ProfileDestination.profile_id == ch_id).count()


def main() -> int:
    apply = "--apply" in sys.argv
    db = SessionLocal()
    merged_groups = 0
    try:
        groups = db.execute(text("""
            SELECT ch.user_id, tok.google_channel_id,
                   array_agg(ch.id ORDER BY ch.id) AS ids
            FROM channels ch
            JOIN oauth_tokens tok ON tok.channel_id = ch.id
            WHERE tok.refresh_token_enc IS NOT NULL AND tok.refresh_token_enc <> ''
            GROUP BY ch.user_id, tok.google_channel_id
            HAVING count(*) > 1
        """)).fetchall()

        if not groups:
            print("No duplicate connections found. Nothing to do.")
            return 0

        # Discover EVERY (table, column) that FKs to channels.id, so the
        # repoint can't miss one (live_streams, campaigns, etc.).
        _fk_cols = [
            (r[0], r[1]) for r in db.execute(text("""
                SELECT tc.table_name, kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage ccu
                  ON tc.constraint_name = ccu.constraint_name
                 AND tc.table_schema = ccu.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND ccu.table_name = 'channels'
                  AND ccu.column_name = 'id'
            """)).fetchall()
        ]
        print(f"FK columns referencing channels.id: "
              f"{[f'{t}.{c}' for t, c in _fk_cols]}")

        print(f"{'APPLY' if apply else 'DRY-RUN'}: {len(groups)} account(s) "
              f"with duplicate connections\n")

        for uid, gcid, ids in groups:
            chans = (db.query(models.Channel)
                     .filter(models.Channel.id.in_(list(ids))).all())
            chans = {c.id: c for c in chans}
            # Choose canonical: has branding > most destinations > lowest id.
            ranked = sorted(
                ids,
                key=lambda i: (
                    0 if _has_branding(db, chans[i]) else 1,
                    -_dest_count(db, i),
                    i,
                ),
            )
            canonical_id = ranked[0]
            dups = ranked[1:]
            canon = chans[canonical_id]
            print(f"account {gcid} (user {uid}): keep #{canonical_id} "
                  f"'{canon.name}', merge {dups}")

            # Winning branding = canonical's; if canonical somehow lacks it,
            # borrow from a duplicate that has it.
            brand_src = canon
            if not _has_branding(db, canon):
                for d in dups:
                    if _has_branding(db, chans[d]):
                        brand_src = chans[d]
                        break

            if not apply:
                src_logo = (getattr(brand_src.oauth_token, "logo_asset_id", None)
                            if brand_src.oauth_token else None) or brand_src.logo_asset_id
                print(f"    would set canonical branding: logo={src_logo} "
                      f"wm={(brand_src.watermark_text or '')!r}")
                for d in dups:
                    print(f"    would repoint upload jobs + destinations of "
                          f"#{d} -> #{canonical_id}, then delete #{d}")
                continue

            # Copy winning branding onto the canonical (channel + token).
            for field in ("watermark_text", "watermark_opacity",
                          "watermark_position", "socials", "logo_asset_id"):
                setattr(canon, field, getattr(brand_src, field, None))
            src_logo = (getattr(brand_src.oauth_token, "logo_asset_id", None)
                        if brand_src.oauth_token else None) or brand_src.logo_asset_id
            if canon.oauth_token is not None:
                canon.oauth_token.logo_asset_id = src_logo

            for d in dups:
                # Repoint EVERY table that references channels.id onto
                # the canonical (upload_jobs_v2, upload_jobs, live_streams,
                # campaigns, …) — discovered dynamically so we never miss
                # an FK. oauth_tokens (deleted) + profile_destinations
                # (handled below) are skipped.
                for tbl, col in _fk_cols:
                    if (tbl, col) in (("oauth_tokens", "channel_id"),
                                      ("profile_destinations", "profile_id")):
                        continue
                    db.execute(text(
                        f"UPDATE {tbl} SET {col} = :c WHERE {col} = :d"),
                        {"c": canonical_id, "d": d})
                # Move destinations not already on the canonical.
                dests = db.query(models.ProfileDestination).filter(
                    models.ProfileDestination.profile_id == d).all()
                existing = {
                    pd.google_channel_id for pd in
                    db.query(models.ProfileDestination).filter(
                        models.ProfileDestination.profile_id == canonical_id).all()
                }
                for pd in dests:
                    if pd.google_channel_id in existing:
                        db.delete(pd)
                    else:
                        pd.profile_id = canonical_id
                # Drop the duplicate's token + channel.
                dup_ch = chans[d]
                if dup_ch.oauth_token is not None:
                    db.delete(dup_ch.oauth_token)
                # corpus (if any) cascades via the channel relationship.
                db.delete(dup_ch)
            db.commit()
            merged_groups += 1
            print(f"    merged. #{canonical_id} now the sole connection.")

        if not apply:
            print(f"\nDRY-RUN complete. Re-run with --apply to perform "
                  f"the merge.")
        else:
            print(f"\nMerged {merged_groups} account group(s). Each real "
                  f"channel now has ONE connection.")
        return 0
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
