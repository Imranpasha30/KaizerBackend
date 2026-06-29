"""Per-account branding check.

For user imranpasha.ahmed@gmail.com, walk every CONNECTED account (a
models.Channel whose oauth_token carries a non-empty refresh_token_enc),
resolve its brand via services.brand_resolver.resolve_brand_profile, and
report whether that account will inject its OWN branding at publish time:

  * logo_present   — resolver returned a logo_asset_id
  * logo_exists    — that UserAsset row exists AND has usable bytes
                     (storage_url / storage_key / local file_path),
                     i.e. NOT a dangling/orphaned asset id
  * watermark_present
  * socials_present
  * fallback_used  — the brand came from an account-level sibling, not the
                     account's own row (post-merge this should never happen)
  * complete       — logo + watermark + socials all set
  * missing        — list of the three that are absent

After the duplicate-merge each account is a single connection with NO
siblings, so the sibling fallback can no longer fill gaps — flag any
account missing logo / watermark / socials.
"""
import json
import os
import sys

sys.path.insert(0, ".")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import warnings

warnings.filterwarnings("ignore")

from database import SessionLocal
import models
from services import brand_resolver

USER_EMAIL = "imranpasha.ahmed@gmail.com"


def _socials_nonempty(socials_json: str) -> bool:
    """True when the resolved socials JSON has at least one entry."""
    if not socials_json:
        return False
    try:
        d = json.loads(socials_json)
    except Exception:
        return False
    if not isinstance(d, dict):
        return False
    # A social link counts only when it has a non-empty value.
    return any((str(v).strip() for v in d.values() if v is not None))


def _asset_exists(db, asset_id) -> bool:
    """A logo asset RENDERS only if the UserAsset row exists and points at
    real bytes — a remote storage url/key OR a local file that is present
    and non-empty. A dangling id (no row, or a row with no usable bytes)
    means no logo will appear."""
    if not asset_id:
        return False
    a = (
        db.query(models.UserAsset)
        .filter(models.UserAsset.id == int(asset_id))
        .first()
    )
    if a is None:
        return False  # dangling FK — row was deleted
    # Remote storage (R2/CDN) — bytes live off-disk.
    if (getattr(a, "storage_url", "") or "").strip():
        return True
    if (getattr(a, "storage_key", "") or "").strip():
        return True
    # Local path — must exist on disk and be non-empty.
    fp = (getattr(a, "file_path", "") or "").strip()
    if fp and os.path.isfile(fp) and os.path.getsize(fp) > 0:
        return True
    return False


def main():
    db = SessionLocal()
    out = {"accounts": [], "any_incomplete": False, "summary": ""}
    try:
        user = (
            db.query(models.User)
            .filter(models.User.email == USER_EMAIL)
            .first()
        )
        if user is None:
            out["summary"] = f"User {USER_EMAIL} not found."
            print(json.dumps(out, indent=2))
            return out

        # All this user's channels that are actually CONNECTED to YouTube
        # (non-empty refresh_token_enc on the 1:1 oauth_token).
        channels = (
            db.query(models.Channel)
            .filter(models.Channel.user_id == user.id)
            .order_by(models.Channel.id)
            .all()
        )
        connected = []
        for ch in channels:
            tok = getattr(ch, "oauth_token", None)
            rt = (getattr(tok, "refresh_token_enc", "") or "").strip() if tok else ""
            if rt:
                connected.append(ch)

        for ch in connected:
            # Resolve WITH the account-level fallback (the real publish path).
            resolved = brand_resolver.resolve_brand_profile(db, ch.id)
            # Resolve WITHOUT fallback to detect whether the fallback fired.
            own = brand_resolver._resolve_for_channel(db, ch.id)
            fallback_used = (
                brand_resolver._brand_is_empty(own)
                and not brand_resolver._brand_is_empty(resolved)
            )

            logo_present = bool(resolved.logo_asset_id)
            logo_exists = _asset_exists(db, resolved.logo_asset_id)
            watermark_present = bool((resolved.watermark_text or "").strip())
            socials_present = _socials_nonempty(resolved.socials_json)

            missing = []
            # Logo only counts as present if it also actually exists.
            if not (logo_present and logo_exists):
                missing.append("logo")
            if not watermark_present:
                missing.append("watermark")
            if not socials_present:
                missing.append("socials")

            complete = not missing

            out["accounts"].append(
                {
                    "name": ch.name or "",
                    "channel_id": int(ch.id),
                    "logo_present": logo_present,
                    "logo_exists": logo_exists,
                    "watermark_present": watermark_present,
                    "socials_present": socials_present,
                    "fallback_used": fallback_used,
                    "complete": complete,
                    "missing": missing,
                }
            )

        out["any_incomplete"] = any(not a["complete"] for a in out["accounts"])
        n = len(out["accounts"])
        n_complete = sum(1 for a in out["accounts"] if a["complete"])
        n_fallback = sum(1 for a in out["accounts"] if a["fallback_used"])
        out["summary"] = (
            f"{n} connected account(s) for {USER_EMAIL}: "
            f"{n_complete} complete, {n - n_complete} incomplete, "
            f"{n_fallback} relying on sibling fallback."
        )
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return out
    finally:
        db.close()


if __name__ == "__main__":
    main()
