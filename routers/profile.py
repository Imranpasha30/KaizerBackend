"""User profile router — avatar upload + creator-rating endpoints.

The avatar (image or GIF) is stored on R2 next to the library uploads
(`library/users/<id>/avatar.<ext>`) so the single-source-of-truth rule
extends to profile pictures too. Creator ratings are upsert-per-rater
and the cached aggregate on the User row gets adjusted by the delta.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter, Depends, File, HTTPException, UploadFile, status,
)
from pydantic import BaseModel
from sqlalchemy.orm import Session

import auth
import models
from database import get_db
from pipeline_core.storage import get_storage_provider

router = APIRouter(prefix="/api/profile", tags=["profile"])

log   = logging.getLogger("kaizer.profile")
audit = logging.getLogger("kaizer.profile.audit")

# Avatar upload constraints — small and tight.
AVATAR_ALLOWED_EXTS  = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
AVATAR_ALLOWED_MIMES = {
    "image/jpeg", "image/png", "image/webp", "image/gif",
}
AVATAR_MAX_BYTES = 8 * 1024 * 1024   # 8 MiB


def _r2():
    """R2 pinned regardless of env — same single-source rule as library."""
    try:
        return get_storage_provider(backend="r2")
    except Exception as exc:
        log.error("profile: R2 storage unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Profile storage (Cloudflare R2) is not configured. "
                "Set R2_BUCKET, R2_ENDPOINT, R2_ACCESS_KEY_ID, "
                "R2_SECRET_ACCESS_KEY in the environment and restart."
            ),
        ) from exc


def _avatar_key(user_id: int, ext: str) -> str:
    ext = (ext or ".jpg").lower()
    if not ext.startswith("."):
        ext = "." + ext
    return f"library/users/{user_id}/avatar{ext}"


# ─── Avatar upload / delete ────────────────────────────────────────────


@router.post("/me/avatar")
async def upload_avatar(
    image: UploadFile = File(...),
    db: Session       = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Upload a profile picture (JPEG, PNG, WebP, or GIF). Overwrites
    any previous avatar — R2 key is deterministic per user."""
    raw = image.filename or "avatar.jpg"
    ext = Path(raw).suffix.lower() or ".jpg"
    if ext not in AVATAR_ALLOWED_EXTS:
        raise HTTPException(
            415,
            f"Unsupported avatar extension {ext!r}. "
            f"Allowed: {sorted(AVATAR_ALLOWED_EXTS)}",
        )
    if image.content_type and image.content_type.split(";")[0].strip() not in AVATAR_ALLOWED_MIMES:
        log.warning("profile: unusual avatar content-type %r", image.content_type)

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"avatar_{user.id}_"))
    local = tmp_dir / f"avatar{ext}"
    try:
        bytes_written = 0
        with local.open("wb") as out:
            while True:
                chunk = await image.read(256 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > AVATAR_MAX_BYTES:
                    raise HTTPException(
                        413,
                        f"Avatar exceeds {AVATAR_MAX_BYTES // (1024 * 1024)} MiB limit",
                    )
                out.write(chunk)
        if bytes_written == 0:
            raise HTTPException(400, "Uploaded avatar is empty")

        r2 = _r2()
        key = _avatar_key(user.id, ext)
        r2.upload(str(local), key,
                  content_type=image.content_type or "image/jpeg")

        # If there was a previous avatar with a different extension, drop
        # it from R2 so we don't accumulate ghost files. Best-effort.
        if user.avatar_key and user.avatar_key != key:
            try:
                r2.delete(user.avatar_key)
            except Exception:
                pass

        user.avatar_key = key
        db.commit()
        audit.info("avatar upload user=%d size=%d", user.id, bytes_written)

        url = ""
        try:
            url = r2.get_url(key, signed=False)
        except Exception:
            url = ""
        return {"ok": True, "avatar_key": key, "avatar_url": url}
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


@router.delete("/me/avatar")
def delete_avatar(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    if not user.avatar_key:
        return {"ok": True}
    try:
        _r2().delete(user.avatar_key)
    except Exception as exc:
        log.warning("profile: R2 delete avatar failed (non-fatal): %s", exc)
    user.avatar_key = ""
    db.commit()
    audit.info("avatar delete user=%d", user.id)
    return {"ok": True}


# ─── Creator rating ────────────────────────────────────────────────────


class CreatorRateIn(BaseModel):
    stars: int


@router.post("/creators/{creator_id}/rate")
def rate_creator(
    creator_id: int,
    body: CreatorRateIn,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Upsert the caller's rating for a creator. Cached
    creator_rating_sum/creator_rating_count on the User row are
    delta-adjusted so listing creators never has to GROUP BY at
    query time. Cannot rate yourself."""
    if creator_id == user.id:
        raise HTTPException(422, "You can't rate yourself")
    stars = int(body.stars or 0)
    if not 1 <= stars <= 5:
        raise HTTPException(422, "stars must be in 1..5")
    creator = db.query(models.User).get(creator_id)
    if not creator or not creator.is_active:
        raise HTTPException(404, "Creator not found")

    existing = db.query(models.CreatorRating).filter_by(
        creator_id=creator_id, rater_id=user.id,
    ).first()
    if existing:
        delta = stars - int(existing.stars or 0)
        existing.stars = stars
        creator.creator_rating_sum = int(creator.creator_rating_sum or 0) + delta
    else:
        db.add(models.CreatorRating(
            creator_id=creator_id, rater_id=user.id, stars=stars,
        ))
        creator.creator_rating_sum   = int(creator.creator_rating_sum or 0) + stars
        creator.creator_rating_count = int(creator.creator_rating_count or 0) + 1

    db.commit(); db.refresh(creator)
    avg = (creator.creator_rating_sum / creator.creator_rating_count) if creator.creator_rating_count else 0.0
    return {
        "ok":           True,
        "stars":        stars,
        "rating_avg":   round(avg, 2),
        "rating_count": int(creator.creator_rating_count or 0),
    }


# ─── Convenience helper for auth.py ───────────────────────────────────

def user_avatar_url(user: models.User) -> str:
    """Resolve the user's avatar to a public/signed URL, returning ''
    when the user has no avatar set OR the storage backend isn't
    configured (silently — avatar is optional)."""
    if not user or not user.avatar_key:
        return ""
    try:
        return _r2().get_url(user.avatar_key, signed=False)
    except Exception:
        return ""
