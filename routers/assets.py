"""User assets router — image upload, list, set-default, delete.

Powers the "Assets" page: a Canva-style library of images the user can
reuse across clips.  Setting one as `is_default_ad` makes the pipeline
use it automatically for every clip when "Use my default image" is on.
"""
from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import List, Optional

from fastapi import (
    APIRouter, Depends, File, Form, HTTPException, Request, UploadFile,
    status,
)
from pydantic import BaseModel
from sqlalchemy.orm import Session

import auth
import models
from database import get_db


router = APIRouter(prefix="/api/assets", tags=["assets"])

BASE_DIR  = Path(__file__).resolve().parent.parent
ASSETS_ROOT = BASE_DIR / "output" / "user_assets"
THUMB_MAX = 512    # longest edge for the thumbnail JPG
ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_BYTES = 20 * 1024 * 1024   # 20 MB per file


def _to_dict(a: models.UserAsset) -> dict:
    return {
        "id":             a.id,
        "filename":       a.filename,
        "kind":           a.kind,
        "mime":           a.mime,
        "size_bytes":     a.size_bytes,
        "width":          a.width,
        "height":         a.height,
        "is_default_ad":  bool(a.is_default_ad),
        "tags":           list(a.tags or []),
        "created_at":     a.created_at.isoformat() if a.created_at else None,
        "url":            f"/api/file/?path={a.file_path}",
        "thumb_url":      f"/api/file/?path={a.thumb_path}" if a.thumb_path else f"/api/file/?path={a.file_path}",
    }


def _thumb_for(src: Path) -> tuple[str, int, int]:
    """Create a <=THUMB_MAX-edge JPG thumbnail next to the source file.

    Returns (thumb_abs_path, width, height).  If Pillow isn't available or
    the source is not an image, returns ("", 0, 0).
    """
    try:
        from PIL import Image
        with Image.open(src) as im:
            w0, h0 = im.size
            im2 = im.convert("RGB")
            im2.thumbnail((THUMB_MAX, THUMB_MAX), Image.LANCZOS)
            thumb_path = src.with_name(f"thumb_{src.stem}.jpg")
            im2.save(thumb_path, "JPEG", quality=85)
            return str(thumb_path), w0, h0
    except Exception as e:
        print(f"[assets] thumb gen failed for {src.name}: {e}")
        return "", 0, 0


# ─── Endpoints ────────────────────────────────────────────────────────────

@router.get("/")
def list_assets(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    rows = (
        db.query(models.UserAsset)
          .filter(models.UserAsset.user_id == user.id)
          .order_by(models.UserAsset.is_default_ad.desc(), models.UserAsset.created_at.desc())
          .all()
    )
    return [_to_dict(a) for a in rows]


@router.post("/upload", status_code=201)
async def upload_asset(
    file: UploadFile = File(...),
    kind: str = Form("image"),
    tags: str = Form(""),
    is_default_ad: bool = Form(False),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    content = await file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(413, f"File too large (>{MAX_BYTES // (1024*1024)} MB).")
    if not content:
        raise HTTPException(400, "Empty upload.")

    mime = file.content_type or mimetypes.guess_type(file.filename or "")[0] or "application/octet-stream"
    if mime not in ALLOWED_MIMES:
        raise HTTPException(415, f"Unsupported image type: {mime}")

    user_dir = ASSETS_ROOT / str(user.id)
    user_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename — keep the extension, strip path/weird chars
    safe_name = Path(file.filename or "upload").name.replace(" ", "_")
    out_path = user_dir / safe_name
    # If a file by that name already exists, suffix with a counter
    if out_path.exists():
        stem, suf = out_path.stem, out_path.suffix
        i = 1
        while out_path.exists():
            out_path = user_dir / f"{stem}_{i}{suf}"
            i += 1
    out_path.write_bytes(content)

    thumb_path, w, h = _thumb_for(out_path)

    # If this upload is marked default → demote any previous default
    if is_default_ad:
        db.query(models.UserAsset).filter(
            models.UserAsset.user_id == user.id,
            models.UserAsset.is_default_ad == True,  # noqa: E712
        ).update({"is_default_ad": False})

    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    row = models.UserAsset(
        user_id=user.id,
        filename=out_path.name,
        file_path=str(out_path.resolve()),
        thumb_path=thumb_path,
        kind=kind or "image",
        mime=mime,
        size_bytes=len(content),
        width=w, height=h,
        is_default_ad=bool(is_default_ad),
        tags=tag_list,
    )
    db.add(row); db.commit(); db.refresh(row)
    return _to_dict(row)


class AssetPatch(BaseModel):
    is_default_ad: Optional[bool] = None
    tags:          Optional[List[str]] = None
    kind:          Optional[str] = None


@router.patch("/{asset_id}")
def patch_asset(
    asset_id: int,
    payload: AssetPatch,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    row = db.query(models.UserAsset).filter(
        models.UserAsset.id == asset_id, models.UserAsset.user_id == user.id,
    ).first()
    if not row:
        raise HTTPException(404, "Asset not found")

    if payload.is_default_ad is True:
        # Demote any existing default so at most one exists per user
        db.query(models.UserAsset).filter(
            models.UserAsset.user_id == user.id,
            models.UserAsset.is_default_ad == True,  # noqa: E712
            models.UserAsset.id != asset_id,
        ).update({"is_default_ad": False})
    if payload.is_default_ad is not None:
        row.is_default_ad = bool(payload.is_default_ad)
    if payload.tags is not None:
        row.tags = list(payload.tags)
    if payload.kind is not None:
        row.kind = payload.kind
    db.commit(); db.refresh(row)
    return _to_dict(row)


@router.delete("/{asset_id}")
def delete_asset(
    asset_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    row = db.query(models.UserAsset).filter(
        models.UserAsset.id == asset_id, models.UserAsset.user_id == user.id,
    ).first()
    if not row:
        raise HTTPException(404, "Asset not found")
    # Best-effort cleanup
    for p in (row.file_path, row.thumb_path):
        if p:
            try: Path(p).unlink(missing_ok=True)
            except Exception: pass
    db.delete(row); db.commit()
    return {"deleted": asset_id}


@router.get("/default")
def get_default(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Convenience endpoint — what's currently the user's default ad image?"""
    row = db.query(models.UserAsset).filter(
        models.UserAsset.user_id == user.id,
        models.UserAsset.is_default_ad == True,  # noqa: E712
    ).first()
    return _to_dict(row) if row else None
