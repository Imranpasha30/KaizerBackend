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


def _normalize_folder(raw: Optional[str]) -> str:
    """Canonical form: no leading/trailing '/', no '..'; empty = root.

    Keeps the path printable — disallows anything that might confuse the
    filesystem later if we ever materialize folders.  We store virtual
    folders only; this is just UI organization.
    """
    s = (raw or "").strip().strip("/")
    if not s:
        return ""
    # Strip any relative-path tricks defensively
    parts = [p for p in s.split("/") if p and p not in (".", "..")]
    # Each part: keep alnum + spaces + - _ to stay URL-safe
    import re as _re
    cleaned = [_re.sub(r"[^\w \-]", "", p).strip() for p in parts]
    cleaned = [p for p in cleaned if p]
    return "/".join(cleaned[:6])   # max depth 6, prevents absurd nesting


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
        "folder_path":    a.folder_path or "",
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
    folder_path: Optional[str] = None,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """All of this user's assets.  When `folder_path` is given, only assets
    in that exact folder are returned (no subtree — keep navigation simple).

    Folder-marker rows are never returned — they're implementation detail.
    """
    q = (
        db.query(models.UserAsset)
          .filter(
              models.UserAsset.user_id == user.id,
              models.UserAsset.kind != "folder_marker",
          )
    )
    if folder_path is not None:
        q = q.filter(models.UserAsset.folder_path == _normalize_folder(folder_path))
    rows = q.order_by(
        models.UserAsset.is_default_ad.desc(),
        models.UserAsset.created_at.desc(),
    ).all()
    return [_to_dict(a) for a in rows]


@router.post("/upload", status_code=201)
async def upload_asset(
    file: UploadFile = File(...),
    kind: str = Form("image"),
    tags: str = Form(""),
    is_default_ad: bool = Form(False),
    folder_path: str = Form(""),
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
        folder_path=_normalize_folder(folder_path),
    )
    db.add(row); db.commit(); db.refresh(row)
    return _to_dict(row)


class AssetPatch(BaseModel):
    is_default_ad: Optional[bool] = None
    tags:          Optional[List[str]] = None
    kind:          Optional[str] = None
    folder_path:   Optional[str]   = None   # "" = move to root; any string = move to that folder


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
    if payload.folder_path is not None:
        # Moving an asset: logos referenced by channels keep working because
        # Channel.logo_asset_id is an FK by id, not by path.  No rewrite
        # needed.  File on disk does NOT move (we store virtual folders only).
        row.folder_path = _normalize_folder(payload.folder_path)
    db.commit(); db.refresh(row)
    return _to_dict(row)


# ─── Folder operations ────────────────────────────────────────────────────

@router.get("/folders")
def list_folders(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Distinct folder paths the user has assets in.  Sorted alphabetically.

    The frontend renders these as the folder navigation (pills / tree) on
    the Assets page.  Root ("") is always listed first as "Root".
    """
    rows = (
        db.query(models.UserAsset.folder_path)
          .filter(models.UserAsset.user_id == user.id)
          .distinct()
          .all()
    )
    folders = sorted({(r[0] or "") for r in rows})
    # Also surface all ancestor folders so nested trees work when a user
    # directly put an asset at "logos/english/" without creating "logos/".
    expanded: set[str] = set(folders)
    for f in list(folders):
        parts = (f or "").split("/")
        for i in range(1, len(parts)):
            expanded.add("/".join(parts[:i]))
    return sorted(expanded)


class FolderCreate(BaseModel):
    path: str


@router.post("/folders", status_code=201)
def create_folder(
    payload: FolderCreate,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Create a named folder BEFORE any asset lives in it.

    Implementation: insert a tiny "folder keeper" row (zero-byte, kind=
    'folder_marker') so the folder shows up in list_folders.  This lets the
    user organize upload targets ahead of time.  The marker is invisible in
    list_assets by kind filter.
    """
    norm = _normalize_folder(payload.path)
    if not norm:
        raise HTTPException(422, "Folder path is required")
    existing = db.query(models.UserAsset).filter(
        models.UserAsset.user_id == user.id,
        models.UserAsset.folder_path == norm,
    ).first()
    if existing:
        return {"path": norm, "created": False, "note": "Folder already exists"}

    keeper = models.UserAsset(
        user_id=user.id,
        filename=".keep",
        file_path="",           # no on-disk file — pure marker
        thumb_path="",
        kind="folder_marker",
        mime="",
        size_bytes=0,
        width=0, height=0,
        is_default_ad=False,
        tags=[],
        folder_path=norm,
    )
    db.add(keeper); db.commit()
    return {"path": norm, "created": True}


class FolderRename(BaseModel):
    old_path: str
    new_path: str


@router.patch("/folders")
def rename_folder(
    payload: FolderRename,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Rename / move a folder.  Every asset whose folder_path matches
    `old_path` (or nests under it) gets the prefix rewritten to `new_path`.

    Channel.logo_asset_id references by asset ID, so cross-references remain
    valid automatically — no path rewriting anywhere else.
    """
    old = _normalize_folder(payload.old_path)
    new = _normalize_folder(payload.new_path)
    if not old:
        raise HTTPException(422, "old_path is required")

    # Find matching folders (exact + nested)
    old_prefix = old + "/"
    rows = (
        db.query(models.UserAsset)
          .filter(
              models.UserAsset.user_id == user.id,
              (models.UserAsset.folder_path == old)
              | (models.UserAsset.folder_path.like(f"{old_prefix}%"))
          )
          .all()
    )
    for a in rows:
        fp = a.folder_path or ""
        if fp == old:
            a.folder_path = new
        elif fp.startswith(old_prefix):
            suffix = fp[len(old_prefix):]
            a.folder_path = (new + "/" + suffix) if new else suffix
    db.commit()
    return {"renamed": len(rows), "old": old, "new": new}


@router.delete("/folders")
def delete_folder(
    path: str,
    cascade: bool = False,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Delete a folder.  When `cascade=true`, also deletes every asset in
    that folder + its subtree.  Otherwise only removes the folder marker
    and moves any orphans to the root.
    """
    norm = _normalize_folder(path)
    if not norm:
        raise HTTPException(422, "path is required")

    prefix = norm + "/"
    q = db.query(models.UserAsset).filter(
        models.UserAsset.user_id == user.id,
        (models.UserAsset.folder_path == norm)
        | (models.UserAsset.folder_path.like(f"{prefix}%")),
    )
    rows = q.all()

    if cascade:
        # Delete all files on disk + rows
        deleted_ids = []
        for a in rows:
            for p in (a.file_path, a.thumb_path):
                if p:
                    try: Path(p).unlink(missing_ok=True)
                    except Exception: pass
            db.delete(a)
            deleted_ids.append(a.id)
        db.commit()
        return {"path": norm, "deleted_asset_ids": deleted_ids, "cascaded": True}
    else:
        # Pull everything back to root; drop folder-markers entirely
        promoted = 0
        for a in rows:
            if a.kind == "folder_marker":
                db.delete(a)
            else:
                a.folder_path = ""
                promoted += 1
        db.commit()
        return {"path": norm, "assets_promoted_to_root": promoted, "cascaded": False}


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
