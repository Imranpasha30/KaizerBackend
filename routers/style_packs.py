"""User-composed style packs — save your own mix, reuse it on any job.

Operator requirement: a creative user who likes the LOOK of one pack,
the MOTION of another and the SOUND of a third can combine them, name
the mix, and use it on future jobs. The creation is PRIVATE to its
owner; admins see every user creation in the admin tab.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

import auth
import models
from database import get_db

router = APIRouter(prefix="/api/style-packs", tags=["style-packs"])

_MAX_PACKS_PER_USER = 40
_COMPONENTS = ("look", "motion", "sound", "cards")


class PackIn(BaseModel):
    name: str
    look: str = "news"
    motion: str = "news"
    sound: str = "news"
    cards: str = "news"


def _row_dict(r: models.UserStylePack) -> dict:
    spec = r.spec or {}
    return {"id": r.id, "name": r.name,
            **{c: spec.get(c, "news") for c in _COMPONENTS},
            "key": f"user:{r.id}"}


@router.get("")
@router.get("/")
def list_my_packs(db: Session = Depends(get_db),
                  user: models.User = Depends(auth.current_user)):
    rows = (db.query(models.UserStylePack)
            .filter(models.UserStylePack.owner_id == user.id)
            .order_by(models.UserStylePack.id.desc()).all())
    return [_row_dict(r) for r in rows]


@router.post("")
@router.post("/")
def create_pack(body: PackIn, db: Session = Depends(get_db),
                user: models.User = Depends(auth.current_user)):
    from pipeline_v4.trailer_styles import STYLES
    name = (body.name or "").strip()[:80]
    if not name:
        raise HTTPException(400, "Give your style a name.")
    spec = {}
    for c in _COMPONENTS:
        v = (getattr(body, c) or "news").strip().lower()
        if v not in STYLES:
            raise HTTPException(400, f"Unknown pack for {c}: {v!r}")
        spec[c] = v
    n = (db.query(models.UserStylePack)
         .filter(models.UserStylePack.owner_id == user.id).count())
    if n >= _MAX_PACKS_PER_USER:
        raise HTTPException(400, f"Style limit reached ({_MAX_PACKS_PER_USER}).")
    row = models.UserStylePack(owner_id=user.id, name=name, spec=spec)
    db.add(row)
    db.commit()
    db.refresh(row)
    return _row_dict(row)


@router.delete("/{pid}")
def delete_pack(pid: int, db: Session = Depends(get_db),
                user: models.User = Depends(auth.current_user)):
    row = db.get(models.UserStylePack, pid)
    if not row or row.owner_id != user.id:      # 404 for others' packs
        raise HTTPException(404, "not found")
    db.delete(row)
    db.commit()
    return {"ok": True}


def resolve_user_style(style: str, user_id: int, db: Session):
    """'user:<id>' → composed TrailerStyle (owner-checked). None if the
    value isn't a user style; raises 404 on someone else's/missing id."""
    s = (style or "").strip().lower()
    if not s.startswith("user:"):
        return None
    try:
        pid = int(s.split(":", 1)[1])
    except ValueError:
        raise HTTPException(404, "unknown style")
    row = db.get(models.UserStylePack, pid)
    if not row or row.owner_id != user_id:
        raise HTTPException(404, "unknown style")
    from pipeline_v4.trailer_styles import compose_pack
    return compose_pack(row.spec or {}, name=row.name)


# ── Admin: EVERY user creation in one place (packs + templates) ──────

@router.get("/admin/user-creations")
def admin_user_creations(db: Session = Depends(get_db),
                         _: models.User = Depends(auth.admin_required)):
    packs = (db.query(models.UserStylePack, models.User.email)
             .join(models.User, models.User.id == models.UserStylePack.owner_id)
             .order_by(models.UserStylePack.id.desc()).limit(500).all())
    tpls = (db.query(models.CustomTemplate, models.User.email)
            .join(models.User, models.User.id == models.CustomTemplate.owner_id)
            .filter(models.CustomTemplate.status != "disabled")
            .order_by(models.CustomTemplate.created_at.desc())
            .limit(500).all())
    return {
        "style_packs": [{**_row_dict(r), "owner": email}
                        for (r, email) in packs],
        "templates": [{"id": t.id, "name": t.name, "owner": email,
                       "visibility": t.visibility,
                       "preview_url": f"/api/templates/{t.id}/preview"}
                      for (t, email) in tpls],
    }
