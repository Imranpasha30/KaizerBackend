"""Channel-groups router — user-defined presets for publish fan-out.

Flow:
  - User creates a group like "English" with 2-3 of their YouTube destinations.
  - PublishModal shows the group as a one-click preset.
  - Picking the group auto-selects every destination in it.

Destinations are stored as `google_channel_id`s (real YouTube channels, not
profile ids) so rename/relink of style profiles doesn't invalidate the group.
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from database import get_db
import models
import auth


router = APIRouter(prefix="/api/channel-groups", tags=["channel-groups"])


# ─── Schemas ──────────────────────────────────────────────────────────────

class ChannelGroupIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""
    google_channel_ids: List[str] = Field(default_factory=list)

    @field_validator("google_channel_ids")
    @classmethod
    def _strip_and_dedupe(cls, v: List[str]) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for x in v or []:
            s = (x or "").strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out


class ChannelGroupPatch(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    google_channel_ids: Optional[List[str]] = None

    @field_validator("google_channel_ids")
    @classmethod
    def _strip_and_dedupe(cls, v):
        if v is None:
            return v
        seen: set[str] = set()
        out: List[str] = []
        for x in v:
            s = (x or "").strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out


def _to_dict(g: models.ChannelGroup) -> dict:
    return {
        "id":                  g.id,
        "name":                g.name,
        "description":         g.description or "",
        "google_channel_ids":  list(g.google_channel_ids or []),
        "created_at":          g.created_at.isoformat() if g.created_at else None,
        "updated_at":          g.updated_at.isoformat() if g.updated_at else None,
    }


def _user_owned_destinations(db: Session, user_id: int) -> set[str]:
    """All google_channel_ids the user's OAuth tokens cover — used to block
    groups from containing destinations they don't actually own."""
    rows = (
        db.query(models.OAuthToken)
          .join(models.Channel, models.Channel.id == models.OAuthToken.channel_id)
          .filter(
              models.Channel.user_id == user_id,
              models.OAuthToken.refresh_token_enc.isnot(None),
              models.OAuthToken.google_channel_id.isnot(None),
          )
          .all()
    )
    return {r.google_channel_id for r in rows if r.google_channel_id}


# ─── Endpoints ────────────────────────────────────────────────────────────

@router.get("/")
def list_groups(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    rows = (
        db.query(models.ChannelGroup)
          .filter(models.ChannelGroup.user_id == user.id)
          .order_by(models.ChannelGroup.name)
          .all()
    )
    return [_to_dict(g) for g in rows]


@router.post("/", status_code=201)
def create_group(
    payload: ChannelGroupIn,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    # Name uniqueness per user
    existing = db.query(models.ChannelGroup).filter(
        models.ChannelGroup.user_id == user.id,
        models.ChannelGroup.name == payload.name,
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Group '{payload.name}' already exists")

    # Filter out destinations the user doesn't own (defensive — client might
    # send stale ids)
    owned = _user_owned_destinations(db, user.id)
    valid_dests = [g for g in payload.google_channel_ids if g in owned]

    g = models.ChannelGroup(
        user_id=user.id,
        name=payload.name,
        description=payload.description or "",
        google_channel_ids=valid_dests,
    )
    db.add(g); db.commit(); db.refresh(g)
    return _to_dict(g)


@router.patch("/{group_id}")
def update_group(
    group_id: int,
    payload: ChannelGroupPatch,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    g = db.query(models.ChannelGroup).filter(
        models.ChannelGroup.id == group_id,
        models.ChannelGroup.user_id == user.id,
    ).first()
    if not g:
        raise HTTPException(status_code=404, detail="Group not found")

    updates = payload.model_dump(exclude_unset=True)

    if "name" in updates and updates["name"] != g.name:
        dup = db.query(models.ChannelGroup).filter(
            models.ChannelGroup.user_id == user.id,
            models.ChannelGroup.name == updates["name"],
            models.ChannelGroup.id != group_id,
        ).first()
        if dup:
            raise HTTPException(status_code=409, detail=f"Group '{updates['name']}' already exists")

    if "google_channel_ids" in updates:
        owned = _user_owned_destinations(db, user.id)
        updates["google_channel_ids"] = [
            gid for gid in updates["google_channel_ids"] if gid in owned
        ]

    for key, val in updates.items():
        setattr(g, key, val)

    db.commit(); db.refresh(g)
    return _to_dict(g)


@router.delete("/{group_id}")
def delete_group(
    group_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    g = db.query(models.ChannelGroup).filter(
        models.ChannelGroup.id == group_id,
        models.ChannelGroup.user_id == user.id,
    ).first()
    if not g:
        raise HTTPException(status_code=404, detail="Group not found")
    db.delete(g); db.commit()
    return {"deleted": group_id}
