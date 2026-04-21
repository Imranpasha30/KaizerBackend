"""Channels router — CRUD for YouTube channel profiles.

Each channel drives SEO generation (title formula, footer, fixed tags,
mandatory hashtags) and upload targeting (linked OAuth token).
"""
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from database import SessionLocal, get_db
import models
import auth
from learning import corpus as learning_corpus


router = APIRouter(prefix="/api/channels", tags=["channels"])


# ─── Schemas ──────────────────────────────────────────────────────────────────

class ChannelIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    handle: str = Field("", max_length=100)
    language: str = Field("te", max_length=10)
    title_formula: str = ""
    desc_style: str = Field("hook_first", max_length=50)
    footer: str = ""
    fixed_tags: List[str] = Field(default_factory=list)
    hashtags: List[str] = Field(default_factory=list)
    mandatory_hashtags: List[str] = Field(default_factory=list)
    is_priority: bool = False

    @field_validator("fixed_tags", "hashtags", "mandatory_hashtags")
    @classmethod
    def _strip_and_dedupe(cls, v: List[str]) -> List[str]:
        seen = set()
        cleaned = []
        for item in v or []:
            s = (item or "").strip()
            if not s or s.lower() in seen:
                continue
            seen.add(s.lower())
            cleaned.append(s)
        return cleaned

    @field_validator("hashtags", "mandatory_hashtags")
    @classmethod
    def _ensure_hashtag_prefix(cls, v: List[str]) -> List[str]:
        return [s if s.startswith("#") else f"#{s}" for s in v]


class ChannelPatch(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    handle: Optional[str] = Field(None, max_length=100)
    language: Optional[str] = Field(None, max_length=10)
    title_formula: Optional[str] = None
    desc_style: Optional[str] = Field(None, max_length=50)
    footer: Optional[str] = None
    fixed_tags: Optional[List[str]] = None
    hashtags: Optional[List[str]] = None
    mandatory_hashtags: Optional[List[str]] = None
    is_priority: Optional[bool] = None

    @field_validator("fixed_tags", "hashtags", "mandatory_hashtags")
    @classmethod
    def _strip_and_dedupe(cls, v):
        if v is None:
            return v
        seen = set()
        cleaned = []
        for item in v:
            s = (item or "").strip()
            if not s or s.lower() in seen:
                continue
            seen.add(s.lower())
            cleaned.append(s)
        return cleaned

    @field_validator("hashtags", "mandatory_hashtags")
    @classmethod
    def _ensure_hashtag_prefix(cls, v):
        if v is None:
            return v
        return [s if s.startswith("#") else f"#{s}" for s in v]


def _to_dict(c: models.Channel) -> dict:
    tok = c.oauth_token
    return {
        "id": c.id,
        "name": c.name,
        "handle": c.handle or "",
        "language": c.language or "te",
        "title_formula": c.title_formula or "",
        "desc_style": c.desc_style or "hook_first",
        "footer": c.footer or "",
        "fixed_tags": c.fixed_tags or [],
        "hashtags": c.hashtags or [],
        "mandatory_hashtags": c.mandatory_hashtags or [],
        "is_priority": bool(c.is_priority),
        "created_at": c.created_at.isoformat() if c.created_at else None,
        "updated_at": c.updated_at.isoformat() if c.updated_at else None,
        "connected": tok is not None and bool(tok.refresh_token_enc),
        "youtube_channel_id": tok.google_channel_id if tok else "",
        "youtube_channel_title": tok.google_channel_title if tok else "",
        "connected_at": tok.connected_at.isoformat() if tok and tok.connected_at else None,
        # Many-to-many: all destinations this profile is permitted to publish to.
        # Auto-includes the profile's own oauth-token destination.
        "allowed_destinations": sorted({
            pd.google_channel_id
            for pd in (c.__dict__.get("_allowed_dests_cache")  # set by bulk loader
                       or [])
        }) if "_allowed_dests_cache" in c.__dict__ else None,
    }


def _load_allowed_destinations(db, profiles: list) -> None:
    """Attach `_allowed_dests_cache` to each Channel so _to_dict can include it
    without N+1 queries."""
    ids = [p.id for p in profiles]
    if not ids:
        return
    rows = (
        db.query(models.ProfileDestination)
          .filter(models.ProfileDestination.profile_id.in_(ids))
          .all()
    )
    by_profile: dict[int, list] = {}
    for r in rows:
        by_profile.setdefault(r.profile_id, []).append(r)
    for p in profiles:
        p.__dict__["_allowed_dests_cache"] = by_profile.get(p.id, [])


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/")
def list_channels(db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    rows = (
        db.query(models.Channel)
          .filter(models.Channel.user_id == user.id)
          .order_by(models.Channel.is_priority.desc(), models.Channel.name)
          .all()
    )
    _load_allowed_destinations(db, rows)
    return [_to_dict(c) for c in rows]


# ─── Many-to-many destinations per profile ───────────────────────────────

@router.put("/{channel_id}/destinations")
def set_profile_destinations(
    channel_id: int,
    payload: dict,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Replace the set of YouTube destinations this profile can publish to.

    Body: `{ "google_channel_ids": ["UC...", "UC..."] }`.
    Only destinations the user actually owns (has an OAuthToken on some
    profile of theirs) are accepted.  The profile's own primary destination
    (if any) is auto-included so toggling it off never orphans the link.
    """
    ch = db.query(models.Channel).filter(
        models.Channel.id == channel_id,
        models.Channel.user_id == user.id,
    ).first()
    if not ch:
        raise HTTPException(status_code=404, detail="Profile not found")

    requested = list((payload or {}).get("google_channel_ids") or [])
    requested = [str(x).strip() for x in requested if str(x).strip()]

    # The user's owned destinations = google_channel_ids from any of their
    # profiles' oauth tokens.  Only these are allowed targets.
    owned_ids = {
        tok.google_channel_id
        for tok in (
            db.query(models.OAuthToken)
              .join(models.Channel, models.Channel.id == models.OAuthToken.channel_id)
              .filter(
                  models.Channel.user_id == user.id,
                  models.OAuthToken.google_channel_id.isnot(None),
                  models.OAuthToken.refresh_token_enc.isnot(None),
              )
              .all()
        )
        if tok.google_channel_id
    }
    final = {gid for gid in requested if gid in owned_ids}

    # Always include this profile's own primary destination if it has one
    if ch.oauth_token and ch.oauth_token.google_channel_id:
        final.add(ch.oauth_token.google_channel_id)

    # Wipe existing links + rewrite
    db.query(models.ProfileDestination).filter(
        models.ProfileDestination.profile_id == ch.id
    ).delete()
    for gid in final:
        db.add(models.ProfileDestination(profile_id=ch.id, google_channel_id=gid))
    db.commit()
    return {"profile_id": ch.id, "google_channel_ids": sorted(final)}


@router.post("/", status_code=201)
def create_channel(payload: ChannelIn, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    existing = db.query(models.Channel).filter(
        models.Channel.name == payload.name,
        models.Channel.user_id == user.id,
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Profile with name '{payload.name}' already exists")

    ch = models.Channel(user_id=user.id, **payload.model_dump())
    db.add(ch)
    db.commit()
    db.refresh(ch)
    return _to_dict(ch)


@router.get("/{channel_id}/")
def get_channel(channel_id: int, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    ch = db.query(models.Channel).filter(
        models.Channel.id == channel_id, models.Channel.user_id == user.id,
    ).first()
    if not ch:
        raise HTTPException(status_code=404, detail="Profile not found")
    return _to_dict(ch)


@router.patch("/{channel_id}/")
def update_channel(channel_id: int, payload: ChannelPatch, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    ch = db.query(models.Channel).filter(
        models.Channel.id == channel_id, models.Channel.user_id == user.id,
    ).first()
    if not ch:
        raise HTTPException(status_code=404, detail="Profile not found")

    updates = payload.model_dump(exclude_unset=True)

    if "name" in updates and updates["name"] != ch.name:
        dup = db.query(models.Channel).filter(
            models.Channel.name == updates["name"],
            models.Channel.user_id == user.id,
            models.Channel.id != channel_id,
        ).first()
        if dup:
            raise HTTPException(status_code=409, detail=f"Profile with name '{updates['name']}' already exists")

    for key, val in updates.items():
        setattr(ch, key, val)

    db.commit()
    db.refresh(ch)
    return _to_dict(ch)


@router.delete("/{channel_id}/")
def delete_channel(channel_id: int, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    ch = db.query(models.Channel).filter(
        models.Channel.id == channel_id, models.Channel.user_id == user.id,
    ).first()
    if not ch:
        raise HTTPException(status_code=404, detail="Profile not found")

    queued = (
        db.query(models.UploadJob)
          .filter(
              models.UploadJob.channel_id == channel_id,
              models.UploadJob.status.in_(["queued", "uploading"]),
          )
          .count()
    )
    if queued:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete — {queued} upload job(s) still active. Cancel them first.",
        )

    db.delete(ch)
    db.commit()
    return {"deleted": channel_id}


# ─── Learning corpus (Phase 7) ────────────────────────────────────────────────

@router.get("/{channel_id}/corpus")
def get_channel_corpus(channel_id: int, db: Session = Depends(get_db)):
    ch = db.query(models.Channel).filter(models.Channel.id == channel_id).first()
    if not ch:
        raise HTTPException(status_code=404, detail="Channel not found")
    row = ch.corpus
    if row is None:
        return {"channel_id": channel_id, "payload": None, "refreshed_at": None}
    return {
        "channel_id":   channel_id,
        "payload":      row.payload or {},
        "refreshed_at": row.refreshed_at.isoformat() if row.refreshed_at else None,
    }


def _run_corpus_refresh(channel_id: int) -> None:
    """Background task — opens its own session to survive request lifecycle."""
    db = SessionLocal()
    try:
        learning_corpus.refresh_channel(db, channel_id)
    except learning_corpus.CorpusError as e:
        print(f"[learn] channel {channel_id} failed: {e}")
    except Exception as e:
        print(f"[learn] channel {channel_id} unexpected error: {e}")
    finally:
        db.close()


@router.post("/{channel_id}/learn", status_code=202)
def learn_channel(
    channel_id: int,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Kick off a one-off corpus refresh for this channel. Non-blocking.

    422 if the channel isn't connected to YouTube yet, since we need its
    google_channel_id to list uploads.
    """
    ch = db.query(models.Channel).filter(models.Channel.id == channel_id).first()
    if not ch:
        raise HTTPException(status_code=404, detail="Channel not found")
    tok = ch.oauth_token
    if not tok or not tok.google_channel_id:
        raise HTTPException(
            status_code=422,
            detail=f"Channel '{ch.name}' is not connected to YouTube. Connect it first to mine its top videos.",
        )

    background.add_task(_run_corpus_refresh, channel_id)
    return {
        "channel_id": channel_id,
        "status":     "queued",
        "note":       "Refresh runs in the background. Poll GET /api/channels/{id}/corpus for the result (usually under 15s).",
    }
