"""Channels router — CRUD for YouTube channel profiles.

Each channel drives SEO generation (title formula, footer, fixed tags,
mandatory hashtags) and upload targeting (linked OAuth token).
"""
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

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
    logo_asset_id: Optional[int] = None
    # "postiz" | "kaizer" | None (= use system default)
    upload_provider: Optional[str] = None
    # Per-channel watermark (applied at upload time by the worker).
    watermark_text: str = ""
    watermark_opacity: float = 0.35
    watermark_position: str = "top-right"
    # Per-channel social links injected into the SEO description footer.
    socials: dict = Field(default_factory=dict)

    @field_validator("upload_provider")
    @classmethod
    def _provider_must_be_valid(cls, v):
        if v is None or v == "":
            return None
        v = str(v).strip().lower()
        if v not in {"postiz", "kaizer"}:
            raise ValueError("upload_provider must be 'postiz', 'kaizer', or null")
        return v

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
    # Pass `null` explicitly to clear the logo.  Pass an int to set it to a
    # UserAsset (ownership validated server-side).
    logo_asset_id: Optional[int] = None
    # Per-channel INTRO video (UserAsset id) — prepended to the branded clip
    # at publish (anti-duplicate). Same semantics as logo: null clears, int
    # sets (ownership validated), mirrored to sibling account profiles.
    intro_asset_id: Optional[int] = None
    # Per-channel YouTube publish defaults (applied to every upload to this
    # channel so the operator needn't open YouTube Studio).
    yt_category_id: Optional[str] = None
    yt_playlist_id: Optional[str] = None
    yt_default_language: Optional[str] = None
    yt_made_for_kids: Optional[bool] = None
    yt_license: Optional[str] = None
    # "postiz" | "kaizer" | "" (= clear → fall back to system default)
    upload_provider: Optional[str] = None
    # Postiz integration id this channel delivers to (when
    # upload_provider='postiz'). Bind it from the Postiz integrations list.
    postiz_integration_id: Optional[str] = None
    # Per-channel watermark + socials. None = leave existing values.
    watermark_text: Optional[str] = None
    watermark_opacity: Optional[float] = None
    watermark_position: Optional[str] = None
    socials: Optional[dict] = None

    @field_validator("upload_provider")
    @classmethod
    def _provider_must_be_valid_patch(cls, v):
        if v is None or v == "":
            return None
        v = str(v).strip().lower()
        if v not in {"postiz", "kaizer"}:
            raise ValueError("upload_provider must be 'postiz', 'kaizer', or null")
        return v

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
    # Logo preview — same resolution order the upload worker uses:
    # OAuthToken.logo_asset_id first (the "VIDEO OVERLAY LOGO" set on
    # the My YouTube Accounts card), then Channel.logo_asset_id (style-
    # profile-level fallback). Without this, the legacy DownloadModal
    # mislabels every channel "no logo" even when the OAuth-token logo
    # IS set and would actually be applied.
    logo_asset = None
    effective_logo_asset_id = None
    try:
        from sqlalchemy.orm import object_session
        sess = object_session(c)
        if sess is not None:
            for source_id in (
                (tok.logo_asset_id if tok and getattr(tok, "logo_asset_id", None) else None),
                c.logo_asset_id,
            ):
                if not source_id:
                    continue
                la = sess.query(models.UserAsset).filter(models.UserAsset.id == source_id).first()
                if la:
                    effective_logo_asset_id = la.id
                    logo_asset = {
                        "id":       la.id,
                        "filename": la.filename,
                        "url":      la.storage_url or (f"/api/file/?path={la.file_path}" if la.file_path else ""),
                        "thumb_url": (
                            getattr(la, "thumb_storage_url", "")
                            or la.storage_url
                            or (f"/api/file/?path={la.thumb_path}" if la.thumb_path else "")
                        ),
                    }
                    break
    except Exception:
        logo_asset = None
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
        "logo_asset_id": c.logo_asset_id,
        # Per-channel intro video (UserAsset id), or null. Prepended to the
        # branded clip at publish; registered in the same brand modal as logo.
        "intro_asset_id": getattr(c, "intro_asset_id", None),
        # Per-channel YouTube publish defaults.
        "yt_category_id": getattr(c, "yt_category_id", None),
        "yt_playlist_id": getattr(c, "yt_playlist_id", None),
        "yt_default_language": getattr(c, "yt_default_language", None),
        "yt_made_for_kids": getattr(c, "yt_made_for_kids", None),
        "yt_license": getattr(c, "yt_license", None),
        # `effective_logo_asset_id` reflects whatever logo the upload
        # worker would actually apply — OAuthToken first, Channel
        # second — so UIs can label "logo configured" correctly. Stays
        # null only when truly no logo is set anywhere.
        "effective_logo_asset_id": effective_logo_asset_id,
        "logo":          logo_asset,
        # null = "use system default" — the UI shows the resolved
        # value via the system-settings endpoint when null.
        "upload_provider": c.upload_provider,
        "postiz_integration_id": getattr(c, "postiz_integration_id", None),
        "watermark_text":     getattr(c, "watermark_text", "") or "",
        "watermark_opacity":  float(getattr(c, "watermark_opacity", 0.35) or 0.35),
        "watermark_position": getattr(c, "watermark_position", "top-right") or "top-right",
        "socials":            getattr(c, "socials", None) or {},
        "created_at": c.created_at.isoformat() if c.created_at else None,
        "updated_at": c.updated_at.isoformat() if c.updated_at else None,
        "connected": tok is not None and bool(tok.refresh_token_enc),
        # 'account' = a connected YouTube channel you PUBLISH to (owns
        # branding). 'style' = a competitor/style reference used only to
        # generate SEO. Lets the UI cleanly split into two tabs.
        "kind": (getattr(c, "kind", None)
                 or ("account" if (tok is not None and bool(tok.refresh_token_enc)) else "style")),
        "youtube_channel_id": tok.google_channel_id if tok else "",
        "youtube_channel_title": tok.google_channel_title if tok else "",
        # Cached at OAuth time; lets the publish UI show the real YT
        # avatar + handle next to the style-profile name instead of
        # just "Personal 3".
        "youtube_channel_thumbnail_url": (tok.channel_thumbnail_url if tok else "") or "",
        "youtube_channel_custom_url":    (tok.channel_custom_url if tok else "") or "",
        "connected_at": tok.connected_at.isoformat() if tok and tok.connected_at else None,
        # Many-to-many: all destinations this profile is permitted to publish to.
        # Auto-includes the profile's own oauth-token destination.
        "allowed_destinations": sorted({
            pd.google_channel_id
            for pd in (c.__dict__.get("_allowed_dests_cache")  # set by bulk loader
                       or [])
        }) if "_allowed_dests_cache" in c.__dict__ else None,
        # Rich metadata — one entry per ProfileDestination, with cached
        # YouTube channel-name / avatar / sub-count populated at OAuth
        # time. The frontend uses this to render the multi-channel
        # picker without re-calling channels.list.
        "destinations": [
            {
                "google_channel_id":     pd.google_channel_id,
                "title":                 pd.channel_title or "",
                "thumbnail_url":         pd.channel_thumbnail_url or "",
                "custom_url":            pd.channel_custom_url or "",
                "subscriber_count":      int(pd.subscriber_count or 0),
                "video_count":           int(pd.video_count or 0),
                "enabled":               bool(getattr(pd, "enabled", True)),
                "is_primary":            (tok is not None
                                          and pd.google_channel_id
                                          and tok.google_channel_id == pd.google_channel_id),
            }
            for pd in (c.__dict__.get("_allowed_dests_cache") or [])
        ] if "_allowed_dests_cache" in c.__dict__ else None,
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
def list_channels(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
    kind: Optional[str] = Query(None, pattern="^(accounts|styles)$"),
):
    """List the user's channels.

    ``kind=accounts`` → only CONNECTED YouTube accounts (publish targets
    that own branding). ``kind=styles`` → only style references
    (competitor channels used to generate SEO). No ``kind`` → the full
    list (back-compat). The two tabs in the UI call this with each kind
    so connected accounts no longer leak into the style list.
    """
    rows = (
        db.query(models.Channel)
          .filter(models.Channel.user_id == user.id)
          .order_by(models.Channel.is_priority.desc(), models.Channel.name)
          .all()
    )
    if kind == "accounts":
        # Connected YouTube accounts you publish to — a live token is required.
        rows = [c for c in rows
                if c.oauth_token is not None and bool(c.oauth_token.refresh_token_enc)]
    elif kind == "styles":
        # Style references only — driven by the persistent `kind` column, NOT
        # by token presence. A disconnected account is kind='account', so it
        # never shows here even though its token is gone.
        rows = [c for c in rows if (getattr(c, "kind", None) or "account") == "style"]
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

    # Preserve cached metadata: snapshot existing rows (with their
    # cached title / thumbnail / sub-count populated at OAuth time)
    # before wipe, so re-created rows keep their pretty UI fields.
    existing_meta = {
        pd.google_channel_id: pd
        for pd in db.query(models.ProfileDestination).filter(
            models.ProfileDestination.profile_id == ch.id
        ).all()
    }

    db.query(models.ProfileDestination).filter(
        models.ProfileDestination.profile_id == ch.id
    ).delete()
    for gid in final:
        prev = existing_meta.get(gid)
        db.add(models.ProfileDestination(
            profile_id=ch.id,
            google_channel_id=gid,
            channel_title=(prev.channel_title if prev else "") or "",
            channel_thumbnail_url=(prev.channel_thumbnail_url if prev else "") or "",
            channel_custom_url=(prev.channel_custom_url if prev else "") or "",
            subscriber_count=int(prev.subscriber_count if prev else 0),
            video_count=int(prev.video_count if prev else 0),
            enabled=bool(prev.enabled if prev is not None else True),
        ))
    db.commit()
    return {"profile_id": ch.id, "google_channel_ids": sorted(final)}


@router.patch("/{channel_id}/destinations/{google_channel_id}")
def toggle_destination(
    channel_id: int,
    google_channel_id: str,
    payload: dict,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Toggle a single ProfileDestination on or off.

    Body: ``{ "enabled": true | false }``. Returns the updated row.
    Used by the multi-channel picker UI to enable / disable individual
    Brand Accounts without rewriting the full destinations list.
    """
    ch = db.query(models.Channel).filter(
        models.Channel.id == channel_id,
        models.Channel.user_id == user.id,
    ).first()
    if not ch:
        raise HTTPException(status_code=404, detail="Profile not found")

    pd = db.query(models.ProfileDestination).filter(
        models.ProfileDestination.profile_id == ch.id,
        models.ProfileDestination.google_channel_id == google_channel_id,
    ).first()
    if not pd:
        raise HTTPException(status_code=404, detail="Destination not found")

    desired = bool((payload or {}).get("enabled", True))
    pd.enabled = desired
    db.commit()
    return {
        "profile_id": ch.id,
        "google_channel_id": google_channel_id,
        "enabled": desired,
    }


def _validate_logo_ownership(db: Session, user_id: int, asset_id: Optional[int]) -> None:
    """Reject logo picks that reference assets outside the user's library."""
    if asset_id is None:
        return
    a = db.query(models.UserAsset).filter(
        models.UserAsset.id == asset_id,
        models.UserAsset.user_id == user_id,
    ).first()
    if not a:
        raise HTTPException(status_code=404, detail=f"Logo asset {asset_id} not found in your library")


class ApplyLogoBulk(BaseModel):
    channel_ids:   List[int]
    logo_asset_id: Optional[int] = None   # null = clear logo on all listed channels


@router.post("/apply-logo")
def apply_logo_to_channels(
    payload: ApplyLogoBulk,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Set (or clear) the same logo on many profiles at once.

    Powers the "use this logo for multiple channels" dropdown on the logo
    picker.  Only touches channels owned by the caller.  If the requested
    logo_asset_id doesn't belong to this user it's rejected outright.
    """
    if not payload.channel_ids:
        raise HTTPException(422, "channel_ids is required and must be non-empty")
    _validate_logo_ownership(db, user.id, payload.logo_asset_id)

    rows = (
        db.query(models.Channel)
          .filter(
              models.Channel.id.in_(payload.channel_ids),
              models.Channel.user_id == user.id,
          )
          .all()
    )
    found_ids = {c.id for c in rows}
    missing = [i for i in payload.channel_ids if i not in found_ids]
    if missing:
        raise HTTPException(404, f"Channel(s) not found: {missing}")

    for ch in rows:
        ch.logo_asset_id = payload.logo_asset_id
    db.commit()
    return {
        "updated": [c.id for c in rows],
        "logo_asset_id": payload.logo_asset_id,
    }


@router.post("/", status_code=201)
def create_channel(payload: ChannelIn, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    existing = db.query(models.Channel).filter(
        models.Channel.name == payload.name,
        models.Channel.user_id == user.id,
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Profile with name '{payload.name}' already exists")

    _validate_logo_ownership(db, user.id, payload.logo_asset_id)
    data = payload.model_dump()
    # If the caller didn't supply per-channel socials, seed them from
    # the user's global Settings socials so new channels start with
    # sensible defaults. The user can still override any field later
    # via the Channels page's ✦ Brand modal.
    if not data.get("socials"):
        try:
            import json as _json
            user_socials = _json.loads(user.socials or "{}") if isinstance(user.socials, str) else (user.socials or {})
            if isinstance(user_socials, dict) and any(user_socials.values()):
                data["socials"] = user_socials
        except Exception:
            pass
    # This endpoint creates STYLE references (competitor channels added on the
    # SEO Settings tab). Owned YouTube accounts are created by the OAuth connect
    # flow, not here. Mark it so it never gets confused with an account.
    data.pop("kind", None)
    ch = models.Channel(user_id=user.id, kind="style", **data)
    db.add(ch)
    db.commit()
    db.refresh(ch)
    return _to_dict(ch)


class BulkBrandIn(BaseModel):
    """Copy branding from ONE configured channel onto many others at once.
    Each toggled field is read from the source channel and written to every
    target. Lets the operator set a logo / watermark / socials on one channel
    then apply it to 30 others in a click instead of editing each by hand."""
    source_channel_id: int
    target_channel_ids: list[int] = Field(default_factory=list)
    copy_logo: bool = False
    copy_watermark: bool = False
    copy_socials: bool = False


@router.post("/bulk-brand")
def bulk_apply_branding(
    payload: BulkBrandIn,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    if not (payload.copy_logo or payload.copy_watermark or payload.copy_socials):
        raise HTTPException(400, "Select at least one of logo / watermark / socials to copy")
    if not payload.target_channel_ids:
        raise HTTPException(400, "Select at least one target channel")

    src = db.query(models.Channel).filter(
        models.Channel.id == payload.source_channel_id,
        models.Channel.user_id == user.id,
    ).first()
    if not src:
        raise HTTPException(404, "Source channel not found")

    # Resolve the source logo: the overlay logo lives on the OAuthToken (set
    # via the account-logo endpoint); fall back to the channel's own field.
    src_logo = None
    if payload.copy_logo:
        if src.oauth_token is not None:
            src_logo = getattr(src.oauth_token, "logo_asset_id", None)
        if src_logo is None:
            src_logo = getattr(src, "logo_asset_id", None)

    targets = db.query(models.Channel).filter(
        models.Channel.id.in_(payload.target_channel_ids),
        models.Channel.user_id == user.id,
    ).all()

    applied = 0
    for ch in targets:
        if ch.id == src.id:
            continue
        if payload.copy_logo:
            ch.logo_asset_id = src_logo
            # Mirror onto the linked YouTube account token (where the overlay
            # logo is actually read from at publish time), like set_account_logo.
            if ch.oauth_token is not None:
                ch.oauth_token.logo_asset_id = src_logo
        if payload.copy_watermark:
            ch.watermark_text     = src.watermark_text
            ch.watermark_opacity  = src.watermark_opacity
            ch.watermark_position = src.watermark_position
        if payload.copy_socials:
            ch.socials = dict(src.socials or {})
        applied += 1

    db.commit()
    fields = [f for f, on in (("logo", payload.copy_logo),
                              ("watermark", payload.copy_watermark),
                              ("socials", payload.copy_socials)) if on]
    return {"ok": True, "applied": applied, "fields": fields,
            "source_channel_id": src.id}


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

    if "logo_asset_id" in updates:
        _validate_logo_ownership(db, user.id, updates["logo_asset_id"])
    if "intro_asset_id" in updates:
        _validate_logo_ownership(db, user.id, updates["intro_asset_id"])

    # Postiz binding must reference an integration THIS user connected
    # (per-user isolation in the shared Postiz org). Empty / None = unbind,
    # always allowed.
    if updates.get("postiz_integration_id"):
        from services.postiz_scope import team_user_ids as _pz_team
        _iid = str(updates["postiz_integration_id"]).strip()
        _team = _pz_team(db, user.id)
        _owned = db.query(models.PostizIntegration.id).filter(
            models.PostizIntegration.user_id.in_(_team),
            models.PostizIntegration.integration_id == _iid,
        ).first()
        if not _owned:
            raise HTTPException(
                status_code=403,
                detail="You can only bind a Postiz channel your team connected",
            )

    for key, val in updates.items():
        setattr(ch, key, val)

    # Account-level branding consistency: when this row is a CONNECTED
    # account and the edit touches brand fields, mirror them onto every
    # sibling profile of the SAME real YouTube account (same
    # google_channel_id). The user thinks of branding as belonging to
    # the account, so all its duplicate profiles must stay in sync.
    _BRAND_FIELDS = {
        "logo_asset_id", "intro_asset_id", "watermark_text", "watermark_opacity",
        "watermark_position", "socials",
    }
    brand_updates = {k: v for k, v in updates.items() if k in _BRAND_FIELDS}
    tok = ch.oauth_token
    gcid = (getattr(tok, "google_channel_id", "") or "").strip() if tok else ""
    if brand_updates and gcid and tok and tok.refresh_token_enc:
        siblings = (
            db.query(models.Channel)
            .join(models.OAuthToken, models.OAuthToken.channel_id == models.Channel.id)
            .filter(
                models.Channel.user_id == user.id,
                models.Channel.id != ch.id,
                models.OAuthToken.google_channel_id == gcid,
                # Only mirror onto REAL connected accounts — never a style
                # reference that happens to carry a stale/empty token row.
                models.OAuthToken.refresh_token_enc.isnot(None),
                models.OAuthToken.refresh_token_enc != "",
            )
            .all()
        )
        for s in siblings:
            for key, val in brand_updates.items():
                setattr(s, key, val)
            # The logo specifically also lives on the OAuth token (the
            # 'VIDEO OVERLAY LOGO' the upload worker reads first).
            if "logo_asset_id" in brand_updates and s.oauth_token is not None:
                s.oauth_token.logo_asset_id = brand_updates["logo_asset_id"]
        if "logo_asset_id" in brand_updates and tok is not None:
            tok.logo_asset_id = brand_updates["logo_asset_id"]

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

    # Clean up dependents that have NO ORM cascade and would otherwise
    # block the FK delete. The connect flow's oauth_states CSRF row is the
    # cause of the "abandoned connect → Personal N → 500 on delete" bug:
    # an incomplete OAuth leaves an oauth_states row referencing this
    # channel. profile_destinations also lack a cascade. The oauth_token /
    # upload_jobs / corpus rows cascade via their relationships.
    db.query(models.OAuthState).filter(
        models.OAuthState.channel_id == channel_id
    ).delete(synchronize_session=False)
    db.query(models.ProfileDestination).filter(
        models.ProfileDestination.profile_id == channel_id
    ).delete(synchronize_session=False)

    db.delete(ch)
    try:
        db.commit()
    except IntegrityError:
        # Some other table still references this channel (e.g. publish
        # history). Fail cleanly with a 409 instead of a raw 500.
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="Cannot delete — this profile is still referenced by other "
                   "records (e.g. publish history). Remove those first.",
        )
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

    Own accounts study via their Connected identity; a study/writing-voice
    channel (a competitor, kind='style') studies its PUBLIC top videos via
    its @handle — no OAuth needed (you can't connect someone else's channel).
    422 only if there's neither a connection nor a resolvable handle.
    """
    ch = db.query(models.Channel).filter(models.Channel.id == channel_id).first()
    if not ch:
        raise HTTPException(status_code=404, detail="Channel not found")
    tok = ch.oauth_token
    connected = bool(tok and tok.google_channel_id)
    has_handle = bool((getattr(ch, "handle", "") or "").strip())
    if not connected and not has_handle:
        raise HTTPException(
            status_code=422,
            detail=(f"'{ch.name}' has no YouTube identity — add its @handle or "
                    f"channel URL (Edit) so we can study its public videos."),
        )
    if not connected:
        # Resolving an @handle → channel id needs the public Data API key; a
        # pasted UC-id / channel URL resolves with no API call, so don't gate it.
        import re as _re
        _h = (getattr(ch, "handle", "") or "").strip()
        _needs_resolve = not _re.fullmatch(r"UC[A-Za-z0-9_-]{20,30}", _h) \
            and "channel/UC" not in _h
        if _needs_resolve:
            from config import settings as _settings
            if not _settings.yt_data_api_key:
                raise HTTPException(
                    status_code=422,
                    detail=("Studying a channel by its public @handle needs "
                            "YOUTUBE_DATA_API_KEY set on the server."),
                )

    background.add_task(_run_corpus_refresh, channel_id)
    return {
        "channel_id": channel_id,
        "status":     "queued",
        "note":       "Refresh runs in the background. Poll GET /api/channels/{id}/corpus for the result (usually under 15s).",
    }
