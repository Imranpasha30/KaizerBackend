"""Admin-only endpoints for cross-platform posting via Postiz.

All routes require ``is_admin=True`` so regular Kaizer users don't
see the cross-post option until we've completed full verification on
each platform (Twitter / IG / LinkedIn / TikTok / …).

Endpoints:
  GET  /api/postiz/status                 → is Postiz reachable + how
                                            many platforms connected
  GET  /api/postiz/integrations           → list connected platforms
                                            (Twitter / IG / LinkedIn / …)
  POST /api/postiz/schedule               → schedule a post across N
                                            integrations with a video URL

The frontend hides this UI behind a `user.is_admin` check too
(belt-and-suspenders so a deep-linker can't render a leaky modal),
but the backend gate here is the source of truth.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

import auth
import models
from database import get_db
from clients import postiz as postiz_client
from services.postiz_scope import team_user_ids as _team_user_ids

logger = logging.getLogger("kaizer.routers.postiz")
router = APIRouter(prefix="/api/postiz", tags=["postiz"])

# The Postiz API key is intentionally env-ONLY (POSTIZ_API_KEY in the
# backend .env). It is never exposed or settable over the API — the
# operator manages it on the server. We only ever READ connectivity
# status + the integrations list here.


# ─── Models ──────────────────────────────────────────────────────────────────

class PostizSchedule(BaseModel):
    integration_ids: list[str]
    text: str = ""
    media_url: Optional[str] = None       # public URL (R2) Postiz fetches
    schedule_at_iso: Optional[str] = None # ISO-8601 UTC; None = now
    type: str = "now"                      # "draft" | "scheduled" | "now"


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/status")
def status(_user: models.User = Depends(auth.admin_required)) -> dict:
    """Cheap reachability probe — used by the admin UI to decide
    whether to render the Cross-post toggle. No data leak: gated by
    admin_required so non-admins can't even see if it's configured."""
    if not postiz_client.is_enabled():
        return {
            "enabled": False,
            "reason": "POSTIZ_API_KEY not set in backend env",
        }
    try:
        integrations = postiz_client.list_integrations()
    except postiz_client.PostizAuthError as e:
        return {"enabled": False, "reason": f"auth: {e}"}
    except postiz_client.PostizError as e:
        return {"enabled": False, "reason": f"unreachable: {e}"}
    return {
        "enabled": True,
        "integration_count": len(integrations),
        "providers": sorted({i.get("provider", "") for i in integrations
                              if i.get("provider")}),
    }


@router.get("/config")
def get_config(_user: models.User = Depends(auth.admin_required)) -> dict:
    """Connectivity status for the Postiz delivery panel — whether the
    env key is set and reachable. The key itself is NEVER returned (not
    even masked); it is managed only via POSTIZ_API_KEY in the backend
    .env. Read-only: there is no endpoint to set the key over the API."""
    key = os.environ.get("POSTIZ_API_KEY", "").strip()
    # Report the SAME base URL the client will actually use (single source
    # of truth) so the admin panel can't display a URL that differs from
    # where requests really go.
    base = postiz_client.base_url()
    out = {
        "key_set": bool(key),
        "base_url": base,
        "reachable": False,
        "reason": "" if key else "no key set",
        "integration_count": 0,
    }
    if key:
        try:
            ints = postiz_client.list_integrations()
            out["reachable"] = True
            out["integration_count"] = len(ints)
        except postiz_client.PostizAuthError as e:
            out["reason"] = f"auth rejected: {e}"
        except postiz_client.PostizError as e:
            out["reason"] = f"unreachable: {e}"
    return out


@router.get("/integrations")
def integrations(_user: models.User = Depends(auth.admin_required)) -> list[dict]:
    """List connected platforms Postiz can post to on this user's
    behalf. Only the fields the UI needs are returned — no tokens."""
    try:
        raw = postiz_client.list_integrations()
    except postiz_client.PostizAuthError as e:
        raise HTTPException(status_code=503, detail=f"Postiz auth: {e}")
    except postiz_client.PostizError as e:
        raise HTTPException(status_code=503, detail=f"Postiz unreachable: {e}")
    return [
        {
            "id":         i.get("id", ""),
            "name":       i.get("name", ""),
            "provider":   i.get("provider", ""),
            "picture":    i.get("picture", ""),
            "identifier": i.get("identifier", ""),
        }
        for i in raw
    ]


@router.post("/schedule")
def schedule(
    payload: PostizSchedule,
    _user: models.User = Depends(auth.admin_required),
) -> dict:
    """Schedule a post on the chosen Postiz integrations.

    ``media_url`` should be a publicly-reachable URL (R2 works); Postiz
    fetches the bytes server-side and uploads to each platform.
    """
    if not payload.integration_ids:
        raise HTTPException(status_code=400, detail="No integrations selected")

    # Postiz attaches media by INTERNAL id (POST /public/v1/upload), NOT by
    # URL — so if a media_url was supplied, fetch the bytes and upload them
    # to Postiz first to mint an id, then reference that id on the post.
    media_id: Optional[str] = None
    media_path: Optional[str] = None
    tmp_path: Optional[str] = None
    try:
        if payload.media_url:
            import tempfile
            import requests
            try:
                with requests.get(payload.media_url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    suffix = os.path.splitext(payload.media_url.split("?")[0])[1] or ".mp4"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                        tmp_path = tf.name
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            if chunk:
                                tf.write(chunk)
            except requests.RequestException as exc:
                raise HTTPException(status_code=502,
                                    detail=f"Could not fetch media_url: {exc}")
            uploaded = postiz_client.upload_file(tmp_path) or {}
            media_id = str(uploaded.get("id") or "") or None
            media_path = uploaded.get("path") or None
            if not media_id:
                raise HTTPException(status_code=502,
                                    detail="Postiz upload returned no media id")

        result = postiz_client.schedule_post(
            integration_ids=payload.integration_ids,
            text=payload.text or "",
            media_id=media_id,
            media_path=media_path,
            schedule_at_iso=payload.schedule_at_iso,
            type_=payload.type or "now",
        )
    except postiz_client.PostizAuthError as e:
        raise HTTPException(status_code=503, detail=f"Postiz auth: {e}")
    except postiz_client.PostizError as e:
        raise HTTPException(status_code=502, detail=f"Postiz scheduling failed: {e}")
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    logger.info("postiz.schedule ok: integrations=%s media=%s",
                payload.integration_ids, bool(media_id))
    return {"ok": True, "postiz_response": result}


# ═══════════════════════════════════════════════════════════════════════════
# Per-user channel connection (single business org + Kaizer-side ownership)
# ───────────────────────────────────────────────────────────────────────────
# Every Kaizer user connects their OWN socials into our ONE Postiz org (the
# env POSTIZ_API_KEY). Kaizer records which integration belongs to which user
# (models.PostizIntegration, integration_id globally unique) so a user can
# only ever see / bind / publish-to / disconnect the channels THEY connected.
# Users never log into or register with Postiz — they click Connect here,
# approve on the platform's own consent screen, and that's it.
# ═══════════════════════════════════════════════════════════════════════════

# Pre-connect snapshots live in the DB (models.PostizConnectSession), NOT
# process memory, so attribution survives multiple uvicorn workers (railway
# runs --workers 2). Channels are TEAM-SHARED: a connected channel belongs to
# the connector's team (AgencyTeam) and any teammate can see/bind/disconnect
# it; a SOLO user (in no team) gets a personal pool of just their own — one
# helper (_team_user_ids) covers both. Connects are serialized only ACROSS
# teams (a DIFFERENT team mid-connecting the same provider → 409), so a team
# never blocks its own members and concurrent same-team connects are safe.
_PENDING_TTL = 600  # seconds — max age of a pending connect session


def _owned_integration_ids(db: Session) -> set[str]:
    """All integration ids already claimed by ANY Kaizer user."""
    return {row[0] for row in db.query(models.PostizIntegration.integration_id).all()}


class PostizConnectIn(BaseModel):
    provider: str
    refresh: Optional[str] = None


@router.get("/me/status")
def me_status(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Reachability + how many channels THIS user has connected."""
    if not postiz_client.is_enabled():
        return {"enabled": False, "reason": "Postiz not configured", "integration_count": 0}
    try:
        org = postiz_client.list_integrations()
    except postiz_client.PostizAuthError as e:
        return {"enabled": False, "reason": f"auth: {e}", "integration_count": 0}
    except postiz_client.PostizError as e:
        return {"enabled": False, "reason": f"unreachable: {e}", "integration_count": 0}
    org_ids = {str(i.get("id")) for i in org if i.get("id")}
    team = _team_user_ids(db, user.id)
    mine = (db.query(models.PostizIntegration)
              .filter(models.PostizIntegration.user_id.in_(team)).all())
    live = [m for m in mine if m.integration_id in org_ids]
    return {"enabled": True, "integration_count": len(live)}


@router.get("/me/integrations")
def me_integrations(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> list[dict]:
    """List ONLY the channels this user connected (owner-filtered), enriched
    with live name/provider/picture from the org list. Ownership rows whose
    integration no longer exists in Postiz are pruned."""
    try:
        org = postiz_client.list_integrations()
    except postiz_client.PostizError:
        org = []
    org_by_id = {str(i.get("id")): i for i in org if i.get("id")}
    team = _team_user_ids(db, user.id)
    out: list[dict] = []
    stale: list = []
    for m in (db.query(models.PostizIntegration)
                .filter(models.PostizIntegration.user_id.in_(team))
                .order_by(models.PostizIntegration.created_at.desc()).all()):
        live = org_by_id.get(m.integration_id)
        if live is None:
            stale.append(m)   # deleted on the Postiz side — prune mapping
            continue
        out.append({
            "integration_id": m.integration_id,
            "provider":   live.get("provider") or m.provider,
            "name":       live.get("name") or m.name,
            "identifier": live.get("identifier") or m.identifier,
            "picture":    live.get("picture") or m.picture,
        })
    if stale:
        for m in stale:
            db.delete(m)
        db.commit()
    return out


@router.post("/me/connect")
def me_connect(
    payload: PostizConnectIn,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Start connecting a social channel for THIS user. Returns the OAuth
    authorize URL to open in a popup. Snapshots the org's current
    integration ids so the matching finalize attributes only the new one."""
    prov = (payload.provider or "").strip().lower()
    if not prov:
        raise HTTPException(status_code=400, detail="provider required")
    team = _team_user_ids(db, user.id)
    refresh_id = (payload.refresh or "").strip() or None
    if refresh_id:
        owned = (db.query(models.PostizIntegration)
                   .filter(models.PostizIntegration.user_id.in_(team),
                           models.PostizIntegration.integration_id == refresh_id)
                   .first())
        if not owned:
            raise HTTPException(status_code=403, detail="Not your team's integration")

    now = time.time()
    # Prune expired sessions + this user's own prior session for this provider.
    db.query(models.PostizConnectSession).filter(
        (models.PostizConnectSession.created_ts < now - _PENDING_TTL)
        | ((models.PostizConnectSession.user_id == user.id)
           & (models.PostizConnectSession.provider == prov))
    ).delete(synchronize_session=False)
    db.flush()

    # Cross-team serialization: 409 ONLY if a DIFFERENT team is mid-connecting
    # this provider (keeps cross-team attribution unambiguous). Same-team
    # concurrent connects are allowed — both channels land in the team pool,
    # so there's nothing to mis-attribute across tenants.
    other_team = (db.query(models.PostizConnectSession.id)
                    .filter(models.PostizConnectSession.provider == prov,
                            models.PostizConnectSession.created_ts >= now - _PENDING_TTL,
                            models.PostizConnectSession.user_id.notin_(team))
                    .first())
    if other_team:
        db.commit()   # persist the prune
        raise HTTPException(
            status_code=409,
            detail=f"Another team is connecting {prov} right now — please retry in a moment.",
        )

    try:
        before = postiz_client.list_integrations()
        res = postiz_client.social_connect_url(prov, refresh=refresh_id)
    except postiz_client.PostizAuthError as e:
        db.rollback()
        raise HTTPException(status_code=503, detail=f"Postiz auth: {e}")
    except postiz_client.PostizError as e:
        db.rollback()
        raise HTTPException(status_code=502, detail=f"Postiz connect failed: {e}")
    url = (res or {}).get("url") or ""
    if not url:
        db.rollback()
        raise HTTPException(status_code=502, detail="Postiz returned no connect URL")

    db.add(models.PostizConnectSession(
        user_id=user.id, provider=prov,
        before_ids=sorted({str(i.get("id")) for i in before if i.get("id")}),
        created_ts=now,
    ))
    db.commit()
    return {"url": url, "provider": prov}


@router.post("/me/connect/finalize")
def me_connect_finalize(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Call after the OAuth popup completes. Diffs the org's integrations
    against this user's pre-connect snapshot and claims the newly-appeared
    channel(s) for them — only ids NOT already owned by anyone (global
    uniqueness prevents stealing). Returns the claimed ids."""
    try:
        org = postiz_client.list_integrations()
    except postiz_client.PostizError as e:
        raise HTTPException(status_code=502, detail=f"Postiz unreachable: {e}")
    org_by_id = {str(i.get("id")): i for i in org if i.get("id")}

    # Active connect sessions for THIS user's TEAM (DB-backed → worker-safe).
    # No active session ⇒ claim NOTHING (never grab a channel blindly).
    now = time.time()
    team = _team_user_ids(db, user.id)
    sessions = (db.query(models.PostizConnectSession)
                  .filter(models.PostizConnectSession.user_id.in_(team),
                          models.PostizConnectSession.created_ts >= now - _PENDING_TTL)
                  .all())
    if not sessions:
        db.query(models.PostizConnectSession).filter(
            models.PostizConnectSession.user_id == user.id
        ).delete(synchronize_session=False)
        db.commit()
        return {"claimed": [], "count": 0, "pending": False}

    before_by_prov: dict[str, set] = {}
    for s in sessions:
        before_by_prov.setdefault(s.provider, set()).update(s.before_ids or [])

    owned_anywhere = _owned_integration_ids(db)
    claimed: list[str] = []
    claimed_provs: set[str] = set()
    for iid, info in org_by_id.items():
        if iid in owned_anywhere:
            continue                       # already owned by someone (incl. me)
        prov = (info.get("provider") or "").strip().lower()
        if prov not in before_by_prov:
            continue                       # not a provider this user is connecting
        if iid in before_by_prov[prov]:
            continue                       # existed before this connect
        db.add(models.PostizIntegration(
            user_id=user.id, integration_id=iid, provider=prov,
            name=(info.get("name") or "")[:255],
            identifier=(info.get("identifier") or "")[:255],
            picture=(info.get("picture") or "")[:500],
        ))
        claimed.append(iid)
        claimed_provs.add(prov)

    # Free MY connect slot(s) for the providers I just claimed (teammates'
    # sessions stay alive for their own polling).
    for s in sessions:
        if s.user_id == user.id and s.provider in claimed_provs:
            db.delete(s)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()   # extremely rare under per-provider serialization
        logger.warning("postiz.me.finalize: claim race for user=%s", user.id)
        return {"claimed": [], "count": 0, "pending": True}
    return {"claimed": claimed, "count": len(claimed), "pending": True}


@router.delete("/me/integrations/{integration_id}")
def me_disconnect(
    integration_id: str,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Disconnect a channel the user's TEAM connected: team-ownership checked,
    deleted on Postiz, mapping removed, and EVERY channel bound to it reverted
    to native so no dangling binding survives."""
    iid = (integration_id or "").strip()
    team = _team_user_ids(db, user.id)
    owned = (db.query(models.PostizIntegration)
               .filter(models.PostizIntegration.user_id.in_(team),
                       models.PostizIntegration.integration_id == iid).first())
    if not owned:
        raise HTTPException(status_code=404, detail="Not your team's integration")
    try:
        postiz_client.delete_integration(iid)
    except postiz_client.PostizAuthError as e:
        raise HTTPException(status_code=503, detail=f"Postiz auth: {e}")
    except postiz_client.PostizError as e:
        raise HTTPException(status_code=502, detail=f"Postiz delete failed: {e}")
    # Clear the binding on EVERY channel referencing this id (it's gone on
    # Postiz now, so any binding is dangling). Cross-team bindings are
    # impossible (bind is team-scoped), so this only touches the team's rows.
    bound = (db.query(models.Channel)
               .filter(models.Channel.postiz_integration_id == iid).all())
    for ch in bound:
        ch.postiz_integration_id = None
        if (ch.upload_provider or "").strip().lower() == "postiz":
            ch.upload_provider = "kaizer"
    db.delete(owned)
    db.commit()
    return {"deleted": iid, "unbound_channels": [c.id for c in bound]}
