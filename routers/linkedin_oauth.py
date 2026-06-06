"""LinkedIn OAuth router.

Mirrors routers/meta_oauth.py — same shape, different platform.

Goes live when these env vars are set:
  LINKEDIN_CLIENT_ID
  LINKEDIN_CLIENT_SECRET
  LINKEDIN_REDIRECT_URI

The redirect URI MUST also be added to the LinkedIn App's
"Authorized redirect URLs" allowlist at
https://www.linkedin.com/developers/apps/<app-id>/auth.

Endpoints
─────────
GET  /api/linkedin/oauth/config        — feature-detect for the frontend
GET  /api/linkedin/oauth/start         — redirect URL to LinkedIn's auth dialog
GET  /api/linkedin/oauth/callback      — receives ?code=&state=; exchanges
                                          + fetches profile + upserts a
                                          LinkedInAccount row.
GET  /api/linkedin/accounts             — list operator's connected
                                          LinkedIn profiles + pages.
DELETE /api/linkedin/accounts/{id}      — disconnect.

Token model
───────────
LinkedIn gives both an access_token (~60 days) AND a refresh_token
(~365 days) on initial OAuth. The refresh_token is exchangeable for a
fresh access_token any time — simpler than Meta's flow. We store
BOTH encrypted; a background worker (TODO) renews access_tokens
before expiry without operator interaction.

Scopes requested
────────────────
- openid                  identity assertion
- profile                 basic profile metadata (name, picture)
- email                   verified email (used as the account label)
- w_member_social         post to personal profile
- w_organization_social   post to Company Pages (Marketing Developer
                          Platform — requires approval, but harmless
                          to request)
- r_organization_admin    discover which Pages the operator can post to
"""
from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

import auth
import models
from database import get_db


router = APIRouter(prefix="/api/linkedin", tags=["linkedin-oauth"])

LINKEDIN_CLIENT_ID = (os.environ.get("LINKEDIN_CLIENT_ID") or "").strip()
LINKEDIN_CLIENT_SECRET = (os.environ.get("LINKEDIN_CLIENT_SECRET") or "").strip()
LINKEDIN_REDIRECT_URI = (os.environ.get("LINKEDIN_REDIRECT_URI") or "").strip()

LINKEDIN_SCOPES = [
    "openid",
    "profile",
    "email",
    "w_member_social",
    "w_organization_social",
    "r_organization_admin",
]

AUTH_BASE = "https://www.linkedin.com/oauth/v2/authorization"
TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"
API_BASE = "https://api.linkedin.com/v2"
USERINFO_URL = "https://api.linkedin.com/v2/userinfo"   # OpenID Connect endpoint


def _is_configured() -> bool:
    return bool(LINKEDIN_CLIENT_ID and LINKEDIN_CLIENT_SECRET and LINKEDIN_REDIRECT_URI)


def _ensure_configured() -> None:
    if not _is_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "LinkedIn OAuth is not configured. Set LINKEDIN_CLIENT_ID, "
                "LINKEDIN_CLIENT_SECRET, LINKEDIN_REDIRECT_URI in the backend "
                "environment and restart."
            ),
        )


def _enc(s: str) -> str:
    if not s:
        return ""
    from auth import _fernet
    return _fernet().encrypt(s.encode("utf-8")).decode("ascii")


def _dec(s: str) -> str:
    if not s:
        return ""
    from auth import _fernet
    try:
        return _fernet().decrypt(s.encode("ascii")).decode("utf-8")
    except Exception:
        return ""


# Same in-memory state store pattern as Meta OAuth. Single-tenant
# Kaizer host; multi-instance deployments should swap to Redis.

_state_store: dict[str, int] = {}
_state_expiry: dict[str, datetime] = {}


def _mint_state(user_id: int) -> str:
    state = secrets.token_urlsafe(24)
    _state_store[state] = user_id
    _state_expiry[state] = datetime.now(timezone.utc) + timedelta(minutes=10)
    now = datetime.now(timezone.utc)
    for k in [k for k, v in _state_expiry.items() if v < now]:
        _state_store.pop(k, None)
        _state_expiry.pop(k, None)
    return state


def _consume_state(state: str) -> Optional[int]:
    exp = _state_expiry.pop(state, None)
    user_id = _state_store.pop(state, None)
    if not user_id or not exp or exp < datetime.now(timezone.utc):
        return None
    return user_id


# ── Endpoints ──────────────────────────────────────────────────────


@router.get("/oauth/config")
def get_config(
    user: models.User = Depends(auth.current_user),
) -> dict:
    return {
        "configured": _is_configured(),
        "client_id_set": bool(LINKEDIN_CLIENT_ID),
        "redirect_uri_set": bool(LINKEDIN_REDIRECT_URI),
        "scopes": LINKEDIN_SCOPES,
    }


@router.get("/oauth/start")
def start_oauth(
    user: models.User = Depends(auth.current_user),
) -> dict:
    _ensure_configured()
    state = _mint_state(user.id)
    params = {
        "response_type": "code",
        "client_id": LINKEDIN_CLIENT_ID,
        "redirect_uri": LINKEDIN_REDIRECT_URI,
        "scope": " ".join(LINKEDIN_SCOPES),
        "state": state,
    }
    return {"redirect_url": f"{AUTH_BASE}?{urlencode(params)}"}


@router.get("/oauth/callback")
async def oauth_callback(
    code: str = Query(...),
    state: str = Query(...),
    db: Session = Depends(get_db),
):
    _ensure_configured()
    user_id = _consume_state(state)
    if not user_id:
        raise HTTPException(400, "Invalid or expired OAuth state")
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(400, "User no longer exists")

    # Step 1: exchange code for tokens. LinkedIn uses form-encoded
    # POST body, NOT querystring like Meta — easy to get wrong.
    async with httpx.AsyncClient(timeout=30) as cx:
        tok = await cx.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "client_id": LINKEDIN_CLIENT_ID,
                "client_secret": LINKEDIN_CLIENT_SECRET,
                "redirect_uri": LINKEDIN_REDIRECT_URI,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if tok.status_code >= 400:
        try:
            body = tok.json()
        except Exception:
            body = {"raw": tok.text[:300]}
        raise HTTPException(502, f"LinkedIn token exchange failed: {body}")
    tdata = tok.json()
    access_token = tdata.get("access_token") or ""
    refresh_token = tdata.get("refresh_token") or ""
    expires_in = int(tdata.get("expires_in", 0))
    if not access_token:
        raise HTTPException(502, f"LinkedIn did not return access_token: {tdata}")

    # Step 2: fetch the user's profile so we know who we just
    # connected. The OpenID Connect userinfo endpoint gives us a
    # stable sub (LinkedIn id) + name + email + picture in one call.
    async with httpx.AsyncClient(timeout=30) as cx:
        ui = await cx.get(
            USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
    if ui.status_code >= 400:
        # Some App configurations don't grant OpenID — fall back to
        # the legacy /me endpoint.
        async with httpx.AsyncClient(timeout=30) as cx:
            ui = await cx.get(
                f"{API_BASE}/me",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "LinkedIn-Version": "202405",
                    "X-Restli-Protocol-Version": "2.0.0",
                },
            )
    profile = ui.json() if ui.status_code < 400 else {}
    li_id = profile.get("sub") or profile.get("id") or ""
    name = (profile.get("name")
            or " ".join(filter(None, [
                profile.get("given_name") or profile.get("localizedFirstName"),
                profile.get("family_name") or profile.get("localizedLastName"),
            ])))
    pic = (profile.get("picture") or "") or ""

    if not li_id:
        raise HTTPException(502, f"Could not resolve LinkedIn id from profile: {profile}")

    urn = f"urn:li:person:{li_id}"

    # Upsert. One row per (user_id, linkedin_id).
    existing = (
        db.query(models.LinkedInAccount)
          .filter(models.LinkedInAccount.user_id == user.id,
                  models.LinkedInAccount.linkedin_id == li_id)
          .first()
    )
    row = existing or models.LinkedInAccount(user_id=user.id, linkedin_id=li_id)
    row.linkedin_urn = urn
    row.profile_name = (name or "")[:255]
    row.profile_pic_url = (pic or "")[:500]
    row.profile_url = f"https://www.linkedin.com/in/{li_id}"
    row.account_type = "person"
    row.access_token_enc = _enc(access_token)
    if refresh_token:
        row.refresh_token_enc = _enc(refresh_token)
    row.token_expiry = (
        datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        if expires_in > 0 else None
    )
    row.granted_scopes = ",".join(LINKEDIN_SCOPES)
    row.last_refreshed_at = datetime.now(timezone.utc)
    if not existing:
        db.add(row)
    db.commit()

    return RedirectResponse(
        url=f"/settings/social?connected=linkedin",
        status_code=303,
    )


@router.get("/accounts")
def list_accounts(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> list[dict]:
    rows = (
        db.query(models.LinkedInAccount)
          .filter(models.LinkedInAccount.user_id == user.id)
          .order_by(models.LinkedInAccount.connected_at.asc())
          .all()
    )
    return [
        {
            "id": r.id,
            "linkedin_id": r.linkedin_id,
            "linkedin_urn": r.linkedin_urn,
            "profile_name": r.profile_name,
            "profile_pic_url": r.profile_pic_url,
            "profile_url": r.profile_url,
            "account_type": r.account_type,
            "granted_scopes": (r.granted_scopes or "").split(",") if r.granted_scopes else [],
            "connected_at": r.connected_at.isoformat() if r.connected_at else None,
            "last_refreshed_at": (
                r.last_refreshed_at.isoformat() if r.last_refreshed_at else None
            ),
            "last_publish_at": (
                r.last_publish_at.isoformat() if r.last_publish_at else None
            ),
            "publishes_today": r.publishes_today or 0,
            "token_expiry": (
                r.token_expiry.isoformat() if r.token_expiry else None
            ),
        }
        for r in rows
    ]


@router.delete("/accounts/{account_id}")
def disconnect(
    account_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    row = (
        db.query(models.LinkedInAccount)
          .filter(models.LinkedInAccount.id == account_id,
                  models.LinkedInAccount.user_id == user.id)
          .first()
    )
    if not row:
        raise HTTPException(404, "LinkedIn account not found")
    db.delete(row)
    db.commit()
    return {"ok": True, "deleted": account_id}


def access_token_for(db: Session, account_id: int) -> Optional[str]:
    """Decrypt + return the current access token. Used by the
    LinkedIn publisher to make API calls."""
    row = db.query(models.LinkedInAccount).filter(
        models.LinkedInAccount.id == account_id
    ).first()
    if not row:
        return None
    return _dec(row.access_token_enc)
