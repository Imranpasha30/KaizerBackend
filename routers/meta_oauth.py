"""Meta (Facebook + Instagram) OAuth router.

Goes live the moment three env vars are set:

  META_APP_ID            (e.g. 1234567890)
  META_APP_SECRET        (Meta App Secret — keep out of source control)
  META_REDIRECT_URI      (https://your-domain.com/api/meta/oauth/callback)

Without those, every endpoint returns a clear "not configured"
response so the editor's UI surfaces a remediation message instead of
a generic 500.

Endpoints
─────────
GET  /api/meta/oauth/config        — feature-detect for the frontend
GET  /api/meta/oauth/start         — redirect URL to Meta's auth dialog
GET  /api/meta/oauth/callback      — receives ?code=...; exchanges
                                      it for tokens; persists a
                                      MetaAccount row per Page the
                                      operator owns + has granted
                                      permission for.
GET  /api/meta/accounts             — list the operator's MetaAccount
                                      rows (for the connected-accounts
                                      UI).
POST /api/meta/oauth/refresh/{id}  — re-exchange for a fresh
                                      long-lived Page token.
DELETE /api/meta/accounts/{id}     — disconnect a Page.

Token model
───────────
Meta gives short-lived USER tokens at OAuth time (~1 hour). Those
can be exchanged for long-lived USER tokens (~60 days). From a
long-lived user token we mint long-lived PAGE tokens — those don't
expire as long as the user token they came from is still alive AND
the user hasn't revoked the app.

So the storage strategy is: persist only the long-lived PAGE token
per Page, and refresh by going through a "minty" user-token exchange
again before the user token expires.

Permissions requested
─────────────────────
- pages_show_list           — list Pages the user manages
- pages_read_engagement     — read Page metadata
- pages_manage_posts        — post videos (REQUIRES App Review for prod)
- pages_manage_metadata     — required by Graph for the video endpoints
- business_management       — multi-account
- instagram_basic           — see the linked IG account
- instagram_content_publish — post Reels (REQUIRES App Review)

In development mode (App not yet reviewed) only the App admin's own
Pages can be posted to — but the OAuth flow + token storage works the
same. Submit for review when you're ready to publish to anyone's Page.
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


router = APIRouter(prefix="/api/meta", tags=["meta-oauth"])


META_APP_ID = (os.environ.get("META_APP_ID") or "").strip()
META_APP_SECRET = (os.environ.get("META_APP_SECRET") or "").strip()
META_REDIRECT_URI = (os.environ.get("META_REDIRECT_URI") or "").strip()
META_GRAPH_API_VERSION = os.environ.get("META_GRAPH_API_VERSION", "v21.0")

META_SCOPES = [
    "pages_show_list",
    "pages_read_engagement",
    "pages_manage_posts",
    "pages_manage_metadata",
    "business_management",
    "instagram_basic",
    "instagram_content_publish",
]

DIALOG_BASE = f"https://www.facebook.com/{META_GRAPH_API_VERSION}/dialog/oauth"
GRAPH_BASE = f"https://graph.facebook.com/{META_GRAPH_API_VERSION}"


def _is_configured() -> bool:
    return bool(META_APP_ID and META_APP_SECRET and META_REDIRECT_URI)


def _ensure_configured() -> None:
    if not _is_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "Meta OAuth is not configured. Set META_APP_ID, "
                "META_APP_SECRET, META_REDIRECT_URI in the backend "
                "environment and restart. See the publishers/ package "
                "header for the App-registration walk-through."
            ),
        )


# ── Encryption helpers — reuse the Fernet key the rest of the codebase uses ─

def _enc(s: str) -> str:
    """Encrypt a token. Lazy import keeps the OAuth module light."""
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


# ── Graph API helpers ──────────────────────────────────────────────


async def _graph_get(path: str, params: dict) -> dict:
    """Hit the Graph API GET endpoint with friendly error mapping."""
    url = f"{GRAPH_BASE}{path}"
    async with httpx.AsyncClient(timeout=30) as cx:
        r = await cx.get(url, params=params)
    if r.status_code >= 400:
        # Surface the real Meta error to the operator UI — these are
        # actionable ("missing permission", "invalid token", etc.).
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text[:500]}
        raise HTTPException(
            status_code=502,
            detail={"upstream_status": r.status_code, "meta_error": body},
        )
    return r.json()


async def _exchange_code_for_user_token(code: str) -> dict:
    """Step 1: short-lived user token from the auth code."""
    params = {
        "client_id": META_APP_ID,
        "client_secret": META_APP_SECRET,
        "redirect_uri": META_REDIRECT_URI,
        "code": code,
    }
    return await _graph_get("/oauth/access_token", params)


async def _exchange_for_long_lived_user_token(short_lived: str) -> dict:
    """Step 2: trade the short-lived user token for a ~60-day token."""
    params = {
        "grant_type": "fb_exchange_token",
        "client_id": META_APP_ID,
        "client_secret": META_APP_SECRET,
        "fb_exchange_token": short_lived,
    }
    return await _graph_get("/oauth/access_token", params)


async def _list_pages_with_tokens(long_lived_user_token: str) -> list[dict]:
    """Step 3: enumerate the user's Pages — each carries its own
    long-lived Page access token. This is the token we persist."""
    params = {
        "access_token": long_lived_user_token,
        "fields": "id,name,category,picture{url},link,access_token,"
                  "instagram_business_account{id,username,profile_picture_url,account_type}",
    }
    data = await _graph_get("/me/accounts", params)
    return data.get("data") or []


# ── State token (CSRF) management ──────────────────────────────────
# Cheap: store per-user state strings in-memory. For multi-instance
# deployments this should move to Redis; for the single-tenant
# Kaizer host this is fine.

_state_store: dict[str, int] = {}  # state_token -> user_id
_state_expiry: dict[str, datetime] = {}


def _mint_state(user_id: int) -> str:
    state = secrets.token_urlsafe(24)
    _state_store[state] = user_id
    _state_expiry[state] = datetime.now(timezone.utc) + timedelta(minutes=10)
    # GC expired states opportunistically.
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
    """Feature-detect for the frontend. Tells the UI whether the
    connect button can do anything yet."""
    return {
        "configured": _is_configured(),
        "app_id_set": bool(META_APP_ID),
        "redirect_uri_set": bool(META_REDIRECT_URI),
        "scopes": META_SCOPES,
        "graph_version": META_GRAPH_API_VERSION,
    }


@router.get("/oauth/start")
def start_oauth(
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Returns the URL the frontend should redirect to. The frontend
    opens it in a new tab — Meta then bounces back to
    META_REDIRECT_URI with ?code=...&state=..."""
    _ensure_configured()
    state = _mint_state(user.id)
    params = {
        "client_id": META_APP_ID,
        "redirect_uri": META_REDIRECT_URI,
        "scope": ",".join(META_SCOPES),
        "response_type": "code",
        "state": state,
    }
    return {"redirect_url": f"{DIALOG_BASE}?{urlencode(params)}"}


@router.get("/oauth/callback")
async def oauth_callback(
    code: str = Query(...),
    state: str = Query(...),
    db: Session = Depends(get_db),
):
    """Meta hits this with ?code=...&state=... after the operator
    approves the permissions. We complete the token exchange + persist
    one MetaAccount row per Page they granted us, then redirect to
    /settings/meta with a success flag for the UI."""
    _ensure_configured()

    user_id = _consume_state(state)
    if not user_id:
        raise HTTPException(400, "Invalid or expired OAuth state")

    # Verify the user still exists (rare race but possible).
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(400, "User no longer exists")

    short = await _exchange_code_for_user_token(code)
    short_token = short.get("access_token", "")
    if not short_token:
        raise HTTPException(502, f"Meta did not return access_token: {short}")

    long = await _exchange_for_long_lived_user_token(short_token)
    long_user_token = long.get("access_token", "")
    expires_in = int(long.get("expires_in", 0))   # seconds
    if not long_user_token:
        raise HTTPException(502, f"Meta did not return long-lived token: {long}")

    pages = await _list_pages_with_tokens(long_user_token)
    persisted: list[dict] = []
    for p in pages:
        page_id = p.get("id") or ""
        page_token = p.get("access_token") or ""
        if not page_id or not page_token:
            continue
        ig = p.get("instagram_business_account") or {}
        # One row per Page — upsert by (user_id, fb_page_id).
        existing = (
            db.query(models.MetaAccount)
              .filter(
                  models.MetaAccount.user_id == user.id,
                  models.MetaAccount.fb_page_id == page_id,
              )
              .first()
        )
        row = existing or models.MetaAccount(user_id=user.id, fb_page_id=page_id)
        row.fb_user_id = (long.get("user_id") or short.get("user_id") or "")[:50]
        row.fb_page_name = (p.get("name") or "")[:255]
        row.fb_page_category = (p.get("category") or "")[:120]
        row.fb_page_picture_url = ((p.get("picture") or {}).get("data") or {}).get("url", "")[:500]
        row.fb_page_url = (p.get("link") or "")[:500]

        row.ig_user_id = (ig.get("id") or "")[:50]
        row.ig_username = (ig.get("username") or "")[:120]
        row.ig_profile_pic_url = (ig.get("profile_picture_url") or "")[:500]
        row.ig_account_type = (ig.get("account_type") or "")[:40]

        row.page_access_token_enc = _enc(page_token)
        # Page tokens are documented as "never expire" but Meta has
        # been known to invalidate them on user changes. We track an
        # expected expiry equal to the long-lived user token's expiry
        # as a conservative lower bound + nudge for the refresh worker.
        row.page_token_expiry = (
            datetime.now(timezone.utc) + timedelta(seconds=expires_in)
            if expires_in > 0 else None
        )
        row.granted_scopes = ",".join(META_SCOPES)
        row.last_refreshed_at = datetime.now(timezone.utc)
        if not existing:
            db.add(row)
        persisted.append({
            "page_id": row.fb_page_id,
            "page_name": row.fb_page_name,
            "has_ig": bool(row.ig_user_id),
        })

    db.commit()

    # Redirect to the frontend's settings page with a query flag so
    # the UI can show a success toast + refresh the accounts list.
    return RedirectResponse(
        url=f"/settings/meta?connected={len(persisted)}",
        status_code=303,
    )


@router.get("/accounts")
def list_accounts(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> list[dict]:
    rows = (
        db.query(models.MetaAccount)
          .filter(models.MetaAccount.user_id == user.id)
          .order_by(models.MetaAccount.connected_at.asc())
          .all()
    )
    return [
        {
            "id": r.id,
            "fb_page_id": r.fb_page_id,
            "fb_page_name": r.fb_page_name,
            "fb_page_picture_url": r.fb_page_picture_url,
            "fb_page_url": r.fb_page_url,
            "fb_page_category": r.fb_page_category,
            "ig_user_id": r.ig_user_id,
            "ig_username": r.ig_username,
            "ig_profile_pic_url": r.ig_profile_pic_url,
            "ig_account_type": r.ig_account_type,
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
                r.page_token_expiry.isoformat() if r.page_token_expiry else None
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
        db.query(models.MetaAccount)
          .filter(models.MetaAccount.id == account_id,
                  models.MetaAccount.user_id == user.id)
          .first()
    )
    if not row:
        raise HTTPException(404, "Meta account not found")
    db.delete(row)
    db.commit()
    return {"ok": True, "deleted": account_id}


def page_access_token_for(db: Session, account_id: int) -> Optional[str]:
    """Helper for the Meta publishers — decrypts and returns the
    current Page access token. Publishers call this instead of
    touching the DB row directly so future token-refresh logic lives
    in one place."""
    row = db.query(models.MetaAccount).filter(
        models.MetaAccount.id == account_id
    ).first()
    if not row:
        return None
    return _dec(row.page_access_token_enc)
