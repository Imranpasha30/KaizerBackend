"""Auth router — register / login / me / logout + Google Sign-In.

Google Sign-In here is distinct from YouTube OAuth.  YouTube OAuth grants
the app permission to upload to a user's channel; this flow just verifies
"who is signing into the app".  It uses the same `YOUTUBE_CLIENT_ID` because
Google's `sign-in with Google` works with any OAuth 2.0 web client.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy.orm import Session

import auth as _auth
import models
from config import settings
from database import get_db


router = APIRouter(prefix="/api/auth", tags=["auth"])


# ─── Schemas ──────────────────────────────────────────────────────────────

class RegisterIn(BaseModel):
    email:    EmailStr
    password: str = Field(..., min_length=6, max_length=200)
    name:     str = ""


class LoginIn(BaseModel):
    email:    EmailStr
    password: str = Field(..., min_length=1, max_length=200)


class GoogleIn(BaseModel):
    credential: str  # the ID token from Google Identity Services


def _public_user(u: models.User) -> dict:
    raw_socials = getattr(u, "socials", None) or {}
    return {
        "id":     u.id,
        "email":  u.email,
        "name":   u.name or u.email.split("@")[0],
        "google": bool(u.google_sub),
        "admin":  bool(u.is_admin),
        "socials": raw_socials if isinstance(raw_socials, dict) else {},
        "created_at":    u.created_at.isoformat() if u.created_at else None,
        "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
    }


def _with_token(u: models.User) -> dict:
    return {"token": _auth.issue_token(u), "user": _public_user(u)}


# ─── Endpoints ────────────────────────────────────────────────────────────

@router.post("/register")
def register(payload: RegisterIn, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()
    if db.query(models.User).filter(models.User.email == email).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists. Try logging in.",
        )
    u = models.User(
        email=email,
        name=(payload.name or "").strip(),
        password_hash=_auth.hash_password(payload.password),
        is_active=True,
    )
    db.add(u); db.commit(); db.refresh(u)
    u.last_login_at = datetime.now(timezone.utc)
    db.commit()
    return _with_token(u)


@router.post("/login")
def login(payload: LoginIn, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()
    u = db.query(models.User).filter(models.User.email == email).first()
    if not u or not u.password_hash or not _auth.verify_password(payload.password, u.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Wrong email or password.",
        )
    if not u.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled.")
    u.last_login_at = datetime.now(timezone.utc)
    db.commit()
    return _with_token(u)


@router.post("/google")
def google_signin(payload: GoogleIn, db: Session = Depends(get_db)):
    """Verify a Google ID token and log the user in (creating an account if new).

    Relies on `YOUTUBE_CLIENT_ID` being the OAuth client whose ID token this
    came from — same client is used for Sign-In With Google + YouTube OAuth.
    """
    client_id = settings.yt_client_id
    if not client_id:
        raise HTTPException(status_code=500, detail="Google Sign-In is not configured.")

    try:
        from google.oauth2 import id_token as _idt
        from google.auth.transport import requests as _grequests
        info = _idt.verify_oauth2_token(
            payload.credential, _grequests.Request(), client_id
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid Google credential: {e}")

    sub   = info.get("sub")
    email = (info.get("email") or "").lower().strip()
    name  = info.get("name") or ""
    if not sub or not email:
        raise HTTPException(status_code=401, detail="Google credential missing required fields.")
    if not info.get("email_verified", False):
        raise HTTPException(status_code=401, detail="Please verify your email with Google before signing in.")

    # Match on google_sub first, fall back to email for linking existing accounts
    u = db.query(models.User).filter(models.User.google_sub == sub).first()
    if not u:
        u = db.query(models.User).filter(models.User.email == email).first()
        if u:
            u.google_sub = sub  # Link existing email-account to this Google identity
        else:
            u = models.User(
                email=email, name=name,
                google_sub=sub, password_hash=None,
                is_active=True,
            )
            db.add(u)
    if not u.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled.")
    u.last_login_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(u)
    return _with_token(u)


@router.get("/me")
def me(user: models.User = Depends(_auth.current_user)):
    return _public_user(user)


# ─── Socials (cross-promo links used by SEO footer) ────────────────────

# Canonical platform keys — free-form values, empty string = remove.
SOCIAL_KEYS = (
    "youtube", "website", "twitter", "instagram", "facebook",
    "whatsapp", "telegram", "linkedin", "tiktok", "threads", "email",
)


class SocialsIn(BaseModel):
    socials: dict  # { "twitter": "@...", "instagram": "https://...", ... }


@router.get("/me/socials")
def get_socials(
    user: models.User = Depends(_auth.current_user),
):
    raw = getattr(user, "socials", None) or {}
    return raw if isinstance(raw, dict) else {}


@router.put("/me/socials")
def put_socials(
    payload: SocialsIn,
    db: Session = Depends(get_db),
    user: models.User = Depends(_auth.current_user),
):
    """Replace the user's social-links map. Unknown keys are kept (future-proof);
    empty-string values are dropped."""
    cleaned = {}
    src = payload.socials or {}
    if not isinstance(src, dict):
        raise HTTPException(status_code=422, detail="socials must be an object")
    for k, v in src.items():
        if not k or not isinstance(k, str):
            continue
        v = (v or "").strip() if isinstance(v, str) else ""
        if v:
            cleaned[k.strip().lower()] = v[:500]
    user.socials = cleaned
    db.commit()
    db.refresh(user)
    return cleaned


@router.post("/logout")
def logout():
    """Stateless — the frontend just forgets the JWT. Here for symmetry."""
    return {"ok": True}


@router.get("/config")
def auth_config():
    """Exposes whether Google Sign-In is available + the client id."""
    return {
        "google_enabled":    bool(settings.yt_client_id),
        "google_client_id":  settings.yt_client_id or "",
        "auth_required":     (
            __import__("os").getenv("KAIZER_AUTH_REQUIRED", "false").lower()
            in ("1", "true", "yes", "on")
        ),
    }
