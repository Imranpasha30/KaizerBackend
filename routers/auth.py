"""Auth router — register / login / me / logout + Google Sign-In.

Google Sign-In here is distinct from YouTube OAuth.  YouTube OAuth grants
the app permission to upload to a user's channel; this flow just verifies
"who is signing into the app".  It uses the same `YOUTUBE_CLIENT_ID` because
Google's `sign-in with Google` works with any OAuth 2.0 web client.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
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
        "is_admin": bool(u.is_admin),
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
def login(payload: LoginIn, request: Request, db: Session = Depends(get_db)):
    # Per-IP brute-force guard. We rate-limit BEFORE checking creds so
    # an attacker can't probe valid emails by timing 401s, and so a
    # password-spray scaled across thousands of accounts still trips
    # the same per-IP bucket.
    from rate_limit import check_ip_rate as _check_ip_rate
    xff = request.headers.get("x-forwarded-for", "")
    ip  = (xff.split(",")[0].strip() if xff else
           (request.client.host if request.client else "unknown"))
    allowed, retry_after, _remaining = _check_ip_rate(ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Too many login attempts. Retry in {int(retry_after)}s.",
            headers={"Retry-After": str(max(1, int(retry_after)))},
        )

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
        # ``clock_skew_in_seconds`` absorbs the small drift between
        # Google's token-issue clock and the local server clock. Without
        # this tolerance a 1-second drift trips "Token used too early"
        # on the user's first sign-in (seen on Windows boxes that haven't
        # NTP-synced recently). 10 s is the canonical google-auth-library
        # default — large enough to forgive normal drift, small enough
        # that replay-window attacks are still bounded.
        info = _idt.verify_oauth2_token(
            payload.credential, _grequests.Request(), client_id,
            clock_skew_in_seconds=10,
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


# ─── Password management ──────────────────────────────────────────────────
#
# Three flows:
#   1. /me/has-password   — frontend asks "should I show Set or Change?"
#   2. /me/password       — authenticated change/set (Google users hit this
#                           with current_password=null to set their first pw)
#   3. /forgot + /reset   — unauthenticated reset via single-use token


class ChangePasswordIn(BaseModel):
    current_password: Optional[str] = Field(None, max_length=200,
        description="Required only if the account already has a password set.")
    new_password: str = Field(..., min_length=8, max_length=200)

    @field_validator("new_password")
    @classmethod
    def _strength(cls, v: str) -> str:
        # Minimal-but-meaningful: 8+ chars + at least one digit OR symbol.
        # Anything stricter just irritates users; password resets are cheap.
        if not any(c.isdigit() or not c.isalnum() for c in v):
            raise ValueError("Add at least one digit or symbol to the new password.")
        return v


class ForgotPasswordIn(BaseModel):
    email: EmailStr


class ResetPasswordIn(BaseModel):
    token:        str = Field(..., min_length=10, max_length=200)
    new_password: str = Field(..., min_length=8, max_length=200)

    @field_validator("new_password")
    @classmethod
    def _strength(cls, v: str) -> str:
        if not any(c.isdigit() or not c.isalnum() for c in v):
            raise ValueError("Add at least one digit or symbol to the new password.")
        return v


@router.get("/me/has-password")
def has_password(user: models.User = Depends(_auth.current_user)):
    """Frontend uses this to label the Settings section.

    Returns ``{has_password, signin_methods}`` so the UI can show "Set a
    password" for Google-only accounts and "Change password" for the rest.
    """
    methods = []
    if user.password_hash: methods.append("password")
    if user.google_sub:    methods.append("google")
    return {
        "has_password":   bool(user.password_hash),
        "signin_methods": methods,
        "google_linked":  bool(user.google_sub),
    }


@router.post("/me/password")
def change_password(
    payload: ChangePasswordIn,
    db: Session = Depends(get_db),
    user: models.User = Depends(_auth.current_user),
):
    """Set or change the password for the currently signed-in user.

    If the account already has a password, ``current_password`` must match.
    Google-only accounts (no password yet) can omit it — this is the
    "Set a password" path that lets a Google user sign in via email later.
    """
    if user.password_hash:
        if not payload.current_password or not _auth.verify_password(
                payload.current_password, user.password_hash):
            raise HTTPException(status_code=400, detail="Current password is incorrect.")

    _auth.set_password(db, user, payload.new_password)
    return {"ok": True, "message": "Password updated."}


@router.post("/forgot-password")
def forgot_password(
    payload: ForgotPasswordIn,
    request: Request,
    db: Session = Depends(get_db),
):
    """Issue a one-shot reset token and email/log it.

    Always returns 200 with the same shape regardless of whether the email
    exists — that prevents email-enumeration. The actual delivery happens
    out-of-band (SMTP if configured, else logged to admin Logs tab).
    """
    email = payload.email.lower().strip()
    # Lazy cleanup of week-old tokens — keeps the table small.
    try:
        _auth.purge_expired_reset_tokens(db)
    except Exception:
        pass

    user = db.query(models.User).filter(models.User.email == email).first()

    # Always behave identically from the caller's POV (anti-enumeration).
    if user and user.is_active:
        # Figure out the public origin to build the reset URL.
        # Prefer the Origin header (real browser); fall back to env.
        import os as _os
        origin = (request.headers.get("origin")
                  or request.headers.get("referer", "").rstrip("/")
                  or _os.getenv("KAIZER_PUBLIC_ORIGIN", "")
                  or "https://test.kaizerx.com")
        # Strip any path from the referer
        if "://" in origin:
            scheme, rest = origin.split("://", 1)
            host = rest.split("/", 1)[0]
            origin = f"{scheme}://{host}"
        raw = _auth.make_reset_token(
            db, user,
            requested_ip=(request.client.host if request.client else "")[:64],
        )
        reset_url = f"{origin}/reset-password?token={raw}"
        _auth.send_reset_email(
            to_email=user.email,
            reset_url=reset_url,
            user_name=user.name or "",
        )

    return {
        "ok": True,
        "message": ("If an account with that email exists, a password-reset "
                    "link has been sent. The link expires in "
                    f"{_auth.RESET_TOKEN_TTL_MIN} minutes."),
    }


@router.get("/reset-password/validate")
def validate_reset(token: str):
    """Cheap pre-check from the Reset page so we can tell the user
    'this link is expired / already used' before they type a new password."""
    db = next(get_db())
    try:
        row = _auth.lookup_valid_reset(db, token)
        if not row:
            return {"valid": False, "reason": "expired_or_invalid"}
        user = db.query(models.User).filter(models.User.id == row.user_id).first()
        return {
            "valid": True,
            "email": user.email if user else None,
            "expires_at": row.expires_at.isoformat() if row.expires_at else None,
        }
    finally:
        db.close()


@router.post("/reset-password")
def reset_password(payload: ResetPasswordIn, db: Session = Depends(get_db)):
    """Consume a reset token and set a new password. Returns a fresh JWT so
    the user is logged in immediately after a successful reset."""
    row = _auth.lookup_valid_reset(db, payload.token)
    if not row:
        raise HTTPException(
            status_code=400,
            detail="This reset link is invalid or has expired. Please request a new one.",
        )
    user = _auth.consume_reset_token(db, row, payload.new_password)
    user.last_login_at = datetime.now(timezone.utc)
    db.commit()
    return _with_token(user)
