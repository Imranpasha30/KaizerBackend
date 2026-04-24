"""Password hashing + JWT helpers + FastAPI `current_user` dependency.

All routes that need the signed-in user should declare `user = Depends(current_user)`.
Legacy (unauthenticated) requests fall back to a single shared "legacy" user so
the old single-tenant data stays accessible while multi-user ramps up.
"""
from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
import bcrypt
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from database import SessionLocal, get_db
import models


# ─── Config ───────────────────────────────────────────────────────────────

JWT_ALGO       = "HS256"
JWT_EXPIRES_HOURS = 24 * 7  # 7 days
LEGACY_USER_EMAIL = "legacy@kaizer.local"


def _jwt_secret() -> str:
    """Persisted in env; generated on first use if missing.

    Persisting in the running process is OK — if the backend restarts,
    tokens signed with a regenerated secret will become invalid and
    clients simply re-login.  For production set KAIZER_JWT_SECRET in .env.
    """
    sec = os.getenv("KAIZER_JWT_SECRET")
    if sec:
        return sec
    # Dev fallback: stable per-process random secret
    global _runtime_secret
    if not globals().get("_runtime_secret"):
        globals()["_runtime_secret"] = secrets.token_urlsafe(48)
    return globals()["_runtime_secret"]


# ─── Password ─────────────────────────────────────────────────────────────
# bcrypt caps input at 72 bytes.  We pre-truncate (a standard workaround —
# widely used; the resulting hash is still secure because the input space
# is already far larger than the keyspace).

def _pw_bytes(raw: str) -> bytes:
    return (raw or "").encode("utf-8")[:72]


def hash_password(raw: str) -> str:
    return bcrypt.hashpw(_pw_bytes(raw), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(raw: str, hashed: str) -> bool:
    if not raw or not hashed:
        return False
    try:
        return bcrypt.checkpw(_pw_bytes(raw), hashed.encode("utf-8"))
    except Exception:
        return False


# ─── JWT ──────────────────────────────────────────────────────────────────

def issue_token(user: models.User) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub":   str(user.id),
        "email": user.email,
        "iat":   int(now.timestamp()),
        "exp":   int((now + timedelta(hours=JWT_EXPIRES_HOURS)).timestamp()),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=JWT_ALGO)


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, _jwt_secret(), algorithms=[JWT_ALGO])
    except jwt.PyJWTError:
        return None


# ─── Legacy user (used for unauthenticated requests + data migration) ────

def ensure_legacy_user(db: Session) -> models.User:
    """Idempotent — returns the shared legacy user, creating it if absent."""
    u = db.query(models.User).filter(models.User.email == LEGACY_USER_EMAIL).first()
    if u:
        return u
    u = models.User(
        email=LEGACY_USER_EMAIL,
        name="Legacy User",
        password_hash=None,
        is_active=True,
        is_admin=True,
    )
    db.add(u); db.commit(); db.refresh(u)
    return u


# ─── Dependencies ────────────────────────────────────────────────────────

def current_user(
    authorization: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
) -> models.User:
    """Returns the signed-in User row.

    - If the request carries `Authorization: Bearer <jwt>` and it decodes
      successfully, returns that user.
    - Otherwise returns the shared legacy user.  This keeps the old
      unauthenticated frontend working during the rollout.  Flip on
      KAIZER_AUTH_REQUIRED=true to enforce login everywhere.
    """
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()

    if token:
        payload = decode_token(token)
        if payload:
            user_id = int(payload.get("sub") or 0)
            if user_id:
                u = db.query(models.User).filter(models.User.id == user_id).first()
                if u and u.is_active:
                    return u
        # Token present but invalid — respond clearly so the frontend can log out
        if os.getenv("KAIZER_AUTH_REQUIRED", "false").lower() in ("1", "true", "yes", "on"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token. Please sign in again.",
            )

    # No token (or invalid while auth is optional) → legacy user fallback
    if os.getenv("KAIZER_AUTH_REQUIRED", "false").lower() in ("1", "true", "yes", "on"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in.",
        )
    return ensure_legacy_user(db)


def current_user_optional(
    authorization: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
) -> Optional[models.User]:
    """Same as current_user but returns None instead of a legacy fallback —
    used by endpoints that behave differently when signed in vs anonymous.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    payload = decode_token(token)
    if not payload:
        return None
    user_id = int(payload.get("sub") or 0)
    if not user_id:
        return None
    u = db.query(models.User).filter(models.User.id == user_id).first()
    return u if (u and u.is_active) else None


async def admin_required(user: "models.User" = Depends(current_user)) -> "models.User":
    """401/403 gate for admin-only endpoints.

    Denies requests from disabled accounts AND from non-admins.  Callers:

        @router.get("/admin/…")
        def ep(user: models.User = Depends(admin_required)):
            ...

    NOTE: Runs *after* `current_user`, so in environments where auth is
    optional a legacy user is returned — that legacy user is flagged
    `is_admin=True` by `ensure_legacy_user()`, so dev still works.
    """
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ─── Field-level encryption helpers ──────────────────────────────────────
# Thin re-exports over `crypto.py`'s Fernet wrapper, kept here so admin
# routes can mask/decrypt sensitive fields without importing crypto directly.
# TODO(Phase-12.1): migrate OAuth access/refresh tokens stored in plaintext
# in any legacy rows to the encrypted form; today they're already written
# to `*_enc` columns via crypto.encrypt at write-time.

def encrypt(plaintext: str) -> str:
    """Fernet-encrypt; empty input → empty output (safe for NULLable cols)."""
    from crypto import encrypt as _enc
    return _enc(plaintext or "")


def decrypt(ciphertext: str) -> str:
    """Reverse of encrypt().  Returns "" for empty input.  Raises CryptoError
    on malformed/tampered payloads — admin endpoints should catch and mask
    to "****" on failure (never leak the raw ciphertext)."""
    from crypto import decrypt as _dec
    return _dec(ciphertext or "")
