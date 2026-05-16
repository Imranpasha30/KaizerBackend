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


# ─── Password reset (forgot-password flow) ────────────────────────────────
#
# Two write paths into ``users.password_hash``:
#  - ``set_password`` for a *logged-in* user who knows their old password (or
#    is a Google-only account setting one for the first time).
#  - ``consume_reset_token`` for the unauthenticated forgot-password flow:
#    the user proves possession of a single-use token mailed/logged earlier.
#
# Token bytes are 32 random URL-safe chars; only the SHA-256 of the token is
# persisted, so a leaked DB doesn't grant impersonation. Tokens live 30 min.

import hashlib

RESET_TOKEN_TTL_MIN = int(os.getenv("KAIZER_RESET_TOKEN_TTL_MIN", "30"))


def _hash_token(token: str) -> str:
    return hashlib.sha256((token or "").encode("utf-8")).hexdigest()


def set_password(db: Session, user: "models.User", new_password: str) -> None:
    """Hash + store a new password. Caller has already authenticated the user
    (either by current password or by a valid reset token)."""
    user.password_hash = hash_password(new_password)
    db.commit()


def make_reset_token(db: Session, user: "models.User", *, requested_ip: str = "") -> str:
    """Issue a one-shot reset token. Returns the raw token (only shown once)."""
    raw = secrets.token_urlsafe(32)
    row = models.PasswordResetToken(
        user_id=user.id,
        token_hash=_hash_token(raw),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=RESET_TOKEN_TTL_MIN),
        requested_ip=(requested_ip or "")[:64],
    )
    db.add(row)
    db.commit()
    return raw


def lookup_valid_reset(db: Session, raw_token: str) -> Optional["models.PasswordResetToken"]:
    """Return the row IFF the token exists, hasn't been used, and hasn't
    expired. Returns None in all the failure cases — never raises."""
    if not raw_token:
        return None
    row = (db.query(models.PasswordResetToken)
             .filter(models.PasswordResetToken.token_hash == _hash_token(raw_token))
             .first())
    if not row:
        return None
    if row.used_at is not None:
        return None
    expires = row.expires_at
    if expires is None:
        return None
    # Normalise to UTC for comparison (timezone-aware DBs return aware; SQLite naive)
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) > expires:
        return None
    return row


def consume_reset_token(db: Session, row: "models.PasswordResetToken",
                        new_password: str) -> "models.User":
    """Apply a verified token: set the new password, mark the token used."""
    user = db.query(models.User).filter(models.User.id == row.user_id).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=400, detail="Account no longer exists")
    user.password_hash = hash_password(new_password)
    row.used_at = datetime.now(timezone.utc)
    db.commit()
    return user


def purge_expired_reset_tokens(db: Session) -> int:
    """Best-effort cleanup of stale rows. Called opportunistically on each
    /forgot request. Returns the number deleted."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    n = (db.query(models.PasswordResetToken)
           .filter(models.PasswordResetToken.created_at < cutoff)
           .delete(synchronize_session=False))
    db.commit()
    return n


# ─── Email delivery (best-effort) ─────────────────────────────────────────
# SMTP is OPTIONAL. When ``KAIZER_SMTP_HOST`` is set the reset link is
# emailed; otherwise the link is just logged so the operator can grab it
# from the admin Logs tab during pre-launch.

def _smtp_configured() -> bool:
    return bool(os.getenv("KAIZER_SMTP_HOST"))


def send_reset_email(*, to_email: str, reset_url: str, user_name: str = "") -> bool:
    """Send a password-reset email via SMTP. Always logs the URL to stdout
    (which the admin Logs tab captures), so dev/pre-launch works without
    SMTP credentials. Returns True if an actual email was sent."""
    # ALWAYS log — admin Logs panel uses this as the dev fallback.
    print(f"[auth.reset] password-reset link for {to_email}: {reset_url}")

    if not _smtp_configured():
        return False

    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    try:
        host = os.environ["KAIZER_SMTP_HOST"]
        port = int(os.environ.get("KAIZER_SMTP_PORT", "587"))
        user = os.environ.get("KAIZER_SMTP_USER", "")
        pwd  = os.environ.get("KAIZER_SMTP_PASS", "")
        sender = os.environ.get("KAIZER_SMTP_FROM", user or "noreply@kaizer.local")
        use_tls = os.environ.get("KAIZER_SMTP_TLS", "true").lower() in ("1", "true", "yes")

        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Reset your Kaizer password"
        msg["From"]    = sender
        msg["To"]      = to_email

        text = (
            f"Hi{(' ' + user_name) if user_name else ''},\n\n"
            f"Someone (hopefully you) asked to reset your Kaizer password. "
            f"Open this link within {RESET_TOKEN_TTL_MIN} minutes to choose a new one:\n\n"
            f"{reset_url}\n\n"
            f"If you didn't ask for this, just ignore this email — your password "
            f"won't change.\n"
        )
        html = (
            f"<p>Hi{(' ' + user_name) if user_name else ''},</p>"
            f"<p>Someone (hopefully you) asked to reset your Kaizer password. "
            f"Open this link within <b>{RESET_TOKEN_TTL_MIN} minutes</b> to choose a new one:</p>"
            f"<p><a href=\"{reset_url}\">{reset_url}</a></p>"
            f"<p>If you didn't ask for this, ignore this email — your password won't change.</p>"
        )
        msg.attach(MIMEText(text, "plain", "utf-8"))
        msg.attach(MIMEText(html, "html",  "utf-8"))

        with smtplib.SMTP(host, port, timeout=10) as s:
            if use_tls:
                s.starttls()
            if user:
                s.login(user, pwd)
            s.sendmail(sender, [to_email], msg.as_string())
        print(f"[auth.reset] SMTP sent to {to_email}")
        return True
    except Exception as exc:
        # Don't surface failure to the caller — we already logged the link.
        print(f"[auth.reset] SMTP send failed for {to_email}: {exc}")
        return False
