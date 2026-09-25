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
import datetime as _dt


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


def _onboarding_done(u: models.User) -> bool:
    """Does this account have an onboarding row?

    Errs towards True. A False here forces the person into a form, so the
    cost of being wrong in that direction is someone locked out of the
    product; the cost of the other direction is a form not shown.
    """
    try:
        from sqlalchemy.orm import object_session
        db = object_session(u)
        if db is None:
            return True
        return db.query(models.OnboardingProfile).filter(
            models.OnboardingProfile.user_id == u.id).first() is not None
    except Exception:
        return True


def _public_user(u: models.User) -> dict:
    raw_socials = getattr(u, "socials", None) or {}
    avatar_url = ""
    try:
        from routers.profile import user_avatar_url as _avatar_url
        avatar_url = _avatar_url(u)
    except Exception:
        avatar_url = ""
    rating_count = int(getattr(u, "creator_rating_count", 0) or 0)
    rating_sum   = int(getattr(u, "creator_rating_sum",   0) or 0)
    rating_avg   = round(rating_sum / rating_count, 2) if rating_count else 0.0
    return {
        "id":     u.id,
        "email":  u.email,
        "name":   u.name or u.email.split("@")[0],
        "google": bool(u.google_sub),
        "is_admin": bool(u.is_admin),
        "is_creative": bool(getattr(u, "is_creative", False)),
        "plan":   (getattr(u, "plan", None) or "free"),
        "avatar_url":   avatar_url,
        "creator_rating_avg":   rating_avg,
        "creator_rating_count": rating_count,
        "socials": raw_socials if isinstance(raw_socials, dict) else {},
        # Has this account filled the one-time details form? Every sign-in
        # path returns _public_user, so adding it here covers password,
        # Google and emailed-code at once.
        #
        # TRUE ON ANY DOUBT. If the table is not there yet, or the session
        # has gone, we say completed rather than risk gating a user we
        # cannot then let through.
        "onboarding_completed": _onboarding_done(u),
        "created_at":    u.created_at.isoformat() if u.created_at else None,
        "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
    }


def _with_token(u: models.User) -> dict:
    return {"token": _auth.issue_token(u), "user": _public_user(u)}


# ─── Endpoints ────────────────────────────────────────────────────────────

def _default_plan_tier_id(db) -> "int | None":
    """Tier assigned to every brand-new account = Free. Without this, new
    users land with plan_tier_id NULL and publishing 400s with
    'plan_tier_unknown'. Returns None only if the plan_tiers table isn't
    seeded (then the publish flow prompts to pick a plan)."""
    try:
        t = db.query(models.PlanTier).filter(models.PlanTier.name == "free").first()
        return t.id if t else None
    except Exception:
        return None


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
        plan_tier_id=_default_plan_tier_id(db),
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
    if u and not u.password_hash:
        # Google-Sign-In-only account: explicit reason instead of the
        # generic "Wrong email or password." Otherwise the user stares
        # at a confusing error and has no idea this account doesn't
        # accept a password — they need to click "Sign in with Google".
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="This account is linked to Google. Click 'Sign in with Google' instead of using a password.",
        )
    if not u or not _auth.verify_password(payload.password, u.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Wrong email or password.",
        )
    if not u.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled.")
    u.last_login_at = datetime.now(timezone.utc)
    db.commit()
    return _with_token(u)

# ─── Sign in with a code sent to your email ───────────────────────────
#
# No password. Type your address, get six digits, type them back. The
# account is created on the first successful code, so there is no separate
# sign-up to get out of step with this.
#
# The two shapes below exist to stop the two things that go wrong with
# emailed codes: REQUESTING one tells you nothing about whether the address
# is known (otherwise the response enumerates the customer list), and
# VERIFYING one is rate-limited per IP and per code (otherwise six digits is
# a million guesses nobody is counting).

class CodeRequestIn(BaseModel):
    email: str


class CodeVerifyIn(BaseModel):
    email: str
    code: str


@router.post("/login-code/request")
def login_code_request(payload: CodeRequestIn, request: Request,
                       db: Session = Depends(get_db)):
    """Email a six-digit sign-in code.

    ALWAYS answers the same way. Whether the address has an account, is
    malformed, or the mail server refused, the caller gets {"ok": true} and
    the same wording -- because anything else turns this endpoint into a
    directory of who is a customer.
    """
    import login_code as _lc
    import mailer as _mail
    from rate_limit import check_ip_rate as _check_ip_rate

    xff = request.headers.get("x-forwarded-for", "")
    ip = (xff.split(",")[0].strip() if xff else
          (request.client.host if request.client else "unknown"))
    allowed, retry_after, _ = _check_ip_rate(ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Too many attempts. Retry in {int(retry_after)}s.",
            headers={"Retry-After": str(max(1, int(retry_after)))})

    said = {"ok": True,
            "message": "If that address can sign in, a code is on its way."}
    email = _lc.normalise_email(payload.email)
    if not _mail.valid_email(email):
        return said

    # Reuse a code issued seconds ago rather than minting a second one --
    # a double-clicked button otherwise leaves the first email dead on
    # arrival, which reads as "the code doesn't work".
    recent = (db.query(models.LoginCode)
                .filter(models.LoginCode.email == email)
                .order_by(models.LoginCode.id.desc()).first())
    if recent is not None and _lc.within_grace(recent):
        print(f"[auth.code] reusing the code issued moments ago for {email}")
        return said

    code = _lc.make_code()
    user = db.query(models.User).filter(models.User.email == email).first()
    row = models.LoginCode(
        email=email, user_id=(user.id if user else None),
        code_hash=_lc.hash_code(code), expires_at=_lc.expiry(),
        attempts=0, requested_ip=ip[:64])
    db.add(row)
    db.commit()

    # ALWAYS printed. On a box with no mail credential, or the day Zoho is
    # down, this is how the operator still gets in.
    print(f"[auth.code] sign-in code for {email}: {code}")
    text, html = _lc.body(code, name=(user.name if user else ""))
    _mail.send(to_email=email, subject=_lc.subject(), text=text, html=html,
               to_name=(user.name if user else ""))
    return said


@router.post("/login-code/verify")
def login_code_verify(payload: CodeVerifyIn, request: Request,
                      db: Session = Depends(get_db)):
    """Exchange the code for a token. Creates the account on first use."""
    import login_code as _lc
    from rate_limit import check_ip_rate as _check_ip_rate

    xff = request.headers.get("x-forwarded-for", "")
    ip = (xff.split(",")[0].strip() if xff else
          (request.client.host if request.client else "unknown"))
    allowed, retry_after, _ = _check_ip_rate(ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Too many attempts. Retry in {int(retry_after)}s.",
            headers={"Retry-After": str(max(1, int(retry_after)))})

    email = _lc.normalise_email(payload.email)
    code = _lc.clean_code(payload.code)
    WRONG = HTTPException(status_code=400,
                          detail="That code is wrong or has expired.")
    if not _lc.looks_like_a_code(code):
        raise WRONG

    row = (db.query(models.LoginCode)
             .filter(models.LoginCode.email == email,
                     models.LoginCode.used_at.is_(None))
             .order_by(models.LoginCode.id.desc()).first())
    if row is None or not _lc.is_live(row):
        raise WRONG

    # Count the try BEFORE comparing, and commit it, so a crash or a
    # disconnect mid-request cannot be used to get a free guess.
    row.attempts = int(row.attempts or 0) + 1
    db.commit()

    if row.code_hash != _lc.hash_code(code):
        left = max(0, _lc.MAX_ATTEMPTS - int(row.attempts or 0))
        raise HTTPException(
            status_code=400,
            detail=(f"That code is wrong. {left} tr{'y' if left == 1 else 'ies'} left."
                    if left else "That code is wrong and has now expired."))

    user = db.query(models.User).filter(models.User.email == email).first()
    created = False
    if user is None:
        # First code IS the sign-up, and it must build the account EXACTLY
        # as /register does. The first version here set only the email,
        # name and an empty password -- which leaves plan_tier_id NULL, and
        # a NULL tier makes every publish 400 with "plan_tier_unknown".
        # A user who signed in by code would have had a working account
        # that silently could not publish.
        #
        # password_hash stays empty on purpose: the password login path
        # already recognises that and says so, rather than failing blankly.
        user = models.User(
            email=email,
            name=email.split("@")[0],
            password_hash="",
            is_active=True,
            plan_tier_id=_default_plan_tier_id(db),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        created = True
        print(f"[auth.code] created an account for {email}")

    row.used_at = _dt.datetime.now(_dt.timezone.utc)
    row.user_id = user.id
    user.last_login_at = _dt.datetime.now(_dt.timezone.utc)
    db.commit()

    # _with_token, not a hand-rolled dict. The frontend reads res.token and
    # the canonical user shape comes from _public_user; a bespoke
    # {"access_token": ...} here would have signed nobody in.
    return {**_with_token(user), "created": created}



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
                plan_tier_id=_default_plan_tier_id(db),
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
        # The sign-in screen only offers the emailed-code option where the
        # backend says it can serve it. Without this key the routes below
        # exist and nothing in the UI reaches them.
        "code_login":        True,
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
