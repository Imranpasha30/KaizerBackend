"""Account requests + server-managed desktop API keys.

The desktop app's account/key lifecycle (operator design, 2026-08-25):

1. A prospective user submits "Request an account" from the desktop login
   screen → an AccountRequest row (password bcrypt-hashed immediately).
2. The admin approves in the console → the User row is minted on the spot
   (Free tier, same defaults as /api/auth/register).
3. At sign-in the desktop fetches /api/desktop/keys-bundle and injects the
   resolved keys into the LOCAL engine's key store; at sign-out the shell
   wipes them. Users may still override any key in Settings afterwards.

Key storage: ManagedApiKey rows — user_id NULL = the DEFAULT bundle every
user gets; a user-specific row overrides that key for that user. Values
are Fernet-encrypted (crypto.py, KAIZER_ENCRYPTION_KEY). Only the names
in INJECTABLE_KEYS are ever accepted or served — server infrastructure
secrets (DB, R2, JWT, SMTP…) can never enter this table.

Spend tracking note: exact per-user spend requires assigning each user
their OWN provider key (per-user rows here) — then every provider console
attributes usage per key. A shared default bundle cannot be split by user
at the provider.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

import auth as _auth
import crypto
import models
from database import get_db

router = APIRouter(tags=["account-requests"])

# The COMPLETE injectable set — audited 2026-08-25 across LIVE/DEV .env +
# engine usage (grep-verified; SARVAM/HF_TOKEN are read by nothing and were
# excluded). Everything the desktop needs to run fully functional:
#   LLM brains, STT, YouTube public data, story-image web search chain,
#   stock photos, HeyGen avatar generation, thumbnail image generation,
#   HeyGen defaults (plain settings, not secrets).
INJECTABLE_KEYS = (
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "DEEPGRAM_API_KEY",
    "YOUTUBE_DATA_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_CSE_ID",
    "PEXELS_API_KEY",
    "HEYGEN_API_KEY",
    "KAIZER_NANO_BANANA_API_KEY",
    "HEYGEN_DEFAULT_AVATAR_ID",
    "HEYGEN_DEFAULT_VOICE_ID",
)


def _require_admin(user: models.User) -> None:
    if not getattr(user, "is_admin", False):
        raise HTTPException(403, "Admin only")


# ─── Public: request an account ───────────────────────────────────────────

class RequestAccountIn(BaseModel):
    email:    EmailStr
    password: str = Field(..., min_length=6, max_length=200)
    name:     str = Field("", max_length=255)
    note:     str = Field("", max_length=1000)


@router.post("/api/auth/request-account")
def request_account(payload: RequestAccountIn, request: Request,
                    db: Session = Depends(get_db)) -> dict:
    """Desktop login screen → 'Request an account'. IP rate-limited with
    the same bucket as login (this is an unauthenticated write)."""
    from rate_limit import check_ip_rate as _check_ip_rate
    xff = request.headers.get("x-forwarded-for", "")
    ip = (xff.split(",")[0].strip() if xff else
          (request.client.host if request.client else "unknown"))
    allowed, retry_after, _rem = _check_ip_rate(ip)
    if not allowed:
        raise HTTPException(429, f"Too many attempts. Retry in {int(retry_after)}s.",
                            headers={"Retry-After": str(max(1, int(retry_after)))})

    email = payload.email.lower().strip()
    if db.query(models.User).filter(models.User.email == email).first():
        raise HTTPException(409, "An account with this email already exists — just sign in.")
    pending = (db.query(models.AccountRequest)
               .filter(models.AccountRequest.email == email,
                       models.AccountRequest.status == "pending").first())
    if pending:
        # Idempotent: re-submitting refreshes the password/note instead of
        # stacking duplicate rows for the admin to wade through.
        pending.password_hash = _auth.hash_password(payload.password)
        pending.name = (payload.name or "").strip()
        pending.note = (payload.note or "").strip()
        db.commit()
        return {"ok": True, "status": "pending",
                "message": "Your request is already with the admin — we updated it."}

    row = models.AccountRequest(
        email=email,
        name=(payload.name or "").strip(),
        password_hash=_auth.hash_password(payload.password),
        note=(payload.note or "").strip(),
        status="pending",
    )
    db.add(row); db.commit()
    return {"ok": True, "status": "pending",
            "message": "Request sent — you'll be able to sign in once the admin approves it."}


# ─── Admin: review requests ───────────────────────────────────────────────

@router.get("/api/admin/account-requests")
def list_requests(
    status: str = Query("pending", pattern="^(pending|approved|rejected|all)$"),
    db: Session = Depends(get_db),
    user: models.User = Depends(_auth.current_user),
) -> list[dict]:
    _require_admin(user)
    q = db.query(models.AccountRequest)
    if status != "all":
        q = q.filter(models.AccountRequest.status == status)
    rows = q.order_by(models.AccountRequest.created_at.desc()).limit(200).all()
    return [{
        "id": r.id, "email": r.email, "name": r.name, "note": r.note,
        "status": r.status,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "decided_at": r.decided_at.isoformat() if r.decided_at else None,
    } for r in rows]


@router.post("/api/admin/account-requests/{req_id}/approve")
def approve_request(req_id: int, db: Session = Depends(get_db),
                    user: models.User = Depends(_auth.current_user)) -> dict:
    _require_admin(user)
    r = db.query(models.AccountRequest).filter(models.AccountRequest.id == req_id).first()
    if not r:
        raise HTTPException(404, "Request not found")
    if r.status != "pending":
        raise HTTPException(409, f"Request already {r.status}")
    if db.query(models.User).filter(models.User.email == r.email).first():
        raise HTTPException(409, "A user with this email now exists — reject the request instead.")

    # Same defaults as /api/auth/register (Free tier so publishing works).
    from routers.auth import _default_plan_tier_id
    u = models.User(
        email=r.email,
        name=r.name or "",
        password_hash=r.password_hash,   # already bcrypt — minted as-is
        is_active=True,
        plan_tier_id=_default_plan_tier_id(db),
    )
    db.add(u)
    r.status = "approved"
    r.decided_at = datetime.now(timezone.utc)
    r.decided_by = user.id
    db.commit(); db.refresh(u)

    # Auto-mint this user's OWN Google keys (YouTube + Gemini) in the
    # operator's project so their usage is tracked per user. NEVER blocks or
    # fails approval — Google being down/unconfigured just leaves the user on
    # the shared bundle, visible + retryable in the Super Admin billing panel.
    minting = False
    try:
        from services import google_key_minter
        if google_key_minter.is_configured():
            google_key_minter.create_pending_rows(db, u.id)
            minting = google_key_minter.start_mint_thread(u.id, u.email)
    except Exception as exc:
        print(f"[account-approve] key mint skipped: {exc}")
    return {"ok": True, "user_id": u.id, "email": u.email, "minting": minting}


@router.post("/api/admin/account-requests/{req_id}/reject")
def reject_request(req_id: int, db: Session = Depends(get_db),
                   user: models.User = Depends(_auth.current_user)) -> dict:
    _require_admin(user)
    r = db.query(models.AccountRequest).filter(models.AccountRequest.id == req_id).first()
    if not r:
        raise HTTPException(404, "Request not found")
    if r.status != "pending":
        raise HTTPException(409, f"Request already {r.status}")
    r.status = "rejected"
    r.decided_at = datetime.now(timezone.utc)
    r.decided_by = user.id
    db.commit()
    return {"ok": True}


# ─── Admin: managed key store ─────────────────────────────────────────────

class ManagedKeyIn(BaseModel):
    # user_id None → the DEFAULT bundle row shared by every user.
    user_id: Optional[int] = None
    name:    str = Field(..., max_length=64)
    # Empty value deletes the row (mirrors the desktop-local keys contract).
    value:   str = Field("", max_length=4096)


def _masked(value: str) -> str:
    return ("•••" + value[-4:]) if len(value) > 6 else "•••"


@router.get("/api/admin/managed-keys")
def list_managed_keys(
    user_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    user: models.User = Depends(_auth.current_user),
) -> dict:
    """Masked view for the admin console. user_id omitted → the default
    bundle; user_id given → that user's overrides only."""
    _require_admin(user)
    q = db.query(models.ManagedApiKey)
    q = q.filter(models.ManagedApiKey.user_id == user_id) if user_id is not None \
        else q.filter(models.ManagedApiKey.user_id.is_(None))
    rows = q.order_by(models.ManagedApiKey.name).all()
    out = []
    for r in rows:
        try:
            val = crypto.decrypt(r.value_enc)
        except Exception:
            val = ""
        out.append({"name": r.name, "masked": _masked(val),
                    "updated_at": r.updated_at.isoformat() if r.updated_at else None})
    return {"user_id": user_id, "keys": out,
            "injectable": list(INJECTABLE_KEYS)}


@router.put("/api/admin/managed-keys")
def put_managed_key(payload: ManagedKeyIn, db: Session = Depends(get_db),
                    user: models.User = Depends(_auth.current_user)) -> dict:
    _require_admin(user)
    name = payload.name.strip().upper()
    if name not in INJECTABLE_KEYS:
        raise HTTPException(400, f"'{name}' is not an injectable key. Allowed: "
                                 f"{', '.join(INJECTABLE_KEYS)}")
    value = (payload.value or "").replace("\r", "").replace("\n", "").strip()
    row = (db.query(models.ManagedApiKey)
           .filter(models.ManagedApiKey.user_id == payload.user_id,
                   models.ManagedApiKey.name == name).first())
    if not value:
        if row:
            db.delete(row); db.commit()
        return {"ok": True, "name": name, "cleared": True}
    if row:
        row.value_enc = crypto.encrypt(value)
    else:
        db.add(models.ManagedApiKey(user_id=payload.user_id, name=name,
                                    value_enc=crypto.encrypt(value)))
    db.commit()
    return {"ok": True, "name": name, "masked": _masked(value)}


# ─── Desktop: the resolved bundle ─────────────────────────────────────────

@router.get("/api/desktop/keys-bundle")
def keys_bundle(db: Session = Depends(get_db),
                user: models.User = Depends(_auth.current_user)) -> dict:
    """The signed-in user's resolved key bundle: default rows overlaid by
    the user's own rows. The desktop shell injects these into the local
    engine at sign-in and wipes them at sign-out."""
    rows = (db.query(models.ManagedApiKey)
            .filter((models.ManagedApiKey.user_id.is_(None)) |
                    (models.ManagedApiKey.user_id == user.id))
            .all())
    resolved: dict[str, str] = {}
    # Defaults first, then user rows override.
    for r in sorted(rows, key=lambda x: 0 if x.user_id is None else 1):
        if r.name not in INJECTABLE_KEYS:
            continue
        try:
            resolved[r.name] = crypto.decrypt(r.value_enc)
        except Exception:
            continue
    return {"keys": resolved, "names": sorted(resolved.keys())}
