# Ported from kaizer-platform@d5fd482 server/routers/desktop.py.
# Changes: (1) licensing surface ONLY — the local render worker endpoints
# (jobs/pending, claim, complete, fail) were NOT ported (no local-render
# job path here yet); (2) ADDED POST /licenses/{license_id}/revoke
# (owner-only, sets revoked=True) which upstream lacks — the device-
# management UI needs it. Activation/limit/require_desktop_license logic
# is verbatim.
"""
routers.desktop
===============
Desktop app machine activation + licensing.

POST /api/desktop/activate — registers (or re-confirms) a machine fingerprint
against the signed-in user's account, enforcing a placeholder per-user
activation limit. GET /api/desktop/licenses — lists the caller's own active
machines (the desktop app's "manage devices" surface, and handy for
verifying activation worked). POST /api/desktop/licenses/{id}/revoke —
owner-only revocation for that same device-management surface.

This is deliberately the SMALL, bounded piece of "License: machine
fingerprint + key, N systems per contract" — not a full licensing system.
No key issuance, no billing tie-in; just "is this machine allowed to run
the app for this account right now."

Authentication
--------------
Activation/license-list/revoke use plain `auth.current_user` (same JWT the
web apps use). Endpoints that must prove they run on a licensed desktop
install use `require_desktop_license` instead — a regular JWT alone is NOT
enough for those; the caller must ALSO present an
`X-Desktop-Machine-Fingerprint` header naming an activated, unrevoked
`DesktopLicense` row for that user.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import auth
import models
from database import get_db

router = APIRouter(prefix="/api/desktop", tags=["desktop"])

# PLACEHOLDER, pending real per-contract economics ("N systems per
# contract" has no locked number yet). Not read from a PlanTier row — a
# flat constant, changed here when a real number is decided.
ACTIVATION_LIMIT = 3


class ActivateRequest(BaseModel):
    machine_fingerprint: str = Field(..., min_length=8, max_length=128)
    machine_label: str = Field(default="", max_length=255)


class LicenseSchema(BaseModel):
    id: int
    machine_fingerprint: str
    machine_label: str
    activated_at: Optional[datetime]
    last_seen_at: Optional[datetime]
    revoked: bool


class ActivateResponse(BaseModel):
    license: LicenseSchema
    active_count: int
    limit: int
    already_activated: bool


def _to_schema(row: models.DesktopLicense) -> LicenseSchema:
    return LicenseSchema(
        id=row.id,
        machine_fingerprint=row.machine_fingerprint,
        machine_label=row.machine_label or "",
        activated_at=row.activated_at,
        last_seen_at=row.last_seen_at,
        revoked=row.revoked,
    )


@router.post("/activate", response_model=ActivateResponse)
def activate_desktop(
    payload: ActivateRequest,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Register this machine fingerprint for the signed-in user.

    Idempotent for a fingerprint already activated on this account (just
    bumps `last_seen_at` — a re-launch of the app is not a new activation).
    Otherwise enforces `ACTIVATION_LIMIT` active (non-revoked) machines per
    user before creating a new row.
    """
    fingerprint = payload.machine_fingerprint.strip()
    if not fingerprint:
        raise HTTPException(status_code=422, detail="machine_fingerprint is required")

    existing = (
        db.query(models.DesktopLicense)
        .filter(
            models.DesktopLicense.user_id == user.id,
            models.DesktopLicense.machine_fingerprint == fingerprint,
        )
        .first()
    )

    if existing:
        if existing.revoked:
            raise HTTPException(
                status_code=403,
                detail="This machine's license was revoked. Contact support to reactivate.",
            )
        existing.last_seen_at = datetime.now(timezone.utc)
        if payload.machine_label:
            existing.machine_label = payload.machine_label
        db.commit()
        db.refresh(existing)
        active_count = (
            db.query(models.DesktopLicense)
            .filter(models.DesktopLicense.user_id == user.id,
                     models.DesktopLicense.revoked == False)  # noqa: E712
            .count()
        )
        return ActivateResponse(
            license=_to_schema(existing),
            active_count=active_count,
            limit=ACTIVATION_LIMIT,
            already_activated=True,
        )

    active_count = (
        db.query(models.DesktopLicense)
        .filter(models.DesktopLicense.user_id == user.id,
                 models.DesktopLicense.revoked == False)  # noqa: E712
        .count()
    )
    if active_count >= ACTIVATION_LIMIT:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Activation limit reached ({active_count}/{ACTIVATION_LIMIT} systems "
                "for this account). Deactivate another machine or contact support to "
                "raise your limit."
            ),
        )

    row = models.DesktopLicense(
        user_id=user.id,
        machine_fingerprint=fingerprint,
        machine_label=payload.machine_label,
        last_seen_at=datetime.now(timezone.utc),
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return ActivateResponse(
        license=_to_schema(row),
        active_count=active_count + 1,
        limit=ACTIVATION_LIMIT,
        already_activated=False,
    )


@router.get("/licenses", response_model=list[LicenseSchema])
def list_desktop_licenses(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """List the signed-in user's own activated machines (newest first)."""
    rows = (
        db.query(models.DesktopLicense)
        .filter(models.DesktopLicense.user_id == user.id)
        .order_by(models.DesktopLicense.activated_at.desc())
        .all()
    )
    return [_to_schema(r) for r in rows]


@router.post("/licenses/{license_id}/revoke", response_model=LicenseSchema)
def revoke_desktop_license(
    license_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Revoke one of the CALLER'S OWN machine licenses (device-management
    UI). NOT in upstream — added for Kaizer X.

    Owner-only: the row must belong to the signed-in user; anyone else's
    license id 404s (existence is not leaked). Sets `revoked=True` rather
    than deleting — activation history stays auditable, and the vendor's
    activate/require_desktop_license paths already treat a revoked row as
    a hard 403. Idempotent: revoking an already-revoked license is a no-op
    success.
    """
    row = (
        db.query(models.DesktopLicense)
        .filter(
            models.DesktopLicense.id == license_id,
            models.DesktopLicense.user_id == user.id,
        )
        .first()
    )
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"License {license_id} not found on this account.",
        )
    if not row.revoked:
        row.revoked = True
        db.commit()
        db.refresh(row)
    return _to_schema(row)


# ─────────────────────────────────────────────────────────────────────────
# Licensed-install proof — dependency for future desktop-only endpoints
# ─────────────────────────────────────────────────────────────────────────

def require_desktop_license(
    x_machine_fingerprint: str = Header(..., alias="X-Desktop-Machine-Fingerprint"),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> models.DesktopLicense:
    """Prove the caller is a licensed desktop install, not just any JWT
    holder. A regular web/mobile session carries the same JWT shape as the
    desktop app, so `auth.current_user` alone can't distinguish "the real
    desktop app for this account" from "any other logged-in client".

    Requires an `X-Desktop-Machine-Fingerprint` header naming a
    `DesktopLicense` row that (a) belongs to the caller's account and
    (b) is not revoked. Bumps `last_seen_at` on every successful call --
    doubles as the desktop app's heartbeat, no separate ping endpoint
    needed.
    """
    fingerprint = (x_machine_fingerprint or "").strip()
    if not fingerprint:
        raise HTTPException(
            status_code=422,
            detail="X-Desktop-Machine-Fingerprint header is required.",
        )
    lic = (
        db.query(models.DesktopLicense)
        .filter(
            models.DesktopLicense.user_id == user.id,
            models.DesktopLicense.machine_fingerprint == fingerprint,
        )
        .first()
    )
    if lic is None:
        raise HTTPException(
            status_code=403,
            detail="No activated desktop license for this machine on this "
                   "account. Call POST /api/desktop/activate first.",
        )
    if lic.revoked:
        raise HTTPException(
            status_code=403,
            detail="This machine's desktop license has been revoked.",
        )
    lic.last_seen_at = datetime.now(timezone.utc)
    db.commit()
    return lic


# ─────────────────────────────────────────────────────────────────────────
# Admin — every user's activated machines (operator: "for admin keep a tab
# where show licenses granted for the user"). NOT in upstream.
# ─────────────────────────────────────────────────────────────────────────

@router.get("/admin/licenses")
def admin_list_licenses(
    db: Session = Depends(get_db),
    admin: models.User = Depends(auth.admin_required),
) -> dict:
    """Every desktop license across all accounts, newest first, with the
    owning user's identity. Fingerprints are shown truncated — the full
    value never leaves the server (it is effectively a device credential)."""
    rows = (
        db.query(models.DesktopLicense, models.User)
        .join(models.User, models.User.id == models.DesktopLicense.user_id)
        .order_by(models.DesktopLicense.activated_at.desc())
        .all()
    )
    out = []
    for lic, owner in rows:
        out.append({
            "id": lic.id,
            "user_id": owner.id,
            "user_email": owner.email,
            "user_name": getattr(owner, "name", "") or "",
            "machine_label": lic.machine_label or "",
            "fingerprint_short": (lic.machine_fingerprint or "")[:12],
            "activated_at": lic.activated_at.isoformat() if lic.activated_at else None,
            "last_seen_at": lic.last_seen_at.isoformat() if lic.last_seen_at else None,
            "revoked": bool(lic.revoked),
        })
    active = sum(1 for r in out if not r["revoked"])
    return {"licenses": out, "total": len(out), "active": active,
            "limit_per_user": ACTIVATION_LIMIT}


@router.post("/admin/licenses/{license_id}/revoke")
def admin_revoke_license(
    license_id: int,
    db: Session = Depends(get_db),
    admin: models.User = Depends(auth.admin_required),
) -> dict:
    """Admin revoke of ANY user's machine license (support flow: a customer
    burned a slot on a VPN-adapter fingerprint change — revoke the stale
    row so they can re-activate). Sets revoked=True, never deletes;
    idempotent."""
    row = db.get(models.DesktopLicense, license_id)
    if row is None:
        raise HTTPException(404, f"License {license_id} not found.")
    if not row.revoked:
        row.revoked = True
        db.commit()
    return {"ok": True, "id": row.id, "revoked": True}
