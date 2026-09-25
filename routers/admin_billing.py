"""Super Admin: per-user Google key minting + usage & billing view.

Endpoints (all admin-gated; full paths, no router prefix — matches
routers/account_requests.py):
  POST /api/admin/users/{id}/mint-keys      → start async mint (pending rows)
  POST /api/admin/users/{id}/revoke-keys    → sync revoke + shared-bundle fallback
  GET  /api/admin/users/{id}/minted-keys    → per-user key status poll
  GET  /api/admin/usage?days=7|30&refresh=  → per-user request counts + billing math
  GET  /api/admin/billing-rates             → operator $/1000 rates
  PUT  /api/admin/billing-rates             → set rates

The usage endpoint does ALL the math server-side (attribution, cost, suggested
charge) so the desktop and web panels render identical numbers with no client
logic. See services/google_key_minter.py + services/google_usage.py.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import auth as _auth
import models
from database import get_db
from services import google_key_minter as gkm
from services import google_usage

router = APIRouter(tags=["admin-billing"])

RATE_METRICS = ("youtube_requests", "gemini_requests")

_CAVEATS = [
    "Counts are API requests, not YouTube quota units (a search costs 100 units but counts as 1 request).",
    "Gemini calls made through the Vertex service-account path are not attributed per user — only API-key calls are.",
    "Per-user token counts are not available from Google — request counts only.",
    "All minted keys share the project's quota pools; per-user metrics are not per-user quotas.",
    "Cloud Monitoring lags real traffic by a few minutes.",
    "Key changes reach a user's desktop at their next sign-in.",
]


def _require_admin(user: models.User) -> None:
    if not getattr(user, "is_admin", False):
        raise HTTPException(403, "Admin only")


# ─── mint / revoke / status ──────────────────────────────────────────

class MintIn(BaseModel):
    names: Optional[list[str]] = None   # subset of KEY_SPECS; default both


def _user_or_404(db: Session, user_id: int) -> models.User:
    u = db.query(models.User).filter(models.User.id == user_id).first()
    if not u:
        raise HTTPException(404, "User not found")
    return u


@router.post("/api/admin/users/{user_id}/mint-keys")
def mint_keys(user_id: int, payload: MintIn = MintIn(),
              db: Session = Depends(get_db),
              user: models.User = Depends(_auth.current_user)) -> dict:
    _require_admin(user)
    u = _user_or_404(db, user_id)
    if not gkm.is_configured():
        raise HTTPException(
            503, "Key minting is not configured (service account / project missing).")
    if gkm.is_minting(user_id):
        raise HTTPException(409, "Minting already in progress for this user.")
    only = payload.names
    if only:
        bad = [n for n in only if n not in gkm.KEY_SPECS]
        if bad:
            raise HTTPException(400, f"Unknown key names: {', '.join(bad)}")
    pending = gkm.create_pending_rows(db, user_id, only)
    started = gkm.start_mint_thread(user_id, u.email, only)
    return {"ok": True, "started": started, "user_id": user_id,
            "keys": [{"env_name": n, "status": "pending"} for n in pending]}


@router.post("/api/admin/users/{user_id}/revoke-keys")
def revoke_keys(user_id: int, payload: MintIn = MintIn(),
                db: Session = Depends(get_db),
                user: models.User = Depends(_auth.current_user)) -> dict:
    _require_admin(user)
    _user_or_404(db, user_id)
    results = gkm.revoke_user_keys(user_id, payload.names)
    ok = all(r["revoked"] for r in results) if results else True
    return {"ok": ok, "user_id": user_id, "results": results,
            "note": "Removed per-user keys fall back to the shared bundle at the "
                    "user's next sign-in."}


@router.get("/api/admin/users/{user_id}/minted-keys")
def minted_keys(user_id: int, db: Session = Depends(get_db),
                user: models.User = Depends(_auth.current_user)) -> dict:
    _require_admin(user)
    _user_or_404(db, user_id)
    rows = (db.query(models.GoogleMintedKey)
              .filter(models.GoogleMintedKey.user_id == user_id).all())
    return {"user_id": user_id, "minting": gkm.is_minting(user_id), "keys": [{
        "env_name": r.env_name, "status": r.status, "error": r.error or "",
        "key_uid": r.key_uid or "", "key_resource_name": r.key_resource_name or "",
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        "revoked_at": r.revoked_at.isoformat() if r.revoked_at else None,
    } for r in rows]}


# ─── billing rates ───────────────────────────────────────────────────

class RateIn(BaseModel):
    cost_per_1000:   float = Field(0.0, ge=0)
    charge_per_1000: float = Field(0.0, ge=0)
    currency:        str   = Field("USD", max_length=8)


class RatesIn(BaseModel):
    rates: dict[str, RateIn]


def _rates_map(db: Session) -> dict[str, dict]:
    rows = {r.metric: r for r in db.query(models.BillingRate).all()}
    out: dict[str, dict] = {}
    for m in RATE_METRICS:
        r = rows.get(m)
        out[m] = {
            "cost_per_1000":   float(r.cost_per_1000) if r else 0.0,
            "charge_per_1000": float(r.charge_per_1000) if r else 0.0,
            "currency":        (r.currency if r else "USD"),
        }
    return out


@router.get("/api/admin/billing-rates")
def get_billing_rates(db: Session = Depends(get_db),
                      user: models.User = Depends(_auth.current_user)) -> dict:
    _require_admin(user)
    return {"rates": _rates_map(db)}


@router.put("/api/admin/billing-rates")
def put_billing_rates(payload: RatesIn, db: Session = Depends(get_db),
                      user: models.User = Depends(_auth.current_user)) -> dict:
    _require_admin(user)
    for metric, rate in payload.rates.items():
        if metric not in RATE_METRICS:
            raise HTTPException(400, f"Unknown billing metric: {metric}")
        row = (db.query(models.BillingRate)
                 .filter(models.BillingRate.metric == metric).first())
        if not row:
            row = models.BillingRate(metric=metric)
            db.add(row)
        row.cost_per_1000 = float(rate.cost_per_1000)
        row.charge_per_1000 = float(rate.charge_per_1000)
        row.currency = rate.currency or "USD"
    db.commit()
    return {"ok": True, "rates": _rates_map(db)}


# ─── usage + billing ─────────────────────────────────────────────────

def _money(x: float) -> float:
    return round(x + 1e-9, 2)


@router.get("/api/admin/usage")
def usage(days: int = Query(7, ge=1, le=90), refresh: int = Query(0),
          db: Session = Depends(get_db),
          user: models.User = Depends(_auth.current_user)) -> dict:
    _require_admin(user)
    raw = google_usage.fetch_usage(days, force=bool(refresh))
    by_cred = raw.get("by_credential", {})
    rates = _rates_map(db)

    # uid -> (user_id, env_name); include revoked rows so historical usage
    # still attributes to the right user.
    uid_to_user: dict[str, int] = {}
    for row in db.query(models.GoogleMintedKey).all():
        if row.key_uid:
            uid_to_user[f"apikey:{row.key_uid}"] = row.user_id

    shared_uids = gkm.lookup_shared_key_uids()   # {"apikey:<uid>": env_name} or {}

    # per-user status map from minted rows
    status_by_user: dict[int, dict] = {}
    for row in db.query(models.GoogleMintedKey).all():
        status_by_user.setdefault(row.user_id, {})[row.env_name] = {
            "status": row.status, "error": row.error or ""}

    def _rate_math(yt: int, gm: int) -> tuple[float, float]:
        cost = (yt / 1000.0) * rates["youtube_requests"]["cost_per_1000"] \
             + (gm / 1000.0) * rates["gemini_requests"]["cost_per_1000"]
        charge = (yt / 1000.0) * rates["youtube_requests"]["charge_per_1000"] \
               + (gm / 1000.0) * rates["gemini_requests"]["charge_per_1000"]
        return _money(cost), _money(charge)

    per_user: dict[int, dict] = {}
    shared = {"youtube_requests": 0, "gemini_requests": 0}
    unattributed = {"youtube_requests": 0, "gemini_requests": 0, "credentials": []}
    for cred, d in by_cred.items():
        yt = int(d.get("youtube_requests", 0))
        gm = int(d.get("gemini_requests", 0))
        if cred in uid_to_user:
            uid = uid_to_user[cred]
            acc = per_user.setdefault(uid, {"youtube_requests": 0, "gemini_requests": 0})
            acc["youtube_requests"] += yt
            acc["gemini_requests"] += gm
        elif cred in shared_uids:
            shared["youtube_requests"] += yt
            shared["gemini_requests"] += gm
        else:
            unattributed["youtube_requests"] += yt
            unattributed["gemini_requests"] += gm
            if cred not in ("(no credential)",):
                unattributed["credentials"].append(cred)

    # list ALL users (cap 500) so zero-usage users still show with key status
    users = (db.query(models.User).order_by(models.User.id.asc()).limit(500).all())
    out_users = []
    tot_yt = tot_gm = 0
    tot_cost = tot_charge = 0.0
    for u in users:
        counts = per_user.get(u.id, {"youtube_requests": 0, "gemini_requests": 0})
        yt = counts["youtube_requests"]; gm = counts["gemini_requests"]
        cost, charge = _rate_math(yt, gm)
        st = status_by_user.get(u.id, {})
        keys = {}
        for env in gkm.KEY_SPECS:
            row = st.get(env)
            keys[env] = row if row else {"status": "shared", "error": ""}
        out_users.append({
            "user_id": u.id, "email": u.email, "name": getattr(u, "name", "") or "",
            "keys": keys, "youtube_requests": yt, "gemini_requests": gm,
            "est_cost": cost, "suggested_charge": charge,
        })
        tot_yt += yt; tot_gm += gm; tot_cost += cost; tot_charge += charge

    return {
        "days": days, "start": raw.get("start"), "end": raw.get("end"),
        "fetched_at": raw.get("fetched_at"), "cached": raw.get("cached", False),
        "monitoring_error": raw.get("error"),
        "configured": gkm.is_configured(),
        "rates": rates,
        "users": out_users,
        "shared": shared,
        "unattributed": unattributed,
        "totals": {"youtube_requests": tot_yt, "gemini_requests": tot_gm,
                   "est_cost": _money(tot_cost), "suggested_charge": _money(tot_charge)},
        "caveats": _CAVEATS,
    }
