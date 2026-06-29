"""Tiny key-value config layer for system-wide toggles.

The single canonical entry point — every reader / writer goes
through these two helpers so we never duplicate default values or
allowed enums across the codebase.

Today's keys:
  - ``upload_provider``      → "postiz" (default) or "kaizer"
                                Picks which path /api/clips/{id}/publish
                                takes for YouTube uploads. Postiz is
                                the default until our app's verification
                                completes; admins can flip via the
                                admin Settings UI.

To add a new setting:
  1. Add a constant + default below.
  2. Read with ``get_system_setting(db, KEY, DEFAULT)``.
  3. Write with ``set_system_setting(db, KEY, value)`` (admin endpoint).
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

import models


# ─── Setting keys + defaults ─────────────────────────────────────────────────

UPLOAD_PROVIDER          = "upload_provider"
UPLOAD_PROVIDER_DEFAULT  = "postiz"
# Whitelist of provider keys the worker can route to. The set is also
# the source of truth for the publishers/__init__.py registry — every
# key here MUST have a registered Publisher subclass, and vice versa.
UPLOAD_PROVIDER_VALID    = {
    "postiz", "kaizer",                       # legacy + native YouTube
    "meta_fb", "meta_ig",                     # Facebook Page + IG Reels
    "x", "linkedin", "tiktok",                # scaffolded for credentials
}

# Delivery mode — controls whether Postiz AUTO-FALLBACK is active.
#   "testing"    → current manual behaviour: a channel uploads natively
#                  (our own quota) UNLESS the user explicitly set it to
#                  Postiz. NO automatic fallback. (Default — safe.)
#   "production" → auto-fallback: native first, and when the YouTube daily
#                  quota can't cover a target, channels that have a bound
#                  Postiz integration are routed to Postiz automatically so
#                  publishing never stalls while our quota is low.
DELIVERY_MODE          = "delivery_mode"
DELIVERY_MODE_DEFAULT  = "testing"
DELIVERY_MODE_VALID    = {"testing", "production"}


# ─── Read / write helpers ────────────────────────────────────────────────────

def get_system_setting(db: Session, key: str, default: str = "") -> str:
    """Return the stored value for ``key``, or ``default`` if absent.

    Never raises — a brand-new install with no rows just sees the
    default everywhere.
    """
    row = db.query(models.SystemSetting).filter(
        models.SystemSetting.key == key
    ).first()
    if not row or row.value is None:
        return default
    return str(row.value)


def set_system_setting(db: Session, key: str, value: str) -> None:
    """Upsert ``key`` → ``value``. Caller commits (lets the endpoint
    bundle multiple changes in one transaction)."""
    row = db.query(models.SystemSetting).filter(
        models.SystemSetting.key == key
    ).first()
    if row is None:
        row = models.SystemSetting(key=key, value=str(value or ""))
        db.add(row)
    else:
        row.value = str(value or "")


# ─── Convenience reads for common settings ──────────────────────────────────

def get_upload_provider(db: Session) -> str:
    """Return 'postiz' or 'kaizer'. Falls back to default + clamps the
    value so a malformed DB entry can't crash the publish flow."""
    v = get_system_setting(db, UPLOAD_PROVIDER, UPLOAD_PROVIDER_DEFAULT).strip().lower()
    return v if v in UPLOAD_PROVIDER_VALID else UPLOAD_PROVIDER_DEFAULT


def get_delivery_mode(db: Session) -> str:
    """Return 'testing' (default) or 'production'. Clamps malformed values
    so the publish flow can never crash on a bad DB entry. 'production'
    enables Postiz auto-fallback on YouTube-quota exhaustion."""
    v = get_system_setting(db, DELIVERY_MODE, DELIVERY_MODE_DEFAULT).strip().lower()
    return v if v in DELIVERY_MODE_VALID else DELIVERY_MODE_DEFAULT
