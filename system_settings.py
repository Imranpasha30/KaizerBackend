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
UPLOAD_PROVIDER_VALID    = {"postiz", "kaizer"}


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
