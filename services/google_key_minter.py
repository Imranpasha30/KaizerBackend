"""Mint / revoke per-user Google API keys via the API Keys API (apikeys v2).

Operator design (2026-08-25): every approved desktop user gets their OWN
YouTube Data + Gemini API keys, created in the operator's Google project and
named with the user's email. The key STRINGS are stored as per-user
ManagedApiKey rows (Fernet) so the existing keys-bundle injection delivers
them to the desktop at sign-in — zero desktop-engine changes. This module
holds only the Google-side lifecycle + the uid→user mapping metadata
(GoogleMintedKey rows) that the usage/billing view joins against.

SOFT-FAIL CONTRACT: nothing here ever raises into a request path. Account
approval calls start_mint_thread() fire-and-forget; if Google is down or
unconfigured the user simply stays on the shared default bundle, visible and
retryable in the Super Admin billing panel.

Pattern-matched to services/quota_sync.py: function-local google imports,
cache_discovery=False, service account from the same creds.

Project / credential resolution (both must be the SAME project as
google_usage.py so metrics and keys line up):
    project: KAIZER_KEYMINT_PROJECT > GOOGLE_CLOUD_PROJECT > KAIZER_GCP_PROJECT
    creds:   KAIZER_KEYMINT_CREDENTIALS > GOOGLE_APPLICATION_CREDENTIALS
             > KAIZER_VERTEX_CREDENTIALS

Limits baked in: Google allows 300 API keys / project (~150 users at 2 keys
each); keyId must be unique per project INCLUDING soft-deleted keys (30-day
window) so we use a random suffix, never a deterministic id.
"""
from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("kaizer.key_minter")

#: env_name -> the Google service the minted key is API-restricted to.
KEY_SPECS: dict[str, dict] = {
    "YOUTUBE_DATA_API_KEY": {"service": "youtube.googleapis.com",            "slug": "yt"},
    "GEMINI_API_KEY":       {"service": "generativelanguage.googleapis.com", "slug": "gl"},
}

_CREATE_POLL_S = 2.0          # LRO poll interval
_CREATE_TIMEOUT_S = 90.0      # give up -> status=failed (retryable via the Mint button)

_mint_lock = threading.Lock()
_inflight: set[int] = set()   # user_ids with a mint thread running (double-click guard)

# Module cache for the shared-bundle key uids (labels shared traffic honestly).
_shared_uids_cache: dict[str, str] = {}
_shared_uids_ts: float = 0.0
_SHARED_TTL_S = 600.0


# ─── config resolution ───────────────────────────────────────────────

def resolve_project() -> str:
    for var in ("KAIZER_KEYMINT_PROJECT", "GOOGLE_CLOUD_PROJECT", "KAIZER_GCP_PROJECT"):
        v = os.environ.get(var, "").strip()
        if v:
            return v
    return ""


def _creds_path() -> str:
    for var in ("KAIZER_KEYMINT_CREDENTIALS", "GOOGLE_APPLICATION_CREDENTIALS",
                "KAIZER_VERTEX_CREDENTIALS"):
        v = os.environ.get(var, "").strip()
        if v:
            return v
    return ""


def is_configured() -> bool:
    """True when a readable service-account json AND a project id are present.
    Never raises."""
    try:
        cp = _creds_path()
        return bool(cp) and os.path.isfile(cp) and bool(resolve_project())
    except Exception:
        return False


def _apikeys_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    creds = service_account.Credentials.from_service_account_file(
        _creds_path(), scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return build("apikeys", "v2", credentials=creds, cache_discovery=False)


# ─── row bookkeeping ─────────────────────────────────────────────────

def create_pending_rows(db, user_id: int, only: Optional[list[str]] = None) -> list[str]:
    """Upsert GoogleMintedKey rows to status='pending' for env names not
    already 'active'. Commits in the caller's session so the admin sees
    'minting' immediately. Returns the env names set pending."""
    import models
    names = [n for n in KEY_SPECS if (only is None or n in only)]
    pending: list[str] = []
    for env_name in names:
        row = (db.query(models.GoogleMintedKey)
                 .filter(models.GoogleMintedKey.user_id == user_id,
                         models.GoogleMintedKey.env_name == env_name).first())
        if row and row.status == "active":
            continue
        if not row:
            row = models.GoogleMintedKey(user_id=user_id, env_name=env_name)
            db.add(row)
        row.status = "pending"
        row.error = ""
        pending.append(env_name)
    if pending:
        db.commit()
    return pending


# ─── mint ────────────────────────────────────────────────────────────

def mint_user_keys(user_id: int, email: str,
                   only: Optional[list[str]] = None) -> list[dict]:
    """Worker body — owns its own SessionLocal and NEVER raises.

    For each env in KEY_SPECS (filtered by `only`) whose row isn't already
    'active': create an API-restricted key, poll the create LRO, fetch the
    key string, store it as a per-user ManagedApiKey (Fernet), and flip the
    row to 'active'. Per-key failures are isolated. Returns a list of
    {env_name, status, error}."""
    import crypto
    import models
    from database import SessionLocal

    results: list[dict] = []
    project = resolve_project()
    db = SessionLocal()
    try:
        names = [n for n in KEY_SPECS if (only is None or n in only)]
        for env_name in names:
            spec = KEY_SPECS[env_name]
            row = (db.query(models.GoogleMintedKey)
                     .filter(models.GoogleMintedKey.user_id == user_id,
                             models.GoogleMintedKey.env_name == env_name).first())
            if row and row.status == "active":
                results.append({"env_name": env_name, "status": "active", "error": ""})
                continue
            if not row:
                row = models.GoogleMintedKey(user_id=user_id, env_name=env_name,
                                             status="pending")
                db.add(row); db.commit(); db.refresh(row)

            try:
                svc = _apikeys_service()
                key_id = f"kaizer-u{user_id}-{spec['slug']}-{secrets.token_hex(3)}"
                display = f"kaizer u{user_id} {spec['slug']} {email}"[:63]
                op = svc.projects().locations().keys().create(
                    parent=f"projects/{project}/locations/global",
                    keyId=key_id,
                    body={
                        "displayName": display,
                        "annotations": {
                            "kaizer_user_id": str(user_id),
                            "kaizer_env": env_name,
                        },
                        "restrictions": {
                            "apiTargets": [{"service": spec["service"]}],
                        },
                    },
                ).execute()

                # keys.create is long-running — poll operations.get.
                deadline = time.monotonic() + _CREATE_TIMEOUT_S
                op_name = op.get("name", "")
                while not op.get("done") and op_name:
                    if time.monotonic() > deadline:
                        raise TimeoutError("create operation timed out")
                    time.sleep(_CREATE_POLL_S)
                    op = svc.operations().get(name=op_name).execute()
                if op.get("error"):
                    raise RuntimeError(str(op["error"])[:300])

                key = op.get("response") or {}
                resource_name = key.get("name", "")
                uid = key.get("uid", "")
                # The key string is fetched via a dedicated call (don't rely on
                # it being present in the LRO response).
                ks = svc.projects().locations().keys().getKeyString(
                    name=resource_name).execute().get("keyString", "")
                if not ks:
                    raise RuntimeError("empty key string returned")

                # Store the secret as a per-user ManagedApiKey row (overwrite).
                mk = (db.query(models.ManagedApiKey)
                        .filter(models.ManagedApiKey.user_id == user_id,
                                models.ManagedApiKey.name == env_name).first())
                if mk:
                    mk.value_enc = crypto.encrypt(ks)
                else:
                    db.add(models.ManagedApiKey(user_id=user_id, name=env_name,
                                                value_enc=crypto.encrypt(ks)))
                row.key_resource_name = resource_name
                row.key_uid = uid
                row.display_name = display
                row.status = "active"
                row.error = ""
                db.commit()
                results.append({"env_name": env_name, "status": "active", "error": ""})
            except Exception as exc:  # isolate per-key
                db.rollback()
                msg = str(exc)[:500]
                try:
                    row.status = "failed"
                    row.error = msg
                    db.commit()
                except Exception:
                    db.rollback()
                log.warning("mint failed user=%s env=%s: %s", user_id, env_name, msg)
                results.append({"env_name": env_name, "status": "failed", "error": msg})
    finally:
        db.close()
    return results


def start_mint_thread(user_id: int, email: str,
                      only: Optional[list[str]] = None) -> bool:
    """Fire-and-forget daemon thread guarded by _inflight so a double-click
    can't double-mint. Returns False if already in flight or not configured."""
    if not is_configured():
        return False
    with _mint_lock:
        if user_id in _inflight:
            return False
        _inflight.add(user_id)

    def _run():
        try:
            mint_user_keys(user_id, email, only)
        finally:
            with _mint_lock:
                _inflight.discard(user_id)

    threading.Thread(target=_run, name=f"keymint-u{user_id}", daemon=True).start()
    return True


def is_minting(user_id: int) -> bool:
    with _mint_lock:
        return user_id in _inflight


# ─── revoke ──────────────────────────────────────────────────────────

def revoke_user_keys(user_id: int,
                     only: Optional[list[str]] = None) -> list[dict]:
    """SYNCHRONOUS revoke. Deletes the Google key (404 = already gone =
    success) and, on success, removes the per-user ManagedApiKey row so the
    user falls back to the shared bundle at next sign-in. Google failure
    keeps the row + records the error for retry. Returns per-key results."""
    import models
    from database import SessionLocal

    results: list[dict] = []
    db = SessionLocal()
    try:
        q = (db.query(models.GoogleMintedKey)
               .filter(models.GoogleMintedKey.user_id == user_id))
        if only is not None:
            q = q.filter(models.GoogleMintedKey.env_name.in_(only))
        rows = q.all()
        for row in rows:
            if row.status == "revoked":
                results.append({"env_name": row.env_name, "revoked": True, "error": ""})
                continue
            err = ""
            ok = True
            if (row.key_resource_name or "").strip():
                try:
                    svc = _apikeys_service()
                    svc.projects().locations().keys().delete(
                        name=row.key_resource_name).execute()
                except Exception as exc:
                    # 404 / NOT_FOUND = already deleted = success.
                    txt = str(exc)
                    if "404" in txt or "NOT_FOUND" in txt.upper():
                        ok = True
                    else:
                        ok = False
                        err = txt[:500]
            if ok:
                row.status = "revoked"
                row.revoked_at = datetime.now(timezone.utc)
                row.error = ""
                mk = (db.query(models.ManagedApiKey)
                        .filter(models.ManagedApiKey.user_id == user_id,
                                models.ManagedApiKey.name == row.env_name).first())
                if mk:
                    db.delete(mk)
                db.commit()
            else:
                row.error = err
                db.commit()
                log.warning("revoke failed user=%s env=%s: %s",
                            user_id, row.env_name, err)
            results.append({"env_name": row.env_name, "revoked": ok, "error": err})
    finally:
        db.close()
    return results


# ─── startup janitor ─────────────────────────────────────────────────

def sweep_stale_pending(max_age_min: int = 15) -> int:
    """Rows stuck 'pending' older than max_age_min (server restarted
    mid-mint) → 'failed'. Called once at startup. Never raises."""
    import models
    from database import SessionLocal
    from datetime import timedelta

    n = 0
    db = SessionLocal()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_min)
        rows = (db.query(models.GoogleMintedKey)
                  .filter(models.GoogleMintedKey.status == "pending",
                          models.GoogleMintedKey.updated_at < cutoff).all())
        for row in rows:
            row.status = "failed"
            row.error = "interrupted by restart"
            n += 1
        if n:
            db.commit()
    except Exception as exc:
        db.rollback()
        log.warning("sweep_stale_pending skipped: %s", exc)
    finally:
        db.close()
    return n


# ─── shared-bundle uid labelling (optional nicety) ───────────────────

def lookup_shared_key_uids() -> dict[str, str]:
    """For each default-bundle (user_id NULL) ManagedApiKey named in
    KEY_SPECS, resolve its Google key uid so the usage view can label the
    shared bundle's traffic 'Shared bundle' instead of 'Unattributed'.
    Returns {"apikey:<uid>": env_name}. Module-cached, soft-fail → {}."""
    global _shared_uids_cache, _shared_uids_ts
    now = time.monotonic()
    if _shared_uids_cache and (now - _shared_uids_ts) < _SHARED_TTL_S:
        return _shared_uids_cache
    out: dict[str, str] = {}
    if not is_configured():
        return out
    try:
        import crypto
        import models
        from database import SessionLocal
        db = SessionLocal()
        try:
            rows = (db.query(models.ManagedApiKey)
                      .filter(models.ManagedApiKey.user_id.is_(None),
                              models.ManagedApiKey.name.in_(list(KEY_SPECS))).all())
            vals = []
            for r in rows:
                try:
                    vals.append((r.name, crypto.decrypt(r.value_enc)))
                except Exception:
                    continue
        finally:
            db.close()
        if not vals:
            return out
        svc = _apikeys_service()
        for env_name, key_string in vals:
            try:
                looked = svc.keys().lookupKey(keyString=key_string).execute()
                name = looked.get("name", "")
                if not name:
                    continue
                key = svc.projects().locations().keys().get(name=name).execute()
                uid = key.get("uid", "")
                if uid:
                    out[f"apikey:{uid}"] = env_name
            except Exception:
                continue
        _shared_uids_cache = out
        _shared_uids_ts = now
    except Exception as exc:
        log.debug("lookup_shared_key_uids soft-fail: %s", exc)
    return out


__all__ = [
    "KEY_SPECS", "is_configured", "resolve_project",
    "create_pending_rows", "mint_user_keys", "start_mint_thread", "is_minting",
    "revoke_user_keys", "sweep_stale_pending", "lookup_shared_key_uids",
]
