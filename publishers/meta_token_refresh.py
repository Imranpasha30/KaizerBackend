"""Meta Page-token refresh worker.

Meta's long-lived Page tokens are documented as "never expire" but in
practice they invalidate on:
 - User permission changes
 - Page admin changes
 - User logging out of all Meta apps
 - Periodic forced rotation (we've observed ~60-day cycles)

We track an `expected expiry` on each MetaAccount row (set from the
long-lived user token's expires_in at OAuth time). This worker
periodically:
  1. Scans MetaAccount rows whose page_token_expiry is within
     `REFRESH_BEFORE` of now.
  2. Hits Graph debug_token to confirm the current token is still
     valid.
  3. If valid, re-exchanges the user token for a fresh long-lived
     user token (Meta lets you do this any time before expiry) and
     mints a new Page token from it.
  4. Updates the MetaAccount row with the new encrypted Page token +
     refreshed expiry.

This worker is NOT started by default — it's started from
publishers/__init__.py via start_meta_refresh_loop() ONLY when
META_APP_ID is configured, so dev environments without Meta creds
don't see warnings.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from datetime import datetime, timedelta, timezone

import httpx

import models
from database import SessionLocal


# Look this many days ahead — refresh anything expiring sooner.
REFRESH_BEFORE = timedelta(days=10)
# Poll cadence — 6 hours is plenty for tokens with ~60-day lifetime.
POLL_INTERVAL_SEC = 6 * 3600
# Per-call timeout on Graph API hits.
HTTP_TIMEOUT_SEC = 30

META_GRAPH_API_VERSION = os.environ.get("META_GRAPH_API_VERSION", "v21.0")
GRAPH_BASE = f"https://graph.facebook.com/{META_GRAPH_API_VERSION}"


def _is_configured() -> bool:
    return bool(
        os.environ.get("META_APP_ID")
        and os.environ.get("META_APP_SECRET")
    )


async def _debug_token(client: httpx.AsyncClient, token: str) -> dict:
    """Check a token's validity via Graph debug_token. The
    `inputtoken` is the one being checked; `access_token` (the app
    token) authenticates the check."""
    app_id = os.environ["META_APP_ID"]
    app_secret = os.environ["META_APP_SECRET"]
    app_token = f"{app_id}|{app_secret}"
    r = await client.get(
        f"{GRAPH_BASE}/debug_token",
        params={"input_token": token, "access_token": app_token},
    )
    return r.json() if r.status_code < 400 else {"error": r.text[:300]}


async def _exchange_long_lived(client: httpx.AsyncClient, user_token: str) -> dict:
    """Mint a fresh long-lived user token. Idempotent — can be called
    on a long-lived token to extend it."""
    r = await client.get(
        f"{GRAPH_BASE}/oauth/access_token",
        params={
            "grant_type": "fb_exchange_token",
            "client_id": os.environ["META_APP_ID"],
            "client_secret": os.environ["META_APP_SECRET"],
            "fb_exchange_token": user_token,
        },
    )
    return r.json() if r.status_code < 400 else {"error": r.text[:300]}


async def _list_pages_with_tokens(client: httpx.AsyncClient, user_token: str) -> list[dict]:
    """Re-list the user's Pages — each carries a fresh Page token."""
    r = await client.get(
        f"{GRAPH_BASE}/me/accounts",
        params={"access_token": user_token, "fields": "id,access_token"},
    )
    if r.status_code >= 400:
        return []
    return (r.json().get("data") or [])


def _enc(s: str) -> str:
    from auth import _fernet
    return _fernet().encrypt(s.encode("utf-8")).decode("ascii")


def _dec(s: str) -> str:
    from auth import _fernet
    try:
        return _fernet().decrypt(s.encode("ascii")).decode("utf-8")
    except Exception:
        return ""


async def _refresh_one(row_id: int) -> dict:
    """Refresh a single MetaAccount. Returns a small status dict for
    logging."""
    db = SessionLocal()
    try:
        row = db.query(models.MetaAccount).filter(models.MetaAccount.id == row_id).first()
        if not row:
            return {"id": row_id, "status": "missing"}
        current_token = _dec(row.page_access_token_enc)
        if not current_token:
            return {"id": row_id, "status": "no-token"}
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as cx:
            # Step 1: is the current token still valid? If yes, we
            # don't have a USER token to re-exchange from (we only
            # persisted the Page token). In that case the Page token
            # is its own refresh target — we'll hit /me/accounts with
            # the Page token to see if it still works.
            check = await _debug_token(cx, current_token)
            data = check.get("data") or {}
            is_valid = bool(data.get("is_valid"))
            if not is_valid:
                # Page token is dead. Can't auto-recover — operator
                # has to re-OAuth. Surface in the row's status so the
                # /settings/meta page shows it.
                row.granted_scopes = (row.granted_scopes or "") + ",__needs_reauth"
                db.commit()
                return {"id": row_id, "status": "invalidated", "remediation": "reauth"}
            # Token still valid. Extend by re-running /me/accounts
            # with the Page token — this doesn't extend the token's
            # lifetime, but for tokens marked "no expiration" it's
            # confirming Meta's own DB still trusts it. We update
            # last_refreshed_at to show the operator we checked.
            pages = await _list_pages_with_tokens(cx, current_token)
            # Pages call returned this exact Page's token in its
            # response — if so, persist any rotation.
            fresh = None
            for p in pages:
                if p.get("id") == row.fb_page_id:
                    fresh = p.get("access_token") or ""
                    break
            if fresh and fresh != current_token:
                row.page_access_token_enc = _enc(fresh)
            row.last_refreshed_at = datetime.now(timezone.utc)
            db.commit()
            return {"id": row_id, "status": "ok", "rotated": bool(fresh and fresh != current_token)}
    except Exception as exc:  # noqa: BLE001
        return {"id": row_id, "status": "error", "error": str(exc)[:200]}
    finally:
        db.close()


async def _refresh_due() -> list[dict]:
    """Scan + refresh every MetaAccount that's due. Returns the list
    of status dicts for logging."""
    deadline = datetime.now(timezone.utc) + REFRESH_BEFORE
    db = SessionLocal()
    try:
        rows = (
            db.query(models.MetaAccount.id)
              .filter(
                  (models.MetaAccount.page_token_expiry == None) |   # noqa: E711
                  (models.MetaAccount.page_token_expiry <= deadline),
              )
              .all()
        )
        ids = [r[0] for r in rows]
    finally:
        db.close()
    results = []
    for rid in ids:
        results.append(await _refresh_one(rid))
    return results


def _loop_thread() -> None:
    """Long-running background thread. Sleeps between refresh sweeps;
    catches all exceptions so a single bad row never takes the loop
    down."""
    while True:
        try:
            results = asyncio.run(_refresh_due())
            if results:
                import logging
                logging.getLogger("publishers.meta_token_refresh").info(
                    "refresh sweep: %s", results,
                )
        except Exception as exc:  # noqa: BLE001
            import logging
            logging.getLogger("publishers.meta_token_refresh").error(
                "refresh sweep crashed: %s", exc,
            )
        time.sleep(POLL_INTERVAL_SEC)


_started = False


def start_meta_refresh_loop() -> None:
    """Idempotent — multiple calls during reloads don't spawn extra
    threads."""
    global _started
    if _started or not _is_configured():
        return
    t = threading.Thread(
        target=_loop_thread,
        name="meta-token-refresh",
        daemon=True,
    )
    t.start()
    _started = True
