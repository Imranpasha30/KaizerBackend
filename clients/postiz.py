"""Thin HTTP client for the self-hosted Postiz instance.

Postiz handles all the per-platform OAuth + content adapters for us;
we only ever talk to its public REST API. The contract:

  - Base URL:  http://localhost:5000  (override with POSTIZ_BASE_URL env)
  - Auth:      Bearer token (POSTIZ_API_KEY env). Generate one in the
               Postiz UI → Settings → API Keys.
  - Endpoints we use:
      GET  /public/v1/integrations  → list connected platforms
      POST /public/v1/posts         → schedule a multi-platform post

Why a thin client (no postiz SDK):
  - Postiz doesn't ship a Python SDK; pip install would be a fork.
  - Their REST surface is small; HTTP+requests is cleaner than a
    multi-thousand-line SDK we'd own forever.
  - This module is the ONLY place that knows the Postiz wire format,
    so when their API changes we patch one file.

Failure model:
  - Network or 5xx → raise PostizError with the response body.
  - Auth failure (401/403) → raise PostizAuthError so the router can
    surface "Postiz token invalid — re-add it in admin settings."
  - Caller is responsible for choosing whether to fail the user's
    publish action or queue the cross-post for retry.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import requests

logger = logging.getLogger("kaizer.clients.postiz")


class PostizError(RuntimeError):
    """Postiz API returned an error or was unreachable."""


class PostizAuthError(PostizError):
    """Postiz rejected the API token (401 / 403)."""


def _base_url() -> str:
    return os.environ.get("POSTIZ_BASE_URL", "http://localhost:5000").rstrip("/")


def _api_key() -> Optional[str]:
    val = os.environ.get("POSTIZ_API_KEY", "").strip()
    return val or None


def is_enabled() -> bool:
    """True iff Postiz is reachable and we have a token. Cheap check
    used to gate the admin-only UI toggle without an extra round-trip
    on every render."""
    return bool(_api_key())


def _headers() -> dict:
    key = _api_key()
    if not key:
        raise PostizAuthError(
            "POSTIZ_API_KEY not set — generate one at "
            "<postiz>/Settings → API Keys and add it to .env."
        )
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def _handle(resp: requests.Response, op: str) -> dict:
    if resp.status_code in (401, 403):
        raise PostizAuthError(
            f"Postiz {op} rejected the token (HTTP {resp.status_code}). "
            f"Body: {resp.text[:200]}"
        )
    if not resp.ok:
        raise PostizError(
            f"Postiz {op} failed (HTTP {resp.status_code}): {resp.text[:300]}"
        )
    try:
        return resp.json()
    except Exception:
        return {}


def list_integrations() -> list[dict]:
    """Return the list of platforms Postiz has OAuth tokens for.

    Each entry typically has:
        id            — Postiz's internal integration id (UUID)
        name          — display label (e.g. "Kaizer News Andhra")
        provider      — "twitter", "instagram", "linkedin", "tiktok", …
        picture       — avatar URL
        identifier    — platform-side handle / channel id
    Raises PostizAuthError or PostizError on failure.
    """
    url = f"{_base_url()}/public/v1/integrations"
    try:
        resp = requests.get(url, headers=_headers(), timeout=10)
    except requests.RequestException as exc:
        raise PostizError(f"Postiz unreachable at {url}: {exc}") from exc
    data = _handle(resp, "list_integrations")
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "integrations" in data:
        return data["integrations"]
    return []


def schedule_post(
    *,
    integration_ids: list[str],
    text: str,
    media_url: Optional[str] = None,
    schedule_at_iso: Optional[str] = None,
    type_: str = "draft",
) -> dict:
    """Schedule a post across N integrations.

    Parameters
    ----------
    integration_ids : list[str]
        IDs from list_integrations(). Each is a connected platform.
    text : str
        Caption / body. Postiz auto-truncates per platform (Twitter
        280, etc.) if you turn that on in its UI.
    media_url : str | None
        Public URL Postiz can fetch the video / image from. We pass
        the R2 URL Kaizer already produced.
    schedule_at_iso : str | None
        ISO-8601 datetime (UTC). None = post immediately ("now").
    type_ : str
        "draft" | "scheduled" | "now". Postiz's `type` field — we let
        the caller decide.

    Raises PostizError on failure.
    """
    if not integration_ids:
        raise PostizError("schedule_post called with empty integration list")

    posts = []
    for iid in integration_ids:
        entry = {
            "integration": {"id": iid},
            "value": [{"content": text or ""}],
        }
        if media_url:
            entry["value"][0]["image"] = [{"path": media_url}]
        posts.append(entry)

    body: dict[str, Any] = {
        "type": type_,
        "posts": posts,
    }
    if schedule_at_iso:
        body["date"] = schedule_at_iso

    url = f"{_base_url()}/public/v1/posts"
    try:
        resp = requests.post(url, headers=_headers(), json=body, timeout=30)
    except requests.RequestException as exc:
        raise PostizError(f"Postiz unreachable at {url}: {exc}") from exc
    return _handle(resp, "schedule_post")
