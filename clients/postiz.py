"""Thin HTTP client for the self-hosted Postiz instance.

Postiz handles all the per-platform OAuth + content adapters for us;
we only ever talk to its public REST API. The contract:

  - Base URL:  http://localhost:5000  (override with POSTIZ_BASE_URL env)
  - Auth:      Bearer token (POSTIZ_API_KEY env). Generate one in the
               Postiz UI → Settings → API Keys.
  - Endpoints we use:
      GET  /public/v1/integrations  → list connected platforms
      POST /public/v1/upload        → upload media, returns {id, path, ...}
      POST /public/v1/posts         → schedule a multi-platform post

Media flow: Postiz post bodies reference media by INTERNAL ID, not
URL — so videos/images must be uploaded via /public/v1/upload first
to mint an id, then attached to the post via ``image: [{id: ...}]``.

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
    # Postiz Cloud's public API expects the raw token in the
    # `Authorization` header (no `Bearer ` prefix). Sending it with
    # the prefix returns 401 "Invalid API key" — verified empirically
    # against /public/v1/integrations 2026-05-07. Other auth schemes
    # (X-API-Key, apikey, etc.) return 401 "No API Key found".
    return {
        "Authorization": key,
        "Content-Type":  "application/json",
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
    raw = data if isinstance(data, list) else (
        data.get("integrations", []) if isinstance(data, dict) else []
    )
    # Normalise — Postiz Cloud's response uses ``identifier`` for the
    # platform name (e.g. "youtube"), and never exposes the YouTube
    # channel id directly. We expose a ``provider`` alias so callers
    # can filter by platform without caring about the wire-format
    # quirks. ``profile`` holds the @handle (with a Postiz-side
    # random suffix like ``-y5p``).
    out: list[dict] = []
    for i in raw:
        item = dict(i)
        item.setdefault("provider", item.get("identifier", ""))
        out.append(item)
    return out


def upload_file(local_path: str, mime_type: Optional[str] = None) -> dict:
    """Upload a local media file to Postiz, return the metadata dict.

    Postiz post bodies attach media by INTERNAL id, not URL — every
    image/video must first be POSTed here to mint an id. The returned
    dict has at minimum ``id`` (UUID), ``name`` (Postiz-renamed),
    ``path`` (public CDN URL on uploads.postiz.com).

    Verified shape against api.postiz.com 2026-05-07:
        {"id":"…","name":"…","originalName":null,
         "path":"https://uploads.postiz.com/…",
         "thumbnail":null,"alt":null}
    """
    import mimetypes, os as _os
    if not _os.path.isfile(local_path):
        raise PostizError(f"upload_file: {local_path!r} does not exist")
    if not mime_type:
        mime_type = mimetypes.guess_type(local_path)[0] or "application/octet-stream"
    url = f"{_base_url()}/public/v1/upload"
    # _headers() forces Content-Type: application/json which breaks
    # multipart — strip it and pass only the auth header.
    auth_only = {"Authorization": _headers()["Authorization"]}
    fname = _os.path.basename(local_path)
    try:
        with open(local_path, "rb") as fh:
            resp = requests.post(
                url,
                headers=auth_only,
                files={"file": (fname, fh, mime_type)},
                timeout=300,  # video bytes can be large
            )
    except requests.RequestException as exc:
        raise PostizError(f"Postiz upload unreachable at {url}: {exc}") from exc
    return _handle(resp, "upload_file")


def schedule_post(
    *,
    integration_ids: list[str],
    text: str,
    media_id: Optional[str] = None,
    media_path: Optional[str] = None,
    schedule_at_iso: Optional[str] = None,
    type_: str = "draft",
    yt_title: Optional[str] = None,
    yt_privacy: str = "public",
    yt_tags: Optional[list[str]] = None,
    yt_made_for_kids: bool = False,
) -> dict:
    """Schedule a post across N integrations.

    Parameters
    ----------
    integration_ids : list[str]
        IDs from list_integrations(). Each is a connected platform.
    text : str
        Caption / body. For YouTube targets this is the description.
        Postiz auto-truncates per platform (Twitter 280, etc.) if
        you turn that on in its UI.
    media_id : str | None
        Postiz file id from upload_file(). Required for posts with media.
    media_path : str | None
        Postiz CDN URL from upload_file()['path']. Sent alongside id —
        Postiz validation requires both fields on each image entry.
    schedule_at_iso : str | None
        ISO-8601 datetime (UTC). None = post immediately ("now").
    type_ : str
        "draft" | "scheduled" | "now". Postiz's `type` field — we let
        the caller decide.
    yt_title : str | None
        YouTube video title (≤100 chars). Required when targeting a
        YouTube integration; Postiz validates this server-side as
        ``posts.0.settings.title``. We auto-truncate from ``text`` if
        the caller doesn't pass one explicitly.
    yt_privacy : str
        YouTube privacy: 'public' | 'private' | 'unlisted'. Required
        per Postiz validation as ``posts.0.settings.type``.
    yt_tags : list[str] | None
        YouTube tags. Sanitised by the caller (no '<', '>', '"', etc.).
        Wire format: ``[{value, label}, ...]`` — matches what the
        Postiz dashboard sends from its own UI.
    yt_made_for_kids : bool
        "Self-declared made for kids" toggle (COPPA). Wire format
        is the string "yes" / "no", which is what Postiz expects in
        ``settings.selfDeclaredMadeForKids``.

    Raises PostizError on failure.
    """
    if not integration_ids:
        raise PostizError("schedule_post called with empty integration list")

    # Default the YT title from the first line of the post body if the
    # caller didn't provide one. Postiz hard-caps at 100 chars.
    if not yt_title:
        first_line = (text or "").splitlines()[0] if text else ""
        yt_title = first_line[:100] or "Untitled"
    yt_title = yt_title[:100]

    # YouTube tags wire format: list of {value, label} objects, same
    # shape the Postiz UI emits. Empty list means "no tags" which YT
    # accepts. Cap at 500 chars combined, individual tag at 100 — we
    # mirror the cap the native uploader's sanitize_tags() applies so
    # the two paths produce identical metadata.
    tag_objs: list[dict] = []
    if yt_tags:
        combined = 0
        seen: set[str] = set()
        for raw in yt_tags:
            t = str(raw).strip()
            while t.startswith("#"):
                t = t[1:].strip()
            t = t.replace("<", "").replace(">", "").replace('"', "").strip()
            if not t:
                continue
            t = t[:100]
            k = t.lower()
            if k in seen:
                continue
            est = len(t) + 2 + (2 if "," in t or " " in t else 0)
            if combined + est > 500:
                break
            seen.add(k)
            tag_objs.append({"value": t, "label": t})
            combined += est

    from datetime import datetime, timezone
    posts = []
    for iid in integration_ids:
        entry: dict[str, Any] = {
            "integration": {"id": iid},
            "value": [{"content": text or ""}],
            # Required by Postiz when the integration is YouTube.
            # Harmless on other platforms — Postiz ignores unknown
            # settings keys per provider, but always validates that
            # title/type exist when YouTube is in the post.
            "settings": {
                "title":                    yt_title,
                "type":                     yt_privacy,
                "tags":                     tag_objs,
                "selfDeclaredMadeForKids":  "yes" if yt_made_for_kids else "no",
            },
        }
        if media_id:
            # Postiz validates that every image entry has BOTH ``id``
            # (its internal file id) and ``path`` (the CDN URL). The
            # path comes from the same upload_file() response.
            img: dict[str, Any] = {"id": media_id}
            if media_path:
                img["path"] = media_path
            entry["value"][0]["image"] = [img]
        posts.append(entry)

    # Postiz Cloud's /public/v1/posts validation requires these top-
    # level fields even when scheduling "now" — without them it 400s
    # with "shortLink/date/tags should not be null or undefined".
    # date defaults to NOW so the "post immediately" semantic still
    # works without requiring callers to compute the timestamp.
    iso_now = datetime.now(timezone.utc).isoformat()
    body: dict[str, Any] = {
        "type":      type_,
        "date":      schedule_at_iso or iso_now,
        "shortLink": False,
        "tags":      [],
        "posts":     posts,
    }

    url = f"{_base_url()}/public/v1/posts"
    try:
        resp = requests.post(url, headers=_headers(), json=body, timeout=30)
    except requests.RequestException as exc:
        raise PostizError(f"Postiz unreachable at {url}: {exc}") from exc
    return _handle(resp, "schedule_post")
