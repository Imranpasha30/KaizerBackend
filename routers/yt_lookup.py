"""Public-data YouTube channel lookup for the Style References flow.

When a user wants to add a Style Reference (e.g. "make my SEO sound
like TV9 Telugu"), pasting a handle or channel ID is way faster than
typing the metadata by hand. This router proxies the YouTube Data
API so the frontend gets a clean normalized dict with name, handle,
language hint, country, subscriber count, thumbnail, description.

Why a backend proxy (instead of calling YouTube from the browser):
  - Hides the API key (we use the same GOOGLE_API_KEY that powers
    Custom Search; no per-user OAuth needed for PUBLIC channel data).
  - Lets us extract handles / IDs from URLs server-side so the user
    can paste anything: ``@TV9Telugu``, ``UCabc...``, the full URL,
    or even free text — we run the right API call.
  - Centralised quota accounting via youtube.quota.reserve().

Endpoints:
  GET /api/yt-lookup/?q=<input>    → first matching channel + metadata
  GET /api/yt-lookup/search?q=<q>  → top 5 candidates (free-text mode)
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

import requests
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

import auth
import models
from db import get_db
from youtube import quota

logger = logging.getLogger("kaizer.routers.yt_lookup")
router = APIRouter(prefix="/api/yt-lookup", tags=["yt-lookup"])

# ─── Helpers ─────────────────────────────────────────────────────────────────

_RE_CHANNEL_ID = re.compile(r"^UC[A-Za-z0-9_-]{20,24}$")
_RE_HANDLE     = re.compile(r"^@?[A-Za-z0-9_.-]{3,30}$")


def _yt_api_key() -> str:
    key = os.environ.get("YOUTUBE_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    return key.strip()


def _classify(q: str) -> tuple[str, str]:
    """Decide which YouTube Data API call to make.

    Returns ``(kind, normalized_value)`` where ``kind`` is one of:
      - ``"id"``     → channels.list?id=…
      - ``"handle"`` → channels.list?forHandle=…
      - ``"text"``   → fallback to search.list, then channels.list
    """
    q = (q or "").strip()
    if not q:
        return "text", ""

    # Strip whitespace, leading "youtube.com/", protocol, etc.
    s = q
    for prefix in ("https://", "http://", "www."):
        if s.startswith(prefix):
            s = s[len(prefix):]
    if s.startswith("youtube.com/"):
        s = s[len("youtube.com/"):]
    if s.startswith("m.youtube.com/"):
        s = s[len("m.youtube.com/"):]
    s = s.split("?", 1)[0].split("#", 1)[0].rstrip("/")

    # /channel/UCxxxx
    if s.startswith("channel/"):
        cid = s[len("channel/"):].split("/", 1)[0]
        if _RE_CHANNEL_ID.match(cid):
            return "id", cid

    # /@handle
    if s.startswith("@"):
        handle = s.split("/", 1)[0]
        return "handle", handle

    # /c/Name (legacy custom URL — only resolvable via search)
    if s.startswith("c/") or s.startswith("user/"):
        return "text", s.split("/", 1)[1].split("/", 1)[0]

    # Plain channel ID without prefix
    if _RE_CHANNEL_ID.match(s):
        return "id", s

    # Plain handle without slash / @
    if _RE_HANDLE.match(s) and not s.startswith("UC"):
        return "handle", "@" + s.lstrip("@")

    # Anything else — free-text search
    return "text", q


def _normalize(item: dict) -> dict:
    """Flatten a YouTube channels.list item into the dict the
    frontend ChannelForm wants."""
    snip = item.get("snippet") or {}
    stats = item.get("statistics") or {}
    branding = item.get("brandingSettings", {}).get("channel", {}) or {}
    thumbs = snip.get("thumbnails") or {}
    thumb_url = (
        (thumbs.get("high") or {}).get("url")
        or (thumbs.get("medium") or {}).get("url")
        or (thumbs.get("default") or {}).get("url")
        or ""
    )

    def _int(v):
        try: return int(v or 0)
        except (ValueError, TypeError): return 0

    custom_url = (snip.get("customUrl") or "").strip()
    if custom_url and not custom_url.startswith("@"):
        custom_url = "@" + custom_url

    # Heuristic language: snippet.defaultLanguage > brandingSettings.country
    lang = (snip.get("defaultLanguage")
            or branding.get("defaultLanguage")
            or "")
    if lang and "-" in lang:
        lang = lang.split("-", 1)[0]

    return {
        "google_channel_id":  item.get("id", ""),
        "name":               snip.get("title", ""),
        "handle":             custom_url,
        "language":           (lang or "").lower(),
        "country":            snip.get("country", "") or "",
        "description":        snip.get("description", "") or "",
        "thumbnail_url":      thumb_url,
        "subscriber_count":   _int(stats.get("subscriberCount")),
        "video_count":        _int(stats.get("videoCount")),
        "view_count":         _int(stats.get("viewCount")),
        "keywords":           (branding.get("keywords") or "").strip(),
    }


# ─── YouTube Data API calls ──────────────────────────────────────────────────

_API = "https://www.googleapis.com/youtube/v3"


def _fetch_by_id(channel_id: str, key: str) -> Optional[dict]:
    r = requests.get(
        f"{_API}/channels",
        params={
            "part": "snippet,statistics,brandingSettings",
            "id":   channel_id,
            "key":  key,
        },
        timeout=10,
    )
    if not r.ok:
        return None
    items = r.json().get("items") or []
    return items[0] if items else None


def _fetch_by_handle(handle: str, key: str) -> Optional[dict]:
    if not handle.startswith("@"):
        handle = "@" + handle
    r = requests.get(
        f"{_API}/channels",
        params={
            "part":      "snippet,statistics,brandingSettings",
            "forHandle": handle,
            "key":       key,
        },
        timeout=10,
    )
    if not r.ok:
        return None
    items = r.json().get("items") or []
    return items[0] if items else None


def _search_text(q: str, key: str, limit: int = 5) -> list[str]:
    """Free-text search — returns up to N channel IDs in popularity order."""
    r = requests.get(
        f"{_API}/search",
        params={
            "part":       "snippet",
            "type":       "channel",
            "q":          q,
            "maxResults": min(limit, 25),
            "key":        key,
        },
        timeout=10,
    )
    if not r.ok:
        return []
    return [
        i.get("snippet", {}).get("channelId", "")
        for i in (r.json().get("items") or [])
        if i.get("snippet", {}).get("channelId")
    ]


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/")
def lookup(
    q: str = Query(..., min_length=2, max_length=200),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Resolve a handle / channel ID / URL / free-text query to ONE
    YouTube channel record.

    Returns the normalized dict the ChannelForm consumes, or 404 if
    no match.
    """
    key = _yt_api_key()
    if not key:
        raise HTTPException(
            status_code=503,
            detail="YOUTUBE_API_KEY (or GOOGLE_API_KEY) not set on the server.",
        )

    # Reserve quota before the call. channels.list = 1 unit;
    # search.list = 100 units, so free-text path costs more.
    kind, val = _classify(q)
    cost = 1 if kind in ("id", "handle") else 101  # search + channels
    if not quota.reserve(db, cost, api_key=key):
        raise HTTPException(
            status_code=429,
            detail="Daily YouTube Data API quota exhausted. Try again tomorrow.",
        )

    item: Optional[dict] = None
    if kind == "id":
        item = _fetch_by_id(val, key)
    elif kind == "handle":
        item = _fetch_by_handle(val, key)
    else:  # text — search then resolve top hit
        ids = _search_text(val, key, limit=1)
        if ids:
            item = _fetch_by_id(ids[0], key)

    if not item:
        raise HTTPException(status_code=404,
                            detail=f"No channel found for {q!r}")
    return _normalize(item)


@router.get("/search")
def search(
    q: str = Query(..., min_length=2, max_length=200),
    limit: int = Query(5, ge=1, le=10),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> list[dict]:
    """Free-text channel search — returns up to N candidates so the
    UI can show a picker if the input is ambiguous."""
    key = _yt_api_key()
    if not key:
        raise HTTPException(
            status_code=503,
            detail="YOUTUBE_API_KEY (or GOOGLE_API_KEY) not set on the server.",
        )
    # search.list = 100 units, channels.list = 1 unit per id.
    cost = 100 + limit
    if not quota.reserve(db, cost, api_key=key):
        raise HTTPException(
            status_code=429,
            detail="Daily YouTube Data API quota exhausted. Try again tomorrow.",
        )

    ids = _search_text(q, key, limit=limit)
    if not ids:
        return []
    # One bulk channels.list call (id=id1,id2,...) is 1 unit total.
    r = requests.get(
        f"{_API}/channels",
        params={
            "part": "snippet,statistics,brandingSettings",
            "id":   ",".join(ids),
            "key":  key,
        },
        timeout=10,
    )
    if not r.ok:
        return []
    return [_normalize(item) for item in (r.json().get("items") or [])]
