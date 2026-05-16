"""SEO validation for Live Studio streams.

Two paths:

  - **User-typed SEO** (``seo_source = "user"``): trusted as-is. We
    light-touch sanitize (strip control chars, hard-cap at YouTube's
    limits) but don't reject. The user owns brand voice; if they want
    a 1-char title we let it through.

  - **AI-generated SEO** (``seo_source = "ai"``): forced through the
    full validator. AI output is brand-consistent only if Kaizer's
    schema bites. Reject + retry if any rule is broken.

YouTube's hard caps (from the Data API v3 docs):
  - title:        ≤ 100 chars
  - description:  ≤ 5000 chars
  - tags:         each ≤ 100 chars; combined (incl. delimiters)
                  ≤ 500 chars; up to ~30 tags effectively
"""
from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ── Hard caps (server-enforced) ─────────────────────────────────

TITLE_MAX_CHARS   = 100
DESC_MAX_CHARS    = 5000
TAG_MAX_CHARS     = 100
TAGS_TOTAL_CHARS  = 500    # YouTube's combined limit
TAGS_MAX_COUNT    = 30


class LiveSeoIn(BaseModel):
    """Input shape — what the frontend POSTs."""
    title:       str
    description: str = ""
    tags:        list[str] = Field(default_factory=list)
    privacy:     str = "unlisted"     # public | unlisted | private
    made_for_kids: bool = False


class LiveSeoOut(BaseModel):
    """Sanitized + validated SEO."""
    title:         str
    description:   str
    tags:          list[str]
    privacy:       str
    made_for_kids: bool


# ── Sanitizers ───────────────────────────────────────────────────

_CTRL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def _strip_ctrl(s: str) -> str:
    """Remove ASCII control characters (newlines + tabs preserved)."""
    return _CTRL_CHARS.sub("", s or "")


def _sanitize_tag(t: str) -> str:
    """One tag — strip surrounding whitespace + #, ban < > "."""
    t = (t or "").strip()
    while t.startswith("#"):
        t = t[1:].strip()
    return t.replace("<", "").replace(">", "").replace('"', "")[:TAG_MAX_CHARS]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        k = it.lower()
        if k in seen or not it:
            continue
        seen.add(k)
        out.append(it)
    return out


# ── Public entry points ─────────────────────────────────────────

def sanitize_for_user_path(payload: LiveSeoIn) -> LiveSeoOut:
    """User-typed path. Hard-cap to YouTube limits, strip control
    chars, but DO NOT reject for missing description / sparse tags.
    The user's voice survives."""
    title = _strip_ctrl(payload.title).strip()[:TITLE_MAX_CHARS]
    desc  = _strip_ctrl(payload.description)[:DESC_MAX_CHARS]

    raw_tags = [_sanitize_tag(t) for t in (payload.tags or [])]
    raw_tags = [t for t in raw_tags if t]
    raw_tags = _dedupe(raw_tags)[:TAGS_MAX_COUNT]
    # Trim to TAGS_TOTAL_CHARS combined budget. YouTube counts each
    # tag plus a separating quote/comma when a tag has spaces.
    combined = 0
    fit: list[str] = []
    for t in raw_tags:
        cost = len(t) + 2 + (2 if (" " in t or "," in t) else 0)
        if combined + cost > TAGS_TOTAL_CHARS:
            break
        fit.append(t)
        combined += cost

    privacy = (payload.privacy or "unlisted").strip().lower()
    if privacy not in ("public", "unlisted", "private"):
        privacy = "unlisted"

    return LiveSeoOut(
        title=title or "Live broadcast",
        description=desc,
        tags=fit,
        privacy=privacy,
        made_for_kids=bool(payload.made_for_kids),
    )


def validate_ai_path(payload: LiveSeoIn) -> LiveSeoOut:
    """AI path — same sanitize PLUS hard validations. Raises
    ``ValueError`` (router translates to 400) when:
      - title is empty after strip
      - title > 100 chars BEFORE truncation (AI must respect the cap)
      - description > 5000 chars
      - any individual tag > 100 chars
      - combined tags > 500 chars
      - tags count > 30
      - privacy is invalid

    These rules exist because AI-generated content is unreviewed and
    we want to catch drift before it ships to YouTube.
    """
    if not (payload.title or "").strip():
        raise ValueError("AI SEO produced an empty title")
    if len(payload.title) > TITLE_MAX_CHARS:
        raise ValueError(
            f"AI title is {len(payload.title)} chars (limit {TITLE_MAX_CHARS})"
        )
    if len(payload.description or "") > DESC_MAX_CHARS:
        raise ValueError(
            f"AI description is {len(payload.description or '')} chars "
            f"(limit {DESC_MAX_CHARS})"
        )
    if len(payload.tags or []) > TAGS_MAX_COUNT:
        raise ValueError(
            f"AI returned {len(payload.tags)} tags (limit {TAGS_MAX_COUNT})"
        )
    for t in (payload.tags or []):
        if len(t) > TAG_MAX_CHARS:
            raise ValueError(f"AI tag exceeds {TAG_MAX_CHARS} chars: {t[:40]}…")
    combined = sum(
        len(t) + 2 + (2 if (" " in t or "," in t) else 0)
        for t in (payload.tags or [])
    )
    if combined > TAGS_TOTAL_CHARS:
        raise ValueError(
            f"AI tag list is {combined} chars combined (limit {TAGS_TOTAL_CHARS})"
        )
    privacy = (payload.privacy or "unlisted").strip().lower()
    if privacy not in ("public", "unlisted", "private"):
        raise ValueError(f"invalid privacy: {payload.privacy!r}")

    # Passed all checks — run through sanitizer for consistency
    # (catches edge-case control chars even if length is OK).
    return sanitize_for_user_path(payload)
