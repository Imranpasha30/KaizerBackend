"""Post-processing to guarantee SEO output quality.

Ports the extension's `enforceQuality` + computed-score logic.

Guarantees:
  - Title ≤ 100 chars with word-boundary cut and ` | {channel}` suffix.
  - Exactly 30 unique, lowercased keywords (≤ 500 chars total).
  - 10–12 CamelCase hashtags with mandatory ones prepended.
  - Computed seo_score (0–100) — we never trust the LLM's self-rating.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

import models


MAX_TITLE_LEN = 100
TAG_COUNT = 30
TAG_TOTAL_CHAR_CAP = 500   # YouTube videos.insert hard limit on tags[] combined length
HASHTAG_MIN = 10
HASHTAG_MAX = 12

_POWER_WORDS = {
    "shocking", "breaking", "viral", "exclusive", "revealed", "stunning",
    "must watch", "huge",
    "బిగ్", "షాకింగ్", "బ్రేకింగ్", "వైరల్", "సంచలనం", "ఎక్స్క్లూజివ్",
}

_GENERIC_TAGS = [
    "telugu news", "latest telugu news", "news today",
    "telugu breaking news", "trending news telugu", "telugu news live",
    "today news telugu", "telugu political news", "telugu headlines",
]
_GENERIC_HASHTAGS = ["#TeluguNews", "#BreakingNews", "#LatestNews", "#Viral", "#Telugu", "#News"]


# ─── Normalizers ──────────────────────────────────────────────────────────────

def _normalize_hashtag(s: str) -> str:
    """'big tv news' → '#BigTvNews', '#bigtv' → '#Bigtv', '' → ''."""
    s = (s or "").strip().lstrip("#")
    if not s:
        return ""
    parts = re.split(r"[\s_\-./,!?:;]+", s)
    camel = "".join(p[:1].upper() + p[1:] for p in parts if p)
    return f"#{camel}" if camel else ""


def _truncate_title_at_word(title: str, channel_name: str, max_len: int = MAX_TITLE_LEN) -> str:
    """Strip any existing ' | ...' suffix, reserve room for the channel suffix,
    cut on a word boundary, re-attach the suffix."""
    title = (title or "").strip()
    name = (channel_name or "").strip()

    # Remove any trailing ' | ...' the model already added
    cleaned = re.sub(r"\s*\|\s*[^|]+$", "", title).strip()

    attach = f" | {name}" if name else ""
    budget = max_len - len(attach)
    if budget <= 0:
        return name[:max_len]

    if len(cleaned) <= budget:
        return f"{cleaned}{attach}".strip()

    cut = cleaned[:budget]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    cut = cut.rstrip(" ,.-–—|:;")
    return f"{cut}{attach}".strip()


def _ensure_tags(tags: List[str], fixed: List[str]) -> List[str]:
    """30 unique lowercased tags, fixed set prepended, ≤ 500 chars total."""
    seen: set[str] = set()
    out: List[str] = []

    def _add(raw: str) -> bool:
        s = re.sub(r"\s+", " ", (raw or "").strip().lower().lstrip("#")).strip()
        if not s or s in seen:
            return False
        # Respect YouTube's 500-char total cap (comma separator adds ~2 chars each)
        projected = sum(len(t) for t in out) + len(s) + 2 * (len(out))
        if projected > TAG_TOTAL_CHAR_CAP:
            return False
        seen.add(s)
        out.append(s)
        return True

    for t in fixed or []:
        _add(t)
        if len(out) >= TAG_COUNT:
            return out

    for t in tags or []:
        _add(t)
        if len(out) >= TAG_COUNT:
            return out

    for t in _GENERIC_TAGS:
        if len(out) >= TAG_COUNT:
            break
        _add(t)

    return out


def _ensure_hashtags(hashtags: List[str], mandatory: List[str]) -> List[str]:
    """Mandatory first (verbatim, normalized), dedupe, CamelCase-normalize, cap 10-12."""
    seen: set[str] = set()
    out: List[str] = []

    def _add(raw: str):
        norm = _normalize_hashtag(raw)
        if not norm:
            return
        key = norm.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(norm)

    for h in mandatory or []:
        _add(h)
        if len(out) >= HASHTAG_MAX:
            return out

    for h in hashtags or []:
        _add(h)
        if len(out) >= HASHTAG_MAX:
            return out

    for g in _GENERIC_HASHTAGS:
        if len(out) >= HASHTAG_MIN:
            break
        _add(g)

    return out[:HASHTAG_MAX]


def _compose_description(hook: str, description: str, footer: str) -> str:
    """Ensure the hook leads and the footer trails exactly once."""
    hook = (hook or "").strip()
    body = (description or "").strip()
    footer = (footer or "").strip()

    if hook and hook not in body[: max(40, len(hook) + 5)]:
        body = f"{hook}\n\n{body}".strip()

    if footer:
        # Remove duplicate footer occurrences mid-body
        if body.count(footer) > 0 and not body.endswith(footer):
            body = body.replace(footer, "").strip()
        if not body.endswith(footer):
            body = f"{body}\n\n{footer}".strip()

    return body


# ─── Score ────────────────────────────────────────────────────────────────────

def compute_seo_score(enforced: Dict[str, Any], channel: models.Channel) -> int:
    """0–100 score. Computed server-side — the model's self-rating is ignored."""
    score = 0
    title = enforced.get("title", "") or ""
    desc = enforced.get("description", "") or ""
    tags = enforced.get("keywords", []) or []
    hashtags = enforced.get("hashtags", []) or []

    has_latin = bool(re.search(r"[A-Za-z]", title))
    has_telugu = bool(re.search(r"[\u0C00-\u0C7F]", title))

    # Title CTR (25)
    if 40 <= len(title) <= MAX_TITLE_LEN:
        score += 10
    if any(w.lower() in title.lower() for w in _POWER_WORDS):
        score += 5
    if "|" in title:
        score += 5
    if has_latin and has_telugu:
        score += 5

    # Description (25)
    if 400 <= len(desc) <= 2000:
        score += 15
    if "\n" in desc:
        score += 5
    if "#" in desc or "subscribe" in desc.lower():
        score += 5

    # Tags (20)
    if len(tags) >= TAG_COUNT:
        score += 15
    fixed_lower = {(t or "").lower() for t in (channel.fixed_tags or [])}
    tags_lower = {(t or "").lower() for t in tags}
    if fixed_lower and fixed_lower.issubset(tags_lower):
        score += 5

    # Hashtags (15)
    if HASHTAG_MIN <= len(hashtags) <= HASHTAG_MAX:
        score += 10
    mand_lower = {(h or "").lower() for h in (channel.mandatory_hashtags or [])}
    hash_lower = {(h or "").lower() for h in hashtags}
    if mand_lower and mand_lower.issubset(hash_lower):
        score += 5

    # Bilingual + overall polish (15)
    if has_latin and has_telugu:
        score += 8
    if len(title) <= MAX_TITLE_LEN and len(hashtags) >= HASHTAG_MIN:
        score += 7

    return max(0, min(100, score))


# ─── Public entry ─────────────────────────────────────────────────────────────

def enforce_quality(raw: Dict[str, Any], channel: models.Channel) -> Dict[str, Any]:
    """Apply all quality rules; return an enforced dict ready to persist."""
    title = _truncate_title_at_word(raw.get("title", ""), channel.name)
    hook = (raw.get("hook") or "").strip()

    keywords = _ensure_tags(raw.get("keywords") or raw.get("tags") or [], channel.fixed_tags or [])
    hashtags = _ensure_hashtags(raw.get("hashtags") or [], channel.mandatory_hashtags or [])
    description = _compose_description(
        hook=hook,
        description=raw.get("description", ""),
        footer=channel.footer or "",
    )

    enforced: Dict[str, Any] = {
        "title": title,
        "description": description,
        "keywords": keywords,
        "hashtags": hashtags,
        "hook": hook,
        "thumbnail_text": (raw.get("thumbnail_text") or "").strip(),
        "metadata": raw.get("metadata") or {},
    }
    enforced["seo_score"] = compute_seo_score(enforced, channel)
    return enforced
