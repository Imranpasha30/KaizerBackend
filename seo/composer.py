"""Publish-time brand-overlay composer.

Takes a channel-agnostic generic SEO (stored on clip.seo) and overlays a
specific destination channel's brand kit to produce the final metadata that
will actually be uploaded to YouTube.

The brand overlay is PURELY mechanical — no Gemini calls, no inference.
Every field injected here comes from the user's own `models.Channel` row,
which means a competitor's brand can never appear in the output by
construction.

The composed output is intentionally VERIFIABLE at publish time:
  - Title suffix is exactly ` | {channel.name}` — no other value allowed.
  - Mandatory hashtags are always first, in the exact order configured.
  - Fixed tags are always present in the keyword list.
  - Footer is always the LAST block of the description.

Publish code should always pass the composed result through
`assert_no_foreign_brand(...)` before enqueueing the upload.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

import models


MAX_TITLE_LEN = 100


# ── Utilities ────────────────────────────────────────────────────────────────

def _normalize_hashtag(s: str) -> str:
    s = (s or "").strip().lstrip("#")
    if not s:
        return ""
    parts = re.split(r"[\s_\-./,!?:;]+", s)
    camel = "".join(p[:1].upper() + p[1:] for p in parts if p)
    return f"#{camel}" if camel else ""


def _dedupe_preserve(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for i in items:
        k = (i or "").strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(i)
    return out


def _truncate_title(title: str, channel_name: str) -> str:
    """Strip any prior `| ...` suffix, reserve room for new `| Name`, word-cut."""
    clean = re.sub(r"\s*\|\s*[^|]+$", "", (title or "").strip()).strip()
    name = (channel_name or "").strip()
    attach = f" | {name}" if name else ""
    budget = MAX_TITLE_LEN - len(attach)
    if budget <= 0:
        return name[:MAX_TITLE_LEN]
    if len(clean) <= budget:
        return f"{clean}{attach}".strip()
    cut = clean[:budget]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    cut = cut.rstrip(" ,.-–—|:;")
    return f"{cut}{attach}".strip()


# ── Public entry ─────────────────────────────────────────────────────────────

def compose(
    generic_seo: Dict[str, Any],
    destination: models.Channel,
    *,
    publish_kind: str = "video",
) -> Dict[str, Any]:
    """Apply destination's brand overlay to the generic SEO.

    The title suffix uses the REAL YouTube channel title (from the linked
    OAuth token) when available — that's what viewers recognize, not the
    internal profile name.  E.g. if the user's YouTube account is "auto
    walla" but their profile row is named "Kaizer News Telugu", the title
    ends with " | auto walla", matching their actual channel brand.

    Falls back to `destination.name` when no OAuth token is linked
    (pre-publish previews on unconnected profiles).
    """
    gs = generic_seo or {}

    # Prefer the real YouTube channel title over the internal profile name
    tok = getattr(destination, "oauth_token", None)
    yt_title = ""
    if tok is not None:
        yt_title = (getattr(tok, "google_channel_title", "") or "").strip()
    brand_name = yt_title or destination.name

    title = _truncate_title(gs.get("title", ""), brand_name)

    # Keywords: mandatory/fixed first, then generic topic tags; cap 30
    fixed = [t for t in (destination.fixed_tags or []) if (t or "").strip()]
    merged_tags = _dedupe_preserve([*fixed, *(gs.get("keywords") or [])])[:30]

    # Hashtags: mandatory (normalized) first, then generic topic hashtags; cap 12
    mandatory = [_normalize_hashtag(h) for h in (destination.mandatory_hashtags or []) if (h or "").strip()]
    mandatory = [h for h in mandatory if h]
    generic_ht = [_normalize_hashtag(h) for h in (gs.get("hashtags") or []) if (h or "").strip()]
    generic_ht = [h for h in generic_ht if h]
    merged_ht = _dedupe_preserve([*mandatory, *generic_ht])[:12]

    # Description: ensure hook in line 1, append destination footer
    body = (gs.get("description") or "").strip()
    hook = (gs.get("hook") or "").strip()
    if hook and hook.lower()[:60] not in body.lower()[:400]:
        body = f"{hook}\n\n{body}".strip()

    footer = (destination.footer or "").strip()
    if footer:
        # Drop any prior instance so a regenerated publish doesn't pile duplicates
        body = body.replace(footer, "").strip()
        body = f"{body}\n\n{footer}".strip()

    # Shorts tag injection — if caller signalled a Shorts publish
    if publish_kind == "short":
        shorts_tag = "#Shorts"
        if shorts_tag.lower() not in title.lower():
            candidate = f"{title} {shorts_tag}"
            title = candidate if len(candidate) <= MAX_TITLE_LEN else title
            if shorts_tag.lower() not in body.lower():
                body = f"{shorts_tag}\n\n{body}".strip()
        if "shorts" not in [t.lower() for t in merged_tags]:
            merged_tags = _dedupe_preserve(["shorts", *merged_tags])[:30]

    return {
        "title":          title,
        "description":    body,
        "keywords":       merged_tags,
        "hashtags":       merged_ht,
        "hook":           hook,
        "thumbnail_text": (gs.get("thumbnail_text") or "").strip(),
        "channel_id":     destination.id,
        "channel_name":   destination.name,
        "composed_from":  gs.get("id") or gs.get("generated_at") or "generic",
    }


# Common English/Telugu/Hindi words that appear both in profile names and in
# everyday news content — "Live" in "HMTV Live" and "Live Performance" aren't
# the same signal.  Skipping these prevents false-positive leak warnings.
_GENERIC_BRAND_TOKENS = {
    "live", "news", "tv", "media", "channel", "official", "hd", "digital",
    "plus", "network", "today", "daily", "24x7", "online", "world",
    "telugu", "hindi", "tamil", "kannada", "malayalam", "bengali", "marathi",
    "gujarati", "english", "india", "bharat", "one", "two",
}


def assert_no_foreign_brand(composed: Dict[str, Any], destination: models.Channel, all_channels: List[models.Channel]) -> List[str]:
    """Final safety gate — return a list of leak warnings (empty when safe).

    Checks that NO channel's branding other than `destination`'s appears in
    the composed output.  Only flags brand-UNIQUE identifiers (the profile's
    distinctive tokens like "RTV", "TV9", "HMTV") — generic news words like
    "Live", "News", "TV" don't count as leaks even if they're part of some
    profile's name.
    """
    warnings: List[str] = []
    title = (composed.get("title") or "").lower()
    desc  = (composed.get("description") or "").lower()
    tags  = " ".join(composed.get("keywords") or []).lower()
    ht    = " ".join(composed.get("hashtags") or []).lower()
    blob  = f"{title} || {desc} || {tags} || {ht}"

    dest_id = destination.id
    dest_tokens = set()
    for src in (destination.name, destination.handle or ""):
        for tok in re.split(r"\s+", (src or "").strip().lower()):
            if len(tok) >= 3:
                dest_tokens.add(tok)
    # Also skip tokens that are the destination's real YouTube title (the
    # brand we ARE publishing as) — those are allowed to appear anywhere.
    tok_link = getattr(destination, "oauth_token", None)
    if tok_link is not None:
        for tok in re.split(r"\s+", (getattr(tok_link, "google_channel_title", "") or "").strip().lower()):
            if len(tok) >= 3:
                dest_tokens.add(tok)

    for ch in (all_channels or []):
        if ch.id == dest_id:
            continue
        for src in (ch.name, ch.handle or ""):
            src = (src or "").strip()
            if not src:
                continue
            src_low = src.lower()
            unique_toks = [
                tok for tok in re.split(r"\s+", src_low)
                if len(tok) >= 4
                and tok not in dest_tokens
                and tok not in _GENERIC_BRAND_TOKENS
            ]
            # Require at least one UNIQUE brand token to be present, not just
            # a generic word like "live".  This eliminates the false-positive
            # that fires on "Live Performance" when a profile is "HMTV Live".
            for tok in unique_toks:
                if re.search(rf"\b{re.escape(tok)}\b", blob):
                    warnings.append(f"'{tok}' (from profile '{ch.name}') appeared in output")
                    break

    return warnings
