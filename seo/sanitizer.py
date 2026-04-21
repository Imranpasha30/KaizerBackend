"""Leak-prevention sanitizer.

Strips any reference to a style-source channel (name, handle, common
hashtag form) from a generated generic SEO.  Runs unconditionally on every
Gemini response — a belt-and-suspenders guarantee that no competitor brand
can land in a user's published video.

Signals cleaned:
  - `| ChannelName` suffix anywhere in title
  - Channel name occurrences in description (case-insensitive)
  - Channel handle ("@name") in description
  - Hashtags whose CamelCase body is the channel name (e.g. #TV9Telugu)
  - Keywords (tags) containing the channel name as a substring
"""
from __future__ import annotations

import re
from typing import Dict, Any, List

import models


def _name_variants(ch: models.Channel) -> List[str]:
    """All surface forms a channel's brand could appear as."""
    out: set[str] = set()
    for src in (ch.name, ch.handle or ""):
        src = (src or "").strip()
        if not src:
            continue
        out.add(src.lower())
        # Spaceless variant — e.g. "TV9 Telugu" → "tv9telugu"
        out.add(re.sub(r"\s+", "", src).lower())
        # Individual tokens ≥ 3 chars
        for tok in re.split(r"[\s_\-./,]+", src):
            if len(tok) >= 3:
                out.add(tok.lower())
    return sorted(out, key=len, reverse=True)  # longest first so replacement is greedy


def sanitize(seo: Dict[str, Any], style_source: models.Channel | None) -> Dict[str, Any]:
    """Return a copy of `seo` with any style_source branding removed.

    Pass-through when `style_source` is None — nothing to scrub against.
    """
    if not style_source:
        return seo

    patterns = _name_variants(style_source)
    if not patterns:
        return seo

    def _strip(text: str) -> str:
        if not text:
            return text
        out = text
        for p in patterns:
            # Word-boundary-ish regex — handles CJK/Devanagari by falling back
            # to literal replacement when \b doesn't apply cleanly.
            esc = re.escape(p)
            out = re.sub(rf"(?i){esc}", "", out)
        # Collapse multi-space + dangling " | "
        out = re.sub(r"\s*\|\s*\|\s*", " | ", out)
        out = re.sub(r"\s{2,}", " ", out)
        out = re.sub(r"\s*\|\s*$", "", out).strip()
        return out

    cleaned = dict(seo)
    cleaned["title"]       = _strip(seo.get("title", ""))
    cleaned["description"] = _strip(seo.get("description", ""))
    cleaned["hook"]        = _strip(seo.get("hook", ""))

    # Keywords / hashtags — drop any that contain a brand variant after strip
    def _ok(s: str) -> bool:
        low = (s or "").lower()
        return not any(p in low for p in patterns)

    cleaned["keywords"] = [k for k in (seo.get("keywords") or []) if _ok(k)]
    cleaned["hashtags"] = [h for h in (seo.get("hashtags") or []) if _ok(h)]

    return cleaned
