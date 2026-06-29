"""Offline dynamic slot-filler for custom templates — "the filler bot".

Given a template's discovered slots (a :class:`~.contract.TemplateContract`) plus the REAL
content the render pipeline already has (:class:`ContentBundle`), this decides — purely,
deterministically, with NO network / DB / Playwright — what text and image goes into each
slot. The renderer then clears any slot this filler could not fill (final render), so a
template author's placeholder text ("Your headline goes here", "Supporting image", a
"...lower third" ticker) can never leak into a published frame.

Pure data in → pure data out. This module imports nothing from the app, so it is trivially
unit-testable and safe to call from any render path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# Per-slot SAFETY bounds only (generous). We do NOT cut text to a tiny limit + "…" anymore —
# that was hiding real content. The renderer auto-fits the font so the FULL text fits its
# box; these caps just stop a pathological (thousands-of-chars) input from breaking layout.
_CAP = {
    "headline": 300, "title": 300, "hook": 300, "subtitle": 300,
    "caption": 300, "body": 2000, "kicker": 60, "cta": 80,
    "ticker": 2000, "marquee": 2000, "watermark": 80,
}
# Star separator between headlines in a derived ticker (matches the newsroom convention).
TICKER_SEP = "  ★  "

# Sentence terminators incl. Indic danda (Devanagari U+0964, Bengali U+09F7) + newlines,
# so first_sentence works on Telugu / Hindi copy, not just Latin.
_SENT_SPLIT = re.compile(r"[.!?।৷॥\n\r]")

# Heuristics for "this looks like author placeholder text, not real content".
_PLACEHOLDER_HINTS = (
    "goes here", "your ", "lorem", "placeholder", "sample", "example",
    "headline here", "dummy", "replace ", "tbd", "todo", "xxxx",
    "supporting image", "image here", "text here", "lower third",
)


@dataclass
class ContentBundle:
    """All the real, already-available content the pipeline can pour into a template.
    Format-agnostic: the orchestrator (which owns the DB + paths) constructs it; the
    filler only reads it. Every field is optional with a safe default."""
    headline: str = ""            # primary headline (native language preferred)
    headline_alt: str = ""        # secondary / English headline (fallback)
    subtitle: str = ""            # one-line supporting line
    body: str = ""                # full summary / longer copy
    kicker: str = ""              # section chip ("BREAKING"); blank -> auto default
    cta: str = ""                 # call-to-action; only used if the template declares one
    ticker: str = ""              # explicit marquee text; blank -> derived from headlines
    channel_name: str = ""        # brand text
    watermark: str = ""           # usually "" at clean-render (stamped per-channel at publish)
    images: list = field(default_factory=list)   # ABSOLUTE supporting-image paths, in order
    logo_path: Optional[str] = None
    text_color: Optional[str] = None             # hex -> brand var --kaizer-text
    stories: list = field(default_factory=list)  # full-form: [{headline, headline_alt, body, ...}]


@dataclass
class SlotFill:
    """The filler's decision for one render."""
    texts: dict = field(default_factory=dict)        # slot key -> text (only filled slots)
    images: dict = field(default_factory=dict)       # slot key -> abs image path
    brand: dict = field(default_factory=dict)        # css var -> value
    unfilled_text: list = field(default_factory=list)
    unfilled_image: list = field(default_factory=list)
    report: dict = field(default_factory=dict)       # slot key -> source description


def first_sentence(text: str, max_len: int = 240) -> str:
    """First sentence of ``text`` (Indic-aware). Returns the FULL first sentence (no "…"
    truncation — the renderer auto-fits the font); only a generous word-boundary safety cut
    is applied to a pathologically long run-on sentence."""
    t = (text or "").strip()
    if not t:
        return ""
    s = (_SENT_SPLIT.split(t, 1)[0] or "").strip() or t
    if len(s) > max_len:
        s = s[:max_len].rsplit(" ", 1)[0].rstrip() or s[:max_len].rstrip()
    return s


def derive_ticker(headlines, channel_name: str = "") -> str:
    """Build a marquee string from a list of headlines joined by the star separator.
    Falls back to the channel name, then "". Centralised so built-in + custom agree."""
    hs = [h.strip() for h in (headlines or []) if h and str(h).strip()]
    if hs:
        return TICKER_SEP.join(hs)
    return (channel_name or "").strip()


def looks_like_placeholder(text: str) -> bool:
    """Heuristic: does this slot's authored inner text read as a placeholder?"""
    t = (text or "").strip().lower()
    if not t:
        return False
    if t.endswith("...") or t.endswith("…"):
        return True
    return any(h in t for h in _PLACEHOLDER_HINTS)


def _cap(key: str, val: str) -> str:
    v = (val or "").strip()
    n = _CAP.get((key or "").lower())
    if n and len(v) > n:
        v = v[:n].rsplit(" ", 1)[0].rstrip() or v[:n].rstrip()   # safety bound only, no "…"
    return v


def _first_story_headline(c: ContentBundle) -> str:
    for st in (c.stories or []):
        h = (st.get("headline") or st.get("headline_alt") or "").strip()
        if h:
            return h
    return ""


def _story_headlines(c: ContentBundle) -> list:
    out = []
    for st in (c.stories or []):
        h = (st.get("headline") or st.get("headline_alt") or "").strip()
        if h:
            out.append(h)
    if not out and c.headline.strip():
        out = [c.headline.strip()]
    return out


def _fill_text_slot(key: str, c: ContentBundle, *, kind: str):
    """Return (value, source) for a text slot. value == "" means: no mapping -> clear it.
    The mapping is a deterministic fallback chain per slot name."""
    k = (key or "").strip().lower()

    if k in ("headline", "title"):
        for f, src in ((c.headline, "headline"), (c.headline_alt, "headline_alt"),
                       (_first_story_headline(c), "story headline"), (c.channel_name, "channel_name")):
            if f and f.strip():
                return f, src
        return "", "BLANK"

    if k == "hook":
        if c.subtitle.strip():
            return c.subtitle, "subtitle"
        fs = first_sentence(c.body)
        if fs:
            return fs, "summary->hook"
        if c.headline.strip():
            return c.headline, "headline"
        return "", "BLANK"

    if k == "subtitle":
        if c.subtitle.strip():
            return c.subtitle, "subtitle"
        fs = first_sentence(c.body, 90)
        if fs:
            return fs, "summary->subtitle"
        return "", "BLANK"

    if k == "kicker":
        if c.kicker.strip():
            return c.kicker, "kicker"
        return ("BREAKING" if kind == "full" else "NEWS"), "auto label"

    if k == "caption":
        fs = first_sentence(c.body, 120)
        if fs:
            return fs, "summary->caption"
        if c.subtitle.strip():
            return c.subtitle, "subtitle"
        if c.headline.strip():
            return c.headline, "headline"
        return "", "BLANK"

    if k == "cta":
        if c.cta.strip():
            return c.cta, "cta"
        # Only derive copy where the author asked for a CTA slot (we're filling one now).
        return ("Watch now" if kind == "short" else ""), ("derived cta" if kind == "short" else "BLANK")

    if k == "body":
        if c.body.strip():
            return c.body, "body"
        if c.subtitle.strip():
            return c.subtitle, "subtitle"
        if c.headline.strip():
            return c.headline, "headline"
        return "", "BLANK"

    if k in ("ticker", "marquee"):
        if c.ticker.strip():
            return c.ticker, "ticker"
        t = derive_ticker(_story_headlines(c), c.channel_name)
        if t:
            return t, "derived ticker"
        return ("BREAKING NEWS" if kind == "full" else "NEWS"), "auto label"

    if k == "watermark":
        if c.watermark.strip():
            return c.watermark, "watermark"
        return "", "BLANK (stamped at publish)"

    # Unknown text:NAME slot — best effort so it isn't left as author placeholder.
    if c.headline.strip():
        return c.headline, "headline (fallback)"
    if c.channel_name.strip():
        return c.channel_name, "channel_name (fallback)"
    return "", "BLANK"


def build_slot_fill(contract, content: ContentBundle, *, kind: str = "short",
                    reuse_images: bool = True) -> SlotFill:
    """Map ``contract``'s slots onto real ``content``. Returns a :class:`SlotFill`.

    Text slots use the per-name fallback chain (``_fill_text_slot``); image slots are
    assigned POSITIONALLY in document order (slot i -> content.images[i], reusing the last
    image for surplus slots when ``reuse_images``). Any slot with no real content is listed
    in ``unfilled_*`` (the renderer hides those so no placeholder leaks)."""
    sf = SlotFill()

    for s in contract.text_slots:
        val, src = _fill_text_slot(s.key, content, kind=kind)
        val = _cap(s.key, val)
        if val:
            sf.texts[s.key] = val
        else:
            sf.unfilled_text.append(s.key)
        sf.report[s.key] = src

    imgs = [p for p in (content.images or []) if p]
    for i, s in enumerate(contract.image_slots):
        chosen = None
        if i < len(imgs):
            chosen = imgs[i]
        elif imgs and reuse_images:
            chosen = imgs[-1]
        if chosen:
            sf.images[s.key] = chosen
            sf.report[s.key] = f"image #{min(i, len(imgs) - 1) + 1}"
        else:
            sf.unfilled_image.append(s.key)
            sf.report[s.key] = "BLANK"

    if content.text_color:
        sf.brand["--kaizer-text"] = content.text_color

    return sf


# ---- upload-time audit (content-free projection: works before any job exists) ----

# Phrased to read naturally after both "auto-filled from {…}" (upload warning) and
# "{slot} ← {…}" (preview) — so no awkward "from derived from".
_TEXT_SOURCE_DESC = {
    "headline": "the story headline", "title": "the story headline",
    "hook": "the story summary", "subtitle": "the story summary",
    "kicker": "an auto label (BREAKING / NEWS)", "caption": "the story summary",
    "cta": "your call-to-action", "body": "the story summary",
    "ticker": "the story headlines", "marquee": "the story headlines",
    "watermark": "your channel watermark (added at publish)",
}


def text_source_desc(key: str) -> str:
    return _TEXT_SOURCE_DESC.get((key or "").strip().lower(), "story headline (fallback)")


def audit_template(contract) -> list:
    """A per-slot projection of WHAT will fill each slot (computed without any job/content),
    plus whether the slot's authored inner text looks like a placeholder. Stored on the
    template so the upload response + picker can show "headline ← story headline", etc."""
    audit = []
    img_n = 0
    for s in contract.slots:
        placeholder = (getattr(s, "placeholder", "") or "").strip()
        entry = {
            "key": s.key, "kind": s.kind, "name": s.name, "tag": s.tag,
            "placeholder": placeholder[:160],
            "is_placeholder": looks_like_placeholder(placeholder),
            "fill_source": "",
            "blank_in_final": False,
        }
        if s.kind == "text":
            entry["fill_source"] = text_source_desc(s.key)
            # watermark is intentionally blank in the master (stamped later)
            entry["blank_in_final"] = (s.key.lower() == "watermark")
        elif s.kind == "image":
            img_n += 1
            entry["fill_source"] = f"supporting image #{img_n}"
        elif s.kind == "logo":
            entry["fill_source"] = "brand logo"
        elif s.kind == "background":
            entry["fill_source"] = "background clip / chosen media"
        elif s.kind == "video":
            entry["fill_source"] = "trimmed clip / chosen media"
        elif s.kind == "intro":
            entry["fill_source"] = "intro clip (if provided)"
        audit.append(entry)
    return audit
