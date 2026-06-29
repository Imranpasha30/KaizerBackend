"""Custom-template contract: slot discovery + canvas parsing.

A developer-uploaded template is plain HTML/CSS (see TEMPLATE_CONTRACT.md). This module
parses the entry HTML and discovers the *slots* Kaizer must fill — any element carrying a
``data-kaizer`` attribute — plus the output canvas size. The rest of the engine
(renderer + composer) consumes the resulting :class:`TemplateContract`.

Pure parsing only (lxml). No rendering, no I/O beyond reading the passed HTML string.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from lxml import html as lxml_html

SLOT_ATTR = "data-kaizer"
DEFAULT_CANVAS = (1080, 1920)
# SECURITY: clamp the template-controlled canvas so a huge "kaizer:canvas" can't OOM
# the render host (a 99999x99999 RGBA buffer is ~40 GB). 4096 covers 4K either way.
MIN_CANVAS_DIM = 64
MAX_CANVAS_DIM = 4096

SlotKind = Literal["video", "background", "intro", "image", "text", "logo"]

# Recognised text-slot aliases (a bare value like "headline") and the generic "text:NAME".
_TEXT_ALIASES = {"headline", "hook", "subtitle", "title", "caption", "cta", "kicker",
                 "body", "ticker", "marquee", "watermark"}
_CANVAS_RE = re.compile(r"^\s*(\d{2,5})\s*[xX×]\s*(\d{2,5})\s*$")


def aspect_kind(canvas_w: int, canvas_h: int) -> str:
    """Categorise a template's output form purely from its canvas aspect — the one
    fact baked into the uploaded code. Landscape (wider than tall) is a *full-form*
    (16:9-ish) template; portrait or square is a *short* (9:16-ish) template. This is
    the source of truth the picker filters on and the render guard checks against, so a
    full-form template can never be rendered as a short (or vice-versa) and crash."""
    return "full" if (canvas_w or 0) > (canvas_h or 0) else "short"


@dataclass
class Slot:
    """One fillable region discovered in a template."""
    kind: SlotKind           # video | background | intro | image | text | logo
    name: str                # the bit after the colon, or the alias; "" for unnamed
    raw: str                 # the verbatim data-kaizer value (e.g. "video:main")
    tag: str                 # the element's tag name (video, div, img, h1, ...)
    default: str = ""        # data-kaizer-default — a bundled media file the user can swap
    placeholder: str = ""    # the author's inner text (used to detect/clear leaking placeholders)
    carousel: bool = False   # image slot opted into a multi-image slideshow (data-kaizer-carousel="1")

    @property
    def key(self) -> str:
        """Stable identifier used to map inputs onto this slot."""
        return self.name or self.kind


@dataclass
class TemplateContract:
    """The machine-understanding of an uploaded template."""
    canvas_w: int = DEFAULT_CANVAS[0]
    canvas_h: int = DEFAULT_CANVAS[1]
    slots: list[Slot] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def video_slots(self) -> list[Slot]:
        return [s for s in self.slots if s.kind == "video"]

    @property
    def background_slots(self) -> list[Slot]:
        return [s for s in self.slots if s.kind == "background"]

    @property
    def intro_slots(self) -> list[Slot]:
        return [s for s in self.slots if s.kind == "intro"]

    @property
    def image_slots(self) -> list[Slot]:
        return [s for s in self.slots if s.kind == "image"]

    @property
    def text_slots(self) -> list[Slot]:
        return [s for s in self.slots if s.kind == "text"]

    @property
    def kind(self) -> str:
        """'short' (portrait/square) or 'full' (landscape) — derived from the canvas."""
        return aspect_kind(self.canvas_w, self.canvas_h)

    def to_json(self) -> dict:
        return {
            "canvas": [self.canvas_w, self.canvas_h],
            "kind": self.kind,
            "slots": [{"kind": s.kind, "name": s.name, "raw": s.raw, "tag": s.tag,
                       "default": s.default, "placeholder": s.placeholder,
                       "carousel": s.carousel}
                      for s in self.slots],
            "warnings": list(self.warnings),
        }


def _classify(raw: str) -> tuple[SlotKind, str] | None:
    """Map a data-kaizer value to (kind, name). Returns None if unrecognised."""
    raw = (raw or "").strip()
    if not raw:
        return None
    base, _, name = raw.partition(":")
    base = base.strip().lower()
    name = name.strip()
    if base in ("video", "clip"):
        return ("video", name)
    if base in ("background", "bg"):
        return ("background", name or "background")
    if base == "intro":
        return ("intro", name or "intro")
    if base in ("image", "img", "photo"):
        return ("image", name)
    if base == "logo":
        return ("logo", name or "logo")
    if base == "text":
        return ("text", name)
    if base in _TEXT_ALIASES:
        return ("text", base)            # bare alias -> a named text slot
    return None


def parse_safe(html_str: str):
    """Parse template HTML without ever raising. Returns (doc_or_None, warnings).
    Handles empty, comment-only, and otherwise-unparseable input by wrapping/falling back,
    so a hostile or broken upload can never crash discovery/render."""
    warnings: list[str] = []
    if not html_str or not html_str.strip():
        return None, ["Empty template HTML."]
    try:
        return lxml_html.fromstring(html_str), warnings
    except Exception:
        pass
    # comment-only / fragment / junk: wrap defensively so lxml has a root element
    try:
        doc = lxml_html.fromstring("<div>" + html_str + "</div>")
        warnings.append("Template had no parseable HTML elements; parsed defensively.")
        return doc, warnings
    except Exception as exc:
        return None, [f"Template could not be parsed ({type(exc).__name__}); using defaults."]


def _parse_canvas(doc) -> tuple[int, int, list[str]]:
    warnings: list[str] = []
    # case-insensitive match on the meta name (authors write KAIZER:CANVAS, Kaizer:Canvas…)
    metas = doc.xpath(
        '//meta[translate(@name,"KAIZERCNVS:","kaizercnvs:")="kaizer:canvas"]/@content')
    if metas:
        m = _CANVAS_RE.match(metas[0])
        if m:
            w, h = int(m.group(1)), int(m.group(2))
            cw = max(MIN_CANVAS_DIM, min(w, MAX_CANVAS_DIM))
            ch = max(MIN_CANVAS_DIM, min(h, MAX_CANVAS_DIM))
            if (cw, ch) != (w, h):
                warnings.append(f"Canvas {w}x{h} clamped to {cw}x{ch} "
                                f"(max {MAX_CANVAS_DIM}px per side).")
            return cw, ch, warnings
        warnings.append(f"Unparseable kaizer:canvas '{metas[0]}', using default 1080x1920.")
    return DEFAULT_CANVAS[0], DEFAULT_CANVAS[1], warnings


def discover(html_str: str, *, canvas: tuple[int, int] | None = None) -> TemplateContract:
    """Parse template HTML → :class:`TemplateContract` (canvas + slots + warnings).

    Never raises. ``canvas`` overrides the parsed canvas (used after inference resolves a
    size from viewport/inline CSS for templates with no kaizer:canvas meta)."""
    doc, warnings = parse_safe(html_str)
    if doc is None:
        cw, ch = (canvas or DEFAULT_CANVAS)
        return TemplateContract(canvas_w=cw, canvas_h=ch, warnings=list(warnings))

    if canvas:
        cw, ch = int(canvas[0]), int(canvas[1])
        cw = max(MIN_CANVAS_DIM, min(cw, MAX_CANVAS_DIM))
        ch = max(MIN_CANVAS_DIM, min(ch, MAX_CANVAS_DIM))
    else:
        cw, ch, cwarn = _parse_canvas(doc)
        warnings = list(warnings) + cwarn
    contract = TemplateContract(canvas_w=cw, canvas_h=ch, warnings=list(warnings))

    seen_raw: set[str] = set()
    for el in doc.xpath(f"//*[@{SLOT_ATTR}]"):
        raw = el.get(SLOT_ATTR, "")
        cls = _classify(raw)
        if cls is None:
            contract.warnings.append(f"Unknown {SLOT_ATTR} value ignored: '{raw}'.")
            continue
        kind, name = cls
        # de-dupe identical raw values for media slots; keep first occurrence
        if kind in ("video", "background", "image") and raw in seen_raw:
            contract.warnings.append(f"Duplicate slot '{raw}' ignored (use distinct names).")
            continue
        seen_raw.add(raw)
        default = (el.get("data-kaizer-default") or "").strip()
        try:
            placeholder = " ".join((el.text_content() or "").split())[:200]
        except Exception:
            placeholder = ""
        # Author opt-in: an image slot marked data-kaizer-carousel="1" becomes a multi-image
        # slideshow (a plain image slot is never silently turned into one).
        is_carousel = (kind == "image" and
                       (el.get("data-kaizer-carousel") or "").strip() in ("1", "true", "yes"))
        contract.slots.append(Slot(kind=kind, name=name, raw=raw,
                                   tag=str(el.tag).lower(), default=default,
                                   placeholder=placeholder, carousel=is_carousel))

    if not contract.video_slots and not contract.background_slots:
        contract.warnings.append(
            "No video/background slot found. Add data-kaizer=\"video\" where the clip goes.")
    return contract
