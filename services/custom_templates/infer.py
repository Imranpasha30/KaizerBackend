"""Tolerant slot + canvas INFERENCE for custom templates — robustness for "any code".

Real users won't all use our ``data-kaizer`` convention. They mark regions with ``id`` /
``class`` names, semantic tags (``<video>``/``<h1>``/``<marquee>``/``<img>``), or other
attributes (``data-role``, ``data-ae-layer`` from After-Effects/Figma exports), and they
declare canvas size via the viewport meta or inline/CSS ``width:…px;height:…px`` rather
than our ``kaizer:canvas`` meta.

``normalize_template(html)`` parses defensively (never raises), INFERS the canvas, and
injects synthetic ``data-kaizer`` markers onto the best-guess elements for any role the
author didn't explicitly mark — returning normalized HTML the renderer/filler understand.
Conservative by design: each single-instance role (headline/ticker/logo/…) is assigned to
at most one element; opaque exports with no recognizable hints are left for the upload
audit to flag rather than mis-mapped.
"""
from __future__ import annotations

import re

from lxml import html as lxml_html

from .contract import (DEFAULT_CANVAS, MAX_CANVAS_DIM, MIN_CANVAS_DIM, SLOT_ATTR,
                       _classify, parse_safe)

# Attributes (besides id/class) that often carry the author's intent for a region.
_MARKER_ATTRS = ("data-role", "data-field", "data-slot", "data-type", "data-name",
                 "data-layer", "data-ae-layer", "data-kaizer-slot", "data-kaizer-role",
                 "role", "itemprop", "aria-label")

# Tags that must never be treated as a content slot (page chrome / the canvas itself).
_SKIP_TAGS = {"html", "head", "body", "style", "script", "meta", "title", "link", "base",
              "noscript", "nav", "header", "footer", "aside", "main"}
# Soft cap on per-element inference passes so a pathological template (tens of thousands of
# nodes) can't burn CPU. Semantic xpath passes still run (they're bounded + fast).
_MAX_INFER_ELEMENTS = 8000

# Ordered role rules — specific keywords first so "subtitle" doesn't match "headline" via
# the shared "title", etc. First keyword that is a substring of the element's haystack wins.
_RULES = [
    (("ticker", "marquee", "crawl", "scroller", "news-bar", "newsbar", "newsticker",
      "news-ticker", "lowerthird", "lower-third", "lower3rd"), ("text", "ticker")),
    (("subtitle", "sub-title", "subhead", "sub-head", "standfirst", "dek"), ("text", "subtitle")),
    (("headline", "head-line", "title", "heading", "story-title", "maintitle",
      "main-title", "title-txt", "title_txt"), ("text", "headline")),
    (("kicker", "eyebrow", "overline"), ("text", "kicker")),
    (("watermark", "water-mark"), ("text", "watermark")),
    (("cta", "call-to-action", "subscribe-btn"), ("text", "cta")),
    (("logo", "brandmark", "brand-mark"), ("logo", "logo")),
    (("background", "backdrop", "bg-video", "bgvideo"), ("background", "background")),
    (("videofill", "video-fill", "video", "clip", "footage", "reel", "mediafill"),
     ("video", "video")),
    (("photo", "picture", "poster", "image", "img-slot", "thumbnail"), ("image", "image")),
    (("summary", "description", "story-body", "bodycopy", "body-copy", "caption"),
     ("text", "body")),
]

_VP_W = re.compile(r"width\s*=\s*(\d{2,5})", re.I)
_VP_H = re.compile(r"height\s*=\s*(\d{2,5})", re.I)
_WH_STYLE = re.compile(r"width\s*:\s*(\d{2,5})px[^}]*?height\s*:\s*(\d{2,5})px", re.I | re.S)
_WXH = re.compile(r"(\d{2,5})\s*[x×]\s*(\d{2,5})")
_ASPECT = re.compile(r"(\d{1,2})\s*[:/]\s*(\d{1,2})")
_FONT_SIZE = re.compile(r"font-size\s*:\s*(\d{2,4})px", re.I)
_MAX_IMG_SLOTS = 8


def _clamp(w: int, h: int) -> tuple[int, int]:
    return (max(MIN_CANVAS_DIM, min(int(w), MAX_CANVAS_DIM)),
            max(MIN_CANVAS_DIM, min(int(h), MAX_CANVAS_DIM)))


def infer_canvas(doc, html: str):
    """Best-guess (w, h) for templates with no kaizer:canvas meta. Returns None when the
    meta is already present (discover will parse it) or nothing usable is found."""
    # explicit meta present (case-insensitive) -> let discover own it
    if doc.xpath('//meta[translate(@name,"KAIZERCNVS:","kaizercnvs:")="kaizer:canvas"]'):
        return None
    # 1) viewport meta: content="width=1080,height=1920"
    for c in doc.xpath('//meta[translate(@name,"VIEWPORT","viewport")="viewport"]/@content'):
        mw, mh = _VP_W.search(c), _VP_H.search(c)
        if mw and mh:
            return _clamp(int(mw.group(1)), int(mh.group(1)))
    # 2) WxH carried on an attribute (data-ae-comp, data-canvas, data-size…)
    for el in doc.iter():
        if not isinstance(getattr(el, "tag", None), str):
            continue
        for a in ("data-ae-comp", "data-canvas", "data-size", "data-dimensions"):
            v = el.get(a)
            if v:
                m = _WXH.search(v)
                if m:
                    return _clamp(int(m.group(1)), int(m.group(2)))
    # 3) inline width:..px;height:..px on body / first sized containers
    cands = doc.xpath('//body') + doc.xpath('//*[@style]')[:12]
    for el in cands:
        m = _WH_STYLE.search(el.get("style", "") or "")
        if m:
            return _clamp(int(m.group(1)), int(m.group(2)))
    # 4) <style> text: e.g. html,body{width:1080px;height:1920px}
    for st in doc.xpath('//style/text()'):
        m = _WH_STYLE.search(st or "")
        if m:
            return _clamp(int(m.group(1)), int(m.group(2)))
    # 5) data-aspect="9:16" -> pick a standard size in that orientation
    for v in doc.xpath('//*[@data-aspect]/@data-aspect'):
        m = _ASPECT.search(v)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return (1920, 1080) if a > b else (1080, 1920)
    return None


def _haystack(el) -> str:
    parts = [el.get("id") or "", el.get("class") or ""]
    for a in _MARKER_ATTRS:
        v = el.get(a)
        if v:
            parts.append(v)
    return " ".join(parts).lower()


def _match_role(hay: str):
    if not hay:
        return None
    for kws, role in _RULES:
        for kw in kws:
            if kw in hay:
                return role
    return None


def infer_slots(doc) -> list:
    """Add synthetic data-kaizer markers for roles the author didn't explicitly mark.
    Mutates ``doc`` in place. Returns the list of (kind, name) assigned."""
    assigned: list = []
    # roles already covered by explicit data-kaizer markers — don't duplicate them.
    explicit_roles: set = set()
    explicit_kinds: set = set()
    for el in doc.xpath(f"//*[@{SLOT_ATTR}]"):
        cls = _classify(el.get(SLOT_ATTR, ""))
        if cls:
            explicit_roles.add(cls[1] or cls[0])
            explicit_kinds.add(cls[0])
    taken: set = set()
    vid_n = [0]
    img_n = [0]

    def has(name: str) -> bool:
        return name in taken or name in explicit_roles

    def has_video() -> bool:
        # A real video/background slot already exists (author marker, <video> tag, or one we
        # added). Don't let class-name keyword matching multiply phantom video slots — a
        # template names many elements "...video..." (wrap, grad, overlay) and each would
        # otherwise become its own video:2 / video:3 slot.
        return ("video" in explicit_kinds or "background" in explicit_kinds or vid_n[0] > 0)

    def mark(el, value: str, kind: str, name: str):
        el.set(SLOT_ATTR, value)
        assigned.append((kind, name))

    def add_video(el):
        vid_n[0] += 1
        mark(el, "video" if vid_n[0] == 1 else f"video:{vid_n[0]}", "video", "video")

    def add_image(el):
        if img_n[0] >= _MAX_IMG_SLOTS:
            return
        img_n[0] += 1
        mark(el, "image" if img_n[0] == 1 else f"image:{img_n[0]}", "image", "image")

    # 1) keyword inference over id/class/marker-attrs
    seen = 0
    for el in doc.iter():
        seen += 1
        if seen > _MAX_INFER_ELEMENTS:
            break
        tag = getattr(el, "tag", None)
        if not isinstance(tag, str) or tag in _SKIP_TAGS:
            continue
        if el.get(SLOT_ATTR):
            continue
        role = _match_role(_haystack(el))
        if not role:
            continue
        kind, name = role
        if kind == "video":
            # Only infer a video slot from a class/id keyword when none exists yet — marking
            # every "...video..."-named element (wrap, grad, overlay) creates phantom slots.
            if not has_video():
                add_video(el)
        elif kind == "image":
            add_image(el)
        elif not has(name):
            taken.add(name)
            mark(el, name, kind, name)

    # 2) semantic-tag fallback for whatever's still missing
    for v in doc.xpath("//video"):
        if not v.get(SLOT_ATTR):
            add_video(v)
    if not has("ticker"):
        for m in doc.xpath("//marquee"):
            if not m.get(SLOT_ATTR):
                taken.add("ticker"); mark(m, "ticker", "text", "ticker"); break
    if not has("headline"):
        for h in (doc.xpath("//h1") or doc.xpath("//h2")):
            if not h.get(SLOT_ATTR):
                taken.add("headline"); mark(h, "headline", "text", "headline"); break
    if not has("subtitle"):
        for h in (doc.xpath("//h2") + doc.xpath("//h3")):
            if not h.get(SLOT_ATTR):
                taken.add("subtitle"); mark(h, "subtitle", "text", "subtitle"); break
    if not has("body"):
        for p in doc.xpath("//p"):
            if p.get(SLOT_ATTR):
                continue
            if len((p.text_content() or "").strip()) >= 12:
                taken.add("body"); mark(p, "body", "text", "body"); break
    for im in doc.xpath("//img"):
        if not im.get(SLOT_ATTR):
            add_image(im)

    # 3) inline-style heuristics for "div soup" (no tags, no keywords, all inline CSS):
    #    a div painted with background-image:url(...) is a photo region; the largest-font
    #    text element is the headline.
    seen = 0
    for el in doc.iter():
        seen += 1
        if seen > _MAX_INFER_ELEMENTS:
            break
        tag = getattr(el, "tag", None)
        if not isinstance(tag, str) or tag in _SKIP_TAGS or el.get(SLOT_ATTR):
            continue
        st = (el.get("style") or "").lower()
        if "background-image" in st and "url(" in st:
            add_image(el)
    if not has("headline"):
        best, best_sz, seen = None, 0, 0
        for el in doc.iter():
            seen += 1
            if seen > _MAX_INFER_ELEMENTS:
                break
            tag = getattr(el, "tag", None)
            if not isinstance(tag, str) or tag in _SKIP_TAGS or el.get(SLOT_ATTR):
                continue
            if not (el.text or "").strip():
                continue
            m = _FONT_SIZE.search(el.get("style") or "")
            if m and int(m.group(1)) > best_sz:
                best_sz, best = int(m.group(1)), el
        if best is not None and best_sz >= 40:
            taken.add("headline"); mark(best, "headline", "text", "headline")

    # 4) last resort: if there is STILL no video/background region, promote the first image
    #    slot to the video target so the trimmed clip always has somewhere to render (an
    #    image-led template plays the clip where its main visual would be).
    has_vid = ("video" in explicit_roles or "background" in explicit_roles
               or any(k in ("video", "background") for k, _ in assigned))
    if not has_vid:
        for el in doc.xpath(f"//*[@{SLOT_ATTR}]"):
            cls = _classify(el.get(SLOT_ATTR, ""))
            if cls and cls[0] == "image":
                el.set(SLOT_ATTR, "video")
                # the element is now a video, not an image — reflect that in the report
                # (replace an inferred image entry if present) so the audit isn't misleading.
                for i, (k, _n) in enumerate(assigned):
                    if k == "image":
                        assigned[i] = ("video", "video")
                        break
                else:
                    assigned.append(("video", "video"))
                break

    dedupe_nested_media_slots(doc)
    return assigned


_MEDIA_KINDS = {"video", "background", "image"}


def dedupe_nested_media_slots(doc) -> int:
    """Remove a media-slot marker from a CONTAINER that wraps another media slot of the SAME
    kind (e.g. a `.video-wrap` around the real `.video` box) — keeps the innermost slot.
    Prevents phantom slots without touching text-over-video (different kinds). Mutates ``doc``;
    returns the count removed. Shared by the offline inferer and the AI-understand pass."""
    removed = 0
    try:
        marked = list(doc.xpath(f"//*[@{SLOT_ATTR}]"))
    except Exception:
        return 0
    for el in marked:
        cls = _classify(el.get(SLOT_ATTR, ""))
        if not cls or cls[0] not in _MEDIA_KINDS:
            continue
        kind0 = cls[0]
        for d in el.iterdescendants():
            try:
                v = d.get(SLOT_ATTR)
            except Exception:
                continue
            if not v:
                continue
            dcls = _classify(v)
            if dcls and dcls[0] == kind0:
                try:
                    del el.attrib[SLOT_ATTR]
                    removed += 1
                except Exception:
                    pass
                break
    return removed


def normalize_template(html: str):
    """Parse defensively, infer canvas + slots, return (normalized_html, canvas|None, warnings).

    The normalized HTML carries synthetic data-kaizer markers for inferred regions so the
    renderer/filler treat an unmarked template exactly like a marked one. Never raises."""
    doc, warns = parse_safe(html)
    if doc is None:
        return html, None, list(warns)
    canvas = None
    try:
        canvas = infer_canvas(doc, html)
    except Exception:
        canvas = None
    assigned = []
    try:
        assigned = infer_slots(doc)
    except Exception:
        assigned = []
    if assigned:
        kinds = {}
        for k, _n in assigned:
            kinds[k] = kinds.get(k, 0) + 1
        desc = ", ".join(f"{v} {k}" for k, v in kinds.items())
        warns.append(f"Auto-detected unmarked region(s) ({desc}) from id/class/tags/attributes. "
                     f"For precise control, add data-kaizer markers.")
    try:
        out = lxml_html.tostring(doc, encoding="unicode")
    except Exception:
        return html, canvas, list(warns)
    return out, canvas, list(warns)


def normalize_and_discover(html: str):
    """Normalize (infer canvas + inject synthetic markers) then discover the contract from
    the normalized HTML. Returns (normalized_html, TemplateContract). Never raises."""
    from .contract import discover
    out, canvas, warns = normalize_template(html)
    c = discover(out, canvas=canvas)
    c.warnings = list(warns) + list(c.warnings)
    return out, c
