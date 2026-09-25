# New for the dual-template merge (design informed by kaizer-platform@d5fd482
# server/pipeline_v4/custom_template.py, which mapped SVG slots onto
# CanvasLayout — OUR custom-template renderer is the HTML/Chromium/ffmpeg
# pipeline instead, so here the parsed SVG becomes a generated HTML ENTRY and
# the entire existing render path runs unchanged).
"""SVG template → HTML wrapper entry.

An uploaded SVG *layout* template (slots marked per
``pipeline_v4.svg_template``) is turned into a normal custom-template bundle:

  * the slot-marked shapes are REMOVED from the artwork (they are
    placeholders — real content lands there), the rest of the SVG is
    sanitized and inlined full-bleed as the design layer;
  * each parsed slot becomes an absolutely-positioned ``data-kaizer`` div at
    the slot's percentage geometry — exactly the markup
    ``services/custom_templates`` already understands (contract discovery,
    Chromium still-frame capture, video hole-punching, ffmpeg compose).

Video reveal needs no z-order care: the renderer's ``_punch_holes`` zeroes
the design PNG's alpha inside every video rect AFTER the screenshot, so
artwork overlapping a video box is cleared no matter how it is stacked.

Security: ``parse_svg_template`` already rejected DOCTYPE/ENTITY (XXE /
billion-laughs) and oversize input. This module strips the *scripting*
surface an SVG can carry — ``<script>``, ``<foreignObject>``, ``on*``
handlers, ``javascript:`` URLs, external ``href``/``xlink:href`` and
external ``url()`` references — before the markup is embedded. The bundle
sanitizer (lxml) and the renderer's file://-only network guard remain as
further layers behind it.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from pipeline_v4.svg_template import ParsedTemplate, _ID_SLOT_RE, _local

_SVG_NS = "http://www.w3.org/2000/svg"
_XLINK_NS = "http://www.w3.org/1999/xlink"

# Element local names that carry script execution — removed with their subtree.
_DANGEROUS_TAGS = {"script", "foreignobject"}

# href values allowed to survive: in-document fragments and safe raster
# data URIs. Everything else (http(s), file, javascript:, data:image/svg,
# data:text/html …) is stripped.
_SAFE_HREF_RE = re.compile(
    r"^(#|data:image/(?:png|jpe?g|gif|webp);)", re.IGNORECASE
)

# url(...) references inside style attributes/values: only in-document
# fragment refs (url(#gradient)) are safe offline.
_EXTERNAL_URL_RE = re.compile(r"url\(\s*(?!['\"]?#)", re.IGNORECASE)

# Canvas normalisation: scale the viewBox to a standard render resolution
# (vector artwork loses nothing) so a tiny authored viewBox (0 0 108 192)
# doesn't produce a tiny low-res render. Clamps mirror the HTML contract.
_TARGET_SHORT_SIDE = 1080
_CANVAS_MIN, _CANVAS_MAX = 64, 4096


def canvas_size_for(parsed: ParsedTemplate) -> tuple[int, int]:
    """(width, height) in px for the wrapper canvas, aspect-true to the
    SVG's viewBox with the SHORT side normalised to 1080. An extreme
    aspect scales BOTH sides down to fit the 4096 cap (a per-dimension
    clamp would break the aspect and misalign every % slot)."""
    vw = max(1.0, float(parsed.viewbox_w))
    vh = max(1.0, float(parsed.viewbox_h))
    if vw <= vh:                      # portrait / square → short-form aspect
        w, h = float(_TARGET_SHORT_SIDE), _TARGET_SHORT_SIDE * vh / vw
    else:                             # landscape → full-form aspect
        h, w = float(_TARGET_SHORT_SIDE), _TARGET_SHORT_SIDE * vw / vh
    long_side = max(w, h)
    if long_side > _CANVAS_MAX:
        scale = _CANVAS_MAX / long_side
        w, h = w * scale, h * scale
    return (max(_CANVAS_MIN, round(w)), max(_CANVAS_MIN, round(h)))


def _is_slot_marked(elem: ET.Element) -> bool:
    if (elem.get("data-kaizer-slot") or "").strip():
        return True
    return bool(_ID_SLOT_RE.match((elem.get("id") or "").strip()))


def sanitize_svg(svg_text: str) -> str:
    """Return the SVG markup with the scripting surface and every
    slot-marked placeholder shape removed. Raises ``ET.ParseError`` on
    malformed XML (callers run ``parse_svg_template`` first, which already
    rejected it with a user-safe message)."""
    root = ET.fromstring(svg_text)

    def _clean(elem: ET.Element) -> None:
        for child in list(elem):
            if (_local(child.tag).lower() in _DANGEROUS_TAGS
                    or _is_slot_marked(child)):
                elem.remove(child)
                continue
            _clean(child)
        for attr in list(elem.attrib):
            local_attr = attr.rsplit("}", 1)[-1].lower()
            val = elem.attrib[attr] or ""
            if local_attr.startswith("on"):
                del elem.attrib[attr]
            elif local_attr.startswith("data-kaizer"):
                # data-kaizer is the ENGINE's own contract attribute —
                # artwork must never carry it (a marker here would make the
                # fill JS treat a design node as a slot).
                del elem.attrib[attr]
            elif local_attr in ("href", "xlink:href") or attr.endswith("}href"):
                if not _SAFE_HREF_RE.match(val.strip()):
                    del elem.attrib[attr]
                elif attr != "href":
                    # Normalise xlink:href → href: survives the bundle
                    # sanitizer's lxml round-trip (which drops namespace
                    # prefixes) and Chromium supports plain href everywhere.
                    elem.attrib["href"] = elem.attrib.pop(attr)
            elif "javascript:" in val.lower():
                del elem.attrib[attr]
            elif local_attr == "style" and _EXTERNAL_URL_RE.search(val):
                del elem.attrib[attr]
            elif _EXTERNAL_URL_RE.search(val) and "url(" in val.lower():
                del elem.attrib[attr]
        # A leftover slot-looking id on a non-slot element (a duplicate the
        # parser ignored) would trip the HTML inference net — drop the id.
        if _ID_SLOT_RE.match((elem.get("id") or "").strip()):
            del elem.attrib["id"]

    _clean(root)
    if _local(root.tag).lower() in _DANGEROUS_TAGS:
        raise ValueError("SVG root element is not renderable")

    ET.register_namespace("", _SVG_NS)
    ET.register_namespace("xlink", _XLINK_NS)
    return ET.tostring(root, encoding="unicode")


# Text slots by index → the contract's text kinds (index 1 is the main
# headline; extras become secondary kinds so each gets distinct story text;
# beyond these, slots get unique text:extraN names — never duplicate keys).
_TEXT_KINDS = ("headline", "subtitle", "caption", "body")


def _slot_div(kaizer: str, s, z: int, extra_style: str = "",
              inner: str = "") -> str:
    style = (f"left:{s.x_pct}%;top:{s.y_pct}%;"
             f"width:{s.w_pct}%;height:{s.h_pct}%;z-index:{z};{extra_style}")
    return (f'  <div class="kx-slot" data-kaizer="{kaizer}" '
            f'style="{style}">{inner}</div>')


def build_wrapper_html(parsed: ParsedTemplate, sanitized_svg: str,
                       name: str = "SVG template") -> str:
    """Generate the bundle's ``index.html`` from a parsed SVG template."""
    w, h = canvas_size_for(parsed)
    lines: list[str] = []

    for i, s in enumerate(sorted((x for x in parsed.slots if x.role == "video"),
                                 key=lambda x: x.index)):
        key = "video" if i == 0 else f"video:video{s.index}"
        lines.append(_slot_div(key, s, z=1))

    for i, s in enumerate(parsed.image_slots):
        key = "image" if i == 0 else f"image:image{s.index}"
        # The fill JS sets only background-image — without cover/center a
        # filled photo tiles or shows a top-left crop at natural size.
        lines.append(_slot_div(
            key, s, z=3,
            extra_style=("background-size:cover;background-position:center;"
                         "background-repeat:no-repeat;")))

    for i, s in enumerate(parsed.text_slots):
        # Distinct contract kinds per slot (duplicate keys would render the
        # same text): headline, subtitle, caption, body, then text:extraN.
        kind = (_TEXT_KINDS[i] if i < len(_TEXT_KINDS)
                else f"text:extra{i + 1}")
        font_px = max(14, round(h * s.h_pct / 100.0 * 0.55))
        lines.append(_slot_div(
            kind, s, z=4,
            extra_style=(
                "display:flex;align-items:center;justify-content:center;"
                "text-align:center;color:#ffffff;font-weight:700;"
                f"font-size:{font_px}px;line-height:1.15;overflow:hidden;"
                "font-family:'Noto Sans','Segoe UI',system-ui,sans-serif;"
                "text-shadow:0 2px 8px rgba(0,0,0,.45);"),
            inner=f"{kind.capitalize()} here"))

    logo = parsed.logo_slot
    if logo is not None:
        lines.append(_slot_div("logo", logo, z=4))

    slots_html = "\n".join(lines)
    title = (name or "SVG template").replace("<", "").replace(">", "")[:120]
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="kaizer:canvas" content="{w}x{h}">
<title>{title}</title>
<style>
  html, body {{ margin: 0; padding: 0; width: {w}px; height: {h}px;
                overflow: hidden; background: transparent; }}
  .kx-slot {{ position: absolute; }}
  .kx-art  {{ position: absolute; left: 0; top: 0; width: {w}px;
              height: {h}px; z-index: 2; pointer-events: none; }}
  .kx-art svg {{ width: 100%; height: 100%; display: block; }}
</style>
</head>
<body>
{slots_html}
  <!-- design artwork (slot placeholders removed; video regions are
       alpha-punched from the captured design after screenshot) -->
  <div class="kx-art">{sanitized_svg}</div>
</body>
</html>
"""
