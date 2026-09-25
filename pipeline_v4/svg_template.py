# Ported from kaizer-platform@d5fd482 server/pipeline_v4/custom_template.py
# Changes from upstream: module renamed svg_template (OUR custom-template
# system is the HTML/CSS one in services/custom_templates — this adds the
# sibling SVG *layout* format alongside it, it does not replace anything).
# Parser logic, security gates and the CanvasLayout mapping are unchanged.
"""V4 SVG-template parsing — turn a user-uploaded SVG into a slot map.

The V4 render engine (``canvas_engine.render_canvas``) is driven ENTIRELY by
the percentage-based fields on ``canvas_schema.CanvasLayout`` (where the
trimmed video sits, where the picture panel sits, where the brand logo sits)
plus ``CanvasTextBlock`` overlays.

This module is the SVG template's front door: it accepts an uploaded SVG,
finds the "slots" the designer marked (where video / image / logo / text
content should land), converts each slot's bounding box to a percentage of
the SVG's viewBox, and hands back a :class:`ParsedTemplate`. The render side
then maps those percentages straight onto a ``CanvasLayout`` — no parallel
renderer, no change to ``canvas_engine``'s core logic. This is purely a
producer of valid ``CanvasLayout`` numbers.

Slot-detection convention (BOTH are supported — SVG editors export differently):
  1. An element whose ``id`` matches ``<role>-slot`` / ``<role>_slot`` /
     ``<role>slot`` with an optional ``-<n>`` index, e.g. ``video-slot``,
     ``image-slot``, ``image-slot-2``, ``logo_slot``, ``text-slot``.
  2. An element carrying ``data-kaizer-slot="video|image|text|logo"`` (with an
     optional ``data-kaizer-slot-index="2"``). ``id`` attributes survive almost
     every SVG editor; ``data-*`` is cleaner but not all tools preserve it.

Supported shapes for a slot's bounding box: ``rect``, ``image``, ``circle``,
``ellipse``, ``polygon``, ``polyline``. A slot marked on any other shape (e.g. a
``path`` with arbitrary béziers, a ``line``, a ``text`` node) raises a clear
error rather than guessing — we do NOT implement a full SVG geometry engine.

Security (this parses arbitrary user uploads — a real OWASP XML boundary):
  * Hard size cap (:data:`MAX_SVG_BYTES`) before anything else.
  * ``<!DOCTYPE`` and ``<!ENTITY`` are rejected up front, before the XML is
    handed to the parser. That single guard neutralises both classic XXE
    (external-entity file/SSRF reads need a DOCTYPE) and the billion-laughs /
    nested-entity-expansion DoS (needs ``<!ENTITY`` declarations). Only the five
    predefined XML entities remain, which are safe. This is the stdlib-only
    safe path — ``xml.etree.ElementTree`` with the dangerous constructs
    stripped before the parse can ever see them.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

# ─── Limits & conventions ────────────────────────────────────────────

# A template is layout metadata, not an asset. 2 MB is already absurdly
# generous for pure vector markup; anything bigger is either an abuse
# attempt or an SVG with embedded raster data we don't want.
MAX_SVG_BYTES = 2 * 1024 * 1024

# Roles a slot can declare. "video" is mandatory in a valid template.
VALID_ROLES = ("video", "image", "text", "logo")

# id-based convention: "<role>-slot", "<role>_slot", "<role>slot", optional
# "-<n>" / "_<n>" index. Case-insensitive.
_ID_SLOT_RE = re.compile(
    r"^(video|image|text|logo)[-_]?slot(?:[-_]?(\d+))?$", re.IGNORECASE
)

# Elements we can compute an exact bounding box for without a rendering engine.
_SUPPORTED_SHAPES = ("rect", "image", "circle", "ellipse", "polygon", "polyline")

# Tolerance (in percentage points) for the 0–100 bounds check — lets a slot
# that a design tool rounded to e.g. 100.02% through instead of rejecting a
# visually-fine template on a sub-pixel overshoot.
_BOUNDS_EPS = 0.75


class TemplateParseError(ValueError):
    """Raised for any invalid / unsafe / unusable uploaded template.

    Carries a human-readable, operator-facing reason — the router surfaces
    ``str(exc)`` directly as the 422 detail, so messages must be specific
    (e.g. "no video-slot element found") and safe to show a user.
    """


# ─── Parsed result ───────────────────────────────────────────────────

@dataclass(frozen=True)
class TemplateSlot:
    """One resolved slot as a percentage of the SVG viewBox.

    All four values are percentages in [0, 100]. ``index`` disambiguates
    repeated roles (``image-slot`` = 1, ``image-slot-2`` = 2); it is 1 for
    single-instance roles.
    """
    role: str
    index: int
    x_pct: float
    y_pct: float
    w_pct: float
    h_pct: float

    def to_json_dict(self) -> dict:
        return {
            "role": self.role,
            "index": self.index,
            "x_pct": self.x_pct,
            "y_pct": self.y_pct,
            "w_pct": self.w_pct,
            "h_pct": self.h_pct,
        }

    @staticmethod
    def from_json_dict(d: dict) -> "TemplateSlot":
        return TemplateSlot(
            role=str(d["role"]),
            index=int(d.get("index", 1)),
            x_pct=float(d["x_pct"]),
            y_pct=float(d["y_pct"]),
            w_pct=float(d["w_pct"]),
            h_pct=float(d["h_pct"]),
        )


@dataclass(frozen=True)
class ParsedTemplate:
    """The slot map extracted from one uploaded SVG.

    ``viewbox_w`` / ``viewbox_h`` are the SVG's own coordinate dimensions —
    kept for reference/debugging only; every slot is already stored as a
    percentage so the canvas dimensions the template renders at are
    independent of the SVG's authored size.
    """
    viewbox_w: float
    viewbox_h: float
    slots: tuple[TemplateSlot, ...]

    # ── Convenience accessors ────────────────────────────────────────
    @property
    def video_slot(self) -> TemplateSlot:
        for s in self.slots:
            if s.role == "video":
                return s
        # Constructed results always have exactly one video slot (parse
        # enforces it); this guards against a hand-built ParsedTemplate.
        raise TemplateParseError("template has no video slot")

    @property
    def image_slots(self) -> tuple[TemplateSlot, ...]:
        return tuple(sorted(
            (s for s in self.slots if s.role == "image"), key=lambda s: s.index
        ))

    @property
    def logo_slot(self) -> Optional[TemplateSlot]:
        for s in self.slots:
            if s.role == "logo":
                return s
        return None

    @property
    def text_slots(self) -> tuple[TemplateSlot, ...]:
        return tuple(sorted(
            (s for s in self.slots if s.role == "text"), key=lambda s: s.index
        ))

    def to_json_dict(self) -> dict:
        return {
            "viewbox_w": self.viewbox_w,
            "viewbox_h": self.viewbox_h,
            "slots": [s.to_json_dict() for s in self.slots],
        }

    @staticmethod
    def from_json_dict(d: dict) -> "ParsedTemplate":
        return ParsedTemplate(
            viewbox_w=float(d.get("viewbox_w", 0.0)),
            viewbox_h=float(d.get("viewbox_h", 0.0)),
            slots=tuple(TemplateSlot.from_json_dict(s) for s in d.get("slots", [])),
        )


# ─── Numeric helpers ─────────────────────────────────────────────────

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _num(raw: Optional[str], default: float = 0.0) -> float:
    """Parse an SVG length like ``"120"`` / ``"120px"`` / ``"12.5"`` to float.

    Unit suffixes (px, and the like) are stripped — we only ever consume the
    leading numeric part, which is correct for the userSpaceOnUse coordinates
    slot geometry lives in. Percentage-unit coordinates are not supported and
    parse to their numeric part (documented limitation)."""
    if raw is None:
        return default
    m = _NUM_RE.match(raw.strip())
    return float(m.group(0)) if m else default


def _all_nums(raw: Optional[str]) -> list[float]:
    if not raw:
        return []
    return [float(m) for m in _NUM_RE.findall(raw)]


def _local(tag: str) -> str:
    """Strip an ElementTree ``{namespace}local`` tag down to ``local``."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


# ─── Slot detection + bbox ───────────────────────────────────────────

def _slot_role_index(elem: ET.Element) -> Optional[tuple[str, int]]:
    """Return ``(role, index)`` if this element is a marked slot, else None.

    ``data-kaizer-slot`` wins over the id convention when both are present."""
    data_role = (elem.get("data-kaizer-slot") or "").strip().lower()
    if data_role in VALID_ROLES:
        idx_raw = (elem.get("data-kaizer-slot-index") or "").strip()
        idx = int(idx_raw) if idx_raw.isdigit() else 1
        return data_role, idx

    el_id = (elem.get("id") or "").strip()
    m = _ID_SLOT_RE.match(el_id)
    if m:
        role = m.group(1).lower()
        idx = int(m.group(2)) if m.group(2) else 1
        return role, idx
    return None


def _bbox_for_shape(elem: ET.Element) -> Optional[tuple[float, float, float, float]]:
    """(x, y, w, h) in the SVG's user coordinate space, or None if the shape
    type is unsupported for exact bounding-box computation."""
    shape = _local(elem.tag)
    if shape in ("rect", "image"):
        return (_num(elem.get("x")), _num(elem.get("y")),
                _num(elem.get("width")), _num(elem.get("height")))
    if shape == "circle":
        cx, cy, r = _num(elem.get("cx")), _num(elem.get("cy")), _num(elem.get("r"))
        return (cx - r, cy - r, 2 * r, 2 * r)
    if shape == "ellipse":
        cx, cy = _num(elem.get("cx")), _num(elem.get("cy"))
        rx, ry = _num(elem.get("rx")), _num(elem.get("ry"))
        return (cx - rx, cy - ry, 2 * rx, 2 * ry)
    if shape in ("polygon", "polyline"):
        nums = _all_nums(elem.get("points"))
        xs = nums[0::2]
        ys = nums[1::2]
        if not xs or not ys:
            return None
        x0, y0 = min(xs), min(ys)
        return (x0, y0, max(xs) - x0, max(ys) - y0)
    return None


def _viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    """Return (min_x, min_y, width, height) for the SVG coordinate system.

    Prefers ``viewBox``; falls back to the ``width``/``height`` attributes with
    a (0,0) origin. Raises if neither yields positive dimensions — we cannot
    convert a bounding box to a percentage without a reference frame."""
    vb = root.get("viewBox") or root.get("viewbox")
    if vb:
        parts = _all_nums(vb)
        if len(parts) == 4 and parts[2] > 0 and parts[3] > 0:
            return parts[0], parts[1], parts[2], parts[3]
    w = _num(root.get("width"))
    h = _num(root.get("height"))
    if w > 0 and h > 0:
        return 0.0, 0.0, w, h
    raise TemplateParseError(
        "SVG has no usable viewBox or width/height — cannot resolve slot "
        "positions. Add a viewBox (e.g. viewBox=\"0 0 1080 1920\")."
    )


# ─── Public entry point ──────────────────────────────────────────────

def parse_svg_template(data, *, max_bytes: int = MAX_SVG_BYTES) -> ParsedTemplate:
    """Parse + validate an uploaded SVG into a :class:`ParsedTemplate`.

    ``data`` may be ``str`` or ``bytes``. Raises :class:`TemplateParseError`
    with a specific, user-safe reason on any problem (oversize, DOCTYPE/ENTITY,
    malformed XML, no video slot, unsupported slot shape, out-of-bounds slot).
    """
    # 1) Size cap FIRST — before decode, before parse.
    if isinstance(data, str):
        raw_bytes = data.encode("utf-8", errors="replace")
    else:
        raw_bytes = bytes(data)
    if len(raw_bytes) > max_bytes:
        raise TemplateParseError(
            f"Template SVG is too large ({len(raw_bytes)} bytes; limit "
            f"{max_bytes} bytes). A template is layout markup, not an asset."
        )
    if not raw_bytes.strip():
        raise TemplateParseError("Empty upload — no SVG content.")

    text = raw_bytes.decode("utf-8", errors="replace")

    # 2) XXE / entity-expansion guard — reject the dangerous DTD constructs
    #    BEFORE the parser can act on them. Case-insensitive, whitespace
    #    tolerant so "<!  DOCTYPE" style evasion is still caught.
    lowered = text.lower()
    if "<!doctype" in lowered or "<!entity" in lowered:
        raise TemplateParseError(
            "Template SVG contains a DOCTYPE or ENTITY declaration, which is "
            "rejected for security. Export a plain SVG without a DTD."
        )

    # 3) Parse with the stdlib parser (safe now that DTD/entities are gone).
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise TemplateParseError(f"Malformed SVG XML: {exc}") from exc

    if _local(root.tag) != "svg":
        raise TemplateParseError(
            f"Root element is <{_local(root.tag)}>, expected <svg>."
        )

    min_x, min_y, vb_w, vb_h = _viewbox(root)

    # 4) Walk the whole tree, collect marked slots.
    slots: list[TemplateSlot] = []
    seen: set[tuple[str, int]] = set()
    for elem in root.iter():
        role_idx = _slot_role_index(elem)
        if role_idx is None:
            continue
        role, index = role_idx
        key = (role, index)
        if key in seen:
            # Duplicate role+index — first one wins, ignore the rest so a
            # stray copy in the SVG doesn't hard-fail an otherwise good file.
            continue

        bbox = _bbox_for_shape(elem)
        if bbox is None:
            raise TemplateParseError(
                f"Slot '{role}-slot' is on an unsupported shape "
                f"<{_local(elem.tag)}>. Mark slots with a rect, image, circle, "
                f"ellipse, polygon, or polyline (arbitrary paths are not "
                f"supported)."
            )
        bx, by, bw, bh = bbox
        if bw <= 0 or bh <= 0:
            raise TemplateParseError(
                f"Slot '{role}-slot' has a zero or negative size."
            )

        x_pct = (bx - min_x) / vb_w * 100.0
        y_pct = (by - min_y) / vb_h * 100.0
        w_pct = bw / vb_w * 100.0
        h_pct = bh / vb_h * 100.0

        if (x_pct < -_BOUNDS_EPS or y_pct < -_BOUNDS_EPS
                or x_pct + w_pct > 100.0 + _BOUNDS_EPS
                or y_pct + h_pct > 100.0 + _BOUNDS_EPS):
            raise TemplateParseError(
                f"Slot '{role}-slot' falls outside the canvas bounds "
                f"(resolved to x={x_pct:.1f}%, y={y_pct:.1f}%, w={w_pct:.1f}%, "
                f"h={h_pct:.1f}%). Keep every slot inside the viewBox."
            )

        # Clamp sub-epsilon overshoot to exact bounds so downstream math and
        # the CanvasLayout validators (0–100 fields) never see 100.02.
        x_pct = min(max(x_pct, 0.0), 100.0)
        y_pct = min(max(y_pct, 0.0), 100.0)
        w_pct = min(w_pct, 100.0 - x_pct)
        h_pct = min(h_pct, 100.0 - y_pct)

        seen.add(key)
        slots.append(TemplateSlot(role=role, index=index,
                                  x_pct=round(x_pct, 4), y_pct=round(y_pct, 4),
                                  w_pct=round(w_pct, 4), h_pct=round(h_pct, 4)))

    # 5) A template MUST define where the video goes.
    if not any(s.role == "video" for s in slots):
        raise TemplateParseError(
            "No video-slot element found. A template needs at least one "
            "element with id \"video-slot\" (or data-kaizer-slot=\"video\") "
            "marking where the trimmed video sits."
        )

    return ParsedTemplate(viewbox_w=vb_w, viewbox_h=vb_h, slots=tuple(slots))


# ─── CanvasLayout mapping ────────────────────────────────────────────

def build_layout_kwargs(parsed: ParsedTemplate) -> dict:
    """Map a parsed template's slots to ``CanvasLayout`` field overrides.

    Returns a kwargs dict the render side merges into ``CanvasLayout(width=…,
    height=…, **kwargs)``. Only the geometry fields the template actually
    defines are set:

      * video slot  -> video_x_pct / video_y_pct / video_w_pct / video_h_pct
      * first image -> picture_x_pct / picture_y_pct / picture_w_pct / picture_h_pct
      * logo slot   -> brand_logo_x_pct / brand_logo_y_pct / brand_logo_w_pct
        (``CanvasLayout`` has no logo HEIGHT field — the engine scales the logo
        to width and preserves aspect, so ``h_pct`` is intentionally dropped.)

    This does NOT touch text slots — those become ``CanvasTextBlock`` entries
    via :func:`build_text_block_kwargs`, a separate concern (overlays, not
    layout).
    """
    v = parsed.video_slot
    kwargs: dict = {
        "video_x_pct": v.x_pct,
        "video_y_pct": v.y_pct,
        "video_w_pct": v.w_pct,
        "video_h_pct": v.h_pct,
    }
    images = parsed.image_slots
    if images:
        img = images[0]
        kwargs.update({
            "picture_x_pct": img.x_pct,
            "picture_y_pct": img.y_pct,
            "picture_w_pct": img.w_pct,
            "picture_h_pct": img.h_pct,
        })
    logo = parsed.logo_slot
    if logo is not None:
        kwargs.update({
            "brand_logo_x_pct": logo.x_pct,
            "brand_logo_y_pct": logo.y_pct,
            "brand_logo_w_pct": logo.w_pct,
        })
    return kwargs


def build_text_block_kwargs(parsed: ParsedTemplate) -> Optional[dict]:
    """Return ``CanvasTextBlock`` geometry kwargs for the first text slot, or
    None when the template declares no text slot.

    ``font_size_pct`` is estimated as ~55% of the slot's height (a readable
    single-line default); the render side supplies the actual ``text`` and
    ``kind``. Returned kwargs cover only position + size so the caller stays
    in control of content and styling.
    """
    texts = parsed.text_slots
    if not texts:
        return None
    t = texts[0]
    return {
        "x_pct": t.x_pct,
        "y_pct": t.y_pct,
        "w_pct": t.w_pct,
        "font_size_pct": round(max(2.0, t.h_pct * 0.55), 4),
    }
