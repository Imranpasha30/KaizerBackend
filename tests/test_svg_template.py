"""SVG template system — parser (ported from kaizer-platform@d5fd482
tests/test_custom_template.py, import path renamed) + OUR wrapper layer
(services/custom_templates/svg_wrap.py): sanitizer security, generated
HTML-entry correctness, and the load-bearing proof that the wrapper is
consumable by the REAL contract-discovery layer the render path uses."""
from __future__ import annotations

import pytest

from pipeline_v4.svg_template import (
    MAX_SVG_BYTES,
    ParsedTemplate,
    TemplateParseError,
    build_layout_kwargs,
    build_text_block_kwargs,
    parse_svg_template,
)
from services.custom_templates import svg_wrap


# ─── helpers ─────────────────────────────────────────────────────────

def _svg(body: str, *, vb: str = "0 0 1000 2000") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}">{body}</svg>'
    )


def _slot_by_role(parsed: ParsedTemplate, role: str, index: int = 1):
    for s in parsed.slots:
        if s.role == role and s.index == index:
            return s
    return None


# ─── valid parsing (ported) ──────────────────────────────────────────

def test_valid_rect_slots_produce_correct_percentages():
    svg = _svg(
        '<rect id="video-slot" x="0" y="0" width="1000" height="1400"/>'
        '<rect id="image-slot" x="100" y="1500" width="800" height="400"/>'
        '<rect id="logo-slot" x="880" y="40" width="100" height="100"/>'
        '<rect id="text-slot" x="50" y="1420" width="900" height="60"/>'
    )
    parsed = parse_svg_template(svg)

    v = _slot_by_role(parsed, "video")
    assert (v.x_pct, v.y_pct, v.w_pct, v.h_pct) == (0.0, 0.0, 100.0, 70.0)

    img = _slot_by_role(parsed, "image")
    assert (img.x_pct, img.y_pct, img.w_pct, img.h_pct) == (10.0, 75.0, 80.0, 20.0)

    logo = _slot_by_role(parsed, "logo")
    assert (logo.x_pct, logo.y_pct, logo.w_pct) == (88.0, 2.0, 10.0)

    txt = _slot_by_role(parsed, "text")
    assert (txt.x_pct, txt.y_pct, txt.w_pct) == (5.0, 71.0, 90.0)


def test_data_attribute_convention_supported():
    svg = _svg(
        '<rect data-kaizer-slot="video" x="0" y="0" width="500" height="1000"/>'
        '<rect data-kaizer-slot="image" data-kaizer-slot-index="2" '
        'x="0" y="1000" width="500" height="500"/>'
    )
    parsed = parse_svg_template(svg)
    v = _slot_by_role(parsed, "video")
    assert v is not None and v.w_pct == 50.0
    img2 = _slot_by_role(parsed, "image", index=2)
    assert img2 is not None and img2.index == 2


def test_indexed_image_slots_via_id():
    svg = _svg(
        '<rect id="video-slot" x="0" y="0" width="1000" height="1000"/>'
        '<rect id="image-slot" x="0" y="1000" width="500" height="500"/>'
        '<rect id="image-slot-2" x="500" y="1000" width="500" height="500"/>'
    )
    parsed = parse_svg_template(svg)
    assert len(parsed.image_slots) == 2
    assert [s.index for s in parsed.image_slots] == [1, 2]


def test_underscore_id_variants_supported():
    for slot_id in ("video_slot", "videoslot", "VIDEO-SLOT"):
        svg = _svg(f'<rect id="{slot_id}" x="0" y="0" width="1000" height="2000"/>')
        parsed = parse_svg_template(svg)
        assert parsed.video_slot.w_pct == 100.0


def test_viewbox_offset_is_respected():
    svg = _svg(
        '<rect id="video-slot" x="100" y="200" width="900" height="1800"/>',
        vb="100 200 900 1800",
    )
    parsed = parse_svg_template(svg)
    v = parsed.video_slot
    assert (v.x_pct, v.y_pct, v.w_pct, v.h_pct) == (0.0, 0.0, 100.0, 100.0)


def test_width_height_fallback_when_no_viewbox():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="800px" height="600px">'
        '<rect id="video-slot" x="0" y="0" width="400" height="600"/>'
        '</svg>'
    )
    parsed = parse_svg_template(svg)
    v = parsed.video_slot
    assert v.w_pct == 50.0 and v.h_pct == 100.0


# ─── shape bounding boxes (ported) ───────────────────────────────────

def test_circle_bbox():
    svg = _svg('<circle id="video-slot" cx="500" cy="1000" r="250"/>')
    v = parse_svg_template(svg).video_slot
    assert (v.x_pct, v.y_pct, v.w_pct, v.h_pct) == (25.0, 37.5, 50.0, 25.0)


def test_ellipse_bbox():
    svg = _svg('<ellipse id="video-slot" cx="500" cy="1000" rx="200" ry="400"/>')
    v = parse_svg_template(svg).video_slot
    assert (v.x_pct, v.w_pct, v.h_pct) == (30.0, 40.0, 40.0)


def test_polygon_bbox():
    svg = _svg('<polygon id="video-slot" points="0,0 1000,0 1000,2000 0,2000"/>')
    v = parse_svg_template(svg).video_slot
    assert (v.x_pct, v.y_pct, v.w_pct, v.h_pct) == (0.0, 0.0, 100.0, 100.0)


# ─── rejection paths (ported) ────────────────────────────────────────

def test_missing_video_slot_rejected():
    svg = _svg('<rect id="image-slot" x="0" y="0" width="100" height="100"/>')
    with pytest.raises(TemplateParseError, match="video-slot"):
        parse_svg_template(svg)


def test_doctype_entity_xxe_rejected():
    malicious = (
        '<?xml version="1.0"?>'
        '<!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<rect id="video-slot" x="0" y="0" width="100" height="100"/>'
        '<text>&xxe;</text></svg>'
    )
    with pytest.raises(TemplateParseError, match="DOCTYPE or ENTITY"):
        parse_svg_template(malicious)


def test_billion_laughs_entity_rejected():
    bomb = (
        '<!DOCTYPE lolz [<!ENTITY lol "lol">'
        '<!ENTITY lol2 "&lol;&lol;&lol;">]>'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<rect id="video-slot" x="0" y="0" width="10" height="10"/></svg>'
    )
    with pytest.raises(TemplateParseError, match="DOCTYPE or ENTITY"):
        parse_svg_template(bomb)


def test_oversized_rejected():
    filler = "<!-- " + "A" * (MAX_SVG_BYTES + 10) + " -->"
    svg = _svg(
        f'<rect id="video-slot" x="0" y="0" width="1000" height="2000"/>{filler}'
    )
    with pytest.raises(TemplateParseError, match="too large"):
        parse_svg_template(svg)


def test_slot_out_of_bounds_rejected():
    svg = _svg('<rect id="video-slot" x="900" y="0" width="500" height="2000"/>')
    with pytest.raises(TemplateParseError, match="outside the canvas bounds"):
        parse_svg_template(svg)


def test_unsupported_shape_for_slot_rejected():
    svg = _svg('<path id="video-slot" d="M0 0 C 100 100 200 0 300 100 Z"/>')
    with pytest.raises(TemplateParseError, match="unsupported shape"):
        parse_svg_template(svg)


def test_non_svg_root_rejected():
    with pytest.raises(TemplateParseError, match="expected <svg>"):
        parse_svg_template('<html><body>nope</body></html>')


def test_malformed_xml_rejected():
    with pytest.raises(TemplateParseError, match="Malformed"):
        parse_svg_template('<svg><rect id="video-slot" </svg>')


def test_empty_rejected():
    with pytest.raises(TemplateParseError, match="Empty"):
        parse_svg_template("   ")


# ─── layout kwargs mapping (ported; fields verified against OUR schema) ──

def test_build_layout_kwargs_maps_all_slots():
    svg = _svg(
        '<rect id="video-slot" x="0" y="0" width="1000" height="1400"/>'
        '<rect id="image-slot" x="100" y="1500" width="800" height="400"/>'
        '<rect id="logo-slot" x="880" y="40" width="100" height="100"/>'
    )
    kw = build_layout_kwargs(parse_svg_template(svg))
    assert kw["video_w_pct"] == 100.0 and kw["video_h_pct"] == 70.0
    assert kw["picture_x_pct"] == 10.0 and kw["picture_w_pct"] == 80.0
    assert kw["brand_logo_x_pct"] == 88.0 and kw["brand_logo_w_pct"] == 10.0
    assert "brand_logo_h_pct" not in kw
    # Every emitted field must exist on OUR CanvasLayout (schema parity).
    from pipeline_v4.canvas_schema import CanvasLayout
    layout = CanvasLayout(width=1080, height=1920, **kw)
    assert layout.video_h_pct == 70.0


def test_build_text_block_kwargs():
    svg = _svg(
        '<rect id="video-slot" x="0" y="0" width="1000" height="1800"/>'
        '<rect id="text-slot" x="50" y="1820" width="900" height="100"/>'
    )
    tb = build_text_block_kwargs(parse_svg_template(svg))
    assert tb is not None
    assert tb["x_pct"] == 5.0 and tb["w_pct"] == 90.0
    assert tb["font_size_pct"] == pytest.approx(2.75, abs=0.01)


def test_build_text_block_kwargs_none_when_absent():
    svg = _svg('<rect id="video-slot" x="0" y="0" width="1000" height="2000"/>')
    assert build_text_block_kwargs(parse_svg_template(svg)) is None


def test_json_roundtrip():
    svg = _svg(
        '<rect id="video-slot" x="0" y="0" width="1000" height="1400"/>'
        '<rect id="image-slot" x="100" y="1500" width="800" height="400"/>'
    )
    parsed = parse_svg_template(svg)
    restored = ParsedTemplate.from_json_dict(parsed.to_json_dict())
    assert restored.to_json_dict() == parsed.to_json_dict()
    assert restored.video_slot.w_pct == 100.0


# ─── OUR wrapper layer: sanitizer security ───────────────────────────

_BASE = ('<rect id="video-slot" x="0" y="0" width="1000" height="1400"/>'
         '<rect fill="#123" x="0" y="1400" width="1000" height="600"/>')


def test_sanitize_strips_scripting_surface():
    svg = _svg(
        _BASE
        + '<script>alert(1)</script>'
        + '<foreignObject x="0" y="0" width="10" height="10">'
          '<body xmlns="http://www.w3.org/1999/xhtml">x</body></foreignObject>'
        + '<circle cx="1" cy="1" r="1" onclick="alert(2)" onload="alert(3)"/>'
        + '<image x="0" y="0" width="10" height="10" href="https://evil/x.png"/>'
        + '<a href="javascript:alert(4)"><text x="1" y="1">t</text></a>'
    )
    out = svg_wrap.sanitize_svg(svg)
    low = out.lower()
    assert "<script" not in low and "foreignobject" not in low
    assert "onclick" not in low and "onload" not in low
    assert "https://evil" not in low and "javascript:" not in low


def test_sanitize_keeps_safe_refs_and_normalizes_xlink():
    svg = _svg(
        _BASE
        + '<defs><linearGradient id="g"><stop offset="0"/></linearGradient></defs>'
        + '<rect x="0" y="0" width="5" height="5" fill="url(#g)"/>'
        + ('<image x="0" y="0" width="10" height="10" '
           'xmlns:xlink="http://www.w3.org/1999/xlink" '
           'xlink:href="data:image/png;base64,iVBORw0KGgo="/>')
    )
    out = svg_wrap.sanitize_svg(svg)
    assert "url(#g)" in out                       # in-document ref survives
    assert "data:image/png;base64" in out         # safe raster data URI kept
    assert "xlink:href=" not in out               # normalised to plain href
    assert 'href="data:image/png' in out


def test_sanitize_removes_slot_placeholder_shapes():
    svg = _svg(_BASE + '<rect id="image-slot" x="0" y="0" width="10" height="10"/>')
    out = svg_wrap.sanitize_svg(svg)
    assert "video-slot" not in out and "image-slot" not in out
    assert 'fill="#123"' in out                   # plain artwork survives


# ─── OUR wrapper layer: generated entry ──────────────────────────────

def _full_svg():
    return _svg(
        '<rect id="video-slot" x="0" y="0" width="1000" height="1400"/>'
        '<rect id="image-slot" x="100" y="1500" width="800" height="400"/>'
        '<rect id="logo-slot" x="880" y="40" width="100" height="100"/>'
        '<rect id="text-slot" x="50" y="1420" width="900" height="60"/>'
    )


def test_canvas_size_normalised_to_1080_short_side():
    parsed = parse_svg_template(_full_svg())          # 1000x2000 portrait
    assert svg_wrap.canvas_size_for(parsed) == (1080, 2160)
    landscape = parse_svg_template(
        _svg('<rect id="video-slot" x="0" y="0" width="1920" height="1080"/>',
             vb="0 0 1920 1080"))
    assert svg_wrap.canvas_size_for(landscape) == (1920, 1080)


def test_wrapper_html_carries_marked_slots_at_percentages():
    parsed = parse_svg_template(_full_svg())
    clean = svg_wrap.sanitize_svg(_full_svg())
    html = svg_wrap.build_wrapper_html(parsed, clean, name="My SVG")
    assert '<meta name="kaizer:canvas" content="1080x2160">' in html
    assert 'data-kaizer="video"' in html
    assert 'data-kaizer="image"' in html
    assert 'data-kaizer="headline"' in html
    assert 'data-kaizer="logo"' in html
    # slot geometry travels as percentages
    assert "left:10.0%;top:75.0%;width:80.0%;height:20.0%" in html
    # the artwork's slot placeholders are gone; the svg element is embedded
    assert "video-slot" not in html and "<svg" in html


def test_inference_never_marks_svg_artwork_as_slots():
    """BLOCKER regression: design-tool layer names inside the SVG artwork
    (id="Background", "logo-mark", "photo-frame", "main-title" — standard
    Figma/Illustrator exports) must NOT become phantom slots. A phantom
    background wiped the artwork pre-screenshot and disabled video hole-
    punching; a phantom logo hijacked the publish brand stamp."""
    from services import custom_templates as ct
    art_svg = _svg(
        '<rect id="video-slot" x="0" y="0" width="1000" height="1400"/>'
        '<g id="Background"><rect x="0" y="0" width="1000" height="2000" fill="#123"/></g>'
        '<g id="logo-mark"><circle cx="60" cy="60" r="40"/></g>'
        '<g id="photo-frame"><rect x="100" y="100" width="200" height="200"/></g>'
        '<text id="main-title" x="500" y="1900">STATIC DESIGN TEXT</text>'
    )
    parsed = parse_svg_template(art_svg)
    clean = svg_wrap.sanitize_svg(art_svg)
    html = svg_wrap.build_wrapper_html(parsed, clean, name="art test")
    _norm, contract = ct.normalize_and_discover(html)
    # ONLY the wrapper's explicit video slot — no phantom background/logo/
    # image/headline from artwork ids.
    kinds = [s.kind for s in contract.slots]
    assert kinds.count("video") == 1
    assert "background" not in kinds and "logo" not in kinds
    assert "image" not in kinds
    assert not any(s.kind == "text" for s in contract.slots)
    # And the artwork itself carries no engine markers after normalization.
    import re as _re
    svg_part = _norm[_norm.index("<svg"):]
    assert not _re.search(r'data-kaizer\s*=', svg_part)


def test_wrapper_text_slots_get_distinct_kinds():
    svg = _svg(
        '<rect id="video-slot" x="0" y="0" width="1000" height="1000"/>'
        + "".join(f'<rect id="text-slot-{i}" x="0" y="{1000 + i * 150}" '
                  f'width="1000" height="100"/>' for i in range(1, 7))
    )
    parsed = parse_svg_template(svg)
    html = svg_wrap.build_wrapper_html(
        parsed, svg_wrap.sanitize_svg(svg), name="six texts")
    for key in ('data-kaizer="headline"', 'data-kaizer="subtitle"',
                'data-kaizer="caption"', 'data-kaizer="body"',
                'data-kaizer="text:extra5"', 'data-kaizer="text:extra6"'):
        assert key in html, key


def test_canvas_clamp_preserves_aspect():
    # 20:1 viewBox — naive per-dimension clamp would break the aspect.
    svg = _svg('<rect id="video-slot" x="0" y="0" width="20000" height="1000"/>',
               vb="0 0 20000 1000")
    w, h = svg_wrap.canvas_size_for(parse_svg_template(svg))
    assert w == 4096 and abs(w / h - 20.0) < 0.2


def test_image_slot_div_has_cover_styling():
    svg = _svg(
        '<rect id="video-slot" x="0" y="0" width="1000" height="1400"/>'
        '<rect id="image-slot" x="100" y="1500" width="800" height="400"/>')
    parsed = parse_svg_template(svg)
    html = svg_wrap.build_wrapper_html(
        parsed, svg_wrap.sanitize_svg(svg), name="img")
    assert "background-size:cover" in html
    assert "background-position:center" in html


def test_wrapper_is_consumable_by_real_contract_discovery():
    """LOAD-BEARING: the generated entry must be understood by the SAME
    services/custom_templates discovery the upload router + render path
    run — video slot found, canvas right, kind derived from aspect."""
    from services import custom_templates as ct
    parsed = parse_svg_template(_full_svg())
    clean = svg_wrap.sanitize_svg(_full_svg())
    html = svg_wrap.build_wrapper_html(parsed, clean, name="My SVG")
    _norm, contract = ct.normalize_and_discover(html)
    assert (contract.canvas_w, contract.canvas_h) == (1080, 2160)
    assert ct.aspect_kind(contract.canvas_w, contract.canvas_h) == "short"
    assert contract.video_slots                    # renderable → status ready
    kinds = {s.kind for s in contract.slots}
    assert "video" in kinds and "image" in kinds and "text" in kinds
