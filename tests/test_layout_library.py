"""UG.5 — layout library: 105 designed layouts + 15 PiP, all validated."""
from __future__ import annotations

from PIL import Image

from pipeline_v4 import layout_library as ll


def test_counts_hit_targets():
    fams = {}
    for l in ll.LAYOUTS.values():
        fams[l.family] = fams.get(l.family, 0) + 1
    non_pip = sum(v for k, v in fams.items() if k != "pip")
    assert non_pip >= 105, f"only {non_pip} layouts"
    assert fams.get("pip", 0) >= 15, f"only {fams.get('pip', 0)} PiP variants"
    keys = list(ll.LAYOUTS)
    assert len(keys) == len(set(keys))


def test_every_layout_is_a_complete_screen():
    """The 'not built blindly' guard: media + text + branding, in-bounds,
    text never buried under media — for EVERY layout."""
    bad = {k: e for k, l in ll.LAYOUTS.items() if (e := ll.validate_layout(l))}
    assert not bad, bad


def test_media_type_support_coverage():
    """Operator: layouts must support background video, background image,
    image carousels — everything."""
    kinds_used = set()
    for l in ll.LAYOUTS.values():
        kinds_used |= {z.kind for z in l.zones}
    for required in ("bg_video", "bg_image", "carousel", "image", "video",
                     "video_b", "chart", "map", "waveform", "pip",
                     "headline", "caption", "ticker", "logo", "watermark",
                     "cta", "panel"):
        assert required in kinds_used, f"no layout uses {required}"
    assert sum(1 for l in ll.LAYOUTS.values()
               if any(z.kind.startswith("bg_") for z in l.zones)) >= 12
    assert sum(1 for l in ll.LAYOUTS.values()
               if any(z.kind == "carousel" for z in l.zones)) >= 4


def test_aspect_families_present():
    aspects = {l.aspect for l in ll.LAYOUTS.values()}
    assert aspects == {"16:9", "9:16", "1:1"}
    assert sum(1 for l in ll.LAYOUTS.values() if l.aspect == "9:16") >= 25
    assert sum(1 for l in ll.LAYOUTS.values() if l.aspect == "1:1") >= 6


def test_every_preview_renders(tmp_path):
    for key, l in ll.LAYOUTS.items():
        p = ll.render_layout_preview(key, str(tmp_path / f"{key}.png"))
        assert p, f"{key} preview failed"
        im = Image.open(p)
        if l.aspect == "9:16":
            assert im.height > im.width, key
        elif l.aspect == "1:1":
            assert im.height == im.width, key
        else:
            assert im.width > im.height, key
    assert ll.render_layout_preview("no_such", str(tmp_path / "x.png")) is None


def test_renderable_subset_maps_to_canvas_pcts():
    """New Job picker chain: every RENDERABLE layout exports valid tile
    percentages the composer honours (video + picture, in-bounds)."""
    for key in ll.RENDERABLE:
        p = ll.to_canvas_pcts(key)
        assert p, key
        for f in ("video_x_pct", "video_y_pct", "video_w_pct", "video_h_pct",
                  "picture_x_pct", "picture_y_pct", "picture_w_pct",
                  "picture_h_pct"):
            assert 0.0 <= p[f] <= 100.0, (key, f)
        assert p["video_w_pct"] > 20 and p["video_h_pct"] > 20, key
    assert ll.to_canvas_pcts("sp_gaming_hud") is None      # not renderable
    assert ll.to_canvas_pcts("nope") is None


def test_library_endpoint_rows():
    from routers.custom_templates import list_library_layouts
    rows = list_library_layouts()
    assert len(rows) == len(ll.RENDERABLE)
    for r in rows:
        assert r["key"] in ll.RENDERABLE
        assert r["name"] and r["preview_url"].endswith("/preview")


def test_layout_to_html_bridge():
    """Fork bridge: eligible layouts generate contract-valid,
    builder-editable HTML (slots + canvas meta + CSS vars)."""
    elig = [k for k in ll.LAYOUTS if ll.fork_eligible(k)]
    assert len(elig) >= 100
    html = ll.layout_to_html("news_bulletin_right")
    for marker in ('data-kaizer="video"', 'data-kaizer="image"',
                   'data-kaizer="ticker"', 'data-kaizer="headline"',
                   'data-kaizer="logo"', 'meta name="kaizer:canvas" content="1920x1080"',
                   "--kaizer-brand"):
        assert marker in html, marker
    v = ll.layout_to_html("short_dual_video")
    assert v.count('data-kaizer="video"') == 2      # both cams are slots
    assert 'content="1080x1920"' in v               # 9:16 canvas
    bg = ll.layout_to_html("studio_bg_classic")
    assert 'data-kaizer="background"' in bg
    assert ll.layout_to_html("no_such") is None
