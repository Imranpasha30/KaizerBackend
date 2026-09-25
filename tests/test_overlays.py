"""UB.1 — overlay generator engine (12 families → the spec's ~190 assets)."""
from __future__ import annotations

from PIL import Image

from pipeline_v4 import overlays as ov


def _open(p):
    assert p is not None
    im = Image.open(p).convert("RGBA")
    return im


def test_banner_full_width_with_color_band(tmp_path):
    p = ov.banner(text="BREAKING NEWS", sub="Major development",
                  out_path=str(tmp_path / "b.png"))
    im = _open(p)
    assert im.width == 1920
    r, g, b, a = im.getpixel((900, 20))
    assert r > 150 and g < 80 and a > 200          # red strip


def test_bug_with_live_dot(tmp_path):
    p = ov.bug(text="LIVE", dot=True, out_path=str(tmp_path / "l.png"))
    im = _open(p)
    assert im.height == 52
    reds = [im.getpixel((x, 26)) for x in range(10, 40)]
    assert any(px[0] > 150 and px[1] < 80 for px in reds)   # the dot


def test_stamp_rotated_transparent_corners(tmp_path):
    p = ov.stamp(text="EXCLUSIVE", out_path=str(tmp_path / "s.png"))
    im = _open(p)
    # rotation-expanded corner is (near-)transparent; bicubic resampling
    # may bleed a few alpha counts into the very corner
    assert im.getpixel((0, 0))[3] < 40
    assert im.width > 200


def test_viewfinder_and_cctv_huds(tmp_path):
    v = _open(ov.frame_hud(style="viewfinder", out_path=str(tmp_path / "v.png")))
    assert v.size == (1920, 1080)
    # REC dot region has red
    region = [v.getpixel((x, y)) for x in range(80, 110) for y in range(90, 116)]
    assert any(px[0] > 150 and px[1] < 90 and px[3] > 200 for px in region)
    c = _open(ov.frame_hud(style="cctv", label="CAM-04",
                           out_path=str(tmp_path / "c.png")))
    # scanline present (semi-transparent dark line at y=0/6/12…)
    assert c.getpixel((960, 6))[3] > 0
    # center is transparent (video shows through). Scanlines are 2px
    # tall starting at multiples of 6, so probe y=303 (between lines).
    assert v.getpixel((960, 540))[3] == 0
    assert c.getpixel((960, 303))[3] == 0


def test_progress_and_countdown_and_panel(tmp_path):
    pr = _open(ov.progress(current=2, total=5, out_path=str(tmp_path / "p.png")))
    assert pr.size == (340, 64)
    cd = _open(ov.countdown(value="3", sub="stories tonight",
                            out_path=str(tmp_path / "cd.png")))
    assert cd.size == (1920, 1080)
    ip = _open(ov.info_panel(title="Key facts", lines=["a", "b"],
                             out_path=str(tmp_path / "i.png")))
    assert ip.width == 520


def test_telugu_text_renders(tmp_path):
    """Indic text must go through PIL fine (never drawtext)."""
    p = ov.banner(text="బ్రేకింగ్ న్యూస్", out_path=str(tmp_path / "te.png"))
    assert p is not None


def test_new_generators_and_hud_styles(tmp_path):
    """UG.4: score/weather/market/poll/social/tag + 3 new HUD styles."""
    assert _open(ov.score_bug(team_a="IND", score_a="187/5", team_b="AUS",
                              score_b="142/7", out_path=str(tmp_path / "sc.png"))).width == 620
    assert ov.weather_chip(city="Vizag", temp="29°C", cond="Heavy rain",
                           out_path=str(tmp_path / "wx.png"))
    assert ov.market_chip(name="SENSEX", value="81,250", change="-610",
                          out_path=str(tmp_path / "mk.png"))
    assert ov.poll_bar(question="Fair verdict?", yes_pct=51,
                       out_path=str(tmp_path / "pl.png"))
    assert ov.social_card(handle="@CityPolice", text="Traffic diverted at NH-44.",
                          out_path=str(tmp_path / "so.png"))
    assert ov.headline_tag(text="FACT CHECK", out_path=str(tmp_path / "tg.png"))
    for style in ("drone", "bodycam", "dashcam"):
        hud = _open(ov.frame_hud(style=style, out_path=str(tmp_path / f"{style}.png")))
        assert hud.size == (1920, 1080), style


def test_registry_rows_all_render(tmp_path):
    assert len(ov.REGISTRY) >= 190, f"only {len(ov.REGISTRY)} rows"
    ids = [r["id"] for r in ov.REGISTRY]
    assert len(set(ids)) == len(ids)
    for row in ov.REGISTRY:
        assert row["gen"] in ov.GENERATORS
        out = ov.render_registry_item(row["id"], str(tmp_path / f"{row['id']}.png"))
        assert out is not None, f"registry row {row['id']} failed to render"
    assert ov.render_registry_item("no_such_id", str(tmp_path / "x.png")) is None


def test_fail_soft_on_bad_input(tmp_path):
    # unknown HUD style → None, never raises
    assert ov.frame_hud(style="hologram", out_path=str(tmp_path / "h.png")) is None
    # unwritable path → None
    assert ov.banner(text="X", out_path=str(tmp_path / "nodir" / "x.png")) is None
