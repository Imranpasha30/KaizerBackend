"""Premium karaoke caption variants — kinetic / chips / glow / flip / impact.

Every new variant must render real caption-state PNGs for an English AND
a Telugu line (the Indic-safe path is the whole point), be registered in
VARIANTS, and auto-enter the Director caption vocabulary (director._vocab
derives it from engine=="karaoke" rows — zero director changes).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from pipeline_v4 import typography as ty

NEW_IDS = ["karaoke_kinetic_yellow", "karaoke_chips_red",
           "karaoke_glow_gold", "karaoke_flip_duo", "karaoke_impact_promo"]

TELUGU_FONT = str(Path(__file__).resolve().parents[1]
                  / "resources" / "fonts" / "NotoSansTelugu-Bold.ttf")


def _w(word, s, e):
    return {"w": word, "s": s, "e": e}


# one caption line each (<=4 words, <=26 chars, no >1s gaps)
EN_WORDS = [_w("Hello", 0.0, 0.4), _w("brave", 0.5, 0.9), _w("world", 1.0, 1.4)]
TE_WORDS = [_w("మోదీ", 0.0, 0.4), _w("పార్లమెంట్", 0.5, 0.9), _w("లో", 1.0, 1.4)]


def test_premium_variants_registered():
    rows = {v["id"]: v for v in ty.VARIANTS}
    for vid in NEW_IDS:
        assert vid in rows, f"{vid} missing from VARIANTS"
        v = rows[vid]
        # engine=="karaoke" is what feeds director._vocab — a typo here
        # would silently drop the variant from the AI's vocabulary
        assert v["engine"] == "karaoke", vid
        assert v["label"] and v["used_for"], vid
        assert v["kwargs"].get("style") and v["kwargs"].get("hilite"), vid
    ids = [v["id"] for v in ty.VARIANTS]
    assert len(ids) == len(set(ids))             # no id collisions
    assert set(NEW_IDS) <= {c["id"] for c in ty.CATALOG}   # admin tab rows


def test_premium_variants_in_director_vocab():
    from pipeline_v4.director import _vocab
    assert set(NEW_IDS) <= _vocab()["captions"]


@pytest.mark.parametrize("vid", NEW_IDS)
def test_premium_variant_renders_english_and_telugu(vid, tmp_path):
    kw = dict(next(v for v in ty.VARIANTS if v["id"] == vid)["kwargs"])
    for tag, words, font in (("en", EN_WORDS, None),
                             ("te", TE_WORDS, TELUGU_FONT)):
        wins = ty.karaoke_captions(words, out_dir=tmp_path, font_path=font,
                                   prefix=f"{vid}_{tag}", **kw)
        # the engine is fail-soft ([] on error) — an empty result here
        # means the variant silently broke, exactly what must not ship
        assert len(wins) >= len(words), f"{vid}/{tag} rendered nothing"
        for p, a, b in wins:
            assert os.path.isfile(p) and os.path.getsize(p) > 0, f"{vid}/{tag}"
            assert b > a
        im = Image.open(wins[0][0]).convert("RGBA")
        assert im.width > 100 and im.height > 40, f"{vid}/{tag}"
        assert im.getchannel("A").getbbox() is not None   # not fully blank
        # the active word moves → frames must be visually distinct
        first = Image.open(wins[0][0]).convert("RGBA").tobytes()
        last = Image.open(wins[-1][0]).convert("RGBA").tobytes()
        assert first != last, f"{vid}/{tag} frames identical"


def test_premium_styles_visually_distinct(tmp_path):
    blobs = set()
    for vid in NEW_IDS:
        kw = dict(next(v for v in ty.VARIANTS if v["id"] == vid)["kwargs"])
        wins = ty.karaoke_captions(EN_WORDS, out_dir=tmp_path,
                                   prefix=f"d_{vid}", **kw)
        # last window = the SETTLED state of the last word for every
        # style (kinetic prepends blink frames, so index from the end)
        blobs.add(Image.open(wins[-1][0]).convert("RGBA").tobytes())
    assert len(blobs) == len(NEW_IDS)


def test_flip_alternates_accent_per_word(tmp_path):
    # same token twice: any pixel difference between the two settled
    # frames can only come from the alternating accent colour
    words = [_w("నమస్తే", 0.0, 0.4), _w("నమస్తే", 0.5, 0.9)]
    wins = ty.karaoke_captions(words, out_dir=tmp_path, style="flip",
                               font_path=TELUGU_FONT, prefix="flipchk")
    assert len(wins) == 2

    def cyanish(p):
        im = Image.open(p).convert("RGBA")
        return sum(1 for r, g, b, a in im.getdata()
                   if a > 200 and r < 120 and g > 150 and b > 150)
    # word 0 = yellow accent (no cyan); word 1 = the hilite2 cyan accent
    assert cyanish(wins[0][0]) < 10
    assert cyanish(wins[1][0]) > 50


def test_premium_variant_admin_previews(tmp_path):
    """render_variant passes the row kwargs through **kw — this is the
    path that must accept the new hilite2 kwarg (admin Features tab)."""
    for vid in NEW_IDS:
        p = ty.render_variant(vid, tmp_path)
        assert p, f"{vid} preview rendered nothing"
        im = Image.open(p).convert("RGBA")
        assert im.width > 40 and im.height > 30, vid
        assert im.getchannel("A").getbbox() is not None, vid


def test_bridge_resolves_premium_ids():
    """v1_bridge._karaoke_params is data-driven from VARIANTS, so the
    new rows resolve without any bridge change (index the result rather
    than unpacking — tolerant of the mapping growing extra fields)."""
    from pipeline_v4 import v1_bridge as vb
    for vid in NEW_IDS:
        row = next(v for v in ty.VARIANTS if v["id"] == vid)
        res = vb._karaoke_params(vid)
        assert res[0] == row["kwargs"]["style"], vid
        assert tuple(res[1])[:3] == tuple(row["kwargs"]["hilite"])[:3], vid


def test_weight_zero_is_pixel_identical(tmp_path):
    """The weight kwarg must be a pure superset: weight=0 keeps every
    existing caller of _styled_text_img byte-identical (old per-story
    caches key on the directive spec, not render code — a drift here
    would hide behind them)."""
    from pipeline_v4.overlays import _font
    f = _font(None, 58)
    base = ty._styled_text_img("పరీక్ష TEST", f)
    again = ty._styled_text_img("పరీక్ష TEST", f, weight=0)
    assert base.tobytes() == again.tobytes()
    heavy = ty._styled_text_img("పరీక్ష TEST", f, weight=2)
    assert heavy.size != base.size or heavy.tobytes() != base.tobytes()
