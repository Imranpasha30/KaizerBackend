"""UB.2 — typography engine: karaoke captions, typewriter, headline crash."""
from __future__ import annotations

from PIL import Image

from pipeline_v4 import typography as ty


def _w(word, s, e):
    return {"w": word, "s": s, "e": e}


WORDS = [_w("మోదీ", 0.2, 0.6), _w("పార్లమెంట్", 0.7, 1.4), _w("లో", 1.5, 1.7),
         _w("ప్రసంగం", 1.8, 2.5),
         # 2s silence → new caption line
         _w("rescue", 4.6, 5.0), _w("teams", 5.1, 5.5), _w("deployed", 5.6, 6.2)]


def test_group_caption_lines_rules():
    lines = ty.group_caption_lines(WORDS, max_words=4, max_gap=1.0)
    assert len(lines) == 2                       # silence split
    assert [w["w"] for w in lines[0]] == ["మోదీ", "పార్లమెంట్", "లో", "ప్రసంగం"]
    assert [w["w"] for w in lines[1]] == ["rescue", "teams", "deployed"]
    # max_words forces a split too
    many = [_w(f"w{i}", i * 0.5, i * 0.5 + 0.4) for i in range(9)]
    assert len(ty.group_caption_lines(many, max_words=4)) == 3
    assert ty.group_caption_lines([]) == []


def test_karaoke_windows_tile_speech(tmp_path):
    wins = ty.karaoke_captions(WORDS, out_dir=tmp_path)
    assert len(wins) == len(WORDS)               # one PNG per word-state
    # windows are word-exact: first starts at 0.2; within a line each
    # window ends where the next word starts
    assert wins[0][1] == 0.2 and wins[0][2] == 0.7
    assert wins[3][2] == 2.5                     # last word holds to line end
    for p, a, b in wins:
        assert b > a
        im = Image.open(p).convert("RGBA")
        assert im.width > 100 and im.height > 60
    # highlighted word is visually different frame to frame
    a0 = Image.open(wins[0][0]).convert("RGBA").tobytes()
    a1 = Image.open(wins[1][0]).convert("RGBA").tobytes()
    assert a0 != a1


def test_typewriter_progression(tmp_path):
    wins = ty.typewriter("CASE FILE #42", out_dir=tmp_path, t_start=1.0)
    assert len(wins) == len("CASE FILE #42")
    assert wins[0][1] == 1.0
    # times strictly increase and the final frame holds
    assert wins[-1][2] - wins[-1][1] > 1.0
    sizes = [Image.open(p).size for p, _, _ in wins[:3]]
    assert len(set(sizes)) == 1                  # constant canvas, growing text
    assert ty.typewriter("", out_dir=tmp_path) == []


def test_headline_crash_spring_frames(tmp_path):
    wins = ty.headline_crash("పెద్ద వార్త", out_dir=tmp_path, frames=12)
    assert len(wins) == 12
    # settled frame holds
    assert wins[-1][2] - wins[-1][1] > 2.0
    # early frame is LARGER content (spring starts oversized): compare
    # opaque bounding boxes of first vs last frame
    def opaque_w(p):
        im = Image.open(p).convert("RGBA")
        alpha = im.getchannel("A")
        return alpha.getbbox()[2] - alpha.getbbox()[0]
    assert opaque_w(wins[0][0]) > opaque_w(wins[-1][0])


def test_fail_soft(tmp_path):
    assert ty.headline_crash("", out_dir=tmp_path) == []
    # unwritable dir → [] never raises
    import os
    bad = os.path.join(str(tmp_path), "no\0dir")
    assert ty.typewriter("x", out_dir=bad) == []


# ── UG.3: the 60-variant registry ────────────────────────────────────

def test_variant_registry_hits_sixty():
    assert len(ty.VARIANTS) >= 60, f"only {len(ty.VARIANTS)} variants"
    ids = [v["id"] for v in ty.VARIANTS]
    assert len(ids) == len(set(ids))             # unique ids
    assert len(ty.CATALOG) == len(ty.VARIANTS)
    for v in ty.VARIANTS:
        assert v["label"] and v["used_for"] and v["engine"]


def test_every_variant_renders(tmp_path):
    """EVERY catalog row must produce a real image — same guard as the
    ffmpeg chains: a variant that can't render never ships silently."""
    for v in ty.VARIANTS:
        p = ty.render_variant(v["id"], tmp_path)
        assert p, f"{v['id']} rendered nothing"
        im = Image.open(p).convert("RGBA")
        assert im.width > 40 and im.height > 30, v["id"]
        # not fully transparent
        assert im.getchannel("A").getbbox() is not None, v["id"]
    assert ty.render_variant("no_such_variant", tmp_path) is None


def test_karaoke_styles_differ(tmp_path):
    words = [_w("hello", 0.0, 0.4), _w("world", 0.5, 0.9)]
    imgs = {}
    for st in ("hilite", "box", "underline", "pop"):
        wins = ty.karaoke_captions(words, out_dir=tmp_path, style=st,
                                   prefix=f"st_{st}")
        imgs[st] = Image.open(wins[0][0]).convert("RGBA").tobytes()
    assert len(set(imgs.values())) == 4          # visually distinct


def test_headline_motions_differ(tmp_path):
    frames = {}
    for m in ("crash", "bounce", "whip_left", "spin_in"):
        wins = ty.headline_crash("TEST", out_dir=tmp_path, frames=8,
                                 motion=m, prefix=f"m_{m}")
        frames[m] = Image.open(wins[2][0]).convert("RGBA").tobytes()
    assert len(set(frames.values())) == 4
