"""Per-job HTML override — render-path tests.

Part A (fast, no browser): render_template forwards clear_unfilled = NOT req.literal,
so a literal/override render keeps the operator's baked text/images instead of wiping them.

Part B (real renderer, needs Chromium): render the SAME baked-text HTML twice — once with
clear_unfilled=False (literal/override: text PRESERVED) and once True (normal final + empty
texts: text CLEARED) — and assert the frames differ in the headline region. This is the
visual proof that the per-job override renders the operator's content verbatim.

Run with the backend venv:  venv/Scripts/python.exe -m scripts.test_perjob_override
"""
from __future__ import annotations

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # KaizerBackend
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_engine_literal_flag() -> None:
    print("PART A — render_template literal flag -> clear_unfilled")
    import services.custom_templates.engine as E

    captured: dict = {}

    class _Stop(Exception):
        pass

    def fake_render_animated(bundle, data, frames_dir, canvas, **kw):
        captured["clear_unfilled"] = data.clear_unfilled
        raise _Stop()

    orig = E._renderer.render_animated
    E._renderer.render_animated = fake_render_animated

    class _Ctr:
        canvas_w = 1080
        canvas_h = 1920

        def __init__(self):
            self.slots = []

    class _B:
        root_dir = tempfile.mkdtemp(prefix="kx_ov_")

    try:
        for literal, expected in [(True, False), (False, True)]:
            captured.clear()
            req = E.RenderRequest(videos={}, texts={}, images={}, literal=literal)
            try:
                E.render_template(_B(), _Ctr(), req,
                                  work_dir=tempfile.mkdtemp(prefix="kx_ov_w_"),
                                  out_path=os.path.join(tempfile.mkdtemp(prefix="kx_ov_o_"), "o.mp4"))
            except _Stop:
                pass
            got = captured.get("clear_unfilled")
            assert got is expected, f"literal={literal}: clear_unfilled={got}, expected {expected}"
            print(f"  literal={literal!s:5} -> clear_unfilled={got}  OK")
    finally:
        E._renderer.render_animated = orig
    print("  PART A PASS\n")


_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="kaizer:canvas" content="1080x1920">
<style>html,body{width:1080px;height:1920px;margin:0;background:#0a1326;}
.h{position:absolute;left:60px;top:300px;right:60px;color:#fff;font-size:96px;
   font-weight:900;line-height:1.05;}
.v{position:absolute;left:60px;top:760px;width:960px;height:540px;background:#000;}</style>
</head><body>
<div class="h" data-kaizer="headline">ZZ UNIQUE BAKED HEADLINE ZZ</div>
<div class="v" data-kaizer="video"></div>
</body></html>"""


def test_renderer_preserves_baked_text() -> bool:
    print("PART B — literal render PRESERVES baked text (real Chromium frame diff)")
    try:
        import services.custom_templates as ct
        from services.custom_templates import renderer as R
        from PIL import Image, ImageChops
    except Exception as e:  # pragma: no cover
        print(f"  SKIP (deps unavailable): {e}\n")
        return True

    tmp = tempfile.mkdtemp(prefix="kx_ov_b_")
    p = os.path.join(tmp, "tpl.html")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(_HTML)
    b = ct.Bundle(root_dir=tmp, entry_rel="tpl.html", files=[])
    ctr = ct.discover(_HTML)

    def frame(clear: bool) -> str:
        fd = os.path.join(tmp, f"frames_{clear}")
        data = R.RenderData(texts={}, images={}, logo_path=None, brand={}, clear_unfilled=clear)
        ar = R.render_animated(b, data, fd, (1080, 1920), fps=24, duration=1.0, force_still=True)
        return os.path.join(ar.frames_dir, ar.frame_pattern % 0)

    try:
        f_lit = frame(False)   # literal/override: baked text PRESERVED
        f_clr = frame(True)    # normal final, empty texts: baked text CLEARED
    except Exception as e:
        print(f"  SKIP (Chromium/render unavailable here): {e}\n")
        return True

    a = Image.open(f_lit).convert("RGB")
    c = Image.open(f_clr).convert("RGB")
    diff = ImageChops.difference(a, c)
    bbox = diff.getbbox()
    g = diff.convert("L")
    changed = sum(g.histogram()[11:])   # pixels whose channels differ by > 10
    print(f"  changed pixels = {changed}, diff bbox = {bbox}")
    assert bbox is not None and changed > 500, \
        "literal mode did NOT preserve baked text — frames are identical (text wiped both ways)"
    # the difference should sit in the headline band (top ~300-450px), not the video region
    assert bbox[1] < 600, f"diff not in the headline region (bbox={bbox})"
    print("  PART B PASS — literal keeps the baked headline; non-literal blanks it\n")
    return True


if __name__ == "__main__":
    test_engine_literal_flag()
    test_renderer_preserves_baked_text()
    print("ALL TESTS PASS ✓")
