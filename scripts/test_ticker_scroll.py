"""End-to-end: a custom template with a ticker slot renders a SCROLLING (marquee) ticker.

Renders a 1920x1080 template (video slot + yellow ticker bar) through the real custom-template
engine in still mode, then proves the ticker MOVES by extracting two frames a couple seconds
apart and asserting the ticker band differs (text scrolled) and is non-empty (text present).

Run:  venv/Scripts/python.exe -m scripts.test_ticker_scroll
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TICKER_Y, TICKER_H = 970, 70   # bottom:40 + height:70 on a 1080 canvas

_HTML = """<!DOCTYPE html><html><head><meta charset="UTF-8">
<meta name="kaizer:canvas" content="1920x1080">
<style>html,body{width:1920px;height:1080px;margin:0;background:#0a1326;}
.v{position:absolute;left:60px;top:80px;width:1000px;height:560px;background:#000;}
.tk{position:absolute;left:0;right:0;bottom:40px;height:70px;background:#ffd23f;color:#0a1326;
   display:flex;align-items:center;font-size:30px;font-weight:800;white-space:nowrap;
   overflow:hidden;padding:0 24px;}</style></head><body>
<div class="v" data-kaizer="video"></div>
<div class="tk" data-kaizer="ticker">placeholder ticker</div>
</body></html>"""


def _make_clip(path: str) -> None:
    ff = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run([ff, "-y", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30:duration=4",
                    "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-shortest",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", path],
                   capture_output=True, timeout=120)


def _frame(mp4: str, t: float, out_png: str) -> bool:
    ff = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run([ff, "-y", "-ss", str(t), "-i", mp4, "-frames:v", "1", out_png],
                   capture_output=True, timeout=120)
    return os.path.isfile(out_png)


def main() -> int:
    import services.custom_templates as ct
    from PIL import Image, ImageChops

    tmp = tempfile.mkdtemp(prefix="kx_tick_")
    print("work:", tmp)
    with open(os.path.join(tmp, "tpl.html"), "w", encoding="utf-8") as fh:
        fh.write(_HTML)
    clip = os.path.join(tmp, "clip.mp4")
    _make_clip(clip)
    assert os.path.isfile(clip), "test clip not made"

    bundle = ct.Bundle(root_dir=tmp, entry_rel="tpl.html", files=[])
    _norm, contract = ct.normalize_and_discover(_HTML)
    out = os.path.join(tmp, "out.mp4")
    req = ct.RenderRequest(
        videos={"video": clip}, texts={"ticker": "SCROLLING TICKER TEST ★ MARQUEE MOVES ★ KAIZER X"},
        main_slot="video", fps=30,
        ticker_text="SCROLLING TICKER TEST ★ MARQUEE MOVES ★ KAIZER X",
    )
    report = ct.render_template(bundle, contract, req, work_dir=os.path.join(tmp, "r"), out_path=out)
    print("report.ticker:", report.get("ticker"), "| out exists:", os.path.isfile(out),
          "| size:", (os.path.getsize(out) if os.path.isfile(out) else 0))
    assert report.get("ticker") is True, "engine did NOT apply a scrolling ticker overlay"
    assert os.path.isfile(out) and os.path.getsize(out) > 2048, "no output video"

    fa, fb = os.path.join(tmp, "a.png"), os.path.join(tmp, "b.png")
    assert _frame(out, 0.5, fa) and _frame(out, 2.8, fb), "frame grab failed"
    a = Image.open(fa).convert("RGB"); b = Image.open(fb).convert("RGB")
    # crop the ticker band (full width) at both timestamps
    box = (0, TICKER_Y - 6, a.width, TICKER_Y + TICKER_H + 6)
    ca, cb = a.crop(box), b.crop(box)
    diff = ImageChops.difference(ca, cb)
    changed = sum(diff.convert("L").histogram()[16:])   # pixels differing > ~16
    # the band must also contain TEXT (not a flat yellow bar): variance within one frame
    extrema = ca.convert("L").getextrema()
    print(f"ticker band changed pixels (t=0.5 vs 2.8) = {changed}; band luma extrema = {extrema}")
    assert changed > 3000, "ticker did NOT scroll — band identical across time (still, not marquee)"
    assert (extrema[1] - extrema[0]) > 40, "ticker band looks blank (no text drawn)"
    print("PASS ✓ ticker scrolls (marquee) and shows text")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("TICKER TEST ERROR:", repr(e))
        raise
