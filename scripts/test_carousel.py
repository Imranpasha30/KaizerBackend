"""End-to-end: a custom template with a CAROUSEL image slot renders a timed slideshow.

Builds a 1280x720 template (video slot + a data-kaizer-carousel image slot), feeds 3 solid-
colour frames (red/green/blue, 2s each, fade), renders, and asserts the slot region shows
red -> green -> blue over time (the slideshow advances + crossfades).

Run:  venv/Scripts/python.exe -m scripts.test_carousel
"""
from __future__ import annotations
import os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# image slot rect on a 1280x720 canvas
IX, IY, IW, IH = 120, 120, 440, 320

_HTML = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<meta name="kaizer:canvas" content="1280x720">
<style>html,body{{width:1280px;height:720px;margin:0;background:#0a1326;}}
.img{{position:absolute;left:{IX}px;top:{IY}px;width:{IW}px;height:{IH}px;background:#222;}}
.v{{position:absolute;left:700px;top:120px;width:440px;height:320px;background:#000;}}</style></head>
<body>
<div class="img" data-kaizer="image" data-kaizer-carousel="1"></div>
<div class="v" data-kaizer="video"></div>
</body></html>"""


def _solid(path, rgb):
    from PIL import Image
    Image.new("RGB", (IW, IH), rgb).save(path)


def _clip(path):
    ff = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run([ff, "-y", "-f", "lavfi", "-i", "color=c=gray:s=640x360:rate=30:d=6",
                    "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-shortest",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", path],
                   capture_output=True, timeout=120)


def _pixel(mp4, t, x, y):
    from PIL import Image
    ff = shutil.which("ffmpeg") or "ffmpeg"
    tmp = mp4 + f".f{int(t*10)}.png"
    subprocess.run([ff, "-y", "-ss", str(t), "-i", mp4, "-frames:v", "1", tmp],
                   capture_output=True, timeout=120)
    return Image.open(tmp).convert("RGB").getpixel((x, y))


def main() -> int:
    import services.custom_templates as ct
    tmp = tempfile.mkdtemp(prefix="kx_car_")
    with open(os.path.join(tmp, "t.html"), "w", encoding="utf-8") as fh:
        fh.write(_HTML)
    red, green, blue = (os.path.join(tmp, c + ".png") for c in ("r", "g", "b"))
    _solid(red, (220, 30, 30)); _solid(green, (30, 200, 30)); _solid(blue, (30, 30, 220))
    clip = os.path.join(tmp, "clip.mp4"); _clip(clip)

    b = ct.Bundle(root_dir=tmp, entry_rel="t.html", files=[])
    _n, contract = ct.normalize_and_discover(_HTML)
    car = [s for s in contract.slots if getattr(s, "carousel", False)]
    print("carousel slots discovered:", [(s.key, s.carousel) for s in car])
    out = os.path.join(tmp, "out.mp4")
    req = ct.RenderRequest(
        videos={"video": clip}, main_slot="video", fps=30,
        images={"image": {"carousel": [
            {"path": red,   "duration_s": 2.0, "effect": "fade",       "effect_duration": 0.4},
            {"path": green, "duration_s": 2.0, "effect": "slide_left", "effect_duration": 0.4},
            {"path": blue,  "duration_s": 2.0, "effect": "zoom_in",    "effect_duration": 0.4},
        ]}},
    )
    rep = ct.render_template(b, contract, req, work_dir=os.path.join(tmp, "r"), out_path=out)
    print("report carousels:", rep.get("carousels"), "| out exists:", os.path.isfile(out))
    assert rep.get("carousels") == 1, "engine did not composite a carousel"
    cx, cy = IX + IW // 2, IY + IH // 2
    p0 = _pixel(out, 0.7, cx, cy)    # frame 1 -> red
    p1 = _pixel(out, 2.7, cx, cy)    # frame 2 -> green
    p2 = _pixel(out, 4.7, cx, cy)    # frame 3 -> blue
    print(f"slot pixel @0.7s={p0} @2.7s={p1} @4.7s={p2}  (expect ~red, ~green, ~blue)")
    ok = (p0[0] > 130 and p0[0] > p0[1] and p0[0] > p0[2] and
          p1[1] > 110 and p1[1] > p1[0] and p1[1] > p1[2] and
          p2[2] > 110 and p2[2] > p2[0] and p2[2] > p2[1])
    print("PASS ✓ carousel advances red->green->blue" if ok else "FAIL: slideshow did not advance")
    assert ok
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("CAROUSEL TEST ERROR:", repr(e)); raise
