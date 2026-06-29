"""Verify logo-slot injection: a template that marks data-kaizer="logo" gets the brand
logo placed AT that slot (not the default corner).

    ../../venv/Scripts/python.exe -m services.custom_templates._selftest_logo
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

_TEMPLATE = """<!doctype html><html><head>
<meta charset="utf-8"><meta name="kaizer:canvas" content="1080x1920">
<style>*{margin:0}html,body{width:1080px;height:1920px;background:#0b0f1a}
.clip{position:absolute;inset:220px 60px 60px 60px;border-radius:28px}
.logo{position:absolute;top:48px;left:48px;width:170px;height:170px}
h1{position:absolute;top:90px;left:250px;color:#fff;font:700 56px sans-serif}</style></head>
<body>
<img class="logo" data-kaizer="logo">
<h1 data-kaizer="headline">Headline</h1>
<video class="clip" data-kaizer="video"></video>
</body></html>"""


def main() -> int:
    from PIL import Image, ImageDraw
    from services.custom_templates import RenderRequest, prepare_bundle, render_template
    from services.custom_templates._selftest import _find_clip

    work = Path(tempfile.mkdtemp(prefix="kx_logo_"))
    clip = _find_clip(str(work))

    logo = work / "logo.png"
    im = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([8, 8, 292, 292], fill=(255, 210, 63, 255), outline=(225, 29, 42, 255), width=14)
    d.text((150, 140), "LOGO", anchor="mm", fill=(11, 15, 26, 255))
    im.save(logo)

    tdir = work / "tmpl"
    tdir.mkdir()
    (tdir / "index.html").write_text(_TEMPLATE, encoding="utf-8")

    bundle, contract = prepare_bundle(str(tdir / "index.html"), str(work / "b"))
    print("logo slots:", [s.key for s in contract.slots if s.kind == "logo"])

    out = str(work / "out.mp4")
    render_template(
        bundle, contract,
        RenderRequest(videos={"video": clip}, texts={"headline": "Logo in slot"},
                      logo_path=str(logo), fps=24, duration=3.0),
        work_dir=str(work / "r"), out_path=out,
    )
    ff = shutil.which("ffmpeg") or "ffmpeg"
    frame = str(work / "frame.png")
    subprocess.run([ff, "-y", "-ss", "1", "-i", out, "-frames:v", "1", frame],
                   capture_output=True, timeout=120)
    print("FRAME", frame, "exists:", os.path.isfile(frame))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
