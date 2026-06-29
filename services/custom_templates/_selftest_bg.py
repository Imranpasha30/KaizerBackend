"""Verify background slot: a full-frame data-kaizer="background" video sits BEHIND a
foreground video tile (different clips). Run:

    ../../venv/Scripts/python.exe -m services.custom_templates._selftest_bg
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
_TPL = """<!doctype html><html><head>
<meta charset="utf-8"><meta name="kaizer:canvas" content="1080x1920">
<style>*{margin:0}html,body{width:1080px;height:1920px;background:#0b0f1a;font-family:sans-serif}
.bg{position:absolute;inset:0}
.tile{position:absolute;left:140px;top:560px;width:800px;height:800px;border-radius:32px;
  border:4px solid #fff;box-shadow:0 30px 80px rgba(0,0,0,.6)}
h1{position:absolute;top:80px;left:0;right:0;text-align:center;color:#fff;font-size:64px;font-weight:800}</style></head>
<body>
<div class="bg" data-kaizer="background"></div>
<h1 data-kaizer="headline">Background test</h1>
<video class="tile" data-kaizer="video"></video>
</body></html>"""


def main() -> int:
    from services.custom_templates import RenderRequest, prepare_bundle, render_template
    from services.custom_templates._selftest import _find_clip

    work = Path(tempfile.mkdtemp(prefix="kx_bg_"))
    main_clip = _find_clip(str(work))                      # talking head (foreground tile)
    repo = HERE.parents[3]
    bg_clip = str(repo / "kaizer-fx" / "out" / "input.mp4")  # colorbars (full-frame bg)

    tdir = work / "tmpl"; tdir.mkdir()
    (tdir / "index.html").write_text(_TPL, encoding="utf-8")
    bundle, contract = prepare_bundle(str(tdir / "index.html"), str(work / "b"))
    print("slots:", [(s.kind, s.key) for s in contract.slots])

    out = str(work / "out.mp4")
    rep = render_template(
        bundle, contract,
        RenderRequest(videos={"video": main_clip, "background": bg_clip},
                      texts={"headline": "Background test"}, main_slot="video",
                      fps=24, duration=3.0),
        work_dir=str(work / "r"), out_path=out)
    print("REPORT:", rep)
    ff = shutil.which("ffmpeg") or "ffmpeg"
    frame = str(work / "frame.png")
    subprocess.run([ff, "-y", "-ss", "1", "-i", out, "-frames:v", "1", frame], capture_output=True, timeout=120)
    print("FRAME", frame, "exists:", os.path.isfile(frame))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
