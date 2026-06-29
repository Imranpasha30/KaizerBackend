"""Animated self-test: render the live_bulletin sample (16:9, 2 videos, blinking dot,
scrolling marquee) and dump frames at several timestamps to verify MOTION was captured.

    ../../venv/Scripts/python.exe -m services.custom_templates._selftest_animated
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "samples" / "live_bulletin"


def main() -> int:
    from services.custom_templates import prepare_bundle, render_template, RenderRequest
    from services.custom_templates._selftest import _find_clip

    work = Path(tempfile.mkdtemp(prefix="kx_anim_"))
    print("work:", work)
    clip = _find_clip(str(work))

    zp = work / "bundle.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        for dp, _d, files in os.walk(SAMPLE):
            for f in files:
                full = os.path.join(dp, f)
                zf.write(full, os.path.relpath(full, SAMPLE).replace("\\", "/"))
    bundle, contract = prepare_bundle(str(zp), str(work / "bundle"))
    print("canvas:", contract.canvas_w, "x", contract.canvas_h)
    print("slots:", [(s.kind, s.key) for s in contract.slots])

    out = str(work / "bulletin.mp4")
    req = RenderRequest(
        videos={"video": clip},                     # fills both video:left + video:right (fallback)
        texts={"headline": "GDP grows 8% this quarter", "hook": "STUDIO"},
        brand={"--kaizer-brand": "#1d4ed8", "--kaizer-accent": "#22d3ee"},
        fps=20, duration=6.0,
    )
    report = render_template(bundle, contract, req, work_dir=str(work / "r"), out_path=out)
    print("REPORT:", report)

    ff = shutil.which("ffmpeg") or "ffmpeg"
    frames = []
    for t in (0.2, 1.5, 3.5):
        fp = str(work / f"frame_t{t}.png")
        subprocess.run([ff, "-y", "-ss", str(t), "-i", out, "-frames:v", "1", fp],
                       capture_output=True, timeout=120)
        frames.append(fp)
        print("FRAME", t, "->", fp, "exists:", os.path.isfile(fp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
