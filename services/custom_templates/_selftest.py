"""Standalone end-to-end self-test for the custom-template engine.

Zips the bundled 'breaking_news' sample, runs the full pipeline (extract -> sanitize ->
discover -> sandboxed render -> ffmpeg composite) on a real clip, and reports. Run with
the backend venv:

    venv/Scripts/python.exe -m services.custom_templates._selftest

Set KAIZER_TEST_CLIP to point at your own source video.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "samples", "breaking_news")


def _zip_sample(dest_zip: str) -> str:
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for dp, _d, files in os.walk(SAMPLE):
            for f in files:
                full = os.path.join(dp, f)
                rel = os.path.relpath(full, SAMPLE).replace("\\", "/")
                zf.write(full, rel)
    return dest_zip


def _find_clip(work: str) -> str:
    env = os.environ.get("KAIZER_TEST_CLIP")
    if env and os.path.isfile(env):
        src = env
    else:
        repo = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
        cands = [
            os.path.join(repo, "kaizer-fx", "out", "web", "9bf2088de316_in.mp4"),
            os.path.join(repo, "kaizer-fx", "out", "input.mp4"),
        ]
        src = next((c for c in cands if os.path.isfile(c)), "")
        if not src:
            raise SystemExit("No test clip found. Set KAIZER_TEST_CLIP=<path to a video>.")
    clip = os.path.join(work, "clip8s.mp4")
    ff = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run([ff, "-y", "-i", src, "-t", "8", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", clip], capture_output=True, timeout=300)
    return clip if os.path.isfile(clip) else src


def main() -> int:
    from . import prepare_bundle, render_template, RenderRequest

    work = tempfile.mkdtemp(prefix="kx_tmpl_")
    print("work dir:", work)
    clip = _find_clip(work)
    print("test clip:", clip)

    zip_path = _zip_sample(os.path.join(work, "bundle.zip"))
    bundle_dir = os.path.join(work, "bundle")
    b, contract = prepare_bundle(zip_path, bundle_dir)
    print("canvas:", contract.canvas_w, "x", contract.canvas_h)
    print("slots:", [(s.kind, s.key) for s in contract.slots])
    if contract.warnings:
        print("warnings:", contract.warnings)

    out = os.path.join(work, "out.mp4")
    req = RenderRequest(
        videos={"video": clip},
        texts={"headline": "GDP grows 8% this quarter", "hook": "Breaking: economy update"},
        brand={"--kaizer-brand": "#1d4ed8", "--kaizer-accent": "#22d3ee"},
        fps=30,
    )
    report = render_template(b, contract, req, work_dir=os.path.join(work, "render"), out_path=out)
    print("REPORT:", report)
    print("OUTPUT:", out, "exists:", os.path.isfile(out),
          "size:", (os.path.getsize(out) if os.path.isfile(out) else 0))
    # frame grab for visual verification
    ff = shutil.which("ffmpeg") or "ffmpeg"
    frame = os.path.join(work, "frame.png")
    subprocess.run([ff, "-y", "-i", out, "-ss", "2", "-frames:v", "1", frame],
                   capture_output=True, timeout=120)
    print("FRAME:", frame, "exists:", os.path.isfile(frame))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
