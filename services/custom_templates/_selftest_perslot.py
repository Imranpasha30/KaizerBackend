"""Verify per-slot media through render_short('custom:N'): a 2-video-slot template gets
TWO DIFFERENT clips (main slot = AI-trim clip; other slot = user-provided media).

    ../../venv/Scripts/python.exe -m services.custom_templates._selftest_perslot
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "samples" / "live_bulletin"          # video:left + video:right


def main() -> int:
    from database import Base, SessionLocal, engine
    import models
    from services import custom_templates as ct
    from services.custom_templates._selftest import _find_clip
    from pipeline_v4 import v1_bridge

    Base.metadata.create_all(bind=engine)
    work = Path(tempfile.mkdtemp(prefix="kx_perslot_"))

    clip_main = _find_clip(str(work))                 # talking head (main → trimmed)
    repo = HERE.parents[3]
    clip_other = str(repo / "kaizer-fx" / "out" / "input.mp4")  # colorbars (other slot)
    print("main clip:", clip_main)
    print("other clip:", clip_other, "exists:", os.path.isfile(clip_other))

    zp = work / "b.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        for dp, _d, fs in os.walk(SAMPLE):
            for f in fs:
                full = os.path.join(dp, f)
                zf.write(full, os.path.relpath(full, SAMPLE).replace("\\", "/"))
    bundle, contract = ct.prepare_bundle(str(zp), str(work / "bundle"))
    print("video slots:", [s.key for s in contract.video_slots])

    db = SessionLocal()
    row = models.CustomTemplate(owner_id=None, name="perslot", slug=f"perslot-{os.getpid()}",
                                visibility="private", status="ready", dir_path=bundle.root_dir,
                                entry_rel=bundle.entry_rel, canvas_w=contract.canvas_w,
                                canvas_h=contract.canvas_h, contract_json=contract.to_json())
    db.add(row); db.commit(); db.refresh(row); tid = row.id

    try:
        out = str(work / "out.mp4")
        inputs = v1_bridge.ShortRenderInputs(
            trimmed_short_path=clip_main,
            title_text="Per-slot media test",
            output_path=out, work_dir=work,
            layout=f"custom:{tid}", language="te",
            template_media={"right": clip_other},      # other slot = colorbars
            main_media_slot="left",                     # main slot = trimmed talking-head
        )
        v1_bridge.render_short(inputs)
        ok = os.path.isfile(out)
        print("rendered:", ok, "size:", os.path.getsize(out) if ok else 0)
        ff = shutil.which("ffmpeg") or "ffmpeg"
        frame = str(work / "frame.png")
        subprocess.run([ff, "-y", "-ss", "1", "-i", out, "-frames:v", "1", frame],
                       capture_output=True, timeout=120)
        print("FRAME", frame, "exists:", os.path.isfile(frame))
    finally:
        db.delete(row); db.commit(); db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
