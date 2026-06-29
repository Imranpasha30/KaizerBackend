"""Integration self-test: render a short through the LIVE render_short('custom:<id>')
path (DB lookup -> bundle -> render -> composite). Seeds a temporary CustomTemplate row
(deleted afterwards). Run with the backend venv from KaizerBackend/:

    ../../venv/Scripts/python.exe -m services.custom_templates._selftest_pipeline
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "samples" / "breaking_news"


def main() -> int:
    from database import Base, SessionLocal, engine
    import models
    from services import custom_templates as ct
    from services.custom_templates._selftest import _find_clip
    from pipeline_v4 import v1_bridge

    Base.metadata.create_all(bind=engine)            # ensure custom_templates table

    work = Path(tempfile.mkdtemp(prefix="kx_pipe_"))
    zpath = work / "bundle.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for dp, _d, files in os.walk(SAMPLE):
            for f in files:
                full = os.path.join(dp, f)
                zf.write(full, os.path.relpath(full, SAMPLE).replace("\\", "/"))
    bundle, contract = ct.prepare_bundle(str(zpath), str(work / "tmpl" / "bundle"))

    db = SessionLocal()
    row = models.CustomTemplate(
        owner_id=None, name="selftest", slug=f"selftest-{os.getpid()}",
        visibility="private", status="ready", dir_path=bundle.root_dir,
        entry_rel=bundle.entry_rel, canvas_w=contract.canvas_w, canvas_h=contract.canvas_h,
        contract_json=contract.to_json(),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    tid = row.id
    print("seeded CustomTemplate id:", tid)

    try:
        clip = _find_clip(str(work))
        out = str(work / "short_custom.mp4")
        inputs = v1_bridge.ShortRenderInputs(
            trimmed_short_path=clip,
            title_text="GDP grows 8% this quarter",
            output_path=out, work_dir=work,
            layout=f"custom:{tid}", language="te",
        )
        v1_bridge.render_short(inputs)
        ok = os.path.isfile(out)
        print("rendered via render_short(custom):", ok,
              "size:", os.path.getsize(out) if ok else 0)
        ff = shutil.which("ffmpeg") or "ffmpeg"
        frame = str(work / "frame.png")
        subprocess.run([ff, "-y", "-i", out, "-ss", "2", "-frames:v", "1", frame],
                       capture_output=True, timeout=120)
        print("FRAME_PATH", frame, "exists:", os.path.isfile(frame))
    finally:
        db.delete(row)
        db.commit()
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
