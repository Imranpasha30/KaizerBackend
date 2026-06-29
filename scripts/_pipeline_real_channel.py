"""Pass a real master video through the REAL branding pipeline using a REAL
channel's resolved brand (logo + watermark) via the GPU overlay — and DO NOT
publish. Writes the branded output + a frame for inspection. Read-only on the DB.
"""
import os
import sys
import time
import subprocess
import tempfile
import shutil

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BACKEND)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from dotenv import load_dotenv
load_dotenv(os.path.join(_BACKEND, ".env"))
from database import SessionLocal
import models
from services import branding as B
from services.brand_resolver import resolve_brand_profile

MASTER = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    _BACKEND, "output", "full_video_shorts_v4", "job_160", "bulletin.mp4")
OUTDIR = r"e:\kaizer new data training\.tmp_pipeline_out"
os.makedirs(OUTDIR, exist_ok=True)

db = SessionLocal()
try:
    names = {c.id: c.name for c in db.query(models.Channel).all()}
    # Prefer a channel that actually has a LOGO; fall back to any with branding.
    picked, resolved = None, None
    pref = [17] + sorted(names.keys())
    seen = set()
    for cid in pref:
        if cid in seen:
            continue
        seen.add(cid)
        try:
            r = resolve_brand_profile(db, int(cid))
        except Exception:
            continue
        if r and (getattr(r, "logo_local_path", None) or getattr(r, "watermark_text", None)):
            picked, resolved = cid, r
            if getattr(r, "logo_local_path", None):
                break
    if not resolved:
        print("No channel with a resolvable brand profile found."); sys.exit(2)

    print(f"CHANNEL: #{picked} {names.get(picked)!r}")
    print(f"  logo_local_path = {resolved.logo_local_path!r}")
    print(f"  watermark = {resolved.watermark_text!r} pos={resolved.watermark_position!r} "
          f"opacity={resolved.watermark_opacity!r}  brand.version={resolved.version!r}")

    cw, ch, dur = B._probe_source(MASTER)
    print(f"MASTER: {os.path.basename(MASTER)}  {cw}x{ch}  {dur:.1f}s")

    wd = tempfile.mkdtemp(prefix="realchan_")
    try:
        bug = (B._prepare_logo_png(resolved.logo_local_path, cw, ch, wd)
               if resolved.logo_local_path else None)
        out = os.path.join(OUTDIR, "branded_real_channel.mp4")
        kwargs = dict(
            source_path=MASTER, out_path=out, work_dir=wd,
            apply_logo=bool(bug), apply_text=bool(resolved.watermark_text),
            logo_local_path=bug, text=(resolved.watermark_text or ""),
            opacity=(resolved.watermark_opacity or 0.85),
            position=(resolved.watermark_position or "lower-center"),
            canvas_w=cw, canvas_h=ch, source_duration=dur,
        )
        gpu_on = os.environ.get("KAIZER_BRAND_GPU_OVERLAY", "0").strip() == "1" and B._gpu_overlay_caps()
        t = time.time()
        if gpu_on:
            B._run_ffmpeg(B._build_gpu_ffmpeg_command(**kwargs), timeout=900)
            path_used = "GPU overlay_cuda"
        else:
            B._run_ffmpeg(B._build_ffmpeg_command(**kwargs), timeout=1200)
            path_used = "CPU overlay"
        dt = time.time() - t

        # Real QC gate
        qc = "PASS"
        try:
            B._qc_branded_artifact(MASTER, out)
        except Exception as e:
            qc = f"FAIL: {e}"
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=codec_name,width,height",
                            "-show_entries", "format=duration", "-of", "csv=p=0", out],
                           capture_output=True, text=True)
        # Frame at 5s for visual inspection
        frame = os.path.join(OUTDIR, "branded_real_channel_frame.png")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "5", "-i", out,
                        "-vframes", "1", frame])
        print(f"\nPATH USED: {path_used}")
        print(f"ENCODE TIME: {dt:.1f}s")
        print(f"OUTPUT: {out}  ({os.path.getsize(out)//1024//1024}MB)")
        print(f"PROBE: {r.stdout.strip()}")
        print(f"QC: {qc}")
        print(f"FRAME: {frame}")
        print("NOT PUBLISHED (branding-only; no upload job, no YouTube/Postiz call).")
    finally:
        shutil.rmtree(wd, ignore_errors=True)
finally:
    db.close()
