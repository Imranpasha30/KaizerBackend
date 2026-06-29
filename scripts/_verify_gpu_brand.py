"""Verify the GPU brand-overlay path THROUGH the real branding code builders.
Compares GPU vs CPU on a real sample: timing, exact dims, QC pass, and writes
frames for a visual check. Cleans up. Zero DB/R2 — pure ffmpeg via branding._*.
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
from services import branding as B

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_BACKEND, "output", "full_video_shorts_v4", "job_160", "bulletin.mp4")
LOGO = os.path.join(_BACKEND, "assests", "kaizer-logo.png")
BENCH = r"e:\kaizer new data training\.tmp_bench"


def _dims(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of", "csv=p=0", p],
                       capture_output=True, text=True)
    return r.stdout.strip()


print("gpu_overlay_caps:", B._gpu_overlay_caps())
cw, ch, dur = B._probe_source(SRC)
print(f"source: {cw}x{ch} dur={dur:.1f}s")
wd = tempfile.mkdtemp(prefix="gpubrand_")
ok = True
try:
    bug = B._prepare_logo_png(LOGO, cw, ch, wd)
    kwargs = dict(source_path=SRC, work_dir=wd, apply_logo=True, apply_text=True,
                  logo_local_path=bug, text="Kaizer 30", opacity=0.85,
                  position="lower-center", canvas_w=cw, canvas_h=ch,
                  source_duration=dur)

    gpu_out = os.path.join(wd, "gpu.mp4")
    t = time.time()
    B._run_ffmpeg(B._build_gpu_ffmpeg_command(out_path=gpu_out, **kwargs), timeout=900)
    gt = time.time() - t

    cpu_out = os.path.join(wd, "cpu.mp4")
    t = time.time()
    B._run_ffmpeg(B._build_ffmpeg_command(out_path=cpu_out, **kwargs), timeout=1200)
    ct = time.time() - t

    gd, cd = _dims(gpu_out), _dims(cpu_out)
    want = f"{cw},{ch}"
    print(f"\nGPU: {gt:5.1f}s  dims={gd}  size={os.path.getsize(gpu_out)//1024//1024}MB")
    print(f"CPU: {ct:5.1f}s  dims={cd}  size={os.path.getsize(cpu_out)//1024//1024}MB")
    print(f"speedup: {ct/gt:.2f}x   want_dims={want}")

    if gd != want:
        print(f"  FAIL: GPU dims {gd} != source {want}"); ok = False
    else:
        print("  PASS: GPU dims exact")
    try:
        B._qc_branded_artifact(SRC, gpu_out); print("  PASS: GPU output passes branding QC gate")
    except Exception as e:
        print(f"  FAIL: GPU QC: {e}"); ok = False

    os.makedirs(BENCH, exist_ok=True)
    for tag, p in (("gpu", gpu_out), ("cpu", cpu_out)):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "5", "-i", p,
                        "-vframes", "1", os.path.join(BENCH, f"vrfy_{tag}.png")])
    print(f"  frames -> {BENCH}\\vrfy_gpu.png / vrfy_cpu.png (visual check)")
finally:
    shutil.rmtree(wd, ignore_errors=True)

print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
