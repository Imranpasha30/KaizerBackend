"""Assert the QC duration fix: the reported false-positives now PASS,
genuine truncation still FAILS."""
import sys
sys.path.insert(0, ".")
import warnings
warnings.filterwarnings("ignore")
from pipeline_v4 import qc

import os
import shutil
import subprocess
import tempfile

ok = True


def expect(label, got, want):
    global ok
    p = (got == want)
    ok = ok and p
    print(("  PASS  " if p else "  FAIL  ") + label + ("" if p else f"  got={got} want={want}"))


if not shutil.which("ffmpeg"):
    print("ffmpeg not on PATH — skipping render-based checks")
    sys.exit(0)

tmp = tempfile.mkdtemp(prefix="kaizer_qcfix_")
try:
    # A real ~40s short rendered slightly short of its 41.8s plan estimate.
    short = os.path.join(tmp, "short.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", "testsrc=size=1080x1920:rate=30:duration=40.4",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=40.4",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
         "-shortest", short],
        capture_output=True, timeout=180, check=True)

    v = qc.verify_render(short, expected_duration=41.80,
                         expected_w=1080, expected_h=1920)
    expect("reported case (40.4s vs 41.8s plan) now PASSES", v, [])

    # Genuine truncation: a 5s file where ~41.8s was planned.
    trunc = os.path.join(tmp, "trunc.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", "testsrc=size=1080x1920:rate=30:duration=5",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
         "-shortest", trunc],
        capture_output=True, timeout=120, check=True)
    v2 = qc.verify_render(trunc, expected_duration=41.80,
                          expected_w=1080, expected_h=1920)
    expect("genuine truncation (5s vs 41.8s) still FAILS", bool(v2), True)
    print("    truncation violation:", v2[0] if v2 else "(none)")

    # Longer than plan (overlays/intro) passes.
    v3 = qc.verify_render(short, expected_duration=35.0,
                          expected_w=1080, expected_h=1920)
    expect("longer-than-plan (40.4s vs 35s) PASSES", v3, [])

    # Wrong resolution still caught.
    v4 = qc.verify_render(short, expected_duration=41.80,
                          expected_w=1920, expected_h=1080)
    expect("wrong resolution still caught", bool(v4), True)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\nQC FIX OK" if ok else "\nQC FIX FAILED")
sys.exit(0 if ok else 1)
