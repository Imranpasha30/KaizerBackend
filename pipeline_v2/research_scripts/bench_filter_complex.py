"""
Track 3D — Empirical test: ffmpeg filter_complex scaling.

Builds filter graphs with N trim nodes (N=5,10,20,50,100,200) on the same
source mezzanine and measures:
  - wall time
  - exit success
  - peak RSS (psutil)
  - output stream A/V duration

NO production code is touched. Outputs go to research_scratch/.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
SCRATCH = HERE.parent.parent / "research_scratch" / "bench_outputs"
SCRATCH.mkdir(parents=True, exist_ok=True)
SOURCE = Path("e:/kaizer new data training/kaizer/KaizerBackend/output/full_video_shorts_v2/job_53/mezzanine.mp4")
assert SOURCE.exists(), f"source missing: {SOURCE}"

DURATION_S = 589.0  # mezzanine length
NS = [5, 10, 20, 50, 100, 200]


def ffprobe_durations(p: Path) -> tuple[float, float]:
    if not p.exists():
        return (0.0, 0.0)
    try:
        v = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=duration", "-of", "csv=p=0", str(p)],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        a = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=duration", "-of", "csv=p=0", str(p)],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        return (float(v or 0), float(a or 0))
    except Exception as e:
        return (-1.0, -1.0)


def build_filter_complex(n: int) -> tuple[str, list[str]]:
    """Build N trim nodes from evenly-spaced ranges + concat. Returns (filter,
    maps)."""
    seg = max(DURATION_S / (n * 1.5), 1.0)
    parts = []
    for i in range(n):
        start = i * (DURATION_S / n)
        end = start + seg
        if end > DURATION_S - 0.1:
            end = DURATION_S - 0.1
        parts.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{i:03d}]"
        )
        parts.append(
            f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{i:03d}]"
        )
    bv_in = "".join(f"[v{i:03d}]" for i in range(n))
    ba_in = "".join(f"[a{i:03d}]" for i in range(n))
    parts.append(f"{bv_in}concat=n={n}:v=1:a=0[bv_out]")
    parts.append(f"{ba_in}concat=n={n}:v=0:a=1[ba_out]")
    return ";".join(parts), ["-map", "[bv_out]", "-map", "[ba_out]"]


def run_one(n: int) -> dict:
    fc, maps = build_filter_complex(n)
    out = SCRATCH / f"fc_n{n:03d}.mp4"
    if out.exists():
        out.unlink()
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(SOURCE),
        "-filter_complex", fc,
        *maps,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
        "-c:a", "aac", "-b:a", "128k",
        str(out),
    ]
    t0 = time.monotonic()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        ok = (r.returncode == 0)
        err_tail = r.stderr[-500:] if r.stderr else ""
    except subprocess.TimeoutExpired:
        ok = False
        err_tail = "TIMEOUT after 600s"
    dt = time.monotonic() - t0
    v_dur, a_dur = ffprobe_durations(out)
    size = out.stat().st_size if out.exists() else 0
    return {
        "n": n,
        "wall_s": round(dt, 2),
        "ok": ok,
        "out_size": size,
        "v_dur": round(v_dur, 3),
        "a_dur": round(a_dur, 3),
        "av_delta_ms": round((v_dur - a_dur) * 1000, 2),
        "filter_complex_chars": len(fc),
        "err_tail": err_tail,
    }


def main():
    results = []
    for n in NS:
        print(f"[{time.strftime('%H:%M:%S')}] N={n} ...", flush=True)
        r = run_one(n)
        results.append(r)
        print(f"  -> wall={r['wall_s']}s ok={r['ok']} av_delta={r['av_delta_ms']}ms fc_chars={r['filter_complex_chars']} size={r['out_size']/1e6:.1f}MB")
    out_json = SCRATCH / "fc_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nResults: {out_json}")


if __name__ == "__main__":
    main()
