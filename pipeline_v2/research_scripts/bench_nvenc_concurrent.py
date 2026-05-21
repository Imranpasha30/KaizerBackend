"""Track 3D-2 — NVENC concurrent session test on RTX 5060.

Spawns K concurrent ffmpeg processes each encoding a 30s mezzanine slice with
h264_nvenc. Records which succeed, which fail with capacity errors.

NO production code modified.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = Path(__file__).resolve()
SCRATCH = HERE.parent.parent / "research_scratch" / "nvenc_outputs"
SCRATCH.mkdir(parents=True, exist_ok=True)
SOURCE = Path("e:/kaizer new data training/kaizer/KaizerBackend/output/full_video_shorts_v2/job_53/mezzanine.mp4")
assert SOURCE.exists()


def encode_one(idx: int) -> dict:
    out = SCRATCH / f"nvenc_{idx:02d}.mp4"
    if out.exists():
        out.unlink()
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        "-ss", f"{(idx * 5) % 500}", "-i", str(SOURCE),
        "-t", "30",
        "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "26",
        "-c:a", "aac", "-b:a", "128k",
        str(out),
    ]
    t0 = time.monotonic()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        ok = (r.returncode == 0)
        err = r.stderr[-300:] if r.stderr else ""
    except subprocess.TimeoutExpired:
        ok = False
        err = "TIMEOUT"
    dt = time.monotonic() - t0
    return {"idx": idx, "ok": ok, "wall_s": round(dt, 2), "size": out.stat().st_size if out.exists() else 0, "err": err}


def run_k(k: int) -> list[dict]:
    print(f"[{time.strftime('%H:%M:%S')}] Spawning K={k} concurrent NVENC ...", flush=True)
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=k) as ex:
        futs = [ex.submit(encode_one, i) for i in range(k)]
        results = [f.result() for f in as_completed(futs)]
    dt = time.monotonic() - t0
    n_ok = sum(1 for r in results if r["ok"])
    print(f"  -> K={k}: {n_ok}/{k} succeeded; total wall={dt:.1f}s")
    for r in results:
        if not r["ok"]:
            print(f"     fail idx={r['idx']}: {r['err'][:200]}")
    return results


def main():
    summary = {}
    for k in [1, 2, 3, 4, 6, 8]:
        results = run_k(k)
        n_ok = sum(1 for r in results if r["ok"])
        summary[k] = {"k": k, "n_ok": n_ok, "n_total": k, "per_run": results}
    out = SCRATCH / "nvenc_results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nResults: {out}")


if __name__ == "__main__":
    main()
