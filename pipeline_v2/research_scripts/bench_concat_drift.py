"""Track 3D-3 — AAC concat drift demonstration.

Reproduces the production drift pattern empirically:
1. Cut N=22 raw clips of 20s each from mezzanine
2. concat via demuxer (-c copy)
3. concat via filter_complex (re-encode)
4. crossfade via acrossfade (3-pass mux mirroring bulletin_crossfade_stitcher.py)

Measures cumulative A/V drift at each stage.

NO production code modified.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve()
SCRATCH = HERE.parent.parent / "research_scratch" / "concat_drift"
SCRATCH.mkdir(parents=True, exist_ok=True)
SOURCE = Path("e:/kaizer new data training/kaizer/KaizerBackend/output/full_video_shorts_v2/job_53/mezzanine.mp4")
assert SOURCE.exists()


def probe(p: Path) -> tuple[float, float]:
    try:
        v = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration", "-of", "csv=p=0", str(p)], capture_output=True, text=True, timeout=30).stdout.strip()
        a = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration", "-of", "csv=p=0", str(p)], capture_output=True, text=True, timeout=30).stdout.strip()
        return float(v or 0), float(a or 0)
    except Exception:
        return -1, -1


def cut_clips(n: int) -> list[Path]:
    """Cut N 20s clips at evenly-spaced positions. Re-encode both A+V to mimic
    pipeline_v2 cut step (NVENC + aac)."""
    clips = []
    duration_each = 20.0
    available = 580.0
    step = available / n
    for i in range(n):
        start = i * step
        out = SCRATCH / f"clip_{i:02d}.mp4"
        if out.exists():
            out.unlink()
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-i", str(SOURCE),
            "-t", f"{duration_each:.3f}",
            "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "26",
            "-c:a", "aac", "-b:a", "128k",
            str(out),
        ]
        subprocess.run(cmd, capture_output=True, timeout=60)
        clips.append(out)
    return clips


def concat_demux(clips: list[Path], out_name: str) -> Path:
    list_file = SCRATCH / f"_list_{out_name}.txt"
    list_file.write_text("\n".join(f"file '{c}'" for c in clips))
    out = SCRATCH / out_name
    if out.exists():
        out.unlink()
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c", "copy",
        str(out),
    ]
    subprocess.run(cmd, capture_output=True, timeout=120)
    return out


def concat_filter(clips: list[Path], out_name: str) -> Path:
    out = SCRATCH / out_name
    if out.exists():
        out.unlink()
    n = len(clips)
    inputs = []
    for c in clips:
        inputs += ["-i", str(c)]
    fc = ""
    for i in range(n):
        fc += f"[{i}:v]setpts=PTS-STARTPTS[v{i}];[{i}:a]asetpts=PTS-STARTPTS[a{i}];"
    fc += "".join(f"[v{i}][a{i}]" for i in range(n))
    fc += f"concat=n={n}:v=1:a=1[vo][ao]"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        *inputs,
        "-filter_complex", fc,
        "-map", "[vo]", "-map", "[ao]",
        "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "26",
        "-c:a", "aac", "-b:a", "128k",
        str(out),
    ]
    subprocess.run(cmd, capture_output=True, timeout=300)
    return out


def single_pass_extract_concat(n: int, out_name: str) -> Path:
    """Item 117's filter graph applied to N evenly spaced cuts as a single
    extract+concat in one ffmpeg invocation."""
    out = SCRATCH / out_name
    if out.exists():
        out.unlink()
    available = 580.0
    step = available / n
    fc_parts = []
    for i in range(n):
        s = i * step
        e = s + 20.0
        fc_parts.append(f"[0:v]trim=start={s:.6f}:end={e:.6f},setpts=PTS-STARTPTS[v{i:02d}]")
        fc_parts.append(f"[0:a]atrim=start={s:.6f}:end={e:.6f},asetpts=PTS-STARTPTS[a{i:02d}]")
    bv_in = "".join(f"[v{i:02d}]" for i in range(n))
    ba_in = "".join(f"[a{i:02d}]" for i in range(n))
    fc_parts.append(f"{bv_in}concat=n={n}:v=1:a=0[vo]")
    fc_parts.append(f"{ba_in}concat=n={n}:v=0:a=1[ao]")
    fc = ";".join(fc_parts)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(SOURCE),
        "-filter_complex", fc,
        "-map", "[vo]", "-map", "[ao]",
        "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "26",
        "-c:a", "aac", "-b:a", "128k",
        str(out),
    ]
    subprocess.run(cmd, capture_output=True, timeout=300)
    return out


def main():
    n = 22
    print(f"=== Cut {n} clips ===")
    t0 = time.monotonic()
    clips = cut_clips(n)
    cut_t = time.monotonic() - t0
    print(f"  cut {n} clips in {cut_t:.1f}s")

    # Per-clip drift
    per_clip = []
    for i, c in enumerate(clips):
        v, a = probe(c)
        per_clip.append({"i": i, "v": v, "a": a, "delta_ms": round((v - a) * 1000, 3)})
    sum_v = sum(r["v"] for r in per_clip)
    sum_a = sum(r["a"] for r in per_clip)
    print(f"  per-clip sum: v={sum_v:.3f}s a={sum_a:.3f}s delta={(sum_v-sum_a)*1000:+.2f}ms")

    print(f"\n=== Method 1: concat demux (-c copy) ===")
    t0 = time.monotonic()
    out1 = concat_demux(clips, "method1_demux.mp4")
    print(f"  {time.monotonic()-t0:.1f}s wall")
    v, a = probe(out1)
    print(f"  v={v:.3f}s a={a:.3f}s delta={(v-a)*1000:+.2f}ms (expected vs sum: v_lost={sum_v-v:+.3f}s a_lost={sum_a-a:+.3f}s)")

    print(f"\n=== Method 2: concat filter (re-encode) ===")
    t0 = time.monotonic()
    out2 = concat_filter(clips, "method2_filter.mp4")
    print(f"  {time.monotonic()-t0:.1f}s wall")
    v, a = probe(out2)
    print(f"  v={v:.3f}s a={a:.3f}s delta={(v-a)*1000:+.2f}ms (vs sum: v_lost={sum_v-v:+.3f}s a_lost={sum_a-a:+.3f}s)")

    print(f"\n=== Method 3: single-pass extract+concat (item 117 style) ===")
    t0 = time.monotonic()
    out3 = single_pass_extract_concat(n, "method3_single_pass.mp4")
    print(f"  {time.monotonic()-t0:.1f}s wall")
    v, a = probe(out3)
    print(f"  v={v:.3f}s a={a:.3f}s delta={(v-a)*1000:+.2f}ms")

    out_json = SCRATCH / "concat_drift_results.json"
    out_json.write_text(json.dumps({"per_clip": per_clip, "sum_v": sum_v, "sum_a": sum_a}, indent=2))
    print(f"\nResults: {out_json}")


if __name__ == "__main__":
    main()
