"""Item 111 live verification: run the new 3-pass stitcher on Job 46's
25 composed_story files and probe the output.

Acceptance:
  - Video duration ≈ sum of input video durations (≈ 475.67s)
  - Audio duration ≈ sum(input audio) - 24*0.08s ≈ 474.08s
  - Format duration: matches min(video, audio) since we use -shortest
  - File plays cleanly (no truncation; this was the bug at 104.5s)
"""
import io
import subprocess
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass

HERE = Path(__file__).resolve().parent
BACKEND_ROOT = HERE.parent.parent
PIPELINE_V2_ROOT = BACKEND_ROOT / "pipeline_v2"
sys.path.insert(0, str(PIPELINE_V2_ROOT))

from pipeline_v2.bulletin_crossfade_stitcher import (
    stitch_bulletin_with_crossfade,
)

JOB46_DIR = BACKEND_ROOT / "output" / "full_video_shorts_v2" / "job_46" / "bulletin"
segs = sorted(str(p) for p in JOB46_DIR.glob("composed_story_*.mp4"))
print(f"Found {len(segs)} composed_story_NN.mp4 files.")

# Write the new bulletin to a fresh path so we don't clobber the old one.
out_path = JOB46_DIR / "bulletin_item111_test.mp4"
work_dir = JOB46_DIR / "_item111_workdir"
work_dir.mkdir(exist_ok=True)

print(f"Running 3-pass crossfade stitcher -> {out_path} ...")
t0 = time.time()
result = stitch_bulletin_with_crossfade(
    segs,
    str(out_path),
    audio_overlap_s=0.08,
    work_dir=str(work_dir),
)
wall = time.time() - t0
print(f"Stitcher returned: total_duration_s = {result.total_duration_s:.3f}s")
print(f"  stories_rendered  = {result.stories_rendered}")
print(f"  stories_skipped   = {result.stories_skipped}")
print(f"  audio_overlap_s   = {result.audio_overlap_s}")
print(f"  video_overlap_s   = {result.video_overlap_s}")
print(f"  wall time         = {wall:.1f}s")
if result.warnings:
    print("  warnings:")
    for w in result.warnings:
        print(f"    {w}")

# Now ffprobe the output for verification.
print()
print("=== ffprobe output ===")
r = subprocess.run(
    ["ffprobe", "-v", "error",
     "-show_entries", "stream=index,codec_type,codec_name,duration,r_frame_rate,nb_frames",
     "-show_entries", "format=duration",
     "-of", "compact",
     str(out_path)],
    capture_output=True, text=True,
)
print(r.stdout)
print()

# Compute expected values from inputs.
def probe_audio(p):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", p],
        capture_output=True, text=True,
    )
    return float((r.stdout or "0").strip() or 0)

def probe_video(p):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", p],
        capture_output=True, text=True,
    )
    return float((r.stdout or "0").strip() or 0)

audio_dur = [probe_audio(s) for s in segs]
video_dur = [probe_video(s) for s in segs]
sum_audio = sum(audio_dur)
sum_video = sum(video_dur)
expected_audio = sum_audio - (len(segs) - 1) * 0.08
print(f"sum(input audio)  = {sum_audio:.3f}s")
print(f"sum(input video)  = {sum_video:.3f}s")
print(f"expected audio    = sum - (N-1)*0.08 = {expected_audio:.3f}s")
print(f"expected video    = sum (no overlap) = {sum_video:.3f}s")
print()

# Re-probe output streams.
def probe_stream(p, sel):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", sel,
         "-show_entries", "stream=duration,nb_frames",
         "-of", "default=noprint_wrappers=1:nokey=1", p],
        capture_output=True, text=True,
    )
    lines = r.stdout.strip().split("\n")
    return [l for l in lines if l]

out_v = probe_stream(str(out_path), "v:0")
out_a = probe_stream(str(out_path), "a:0")
print(f"output video stream: {out_v}")
print(f"output audio stream: {out_a}")

# Pass/fail.
out_v_dur = float(out_v[0]) if out_v else 0
out_a_dur = float(out_a[0]) if out_a else 0
print()
print("=== Verdict ===")
verdict_video = "PASS" if abs(out_v_dur - sum_video) < 2.0 or abs(out_v_dur - expected_audio) < 2.0 else "FAIL"
verdict_audio = "PASS" if abs(out_a_dur - expected_audio) < 0.2 else "FAIL"
print(f"Video duration: {out_v_dur:.3f}s (expect ~{expected_audio:.3f}s after -shortest, or ~{sum_video:.3f}s without): {verdict_video}")
print(f"Audio duration: {out_a_dur:.3f}s (expect {expected_audio:.3f}s ±0.2): {verdict_audio}")
print(f"Old bulletin had video=104.5s (broken). New has video={out_v_dur:.1f}s.")
