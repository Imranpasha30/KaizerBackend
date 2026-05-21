"""Item 112 v2 -- corrected A/V drift measurement.

v1 used ffprobe's -read_intervals to find packets near a target
time. That's broken for VIDEO because video seeks land on the
previous keyframe (every ~8s in our mezzanine -> bulletin chain).
Audio seeks are sample-accurate so audio probes returned correct
PTS, but video probes returned PTS values rounded back to keyframes
-- making the script report 3-6s of phantom "drift" that was
actually seek-rounding error.

v2 strategy:
  1. Trust the stream-level durations reported by ffprobe (these
     come from container metadata + decoded stream length, not
     from seeking).
  2. Probe each composed_story_NN.mp4's INDIVIDUAL audio/video
     durations. The intra-segment delta reveals whether the Stage 4
     cut step is producing aligned slices.
  3. Compute cumulative drift across segments analytically:
     - per-segment intra-drift (audio_dur - video_dur)
     - acrossfade savings ((N-1) * audio_overlap)
     - expected bulletin audio = sum(audio) - savings
     - expected bulletin video = sum(video)
     - measured bulletin audio/video from final file
     - any unaccounted-for drift = the bug we're chasing

USAGE:
  python pipeline_v2/scripts/drift_measure_v2.py <bulletin.mp4>
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass


def _ffprobe_json(args):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-of", "json"] + args,
        capture_output=True, text=True, timeout=120,
    )
    return json.loads(r.stdout) if r.stdout.strip() else {}


def _stream_durs(path):
    """Return (video_dur, audio_dur, format_dur, video_frames)."""
    d = _ffprobe_json([
        "-show_entries", "stream=index,codec_type,duration,nb_frames",
        "-show_entries", "format=duration",
        path,
    ])
    streams = d.get("streams", [])
    fmt = d.get("format", {})
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    a = next((s for s in streams if s.get("codec_type") == "audio"), {})
    return (
        float(v.get("duration", 0) or 0),
        float(a.get("duration", 0) or 0),
        float(fmt.get("duration", 0) or 0),
        int(v.get("nb_frames", 0) or 0),
    )


def main(argv):
    if len(argv) < 2:
        print("usage: drift_measure_v2.py <bulletin.mp4>", file=sys.stderr)
        return 2
    bul = Path(argv[1]).resolve()
    if not bul.exists():
        print(f"ERROR: not found: {bul}", file=sys.stderr)
        return 2

    print("=" * 72)
    print(f"Drift measurement (v2): {bul.name}")
    print("=" * 72)

    # Bulletin-level streams.
    bv, ba, bf, bvframes = _stream_durs(str(bul))
    print(f"Bulletin file streams:")
    print(f"  video stream:  {bv:.4f}s  ({bvframes} frames @ "
          f"30fps = {bvframes/30:.4f}s expected)")
    print(f"  audio stream:  {ba:.4f}s")
    print(f"  format:        {bf:.4f}s")
    print(f"  V - A global:  {(bv - ba) * 1000:+.1f} ms")
    print(f"  frame-count vs duration mismatch (video): "
          f"{(bv - bvframes/30.0) * 1000:+.1f} ms")
    print()

    # Per-composed_story probes.
    bdir = bul.parent
    segs = sorted(bdir.glob("composed_story_*.mp4"))
    if not segs:
        print("(no composed_story segments alongside bulletin)")
        return 0

    print(f"Composed segments (n={len(segs)}):")
    print(f"  {'idx':>3}  {'v_dur':>9}  {'a_dur':>9}  {'a-v':>8}  {'cumul_a':>10}  {'cumul_v':>10}")
    print("  " + "-" * 66)
    cum_v = 0.0
    cum_a = 0.0
    intra_drift_total = 0.0
    seg_data = []
    for i, p in enumerate(segs):
        v, a, _, _ = _stream_durs(str(p))
        delta = a - v
        intra_drift_total += delta
        cum_a += a
        cum_v += v
        seg_data.append({"v": v, "a": a, "delta": delta,
                         "cum_a": cum_a, "cum_v": cum_v})
        if i < 5 or i >= len(segs) - 5:
            print(f"  {i:>3}  {v:>8.3f}s  {a:>8.3f}s  "
                  f"{delta * 1000:>+6.1f}ms  {cum_a:>9.3f}s  {cum_v:>9.3f}s")
        elif i == 5:
            print(f"  ...   (segments 5..{len(segs)-6} omitted)")

    print()
    print(f"Aggregates:")
    print(f"  sum(audio segments)  = {cum_a:.4f}s")
    print(f"  sum(video segments)  = {cum_v:.4f}s")
    print(f"  intra-segment audio-video drift (sum): "
          f"{intra_drift_total * 1000:+.1f} ms")
    print()

    # Item 111 model: bulletin audio = sum(a) - (N-1) * 0.08;
    # bulletin video = sum(v) (concat-demuxer, no overlap).
    # Then -shortest mux trims to min(audio_dur, video_dur).
    n = len(segs)
    audio_overlap = 0.08
    expected_a = cum_a - (n - 1) * audio_overlap
    expected_v = cum_v
    expected_short = min(expected_a, expected_v)
    print(f"Item-111 stitcher model:")
    print(f"  expected audio (sum_a - (N-1)*0.08): {expected_a:.4f}s")
    print(f"  expected video (sum_v, hard concat): {expected_v:.4f}s")
    print(f"  expected after -shortest mux:        {expected_short:.4f}s")
    print()

    # Discrepancies.
    a_diff = (ba - expected_a) * 1000
    v_diff = (bv - expected_v) * 1000
    short_diff = (bf - expected_short) * 1000
    print(f"Stitcher delivered:")
    print(f"  audio actual:  {ba:.4f}s  (model expected {expected_a:.4f}s; "
          f"delta {a_diff:+.1f} ms)")
    print(f"  video actual:  {bv:.4f}s  (model expected {expected_v:.4f}s; "
          f"delta {v_diff:+.1f} ms)")
    print(f"  format actual: {bf:.4f}s  (model expected {expected_short:.4f}s; "
          f"delta {short_diff:+.1f} ms)")
    print()

    # Verdicts.
    print(f"Lip-sync diagnosis:")
    if abs(bv - ba) <= 0.033:
        print(f"  Global V/A delta {(bv - ba) * 1000:+.1f} ms is WITHIN ONE FRAME (33ms).")
        print(f"  The file itself is acceptably synced.")
        if abs(intra_drift_total) > 0.2:
            print(f"  BUT intra-segment cumulative drift = "
                  f"{intra_drift_total * 1000:+.1f} ms.")
            print(f"  This means each composed_story has audio LONGER than video by")
            print(f"  ~30 ms each. Stage 4's cut step is producing slices where")
            print(f"  audio + video are NOT the same source-time range.")
            print(f"  -> mouth movements (video) lag the spoken words (audio)")
            print(f"     by ~{intra_drift_total / n * 1000:.1f} ms per segment.")
            print(f"  -> this would be PERCEIVED as lip-sync drift even though")
            print(f"     the stream durations match.")
        else:
            print(f"  Intra-segment drift {intra_drift_total * 1000:+.1f} ms is also small.")
            print(f"  If lip-sync is still bad, the issue is elsewhere "
                  f"(re-encoder PTS handling, mux behavior).")
    else:
        print(f"  Global V/A delta {(bv - ba) * 1000:+.1f} ms exceeds one-frame "
              f"tolerance.")
        print(f"  Something in the stitch chain is misaligning streams.")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
