"""Item 112 -- A/V drift measurement at user-specified bulletin timestamps.

Used to verify whether user-perceived lip-sync drift matches what is
actually in the bulletin file. Inputs:
  - path to a bulletin_with_overlays.mp4 (or bulletin.mp4)
  - one or more bulletin-playback-time timestamps (in seconds OR
    "MM:SS" / "HH:MM:SS")

For each timestamp T the script reports:
  1. Bulletin-time T (sec)
  2. Composed_story_NN segment that T falls into (and how far in)
  3. Bulletin-stream audio PTS observed at the nearest audio packet
  4. Bulletin-stream video PTS observed at the nearest video frame
  5. Local A/V drift = video_PTS - audio_PTS (milliseconds)
  6. Source mezzanine time T would represent at this composed-story
     position (if Stage 2 sub-cut data is available in the job log)
  7. Verdict bucket:
       <=  33 ms  ::  acceptable (within one 30fps frame)
        <= 100 ms  ::  marginal (perceptible to a sensitive viewer)
        >  100 ms  ::  severe (clearly out of sync)

Aggregate verdict across all sampled timestamps:
  - average drift
  - drift trend (constant / accelerating / decelerating) -- a constant
    offset usually means a baked-in V/A start mismatch; an accelerating
    drift means timebase divergence between streams

USAGE:
  python pipeline_v2/scripts/drift_measure.py <bulletin.mp4> T1 [T2 T3 ...]

Examples:
  python pipeline_v2/scripts/drift_measure.py \
      output/full_video_shorts_v2/job_47/bulletin/bulletin_with_overlays.mp4 \
      02:14 04:32 06:01
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass

FRAME_30FPS_MS = 1000.0 / 30.0   # 33.333 ms
ACCEPTABLE_MS = 33.0
MARGINAL_MS = 100.0


def _parse_timestamp(s: str) -> float:
    """Accept '90', '01:30', '01:30.500', '0:01:30' -> 90.0s."""
    s = s.strip()
    if ":" not in s:
        return float(s)
    parts = s.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Bad timestamp: {s!r}")


def _ffprobe_json(args: list[str]) -> dict:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-of", "json"] + args,
        capture_output=True, text=True, timeout=120,
    )
    return json.loads(r.stdout) if r.stdout.strip() else {}


def _probe_streams(path: str) -> dict:
    """Top-level audio + video stream details."""
    d = _ffprobe_json([
        "-show_streams",
        "-show_entries",
        "stream=index,codec_type,codec_name,duration,nb_frames,"
        "r_frame_rate,avg_frame_rate,sample_rate,start_time,time_base",
        path,
    ])
    return d


def _ffprobe_packets(path: str, around_s: float, window_s: float, stream_sel: str) -> list[dict]:
    """Get packet list (pts, dts, duration) in a window around the given time."""
    lo = max(0.0, around_s - window_s)
    interval = f"{lo}%+{window_s * 2:.3f}"
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-of", "json",
         "-read_intervals", interval,
         "-select_streams", stream_sel,
         "-show_packets",
         "-show_entries", "packet=pts_time,dts_time,duration_time",
         path],
        capture_output=True, text=True, timeout=120,
    )
    if not r.stdout.strip():
        return []
    return (json.loads(r.stdout) or {}).get("packets", [])


def _nearest_packet_pts(packets: list[dict], t: float) -> Optional[float]:
    """Find the packet whose pts_time is closest to t. Returns the
    pts_time (float) or None if no packets available."""
    candidates = []
    for p in packets:
        pt = p.get("pts_time")
        if pt is None or pt == "":
            continue
        try:
            candidates.append(float(pt))
        except (TypeError, ValueError):
            continue
    if not candidates:
        return None
    return min(candidates, key=lambda pt: abs(pt - t))


def _measure_drift_at(path: str, t: float) -> dict:
    """Probe the audio and video packets nearest bulletin-time t,
    compute A/V drift (video_pts - audio_pts) in milliseconds."""
    a_pkts = _ffprobe_packets(path, t, window_s=1.0, stream_sel="a:0")
    v_pkts = _ffprobe_packets(path, t, window_s=1.0, stream_sel="v:0")
    a_pts = _nearest_packet_pts(a_pkts, t)
    v_pts = _nearest_packet_pts(v_pkts, t)
    if a_pts is None or v_pts is None:
        return {
            "t": t,
            "audio_pts": a_pts,
            "video_pts": v_pts,
            "drift_ms": None,
            "drift_verdict": "no-packets",
            "n_audio_packets": len(a_pkts),
            "n_video_packets": len(v_pkts),
        }
    drift_s = v_pts - a_pts
    drift_ms = drift_s * 1000.0
    abs_ms = abs(drift_ms)
    if abs_ms <= ACCEPTABLE_MS:
        verdict = "OK"
    elif abs_ms <= MARGINAL_MS:
        verdict = "MARGINAL"
    else:
        verdict = "SEVERE"
    return {
        "t": t,
        "audio_pts": a_pts,
        "video_pts": v_pts,
        "drift_ms": drift_ms,
        "drift_verdict": verdict,
        "n_audio_packets": len(a_pkts),
        "n_video_packets": len(v_pkts),
    }


# ---- Composed-story segment mapping ---------------------------------


def _probe_segment_durations(job_bulletin_dir: Path) -> list[dict]:
    """Probe all composed_story_NN.mp4 files in order; return per-segment
    video_dur, audio_dur, and (computed) running bulletin-time offsets."""
    files = sorted(job_bulletin_dir.glob("composed_story_*.mp4"))
    out: list[dict] = []
    for p in files:
        a = _ffprobe_json(["-select_streams", "a:0",
                           "-show_entries", "stream=duration", str(p)])
        v = _ffprobe_json(["-select_streams", "v:0",
                           "-show_entries", "stream=duration", str(p)])
        a_dur = float(a.get("streams", [{}])[0].get("duration", 0) or 0)
        v_dur = float(v.get("streams", [{}])[0].get("duration", 0) or 0)
        out.append({
            "name": p.name,
            "audio_dur": a_dur,
            "video_dur": v_dur,
        })
    return out


def _build_segment_offsets(segments: list[dict], audio_overlap_s: float = 0.08) -> list[dict]:
    """Compute the running bulletin-time start offset of each segment.

    Item 111 model: video is concat-only (no overlap), audio uses
    acrossfade (subtracts one overlap per chained pair). At the
    bulletin level the file has video_dur and audio_dur slightly
    different; -shortest aligns them on the shorter. For drift
    measurement at bulletin-time T, we use the CUMULATIVE audio
    timeline since audio playback is the perceptual reference.
    """
    running_audio = 0.0
    running_video = 0.0
    for i, seg in enumerate(segments):
        seg["audio_start"] = running_audio
        seg["video_start"] = running_video
        running_audio += seg["audio_dur"]
        if i < len(segments) - 1:
            running_audio -= audio_overlap_s
        running_video += seg["video_dur"]
        seg["audio_end"] = running_audio
        seg["video_end"] = running_video
    return segments


def _find_segment(segments: list[dict], t: float) -> Optional[dict]:
    for seg in segments:
        if seg["audio_start"] <= t < seg["audio_end"]:
            return seg
    return None


# ---- Optional: source-mezzanine cross-reference ---------------------


_SUBCUT_RE = re.compile(
    r"\[bulletin sub (\d+)\]\s+"
    r"(\d{2}:\d{2}\.\d{3})\s+->\s+(\d{2}:\d{2}\.\d{3})\s+\(([\d.]+)s\)"
)


def _parse_subcuts_from_job_log(job_log_text: str) -> list[dict]:
    """Find ``[bulletin sub N] HH:MM.mmm -> HH:MM.mmm (Xs)`` lines in
    the job log -- they map composed_story_NN to a SOURCE mezzanine
    time range. Returns list of {idx, src_start_sec, src_end_sec, dur_s}."""
    out: list[dict] = []
    for m in _SUBCUT_RE.finditer(job_log_text or ""):
        idx = int(m.group(1))
        src_start = _parse_timestamp(m.group(2))
        src_end = _parse_timestamp(m.group(3))
        out.append({
            "idx": idx,
            "src_start_sec": src_start,
            "src_end_sec": src_end,
            "dur_s": float(m.group(4)),
        })
    return out


def _load_job_log_and_transcript(bulletin_path: Path) -> dict:
    """Attempt to locate the parent job directory, then load
    transcript.json + the per-job log. Returns {transcript_words,
    sub_cuts, found}."""
    out = {"transcript_words": [], "sub_cuts": [], "found": False}
    # bulletin_with_overlays.mp4 lives in <job>/bulletin/. Walk up.
    job_dir = bulletin_path.parent.parent
    if not job_dir.is_dir():
        return out
    tr_path = job_dir / "transcript.json"
    if tr_path.exists():
        try:
            d = json.loads(tr_path.read_text(encoding="utf-8"))
            out["transcript_words"] = d.get("words", [])
        except Exception:
            pass
    # Try DB log via SessionLocal -- skip if unavailable.
    try:
        backend_root = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(backend_root))
        from dotenv import load_dotenv
        load_dotenv(backend_root / ".env", override=True)
        from database import SessionLocal
        from sqlalchemy import text as sqltext
        # Job id from the dir name "job_NN".
        m = re.search(r"job_(\d+)$", str(job_dir.name))
        if m:
            jid = int(m.group(1))
            s = SessionLocal()
            r = s.execute(
                sqltext("SELECT log FROM jobs WHERE id = :id"),
                {"id": jid},
            ).fetchone()
            s.close()
            if r and r[0]:
                out["sub_cuts"] = _parse_subcuts_from_job_log(r[0])
    except Exception as exc:
        print(f"  [note] sub-cut DB lookup failed (non-fatal): {exc}",
              file=sys.stderr)
    out["found"] = bool(out["transcript_words"] or out["sub_cuts"])
    return out


def _word_at_source_time(words: list[dict], src_t: float) -> Optional[dict]:
    for w in words:
        s = float(w.get("s") or 0)
        e = float(w.get("e") or 0)
        if s <= src_t <= e:
            return w
    return None


# ---- Main report ----------------------------------------------------


def _format_mmss(t: float) -> str:
    m = int(t // 60)
    s = t % 60
    return f"{m:02d}:{s:06.3f}"


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2

    bulletin = Path(argv[1]).resolve()
    if not bulletin.exists():
        print(f"ERROR: bulletin not found: {bulletin}", file=sys.stderr)
        return 2

    raw_ts = argv[2:]
    try:
        timestamps = [_parse_timestamp(s) for s in raw_ts]
    except ValueError as e:
        print(f"ERROR parsing timestamps: {e}", file=sys.stderr)
        return 2

    print("=" * 72)
    print(f"A/V drift measurement: {bulletin.name}")
    print("=" * 72)

    # Top-level stream info.
    streams_info = _probe_streams(str(bulletin))
    streams = streams_info.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not v or not a:
        print("ERROR: bulletin missing video or audio stream", file=sys.stderr)
        return 2

    v_dur = float(v.get("duration", 0) or 0)
    a_dur = float(a.get("duration", 0) or 0)
    file_drift_ms = (v_dur - a_dur) * 1000.0
    print(f"Streams:")
    print(f"  video: {v_dur:.3f}s  ({v.get('nb_frames')} frames, "
          f"r_frame_rate={v.get('r_frame_rate')})")
    print(f"  audio: {a_dur:.3f}s  (sample_rate={a.get('sample_rate')})")
    print(f"  global delta (video - audio): {file_drift_ms:+.1f} ms")
    print()

    # Segment offsets.
    bulletin_dir = bulletin.parent
    segments = _probe_segment_durations(bulletin_dir)
    if segments:
        segments = _build_segment_offsets(segments)
        print(f"Composed segments found: {len(segments)}")
        # Print first 3 + last 3 for context.
        previews = segments if len(segments) <= 6 else (segments[:3]
                    + [{"name": "...", "audio_start": 0, "audio_end": 0,
                        "video_start": 0, "video_end": 0,
                        "audio_dur": 0, "video_dur": 0}]
                    + segments[-3:])
        for s in previews:
            if s["name"] == "...":
                print(f"  ...")
                continue
            print(f"  {s['name']:<28}  v={s['video_dur']:6.2f}s  "
                  f"a={s['audio_dur']:6.2f}s  bulletin_time={_format_mmss(s['audio_start'])} -> {_format_mmss(s['audio_end'])}")
    else:
        print("(no composed_story_*.mp4 found in bulletin dir)")
    print()

    # Optional cross-reference data.
    aux = _load_job_log_and_transcript(bulletin)
    if aux["found"]:
        print(f"Cross-reference data:")
        if aux["transcript_words"]:
            print(f"  transcript_words: {len(aux['transcript_words'])} entries")
        if aux["sub_cuts"]:
            print(f"  sub_cuts (from job log): {len(aux['sub_cuts'])}")
    else:
        print("(no transcript or sub-cut data found -- source mapping disabled)")
    print()

    # Per-timestamp measurement.
    print(f"{'bulletin_t':>10}  {'seg':>3}  {'a_pts':>10}  {'v_pts':>10}  {'drift_ms':>9}  {'verdict':>8}  {'source word'}")
    print("-" * 100)
    drifts: list[float] = []
    for t in timestamps:
        m = _measure_drift_at(str(bulletin), t)
        seg = _find_segment(segments, t) if segments else None
        seg_label = ""
        if seg:
            seg_idx = segments.index(seg)
            seg_label = f"{seg_idx:>2}"
        # Source-word cross-reference: bulletin-time t -> segment ->
        # source mezzanine time via sub-cuts -> word lookup.
        word_str = ""
        if seg and aux["sub_cuts"] and aux["transcript_words"]:
            seg_idx = segments.index(seg)
            if seg_idx < len(aux["sub_cuts"]):
                sc = aux["sub_cuts"][seg_idx]
                # Position within the segment in bulletin time:
                offset_in_seg = t - seg["audio_start"]
                src_t = sc["src_start_sec"] + offset_in_seg
                w = _word_at_source_time(aux["transcript_words"], src_t)
                if w:
                    word_str = f"[src {_format_mmss(src_t)}] {w.get('w', '?')}"
                else:
                    word_str = f"[src {_format_mmss(src_t)}] (silence/gap)"
        a_pts_s = f"{m['audio_pts']:.3f}" if m['audio_pts'] is not None else "-"
        v_pts_s = f"{m['video_pts']:.3f}" if m['video_pts'] is not None else "-"
        drift_s = f"{m['drift_ms']:+8.1f}" if m['drift_ms'] is not None else "    -   "
        print(f"{_format_mmss(t):>10}  {seg_label:>3}  {a_pts_s:>10}  {v_pts_s:>10}  {drift_s:>9}  {m['drift_verdict']:>8}  {word_str}")
        if m['drift_ms'] is not None:
            drifts.append(m['drift_ms'])

    print()
    if drifts:
        avg = sum(drifts) / len(drifts)
        mn, mx = min(drifts), max(drifts)
        spread = mx - mn
        print(f"Summary:")
        print(f"  samples       : {len(drifts)}")
        print(f"  avg drift     : {avg:+.1f} ms")
        print(f"  min / max     : {mn:+.1f} / {mx:+.1f} ms (spread {spread:.1f} ms)")
        print(f"  file global Δ : {file_drift_ms:+.1f} ms")
        # Trend.
        if len(drifts) >= 3:
            first_half = sum(drifts[:len(drifts)//2]) / max(1, len(drifts)//2)
            second_half = sum(drifts[len(drifts)//2:]) / max(1, len(drifts) - len(drifts)//2)
            trend_delta = second_half - first_half
            if abs(trend_delta) < 10:
                trend = "constant (baked-in offset)"
            elif trend_delta > 0:
                trend = f"accelerating drift (+{trend_delta:.0f} ms over the bulletin)"
            else:
                trend = f"decelerating drift ({trend_delta:.0f} ms over the bulletin)"
            print(f"  trend         : {trend}")
        print()
        # Bucket the verdict.
        max_abs = max(abs(d) for d in drifts)
        if max_abs <= ACCEPTABLE_MS:
            print(f"  VERDICT: ACCEPTABLE (max |drift| {max_abs:.1f} ms <= {ACCEPTABLE_MS:.0f} ms)")
        elif max_abs <= MARGINAL_MS:
            print(f"  VERDICT: MARGINAL (max |drift| {max_abs:.1f} ms)")
        else:
            print(f"  VERDICT: SEVERE (max |drift| {max_abs:.1f} ms)")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
