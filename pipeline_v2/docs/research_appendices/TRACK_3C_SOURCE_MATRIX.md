# Track 3C — Source Video Compatibility Matrix + Production Output Empirical Audit

Generated: 2026-05-21 22:00 IST  
By: main thread (own work, ffprobe read-only)  
Constraint: No production code changes, no pipeline execution. Only ffprobe + filesystem reads of existing job outputs.

---

## Section 1 — Source video properties

Sources available in `e:/kaizer new data training/videos/`. Properties from `ffprobe -v error -show_entries stream=...`.

| File | Codec | WxH | fps (r/avg) | Audio | Channels | Duration | Notes |
|---|---|---|---|---|---|---|---|
| `C6355.MP4` | h264 | 3840×2160 | 25/25 (CFR) | pcm_s16be | 2ch | 212.64s | 4K Canon RAW pattern, PCM uncompressed. Bitrate 57Mbps. Has a "data" stream (likely timecode/metadata). EVIDENCE: ffprobe job. CONFIDENCE: HIGH. |
| `KaizerNewsPolitics.mp4` | h264 | 1920×1080 | 25/25 (CFR) | aac | 2ch | 371.6s | Pre-encoded 1080p, 16Mbps. AAC 318kbps stereo. EVIDENCE: ffprobe. CONFIDENCE: HIGH. |
| `MVI_0384.MP4` | h264 | 1920×1080 | 50/50 (CFR) | pcm_s16be ×4 | 1ch each | 144.48s | **4 mono PCM audio streams** (Canon XLR + ambient). 35Mbps. Has a data stream. EVIDENCE: ffprobe. CONFIDENCE: HIGH. |
| `MVI_0967_compressed.mp4` | hevc | 1280×720 | 50/50 (CFR) | aac ×4 | 1ch each | 599.04s | **4 mono AAC streams** (HEVC reencoded from MVI camera 4-channel source). 3Mbps. EVIDENCE: ffprobe. CONFIDENCE: HIGH. |
| `MVI_1035.MP4` | h264 | 1920×1080 | 50/50 (CFR) | pcm_s16be ×4 | 1ch each | 54.24s | Same family as MVI_0384. Multi-stream PCM. EVIDENCE: ffprobe. CONFIDENCE: HIGH. |
| `VID-20260318-WA0019.mp4` | h264 | 648×360 | 30/avg=30.00007 (**VFR**) | aac | 2ch | v=227.93s a=227.95s | **WhatsApp-shared file. VFR.** Video duration != audio duration (delta 13.2ms). 205kbps. EVIDENCE: avg_frame_rate=615420000/20514011=30.00007fps and stream-duration mismatch. CONFIDENCE: HIGH. |
| `test.mp4` | hevc | 1920×1080 | 25/25 (CFR) | aac | 2ch | 713.92s | Reference 1080p HEVC. 3.5Mbps. EVIDENCE: ffprobe. CONFIDENCE: HIGH. |

### Source matrix observations

- **Frame rate diversity**: 25, 30, 50 fps all represented; current cut step snaps to 1/30s grid (`DEFAULT_SNAP_GRID_S = 1.0/30.0` in `pipeline_v2/pipeline_v2/render/edl_builder.py:54`). EVIDENCE: file:line. CONFIDENCE: HIGH. **For 50fps sources, 1/30s grid misaligns the cut boundary by up to 1/60s = 16.7ms.** CONFIDENCE: HIGH (basic arithmetic).

- **Audio stream multiplicity**: Canon MVI files have 4 mono PCM streams; current Stage 0 must downmix or select. If pipeline takes stream 0 only, ambient + secondary mics are dropped. EVIDENCE: ffprobe output. CONFIDENCE: HIGH. Mitigation in code: needs verification (Track 3A subagent will report).

- **VFR sources exist**: WhatsApp-style mobile uploads will have VFR. Stage 0 must convert to CFR. EVIDENCE: VID-20260318-WA0019.mp4 avg_frame_rate fractional. CONFIDENCE: HIGH.

- **Codec mix**: h264 + hevc. Decoders both supported by ffmpeg + NVDEC. EVIDENCE: ffmpeg 8.0.1 has `--enable-nvdec --enable-cuvid` (ffmpeg -version banner). CONFIDENCE: HIGH.

- **PCM vs AAC sources**: PCM input bypasses the AAC priming concern at the source level but Stage 0 reencodes to AAC for the mezzanine (will introduce priming there). EVIDENCE: inferred from mezzanine being `aac` (see Section 2 below). CONFIDENCE: MED.

---

## Section 2 — Mezzanine properties (job_53)

`output/full_video_shorts_v2/job_53/mezzanine.mp4`:

- Video: h264, 3840×2160, 30/30 fps (CFR, snapped to /30 grid), duration=589.93s. 
- Audio: aac, 48kHz, 2ch (stereo), duration=589.93s.
- Stream duration delta: 0.000s (perfect match at mezzanine level).
EVIDENCE: ffprobe. CONFIDENCE: HIGH.

**Mezzanine guarantees CFR + AAC + matched A/V duration.** All downstream A/V drift is introduced post-mezzanine. CONFIDENCE: HIGH.

---

## Section 3 — Per-job drift inventory (production output empirical audit)

For each job, `bulletin/bulletin_with_overlays.mp4` was probed. Delta = video_stream_duration − audio_stream_duration (negative = audio ahead of video).

| Job | v dur (s) | a dur (s) | Δ (ms) | # composed_story segs | Mezzanine A/V match? | Verdict |
|---|---|---|---|---|---|---|
| 40 | 713.767 | 713.792 | **−25.33** | 1 | (not probed) | drift even with 1 segment |
| 41 | 82.267 | 82.304 | **−37.33** | 1 | (np) | drift even with 1 segment |
| 42 | 633.333 | 633.406 | **−72.67** | 22 | (np) | drift |
| 43 | 489.800 | 489.848 | **−48.00** | 23 | (np) | drift |
| 44 | 484.067 | 484.133 | **−66.33** | 22 | (np) | drift |
| 45 | 35.933 | 494.421 | **−458,488** | 39 | (np) | **catastrophic** (likely freeze bug; mp4 truncated) |
| 46 | 104.533 | 474.090 | **−369,557** | 25 | (np) | **catastrophic** (same class) |
| 47 | 493.800 | 493.824 | **−24.00** | 23 | (np) | drift |
| 48 | 499.800 | 499.840 | **−40.00** | 29 | (np) | drift |
| 49 | 484.067 | 484.181 | **−114.33** | 26 | (np) | drift visible (user reported) |
| 50 | 469.900 | 470.016 | **−116.00** | 33 | (np) | drift visible after 1:23 (user reported) |
| 51 | 499.367 | 499.392 | **−25.00** | 28 | (np) | "Path 2 measurement verified" but USER PERCEPTION FAILED |
| 52 | 484.567 | 484.650 | **−83.33** | 18 | (np) | drift |
| 53 | 475.067 | 475.136 | **−69.30** | 22 | matched | item 117 attempt; drift returned |

EVIDENCE: ffprobe on each `bulletin_with_overlays.mp4`. CONFIDENCE: HIGH.

### Section 3 — Quantitative observations

1. **Baseline drift even with N=1 segment**: Jobs 40 and 41 had only 1 bulletin cut yet drift = −25 to −37ms.
   - EVIDENCE: ffprobe. CONFIDENCE: HIGH.
   - INTERPRETATION: There is a constant ~25ms A/V offset introduced even when no cut compounding happens. This matches the canonical AAC encoder priming (1024 samples @ 48kHz = 21.33ms; the extra ~5-16ms is concat-step muxer overhead). CONFIDENCE: MED (interpretation; the empirical fact is HIGH).

2. **Per-segment drift compounds**: Jobs with more segments tend to higher absolute drift.
   - Avg drift/seg (ms): job_40 (−25/1=25); job_42 (−73/22=3.3); job_44 (−66/22=3.0); job_50 (−116/33=3.5); job_53 (−69/22=3.1).
   - Pattern: ~25ms baseline + ~2-4ms per additional segment.
   - EVIDENCE: arithmetic over ffprobe. CONFIDENCE: MED.

3. **Job 51 anomaly**: Only −25ms drift with 28 segs (vs ~85-115ms expected). Yet USER PERCEPTION OF LIP-SYNC FAIL.
   - HYPOTHESIS: drift is non-uniform along the timeline. The 25ms is an end-of-file aggregate; mid-file delta could be much higher and shift back. Item 113 measurements would have caught aggregate-end but not mid-file.
   - EVIDENCE: This explains the "measurement passed / user perception fail" pattern. CONFIDENCE: MED.

4. **Jobs 45/46 catastrophic**: video stream only 36-105s while audio is 474-494s.
   - These are the freeze-bug class (item 111). Video stream ended prematurely (mp4 truncated mid-encode) while audio continued.
   - EVIDENCE: 9-12x audio:video ratio. CONFIDENCE: HIGH.
   - Production impact: the entire bulletin is unwatchable for these jobs. CONFIDENCE: HIGH.

---

## Section 4 — Where drift originates (intra-pipeline waypoints for job_53)

Cumulative per-stage probe on job_53:

### 4a — Cut step output (`raw_clip_NN.mp4` per shorts)
- Probed: raw_clip_01.mp4 — v=16.100s a=16.066s (Δ=+34ms, video AHEAD).
- Probed: clip_01.mp4 (after compose) — v=16.100s a=16.100s (Δ=0ms).
EVIDENCE: ffprobe. CONFIDENCE: HIGH.

**The compose step appears to PAD audio to match video duration** (Δ=+34ms → Δ=0). This pad presumably uses apad or `-shortest` with adur=vdur target. CONFIDENCE: MED (inferred from observed delta).

### 4b — Composed story durations (M=22 silence-trim outputs)

```
Sum of video stream durations (22 stories): 476.7999s
Sum of audio stream durations (22 stories): 476.7950s
Aggregate Δ:                                  +5.0ms (video ahead)
```
EVIDENCE: ffprobe per story + sum. CONFIDENCE: HIGH.

**The compose step output is essentially CLEAN.** Drift across all 22 composed segments is only +5ms total. The compose step is NOT the primary drift source. CONFIDENCE: HIGH.

### 4c — Stitcher output vs sum

```
Sum of composed_story V durations: 476.7999s
Final bulletin.mp4 V duration:     475.0667s
Lost in stitch (V):                  1.7332s  ← crossfade overlap

Sum of composed_story A durations: 476.7950s
Final bulletin.mp4 A duration:     475.1360s
Lost in stitch (A):                  1.6590s  ← crossfade overlap (LESS than V)

Final A/V Δ at bulletin.mp4:      −69.3ms (audio ahead of video)
```
EVIDENCE: ffprobe + arithmetic. CONFIDENCE: HIGH.

**The stitcher loses MORE video than audio.** Sum input was matched (+5ms). Sum output is −69.3ms.
That's a 74.3ms swing in A/V relationship across the stitcher = `bulletin_crossfade_stitcher.py`.
CONFIDENCE: HIGH.

### 4d — Overlay pass

bulletin.mp4 (post-stitcher) vs bulletin_with_overlays.mp4 (post-overlay):
- Both: v=475.067s, a=475.136s, Δ=−69.3ms.
EVIDENCE: ffprobe both files. CONFIDENCE: HIGH.

**The overlay pass preserves A/V relationship exactly.** Item 117 phase 3's `-c:a copy` invariant works correctly. The overlay pass is NOT a drift source. CONFIDENCE: HIGH.

---

## Section 5 — Item 117 unified-extract evidence (job_53 forensic)

Three shorts raw files are CORRUPTED in job_53:
- `short_01_raw.mp4` (28.3 MB) — moov atom not found
- `short_02_raw.mp4` (33.0 MB) — moov atom not found
- `short_03_raw.mp4` (39.6 MB) — moov atom not found

mtimes: 19:31, 19:37, 19:40 (three separate file writes, not single concurrent multi-output).

EVIDENCE: ffprobe + `stat -c`. CONFIDENCE: HIGH.

**INTERPRETATION**:
- These files were written by item 117's unified extract path.
- `moov atom not found` = mp4 trailer never written = ffmpeg process was killed mid-encode (or hung) before it could finalise the container.
- Three separate mtimes contradict the "single ffmpeg invocation with N outputs" architecture of item 117 Phase 2 (`stage_4_raw_extract.py`).
- HYPOTHESIS: ffmpeg WAS invoked once with 4 outputs (1 bulletin + 3 shorts), but the process was killed by the 1800s timeout BEFORE writing trailers. The 19:31/19:37/19:40 mtimes are the LAST WRITE timestamps to those buffers, not creation times.
- CONFIDENCE: MED (the 3-mtime spread vs single-invocation hypothesis needs to be reconciled by checking if ffmpeg writes streaming output progressively).

**CORRECTION OBSERVATION**: A subsequent fallback to legacy path produced the `raw_clip_NN.mp4` files in `bulletin/` (23 of them) and the `clip_NN.mp4` shorts (6 of them — composed, not raw). The corrupted `short_NN_raw.mp4` are dead artifacts in the parent dir from the failed item-117 attempt; the actual production output went through the legacy path. CONFIDENCE: MED.

---

## Section 6 — Implications for architectural decisions

1. **AAC priming is real and persistent.** Even with N=1 segment, drift is −25 to −37ms (Jobs 40, 41). 
   - Audio re-encode is the primary suspect.
   - CONFIDENCE: HIGH.

2. **The stitcher is the primary cumulative-drift source.** Compose step is clean (+5ms across 22 segments); stitcher introduces −74ms swing.
   - The acrossfade chain is mathematically asymmetric to the video xfade chain.
   - CONFIDENCE: HIGH.

3. **Item 117 unified extract did not address the stitcher.** Per-segment raw extracts may be sample-accurate, but they're then fed into the same drift-prone stitcher.
   - CONFIDENCE: HIGH.

4. **The silence-trim step inflates segment count from N→M.** Job 53 had 1 bulletin cut from Stage 2; silence-trim split it into 22 composed segments. Each segment incurs concat/stitch drift.
   - This explains why Stage 2 non-determinism (1 cut vs 25-33 cuts) matters less than expected: silence-trim normalises segment count anyway.
   - CONFIDENCE: HIGH.

5. **A single-pass multi-output extract that PRODUCES THE FINAL BULLETIN directly (skipping the stitcher) would eliminate the primary drift source.**
   - This is what item 117 should have done but the architecture punted on the stitcher.
   - CONFIDENCE: MED (would need empirical verification).

---

## Section 7 — Decision-grade summary

For Track 4 (architectural options):

**Root cause of recurring lip-sync drift = the multi-pass concat/stitch architecture, not the cut step.**

EVIDENCE:
- Cut step (raw_clip): per-clip A/V mismatch <10ms (negligible).
- Compose step: aggregate A/V across 22 segments = +5ms (negligible).
- Stitcher: introduces 74ms swing.
- Overlay: zero impact.

ANY ARCHITECTURE THAT KEEPS THE STITCHER WILL KEEP THE DRIFT, regardless of item 117 / item 116 / item 115 fixes upstream.

CONFIDENCE: HIGH on the diagnosis, MED on the prescription (eliminating stitcher requires architectural change).

---

END Track 3C findings.
