# TRACK 3: CURRENT PIPELINE DEEP DIVE

## 3C. SOURCE VIDEO PROPERTY MATRIX

This matrix identifies the fundamental video properties of the various source videos we encounter. Inconsistencies here (like VFR vs CFR, or variable timebases) are often the root cause of drift when processing blindly through `ffmpeg`.

| Video File | Codec | R Frame Rate | Avg Frame Rate | Time Base | Duration (s) | Hypothesis on Sync Risk |
|---|---|---|---|---|---|---|
| **C7040.mp4** | h264 | 25/1 | 25/1 | 1/12800 | 589.92 | LOW risk. Clean CFR. Drift here implies pipeline bugs, not source issues. |
| **test.mp4** | hevc | 25/1 | 25/1 | 1/57600 | 713.88 | LOW-MED. HEVC decode can be slower, potentially causing desync on weak hardware, but CFR is solid. |
| **MVI_0967_compressed.mp4** | hevc | 50/1 | 50/1 | 1/57600 | 599.04 | MED. 50fps HEVC requires double the decoding power. |
| **C0004 (1).mp4** | h264 | 50/1 | 50/1 | 1/12800 | 756.00 | MED. 50fps source needs careful handling if mixed with 25fps graphics/overlays. |
| **MVI_1384.MP4** | h264 | 50/1 | 50/1 | 1/50000 | 171.36 | MED. 50fps CFR. |
| **WhatsApp Video...** | h264 | 25/1 | 25/1 | 1/25000 | 82.28 | **HIGH risk.** WhatsApp aggressively compresses and often outputs VFR (Variable Frame Rate) disguised as CFR, leading to audio drift if not explicitly converted during mezzanine generation. |
| **fuul video_compressed** | hevc | 25/1 | 25/1 | 1/57600 | 713.88 | LOW-MED risk. |

**Observation**: The variety in timebases (from `1/12800` to `1/57600`) means that doing direct stream copy (`-c copy`) or concatenation without re-encoding to a unified mezzanine timebase will *always* result in PTS (Presentation Timestamp) drift. Any architectural solution must enforce a strict, unified timebase at the ingest boundary (Stage 0).

## 3A. DRIFT INTRODUCTION POINT MAP (In Progress)

### Stage 0: Mezzanine Generation
- *Investigation pending code review.*

### Stage 4: Cut Step (Legacy Path)
- *Investigation pending code review.*

### Stage 4: Stitcher & Overlays
- *Investigation pending code review.*

### Item 117 Unified Extract
- *Investigation pending code review.*
