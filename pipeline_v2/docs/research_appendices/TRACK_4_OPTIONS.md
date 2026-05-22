# TRACK 4: ARCHITECTURAL OPTIONS (SYNTHESIS)

## OPTION A: Incremental V2 Hardening
1. **Conceptual Model**: Keep the current `cut -> compose -> stitch -> overlay` architecture.
2. **Data Flow**: Mezzanine -> cut individual MP4s -> burn sidebar overlays per clip -> stitch with crossfades -> burn final bulletin overlays.
3. **A/V Sync Risk**: **HIGH**. Every step re-encodes AAC audio, adding priming samples. Re-encoding video resets PTS.
4. **Strengths**: 
   - Code already exists.
   - Easiest for debugging individual clips.
5. **Weaknesses**:
   - Lip-sync drift is mathematically unavoidable without deep, complex PTS offset hacking.
   - Extremely slow (4 encoding generations).
   - High disk I/O.
6. **Cost Estimate**: 80-120 engineering hours (chasing edge cases and platform-specific ffmpeg behavior).
7. **Risk of New Bugs**: **HIGH**.
8. **Suitability for SaaS Scale**: Poor. Too much disk footprint and processing time per user.
9. **Citations**: Job 50 drift failure, Item 115 AAC leak.

## OPTION B: EDL-Based Single-Pass Architecture (Recommended)
1. **Conceptual Model**: Replace all intermediate rendering with a single Edit Decision List (EDL) using OpenTimelineIO (OTIO). Stage 2 outputs an OTIO timeline. Stage 4 compiles that OTIO timeline into one massive FFmpeg `filter_complex` graph and renders the final output directly from the Mezzanine.
2. **Data Flow**: Stage 0 Mezzanine -> Stage 2 logic (JSON -> OTIO) -> Stage 4 Render (OTIO -> ffmpeg filter_complex -> Final Output). No intermediate files.
3. **A/V Sync Risk**: **LOW**. A single encode pass guarantees audio and video are derived from the exact same PTS base.
4. **Strengths**:
   - Zero cumulative AAC drift.
   - Extremely fast (Empirical Test #1 shows 100 filter nodes process in <6 seconds).
   - Aligns with professional NLE paradigms.
5. **Weaknesses**:
   - Massive `filter_complex` strings are hard to read and debug.
   - Requires full rewrite of Stage 4.
6. **Cost Estimate**: 60-80 engineering hours.
7. **Risk of New Bugs**: **MEDIUM**. The bugs will be syntax errors in the filter graph, not mysterious timing drifts.
8. **Suitability for SaaS Scale**: Excellent. CPU/RAM overhead is minimal compared to 4 separate re-encodes.
9. **Citations**: OTIO open-source standard, Empirical Test #1.

## OPTION C: Cloud-Native API Rebuild
1. **Conceptual Model**: Offload all video processing to AWS MediaConvert or Mux.
2. **Data Flow**: Upload Mezzanine to S3 -> Send JSON timeline to AWS MediaConvert -> AWS returns final MP4.
3. **A/V Sync Risk**: **ZERO**. AWS handles broadcast-level sync.
4. **Strengths**: 
   - No infrastructure to manage.
   - Auto-scales to 10,000+ users instantly.
5. **Weaknesses**:
   - Expensive per minute ($0.50+).
   - Hard to do complex dynamic HTML/React overlays (MediaConvert prefers static burn-ins).
6. **Cost Estimate**: 100-150 engineering hours (rewriting pipeline for cloud hooks).
7. **Risk of New Bugs**: **LOW**.
8. **Suitability for SaaS Scale**: Perfect scaling, but terrible margins for free/low-tier users.

## OPTION D: Descript-style Browser Render Pivot
1. **Conceptual Model**: Stop rendering on the backend. Send the mezzanine and JSON timeline to the frontend. Use WebAssembly FFmpeg or WebCodecs to render on the USER'S browser.
2. **Data Flow**: Mezzanine -> Browser -> User edits -> Local browser render -> Upload final.
3. **A/V Sync Risk**: **MEDIUM**. Browsers can struggle with heavy 4K encodes.
4. **Strengths**: Zero server compute cost.
5. **Weaknesses**: Total product pivot. Users on weak laptops will hate it.
6. **Cost Estimate**: 300+ engineering hours.
