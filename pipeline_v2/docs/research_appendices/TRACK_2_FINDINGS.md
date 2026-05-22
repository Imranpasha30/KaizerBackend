# TRACK 2: BROADCAST/EDITING TOOL ARCHITECTURE

## 2A. EDIT DECISION LISTS (EDLs)
- **CMX 3600**: The ancient standard. Very basic, just IN/OUT timecodes. Too limited for overlays or complex text.
- **FCPXML / Premiere XML**: Extremely verbose but industry standard. Good for exporting to NLEs, but complex to parse/write natively in python without a wrapper.
- **OpenTimelineIO (otio)**: Modern, open-source API (Python/C++) by Pixar/ASWF. 
  - *Critique*: OTIO is perfect for representing Stage 2 decisions as a proper EDL. It supports clips, transitions, and nested tracks. We could build our timeline in OTIO, then compile that OTIO object down to an FFmpeg filter graph. This separates "editing logic" from "ffmpeg syntax".

## 2B. RENDER ENGINES & EMPIRICAL FFmpeg TESTS

**Empirical Test #1: filter_complex Scaling**
*Test: Build filter graphs with 5/10/20/50/100 trim nodes concatenated.*
- H264 (C7040.mp4) @ 100 nodes: Success. 5.64s elapsed.
- HEVC (test.mp4) @ 100 nodes: Success. 2.03s elapsed.
*Insight*: FFmpeg's `filter_complex` easily handles 100+ trim nodes without crashing or running out of memory. This validates that a single-pass monolithic filter graph is computationally viable and likely much faster than writing 100 intermediate temp files to disk.

**Empirical Test #2: NVENC Concurrent Session Limits**
*Test: Run 1 to 8 simultaneous NVENC encodes on RTX 5060.*
- 8 concurrent sessions succeeded in 4.01s total.
- *Insight*: NVIDIA recently unlocked the NVENC session limit on consumer GPUs (previously locked to 3, now essentially bounded by VRAM). We can run highly parallel extraction if we use separate processes.

**Subprocess pipe behavior:**
- Nondeterministic timeouts on Windows often stem from unread stdout/stderr pipes filling the OS buffer and deadlocking the child process. The code must ensure `stderr=subprocess.PIPE` is continuously read or sent to `DEVNULL`.

## 2C. CLOUD VIDEO PROCESSING
- **AWS MediaConvert**: High-end broadcast, slow start times. Not ideal for same-hour news.
- **Mux / Cloudflare Stream**: Excellent for delivery, weak on complex multi-layer editing/composition.
- **Runway / Synthesia**: Too expensive API cost for high-volume 1000+ user SaaS.
- **Cost Analysis**: A 5-minute bulletin with multi-layer graphics on AWS MediaLive/MediaConvert can cost $0.50-$2.00 per render. Self-hosting a consumer RTX GPU (like the 5060) can render thousands of minutes a day for a flat hardware cost. For 1000+ users, self-hosted GPU rendering is vastly superior economically.

## 2E. SYNTHESIS
Current Kaizer V2 cuts individual clips to disk, then stitches them. This violates the NLE pattern. Professional NLEs (Premiere, Resolve) never cut intermediate files; they read the source once, apply a timeline model, and render the final output in one pass.
- **Accidental divergence**: Kaizer's multi-step render (extract -> compose -> stitch) introduces multiple points of a/v sync failure (AAC priming, PTS resets).
- **Pattern to adopt**: Build an OTIO timeline memory object, compile it to a single `filter_complex` script, and render in ONE pass directly from the mezzanine.
