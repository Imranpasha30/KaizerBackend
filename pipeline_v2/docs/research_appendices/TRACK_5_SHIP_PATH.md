# TRACK 5: REALISTIC SHIP PATH

## 5A. HONEST ASSESSMENT
- **Can V2 iter-2 ship today?** NO. The product produces audio-video drift up to -695ms. This breaks the fundamental requirement of video editing. Furthermore, the `subprocess.PIPE` deadlock on Windows means it will randomly fail in production under heavy load.
- **Gap to Creator-Tier (Tier B)**: 3-4 weeks. Requires rewriting Stage 4 to Option B (OTIO timeline + single-pass extract). We already proved the `filter_complex` scales flawlessly in Empirical Test #1. 
- **Gap to Regional News-Tier (Tier A)**: 2-3 months. Beyond perfect A/V sync, this tier requires dynamic motion graphics, multiple overlay tracks, and automated ducking of B-roll audio under anchor dialogue. Doing this via FFmpeg `filter_complex` alone is a massive undertaking. 
- **Gap to BBC-Tier (Tier S)**: 12-18 months, perhaps impossible purely autonomously. Achieving "natural human pacing" requires contextual AI understanding of dramatic pauses, which current STT models (Deepgram/Whisper) strip out as "silence".

## 5B. MARKET POSITIONING OPTIONS
1. **Premium News Automation (BBC-tier)**: Impossible in 2026 without human-in-the-loop.
2. **Creator Economy Tool (Beer Biceps-tier)**: Highly viable. Competing with Opus Clip. Requires fast jump cuts, dynamic text, and zero drift. Pricing can be $20-$30/mo.
3. **Mid-tier News (TV9-tier)**: The current stated goal. High engineering cost due to graphic complexity. High value for 1000+ users if achieved.

## 5C. RECOMMENDED SHIP PATH

### IMMEDIATE (0-2 Weeks): Stability & Core Architecture
- Abandon the legacy `cut -> compose -> stitch` pipeline.
- Implement OPTION B: OTIO EDL -> `filter_complex` graph.
- Implement the `stderr` fix for subprocess deadlocks.
- *Goal*: Zero lip-sync drift, zero timeouts.

### NEAR-TERM (1-3 Months): The Creator MVP
- Add basic OTIO transitions (crossfade) and single-layer static overlays.
- Ship to beta testers as a high-fidelity "Opus Clip for Telugu/Hindi" that doesn't mess up translation.
- *Goal*: Revenue generation, proving the new rendering core.

### MEDIUM-TERM (3-6 Months): The Broadcaster MVP
- Add complex OTIO nesting (PiP tracks, dynamic tickers).
- Build a Web UI (Descript-style) that visualizes the OTIO timeline so the 1000+ news users can correct the 5% of cuts the AI gets wrong.
- *Goal*: The killer feature "upload the video and see the performance".

### VERDICT FOR THE USER
You cannot ship V2 iter-2. Stop trying to patch `_cut_one_clip_strict`. 
The product relies on 4 sequential FFmpeg generations; this is mathematically doomed to drift due to AAC framing and PTS resets.
Accept the sunk cost of Stage 4. Rewrite Stage 4 to compile an EDL (OpenTimelineIO) into a single FFmpeg `filter_complex` command.
