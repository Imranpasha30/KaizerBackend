# TRACK 3D: FAILURE MODE INVENTORY

## Failure 1: Job 50 lip-sync drift after compose-step AAC residue
- **Root cause**: FFmpeg's AAC encoder introduces priming samples (usually 1024 or 2048 samples) at the start of every encoded AAC file. When the legacy pipeline cuts clips into individual MP4 files, each clip gets these priming samples. When stitched back together, the total audio track grows longer than the video track, causing progressive lip-sync drift (-695ms over a bulletin).
- **Fix attempted**: Item 117 (Unified Extract) attempted to solve this by decoding the mezzanine once and extracting directly without intermediate cut-recompilation.
- **Evidence**: `track3_output.txt` shows A/V duration deltas.

## Failure 2: Item 117 production timeout
- **Root cause**: `stage_4_raw_extract.py:326` calls `subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)`. When FFmpeg generates a massive `filter_complex` graph (e.g. 50+ cuts), it can write thousands of lines of warnings to `stderr`. On Windows, the OS pipe buffer for stdout/stderr is typically 64KB. If `capture_output=True` is used and the buffer fills before the Python process reads it, the OS blocks FFmpeg from writing more, causing a deadlock. The `timeout_s` eventually hits and kills the job.
- **Evidence**: Standard Windows/Python `subprocess.PIPE` deadlock behavior. The fix is to either redirect stderr to a file descriptor or use `asyncio.create_subprocess_exec` and read stream incrementally.

## Failure 3: Item 116 cut step -to bug
- **Root cause**: When using `-ss` and `-to` without re-encoding (stream copy), ffmpeg cuts on nearest I-frames (keyframes). Since the mezzanine is H264 with typical GOP sizes (e.g., 30-250 frames), the actual cut can be off by up to several seconds from the requested timestamp, destroying continuity.

## 3E. ALL OPEN BACKLOG ITEMS REVIEW
- The backlog identifies 17+ piecemeal fixes (Item 111 freeze, Item 115 AAC priming, Item 117 extract timeout). 
- **Assessment**: The current piecemeal approach is fundamentally flawed. Fixing AAC priming in cut-step just uncovers PTS reset bugs in the stitcher. Fixing the stitcher uncovers memory leaks in overlay. The entire `cut -> compose -> stitch -> overlay` architecture (4 sequential encoding generations) is lossy, slow, and mathematically impossible to keep perfectly synced without a unified timeline model.
