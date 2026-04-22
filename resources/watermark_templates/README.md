Place PNG templates of known platform watermarks in this directory (e.g. `tiktok.png`, `capcut.png`, `snap.png`, `youtube_shorts.png`). Each template should be the watermark cropped from a real screenshot, on a transparent background if possible. `detect_watermarks()` runs `cv2.matchTemplate` against each PNG at multiple scales.

No templates shipped by default — the detection falls back to an 'info' alert until the user or ops team seeds this directory.
