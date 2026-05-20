"""Print transcript words near the 4 phase-2 partial-restart spots
Gemini Pro flagged in Job 44's verification audit."""
import io
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).resolve()
JOB_44 = HERE.parent.parent.parent / "output" / "full_video_shorts_v2" / "job_44"
data = json.loads((JOB_44 / "transcript.json").read_text(encoding="utf-8"))
words = data["words"]
print(f"Total words: {len(words)}")

spots = [("07:11", 431.0), ("07:33", 453.0), ("08:02", 482.0), ("08:12", 492.0)]
for label, sec in spots:
    print(f"\n=== {label} ({sec}s) ===")
    for i, w in enumerate(words):
        if abs(w["s"] - sec) < 6:
            print(f"  [{i:4d}] {w['s']:7.3f}-{w['e']:7.3f}: {w['w']!r}")
