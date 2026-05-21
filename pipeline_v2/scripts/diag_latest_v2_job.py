"""Quick diag: report the latest V2 job + tail of its log."""
import io
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass

HERE = Path(__file__).resolve().parent
BACKEND_ROOT = HERE.parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from dotenv import load_dotenv
load_dotenv(BACKEND_ROOT / ".env", override=True)

from database import SessionLocal
from sqlalchemy import text

sql = """
SELECT id, video_name, status, finished_at, output_dir,
       transition_style, log
FROM jobs
WHERE platform = 'full_video_shorts_v2'
ORDER BY id DESC
LIMIT 1
"""

s = SessionLocal()
r = s.execute(text(sql)).fetchone()
s.close()

print(f"id            : {r[0]}")
print(f"video_name    : {r[1]}")
print(f"status        : {r[2]}")
print(f"finished_at   : {r[3]}")
print(f"output_dir    : {r[4]}")
print(f"transition_st : {r[5]}")
print()
print("log (tail 4000 chars):")
print((r[6] or "")[-4000:])
