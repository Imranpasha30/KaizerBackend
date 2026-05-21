"""Find any job referencing Personal_10 across all platforms."""
import io
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

s = SessionLocal()
rows = s.execute(text(
    "SELECT id, platform, video_name, status, finished_at, output_dir "
    "FROM jobs "
    "WHERE video_name ILIKE '%Personal%' OR video_name ILIKE '%personal%' "
    "ORDER BY id DESC LIMIT 10"
)).fetchall()
print(f"Jobs with Personal in name: {len(rows)}")
for r in rows:
    print(f"  id={r[0]:>3}  platform={r[1]:<30}  name={r[2]!r:<40}  status={r[3]:<10}  finished={r[4]}")
    print(f"        out={r[5]}")

print()
print("Latest 8 jobs (any platform):")
rows = s.execute(text(
    "SELECT id, platform, video_name, status, finished_at "
    "FROM jobs ORDER BY id DESC LIMIT 8"
)).fetchall()
for r in rows:
    print(f"  id={r[0]:>3}  platform={r[1]:<30}  name={r[2]!r:<40}  status={r[3]:<10}  finished={r[4]}")
s.close()
