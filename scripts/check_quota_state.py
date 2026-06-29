"""Print today's quota usage vs cap (debugging test-run residue)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import SessionLocal
from youtube import quota_v2

db = SessionLocal()
try:
    print(quota_v2.snapshot(db))
finally:
    db.close()
