"""Live smoke of the quota-truth + job-wise Publishes endpoints."""
import json
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import httpx

import auth
import models
from database import SessionLocal

db = SessionLocal()
# Prefer a user who actually OWNS publish tasks so the enrichment
# assertions run against real rows.
owner_row = (
    db.query(models.PublishTask.user_id)
    .order_by(models.PublishTask.id.desc())
    .first()
)
if owner_row:
    u = db.query(models.User).filter(models.User.id == owner_row[0]).first()
else:
    u = db.query(models.User).filter(models.User.is_active == True).order_by(  # noqa: E712
        models.User.id).first()
print(f"smoking as user id={u.id} ({u.email})")
tok = auth.issue_token(u)
db.close()
H = {"Authorization": f"Bearer {tok}"}
BASE = "http://localhost:8000"

with httpx.Client(timeout=30) as c:
    q = c.get(f"{BASE}/api/quota", headers=H).json()
    print("GET /api/quota ->", q)
    assert q["limit"] == 10000, f"limit should be the real 10000, got {q['limit']}"
    assert q["used"] == 0, f"used should be 0 (rebased), got {q['used']}"

    lst = c.get(f"{BASE}/api/publish-tasks", params={"limit": 5}, headers=H).json()
    print(f"GET /api/publish-tasks -> total={lst['total']}, "
          f"items_on_page={len(lst['items'])}")
    enrich_keys = {"video_title", "thumb_url", "channels_preview",
                   "parked_count", "retrying_count", "in_flight_count",
                   "cancelled_count", "publish_kind"}
    if lst["items"]:
        item = lst["items"][0]
        missing = enrich_keys - set(item.keys())
        assert not missing, f"missing enrichment keys: {missing}"
        print("  first card:", json.dumps({
            k: item[k] for k in ("id", "status", "target_count",
                                 "completed_count", "failed_count",
                                 "video_title", "channels_preview")
        }, ensure_ascii=False)[:300])

        det = c.get(f"{BASE}/api/publish-tasks/{item['id']}", headers=H)
        print(f"GET /api/publish-tasks/{item['id']} -> {det.status_code}")
        if det.status_code == 200:
            d = det.json()
            kids = d.get("upload_jobs", [])
            print(f"  children: {len(kids)}")
            if kids:
                k0 = kids[0]
                child_keys = {"channel_name", "video_url", "last_error",
                              "bytes_uploaded", "next_attempt_at", "branded"}
                missing = child_keys - set(k0.keys())
                assert not missing, f"missing child keys: {missing}"
                print("  first child:", json.dumps({
                    kk: k0[kk] for kk in ("id", "status", "channel_name",
                                          "upload_path", "video_url")
                }, ensure_ascii=False)[:300])
    else:
        print("  (no publish tasks for this user — shape check on items "
              "skipped; endpoint healthy)")

print("\nPUBLISHES API SMOKE: PASS")
