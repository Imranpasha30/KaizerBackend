"""Diagnose the REAL YouTube Data API quota allocation.

(1) Our recorded usage today (forensic log + burn log) — what WE think
    we burned.
(2) The service-account project vs the OAuth-client project — the
    quota belongs to the OAuth client's project.
(3) Best-effort: query Google's ACTUAL assigned quota via the Service
    Usage API, if a usable credential exists for that project.

Run from KaizerBackend/:  python scripts/check_yt_quota.py
"""
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dotenv import load_dotenv  # noqa: E402
load_dotenv()


def section(t):
    print("\n" + "=" * 64 + "\n" + t + "\n" + "=" * 64)


# ── (1) Our recorded usage ──────────────────────────────────────────
section("(1) What OUR system recorded as used")
try:
    from sqlalchemy import text
    from database import SessionLocal
    db = SessionLocal()
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # Forensic log (youtube_api_calls) — actual API calls made.
        for tbl, costcol in (("youtube_api_calls", "quota_cost"),
                             ("youtube_api_calls", "cost"),
                             ("youtube_api_calls", "units")):
            try:
                rows = db.execute(text(
                    f"SELECT operation, count(*), COALESCE(sum({costcol}),0) "
                    f"FROM youtube_api_calls "
                    f"WHERE CAST(created_at AS DATE) = CAST(now() AS DATE) "
                    f"GROUP BY operation ORDER BY 3 DESC"
                )).fetchall()
                print(f"  forensic youtube_api_calls today (cost col '{costcol}'):")
                tot = 0
                for op, n, c in rows:
                    print(f"    {op:<28} calls={n:<4} units={int(c)}")
                    tot += int(c)
                print(f"    {'TOTAL':<28} units={tot}")
                break
            except Exception:
                db.rollback()
                continue
    finally:
        db.close()
except Exception as e:
    print("  (forensic log query failed:", e, ")")

# quota_v2 snapshot — what the GATE thinks (uses the env cap, not real)
try:
    from database import SessionLocal
    from youtube import quota_v2
    db = SessionLocal()
    try:
        snap = quota_v2.snapshot(db)
    finally:
        db.close()
    print("\n  quota_v2 gate snapshot:", snap)
    print("  NOTE: 'limit' above is KAIZER_YT_DAILY_QUOTA_CAP (our env"
          " placeholder), NOT Google's real allocation.")
except Exception as e:
    print("  (quota_v2 snapshot failed:", e, ")")


# ── (2) Project identity ────────────────────────────────────────────
section("(2) Which GCP project owns the YouTube quota?")
client_id = os.environ.get("YOUTUBE_CLIENT_ID", "")
oauth_proj_num = client_id.split("-")[0] if client_id else "?"
print(f"  OAuth client project NUMBER (owns YT quota): {oauth_proj_num}")
print(f"  KAIZER_GCP_PROJECT (Vertex/Gemini):          "
      f"{os.environ.get('KAIZER_GCP_PROJECT', '?')}")

sa_project = None
sa_email = None
cred_path = os.environ.get("KAIZER_VERTEX_CREDENTIALS") \
    or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if cred_path and os.path.isfile(cred_path):
    try:
        with open(cred_path, encoding="utf-8") as f:
            sa = json.load(f)
        sa_project = sa.get("project_id")
        sa_email = sa.get("client_email")
        print(f"  Service-account project_id:                  {sa_project}")
        print(f"  Service-account email:                       {sa_email}")
    except Exception as e:
        print("  (couldn't read service account json:", e, ")")
else:
    print("  (no service-account JSON file found at the configured path)")


# ── (3) Query Google's ACTUAL assigned quota ────────────────────────
section("(3) Google's ACTUAL assigned YouTube quota (Service Usage API)")
print("  YouTube Data API exposes NO endpoint for the quota limit, so")
print("  we ask the Service Usage API instead. This needs a credential")
print("  with serviceusage.quotas.get on the OAuth-client's project.")
print()

try:
    import google.auth
    from googleapiclient.discovery import build

    creds = None
    used = None
    # Prefer the service account if present; else Application Default.
    if cred_path and os.path.isfile(cred_path):
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(
            cred_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        used = f"service account ({sa_email})"
    else:
        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"])
        used = "application default credentials"
    print(f"  trying with: {used}")

    su = build("serviceusage", "v1beta1", credentials=creds,
               cache_discovery=False)

    # The quota lives on the OAuth client's project. We only have its
    # NUMBER; Service Usage accepts projects/<number>.
    targets = []
    if oauth_proj_num and oauth_proj_num.isdigit():
        targets.append(oauth_proj_num)
    if sa_project:
        targets.append(sa_project)

    found = False
    for proj in targets:
        # parent pattern is projects/<id>/services/<svc>; the method
        # appends consumerQuotaMetrics itself.
        parent = f"projects/{proj}/services/youtube.googleapis.com"
        try:
            resp = su.services().consumerQuotaMetrics().list(
                parent=parent, view="BASIC").execute()
        except Exception as e:
            print(f"  project {proj}: query failed -> {str(e)[:160]}")
            continue
        metrics = resp.get("metrics", [])
        if not metrics:
            print(f"  project {proj}: no metrics returned "
                  f"(API maybe not enabled here)")
            continue
        print(f"\n  ✅ project {proj} — YouTube Data API quota metrics:")
        for m in metrics:
            disp = m.get("displayName", m.get("metric", "?"))
            for lim in m.get("consumerQuotaLimits", []):
                for b in lim.get("quotaBuckets", []):
                    eff = b.get("effectiveLimit")
                    default = b.get("defaultLimit")
                    if eff is not None:
                        print(f"     {disp}: effective_limit={eff} "
                              f"(default={default})")
                        found = True
        if found:
            break
    if not found:
        print("\n  Could not read the effective limit with our credentials")
        print("  (the service account is almost certainly in a DIFFERENT")
        print("   project than the one that owns the YouTube quota).")
except Exception as e:
    print(f"  Service Usage query unavailable: {str(e)[:200]}")

# ── Verdict ─────────────────────────────────────────────────────────
section("HOW TO CONFIRM DEFINITIVELY (manual, 20 seconds)")
print(f"  Open the YouTube Data API quota page for the OWNING project:")
print(f"  https://console.cloud.google.com/apis/api/"
      f"youtube.googleapis.com/quotas?project={oauth_proj_num}")
print()
print("  Look at 'Queries per day'. If it reads 10,000 you were NOT")
print("  raised; anything higher (e.g. 1,000,000) means the increase")
print("  was granted. Used 31k without a 403 quotaExceeded == it's")
print("  already above 10,000.")
