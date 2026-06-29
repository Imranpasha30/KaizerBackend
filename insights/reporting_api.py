"""YouTube Reporting API v1 — per-video thumbnail impressions + CTR (the real Studio CTR).

These two metrics exist ONLY in the bulk **Reporting API** (report type
``channel_reach_basic_a1``) — NOT in the Analytics API ``reports.query`` (which is why the
legacy ``analytics/ctr.py`` silently reads CTR=0). The Reporting API is job-based + async:

  1. Create ONE reporting job per channel — idempotent (reuse if it already exists).
  2. YouTube generates daily CSV reports. On job creation it backfills only the **prior 30
     days** (NOT full history) and accrues forward; the first report lands **~48h** later.
  3. Poll once/day, download new reports, dedup re-issued periods by the latest
     ``createTime``, parse 5 columns by header, and **impression-weight** the CTR.

PRODUCT CAVEAT (surface in UI): thumbnail-CTR history begins ~30 days before the job was
created and grows forward; videos older than that have no CTR — it cannot be recovered.
So create the job at channel-connect time so the 30-day clock starts ASAP. Scope
``yt-analytics.readonly`` is sufficient (do NOT request the monetary scope).

The pure aggregation (``parse_csv`` + ``aggregate_thumbnail_ctr``) is unit-tested in
``scripts/test_insights_reporting.py``; the I/O wrappers degrade to ``{}`` on any failure
so the analyzer transparently falls back to its early-velocity CTR proxy.
"""
from __future__ import annotations

import csv
import io
import logging
from typing import List, Optional

log = logging.getLogger("kaizer.insights.reporting")

# channel_reach_basic_a1 = per-video THUMBNAIL impressions + thumbnail CTR (the Studio "CTR").
# This is the ONLY report we schedule for CTR.
#   • Do NOT use channel_cards_a1 — that's in-video info-CARD click rate, a different metric.
#   • channel_reach_combined_a1 adds traffic-source/device breakdown — a FUTURE option; not now.
REPORT_TYPE_ID = "channel_reach_basic_a1"
JOB_NAME = "kaizer-insights-thumb-ctr"
ANALYTICS_SCOPE = "https://www.googleapis.com/auth/yt-analytics.readonly"

# CSV columns — parse by HEADER NAME, never position (Google may append columns).
COL_DATE = "date"
COL_VIDEO = "video_id"
COL_IMPR = "video_thumbnail_impressions"
COL_CTR = "video_thumbnail_impressions_ctr"   # a fraction, e.g. 0.0543 = 5.43%


# ── pure aggregation (no I/O — unit-tested) ──────────────────────────────

def parse_csv(text: str) -> List[dict]:
    """Parse a report CSV (header row) into dict rows. Empty / header-only → []."""
    if not text or not text.strip():
        return []
    try:
        return list(csv.DictReader(io.StringIO(text)))
    except Exception:
        return []


def aggregate_thumbnail_ctr(reports: List[dict]) -> dict:
    """Pure. Given report items ``{"startTime","createTime","csv"}``, dedup re-issued
    periods (same ``startTime`` → keep the latest ``createTime``), then per video sum
    impressions and impression-weight CTR across days. Returns
    ``{video_id: {"impressions": int, "ctr": float 0..1, "since": "YYYY-MM-DD"}}``.

    CTR MUST be impression-weighted: ``sum(impr*ctr_day) / sum(impr)`` — never an
    unweighted average of the per-day CTR column.
    """
    # 1) dedup by period (startTime) keeping the report with the max createTime
    by_period: dict = {}
    for r in reports or []:
        st = r.get("startTime") or ""
        ct = r.get("createTime") or ""
        cur = by_period.get(st)
        if cur is None or ct >= (cur.get("createTime") or ""):
            by_period[st] = r
    # 2) accumulate per video: [sum_impr, sum_impr*ctr, earliest_date]
    acc: dict = {}
    for r in by_period.values():
        for row in parse_csv(r.get("csv") or ""):
            vid = (row.get(COL_VIDEO) or "").strip()
            if not vid:
                continue
            try:
                impr = int(float(row.get(COL_IMPR) or 0))
            except Exception:
                impr = 0
            try:
                ctr = float(row.get(COL_CTR) or 0)
            except Exception:
                ctr = 0.0
            date = (row.get(COL_DATE) or "").strip()
            a = acc.setdefault(vid, [0, 0.0, date])
            a[0] += impr
            a[1] += impr * ctr
            if date and (not a[2] or date < a[2]):
                a[2] = date
    out: dict = {}
    for vid, (sum_impr, sum_impr_ctr, since) in acc.items():
        out[vid] = {
            "impressions": int(sum_impr),
            "ctr": round(sum_impr_ctr / sum_impr, 6) if sum_impr > 0 else 0.0,
            "since": since,
        }
    return out


# ── I/O wrappers (live API — degrade to {} on any failure) ───────────────

def _reporting_client(creds):
    from googleapiclient.discovery import build
    return build("youtubereporting", "v1", credentials=creds, cache_discovery=False)


def ensure_job(client, *, name: str = JOB_NAME) -> Optional[str]:
    """Reporting job id for ``channel_reach_basic_a1``, creating it if missing. Idempotent —
    never creates a second job for the same report type. None on failure. Call this at
    channel-connect time so the 30-day backfill clock starts as early as possible."""
    try:
        page = None
        while True:
            resp = client.jobs().list(includeSystemManaged=True, pageToken=page).execute()
            for job in (resp.get("jobs") or []):
                if job.get("reportTypeId") == REPORT_TYPE_ID and not job.get("systemManaged"):
                    return job.get("id")
            page = resp.get("nextPageToken")
            if not page:
                break
        created = client.jobs().create(
            body={"reportTypeId": REPORT_TYPE_ID, "name": name}).execute()
        return created.get("id")
    except Exception as exc:
        log.warning("reporting ensure_job failed: %s", str(exc)[:200])
        return None


def list_reports(client, job_id: str, *, created_after: Optional[str] = None) -> List[dict]:
    """All report metadata for the job (paginated). ``created_after`` = RFC3339 cursor so a
    daily poll only pulls new reports."""
    out: List[dict] = []
    page = None
    while True:
        kwargs = {"jobId": job_id}
        if created_after:
            kwargs["createdAfter"] = created_after
        if page:
            kwargs["pageToken"] = page
        resp = client.jobs().reports().list(**kwargs).execute()
        out.extend(resp.get("reports") or [])
        page = resp.get("nextPageToken")
        if not page:
            break
    return out


def download_csv(client, download_url: str) -> str:
    """GET a report's CSV body (authorized via the client's http)."""
    try:
        _resp, content = client._http.request(download_url)  # AuthorizedHttp carries creds
        if isinstance(content, bytes):
            return content.decode("utf-8", "replace")
        return content or ""
    except Exception as exc:
        log.info("reporting download failed: %s", str(exc)[:160])
        return ""


def fetch_thumbnail_ctr(db, channel_id: int, *, created_after: Optional[str] = None) -> dict:
    """Full flow: creds → client → ensure job → list+download reports → aggregate. Returns
    ``{video_id: {impressions, ctr, since}}`` (empty when scope/creds missing, the job is
    freshly created with no reports yet, or any API error → caller uses the CTR proxy)."""
    try:
        import models
        from youtube import oauth as yt_oauth
        token = (db.query(models.OAuthToken)
                 .filter(models.OAuthToken.channel_id == int(channel_id)).first())
        if token is None or ANALYTICS_SCOPE not in (getattr(token, "scopes", "") or ""):
            return {}
        creds = yt_oauth.get_credentials(db, int(channel_id))
        client = _reporting_client(creds)
    except Exception as exc:
        log.info("reporting creds/scope unavailable for channel=%s: %s", channel_id, str(exc)[:120])
        return {}

    job_id = ensure_job(client)
    if not job_id:
        return {}
    try:
        reports = []
        for m in list_reports(client, job_id, created_after=created_after):
            url = m.get("downloadUrl")
            if not url:
                continue
            reports.append({
                "startTime": m.get("startTime"),
                "createTime": m.get("createTime"),
                "csv": download_csv(client, url),
            })
        return aggregate_thumbnail_ctr(reports)
    except Exception as exc:
        log.info("reporting fetch failed for channel=%s: %s", channel_id, str(exc)[:160])
        return {}


def ensure_job_for_channel(db, channel_id: int) -> Optional[str]:
    """Connect-time bootstrap: create the reporting job now so the 30-day backfill starts.
    Safe to call repeatedly (idempotent). Returns the job id or None."""
    try:
        import models
        from youtube import oauth as yt_oauth
        token = (db.query(models.OAuthToken)
                 .filter(models.OAuthToken.channel_id == int(channel_id)).first())
        if token is None or ANALYTICS_SCOPE not in (getattr(token, "scopes", "") or ""):
            return None
        creds = yt_oauth.get_credentials(db, int(channel_id))
        return ensure_job(_reporting_client(creds))
    except Exception as exc:
        log.info("reporting ensure_job_for_channel failed: %s", str(exc)[:120])
        return None
