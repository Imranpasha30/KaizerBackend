"""Per-credential Google API request counts from Cloud Monitoring (v3).

Reads serviceruntime.googleapis.com/api/request_count on the monitored
resource `consumed_api`, grouped by the resource label `credential_id`.
For API-key traffic that label is "apikey:<uid>" where <uid> is exactly the
Key.uid we store in GoogleMintedKey.key_uid — that's the join key that maps
requests back to a user (services/google_key_minter.py mints the keys).

10-minute TTL cache (operator asked 5-15 min) so the admin panel doesn't
hammer Monitoring. Never raises: on failure it serves the last cached data
with an `error` field, or an empty result if the cache is cold.

HONEST LIMIT: there is no per-CREDENTIAL token metric for
generativelanguage.googleapis.com — token usage exists only as per-PROJECT
quota (`consumer_quota`, which has no credential label). So v1 attributes
request counts per user, not tokens. The response reserves an optional
`project_tokens` field so a project-wide token line can slot in later
without a contract change.

Project + credentials are resolved by google_key_minter.resolve_project() /
its creds helper so metrics and minted keys ALWAYS target the same project.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

log = logging.getLogger("kaizer.google_usage")

#: service host -> the metric key we report under.
SERVICE_TO_METRIC_KEY = {
    "youtube.googleapis.com":            "youtube_requests",
    "generativelanguage.googleapis.com": "gemini_requests",
}

CACHE_TTL_S = 600.0                          # 10 min
_cache: dict[int, dict] = {}                 # days -> {"ts": monotonic, "data": {...}}
_cache_lock = threading.Lock()


def _rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _monitoring_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from services import google_key_minter as gkm
    creds = service_account.Credentials.from_service_account_file(
        gkm._creds_path(), scopes=["https://www.googleapis.com/auth/monitoring.read"],
    )
    return build("monitoring", "v3", credentials=creds, cache_discovery=False)


def _query_service_requests(mon, project: str, service: str,
                            start_iso: str, end_iso: str,
                            align_s: int) -> dict[str, int]:
    """One projects.timeSeries.list for one service. Returns
    {credential_id: total_request_count}. Follows nextPageToken."""
    filt = (
        'metric.type = "serviceruntime.googleapis.com/api/request_count" '
        'AND resource.type = "consumed_api" '
        f'AND resource.label."service" = "{service}"'
    )
    out: dict[str, int] = {}
    page_token = None
    while True:
        resp = mon.projects().timeSeries().list(
            name=f"projects/{project}",
            filter=filt,
            interval_startTime=start_iso,
            interval_endTime=end_iso,
            aggregation_alignmentPeriod=f"{align_s}s",
            aggregation_perSeriesAligner="ALIGN_SUM",
            aggregation_crossSeriesReducer="REDUCE_SUM",
            aggregation_groupByFields=['resource.label."credential_id"'],
            pageToken=page_token,
        ).execute()
        for series in resp.get("timeSeries", []):
            cred = (series.get("resource", {}).get("labels", {})
                    .get("credential_id") or "(no credential)")
            total = 0
            for pt in series.get("points", []):
                v = pt.get("value", {})
                try:
                    total += int(v.get("int64Value", v.get("doubleValue", 0)) or 0)
                except (TypeError, ValueError):
                    continue
            out[cred] = out.get(cred, 0) + total
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return out


def fetch_usage(days: int, force: bool = False) -> dict:
    """Return per-credential request counts over the last `days`, cached.

    {"by_credential": {"apikey:<uid>": {"youtube_requests": int,
                                        "gemini_requests": int}, ...},
     "start": iso, "end": iso, "fetched_at": iso, "cached": bool,
     "error": Optional[str]}
    NEVER raises."""
    days = max(1, min(90, int(days)))
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(days)
        if hit and not force and (now - hit["ts"]) < CACHE_TTL_S:
            data = dict(hit["data"]); data["cached"] = True
            return data

    from services import google_key_minter as gkm
    project = gkm.resolve_project()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    result = {
        "by_credential": {},
        "start": _rfc3339(start), "end": _rfc3339(end),
        "fetched_at": end.isoformat(), "cached": False, "error": None,
    }
    if not gkm.is_configured():
        result["error"] = "Key minting / monitoring is not configured on the server."
        return result
    try:
        mon = _monitoring_service()
        align_s = days * 86400
        by_cred: dict[str, dict] = {}
        for service, metric_key in SERVICE_TO_METRIC_KEY.items():
            counts = _query_service_requests(
                mon, project, service, result["start"], result["end"], align_s)
            for cred, n in counts.items():
                by_cred.setdefault(cred, {})[metric_key] = n
        # normalize: every credential has both keys
        for cred, d in by_cred.items():
            d.setdefault("youtube_requests", 0)
            d.setdefault("gemini_requests", 0)
        result["by_credential"] = by_cred
        with _cache_lock:
            _cache[days] = {"ts": now, "data": dict(result)}
    except Exception as exc:
        msg = str(exc)[:400]
        log.warning("fetch_usage failed: %s", msg)
        with _cache_lock:
            hit = _cache.get(days)
        if hit:
            data = dict(hit["data"]); data["cached"] = True; data["error"] = msg
            return data
        result["error"] = msg
    return result


__all__ = ["fetch_usage", "SERVICE_TO_METRIC_KEY", "CACHE_TTL_S"]
