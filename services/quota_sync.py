"""Sync the REAL Google-assigned YouTube Data API daily quota.

Why this exists: the system used to run on a fantasy — the quota gate
read ``KAIZER_YT_DAILY_QUOTA_CAP=1,000,000`` (a placeholder for a quota
grant that was never approved) while Google's actual assigned limit for
this project is the stock **10,000 units/day** (verified live via the
Service Usage API). Real traffic would have sailed past our gate
straight into Google's hard 403.

The YouTube Data API exposes no endpoint for its own limit, so we read
it from the **Service Usage API** using the project's existing service
account (``KAIZER_VERTEX_CREDENTIALS``):

    GET v1beta1/projects/{project}/services/youtube.googleapis.com
        /consumerQuotaMetrics

…and pick the "Queries" metric's per-day limit (unit ``1/d/…``). The
leader-elected cron (``services/cron_runner.py``) calls
:func:`run_quota_sync` hourly and persists the value to the
``system_settings`` KV table — so when Google eventually grants the
increase, the whole platform picks it up within an hour, no deploy.

Resolution order for the effective cap (:func:`resolved_daily_cap`):

    1. ``KAIZER_YT_DAILY_QUOTA_CAP`` env — explicit ops override only
       (leave UNSET in normal operation; the synced value should rule)
    2. ``system_settings['yt_quota_daily_cap_google']`` — synced truth
    3. 10,000 — Google's stock default (the SAFE fallback; the old 1M
       fallback is exactly the bug this module kills)
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("kaizer.quota_sync")

SETTING_CAP_KEY = "yt_quota_daily_cap_google"
SETTING_SYNCED_AT_KEY = "yt_quota_cap_synced_at"

#: Google's stock per-project default — the safe fallback when neither
#: an env override nor a synced value exists.
STOCK_DEFAULT_CAP = 10_000

#: In-process TTL cache so the per-reserve hot path doesn't query
#: system_settings on every call.
_CACHE_TTL_S = 60.0
_cache: dict = {"ts": 0.0, "cap": None}


# ─── Live fetch from Google ──────────────────────────────────────────


def fetch_google_daily_cap() -> Optional[int]:
    """Ask Google for the ACTUAL per-day YouTube quota limit.

    Returns the effective per-day "Queries" limit in units, or None on
    any failure (missing creds, IAM denied, API disabled, …). Never
    raises — the caller falls back to the cached/stock value.
    """
    cred_path = (
        os.environ.get("KAIZER_VERTEX_CREDENTIALS")
        or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        or ""
    )
    project = os.environ.get("KAIZER_GCP_PROJECT", "").strip()
    if not cred_path or not os.path.isfile(cred_path) or not project:
        log.warning(
            "quota_sync: missing service-account creds or KAIZER_GCP_PROJECT; "
            "cannot fetch the real cap",
        )
        return None
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        creds = service_account.Credentials.from_service_account_file(
            cred_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        su = build("serviceusage", "v1beta1", credentials=creds,
                   cache_discovery=False)
        parent = f"projects/{project}/services/youtube.googleapis.com"
        resp = su.services().consumerQuotaMetrics().list(
            parent=parent, view="BASIC",
        ).execute()

        best: Optional[int] = None
        for metric in resp.get("metrics", []):
            # The daily project quota is the "Queries" metric's 1/d
            # limit. Other metrics (Search Queries, Video Uploads, …)
            # are sub-quotas we don't gate on.
            disp = (metric.get("displayName") or "").strip().lower()
            if disp != "queries":
                continue
            for lim in metric.get("consumerQuotaLimits", []):
                unit = str(lim.get("unit", ""))
                if "/d/" not in unit and not unit.startswith("1/d"):
                    continue
                for bucket in lim.get("quotaBuckets", []):
                    # The default (project-wide) bucket carries no
                    # dimensions; region/user overrides do.
                    if bucket.get("dimensions"):
                        continue
                    eff = bucket.get("effectiveLimit",
                                     bucket.get("defaultLimit"))
                    try:
                        val = int(eff)
                    except (TypeError, ValueError):
                        continue
                    if val > 0:
                        best = val
        if best is None:
            log.warning(
                "quota_sync: Service Usage answered but no per-day "
                "'Queries' limit found for %s", project,
            )
        return best
    except Exception as exc:
        log.warning("quota_sync: fetch failed: %s", str(exc)[:300])
        return None


# ─── Persist + resolve ───────────────────────────────────────────────


def sync_to_settings() -> Optional[int]:
    """Fetch the real cap and persist it. Returns the cap, or None."""
    cap = fetch_google_daily_cap()
    if cap is None:
        return None
    from database import SessionLocal
    from system_settings import get_system_setting, set_system_setting

    db = SessionLocal()
    try:
        prev = get_system_setting(db, SETTING_CAP_KEY, "")
        set_system_setting(db, SETTING_CAP_KEY, str(cap))
        set_system_setting(
            db, SETTING_SYNCED_AT_KEY,
            datetime.now(timezone.utc).isoformat(),
        )
        db.commit()
        _cache["ts"] = 0.0  # bust the TTL cache immediately
        if str(cap) != (prev or ""):
            log.info(
                "quota_sync: Google YouTube daily cap synced: %s -> %d "
                "units/day", prev or "(none)", cap,
            )
            print(f"[quota_sync] Google YouTube daily quota = {cap} units/day"
                  + (f" (was {prev})" if prev else ""))
    except Exception:
        db.rollback()
        log.exception("quota_sync: persist failed")
        return None
    finally:
        db.close()
    return cap


def resolved_daily_cap() -> int:
    """The effective daily cap the gates must enforce. See module
    docstring for the resolution order. TTL-cached (60s) because the
    quota gate calls this on every reserve()."""
    # 1) Explicit env override (emergency lever — normally UNSET).
    raw = os.environ.get("KAIZER_YT_DAILY_QUOTA_CAP", "").strip()
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            pass

    # 2) Synced value (TTL-cached).
    now = time.monotonic()
    if _cache["cap"] is not None and (now - _cache["ts"]) < _CACHE_TTL_S:
        return int(_cache["cap"])
    cap = STOCK_DEFAULT_CAP
    try:
        from database import SessionLocal
        from system_settings import get_system_setting

        db = SessionLocal()
        try:
            stored = get_system_setting(db, SETTING_CAP_KEY, "")
        finally:
            db.close()
        if stored:
            v = int(stored)
            if v > 0:
                cap = v
    except Exception:
        # Settings table missing / DB down → stock default. NEVER the
        # old 1M fantasy.
        log.debug("quota_sync: resolved_daily_cap fell back to stock default",
                  exc_info=True)
    _cache["cap"] = cap
    _cache["ts"] = now
    return cap


def run_quota_sync() -> None:
    """Cron body (leader-elected, hourly)."""
    sync_to_settings()


__all__ = [
    "fetch_google_daily_cap",
    "sync_to_settings",
    "resolved_daily_cap",
    "run_quota_sync",
    "STOCK_DEFAULT_CAP",
    "SETTING_CAP_KEY",
]
