"""Insights / Trend Finder — API endpoints (isolated module router).

Ties the pipeline together: pick a connected channel → analyze (ingest → diagnose → report)
→ fetch / export the report. Every endpoint is ownership-scoped (a user can only analyze
their OWN connected channels; ingestion re-checks ownership too).

Kaizer X is a production & analytics tool — these endpoints diagnose editorial/content
performance, never operate channels for bulk uploads/monetization.
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

import auth
import models
from database import get_db, SessionLocal
from insights import ingest, analyze, report, reporting_api
from insights import models as im

log = logging.getLogger("kaizer.insights.router")
router = APIRouter(prefix="/api/insights", tags=["insights"])

ANALYTICS_SCOPE = "https://www.googleapis.com/auth/yt-analytics.readonly"
_PROVIDERS = {"gemini", "deterministic"}

# Per-user daily cap on FRESH channel pulls (each burns shared YouTube Data API quota).
# ADMINS ARE EXEMPT (unlimited). Regular users are capped so the project quota can't be
# exhausted. Tune via env; the 24h snapshot cache already prevents re-pulling a channel.
INSIGHTS_DAILY_LIMIT = max(1, int(os.environ.get("KAIZER_INSIGHTS_DAILY_LIMIT", "5") or 5))
# Platform-wide ceiling on fresh pulls/24h (protects the shared 10k/day Data API quota even
# if many users each spend their allowance). Admins bypass everything.
INSIGHTS_GLOBAL_LIMIT = max(INSIGHTS_DAILY_LIMIT,
                            int(os.environ.get("KAIZER_INSIGHTS_GLOBAL_LIMIT", "200") or 200))


def _enforce_quota(db: Session, user: models.User, gcid: str, force: bool) -> bool:
    """Return the effective `force` after quota policy. Admins: unlimited (force honored).
    Everyone else: a channel already pulled in the last 24h serves the cache (force ignored);
    brand-new pulls are capped per-user (INSIGHTS_DAILY_LIMIT) AND platform-wide
    (INSIGHTS_GLOBAL_LIMIT) → 429 over either."""
    if getattr(user, "is_admin", False):
        return force                                   # admin: no quota, full control
    day_ago = datetime.now(timezone.utc) - timedelta(hours=24)
    recent_for_chan = (db.query(im.ChannelSnapshot.id)
                       .filter(im.ChannelSnapshot.user_id == user.id,
                               im.ChannelSnapshot.google_channel_id == gcid,
                               im.ChannelSnapshot.created_at >= day_ago).first())
    if recent_for_chan:
        return False                                   # serve today's cached snapshot — never re-burn
    # this would be a NEW pull → check per-user then platform-wide caps
    fresh_today = (db.query(im.ChannelSnapshot.id)
                   .filter(im.ChannelSnapshot.user_id == user.id,
                           im.ChannelSnapshot.created_at >= day_ago).count())
    if fresh_today >= INSIGHTS_DAILY_LIMIT:
        raise HTTPException(
            429, f"You've reached today's limit of {INSIGHTS_DAILY_LIMIT} channel analyses. Your "
                 f"existing reports stay available; new analyses reset within 24 hours.")
    global_today = (db.query(im.ChannelSnapshot.id)
                    .filter(im.ChannelSnapshot.created_at >= day_ago).count())
    if global_today >= INSIGHTS_GLOBAL_LIMIT:
        raise HTTPException(
            429, "Insights is at today's analysis capacity across the platform — please try again "
                 "in a little while. Your existing reports are still available.")
    return force


@router.get("/channels")
def list_insight_channels(db: Session = Depends(get_db),
                          user: models.User = Depends(auth.current_user)):
    """The user's connected YouTube channels (for the analyzer's channel picker)."""
    rows = (db.query(models.OAuthToken)
            .join(models.Channel, models.Channel.id == models.OAuthToken.channel_id)
            .filter(models.Channel.user_id == user.id,
                    models.OAuthToken.refresh_token_enc != "")
            .all())
    seen, out = set(), []
    for t in rows:
        g = t.google_channel_id
        if not g or g in seen:
            continue
        seen.add(g)
        out.append({
            "google_channel_id": g,
            "title": t.google_channel_title or g,
            "has_analytics": ANALYTICS_SCOPE in (t.scopes or ""),
        })
    return out


class AnalyzeIn(BaseModel):
    google_channel_id: str
    force: bool = False                 # bypass the 24h snapshot cache
    provider: str = "gemini"            # gemini | deterministic


# ── Background analysis (a full pull of a big channel would time out a sync HTTP request) ──
# In-memory run state, mirroring the V4 render pattern. run_id encodes the owner for polling
# auth. The worker uses its OWN DB session (the request session is gone by then).
_RUNS: dict = {}
_RUNS_LOCK = threading.Lock()
_RUN_SEM = threading.BoundedSemaphore(max(1, int(os.environ.get("KAIZER_INSIGHTS_CONCURRENCY", "2") or 2)))


def _set_run(run_id: str, **kv) -> None:
    with _RUNS_LOCK:
        cur = _RUNS.get(run_id) or {}
        cur.update(kv)
        _RUNS[run_id] = cur


def _get_run(run_id: str) -> dict:
    with _RUNS_LOCK:
        return dict(_RUNS.get(run_id) or {})


def _prune_runs() -> None:
    cutoff = datetime.now(timezone.utc).timestamp() - 3600
    with _RUNS_LOCK:
        for rid in [k for k, v in _RUNS.items() if (v.get("ts") or 0) < cutoff]:
            _RUNS.pop(rid, None)


def _analyze_worker(run_id: str, user_id: int, gcid: str, provider: str, eff_force: bool) -> None:
    if not _RUN_SEM.acquire(timeout=600):
        _set_run(run_id, status="failed", error="Server busy — please try again shortly.")
        return
    db = SessionLocal()
    try:
        _set_run(run_id, status="running", msg="Reading your channel history…", progress=25)
        snap = ingest.run_ingestion(db, user_id, gcid, force=eff_force)
        _set_run(run_id, status="running", msg="Diagnosing performance…", progress=65)
        run = analyze.analyze_snapshot(db, snap.id)
        if run.status != "ok":
            _set_run(run_id, status="failed", error=f"Analysis failed: {(run.error or '')[:240]}")
            return
        _set_run(run_id, status="running", msg="Writing your report…", progress=88)
        rv = report.generate_report(db, run.id, provider=provider)
        _set_run(run_id, status="done", progress=100, msg="Complete", result={
            "snapshot_id": snap.id, "run_id": run.id, "report_id": rv.id, "version": rv.version,
            "mode": run.maturity, "access_mode": snap.access_mode, "provider": rv.provider,
            "videos_analyzed": snap.videos_ingested, "channel_title": snap.channel_title,
            "exec_summary": rv.exec_summary, "report_md": rv.report_md, "report_json": rv.report_json,
        })
    except RuntimeError as exc:            # ownership / not-connected
        _set_run(run_id, status="failed", error=str(exc)[:300])
    except Exception as exc:
        log.exception("insights async analysis failed for %s", gcid)
        _set_run(run_id, status="failed", error=f"Analysis failed: {str(exc)[:280]}")
    finally:
        db.close()
        _RUN_SEM.release()


@router.post("/analyze")
def analyze_channel(body: AnalyzeIn, db: Session = Depends(get_db),
                    user: models.User = Depends(auth.current_user)):
    """Queue the full pipeline for one of the user's channels in the BACKGROUND and return a
    run_id to poll (a fresh pull of a 10k-video channel takes minutes — too long to block on).
    Quota is enforced HERE (so 429 is synchronous); ownership is enforced in ingestion."""
    gcid = (body.google_channel_id or "").strip()
    if not gcid:
        raise HTTPException(400, "google_channel_id is required")
    provider = body.provider if body.provider in _PROVIDERS else "gemini"
    # Quota policy: admins unlimited; regular users capped (protects the shared Data API quota).
    eff_force = _enforce_quota(db, user, gcid, body.force)
    _prune_runs()
    run_id = f"{user.id}:{gcid}:{int(datetime.now(timezone.utc).timestamp() * 1000)}"
    _set_run(run_id, status="queued", msg="Queued…", progress=5, error=None, result=None,
             ts=datetime.now(timezone.utc).timestamp())
    threading.Thread(target=_analyze_worker, args=(run_id, user.id, gcid, provider, eff_force),
                     daemon=True, name=f"insights-{run_id[:40]}").start()
    return {"run_id": run_id, "status": "queued"}


@router.get("/analyze-status/{run_id}")
def analyze_status(run_id: str, user: models.User = Depends(auth.current_user)):
    """Poll a background analysis. Returns {status, msg, progress, error?, result?}. When
    status=='done', `result` is the full report payload."""
    if not (run_id.startswith(f"{user.id}:") or getattr(user, "is_admin", False)):
        raise HTTPException(403, "Not your analysis run")
    st = _get_run(run_id)
    if not st:
        raise HTTPException(404, "Unknown or expired analysis run — start a new analysis.")
    st.pop("ts", None)
    return st


@router.get("/latest/{google_channel_id}")
def latest_report(google_channel_id: str, db: Session = Depends(get_db),
                  user: models.User = Depends(auth.current_user)):
    """Most recent report for a channel (no re-run). 404 if none yet."""
    rv = (db.query(im.ReportVersion)
          .join(im.AnalysisRun, im.AnalysisRun.id == im.ReportVersion.analysis_run_id)
          .join(im.ChannelSnapshot, im.ChannelSnapshot.id == im.AnalysisRun.snapshot_id)
          .filter(im.ReportVersion.user_id == user.id,
                  im.ChannelSnapshot.google_channel_id == google_channel_id)
          .order_by(im.ReportVersion.created_at.desc()).first())
    if not rv:
        raise HTTPException(404, "No analysis yet for this channel — run one first.")
    run = db.query(im.AnalysisRun).filter(im.AnalysisRun.id == rv.analysis_run_id).first()
    return {
        "report_id": rv.id, "version": rv.version, "provider": rv.provider,
        "mode": run.maturity if run else None,
        "created_at": rv.created_at, "exec_summary": rv.exec_summary,
        "report_md": rv.report_md, "report_json": rv.report_json,
    }


@router.get("/export/{report_id}")
def export_report(report_id: int, db: Session = Depends(get_db),
                  user: models.User = Depends(auth.current_user)):
    """Download a report as a markdown file (ownership-checked)."""
    rv = (db.query(im.ReportVersion)
          .filter(im.ReportVersion.id == report_id, im.ReportVersion.user_id == user.id).first())
    if not rv:
        raise HTTPException(404, "Report not found")
    return Response(
        content=rv.report_md or "",
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="kaizer-insights-{report_id}.md"'},
    )


@router.post("/bootstrap/{google_channel_id}")
def bootstrap_ctr(google_channel_id: str, db: Session = Depends(get_db),
                  user: models.User = Depends(auth.current_user)):
    """Create the Reporting-API job NOW so the 30-day thumbnail-CTR backfill clock starts
    (call at channel-connect time; idempotent). Returns whether CTR collection is armed."""
    owner = ingest._owner_channel(db, user.id, google_channel_id)
    if not owner:
        raise HTTPException(403, "You don't own a profile linked to this channel.")
    job_id = reporting_api.ensure_job_for_channel(db, owner.id)
    return {
        "armed": bool(job_id),
        "note": ("Thumbnail-CTR collection started. First data in ~48h; YouTube backfills only "
                 "~30 days, then grows daily. Older videos won't have CTR." if job_id else
                 "Could not start CTR collection (analytics access may not be granted)."),
    }
