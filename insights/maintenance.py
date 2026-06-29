"""Insights background maintenance — daily CTR refresh + snapshot pruning.

These are the multi-tenant "set-and-forget" jobs. They're plain functions (the scheduler
wires them up separately) so they're testable and can also be run on demand.

  refresh_thumbnail_ctr  — for each connected channel, download the latest Reporting-API
                           reach report and write CTR into that channel's MOST RECENT
                           snapshot's video rows (no full re-pull, ~free quota).
  prune_snapshots        — keep the newest N snapshots per (user, channel); delete older
                           ones + their video metrics / runs / reports (bound DB growth).

Both iterate ALL users' channels, but each fetch uses that channel's own owner credentials
(reporting_api re-checks scope + ownership), so isolation holds.
"""
from __future__ import annotations

import logging
from typing import Optional

from insights import models as im
from insights import reporting_api as RA

log = logging.getLogger("kaizer.insights.maintenance")

ANALYTICS_SCOPE = "https://www.googleapis.com/auth/yt-analytics.readonly"


def refresh_thumbnail_ctr_for_channel(db, owner_channel_id: int) -> int:
    """Download this channel's latest reach report and write CTR into its most-recent
    snapshot's VideoMetric rows. Returns the number of videos updated (0 if no report yet)."""
    snap = (db.query(im.ChannelSnapshot)
            .filter(im.ChannelSnapshot.channel_id == owner_channel_id)
            .order_by(im.ChannelSnapshot.created_at.desc()).first())
    if not snap:
        return 0
    ctr_map = RA.fetch_thumbnail_ctr(db, owner_channel_id) or {}
    if not ctr_map:
        return 0
    updated = 0
    rows = (db.query(im.VideoMetric)
            .filter(im.VideoMetric.snapshot_id == snap.id,
                    im.VideoMetric.video_id.in_(list(ctr_map.keys()))).all())
    for vm in rows:
        c = ctr_map.get(vm.video_id)
        if not c:
            continue
        vm.impressions = c.get("impressions")
        vm.impressions_ctr = c.get("ctr")
        updated += 1
    if updated:
        # clear the "reporting_populating" note once real CTR has landed
        if snap.error in ("reporting_populating", "reporting_unavailable"):
            snap.error = None
        db.commit()
    return updated


def refresh_thumbnail_ctr(db, *, limit: Optional[int] = None) -> dict:
    """Daily poll: refresh thumbnail CTR for every connected channel that granted analytics.
    Cheap (bulk report download, separate quota pool). Best-effort per channel."""
    import models
    toks = (db.query(models.OAuthToken)
            .filter(models.OAuthToken.refresh_token_enc != "").all())
    seen, channels, updated_total, touched = set(), 0, 0, 0
    for t in toks:
        if not t.channel_id or t.channel_id in seen:
            continue
        seen.add(t.channel_id)
        if ANALYTICS_SCOPE not in (getattr(t, "scopes", "") or ""):
            continue
        channels += 1
        if limit and channels > limit:
            break
        try:
            n = refresh_thumbnail_ctr_for_channel(db, t.channel_id)
            updated_total += n
            if n:
                touched += 1
        except Exception as exc:
            log.info("insights CTR refresh failed for channel %s: %s", t.channel_id, str(exc)[:160])
    log.info("insights CTR refresh: %d channels, %d videos updated across %d channels",
             channels, updated_total, touched)
    return {"channels": channels, "videos_updated": updated_total, "channels_touched": touched}


def prune_snapshots(db, *, keep: int = 2) -> dict:
    """Keep the newest `keep` snapshots per (user, channel); delete older snapshots + their
    video metrics / analysis runs / report versions. Bounds DB growth (VideoMetric is the bulk)."""
    keep = max(1, int(keep))
    rows = (db.query(im.ChannelSnapshot.id, im.ChannelSnapshot.user_id,
                     im.ChannelSnapshot.google_channel_id)
            .order_by(im.ChannelSnapshot.created_at.desc()).all())
    by_key: dict = {}
    for sid, uid, gcid in rows:
        by_key.setdefault((uid, gcid), []).append(sid)
    old_ids = [sid for ids in by_key.values() for sid in ids[keep:]]
    if not old_ids:
        return {"pruned_snapshots": 0}
    # delete children first (robust on SQLite where FK cascade may be off), then snapshots.
    run_ids = [r[0] for r in db.query(im.AnalysisRun.id)
               .filter(im.AnalysisRun.snapshot_id.in_(old_ids)).all()]
    if run_ids:
        db.query(im.ReportVersion).filter(
            im.ReportVersion.analysis_run_id.in_(run_ids)).delete(synchronize_session=False)
    db.query(im.AnalysisRun).filter(
        im.AnalysisRun.snapshot_id.in_(old_ids)).delete(synchronize_session=False)
    db.query(im.VideoMetric).filter(
        im.VideoMetric.snapshot_id.in_(old_ids)).delete(synchronize_session=False)
    db.query(im.ChannelSnapshot).filter(
        im.ChannelSnapshot.id.in_(old_ids)).delete(synchronize_session=False)
    db.commit()
    log.info("insights prune: removed %d old snapshots (+children)", len(old_ids))
    return {"pruned_snapshots": len(old_ids)}


def run_daily_maintenance(db, *, keep: int = 2) -> dict:
    """One daily pass: refresh CTR for all channels, then prune old snapshots."""
    ctr = refresh_thumbnail_ctr(db)
    pruned = prune_snapshots(db, keep=keep)
    return {"ctr": ctr, "prune": pruned}
