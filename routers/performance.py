"""Performance router — feedback loop read API."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session

import auth
from database import get_db
import models
from analytics import calibrator, channel_catalog, poller


router = APIRouter(prefix="/api/performance", tags=["performance"])


@router.get("/leaderboard")
def leaderboard(
    limit: int = Query(20, ge=1, le=100),
    channel_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    return calibrator.channel_leaderboard(db, limit=limit, channel_id=channel_id)


@router.get("/calibration")
def calibration(
    channel_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    return calibrator.calibration_report(db, channel_id=channel_id)


@router.get("/channels")
def channels_summary(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Per-YouTube-channel summary cards for the Performance header.

    Returns one card per real YT channel the requesting user has
    connected (every OAuthToken with a refresh token), with
    aggregates drawn from the ``channel_videos`` cache.  Channels
    that haven't been synced yet still appear (with
    ``needs_sync: true`` and zeroed stats) so the UI can prompt the
    user to hit "Sync from YouTube".
    """
    return calibrator.channel_summary(db, user_id=user.id)


@router.get("/seo-learning")
def seo_learning(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Per-channel SEO feedback-loop learning — powers the channel-wise
    Insights tab. For each connected channel: what the system has learnt from
    past performance (its 'winning keywords'), which signal it's using (CTR
    once that channel is re-approved for analytics, else views), how much data
    it has, and whether it's actively shaping SEO yet. Read-only; safe."""
    from seo.performance_profile import build_channel_profile
    from analytics.ctr import token_has_analytics_scope

    out = []
    all_chans = db.query(models.Channel).filter(models.Channel.user_id == user.id).all()
    # Only OUR connected publish channels learn from their OWN results here.
    # Style-reference / competitor channels (no usable OAuth) are analysed
    # separately via their PUBLIC performance (learning/corpus.py) — they do not
    # belong in "what WE learnt", so keep them out of this tab.
    chans = [c for c in all_chans
             if c.oauth_token is not None and bool(c.oauth_token.refresh_token_enc)]
    for ch in chans:
        try:
            prof = build_channel_profile(db, ch.id)
        except Exception:
            prof = {"ready": False, "n_videos": 0, "winning_keywords": [], "signal": "views_per_hour"}
        tok = (db.query(models.OAuthToken)
                 .filter(models.OAuthToken.channel_id == ch.id).first())
        # REAL learned state (learning/seo_learning.py): the measured
        # policy the generator actually consumes + how much real CTR the
        # channel has — so the UI shows genuine learning, never a mock.
        policy = None
        last_learned_at = None
        ctr_rows = 0
        try:
            from learning.seo_learning import latest_policy
            policy = latest_policy(db, ch.id)
            snap = (db.query(models.SeoLearningSnapshot)
                    .filter(models.SeoLearningSnapshot.channel_id == ch.id)
                    .order_by(models.SeoLearningSnapshot.computed_at.desc())
                    .first())
            if snap is not None:
                last_learned_at = snap.computed_at
            ctr_rows = (db.query(models.TrainingSample)
                        .filter(models.TrainingSample.channel_id == ch.id)
                        .filter(models.TrainingSample.ctr.isnot(None))
                        .count())
        except Exception:
            pass
        out.append({
            "channel_id": ch.id,
            "channel_name": ch.name,
            "ready": bool(prof.get("ready")),
            "signal": prof.get("signal", "views_per_hour"),
            "ctr_unlocked": bool(tok and token_has_analytics_scope(tok)),
            # ONE truth for "videos learnt from": the learned policy's
            # measured base when it exists (matches the curve panel), the
            # raw catalogue count only before any learning has run.
            "videos_sampled": ((policy or {}).get("based_on")
                               or prof.get("n_videos", 0)),
            "ctr_rows": ctr_rows,
            # ONE keyword system: the learned, brand-filtered, performance-
            # weighted terms when a policy exists; the legacy profile list
            # only as fallback (it was brand echo — "kaizer, news" on every
            # card).
            "winning_keywords": ((policy or {}).get("top_keywords")
                                 or prof.get("winning_keywords", [])),
            "top_titles": prof.get("top_titles", []),
            "top_views_per_hour": prof.get("top_views_per_hour"),
            "policy": policy,
            "use_competitor_intel": bool(
                getattr(ch, "use_competitor_intel", False)),
            "last_learned_at": (last_learned_at.isoformat()
                                if last_learned_at else None),
            "status": ("steering SEO" if policy
                       else ("learning" if prof.get("ready")
                             else "collecting data")),
        })
    # Learning channels first, then most data.
    out.sort(key=lambda c: (not c["ready"], -c["videos_sampled"]))
    return {
        "channels": out,
        "seo_engine": (getattr(user, "seo_engine", "") or "gemini"),
        "note": "Each channel learns from its own results. CTR is used once you "
                "re-approve that channel for analytics; views are used until then.",
    }


@router.post("/seo-engine")
def set_seo_engine(
    engine: str = Query(...),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """User-level SEO writer choice: gemini (default) or claude. Same
    prompts/verifier/learning either way — only the writer swaps; Claude
    failures auto-fall back to Gemini at generation time."""
    e = (engine or "").strip().lower()
    if e not in ("gemini", "claude"):
        raise HTTPException(400, "engine must be 'gemini' or 'claude'")
    row = db.query(models.User).filter(models.User.id == user.id).first()
    row.seo_engine = e
    db.commit()
    return {"ok": True, "seo_engine": e}


@router.get("/seo-learning/curves")
def seo_learning_curves(
    channel_id: int = Query(...),
    days: int = Query(30, ge=7, le=90),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Time-series behind the SEO-Learning graphs: per-day performance of
    this channel's published videos (views/hour + real CTR) plus the
    snapshot timeline (how the learned policy evolved). All values are
    MEASURED — an empty series means the channel genuinely has no data
    for the window, and the UI says so instead of faking a curve."""
    from datetime import datetime, timedelta, timezone

    ch = (db.query(models.Channel)
          .filter(models.Channel.id == channel_id,
                  models.Channel.user_id == user.id).first())
    if ch is None:
        raise HTTPException(404, "channel not found")

    since = datetime.now(timezone.utc) - timedelta(days=int(days))
    # Merged REAL data: the channel's full YouTube catalogue (every upload,
    # real view counts/ages) + Kaizer TrainingSamples (real CTR/keywords).
    from learning.seo_learning import merged_rows
    rows = sorted(merged_rows(db, channel_id, since),
                  key=lambda r: r.first_seen_at or since)
    daily: dict = {}
    for r in rows:
        d = (r.first_seen_at or since).date().isoformat()
        b = daily.setdefault(d, {"date": d, "published": 0, "vph": [],
                                 "ctr": [], "impressions": 0})
        b["published"] += 1
        b["vph"].append(max(0.0, float(r.views_per_hour or 0.0)))
        if r.ctr is not None:
            b["ctr"].append(float(r.ctr))
        if getattr(r, "impressions", None):
            b["impressions"] += int(r.impressions)
    series = []
    for d in sorted(daily):
        b = daily[d]
        series.append({
            "date": d,
            "published": b["published"],
            "avg_vph": round(sum(b["vph"]) / len(b["vph"]), 2) if b["vph"] else 0.0,
            "avg_ctr": (round(sum(b["ctr"]) / len(b["ctr"]), 3)
                        if b["ctr"] else None),
            "impressions": b["impressions"],
        })

    snaps = (db.query(models.SeoLearningSnapshot)
             .filter(models.SeoLearningSnapshot.channel_id == channel_id)
             .filter(models.SeoLearningSnapshot.window_days == int(days)
                     if int(days) in (7, 30, 90)
                     else models.SeoLearningSnapshot.window_days == 30)
             .order_by(models.SeoLearningSnapshot.computed_at.asc())
             .all())
    learning_timeline = [{
        "computed_at": s.computed_at.isoformat() if s.computed_at else None,
        "samples": s.samples,
        "has_policy": bool((s.payload or {}).get("policy")),
        "ctr_coverage": (s.payload or {}).get("ctr_coverage", 0),
    } for s in snaps[-60:]]

    latest = snaps[-1].payload if snaps else None

    # Honesty instruments: does the score predict reality, and is there
    # measured uplift? Both ride the curves response for the UI.
    audit = uplift = None
    try:
        from learning.seo_learning import score_reality_audit, uplift_report
        audit = score_reality_audit(db, channel_id)
        uplift = uplift_report(db, channel_id)
    except Exception:
        pass
    return {
        "channel_id": channel_id,
        "days": int(days),
        "series": series,
        "learning_timeline": learning_timeline,
        "latest": latest,
        "score_audit": audit,
        "uplift": uplift,
    }


@router.get("/seo-learning/best-times")
def seo_learning_best_times(
    channel_id: int = Query(...),
    days: int = Query(3650),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Best-time-to-post report: the channel's REAL upload-hour + weekday
    performance (avg views/hour + real CTR + sample counts, IST) plus a
    recommendation. Every number is measured — honest empty-state when the
    history is too thin to trust a time. `days` picks the learning window
    (30 / 90 / 3650=all-time; all-time is the default since upload-time
    patterns are most stable over the full catalogue)."""
    if days not in (30, 90, 3650):
        days = 3650
    ch = (db.query(models.Channel)
          .filter(models.Channel.id == channel_id,
                  models.Channel.user_id == user.id).first())
    if ch is None:
        raise HTTPException(404, "channel not found")
    from learning.seo_learning import best_times_report
    rep = best_times_report(db, channel_id, days)
    rep["channel_name"] = ch.name
    return rep


@router.get("/seo-learning/weekly-uplift")
def seo_learning_weekly_uplift(
    channel_id: int = Query(...),
    weeks: int = Query(8, ge=2, le=26),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Week-by-week measured rollup (published count + avg views/hour + real
    CTR + week-over-week %) — the multi-week view behind the single uplift
    chip. Honest empty-state until >=2 measured weeks exist."""
    ch = (db.query(models.Channel)
          .filter(models.Channel.id == channel_id,
                  models.Channel.user_id == user.id).first())
    if ch is None:
        raise HTTPException(404, "channel not found")
    from learning.seo_learning import weekly_uplift
    rep = weekly_uplift(db, channel_id, weeks)
    rep["channel_name"] = ch.name
    return rep


# ── Competitor intelligence (Insights → Competitors) ───────────────

@router.get("/competitors")
def competitors_list(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Tracked rivals + their learned formula (public data only)."""
    from learning.competitor_intel import latest_competitor_payload
    comps = (db.query(models.CompetitorChannel)
             .filter(models.CompetitorChannel.user_id == user.id).all())
    out = []
    for c in comps:
        payload = latest_competitor_payload(db, c.id)
        out.append({
            "id": c.id, "name": c.name, "handle": c.handle,
            "youtube_channel_id": c.youtube_channel_id,
            "active": bool(c.active),
            "samples": (payload or {}).get("samples", 0),
            "policy": (payload or {}).get("policy"),
            "top_topics": ((payload or {}).get("top_topics") or [])[:5],
            "top_tags": ((payload or {}).get("top_tags") or [])[:8],
        })
    return {"competitors": out,
            "note": "Public YouTube data (views/titles/tags). Rival CTR is "
                    "private and never shown or faked."}


@router.post("/competitors/{comp_id}/learn")
def competitor_learn(
    comp_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Deep-sync the rival's public catalogue (with tags) + learn."""
    from learning.competitor_intel import learn_competitor, sync_competitor
    comp = (db.query(models.CompetitorChannel)
            .filter(models.CompetitorChannel.id == comp_id,
                    models.CompetitorChannel.user_id == user.id).first())
    if comp is None:
        raise HTTPException(404, "competitor not found")
    synced = sync_competitor(db, comp)
    payload = learn_competitor(db, comp)
    return {"ok": True, "synced": synced,
            "samples": payload.get("samples", 0),
            "policy": payload.get("policy")}


@router.post("/seo-learning/competitor-toggle")
def competitor_toggle(
    channel_id: int = Query(...),
    enabled: bool = Query(...),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Per-channel opt-in: feed competitor intelligence into this
    channel's SEO generation."""
    ch = (db.query(models.Channel)
          .filter(models.Channel.id == channel_id,
                  models.Channel.user_id == user.id).first())
    if ch is None:
        raise HTTPException(404, "channel not found")
    ch.use_competitor_intel = bool(enabled)
    db.commit()
    return {"ok": True, "channel_id": channel_id,
            "use_competitor_intel": bool(enabled)}


# Server-side relearn state — survives the browser navigating away and
# back (the UI polls relearn-status; a run in flight shows as running no
# matter how often the tab remounts).
_RELEARN_LOCK = __import__("threading").Lock()
_RELEARN: dict = {}   # channel_id -> {state, msg, started_at, result}


def _relearn_worker(channel_ids: list[int], user_id: int) -> None:
    """Deep-learn channels in a daemon thread with its OWN session:
    1) deep-sync the channel's FULL catalogue (max 2000 uploads — the
       default 200-video sync cap hid most of a 1000+-video channel's
       real history), 2) poll fresh stats, 3) ingest real thumbnail CTR,
    4) recompute + persist the policy snapshots."""
    from database import SessionLocal
    from learning.seo_learning import learn_channel
    from analytics.channel_catalog import sync_channel_videos
    db = SessionLocal()
    try:
        for cid in channel_ids:
            def _set(msg, state="running", result=None):
                with _RELEARN_LOCK:
                    _RELEARN[cid] = {"state": state, "msg": msg,
                                     "result": result}
            try:
                _set("syncing full channel history from YouTube…")
                tok = (db.query(models.OAuthToken)
                       .filter(models.OAuthToken.channel_id == cid).first())
                gcid = (getattr(tok, "google_channel_id", "") or "") if tok else ""
                if gcid:
                    try:
                        sync_channel_videos(db, user_id, gcid, max_videos=2000)
                    except Exception as exc:
                        print(f"[seo-learning] deep sync soft-fail ch={cid}: "
                              f"{str(exc)[:120]}", flush=True)
                _set("polling latest video stats…")
                try:
                    poller.poll_once(channel_id=cid)
                except Exception:
                    pass
                _set("ingesting real CTR + computing what works…")
                res = learn_channel(db, cid)
                _set("done", state="done", result=res)
            except Exception as exc:
                _set(f"failed: {str(exc)[:160]}", state="error")
    finally:
        try:
            db.close()
        except Exception:
            pass


@router.post("/seo-learning/relearn")
def seo_learning_relearn(
    channel_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Learn NOW (background): deep-sync the channel's full YouTube
    catalogue, poll fresh stats, ingest REAL thumbnail CTR, recompute the
    policy snapshots. Returns immediately; poll relearn-status."""
    import threading

    if channel_id is not None:
        chans = (db.query(models.Channel)
                 .filter(models.Channel.id == channel_id,
                         models.Channel.user_id == user.id).all())
    else:
        chans = (db.query(models.Channel)
                 .filter(models.Channel.user_id == user.id).all())
        chans = [c for c in chans if c.oauth_token is not None]
    if not chans:
        raise HTTPException(404, "no matching channels")

    ids = []
    with _RELEARN_LOCK:
        for ch in chans:
            cur = _RELEARN.get(ch.id) or {}
            if cur.get("state") == "running":
                continue                      # already learning — don't stack
            _RELEARN[ch.id] = {"state": "running", "msg": "queued…"}
            ids.append(ch.id)
    if ids:
        threading.Thread(target=_relearn_worker, args=(ids, user.id),
                         daemon=True, name="seo-relearn").start()
    return {"ok": True, "started": ids,
            "already_running": [c.id for c in chans if c.id not in ids]}


@router.get("/seo-learning/relearn-status")
def seo_learning_relearn_status(
    channel_id: int = Query(...),
    user: models.User = Depends(auth.current_user),
):
    """State of a channel's relearn run — lets the UI keep showing the
    learning animation even after navigating away and back."""
    with _RELEARN_LOCK:
        st = dict(_RELEARN.get(channel_id) or {"state": "idle", "msg": ""})
    return st


# ─── Phase 2: full-channel video catalogue ────────────────────────────


def _assert_owns_channel(db: Session, user: models.User,
                          google_channel_id: str) -> None:
    """Reject if the user has no OAuth token for ``google_channel_id``."""
    owned = (
        db.query(models.OAuthToken)
          .join(models.Channel, models.Channel.id == models.OAuthToken.channel_id)
          .filter(
              models.Channel.user_id == user.id,
              models.OAuthToken.google_channel_id == google_channel_id,
          )
          .first()
    )
    if not owned:
        raise HTTPException(
            status_code=404,
            detail=f"You don't have a connected YouTube channel matching "
                   f"{google_channel_id!r}.",
        )


@router.post("/yt/{google_channel_id}/sync")
def sync_yt_channel(
    google_channel_id: str,
    background: BackgroundTasks,
    max_videos: int = Query(200, ge=1, le=500),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Fetch the channel's uploads playlist from YouTube and cache it.

    Runs synchronously so the UI can show "synced N videos" right after
    the click — at 200 videos this is still <10 seconds on the wire.
    For massively larger channels the caller can lower max_videos
    or we can promote this to a background task later.
    """
    _assert_owns_channel(db, user, google_channel_id)
    try:
        result = channel_catalog.sync_channel_videos(
            db, user.id, google_channel_id, max_videos=max_videos
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return result


@router.get("/yt/{google_channel_id}/videos")
def list_yt_videos(
    google_channel_id: str,
    limit: int = Query(50, ge=1, le=500),
    offset: int = 0,
    q: Optional[str] = None,
    order_by: str = Query("views", regex=r"^(views|likes|comments|published)$"),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Paginated list of cached videos for one connected YT channel.

    ``q`` is a case-insensitive substring match on the title; useful
    for the explorer dropdown.  ``order_by`` controls the sort axis —
    default is by view count (descending) which is what users almost
    always want when browsing their channel's catalogue.
    """
    _assert_owns_channel(db, user, google_channel_id)
    base = (
        db.query(models.ChannelVideo)
          .filter(
              models.ChannelVideo.user_id == user.id,
              models.ChannelVideo.google_channel_id == google_channel_id,
          )
    )
    if q:
        base = base.filter(models.ChannelVideo.title.ilike(f"%{q.strip()}%"))
    order_col = {
        "views":     models.ChannelVideo.view_count,
        "likes":     models.ChannelVideo.like_count,
        "comments":  models.ChannelVideo.comment_count,
        "published": models.ChannelVideo.published_at,
    }[order_by]
    rows = (base
            .order_by(order_col.desc())
            .offset(offset)
            .limit(limit)
            .all())
    total = base.count()
    return {
        "google_channel_id": google_channel_id,
        "total":             total,
        "limit":             limit,
        "offset":            offset,
        "videos": [
            {
                "video_id":         r.video_id,
                "video_url":        f"https://youtu.be/{r.video_id}",
                "title":            r.title or "",
                "thumbnail_url":    r.thumbnail_url or "",
                "published_at":     r.published_at.isoformat() if r.published_at else None,
                "duration_seconds": int(r.duration_seconds or 0),
                "view_count":       int(r.view_count    or 0),
                "like_count":       int(r.like_count    or 0),
                "comment_count":    int(r.comment_count or 0),
                "engagement_rate":  round(
                    ((r.like_count or 0) + (r.comment_count or 0))
                    / max(r.view_count or 1, 1) * 100, 2
                ),
                "last_synced_at":   r.last_synced_at.isoformat() if r.last_synced_at else None,
            }
            for r in rows
        ],
    }


@router.get("/yt/{google_channel_id}/percentiles")
def channel_percentiles(
    google_channel_id: str,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """View / like / comment percentile distribution for one channel.

    Drives the "where does this clip rank?" bars on the compare card.
    """
    _assert_owns_channel(db, user, google_channel_id)
    return channel_catalog.channel_percentiles(db, user.id, google_channel_id)


# ─── Phase 3: single-video comparison ─────────────────────────────────


@router.get("/compare/video/{video_id}")
def compare_one_video(
    video_id: str,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """One video's stats + its rank in its channel + 3 nearby peers.

    The video must be in the cache — caller is expected to sync the
    channel first.  Used by the "compare this video" card on the
    Performance page.
    """
    try:
        return channel_catalog.compare_video(db, user.id, video_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ─── Phase 3b: one video, ranked across every connected channel ───


@router.get("/compare/video/{video_id}/across-channels")
def compare_video_across_channels(
    video_id: str,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Rank one video within EVERY connected channel's distribution.

    Answers "if I'd posted this video on channel B, where would it
    sit?".  The target must already be in the cache (synced from its
    home channel).  Channels that haven't been synced themselves
    appear in the response with ``needs_sync: true`` so the UI can
    prompt for that action rather than pretend the video would rank
    at the top.
    """
    try:
        return channel_catalog.compare_video_across_channels(
            db, user.id, video_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ─── Phase 4: cross-channel comparison ────────────────────────────────


@router.get("/compare/channels")
def compare_all_channels(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Side-by-side per-channel rollup: median views, cadence, etc.

    Returns one row per connected YT channel the user has synced
    catalogue data for.  Cards / bars on the Performance page render
    this row-by-row so the user can compare their own channels.
    """
    return channel_catalog.compare_channels(db, user.id)


@router.get("/history/{upload_job_id}")
def history(upload_job_id: int, db: Session = Depends(get_db)):
    """Time series of samples for one upload."""
    rows = (
        db.query(models.ClipPerformance)
          .filter(models.ClipPerformance.upload_job_id == upload_job_id)
          .order_by(models.ClipPerformance.sampled_at.asc())
          .all()
    )
    return [{
        "sampled_at":          r.sampled_at.isoformat() if r.sampled_at else None,
        "hours_since_publish": round(r.hours_since_publish or 0, 2),
        "views":               r.views,
        "likes":               r.likes,
        "comments":            r.comments,
        "seo_score":           r.seo_score,
    } for r in rows]


@router.post("/poll")
def trigger_poll(
    background: BackgroundTasks,
    channel_id: Optional[int] = None,
):
    """Force a stats poll now instead of waiting for the next cron tick.

    When ``channel_id`` is supplied, only uploads for that style profile
    are sampled — lets the admin refresh one channel's numbers in
    isolation instead of burning quota on every connected channel.
    Omit the param to keep the original "poll everything" behaviour.
    """
    background.add_task(poller.poll_once, channel_id=channel_id)
    return {"triggered": True, "channel_id": channel_id}
