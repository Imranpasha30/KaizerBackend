"""Training-data collection — turns each published video into ONE clean,
training-ready row (``models.TrainingSample``): the content + the exact SEO that
shipped (FEATURES) joined to the measured outcome (LABELS) + meta.

Snapshotted at poll time so the export needs NO cleaning at train time — feed
it straight to a model. The features are stored verbatim (so they survive later
clip edits/deletes), plus derived numeric features for classic ML.

Entry points:
  * ``record_sample(db, ...)``  — called by the analytics poller per video.
  * ``backfill(db)``            — (re)build samples from existing performance.
  * ``overview(db)``            — aggregate stats for the admin dashboard.
  * ``iter_rows(db)`` / ``to_jsonl_line(row)`` — the .jsonl export.

Everything is best-effort: learning collection must NEVER break the poll.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterator, Optional

import models


def _parse_seo(clip) -> dict:
    raw = getattr(clip, "seo", None)
    if not raw:
        return {}
    try:
        d = json.loads(raw)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _vph(views: int, hours: float) -> float:
    """Views-per-hour — the rate label (fair across videos of different ages).
    Below ~0.5h we don't divide (too noisy) and just use raw views."""
    return round((float(views) / hours) if hours >= 0.5 else float(views), 3)


def record_sample(
    db,
    *,
    upload_job=None,
    clip=None,
    channel=None,
    views: int = 0,
    likes: int = 0,
    comments: int = 0,
    hours_since_publish: float = 0.0,
    platform: str = "youtube",
    ctr: Optional[float] = None,
    impressions: Optional[int] = None,
) -> Optional["models.TrainingSample"]:
    """Upsert ONE training-ready row for a published video (keyed by video_id),
    updating it to the LATEST outcome. Does NOT commit — the caller owns the
    transaction. Returns the row or None. Never raises."""
    try:
        vid = (getattr(upload_job, "video_id", "") or "").strip()
        if not vid:
            return None
        ch_id = getattr(upload_job, "channel_id", None)
        if channel is None and ch_id is not None:
            channel = db.query(models.Channel).filter(
                models.Channel.id == ch_id).first()

        seo = _parse_seo(clip)
        title = str(seo.get("title", "") or "")
        desc = str(seo.get("description", "") or "")
        kws = [str(k).strip() for k in (seo.get("keywords") or []) if str(k).strip()][:40]
        tags = [str(h).strip() for h in (seo.get("hashtags") or []) if str(h).strip()][:40]
        content = str(getattr(clip, "text", "") or "")[:2000]
        kind = str(getattr(clip, "frame_type", "") or "")
        language = str(getattr(channel, "language", "") or "")
        ch_name = str(getattr(channel, "name", "") or "")
        user_id = getattr(channel, "user_id", None)
        try:
            score = int(seo.get("seo_score") or seo.get("tool_score") or 0)
        except Exception:
            score = 0
        hours = max(0.0, float(hours_since_publish or 0.0))

        row = db.query(models.TrainingSample).filter(
            models.TrainingSample.video_id == vid).first()
        if row is None:
            row = models.TrainingSample(video_id=vid)
            db.add(row)

        # meta (only overwrite when we have a value)
        if user_id is not None:
            row.user_id = user_id
        if ch_id is not None:
            row.channel_id = ch_id
        if getattr(clip, "id", None):
            row.clip_id = clip.id
        if getattr(upload_job, "id", None):
            row.upload_job_id = upload_job.id
        if ch_name:
            row.channel_name = ch_name
        row.platform = platform or "youtube"
        if language:
            row.language = language
        if kind:
            row.kind = kind
        # features (verbatim snapshot)
        row.content_text = content
        row.seo_title = title[:300]
        row.seo_description = desc[:5000]
        row.seo_keywords = kws
        row.seo_hashtags = tags
        row.style_source_id = seo.get("style_source_id")
        row.title_len = len(title)
        row.desc_len = len(desc)
        row.keyword_count = len(kws)
        row.hashtag_count = len(tags)
        row.seo_score = score
        # A/B exploration ledger: the hook form this SEO deliberately
        # probed (stamped by the generator), or None for exploit rounds.
        try:
            row.explored_hook = (str(seo.get("explored_hook") or "")[:12]
                                 or None)
        except Exception:
            pass
        # labels (latest)
        row.views = int(views or 0)
        row.likes = int(likes or 0)
        row.comments = int(comments or 0)
        row.hours_since_publish = hours
        row.views_per_hour = _vph(int(views or 0), hours)
        if ctr is not None:
            try:
                row.ctr = float(ctr)
            except Exception:
                pass
        if impressions is not None:
            try:
                row.impressions = int(impressions)
            except Exception:
                pass
        return row
    except Exception:
        return None


def backfill(db) -> Dict[str, int]:
    """(Re)build training samples from the LATEST ClipPerformance row per video
    + its clip/channel. Idempotent (upsert by video_id). One-time bootstrap so
    the dataset has data before the next poll. Best-effort; commits itself."""
    built = 0
    skipped = 0
    try:
        # Latest performance row per video_id.
        rows = (
            db.query(models.ClipPerformance)
              .filter(models.ClipPerformance.video_id != "")
              .order_by(models.ClipPerformance.video_id,
                        models.ClipPerformance.sampled_at.desc())
              .all()
        )
        seen: set[str] = set()
        for perf in rows:
            vid = (perf.video_id or "").strip()
            if not vid or vid in seen:
                continue
            seen.add(vid)
            clip = (db.query(models.Clip).filter(models.Clip.id == perf.clip_id).first()
                    if perf.clip_id else None)
            channel = (db.query(models.Channel).filter(
                models.Channel.id == perf.channel_id).first()
                if perf.channel_id else None)
            uj = (db.query(models.UploadJob).filter(
                models.UploadJob.id == perf.upload_job_id).first()
                if perf.upload_job_id else None)
            # Synthesize an upload_job-shaped object if the legacy row is gone.
            if uj is None:
                from types import SimpleNamespace
                uj = SimpleNamespace(video_id=vid, channel_id=perf.channel_id, id=perf.upload_job_id)
            r = record_sample(
                db, upload_job=uj, clip=clip, channel=channel,
                views=perf.views, likes=perf.likes, comments=perf.comments,
                hours_since_publish=perf.hours_since_publish,
            )
            if r is not None:
                built += 1
            else:
                skipped += 1
        db.commit()
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
    return {"built": built, "skipped": skipped}


def overview(db) -> Dict[str, Any]:
    """Aggregate stats for the admin Learning dashboard."""
    out: Dict[str, Any] = {
        "total": 0, "with_views": 0, "with_ctr": 0,
        "languages": {}, "platforms": {}, "by_channel": [],
        "avg_views_per_hour": 0.0, "top_keywords": [],
    }
    try:
        rows = db.query(models.TrainingSample).all()
        out["total"] = len(rows)
        if not rows:
            return out
        from collections import Counter
        kw_perf: Dict[str, float] = {}
        kw_n: Dict[str, int] = {}
        lang_c: Counter = Counter()
        plat_c: Counter = Counter()
        ch_agg: Dict[int, Dict[str, Any]] = {}
        vph_sum = 0.0
        for r in rows:
            if (r.views or 0) > 0:
                out["with_views"] += 1
            if r.ctr is not None:
                out["with_ctr"] += 1
            lang_c[r.language or "?"] += 1
            plat_c[r.platform or "?"] += 1
            vph_sum += float(r.views_per_hour or 0.0)
            ca = ch_agg.setdefault(r.channel_id or 0, {
                "channel_id": r.channel_id, "channel_name": r.channel_name or "",
                "samples": 0, "views": 0})
            ca["samples"] += 1
            ca["views"] += int(r.views or 0)
            # winning keywords weighted by views-per-hour
            for k in (r.seo_keywords or []):
                kw_perf[k] = kw_perf.get(k, 0.0) + float(r.views_per_hour or 0.0)
                kw_n[k] = kw_n.get(k, 0) + 1
        out["languages"] = dict(lang_c)
        out["platforms"] = dict(plat_c)
        out["avg_views_per_hour"] = round(vph_sum / max(1, len(rows)), 3)
        out["by_channel"] = sorted(ch_agg.values(), key=lambda c: c["views"], reverse=True)[:20]
        # top winning keywords (need ≥2 samples to count, ranked by total vph)
        ranked = sorted(
            ((k, kw_perf[k], kw_n[k]) for k in kw_perf if kw_n[k] >= 2),
            key=lambda t: t[1], reverse=True)[:30]
        out["top_keywords"] = [{"keyword": k, "score": round(v, 2), "n": n} for k, v, n in ranked]
    except Exception:
        pass
    return out


def iter_rows(db, limit: Optional[int] = None) -> Iterator["models.TrainingSample"]:
    q = db.query(models.TrainingSample).order_by(models.TrainingSample.id.asc())
    if limit:
        q = q.limit(int(limit))
    for r in q:
        yield r


def to_export_dict(r: "models.TrainingSample") -> Dict[str, Any]:
    """The clean, training-ready shape. ``input`` = what you'd give a model to
    generate SEO; ``output`` = the SEO that shipped; ``label`` = the outcome;
    ``features`` = derived numerics. Directly usable for SFT / reward modeling /
    classic ML — no preprocessing needed."""
    return {
        "input": {
            "content": r.content_text or "",
            "language": r.language or "",
            "kind": r.kind or "",
            "platform": r.platform or "",
            "style_source_id": r.style_source_id,
        },
        "output": {
            "title": r.seo_title or "",
            "description": r.seo_description or "",
            "keywords": r.seo_keywords or [],
            "hashtags": r.seo_hashtags or [],
        },
        "features": {
            "title_len": r.title_len or 0,
            "desc_len": r.desc_len or 0,
            "keyword_count": r.keyword_count or 0,
            "hashtag_count": r.hashtag_count or 0,
            "seo_score": r.seo_score or 0,
        },
        "label": {
            "views": r.views or 0,
            "likes": r.likes or 0,
            "comments": r.comments or 0,
            "hours_since_publish": round(float(r.hours_since_publish or 0.0), 2),
            "views_per_hour": round(float(r.views_per_hour or 0.0), 3),
            "ctr": r.ctr,
        },
        "meta": {
            "video_id": r.video_id,
            "channel_id": r.channel_id,
            "channel_name": r.channel_name or "",
            "user_id": r.user_id,
            "clip_id": r.clip_id,
            "captured_at": r.captured_at.isoformat() if r.captured_at else None,
        },
    }
