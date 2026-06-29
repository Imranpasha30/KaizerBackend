"""Pipeline → Campaign → Upload orchestrator.

Called when a pipeline Job finishes. Fans clips out to each configured
channel, generates SEO if missing, picks the next slot per channel, and
inserts UploadJob rows. Translation fan-out (Phase D) and thumbnail A/B
(Phase C) are invoked here too when enabled.

Runs in a daemon thread — the pipeline just calls `auto_enqueue(job_id)`
and returns.
"""
from __future__ import annotations

import threading
import traceback
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

import models
from database import SessionLocal
from campaigns import scheduler as slot_scheduler


def auto_enqueue_async(job_id: int) -> None:
    """Non-blocking entrypoint for the pipeline runner."""
    threading.Thread(
        target=_auto_enqueue_safe,
        args=(job_id,),
        daemon=True,
        name=f"campaign-enqueue-job-{job_id}",
    ).start()


def _auto_enqueue_safe(job_id: int) -> None:
    try:
        auto_enqueue(job_id)
    except Exception:
        traceback.print_exc()


def auto_enqueue(job_id: int) -> dict:
    """For every campaign attached to this job, fan out all rendered clips."""
    db = SessionLocal()
    try:
        links = (
            db.query(models.JobCampaign)
              .filter(models.JobCampaign.job_id == job_id)
              .filter(models.JobCampaign.status.in_(["pending", "failed"]))
              .all()
        )
        if not links:
            return {"job_id": job_id, "enqueued": 0, "note": "no campaigns attached"}

        total_queued = 0
        for link in links:
            try:
                link.status = "seo"
                db.commit()
                queued = _enqueue_job_on_campaign(db, job_id, link.campaign_id)
                total_queued += queued
                link.status = "scheduled" if queued else "done"
                db.commit()
            except Exception as e:
                link.status = "failed"
                link.last_error = str(e)[:500]
                db.commit()

        return {"job_id": job_id, "enqueued": total_queued, "campaigns": len(links)}
    finally:
        db.close()


def _resolve_campaign_user(db: Session, camp: models.Campaign, job_id: int):
    """The owner whose YouTube credentials + credits the publish runs under.
    Prefer the campaign's owner; fall back to the job's owner."""
    uid = getattr(camp, "user_id", None)
    if uid is None:
        job = db.query(models.Job).filter(models.Job.id == job_id).first()
        uid = getattr(job, "user_id", None)
    if uid is None:
        return None
    return db.query(models.User).filter(models.User.id == uid).first()


def _detect_publish_kind(clip: models.Clip) -> str:
    """Bulletin / full-length renders publish as 'video'; the vertical shorts
    a V4 job emits publish as 'short'."""
    ft = (getattr(clip, "frame_type", "") or "").lower()
    if ft in ("bulletin", "raw_upload", "youtube_full"):
        return "video"
    return "short"


def _already_published_v2(db: Session, clip_id: int, channel_id: int) -> bool:
    """True if this clip already has a live (non-dead) V2 upload to this
    channel — guards against double-publishing on re-runs / retries."""
    master = (
        db.query(models.MasterVideo)
          .filter(models.MasterVideo.clip_id == int(clip_id))
          .first()
    )
    if not master:
        return False
    return (
        db.query(models.UploadJobV2.id)
          .join(models.PublishTask,
                models.UploadJobV2.publish_task_id == models.PublishTask.id)
          .filter(models.PublishTask.master_video_id == master.id)
          .filter(models.UploadJobV2.channel_id == int(channel_id))
          .filter(models.UploadJobV2.status.notin_(["cancelled", "failed"]))
          .first() is not None
    )


def _enqueue_job_on_campaign(db: Session, job_id: int, campaign_id: int) -> int:
    """Fan a finished job's clips out to the campaign's channels via the LIVE
    V2 publish path (MasterVideo → PublishTask → UploadJobV2 on the durable
    queue), spaced per channel. Reuses the exact bridge the manual publish UI
    uses, so campaigns inherit per-channel SEO, branding, scheduling, credit
    reservation and idempotency for free."""
    camp = db.query(models.Campaign).filter(models.Campaign.id == campaign_id).first()
    if not camp or not camp.active:
        return 0

    user = _resolve_campaign_user(db, camp, job_id)
    if user is None:
        print(f"[campaigns] no owner user for campaign {campaign_id} — skipping")
        return 0

    clips = (
        db.query(models.Clip)
          .filter(models.Clip.job_id == job_id)
          .filter(models.Clip.file_path != "")
          .order_by(models.Clip.clip_index.asc())
          .all()
    )
    if not clips:
        return 0

    if camp.auto_translate_to or camp.thumbnail_ab:
        print(f"[campaigns] note: auto_translate_to / thumbnail_ab are not yet "
              f"wired to the V2 publish path — skipped for campaign {campaign_id}")

    # Lazy import — routers.youtube_upload pulls in services.* which would
    # create a circular import if loaded at module top.
    from routers.youtube_upload import PublishRequest, _legacy_to_v2_redirect

    queued = 0
    for channel_id in (camp.channel_ids or []):
        channel = db.query(models.Channel).filter(models.Channel.id == channel_id).first()
        if not channel:
            continue
        if not channel.oauth_token or not channel.oauth_token.refresh_token_enc:
            continue  # not a connected publish target — skip silently

        for clip in clips:
            if _already_published_v2(db, clip.id, channel_id):
                continue

            # Daily cap — count this channel's live + scheduled uploads today.
            if camp.daily_cap and slot_scheduler.count_scheduled_today(db, channel_id) >= camp.daily_cap:
                break

            # auto-SEO: generate ONLY when the clip has none yet. The publish
            # path composes per-channel branding onto clip.seo at dispatch.
            if camp.auto_seo and not (clip.seo or "").strip():
                _generate_seo_inline(db, clip.id, channel_id)
                db.refresh(clip)

            payload = PublishRequest(
                channel_ids=[int(channel_id)],
                # VISIBILITY IS DECIDED ONLY IN THE PUBLISH PANEL (operator rule
                # 2026-06-18). A plan must never silently force a video private.
                # Plan-published clips go PUBLIC immediately. If the operator
                # wants timed/scheduled release, they choose "Scheduled" in the
                # Publish panel — the single place visibility is controlled.
                # (Scheduled drip used to upload private + publish_at so YouTube
                # auto-flipped it public; that hid videos as private, which is the
                # exact confusion being removed here.)
                privacy_status="public",
                publish_at=None,
                use_seo=True,
                publish_kind=_detect_publish_kind(clip),
                brand_mode="per_channel",
            )
            try:
                _legacy_to_v2_redirect(db, user, int(clip.id), payload)
                queued += 1
            except Exception as e:
                try:
                    db.rollback()
                except Exception:
                    pass
                detail = getattr(e, "detail", None) or str(e)
                print(f"[campaigns] publish failed clip={clip.id} ch={channel_id}: {detail}")

    return queued


def _already_queued(db: Session, clip_id: int, channel_id: int) -> bool:
    return db.query(models.UploadJob).filter(
        models.UploadJob.clip_id == clip_id,
        models.UploadJob.channel_id == channel_id,
        models.UploadJob.status.in_(["queued", "uploading", "processing", "done"]),
    ).first() is not None


def _generate_seo_inline(db: Session, clip_id: int, channel_id: int) -> None:
    """Blocking SEO generation for the orchestrator path (non-pipeline thread)."""
    from seo import generator as seo_gen
    clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
    channel = db.query(models.Channel).filter(models.Channel.id == channel_id).first()
    if not clip or not channel:
        return
    try:
        seo_gen.generate_for_clip(clip, channel, db, include_news=True, include_corpus=True)
    except Exception as e:
        print(f"[campaign] inline SEO failed for clip {clip_id}: {e}")


def _compose_metadata(clip: models.Clip, channel: models.Channel):
    """Resolve title/description/tags from clip.seo. Falls back to raw text."""
    import json as _json
    seo = {}
    if clip.seo:
        try: seo = _json.loads(clip.seo)
        except (ValueError, TypeError): seo = {}
    title = (seo.get("title") or clip.text or f"Kaizer clip #{clip.id}")[:100]
    description = seo.get("description") or ""
    tags = seo.get("keywords") or []
    deduped, total = [], 0
    for t in tags:
        t = (t or "").strip()
        if not t: continue
        if total + len(t) + 2 > 500: break
        deduped.append(t); total += len(t) + 2
    return title, description, deduped


def _translate_and_fanout(db: Session, clip: models.Clip, source_channel: models.Channel,
                          camp: models.Campaign, source_upload_id: int) -> None:
    """Phase D — for each target language, find a channel speaking it and
    queue a parallel upload with translated SEO."""
    try:
        from translation import translator
    except Exception:
        return

    for lang in (camp.auto_translate_to or []):
        lang = (lang or "").lower().strip()
        if not lang or lang == (source_channel.language or "te").lower():
            continue
        # Find a connected channel whose language matches AND is in the campaign's channel set.
        target = (
            db.query(models.Channel)
              .filter(models.Channel.language == lang)
              .filter(models.Channel.id.in_(camp.channel_ids or []))
              .first()
        )
        if not target or not target.oauth_token:
            continue

        try:
            translator.ensure_translation(db, clip.id, lang, target.id)
        except Exception as e:
            print(f"[campaign] translation to {lang} failed: {e}")


def _queue_thumbnail_variants(db: Session, clip: models.Clip, upload_id: int) -> None:
    """Phase C — register A/B variants for later swap (runs at +6h via scheduler)."""
    try:
        from thumbnails_ab import generator as thumb_gen
        thumb_gen.register_variants(db, upload_id, clip)
    except Exception as e:
        print(f"[campaign] thumbnail variant registration failed: {e}")
