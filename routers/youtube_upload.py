"""Upload endpoints — enqueue publish, list queue, detail, cancel, progress SSE."""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from database import SessionLocal, get_db
import models
import auth
from youtube import quota


router = APIRouter(prefix="/api", tags=["youtube-upload"])


# ─── Schemas ──────────────────────────────────────────────────────────────

class PublishRequest(BaseModel):
    # Legacy single-target support — kept so existing frontend still works.
    channel_id:     Optional[int] = None
    # New: fan-out to multiple style profiles in one call.  Each entry creates
    # its own UploadJob queued independently.  If set, `channel_id` is ignored.
    channel_ids:    Optional[List[int]] = None
    privacy_status: str = "private"
    publish_at:     Optional[datetime] = None
    use_seo:        bool = True
    # "short" → append #Shorts hashtag (YouTube auto-classifies vertical ≤60s
    # clips with that hashtag as a Short). "video" → standard upload.
    publish_kind:   str = "video"
    # When set, force every destination to use THIS channel's SEO variant
    # from clip.seo_variants (overrides the per-destination auto-match).
    seo_variant_override: Optional[int] = None
    # Per-destination override map: { "<dest_channel_id>": <variant_channel_id> }.
    # Wins over `seo_variant_override` for destinations it specifies.  Lets a
    # user publish Auto Wala with "Suman TV Live" voice AND Cyber Sphere with
    # "Personal 2" voice in the same click.
    variant_by_channel: Optional[dict[str, int]] = None
    # Optional overrides if use_seo is False or the user wants to tweak at publish time
    title:          Optional[str] = Field(None, max_length=150)
    description:    Optional[str] = None
    tags:           Optional[List[str]] = None
    category_id:    Optional[str] = None
    made_for_kids:  bool = False

    @field_validator("privacy_status")
    @classmethod
    def _privacy(cls, v: str) -> str:
        v = (v or "").lower().strip()
        if v not in ("public", "private", "unlisted"):
            raise ValueError("privacy_status must be public | private | unlisted")
        return v

    @field_validator("publish_kind")
    @classmethod
    def _publish_kind(cls, v: str) -> str:
        v = (v or "video").lower().strip()
        if v not in ("short", "video"):
            raise ValueError("publish_kind must be 'short' or 'video'")
        return v

    @field_validator("publish_at")
    @classmethod
    def _publish_at(cls, v: Optional[datetime]) -> Optional[datetime]:
        if v is None:
            return None
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        if v <= datetime.now(timezone.utc):
            raise ValueError("publish_at must be in the future")
        return v


# ─── Helpers ──────────────────────────────────────────────────────────────

def _to_dict(job: models.UploadJob) -> dict:
    clip = job.clip
    channel = job.channel
    return {
        "id":             job.id,
        "clip_id":        job.clip_id,
        "channel_id":     job.channel_id,
        "channel_name":   channel.name if channel else None,
        "clip_filename":  clip.filename if clip else None,
        "clip_thumb_url": f"/api/file/?path={clip.thumb_path}" if clip and clip.thumb_path else "",
        "status":         job.status,
        "privacy_status": job.privacy_status,
        "publish_kind":   job.publish_kind or "video",
        "publish_at":     job.publish_at.isoformat() if job.publish_at else None,
        "title":          job.title,
        "description":    job.description,
        "tags":           list(job.tags or []),
        "category_id":    job.category_id,
        "made_for_kids":  bool(job.made_for_kids),
        "video_id":       job.video_id or None,
        "video_url":      f"https://youtu.be/{job.video_id}" if job.video_id else None,
        "bytes_uploaded": job.bytes_uploaded or 0,
        "bytes_total":    job.bytes_total or 0,
        "progress_pct":   round(100 * (job.bytes_uploaded or 0) / max(job.bytes_total or 1, 1), 1),
        "attempts":       job.attempts or 0,
        "last_error":     job.last_error or "",
        "log":            job.log or "",
        "created_at":     job.created_at.isoformat() if job.created_at else None,
        "updated_at":     job.updated_at.isoformat() if job.updated_at else None,
    }


def _compose_metadata(
    clip: models.Clip,
    channel: models.Channel,
    payload: PublishRequest,
) -> tuple[str, str, list[str]]:
    """Resolve title / description / tags for an upload to `channel`.

    Content + Brand Overlay model:
      - `clip.seo` holds the GENERIC (channel-agnostic) SEO produced by the
        generator.
      - `seo.composer.compose(generic, channel)` overlays the destination's
        name suffix, mandatory hashtags, fixed tags, and footer.
      - Result is the exact metadata uploaded to YouTube for this destination.

    Legacy `clip.seo_variants` is still read as a fallback for older clips
    that were generated before the refactor (they stored per-channel full
    SEO blobs instead of a single generic one).  When present, we apply the
    variant AS-IS for that destination (no composer re-pass — it's already
    channel-scoped from the old flow).

    Shorts handling is performed inside composer.compose().
    """
    # Manual override wins everything — user typed title/desc in PublishModal
    if not payload.use_seo:
        title = (payload.title or clip.text or f"Kaizer clip #{clip.id}")[:100]
        description = payload.description or ""
        tags = payload.tags or []
        if payload.publish_kind == "short":
            if "shorts" not in [t.lower() for t in tags]:
                tags = ["shorts", *tags]
        from youtube.uploader import sanitize_tags
        return title, description, sanitize_tags(tags)

    # ─── Try generic SEO + brand overlay (new path) ───
    generic: dict = {}
    if clip.seo:
        try:
            g = json.loads(clip.seo)
            if isinstance(g, dict) and g.get("title"):
                generic = g
        except (ValueError, TypeError):
            pass

    if generic:
        from seo.composer import compose
        composed = compose(generic, channel, publish_kind=payload.publish_kind)
        title = (payload.title or composed["title"])[:100]
        description = payload.description or composed["description"]
        tags = payload.tags if payload.tags is not None else composed["keywords"]
        from youtube.uploader import sanitize_tags
        return title, description, sanitize_tags(tags)

    # ─── Legacy fallback: per-channel variants from old generation runs ───
    try:
        variants = json.loads(clip.seo_variants or "{}")
    except (ValueError, TypeError):
        variants = {}
    if not isinstance(variants, dict):
        variants = {}

    per_dest = payload.variant_by_channel or {}
    dest_key = str(channel.id)
    legacy_seo: dict = {}
    if dest_key in per_dest:
        v_key = str(per_dest[dest_key])
        if v_key in variants:
            legacy_seo = variants[v_key] or {}
    if not legacy_seo and payload.seo_variant_override is not None:
        key = str(payload.seo_variant_override)
        if key in variants:
            legacy_seo = variants[key] or {}
    if not legacy_seo and dest_key in variants:
        legacy_seo = variants[dest_key] or {}

    title = (payload.title
             or legacy_seo.get("title")
             or clip.text
             or f"Kaizer clip #{clip.id}")[:100]
    description = payload.description or legacy_seo.get("description") or ""
    tags = payload.tags if payload.tags is not None else (legacy_seo.get("keywords") or [])

    if payload.publish_kind == "short":
        shorts_tag = "#Shorts"
        if shorts_tag.lower() not in title.lower():
            candidate = f"{title} {shorts_tag}"
            if len(candidate) <= 100:
                title = candidate
            elif shorts_tag.lower() not in description.lower():
                description = (shorts_tag + "\n\n" + description).strip()
        if "shorts" not in [t.lower() for t in tags]:
            tags = ["shorts", *tags]

    from youtube.uploader import sanitize_tags
    return title, description, sanitize_tags(tags)


# ─── Endpoints ────────────────────────────────────────────────────────────

@router.post("/clips/{clip_id}/publish")
def publish_clip(clip_id: int, payload: PublishRequest, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    """Publish a clip to one or more YouTube destinations in one call.

    Accepts either the legacy `channel_id` (single) or the newer `channel_ids`
    list (fan-out → one UploadJob per entry).  Returns a single dict when only
    one target was requested, or `{jobs: [...]}` for a fan-out.
    """
    clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    if not clip.file_path:
        raise HTTPException(status_code=422, detail="Clip has no rendered file — run the pipeline first")

    # Resolve target list — dedupe but preserve order
    raw_ids: list[int] = []
    if payload.channel_ids:
        raw_ids.extend(payload.channel_ids)
    if payload.channel_id is not None:
        raw_ids.append(payload.channel_id)
    seen: set[int] = set()
    target_ids = [i for i in raw_ids if not (i in seen or seen.add(i))]
    if not target_ids:
        raise HTTPException(status_code=422, detail="At least one destination must be selected.")

    # Look up all targets up-front and validate them before creating any jobs.
    channels = db.query(models.Channel).filter(models.Channel.id.in_(target_ids)).all()
    by_id = {c.id: c for c in channels}
    missing = [i for i in target_ids if i not in by_id]
    if missing:
        raise HTTPException(status_code=404, detail=f"Profile(s) not found: {missing}")

    for cid in target_ids:
        ch = by_id[cid]
        if not ch.oauth_token or not ch.oauth_token.refresh_token_enc:
            raise HTTPException(
                status_code=409,
                detail=f"Style profile '{ch.name}' is not linked to YouTube. Open Style Profiles → Link my YT.",
            )

    # Accept either a legacy clip.seo OR a per-channel variant
    if payload.use_seo:
        try:
            _variants_check = json.loads(clip.seo_variants or "{}")
        except (ValueError, TypeError):
            _variants_check = {}
        if not clip.seo and not (isinstance(_variants_check, dict) and _variants_check):
            raise HTTPException(
                status_code=409,
                detail="Clip has no SEO metadata. Generate SEO first, or pass use_seo=false with manual title/description.",
            )

    # Scheduled uploads must be private-until-publish
    if payload.publish_at and payload.privacy_status != "private":
        raise HTTPException(
            status_code=422,
            detail="Scheduled uploads require privacy_status='private' (YouTube flips it to public at publish_at).",
        )

    # Fan-out: one UploadJob per target, each with independently composed metadata
    created: list[models.UploadJob] = []
    for cid in target_ids:
        channel = by_id[cid]
        title, description, tags = _compose_metadata(clip, channel, payload)
        job = models.UploadJob(
            user_id=user.id,
            clip_id=clip.id,
            channel_id=channel.id,
            status="queued",
            privacy_status=payload.privacy_status,
            publish_kind=payload.publish_kind,
            publish_at=payload.publish_at,
            title=title,
            description=description,
            tags=tags,
            category_id=payload.category_id or "25",
            made_for_kids=payload.made_for_kids,
        )
        db.add(job)
        created.append(job)
    db.commit()
    for j in created:
        db.refresh(j)

    # Single-target → legacy shape; fan-out → list under `jobs`
    if len(created) == 1:
        return _to_dict(created[0])
    return {"jobs": [_to_dict(j) for j in created], "count": len(created)}


@router.get("/uploads")
def list_uploads(
    status_filter: Optional[str] = Query(None, alias="status"),
    channel_id:    Optional[int] = None,
    limit:         int = 100,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    q = db.query(models.UploadJob).filter(models.UploadJob.user_id == user.id)
    if status_filter:
        q = q.filter(models.UploadJob.status == status_filter)
    if channel_id:
        q = q.filter(models.UploadJob.channel_id == channel_id)
    rows = q.order_by(models.UploadJob.created_at.desc()).limit(limit).all()
    return [_to_dict(r) for r in rows]


@router.get("/uploads/{upload_id}")
def get_upload(upload_id: int, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    row = db.query(models.UploadJob).filter(
        models.UploadJob.id == upload_id, models.UploadJob.user_id == user.id,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Upload not found")
    return _to_dict(row)


@router.delete("/uploads/{upload_id}")
def cancel_upload(upload_id: int, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    row = db.query(models.UploadJob).filter(
        models.UploadJob.id == upload_id, models.UploadJob.user_id == user.id,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Upload not found")
    if row.status in ("done", "failed", "cancelled"):
        return {"upload_id": upload_id, "status": row.status, "cancelled": False,
                "note": f"already terminal ({row.status})"}
    if row.status == "uploading":
        # We can only mark the row; in-flight chunks will notice on next checkpoint
        row.status = "cancelled"
        row.last_error = (row.last_error or "") + "\n[user] cancelled"
        db.commit()
        return {"upload_id": upload_id, "status": "cancelled", "cancelled": True,
                "note": "in-flight chunks may continue briefly"}
    row.status = "cancelled"
    db.commit()
    return {"upload_id": upload_id, "status": "cancelled", "cancelled": True}


@router.post("/uploads/{upload_id}/retry")
def retry_upload(upload_id: int, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    row = db.query(models.UploadJob).filter(
        models.UploadJob.id == upload_id, models.UploadJob.user_id == user.id,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Upload not found")
    if row.status not in ("failed", "cancelled"):
        raise HTTPException(status_code=409, detail=f"Cannot retry while status={row.status}")
    row.status = "queued"
    row.attempts = 0
    row.last_error = ""
    # Keep upload_uri + bytes_uploaded so we resume where we left off
    db.commit()
    return _to_dict(row)


@router.get("/uploads/{upload_id}/log")
def stream_log(upload_id: int):
    """SSE stream — emits job dict every POLL_MS until status is terminal."""
    def event_stream():
        last_payload: Optional[str] = None
        # 15 minutes of polling max; UI should give up or reconnect
        for _ in range(15 * 60 * 2):
            db = SessionLocal()
            try:
                row = db.query(models.UploadJob).filter(models.UploadJob.id == upload_id).first()
                if not row:
                    yield f"data: {json.dumps({'error': 'not found'})}\n\n"
                    return
                payload = json.dumps(_to_dict(row), ensure_ascii=False, default=str)
                if payload != last_payload:
                    yield f"data: {payload}\n\n"
                    last_payload = payload
                if row.status in ("done", "failed", "cancelled"):
                    return
            finally:
                db.close()
            time.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/quota")
def get_quota(db: Session = Depends(get_db)):
    """Lightweight quota snapshot for the UI."""
    return quota.snapshot(db)
