"""SEO router — per-clip SEO generation, status polling, manual edit, batch."""
from __future__ import annotations

import json
import threading
import traceback
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import SessionLocal, get_db
import models
import auth
from seo import enforcer, generator


router = APIRouter(prefix="/api", tags=["seo"])


# ─── Request schemas ──────────────────────────────────────────────────────────

class GenerateSEORequest(BaseModel):
    # Either `channel_id` (single, legacy) or `channel_ids` (multi, preferred).
    channel_id:  Optional[int] = None
    channel_ids: Optional[List[int]] = None
    force: bool = False
    include_news: bool = True


class BulkGenerateRequest(BaseModel):
    channel_id: int
    force: bool = False
    include_news: bool = True


class UpdateSEORequest(BaseModel):
    title: Optional[str] = Field(None, max_length=150)
    description: Optional[str] = None
    keywords: Optional[List[str]] = None
    hashtags: Optional[List[str]] = None
    hook: Optional[str] = None
    thumbnail_text: Optional[str] = None


# ─── In-process status tracker ────────────────────────────────────────────────
# (This is intentionally simple. A DB-backed table would survive restarts but
#  SEO generation completes in <30 s so ephemeral is fine.)

_generation_status: dict[int, str] = {}   # clip_id -> "generating" | "done" | "error: ..."
_generation_lock = threading.Lock()


def _set_status(clip_id: int, status: str) -> None:
    with _generation_lock:
        _generation_status[clip_id] = status


def _get_status(clip_id: int) -> str:
    with _generation_lock:
        return _generation_status.get(clip_id, "idle")


def _run_generate(clip_id: int, channel_ids: list[int], include_news: bool) -> None:
    """Background worker — generates SEO for each channel sequentially.

    Each successful channel adds a variant under clip.seo_variants[channel_id].
    clip.seo ends up as the last successfully generated variant (for back-compat).
    """
    db = SessionLocal()
    try:
        clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
        if not clip:
            _set_status(clip_id, "error: clip not found")
            return

        import time as _time
        errors: list[str] = []
        ok_count = 0
        for idx, cid in enumerate(channel_ids):
            channel = db.query(models.Channel).filter(models.Channel.id == cid).first()
            if not channel:
                errors.append(f"channel {cid}: not found")
                continue
            _set_status(clip_id, f"generating ({ok_count + 1}/{len(channel_ids)}: {channel.name})")
            try:
                generator.generate_seo_for_clip(clip, channel, db=db, include_news=include_news)
                ok_count += 1
            except Exception as e:
                traceback.print_exc()
                errors.append(f"{channel.name}: {str(e)[:200]}")
                # keep going — one channel failing shouldn't abort the whole batch
            # Small breather between variants — safety net against bursty
            # per-minute limits.  With paid tier this is essentially free;
            # on free tier (10 RPM) 2s keeps us well under the limit.
            if idx < len(channel_ids) - 1:
                _time.sleep(2)

        if ok_count == 0:
            _set_status(clip_id, f"error: {'; '.join(errors) or 'no SEO generated'}")
        elif errors:
            _set_status(clip_id, f"done_with_errors ({ok_count}/{len(channel_ids)}): {'; '.join(errors[:2])}")
        else:
            _set_status(clip_id, "done")
    except Exception as e:
        traceback.print_exc()
        _set_status(clip_id, f"error: {str(e)[:200]}")
    finally:
        db.close()


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/clips/{clip_id}/seo/generate")
def generate_clip_seo(
    clip_id: int,
    payload: GenerateSEORequest,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Kick off SEO generation for one OR many channels in a background thread.

    - If `channel_ids` is set → fan out, producing one variant per channel.
    - If only `channel_id` is set → single-channel path (legacy).
    Poll `.../seo/status` for result.
    """
    clip = (
        db.query(models.Clip)
          .join(models.Job, models.Clip.job_id == models.Job.id)
          .filter(models.Clip.id == clip_id, models.Job.user_id == user.id)
          .first()
    )
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    raw_ids: list[int] = []
    if payload.channel_ids:
        raw_ids.extend(payload.channel_ids)
    if payload.channel_id is not None:
        raw_ids.append(payload.channel_id)
    seen: set[int] = set()
    target_ids = [i for i in raw_ids if not (i in seen or seen.add(i))]
    if not target_ids:
        raise HTTPException(status_code=422, detail="At least one channel must be specified.")

    # Validate all channels belong to this user
    channels = (
        db.query(models.Channel)
          .filter(
              models.Channel.id.in_(target_ids),
              models.Channel.user_id == user.id,
          )
          .all()
    )
    found_ids = {c.id for c in channels}
    missing = [i for i in target_ids if i not in found_ids]
    if missing:
        raise HTTPException(status_code=404, detail=f"Channel(s) not found: {missing}")

    if not payload.force:
        try:
            variants = json.loads(clip.seo_variants or "{}")
        except (ValueError, TypeError):
            variants = {}
        # If every requested channel already has a variant, reject unless forced
        already = [cid for cid in target_ids if str(cid) in variants]
        if len(already) == len(target_ids) and already:
            raise HTTPException(
                status_code=409,
                detail="SEO already exists for all requested channels. Pass force=true to regenerate.",
            )

    if _get_status(clip_id).startswith("generating"):
        return {"status": "generating", "clip_id": clip_id}

    _set_status(clip_id, f"generating (0/{len(target_ids)})")
    threading.Thread(
        target=_run_generate,
        args=(clip_id, target_ids, payload.include_news),
        daemon=True,
    ).start()

    return {
        "status": "generating",
        "clip_id": clip_id,
        "channel_ids": target_ids,
        "variant_count": len(target_ids),
    }


@router.get("/clips/{clip_id}/seo/status")
def get_seo_status(clip_id: int, db: Session = Depends(get_db)):
    clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    status = _get_status(clip_id)
    seo = None
    if clip.seo:
        try:
            seo = json.loads(clip.seo)
        except (ValueError, TypeError):
            seo = None

    return {"clip_id": clip_id, "status": status, "seo": seo}


@router.put("/clips/{clip_id}/seo")
def update_clip_seo(
    clip_id: int,
    payload: UpdateSEORequest,
    db: Session = Depends(get_db),
):
    """Manual edit — merges user changes into the existing SEO JSON and
    re-enforces constraints + score."""
    clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    if not clip.seo:
        raise HTTPException(
            status_code=409,
            detail="No SEO on this clip yet — generate first",
        )

    try:
        seo = json.loads(clip.seo)
    except (ValueError, TypeError):
        raise HTTPException(status_code=500, detail="Existing SEO is corrupt — regenerate")

    updates = payload.model_dump(exclude_unset=True)
    for key, val in updates.items():
        seo[key] = val

    # Re-enforce title length + hashtag shape, then recompute score
    channel = (
        db.query(models.Channel).filter(models.Channel.id == seo.get("channel_id")).first()
        if seo.get("channel_id") else None
    )
    if channel:
        # Only re-enforce the edited fields
        if "title" in updates:
            seo["title"] = enforcer._truncate_title_at_word(seo["title"], channel.name)
        if "hashtags" in updates:
            seo["hashtags"] = enforcer._ensure_hashtags(seo["hashtags"], channel.mandatory_hashtags or [])
        if "keywords" in updates:
            seo["keywords"] = enforcer._ensure_tags(seo["keywords"], channel.fixed_tags or [])
        seo["seo_score"] = enforcer.compute_seo_score(seo, channel)

    seo["edited_by_user"] = True
    seo["edited_at"] = datetime.now(timezone.utc).isoformat()

    clip.seo = json.dumps(seo, ensure_ascii=False)
    db.commit()

    return {"clip_id": clip_id, "seo": seo}


@router.delete("/clips/{clip_id}/seo")
def clear_clip_seo(clip_id: int, db: Session = Depends(get_db)):
    clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    clip.seo = ""
    db.commit()
    _set_status(clip_id, "idle")
    return {"clip_id": clip_id, "cleared": True}


@router.post("/jobs/{job_id}/seo/generate-all")
def generate_job_seo(
    job_id: int,
    payload: BulkGenerateRequest,
    db: Session = Depends(get_db),
):
    """Fan-out SEO generation over every clip in a job."""
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    channel = db.query(models.Channel).filter(models.Channel.id == payload.channel_id).first()
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    targeted = 0
    skipped = 0
    for clip in job.clips:
        if clip.seo and not payload.force:
            skipped += 1
            continue
        _set_status(clip.id, "generating")
        threading.Thread(
            target=_run_generate,
            args=(clip.id, payload.channel_id, payload.include_news),
            daemon=True,
        ).start()
        targeted += 1

    return {
        "job_id": job_id,
        "targeted": targeted,
        "skipped": skipped,
        "total_clips": len(job.clips),
        "channel_id": payload.channel_id,
    }
