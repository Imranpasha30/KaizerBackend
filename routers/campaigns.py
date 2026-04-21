"""Campaigns router — CRUD for auto-publish playbooks + attach-to-job + manual trigger."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from database import get_db
import models
import auth
from campaigns import orchestrator


router = APIRouter(prefix="/api/campaigns", tags=["campaigns"])


# ─── Schemas ──────────────────────────────────────────────────────────────

class CampaignIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    channel_ids: List[int] = Field(default_factory=list)
    spacing_minutes: int = Field(120, ge=10, le=1440)
    privacy_status: str = "private"
    auto_seo: bool = True
    auto_translate_to: List[str] = Field(default_factory=list)
    daily_cap: int = Field(0, ge=0, le=100)
    quiet_hours_start: int = Field(0, ge=0, le=23)
    quiet_hours_end: int = Field(0, ge=0, le=23)
    thumbnail_ab: bool = False
    active: bool = True

    @field_validator("privacy_status")
    @classmethod
    def _priv(cls, v: str) -> str:
        v = (v or "").lower().strip()
        if v not in ("public", "private", "unlisted"):
            raise ValueError("privacy_status must be public | private | unlisted")
        return v

    @field_validator("auto_translate_to")
    @classmethod
    def _langs(cls, v: List[str]) -> List[str]:
        return [s.lower().strip()[:5] for s in (v or []) if s.strip()]


class CampaignPatch(BaseModel):
    name: Optional[str] = None
    channel_ids: Optional[List[int]] = None
    spacing_minutes: Optional[int] = Field(None, ge=10, le=1440)
    privacy_status: Optional[str] = None
    auto_seo: Optional[bool] = None
    auto_translate_to: Optional[List[str]] = None
    daily_cap: Optional[int] = Field(None, ge=0, le=100)
    quiet_hours_start: Optional[int] = Field(None, ge=0, le=23)
    quiet_hours_end: Optional[int] = Field(None, ge=0, le=23)
    thumbnail_ab: Optional[bool] = None
    active: Optional[bool] = None


def _to_dict(c: models.Campaign) -> dict:
    return {
        "id": c.id,
        "name": c.name,
        "channel_ids": list(c.channel_ids or []),
        "spacing_minutes": c.spacing_minutes or 120,
        "privacy_status": c.privacy_status or "private",
        "auto_seo": bool(c.auto_seo),
        "auto_translate_to": list(c.auto_translate_to or []),
        "daily_cap": c.daily_cap or 0,
        "quiet_hours_start": c.quiet_hours_start or 0,
        "quiet_hours_end": c.quiet_hours_end or 0,
        "thumbnail_ab": bool(c.thumbnail_ab),
        "active": bool(c.active),
        "created_at": c.created_at.isoformat() if c.created_at else None,
        "updated_at": c.updated_at.isoformat() if c.updated_at else None,
    }


# ─── Endpoints ────────────────────────────────────────────────────────────

@router.get("/")
def list_campaigns(db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    rows = (
        db.query(models.Campaign)
          .filter(models.Campaign.user_id == user.id)
          .order_by(models.Campaign.name)
          .all()
    )
    return [_to_dict(c) for c in rows]


@router.post("/", status_code=201)
def create_campaign(payload: CampaignIn, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    if db.query(models.Campaign).filter(
        models.Campaign.name == payload.name,
        models.Campaign.user_id == user.id,
    ).first():
        raise HTTPException(status_code=409, detail=f"Campaign '{payload.name}' already exists")
    c = models.Campaign(user_id=user.id, **payload.model_dump())
    db.add(c); db.commit(); db.refresh(c)
    return _to_dict(c)


@router.get("/{campaign_id}")
def get_campaign(campaign_id: int, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    c = db.query(models.Campaign).filter(
        models.Campaign.id == campaign_id, models.Campaign.user_id == user.id,
    ).first()
    if not c:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return _to_dict(c)


@router.patch("/{campaign_id}")
def update_campaign(campaign_id: int, payload: CampaignPatch, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    c = db.query(models.Campaign).filter(
        models.Campaign.id == campaign_id, models.Campaign.user_id == user.id,
    ).first()
    if not c:
        raise HTTPException(status_code=404, detail="Campaign not found")
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(c, k, v)
    db.commit(); db.refresh(c)
    return _to_dict(c)


@router.delete("/{campaign_id}")
def delete_campaign(campaign_id: int, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    c = db.query(models.Campaign).filter(
        models.Campaign.id == campaign_id, models.Campaign.user_id == user.id,
    ).first()
    if not c:
        raise HTTPException(status_code=404, detail="Campaign not found")
    db.delete(c); db.commit()
    return {"deleted": campaign_id}


# ─── Job attachment + manual trigger ──────────────────────────────────────

class AttachRequest(BaseModel):
    job_id: int
    auto_run: bool = True


@router.post("/{campaign_id}/attach")
def attach_to_job(
    campaign_id: int,
    payload: AttachRequest,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
):
    c = db.query(models.Campaign).filter(models.Campaign.id == campaign_id).first()
    if not c:
        raise HTTPException(status_code=404, detail="Campaign not found")
    j = db.query(models.Job).filter(models.Job.id == payload.job_id).first()
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")

    existing = (
        db.query(models.JobCampaign)
          .filter(models.JobCampaign.job_id == payload.job_id,
                  models.JobCampaign.campaign_id == campaign_id)
          .first()
    )
    if not existing:
        link = models.JobCampaign(job_id=payload.job_id, campaign_id=campaign_id, status="pending")
        db.add(link); db.commit()

    if payload.auto_run and j.status == "done":
        background.add_task(orchestrator.auto_enqueue, payload.job_id)
    return {"job_id": payload.job_id, "campaign_id": campaign_id, "attached": True}


@router.post("/{campaign_id}/run/{job_id}")
def manual_run(campaign_id: int, job_id: int, background: BackgroundTasks, db: Session = Depends(get_db)):
    """Force a one-shot fan-out even if the campaign wasn't pre-attached."""
    c = db.query(models.Campaign).filter(models.Campaign.id == campaign_id).first()
    if not c:
        raise HTTPException(status_code=404, detail="Campaign not found")
    j = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")
    if j.status != "done":
        raise HTTPException(status_code=409, detail=f"Job status={j.status} — must be 'done' before fan-out")

    existing = (
        db.query(models.JobCampaign)
          .filter(models.JobCampaign.job_id == job_id,
                  models.JobCampaign.campaign_id == campaign_id)
          .first()
    )
    if not existing:
        db.add(models.JobCampaign(job_id=job_id, campaign_id=campaign_id, status="pending"))
        db.commit()
    else:
        existing.status = "pending"
        db.commit()

    background.add_task(orchestrator.auto_enqueue, job_id)
    return {"job_id": job_id, "campaign_id": campaign_id, "queued": True}


@router.get("/job/{job_id}")
def list_job_campaigns(job_id: int, db: Session = Depends(get_db)):
    rows = (
        db.query(models.JobCampaign)
          .filter(models.JobCampaign.job_id == job_id)
          .all()
    )
    return [{
        "id": r.id, "job_id": r.job_id, "campaign_id": r.campaign_id,
        "status": r.status, "last_error": r.last_error or "",
        "created_at": r.created_at.isoformat() if r.created_at else None,
    } for r in rows]
