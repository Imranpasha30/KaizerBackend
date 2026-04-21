"""Translation router — on-demand translate + fan-out + list."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from database import SessionLocal, get_db
import models
from translation import translator


router = APIRouter(prefix="/api", tags=["translation"])


class TranslateRequest(BaseModel):
    language: str = Field(..., min_length=2, max_length=5)
    target_channel_id: int

    @field_validator("language")
    @classmethod
    def _lower(cls, v: str) -> str:
        return v.lower().strip()


def _run_translate(clip_id: int, language: str, target_channel_id: int) -> None:
    db = SessionLocal()
    try:
        translator.ensure_translation(db, clip_id, language, target_channel_id)
    except translator.TranslationError as e:
        print(f"[translate] clip {clip_id} → {language}: {e}")
    except Exception as e:
        print(f"[translate] clip {clip_id} → {language} unexpected: {e}")
    finally:
        db.close()


@router.post("/clips/{clip_id}/translate", status_code=202)
def translate_clip(
    clip_id: int,
    payload: TranslateRequest,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
):
    clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    target = db.query(models.Channel).filter(models.Channel.id == payload.target_channel_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="Target channel not found")
    if not target.oauth_token or not target.oauth_token.refresh_token_enc:
        raise HTTPException(status_code=409,
            detail=f"Target channel '{target.name}' is not connected to YouTube.")

    background.add_task(_run_translate, clip_id, payload.language, payload.target_channel_id)
    return {"clip_id": clip_id, "language": payload.language,
            "target_channel_id": payload.target_channel_id, "status": "queued"}


@router.get("/clips/{clip_id}/translations")
def list_translations(clip_id: int, db: Session = Depends(get_db)):
    rows = db.query(models.ClipTranslation).filter(
        models.ClipTranslation.clip_id == clip_id,
    ).all()
    return [{
        "language": r.language,
        "payload":  r.payload or {},
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "updated_at": r.updated_at.isoformat() if r.updated_at else None,
    } for r in rows]
