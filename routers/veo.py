"""Veo video generation router.

Flow (all async in a background thread):
  1. Look up the trending topic (must belong to current user's competitor)
  2. Build a Veo prompt from the topic
  3. Call Veo 3, save MP4
  4. Auto-generate a thumbnail from frame 1 (ffmpeg)
  5. Create a Job + Clip row so the video flows through the existing SEO /
     Publish / Uploads pipeline just like a normally-rendered clip
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

import auth
import models
from database import SessionLocal, get_db
from veo import generator as veo_gen
from veo import prompt_builder


router = APIRouter(prefix="/api/veo", tags=["veo"])


# ─── Status tracker (in-memory; fine for single-box deploy) ──────────────

_status: dict[int, dict] = {}   # topic_id → {"state": str, "job_id": int|None, "error": str}
_lock = threading.Lock()


def _set_status(topic_id: int, **kw):
    with _lock:
        cur = _status.get(topic_id, {})
        cur.update(kw)
        _status[topic_id] = cur


def _get_status(topic_id: int) -> dict:
    with _lock:
        return dict(_status.get(topic_id) or {"state": "idle"})


# ─── Background worker ───────────────────────────────────────────────────

def _run_veo_job(topic_id: int, user_id: int, platform: str, language: str) -> None:
    """Generate → save → create Job+Clip → kick SEO.  Own DB session."""
    db = SessionLocal()
    try:
        topic = db.query(models.TrendingTopic).filter(
            models.TrendingTopic.id == topic_id
        ).first()
        if not topic:
            _set_status(topic_id, state="error", error="Topic not found")
            return

        _set_status(topic_id, state="building_prompt")
        prompt_cfg = prompt_builder.build_prompt(topic, platform=platform)

        # Output dir under standard OUTPUT_ROOT so /api/file/ serves it
        from runner import OUTPUT_ROOT
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(OUTPUT_ROOT) / "veo" / f"topic_{topic_id}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        video_path = out_dir / "clip_veo.mp4"

        _set_status(topic_id, state="generating")
        try:
            meta = veo_gen.generate_video(
                prompt=prompt_cfg["prompt"],
                out_path=video_path,
                aspect_ratio=prompt_cfg["aspect_ratio"],
                duration_seconds=prompt_cfg["duration_seconds"],
                negative_prompt=prompt_cfg.get("negative_prompt", ""),
                progress_cb=lambda m: _set_status(topic_id, state=f"generating: {m}"),
            )
        except veo_gen.VeoError as e:
            _set_status(topic_id, state="error", error=str(e))
            return

        # Thumbnail via ffmpeg (reuse pipeline's FFMPEG_BIN discovery)
        thumb_path = out_dir / "thumb_veo.jpg"
        try:
            import subprocess, sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline_core"))
            from pipeline import FFMPEG_BIN  # type: ignore
            subprocess.run(
                [FFMPEG_BIN, "-y", "-i", str(video_path),
                 "-vframes", "1", "-q:v", "2", str(thumb_path)],
                capture_output=True, check=True, timeout=30,
            )
        except Exception as e:
            print(f"[veo] thumb gen failed: {e}")

        # Create standard Job + Clip so the rest of the app works unchanged.
        # frame_layout=veo_generated signals "no compose step needed".
        job = models.Job(
            user_id=user_id,
            platform=platform,
            frame_layout="veo_generated",
            video_name=f"veo_topic_{topic_id}.mp4",
            language=language,
            status="done",
            log=f"[veo] topic #{topic_id}\nprompt: {prompt_cfg['prompt'][:500]}",
            output_dir=str(out_dir),
        )
        db.add(job); db.commit(); db.refresh(job)

        clip = models.Clip(
            job_id=job.id,
            clip_index=0,
            filename=video_path.name,
            file_path=str(video_path),
            thumb_path=str(thumb_path) if thumb_path.exists() else "",
            image_path="",
            duration=float(prompt_cfg["duration_seconds"]),
            frame_type="veo_generated",
            text=(topic.topic_summary or topic.video_title or "")[:300],
            sentiment="",
            entities=json.dumps(topic.keywords or []),
            card_params=json.dumps({}),
            section_pct=json.dumps({}),
            follow_params=json.dumps({}),
            meta=json.dumps({
                "source":     "veo",
                "topic_id":   topic_id,
                "veo_model":  meta.get("model", ""),
                "prompt":     prompt_cfg["prompt"],
                "aspect":     prompt_cfg["aspect_ratio"],
                "duration":   prompt_cfg["duration_seconds"],
                "language":   language,
                "platform":   platform,
                "original_title": topic.video_title,
            }),
        )
        db.add(clip); db.commit(); db.refresh(clip)

        # Mark the topic as used so the trending UI hides it
        topic.used_for_job_id = job.id
        db.commit()

        _set_status(topic_id, state="done", job_id=job.id, clip_id=clip.id)
    except Exception as e:
        traceback.print_exc()
        _set_status(topic_id, state="error", error=str(e)[:400])
    finally:
        db.close()


# ─── Endpoints ───────────────────────────────────────────────────────────

class GenFromTopic(BaseModel):
    platform: str = "youtube_short"
    language: str = "te"


@router.post("/generate-from-topic/{topic_id}")
def generate_from_topic(
    topic_id: int,
    payload: GenFromTopic,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Kick off Veo 3 video generation for one trending topic."""
    topic = (
        db.query(models.TrendingTopic)
          .join(models.CompetitorChannel,
                models.CompetitorChannel.id == models.TrendingTopic.source_channel_id)
          .filter(
              models.TrendingTopic.id == topic_id,
              models.CompetitorChannel.user_id == user.id,
          )
          .first()
    )
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")

    cur = _get_status(topic_id)
    if cur.get("state", "idle").startswith("generating"):
        return {"status": cur, "topic_id": topic_id}

    _set_status(topic_id, state="queued", job_id=None, error="")
    threading.Thread(
        target=_run_veo_job,
        args=(topic_id, user.id, payload.platform, payload.language),
        daemon=True,
    ).start()
    return {"status": "queued", "topic_id": topic_id}


@router.get("/status/{topic_id}")
def get_status(
    topic_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),  # noqa: auth gate
):
    return _get_status(topic_id)
