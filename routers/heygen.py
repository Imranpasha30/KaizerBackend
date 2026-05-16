"""HeyGen avatar video generation router — replaces Veo 3 in Trending.

Endpoints
---------
  GET  /api/heygen/avatars
  GET  /api/heygen/voices
  GET  /api/heygen/defaults
  PUT  /api/heygen/defaults
  POST /api/heygen/generate-from-topic/{topic_id}
  GET  /api/heygen/status/{topic_id}

Flow for ``generate-from-topic``:
  1. Verify topic belongs to current user's competitor channel.
  2. Spawn a background daemon thread that:
     - fetches transcript (captions, Whisper fallback)
     - compresses into a ~700-char avatar script via Gemini
     - submits to HeyGen v2/video/generate
     - polls v1/video_status.get every 6 s until completed / failed
     - downloads the MP4 + auto-thumbnail
     - creates Job + Clip rows so the asset flows through the
       existing SEO / Publish pipeline (identical to Veo's handoff)
  3. UI polls ``/status/{topic_id}`` for live state.
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
from heygen import client as heygen_client
from heygen import script_builder
from heygen import transcript as heygen_transcript


router = APIRouter(prefix="/api/heygen", tags=["heygen"])


# ── In-memory status tracker (matches Veo's pattern) ─────────────
# Per-topic-id keyed; UI polls /status/{topic_id} every few seconds.
# state values:
#   idle | queued | transcribing | scripting | heygen_pending
#   heygen_processing | downloading | done | error

_status: dict[int, dict] = {}
_lock   = threading.Lock()


def _set_status(topic_id: int, **kw) -> None:
    with _lock:
        cur = _status.get(topic_id) or {}
        cur.update(kw)
        cur["updated_at"] = time.time()
        _status[topic_id] = cur


def _get_status(topic_id: int) -> dict:
    with _lock:
        return dict(_status.get(topic_id) or {"state": "idle"})


# ── Background worker ─────────────────────────────────────────────

def _run_heygen_job(
    topic_id: int,
    user_id: int,
    platform: str,
    language: str,
    avatar_id: str,
    voice_id: str,
) -> None:
    """End-to-end HeyGen generation for one trending topic.
    Owns its own DB session — never touches the request-scoped one."""
    db = SessionLocal()
    try:
        # 1) Topic lookup with tenancy check.
        topic = (
            db.query(models.TrendingTopic)
              .join(models.CompetitorChannel,
                    models.CompetitorChannel.id == models.TrendingTopic.source_channel_id)
              .filter(
                  models.TrendingTopic.id == topic_id,
                  models.CompetitorChannel.user_id == user_id,
              )
              .first()
        )
        if not topic:
            _set_status(topic_id, state="error",
                        error="Topic not found or not yours")
            return

        if not topic.video_url:
            _set_status(topic_id, state="error",
                        error="Topic has no video_url to transcribe")
            return

        # 2) Transcript fetch.
        _set_status(topic_id, state="transcribing", progress=10,
                    message="Pulling video transcript…")
        try:
            tr = heygen_transcript.fetch_transcript(
                topic.video_url, language=language,
            )
        except heygen_transcript.TranscriptError as exc:
            _set_status(topic_id, state="error", error=str(exc)[:300])
            return

        # 3) Script compression.
        _set_status(topic_id, state="scripting", progress=30,
                    message="Building avatar script (Gemini)…")
        keywords = []
        try:
            if topic.keywords:
                if isinstance(topic.keywords, str):
                    keywords = json.loads(topic.keywords)
                elif isinstance(topic.keywords, list):
                    keywords = topic.keywords
        except (ValueError, TypeError):
            keywords = []
        script_pack = script_builder.build_script(
            topic_title    = topic.video_title or "",
            topic_summary  = topic.topic_summary or "",
            topic_keywords = keywords,
            transcript     = tr["transcript"],
            transcript_source = tr.get("source", "captions"),
            language       = language or tr.get("language") or "te",
        )
        script_text = script_pack["script"]
        if not script_text:
            _set_status(topic_id, state="error",
                        error="Script builder produced empty output")
            return

        # 4) Submit to HeyGen.
        _set_status(topic_id, state="heygen_pending", progress=45,
                    message="Submitted to HeyGen — waiting for render…")
        try:
            video_id = heygen_client.generate_video(
                avatar_id = avatar_id,
                voice_id  = voice_id,
                script    = script_text,
                width     = 1080 if platform.endswith("short") or platform.endswith("reel") else 1920,
                height    = 1920 if platform.endswith("short") or platform.endswith("reel") else 1080,
            )
        except heygen_client.HeyGenError as exc:
            _set_status(topic_id, state="error", error=str(exc)[:400])
            return

        # 5) Poll until done / failed. 6 s interval, 15 min cap.
        _set_status(topic_id, state="heygen_processing", progress=55,
                    message="HeyGen rendering avatar…",
                    heygen_video_id=video_id)
        deadline = time.time() + 15 * 60
        last_status: Optional[dict] = None
        while time.time() < deadline:
            try:
                last_status = heygen_client.get_status(video_id=video_id)
            except heygen_client.HeyGenError as exc:
                _set_status(topic_id, state="error",
                            error=f"poll failed: {exc}"[:400])
                return
            st = last_status.get("status")
            if st == "completed":
                break
            if st == "failed":
                err = last_status.get("error") or {}
                msg = err.get("message") if isinstance(err, dict) else str(err)
                _set_status(topic_id, state="error",
                            error=f"HeyGen failed: {msg or 'unknown'}"[:400])
                return
            # Linear progress between 55 and 88 over the wait window.
            elapsed = 15 * 60 - max(0, deadline - time.time())
            pct = min(88, 55 + int((elapsed / (15 * 60)) * 33))
            _set_status(topic_id, state="heygen_processing", progress=pct,
                        message=f"HeyGen status: {st}")
            time.sleep(6)
        else:
            _set_status(topic_id, state="error",
                        error="HeyGen render exceeded 15 min timeout")
            return

        if not last_status or not last_status.get("video_url"):
            _set_status(topic_id, state="error",
                        error="HeyGen returned no video_url on completed")
            return

        # 6) Download MP4 + thumbnail.
        from runner import OUTPUT_ROOT
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(OUTPUT_ROOT) / "heygen" / f"topic_{topic_id}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        video_path = out_dir / "clip_heygen.mp4"
        thumb_path = out_dir / "thumb_heygen.jpg"

        _set_status(topic_id, state="downloading", progress=90,
                    message="Downloading rendered video…")
        try:
            heygen_client.download_to_file(
                last_status["video_url"], str(video_path),
            )
        except heygen_client.HeyGenError as exc:
            _set_status(topic_id, state="error",
                        error=f"download failed: {exc}"[:400])
            return

        # Thumbnail — frame 0 via ffmpeg (mirror Veo's helper).
        try:
            import subprocess, sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline_core"))
            from pipeline import FFMPEG_BIN  # type: ignore
            subprocess.run(
                [FFMPEG_BIN, "-y", "-i", str(video_path),
                 "-vframes", "1", "-q:v", "2", str(thumb_path)],
                capture_output=True, check=True, timeout=30,
            )
        except Exception as exc:
            print(f"[heygen] thumb gen failed: {exc}")

        # 7) Create Job + Clip rows so the asset flows through the
        # existing editor / SEO / publish pipeline. Identical shape
        # to Veo's handoff (lines 106-145 of routers/veo.py).
        duration_s = float(last_status.get("duration") or tr.get("duration_s") or 0.0)
        job = models.Job(
            user_id      = user_id,
            platform     = platform,
            frame_layout = "heygen_generated",
            video_name   = f"heygen_topic_{topic_id}.mp4",
            language     = language,
            status       = "done",
            log          = (
                f"[heygen] topic #{topic_id}\n"
                f"transcript_source={tr.get('source')}, "
                f"script_source={script_pack.get('source')}, "
                f"avatar={avatar_id}, voice={voice_id}\n"
                f"script[:300]: {script_text[:300]}"
            ),
            output_dir   = str(out_dir),
        )
        db.add(job); db.commit(); db.refresh(job)

        clip = models.Clip(
            job_id     = job.id,
            clip_index = 0,
            filename   = video_path.name,
            file_path  = str(video_path),
            thumb_path = str(thumb_path) if thumb_path.exists() else "",
            image_path = "",
            duration   = duration_s,
            frame_type = "heygen_generated",
            text       = (topic.topic_summary or topic.video_title or "")[:300],
            sentiment  = "",
            entities   = json.dumps(keywords),
            card_params   = json.dumps({}),
            section_pct   = json.dumps({}),
            follow_params = json.dumps({}),
            meta = json.dumps({
                "source":             "heygen",
                "topic_id":           topic_id,
                "heygen_video_id":    video_id,
                "avatar_id":          avatar_id,
                "voice_id":           voice_id,
                "script":             script_text,
                "script_source":      script_pack.get("source"),
                "script_model":       script_pack.get("model"),
                "transcript_source":  tr.get("source"),
                "transcript_chars":   len(tr.get("transcript") or ""),
                "language":           language,
                "platform":           platform,
                "original_title":     topic.video_title,
            }),
        )
        db.add(clip); db.commit(); db.refresh(clip)

        # Mark the topic as used so the trending UI hides it.
        topic.used_for_job_id = job.id
        db.commit()

        _set_status(topic_id, state="done", progress=100,
                    message="Done — clip ready in editor",
                    job_id=job.id, clip_id=clip.id)
    except Exception as exc:
        traceback.print_exc()
        _set_status(topic_id, state="error", error=str(exc)[:400])
    finally:
        db.close()


# ── Avatar / voice listing (proxies to HeyGen) ────────────────────

@router.get("/avatars")
def list_avatars(user: models.User = Depends(auth.current_user)) -> dict:
    """Return the user's available HeyGen avatars. Backed by the
    server-side ``HEYGEN_API_KEY`` — UI never sees the key."""
    try:
        return heygen_client.list_avatars()
    except heygen_client.HeyGenAuthError as exc:
        raise HTTPException(401, f"HeyGen auth: {exc}")
    except heygen_client.HeyGenError as exc:
        raise HTTPException(502, f"HeyGen error: {exc}")


@router.get("/voices")
def list_voices(user: models.User = Depends(auth.current_user)) -> dict:
    """Return the HeyGen voice library. UI filters by language."""
    try:
        voices = heygen_client.list_voices()
    except heygen_client.HeyGenAuthError as exc:
        raise HTTPException(401, f"HeyGen auth: {exc}")
    except heygen_client.HeyGenError as exc:
        raise HTTPException(502, f"HeyGen error: {exc}")
    return {"voices": voices, "count": len(voices)}


# ── User-level defaults (avatar + voice) ──────────────────────────

class HeyGenDefaults(BaseModel):
    avatar_id: Optional[str] = None
    voice_id:  Optional[str] = None


@router.get("/defaults")
def get_defaults(
    db:   Session       = Depends(get_db),
    user: models.User   = Depends(auth.current_user),
) -> dict:
    """Return the user's saved default avatar + voice. Empty strings
    when unset (frontend falls back to picker UI). Also returns the
    server-wide ``HEYGEN_DEFAULT_*`` env values as a secondary
    fallback so a brand-new user still gets a sensible default."""
    import os
    return {
        "avatar_id": getattr(user, "heygen_avatar_id", None) or "",
        "voice_id":  getattr(user, "heygen_voice_id",  None) or "",
        "env_default_avatar_id": os.environ.get("HEYGEN_DEFAULT_AVATAR_ID", "").strip(),
        "env_default_voice_id":  os.environ.get("HEYGEN_DEFAULT_VOICE_ID", "").strip(),
    }


@router.put("/defaults")
def put_defaults(
    payload: HeyGenDefaults,
    db:      Session     = Depends(get_db),
    user:    models.User = Depends(auth.current_user),
) -> dict:
    """Save the user's preferred avatar + voice. Either field can be
    None / empty — sending an empty string clears that preference."""
    if hasattr(user, "heygen_avatar_id"):
        user.heygen_avatar_id = (payload.avatar_id or "").strip()[:64] or None
    if hasattr(user, "heygen_voice_id"):
        user.heygen_voice_id  = (payload.voice_id  or "").strip()[:64] or None
    db.add(user); db.commit(); db.refresh(user)
    return {
        "ok":         True,
        "avatar_id":  getattr(user, "heygen_avatar_id", None) or "",
        "voice_id":   getattr(user, "heygen_voice_id",  None) or "",
    }


# ── Kick generation ──────────────────────────────────────────────

class GenFromTopic(BaseModel):
    platform:  str = "youtube_short"   # youtube_short | youtube_full | instagram_reel
    language:  str = "te"
    avatar_id: str
    voice_id:  str


@router.post("/generate-from-topic/{topic_id}")
def generate_from_topic(
    topic_id: int,
    payload:  GenFromTopic,
    db:       Session     = Depends(get_db),
    user:     models.User = Depends(auth.current_user),
) -> dict:
    """Kick off a HeyGen avatar generation for one trending topic.
    Returns immediately; UI polls /status/{topic_id}."""
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
        raise HTTPException(404, "Topic not found")

    if not payload.avatar_id.strip():
        raise HTTPException(400, "avatar_id required")
    if not payload.voice_id.strip():
        raise HTTPException(400, "voice_id required")

    cur = _get_status(topic_id)
    if cur.get("state") in ("queued", "transcribing", "scripting",
                            "heygen_pending", "heygen_processing",
                            "downloading"):
        return {"status": "already_running", "current": cur, "topic_id": topic_id}

    _set_status(topic_id, state="queued", progress=0,
                message="Queued", job_id=None, clip_id=None, error="")
    threading.Thread(
        target=_run_heygen_job,
        args=(topic_id, user.id, payload.platform, payload.language,
              payload.avatar_id.strip(), payload.voice_id.strip()),
        daemon=True,
    ).start()
    return {"status": "queued", "topic_id": topic_id}


@router.get("/status/{topic_id}")
def get_status(
    topic_id: int,
    user: models.User = Depends(auth.current_user),  # auth gate
) -> dict:
    """Return the current state of a HeyGen generation job."""
    return _get_status(topic_id)
