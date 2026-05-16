"""Live Studio router — bulk RTMP-live publishing.

Endpoints
---------
  POST   /api/live-studio/batches
       → create a new batch. Body: list of videos, each with picked
         channels + duration + per-channel SEO. Returns the LiveBatch
         id + a list of LiveStream ids (the per-(video×channel) units
         the browser will then upload chunks for).

  POST   /api/live-studio/streams/{id}/chunk   (raw bytes)
       → upload one chunk. Header: Content-Range: bytes <s>-<e>/<t>.
         Server appends to the growing temp file.

  POST   /api/live-studio/streams/{id}/start
       → mark upload complete (or "good enough" threshold reached).
         Triggers the background ffmpeg push.

  POST   /api/live-studio/streams/{id}/cancel
       → kill a running stream cleanly.

  GET    /api/live-studio/batches
  GET    /api/live-studio/batches/{id}
  GET    /api/live-studio/streams/{id}
       → status reads.

  POST   /api/live-studio/seo/validate
       → run AI-generated SEO through the strict validator before the
         user clicks "create batch". Returns 200 with the sanitized
         payload on success, 400 on failure.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
from datetime import datetime, timezone
from typing import Optional

from fastapi import (
    APIRouter, Depends, Header, HTTPException, Request, status,
)
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

import auth
import models
from database import SessionLocal, get_db
from live_studio import concurrency
from live_studio import orchestrator as live_orch
from live_studio import seo as live_seo
from live_studio import uploads as live_uploads


router = APIRouter(prefix="/api/live-studio", tags=["live-studio"])


# ─── Pydantic request models ─────────────────────────────────────

class BatchVideoIn(BaseModel):
    """One video in the batch. The browser uploads this file via
    chunks AFTER batch creation (we need the stream id back first)."""
    filename:      str = Field(..., min_length=1, max_length=255)
    size_bytes:    int = Field(0, ge=0)
    duration_hours:float = Field(1.0, gt=0, le=24)
    channel_ids:   list[int] = Field(..., min_length=1)
    seo_source:    str = Field("user", pattern="^(user|ai)$")
    seo:           live_seo.LiveSeoIn

    @field_validator("filename")
    @classmethod
    def _safe_filename(cls, v: str) -> str:
        # Drop directory parts; we use the slot's stream_id as the
        # canonical filename so collisions are impossible.
        return os.path.basename(v.strip())


class BatchCreateIn(BaseModel):
    videos: list[BatchVideoIn] = Field(..., min_length=1, max_length=50)


class BatchStreamOut(BaseModel):
    id:           int
    video_slot:   int
    channel_id:   int
    status:       str
    progress_pct: int
    title:        str
    duration_hours: float


class BatchOut(BaseModel):
    id:           int
    public_id:    str
    status:       str
    message:      Optional[str]
    total_streams: int
    streams_done: int
    streams_failed: int
    streams:      list[BatchStreamOut]
    created_at:   Optional[datetime]


# ─── Helpers ─────────────────────────────────────────────────────

def _batch_to_out(batch: models.LiveBatch, streams: list[models.LiveStream]) -> BatchOut:
    return BatchOut(
        id=batch.id,
        public_id=batch.public_id,
        status=batch.status,
        message=batch.message,
        total_streams=batch.total_streams,
        streams_done=batch.streams_done,
        streams_failed=batch.streams_failed,
        streams=[
            BatchStreamOut(
                id=s.id, video_slot=s.video_slot,
                channel_id=s.channel_id or 0,
                status=s.status, progress_pct=s.progress_pct,
                title=s.title or "",
                duration_hours=float(s.target_hours or 0),
            ) for s in streams
        ],
        created_at=batch.created_at,
    )


def _owned_stream(db: Session, stream_id: int, user_id: int) -> models.LiveStream:
    """Fetch + enforce tenancy. 404 (not 403) on foreign rows so an
    attacker can't enumerate stream ids."""
    row = db.query(models.LiveStream).filter(
        models.LiveStream.id == stream_id,
        models.LiveStream.user_id == user_id,
    ).first()
    if not row:
        raise HTTPException(404, "stream not found")
    return row


# ─── Endpoints ────────────────────────────────────────────────────

@router.get("/channels")
def list_oauth_channels(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Return the **real YouTube accounts** the user has connected
    via OAuth (i.e. each row on the "My accounts" tab — Cyber Sphere,
    Auto Wala, Kaizer 15, …). One row per ``OAuthToken``.

    Note: the ``id`` field is still ``Channel.id`` (the style profile
    that owns the OAuth, 1:1 with OAuthToken). The orchestrator uses
    that to call ``youtube.oauth.get_credentials(db, channel_id)``.
    The user-facing name + handle come from the OAuthToken's cached
    YouTube metadata so the UI shows the actual YT channel branding
    (not the internal style-profile label).
    """
    tokens = (
        db.query(models.OAuthToken)
          .join(models.Channel, models.Channel.id == models.OAuthToken.channel_id)
          .filter(models.Channel.user_id == user.id)
          .order_by(models.OAuthToken.connected_at.desc())
          .all()
    )
    # Dedupe by ``google_channel_id`` — when a single YT account has
    # been re-connected through multiple style profiles (which happens
    # if the user picked the same Brand row twice in Google's OAuth
    # picker), we get N rows for the same actual YouTube channel.
    # Keep the most-recently-connected token's channel_id.
    seen_gcid: set[str] = set()
    out = []
    for tok in tokens:
        if not (tok.refresh_token_enc or "").strip():
            continue
        gcid = (tok.google_channel_id or "").strip()
        # Empty google_channel_id is rare but possible on a stale
        # token — still show but key by token id so we don't merge
        # un-keyed rows together.
        dedupe_key = gcid or f"_no_gcid_{tok.id}"
        if dedupe_key in seen_gcid:
            continue
        seen_gcid.add(dedupe_key)
        out.append({
            # Channel.id — what the orchestrator needs internally.
            "id":            tok.channel_id,
            # Real YT channel name (e.g. "Cyber Sphere").
            "name":          tok.google_channel_title or f"Channel #{tok.channel_id}",
            # @handle from YouTube — display only.
            "handle":        tok.channel_custom_url or "",
            "youtube_channel_id": gcid,
            "avatar_url":    tok.channel_thumbnail_url or "",
            "subscriber_count": int(tok.subscriber_count or 0),
            "video_count":   int(tok.video_count or 0),
            # Connected_at lets the UI sort newest-first.
            "connected_at":  tok.connected_at,
        })
    return {"channels": out, "count": len(out)}


class SeoGenerateIn(BaseModel):
    """Generate SEO from a short user-typed brief. The Live Studio
    use case is "user uploaded a finished video and wants Kaizer to
    write title/description/tags for it" — no transcript, no
    pipeline, just a one-paragraph brief."""
    brief:    str = Field(..., min_length=4, max_length=2000)
    channel_id: Optional[int] = None
    language: str = Field("te", min_length=2, max_length=8)
    privacy:  str = Field("unlisted")


@router.post("/seo/generate")
def generate_seo(
    payload: SeoGenerateIn,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Generate SEO using Kaizer's **editor SEO machinery** — same
    Gemini model chain, same system-prompt template, same writing
    voice that the editor's "Generate SEO" button uses. The only
    difference is the user-prompt: we feed a brief (Live Studio
    doesn't have a Clip/transcript), not a topic extracted from a
    Clip row.

    Output still goes through the strict AI-path validator (YouTube
    hard caps) before returning, so the result is brand-safe.
    """
    import os
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        raise HTTPException(503, "GEMINI_API_KEY not configured — fill SEO manually")

    # Style-source channel = the user's style profile bound to this
    # YT account. Gives Gemini the title-formula + desc-style
    # references the editor uses.
    style_source = None
    if payload.channel_id:
        style_source = db.query(models.Channel).filter(
            models.Channel.id == payload.channel_id,
            models.Channel.user_id == user.id,
        ).first()

    # Reuse the editor's full machinery: system-prompt builder,
    # Gemini call chain, sanitizer, AND verifier-driven retry loop.
    # Same TARGET_SCORE (95) + same MAX_RETRIES (4 retries past
    # the first attempt). Result is identical quality to the
    # editor's button.
    try:
        from seo import prompts as seo_prompts
        from seo import generator as seo_gen
        from seo import sanitizer as seo_sanitizer
        from seo import verifier as seo_verifier
    except ImportError as exc:
        raise HTTPException(503, f"Kaizer SEO module unavailable: {exc}")

    try:
        system_prompt = seo_prompts.build_system_prompt(
            language=payload.language or "te",
            style_source=style_source,
            target_score=seo_gen.TARGET_SCORE,
        )
    except Exception as exc:
        raise HTTPException(500, f"system prompt build failed: {exc}")

    brief_text = payload.brief.strip()[:2000]

    def _build_user_prompt(retry_feedback: list[str] | None) -> str:
        """Live Studio's user prompt — analogous to
        ``seo.prompts.build_user_prompt(clip=...)`` but with the
        operator's brief in place of a Clip's transcript.

        The retry-feedback block tells Gemini exactly which scoring
        rules it failed on the previous attempt so the next attempt
        targets the right fixes.
        """
        parts = [
            "Generate a single JSON object with keys "
            "{title, description, tags, hashtags} for the topic below. "
            "There is no transcript or upstream metadata — only the "
            "operator's brief. Treat the brief as the entire context. "
            "Score target: ≥{} (verifier-graded).".format(seo_gen.TARGET_SCORE),
            "",
            "TOPIC / BRIEF:",
            f'"""\n{brief_text}\n"""',
        ]
        if retry_feedback:
            parts += [
                "",
                "RETRY FEEDBACK — your previous attempt failed these "
                "specific scoring rules. Fix EACH ONE in this attempt:",
            ]
            for r in retry_feedback[:8]:
                parts.append(f"  - {r}")
        return "\n".join(parts)

    # Same retry loop as ``seo.generator.generate_seo_for_clip``.
    best: dict | None = None
    best_score = -1
    best_report: dict | None = None
    model_used = seo_gen.GEMINI_MODEL
    attempts_log: list[dict] = []
    retry_feedback: list[str] = []
    total_rounds = seo_gen.MAX_RETRIES + 1

    for attempt in range(1, total_rounds + 1):
        user_prompt = _build_user_prompt(retry_feedback if attempt > 1 else None)
        try:
            raw, model_used = seo_gen._call_gemini(
                system_prompt, user_prompt,
                db=db, user_id=user.id, job_id=None, clip_id=None,
            )
        except seo_gen.SEOGenerationError as exc:
            if best:
                # All models exhausted but we have a candidate — ship it.
                print(f"[live-studio/seo] attempt {attempt}: models exhausted; "
                      f"keeping best score={best_score}")
                break
            raise HTTPException(502, f"Gemini SEO failed: {exc}")

        if not isinstance(raw, dict):
            print(f"[live-studio/seo] attempt {attempt}: non-dict response, skip")
            continue

        # Mechanical cleanups + sanitize, identical to editor path.
        raw["title"] = seo_gen._strip_trailing_suffix(raw.get("title", ""))
        try:
            cleaned = seo_sanitizer.sanitize(raw, style_source)
        except Exception as exc:
            print(f"[live-studio/seo] attempt {attempt}: sanitize crashed: {exc}")
            continue

        # Verifier scores against the brief as the topic. No
        # trend_keywords / news_items since Live Studio doesn't run
        # the research phase — that's fine, verifier handles empty.
        report = seo_verifier.verify(
            cleaned, clip_topic=brief_text,
            trend_keywords=[], news_items=[],
        )
        attempts_log.append({
            "attempt": attempt,
            "score":   int(report.get("score") or 0),
            "reasons": (report.get("reasons") or [])[:6],
            "model":   model_used,
        })
        print(f"[live-studio/seo] attempt {attempt}: score={report.get('score')} "
              f"fails={len(report.get('reasons') or [])}")

        if (report.get("score") or 0) > best_score:
            best_score  = int(report.get("score") or 0)
            best        = cleaned
            best_report = report

        if best_score >= seo_gen.TARGET_SCORE:
            break

        retry_feedback = report.get("reasons") or []

    if not best:
        raise HTTPException(502, "SEO generation produced no usable candidates")

    # Strict-validate the final result against YouTube hard caps.
    try:
        validated = live_seo.validate_ai_path(live_seo.LiveSeoIn(
            title       = (best.get("title") or "").strip(),
            description = (best.get("description") or "").strip(),
            tags        = list(best.get("tags") or best.get("keywords") or []),
            privacy     = payload.privacy or "unlisted",
            made_for_kids = False,
        ))
    except ValueError as exc:
        raise HTTPException(
            422,
            f"Editor SEO best-of-{len(attempts_log)} attempt(s) failed "
            f"strict validator: {exc}. Edit manually or retry generation."
        )

    return {
        "ok":       True,
        "seo":      validated.model_dump(),
        "raw":      best,
        "model":    model_used,
        "source":   "kaizer_editor_seo_retry",
        "attempts": len(attempts_log),
        "best_score":   best_score,
        "target_score": seo_gen.TARGET_SCORE,
        "attempts_log": attempts_log,
        "reasons":  (best_report or {}).get("reasons", [])[:6],
    }


@router.post("/seo/validate")
def validate_seo(
    payload: live_seo.LiveSeoIn,
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Run the STRICT (AI-path) validator. Returns the sanitized
    payload on success, 400 with a clear message on failure.

    Frontend calls this after invoking the AI SEO generator (to
    confirm the AI output is brand-safe) BEFORE the user clicks
    "create batch". Lets us reject and re-roll the AI before any
    DB write happens.
    """
    try:
        cleaned = live_seo.validate_ai_path(payload)
    except ValueError as exc:
        raise HTTPException(400, f"SEO validation failed: {exc}")
    return {"ok": True, "seo": cleaned.model_dump()}


@router.post("/batches")
def create_batch(
    payload: BatchCreateIn,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> BatchOut:
    """Create a LiveBatch + N*M LiveStream rows. Returns the batch
    + the per-stream ids the browser will use to push chunks.

    Each video × channel produces one stream row. SEO is sanitized
    or validated per the per-video ``seo_source`` flag.
    """
    # Validate channel ids belong to the current user.
    all_channel_ids = sorted({c for v in payload.videos for c in v.channel_ids})
    if not all_channel_ids:
        raise HTTPException(400, "no channels picked")
    owned = {
        c.id: c for c in db.query(models.Channel).filter(
            models.Channel.id.in_(all_channel_ids),
            models.Channel.user_id == user.id,
        ).all()
    }
    missing = set(all_channel_ids) - set(owned.keys())
    if missing:
        raise HTTPException(403, f"channels not yours: {sorted(missing)}")

    # Process each video's SEO once (cheaper than per-channel).
    per_video_seo: list[live_seo.LiveSeoOut] = []
    for vi, vid in enumerate(payload.videos):
        try:
            if vid.seo_source == "ai":
                per_video_seo.append(live_seo.validate_ai_path(vid.seo))
            else:
                per_video_seo.append(live_seo.sanitize_for_user_path(vid.seo))
        except ValueError as exc:
            raise HTTPException(400, f"video[{vi}] SEO invalid: {exc}")

    # Build the rows. One LiveStream per (video, channel) pair.
    batch = models.LiveBatch(
        user_id=user.id,
        public_id=secrets.token_urlsafe(8),
        status="queued",
        total_streams=sum(len(v.channel_ids) for v in payload.videos),
    )
    db.add(batch); db.commit(); db.refresh(batch)

    streams: list[models.LiveStream] = []
    for vi, vid in enumerate(payload.videos):
        cleaned = per_video_seo[vi]
        for cid in vid.channel_ids:
            row = models.LiveStream(
                batch_id=batch.id,
                user_id=user.id,
                channel_id=cid,
                video_slot=vi,
                status="queued",
                target_hours=vid.duration_hours,
                upload_total=vid.size_bytes or None,
                seo_source=vid.seo_source,
                title=cleaned.title,
                description=cleaned.description,
                tags_json=json.dumps(cleaned.tags, ensure_ascii=False),
                privacy=cleaned.privacy,
                made_for_kids=cleaned.made_for_kids,
            )
            db.add(row); db.commit(); db.refresh(row)
            # The upload path needs the row id so we set it now.
            row.upload_path = live_uploads.upload_path_for(row.id)
            db.commit()
            streams.append(row)

    return _batch_to_out(batch, streams)


@router.post("/streams/{stream_id}/chunk")
async def upload_chunk(
    stream_id: int,
    request: Request,
    content_range: Optional[str] = Header(default=None, alias="Content-Range"),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Append one chunk to the growing temp file for ``stream_id``.

    Headers required:
      Content-Range: bytes <start>-<end>/<total>
    Body: raw bytes of the chunk (no multipart wrapper — keeps it
          fast and removes a memory copy).
    """
    row = _owned_stream(db, stream_id, user.id)
    if row.upload_done:
        raise HTTPException(409, "upload already complete")

    parsed = live_uploads.parse_content_range(content_range)
    if not parsed:
        raise HTTPException(400, "missing or malformed Content-Range header")
    start, end, total = parsed

    body = await request.body()
    try:
        result = live_uploads.write_chunk(
            stream_id=stream_id, chunk=body,
            start=start, end=end, total=total,
        )
    except (ValueError, OSError) as exc:
        raise HTTPException(400, f"chunk write failed: {exc}")

    # Update the DB row.
    row.upload_bytes = int(result["bytes_written"])
    if total is not None:
        row.upload_total = total
    if result["is_complete"]:
        row.upload_done = True
        if row.status == "queued":
            row.status = "uploaded"
            row.message = "upload complete, awaiting broadcast slot"
    elif row.status == "queued":
        row.status = "uploading"
    db.commit()

    return {
        "bytes_written": row.upload_bytes,
        "total":         row.upload_total,
        "is_complete":   bool(row.upload_done),
        "status":        row.status,
    }


@router.post("/streams/{stream_id}/start")
def start_stream(
    stream_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Tell the orchestrator to kick off the ffmpeg push for this
    stream. The orchestrator is implemented in a follow-up commit —
    for now this endpoint marks the row ``streaming`` and the actual
    ffmpeg launch is wired up next phase.
    """
    row = _owned_stream(db, stream_id, user.id)
    if not row.upload_done and row.upload_bytes < live_uploads.CHUNK_THRESHOLD_BYTES:
        raise HTTPException(
            409,
            f"not enough video buffered yet "
            f"({row.upload_bytes}/{live_uploads.CHUNK_THRESHOLD_BYTES} bytes)",
        )
    if row.status in ("streaming", "done", "starting", "provisioning"):
        return {"ok": True, "status": row.status, "note": "already started"}

    row.status = "starting"
    row.message = "spawning broadcast worker…"
    row.started_at = datetime.now(timezone.utc)
    db.commit()

    # Spawn the daemon worker. It owns its own DB session — we don't
    # block the request thread.
    live_orch.kick_off(row.id)
    return {"ok": True, "status": row.status}


@router.post("/streams/{stream_id}/cancel")
def cancel_stream(
    stream_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    row = _owned_stream(db, stream_id, user.id)
    if row.status in ("done", "failed", "canceled"):
        return {"ok": True, "status": row.status, "note": "terminal state"}
    row.status = "canceled"
    row.message = "canceled by user"
    row.finished_at = datetime.now(timezone.utc)
    db.commit()
    # Signal the running ffmpeg (if any) to exit cleanly. The worker
    # will also cleanup the temp file in its finally block. If no
    # worker is running yet (cancel before /start), do the cleanup
    # ourselves so the disk doesn't leak.
    if not live_orch.request_cancel(stream_id):
        live_uploads.delete_upload(stream_id)
    return {"ok": True, "status": row.status}


@router.get("/batches")
def list_batches(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
    limit: int = 50,
) -> dict:
    rows = (
        db.query(models.LiveBatch)
          .filter(models.LiveBatch.user_id == user.id)
          .order_by(models.LiveBatch.created_at.desc())
          .limit(min(max(1, limit), 200))
          .all()
    )
    return {
        "batches": [
            {
                "id":           b.id,
                "public_id":    b.public_id,
                "status":       b.status,
                "message":      b.message,
                "total":        b.total_streams,
                "done":         b.streams_done,
                "failed":       b.streams_failed,
                "created_at":   b.created_at,
            } for b in rows
        ],
        "count": len(rows),
    }


@router.get("/batches/{batch_id}")
def get_batch(
    batch_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> BatchOut:
    batch = db.query(models.LiveBatch).filter(
        models.LiveBatch.id == batch_id,
        models.LiveBatch.user_id == user.id,
    ).first()
    if not batch:
        raise HTTPException(404, "batch not found")
    streams = (
        db.query(models.LiveStream)
          .filter(models.LiveStream.batch_id == batch.id)
          .order_by(models.LiveStream.video_slot.asc(),
                    models.LiveStream.id.asc())
          .all()
    )
    return _batch_to_out(batch, streams)


@router.get("/streams/{stream_id}")
def get_stream(
    stream_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    row = _owned_stream(db, stream_id, user.id)
    return {
        "id":             row.id,
        "batch_id":       row.batch_id,
        "video_slot":     row.video_slot,
        "channel_id":     row.channel_id,
        "status":         row.status,
        "progress_pct":   row.progress_pct,
        "message":        row.message,
        "error":          row.error,
        "upload_bytes":   row.upload_bytes,
        "upload_total":   row.upload_total,
        "upload_done":    row.upload_done,
        "target_hours":   row.target_hours,
        "started_at":     row.started_at,
        "finished_at":    row.finished_at,
        "title":          row.title,
        "description":    row.description,
        "tags":           json.loads(row.tags_json) if row.tags_json else [],
        "privacy":        row.privacy,
        "yt_video_id":    row.yt_video_id,
        "backup_url":     row.backup_url,
    }


@router.get("/health")
def health(user: models.User = Depends(auth.current_user)) -> dict:
    """Capacity check — exposes how many concurrent slots are still
    free so the UI can warn before the user submits a batch that
    would queue."""
    return {
        "ok": True,
        "concurrency": concurrency.stats(),
        "upload_threshold_bytes": live_uploads.CHUNK_THRESHOLD_BYTES,
    }
