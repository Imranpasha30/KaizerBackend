"""Quick Publish — already-edited video → SEO → thumbnail → fanout.

Wires the /quick-publish page end-to-end for users who DON'T need the
editing pipeline: the video arrived finished via ``POST
/api/clips/raw-upload/`` (Job status=done + Clip + storage mirror), and
publishing rides the existing v2 fanout (per-channel logo + watermark
via the Branding worker — the raw-upload MasterVideo is synthesised
with ``clean_master=True`` in routers/youtube_upload.py — and
per-channel socials/footer via the SEO composer).

This router only adds the three glue surfaces, all REUSE:

* quick-seo      — three modes, persisted into ``Clip.seo`` (the exact
                   JSON shape the publish composer already consumes):
                     manual      0 AI calls (user-typed)
                     description 1 Gemini call on ≤2k chars
                     transcript  1 Deepgram pass + 1 Gemini call on
                                 ≤7k transcript chars (cached on the
                                 clip so thumbnails never re-pay it)
* quick-thumbnail/upload | /generate — stores the image in object
                   storage and stamps ``clip.meta.publish_thumbnail_key``
                   which the legacy→v2 redirect forwards as
                   thumbnail_source='user_uploaded' (dispatch already
                   downloads it and runs thumbnails.set).
* quick-state    — wizard resume after a page refresh.

Token-budget rules (user requirement): single-shot AI calls only, hard
input caps, no retries, transcript cached.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import auth
import models
from database import get_db

log = logging.getLogger("kaizer.quick_publish")

router = APIRouter(prefix="/api", tags=["quick-publish"])

_TRANSCRIPT_CAP_CHARS = 7_000   # SEO quality plateaus well before this
_DESCRIPTION_CAP_CHARS = 2_000
_TRANSCRIPT_CACHE_CHARS = 4_000  # stored on clip.meta for thumbnail reuse
_THUMB_MAX_BYTES = 4 * 1024 * 1024


# ─── Helpers ─────────────────────────────────────────────────────────

def _seo_belt(job_id, user_id, stage: str, status: str) -> None:
    """Emit a SEO-lane transition for the admin Pipeline Flow belt so Quick
    Publish SEO shows on the conveyor alongside the editor's. Stages:
    'generate' (base SEO) | 'per_channel'. Best-effort — telemetry must NEVER
    break publishing."""
    if not job_id:
        return
    try:
        from services import stage_events as _se
        env = _se.Envelope(tenant_id=user_id, user_id=user_id, job_id=int(job_id),
                           clip_id=None, channel_id=None, label=f"quick job {job_id}")
        _se.emit_lane(env, "seo", stage, status)
    except Exception:
        pass


def _owned_clip(db: Session, user: models.User, clip_id: int) -> models.Clip:
    clip = db.query(models.Clip).filter(models.Clip.id == int(clip_id)).first()
    if clip is None:
        raise HTTPException(status_code=404, detail="Clip not found")
    job = db.query(models.Job).filter(models.Job.id == clip.job_id).first()
    if job is not None and job.user_id is not None:
        if int(job.user_id) != int(user.id) and not bool(user.is_admin):
            raise HTTPException(status_code=403, detail="Not your clip")
    return clip


def _meta(clip: models.Clip) -> dict:
    try:
        m = json.loads(clip.meta) if clip.meta else {}
        return m if isinstance(m, dict) else {}
    except Exception:
        return {}


def _save_meta(db: Session, clip: models.Clip, meta: dict) -> None:
    clip.meta = json.dumps(meta, ensure_ascii=False)
    db.add(clip)


def _seo_kind(clip: models.Clip) -> str:
    """'short' vs 'bulletin' for the SEO prompt — Shorts get the
    #Shorts-aware treatment."""
    platform = str(_meta(clip).get("platform") or "")
    return "short" if platform == "youtube_short" else "bulletin"


def _materialize_video(clip: models.Clip, tmp_dir: str) -> str:
    """Local path to the clip's video: the on-disk file when present
    (STORAGE_BACKEND=local keeps it), else download via the storage
    provider (R2 deployments delete the local copy after mirroring)."""
    fp = (clip.file_path or "").strip()
    if fp and os.path.isfile(fp):
        return fp
    key = (getattr(clip, "storage_key", "") or "").strip()
    if not key:
        raise HTTPException(
            status_code=422,
            detail="Video unavailable: no local file and no storage key. "
                   "Re-upload the video.",
        )
    from pipeline_core.storage import get_storage_provider
    local = os.path.join(tmp_dir, "video.mp4")
    try:
        get_storage_provider().download(key, local)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not fetch the video from storage: {exc}",
        )
    if not os.path.isfile(local) or os.path.getsize(local) == 0:
        raise HTTPException(
            status_code=502, detail="Storage returned an empty video file",
        )
    return local


def _persist_seo(db: Session, clip: models.Clip, seo: dict) -> dict:
    clip.seo = json.dumps(seo, ensure_ascii=False)
    db.add(clip)
    db.commit()
    return seo


def _norm_hashtags(tags: List[str]) -> List[str]:
    out = []
    for t in tags or []:
        t = str(t or "").strip()
        if not t:
            continue
        out.append(t if t.startswith("#") else f"#{t}")
    return out[:12]


# ─── quick-seo ───────────────────────────────────────────────────────


class QuickSeoRequest(BaseModel):
    mode: str = Field(..., pattern="^(manual|description|transcript)$")
    language: str = "te"
    # manual mode fields (description doubles as the prompt text in
    # description mode):
    title: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    hashtags: List[str] = Field(default_factory=list)
    # Optional competitor style reference (a kind:styles Channel from SEO
    # Settings). When set, AI modes write the title/description in that
    # channel's voice instead of the default. Ignored in manual mode.
    style_source_id: Optional[int] = None
    # Run the improve-to-85 loop on the base SEO (AI modes). False = single-pass (used when
    # the base is only a SEED for per-channel SEO, so we don't spend extra passes improving it).
    improve: bool = True


def _resolve_style_source(db: Session, user: models.User, style_source_id):
    """Load a user-owned style-reference Channel, or None. Raises 404 if
    unknown/not owned, 409 if it's a connected publish account."""
    if not style_source_id:
        return None
    ch = (db.query(models.Channel)
          .filter(models.Channel.id == style_source_id,
                  models.Channel.user_id == user.id)
          .first())
    if ch is None:
        raise HTTPException(status_code=404, detail="style_source not found")
    tok = ch.oauth_token
    if tok is not None and (tok.refresh_token_enc or "").strip():
        raise HTTPException(
            status_code=409,
            detail="style_source points at a connected account, not a "
                   "style reference",
        )
    return ch


@router.post("/clips/{clip_id}/quick-seo")
def quick_seo(
    clip_id: int,
    body: QuickSeoRequest,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    clip = _owned_clip(db, user, clip_id)
    t0 = time.monotonic()
    language = (body.language or "te").strip() or "te"

    if body.mode == "manual":
        title = (body.title or "").strip()
        if not title:
            raise HTTPException(status_code=400, detail="Title is required")
        seo = {
            "title": title[:100],
            "description": (body.description or "").strip(),
            "keywords": [str(t).strip() for t in (body.tags or []) if str(t).strip()][:30],
            "hashtags": _norm_hashtags(body.hashtags),
            "hook": "",
            "thumbnail_text": "",
            "metadata": {},
            "language": language,
            "model": "manual",
            "edited_by_user": True,
        }
        # Keep the working title in sync — it's the publish fallback.
        clip.text = title[:120]
        _persist_seo(db, clip, seo)
        return {"seo": seo, "elapsed_s": round(time.monotonic() - t0, 1)}

    # AI modes — generate, then (when improve=True) KEEP IMPROVING until 85+ (attempt cap).
    from pipeline_v4.seo_provider import SeoInput, generate_seo, generate_seo_to_score
    _seo_belt(clip.job_id, user.id, "generate", "entered")

    kind = _seo_kind(clip)
    style_source = _resolve_style_source(db, user, body.style_source_id)
    transcript_chars = 0

    if body.mode == "description":
        desc = (body.description or "").strip()
        if not desc:
            raise HTTPException(
                status_code=400,
                detail="Describe the video so SEO can be generated from it",
            )
        inp = SeoInput(
            kind=kind,
            language=language,
            title_native=(clip.text or "").strip(),
            summary=desc[:_DESCRIPTION_CAP_CHARS],
            style_source_id=body.style_source_id,
        )
    else:  # transcript
        tmp = tempfile.mkdtemp(prefix="kaizer_quickseo_")
        try:
            from pipeline_v4.trim_engine import (
                _deepgram_words, _extract_audio_mp3,
            )
            video = _materialize_video(clip, tmp)
            audio = os.path.join(tmp, "audio.mp3")
            try:
                _extract_audio_mp3(video, audio)
            except Exception as exc:
                raise HTTPException(
                    status_code=502, detail=f"Audio extraction failed: {exc}",
                )
            try:
                words, _dur = _deepgram_words(audio, language=language or "multi")
            except Exception as exc:
                raise HTTPException(
                    status_code=502, detail=f"Transcription failed: {exc}",
                )
            transcript = " ".join(
                str(w.get("w") or "") for w in (words or [])
            ).strip()
            if not transcript:
                raise HTTPException(
                    status_code=422,
                    detail="No speech detected in the video — try the "
                           "'From my description' mode instead",
                )
            transcript = transcript[:_TRANSCRIPT_CAP_CHARS]
            transcript_chars = len(transcript)
            # Cache for the thumbnail generator — never re-pay Deepgram.
            meta = _meta(clip)
            meta["transcript_excerpt"] = transcript[:_TRANSCRIPT_CACHE_CHARS]
            _save_meta(db, clip, meta)
            inp = SeoInput(
                kind=kind,
                language=language,
                title_native=(clip.text or "").strip(),
                body=transcript,
                style_source_id=body.style_source_id,
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # Improve-until-85 (re-generates feeding the score-checker's suggestions back in), OR a
    # single pass when this base is only a seed for per-channel SEO (improve=False) — saves passes.
    if body.improve:
        seo = generate_seo_to_score(inp, style_source=style_source, target_score=85, max_attempts=4)
    else:
        seo = generate_seo(inp, style_source=style_source)
    if not (seo.get("title") or "").strip():
        raise HTTPException(
            status_code=502,
            detail="SEO generation returned nothing usable — try again or "
                   "write it manually",
        )
    seo["language"] = seo.get("language") or language
    _persist_seo(db, clip, seo)
    _seo_belt(clip.job_id, user.id, "generate", "exited")
    return {
        "seo": seo,
        "transcript_chars": transcript_chars,
        "elapsed_s": round(time.monotonic() - t0, 1),
    }


# ─── quick-seo: per-channel (channel-wise SEO) ───────────────────────


class QuickPerChannelSeoIn(BaseModel):
    # per_channel = each connected channel gets its OWN keyword-tuned SEO;
    # shared = revert to one SEO for all (clears our per-channel markers).
    mode: str = Field("per_channel", pattern="^(per_channel|shared)$")
    # Limit to these channel ids; default = all the user's connected accounts.
    channel_ids: Optional[List[int]] = None


@router.post("/clips/{clip_id}/quick-seo/per-channel")
def quick_seo_per_channel(
    clip_id: int,
    body: QuickPerChannelSeoIn,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Channel-wise SEO for Quick Publish — same engine the V4 editor uses
    (``apply_per_channel_seo``), but the base SEO comes from the clip's own
    generated ``clip.seo`` (not a canvas). For each connected channel we adapt
    the base SEO with that channel's winning keywords + the clip's content
    keywords, and persist it into ``clip.seo_variants[channel_id]`` marked
    ``_per_channel`` — which ``upload_dispatch._compose_metadata`` then applies
    PER CHANNEL at publish. Idempotent; never touches legacy (non-marked)
    variants. Emits to the admin SEO conveyor belt."""
    clip = _owned_clip(db, user, clip_id)
    base_seo = json.loads(clip.seo) if (clip.seo or "").strip() else {}
    if not (base_seo.get("title") or "").strip():
        raise HTTPException(status_code=400,
                            detail="Generate the base SEO first, then split it per channel.")

    _seo_belt(clip.job_id, user.id, "per_channel", "entered")
    try:
        variants = json.loads(clip.seo_variants or "{}") if clip.seo_variants else {}
        if not isinstance(variants, dict):
            variants = {}
    except Exception:
        variants = {}
    # Clean slate for our markers; keep any legacy (non-marked) variants.
    variants = {k: v for k, v in variants.items()
                if not (isinstance(v, dict) and v.get("_per_channel"))}

    if body.mode == "shared":
        clip.seo_variants = json.dumps(variants)
        db.add(clip); db.commit()
        _seo_belt(clip.job_id, user.id, "per_channel", "exited")
        return {"ok": True, "mode": "shared", "applied": 0, "previews": []}

    # Connected publish accounts only (optionally narrowed to channel_ids).
    channels = [c for c in db.query(models.Channel).filter(
                    models.Channel.user_id == user.id).all()
                if c.oauth_token is not None and bool(c.oauth_token.refresh_token_enc)]
    if body.channel_ids:
        wanted = set(int(i) for i in body.channel_ids)
        channels = [c for c in channels if c.id in wanted]
    if not channels:
        raise HTTPException(status_code=409,
                            detail="No connected channels to generate per-channel SEO for.")

    # Content keywords from the clip text + cached transcript excerpt.
    content = (clip.text or "")
    try:
        m = _meta(clip)
        content = (content + " " + (m.get("transcript_excerpt") or "")).strip()
    except Exception:
        pass
    try:
        from seo.score_checker import _content_keywords
        ckw = _content_keywords(content, top=10)
    except Exception:
        ckw = []

    # Per-channel winning keywords (the feedback loop), then adapt.
    from seo.performance_profile import build_channel_profile
    from seo.per_channel import build_channel_previews
    try:
        from analytics.channel_catalog import ensure_synced as _ensure_synced
    except Exception:
        _ensure_synced = None
    fb: dict = {}
    for ch in channels:
        # Pull the channel's REAL YouTube catalogue (best-effort) so a
        # monetized channel's 100s of native videos feed the winning-keyword
        # profile — instead of 'no history'. Cached + only re-synced weekly.
        try:
            gcid = (ch.oauth_token.google_channel_id or "") if ch.oauth_token else ""
            if _ensure_synced is not None and gcid:
                _ensure_synced(db, user.id, gcid)
        except Exception:
            pass
        try:
            p = build_channel_profile(db, ch.id)
            if p.get("ready"):
                fb[ch.id] = {"winning_keywords": p.get("winning_keywords", [])}
        except Exception:
            pass
    # generate=True -> WRITE a DISTINCT title/description per channel with the real SEO engine
    # (variation_index + avoid_titles dedupe across channels), not just the base SEO + a channel
    # suffix. Mirrors the V4 editor's "apply-per-channel". 1 AI call/channel (the button shows
    # "Tailoring…"); falls back to the adapted base per channel if a write fails.
    previews = build_channel_previews(
        base_seo=base_seo, channels=channels,
        feedback_by_channel=fb, content_keywords=ckw, mode="per_channel",
        generate=True, content_text=content, language=(getattr(clip, "language", None) or "te"),
    )
    applied = 0
    for p in previews:
        cid = p.get("channel_id")
        if cid is None:
            continue
        _v = {
            "title": p.get("title", ""),
            "description": p.get("description", ""),
            "tags": p.get("tags", []),
            "_per_channel": True,
        }
        if p.get("hashtags"):
            _v["hashtags"] = p.get("hashtags")
        if p.get("seo_score") is not None:
            _v["seo_score"] = p.get("seo_score")
        variants[str(cid)] = _v
        applied += 1
    clip.seo_variants = json.dumps(variants)
    db.add(clip); db.commit()
    _seo_belt(clip.job_id, user.id, "per_channel", "exited")
    return {"ok": True, "mode": "per_channel", "applied": applied, "previews": previews}


# ─── quick-thumbnail ─────────────────────────────────────────────────


def _store_thumbnail(
    db: Session, user: models.User, clip: models.Clip,
    local_path: str, content_type: str,
) -> dict:
    from pipeline_core.storage import get_storage_provider
    ext = "png" if "png" in content_type else "jpg"
    key = f"raw_uploads/{int(user.id)}/clip_{int(clip.id)}/thumb_{int(time.time())}.{ext}"
    stored = get_storage_provider().upload(local_path, key, content_type=content_type)
    meta = _meta(clip)
    meta["publish_thumbnail_key"] = stored.key
    _save_meta(db, clip, meta)
    clip.thumb_storage_url = stored.url
    db.add(clip)
    db.commit()
    return {"thumb_url": stored.url, "key": stored.key}


@router.post("/clips/{clip_id}/quick-thumbnail/upload")
def quick_thumbnail_upload(
    clip_id: int,
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    clip = _owned_clip(db, user, clip_id)
    ct = (image.content_type or "").lower()
    if ct not in ("image/jpeg", "image/jpg", "image/png"):
        raise HTTPException(status_code=400, detail="Thumbnail must be JPG or PNG")
    data = image.file.read(_THUMB_MAX_BYTES + 1)
    if len(data) > _THUMB_MAX_BYTES:
        raise HTTPException(status_code=400, detail="Thumbnail must be ≤ 4 MB")
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload")
    tmp = tempfile.mkdtemp(prefix="kaizer_quickthumb_")
    try:
        local = os.path.join(tmp, f"thumb.{'png' if 'png' in ct else 'jpg'}")
        with open(local, "wb") as f:
            f.write(data)
        return _store_thumbnail(db, user, clip, local, ct)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class QuickThumbGenRequest(BaseModel):
    style: str = "symbolic"  # thumbnail_styles key; UI may expose later


@router.post("/clips/{clip_id}/quick-thumbnail/generate")
def quick_thumbnail_generate(
    clip_id: int,
    body: QuickThumbGenRequest = QuickThumbGenRequest(),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """AI thumbnail from the clip's SEO (+ cached transcript when the
    transcript SEO mode ran). Reuses the V4 director end-to-end via a
    minimal synthesized Canvas — one planner call, one prompt call, one
    image call."""
    clip = _owned_clip(db, user, clip_id)
    try:
        seo_raw = json.loads(clip.seo) if clip.seo else {}
    except Exception:
        seo_raw = {}
    if not (isinstance(seo_raw, dict) and (seo_raw.get("title") or "").strip()):
        raise HTTPException(
            status_code=409,
            detail="Generate or write the SEO first — the thumbnail is "
                   "designed from it",
        )

    from pipeline_v4.canvas_schema import (
        Canvas, CanvasLayout, CanvasSEO, CanvasStory,
    )
    from pipeline_v4.thumbnail_ai import make_thumbnail_for_canvas

    meta = _meta(clip)
    language = str(seo_raw.get("language") or meta.get("language") or "te")
    duration = float(clip.duration or 0) or 1.0
    canvas = Canvas(
        kind=("short" if _seo_kind(clip) == "short" else "bulletin"),
        output_filename="quick_publish.mp4",
        layout=CanvasLayout(),
        stories=[CanvasStory(
            story_index=0,
            video_t_start=0.0,
            video_t_end=max(1.0, duration),
            title_native=str(seo_raw.get("title") or clip.text or ""),
            summary=str(seo_raw.get("description") or "")[:400],
        )],
        trimmed_video_path="quick_publish_unused",
        seo=CanvasSEO(
            title=str(seo_raw.get("title") or ""),
            description=str(seo_raw.get("description") or ""),
            hook=str(seo_raw.get("hook") or ""),
            thumbnail_text=str(seo_raw.get("thumbnail_text") or ""),
            language=language,
        ),
    )

    tmp = tempfile.mkdtemp(prefix="kaizer_quickthumb_")
    try:
        out_path = os.path.join(tmp, "thumb.png")
        try:
            saved, prompt = make_thumbnail_for_canvas(
                canvas=canvas,
                out_path=out_path,
                language=language,
                transcript_excerpt=str(meta.get("transcript_excerpt") or ""),
                style=(body.style or "symbolic"),
                engine="gemini",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail=f"Thumbnail generation failed: {exc}",
            )
        if not saved or not os.path.isfile(saved):
            raise HTTPException(
                status_code=502,
                detail="Thumbnail model returned no image — try again",
            )
        result = _store_thumbnail(db, user, clip, saved, "image/png")
        # Keep the prompt so a future "tweak" flow can iterate cheaply.
        meta = _meta(clip)
        meta["thumbnail_prompt"] = (prompt or "")[:2000]
        _save_meta(db, clip, meta)
        db.commit()
        return result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── quick-state (wizard resume) ─────────────────────────────────────


@router.get("/clips/{clip_id}/quick-state")
def quick_state(
    clip_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    clip = _owned_clip(db, user, clip_id)
    meta = _meta(clip)
    seo = None
    try:
        parsed = json.loads(clip.seo) if clip.seo else None
        if isinstance(parsed, dict) and (parsed.get("title") or "").strip():
            seo = parsed
    except Exception:
        pass
    return {
        "clip_id": int(clip.id),
        "job_id": int(clip.job_id),
        "filename": clip.filename,
        "duration": float(clip.duration or 0),
        "platform": meta.get("platform") or "youtube_full",
        "language": meta.get("language") or "te",
        "working_title": clip.text or "",
        "seo": seo,
        "thumb_url": clip.thumb_storage_url or "",
        "publish_thumbnail_key": meta.get("publish_thumbnail_key") or "",
        "has_transcript": bool(meta.get("transcript_excerpt")),
        "storage_url": clip.storage_url or "",
    }


__all__ = ["router"]
