import os
import json
import time
import shutil
import mimetypes
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from dotenv import load_dotenv

load_dotenv()

from database import engine, SessionLocal, Base, get_db
import models
import runner
import auth

from routers.auth import router as auth_router
from routers.channels import router as channels_router
from routers.seo import router as seo_router
from routers.youtube_oauth import router as youtube_oauth_router
from routers.youtube_upload import router as youtube_upload_router
from routers.campaigns import router as campaigns_router
from routers.performance import router as performance_router
from routers.translation import router as translation_router
from routers.trending import router as trending_router
from routers.assets import router as assets_router
from routers.veo import router as veo_router
from routers.channel_groups import router as channel_groups_router
from routers.billing import router as billing_router
from routers.job_progress import router as job_progress_router
from routers.feedback import router as feedback_router
from seo.default_channels import seed_channels
from youtube import worker as upload_worker
from learning import scheduler as corpus_scheduler


def _migrate_schema():
    """Add missing columns to existing tables — safe to run on every startup."""
    from sqlalchemy import text, inspect
    with engine.connect() as conn:
        inspector = inspect(engine)

        # ── jobs table columns ──────────────────────────────────────────
        existing_jobs = {c["name"] for c in inspector.get_columns("jobs")}
        job_additions = {
            "frame_layout": "VARCHAR(50)",
            "video_name":   "VARCHAR(255)",
            "error":        "TEXT",
            "language":     "VARCHAR(10) DEFAULT 'te'",
            "user_id":      "INTEGER",
            "started_at":   "DATETIME",
            "finished_at":  "DATETIME",
        }
        for col, dtype in job_additions.items():
            if col not in existing_jobs:
                conn.execute(text(f"ALTER TABLE jobs ADD COLUMN {col} {dtype}"))

        # ── user_id on multi-tenant tables ──────────────────────────────
        for tbl in ("channels", "upload_jobs", "campaigns", "competitor_channels"):
            if inspector.has_table(tbl):
                cols = {c["name"] for c in inspector.get_columns(tbl)}
                if "user_id" not in cols:
                    conn.execute(text(f"ALTER TABLE {tbl} ADD COLUMN user_id INTEGER"))

        # ── channels.logo_asset_id (per-channel video overlay logo) ─────
        if inspector.has_table("channels"):
            cols = {c["name"] for c in inspector.get_columns("channels")}
            if "logo_asset_id" not in cols:
                conn.execute(text("ALTER TABLE channels ADD COLUMN logo_asset_id INTEGER"))

        # ── user_assets.folder_path (virtual folders for organization) ──
        if inspector.has_table("user_assets"):
            cols = {c["name"] for c in inspector.get_columns("user_assets")}
            if "folder_path" not in cols:
                conn.execute(text("ALTER TABLE user_assets ADD COLUMN folder_path VARCHAR(255) DEFAULT ''"))

        # ── users.socials (cross-promo links for SEO footer) ────────────
        if inspector.has_table("users"):
            ucols = {c["name"] for c in inspector.get_columns("users")}
            if "socials" not in ucols:
                conn.execute(text("ALTER TABLE users ADD COLUMN socials TEXT DEFAULT '{}'"))

            # ── users: billing / subscription columns ────────────────────
            user_billing_additions = {
                "plan":                   "VARCHAR(20) DEFAULT 'free'",
                "plan_cycle":             "VARCHAR(10) DEFAULT 'monthly'",
                "plan_renews_at":         "TIMESTAMP NULL" if engine.dialect.name == "postgresql" else "DATETIME",
                "stripe_customer_id":     "VARCHAR(64)",
                "stripe_subscription_id": "VARCHAR(64)",
                "monthly_clip_count":     "INTEGER DEFAULT 0",
                "usage_reset_at":         "TIMESTAMP NULL" if engine.dialect.name == "postgresql" else "DATETIME",
            }
            for col, dtype in user_billing_additions.items():
                if col not in ucols:
                    conn.execute(text(f"ALTER TABLE users ADD COLUMN {col} {dtype}"))

        # ── profile_destinations — seed from existing OAuthTokens so
        # pre-existing 1:1 profile↔destination links become rows in the
        # new many-to-many table without the user having to re-link them.
        # Dialect-aware: SQLite uses INSERT OR IGNORE, Postgres uses
        # ON CONFLICT DO NOTHING on the (profile_id, google_channel_id) UK.
        if inspector.has_table("profile_destinations") and inspector.has_table("oauth_tokens"):
            dialect = engine.dialect.name
            if dialect == "postgresql":
                conn.execute(text("""
                    INSERT INTO profile_destinations (profile_id, google_channel_id)
                    SELECT channel_id, google_channel_id
                    FROM oauth_tokens
                    WHERE google_channel_id IS NOT NULL AND google_channel_id != ''
                    ON CONFLICT DO NOTHING
                """))
            else:
                conn.execute(text("""
                    INSERT OR IGNORE INTO profile_destinations (profile_id, google_channel_id)
                    SELECT channel_id, google_channel_id
                    FROM oauth_tokens
                    WHERE google_channel_id IS NOT NULL AND google_channel_id != ''
                """))

        # Migrate old 'frame' → 'frame_layout' if needed
        if "frame" in existing_jobs and "frame_layout" in existing_jobs:
            conn.execute(text(
                "UPDATE jobs SET frame_layout = frame WHERE frame_layout IS NULL"
            ))

        # ── clips table columns ─────────────────────────────────────────
        existing_clips = {c["name"] for c in inspector.get_columns("clips")}
        clip_additions = {
            "thumb_path":    "VARCHAR(500)",
            "image_path":    "VARCHAR(500)",
            "frame_type":    "VARCHAR(50)",
            "text":          "TEXT",
            "card_params":   "TEXT",
            "section_pct":   "TEXT",
            "follow_params": "TEXT",
            "seo":           "TEXT",
            "seo_variants":  "TEXT DEFAULT '{}'",
        }
        for col, dtype in clip_additions.items():
            if col not in existing_clips:
                conn.execute(text(f"ALTER TABLE clips ADD COLUMN {col} {dtype}"))

        # ── upload_jobs table columns (new: publish_kind) ────────────────
        if inspector.has_table("upload_jobs"):
            existing_uploads = {c["name"] for c in inspector.get_columns("upload_jobs")}
            upload_additions = {
                "publish_kind": "VARCHAR(10) DEFAULT 'video'",
            }
            for col, dtype in upload_additions.items():
                if col not in existing_uploads:
                    conn.execute(text(f"ALTER TABLE upload_jobs ADD COLUMN {col} {dtype}"))

        # ── oauth_tokens: cached YouTube-channel metadata ───────────────
        # Populated on OAuth connect + manual refresh.  Eliminates repeated
        # YT Data API calls for display-only fields (thumbnail, subs, etc.)
        if inspector.has_table("oauth_tokens"):
            existing_tokens = {c["name"] for c in inspector.get_columns("oauth_tokens")}
            token_additions = {
                "channel_description":   "TEXT DEFAULT ''",
                "channel_thumbnail_url": "VARCHAR(500) DEFAULT ''",
                "channel_custom_url":    "VARCHAR(120) DEFAULT ''",
                "channel_country":       "VARCHAR(10) DEFAULT ''",
                "subscriber_count":      "INTEGER DEFAULT 0",
                "video_count":           "INTEGER DEFAULT 0",
                "view_count":            "BIGINT DEFAULT 0",
                "metadata_cached_at":    "TIMESTAMP NULL" if engine.dialect.name == "postgresql" else "DATETIME",
                # Per-YT-account logo (lives here, NOT on Channel which is a
                # style template).  NULL = no overlay on rendered videos.
                "logo_asset_id":         "INTEGER",
            }
            for col, dtype in token_additions.items():
                if col not in existing_tokens:
                    conn.execute(text(f"ALTER TABLE oauth_tokens ADD COLUMN {col} {dtype}"))

        conn.commit()


def _seed_defaults():
    """Populate new DB with default channel profiles + ensure legacy user exists,
    and backfill any pre-auth rows with NULL user_id onto that legacy user.
    Idempotent — safe to run on every startup.
    """
    from sqlalchemy import text as _text
    from auth import ensure_legacy_user

    db = SessionLocal()
    try:
        # Legacy user — owns all pre-existing data from the single-tenant era
        legacy = ensure_legacy_user(db)

        # Backfill — any row with NULL user_id belongs to legacy until a real
        # user claims it.  Cheap UPDATE; noop when there's nothing to backfill.
        for tbl in ("jobs", "channels", "upload_jobs", "campaigns", "competitor_channels"):
            try:
                db.execute(_text(f"UPDATE {tbl} SET user_id = :uid WHERE user_id IS NULL"),
                           {"uid": legacy.id})
            except Exception as e:
                print(f"[startup] backfill {tbl} skipped: {e}")
        db.commit()

        added = seed_channels(db)
        if added:
            # Freshly seeded channels also belong to legacy user
            db.execute(_text("UPDATE channels SET user_id = :uid WHERE user_id IS NULL"),
                       {"uid": legacy.id})
            db.commit()
            print(f"[startup] Seeded {added} default channel(s) for legacy user")
    finally:
        db.close()


# Create tables if they don't exist, then safely add any missing columns
Base.metadata.create_all(bind=engine)
_migrate_schema()
_seed_defaults()

BASE_DIR    = Path(__file__).parent
MEDIA_ROOT  = BASE_DIR / "media"
OUTPUT_ROOT = Path(os.getenv("KAIZER_OUTPUT_ROOT", "/tmp/kaizer_output"))
MEDIA_ROOT.mkdir(exist_ok=True)
OUTPUT_ROOT.mkdir(exist_ok=True)

app = FastAPI(title="Kaizer Pipeline API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
# Auth (register / login / Google sign-in / me).
app.include_router(auth_router)
# Phases 1–7: channels, SEO, OAuth, uploads.
app.include_router(channels_router)
app.include_router(seo_router)
app.include_router(youtube_oauth_router)
app.include_router(youtube_upload_router)
# Billion-dollar phases A–E: campaigns, performance, translation, trending radar.
app.include_router(campaigns_router)
app.include_router(performance_router)
app.include_router(translation_router)
app.include_router(trending_router)
app.include_router(assets_router)
app.include_router(veo_router)
app.include_router(channel_groups_router)
app.include_router(billing_router)
app.include_router(job_progress_router)   # Phase 2B — job progress endpoint
app.include_router(feedback_router)       # Phase 3.5 — post-publish feedback endpoint


# ── Upload worker lifecycle ──────────────────────────────────────────────────
@app.on_event("startup")
async def _start_upload_worker():
    try:
        await upload_worker.start()
        print("[startup] upload worker running")
    except Exception as e:
        print(f"[startup] WARN: upload worker failed to start: {e}")


@app.on_event("shutdown")
async def _stop_upload_worker():
    await upload_worker.stop()


# ── Channel learning cron (Phase 7) ──────────────────────────────────────────
@app.on_event("startup")
async def _start_corpus_scheduler():
    try:
        corpus_scheduler.start()
    except Exception as e:
        print(f"[startup] WARN: corpus scheduler failed to start: {e}")


@app.on_event("shutdown")
async def _stop_corpus_scheduler():
    corpus_scheduler.stop()

# ── Static config ────────────────────────────────────────────────────────────

PLATFORMS = {
    "instagram_reel": {"label": "Instagram Reel", "width": 1080, "height": 1920},
    "youtube_short":  {"label": "YouTube Short",  "width": 1080, "height": 1920},
    "youtube_full":   {"label": "YouTube Full",   "width": 1920, "height": 1080},
}

FRAME_LAYOUTS = {
    "torn_card":  "Torn Card — Classic torn-edge news card layout",
    "follow_bar": "Follow Bar — News card with follow bar at bottom",
    "minimal":    "Minimal — Clean single-section layout",
}

# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"app": "Kaizer Pipeline API", "docs": "/docs", "health": "/api/health/"}

@app.get("/api/health/")
def health():
    return {"status": "ok"}

# ── Config ───────────────────────────────────────────────────────────────────

@app.get("/api/platforms/")
def get_platforms():
    return PLATFORMS

@app.get("/api/frame-layouts/")
def get_frame_layouts():
    return FRAME_LAYOUTS

@app.get("/api/fonts/{filename}")
def serve_font(filename: str):
    from fastapi.responses import FileResponse
    font_path = BASE_DIR / "resources" / "fonts" / filename
    if not font_path.exists():
        raise HTTPException(status_code=404, detail="Font not found")
    return FileResponse(font_path, media_type="font/ttf", headers={
        "Cache-Control": "public, max-age=86400",
        "Access-Control-Allow-Origin": "*",
    })

@app.get("/api/frames/")          # legacy alias
def get_frames():
    return FRAME_LAYOUTS

# ── Jobs ─────────────────────────────────────────────────────────────────────

@app.get("/api/jobs/")
def list_jobs(db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    jobs = (
        db.query(models.Job)
          .filter(models.Job.user_id == user.id)
          .order_by(models.Job.created_at.desc())
          .all()
    )
    return [
        {
            "id": j.id,
            "status": j.status,
            "platform": j.platform,
            "frame_layout": j.frame_layout,
            "video_name": j.video_name,
            "language": j.language or "te",
            "created_at": j.created_at,
            "clip_count": len(j.clips),
        }
        for j in jobs
    ]


@app.post("/api/jobs/create/")
async def create_job(
    video: UploadFile = File(...),
    platform: str = Form(...),
    frame_layout: str = Form(...),
    language: str = Form("te"),
    use_default_image: bool = Form(False),
    # Optional: pick which style-profile's logo to overlay.  Resolved to the
    # UserAsset.file_path at render time.  Absent / invalid = no logo.
    logo_channel_id: Optional[int] = Form(None),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    import languages as _langs
    lang_cfg = _langs.get(language)  # falls back to default if invalid

    upload_dir = MEDIA_ROOT / "uploads"
    upload_dir.mkdir(exist_ok=True)
    video_path = upload_dir / video.filename

    with open(video_path, "wb") as f:
        f.write(await video.read())

    # ── Input validation gate (Phase 1) ──────────────────────────────────────
    # Validate the uploaded file before creating a job row or starting the
    # pipeline.  Hard errors (wrong codec, corrupt file, etc.) return HTTP 400
    # immediately so the user gets actionable feedback without wasting a job
    # slot.  Soft warnings are attached to the job's log field so they surface
    # in the UI after the pipeline completes.
    _validation_warnings: list[str] = []
    try:
        import sys as _sys
        _sys.path.insert(0, str(BASE_DIR / "pipeline_core"))
        from pipeline_core.validator import validate_input as _validate_input  # type: ignore
        _val_result = _validate_input(str(video_path))
        if not _val_result.ok:
            # Clean up uploaded file so it doesn't accumulate on disk
            try:
                video_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise HTTPException(
                status_code=400,
                detail={
                    "errors": _val_result.errors,
                    "warnings": _val_result.warnings,
                },
            )
        _validation_warnings = _val_result.warnings
    except HTTPException:
        raise
    except Exception as _ve:
        # Validator import failure or unexpected error — log and continue so a
        # broken validator never blocks all uploads.  The pipeline itself will
        # fail if the file is truly bad.
        import logging as _logging
        _logging.getLogger("kaizer.pipeline.validator").warning(
            "create_job: validator error (non-fatal): %s", _ve
        )

    _warning_prefix = ""
    if _validation_warnings:
        _warning_prefix = "[input warnings] " + "; ".join(_validation_warnings) + "\n"

    job = models.Job(
        user_id=user.id,
        platform=platform,
        frame_layout=frame_layout,
        video_name=video.filename,
        language=lang_cfg.code,
        status="pending",
        log=_warning_prefix,
        output_dir=str(OUTPUT_ROOT),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # If the user opted in, look up their default asset and pass the absolute
    # path to the pipeline so every clip uses it instead of stock photos.
    default_img_path = ""
    if use_default_image:
        asset = (
            db.query(models.UserAsset)
              .filter(
                  models.UserAsset.user_id == user.id,
                  models.UserAsset.is_default_ad == True,  # noqa: E712
              )
              .first()
        )
        if asset and asset.file_path and Path(asset.file_path).exists():
            default_img_path = asset.file_path

    # Per-destination logos: the pipeline now renders a CLEAN MASTER with no
    # logo overlay.  The upload worker applies each destination's logo at
    # publish time (youtube/logo_overlay.py) so Auto Wala videos get Auto
    # Wala's logo, Cyber Sphere videos get Cyber Sphere's.  Set
    # KAIZER_BAKE_LOGO_AT_RENDER=true in .env to restore the old single-logo
    # behavior (faster renders, but same logo across all destinations).
    _BAKE_AT_RENDER = (os.environ.get("KAIZER_BAKE_LOGO_AT_RENDER", "") or "").lower() == "true"
    default_logo_path = ""

    def _resolve_asset(asset_id):
        if not asset_id:
            return ""
        a = db.query(models.UserAsset).filter(
            models.UserAsset.id == asset_id,
            models.UserAsset.user_id == user.id,
        ).first()
        if a and a.file_path and Path(a.file_path).exists():
            return a.file_path
        return ""

    # Only bake a logo at render time when the explicit legacy flag is on.
    # Default path: render clean master, let upload worker overlay per-
    # destination (handled in youtube/logo_overlay.py + worker.py).
    if _BAKE_AT_RENDER and logo_channel_id:
        ch = (
            db.query(models.Channel)
              .filter(
                  models.Channel.id == logo_channel_id,
                  models.Channel.user_id == user.id,
              )
              .first()
        )
        tok = ch.oauth_token if ch else None
        asset_id = (tok.logo_asset_id if tok and tok.logo_asset_id
                    else (ch.logo_asset_id if ch else None))
        default_logo_path = _resolve_asset(asset_id)

    runner.run_pipeline(
        job_id=job.id,
        video_path=str(video_path),
        platform=platform,
        frame=frame_layout,
        language=lang_cfg.code,
        default_image=default_img_path,
        default_logo=default_logo_path,
        db_session_factory=SessionLocal,
    )

    return {"id": job.id, "status": job.status, "language": lang_cfg.code}


@app.get("/api/languages/")
def list_languages():
    """Language picker payload for the frontend New Job form."""
    import languages as _langs
    return _langs.list_options()


@app.post("/api/clips/raw-upload/")
async def raw_upload(
    video: UploadFile = File(...),
    title:    str = Form(""),
    language: str = Form("te"),
    platform: str = Form("youtube_full"),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Upload an already-edited video directly as a publishable Clip.

    Skips the whole Gemini-cut / compose pipeline. Creates a tiny Job with
    status='done' so all existing Editor / SEO / Publish / Uploads flows
    work against the resulting Clip unchanged.
    """
    import subprocess, sys
    import languages as _langs
    lang_cfg = _langs.get(language)

    # Save the file under a predictable path so /api/file/ allowlist accepts it
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    upload_dir = BASE_DIR / "output" / "raw_uploads" / timestamp
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(video.filename or "upload.mp4").name
    video_path = upload_dir / safe_name
    with open(video_path, "wb") as f:
        f.write(await video.read())

    # ffprobe for duration — silently defaults to 0 if the binary is missing
    duration = 0.0
    try:
        sys.path.insert(0, str(BASE_DIR / "pipeline_core"))
        from pipeline import FFMPEG_BIN, get_video_info  # type: ignore
        info = get_video_info(str(video_path))
        if info:
            duration = float(info.get("duration") or 0)
    except Exception:
        FFMPEG_BIN = "ffmpeg"  # Fallback — hope it's on PATH

    # Auto-generate a thumbnail from the first frame
    thumb_path = upload_dir / f"thumb_{safe_name}.jpg"
    try:
        subprocess.run(
            [FFMPEG_BIN, "-y", "-i", str(video_path),
             "-vframes", "1", "-q:v", "2", str(thumb_path)],
            capture_output=True, check=True, timeout=30,
        )
    except Exception as e:
        print(f"[raw-upload] thumb gen failed: {e}")
        thumb_path = None  # type: ignore

    # Minimal Job acting as a container for the standalone Clip
    job = models.Job(
        user_id=user.id,
        platform=platform,
        frame_layout="raw_upload",
        video_name=safe_name,
        language=lang_cfg.code,
        status="done",
        log="[raw-upload] no pipeline run — user-edited video",
        output_dir=str(upload_dir),
    )
    db.add(job); db.commit(); db.refresh(job)

    display_title = (title or "").strip() or Path(safe_name).stem

    clip = models.Clip(
        job_id=job.id,
        clip_index=0,
        filename=safe_name,
        file_path=str(video_path),
        thumb_path=str(thumb_path) if thumb_path and thumb_path.exists() else "",
        image_path="",
        duration=duration,
        frame_type="raw_upload",
        text=display_title,
        sentiment="",
        entities=json.dumps([]),
        card_params=json.dumps({}),
        section_pct=json.dumps({}),
        follow_params=json.dumps({}),
        meta=json.dumps({
            "raw_upload": True,
            "platform": platform,
            "language": lang_cfg.code,
            "original_filename": safe_name,
        }),
    )
    db.add(clip); db.commit(); db.refresh(clip)

    return {
        "job_id":   job.id,
        "clip_id":  clip.id,
        "duration": duration,
        "language": lang_cfg.code,
    }


@app.get("/api/jobs/{job_id}/")
def get_job(job_id: int, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    job = db.query(models.Job).filter(
        models.Job.id == job_id, models.Job.user_id == user.id,
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job.id,
        "status": job.status,
        "platform": job.platform,
        "frame_layout": job.frame_layout,
        "video_name": job.video_name,
        "language": job.language or "te",
        "log": job.log,
        "created_at":  job.created_at,
        "started_at":  job.started_at.isoformat()  if job.started_at  else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "elapsed_seconds": _elapsed_seconds(job),
        "clips": [_clip_dict(c) for c in job.clips],
    }


@app.get("/api/jobs/{job_id}/status/")
def get_job_status(job_id: int, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    job = db.query(models.Job).filter(
        models.Job.id == job_id, models.Job.user_id == user.id,
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    log_lines = (job.log or "").split("\n") if job.log else []
    progress_pct = _estimate_progress(log_lines, job.status)
    return {
        "status": job.status,
        "progress_pct": progress_pct,
        "log_lines": log_lines,
        "error": job.error or "",
        "started_at":  job.started_at.isoformat()  if job.started_at  else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "elapsed_seconds": _elapsed_seconds(job),
    }


def _elapsed_seconds(job: "models.Job") -> int | None:
    """Wall-clock runtime of the pipeline.  Live-counting while running."""
    start = job.started_at
    if not start:
        return None
    from datetime import datetime, timezone
    # SQLite strips tzinfo — treat stored naive datetimes as UTC.
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    end = job.finished_at
    if end is not None and end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    if end is None:
        end = datetime.now(timezone.utc)
    try:
        return max(0, int((end - start).total_seconds()))
    except Exception:
        return None


def _meta_path_from_log(log: str) -> Path | None:
    """Mine the stored job log for the exact editor_meta.json path.

    Handles three cases, in priority order:
      1. `[kaizer:meta] <abs_path>`   — emitted by pipeline.py post-write
      2. `Output: <dir>`              — legacy breadcrumb before the marker existed
    """
    if not log:
        return None
    import re
    # Priority 1: explicit marker
    for m in re.finditer(r"^\s*\[kaizer:meta\]\s+(.+?)\s*$", log, re.MULTILINE):
        p = Path(m.group(1).strip())
        if p.exists():
            return p
    # Priority 2: "Output: <dir>" printed by run_pipeline banner
    for m in re.finditer(r"^\s*Output:\s+(.+?)\s*$", log, re.MULTILINE):
        candidate = Path(m.group(1).strip()) / "editor_meta.json"
        if not candidate.is_absolute():
            candidate = (BASE_DIR / candidate).resolve()
        if candidate.exists():
            return candidate
    return None


@app.post("/api/jobs/{job_id}/reimport/")
def reimport_clips(job_id: int, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    """Re-run clip import for a job whose pipeline already finished.

    Used when `_import_clips` failed silently (e.g. the earlier cp1252 bug) or
    when a fresh editor_meta.json has been written to disk and the DB is stale.
    Clears any existing clip rows first so the import is idempotent.
    """
    job = db.query(models.Job).filter(
        models.Job.id == job_id, models.Job.user_id == user.id,
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Drop any stale Clip rows for this job so we don't accumulate duplicates
    db.query(models.Clip).filter(models.Clip.job_id == job_id).delete()
    db.commit()

    # Prefer the exact meta path mined from the job's log output — avoids
    # picking up an unrelated editor_meta.json from an older run via rglob.
    meta_override = _meta_path_from_log(job.log or "")

    try:
        runner._import_clips(job, db, meta_override=meta_override)
    except Exception as e:
        job.status = "failed"
        job.error = f"Reimport failed: {e}"
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))

    db.refresh(job)
    if not job.clips:
        job.status = "failed"
        job.error = "Reimport found 0 clips on disk."
        db.commit()
        raise HTTPException(
            status_code=422,
            detail="No clips found on disk for this job. Check the output directory.",
        )

    job.status = "done"
    job.error = ""
    db.commit()
    return {"imported": len(job.clips), "output_dir": job.output_dir}


@app.post("/api/jobs/{job_id}/export/")
def export_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    export_dir = BASE_DIR / "output" / "exports" / str(job_id)
    export_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for clip in job.clips:
        if clip.file_path and Path(clip.file_path).exists():
            dest = export_dir / Path(clip.file_path).name
            shutil.copy2(clip.file_path, dest)
            count += 1

    return {"count": count, "export_dir": str(export_dir)}


@app.delete("/api/jobs/{job_id}/delete/")
def delete_job(job_id: int, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    job = db.query(models.Job).filter(
        models.Job.id == job_id, models.Job.user_id == user.id,
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    db.delete(job)
    db.commit()
    return {"deleted": job_id}


@app.get("/api/jobs/{job_id}/log/")
def stream_log(job_id: int):
    """SSE: stream log until job completes."""
    def event_stream():
        last_len = 0
        for _ in range(600):
            db = SessionLocal()
            job = db.query(models.Job).filter(models.Job.id == job_id).first()
            db.close()
            if not job:
                break
            log = job.log or ""
            if len(log) > last_len:
                yield f"data: {json.dumps({'log': log[last_len:], 'status': job.status})}\n\n"
                last_len = len(log)
            if job.status in ("done", "failed"):
                yield f"data: {json.dumps({'log': '', 'status': job.status, 'done': True})}\n\n"
                break
            time.sleep(1)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ── Clips ────────────────────────────────────────────────────────────────────

@app.get("/api/clips/{clip_id}/")
def get_clip(clip_id: int, db: Session = Depends(get_db)):
    clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    return _clip_dict(clip)


@app.post("/api/clips/{clip_id}/rerender/")
async def rerender_clip(clip_id: int, request: Request, db: Session = Depends(get_db)):
    clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    edits = await request.json()

    # Persist new params to DB
    if "text" in edits:
        clip.text = edits["text"]
    if "frame_type" in edits:
        clip.frame_type = edits["frame_type"]
    if "section_pct" in edits:
        clip.section_pct = json.dumps(edits["section_pct"])
    if "follow_params" in edits:
        clip.follow_params = json.dumps(edits["follow_params"])

    card_params = json.loads(clip.card_params or "{}")
    for key in ("font_size", "text_color", "font_file", "card_style"):
        if key in edits:
            card_params[key] = edits[key]
    clip.card_params = json.dumps(card_params)

    db.commit()

    # Actually re-compose the clip using pipeline functions
    meta = json.loads(clip.meta or "{}")
    raw_path = meta.get("raw_path", "")
    if not raw_path:
        raise HTTPException(status_code=422, detail="No source video path in clip metadata — cannot rerender")
    if not Path(raw_path).exists():
        raise HTTPException(status_code=410, detail="Source video has expired (server was redeployed) — please re-run the pipeline job to regenerate clips")

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _recompose_clip, clip, meta, edits, db)
    except Exception as e:
        import traceback
        print(f"[rerender] compose error:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Rerender failed: {e}")

    return _clip_dict(clip)


def _recompose_clip(clip, meta, edits, db):
    """Re-compose a clip using the pipeline's compose functions with updated params."""
    import subprocess, sys

    raw_path = meta.get("raw_path", "")
    out_path = clip.file_path
    preset = meta.get("preset", {"width": 1080, "height": 1920})
    frame_type = clip.frame_type or meta.get("frame_type", "follow_bar")
    title_text = clip.text or meta.get("text", "KAIZER NEWS")
    image_path = clip.image_path or meta.get("image_path", "")

    card_params = json.loads(clip.card_params or "{}")
    follow_params = json.loads(clip.follow_params or "{}")
    section_pct = json.loads(clip.section_pct or "{}")

    # Import pipeline compose functions
    sys.path.insert(0, str(BASE_DIR / "pipeline_core"))
    from pipeline import compose_clip, compose_follow_bar, compose_split_frame, FFMPEG_BIN

    if frame_type == "follow_bar":
        compose_follow_bar(
            raw_path, out_path, preset,
            title_text=title_text,
            font_file=card_params.get("font_file", "Ponnala-Regular.ttf"),
            text_color=follow_params.get("text_color", card_params.get("text_color", "#ffff00")),
            text_size=int(card_params.get("font_size", 60)),
            bg_color=follow_params.get("bg_color", "#1a0a2e"),
            follow_text=follow_params.get("follow_text", "FOLLOW KAIZER NEWS TELUGU"),
            follow_text_color=follow_params.get("follow_text_color", "#ffffff"),
            velvet_style=follow_params.get("velvet_style"),
        )
    elif frame_type == "split_frame":
        compose_split_frame(raw_path, image_path, out_path, preset)
    else:
        # torn_card
        # card_style may be stored nested (after editor rerender) or flat in card_params
        # (original pipeline format uses card_c0/card_c1/edge/jag/... at top level).
        # Support both by falling back to the flat card_params keys.
        cs = card_params.get("card_style") or {
            k: v for k, v in card_params.items()
            if k not in ("font_size", "font_file", "text_color")
        }
        compose_clip(
            raw_path, image_path, title_text, out_path, preset,
            font_size=card_params.get("font_size", 52),
            text_color=card_params.get("text_color", "#ffffff"),
            font_file=card_params.get("font_file", "Ponnala-Regular.ttf"),
            section_pct=section_pct or None,
            card_style=cs or None,
        )

    # Regenerate thumbnail
    thumb_path = clip.thumb_path
    if thumb_path and out_path:
        try:
            subprocess.run(
                [FFMPEG_BIN, "-y", "-i", out_path, "-vframes", "1", "-q:v", "2", thumb_path],
                capture_output=True, check=True, timeout=30,
            )
        except Exception:
            pass


@app.post("/api/clips/{clip_id}/upload-image/")
async def upload_image(clip_id: int, image: UploadFile = File(...), db: Session = Depends(get_db)):
    clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    img_dir = MEDIA_ROOT / "images"
    img_dir.mkdir(exist_ok=True)
    img_path = img_dir / f"clip_{clip_id}_{image.filename}"

    with open(img_path, "wb") as f:
        f.write(await image.read())

    clip.image_path = str(img_path)
    db.commit()

    return {
        "image_path": str(img_path),
        "image_url": f"/api/file/?path={img_path}",
    }

# ── File serving (path-restricted, with Range support) ──────────────────────

def _allowed_file_roots() -> list[Path]:
    """Absolute paths the /api/file/ endpoint is permitted to read from."""
    roots = [
        BASE_DIR / "output",
        BASE_DIR / "media",
        MEDIA_ROOT,
        OUTPUT_ROOT,
    ]
    return [r.resolve() for r in roots if r.exists() or True]


def _is_safe_path(p: Path) -> bool:
    """Block path traversal: requested file must resolve under an allowed root."""
    try:
        resolved = p.resolve()
    except Exception:
        return False
    for root in _allowed_file_roots():
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


@app.get("/api/file/")
async def serve_file(path: str, request: Request):
    file_path = Path(path)
    if not _is_safe_path(file_path):
        raise HTTPException(status_code=403, detail="Path is not under an allowed root")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found — clips are ephemeral and expire on redeploy")

    mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    file_size = file_path.stat().st_size
    filename = file_path.name
    range_header = request.headers.get("range", "")

    # Common headers for all responses
    base_headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Access-Control-Expose-Headers": "Content-Disposition, Content-Length, Accept-Ranges",
    }

    if range_header.startswith("bytes="):
        start_str, _, end_str = range_header[6:].partition("-")
        start = int(start_str)
        end = int(end_str) if end_str else file_size - 1
        length = end - start + 1

        def _iter():
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            _iter(), status_code=206, media_type=mime,
            headers={
                **base_headers,
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Content-Length": str(length),
            },
        )

    return StreamingResponse(
        open(file_path, "rb"), media_type=mime,
        headers={**base_headers, "Content-Length": str(file_size)},
    )

# ── Helpers ──────────────────────────────────────────────────────────────────

def _clip_dict(c):
    def _furl(path):
        """Build /api/file/ URL — no exists() check; let endpoint handle 404."""
        return f"/api/file/?path={path}" if path else ""

    meta = json.loads(c.meta or "{}")
    raw_path = meta.get("raw_path", "")

    seo = None
    if c.seo:
        try:
            seo = json.loads(c.seo)
        except (ValueError, TypeError):
            seo = None

    return {
        "id":           c.id,
        "job_id":       c.job_id,
        "clip_index":   c.clip_index,
        "filename":     c.filename,
        "file_path":    c.file_path,
        "thumb_path":   c.thumb_path or "",
        "thumb_url":    _furl(c.thumb_path),
        "image_path":   c.image_path or "",
        "image_url":    _furl(c.image_path),
        "raw_url":      _furl(raw_path),
        "duration":     c.duration,
        "frame_type":   c.frame_type,
        "text":         c.text,
        "sentiment":    c.sentiment,
        "entities":     json.loads(c.entities or "[]"),
        "card_params":  json.loads(c.card_params or "{}"),
        "section_pct":  json.loads(c.section_pct or "{}"),
        "follow_params":json.loads(c.follow_params or "{}"),
        "meta":         meta,
        "video_url":    _furl(c.file_path),
        "seo":          seo,
        "seo_variants": (lambda raw: (json.loads(raw) if raw else {}) or {})(getattr(c, "seo_variants", "") or "{}"),
    }


def _estimate_progress(log_lines: list, status: str) -> int:
    if status == "done":
        return 100
    if status == "failed":
        return 0
    # Estimate from pipeline step markers in log
    steps_found = sum(1 for l in log_lines if "STEP" in l.upper())
    return min(90, steps_found * 9)
