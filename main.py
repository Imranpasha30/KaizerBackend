import os
import json
import time
import shutil
import mimetypes
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from dotenv import load_dotenv

load_dotenv()

from database import engine, SessionLocal, Base, get_db
import models
import runner


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
        }
        for col, dtype in job_additions.items():
            if col not in existing_jobs:
                conn.execute(text(f"ALTER TABLE jobs ADD COLUMN {col} {dtype}"))

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
        }
        for col, dtype in clip_additions.items():
            if col not in existing_clips:
                conn.execute(text(f"ALTER TABLE clips ADD COLUMN {col} {dtype}"))

        conn.commit()


# Create tables if they don't exist, then safely add any missing columns
Base.metadata.create_all(bind=engine)
_migrate_schema()

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
def list_jobs(db: Session = Depends(get_db)):
    jobs = db.query(models.Job).order_by(models.Job.created_at.desc()).all()
    return [
        {
            "id": j.id,
            "status": j.status,
            "platform": j.platform,
            "frame_layout": j.frame_layout,
            "video_name": j.video_name,
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
    db: Session = Depends(get_db),
):
    upload_dir = MEDIA_ROOT / "uploads"
    upload_dir.mkdir(exist_ok=True)
    video_path = upload_dir / video.filename

    with open(video_path, "wb") as f:
        f.write(await video.read())

    job = models.Job(
        platform=platform,
        frame_layout=frame_layout,
        video_name=video.filename,
        status="pending",
        log="",
        output_dir=str(OUTPUT_ROOT),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    runner.run_pipeline(
        job_id=job.id,
        video_path=str(video_path),
        platform=platform,
        frame=frame_layout,
        db_session_factory=SessionLocal,
    )

    return {"id": job.id, "status": job.status}


@app.get("/api/jobs/{job_id}/")
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job.id,
        "status": job.status,
        "platform": job.platform,
        "frame_layout": job.frame_layout,
        "video_name": job.video_name,
        "log": job.log,
        "created_at": job.created_at,
        "clips": [_clip_dict(c) for c in job.clips],
    }


@app.get("/api/jobs/{job_id}/status/")
def get_job_status(job_id: int, db: Session = Depends(get_db)):
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    log_lines = (job.log or "").split("\n") if job.log else []
    progress_pct = _estimate_progress(log_lines, job.status)
    return {
        "status": job.status,
        "progress_pct": progress_pct,
        "log_lines": log_lines,
        "error": job.error or "",
    }


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
def delete_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
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
        _recompose_clip(clip, meta, edits, db)
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

# ── File serving (any path, with Range support) ──────────────────────────────

@app.get("/api/file/")
async def serve_file(path: str, request: Request):
    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found — clips are ephemeral and expire on redeploy")

    mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    file_size = file_path.stat().st_size
    filename = file_path.name
    range_header = request.headers.get("range", "")

    # Common headers for all responses
    base_headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": f'inline; filename="{filename}"',
        "Access-Control-Expose-Headers": "Content-Disposition",
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
    }


def _estimate_progress(log_lines: list, status: str) -> int:
    if status == "done":
        return 100
    if status == "failed":
        return 0
    # Estimate from pipeline step markers in log
    steps_found = sum(1 for l in log_lines if "STEP" in l.upper())
    return min(90, steps_found * 9)
