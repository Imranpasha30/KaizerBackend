import os
import json
import time
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

# Create tables on startup
Base.metadata.create_all(bind=engine)

BASE_DIR = Path(__file__).parent
MEDIA_ROOT = BASE_DIR / "media"
OUTPUT_ROOT = Path(os.getenv("KAIZER_OUTPUT_ROOT", str(BASE_DIR / "output")))
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

# ── Static data ──────────────────────────────────────────────────────────────

PLATFORMS = {
    "instagram_reel": {"label": "Instagram Reel", "width": 1080, "height": 1920},
    "youtube_short":  {"label": "YouTube Short",  "width": 1080, "height": 1920},
    "youtube_full":   {"label": "YouTube Full",   "width": 1920, "height": 1080},
}

FRAMES = {
    "news_standard": {"label": "News Standard"},
    "news_modern":   {"label": "News Modern"},
    "news_minimal":  {"label": "News Minimal"},
}

# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/api/health/")
def health():
    return {"status": "ok"}

# ── Platforms & Frames ───────────────────────────────────────────────────────

@app.get("/api/platforms/")
def get_platforms():
    return PLATFORMS

@app.get("/api/frames/")
def get_frames():
    return FRAMES

# ── Jobs ─────────────────────────────────────────────────────────────────────

@app.get("/api/jobs/")
def list_jobs(db: Session = Depends(get_db)):
    jobs = db.query(models.Job).order_by(models.Job.created_at.desc()).all()
    return [
        {
            "id": j.id,
            "status": j.status,
            "platform": j.platform,
            "frame": j.frame,
            "video_filename": j.video_filename,
            "created_at": j.created_at,
            "clip_count": len(j.clips),
        }
        for j in jobs
    ]


@app.post("/api/jobs/")
async def create_job(
    video: UploadFile = File(...),
    platform: str = Form(...),
    frame: str = Form(...),
    db: Session = Depends(get_db),
):
    upload_dir = MEDIA_ROOT / "uploads"
    upload_dir.mkdir(exist_ok=True)
    video_path = upload_dir / video.filename

    with open(video_path, "wb") as f:
        f.write(await video.read())

    job = models.Job(
        platform=platform,
        frame=frame,
        video_filename=video.filename,
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
        frame=frame,
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
        "frame": job.frame,
        "video_filename": job.video_filename,
        "log": job.log,
        "created_at": job.created_at,
        "clips": [_clip_dict(c) for c in job.clips],
    }


@app.get("/api/jobs/{job_id}/log/")
def stream_log(job_id: int):
    """SSE: stream log updates until job completes."""
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


# ── Video serving with HTTP Range support ────────────────────────────────────

@app.get("/api/media/{clip_id}/video/")
async def serve_video(clip_id: int, request: Request, db: Session = Depends(get_db)):
    clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    path = Path(clip.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    file_size = path.stat().st_size
    range_header = request.headers.get("range", "")

    if range_header.startswith("bytes="):
        start_str, _, end_str = range_header[6:].partition("-")
        start = int(start_str)
        end = int(end_str) if end_str else file_size - 1
        length = end - start + 1

        def _iter():
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            _iter(), status_code=206, media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
            },
        )

    return StreamingResponse(
        open(path, "rb"), media_type="video/mp4",
        headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
    )


# ── Helper ───────────────────────────────────────────────────────────────────

def _clip_dict(c):
    return {
        "id": c.id,
        "job_id": c.job_id,
        "clip_index": c.clip_index,
        "filename": c.filename,
        "file_path": c.file_path,
        "duration": c.duration,
        "sentiment": c.sentiment,
        "entities": json.loads(c.entities or "[]"),
        "meta": json.loads(c.meta or "{}"),
        "video_url": f"/api/media/{c.id}/video/",
    }
