"""Admin REST surface — Phase 12.

Every endpoint is gated by `auth.admin_required`.  Responses NEVER include
raw prompts, OAuth access/refresh tokens, or signed R2 URLs — any sensitive
field is masked to "****".

Endpoints
---------
* GET  /api/admin/system                — live OS + GPU + process metrics
* GET  /api/admin/users                 — paginated user list with aggregates
* GET  /api/admin/users/{id}            — single user drill-down
* POST /api/admin/users/{id}/toggle-admin
* GET  /api/admin/jobs                  — paginated jobs across users
* GET  /api/admin/jobs/{id}             — drill-down: clips, uploads, gemini
* GET  /api/admin/gemini-usage          — aggregate analytics
* GET  /api/admin/live-events           — currently running live sessions
* GET  /api/admin/audit                 — last-50 logins / failures / errors
"""
from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import psutil
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

import auth
import models
from database import get_db


router = APIRouter(prefix="/api/admin", tags=["admin"])


# Track process start-time once at import; `psutil.Process().create_time()`
# is canonical but this fallback keeps the metric useful even if the process
# doesn't exist in psutil's table (unusual).
_PROCESS_START_UTC = datetime.now(timezone.utc)


# ─── Helpers ──────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _mask() -> str:
    """Canonical redaction token for sensitive fields in admin responses."""
    return "****"


def _gpu_metrics() -> dict:
    """Query nvidia-smi once — returns {} if the binary isn't on PATH or fails."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {}
    try:
        out = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.5,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return {}
        line = out.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            return {}
        return {
            "name":                parts[0],
            "memory_total_mb":     int(float(parts[1])),
            "memory_used_mb":      int(float(parts[2])),
            "utilization_percent": int(float(parts[3])),
            "temperature_c":       int(float(parts[4])),
        }
    except Exception:
        return {}


def _process_metrics() -> dict:
    try:
        p = psutil.Process()
        try:
            rss = p.memory_info().rss
        except Exception:
            rss = 0
        try:
            threads = p.num_threads()
        except Exception:
            threads = 0
        try:
            start_ts = p.create_time()
            uptime_s = max(0, int(_utcnow().timestamp() - start_ts))
        except Exception:
            uptime_s = int((_utcnow() - _PROCESS_START_UTC).total_seconds())
        return {
            "pid":       p.pid,
            "rss_gb":    round(rss / (1024 ** 3), 3),
            "threads":   threads,
            "uptime_s":  uptime_s,
        }
    except Exception:
        return {
            "pid": os.getpid(),
            "rss_gb": 0.0,
            "threads": 0,
            "uptime_s": int((_utcnow() - _PROCESS_START_UTC).total_seconds()),
        }


def _live_events_running() -> int:
    """Count currently-live event sessions known to the in-process registry."""
    try:
        from routers.live_director import _SESSIONS
        return len(_SESSIONS)
    except Exception:
        return 0


def _file_size_bytes(path: str) -> int:
    """Quick non-raising file-size lookup — returns 0 if the file is gone."""
    if not path:
        return 0
    try:
        return os.path.getsize(path)
    except Exception:
        return 0


def _storage_bytes_for_user(db: Session, user_id: int) -> int:
    """Sum of on-disk clip + asset file sizes for a user.

    SQLite friendly — we walk the rows rather than trying to compute
    server-side because SQLAlchemy can't stat() a file.  Caps at ~2000 rows
    per kind to keep the endpoint snappy.
    """
    total = 0
    clips = (
        db.query(models.Clip)
          .join(models.Job, models.Clip.job_id == models.Job.id)
          .filter(models.Job.user_id == user_id)
          .limit(2000)
          .all()
    )
    for c in clips:
        total += _file_size_bytes(c.file_path or "")
        total += _file_size_bytes(c.thumb_path or "")
    assets = (
        db.query(models.UserAsset)
          .filter(models.UserAsset.user_id == user_id)
          .limit(2000)
          .all()
    )
    for a in assets:
        total += _file_size_bytes(a.file_path or "")
    return total


def _since(days: int) -> datetime:
    return _utcnow() - timedelta(days=max(1, int(days)))


# ─── Shared serializers ───────────────────────────────────────────────────

def _user_row(
    db: Session,
    u: models.User,
    *,
    jobs_count: int,
    clips_count: int,
    storage_bytes: int,
    gemini_calls_30d: int,
    gemini_cost_30d: float,
) -> dict:
    return {
        "id":         u.id,
        "email":      u.email,
        "name":       u.name or "",
        "is_admin":   bool(u.is_admin),
        "is_active":  bool(u.is_active),
        "plan":       u.plan or "free",
        "created_at":    _iso(u.created_at),
        "last_login_at": _iso(u.last_login_at),
        "jobs_count":   int(jobs_count),
        "clips_count":  int(clips_count),
        "storage_mb":   round(storage_bytes / (1024 ** 2), 2),
        "gemini_calls_30d":   int(gemini_calls_30d),
        "gemini_cost_usd_30d": round(float(gemini_cost_30d), 4),
    }


def _job_row(db: Session, j: models.Job, *, user_email: str = "") -> dict:
    # Cheap — Job.clips is already a relationship.  For the listing endpoint
    # we compute it via a subquery instead (see /jobs); this serializer is
    # used in both places and accepts an already-loaded job.
    clips_count = (
        db.query(func.count(models.Clip.id))
          .filter(models.Clip.job_id == j.id)
          .scalar() or 0
    )
    gemini_cost = (
        db.query(func.coalesce(func.sum(models.GeminiCall.cost_usd), 0.0))
          .filter(models.GeminiCall.job_id == j.id)
          .scalar() or 0.0
    )
    return {
        "id":          j.id,
        "user_id":     j.user_id,
        "user_email":  user_email,
        "status":      j.status,
        "platform":    j.platform,
        "language":    j.language,
        "video_name":  j.video_name,
        "created_at":  _iso(j.created_at),
        "started_at":  _iso(j.started_at),
        "finished_at": _iso(j.finished_at),
        "error":       (j.error or "")[:500],
        "clips_count": int(clips_count),
        "gemini_cost_usd": round(float(gemini_cost), 4),
    }


def _gemini_call_row(g: models.GeminiCall) -> dict:
    """NEVER include a prompt or response body — this endpoint is for admin
    accounting only."""
    return {
        "id":            g.id,
        "user_id":       g.user_id,
        "job_id":        g.job_id,
        "clip_id":       g.clip_id,
        "model":         g.model,
        "purpose":       g.purpose or "",
        "prompt_tokens": int(g.prompt_tokens or 0),
        "output_tokens": int(g.output_tokens or 0),
        "total_tokens":  int(g.total_tokens or 0),
        "cost_usd":      round(float(g.cost_usd or 0.0), 4),
        "latency_ms":    int(g.latency_ms or 0),
        "status":        g.status or "",
        "error":         (g.error or "")[:300],
        "created_at":    _iso(g.created_at),
    }


# ─── Endpoints: system ────────────────────────────────────────────────────

@router.get("/system")
def system_metrics(_: models.User = Depends(auth.admin_required)) -> dict:
    """Live CPU / RAM / disk / GPU / process metrics + live-event count."""
    cpu_percent = psutil.cpu_percent(interval=0.0)   # non-blocking snapshot
    cpu_count = psutil.cpu_count(logical=True) or 0
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(os.path.abspath(os.sep))
    return {
        "cpu_percent":    round(float(cpu_percent), 1),
        "cpu_count":      int(cpu_count),
        "ram_total_gb":   round(vm.total / (1024 ** 3), 2),
        "ram_used_gb":    round(vm.used  / (1024 ** 3), 2),
        "ram_percent":    round(float(vm.percent), 1),
        "disk_total_gb":  round(disk.total / (1024 ** 3), 2),
        "disk_used_gb":   round(disk.used  / (1024 ** 3), 2),
        "disk_percent":   round(float(disk.percent), 1),
        "gpu":            _gpu_metrics(),
        "process":        _process_metrics(),
        "live_events_running": _live_events_running(),
        "timestamp":      _iso(_utcnow()),
    }


# ─── Endpoints: users ─────────────────────────────────────────────────────

@router.get("/users")
def list_users(
    q: str = Query("", description="Search email substring"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    _: models.User = Depends(auth.admin_required),
) -> dict:
    base = db.query(models.User)
    if q:
        base = base.filter(models.User.email.ilike(f"%{q}%"))
    total = base.count()
    users = (
        base.order_by(desc(models.User.created_at))
            .offset(offset)
            .limit(limit)
            .all()
    )

    since_30d = _since(30)
    out = []
    for u in users:
        jobs_count = (
            db.query(func.count(models.Job.id))
              .filter(models.Job.user_id == u.id)
              .scalar() or 0
        )
        clips_count = (
            db.query(func.count(models.Clip.id))
              .join(models.Job, models.Clip.job_id == models.Job.id)
              .filter(models.Job.user_id == u.id)
              .scalar() or 0
        )
        calls_30d = (
            db.query(func.count(models.GeminiCall.id))
              .filter(models.GeminiCall.user_id == u.id)
              .filter(models.GeminiCall.created_at >= since_30d)
              .scalar() or 0
        )
        cost_30d = (
            db.query(func.coalesce(func.sum(models.GeminiCall.cost_usd), 0.0))
              .filter(models.GeminiCall.user_id == u.id)
              .filter(models.GeminiCall.created_at >= since_30d)
              .scalar() or 0.0
        )
        storage_bytes = _storage_bytes_for_user(db, u.id)
        out.append(_user_row(
            db, u,
            jobs_count=jobs_count,
            clips_count=clips_count,
            storage_bytes=storage_bytes,
            gemini_calls_30d=calls_30d,
            gemini_cost_30d=cost_30d,
        ))
    return {"total": int(total), "users": out}


@router.get("/users/{user_id}")
def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    _: models.User = Depends(auth.admin_required),
) -> dict:
    u = db.query(models.User).filter(models.User.id == user_id).first()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")

    since_30d = _since(30)
    jobs_count = (
        db.query(func.count(models.Job.id))
          .filter(models.Job.user_id == u.id)
          .scalar() or 0
    )
    clips_count = (
        db.query(func.count(models.Clip.id))
          .join(models.Job, models.Clip.job_id == models.Job.id)
          .filter(models.Job.user_id == u.id)
          .scalar() or 0
    )
    calls_30d = (
        db.query(func.count(models.GeminiCall.id))
          .filter(models.GeminiCall.user_id == u.id)
          .filter(models.GeminiCall.created_at >= since_30d)
          .scalar() or 0
    )
    cost_30d = (
        db.query(func.coalesce(func.sum(models.GeminiCall.cost_usd), 0.0))
          .filter(models.GeminiCall.user_id == u.id)
          .filter(models.GeminiCall.created_at >= since_30d)
          .scalar() or 0.0
    )
    storage_bytes = _storage_bytes_for_user(db, u.id)

    base = _user_row(
        db, u,
        jobs_count=jobs_count,
        clips_count=clips_count,
        storage_bytes=storage_bytes,
        gemini_calls_30d=calls_30d,
        gemini_cost_30d=cost_30d,
    )

    # Recent jobs (compact rows to keep payload small)
    recent_jobs = (
        db.query(models.Job)
          .filter(models.Job.user_id == u.id)
          .order_by(desc(models.Job.created_at))
          .limit(20)
          .all()
    )
    base["recent_jobs"] = [
        {
            "id":         j.id,
            "status":     j.status,
            "platform":   j.platform,
            "video_name": j.video_name,
            "language":   j.language,
            "created_at": _iso(j.created_at),
            "finished_at": _iso(j.finished_at),
        }
        for j in recent_jobs
    ]

    # Recent Gemini calls — NO prompts, NO responses.
    recent_calls = (
        db.query(models.GeminiCall)
          .filter(models.GeminiCall.user_id == u.id)
          .order_by(desc(models.GeminiCall.created_at))
          .limit(20)
          .all()
    )
    base["recent_gemini_calls"] = [_gemini_call_row(g) for g in recent_calls]

    # Storage breakdown by kind (bytes on disk — approximate).
    storage_breakdown: dict[str, int] = {"clips": 0, "thumbs": 0, "assets": 0}
    clips = (
        db.query(models.Clip)
          .join(models.Job, models.Clip.job_id == models.Job.id)
          .filter(models.Job.user_id == u.id)
          .limit(2000)
          .all()
    )
    for c in clips:
        storage_breakdown["clips"]  += _file_size_bytes(c.file_path or "")
        storage_breakdown["thumbs"] += _file_size_bytes(c.thumb_path or "")
    assets = (
        db.query(models.UserAsset)
          .filter(models.UserAsset.user_id == u.id)
          .limit(2000)
          .all()
    )
    for a in assets:
        storage_breakdown["assets"] += _file_size_bytes(a.file_path or "")

    base["storage_breakdown_mb"] = {
        k: round(v / (1024 ** 2), 2) for k, v in storage_breakdown.items()
    }
    return base


@router.post("/users/{user_id}/toggle-admin")
def toggle_admin(
    user_id: int,
    db: Session = Depends(get_db),
    acting: models.User = Depends(auth.admin_required),
) -> dict:
    target = db.query(models.User).filter(models.User.id == user_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    # Refuse self-demotion — prevents locking the last admin out of the panel.
    # (If target is a different user, is_admin just flips either way.)
    if target.id == acting.id and target.is_admin:
        raise HTTPException(status_code=400, detail="Cannot demote yourself")

    target.is_admin = not bool(target.is_admin)
    db.commit()
    return {"id": target.id, "is_admin": bool(target.is_admin)}


# ─── Endpoints: jobs ──────────────────────────────────────────────────────

@router.get("/jobs")
def list_jobs(
    q: str = Query("", description="Search video_name substring"),
    status: str = Query("", description="Filter by status"),
    user_id: Optional[int] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    _: models.User = Depends(auth.admin_required),
) -> dict:
    base = db.query(models.Job)
    if q:
        base = base.filter(models.Job.video_name.ilike(f"%{q}%"))
    if status:
        base = base.filter(models.Job.status == status)
    if user_id is not None:
        base = base.filter(models.Job.user_id == user_id)
    total = base.count()
    jobs = (
        base.order_by(desc(models.Job.created_at))
            .offset(offset)
            .limit(limit)
            .all()
    )

    # Pre-load user emails in one query to avoid N+1.
    user_ids = [j.user_id for j in jobs if j.user_id]
    emails: dict[int, str] = {}
    if user_ids:
        rows = (
            db.query(models.User.id, models.User.email)
              .filter(models.User.id.in_(user_ids))
              .all()
        )
        emails = {uid: email for (uid, email) in rows}

    return {
        "total": int(total),
        "jobs":  [_job_row(db, j, user_email=emails.get(j.user_id or 0, "")) for j in jobs],
    }


@router.get("/jobs/{job_id}")
def get_job_detail(
    job_id: int,
    db: Session = Depends(get_db),
    _: models.User = Depends(auth.admin_required),
) -> dict:
    j = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")

    user_email = ""
    if j.user_id:
        u = db.query(models.User).filter(models.User.id == j.user_id).first()
        if u:
            user_email = u.email

    row = _job_row(db, j, user_email=user_email)

    # Clips — compact
    clips = (
        db.query(models.Clip)
          .filter(models.Clip.job_id == j.id)
          .order_by(models.Clip.clip_index.asc())
          .all()
    )
    row["clips"] = [
        {
            "id":         c.id,
            "clip_index": c.clip_index,
            "filename":   c.filename,
            "duration":   c.duration,
            "frame_type": c.frame_type,
            "text":       (c.text or "")[:200],
            # Public storage URL only — signed URLs are never exposed here.
            "storage_url": c.storage_url or "",
        }
        for c in clips
    ]

    # Upload jobs for this job's clips (flattened).
    clip_ids = [c.id for c in clips]
    uploads = []
    if clip_ids:
        uploads_rows = (
            db.query(models.UploadJob)
              .filter(models.UploadJob.clip_id.in_(clip_ids))
              .order_by(desc(models.UploadJob.created_at))
              .all()
        )
        for uj in uploads_rows:
            uploads.append({
                "id":         uj.id,
                "clip_id":    uj.clip_id,
                "channel_id": uj.channel_id,
                "status":     uj.status,
                "video_id":   uj.video_id,
                "attempts":   uj.attempts,
                "last_error": (uj.last_error or "")[:300],
                "created_at": _iso(uj.created_at),
            })
    row["uploads"] = uploads

    # Gemini calls tied to this job — NO prompts, NO responses.
    gemini = (
        db.query(models.GeminiCall)
          .filter(models.GeminiCall.job_id == j.id)
          .order_by(desc(models.GeminiCall.created_at))
          .all()
    )
    row["gemini_calls"] = [_gemini_call_row(g) for g in gemini]

    return row


# ─── Endpoints: gemini analytics ──────────────────────────────────────────

@router.get("/gemini-usage")
def gemini_usage(
    days: int = Query(30, ge=1, le=365),
    user_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    _: models.User = Depends(auth.admin_required),
) -> dict:
    since = _since(days)
    base = db.query(models.GeminiCall).filter(models.GeminiCall.created_at >= since)
    if user_id is not None:
        base = base.filter(models.GeminiCall.user_id == user_id)

    total_calls = base.count()
    total_tokens = (
        base.with_entities(func.coalesce(func.sum(models.GeminiCall.total_tokens), 0)).scalar()
        or 0
    )
    total_cost = (
        base.with_entities(func.coalesce(func.sum(models.GeminiCall.cost_usd), 0.0)).scalar()
        or 0.0
    )
    # Video accounting totals — Gemini bills video uploads per-second on top
    # of prompt tokens, so the admin needs these numbers to price tiers.
    total_bytes = (
        base.with_entities(func.coalesce(func.sum(models.GeminiCall.file_bytes), 0)).scalar()
        or 0
    )
    total_video_seconds = (
        base.with_entities(func.coalesce(func.sum(models.GeminiCall.video_duration_s), 0.0)).scalar()
        or 0.0
    )

    # Per-day buckets.  SQLite lacks date_trunc — use dialect-specific date
    # formatting.  Postgres → to_char; SQLite → strftime.  The fallback is
    # a server-side cast-to-text that works on every other dialect.
    dialect = db.bind.dialect.name if db.bind is not None else "sqlite"
    if dialect == "sqlite":
        day_expr = func.strftime("%Y-%m-%d", models.GeminiCall.created_at).label("day")
    elif dialect == "postgresql":
        day_expr = func.to_char(models.GeminiCall.created_at, "YYYY-MM-DD").label("day")
    else:
        from sqlalchemy import String as _String
        day_expr = func.substr(func.cast(models.GeminiCall.created_at, _String()), 1, 10).label("day")

    by_day_rows = (
        base.with_entities(
            day_expr,
            func.count(models.GeminiCall.id),
            func.coalesce(func.sum(models.GeminiCall.cost_usd), 0.0),
            func.coalesce(func.sum(models.GeminiCall.file_bytes), 0),
            func.coalesce(func.sum(models.GeminiCall.video_duration_s), 0.0),
        )
        .group_by("day")
        .order_by("day")
        .all()
    )
    by_day = [
        {
            "date":             r[0],
            "calls":            int(r[1] or 0),
            "cost_usd":         round(float(r[2] or 0.0), 4),
            "bytes":            int(r[3] or 0),
            "video_seconds":    round(float(r[4] or 0.0), 1),
        }
        for r in by_day_rows
    ]

    by_user_rows = (
        base.with_entities(
            models.GeminiCall.user_id,
            func.count(models.GeminiCall.id),
            func.coalesce(func.sum(models.GeminiCall.cost_usd), 0.0),
            func.coalesce(func.sum(models.GeminiCall.file_bytes), 0),
            func.coalesce(func.sum(models.GeminiCall.video_duration_s), 0.0),
        )
        .group_by(models.GeminiCall.user_id)
        .all()
    )
    emails: dict[int, str] = {}
    uids = [row[0] for row in by_user_rows if row[0]]
    if uids:
        for (uid, email) in db.query(models.User.id, models.User.email).filter(models.User.id.in_(uids)).all():
            emails[uid] = email
    by_user = [
        {
            "user_id":        r[0],
            "email":          emails.get(r[0] or 0, ""),
            "calls":          int(r[1] or 0),
            "cost_usd":       round(float(r[2] or 0.0), 4),
            "bytes":          int(r[3] or 0),
            "video_seconds":  round(float(r[4] or 0.0), 1),
        }
        for r in by_user_rows
    ]

    by_model_rows = (
        base.with_entities(
            models.GeminiCall.model,
            func.count(models.GeminiCall.id),
            func.coalesce(func.sum(models.GeminiCall.total_tokens), 0),
            func.coalesce(func.sum(models.GeminiCall.cost_usd), 0.0),
            func.coalesce(func.sum(models.GeminiCall.file_bytes), 0),
            func.coalesce(func.sum(models.GeminiCall.video_duration_s), 0.0),
        )
        .group_by(models.GeminiCall.model)
        .all()
    )
    by_model = [
        {
            "model":          r[0] or "",
            "calls":          int(r[1] or 0),
            "tokens":         int(r[2] or 0),
            "cost_usd":       round(float(r[3] or 0.0), 4),
            "bytes":          int(r[4] or 0),
            "video_seconds":  round(float(r[5] or 0.0), 1),
        }
        for r in by_model_rows
    ]

    by_purpose_rows = (
        base.with_entities(
            models.GeminiCall.purpose,
            func.count(models.GeminiCall.id),
            func.coalesce(func.sum(models.GeminiCall.cost_usd), 0.0),
            func.coalesce(func.sum(models.GeminiCall.file_bytes), 0),
            func.coalesce(func.sum(models.GeminiCall.video_duration_s), 0.0),
        )
        .group_by(models.GeminiCall.purpose)
        .all()
    )
    by_purpose = [
        {
            "purpose":        r[0] or "",
            "calls":          int(r[1] or 0),
            "cost_usd":       round(float(r[2] or 0.0), 4),
            "bytes":          int(r[3] or 0),
            "video_seconds":  round(float(r[4] or 0.0), 1),
        }
        for r in by_purpose_rows
    ]

    return {
        "total_calls":          int(total_calls),
        "total_tokens":         int(total_tokens),
        "total_cost_usd":       round(float(total_cost), 4),
        "total_bytes":          int(total_bytes),
        "total_video_seconds":  round(float(total_video_seconds), 1),
        "by_day":          by_day,
        "by_user":         by_user,
        "by_model":        by_model,
        "by_purpose":      by_purpose,
    }


# ─── Endpoints: live events ───────────────────────────────────────────────

@router.get("/live-events")
def list_live_events(
    db: Session = Depends(get_db),
    _: models.User = Depends(auth.admin_required),
) -> dict:
    """Currently running live sessions — reads the in-process _SESSIONS
    registry.  Gracefully returns an empty list if the module isn't loaded
    or the attribute has moved."""
    try:
        from routers.live_director import _SESSIONS
    except Exception:
        return {"running": []}

    out = []
    for event_id, session in list(_SESSIONS.items()):
        # Best-effort stats extraction — don't trust internal structure.
        cam_ids: list[str] = []
        workers_count = 0
        try:
            cam_ids = sorted(getattr(session, "rings", {}).keys())
        except Exception:
            cam_ids = []
        try:
            workers_count = len(getattr(session, "webrtc_workers", {}) or {})
        except Exception:
            workers_count = 0

        decision_count = (
            db.query(func.count(models.DirectorLogEntry.id))
              .filter(models.DirectorLogEntry.event_id == event_id)
              .scalar() or 0
        )
        # Pull the event row so admins see the name + status, not just the id.
        ev = db.query(models.LiveEvent).filter(models.LiveEvent.id == event_id).first()
        out.append({
            "event_id":       event_id,
            "name":           ev.name if ev else "",
            "status":         ev.status if ev else "",
            "starts_at":      _iso(ev.starts_at) if ev else None,
            "camera_ids":     cam_ids,
            "workers_count":  int(workers_count),
            "ws_clients":     len(getattr(session, "ws_clients", []) or []),
            "decision_count": int(decision_count),
        })
    return {"running": out}


# ─── Endpoints: audit ────────────────────────────────────────────────────

@router.get("/audit")
def audit(
    db: Session = Depends(get_db),
    _: models.User = Depends(auth.admin_required),
) -> dict:
    """Recent events of interest — helps admins spot incidents quickly.

    We don't have a dedicated audit log table (yet), so this is synthesized
    from existing data:
      * logins          — users ordered by last_login_at desc (top 50)
      * failed jobs     — Jobs with status='failed' (top 50)
      * upload errors   — UploadJobs with non-empty last_error (top 50)
      * admin actions   — agency_audit_log entries (top 20, if any)
    """
    logins = (
        db.query(models.User)
          .filter(models.User.last_login_at.isnot(None))
          .order_by(desc(models.User.last_login_at))
          .limit(50)
          .all()
    )
    failed_jobs = (
        db.query(models.Job)
          .filter(models.Job.status == "failed")
          .order_by(desc(models.Job.created_at))
          .limit(50)
          .all()
    )
    upload_errors = (
        db.query(models.UploadJob)
          .filter(models.UploadJob.last_error != "")
          .order_by(desc(models.UploadJob.updated_at))
          .limit(50)
          .all()
    )
    admin_actions = []
    try:
        rows = (
            db.query(models.AgencyAuditLog)
              .order_by(desc(models.AgencyAuditLog.timestamp))
              .limit(20)
              .all()
        )
        for r in rows:
            admin_actions.append({
                "id":            r.id,
                "agency_id":     r.agency_id,
                "actor_user_id": r.actor_user_id,
                "action":        r.action,
                "target_kind":   r.target_kind,
                "target_id":     r.target_id,
                "timestamp":     _iso(r.timestamp),
            })
    except Exception:
        admin_actions = []

    return {
        "logins": [
            {"id": u.id, "email": u.email, "last_login_at": _iso(u.last_login_at)}
            for u in logins
        ],
        "failed_jobs": [
            {
                "id":         j.id,
                "user_id":    j.user_id,
                "status":     j.status,
                "video_name": j.video_name,
                "error":      (j.error or "")[:500],
                "created_at": _iso(j.created_at),
            }
            for j in failed_jobs
        ],
        "upload_errors": [
            {
                "id":         uj.id,
                "clip_id":    uj.clip_id,
                "channel_id": uj.channel_id,
                "status":     uj.status,
                "attempts":   uj.attempts,
                "last_error": (uj.last_error or "")[:500],
                "updated_at": _iso(uj.updated_at),
            }
            for uj in upload_errors
        ],
        "admin_actions": admin_actions,
    }


# ─── Endpoints: oauth token safety check ──────────────────────────────────
# This is here for completeness — make sure no endpoint ever ships raw
# OAuth tokens.  If a future admin endpoint serializes OAuthToken, it must
# mask these three fields manually.  See auth.encrypt/decrypt helpers.
#
# Sensitive field allow-list (masked to "****" in any admin response):
_MASK_FIELDS = {
    "access_token", "access_token_enc",
    "refresh_token", "refresh_token_enc",
    "client_secret",
}


def _mask_oauth(row: dict) -> dict:
    """Helper: walk a dict and mask any key in `_MASK_FIELDS` to '****'.
    Reserved for future endpoints that expose OAuthToken rows."""
    safe = dict(row)
    for k in list(safe.keys()):
        if k in _MASK_FIELDS:
            safe[k] = _mask()
    return safe
