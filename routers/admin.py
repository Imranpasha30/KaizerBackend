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

import asyncio
import json
from queue import Empty

import psutil
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import case, desc, func
from sqlalchemy.orm import Session

import auth
import models
import system_observer
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


@router.get("/rtmp/activity")
def rtmp_activity(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
    _:  models.User = Depends(auth.admin_required),
) -> dict:
    """Consolidated audit log of the RTMP-live agent.

    Surfaces three rollups, all derived from existing tables (no new
    schema needed):

      * ``totals``      — minted broadcasts, successful pushes, failures,
                          and approximate quota saved vs the regular
                          ``videos.insert`` path
      * ``recent``      — last 25 broadcasts with channel, duration,
                          status, watch URL
      * ``by_channel``  — per-channel counts so you can see which
                          channels are riding the RTMP path

    Data sources:
      - ``YouTubeApiCall`` rows where operation starts with 'live*' or
        is ``videos.insert`` (used to count minted broadcasts and
        compute quota saved).
      - ``UploadJob`` rows where ``upload_provider='native_rtmp'``
        (used for per-broadcast outcomes + recent list).
    """
    cutoff = _utcnow() - timedelta(days=int(days))

    # ── 1) Total broadcasts minted (one per liveBroadcasts.insert) ──
    mint_count = (
        db.query(func.count(models.YouTubeApiCall.id))
          .filter(models.YouTubeApiCall.operation == "liveBroadcasts.insert")
          .filter(models.YouTubeApiCall.created_at >= cutoff)
          .scalar()
    ) or 0

    # ── 2) UploadJob outcomes for the RTMP path ────────────────────
    rtmp_jobs_total = (
        db.query(func.count(models.UploadJob.id))
          .filter(models.UploadJob.upload_provider == "native_rtmp")
          .filter(models.UploadJob.created_at >= cutoff)
          .scalar()
    ) or 0
    rtmp_jobs_done = (
        db.query(func.count(models.UploadJob.id))
          .filter(models.UploadJob.upload_provider == "native_rtmp")
          .filter(models.UploadJob.status == "done")
          .filter(models.UploadJob.created_at >= cutoff)
          .scalar()
    ) or 0
    rtmp_jobs_failed = (
        db.query(func.count(models.UploadJob.id))
          .filter(models.UploadJob.upload_provider == "native_rtmp")
          .filter(models.UploadJob.status == "failed")
          .filter(models.UploadJob.created_at >= cutoff)
          .scalar()
    ) or 0
    rtmp_jobs_inflight = (
        db.query(func.count(models.UploadJob.id))
          .filter(models.UploadJob.upload_provider == "native_rtmp")
          .filter(models.UploadJob.status.in_(["queued", "uploading", "processing"]))
          .filter(models.UploadJob.created_at >= cutoff)
          .scalar()
    ) or 0

    # ── 3) Quota burn comparison (units saved) ─────────────────────
    # Successful native_rtmp jobs would each have cost 1600 units via
    # videos.insert. Actual cost is ~250 (mint + bind + transition +
    # thumbnail). The delta is what you saved.
    UNITS_VIDEOS_INSERT = 1600
    UNITS_RTMP_TYPICAL  = 250
    units_saved = rtmp_jobs_done * (UNITS_VIDEOS_INSERT - UNITS_RTMP_TYPICAL)
    units_spent_rtmp = rtmp_jobs_done * UNITS_RTMP_TYPICAL

    # ── 4) Recent broadcasts (last 25) ─────────────────────────────
    recent_rows = (
        db.query(models.UploadJob, models.Channel)
          .outerjoin(models.Channel, models.Channel.id == models.UploadJob.channel_id)
          .filter(models.UploadJob.upload_provider == "native_rtmp")
          .filter(models.UploadJob.created_at >= cutoff)
          .order_by(desc(models.UploadJob.created_at))
          .limit(25)
          .all()
    )
    recent = []
    for j, ch in recent_rows:
        recent.append({
            "job_id":       int(j.id),
            "clip_id":      int(j.clip_id) if j.clip_id else None,
            "channel_id":   int(j.channel_id) if j.channel_id else None,
            "channel_name": (ch.name if ch else "") or "",
            "video_id":     j.video_id or "",
            "watch_url":    (f"https://www.youtube.com/watch?v={j.video_id}"
                             if j.video_id else ""),
            "status":       j.status,
            "title":        (j.title or "")[:120],
            "bytes_uploaded": int(j.bytes_uploaded or 0),
            "bytes_total":  int(j.bytes_total or 0),
            "attempts":     int(j.attempts or 0),
            "last_error":   (j.last_error or "")[:200],
            "created_at":   _iso(j.created_at),
            "updated_at":   _iso(j.updated_at),
        })

    # ── 5) Per-channel rollup ──────────────────────────────────────
    by_channel_rows = (
        db.query(
            models.Channel.id,
            models.Channel.name,
            func.count(models.UploadJob.id),
            func.sum(
                case((models.UploadJob.status == "done", 1), else_=0)
            ),
            func.sum(
                case((models.UploadJob.status == "failed", 1), else_=0)
            ),
        )
          .join(models.UploadJob, models.UploadJob.channel_id == models.Channel.id)
          .filter(models.UploadJob.upload_provider == "native_rtmp")
          .filter(models.UploadJob.created_at >= cutoff)
          .group_by(models.Channel.id, models.Channel.name)
          .order_by(desc(func.count(models.UploadJob.id)))
          .limit(20)
          .all()
    )
    by_channel = [
        {
            "channel_id":   int(r[0]),
            "channel_name": r[1] or "",
            "total":        int(r[2] or 0),
            "done":         int(r[3] or 0),
            "failed":       int(r[4] or 0),
        }
        for r in by_channel_rows
    ]

    return {
        "window": {
            "days":  int(days),
            "start": _iso(cutoff),
            "end":   _iso(_utcnow()),
        },
        "enabled": (
            os.environ.get("KAIZER_NATIVE_RTMP_ENABLED", "false").lower()
            in ("1", "true", "yes", "on")
        ),
        "concurrency": int(os.environ.get("KAIZER_RTMP_CONCURRENCY", "2") or 2),
        "totals": {
            "broadcasts_minted":  int(mint_count),
            "jobs_total":         int(rtmp_jobs_total),
            "jobs_done":          int(rtmp_jobs_done),
            "jobs_failed":        int(rtmp_jobs_failed),
            "jobs_inflight":      int(rtmp_jobs_inflight),
            "success_rate_pct":   round(
                (rtmp_jobs_done / rtmp_jobs_total * 100.0) if rtmp_jobs_total else 0.0,
                1,
            ),
            "units_spent_rtmp":   int(units_spent_rtmp),
            "units_saved":        int(units_saved),
            "videos_insert_equivalent_cost": int(rtmp_jobs_done * UNITS_VIDEOS_INSERT),
        },
        "recent":     recent,
        "by_channel": by_channel,
    }


@router.get("/system/history")
def system_history(
    hours: float = Query(24.0, ge=0.1, le=24 * 14,
                         description="Window size in hours (max 14d retention)."),
    bucket_minutes: int = Query(0, ge=0, le=240,
                                description="If >0, bucket samples for chart density."),
    db: Session = Depends(get_db),
    _: models.User = Depends(auth.admin_required),
) -> dict:
    """Persistent CPU/RAM/GPU/disk history sampled by ``system_observer``.

    Returns raw rows by default (one per ~30s). Pass ``bucket_minutes=15``
    for a smoother long-window chart; the bucketing happens in Python
    rather than SQL so the dialect doesn't matter.
    """
    cutoff = _utcnow() - timedelta(hours=float(hours))
    rows = (
        db.query(models.SystemMetric)
          .filter(models.SystemMetric.ts >= cutoff)
          .order_by(models.SystemMetric.ts.asc())
          .all()
    )

    def _row(r):
        return {
            "ts":               _iso(r.ts),
            "cpu_percent":      r.cpu_percent,
            "ram_percent":      r.ram_percent,
            "ram_used_gb":      r.ram_used_gb,
            "disk_percent":     r.disk_percent,
            "gpu_util":         r.gpu_util,
            "gpu_mem_used_mb":  r.gpu_mem_used_mb,
            "gpu_mem_total_mb": r.gpu_mem_total_mb,
            "gpu_temp_c":       r.gpu_temp_c,
            "proc_rss_gb":      r.proc_rss_gb,
            "proc_threads":     r.proc_threads,
            "live_events":      r.live_events,
            "net_rx_bps":       r.net_rx_bps,
            "net_tx_bps":       r.net_tx_bps,
            # Kaizer-only rollup (the actual stack footprint, no Chrome/VS Code noise)
            "kaizer_cpu_percent":  getattr(r, "kaizer_cpu_percent",  None),
            "kaizer_rss_gb":       getattr(r, "kaizer_rss_gb",       None),
            "kaizer_proc_count":   getattr(r, "kaizer_proc_count",   None),
            "kaizer_ffmpeg_count": getattr(r, "kaizer_ffmpeg_count", None),
            "kaizer_gpu_util":     getattr(r, "kaizer_gpu_util",     None),
        }

    if bucket_minutes <= 0 or len(rows) < 4:
        samples = [_row(r) for r in rows]
    else:
        # Bucket by floor-to-N-minutes. p95 over each bucket so the chart
        # shows realistic peaks rather than averaging the bursts away.
        bucket_s = bucket_minutes * 60
        buckets: dict[int, list[models.SystemMetric]] = {}
        anchor = rows[0].ts.timestamp() if rows else 0
        for r in rows:
            k = int((r.ts.timestamp() - anchor) // bucket_s)
            buckets.setdefault(k, []).append(r)

        def _pct(values, p):
            xs = sorted([v for v in values if v is not None])
            if not xs:
                return None
            idx = min(len(xs) - 1, max(0, int(round((p / 100) * (len(xs) - 1)))))
            return xs[idx]

        samples = []
        for k in sorted(buckets):
            chunk = buckets[k]
            mid = chunk[len(chunk) // 2]
            samples.append({
                "ts":               _iso(mid.ts),
                # p95 for utilisation curves — captures real peaks.
                "cpu_percent":      _pct([r.cpu_percent  for r in chunk], 95),
                "ram_percent":      _pct([r.ram_percent  for r in chunk], 95),
                "ram_used_gb":      _pct([r.ram_used_gb  for r in chunk], 95),
                "disk_percent":     _pct([r.disk_percent for r in chunk], 95),
                "gpu_util":         _pct([r.gpu_util     for r in chunk], 95),
                "gpu_mem_used_mb":  _pct([r.gpu_mem_used_mb for r in chunk], 95),
                "gpu_mem_total_mb": chunk[-1].gpu_mem_total_mb,
                "gpu_temp_c":       _pct([r.gpu_temp_c   for r in chunk], 95),
                "proc_rss_gb":      _pct([r.proc_rss_gb  for r in chunk], 95),
                "proc_threads":     _pct([r.proc_threads for r in chunk], 95),
                "live_events":      _pct([r.live_events  for r in chunk], 95),
                "net_rx_bps":       _pct([r.net_rx_bps   for r in chunk], 95),
                "net_tx_bps":       _pct([r.net_tx_bps   for r in chunk], 95),
                # Kaizer-only rollup
                "kaizer_cpu_percent":  _pct([getattr(r, "kaizer_cpu_percent",  None) for r in chunk], 95),
                "kaizer_rss_gb":       _pct([getattr(r, "kaizer_rss_gb",       None) for r in chunk], 95),
                "kaizer_proc_count":   _pct([getattr(r, "kaizer_proc_count",   None) for r in chunk], 95),
                "kaizer_ffmpeg_count": _pct([getattr(r, "kaizer_ffmpeg_count", None) for r in chunk], 95),
                "kaizer_gpu_util":     _pct([getattr(r, "kaizer_gpu_util",     None) for r in chunk], 95),
            })

    # Headline summary across the window — drives the capacity tab's KPI tiles.
    def _series_summary(key, scale=1.0):
        xs = sorted([getattr(r, key) for r in rows if getattr(r, key) is not None])
        if not xs:
            return {"avg": None, "p95": None, "max": None}
        avg = sum(xs) / len(xs)
        p95 = xs[min(len(xs) - 1, max(0, int(round(0.95 * (len(xs) - 1)))))]
        return {
            "avg": round(avg * scale, 2),
            "p95": round(p95 * scale, 2),
            "max": round(xs[-1] * scale, 2),
        }

    summary = {
        # Whole-machine context (Chrome / VS Code / etc included)
        "cpu_percent":    _series_summary("cpu_percent"),
        "ram_percent":    _series_summary("ram_percent"),
        "ram_used_gb":    _series_summary("ram_used_gb"),
        "gpu_util":       _series_summary("gpu_util"),
        "gpu_mem_used_mb":_series_summary("gpu_mem_used_mb"),
        "proc_rss_gb":    _series_summary("proc_rss_gb"),
        "live_events":    _series_summary("live_events"),
        # Kaizer-only rollup — the cloud-sizing numbers
        "kaizer_cpu_percent":  _series_summary("kaizer_cpu_percent"),
        "kaizer_rss_gb":       _series_summary("kaizer_rss_gb"),
        "kaizer_proc_count":   _series_summary("kaizer_proc_count"),
        "kaizer_ffmpeg_count": _series_summary("kaizer_ffmpeg_count"),
        "kaizer_gpu_util":     _series_summary("kaizer_gpu_util"),
    }

    return {
        "window": {
            "hours":  float(hours),
            "start":  _iso(cutoff),
            "end":    _iso(_utcnow()),
            "buckets_minutes": int(bucket_minutes),
        },
        "summary":  summary,
        "samples":  samples,
        "raw_count": len(rows),
    }


# ─── Endpoints: live log tail ─────────────────────────────────────────────

@router.get("/logs/recent")
def logs_recent(
    limit: int = Query(500, ge=1, le=2000),
    level: Optional[str] = Query(None, description="Filter to one of: info|warning|error|debug"),
    _: models.User = Depends(auth.admin_required),
) -> dict:
    """Return the most recent N lines from the in-memory ring buffer."""
    items = system_observer.get_buffer().snapshot(limit=int(limit), level=level)
    return {"lines": items, "count": len(items)}


@router.get("/logs/stream")
async def logs_stream(
    request: Request,
    _: models.User = Depends(auth.admin_required),
):
    """SSE endpoint that pushes every new log line as it lands in the buffer.

    Each event payload is a JSON object: ``{id, ts, level, source, line}``.
    Replays nothing — the client should call ``/logs/recent`` first for the
    backlog, then connect here for the live tail.
    """
    buf = system_observer.get_buffer()
    queue = buf.subscribe()

    async def _gen():
        try:
            yield "retry: 3000\n\n"   # tell EventSource to reconnect quickly
            while True:
                if await request.is_disconnected():
                    break
                try:
                    entry = queue.get_nowait()
                except Empty:
                    # Yield a heartbeat comment every 15s so proxies don't kill us.
                    await asyncio.sleep(0.5)
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"
        finally:
            buf.unsubscribe(queue)

    return StreamingResponse(_gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",   # disable nginx/cloudflare buffering
    })


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


# ─── System settings (admin-only key/value config) ───────────────────────────
import system_settings as _ss


@router.get("/settings")
def list_settings(
    db: Session = Depends(get_db),
    _u: models.User = Depends(auth.admin_required),
) -> dict:
    """Return every known setting + its current value (or default)."""
    return {
        "upload_provider": {
            "value":   _ss.get_upload_provider(db),
            "default": _ss.UPLOAD_PROVIDER_DEFAULT,
            "options": sorted(_ss.UPLOAD_PROVIDER_VALID),
            "description": (
                "Which path /api/clips/{id}/publish takes. 'postiz' "
                "routes uploads through the self/cloud-hosted Postiz "
                "instance (covers YouTube + 14 other platforms). "
                "'kaizer' uses our native YouTube OAuth path "
                "(YouTube only). Default is postiz until our app's "
                "verification is complete; admin can flip per-need."
            ),
        },
    }


@router.put("/settings/{key}")
def update_setting(
    key: str,
    payload: dict,
    db: Session = Depends(get_db),
    _u: models.User = Depends(auth.admin_required),
) -> dict:
    """Update one system setting. Body: ``{"value": "<new>"}``."""
    new_val = (payload or {}).get("value", "")
    if not isinstance(new_val, str):
        raise HTTPException(status_code=400, detail="value must be a string")

    if key == _ss.UPLOAD_PROVIDER:
        if new_val.strip().lower() not in _ss.UPLOAD_PROVIDER_VALID:
            raise HTTPException(
                status_code=400,
                detail=f"upload_provider must be one of {sorted(_ss.UPLOAD_PROVIDER_VALID)}",
            )
        new_val = new_val.strip().lower()
    else:
        raise HTTPException(status_code=404,
                            detail=f"Unknown setting: {key}")

    _ss.set_system_setting(db, key, new_val)
    db.commit()
    return {"key": key, "value": new_val}


# Public endpoint — every authenticated user can READ the active
# upload provider (so the publish modal can show "Uploading via X" to
# everyone). Writes still require admin.
@router.get("/settings/upload-provider/public")
def get_upload_provider_public(
    db: Session = Depends(get_db),
    _u: models.User = Depends(auth.current_user),
) -> dict:
    return {"upload_provider": _ss.get_upload_provider(db)}


# ─── Queue / Worker observability (Stage 2 of Redis migration) ────────────

@router.get("/queue/stats")
def admin_queue_stats(_: models.User = Depends(auth.admin_required)) -> dict:
    """Snapshot of the Redis Streams queue state — stream length, PEL,
    per-consumer pending, DLQ size. Returns ``ok=False`` when Redis is
    unreachable (the API still serves so the admin UI shows the error
    rather than 500ing)."""
    from redis_queue import queue_stats
    return queue_stats()


@router.get("/queue/dlq")
def admin_queue_dlq(
    count: int = Query(50, ge=1, le=500),
    _: models.User = Depends(auth.admin_required),
) -> dict:
    """List recent DLQ entries (newest first). Use ``count`` to scope
    the page size — defaults to 50."""
    from redis_queue import list_dlq, QueueError
    try:
        return {"entries": list_dlq(count=count)}
    except QueueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/queue/dlq/{message_id}/replay")
def admin_queue_replay(
    message_id: str,
    _: models.User = Depends(auth.admin_required),
) -> dict:
    """Pop the named DLQ entry and re-enqueue its job_id to the main
    upload stream. The message_id format is the Redis stream id
    (e.g. ``1778312608609-0``) — get it from ``GET /queue/dlq``.

    Returns the new stream message id of the replay so admin tools can
    correlate."""
    from redis_queue import replay_from_dlq, QueueError
    try:
        return replay_from_dlq(message_id)
    except QueueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ─── Gemini cache (Stage 4 of Redis migration) ──────────────────────────

@router.get("/cache/gemini")
def admin_gemini_cache_stats(_: models.User = Depends(auth.admin_required)) -> dict:
    """Per-kind cache hits/misses/writes + a totals roll-up.

    Use the hit_rate to decide whether a kind is paying its keep —
    < 30% probably means the cache key is too narrow (over-keying on
    something that should be canonicalised) or the temperature is
    too high (creative variation defeating the cache).
    """
    import gemini_cache
    return gemini_cache.stats()


@router.delete("/cache/gemini")
def admin_gemini_cache_reset(
    kind: Optional[str] = Query(None,
        description="Specific kind to wipe ('video', 'translation', etc.) — "
                    "omit to reset all kinds."),
    _: models.User = Depends(auth.admin_required),
) -> dict:
    """Wipe cached Gemini responses. Use after a prompt change or model
    upgrade to make sure stale cached outputs aren't served."""
    import gemini_cache
    if kind and kind not in gemini_cache.KINDS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown cache kind {kind!r}. "
                                   f"Valid: {sorted(gemini_cache.KINDS.keys())}")
    deleted = gemini_cache.reset(kind)
    return {"ok": True, "kind": kind or "*", "keys_deleted": deleted}


# ─── Per-tenant rate-limit inspection (Stage 5) ─────────────────────────

@router.get("/rate-limits/buckets")
def admin_rate_limit_config(_: models.User = Depends(auth.admin_required)) -> dict:
    """Return the plan→bucket→(burst, rate_per_s) config table.

    The frontend uses this to render a "your plan's caps" panel; ops
    use it to confirm a rate-limit change rolled out before debugging
    a 429."""
    import rate_limit
    return {
        "default_plan": rate_limit.DEFAULT_PLAN,
        "limits": {
            plan: {
                bucket: {"burst": cfg[0], "rate_per_s": cfg[1]}
                for bucket, cfg in buckets.items()
            }
            for plan, buckets in rate_limit.PLAN_LIMITS.items()
        },
        "auth_ip": {
            "burst":      rate_limit.AUTH_IP_BURST,
            "rate_per_s": rate_limit.AUTH_IP_RATE,
        },
    }


@router.get("/rate-limits/state")
def admin_rate_limit_state(
    bucket: str = Query("create"),
    plan:   str = Query("free"),
    who:    str = Query(..., description="user:<id> or ip:<addr>"),
    _:      models.User = Depends(auth.admin_required),
) -> dict:
    """Inspect a tenant's current bucket state — supports the "why am
    I being rate-limited?" support flow without granting Redis-CLI
    access. ``who`` matches the format the limiter writes:
    ``user:<id>`` or ``ip:<addr>``."""
    import rate_limit
    return rate_limit.bucket_state(bucket, plan, who)


@router.delete("/rate-limits/state")
def admin_rate_limit_reset(
    bucket: str = Query("create"),
    plan:   str = Query("free"),
    who:    str = Query(..., description="user:<id> or ip:<addr>"),
    _:      models.User = Depends(auth.admin_required),
) -> dict:
    """Wipe a tenant's bucket. Use during incident response to unblock
    a customer whose legitimate burst tripped the cap."""
    import rate_limit
    deleted = rate_limit.reset_bucket(bucket, plan, who)
    return {"ok": True, "deleted": deleted, "key": f"kaizer:rl:{bucket}:{plan}:{who}"}


# ─── Storage promotion (local → R2) ─────────────────────────────────────
#
# Backs the admin "Promote to R2" button. Walks every Clip + UserAsset
# row whose storage_backend='local' (or whose thumb/image URL begins
# with /media/) and uploads the underlying file to R2 with the same
# key, then updates the row. Idempotent + dry-run-by-default.

@router.post("/storage/promote-to-r2")
def admin_storage_promote_to_r2(
    dry_run: bool = Query(True, description="When True (default), nothing is uploaded or written. Set to False for the live run."),
    db: Session = Depends(get_db),
    _:  models.User = Depends(auth.admin_required),
) -> dict:
    """Promote every local-stored Clip + UserAsset row to R2.

    Run dry-run FIRST. Inspect the returned ``totals`` and ``tables``
    fields, then re-run with ``?dry_run=false`` to commit. Safe to
    re-run after a partial failure — rows already on R2 are skipped.
    """
    from storage_migration import promote_local_to_r2
    return promote_local_to_r2(db, dry_run=dry_run)


# ─── Phase 13 — Unified Usage Dashboard ────────────────────────────────
# One endpoint, one JSON payload, all the breakdowns the UI needs.
#
# Returning everything in one call lets the dashboard render with a
# single HTTP round-trip and ensures every panel reflects the same
# time window — no risk of the "Today" tile lagging the daily chart
# by a polling cycle.

# Daily YouTube Data API cap on a standard project.  Used to compute
# the percentage-of-cap-used gauge.  When the user upgrades their
# Google Cloud quota grant, bump this via env (KAIZER_YT_DAILY_CAP).
_YT_DAILY_CAP = int(os.environ.get("KAIZER_YT_DAILY_CAP", "10000"))


@router.get("/usage/dashboard")
def usage_dashboard(
    days: int = Query(30, ge=1, le=365, description="History window in days. Defaults to 30."),
    db: Session = Depends(get_db),
    _:  models.User = Depends(auth.admin_required),
) -> dict:
    """All-providers usage dashboard payload.

    Returns one dict with these top-level sections:
      * window        — period summary (start/end dates, days)
      * overview      — totals per provider for the window + today
      * timeseries    — daily per-provider totals for the line chart
      * gemini        — purpose / model / user / job / video breakdowns
      * openai        — purpose / model / user / job breakdowns
      * youtube       — operation / channel / video breakdowns + cap %
      * recent_calls  — last 25 calls per provider for the live tail
    """
    now      = _utcnow()
    window_start = now - timedelta(days=int(days))
    today_start  = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # ─── Helpers ────────────────────────────────────────────────
    def _scalar(q):
        v = q.scalar()
        return float(v or 0)

    def _scalar_int(q):
        v = q.scalar()
        return int(v or 0)

    def _row_to_dict(headers, rows):
        return [dict(zip(headers, r)) for r in rows]

    # ─── Overview totals ────────────────────────────────────────
    def _gemini_total(start):
        return _scalar(
            db.query(func.coalesce(func.sum(models.GeminiCall.cost_usd), 0.0))
              .filter(models.GeminiCall.created_at >= start)
        )

    def _openai_total(start):
        return _scalar(
            db.query(func.coalesce(func.sum(models.OpenAiCall.cost_usd), 0.0))
              .filter(models.OpenAiCall.created_at >= start)
        )

    def _yt_quota_total(start):
        return _scalar_int(
            db.query(func.coalesce(func.sum(models.YouTubeApiCall.quota_cost), 0))
              .filter(models.YouTubeApiCall.created_at >= start)
              .filter(models.YouTubeApiCall.success == True)  # only successful calls burn quota
        )

    def _gemini_calls(start):
        return _scalar_int(
            db.query(func.count(models.GeminiCall.id))
              .filter(models.GeminiCall.created_at >= start)
        )

    def _openai_calls(start):
        return _scalar_int(
            db.query(func.count(models.OpenAiCall.id))
              .filter(models.OpenAiCall.created_at >= start)
        )

    def _yt_calls(start):
        return _scalar_int(
            db.query(func.count(models.YouTubeApiCall.id))
              .filter(models.YouTubeApiCall.created_at >= start)
        )

    overview = {
        "gemini": {
            "today_cost_usd":  round(_gemini_total(today_start),  4),
            "window_cost_usd": round(_gemini_total(window_start), 4),
            "today_calls":     _gemini_calls(today_start),
            "window_calls":    _gemini_calls(window_start),
        },
        "openai": {
            "today_cost_usd":  round(_openai_total(today_start),  4),
            "window_cost_usd": round(_openai_total(window_start), 4),
            "today_calls":     _openai_calls(today_start),
            "window_calls":    _openai_calls(window_start),
        },
        "youtube": {
            "today_quota":   _yt_quota_total(today_start),
            "window_quota":  _yt_quota_total(window_start),
            "today_calls":   _yt_calls(today_start),
            "window_calls":  _yt_calls(window_start),
            "daily_cap":     _YT_DAILY_CAP,
            "today_pct":     round(100.0 * _yt_quota_total(today_start) / max(_YT_DAILY_CAP, 1), 1),
        },
        "total_cost_today_usd":  round(
            _gemini_total(today_start) + _openai_total(today_start), 4
        ),
        "total_cost_window_usd": round(
            _gemini_total(window_start) + _openai_total(window_start), 4
        ),
    }

    # ─── Daily timeseries (last N days) ─────────────────────────
    # Postgres date_trunc; SQLAlchemy bridges to SQLite's strftime via
    # the func layer (we use date() to be dialect-agnostic).
    day_col_g = func.date(models.GeminiCall.created_at).label("d")
    day_col_o = func.date(models.OpenAiCall.created_at).label("d")
    day_col_y = func.date(models.YouTubeApiCall.created_at).label("d")

    g_rows = (
        db.query(
            day_col_g,
            func.coalesce(func.sum(models.GeminiCall.cost_usd),     0.0),
            func.coalesce(func.sum(models.GeminiCall.total_tokens), 0),
            func.count(models.GeminiCall.id),
        )
        .filter(models.GeminiCall.created_at >= window_start)
        .group_by(day_col_g).order_by(day_col_g).all()
    )
    o_rows = (
        db.query(
            day_col_o,
            func.coalesce(func.sum(models.OpenAiCall.cost_usd),    0.0),
            func.coalesce(func.sum(models.OpenAiCall.image_count), 0),
            func.count(models.OpenAiCall.id),
        )
        .filter(models.OpenAiCall.created_at >= window_start)
        .group_by(day_col_o).order_by(day_col_o).all()
    )
    y_rows = (
        db.query(
            day_col_y,
            func.coalesce(func.sum(models.YouTubeApiCall.quota_cost), 0),
            func.count(models.YouTubeApiCall.id),
        )
        .filter(models.YouTubeApiCall.created_at >= window_start)
        .filter(models.YouTubeApiCall.success == True)
        .group_by(day_col_y).order_by(day_col_y).all()
    )

    timeseries = {
        "gemini": _row_to_dict(("day","cost_usd","tokens","calls"), [
            (str(r[0]), round(float(r[1] or 0), 4), int(r[2] or 0), int(r[3] or 0))
            for r in g_rows
        ]),
        "openai": _row_to_dict(("day","cost_usd","images","calls"), [
            (str(r[0]), round(float(r[1] or 0), 4), int(r[2] or 0), int(r[3] or 0))
            for r in o_rows
        ]),
        "youtube": _row_to_dict(("day","quota","calls"), [
            (str(r[0]), int(r[1] or 0), int(r[2] or 0))
            for r in y_rows
        ]),
    }

    # ─── Gemini breakdowns ─────────────────────────────────────
    g_by_purpose = (
        db.query(
            models.GeminiCall.purpose,
            func.coalesce(func.sum(models.GeminiCall.total_tokens), 0),
            func.coalesce(func.sum(models.GeminiCall.cost_usd),     0.0),
            func.count(models.GeminiCall.id),
        )
        .filter(models.GeminiCall.created_at >= window_start)
        .group_by(models.GeminiCall.purpose)
        .order_by(desc(func.sum(models.GeminiCall.cost_usd)))
        .all()
    )
    g_by_model = (
        db.query(
            models.GeminiCall.model,
            func.coalesce(func.sum(models.GeminiCall.total_tokens), 0),
            func.coalesce(func.sum(models.GeminiCall.cost_usd),     0.0),
            func.count(models.GeminiCall.id),
        )
        .filter(models.GeminiCall.created_at >= window_start)
        .group_by(models.GeminiCall.model)
        .order_by(desc(func.sum(models.GeminiCall.cost_usd)))
        .all()
    )
    g_top_users = (
        db.query(
            models.GeminiCall.user_id,
            models.User.email,
            func.coalesce(func.sum(models.GeminiCall.cost_usd),     0.0),
            func.coalesce(func.sum(models.GeminiCall.total_tokens), 0),
            func.count(models.GeminiCall.id),
        )
        .outerjoin(models.User, models.User.id == models.GeminiCall.user_id)
        .filter(models.GeminiCall.created_at >= window_start)
        .group_by(models.GeminiCall.user_id, models.User.email)
        .order_by(desc(func.sum(models.GeminiCall.cost_usd)))
        .limit(10).all()
    )
    g_top_jobs = (
        db.query(
            models.GeminiCall.job_id,
            func.coalesce(func.sum(models.GeminiCall.cost_usd),         0.0),
            func.coalesce(func.sum(models.GeminiCall.total_tokens),     0),
            func.coalesce(func.sum(models.GeminiCall.file_bytes),       0),
            func.coalesce(func.sum(models.GeminiCall.video_duration_s), 0.0),
            func.count(models.GeminiCall.id),
        )
        .filter(models.GeminiCall.created_at >= window_start)
        .filter(models.GeminiCall.job_id.isnot(None))
        .group_by(models.GeminiCall.job_id)
        .order_by(desc(func.sum(models.GeminiCall.cost_usd)))
        .limit(15).all()
    )
    # Video-size & duration buckets so the user can see the
    # cost-per-byte and cost-per-second story.
    g_video_only = (
        db.query(
            func.coalesce(func.sum(models.GeminiCall.cost_usd),         0.0),
            func.coalesce(func.sum(models.GeminiCall.file_bytes),       0),
            func.coalesce(func.sum(models.GeminiCall.video_duration_s), 0.0),
            func.count(models.GeminiCall.id),
        )
        .filter(models.GeminiCall.created_at >= window_start)
        .filter(models.GeminiCall.file_bytes > 0)
        .one()
    )

    gemini = {
        "by_purpose": _row_to_dict(("purpose","tokens","cost_usd","calls"), [
            (r[0] or "(unset)", int(r[1] or 0), round(float(r[2] or 0), 4), int(r[3] or 0))
            for r in g_by_purpose
        ]),
        "by_model": _row_to_dict(("model","tokens","cost_usd","calls"), [
            (r[0] or "(unknown)", int(r[1] or 0), round(float(r[2] or 0), 4), int(r[3] or 0))
            for r in g_by_model
        ]),
        "top_users": _row_to_dict(("user_id","email","cost_usd","tokens","calls"), [
            (int(r[0]) if r[0] else None, r[1] or "(system)",
             round(float(r[2] or 0), 4), int(r[3] or 0), int(r[4] or 0))
            for r in g_top_users
        ]),
        "top_jobs": _row_to_dict(
            ("job_id","cost_usd","tokens","file_bytes","duration_s","calls"),
            [
                (int(r[0]), round(float(r[1] or 0), 4), int(r[2] or 0),
                 int(r[3] or 0), round(float(r[4] or 0), 1), int(r[5] or 0))
                for r in g_top_jobs
            ],
        ),
        "video_uploads": {
            "cost_usd":      round(float(g_video_only[0] or 0), 4),
            "total_bytes":   int(g_video_only[1] or 0),
            "total_seconds": round(float(g_video_only[2] or 0), 1),
            "call_count":    int(g_video_only[3] or 0),
            "cost_per_minute_usd": round(
                (float(g_video_only[0] or 0) /
                 max(float(g_video_only[2] or 0) / 60.0, 0.001)),
                5,
            ) if g_video_only[2] else 0.0,
        },
    }

    # ─── OpenAI breakdowns ─────────────────────────────────────
    o_by_purpose = (
        db.query(
            models.OpenAiCall.purpose,
            func.coalesce(func.sum(models.OpenAiCall.image_count), 0),
            func.coalesce(func.sum(models.OpenAiCall.cost_usd),    0.0),
            func.count(models.OpenAiCall.id),
        )
        .filter(models.OpenAiCall.created_at >= window_start)
        .group_by(models.OpenAiCall.purpose)
        .order_by(desc(func.sum(models.OpenAiCall.cost_usd)))
        .all()
    )
    o_by_model = (
        db.query(
            models.OpenAiCall.model,
            func.coalesce(func.sum(models.OpenAiCall.image_count), 0),
            func.coalesce(func.sum(models.OpenAiCall.cost_usd),    0.0),
            func.count(models.OpenAiCall.id),
        )
        .filter(models.OpenAiCall.created_at >= window_start)
        .group_by(models.OpenAiCall.model)
        .order_by(desc(func.sum(models.OpenAiCall.cost_usd)))
        .all()
    )
    o_top_users = (
        db.query(
            models.OpenAiCall.user_id,
            models.User.email,
            func.coalesce(func.sum(models.OpenAiCall.cost_usd),    0.0),
            func.coalesce(func.sum(models.OpenAiCall.image_count), 0),
            func.count(models.OpenAiCall.id),
        )
        .outerjoin(models.User, models.User.id == models.OpenAiCall.user_id)
        .filter(models.OpenAiCall.created_at >= window_start)
        .group_by(models.OpenAiCall.user_id, models.User.email)
        .order_by(desc(func.sum(models.OpenAiCall.cost_usd)))
        .limit(10).all()
    )
    o_top_jobs = (
        db.query(
            models.OpenAiCall.job_id,
            func.coalesce(func.sum(models.OpenAiCall.cost_usd),    0.0),
            func.coalesce(func.sum(models.OpenAiCall.image_count), 0),
            func.count(models.OpenAiCall.id),
        )
        .filter(models.OpenAiCall.created_at >= window_start)
        .filter(models.OpenAiCall.job_id.isnot(None))
        .group_by(models.OpenAiCall.job_id)
        .order_by(desc(func.sum(models.OpenAiCall.cost_usd)))
        .limit(15).all()
    )

    openai = {
        "by_purpose": _row_to_dict(("purpose","images","cost_usd","calls"), [
            (r[0] or "(unset)", int(r[1] or 0), round(float(r[2] or 0), 4), int(r[3] or 0))
            for r in o_by_purpose
        ]),
        "by_model": _row_to_dict(("model","images","cost_usd","calls"), [
            (r[0] or "(unknown)", int(r[1] or 0), round(float(r[2] or 0), 4), int(r[3] or 0))
            for r in o_by_model
        ]),
        "top_users": _row_to_dict(("user_id","email","cost_usd","images","calls"), [
            (int(r[0]) if r[0] else None, r[1] or "(system)",
             round(float(r[2] or 0), 4), int(r[3] or 0), int(r[4] or 0))
            for r in o_top_users
        ]),
        "top_jobs": _row_to_dict(("job_id","cost_usd","images","calls"), [
            (int(r[0]), round(float(r[1] or 0), 4), int(r[2] or 0), int(r[3] or 0))
            for r in o_top_jobs
        ]),
    }

    # ─── YouTube breakdowns ────────────────────────────────────
    y_by_op = (
        db.query(
            models.YouTubeApiCall.operation,
            func.coalesce(func.sum(models.YouTubeApiCall.quota_cost), 0),
            func.count(models.YouTubeApiCall.id),
        )
        .filter(models.YouTubeApiCall.created_at >= window_start)
        .filter(models.YouTubeApiCall.success == True)
        .group_by(models.YouTubeApiCall.operation)
        .order_by(desc(func.sum(models.YouTubeApiCall.quota_cost)))
        .all()
    )
    y_by_channel = (
        db.query(
            models.YouTubeApiCall.google_channel_id,
            func.coalesce(func.sum(models.YouTubeApiCall.quota_cost), 0),
            func.count(models.YouTubeApiCall.id),
        )
        .filter(models.YouTubeApiCall.created_at >= window_start)
        .filter(models.YouTubeApiCall.success == True)
        .filter(models.YouTubeApiCall.google_channel_id != "")
        .group_by(models.YouTubeApiCall.google_channel_id)
        .order_by(desc(func.sum(models.YouTubeApiCall.quota_cost)))
        .limit(10).all()
    )
    y_by_kind = (
        db.query(
            models.YouTubeApiCall.publish_kind,
            func.coalesce(func.sum(models.YouTubeApiCall.quota_cost),       0),
            func.coalesce(func.sum(models.YouTubeApiCall.file_bytes),       0),
            func.coalesce(func.sum(models.YouTubeApiCall.duration_seconds), 0.0),
            func.count(models.YouTubeApiCall.id),
        )
        .filter(models.YouTubeApiCall.created_at >= window_start)
        .filter(models.YouTubeApiCall.success == True)
        .filter(models.YouTubeApiCall.publish_kind != "")
        .group_by(models.YouTubeApiCall.publish_kind)
        .order_by(desc(func.sum(models.YouTubeApiCall.quota_cost)))
        .all()
    )
    y_top_videos = (
        db.query(
            models.YouTubeApiCall.video_id,
            models.YouTubeApiCall.publish_kind,
            func.coalesce(func.sum(models.YouTubeApiCall.quota_cost), 0),
            func.coalesce(func.max(models.YouTubeApiCall.file_bytes), 0),
            func.coalesce(func.max(models.YouTubeApiCall.duration_seconds), 0.0),
            func.count(models.YouTubeApiCall.id),
        )
        .filter(models.YouTubeApiCall.created_at >= window_start)
        .filter(models.YouTubeApiCall.success == True)
        .filter(models.YouTubeApiCall.video_id != "")
        .group_by(models.YouTubeApiCall.video_id, models.YouTubeApiCall.publish_kind)
        .order_by(desc(func.sum(models.YouTubeApiCall.quota_cost)))
        .limit(15).all()
    )

    youtube = {
        "by_operation": _row_to_dict(("operation","quota","calls"), [
            (r[0] or "(unknown)", int(r[1] or 0), int(r[2] or 0))
            for r in y_by_op
        ]),
        "by_channel": _row_to_dict(("google_channel_id","quota","calls"), [
            (r[0], int(r[1] or 0), int(r[2] or 0))
            for r in y_by_channel
        ]),
        "by_publish_kind": _row_to_dict(
            ("kind","quota","total_bytes","total_seconds","calls"),
            [
                (r[0] or "(unset)", int(r[1] or 0), int(r[2] or 0),
                 round(float(r[3] or 0), 1), int(r[4] or 0))
                for r in y_by_kind
            ],
        ),
        "top_videos": _row_to_dict(
            ("video_id","kind","quota","file_bytes","duration_seconds","calls"),
            [
                (r[0], r[1] or "(unset)", int(r[2] or 0), int(r[3] or 0),
                 round(float(r[4] or 0), 1), int(r[5] or 0))
                for r in y_top_videos
            ],
        ),
    }

    # ─── Recent calls tail (last 25 per provider) ─────────────
    def _gemini_recent():
        rows = (
            db.query(models.GeminiCall)
              .order_by(desc(models.GeminiCall.created_at))
              .limit(25).all()
        )
        return [{
            "id":         r.id,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "user_id":    r.user_id,
            "job_id":     r.job_id,
            "clip_id":    r.clip_id,
            "model":      r.model,
            "purpose":    r.purpose,
            "tokens":     int(r.total_tokens or 0),
            "cost_usd":   round(float(r.cost_usd or 0), 5),
            "latency_ms": int(r.latency_ms or 0),
            "status":     r.status,
        } for r in rows]

    def _openai_recent():
        rows = (
            db.query(models.OpenAiCall)
              .order_by(desc(models.OpenAiCall.created_at))
              .limit(25).all()
        )
        return [{
            "id":         r.id,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "user_id":    r.user_id,
            "job_id":     r.job_id,
            "clip_id":    r.clip_id,
            "model":      r.model,
            "purpose":    r.purpose,
            "image_size":    r.image_size,
            "image_quality": r.image_quality,
            "image_count":   int(r.image_count or 0),
            "cost_usd":      round(float(r.cost_usd or 0), 5),
            "latency_ms":    int(r.latency_ms or 0),
            "status":        r.status,
        } for r in rows]

    def _youtube_recent():
        rows = (
            db.query(models.YouTubeApiCall)
              .order_by(desc(models.YouTubeApiCall.created_at))
              .limit(25).all()
        )
        return [{
            "id":           r.id,
            "created_at":   r.created_at.isoformat() if r.created_at else None,
            "user_id":      r.user_id,
            "job_id":       r.job_id,
            "channel_id":   r.channel_id,
            "google_channel_id": r.google_channel_id,
            "video_id":     r.video_id,
            "operation":    r.operation,
            "quota_cost":   int(r.quota_cost or 0),
            "publish_kind": r.publish_kind,
            "file_bytes":   int(r.file_bytes or 0),
            "duration_seconds": round(float(r.duration_seconds or 0), 1),
            "success":      bool(r.success),
            "http_status":  int(r.http_status or 0),
        } for r in rows]

    return {
        "window": {
            "days":  int(days),
            "start": window_start.isoformat(),
            "end":   now.isoformat(),
        },
        "overview":     overview,
        "timeseries":   timeseries,
        "gemini":       gemini,
        "openai":       openai,
        "youtube":      youtube,
        "recent_calls": {
            "gemini":  _gemini_recent(),
            "openai":  _openai_recent(),
            "youtube": _youtube_recent(),
        },
    }
