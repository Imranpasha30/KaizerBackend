"""Publish Tasks v2 router — POST /api/publish-tasks (feature-flagged).

The legacy publish path at ``POST /api/clips/:id/publish`` (in
``routers/youtube_upload.py``) stays untouched. This router is
purely additive and gated behind the ``KAIZER_NEW_PUBLISH_PATH=1``
env var per CONTRACTS.md §2; until the flag is flipped, every POST
returns 503 with a stable error code so callers can detect the
disabled state without parsing strings.

Conventions match ``routers/youtube_upload.py``:
  - ``db: Session = Depends(get_db)`` for DB injection
  - ``user: models.User = Depends(auth.current_user)`` for auth
  - error bodies use ``{code: ..., message: ..., ...}`` shape so the
    frontend can switch on ``code`` without scraping ``message``.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import bindparam as _sql_bindparam
from sqlalchemy import text as _sql_text
from sqlalchemy.orm import Session

import auth
import models
from database import get_db
from rate_limit import rate_limited as _rate_limited  # Wave 2 — same dependency youtube_upload uses
from services import credits as credits_svc
from services import fanout as fanout_svc

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["publish-tasks-v2"])


# ─── Feature flag ─────────────────────────────────────────────────────────


_FLAG_NAME = "KAIZER_NEW_PUBLISH_PATH"


def _new_path_enabled() -> bool:
    """Re-read every request so an ops flip doesn't require a restart."""
    return (os.environ.get(_FLAG_NAME, "0") or "0").strip() == "1"


def _ensure_enabled() -> None:
    if not _new_path_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "new_publish_path_disabled",
                "message": (
                    "POST /api/publish-tasks is feature-flag-gated. "
                    f"Set {_FLAG_NAME}=1 in env to enable."
                ),
            },
        )


# ─── Request / response schemas ───────────────────────────────────────────


class PublishVersionInputs(BaseModel):
    """Caller-supplied opaque version strings (Decision 10).

    Phase 2 F-agent defines the canonical ``seo_version`` formula;
    in Phase 1 it's just a stable string the caller computes.
    """
    brand_profile_version: str = Field(..., min_length=1, max_length=64)
    seo_version: str = Field(..., min_length=1, max_length=64)
    metadata_version: str = Field(..., min_length=1, max_length=64)


class PublishTaskTargetIn(BaseModel):
    channel_id: int
    upload_path: Literal["direct", "rtmp"]
    publish_kind: Literal["video", "short"]
    brand_profile_id: Optional[int] = None
    thumbnail_source: Optional[Literal["pipeline_generated", "user_override", "user_uploaded"]] = None
    thumbnail_r2_key: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    publish_version_inputs: PublishVersionInputs


class PublishTaskCreate(BaseModel):
    master_video_id: int
    targets: List[PublishTaskTargetIn] = Field(..., min_length=1)
    priority: Literal["critical", "high", "normal", "low"] = "normal"

    @field_validator("targets")
    @classmethod
    def _at_least_one(cls, v: List[PublishTaskTargetIn]) -> List[PublishTaskTargetIn]:
        if not v:
            raise ValueError("at least one target required")
        return v


class PublishTaskCreated(BaseModel):
    publish_task_id: int
    upload_job_ids: List[int]
    predicted_credit_total: int
    predicted_quota_units_total: int
    quota_pre_flight_ok: bool
    deduped: bool = False  # True iff the request matched an existing task


class UploadJobV2Out(BaseModel):
    id: int
    publish_task_id: int
    channel_id: int
    upload_path: str
    publish_kind: str
    status: str
    idempotency_key: str
    publish_version: str
    predicted_quota_units: int
    predicted_credit_cost: int
    youtube_video_id: Optional[str] = None
    thumbnail_source: Optional[str] = None
    thumbnail_r2_key: Optional[str] = None
    brand_profile_id: Optional[int] = None
    oauth_token_id: Optional[int] = None
    priority_at_dispatch: Optional[str] = None
    attempts: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    # ── Audit-view enrichment (additive — job-wise Publishes UI) ──
    channel_name: Optional[str] = None
    channel_handle: Optional[str] = None
    video_url: Optional[str] = None          # https://youtu.be/{id}
    last_error: str = ""
    bytes_uploaded: int = 0
    next_attempt_at: Optional[datetime] = None  # "retrying in ~Ns" hint
    branded: bool = False                    # branded artifact cached


class PublishTaskOut(BaseModel):
    id: int
    user_id: int
    master_video_id: int
    priority: str
    status: str
    target_count: int
    completed_count: int
    failed_count: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    upload_jobs: List[UploadJobV2Out] = Field(default_factory=list)


class PublishTaskListItem(BaseModel):
    id: int
    master_video_id: int
    priority: str
    status: str
    target_count: int
    completed_count: int
    failed_count: int
    created_at: Optional[datetime] = None
    # ── Audit-view enrichment (additive — job-wise Publishes UI) ──
    video_title: Optional[str] = None        # from the source Clip's SEO/text
    thumb_url: Optional[str] = None          # clip thumb (storage URL or /api/file/)
    publish_kind: Optional[str] = None       # video | short (first child)
    channels_preview: List[str] = Field(default_factory=list)  # first 3 names
    # Live child-status rollup (completed/failed come from the counters
    # above; these three cover the in-between states the cards show):
    parked_count: int = 0
    retrying_count: int = 0                  # queued w/ attempts>0 (backoff)
    in_flight_count: int = 0                 # queued/claimed/branding/uploading
    cancelled_count: int = 0
    # Grouping + delivery (job-wise Publishes UI): PublishTasks that share a
    # source_job_id came from the SAME render (the full video + its shorts);
    # delivery = the route(s) these channels publish through
    # ("direct"|"rtmp"|"postiz"|"mixed") so the card shows HOW it's going out.
    source_job_id: Optional[int] = None
    delivery: str = ""


class PublishTaskList(BaseModel):
    items: List[PublishTaskListItem]
    limit: int
    offset: int
    total: int


# ─── Serialisers ─────────────────────────────────────────────────────────


def _serialise_job(
    job: models.UploadJobV2,
    channel: Optional[models.Channel] = None,
) -> UploadJobV2Out:
    vid = (job.youtube_video_id or "").strip()
    return UploadJobV2Out(
        id=int(job.id),
        publish_task_id=int(job.publish_task_id),
        channel_id=int(job.channel_id) if job.channel_id is not None else 0,
        upload_path=str(job.upload_path),
        publish_kind=str(job.publish_kind),
        status=str(job.status),
        idempotency_key=str(job.idempotency_key),
        publish_version=str(job.publish_version),
        predicted_quota_units=int(job.predicted_quota_units),
        predicted_credit_cost=int(job.predicted_credit_cost),
        youtube_video_id=job.youtube_video_id,
        thumbnail_source=job.thumbnail_source,
        thumbnail_r2_key=job.thumbnail_r2_key,
        brand_profile_id=job.brand_profile_id,
        oauth_token_id=job.oauth_token_id,
        priority_at_dispatch=job.priority_at_dispatch,
        attempts=int(job.attempts or 0),
        created_at=job.created_at,
        updated_at=job.updated_at,
        # Audit-view enrichment.
        channel_name=(channel.name if channel else None),
        channel_handle=(getattr(channel, "handle", None) if channel else None),
        video_url=(f"https://youtu.be/{vid}" if vid else None),
        last_error=(job.last_error or ""),
        bytes_uploaded=int(job.bytes_uploaded or 0),
        next_attempt_at=getattr(job, "next_attempt_at", None),
        branded=bool((job.branded_artifact_r2_key or "").strip()),
    )


def _clip_display(db: Session, clip_ids: List[int]) -> dict:
    """{clip_id: {"title": ..., "thumb_url": ...}} for the card header.

    Title preference: parsed SEO title → clip.text (first line) →
    filename. Thumb mirrors the legacy _to_dict logic in
    routers/youtube_upload.py (storage URL first, /api/file/ fallback).
    """
    out: dict = {}
    ids = [int(c) for c in clip_ids if c]
    if not ids:
        return out
    clips = db.query(models.Clip).filter(models.Clip.id.in_(ids)).all()
    for clip in clips:
        title = ""
        if clip.seo:
            try:
                parsed = json.loads(clip.seo)
                if isinstance(parsed, dict):
                    title = str(parsed.get("title") or "").strip()
            except Exception:
                pass
        if not title and (clip.text or "").strip():
            title = (clip.text or "").strip().splitlines()[0][:120]
        if not title:
            title = clip.filename or f"Clip #{clip.id}"
        thumb = (
            (getattr(clip, "thumb_storage_url", "") or "")
            or (f"/api/file/?path={clip.thumb_path}" if clip.thumb_path else "")
        )
        out[int(clip.id)] = {"title": title, "thumb_url": (thumb or None)}
    return out


def _serialise_task(task: models.PublishTask, jobs: List[models.UploadJobV2]) -> PublishTaskOut:
    return PublishTaskOut(
        id=int(task.id),
        user_id=int(task.user_id),
        master_video_id=int(task.master_video_id),
        priority=str(task.priority),
        status=str(task.status),
        target_count=int(task.target_count or 0),
        completed_count=int(task.completed_count or 0),
        failed_count=int(task.failed_count or 0),
        created_at=task.created_at,
        updated_at=task.updated_at,
        upload_jobs=[_serialise_job(j) for j in jobs],
    )


def _build_existing_result(
    db: Session, publish_task_id: int
) -> PublishTaskCreated:
    """When an idempotency-key collision means a PublishTask already
    exists, return its current rollup so the client sees the same
    response shape as a fresh-create (brief §2: dedupe-by-design)."""
    task = (
        db.query(models.PublishTask)
        .filter(models.PublishTask.id == publish_task_id)
        .first()
    )
    if task is None:
        # Shouldn't happen — but if the row vanished mid-flight, surface 409.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "duplicate_publish_version_lost_parent",
                "message": (
                    f"upload_jobs_v2 row exists pointing at PublishTask id="
                    f"{publish_task_id} which is no longer in the DB"
                ),
            },
        )
    jobs = (
        db.query(models.UploadJobV2)
        .filter(models.UploadJobV2.publish_task_id == publish_task_id)
        .order_by(models.UploadJobV2.id.asc())
        .all()
    )
    return PublishTaskCreated(
        publish_task_id=int(task.id),
        upload_job_ids=[int(j.id) for j in jobs],
        predicted_credit_total=sum(int(j.predicted_credit_cost) for j in jobs),
        predicted_quota_units_total=sum(int(j.predicted_quota_units) for j in jobs),
        quota_pre_flight_ok=True,
        deduped=True,
    )


# ─── POST /api/publish-tasks ──────────────────────────────────────────────


def _to_dataclass_request(payload: PublishTaskCreate) -> fanout_svc.PublishTaskRequest:
    targets = [
        fanout_svc.FanoutTarget(
            channel_id=t.channel_id,
            upload_path=t.upload_path,
            publish_kind=t.publish_kind,
            brand_profile_id=t.brand_profile_id,
            thumbnail_source=t.thumbnail_source,
            thumbnail_r2_key=t.thumbnail_r2_key,
            scheduled_at=t.scheduled_at,
            brand_profile_version=t.publish_version_inputs.brand_profile_version,
            seo_version=t.publish_version_inputs.seo_version,
            metadata_version=t.publish_version_inputs.metadata_version,
        )
        for t in payload.targets
    ]
    return fanout_svc.PublishTaskRequest(
        master_video_id=payload.master_video_id,
        targets=targets,
        priority=payload.priority,
    )


@router.post(
    "/publish-tasks",
    response_model=PublishTaskCreated,
    status_code=status.HTTP_201_CREATED,
)
def create_publish_task(
    payload: PublishTaskCreate,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
    # Wave 2 (API scale): plan-aware token bucket — runs BEFORE the
    # endpoint body so an over-quota tenant 429s without DB work.
    _rl=Depends(_rate_limited("create")),
):
    """Create a PublishTask + N UploadJobV2 rows; enqueue on the Scheduler.

    Feature-flag-gated by ``KAIZER_NEW_PUBLISH_PATH=1``. Disabled by
    default; returns 503 with ``code='new_publish_path_disabled'`` so
    clients can detect the dormant state without scraping ``message``.
    """
    _ensure_enabled()

    request_dc = _to_dataclass_request(payload)

    try:
        result = fanout_svc.create_publish_task(db, user, request_dc)
    except fanout_svc.DuplicatePublishVersionError as exc:
        # Dedupe-by-design (brief §2): same idempotency_key → return
        # the existing PublishTask's rollup as 200 OK.
        db.commit()  # commit any refund rows the fanout layer wrote
        existing = _build_existing_result(db, exc.existing_publish_task_id)
        return existing
    except fanout_svc.PlanTierMissingError as exc:
        db.rollback()
        log.error("publish_tasks: plan_tier missing for user_id=%d: %s", user.id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": exc.code, "message": str(exc)},
        )
    except fanout_svc.PlanTierViolationError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": exc.code, "message": str(exc), **exc.extra},
        )
    except fanout_svc.ChannelOwnershipError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": exc.code, "message": str(exc), **exc.extra},
        )
    except fanout_svc.MasterVideoNotReadyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": exc.code, "message": str(exc), **exc.extra},
        )
    except fanout_svc.ThumbnailSourceMismatchError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": exc.code, "message": str(exc), **exc.extra},
        )
    except fanout_svc.OAuthTokenMissingError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": exc.code, "message": str(exc), **exc.extra},
        )
    except credits_svc.InsufficientCreditsError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail={
                "code": "insufficient_credits",
                "message": str(exc),
                "balance": exc.balance,
                "needed": exc.needed,
            },
        )
    except fanout_svc.FanoutError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": exc.code, "message": str(exc), **exc.extra},
        )
    except Exception:
        db.rollback()
        log.exception("publish_tasks: unexpected error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "internal_error", "message": "see server log"},
        )

    db.commit()
    return PublishTaskCreated(
        publish_task_id=result.publish_task_id,
        upload_job_ids=result.upload_job_ids,
        predicted_credit_total=result.predicted_credit_total,
        predicted_quota_units_total=result.predicted_quota_units_total,
        quota_pre_flight_ok=result.quota_pre_flight_ok,
        deduped=False,
    )


# ─── GET /api/publish-tasks/:id ───────────────────────────────────────────


@router.get(
    "/publish-tasks/{publish_task_id}",
    response_model=PublishTaskOut,
)
def get_publish_task(
    publish_task_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Return one PublishTask + its child UploadJobV2 rows.

    No feature flag — read-only is safe to expose pre-cutover.
    """
    task = (
        db.query(models.PublishTask)
        .filter(models.PublishTask.id == publish_task_id)
        .first()
    )
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "publish_task_not_found",
                "message": f"publish_task_id={publish_task_id} not found",
            },
        )
    if int(task.user_id) != int(user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "publish_task_not_owned",
                "message": f"publish_task_id={publish_task_id} belongs to another user",
            },
        )

    jobs = (
        db.query(models.UploadJobV2)
        .filter(models.UploadJobV2.publish_task_id == publish_task_id)
        .order_by(models.UploadJobV2.id.asc())
        .all()
    )
    # One channels query for the whole task — names/handles for the
    # per-channel audit table.
    ch_ids = {int(j.channel_id) for j in jobs if j.channel_id is not None}
    channels = {}
    if ch_ids:
        channels = {
            int(c.id): c
            for c in db.query(models.Channel)
            .filter(models.Channel.id.in_(list(ch_ids)))
            .all()
        }
    out = _serialise_task(task, [])
    out.upload_jobs = [
        _serialise_job(j, channels.get(int(j.channel_id) if j.channel_id else 0))
        for j in jobs
    ]
    return out


# ─── GET /api/publish-tasks (list, paged) ─────────────────────────────────


@router.get(
    "/publish-tasks",
    response_model=PublishTaskList,
)
def list_publish_tasks(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status_filter: Optional[str] = Query(None, alias="status"),
):
    """Current user's PublishTasks. Newest first."""
    q = (
        db.query(models.PublishTask)
        .filter(models.PublishTask.user_id == user.id)
    )
    if status_filter:
        q = q.filter(models.PublishTask.status == status_filter)

    total = q.count()
    rows = (
        q.order_by(models.PublishTask.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    # ── Audit-view enrichment: ONE children query for the whole page ──
    # (≤100 tasks), then small lookups for clip titles/thumbs + channel
    # names. Keeps the list endpoint at 4 queries regardless of page size.
    task_ids = [int(r.id) for r in rows]
    by_task: dict[int, list] = {tid: [] for tid in task_ids}
    if task_ids:
        kids = (
            db.query(
                models.UploadJobV2.publish_task_id,
                models.UploadJobV2.status,
                models.UploadJobV2.channel_id,
                models.UploadJobV2.clip_id,
                models.UploadJobV2.publish_kind,
                models.UploadJobV2.attempts,
                models.UploadJobV2.next_attempt_at,
                models.UploadJobV2.upload_path,
            )
            .filter(models.UploadJobV2.publish_task_id.in_(task_ids))
            .order_by(models.UploadJobV2.id.asc())
            .all()
        )
        for k in kids:
            by_task.setdefault(int(k[0]), []).append(k)

    # Channel names (preview) + clip display info in two batched queries.
    all_ch_ids = {int(k[2]) for ks in by_task.values() for k in ks if k[2]}
    ch_names: dict[int, str] = {}
    if all_ch_ids:
        ch_names = {
            int(c.id): (c.name or f"Channel #{c.id}")
            for c in db.query(models.Channel.id, models.Channel.name)
            .filter(models.Channel.id.in_(list(all_ch_ids)))
            .all()
        }
    first_clip_by_task = {
        tid: next((int(k[3]) for k in ks if k[3]), None)
        for tid, ks in by_task.items()
    }
    clip_info = _clip_display(
        db, [c for c in first_clip_by_task.values() if c],
    )

    # Source job per task (master.source_upload_id): PublishTasks sharing one
    # source_upload_id are clips of the SAME render (full video + its shorts),
    # so the UI can group them under one render header.
    master_ids = {int(r.master_video_id) for r in rows if r.master_video_id is not None}
    job_by_master: dict[int, int] = {}
    if master_ids:
        for mid, src in (db.query(models.MasterVideo.id, models.MasterVideo.source_upload_id)
                           .filter(models.MasterVideo.id.in_(list(master_ids))).all()):
            if src is not None:
                job_by_master[int(mid)] = int(src)

    now = datetime.utcnow()
    items: List[PublishTaskListItem] = []
    for r in rows:
        ks = by_task.get(int(r.id), [])
        parked = sum(1 for k in ks if k[1] == "parked_quota")
        cancelled = sum(1 for k in ks if k[1] == "cancelled")
        retrying = sum(
            1 for k in ks
            if k[1] == "queued" and int(k[5] or 0) > 0
        )
        in_flight = sum(
            1 for k in ks
            if k[1] in ("queued", "claimed", "branding",
                        "ready_to_upload", "uploading")
        )
        names = [ch_names.get(int(k[2]), "") for k in ks if k[2]]
        names = [n for n in names if n]
        clip = clip_info.get(first_clip_by_task.get(int(r.id)) or -1, {})
        paths = sorted({str(k[7] or "").strip().lower() for k in ks if k[7]})
        delivery = "" if not paths else (paths[0] if len(paths) == 1 else "mixed")
        items.append(PublishTaskListItem(
            id=int(r.id),
            master_video_id=int(r.master_video_id),
            priority=str(r.priority),
            status=str(r.status),
            target_count=int(r.target_count or 0),
            completed_count=int(r.completed_count or 0),
            failed_count=int(r.failed_count or 0),
            created_at=r.created_at,
            video_title=clip.get("title"),
            thumb_url=clip.get("thumb_url"),
            publish_kind=(str(ks[0][4]) if ks else None),
            channels_preview=names[:3],
            parked_count=parked,
            retrying_count=retrying,
            in_flight_count=in_flight,
            cancelled_count=cancelled,
        ))
    return PublishTaskList(items=items, limit=limit, offset=offset, total=total)


# ─── Per-child actions (job-wise Publishes UI) ────────────────────────────
#
# Mirror the legacy /uploads/{id}/retry + DELETE semantics for the v2
# pipeline. Money invariants:
#   * failed / cancelled / parked children were REFUNDED at terminalisation
#     → a retry must RE-RESERVE credits (reuse the unpark pattern), else
#       the re-upload is free.
#   * cancel refunds idempotently (skip when a refund row already exists).
# Counter invariants:
#   * retry of failed/cancelled decrements the parent's failed_count and
#     reopens it to 'dispatched'; the worker's atomic finalize (or the
#     hourly counter reconciler) re-derives the terminal status later.

_RETRYABLE_STATUSES = ("failed", "cancelled", "parked_quota")
_CANCELLABLE_STATUSES = ("queued", "parked_quota")

_BUMP_FAILED_SQL = """
UPDATE publish_tasks
SET failed_count = failed_count + 1,
    status = CASE
        WHEN completed_count + (failed_count + 1) >= target_count THEN
            CASE WHEN (failed_count + 1) = 0 THEN 'completed'
                 WHEN completed_count = 0 THEN 'failed'
                 ELSE 'partial_failed' END
        ELSE status
    END,
    updated_at = CURRENT_TIMESTAMP
WHERE id = :tid
"""

_REOPEN_SQL = """
UPDATE publish_tasks
SET failed_count = CASE WHEN failed_count >= :dec THEN failed_count - :dec
                        ELSE 0 END,
    status = 'dispatched',
    updated_at = CURRENT_TIMESTAMP
WHERE id = :tid
"""


def _owned_task(db: Session, user: models.User, publish_task_id: int) -> models.PublishTask:
    task = (
        db.query(models.PublishTask)
        .filter(models.PublishTask.id == publish_task_id)
        .first()
    )
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "publish_task_not_found",
                    "message": f"publish_task_id={publish_task_id} not found"},
        )
    if int(task.user_id) != int(user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "publish_task_not_owned",
                    "message": "belongs to another user"},
        )
    return task


def _refund_exists(db: Session, user_id: int, upload_job_id: int) -> bool:
    return (
        db.query(models.CreditLedger)
        .filter(
            models.CreditLedger.user_id == user_id,
            models.CreditLedger.upload_job_id == upload_job_id,
            models.CreditLedger.reason == "refund",
        )
        .first()
        is not None
    )


def _requeue_children(
    db: Session,
    task: models.PublishTask,
    jobs: List[models.UploadJobV2],
) -> dict:
    """Shared retry core: flip children back to 'queued', reopen the
    parent's counters, re-reserve credits (insufficient balance fails
    the child again via fail_terminal — same as the unpark cron)."""
    ids = [int(j.id) for j in jobs]
    if not ids:
        return {"retried": 0, "job_ids": []}
    prev_failed_like = sum(1 for j in jobs if j.status in ("failed", "cancelled"))

    db.execute(
        _sql_text(
            "UPDATE upload_jobs_v2 "
            "SET status='queued', attempts=0, "
            "    next_attempt_at=CURRENT_TIMESTAMP, claimed_by=NULL, "
            "    lease_expires_at=NULL, last_error=NULL, finished_at=NULL, "
            "    updated_at=CURRENT_TIMESTAMP "
            "WHERE id IN :ids"
        ).bindparams(_sql_bindparam("ids", expanding=True)),
        {"ids": ids},
    )
    if prev_failed_like:
        db.execute(_sql_text(_REOPEN_SQL),
                   {"dec": prev_failed_like, "tid": int(task.id)})
    else:
        db.execute(_sql_text(
            "UPDATE publish_tasks SET status='dispatched', "
            "updated_at=CURRENT_TIMESTAMP WHERE id = :tid"
        ), {"tid": int(task.id)})
    db.commit()

    # Re-reserve credits — terminal/parked children were refunded.
    # Reuses the unpark cron's helper (handles InsufficientCredits by
    # terminal-failing the child WITHOUT a double refund).
    from services.cron_runner import _re_reserve_credits
    _re_reserve_credits([
        {
            "id": int(j.id),
            "user_id": int(task.user_id),
            "predicted_credit_cost": int(j.predicted_credit_cost or 0),
            "upload_path": str(j.upload_path or "direct"),
            "publish_kind": str(j.publish_kind or "video"),
            "predicted_quota_units": int(j.predicted_quota_units or 0),
        }
        for j in jobs
        if int(j.predicted_credit_cost or 0) > 0
        and _refund_exists(db, int(task.user_id), int(j.id))
    ])
    return {"retried": len(ids), "job_ids": ids}


@router.post("/publish-tasks/{publish_task_id}/jobs/{job_id}/retry")
def retry_publish_job(
    publish_task_id: int,
    job_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Re-queue one failed / cancelled / quota-parked child."""
    task = _owned_task(db, user, publish_task_id)
    job = (
        db.query(models.UploadJobV2)
        .filter(
            models.UploadJobV2.id == job_id,
            models.UploadJobV2.publish_task_id == publish_task_id,
        )
        .first()
    )
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "upload_job_not_found",
                    "message": f"job_id={job_id} not in this publish"},
        )
    if job.status not in _RETRYABLE_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "not_retryable",
                    "message": f"status={job.status!r} — only "
                               f"{'/'.join(_RETRYABLE_STATUSES)} can be retried"},
        )
    result = _requeue_children(db, task, [job])
    db.expire_all()
    fresh = db.query(models.UploadJobV2).filter(
        models.UploadJobV2.id == job_id).first()
    return {
        **result,
        "job_status": (fresh.status if fresh else "queued"),
        "last_error": ((fresh.last_error or "") if fresh else ""),
    }


@router.post("/publish-tasks/{publish_task_id}/jobs/{job_id}/cancel")
def cancel_publish_job(
    publish_task_id: int,
    job_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Cancel one queued / quota-parked child (idempotent refund).
    In-flight children (claimed/branding/uploading) return 409 — the
    upload is already burning quota; let it finish or fail."""
    task = _owned_task(db, user, publish_task_id)
    job = (
        db.query(models.UploadJobV2)
        .filter(
            models.UploadJobV2.id == job_id,
            models.UploadJobV2.publish_task_id == publish_task_id,
        )
        .first()
    )
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "upload_job_not_found",
                    "message": f"job_id={job_id} not in this publish"},
        )
    res = db.execute(
        _sql_text(
            "UPDATE upload_jobs_v2 "
            "SET status='cancelled', claimed_by=NULL, lease_expires_at=NULL, "
            "    finished_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP "
            "WHERE id = :jid AND status IN :ok"
        ).bindparams(_sql_bindparam("ok", expanding=True)),
        {"jid": int(job_id), "ok": list(_CANCELLABLE_STATUSES)},
    )
    if not res.rowcount:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "not_cancellable",
                    "message": f"status={job.status!r} — only "
                               f"{'/'.join(_CANCELLABLE_STATUSES)} can be "
                               f"cancelled (in-flight uploads must finish)"},
        )
    cost = int(job.predicted_credit_cost or 0)
    if cost > 0 and not _refund_exists(db, int(task.user_id), int(job_id)):
        credits_svc.refund(
            db, user_id=int(task.user_id), cost=cost, upload_job_id=int(job_id),
        )
    db.execute(_sql_text(_BUMP_FAILED_SQL), {"tid": int(task.id)})
    db.commit()
    return {"cancelled": True, "job_id": int(job_id)}


@router.post("/publish-tasks/{publish_task_id}/retry-failed")
def retry_all_failed(
    publish_task_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Bulk re-queue of every failed / cancelled child in one publish."""
    task = _owned_task(db, user, publish_task_id)
    jobs = (
        db.query(models.UploadJobV2)
        .filter(
            models.UploadJobV2.publish_task_id == publish_task_id,
            models.UploadJobV2.status.in_(["failed", "cancelled"]),
        )
        .all()
    )
    if not jobs:
        return {"retried": 0, "job_ids": []}
    return _requeue_children(db, task, jobs)


__all__ = ["router"]
