"""Phase 2.E — Upload dispatch service.

Single entry point for the Phase 2 upload worker. The Scheduler's
``_run_job()`` will (once the orchestrator wires it up — not in this
agent's scope) call ``upload_dispatch.process(upload_job_id)`` while
holding the network slot, and this module:

  1. Opens a DB session, loads the UploadJobV2 + linked rows.
  2. Defense-in-depth plan-tier check (Fanout already gates Free Direct;
     we re-check so a bypass at the publish API still trips here).
  3. Ensures the branded artifact exists by calling
     ``services.branding.process_upload_job(upload_job_id)``.
  4. Idempotency pre-check: if a prior attempt recorded a
     ``youtube_video_id``, confirm via ``videos.list`` (1 unit) and
     short-circuit. NEVER ``search.list`` (brief §9).
  5. Reserves credits + quota (via the F-agent stubs).
  6. Composes metadata via ``seo.composer.compose(...)``.
  7. Routes on ``upload_path``:
       - 'direct': ``youtube.uploader_v2.upload_video(...)``
       - 'rtmp':   ``youtube.rtmp_agent_v2.upload_via_rtmp(...)``
  8. On success: records ``youtube_video_id``, sets status='completed',
     calls F-agent's ``idempotency.record_success`` (when available).
  9. For Full Videos only: ``thumbnails.set`` (50u) — gated by a
     separate ``quota.reserve(COST_THUMBNAIL_SET)``. For Shorts: skip.
 10. On quotaExceeded: refunds credits, sets status='parked_quota',
     does NOT bump attempts.
 11. On transient: persists progress (bytes_uploaded, upload_uri),
     re-raises so the scheduler can re-enqueue with backoff.
 12. On permanent: refunds credits (so user isn't charged for our bug),
     status='failed', last_error=str(e)[:1000].

HARD CONSTRAINTS (brief §9 + task spec)
---------------------------------------
* No `search.list` calls. Idempotency confirmation = `videos.list` (1u).
* Free-tier Direct is rejected at the start of ``process()`` with
  ``PlanTierViolation`` (defense-in-depth).
* Credit / quota numbers come from a single constants table here
  (mirrored from Fanout's ``_CREDIT_COST`` / ``_QUOTA_UNITS`` per
  Decision 4). Hardcoded numbers live nowhere else in this file.
* Audio is `-c:a copy` everywhere (the branding stage already handled
  video; we don't re-encode).
* Every YouTube API call is wrapped in ``log_youtube_call`` by the
  underlying uploader_v2 / rtmp_provider modules — forensic log
  preserved (brief §2).
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Tuple

from sqlalchemy import update as _sa_update

from database import SessionLocal
import models

# Durable-queue contract (Wave 1). Only exercised when ``process()`` is
# called WITH a worker_id (i.e. from services.publish_worker under
# KAIZER_DURABLE_QUEUE=1). The legacy scheduler path (worker_id=None)
# keeps the exact pre-existing raise/persist behaviour.
from services import job_queue

# Branding is owned by the D-agent and is REQUIRED — its absence is a
# real error. Don't try-import it.
from services import branding

# F-agent surfaces — degrade gracefully if the full impl hasn't landed.
from services import credits

# F-agent's predicted-vs-actual ledger (Phase 2.F). MUST be called once
# per YouTube API call site (brief §2 + §7, CONTRACTS §4.5). We import
# at module-load time because every successful upload path writes ≥ 1
# row and we want an import-time failure to be loud rather than a
# silent ledger-skip at runtime.
from services import burn_log

try:  # Full F-agent impl may not exist at smoke-test time.
    from services import idempotency as _idem_module  # noqa: F401
    _IDEMPOTENCY_FULL = hasattr(_idem_module, "check_or_register")
except Exception:
    _idem_module = None
    _IDEMPOTENCY_FULL = False

try:  # F-agent's quota_v2 may not exist yet.
    from youtube import quota_v2 as quota_mod  # type: ignore
    _QUOTA_V2 = True
except Exception:
    from youtube import quota as quota_mod  # type: ignore
    _QUOTA_V2 = False

# Storage provider for R2 access.
from pipeline_core.storage import get_storage_provider

# OAuth is preserved per the brief — we call it directly.
from youtube import oauth as _yt_oauth

# Direct + RTMP path modules.
from youtube import uploader_v2
from youtube import rtmp_agent_v2

# SEO composer (existing logic must be reused).
try:
    from seo.composer import compose as _seo_compose
except Exception:
    _seo_compose = None  # type: ignore


log = logging.getLogger("kaizer.upload_dispatch")


# ─── Credit / quota constants — single source of truth ──────────────
# These MUST match ``services/fanout.py`` so the Fanout pre-flight
# reservation and the dispatch reservation see the same numbers.
# Decision 4 (DECISIONS.md):
#   Direct = 16 credits / ~1600 quota (full video adds 50u thumb = 1650)
#   RTMP   = 2  credits / ~150  quota (full video adds 50u thumb = 200)

CREDIT_COST: dict[tuple[str, str], int] = {
    ("direct", "video"): 16,
    ("direct", "short"): 16,
    ("rtmp",   "video"): 2,
    ("rtmp",   "short"): 2,
    # Postiz never calls the YouTube API → 0 credits / 0 quota.
    ("postiz", "video"): 0,
    ("postiz", "short"): 0,
}

QUOTA_UNITS: dict[tuple[str, str], int] = {
    ("direct", "video"): 1650,  # 1600 insert + 50 thumb
    ("direct", "short"): 1600,  # no thumb on Shorts
    ("rtmp",   "video"): 200,   # 3 × 50 + 50 thumb
    ("rtmp",   "short"): 150,   # 3 × 50, no thumb
    ("postiz", "video"): 0,
    ("postiz", "short"): 0,
}

# Cost of the upload-only call (separating thumb so the Full Video
# branch can reserve it independently per brief §2 / task spec step 9).
# NOTE: the "direct" entry is NO LONGER used for gating — videos.insert
# moved to its own count-based 100/day bucket (2026-06-01), so direct
# uploads gate via quota_v2.reserve_upload(), not this units map. Kept
# for the rtmp/postiz Queries-pool reservations only.
QUOTA_UPLOAD_ONLY: dict[str, int] = {
    "direct": 100,   # repriced 1,600 → 100; gated by count, see _quota_reserve_upload
    "rtmp":   150,
    # 0 → the postiz branch skips the YouTube quota reservation entirely.
    "postiz": 0,
}

COST_THUMBNAIL_SET = 50  # Mirrored from youtube/quota.py:82.
COST_VIDEOS_LIST = 1     # Idempotency confirmation probe.


# ─── Burn-log helper (Phase 2.F predicted-vs-actual ledger) ─────────
#
# This helper lives here (rather than as a public surface in
# ``services.burn_log``) because it encodes upload-dispatch policy:
# (a) how to classify an exception into a quota_burn_log.observed_outcome,
# (b) when to swallow vs propagate. The public burn_log API stays
# focused on the DB write — no classification logic there.
#
# Wired into every YouTube API call site invoked from upload_dispatch
# (videos.insert, thumbnails.set, the 3 RTMP triple-call rows, the
# videos.list idempotency probe) AND the quota-park branch. The
# orthogonal forensic log (``learning.youtube_quota_log``) is
# preserved per brief §0 — we do NOT double-write to youtube_api_calls.


def _burn_for_call(
    db,
    *,
    upload_job_id,
    operation: str,
    predicted_cost: int,
    exc: Optional[BaseException] = None,
    http_status: Optional[int] = None,
) -> None:
    """Write one ``quota_burn_log`` row classifying the outcome.

    ``upload_job_id`` here is the ``upload_jobs_v2.id`` — burn_log's
    FK points at the v2 table (models.py:1747), so we pass the real
    id (NOT the legacy-bridge value the forensic log uses).

    Never raises; burn_log.log_predicted_and_actual itself swallows
    DB errors so a ledger-write failure cannot cascade-kill an upload.
    """
    if db is None:
        return
    if exc is None:
        try:
            burn_log.log_predicted_and_actual(
                db,
                upload_job_id=upload_job_id,
                operation=operation,
                predicted_cost=int(predicted_cost),
                http_status=int(http_status if http_status is not None else 200),
                observed_outcome="success",
            )
        except Exception:
            pass
        return

    # Determine outcome from the exception class / status / reason.
    status = int(http_status) if http_status is not None else 0
    if status == 0:
        # Try to pull HTTP status from a googleapiclient HttpError.
        resp = getattr(exc, "resp", None)
        status = int(getattr(resp, "status", 0) or 0)

    outcome = "permanent_error"
    if isinstance(exc, uploader_v2.QuotaExceededError):
        outcome = "quota_exceeded"
    elif isinstance(exc, uploader_v2.TransientUploadError):
        outcome = "transient_error"
    elif isinstance(exc, rtmp_agent_v2.TransientRtmpUploadError):
        outcome = "transient_error"
    else:
        # Fall back to inspecting the raw HttpError shape.
        reason = ""
        try:
            import json as _json
            content = getattr(exc, "content", None)
            if isinstance(content, (bytes, bytearray)):
                data = _json.loads(content.decode("utf-8")) if content else {}
            elif isinstance(content, str):
                data = _json.loads(content) if content else {}
            else:
                data = {}
            errs = (data.get("error") or {}).get("errors") or []
            if errs:
                reason = str(errs[0].get("reason") or "")
        except Exception:
            pass
        msg = str(exc).lower()
        if reason in {"quotaExceeded", "dailyLimitExceeded"} or (
            status == 403 and "quota" in msg
        ):
            outcome = "quota_exceeded"
        elif status in {500, 502, 503, 504} or reason in {
            "rateLimitExceeded", "userRateLimitExceeded",
            "internalError", "backendError",
        }:
            outcome = "transient_error"

    try:
        burn_log.log_predicted_and_actual(
            db,
            upload_job_id=upload_job_id,
            operation=operation,
            predicted_cost=int(predicted_cost),
            http_status=status,
            observed_outcome=outcome,
        )
    except Exception:
        pass


# ─── Errors the dispatch surfaces to the scheduler ──────────────────


class PlanTierViolation(Exception):
    """Defense-in-depth: Free user requested Direct path; Fanout should
    have rejected this. Surfacing as a hard error means the scheduler
    marks the job failed (no retry — user must change the path)."""


class UploadJobNotFound(Exception):
    """The upload_job_id passed to ``process()`` does not exist."""


class OAuthTokenMissing(Exception):
    """The channel's OAuth token row is missing — user must re-link."""


class OwnershipViolation(Exception):
    """ISOLATION INVARIANT I3 (fail-closed): the destination channel does
    NOT belong to the user who owns this job. A correct system can never
    reach this — fanout only ever targets the owner's channels — but on a
    shared staged pipeline a mis-routed unit must be HARD-stopped before it
    can deliver one user's video to another user's channel. OAuth would
    also reject (the refresh token is the owner's), but we do not rely on
    that: we assert ownership at the upload boundary and refuse to publish.
    Terminal, no retry — a mismatch is a bug, not a transient fault."""


class StaleClaim(Exception):
    """Fence violation: this worker no longer owns the job (its lease
    expired and the reaper handed the row to someone else). The correct
    response is to abort silently — every further write would no-op
    against the fence anyway, and the new owner is already driving."""


class DeadlineExceeded(Exception):
    """Cooperative dispatch deadline hit. Classified as transient: the
    job requeues with backoff and resumable progress (upload_uri /
    bytes_uploaded) intact."""


# ─── Durable-queue outcome contract ──────────────────────────────────


class Outcome(str, Enum):
    """What ``process()`` reports back to the publish worker. The worker
    maps these to PublishTask counter bumps — it NEVER overwrites job
    status from a generic exception handler (the bug that used to turn
    transient 503s into permanent, credit-burning failures)."""

    COMPLETED = "completed"        # video is on YouTube
    PARKED_QUOTA = "parked_quota"  # quota exhausted; un-parker will retry
    RETRY = "retry"                # transient; requeued with backoff
    FAILED = "failed"              # permanent; credits refunded
    SKIPPED = "skipped"            # stale claim / nothing to do
    HANDOFF = "handoff"            # branding done; slot released, upload re-queued
                                   # as its own unit (KAIZER_SPLIT_STAGES conveyor)


# ─── Fenced writes + cooperative deadline (durable mode) ─────────────


def _guarded_job_update(
    db,
    job_id: int,
    worker_id: Optional[str],
    values: dict,
    *,
    require_status_in: Optional[tuple[str, ...]] = None,
) -> Optional[bool]:
    """UPDATE upload_jobs_v2 with the worker fence.

    When ``worker_id`` is set, the WHERE clause requires
    ``claimed_by = worker_id`` — a zombie thread (abandoned by the
    watchdog, lease reaped, job reclaimed) can never clobber the new
    owner's state. ``require_status_in`` additionally guards stage
    progression (e.g. you can only flip to 'uploading' from an active
    pre-upload status — never from 'completed').

    Tri-state return so callers can distinguish a true fence miss from
    an infrastructure blip:
      True  → row updated (we still own it)
      False → fence miss (rowcount 0 — another actor owns the row)
      None  → DB error (unknown; caller decides whether to fail open)
    """
    stmt = _sa_update(models.UploadJobV2).where(
        models.UploadJobV2.id == int(job_id)
    )
    if worker_id is not None:
        stmt = stmt.where(models.UploadJobV2.claimed_by == worker_id[:64])
    if require_status_in is not None:
        stmt = stmt.where(models.UploadJobV2.status.in_(list(require_status_in)))
    try:
        res = db.execute(stmt.values(**values))
        db.commit()
        return bool(res.rowcount and res.rowcount > 0)
    except Exception:
        db.rollback()
        log.exception(
            "upload_dispatch: guarded update failed for job=%d", job_id
        )
        return None


def _check_deadline(deadline: Optional[float], job_id: int) -> None:
    """Cooperative timeout — called between stages and from the upload
    progress callback. Raises DeadlineExceeded (→ transient requeue,
    progress preserved) once the monotonic deadline passes. This is the
    layer that actually works for sync code running in a thread; the
    worker's asyncio watchdog is only the backstop."""
    if deadline is not None and time.monotonic() > deadline:
        raise DeadlineExceeded(
            f"dispatch deadline exceeded for job={job_id}"
        )


# ─── Worker identity (idempotency attribution) ──────────────────────


def _worker_id() -> str:
    """A stable-per-process identifier for the worker, used in the
    publish_attempts.worker_id column when F-agent's idempotency layer
    is enabled."""
    pid = os.getpid()
    host = os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME") or "host"
    return f"{host}:{pid}"[:64]


def _log_attr_upload_job_id(v2_id: Optional[int]) -> Optional[int]:
    """Return a value safe to pass as ``log_youtube_call(upload_job_id=…)``.

    ``youtube_api_calls.upload_job_id`` is FK'd to the LEGACY
    ``upload_jobs.id`` (models.py:1115) — not ``upload_jobs_v2.id``. A
    v2 row id will FK-violate on the INSERT and the forensic log row
    gets dropped (silently logged as a WARNING in the log wrapper).

    Until the A-agent migrates the FK to ``upload_jobs_v2.id``, we pass
    ``None`` here so the forensic log row IS written (attribution
    falls back to user_id / channel_id / video_id, which is enough
    for the admin dashboards to group). Setting
    ``KAIZER_UPLOAD_V2_LOG_PASS_ID=1`` opts back in (use only after
    the FK migration ships).
    """
    if v2_id is None:
        return None
    if os.environ.get("KAIZER_UPLOAD_V2_LOG_PASS_ID", "0").strip() == "1":
        return int(v2_id)
    return None


# ─── Helpers ────────────────────────────────────────────────────────


def _gcid_for_channel(db, channel_id: int) -> str:
    """Resolve google_channel_id for log attribution. Returns '' on miss."""
    tok = (
        db.query(models.OAuthToken)
        .filter(models.OAuthToken.channel_id == channel_id)
        .first()
    )
    if tok is None:
        return ""
    return (getattr(tok, "google_channel_id", "") or "")[:50]


def _clip_for_job(db, job: models.UploadJobV2) -> Optional[models.Clip]:
    """Return the Clip row whose SEO drives this upload's metadata.

    The V2 fan-out path does NOT stamp ``clip_id`` onto the UploadJobV2,
    so the direct ``job.clip_id`` lookup is almost always None — which
    made _compose_metadata fall back to a bare "<channel> #<id>" title
    with an EMPTY description (no SEO, no socials). Recover the clip from
    the job's MasterVideo instead: there is one master per clip, and
    ``master_videos.clip_id`` is populated at materialisation, so it's the
    reliable link. We keep the direct ``job.clip_id`` path first for any
    legacy rows that do carry it.
    """
    if job.clip_id:
        c = db.query(models.Clip).filter(models.Clip.id == job.clip_id).first()
        if c is not None:
            return c
    pt = (
        db.query(models.PublishTask)
        .filter(models.PublishTask.id == job.publish_task_id)
        .first()
    )
    if pt is not None and pt.master_video_id:
        mv = (
            db.query(models.MasterVideo)
            .filter(models.MasterVideo.id == pt.master_video_id)
            .first()
        )
        if mv is not None and getattr(mv, "clip_id", None):
            return (
                db.query(models.Clip)
                .filter(models.Clip.id == mv.clip_id)
                .first()
            )
    return None


def _compose_metadata(
    db,
    job: models.UploadJobV2,
) -> Tuple[str, str, list[str], str, bool]:
    """Compose (title, description, tags, category_id, made_for_kids).

    Reuses ``seo.composer.compose(generic, channel, publish_kind=...)``
    for the brand overlay step. The UploadJobV2 schema (CONTRACTS §3.3
    as shipped by the A-agent) does NOT carry title/description/tags as
    persisted columns — composer-resolved values are recomputed each
    dispatch from the linked Clip's SEO. We still respect transient
    attributes via ``getattr(...)`` so test rigs that prepopulate the
    row's title/description (e.g. monkey-patched attrs for smoke tests)
    can still pin the metadata.

    Returns a tuple matching what uploader_v2.upload_video expects.
    """
    channel = db.query(models.Channel).filter(
        models.Channel.id == job.channel_id
    ).first()

    # Transient overrides (test rigs / future admin-override columns).
    pre_title = (getattr(job, "title", None) or "")
    pre_desc = (getattr(job, "description", None) or "")
    pre_tags = list(getattr(job, "tags", None) or [])
    pre_cat = (getattr(job, "category_id", None) or "25")
    pre_mfk = bool(getattr(job, "made_for_kids", False) or False)

    if channel is None:
        # Channel deleted between Fanout and dispatch — fail soft with
        # whatever transient overrides we have, but never block.
        return (
            (pre_title or "Untitled")[:100],
            pre_desc,
            pre_tags,
            pre_cat,
            pre_mfk,
        )

    # If a title + description are already pinned on the row instance
    # (e.g. retry path or smoke-test override), use them as-is.
    if pre_title and pre_desc:
        return (pre_title[:100], pre_desc, pre_tags, pre_cat, pre_mfk)

    # Compose via the existing seo.composer (brief §2 + CONTRACTS §4):
    # parse the linked Clip's SEO JSON, then overlay the channel brand.
    clip = _clip_for_job(db, job)
    generic: dict = {}
    if clip is not None:
        import json as _json
        # PER-CHANNEL SEO (paid mode): if the operator applied a tailored SEO
        # for THIS channel it carries a "_per_channel" marker — use it as the
        # base so this channel publishes with its own title/tags (the composer
        # then overlays this channel's socials as usual). Variants WITHOUT the
        # marker (legacy/stale) are ignored here, so shared-mode clips — the
        # free default — are byte-for-byte unchanged.
        try:
            _variants = _json.loads(clip.seo_variants or "{}") if clip.seo_variants else {}
            if isinstance(_variants, dict):
                _v = _variants.get(str(job.channel_id)) or _variants.get(job.channel_id)
                if isinstance(_v, dict) and _v.get("_per_channel") and _v.get("title"):
                    generic = {k: val for k, val in _v.items() if k != "_per_channel"}
                    log.info("upload_dispatch: job=%d using per-channel SEO for channel=%s",
                             job.id, job.channel_id)
        except Exception:
            pass
        if not generic and clip.seo:
            try:
                parsed = _json.loads(clip.seo)
                if isinstance(parsed, dict) and parsed.get("title"):
                    generic = parsed
            except Exception:
                pass

    # PER-PLATFORM SEO: resolve which platform this job targets (YouTube vs
    # Instagram/Facebook via Postiz). For the social platforms, lazily
    # generate + cache a native caption variant (one Gemini call per clip+
    # platform) and attach it to `generic` so the composer's platform branch
    # shapes the post text. YouTube is unaffected (publish_platform="youtube").
    publish_platform = "youtube"
    try:
        from services.platform_variants import (
            resolve_publish_platform, ensure_platform_variant,
        )
        publish_platform = resolve_publish_platform(db, job)
        if publish_platform in ("instagram", "facebook") and clip is not None:
            _variant = ensure_platform_variant(db, clip, publish_platform)
            if _variant and _variant.get("caption"):
                if not isinstance(generic, dict):
                    generic = {}
                generic.setdefault("platform_variants", {})[publish_platform] = _variant
    except Exception as exc:
        log.warning(
            "upload_dispatch._compose_metadata: platform resolve/gen failed (%s)",
            exc,
        )

    title = pre_title
    description = pre_desc
    tags = pre_tags
    if generic and _seo_compose is not None:
        try:
            composed = _seo_compose(
                generic, channel, publish_kind=(job.publish_kind or "video"),
                platform=publish_platform,
            )
            title = title or composed.get("title", "")
            description = description or composed.get("description", "")
            if not tags:
                tags = list(composed.get("keywords") or [])
        except Exception as exc:
            log.warning(
                "upload_dispatch._compose_metadata: composer failed (%s); using fallback",
                exc,
            )

    if not title:
        title = (
            (clip.text if clip and clip.text else "")
            or f"{channel.name or 'Kaizer'} #{job.id}"
        )[:100]
    if not description:
        # Even if compose() returned nothing usable, the channel's
        # footer is the minimum-viable description.
        description = (channel.footer or "").strip()
    return (title[:100], description, tags, pre_cat, pre_mfk)


def _download_branded_artifact(cache_key: str, dest_dir: str) -> str:
    """Materialize the branded artifact from R2 to a local path so
    ffmpeg / the YouTube API can read it. Returns the local path."""
    provider = get_storage_provider()
    local_path = os.path.join(dest_dir, "_branded.mp4")
    provider.download(cache_key, local_path)
    if not os.path.isfile(local_path) or os.path.getsize(local_path) == 0:
        raise RuntimeError(
            f"upload_dispatch: branded artifact missing/empty after R2 download "
            f"(key={cache_key!r}, local={local_path!r})"
        )
    return local_path


def _resolve_user_and_plan(db, job: models.UploadJobV2) -> Tuple[int, str]:
    """Resolve (user_id, plan_tier_name) via PublishTask → User → PlanTier.

    Returns ('pro') as a defensive fallback when the chain has missing
    rows so the dispatch can still log a clear failure rather than
    crashing with a NoneType.
    """
    pt = db.query(models.PublishTask).filter(
        models.PublishTask.id == job.publish_task_id
    ).first()
    if pt is None:
        raise UploadJobNotFound(
            f"PublishTask id={job.publish_task_id} for job={job.id} missing"
        )
    user = db.query(models.User).filter(models.User.id == pt.user_id).first()
    if user is None:
        raise UploadJobNotFound(f"User id={pt.user_id} for job={job.id} missing")
    plan_tier = None
    if getattr(user, "plan_tier_id", None) is not None:
        plan_tier = db.query(models.PlanTier).filter(
            models.PlanTier.id == user.plan_tier_id
        ).first()
    plan_tier_name = (
        str(plan_tier.name) if plan_tier and plan_tier.name else "pro"
    )
    return int(user.id), plan_tier_name


def _plan_tier_row(db, user_id: int) -> Optional[models.PlanTier]:
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user is None or getattr(user, "plan_tier_id", None) is None:
        return None
    return db.query(models.PlanTier).filter(
        models.PlanTier.id == user.plan_tier_id
    ).first()


def _quota_reserve(db, cost: int) -> bool:
    """Dispatch between ``youtube.quota_v2.reserve_v2`` (F-agent's
    refactor) and the legacy ``youtube.quota.reserve`` so this module
    works at every cutover stage (CONTRACTS §4.5)."""
    if _QUOTA_V2 and hasattr(quota_mod, "reserve_v2"):
        try:
            # F-agent's v2 signature takes (db, cost).
            return bool(quota_mod.reserve_v2(db, cost))
        except TypeError:
            # Different signature variant — fall through.
            pass
    return bool(quota_mod.reserve(db, cost))


def _quota_reserve_upload(db) -> bool:
    """Reserve ONE videos.insert against the granular uploads bucket
    (100/day since 2026-06-01) when the v2 gate is live. videos.insert
    no longer draws from the 10,000 'Queries' pool, so we gate on a
    COUNT (≤100/day) instead of the dead 1,600-unit reservation. Falls
    back to charging the real 100-unit cost on the legacy gate."""
    if _QUOTA_V2 and hasattr(quota_mod, "reserve_upload"):
        try:
            return bool(quota_mod.reserve_upload(db))
        except TypeError:
            pass
    return _quota_reserve(db, 100)


def _quota_refund_upload(db) -> None:
    """Give back the ONE videos.insert reservation taken at the gate when
    a direct upload attempt did not complete (failed / parked / abandoned)
    — so failed retries don't permanently burn the 100/day bucket. Only
    the count-based v2 bucket is refundable; the legacy units path has no
    separate upload bucket, so it's a no-op there. Best-effort."""
    try:
        if _QUOTA_V2 and hasattr(quota_mod, "refund_upload"):
            quota_mod.refund_upload(db)
    except Exception:
        pass


def _refund_already(db, user_id: int, upload_job_id: int) -> bool:
    """Return True if a refund row for this upload_job_id already
    exists. Phase-1 credits stub puts the dedupe burden on the caller —
    we check before issuing another refund so a retry path doesn't
    double-refund (which would leak credits to the user)."""
    row = (
        db.query(models.CreditLedger)
        .filter(
            models.CreditLedger.user_id == user_id,
            models.CreditLedger.upload_job_id == upload_job_id,
            models.CreditLedger.reason == "refund",
        )
        .first()
    )
    return row is not None


def _idempotency_short_circuit(
    db,
    job: models.UploadJobV2,
    creds,
    google_channel_id: str,
) -> bool:
    """If a prior attempt recorded a ``youtube_video_id``, confirm via
    ``videos.list`` (1 unit) and short-circuit. Brief §2 + §9 forbid
    ``search.list``.

    Returns True when the dispatch should short-circuit (job is now
    'completed' in the DB). Returns False otherwise.
    """
    prior_vid = (job.youtube_video_id or "").strip()
    if not prior_vid:
        return False
    try:
        confirmed = uploader_v2.confirm_video_exists(
            creds,
            prior_vid,
            db=db,
            upload_job_id=_log_attr_upload_job_id(job.id),
            upload_job_id_v2=int(job.id),
            user_id=_user_id_for_job(db, job),
            channel_id=job.channel_id,
            google_channel_id=google_channel_id,
        )
    except Exception as exc:
        log.warning(
            "upload_dispatch: idempotency probe failed for job=%d "
            "video_id=%s — proceeding to re-upload: %s",
            job.id, prior_vid, exc,
        )
        return False
    if not confirmed:
        log.info(
            "upload_dispatch: prior video_id=%s for job=%d NOT confirmed via "
            "videos.list — proceeding to re-upload",
            prior_vid, job.id,
        )
        return False

    job.status = "completed"
    if job.finished_at is None:
        job.finished_at = datetime.utcnow()
    # Release the durable-queue claim — a completed row must not sit on
    # a lease (the reaper would otherwise requeue it at expiry... it
    # wouldn't, status filters protect us, but clean is clean).
    job.claimed_by = None
    job.lease_expires_at = None
    db.add(job)
    db.commit()
    log.info(
        "upload_dispatch: job=%d short-circuited via videos.list confirmation "
        "(youtube_video_id=%s)", job.id, prior_vid,
    )
    return True


def _user_id_for_job(db, job: models.UploadJobV2) -> Optional[int]:
    pt = db.query(models.PublishTask).filter(
        models.PublishTask.id == job.publish_task_id
    ).first()
    return int(pt.user_id) if pt else None


def _build_stage_env(job, user_id, channel):
    """Build the immutable factory telemetry envelope for this upload unit.
    Returns None on any failure — telemetry is strictly best-effort."""
    try:
        from services import stage_events as _se
        label = ""
        if channel is not None:
            label = (getattr(channel, "name", "") or "")[:40]
        return _se.Envelope(
            tenant_id=user_id,
            user_id=user_id,
            job_id=getattr(job, "clip_id", None),
            clip_id=getattr(job, "clip_id", None),
            channel_id=getattr(job, "channel_id", None),
            upload_job_id=getattr(job, "id", None),
            label=label,
        )
    except Exception:
        return None


def _emit_stage(env, stage: str, status: str, msg: str = "") -> None:
    """Emit one factory stage event. No-op if telemetry is unavailable."""
    if env is None:
        return
    try:
        from services import stage_events as _se
        _se.emit(env, stage, status, msg)
    except Exception:
        pass


# ─── Main entry point ──────────────────────────────────────────────


def process(
    upload_job_id: int,
    worker_id: Optional[str] = None,
    deadline: Optional[float] = None,
) -> Outcome:
    """Dispatch an UploadJobV2 to the correct upload path.

    Two calling modes:

    * **Legacy** (``worker_id=None``) — the in-memory scheduler's
      ``_run_job()`` holds the network slot and calls this; exceptions
      propagate exactly as before (KAIZER_DURABLE_QUEUE=0 path).
    * **Durable** (``worker_id`` set, from ``services.publish_worker``)
      — this function owns ALL terminal/retry persistence and returns
      an :class:`Outcome`; it never lets a transient error escape as a
      job-killing exception. Every status write is fenced with
      ``claimed_by = worker_id`` and ``deadline`` (``time.monotonic()``
      basis) is enforced cooperatively between stages and per upload
      chunk.
    """
    durable = worker_id is not None
    # Stage-split conveyor (flag-gated, durable mode only). When ON, a worker
    # releases its slot the moment branding finishes (row at 'ready_to_upload')
    # by re-queueing the job; the upload phase is then claimed as an independent
    # unit, so branding of the NEXT job overlaps the upload of this one instead
    # of the brand stage idling. OFF (default) = current serial behavior,
    # byte-identical. Inert whenever KAIZER_DURABLE_QUEUE=0 (durable is False).
    split_stages = durable and os.environ.get("KAIZER_SPLIT_STAGES", "0").strip() == "1"
    db = SessionLocal()
    work_dir: Optional[str] = None
    job: Optional[models.UploadJobV2] = None
    user_id: Optional[int] = None
    publish_kind = "video"
    upload_path = "direct"
    credit_cost = 0
    # Track the videos.insert (direct) 100/day-bucket reservation so we can
    # REFUND it if this attempt doesn't complete — only successful uploads
    # should count against the bucket. Cleared right before the COMPLETED
    # return; refunded in the finally when reserved-but-not-succeeded.
    upload_reserved = False
    upload_succeeded = False
    # Factory live-view telemetry envelope (ISOLATION INVARIANT I2). Built
    # once identity is known; guarded so a telemetry gap never breaks work.
    _stage_env = None
    _cur_stage = None
    try:
        # ── 1) Load row + linked context ───────────────────────────
        job = db.query(models.UploadJobV2).filter(
            models.UploadJobV2.id == int(upload_job_id)
        ).first()
        if job is None:
            raise UploadJobNotFound(f"UploadJobV2 id={upload_job_id} not found")

        publish_kind = (job.publish_kind or "video")
        upload_path = (job.upload_path or "direct")
        _check_deadline(deadline, int(upload_job_id))
        try:
            credit_cost = int(
                job.predicted_credit_cost
                or CREDIT_COST.get((upload_path, publish_kind), 0)
            )
        except Exception:
            credit_cost = CREDIT_COST.get((upload_path, publish_kind), 0)

        user_id, plan_tier_name = _resolve_user_and_plan(db, job)

        # ── 2) Plan-tier defense-in-depth (Decision 6) ─────────────
        plan_tier = _plan_tier_row(db, user_id)
        if plan_tier is not None:
            if upload_path == "direct" and not bool(plan_tier.direct_path_allowed):
                raise PlanTierViolation(
                    f"plan_tier={plan_tier.name!r} disallows upload_path='direct' "
                    f"(job_id={job.id}, user_id={user_id})"
                )

        gcid = _gcid_for_channel(db, job.channel_id)

        # ── 2b) Delivery ownership assertion (ISOLATION INVARIANT I3) ─
        # Fail-closed: the destination channel MUST belong to the user who
        # owns this job. Fanout only ever targets the owner's channels, so
        # a correct system never trips this — but on a shared staged
        # pipeline a mis-routed unit must be HARD-stopped here, before it
        # can brand, mint OAuth, or deliver one user's video to another
        # user's channel. user_id was resolved above via PublishTask.
        _dest_channel = (
            db.query(models.Channel)
            .filter(models.Channel.id == job.channel_id)
            .first()
        )
        if _dest_channel is None:
            raise OwnershipViolation(
                f"destination channel_id={job.channel_id} not found "
                f"(job_id={job.id}, user_id={user_id})"
            )
        if int(_dest_channel.user_id or 0) != int(user_id):
            raise OwnershipViolation(
                f"channel_id={job.channel_id} is owned by "
                f"user_id={_dest_channel.user_id!r}, but job_id={job.id} "
                f"belongs to user_id={user_id} — refusing to deliver"
            )

        # Build the factory telemetry envelope now that identity is verified.
        _stage_env = _build_stage_env(job, user_id, _dest_channel)

        # ── 3) Ensure branded artifact exists ──────────────────────
        _cur_stage = "brand"
        _emit_stage(_stage_env, "brand", "entered")
        cache_key = (job.branded_artifact_r2_key or "").strip()
        # Did this row ALREADY have a branded artifact when we entered? If so,
        # this is the UPLOAD-phase re-entry of a stage-split handoff (or a crash
        # recovery) — branding is skipped below and we must NOT hand off again.
        _entry_had_brand = bool(cache_key)
        if not cache_key:
            if (getattr(job, "brand_mode", "per_channel") or "per_channel") == "as_is":
                # Source is ALREADY branded (user's video has its own logo +
                # watermark). Upload the master VERBATIM to every channel — no
                # brand resolution, no overlay, no ffmpeg. We point straight at
                # the master's r2_key, so 'as_is' never mints a branded/ key
                # and can never collide with per-channel branded artifacts.
                _mid = branding._resolve_master_video_id(db, job)
                _master = db.query(models.MasterVideo).filter(
                    models.MasterVideo.id == int(_mid)
                ).first()
                cache_key = (getattr(_master, "r2_key", "") or "").strip()
                if not cache_key:
                    raise RuntimeError(
                        f"as_is publish: master_video {_mid} has no r2_key "
                        f"(job={job.id})"
                    )
                log.info("upload_dispatch: job=%d brand_mode=as_is — uploading "
                         "master verbatim key=%r (no overlay)", job.id, cache_key)
                if durable:
                    _guarded_job_update(
                        db, job.id, worker_id,
                        {"branded_artifact_r2_key": cache_key,
                         "status": "ready_to_upload"},
                    )
                else:
                    job.branded_artifact_r2_key = cache_key
                    job.status = "ready_to_upload"
                    db.add(job); db.commit()
                db.expire(job)
                job = db.query(models.UploadJobV2).filter(
                    models.UploadJobV2.id == int(upload_job_id)
                ).first()
                assert job is not None
            else:
                # Branding agent will write branded_artifact_r2_key + flip
                # status to 'ready_to_upload'. Re-load after.
                _check_deadline(deadline, job.id)
                cache_key = branding.process_upload_job(job.id)
                db.expire(job)
                job = db.query(models.UploadJobV2).filter(
                    models.UploadJobV2.id == int(upload_job_id)
                ).first()
                assert job is not None  # belt-and-braces — row obviously exists
                cache_key = (job.branded_artifact_r2_key or "").strip() or cache_key
        _emit_stage(_stage_env, "brand", "exited")

        # ── 3b) Stage-split conveyor handoff (flag-gated, durable only) ──
        # Branding just finished and the row is 'ready_to_upload'. Release THIS
        # worker slot now so it can brand the NEXT queued job, while the upload
        # phase runs as a separately-claimed unit. The re-claim re-enters
        # process(), finds branded_artifact_r2_key set (_entry_had_brand=True),
        # SKIPS branding, and runs only the upload — the exact path crash
        # recovery already exercises. No quota is reserved yet (step 7 is below),
        # so the handoff burns nothing. Guard on `not _entry_had_brand` makes the
        # upload-phase re-entry never hand off again (no infinite loop).
        if split_stages and not _entry_had_brand:
            # Brand stage already emitted 'exited' above; clear _cur_stage so the
            # finally does NOT emit a spurious 'brand failed' for this clean exit.
            _cur_stage = None
            state = job_queue.requeue_for_stage_handoff(
                job.id, worker_id, "stage-split: brand done, hand off to upload",
            )
            log.info(
                "upload_dispatch: job=%d brand→upload handoff (%s) — slot released",
                job.id, state,
            )
            return Outcome.HANDOFF

        # ── 4) Mint OAuth credentials ──────────────────────────────
        # Postiz delivery never calls the YouTube Data API, so it needs no
        # YouTube credentials (and must not fail when they are absent or
        # expired — Postiz owns the platform OAuth on its side).
        creds = None
        if upload_path != "postiz":
            try:
                creds = _yt_oauth.get_credentials(db, job.channel_id)
            except Exception as exc:
                raise OAuthTokenMissing(
                    f"OAuth credentials for channel_id={job.channel_id} failed: {exc}"
                ) from exc

        # ── 5) Idempotency pre-check (videos.list, 1u) ─────────────
        # Skipped for Postiz: the confirmation probe is a YouTube
        # videos.list call, and a Postiz post id is not a YouTube video
        # id. Postiz relies on the idempotency_key UNIQUE constraint only.
        if upload_path != "postiz":
            # Prefer F-agent's full impl when available.
            if _IDEMPOTENCY_FULL and _idem_module is not None:
                try:
                    result = _idem_module.check_or_register(  # type: ignore[attr-defined]
                        db,
                        key=job.idempotency_key,
                        worker_id=(worker_id or _worker_id()),
                        upload_job_id=job.id,
                    )
                    # We treat both AlreadyCompleted and our own
                    # videos.list confirmation as the short-circuit path
                    # below — keep the contract simple here in Phase 2.
                    if getattr(result, "youtube_video_id", None):
                        # Recheck via videos.list (1u) — brief §2 says we
                        # confirm even when the row says "completed".
                        if _idempotency_short_circuit(db, job, creds, gcid):
                            return Outcome.COMPLETED
                except Exception as exc:
                    log.warning(
                        "upload_dispatch: F-agent idempotency stub call raised "
                        "(%s); falling back to row-level probe", exc,
                    )

            # Row-level probe — short-circuit without calling F-agent at all.
            if _idempotency_short_circuit(db, job, creds, gcid):
                return Outcome.COMPLETED

        # ── 6) Compose metadata ────────────────────────────────────
        title, description, tags, category_id, made_for_kids = _compose_metadata(
            db, job,
        )

        # ── 7) Reserve quota for the upload-only operation ─────────
        # The thumbnail step is reserved separately (step 10).
        #
        # direct = videos.insert: own granular bucket (100/day, count-
        #   based) since 2026-06-01 — NOT the 10,000 "Queries" pool.
        # rtmp   = liveBroadcasts.*: still the Queries pool (~150u).
        # postiz = 0: never calls the YouTube API.
        if upload_path == "direct":
            quota_ok = _quota_reserve_upload(db)
            upload_reserved = bool(quota_ok)  # refund this on non-success
        else:
            upload_quota = QUOTA_UPLOAD_ONLY.get(upload_path, 0)
            quota_ok = (upload_quota <= 0) or _quota_reserve(db, upload_quota)
        if not quota_ok:
            # Park-not-fail (brief §9) — refund any credits the Fanout
            # service deducted at task creation time.
            log.info(
                "upload_dispatch: job=%d parked (quota exhausted for upload step)",
                job.id,
            )
            # Burn-log the parked operation BEFORE refunding so the
            # predicted-vs-actual ledger gets a was_quota_exceeded=True
            # row attributed to the upload that would-have-burned the
            # quota (brief §2 + §7; CONTRACTS §3.6).
            _burn_park_quota(db, job, upload_path, publish_kind)
            _park_quota(db, job, user_id, credit_cost, worker_id=worker_id)
            return Outcome.PARKED_QUOTA

        # ── 8) Download branded artifact to local disk ─────────────
        work_dir = tempfile.mkdtemp(prefix="kaizer_upload_dispatch_")
        try:
            branded_local = _download_branded_artifact(cache_key, work_dir)
        except Exception as exc:
            raise RuntimeError(
                f"upload_dispatch: branded artifact download failed "
                f"(job={job.id}, key={cache_key!r}): {exc}"
            ) from exc

        # ── 9) Flip status to 'uploading' before the network call ──
        # Durable mode: fenced + progression-guarded. If the lease was
        # reaped (claimed_by changed) or a parallel actor already
        # completed the row, this flip fails and we abort BEFORE
        # burning a 1600-unit videos.insert on a job we don't own.
        _check_deadline(deadline, job.id)
        if durable:
            flipped = _guarded_job_update(
                db, job.id, worker_id,
                {
                    "status": "uploading",
                    "started_at": (job.started_at or datetime.utcnow()),
                },
                require_status_in=("claimed", "branding", "ready_to_upload"),
            )
            if not flipped:
                raise StaleClaim(
                    f"job={job.id} not owned by worker={worker_id} at upload flip"
                )
            db.expire(job)
        else:
            job.status = "uploading"
            if job.started_at is None:
                job.started_at = datetime.utcnow()
            db.add(job)
            db.commit()

        # ── 10) Branch on upload_path ──────────────────────────────
        _cur_stage = "upload"
        _emit_stage(_stage_env, "upload", "entered", upload_path)
        if upload_path == "direct":
            result = _do_direct_upload(
                db, job, creds, branded_local,
                title=title,
                description=description,
                tags=tags,
                category_id=category_id,
                made_for_kids=made_for_kids,
                publish_kind=publish_kind,
                google_channel_id=gcid,
                user_id=user_id,
                worker_id=worker_id,
                deadline=deadline,
            )
        elif upload_path == "rtmp":
            result = _do_rtmp_upload(
                db, job, creds, branded_local,
                title=title,
                description=description,
                publish_kind=publish_kind,
                google_channel_id=gcid,
                user_id=user_id,
                worker_id=worker_id,
                deadline=deadline,
            )
        elif upload_path == "postiz":
            result = _do_postiz_upload(
                db, job, branded_local,
                title=title,
                description=description,
                tags=tags,
                publish_kind=publish_kind,
                user_id=user_id,
                worker_id=worker_id,
                deadline=deadline,
            )
        else:
            raise RuntimeError(
                f"upload_dispatch: unknown upload_path={upload_path!r} for job={job.id}"
            )

        video_id = str(result.get("video_id") or "")
        if not video_id:
            raise RuntimeError(
                f"upload_dispatch: upload returned no video_id (job={job.id}, result={result!r})"
            )

        # ── 11) Persist success ────────────────────────────────────
        bytes_done = int(
            result.get("bytes_uploaded")
            or result.get("bytes_pushed")
            or 0
        )
        if durable:
            # Deliberately NOT worker-fenced: recording a real YouTube
            # video id must never be blocked (blocking it is what
            # causes duplicate uploads). Guarded only against
            # overwriting a DIFFERENT already-recorded id — under the
            # durable queue's claim fencing that means a genuine
            # double-upload worth a loud log.
            _persist_success(db, job, video_id, bytes_done)
            db.expire(job)
        else:
            # Legacy scheduler path: byte-identical to the pre-Wave-1
            # behaviour (plain last-writer-wins ORM write). The legacy
            # heap can double-dispatch under test/race conditions and
            # its semantics expect the second writer to win silently.
            job.youtube_video_id = video_id[:32]
            job.bytes_uploaded = int(bytes_done or job.bytes_uploaded or 0)
            job.status = "completed"
            job.finished_at = datetime.utcnow()
            db.add(job)
            db.commit()

        # Notify the F-agent's idempotency layer (if available) so
        # publish_attempts gets the success row.
        if _IDEMPOTENCY_FULL and _idem_module is not None:
            try:
                _idem_module.record_success(  # type: ignore[attr-defined]
                    db,
                    key=job.idempotency_key,
                    youtube_video_id=video_id,
                )
            except Exception as exc:
                log.warning(
                    "upload_dispatch: idempotency.record_success raised (%s); "
                    "row already marked completed",
                    exc,
                )

        # ── 12) Thumbnail step (Full Video only) ───────────────────
        # Postiz never gets a YouTube thumbnails.set call — video_id is a
        # Postiz post id (not a YouTube video id) and there are no YT creds.
        if publish_kind == "video" and upload_path != "postiz":
            _maybe_set_thumbnail(
                db, job, creds, video_id, gcid, user_id,
            )
        else:
            log.info(
                "upload_dispatch: job=%d publish_kind=short — skipping thumbnail "
                "(YouTube API does not support custom thumbnails on Shorts)",
                job.id,
            )

        log.info(
            "upload_dispatch: job=%d completed video_id=%s bytes_uploaded=%d path=%s kind=%s",
            job.id, video_id, job.bytes_uploaded or 0, upload_path, publish_kind,
        )
        upload_succeeded = True   # keep the bucket reservation (real upload)
        _emit_stage(_stage_env, "upload", "exited", video_id[:24])
        return Outcome.COMPLETED

    except StaleClaim as exc:
        # We lost the lease mid-flight; the reaper handed the job to a
        # new owner. Abort silently — every write we could make would
        # no-op against the fence, and the new owner is driving.
        log.warning("upload_dispatch: stale claim — aborting: %s", exc)
        return Outcome.SKIPPED
    except UploadJobNotFound as exc:
        log.error("upload_dispatch: %s", exc)
        if not durable:
            raise
        return Outcome.SKIPPED
    except PlanTierViolation as exc:
        # Permanent — refund credits + mark failed; user must change.
        log.error("upload_dispatch: PlanTierViolation: %s", exc)
        _refund_and_fail(db, job, user_id, credit_cost, str(exc),
                         worker_id=worker_id)
        if not durable:
            raise
        return Outcome.FAILED
    except OAuthTokenMissing as exc:
        # Permanent — user must re-link the channel.
        log.error("upload_dispatch: OAuthTokenMissing: %s", exc)
        _refund_and_fail(db, job, user_id, credit_cost, str(exc),
                         worker_id=worker_id)
        if not durable:
            raise
        return Outcome.FAILED
    except OwnershipViolation as exc:
        # ISOLATION INVARIANT I3 — permanent, NEVER retry. A mis-routed
        # unit (channel not owned by the job's user) is a bug, not a
        # transient fault; retrying would just re-attempt the same illegal
        # delivery. Refund + terminal-fail so it surfaces loudly and the
        # video is never published to the wrong channel.
        log.error("upload_dispatch: OwnershipViolation (BLOCKED delivery): %s", exc)
        _refund_and_fail(db, job, user_id, credit_cost, str(exc),
                         worker_id=worker_id)
        if not durable:
            raise
        return Outcome.FAILED
    except uploader_v2.QuotaExceededError as exc:
        log.warning("upload_dispatch: QuotaExceededError on direct path: %s", exc)
        # NB: the burn_log row for videos.insert was already written
        # inside uploader_v2.upload_video at the exception site
        # (predicted=1600, outcome='quota_exceeded', was_quota_exceeded=True).
        # We deliberately do NOT call _burn_park_quota here — that
        # would write a second was_quota_exceeded row for the same
        # physical event and inflate the dashboard's "quota_exceeded today"
        # panel by 2× for every 403 from videos.insert.
        _park_quota(db, job, user_id, credit_cost, worker_id=worker_id)
        return Outcome.PARKED_QUOTA
    except (uploader_v2.TransientUploadError,
            rtmp_agent_v2.TransientRtmpUploadError,
            DeadlineExceeded) as exc:
        log.warning("upload_dispatch: transient: %s", exc)
        if durable:
            # Durable path: requeue with backoff (progress preserved).
            # The old scheduler used to catch the re-raise below and
            # overwrite this to 'failed' — THE critical bug. The worker
            # never does that; it only reads the Outcome.
            return _retry_or_exhaust(db, job, worker_id, user_id,
                                     credit_cost, str(exc))
        _persist_transient(db, job, str(exc))
        raise
    except uploader_v2.UploadError as exc:
        log.exception("upload_dispatch: permanent UploadError: %s", exc)
        _refund_and_fail(db, job, user_id, credit_cost, str(exc),
                         worker_id=worker_id)
        if not durable:
            raise
        return Outcome.FAILED
    except rtmp_agent_v2.RtmpUploadError as exc:
        log.exception("upload_dispatch: permanent RtmpUploadError: %s", exc)
        _refund_and_fail(db, job, user_id, credit_cost, str(exc),
                         worker_id=worker_id)
        if not durable:
            raise
        return Outcome.FAILED
    except Exception as exc:
        # Postiz auth/permission failures (401/403 — bad/expired key or no
        # subscription) are PERMANENT: fail fast and refund rather than
        # burning the retry budget against a key that will never work.
        from clients.postiz import PostizAuthError as _PostizAuthError
        if isinstance(exc, _PostizAuthError):
            log.error("upload_dispatch: permanent Postiz auth failure: %s", exc)
            _refund_and_fail(db, job, user_id, credit_cost, str(exc),
                             worker_id=worker_id)
            if not durable:
                raise
            return Outcome.FAILED
        log.exception("upload_dispatch: unexpected error: %s", exc)
        if durable:
            # KEY INVERSION (Wave 1): an unexpected exception (DB blip,
            # R2 hiccup, our own bug) is treated as TRANSIENT — retried
            # with backoff and bounded by the attempts cap — instead of
            # a terminal, credit-burning failure.
            return _retry_or_exhaust(db, job, worker_id, user_id,
                                     credit_cost, str(exc))
        _refund_and_fail(db, job, user_id, credit_cost, str(exc))
        raise
    finally:
        # Factory live-view hygiene: if this attempt entered a station but did
        # not complete (failed / parked / stale-claim / abandoned), emit a
        # terminal event so the unit is removed from the station's live
        # occupancy and never lingers as a ghost card. Success already emitted
        # its own 'exited' above (so it is skipped here).
        if _stage_env is not None and _cur_stage and not upload_succeeded:
            _emit_stage(_stage_env, _cur_stage, "failed")
        # Refund the videos.insert (direct) 100/day-bucket reservation when
        # this attempt reserved it but did NOT complete (failed / parked /
        # quota-403 / stale-claim / abandoned). Each retry re-reserves, so
        # only the attempt that actually completes leaves +1 in the bucket.
        if upload_reserved and not upload_succeeded:
            _quota_refund_upload(db)
        if work_dir is not None and os.path.isdir(work_dir):
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass
        try:
            db.close()
        except Exception:
            pass


# ─── Path branches ──────────────────────────────────────────────────


def _do_direct_upload(
    db, job: models.UploadJobV2, creds, branded_local: str,
    *, title: str, description: str, tags: list[str],
    category_id: str, made_for_kids: bool,
    publish_kind: str, google_channel_id: str,
    user_id: Optional[int],
    worker_id: Optional[str] = None,
    deadline: Optional[float] = None,
) -> dict:
    """Run the Direct ``videos.insert`` resumable upload, with the
    upload_uri + bytes_uploaded checkpoints persisted to the DB so
    a process restart can resume. In durable mode the checkpoints are
    fenced and each chunk enforces the cooperative deadline."""

    def _on_uri(new_uri: str) -> None:
        """Persist the resumable session URI after the first negotiation.
        Closes the DISCOVERY §2 gap. Fenced in durable mode (a fence
        miss here is harmless — the URI just isn't checkpointed)."""
        if worker_id is not None:
            _guarded_job_update(
                db, job.id, worker_id,
                {"upload_uri": (new_uri[:2000] if new_uri else None)},
            )
            return
        try:
            job.upload_uri = new_uri[:2000] if new_uri else None
            db.add(job)
            db.commit()
        except Exception:
            db.rollback()

    def _on_bytes(uploaded: int) -> None:
        """Checkpoint bytes_uploaded after each chunk. Durable mode:
        enforce the cooperative deadline, and abort promptly via
        StaleClaim on a TRUE fence miss (we were reaped — stop burning
        bandwidth on a job someone else now owns). A DB blip (None)
        fails open: the upload continues, only the checkpoint is lost."""
        _check_deadline(deadline, job.id)
        if worker_id is not None:
            ok = _guarded_job_update(
                db, job.id, worker_id,
                {"bytes_uploaded": int(uploaded or 0)},
            )
            if ok is False:
                raise StaleClaim(f"job={job.id} reaped mid-upload")
            return
        try:
            job.bytes_uploaded = int(uploaded or 0)
            db.add(job)
            db.commit()
        except Exception:
            db.rollback()

    # UploadJobV2 now carries privacy_status + publish_at (populated by
    # fan-out from the PublishRequest), so the user's public/unlisted/
    # private choice is honoured. ``getattr`` + 'private' fallback stays
    # as defence for any legacy row written before the columns existed.
    privacy_status = (getattr(job, "privacy_status", None) or "private")
    publish_at = getattr(job, "publish_at", None)

    return uploader_v2.upload_video(
        creds=creds,
        branded_path=branded_local,
        title=title,
        description=description,
        tags=tags,
        category_id=category_id,
        privacy_status=privacy_status,
        publish_at=publish_at,
        made_for_kids=made_for_kids,
        publish_kind=publish_kind,
        progress_cb=None,
        upload_uri=(job.upload_uri or None),
        on_uri_obtained=_on_uri,
        on_bytes_uploaded=_on_bytes,
        db=db,
        upload_job_id=_log_attr_upload_job_id(job.id),
        upload_job_id_v2=int(job.id),
        user_id=user_id,
        channel_id=job.channel_id,
        google_channel_id=google_channel_id,
        clip_id=job.clip_id,
    )


def _do_rtmp_upload(
    db, job: models.UploadJobV2, creds, branded_local: str,
    *, title: str, description: str, publish_kind: str,
    google_channel_id: str, user_id: Optional[int],
    worker_id: Optional[str] = None,
    deadline: Optional[float] = None,
) -> dict:
    """Run the RTMP path via the v2 wrapper. Returns the same shape as
    the Direct path so the caller can persist uniformly."""

    def _on_bytes(uploaded: int) -> None:
        _check_deadline(deadline, job.id)
        if worker_id is not None:
            ok = _guarded_job_update(
                db, job.id, worker_id,
                {"bytes_uploaded": int(uploaded or 0)},
            )
            if ok is False:
                raise StaleClaim(f"job={job.id} reaped mid-push")
            return
        try:
            job.bytes_uploaded = int(uploaded or 0)
            db.add(job)
            db.commit()
        except Exception:
            db.rollback()

    # Use clip duration when available so push_to_rtmp's 95% threshold
    # is meaningful. Falls back to None (the wrapper defaults to 60s).
    duration = None
    clip = _clip_for_job(db, job)
    if clip is not None:
        try:
            d = float(getattr(clip, "duration", 0) or 0)
            duration = d if d > 0 else None
        except Exception:
            duration = None

    privacy_status = (getattr(job, "privacy_status", None) or "private")

    return rtmp_agent_v2.upload_via_rtmp(
        creds=creds,
        branded_path=branded_local,
        title=title,
        description=description,
        privacy_status=privacy_status,
        publish_kind=publish_kind,
        expected_duration_s=duration,
        db=db,
        upload_job_id=_log_attr_upload_job_id(job.id),
        upload_job_id_v2=int(job.id),
        user_id=user_id,
        channel_id=job.channel_id,
        google_channel_id=google_channel_id,
        clip_id=job.clip_id,
        on_bytes_uploaded=_on_bytes,
    )


def _do_postiz_upload(
    db, job: models.UploadJobV2, branded_local: str,
    *, title: str, description: str, tags: list[str],
    publish_kind: str, user_id: Optional[int],
    worker_id: Optional[str] = None,
    deadline: Optional[float] = None,
) -> dict:
    """Deliver the branded artifact via Postiz instead of YouTube directly.

    Uploads the already-branded MP4 (logo + watermark baked in by the
    branding worker) to Postiz, then creates a 'post now' to the channel's
    bound Postiz integration carrying the composed title/description (with
    per-channel socials) + tags. Returns ``{video_id, bytes_uploaded}`` to
    match the direct/rtmp helpers so the success path persists uniformly —
    ``video_id`` is the Postiz post/media id, NOT a YouTube video id."""
    from clients import postiz as _postiz

    iid = (getattr(job, "postiz_integration_id", "") or "").strip()
    if not iid:
        raise RuntimeError(
            f"upload_dispatch: postiz job={job.id} has no postiz_integration_id "
            f"(channel not bound to a Postiz integration)"
        )

    _check_deadline(deadline, job.id)
    media = _postiz.upload_file(branded_local) or {}
    media_id = str(media.get("id") or "")
    media_path = media.get("path") or None
    if not media_id:
        raise RuntimeError(
            f"upload_dispatch: postiz upload returned no media id (job={job.id})"
        )

    _check_deadline(deadline, job.id)
    privacy_status = (getattr(job, "privacy_status", None) or "private")
    made_for_kids = bool(getattr(job, "made_for_kids", False) or False)
    # Only attach YouTube-specific settings (title/tags/made-for-kids) when the
    # Postiz integration IS YouTube. For Instagram/Facebook the post is just the
    # caption (`text`); sending yt_tags there could trip IG's hashtag rules.
    try:
        from services.platform_variants import resolve_publish_platform
        _is_yt = (resolve_publish_platform(db, job) == "youtube")
    except Exception:
        _is_yt = True
    resp = _postiz.schedule_post(
        integration_ids=[iid],
        text=description or "",
        media_id=media_id,
        media_path=media_path,
        type_="now",
        yt_title=((title or "")[:100] if _is_yt else ""),
        yt_privacy=privacy_status,
        yt_tags=(list(tags or []) if _is_yt else []),
        yt_made_for_kids=(made_for_kids if _is_yt else False),
    )
    # Postiz returns either a dict or a list of created posts; extract a
    # stable id, falling back to the media id so the success contract
    # (non-empty video_id) always holds.
    post_id = ""
    if isinstance(resp, dict):
        post_id = str(resp.get("id") or "")
        if not post_id and isinstance(resp.get("posts"), list) and resp["posts"]:
            post_id = str((resp["posts"][0] or {}).get("id") or "")
    elif isinstance(resp, list) and resp:
        post_id = str((resp[0] or {}).get("id") or "")

    log.info(
        "upload_dispatch: job=%d postiz post created integration=%s post_id=%s",
        job.id, iid, (post_id or media_id),
    )
    return {"video_id": (post_id or media_id), "bytes_uploaded": 0}


def _maybe_set_thumbnail(
    db, job: models.UploadJobV2, creds, video_id: str,
    google_channel_id: str, user_id: Optional[int],
) -> None:
    """Reserve thumbnail quota independently, then set the thumbnail.

    Skipped entirely if no ``thumbnail_r2_key`` is set on the row.
    On quota exhaustion: log a warning and continue (the video itself
    is already up; we don't fail the whole job).
    """
    thumb_key = (job.thumbnail_r2_key or "").strip()
    if not thumb_key:
        log.info(
            "upload_dispatch: job=%d no thumbnail_r2_key — skipping thumbnails.set",
            job.id,
        )
        return

    if not _quota_reserve(db, COST_THUMBNAIL_SET):
        log.warning(
            "upload_dispatch: job=%d thumbnail quota exhausted — skipping "
            "thumbnails.set (video itself uploaded OK; user may set the thumbnail manually)",
            job.id,
        )
        # Burn-log the would-have-been thumbnails.set so the
        # predicted-vs-actual ledger reflects the deferred 50u burn
        # (was_quota_exceeded=True) — brief §2.
        try:
            burn_log.log_predicted_and_actual(
                db,
                upload_job_id=int(job.id),
                operation="thumbnails.set",
                predicted_cost=COST_THUMBNAIL_SET,
                http_status=403,
                observed_outcome="quota_exceeded",
            )
        except Exception:
            pass
        return

    # Download the thumbnail to local disk for the set call.
    tmp_dir = tempfile.mkdtemp(prefix="kaizer_thumb_")
    try:
        local_thumb = os.path.join(tmp_dir, "thumb.jpg")
        try:
            get_storage_provider().download(thumb_key, local_thumb)
        except Exception as exc:
            log.warning(
                "upload_dispatch: job=%d thumbnail R2 download failed (%s); skipping",
                job.id, exc,
            )
            return
        try:
            uploader_v2.set_thumbnail(
                creds=creds,
                video_id=video_id,
                thumb_local_path=local_thumb,
                db=db,
                upload_job_id=_log_attr_upload_job_id(job.id),
                upload_job_id_v2=int(job.id),
                user_id=user_id,
                channel_id=job.channel_id,
                google_channel_id=google_channel_id,
                clip_id=job.clip_id,
            )
        except Exception as exc:
            # Thumbnail failure is non-fatal — the video itself is live.
            log.warning(
                "upload_dispatch: job=%d thumbnails.set failed (%s); video remains uploaded",
                job.id, exc,
            )
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


# ─── Failure handling ──────────────────────────────────────────────


def _burn_park_quota(
    db,
    job: Optional[models.UploadJobV2],
    upload_path: str,
    publish_kind: str,
) -> None:
    """Write a ``quota_burn_log`` row for the inline quota-park case
    (``_quota_reserve`` returned False BEFORE any YouTube API call).

    The row captures the call that WOULD have been made:
      * direct  → ``videos.insert``           (1,600 units)
      * rtmp    → ``liveBroadcasts.insert``   (50 units; first of the
                                              RTMP triple — none of the
                                              3 ran because we never got
                                              past the gate)

    ``observed_outcome='quota_exceeded'`` flips
    ``was_quota_exceeded=True`` automatically (see burn_log.py:120).
    """
    if job is None or db is None:
        return
    if upload_path == "direct":
        op = "videos.insert"
        cost = 100   # repriced 1,600 → 100 on 2025-12-04
    elif upload_path == "rtmp":
        op = "liveBroadcasts.insert"
        cost = 50
    else:
        # Unknown path — still log a row so the bucket isn't silent.
        op = "videos.insert"
        cost = 100
    try:
        burn_log.log_predicted_and_actual(
            db,
            upload_job_id=int(job.id),
            operation=op,
            predicted_cost=cost,
            http_status=403,
            observed_outcome="quota_exceeded",
        )
    except Exception:
        pass


def _persist_success(
    db,
    job: models.UploadJobV2,
    video_id: str,
    bytes_done: int,
) -> None:
    """Record a landed upload. NOT worker-fenced (a recorded YouTube
    video id is truth and blocking it would cause duplicate uploads) —
    guarded only against overwriting a DIFFERENT already-recorded id.
    Clears the claim columns so the row stops occupying a lease."""
    vid = (video_id or "")[:32]
    values: dict = {
        "youtube_video_id": vid,
        "status": "completed",
        "finished_at": datetime.utcnow(),
        "claimed_by": None,
        "lease_expires_at": None,
    }
    if bytes_done > 0:
        values["bytes_uploaded"] = int(bytes_done)
    try:
        res = db.execute(
            _sa_update(models.UploadJobV2)
            .where(
                models.UploadJobV2.id == int(job.id),
                (
                    models.UploadJobV2.youtube_video_id.is_(None)
                    | (models.UploadJobV2.youtube_video_id == vid)
                ),
            )
            .values(**values)
        )
        db.commit()
        if not res.rowcount:
            log.error(
                "upload_dispatch: job=%d uploaded video_id=%s but a DIFFERENT "
                "video id is already recorded — possible duplicate upload; "
                "investigate publish_attempts for this idempotency_key",
                job.id, vid,
            )
    except Exception:
        db.rollback()
        log.exception(
            "upload_dispatch: _persist_success failed for job=%d", job.id
        )
        raise


def _retry_or_exhaust(
    db,
    job: Optional[models.UploadJobV2],
    worker_id: Optional[str],
    user_id: Optional[int],
    credit_cost: int,
    message: str,
) -> Outcome:
    """Durable-mode transient handler: requeue with backoff, or — when
    the attempts cap is spent — terminal-fail with the idempotent
    refund. job_queue owns the backoff math + fencing."""
    if job is None:
        return Outcome.FAILED
    state = job_queue.requeue_for_retry(job.id, worker_id, message)
    if state == "requeued":
        return Outcome.RETRY
    if state == "exhausted":
        _refund_and_fail(
            db, job, user_id, credit_cost,
            f"retries exhausted ({job_queue.max_attempts()}): {message}"[:1000],
            worker_id=worker_id,
        )
        return Outcome.FAILED
    # 'stale' — another actor owns the row now.
    return Outcome.SKIPPED


def fail_terminal(
    upload_job_id: int,
    worker_id: Optional[str],
    reason: str,
) -> None:
    """Public terminal-fail used by the publish worker's watchdog and
    the cron's exhausted-attempts sweep. Loads its own session, resolves
    the user for the refund, and applies the fenced fail."""
    db = SessionLocal()
    try:
        job = db.query(models.UploadJobV2).filter(
            models.UploadJobV2.id == int(upload_job_id)
        ).first()
        if job is None:
            return
        try:
            user_id, _tier = _resolve_user_and_plan(db, job)
        except Exception:
            user_id = job.user_id if getattr(job, "user_id", None) else None
        cost = int(job.predicted_credit_cost or 0)
        _refund_and_fail(db, job, user_id, cost, reason, worker_id=worker_id)
    finally:
        try:
            db.close()
        except Exception:
            pass


def _park_quota(
    db,
    job: Optional[models.UploadJobV2],
    user_id: Optional[int],
    credit_cost: int,
    worker_id: Optional[str] = None,
) -> None:
    """Quota-exhausted: flip status to 'parked_quota' (fenced when a
    worker_id is given — a stale worker must not park a reclaimed job),
    then refund credits (brief §9 says no credit burn). The un-parker
    cron re-reserves credits when the quota window resets."""
    if job is None or user_id is None:
        return
    try:
        stmt = _sa_update(models.UploadJobV2).where(
            models.UploadJobV2.id == int(job.id)
        )
        if worker_id is not None:
            stmt = stmt.where(models.UploadJobV2.claimed_by == worker_id[:64])
        res = db.execute(stmt.values(
            status="parked_quota",
            last_error="daily quota exhausted; will retry when the window resets",
            claimed_by=None,
            lease_expires_at=None,
        ))
        if not res.rowcount:
            db.rollback()
            log.warning(
                "upload_dispatch: _park_quota fence miss for job=%d "
                "(reclaimed by another worker) — skipping refund", job.id,
            )
            return
        if credit_cost > 0 and not _refund_already(db, user_id, job.id):
            credits.refund(
                db, user_id=user_id, cost=credit_cost, upload_job_id=job.id,
            )
        db.commit()
    except Exception:
        db.rollback()
        log.exception(
            "upload_dispatch: _park_quota failed for job=%d",
            getattr(job, "id", -1),
        )


def _persist_transient(
    db,
    job: Optional[models.UploadJobV2],
    message: str,
) -> None:
    """Transient: bump attempts + persist progress + re-queue.

    The caller (scheduler) re-raises this so the dispatcher's retry
    loop picks it up; we just persist state here.
    """
    if job is None:
        return
    try:
        job.attempts = int(job.attempts or 0) + 1
        job.status = "queued"
        job.last_error = (message or "transient")[:1000]
        db.add(job)
        db.commit()
    except Exception:
        db.rollback()


def _refund_and_fail(
    db,
    job: Optional[models.UploadJobV2],
    user_id: Optional[int],
    credit_cost: int,
    message: str,
    worker_id: Optional[str] = None,
) -> None:
    """Permanent failure: flip status to 'failed' (fenced when a
    worker_id is given), then refund credits in the same transaction so
    the user isn't charged for our bug (brief §9 implicit). Fence miss
    (the row was reclaimed) → no write, no refund — the new owner is
    authoritative."""
    if job is None:
        return
    try:
        stmt = _sa_update(models.UploadJobV2).where(
            models.UploadJobV2.id == int(job.id)
        )
        if worker_id is not None:
            stmt = stmt.where(models.UploadJobV2.claimed_by == worker_id[:64])
        res = db.execute(stmt.values(
            status="failed",
            last_error=(message or "permanent failure")[:1000],
            finished_at=datetime.utcnow(),
            claimed_by=None,
            lease_expires_at=None,
        ))
        if not res.rowcount:
            db.rollback()
            log.warning(
                "upload_dispatch: _refund_and_fail fence miss for job=%d "
                "(reclaimed by another worker) — skipping refund", job.id,
            )
            return
        if (
            user_id is not None
            and credit_cost > 0
            and not _refund_already(db, user_id, job.id)
        ):
            credits.refund(
                db, user_id=user_id, cost=credit_cost, upload_job_id=job.id,
            )
        db.commit()
    except Exception:
        db.rollback()
        log.exception(
            "upload_dispatch: _refund_and_fail failed for job=%d",
            getattr(job, "id", -1),
        )


__all__ = [
    "process",
    "fail_terminal",
    "Outcome",
    "PlanTierViolation",
    "UploadJobNotFound",
    "OAuthTokenMissing",
    "OwnershipViolation",
    "StaleClaim",
    "DeadlineExceeded",
    "CREDIT_COST",
    "QUOTA_UNITS",
    "QUOTA_UPLOAD_ONLY",
    "COST_THUMBNAIL_SET",
]
