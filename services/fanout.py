"""Fanout service — Phase 1.B.

Owns the ``POST /api/publish-tasks`` business logic. The router (in
``routers/publish_tasks.py``) is the only file that calls into this
module; everything else is internal.

REQUEST FLOW (CONTRACTS.md §4.1, brief §6.3)
--------------------------------------------
Per ``create_publish_task(db, user, request)``, inside ONE transaction:
  1. Resolve user.plan_tier; raise PlanTierMissingError if NULL.
  2. Plan-tier enforcement (Decision 6):
       - direct_path on a tier where direct_path_allowed is False -> 403
       - distinct channels in targets exceeds max_channels -> 403
       - daily-publish cap → warn-only in Phase 1 (F-agent gates fully in P2)
  3. MasterVideo MUST exist and be ``status='ready'``.
  4. Every target.channel_id MUST belong to the user.
  5. ``thumbnail_source`` is NULL iff ``publish_kind == 'short'`` (CONTRACTS §3.3).
  6. Compute predicted_credit_cost per target (Decision 4):
        direct → 16 cr, rtmp → 2 cr (both for video and short).
  7. Compute predicted_quota_units per target (brief §1 + §6 + task spec):
        direct + video → 1650 (1600 insert + 50 thumb)
        direct + short → 1600 (no thumb)
        rtmp + video  → 200  (3×50 + 50 thumb)
        rtmp + short  → 150  (3×50, no thumb)
  8. credits.reserve(...) per target. If any later target fails, refund
     every already-reserved target before re-raising (caller's transaction
     also rolls back, so this is mainly belt-and-braces for the
     dataclass return values).
  9. Insert one PublishTask row with status='fanning_out'.
 10. Insert N UploadJobV2 rows. ``idempotency_key`` is computed via
     ``services.idempotency.compute_key`` (NEVER inlined here so the
     F-agent can change the formula in Phase 2 with one file edit).
     ``publish_version`` = brand[:8] + ':' + seo[:8] + ':' + metadata[:8]
     concatenation per task spec.
 11. Resolve oauth_token_id by looking up OAuthToken WHERE channel_id=cid.
     If missing -> OAuthTokenMissingError.
 12. PublishTask.status = 'dispatched'.
 13. scheduler_enqueue(upload_job_id, priority, user_id, plan_tier_name)
     per job. The Scheduler agent (C) is building scheduler.py in
     parallel; this module imports it lazily and falls back to a no-op
     warning log if the import fails so Phase 1.B is mergeable
     independently of Phase 1.C.
 14. Commit.

IDEMPOTENCY (brief §2)
----------------------
If a UploadJobV2 with the same idempotency_key already exists, we DO
NOT create a duplicate. We raise DuplicatePublishVersionError carrying
the existing PublishTask id; the router maps this to HTTP 200 (this is
dedupe-by-design, not an error).

PUBLISH VERSION FORMULA (Decision 10)
-------------------------------------
Phase 1: caller supplies (brand_profile_version, seo_version,
metadata_version) as opaque stable strings; we concatenate the
first 8 chars of each separated by ':' into a stored ``publish_version``
column (40 chars headroom). The F-agent will redefine ``seo_version``
canonically in Phase 2 and replay-validate every persisted row at
boot. The runbook drains the publish queue before that switch so
in-flight rows can't mismatch.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import models
from services import credits as credits_svc
from services import idempotency as idempotency_svc

log = logging.getLogger(__name__)


# ─── Scheduler import (Phase 1.C is in-flight; degrade gracefully) ────────
#
# Scheduler agent C is shipping ``services/scheduler.py`` (or a
# ``services/scheduler/`` package) in parallel with this file. We import
# lazily so a startup-time absence does NOT prevent the router from
# loading. The fallback no-op keeps Phase 1.B independently mergeable.

def _scheduler_enqueue_fallback(
    upload_job_id: int,
    priority: str,
    user_id: int,
    plan_tier_name: str,
) -> None:
    log.warning(
        "scheduler_enqueue fallback (Phase 1.C not yet landed): "
        "job=%d priority=%s user=%d tier=%s",
        upload_job_id, priority, user_id, plan_tier_name,
    )


def _resolve_scheduler_enqueue():
    """Try in priority order: package, module, sibling ``scheduler_service``.
    Returns the chosen callable. Logs the choice once at startup time."""
    try:
        from services.scheduler import scheduler_enqueue  # type: ignore
        log.info("fanout: using services.scheduler.scheduler_enqueue")
        return scheduler_enqueue
    except Exception:
        pass
    try:
        from services.scheduler import enqueue as scheduler_enqueue  # type: ignore
        log.info("fanout: using services.scheduler.enqueue")
        return scheduler_enqueue
    except Exception:
        pass
    try:
        from services.scheduler.scheduler_service import enqueue as scheduler_enqueue  # type: ignore
        log.info("fanout: using services.scheduler.scheduler_service.enqueue")
        return scheduler_enqueue
    except Exception:
        pass
    log.warning(
        "fanout: scheduler module not present; using no-op fallback. "
        "Jobs will be persisted to upload_jobs_v2 but not dispatched."
    )
    return _scheduler_enqueue_fallback


_SCHEDULER_ENQUEUE = _resolve_scheduler_enqueue()


# ─── Request / response dataclasses (router builds these from Pydantic) ───


@dataclass
class FanoutTarget:
    channel_id: int
    upload_path: str            # 'direct' | 'rtmp' | 'postiz'
    publish_kind: str           # 'video' | 'short'
    brand_profile_id: Optional[int] = None
    thumbnail_source: Optional[str] = None   # NULL iff publish_kind == 'short'
    thumbnail_r2_key: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    # Publish intent carried to the UploadJobV2 so dispatch honours the
    # user's choice instead of hard-defaulting to private.
    privacy_status: str = "private"   # 'public' | 'unlisted' | 'private'
    brand_profile_version: str = ""
    seo_version: str = ""
    metadata_version: str = ""
    # Set by the channel-wise override below when the channel is
    # configured for Postiz delivery. NULL for direct/rtmp targets.
    postiz_integration_id: Optional[str] = None
    # Branding mode: 'per_channel' (overlay this channel's logo+watermark at
    # upload, default) | 'as_is' (source already branded — upload verbatim).
    brand_mode: str = "per_channel"
    # Logo/watermark placement when overlaying: 'template' (use the template's
    # marked slot) | 'channel' (use this channel's own position instead).
    brand_placement: str = "template"


@dataclass
class PublishTaskRequest:
    master_video_id: int
    targets: List[FanoutTarget] = field(default_factory=list)
    priority: str = "normal"     # 'critical' | 'high' | 'normal' | 'low'


@dataclass
class PublishTaskResult:
    publish_task_id: int
    upload_job_ids: List[int]
    predicted_credit_total: int
    predicted_quota_units_total: int
    quota_pre_flight_ok: bool   # Always True in Phase 1; F-agent gates in P2.
    # Channels skipped because this exact (master, channel, version) was
    # already published — the publish still succeeds for the fresh ones.
    skipped_count: int = 0


# ─── Exceptions (all carry a stable ``code`` the router maps to HTTP) ─────


class FanoutError(Exception):
    """Base class for Fanout service exceptions. ``code`` is a stable
    machine-readable error code the router exposes to the client; HTTP
    status is decided by the router (see ``routers/publish_tasks.py``)."""

    code: str = "fanout_error"

    def __init__(self, message: str, *, code: Optional[str] = None, **extra) -> None:
        super().__init__(message)
        if code:
            self.code = code
        self.extra = extra


class PlanTierMissingError(FanoutError):
    code = "plan_tier_unknown"


class PlanTierViolationError(FanoutError):
    code = "plan_tier_violation"


class ChannelOwnershipError(FanoutError):
    code = "channel_not_owned"


class MasterVideoNotReadyError(FanoutError):
    code = "master_video_not_ready"


class ThumbnailSourceMismatchError(FanoutError):
    code = "thumbnail_source_mismatch"


class OAuthTokenMissingError(FanoutError):
    code = "oauth_token_missing"


class DuplicatePublishVersionError(FanoutError):
    """Carry the existing PublishTask id so the router can return its
    existing result instead of a hard error (brief §2 idempotency)."""
    code = "duplicate_publish_version"

    def __init__(self, message: str, *, existing_publish_task_id: int) -> None:
        super().__init__(message, code="duplicate_publish_version")
        self.existing_publish_task_id = existing_publish_task_id


# ─── Credit + quota cost tables (Decision 4 + task spec; DO NOT INVENT) ───


_CREDIT_COST = {
    ("direct", "video"): 16,
    ("direct", "short"): 16,
    ("rtmp",   "video"): 2,
    ("rtmp",   "short"): 2,
    # Postiz hands the branded artifact to a 3rd party; it never touches
    # the YouTube Data API, so 0 YouTube quota and 0 credits.
    ("postiz", "video"): 0,
    ("postiz", "short"): 0,
}

_QUOTA_UNITS = {
    ("direct", "video"): 1650,   # 1600 insert + 50 thumb
    ("direct", "short"): 1600,   # no thumb on Shorts
    ("rtmp",   "video"): 200,    # 3 × 50 (insert/insert/bind) + 50 thumb
    ("rtmp",   "short"): 150,    # 3 × 50, no thumb
    ("postiz", "video"): 0,      # no YouTube API call
    ("postiz", "short"): 0,
}

_VALID_PATHS = {"direct", "rtmp", "postiz"}
_VALID_KINDS = {"video", "short"}
_VALID_PRIORITIES = {"critical", "high", "normal", "low"}
_VALID_THUMB_SOURCES = {"pipeline_generated", "user_override", "user_uploaded"}


def _cost_pair(path: str, kind: str) -> tuple[int, int]:
    key = (path, kind)
    if key not in _CREDIT_COST:
        raise FanoutError(
            f"unsupported (upload_path={path!r}, publish_kind={kind!r}) combo",
            code="invalid_target",
        )
    return _CREDIT_COST[key], _QUOTA_UNITS[key]


def _short_publish_version(brand_v: str, seo_v: str, metadata_v: str) -> str:
    """brand_profile_version[:8] + ':' + seo_version[:8] + ':' + metadata_version[:8].

    Total length ≤ 26 chars, well under the 40-char column cap. F-agent
    Phase 2 may swap the formula; this is the one place to change it.
    """
    def _take(s: str) -> str:
        return (s or "")[:8]
    return f"{_take(brand_v)}:{_take(seo_v)}:{_take(metadata_v)}"


# ─── Plan-tier validation ─────────────────────────────────────────────────


def _enforce_plan_tier(
    plan_tier: "models.PlanTier",
    targets: List[FanoutTarget],
) -> None:
    if plan_tier is None:
        raise PlanTierMissingError(
            "user.plan_tier is NULL; backfill is required",
            code="plan_tier_unknown",
        )

    # Decision 6: Free tier rejects Direct.
    if not bool(plan_tier.direct_path_allowed):
        offending = [t for t in targets if t.upload_path == "direct"]
        if offending:
            raise PlanTierViolationError(
                f"plan_tier={plan_tier.name!r} disallows upload_path='direct' "
                f"({len(offending)} target(s) requested it)",
                code="direct_path_requires_pro",
                offending_channel_ids=[t.channel_id for t in offending],
            )

    # Decision 6: max channels per publish task.
    if plan_tier.max_channels is not None and plan_tier.max_channels >= 0:
        distinct_channels = {t.channel_id for t in targets}
        if len(distinct_channels) > plan_tier.max_channels:
            raise PlanTierViolationError(
                f"plan_tier={plan_tier.name!r} max_channels={plan_tier.max_channels} "
                f"but {len(distinct_channels)} distinct channel(s) requested",
                code="channel_cap_exceeded",
                requested=len(distinct_channels),
                max_channels=int(plan_tier.max_channels),
            )

    # Daily-publish cap (Phase 1 only logs; F-agent gates in Phase 2).
    if plan_tier.max_publishes_per_day is not None and plan_tier.max_publishes_per_day >= 0:
        log.info(
            "fanout: plan_tier=%s max_publishes_per_day=%d — gating deferred to F-agent (Phase 2)",
            plan_tier.name, plan_tier.max_publishes_per_day,
        )


def _validate_target_shape(t: FanoutTarget) -> None:
    if t.upload_path not in _VALID_PATHS:
        raise FanoutError(
            f"upload_path must be one of {_VALID_PATHS}; got {t.upload_path!r}",
            code="invalid_upload_path",
        )
    if t.publish_kind not in _VALID_KINDS:
        raise FanoutError(
            f"publish_kind must be one of {_VALID_KINDS}; got {t.publish_kind!r}",
            code="invalid_publish_kind",
        )
    # Postiz delivery sets no YouTube thumbnail (the YouTube Data API is
    # never called), so the native thumbnail-source rules don't apply.
    if t.upload_path == "postiz":
        return
    # CONTRACTS.md §3.3 / brief §2 "Thumbnails": NULL iff publish_kind=='short'.
    if t.publish_kind == "short":
        if t.thumbnail_source is not None or t.thumbnail_r2_key is not None:
            raise ThumbnailSourceMismatchError(
                "publish_kind='short' must have thumbnail_source=None and thumbnail_r2_key=None",
                code="thumbnail_source_mismatch",
                channel_id=t.channel_id,
            )
    else:  # video
        if t.thumbnail_source is None:
            raise ThumbnailSourceMismatchError(
                "publish_kind='video' requires a non-null thumbnail_source",
                code="thumbnail_source_mismatch",
                channel_id=t.channel_id,
            )
        if t.thumbnail_source not in _VALID_THUMB_SOURCES:
            raise ThumbnailSourceMismatchError(
                f"thumbnail_source must be one of {_VALID_THUMB_SOURCES}; got {t.thumbnail_source!r}",
                code="thumbnail_source_mismatch",
                channel_id=t.channel_id,
            )


# ─── Channel + oauth-token resolution ─────────────────────────────────────


def _load_channels_for_user(
    db: Session, user_id: int, channel_ids: List[int]
) -> dict[int, "models.Channel"]:
    if not channel_ids:
        return {}
    rows = (
        db.query(models.Channel)
        .filter(models.Channel.id.in_(channel_ids))
        .all()
    )
    by_id: dict[int, models.Channel] = {}
    for ch in rows:
        if ch.user_id != user_id:
            raise ChannelOwnershipError(
                f"channel_id={ch.id} does not belong to user_id={user_id}",
                code="channel_not_owned",
                channel_id=ch.id,
            )
        by_id[ch.id] = ch
    missing = [cid for cid in channel_ids if cid not in by_id]
    if missing:
        raise ChannelOwnershipError(
            f"channel_id(s) {missing} not found",
            code="channel_not_owned",
            missing=missing,
        )
    return by_id


def _resolve_oauth_token_id(db: Session, channel_id: int) -> int:
    tok = (
        db.query(models.OAuthToken)
        .filter(models.OAuthToken.channel_id == channel_id)
        .first()
    )
    if tok is None:
        raise OAuthTokenMissingError(
            f"channel_id={channel_id} has no OAuthToken row",
            code="oauth_token_missing",
            channel_id=channel_id,
        )
    return int(tok.id)


# ─── MAIN ENTRY POINT ─────────────────────────────────────────────────────


def create_publish_task(
    db: Session,
    user: "models.User",
    request: PublishTaskRequest,
) -> PublishTaskResult:
    """Atomic PublishTask + N UploadJobV2 creation + scheduler dispatch.

    All exceptions raised here inherit ``FanoutError`` and carry a
    stable ``code``. The router maps them to HTTP status codes.
    """
    if request.priority not in _VALID_PRIORITIES:
        raise FanoutError(
            f"priority must be one of {_VALID_PRIORITIES}; got {request.priority!r}",
            code="invalid_priority",
        )
    if not request.targets:
        raise FanoutError(
            "targets must contain at least one channel",
            code="no_targets",
        )

    # Step 1 + 2 — plan tier.
    plan_tier = getattr(user, "plan_tier", None)
    _enforce_plan_tier(plan_tier, request.targets)

    # Step 3 — MasterVideo exists + ready.
    master = (
        db.query(models.MasterVideo)
        .filter(models.MasterVideo.id == request.master_video_id)
        .first()
    )
    if master is None:
        raise MasterVideoNotReadyError(
            f"MasterVideo id={request.master_video_id} not found",
            code="master_video_not_ready",
        )
    if master.status != "ready":
        raise MasterVideoNotReadyError(
            f"MasterVideo id={request.master_video_id} status={master.status!r} (need 'ready')",
            code="master_video_not_ready",
            status=master.status,
        )

    # Step 4 — channel ownership.
    distinct_channel_ids = list({t.channel_id for t in request.targets})
    by_id = _load_channels_for_user(db, user.id, distinct_channel_ids)

    # Step 4b — channel-wise Postiz delivery + production auto-fallback.
    #
    #   testing mode (default): a channel is delivered via Postiz ONLY when
    #     the user explicitly set upload_provider='postiz' AND bound a
    #     postiz_integration_id. Native (our quota) otherwise. NO fallback.
    #
    #   production mode: the same explicit rule, PLUS auto-fallback — a
    #     NATIVE target whose channel has a bound postiz_integration_id is
    #     routed to Postiz when the YouTube daily quota can't cover it, so
    #     publishing never stalls while our quota is low. We debit a running
    #     "remaining" as we walk targets so a multi-clip publish only falls
    #     back the overflow. Derived server-side (never from the client) so
    #     the request contract stays direct/rtmp and the path is inert until
    #     a channel is bound.
    from system_settings import get_delivery_mode
    _delivery_mode = get_delivery_mode(db)
    _yt_remaining = None
    if _delivery_mode == "production":
        try:
            from youtube import quota_v2 as _quota_v2
            _yt_remaining = int(_quota_v2.snapshot(db).get("remaining", 0))
        except Exception:
            log.warning("fanout: quota snapshot failed; auto-fallback disabled "
                        "for this request", exc_info=True)
            _yt_remaining = None  # fail safe → behave like testing (no fallback)

    for t in request.targets:
        ch = by_id.get(t.channel_id)
        if ch is None:
            continue
        _prov = (getattr(ch, "upload_provider", "") or "").strip().lower()
        _bound_iid = (getattr(ch, "postiz_integration_id", "") or "").strip()

        # Defense-in-depth: honour a binding ONLY if the integration is STILL
        # owned by THIS channel's user. Ownership can't legitimately cross
        # users, but this makes a stale/forged binding non-exploitable at
        # publish time — we fall through to native instead of ever publishing
        # to a channel the user doesn't own.
        if _bound_iid:
            from services.postiz_scope import team_user_ids as _pz_team
            _owner_team = _pz_team(db, ch.user_id)
            _owns = db.query(models.PostizIntegration.id).filter(
                models.PostizIntegration.integration_id == _bound_iid,
                models.PostizIntegration.user_id.in_(_owner_team),
            ).first()
            if not _owns:
                log.warning("fanout: channel=%s bound to postiz integration %r "
                            "not owned by the channel owner's team (user=%s) — "
                            "ignoring binding (native)",
                            t.channel_id, _bound_iid, ch.user_id)
                _bound_iid = ""

        if _prov == "postiz" and _bound_iid:
            # Explicit per-channel Postiz delivery (both modes).
            t.upload_path = "postiz"
            t.postiz_integration_id = _bound_iid
            continue

        # Native target. In production, walk a running YouTube-quota budget so
        # a multi-clip publish only falls back the OVERFLOW — and debit EVERY
        # native clip (bound or not) since they all consume our real quota,
        # otherwise unbound clips would hide the shortfall from bound ones.
        if _yt_remaining is not None:
            try:
                _, _qq = _cost_pair(t.upload_path, t.publish_kind)
            except Exception:
                _qq = 0  # let Step 5 shape-validation surface the real error
            if _bound_iid and _qq > _yt_remaining:
                t.upload_path = "postiz"
                t.postiz_integration_id = _bound_iid
                log.info("fanout: production auto-fallback channel=%s → postiz "
                         "(needed=%s quota, remaining=%s)",
                         t.channel_id, _qq, _yt_remaining)
            else:
                _yt_remaining -= _qq

    # Step 5 — per-target shape validation (thumbnail rules included).
    for t in request.targets:
        _validate_target_shape(t)

    # Resolve OAuth tokens up-front so we don't half-create then fail.
    oauth_by_channel: dict[int, int] = {}
    for cid in distinct_channel_ids:
        oauth_by_channel[cid] = _resolve_oauth_token_id(db, cid)

    # Step 6 + 7 — predicted costs per target.
    per_target_credit_cost: list[int] = []
    per_target_quota_units: list[int] = []
    for t in request.targets:
        cc, qq = _cost_pair(t.upload_path, t.publish_kind)
        per_target_credit_cost.append(cc)
        per_target_quota_units.append(qq)

    predicted_credit_total = sum(per_target_credit_cost)
    predicted_quota_units_total = sum(per_target_quota_units)

    # ── ALL VALIDATION DONE ───────────────────────────────────────────────
    # Beyond this point we start mutating the DB. The router runs us
    # inside a single SQLAlchemy session; if anything raises, the
    # router's exception handler rolls back the transaction (no
    # partial PublishTask, no leaked ledger rows).

    # Step 9 — create PublishTask row with status='fanning_out'.
    publish_task = models.PublishTask(
        user_id=user.id,
        master_video_id=request.master_video_id,
        priority=request.priority,
        status="fanning_out",
        target_count=len(request.targets),
        completed_count=0,
        failed_count=0,
    )
    db.add(publish_task)
    db.flush()  # populate publish_task.id

    # Step 10 — N UploadJobV2 rows.
    upload_job_ids: list[int] = []
    reserved_ledger_ids: list[tuple[int, int, int]] = []  # (user_id, cost, upload_job_id)

    # PER-CHANNEL idempotency pre-filter. A target already published with
    # this exact (master, channel, version) is SKIPPED — but brand-new
    # channels in the SAME request still publish. The old code raised on
    # the FIRST duplicate and aborted the WHOLE request, so adding one
    # already-published channel silently blocked every NEW channel too
    # (the user publishing the same clip to a different channel must be
    # allowed). These are pure reads — no mutation — so there is nothing
    # to roll back when every target turns out to be a duplicate.
    fresh: list = []
    skipped_dupes = 0
    existing_dupe_ptid = 0
    for t, credit_cost, quota_units in zip(
        request.targets, per_target_credit_cost, per_target_quota_units
    ):
        publish_version = _short_publish_version(
            t.brand_profile_version, t.seo_version, t.metadata_version,
        )
        idem_key = idempotency_svc.compute_key(
            master_video_id=request.master_video_id,
            channel_id=t.channel_id,
            publish_version=publish_version,
        )
        existing = (
            db.query(models.UploadJobV2)
            .filter(models.UploadJobV2.idempotency_key == idem_key)
            .first()
        )
        if existing is not None:
            status = (existing.status or "").strip().lower()
            if status in ("cancelled", "failed"):
                # The prior attempt for this (master, channel, version) is
                # DEAD — cancelled or permanently failed. It must NOT block a
                # genuine re-publish (the user cancelled it on purpose, or it
                # errored). Reclaim its idempotency_key by deleting the dead
                # row (publish_attempts cascade; credit_ledger / quota_burn_log
                # FKs are SET NULL), then publish this target fresh.
                db.delete(existing)
                db.flush()
                fresh.append((t, credit_cost, quota_units, publish_version, idem_key))
                continue
            # Completed or in-flight (queued/claimed/branding/uploading/
            # parked) → a REAL duplicate. Skip just this channel; the other
            # (new) channels in the request still publish.
            skipped_dupes += 1
            if not existing_dupe_ptid:
                existing_dupe_ptid = int(existing.publish_task_id)
            continue
        fresh.append((t, credit_cost, quota_units, publish_version, idem_key))

    # Every requested channel was already published with this version →
    # nothing new to do. Surface as dedupe so the router returns the
    # existing publish (the "already published" path).
    if not fresh:
        raise DuplicatePublishVersionError(
            f"all {len(request.targets)} target(s) already published with "
            f"this version (master_video_id={request.master_video_id})",
            existing_publish_task_id=existing_dupe_ptid,
        )

    # Anti-duplicate stagger (flag-gated, default OFF). When
    # KAIZER_STAGGER_MINUTES > 0, spread the PUBLIC go-live of a multi-channel
    # fan-out across spaced slots so YouTube doesn't see N near-identical
    # videos appear at once. Uses the EXISTING publish_at → YouTube native
    # scheduled-publish wiring (requires privacy=private, which we force for
    # staggered jobs). The first channel (idx 0) is unchanged; each subsequent
    # channel is pushed +N minutes. Never overrides a publish_at the caller
    # already set (e.g. Campaigns slots). OFF ⇒ byte-identical to before.
    try:
        _stagger_min = int(os.environ.get("KAIZER_STAGGER_MINUTES", "0") or "0")
    except ValueError:
        _stagger_min = 0
    _stagger_min = max(0, min(720, _stagger_min))
    _stagger_base = datetime.now(timezone.utc)

    for _idx, (t, credit_cost, quota_units, publish_version, idem_key) in enumerate(fresh):
        # Step 8 — reserve credits per target (writes one ledger row).
        # We reserve BEFORE creating the UploadJobV2 row so the
        # credit_ledger.upload_job_id can be NULL on a failure that
        # rolls back; F-agent's Phase 2 row-lock variant can keep this
        # ordering or invert it without affecting the contract.
        # Postiz delivery costs 0 credits / 0 YouTube quota, so it skips
        # the ledger entirely — credits_svc.reserve() rejects cost<=0 by
        # contract, and there's nothing to refund on failure.
        is_postiz = (t.upload_path == "postiz")
        reason = (
            "upload_direct" if t.upload_path == "direct"
            else ("upload_postiz" if is_postiz else "upload_rtmp")
        )
        if not is_postiz:
            try:
                credits_svc.reserve(
                    db,
                    user_id=user.id,
                    cost=credit_cost,
                    reason=reason,
                    upload_job_id=None,            # back-filled below
                    path=t.upload_path,
                    publish_kind=t.publish_kind,
                    predicted_quota_units=quota_units,
                )
            except credits_svc.InsufficientCreditsError:
                # Refund the already-reserved targets so user balance is
                # unchanged when the router catches and 402's. (The router
                # also rolls back the SAVEPOINT — this is belt + braces.)
                for u_id, cost, ujob_id in reserved_ledger_ids:
                    credits_svc.refund(
                        db, user_id=u_id, cost=cost, upload_job_id=ujob_id,
                    )
                raise

        # Effective publish time + privacy. Default = the caller's intent
        # (unchanged). When stagger is ON and the caller did NOT set an
        # explicit time, push every channel after the first to a spaced slot
        # and force private (YouTube scheduled-publish requires private).
        _eff_publish_at = t.scheduled_at
        _eff_privacy = (t.privacy_status or "private")
        if _stagger_min > 0 and _idx > 0 and t.scheduled_at is None:
            _eff_publish_at = _stagger_base + timedelta(minutes=_stagger_min * _idx)
            _eff_privacy = "private"

        job = models.UploadJobV2(
            publish_task_id=publish_task.id,
            channel_id=t.channel_id,
            oauth_token_id=oauth_by_channel[t.channel_id],
            brand_profile_id=t.brand_profile_id,
            upload_path=t.upload_path,
            postiz_integration_id=(t.postiz_integration_id if is_postiz else None),
            publish_kind=t.publish_kind,
            brand_mode=(getattr(t, "brand_mode", "per_channel") or "per_channel"),
            brand_placement=(getattr(t, "brand_placement", "template") or "template"),
            # Carry the publish intent so dispatch uploads with the user's
            # chosen privacy (and scheduled time) instead of forcing private.
            # (Anti-duplicate stagger may override these — see above.)
            privacy_status=_eff_privacy,
            publish_at=_eff_publish_at,
            thumbnail_source=t.thumbnail_source,
            thumbnail_r2_key=t.thumbnail_r2_key,
            status="queued",
            attempts=0,
            idempotency_key=idem_key,
            publish_version=publish_version,
            predicted_quota_units=quota_units,
            predicted_credit_cost=credit_cost,
            priority_at_dispatch=request.priority,
            # Durable queue (Wave 1): denormalized so the SKIP LOCKED
            # claim query + per-user active-cap aggregate need no joins.
            user_id=int(user.id),
            priority=request.priority,
        )
        db.add(job)
        try:
            db.flush()
        except IntegrityError as exc:
            # A concurrent request raced us between the SELECT above
            # and our INSERT. Treat the conflict as dedupe-by-design.
            db.rollback()
            log.warning(
                "fanout: idempotency_key collision on insert; treating as duplicate: %s",
                exc,
            )
            # Re-find the existing row's publish_task_id (after rollback
            # we need a fresh query).
            existing = (
                db.query(models.UploadJobV2)
                .filter(models.UploadJobV2.idempotency_key == idem_key)
                .first()
            )
            existing_ptid = int(existing.publish_task_id) if existing else 0
            raise DuplicatePublishVersionError(
                f"idempotency_key={idem_key} collided on insert",
                existing_publish_task_id=existing_ptid,
            )

        upload_job_ids.append(int(job.id))

        # Postiz jobs wrote no ledger row (0 cost), so there's nothing to
        # track for refund and nothing to backfill.
        if not is_postiz:
            reserved_ledger_ids.append((user.id, credit_cost, int(job.id)))

            # Backfill the ledger row's upload_job_id with the new job id.
            # We update the latest matching ledger row in-place.
            latest_ledger = (
                db.query(models.CreditLedger)
                .filter(
                    models.CreditLedger.user_id == user.id,
                    models.CreditLedger.reason == reason,
                    models.CreditLedger.upload_job_id.is_(None),
                )
                .order_by(models.CreditLedger.id.desc())
                .first()
            )
            if latest_ledger is not None:
                latest_ledger.upload_job_id = job.id
                db.add(latest_ledger)

    # target_count was provisionally set to the full request size; reset
    # it to ONLY the freshly-created jobs so completion math is correct
    # (skipped duplicates were never queued and must not be awaited).
    publish_task.target_count = len(upload_job_ids)

    # Step 12 — flip PublishTask to 'dispatched'.
    publish_task.status = "dispatched"
    db.add(publish_task)
    db.flush()

    # Step 13 — enqueue onto the Scheduler (or no-op fallback).
    # Durable queue (Wave 1): a committed status='queued' row IS the
    # enqueue — the SKIP LOCKED worker claims straight from the table,
    # so there is nothing to push and nothing that can be lost.
    if (os.environ.get("KAIZER_DURABLE_QUEUE", "0") or "0").strip() != "1":
        plan_tier_name = getattr(plan_tier, "name", "pro") if plan_tier else "pro"
        for ujob_id in upload_job_ids:
            try:
                _SCHEDULER_ENQUEUE(ujob_id, request.priority, user.id, plan_tier_name)
            except Exception as exc:
                # Never let scheduler enqueue failures abort the create —
                # rows are persisted, the legacy backfill or scheduler
                # startup recovery can pick them up. Log loudly though.
                log.exception(
                    "fanout: scheduler_enqueue failed for upload_job_id=%d: %s",
                    ujob_id, exc,
                )

    return PublishTaskResult(
        publish_task_id=int(publish_task.id),
        upload_job_ids=upload_job_ids,
        predicted_credit_total=int(predicted_credit_total),
        predicted_quota_units_total=int(predicted_quota_units_total),
        quota_pre_flight_ok=True,
        skipped_count=skipped_dupes,
    )


__all__ = [
    "FanoutTarget",
    "PublishTaskRequest",
    "PublishTaskResult",
    "create_publish_task",
    "FanoutError",
    "PlanTierMissingError",
    "PlanTierViolationError",
    "ChannelOwnershipError",
    "MasterVideoNotReadyError",
    "ThumbnailSourceMismatchError",
    "OAuthTokenMissingError",
    "DuplicatePublishVersionError",
]
