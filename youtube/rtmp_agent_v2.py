"""Phase 2.E — RTMP upload wrapper (v2).

Thin v2 wrapper that calls the existing ``youtube/rtmp_provider.py`` +
``youtube/rtmp_pusher.py`` modules. Those are already well-structured
and every external call is already wrapped in ``log_youtube_call`` —
we do NOT duplicate them.

What this module adds on top of the existing surface:

* A clean, dispatch-friendly signature: takes ``creds``, a local
  ``branded_path`` (downloaded from R2 by ``upload_dispatch``), and
  metadata. Returns ``{video_id, bytes_pushed, http_status}`` so the
  upload_dispatch.process() return shape matches the Direct path.
* Synthetic ``job``/``channel`` shims so the existing log wrappers
  attribute the call to the right (user, upload_job, channel) row
  without forcing the caller to construct a full SQLAlchemy ORM
  object. The legacy callers pass real ``models.UploadJob`` / Channel
  instances; the v2 caller passes a tiny ``_LogShim`` dataclass with
  the same attribute names. The forensic log is preserved either way.
* Quota cost (per Decision 4 / brief §1):
    - liveBroadcasts.insert  = 50 units
    - liveStreams.insert     = 50 units
    - liveBroadcasts.bind    = 50 units
    Total: 150 units (RTMP path).
  The thumbnail step (50 units) is NOT included here — that's owned
  by ``upload_dispatch.process()`` and gated by an independent
  ``quota.reserve(COST_THUMBNAIL_SET)`` so an upload doesn't succeed
  only to fail on the thumbnail.

The 1,600-unit difference between Direct and RTMP is exactly why
RTMP exists (brief §1).
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

from google.oauth2.credentials import Credentials

# Resilient imports: in unit-test/mocked environments the rtmp_*
# modules can still be import-time available (they don't require
# network at import). We import them lazily inside the function so
# the module loads cleanly even when one of them is mock-replaced.

log = logging.getLogger("kaizer.rtmp_agent_v2")


# ─── Errors (mirror uploader_v2 taxonomy for caller uniformity) ─────


class RtmpUploadError(Exception):
    """Terminal failure during the RTMP path. Caller refunds credits."""


class TransientRtmpUploadError(RtmpUploadError):
    """Retryable RTMP failure (5xx, push timeout, etc.). Caller re-queues."""


# ─── Burn-log helper (Phase 2.F predicted-vs-actual ledger) ─────────
#
# Per brief §2 + §7 and CONTRACTS §4.5: every YouTube API call site MUST
# write one ``quota_burn_log`` row. The RTMP path makes 3 YouTube API
# calls inside ``rtmp_provider.obtain_rtmp_target()`` (liveBroadcasts.insert,
# liveStreams.insert, liveBroadcasts.bind — 50 quota units each). We
# CHOSE to write 3 SEPARATE burn_log rows (one per logical operation)
# rather than one aggregated 150-unit row, so the predicted-vs-actual
# delta is attributable per-operation for the admin dashboard.
# All 3 succeed-or-fail together (obtain_rtmp_target raises atomically
# on any sub-call failure), so the outcome flag is shared.


_RTMP_TRIPLE = (
    ("liveBroadcasts.insert", 50),
    ("liveStreams.insert", 50),
    ("liveBroadcasts.bind", 50),
)
_QUOTA_REASONS = {"quotaExceeded", "dailyLimitExceeded"}
_TRANSIENT_STATUS = {500, 502, 503, 504}
_TRANSIENT_REASONS = {
    "rateLimitExceeded", "userRateLimitExceeded", "internalError",
    "backendError",
}


def _extract_reason(exc: BaseException) -> str:
    """Pull a YouTube error 'reason' out of an HttpError-ish exception.
    Returns '' on miss — the helper is best-effort."""
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
            return str(errs[0].get("reason") or "")
    except Exception:
        pass
    return ""


def _classify_outcome(exc: BaseException, status: int) -> str:
    reason = _extract_reason(exc)
    msg = str(exc).lower()
    if reason in _QUOTA_REASONS or (status == 403 and "quota" in msg):
        return "quota_exceeded"
    if status in _TRANSIENT_STATUS or reason in _TRANSIENT_REASONS:
        return "transient_error"
    return "permanent_error"


def _burn_rtmp_triple(
    db,
    *,
    upload_job_id_v2,
    exc: Optional[BaseException] = None,
) -> None:
    """Write the 3 burn_log rows for the RTMP triple-call (mint
    broadcast + stream + bind). All 3 share the same outcome because
    rtmp_provider.obtain_rtmp_target() raises atomically on any sub-call
    failure.

    Per task spec: we picked "3 separate rows" over "1 aggregated row"
    to keep predicted-vs-actual analytics per-operation.
    """
    if db is None:
        return
    try:
        from services import burn_log as _bl
    except Exception:
        return

    if exc is None:
        for op, cost in _RTMP_TRIPLE:
            try:
                _bl.log_predicted_and_actual(
                    db,
                    upload_job_id=upload_job_id_v2,
                    operation=op,
                    predicted_cost=cost,
                    http_status=200,
                    observed_outcome="success",
                )
            except Exception:
                pass
        return

    status = int(getattr(getattr(exc, "resp", None), "status", 0) or 0)
    outcome = _classify_outcome(exc, status)
    for op, cost in _RTMP_TRIPLE:
        try:
            _bl.log_predicted_and_actual(
                db,
                upload_job_id=upload_job_id_v2,
                operation=op,
                predicted_cost=cost,
                http_status=status,
                observed_outcome=outcome,
            )
        except Exception:
            pass


# ─── Log-attribution shim ───────────────────────────────────────────


@dataclass
class _LogShim:
    """Minimal struct that quacks like ``models.UploadJob`` for the
    existing rtmp_provider log-call wrappers.

    The legacy ``rtmp_provider.obtain_rtmp_target(...)`` reads
    ``job.user_id`` / ``job.id`` / ``job.clip_id`` / ``job.channel_id``
    / ``job.publish_kind`` and ``channel.oauth_token.google_channel_id``
    for attribution. We satisfy those reads with simple attribute
    access without needing the full ORM row.
    """
    user_id: Optional[int]
    id: Optional[int]
    clip_id: Optional[int]
    channel_id: Optional[int]
    publish_kind: str = "video"


@dataclass
class _ChannelShim:
    """Shim that satisfies ``rtmp_provider._gcid_from_channel(channel)``
    by exposing a ``.oauth_token.google_channel_id`` chain."""
    oauth_token: Any  # has .google_channel_id


@dataclass
class _OAuthTokenShim:
    google_channel_id: str = ""


# ─── Main entry point ───────────────────────────────────────────────


def upload_via_rtmp(
    creds: Credentials,
    branded_path: str,
    title: str,
    description: str,
    *,
    privacy_status: str = "private",
    publish_kind: str = "video",
    expected_duration_s: Optional[float] = None,
    db: Optional[Any] = None,
    upload_job_id: Optional[int] = None,
    upload_job_id_v2: Optional[int] = None,
    user_id: Optional[int] = None,
    channel_id: Optional[int] = None,
    google_channel_id: str = "",
    clip_id: Optional[int] = None,
    on_bytes_uploaded: Optional[Callable[[int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Run the full RTMP lifecycle for a single branded artifact.

    Steps (each one of 1-3 wrapped in ``log_youtube_call`` by the
    existing rtmp_provider module — forensic log preserved):

      1. ``rtmp_provider.obtain_rtmp_target(...)``
         → 50 + 50 + 50 = 150 quota units
         → returns broadcast_id, stream_id, ingest_url, stream_key, video_id

      2. ``rtmp_pusher.push_to_rtmp(...)``
         → real-time ffmpeg RTMPS push (zero quota units)

      3. ``rtmp_provider.finalize_broadcast(...)``
         → liveBroadcasts.list (1u) polls + liveBroadcasts.transition
           (≤50u depending on path; usually 0 because auto-stop fires)

    The thumbnail step (50u) is NOT invoked here. ``upload_dispatch``
    owns it, gated by an independent ``quota.reserve(COST_THUMBNAIL_SET)``
    and skipped entirely when ``publish_kind == 'short'``.

    Returns: ``{"video_id": str, "bytes_pushed": int, "http_status": int}``.

    Raises:
        RtmpUploadError              - permanent failure
        TransientRtmpUploadError     - retryable (5xx, push timeout)
    """
    # Lazy import so test environments can monkeypatch.
    from youtube import rtmp_provider as _rp
    from youtube import rtmp_pusher as _rp_pusher

    if not os.path.isfile(branded_path):
        raise RtmpUploadError(f"Branded artifact missing: {branded_path}")

    file_size = os.path.getsize(branded_path)
    if file_size <= 0:
        raise RtmpUploadError(f"Branded artifact empty: {branded_path}")

    job_shim = _LogShim(
        user_id=user_id,
        id=upload_job_id,
        clip_id=clip_id,
        channel_id=channel_id,
        publish_kind=(publish_kind or "video")[:10],
    )
    channel_shim = _ChannelShim(
        oauth_token=_OAuthTokenShim(google_channel_id=google_channel_id or ""),
    )

    # ── 1) Mint broadcast + stream + bind (150 quota units) ─────────
    try:
        target = _rp.obtain_rtmp_target(
            creds,
            job=job_shim,
            channel=channel_shim,
            title=title,
            description=description,
            privacy_status=privacy_status,
            db=db,
        )
    except Exception as exc:
        # All 3 sub-calls share the same failure outcome (the function
        # raises atomically). Log all 3 burn_log rows with the same
        # observed_outcome so the predicted-vs-actual ledger reflects
        # the 150 units that would have been charged.
        _burn_rtmp_triple(db, upload_job_id_v2=upload_job_id_v2, exc=exc)
        # rtmp_provider raises RtmpProviderError / TransientRtmpError;
        # map both to our taxonomy.
        if exc.__class__.__name__ == "TransientRtmpError":
            raise TransientRtmpUploadError(
                f"rtmp_provider.obtain_rtmp_target transient: {exc}"
            ) from exc
        raise RtmpUploadError(
            f"rtmp_provider.obtain_rtmp_target failed: {exc}"
        ) from exc

    # Success: 3 burn_log rows (one per logical API call), shared
    # outcome='success' (all 3 succeeded inside obtain_rtmp_target).
    _burn_rtmp_triple(db, upload_job_id_v2=upload_job_id_v2)

    broadcast_id = target.get("broadcast_id") or ""
    stream_id = target.get("stream_id") or ""
    ingest_url = target.get("ingest_url") or ""
    stream_key = target.get("stream_key") or ""
    video_id = target.get("video_id") or broadcast_id
    if not (broadcast_id and ingest_url and stream_key):
        raise RtmpUploadError(
            f"rtmp_provider returned incomplete target: {target!r}"
        )

    log.info(
        "rtmp_agent_v2: minted broadcast=%s stream=%s ingest=%s",
        broadcast_id, stream_id, ingest_url,
    )

    # ── 2) Real-time RTMPS push (zero quota units) ──────────────────
    seconds_pushed = 0.0
    bytes_pushed = 0

    def _progress(pushed_s: float, total_s: float) -> None:
        nonlocal seconds_pushed, bytes_pushed
        seconds_pushed = max(seconds_pushed, float(pushed_s or 0.0))
        if total_s > 0:
            # Translate "seconds pushed → bytes" proportionally so the
            # caller's progress UI keeps working without a new column.
            bytes_pushed = int(
                file_size * min(1.0, pushed_s / total_s)
            )
        if on_bytes_uploaded is not None:
            try:
                on_bytes_uploaded(bytes_pushed)
            except Exception:
                pass

    cancel = cancel_event or threading.Event()
    dur = expected_duration_s if expected_duration_s and expected_duration_s > 0 else 60.0

    try:
        result = _rp_pusher.push_to_rtmp(
            input_path=branded_path,
            ingest_url=ingest_url,
            stream_key=stream_key,
            expected_duration_s=dur,
            progress_cb=_progress,
            cancel_event=cancel,
            log_prefix=f"[rtmp_v2 job={upload_job_id}]",
        )
    except Exception as exc:
        # Clean up the orphan broadcast (best-effort) so we don't leave
        # junk on the channel.
        _safe_finalize(_rp, creds, job_shim, channel_shim, broadcast_id, db)
        if exc.__class__.__name__ in ("PushFailed", "TransientRtmpError"):
            raise TransientRtmpUploadError(
                f"rtmp_pusher.push_to_rtmp failed: {exc}"
            ) from exc
        raise RtmpUploadError(
            f"rtmp_pusher.push_to_rtmp failed: {exc}"
        ) from exc

    bytes_pushed = file_size
    seconds_pushed = max(
        seconds_pushed,
        float((result or {}).get("seconds_pushed", 0.0) or 0.0),
    )
    if on_bytes_uploaded is not None:
        try:
            on_bytes_uploaded(bytes_pushed)
        except Exception:
            pass

    # ── 3) Finalize: poll + transition to complete ──────────────────
    # finalize_broadcast also accepts a thumbnail path; we pass None
    # because the dispatch layer owns thumbnails (separate quota reserve).
    try:
        _rp.finalize_broadcast(
            creds,
            job=job_shim,
            channel=channel_shim,
            broadcast_id=broadcast_id,
            thumbnail_path=None,
            db=db,
        )
    except Exception as exc:
        # The push already succeeded — the video IS on YouTube.
        # Finalize failures are downgraded to a warning, NOT raised.
        log.warning(
            "rtmp_agent_v2: finalize_broadcast non-fatal failure (video pushed OK): %s",
            exc,
        )

    return {
        "video_id": str(video_id),
        "bytes_pushed": int(bytes_pushed),
        "http_status": 200,
    }


def _safe_finalize(rp_mod, creds, job_shim, channel_shim, broadcast_id, db) -> None:
    """Best-effort orphan cleanup. Never raises."""
    try:
        rp_mod.finalize_broadcast(
            creds,
            job=job_shim,
            channel=channel_shim,
            broadcast_id=broadcast_id,
            thumbnail_path=None,
            db=db,
        )
    except Exception as exc:
        log.warning(
            "rtmp_agent_v2: best-effort finalize after push failure failed (broadcast %s): %s",
            broadcast_id, exc,
        )


__all__ = [
    "RtmpUploadError",
    "TransientRtmpUploadError",
    "upload_via_rtmp",
]
