"""YouTube Live Streaming API helpers — mint, bind, transition, finalize.

This module is the "credential agent" the operator asked for. Given a
YouTube OAuth credential, it produces a fresh RTMPS push target
(``rtmps://...`` + stream key) by calling three YouTube Data API
endpoints in sequence:

  1. ``liveBroadcasts.insert``  — creates the broadcast (the public-
     facing "video" that will appear on the channel after streaming)
  2. ``liveStreams.insert``      — creates the ingestion endpoint
     (the RTMP URL + key)
  3. ``liveBroadcasts.bind``     — couples the two

Total quota: ~150 units to mint, ~50 to start, ~50 to finalize,
~50 for thumbnail = **~250 units per video**, vs 1,600 for
``videos.insert``. That's the entire reason this path exists.

EVERY external call here is:
  * wrapped in ``log_youtube_call`` so it appears in the admin Usage
    dashboard with correct attribution (channel, job, user, quota)
  * subject to explicit return-value validation — no "assume success"
  * isolated so a partial failure leaves cleanable state, not orphans

Failure model: any function that needs to mutate broadcast state and
fails raises ``RtmpProviderError`` with a human-readable message. The
caller (rtmp_agent) marks the upload-job failed and surfaces the error
to the admin Logs tab.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials

import models


# ─── Errors ──────────────────────────────────────────────────────────

class RtmpProviderError(Exception):
    """Terminal failure during broadcast credential management."""


class TransientRtmpError(RtmpProviderError):
    """Retryable — network blip, 5xx, rate-limit."""


# ─── Quota wrapper (resilient import) ────────────────────────────────

try:
    from learning.youtube_quota_log import log_youtube_call as _log_yt
except Exception:
    _log_yt = None


@contextmanager
def _maybe_log_yt(**kwargs):
    """Wrap an API call with quota logging when available, no-op otherwise.
    Always yields the call handle (or None) so the call sites have a
    uniform shape."""
    if _log_yt is None:
        yield None
    else:
        with _log_yt(**kwargs) as call:
            yield call


# ─── YouTube client builder ──────────────────────────────────────────

def _yt(creds: Credentials):
    """Return a YouTube Data API v3 client bound to ``creds``."""
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def _gcid_from_channel(channel: Optional["models.Channel"]) -> str:
    """Extract the destination ``google_channel_id`` for log attribution.
    Returns "" if anything is missing — log row just lands with an
    empty channel column, which the dashboard already handles."""
    if channel is None:
        return ""
    tok = getattr(channel, "oauth_token", None)
    if tok is None:
        return ""
    return (getattr(tok, "google_channel_id", "") or "")[:50]


# ─── Mint: insert + insert + bind ────────────────────────────────────

def obtain_rtmp_target(
    creds: Credentials,
    *,
    job: "models.UploadJob",
    channel: Optional["models.Channel"],
    title: str,
    description: str = "",
    privacy_status: str = "private",
    scheduled_start: Optional[datetime] = None,
    enable_auto_start: bool = True,
    enable_auto_stop: bool = True,
    db=None,
) -> dict:
    """Mint a fresh RTMPS push target on the channel.

    Returns a dict with the ingest credentials AND the broadcast/stream
    resource IDs (needed later to bind, transition, finalize):

        {
            "broadcast_id":   "abc123...",
            "stream_id":      "def456...",
            "ingest_url":     "rtmps://a.rtmps.youtube.com/live2",
            "stream_key":     "xxxx-xxxx-xxxx-xxxx",
            "video_id":       "abc123...",   # same as broadcast_id for
                                              # past-stream classification
        }

    On failure: raises ``RtmpProviderError`` (terminal) or
    ``TransientRtmpError`` (caller should retry once).

    Cost: 3 × 50 = 150 quota units (insert broadcast + insert stream +
    bind), all logged to the admin Usage dashboard.
    """
    yt = _yt(creds)
    gcid = _gcid_from_channel(channel)

    # ── 1) liveBroadcasts.insert ────────────────────────────────────
    body_broadcast = {
        "snippet": {
            "title":       (title or "Untitled bulletin")[:100],
            "description": (description or "")[:5000],
            "scheduledStartTime": (
                (scheduled_start or datetime.now(timezone.utc) + timedelta(seconds=5))
                .astimezone(timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ")
            ),
        },
        "status": {
            "privacyStatus":          (privacy_status or "private").lower(),
            "selfDeclaredMadeForKids": False,
        },
        "contentDetails": {
            # Auto-transitioning is critical: without it, the broadcast
            # sits in "testing" forever and the agent has to explicitly
            # transition through every state.
            "enableAutoStart": bool(enable_auto_start),
            "enableAutoStop":  bool(enable_auto_stop),
            "enableContentEncryption": False,
            "enableDvr":               True,
            "monitorStream": {
                "enableMonitorStream": False,
            },
        },
    }
    broadcast_id = ""
    try:
        with _maybe_log_yt(
            db=db,
            user_id=getattr(job, "user_id", None),
            upload_job_id=getattr(job, "id", None),
            clip_id=getattr(job, "clip_id", None),
            channel_id=getattr(job, "channel_id", None),
            google_channel_id=gcid,
            operation="liveBroadcasts.insert",
            publish_kind=(getattr(job, "publish_kind", "") or "")[:10],
        ):
            resp = yt.liveBroadcasts().insert(
                part="snippet,status,contentDetails",
                body=body_broadcast,
            ).execute()
        broadcast_id = (resp or {}).get("id", "")
        if not broadcast_id:
            raise RtmpProviderError(f"liveBroadcasts.insert returned no id: {resp}")
    except HttpError as e:
        raise _wrap_http(e, "liveBroadcasts.insert") from e

    # ── 2) liveStreams.insert ───────────────────────────────────────
    body_stream = {
        "snippet": {
            "title": (title or "Kaizer ingest")[:100],
        },
        "cdn": {
            # Variable-bitrate keeps quality up while letting the
            # encoder choose what works. resolution+frameRate=variable
            # is the safest default for any video we throw at it.
            "frameRate":   "variable",
            "ingestionType": "rtmp",
            "resolution":  "variable",
        },
        "contentDetails": {
            # Sticky stream resources are reusable — we still create
            # one per upload for cleanness + per-stream quota tracking.
            "isReusable": False,
        },
    }
    stream_id = ""
    ingest_url = ""
    stream_key = ""
    try:
        with _maybe_log_yt(
            db=db,
            user_id=getattr(job, "user_id", None),
            upload_job_id=getattr(job, "id", None),
            clip_id=getattr(job, "clip_id", None),
            channel_id=getattr(job, "channel_id", None),
            google_channel_id=gcid,
            operation="liveStreams.insert",
        ):
            sresp = yt.liveStreams().insert(
                part="snippet,cdn,contentDetails",
                body=body_stream,
            ).execute()
        stream_id  = (sresp or {}).get("id", "")
        ingestion  = ((sresp or {}).get("cdn") or {}).get("ingestionInfo") or {}
        ingest_url = ingestion.get("ingestionAddress", "") or ""
        stream_key = ingestion.get("streamName", "") or ""
        # Prefer RTMPS — YouTube returns both `ingestionAddress` (rtmp)
        # and `rtmpsIngestionAddress` (rtmps). Switch to the encrypted
        # one when present (it's free on every YT account).
        rtmps = ingestion.get("rtmpsIngestionAddress", "")
        if rtmps:
            ingest_url = rtmps
        if not stream_id or not ingest_url or not stream_key:
            raise RtmpProviderError(
                f"liveStreams.insert missing ingest fields: id={stream_id!r}, "
                f"url={ingest_url!r}, key_set={bool(stream_key)}"
            )
    except HttpError as e:
        # Best-effort cleanup of the orphan broadcast we just created.
        _safe_delete_broadcast(yt, broadcast_id)
        raise _wrap_http(e, "liveStreams.insert") from e

    # ── 3) liveBroadcasts.bind ──────────────────────────────────────
    try:
        with _maybe_log_yt(
            db=db,
            user_id=getattr(job, "user_id", None),
            upload_job_id=getattr(job, "id", None),
            clip_id=getattr(job, "clip_id", None),
            channel_id=getattr(job, "channel_id", None),
            google_channel_id=gcid,
            operation="liveBroadcasts.bind",
        ):
            yt.liveBroadcasts().bind(
                id=broadcast_id,
                part="id,contentDetails",
                streamId=stream_id,
            ).execute()
    except HttpError as e:
        _safe_delete_stream(yt, stream_id)
        _safe_delete_broadcast(yt, broadcast_id)
        raise _wrap_http(e, "liveBroadcasts.bind") from e

    return {
        "broadcast_id": broadcast_id,
        "stream_id":    stream_id,
        "ingest_url":   ingest_url,
        "stream_key":   stream_key,
        "video_id":     broadcast_id,   # past-stream URL = /watch?v=broadcast_id
    }


# ─── Finalize: transition complete + set thumbnail ───────────────────

def finalize_broadcast(
    creds: Credentials,
    *,
    job: "models.UploadJob",
    channel: Optional["models.Channel"],
    broadcast_id: str,
    thumbnail_path: Optional[str] = None,
    db=None,
) -> None:
    """Move the broadcast to ``complete`` and (best-effort) set the
    custom thumbnail.

    Idempotent on the transition: re-calling on an already-complete
    broadcast is silently OK (YT returns 403 redundantTransition, we
    treat that as success).
    """
    yt = _yt(creds)
    gcid = _gcid_from_channel(channel)

    # ── 1) Wait briefly for YT to register the stream as "active",
    # otherwise the transition rejects with "errorStreamInactive".
    # We don't block forever — 30 s is plenty after ffmpeg started.
    _wait_for_stream_active(yt, broadcast_id, timeout_s=30)

    # ── 2) Transition to "complete" — closes the live broadcast and
    # locks the recording as a past-stream video on the channel.
    try:
        with _maybe_log_yt(
            db=db,
            user_id=getattr(job, "user_id", None),
            upload_job_id=getattr(job, "id", None),
            clip_id=getattr(job, "clip_id", None),
            channel_id=getattr(job, "channel_id", None),
            google_channel_id=gcid,
            video_id=broadcast_id[:50],
            operation="liveBroadcasts.transition",
        ):
            yt.liveBroadcasts().transition(
                broadcastStatus="complete",
                id=broadcast_id,
                part="status",
            ).execute()
    except HttpError as e:
        # YT often returns 403 redundantTransition when auto-stop
        # already fired (because the encoder cleanly stopped sending
        # frames). That's success, not a failure.
        if _is_redundant_transition(e):
            print(f"[rtmp-provider] broadcast {broadcast_id} already complete (auto-stop fired)")
        else:
            raise _wrap_http(e, "liveBroadcasts.transition(complete)") from e

    # ── 3) Set the custom thumbnail (best-effort, never raises).
    if thumbnail_path and os.path.isfile(thumbnail_path):
        try:
            from youtube.uploader import set_thumbnail as _set_thumb
            _set_thumb(creds, broadcast_id, thumbnail_path, job=job)
        except Exception as exc:
            print(f"[rtmp-provider] thumbnail set failed (non-fatal): {exc}")


# ─── Polling helper: is the stream actively ingesting? ───────────────

def _wait_for_stream_active(yt, broadcast_id: str, *, timeout_s: int = 30) -> bool:
    """Block until ``broadcastStatus`` of the bound stream becomes
    `live` or `active`, or ``timeout_s`` elapses. Returns True if active,
    False on timeout — caller decides whether to proceed anyway.

    These status polls are cheap (1 unit each via liveBroadcasts.list)
    and the call is rate-limited internally to once per 2 s."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            r = yt.liveBroadcasts().list(
                part="status,id",
                id=broadcast_id,
            ).execute()
            items = (r or {}).get("items") or []
            if items:
                lifecycle = (items[0].get("status") or {}).get("lifeCycleStatus") or ""
                if lifecycle in ("live", "liveStarting"):
                    return True
                if lifecycle == "complete":
                    return True   # already done, caller will handle redundantTransition
        except HttpError:
            pass
        time.sleep(2.0)
    return False


# ─── Cleanup helpers (best-effort, never raise) ─────────────────────

def _safe_delete_broadcast(yt, broadcast_id: str) -> None:
    if not broadcast_id:
        return
    try:
        yt.liveBroadcasts().delete(id=broadcast_id).execute()
    except Exception as exc:
        print(f"[rtmp-provider] cleanup broadcast {broadcast_id} failed: {exc}")


def _safe_delete_stream(yt, stream_id: str) -> None:
    if not stream_id:
        return
    try:
        yt.liveStreams().delete(id=stream_id).execute()
    except Exception as exc:
        print(f"[rtmp-provider] cleanup stream {stream_id} failed: {exc}")


# ─── HTTP error classification ───────────────────────────────────────

_TRANSIENT_STATUS  = {500, 502, 503, 504}
_TRANSIENT_REASONS = {
    "rateLimitExceeded", "userRateLimitExceeded", "internalError",
    "backendError",
}


def _parse_error(e: HttpError) -> tuple[int, str]:
    status = getattr(e.resp, "status", 0) if hasattr(e, "resp") else 0
    reason = ""
    try:
        import json as _json
        content = e.content if isinstance(e.content, (bytes, bytearray)) else b""
        data = _json.loads(content.decode("utf-8")) if content else {}
        errs = (data.get("error") or {}).get("errors") or []
        if errs:
            reason = errs[0].get("reason") or ""
    except Exception:
        pass
    return status, reason


def _wrap_http(e: HttpError, op: str) -> RtmpProviderError:
    status, reason = _parse_error(e)
    msg = f"{op} failed: HTTP {status} {reason}"
    if status in _TRANSIENT_STATUS or reason in _TRANSIENT_REASONS:
        return TransientRtmpError(msg)
    return RtmpProviderError(msg)


def _is_redundant_transition(e: HttpError) -> bool:
    _, reason = _parse_error(e)
    return reason == "redundantTransition"
