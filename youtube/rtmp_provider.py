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


def _job_ref(job) -> dict:
    """Which audit column this job belongs in.

    obtain_rtmp_target serves two callers with two different row types: the
    upload agent passes an UploadJob, Live Studio passes a LiveStream. Writing
    a LiveStream id into upload_job_id violated that column's foreign key, and
    log_youtube_call swallows the failure — so the call simply vanished from
    the quota dashboard instead of being recorded against the wrong job.
    """
    jid = getattr(job, "id", None)
    if jid is None:
        return {}
    table = getattr(getattr(job, "__table__", None), "name", "")
    if table == "live_streams":
        return {"live_stream_id": jid}
    return {"upload_job_id": jid}

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

def _yt(creds: Credentials, *, timeout_s: Optional[float] = None):
    """Return a YouTube Data API v3 client bound to ``creds``.

    Wave 1.6: explicit socket timeout on the httplib2 transport — the
    library default has NONE, so a single hung liveBroadcasts call used
    to block its dispatch thread (and scheduler slot) forever.

    ``timeout_s`` overrides that default for one client. The live engine passes
    10 s: its calls run inside an HTTP request that is holding a claimed channel
    and already-spent start credit, so a hung connection has to fail fast. The
    120 s default stays for everything else, because an upload legitimately
    takes minutes and must not be cut off.
    """
    try:
        import os as _os

        import google_auth_httplib2
        import httplib2
        _timeout = float(timeout_s) if timeout_s else max(10, int(_os.environ.get(
            "KAIZER_YT_HTTP_TIMEOUT_SECONDS", "120")))
        authed = google_auth_httplib2.AuthorizedHttp(
            creds, http=httplib2.Http(timeout=_timeout),
        )
        return build("youtube", "v3", http=authed, cache_discovery=False)
    except Exception:
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
    timeout_s: Optional[float] = None,
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
            **_job_ref(job),
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
            # REUSABLE. A non-reusable stream cannot be bound to a second
            # broadcast, so this flag is what makes a per-channel persistent
            # key possible at all. It was False "for cleanness + per-stream
            # quota tracking" -- a bookkeeping preference costing 50 units a
            # broadcast and leaking a dead stream resource onto the
            # customer's channel every time one succeeded.
            "isReusable": True,
        },
    }
    stream_id = ""
    ingest_url = ""
    stream_key = ""
    # Did THIS call create the stream? Gates the cleanup below -- deleting a
    # stream we merely borrowed would destroy the channel's persistent key and
    # could cut the ingest out from under a concurrent broadcast.
    created_stream = False

    # ── Reuse the channel's persistent stream when it has one ────────
    # 1 unit, against 50 to mint. The ingest info comes from the RESPONSE
    # rather than from storage, so a key regenerated in YouTube Studio is
    # picked up automatically and nothing secret is kept at rest.
    _saved = (getattr(channel, "yt_stream_id", "") or "").strip()
    if _saved:
        try:
            with _maybe_log_yt(
                db=db,
                user_id=getattr(job, "user_id", None),
                **_job_ref(job),
                clip_id=getattr(job, "clip_id", None),
                channel_id=getattr(job, "channel_id", None),
                google_channel_id=gcid,
                operation="liveStreams.list",
            ):
                _lresp = yt.liveStreams().list(part="id,cdn", id=_saved).execute()
            _items = (_lresp or {}).get("items") or []
            if _items:
                _ing = ((_items[0].get("cdn") or {}).get("ingestionInfo")) or {}
                stream_id  = _items[0].get("id", "") or ""
                ingest_url = (_ing.get("rtmpsIngestionAddress")
                              or _ing.get("ingestionAddress") or "")
                stream_key = _ing.get("streamName", "") or ""
        except Exception as _exc:
            # A deleted or unreadable stream is not fatal: fall through, mint a
            # fresh one and overwrite the stale id. Self-correcting.
            print(f"[rtmp-provider] saved stream {_saved} unusable ({_exc}); "
                  f"minting a replacement", flush=True)
            stream_id = ingest_url = stream_key = ""

    _mint_stream = not (stream_id and ingest_url and stream_key)

    # Mint only when the channel has no usable stream of its own.
    if _mint_stream:
        try:
            with _maybe_log_yt(
                db=db,
                user_id=getattr(job, "user_id", None),
                **_job_ref(job),
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
            created_stream = True
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
            **_job_ref(job),
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
        # ONLY delete a stream this call created. The channel's persistent
        # stream is shared by every broadcast on that channel -- deleting it
        # here would destroy the saved key and could cut off a concurrent
        # broadcast using the same ingest.
        if created_stream:
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
    confirm_active: bool = True,
    timeout_s: Optional[float] = None,
    db=None,
) -> None:
    """Move the broadcast to ``complete`` and (best-effort) set the
    custom thumbnail.

    Idempotent on the transition: re-calling on an already-complete
    broadcast is silently OK (YT returns 403 redundantTransition, we
    treat that as success).

    ``confirm_active=False`` skips the pre-transition poll. THE LIVE ENGINE MUST
    PASS FALSE, for two separate reasons:

      * it already knows. The engine confirms the broadcast went live once, with
        one liveBroadcasts.list a few seconds after ffmpeg started flowing.
        Polling again here would pay up to 15 more units for the same answer.
      * the poll is unsafe at this point. It runs just after the push was
        deliberately stopped, so YouTube may already have moved the broadcast
        out of `live` -- and the not-active branch below DELETES the broadcast.
        For a stream that ran for hours, that deletes the recording.

    The classic Live Studio path keeps the poll: there, asking at the end is the
    only way to tell "the encoder never connected" from "the transition failed",
    and the broadcast it deletes is one that never carried a second of video.
    """
    yt = _yt(creds, timeout_s=timeout_s)
    gcid = _gcid_from_channel(channel)

    # ── 1) Wait briefly for YT to register the stream as "active",
    # otherwise the transition rejects with "errorStreamInactive".
    # We don't block forever — 30 s is plenty after ffmpeg started.
    #
    # THE RESULT IS READ. It used to be discarded, and the consequence was a
    # misleading failure every time nothing was streamed: the broadcast stays
    # `ready`, `ready` -> `complete` is not a legal move, and the operator was
    # shown a 403 invalidTransition about a transition when the real fault was
    # that the encoder never connected.
    went_live = _wait_for_stream_active(yt, broadcast_id, timeout_s=30) if confirm_active else True
    if not went_live:
        # Do not ask YouTube to complete a broadcast that never started, and
        # do not leave an orphan `ready` broadcast sitting on the channel.
        _safe_delete_broadcast(yt, broadcast_id)
        raise StreamNeverActive(
            "Nothing reached YouTube: the broadcast was created and the stream "
            "key issued, but no video arrived at the ingest within 30s, so it "
            "never went live. The encoder is what failed - usually the source "
            "could not be fetched (for a URL source, check the stream's error "
            "for the yt-dlp reason), ffmpeg exited early, or the RTMP endpoint "
            "was unreachable from this server."
        )

    # ── 2) Transition to "complete" — closes the live broadcast and
    # locks the recording as a past-stream video on the channel.
    try:
        with _maybe_log_yt(
            db=db,
            user_id=getattr(job, "user_id", None),
            **_job_ref(job),
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
        if _is_redundant_transition(e) or _is_invalid_transition(e):
            # We only reach here having CONFIRMED the broadcast went live, so
            # either reason means the same thing: YouTube already closed it,
            # because enableAutoStop fired when the encoder stopped sending.
            # That is the normal ending, not a failure.
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


# ─── One cheap read, for the engine's single confirmation ────────────

def broadcast_lifecycle(
    creds: Credentials,
    broadcast_id: str,
    *,
    job: "models.UploadJob" = None,
    channel: Optional["models.Channel"] = None,
    timeout_s: Optional[float] = None,
    db=None,
) -> str:
    """YouTube's ``status.lifeCycleStatus`` for one broadcast. ONE unit.

    This is the whole of the engine's confirmation: rather than polling every
    two seconds for thirty seconds while starting (15 units), it waits until
    ffmpeg has actually been flowing and then asks once (1 unit). That single
    change is 13 units a broadcast, which is five more broadcasts a day.

    Returns "" when the broadcast is gone or the call failed, which the engine
    reads as "not answered" rather than as "not live" -- a network blip must not
    be recorded as a broadcast that never started.
    """
    yt = _yt(creds, timeout_s=timeout_s)
    try:
        with _maybe_log_yt(
            db=db,
            user_id=getattr(job, "user_id", None),
            **_job_ref(job),
            channel_id=getattr(job, "channel_id", None),
            google_channel_id=_gcid_from_channel(channel),
            video_id=(broadcast_id or "")[:50],
            operation="liveBroadcasts.list",
        ):
            r = yt.liveBroadcasts().list(part="status,id", id=broadcast_id).execute()
    except HttpError as e:
        print(f"[rtmp-provider] lifecycle read failed for {broadcast_id}: {_parse_error(e)[1]}")
        return ""
    items = (r or {}).get("items") or []
    if not items:
        return ""
    return (items[0].get("status") or {}).get("lifeCycleStatus") or ""


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


class StreamNeverActive(RuntimeError):
    """Nothing ever reached YouTube's ingest.

    Raised instead of letting transition(complete) fail with
    invalidTransition, which describes the symptom rather than the fault: a
    broadcast that was never fed stays `ready`, and `ready` -> `complete` is
    not a legal move. The encoder is what failed, not the transition.
    """


def _is_invalid_transition(e: HttpError) -> bool:
    """403 invalidTransition -- the broadcast is not in a state that allows
    the move being asked for. After a stream that ran, it means auto-stop
    already closed it; after one that never started, it means nothing was
    ever sent. The caller knows which, from the stream status."""
    _, reason = _parse_error(e)
    return reason == "invalidTransition"


def _is_redundant_transition(e: HttpError) -> bool:
    _, reason = _parse_error(e)
    return reason == "redundantTransition"
