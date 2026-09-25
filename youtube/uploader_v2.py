"""Phase 2.E — Direct upload (videos.insert) refactor.

Refactor of ``youtube/uploader.py:upload_video()`` for the v2 publish
pipeline. Key differences from the legacy module:

* Input is the **local path to the branded artifact** (downloaded from
  R2 by ``services.upload_dispatch`` before calling here), NOT the raw
  clip path.
* The resumable session ``upload_uri`` is **persisted to the DB** after
  Google returns it (via the ``on_uri_obtained`` callback). The legacy
  code never wrote it (DISCOVERY §2 gap closed here).
* The thumbnail step is **NOT** invoked here. The dispatch layer owns
  ``thumbnails.set`` because it must be gated by an independent
  ``quota.reserve(COST_THUMBNAIL_SET)`` AND it must be skipped entirely
  for Shorts.
* Audio is whatever the branded artifact already has (``-c:a copy``
  from the branding stage). No re-encoding here.
* Every YouTube API call is wrapped in ``log_youtube_call`` so the
  forensic ``youtube_api_calls`` table gets exactly one row per real
  ``videos.insert`` burn (1,600 quota units).

NOTE — the legacy ``youtube/uploader.py`` stays alive behind the
feature flag. This module is a sibling, not a replacement-in-place.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

log = logging.getLogger("kaizer.uploader_v2")


# Resumable-upload chunk size. With the httplib2 308-redirect bug fixed
# in _yt() (follow_redirects=False), MULTI-chunk uploads now work — so
# large longform videos upload reliably across many chunks. This is just
# a tuning knob: bigger = fewer requests but more RAM per upload (and
# coarser resume granularity); since fan-out runs several uploads
# concurrently, a moderate default keeps memory sane. Clips that fit in
# one chunk still upload in a single request. Must be a multiple of
# 256 KB (resumable requirement). Override via KAIZER_YT_CHUNK_MB.
import os as _os
try:
    _chunk_mb = int(_os.environ.get("KAIZER_YT_CHUNK_MB", "50") or "50")
    if _chunk_mb <= 0:
        _chunk_mb = 50
except ValueError:
    _chunk_mb = 50
CHUNK_SIZE = _chunk_mb * 1024 * 1024


# ─── Burn-log helper (Phase 2.F predicted-vs-actual ledger) ────────────
#
# Per brief §2 + §7 and CONTRACTS §4.5: every YouTube API call site MUST
# write one ``quota_burn_log`` row recording (predicted_cost, http_status,
# observed_outcome). This is ORTHOGONAL to the existing ``log_youtube_call``
# wrapper (which writes ``youtube_api_calls`` — preserved per brief §0).
# Implemented as a private helper here so the call-site bodies stay flat.


def _extract_reason(exc: BaseException) -> str:
    """Pull the YouTube error reason (e.g. 'quotaExceeded') out of an
    HttpError payload. Returns '' on miss."""
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


def _burn_for_call(
    db,
    *,
    upload_job_id_v2,
    operation: str,
    predicted_cost: int,
    exc: Optional[BaseException] = None,
    http_status: Optional[int] = None,
) -> None:
    """Write one ``quota_burn_log`` row classifying the call outcome.

    Safe to call even when ``db`` is None (smoke tests sometimes pass
    None) — we no-op. Never raises; burn_log itself swallows DB errors.
    """
    if db is None:
        return
    try:
        from services import burn_log as _bl
    except Exception:
        return

    if exc is None:
        try:
            _bl.log_predicted_and_actual(
                db,
                upload_job_id=upload_job_id_v2,
                operation=operation,
                predicted_cost=int(predicted_cost),
                http_status=int(http_status if http_status is not None else 200),
                observed_outcome="success",
            )
        except Exception:
            pass
        return

    status = int(http_status) if http_status is not None else int(
        getattr(getattr(exc, "resp", None), "status", 0) or 0
    )
    reason = _extract_reason(exc)
    msg = str(exc).lower()
    if reason in _QUOTA_REASONS or (status == 403 and "quota" in msg):
        outcome = "quota_exceeded"
    elif status in _TRANSIENT_STATUS or reason in _TRANSIENT_REASONS:
        outcome = "transient_error"
    else:
        outcome = "permanent_error"
    try:
        _bl.log_predicted_and_actual(
            db,
            upload_job_id=upload_job_id_v2,
            operation=operation,
            predicted_cost=int(predicted_cost),
            http_status=status,
            observed_outcome=outcome,
        )
    except Exception:
        pass


# ─── Errors (mirror legacy taxonomy) ──────────────────────────────────


class UploadError(Exception):
    """Terminal error — caller refunds credits and marks job failed."""


class TransientUploadError(UploadError):
    """Temporary — caller persists progress and re-queues with backoff."""


class QuotaExceededError(UploadError):
    """403 quotaExceeded / dailyLimitExceeded — caller refunds credits
    and parks the job WITHOUT bumping the attempts counter."""


# ─── YouTube client ─────────────────────────────────────────────────


def _http_timeout_s() -> int:
    """Per-HTTP-call socket timeout. Without this a single hung Google
    call blocks the dispatch thread (and its slot) FOREVER — the #3
    critical finding of the Wave-1 reliability audit."""
    try:
        return max(10, int(os.environ.get("KAIZER_YT_HTTP_TIMEOUT_SECONDS", "120")))
    except Exception:
        return 120


def _yt(creds: Credentials):
    """Build the client with an explicit socket timeout on the
    underlying httplib2 transport (default has NO timeout)."""
    try:
        import google_auth_httplib2  # ships with googleapiclient
        import httplib2
        base = httplib2.Http(timeout=_http_timeout_s())
        # CRITICAL for large / multi-chunk resumable uploads (longform
        # videos). The resumable protocol returns "308 Resume Incomplete"
        # between chunks — that 308 carries a Range header, NOT a Location.
        # httplib2 defaults to follow_redirects=True and treats the 308 as
        # an HTTP redirect, looks for the absent Location header, and raises
        # RedirectMissingLocation ("Redirected but the response is missing a
        # Location: header") — which killed EVERY video big enough to span
        # more than one chunk (Shorts fit in one chunk, so they worked).
        # Turning redirect-following off hands the 308 back to
        # googleapiclient, which reads the Range header and resumes
        # correctly. The YouTube Data API itself uses no real redirects, so
        # this is safe for the non-upload calls on the same client.
        base.follow_redirects = False
        authed = google_auth_httplib2.AuthorizedHttp(creds, http=base)
        return build("youtube", "v3", http=authed, cache_discovery=False)
    except Exception:
        # Fallback: library default transport (no timeout) — the
        # publish worker's watchdog still backstops a hang.
        return build("youtube", "v3", credentials=creds, cache_discovery=False)


# ─── Resumable session probe (Wave 1.6 — real resume support) ───────


def _probe_resumable_session(upload_uri: str, size: int) -> dict:
    """Ask Google for the state of a persisted resumable session.

    Returns one of:
      {"state": "completed", "video_id": str}  — the upload actually
          landed before the previous worker died; ZERO re-upload needed
      {"state": "active", "offset": int}       — resume from offset
      {"state": "dead"}                        — 404/410/expired (Google
          invalidates sessions after ~7 days) → start a fresh session

    A session-status PUT costs ZERO quota — it's not an API method.
    Never raises; any error maps to "dead" (fresh upload is always a
    safe fallback, and the idempotency probe already guarded against
    true duplicates before we got here).
    """
    try:
        import json as _json

        import httplib2
        h = httplib2.Http(timeout=60)
        resp, content = h.request(
            upload_uri,
            method="PUT",
            headers={"Content-Range": f"bytes */{int(size)}"},
            body=b"",
        )
        status = int(resp.status)
        if status in (200, 201):
            # Session already completed — parse the video resource.
            try:
                data = _json.loads(content.decode("utf-8"))
                vid = str(data.get("id") or "")
            except Exception:
                vid = ""
            if vid:
                return {"state": "completed", "video_id": vid}
            return {"state": "dead"}
        if status == 308:
            rng = str(resp.get("range") or resp.get("Range") or "")
            try:
                offset = int(rng.split("-")[1]) + 1 if "-" in rng else 0
            except Exception:
                offset = 0
            return {"state": "active", "offset": offset}
        return {"state": "dead"}
    except Exception:
        return {"state": "dead"}


# ─── Tag sanitiser (copied from uploader.py:41 for refactor isolation) ─


def sanitize_tags(raw: Optional[list]) -> list[str]:
    """Scrub tags so YouTube's ``videos.insert`` doesn't 400 with
    ``invalidTags``. Same rules as the legacy uploader:
      - no '<' or '>' anywhere
      - no leading '#'
      - each tag ≤ 100 chars
      - joined length (commas + quotes) ≤ 500 chars
      - no empty strings, no duplicates (case-insensitive)
    """
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    combined = 0
    for item in raw:
        if not item:
            continue
        t = str(item).strip()
        while t.startswith("#"):
            t = t[1:].strip()
        t = t.replace("<", "").replace(">", "").replace('"', "").strip()
        if not t:
            continue
        t = t[:100]
        key = t.lower()
        if key in seen:
            continue
        est = len(t) + 2 + (2 if "," in t or " " in t else 0)
        if combined + est > 500:
            break
        seen.add(key)
        out.append(t)
        combined += est
    return out


# ─── Body builder ───────────────────────────────────────────────────


def add_to_playlist(creds: Credentials, video_id: str, playlist_id: str) -> dict:
    """Best-effort: add an uploaded video to a playlist. NEVER raises —
    playlist membership is non-critical, so a failure must not fail the
    publish (the video is already live). Returns {"ok": bool, ...}."""
    if not (video_id and playlist_id):
        return {"ok": False, "skipped": True}
    try:
        yt = _yt(creds)
        yt.playlistItems().insert(
            part="snippet",
            body={"snippet": {
                "playlistId": str(playlist_id),
                "resourceId": {"kind": "youtube#video", "videoId": str(video_id)},
            }},
        ).execute()
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:300]}


def _build_body(
    *,
    title: str,
    description: str,
    tags: list[str],
    category_id: str,
    privacy_status: str,
    publish_at: Optional[Any],
    made_for_kids: bool,
    publish_kind: str,
    default_language: str = "te",
    license_kind: str = "youtube",
) -> dict:
    """Construct the ``videos.insert`` body.

    Mirrors ``uploader.py:_build_body`` but takes scalar args (no
    UploadJob coupling) so the v2 callers can build it from any source
    of truth. Honours ``publishAt`` only when ``privacyStatus='private'``
    (Google's requirement).

    For Shorts: ensures ``#Shorts`` is present somewhere in the
    description so YouTube's classifier sees it. (The composer/Fanout
    should have already added it via SEO composition; this is belt-
    and-braces because the API behaviour drifts otherwise.)
    """
    effective_desc = description or ""
    if publish_kind == "short":
        if "#shorts" not in effective_desc.lower():
            # Prepend on a new line so it's visible and high-signal.
            effective_desc = (
                f"#Shorts\n\n{effective_desc}".strip()
                if effective_desc
                else "#Shorts"
            )

    _lang = (default_language or "te")
    _lic = (license_kind or "youtube")
    if _lic not in ("youtube", "creativeCommon"):
        _lic = "youtube"
    body: dict = {
        "snippet": {
            "title": (title or "Untitled")[:100],
            "description": effective_desc,
            "tags": sanitize_tags(tags or []),
            "categoryId": category_id or "25",
            "defaultLanguage": _lang,
            "defaultAudioLanguage": _lang,
        },
        "status": {
            "privacyStatus": (privacy_status or "private").lower(),
            "selfDeclaredMadeForKids": bool(made_for_kids),
            "license": _lic,
            "embeddable": True,
            "publicStatsViewable": True,
        },
    }
    if publish_at and body["status"]["privacyStatus"] == "private":
        dt = publish_at
        if hasattr(dt, "tzinfo") and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if hasattr(dt, "astimezone"):
            body["status"]["publishAt"] = dt.astimezone(
                timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
    return body


# ─── Main entry point ───────────────────────────────────────────────


def upload_video(
    creds: Credentials,
    branded_path: str,
    title: str,
    description: str,
    tags: list[str],
    *,
    category_id: str = "25",
    privacy_status: str = "private",
    publish_at: Optional[Any] = None,
    made_for_kids: bool = False,
    publish_kind: str = "video",
    default_language: str = "te",
    license_kind: str = "youtube",
    progress_cb: Optional[Callable[[int, int], None]] = None,
    upload_uri: Optional[str] = None,
    on_uri_obtained: Optional[Callable[[str], None]] = None,
    on_bytes_uploaded: Optional[Callable[[int], None]] = None,
    db: Optional[Any] = None,
    upload_job_id: Optional[int] = None,
    upload_job_id_v2: Optional[int] = None,
    user_id: Optional[int] = None,
    channel_id: Optional[int] = None,
    google_channel_id: str = "",
    clip_id: Optional[int] = None,
    job_id: Optional[int] = None,
) -> dict:
    """Run (or resume) the resumable ``videos.insert`` upload.

    Returns ``{"video_id": str, "bytes_uploaded": int, "http_status": int}``.

    Raises:
        UploadError                - permanent failure
        TransientUploadError       - retryable
        QuotaExceededError         - 403 quotaExceeded; caller refunds
                                     credits and parks the job
    """
    path = Path(branded_path)
    if not path.exists():
        raise UploadError(f"Branded artifact missing: {branded_path}")

    size = path.stat().st_size
    if size <= 0:
        raise UploadError(f"Branded artifact empty: {branded_path}")

    # Resilient import — the forensic log is optional only in unit-test
    # environments (it's always present in prod).
    try:
        from learning.youtube_quota_log import log_youtube_call as _log_yt
    except Exception:
        _log_yt = None

    body = _build_body(
        title=title,
        description=description,
        tags=tags or [],
        category_id=category_id,
        privacy_status=privacy_status,
        publish_at=publish_at,
        made_for_kids=made_for_kids,
        publish_kind=publish_kind,
        default_language=default_language,
        license_kind=license_kind,
    )

    media: Optional[MediaFileUpload] = None
    http_status = 0
    try:
        yt = _yt(creds)

        # ── Resume probe (Wave 1.6) ─────────────────────────────────
        # A persisted upload_uri used to be accepted-and-ignored here —
        # every retry restarted a fresh videos.insert (another 1,600u).
        # Now: probe the session. completed → return the video id with
        # ZERO new quota; active → resume the SAME session from the
        # confirmed offset; dead (Google expires sessions ~7 days, or
        # 404/410) → clear the stale checkpoint and start fresh.
        _resume_offset: Optional[int] = None
        if upload_uri:
            probe = _probe_resumable_session(str(upload_uri), int(size))
            state = probe.get("state")
            if state == "completed" and probe.get("video_id"):
                vid = str(probe["video_id"])
                log.info(
                    "uploader_v2: resumable session already completed "
                    "(video_id=%s) — short-circuiting with zero quota", vid,
                )
                if on_bytes_uploaded is not None:
                    try:
                        on_bytes_uploaded(int(size))
                    except Exception:
                        pass
                return {
                    "video_id": vid,
                    "bytes_uploaded": int(size),
                    "http_status": 200,
                }
            elif state == "active":
                _resume_offset = int(probe.get("offset") or 0)
                log.info(
                    "uploader_v2: resuming session at byte %d/%d",
                    _resume_offset, size,
                )
            else:
                log.info(
                    "uploader_v2: persisted upload session is dead — "
                    "starting a fresh videos.insert",
                )
                if on_uri_obtained is not None:
                    try:
                        on_uri_obtained("")  # clears the stale checkpoint
                    except Exception:
                        pass
                upload_uri = None

        media = MediaFileUpload(
            str(path),
            chunksize=CHUNK_SIZE,
            resumable=True,
            mimetype="video/mp4",
        )

        # The entire resumable session is ONE quota event (1600 units).
        # Wrap so the forensic log gets exactly one row per insert.
        with (_log_yt(
                db=db,
                user_id=user_id,
                job_id=job_id,
                clip_id=clip_id,
                upload_job_id=upload_job_id,
                channel_id=channel_id,
                google_channel_id=(google_channel_id or "")[:50],
                operation="videos.insert",
                publish_kind=(publish_kind or "")[:10],
                file_bytes=int(size),
              ) if _log_yt is not None else _nullctx()) as _call:
            request = yt.videos().insert(
                part="snippet,status",
                body=body,
                media_body=media,
                notifySubscribers=False,
            )

            # Attach the live session so next_chunk() skips initiation
            # (no second 1,600u event) and continues from the offset
            # Google confirmed in the probe above.
            if upload_uri and _resume_offset is not None:
                try:
                    request.resumable_uri = str(upload_uri)
                    request.resumable_progress = int(_resume_offset)
                except Exception:
                    log.warning(
                        "uploader_v2: could not attach resume session; "
                        "falling back to fresh upload",
                    )

            # ── Persist resumable upload URI (DISCOVERY §2 gap) ──
            # googleapiclient sets ``request.resumable_uri`` after the
            # first chunk negotiation. We surface it back to the caller
            # so the DB row gets ``upload_uri`` checkpointed; the next
            # process restart can resume from that URI.
            response = None
            uploaded_bytes = 0
            last_emitted = -1
            first_chunk_done = False

            while response is None:
                try:
                    status, response = request.next_chunk()
                except HttpError as exc:
                    # Burn-log the actual outcome (quota_exceeded /
                    # transient / permanent) BEFORE re-raising so the
                    # predicted-vs-actual ledger captures every API
                    # failure, not just successes (brief §2 + §7).
                    _burn_for_call(
                        db,
                        upload_job_id_v2=upload_job_id_v2,
                        operation="videos.insert",
                        predicted_cost=100,
                        exc=exc,
                    )
                    raise _wrap_http(exc) from exc
                except Exception as exc:
                    _burn_for_call(
                        db,
                        upload_job_id_v2=upload_job_id_v2,
                        operation="videos.insert",
                        predicted_cost=100,
                        exc=exc,
                        http_status=0,
                    )
                    raise TransientUploadError(
                        f"chunk failed: {exc}"
                    ) from exc

                # Capture the resumable URI after the first negotiation.
                if not first_chunk_done:
                    first_chunk_done = True
                    new_uri = getattr(request, "resumable_uri", None)
                    if new_uri and on_uri_obtained is not None:
                        try:
                            on_uri_obtained(str(new_uri))
                        except Exception:
                            pass

                if status is not None:
                    uploaded_bytes = int(
                        getattr(status, "resumable_progress", 0) or 0
                    )
                    if on_bytes_uploaded is not None:
                        try:
                            on_bytes_uploaded(uploaded_bytes)
                        except Exception:
                            pass
                    if progress_cb is not None and uploaded_bytes != last_emitted:
                        try:
                            progress_cb(uploaded_bytes, size)
                        except Exception:
                            pass
                        last_emitted = uploaded_bytes

            if not response or "id" not in response:
                raise UploadError(
                    f"videos.insert returned no id: {response!r}"
                )

            video_id = str(response["id"])
            uploaded_bytes = size
            http_status = 200
            if on_bytes_uploaded is not None:
                try:
                    on_bytes_uploaded(uploaded_bytes)
                except Exception:
                    pass
            if progress_cb is not None:
                try:
                    progress_cb(size, size)
                except Exception:
                    pass
            if _call is not None and hasattr(_call, "record_video_id"):
                try:
                    _call.record_video_id(video_id)
                except Exception:
                    pass
            if _call is not None and hasattr(_call, "record_http_status"):
                try:
                    _call.record_http_status(http_status)
                except Exception:
                    pass

            # Success: one burn_log row at the published 100u cost
            # (videos.insert was repriced 1,600 → 100 on 2025-12-04).
            _burn_for_call(
                db,
                upload_job_id_v2=upload_job_id_v2,
                operation="videos.insert",
                predicted_cost=1600,
                http_status=http_status,
            )

            return {
                "video_id": video_id,
                "bytes_uploaded": int(uploaded_bytes),
                "http_status": int(http_status),
            }
    finally:
        # googleapiclient keeps a file handle on MediaFileUpload —
        # closing it lets the next op on this path proceed on Windows.
        try:
            if media is not None and media.stream() is not None:
                media.stream().close()
        except Exception:
            pass


# ─── Thumbnail helper (delegate to the legacy module so we don't fork) ─


def set_thumbnail(
    creds: Credentials,
    video_id: str,
    thumb_local_path: str,
    *,
    db: Optional[Any] = None,
    upload_job_id: Optional[int] = None,
    upload_job_id_v2: Optional[int] = None,
    user_id: Optional[int] = None,
    channel_id: Optional[int] = None,
    google_channel_id: str = "",
    clip_id: Optional[int] = None,
) -> dict:
    """Set the video's thumbnail (50 quota units).

    Wraps the call in ``log_youtube_call`` and raises the same
    UploadError taxonomy as ``upload_video``. The caller MUST have
    independently reserved ``COST_THUMBNAIL_SET`` quota AND must not
    call this for Shorts (the YouTube API does not support custom
    thumbnails on Shorts — brief §2 table).

    Returns ``{"http_status": int}``.
    """
    if not thumb_local_path or not os.path.isfile(thumb_local_path):
        raise UploadError(f"Thumbnail file missing: {thumb_local_path!r}")

    try:
        from learning.youtube_quota_log import log_youtube_call as _log_yt
    except Exception:
        _log_yt = None

    yt = _yt(creds)
    media = MediaFileUpload(
        thumb_local_path, mimetype="image/jpeg", resumable=False,
    )
    try:
        with (_log_yt(
                db=db,
                user_id=user_id,
                job_id=None,
                clip_id=clip_id,
                upload_job_id=upload_job_id,
                channel_id=channel_id,
                google_channel_id=(google_channel_id or "")[:50],
                video_id=(video_id or "")[:50],
                operation="thumbnails.set",
              ) if _log_yt is not None else _nullctx()) as _call:
            try:
                yt.thumbnails().set(
                    videoId=video_id, media_body=media,
                ).execute()
            except HttpError as exc:
                _burn_for_call(
                    db,
                    upload_job_id_v2=upload_job_id_v2,
                    operation="thumbnails.set",
                    predicted_cost=50,
                    exc=exc,
                )
                raise _wrap_http(exc) from exc

            if _call is not None and hasattr(_call, "record_http_status"):
                try:
                    _call.record_http_status(200)
                except Exception:
                    pass
            _burn_for_call(
                db,
                upload_job_id_v2=upload_job_id_v2,
                operation="thumbnails.set",
                predicted_cost=50,
                http_status=200,
            )
            return {"http_status": 200}
    finally:
        try:
            if media is not None and media.stream() is not None:
                media.stream().close()
        except Exception:
            pass


# ─── Idempotency confirmation via videos.list (1 unit) ─────────────


def confirm_video_exists(
    creds: Credentials,
    video_id: str,
    *,
    db: Optional[Any] = None,
    upload_job_id: Optional[int] = None,
    upload_job_id_v2: Optional[int] = None,
    user_id: Optional[int] = None,
    channel_id: Optional[int] = None,
    google_channel_id: str = "",
) -> bool:
    """Return True if YouTube has a video with the given id.

    Used by the idempotency short-circuit (brief §2 / §9): when we
    already recorded a ``youtube_video_id`` from a prior attempt and
    are about to re-upload, we confirm via ``videos.list`` (1 unit).
    NEVER ``search.list`` (100 units, brief §9).
    """
    if not video_id:
        return False
    try:
        from learning.youtube_quota_log import log_youtube_call as _log_yt
    except Exception:
        _log_yt = None

    yt = _yt(creds)
    try:
        with (_log_yt(
                db=db,
                user_id=user_id,
                upload_job_id=upload_job_id,
                channel_id=channel_id,
                google_channel_id=(google_channel_id or "")[:50],
                video_id=(video_id or "")[:50],
                operation="videos.list",
              ) if _log_yt is not None else _nullctx()) as _call:
            try:
                resp = yt.videos().list(
                    part="id,status",
                    id=video_id,
                ).execute()
            except HttpError as exc:
                _burn_for_call(
                    db,
                    upload_job_id_v2=upload_job_id_v2,
                    operation="videos.list",
                    predicted_cost=1,
                    exc=exc,
                )
                # If videos.list itself fails transiently, surface as
                # transient so the caller can retry; on terminal HTTP
                # we still raise so the dispatch can log.
                raise _wrap_http(exc) from exc
            if _call is not None and hasattr(_call, "record_http_status"):
                try:
                    _call.record_http_status(200)
                except Exception:
                    pass
            _burn_for_call(
                db,
                upload_job_id_v2=upload_job_id_v2,
                operation="videos.list",
                predicted_cost=1,
                http_status=200,
            )
            items = (resp or {}).get("items") or []
            return bool(items)
    except UploadError:
        # videos.list never burns 1,600 units, so a permanent error here
        # means the video genuinely cannot be confirmed. Return False
        # and let the caller decide what to do.
        return False


# ─── HTTP error classification (copied from uploader.py:326 for isolation) ─


_TRANSIENT_STATUS = {500, 502, 503, 504}
_TRANSIENT_REASONS = {
    "rateLimitExceeded", "userRateLimitExceeded", "internalError",
    "backendError", "uploadLimitExceeded",
}
_QUOTA_REASONS = {"quotaExceeded", "dailyLimitExceeded"}


def _wrap_http(exc: HttpError) -> UploadError:
    """Classify a Google HttpError into transient / permanent / quota
    so the dispatch layer can route each to the correct refund + park
    semantics (brief §9)."""
    status = getattr(exc.resp, "status", 0) if hasattr(exc, "resp") else 0
    reason = ""
    try:
        import json as _json
        content = exc.content if isinstance(exc.content, (bytes, bytearray)) else b""
        data = _json.loads(content.decode("utf-8")) if content else {}
        errs = (data.get("error") or {}).get("errors") or []
        if errs:
            reason = errs[0].get("reason") or ""
    except Exception:
        pass

    if reason in _QUOTA_REASONS:
        return QuotaExceededError(f"quota {status} {reason}: {exc}")
    if status in _TRANSIENT_STATUS or reason in _TRANSIENT_REASONS:
        return TransientUploadError(f"transient {status} {reason}: {exc}")
    return UploadError(f"permanent {status} {reason}: {exc}")


from contextlib import contextmanager as _contextmanager


@_contextmanager
def _nullctx():
    yield None


__all__ = [
    "CHUNK_SIZE",
    "UploadError",
    "TransientUploadError",
    "QuotaExceededError",
    "upload_video",
    "set_thumbnail",
    "confirm_video_exists",
    "sanitize_tags",
]
