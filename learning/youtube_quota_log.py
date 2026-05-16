"""YouTube Data API call accounting wrapper.

Every site that calls a YouTube Data API method (``yt.videos().insert()``,
``yt.thumbnails().set()``, ``yt.playlistItems().list()``, etc.) should
wrap the call in ``log_youtube_call(...)``.  The wrapper writes one
``YouTubeApiCall`` row per call so the admin Usage dashboard can answer:

* "Which videos burned the most quota today?"
* "What % of the 10 000-unit daily cap have we used?"
* "Which user / channel is driving us toward the cap?"

YouTube doesn't bill dollars — it bills *quota units* against a daily
cap (10 000 / day per Google Cloud project by default).  Costs come
from a fixed table maintained in this file.

Usage
-----

    from learning.youtube_quota_log import log_youtube_call

    with log_youtube_call(
        db=db,
        user_id=user.id,
        job_id=job.id,
        upload_job_id=upload_job.id,
        channel_id=channel.id,
        google_channel_id=tok.google_channel_id,
        operation="videos.insert",
        publish_kind="short",
        file_bytes=path.stat().st_size,
        duration_seconds=clip.duration_s,
    ) as call:
        resp = request.next_chunk()
        call.record_video_id(resp[1])      # if the call returned a video id
        return resp
"""
from __future__ import annotations

import logging
import time
import traceback
from contextlib import contextmanager
from typing import Any, Optional

from sqlalchemy.orm import Session

import models


logger = logging.getLogger("kaizer.youtube_quota_log")


# ─── YouTube Data API operation costs (quota units) ──────────────────
# Source: https://developers.google.com/youtube/v3/determine_quota_cost
# Default daily cap = 10 000 units.  Uploads are the dominant cost.
OPERATION_QUOTA: dict[str, int] = {
    "videos.insert":       1600,   # uploading a video — by far the heaviest
    "videos.update":         50,
    "videos.delete":         50,
    "videos.list":            1,
    "videos.rate":           50,
    "thumbnails.set":        50,
    "playlistItems.list":     1,
    "playlistItems.insert":  50,
    "channels.list":          1,
    "channels.update":       50,
    "search.list":          100,
    "captions.list":         50,
    "captions.insert":      400,
}
_DEFAULT_OP_COST = 1


def quota_for(operation: str) -> int:
    """Return the documented quota cost for an operation.  Unknown
    operations fall back to 1 unit so the count is at least non-zero."""
    return int(OPERATION_QUOTA.get(operation or "", _DEFAULT_OP_COST))


class _Call:
    """Handle returned from ``log_youtube_call(...)`` so the caller can
    set the video_id once they have it (the upload doesn't return until
    the resumable session completes)."""

    def __init__(
        self,
        operation: str,
        publish_kind: str = "",
        file_bytes: int = 0,
        duration_seconds: float = 0.0,
        video_id: str = "",
    ):
        self.operation       = operation
        self.publish_kind    = publish_kind or ""
        self.file_bytes      = int(file_bytes or 0)
        self.duration_seconds = float(duration_seconds or 0.0)
        self.video_id        = video_id or ""
        self.quota_cost      = quota_for(operation)
        self.success         = True
        self.http_status     = 0
        self.error           = ""

    def record_video_id(self, video_id: str) -> None:
        """For ``videos.insert`` — call once the upload completes so we
        can correlate quota burn → published video."""
        if video_id:
            self.video_id = str(video_id)

    def record_http_status(self, status: int) -> None:
        """Optional — set the HTTP status the API returned.  Useful when
        the call succeeded but the server told us we were partially
        rate-limited (e.g. 403 quotaExceeded mid-stream)."""
        try:
            self.http_status = int(status)
        except Exception:
            pass


@contextmanager
def log_youtube_call(
    db: Optional[Session],
    *,
    user_id: Optional[int] = None,
    job_id: Optional[int] = None,
    clip_id: Optional[int] = None,
    upload_job_id: Optional[int] = None,
    channel_id: Optional[int] = None,
    google_channel_id: str = "",
    operation: str,
    publish_kind: str = "",
    file_bytes: int = 0,
    duration_seconds: float = 0.0,
    video_id: str = "",
):
    """Context manager that writes a ``YouTubeApiCall`` row when the
    ``with`` block exits, success or failure.

    The quota cost is looked up from ``OPERATION_QUOTA`` — the caller
    doesn't have to pass it (YouTube's per-operation costs are public
    and rarely change).
    """
    call = _Call(
        operation=operation,
        publish_kind=publish_kind,
        file_bytes=file_bytes,
        duration_seconds=duration_seconds,
        video_id=video_id,
    )
    start = time.monotonic()
    try:
        yield call
    except BaseException as e:
        call.success = False
        msg = str(e)
        # googleapiclient HttpError exposes resp.status; pull it where
        # possible so the dashboard can distinguish 403-quota from
        # transient 5xx.
        try:
            resp = getattr(e, "resp", None)
            if resp is not None:
                call.http_status = int(getattr(resp, "status", 0) or 0)
        except Exception:
            pass
        call.error = (msg or e.__class__.__name__)[:1000]
        raise
    finally:
        latency_ms = int((time.monotonic() - start) * 1000)
        _owned_session = False
        session: Optional[Session] = db
        try:
            if session is None:
                try:
                    from database import SessionLocal
                    session = SessionLocal()
                    _owned_session = True
                except Exception:
                    session = None

            if session is not None:
                row = models.YouTubeApiCall(
                    user_id=user_id,
                    job_id=job_id,
                    clip_id=clip_id,
                    upload_job_id=upload_job_id,
                    channel_id=channel_id,
                    google_channel_id=(google_channel_id or "")[:50],
                    video_id=(call.video_id or "")[:50],
                    operation=(operation or "")[:64],
                    quota_cost=int(call.quota_cost or 0),
                    publish_kind=(call.publish_kind or "")[:10],
                    file_bytes=int(call.file_bytes or 0),
                    duration_seconds=float(call.duration_seconds or 0.0),
                    success=bool(call.success),
                    http_status=int(call.http_status or 0),
                    error=call.error,
                )
                session.add(row)
                session.commit()
        except Exception as log_err:
            logger.warning(
                "youtube_quota_log: failed to persist row "
                "(op=%s, user=%s, job=%s): %s\n%s",
                operation, user_id, job_id, log_err,
                traceback.format_exc(limit=3),
            )
            try:
                if session is not None:
                    session.rollback()
            except Exception:
                pass
        finally:
            if _owned_session and session is not None:
                try:
                    session.close()
                except Exception:
                    pass
        # latency isn't a column today — kept in the local for future use
        _ = latency_ms
