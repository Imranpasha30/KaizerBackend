"""Meta Resumable Upload helper.

Meta's Graph API supports two upload styles for video:
  1. One-shot multipart  — for files ≤100 MB; what we use today.
  2. Resumable session   — for files >100 MB up to the 4 GB cap.
                           Three-step: START → TRANSFER (chunked) → FINISH.

Our bulletins typically run several minutes at 1080p, which usually
clears 100 MB. Without this helper, the FB Page publisher would refuse
or 400 on those files.

This module is a thin wrapper around the Resumable Upload endpoints
so the FB/IG publishers can call ``post_resumable(...)`` instead of
the one-shot path when ``os.path.getsize(file) > 100 MB``.

Reference:
  https://developers.facebook.com/docs/video-api/guides/publishing/
"""
from __future__ import annotations

import os
from typing import Optional

import httpx


META_GRAPH_API_VERSION = os.environ.get("META_GRAPH_API_VERSION", "v21.0")
GRAPH_BASE = f"https://graph.facebook.com/{META_GRAPH_API_VERSION}"

# Threshold above which we MUST use resumable. Spec is 100 MB.
RESUMABLE_THRESHOLD_BYTES = 100 * 1024 * 1024
# Per-chunk size — Meta accepts up to ~50 MB but smaller chunks make
# transient retries cheaper.
CHUNK_BYTES = 8 * 1024 * 1024


class MetaUploadError(Exception):
    """Raised when any of the three steps fails. Carries the Meta
    error body for upstream propagation into a Terminal/Transient
    Publish error."""
    def __init__(self, message: str, *, status: int = 0, body: Optional[dict] = None):
        super().__init__(message)
        self.status = status
        self.body = body or {}


def _raise_if_error(r: httpx.Response, what: str) -> None:
    if r.status_code < 400:
        return
    try:
        body = r.json()
    except Exception:
        body = {"raw": r.text[:500]}
    raise MetaUploadError(
        f"resumable {what} failed: {r.status_code} {body}",
        status=r.status_code,
        body=body,
    )


def post_resumable_to_page(
    *,
    page_id: str,
    access_token: str,
    file_path: str,
    title: str = "",
    description: str = "",
    endpoint_path: str = "videos",
) -> dict:
    """Three-step resumable upload to a FB Page's video endpoint.

    Returns the full response body of the FINISH call, which carries
    the {id} of the published video post.
    """
    file_size = os.path.getsize(file_path)

    with httpx.Client(timeout=600) as cx:
        # ── Step 1: START ──
        # Creates an upload_session_id + reports the offset we should
        # start writing at.
        r1 = cx.post(
            f"{GRAPH_BASE}/{page_id}/{endpoint_path}",
            data={
                "access_token": access_token,
                "upload_phase": "start",
                "file_size": file_size,
            },
        )
        _raise_if_error(r1, "start")
        session = r1.json()
        upload_session_id = session.get("upload_session_id")
        if not upload_session_id:
            raise MetaUploadError(f"start returned no upload_session_id: {session}")
        start_offset = int(session.get("start_offset", 0))
        end_offset   = int(session.get("end_offset", min(CHUNK_BYTES, file_size)))

        # ── Step 2: TRANSFER (chunks) ──
        # Each chunk POST returns the NEXT [start_offset, end_offset)
        # we should send. Loop until start_offset == file_size.
        with open(file_path, "rb") as fh:
            while start_offset < file_size:
                fh.seek(start_offset)
                chunk = fh.read(end_offset - start_offset)
                if not chunk:
                    break
                r2 = cx.post(
                    f"{GRAPH_BASE}/{page_id}/{endpoint_path}",
                    data={
                        "access_token": access_token,
                        "upload_phase": "transfer",
                        "upload_session_id": upload_session_id,
                        "start_offset": start_offset,
                    },
                    files={
                        "video_file_chunk": ("chunk", chunk, "application/octet-stream"),
                    },
                )
                _raise_if_error(r2, f"transfer @{start_offset}")
                body2 = r2.json()
                start_offset = int(body2.get("start_offset", start_offset))
                end_offset   = int(body2.get("end_offset", start_offset))

        # ── Step 3: FINISH ──
        r3 = cx.post(
            f"{GRAPH_BASE}/{page_id}/{endpoint_path}",
            data={
                "access_token": access_token,
                "upload_phase": "finish",
                "upload_session_id": upload_session_id,
                "title": title[:200] if title else "",
                "description": description[:5000] if description else "",
            },
        )
        _raise_if_error(r3, "finish")
        return r3.json()
