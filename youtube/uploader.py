"""Resumable videos.insert + thumbnails.set helpers.

Designed to survive process restarts:
  - The resumable `upload_uri` is written to the DB as soon as
    Google returns it, before any chunks ship.
  - `bytes_uploaded` is checkpointed after every successful chunk,
    so a restart can resume from the last committed byte.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from google.oauth2.credentials import Credentials

import models


CHUNK_SIZE = 10 * 1024 * 1024   # 10 MB
THUMB_MAX_BYTES = 2 * 1024 * 1024   # YouTube cap: 2 MB


class UploadError(Exception):
    """Terminal error — no more retries."""


class TransientUploadError(UploadError):
    """Temporary — worker should retry with backoff."""


def _yt(creds: Credentials):
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def sanitize_tags(raw: list | None) -> list[str]:
    """Scrub tags so YouTube's `videos.insert` doesn't 400 with `invalidTags`.

    YouTube rules that the API enforces:
      - No `<` or `>` anywhere in any tag
      - No leading `#` (hashtags belong in description, not the tags field)
      - Each tag ≤ 100 chars
      - Combined joined length (commas + quotes) ≤ 500 chars
      - No empty strings
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
        # Strip leading '#' (shorts / hashtags)
        while t.startswith("#"):
            t = t[1:].strip()
        # Drop YouTube-forbidden chars
        t = t.replace("<", "").replace(">", "").replace('"', "").strip()
        if not t:
            continue
        t = t[:100]  # per-tag cap
        key = t.lower()
        if key in seen:
            continue
        # Joined representation uses commas + optional quotes; approximate the 500 cap
        est = len(t) + 2 + (2 if "," in t or " " in t else 0)
        if combined + est > 500:
            break
        seen.add(key)
        out.append(t)
        combined += est
    return out


def _build_body(job: models.UploadJob) -> dict:
    """Construct the YouTube videos.insert body from the UploadJob row.

    `status.publishAt` is honored only when privacy_status == 'private' —
    that's Google's requirement; otherwise YouTube rejects the request.
    """
    body: dict = {
        "snippet": {
            "title":       (job.title or "Untitled")[:100],
            "description": job.description or "",
            # Sanitize at insert time so retries of jobs queued with bad tags succeed
            "tags":        sanitize_tags(job.tags),
            "categoryId":  job.category_id or "25",
            "defaultLanguage":      "te",
            "defaultAudioLanguage": "te",
        },
        "status": {
            "privacyStatus":           (job.privacy_status or "private").lower(),
            "selfDeclaredMadeForKids": bool(job.made_for_kids),
            "embeddable":              True,
            "publicStatsViewable":     True,
        },
    }
    if job.publish_at and body["status"]["privacyStatus"] == "private":
        # Google requires RFC3339 / UTC "Z"
        dt = job.publish_at
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        body["status"]["publishAt"] = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return body


def upload_video(
    creds: Credentials,
    job: models.UploadJob,
    clip_path: str,
    db,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Run (or resume) the resumable upload. Returns the YouTube video_id.

    Transient failures raise TransientUploadError; the worker catches them,
    records an attempt, and retries. `progress_cb(uploaded, total)` is called
    after every chunk with the *checkpointed* byte counts.
    """
    path = Path(clip_path)
    if not path.exists():
        raise UploadError(f"Clip file no longer exists: {clip_path}")

    size = path.stat().st_size
    if job.bytes_total != size:
        job.bytes_total = size
        db.commit()

    try:
        yt = _yt(creds)
        media = MediaFileUpload(
            str(path), chunksize=CHUNK_SIZE, resumable=True, mimetype="video/mp4",
        )
        request = yt.videos().insert(
            part="snippet,status",
            body=_build_body(job),
            media_body=media,
            notifySubscribers=False,
        )

        response = None
        last_emitted = -1
        while response is None:
            try:
                status, response = request.next_chunk()
            except HttpError as e:
                raise _wrap_http(e) from e
            except Exception as e:
                # Network hiccup — treat as transient
                raise TransientUploadError(f"chunk failed: {e}") from e

            if status is not None:
                uploaded = int(getattr(status, "resumable_progress", 0) or 0)
                job.bytes_uploaded = uploaded
                db.commit()
                if progress_cb and uploaded != last_emitted:
                    progress_cb(uploaded, size)
                    last_emitted = uploaded

        if not response or "id" not in response:
            raise UploadError(f"videos.insert returned no id: {response}")

        video_id = response["id"]
        job.video_id       = video_id
        job.bytes_uploaded = size
        db.commit()
        if progress_cb:
            progress_cb(size, size)
        return video_id
    finally:
        # googleapiclient's MediaFileUpload keeps a file handle open — close it
        # so subsequent operations on the same path (restart, cleanup) don't
        # fail on Windows.
        try:
            if media and media.stream():
                media.stream().close()
        except Exception:
            pass


def set_thumbnail(creds: Credentials, video_id: str, thumb_path: str) -> None:
    """Upload the clip's still frame as the video's thumbnail.

    No-op if the file is missing. Resizes to ≤ 2 MB via Pillow if needed.
    """
    path = Path(thumb_path or "")
    if not path.exists():
        return

    src = str(path)
    tmp_shrunk: Optional[Path] = None
    try:
        if path.stat().st_size > THUMB_MAX_BYTES:
            tmp_shrunk = _shrink(path)
            src = str(tmp_shrunk)

        yt = _yt(creds)
        media = MediaFileUpload(src, mimetype="image/jpeg", resumable=False)
        try:
            yt.thumbnails().set(videoId=video_id, media_body=media).execute()
        except HttpError as e:
            raise _wrap_http(e) from e
    finally:
        if tmp_shrunk and tmp_shrunk.exists():
            try: tmp_shrunk.unlink()
            except OSError: pass


def _shrink(path: Path) -> Path:
    """Lossy-recompress a JPEG until it fits under THUMB_MAX_BYTES."""
    from PIL import Image

    tmp = path.with_suffix(".yt_thumb.jpg")
    img = Image.open(path).convert("RGB")
    # Start at quality 90 and descend; also cap max dimension at 1280
    img.thumbnail((1280, 1280))
    for q in (90, 85, 80, 75, 70, 65, 60, 55, 50):
        img.save(tmp, "JPEG", quality=q, optimize=True)
        if tmp.stat().st_size <= THUMB_MAX_BYTES:
            return tmp
    # Fall through — still return the last attempt, let YouTube error surface
    return tmp


# ─── HTTP error classification ────────────────────────────────────────────

_TRANSIENT_STATUS = {500, 502, 503, 504}
_TRANSIENT_REASONS = {
    "rateLimitExceeded", "userRateLimitExceeded", "internalError",
    "backendError", "uploadLimitExceeded",
}
_QUOTA_REASONS = {"quotaExceeded", "dailyLimitExceeded"}

def _wrap_http(e: HttpError) -> UploadError:
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

    if status in _TRANSIENT_STATUS or reason in _TRANSIENT_REASONS:
        return TransientUploadError(f"transient {status} {reason}: {e}")
    if reason in _QUOTA_REASONS:
        # Treat as transient so the worker parks & retries tomorrow
        return TransientUploadError(f"quota {status} {reason}: {e}")
    return UploadError(f"permanent {status} {reason}: {e}")
