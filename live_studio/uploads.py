"""Chunked upload writer for Live Studio.

Protocol
--------
Browser POSTs chunks of the video file to the same URL, one chunk per
request. Each request carries:

  Content-Range: bytes <start>-<end>/<total>
  Content-Type:  application/octet-stream
  Body:          raw bytes of the chunk

The server appends each chunk to a growing temp file. When the last
chunk lands (i.e. ``end + 1 == total``), the writer marks ``upload_done``
on the LiveStream row.

Why not TUS
-----------
TUS is the standard but it needs three endpoints (HEAD/POST/PATCH)
and an upload-id negotiation step. Our flow is simpler: the
LiveStream row IS the upload session. Browser already has its id
(returned at batch-create time) so we just use POST per chunk and
the row's ``upload_bytes`` + ``upload_total`` track progress.

Resumability
------------
If a chunk fails mid-flight, the browser retries from the last
``upload_bytes`` value (it asks via GET first if it doesn't remember).
The server is idempotent w.r.t. re-sending the same byte range
(seek + write), so duplicate chunks land cleanly.

Buffer threshold
----------------
The orchestrator watches ``upload_bytes`` and starts ffmpeg as soon as
``CHUNK_THRESHOLD`` (~5 MB) is reached — enough video data for ffmpeg
to read MP4 metadata + start streaming without stalling for the moov
atom (if the input is fragmented or fast-start optimized; otherwise
we wait for ``upload_done``).
"""
from __future__ import annotations

import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Optional


# 5 MB minimum buffered before broadcast can start (sane default for
# 1080p H.264 at ~5 Mbps — that's ~8 seconds of video).
CHUNK_THRESHOLD_BYTES = 5 * 1024 * 1024


_UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "kaizer-live-studio")
os.makedirs(_UPLOAD_DIR, exist_ok=True)


# Per-stream write locks so concurrent chunk POSTs to the same row
# don't race. Cleared when the upload completes.
_WRITE_LOCKS: dict[int, threading.Lock] = {}
_LOCKS_LOCK = threading.Lock()


def _lock_for(stream_id: int) -> threading.Lock:
    with _LOCKS_LOCK:
        lk = _WRITE_LOCKS.get(stream_id)
        if lk is None:
            lk = threading.Lock()
            _WRITE_LOCKS[stream_id] = lk
        return lk


def _release_lock(stream_id: int) -> None:
    with _LOCKS_LOCK:
        _WRITE_LOCKS.pop(stream_id, None)


_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$")


def parse_content_range(header: Optional[str]) -> Optional[tuple[int, int, Optional[int]]]:
    """Parse ``Content-Range: bytes 0-5242879/123456789`` →
    ``(start, end, total)``. ``end`` is the LAST byte (inclusive,
    per RFC 9110). ``total`` is None when the client sent ``*``.

    Returns None when the header is missing or malformed."""
    if not header:
        return None
    m = _RANGE_RE.match(header.strip())
    if not m:
        return None
    start = int(m.group(1))
    end   = int(m.group(2))
    total = None if m.group(3) == "*" else int(m.group(3))
    if start < 0 or end < start:
        return None
    return start, end, total


def upload_path_for(stream_id: int, suffix: str = ".mp4") -> str:
    """Stable absolute path for the growing temp file. Idempotent —
    just builds the name, doesn't touch the disk."""
    safe_suffix = (suffix or ".mp4").lower()
    if not safe_suffix.startswith("."):
        safe_suffix = "." + safe_suffix
    safe_suffix = re.sub(r"[^.\w]", "", safe_suffix)[:8] or ".mp4"
    return os.path.join(_UPLOAD_DIR, f"stream-{stream_id}{safe_suffix}")


def write_chunk(
    *,
    stream_id: int,
    chunk: bytes,
    start: int,
    end: int,
    total: Optional[int],
    suffix: str = ".mp4",
) -> dict:
    """Append (or seek+write) one chunk to the stream's temp file.

    Returns ``{bytes_written_so_far, is_complete, path}``. The
    caller updates the DB row from these values.
    """
    if not chunk:
        raise ValueError("empty chunk body")
    expected_len = end - start + 1
    if len(chunk) != expected_len:
        raise ValueError(
            f"chunk size mismatch: header says {expected_len} bytes "
            f"({start}-{end}), got {len(chunk)} bytes in body"
        )

    path = upload_path_for(stream_id, suffix)
    lock = _lock_for(stream_id)
    with lock:
        mode = "r+b" if os.path.exists(path) else "w+b"
        with open(path, mode) as fh:
            fh.seek(start)
            fh.write(chunk)
            fh.flush()
            os.fsync(fh.fileno())
            cur = fh.tell()    # last byte we just wrote + 1

    is_complete = bool(total) and cur >= total
    if is_complete:
        # Last chunk just landed — release the per-stream lock so the
        # writer dict doesn't leak.
        _release_lock(stream_id)
    return {"bytes_written": cur, "is_complete": is_complete, "path": path}


def current_size(stream_id: int, suffix: str = ".mp4") -> int:
    """Return the current size of the partial upload, or 0 if not
    started yet. Used by the orchestrator to decide when to start
    ffmpeg (after ``CHUNK_THRESHOLD_BYTES``)."""
    path = upload_path_for(stream_id, suffix)
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def delete_upload(stream_id: int, suffix: str = ".mp4") -> None:
    """Clean up the temp file. Best-effort — broadcasts can still
    fail after the cleanup runs (e.g. R2 backup completed), so the
    R2 copy is the durable artifact."""
    path = upload_path_for(stream_id, suffix)
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as exc:
        print(f"[live_studio/uploads] cleanup skip {path}: {exc}")
    _release_lock(stream_id)
