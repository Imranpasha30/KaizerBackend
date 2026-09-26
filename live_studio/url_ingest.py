"""yt-dlp ingest for Live Studio.

When a user pastes a YouTube URL in Live Studio, the streams it
creates need a video file on disk before the existing broadcast
pipeline can kick in. This module handles that bridge:

  1. Verify the URL with yt-dlp's metadata extractor (no download).
  2. Download the best MP4 (≤1080p) to ONE shared path per
     ``(batch_id, video_slot)``.
  3. Hard-link the downloaded file to every ``LiveStream.upload_path``
     that shares the slot — so each broadcast row has its own file
     handle without doubling disk usage.
  4. Mark each row ``upload_done=True`` and status ``uploaded`` so the
     standard ``/streams/{id}/start`` endpoint picks them up.

Failures land in ``LiveStream.error`` with a human-readable message;
no exception leaks to the request handler.
"""
from __future__ import annotations

import os
import sys as _sys
import shutil as _shutil
import re
import subprocess
import threading
import traceback
from pathlib import Path
from typing import Optional

import models
from database import SessionLocal
from live_studio import uploads as live_uploads


# Limit downloads to a sane resolution so disk + uplink don't blow up
# on 4K/8K sources. 1080p is YouTube's max ingest tier for most
# accounts anyway.
# The same ffmpeg the streamer uses. yt-dlp needs it for HLS sources.
_FFMPEG_BIN = (os.environ.get("FFMPEG_BIN")
               or os.environ.get("KAIZER_FFMPEG_BIN")
               or _shutil.which("ffmpeg")
               or "ffmpeg")

# ── yt-dlp, RESOLVED rather than hoped for ───────────────────────────
# A bare "yt-dlp" resolves only if the venv's bin directory is exported on
# PATH for the CHILD process. On the buildpack deploy that serves production
# it is not, and every URL broadcast died with "[Errno 2] No such file or
# directory: 'yt-dlp'" after the YouTube broadcast had already been minted.
def _resolve_ytdlp() -> list:
    explicit = (os.environ.get("YTDLP_BIN")
                or os.environ.get("KAIZER_YTDLP_BIN"))
    if explicit:
        return [explicit]
    found = _shutil.which("yt-dlp")
    if found:
        return [found]
    # The console script is missing, but the PACKAGE may still be importable
    # -- the normal state of a venv whose bin dir is not on the child's PATH.
    # `python -m yt_dlp` is the same CLI in the same kind of subprocess; it is
    # NOT yt-dlp's in-process Python API, which is what _ytdlp_download's
    # docstring warns against.
    try:
        import importlib.util
        if importlib.util.find_spec("yt_dlp") is not None:
            return [_sys.executable, "-m", "yt_dlp"]
    except Exception:
        pass
    # Nothing found. Keep the bare name so the failure names itself.
    return ["yt-dlp"]


_YTDLP_CMD = _resolve_ytdlp()


_YTDLP_FORMAT = "bv*[height<=1080][vcodec^=avc1]+ba[acodec^=mp4a]/bv*[height<=1080][vcodec^=avc1]+ba/b[height<=1080][vcodec^=avc1]/bv*[height<=1080]+ba/b"

_YT_URL_RE = re.compile(
    r"^https?://(?:www\.|m\.|music\.)?"
    r"(?:youtube\.com/(?:watch\?v=|shorts/|live/|embed/)|youtu\.be/)"
    r"[\w\-]{6,}",
    re.IGNORECASE,
)


def is_supported_url(url: str) -> bool:
    return bool(_YT_URL_RE.match((url or "").strip()))


def _shared_path(batch_id: int, video_slot: int) -> str:
    """Single canonical path that ALL streams in a (batch, slot)
    will hard-link to. Lives in the same temp dir the chunk uploads
    use so cleanup paths see it naturally."""
    base = os.path.dirname(live_uploads.upload_path_for(0))
    return os.path.join(base, f"url-b{batch_id}-v{video_slot}.mp4")


def _hardlink_or_copy(src: str, dst: str) -> None:
    """Replicate ``src`` to ``dst`` via a hardlink (cheap, same
    inode). Falls back to ``copy2`` if the filesystem refuses
    (cross-volume, FAT32, etc.). Idempotent: removes existing dst
    first."""
    try:
        if os.path.exists(dst):
            os.remove(dst)
    except OSError:
        pass
    try:
        os.link(src, dst)
    except OSError:
        # Cross-volume or unsupported — fall back to a full copy.
        import shutil
        shutil.copy2(src, dst)


def _ytdlp_download(url: str, out_path: str) -> tuple[bool, str]:
    """Blocking download via the yt-dlp CLI. Returns ``(ok, log_tail)``.

    Why the CLI not the python module: yt-dlp's Python API drives
    output through callbacks that don't behave well from a daemon
    thread on Windows. The CLI is rock-solid + we get progress on
    stderr for free.
    """
    # Cap to 1080p and force the merged output to be the path we asked
    # for (no auto-numbering of duplicates).
    cmd = [
        *_YTDLP_CMD,
        # HLS sources need ffmpeg; yt-dlp looks on PATH unless told.
        *(["--ffmpeg-location", _FFMPEG_BIN] if os.path.isfile(_FFMPEG_BIN) else []),
        "--no-progress",
        "--no-colors",
        "--no-warnings",
        "--retries", "3",
        "-f", _YTDLP_FORMAT,
        "--merge-output-format", "mp4",
        "-o", out_path,
        url,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60 * 60,    # 1 h hard ceiling per download
        )
    except subprocess.TimeoutExpired:
        return False, "yt-dlp download exceeded 1h timeout"
    except FileNotFoundError:
        return False, "yt-dlp not installed on the server PATH"

    if proc.returncode != 0 or not os.path.isfile(out_path):
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        return False, "yt-dlp failed: " + " | ".join(tail)
    return True, ""


def _mark_failed(stream_ids: list[int], reason: str) -> None:
    sess = SessionLocal()
    try:
        for sid in stream_ids:
            row = sess.query(models.LiveStream).get(sid)
            if not row:
                continue
            row.status = "failed"
            row.error = reason[:2000]
            # The reason, not the category: "URL ingest failed" told the
            # operator only that the thing that failed was the thing that
            # failed. yt-dlp's own words are what identify the fault.
            row.message = ("failed: " + " ".join(str(reason).split()))[:512]
            print(f"[live_studio] STREAM {sid} -> failed | via url_ingest"
                  f" | err={' '.join(str(reason).split())[:2000]}", flush=True)
        sess.commit()
    finally:
        sess.close()


def _mark_ready(stream_ids: list[int], paths_by_stream: dict[int, str],
                size_bytes: int) -> None:
    sess = SessionLocal()
    try:
        for sid in stream_ids:
            row = sess.query(models.LiveStream).get(sid)
            if not row:
                continue
            row.upload_path = paths_by_stream[sid]
            row.upload_bytes = size_bytes
            row.upload_total = size_bytes
            row.upload_done = True
            row.status = "uploaded"
            row.message = "downloaded from URL; awaiting broadcast slot"
        sess.commit()
    finally:
        sess.close()


def mark_passthrough_ready(
    *,
    batch_id: int,
    video_slot: int,
    source_url: str,
    stream_ids: list[int],
) -> None:
    """Pass-through (OBS-style) URL streams skip the server-side
    download entirely. The orchestrator pipes yt-dlp straight into
    ffmpeg when it sees ``source_url`` on the row.

    All we have to do here is flip every stream sharing the slot to
    ``uploaded`` (the marker the existing /start endpoint requires)
    so the user's "Start broadcasting" submission flows the row into
    the orchestrator without any wait-for-download polling.
    """
    if not is_supported_url(source_url):
        _mark_failed(stream_ids,
            f"unsupported URL (must be a youtube.com / youtu.be link): {source_url}")
        return

    sess = SessionLocal()
    try:
        for sid in stream_ids:
            row = sess.query(models.LiveStream).get(sid)
            if not row:
                continue
            row.source_url = source_url[:1024]
            row.upload_done = True             # /start gate
            row.upload_bytes = 0
            row.upload_total = 0
            row.upload_path = ""               # explicitly empty — orchestrator
                                               # uses source_url instead.
            row.status = "uploaded"
            row.message = "URL ready — passthrough live (no download)"
        sess.commit()
    finally:
        sess.close()


# Legacy download-then-broadcast helper — kept here so the previously
# wired call sites compile, but the router now drives URL streams
# through ``kick_off`` → ``mark_passthrough_ready``. Delete this in a
# follow-up once nothing else references it.
def ingest_url_for_slot(
    *,
    batch_id: int,
    video_slot: int,
    source_url: str,
    stream_ids: list[int],
) -> None:
    try:
        if not is_supported_url(source_url):
            _mark_failed(stream_ids,
                f"unsupported URL (must be a youtube.com / youtu.be link): {source_url}")
            return

        sess = SessionLocal()
        try:
            for sid in stream_ids:
                row = sess.query(models.LiveStream).get(sid)
                if not row:
                    continue
                row.status = "downloading"
                row.message = f"server is downloading from {source_url[:120]}…"
                row.source_url = source_url[:1024]
            sess.commit()
        finally:
            sess.close()

        out_path = _shared_path(batch_id, video_slot)
        Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
        ok, err = _ytdlp_download(source_url, out_path)
        if not ok:
            _mark_failed(stream_ids, err)
            try:
                if os.path.isfile(out_path):
                    os.remove(out_path)
            except OSError:
                pass
            return

        size = os.path.getsize(out_path)

        # 3) Hard-link the file to each stream's canonical upload_path
        # so the orchestrator's existing cleanup (which unlinks per
        # stream) works without cross-deletes. Hardlinks share the
        # inode → space cost is one copy regardless of N.
        paths: dict[int, str] = {}
        for sid in stream_ids:
            stream_path = live_uploads.upload_path_for(sid)
            try:
                _hardlink_or_copy(out_path, stream_path)
                paths[sid] = stream_path
            except OSError as exc:
                _mark_failed(
                    [sid],
                    f"could not replicate download to stream {sid}: {exc}",
                )
        if not paths:
            return

        _mark_ready(list(paths.keys()), paths, size)

    except Exception:
        tb = traceback.format_exc()
        print(f"[live_studio/url_ingest] ingest crashed for batch={batch_id} "
              f"slot={video_slot}:\n{tb}")
        _mark_failed(stream_ids, f"ingest crashed: {tb.splitlines()[-1][:200]}")


def kick_off(*, batch_id: int, video_slot: int, source_url: str,
             stream_ids: list[int]) -> None:
    """OBS-style passthrough: flip the rows to ``uploaded`` immediately
    so /start can run. The orchestrator does the actual yt-dlp →
    ffmpeg → YouTube pipe inline once it acquires a slot. No disk
    landing, no looping — broadcast ends when the source ends.

    Synchronous (no thread) because it's a single UPDATE statement.
    """
    mark_passthrough_ready(
        batch_id=batch_id, video_slot=video_slot,
        source_url=source_url, stream_ids=stream_ids,
    )
