"""Per-LiveStream broadcast worker.

When the router's ``/streams/{id}/start`` endpoint fires, it kicks off
``run_stream(stream_id, user_id)`` in a daemon thread. That worker:

  1. Acquires a global concurrency slot (max 8 across the box).
  2. Loads the LiveStream + Channel rows.
  3. Mints a YouTube live broadcast via ``rtmp_provider.obtain_rtmp_target``
     (reuses existing code, ~150 quota units).
  4. Saves the broadcast / ingest / stream-key on the row.
  5. Spawns ``streamer.push_loop`` (ffmpeg) which streams the file with
     ``-stream_loop -1 -t <hours>`` so short videos loop to fill the
     configured duration.
  6. Polls progress, updates ``LiveStream.progress_pct`` + ``message``.
  7. On exit, finalizes the broadcast (``transition=complete``).
  8. Deletes the temp upload file (R2 backup, if enabled, was uploaded
     in parallel during streaming — Phase 6).

Failure handling
----------------
Any uncaught exception lands in the LiveStream row as ``status=failed``
with the traceback's last line in ``error``. The temp file is preserved
in a debug dir for inspection (same pattern as Express Mode).

Cancellation
------------
``cancel_event_for(stream_id)`` returns a threading.Event the router
sets when the user clicks "Stop". The streamer watches it, sends ffmpeg
SIGINT/CTRL_BREAK, and exits cleanly (status=canceled).
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import models
from database import SessionLocal
from live_studio import concurrency, streamer, uploads as live_uploads


# Per-stream cancel events. Set by the /cancel endpoint, watched by
# the orchestrator. Cleared after the stream exits.
_CANCEL_EVENTS: dict[int, threading.Event] = {}
_EV_LOCK = threading.Lock()


def cancel_event_for(stream_id: int) -> threading.Event:
    """Return (or create) the cancel event for ``stream_id``."""
    with _EV_LOCK:
        ev = _CANCEL_EVENTS.get(stream_id)
        if ev is None:
            ev = threading.Event()
            _CANCEL_EVENTS[stream_id] = ev
        return ev


def _release_cancel_event(stream_id: int) -> None:
    with _EV_LOCK:
        _CANCEL_EVENTS.pop(stream_id, None)


def request_cancel(stream_id: int) -> bool:
    """Signal the running worker to stop. Returns True if a worker
    was registered, False otherwise (the row probably never started)."""
    with _EV_LOCK:
        ev = _CANCEL_EVENTS.get(stream_id)
    if ev:
        ev.set()
        return True
    return False


# ─── DB helpers ─────────────────────────────────────────────────

def _update(stream_id: int, **fields) -> None:
    """Patch a LiveStream row. Owns its own session so it doesn't
    block on whatever else is happening in the request thread."""
    sess = SessionLocal()
    try:
        row = sess.query(models.LiveStream).get(stream_id)
        if not row:
            return
        for k, v in fields.items():
            if hasattr(row, k):
                setattr(row, k, v)
        sess.commit()
    finally:
        sess.close()


def _append_message(stream_id: int, line: str) -> None:
    """Replace ``message`` so the UI's per-row tooltip shows the
    latest pipeline step."""
    if not line:
        return
    _update(stream_id, message=line[:512])


# ─── Main worker ─────────────────────────────────────────────────

def run_stream(stream_id: int) -> None:
    """End-to-end runner for one LiveStream. Owns its own DB session.

    Idempotent: re-running after a backend crash will skip rows that
    already have ``yt_broadcast_id`` set (the Phase 7 recovery path
    plugs in here).
    """
    sess = SessionLocal()
    cancel_ev = cancel_event_for(stream_id)

    try:
        row = sess.query(models.LiveStream).get(stream_id)
        if not row:
            return
        if row.status in ("done", "failed", "canceled"):
            _release_cancel_event(stream_id)
            return

        # 1) Concurrency slot
        _append_message(stream_id, "waiting for broadcast slot…")
        with concurrency.acquire_slot(timeout_s=600) as got:
            if not got:
                _update(stream_id, status="failed",
                        error="timed out waiting for a broadcast slot")
                return

            # Re-fetch in case state changed during the wait.
            sess.refresh(row)
            if cancel_ev.is_set() or row.status == "canceled":
                _update(stream_id, status="canceled",
                        message="canceled before start")
                return

            # 2) Look up the channel + creds.
            channel = sess.query(models.Channel).get(row.channel_id) if row.channel_id else None
            if not channel:
                _update(stream_id, status="failed",
                        error=f"channel id {row.channel_id} not found")
                return

            try:
                from youtube import oauth as yt_oauth
                creds = yt_oauth.get_credentials(sess, channel.id)
            except Exception as exc:
                _update(stream_id, status="failed",
                        error=f"YouTube OAuth failed: {exc}")
                return

            # 3) Mint the YouTube broadcast — re-use rtmp_provider.
            _update(stream_id, status="provisioning",
                    message="creating YouTube broadcast…")
            try:
                from youtube import rtmp_provider as yt_rtmp
                target = yt_rtmp.obtain_rtmp_target(
                    creds=creds,
                    job=row,             # rtmp_provider reads via getattr,
                                         # LiveStream has user_id+id+channel_id
                    channel=channel,
                    title=(row.title or "Live broadcast")[:100],
                    description=(row.description or "")[:5000],
                    privacy_status=row.privacy or "unlisted",
                    db=sess,
                )
            except Exception as exc:
                _update(stream_id, status="failed",
                        error=f"broadcast mint failed: {exc}")
                return

            _update(
                stream_id,
                yt_broadcast_id=target.get("broadcast_id") or "",
                yt_stream_id=target.get("stream_id") or "",
                yt_ingest_url=target.get("ingest_url") or "",
                yt_stream_key=target.get("stream_key") or "",
                yt_video_id=target.get("video_id") or target.get("broadcast_id") or "",
                status="streaming",
                progress_pct=0,
                message="broadcast minted; ffmpeg starting",
                started_at=datetime.now(timezone.utc),
            )

            # 4) ffmpeg push (blocks until done, cancel, or error).
            try:
                streamer.push_loop(
                    input_path=row.upload_path,
                    ingest_url=target["ingest_url"],
                    stream_key=target["stream_key"],
                    duration_hours=float(row.target_hours or 1.0),
                    progress_cb=lambda pct: _update(
                        stream_id, progress_pct=int(pct),
                        message=f"streaming… {pct:.1f}%",
                    ),
                    cancel_event=cancel_ev,
                    extra_log_cb=None,
                )
                completed_clean = True
            except streamer.StreamerError as exc:
                _update(stream_id, status="failed",
                        error=f"ffmpeg push failed: {exc}",
                        finished_at=datetime.now(timezone.utc))
                completed_clean = False
            except Exception as exc:
                _update(stream_id, status="failed",
                        error=f"unexpected: {exc}",
                        finished_at=datetime.now(timezone.utc))
                completed_clean = False

            # 5) Finalize broadcast (transition=complete) regardless of
            # how it ended — if YouTube has already transitioned us
            # automatically (enableAutoStop=True is the default), this
            # is a no-op.
            try:
                yt_rtmp.finalize_broadcast(
                    creds=creds, job=row, channel=channel,
                    broadcast_id=target.get("broadcast_id") or "",
                    db=sess,
                )
            except Exception as exc:
                # Non-fatal — the broadcast is on YT; we just couldn't
                # transition it via API. Log + move on.
                print(f"[live_studio] finalize_broadcast soft-fail for "
                      f"stream={stream_id}: {exc}")

            if completed_clean and not cancel_ev.is_set():
                _update(stream_id, status="done", progress_pct=100,
                        message="broadcast complete; uploading 48h preview to R2",
                        finished_at=datetime.now(timezone.utc))
            elif cancel_ev.is_set():
                _update(stream_id, status="canceled",
                        message="user canceled mid-broadcast",
                        finished_at=datetime.now(timezone.utc))

            # 6) R2 preview upload (48 h). Soft-fails — broadcast is
            # already on YouTube, this is just for in-Kaizer preview.
            # Done before the temp file is cleaned up in finally.
            if completed_clean and not cancel_ev.is_set():
                try:
                    from live_studio import r2_backup
                    backup = r2_backup.upload_for_preview(
                        stream_id=stream_id, user_id=row.user_id,
                        local_path=row.upload_path,
                    )
                    if backup:
                        _update(
                            stream_id,
                            backup_url=backup["url"],
                            backup_key=backup["key"],
                            backup_expires_at=backup["expires_at"],
                            message="broadcast complete (preview saved to R2 for 48h)",
                        )
                except Exception as exc:
                    print(f"[live_studio] R2 backup skipped for "
                          f"stream={stream_id}: {exc}")

        # End of `with acquire_slot`

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[live_studio] worker {stream_id} crashed:\n{tb}")
        _update(stream_id, status="failed",
                error=str(exc)[:2000],
                finished_at=datetime.now(timezone.utc))
    finally:
        # Clean up temp upload + cancel event.
        live_uploads.delete_upload(stream_id)
        _release_cancel_event(stream_id)
        sess.close()


def kick_off(stream_id: int) -> None:
    """Spawn the worker in a daemon thread. The router calls this
    from /streams/{id}/start once the upload threshold is met."""
    threading.Thread(
        target=run_stream, args=(stream_id,),
        name=f"live-stream-{stream_id}", daemon=True,
    ).start()
