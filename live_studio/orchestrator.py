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

# Shared branding cache (Quick Live conveyor): channels with the SAME
# effective branding (watermark.brand_signature) broadcasting the SAME
# source share ONE stamped file instead of rendering N identical copies.
# Keyed by the shared file path; per-key locks serialize the first stamp,
# refcounts delay deletion until the last sharer's finally block.
_BRAND_STATE_LOCK = threading.Lock()
_BRAND_LOCKS: dict[str, threading.Lock] = {}
_BRAND_REFS: dict[str, int] = {}


def _brand_lock_for(path: str) -> threading.Lock:
    with _BRAND_STATE_LOCK:
        lk = _BRAND_LOCKS.get(path)
        if lk is None:
            lk = threading.Lock()
            _BRAND_LOCKS[path] = lk
        return lk


def _brand_ref(path: str) -> None:
    with _BRAND_STATE_LOCK:
        _BRAND_REFS[path] = _BRAND_REFS.get(path, 0) + 1


def _brand_unref_and_maybe_delete(path: str) -> None:
    """Drop one reference; delete the shared file when the last sharer
    leaves. A Windows delete of a file another ffmpeg still has open
    fails harmlessly (caught) — the next stamp regenerates it."""
    with _BRAND_STATE_LOCK:
        n = _BRAND_REFS.get(path, 1) - 1
        if n > 0:
            _BRAND_REFS[path] = n
            return
        _BRAND_REFS.pop(path, None)
        _BRAND_LOCKS.pop(path, None)
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


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
    block on whatever else is happening in the request thread.

    Also does two things for every terminal status, here rather than at the
    seven call sites so the eighth cannot forget:

    * LOGS the reason. The `error` was only ever written to Postgres, and a
      hosted deployment's logs come from stdout, so on production the reason
      for a failure was unreachable.

    * REPLACES A STALE `message`. The UI renders `message`, and a failure that
      passed only status+error left whatever step ran last still showing --
      which is how a failed row came to read "queued - waiting for an
      available broadcast slot" long after it had stopped queueing.
    """
    sess = SessionLocal()
    try:
        row = sess.query(models.LiveStream).get(stream_id)
        if not row:
            return

        # A failure that carries a reason but no message would otherwise leave
        # the previous step's message on screen, describing something that is
        # no longer true.
        if (fields.get("status") == "failed" and fields.get("error")
                and not fields.get("message")):
            fields["message"] = ("failed: "
                                 + " ".join(str(fields["error"]).split()))[:512]

        for k, v in fields.items():
            if hasattr(row, k):
                setattr(row, k, v)
        sess.commit()

        status = fields.get("status")
        if status in ("failed", "canceled", "done"):
            # ONE line per event: log collectors split on newlines, and a
            # reason spread over five lines is a reason nobody greps out.
            err = str(fields.get("error") or getattr(row, "error", "") or "")
            msg = str(fields.get("message") or getattr(row, "message", "") or "")
            print(
                f"[live_studio] STREAM {stream_id} -> {status}"
                f" | channel={getattr(row, 'channel_id', None)}"
                f" | batch={getattr(row, 'batch_id', None)}"
                f" | source={getattr(row, 'source_url', None) or 'upload'}"
                f" | broadcast={getattr(row, 'yt_broadcast_id', None)}"
                f" | msg={' '.join(msg.split())[:200]}"
                f" | err={' '.join(err.split())[:2000] or '(none recorded)'}",
                flush=True,
            )
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
    # Stamped branding temp file (Quick Live conveyor) — tracked out here
    # so the finally block can clean it up on ANY exit path.
    branded_tmp: Optional[str] = None

    try:
        row = sess.query(models.LiveStream).get(stream_id)
        if not row:
            return
        if row.status in ("done", "failed", "canceled"):
            _release_cancel_event(stream_id)
            return

        # 1) Concurrency slot. Queued workers may wait a long time — a
        # 24h broadcast occupying slot 1 means the 11th submission has
        # to wait the full 24h before it even starts. The cap is
        # KAIZER_LIVE_STUDIO_CONCURRENCY (default 10); the wait timeout
        # is KAIZER_LIVE_STUDIO_QUEUE_TIMEOUT_S (default 48h).
        _update(stream_id, status="queued",
                message="queued — waiting for an available broadcast slot")
        with concurrency.acquire_slot(timeout_s=concurrency.QUEUE_TIMEOUT_S) as got:
            if not got:
                _update(stream_id, status="failed",
                        error=f"timed out waiting for a broadcast slot "
                              f"(after {concurrency.QUEUE_TIMEOUT_S:.0f}s)")
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

            # 2b) Quick Live branding conveyor — stamp the channel's
            # logo/watermark onto the source BEFORE minting the broadcast
            # (so YouTube never sees a moment of unbranded footage).
            # Soft-fails: an unbranded live beats no live, and the reason
            # is surfaced in the row message. Skipped for passthrough URL
            # streams (no local file to stamp). The stamped temp file is
            # recorded on the row and deleted in the finally block.
            stream_input_path = row.upload_path
            if (getattr(row, "apply_branding", False)
                    and not (getattr(row, "source_url", "") or "").strip()):
                _update(stream_id, status="branding",
                        message="adding channel branding (logo/watermark)…")
                try:
                    import hashlib
                    from pipeline_v4 import watermark as _wm
                    _owner = sess.query(models.User).get(row.user_id)
                    _sig = _wm.brand_signature(
                        channel=channel, user=_owner, db=sess)
                    if _sig is None:
                        # Honest no-op: nothing configured to stamp. The
                        # UI asks BEFORE launch (has_branding on /channels);
                        # this covers settings changed mid-flight.
                        _append_message(
                            stream_id,
                            "branding not configured on this channel — "
                            "streaming the original (unbranded)")
                    else:
                        # Channels with identical effective branding AND the
                        # same source content share ONE stamped file.
                        # Hardlinked from-video sources share size+mtime, so
                        # (size, mtime_ns, signature) identifies the output.
                        _st = os.stat(row.upload_path)
                        _key = hashlib.sha1(
                            f"{_st.st_size}|{_st.st_mtime_ns}|{_sig}"
                            .encode("utf-8")).hexdigest()[:16]
                        _tmp_dir = os.path.dirname(
                            live_uploads.upload_path_for(row.id))
                        _shared = os.path.join(
                            _tmp_dir, f"brandshare-{_key}.mp4")
                        with _brand_lock_for(_shared):
                            if not os.path.isfile(_shared):
                                _stamped = _wm.stamp_for_channel(
                                    source_path=row.upload_path,
                                    channel=channel,
                                    user=_owner,
                                    db=sess,
                                    # NEVER the source's own dir — from-video
                                    # streams can point at a render-output
                                    # file; keep temp artifacts in the
                                    # live-studio upload dir.
                                    work_dir=_tmp_dir,
                                )
                                if (_stamped and _stamped != row.upload_path
                                        and os.path.isfile(_stamped)):
                                    os.replace(_stamped, _shared)
                            if os.path.isfile(_shared):
                                _brand_ref(_shared)
                                branded_tmp = _shared
                                stream_input_path = _shared
                                _update(stream_id, branded_path=_shared,
                                        message="channel branding applied")
                            else:
                                _append_message(
                                    stream_id,
                                    "branding produced no change — "
                                    "streaming the original")
                except Exception as exc:
                    _append_message(
                        stream_id,
                        f"branding failed — streaming the original "
                        f"({str(exc)[:200]})")
                if cancel_ev.is_set():
                    _update(stream_id, status="canceled",
                            message="canceled during branding",
                            finished_at=datetime.now(timezone.utc))
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
                    # Looping a copy-stream (-stream_loop -1 -c:v copy) produces
                    # a micro-gap at each loop boundary. With YouTube's default
                    # enableAutoStop=True those gaps make YouTube END the
                    # broadcast early (operator: "set 9h, closed before time").
                    # We own the duration via ffmpeg -t and finalize_broadcast
                    # at the end, so tell YouTube NOT to auto-stop.
                    enable_auto_stop=False,
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

            # Remember the channel's stream so the NEXT broadcast reuses it
            # (liveStreams.list, 1 unit) instead of minting another (50) and
            # abandoning the old one on the customer's channel. Without this
            # the reuse path above never has an id to find and the whole
            # feature is inert.
            _sid = target.get("stream_id") or ""
            if _sid and channel is not None and getattr(channel, "yt_stream_id", None) != _sid:
                try:
                    channel.yt_stream_id = _sid
                    sess.commit()
                    print(f"[live_studio] channel {channel.id} stream id saved: {_sid}",
                          flush=True)
                except Exception as _exc:
                    sess.rollback()
                    print(f"[live_studio] could not save stream id for channel "
                          f"{getattr(channel, 'id', '?')}: {_exc}", flush=True)

            # 3b) Apply the user-uploaded thumbnail, if any. Soft-fail —
            # YouTube will fall back to an auto-picked frame, and the
            # broadcast itself is already minted. Burns 50 quota units
            # per call; the helper resizes to ≤2 MB JPEG as needed.
            thumb_path = (getattr(row, "thumbnail_path", "") or "").strip()
            yt_video_id = target.get("video_id") or target.get("broadcast_id") or ""
            if thumb_path and yt_video_id and os.path.isfile(thumb_path):
                try:
                    from youtube import uploader as yt_uploader
                    yt_uploader.set_thumbnail(
                        creds=creds, video_id=yt_video_id,
                        thumb_path=thumb_path, job=row,
                    )
                    _append_message(stream_id, "thumbnail applied; ffmpeg starting")
                except Exception as exc:
                    print(f"[live_studio] set_thumbnail soft-fail for "
                          f"stream={stream_id} video={yt_video_id}: {exc}")

            # 4) ffmpeg push (blocks until done, cancel, or error).
            # Two modes based on the row's source:
            #   - source_url present → OBS-style passthrough pipe
            #     (yt-dlp | ffmpeg → YouTube). No looping, broadcast
            #     ends when the source video ends.
            #   - upload_path present → loop-on-disk mode (existing).
            #     Loops a finished file for the configured duration.
            src_url = (getattr(row, "source_url", "") or "").strip()
            try:
                if src_url:
                    _append_message(
                        stream_id,
                        "passthrough live (no loop, ends when source ends)",
                    )
                    streamer.push_passthrough(
                        source_url=src_url,
                        ingest_url=target["ingest_url"],
                        stream_key=target["stream_key"],
                        progress_cb=lambda pct: _update(
                            stream_id, progress_pct=int(pct),
                            message=f"streaming (passthrough)… {pct:.1f}%",
                        ),
                        cancel_event=cancel_ev,
                        extra_log_cb=None,
                    )
                else:
                    streamer.push_loop(
                        input_path=stream_input_path,
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
                # Race guard: if the user hit Stop right as ffmpeg exited
                # non-zero, honour the cancel — a user stop is NOT a failure.
                if cancel_ev.is_set():
                    _update(stream_id, status="canceled",
                            message="user canceled mid-broadcast",
                            finished_at=datetime.now(timezone.utc))
                else:
                    _update(stream_id, status="failed",
                            error=f"ffmpeg push failed: {exc}",
                            finished_at=datetime.now(timezone.utc))
                completed_clean = False
            except Exception as exc:
                if cancel_ev.is_set():
                    _update(stream_id, status="canceled",
                            message="user canceled mid-broadcast",
                            finished_at=datetime.now(timezone.utc))
                else:
                    _update(stream_id, status="failed",
                            error=f"unexpected: {exc}",
                            finished_at=datetime.now(timezone.utc))
                completed_clean = False

            # 5) Finalize broadcast (transition=complete) regardless of
            # how it ended — if YouTube has already transitioned us
            # NOT the default here: obtain_rtmp_target is called with
            # enable_auto_stop=False, because loop micro-gaps make
            # YouTube's auto-stop end the broadcast early. This
            # transition is therefore the ONLY closer for a Live Studio
            # broadcast -- and with a reusable per-channel stream, an
            # un-closed broadcast blocks the channel's next one.
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
            # SKIPPED for passthrough URL streams: there's no local
            # file to upload (yt-dlp piped straight to ffmpeg).
            # YouTube keeps the durable copy of the broadcast either way.
            if completed_clean and not cancel_ev.is_set() and not src_url:
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
        # Clean up temp upload + branding artifact + cancel event. The
        # delete only ever touches the live-studio temp dir — from-video
        # streams' original render files are never at these paths.
        # Shared branded files (brandshare-*) are refcounted: the file
        # survives until the LAST channel sharing it finishes.
        live_uploads.delete_upload(stream_id)
        if branded_tmp:
            _brand_unref_and_maybe_delete(branded_tmp)
        _release_cancel_event(stream_id)
        sess.close()


def kick_off(stream_id: int) -> None:
    """Spawn the worker in a daemon thread. The router calls this
    from /streams/{id}/start once the upload threshold is met."""
    threading.Thread(
        target=run_stream, args=(stream_id,),
        name=f"live-stream-{stream_id}", daemon=True,
    ).start()
