"""RTMP-live upload agent — autonomous lifecycle for a single job.

The "parallel agent" the operator asked for: hands a rendered clip to
YouTube via RTMPS instead of ``videos.insert``, surviving the entire
real-time push and reporting progress along the way.

The agent is **stateless across calls** — each invocation processes
exactly one ``UploadJob`` row. Parallelism is handled by the existing
upload-worker's thread pool (``KAIZER_WORKER_CONCURRENCY``), which
gives us "N concurrent RTMP pushes" for free without standing up a
second worker tier.

Lifecycle per job
-----------------
::

    queued  →  uploading  →  streaming  →  finalizing  →  done
        ↓             ↓            ↓              ↓
       fail-on-any-step (no orphan broadcasts)

Each transition is persisted to the DB BEFORE the next external call,
so a crash mid-push leaves a recoverable record (the broadcast on
YouTube's side will auto-stop because cdcontentDetails.enableAutoStop
is True, and our reconciler picks up the orphan on the next run).
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Optional

import models
from sqlalchemy.orm import Session

from youtube import oauth, uploader, rtmp_provider, rtmp_pusher

# Optional ffprobe for accurate duration. Pipeline already stores
# duration on Clip; we use that as the source of truth and only fall
# back to probing the file when missing.
try:
    from pipeline_core.hw_accel import probe_duration_seconds  # type: ignore
except Exception:
    probe_duration_seconds = None


# ─── Helpers ─────────────────────────────────────────────────────────

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _clip_duration_seconds(clip: "models.Clip", local_path: str) -> float:
    """Best-effort duration in seconds. Falls back to ffprobe if the
    Clip row doesn't carry a duration (older rows or beta renders)."""
    try:
        d = float(getattr(clip, "duration", 0) or 0)
        if d > 0:
            return d
    except (TypeError, ValueError):
        pass
    if probe_duration_seconds is not None and local_path:
        try:
            return float(probe_duration_seconds(local_path) or 0)
        except Exception:
            pass
    # Last-resort default — won't pass the 95% threshold but prevents
    # an immediate divide-by-zero.
    return 60.0


# ─── Main entry point — called from worker._process() ────────────────

def process_via_rtmp(
    db: Session,
    job: "models.UploadJob",
    clip: "models.Clip",
    clip_path: str,
    *,
    append_log,
    on_finish,
    on_fail,
) -> None:
    """Run the full RTMP lifecycle for one ``UploadJob``.

    Parameters
    ----------
    db, job, clip, clip_path
        Standard worker inputs. ``clip_path`` is the resolved local
        ``.mp4`` (beta-render-aware).
    append_log(job, msg)
        Reuse the worker's row-logger so all output lands in the same
        place the admin Logs / Job Detail page reads from.
    on_finish(video_id)
        Called once after a clean push + finalize. The worker uses
        this to flip ``UploadJob.status = 'completed'`` and stash the
        final ``video_id`` / ``video_url``.
    on_fail(message)
        Called on terminal failure. Worker flips status to 'failed'.
        Transient failures (TransientRtmpError, network blips) get
        re-raised so the worker's retry policy kicks in.
    """
    # ── 0) Resolve OAuth + destination channel ──────────────────────
    if not job.channel_id:
        on_fail("RTMP: no destination channel bound to job")
        return
    try:
        creds = oauth.get_credentials(db, job.channel_id)
    except oauth.OAuthError as e:
        on_fail(f"RTMP: OAuth refresh failed — {e}")
        return

    channel = db.query(models.Channel).filter(
        models.Channel.id == job.channel_id
    ).first()
    if channel is None:
        on_fail("RTMP: channel row missing — re-link the channel")
        return

    title = (job.title or clip.filename or "Bulletin")[:100]
    description = job.description or ""
    privacy = (job.privacy_status or "private").lower()
    duration_s = _clip_duration_seconds(clip, clip_path)

    append_log(job,
        f"[rtmp-agent] route=native_rtmp duration={duration_s:.1f}s "
        f"channel_id={job.channel_id} privacy={privacy}")
    db.commit()

    # ── 1) Mint broadcast + stream + bind ───────────────────────────
    # 150 quota units total — logged to admin Usage automatically.
    try:
        target = rtmp_provider.obtain_rtmp_target(
            creds,
            job=job,
            channel=channel,
            title=title,
            description=description,
            privacy_status=privacy,
            db=db,
        )
    except rtmp_provider.TransientRtmpError:
        # Let the worker's outer retry policy handle this — don't burn
        # an attempt on a 503.
        raise
    except rtmp_provider.RtmpProviderError as e:
        on_fail(f"RTMP: provider mint failed — {e}")
        return

    broadcast_id = target["broadcast_id"]
    stream_id    = target["stream_id"]
    ingest_url   = target["ingest_url"]
    stream_key   = target["stream_key"]
    video_id     = target["video_id"]

    # Persist the broadcast id RIGHT NOW so we can clean it up if we
    # die after this point. Use the existing video_id column — admin
    # Usage dashboard already correlates calls by it.
    try:
        job.video_id = video_id
        db.commit()
    except Exception:
        db.rollback()

    append_log(job,
        f"[rtmp-agent] minted broadcast={broadcast_id} stream={stream_id} "
        f"→ pushing to {ingest_url}")
    db.commit()

    # ── 2) Real-time RTMP push (ffmpeg) ─────────────────────────────
    cancel = threading.Event()

    # Make sure bytes_total reflects file size (used by progress UI).
    try:
        size = os.path.getsize(clip_path)
        job.bytes_total = int(size)
        db.commit()
    except OSError:
        size = 0

    def _progress(pushed_s: float, total_s: float) -> None:
        # Translate "seconds pushed" → "bytes uploaded" proportionally
        # so the existing progress UI (which tracks bytes) keeps
        # working without a new column.
        if total_s <= 0 or size <= 0:
            return
        bytes_done = int(size * min(1.0, pushed_s / total_s))
        try:
            job.bytes_uploaded = bytes_done
            db.commit()
        except Exception:
            db.rollback()

    try:
        result = rtmp_pusher.push_to_rtmp(
            input_path=clip_path,
            ingest_url=ingest_url,
            stream_key=stream_key,
            expected_duration_s=duration_s,
            progress_cb=_progress,
            cancel_event=cancel,
            log_prefix=f"[rtmp-agent job={job.id}]",
        )
    except rtmp_pusher.PushFailed as e:
        append_log(job, f"[rtmp-agent] push failed — {e}")
        db.commit()
        # Clean up the (now-orphaned) broadcast so we don't leave junk
        # on the channel. Best-effort; never raises.
        _best_effort_finalize(creds, job, channel, broadcast_id, db,
                              note="cleanup after push failure")
        on_fail(f"RTMP push failed: {str(e)[:300]}")
        return

    append_log(job,
        f"[rtmp-agent] push complete — {result['seconds_pushed']:.1f}s sent "
        f"(target {duration_s:.1f}s). Finalising broadcast.")
    db.commit()

    # ── 3) Transition to complete + thumbnail ───────────────────────
    try:
        rtmp_provider.finalize_broadcast(
            creds,
            job=job,
            channel=channel,
            broadcast_id=broadcast_id,
            thumbnail_path=getattr(clip, "thumb_path", None),
            db=db,
        )
    except rtmp_provider.RtmpProviderError as e:
        # The PUSH succeeded — the video IS on YouTube — but the
        # transition call failed. We've still produced output; record
        # as completed-with-warning rather than failed.
        append_log(job,
            f"[rtmp-agent][warn] finalize call failed but video was pushed: {e}")
        db.commit()

    on_finish(video_id)


# ─── Internal: cleanup helper ────────────────────────────────────────

def _best_effort_finalize(creds, job, channel, broadcast_id, db, *, note: str) -> None:
    """When something goes wrong mid-flight, try to release the
    broadcast cleanly. Failures here are logged but never raised —
    cleanup must never mask the original error."""
    try:
        rtmp_provider.finalize_broadcast(
            creds,
            job=job,
            channel=channel,
            broadcast_id=broadcast_id,
            thumbnail_path=None,
            db=db,
        )
        print(f"[rtmp-agent] {note}: broadcast {broadcast_id} transitioned to complete")
    except Exception as exc:
        print(f"[rtmp-agent] {note}: finalize failed (broadcast {broadcast_id}): {exc}")


# ─── Startup reconciler: clean up orphan broadcasts ─────────────────
#
# Called once from main.py on backend boot. If a previous process died
# mid-stream, the broadcast on YouTube's side will auto-stop (because
# ``enableAutoStop=True``) and we'll have an UploadJob row stuck in
# 'uploading' with a video_id already set. The reconciler:
#
#   1. Finds those rows (provider=native_rtmp, status=uploading,
#      video_id non-empty, last update older than a safe window).
#   2. For each: re-uses OAuth creds, calls liveBroadcasts.list to
#      check the actual broadcast state on YouTube's side.
#   3. If it's 'complete' (auto-stop fired) → flip job to 'done' with
#      the existing video_id. Quota-free recovery.
#   4. If it's still 'live' but no encoder pushing → call transition
#      to 'complete' so it doesn't sit consuming a live-slot.
#   5. If the broadcast no longer exists or is in a terminal-failure
#      state → flip job to 'failed' with a clear reason.
#
# The reconciler is silent (no errors raised) when there's nothing to
# do. Disabled entirely when ``KAIZER_NATIVE_RTMP_ENABLED`` is off.

def reconcile_orphan_broadcasts() -> None:
    """Run once on backend startup. Idempotent; safe to call repeatedly."""
    if os.environ.get("KAIZER_NATIVE_RTMP_ENABLED", "false").lower() not in (
        "1", "true", "yes", "on"
    ):
        return

    from database import SessionLocal
    from datetime import timedelta

    db = SessionLocal()
    try:
        # Only look at rows that were mid-stream long enough that a
        # live encoder couldn't still be pushing. 10 minutes is safe:
        # any push that takes longer would have updated bytes_uploaded
        # which auto-bumps updated_at on commit.
        cutoff = _now_utc() - timedelta(minutes=10)
        orphans = (
            db.query(models.UploadJob)
              .filter(models.UploadJob.upload_provider == "native_rtmp")
              .filter(models.UploadJob.status == "uploading")
              .filter(models.UploadJob.video_id != "")
              .filter(models.UploadJob.updated_at < cutoff)
              .all()
        )
        if not orphans:
            return
        print(f"[rtmp-agent] reconciling {len(orphans)} orphan RTMP broadcast(s) "
              f"from a previous run")

        for job in orphans:
            try:
                _reconcile_one(db, job)
            except Exception as exc:
                # Never let one bad row stop the sweep.
                print(f"[rtmp-agent][warn] reconcile job={job.id} failed: {exc}")
    finally:
        db.close()


def _reconcile_one(db: Session, job: "models.UploadJob") -> None:
    """Reconcile a single orphan job."""
    if not job.channel_id or not job.video_id:
        return

    try:
        creds = oauth.get_credentials(db, job.channel_id)
    except oauth.OAuthError as exc:
        print(f"[rtmp-agent] reconcile job={job.id}: OAuth failed ({exc}) "
              f"— marking failed")
        job.status = "failed"
        job.last_error = f"reconcile: OAuth refresh failed ({exc})"
        db.commit()
        return

    channel = db.query(models.Channel).filter(
        models.Channel.id == job.channel_id
    ).first()
    broadcast_id = job.video_id
    yt = rtmp_provider._yt(creds)   # safe internal use

    # Check current state of the broadcast on YouTube's side.
    try:
        resp = yt.liveBroadcasts().list(
            part="status,id", id=broadcast_id,
        ).execute()
    except Exception as exc:
        print(f"[rtmp-agent] reconcile job={job.id}: list broadcast failed ({exc})")
        return

    items = (resp or {}).get("items") or []
    if not items:
        # Broadcast was deleted (or never existed). Mark job failed.
        print(f"[rtmp-agent] reconcile job={job.id}: broadcast {broadcast_id} "
              f"no longer exists on YouTube — marking failed")
        job.status = "failed"
        job.last_error = "reconcile: broadcast disappeared from YouTube (deleted?)"
        db.commit()
        return

    lifecycle = (items[0].get("status") or {}).get("lifeCycleStatus") or ""
    print(f"[rtmp-agent] reconcile job={job.id}: broadcast {broadcast_id} "
          f"lifecycle={lifecycle!r}")

    if lifecycle in ("complete", "ready"):
        # Auto-stop already finalised the broadcast. The video is on
        # the channel and reachable at /watch?v=broadcast_id. Just
        # update our job row.
        job.status = "done"
        job.last_error = ""
        db.commit()
        print(f"[rtmp-agent] reconcile job={job.id}: marked done — "
              f"https://www.youtube.com/watch?v={broadcast_id}")
        return

    if lifecycle in ("live", "liveStarting", "testing"):
        # Stream is still showing as live but nothing's pushing. Force
        # transition to complete so the broadcast doesn't hang there
        # consuming a live-slot.
        try:
            _best_effort_finalize(creds, job, channel, broadcast_id, db,
                                  note=f"reconcile job={job.id}")
            job.status = "done"
            job.last_error = ""
            db.commit()
            print(f"[rtmp-agent] reconcile job={job.id}: force-transitioned "
                  f"to complete, marked done")
        except Exception as exc:
            print(f"[rtmp-agent] reconcile job={job.id}: force-transition "
                  f"failed ({exc})")
        return

    # Anything else (revoked, abandoned) → mark failed; user can re-publish.
    print(f"[rtmp-agent] reconcile job={job.id}: lifecycle={lifecycle!r} not "
          f"recoverable — marking failed")
    job.status = "failed"
    job.last_error = f"reconcile: broadcast in non-recoverable state {lifecycle!r}"
    db.commit()
