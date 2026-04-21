"""APScheduler — daily corpus refresh + hourly analytics/thumb-swap + 2h trending radar.

All crons run on a single AsyncIOScheduler instance bound to Asia/Kolkata.
Idempotent: start() is a no-op if already running.
"""
from __future__ import annotations

import os
import traceback
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from database import SessionLocal
from learning import corpus


IST_TZ = "Asia/Kolkata"
CRON_HOUR, CRON_MINUTE = 4, 0

_scheduler: Optional[AsyncIOScheduler] = None


def _flag(name: str, default: bool) -> bool:
    """Read a KAIZER_ENABLE_* env flag; tolerates 'true/false/1/0/yes/no'."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _run_refresh_job() -> None:
    """Job body — opens its own DB session, swallows exceptions."""
    db = SessionLocal()
    try:
        summary = corpus.refresh_all_priority(db)
        refreshed = summary.get("refreshed") or []
        failed    = summary.get("failed") or []
        print(f"[corpus-cron] refreshed={len(refreshed)} failed={len(failed)}")
        if failed:
            for f in failed:
                print(f"[corpus-cron]   {f.get('channel')}: {f.get('error')}")
    except Exception:
        traceback.print_exc()
    finally:
        db.close()


def _run_analytics_poll() -> None:
    try:
        from analytics import poller
        summary = poller.poll_once()
        print(f"[analytics-cron] {summary}")
    except Exception:
        traceback.print_exc()


def _run_thumb_swap() -> None:
    try:
        from thumbnails_ab import swapper
        summary = swapper.run_once()
        print(f"[thumb-ab-cron] {summary}")
    except Exception:
        traceback.print_exc()


def _run_trending_radar() -> None:
    try:
        from trending import radar
        summary = radar.refresh_all()
        print(f"[trending-cron] {summary}")
    except Exception:
        traceback.print_exc()


def start() -> None:
    """Register background jobs. Each is individually gated by env flag.

    Default is OFF during dev so Gemini/YT-Data quota isn't silently burned.
    Flip individual flags to re-enable:
      KAIZER_ENABLE_SCHEDULERS=true      # master switch (default false)
      KAIZER_ENABLE_CORPUS=true          # daily at 04:00 IST  (Gemini text calls)
      KAIZER_ENABLE_ANALYTICS=true       # hourly              (YT Data API)
      KAIZER_ENABLE_THUMB_AB=true        # hourly              (Gemini text calls)
      KAIZER_ENABLE_TRENDING=true        # every 2h            (Gemini text per new video)
    """
    global _scheduler
    if _scheduler is not None:
        return

    master = _flag("KAIZER_ENABLE_SCHEDULERS", False)
    if not master:
        print("[scheduler] DISABLED — set KAIZER_ENABLE_SCHEDULERS=true to turn on. "
              "All background Gemini/YT-Data calls are suppressed to preserve quota.")
        return

    sched = AsyncIOScheduler(timezone=IST_TZ)
    enabled = []

    if _flag("KAIZER_ENABLE_CORPUS", True):
        sched.add_job(
            _run_refresh_job,
            CronTrigger(hour=CRON_HOUR, minute=CRON_MINUTE, timezone=IST_TZ),
            id="channel-corpus-refresh",
            replace_existing=True, misfire_grace_time=3600,
            coalesce=True, max_instances=1,
        )
        enabled.append(f"corpus daily {CRON_HOUR:02d}:{CRON_MINUTE:02d} IST")

    if _flag("KAIZER_ENABLE_ANALYTICS", True):
        sched.add_job(
            _run_analytics_poll,
            IntervalTrigger(hours=1),
            id="analytics-poll",
            replace_existing=True, misfire_grace_time=1800,
            coalesce=True, max_instances=1,
        )
        enabled.append("analytics hourly")

    if _flag("KAIZER_ENABLE_THUMB_AB", True):
        sched.add_job(
            _run_thumb_swap,
            IntervalTrigger(hours=1),
            id="thumbnail-ab-swap",
            replace_existing=True, misfire_grace_time=1800,
            coalesce=True, max_instances=1,
        )
        enabled.append("thumb-ab hourly")

    if _flag("KAIZER_ENABLE_TRENDING", True):
        sched.add_job(
            _run_trending_radar,
            IntervalTrigger(hours=2),
            id="trending-radar",
            replace_existing=True, misfire_grace_time=3600,
            coalesce=True, max_instances=1,
        )
        enabled.append("trending every 2h")

    sched.start()
    _scheduler = sched
    print(f"[scheduler] started: {', '.join(enabled) if enabled else 'NO JOBS (all disabled)'}")


def stop() -> None:
    global _scheduler
    if _scheduler is None:
        return
    try:
        _scheduler.shutdown(wait=False)
    except Exception:
        pass
    _scheduler = None
