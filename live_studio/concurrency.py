"""Global concurrency guard for Live Studio streams.

Why a single global cap:
  - Each concurrent ffmpeg push at 1080p ~ 4-8 Mbps uplink + 1 CPU
    core. Even a strong home line + RTX 5060 max out around 8-10
    parallel streams before YouTube starts dropping the worst ones.
  - YouTube has its own rate-limiting on liveBroadcasts.insert; 5-10
    new broadcasts/min is the safe rate.

The semaphore is process-wide. On a multi-worker uvicorn setup this
would need Redis to coordinate; current Kaizer deploy is single-
worker so an in-process semaphore is enough.
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Iterator


# Env override so cloud workers can bump it without code change.
MAX_CONCURRENT = max(1, int(
    (os.environ.get("KAIZER_LIVE_STUDIO_CONCURRENCY") or "8").strip()
))


_GLOBAL_SLOT = threading.BoundedSemaphore(MAX_CONCURRENT)


@contextmanager
def acquire_slot(*, blocking: bool = True, timeout_s: float = 600) -> Iterator[bool]:
    """Acquire one global stream slot.

    Use as::

        with acquire_slot() as got:
            if not got:
                state.update(jid, status="error", message="server busy")
                return
            run_ffmpeg(...)

    ``blocking=True`` waits up to ``timeout_s`` seconds for a slot.
    ``blocking=False`` returns immediately (False if no slot).
    """
    got = _GLOBAL_SLOT.acquire(blocking=blocking, timeout=timeout_s)
    try:
        yield got
    finally:
        if got:
            try:
                _GLOBAL_SLOT.release()
            except ValueError:
                pass


def stats() -> dict:
    """Snapshot for debug / admin UI."""
    return {
        "max":     MAX_CONCURRENT,
        # ``_value`` is implementation-defined but stable across CPython.
        "free":    getattr(_GLOBAL_SLOT, "_value", None),
    }
