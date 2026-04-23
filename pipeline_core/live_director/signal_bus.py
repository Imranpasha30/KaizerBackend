"""
kaizer.pipeline.live_director.signal_bus
==========================================
Asyncio fan-in for ``SignalFrame`` objects produced by all analyzers across
all cameras.

Architecture
------------
- A single ``asyncio.Queue`` receives frames from N analyzer coroutines
  (many producers).
- A running dict ``_latest`` maps ``cam_id → SignalFrame`` for O(1) director
  access to the most-recent signal per camera.
- ``subscribe()`` is an async generator that yields frames in arrival order.
  Multiple subscribers each get their own view (implemented via per-subscriber
  secondary queues so the bus is a proper fan-out, not a competing consumer).

Overflow handling
-----------------
When the queue reaches *maxsize*, ``publish()`` uses ``put_nowait`` and
catches ``QueueFull`` to drop the oldest item first, then retries. This
ensures slow consumers never block the ingest pipeline.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Optional

from pipeline_core.live_director.signals import SignalFrame

logger = logging.getLogger("kaizer.pipeline.live_director.signal_bus")


class SignalBus:
    """Fan-in / fan-out asyncio bus for ``SignalFrame`` objects.

    Parameters
    ----------
    maxsize : Maximum queue depth before oldest frames are dropped (default 10 000).
    """

    def __init__(self, *, maxsize: int = 10_000) -> None:
        self._maxsize    = maxsize
        self._queue: asyncio.Queue[SignalFrame] = asyncio.Queue(maxsize=maxsize)
        self._latest: dict[str, SignalFrame] = {}
        self._lock       = asyncio.Lock()
        # Subscriber queues: each subscribe() call adds an entry here.
        self._subscribers: list[asyncio.Queue[Optional[SignalFrame]]] = []

    # ── Write API ────────────────────────────────────────────────────────────

    async def publish(self, frame: SignalFrame) -> None:
        """Publish a ``SignalFrame`` to all subscribers.

        If every subscriber queue is at capacity, the oldest item is dropped
        to make room — slow consumers shed load rather than blocking ingest.
        Updates ``_latest`` so ``latest_per_camera()`` always reflects current state.
        """
        async with self._lock:
            self._latest[frame.cam_id] = frame
            for sq in self._subscribers:
                if sq.full():
                    try:
                        sq.get_nowait()  # drop oldest
                    except asyncio.QueueEmpty:
                        pass
                try:
                    sq.put_nowait(frame)
                except asyncio.QueueFull:
                    logger.debug(
                        "SignalBus: subscriber queue full for cam=%s — frame dropped",
                        frame.cam_id,
                    )

    # ── Read API ─────────────────────────────────────────────────────────────

    async def subscribe(self) -> AsyncIterator[SignalFrame]:
        """Async generator that yields every published ``SignalFrame`` in order.

        Each call to ``subscribe()`` creates an independent view of the stream.
        The generator runs until the caller breaks / closes it — at that point
        the subscriber queue is removed.

        Usage::

            async for frame in bus.subscribe():
                process(frame)
        """
        sq: asyncio.Queue[Optional[SignalFrame]] = asyncio.Queue(maxsize=self._maxsize)
        async with self._lock:
            self._subscribers.append(sq)
        try:
            while True:
                frame = await sq.get()
                if frame is None:
                    break  # sentinel from close()
                yield frame
        finally:
            async with self._lock:
                try:
                    self._subscribers.remove(sq)
                except ValueError:
                    pass

    async def latest_per_camera(self) -> dict[str, SignalFrame]:
        """Return a shallow copy of the most-recent-frame-per-camera dict.

        O(1) per camera — no queue traversal. Returns an empty dict if no
        frames have been published yet.
        """
        async with self._lock:
            return dict(self._latest)

    async def close(self) -> None:
        """Send sentinel to all active subscribers so their generators exit cleanly."""
        async with self._lock:
            for sq in self._subscribers:
                try:
                    sq.put_nowait(None)
                except asyncio.QueueFull:
                    pass
