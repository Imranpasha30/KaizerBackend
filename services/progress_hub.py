"""Progress hub — shared DB tailers + WebSocket fan-out (Wave 3).

Why this shape: the V4 render runs in a SUBPROCESS (its log lines land
in the ``jobs`` row via its own DB connection) and the publish worker
may run in a DIFFERENT PROCESS entirely (standalone mode) — so a
callback-based pub/sub inside the web process cannot see either
reliably. The robust source of truth in every topology is the DB row.

The hub therefore manages **one tailer per topic**: the first WebSocket
subscriber starts a polling coroutine (1-2s interval); every further
subscriber for the same topic just attaches a queue. 50 viewers of one
job = ONE DB query per tick, not 50 — that's the fan-out win over the
legacy per-client 2s HTTP polling.

Backplane note: when multi-host web tiers arrive, this module is the
seam — replace the tailer with a Redis pub/sub consumer per topic and
nothing above (router) or below (DB) changes.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Awaitable, Callable, Optional

log = logging.getLogger("kaizer.progress_hub")

# A tail function: async generator that yields payload dicts forever
# (or until the topic is finished — yield a final dict with
# ``{"final": True}`` to close all subscribers).
TailFactory = Callable[[], AsyncIterator[dict]]

_QUEUE_MAX = 100  # per-subscriber buffer; slow clients drop oldest


class _Topic:
    def __init__(self, name: str) -> None:
        self.name = name
        self.subscribers: set[asyncio.Queue] = set()
        self.tail_task: Optional[asyncio.Task] = None


class ProgressHub:
    def __init__(self) -> None:
        self._topics: dict[str, _Topic] = {}
        self._lock = asyncio.Lock()

    async def subscribe(
        self,
        topic_name: str,
        tail_factory: TailFactory,
    ) -> tuple[asyncio.Queue, Callable[[], Awaitable[None]]]:
        """Attach to ``topic_name``; starts the shared tailer if this is
        the first subscriber. Returns (queue, detach) — caller MUST
        await detach() on disconnect (the last detach stops the tailer).
        """
        async with self._lock:
            topic = self._topics.get(topic_name)
            if topic is None:
                topic = _Topic(topic_name)
                self._topics[topic_name] = topic
            q: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAX)
            topic.subscribers.add(q)
            if topic.tail_task is None or topic.tail_task.done():
                topic.tail_task = asyncio.create_task(
                    self._run_tailer(topic, tail_factory),
                    name=f"hub-tail:{topic_name}",
                )

        async def detach() -> None:
            async with self._lock:
                topic.subscribers.discard(q)
                if not topic.subscribers:
                    if topic.tail_task is not None:
                        topic.tail_task.cancel()
                    self._topics.pop(topic_name, None)

        return q, detach

    async def _run_tailer(self, topic: _Topic, tail_factory: TailFactory) -> None:
        try:
            async for payload in tail_factory():
                self._fan_out(topic, payload)
                if payload.get("final"):
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("progress_hub: tailer for %s crashed", topic.name)
            self._fan_out(topic, {"type": "error",
                                  "error": "progress tail crashed", "final": True})
        finally:
            async with self._lock:
                # Wake subscribers so their reads return promptly.
                for q in list(topic.subscribers):
                    self._offer(q, {"type": "closed", "final": True})

    def _fan_out(self, topic: _Topic, payload: dict) -> None:
        for q in list(topic.subscribers):
            self._offer(q, payload)

    @staticmethod
    def _offer(q: asyncio.Queue, payload: dict) -> None:
        """Non-blocking put — a slow consumer drops its OLDEST update
        (progress frames are state snapshots; the newest always wins)."""
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            try:
                q.get_nowait()
            except Exception:
                pass
            try:
                q.put_nowait(payload)
            except Exception:
                pass

    def snapshot(self) -> dict:
        return {
            "topics": len(self._topics),
            "subscribers": sum(len(t.subscribers) for t in self._topics.values()),
            "names": list(self._topics)[:50],
        }


#: process-wide singleton — the WS router and admin snapshot share it.
hub = ProgressHub()

__all__ = ["hub", "ProgressHub"]
