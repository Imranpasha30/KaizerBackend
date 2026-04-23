"""Analyzer abstract base + AnalyzerConfig dataclass + shared run loop."""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from pipeline_core.live_director.ring_buffer import CameraRingBuffer
from pipeline_core.live_director.signal_bus import SignalBus
from pipeline_core.live_director.signals import SignalFrame

logger = logging.getLogger("kaizer.pipeline.live_director.analyzers.base")


@dataclass
class AnalyzerConfig:
    cam_id: str                    # which camera this analyzer works on
    interval_s: float = 0.3        # how often to evaluate
    enabled: bool = True


class Analyzer(ABC):
    """Base class: subclasses implement analyze() and set `name`."""
    name: str = "base"

    def __init__(
        self,
        config: AnalyzerConfig,
        ring: CameraRingBuffer,
        bus: SignalBus,
    ) -> None:
        self.config = config
        self.ring = ring
        self.bus = bus
        self._log = logging.getLogger(
            f"kaizer.pipeline.live_director.analyzers.{self.name}"
        )

    async def run(self) -> None:
        """Outer loop. Evaluates every interval_s. Never raises — logs and
        continues. Exits on asyncio.CancelledError."""
        while True:
            try:
                await asyncio.sleep(self.config.interval_s)
                if not self.config.enabled:
                    continue
                partial = await self.analyze()
                if partial is not None:
                    await self.bus.publish(partial)
            except asyncio.CancelledError:
                self._log.debug("[%s/%s] run() cancelled", self.config.cam_id, self.name)
                return
            except Exception as exc:
                self._log.error(
                    "[%s/%s] analyzer crashed: %s — continuing",
                    self.config.cam_id, self.name, exc,
                )

    @abstractmethod
    async def analyze(self) -> Optional[SignalFrame]:
        """Return a SignalFrame (cam_id + t + this analyzer's field(s) populated)
        or None when there's nothing meaningful to publish this tick."""
        ...

    def _now(self) -> float:
        return time.monotonic()
