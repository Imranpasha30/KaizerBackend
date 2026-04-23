"""Motion analyzer: grayscale abs-diff magnitude between consecutive frames."""
from __future__ import annotations

from typing import Optional

import numpy as np

from pipeline_core.live_director.analyzers.base import Analyzer
from pipeline_core.live_director.signals import SignalFrame
from pipeline_core.live_director.analyzers import ANALYZER_FRAME_DOWNSAMPLE


class MotionAnalyzer(Analyzer):
    name = "motion"

    def __init__(self, config, ring, bus) -> None:
        super().__init__(config, ring, bus)
        self._downsample = ANALYZER_FRAME_DOWNSAMPLE
        self._prev_gray: Optional[np.ndarray] = None

    async def analyze(self) -> Optional[SignalFrame]:
        try:
            frames = await self.ring.latest_video(count=2)
        except Exception as exc:
            self._log.debug("motion: latest_video failed: %s", exc)
            return None
        if not frames or len(frames) < 1:
            return None

        import cv2  # type: ignore

        t, frame = frames[-1]
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        new_w = max(64, int(w * self._downsample))
        new_h = max(64, int(h * self._downsample))
        small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        motion_mag = 0.0
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            diff = cv2.absdiff(self._prev_gray, gray)
            motion_mag = float(np.mean(diff)) / 255.0

        # Cache current frame for next tick. Small (~76 KB at 360×640×1).
        self._prev_gray = gray

        return SignalFrame(
            cam_id=self.config.cam_id,
            t=t,
            motion_mag=motion_mag,
        )
