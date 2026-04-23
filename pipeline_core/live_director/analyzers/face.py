"""Face analyzer: cv2 Haar cascade + largest-bbox size normalisation."""
from __future__ import annotations

from typing import Optional

import numpy as np

from pipeline_core.live_director.analyzers.base import Analyzer
from pipeline_core.live_director.signals import SignalFrame
from pipeline_core.live_director.analyzers import ANALYZER_FRAME_DOWNSAMPLE


class FaceAnalyzer(Analyzer):
    name = "face"

    def __init__(self, config, ring, bus) -> None:
        super().__init__(config, ring, bus)
        self._cascade = None
        self._downsample = ANALYZER_FRAME_DOWNSAMPLE

    def _ensure_cascade(self):
        if self._cascade is None:
            import cv2  # type: ignore
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(cascade_path)
            if self._cascade.empty():
                raise RuntimeError(f"Haar cascade failed to load: {cascade_path}")

    async def analyze(self) -> Optional[SignalFrame]:
        try:
            frames = await self.ring.latest_video(count=1)
        except Exception as exc:
            self._log.debug("face: latest_video failed: %s", exc)
            return None
        if not frames:
            return None
        t, frame = frames[0]
        if frame is None or frame.size == 0:
            return None

        import cv2  # type: ignore
        self._ensure_cascade()

        # Downsample once
        h, w = frame.shape[:2]
        new_w = max(64, int(w * self._downsample))
        new_h = max(64, int(h * self._downsample))
        small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=4, minSize=(24, 24)
        )

        face_present = False
        face_size_norm = 0.0
        if len(faces) > 0:
            face_present = True
            areas = [fw * fh for (_, _, fw, fh) in faces]
            i_max = int(np.argmax(areas))
            _, _, fw, fh = faces[i_max]
            face_size_norm = float(fw * fh) / float(new_w * new_h)

        # Return minimal SignalFrame with face fields populated.
        return SignalFrame(
            cam_id=self.config.cam_id,
            t=t,
            face_present=face_present,
            face_size_norm=face_size_norm,
            face_identity=None,
        )
