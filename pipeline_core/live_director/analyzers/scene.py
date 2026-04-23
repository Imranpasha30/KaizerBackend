"""Scene analyzer: heuristic classifier (stage / crowd / closeup / wide / graphic)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from pipeline_core.live_director.analyzers.base import Analyzer
from pipeline_core.live_director.signals import SignalFrame
from pipeline_core.live_director.analyzers import ANALYZER_FRAME_DOWNSAMPLE


# Heuristic thresholds (tuned later per event type via vertical packs)
_CLOSEUP_FACE_SIZE = 0.08
_STAGE_FACE_SIZE = 0.015
_CROWD_FACE_COUNT = 3
_STAGE_DIM_BRIGHTNESS = 0.18
_STAGE_SAT_HIGH = 0.4
_GRAPHIC_SAT = 0.55
_GRAPHIC_BRIGHT = 0.4


class SceneAnalyzer(Analyzer):
    name = "scene"

    def __init__(self, config, ring, bus) -> None:
        super().__init__(config, ring, bus)
        self._downsample = ANALYZER_FRAME_DOWNSAMPLE
        self._cascade = None

    def _ensure_cascade(self):
        if self._cascade is None:
            import cv2  # type: ignore
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(cascade_path)

    async def analyze(self) -> Optional[SignalFrame]:
        try:
            frames = await self.ring.latest_video(count=1)
        except Exception as exc:
            self._log.debug("scene: latest_video failed: %s", exc)
            return None
        if not frames:
            return None
        t, frame = frames[0]
        if frame is None or frame.size == 0:
            return None

        import cv2  # type: ignore
        self._ensure_cascade()

        h, w = frame.shape[:2]
        new_w = max(64, int(w * self._downsample))
        new_h = max(64, int(h * self._downsample))
        small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Brightness (grayscale mean / 255)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray)) / 255.0

        # Saturation (HSV S channel)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        saturation = float(np.mean(hsv[:, :, 1])) / 255.0

        # Face detection on same downsample
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=4, minSize=(20, 20)
        )
        face_count = len(faces)
        max_face_size_norm = 0.0
        if face_count > 0:
            areas = [fw * fh for (_, _, fw, fh) in faces]
            max_face_size_norm = max(areas) / float(new_w * new_h)

        # Classification
        if face_count >= _CROWD_FACE_COUNT:
            scene = "crowd"
        elif face_count >= 1 and max_face_size_norm > _CLOSEUP_FACE_SIZE:
            scene = "closeup"
        elif face_count >= 1 and max_face_size_norm >= _STAGE_FACE_SIZE:
            scene = "stage"
        elif brightness < _STAGE_DIM_BRIGHTNESS and saturation > _STAGE_SAT_HIGH:
            scene = "stage"
        elif saturation > _GRAPHIC_SAT and brightness > _GRAPHIC_BRIGHT:
            scene = "graphic"
        else:
            scene = "wide"

        return SignalFrame(
            cam_id=self.config.cam_id,
            t=t,
            scene=scene,
        )
