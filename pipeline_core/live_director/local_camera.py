"""
kaizer.pipeline.live_director.local_camera
============================================
OpenCV-based ingest for locally-attached cameras (laptop webcam, USB cam,
IP cam via RTSP URL). Reads BGR frames straight from the camera + writes
them into the CameraRingBuffer, where the existing Phase 6.2 analyzers
pick them up.

Why this exists
---------------
The phone-browser → WebSocket → MediaRecorder → ffmpeg-stdin path
(WebRTCIngestWorker) proved fragile across browsers: codec negotiation,
matroska fragment handling, and stdin piping on Windows all introduce
different failure modes. For a predictable test/demo flow (and for
customers who bring a pro camera + USB capture card), reading a local
camera directly is strictly simpler and more robust.

Interface parity with WebRTCIngestWorker
----------------------------------------
Same start/stop/stats shape — so the /start endpoint can swap in either
worker type depending on the camera's role_hints without the rest of
the pipeline caring.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional, Union

from pipeline_core.live_director.ring_buffer import CameraRingBuffer
from pipeline_core.live_director.signal_bus import SignalBus

logger = logging.getLogger("kaizer.pipeline.live_director.local_camera")


@dataclass
class LocalCameraConfig:
    """Tuning knobs for a single local-camera ingest worker."""
    # Source: int device index for webcam, str for RTSP / HTTP URL
    source: Union[int, str] = 0
    analyzer_width:  int = 320
    analyzer_height: int = 240
    target_fps:      int = 15
    # How many times to retry opening the source before giving up
    open_retries: int = 3
    open_retry_delay_s: float = 1.0


@dataclass
class LocalCameraStats:
    running:            bool = False
    source:             str  = ""
    frames_read:        int  = 0
    frames_failed:      int  = 0
    last_error:         str  = ""
    started_at:         float = 0.0


class LocalCameraWorker:
    """Reads a local camera / RTSP URL + pushes BGR frames to the ring.

    Usage:
        worker = LocalCameraWorker(
            event_id=1, cam_id="cam_laptop",
            ring=ring, bus=bus,
            config=LocalCameraConfig(source=0),
        )
        await worker.start()
        ...
        await worker.stop()
    """

    def __init__(
        self,
        *,
        event_id: int,
        cam_id: str,
        ring: CameraRingBuffer,
        bus: Optional[SignalBus] = None,
        config: Optional[LocalCameraConfig] = None,
    ) -> None:
        self.event_id = event_id
        self.cam_id   = cam_id
        self.ring     = ring
        self.bus      = bus
        self.config   = config or LocalCameraConfig()
        self._task: Optional[asyncio.Task] = None
        self._running: bool = False
        self._event_start_t: float = 0.0
        self._stats = LocalCameraStats(source=str(self.config.source))

    @property
    def is_running(self) -> bool:
        return self._running

    def stats(self) -> dict:
        uptime = (time.monotonic() - self._event_start_t) if self._event_start_t else 0.0
        return {
            "cam_id":        self.cam_id,
            "running":       self._running,
            "source":        self._stats.source,
            "frames_read":   self._stats.frames_read,
            "frames_failed": self._stats.frames_failed,
            "uptime_s":      round(uptime, 2),
            "last_error":    self._stats.last_error,
            # Mirror the webrtc-worker schema so the Debug panel's CamCard
            # can render it without branching on worker type.
            "chunks_in":            0,
            "chunks_dropped":       0,
            "bytes_in":             0,
            "frames_decoded":       self._stats.frames_read,
            "audio_packets_decoded": 0,
            "ffmpeg_restarts_video": 0,
            "ffmpeg_restarts_audio": 0,
            "queue_depth":    0,
            "queue_capacity": 0,
        }

    async def start(self) -> None:
        if self._running:
            logger.warning("[%s] start() called but already running", self.cam_id)
            return
        self._running = True
        self._event_start_t = time.monotonic()
        self._stats.started_at = self._event_start_t
        logger.info(
            "[%s] LocalCameraWorker starting source=%s target=%dx%d@%dfps",
            self.cam_id, self.config.source,
            self.config.analyzer_width, self.config.analyzer_height,
            self.config.target_fps,
        )
        self._task = asyncio.create_task(
            self._capture_loop(), name=f"local-cam-{self.cam_id}",
        )

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._task = None
        logger.info("[%s] LocalCameraWorker stopped", self.cam_id)

    async def _capture_loop(self) -> None:
        """Blocking cv2 reads run in a thread executor so we never stall
        the event loop. Handles open-retries + auto-reconnect on read fail.
        """
        import cv2  # lazy
        import numpy as np

        loop = asyncio.get_event_loop()
        frame_interval = 1.0 / max(1, self.config.target_fps)
        src = self.config.source
        cap = None

        # ── Open (with retries) ───────────────────────────────────────────
        def _open():
            # CAP_DSHOW on Windows avoids a 2-3s MSMF cold-start.
            backend = cv2.CAP_DSHOW if isinstance(src, int) else cv2.CAP_ANY
            c = cv2.VideoCapture(src, backend)
            if not c.isOpened():
                c.release()
                return None
            try:
                c.set(cv2.CAP_PROP_FRAME_WIDTH,  self.config.analyzer_width)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.analyzer_height)
                c.set(cv2.CAP_PROP_FPS,          self.config.target_fps)
            except Exception:
                pass
            return c

        for attempt in range(self.config.open_retries):
            try:
                cap = await loop.run_in_executor(None, _open)
            except Exception as exc:
                self._stats.last_error = f"open failed: {exc}"
            if cap is not None:
                break
            await asyncio.sleep(self.config.open_retry_delay_s)

        if cap is None:
            self._stats.last_error = f"could not open source {src!r}"
            logger.error("[%s] %s", self.cam_id, self._stats.last_error)
            self._running = False
            return

        logger.info("[%s] camera opened source=%s", self.cam_id, src)

        try:
            while self._running:
                tick_start = time.monotonic()
                try:
                    ok, frame = await loop.run_in_executor(None, cap.read)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._stats.last_error = f"read error: {exc}"
                    self._stats.frames_failed += 1
                    await asyncio.sleep(0.1)
                    continue

                if not ok or frame is None:
                    self._stats.frames_failed += 1
                    self._stats.last_error = "read returned no frame"
                    # Don't spin: a failed read usually means a transient USB hiccup
                    await asyncio.sleep(0.05)
                    continue

                # Analyzers operate on a small frame — resize here so the
                # ring stores only what's needed.
                try:
                    if frame.shape[0] != self.config.analyzer_height or frame.shape[1] != self.config.analyzer_width:
                        frame = cv2.resize(
                            frame,
                            (self.config.analyzer_width, self.config.analyzer_height),
                            interpolation=cv2.INTER_AREA,
                        )
                except Exception:
                    pass

                t = time.monotonic() - self._event_start_t
                try:
                    await self.ring.push_video(t, frame)
                    self._stats.frames_read += 1
                    # Clear last_error on successful read to avoid stale data
                    if self._stats.last_error.startswith("read"):
                        self._stats.last_error = ""
                except Exception as exc:
                    self._stats.last_error = f"ring push failed: {exc}"

                # Pace to target fps (cap.read usually already paces but be safe)
                elapsed = time.monotonic() - tick_start
                sleep_for = frame_interval - elapsed
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)

        except asyncio.CancelledError:
            logger.debug("[%s] capture loop cancelled", self.cam_id)
            raise
        except Exception as exc:
            self._stats.last_error = f"loop crashed: {exc}"
            logger.error("[%s] capture loop crashed: %s", self.cam_id, exc)
        finally:
            try:
                cap.release()
            except Exception:
                pass
