"""
kaizer.pipeline.live_director.ingest
======================================
One asyncio task per camera.  Spawns two FFmpeg subprocesses per camera
(video + audio, separate pipes for simpler buffer management) that read
from ``rtmp://<host>:1935/live/<event_id>/<cam_id>`` and push raw frames /
PCM samples into a ``CameraRingBuffer``.

Design decisions
----------------
- Separate video and audio processes avoids muxer synchronisation complexity.
- Fixed-size stdout reads match the exact frame/chunk byte counts so no
  partial-frame logic is needed.
- On subprocess death the worker logs the error and restarts with exponential
  backoff (1 s → 2 s → 4 s … capped at 30 s) — it NEVER raises.
- ``asyncio.create_subprocess_exec`` is used throughout (no shell=True).
- numpy arrays are created with ``np.frombuffer`` + ``.copy()`` to avoid
  aliasing into the raw bytes object.

Video frame size : 1080 × 1920 × 3 = 6 220 800 bytes  (bgr24)
Audio chunk size : 16 000 × 2 × 0.32 = 10 240 bytes    (s16le mono, 320 ms)
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import numpy as np

from pipeline_core.live_director.signals import CameraConfig
from pipeline_core.live_director.ring_buffer import CameraRingBuffer
from pipeline_core.pipeline import FFMPEG_BIN

logger = logging.getLogger("kaizer.pipeline.live_director.ingest")

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_WIDTH  = 1080
_DEFAULT_HEIGHT = 1920
_DEFAULT_FPS    = 30
_DEFAULT_HOST   = "localhost"
_DEFAULT_PORT   = 1935

# 320 ms audio chunk at 16 kHz, mono int16
_AUDIO_SAMPLE_RATE   = 16_000
_AUDIO_CHUNK_MS      = 320
_AUDIO_CHUNK_SAMPLES = int(_AUDIO_SAMPLE_RATE * _AUDIO_CHUNK_MS / 1000)  # 5 120
_AUDIO_BYTES_PER_CHUNK = _AUDIO_CHUNK_SAMPLES * 2  # int16 = 2 bytes/sample = 10 240

_MAX_BACKOFF_S   = 30.0
_MIN_RESTART_S   = 1.0


# ─────────────────────────────────────────────────────────────────────────────


class IngestWorker:
    """Per-camera RTMP ingestion worker.

    Spawns one FFmpeg process for raw video and one for raw audio.  Both
    processes restart automatically on crash with exponential backoff.

    Parameters
    ----------
    event_id    : Live-event database ID (used in the RTMP path).
    camera      : Static camera configuration.
    ring        : The target ``CameraRingBuffer`` for this camera.
    rtmp_host   : RTMP server hostname (default ``localhost``).
    rtmp_port   : RTMP server port (default ``1935``).
    target_fps  : Target frame rate for the video subprocess (default ``30``).
    target_size : ``(width, height)`` for the decoded video frames.
    """

    def __init__(
        self,
        event_id: int,
        camera: CameraConfig,
        ring: CameraRingBuffer,
        *,
        rtmp_host: str = _DEFAULT_HOST,
        rtmp_port: int = _DEFAULT_PORT,
        target_fps: int = _DEFAULT_FPS,
        target_size: tuple[int, int] = (_DEFAULT_WIDTH, _DEFAULT_HEIGHT),
    ) -> None:
        self.event_id   = event_id
        self.camera     = camera
        self.ring       = ring
        self.rtmp_host  = rtmp_host
        self.rtmp_port  = rtmp_port
        self.target_fps = target_fps
        self.target_size = target_size  # (width, height)

        self._video_task: Optional[asyncio.Task] = None
        self._audio_task: Optional[asyncio.Task] = None
        self._running    = False
        self._event_start_t: float = 0.0   # monotonic reference set on start()

    # ── Public API ────────────────────────────────────────────────────────────

    def rtmp_url(self) -> str:
        """Return the RTMP ingest URL for this camera.

        ``rtmp://<host>:<port>/live/<event_id>/<cam_id>``
        """
        return (
            f"rtmp://{self.rtmp_host}:{self.rtmp_port}"
            f"/live/{self.event_id}/{self.camera.id}"
        )

    @property
    def is_alive(self) -> bool:
        """True if both reader tasks are currently running."""
        return (
            self._running
            and self._video_task is not None
            and not self._video_task.done()
            and self._audio_task is not None
            and not self._audio_task.done()
        )

    async def start(self) -> None:
        """Launch the video + audio FFmpeg subprocesses.

        Each subprocess runs in its own asyncio task with automatic restart
        on crash (exponential backoff, capped at 30 s).  This method returns
        immediately after spawning the tasks.
        """
        if self._running:
            logger.warning(
                "[%s] IngestWorker.start() called but already running — ignoring",
                self.camera.id,
            )
            return

        self._running = True
        self._event_start_t = time.monotonic()
        logger.info(
            "[%s] Starting ingest worker for event=%d url=%s",
            self.camera.id, self.event_id, self.rtmp_url(),
        )
        self._video_task = asyncio.create_task(
            self._run_loop(self._read_video, "video"),
            name=f"ingest-video-{self.camera.id}",
        )
        self._audio_task = asyncio.create_task(
            self._run_loop(self._read_audio, "audio"),
            name=f"ingest-audio-{self.camera.id}",
        )

    async def stop(self) -> None:
        """Cancel the ingest tasks and wait for them to finish."""
        self._running = False
        for task in (self._video_task, self._audio_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("[%s] IngestWorker stopped.", self.camera.id)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _elapsed(self) -> float:
        """Monotonic seconds since ``start()`` was called."""
        return time.monotonic() - self._event_start_t

    def _build_video_cmd(self) -> list[str]:
        """Build the FFmpeg argument list for raw BGR24 video.

        Exact command:
            ffmpeg -hide_banner -loglevel warning
                   -i <rtmp_url>
                   -f rawvideo -pix_fmt bgr24 -s WxH -r <fps>
                   pipe:1
        """
        w, h = self.target_size
        return [
            FFMPEG_BIN,
            "-hide_banner", "-loglevel", "warning",
            "-i", self.rtmp_url(),
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", str(self.target_fps),
            "pipe:1",
        ]

    def _build_audio_cmd(self) -> list[str]:
        """Build the FFmpeg argument list for raw s16le mono PCM audio.

        Exact command:
            ffmpeg -hide_banner -loglevel warning
                   -i <rtmp_url>
                   -vn -f s16le -ac 1 -ar 16000
                   pipe:1
        """
        return [
            FFMPEG_BIN,
            "-hide_banner", "-loglevel", "warning",
            "-i", self.rtmp_url(),
            "-vn",
            "-f", "s16le",
            "-ac", "1",
            "-ar", str(_AUDIO_SAMPLE_RATE),
            "pipe:1",
        ]

    def _frame_bytes(self) -> int:
        """Bytes per raw BGR24 video frame."""
        w, h = self.target_size
        return w * h * 3

    async def _run_loop(self, reader_coro, kind: str) -> None:
        """Outer restart loop with exponential backoff.

        Calls *reader_coro(process)* in a tight loop.  On any exception or
        process death: log, wait, restart.  Never raises.
        """
        backoff = _MIN_RESTART_S
        while self._running:
            try:
                await self._spawn_and_read(reader_coro, kind)
            except asyncio.CancelledError:
                logger.debug("[%s/%s] reader task cancelled.", self.camera.id, kind)
                return
            except Exception as exc:
                logger.error(
                    "[%s/%s] reader loop crashed: %s — restarting in %.1fs",
                    self.camera.id, kind, exc, backoff,
                )
            if not self._running:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _MAX_BACKOFF_S)

        logger.debug("[%s/%s] reader loop exiting (running=False).", self.camera.id, kind)

    async def _spawn_and_read(self, reader_coro, kind: str) -> None:
        """Spawn the appropriate FFmpeg subprocess and run the reader coroutine.

        Cleans up the subprocess on exit regardless of reason.
        """
        cmd = self._build_video_cmd() if kind == "video" else self._build_audio_cmd()
        logger.debug("[%s/%s] spawning: %s", self.camera.id, kind, " ".join(cmd))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await reader_coro(proc)
        finally:
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            try:
                await proc.wait()
            except Exception:
                pass

    async def _read_video(self, proc: asyncio.subprocess.Process) -> None:
        """Read fixed-size BGR24 frame chunks from stdout and push to ring."""
        frame_bytes = self._frame_bytes()
        w, h = self.target_size
        assert proc.stdout is not None

        while self._running:
            data = await proc.stdout.readexactly(frame_bytes)
            if not data:
                logger.warning("[%s/video] stdout closed — subprocess exited.", self.camera.id)
                return
            frame = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3)).copy()
            t = self._elapsed()
            await self.ring.push_video(t, frame)

    async def _read_audio(self, proc: asyncio.subprocess.Process) -> None:
        """Read fixed-size s16le mono PCM chunks from stdout and push to ring."""
        assert proc.stdout is not None

        while self._running:
            data = await proc.stdout.readexactly(_AUDIO_BYTES_PER_CHUNK)
            if not data:
                logger.warning("[%s/audio] stdout closed — subprocess exited.", self.camera.id)
                return
            samples = np.frombuffer(data, dtype=np.int16).copy()
            t = self._elapsed()
            await self.ring.push_audio(t, samples)
