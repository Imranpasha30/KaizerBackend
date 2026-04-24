"""
kaizer.pipeline.live_director.webrtc_ingest
============================================
Browser-webm → CameraRingBuffer bridge for the "phone as camera" flow.

A WebRTCIngestWorker owns one phone session. Webm chunks arriving on the
phone's WebSocket are fanned into two FFmpeg subprocesses:

    chunk_queue ──┬── stdin ──► ffmpeg(video) ── stdout ──► BGR frames
                  │
                  └── stdin ──► ffmpeg(audio) ── stdout ──► s16le PCM

Each decoded frame / PCM chunk is pushed into this camera's
CameraRingBuffer — the existing Phase 6.2 analyzers (face, motion, scene,
audio, reaction, beat) already poll that ring, so no analyzer changes
are needed. Once the bridge is running, the Director starts receiving
real SignalFrames from phone feeds and autonomous cuts fire for free.

Memory budget
-------------
Per worker:
  - Analyzer-resolution frames (320×240×3 = ~225 KB) × ring depth (60) ≈ 14 MB
  - Audio buffer (5 s × 16 kHz × 2 B) ≈ 160 KB
  - Chunk queue (≤ 30 webm chunks × ~30 KB each) ≈ 1 MB
  - Two ffmpeg subprocesses ≈ 60-100 MB combined
  - Numpy / opencv overhead per analyzer tick (transient)
Total hard cap: ~150 MB RSS per phone. 10 phones ≈ 1.5 GB.

Design decisions
----------------
- Two ffmpeg subprocs (video + audio) mirror the existing IngestWorker
  pattern. Simpler than one-proc-with-many-pipes on Windows.
- Drop oldest chunk when the chunk queue fills — never block the phone's
  producer. Back-pressure on a live camera would stall the WebSocket and
  surface as "choppy" on the director's side. Dropping 100 ms of video on
  a slow tick is strictly better than a 2 s stall.
- FFmpeg restarts with exponential backoff on crash (1 s → 30 s cap).
- Stdin-closing (worker shutdown) flushes the webm stream — ffmpeg exits
  cleanly.
- All heavy libs (numpy, opencv) are lazy-imported inside the analyzer
  callbacks, not at module level — matches the analyzers themselves so
  a dormant worker doesn't hold 200 MB of unused opencv state.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from pipeline_core.live_director.ring_buffer import CameraRingBuffer
from pipeline_core.live_director.signal_bus import SignalBus
from pipeline_core.pipeline import FFMPEG_BIN

logger = logging.getLogger("kaizer.pipeline.live_director.webrtc_ingest")


# ── Tuning ────────────────────────────────────────────────────────────────────

_MIN_RESTART_S   = 0.5
_MAX_BACKOFF_S   = 30.0


@dataclass
class WebRTCIngestConfig:
    """Tuning knobs for a single phone ingest worker."""
    analyzer_width:  int   = 320     # analyzer-resolution video
    analyzer_height: int   = 240
    analyzer_fps:    int   = 15      # 15 fps is enough for director decisions
    audio_sample_rate:  int = 16_000  # webrtcvad requires 8/16/32 kHz
    audio_chunk_ms:     int = 320     # matches the ring buffer's assumed chunk size
    max_chunk_queue:    int = 30      # ~15 s of 500 ms webm chunks
    ffmpeg_restart_initial_backoff_s: float = _MIN_RESTART_S
    ffmpeg_restart_max_backoff_s:     float = _MAX_BACKOFF_S


@dataclass
class WebRTCStats:
    """Live counters surfaced via `worker.stats()` for health checks + UI."""
    chunks_in:              int = 0
    chunks_dropped:         int = 0
    bytes_in:               int = 0
    frames_decoded:         int = 0
    audio_packets_decoded:  int = 0
    ffmpeg_restarts_video:  int = 0
    ffmpeg_restarts_audio:  int = 0
    last_error:             str = ""
    started_at:             float = 0.0


class WebRTCIngestWorker:
    """Per-phone-session webm → ring-buffer bridge.

    Usage
    -----
        worker = WebRTCIngestWorker(
            event_id=42, cam_id="phone_abc",
            ring=event_ring_for_this_cam, bus=session.bus,
        )
        await worker.start()
        # on every incoming WebSocket binary chunk:
        await worker.push_chunk(webm_bytes)
        # at event stop:
        await worker.stop()

    Lifecycle notes
    ---------------
    - `start()` spawns two ffmpeg subprocs + three asyncio tasks
      (chunk-pump, video-reader, audio-reader). It is safe to call once.
    - `push_chunk()` is non-blocking on a full queue — oldest chunk is
      dropped so the phone's websocket producer never stalls.
    - `stop()` is idempotent and cancels every task + closes stdin on
      both subprocs + waits for exits.
    - FFmpeg dying does NOT kill the worker — the supervisor loop relaunches
      with exponential backoff. During the gap the ring simply stops getting
      fresh frames; the director rolls with the last-known state.
    """

    def __init__(
        self,
        *,
        event_id: int,
        cam_id: str,
        ring: CameraRingBuffer,
        bus: Optional[SignalBus] = None,
        config: Optional[WebRTCIngestConfig] = None,
        on_error: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.event_id = event_id
        self.cam_id   = cam_id
        self.ring     = ring
        self.bus      = bus
        self.config   = config or WebRTCIngestConfig()
        self._on_error = on_error

        self._chunk_queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=self.config.max_chunk_queue,
        )
        self._video_proc: Optional[asyncio.subprocess.Process] = None
        self._audio_proc: Optional[asyncio.subprocess.Process] = None
        self._chunk_pump_task: Optional[asyncio.Task] = None
        self._video_loop_task: Optional[asyncio.Task] = None
        self._audio_loop_task: Optional[asyncio.Task] = None
        self._running: bool = False
        self._event_start_t: float = 0.0
        self._stats = WebRTCStats()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    def stats(self) -> dict:
        """Snapshot of the live counters + queue depth + uptime."""
        uptime = (time.monotonic() - self._event_start_t) if self._event_start_t else 0.0
        return {
            "cam_id":                 self.cam_id,
            "running":                self._running,
            "uptime_s":               round(uptime, 2),
            "chunks_in":              self._stats.chunks_in,
            "chunks_dropped":         self._stats.chunks_dropped,
            "bytes_in":               self._stats.bytes_in,
            "frames_decoded":         self._stats.frames_decoded,
            "audio_packets_decoded":  self._stats.audio_packets_decoded,
            "ffmpeg_restarts_video":  self._stats.ffmpeg_restarts_video,
            "ffmpeg_restarts_audio":  self._stats.ffmpeg_restarts_audio,
            "queue_depth":            self._chunk_queue.qsize(),
            "queue_capacity":         self.config.max_chunk_queue,
            "last_error":             self._stats.last_error,
        }

    async def start(self) -> None:
        """Launch the two FFmpeg subprocesses + start the pump/reader tasks.

        Safe to call once. Idempotent subsequent calls log a warning.
        """
        if self._running:
            logger.warning("[%s] start() called but already running", self.cam_id)
            return
        self._running = True
        self._event_start_t = time.monotonic()
        self._stats.started_at = self._event_start_t
        logger.info(
            "[%s] WebRTCIngestWorker starting (event=%d, analyzer=%dx%d@%d)",
            self.cam_id, self.event_id,
            self.config.analyzer_width, self.config.analyzer_height,
            self.config.analyzer_fps,
        )
        self._chunk_pump_task = asyncio.create_task(
            self._chunk_pump_loop(), name=f"webrtc-pump-{self.cam_id}",
        )
        self._video_loop_task = asyncio.create_task(
            self._supervisor("video"), name=f"webrtc-video-{self.cam_id}",
        )
        self._audio_loop_task = asyncio.create_task(
            self._supervisor("audio"), name=f"webrtc-audio-{self.cam_id}",
        )

    async def push_chunk(self, chunk: bytes) -> None:
        """Enqueue one webm chunk. If the queue is full, DROP the oldest
        chunk and enqueue the new one — never block the phone's producer.
        """
        if not chunk:
            return
        self._stats.chunks_in += 1
        self._stats.bytes_in += len(chunk)
        try:
            self._chunk_queue.put_nowait(chunk)
        except asyncio.QueueFull:
            # Drop oldest, retry once
            try:
                self._chunk_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._stats.chunks_dropped += 1
            try:
                self._chunk_queue.put_nowait(chunk)
            except asyncio.QueueFull:
                # Extremely unlikely — consumer is completely stuck
                self._stats.chunks_dropped += 1

    async def stop(self) -> None:
        """Terminate ffmpeg subprocs, cancel all tasks. Idempotent."""
        if not self._running:
            return
        self._running = False
        logger.info("[%s] WebRTCIngestWorker stopping", self.cam_id)
        # Close stdin on both subprocs so ffmpeg flushes + exits cleanly
        for proc in (self._video_proc, self._audio_proc):
            if proc and proc.stdin and not proc.stdin.is_closing():
                try:
                    proc.stdin.close()
                except Exception:
                    pass
        # Cancel tasks
        for task in (
            self._chunk_pump_task, self._video_loop_task, self._audio_loop_task,
        ):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        # Final kill if still alive
        for proc in (self._video_proc, self._audio_proc):
            if proc and proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    await proc.wait()
                except Exception:
                    pass
        self._video_proc = None
        self._audio_proc = None

    # ── Internal: chunk fan-out ──────────────────────────────────────────────

    async def _chunk_pump_loop(self) -> None:
        """Read from the chunk queue and write to both ffmpeg stdins.

        If a subproc's stdin is closed / broken (mid-restart), skip silently
        — the supervisor will restart it.
        """
        while self._running:
            try:
                chunk = await self._chunk_queue.get()
            except asyncio.CancelledError:
                return
            for proc in (self._video_proc, self._audio_proc):
                if proc and proc.stdin and not proc.stdin.is_closing():
                    try:
                        proc.stdin.write(chunk)
                        await proc.stdin.drain()
                    except (BrokenPipeError, ConnectionResetError, RuntimeError):
                        pass
                    except Exception as exc:
                        logger.debug("[%s] stdin write failed: %s", self.cam_id, exc)

    # ── Internal: ffmpeg supervisor ───────────────────────────────────────────

    async def _supervisor(self, kind: str) -> None:
        """Spawn + re-spawn ffmpeg with exponential backoff until stop()."""
        backoff = self.config.ffmpeg_restart_initial_backoff_s
        while self._running:
            try:
                await self._spawn_and_read(kind)
                backoff = self.config.ffmpeg_restart_initial_backoff_s
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._stats.last_error = f"{kind}: {exc}"
                logger.error(
                    "[%s/%s] ffmpeg crashed: %s — restart in %.1fs",
                    self.cam_id, kind, exc, backoff,
                )
                if kind == "video":
                    self._stats.ffmpeg_restarts_video += 1
                else:
                    self._stats.ffmpeg_restarts_audio += 1
                if self._on_error:
                    try:
                        self._on_error(kind, str(exc))
                    except Exception:
                        pass
            if not self._running:
                break
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                return
            backoff = min(backoff * 2, self.config.ffmpeg_restart_max_backoff_s)

    async def _spawn_and_read(self, kind: str) -> None:
        """Spawn one ffmpeg subproc and drive its output reader.

        Captures stderr into a rolling buffer surfaced via stats().last_error
        so the debug panel can see why ffmpeg is refusing to decode.
        Cleans up the subproc on any exit path.
        """
        cmd = self._build_cmd(kind)
        logger.debug("[%s/%s] spawning: %s", self.cam_id, kind, " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if kind == "video":
            self._video_proc = proc
        else:
            self._audio_proc = proc

        # Concurrent stderr reader so ffmpeg complaints reach the debug panel.
        stderr_task = asyncio.create_task(
            self._drain_stderr(proc, kind), name=f"webrtc-err-{self.cam_id}-{kind}",
        )

        try:
            if kind == "video":
                await self._read_video(proc)
            else:
                await self._read_audio(proc)
        finally:
            # Cancel stderr reader first so it doesn't block on a killed pipe.
            stderr_task.cancel()
            try:
                await stderr_task
            except (asyncio.CancelledError, Exception):
                pass
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            try:
                await proc.wait()
            except Exception:
                pass

    async def _drain_stderr(self, proc: asyncio.subprocess.Process, kind: str) -> None:
        """Stream stderr into a rolling 2KB snippet on last_error.
        Each non-empty line is also logged at debug level.
        """
        if proc.stderr is None:
            return
        rolling = bytearray()
        while True:
            try:
                line = await proc.stderr.readline()
            except asyncio.CancelledError:
                raise
            except Exception:
                return
            if not line:
                return
            rolling.extend(line)
            if len(rolling) > 2048:
                rolling = rolling[-2048:]
            try:
                txt = line.decode("utf-8", errors="replace").rstrip()
            except Exception:
                txt = "<undecodable>"
            if txt:
                logger.debug("[%s/%s] ffmpeg: %s", self.cam_id, kind, txt)
                # Always expose the most-recent line so operators can see live
                # decode warnings even when ffmpeg isn't crashing.
                self._stats.last_error = f"{kind}: {txt[:200]}"

    def _build_cmd(self, kind: str) -> list[str]:
        """Build the ffmpeg command for one kind (video | audio).

        MediaRecorder on the phone emits a matroska-subset (webm). Using
        `-f matroska,webm` accepts both strict-webm and generic-matroska
        outputs.  `-fflags +nobuffer`, `-flags low_delay`, `-probesize 32`
        and `-analyzeduration 0` make ffmpeg start emitting frames as soon
        as it sees the first Cluster — without these, ffmpeg can buffer
        multi-second chunks before producing any output. That was the
        cause of the "ring fps below 1 despite chunks arriving" error.
        """
        common_input = [
            "-hide_banner", "-loglevel", "warning",
            "-fflags", "+nobuffer+genpts+discardcorrupt",
            "-flags", "low_delay",
            "-probesize", "32",
            "-analyzeduration", "0",
            "-f", "matroska,webm",
            "-i", "pipe:0",
        ]
        if kind == "video":
            return [
                FFMPEG_BIN,
                *common_input,
                "-an",
                "-vf", f"fps={self.config.analyzer_fps},"
                       f"scale={self.config.analyzer_width}:{self.config.analyzer_height}",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "pipe:1",
            ]
        return [
            FFMPEG_BIN,
            *common_input,
            "-vn",
            "-ac", "1", "-ar", str(self.config.audio_sample_rate),
            "-f", "s16le",
            "pipe:1",
        ]

    async def _read_video(self, proc: asyncio.subprocess.Process) -> None:
        """Read fixed-size BGR frames from ffmpeg stdout, push to ring."""
        import numpy as np  # lazy

        w, h = self.config.analyzer_width, self.config.analyzer_height
        frame_bytes = w * h * 3
        assert proc.stdout is not None

        while self._running:
            try:
                data = await proc.stdout.readexactly(frame_bytes)
            except asyncio.IncompleteReadError:
                return
            if not data:
                return
            frame = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3)).copy()
            t = time.monotonic() - self._event_start_t
            await self.ring.push_video(t, frame)
            self._stats.frames_decoded += 1

    async def _read_audio(self, proc: asyncio.subprocess.Process) -> None:
        """Read fixed-size PCM packets from ffmpeg stdout, push to ring."""
        import numpy as np  # lazy

        samples_per_chunk = int(
            self.config.audio_sample_rate * self.config.audio_chunk_ms / 1000
        )
        bytes_per_chunk = samples_per_chunk * 2  # int16 = 2 B/sample
        assert proc.stdout is not None

        while self._running:
            try:
                data = await proc.stdout.readexactly(bytes_per_chunk)
            except asyncio.IncompleteReadError:
                return
            if not data:
                return
            samples = np.frombuffer(data, dtype=np.int16).copy()
            t = time.monotonic() - self._event_start_t
            await self.ring.push_audio(t, samples)
            self._stats.audio_packets_decoded += 1
