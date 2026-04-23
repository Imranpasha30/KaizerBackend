"""
kaizer.pipeline.live_director.ring_buffer
==========================================
Async-safe per-camera circular buffer holding the last ~5 s of decoded
video frames and raw audio samples.

Design
------
- Video and audio kept in separate ``collections.deque(maxlen=...)`` instances
  so analyzers can pull at different rates without coupling.
- All mutations and reads are guarded by a single ``asyncio.Lock`` so multiple
  concurrent analyzer coroutines can share the same buffer safely.
- ``stats()`` returns running health metrics consumed by the UI overlay and
  the health-check DirectorEvent.

Frame layout
------------
Video  : ``(timestamp_s: float, frame: np.ndarray)``  — BGR24, shape (H, W, 3)
Audio  : ``(timestamp_s: float, samples: np.ndarray)`` — int16 mono flat array
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import NamedTuple

import numpy as np

logger = logging.getLogger("kaizer.pipeline.live_director.ring_buffer")

# ── Internal frame containers ─────────────────────────────────────────────────

class _VideoFrame(NamedTuple):
    t: float
    frame: np.ndarray


class _AudioChunk(NamedTuple):
    t: float
    samples: np.ndarray   # int16, mono, 16 kHz


# ─────────────────────────────────────────────────────────────────────────────


class CameraRingBuffer:
    """Per-camera circular frame + audio buffer.

    Parameters
    ----------
    cam_id          : Camera identifier (for logging).
    video_max_frames: Maximum video frames to retain (default 60 ≈ 2 s @ 30 fps).
        2 s is enough lookback for every analyzer currently defined (face /
        motion / scene). A 1080×1920×3 BGR frame is 6 MB → 60 frames per
        camera = 360 MB. On a 4-camera event that's 1.44 GB for video
        buffers alone; any larger and the 32 GB host OOMs.
    audio_max_samples: Maximum audio samples to retain (default 80 000 ≈ 5 s @ 16 kHz).
        Audio is cheap (16 kHz mono int16 = 32 KB/s), so 5 s stays.
    """

    def __init__(
        self,
        cam_id: str,
        *,
        video_max_frames: int = 60,    # ~2 s @ 30 fps — 360 MB per cam @ 1080×1920
        audio_max_samples: int = 80_000,  # ~5 s @ 16 kHz — cheap, keep 5 s
    ) -> None:
        self.cam_id = cam_id
        self._video_max = video_max_frames
        self._audio_max = audio_max_samples

        self._video: deque[_VideoFrame] = deque(maxlen=video_max_frames)
        # Audio is stored as a flat deque of individual *samples* so that
        # arbitrary-length windows can be sliced without bookkeeping overhead.
        # Each element is a (t, samples_array) pair; we cap the deque length
        # at a number of *chunks* instead. We translate audio_max_samples to
        # a generous chunk count (each 320 ms push ≈ 5 120 samples at 16 kHz).
        _chunk_samples = 5_120  # 320 ms @ 16 kHz per push
        _max_chunks = max(1, audio_max_samples // _chunk_samples) + 1
        self._audio: deque[_AudioChunk] = deque(maxlen=_max_chunks)

        self._lock = asyncio.Lock()

        # Stats tracking
        self._frames_pushed: int = 0
        self._frames_dropped: int = 0
        self._push_times: deque[float] = deque(maxlen=90)  # wall-clock for fps
        self._last_video_t: float | None = None

    # ── Write API ────────────────────────────────────────────────────────────

    async def push_video(self, t: float, frame: np.ndarray) -> None:
        """Append one decoded BGR frame to the video ring.

        If the deque is already at *maxlen* the oldest frame is silently evicted
        (standard ``deque`` behaviour). The dropped-frames counter is incremented
        whenever the deque was full before the push.
        """
        async with self._lock:
            was_full = len(self._video) == self._video_max
            self._video.append(_VideoFrame(t=t, frame=frame))
            self._frames_pushed += 1
            if was_full:
                self._frames_dropped += 1
            wall = time.monotonic()
            self._push_times.append(wall)
            self._last_video_t = t

    async def push_audio(self, t: float, samples: np.ndarray) -> None:
        """Append one audio chunk (int16 flat array, mono 16 kHz)."""
        async with self._lock:
            self._audio.append(_AudioChunk(t=t, samples=samples))

    # ── Read API ─────────────────────────────────────────────────────────────

    async def latest_video(self, count: int = 1) -> list[tuple[float, np.ndarray]]:
        """Return the *count* most-recent video frames as ``(t, frame)`` pairs.

        Returns fewer than *count* items if the buffer is not yet full.
        """
        async with self._lock:
            frames = list(self._video)
            tail = frames[-count:] if count <= len(frames) else frames
            return [(f.t, f.frame) for f in tail]

    async def latest_audio(self, seconds: float = 0.3) -> tuple[float, np.ndarray]:
        """Return a contiguous PCM array covering the last *seconds* of audio.

        Returns ``(earliest_t, samples_array)``.  If there is not enough audio
        buffered, returns whatever is available (may be shorter than requested).
        The returned array is a new allocation so callers can modify it freely.
        """
        sample_rate = 16_000
        needed = int(seconds * sample_rate)
        async with self._lock:
            chunks = list(self._audio)

        if not chunks:
            return (0.0, np.zeros(0, dtype=np.int16))

        # Concatenate all chunks and take the last *needed* samples.
        all_samples = np.concatenate([c.samples for c in chunks])
        if len(all_samples) >= needed:
            window = all_samples[-needed:]
        else:
            window = all_samples

        earliest_t = chunks[0].t
        return (earliest_t, window.copy())

    # ── Health / stats ───────────────────────────────────────────────────────

    async def stats(self) -> dict:
        """Return a health snapshot dict for UI overlay and DirectorEvent payloads.

        Keys
        ----
        fps             : Measured video push rate over the last 3 s.
        frames_buffered : Current number of frames in the video deque.
        dropped_frames  : Cumulative number of frames evicted while full.
        lag_ms          : Estimated lag: difference between wall-clock now and
                          the monotonic timestamp of the most-recent frame,
                          expressed in milliseconds. 0 if no frames yet.
        """
        async with self._lock:
            frames_buffered = len(self._video)
            dropped = self._frames_dropped
            push_times = list(self._push_times)
            last_t = self._last_video_t

        now_wall = time.monotonic()

        # FPS: count how many pushes happened in the last 3 s
        window_s = 3.0
        recent = [pt for pt in push_times if now_wall - pt <= window_s]
        fps = len(recent) / window_s if recent else 0.0

        # Lag: approximate by comparing wall-clock intervals
        if len(push_times) >= 2:
            elapsed_wall = push_times[-1] - push_times[0]
            elapsed_t = (last_t or 0.0)
            # A simple heuristic: if push_times window is shorter than actual t
            # window there is buffering lag. For most cases last push < 1s ago.
            lag_ms = max(0.0, (now_wall - push_times[-1]) * 1000.0)
        else:
            lag_ms = 0.0

        return {
            "cam_id": self.cam_id,
            "fps": round(fps, 2),
            "frames_buffered": frames_buffered,
            "dropped_frames": dropped,
            "lag_ms": round(lag_ms, 1),
        }
