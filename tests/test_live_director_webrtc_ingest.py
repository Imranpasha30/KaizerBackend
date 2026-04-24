"""
Phase 10.1 — WebRTCIngestWorker unit tests.

Mocks asyncio.create_subprocess_exec so no real ffmpeg runs. Covers the
chunk fan-out, drop-oldest back-pressure, stat counters, start/stop
lifecycle, and that video/audio frames reach the ring buffer when the
mock ffmpeg stdout is fed fixed-size payloads.
"""
from __future__ import annotations

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from pipeline_core.live_director.ring_buffer import CameraRingBuffer
from pipeline_core.live_director.signal_bus import SignalBus
from pipeline_core.live_director.webrtc_ingest import (
    WebRTCIngestConfig,
    WebRTCIngestWorker,
)


def _mock_proc(stdout_payload: bytes = b"") -> MagicMock:
    """Build a MagicMock that looks like an asyncio.subprocess.Process.

    stdout.readexactly returns successive slices of `stdout_payload` then
    raises IncompleteReadError (EOF) so readers exit cleanly.
    """
    proc = MagicMock()
    proc.returncode = None
    proc.stdin = MagicMock()
    proc.stdin.is_closing = MagicMock(return_value=False)
    proc.stdin.close = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()

    proc.stdout = MagicMock()
    buf = {"offset": 0, "data": stdout_payload}

    async def _readexactly(n):
        start = buf["offset"]
        end = start + n
        if end > len(buf["data"]):
            # Simulate EOF
            raise asyncio.IncompleteReadError(partial=b"", expected=n)
        buf["offset"] = end
        return buf["data"][start:end]

    proc.stdout.readexactly = _readexactly

    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=0)
    return proc


@pytest.mark.asyncio
async def test_push_chunk_enqueues_when_running():
    ring = CameraRingBuffer("cam1", video_max_frames=10, audio_max_samples=16000)
    bus = SignalBus()
    worker = WebRTCIngestWorker(
        event_id=1, cam_id="cam1", ring=ring, bus=bus,
        config=WebRTCIngestConfig(max_chunk_queue=5),
    )
    # Don't actually start — we just want to verify push_chunk bumps counters.
    # First chunk is captured as _init_chunk (EBML header preserve); chunks 2
    # and 3 go through the normal queue.
    for i in range(3):
        await worker.push_chunk(b"x" * 100)
    s = worker.stats()
    assert s["chunks_in"] == 3
    assert s["bytes_in"] == 300
    assert s["chunks_dropped"] == 0
    assert s["queue_depth"] == 2          # 3 pushed minus 1 captured as init
    assert worker._init_chunk == b"x" * 100


@pytest.mark.asyncio
async def test_push_chunk_drops_oldest_when_queue_full():
    ring = CameraRingBuffer("cam1")
    bus = SignalBus()
    worker = WebRTCIngestWorker(
        event_id=1, cam_id="cam1", ring=ring, bus=bus,
        config=WebRTCIngestConfig(max_chunk_queue=3),
    )
    # 5 pushed: chunk 1 → init_chunk; chunks 2-5 → queue (capacity 3).
    # So 4 enqueues into a 3-slot queue → 1 drop.
    for i in range(5):
        await worker.push_chunk(bytes([i]) * 10)
    s = worker.stats()
    assert s["chunks_in"] == 5
    assert s["chunks_dropped"] == 1
    assert s["queue_depth"] == 3
    assert worker._init_chunk == bytes([0]) * 10


@pytest.mark.asyncio
async def test_push_chunk_ignores_empty_payload():
    ring = CameraRingBuffer("cam1")
    bus = SignalBus()
    worker = WebRTCIngestWorker(event_id=1, cam_id="cam1", ring=ring, bus=bus)
    await worker.push_chunk(b"")
    assert worker.stats()["chunks_in"] == 0
    assert worker.stats()["queue_depth"] == 0


@pytest.mark.asyncio
async def test_start_spawns_two_ffmpeg_subprocesses():
    ring = CameraRingBuffer("cam1")
    bus = SignalBus()
    worker = WebRTCIngestWorker(event_id=1, cam_id="cam1", ring=ring, bus=bus)

    with patch(
        "asyncio.create_subprocess_exec",
        new=AsyncMock(side_effect=lambda *a, **kw: _mock_proc()),
    ) as spawn:
        await worker.start()
        # Pump pushes against both stdins; give the supervisor tasks a moment
        # to reach create_subprocess_exec.
        await asyncio.sleep(0.05)
        assert spawn.call_count >= 2  # video + audio
    await worker.stop()


@pytest.mark.asyncio
async def test_stop_is_idempotent():
    ring = CameraRingBuffer("cam1")
    bus = SignalBus()
    worker = WebRTCIngestWorker(event_id=1, cam_id="cam1", ring=ring, bus=bus)
    await worker.stop()  # not-running → no-op
    await worker.stop()  # still no-op


@pytest.mark.asyncio
async def test_stop_closes_stdins_and_waits():
    ring = CameraRingBuffer("cam1")
    bus = SignalBus()
    worker = WebRTCIngestWorker(event_id=1, cam_id="cam1", ring=ring, bus=bus)

    procs = []

    def _make(*a, **kw):
        p = _mock_proc()
        procs.append(p)
        return p

    with patch("asyncio.create_subprocess_exec", new=AsyncMock(side_effect=_make)):
        await worker.start()
        await asyncio.sleep(0.05)  # let supervisors reach spawn
        await worker.stop()

    # Every spawned proc should have had stdin.close() called at least once.
    assert len(procs) >= 2
    for p in procs:
        assert p.stdin.close.called or p.kill.called


@pytest.mark.asyncio
async def test_chunk_pump_forwards_to_both_stdins():
    ring = CameraRingBuffer("cam1")
    bus = SignalBus()
    cfg = WebRTCIngestConfig(max_chunk_queue=10)
    worker = WebRTCIngestWorker(event_id=1, cam_id="cam1", ring=ring, bus=bus, config=cfg)

    procs = []

    def _make(*a, **kw):
        p = _mock_proc()
        procs.append(p)
        return p

    with patch("asyncio.create_subprocess_exec", new=AsyncMock(side_effect=_make)):
        await worker.start()
        # Wait for both subprocs to be referenced by the worker
        for _ in range(20):
            if worker._video_proc is not None and worker._audio_proc is not None:
                break
            await asyncio.sleep(0.01)
        await worker.push_chunk(b"ABC")
        await worker.push_chunk(b"XYZ")
        # Give the pump loop a chance to drain
        await asyncio.sleep(0.05)

    # Both subprocs should have received the writes
    assert len(procs) >= 2
    writes_v = [c.args[0] for c in procs[0].stdin.write.call_args_list]
    writes_a = [c.args[0] for c in procs[1].stdin.write.call_args_list]
    assert b"ABC" in writes_v
    assert b"XYZ" in writes_v
    assert b"ABC" in writes_a
    assert b"XYZ" in writes_a
    await worker.stop()


@pytest.mark.asyncio
async def test_video_reader_pushes_decoded_frames_to_ring():
    cfg = WebRTCIngestConfig(analyzer_width=4, analyzer_height=2, analyzer_fps=15)
    ring = CameraRingBuffer("cam1", video_max_frames=10)
    bus = SignalBus()
    worker = WebRTCIngestWorker(
        event_id=1, cam_id="cam1", ring=ring, bus=bus, config=cfg,
    )
    frame_bytes = 4 * 2 * 3  # 24 bytes per frame at 4x2
    payload = b"\xaa" * (frame_bytes * 3)  # 3 frames

    def _make(*a, **kw):
        # The FIRST subproc is the video one (audio second), per _supervisor
        # dispatch order. Give video a payload, audio empty.
        if _make.n == 0:
            _make.n += 1
            return _mock_proc(stdout_payload=payload)
        _make.n += 1
        return _mock_proc(stdout_payload=b"")
    _make.n = 0

    with patch("asyncio.create_subprocess_exec", new=AsyncMock(side_effect=_make)):
        await worker.start()
        # Wait for the video reader to chew through the payload
        for _ in range(30):
            if worker.stats()["frames_decoded"] >= 3:
                break
            await asyncio.sleep(0.02)
        await worker.stop()

    assert worker.stats()["frames_decoded"] >= 3
    # Ring should have received frames too
    latest = await ring.latest_video(count=3)
    assert len(latest) == 3
    for t, frame in latest:
        assert frame.shape == (2, 4, 3)


@pytest.mark.asyncio
async def test_audio_reader_pushes_pcm_to_ring():
    cfg = WebRTCIngestConfig(
        audio_sample_rate=16_000, audio_chunk_ms=100,  # 1600 samples = 3200 B
    )
    ring = CameraRingBuffer("cam1")
    bus = SignalBus()
    worker = WebRTCIngestWorker(
        event_id=1, cam_id="cam1", ring=ring, bus=bus, config=cfg,
    )
    bytes_per_chunk = 1600 * 2  # 3200
    payload = b"\x00\x01" * (1600 * 2)  # 2 chunks

    def _make(*a, **kw):
        # Dispatch: video first (empty), audio second (our payload)
        if _make.n == 0:
            _make.n += 1
            return _mock_proc(stdout_payload=b"")
        _make.n += 1
        return _mock_proc(stdout_payload=payload)
    _make.n = 0

    with patch("asyncio.create_subprocess_exec", new=AsyncMock(side_effect=_make)):
        await worker.start()
        for _ in range(30):
            if worker.stats()["audio_packets_decoded"] >= 2:
                break
            await asyncio.sleep(0.02)
        await worker.stop()

    assert worker.stats()["audio_packets_decoded"] >= 2


@pytest.mark.asyncio
async def test_stats_snapshot_shape():
    worker = WebRTCIngestWorker(
        event_id=1, cam_id="cam1",
        ring=CameraRingBuffer("cam1"), bus=SignalBus(),
    )
    s = worker.stats()
    # Schema check — every expected key is present
    required = {
        "cam_id", "running", "uptime_s", "chunks_in", "chunks_dropped",
        "bytes_in", "frames_decoded", "audio_packets_decoded",
        "ffmpeg_restarts_video", "ffmpeg_restarts_audio",
        "queue_depth", "queue_capacity", "last_error",
    }
    assert required.issubset(s.keys())
