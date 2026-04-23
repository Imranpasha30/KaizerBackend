"""
tests/test_live_director_foundation.py
========================================
Phase 6 Foundation — ≥20 test cases covering:
  - signals.py      : all 4 dataclasses
  - ring_buffer.py  : push/pop, eviction, concurrency, stats
  - signal_bus.py   : publish/subscribe, latest_per_camera, overflow
  - ingest.py       : URL construction, constructor, ffmpeg args, restart, frame math
"""
from __future__ import annotations

import asyncio
import dataclasses
import math
from unittest.mock import AsyncMock, MagicMock, patch, call

import numpy as np
import pytest
import pytest_asyncio

# ── imports under test ────────────────────────────────────────────────────────
from pipeline_core.live_director.signals import (
    CameraConfig,
    CameraSelection,
    DirectorEvent,
    SignalFrame,
)
from pipeline_core.live_director.ring_buffer import CameraRingBuffer
from pipeline_core.live_director.signal_bus import SignalBus
from pipeline_core.live_director.ingest import (
    IngestWorker,
    _AUDIO_BYTES_PER_CHUNK,
    _AUDIO_CHUNK_SAMPLES,
)


# ══════════════════════════════════════════════════════════════════════════════
# §1 — signals.py
# ══════════════════════════════════════════════════════════════════════════════


class TestCameraConfig:
    def test_required_fields(self):
        cfg = CameraConfig(id="cam1", label="Stage Left")
        assert cfg.id == "cam1"
        assert cfg.label == "Stage Left"

    def test_defaults(self):
        cfg = CameraConfig(id="cam1", label="Stage Left")
        assert cfg.mic_id is None
        assert cfg.role_hints == []

    def test_role_hints_preserved(self):
        hints = ["stage", "wide", "closeup_artist_1"]
        cfg = CameraConfig(id="cam2", label="Wide", role_hints=hints)
        assert cfg.role_hints == hints

    def test_role_hints_not_shared(self):
        """Each instance gets its own default list — no mutable-default aliasing."""
        a = CameraConfig(id="a", label="A")
        b = CameraConfig(id="b", label="B")
        a.role_hints.append("stage")
        assert b.role_hints == [], "default_factory must give independent lists"

    def test_mic_id_set(self):
        cfg = CameraConfig(id="c", label="C", mic_id="mic_front")
        assert cfg.mic_id == "mic_front"


class TestSignalFrame:
    def test_minimal_construction(self):
        sf = SignalFrame(cam_id="cam1", t=1.5)
        assert sf.cam_id == "cam1"
        assert sf.t == 1.5

    def test_defaults_are_sane(self):
        sf = SignalFrame(cam_id="x", t=0.0)
        assert sf.audio_rms == 0.0
        assert sf.vad_speaking is False
        assert sf.face_present is False
        assert sf.face_size_norm == 0.0
        assert sf.face_identity is None
        assert sf.scene == "unknown"
        assert sf.motion_mag == 0.0
        assert sf.reaction is None
        assert sf.beat_phase is None

    def test_all_fields_set(self):
        sf = SignalFrame(
            cam_id="cam3", t=10.0,
            audio_rms=0.8, vad_speaking=True,
            face_present=True, face_size_norm=0.25,
            face_identity="artist_1", scene="stage",
            motion_mag=0.3, reaction="cheer", beat_phase=0.5,
        )
        assert sf.face_identity == "artist_1"
        assert sf.beat_phase == 0.5

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(SignalFrame)


class TestCameraSelection:
    def test_defaults(self):
        sel = CameraSelection(t=5.0, cam_id="cam1")
        assert sel.transition == "cut"
        assert sel.confidence == 0.0
        assert sel.reason == ""

    def test_full_construction(self):
        sel = CameraSelection(t=7.5, cam_id="cam2", transition="dissolve",
                               confidence=0.9, reason="beat cut")
        assert sel.reason == "beat cut"


class TestDirectorEvent:
    def test_defaults(self):
        ev = DirectorEvent(t=1.0, kind="selection")
        assert ev.payload == {}

    def test_payload_not_shared(self):
        a = DirectorEvent(t=1.0, kind="health")
        b = DirectorEvent(t=2.0, kind="health")
        a.payload["key"] = "value"
        assert "key" not in b.payload

    def test_kind_values(self):
        for kind in ("selection", "override", "camera_lost", "health"):
            ev = DirectorEvent(t=0.0, kind=kind)
            assert ev.kind == kind


# ══════════════════════════════════════════════════════════════════════════════
# §2 — ring_buffer.py
# ══════════════════════════════════════════════════════════════════════════════


def _make_frame(h=1920, w=1080, val=0):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _make_audio(samples=5120, val=100):
    return np.full(samples, val, dtype=np.int16)


class TestCameraRingBuffer:
    @pytest.mark.asyncio
    async def test_push_video_and_latest_video(self):
        buf = CameraRingBuffer("cam1")
        frame = _make_frame(val=42)
        await buf.push_video(1.0, frame)
        results = await buf.latest_video(1)
        assert len(results) == 1
        t, f = results[0]
        assert t == 1.0
        np.testing.assert_array_equal(f, frame)

    @pytest.mark.asyncio
    async def test_push_audio_and_latest_audio_sample_count(self):
        buf = CameraRingBuffer("cam1")
        sample_rate = 16_000
        window_s = 0.3
        expected = int(sample_rate * window_s)   # 4800

        # Push exactly enough samples
        samples = _make_audio(expected, val=50)
        await buf.push_audio(0.0, samples)

        _, out = await buf.latest_audio(window_s)
        assert len(out) == expected

    @pytest.mark.asyncio
    async def test_latest_audio_partial_when_not_enough_buffered(self):
        buf = CameraRingBuffer("cam1")
        await buf.push_audio(0.0, _make_audio(100))
        _, out = await buf.latest_audio(0.3)   # wants 4800 samples, only 100 available
        assert len(out) == 100

    @pytest.mark.asyncio
    async def test_oldest_frame_evicted_at_maxlen(self):
        maxlen = 5
        buf = CameraRingBuffer("cam1", video_max_frames=maxlen)
        for i in range(maxlen + 3):
            await buf.push_video(float(i), _make_frame(val=i))

        results = await buf.latest_video(maxlen + 3)
        # Should have at most maxlen frames
        assert len(results) <= maxlen
        # Oldest must be gone — first t should be >= 3
        ts = [r[0] for r in results]
        assert ts[0] >= 3.0

    @pytest.mark.asyncio
    async def test_stats_returns_required_keys(self):
        buf = CameraRingBuffer("cam1")
        await buf.push_video(1.0, _make_frame())
        s = await buf.stats()
        assert "fps" in s
        assert "frames_buffered" in s
        assert "dropped_frames" in s
        assert "lag_ms" in s

    @pytest.mark.asyncio
    async def test_stats_frames_buffered(self):
        buf = CameraRingBuffer("cam1")
        for i in range(5):
            await buf.push_video(float(i), _make_frame())
        s = await buf.stats()
        assert s["frames_buffered"] == 5

    @pytest.mark.asyncio
    async def test_dropped_frames_counter(self):
        maxlen = 3
        buf = CameraRingBuffer("cam1", video_max_frames=maxlen)
        for i in range(maxlen + 2):
            await buf.push_video(float(i), _make_frame())
        s = await buf.stats()
        assert s["dropped_frames"] == 2

    @pytest.mark.asyncio
    async def test_concurrent_push_and_read(self):
        """Hammer test: 2 concurrent pushers + 1 reader — no deadlock or corruption."""
        buf = CameraRingBuffer("cam_hammer", video_max_frames=60)
        results = []

        async def pusher(offset: int):
            for i in range(20):
                await buf.push_video(float(offset + i), _make_frame(val=offset + i))
                await asyncio.sleep(0)

        async def reader():
            for _ in range(10):
                frames = await buf.latest_video(5)
                results.append(len(frames))
                await asyncio.sleep(0)

        await asyncio.gather(pusher(0), pusher(100), reader())
        # Just assert no exception was raised and reader got some frames
        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_latest_video_returns_most_recent(self):
        buf = CameraRingBuffer("cam1")
        for i in range(10):
            await buf.push_video(float(i), _make_frame(val=i))
        frames = await buf.latest_video(3)
        ts = [f[0] for f in frames]
        assert ts == [7.0, 8.0, 9.0]

    @pytest.mark.asyncio
    async def test_empty_buffer_latest_video(self):
        buf = CameraRingBuffer("cam1")
        results = await buf.latest_video(5)
        assert results == []

    @pytest.mark.asyncio
    async def test_empty_buffer_latest_audio(self):
        buf = CameraRingBuffer("cam1")
        t, samples = await buf.latest_audio(0.3)
        assert t == 0.0
        assert len(samples) == 0


# ══════════════════════════════════════════════════════════════════════════════
# §3 — signal_bus.py
# ══════════════════════════════════════════════════════════════════════════════


class TestSignalBus:
    @pytest.mark.asyncio
    async def test_publish_subscribe_roundtrip(self):
        bus = SignalBus()
        received = []

        async def consume():
            async for frame in bus.subscribe():
                received.append(frame)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)  # let consumer register

        frame = SignalFrame(cam_id="cam1", t=1.0, audio_rms=0.5)
        await bus.publish(frame)
        await asyncio.sleep(0)
        await bus.close()
        await task

        assert len(received) == 1
        assert received[0].cam_id == "cam1"
        assert received[0].audio_rms == 0.5

    @pytest.mark.asyncio
    async def test_latest_per_camera_one_entry_per_cam(self):
        bus = SignalBus()
        await bus.publish(SignalFrame(cam_id="cam_a", t=1.0))
        await bus.publish(SignalFrame(cam_id="cam_b", t=1.1))
        await bus.publish(SignalFrame(cam_id="cam_a", t=2.0, audio_rms=0.9))

        latest = await bus.latest_per_camera()
        assert set(latest.keys()) == {"cam_a", "cam_b"}
        # cam_a should reflect the second publish (t=2.0)
        assert latest["cam_a"].t == 2.0
        assert latest["cam_a"].audio_rms == 0.9

    @pytest.mark.asyncio
    async def test_queue_overflow_drops_oldest_first(self):
        """With maxsize=3 and 5 publishes the first 2 items are dropped."""
        bus = SignalBus(maxsize=3)
        received = []

        async def consume():
            async for frame in bus.subscribe():
                received.append(frame.t)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)

        for i in range(5):
            await bus.publish(SignalFrame(cam_id="c", t=float(i)))
        await asyncio.sleep(0)
        await bus.close()
        await task

        # Queue can hold at most 3; the first 2 were dropped
        assert len(received) <= 3
        # The most recent frame (t=4.0) must be present
        assert 4.0 in received

    @pytest.mark.asyncio
    async def test_multiple_subscribers_each_get_frame(self):
        bus = SignalBus()
        received_a, received_b = [], []

        async def consume(store):
            async for frame in bus.subscribe():
                store.append(frame.cam_id)

        ta = asyncio.create_task(consume(received_a))
        tb = asyncio.create_task(consume(received_b))
        await asyncio.sleep(0)

        await bus.publish(SignalFrame(cam_id="camX", t=0.0))
        await asyncio.sleep(0)
        await bus.close()
        await asyncio.gather(ta, tb)

        assert "camX" in received_a
        assert "camX" in received_b


# ══════════════════════════════════════════════════════════════════════════════
# §4 — ingest.py
# ══════════════════════════════════════════════════════════════════════════════


def _make_worker(event_id=1, cam_id="cam1", host="localhost", port=1935):
    cam = CameraConfig(id=cam_id, label="Test Cam")
    ring = CameraRingBuffer(cam_id)
    return IngestWorker(event_id=event_id, camera=cam, ring=ring,
                        rtmp_host=host, rtmp_port=port)


class TestIngestWorker:
    def test_rtmp_url(self):
        w = _make_worker(event_id=42, cam_id="cam_stage", host="10.0.0.1", port=1935)
        assert w.rtmp_url() == "rtmp://10.0.0.1:1935/live/42/cam_stage"

    def test_rtmp_url_default_host(self):
        w = _make_worker()
        assert w.rtmp_url() == "rtmp://localhost:1935/live/1/cam1"

    def test_constructor_stores_camera_and_event_id(self):
        cam = CameraConfig(id="cam_x", label="X")
        ring = CameraRingBuffer("cam_x")
        w = IngestWorker(event_id=99, camera=cam, ring=ring)
        assert w.event_id == 99
        assert w.camera is cam
        assert w.ring is ring

    def test_is_alive_false_before_start(self):
        w = _make_worker()
        assert w.is_alive is False

    def test_frame_size_math(self):
        """1080 × 1920 × 3 = 6 220 800 bytes per bgr24 frame."""
        w = _make_worker()
        assert w._frame_bytes() == 1080 * 1920 * 3
        assert w._frame_bytes() == 6_220_800

    def test_audio_chunk_size_constant(self):
        """320 ms @ 16 kHz mono int16 = 5120 samples = 10 240 bytes."""
        assert _AUDIO_CHUNK_SAMPLES == 5_120
        assert _AUDIO_BYTES_PER_CHUNK == 10_240

    def test_video_ffmpeg_cmd(self):
        w = _make_worker(event_id=7, cam_id="cam_crowd")
        cmd = w._build_video_cmd()
        assert "-hide_banner" in cmd
        assert "-loglevel" in cmd
        assert "warning" in cmd
        assert "-i" in cmd
        url_idx = cmd.index("-i") + 1
        assert cmd[url_idx] == "rtmp://localhost:1935/live/7/cam_crowd"
        assert "-f" in cmd
        assert "rawvideo" in cmd
        assert "-pix_fmt" in cmd
        assert "bgr24" in cmd
        assert "-s" in cmd
        # size arg is WxH
        s_idx = cmd.index("-s") + 1
        assert cmd[s_idx] == "1080x1920"
        assert "pipe:1" in cmd

    def test_audio_ffmpeg_cmd(self):
        w = _make_worker(event_id=7, cam_id="cam_crowd")
        cmd = w._build_audio_cmd()
        assert "-hide_banner" in cmd
        assert "-vn" in cmd
        assert "-f" in cmd
        assert "s16le" in cmd
        assert "-ac" in cmd
        ac_idx = cmd.index("-ac") + 1
        assert cmd[ac_idx] == "1"
        assert "-ar" in cmd
        ar_idx = cmd.index("-ar") + 1
        assert cmd[ar_idx] == "16000"
        assert "pipe:1" in cmd

    @pytest.mark.asyncio
    async def test_start_spawns_tasks(self):
        """start() creates two asyncio tasks (video + audio reader loops)."""
        w = _make_worker()

        # Patch _run_loop so tasks immediately return
        async def _noop(*args, **kwargs):
            pass

        with patch.object(w, "_run_loop", side_effect=_noop):
            await w.start()
            assert w._video_task is not None
            assert w._audio_task is not None
            await w.stop()

    @pytest.mark.asyncio
    async def test_start_called_twice_is_idempotent(self):
        """Second start() call is a no-op (logs warning, no duplicate tasks)."""
        w = _make_worker()

        async def _noop(*args, **kwargs):
            pass

        with patch.object(w, "_run_loop", side_effect=_noop):
            await w.start()
            first_video_task = w._video_task
            await w.start()   # second call
            assert w._video_task is first_video_task  # same task object
            await w.stop()

    @pytest.mark.asyncio
    async def test_subprocess_death_triggers_restart_with_backoff(self):
        """_run_loop should call the reader at least twice on repeated failures.

        _spawn_and_read is actually invoked as _spawn_and_read(reader_coro, kind)
        — two args — so our fake must match that signature or the TypeError
        prevents _running from ever being flipped and the loop spins until
        pytest timeout.
        """
        w = _make_worker()
        call_count = 0

        async def _failing_spawn(reader_coro, kind):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("simulated subprocess death")
            # On third call, signal stop so the loop exits.
            w._running = False

        async def _instant_sleep(secs):
            pass  # skip actual backoff delay in tests

        with patch.object(w, "_spawn_and_read", side_effect=_failing_spawn), \
             patch("asyncio.sleep", side_effect=_instant_sleep):
            w._running = True
            # reader_coro is irrelevant — our mock replaces _spawn_and_read
            # which is the only consumer of it.
            await w._run_loop(None, "video")

        assert call_count >= 2, "reader must be retried after crash"

    @pytest.mark.asyncio
    async def test_frame_size_matches_one_push_video_call(self):
        """Reading exactly frame_bytes from stdout triggers exactly one push_video."""
        w = _make_worker()
        w._running = True
        w._event_start_t = 0.0

        frame_bytes = w._frame_bytes()
        raw = bytes(frame_bytes)  # zero-filled fake frame

        mock_proc = MagicMock()
        mock_proc.stdout = AsyncMock()
        mock_proc.stdout.readexactly = AsyncMock(side_effect=[raw, asyncio.CancelledError()])

        push_calls = []

        async def _fake_push(t, frame):
            push_calls.append((t, frame.shape))

        with patch.object(w.ring, "push_video", side_effect=_fake_push):
            try:
                await w._read_video(mock_proc)
            except asyncio.CancelledError:
                pass

        assert len(push_calls) == 1
        _, shape = push_calls[0]
        assert shape == (1920, 1080, 3)
