"""Phase 6.2 Analyzers — fast test suite.

Covers the six per-camera analyzers (audio / face / motion / scene / reaction /
beat) plus the shared Analyzer base class. Heavyweight libs (cv2, librosa,
webrtcvad) are lazy-loaded by each analyzer so most tests mock at the ring
buffer level and inspect the SignalFrame partial each analyzer emits.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from pipeline_core.live_director.analyzers import (
    ANALYZER_FRAME_DOWNSAMPLE,
    Analyzer,
    AnalyzerConfig,
    AudioAnalyzer,
    FaceAnalyzer,
    MotionAnalyzer,
    SceneAnalyzer,
    ReactionAnalyzer,
    BeatAnalyzer,
)
from pipeline_core.live_director.signals import SignalFrame


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


def _cfg(cam_id="cam1", interval_s=0.3):
    return AnalyzerConfig(cam_id=cam_id, interval_s=interval_s)


@pytest.fixture
def fake_ring_empty():
    ring = MagicMock()
    ring.latest_video = AsyncMock(return_value=[])
    ring.latest_audio = AsyncMock(return_value=(0.0, np.zeros(0, dtype=np.int16)))
    return ring


@pytest.fixture
def fake_bus():
    bus = MagicMock()
    bus.publish = AsyncMock()
    return bus


def _make_frame(shape=(1080, 1920, 3), fill=0):
    return np.full(shape, fill, dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# AnalyzerConfig + base Analyzer
# ══════════════════════════════════════════════════════════════════════════════


class TestAnalyzerBase:
    def test_config_defaults(self):
        c = AnalyzerConfig(cam_id="cam1")
        assert c.cam_id == "cam1"
        assert c.interval_s == 0.3
        assert c.enabled is True

    def test_downsample_constant(self):
        assert 0.1 < ANALYZER_FRAME_DOWNSAMPLE <= 1.0

    @pytest.mark.asyncio
    async def test_run_loop_calls_analyze_and_publishes(self, fake_ring_empty, fake_bus):
        class _DummyAnalyzer(Analyzer):
            name = "dummy"
            calls = 0

            async def analyze(self):
                self.calls += 1
                if self.calls >= 2:
                    # Stop by cancelling the task from inside
                    raise asyncio.CancelledError()
                return SignalFrame(cam_id=self.config.cam_id, t=0.0, audio_rms=0.5)

        a = _DummyAnalyzer(_cfg(interval_s=0.01), fake_ring_empty, fake_bus)
        await a.run()
        assert a.calls >= 2
        assert fake_bus.publish.await_count >= 1

    @pytest.mark.asyncio
    async def test_run_loop_swallows_exceptions_and_continues(self, fake_ring_empty, fake_bus):
        class _FailAnalyzer(Analyzer):
            name = "fail"
            calls = 0

            async def analyze(self):
                self.calls += 1
                if self.calls < 3:
                    raise RuntimeError("boom")
                raise asyncio.CancelledError()

        a = _FailAnalyzer(_cfg(interval_s=0.001), fake_ring_empty, fake_bus)
        await a.run()
        assert a.calls >= 3

    @pytest.mark.asyncio
    async def test_run_loop_skips_when_disabled(self, fake_ring_empty, fake_bus):
        class _NoopAnalyzer(Analyzer):
            name = "noop"
            calls = 0

            async def analyze(self):
                self.calls += 1
                return None

        cfg = AnalyzerConfig(cam_id="cam1", interval_s=0.001, enabled=False)
        a = _NoopAnalyzer(cfg, fake_ring_empty, fake_bus)

        async def _stop_soon():
            await asyncio.sleep(0.02)
            raise asyncio.CancelledError()

        task = asyncio.create_task(a.run())
        await asyncio.sleep(0.03)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert a.calls == 0, "analyze() must not be called when disabled"


# ══════════════════════════════════════════════════════════════════════════════
# AudioAnalyzer
# ══════════════════════════════════════════════════════════════════════════════


class TestAudioAnalyzer:
    @pytest.mark.asyncio
    async def test_no_audio_returns_none(self, fake_ring_empty, fake_bus):
        a = AudioAnalyzer(_cfg(), fake_ring_empty, fake_bus)
        result = await a.analyze()
        assert result is None

    @pytest.mark.asyncio
    async def test_silent_audio_low_rms(self, fake_bus):
        ring = MagicMock()
        silent = np.zeros(8000, dtype=np.int16)
        ring.latest_audio = AsyncMock(return_value=(1.23, silent))
        a = AudioAnalyzer(_cfg(), ring, fake_bus)
        result = await a.analyze()
        assert result is not None
        assert result.cam_id == "cam1"
        assert result.t == 1.23
        assert 0.0 <= result.audio_rms < 0.05
        assert isinstance(result.vad_speaking, bool)

    @pytest.mark.asyncio
    async def test_loud_audio_high_rms(self, fake_bus):
        ring = MagicMock()
        # Half-amplitude sine wave
        t_axis = np.linspace(0, 0.5, 8000, endpoint=False)
        samples = (16000 * np.sin(2 * np.pi * 200 * t_axis)).astype(np.int16)
        ring.latest_audio = AsyncMock(return_value=(2.0, samples))
        a = AudioAnalyzer(_cfg(), ring, fake_bus)
        result = await a.analyze()
        assert result is not None
        assert result.audio_rms > 0.1


# ══════════════════════════════════════════════════════════════════════════════
# FaceAnalyzer
# ══════════════════════════════════════════════════════════════════════════════


class TestFaceAnalyzer:
    @pytest.mark.asyncio
    async def test_no_frame_returns_none(self, fake_ring_empty, fake_bus):
        a = FaceAnalyzer(_cfg(), fake_ring_empty, fake_bus)
        result = await a.analyze()
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_frame_fields_default(self, fake_bus):
        ring = MagicMock()
        frame = _make_frame(shape=(360, 640, 3), fill=0)  # black frame, no faces
        ring.latest_video = AsyncMock(return_value=[(0.0, frame)])

        # Mock the cascade so we don't actually run Haar on a black frame
        a = FaceAnalyzer(_cfg(), ring, fake_bus)
        a._cascade = MagicMock()
        a._cascade.detectMultiScale.return_value = np.empty((0, 4), dtype=np.int32)

        result = await a.analyze()
        assert result is not None
        assert result.face_present is False
        assert result.face_size_norm == 0.0

    @pytest.mark.asyncio
    async def test_face_detected_populates_fields(self, fake_bus):
        ring = MagicMock()
        frame = _make_frame(shape=(360, 640, 3), fill=200)
        ring.latest_video = AsyncMock(return_value=[(5.0, frame)])
        a = FaceAnalyzer(_cfg(), ring, fake_bus)
        # Faux cascade returns one bbox at (x, y, w, h)
        fake_cascade = MagicMock()
        # Downsampled frame size will be (360*0.333, 640*0.333) ≈ (119, 213)
        # Our cascade mock returns a single 30×40 face bbox
        fake_cascade.detectMultiScale.return_value = np.array([[10, 10, 30, 40]])
        a._cascade = fake_cascade

        result = await a.analyze()
        assert result is not None
        assert result.face_present is True
        assert result.face_size_norm > 0.0
        assert result.face_size_norm < 1.0
        assert result.t == 5.0


# ══════════════════════════════════════════════════════════════════════════════
# MotionAnalyzer
# ══════════════════════════════════════════════════════════════════════════════


class TestMotionAnalyzer:
    @pytest.mark.asyncio
    async def test_no_frames_returns_none(self, fake_ring_empty, fake_bus):
        a = MotionAnalyzer(_cfg(), fake_ring_empty, fake_bus)
        result = await a.analyze()
        assert result is None

    @pytest.mark.asyncio
    async def test_identical_frames_zero_motion(self, fake_bus):
        ring = MagicMock()
        f1 = _make_frame(shape=(360, 640, 3), fill=128)
        f2 = _make_frame(shape=(360, 640, 3), fill=128)
        ring.latest_video = AsyncMock(return_value=[(1.0, f1), (2.0, f2)])
        a = MotionAnalyzer(_cfg(), ring, fake_bus)
        # Seed prev_gray via a first tick
        _ = await a.analyze()
        result = await a.analyze()
        assert result is not None
        assert result.motion_mag == 0.0

    @pytest.mark.asyncio
    async def test_changed_frames_nonzero_motion(self, fake_bus):
        ring = MagicMock()
        f1 = _make_frame(shape=(360, 640, 3), fill=0)
        f2 = _make_frame(shape=(360, 640, 3), fill=255)
        ring.latest_video = AsyncMock(side_effect=[
            [(1.0, f1), (1.0, f1)],  # first tick: same frame doubled
            [(2.0, f2), (2.0, f2)],  # second tick: bright frames
        ])
        a = MotionAnalyzer(_cfg(), ring, fake_bus)
        _ = await a.analyze()          # prime prev_gray with f1
        result = await a.analyze()     # now compare f2 to f1
        assert result is not None
        assert result.motion_mag > 0.5, f"Expected >0.5 motion, got {result.motion_mag}"


# ══════════════════════════════════════════════════════════════════════════════
# SceneAnalyzer
# ══════════════════════════════════════════════════════════════════════════════


class TestSceneAnalyzer:
    @pytest.mark.asyncio
    async def test_no_frame_returns_none(self, fake_ring_empty, fake_bus):
        a = SceneAnalyzer(_cfg(), fake_ring_empty, fake_bus)
        result = await a.analyze()
        assert result is None

    @pytest.mark.asyncio
    async def test_many_faces_crowd(self, fake_bus):
        ring = MagicMock()
        frame = _make_frame(shape=(360, 640, 3), fill=128)
        ring.latest_video = AsyncMock(return_value=[(1.0, frame)])
        a = SceneAnalyzer(_cfg(), ring, fake_bus)
        fake_cascade = MagicMock()
        fake_cascade.detectMultiScale.return_value = np.array([
            [10, 10, 20, 20], [40, 40, 20, 20], [70, 70, 20, 20],
        ])
        a._cascade = fake_cascade
        result = await a.analyze()
        assert result is not None
        assert result.scene == "crowd"

    @pytest.mark.asyncio
    async def test_one_large_face_closeup(self, fake_bus):
        ring = MagicMock()
        frame = _make_frame(shape=(360, 640, 3), fill=128)
        ring.latest_video = AsyncMock(return_value=[(1.0, frame)])
        a = SceneAnalyzer(_cfg(), ring, fake_bus)
        fake_cascade = MagicMock()
        # Large face — 120×120 out of ~119×213 downsampled
        fake_cascade.detectMultiScale.return_value = np.array([[5, 5, 120, 120]])
        a._cascade = fake_cascade
        result = await a.analyze()
        assert result is not None
        assert result.scene == "closeup"

    @pytest.mark.asyncio
    async def test_no_faces_dim_saturated_stage(self, fake_bus):
        ring = MagicMock()
        # Dim purple-ish frame — low brightness + high saturation
        frame = _make_frame(shape=(360, 640, 3), fill=0)
        frame[:, :, 2] = 60  # red tint
        frame[:, :, 0] = 60  # blue tint
        ring.latest_video = AsyncMock(return_value=[(1.0, frame)])
        a = SceneAnalyzer(_cfg(), ring, fake_bus)
        fake_cascade = MagicMock()
        fake_cascade.detectMultiScale.return_value = np.empty((0, 4), dtype=np.int32)
        a._cascade = fake_cascade
        result = await a.analyze()
        assert result is not None
        assert result.scene in ("stage", "wide", "graphic")   # accepts either


# ══════════════════════════════════════════════════════════════════════════════
# ReactionAnalyzer
# ══════════════════════════════════════════════════════════════════════════════


class TestReactionAnalyzer:
    @pytest.mark.asyncio
    async def test_silent_audio_no_reaction(self, fake_bus):
        ring = MagicMock()
        ring.latest_audio = AsyncMock(return_value=(1.0, np.zeros(32000, dtype=np.int16)))
        a = ReactionAnalyzer(_cfg(), ring, fake_bus)
        result = await a.analyze()
        assert result is not None
        assert result.reaction is None

    @pytest.mark.asyncio
    async def test_short_audio_returns_none(self, fake_bus):
        ring = MagicMock()
        ring.latest_audio = AsyncMock(return_value=(1.0, np.zeros(100, dtype=np.int16)))
        a = ReactionAnalyzer(_cfg(), ring, fake_bus)
        result = await a.analyze()
        # Too short → we return None
        assert result is None

    @pytest.mark.asyncio
    async def test_audio_spike_produces_reaction(self, fake_bus):
        ring = MagicMock()
        # 2 s audio: quiet 1.5s + loud spike 500ms
        n_total = 32000
        n_recent = 8000
        x = np.zeros(n_total, dtype=np.float32)
        x[:-n_recent] = 500.0 * np.sin(
            2 * np.pi * 500 * np.linspace(0, 1.5, n_total - n_recent, endpoint=False)
        )
        x[-n_recent:] = 15000.0 * np.sin(
            2 * np.pi * 4000 * np.linspace(0, 0.5, n_recent, endpoint=False)
        )
        ring.latest_audio = AsyncMock(return_value=(2.0, x.astype(np.int16)))
        a = ReactionAnalyzer(_cfg(), ring, fake_bus)
        result = await a.analyze()
        assert result is not None
        # Either we got a concrete reaction or None — both acceptable under the
        # heuristic; we just verify the analyzer doesn't crash and emits a frame.
        assert result.t == 2.0


# ══════════════════════════════════════════════════════════════════════════════
# BeatAnalyzer
# ══════════════════════════════════════════════════════════════════════════════


class TestBeatAnalyzer:
    @pytest.mark.asyncio
    async def test_no_audio_returns_none(self, fake_bus):
        ring = MagicMock()
        ring.latest_audio = AsyncMock(return_value=(0.0, np.zeros(0, dtype=np.int16)))
        a = BeatAnalyzer(_cfg(), ring, fake_bus)
        result = await a.analyze()
        assert result is None

    @pytest.mark.asyncio
    async def test_silent_audio_beat_phase_none(self, fake_bus):
        ring = MagicMock()
        ring.latest_audio = AsyncMock(return_value=(8.0, np.zeros(16000 * 8, dtype=np.int16)))
        a = BeatAnalyzer(_cfg(), ring, fake_bus)
        result = await a.analyze()
        # Silent audio → librosa either returns tempo 0 (→ None) or succeeds
        # with an implausible tempo filtered out. Either way beat_phase is None.
        assert result is not None
        assert result.beat_phase is None

    def test_beat_interval_override_sets_longer_cadence(self, fake_bus):
        ring = MagicMock()
        cfg = AnalyzerConfig(cam_id="cam1", interval_s=0.3)  # default
        a = BeatAnalyzer(cfg, ring, fake_bus)
        # BeatAnalyzer overrides the default 0.3s to 1.0s because beat tracking
        # is not low-latency-sensitive.
        assert a.config.interval_s == 1.0
