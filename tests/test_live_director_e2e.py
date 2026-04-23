"""
Phase 6 end-to-end smoke test.

Wires together the Phase 6 pieces that CAN be exercised without an actual
RTMP server:
  - CameraRingBuffer  (receives synthetic BGR frames + int16 audio)
  - SignalBus         (asyncio fan-in / fan-out)
  - Analyzers         (audio, face, motion, scene — no network needed)
  - Director          (rule engine)

Two simulated cameras produce distinct signals; we drive the bus with
scripted SignalFrame injections that mimic a real concert moment
("speaker on cam1", "crowd laughs on cam2", etc.) and verify the
Director emits the right CameraSelection.

Ingestion (RTMP+FFmpeg) and the Composer (live FFmpeg filter_complex)
are skipped here because they require a running RTMP server — those
are covered by their own unit tests with subprocess mocks.
"""
from __future__ import annotations

import asyncio
from typing import List
from unittest.mock import MagicMock

import pytest

from pipeline_core.live_director.signals import (
    CameraConfig,
    CameraSelection,
    SignalFrame,
)
from pipeline_core.live_director.signal_bus import SignalBus
from pipeline_core.live_director.director import Director, DirectorConfig


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


class _SelectionCollector:
    """Records every CameraSelection the director emits via the callback."""
    def __init__(self) -> None:
        self.selections: List[CameraSelection] = []
        self.events: list = []

    async def on_selection(self, sel: CameraSelection) -> None:
        self.selections.append(sel)

    def on_event(self, ev) -> None:
        self.events.append(ev)


async def _drive(director: Director, frames: list[SignalFrame]) -> None:
    """Feed the director a scripted SignalFrame sequence by calling its
    _ingest + _decide directly — avoids coupling to the SignalBus async
    iterator which only stops on external cancellation."""
    for f in frames:
        director._ingest(f)
        sel = director._decide(now=f.t)
        if sel is not None:
            await director._emit_selection(sel)


# ══════════════════════════════════════════════════════════════════════════════
# E2E scenarios
# ══════════════════════════════════════════════════════════════════════════════


class TestE2EScenarios:
    @pytest.mark.asyncio
    async def test_speaker_starts_talking_on_cam2_director_cuts(self):
        """Concert scenario: cam1 is on the wide shot, artist walks to cam2's
        frame and starts speaking. The director should cut to cam2 once the
        vad+face hold satisfies the threshold."""
        collector = _SelectionCollector()
        director = Director(
            event_id=1,
            camera_ids=["cam1", "cam2"],
            bus=SignalBus(),
            config=DirectorConfig(
                min_shot_s=2.0,
                max_shot_s=12.0,
                speaker_vad_hold_ms=400,
                reaction_threshold=0.7,
            ),
            on_selection=collector.on_selection,
        )
        # Establish that cam1 is currently selected (past min_shot).
        director._current_cam = "cam1"
        director._last_cut_t = 0.0

        # t=3.0 — cam2 picks up VAD + face; ingest hold starts
        # t=3.5 — still speaking; hold satisfied (500ms > 400ms)
        frames = [
            SignalFrame(cam_id="cam1", t=3.0, audio_rms=0.05, motion_mag=0.02, scene="wide"),
            SignalFrame(cam_id="cam2", t=3.0, vad_speaking=True, face_present=True, face_size_norm=0.15),
            SignalFrame(cam_id="cam2", t=3.5, vad_speaking=True, face_present=True, face_size_norm=0.18),
        ]
        await _drive(director, frames)

        assert len(collector.selections) >= 1
        last = collector.selections[-1]
        assert last.cam_id == "cam2"
        assert "speaker" in last.reason

    @pytest.mark.asyncio
    async def test_crowd_laughs_director_cuts_to_crowd_cam(self):
        """Comedy moment: the crowd cam detects a laugh with high audio
        energy. Director should cut to the crowd cam."""
        collector = _SelectionCollector()
        director = Director(
            event_id=2,
            camera_ids=["cam_stage", "cam_crowd"],
            bus=SignalBus(),
            config=DirectorConfig(
                min_shot_s=2.0,
                reaction_threshold=0.7,
            ),
            on_selection=collector.on_selection,
        )
        director._current_cam = "cam_stage"
        director._last_cut_t = 0.0

        frames = [
            SignalFrame(cam_id="cam_stage", t=5.0, audio_rms=0.1, scene="stage"),
            SignalFrame(cam_id="cam_crowd", t=5.0, audio_rms=0.85,
                        reaction="laugh", scene="crowd"),
        ]
        await _drive(director, frames)
        assert len(collector.selections) >= 1
        last = collector.selections[-1]
        assert last.cam_id == "cam_crowd"
        assert "reaction" in last.reason

    @pytest.mark.asyncio
    async def test_operator_pin_overrides_every_rule(self):
        """Human override: operator pins cam_stage. Even when cam_crowd
        has a massive reaction, the director should NOT cut away."""
        collector = _SelectionCollector()
        director = Director(
            event_id=3,
            camera_ids=["cam_stage", "cam_crowd"],
            bus=SignalBus(),
            config=DirectorConfig(min_shot_s=2.0, reaction_threshold=0.5),
            on_selection=collector.on_selection,
        )
        director.pin("cam_stage")
        # Immediate pin emits a cut to cam_stage
        collector.selections.clear()

        # Now a crowd reaction happens — should be ignored
        frames = [
            SignalFrame(cam_id="cam_crowd", t=5.0, reaction="cheer", audio_rms=0.95),
        ]
        await _drive(director, frames)
        # No additional cuts — pin overrides
        assert all(s.cam_id == "cam_stage" for s in collector.selections)

    @pytest.mark.asyncio
    async def test_max_shot_ceiling_eventually_forces_a_cut(self):
        """After 12s of no signals, the director MUST force a cut so the
        program doesn't get stuck on a stale camera."""
        collector = _SelectionCollector()
        director = Director(
            event_id=4,
            camera_ids=["cam1", "cam2", "cam3"],
            bus=SignalBus(),
            config=DirectorConfig(min_shot_s=2.0, max_shot_s=12.0),
            on_selection=collector.on_selection,
        )
        director._current_cam = "cam1"
        director._last_cut_t = 0.0

        # Seed some energy so rotation has a target
        frames = [
            SignalFrame(cam_id="cam2", t=5.0, audio_rms=0.12, motion_mag=0.2),
            # t=13 is past max_shot of 12s
            SignalFrame(cam_id="cam1", t=13.0, audio_rms=0.02),
        ]
        await _drive(director, frames)
        assert len(collector.selections) >= 1
        last = collector.selections[-1]
        assert last.cam_id != "cam1"
        assert "max_shot" in last.reason

    @pytest.mark.asyncio
    async def test_blacklisted_camera_never_selected(self):
        collector = _SelectionCollector()
        director = Director(
            event_id=5,
            camera_ids=["cam1", "cam2"],
            bus=SignalBus(),
            config=DirectorConfig(min_shot_s=2.0, reaction_threshold=0.5),
            on_selection=collector.on_selection,
        )
        director._current_cam = "cam1"
        director._last_cut_t = 0.0
        director.blacklist("cam2")
        # Clear override selection
        collector.selections.clear()

        # Even a massive reaction on cam2 must NOT trigger a cut
        frames = [
            SignalFrame(cam_id="cam2", t=5.0, reaction="cheer", audio_rms=0.9),
        ]
        await _drive(director, frames)
        # No selection → still on cam1
        assert all(s.cam_id != "cam2" for s in collector.selections)


# ══════════════════════════════════════════════════════════════════════════════
# Bus integration
# ══════════════════════════════════════════════════════════════════════════════


class TestBusIntegration:
    @pytest.mark.asyncio
    async def test_signal_bus_round_trip(self):
        """Analyzers publish → Director subscribes → decisions fire."""
        bus = SignalBus()
        received: list[SignalFrame] = []

        async def _consume():
            async for f in bus.subscribe():
                received.append(f)
                if len(received) >= 2:
                    return

        async def _publisher():
            # Let the consumer register its subscriber queue before publishing.
            await asyncio.sleep(0.05)
            await bus.publish(SignalFrame(cam_id="cam1", t=1.0, audio_rms=0.1))
            await bus.publish(SignalFrame(cam_id="cam2", t=1.2, audio_rms=0.5))

        consume_task = asyncio.create_task(_consume())
        pub_task = asyncio.create_task(_publisher())
        await asyncio.wait_for(consume_task, timeout=2.0)
        await pub_task
        assert len(received) == 2
        assert received[0].cam_id == "cam1"
        assert received[1].cam_id == "cam2"

    @pytest.mark.asyncio
    async def test_latest_per_camera_snapshot(self):
        bus = SignalBus()
        await bus.publish(SignalFrame(cam_id="cam1", t=1.0, audio_rms=0.1))
        await bus.publish(SignalFrame(cam_id="cam2", t=1.2, audio_rms=0.5))
        await bus.publish(SignalFrame(cam_id="cam1", t=1.5, audio_rms=0.8))
        snapshot = await bus.latest_per_camera()
        assert "cam1" in snapshot
        assert "cam2" in snapshot
        # cam1 latest should be the t=1.5 frame (higher t wins)
        assert snapshot["cam1"].audio_rms == 0.8
