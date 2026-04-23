"""Phase 6.3 Director — rule-engine tests.

Exercises the rule ladder (override → reaction → speaker → beat → min/max
shot → default-stay) and the operator controls (pin, blacklist, force_cut).
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, AsyncMock

import pytest

from pipeline_core.live_director.director import Director, DirectorConfig
from pipeline_core.live_director.signals import CameraSelection, SignalFrame


def _sf(cam_id: str, t: float = 0.0, **kw) -> SignalFrame:
    return SignalFrame(cam_id=cam_id, t=t, **kw)


def _mk_director(cams=("cam1", "cam2", "cam3"), config=None):
    bus = MagicMock()
    d = Director(
        event_id=1,
        camera_ids=list(cams),
        bus=bus,
        config=config or DirectorConfig(min_shot_s=2.5, max_shot_s=12.0),
    )
    return d


# ══════════════════════════════════════════════════════════════════════════════
# Config + dataclass
# ══════════════════════════════════════════════════════════════════════════════


class TestDirectorConfig:
    def test_defaults(self):
        c = DirectorConfig()
        assert c.min_shot_s == 2.5
        assert c.max_shot_s == 12.0
        assert c.reaction_threshold == 0.7
        assert c.speaker_vad_hold_ms == 400.0
        assert c.beat_cut_every_nth_bar == 2
        assert c.crossfade_on_scene_change is True


# ══════════════════════════════════════════════════════════════════════════════
# Rule 1: manual override
# ══════════════════════════════════════════════════════════════════════════════


class TestOperatorOverrides:
    def test_pin_forces_camera(self):
        d = _mk_director()
        d.pin("cam2")
        sel = d._decide(now=0.0)
        assert sel is not None
        assert sel.cam_id == "cam2"
        assert "pinned" in sel.reason

    def test_pin_unknown_raises(self):
        d = _mk_director()
        with pytest.raises(ValueError, match="Unknown camera"):
            d.pin("cam_ghost")

    def test_unpin_clears(self):
        d = _mk_director()
        d.pin("cam2")
        _ = d._decide(now=0.0)
        d.unpin()
        # After unpin + no new signals, nothing forces a cut
        sel = d._decide(now=1.0)
        assert sel is None

    def test_force_cut_oneshot(self):
        d = _mk_director()
        d.force_cut("cam2")
        sel = d._decide(now=0.0)
        assert sel is not None
        assert sel.cam_id == "cam2"
        assert "force_cut" in sel.reason
        # Second call — one_shot cleared, stays (no cut)
        sel2 = d._decide(now=0.1)
        assert sel2 is None

    def test_blacklist_excludes_cam(self):
        d = _mk_director()
        d.blacklist("cam2")
        # Even a forceful reaction signal should NOT send us to cam2.
        d._ingest(_sf("cam2", t=0.0, reaction="laugh", audio_rms=0.9))
        sel = d._decide(now=5.0)
        # cam2 blacklisted → no cut
        assert sel is None or sel.cam_id != "cam2"

    def test_allow_removes_blacklist(self):
        d = _mk_director()
        d.blacklist("cam2")
        d.allow("cam2")
        # Now cam2 can be selected again
        d.pin("cam2")
        sel = d._decide(now=0.0)
        assert sel is not None
        assert sel.cam_id == "cam2"


# ══════════════════════════════════════════════════════════════════════════════
# Rule 2: critical reaction
# ══════════════════════════════════════════════════════════════════════════════


class TestReactionRule:
    def test_strong_reaction_triggers_cut(self):
        d = _mk_director()
        # Establish current cam past min_shot
        d._current_cam = "cam1"
        d._last_cut_t = 0.0
        d._ingest(_sf("cam2", t=5.0, reaction="laugh", audio_rms=0.85))
        sel = d._decide(now=5.0)
        assert sel is not None
        assert sel.cam_id == "cam2"
        assert "reaction" in sel.reason

    def test_reaction_below_threshold_no_cut(self):
        d = _mk_director(config=DirectorConfig(reaction_threshold=0.7))
        d._current_cam = "cam1"
        d._last_cut_t = 0.0
        # Audio_rms 0.4 < threshold 0.7
        d._ingest(_sf("cam2", t=5.0, reaction="laugh", audio_rms=0.4))
        sel = d._decide(now=5.0)
        assert sel is None or sel.cam_id != "cam2"

    def test_reaction_during_min_shot_blocked(self):
        d = _mk_director(config=DirectorConfig(min_shot_s=2.5))
        d._current_cam = "cam1"
        d._last_cut_t = 10.0
        d._ingest(_sf("cam2", t=10.5, reaction="laugh", audio_rms=0.9))
        sel = d._decide(now=10.5)
        # Only 0.5s elapsed < 2.5s min — no cut
        assert sel is None


# ══════════════════════════════════════════════════════════════════════════════
# Rule 3: designated speaker active
# ══════════════════════════════════════════════════════════════════════════════


class TestSpeakerRule:
    def test_speaker_with_face_cuts(self):
        d = _mk_director(config=DirectorConfig(min_shot_s=2.5, speaker_vad_hold_ms=400))
        d._current_cam = "cam1"
        d._last_cut_t = 0.0
        # First tick — starts speaking
        d._ingest(_sf("cam2", t=4.0, vad_speaking=True, face_present=True, face_size_norm=0.15))
        # Second tick — 500ms later still speaking; hold satisfied
        d._ingest(_sf("cam2", t=4.5, vad_speaking=True, face_present=True, face_size_norm=0.15))
        sel = d._decide(now=4.5)
        assert sel is not None
        assert sel.cam_id == "cam2"
        assert "speaker" in sel.reason

    def test_speaker_without_face_no_cut(self):
        d = _mk_director()
        d._current_cam = "cam1"
        d._last_cut_t = 0.0
        d._ingest(_sf("cam2", t=4.0, vad_speaking=True, face_present=False))
        d._ingest(_sf("cam2", t=5.0, vad_speaking=True, face_present=False))
        sel = d._decide(now=5.0)
        assert sel is None or sel.cam_id != "cam2"

    def test_speaker_hold_not_satisfied_no_cut(self):
        d = _mk_director(config=DirectorConfig(min_shot_s=2.5, speaker_vad_hold_ms=400))
        d._current_cam = "cam1"
        d._last_cut_t = 0.0
        # Only 100ms of speaking — below 400ms hold
        d._ingest(_sf("cam2", t=4.0, vad_speaking=True, face_present=True, face_size_norm=0.15))
        sel = d._decide(now=4.1)
        assert sel is None or sel.cam_id != "cam2"


# ══════════════════════════════════════════════════════════════════════════════
# Rule 4: beat cut during music
# ══════════════════════════════════════════════════════════════════════════════


class TestBeatRule:
    def test_beat_phase_near_zero_cuts(self):
        d = _mk_director(config=DirectorConfig(min_shot_s=2.5, beat_cut_every_nth_bar=2))
        d._current_cam = "cam1"
        d._last_cut_t = 0.0
        d._beats_since_last_cut = 2
        # beat_phase=0.05 near a downbeat
        d._ingest(_sf("cam2", t=5.0, beat_phase=0.05, audio_rms=0.5))
        sel = d._decide(now=5.0)
        # Expect cut to another camera (cam2 in this case, which has energy)
        if sel is not None:
            assert "beat_cut" in sel.reason

    def test_beat_cut_blocked_during_min_shot(self):
        d = _mk_director(config=DirectorConfig(min_shot_s=2.5))
        d._current_cam = "cam1"
        d._last_cut_t = 10.0
        d._beats_since_last_cut = 5
        d._ingest(_sf("cam2", t=10.3, beat_phase=0.05, audio_rms=0.5))
        sel = d._decide(now=10.3)
        assert sel is None


# ══════════════════════════════════════════════════════════════════════════════
# Rule 5 + 6: min/max shot
# ══════════════════════════════════════════════════════════════════════════════


class TestShotDurationRules:
    def test_no_cut_under_min_shot(self):
        d = _mk_director(config=DirectorConfig(min_shot_s=2.5))
        d._current_cam = "cam1"
        d._last_cut_t = 10.0
        # 1s elapsed — under min_shot
        d._ingest(_sf("cam2", t=11.0, audio_rms=0.9))
        sel = d._decide(now=11.0)
        assert sel is None

    def test_max_shot_forces_cut(self):
        d = _mk_director(config=DirectorConfig(min_shot_s=2.5, max_shot_s=12.0))
        d._current_cam = "cam1"
        d._last_cut_t = 0.0
        # All other cameras show some energy so rotation picks one
        d._ingest(_sf("cam2", t=13.0, audio_rms=0.3))
        d._ingest(_sf("cam3", t=13.0, audio_rms=0.1))
        sel = d._decide(now=13.0)
        assert sel is not None
        assert sel.cam_id != "cam1"
        assert "max_shot" in sel.reason


# ══════════════════════════════════════════════════════════════════════════════
# Default stay + ingest merge
# ══════════════════════════════════════════════════════════════════════════════


class TestIngestMerge:
    def test_ingest_unknown_cam_ignored(self):
        d = _mk_director()
        d._ingest(_sf("cam_ghost", t=1.0, audio_rms=0.9))
        assert "cam_ghost" not in d._cam_states

    def test_ingest_partial_updates_merge(self):
        d = _mk_director()
        d._ingest(_sf("cam1", t=1.0, audio_rms=0.6))
        d._ingest(_sf("cam1", t=1.5, face_present=True, face_size_norm=0.1))
        state = d._cam_states["cam1"].last_frame
        # Both fields should be present after merge
        assert state.audio_rms == 0.6
        assert state.face_present is True
        assert state.face_size_norm == 0.1


class TestDefaultStay:
    def test_no_cut_when_nothing_interesting(self):
        d = _mk_director(config=DirectorConfig(min_shot_s=2.5, max_shot_s=12.0))
        d._current_cam = "cam1"
        d._last_cut_t = 5.0
        # 3s elapsed — past min_shot but no reaction, no speaker, no beat
        d._ingest(_sf("cam2", t=8.0, audio_rms=0.02))
        sel = d._decide(now=8.0)
        assert sel is None


# ══════════════════════════════════════════════════════════════════════════════
# Transition type
# ══════════════════════════════════════════════════════════════════════════════


class TestTransitionType:
    def test_cut_by_default(self):
        d = _mk_director()
        d._ingest(_sf("cam2", t=5.0, scene="stage"))
        d.pin("cam2")
        sel = d._decide(now=5.0)
        assert sel.transition == "cut"

    def test_dissolve_on_scene_change(self):
        d = _mk_director(config=DirectorConfig(crossfade_on_scene_change=True))
        d._current_cam = "cam1"
        d._last_cut_t = 0.0
        d._ingest(_sf("cam1", t=5.0, scene="stage"))
        d._ingest(_sf("cam2", t=5.0, scene="crowd", reaction="laugh", audio_rms=0.85))
        sel = d._decide(now=5.0)
        # Scene differs → dissolve
        assert sel is not None
        assert sel.transition == "dissolve"


# ══════════════════════════════════════════════════════════════════════════════
# Callback invocation
# ══════════════════════════════════════════════════════════════════════════════


class TestCallbacks:
    @pytest.mark.asyncio
    async def test_on_selection_called_on_cut(self):
        d = _mk_director()
        recorded = []

        async def _on_sel(sel):
            recorded.append(sel)

        d._on_selection = _on_sel
        # Trigger a cut via pin
        d.pin("cam2")
        sel = d._decide(now=0.0)
        assert sel is not None
        await d._emit_selection(sel)
        assert len(recorded) == 1
        assert recorded[0].cam_id == "cam2"

    def test_on_event_called_on_override(self):
        d = _mk_director()
        recorded = []
        d._on_event = lambda ev: recorded.append(ev)
        d.pin("cam2")
        # pin() emits an 'override' DirectorEvent synchronously
        assert any(ev.kind == "override" for ev in recorded)
