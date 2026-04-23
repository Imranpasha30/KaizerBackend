"""Phase 6.4 Composer — tests for pure + subprocess-mocked paths."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline_core.live_director.composer import Composer, ComposerConfig
from pipeline_core.live_director.signals import CameraSelection


# ══════════════════════════════════════════════════════════════════════════════
# ComposerConfig
# ══════════════════════════════════════════════════════════════════════════════


class TestComposerConfig:
    def test_defaults(self):
        c = ComposerConfig()
        assert c.segment_seconds == 600
        assert c.bitrate_kbps == 6000
        assert c.audio_bitrate_kbps == 192
        assert c.width == 1920
        assert c.height == 1080
        assert c.fps == 30
        assert c.include_cue_sheet is True


# ══════════════════════════════════════════════════════════════════════════════
# build_streamselect_filter — pure string construction
# ══════════════════════════════════════════════════════════════════════════════


class TestStreamselectFilter:
    def test_two_cameras(self):
        c = Composer()
        flt = c.build_streamselect_filter(2)
        assert "[0:v][1:v]streamselect=inputs=2:map=0[vout]" in flt
        assert "[0:a][1:a]astreamselect=inputs=2:map=0[aout]" in flt

    def test_four_cameras(self):
        c = Composer()
        flt = c.build_streamselect_filter(4)
        assert "streamselect=inputs=4" in flt
        assert "astreamselect=inputs=4" in flt
        # All pads present
        for i in range(4):
            assert f"[{i}:v]" in flt
            assert f"[{i}:a]" in flt

    def test_zero_cameras_raises(self):
        c = Composer()
        with pytest.raises(ValueError, match="at least 1 camera"):
            c.build_streamselect_filter(0)


# ══════════════════════════════════════════════════════════════════════════════
# build_ffmpeg_cmd — full command assembly
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildFFmpegCmd:
    def test_two_cameras_no_rtmp_push(self):
        c = Composer()
        cmd = c.build_ffmpeg_cmd(
            camera_rtmp_urls=[
                "rtmp://localhost:1935/live/42/cam1",
                "rtmp://localhost:1935/live/42/cam2",
            ],
            output_path_template="/tmp/event_42/program_%03d.mp4",
        )
        # ffmpeg binary invoked
        assert cmd[0] == "ffmpeg"
        # Both inputs present
        assert cmd.count("-i") == 2
        assert "rtmp://localhost:1935/live/42/cam1" in cmd
        assert "rtmp://localhost:1935/live/42/cam2" in cmd
        # filter_complex present
        assert "-filter_complex" in cmd
        # segmented output
        assert "-f" in cmd
        assert "segment" in cmd
        assert "/tmp/event_42/program_%03d.mp4" in cmd
        # No second sink
        assert "flv" not in cmd

    def test_with_rtmp_push_adds_second_sink(self):
        c = Composer()
        cmd = c.build_ffmpeg_cmd(
            camera_rtmp_urls=["rtmp://x/live/1"],
            output_path_template="/tmp/prog_%03d.mp4",
            rtmp_push_url="rtmp://a.rtmp.youtube.com/live2/XYZ",
        )
        assert "rtmp://a.rtmp.youtube.com/live2/XYZ" in cmd
        assert "flv" in cmd

    def test_no_cameras_raises(self):
        c = Composer()
        with pytest.raises(ValueError, match="at least one camera url"):
            c.build_ffmpeg_cmd(camera_rtmp_urls=[], output_path_template="/tmp/x.mp4")


# ══════════════════════════════════════════════════════════════════════════════
# selection_to_sendcmd — serialisation
# ══════════════════════════════════════════════════════════════════════════════


class TestSelectionToSendcmd:
    def test_first_camera_map_0(self):
        c = Composer()
        sel = CameraSelection(t=5.0, cam_id="cam1", transition="cut",
                              confidence=0.9, reason="speaker")
        line = c.selection_to_sendcmd(sel, ["cam1", "cam2", "cam3"])
        assert "streamselect map 0" in line
        assert "astreamselect map 0" in line
        assert line.endswith("\n")

    def test_second_camera_map_1(self):
        c = Composer()
        sel = CameraSelection(t=5.0, cam_id="cam2", transition="cut",
                              confidence=0.9, reason="")
        line = c.selection_to_sendcmd(sel, ["cam1", "cam2", "cam3"])
        assert "streamselect map 1" in line
        assert "astreamselect map 1" in line

    def test_unknown_camera_raises(self):
        c = Composer()
        sel = CameraSelection(t=5.0, cam_id="cam_ghost", transition="cut",
                              confidence=0.9, reason="")
        with pytest.raises(ValueError, match="not in camera_ids"):
            c.selection_to_sendcmd(sel, ["cam1", "cam2"])


# ══════════════════════════════════════════════════════════════════════════════
# Cue-sheet lifecycle
# ══════════════════════════════════════════════════════════════════════════════


class TestCueSheet:
    def test_write_cue_sheet_empty_run_raises(self):
        c = Composer()
        with pytest.raises(RuntimeError):
            c.write_cue_sheet("/tmp/dummy.json")

    def test_write_cue_sheet_after_selections(self, tmp_path):
        c = Composer()
        # Manually populate a run state (no ffmpeg)
        from pipeline_core.live_director.composer import ComposerRunState
        c._state = ComposerRunState(
            event_id=1,
            camera_ids=["cam1", "cam2"],
            output_dir=str(tmp_path),
            started_at=100.0,
        )
        # Simulate two selections
        for t, cam in ((5.0, "cam1"), (10.5, "cam2")):
            sel = CameraSelection(t=t, cam_id=cam, transition="cut",
                                  confidence=0.8, reason="test")
            c._state.cues.append(
                __import__("pipeline_core.live_director.composer",
                           fromlist=["_CueEntry"])._CueEntry(
                    t=sel.t, cam_id=sel.cam_id, transition=sel.transition,
                    confidence=sel.confidence, reason=sel.reason,
                )
            )
        out = tmp_path / "cues.json"
        c.write_cue_sheet(str(out))
        payload = json.loads(out.read_text())
        assert payload["event_id"] == 1
        assert len(payload["cues"]) == 2
        assert payload["cues"][0]["cam_id"] == "cam1"
        assert payload["cues"][1]["cam_id"] == "cam2"


# ══════════════════════════════════════════════════════════════════════════════
# Live subprocess — mocked
# ══════════════════════════════════════════════════════════════════════════════


class TestStartLive:
    @pytest.mark.asyncio
    async def test_start_live_spawns_subprocess(self, tmp_path, monkeypatch):
        c = Composer()
        fake_proc = MagicMock()
        fake_proc.stdin = MagicMock()
        fake_proc.stdin.write = MagicMock()
        fake_proc.stdin.drain = AsyncMock()

        async def _fake_create(*args, **kwargs):
            return fake_proc

        monkeypatch.setattr(
            "asyncio.create_subprocess_exec", AsyncMock(side_effect=_fake_create),
        )

        await c.start_live(
            event_id=42,
            camera_ids=["cam1", "cam2"],
            camera_rtmp_urls=["rtmp://x/cam1", "rtmp://x/cam2"],
            output_dir=str(tmp_path),
        )
        assert c._running is True
        assert c._state is not None
        assert c._state.event_id == 42

    @pytest.mark.asyncio
    async def test_start_live_url_mismatch_raises(self):
        c = Composer()
        with pytest.raises(ValueError, match="length mismatch"):
            await c.start_live(
                event_id=42,
                camera_ids=["cam1", "cam2"],
                camera_rtmp_urls=["rtmp://x/cam1"],  # only 1 URL for 2 cams
                output_dir="/tmp/whatever",
            )


class TestApplySelection:
    @pytest.mark.asyncio
    async def test_apply_selection_writes_sendcmd_and_records_cue(self, tmp_path):
        c = Composer()
        # Fake running state (no real ffmpeg)
        from pipeline_core.live_director.composer import ComposerRunState
        c._state = ComposerRunState(
            event_id=1,
            camera_ids=["cam1", "cam2"],
            output_dir=str(tmp_path),
        )
        fake_proc = MagicMock()
        fake_proc.stdin = MagicMock()
        fake_proc.stdin.write = MagicMock()
        fake_proc.stdin.drain = AsyncMock()
        c._proc = fake_proc

        sel = CameraSelection(t=5.0, cam_id="cam2", transition="cut",
                              confidence=0.9, reason="speaker")
        await c.apply_selection(sel)

        # stdin.write called with the sendcmd line
        fake_proc.stdin.write.assert_called_once()
        written = fake_proc.stdin.write.call_args.args[0]
        assert b"streamselect map 1" in written
        assert b"astreamselect map 1" in written
        # Cue recorded
        assert len(c._state.cues) == 1
        assert c._state.cues[0].cam_id == "cam2"

    @pytest.mark.asyncio
    async def test_apply_selection_unknown_cam_logs_skip(self, tmp_path):
        c = Composer()
        from pipeline_core.live_director.composer import ComposerRunState
        c._state = ComposerRunState(
            event_id=1,
            camera_ids=["cam1", "cam2"],
            output_dir=str(tmp_path),
        )
        fake_proc = MagicMock()
        fake_proc.stdin = MagicMock()
        fake_proc.stdin.write = MagicMock()
        fake_proc.stdin.drain = AsyncMock()
        c._proc = fake_proc

        sel = CameraSelection(t=5.0, cam_id="cam_ghost", transition="cut",
                              confidence=0.9, reason="")
        # Should not raise
        await c.apply_selection(sel)
        # No stdin write
        fake_proc.stdin.write.assert_not_called()

    @pytest.mark.asyncio
    async def test_apply_selection_before_start_live_logs_ignore(self):
        c = Composer()  # no start_live called → no state
        sel = CameraSelection(t=5.0, cam_id="cam1", transition="cut",
                              confidence=0.9, reason="")
        # Should not raise, just log
        await c.apply_selection(sel)
