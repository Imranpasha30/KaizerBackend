"""Phase 6.5 Output — HLSSink + ISORecorder + OutputStack tests."""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline_core.live_director.output import (
    HLSSink,
    HLSConfig,
    ISORecorder,
    ISOConfig,
    OutputStack,
)
from pipeline_core.live_director.signals import CameraConfig


# ══════════════════════════════════════════════════════════════════════════════
# HLSSink
# ══════════════════════════════════════════════════════════════════════════════


class TestHLSSinkConfig:
    def test_defaults(self):
        c = HLSConfig(output_dir="/tmp/hls")
        assert c.segment_seconds == 4
        assert c.playlist_size == 6
        assert "delete_segments" in c.hls_flags


class TestHLSSinkBuildCmd:
    def test_cmd_has_hls_output(self, tmp_path):
        sink = HLSSink(
            input_url="rtmp://localhost:1935/live/42/program",
            config=HLSConfig(output_dir=str(tmp_path)),
        )
        cmd = sink.build_cmd()
        assert cmd[0] == "ffmpeg"
        assert "-i" in cmd
        assert "rtmp://localhost:1935/live/42/program" in cmd
        assert "-f" in cmd
        assert "hls" in cmd
        assert "-hls_time" in cmd
        assert "4" in cmd
        assert any("program.m3u8" in x for x in cmd)

    def test_cmd_stream_copies_no_reencode(self, tmp_path):
        sink = HLSSink("rtmp://x/y", HLSConfig(output_dir=str(tmp_path)))
        cmd = sink.build_cmd()
        assert "-c" in cmd
        idx = cmd.index("-c")
        assert cmd[idx + 1] == "copy"

    def test_playlist_path(self, tmp_path):
        sink = HLSSink("rtmp://x/y", HLSConfig(output_dir=str(tmp_path)))
        assert sink.playlist_path.endswith("program.m3u8")


class TestHLSSinkLifecycle:
    @pytest.mark.asyncio
    async def test_start_spawns_subprocess(self, tmp_path, monkeypatch):
        sink = HLSSink("rtmp://x/y", HLSConfig(output_dir=str(tmp_path)))
        fake_proc = MagicMock()
        fake_proc.wait = AsyncMock(return_value=0)
        fake_proc.terminate = MagicMock()

        async def _fake_create(*args, **kwargs):
            return fake_proc

        monkeypatch.setattr(
            "asyncio.create_subprocess_exec",
            AsyncMock(side_effect=_fake_create),
        )
        await sink.start()
        # Give the loop one chance to enter _run_loop
        import asyncio as _a
        await _a.sleep(0.01)
        await sink.stop()
        # After stop, no proc running
        assert sink._proc is None


# ══════════════════════════════════════════════════════════════════════════════
# ISORecorder
# ══════════════════════════════════════════════════════════════════════════════


def _cam(cid="cam1"):
    return CameraConfig(id=cid, label=f"Label {cid}")


class TestISORecorderConfig:
    def test_defaults(self, tmp_path):
        c = ISOConfig(
            camera=_cam("cam1"),
            rtmp_url="rtmp://x/cam1",
            output_dir=str(tmp_path),
        )
        assert c.segment_seconds == 600

    def test_camera_dir(self, tmp_path):
        rec = ISORecorder(ISOConfig(
            camera=_cam("cam1"), rtmp_url="rtmp://x/cam1",
            output_dir=str(tmp_path),
        ))
        expected = os.path.join(str(tmp_path), "iso", "cam1").replace("\\", "/")
        assert rec.camera_dir == expected


class TestISORecorderBuildCmd:
    def test_cmd_stream_copies(self, tmp_path):
        rec = ISORecorder(ISOConfig(
            camera=_cam("cam1"),
            rtmp_url="rtmp://x/cam1",
            output_dir=str(tmp_path),
        ))
        cmd = rec.build_cmd()
        assert cmd[0] == "ffmpeg"
        assert "rtmp://x/cam1" in cmd
        # Stream copy, no re-encode
        assert "-c" in cmd
        idx = cmd.index("-c")
        assert cmd[idx + 1] == "copy"
        # Segmented output
        assert "-f" in cmd
        assert "segment" in cmd
        # Segment format MP4
        assert "-segment_format" in cmd
        assert "mp4" in cmd

    def test_cmd_includes_segment_time(self, tmp_path):
        rec = ISORecorder(ISOConfig(
            camera=_cam("cam1"),
            rtmp_url="rtmp://x/cam1",
            output_dir=str(tmp_path),
            segment_seconds=300,
        ))
        cmd = rec.build_cmd()
        assert "-segment_time" in cmd
        idx = cmd.index("-segment_time")
        assert cmd[idx + 1] == "300"


class TestISORecorderLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop_round_trip(self, tmp_path, monkeypatch):
        rec = ISORecorder(ISOConfig(
            camera=_cam("cam1"),
            rtmp_url="rtmp://x/cam1",
            output_dir=str(tmp_path),
        ))
        fake_proc = MagicMock()
        fake_proc.wait = AsyncMock(return_value=0)
        fake_proc.terminate = MagicMock()

        async def _fake_create(*args, **kwargs):
            return fake_proc

        monkeypatch.setattr(
            "asyncio.create_subprocess_exec",
            AsyncMock(side_effect=_fake_create),
        )
        await rec.start()
        import asyncio as _a
        await _a.sleep(0.01)
        await rec.stop()
        assert rec._proc is None


# ══════════════════════════════════════════════════════════════════════════════
# OutputStack
# ══════════════════════════════════════════════════════════════════════════════


class TestOutputStack:
    @pytest.mark.asyncio
    async def test_start_stop_iterates_recorders(self, tmp_path, monkeypatch):
        fake_proc = MagicMock()
        fake_proc.wait = AsyncMock(return_value=0)
        fake_proc.terminate = MagicMock()

        async def _fake_create(*args, **kwargs):
            return fake_proc

        monkeypatch.setattr(
            "asyncio.create_subprocess_exec",
            AsyncMock(side_effect=_fake_create),
        )

        stack = OutputStack(
            composer=MagicMock(),
            hls=HLSSink("rtmp://x/y", HLSConfig(output_dir=str(tmp_path / "hls"))),
            iso_recorders=[
                ISORecorder(ISOConfig(
                    camera=_cam("cam1"),
                    rtmp_url="rtmp://x/cam1",
                    output_dir=str(tmp_path),
                )),
                ISORecorder(ISOConfig(
                    camera=_cam("cam2"),
                    rtmp_url="rtmp://x/cam2",
                    output_dir=str(tmp_path),
                )),
            ],
        )
        await stack.start_all()
        import asyncio as _a
        await _a.sleep(0.01)
        await stack.stop_all()
        assert stack.hls._proc is None
        for rec in stack.iso_recorders:
            assert rec._proc is None

    @pytest.mark.asyncio
    async def test_stack_optional_hls(self, tmp_path):
        stack = OutputStack(composer=MagicMock(), hls=None, iso_recorders=[])
        # Start/stop without any sinks — should just work
        await stack.start_all()
        await stack.stop_all()
