"""
Phase 7.6 — composer layout + chroma-injection tests.

Covers:
  - build_layout_filter for every layout (single/split/pip/quad/bridge)
  - _build_layout_with_chroma_labels — chroma output labels replace [K:v] pads
  - build_ffmpeg_cmd respects layout + bridge_asset_path
  - apply_selection: fast path (sendcmd) for single-cam cuts vs slow path
    (respawn) for layout changes
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline_core.live_director.chroma import ChromaConfig
from pipeline_core.live_director.composer import (
    Composer,
    ComposerConfig,
    _build_layout_with_chroma_labels,
    build_layout_filter,
)
from pipeline_core.live_director.signals import CameraSelection


# ── Pure filter graph builders ─────────────────────────────────────────────


class TestBuildLayoutFilter:
    def test_single_uses_streamselect_so_sendcmd_can_swap(self):
        graph = build_layout_filter("single", [0], 2, 1920, 1080)
        assert "streamselect=inputs=2:map=0" in graph
        assert "astreamselect=inputs=2:map=0" in graph
        assert "[vout]" in graph and "[aout]" in graph

    def test_single_respects_primary_cam_index(self):
        graph = build_layout_filter("single", [1], 2, 1920, 1080)
        assert "streamselect=inputs=2:map=1" in graph

    def test_split2_hstack_two_cams(self):
        graph = build_layout_filter("split2_hstack", [0, 1], 2, 1920, 1080)
        assert "hstack=inputs=2" in graph
        # Each half is width/2 x height
        assert "scale=960:1080" in graph
        assert "[vout]" in graph

    def test_split2_hstack_wrong_arity_raises(self):
        with pytest.raises(ValueError):
            build_layout_filter("split2_hstack", [0], 2, 1920, 1080)

    def test_split2_vstack_halves_height(self):
        graph = build_layout_filter("split2_vstack", [0, 1], 2, 1920, 1080)
        assert "vstack=inputs=2" in graph
        assert "scale=1920:540" in graph

    def test_pip_overlay_bottom_right(self):
        graph = build_layout_filter("pip", [0, 1], 2, 1920, 1080)
        assert "overlay=" in graph
        # PIP size ≈ 1/4 frame
        assert "scale=480:270" in graph
        # Primary still audio
        assert "[0:a]acopy[aout]" in graph

    def test_pip_wrong_arity_raises(self):
        with pytest.raises(ValueError):
            build_layout_filter("pip", [0, 1, 2], 3, 1920, 1080)

    def test_quad_xstack_with_2x2_layout(self):
        graph = build_layout_filter("quad", [0, 1, 2, 3], 4, 1920, 1080)
        assert "xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0" in graph
        # Each cell is width/2 x height/2
        assert "scale=960:540" in graph

    def test_quad_wrong_arity_raises(self):
        with pytest.raises(ValueError):
            build_layout_filter("quad", [0, 1, 2], 3, 1920, 1080)

    def test_bridge_scales_asset_to_full_frame(self):
        graph = build_layout_filter(
            "bridge", [0], 2, 1920, 1080, bridge_input_idx=2,
        )
        assert "scale=1920:1080" in graph
        assert "crop=1920:1080" in graph
        assert "[2:v]" in graph
        assert "[vout]" in graph

    def test_bridge_without_input_idx_raises(self):
        with pytest.raises(ValueError):
            build_layout_filter("bridge", [0], 2, 1920, 1080)

    def test_unknown_layout_raises(self):
        with pytest.raises(ValueError, match="unknown layout"):
            build_layout_filter("holodeck", [0], 1, 1920, 1080)


# ── Chroma-labelled variant ─────────────────────────────────────────────────


class TestBuildLayoutWithChromaLabels:
    def test_single_uses_chroma_pads_instead_of_raw_video(self):
        graph = _build_layout_with_chroma_labels(
            "single", [1], 2, 1920, 1080,
            chroma_labels=["kchroma_0", "kchroma_1"],
        )
        assert "[kchroma_0]" in graph
        assert "[kchroma_1]" in graph
        # Audio still comes from raw pads
        assert "[0:a]" in graph and "[1:a]" in graph
        assert "streamselect=inputs=2:map=1[vout]" in graph

    def test_split2_hstack_uses_chroma_pads(self):
        graph = _build_layout_with_chroma_labels(
            "split2_hstack", [0, 1], 2, 1920, 1080,
            chroma_labels=["kchroma_0", "kchroma_1"],
        )
        assert "[kchroma_0]scale=960:1080" in graph
        assert "[kchroma_1]scale=960:1080" in graph
        assert "hstack=inputs=2" in graph
        # Audio from primary cam's raw pad
        assert "[0:a]acopy[aout]" in graph

    def test_pip_uses_chroma_pads_for_main_and_pip(self):
        graph = _build_layout_with_chroma_labels(
            "pip", [0, 1], 2, 1920, 1080,
            chroma_labels=["kchroma_0", "kchroma_1"],
        )
        assert "[kchroma_0]scale=1920:1080" in graph
        assert "[kchroma_1]scale=480:270" in graph
        assert "overlay=" in graph

    def test_quad_uses_chroma_pads_for_all_four(self):
        graph = _build_layout_with_chroma_labels(
            "quad", [0, 1, 2, 3], 4, 1920, 1080,
            chroma_labels=[f"kchroma_{i}" for i in range(4)],
        )
        for i in range(4):
            assert f"[kchroma_{i}]scale=960:540" in graph
        assert "xstack=inputs=4" in graph

    def test_bridge_ignores_chroma_and_delegates(self):
        graph = _build_layout_with_chroma_labels(
            "bridge", [0], 2, 1920, 1080,
            chroma_labels=["kchroma_0", "kchroma_1"],
            bridge_input_idx=3,
        )
        # Bridge uses raw bridge asset input, not chroma-labelled pads.
        assert "[3:v]" in graph


# ── build_ffmpeg_cmd with layout + bridge + chroma ──────────────────────────


class TestBuildFFmpegCmdLayout:
    def test_default_single_layout_when_not_specified(self):
        c = Composer(ComposerConfig())
        cmd = c.build_ffmpeg_cmd(
            ["rtmp://x/cam1", "rtmp://x/cam2"],
            "/tmp/p_%03d.mp4",
        )
        joined = " ".join(cmd)
        assert "streamselect=inputs=2:map=0" in joined

    def test_layout_split_emits_hstack_in_filter_complex(self):
        c = Composer(ComposerConfig())
        cmd = c.build_ffmpeg_cmd(
            ["rtmp://x/a", "rtmp://x/b"],
            "/tmp/p_%03d.mp4",
            layout="split2_hstack",
            layout_cam_indices=[0, 1],
        )
        joined = " ".join(cmd)
        assert "hstack=inputs=2" in joined

    def test_layout_bridge_requires_asset_path(self):
        c = Composer(ComposerConfig())
        with pytest.raises(ValueError, match="bridge_asset_path"):
            c.build_ffmpeg_cmd(
                ["rtmp://x/cam1"],
                "/tmp/p_%03d.mp4",
                layout="bridge",
                layout_cam_indices=[0],
                bridge_asset_path="",
            )

    def test_layout_bridge_image_asset_uses_loop_flag(self):
        c = Composer(ComposerConfig())
        cmd = c.build_ffmpeg_cmd(
            ["rtmp://x/cam1"],
            "/tmp/p_%03d.mp4",
            layout="bridge",
            layout_cam_indices=[0],
            bridge_asset_path="/assets/title.png",
        )
        # -loop 1 appears once (for the bridge image input)
        assert cmd.count("-loop") == 1
        assert "/assets/title.png" in cmd

    def test_layout_bridge_video_asset_uses_stream_loop(self):
        c = Composer(ComposerConfig())
        cmd = c.build_ffmpeg_cmd(
            ["rtmp://x/cam1"],
            "/tmp/p_%03d.mp4",
            layout="bridge",
            layout_cam_indices=[0],
            bridge_asset_path="/assets/titlecard.mp4",
        )
        assert "-stream_loop" in cmd
        assert "/assets/titlecard.mp4" in cmd

    def test_chroma_configs_add_extra_inputs_and_filter(self, tmp_path):
        # Real file so chroma config validates OK via path existence (if needed)
        bg = tmp_path / "bg.png"
        bg.write_bytes(b"\x89PNG")
        cfg = ComposerConfig(
            chroma_configs={
                0: ChromaConfig(
                    bg_asset_path=str(bg),
                    bg_asset_kind="image",
                )
            },
        )
        c = Composer(cfg)
        cmd = c.build_ffmpeg_cmd(
            ["rtmp://x/cam1", "rtmp://x/cam2"],
            "/tmp/p_%03d.mp4",
            layout="single",
            layout_cam_indices=[0],
        )
        joined = " ".join(cmd)
        # chroma filter fragment is present
        assert "chromakey" in joined
        # kchroma_0 (keyed) and kchroma_1 (passthrough) both reach streamselect
        assert "kchroma_0" in joined and "kchroma_1" in joined
        # The BG PNG got loaded as an extra input
        assert str(bg) in cmd


# ── apply_selection routing: fast vs slow path ──────────────────────────────


class TestApplySelectionLayoutRouting:
    @pytest.mark.asyncio
    async def test_single_cam_cut_uses_sendcmd_fast_path(self):
        c = Composer(ComposerConfig())
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock()
        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ):
            await c.start_live(
                event_id=1,
                camera_ids=["cam1", "cam2"],
                camera_rtmp_urls=["rtmp://x/1", "rtmp://x/2"],
                output_dir="/tmp/evt1",
            )
        # Reset spawn count (already spawned once by start_live)
        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ) as spawn:
            # Same layout, different cam
            sel = CameraSelection(
                t=5.0, cam_id="cam2",
                layout="single", layout_cams=["cam2"],
                confidence=0.9, reason="speaker",
            )
            await c.apply_selection(sel)
            assert spawn.call_count == 0  # no respawn
        # sendcmd was written
        assert mock_proc.stdin.write.called
        written = mock_proc.stdin.write.call_args[0][0]
        assert b"astreamselect map 1" in written

    @pytest.mark.asyncio
    async def test_layout_change_respawns(self):
        c = Composer(ComposerConfig())
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock()
        mock_proc.terminate = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)
        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ):
            await c.start_live(
                event_id=1,
                camera_ids=["cam1", "cam2"],
                camera_rtmp_urls=["rtmp://x/1", "rtmp://x/2"],
                output_dir="/tmp/evt1",
            )
        # Going to split-screen → respawn expected
        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ) as spawn:
            sel = CameraSelection(
                t=6.0, cam_id="cam1",
                layout="split2_hstack", layout_cams=["cam1", "cam2"],
                confidence=0.85, reason="joke_laugh",
            )
            await c.apply_selection(sel)
            assert spawn.call_count == 1  # respawn happened
        assert mock_proc.terminate.called
        assert c._state.current_layout == "split2_hstack"
        assert c._state.current_layout_cams == ["cam1", "cam2"]

    @pytest.mark.asyncio
    async def test_bridge_layout_respawns_with_asset(self):
        c = Composer(ComposerConfig())
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock()
        mock_proc.terminate = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)
        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ):
            await c.start_live(
                event_id=1,
                camera_ids=["cam1"],
                camera_rtmp_urls=["rtmp://x/1"],
                output_dir="/tmp/evt1",
            )
        spawn_mock = AsyncMock(return_value=mock_proc)
        with patch("asyncio.create_subprocess_exec", new=spawn_mock):
            sel = CameraSelection(
                t=10.0, cam_id="cam1",
                layout="bridge", layout_cams=["cam1"],
                bridge_asset_url="/assets/titlecard.png",
                confidence=0.95, reason="bridge (dead_air)",
            )
            await c.apply_selection(sel)
            assert spawn_mock.call_count == 1
            # The new cmd must include the bridge asset
            spawned_cmd = spawn_mock.call_args[0]
            assert "/assets/titlecard.png" in spawned_cmd
