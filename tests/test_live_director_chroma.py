"""Phase 7.2 Chroma — tests for pure filter-graph builder + validation."""
from __future__ import annotations

import pytest

from pipeline_core.live_director.chroma import (
    ChromaConfig,
    build_chroma_chain_for_all,
    build_chroma_filter_fragment,
    build_chroma_input_args,
    chroma_passthrough,
    detect_bg_kind,
    validate_chroma_config,
)


# ══════════════════════════════════════════════════════════════════════════════
# detect_bg_kind
# ══════════════════════════════════════════════════════════════════════════════


class TestDetectBgKind:
    def test_detect_bg_kind_image_extensions(self):
        assert detect_bg_kind("bg.png") == "image"
        assert detect_bg_kind("bg.JPG") == "image"
        assert detect_bg_kind("bg.jpeg") == "image"
        assert detect_bg_kind("bg.webp") == "image"
        assert detect_bg_kind("bg.bmp") == "image"

    def test_detect_bg_kind_video_extensions(self):
        assert detect_bg_kind("bg.mp4") == "video"
        assert detect_bg_kind("bg.mov") == "video"
        assert detect_bg_kind("bg.webm") == "video"
        assert detect_bg_kind("bg.mkv") == "video"

    def test_detect_bg_kind_gif_is_video(self):
        assert detect_bg_kind("bg.gif") == "video"

    def test_detect_bg_kind_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown bg asset extension"):
            detect_bg_kind("bg.xyz")


# ══════════════════════════════════════════════════════════════════════════════
# validate_chroma_config
# ══════════════════════════════════════════════════════════════════════════════


class TestValidateChromaConfig:
    def test_validate_resolves_auto_kind(self, tmp_path):
        bg = tmp_path / "bg.png"
        bg.write_bytes(b"\x89PNG\r\n")  # dummy content, only extension matters
        cfg = ChromaConfig(bg_asset_path=str(bg), bg_asset_kind="auto")
        validate_chroma_config(cfg)
        assert cfg.bg_asset_kind == "image"

    def test_validate_enabled_without_path_raises(self):
        cfg = ChromaConfig(bg_asset_path="", enabled=True)
        with pytest.raises(ValueError, match="bg_asset_path"):
            validate_chroma_config(cfg)

    def test_validate_missing_file_raises_filenotfound(self, tmp_path):
        cfg = ChromaConfig(bg_asset_path=str(tmp_path / "does_not_exist.png"))
        with pytest.raises(FileNotFoundError):
            validate_chroma_config(cfg)

    def test_validate_bad_similarity_raises(self, tmp_path):
        bg = tmp_path / "bg.png"
        bg.write_bytes(b"\x89PNG")
        cfg = ChromaConfig(bg_asset_path=str(bg), similarity=1.5)
        with pytest.raises(ValueError, match="similarity"):
            validate_chroma_config(cfg)


# ══════════════════════════════════════════════════════════════════════════════
# build_chroma_input_args
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildChromaInputArgs:
    def test_build_chroma_input_args_image_uses_loop(self):
        cfg = ChromaConfig(
            bg_asset_path="/tmp/bg.png", bg_asset_kind="image"
        )
        args = build_chroma_input_args(cfg, input_index=3)
        assert args == ["-loop", "1", "-i", "/tmp/bg.png"]

    def test_build_chroma_input_args_video_uses_stream_loop(self):
        cfg = ChromaConfig(
            bg_asset_path="/tmp/bg.mp4", bg_asset_kind="video"
        )
        args = build_chroma_input_args(cfg, input_index=3)
        assert args == ["-stream_loop", "-1", "-i", "/tmp/bg.mp4"]

    def test_build_chroma_input_args_disabled_returns_empty(self):
        cfg = ChromaConfig(
            bg_asset_path="/tmp/bg.png", bg_asset_kind="image", enabled=False
        )
        assert build_chroma_input_args(cfg, input_index=3) == []


# ══════════════════════════════════════════════════════════════════════════════
# build_chroma_filter_fragment
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildChromaFilterFragment:
    def test_fragment_enabled_has_chromakey_and_overlay(self):
        cfg = ChromaConfig(
            bg_asset_path="/tmp/bg.png", bg_asset_kind="image"
        )
        frag = build_chroma_filter_fragment(
            camera_pad_in="[0:v]",
            bg_pad_in="[2:v]",
            width=1920, height=1080,
            config=cfg,
            out_label="kchroma_0",
        )
        assert "chromakey=0x00d639:0.12:0.08" in frag
        assert "overlay=shortest=1" in frag
        assert "[kchroma_0]" in frag
        assert "[0:v]" in frag
        assert "[2:v]" in frag

    def test_fragment_cover_fit_uses_crop(self):
        cfg = ChromaConfig(
            bg_asset_path="/tmp/bg.png", bg_asset_kind="image", bg_fit="cover"
        )
        frag = build_chroma_filter_fragment(
            "[0:v]", "[2:v]", 1920, 1080, cfg, "kchroma_0"
        )
        assert "force_original_aspect_ratio=increase" in frag
        assert "crop=1920:1080" in frag

    def test_fragment_contain_fit_uses_pad(self):
        cfg = ChromaConfig(
            bg_asset_path="/tmp/bg.png", bg_asset_kind="image", bg_fit="contain"
        )
        frag = build_chroma_filter_fragment(
            "[0:v]", "[2:v]", 1920, 1080, cfg, "kchroma_0"
        )
        assert "force_original_aspect_ratio=decrease" in frag
        assert "pad=1920:1080" in frag

    def test_fragment_stretch_fit_uses_plain_scale(self):
        cfg = ChromaConfig(
            bg_asset_path="/tmp/bg.png", bg_asset_kind="image", bg_fit="stretch"
        )
        frag = build_chroma_filter_fragment(
            "[0:v]", "[2:v]", 1920, 1080, cfg, "kchroma_0"
        )
        assert "scale=1920:1080" in frag
        assert "force_original_aspect_ratio" not in frag
        assert "crop=" not in frag
        assert "pad=" not in frag

    def test_fragment_disabled_is_passthrough(self):
        cfg = ChromaConfig(
            bg_asset_path="/tmp/bg.png", bg_asset_kind="image", enabled=False
        )
        frag = build_chroma_filter_fragment(
            "[0:v]", "[2:v]", 1920, 1080, cfg, "kchroma_0"
        )
        assert "null" in frag
        assert "chromakey" not in frag
        assert "[kchroma_0]" in frag


# ══════════════════════════════════════════════════════════════════════════════
# chroma_passthrough
# ══════════════════════════════════════════════════════════════════════════════


class TestChromaPassthrough:
    def test_passthrough_shape(self):
        assert chroma_passthrough("[0:v]", "kchroma_0") == "[0:v]null[kchroma_0]"


# ══════════════════════════════════════════════════════════════════════════════
# build_chroma_chain_for_all
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildChromaChainForAll:
    def test_chroma_chain_all_mixes_enabled_and_disabled(self):
        # 3 cams; only cam 1 has chroma.
        cfg_mid = ChromaConfig(
            bg_asset_path="/tmp/bg.png", bg_asset_kind="image"
        )
        chroma_configs = {0: None, 1: cfg_mid, 2: None}
        extra_inputs, fragments, output_labels = build_chroma_chain_for_all(
            camera_count=3, width=1920, height=1080,
            chroma_configs=chroma_configs,
        )
        # 1 BG → 1 extra input group (with -loop 1 -i path)
        assert extra_inputs == ["-loop", "1", "-i", "/tmp/bg.png"]
        # 3 fragments, one per camera
        assert len(fragments) == 3
        # 3 output labels
        assert output_labels == ["kchroma_0", "kchroma_1", "kchroma_2"]
        # BG input for cam 1 should reference index 3 (= camera_count)
        assert "[3:v]" in fragments[1]
        # Cam 0 and cam 2 are passthrough nulls
        assert "null" in fragments[0]
        assert "null" in fragments[2]
        # Cam 1 fragment has chromakey
        assert "chromakey" in fragments[1]

    def test_chroma_chain_no_enabled_returns_no_extra_inputs(self):
        chroma_configs = {0: None, 1: None}
        extra_inputs, fragments, output_labels = build_chroma_chain_for_all(
            camera_count=2, width=1280, height=720,
            chroma_configs=chroma_configs,
        )
        assert extra_inputs == []
        assert len(fragments) == 2
        assert output_labels == ["kchroma_0", "kchroma_1"]
        for f in fragments:
            assert "null" in f
            assert "chromakey" not in f

    def test_chroma_chain_output_labels_are_unique_and_ordered(self):
        chroma_configs = {
            0: ChromaConfig(bg_asset_path="/tmp/a.png", bg_asset_kind="image"),
            1: None,
            2: ChromaConfig(bg_asset_path="/tmp/b.mp4", bg_asset_kind="video"),
            3: None,
        }
        _extra, _frags, labels = build_chroma_chain_for_all(
            camera_count=4, width=1920, height=1080,
            chroma_configs=chroma_configs,
        )
        assert labels == ["kchroma_0", "kchroma_1", "kchroma_2", "kchroma_3"]
        assert len(set(labels)) == len(labels)
