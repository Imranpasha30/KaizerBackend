"""
kaizer.pipeline.live_director.chroma
=====================================
Chroma keying (green/blue screen) + background-asset compositing for
individual live camera feeds.

Scope
-----
Pure filter-graph builder. Emits the FFmpeg ``-filter_complex`` sub-graph
fragments and input arg lists that the Composer splices into its main
command when it builds the live switcher pipeline.

Integration contract
--------------------
The Composer's main filter_complex expects uniform per-camera output
labels (``[kchroma_0][kchroma_1]…``). For cameras without chroma config
we still emit a ``null`` passthrough fragment so the downstream
streamselect sees a consistent label scheme.

Background assets may be images (looped via ``-loop 1``) or videos
(looped via ``-stream_loop -1``). Either way they are overlaid under the
keyed camera feed and scaled to the camera's native resolution according
to ``bg_fit`` (cover / contain / stretch).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("kaizer.pipeline.live_director.chroma")


# ─────────────────────────────────────────────────────────────────────────────
# Extension tables for BG-asset sniffing
# ─────────────────────────────────────────────────────────────────────────────

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".gif"}


# ─────────────────────────────────────────────────────────────────────────────
# ChromaConfig
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ChromaConfig:
    """Per-camera chroma-key + background-asset configuration.

    color          : Key colour as an FFmpeg-accepted hex literal.
    similarity     : 0..1 — ``chromakey`` similarity parameter.
    blend          : 0..1 — ``chromakey`` blend parameter.
    bg_asset_path  : Local filesystem path to the BG image or video.
                     "" disables chroma for this camera.
    bg_asset_kind  : "auto" | "image" | "video". "auto" sniffs from
                     extension at validate time.
    bg_fit         : "cover" | "contain" | "stretch". How the BG scales
                     to the camera's output resolution.
    enabled        : Master switch. False → passthrough fragment only.
    """

    color: str = "0x00d639"
    similarity: float = 0.12
    blend: float = 0.08
    bg_asset_path: str = ""
    bg_asset_kind: str = "auto"
    bg_fit: str = "cover"
    enabled: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers
# ─────────────────────────────────────────────────────────────────────────────


def detect_bg_kind(path: str) -> str:
    """Sniff "image" or "video" from a BG asset's file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _VIDEO_EXTS:
        return "video"
    raise ValueError(f"unknown bg asset extension: {ext!r} (path={path!r})")


def validate_chroma_config(config: ChromaConfig) -> None:
    """Validate a ChromaConfig in place. Resolves ``kind == 'auto'``."""
    if not config.enabled:
        return
    if not config.bg_asset_path:
        raise ValueError("chroma config enabled but bg_asset_path is empty")
    if not os.path.exists(config.bg_asset_path):
        raise FileNotFoundError(
            f"chroma bg asset not found: {config.bg_asset_path!r}"
        )
    if not (0.0 <= config.similarity <= 1.0):
        raise ValueError(
            f"chroma similarity must be in [0,1], got {config.similarity!r}"
        )
    if not (0.0 <= config.blend <= 1.0):
        raise ValueError(
            f"chroma blend must be in [0,1], got {config.blend!r}"
        )
    if config.bg_asset_kind == "auto":
        config.bg_asset_kind = detect_bg_kind(config.bg_asset_path)


def build_chroma_input_args(config: ChromaConfig, input_index: int) -> list[str]:
    """Return the ``-i`` segment that pulls in the BG asset.

    ``input_index`` is accepted for future-proofing / logging even though
    FFmpeg itself doesn't need the index in the arg list — inputs are
    positional in the command.
    """
    if not config.enabled or not config.bg_asset_path:
        return []
    kind = config.bg_asset_kind
    if kind == "auto":
        kind = detect_bg_kind(config.bg_asset_path)
    if kind == "image":
        return ["-loop", "1", "-i", config.bg_asset_path]
    if kind == "video":
        return ["-stream_loop", "-1", "-i", config.bg_asset_path]
    raise ValueError(f"unsupported bg_asset_kind: {kind!r}")


def _scale_clause(width: int, height: int, bg_fit: str) -> str:
    """Return the BG-scaling fragment according to ``bg_fit``."""
    if bg_fit == "cover":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}"
        )
    if bg_fit == "contain":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        )
    if bg_fit == "stretch":
        return f"scale={width}:{height}"
    raise ValueError(f"unsupported bg_fit: {bg_fit!r}")


def chroma_passthrough(camera_pad_in: str, out_label: str) -> str:
    """Return a ``null`` passthrough fragment so label naming stays uniform."""
    return f"{camera_pad_in}null[{out_label}]"


def build_chroma_filter_fragment(
    camera_pad_in: str,
    bg_pad_in: str,
    width: int,
    height: int,
    config: ChromaConfig,
    out_label: str,
) -> str:
    """Return a filter_complex sub-graph for one chroma-keyed camera.

    Shape (cam at [0:v], bg at [2:v], out label ``kchroma_0``)::

        [2:v]scale=W:H:...,crop=W:H[kbg_0];
        [0:v]chromakey=0x00d639:0.12:0.08[kfg_0];
        [kbg_0][kfg_0]overlay=shortest=1[kchroma_0]
    """
    if not config.enabled or not config.bg_asset_path:
        return chroma_passthrough(camera_pad_in, out_label)

    scale = _scale_clause(width, height, config.bg_fit)
    bg_label = f"kbg_{out_label}"
    fg_label = f"kfg_{out_label}"
    return (
        f"{bg_pad_in}{scale}[{bg_label}];"
        f"{camera_pad_in}chromakey={config.color}:{config.similarity}:{config.blend}[{fg_label}];"
        f"[{bg_label}][{fg_label}]overlay=shortest=1[{out_label}]"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Composer-facing aggregator
# ─────────────────────────────────────────────────────────────────────────────


def build_chroma_chain_for_all(
    camera_count: int,
    width: int,
    height: int,
    chroma_configs: dict[int, Optional[ChromaConfig]],
) -> tuple[list[str], list[str], list[str]]:
    """High-level helper that produces the complete chroma layer.

    Returns a tuple ``(extra_inputs, fragments, output_labels)``:
      - ``extra_inputs`` — additional ``-i`` args (with ``-loop`` /
        ``-stream_loop``) to splice before ``-filter_complex``.
      - ``fragments``    — one filter_complex sub-graph per camera,
        joined with ``;`` by the Composer.
      - ``output_labels``— unified per-camera labels (``kchroma_K``) the
        streamselect should consume.

    BG inputs land at indexes starting at ``camera_count``.
    """
    if camera_count < 0:
        raise ValueError("camera_count must be >= 0")

    extra_inputs: list[str] = []
    fragments: list[str] = []
    output_labels: list[str] = []

    next_bg_index = camera_count
    for cam_idx in range(camera_count):
        out_label = f"kchroma_{cam_idx}"
        cam_pad = f"[{cam_idx}:v]"
        cfg = chroma_configs.get(cam_idx)

        if cfg is None or not cfg.enabled or not cfg.bg_asset_path:
            fragments.append(chroma_passthrough(cam_pad, out_label))
        else:
            bg_idx = next_bg_index
            next_bg_index += 1
            extra_inputs.extend(build_chroma_input_args(cfg, bg_idx))
            bg_pad = f"[{bg_idx}:v]"
            fragments.append(
                build_chroma_filter_fragment(
                    camera_pad_in=cam_pad,
                    bg_pad_in=bg_pad,
                    width=width,
                    height=height,
                    config=cfg,
                    out_label=out_label,
                )
            )
        output_labels.append(out_label)

    return extra_inputs, fragments, output_labels
