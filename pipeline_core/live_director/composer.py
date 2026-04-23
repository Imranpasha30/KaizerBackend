"""
kaizer.pipeline.live_director.composer
=======================================
Program output composer — translates CameraSelection decisions into a live
FFmpeg `streamselect` command driven at runtime via a sendcmd channel.

Architecture
------------
One long-running FFmpeg subprocess with:
  - N video inputs (each a camera's RTMP stream)
  - N audio inputs (same streams, audio side)
  - [v0][v1]…[vN-1]streamselect=inputs=N:map=0[vout]
    [a0][a1]…[aN-1]astreamselect=inputs=N:map=0[aout]
  - Output to:
    * rotating 10-minute MP4 segments on local disk
    * (later) RTMP push to YouTube Live / Twitch
  - sendcmd socket for runtime re-mapping: each CameraSelection writes
    `streamselect map=K, astreamselect map=K` to the sendcmd pipe.

This module handles:
  1. Building the FFmpeg command line (pure — testable without running ffmpeg)
  2. Spawning + supervising the subprocess (auto-restart on crash)
  3. Serialising a CameraSelection to the sendcmd channel
  4. Emitting a cue-sheet (JSON) alongside the program MP4 for post-
     production editors

Lower-third + title overlays
----------------------------
Two paths are offered:
  - Baked via the filter_complex `drawtext`/`overlay` chain (simple static
    overlays like event title + clock).
  - Dynamic PNG overlays via captions.render_caption + a second sendcmd
    queue that toggles their `enable=between(t,START,END)` windows.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

from pipeline_core.live_director.chroma import (
    ChromaConfig,
    build_chroma_chain_for_all,
)
from pipeline_core.live_director.signals import CameraSelection, DirectorEvent

logger = logging.getLogger("kaizer.pipeline.live_director.composer")


@dataclass
class ComposerConfig:
    """Tuneable composer parameters."""
    segment_seconds: int = 600         # rotating MP4 segment length
    bitrate_kbps: int = 6000           # program video bitrate
    audio_bitrate_kbps: int = 192
    encoder_preset: str = "veryfast"   # low-latency tuning for live
    width: int = 1920
    height: int = 1080
    fps: int = 30
    include_cue_sheet: bool = True
    cue_sheet_path: str = ""           # if empty, derives from output_dir
    # Per-camera chroma configs, keyed by 0-based camera index matching the
    # order passed to start_live(camera_ids=...). Cameras without an entry
    # are treated as no-chroma (pass-through).
    chroma_configs: dict = field(default_factory=dict)


@dataclass
class _CueEntry:
    """One entry in the cue-sheet JSON produced alongside the program MP4."""
    t: float
    cam_id: str
    transition: str
    confidence: float
    reason: str


@dataclass
class ComposerRunState:
    """Running state of a live composer session."""
    event_id: int
    camera_ids: list[str]
    output_dir: str
    camera_rtmp_urls: list[str] = field(default_factory=list)
    current_cam_idx: int = 0
    cues: list[_CueEntry] = field(default_factory=list)
    started_at: float = 0.0
    program_path_template: str = ""
    # Active layout being rendered. Defaults to 'single' on first cam.
    current_layout: str = "single"
    current_layout_cams: list[str] = field(default_factory=list)
    current_bridge_asset: str = ""
    rtmp_push_url: Optional[str] = None


# ── Layout filter graph builders (pure, testable without ffmpeg) ─────────────

def _pad_v(idx: int) -> str:
    return f"[{idx}:v]"


def _pad_a(idx: int) -> str:
    return f"[{idx}:a]"


def build_layout_filter(
    layout: str,
    cam_indices: list[int],
    total_cam_count: int,
    width: int,
    height: int,
    *,
    bridge_input_idx: Optional[int] = None,
) -> str:
    """Return a filter_complex string producing [vout] + [aout] for a layout.

    Parameters
    ----------
    layout            : 'single' | 'split2_hstack' | 'split2_vstack' | 'pip'
                        | 'quad' | 'bridge'
    cam_indices       : Which cameras participate (0-based indices into the
                        ffmpeg input list), primary first. Length must match
                        the layout's arity (single=1, split=2, pip=2, quad=4,
                        bridge ignored when a bridge_input_idx is given).
    total_cam_count   : Total number of cameras routed to the composer; used
                        only by the 'single' layout to emit a sendcmd-driven
                        streamselect that can swap the live cam without a
                        respawn.
    width, height     : Program output resolution.
    bridge_input_idx  : For layout='bridge', the ffmpeg input index that
                        points at the bridge asset (image or looping video).

    Returns
    -------
    A semicolon-separated filter chain terminated by the [vout]+[aout] labels.

    Audio policy
    ------------
    Multi-cam layouts use the PRIMARY cam's audio (viewers follow whoever is
    speaking; mixing multiple cams of crowd audio sounds muddy). Override by
    post-processing [aout] in a consumer filter if needed.
    """
    if layout == "single":
        # Use streamselect so sendcmd can swap without a respawn. All N camera
        # inputs feed in; map=K picks the active one.
        n = max(total_cam_count, 1)
        v_pads = "".join(_pad_v(i) for i in range(n))
        a_pads = "".join(_pad_a(i) for i in range(n))
        primary = cam_indices[0] if cam_indices else 0
        return (
            f"{v_pads}streamselect=inputs={n}:map={primary}[vout];"
            f"{a_pads}astreamselect=inputs={n}:map={primary}[aout]"
        )

    if layout in ("split2_hstack", "split2_vstack"):
        if len(cam_indices) != 2:
            raise ValueError(f"{layout} needs exactly 2 cam_indices")
        a_idx, b_idx = cam_indices
        half_w = width // 2 if layout == "split2_hstack" else width
        half_h = height if layout == "split2_hstack" else height // 2
        stack = "hstack" if layout == "split2_hstack" else "vstack"
        return (
            f"{_pad_v(a_idx)}scale={half_w}:{half_h}[lv_a];"
            f"{_pad_v(b_idx)}scale={half_w}:{half_h}[lv_b];"
            f"[lv_a][lv_b]{stack}=inputs=2[vout];"
            f"{_pad_a(a_idx)}acopy[aout]"
        )

    if layout == "pip":
        if len(cam_indices) != 2:
            raise ValueError("pip needs exactly 2 cam_indices")
        main_idx, pip_idx = cam_indices
        pip_w, pip_h = width // 4, height // 4
        pip_x, pip_y = width - pip_w - 32, height - pip_h - 32
        return (
            f"{_pad_v(main_idx)}scale={width}:{height}[lv_main];"
            f"{_pad_v(pip_idx)}scale={pip_w}:{pip_h}[lv_pip];"
            f"[lv_main][lv_pip]overlay=x={pip_x}:y={pip_y}[vout];"
            f"{_pad_a(main_idx)}acopy[aout]"
        )

    if layout == "quad":
        if len(cam_indices) != 4:
            raise ValueError("quad needs exactly 4 cam_indices")
        q_w, q_h = width // 2, height // 2
        parts = [
            f"{_pad_v(idx)}scale={q_w}:{q_h}[lv_q{i}]"
            for i, idx in enumerate(cam_indices)
        ]
        stack = "[lv_q0][lv_q1][lv_q2][lv_q3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[vout]"
        primary = cam_indices[0]
        return ";".join(parts) + ";" + stack + ";" + f"{_pad_a(primary)}acopy[aout]"

    if layout == "bridge":
        if bridge_input_idx is None:
            raise ValueError("bridge requires bridge_input_idx")
        # If we have any camera audio, carry it through; otherwise emit silence.
        if total_cam_count > 0:
            primary = cam_indices[0] if cam_indices else 0
            audio_chain = f"{_pad_a(primary)}acopy[aout]"
        else:
            audio_chain = "anullsrc=channel_layout=stereo:sample_rate=48000[aout]"
        return (
            f"{_pad_v(bridge_input_idx)}scale={width}:{height}:"
            f"force_original_aspect_ratio=increase,crop={width}:{height}[vout];"
            + audio_chain
        )

    raise ValueError(f"unknown layout {layout!r}")


def _build_layout_with_chroma_labels(
    layout: str,
    cam_indices: list[int],
    total_cam_count: int,
    width: int,
    height: int,
    chroma_labels: list[str],
    *,
    bridge_input_idx: Optional[int] = None,
) -> str:
    """Variant of build_layout_filter that consumes chroma output labels
    (kchroma_0, kchroma_1, …) instead of raw [K:v] input pads. Audio pads
    still come from [K:a] since chroma only applies to video."""
    def cv(idx: int) -> str:
        return f"[{chroma_labels[idx]}]" if idx < len(chroma_labels) else _pad_v(idx)

    if layout == "single":
        # Post-chroma: we need to pick the active cam's video. Use streamselect
        # with kchroma_K labels; audio still uses raw astreamselect on [K:a].
        n = max(total_cam_count, 1)
        v_pads = "".join(cv(i) for i in range(n))
        a_pads = "".join(_pad_a(i) for i in range(n))
        primary = cam_indices[0] if cam_indices else 0
        return (
            f"{v_pads}streamselect=inputs={n}:map={primary}[vout];"
            f"{a_pads}astreamselect=inputs={n}:map={primary}[aout]"
        )

    if layout in ("split2_hstack", "split2_vstack"):
        if len(cam_indices) != 2:
            raise ValueError(f"{layout} needs exactly 2 cam_indices")
        a_idx, b_idx = cam_indices
        half_w = width // 2 if layout == "split2_hstack" else width
        half_h = height if layout == "split2_hstack" else height // 2
        stack = "hstack" if layout == "split2_hstack" else "vstack"
        return (
            f"{cv(a_idx)}scale={half_w}:{half_h}[lv_a];"
            f"{cv(b_idx)}scale={half_w}:{half_h}[lv_b];"
            f"[lv_a][lv_b]{stack}=inputs=2[vout];"
            f"{_pad_a(a_idx)}acopy[aout]"
        )

    if layout == "pip":
        if len(cam_indices) != 2:
            raise ValueError("pip needs exactly 2 cam_indices")
        main_idx, pip_idx = cam_indices
        pip_w, pip_h = width // 4, height // 4
        pip_x, pip_y = width - pip_w - 32, height - pip_h - 32
        return (
            f"{cv(main_idx)}scale={width}:{height}[lv_main];"
            f"{cv(pip_idx)}scale={pip_w}:{pip_h}[lv_pip];"
            f"[lv_main][lv_pip]overlay=x={pip_x}:y={pip_y}[vout];"
            f"{_pad_a(main_idx)}acopy[aout]"
        )

    if layout == "quad":
        if len(cam_indices) != 4:
            raise ValueError("quad needs exactly 4 cam_indices")
        q_w, q_h = width // 2, height // 2
        parts = [
            f"{cv(idx)}scale={q_w}:{q_h}[lv_q{i}]"
            for i, idx in enumerate(cam_indices)
        ]
        stack = "[lv_q0][lv_q1][lv_q2][lv_q3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[vout]"
        primary = cam_indices[0]
        return ";".join(parts) + ";" + stack + ";" + f"{_pad_a(primary)}acopy[aout]"

    if layout == "bridge":
        # Bridge ignores chroma labels — the asset is the whole frame.
        return build_layout_filter(
            layout, cam_indices, total_cam_count, width, height,
            bridge_input_idx=bridge_input_idx,
        )

    raise ValueError(f"unknown layout {layout!r}")


class Composer:
    """Builds FFmpeg streamselect commands + drives them via sendcmd.

    Usage (live)
    ------------
        comp = Composer(config)
        await comp.start_live(
            event_id=42,
            camera_ids=['cam1', 'cam2'],
            camera_rtmp_urls=['rtmp://localhost:1935/live/42/cam1',
                              'rtmp://localhost:1935/live/42/cam2'],
            output_dir='/tmp/event_42',
        )
        # Wire the director:
        director._on_selection = comp.apply_selection
        # ... director runs, fires selections, comp relays them to ffmpeg sendcmd

    Usage (offline, from a cue-sheet)
    ---------------------------------
        comp.apply_cue_sheet(
            cue_sheet_path='/tmp/event_42/cues.json',
            camera_mp4_paths={'cam1': '.../cam1_iso.mp4', ...},
            output_path='.../program.mp4',
        )
    """

    def __init__(self, config: Optional[ComposerConfig] = None) -> None:
        self.config = config or ComposerConfig()
        self._state: Optional[ComposerRunState] = None
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._sendcmd_fifo_path: Optional[str] = None
        self._running: bool = False

    # ── Pure helpers (testable without ffmpeg) ────────────────────────────────

    def build_streamselect_filter(self, n_cameras: int) -> str:
        """Return the -filter_complex string for N cameras.

        The `streamselect`/`astreamselect` filters map=K param is what we
        toggle at runtime via sendcmd; v0, v1, … are the video inputs in
        order, a0, a1, … are the audio inputs.
        """
        if n_cameras < 1:
            raise ValueError("need at least 1 camera")
        v_pads = "".join(f"[{i}:v]" for i in range(n_cameras))
        a_pads = "".join(f"[{i}:a]" for i in range(n_cameras))
        return (
            f"{v_pads}streamselect=inputs={n_cameras}:map=0[vout];"
            f"{a_pads}astreamselect=inputs={n_cameras}:map=0[aout]"
        )

    def build_ffmpeg_cmd(
        self,
        camera_rtmp_urls: list[str],
        output_path_template: str,
        *,
        rtmp_push_url: Optional[str] = None,
        layout: str = "single",
        layout_cam_indices: Optional[list[int]] = None,
        bridge_asset_path: str = "",
    ) -> list[str]:
        """Build the full ffmpeg command for a live composer session.

        output_path_template example: '/tmp/event_42/program_%03d.mp4'
        (FFmpeg segment muxer fills %03d with the segment index).

        When rtmp_push_url is supplied, ffmpeg also pushes the program feed
        there via `-f flv` output.

        Layout handling
        ---------------
        layout='single' (default) uses the streamselect-based graph — sendcmd
        can then swap the active cam without respawning. Any other layout
        produces a fixed filter_complex and requires a respawn to change.

        Chroma handling
        ---------------
        Camera feeds with an entry in self.config.chroma_configs are routed
        through chromakey + background overlay before the layout stage. BG
        inputs (images/videos) are appended after all camera -i inputs; the
        bridge asset (when layout='bridge') sits after the BG inputs.
        """
        if not camera_rtmp_urls:
            raise ValueError("need at least one camera url")

        from pipeline_core.hw_accel import h264_args

        enc_args = h264_args(
            bitrate_kbps=self.config.bitrate_kbps,
            maxrate_kbps=max(self.config.bitrate_kbps + 2000, self.config.bitrate_kbps),
            bufsize_kbps=max(self.config.bitrate_kbps * 2, 8000),
            cpu_preset=self.config.encoder_preset,
        )
        audio_args = [
            "-c:a", "aac",
            "-b:a", f"{self.config.audio_bitrate_kbps}k",
            "-ar", "48000",
        ]

        cmd: list[str] = ["ffmpeg", "-hide_banner", "-y"]

        # Primary inputs: cameras
        for url in camera_rtmp_urls:
            cmd += ["-i", url]

        n_cams = len(camera_rtmp_urls)

        # Chroma background inputs (if any)
        chroma_extra_inputs, chroma_fragments, chroma_labels = \
            build_chroma_chain_for_all(
                n_cams, self.config.width, self.config.height,
                self.config.chroma_configs or {},
            )
        for extra in chroma_extra_inputs:
            cmd += extra if isinstance(extra, list) else [extra]

        # Bridge asset input (image or looping video) when layout='bridge'
        bridge_input_idx: Optional[int] = None
        if layout == "bridge":
            if not bridge_asset_path:
                raise ValueError("layout='bridge' needs bridge_asset_path")
            # Loop image or stream-loop video
            _, ext = os.path.splitext(bridge_asset_path.lower())
            if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                cmd += ["-loop", "1", "-i", bridge_asset_path]
            else:
                cmd += ["-stream_loop", "-1", "-i", bridge_asset_path]
            # Index = n_cams + len(bg inputs)
            bridge_input_idx = n_cams + sum(
                1 for _ in chroma_extra_inputs
                if isinstance(_, list) and any(v == "-i" for v in _)
            )
            # Fallback: simple count of "-i" flags in chroma_extra_inputs
            total_is_after_cams = 0
            for item in chroma_extra_inputs:
                if isinstance(item, list):
                    total_is_after_cams += item.count("-i")
            bridge_input_idx = n_cams + total_is_after_cams

        # Layout filter — if chroma is active, the layout must consume chroma
        # output labels instead of raw [K:v] pads. For simplicity, when chroma
        # is in play we prepend the chroma fragments and rewrite the layout
        # to pull from kchroma_K labels. Otherwise, layout reads directly
        # from [K:v] / [K:a].
        layout_cams = layout_cam_indices or [0]
        if chroma_fragments:
            # Rewrite layout to use kchroma_K video pads; audio pads unchanged.
            layout_graph = _build_layout_with_chroma_labels(
                layout, layout_cams, n_cams,
                self.config.width, self.config.height,
                chroma_labels,
                bridge_input_idx=bridge_input_idx,
            )
            filter_chain = ";".join(chroma_fragments) + ";" + layout_graph
        else:
            filter_chain = build_layout_filter(
                layout, layout_cams, n_cams,
                self.config.width, self.config.height,
                bridge_input_idx=bridge_input_idx,
            )

        cmd += [
            "-filter_complex", filter_chain,
            "-map", "[vout]",
            "-map", "[aout]",
        ]

        # Encoders
        cmd += enc_args + audio_args

        # Segmented MP4 output
        cmd += [
            "-f", "segment",
            "-segment_time", str(self.config.segment_seconds),
            "-reset_timestamps", "1",
            "-segment_format", "mp4",
            output_path_template,
        ]

        # Optional second sink — RTMP push
        if rtmp_push_url:
            cmd += [
                "-c:v", "copy", "-c:a", "copy",
                "-f", "flv",
                rtmp_push_url,
            ]

        return cmd

    def selection_to_sendcmd(
        self, selection: CameraSelection, camera_ids: list[str]
    ) -> str:
        """Serialise a CameraSelection into the sendcmd wire format.

        Format: one line per effect, terminated by newline:
            streamselect map K, astreamselect map K

        FFmpeg reads sendcmd from either a file (polled) or a pipe; we use
        stdin-driven `-f lavfi -filter_complex …` with sendcmd attached to
        the streamselect filter via its label.
        """
        try:
            idx = camera_ids.index(selection.cam_id)
        except ValueError:
            raise ValueError(
                f"cam_id {selection.cam_id!r} not in camera_ids {camera_ids!r}"
            )
        return f"streamselect map {idx}, astreamselect map {idx}\n"

    # ── Cue-sheet helpers ────────────────────────────────────────────────────

    def write_cue_sheet(self, path: str) -> None:
        """Dump the current decision timeline to a JSON file."""
        if self._state is None:
            raise RuntimeError("composer has no run state")
        payload = {
            "event_id": self._state.event_id,
            "camera_ids": self._state.camera_ids,
            "started_at": self._state.started_at,
            "cues": [
                {
                    "t": c.t, "cam_id": c.cam_id,
                    "transition": c.transition,
                    "confidence": c.confidence,
                    "reason": c.reason,
                }
                for c in self._state.cues
            ],
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def apply_cue_sheet(
        self,
        cue_sheet_path: str,
        camera_mp4_paths: dict[str, str],
        output_path: str,
    ) -> None:
        """Offline re-compose a program MP4 from per-camera ISO recordings
        + a previously-written cue sheet.

        Uses ffmpeg concat demuxer. Each cue becomes one segment spec
        [cam_iso, t_start, t_end]. Writes a temp concat listfile and runs
        ffmpeg -f concat.
        """
        import tempfile
        with open(cue_sheet_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        cues = payload.get("cues", [])
        if not cues:
            raise ValueError(f"cue sheet {cue_sheet_path!r} has no cues")

        # Build segment list: pairs (t_start, t_end, cam_id) by walking cues.
        segments: list[tuple[float, float, str]] = []
        for i, cue in enumerate(cues):
            t_start = float(cue["t"])
            t_end = float(cues[i + 1]["t"]) if i + 1 < len(cues) else float(cue["t"]) + 600.0
            segments.append((t_start, t_end, cue["cam_id"]))

        # Emit concat listfile entries, one per segment
        listfile = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        try:
            for (start, end, cam) in segments:
                src = camera_mp4_paths.get(cam)
                if not src or not os.path.exists(src):
                    logger.warning("cue cam %s missing ISO mp4 %r — skipping", cam, src)
                    continue
                listfile.write(f"file '{src.replace(chr(39), chr(92) + chr(39))}'\n")
                listfile.write(f"inpoint {start}\n")
                listfile.write(f"outpoint {end}\n")
            listfile.close()
            cmd = [
                "ffmpeg", "-hide_banner", "-y",
                "-f", "concat", "-safe", "0",
                "-i", listfile.name,
                "-c", "copy",
                output_path,
            ]
            logger.info("apply_cue_sheet: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg concat failed (exit {result.returncode}): {result.stderr.strip()}"
                )
        finally:
            try:
                os.unlink(listfile.name)
            except Exception:
                pass

    # ── Live runtime (subprocess + sendcmd) ──────────────────────────────────

    async def start_live(
        self,
        event_id: int,
        camera_ids: list[str],
        camera_rtmp_urls: list[str],
        output_dir: str,
        *,
        rtmp_push_url: Optional[str] = None,
    ) -> None:
        """Spawn the long-running FFmpeg switcher + prepare sendcmd channel."""
        if len(camera_ids) != len(camera_rtmp_urls):
            raise ValueError("camera_ids and camera_rtmp_urls length mismatch")

        os.makedirs(output_dir, exist_ok=True)
        program_tmpl = os.path.join(output_dir, "program_%03d.mp4").replace("\\", "/")

        self._state = ComposerRunState(
            event_id=event_id,
            camera_ids=list(camera_ids),
            camera_rtmp_urls=list(camera_rtmp_urls),
            output_dir=output_dir,
            program_path_template=program_tmpl,
            started_at=time.time(),
            current_layout="single",
            current_layout_cams=[camera_ids[0]] if camera_ids else [],
            rtmp_push_url=rtmp_push_url,
        )
        cmd = self.build_ffmpeg_cmd(
            camera_rtmp_urls, program_tmpl,
            rtmp_push_url=rtmp_push_url,
            layout="single",
            layout_cam_indices=[0],
        )
        logger.info("composer: starting live — %s", " ".join(cmd[:6]) + " …")
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        self._running = True

    async def stop_live(self) -> None:
        """Gracefully terminate the ffmpeg subprocess."""
        if self._proc is None:
            return
        self._running = False
        try:
            self._proc.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            self._proc.kill()
        self._proc = None

    async def apply_selection(self, selection: CameraSelection) -> None:
        """Director callback: turn a CameraSelection into either a sendcmd
        (fast path, single-cam cut within the same layout) or a respawn
        (slow path, any layout / bridge / chroma-config change).

        Also records the decision for the cue sheet.
        """
        if self._state is None:
            logger.warning("composer: apply_selection before start_live — ignoring")
            return
        self._state.cues.append(_CueEntry(
            t=selection.t, cam_id=selection.cam_id,
            transition=selection.transition,
            confidence=selection.confidence,
            reason=selection.reason,
        ))
        try:
            self._state.current_cam_idx = self._state.camera_ids.index(selection.cam_id)
        except ValueError:
            logger.error("composer: unknown cam_id %s", selection.cam_id)
            return

        # Fast path: staying in 'single' layout, only the active cam changed.
        layout_unchanged = (
            selection.layout == "single"
            and self._state.current_layout == "single"
        )
        if layout_unchanged:
            line = self.selection_to_sendcmd(selection, self._state.camera_ids)
            if self._proc is not None and self._proc.stdin is not None:
                try:
                    self._proc.stdin.write(line.encode("utf-8"))
                    await self._proc.stdin.drain()
                except Exception as exc:
                    logger.error("composer: sendcmd write failed: %s", exc)
            self._state.current_layout_cams = [selection.cam_id]
            return

        # Slow path: layout, layout cams, or bridge asset changed → respawn.
        await self._respawn_with_layout(selection)

    async def _respawn_with_layout(self, selection: CameraSelection) -> None:
        """Stop the current ffmpeg subprocess, rebuild the cmd with the new
        layout / bridge / chroma state, and relaunch. There is an ~1s gap in
        output (acceptable — layout changes are rare by design)."""
        if self._state is None:
            return
        # Translate cam_ids to indices
        try:
            layout_cam_indices = [
                self._state.camera_ids.index(c) for c in selection.layout_cams
            ] or [self._state.camera_ids.index(selection.cam_id)]
        except ValueError as exc:
            logger.error("composer: respawn — unknown cam in layout_cams: %s", exc)
            return

        # Kill current proc
        if self._proc is not None:
            try:
                self._proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._proc.kill()
            self._proc = None

        # Build new cmd
        bridge_path = selection.bridge_asset_url if selection.layout == "bridge" else ""
        cmd = self.build_ffmpeg_cmd(
            self._state.camera_rtmp_urls,
            self._state.program_path_template,
            rtmp_push_url=self._state.rtmp_push_url,
            layout=selection.layout,
            layout_cam_indices=layout_cam_indices,
            bridge_asset_path=bridge_path,
        )
        logger.info(
            "composer: respawn — layout=%s cams=%s",
            selection.layout, selection.layout_cams,
        )
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        self._state.current_layout = selection.layout
        self._state.current_layout_cams = list(selection.layout_cams)
        self._state.current_bridge_asset = bridge_path
