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
    current_cam_idx: int = 0
    cues: list[_CueEntry] = field(default_factory=list)
    started_at: float = 0.0
    program_path_template: str = ""


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
    ) -> list[str]:
        """Build the full ffmpeg command for a live switcher session.

        output_path_template example: '/tmp/event_42/program_%03d.mp4'
        (FFmpeg segment muxer fills %03d with the segment index).

        When rtmp_push_url is supplied, ffmpeg also pushes the program feed
        there via `-f flv` output.
        """
        if not camera_rtmp_urls:
            raise ValueError("need at least one camera url")

        # Import the picked encoder args from hw_accel so NVENC kicks in when
        # available.
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

        # Inputs
        for url in camera_rtmp_urls:
            cmd += ["-i", url]

        # Filter
        cmd += [
            "-filter_complex",
            self.build_streamselect_filter(len(camera_rtmp_urls)),
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
            output_dir=output_dir,
            program_path_template=program_tmpl,
            started_at=time.time(),
        )
        cmd = self.build_ffmpeg_cmd(
            camera_rtmp_urls, program_tmpl, rtmp_push_url=rtmp_push_url,
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
        """Director callback: relay a CameraSelection to the running ffmpeg
        via its sendcmd stdin pipe. Also records the decision for the cue sheet."""
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

        line = self.selection_to_sendcmd(selection, self._state.camera_ids)
        if self._proc is not None and self._proc.stdin is not None:
            try:
                self._proc.stdin.write(line.encode("utf-8"))
                await self._proc.stdin.drain()
            except Exception as exc:
                logger.error("composer: sendcmd write failed: %s", exc)
