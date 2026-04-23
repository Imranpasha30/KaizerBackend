"""
kaizer.pipeline.live_director.output
=====================================
Output sinks for the live director: HLS manifest/segments for browser
preview, per-camera ISO recorders for post-event editing.

The Composer (Wave 6.4) already handles:
  - Segmented MP4 FileSink — rotating 10-min chunks of the PROGRAM feed
  - Optional RTMPSink      — direct FLV push to YouTube Live / Twitch

This module adds:
  - HLSSink       : wraps the program FLV output into .m3u8 + .ts for
                    low-latency in-browser playback (HLS.js on frontend)
  - ISORecorder   : one ffmpeg subprocess per camera, stream-copying
                    its RTMP feed to rotating .mp4 segments. These are
                    the untouched camera recordings post-production
                    uses, plus the seed data for the training-flywheel
                    video_hash computation.

Each sink is a small async class with .start() and .stop(). All
subprocesses auto-restart on crash with exponential backoff, identical
to IngestWorker's restart strategy.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from pipeline_core.live_director.signals import CameraConfig

logger = logging.getLogger("kaizer.pipeline.live_director.output")

_MIN_RESTART_S = 1.0
_MAX_RESTART_S = 30.0


# ══════════════════════════════════════════════════════════════════════════════
# HLSSink
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class HLSConfig:
    """HLS output parameters."""
    output_dir: str                  # where .m3u8 + .ts files go
    segment_seconds: int = 4         # 4s is the low-latency sweet spot
    playlist_size: int = 6           # rolling window; older chunks deleted
    hls_flags: str = "delete_segments+append_list+omit_endlist"


class HLSSink:
    """Spawns an ffmpeg subprocess that consumes the program RTMP/SRT URL
    (or re-reads the program MP4 segments) and emits HLS manifest + chunks."""

    def __init__(
        self,
        input_url: str,
        config: HLSConfig,
    ) -> None:
        self.input_url = input_url
        self.config = config
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._task: Optional[asyncio.Task] = None
        self._running: bool = False

    @property
    def playlist_path(self) -> str:
        return os.path.join(self.config.output_dir, "program.m3u8").replace("\\", "/")

    def build_cmd(self) -> list[str]:
        os.makedirs(self.config.output_dir, exist_ok=True)
        segment_tmpl = os.path.join(
            self.config.output_dir, "program_%05d.ts",
        ).replace("\\", "/")
        return [
            "ffmpeg", "-hide_banner", "-y",
            "-i", self.input_url,
            "-c", "copy",
            "-f", "hls",
            "-hls_time", str(self.config.segment_seconds),
            "-hls_list_size", str(self.config.playlist_size),
            "-hls_flags", self.config.hls_flags,
            "-hls_segment_filename", segment_tmpl,
            self.playlist_path,
        ]

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="hls-sink")

    async def stop(self) -> None:
        self._running = False
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
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run_loop(self) -> None:
        backoff = _MIN_RESTART_S
        while self._running:
            cmd = self.build_cmd()
            logger.info("hls-sink: spawning — %s", " ".join(cmd[:5]) + " …")
            try:
                self._proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                await self._proc.wait()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("hls-sink: spawn failed: %s", exc)
            if not self._running:
                return
            logger.warning("hls-sink: ffmpeg exited — restarting in %.1fs", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _MAX_RESTART_S)


# ══════════════════════════════════════════════════════════════════════════════
# ISORecorder — per-camera stream-copy to rotating MP4 segments
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ISOConfig:
    """Per-camera ISO recorder config."""
    camera: CameraConfig
    rtmp_url: str
    output_dir: str                  # per-event dir; recorder writes a cam subdir
    segment_seconds: int = 600       # 10-minute chunks


class ISORecorder:
    """Records ONE camera's RTMP feed to rotating MP4 chunks, stream-copy
    (no re-encode). Output path: <output_dir>/iso/<cam_id>/%03d.mp4."""

    def __init__(self, config: ISOConfig) -> None:
        self.config = config
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._task: Optional[asyncio.Task] = None
        self._running: bool = False

    @property
    def camera_dir(self) -> str:
        return os.path.join(
            self.config.output_dir, "iso", self.config.camera.id,
        ).replace("\\", "/")

    def build_cmd(self) -> list[str]:
        os.makedirs(self.camera_dir, exist_ok=True)
        tmpl = os.path.join(self.camera_dir, "%03d.mp4").replace("\\", "/")
        return [
            "ffmpeg", "-hide_banner", "-y",
            "-i", self.config.rtmp_url,
            "-c", "copy",
            "-f", "segment",
            "-segment_time", str(self.config.segment_seconds),
            "-reset_timestamps", "1",
            "-segment_format", "mp4",
            tmpl,
        ]

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name=f"iso-{self.config.camera.id}")

    async def stop(self) -> None:
        self._running = False
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
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run_loop(self) -> None:
        backoff = _MIN_RESTART_S
        while self._running:
            cmd = self.build_cmd()
            logger.info(
                "iso-recorder[%s]: spawning — %s",
                self.config.camera.id, " ".join(cmd[:5]) + " …",
            )
            try:
                self._proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                await self._proc.wait()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error(
                    "iso-recorder[%s]: spawn failed: %s",
                    self.config.camera.id, exc,
                )
            if not self._running:
                return
            logger.warning(
                "iso-recorder[%s]: ffmpeg exited — restarting in %.1fs",
                self.config.camera.id, backoff,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _MAX_RESTART_S)


# ══════════════════════════════════════════════════════════════════════════════
# OutputStack — convenience bundle
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class OutputStack:
    """Aggregates a composer, optional HLSSink, and N ISORecorders.

    Usage
    -----
        stack = OutputStack(
            composer=composer,
            hls=HLSSink(...),
            iso_recorders=[ISORecorder(cfg1), ISORecorder(cfg2)],
        )
        await stack.start_all()
        # ... event runs ...
        await stack.stop_all()
    """
    composer: object    # Composer instance (pipeline_core.live_director.composer.Composer)
    hls: Optional[HLSSink] = None
    iso_recorders: list[ISORecorder] = field(default_factory=list)

    async def start_all(self) -> None:
        # Composer must already be started via its own start_live(). We
        # just start HLS + ISO here.
        if self.hls is not None:
            await self.hls.start()
        for rec in self.iso_recorders:
            await rec.start()

    async def stop_all(self) -> None:
        if self.hls is not None:
            await self.hls.stop()
        await asyncio.gather(
            *(rec.stop() for rec in self.iso_recorders),
            return_exceptions=True,
        )
        # Composer stop is caller's responsibility — it owns the program
        # subprocess lifetime.
