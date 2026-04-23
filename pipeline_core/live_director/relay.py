"""
kaizer.pipeline.live_director.relay
====================================
RTMP relay module — tees the program output to one or more public ingest
endpoints (YouTube Live, Twitch, Facebook Live, custom RTMP) with
per-destination auto-reconnect and exponential backoff.

Architecture
------------
One `RTMPRelay` instance owns N `RelayDestination`s. For each enabled
destination we spawn a supervisor coroutine that:
  1. Builds an ffmpeg stream-copy command (source → -c copy -f flv → RTMP).
  2. Launches ffmpeg via `asyncio.create_subprocess_exec`.
  3. Awaits its exit.
  4. On unclean exit: sleeps with capped exponential backoff and respawns.
  5. Tracks state in a `RelayStatus` snapshot keyed by destination id.

The relay does NOT re-encode. Source must already be a broadcast-ready
stream — normally the composer's HLS manifest (http://…/playlist.m3u8)
or the local program MP4 being written by the composer. This keeps CPU
near zero per destination, so 4-way tee-ing to YouTube + Twitch +
Facebook + a backup ingest is cheap.

Reconnect strategy
------------------
Two layers stack:
  - FFmpeg's own `-reconnect` flags recover from transient HLS source
    hiccups without killing the process.
  - Our supervisor loop respawns ffmpeg after unrecoverable exits with
    `min(initial * 2**attempts, max_backoff)` sleep between attempts.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Union

logger = logging.getLogger("kaizer.pipeline.live_director.relay")


# ══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class RelayDestination:
    """One RTMP ingest endpoint (YouTube Live, Twitch, custom, …)."""
    id: str
    name: str
    rtmp_url: str
    enabled: bool = True
    reconnect_max_attempts: int = 0          # 0 = infinite retries
    reconnect_initial_backoff_s: float = 1.0
    reconnect_max_backoff_s: float = 30.0


@dataclass
class RelayStatus:
    """Read-only snapshot of one destination's current state."""
    destination_id: str
    state: str = "idle"          # idle | connecting | live | reconnecting | failed | stopped
    attempts: int = 0
    last_error: str = ""
    started_at: float = 0.0      # monotonic timestamp, 0 if never started
    uptime_s: float = 0.0        # 0 if not currently live


# ══════════════════════════════════════════════════════════════════════════════
# RTMPRelay
# ══════════════════════════════════════════════════════════════════════════════


class RTMPRelay:
    """Tees one program source to N RTMP destinations, with per-destination
    supervisor coroutines that auto-reconnect on ffmpeg exit.

    Usage
    -----
        relay = RTMPRelay(
            source_url="http://127.0.0.1:8080/event_42/playlist.m3u8",
            destinations=[
                RelayDestination(id="youtube_main", name="YouTube",
                                 rtmp_url="rtmp://a.rtmp.youtube.com/live2/XXX"),
                RelayDestination(id="twitch_backup", name="Twitch",
                                 rtmp_url="rtmp://live.twitch.tv/app/YYY"),
            ],
        )
        await relay.start()
        # ... event runs, relay auto-reconnects on drops ...
        await relay.stop()
    """

    def __init__(
        self,
        *,
        source_url: str,
        destinations: list[RelayDestination],
    ) -> None:
        self.source_url = source_url
        self._destinations: dict[str, RelayDestination] = {
            d.id: d for d in destinations
        }
        self._statuses: dict[str, RelayStatus] = {
            d.id: RelayStatus(destination_id=d.id) for d in destinations
        }
        self._tasks: dict[str, asyncio.Task] = {}
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._stop_event: asyncio.Event = asyncio.Event()
        self._started: bool = False

    # ── Pure helpers (testable without ffmpeg) ────────────────────────────────

    def _is_http_source(self) -> bool:
        return self.source_url.startswith("http://") or self.source_url.startswith("https://")

    def build_ffmpeg_cmd(self, dest: RelayDestination) -> list[str]:
        """Build the ffmpeg stream-copy command for one destination.

        HLS/HTTP source: adds `-reconnect 1 -reconnect_streamed 1
        -reconnect_delay_max 10` so ffmpeg itself rides out transient
        source hiccups.

        Local file/FIFO source: adds `-re` to pace realtime (the composer
        may still be actively writing).

        Always stream-copies — no encoder args, near-zero CPU per tee.
        """
        cmd: list[str] = ["ffmpeg", "-hide_banner"]

        if self._is_http_source():
            cmd += [
                "-reconnect", "1",
                "-reconnect_streamed", "1",
                "-reconnect_delay_max", "10",
            ]
        else:
            # Local file / FIFO — pace at realtime so downstream ingest
            # doesn't get a burst-read that trips its buffer.
            cmd += ["-re"]

        cmd += [
            "-i", self.source_url,
            "-c", "copy",
            "-f", "flv",
            dest.rtmp_url,
        ]
        return cmd

    # ── Public lifecycle ──────────────────────────────────────────────────────

    async def start(self) -> None:
        """Spawn one supervisor coroutine per enabled destination.

        Safe to call once. Subsequent calls are no-ops.
        """
        if self._started:
            logger.warning("relay: start() called twice — ignoring")
            return
        self._started = True
        self._stop_event.clear()
        for dest in self._destinations.values():
            if not dest.enabled:
                logger.info("relay[%s]: disabled, skipping", dest.id)
                continue
            self._tasks[dest.id] = asyncio.create_task(
                self._run_supervisor(dest), name=f"relay-{dest.id}",
            )
        logger.info(
            "relay: started %d supervisor(s) for source=%s",
            len(self._tasks), self.source_url,
        )

    async def stop(self) -> None:
        """Cancel every supervisor, kill every subprocess, mark all stopped.

        Idempotent.
        """
        self._stop_event.set()
        # Kill subprocesses first so waiting supervisors wake up.
        for dest_id, proc in list(self._procs.items()):
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            except Exception as exc:
                logger.error("relay[%s]: terminate failed: %s", dest_id, exc)
        # Give each proc a moment to exit cleanly, else kill.
        for dest_id, proc in list(self._procs.items()):
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            except Exception:
                pass
        self._procs.clear()
        # Cancel and await supervisor tasks.
        for task in list(self._tasks.values()):
            task.cancel()
        for task in list(self._tasks.values()):
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error("relay: supervisor task raised on stop: %s", exc)
        self._tasks.clear()
        # Transition all remaining statuses to "stopped".
        for status in self._statuses.values():
            status.state = "stopped"
            status.uptime_s = 0.0
        self._started = False

    async def add_destination(self, dest: RelayDestination) -> None:
        """Hot-add a destination. If relay is running, spawn its supervisor."""
        if dest.id in self._destinations:
            raise ValueError(f"destination id {dest.id!r} already exists")
        self._destinations[dest.id] = dest
        self._statuses[dest.id] = RelayStatus(destination_id=dest.id)
        if self._started and dest.enabled:
            self._tasks[dest.id] = asyncio.create_task(
                self._run_supervisor(dest), name=f"relay-{dest.id}",
            )
            logger.info("relay[%s]: hot-added supervisor", dest.id)

    async def remove_destination(self, destination_id: str) -> None:
        """Cancel that destination's supervisor, kill its ffmpeg, drop state."""
        if destination_id not in self._destinations:
            return
        # Kill the subprocess so the supervisor's `await wait()` unblocks.
        proc = self._procs.pop(destination_id, None)
        if proc is not None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            except Exception:
                pass
        # Cancel supervisor task.
        task = self._tasks.pop(destination_id, None)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error(
                    "relay[%s]: supervisor task raised on remove: %s",
                    destination_id, exc,
                )
        self._destinations.pop(destination_id, None)
        self._statuses.pop(destination_id, None)
        logger.info("relay[%s]: removed", destination_id)

    def get_status(
        self, destination_id: Optional[str] = None,
    ) -> Union[RelayStatus, list[RelayStatus]]:
        """Return a status snapshot — one destination if id given, else all."""
        if destination_id is not None:
            status = self._statuses[destination_id]
            return self._snapshot(status)
        return [self._snapshot(s) for s in self._statuses.values()]

    def is_running(self) -> bool:
        """True if any supervisor task is still alive."""
        return any(not t.done() for t in self._tasks.values())

    # ── Internals ────────────────────────────────────────────────────────────

    def _snapshot(self, status: RelayStatus) -> RelayStatus:
        """Return a copy of a status with uptime recalculated from now."""
        uptime = 0.0
        if status.state == "live" and status.started_at > 0:
            uptime = max(0.0, time.monotonic() - status.started_at)
        return RelayStatus(
            destination_id=status.destination_id,
            state=status.state,
            attempts=status.attempts,
            last_error=status.last_error,
            started_at=status.started_at,
            uptime_s=uptime,
        )

    async def _run_supervisor(self, dest: RelayDestination) -> None:
        """Per-destination infinite loop: spawn ffmpeg, await exit, backoff,
        respawn. Exits cleanly on stop_event or cancel."""
        status = self._statuses[dest.id]
        attempts = 0
        try:
            while not self._stop_event.is_set():
                status.state = "connecting"
                status.attempts = attempts
                cmd = self.build_ffmpeg_cmd(dest)
                logger.info(
                    "relay[%s]: spawning — %s", dest.id,
                    " ".join(cmd[:6]) + " …",
                )
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.PIPE,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    status.last_error = f"spawn failed: {exc}"
                    logger.error("relay[%s]: spawn failed: %s", dest.id, exc)
                    # Treat as a failed attempt and back off.
                    attempts += 1
                    if (
                        dest.reconnect_max_attempts
                        and attempts > dest.reconnect_max_attempts
                    ):
                        status.state = "failed"
                        break
                    status.state = "reconnecting"
                    backoff = min(
                        dest.reconnect_initial_backoff_s * (2 ** attempts),
                        dest.reconnect_max_backoff_s,
                    )
                    await asyncio.sleep(backoff)
                    continue

                self._procs[dest.id] = proc
                status.state = "live"
                status.started_at = time.monotonic()
                status.last_error = ""

                try:
                    rc = await proc.wait()
                except asyncio.CancelledError:
                    raise
                finally:
                    self._procs.pop(dest.id, None)

                # Clean stop requested → exit the loop.
                if self._stop_event.is_set():
                    break

                # Unclean exit: count the attempt and back off.
                attempts += 1
                status.attempts = attempts
                status.last_error = f"ffmpeg exited rc={rc}"
                logger.warning(
                    "relay[%s]: ffmpeg exited rc=%s (attempt %d)",
                    dest.id, rc, attempts,
                )

                if (
                    dest.reconnect_max_attempts
                    and attempts > dest.reconnect_max_attempts
                ):
                    status.state = "failed"
                    logger.error(
                        "relay[%s]: max attempts (%d) exceeded — giving up",
                        dest.id, dest.reconnect_max_attempts,
                    )
                    break

                status.state = "reconnecting"
                backoff = min(
                    dest.reconnect_initial_backoff_s * (2 ** attempts),
                    dest.reconnect_max_backoff_s,
                )
                logger.info(
                    "relay[%s]: reconnecting in %.2fs", dest.id, backoff,
                )
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    raise
        except asyncio.CancelledError:
            # Cancelled: ensure subprocess dies, mark stopped, propagate.
            proc = self._procs.pop(dest.id, None)
            if proc is not None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                except Exception:
                    pass
            status.state = "stopped"
            raise
        except BaseException as exc:
            # Unexpected: record and mark failed (do not swallow silently).
            status.last_error = f"supervisor crashed: {exc}"
            status.state = "failed"
            logger.exception("relay[%s]: supervisor crashed", dest.id)
            return

        # Normal loop exit (clean stop or max attempts): set terminal state.
        if status.state not in ("failed",):
            status.state = "stopped"
