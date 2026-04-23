"""Phase 7.1 RTMPRelay — pure cmd-building + supervisor loop tests."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline_core.live_director.relay import (
    RTMPRelay,
    RelayDestination,
    RelayStatus,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _dest(
    dest_id: str = "youtube_main",
    *,
    name: str = "YouTube Main",
    rtmp_url: str = "rtmp://a.rtmp.youtube.com/live2/xxxx-xxxx",
    enabled: bool = True,
    reconnect_max_attempts: int = 0,
    reconnect_initial_backoff_s: float = 0.01,
    reconnect_max_backoff_s: float = 0.05,
) -> RelayDestination:
    return RelayDestination(
        id=dest_id,
        name=name,
        rtmp_url=rtmp_url,
        enabled=enabled,
        reconnect_max_attempts=reconnect_max_attempts,
        reconnect_initial_backoff_s=reconnect_initial_backoff_s,
        reconnect_max_backoff_s=reconnect_max_backoff_s,
    )


def _live_proc():
    """MagicMock ffmpeg process whose wait() blocks forever until terminated."""
    proc = MagicMock()
    exit_event = asyncio.Event()

    async def _wait():
        await exit_event.wait()
        return 0

    def _terminate():
        exit_event.set()

    def _kill():
        exit_event.set()

    proc.wait = AsyncMock(side_effect=_wait)
    proc.terminate = MagicMock(side_effect=_terminate)
    proc.kill = MagicMock(side_effect=_kill)
    proc._exit_event = exit_event  # exposed for test control
    return proc


def _dying_proc(rc: int = 1):
    """MagicMock ffmpeg process whose wait() returns immediately (crash)."""
    proc = MagicMock()
    proc.wait = AsyncMock(return_value=rc)
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    return proc


# ══════════════════════════════════════════════════════════════════════════════
# build_ffmpeg_cmd — pure
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildFFmpegCmd:
    def test_build_ffmpeg_cmd_hls_source_adds_reconnect_flags(self):
        relay = RTMPRelay(
            source_url="http://host/playlist.m3u8",
            destinations=[_dest()],
        )
        dest = relay._destinations["youtube_main"]
        cmd = relay.build_ffmpeg_cmd(dest)

        assert cmd[0] == "ffmpeg"
        assert "-reconnect" in cmd
        idx = cmd.index("-reconnect")
        assert cmd[idx + 1] == "1"
        assert "-reconnect_streamed" in cmd
        assert "-reconnect_delay_max" in cmd
        # -i with the source URL present
        assert "-i" in cmd
        i_idx = cmd.index("-i")
        assert cmd[i_idx + 1] == "http://host/playlist.m3u8"
        # stream copy
        assert "-c" in cmd
        c_idx = cmd.index("-c")
        assert cmd[c_idx + 1] == "copy"
        # flv output
        assert "-f" in cmd
        f_idx = cmd.index("-f")
        assert cmd[f_idx + 1] == "flv"
        # dest URL is the final arg
        assert cmd[-1] == "rtmp://a.rtmp.youtube.com/live2/xxxx-xxxx"

    def test_build_ffmpeg_cmd_file_source_adds_re_flag(self):
        relay = RTMPRelay(
            source_url="/path/to/program.mp4",
            destinations=[_dest()],
        )
        cmd = relay.build_ffmpeg_cmd(relay._destinations["youtube_main"])
        assert "-re" in cmd
        # No -reconnect for local sources
        assert "-reconnect" not in cmd
        # Source is still there
        i_idx = cmd.index("-i")
        assert cmd[i_idx + 1] == "/path/to/program.mp4"

    def test_build_ffmpeg_cmd_stream_copy_only(self):
        relay = RTMPRelay(
            source_url="http://host/playlist.m3u8",
            destinations=[_dest()],
        )
        cmd = relay.build_ffmpeg_cmd(relay._destinations["youtube_main"])
        # No encoder flags
        for encoder_flag in ("-c:v", "-c:a", "-b:v", "-b:a", "-preset", "libx264", "h264_nvenc"):
            assert encoder_flag not in cmd, f"unexpected encoder flag {encoder_flag!r} in cmd"
        # -c copy is the single codec directive
        assert cmd.count("-c") == 1
        c_idx = cmd.index("-c")
        assert cmd[c_idx + 1] == "copy"

    def test_build_ffmpeg_cmd_https_source_treated_as_http(self):
        relay = RTMPRelay(
            source_url="https://host/playlist.m3u8",
            destinations=[_dest()],
        )
        cmd = relay.build_ffmpeg_cmd(relay._destinations["youtube_main"])
        assert "-reconnect" in cmd
        assert "-re" not in cmd


# ══════════════════════════════════════════════════════════════════════════════
# Supervisor lifecycle — mocked subprocess
# ══════════════════════════════════════════════════════════════════════════════


class TestMultiDestinationSpawn:
    @pytest.mark.asyncio
    async def test_multi_destination_spawns_one_process_per_destination(
        self, monkeypatch,
    ):
        dests = [
            _dest("youtube_main", rtmp_url="rtmp://yt/live/AAA"),
            _dest("twitch_backup", rtmp_url="rtmp://twitch/live/BBB"),
            _dest("fb_live", rtmp_url="rtmp://fb/live/CCC"),
        ]
        spawn_count = {"n": 0}

        async def _fake_create(*args, **kwargs):
            spawn_count["n"] += 1
            return _live_proc()

        fake = AsyncMock(side_effect=_fake_create)
        monkeypatch.setattr("asyncio.create_subprocess_exec", fake)

        relay = RTMPRelay(
            source_url="http://host/playlist.m3u8",
            destinations=dests,
        )
        await relay.start()
        # Give supervisors a chance to enter the loop + call spawn
        for _ in range(20):
            await asyncio.sleep(0.01)
            if spawn_count["n"] >= len(dests):
                break

        assert spawn_count["n"] == len(dests)
        await relay.stop()


class TestCleanStop:
    @pytest.mark.asyncio
    async def test_clean_stop_transitions_status_to_stopped(self, monkeypatch):
        dests = [_dest("d1"), _dest("d2")]

        async def _fake_create(*args, **kwargs):
            return _live_proc()

        monkeypatch.setattr(
            "asyncio.create_subprocess_exec",
            AsyncMock(side_effect=_fake_create),
        )

        relay = RTMPRelay(
            source_url="http://host/playlist.m3u8",
            destinations=dests,
        )
        await relay.start()
        # Let supervisors reach the `live` state
        for _ in range(30):
            await asyncio.sleep(0.01)
            if all(relay._statuses[d.id].state == "live" for d in dests):
                break

        await relay.stop()

        for d in dests:
            assert relay._statuses[d.id].state == "stopped"
        assert relay.is_running() is False


# ══════════════════════════════════════════════════════════════════════════════
# Respawn + backoff
# ══════════════════════════════════════════════════════════════════════════════


class TestRespawnBackoff:
    @pytest.mark.asyncio
    async def test_subprocess_death_triggers_respawn_with_backoff(
        self, monkeypatch,
    ):
        # First proc dies immediately (rc=1). Second proc is a live one so
        # we can observe the reconnect → live transition.
        call_log: list[float] = []
        procs_to_return: list[object] = []

        dying = _dying_proc(rc=1)
        live = _live_proc()
        procs_to_return.extend([dying, live])

        loop = asyncio.get_event_loop()

        async def _fake_create(*args, **kwargs):
            call_log.append(loop.time())
            return procs_to_return.pop(0)

        monkeypatch.setattr(
            "asyncio.create_subprocess_exec",
            AsyncMock(side_effect=_fake_create),
        )

        dest = _dest(
            "d1",
            reconnect_initial_backoff_s=0.05,
            reconnect_max_backoff_s=0.2,
        )
        relay = RTMPRelay(
            source_url="http://host/playlist.m3u8",
            destinations=[dest],
        )

        # Track that we passed through "reconnecting"
        saw_reconnecting = {"flag": False}
        orig_sleep = asyncio.sleep

        async def _probe_sleep(delay, *a, **k):
            if relay._statuses["d1"].state == "reconnecting":
                saw_reconnecting["flag"] = True
            return await orig_sleep(delay, *a, **k)

        monkeypatch.setattr("asyncio.sleep", _probe_sleep)

        await relay.start()
        # Wait until the second spawn (live proc) has been called
        for _ in range(80):
            await orig_sleep(0.01)
            if len(call_log) >= 2:
                break

        assert len(call_log) >= 2
        # Second spawn must be at least `initial_backoff_s` after first
        gap = call_log[1] - call_log[0]
        assert gap >= 0.04, f"respawn gap {gap:.3f}s shorter than backoff"
        assert saw_reconnecting["flag"] is True

        await relay.stop()


class TestMaxAttemptsFailure:
    @pytest.mark.asyncio
    async def test_reconnect_max_attempts_exceeded_marks_failed(
        self, monkeypatch,
    ):
        # Every spawn returns a dying proc — with max_attempts=2 we
        # expect: attempt#0 (initial) dies → attempt#1 dies → attempt#2
        # dies → counter=3 > 2 → state=failed, loop exits.
        spawn_count = {"n": 0}

        async def _fake_create(*args, **kwargs):
            spawn_count["n"] += 1
            return _dying_proc(rc=1)

        monkeypatch.setattr(
            "asyncio.create_subprocess_exec",
            AsyncMock(side_effect=_fake_create),
        )

        dest = _dest(
            "d1",
            reconnect_max_attempts=2,
            reconnect_initial_backoff_s=0.005,
            reconnect_max_backoff_s=0.01,
        )
        relay = RTMPRelay(
            source_url="http://host/playlist.m3u8",
            destinations=[dest],
        )
        await relay.start()

        # Wait for the supervisor task to complete.
        task = relay._tasks["d1"]
        await asyncio.wait_for(task, timeout=2.0)

        assert relay._statuses["d1"].state == "failed"
        assert spawn_count["n"] == 3  # initial + 2 retries
        await relay.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Hot add / remove
# ══════════════════════════════════════════════════════════════════════════════


class TestHotAddRemove:
    @pytest.mark.asyncio
    async def test_add_destination_hot_adds_to_running_relay(self, monkeypatch):
        procs_created: list[object] = []

        async def _fake_create(*args, **kwargs):
            p = _live_proc()
            procs_created.append(p)
            return p

        monkeypatch.setattr(
            "asyncio.create_subprocess_exec",
            AsyncMock(side_effect=_fake_create),
        )

        relay = RTMPRelay(
            source_url="http://host/playlist.m3u8",
            destinations=[_dest("d1")],
        )
        await relay.start()
        # Wait for d1 to be spawned
        for _ in range(20):
            await asyncio.sleep(0.01)
            if len(procs_created) >= 1:
                break

        await relay.add_destination(_dest("d2", rtmp_url="rtmp://x/y/BBB"))
        for _ in range(20):
            await asyncio.sleep(0.01)
            if len(procs_created) >= 2:
                break

        assert len(procs_created) == 2
        assert "d1" in relay._tasks
        assert "d2" in relay._tasks
        assert not relay._tasks["d1"].done()
        assert not relay._tasks["d2"].done()
        await relay.stop()

    @pytest.mark.asyncio
    async def test_remove_destination_kills_that_subprocess_only(
        self, monkeypatch,
    ):
        procs_by_order: list[object] = []

        async def _fake_create(*args, **kwargs):
            p = _live_proc()
            procs_by_order.append(p)
            return p

        monkeypatch.setattr(
            "asyncio.create_subprocess_exec",
            AsyncMock(side_effect=_fake_create),
        )

        relay = RTMPRelay(
            source_url="http://host/playlist.m3u8",
            destinations=[_dest("d1"), _dest("d2")],
        )
        await relay.start()

        # Wait for both procs to spawn
        for _ in range(30):
            await asyncio.sleep(0.01)
            if len(procs_by_order) >= 2:
                break
        assert len(procs_by_order) >= 2

        # Snapshot which proc belongs to which dest *before* removal
        # (the dict is cleared after termination)
        d1_proc = relay._procs.get("d1")
        d2_proc = relay._procs.get("d2")
        assert d1_proc is not None and d2_proc is not None

        await relay.remove_destination("d1")

        # d1's proc got terminated (its event fired)
        assert d1_proc.terminate.called or d1_proc.kill.called
        # d1 cleared out
        assert "d1" not in relay._tasks
        assert "d1" not in relay._destinations
        # d2 still running, not terminated
        assert "d2" in relay._tasks
        assert not relay._tasks["d2"].done()
        assert not d2_proc.terminate.called

        await relay.stop()


class TestAddDuplicate:
    @pytest.mark.asyncio
    async def test_add_destination_duplicate_id_raises(self):
        relay = RTMPRelay(
            source_url="http://host/playlist.m3u8",
            destinations=[_dest("d1")],
        )
        with pytest.raises(ValueError, match="already exists"):
            await relay.add_destination(_dest("d1", rtmp_url="rtmp://x/y/Z"))


# ══════════════════════════════════════════════════════════════════════════════
# Status queries
# ══════════════════════════════════════════════════════════════════════════════


class TestGetStatus:
    @pytest.mark.asyncio
    async def test_get_status_single_and_all(self):
        relay = RTMPRelay(
            source_url="http://host/playlist.m3u8",
            destinations=[_dest("d1"), _dest("d2")],
        )
        # Single
        s = relay.get_status("d1")
        assert isinstance(s, RelayStatus)
        assert s.destination_id == "d1"
        assert s.state == "idle"

        # All
        all_s = relay.get_status()
        assert isinstance(all_s, list)
        assert len(all_s) == 2
        ids = {s.destination_id for s in all_s}
        assert ids == {"d1", "d2"}


# ══════════════════════════════════════════════════════════════════════════════
# Backoff cap
# ══════════════════════════════════════════════════════════════════════════════


class TestBackoffCap:
    @pytest.mark.asyncio
    async def test_reconnect_backoff_caps_at_max(self, monkeypatch):
        # Every spawn dies → many reconnects → verify each sleep <= max.
        sleep_calls: list[float] = []
        orig_sleep = asyncio.sleep

        async def _recording_sleep(delay, *a, **k):
            sleep_calls.append(float(delay))
            # fast-forward: don't actually sleep that long
            return await orig_sleep(0, *a, **k)

        async def _fake_create(*args, **kwargs):
            return _dying_proc(rc=1)

        monkeypatch.setattr(
            "asyncio.create_subprocess_exec",
            AsyncMock(side_effect=_fake_create),
        )
        monkeypatch.setattr("asyncio.sleep", _recording_sleep)

        dest = _dest(
            "d1",
            reconnect_max_attempts=20,
            reconnect_initial_backoff_s=0.1,
            reconnect_max_backoff_s=2.0,
        )
        relay = RTMPRelay(
            source_url="http://host/playlist.m3u8",
            destinations=[dest],
        )
        await relay.start()
        task = relay._tasks["d1"]
        await asyncio.wait_for(task, timeout=5.0)

        # Many sleeps should have been recorded and all are <= 2.0
        assert len(sleep_calls) >= 5
        for delay in sleep_calls:
            assert delay <= 2.0 + 1e-9, f"backoff {delay} exceeded cap"
        # Later sleeps should hit the cap
        assert max(sleep_calls) == pytest.approx(2.0)

        await relay.stop()
