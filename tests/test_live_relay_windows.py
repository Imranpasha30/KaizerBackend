"""The Go relay, actually running on Windows.

The engine's README says the relay had only ever been run on Linux and that
`go build` for Windows was untested, and the handoff asks for anything
Windows-specific to be reported. Compiling was never the question — it compiled
first time. These are the things the integration now depends on that had never
executed on this OS:

  * the Go RTMP client publishes at all;
  * ONE ffmpeg feeds EVERY channel, and they get the same signal — the claim the
    whole architecture rests on;
  * health can be read from out_time, because an RTMP output reports no size
    (ffmpeg prints total_size=N/A, which is what made an earlier version restart
    a perfectly healthy stream sixteen times);
  * --instance comes back out of /stats, which is how a restarted worker proves
    the pid it recorded is still its relay and not a process that inherited the
    number;
  * removing one channel does not disturb another;
  * POST /shutdown retires it cleanly, which is the only way to stop a relay
    that cannot be adopted.

ffmpeg stands in for YouTube (`-listen 1` makes it an RTMP server), so this needs
no extra binary. It is a real network test and takes about twenty seconds.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request

import pytest

# This deployment's relay, and the ffmpeg the application resolved -- not a
# path that happens to exist on the build machine. Run on LIVE these would
# otherwise exercise DEV's binaries and report on the wrong ones.
import live_integration as _li  # noqa: E402

_STACK = _li.resolve_stack_dir()
RELAY = (pathlib.Path(_STACK) / "bin" / "kaizer-relay.exe") if _STACK else pathlib.Path("kaizer-relay.exe")
pytestmark = [
    pytest.mark.skipif(os.name != "nt", reason="this file is about Windows specifically"),
    pytest.mark.skipif(not RELAY.is_file(),
                       reason=f"{RELAY} is not built; run kaizer-live-stack/deploy/build_relay.ps1"),
]

# Two channels' worth of keys, long enough that mask() has something to hide.
KEYS = ("abcd-efgh-ijkl-mnop", "wxyz-1234-5678-9012")
TOKEN = "pytest-instance-token"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _api(addr, method, path, body=None, timeout=5):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://{addr}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


@pytest.fixture(scope="module")
def ff():
    # What the application itself resolved, so this tests the pair a broadcast
    # would actually use rather than whichever build happens to be on PATH.
    try:
        _s = _li.build_settings()
        if pathlib.Path(_s.ffmpeg).is_file() and pathlib.Path(_s.ffprobe).is_file():
            return _s.ffmpeg, _s.ffprobe
    except Exception:
        pass
    cand = shutil.which("ffmpeg")
    if cand:
        return cand, (shutil.which("ffprobe") or str(pathlib.Path(cand).with_name("ffprobe.exe")))
    pytest.skip("ffmpeg is not available")


@pytest.fixture(scope="module")
def source(ff, tmp_path_factory):
    """A stream-ready clip: H.264 + AAC, keyframes every second."""
    ffmpeg, _ = ff
    out = tmp_path_factory.mktemp("relay") / "src.mp4"
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30:duration=6",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
         "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
         "-movflags", "+faststart", str(out)], check=True)
    return out


@pytest.fixture
def rig(ff, source, tmp_path):
    """Two RTMP sinks and one relay, spawned exactly as the worker spawns it:
    detached, writing to a log file rather than a pipe."""
    ffmpeg, ffprobe = ff
    ports = [_free_port(), _free_port()]
    sinks, files, errs = [], [], []
    for key, port in zip(KEYS, ports):
        f = tmp_path / f"sink_{key[:4]}.flv"
        e = open(tmp_path / f"sink_{key[:4]}.err", "wb")
        files.append(f)
        errs.append(e)
        sinks.append(subprocess.Popen(
            [ffmpeg, "-hide_banner", "-loglevel", "warning", "-listen", "1", "-flush_packets", "1",
             "-i", f"rtmp://127.0.0.1:{port}/live/{key}",
             "-c", "copy", "-flush_packets", "1", "-y", str(f)],
            stdout=subprocess.DEVNULL, stderr=e))
    time.sleep(2.0)
    for name, p, e in zip(KEYS, sinks, errs):
        if p.poll() is not None:
            for x in sinks:
                if x.poll() is None:
                    x.kill()
            pytest.skip(f"an RTMP sink could not listen (port in use?): {name}")

    log = tmp_path / "relay.log"
    fh = open(log, "wb", buffering=0)
    relay = subprocess.Popen(
        [str(RELAY), "--listen", "127.0.0.1:0", "--source", str(source), "--loop=true",
         "--ffmpeg", ffmpeg, "--ffprobe", ffprobe, "--instance", TOKEN,
         "--orphan-timeout", "120s"],
        stdin=subprocess.DEVNULL, stdout=fh, stderr=fh,
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, as worker._detached() does
        creationflags=0x00000008 | 0x00000200)

    addr = ""
    for _ in range(200):
        if relay.poll() is not None:
            break
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("LISTEN "):
                addr = line.split(None, 1)[1].strip()
        if addr:
            break
        time.sleep(0.1)
    if not addr:
        for p in [relay, *sinks]:
            if p.poll() is None:
                p.kill()
        pytest.fail(f"the relay never reported a control address: {log.read_text(errors='replace')[-400:]}")

    yield {"addr": addr, "relay": relay, "ports": ports, "files": files,
           "log": log, "ffprobe": ffprobe}

    for p in [relay, *sinks]:
        if p.poll() is None:
            p.kill()
    fh.close()
    for e in errs:
        e.close()


def _wait_live(addr, n, timeout=45):
    deadline = time.time() + timeout
    st = {}
    while time.time() < deadline:
        st = _api(addr, "GET", "/stats")
        outs = st.get("outputs", {})
        if len(outs) == n and all(o.get("state") == "live" for o in outs.values()):
            return st
        time.sleep(0.5)
    pytest.fail(f"channels never went live: {json.dumps(st.get('outputs', {}))[:400]}")


def test_the_relay_reports_its_control_address_through_a_log_file(rig):
    """A file, never a pipe. A pipe belongs to the worker, and once the worker
    exits nothing drains it: the OS buffer fills at about 64 KB and the relay
    blocks for ever on its next write — the stream freezes while every process
    involved still looks alive."""
    assert rig["addr"].startswith("127.0.0.1:")
    assert "LISTEN " in rig["log"].read_text(errors="replace")


def test_the_instance_token_comes_back_so_adoption_can_be_exact(rig):
    """A pid is a weak claim — pids get reused, and a relay recorded minutes ago
    may be a different process by now. The token cannot be forged by whatever
    inherited the number."""
    st = _api(rig["addr"], "GET", "/stats")
    assert st.get("instance") == TOKEN


def test_one_ffmpeg_feeds_every_channel_with_the_same_signal(rig):
    """The claim the whole architecture rests on: not one ffmpeg per channel."""
    addr = rig["addr"]
    for i, (key, port) in enumerate(zip(KEYS, rig["ports"])):
        _api(addr, "PUT", f"/outputs/ch{i}", {"url": f"rtmp://127.0.0.1:{port}/live/{key}", "gen": 1})
    st = _wait_live(addr, 2)

    # One reader process for the video, whatever the number of channels.
    assert st["reader"].get("pid"), "no reader ffmpeg is running"

    first = {c: int(o["bytes"]) for c, o in st["outputs"].items()}
    t0 = float(st["reader"]["out_time_s"])
    time.sleep(6)
    st = _api(addr, "GET", "/stats")

    assert all(int(o["bytes"]) > first[c] for c, o in st["outputs"].items()), \
        "a channel stopped receiving bytes"
    # HEALTH MUST COME FROM out_time, NOT FROM A SIZE. An RTMP output has no
    # file size -- ffmpeg reports total_size=N/A -- and an earlier version read
    # that as "no progress" and restarted a perfectly healthy stream sixteen
    # times before the cause was found.
    assert float(st["reader"]["out_time_s"]) > t0, "out_time did not advance"

    sizes = [f.stat().st_size for f in rig["files"]]
    assert all(z > 20000 for z in sizes), f"a sink received nothing: {sizes}"
    probes = [subprocess.run([rig["ffprobe"], "-v", "error", "-show_entries",
                              "stream=codec_name,width,height", "-of", "csv=p=0", str(f)],
                             capture_output=True, text=True).stdout.strip() for f in rig["files"]]
    assert probes[0] == probes[1] and probes[0], \
        f"the two channels did not receive the same stream: {probes}"


def test_stats_masks_the_stream_key(rig):
    """The worker copies this towards the admin panel, so a full destination
    here would end up in a browser."""
    addr = rig["addr"]
    _api(addr, "PUT", "/outputs/ch0", {"url": f"rtmp://127.0.0.1:{rig['ports'][0]}/live/{KEYS[0]}", "gen": 1})
    _wait_live(addr, 1)
    blob = json.dumps(_api(addr, "GET", "/stats"), ensure_ascii=False)
    assert KEYS[0] not in blob, "the stream key was exposed in /stats"
    assert KEYS[0][:4] + "\u2026" in blob, "the masked form should still be recognisable"


def test_removing_one_channel_leaves_the_other_streaming(rig):
    """Adding, stopping and restarting one channel while the others keep running
    is the reason for a relay rather than an ffmpeg per channel."""
    addr = rig["addr"]
    for i, (key, port) in enumerate(zip(KEYS, rig["ports"])):
        _api(addr, "PUT", f"/outputs/ch{i}", {"url": f"rtmp://127.0.0.1:{port}/live/{key}", "gen": 1})
    _wait_live(addr, 2)
    before = int(_api(addr, "GET", "/stats")["outputs"]["ch1"]["bytes"])

    _api(addr, "DELETE", "/outputs/ch0")
    time.sleep(4)
    st = _api(addr, "GET", "/stats")
    assert "ch0" not in st["outputs"], "the removed channel is still attached"
    assert int(st["outputs"]["ch1"]["bytes"]) > before, "removing one channel disturbed the other"


def test_shutdown_retires_the_relay_cleanly(rig):
    """The only way to stop a relay a worker finds running but cannot adopt.
    Without it the alternative would be killing a recorded pid, which by then
    may belong to something else entirely."""
    # A reset here IS the shutdown: the relay can close its listener before the
    # 202 finishes crossing the socket, which surfaces as ConnectionResetError
    # under load. What is being tested is whether the process goes away, so the
    # transport's opinion of the last byte is not the evidence -- the exit is.
    try:
        _api(rig["addr"], "POST", "/shutdown", {})
    except (urllib.error.URLError, OSError) as exc:
        print(f"    (shutdown response was cut off, which is expected: {exc})")
    for _ in range(80):
        if rig["relay"].poll() is not None:
            break
        time.sleep(0.25)
    assert rig["relay"].poll() is not None, "the relay ignored POST /shutdown"


# ─────────────────────────────────────────────────────────────────────
# The crash that reached production, and the reason no test saw it.
#
# A relay carrying a real broadcast died with:
#
#     panic: runtime error: index out of range [1] with length 1
#     joy4/format/rtmp.(*Conn).WritePacket   rtmp.go:860
#
# joy4 does `self.streams[pkt.Idx]` with no bounds check. A Go panic ends
# every goroutine, so one destination took every channel on that video off
# air at once.
#
# HOW A PACKET GETS AN INDEX THE CONNECTION NEVER HEARD OF. The relay reads
# FLV from ffmpeg, which is invoked `-map 0:v:0 -map 0:a:0?` -- the audio
# mapping is OPTIONAL. A source with no audio therefore produces an FLV whose
# file-header flags say video only (byte 4 == 1 rather than 5), joy4's prober
# stops at one stream, and the relay FREEZES that list:
#
#     if r.streams == nil { r.streams = streams }
#     else if !sameCodec(...) { r.readerErr.Store("source codec changed...") }
#
# Every RTMP connection then publishes a one-stream header. When ffmpeg is
# restarted -- on a crash, or at the end of a file that was still uploading --
# and the source now carries audio, the demuxer hands out packets with Idx 1.
# sameCodec DETECTS exactly this (it compares len) and only records a string;
# the packets keep flowing, and the next one panics the process.
#
# WHY THE SUITE MISSED IT. Every relay fixture in both repos is built
# `testsrc + sine -> libx264 + aac`: two streams, always, from frame zero. No
# test ever gave the relay a video-only source, and none ever changed the
# source's stream count across a restart. The ffmpeg-restart tests that do
# exist are Linux-only (SIGKILL, `ps --ppid`), so on Windows -- which is what
# runs this in production -- the restart path was never exercised at all.
#
# This test creates precisely that state and asserts the relay SURVIVES it.
# ─────────────────────────────────────────────────────────────────────

def _mk(ffmpeg, path, *, audio: bool, seconds: int = 4):
    """A tiny clip, with or without an audio track."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", f"testsrc=size=320x180:rate=30:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-g", "30", "-keyint_min", "30", "-sc_threshold", "0"]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2"]
    cmd += ["-movflags", "+faststart", str(path)]
    subprocess.run(cmd, check=True)
    return path


def test_a_video_only_source_declares_one_stream_in_its_flv_header(ff, tmp_path):
    """The precondition, isolated: this is what makes the frozen list wrong.

    byte 4 of an FLV header is the flags: bit0 video, bit2 audio. 1 means the
    demuxer will stop probing at a single stream.
    """
    ffmpeg, _ = ff
    for audio, want in ((False, 1), (True, 5)):
        src = _mk(ffmpeg, tmp_path / f"src_{audio}.mp4", audio=audio)
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(src),
             "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", "-f", "flv", "-"],
            capture_output=True).stdout
        assert len(out) > 9, "ffmpeg produced no FLV"
        assert out[4] == want, (
            f"audio={audio}: FLV flags byte is {out[4]}, expected {want}. The "
            f"relay's stream list is derived from this.")


def test_the_relay_survives_a_source_whose_stream_count_changes(ff, tmp_path):
    """The production crash, reproduced end to end.

    Start on a video-only source so the relay freezes a one-stream header, then
    put a file WITH audio at the same path and restart ffmpeg. The demuxer then
    emits Idx 1 packets against a header that declared one stream -- which used
    to panic joy4 and kill every channel.

    Asserts both halves: that the bad packets really arrive (`mismatched` > 0,
    otherwise this test would pass without reproducing anything), and that the
    relay is still alive and still publishing afterwards.
    """
    ffmpeg, ffprobe = ff
    a = _mk(ffmpeg, tmp_path / "a_videoonly.mp4", audio=False)
    b = _mk(ffmpeg, tmp_path / "b_withaudio.mp4", audio=True)
    live = tmp_path / "live_source.mp4"
    shutil.copy(a, live)

    port = _free_port()
    sink_f = tmp_path / "sink.flv"
    sink_e = open(tmp_path / "sink.err", "wb")
    sink = subprocess.Popen(
        [ffmpeg, "-hide_banner", "-loglevel", "warning", "-listen", "1",
         "-i", f"rtmp://127.0.0.1:{port}/live/k", "-c", "copy", "-y", str(sink_f)],
        stdout=subprocess.DEVNULL, stderr=sink_e)
    time.sleep(2.0)
    if sink.poll() is not None:
        sink_e.close()
        pytest.skip("the RTMP sink could not listen (port in use?)")

    log = tmp_path / "relay.log"
    fh = open(log, "wb", buffering=0)
    relay = subprocess.Popen(
        [str(RELAY), "--listen", "127.0.0.1:0", "--source", str(live), "--loop=true",
         "--ffmpeg", ffmpeg, "--ffprobe", ffprobe, "--instance", "streamswap",
         "--orphan-timeout", "120s"],
        stdin=subprocess.DEVNULL, stdout=fh, stderr=fh,
        creationflags=0x00000008 | 0x00000200)
    try:
        addr = ""
        for _ in range(200):
            if relay.poll() is not None:
                break
            for line in log.read_text(errors="replace").splitlines():
                if line.startswith("LISTEN "):
                    addr = line.split(None, 1)[1].strip()
            if addr:
                break
            time.sleep(0.1)
        assert addr, f"relay never listened:\n{log.read_text(errors='replace')[-400:]}"

        _api(addr, "PUT", "/outputs/ch0",
             {"url": f"rtmp://127.0.0.1:{port}/live/k", "gen": 1})
        time.sleep(5)
        first = _api(addr, "GET", "/stats")
        assert list(first["outputs"].values())[0].get("packets", 0) > 0, \
            "nothing was published before the swap; the test proves nothing"

        # The swap: same path, now with an audio track, and force the restart.
        shutil.copy(b, live)
        pid = (first.get("reader") or {}).get("pid")
        if pid:
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
        time.sleep(12)

        after = _api(addr, "GET", "/stats")
        out = list(after["outputs"].values())[0]

        # Did we actually reproduce it? Without this the test could pass by
        # simply never creating the condition.
        assert out.get("mismatched", 0) > 0, (
            "no out-of-range packets arrived, so the panic was never provoked "
            f"-- this test is not testing anything. stats={after}")

        # And the point: it did not die.
        assert relay.poll() is None, (
            "the relay exited when the source's stream count changed:\n"
            + log.read_text(errors="replace")[-900:])
        assert "panic" not in log.read_text(errors="replace").lower(), \
            "the relay panicked"
        assert out.get("state") in ("live", "reconnecting"), out
    finally:
        for p in (relay, sink):
            if p.poll() is None:
                p.kill()
        fh.close()
        sink_e.close()
