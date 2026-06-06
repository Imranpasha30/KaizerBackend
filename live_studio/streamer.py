"""ffmpeg launcher for Live Studio — push a video file to YouTube
RTMPS with looping + duration cap.

Why a new module instead of reusing ``youtube/rtmp_pusher.py``
-------------------------------------------------------------
The existing pusher streams a known-length file once. Live Studio
needs:
  - ``-stream_loop -1`` so a 30 min video can fill a 6 hour broadcast
  - ``-t <hours_in_sec>`` to hard-cap the broadcast at the requested
    duration regardless of how many loops fit
  - Cancellation via an external event (admin "Stop stream" button)
  - Progress in % of the configured duration (not % of file length)

Rather than fork the existing pusher and risk drift, we call ffmpeg
directly here with a similar shape but the loop-aware flags.

ffmpeg invocation
-----------------
    ffmpeg -re -stream_loop -1 -i <input> -t <secs>
           -c:v copy -c:a aac -b:a 128k -ar 44100
           -reconnect 1 -reconnect_at_eof 1 -reconnect_streamed 1
           -reconnect_delay_max 30
           -f flv <rtmps_url>

``-re`` paces frames at native rate (YouTube ingest requires real-time).
``-stream_loop -1`` loops the input forever — combined with ``-t`` it
stops at exactly the target duration.
``-c:v copy`` avoids re-encoding (fast, no CPU/GPU cost). If the input
isn't H.264 we'd need to transcode; v1 assumes user uploads MP4/H.264.
``-reconnect`` survives transient WS/TCP hiccups to YouTube.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from typing import Callable, Optional


_FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
_TIME_RE    = re.compile(r"time=(-?\d+):(\d+):(\d+\.?\d*)")


class StreamerError(RuntimeError):
    """ffmpeg push failed terminally."""


def _hms_to_sec(h: str, m: str, s: str) -> float:
    try:
        return int(h) * 3600 + int(m) * 60 + float(s)
    except (ValueError, TypeError):
        return 0.0


def push_loop(
    *,
    input_path: str,
    ingest_url: str,
    stream_key: str,
    duration_hours: float,
    progress_cb: Optional[Callable[[float], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    extra_log_cb: Optional[Callable[[str], None]] = None,
) -> bool:
    """Push ``input_path`` to ``ingest_url + stream_key`` for
    ``duration_hours``, looping if the input is shorter than the
    target duration.

    Returns True on natural completion (ffmpeg exited 0 OR the cancel
    event was set after we hit at least 90% of target duration).
    Raises ``StreamerError`` on any other terminal failure.

    ``progress_cb(pct)`` is called with progress as a 0-100 float
    every ~1s, computed against the configured duration (not the
    input file length).
    """
    if duration_hours <= 0:
        raise StreamerError(f"duration_hours must be > 0, got {duration_hours}")
    if not os.path.isfile(input_path):
        raise StreamerError(f"input not found: {input_path}")
    if not ingest_url or not stream_key:
        raise StreamerError("ingest_url + stream_key required")

    target_sec = int(round(duration_hours * 3600))
    rtmps_url  = f"{ingest_url.rstrip('/')}/{stream_key}"

    args: list[str] = [
        _FFMPEG_BIN,
        "-hide_banner", "-loglevel", "info",
        "-re",                       # real-time pacing
        "-stream_loop", "-1",        # loop input forever (capped by -t)
        "-i", input_path,
        "-t", str(target_sec),       # stop at target duration
        "-c:v", "copy",              # no re-encode
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        # Reconnect across transient ingest hiccups (rare but does happen).
        "-reconnect",            "1",
        "-reconnect_at_eof",     "1",
        "-reconnect_streamed",   "1",
        "-reconnect_delay_max",  "30",
        "-f", "flv",
        rtmps_url,
    ]

    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, bufsize=0)

    last_progress = 0.0
    last_update = 0.0

    def _stderr_pump() -> None:
        nonlocal last_progress, last_update
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip()
            if extra_log_cb and line:
                extra_log_cb(line[-300:])
            m = _TIME_RE.search(line)
            if m:
                cur = _hms_to_sec(*m.groups())
                pct = min(99.5, (cur / target_sec) * 100.0) if target_sec else 0.0
                # Throttle UI updates to ~1Hz.
                now = time.time()
                if pct - last_progress > 0.5 or now - last_update > 1.0:
                    last_progress = pct
                    last_update = now
                    if progress_cb:
                        try: progress_cb(pct)
                        except Exception: pass

    pump = threading.Thread(target=_stderr_pump, name="ffmpeg-stderr",
                            daemon=True)
    pump.start()

    # Watch for cancellation. ffmpeg ignores SIGTERM gracefully on
    # Windows — we send SIGINT (CTRL_BREAK_EVENT on Windows) first
    # so the flv muxer flushes properly.
    cancelled_clean = False
    try:
        while True:
            if proc.poll() is not None:
                break
            if cancel_event and cancel_event.is_set():
                # If we've already pushed ≥ 90% of the configured
                # duration, treat a cancel as a clean stop.
                cancelled_clean = (last_progress >= 90.0)
                _terminate(proc)
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        _terminate(proc)
        raise
    finally:
        pump.join(timeout=2)

    code = proc.returncode
    if code == 0 or cancelled_clean:
        if progress_cb:
            try: progress_cb(100.0)
            except Exception: pass
        return True

    # Best-effort tail of stderr for the error message.
    tail = ""
    try:
        if proc.stderr:
            tail = (proc.stderr.read() or b"").decode("utf-8", errors="replace")[-600:]
    except OSError:
        pass
    raise StreamerError(f"ffmpeg exited {code}: {tail}")


def push_passthrough(
    *,
    source_url: str,
    ingest_url: str,
    stream_key: str,
    progress_cb: Optional[Callable[[float], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    extra_log_cb: Optional[Callable[[str], None]] = None,
) -> bool:
    """OBS-style pass-through broadcast for URL sources.

    Pipes ``yt-dlp -o -`` straight into ffmpeg, which copies the
    packets to YouTube's RTMP ingest. No disk landing, no loop —
    the broadcast naturally ends when the source video ends or
    when the operator cancels.

    Differences from ``push_loop``:
      - no ``-stream_loop`` flag (one-shot read)
      - no ``-t`` duration cap (runs as long as the source does)
      - input is ``pipe:0`` fed by yt-dlp
      - progress is reported against the source's discovered duration
        (extracted from ffmpeg's first ``Duration:`` log line) — falls
        back to monotonic elapsed seconds if duration can't be parsed
    """
    if not source_url:
        raise StreamerError("source_url is required")
    if not ingest_url or not stream_key:
        raise StreamerError("ingest_url + stream_key required")

    rtmps_url = f"{ingest_url.rstrip('/')}/{stream_key}"

    # 1) yt-dlp produces a single mp4 stream on stdout. We ask for the
    # best ≤1080p video+audio merged into an fragmented mp4 so ffmpeg
    # can consume it without seeking (pipes aren't seekable).
    ytdlp_args = [
        "yt-dlp",
        "--no-progress", "--no-colors", "--no-warnings",
        "--no-part",
        "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/b",
        "--merge-output-format", "mp4",
        "--postprocessor-args", "ffmpeg:-movflags +frag_keyframe+empty_moov+default_base_moof",
        "-o", "-",
        source_url,
    ]

    # 2) ffmpeg reads from stdin and pushes to RTMP. -re paces frames at
    # source rate so YouTube's ingest doesn't reject us for going too
    # fast. No loop, no -t.
    ff_args = [
        _FFMPEG_BIN,
        "-hide_banner", "-loglevel", "info",
        "-re",
        "-i", "pipe:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
        "-reconnect",            "1",
        "-reconnect_at_eof",     "1",
        "-reconnect_streamed",   "1",
        "-reconnect_delay_max",  "30",
        "-f", "flv",
        rtmps_url,
    ]

    try:
        ytdlp = subprocess.Popen(
            ytdlp_args,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=0,
        )
    except FileNotFoundError as exc:
        raise StreamerError(f"yt-dlp not on PATH: {exc}")

    try:
        ff = subprocess.Popen(
            ff_args,
            stdin=ytdlp.stdout,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            bufsize=0,
        )
    except FileNotFoundError as exc:
        try: ytdlp.kill()
        except Exception: pass
        raise StreamerError(f"ffmpeg not on PATH: {exc}")

    # Close our handle on yt-dlp's stdout — only ffmpeg should be reading
    # from it, otherwise yt-dlp won't receive SIGPIPE when ffmpeg exits.
    if ytdlp.stdout:
        ytdlp.stdout.close()

    last_progress = 0.0
    last_update = 0.0
    discovered_dur_sec = 0.0
    started_at = time.time()

    _DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)")

    def _stderr_pump() -> None:
        nonlocal last_progress, last_update, discovered_dur_sec
        for raw in iter(ff.stderr.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip()
            if extra_log_cb and line:
                extra_log_cb(line[-300:])
            if discovered_dur_sec == 0:
                dm = _DUR_RE.search(line)
                if dm:
                    discovered_dur_sec = _hms_to_sec(*dm.groups())
            m = _TIME_RE.search(line)
            if m:
                cur = _hms_to_sec(*m.groups())
                if discovered_dur_sec > 0:
                    pct = min(99.5, (cur / discovered_dur_sec) * 100.0)
                else:
                    # Unknown duration — show elapsed wall-clock minutes
                    # as a "running" indicator (caps at 99 so the UI
                    # never claims "done" while pipe is still live).
                    pct = min(99.0, (time.time() - started_at) / 60.0)
                now = time.time()
                if pct - last_progress > 0.5 or now - last_update > 1.0:
                    last_progress = pct
                    last_update = now
                    if progress_cb:
                        try: progress_cb(pct)
                        except Exception: pass

    pump = threading.Thread(target=_stderr_pump, name="ffmpeg-stderr-pass",
                            daemon=True)
    pump.start()

    cancelled_by_user = False
    try:
        while True:
            if ff.poll() is not None:
                break
            if cancel_event and cancel_event.is_set():
                cancelled_by_user = True
                _terminate(ff)
                _terminate(ytdlp)
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        _terminate(ff)
        _terminate(ytdlp)
        raise
    finally:
        # Make sure yt-dlp doesn't outlive ffmpeg (SIGPIPE handling on
        # Windows is unreliable — kill it explicitly).
        if ytdlp.poll() is None:
            _terminate(ytdlp)
        pump.join(timeout=2)

    code = ff.returncode
    if code == 0 or cancelled_by_user:
        if progress_cb:
            try: progress_cb(100.0)
            except Exception: pass
        return True

    tail = ""
    try:
        if ff.stderr:
            tail = (ff.stderr.read() or b"").decode("utf-8", errors="replace")[-600:]
    except OSError:
        pass
    raise StreamerError(f"ffmpeg exited {code} (passthrough): {tail}")


def _terminate(proc: subprocess.Popen) -> None:
    """Best-effort kill: SIGINT first (lets flv muxer flush), then
    SIGKILL after 10 s grace. Same pattern as rtmp_pusher.py."""
    try:
        if os.name == "nt":
            proc.send_signal(subprocess.signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        else:
            proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass
