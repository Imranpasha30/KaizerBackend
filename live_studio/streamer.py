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
import sys as _sys
import shutil as _shutil
import re
import subprocess
import threading
import time
from collections import deque
from typing import Callable, Optional


# Resolved to a REAL PATH, not a bare name: it is handed to yt-dlp with
# --ffmpeg-location for HLS sources, and a bare "ffmpeg" would fail the
# isfile() guard and silently omit the flag.
_FFMPEG_BIN = (os.environ.get("FFMPEG_BIN")
               or os.environ.get("KAIZER_FFMPEG_BIN")
               or _shutil.which("ffmpeg")
               or "ffmpeg")

# ── yt-dlp, RESOLVED rather than hoped for ───────────────────────────
# A bare "yt-dlp" resolves only if the venv's bin directory is exported on
# PATH for the CHILD process. On the buildpack deploy that serves production
# it is not, and every URL broadcast died with "[Errno 2] No such file or
# directory: 'yt-dlp'" after the YouTube broadcast had already been minted.
def _resolve_ytdlp() -> list:
    explicit = (os.environ.get("YTDLP_BIN")
                or os.environ.get("KAIZER_YTDLP_BIN"))
    if explicit:
        return [explicit]
    found = _shutil.which("yt-dlp")
    if found:
        return [found]
    # The console script is missing, but the PACKAGE may still be importable
    # -- the normal state of a venv whose bin dir is not on the child's PATH.
    # `python -m yt_dlp` is the same CLI in the same kind of subprocess; it is
    # NOT yt-dlp's in-process Python API, which is what _ytdlp_download's
    # docstring warns against.
    try:
        import importlib.util
        if importlib.util.find_spec("yt_dlp") is not None:
            return [_sys.executable, "-m", "yt_dlp"]
    except Exception:
        pass
    # Nothing found. Keep the bare name so the failure names itself.
    return ["yt-dlp"]


_YTDLP_CMD = _resolve_ytdlp()

_TIME_RE    = re.compile(r"time=(-?\d+):(\d+):(\d+\.?\d*)")

# On Windows, ``CTRL_BREAK_EVENT`` (used by ``_terminate`` to stop a broadcast)
# is delivered to the ENTIRE console process group. If the ffmpeg/yt-dlp child
# shares the parent's group, cancelling a live ALSO signals the uvicorn backend
# and shuts it down (this is why the server died whenever a live was stopped).
# Spawning each child with CREATE_NEW_PROCESS_GROUP puts it in its own group so
# the cancel signal stays contained to ffmpeg/yt-dlp and never reaches uvicorn.
# The attr only exists on Windows; falls back to 0 (no-op) elsewhere.
_NEW_GROUP_FLAGS = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


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
                            stderr=subprocess.PIPE, bufsize=0,
                            creationflags=_NEW_GROUP_FLAGS)

    last_progress = 0.0
    last_update = 0.0
    # The pump thread drains stderr line-by-line, so a later proc.stderr.read()
    # is always empty. Keep the tail here so the error message is diagnosable.
    stderr_tail: deque = deque(maxlen=80)

    def _stderr_pump() -> None:
        nonlocal last_progress, last_update
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                stderr_tail.append(line)
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
                # A user-initiated stop is ALWAYS a clean stop, not a failure —
                # regardless of how far the broadcast got. (Matches passthrough.)
                cancelled_clean = True
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

    # The pump thread already drained stderr — use the captured ring buffer
    # (proc.stderr.read() here would always return empty).
    tail = "\n".join(stderr_tail)[-900:]
    raise StreamerError(f"ffmpeg exited {code}: {tail or '(no stderr captured)'}")


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
        *_YTDLP_CMD,
        "--no-progress", "--no-colors", "--no-warnings",
        "--no-part",
        "-f", "bv*[height<=1080][vcodec^=avc1]+ba[acodec^=mp4a]/bv*[height<=1080][vcodec^=avc1]+ba/b[height<=1080][vcodec^=avc1]/bv*[height<=1080]+ba/b",
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
            # NOT DEVNULL: when yt-dlp cannot fetch the source it says
            # exactly why here, and discarding it left only
            # ffmpeg's "Invalid data ... pipe:0".
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0,
            creationflags=_NEW_GROUP_FLAGS,
        )
    except FileNotFoundError as exc:
        raise StreamerError(
            f"yt-dlp could not be run as {' '.join(_YTDLP_CMD)!r}: {exc}. "
            f"Install it on the server (pip install yt-dlp) or set YTDLP_BIN "
            f"to its full path.")

    try:
        ff = subprocess.Popen(
            ff_args,
            stdin=ytdlp.stdout,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            bufsize=0,
            creationflags=_NEW_GROUP_FLAGS,
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
    stderr_tail: deque = deque(maxlen=80)
    # yt-dlp's own words, kept for the error message -- AND, just as
    # importantly, READ AT ALL. stderr is a pipe; a pipe nobody drains fills at
    # 64 KB and blocks the writer for ever. Turning stderr into a pipe without
    # this thread is what hung stream 123: three live processes, 0 CPU, and an
    # ingest reporting noData.
    _ytdlp_tail: deque = deque(maxlen=20)

    def _ytdlp_pump() -> None:
        if not ytdlp.stderr:
            return
        for raw in iter(ytdlp.stderr.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                _ytdlp_tail.append(line)
                if extra_log_cb:
                    extra_log_cb(("yt-dlp: " + line)[-300:])

    _DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)")

    def _stderr_pump() -> None:
        nonlocal last_progress, last_update, discovered_dur_sec
        for raw in iter(ff.stderr.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                stderr_tail.append(line)
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
    ytpump = threading.Thread(target=_ytdlp_pump, name="ytdlp-stderr",
                              daemon=True)
    ytpump.start()

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

    # Pump already drained stderr — use the captured ring buffer.
    tail = "\n".join(stderr_tail)[-900:]
    # yt-dlp's words are the diagnosis when ffmpeg's complaint is only that
    # stdin held nothing usable.
    yt_tail = " | ".join(list(_ytdlp_tail)[-4:])
    yt_rc = ytdlp.poll()
    if yt_tail and ("pipe:0" in (tail or "") or "Invalid data" in (tail or "")
                    or (yt_rc not in (0, None))):
        raise StreamerError(
            f"the source could not be fetched (yt-dlp exited {yt_rc}): {yt_tail} "
            f"- ffmpeg then had nothing to send: {tail or '(no stderr captured)'}"
        )
    raise StreamerError(
        f"ffmpeg exited {code} (passthrough): {tail or '(no stderr captured)'}"
        + (f" [yt-dlp: {yt_tail}]" if yt_tail else "")
    )


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
