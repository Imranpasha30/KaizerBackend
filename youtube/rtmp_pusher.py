"""ffmpeg-driven RTMPS push to YouTube.

Given a rendered ``.mp4`` and an ingest URL + stream key from
``rtmp_provider.obtain_rtmp_target()``, this module pushes the file to
YouTube in real time and reports progress.

Why ffmpeg + ``-re``
--------------------
YouTube's RTMP ingest enforces real-time frame pacing. ``ffmpeg -re``
re-reads the input at native frame rate so YT receives frames at the
same rate it would receive a live camera feed. Without ``-re`` the
broadcast will fail with "video bitrate is too high" within seconds.

What this module guarantees
---------------------------
1. ffmpeg runs in a subprocess with stderr captured (not lost).
2. Progress is parsed from ffmpeg's stderr (``time=hh:mm:ss``) so the
   caller can update Job.progress in real-time.
3. The function only returns ``True`` (success) when ffmpeg exits 0
   AND we've pushed at least 95% of the expected duration. Anything
   else raises ``PushFailed``.
4. The subprocess is **never orphaned**: cancellation triggers
   ``proc.terminate()`` then ``proc.kill()`` with a 10-second grace.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from typing import Callable, Optional


class PushFailed(Exception):
    """Terminal — ffmpeg did not complete a successful push."""


_FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")


# Detection for the "duration=" / "time=" markers that ffmpeg emits.
# Format: ``time=00:01:23.45``. We tolerate negative / huge values
# defensively (corrupt input streams sometimes throw weird ones).
_TIME_RE = re.compile(r"time=(-?\d+):(\d+):(\d+\.?\d*)")


def _hms_to_seconds(h: str, m: str, s: str) -> float:
    try:
        return int(h) * 3600 + int(m) * 60 + float(s)
    except (ValueError, TypeError):
        return 0.0


def push_to_rtmp(
    *,
    input_path: str,
    ingest_url: str,
    stream_key: str,
    expected_duration_s: float,
    progress_cb: Optional[Callable[[float, float], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    log_prefix: str = "[rtmp-pusher]",
    extra_ffmpeg_args: Optional[list[str]] = None,
) -> dict:
    """Push ``input_path`` to ``ingest_url/stream_key`` in real time.

    Returns ``{ok: True, seconds_pushed: float, exit_code: int}`` on
    success. On failure raises ``PushFailed`` with a message that
    includes the last 1 KB of ffmpeg stderr (admin Logs tab catches
    this for diagnosis).

    Parameters
    ----------
    expected_duration_s
        Used for two things: (1) the success threshold (we require
        ``seconds_pushed >= 0.95 * expected_duration_s``) and (2) the
        progress percentage handed to ``progress_cb``.
    progress_cb(seconds_pushed, total)
        Called every ~2 seconds while ffmpeg runs. Used by the agent to
        update ``UploadJob.progress``.
    cancel_event
        If set during execution, ffmpeg is terminated gracefully.
        ``PushFailed("cancelled")`` is raised.
    extra_ffmpeg_args
        Hook for tests / overrides. Inserted before the output URL.
    """
    if not os.path.isfile(input_path):
        raise PushFailed(f"input file does not exist: {input_path}")
    if not ingest_url or not stream_key:
        raise PushFailed("ingest_url + stream_key both required")
    if expected_duration_s <= 0:
        raise PushFailed(f"expected_duration_s must be > 0 (got {expected_duration_s})")

    target = f"{ingest_url.rstrip('/')}/{stream_key}"

    # ── ffmpeg command ──────────────────────────────────────────────
    # ``-re``                     real-time read pacing (CRITICAL)
    # ``-i``                      input file
    # ``-c:v copy``               try direct video stream copy first —
    #                             our pipeline outputs H.264 already so
    #                             no re-encode is needed in 99% of cases
    # ``-c:a aac -b:a 128k -ar 44100``   audio always re-encoded to AAC
    #                             (YT requires AAC; some inputs are
    #                             MP3/Opus). 44.1 kHz + 128k is the
    #                             YT-recommended audio profile.
    # ``-f flv``                  flv container = the RTMP wire format
    # ``-loglevel info``          enough to track ``time=`` markers
    # ``-stats``                  emit periodic progress to stderr
    # ``-y``                      overwrite output target if reconnect
    cmd = [
        _FFMPEG_BIN,
        "-hide_banner",
        "-loglevel", "info",
        "-stats",
        "-re",
        "-i", input_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
        "-bufsize", "6000k",
        "-maxrate", "8000k",
        "-pix_fmt", "yuv420p",
        # TCP retry: short transient drops are recovered without
        # losing the broadcast. RTMP itself doesn't have resume so
        # this only helps for sub-second TCP blips.
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "10",
    ]
    if extra_ffmpeg_args:
        cmd.extend(extra_ffmpeg_args)
    cmd.extend([
        "-f", "flv",
        target,
    ])

    print(f"{log_prefix} pushing {os.path.basename(input_path)} "
          f"(~{expected_duration_s:.0f}s) to {ingest_url}")

    # Pipe stderr through so we can parse progress markers. stdout is
    # silent on these flags.
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,           # line-buffered
        encoding="utf-8",
        errors="replace",
    )

    # Rolling tail of stderr — we keep the last 8 KB so a failure
    # report has context without dumping megabytes.
    stderr_tail: list[str] = []
    STDERR_TAIL_BYTES = 8192

    seconds_pushed = 0.0
    last_cb = 0.0

    try:
        for line in iter(proc.stderr.readline, ""):
            # Cancellation check (cheap, every line).
            if cancel_event is not None and cancel_event.is_set():
                _terminate_proc(proc)
                raise PushFailed("cancelled by caller")

            line_stripped = line.rstrip("\r\n")
            stderr_tail.append(line_stripped)
            # Bound the tail.
            if sum(len(s) + 1 for s in stderr_tail) > STDERR_TAIL_BYTES:
                # Drop from the front until we're under the cap.
                while stderr_tail and sum(len(s) + 1 for s in stderr_tail) > STDERR_TAIL_BYTES:
                    stderr_tail.pop(0)

            # Parse the latest ``time=`` marker.
            m = _TIME_RE.search(line)
            if m:
                seconds_pushed = _hms_to_seconds(*m.groups())
                now = time.monotonic()
                if progress_cb and (now - last_cb >= 2.0):
                    try:
                        progress_cb(seconds_pushed, expected_duration_s)
                    except Exception:
                        pass
                    last_cb = now
    finally:
        # readline returns "" on EOF — wait for the process to exit
        # cleanly (it has at this point), but don't block forever.
        try:
            exit_code = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _terminate_proc(proc)
            exit_code = -1

    # Final progress emit so the UI lands on the real number.
    if progress_cb:
        try:
            progress_cb(seconds_pushed, expected_duration_s)
        except Exception:
            pass

    if exit_code != 0:
        tail = "\n".join(stderr_tail[-40:])   # last ~40 lines
        raise PushFailed(
            f"ffmpeg exited {exit_code} after {seconds_pushed:.1f}s of "
            f"{expected_duration_s:.1f}s. stderr tail:\n{tail}"
        )

    # ffmpeg said it succeeded — but we also verify we got most of the
    # expected duration. If we cut out at 10% YT records a 10% video.
    if seconds_pushed < expected_duration_s * 0.95:
        tail = "\n".join(stderr_tail[-20:])
        raise PushFailed(
            f"ffmpeg exited 0 but only {seconds_pushed:.1f}s of "
            f"{expected_duration_s:.1f}s were pushed "
            f"({100*seconds_pushed/expected_duration_s:.0f}%). "
            f"YouTube will record a truncated video. stderr:\n{tail}"
        )

    print(f"{log_prefix} push complete: {seconds_pushed:.1f}s pushed, exit=0")
    return {
        "ok":              True,
        "seconds_pushed":  seconds_pushed,
        "exit_code":       exit_code,
    }


def _terminate_proc(proc: subprocess.Popen) -> None:
    """Graceful → forced shutdown of an ffmpeg child."""
    if proc.poll() is not None:
        return
    try:
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
