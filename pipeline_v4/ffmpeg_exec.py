"""Thin ffmpeg subprocess runner for the V4 pipeline (Wave 4, item A).

Centralises retry + NVENC→CPU fallback for the heavy V4 ffmpeg passes:

  - Every call gets ``retry`` extra attempts on a non-zero exit.
  - When the failure stderr matches the known NVENC breakage patterns
    (driver session exhaustion, CUDA init failure, no capable device,
    encoder OOM) AND the command actually uses ``h264_nvenc``, the
    retry is performed on CPU: the NVENC encoder arg block is rewritten
    to the libx264 equivalent and any ``-hwaccel cuda`` /
    ``-hwaccel_output_format`` input options are dropped. This means a
    flaky / oversubscribed GPU degrades a render to CPU instead of
    failing the job.
  - Every retry (and the final failure) logs the stderr tail (last 800
    chars) so the job log carries enough forensics.

Only the call sites that Wave 4 touches use this runner — untouched
legacy call sites keep their inline ``subprocess.run`` behaviour.
"""
from __future__ import annotations

import re
import subprocess
from typing import List

import os

# Case-insensitive substrings that identify an NVENC/CUDA-specific
# failure (as opposed to e.g. a bad filtergraph, which would fail on
# CPU exactly the same way).
_NVENC_FAILURE_SUBSTRINGS = (
    "openencodesessionex",   # session limit / driver refusal
    "no capable devices",    # no NVENC-capable GPU visible
    "cannot init cuda",      # CUDA context creation failed
    "nvenc",                 # generic h264_nvenc error chatter
    # Leaked/exhausted sessions right after force-killed renders (job
    # 601): the encoder never opens, and the run ALSO reports "received
    # no packets" — the encoder is the root, so this must outrank the
    # stream-failure rung.
    "could not open encoder",
)

# Failures where the OUTPUT came out empty / no frames decoded. The classic
# cause is GPU input-seek: `-hwaccel cuda` + `-ss` before `-i` can decode zero
# frames on some sources/positions, so ffmpeg writes a file with no stream.
# These are NOT nvenc-ENCODE failures (the encoder never even ran), so the
# right mitigation is to drop GPU DECODE (-hwaccel) and re-run on CPU decode.
_STREAM_FAILURE_SUBSTRINGS = (
    "does not contain any stream",
    "output file is empty",
    "received no packets",
    "no frames",
    # Decoder-side death signatures (job 600 re-render): NVDEC session
    # exhaustion under parallel slice+compose surfaces as the h264
    # decoder's "no frame!" plus AVERROR_EXTERNAL — the ENCODER never
    # ran, so dropping GPU decode is the right first rung.
    "no frame!",
    "generic error in an external library",
)

# NVENC preset names (p1..p7) — used to recognise (and drop) the NVENC
# ``-preset pN`` pair during the CPU rewrite.
_NVENC_PRESET_RE = re.compile(r"^p[1-7]$")

_STDERR_TAIL_CHARS = 800


def _threads_already_one(cmd: List[str]) -> bool:
    """True when the command already runs the filtergraph single-threaded —
    the OOM rung has nowhere lower to go and must cascade to a CPU rewrite."""
    try:
        i = cmd.index("-filter_complex_threads")
        return i + 1 < len(cmd) and str(cmd[i + 1]) == "1"
    except ValueError:
        return False


def _is_nvenc_failure(stderr: str) -> bool:
    """True when stderr looks like an NVENC/CUDA failure (case-insensitive)."""
    s = (stderr or "").lower()
    if any(pat in s for pat in _NVENC_FAILURE_SUBSTRINGS):
        return True
    # "out of memory" near "cuda" — encoder VRAM exhaustion.
    if "out of memory" in s and "cuda" in s:
        return True
    return False


def _is_stream_failure(stderr: str) -> bool:
    """True when ffmpeg produced an empty / stream-less output (decode-side
    failure), as opposed to an encoder failure."""
    s = (stderr or "").lower()
    return any(pat in s for pat in _STREAM_FAILURE_SUBSTRINGS)


def _is_oom_failure(stderr: str) -> bool:
    """True when ffmpeg died on memory allocation. ffmpeg 8's THREADED
    filtergraph executor (the ``fc#N`` tasks) keeps frame queues per
    filter link — a branch-heavy graph can demand tens of GB even on a
    box with plenty of free RAM (job 600: single compose OOM'd with
    25GB free). Single-threading the graph collapses that memory."""
    s = (stderr or "").lower()
    return "cannot allocate memory" in s or "error code: -12" in s


def _uses_hwaccel(cmd: List[str]) -> bool:
    return "-hwaccel" in cmd or "-hwaccel_output_format" in cmd


def _drop_hwaccel(cmd: List[str]) -> List[str]:
    """Return ``cmd`` with GPU-decode input options removed (CPU decode),
    leaving the encoder + everything else intact. Used to recover from a
    GPU input-seek that decoded zero frames."""
    out: List[str] = []
    i = 0
    n = len(cmd)
    while i < n:
        tok = cmd[i]
        if tok in ("-hwaccel", "-hwaccel_output_format") and i + 1 < n:
            i += 2
            continue
        out.append(tok)
        i += 1
    return out


def _rewrite_nvenc_to_cpu(cmd: List[str]) -> List[str]:
    """Rewrite an ``h264_nvenc`` command to its libx264 equivalent.

    Swaps ``-c:v h264_nvenc`` for ``-c:v libx264 -preset medium -crf 20``
    and drops the NVENC-only knobs (``-preset pN``, ``-tune hq``,
    ``-rc vbr``, ``-cq N``, ``-b:v 0``) plus any GPU-decode input
    options (``-hwaccel ...`` / ``-hwaccel_output_format ...``).
    Everything else (inputs, filtergraph, audio args) is preserved
    verbatim so the CPU retry produces the same output content.
    """
    out: List[str] = []
    i = 0
    n = len(cmd)
    while i < n:
        tok = cmd[i]
        nxt = cmd[i + 1] if i + 1 < n else ""
        # Drop GPU-decode input options (option + its value).
        if tok in ("-hwaccel", "-hwaccel_output_format") and i + 1 < n:
            i += 2
            continue
        # Swap the encoder itself; emit the libx264 quality block here.
        if tok == "-c:v" and nxt == "h264_nvenc":
            out += ["-c:v", "libx264", "-preset", "medium", "-crf", "20"]
            i += 2
            continue
        # Drop the NVENC preset pair (we already emitted -preset medium).
        if tok == "-preset" and _NVENC_PRESET_RE.match(nxt or ""):
            i += 2
            continue
        # NVENC-only knobs — invalid (or wrong) for libx264.
        if tok == "-tune" and (nxt or "").lower() == "hq":
            i += 2
            continue
        if tok == "-rc" and i + 1 < n:
            i += 2
            continue
        if tok == "-cq" and i + 1 < n:
            i += 2
            continue
        if tok == "-b:v" and nxt == "0":
            i += 2
            continue
        out.append(tok)
        i += 1
    return out


def run_ffmpeg(
    cmd: List[str],
    *,
    timeout: int,
    log_label: str,
    retry: int = 1,
) -> subprocess.CompletedProcess:
    """Run an ffmpeg command with retry + NVENC→CPU fallback.

    Parameters
    ----------
    cmd
        Full argv (including the ffmpeg binary).
    timeout
        Per-attempt timeout in seconds (every V4 ffmpeg call already
        had one — this preserves that invariant).
    log_label
        Short tag for log lines, e.g. ``"compose_story_03"``.
    retry
        Extra attempts after the first failure (default 1 → 2 attempts
        total).

    Returns the successful ``CompletedProcess``. Raises ``RuntimeError``
    carrying the stderr tail after the final failure.
    """
    attempt_cmd = list(cmd)
    total_attempts = max(1, int(retry) + 1)
    last_tail = ""
    last_rc: object = None
    dropped_hwaccel = False
    rewrote_cpu = False

    for attempt in range(1, total_attempts + 1):
        try:
            proc = subprocess.run(
                attempt_cmd, capture_output=True, text=True, timeout=timeout,
                # ffmpeg/ffprobe print filenames + container metadata in the
                # system codepage (latin1), so their stderr routinely carries
                # non-UTF-8 bytes (e.g. 0xf6 'ö' from Sony camera tags). The
                # default strict decode crashes the stdlib reader THREAD with
                # UnicodeDecodeError — spamming the backend log and losing the
                # stderr we classify failures by. errors="replace" keeps the
                # text intact-enough and never raises.
                errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            stderr = exc.stderr
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            last_tail = (stderr or f"(no stderr; timed out after {timeout}s)")[-_STDERR_TAIL_CHARS:]
            last_rc = "timeout"
            print(
                f"[ffmpeg/{log_label}] attempt {attempt}/{total_attempts} "
                f"timed out after {timeout}s; stderr tail:\n{last_tail}",
                flush=True,
            )
            continue  # a timeout is not an NVENC signature — plain retry

        if proc.returncode == 0:
            return proc

        stderr = proc.stderr or ""
        last_tail = stderr[-_STDERR_TAIL_CHARS:]
        last_rc = proc.returncode
        print(
            f"[ffmpeg/{log_label}] attempt {attempt}/{total_attempts} failed "
            f"rc={proc.returncode}; stderr tail:\n{last_tail}",
            flush=True,
        )
        if attempt < total_attempts:
            # Mitigation cascade, each applied at most once:
            #   0. OOM on a -filter_complex command -> single-thread the
            #      filtergraph (ffmpeg 8's threaded executor queues frames
            #      per link; huge graphs demand tens of GB — threads=1
            #      collapses it). Keeps GPU decode/encode.
            #   1. Empty/stream-less output WITH GPU decode  -> drop -hwaccel
            #      (the encoder never ran; the GPU input-seek decoded 0 frames).
            #      Keep the encoder so the fast NVENC encode is retained.
            #   2. NVENC ENCODE failure                      -> full CPU rewrite
            #      (libx264 + drop -hwaccel).
            #   3. Any other failure on a still-GPU command   -> full CPU rewrite
            #      as a last resort before giving up.
            if (_is_oom_failure(stderr) and "-filter_complex" in attempt_cmd
                    and not _threads_already_one(attempt_cmd)):
                # LOWER filter threads to 1 (the callers now set a moderate
                # proactive cap, so the OOM path must REWRITE the value, not
                # only add it). Single-threaded runs several times slower —
                # give the rescue attempt a matching time budget.
                if "-filter_complex_threads" in attempt_cmd:
                    _ti = attempt_cmd.index("-filter_complex_threads")
                    attempt_cmd[_ti + 1] = "1"
                else:
                    _fci = attempt_cmd.index("-filter_complex")
                    attempt_cmd = (attempt_cmd[:_fci]
                                   + ["-filter_complex_threads", "1"]
                                   + attempt_cmd[_fci:])
                timeout = int(timeout * 4)
                print(
                    f"[ffmpeg/{log_label}] out-of-memory in the filtergraph -- "
                    f"retrying single-threaded (-filter_complex_threads 1, "
                    f"timeout {timeout}s)",
                    flush=True,
                )
            elif _is_nvenc_failure(stderr) and "h264_nvenc" in attempt_cmd and not rewrote_cpu:
                # Checked BEFORE the stream rung: an encoder that never
                # opened also yields "received no packets" (job 601) —
                # dropping GPU decode there wastes the retry on the wrong
                # half. Full CPU rewrite is the correct medicine.
                attempt_cmd = _rewrite_nvenc_to_cpu(attempt_cmd)
                rewrote_cpu = True
                dropped_hwaccel = True
                print(
                    f"[ffmpeg/{log_label}] NVENC failure detected -- "
                    f"retrying on CPU (libx264)",
                    flush=True,
                )
            elif (_is_stream_failure(stderr) and _uses_hwaccel(attempt_cmd)
                    and not dropped_hwaccel and not rewrote_cpu):
                attempt_cmd = _drop_hwaccel(attempt_cmd)
                dropped_hwaccel = True
                print(
                    f"[ffmpeg/{log_label}] empty output with GPU decode -- "
                    f"retrying with CPU decode (-hwaccel dropped)",
                    flush=True,
                )
            elif "h264_nvenc" in attempt_cmd and not rewrote_cpu:
                attempt_cmd = _rewrite_nvenc_to_cpu(attempt_cmd)
                rewrote_cpu = True
                dropped_hwaccel = True
                print(
                    f"[ffmpeg/{log_label}] retrying on CPU (libx264) as a fallback",
                    flush=True,
                )
            else:
                print(f"[ffmpeg/{log_label}] retrying unchanged command", flush=True)

    # Forensics: persist the exact final failing command next to its
    # output file, so an OOM/hang can be reproduced and bisected by hand
    # (<output>.failcmd.txt — job 600 cost three blind re-render rounds
    # without this).
    try:
        _out = str(cmd[-1]) if cmd else ""
        if _out and (os.path.sep in _out or ":" in _out):
            with open(_out + ".failcmd.txt", "w", encoding="utf-8") as _fh:
                _fh.write(" ".join(str(t) for t in attempt_cmd))
    except Exception:
        pass
    raise RuntimeError(
        f"{log_label}: ffmpeg failed after {total_attempts} attempt(s) "
        f"(rc={last_rc}): {last_tail}"
    )  