"""Capacity planning + live log capture for the admin dashboard.

This module owns two side-effects that run for the lifetime of the uvicorn
process:

1. **Persistent system-utilisation sampler.** A daemon thread takes a
   psutil + nvidia-smi snapshot every ~30 s and inserts one row into
   ``system_metrics``. Old rows are pruned after 14 days. The admin
   "Capacity" tab reads from this table to size the future cloud
   deployment ("how much CPU/RAM/GPU do we actually need at peak?").

2. **In-memory log ring buffer.** Everything uvicorn would have printed
   to the terminal (FastAPI startup banner, ``print()`` calls, the
   uvicorn access log) is teed into a ring buffer and broadcast to any
   SSE subscriber. That feeds the admin "Logs" tab so the operator can
   watch the live terminal output from the browser without needing to
   keep a PowerShell window around.

Why a separate module
---------------------
Both pieces want to be in place *before* any router code runs (so the
sampler captures startup-time metrics and the buffer captures the FastAPI
boot banner). ``main.py`` imports this and calls ``install_log_capture()``
+ ``start_metric_sampler()`` near the top of its initialisation flow.

Failure mode
------------
Everything here is best-effort: a psutil exception, a missing nvidia-smi,
a DB hiccup — none of them are allowed to take down uvicorn. The sampler
loop swallows all errors and just retries on the next tick.
"""
from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from queue import Queue, Empty
from typing import Optional

import psutil

# ── Tunables (env-overridable) ────────────────────────────────────────

# Cadence of the sampler thread. 30s is a sweet spot: gives ~2,880 samples
# per day, enough to spot a 1-minute compose burst but cheap to store
# (~28k rows / 10 days). Set via env if you want denser/sparser data.
_SAMPLE_INTERVAL_S = int(os.environ.get("KAIZER_METRIC_INTERVAL_S", "30"))

# How long persisted samples live. After this, the daemon prunes rows.
_RETENTION_DAYS    = int(os.environ.get("KAIZER_METRIC_RETENTION_DAYS", "14"))

# Max log lines kept in the in-memory ring buffer. 2000 lines is ~200 KB
# at typical log-line lengths — fits any browser comfortably and gives the
# operator a few hours of recent activity to scroll back through.
_LOG_BUFFER_SIZE   = int(os.environ.get("KAIZER_LOG_BUFFER_SIZE", "2000"))


# ── Log ring buffer (multi-subscriber, thread-safe) ──────────────────

class _LogRingBuffer:
    """Bounded, thread-safe ring of log lines + fan-out to SSE subscribers.

    Append is O(1). Snapshot (for the REST endpoint that returns the
    backlog) copies under the lock. Each SSE subscriber gets a Queue
    that ``append`` pushes onto; the queue is bounded so a slow consumer
    can't pin memory — overflow drops oldest.
    """

    def __init__(self, maxlen: int = _LOG_BUFFER_SIZE) -> None:
        self._buf:  deque[dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._subs: set[Queue] = set()
        self._next_id = 1

    def append(self, level: str, source: str, line: str) -> None:
        # Drop trailing newlines so the SSE payload is one event per line.
        line = (line or "").rstrip("\r\n")
        if not line:
            return
        ts = datetime.now(timezone.utc).isoformat()
        with self._lock:
            seq = self._next_id
            self._next_id += 1
            entry = {"id": seq, "ts": ts, "level": level, "source": source, "line": line}
            self._buf.append(entry)
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(entry)
            except Exception:
                pass   # subscriber's queue full — drop, they'll resync on reconnect

    def snapshot(self, limit: int = 500, level: Optional[str] = None) -> list[dict]:
        with self._lock:
            items = list(self._buf)
        if level:
            items = [e for e in items if e["level"] == level]
        return items[-limit:]

    def subscribe(self) -> Queue:
        q: Queue = Queue(maxsize=4096)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: Queue) -> None:
        with self._lock:
            self._subs.discard(q)


_BUFFER = _LogRingBuffer()


def get_buffer() -> _LogRingBuffer:
    """Public accessor for the singleton buffer (used by admin endpoints)."""
    return _BUFFER


# ── stdout/stderr tee ────────────────────────────────────────────────

class _StreamTee(io.TextIOBase):
    """File-like that writes through to the wrapped stream AND the ring
    buffer. Wraps ``sys.stdout`` / ``sys.stderr`` so naked ``print()``
    calls + the FastAPI boot banner are captured.
    """

    def __init__(self, wrapped, level: str, source: str) -> None:
        self._wrapped = wrapped
        self._level   = level
        self._source  = source
        self._partial = ""

    def writable(self) -> bool:        # noqa: D401
        return True

    def write(self, data: str) -> int:
        try:
            self._wrapped.write(data)
        except Exception:
            pass
        # Buffer line-at-a-time so multi-line writes split into discrete events.
        self._partial += data
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            _BUFFER.append(self._level, self._source, line)
        return len(data)

    def flush(self) -> None:
        try:
            self._wrapped.flush()
        except Exception:
            pass
        if self._partial:
            _BUFFER.append(self._level, self._source, self._partial)
            self._partial = ""

    # passthroughs uvicorn / loguru may poke
    def fileno(self):   return self._wrapped.fileno()
    def isatty(self):   return self._wrapped.isatty()

    # `pipeline.py` (and other modules) call `sys.stdout.reconfigure(...)`
    # to flip the underlying encoding. Forward to the wrapped stream and
    # swallow gracefully when it isn't a TextIOWrapper (older Python versions
    # or redirected file streams).
    def reconfigure(self, **kwargs):
        fn = getattr(self._wrapped, "reconfigure", None)
        if callable(fn):
            try: return fn(**kwargs)
            except Exception: return None
        return None

    @property
    def encoding(self):
        return getattr(self._wrapped, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._wrapped, "errors", "strict")

    # Final safety net: any other attribute the runtime expects (closed,
    # buffer, mode, name, ...) falls through to the wrapped stream so we
    # never break a library that expects a "real" file-like.
    def __getattr__(self, name):
        # __getattr__ is only called when normal lookup fails, so we won't
        # recurse on the attrs we define above.
        if name.startswith("_"):
            raise AttributeError(name)
        wrapped = self.__dict__.get("_wrapped")
        if wrapped is None:
            raise AttributeError(name)
        return getattr(wrapped, name)


# ── logging handler ──────────────────────────────────────────────────

class _BufferingHandler(logging.Handler):
    """Logging handler that pushes formatted records into the ring buffer.

    Attached to the root logger + ``uvicorn`` + ``uvicorn.access`` +
    ``uvicorn.error`` so structured log calls (which bypass stdout) also
    show up in the live tail.
    """

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            level = record.levelname.lower()
            source = record.name
            msg = self.format(record)
            _BUFFER.append(level, source, msg)
        except Exception:
            pass


def install_log_capture() -> None:
    """Wire stdout/stderr tees + a logging handler that fan out to the buffer.

    Idempotent: re-imports / double-calls are safe.
    """
    # Avoid double-wrapping when uvicorn --reload restarts the module.
    if getattr(sys.stdout, "_kaizer_tee", False):
        return

    tee_out = _StreamTee(sys.stdout, level="info",  source="stdout")
    tee_err = _StreamTee(sys.stderr, level="error", source="stderr")
    tee_out._kaizer_tee = True   # type: ignore[attr-defined]
    tee_err._kaizer_tee = True   # type: ignore[attr-defined]
    sys.stdout = tee_out         # type: ignore[assignment]
    sys.stderr = tee_err         # type: ignore[assignment]

    handler = _BufferingHandler(level=logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s — %(message)s"))

    for logger_name in ("", "uvicorn", "uvicorn.access", "uvicorn.error",
                        "fastapi", "kaizer"):
        lg = logging.getLogger(logger_name)
        # Don't lower an already-permissive level set elsewhere.
        if lg.level == logging.NOTSET or lg.level > logging.INFO:
            lg.setLevel(logging.INFO)
        # Avoid attaching twice on hot-reload.
        if not any(isinstance(h, _BufferingHandler) for h in lg.handlers):
            lg.addHandler(handler)


# ── System sampler ───────────────────────────────────────────────────

def _gpu_snapshot() -> dict:
    """nvidia-smi one-shot — returns {} when no GPU or the binary is gone."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {}
    try:
        out = subprocess.run(
            [nvidia_smi,
             "--query-gpu=memory.total,memory.used,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.5,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return {}
        parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
        if len(parts) < 4:
            return {}
        return {
            "memory_total_mb": int(float(parts[0])),
            "memory_used_mb":  int(float(parts[1])),
            "utilization":     float(parts[2]),
            "temperature_c":   int(float(parts[3])),
        }
    except Exception:
        return {}


# ── Kaizer-family process rollup ─────────────────────────────────────
#
# psutil.cpu_percent() on a process is delta-based: it returns 0.0 on the
# first call and only reports a real % on subsequent calls. We cache
# Process objects in a dict so consecutive ticks see real utilisation
# instead of always zero. Stale (dead) PIDs are evicted automatically.
_FAMILY_PROCS: dict[int, psutil.Process] = {}

# Substrings we use to recognise "this is part of the Kaizer stack" by
# command line. Conservative: only patterns specific to our processes.
_FAMILY_NEEDLES = (
    "main:app",                 # uvicorn main:app
    "uvicorn",
    "pipeline.py",              # spawned by runner.py
    "kaizerFrontned",           # vite dev server cwd
    "kaizer-test",              # cloudflared tunnel name
    "cloudflared",
)


def _is_kaizer_family_cmdline(cmdline: list[str] | None) -> bool:
    if not cmdline:
        return False
    joined = " ".join(cmdline).lower()
    return any(n.lower() in joined for n in _FAMILY_NEEDLES)


def _kaizer_family_snapshot(uvicorn_proc: psutil.Process) -> dict:
    """Sum CPU%, RSS, and counts across the whole Kaizer process tree.

    Members of the family:
      1. uvicorn itself
      2. All descendants of uvicorn (pipeline.py subprocess, ffmpeg, ...)
      3. Any standalone process whose cmdline matches one of the
         ``_FAMILY_NEEDLES`` patterns — catches vite (separate cmd.exe →
         node tree) and cloudflared (started by start_kaizer.bat outside
         our parent).

    Returns ``{cpu_percent, rss_gb, proc_count, ffmpeg_count, gpu_util}``.
    GPU per-process is best-effort via ``nvidia-smi pmon`` — returns
    ``None`` when no GPU or the tool isn't on PATH.
    """
    pids: set[int] = set()
    family_procs: list[psutil.Process] = []
    ffmpeg_count = 0

    # 1+2. uvicorn + descendants
    try:
        if uvicorn_proc.is_running():
            pids.add(uvicorn_proc.pid)
            family_procs.append(uvicorn_proc)
            for child in uvicorn_proc.children(recursive=True):
                try:
                    if child.is_running():
                        pids.add(child.pid)
                        family_procs.append(child)
                        if "ffmpeg" in (child.name() or "").lower():
                            ffmpeg_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    # 3. Standalone family processes (vite, cloudflared, stray ffmpegs)
    try:
        for p in psutil.process_iter(attrs=("pid", "name", "cmdline")):
            try:
                pid = p.info["pid"]
                if pid in pids:
                    continue
                name = (p.info.get("name") or "").lower()
                cmdline = p.info.get("cmdline")
                if "ffmpeg" in name:
                    pids.add(pid); family_procs.append(p); ffmpeg_count += 1; continue
                if _is_kaizer_family_cmdline(cmdline):
                    pids.add(pid); family_procs.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass

    # Drop any cached Process whose PID isn't in the current family
    stale = [k for k in _FAMILY_PROCS if k not in pids]
    for k in stale:
        _FAMILY_PROCS.pop(k, None)

    total_cpu  = 0.0
    total_rss  = 0
    counted    = 0
    for proc in family_procs:
        try:
            # Cache the Process so cpu_percent's delta has a previous reading.
            # On first sight of a PID we prime cpu_percent (it'd return 0
            # uselessly) but we DO count RSS since it's instantaneous.
            cached = _FAMILY_PROCS.get(proc.pid)
            first_sight = cached is None or not cached.is_running()
            if first_sight:
                _FAMILY_PROCS[proc.pid] = proc
                cached = proc
                cached.cpu_percent(interval=None)   # prime, ignore result
                cpu = 0.0
            else:
                cpu = cached.cpu_percent(interval=None) or 0.0
            rss = cached.memory_info().rss
            total_cpu += cpu
            total_rss += rss
            counted += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            _FAMILY_PROCS.pop(proc.pid, None)
            continue

    # Normalise CPU to whole-machine % (psutil returns per-core summed).
    cpu_count = psutil.cpu_count(logical=True) or 1
    family_cpu_pct = min(100.0, total_cpu / cpu_count)

    # GPU per-process via nvidia-smi pmon (best-effort).
    gpu_util = _kaizer_family_gpu(pids)

    return {
        "kaizer_cpu_percent":  round(family_cpu_pct, 1),
        "kaizer_rss_gb":       round(total_rss / (1024 ** 3), 3),
        "kaizer_proc_count":   int(len(family_procs)),
        "kaizer_ffmpeg_count": int(ffmpeg_count),
        "kaizer_gpu_util":     gpu_util,
    }


def _kaizer_family_gpu(family_pids: set[int]) -> Optional[float]:
    """Sum GPU SM% across all Kaizer PIDs via ``nvidia-smi pmon -c 1``.

    `pmon` shows per-PID GPU usage. We sum the `sm` (streaming-multiprocessor
    utilisation) column for every row whose PID is in ``family_pids``.
    Returns None when no GPU, no nvidia-smi, or nothing in our family is
    touching the GPU.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi or not family_pids:
        return None
    try:
        out = subprocess.run(
            [nvidia_smi, "pmon", "-c", "1", "-s", "u"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        # Format (whitespace-separated):
        #   # gpu    pid  type    sm    mem    enc   dec   command
        #     0   1234     C     45     12      0     0   python
        total = 0.0
        any_match = False
        for line in out.stdout.splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                pid = int(parts[1])
                sm  = parts[3]
            except ValueError:
                continue
            if pid not in family_pids:
                continue
            try:
                total += float(sm)
                any_match = True
            except ValueError:
                continue
        return round(total, 1) if any_match else 0.0
    except Exception:
        return None


_PREV_NET_COUNTERS: dict = {"rx": None, "tx": None, "ts": None}


def _network_delta() -> tuple[Optional[int], Optional[int]]:
    """Bytes-per-second since the previous call. Returns (rx_bps, tx_bps)."""
    global _PREV_NET_COUNTERS
    try:
        c = psutil.net_io_counters()
    except Exception:
        return (None, None)
    now = time.monotonic()
    prev_rx, prev_tx, prev_ts = (
        _PREV_NET_COUNTERS["rx"],
        _PREV_NET_COUNTERS["tx"],
        _PREV_NET_COUNTERS["ts"],
    )
    _PREV_NET_COUNTERS = {"rx": c.bytes_recv, "tx": c.bytes_sent, "ts": now}
    if prev_rx is None or prev_ts is None or now <= prev_ts:
        return (None, None)
    dt = now - prev_ts
    return (
        max(0, int((c.bytes_recv - prev_rx) / dt)),
        max(0, int((c.bytes_sent - prev_tx) / dt)),
    )


def _take_snapshot(proc: psutil.Process) -> dict:
    """Build one row for the system_metrics table."""
    vm   = psutil.virtual_memory()
    disk = psutil.disk_usage(os.path.abspath(os.sep))
    cpu_pct = psutil.cpu_percent(interval=0.0)   # non-blocking
    gpu = _gpu_snapshot()
    rx, tx = _network_delta()
    try:
        rss = proc.memory_info().rss
    except Exception:
        rss = 0
    try:
        threads = proc.num_threads()
    except Exception:
        threads = 0

    # Live-event count — best-effort; admin/system has the same import dance.
    live = 0
    try:
        from routers.live_director import _SESSIONS   # type: ignore
        live = len(_SESSIONS)
    except Exception:
        pass

    # Kaizer-family rollup — the actual footprint of OUR stack, ignoring
    # whatever else is running on the box.
    family = _kaizer_family_snapshot(proc)

    return {
        "cpu_percent":     round(float(cpu_pct), 1),
        "cpu_count":       psutil.cpu_count(logical=True) or 0,
        "ram_percent":     round(float(vm.percent), 1),
        "ram_used_gb":     round(vm.used  / (1024 ** 3), 3),
        "ram_total_gb":    round(vm.total / (1024 ** 3), 3),
        "disk_percent":    round(float(disk.percent), 1),
        "disk_used_gb":    round(disk.used  / (1024 ** 3), 2),
        "disk_total_gb":   round(disk.total / (1024 ** 3), 2),
        "gpu_util":        gpu.get("utilization"),
        "gpu_mem_used_mb": gpu.get("memory_used_mb"),
        "gpu_mem_total_mb":gpu.get("memory_total_mb"),
        "gpu_temp_c":      gpu.get("temperature_c"),
        "proc_rss_gb":     round(rss / (1024 ** 3), 3),
        "proc_threads":    int(threads),
        "live_events":     int(live),
        "net_rx_bps":      rx,
        "net_tx_bps":      tx,
        # Kaizer-only (the rollup you size the cloud server against)
        "kaizer_cpu_percent":  family["kaizer_cpu_percent"],
        "kaizer_rss_gb":       family["kaizer_rss_gb"],
        "kaizer_proc_count":   family["kaizer_proc_count"],
        "kaizer_ffmpeg_count": family["kaizer_ffmpeg_count"],
        "kaizer_gpu_util":     family["kaizer_gpu_util"],
    }


_SAMPLER_STARTED = False


def start_metric_sampler() -> None:
    """Spawn the daemon thread that writes one system_metrics row per tick.

    Idempotent: subsequent calls are no-ops (uvicorn --reload re-imports).
    """
    global _SAMPLER_STARTED
    if _SAMPLER_STARTED:
        return
    _SAMPLER_STARTED = True

    t = threading.Thread(target=_sampler_loop, name="kaizer-metric-sampler",
                         daemon=True)
    t.start()


def _sampler_loop() -> None:
    # Local imports so the module can be imported in tests that don't have
    # the full DB stack on path.
    try:
        from database import SessionLocal
        import models
    except Exception as exc:
        _BUFFER.append("warning", "sampler",
                       f"sampler disabled — could not import DB: {exc}")
        return

    proc = psutil.Process()
    _BUFFER.append("info", "sampler",
                   f"system_observer started — interval={_SAMPLE_INTERVAL_S}s "
                   f"retention={_RETENTION_DAYS}d")

    # Prime psutil's cpu_percent so the first real sample isn't 0.
    psutil.cpu_percent(interval=None)

    last_prune = 0.0
    while True:
        try:
            snap = _take_snapshot(proc)
            db = SessionLocal()
            try:
                row = models.SystemMetric(**snap)
                db.add(row)
                db.commit()
            finally:
                db.close()
        except Exception as exc:
            # Swallow — the loop must never die.
            _BUFFER.append("warning", "sampler", f"sample failed: {exc}")

        # Prune once an hour (cheap; deletes a few thousand rows max).
        now = time.monotonic()
        if now - last_prune > 3600:
            last_prune = now
            try:
                cutoff = datetime.now(timezone.utc) - timedelta(days=_RETENTION_DAYS)
                db = SessionLocal()
                try:
                    deleted = (db.query(models.SystemMetric)
                                 .filter(models.SystemMetric.ts < cutoff)
                                 .delete(synchronize_session=False))
                    db.commit()
                    if deleted:
                        _BUFFER.append("info", "sampler",
                                       f"pruned {deleted} samples older than "
                                       f"{_RETENTION_DAYS}d")
                finally:
                    db.close()
            except Exception as exc:
                _BUFFER.append("warning", "sampler", f"prune failed: {exc}")

        time.sleep(_SAMPLE_INTERVAL_S)
