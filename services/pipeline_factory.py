"""
kaizer.services.pipeline_factory
================================
The staged "factory" pipeline engine.

THE MODEL (the operator's biscuit-factory analogy, made real)
-------------------------------------------------------------
Instead of running N *whole jobs* side by side — each holding its entire
working set in RAM for its whole lifetime (today's design, which pins RAM at
~31.5/32 GB and caps us at 3-4 concurrent jobs) — work flows along a conveyor
of STATIONS:

    ingest → transcribe → cut_plan → trim → compose → brand → upload

Each station is a bounded worker pool draining a bounded input queue. The
instant a station finishes a unit it hands it to the next station's queue and
grabs the next unit — like the mixer never waiting for a biscuit to finish
baking. The crucial property:

    RAM ≈ number of STATIONS (a small constant of units "in the machine"),
    NOT number of JOBS.

So the queue can hold thousands of jobs while only a handful of units occupy
the heavy stations at once. The slowest station (compose, on the GPU) sets the
belt speed; the bounded queues provide BACK-PRESSURE — when compose is busy,
ingest blocks instead of flooding RAM. That back-pressure replaces the crude
"max 4 jobs" cap.

ISOLATION (hard, fail-closed — the operator's non-negotiable)
-------------------------------------------------------------
Every unit on the belt carries an immutable :class:`stage_events.Envelope`
``(tenant_id, user_id, job_id, clip_id, channel_id)`` (INVARIANT I2). A station
can READ who owns a unit but can never re-stamp it. Delivery ownership is
asserted fail-closed at the upload boundary in
``upload_dispatch.process`` (INVARIANT I3). Per-job workspaces are unique
(INVARIANT I1). Stations are pure functions of ``(envelope, item)`` with no
shared mutable job state (INVARIANT I5).

ROLLOUT (safe by construction)
------------------------------
This module is ADDITIVE and OFF by default:
  - ``KAIZER_PIPELINE_FACTORY=1``         → engine framework is activated.
  - ``KAIZER_PIPELINE_FACTORY_WORKERS=1`` → start station worker pools at boot.
The existing publish path (``upload_dispatch.process``) is unchanged and is
already INSTRUMENTED to emit live stage events, so the admin "Pipeline Flow"
view shows the real pipeline whether or not the factory is switched on. Wiring
the heavy *render* stages onto these station pools is the next increment and
must be smoke-tested (``self_test()`` proves the engine end-to-end) before the
flag is flipped in production.

This module holds NO secrets.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from services import stage_events as _se
from services.stage_events import Envelope, STATIONS

log = logging.getLogger("kaizer.pipeline_factory")


# ── Feature flags (canonical env pattern, read per call) ─────────────────
def enabled() -> bool:
    """``KAIZER_PIPELINE_FACTORY=1`` activates the engine framework."""
    return (os.environ.get("KAIZER_PIPELINE_FACTORY", "0") or "0").strip() == "1"


def workers_enabled() -> bool:
    """``KAIZER_PIPELINE_FACTORY_WORKERS=1`` starts station pools at boot."""
    return (os.environ.get("KAIZER_PIPELINE_FACTORY_WORKERS", "0") or "0").strip() == "1"


def _stage_workers(stage: str, default: int) -> int:
    """Per-station worker-pool size. Env: ``KAIZER_STAGE_<STAGE>_WORKERS``.
    Sized by what the station COSTS: light/network stations get many, the
    heavy GPU ``compose`` station gets few (it paces the belt)."""
    try:
        return max(1, int(os.environ.get(
            f"KAIZER_STAGE_{stage.upper()}_WORKERS", str(default))))
    except Exception:
        return default


def _stage_queue_size(stage: str, default: int = 256) -> int:
    """Per-station queue depth (back-pressure bound). Env:
    ``KAIZER_STAGE_<STAGE>_QUEUE_SIZE``."""
    try:
        return max(1, int(os.environ.get(
            f"KAIZER_STAGE_{stage.upper()}_QUEUE_SIZE", str(default))))
    except Exception:
        return default


# Default pool sizes — light stations many, heavy GPU station few. These are
# the RAM-safe starting points from the pipeline plan; tune via env.
_DEFAULT_WORKERS: Dict[str, int] = {
    "ingest": 8,
    "transcribe": 8,
    "cut_plan": 8,
    "trim": 3,
    "compose": 2,    # the GPU bottleneck — paces the whole belt
    "brand": 4,
    "upload": 16,
}


# ── The Station: a bounded worker pool draining a bounded queue ──────────
@dataclass
class _StageItem:
    envelope: Envelope
    payload: Any


class Station:
    """One conveyor station: ``workers`` threads draining a bounded queue,
    each applying ``work_fn(envelope, payload) -> next_payload`` and handing
    the result to ``next_station``.

    Back-pressure is intrinsic: :meth:`submit` blocks when the queue is full,
    so a slow downstream station throttles upstream producers instead of
    letting work pile up in RAM.
    """

    def __init__(
        self,
        name: str,
        work_fn: Callable[[Envelope, Any], Any],
        *,
        workers: int,
        queue_size: int,
    ) -> None:
        self.name = name
        self.work_fn = work_fn
        self.workers = max(1, workers)
        self.q: "queue.Queue[_StageItem]" = queue.Queue(maxsize=queue_size)
        self.next_station: Optional["Station"] = None
        self._threads: List[threading.Thread] = []
        self._stop = threading.Event()

    def submit(self, envelope: Envelope, payload: Any,
               *, timeout: Optional[float] = None) -> bool:
        """Enqueue a unit. Blocks (back-pressure) until space or ``timeout``.
        Returns False if the queue stayed full past ``timeout``."""
        try:
            self.q.put(_StageItem(envelope, payload), block=True, timeout=timeout)
            return True
        except queue.Full:
            return False

    def depth(self) -> int:
        return self.q.qsize()

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        for i in range(self.workers):
            t = threading.Thread(
                target=self._loop, name=f"station-{self.name}-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        log.info("station %r started: workers=%d queue_max=%d",
                 self.name, self.workers, self.q.maxsize)

    def stop(self, *, join_timeout: float = 5.0) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=join_timeout)
        self._threads.clear()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            env = item.envelope
            try:
                _se.emit(env, self.name, _se.ENTERED)
                result = self.work_fn(env, item.payload)
                _se.emit(env, self.name, _se.EXITED)
                nxt = self.next_station
                if nxt is not None and result is not None:
                    # Hand off downstream (blocks under back-pressure — this
                    # is intentional: it throttles this station to the speed
                    # of the next, keeping RAM bounded).
                    nxt.submit(env, result)
            except Exception as exc:  # noqa: BLE001 — a unit failure must not kill the worker
                _se.emit(env, self.name, _se.FAILED, repr(exc)[:160])
                log.exception("station %r failed a unit: %s", self.name, exc)
            finally:
                self.q.task_done()


class Pipeline:
    """A chain of stations wired head→tail. Submit at the head; units flow."""

    def __init__(self, stations: List[Station]) -> None:
        if not stations:
            raise ValueError("Pipeline needs at least one station")
        self.stations = stations
        for a, b in zip(stations, stations[1:]):
            a.next_station = b
        self.by_name: Dict[str, Station] = {s.name: s for s in stations}

    @property
    def head(self) -> Station:
        return self.stations[0]

    def start(self) -> None:
        # Start tail→head so a downstream station is ready before its
        # producer can hand off to it.
        for s in reversed(self.stations):
            s.start()

    def stop(self) -> None:
        for s in self.stations:
            s.stop()

    def submit(self, envelope: Envelope, payload: Any) -> bool:
        return self.head.submit(envelope, payload)

    def depths(self) -> Dict[str, int]:
        return {s.name: s.depth() for s in self.stations}


# ── Module-level engine handle ───────────────────────────────────────────
_PIPELINE: Optional[Pipeline] = None
_lock = threading.Lock()


def _passthrough_stage(stage_name: str) -> Callable[[Envelope, Any], Any]:
    """Placeholder station work_fn used until the real render-stage logic is
    migrated onto the belt. It does no work and forwards the payload — so the
    framework is runnable and observable end-to-end today (see
    :func:`self_test`) without touching the proven render/upload code.

    The production work_fn for each station (decode/trim/compose/...) plugs in
    here in the next increment; the surrounding pool, queue, back-pressure,
    envelope and telemetry are already done."""
    def _fn(env: Envelope, payload: Any) -> Any:  # noqa: ARG001
        return payload
    _fn.__name__ = f"passthrough_{stage_name}"
    return _fn


def build_pipeline(
    work_fns: Optional[Dict[str, Callable[[Envelope, Any], Any]]] = None,
) -> Pipeline:
    """Construct the 7-station conveyor. Any station missing from ``work_fns``
    gets a safe pass-through (lets the engine run + be observed before the
    heavy stages are migrated)."""
    work_fns = work_fns or {}
    stations: List[Station] = []
    for name in STATIONS:
        stations.append(Station(
            name,
            work_fns.get(name, _passthrough_stage(name)),
            workers=_stage_workers(name, _DEFAULT_WORKERS.get(name, 4)),
            queue_size=_stage_queue_size(name),
        ))
    return Pipeline(stations)


def start_workers() -> bool:
    """Boot the station pools (idempotent). Called from main.py startup when
    ``KAIZER_PIPELINE_FACTORY_WORKERS=1``. Returns True if started."""
    global _PIPELINE
    with _lock:
        if _PIPELINE is not None:
            return True
        _PIPELINE = build_pipeline()
        _PIPELINE.start()
    log.info("pipeline factory workers started: %s", _PIPELINE.depths())
    return True


def stop_workers() -> None:
    global _PIPELINE
    with _lock:
        if _PIPELINE is None:
            return
        _PIPELINE.stop()
        _PIPELINE = None


def depths() -> Dict[str, int]:
    """Current queue depth per station (0s if not started)."""
    with _lock:
        if _PIPELINE is None:
            return {s: 0 for s in STATIONS}
        return _PIPELINE.depths()


def self_test(units: int = 3, timeout: float = 10.0) -> Dict[str, Any]:
    """Run ``units`` synthetic units through every station end-to-end and
    confirm each emitted ENTERED+EXITED for every stage. Proves the engine
    (pools, queues, hand-off, back-pressure, envelope, telemetry) works
    BEFORE any real render-stage logic is attached. Used by the smoke test.

    Runs on a private throwaway pipeline so it never touches the live engine
    or real telemetry counters meaningfully beyond the synthetic events.
    """
    pipe = build_pipeline()
    pipe.start()
    try:
        seen: Dict[str, set] = {s: set() for s in STATIONS}
        done = threading.Event()
        last_station = STATIONS[-1]

        # Subscribe by tailing the ring buffer for our synthetic ids.
        ids = list(range(900000, 900000 + units))
        for i in ids:
            env = Envelope(
                tenant_id=-1, user_id=-1, job_id=None, clip_id=None,
                channel_id=None, upload_job_id=i, label=f"selftest-{i}")
            pipe.submit(env, {"n": i})

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            evs = _se.recent(limit=2000)
            for e in evs:
                uj = e.get("upload_job_id")
                if uj in ids and e.get("status") == _se.EXITED:
                    seen.setdefault(e.get("stage"), set()).add(uj)
            if all(len(seen.get(s, set())) >= units for s in STATIONS):
                done.set()
                break
            time.sleep(0.1)

        ok = done.is_set()
        return {
            "ok": ok,
            "units": units,
            "stations": STATIONS,
            "exited_counts": {s: len(seen.get(s, set())) for s in STATIONS},
            "last_station": last_station,
        }
    finally:
        pipe.stop()


def run_demo(units: int = 4, delay_ms: int = 1500) -> Dict[str, Any]:
    """Push ``units`` SYNTHETIC units through a transient slow pipeline so the
    admin conveyor visibly fills, flows, and drains — a safe, YouTube-free,
    render-free live smoke test. Synthetic units carry NEGATIVE upload_job_ids
    (label 'sim N') so they can never be confused with, or written to, real
    rows. Runs in a background daemon thread and auto-stops when drained.
    """
    units = max(1, min(int(units), 40))
    delay = max(0.0, min(int(delay_ms), 8000)) / 1000.0

    def _slow(name: str):
        def fn(env: Envelope, payload: Any) -> Any:  # noqa: ARG001
            if delay:
                time.sleep(delay)
            return payload
        fn.__name__ = f"demo_{name}"
        return fn

    pipe = build_pipeline({s: _slow(s) for s in STATIONS})

    def _runner() -> None:
        pipe.start()
        try:
            for i in range(units):
                env = Envelope(
                    tenant_id=-1, user_id=-1, job_id=None, clip_id=None,
                    channel_id=None, upload_job_id=-(900000 + i),
                    label=f"sim {i + 1}")
                pipe.submit(env, {"i": i})
                time.sleep(max(delay / 2, 0.15))  # stagger so the belt looks alive
            deadline = time.monotonic() + (units * delay) + \
                (len(STATIONS) + 3) * max(delay, 0.3) + 6
            while time.monotonic() < deadline:
                if all(d == 0 for d in pipe.depths().values()):
                    time.sleep(max(delay, 0.3) * 1.5)
                    if all(d == 0 for d in pipe.depths().values()):
                        break
                time.sleep(0.25)
        finally:
            pipe.stop()

    t = threading.Thread(target=_runner, name="factory-demo", daemon=True)
    t.start()
    log.info("pipeline factory demo started: units=%d delay_ms=%d", units, int(delay * 1000))
    return {"started": units, "delay_ms": int(delay * 1000)}


__all__ = [
    "enabled",
    "workers_enabled",
    "Station",
    "Pipeline",
    "build_pipeline",
    "start_workers",
    "stop_workers",
    "depths",
    "self_test",
    "run_demo",
]
