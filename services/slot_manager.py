"""Slot Manager — Phase 1.C (Scheduler agent).

Resource book-keeper for the weighted-fair scheduler. Tracks two
in-process token pools (CPU + network) plus a per-user
``BoundedSemaphore`` keyed by ``user_id`` whose capacity is the user's
plan-tier ``slot_cap_active_uploads`` (Decision 3:
free=5 / pro=20 / enterprise=100).

Token totals come from env at construct time (Decision 9):
  - ``KAIZER_SCHED_CPU_TOKENS``  (default 4)
  - ``KAIZER_SCHED_NET_TOKENS``  (default 16)

CPU tokens are reserved for the Phase 2 Branding Worker (overlay pass);
network tokens are reserved for upload workers. In Phase 1 the scheduler
synthesises only the upload step, so only the network pool is exercised
by the smoke test — the CPU pool is constructed identically so the
Phase 2 D-agent can plug straight in.

A per-user semaphore is rebuilt only when its cap changes (e.g. a plan
upgrade). The previous semaphore object stays alive while any in-flight
jobs hold tokens on it; the GC reclaims it after they release. We do
NOT try to migrate held tokens to the new sem — that would require
coordination this Phase doesn't have. Phase 3 may revisit.

All public methods are async. ``acquire`` is an
``@asynccontextmanager`` so callers say::

    async with slot_manager.acquire(user_id, cap, 'network'):
        ... do the work ...

and release happens in the ``finally`` even on cancellation. Order of
acquisition is per-user-slot first, then the global token, so two
simultaneous Free users requesting all 5 tokens each don't deadlock at
the global pool while neither has crossed their own gate.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

log = logging.getLogger("kaizer.scheduler.slot_manager")

TokenKind = Literal["cpu", "network"]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        n = int(raw)
        if n <= 0:
            log.warning(
                "slot_manager: env %s=%r is non-positive; falling back to %d",
                name, raw, default,
            )
            return default
        return n
    except ValueError:
        log.warning(
            "slot_manager: env %s=%r is not an int; falling back to %d",
            name, raw, default,
        )
        return default


def cpu_tokens_from_env() -> int:
    return _env_int("KAIZER_SCHED_CPU_TOKENS", 4)


def net_tokens_from_env() -> int:
    return _env_int("KAIZER_SCHED_NET_TOKENS", 16)


class SlotManager:
    """In-process semaphore wrapper.

    Two global ``asyncio.Semaphore`` pools (CPU, network) plus a dict
    of per-user ``BoundedSemaphore`` instances.
    """

    def __init__(self, cpu_tokens: int, net_tokens: int) -> None:
        if cpu_tokens <= 0 or net_tokens <= 0:
            raise ValueError(
                f"SlotManager requires positive token totals; "
                f"got cpu={cpu_tokens} net={net_tokens}"
            )
        self._cpu = asyncio.Semaphore(cpu_tokens)
        self._net = asyncio.Semaphore(net_tokens)
        self._cpu_total = cpu_tokens
        self._net_total = net_tokens
        self._cpu_in_use = 0
        self._net_in_use = 0
        # Per-user semaphores. Replaced (NOT mutated) when a user's cap
        # changes — see _user_sem.
        self._user_sems: dict[int, asyncio.BoundedSemaphore] = {}
        self._user_caps: dict[int, int] = {}
        self._user_in_use: dict[int, int] = {}
        # Guards the in-use counters + dict mutations.
        self._lock = asyncio.Lock()

    # ─── Per-user semaphore lookup ─────────────────────────────────────

    def _user_sem(self, user_id: int, cap: int) -> asyncio.BoundedSemaphore:
        """Return (or create) the user's semaphore.

        If the cap has changed since last time, build a fresh sem.
        Existing holders of the previous sem will release into it
        normally; the GC reclaims it once no references remain. New
        acquirers will block on the new sem.
        """
        if cap <= 0:
            # Defensive fallback — should never happen but a 0 cap would
            # mean "no one can ever upload for this user" which is worse
            # than letting the Free default through.
            log.warning(
                "slot_manager: user_id=%d got non-positive cap=%d; using Free fallback 5",
                user_id, cap,
            )
            cap = 5
        sem = self._user_sems.get(user_id)
        if sem is None or self._user_caps.get(user_id) != cap:
            sem = asyncio.BoundedSemaphore(cap)
            self._user_sems[user_id] = sem
            self._user_caps[user_id] = cap
            # Reset the in-use counter for snapshot purposes — held
            # tokens on the OLD sem don't affect the new one's count.
            self._user_in_use[user_id] = 0
            log.debug(
                "slot_manager: user_id=%d sem (re)built with cap=%d", user_id, cap,
            )
        return sem

    # ─── Public acquire context manager ────────────────────────────────

    @asynccontextmanager
    async def acquire(
        self, user_id: int, user_cap: int, token_kind: TokenKind,
    ):
        """Acquire BOTH the per-user slot AND a global token of the
        given kind. Releases both on context exit. Cancellation-safe.

        Order matters: per-user slot first, then global token. This
        avoids a deadlock where two users sat on the global pool while
        neither held their own slot.
        """
        if token_kind not in ("cpu", "network"):
            raise ValueError(f"token_kind must be 'cpu' or 'network'; got {token_kind!r}")
        user_sem = self._user_sem(user_id, user_cap)
        await user_sem.acquire()
        async with self._lock:
            self._user_in_use[user_id] = self._user_in_use.get(user_id, 0) + 1
        try:
            pool_sem = self._cpu if token_kind == "cpu" else self._net
            await pool_sem.acquire()
            async with self._lock:
                if token_kind == "cpu":
                    self._cpu_in_use += 1
                else:
                    self._net_in_use += 1
            try:
                yield
            finally:
                async with self._lock:
                    if token_kind == "cpu":
                        self._cpu_in_use -= 1
                    else:
                        self._net_in_use -= 1
                pool_sem.release()
        finally:
            async with self._lock:
                self._user_in_use[user_id] = max(
                    0, self._user_in_use.get(user_id, 0) - 1
                )
            user_sem.release()

    # ─── Observability ─────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Synchronous read-only snapshot of current pool state.

        Phase 1 returns plain counters; the G-agent's Prometheus
        collectors will hook into the same instance in Phase 3.
        """
        return {
            "cpu_total": self._cpu_total,
            "cpu_in_use": self._cpu_in_use,
            "cpu_free": self._cpu_total - self._cpu_in_use,
            "net_total": self._net_total,
            "net_in_use": self._net_in_use,
            "net_free": self._net_total - self._net_in_use,
            "user_caps": dict(self._user_caps),
            "user_in_use": dict(self._user_in_use),
        }


__all__ = [
    "SlotManager",
    "TokenKind",
    "cpu_tokens_from_env",
    "net_tokens_from_env",
]
