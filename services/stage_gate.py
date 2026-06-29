"""
kaizer.services.stage_gate
==========================
Cross-process **and cross-machine** counting semaphore for the heavy render
stages — the factory's core mechanism applied to Kaizer's real architecture
(V4 render runs as one subprocess per job).

Why this exists
---------------
The render bottleneck is the single NVENC encoder + the RAM spike in the
compose/canvas stage. Today the *whole job* is capped by one semaphore
(``KAIZER_PIPELINE_CONCURRENCY``, default 2) — so a job that is merely
transcribing (network, no GPU, no RAM spike) still occupies a precious encode
slot. That throttles throughput badly.

The fix (the factory principle): **bound only the heavy stage**, and let the
light stages run many-at-once. We raise overall admission
(``KAIZER_PIPELINE_CONCURRENCY``) and wrap the encode-heavy work in a gate of
size ``KAIZER_STAGE_ENCODE_MAX`` (≈ the NVENC count, default 2). Result: many
jobs flow through ingest/transcribe/cut-plan concurrently while at most N are
encoding at once — encoder protected, RAM bounded, throughput up.

Cross-machine by construction
-----------------------------
The gate uses **PostgreSQL session-level advisory locks**
(``pg_try_advisory_lock``). Because every render worker — on this box or any
other — shares the same Postgres, the gate bounds total simultaneous encodes
**across the whole fleet**. That is the lever from 6 → 100 → 2000: add machines
pointed at the same Postgres + R2; the gate keeps the cluster's encoder/RAM
within policy. Tune the bound per deployment.

Safety
------
- OFF by default (``KAIZER_STAGE_GATE`` unset/0) → :func:`gate` is a no-op, so
  the render path is byte-identical to today until you switch it on.
- A dedicated DB connection is held only for the gated section and the lock is
  explicitly released; if Postgres is unavailable the gate degrades to a no-op
  (never blocks a render on telemetry/coordination failure).
- Holds NO secrets.
"""

from __future__ import annotations

import os
import socket
import time
import zlib
from contextlib import contextmanager
from typing import Optional


def enabled() -> bool:
    """``KAIZER_STAGE_GATE=1`` activates the cross-process/machine gate."""
    return (os.environ.get("KAIZER_STAGE_GATE", "0") or "0").strip() == "1"


# Small per-gate slot so different stages never collide within a host range.
_GATE_SLOT = {"encode": 1000, "compose": 2000, "trim": 3000}


def _host_base() -> int:
    """Per-host key base. NVENC is a PHYSICAL per-machine resource, so by
    default each host gets its OWN semaphore (independent slots) even when many
    machines share one Postgres — that's what lets the fleet scale: 10 boxes ×
    encode_max=2 = 20 simultaneous encodes cluster-wide, 2 per box.

    Set ``KAIZER_STAGE_GATE_SCOPE=global`` for a resource that is genuinely
    shared across the fleet (none today). crc32(hostname) is STABLE across all
    processes on a host (unlike hash()), so every render subprocess on the same
    box maps to the same slots → correct mutual exclusion."""
    scope = (os.environ.get("KAIZER_STAGE_GATE_SCOPE", "host") or "host").strip().lower()
    if scope == "global":
        return 0
    # Keep the final key < 2^31 so the single-bigint advisory lock maps to
    # classid=0/objid=key in pg_locks (clean status() lookup). 20000 host
    # slots is ample for a real fleet; collisions are birthday-rare.
    return (zlib.crc32(socket.gethostname().encode("utf-8")) % 20000) * 100000


def _key(name: str, i: int) -> int:
    return _host_base() + _GATE_SLOT.get(name, 9000) + i


def _max_for(name: str) -> int:
    """Slots for a gate. Env: ``KAIZER_STAGE_<NAME>_MAX``. Default 2 (one
    mid-range NVENC engine; >2 simultaneous encodes thrash it)."""
    try:
        return max(0, int(os.environ.get(f"KAIZER_STAGE_{name.upper()}_MAX", "2")))
    except Exception:
        return 2


def acquire(name: str = "encode", *, poll: float = 0.75,
            timeout: Optional[float] = None):
    """Acquire one slot of the named gate. Blocks (back-pressure) until a slot
    is free across the whole fleet, or ``timeout`` seconds elapse. Returns an
    opaque token to pass to :func:`release`, or ``None`` when the gate is
    disabled / unavailable (caller proceeds ungated)."""
    if not enabled():
        return None
    max_n = _max_for(name)
    if max_n <= 0:
        return None
    try:
        from database import engine
        from sqlalchemy import text
    except Exception:
        return None
    try:
        conn = engine.connect()
    except Exception:
        return None
    deadline = None if timeout is None else (time.monotonic() + timeout)
    try:
        while True:
            for i in range(max_n):
                key = _key(name, i)
                try:
                    got = conn.execute(
                        text("SELECT pg_try_advisory_lock(:k)"), {"k": key}
                    ).scalar()
                except Exception:
                    # DB hiccup — fail open so a render never hangs on the gate.
                    try:
                        conn.close()
                    except Exception:
                        pass
                    return None
                if got:
                    return {"conn": conn, "key": key, "name": name}
            if deadline is not None and time.monotonic() > deadline:
                try:
                    conn.close()
                except Exception:
                    pass
                return None
            time.sleep(poll)
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return None


def release(token) -> None:
    """Release a slot acquired via :func:`acquire`. Safe on ``None``."""
    if not token:
        return
    conn = token.get("conn")
    key = token.get("key")
    try:
        from sqlalchemy import text
        if conn is not None and key is not None:
            conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})
    except Exception:
        pass
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass


@contextmanager
def gate(name: str = "encode", *, poll: float = 0.75,
         timeout: Optional[float] = None):
    """Context-manager form: ``with stage_gate.gate("encode"): <heavy work>``.
    No-op when disabled."""
    tok = acquire(name, poll=poll, timeout=timeout)
    try:
        yield tok
    finally:
        release(tok)


def status() -> dict:
    """Best-effort: how many slots of each gate are currently held (for the
    admin view / ops). Counts locks in pg_locks for our key ranges."""
    out = {"enabled": enabled(), "gates": {}}
    if not enabled():
        return out
    try:
        from database import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            for name in _GATE_SLOT:
                mx = _max_for(name)
                lo = _key(name, 0)
                hi = lo + max(mx, 1)
                held = conn.execute(
                    text("SELECT count(*) FROM pg_locks WHERE locktype='advisory' "
                         "AND classid=0 AND objid >= :lo AND objid < :hi"),
                    {"lo": lo, "hi": hi},
                ).scalar() or 0
                out["gates"][name] = {"held": int(held), "max": mx}
    except Exception:
        pass
    return out


__all__ = ["enabled", "acquire", "release", "gate", "status"]
