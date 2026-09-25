"""Process-wide pacing gate for burst-prone Gemini calls.

Job 611 (16 stories) fired 16 image-prompt calls and 16 image-timing
calls in a burst; Vertex enforces a PER-MINUTE quota, so nearly every
call 429'd, slept a blind 30s, frequently 429'd again, and fell back to
heuristics — ~13 minutes of Stage 2 spent colliding, with a QUALITY loss
on every story that fell back (no word-synced image placement).

The same total work, spaced at 60/RPM seconds per request, completes in
~N*gap with zero collisions. This gate serializes the REQUEST STARTS
(token-bucket-of-one with a monotonic next-slot clock) across all
threads of the orchestrator process; callers invoke ``pace()`` right
before a Gemini request. Env: KAIZER_GEMINI_RPM (default 10; 0 disables
pacing). Fail-open and dependency-free.
"""
from __future__ import annotations

import os
import threading
import time

_lock = threading.Lock()
_next_slot = 0.0


def _rpm() -> float:
    try:
        return float(os.environ.get("KAIZER_GEMINI_RPM", "10") or "10")
    except (TypeError, ValueError):
        return 10.0


def pace() -> float:
    """Block until this thread may start a Gemini request; returns the
    seconds actually slept. No-op (0.0) when pacing is disabled."""
    rpm = _rpm()
    if rpm <= 0:
        return 0.0
    gap = 60.0 / rpm
    global _next_slot
    with _lock:
        now = time.monotonic()
        wait = max(0.0, _next_slot - now)
        _next_slot = max(now, _next_slot) + gap
    if wait > 0:
        time.sleep(wait)
    return wait
