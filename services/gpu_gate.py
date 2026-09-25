"""ONE process-wide gate for ad-hoc GPU renders outside the V4 queue.

This host HARD-RESETS under concurrent GPU load (operator rule: one render
at a time). The avatar router (MuseTalk/EchoMimic — minutes of GPU) and the
podcast router (up to three sequential NVENC ffmpeg encodes) each shipped
their own module-local Semaphore(1); two private gates in two modules never
serialize against EACH OTHER. Both routers must import THIS one.

Known limit (documented, accepted v1): the gate is per-process — it cannot
see the V4 render queue's separate orchestrator processes. Queuing policy
across the whole box is the OPERATOR'S open concurrency decision; until
then, keep ad-hoc renders serialized among themselves and rely on the
operator running one pipeline job at a time.
"""
from __future__ import annotations

import threading

# Acquire around any local (non-remote-API) generate/render call.
GPU_GATE = threading.Semaphore(1)
