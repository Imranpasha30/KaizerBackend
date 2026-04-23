"""
kaizer.pipeline.live_director
================================
Phase 6 — Autonomous Live Director subsystem.

Package structure
-----------------
signals.py    — Shared dataclasses (CameraConfig, SignalFrame,
                CameraSelection, DirectorEvent).
ring_buffer.py — Async-safe per-camera frame + audio ring buffer.
ingest.py     — RTMP ingestion worker (one asyncio task per camera).
signal_bus.py — Asyncio fan-in / fan-out for SignalFrame objects.

Subsequent waves (analyzers/, director.py, composer.py, output.py)
all consume these primitives.
"""
from __future__ import annotations

from pipeline_core.live_director.signals import (
    CameraConfig,
    CameraSelection,
    DirectorEvent,
    SignalFrame,
)
from pipeline_core.live_director.ring_buffer import CameraRingBuffer
from pipeline_core.live_director.ingest import IngestWorker
from pipeline_core.live_director.signal_bus import SignalBus

__all__ = [
    "CameraConfig",
    "CameraSelection",
    "DirectorEvent",
    "SignalFrame",
    "CameraRingBuffer",
    "IngestWorker",
    "SignalBus",
]
