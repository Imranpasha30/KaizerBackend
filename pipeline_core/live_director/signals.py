"""
kaizer.pipeline.live_director.signals
======================================
Stable shared dataclasses used by every Phase-6 layer:
ingest, analyzers, director, composer, and frontend WebSocket bridge.

Define once here; import everywhere. Never mutate these after creation —
treat as value objects. All fields carry defaults so callers can do
partial construction and fill in the rest incrementally.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("kaizer.pipeline.live_director.signals")


# ─────────────────────────────────────────────────────────────────────────────
# §6.1.3 — Stable signal types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CameraConfig:
    """Static configuration for one camera feed in a live event.

    id          : Short machine-readable identifier, e.g. ``"cam_stage_left"``.
    label       : Human display name, e.g. ``"Stage Left Cam"``.
    mic_id      : Optional mic feed ID. ``None`` = use camera's embedded audio.
    role_hints  : Free-form tags that bias director rules, e.g.
                  ``["stage", "wide", "closeup_artist_1"]``.
    """

    id: str
    label: str
    mic_id: Optional[str] = None
    role_hints: list[str] = field(default_factory=list)


@dataclass
class SignalFrame:
    """One analysis snapshot for a single camera at a point in time.

    Produced by analyzers and published to the SignalBus. The director
    consumes a stream of these to make cut decisions.

    cam_id          : Which camera this snapshot belongs to.
    t               : Monotonic seconds since event start.
    audio_rms       : Normalised RMS energy 0.0–1.0.
    vad_speaking    : True when a human voice is detected (webrtcvad).
    face_present    : True when at least one face is detected in frame.
    face_size_norm  : Largest face bounding-box area ÷ total frame area (0–1).
    face_identity   : Artist label from the per-event embedding registry, or None.
    scene           : Classifier output: 'stage' | 'crowd' | 'wide' |
                      'closeup' | 'graphic' | 'unknown'.
    motion_mag      : Optical-flow magnitude, normalised 0.0–1.0.
    reaction        : Audio-event tag: 'laugh' | 'cheer' | 'clap' | 'boo' | None.
    beat_phase      : Fractional position within the current musical bar (0.0–1.0),
                      or None if beat tracking is not active.
    """

    cam_id: str
    t: float
    audio_rms: float = 0.0
    vad_speaking: bool = False
    face_present: bool = False
    face_size_norm: float = 0.0
    face_identity: Optional[str] = None
    scene: str = "unknown"
    motion_mag: float = 0.0
    reaction: Optional[str] = None
    beat_phase: Optional[float] = None


@dataclass
class CameraSelection:
    """A director decision: which camera to put on-program at time *t*.

    t           : Monotonic seconds since event start.
    cam_id      : The selected camera's ID.
    transition  : 'cut' (v1 only) | 'dissolve' (v2).
    confidence  : 0.0–1.0 score from the winning rule.
    reason      : Human-readable explanation, e.g.
                  ``"artist speaking + face in frame"``.
    """

    t: float
    cam_id: str
    transition: str = "cut"
    confidence: float = 0.0
    reason: str = ""


@dataclass
class DirectorEvent:
    """Envelope type for all events emitted by the director engine.

    Delivered over the live WebSocket to the control surface.

    t       : Monotonic seconds since event start.
    kind    : 'selection' | 'override' | 'camera_lost' | 'health'
    payload : Kind-specific dict (CameraSelection fields, override cam_id,
              health metrics, etc.).
    """

    t: float
    kind: str          # 'selection' | 'override' | 'camera_lost' | 'health'
    payload: dict = field(default_factory=dict)
