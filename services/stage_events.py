"""
kaizer.services.stage_events
============================
Lightweight, thread-safe in-memory event backbone for the staged "factory"
pipeline and the admin "Pipeline Flow" live view.

Why in-memory (not a DB table or Redis pub/sub)?
  - Single-box Windows deployment: a WebSocket fan-out per viewer is overkill.
  - The admin view only needs "what is happening right now + the last few
    hundred transitions", which a bounded ring buffer serves in O(1) append /
    O(N) snapshot under one lock (the exact pattern proven by
    ``services.branding._snapshot`` and ``system_observer._LogRingBuffer``).
  - Redis pub/sub is transient (events vanish with no subscriber); the ring
    buffer survives between polls so the 1 s admin poll never misses a beat.

Every event carries the IMMUTABLE IDENTITY ENVELOPE
``(tenant_id, user_id, job_id, clip_id, channel_id)`` so the live view — and
any later forensic dump — can always attribute a unit of work to exactly one
user. This is ISOLATION INVARIANT I2 made observable: a unit's identity rides
with it through every station and is stamped on every event.

This module holds NO secrets and NO job payload — only ids, a stage name, a
status, and a short human message. It must never raise into a caller: a
telemetry failure can never break a render or an upload.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

# ── Canonical station order (the conveyor) ───────────────────────────────
# The single source of truth for the factory's stations AND the admin view's
# swim-lane order. Render stations produce the master/clips; publish stations
# brand + deliver per channel. Keep this list and the labels in sync with the
# frontend AdminPipelineFlow.jsx STATIONS constant.
STATIONS: List[str] = [
    "ingest",
    "transcribe",
    "cut_plan",
    "trim",
    "compose",
    "brand",
    "upload",
]

STATION_LABELS: Dict[str, str] = {
    "ingest": "Ingest",
    "transcribe": "Transcribe",
    "cut_plan": "Cut plan",
    "trim": "Trim / encode",
    "compose": "Compose",
    "brand": "Brand overlay",
    "upload": "Upload",
}

# ── Extra conveyor LANES (separate belts in the admin view) ──────────────
# The main render+delivery pipeline above is the "render" lane. These extra
# lanes get their OWN belts so editor re-renders and SEO generation are tracked
# independently of fresh job renders. Each lane has its own occupancy store
# (``_lane_occupancy``) so it never mixes with the render belt.
LANE_DEFS: Dict[str, Dict[str, Any]] = {
    "rerender": {
        "label": "Re-render (editor)",
        "stations": ["slice", "compose", "encode"],
        "labels": {"slice": "Slice", "compose": "Compose", "encode": "Encode"},
    },
    "seo": {
        "label": "SEO pipeline",
        # "compare" = head-to-head-with-a-YouTube-video + regenerate-from-that
        # comparison; it precedes a fresh generate in the editor's SEO flow.
        "stations": ["compare", "generate", "score", "per_channel", "platform"],
        "labels": {"compare": "Compare", "generate": "Generate", "score": "Score",
                   "per_channel": "Per-channel", "platform": "Platform"},
    },
    # Editor "Render all channels" — one vehicle per channel, each gets its own
    # branded clip (logo + watermark + zoom + nudge + that channel's intro).
    "channel_render": {
        "label": "Channel videos (editor)",
        "stations": ["queued", "rendering", "ready"],
        "labels": {"queued": "Queued", "rendering": "Rendering", "ready": "Ready"},
    },
}

# Event status verbs.
ENTERED = "entered"
PROGRESSED = "progressed"
EXITED = "exited"
FAILED = "failed"


def canonical_station(raw: Optional[str]) -> str:
    """Map any pipeline stage string (V4 'step1_trim', V2 'stage_1_transcribe',
    a publish status, …) onto one of the 7 canonical STATIONS. Single source of
    truth so the engine, the V4 instrumentation, and the admin view agree."""
    s = (raw or "").lower()
    if any(k in s for k in ("ingest", "download", "probe", "pool", "source")):
        return "ingest"
    if any(k in s for k in ("transcri", "asr", "deepgram", "whisper", "stt", "audio")):
        return "transcribe"
    if any(k in s for k in ("plan", "cut", "keep", "analy", "story", "stories", "claude", "gemini")):
        return "cut_plan"
    if any(k in s for k in ("trim", "concat", "step1")):
        return "trim"
    if any(k in s for k in ("brand", "watermark")):
        return "brand"
    if any(k in s for k in ("upload", "publish", "deliver")):
        return "upload"
    # canvas / compose / render / stitch / overlay / carousel / bulletin /
    # short / step2 / materialise / finalize → the heavy compose station.
    return "compose"

_RING_MAX = 5000  # ~2.5 MB worst case; negligible.

_lock = threading.Lock()
_ring: Deque[Dict[str, Any]] = deque(maxlen=_RING_MAX)
# Live occupancy: stage -> { job_key: card_dict }. A job_key is present while it
# sits AT that station (between ENTERED and EXITED/FAILED). This lets the admin
# view show "what is in the machine right now" — counts AND cards — straight
# from the factory's own bookkeeping, no DB query needed.
_occupancy: Dict[str, Dict[str, Dict[str, Any]]] = {s: {} for s in STATIONS}
# Per-lane occupancy for the EXTRA belts (rerender / seo), kept separate from
# the render belt above: { lane: { station: { key: card } } }.
_lane_occupancy: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
    lane: {s: {} for s in d["stations"]} for lane, d in LANE_DEFS.items()
}
# Rolling counters since process start (cheap KPIs for the header).
_counters: Dict[str, int] = {"emitted": 0, "completed": 0, "failed": 0}


@dataclass(frozen=True)
class Envelope:
    """ISOLATION INVARIANT I2 — the immutable identity that rides every stage.

    Frozen on purpose: a station can READ who owns the unit but can never
    re-stamp it onto a different user. ``tenant_id`` mirrors ``user_id`` in
    the current single-tenant-per-user model (user IS the tenant boundary),
    but is carried separately so a future tenant split needs no signature
    change.
    """

    tenant_id: Optional[int]
    user_id: Optional[int]
    job_id: Optional[int]          # render Job.id (the source-video job)
    clip_id: Optional[int]
    channel_id: Optional[int]
    upload_job_id: Optional[int] = None  # UploadJobV2.id (publish unit)
    label: str = ""               # short human label for the card, e.g. channel name

    def key(self) -> str:
        """Stable per-unit key for occupancy bookkeeping. Prefers the upload
        job id (one per channel) then the render job id."""
        if self.upload_job_id is not None:
            return f"u{self.upload_job_id}"
        base = f"j{self.job_id}" if self.job_id is not None else f"c{self.clip_id}"
        # Per-channel editor renders share one job/clip but differ by channel,
        # so suffix the channel to keep each its OWN unit on the conveyor.
        # No effect on existing lanes: publish units take the u{...} branch
        # above, and render/rerender/SEO emits carry channel_id=None.
        if self.channel_id is not None:
            return f"{base}.ch{self.channel_id}"
        return base

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "job_id": self.job_id,
            "clip_id": self.clip_id,
            "channel_id": self.channel_id,
            "upload_job_id": self.upload_job_id,
            "label": self.label,
        }


def emit(
    envelope: Envelope,
    stage: str,
    status: str,
    message: str = "",
) -> None:
    """Record one stage transition. NEVER raises — telemetry must not break work.

    ``ENTERED`` adds the unit to the station's live occupancy; ``EXITED`` /
    ``FAILED`` removes it (and, on the final ``upload`` station, bumps the
    completed/failed KPI counters).
    """
    try:
        now_wall = time.time()
        rec = {
            "ts": now_wall,
            "stage": stage,
            "status": status,
            "message": (message or "")[:240],
            **envelope.as_dict(),
        }
        key = envelope.key()
        uj = envelope.upload_job_id
        kind = "publish" if uj is not None else "render"
        with _lock:
            _ring.append(rec)
            _counters["emitted"] += 1
            occ = _occupancy.setdefault(stage, {})
            if status == ENTERED:
                occ[key] = {
                    "id": key,
                    "label": (envelope.label or "")[:40] or key,
                    "user_id": envelope.user_id,
                    "kind": kind,
                    "since": now_wall,
                }
            else:
                occ.pop(key, None)
                if status == FAILED:
                    _counters["failed"] += 1
                elif status == EXITED and stage == STATIONS[-1]:
                    _counters["completed"] += 1
    except Exception:
        # Telemetry is best-effort. Swallow everything.
        pass


# Lane-belt visibility tuning. Many SEO stations (score/per_channel/platform)
# fire ENTERED+EXITED back-to-back inside one synchronous request, so the
# 1.5s admin poll almost never lands while they're occupied → the belt looked
# "not connected". We keep an exited card visible for a short LINGER window so
# the blip is seen, and we evict an ENTERED card that NEVER exited (a ghost
# leaked by a ``raise`` between entered/exited) after STUCK_TTL.
_LANE_LINGER_S = 6.0
_LANE_STUCK_TTL_S = 90.0


def emit_lane(
    envelope: Envelope,
    lane: str,
    stage: str,
    status: str,
    message: str = "",
) -> None:
    """Record a transition on an EXTRA lane (``rerender`` / ``seo``). Mirrors
    ``emit`` but writes to that lane's own occupancy so the admin view shows it
    as a separate belt. Unknown lane/stage → no-op. NEVER raises."""
    try:
        if lane not in _lane_occupancy:
            return
        lane_occ = _lane_occupancy[lane]
        if stage not in lane_occ:
            return
        now_wall = time.time()
        rec = {
            "ts": now_wall, "lane": lane, "stage": stage, "status": status,
            "message": (message or "")[:240], **envelope.as_dict(),
        }
        key = envelope.key()
        with _lock:
            _ring.append(rec)
            _counters["emitted"] += 1
            occ = lane_occ[stage]
            if status == ENTERED:
                occ[key] = {
                    "id": key,
                    "label": (envelope.label or "")[:40] or key,
                    "user_id": envelope.user_id,
                    "kind": lane,
                    "since": now_wall,
                    "exited_at": None,
                }
            else:
                # Don't drop instantly — stamp exited_at so lane_belts can
                # LINGER the card briefly (sub-second SEO blips were invisible
                # to the admin poll). lane_belts prunes it after the linger.
                if key in occ:
                    occ[key]["exited_at"] = now_wall
    except Exception:
        pass


def emit_lane_here(lane: str, stage: str, status: str, message: str = "") -> None:
    """Emit on an extra lane using the thread-bound envelope. No-op if nothing
    bound. Used by the orchestrator's SEO submit/collect (already bound)."""
    env = getattr(_local, "env", None)
    if env is None:
        return
    emit_lane(env, lane, stage, status, message)


# ── Thread-local "current unit" binding ─────────────────────────────────
# The orchestrator binds the job's Envelope once at the top of a render run;
# deep helpers (trim/transcribe/cut-plan) then emit via emit_here() WITHOUT
# threading job_id through every signature. Each render job runs on its own
# worker thread, so the binding is per-job. If two jobs ever share a thread
# the worst case is a mislabelled telemetry card — never a delivery/isolation
# error (delivery ownership is enforced separately, fail-closed, in
# upload_dispatch).
_local = threading.local()


def bind(envelope: "Envelope") -> None:
    try:
        _local.env = envelope
    except Exception:
        pass


def unbind() -> None:
    try:
        _local.env = None
    except Exception:
        pass


def current() -> "Optional[Envelope]":
    return getattr(_local, "env", None)


def emit_here(stage: str, status: str, message: str = "") -> None:
    """Emit using the thread-bound envelope, mapping ``stage`` to a canonical
    station. No-op if nothing is bound. Never raises."""
    env = getattr(_local, "env", None)
    if env is None:
        return
    emit(env, canonical_station(stage), status, message)


def recent(limit: int = 200, job_id: Optional[int] = None,
           user_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return the most recent events (newest last), optionally filtered."""
    try:
        with _lock:
            items = list(_ring)
    except Exception:
        return []
    if job_id is not None:
        items = [e for e in items if e.get("job_id") == job_id]
    if user_id is not None:
        items = [e for e in items if e.get("user_id") == user_id]
    return items[-max(1, min(limit, _RING_MAX)):]


def occupancy_snapshot() -> Dict[str, int]:
    """Live count of units currently AT each station (factory bookkeeping)."""
    try:
        with _lock:
            return {s: len(_occupancy.get(s, {})) for s in STATIONS}
    except Exception:
        return {s: 0 for s in STATIONS}


def occupancy_cards(limit_per: int = 8) -> Dict[str, List[Dict[str, Any]]]:
    """Live unit cards currently AT each station (newest first), for the admin
    conveyor. ``since`` is rendered as ISO so the frontend can age it."""
    from datetime import datetime, timezone
    out: Dict[str, List[Dict[str, Any]]] = {s: [] for s in STATIONS}
    try:
        with _lock:
            for s in STATIONS:
                items = sorted(
                    _occupancy.get(s, {}).values(),
                    key=lambda c: c.get("since", 0), reverse=True)[:limit_per]
                cards = []
                for c in items:
                    ts = c.get("since")
                    try:
                        iso = datetime.fromtimestamp(
                            float(ts), timezone.utc).isoformat() if ts else None
                    except Exception:
                        iso = None
                    cards.append({
                        "id": c.get("id"), "label": c.get("label"),
                        "user_id": c.get("user_id"), "kind": c.get("kind"),
                        "since": iso,
                    })
                out[s] = cards
    except Exception:
        return {s: [] for s in STATIONS}
    return out


def lane_belts(limit_per: int = 8) -> List[Dict[str, Any]]:
    """Return the EXTRA belts (rerender / seo) fully shaped for the admin view:
    ``[{key, label, stations: [{key, label, count, cards:[...]}]}]``. Cards are
    newest-first with ISO ``since`` so the frontend can age them."""
    from datetime import datetime, timezone
    now_wall = time.time()
    out: List[Dict[str, Any]] = []
    try:
        with _lock:
            for lane, d in LANE_DEFS.items():
                stations = []
                lane_occ = _lane_occupancy.get(lane, {})
                for s in d["stations"]:
                    occ_s = lane_occ.get(s, {})
                    # Select VISIBLE cards + prune expired in place:
                    #  - ENTERED, not yet exited: visible until STUCK_TTL (then
                    #    evicted as a ghost leaked by a raise between emits).
                    #  - EXITED: visible for LINGER seconds so sub-second blips
                    #    are caught by the poll, then dropped.
                    visible = []
                    for k, c in list(occ_s.items()):
                        ex = c.get("exited_at")
                        if ex is None:
                            if now_wall - c.get("since", now_wall) > _LANE_STUCK_TTL_S:
                                occ_s.pop(k, None)
                                continue
                        else:
                            if now_wall - ex > _LANE_LINGER_S:
                                occ_s.pop(k, None)
                                continue
                        visible.append(c)
                    items = sorted(
                        visible, key=lambda c: c.get("since", 0), reverse=True)[:limit_per]
                    cards = []
                    for c in items:
                        ts = c.get("since")
                        try:
                            iso = datetime.fromtimestamp(
                                float(ts), timezone.utc).isoformat() if ts else None
                        except Exception:
                            iso = None
                        cards.append({"id": c.get("id"), "label": c.get("label"),
                                      "user_id": c.get("user_id"), "kind": c.get("kind"),
                                      "since": iso})
                    stations.append({
                        "key": s,
                        "label": d["labels"].get(s, s.title()),
                        "count": len(visible),
                        "cards": cards,
                    })
                out.append({"key": lane, "label": d["label"], "stations": stations})
    except Exception:
        return []
    return out


def counters_snapshot() -> Dict[str, int]:
    try:
        with _lock:
            return dict(_counters)
    except Exception:
        return {"emitted": 0, "completed": 0, "failed": 0}


def finish(envelope: "Optional[Envelope]", status: str = "exited") -> None:
    """Terminal cleanup for a unit: remove its key from EVERY station's
    occupancy (no ghost cards no matter where it died), bump the right
    counter, and append one terminal ring event. Called on job done/failed."""
    if envelope is None:
        return
    try:
        key = envelope.key()
        found = None
        now_wall = time.time()
        with _lock:
            for s in STATIONS:
                if key in _occupancy.get(s, {}):
                    found = s
                    _occupancy[s].pop(key, None)
            if status == FAILED or status == "failed":
                _counters["failed"] += 1
                st = FAILED
            else:
                _counters["completed"] += 1
                st = EXITED
            _ring.append({
                "ts": now_wall, "stage": found or STATIONS[-1], "status": st,
                "message": "", **envelope.as_dict(),
            })
    except Exception:
        pass


def reset() -> None:
    """Test helper — clear all state."""
    with _lock:
        _ring.clear()
        for s in _occupancy:
            _occupancy[s].clear()
        for lane in _lane_occupancy:
            for s in _lane_occupancy[lane]:
                _lane_occupancy[lane][s].clear()
        for k in _counters:
            _counters[k] = 0


__all__ = [
    "STATIONS",
    "STATION_LABELS",
    "LANE_DEFS",
    "ENTERED",
    "PROGRESSED",
    "EXITED",
    "FAILED",
    "canonical_station",
    "Envelope",
    "emit",
    "emit_lane",
    "emit_lane_here",
    "lane_belts",
    "bind",
    "unbind",
    "current",
    "emit_here",
    "recent",
    "occupancy_snapshot",
    "occupancy_cards",
    "counters_snapshot",
    "finish",
    "reset",
]
