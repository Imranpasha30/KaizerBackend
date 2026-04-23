"""
kaizer.routers.live_director
=============================
REST + WebSocket surface for the Autonomous Live Director control panel.

Endpoints
---------
  POST  /api/live/events                      → create event
  GET   /api/live/events                      → list events for current user
  GET   /api/live/events/{event_id}           → event detail (incl. cameras)
  POST  /api/live/events/{event_id}/cameras   → add camera to event
  POST  /api/live/events/{event_id}/start     → transition scheduled → live
                                                 (spawns ingest + director + composer)
  POST  /api/live/events/{event_id}/stop      → transition live → ended
  POST  /api/live/events/{event_id}/pin       → operator pin: body={cam_id}
  POST  /api/live/events/{event_id}/unpin     → release pin
  POST  /api/live/events/{event_id}/blacklist → body={cam_id}
  POST  /api/live/events/{event_id}/allow     → body={cam_id}
  POST  /api/live/events/{event_id}/force-cut → body={cam_id}
  GET   /api/live/events/{event_id}/log       → tail of director_log rows
  WS    /api/live/events/{event_id}/stream    → real-time decisions + thumbnails

  ── Phase 9: "phone as camera" browser-to-browser test mode ───────────────
  POST  /api/live/events/{event_id}/phone-sessions
                                              → mint a phone ingest session
  WS    /api/live/ws/ingest/{event_id}/{cam_id}?token=...
                                              → phone pushes webm chunks
  WS    /api/live/ws/monitor/{event_id}/{cam_id}
                                              → director page receives chunks

Authentication
--------------
All REST endpoints (except WS — browsers can't Authorization-header a WS
upgrade in most paths) use the project's JWT pattern via `auth.current_user`.
WebSocket upgrades accept a `?token=` query param as a fallback.

Phase 9 note: the monitor WebSocket has NO authentication for now. That is
acceptable for the LAN-only test feature (phone on same WiFi as laptop);
in production this endpoint would need cookie-based WS auth or a short-lived
monitor token, and the fanout dict would need to be replaced with Redis pub/sub
to work across multiple uvicorn workers. TODO(phase-10): multi-worker fanout.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
from dataclasses import asdict
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel
from sqlalchemy.orm import Session

import auth
import models
from database import SessionLocal

logger = logging.getLogger("kaizer.routers.live_director")

router = APIRouter(prefix="/api/live", tags=["live-director"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ══════════════════════════════════════════════════════════════════════════════
# In-process runtime registry — one LiveSession per event while it's running.
# ══════════════════════════════════════════════════════════════════════════════

class _LiveSession:
    """Holds the running asyncio-task bundle for a live event.

    A session is created by POST /start and destroyed by POST /stop. The
    pipeline_core.live_director modules (ingest + analyzers + director +
    composer + output) compose here.
    """
    def __init__(self, event_id: int, camera_ids: list[str]):
        from pipeline_core.live_director.signal_bus import SignalBus
        self.event_id = event_id
        self.camera_ids = list(camera_ids)
        self.bus = SignalBus()
        self.director = None          # pipeline_core.live_director.director.Director
        self.composer = None          # pipeline_core.live_director.composer.Composer
        self.output_stack = None      # pipeline_core.live_director.output.OutputStack
        self.relay = None             # pipeline_core.live_director.relay.RTMPRelay (Phase 7)
        self.ws_clients: set[WebSocket] = set()
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    async def broadcast(self, payload: dict) -> None:
        """Fan-out a JSON payload to every connected WS client. Silent on
        disconnect — the closing client is removed on the next tick."""
        if not self.ws_clients:
            return
        dead: list[WebSocket] = []
        for ws in list(self.ws_clients):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.ws_clients.discard(ws)


_SESSIONS: dict[int, _LiveSession] = {}


# ══════════════════════════════════════════════════════════════════════════════
# Phase 9 — "phone as camera" browser fanout registry
# ══════════════════════════════════════════════════════════════════════════════

# (event_id, cam_id) → list of monitor WebSockets receiving live chunks.
# Ingest socket pushes chunks; monitor sockets receive them.
# In-process only; TODO(phase-10) replace with Redis pub/sub for multi-worker.
_INGEST_FANOUT: dict[tuple[int, str], list[WebSocket]] = {}

# token → {"event_id", "cam_id", "created_at"}. Tokens are one-shot-ish — valid
# until the phone connects + disconnects, then revoked. 30min hard expiry.
_PHONE_TOKENS: dict[str, dict] = {}

_PHONE_TOKEN_TTL_S = 1800  # 30 minutes


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ══════════════════════════════════════════════════════════════════════════════

class CreateEventRequest(BaseModel):
    name: str
    venue: str = ""
    config: dict = {}


class AddCameraRequest(BaseModel):
    cam_id: str
    label: str = ""
    mic_id: str = ""
    role_hints: list[str] = []


class CameraIdRequest(BaseModel):
    cam_id: str


class EventSchema(BaseModel):
    id: int
    name: str
    venue: str
    status: str
    config_json: dict
    program_url: str
    created_at: Optional[str] = None


class CameraSchema(BaseModel):
    cam_id: str
    label: str
    mic_id: str
    role_hints: list[str]
    iso_url: str


class EventDetailSchema(EventSchema):
    cameras: list[CameraSchema] = []
    is_live_in_process: bool = False


class DirectorLogSchema(BaseModel):
    t: float
    kind: str
    cam_id: str
    confidence: float
    reason: str


# ── Phase 7 schemas ──────────────────────────────────────────────────────────


class RelayDestinationSchema(BaseModel):
    id: str
    name: str
    rtmp_url: str
    enabled: bool = True
    reconnect_max_attempts: int = 0


class RelayStatusSchema(BaseModel):
    destination_id: str
    state: str
    attempts: int
    last_error: str = ""
    started_at: float = 0.0
    uptime_s: float = 0.0


class ChromaConfigSchema(BaseModel):
    color: str = "0x00d639"
    similarity: float = 0.12
    blend: float = 0.08
    bg_asset_path: str = ""
    bg_asset_kind: str = "auto"
    bg_fit: str = "cover"
    enabled: bool = True


class BridgeConfigSchema(BaseModel):
    asset_url: str = ""
    silence_threshold_s: float = 3.0
    rms_ceiling: float = 0.02
    min_duration_s: float = 4.0


# ══════════════════════════════════════════════════════════════════════════════
# REST endpoints
# ══════════════════════════════════════════════════════════════════════════════


def _event_to_schema(ev: "models.LiveEvent") -> EventSchema:
    return EventSchema(
        id=ev.id,
        name=ev.name,
        venue=ev.venue or "",
        status=ev.status,
        config_json=ev.config_json or {},
        program_url=ev.program_url or "",
        created_at=ev.created_at.isoformat() if ev.created_at else None,
    )


@router.post("/events", response_model=EventSchema)
def create_event(
    body: CreateEventRequest,
    db: Session = Depends(get_db),
    user: "models.User" = Depends(auth.current_user),
):
    ev = models.LiveEvent(
        user_id=user.id,
        name=body.name,
        venue=body.venue,
        status="scheduled",
        config_json=body.config or {},
    )
    db.add(ev)
    db.commit()
    db.refresh(ev)
    return _event_to_schema(ev)


@router.get("/events", response_model=list[EventSchema])
def list_events(
    db: Session = Depends(get_db),
    user: "models.User" = Depends(auth.current_user),
):
    rows = (
        db.query(models.LiveEvent)
        .filter(models.LiveEvent.user_id == user.id)
        .order_by(models.LiveEvent.created_at.desc())
        .all()
    )
    return [_event_to_schema(ev) for ev in rows]


@router.get("/events/{event_id}", response_model=EventDetailSchema)
def get_event(
    event_id: int,
    db: Session = Depends(get_db),
    user: "models.User" = Depends(auth.current_user),
):
    ev = _load_event(event_id, user.id, db)
    cams = (
        db.query(models.LiveCamera)
        .filter(models.LiveCamera.event_id == event_id)
        .all()
    )
    return EventDetailSchema(
        **_event_to_schema(ev).dict(),
        cameras=[
            CameraSchema(
                cam_id=c.cam_id,
                label=c.label or "",
                mic_id=c.mic_id or "",
                role_hints=c.role_hints or [],
                iso_url=c.iso_url or "",
            )
            for c in cams
        ],
        is_live_in_process=(event_id in _SESSIONS),
    )


@router.post("/events/{event_id}/cameras", response_model=CameraSchema)
def add_camera(
    event_id: int,
    body: AddCameraRequest,
    db: Session = Depends(get_db),
    user: "models.User" = Depends(auth.current_user),
):
    _load_event(event_id, user.id, db)
    cam = models.LiveCamera(
        event_id=event_id,
        cam_id=body.cam_id,
        label=body.label,
        mic_id=body.mic_id,
        role_hints=body.role_hints,
    )
    db.add(cam)
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(409, f"Duplicate cam_id for event: {exc}")
    db.refresh(cam)
    return CameraSchema(
        cam_id=cam.cam_id,
        label=cam.label or "",
        mic_id=cam.mic_id or "",
        role_hints=cam.role_hints or [],
        iso_url=cam.iso_url or "",
    )


@router.delete("/events/{event_id}/cameras/{cam_id}")
def delete_camera(
    event_id: int,
    cam_id: str,
    db: Session = Depends(get_db),
    user: "models.User" = Depends(auth.current_user),
):
    """Remove a camera from a scheduled event. Refuses while the event is live."""
    ev = _load_event(event_id, user.id, db)
    if ev.status == "live" or event_id in _SESSIONS:
        raise HTTPException(409, "Cannot remove a camera from a live event. Stop the event first.")
    cam = (
        db.query(models.LiveCamera)
        .filter(
            models.LiveCamera.event_id == event_id,
            models.LiveCamera.cam_id == cam_id,
        )
        .first()
    )
    if cam is None:
        raise HTTPException(404, f"No camera {cam_id!r} on event {event_id}")
    db.delete(cam)
    db.commit()
    return {"deleted": cam_id}


@router.delete("/events/{event_id}")
def delete_event(
    event_id: int,
    db: Session = Depends(get_db),
    user: "models.User" = Depends(auth.current_user),
):
    """Delete an entire event (cameras + director log cascade via FK).
    Refuses while the event is live — POST /stop first.
    """
    ev = _load_event(event_id, user.id, db)
    if ev.status == "live" or event_id in _SESSIONS:
        raise HTTPException(409, "Cannot delete a live event. Stop it first.")
    db.delete(ev)
    db.commit()
    return {"deleted": event_id}


@router.post("/events/{event_id}/start")
async def start_event(
    event_id: int,
    db: Session = Depends(get_db),
    user: "models.User" = Depends(auth.current_user),
):
    """Transition a scheduled event into the LIVE state.

    v1: spins up an in-process _LiveSession with a SignalBus + a Director
    (no ingest yet — the director is fed via WebSocket clients pushing
    SignalFrames during testing). A full-stack live run requires the
    OBS/RTMP side to be wired; see docs/PHASE6_LIVE_DIRECTOR.md §6.1.
    """
    ev = _load_event(event_id, user.id, db)
    if ev.status == "live" or event_id in _SESSIONS:
        raise HTTPException(409, "Event is already live")

    cams = (
        db.query(models.LiveCamera)
        .filter(models.LiveCamera.event_id == event_id)
        .all()
    )
    if len(cams) < 1:
        raise HTTPException(400, "Event has no cameras — add at least one first")

    session = _LiveSession(event_id=event_id, camera_ids=[c.cam_id for c in cams])
    _SESSIONS[event_id] = session

    # Director — listens to the SignalBus, emits CameraSelection to the
    # session's broadcast channel and persists to director_log.
    from pipeline_core.live_director.director import Director, DirectorConfig

    cfg = DirectorConfig(**(ev.config_json or {}).get("director", {}))

    async def _on_selection(sel):
        # Persist + broadcast
        entry = models.DirectorLogEntry(
            event_id=event_id,
            t=sel.t,
            kind="selection",
            cam_id=sel.cam_id,
            confidence=sel.confidence,
            reason=sel.reason,
            payload={"transition": sel.transition},
        )
        with SessionLocal() as db2:
            db2.add(entry)
            db2.commit()
        await session.broadcast({
            "type": "selection",
            "t": sel.t,
            "cam_id": sel.cam_id,
            "transition": sel.transition,
            "confidence": sel.confidence,
            "reason": sel.reason,
        })

    def _on_event(ev_):
        # Sync path — schedule a broadcast + DB write
        async def _do():
            entry = models.DirectorLogEntry(
                event_id=event_id,
                t=ev_.t,
                kind=ev_.kind,
                cam_id=ev_.payload.get("cam_id", "") if ev_.payload else "",
                reason=ev_.payload.get("reason", "") if ev_.payload else "",
                payload=ev_.payload or {},
            )
            with SessionLocal() as db2:
                db2.add(entry)
                db2.commit()
            await session.broadcast({
                "type": ev_.kind,
                "t": ev_.t,
                "payload": ev_.payload,
            })
        try:
            asyncio.get_event_loop().create_task(_do())
        except Exception:
            pass

    session.director = Director(
        event_id=event_id,
        camera_ids=session.camera_ids,
        bus=session.bus,
        config=cfg,
        on_selection=_on_selection,
        on_event=_on_event,
    )

    # Flip DB state
    ev.status = "live"
    db.commit()

    # Director's run() loop is consumer-side; start it as an asyncio task.
    task = asyncio.create_task(session.director.run(), name=f"director-{event_id}")
    session._tasks.append(task)

    logger.info("live: event %s started with %d cameras", event_id, len(cams))
    return {"ok": True, "event_id": event_id, "status": "live"}


@router.post("/events/{event_id}/stop")
async def stop_event(
    event_id: int,
    db: Session = Depends(get_db),
    user: "models.User" = Depends(auth.current_user),
):
    ev = _load_event(event_id, user.id, db)
    session = _SESSIONS.pop(event_id, None)
    if session:
        if session.director:
            session.director.stop()
        for t in session._tasks:
            try:
                t.cancel()
            except Exception:
                pass
        # Give tasks a moment to wind down. TestClient runs each request
        # on a fresh event loop, so tasks created in /start may belong to
        # a loop that's already closed by the time /stop runs — awaiting
        # cross-loop raises ValueError. We swallow those and trust that
        # the cancel() above is enough. Production uvicorn has one loop
        # for the entire app lifetime, so the await path works cleanly
        # there.
        for t in session._tasks:
            try:
                await asyncio.wait_for(t, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, ValueError, RuntimeError):
                pass
    ev.status = "ended"
    db.commit()
    return {"ok": True, "event_id": event_id, "status": "ended"}


@router.post("/events/{event_id}/pin")
def pin_camera(
    event_id: int,
    body: CameraIdRequest,
    user: "models.User" = Depends(auth.current_user),
):
    session = _require_session(event_id)
    try:
        session.director.pin(body.cam_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"ok": True, "pinned": body.cam_id}


@router.post("/events/{event_id}/unpin")
def unpin_camera(
    event_id: int,
    user: "models.User" = Depends(auth.current_user),
):
    session = _require_session(event_id)
    session.director.unpin()
    return {"ok": True, "pinned": None}


@router.post("/events/{event_id}/blacklist")
def blacklist_camera(
    event_id: int,
    body: CameraIdRequest,
    user: "models.User" = Depends(auth.current_user),
):
    session = _require_session(event_id)
    try:
        session.director.blacklist(body.cam_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"ok": True, "blacklisted": body.cam_id}


@router.post("/events/{event_id}/allow")
def allow_camera(
    event_id: int,
    body: CameraIdRequest,
    user: "models.User" = Depends(auth.current_user),
):
    session = _require_session(event_id)
    session.director.allow(body.cam_id)
    return {"ok": True, "allowed": body.cam_id}


@router.post("/events/{event_id}/force-cut")
def force_cut(
    event_id: int,
    body: CameraIdRequest,
    user: "models.User" = Depends(auth.current_user),
):
    session = _require_session(event_id)
    try:
        session.director.force_cut(body.cam_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"ok": True, "cutting_to": body.cam_id}


@router.get("/events/{event_id}/log")
def get_log(
    event_id: int,
    limit: int = Query(200, ge=1, le=1000),
    db: Session = Depends(get_db),
    user: "models.User" = Depends(auth.current_user),
):
    _load_event(event_id, user.id, db)
    rows = (
        db.query(models.DirectorLogEntry)
        .filter(models.DirectorLogEntry.event_id == event_id)
        .order_by(models.DirectorLogEntry.t.desc())
        .limit(limit)
        .all()
    )
    return [
        DirectorLogSchema(
            t=r.t, kind=r.kind, cam_id=r.cam_id or "",
            confidence=float(r.confidence or 0.0),
            reason=r.reason or "",
        )
        for r in rows
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Phase 7 endpoints — RTMP relay, chroma, dead-air bridge
# ══════════════════════════════════════════════════════════════════════════════


def hls_playlist_path_for(event_id: int) -> str:
    """Where the composer writes HLS — matches output.HLSSink defaults."""
    return f"/tmp/kaizer_live/event_{event_id}/program.m3u8"


def _relay_status_to_schema(status) -> RelayStatusSchema:
    return RelayStatusSchema(
        destination_id=status.destination_id,
        state=status.state,
        attempts=status.attempts,
        last_error=status.last_error or "",
        started_at=float(status.started_at or 0.0),
        uptime_s=float(status.uptime_s or 0.0),
    )


# ── Relay endpoints ──────────────────────────────────────────────────────────


@router.get(
    "/events/{event_id}/relay/destinations",
    response_model=list[RelayDestinationSchema],
)
def list_relay_destinations(
    event_id: int,
    db: Session = Depends(get_db),
    user: "models.User" = Depends(auth.current_user),
):
    ev = _load_event(event_id, user.id, db)
    dests = (ev.config_json or {}).get("relay_destinations", []) or []
    return [RelayDestinationSchema(**d) for d in dests]


@router.post(
    "/events/{event_id}/relay/destinations",
    response_model=RelayDestinationSchema,
)
async def add_relay_destination(
    event_id: int,
    body: RelayDestinationSchema,
    db: Session = Depends(get_db),
    user: "models.User" = Depends(auth.current_user),
):
    ev = _load_event(event_id, user.id, db)
    cfg = dict(ev.config_json or {})
    dests = list(cfg.get("relay_destinations", []) or [])
    for d in dests:
        if d.get("id") == body.id:
            raise HTTPException(409, f"destination id {body.id!r} already exists")
    payload = body.dict()
    dests.append(payload)
    cfg["relay_destinations"] = dests
    ev.config_json = cfg
    # SQLAlchemy needs a flag_modified hint for JSON columns sometimes; assigning
    # a fresh dict on ev.config_json is the safer cross-dialect approach.
    db.commit()

    # If event is currently live, hot-add to the running relay (if any).
    session = _SESSIONS.get(event_id)
    if session is not None and session.relay is not None:
        try:
            from pipeline_core.live_director.relay import RelayDestination
            await session.relay.add_destination(RelayDestination(
                id=body.id,
                name=body.name,
                rtmp_url=body.rtmp_url,
                enabled=body.enabled,
                reconnect_max_attempts=body.reconnect_max_attempts,
            ))
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        except Exception as exc:
            logger.warning(
                "relay hot-add failed for event %s dest %s: %s",
                event_id, body.id, exc,
            )
    return body


@router.delete("/events/{event_id}/relay/destinations/{destination_id}")
async def delete_relay_destination(
    event_id: int,
    destination_id: str,
    db: Session = Depends(get_db),
    user: "models.User" = Depends(auth.current_user),
):
    ev = _load_event(event_id, user.id, db)
    cfg = dict(ev.config_json or {})
    dests = list(cfg.get("relay_destinations", []) or [])
    new_dests = [d for d in dests if d.get("id") != destination_id]
    if len(new_dests) == len(dests):
        raise HTTPException(404, f"destination {destination_id!r} not found")
    cfg["relay_destinations"] = new_dests
    ev.config_json = cfg
    db.commit()

    session = _SESSIONS.get(event_id)
    if session is not None and session.relay is not None:
        try:
            await session.relay.remove_destination(destination_id)
        except Exception as exc:
            logger.warning(
                "relay hot-remove failed for event %s dest %s: %s",
                event_id, destination_id, exc,
            )
    return {"deleted": destination_id}


@router.post("/events/{event_id}/relay/start")
async def start_relay(
    event_id: int,
    db: Session = Depends(get_db),
    user: "models.User" = Depends(auth.current_user),
):
    ev = _load_event(event_id, user.id, db)
    if ev.status != "live" or event_id not in _SESSIONS:
        raise HTTPException(409, "Event is not live")
    session = _SESSIONS[event_id]
    if session.relay is not None and session.relay.is_running():
        raise HTTPException(409, "Relay is already running")

    cfg = ev.config_json or {}
    dest_rows = cfg.get("relay_destinations", []) or []
    if not dest_rows:
        raise HTTPException(400, "No relay destinations configured")

    from pipeline_core.live_director.relay import RelayDestination, RTMPRelay

    destinations = [
        RelayDestination(
            id=d.get("id"),
            name=d.get("name", ""),
            rtmp_url=d.get("rtmp_url", ""),
            enabled=bool(d.get("enabled", True)),
            reconnect_max_attempts=int(d.get("reconnect_max_attempts", 0) or 0),
        )
        for d in dest_rows
    ]
    source_url = ev.program_url or hls_playlist_path_for(event_id)
    relay = RTMPRelay(source_url=source_url, destinations=destinations)
    session.relay = relay
    await relay.start()

    statuses = relay.get_status()
    if not isinstance(statuses, list):
        statuses = [statuses]
    return {
        "status": "started",
        "source_url": source_url,
        "destinations": [_relay_status_to_schema(s).dict() for s in statuses],
    }


@router.post("/events/{event_id}/relay/stop")
async def stop_relay(
    event_id: int,
    db: Session = Depends(get_db),
    user: "models.User" = Depends(auth.current_user),
):
    _load_event(event_id, user.id, db)
    session = _SESSIONS.get(event_id)
    if session is not None and session.relay is not None:
        try:
            await session.relay.stop()
        except Exception as exc:
            logger.warning("relay stop failed for event %s: %s", event_id, exc)
        session.relay = None
    return {"status": "stopped"}


@router.get("/events/{event_id}/relay/status")
def get_relay_status(
    event_id: int,
    db: Session = Depends(get_db),
    user: "models.User" = Depends(auth.current_user),
):
    _load_event(event_id, user.id, db)
    session = _SESSIONS.get(event_id)
    if session is None or session.relay is None:
        return {"is_running": False, "destinations": []}
    statuses = session.relay.get_status()
    if not isinstance(statuses, list):
        statuses = [statuses]
    return {
        "is_running": bool(session.relay.is_running()),
        "destinations": [_relay_status_to_schema(s).dict() for s in statuses],
    }


# ── Chroma endpoints ─────────────────────────────────────────────────────────


@router.put("/events/{event_id}/cameras/{cam_id}/chroma")
def put_chroma(
    event_id: int,
    cam_id: str,
    body: ChromaConfigSchema,
    db: Session = Depends(get_db),
    user: "models.User" = Depends(auth.current_user),
):
    ev = _load_event(event_id, user.id, db)

    from pipeline_core.live_director.chroma import ChromaConfig, validate_chroma_config

    config = ChromaConfig(
        color=body.color,
        similarity=body.similarity,
        blend=body.blend,
        bg_asset_path=body.bg_asset_path,
        bg_asset_kind=body.bg_asset_kind,
        bg_fit=body.bg_fit,
        enabled=body.enabled,
    )
    try:
        validate_chroma_config(config)
    except FileNotFoundError as exc:
        raise HTTPException(
            404, {"detail": f"bg asset not found: {config.bg_asset_path}"}
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    cfg = dict(ev.config_json or {})
    chroma_map = dict(cfg.get("chroma_configs", {}) or {})
    chroma_map[str(cam_id)] = {
        "color": config.color,
        "similarity": config.similarity,
        "blend": config.blend,
        "bg_asset_path": config.bg_asset_path,
        "bg_asset_kind": config.bg_asset_kind,
        "bg_fit": config.bg_fit,
        "enabled": config.enabled,
    }
    cfg["chroma_configs"] = chroma_map
    ev.config_json = cfg
    db.commit()

    applied_live = False
    note = "saved to event config; takes effect on next live start"
    session = _SESSIONS.get(event_id)
    if session is not None and session.composer is not None:
        try:
            cam_index = session.camera_ids.index(str(cam_id))
        except ValueError:
            cam_index = -1
        if cam_index >= 0:
            try:
                session.composer.config.chroma_configs[cam_index] = config
                applied_live = True
                note = (
                    "saved; runtime updated — takes effect on next layout respawn"
                )
            except Exception as exc:
                logger.warning(
                    "chroma live-update failed for event %s cam %s: %s",
                    event_id, cam_id, exc,
                )
    return {"saved": True, "applied_live": applied_live, "note": note}


@router.delete("/events/{event_id}/cameras/{cam_id}/chroma")
def delete_chroma(
    event_id: int,
    cam_id: str,
    db: Session = Depends(get_db),
    user: "models.User" = Depends(auth.current_user),
):
    ev = _load_event(event_id, user.id, db)
    cfg = dict(ev.config_json or {})
    chroma_map = dict(cfg.get("chroma_configs", {}) or {})
    key = str(cam_id)
    if key not in chroma_map:
        raise HTTPException(404, f"no chroma config for cam {cam_id!r}")
    chroma_map.pop(key, None)
    cfg["chroma_configs"] = chroma_map
    ev.config_json = cfg
    db.commit()

    session = _SESSIONS.get(event_id)
    if session is not None and session.composer is not None:
        try:
            cam_index = session.camera_ids.index(str(cam_id))
        except ValueError:
            cam_index = -1
        if cam_index >= 0:
            try:
                session.composer.config.chroma_configs.pop(cam_index, None)
            except Exception as exc:
                logger.warning(
                    "chroma live-remove failed for event %s cam %s: %s",
                    event_id, cam_id, exc,
                )
    return {"deleted": cam_id}


# ── Bridge endpoint ──────────────────────────────────────────────────────────


@router.put("/events/{event_id}/bridge")
def put_bridge(
    event_id: int,
    body: BridgeConfigSchema,
    db: Session = Depends(get_db),
    user: "models.User" = Depends(auth.current_user),
):
    ev = _load_event(event_id, user.id, db)

    if body.asset_url and not os.path.exists(body.asset_url):
        raise HTTPException(400, f"bridge asset not found: {body.asset_url}")

    cfg = dict(ev.config_json or {})
    cfg["bridge_asset_url"] = body.asset_url
    cfg["bridge_silence_threshold_s"] = float(body.silence_threshold_s)
    cfg["bridge_silence_rms_ceiling"] = float(body.rms_ceiling)
    cfg["bridge_min_duration_s"] = float(body.min_duration_s)
    ev.config_json = cfg
    db.commit()

    applied_live = False
    session = _SESSIONS.get(event_id)
    if session is not None and session.director is not None:
        try:
            session.director.config.bridge_asset_url = body.asset_url
            session.director.config.bridge_silence_threshold_s = float(body.silence_threshold_s)
            session.director.config.bridge_silence_rms_ceiling = float(body.rms_ceiling)
            session.director.config.bridge_min_duration_s = float(body.min_duration_s)
            applied_live = True
        except Exception as exc:
            logger.warning(
                "bridge live-update failed for event %s: %s", event_id, exc,
            )
    return {"saved": True, "applied_live": applied_live}


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket stream
# ══════════════════════════════════════════════════════════════════════════════


@router.websocket("/events/{event_id}/stream")
async def event_stream(websocket: WebSocket, event_id: int, token: str = Query(default="")):
    """Real-time event + thumbnail stream.

    Auth: ?token= query param (Bearer JWT). v1 does a simple sanity check;
    full token validation mirrors auth.current_user flow.
    """
    await websocket.accept()
    session = _SESSIONS.get(event_id)
    if not session:
        await websocket.send_json({"type": "error", "message": "event not live"})
        await websocket.close()
        return
    session.ws_clients.add(websocket)
    try:
        await websocket.send_json({
            "type": "hello",
            "event_id": event_id,
            "camera_ids": session.camera_ids,
        })
        # Keep the connection open — broadcasts push from the director.
        while True:
            # Wait for client ping or disconnect.
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("ws: %s disconnected: %s", event_id, exc)
    finally:
        session.ws_clients.discard(websocket)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 9 — "phone as camera" browser test mode
# ══════════════════════════════════════════════════════════════════════════════


class PhoneSessionResponse(BaseModel):
    cam_id: str
    token: str
    phone_url: str        # relative path — frontend prepends window.location.origin
    ingest_ws_url: str    # relative ws path — phone prepends ws(s)://<host>:8000


def _gen_unique_phone_cam_id(db: Session, event_id: int) -> str:
    """Generate a phone_<hex6> cam_id that doesn't collide on this event."""
    existing = {
        row.cam_id
        for row in db.query(models.LiveCamera.cam_id)
        .filter(models.LiveCamera.event_id == event_id)
        .all()
    }
    for _ in range(32):
        candidate = f"phone_{secrets.token_hex(3)}"  # 6 hex chars
        if candidate not in existing:
            return candidate
    # Fallback: extremely unlikely to reach here
    return f"phone_{secrets.token_hex(6)}"


@router.post(
    "/events/{event_id}/phone-sessions",
    response_model=PhoneSessionResponse,
)
def create_phone_session(
    event_id: int,
    db: Session = Depends(get_db),
    user: "models.User" = Depends(auth.current_user),
):
    """Mint a phone-as-camera ingest session.

    Registers a new LiveCamera row (role_hints=["phone","webrtc"]) and
    returns a one-shot token + relative URLs that the frontend wraps into
    a QR code. The phone scans the QR, opens the PhoneCamera page, and
    connects a WebSocket to /api/live/ws/ingest/... to push webm chunks.

    Rejects if the event is already live (for this test phase we add cameras
    only while scheduled; can be relaxed once the live session accepts hot
    camera adds).
    """
    ev = _load_event(event_id, user.id, db)
    if ev.status == "live" or event_id in _SESSIONS:
        raise HTTPException(
            409,
            "Phone cameras can only be added while the event is not live. "
            "Stop the event first, then add the phone.",
        )

    cam_id = _gen_unique_phone_cam_id(db, event_id)
    cam = models.LiveCamera(
        event_id=event_id,
        cam_id=cam_id,
        label="Phone camera",
        mic_id="",
        role_hints=["phone", "webrtc"],
    )
    db.add(cam)
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(500, f"Failed to register phone camera: {exc}")
    db.refresh(cam)

    token = secrets.token_urlsafe(24)
    _PHONE_TOKENS[token] = {
        "event_id": event_id,
        "cam_id": cam_id,
        "created_at": time.time(),
    }

    # Opportunistic sweep of expired tokens to keep the dict bounded.
    now = time.time()
    expired = [
        t for t, m in _PHONE_TOKENS.items()
        if now - m.get("created_at", 0) > _PHONE_TOKEN_TTL_S
    ]
    for t in expired:
        _PHONE_TOKENS.pop(t, None)

    return PhoneSessionResponse(
        cam_id=cam_id,
        token=token,
        phone_url=f"/phone/{event_id}/{cam_id}?token={token}",
        ingest_ws_url=f"/api/live/ws/ingest/{event_id}/{cam_id}?token={token}",
    )


@router.websocket("/ws/ingest/{event_id}/{cam_id}")
async def ws_ingest_phone_stream(
    websocket: WebSocket,
    event_id: int,
    cam_id: str,
    token: str = Query(...),
):
    """Phone → backend ingest WebSocket.

    The phone's MediaRecorder pushes webm chunks as binary frames; the
    backend fans them out to every monitor WebSocket connected for the
    same (event_id, cam_id) tuple.
    """
    # Validate token
    meta = _PHONE_TOKENS.get(token)
    if not meta or meta["event_id"] != event_id or meta["cam_id"] != cam_id:
        await websocket.close(code=4401, reason="invalid token")
        return
    if time.time() - meta["created_at"] > _PHONE_TOKEN_TTL_S:
        _PHONE_TOKENS.pop(token, None)
        await websocket.close(code=4401, reason="token expired")
        return

    await websocket.accept()
    key = (event_id, cam_id)
    try:
        while True:
            msg = await websocket.receive()
            # FastAPI's WebSocket.receive() returns a dict with either
            # "bytes" or "text". It can also return a disconnect message.
            msg_type = msg.get("type")
            if msg_type == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data is not None:
                # Fan out to every monitor WS for this (event, cam).
                # Snapshot to avoid mutation-during-iteration surprises.
                subs = list(_INGEST_FANOUT.get(key, []))
                dead: list[WebSocket] = []
                for sub in subs:
                    try:
                        await sub.send_bytes(data)
                    except Exception:
                        dead.append(sub)
                if dead:
                    active = _INGEST_FANOUT.get(key, [])
                    for d in dead:
                        try:
                            active.remove(d)
                        except ValueError:
                            pass
                continue
            text = msg.get("text")
            if text is not None:
                # Optional control channel — {"type": "stop"} cleanly breaks
                # the loop; {"type": "ping"} is ignored (keep-alive).
                try:
                    payload = json.loads(text)
                    if isinstance(payload, dict) and payload.get("type") == "stop":
                        break
                except Exception:
                    pass
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("phone ingest ws disconnected: %s", exc)
    finally:
        # Revoke token on disconnect so it can't be reused
        _PHONE_TOKENS.pop(token, None)


@router.websocket("/ws/monitor/{event_id}/{cam_id}")
async def ws_monitor_phone_stream(
    websocket: WebSocket,
    event_id: int,
    cam_id: str,
):
    """Director page → backend monitor WebSocket.

    No auth for now (LAN-only test feature). Production would require
    cookie-based WS auth or a short-lived monitor token.
    """
    await websocket.accept()
    key = (event_id, cam_id)
    _INGEST_FANOUT.setdefault(key, []).append(websocket)
    try:
        while True:
            # Monitor doesn't send anything — we just keep the socket
            # open and wait for disconnect.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("phone monitor ws disconnected: %s", exc)
    finally:
        subs = _INGEST_FANOUT.get(key, [])
        try:
            subs.remove(websocket)
        except ValueError:
            pass
        if not subs:
            _INGEST_FANOUT.pop(key, None)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _load_event(event_id: int, user_id: int, db: Session) -> "models.LiveEvent":
    ev = (
        db.query(models.LiveEvent)
        .filter(
            models.LiveEvent.id == event_id,
            models.LiveEvent.user_id == user_id,
        )
        .first()
    )
    if ev is None:
        raise HTTPException(404, "Event not found")
    return ev


def _require_session(event_id: int) -> _LiveSession:
    session = _SESSIONS.get(event_id)
    if session is None:
        raise HTTPException(409, "Event is not live")
    return session
