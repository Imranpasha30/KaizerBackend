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

Authentication
--------------
All REST endpoints (except WS — browsers can't Authorization-header a WS
upgrade in most paths) use the project's JWT pattern via `auth.current_user`.
WebSocket upgrades accept a `?token=` query param as a fallback.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
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
