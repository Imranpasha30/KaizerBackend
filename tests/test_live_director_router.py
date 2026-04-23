"""Phase 6.6 backend — live-director router tests.

Uses an in-memory SQLite DB + dependency override, same pattern as
test_feedback_api.py. Auth is stubbed via dependency_overrides.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import auth
import models
from database import Base


@pytest.fixture
def app_and_session(monkeypatch):
    engine = create_engine(
        "sqlite:///file:kaizer_live_router_test?mode=memory&cache=shared&uri=true",
        future=True,
        # TestClient runs request handlers in a threadpool; default SQLite
        # connections raise when a connection created in the setup thread
        # is re-used in the handler thread. The shared-cache file URI makes
        # this safe.
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    # Seed a user
    s = TestingSessionLocal()
    try:
        user = models.User(id=1, email="live@test", name="Live Tester")
        s.add(user)
        s.commit()
    finally:
        s.close()

    from main import app
    from routers import live_director as ld_router

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    def override_current_user():
        db = TestingSessionLocal()
        try:
            return db.query(models.User).filter(models.User.id == 1).first()
        finally:
            db.close()

    app.dependency_overrides[ld_router.get_db] = override_get_db
    app.dependency_overrides[auth.current_user] = override_current_user

    try:
        yield app, TestingSessionLocal
    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(engine)


@pytest.fixture
def client(app_and_session):
    app, _ = app_and_session
    return TestClient(app)


# ══════════════════════════════════════════════════════════════════════════════
# Event CRUD
# ══════════════════════════════════════════════════════════════════════════════


class TestEventCRUD:
    def test_create_event(self, client):
        r = client.post("/api/live/events", json={"name": "Test Concert"})
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "Test Concert"
        assert body["status"] == "scheduled"
        assert body["id"] >= 1

    def test_list_events(self, client):
        client.post("/api/live/events", json={"name": "E1"})
        client.post("/api/live/events", json={"name": "E2"})
        r = client.get("/api/live/events")
        assert r.status_code == 200
        names = [e["name"] for e in r.json()]
        assert "E1" in names and "E2" in names

    def test_get_event_detail(self, client):
        r1 = client.post("/api/live/events", json={"name": "Tour 2026"})
        ev_id = r1.json()["id"]
        r2 = client.get(f"/api/live/events/{ev_id}")
        assert r2.status_code == 200
        body = r2.json()
        assert body["id"] == ev_id
        assert body["cameras"] == []
        assert body["is_live_in_process"] is False

    def test_get_nonexistent_event_404(self, client):
        r = client.get("/api/live/events/99999")
        assert r.status_code == 404


# ══════════════════════════════════════════════════════════════════════════════
# Cameras
# ══════════════════════════════════════════════════════════════════════════════


class TestCameras:
    def test_add_camera(self, client):
        ev_id = client.post("/api/live/events", json={"name": "X"}).json()["id"]
        r = client.post(
            f"/api/live/events/{ev_id}/cameras",
            json={"cam_id": "cam1", "label": "Stage Left", "role_hints": ["stage"]},
        )
        assert r.status_code == 200
        assert r.json()["cam_id"] == "cam1"
        assert r.json()["role_hints"] == ["stage"]

    def test_add_duplicate_camera_conflict(self, client):
        ev_id = client.post("/api/live/events", json={"name": "X"}).json()["id"]
        client.post(
            f"/api/live/events/{ev_id}/cameras",
            json={"cam_id": "cam1", "label": "A"},
        )
        r = client.post(
            f"/api/live/events/{ev_id}/cameras",
            json={"cam_id": "cam1", "label": "B"},
        )
        assert r.status_code == 409


# ══════════════════════════════════════════════════════════════════════════════
# Lifecycle start/stop + operator controls
# ══════════════════════════════════════════════════════════════════════════════


class TestLifecycle:
    def test_start_without_cameras_400(self, client):
        ev_id = client.post("/api/live/events", json={"name": "X"}).json()["id"]
        r = client.post(f"/api/live/events/{ev_id}/start")
        assert r.status_code == 400

    def test_start_with_cameras_flips_status(self, client):
        ev_id = client.post("/api/live/events", json={"name": "X"}).json()["id"]
        client.post(f"/api/live/events/{ev_id}/cameras",
                    json={"cam_id": "cam1", "label": "A"})
        client.post(f"/api/live/events/{ev_id}/cameras",
                    json={"cam_id": "cam2", "label": "B"})
        r = client.post(f"/api/live/events/{ev_id}/start")
        assert r.status_code == 200
        assert r.json()["status"] == "live"
        # Stop afterwards to clean up
        client.post(f"/api/live/events/{ev_id}/stop")

    def test_double_start_409(self, client):
        ev_id = client.post("/api/live/events", json={"name": "X"}).json()["id"]
        client.post(f"/api/live/events/{ev_id}/cameras",
                    json={"cam_id": "cam1", "label": "A"})
        client.post(f"/api/live/events/{ev_id}/start")
        r = client.post(f"/api/live/events/{ev_id}/start")
        assert r.status_code == 409
        client.post(f"/api/live/events/{ev_id}/stop")

    def test_stop_flips_to_ended(self, client):
        ev_id = client.post("/api/live/events", json={"name": "X"}).json()["id"]
        client.post(f"/api/live/events/{ev_id}/cameras",
                    json={"cam_id": "cam1", "label": "A"})
        client.post(f"/api/live/events/{ev_id}/start")
        r = client.post(f"/api/live/events/{ev_id}/stop")
        assert r.status_code == 200
        assert r.json()["status"] == "ended"


# ══════════════════════════════════════════════════════════════════════════════
# Operator controls
# ══════════════════════════════════════════════════════════════════════════════


class TestOperatorControls:
    def test_pin_not_live_409(self, client):
        ev_id = client.post("/api/live/events", json={"name": "X"}).json()["id"]
        r = client.post(f"/api/live/events/{ev_id}/pin", json={"cam_id": "cam1"})
        assert r.status_code == 409

    def test_pin_unknown_camera_400(self, client):
        ev_id = client.post("/api/live/events", json={"name": "X"}).json()["id"]
        client.post(f"/api/live/events/{ev_id}/cameras",
                    json={"cam_id": "cam1", "label": "A"})
        client.post(f"/api/live/events/{ev_id}/start")
        r = client.post(f"/api/live/events/{ev_id}/pin", json={"cam_id": "cam_ghost"})
        assert r.status_code == 400
        client.post(f"/api/live/events/{ev_id}/stop")

    def test_pin_unpin_roundtrip(self, client):
        ev_id = client.post("/api/live/events", json={"name": "X"}).json()["id"]
        client.post(f"/api/live/events/{ev_id}/cameras",
                    json={"cam_id": "cam1", "label": "A"})
        client.post(f"/api/live/events/{ev_id}/start")
        r1 = client.post(f"/api/live/events/{ev_id}/pin", json={"cam_id": "cam1"})
        assert r1.status_code == 200
        assert r1.json()["pinned"] == "cam1"
        r2 = client.post(f"/api/live/events/{ev_id}/unpin")
        assert r2.status_code == 200
        assert r2.json()["pinned"] is None
        client.post(f"/api/live/events/{ev_id}/stop")

    def test_blacklist_allow_roundtrip(self, client):
        ev_id = client.post("/api/live/events", json={"name": "X"}).json()["id"]
        client.post(f"/api/live/events/{ev_id}/cameras",
                    json={"cam_id": "cam1", "label": "A"})
        client.post(f"/api/live/events/{ev_id}/start")
        r1 = client.post(f"/api/live/events/{ev_id}/blacklist", json={"cam_id": "cam1"})
        assert r1.status_code == 200 and r1.json()["blacklisted"] == "cam1"
        r2 = client.post(f"/api/live/events/{ev_id}/allow", json={"cam_id": "cam1"})
        assert r2.status_code == 200 and r2.json()["allowed"] == "cam1"
        client.post(f"/api/live/events/{ev_id}/stop")

    def test_force_cut(self, client):
        ev_id = client.post("/api/live/events", json={"name": "X"}).json()["id"]
        client.post(f"/api/live/events/{ev_id}/cameras",
                    json={"cam_id": "cam1", "label": "A"})
        client.post(f"/api/live/events/{ev_id}/cameras",
                    json={"cam_id": "cam2", "label": "B"})
        client.post(f"/api/live/events/{ev_id}/start")
        r = client.post(f"/api/live/events/{ev_id}/force-cut", json={"cam_id": "cam2"})
        assert r.status_code == 200
        assert r.json()["cutting_to"] == "cam2"
        client.post(f"/api/live/events/{ev_id}/stop")


# ══════════════════════════════════════════════════════════════════════════════
# Log endpoint
# ══════════════════════════════════════════════════════════════════════════════


class TestLogEndpoint:
    def test_empty_log(self, client):
        ev_id = client.post("/api/live/events", json={"name": "X"}).json()["id"]
        r = client.get(f"/api/live/events/{ev_id}/log")
        assert r.status_code == 200
        assert r.json() == []

    def test_log_limit_param(self, client):
        ev_id = client.post("/api/live/events", json={"name": "X"}).json()["id"]
        r = client.get(f"/api/live/events/{ev_id}/log?limit=10")
        assert r.status_code == 200
