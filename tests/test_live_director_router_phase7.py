"""Phase 7.7 backend — live-director router Phase 7 endpoint tests.

Covers RTMP relay config CRUD, chroma per-camera config, and dead-air
bridge config. Mirrors the style of test_live_director_router.py —
in-memory SQLite + dependency override for auth.current_user.
"""
from __future__ import annotations

import os
import tempfile

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
        "sqlite:///file:kaizer_live_router_phase7_test?mode=memory&cache=shared&uri=true",
        future=True,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    # Seed a user
    s = TestingSessionLocal()
    try:
        user = models.User(id=1, email="live7@test", name="Phase7 Tester")
        s.add(user)
        s.commit()
    finally:
        s.close()

    from main import app
    from routers import live_director as ld_router

    # Clear any left-over in-process sessions from previous tests.
    ld_router._SESSIONS.clear()

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
        ld_router._SESSIONS.clear()
        Base.metadata.drop_all(engine)


@pytest.fixture
def client(app_and_session):
    app, _ = app_and_session
    return TestClient(app)


def _new_event(client, name: str = "P7 Event") -> int:
    r = client.post("/api/live/events", json={"name": name})
    assert r.status_code == 200, r.text
    return r.json()["id"]


# ══════════════════════════════════════════════════════════════════════════════
# Relay — destination CRUD
# ══════════════════════════════════════════════════════════════════════════════


class TestRelayDestinations:
    def test_list_relay_destinations_empty(self, client):
        ev_id = _new_event(client)
        r = client.get(f"/api/live/events/{ev_id}/relay/destinations")
        assert r.status_code == 200
        assert r.json() == []

    def test_add_relay_destination_persists_to_config_json(self, client):
        ev_id = _new_event(client)
        payload = {
            "id": "yt_main",
            "name": "YouTube Main",
            "rtmp_url": "rtmp://a.rtmp.youtube.com/live2/XXX",
            "enabled": True,
            "reconnect_max_attempts": 0,
        }
        r = client.post(
            f"/api/live/events/{ev_id}/relay/destinations", json=payload,
        )
        assert r.status_code == 200, r.text
        assert r.json()["id"] == "yt_main"

        r2 = client.get(f"/api/live/events/{ev_id}/relay/destinations")
        assert r2.status_code == 200
        body = r2.json()
        assert len(body) == 1
        assert body[0]["id"] == "yt_main"
        assert body[0]["rtmp_url"] == "rtmp://a.rtmp.youtube.com/live2/XXX"

    def test_delete_relay_destination_removes_from_list(self, client):
        ev_id = _new_event(client)
        payload = {
            "id": "twitch_a",
            "name": "Twitch",
            "rtmp_url": "rtmp://live.twitch.tv/app/YYY",
        }
        r = client.post(
            f"/api/live/events/{ev_id}/relay/destinations", json=payload,
        )
        assert r.status_code == 200

        r_del = client.delete(
            f"/api/live/events/{ev_id}/relay/destinations/twitch_a",
        )
        assert r_del.status_code == 200
        assert r_del.json()["deleted"] == "twitch_a"

        r_list = client.get(f"/api/live/events/{ev_id}/relay/destinations")
        assert r_list.status_code == 200
        assert r_list.json() == []

    def test_delete_unknown_destination_returns_404(self, client):
        ev_id = _new_event(client)
        r = client.delete(
            f"/api/live/events/{ev_id}/relay/destinations/ghost_dest",
        )
        assert r.status_code == 404


# ══════════════════════════════════════════════════════════════════════════════
# Relay — start / status guards
# ══════════════════════════════════════════════════════════════════════════════


class TestRelayLifecycle:
    def test_relay_start_409_when_event_not_live(self, client):
        ev_id = _new_event(client)
        # Add a dest so the 409 can't be "no destinations".
        client.post(
            f"/api/live/events/{ev_id}/relay/destinations",
            json={
                "id": "d1", "name": "X",
                "rtmp_url": "rtmp://example.com/live/abc",
            },
        )
        r = client.post(f"/api/live/events/{ev_id}/relay/start")
        assert r.status_code == 409

    def test_relay_status_no_session_returns_not_running(self, client):
        ev_id = _new_event(client)
        r = client.get(f"/api/live/events/{ev_id}/relay/status")
        assert r.status_code == 200
        body = r.json()
        assert body["is_running"] is False
        assert body["destinations"] == []


# ══════════════════════════════════════════════════════════════════════════════
# Chroma — per-camera config CRUD
# ══════════════════════════════════════════════════════════════════════════════


class TestChromaConfig:
    def test_put_chroma_saves_to_config_json(self, client, tmp_path):
        ev_id = _new_event(client)
        # Create a real bg asset (png extension so validate doesn't reject).
        bg = tmp_path / "bg.png"
        bg.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)

        payload = {
            "color": "0x00d639",
            "similarity": 0.2,
            "blend": 0.1,
            "bg_asset_path": str(bg),
            "bg_asset_kind": "image",
            "bg_fit": "cover",
            "enabled": True,
        }
        r = client.put(
            f"/api/live/events/{ev_id}/cameras/cam1/chroma", json=payload,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["saved"] is True

        # Fetch event detail and verify config_json persisted.
        r2 = client.get(f"/api/live/events/{ev_id}")
        assert r2.status_code == 200
        cfg = r2.json()["config_json"]
        assert "chroma_configs" in cfg
        assert "cam1" in cfg["chroma_configs"]
        assert cfg["chroma_configs"]["cam1"]["bg_asset_path"] == str(bg)

    def test_put_chroma_missing_file_returns_404(self, client):
        ev_id = _new_event(client)
        payload = {
            "bg_asset_path": "/definitely/not/a/real/path/bg.png",
            "bg_asset_kind": "image",
            "enabled": True,
        }
        r = client.put(
            f"/api/live/events/{ev_id}/cameras/cam1/chroma", json=payload,
        )
        assert r.status_code == 404

    def test_put_chroma_bad_similarity_returns_400(self, client, tmp_path):
        ev_id = _new_event(client)
        bg = tmp_path / "bg.png"
        bg.write_bytes(b"\x89PNG\r\n\x1a\n")
        payload = {
            "similarity": 2.0,       # out of [0,1] → ValueError
            "bg_asset_path": str(bg),
            "bg_asset_kind": "image",
            "enabled": True,
        }
        r = client.put(
            f"/api/live/events/{ev_id}/cameras/cam1/chroma", json=payload,
        )
        assert r.status_code == 400

    def test_delete_chroma_removes_entry(self, client, tmp_path):
        ev_id = _new_event(client)
        bg = tmp_path / "bg.png"
        bg.write_bytes(b"\x89PNG\r\n\x1a\n")

        client.put(
            f"/api/live/events/{ev_id}/cameras/cam1/chroma",
            json={
                "bg_asset_path": str(bg),
                "bg_asset_kind": "image",
                "enabled": True,
            },
        )
        r = client.delete(f"/api/live/events/{ev_id}/cameras/cam1/chroma")
        assert r.status_code == 200
        assert r.json()["deleted"] == "cam1"

        # 404 on second delete.
        r2 = client.delete(f"/api/live/events/{ev_id}/cameras/cam1/chroma")
        assert r2.status_code == 404


# ══════════════════════════════════════════════════════════════════════════════
# Bridge — dead-air asset config
# ══════════════════════════════════════════════════════════════════════════════


class TestBridgeConfig:
    def test_put_bridge_saves_config(self, client, tmp_path):
        ev_id = _new_event(client)
        asset = tmp_path / "bridge.mp4"
        asset.write_bytes(b"\x00" * 32)

        payload = {
            "asset_url": str(asset),
            "silence_threshold_s": 2.5,
            "rms_ceiling": 0.01,
            "min_duration_s": 5.0,
        }
        r = client.put(f"/api/live/events/{ev_id}/bridge", json=payload)
        assert r.status_code == 200, r.text
        assert r.json()["saved"] is True

        r2 = client.get(f"/api/live/events/{ev_id}")
        cfg = r2.json()["config_json"]
        assert cfg["bridge_asset_url"] == str(asset)
        assert cfg["bridge_silence_threshold_s"] == 2.5
        assert cfg["bridge_silence_rms_ceiling"] == 0.01
        assert cfg["bridge_min_duration_s"] == 5.0

    def test_put_bridge_nonexistent_asset_returns_400(self, client):
        ev_id = _new_event(client)
        payload = {
            "asset_url": "/does/not/exist/bridge.mp4",
            "silence_threshold_s": 3.0,
            "rms_ceiling": 0.02,
            "min_duration_s": 4.0,
        }
        r = client.put(f"/api/live/events/{ev_id}/bridge", json=payload)
        assert r.status_code == 400
