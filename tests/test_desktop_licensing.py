"""tests/test_desktop_licensing.py — desktop machine activation + licensing.

Exercises routers/desktop.py (ported from kaizer-platform@d5fd482):
  * first activation of a fingerprint creates a DesktopLicense row.
  * re-activating the same fingerprint is idempotent (bumps last_seen_at,
    does not create a second row, does not count twice against the limit).
  * the placeholder ACTIVATION_LIMIT (3) is enforced — the 4th distinct
    fingerprint for one user is rejected with 403.
  * require_desktop_license: missing X-Desktop-Machine-Fingerprint header
    is rejected; an unactivated fingerprint is 403; a revoked fingerprint
    is 403; a valid call bumps last_seen_at (heartbeat).
  * POST /licenses/{id}/revoke is owner-only (someone else's id 404s and
    leaves the row untouched); a revoked fingerprint can't re-activate.
  * GET /licenses is scoped to the caller's own user_id.

Same in-memory-SQLite + dependency-override conventions as
tests/test_live_director_router.py / test_feedback_api.py, but on a
minimal FastAPI app carrying just this router (it is not yet wired into
main.py — the orchestrator does that).
"""
from __future__ import annotations

import time

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import auth
import models
from database import Base, get_db
from routers import desktop as desktop_router


@pytest.fixture
def app_env():
    engine = create_engine(
        "sqlite:///file:kaizer_desktop_license_test?mode=memory&cache=shared&uri=true",
        future=True,
        # TestClient runs handlers in a threadpool; the shared-cache file
        # URI + check_same_thread=False make cross-thread reuse safe.
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, expire_on_commit=False,
    )

    s = TestingSessionLocal()
    try:
        s.add_all([
            models.User(id=1, email="owner@x.com", name="owner"),
            models.User(id=2, email="other@x.com", name="other"),
        ])
        s.commit()
    finally:
        s.close()

    # Minimal app: just the desktop router + a probe route so
    # require_desktop_license (a dependency with no production consumer
    # yet) is exercised through real request plumbing.
    app = FastAPI()
    app.include_router(desktop_router.router)

    @app.get("/api/desktop/_probe")
    def _probe(
        lic: models.DesktopLicense = Depends(desktop_router.require_desktop_license),
    ):
        return {"license_id": lic.id, "fingerprint": lic.machine_fingerprint}

    acting = {"user_id": 1}

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    def override_user():
        db = TestingSessionLocal()
        try:
            return db.query(models.User).filter(
                models.User.id == acting["user_id"]).first()
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[auth.current_user] = override_user
    try:
        yield app, TestingSessionLocal, acting
    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(engine)


@pytest.fixture
def client(app_env):
    app, _, _ = app_env
    return TestClient(app)


def _row(Session, user_id, fingerprint):
    db = Session()
    try:
        return (
            db.query(models.DesktopLicense)
            .filter(models.DesktopLicense.user_id == user_id,
                    models.DesktopLicense.machine_fingerprint == fingerprint)
            .first()
        )
    finally:
        db.close()


def _count(Session, user_id):
    db = Session()
    try:
        return (
            db.query(models.DesktopLicense)
            .filter(models.DesktopLicense.user_id == user_id)
            .count()
        )
    finally:
        db.close()


# ───────────────────────────────────────────────────────────────────
# Activation
# ───────────────────────────────────────────────────────────────────

def test_first_activation_creates_license(app_env, client):
    _, Session, _ = app_env
    r = client.post("/api/desktop/activate", json={
        "machine_fingerprint": "fp-aaaaaaaa", "machine_label": "Imran-PC",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["already_activated"] is False
    assert body["active_count"] == 1
    assert body["limit"] == 3
    assert body["license"]["machine_fingerprint"] == "fp-aaaaaaaa"
    assert body["license"]["revoked"] is False
    assert body["license"]["last_seen_at"] is not None

    row = _row(Session, 1, "fp-aaaaaaaa")
    assert row is not None
    assert row.machine_label == "Imran-PC"
    assert row.revoked is False


def test_reactivate_same_fingerprint_idempotent_and_bumps_last_seen(app_env, client):
    _, Session, _ = app_env
    r1 = client.post("/api/desktop/activate",
                     json={"machine_fingerprint": "fp-bbbbbbbb"})
    assert r1.status_code == 200, r1.text
    seen1 = _row(Session, 1, "fp-bbbbbbbb").last_seen_at
    assert seen1 is not None

    time.sleep(0.02)  # ensure a measurable clock delta
    r2 = client.post("/api/desktop/activate",
                     json={"machine_fingerprint": "fp-bbbbbbbb"})
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["already_activated"] is True
    assert body["active_count"] == 1          # still one machine, not two
    assert _count(Session, 1) == 1            # no new row

    seen2 = _row(Session, 1, "fp-bbbbbbbb").last_seen_at
    assert seen2 > seen1                      # heartbeat bumped


def test_fourth_device_rejected_at_activation_limit(app_env, client):
    _, Session, _ = app_env
    for i in range(desktop_router.ACTIVATION_LIMIT):      # 3 machines OK
        r = client.post("/api/desktop/activate",
                        json={"machine_fingerprint": f"fp-limit-{i}00"})
        assert r.status_code == 200, r.text
    r4 = client.post("/api/desktop/activate",
                     json={"machine_fingerprint": "fp-limit-4th"})
    assert r4.status_code == 403
    assert "Activation limit reached" in r4.json()["detail"]
    assert _count(Session, 1) == 3            # 4th row never created


def test_blank_fingerprint_rejected(client):
    r = client.post("/api/desktop/activate", json={"machine_fingerprint": ""})
    assert r.status_code == 422               # pydantic min_length=8


# ───────────────────────────────────────────────────────────────────
# require_desktop_license (via the probe route)
# ───────────────────────────────────────────────────────────────────

def test_missing_fingerprint_header_rejected(client):
    r = client.get("/api/desktop/_probe")     # no header at all
    assert r.status_code == 422


def test_unactivated_fingerprint_rejected(client):
    r = client.get("/api/desktop/_probe",
                   headers={"X-Desktop-Machine-Fingerprint": "fp-never-seen"})
    assert r.status_code == 403
    assert "No activated desktop license" in r.json()["detail"]


def test_valid_license_passes_and_heartbeats(app_env, client):
    _, Session, _ = app_env
    client.post("/api/desktop/activate",
                json={"machine_fingerprint": "fp-cccccccc"})
    seen1 = _row(Session, 1, "fp-cccccccc").last_seen_at
    time.sleep(0.02)
    r = client.get("/api/desktop/_probe",
                   headers={"X-Desktop-Machine-Fingerprint": "fp-cccccccc"})
    assert r.status_code == 200, r.text
    assert r.json()["fingerprint"] == "fp-cccccccc"
    seen2 = _row(Session, 1, "fp-cccccccc").last_seen_at
    assert seen2 > seen1                      # last_seen_at heartbeat


def test_revoked_fingerprint_rejected_by_require_desktop_license(app_env, client):
    _, Session, _ = app_env
    lic_id = client.post("/api/desktop/activate", json={
        "machine_fingerprint": "fp-dddddddd"}).json()["license"]["id"]
    assert client.post(
        f"/api/desktop/licenses/{lic_id}/revoke").status_code == 200
    r = client.get("/api/desktop/_probe",
                   headers={"X-Desktop-Machine-Fingerprint": "fp-dddddddd"})
    assert r.status_code == 403
    assert "revoked" in r.json()["detail"]


# ───────────────────────────────────────────────────────────────────
# Revoke endpoint (Kaizer X addition)
# ───────────────────────────────────────────────────────────────────

def test_revoke_owner_only(app_env, client):
    _, Session, acting = app_env
    lic_id = client.post("/api/desktop/activate", json={
        "machine_fingerprint": "fp-eeeeeeee"}).json()["license"]["id"]

    acting["user_id"] = 2                     # someone else tries to revoke
    r = client.post(f"/api/desktop/licenses/{lic_id}/revoke")
    assert r.status_code == 404               # existence not leaked
    assert _row(Session, 1, "fp-eeeeeeee").revoked is False   # untouched

    acting["user_id"] = 1                     # the owner revokes
    r = client.post(f"/api/desktop/licenses/{lic_id}/revoke")
    assert r.status_code == 200, r.text
    assert r.json()["revoked"] is True
    assert _row(Session, 1, "fp-eeeeeeee").revoked is True

    # idempotent second revoke
    r = client.post(f"/api/desktop/licenses/{lic_id}/revoke")
    assert r.status_code == 200
    assert r.json()["revoked"] is True

    # a revoked fingerprint cannot re-activate
    r = client.post("/api/desktop/activate",
                    json={"machine_fingerprint": "fp-eeeeeeee"})
    assert r.status_code == 403
    assert "revoked" in r.json()["detail"]


def test_revoke_unknown_id_404(client):
    assert client.post("/api/desktop/licenses/99999/revoke").status_code == 404


# ───────────────────────────────────────────────────────────────────
# License list scoping
# ───────────────────────────────────────────────────────────────────

def test_licenses_list_scoped_to_caller(app_env, client):
    _, _, acting = app_env
    client.post("/api/desktop/activate",
                json={"machine_fingerprint": "fp-user1-aaaa"})
    acting["user_id"] = 2
    client.post("/api/desktop/activate",
                json={"machine_fingerprint": "fp-user2-bbbb"})

    r = client.get("/api/desktop/licenses")   # still acting as user 2
    assert r.status_code == 200
    fps = [l["machine_fingerprint"] for l in r.json()]
    assert fps == ["fp-user2-bbbb"]

    acting["user_id"] = 1
    r = client.get("/api/desktop/licenses")
    fps = [l["machine_fingerprint"] for l in r.json()]
    assert fps == ["fp-user1-aaaa"]
