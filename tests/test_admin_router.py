"""Tests for Phase 12 — admin router + gemini call logger.

Uses an in-memory SQLite DB + FastAPI dependency_overrides, mirroring the
pattern in tests/test_live_director_router.py.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import auth
import models
from database import Base


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def app_and_session():
    """Fresh in-memory SQLite per test — isolation over speed."""
    engine = create_engine(
        "sqlite:///file:kaizer_admin_test?mode=memory&cache=shared&uri=true",
        future=True,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    # Seed an admin + a regular user
    s = TestingSessionLocal()
    try:
        admin = models.User(id=1, email="admin@test", name="Admin", is_admin=True,  is_active=True)
        regular = models.User(id=2, email="user@test",  name="User",  is_admin=False, is_active=True)
        s.add(admin); s.add(regular); s.commit()
    finally:
        s.close()

    from main import app

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    # Default: run tests as the admin user.  Individual tests flip this
    # back to regular/disabled users to exercise the admin gate.
    def override_current_user():
        db = TestingSessionLocal()
        try:
            return db.query(models.User).filter(models.User.id == 1).first()
        finally:
            db.close()

    from database import get_db as real_get_db
    app.dependency_overrides[real_get_db] = override_get_db
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


@pytest.fixture
def SessionFactory(app_and_session):
    _, Factory = app_and_session
    return Factory


# ══════════════════════════════════════════════════════════════════════════════
# /system
# ══════════════════════════════════════════════════════════════════════════════

class TestSystemMetrics:
    def test_system_returns_required_keys(self, client):
        r = client.get("/api/admin/system")
        assert r.status_code == 200
        body = r.json()
        for key in ("cpu_percent", "cpu_count", "ram_total_gb", "ram_used_gb",
                    "ram_percent", "disk_total_gb", "disk_used_gb", "disk_percent",
                    "gpu", "process", "live_events_running", "timestamp"):
            assert key in body, f"missing key: {key}"
        assert isinstance(body["process"], dict)
        # gpu may be {} on a box without nvidia-smi — that's OK
        assert isinstance(body["gpu"], dict)


# ══════════════════════════════════════════════════════════════════════════════
# Admin gate
# ══════════════════════════════════════════════════════════════════════════════

class TestAdminGate:
    def test_non_admin_blocked(self, app_and_session):
        app, Factory = app_and_session

        def override_as_regular():
            db = Factory()
            try:
                return db.query(models.User).filter(models.User.id == 2).first()
            finally:
                db.close()

        app.dependency_overrides[auth.current_user] = override_as_regular
        c = TestClient(app)
        r = c.get("/api/admin/users")
        assert r.status_code == 403
        assert "Admin" in r.json().get("detail", "")

    def test_admin_allowed(self, client):
        r = client.get("/api/admin/users")
        assert r.status_code == 200
        body = r.json()
        assert "total" in body and "users" in body
        assert isinstance(body["users"], list)
        # Both seeded users should appear
        emails = {u["email"] for u in body["users"]}
        assert "admin@test" in emails
        assert "user@test" in emails


# ══════════════════════════════════════════════════════════════════════════════
# /users drill-down + toggle-admin
# ══════════════════════════════════════════════════════════════════════════════

class TestUserDetail:
    def test_user_drilldown(self, client):
        r = client.get("/api/admin/users/2")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == 2
        assert body["email"] == "user@test"
        assert "recent_jobs"         in body
        assert "recent_gemini_calls" in body
        assert "storage_breakdown_mb" in body

    def test_user_not_found(self, client):
        r = client.get("/api/admin/users/99999")
        assert r.status_code == 404


class TestToggleAdmin:
    def test_promote_regular(self, client):
        r = client.post("/api/admin/users/2/toggle-admin")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == 2
        assert body["is_admin"] is True

    def test_self_demotion_refused(self, client):
        # Acting admin has id=1 (set via dependency_overrides fixture)
        r = client.post("/api/admin/users/1/toggle-admin")
        assert r.status_code == 400
        assert "demote" in r.json()["detail"].lower()


# ══════════════════════════════════════════════════════════════════════════════
# /gemini-usage — empty buckets
# ══════════════════════════════════════════════════════════════════════════════

class TestGeminiUsageShape:
    def test_empty_db_returns_zero_buckets(self, client):
        r = client.get("/api/admin/gemini-usage")
        assert r.status_code == 200
        body = r.json()
        for key in ("total_calls", "total_tokens", "total_cost_usd",
                    "by_day", "by_user", "by_model", "by_purpose"):
            assert key in body
        assert body["total_calls"] == 0
        assert body["total_tokens"] == 0
        assert body["total_cost_usd"] == 0.0
        assert body["by_day"]     == []
        assert body["by_user"]    == []
        assert body["by_model"]   == []
        assert body["by_purpose"] == []


# ══════════════════════════════════════════════════════════════════════════════
# log_gemini_call — context manager behavior
# ══════════════════════════════════════════════════════════════════════════════

class _FakeUsage:
    def __init__(self, p=10, o=20, t=30):
        self.prompt_token_count = p
        self.candidates_token_count = o
        self.total_token_count = t


class _FakeResp:
    def __init__(self):
        self.usage_metadata = _FakeUsage()
        self.text = "ok"


class TestLogGeminiCall:
    def test_writes_ok_row_on_success(self, SessionFactory):
        from learning.gemini_log import log_gemini_call

        with SessionFactory() as db:
            with log_gemini_call(
                db=db, user_id=1, job_id=None, clip_id=None,
                model="gemini-2.0-flash", purpose="seo",
            ) as call:
                call.record(_FakeResp())
            # Commit happened inside the wrapper's finally — verify row exists
            rows = db.query(models.GeminiCall).all()
            assert len(rows) == 1
            r = rows[0]
            assert r.model == "gemini-2.0-flash"
            assert r.purpose == "seo"
            assert r.status == "ok"
            assert r.prompt_tokens == 10
            assert r.output_tokens == 20
            assert r.total_tokens == 30
            assert r.cost_usd > 0.0

    def test_writes_error_row_and_reraises(self, SessionFactory):
        from learning.gemini_log import log_gemini_call

        class _Boom(RuntimeError):
            pass

        with SessionFactory() as db:
            raised = False
            try:
                with log_gemini_call(
                    db=db, user_id=1,
                    model="gemini-2.5-flash", purpose="seo",
                ) as _call:
                    raise _Boom("simulated gemini failure")
            except _Boom:
                raised = True
            assert raised, "original exception must propagate"

            rows = db.query(models.GeminiCall).all()
            assert len(rows) == 1
            assert rows[0].status == "error"
            assert "simulated" in (rows[0].error or "")

    def test_429_is_classified_as_rate_limited(self, SessionFactory):
        from learning.gemini_log import log_gemini_call

        with SessionFactory() as db:
            try:
                with log_gemini_call(
                    db=db, model="gemini-2.5-flash", purpose="seo",
                ) as _call:
                    raise RuntimeError("429 Too Many Requests — quota exhausted")
            except RuntimeError:
                pass

            rows = db.query(models.GeminiCall).all()
            assert len(rows) == 1
            assert rows[0].status == "rate_limited"
