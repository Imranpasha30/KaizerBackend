"""Tests for routers.feedback — GET /api/uploads/{id}/feedback."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import models
from database import Base


@pytest.fixture
def test_app(monkeypatch):
    """Spin up FastAPI app with in-memory SQLite for isolated testing."""
    # Shared in-memory DB (file-based URI so multiple connections see it)
    engine = create_engine(
        "sqlite:///file:kaizer_fb_test?mode=memory&cache=shared&uri=true",
        future=True,
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    from main import app
    from routers import feedback as fb_router

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[fb_router.get_db] = override_get_db
    try:
        yield app, TestingSessionLocal
    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(engine)


def _seed(db, *, upload_id=1, narrative_meta=None, publish_kind="short"):
    u = models.User(id=1, email="u@x.com", name="u")
    db.add(u)
    c = models.Channel(id=1, user_id=1, name="c")
    db.add(c)
    job = models.Job(id=100, user_id=1, status="done",
                     platform="youtube_short", video_name="t")
    db.add(job)
    db.flush()
    clip = models.Clip(
        id=1, job_id=100, clip_index=0, duration=45.0,
        meta=json.dumps(narrative_meta or {}),
    )
    db.add(clip)
    up = models.UploadJob(
        id=upload_id, user_id=1, clip_id=1, channel_id=1,
        status="done", publish_kind=publish_kind, title="t",
        video_id="v1",
    )
    db.add(up)
    db.commit()


def test_feedback_route_404_on_nonexistent_upload(test_app):
    app, _ = test_app
    client = TestClient(app)
    r = client.get("/api/uploads/99999/feedback")
    assert r.status_code == 404


def test_feedback_route_returns_no_analytics_without_retention(test_app):
    app, Session = test_app
    db = Session()
    _seed(db)
    db.close()
    client = TestClient(app)
    r = client.get("/api/uploads/1/feedback")
    assert r.status_code == 200
    body = r.json()
    assert body["upload_job_id"] == 1
    assert body["status"] == "no_analytics"
    assert body["retention_curve"] == []
    assert isinstance(body["recommendations"], list)


def test_feedback_response_schema_complete(test_app):
    app, Session = test_app
    db = Session()
    _seed(db, narrative_meta={
        "narrative_role": "climax",
        "hook_score": 0.7,
        "completion_score": 0.6,
    })
    db.close()
    client = TestClient(app)
    r = client.get("/api/uploads/1/feedback")
    assert r.status_code == 200
    body = r.json()
    for key in (
        "upload_job_id", "status", "retention_curve", "dropoffs",
        "recommendations", "explainability", "warnings",
    ):
        assert key in body, f"Response missing field {key!r}"


def test_feedback_route_explainability_surfaces_narrative_meta(test_app):
    app, Session = test_app
    db = Session()
    _seed(db, narrative_meta={
        "narrative_role": "turn",
        "hook_score": 0.55,
    })
    db.close()
    client = TestClient(app)
    r = client.get("/api/uploads/1/feedback")
    assert r.status_code == 200
    body = r.json()
    assert body["explainability"]["narrative_role"] == "turn"
    assert body["explainability"]["hook_score"] == 0.55


def test_feedback_recommendations_have_required_fields(test_app):
    app, Session = test_app
    db = Session()
    _seed(db)
    db.close()
    client = TestClient(app)
    r = client.get("/api/uploads/1/feedback")
    body = r.json()
    for rec in body["recommendations"]:
        assert "kind" in rec
        assert "message" in rec
        assert "actionable" in rec


def test_feedback_route_youtube_api_key_query_param_accepted(test_app):
    """Passing ?youtube_api_key=fake must not 500 — stub path returns empty curve."""
    app, Session = test_app
    db = Session()
    _seed(db)
    db.close()
    client = TestClient(app)
    r = client.get("/api/uploads/1/feedback?youtube_api_key=synthetic")
    assert r.status_code == 200
    # Still stub — v1 returns no_analytics even with a key
    assert r.json()["status"] == "no_analytics"
