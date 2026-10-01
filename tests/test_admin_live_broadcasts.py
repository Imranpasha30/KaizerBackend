"""Admin view of Live Studio: what is on air, who ran it, what failed.

routers/admin.py had no reference to live_streams at all before this endpoint,
so an operator could not see broadcast activity across accounts. The two
properties worth defending are that the counts tell the truth, and that the
response never carries a secret -- live_streams holds yt_stream_key in PLAIN
TEXT, plus an ingest url and a signed backup url, none of which may reach a
browser.

Same in-memory SQLite + dependency_overrides pattern as tests/test_admin_router.py.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import auth
import models
from database import Base


def _now():
    return datetime.now(timezone.utc)


@pytest.fixture
def app_and_session():
    engine = create_engine(
        "sqlite:///file:kaizer_adminlive_test?mode=memory&cache=shared&uri=true",
        future=True,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    Factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    s = Factory()
    try:
        s.add(models.User(id=1, email="admin@test", name="Admin",
                          is_admin=True, is_active=True))
        s.add(models.User(id=2, email="rita@test", name="Rita",
                          is_admin=False, is_active=True))
        s.add(models.Channel(id=10, user_id=2, name="Cyber Sphere"))
        s.add(models.Channel(id=11, user_id=2, name="Second Channel"))

        # One of each thing the operator needs to see.
        s.add(models.LiveStream(
            id=100, batch_id="B1", user_id=2, channel_id=10, video_slot=0,
            status="streaming", title="On air bulletin",
            started_at=_now() - timedelta(minutes=20),
            progress_pct=42, created_at=_now() - timedelta(minutes=21)))
        s.add(models.LiveStream(
            id=101, batch_id="B1", user_id=2, channel_id=11, video_slot=0,
            status="done", title="Finished bulletin", yt_video_id="VID-OK",
            started_at=_now() - timedelta(hours=3),
            finished_at=_now() - timedelta(hours=2),
            created_at=_now() - timedelta(hours=3)))
        s.add(models.LiveStream(
            id=102, batch_id="B2", user_id=2, channel_id=10, video_slot=0,
            status="failed", title="Broken one",
            source_url="https://www.youtube.com/live/abc",
            error="yt-dlp not on PATH", created_at=_now() - timedelta(hours=1)))
        s.add(models.LiveStream(
            id=103, batch_id="B3", user_id=2, channel_id=10, video_slot=0,
            status="canceled", title="Stopped", created_at=_now() - timedelta(hours=4)))
        # Old, and still running: must appear in live_now despite the window.
        s.add(models.LiveStream(
            id=104, batch_id="B0", user_id=2, channel_id=11, video_slot=0,
            status="streaming", title="Long runner",
            started_at=_now() - timedelta(days=40),
            created_at=_now() - timedelta(days=40)))
        s.commit()
    finally:
        s.close()

    from main import app
    from database import get_db as real_get_db

    def override_get_db():
        db = Factory()
        try:
            yield db
        finally:
            db.close()

    def as_admin():
        db = Factory()
        try:
            return db.query(models.User).get(1)
        finally:
            db.close()

    app.dependency_overrides[real_get_db] = override_get_db
    app.dependency_overrides[auth.current_user] = as_admin
    try:
        yield app, Factory
    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(engine)


@pytest.fixture
def client(app_and_session):
    return TestClient(app_and_session[0])


def _get(client, days=7):
    r = client.get(f"/api/admin/live-broadcasts?days={days}")
    assert r.status_code == 200, r.text
    return r.json()


# ── the gate ─────────────────────────────────────────────────────────

def test_a_non_admin_is_refused(app_and_session):
    app, Factory = app_and_session

    def as_regular():
        db = Factory()
        try:
            return db.query(models.User).get(2)
        finally:
            db.close()

    app.dependency_overrides[auth.current_user] = as_regular
    r = TestClient(app).get("/api/admin/live-broadcasts")
    assert r.status_code == 403


# ── shape ────────────────────────────────────────────────────────────

def test_the_payload_has_every_section(client):
    b = _get(client)
    for k in ("window", "totals", "live_now", "recent", "by_user", "by_channel"):
        assert k in b, f"missing {k}"


# ── counts that tell the truth ───────────────────────────────────────

def test_live_now_is_not_windowed(client):
    """A broadcast started 40 days ago can still be running. Filtering
    live_now by the date range would hide exactly the row an operator opens
    this page to find."""
    b = _get(client, days=7)
    ids = {r["id"] for r in b["live_now"]}
    assert 100 in ids, "the recent streaming row is missing"
    assert 104 in ids, "a still-running broadcast older than the window was hidden"
    assert b["totals"]["live_now"] == 2


def test_terminal_rows_are_not_counted_as_live(client):
    b = _get(client, days=90)
    ids = {r["id"] for r in b["live_now"]}
    assert ids.isdisjoint({101, 102, 103})


def test_success_rate_ignores_broadcasts_still_running(client):
    """Counting in-flight broadcasts as failures would make a busy day look
    broken. 1 done / (1 done + 1 failed) = 0.5."""
    b = _get(client, days=7)
    assert b["totals"]["done"] == 1
    assert b["totals"]["failed"] == 1
    assert b["totals"]["success_rate"] == 0.5


def test_success_rate_is_null_when_nothing_has_finished(app_and_session):
    app, Factory = app_and_session
    s = Factory()
    try:
        for r in s.query(models.LiveStream).filter(
                models.LiveStream.status.in_(("done", "failed"))).all():
            s.delete(r)
        s.commit()
    finally:
        s.close()
    b = TestClient(app).get("/api/admin/live-broadcasts?days=7").json()
    assert b["totals"]["success_rate"] is None


# ── the operator's actual questions ──────────────────────────────────

def test_a_successful_broadcast_carries_its_watch_link(client):
    b = _get(client, days=7)
    row = next(r for r in b["recent"] if r["id"] == 101)
    assert row["watch_url"] == "https://www.youtube.com/watch?v=VID-OK"


def test_a_row_with_no_video_id_has_no_link(client):
    b = _get(client, days=7)
    row = next(r for r in b["recent"] if r["id"] == 102)
    assert row["watch_url"] is None


def test_a_failure_carries_its_reason(client):
    b = _get(client, days=7)
    row = next(r for r in b["recent"] if r["id"] == 102)
    assert row["status"] == "failed"
    assert "yt-dlp" in (row["error"] or "")


def test_url_and_upload_sources_are_distinguishable(client):
    """The two paths fail in completely different ways, so the operator has
    to be able to tell them apart without opening the row."""
    b = _get(client, days=7)
    rows = {r["id"]: r for r in b["recent"]}
    assert rows[102]["source"] == "url"
    assert rows[101]["source"] == "upload"


def test_rows_name_the_user_and_channel(client):
    b = _get(client, days=7)
    row = next(r for r in b["recent"] if r["id"] == 102)
    assert row["user_email"] == "rita@test"
    assert row["channel_name"] == "Cyber Sphere"


def test_duration_is_computed_for_finished_rows(client):
    b = _get(client, days=7)
    row = next(r for r in b["recent"] if r["id"] == 101)
    assert row["duration_s"] == pytest.approx(3600, abs=120)


def test_rollups_group_by_user_and_channel(client):
    b = _get(client, days=7)
    assert any(u["email"] == "rita@test" for u in b["by_user"])
    names = {c["name"] for c in b["by_channel"]}
    assert "Cyber Sphere" in names


# ── secrets must not travel ──────────────────────────────────────────

def test_no_stream_key_or_ingest_url_is_ever_returned(app_and_session):
    """live_streams stores yt_stream_key in PLAIN TEXT alongside the ingest
    url and a signed backup url. The response is built from named fields
    rather than serialised off the row precisely so these cannot leak."""
    app, Factory = app_and_session
    s = Factory()
    try:
        row = s.query(models.LiveStream).get(102)
        row.yt_stream_key = "SUPER-SECRET-KEY"
        row.yt_ingest_url = "rtmps://a.rtmps.youtube.com/live2"
        row.backup_url = "https://r2.example/signed?token=SECRET-TOKEN"
        s.commit()
    finally:
        s.close()

    raw = TestClient(app).get("/api/admin/live-broadcasts?days=7").text
    assert "SUPER-SECRET-KEY" not in raw
    assert "SECRET-TOKEN" not in raw
    assert "rtmps://" not in raw
    for forbidden in ("yt_stream_key", "yt_ingest_url", "backup_url", "upload_path"):
        assert forbidden not in raw, f"{forbidden} leaked into the admin payload"


# ── the sweeper that would otherwise lie ─────────────────────────────

def test_the_orphan_sweeper_covers_every_non_terminal_status():
    """A stream interrupted during the branding pass used to be missed by the
    restart sweeper, so it stayed 'branding' for ever and would show as
    permanently on air in this tab."""
    import inspect
    import main
    src = inspect.getsource(main._live_studio_orphan_sweeper)
    for status in ("starting", "provisioning", "streaming", "queued",
                   "downloading", "uploaded", "uploading", "branding", "preparing"):
        assert f"'{status}'" in src, f"the sweeper ignores {status!r}"

    # It is a Python loop now, not one UPDATE. It had to become one: a single
    # statement cannot ask the engine whether a row is still live, and the
    # engine's relays run in their own process, so they survive the restart this
    # sweeper is cleaning up after. Burying them marked a streaming broadcast
    # failed AND -- because cancel_stream returns early on a terminal status --
    # disarmed the customer's own Stop button.
    assert "engine_is_carrying" in src, "the sweeper buries without asking again"
    assert "UPDATE live_streams" not in src


# ── did manual mode really cost nothing? ─────────────────────────────

def test_credit_proof_is_admin_only(app_and_session):
    app, Factory = app_and_session

    def as_regular():
        db = Factory()
        try:
            return db.query(models.User).get(2)
        finally:
            db.close()

    app.dependency_overrides[auth.current_user] = as_regular
    assert TestClient(app).get("/api/admin/live/credit-proof").status_code == 403


def test_credit_proof_measures_spend_from_the_audit_log(app_and_session):
    """The claim that manual mode is free has to be a MEASUREMENT, not a
    sentence in a report. Every API call this backend makes is recorded with
    its published unit cost, so the spend is a fact."""
    app, Factory = app_and_session
    s = Factory()
    try:
        # one connected broadcast: insert + bind + transition, against the
        # live_stream_id column that the FK bug used to reject
        for op, cost in (("liveBroadcasts.insert", 50),
                         ("liveBroadcasts.bind", 50),
                         ("liveBroadcasts.transition", 50)):
            s.add(models.YouTubeApiCall(user_id=2, live_stream_id=101,
                                        operation=op, quota_cost=cost,
                                        success=True, created_at=_now()))
        s.commit()
    finally:
        s.close()

    b = TestClient(app).get("/api/admin/live/credit-proof?hours=24").json()
    assert b["connected"]["broadcasts_with_api_calls"] == 1
    assert b["connected"]["units"] == 150
    assert b["connected"]["units_per_broadcast"] == 150.0
    ops = {r["operation"]: r["units"] for r in b["by_operation"]}
    assert ops["liveBroadcasts.insert"] == 50
    assert b["total_units_all_operations"] == 150


def test_credit_proof_says_what_it_found_in_words(app_and_session):
    """An operator should not have to compare two numbers to learn the
    answer."""
    b = TestClient(app_and_session[0]).get("/api/admin/live/credit-proof").json()
    assert isinstance(b["verdict"], str) and len(b["verdict"]) > 20


def test_credit_proof_attributes_nothing_to_manual_broadcasts(app_and_session):
    """A manual broadcast makes no API calls at all, so it has no rows. Any
    row appearing against one would mean the feature is not doing what it
    says — which is exactly what this view exists to catch."""
    b = TestClient(app_and_session[0]).get("/api/admin/live/credit-proof").json()
    assert b["connected"]["broadcasts_with_api_calls"] == 0
    assert b["connected"]["units"] == 0
    assert b["engine_started"]["manual"] == 0
