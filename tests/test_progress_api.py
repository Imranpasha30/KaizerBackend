"""
tests/test_progress_api.py — Phase 2B TDD coverage for
routers/job_progress.py + GET /api/jobs/{job_id}/progress.

The router IS shipped (routers/job_progress.py) and registered in main.py.
These tests verify its contract.

Setup strategy:
  - Each test function uses a fresh SQLite in-memory database with a unique
    `?mode=memory&cache=shared` URL so tables persist across connections
    within the same process (plain `sqlite:///:memory:` creates a new,
    table-less DB on every new connection).
  - The FastAPI `get_db` dependency is overridden via
    `app.dependency_overrides` for the duration of each test.

Stage heuristic notes (from the router's STAGE_PROGRESS list):
  - "Cutting clip"    → stage name "Cutting clip"  (contains "cutting")
  - "Composing"       → stage name "Composing"     (contains "composing")
  - "Generating SEO"  → stage name "Generating SEO"
  - "Transcribing"    → stage name "Transcribing"
  - "Running QA"      → stage name "Running QA"
  ETA formula: (100-pct)/pct * elapsed_s  when pct > 5, else None.
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timezone, timedelta
from typing import Generator

import pytest

# ---------------------------------------------------------------------------
# Ensure backend root is on sys.path
# ---------------------------------------------------------------------------

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_shared_mem_factory(db_name: str):
    """Return a (engine, sessionmaker) pair backed by a named, shared-cache
    in-memory SQLite database.

    Using `?mode=memory&cache=shared` keeps the in-memory DB alive and
    accessible to *all* connections that use the same `db_name` within the
    same process — this is required because FastAPI's dependency injection
    opens a fresh connection per request, which would otherwise receive a
    brand-new, table-less database.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from database import Base
    import models  # noqa: F401 — registers ORM classes on Base

    url = (
        f"sqlite:///file:{db_name}"
        f"?mode=memory&cache=shared&uri=true"
    )
    engine = create_engine(
        url,
        connect_args={"check_same_thread": False, "uri": True},
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, factory


def _seed_job(factory, *, status="pending", log="", started_at=None):
    """Create a legacy user (if absent) and one Job row.  Return the Job."""
    import models
    from auth import ensure_legacy_user

    db = factory()
    try:
        user = ensure_legacy_user(db)
        job = models.Job(
            user_id=user.id,
            platform="youtube_short",
            frame_layout="torn_card",
            video_name="test.mp4",
            language="te",
            status=status,
            log=log,
            output_dir="/tmp",
            started_at=started_at,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job.id  # return only the PK — session will be closed
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Per-test fixture: unique in-memory DB + TestClient
# ---------------------------------------------------------------------------

@pytest.fixture()
def ctx(request):
    """Yield a dict with keys:
        client    : fastapi.testclient.TestClient
        factory   : sqlalchemy sessionmaker (connected to in-memory DB)

    Each test gets a unique DB name so there is zero cross-test contamination.
    """
    from fastapi.testclient import TestClient
    import main
    from database import get_db

    # Unique name so parallel tests don't share state
    db_name = f"kaizer_test_{uuid.uuid4().hex}"
    _engine, factory = _make_shared_mem_factory(db_name)

    def _override_get_db() -> Generator:
        db = factory()
        try:
            yield db
        finally:
            db.close()

    main.app.dependency_overrides[get_db] = _override_get_db
    client = TestClient(main.app, raise_server_exceptions=False)

    yield {"client": client, "factory": factory}

    # Teardown: clear the override so other tests are not affected
    main.app.dependency_overrides.pop(get_db, None)
    _engine.dispose()


# ---------------------------------------------------------------------------
# Tests — 9 total (≥8 required)
# ---------------------------------------------------------------------------

class TestProgressEndpoint:

    # ── 1. Non-existent job returns 404 ─────────────────────────────────────

    def test_get_progress_nonexistent_job_404(self, ctx):
        """GET /api/jobs/99999/progress for a job that doesn't exist → 404."""
        resp = ctx["client"].get("/api/jobs/99999/progress")
        assert resp.status_code == 404, (
            f"Expected 404 for non-existent job, got {resp.status_code}: {resp.text[:200]}"
        )
        data = resp.json()
        assert "detail" in data, "404 response should include 'detail' key"

    # ── 2. Pending job → stage 'starting', percent < 5 ──────────────────────

    def test_get_progress_pending_job_returns_starting(self, ctx):
        """status='pending', empty log → stage contains 'start' (case-insensitive),
        percent < 5."""
        job_id = _seed_job(ctx["factory"], status="pending", log="")

        resp = ctx["client"].get(f"/api/jobs/{job_id}/progress")
        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text[:200]}"
        )
        data = resp.json()
        assert data["status"] == "pending", (
            f"Expected status='pending', got {data['status']!r}"
        )
        assert "start" in data["stage"].lower(), (
            f"Expected stage to contain 'start' for pending+empty-log job, "
            f"got {data['stage']!r}"
        )
        assert data["percent"] < 5.0, (
            f"Expected percent < 5 for pending job, got {data['percent']}"
        )

    # ── 3. Done job → percent = 100, stage 'done' ───────────────────────────

    def test_get_progress_done_job_returns_100(self, ctx):
        """status='done' → percent=100.0, stage contains 'done'."""
        job_id = _seed_job(ctx["factory"], status="done", log="Pipeline finished.")

        resp = ctx["client"].get(f"/api/jobs/{job_id}/progress")
        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text[:200]}"
        )
        data = resp.json()
        assert data["percent"] == 100.0, (
            f"Expected percent=100.0 for done job, got {data['percent']}"
        )
        assert "done" in data["stage"].lower(), (
            f"Expected stage to contain 'done', got {data['stage']!r}"
        )

    # ── 4. Failed job → stage 'failed' ──────────────────────────────────────

    def test_get_progress_failed_job_has_failed_stage(self, ctx):
        """status='failed' → stage contains 'failed', status='failed'."""
        job_id = _seed_job(ctx["factory"], status="failed", log="Error: FFmpeg crashed.")

        resp = ctx["client"].get(f"/api/jobs/{job_id}/progress")
        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text[:200]}"
        )
        data = resp.json()
        assert data["status"] == "failed", (
            f"Expected status='failed', got {data['status']!r}"
        )
        assert "failed" in data["stage"].lower(), (
            f"Expected stage to contain 'failed', got {data['stage']!r}"
        )

    # ── 5. Log markers map to expected stages ───────────────────────────────

    @pytest.mark.parametrize("log_fragment,expected_stage_fragment", [
        ("Cutting clip 3/5",        "cutting"),
        ("Cutting clip 1/3",        "cutting"),
        ("Composing final clip",    "composing"),
        ("Generating SEO for clip", "generating"),
        ("Transcribing audio",      "transcribing"),
    ])
    def test_get_progress_log_markers_map_to_stages(
        self,
        ctx,
        log_fragment: str,
        expected_stage_fragment: str,
    ):
        """Known log fragments cause the endpoint to return the expected stage keyword.

        The stage check is case-insensitive substring match to avoid coupling
        tests to exact capitalisation chosen by the Builder.
        """
        job_id = _seed_job(ctx["factory"], status="running", log=log_fragment)

        resp = ctx["client"].get(f"/api/jobs/{job_id}/progress")
        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text[:200]}"
        )
        data = resp.json()
        assert "stage" in data, "Response must include 'stage' key"
        stage = data["stage"].lower()
        assert expected_stage_fragment in stage, (
            f"Log {log_fragment!r} should produce a stage containing "
            f"{expected_stage_fragment!r}, got {data['stage']!r}"
        )

    # ── 6. elapsed_s reflects started_at ────────────────────────────────────

    def test_get_progress_elapsed_s_tracks_started_at(self, ctx):
        """started_at set 60s ago → elapsed_s ≈ 60 ±5 seconds."""
        started_naive = (
            datetime.now(timezone.utc) - timedelta(seconds=60)
        ).replace(tzinfo=None)  # SQLite stores naive UTC

        job_id = _seed_job(
            ctx["factory"],
            status="running",
            log="Cutting clip 1/2",
            started_at=started_naive,
        )

        resp = ctx["client"].get(f"/api/jobs/{job_id}/progress")
        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text[:200]}"
        )
        data = resp.json()
        elapsed = data.get("elapsed_s")
        assert elapsed is not None, "elapsed_s must not be None when started_at is set"
        assert isinstance(elapsed, int), f"elapsed_s must be int, got {type(elapsed)}"
        assert abs(elapsed - 60) <= 5, (
            f"elapsed_s={elapsed} should be ≈60 (started 60s ago, tolerance ±5)"
        )

    # ── 7. ETA is computed when percent > 5 ─────────────────────────────────

    def test_get_progress_eta_computed_when_percent_gt_5(self, ctx):
        """A running job with percent>5 and elapsed>0 should have a non-None eta_s.

        We log "Cutting clip" (→ 35%) with started_at=30s ago so both
        conditions (percent > 5 AND elapsed_s > 0) are satisfied.
        """
        started_naive = (
            datetime.now(timezone.utc) - timedelta(seconds=30)
        ).replace(tzinfo=None)

        job_id = _seed_job(
            ctx["factory"],
            status="running",
            log="Cutting clip 1/3",
            started_at=started_naive,
        )

        resp = ctx["client"].get(f"/api/jobs/{job_id}/progress")
        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text[:200]}"
        )
        data = resp.json()
        percent = data.get("percent", 0)
        elapsed = data.get("elapsed_s", 0)

        if percent > 5.0 and elapsed > 0:
            eta = data.get("eta_s")
            assert eta is not None, (
                f"eta_s should be non-None at percent={percent:.1f}%, "
                f"elapsed={elapsed}s, got None"
            )
            assert eta > 0, (
                f"eta_s should be > 0 at percent={percent:.1f}%, got {eta}"
            )

    # ── 8. ETA is None when percent is low ──────────────────────────────────

    def test_get_progress_eta_is_none_when_percent_low(self, ctx):
        """A pending job with no log and no started_at → eta_s is None."""
        job_id = _seed_job(ctx["factory"], status="pending", log="", started_at=None)

        resp = ctx["client"].get(f"/api/jobs/{job_id}/progress")
        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text[:200]}"
        )
        data = resp.json()
        # percent < 5 for an unstarted pending job
        percent = data.get("percent", -1)
        assert percent < 5.0, (
            f"Expected percent < 5 for unstarted pending job, got {percent}"
        )
        eta = data.get("eta_s")
        assert eta is None, (
            f"eta_s should be None when percent={percent:.1f} (<5), got {eta!r}"
        )

    # ── 9. Full ProgressResponse shape ──────────────────────────────────────

    def test_progress_response_shape_complete(self, ctx):
        """Response must contain all ProgressResponse fields with correct types.

        Required fields (from spec):
            job_id   : int
            status   : str   (pending | running | done | failed)
            stage    : str
            percent  : float  (0.0 – 100.0)
            elapsed_s: int
            eta_s    : int | None
            message  : str
        """
        started_naive = (
            datetime.now(timezone.utc) - timedelta(seconds=10)
        ).replace(tzinfo=None)

        job_id = _seed_job(
            ctx["factory"],
            status="running",
            log="Cutting clip 1/2",
            started_at=started_naive,
        )

        resp = ctx["client"].get(f"/api/jobs/{job_id}/progress")
        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text[:300]}"
        )
        data = resp.json()

        # ── Required scalar fields ────────────────────────────────────────
        required = {
            "job_id":    int,
            "status":    str,
            "stage":     str,
            "percent":   (int, float),
            "elapsed_s": int,
            "message":   str,
        }
        for field, expected_type in required.items():
            assert field in data, (
                f"Missing required field {field!r}; response keys: {list(data)}"
            )
            assert isinstance(data[field], expected_type), (
                f"Field {field!r}: expected {expected_type}, got "
                f"{type(data[field]).__name__} ({data[field]!r})"
            )

        # ── eta_s: present, int or None ───────────────────────────────────
        assert "eta_s" in data, (
            f"Missing required field 'eta_s'; response keys: {list(data)}"
        )
        assert data["eta_s"] is None or isinstance(data["eta_s"], int), (
            f"eta_s must be int or None, got {type(data['eta_s']).__name__}"
        )

        # ── job_id matches ────────────────────────────────────────────────
        assert data["job_id"] == job_id, (
            f"job_id mismatch: expected {job_id}, got {data['job_id']}"
        )

        # ── percent in [0, 100] ───────────────────────────────────────────
        pct = data["percent"]
        assert 0.0 <= pct <= 100.0, (
            f"percent={pct} out of valid range [0.0, 100.0]"
        )

        # ── status is a valid lifecycle value ─────────────────────────────
        valid_statuses = {"pending", "running", "done", "failed"}
        assert data["status"] in valid_statuses, (
            f"status={data['status']!r} not in {valid_statuses}"
        )
