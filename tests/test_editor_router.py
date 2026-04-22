"""
tests/test_editor_router.py
===========================
Wave 2A — 10-test suite for routers/editor.py.

Coverage
--------
  1.  test_get_styles_returns_five_packs
  2.  test_get_styles_shape_correct
  3.  test_render_beta_nonexistent_clip_404
  4.  test_render_beta_invalid_style_pack_400
  5.  test_render_beta_happy_path_returns_paths_and_urls
  6.  test_render_beta_qa_failure_surfaces_warnings
  7.  test_get_latest_render_returns_404_when_none_cached
  8.  test_get_latest_render_returns_cached_json_when_present
  9.  test_render_beta_respects_platform_param
  10. test_render_beta_hook_text_forwarded

Strategy
--------
- In-memory SQLite using shared-cache URIs (same pattern as test_progress_api).
- `render_beta` is always mocked — no real ffmpeg is invoked.
- The `latest.json` cache is written to a tmp directory via monkeypatching of
  `routers.editor.BETA_RENDERS_ROOT`.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# ---------------------------------------------------------------------------
# Ensure backend root is on sys.path
# ---------------------------------------------------------------------------

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(db_name: str):
    """Return (engine, sessionmaker) backed by a named shared-memory SQLite."""
    from database import Base
    import models  # noqa: F401 — registers ORM classes

    url = f"sqlite:///file:{db_name}?mode=memory&cache=shared&uri=true"
    engine = create_engine(
        url,
        connect_args={"check_same_thread": False, "uri": True},
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, factory


def _seed_clip(factory, *, clip_id: int = 1, file_path: str = "/tmp/master.mp4") -> int:
    """Insert a minimal User + Job + Clip row.  Returns clip_id."""
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
            status="done",
            output_dir="/tmp",
        )
        db.add(job)
        db.flush()
        clip = models.Clip(
            id=clip_id,
            job_id=job.id,
            clip_index=0,
            file_path=file_path,
            duration=30.0,
        )
        db.add(clip)
        db.commit()
        return clip_id
    finally:
        db.close()


def _fake_result(
    *,
    current_path: str = "/tmp/master.mp4",
    beta_path: str = "/tmp/master_beta_cinematic.mp4",
    style_pack: str = "cinematic",
    effects_applied: list[str] | None = None,
    render_time_s: float = 1.23,
    qa_ok: bool = True,
    warnings: list[str] | None = None,
):
    """Build a BetaRenderResult without importing from pipeline_core."""
    from pipeline_core.editor_pro import BetaRenderResult
    return BetaRenderResult(
        current_path=current_path,
        beta_path=beta_path,
        style_pack=style_pack,
        effects_applied=effects_applied or ["color_grade:cinematic_warm"],
        render_time_s=render_time_s,
        qa_ok=qa_ok,
        warnings=warnings or [],
    )


# ---------------------------------------------------------------------------
# Per-test fixture: unique in-memory DB + TestClient + tmp beta_renders dir
# ---------------------------------------------------------------------------

@pytest.fixture()
def ctx(request, tmp_path, monkeypatch):
    """Yield dict with keys: client, factory.

    - Unique in-memory SQLite per test (no cross-test contamination).
    - BETA_RENDERS_ROOT is redirected to a per-test tmp_path subdirectory so
      latest.json writes/reads never touch the real output tree.
    - auth.current_user is overridden to return the legacy user (no token
      needed in tests).
    """
    import main
    from database import get_db
    import auth
    import routers.editor as editor_mod

    db_name = f"kaizer_editor_test_{uuid.uuid4().hex}"
    engine, factory = _make_db(db_name)

    # Override DB dependency
    def _override_get_db() -> Generator:
        db = factory()
        try:
            yield db
        finally:
            db.close()

    # Override auth so tests don't need Bearer tokens.
    # IMPORTANT: the override must be a zero-argument callable — FastAPI
    # introspects the signature and tries to inject any named params as
    # query/header dependencies, which breaks non-trivial signatures.
    from auth import ensure_legacy_user

    def _override_current_user():
        real_db = factory()
        try:
            return ensure_legacy_user(real_db)
        finally:
            real_db.close()

    # Point BETA_RENDERS_ROOT at tmp_path so disk writes stay isolated
    monkeypatch.setattr(editor_mod, "BETA_RENDERS_ROOT", tmp_path / "beta_renders")

    main.app.dependency_overrides[get_db] = _override_get_db
    main.app.dependency_overrides[auth.current_user] = _override_current_user

    client = TestClient(main.app, raise_server_exceptions=False)

    yield {"client": client, "factory": factory, "tmp_path": tmp_path}

    main.app.dependency_overrides.pop(get_db, None)
    main.app.dependency_overrides.pop(auth.current_user, None)
    engine.dispose()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEditorStyles:

    def test_get_styles_returns_five_packs(self, ctx):
        """GET /api/editor/styles → 200 with exactly 5 items."""
        resp = ctx["client"].get("/api/editor/styles")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        assert isinstance(data, list), "Response should be a list"
        assert len(data) == 5, f"Expected 5 style packs, got {len(data)}"

    def test_get_styles_shape_correct(self, ctx):
        """Each style pack must have all required fields with string values."""
        resp = ctx["client"].get("/api/editor/styles")
        assert resp.status_code == 200
        required_fields = {
            "name", "label", "description", "transition",
            "color_preset", "text_animation", "caption_animation",
        }
        for pack in resp.json():
            for f in required_fields:
                assert f in pack, f"Style pack missing field {f!r}: {pack}"
                assert isinstance(pack[f], str), (
                    f"Field {f!r} should be str, got {type(pack[f])}: {pack[f]!r}"
                )
            # motion is str | None — just assert key present
            assert "motion" in pack, f"Style pack missing 'motion' key: {pack}"


class TestRenderBetaPost:

    def test_render_beta_nonexistent_clip_404(self, ctx):
        """POST /api/editor/render-beta with clip_id that does not exist → 404."""
        with patch("routers.editor.render_beta") as mock_rb:
            resp = ctx["client"].post(
                "/api/editor/render-beta",
                json={"clip_id": 99999, "style_pack": "cinematic"},
            )
        assert resp.status_code == 404, (
            f"Expected 404 for non-existent clip, got {resp.status_code}: {resp.text[:200]}"
        )
        assert "detail" in resp.json()

    def test_render_beta_invalid_style_pack_400(self, ctx):
        """POST with unknown style_pack → 400 before any render is called."""
        _seed_clip(ctx["factory"], clip_id=1)
        with patch("routers.editor.render_beta") as mock_rb:
            resp = ctx["client"].post(
                "/api/editor/render-beta",
                json={"clip_id": 1, "style_pack": "nonexistent_pack_xyz"},
            )
            mock_rb.assert_not_called()
        assert resp.status_code == 400, (
            f"Expected 400 for invalid style_pack, got {resp.status_code}: {resp.text[:200]}"
        )
        detail = resp.json()["detail"]
        assert "nonexistent_pack_xyz" in detail or "Unknown" in detail, (
            f"400 detail should mention the bad pack name: {detail!r}"
        )

    def test_render_beta_happy_path_returns_paths_and_urls(self, ctx):
        """Seed a clip, mock render_beta, assert full response shape."""
        _seed_clip(ctx["factory"], clip_id=2, file_path="/tmp/clip2.mp4")
        fake = _fake_result(
            current_path="/tmp/clip2.mp4",
            beta_path="/fake/output/beta_renders/clip_2/clip2_beta_cinematic.mp4",
        )

        with patch("routers.editor.render_beta", return_value=fake) as mock_rb:
            resp = ctx["client"].post(
                "/api/editor/render-beta",
                json={"clip_id": 2, "style_pack": "cinematic"},
            )

        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text[:300]}"
        )
        data = resp.json()
        required = {
            "clip_id", "current_path", "current_url", "beta_path", "beta_url",
            "style_pack", "effects_applied", "render_time_s", "qa_ok", "warnings",
        }
        for key in required:
            assert key in data, f"Response missing field {key!r}"

        assert data["clip_id"] == 2
        assert data["style_pack"] == "cinematic"
        assert isinstance(data["effects_applied"], list)
        assert isinstance(data["qa_ok"], bool)
        assert isinstance(data["warnings"], list)
        assert data["render_time_s"] == pytest.approx(1.23, abs=0.01)
        # URLs must start with /media/ or /api/file/
        assert data["current_url"].startswith("/") or data["current_url"].startswith("http")
        assert data["beta_url"].startswith("/") or data["beta_url"].startswith("http")

    def test_render_beta_qa_failure_surfaces_warnings(self, ctx):
        """Mock render_beta to return qa_ok=False with warnings; response carries them."""
        _seed_clip(ctx["factory"], clip_id=3, file_path="/tmp/clip3.mp4")
        fake = _fake_result(
            current_path="/tmp/clip3.mp4",
            beta_path="/fake/output/beta_renders/clip_3/clip3_beta_cinematic.mp4",
            qa_ok=False,
            warnings=["QA error: resolution too low", "QA error: bitrate under threshold"],
        )

        with patch("routers.editor.render_beta", return_value=fake):
            resp = ctx["client"].post(
                "/api/editor/render-beta",
                json={"clip_id": 3, "style_pack": "cinematic"},
            )

        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        assert data["qa_ok"] is False, "qa_ok should be False"
        assert len(data["warnings"]) == 2, f"Expected 2 warnings, got {data['warnings']}"
        assert "QA error: resolution too low" in data["warnings"]

    def test_render_beta_respects_platform_param(self, ctx):
        """platform='instagram_reel' must be forwarded to render_beta."""
        _seed_clip(ctx["factory"], clip_id=4, file_path="/tmp/clip4.mp4")
        fake = _fake_result(
            current_path="/tmp/clip4.mp4",
            beta_path="/fake/output/beta_renders/clip_4/clip4_beta_vibrant.mp4",
            style_pack="vibrant",
        )

        with patch("routers.editor.render_beta", return_value=fake) as mock_rb:
            resp = ctx["client"].post(
                "/api/editor/render-beta",
                json={
                    "clip_id": 4,
                    "style_pack": "vibrant",
                    "platform": "instagram_reel",
                },
            )

        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text[:200]}"
        call_kwargs = mock_rb.call_args
        # render_beta is called as render_beta(master_path, style_pack=..., platform=..., ...)
        assert call_kwargs is not None, "render_beta should have been called"
        kwargs = call_kwargs.kwargs
        assert kwargs.get("platform") == "instagram_reel", (
            f"Expected platform='instagram_reel', got {kwargs.get('platform')!r}"
        )

    def test_render_beta_hook_text_forwarded(self, ctx):
        """hook_text='BREAKING' must be forwarded as a kwarg to render_beta."""
        _seed_clip(ctx["factory"], clip_id=5, file_path="/tmp/clip5.mp4")
        fake = _fake_result(
            current_path="/tmp/clip5.mp4",
            beta_path="/fake/output/beta_renders/clip_5/clip5_beta_news_flash.mp4",
            style_pack="news_flash",
        )

        with patch("routers.editor.render_beta", return_value=fake) as mock_rb:
            resp = ctx["client"].post(
                "/api/editor/render-beta",
                json={
                    "clip_id": 5,
                    "style_pack": "news_flash",
                    "hook_text": "BREAKING",
                },
            )

        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text[:200]}"
        call_kwargs = mock_rb.call_args
        assert call_kwargs is not None, "render_beta should have been called"
        kwargs = call_kwargs.kwargs
        assert kwargs.get("hook_text") == "BREAKING", (
            f"Expected hook_text='BREAKING', got {kwargs.get('hook_text')!r}"
        )


class TestGetLatestRender:

    def test_get_latest_render_returns_404_when_none_cached(self, ctx):
        """GET /api/editor/render-beta/{clip_id} with no prior render → 404."""
        resp = ctx["client"].get("/api/editor/render-beta/88888")
        assert resp.status_code == 404, (
            f"Expected 404 when no render cached, got {resp.status_code}: {resp.text[:200]}"
        )
        assert "detail" in resp.json()

    def test_get_latest_render_returns_cached_json_when_present(self, ctx, monkeypatch):
        """Seed latest.json manually → GET returns its content as RenderBetaResponse."""
        import routers.editor as editor_mod

        clip_id = 77
        beta_renders_root: Path = ctx["tmp_path"] / "beta_renders"
        # Align with the monkeypatched BETA_RENDERS_ROOT
        monkeypatch.setattr(editor_mod, "BETA_RENDERS_ROOT", beta_renders_root)

        clip_dir = beta_renders_root / f"clip_{clip_id}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        latest_json = clip_dir / "latest.json"
        payload = {
            "style_pack": "calm",
            "beta_path": f"/fake/output/beta_renders/clip_{clip_id}/master_beta_calm.mp4",
            "current_path": "/fake/output/raw/master.mp4",
            "effects_applied": ["color_grade:cool_blue", "motion:parallax_still"],
            "qa_ok": True,
            "warnings": [],
            "rendered_at": "2026-04-22T10:00:00+00:00",
        }
        latest_json.write_text(json.dumps(payload), encoding="utf-8")

        resp = ctx["client"].get(f"/api/editor/render-beta/{clip_id}")
        assert resp.status_code == 200, (
            f"Expected 200 for seeded latest.json, got {resp.status_code}: {resp.text[:300]}"
        )
        data = resp.json()
        assert data["clip_id"] == clip_id
        assert data["style_pack"] == "calm"
        assert data["qa_ok"] is True
        assert "color_grade:cool_blue" in data["effects_applied"]
        assert data["warnings"] == []
        # render_time_s is 0.0 for cached responses
        assert data["render_time_s"] == 0.0
