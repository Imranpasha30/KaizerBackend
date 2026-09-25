"""Unit 6 — editor 'Sync images to speech' endpoint + canvas round-trip."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import models
from database import Base
from pipeline_v4.canvas_schema import V4JobCanvas


def _canvas_doc(job_id: int, out_dir: Path) -> dict:
    return {
        "job_id": job_id,
        "language": "te",
        "trimmed_bulletin_path": str(out_dir / "trimmed_bulletin.mp4"),
        "bulletin": {
            "kind": "bulletin",
            "output_filename": "bulletin.mp4",
            "layout": {},
            "trimmed_video_path": str(out_dir / "trimmed_bulletin.mp4"),
            "stories": [{
                "story_index": 0,
                "video_t_start": 0.0,
                "video_t_end": 20.0,
                "title_native": "టెస్ట్",
                "title_english": "test",
                "summary": "s",
                "images": [
                    {"src": "a.jpg", "t_start": 0.0, "t_end": 4.0,
                     "label": "Modi parliament"},
                    {"src": "b.jpg", "t_start": 4.0, "t_end": 8.0,
                     "label": "Hyderabad floods"},
                    # Operator-pinned window — must survive verbatim.
                    {"src": "c.jpg", "t_start": 15.0, "t_end": 18.0,
                     "label": "pinned chart", "timing_mode": "pinned"},
                ],
            }],
        },
    }


@pytest.fixture
def app_env(monkeypatch, tmp_path):
    engine = create_engine(
        "sqlite:///file:kaizer_resync_test?mode=memory&cache=shared&uri=true",
        future=True, connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    from main import app
    import auth
    from routers import v4_editor as v4r

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    out_dir = tmp_path / "job42"
    out_dir.mkdir()

    db = TestingSessionLocal()
    db.add(models.User(id=1, email="t@x.com", name="t"))
    db.add(models.Job(id=42, user_id=1, status="done",
                      platform="full_video_shorts_v4", video_name="v",
                      output_dir=str(out_dir)))
    db.commit(); db.close()

    (out_dir / "canvas.json").write_text(
        json.dumps(_canvas_doc(42, out_dir), ensure_ascii=False), encoding="utf-8")

    app.dependency_overrides[v4r.get_db] = override_get_db
    u = models.User(); u.id = 1; u.email = "t@x.com"
    app.dependency_overrides[auth.current_user] = lambda: u
    try:
        yield app, out_dir
    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(engine)


def test_resync_409_without_word_sidecar(app_env):
    app, _ = app_env
    r = TestClient(app).post("/api/v4/jobs/42/bulletin/resync-timings", json={})
    assert r.status_code == 409


def test_resync_replaces_content_windows_keeps_pinned(app_env, monkeypatch):
    app, out_dir = app_env
    (out_dir / "story_words.json").write_text(json.dumps({
        "schema": 1, "language": "te",
        "stories": {"0": [{"w": "మోదీ", "s": 1.0, "e": 1.4},
                          {"w": "వరదలు", "s": 9.0, "e": 9.6}]},
    }), encoding="utf-8")

    captured: dict = {}

    def fake_decide(**kw):
        captured.update(kw)
        return [
            {"pool_index": 0, "t_start": 0.5, "t_end": 4.0,
             "confidence": 0.9, "importance": 0.8, "matched_text": "మోదీ"},
            {"pool_index": 1, "t_start": 8.5, "t_end": 12.0,
             "confidence": 0.7, "importance": 0.3, "matched_text": "వరదలు"},
        ]

    from pipeline_v4 import image_timing
    monkeypatch.setattr(image_timing, "decide_story_timings", fake_decide)

    r = TestClient(app).post("/api/v4/jobs/42/bulletin/resync-timings", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["stories_resynced"] == [0]
    assert "0" in body["images_by_story"]

    # The engine received the pinned reservation + the deduped content pool.
    assert captured["pinned_windows"] == ((15.0, 18.0),)
    assert [p["label"] for p in captured["pool"]] == ["Modi parliament", "Hyderabad floods"]

    # Canvas on disk: pinned image untouched; content windows replaced.
    saved = V4JobCanvas.model_validate(
        json.loads((out_dir / "canvas.json").read_text(encoding="utf-8")))
    imgs = saved.bulletin.stories[0].images
    assert len(imgs) == 3
    pinned = [i for i in imgs if i.timing_mode == "pinned"]
    assert len(pinned) == 1
    assert (pinned[0].src, pinned[0].t_start, pinned[0].t_end) == ("c.jpg", 15.0, 18.0)
    by_src = {i.src: i for i in imgs}
    assert (by_src["a.jpg"].t_start, by_src["a.jpg"].t_end) == (0.5, 4.0)
    assert by_src["a.jpg"].confidence == 0.9
    assert by_src["a.jpg"].matched_text == "మోదీ"
    assert (by_src["b.jpg"].t_start, by_src["b.jpg"].t_end) == (8.5, 12.0)
    # sorted by t_start
    assert [i.t_start for i in imgs] == sorted(i.t_start for i in imgs)


def test_resync_engine_none_leaves_canvas_untouched(app_env, monkeypatch):
    app, out_dir = app_env
    (out_dir / "story_words.json").write_text(json.dumps({
        "schema": 1, "stories": {"0": [{"w": "x", "s": 1.0, "e": 1.2}]},
    }), encoding="utf-8")
    from pipeline_v4 import image_timing
    monkeypatch.setattr(image_timing, "decide_story_timings", lambda **kw: None)
    before = (out_dir / "canvas.json").read_text(encoding="utf-8")
    r = TestClient(app).post("/api/v4/jobs/42/bulletin/resync-timings", json={})
    assert r.status_code == 200
    assert r.json()["stories_resynced"] == []
    assert (out_dir / "canvas.json").read_text(encoding="utf-8") == before


def test_canvas_roundtrip_preserves_sync_fields(app_env):
    """PUT canvas → GET canvas keeps the Phase-1 image fields intact."""
    app, out_dir = app_env
    doc = _canvas_doc(42, out_dir)
    img0 = doc["bulletin"]["stories"][0]["images"][0]
    img0.update({"timing_mode": "pinned", "confidence": 0.83,
                 "importance": 0.9, "spotlight": "fullscreen",
                 "matched_text": "మోదీ పార్లమెంట్"})
    client = TestClient(app)
    r = client.put("/api/v4/jobs/42/canvas", json={"canvas": doc})
    assert r.status_code == 200, r.text
    saved = json.loads((out_dir / "canvas.json").read_text(encoding="utf-8"))
    got = saved["bulletin"]["stories"][0]["images"][0]
    assert got["timing_mode"] == "pinned"
    assert got["confidence"] == 0.83
    assert got["spotlight"] == "fullscreen"
    assert got["matched_text"] == "మోదీ పార్లమెంట్"
