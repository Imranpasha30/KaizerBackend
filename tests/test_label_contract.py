"""Tests for the image label contract ("name-tag contract") — Unit 1.

Every image entering a video must be able to carry a subject label:
  - user uploads: optional `description` form field on /api/assets/upload
  - post-hoc edit: PATCH /api/assets/{id} with {"description": ...}
  - optional vision auto-caption for unlabeled uploads
    (KAIZER_AUTO_CAPTION_UPLOADS=1, background task, fail-soft)
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

import models
from database import Base


PNG_1PX = (  # smallest valid PNG (1x1 transparent)
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xfa\xcf"
    b"\xc0\xf0\x1f\x00\x05\x05\x02\x00_\xc8\xf1\xd2\x00\x00\x00\x00IEND"
    b"\xaeB`\x82"
)


@pytest.fixture
def test_app(monkeypatch, tmp_path):
    """FastAPI app on an in-memory SQLite DB with auth + storage stubbed."""
    engine = create_engine(
        "sqlite:///file:kaizer_label_test?mode=memory&cache=shared&uri=true",
        future=True,
        # async endpoints run on the event-loop thread, sync deps on the
        # TestClient thread pool — SQLite must allow cross-thread use here
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    from main import app
    import auth
    import database
    from routers import assets as assets_router

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    # Seed the fake logged-in user
    db = TestingSessionLocal()
    user = models.User(id=1, email="t@x.com", name="t")
    db.add(user)
    db.commit()
    db.close()

    app.dependency_overrides[assets_router.get_db] = override_get_db
    app.dependency_overrides[auth.current_user] = lambda: _fake_user()

    # Keep test files out of the real assets tree; skip real storage backends.
    monkeypatch.setattr(assets_router, "ASSETS_ROOT", tmp_path)
    monkeypatch.setattr(
        assets_router, "get_storage_provider",
        lambda: (_ for _ in ()).throw(RuntimeError("no storage in tests")),
    )
    # Background caption task opens its own session — point it at the test DB.
    monkeypatch.setattr(database, "SessionLocal", TestingSessionLocal)

    try:
        yield app, TestingSessionLocal
    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(engine)


def _fake_user():
    u = models.User()
    u.id = 1
    u.email = "t@x.com"
    return u


def _upload(client, *, description=None):
    data = {"kind": "image"}
    if description is not None:
        data["description"] = description
    return client.post(
        "/api/assets/upload",
        files={"file": ("pic.png", io.BytesIO(PNG_1PX), "image/png")},
        data=data,
    )


def test_upload_with_description_persists_label(test_app):
    app, Session = test_app
    client = TestClient(app)
    r = _upload(client, description="  CM at press meet  ")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["description"] == "CM at press meet"
    db = Session()
    row = db.query(models.UserAsset).filter_by(id=body["id"]).one()
    assert row.description == "CM at press meet"
    db.close()


def test_upload_without_description_defaults_empty(test_app, monkeypatch):
    app, _ = test_app
    monkeypatch.delenv("KAIZER_AUTO_CAPTION_UPLOADS", raising=False)
    client = TestClient(app)
    r = _upload(client)
    assert r.status_code == 201, r.text
    assert r.json()["description"] == ""


def test_patch_description_updates_label(test_app):
    app, _ = test_app
    client = TestClient(app)
    aid = _upload(client).json()["id"]
    r = client.patch(f"/api/assets/{aid}", json={"description": "flooded street"})
    assert r.status_code == 200, r.text
    assert r.json()["description"] == "flooded street"


def test_auto_caption_fills_empty_label(test_app, monkeypatch):
    """Env on + no description → background vision caption lands in the row."""
    app, Session = test_app
    monkeypatch.setenv("KAIZER_AUTO_CAPTION_UPLOADS", "1")
    from pipeline_v4 import image_ai
    monkeypatch.setattr(image_ai, "caption_image",
                        lambda path, mime="image/jpeg": "politician at podium")
    client = TestClient(app)  # runs background tasks after the response
    r = _upload(client)
    assert r.status_code == 201, r.text
    db = Session()
    row = db.query(models.UserAsset).filter_by(id=r.json()["id"]).one()
    assert row.description == "politician at podium"
    db.close()


def test_auto_caption_never_overwrites_user_label(test_app, monkeypatch):
    app, Session = test_app
    monkeypatch.setenv("KAIZER_AUTO_CAPTION_UPLOADS", "1")
    from pipeline_v4 import image_ai
    monkeypatch.setattr(image_ai, "caption_image",
                        lambda path, mime="image/jpeg": "WRONG")
    client = TestClient(app)
    r = _upload(client, description="my own label")
    db = Session()
    row = db.query(models.UserAsset).filter_by(id=r.json()["id"]).one()
    assert row.description == "my own label"
    db.close()


def test_caption_image_fail_soft(monkeypatch):
    """caption_image returns '' (never raises) on unreadable input."""
    from pipeline_v4.image_ai import caption_image
    assert caption_image("Z:/nope/missing.jpg") == ""


def test_migration_idempotent_and_column_present():
    """_migrate_schema ran at import; a second run must be a no-op, and the
    user_assets.description column must exist in the live DEV schema."""
    import main as main_mod
    from database import engine
    main_mod._migrate_schema()  # second run — idempotent, must not raise
    cols = {c["name"] for c in inspect(engine).get_columns("user_assets")}
    assert "description" in cols


# ─── Unit 2: AI-image beat labels + pool label ingestion ─────────────


def test_parse_beat_list_object_shape():
    from pipeline_v4.image_ai import _parse_beat_list
    raw = ('```json\n[{"prompt": "wide shot of parliament at dusk", '
           '"label": "parliament at dusk"},\n'
           '{"prompt": "close-up of ballot box"}]\n```')
    beats = _parse_beat_list(raw)
    assert len(beats) == 2
    assert beats[0] == {"prompt": "wide shot of parliament at dusk",
                        "label": "parliament at dusk"}
    # missing label → derived from the prompt, never empty
    assert beats[1]["prompt"] == "close-up of ballot box"
    assert beats[1]["label"]


def test_parse_beat_list_plain_strings_backcompat():
    from pipeline_v4.image_ai import _parse_beat_list, _parse_prompt_list
    raw = '["prompt one about floods", "prompt two about rescue crews"]'
    beats = _parse_beat_list(raw)
    assert [b["prompt"] for b in beats] == ["prompt one about floods",
                                            "prompt two about rescue crews"]
    assert all(b["label"] for b in beats)
    # the old shim still returns plain prompts
    assert _parse_prompt_list(raw) == [b["prompt"] for b in beats]


def test_derive_label_strips_stock_preamble():
    from pipeline_v4.image_ai import derive_label
    lab = derive_label(
        "Cinematic 16:9 news B-roll photograph about: Hyderabad floods. "
        "Documentary style, natural lighting."
    )
    assert lab == "Hyderabad floods"
    # ≤8 words cap
    long = derive_label("one two three four five six seven eight nine ten")
    assert len(long.split()) <= 8
    assert derive_label("") == ""


def test_ingest_image_pool_reads_label_env(tmp_path, monkeypatch):
    """KAIZER_BULLETIN_IMAGE_LABELS_B64 (b64 JSON path→label) stamps real
    subject labels onto pool entries; unmapped files keep the stem."""
    import base64 as b64
    import json as js
    from pipeline_v4 import orchestrator as orch

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    img_a = src_dir / "photo_123.jpg"
    img_b = src_dir / "other.jpg"
    img_a.write_bytes(b"x" * 10)
    img_b.write_bytes(b"y" * 10)

    labels = {str(img_a): "సీఎం ప్రెస్ మీట్ (CM press meet)"}
    monkeypatch.setenv(
        "KAIZER_BULLETIN_IMAGE_LABELS_B64",
        b64.b64encode(js.dumps(labels, ensure_ascii=False).encode("utf-8")).decode(),
    )
    out_dir = tmp_path / "job"
    out_dir.mkdir()
    pool = orch._ingest_image_pool(out_dir, f"{img_a}|{img_b}")
    by_name = {p["filename"]: p for p in pool}
    assert by_name["photo_123.jpg"]["label"] == "సీఎం ప్రెస్ మీట్ (CM press meet)"
    assert by_name["other.jpg"]["label"] == "other"   # stem fallback


def test_ingest_image_pool_bad_label_env_falls_back(tmp_path, monkeypatch):
    from pipeline_v4 import orchestrator as orch
    src = tmp_path / "pic.jpg"
    src.write_bytes(b"z" * 10)
    monkeypatch.setenv("KAIZER_BULLETIN_IMAGE_LABELS_B64", "!!!not-base64!!!")
    out_dir = tmp_path / "job2"
    out_dir.mkdir()
    pool = orch._ingest_image_pool(out_dir, str(src))
    assert pool[0]["label"] == "pic"
