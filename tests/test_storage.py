"""
tests/test_storage.py
=====================
Phase 5 — Storage abstraction unit tests.

Coverage
--------
LocalStorage (6 tests):
  test_local_upload_creates_file_in_root_dir
  test_local_download_copies_out
  test_local_delete_removes
  test_local_exists_true_false
  test_local_get_url_is_media_path
  test_ensure_local_returns_existing_local_path_unchanged

R2Storage (10 tests, boto3 mocked):
  test_r2_upload_calls_upload_file_with_correct_args
  test_r2_upload_includes_content_type
  test_r2_download_invokes_download_file
  test_r2_delete_invokes_delete_object
  test_r2_exists_true_when_head_succeeds
  test_r2_exists_false_on_404_clienterror
  test_r2_get_url_signed_calls_generate_presigned_url
  test_r2_get_url_public_when_base_url_set
  test_r2_get_url_falls_back_to_signed_when_no_base_url
  test_r2_upload_returns_public_url_when_base_url_set

Factory (4 tests):
  test_get_storage_provider_local_default
  test_get_storage_provider_r2_env
  test_get_storage_provider_unknown_raises_valueerror
  test_get_storage_provider_cached_per_backend

Router integration (2 tests):
  test_render_beta_returns_storage_url_when_r2
  test_render_beta_local_backend_returns_media_path

Slow / integration (1 test, skipped by default):
  test_r2_round_trip_real_network
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch, call

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

def _make_r2_env(monkeypatch, *, public_base_url: str = "") -> None:
    """Set the minimum R2 env vars so R2Storage() can be constructed."""
    monkeypatch.setenv("R2_BUCKET", "test-bucket")
    monkeypatch.setenv("R2_ENDPOINT", "https://test.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "AKID1234567890abcdef")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "supersecretkey")
    monkeypatch.setenv("R2_PUBLIC_BASE_URL", public_base_url)


def _make_fake_client() -> MagicMock:
    """Return a MagicMock that looks enough like a boto3 S3 client."""
    client = MagicMock()
    client.generate_presigned_url.return_value = (
        "https://test.r2.cloudflarestorage.com/test-bucket/clips/1/master.mp4"
        "?X-Amz-Expires=3600&X-Amz-Signature=fakesig"
    )
    return client


# ---------------------------------------------------------------------------
# LocalStorage tests
# ---------------------------------------------------------------------------

class TestLocalStorage:

    def test_local_upload_creates_file_in_root_dir(self, tmp_path):
        """upload() copies the source file into root_dir/<key>."""
        from pipeline_core.storage import LocalStorage

        src = tmp_path / "source.mp4"
        src.write_bytes(b"fake-video-data")

        root = tmp_path / "storage_root"
        storage = LocalStorage(str(root))
        stored = storage.upload(str(src), "clips/1/master.mp4")

        dest = root / "clips" / "1" / "master.mp4"
        assert dest.exists(), f"Expected {dest} to exist after upload"
        assert dest.read_bytes() == b"fake-video-data"
        assert stored.key == "clips/1/master.mp4"
        assert stored.backend == "local"
        assert stored.size_bytes == len(b"fake-video-data")
        assert stored.url == "/media/clips/1/master.mp4"

    def test_local_download_copies_out(self, tmp_path):
        """download() copies the stored file to the requested local path."""
        from pipeline_core.storage import LocalStorage

        root = tmp_path / "storage_root"
        stored_file = root / "clips" / "2" / "master.mp4"
        stored_file.parent.mkdir(parents=True)
        stored_file.write_bytes(b"clip-content")

        storage = LocalStorage(str(root))
        dest = tmp_path / "downloads" / "out.mp4"
        result = storage.download("clips/2/master.mp4", str(dest))

        assert result == str(dest)
        assert dest.read_bytes() == b"clip-content"

    def test_local_delete_removes(self, tmp_path):
        """delete() removes the file from the root directory."""
        from pipeline_core.storage import LocalStorage

        root = tmp_path / "storage_root"
        stored_file = root / "clips" / "3" / "master.mp4"
        stored_file.parent.mkdir(parents=True)
        stored_file.write_bytes(b"data")

        storage = LocalStorage(str(root))
        assert storage.exists("clips/3/master.mp4")

        storage.delete("clips/3/master.mp4")
        assert not stored_file.exists()
        assert not storage.exists("clips/3/master.mp4")

    def test_local_exists_true_false(self, tmp_path):
        """exists() returns True for present keys, False for absent ones."""
        from pipeline_core.storage import LocalStorage

        root = tmp_path / "storage_root"
        present = root / "a" / "b.mp4"
        present.parent.mkdir(parents=True)
        present.write_bytes(b"x")

        storage = LocalStorage(str(root))
        assert storage.exists("a/b.mp4") is True
        assert storage.exists("a/missing.mp4") is False
        assert storage.exists("nonexistent/key.mp4") is False

    def test_local_get_url_is_media_path(self, tmp_path):
        """get_url() always returns /media/<key> regardless of signed flag."""
        from pipeline_core.storage import LocalStorage

        storage = LocalStorage(str(tmp_path / "root"))

        assert storage.get_url("clips/42/master.mp4") == "/media/clips/42/master.mp4"
        assert storage.get_url("clips/42/master.mp4", signed=True) == "/media/clips/42/master.mp4"
        assert storage.get_url("clips/42/master.mp4", signed=False) == "/media/clips/42/master.mp4"

    def test_ensure_local_returns_existing_local_path_unchanged(self, tmp_path):
        """ensure_local() returns the path as-is when the file exists on disk."""
        from pipeline_core.storage import LocalStorage

        root = tmp_path / "storage_root"
        storage = LocalStorage(str(root))

        existing = tmp_path / "existing_clip.mp4"
        existing.write_bytes(b"already here")

        result = storage.ensure_local(str(existing))
        assert result == str(existing)


# ---------------------------------------------------------------------------
# R2Storage tests (boto3 mocked)
# ---------------------------------------------------------------------------

class TestR2Storage:

    def _make_storage(self, monkeypatch, *, public_base_url: str = "") -> tuple:
        """Return (R2Storage instance, fake_boto3_client)."""
        _make_r2_env(monkeypatch, public_base_url=public_base_url)
        fake_client = _make_fake_client()

        from pipeline_core.storage import R2Storage
        storage = R2Storage()
        # Inject the fake client directly — bypasses lazy-init boto3.client call
        storage._client = fake_client
        return storage, fake_client

    def test_r2_upload_calls_upload_file_with_correct_args(self, tmp_path, monkeypatch):
        """upload() calls client.upload_file(local_path, bucket, key, ExtraArgs)."""
        storage, client = self._make_storage(monkeypatch)

        src = tmp_path / "clip.mp4"
        src.write_bytes(b"data")

        storage.upload(str(src), "clips/1/master.mp4", content_type="video/mp4")

        client.upload_file.assert_called_once_with(
            str(src),
            "test-bucket",
            "clips/1/master.mp4",
            ExtraArgs={"ContentType": "video/mp4"},
        )

    def test_r2_upload_includes_content_type(self, tmp_path, monkeypatch):
        """upload() guesses content_type from extension when not supplied."""
        storage, client = self._make_storage(monkeypatch)

        src = tmp_path / "thumb.jpg"
        src.write_bytes(b"jpeg")

        storage.upload(str(src), "thumbs/1.jpg")

        _, call_kwargs = client.upload_file.call_args
        extra = call_kwargs.get("ExtraArgs") or client.upload_file.call_args[1].get("ExtraArgs") or client.upload_file.call_args[0][3]
        # Verify ContentType was set to something image-related
        assert "image" in extra["ContentType"], (
            f"Expected image/* content type for .jpg, got {extra['ContentType']!r}"
        )

    def test_r2_download_invokes_download_file(self, tmp_path, monkeypatch):
        """download() calls client.download_file(bucket, key, local_path)."""
        storage, client = self._make_storage(monkeypatch)

        dest = tmp_path / "out.mp4"
        result = storage.download("clips/1/master.mp4", str(dest))

        client.download_file.assert_called_once_with(
            "test-bucket", "clips/1/master.mp4", str(dest)
        )
        assert result == str(dest)

    def test_r2_delete_invokes_delete_object(self, monkeypatch):
        """delete() calls client.delete_object(Bucket=..., Key=...)."""
        storage, client = self._make_storage(monkeypatch)

        storage.delete("clips/1/master.mp4")

        client.delete_object.assert_called_once_with(
            Bucket="test-bucket", Key="clips/1/master.mp4"
        )

    def test_r2_exists_true_when_head_succeeds(self, monkeypatch):
        """exists() returns True when head_object raises no exception."""
        storage, client = self._make_storage(monkeypatch)
        client.head_object.return_value = {"ContentLength": 1234}

        assert storage.exists("clips/1/master.mp4") is True
        client.head_object.assert_called_once_with(
            Bucket="test-bucket", Key="clips/1/master.mp4"
        )

    def test_r2_exists_false_on_404_clienterror(self, monkeypatch):
        """exists() returns False when head_object raises a 404 ClientError."""
        storage, client = self._make_storage(monkeypatch)

        from botocore.exceptions import ClientError
        error_response = {"Error": {"Code": "404", "Message": "Not Found"}}
        client.head_object.side_effect = ClientError(error_response, "HeadObject")

        assert storage.exists("clips/missing/master.mp4") is False

    def test_r2_get_url_signed_calls_generate_presigned_url(self, monkeypatch):
        """get_url(signed=True) calls generate_presigned_url with correct params."""
        storage, client = self._make_storage(monkeypatch)

        url = storage.get_url("clips/1/master.mp4", signed=True, expires_s=1800)

        client.generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={"Bucket": "test-bucket", "Key": "clips/1/master.mp4"},
            ExpiresIn=1800,
        )
        assert url == client.generate_presigned_url.return_value

    def test_r2_get_url_public_when_base_url_set(self, monkeypatch):
        """get_url(signed=False) returns public CDN URL when R2_PUBLIC_BASE_URL is set."""
        storage, client = self._make_storage(
            monkeypatch, public_base_url="https://cdn.example.com"
        )

        url = storage.get_url("clips/1/master.mp4", signed=False)

        assert url == "https://cdn.example.com/clips/1/master.mp4"
        client.generate_presigned_url.assert_not_called()

    def test_r2_get_url_falls_back_to_signed_when_no_base_url(self, monkeypatch):
        """get_url(signed=False) falls back to signed URL when R2_PUBLIC_BASE_URL is blank."""
        storage, client = self._make_storage(monkeypatch, public_base_url="")

        url = storage.get_url("clips/1/master.mp4", signed=False)

        client.generate_presigned_url.assert_called_once()
        assert url == client.generate_presigned_url.return_value

    def test_r2_upload_returns_public_url_when_base_url_set(self, tmp_path, monkeypatch):
        """upload() uses public CDN URL in StoredObject.url when R2_PUBLIC_BASE_URL is set."""
        storage, client = self._make_storage(
            monkeypatch, public_base_url="https://cdn.example.com"
        )

        src = tmp_path / "clip.mp4"
        src.write_bytes(b"video")

        stored = storage.upload(str(src), "clips/99/master.mp4", content_type="video/mp4")

        assert stored.url == "https://cdn.example.com/clips/99/master.mp4"
        # generate_presigned_url should NOT be called when public_base_url is set
        client.generate_presigned_url.assert_not_called()


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

class TestGetStorageProvider:

    def test_get_storage_provider_local_default(self, monkeypatch, tmp_path):
        """get_storage_provider() returns LocalStorage when STORAGE_BACKEND is unset."""
        monkeypatch.delenv("STORAGE_BACKEND", raising=False)
        # Clear cache so this test gets a fresh provider
        from pipeline_core import storage as storage_mod
        storage_mod._PROVIDER_CACHE.clear()

        from pipeline_core.storage import get_storage_provider, LocalStorage
        provider = get_storage_provider()
        assert isinstance(provider, LocalStorage)
        assert provider.name == "local"

    def test_get_storage_provider_r2_env(self, monkeypatch, tmp_path):
        """get_storage_provider() returns R2Storage when STORAGE_BACKEND=r2."""
        _make_r2_env(monkeypatch)
        monkeypatch.setenv("STORAGE_BACKEND", "r2")

        from pipeline_core import storage as storage_mod
        storage_mod._PROVIDER_CACHE.clear()

        from pipeline_core.storage import get_storage_provider, R2Storage
        provider = get_storage_provider()
        assert isinstance(provider, R2Storage)
        assert provider.name == "r2"

    def test_get_storage_provider_unknown_raises_valueerror(self, monkeypatch):
        """get_storage_provider() raises ValueError for unknown backend names."""
        monkeypatch.setenv("STORAGE_BACKEND", "azure_blob")

        from pipeline_core import storage as storage_mod
        storage_mod._PROVIDER_CACHE.clear()

        from pipeline_core.storage import get_storage_provider
        with pytest.raises(ValueError, match="Unknown storage backend"):
            get_storage_provider()

    def test_get_storage_provider_cached_per_backend(self, monkeypatch, tmp_path):
        """get_storage_provider() returns the same instance on repeated calls."""
        monkeypatch.delenv("STORAGE_BACKEND", raising=False)

        from pipeline_core import storage as storage_mod
        storage_mod._PROVIDER_CACHE.clear()

        from pipeline_core.storage import get_storage_provider
        p1 = get_storage_provider("local")
        p2 = get_storage_provider("local")
        assert p1 is p2, "Expected the same cached instance for the same backend"

    def test_get_storage_provider_explicit_backend_arg(self, monkeypatch, tmp_path):
        """get_storage_provider('local') works even when env says 'r2'."""
        _make_r2_env(monkeypatch)
        monkeypatch.setenv("STORAGE_BACKEND", "r2")

        from pipeline_core import storage as storage_mod
        storage_mod._PROVIDER_CACHE.clear()

        from pipeline_core.storage import get_storage_provider, LocalStorage
        provider = get_storage_provider("local")
        assert isinstance(provider, LocalStorage)


# ---------------------------------------------------------------------------
# Router integration tests
# ---------------------------------------------------------------------------

def _make_router_db(db_name: str):
    """Return (engine, sessionmaker) backed by a named shared-memory SQLite."""
    from database import Base
    import models  # noqa: F401
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    url = f"sqlite:///file:{db_name}?mode=memory&cache=shared&uri=true"
    engine = create_engine(
        url, connect_args={"check_same_thread": False, "uri": True}
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, factory


def _seed_router_clip(
    factory,
    *,
    clip_id: int = 1,
    file_path: str = "/tmp/master.mp4",
) -> int:
    import models
    from auth import ensure_legacy_user
    from sqlalchemy.orm import sessionmaker

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


@pytest.fixture()
def router_ctx(tmp_path, monkeypatch):
    """TestClient fixture with isolated DB, tmp beta renders dir, and auth bypass."""
    import main
    from database import get_db
    import auth
    import routers.editor as editor_mod
    from fastapi.testclient import TestClient

    db_name = f"kaizer_storage_test_{uuid.uuid4().hex}"
    engine, factory = _make_router_db(db_name)

    def _override_get_db() -> Generator:
        db = factory()
        try:
            yield db
        finally:
            db.close()

    from auth import ensure_legacy_user

    def _override_current_user():
        real_db = factory()
        try:
            return ensure_legacy_user(real_db)
        finally:
            real_db.close()

    monkeypatch.setattr(editor_mod, "BETA_RENDERS_ROOT", tmp_path / "beta_renders")

    main.app.dependency_overrides[get_db] = _override_get_db
    main.app.dependency_overrides[auth.current_user] = _override_current_user

    client = TestClient(main.app, raise_server_exceptions=False)

    yield {"client": client, "factory": factory, "tmp_path": tmp_path}

    main.app.dependency_overrides.pop(get_db, None)
    main.app.dependency_overrides.pop(auth.current_user, None)
    engine.dispose()


class TestRenderBetaStorageIntegration:

    def test_render_beta_returns_storage_url_when_r2(
        self, router_ctx, monkeypatch, tmp_path
    ):
        """When STORAGE_BACKEND=r2, the response beta_url is the StoredObject.url."""
        from pipeline_core.editor_pro import BetaRenderResult
        from pipeline_core.storage import StoredObject

        # Create a real temp file for the beta_path so os.path.isfile() passes
        fake_beta = tmp_path / "beta.mp4"
        fake_beta.write_bytes(b"fake-mp4")

        fake_result = BetaRenderResult(
            current_path="/tmp/master.mp4",
            beta_path=str(fake_beta),
            style_pack="cinematic",
            effects_applied=["color_grade:cinematic_warm"],
            render_time_s=0.5,
            qa_ok=True,
            warnings=[],
        )

        fake_stored = StoredObject(
            key="beta_renders/clip_10/cinematic.mp4",
            url="https://r2.example.com/beta_renders/clip_10/cinematic.mp4",
            backend="r2",
            size_bytes=1024,
            etag="abc123",
        )

        monkeypatch.setenv("STORAGE_BACKEND", "r2")

        _seed_router_clip(router_ctx["factory"], clip_id=10, file_path="/tmp/master.mp4")

        with patch("routers.editor.render_beta", return_value=fake_result), \
             patch("routers.editor.get_storage_provider") as mock_provider_factory:

            mock_provider = MagicMock()
            mock_provider.name = "r2"
            mock_provider.upload.return_value = fake_stored
            mock_provider_factory.return_value = mock_provider

            resp = router_ctx["client"].post(
                "/api/editor/render-beta",
                json={"clip_id": 10, "style_pack": "cinematic"},
            )

        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text[:300]}"
        data = resp.json()
        assert data["beta_url"] == fake_stored.url, (
            f"Expected R2 URL {fake_stored.url!r}, got {data['beta_url']!r}"
        )

    def test_render_beta_local_backend_returns_media_path(
        self, router_ctx, monkeypatch, tmp_path
    ):
        """When STORAGE_BACKEND=local, beta_url is a /media/ path (unchanged behavior)."""
        from pipeline_core.editor_pro import BetaRenderResult

        import routers.editor as editor_mod
        beta_renders = tmp_path / "beta_renders"
        monkeypatch.setattr(editor_mod, "BETA_RENDERS_ROOT", beta_renders)

        # Create a fake output dir that the path_to_url helper can relativise
        clip_dir = beta_renders / "clip_20"
        clip_dir.mkdir(parents=True, exist_ok=True)
        fake_beta = clip_dir / "master_beta_calm.mp4"
        fake_beta.write_bytes(b"data")

        fake_result = BetaRenderResult(
            current_path="/tmp/master.mp4",
            beta_path=str(fake_beta),
            style_pack="calm",
            effects_applied=[],
            render_time_s=0.1,
            qa_ok=True,
            warnings=[],
        )

        monkeypatch.setenv("STORAGE_BACKEND", "local")
        from pipeline_core import storage as storage_mod
        storage_mod._PROVIDER_CACHE.clear()

        _seed_router_clip(router_ctx["factory"], clip_id=20, file_path="/tmp/master.mp4")

        with patch("routers.editor.render_beta", return_value=fake_result):
            # Also patch OUTPUT_DIR so _path_to_url can relativise the beta path
            monkeypatch.setattr(editor_mod, "OUTPUT_DIR", beta_renders)

            resp = router_ctx["client"].post(
                "/api/editor/render-beta",
                json={"clip_id": 20, "style_pack": "calm"},
            )

        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text[:300]}"
        data = resp.json()
        beta_url = data["beta_url"]
        assert beta_url.startswith("/media/") or beta_url.startswith("/api/file/"), (
            f"Expected /media/ or /api/file/ URL for local backend, got {beta_url!r}"
        )


# ---------------------------------------------------------------------------
# Slow / real-network integration test
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("R2_INTEGRATION_TEST", "0") != "1",
    reason="Real R2 network test — set R2_INTEGRATION_TEST=1 to run",
)
def test_r2_round_trip_real_network(tmp_path):
    """PUT / GET / DELETE against the real R2 bucket.

    Skipped unless R2_INTEGRATION_TEST=1 is set in the environment.
    Credentials are read from the environment / .env file — the same ones
    used by the production server.

    Steps:
      1. Upload a small test file.
      2. Check exists() returns True.
      3. Download it and verify contents match.
      4. Delete it and confirm exists() returns False.
    """
    from pipeline_core.storage import R2Storage

    storage = R2Storage()

    # Use a unique key so parallel test runs don't collide.
    unique_key = f"_ci_test/{uuid.uuid4().hex}/round_trip.txt"
    content = b"kaizer-r2-round-trip-test"

    src = tmp_path / "upload.txt"
    src.write_bytes(content)

    # Upload
    stored = storage.upload(str(src), unique_key, content_type="text/plain")
    assert stored.key == unique_key
    assert stored.backend == "r2"

    try:
        # Exists
        assert storage.exists(unique_key), "Key should exist right after upload"

        # Download
        dest = tmp_path / "download.txt"
        storage.download(unique_key, str(dest))
        assert dest.read_bytes() == content, "Downloaded content must match uploaded content"

        # Signed URL (smoke — just check it looks like a URL)
        url = storage.get_url(unique_key, signed=True, expires_s=60)
        assert url.startswith("https://"), f"Expected HTTPS URL, got {url!r}"

    finally:
        # Always clean up, even on assertion failures
        storage.delete(unique_key)
        assert not storage.exists(unique_key), "Key should be gone after delete"
