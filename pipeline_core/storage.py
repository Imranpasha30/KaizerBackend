"""
kaizer.pipeline.storage
========================
Phase 5 — Cloud storage abstraction layer.

Provides a uniform interface over Cloudflare R2 (S3-compatible) and a
local-disk fallback.  Callers import ``get_storage_provider()`` and never
touch the concrete classes directly.

Usage
-----
    from pipeline_core.storage import get_storage_provider, StoredObject

    provider = get_storage_provider()          # respects STORAGE_BACKEND env
    stored = provider.upload("/tmp/clip.mp4", "clips/42/master.mp4",
                              content_type="video/mp4")
    print(stored.url)                          # fetchable URL

Rules
-----
- Never falls back silently from R2 to local.  If R2 is configured and a
  call fails, the exception is logged loudly and re-raised.
- The boto3 client is lazy-initialised so import time is unaffected.
- Never logs the secret access key.  Only the access key ID (truncated) is
  included in log output.
- Forward slashes are used in all storage keys regardless of OS.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import shutil
import tempfile
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("kaizer.pipeline.storage")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StoredObject:
    """Represents a file that has been persisted to a storage backend."""

    key: str              # storage-backend key, e.g. "clips/42/master.mp4"
    url: str              # fetchable URL (signed for R2, /media/<key> for Local)
    backend: str          # 'r2' | 'local'
    size_bytes: int
    etag: str = ""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class StorageProvider(ABC):
    """Uniform interface for storage backends."""

    name: str  # 'r2' | 'local'

    # -- Abstract methods --------------------------------------------------

    @abstractmethod
    def upload(
        self,
        local_path: str,
        key: str,
        *,
        content_type: Optional[str] = None,
    ) -> StoredObject:
        """Upload *local_path* to the backend under *key*.

        Parameters
        ----------
        local_path:
            Absolute path to the file on local disk.
        key:
            Destination storage key (forward-slash separated).
        content_type:
            Optional MIME type.  Guessed from extension when omitted.

        Returns
        -------
        StoredObject
            Metadata for the uploaded object.
        """

    @abstractmethod
    def download(self, key: str, local_path: str) -> str:
        """Fetch storage *key* to *local_path*. Returns *local_path*."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove *key* from the backend."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return True if *key* exists in the backend, False otherwise."""

    @abstractmethod
    def get_url(
        self,
        key: str,
        *,
        signed: bool = True,
        expires_s: int = 3600,
    ) -> str:
        """Return a URL for *key*.

        Parameters
        ----------
        key:
            Storage key to build a URL for.
        signed:
            When True, return a pre-signed / time-limited URL.
            When False, return a public CDN URL (only meaningful for R2 when
            ``R2_PUBLIC_BASE_URL`` is configured).
        expires_s:
            Expiry window for signed URLs (default: 3600 seconds = 1 hour).
        """

    # -- Shared concrete helper -------------------------------------------

    def ensure_local(
        self,
        url_or_key: str,
        *,
        cache_dir: Optional[str] = None,
    ) -> str:
        """Return a local readable path for *url_or_key*.

        Resolution order:

        1. If *url_or_key* is an existing local file path → return it as-is.
        2. Strip ``file://`` prefix if present, then check again.
        3. Treat *url_or_key* as a storage key and download it to a
           deterministic cache path.  Re-uses the cached file if it already
           exists so repeated calls are cheap.

        Parameters
        ----------
        url_or_key:
            Either an existing local file path, a ``file://`` URI, or a
            storage backend key.
        cache_dir:
            Directory to download into when the file is remote.  Defaults
            to ``<tempfile.gettempdir()>/kaizer_storage_cache``.
        """
        # ------------------------------------------------------------------
        # 1. Strip file:// prefix
        # ------------------------------------------------------------------
        path = url_or_key
        if path.startswith("file://"):
            path = path[7:]

        # ------------------------------------------------------------------
        # 2. Already a local file?
        # ------------------------------------------------------------------
        if os.path.isfile(path):
            return path

        # ------------------------------------------------------------------
        # 3. Download to deterministic cache path
        # ------------------------------------------------------------------
        key = url_or_key
        # Sanitise the key to a safe filename component.
        safe = key.replace("/", "_").replace("\\", "_").lstrip("_")
        dest_dir = cache_dir or os.path.join(
            tempfile.gettempdir(), "kaizer_storage_cache"
        )
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, safe)

        if os.path.isfile(dest_path):
            logger.debug(
                "storage.ensure_local: cache hit for key=%r → %s", key, dest_path
            )
            return dest_path

        logger.info(
            "storage.ensure_local: downloading key=%r → %s", key, dest_path
        )
        return self.download(key, dest_path)


# ---------------------------------------------------------------------------
# Local storage provider
# ---------------------------------------------------------------------------


class LocalStorage(StorageProvider):
    """Stores files on the local filesystem under *root_dir*.

    URLs are ``/media/<key>`` — served by the FastAPI ``/media`` static mount.
    """

    name: str = "local"

    _local_logger = logging.getLogger("kaizer.pipeline.storage.local")

    def __init__(self, root_dir: str) -> None:
        """
        Parameters
        ----------
        root_dir:
            Absolute path to the directory that acts as the storage root.
            Created automatically if it does not exist.
        """
        self.root_dir: str = str(Path(root_dir).resolve())
        os.makedirs(self.root_dir, exist_ok=True)
        self._local_logger.debug(
            "LocalStorage initialised with root_dir=%r", self.root_dir
        )

    # -- Helpers ----------------------------------------------------------

    def _full_path(self, key: str) -> str:
        """Resolve a storage key to its absolute filesystem path."""
        # Normalise forward slashes; avoid path traversal.
        safe_key = key.replace("\\", "/").lstrip("/")
        return os.path.join(self.root_dir, *safe_key.split("/"))

    # -- Interface --------------------------------------------------------

    def upload(
        self,
        local_path: str,
        key: str,
        *,
        content_type: Optional[str] = None,
    ) -> StoredObject:
        dest = self._full_path(key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(local_path, dest)
        size = os.path.getsize(dest)
        url = f"/media/{key.replace(chr(92), '/').lstrip('/')}"
        self._local_logger.info(
            "uploaded local_path=%r → key=%r (%d bytes)", local_path, key, size
        )
        return StoredObject(
            key=key,
            url=url,
            backend=self.name,
            size_bytes=size,
            etag="",
        )

    def download(self, key: str, local_path: str) -> str:
        src = self._full_path(key)
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        shutil.copy2(src, local_path)
        self._local_logger.info(
            "downloaded key=%r → %r", key, local_path
        )
        return local_path

    def delete(self, key: str) -> None:
        path = self._full_path(key)
        if os.path.isfile(path):
            os.remove(path)
            self._local_logger.info("deleted key=%r", key)
        else:
            self._local_logger.warning(
                "delete called on non-existent key=%r", key
            )

    def exists(self, key: str) -> bool:
        return os.path.isfile(self._full_path(key))

    def get_url(
        self,
        key: str,
        *,
        signed: bool = True,
        expires_s: int = 3600,
    ) -> str:
        # signed flag is meaningless for local storage; always return /media/<key>
        return f"/media/{key.replace(chr(92), '/').lstrip('/')}"


# ---------------------------------------------------------------------------
# Cloudflare R2 (S3-compatible) storage provider
# ---------------------------------------------------------------------------


def _guess_content_type(path: str) -> str:
    """Guess MIME type from file extension; fall back to application/octet-stream."""
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


class R2Storage(StorageProvider):
    """Stores files on Cloudflare R2 via the S3-compatible boto3 API.

    The boto3 client is lazy-initialised (first call to any method) to keep
    import time fast.

    Environment variables read in the constructor
    ---------------------------------------------
    R2_BUCKET            : bucket name (required)
    R2_ENDPOINT          : endpoint URL, e.g. https://<account>.r2.cloudflarestorage.com
    R2_ACCESS_KEY_ID     : S3 access key ID
    R2_SECRET_ACCESS_KEY : S3 secret access key (never logged)
    R2_PUBLIC_BASE_URL   : optional CDN / public-access base URL
    """

    name: str = "r2"

    _r2_logger = logging.getLogger("kaizer.pipeline.storage.r2")

    def __init__(
        self,
        *,
        bucket: Optional[str] = None,
        endpoint: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        public_base_url: Optional[str] = None,
        key_prefix: Optional[str] = None,
    ) -> None:
        self.bucket: str = (
            bucket or os.environ.get("R2_BUCKET", "")
        )
        self.endpoint: str = (
            endpoint or os.environ.get("R2_ENDPOINT", "")
        )
        self.access_key_id: str = (
            access_key_id or os.environ.get("R2_ACCESS_KEY_ID", "")
        )
        # Secret is stored but NEVER logged.
        self._secret_access_key: str = (
            secret_access_key or os.environ.get("R2_SECRET_ACCESS_KEY", "")
        )
        raw_base = (
            public_base_url
            if public_base_url is not None
            else os.environ.get("R2_PUBLIC_BASE_URL", "")
        )
        # Treat empty string as "not set".
        self.public_base_url: str = (raw_base or "").strip()

        # Key prefix: transparent namespace inside the bucket so dev/local/
        # prod traffic coexist without collisions. A caller passes
        # "clips/42/x.mp4" and internally the object lives at
        # "<prefix>clips/42/x.mp4". Returned StoredObject.key stays
        # unprefixed — prefix is an internal detail.
        raw_prefix = (
            key_prefix
            if key_prefix is not None
            else os.environ.get("R2_KEY_PREFIX", "")
        )
        self.key_prefix: str = self._normalize_prefix(raw_prefix)

        if not self.bucket:
            raise ValueError(
                "R2Storage: R2_BUCKET is not set. "
                "Ensure the environment variable is configured."
            )
        if not self.endpoint:
            raise ValueError(
                "R2Storage: R2_ENDPOINT is not set. "
                "Ensure the environment variable is configured."
            )

        # Lazy client: created on first access via _get_client()
        self._client = None
        self._client_lock: threading.Lock = threading.Lock()

        # Log partial key ID only — never the secret.
        partial_id = (self.access_key_id[:8] + "…") if len(self.access_key_id) > 8 else self.access_key_id
        self._r2_logger.info(
            "R2Storage configured: bucket=%r endpoint=%r access_key_id=%s prefix=%r",
            self.bucket,
            self.endpoint,
            partial_id,
            self.key_prefix,
        )

    @staticmethod
    def _normalize_prefix(raw: str) -> str:
        """Normalise an R2 key prefix: strip leading slashes, ensure a
        trailing slash when non-empty, collapse doubles. Empty → empty."""
        if not raw:
            return ""
        s = raw.strip().replace("\\", "/").lstrip("/")
        if not s:
            return ""
        # Collapse consecutive slashes
        while "//" in s:
            s = s.replace("//", "/")
        if not s.endswith("/"):
            s = s + "/"
        return s

    def _k(self, key: str) -> str:
        """Apply key_prefix to a caller-supplied key. No-op if no prefix
        or if the caller already included the prefix (idempotent)."""
        if not self.key_prefix:
            return key
        k = key.replace("\\", "/").lstrip("/")
        if k.startswith(self.key_prefix):
            return k
        return self.key_prefix + k

    # -- Lazy boto3 client ------------------------------------------------

    def _get_client(self):  # type: ignore[return]
        """Return the boto3 S3 client, initialising it on first call."""
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                try:
                    import boto3  # type: ignore[import]
                except ImportError as exc:
                    raise RuntimeError(
                        "boto3 is required for R2Storage. "
                        "Install it with: pip install boto3"
                    ) from exc

                self._client = boto3.client(
                    "s3",
                    endpoint_url=self.endpoint,
                    aws_access_key_id=self.access_key_id,
                    aws_secret_access_key=self._secret_access_key,
                    region_name="auto",
                )
                self._r2_logger.debug(
                    "boto3 S3 client initialised for endpoint=%r", self.endpoint
                )
        return self._client

    # -- Interface --------------------------------------------------------

    def upload(
        self,
        local_path: str,
        key: str,
        *,
        content_type: Optional[str] = None,
    ) -> StoredObject:
        resolved_ct = content_type or _guess_content_type(local_path)
        full_key = self._k(key)
        client = self._get_client()
        self._r2_logger.info(
            "uploading %r → bucket=%r key=%r content_type=%r",
            local_path,
            self.bucket,
            full_key,
            resolved_ct,
        )
        try:
            client.upload_file(
                local_path,
                self.bucket,
                full_key,
                ExtraArgs={"ContentType": resolved_ct},
            )
        except Exception as exc:
            self._r2_logger.error(
                "R2 upload FAILED for key=%r: %s", full_key, exc
            )
            raise

        size = os.path.getsize(local_path)

        # Determine the public URL for the stored object (URL always
        # reflects the actual prefixed storage location).
        if self.public_base_url:
            url = f"{self.public_base_url.rstrip('/')}/{full_key}"
        else:
            url = self.get_url(key, signed=True)

        self._r2_logger.info(
            "upload complete: key=%r size=%d url=%r", full_key, size, url
        )
        return StoredObject(
            key=full_key,
            url=url,
            backend=self.name,
            size_bytes=size,
            etag="",
        )

    def download(self, key: str, local_path: str) -> str:
        full_key = self._k(key)
        client = self._get_client()
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        self._r2_logger.info(
            "downloading bucket=%r key=%r → %r", self.bucket, full_key, local_path
        )
        try:
            client.download_file(self.bucket, full_key, local_path)
        except Exception as exc:
            self._r2_logger.error(
                "R2 download FAILED for key=%r: %s", full_key, exc
            )
            raise
        return local_path

    def delete(self, key: str) -> None:
        full_key = self._k(key)
        client = self._get_client()
        self._r2_logger.info(
            "deleting bucket=%r key=%r", self.bucket, full_key
        )
        try:
            client.delete_object(Bucket=self.bucket, Key=full_key)
        except Exception as exc:
            self._r2_logger.error(
                "R2 delete FAILED for key=%r: %s", full_key, exc
            )
            raise

    def exists(self, key: str) -> bool:
        full_key = self._k(key)
        client = self._get_client()
        try:
            client.head_object(Bucket=self.bucket, Key=full_key)
            return True
        except Exception as exc:
            # botocore raises ClientError for 404 and other errors.
            try:
                from botocore.exceptions import ClientError  # type: ignore[import]
                if isinstance(exc, ClientError):
                    code = exc.response.get("Error", {}).get("Code", "")
                    if code in ("404", "NoSuchKey"):
                        return False
            except ImportError:
                pass
            # Unexpected error — log and re-raise so callers notice.
            self._r2_logger.error(
                "R2 exists check FAILED for key=%r: %s", full_key, exc
            )
            raise

    def get_url(
        self,
        key: str,
        *,
        signed: bool = True,
        expires_s: int = 3600,
    ) -> str:
        full_key = self._k(key)
        if signed:
            client = self._get_client()
            url = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": full_key},
                ExpiresIn=expires_s,
            )
            return url

        if self.public_base_url:
            return f"{self.public_base_url.rstrip('/')}/{full_key}"

        # No public base URL configured → fall back to signed URL for safety.
        self._r2_logger.warning(
            "get_url(signed=False) called but R2_PUBLIC_BASE_URL is not set; "
            "falling back to signed URL for key=%r",
            full_key,
        )
        client = self._get_client()
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": full_key},
            ExpiresIn=expires_s,
        )


# ---------------------------------------------------------------------------
# Module-level factory + per-process cache
# ---------------------------------------------------------------------------

_PROVIDER_CACHE: dict[str, StorageProvider] = {}
_PROVIDER_LOCK: threading.Lock = threading.Lock()

# Resolve the backend root for local storage from the same conventions as
# pipeline.py (BASE_DIR is the parent of pipeline_core/).
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_storage_provider(backend: Optional[str] = None) -> StorageProvider:
    """Return the configured storage provider, cached per process.

    Resolution order for the backend name
    --------------------------------------
    1. *backend* argument (when supplied by caller)
    2. ``STORAGE_BACKEND`` environment variable
    3. ``'local'`` (safe default — no cloud credentials needed)

    Parameters
    ----------
    backend:
        ``'r2'`` or ``'local'``.  Pass ``None`` to use the env var / default.

    Returns
    -------
    StorageProvider
        A cached instance of the corresponding provider.

    Raises
    ------
    ValueError
        If an unknown backend name is specified.
    """
    resolved = (
        backend
        or os.environ.get("STORAGE_BACKEND", "local")
    ).lower().strip()

    if resolved in _PROVIDER_CACHE:
        return _PROVIDER_CACHE[resolved]

    with _PROVIDER_LOCK:
        # Double-checked locking
        if resolved in _PROVIDER_CACHE:
            return _PROVIDER_CACHE[resolved]

        provider: StorageProvider
        if resolved == "local":
            local_root = os.environ.get(
                "KAIZER_OUTPUT_ROOT",
                os.path.join(_BASE_DIR, "output"),
            )
            provider = LocalStorage(local_root)
            logger.info("Storage provider: local (root=%r)", local_root)

        elif resolved == "r2":
            provider = R2Storage()
            logger.info(
                "Storage provider: R2 (bucket=%r)", provider.bucket  # type: ignore[attr-defined]
            )

        else:
            raise ValueError(
                f"Unknown storage backend {resolved!r}. "
                "Valid options are: 'r2', 'local'."
            )

        _PROVIDER_CACHE[resolved] = provider
        return provider
