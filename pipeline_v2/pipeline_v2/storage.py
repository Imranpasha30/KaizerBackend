"""V2 storage layer.

Thin namespace wrapper over v1's ``pipeline_core.storage.R2Storage``. We
compose v1 (no monkey-patching, no fork) so v2 inherits its lazy boto3
init, signed-URL handling, secret hygiene, and ``R2_KEY_PREFIX`` env
behaviour.

Key naming
----------
- Per-job V2 outputs live at ``jobs/{job_id}/v2/{relative}``. This sits
  alongside v1 keys (``jobs/{job_id}/...`` without ``/v2/``) in the same
  bucket — v1 keys are never read or written from here.
- Per-job V2 raw inputs (the user's source video, transient) live at
  ``v2-raw/{job_id}/{filename}``. Separate top-level prefix so the
  24-hour TTL lifecycle rule can target ``v2-raw/`` exactly without
  affecting v1 or v2 outputs.
- Per-job content-hash cache lives at
  ``jobs/{job_id}/v2/.cache/{digest[:2]}/{digest}`` — re-uploading the
  same bytes within a job is skipped.

Lifecycle policy
----------------
``setup_v2_raw_lifecycle()`` configures the 24-hour TTL on the
``v2-raw/`` prefix. It is **merge-safe**: it reads existing bucket
lifecycle rules, removes any prior copy of our rule, appends ours, and
puts the merged set back. v1's rules (if any) survive untouched.

No silent exceptions
--------------------
Every R2 call surfaces failures — the Inngest retry layer handles
backoff. ``exists()`` is the one place we recognise the 404 case and
return ``False`` instead of raising; everything else re-raises.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Optional

# v1 storage layer — same bucket, same boto3 client setup. Read-only
# compose; v1 module is never modified from here.
from pipeline_core.storage import R2Storage, StoredObject

logger = logging.getLogger("pipeline_v2.storage")


# --- Key naming ---------------------------------------------------------

V2_OUTPUT_PREFIX = "jobs"          # jobs/{job_id}/v2/...
V2_OUTPUT_SUBPATH = "v2"
V2_RAW_PREFIX = "v2-raw"           # v2-raw/{job_id}/... (lifecycle target)
V2_CACHE_SUBPATH = ".cache"

V2_RAW_LIFECYCLE_RULE_ID_BASE = "kaizer-v2-raw-24h-ttl"
V2_RAW_TTL_DAYS = 1


def _lifecycle_rule_id(key_prefix: str) -> str:
    """Compose the lifecycle rule id, namespaced by R2_KEY_PREFIX.

    The kaizernews bucket is shared between the ``local/`` (dev) and
    ``prod/`` namespaces. Each environment gets its own rule so applying
    in one env doesn't clobber the other's TTL.
    """
    if not key_prefix:
        return V2_RAW_LIFECYCLE_RULE_ID_BASE
    ns = key_prefix.strip("/").replace("/", "-")
    return f"{V2_RAW_LIFECYCLE_RULE_ID_BASE}-{ns}" if ns else V2_RAW_LIFECYCLE_RULE_ID_BASE


# Back-compat: the no-prefix rule id callers used to import directly.
V2_RAW_LIFECYCLE_RULE_ID = V2_RAW_LIFECYCLE_RULE_ID_BASE


def _safe_job_id(job_id: str) -> str:
    """Reject anything that could escape the per-job prefix."""
    if not job_id:
        raise ValueError("V2Storage: job_id must be non-empty")
    if "/" in job_id or "\\" in job_id or ".." in job_id:
        raise ValueError(f"V2Storage: invalid job_id {job_id!r}")
    return job_id


def output_key(job_id: str, relative_path: str) -> str:
    """``jobs/{job_id}/v2/{relative}`` — the canonical v2 output key."""
    _safe_job_id(job_id)
    rel = relative_path.replace("\\", "/").lstrip("/")
    return f"{V2_OUTPUT_PREFIX}/{job_id}/{V2_OUTPUT_SUBPATH}/{rel}"


def raw_key(job_id: str, filename: str) -> str:
    """``v2-raw/{job_id}/{filename}`` — 24h TTL prefix for raw inputs."""
    _safe_job_id(job_id)
    fn = filename.replace("\\", "/").lstrip("/")
    return f"{V2_RAW_PREFIX}/{job_id}/{fn}"


def cache_key(job_id: str, digest: str) -> str:
    """``jobs/{job_id}/v2/.cache/{digest[:2]}/{digest}``."""
    _safe_job_id(job_id)
    if not digest or "/" in digest or "\\" in digest:
        raise ValueError(f"V2Storage: bad digest {digest!r}")
    return (
        f"{V2_OUTPUT_PREFIX}/{job_id}/{V2_OUTPUT_SUBPATH}/"
        f"{V2_CACHE_SUBPATH}/{digest[:2]}/{digest}"
    )


# --- Content hashing ----------------------------------------------------


def content_hash(local_path: str, algo: str = "sha256") -> str:
    """Stream-hash a file. 1 MiB chunks so 30-min videos don't OOM."""
    h = hashlib.new(algo)
    with open(local_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- V2Storage ----------------------------------------------------------


class V2Storage:
    """Per-job storage handle for the v2 pipeline.

    Construction:
        V2Storage(job_id, r2=None)   # r2 defaults to env-configured R2Storage()

    The ``r2`` keyword is the test-injection seam; pass a mock there to
    avoid hitting real R2 from unit tests.
    """

    def __init__(self, job_id: str, *, r2: Optional[R2Storage] = None):
        self.job_id = _safe_job_id(job_id)
        # Compose v1's R2Storage: same bucket, same boto3 client. Lazy
        # init means we don't actually open a network connection until
        # the first upload/download call.
        self.r2 = r2 if r2 is not None else R2Storage()

    # ---- Outputs (jobs/{job_id}/v2/...) -------------------------------

    def upload(
        self,
        local_path: str,
        relative_key: str,
        *,
        content_type: Optional[str] = None,
    ) -> StoredObject:
        return self.r2.upload(
            local_path,
            output_key(self.job_id, relative_key),
            content_type=content_type,
        )

    def download(self, relative_key: str, local_path: str) -> str:
        return self.r2.download(output_key(self.job_id, relative_key), local_path)

    def delete(self, relative_key: str) -> None:
        self.r2.delete(output_key(self.job_id, relative_key))

    def exists(self, relative_key: str) -> bool:
        return self.r2.exists(output_key(self.job_id, relative_key))

    def get_url(
        self,
        relative_key: str,
        *,
        signed: bool = True,
        expires_s: int = 3600,
    ) -> str:
        return self.r2.get_url(
            output_key(self.job_id, relative_key),
            signed=signed,
            expires_s=expires_s,
        )

    # ---- Raw input (v2-raw/{job_id}/..., 24h TTL) ---------------------

    def upload_raw(
        self,
        local_path: str,
        filename: str,
        *,
        content_type: Optional[str] = None,
    ) -> StoredObject:
        """Upload a raw input that will be lifecycle-expired in 24h.

        Use this for the user's source video. The mezzanine, audio, and
        every downstream artifact go through ``upload()`` so they
        survive past the TTL window.
        """
        return self.r2.upload(
            local_path,
            raw_key(self.job_id, filename),
            content_type=content_type,
        )

    # ---- Content-hash dedup ------------------------------------------

    def upload_cached(
        self,
        local_path: str,
        *,
        content_type: Optional[str] = None,
    ) -> StoredObject:
        """Upload only if no object with the same content hash exists.

        Returns a ``StoredObject`` either way — on cache hit, the
        returned object references the existing key without performing
        a re-upload.
        """
        digest = content_hash(local_path)
        key = cache_key(self.job_id, digest)
        if self.r2.exists(key):
            size = os.path.getsize(local_path)
            url = self.r2.get_url(key, signed=True)
            # v1's R2Storage.upload() returns StoredObject.key as the
            # POST-PREFIX (wire) key. Match that contract on the cache-hit
            # path too, so consumers see the same shape regardless of
            # hit/miss.
            full_key = self.r2._k(key)
            logger.info(
                "v2 cache hit job=%s key=%s (skipped %d-byte upload)",
                self.job_id, full_key, size,
            )
            return StoredObject(
                key=full_key,
                url=url,
                backend=self.r2.name,
                size_bytes=size,
                etag="",
            )
        return self.r2.upload(local_path, key, content_type=content_type)


# --- Bucket lifecycle policy -------------------------------------------


def setup_v2_raw_lifecycle(
    r2: Optional[R2Storage] = None,
    *,
    apply: bool = False,
) -> dict:
    """Configure the v2-raw 24h TTL lifecycle rule on the bucket.

    Merge-safe: reads existing lifecycle rules, removes any prior copy
    of our rule (idempotent re-apply), appends ours, writes back. v1
    rules are not modified.

    Args:
        r2: R2Storage instance. Defaults to env-configured.
        apply: If False (default), return the planned config without
            writing. If True, perform the PutBucketLifecycleConfiguration
            call.

    Returns:
        Dict with keys:
          - ``rules``: list of rule dicts that would be / were applied.
          - ``applied``: bool, whether we actually wrote to the bucket.
    """
    r2 = r2 if r2 is not None else R2Storage()
    client = r2._get_client()  # noqa: SLF001 — composed by design

    # The actual prefix on the wire is {R2_KEY_PREFIX}{V2_RAW_PREFIX}/ —
    # e.g. local/v2-raw/ in dev, prod/v2-raw/ in prod. The rule id is
    # namespaced the same way so dev and prod can coexist in the shared
    # bucket without overwriting each other.
    key_prefix = getattr(r2, "key_prefix", "") or ""
    rule_id = _lifecycle_rule_id(key_prefix)
    rule_prefix = f"{key_prefix}{V2_RAW_PREFIX}/"

    v2_rule = {
        "ID": rule_id,
        "Status": "Enabled",
        "Filter": {"Prefix": rule_prefix},
        "Expiration": {"Days": V2_RAW_TTL_DAYS},
    }

    existing_rules: list[dict] = []
    try:
        resp = client.get_bucket_lifecycle_configuration(Bucket=r2.bucket)
        existing_rules = resp.get("Rules", []) or []
    except Exception as exc:
        # Only swallow the "no policy yet" case. Anything else surfaces
        # so callers / Inngest see real failures.
        from botocore.exceptions import ClientError

        if isinstance(exc, ClientError):
            code = exc.response.get("Error", {}).get("Code", "")
            if code != "NoSuchLifecycleConfiguration":
                logger.error("get_bucket_lifecycle_configuration failed: %s", exc)
                raise
        else:
            raise

    merged = [r for r in existing_rules if r.get("ID") != rule_id]
    merged.append(v2_rule)

    if apply:
        client.put_bucket_lifecycle_configuration(
            Bucket=r2.bucket,
            LifecycleConfiguration={"Rules": merged},
        )
        logger.info(
            "applied v2 raw lifecycle rule to bucket=%s (%d total rules)",
            r2.bucket, len(merged),
        )

    return {"rules": merged, "applied": apply}
