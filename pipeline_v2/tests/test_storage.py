"""Unit tests for ``pipeline_v2.storage``.

Strategy: pass a mock ``R2Storage`` to ``V2Storage`` so no real R2
connection happens. The boto3 client itself is also mocked for the
lifecycle-rule test (the only test that calls into ``r2._get_client()``).

Real-R2 round-trip is verified manually by the user (the plan's Step 2
acceptance check). This file covers everything that can be tested
deterministically without network credentials.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from unittest.mock import MagicMock, call

import pytest
from botocore.exceptions import ClientError

from pipeline_core.storage import StoredObject
from pipeline_v2 import storage as v2storage
from pipeline_v2.storage import (
    V2Storage,
    V2_RAW_LIFECYCLE_RULE_ID,
    V2_RAW_PREFIX,
    V2_RAW_TTL_DAYS,
    cache_key,
    content_hash,
    output_key,
    raw_key,
    setup_v2_raw_lifecycle,
)


# --------------------------------------------------------------------- #
# Fixtures                                                              #
# --------------------------------------------------------------------- #


@pytest.fixture
def fake_r2():
    """A MagicMock that looks like an R2Storage instance.

    ``_k`` is wired as identity so cache-hit code paths that re-derive
    the post-prefix key behave correctly in tests. Real R2Storage._k
    prepends ``key_prefix`` — with key_prefix="" the two are equivalent.
    """
    m = MagicMock()
    m.name = "r2"
    m.bucket = "test-bucket"
    m.key_prefix = ""           # default: no global key prefix
    m._k.side_effect = lambda k: k
    return m


@pytest.fixture
def store(fake_r2):
    return V2Storage(job_id="job-abc", r2=fake_r2)


@pytest.fixture
def tmp_file(tmp_path):
    p = tmp_path / "src.bin"
    p.write_bytes(b"hello kaizer v2")
    return str(p)


# --------------------------------------------------------------------- #
# Key composition                                                       #
# --------------------------------------------------------------------- #


class TestKeyComposition:
    def test_output_key_basic(self):
        assert output_key("abc", "stage_0/mezzanine.mp4") == "jobs/abc/v2/stage_0/mezzanine.mp4"

    def test_output_key_normalises_leading_slash(self):
        assert output_key("abc", "/stage_0/x.mp4") == "jobs/abc/v2/stage_0/x.mp4"

    def test_output_key_normalises_backslashes(self):
        assert output_key("abc", "stage_0\\x.mp4") == "jobs/abc/v2/stage_0/x.mp4"

    def test_raw_key_uses_separate_top_level_prefix(self):
        # CRITICAL: raw uploads must NOT live under jobs/{id}/v2/ — they
        # need to be under v2-raw/ so the lifecycle prefix filter matches
        # exactly without snagging outputs.
        k = raw_key("abc", "source.mp4")
        assert k == "v2-raw/abc/source.mp4"
        assert not k.startswith("jobs/")

    def test_cache_key_shards_by_prefix(self):
        digest = "a" * 64
        assert cache_key("abc", digest) == f"jobs/abc/v2/.cache/aa/{'a' * 64}"

    def test_cache_key_rejects_bad_digest(self):
        with pytest.raises(ValueError):
            cache_key("abc", "")
        with pytest.raises(ValueError):
            cache_key("abc", "foo/bar")


class TestSafeJobId:
    @pytest.mark.parametrize("bad", ["", "../etc", "a/b", "a\\b", "..", "../"])
    def test_traversal_attempts_rejected(self, bad):
        with pytest.raises(ValueError):
            v2storage._safe_job_id(bad)

    @pytest.mark.parametrize("good", ["abc", "job-123", "20260518-153012-xyz"])
    def test_normal_ids_accepted(self, good):
        assert v2storage._safe_job_id(good) == good


# --------------------------------------------------------------------- #
# Content hashing                                                       #
# --------------------------------------------------------------------- #


class TestContentHash:
    def test_known_sha256(self, tmp_path):
        p = tmp_path / "a.bin"
        p.write_bytes(b"hello kaizer v2")
        # sha256 of "hello kaizer v2"
        expected = hashlib.sha256(b"hello kaizer v2").hexdigest()
        assert content_hash(str(p)) == expected

    def test_deterministic_across_calls(self, tmp_file):
        assert content_hash(tmp_file) == content_hash(tmp_file)


# --------------------------------------------------------------------- #
# V2Storage thin wrappers                                                #
# --------------------------------------------------------------------- #


class TestV2StorageWrappers:
    def test_upload_prefixes_key(self, store, fake_r2, tmp_file):
        store.upload(tmp_file, "stage_0/mezz.mp4", content_type="video/mp4")
        fake_r2.upload.assert_called_once_with(
            tmp_file,
            "jobs/job-abc/v2/stage_0/mezz.mp4",
            content_type="video/mp4",
        )

    def test_upload_raw_uses_v2_raw_prefix(self, store, fake_r2, tmp_file):
        store.upload_raw(tmp_file, "source.mp4", content_type="video/mp4")
        called_key = fake_r2.upload.call_args.args[1]
        assert called_key == "v2-raw/job-abc/source.mp4"
        # The raw path MUST NOT touch jobs/ — that would break the
        # lifecycle scoping.
        assert not called_key.startswith("jobs/")

    def test_download_prefixes_key(self, store, fake_r2):
        store.download("out.json", "/tmp/o.json")
        fake_r2.download.assert_called_once_with("jobs/job-abc/v2/out.json", "/tmp/o.json")

    def test_delete_only_targets_v2_keys(self, store, fake_r2):
        store.delete("stage_2/output.json")
        fake_r2.delete.assert_called_once_with("jobs/job-abc/v2/stage_2/output.json")
        # Belt-and-suspenders: no v1 key (jobs/job-abc/anything-not-under-v2)
        # is ever passed to delete().
        for c in fake_r2.delete.call_args_list:
            assert "/v2/" in c.args[0]

    def test_exists_prefixes_key(self, store, fake_r2):
        fake_r2.exists.return_value = True
        assert store.exists("foo")
        fake_r2.exists.assert_called_once_with("jobs/job-abc/v2/foo")

    def test_get_url_prefixes_key(self, store, fake_r2):
        fake_r2.get_url.return_value = "https://signed.example/x"
        url = store.get_url("foo", signed=True, expires_s=120)
        assert url == "https://signed.example/x"
        fake_r2.get_url.assert_called_once_with(
            "jobs/job-abc/v2/foo", signed=True, expires_s=120
        )

    def test_bad_job_id_rejected_at_init(self, fake_r2):
        with pytest.raises(ValueError):
            V2Storage(job_id="../etc/passwd", r2=fake_r2)


# --------------------------------------------------------------------- #
# Content-hash cache                                                     #
# --------------------------------------------------------------------- #


class TestUploadCached:
    def test_cache_miss_uploads(self, store, fake_r2, tmp_file):
        fake_r2.exists.return_value = False
        fake_r2.upload.return_value = StoredObject(
            key="x", url="u", backend="r2", size_bytes=15
        )

        store.upload_cached(tmp_file, content_type="application/octet-stream")

        # Exactly one upload happened, at the cache key.
        fake_r2.upload.assert_called_once()
        called_key = fake_r2.upload.call_args.args[1]
        assert called_key.startswith("jobs/job-abc/v2/.cache/")

    def test_cache_hit_returns_existing_without_upload(self, store, fake_r2, tmp_file):
        fake_r2.exists.return_value = True
        fake_r2.get_url.return_value = "https://signed.example/cached"

        result = store.upload_cached(tmp_file)

        fake_r2.upload.assert_not_called()       # the whole point
        assert isinstance(result, StoredObject)
        assert result.backend == "r2"
        assert result.size_bytes == os.path.getsize(tmp_file)
        assert result.url == "https://signed.example/cached"
        assert result.key.startswith("jobs/job-abc/v2/.cache/")

    def test_cache_key_is_deterministic_for_same_content(self, store, fake_r2, tmp_path):
        # Two files with identical content produce the same cache key.
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"identical")
        b.write_bytes(b"identical")
        fake_r2.exists.return_value = False
        fake_r2.upload.side_effect = lambda lp, k, **_: StoredObject(
            key=k, url="u", backend="r2", size_bytes=os.path.getsize(lp)
        )

        ra = store.upload_cached(str(a))
        rb = store.upload_cached(str(b))

        assert ra.key == rb.key

    def test_cache_hit_returns_post_prefix_key(self, fake_r2, tmp_file):
        # Regression: when R2_KEY_PREFIX is set, cache-MISS goes through
        # v1's upload() which applies the prefix, so its StoredObject.key
        # is post-prefix. The HIT branch must match — otherwise consumers
        # see different key shapes for the same content depending on
        # whether they're the first or second caller.
        fake_r2.key_prefix = "local/"
        fake_r2._k.side_effect = lambda k: f"local/{k}"
        fake_r2.exists.return_value = True
        fake_r2.get_url.return_value = "https://signed.example/cached"

        store = V2Storage(job_id="job-abc", r2=fake_r2)
        result = store.upload_cached(tmp_file)

        fake_r2.upload.assert_not_called()
        assert result.key.startswith("local/jobs/job-abc/v2/.cache/"), (
            f"cache-hit key must be post-prefix, got {result.key!r}"
        )


# --------------------------------------------------------------------- #
# Lifecycle rule setup                                                   #
# --------------------------------------------------------------------- #


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "GetBucketLifecycle")


class TestLifecycleSetup:
    def _mk_r2(self, *, existing_rules=None, raise_no_policy=False, key_prefix=""):
        """Build an R2 mock with a stubbed boto3 client."""
        r2 = MagicMock()
        r2.bucket = "test-bucket"
        r2.key_prefix = key_prefix
        client = MagicMock()
        if raise_no_policy:
            client.get_bucket_lifecycle_configuration.side_effect = _client_error(
                "NoSuchLifecycleConfiguration"
            )
        else:
            client.get_bucket_lifecycle_configuration.return_value = {
                "Rules": existing_rules or []
            }
        r2._get_client.return_value = client
        return r2, client

    def test_no_existing_rules_plans_v2_rule(self):
        r2, _client = self._mk_r2(raise_no_policy=True)
        result = setup_v2_raw_lifecycle(r2, apply=False)
        assert result["applied"] is False
        ids = [r["ID"] for r in result["rules"]]
        assert ids == [V2_RAW_LIFECYCLE_RULE_ID]
        v2_rule = result["rules"][0]
        assert v2_rule["Status"] == "Enabled"
        assert v2_rule["Filter"]["Prefix"] == f"{V2_RAW_PREFIX}/"
        assert v2_rule["Expiration"]["Days"] == V2_RAW_TTL_DAYS

    def test_existing_v1_rules_are_preserved(self):
        existing = [
            {
                "ID": "v1-clip-image-cleanup",
                "Status": "Enabled",
                "Filter": {"Prefix": "clips/"},
                "Expiration": {"Days": 90},
            }
        ]
        r2, _ = self._mk_r2(existing_rules=existing)
        result = setup_v2_raw_lifecycle(r2, apply=False)
        ids = [r["ID"] for r in result["rules"]]
        assert "v1-clip-image-cleanup" in ids       # v1 untouched
        assert V2_RAW_LIFECYCLE_RULE_ID in ids      # v2 added

    def test_prior_v2_rule_is_replaced_not_duplicated(self):
        # Idempotent re-apply: running setup twice must not produce
        # duplicate rule IDs.
        existing = [
            {
                "ID": V2_RAW_LIFECYCLE_RULE_ID,
                "Status": "Enabled",
                "Filter": {"Prefix": f"{V2_RAW_PREFIX}/"},
                "Expiration": {"Days": 7},   # old TTL — should be overwritten
            },
            {
                "ID": "some-other-rule",
                "Status": "Enabled",
                "Filter": {"Prefix": "other/"},
                "Expiration": {"Days": 90},
            },
        ]
        r2, _ = self._mk_r2(existing_rules=existing)
        result = setup_v2_raw_lifecycle(r2, apply=False)
        ids = [r["ID"] for r in result["rules"]]
        assert ids.count(V2_RAW_LIFECYCLE_RULE_ID) == 1   # not duplicated
        v2_rule = next(r for r in result["rules"] if r["ID"] == V2_RAW_LIFECYCLE_RULE_ID)
        assert v2_rule["Expiration"]["Days"] == V2_RAW_TTL_DAYS   # new TTL
        assert "some-other-rule" in ids                  # other rule preserved

    def test_apply_true_calls_put_bucket_lifecycle(self):
        r2, client = self._mk_r2(raise_no_policy=True)
        result = setup_v2_raw_lifecycle(r2, apply=True)
        assert result["applied"] is True
        client.put_bucket_lifecycle_configuration.assert_called_once()
        kwargs = client.put_bucket_lifecycle_configuration.call_args.kwargs
        assert kwargs["Bucket"] == "test-bucket"
        rule_ids = [r["ID"] for r in kwargs["LifecycleConfiguration"]["Rules"]]
        assert V2_RAW_LIFECYCLE_RULE_ID in rule_ids

    def test_apply_false_does_not_write(self):
        r2, client = self._mk_r2(raise_no_policy=True)
        setup_v2_raw_lifecycle(r2, apply=False)
        client.put_bucket_lifecycle_configuration.assert_not_called()

    def test_get_lifecycle_other_error_surfaces(self):
        r2, client = self._mk_r2()
        client.get_bucket_lifecycle_configuration.side_effect = _client_error("AccessDenied")
        with pytest.raises(ClientError):
            setup_v2_raw_lifecycle(r2, apply=False)

    def test_rule_prefix_honors_r2_key_prefix(self):
        # When R2_KEY_PREFIX is set (e.g. local/ in dev), the lifecycle
        # rule prefix must include it — otherwise the TTL would never
        # match the actual wire keys (local/v2-raw/...).
        r2, _ = self._mk_r2(raise_no_policy=True, key_prefix="local/")
        result = setup_v2_raw_lifecycle(r2, apply=False)
        v2_rule = next(r for r in result["rules"]
                       if r["ID"].startswith("kaizer-v2-raw-24h-ttl"))
        assert v2_rule["Filter"]["Prefix"] == "local/v2-raw/"

    def test_rule_id_namespaced_by_key_prefix(self):
        # local and prod must have DISTINCT rule IDs so an apply in one
        # env doesn't clobber the other's TTL in the shared bucket.
        r2_local, _ = self._mk_r2(raise_no_policy=True, key_prefix="local/")
        r2_prod, _ = self._mk_r2(raise_no_policy=True, key_prefix="prod/")
        local_rule = setup_v2_raw_lifecycle(r2_local, apply=False)["rules"][-1]
        prod_rule = setup_v2_raw_lifecycle(r2_prod, apply=False)["rules"][-1]
        assert local_rule["ID"] == "kaizer-v2-raw-24h-ttl-local"
        assert prod_rule["ID"] == "kaizer-v2-raw-24h-ttl-prod"
        assert local_rule["ID"] != prod_rule["ID"]

    def test_rule_id_unprefixed_when_no_key_prefix(self):
        # Back-compat: env without R2_KEY_PREFIX uses the bare base id.
        r2, _ = self._mk_r2(raise_no_policy=True, key_prefix="")
        result = setup_v2_raw_lifecycle(r2, apply=False)
        ids = [r["ID"] for r in result["rules"]]
        assert "kaizer-v2-raw-24h-ttl" in ids

    def test_dev_apply_does_not_clobber_prod_rule(self):
        # Scenario: prod already applied its rule; dev re-applies its own.
        # The merge logic must preserve the prod rule untouched.
        existing = [{
            "ID": "kaizer-v2-raw-24h-ttl-prod",
            "Status": "Enabled",
            "Filter": {"Prefix": "prod/v2-raw/"},
            "Expiration": {"Days": 1},
        }]
        r2, _ = self._mk_r2(existing_rules=existing, key_prefix="local/")
        result = setup_v2_raw_lifecycle(r2, apply=False)
        ids = [r["ID"] for r in result["rules"]]
        assert "kaizer-v2-raw-24h-ttl-prod" in ids       # untouched
        assert "kaizer-v2-raw-24h-ttl-local" in ids      # added


# --------------------------------------------------------------------- #
# Belt-and-suspenders: V1 key isolation                                  #
# --------------------------------------------------------------------- #


class TestV1Isolation:
    """Every key V2Storage emits must be under jobs/{id}/v2/, v2-raw/{id}/,
    or jobs/{id}/v2/.cache/. Anything outside those namespaces would risk
    colliding with v1's jobs/{id}/... keys.
    """

    def test_all_emitted_keys_are_v2_scoped(self, store, fake_r2, tmp_file):
        fake_r2.exists.return_value = False
        fake_r2.upload.return_value = StoredObject(
            key="x", url="u", backend="r2", size_bytes=1
        )
        store.upload(tmp_file, "a.txt")
        store.upload_raw(tmp_file, "b.mp4")
        store.upload_cached(tmp_file)
        store.delete("c.txt")
        store.exists("d.txt")
        store.get_url("e.txt")

        # Collect every key that flowed into the underlying R2 mock.
        keys: list[str] = []
        for m in (fake_r2.upload, fake_r2.delete, fake_r2.exists, fake_r2.get_url):
            for c in m.call_args_list:
                # upload / download positional: (local, key)
                # delete / exists / get_url positional: (key,)
                pos = c.args
                if len(pos) >= 2:
                    keys.append(pos[1])
                elif len(pos) == 1:
                    keys.append(pos[0])

        assert keys, "expected at least one R2 call to inspect"
        for k in keys:
            assert k.startswith("jobs/job-abc/v2/") or k.startswith("v2-raw/job-abc/"), (
                f"v2 storage emitted a non-v2 key: {k!r}"
            )
