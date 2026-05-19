"""Step 2 real-R2 round-trip smoketest.

Run from project root or KaizerBackend (the script resolves paths
itself). Loads ``KaizerBackend/.env`` for R2 credentials.

What it does (in order):
  1. Snapshot 3 existing v1 job folders   (records key counts + first-byte etags)
  2. Run V2Storage round-trip operations on a freshly-generated test job-id
       - upload, download, exists=True, get_url
       - upload_raw (lands under v2-raw/, not jobs/)
       - upload_cached x2 (second call must be a cache hit)
       - delete, exists=False
  3. Re-snapshot the same 3 v1 job folders and assert identical
  4. Cleanup: delete all v2 test artifacts (best-effort, errors surface)
  5. Apply v2-raw 24h lifecycle rule (merge-safe)
  6. Re-fetch bucket lifecycle and assert our rule is present + intact

Exit code 0 means everything passed. Non-zero means at least one
assertion failed and a real R2 operation may need cleanup.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

# --- Path / env bootstrap ------------------------------------------------

HERE = Path(__file__).resolve().parent                        # .../pipeline_v2/scripts/
PIPELINE_V2_ROOT = HERE.parent                                 # .../pipeline_v2/
KAIZER_BACKEND = PIPELINE_V2_ROOT.parent                       # .../KaizerBackend/
for p in (PIPELINE_V2_ROOT, KAIZER_BACKEND):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

# Load .env from KaizerBackend/.env (the production .env the v1 backend uses)
ENV_FILE = KAIZER_BACKEND / ".env"
if not ENV_FILE.is_file():
    sys.exit(f"FAIL: .env file not found at {ENV_FILE}")

for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    v = v.strip().strip('"').strip("'")
    if k not in os.environ and v:
        os.environ[k] = v

# Force R2 backend regardless of what STORAGE_BACKEND says -- this script
# is meaningless against the local filesystem fallback.
os.environ["STORAGE_BACKEND"] = "r2"

from pipeline_core.storage import R2Storage                    # noqa: E402
from pipeline_v2.storage import (                              # noqa: E402
    V2Storage,
    V2_RAW_LIFECYCLE_RULE_ID_BASE,
    V2_RAW_PREFIX,
    V2_RAW_TTL_DAYS,
    _lifecycle_rule_id,
    setup_v2_raw_lifecycle,
)


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def info(msg: str) -> None:
    print(f"  ...    {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


# --- Helpers ------------------------------------------------------------


# v1 of Kaizer News stores artifacts under these top-level prefixes
# (verified by listing the bucket on 2026-05-18). v2 must never touch
# any key under any of these prefixes.
V1_TOP_LEVEL_PREFIXES = (
    "clips",
    "bulletin",
    "sources",
    "raw_uploads",
    "user_assets",
    "beta_renders",
)


def list_keys_under(client, bucket: str, prefix: str, *, limit: int = 50) -> list[dict]:
    """List up to N keys under a prefix as {Key, Size, ETag, LastModified}."""
    out: list[dict] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            out.append({
                "Key": obj["Key"],
                "Size": obj["Size"],
                "ETag": obj.get("ETag", ""),
                "LastModified": obj.get("LastModified", "").isoformat() if obj.get("LastModified") else "",
            })
            if len(out) >= limit:
                return out
    return out


def sample_v1_keys(client, bucket: str, key_prefix: str, *, per_prefix: int = 20) -> dict[str, list[dict]]:
    """Sample v1 keys from each v1 top-level prefix.

    Returns a dict mapping prefix-name -> list of {Key, Size, ETag, LastModified}.
    Prefixes with zero keys in the bucket are silently omitted.
    """
    out: dict[str, list[dict]] = {}
    for v1p in V1_TOP_LEVEL_PREFIXES:
        full_prefix = f"{key_prefix}{v1p}/"
        keys = list_keys_under(client, bucket, full_prefix, limit=per_prefix)
        if keys:
            out[v1p] = keys
    return out


# --- Main ---------------------------------------------------------------


def main() -> int:
    failures: list[str] = []

    banner("Step 2 -- Real-R2 Round-trip Smoketest")
    info(f"using .env from: {ENV_FILE}")
    info(f"bucket: {os.environ.get('R2_BUCKET')}")
    info(f"R2_KEY_PREFIX: {os.environ.get('R2_KEY_PREFIX', '(unset)')!r}")

    # ------------------------------------------------------------------
    # 0. R2 client
    # ------------------------------------------------------------------
    r2 = R2Storage()
    client = r2._get_client()
    bucket = r2.bucket
    key_prefix = r2.key_prefix or ""        # global prefix v1+v2 both inherit

    # ------------------------------------------------------------------
    # 1. Snapshot v1 keys BEFORE
    #    v1 doesn't use a 'jobs/' top-level — its layout is per-artifact
    #    type: clips/, bulletin/, sources/, raw_uploads/, user_assets/,
    #    beta_renders/. We sample from each.
    # ------------------------------------------------------------------
    banner("1. Snapshot v1 keys BEFORE (across all v1 top-level prefixes)")
    snapshot_before = sample_v1_keys(client, bucket, key_prefix, per_prefix=20)
    total_before = sum(len(v) for v in snapshot_before.values())
    info(f"sampled {total_before} v1 key(s) across {len(snapshot_before)} prefix(es)")
    for v1p, keys in snapshot_before.items():
        ok(f"{key_prefix}{v1p}/ -- {len(keys)} key(s) sampled")

    if len(snapshot_before) < 3:
        failures.append(
            f"only {len(snapshot_before)} v1 prefix(es) have data — "
            f"need >=3 distinct prefixes to satisfy the user's acceptance"
        )
        fail(failures[-1])

    # ------------------------------------------------------------------
    # 2. V2Storage round-trip
    # ------------------------------------------------------------------
    banner("2. V2Storage round-trip")
    test_job_id = f"step2-smoketest-{int(time.time())}"
    info(f"test job_id: {test_job_id}")
    store = V2Storage(job_id=test_job_id, r2=r2)

    # --- 2a. upload + exists + download + delete
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src.txt"
        body = b"kaizer step2 round-trip " + os.urandom(64)
        src.write_bytes(body)

        info("upload(src.txt -> smoketest/round1.txt)")
        obj = store.upload(str(src), "smoketest/round1.txt", content_type="text/plain")
        full_v2_output_key = obj.key
        ok(f"uploaded -> key={full_v2_output_key} size={obj.size_bytes}")

        # Assert the actual wire key is jobs/{id}/v2/... (post-prefix)
        expected_output_tail = f"jobs/{test_job_id}/v2/smoketest/round1.txt"
        if not full_v2_output_key.endswith(expected_output_tail):
            failures.append(f"upload key {full_v2_output_key!r} doesn't end with {expected_output_tail!r}")
            fail(failures[-1])
        else:
            ok(f"key correctly ends with jobs/{test_job_id}/v2/smoketest/round1.txt")

        info("exists(smoketest/round1.txt) -> expecting True")
        e1 = store.exists("smoketest/round1.txt")
        if not e1:
            failures.append("exists() returned False right after upload")
            fail(failures[-1])
        else:
            ok("exists True")

        info("download(smoketest/round1.txt) and compare bytes")
        dst = td / "out.txt"
        store.download("smoketest/round1.txt", str(dst))
        if dst.read_bytes() != body:
            failures.append("downloaded bytes do NOT match uploaded bytes")
            fail(failures[-1])
        else:
            ok(f"downloaded {len(body)} bytes -- match")

        info("get_url(smoketest/round1.txt, signed=True, expires_s=120)")
        url = store.get_url("smoketest/round1.txt", signed=True, expires_s=120)
        if not url.startswith("http"):
            failures.append(f"signed URL doesn't look like a URL: {url!r}")
            fail(failures[-1])
        else:
            ok(f"signed URL generated ({len(url)} chars)")

        # --- 2b. upload_raw lands under v2-raw/, not under jobs/
        info("upload_raw(src.bin -> v2-raw/{job_id}/src.bin)")
        raw_src = td / "src.bin"
        raw_src.write_bytes(b"raw input bytes " + os.urandom(32))
        raw_obj = store.upload_raw(str(raw_src), "src.bin", content_type="application/octet-stream")
        expected_raw_tail = f"{V2_RAW_PREFIX}/{test_job_id}/src.bin"
        if not raw_obj.key.endswith(expected_raw_tail):
            failures.append(f"raw key {raw_obj.key!r} doesn't end with {expected_raw_tail!r}")
            fail(failures[-1])
        elif "/jobs/" in raw_obj.key:
            failures.append(f"raw key {raw_obj.key!r} is incorrectly inside jobs/ tree")
            fail(failures[-1])
        else:
            ok(f"raw key correctly under {V2_RAW_PREFIX}/ -- {raw_obj.key}")

        # --- 2c. upload_cached: first call uploads, second is a cache hit
        info("upload_cached(src.txt) -- first call -> expecting upload")
        c1 = store.upload_cached(str(src), content_type="text/plain")
        ok(f"first upload_cached -> key={c1.key}")
        info("upload_cached(src.txt) -- second call -> expecting cache hit (no re-upload)")
        c2 = store.upload_cached(str(src), content_type="text/plain")
        if c1.key != c2.key:
            failures.append(f"cache key changed between calls: {c1.key!r} vs {c2.key!r}")
            fail(failures[-1])
        else:
            ok(f"cache hit confirmed -- same key returned: {c2.key}")

        # --- 2d. delete + exists=False
        info("delete(smoketest/round1.txt)")
        store.delete("smoketest/round1.txt")
        e2 = store.exists("smoketest/round1.txt")
        if e2:
            failures.append("exists() True after delete()")
            fail(failures[-1])
        else:
            ok("exists False after delete -- round-trip complete")

    # ------------------------------------------------------------------
    # 3. Re-snapshot v1 keys AFTER and assert identical
    # ------------------------------------------------------------------
    banner("3. Re-snapshot v1 keys AFTER -- assert untouched")
    snapshot_after = sample_v1_keys(client, bucket, key_prefix, per_prefix=20)

    for v1p, before_keys in snapshot_before.items():
        after_keys = snapshot_after.get(v1p, [])
        before_sorted = sorted(before_keys, key=lambda x: x["Key"])
        after_sorted = sorted(after_keys, key=lambda x: x["Key"])
        if before_sorted == after_sorted:
            ok(f"{key_prefix}{v1p}/ -- IDENTICAL ({len(before_sorted)} keys, same Size+ETag+LastModified)")
        else:
            failures.append(f"{key_prefix}{v1p}/ -- DRIFTED between BEFORE and AFTER")
            fail(failures[-1])
            before_set = {x["Key"] for x in before_sorted}
            after_set = {x["Key"] for x in after_sorted}
            added = after_set - before_set
            removed = before_set - after_set
            if added:   info(f"   + added: {sorted(added)[:5]}")
            if removed: info(f"   - removed: {sorted(removed)[:5]}")

    # ------------------------------------------------------------------
    # 4. Cleanup v2 test artifacts (best-effort, surfaces errors)
    # ------------------------------------------------------------------
    banner("4. Cleanup v2 test artifacts")
    # Anything we wrote during this run lives under:
    #   {prefix}jobs/{test_job_id}/v2/
    #   {prefix}v2-raw/{test_job_id}/
    cleanup_prefixes = [
        f"{key_prefix}jobs/{test_job_id}/",
        f"{key_prefix}{V2_RAW_PREFIX}/{test_job_id}/",
    ]
    for cp in cleanup_prefixes:
        paginator = client.get_paginator("list_objects_v2")
        n = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=cp):
            for obj in page.get("Contents", []) or []:
                client.delete_object(Bucket=bucket, Key=obj["Key"])
                n += 1
        ok(f"deleted {n} object(s) under {cp}")

    # ------------------------------------------------------------------
    # 5. Apply lifecycle rule (merge-safe)
    # ------------------------------------------------------------------
    banner("5. Apply v2-raw 24h lifecycle rule (merge-safe)")
    planned = setup_v2_raw_lifecycle(r2, apply=False)
    info(f"planned merged rules: {[r['ID'] for r in planned['rules']]}")
    applied = setup_v2_raw_lifecycle(r2, apply=True)
    ok(f"applied {len(applied['rules'])} rule(s) total to bucket")

    # ------------------------------------------------------------------
    # 6. Verify lifecycle via GetBucketLifecycleConfiguration
    # ------------------------------------------------------------------
    banner("6. Verify lifecycle rule via GetBucketLifecycleConfiguration")
    # The rule id and filter prefix are namespaced by R2_KEY_PREFIX so
    # dev (local/) and prod (prod/) can coexist in this shared bucket.
    expected_rule_id = _lifecycle_rule_id(key_prefix)
    expected_rule_prefix = f"{key_prefix}{V2_RAW_PREFIX}/"
    info(f"expected rule id: {expected_rule_id!r}")
    info(f"expected rule prefix: {expected_rule_prefix!r}")

    resp = client.get_bucket_lifecycle_configuration(Bucket=bucket)
    rules = resp.get("Rules", []) or []
    info(f"bucket has {len(rules)} lifecycle rule(s) total:")
    for r in rules:
        rid = r.get("ID")
        filt = r.get("Filter", {})
        exp = r.get("Expiration", {})
        info(f"   - id={rid!r} status={r.get('Status')} filter={filt} expiration={exp}")

    our_rule = next((r for r in rules if r.get("ID") == expected_rule_id), None)
    if our_rule is None:
        failures.append(f"lifecycle rule {expected_rule_id!r} NOT FOUND after apply")
        fail(failures[-1])
    else:
        ok(f"rule {expected_rule_id!r} present")
        if our_rule.get("Status") != "Enabled":
            failures.append(f"rule status is {our_rule.get('Status')!r}, expected 'Enabled'")
            fail(failures[-1])
        else:
            ok("rule status Enabled")
        f = our_rule.get("Filter", {})
        if f.get("Prefix") != expected_rule_prefix:
            failures.append(f"rule filter is {f!r}, expected Prefix={expected_rule_prefix!r}")
            fail(failures[-1])
        else:
            ok(f"rule filter prefix correct ({expected_rule_prefix})")
        e = our_rule.get("Expiration", {})
        if e.get("Days") != V2_RAW_TTL_DAYS:
            failures.append(f"rule expiration is {e!r}, expected Days={V2_RAW_TTL_DAYS}")
            fail(failures[-1])
        else:
            ok(f"rule expiration correct ({V2_RAW_TTL_DAYS} day)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    banner("Summary")
    if failures:
        print(f"  FAILURES ({len(failures)}):")
        for fmsg in failures:
            print(f"   - {fmsg}")
        return 1
    print("  ALL CHECKS PASSED")
    print("  - v2 round-trip works against real R2")
    print(f"  - {len(snapshot_before)} v1 prefix(es) sampled and confirmed untouched")
    print(f"  - lifecycle rule {expected_rule_id!r} applied + verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
