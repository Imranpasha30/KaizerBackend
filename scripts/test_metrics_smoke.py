"""Smoke test for the /metrics Prometheus endpoint — Phase 3.G.

Run AFTER the backend has been (re)started so it picks up the new
routers + prometheus_client import.

USAGE
-----
    cd "e:/kaizer new data training/kaizer/KaizerBackend"
    python scripts/test_metrics_smoke.py

ENV
---
    KAIZER_BACKEND_URL (default: http://127.0.0.1:8000)

EXIT CODES
----------
    0 = PASS
    1 = FAIL (non-200, missing metric, or transport error)
"""
from __future__ import annotations

import os
import sys

import requests


BASE_URL = os.environ.get("KAIZER_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

# The 11 metric names CONTRACTS §4 G-agent + PHASE3 plan require.
EXPECTED_METRICS = [
    "kaizer_publish_task_total",
    "kaizer_upload_job_total",
    "kaizer_slot_active",
    "kaizer_queue_depth",
    "kaizer_credit_balance",
    "kaizer_quota_burn_predicted_total",
    "kaizer_quota_burn_actual_total",
    "kaizer_quota_burn_delta",
    "kaizer_upload_throughput_bytes",
    "kaizer_critical_queue_latency_seconds",
    "kaizer_recovery_orphan_total",
]


def main() -> int:
    url = f"{BASE_URL}/metrics"
    print(f"[smoke] GET {url}")
    try:
        r = requests.get(url, timeout=10)
    except Exception as exc:
        print(f"[FAIL] transport error: {exc}")
        return 1

    if r.status_code != 200:
        print(f"[FAIL] /metrics returned {r.status_code}")
        print(f"  body: {r.text[:500]}")
        return 1

    ct = r.headers.get("content-type", "")
    if "text/plain" not in ct:
        print(f"[FAIL] unexpected content-type: {ct!r}")
        return 1

    text = r.text
    # We look for each metric name as a TYPE/HELP comment or as a value
    # line. Prometheus' generate_latest always emits the metric name in
    # the # TYPE line even when no samples have been recorded.
    missing = []
    for name in EXPECTED_METRICS:
        if name not in text:
            missing.append(name)

    if missing:
        print(f"[FAIL] missing metrics in /metrics output: {missing}")
        # Print a small slice of what we did get for debugging.
        sample_lines = [
            ln for ln in text.splitlines()
            if ln.startswith("# TYPE")
        ][:30]
        print("  observed # TYPE lines (first 30):")
        for ln in sample_lines:
            print(f"    {ln}")
        return 1

    # Sanity: count how many sample lines we got per metric (zero is OK
    # for a fresh DB, but each metric MUST have a # TYPE line).
    type_lines = [ln for ln in text.splitlines() if ln.startswith("# TYPE")]
    print(f"[PASS] all {len(EXPECTED_METRICS)} metrics present in /metrics output")
    print(f"  body_size_bytes={len(text)}")
    print(f"  type_line_count={len(type_lines)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
