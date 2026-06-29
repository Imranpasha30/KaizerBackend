"""Wave 1.2 SQL smoke — executes every job_queue statement against the
live dev Postgres (empty-queue safe; no fixtures needed). Catches CTE /
window-function / cast errors that only surface at execution time.

Run from KaizerBackend/:  python scripts/check_wave1_queue_sql.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    from services import job_queue

    claimed = job_queue.claim_jobs("smoke:check:00000000", batch=2)
    print(f"claim_jobs OK (claimed {len(claimed)} — expected 0 on idle dev)")
    # If the idle assumption is wrong and we DID claim something real,
    # hand it straight back so nothing is disturbed.
    for j in claimed:
        job_queue.requeue_for_retry(j.id, "smoke:check:00000000",
                                    "smoke test — returned to queue")
        print(f"  returned job {j.id} to queue")

    reaped = job_queue.reap_expired()
    print(f"reap_expired OK (reaped {len(reaped)})")

    unparked = job_queue.unpark_batch(1)
    print(f"unpark_batch OK (unparked {len(unparked)})")

    exhausted = job_queue.list_exhausted(5)
    print(f"list_exhausted OK ({len(exhausted)} rows)")

    snap = job_queue.queue_depth_snapshot()
    print(f"queue_depth_snapshot OK: {snap}")

    renewed = job_queue.renew_leases("smoke:check:00000000", [999999999])
    print(f"renew_leases OK (renewed {len(renewed)} of 1 bogus id)")

    state = job_queue.requeue_for_retry(999999999, None, "bogus")
    assert state == "stale", f"bogus requeue returned {state!r}"
    print("requeue_for_retry(bogus) OK -> stale")

    for a in (1, 2, 3, 4, 5):
        d = job_queue.backoff_seconds(a)
        print(f"backoff(attempt={a}) ~ {d}s")

    print("QUEUE SQL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
