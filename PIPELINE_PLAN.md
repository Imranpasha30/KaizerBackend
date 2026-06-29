# Brand→Upload Pipeline — autonomous build plan

**Context:** operator (Imran) is away. Build this the moment the current
publish run drains (watcher: `scripts/_wait_drain.py`, bg task). Email
updates to imranpasha.ahmed@gmail.com via `scripts/send_update_email.py`.

## Problem
The durable-queue worker (`services/publish_worker.py`) runs each job's
**branding + uploading as one unit** in a thread, holding its concurrency
slot for the whole lifecycle. While a job is in the network-bound
`uploading` phase, the GPU sits idle but the slot is still held → the next
clip can't start branding. No overlap between GPU work and network work.

`upload_dispatch.process(job_id)` drives: queued → branding →
ready_to_upload → uploading → completed | failed | parked_quota.
The `ready_to_upload` status is the natural handoff point (already exists).

## Goal
Two independent stages with separate concurrency caps:
- **Brand stage** (GPU-bound): claim `queued` → brand → set
  `ready_to_upload`, then RELEASE the slot. cap ~ KAIZER_SCHED_CPU_TOKENS (4).
- **Upload stage** (network-bound): claim `ready_to_upload` → upload →
  `completed`. cap ~ KAIZER_SCHED_NET_TOKENS (16).
Result: while N jobs upload over the network, N NEW jobs brand on the GPU.

## Approach (safest)
1. Split `upload_dispatch.process` into:
   - `process_branding(job_id)`  : queued → (brand, QC, R2) → ready_to_upload.
     Includes the existing quota/idempotency pre-checks as appropriate.
   - `process_upload(job_id)`    : ready_to_upload → (quota reserve if not
     already, YouTube/RTMP/Postiz upload) → completed | failed | parked_quota.
   Keep ALL existing fencing (claimed_by guard), idempotency, credit
   reserve/refund, quota burn-log, and error/park handling intact.
2. `publish_worker` main loop claims BOTH statuses each tick:
   - up to (brand_cap - branding_in_flight) jobs in status `queued`  → brand pool
   - up to (upload_cap - upload_in_flight) jobs in status `ready_to_upload` → upload pool
   Each stage uses the durable-queue SKIP-LOCKED claim + lease/fence.
3. **Revert flag**: `KAIZER_BRAND_UPLOAD_PIPELINE` (default off until
   verified). When off, keep the current single-unit `process()` path
   EXACTLY as today. Flip on only after the regression suites + a live
   smoke pass. This guarantees the system still works for the absent
   operator if the pipeline has a subtle bug.

## Risks / must-not-break
- Durable-queue fencing: each stage must claim with its own lease and
  guard writes with `claimed_by = worker_id` (StaleClaim no-ops).
- Idempotency key + credit reserve/refund must fire exactly once.
- A job that fails branding must NOT advance to upload.
- parked_quota must still park correctly (in the upload stage).
- `_run_job` / scheduler path (non-durable) — leave as-is or mirror;
  durable queue is the live path (KAIZER_DURABLE_QUEUE=1).

## Verification (before flipping the flag on)
1. `python -m py_compile` all touched files.
2. `scripts/test_durable_queue.py` (claim/fence/retry/refund) — must pass
   the non-quota-park assertions (the 7 quota-park assertions are stale,
   pre-existing).
3. `scripts/test_upload_dispatch_smoke.py` — direct + rtmp + idempotency.
4. Restart, health `{status:ok}`.
5. Live smoke: publish 1 fresh clip to 2-3 channels; confirm it goes
   queued→branding→ready_to_upload→uploading→completed across the two
   stages, and that a 2nd batch starts branding while the 1st uploads
   (GPU busy during uploads).
6. Email Imran the result (pass → flag on; fail → flag off, system safe).
