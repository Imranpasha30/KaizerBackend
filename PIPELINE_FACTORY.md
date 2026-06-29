# Staged "Factory" Pipeline + Live Admin Flow — build & ops

This documents what shipped, how to switch it on **safely**, what is **verified**
vs what still needs a **live smoke test**, and the hard isolation guarantees.

## What shipped

### 1. Isolation hardening (always on — hardens the current system)
- **I3 — delivery ownership assertion** (`services/upload_dispatch.py`,
  `process()`): before branding/OAuth/upload, the destination channel **must**
  belong to the job's owner (`Channel.user_id == job user_id`), else a
  fail-closed, terminal `OwnershipViolation` (no retry). Stops one user's video
  ever reaching another user's channel — by assertion, not by luck.
- **I1 — storage cache path collision** (`pipeline_core/storage.py`,
  `ensure_local`): cache filenames are now the **sha256 of the full key**
  (the old `/`→`_` scheme could collapse two distinct keys to one file), and
  downloads land in a unique temp file and are **atomically renamed** in, so a
  reader can never observe a half-written file.

### 2. Live telemetry (always on — powers the admin view)
- `services/stage_events.py` — thread-safe in-memory ring buffer + the
  immutable identity `Envelope (tenant_id, user_id, job_id, clip_id, channel_id)`
  + per-station live occupancy + counters. No secrets, no payload; `emit()`
  never raises into a caller.
- `services/upload_dispatch.py` is **instrumented** to emit `brand`/`upload`
  station transitions for every real publish job — so the admin Pipeline Flow
  view shows the **real** pipeline today, with or without the factory engine on.

### 3. The factory engine (additive, OFF by default)
- `services/pipeline_factory.py` — the staged conveyor: per-station bounded
  worker pools + bounded queues + intrinsic **back-pressure** (submit blocks when
  full) + the identity envelope + the 7 canonical stations
  (`ingest → transcribe → cut_plan → trim → compose → brand → upload`).
- `main.py` starts the station pools **only** when
  `KAIZER_PIPELINE_FACTORY_WORKERS=1` (flag-gated startup + shutdown hooks).

### 4. Admin "Pipeline Flow" tab (live + clickable)
- Backend: `GET /api/admin/pipeline/flow` (station occupancy from the real DB +
  recent transitions + queue depths + today's throughput) and
  `GET /api/admin/pipeline/unit?kind=publish|render&id=<id>` (tap a unit → a
  plain-language **diagnosis of why it is where it is / why it is stuck** +
  full state + stage history + render log tail). Both `admin_required`.
- Frontend: `src/pages/AdminPipelineFlow.jsx` — animated conveyor of the 7
  stations with flowing belt dots, per-station unit cards, a KPI rail, a live
  transition ticker, and **click-to-inspect**: tap any unit card (or a ticker
  row) to open the diagnosis modal. Units idle > 15 min get a warning ring that
  invites a tap. Registered as the **Pipeline Flow** admin tab.

## Env flags (all default to the current behaviour)

| Flag | Default | Effect |
|---|---|---|
| `KAIZER_PIPELINE_FACTORY` | `0` | Master switch for the engine (shown as "Engine ON/OFF" in the admin view; reserved to route render stages onto the belt in the next increment). |
| `KAIZER_PIPELINE_FACTORY_WORKERS` | `0` | Boots the station worker pools at startup. |
| `KAIZER_STAGE_<STATION>_WORKERS` | per-station (compose=2, upload=16, …) | Pool size per station. The heavy GPU `compose` station is deliberately small — it paces the belt. |
| `KAIZER_STAGE_<STATION>_QUEUE_SIZE` | `256` | Back-pressure bound per station. |

Stations: `INGEST TRANSCRIBE CUT_PLAN TRIM COMPOSE BRAND UPLOAD`.

## Verified now (automated)
- All backend modules compile + import; flags default OFF.
- `pipeline_factory.self_test()` — 3 synthetic units flow through **all 7
  stations**, each emitting entered+exited (pools/queues/hand-off/back-pressure/
  envelope/telemetry proven end-to-end).
- `GET /pipeline/flow` executes against the **real DB** (no SQL/ORM errors).
- `GET /pipeline/unit` returns a correct diagnosis for a real job (#1766 →
  "Delivered successfully") and a clean 404 for a missing id.
- Diagnosis brain unit-tested: failed / parked-quota / lease-expired / backing-off
  → `stuck=true` with correct severity; healthy queued → not stuck.
- Frontend production build passes (2333 modules) with the new tab + modal.

## Needs a LIVE smoke test before flipping render onto the belt
The publish path is unchanged and fully observable. The heavy **render** stages
(the actual 31 GB RAM hog) are not yet executing **on** the station pools — that
is the next increment. Before `KAIZER_PIPELINE_FACTORY_WORKERS=1` is trusted in
prod with real render work:
1. Start with workers on in a staging run: `KAIZER_PIPELINE_FACTORY_WORKERS=1`.
2. Publish a small real batch; watch the **Pipeline Flow** tab — confirm units
   flow, occupancy is bounded, RAM stays flat, no ghost cards.
3. Force a failure (e.g. revoke a channel token) and confirm the unit shows as
   **stuck** with the right reason on tap, and that `OwnershipViolation` never
   fires for a legitimate job.
4. Only then wire each render stage's real work function into
   `build_pipeline(work_fns=…)` and re-run the smoke test per stage.

## Isolation invariants status
- **I1 sealed paths** — render workspaces already unique; storage cache fixed. ✅
- **I2 identity envelope** — implemented, rides every event. ✅
- **I3 delivery ownership** — fail-closed assertion at the upload boundary. ✅
- **I4 identity-keyed idempotency** — current key is user-scoped (no cross-user
  collision); extend to include tenant+user+clip explicitly when render stages
  land. ⏳
- **I5 no shared mutable state** — stations are pure `(envelope, item)`; only the
  metrics ring buffer is shared (no job payload). ✅

---

# Part 3 — V4 scaling: the NVENC stage gate (6 → 100 → 2000)

V4 render runs as **one subprocess per job**, capped by `KAIZER_PIPELINE_CONCURRENCY`
(was the default **2**) — set for the single NVENC encoder (">2 simultaneous
encodes thrash it"). The waste: a job merely *transcribing* (network, no GPU)
still burned one of those 2 whole-job slots.

**The fix (factory principle on the real architecture):** bound only the
**encode-heavy work** (trim ffmpeg + compose) with a counting semaphore, and
raise overall admission so the light stages run many-at-once.

- `services/stage_gate.py` — a counting semaphore built on **Postgres
  advisory locks**, so it bounds simultaneous encodes **across processes AND
  machines**. Host-scoped by default (`KAIZER_STAGE_GATE_SCOPE=host`): each box
  bounds its own NVENC, so N boxes × `encode_max` scale linearly.
- Wired into `trim_engine.run_step1` (the atomic ffmpeg) and
  `orchestrator.run_job` (the compose stage), both sharing the `encode` gate —
  so total heavy ffmpeg across all jobs ≤ `KAIZER_STAGE_ENCODE_MAX`.
- `GET /api/admin/pipeline/flow` returns `encode_gate {held,max}`; the admin
  header shows an **Encode N/2** chip.

**Live config (in `.env`):**
```
KAIZER_STAGE_GATE=1
KAIZER_STAGE_ENCODE_MAX=2     # = NVENC engines on this box
KAIZER_PIPELINE_CONCURRENCY=8 # render jobs admitted at once (was 2)
```

**Verified:** the gate caps concurrency to 2 under a 5-thread race; `status()`
reports `held` accurately; backend live with `encode_gate enabled max=2`.

## The honest 6 → 100 → 2000 math
- **This box, today:** admission 2 → **8** jobs in flight; ≤2 encode at once
  (encoder + carousel-RAM spike protected). Light stages (ingest/transcribe/
  cut-plan) no longer block on encode slots. Tune `ENCODE_MAX`/`CONCURRENCY`
  while watching **Pipeline Flow** + the system monitor; expect a few-x gain.
- **100:** **horizontal** — ~12–15 boxes on the **same Postgres + R2**, each
  pulling from the durable queue, each gating its own NVENC. Already enabled by
  the stateless subprocess + shared-coordination design; just add machines.
- **2000 (the 30-min spike):** burst ~100 boxes (or cloud GPU instances) for
  the window, then scale to zero. The gate keeps each box safe; the queue
  distributes work; R2 holds artifacts. No single box can do 2000 — this is a
  fleet number, and the architecture is now fleet-ready.

**Still a future optimization (not blocking):** split `run_step1` so
transcribe/cut-plan are their own ungated pools (today they ride inside the
gated trim subprocess but are cheap); and move artifacts fully through R2 for
zero-shared-disk horizontal. Neither is required to start scaling out.
