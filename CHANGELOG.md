# Changelog

All notable changes to the Kaizer backend. Most recent entries first.

## 2026-05-19 — V2 Beta launch (Phase 14)

Added the multi-stage V2 pipeline as an opt-in 5th platform card:
**"Full Video + Shorts (V2 Beta)"**. The four existing V1 platforms
(Instagram Reel, YouTube Short, YouTube Full, Full Video + Shorts)
ship unchanged — V2 is fully side-by-side behind the
`KAIZER_V2_ENABLED` feature flag.

### V2 pipeline
- **Stage 1** — Deepgram Nova-3 multilingual STT (replaces Whisper-Groq which is broken on Telugu — backlog item 59).
- **Stage 2** — Gemini 2.5 Pro continuity editor (semantic cut detection + skipped-segment categorization).
- **Stage 2.5** — Gemini 2.5 Flash entity canonicalizer (6-cap PERSON / ORG / LOCATION canon).
- **Stages 3a/3b/3c** — Gemini 2.5 Flash, parallel via `asyncio.gather` inside one Inngest step:
  - 3a: 5–10 shorts with duration constraint (Option E hybrid: lenient parse + 3-tier outcome).
  - 3b: native-script SEO metadata.
  - 3c: ID-reuse image plan.
- **Stage 4** — ports V1 rendering as-is; adds `ImageSourcer` for search-first PERSON-no-generate image policy.
- Orchestrated by Inngest (SDK 0.5.18), 7 step boundaries, full Pydantic state round-trip.
- Two-layer cancellation: cooperative `_check_cancelled` + SIGKILL via `_V2WorkerProxy` in V1's `_ACTIVE_PROCS` registry (Step 12.3 verified).
- D-10.10 idempotency: stable `Event.id = f"job-{job_id}"` dedupes duplicate submissions within Inngest's 24h window (Step 12.4 verified).

### V2 Beta launch features (Step 13 / Phase 14)
- **Job naming** (D-13.11) — every job gets an optional human-readable name. Form field on `/new`, inline-editable on `/jobs/:id` via PATCH `/api/jobs/:id/rename/`. Defaults to first 80 chars of the source filename when left blank.
- **Job feedback** (D-13.8 + D-13.13) — 0-100 rating + optional comment captured on `/jobs/:id` after a job reaches `status='done'`. One feedback per (job, user) — enforced by unique constraint, returns 409 on duplicate. Stored in the new `job_feedback` table; user_id `SET NULL` on user deletion so aggregate stats survive.
- **User V2 stats** (D-13.12 user-facing) — `/v2-stats` page + `GET /api/v2/stats/` endpoint. Per-user aggregates (counts, success rate, average rating).
- **Admin V2 dashboard** (D-13.8 + D-13.12 admin-facing) — `/admin/v2` tab. Failure breakdown by `permanent:*` slug, rating distribution, cancellation rate, paginated feedback list joined with job + user metadata.
- **Beta visual indicators** (D-13.7) — amber BETA pill on the V2 platform card (NewJob wizard) + on each V2 job row (Home) + in the page title (JobDetail).
- **Cancellation state pill** (Step 12.5) — V2StagePill shifts to amber "Cancellation requested — finishing X" when `cancel_requested=true`, communicating the ~14s–90s gap between click and `Job.status='failed'`.
- **STT provider warnings** (Step 12.5 + backlog item 59) — Indian-language pickers surface a "Recommended for Telugu/Hindi" badge on Deepgram and a warning banner on Whisper-Groq.

### Operations
- **Preflight script** — `pipeline_v2/scripts/preflight_v2_launch.py`. Verifies env vars, `INNGEST_DEV=0`, V2 flag truthy, DB reachable, Inngest Cloud reachable. Run before flipping prod traffic.
- **Runbook** — `pipeline_v2/RUNBOOK.md`. Rollback procedure, daily monitoring SQL, common failure modes, escalation tree.
- **GCP spend cap** — operator must set a $50/day Gemini API cap in GCP console (D-13.6 critical) — this is the only physical guard against runaway loops, since the internal cost ledger is known-broken (backlog item 62).

### Schema (migration ledger: `docs/MIGRATIONS.md` Phase 14 + 15)
- `jobs.name VARCHAR(120) NULL`
- `job_feedback` table — rating CHECK [0,100], unique (job_id, user_id), job_id ON DELETE CASCADE, user_id ON DELETE SET NULL.

### Pre-flight env-var contract (D-13.2)
| Var                           | Purpose                                                       |
|-------------------------------|---------------------------------------------------------------|
| `KAIZER_V2_ENABLED=1`         | Gates the V2 platform card + the Inngest webhook mount.       |
| `INNGEST_EVENT_KEY`           | Inngest Cloud event-send auth.                                |
| `INNGEST_SIGNING_KEY`         | Inngest Cloud webhook signature validation.                   |
| `INNGEST_DEV=0` (or unset)    | Must be off in prod or Cloud will reject.                     |
| `KAIZER_STT_DEFAULT_PROVIDER=deepgram` | Indian-language STT routes to Deepgram (item 59).    |
| `DEEPGRAM_API_KEY`            | Stage 1.                                                      |
| `GEMINI_API_KEY`              | Stages 2 / 2.5 / 3a / 3b / 3c.                                |
| `OPENAI_API_KEY` (optional)   | Stage 4 image fallback.                                       |

### Rollback (D-13.5)
Flip `KAIZER_V2_ENABLED=0` and restart the backend. The V2 platform card disappears from `/api/platforms/`, the `/api/inngest` webhook unmounts, and the four V1 platforms are unaffected. Time to rollback: <2 minutes.

---

## Earlier history

Prior changes are tracked in git commit history. This changelog
starts at the V2 Beta launch because that's the first user-facing
shipment significant enough to warrant a formal entry.
