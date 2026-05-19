# Kaizer — Postgres migrations

The dev flow uses SQLite and relies on `Base.metadata.create_all()` +
`_migrate_schema()` (main.py) to evolve the schema lazily on every startup.

Production Postgres runs this migration ledger manually — copy/paste each
section into `psql` in order.  Each section is idempotent via
`IF NOT EXISTS` guards.

---

## Phase 12 — admin panel / Gemini call accounting

Adds one new table: `gemini_calls`.  One row is written per Gemini SDK
call (see `learning/gemini_log.py`).  Raw prompts + responses are NEVER
stored here — only the metadata needed for per-user quota tracking, cost
estimates and the admin analytics dashboard.

```sql
CREATE TABLE IF NOT EXISTS gemini_calls (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER REFERENCES users(id)  ON DELETE SET NULL,
    job_id          INTEGER REFERENCES jobs(id)   ON DELETE SET NULL,
    clip_id         INTEGER REFERENCES clips(id)  ON DELETE SET NULL,
    model           VARCHAR(64)  NOT NULL,
    purpose         VARCHAR(64)  DEFAULT '',
    prompt_tokens   INTEGER DEFAULT 0,
    output_tokens   INTEGER DEFAULT 0,
    total_tokens    INTEGER DEFAULT 0,
    cost_usd        DOUBLE PRECISION DEFAULT 0.0,
    latency_ms      INTEGER DEFAULT 0,
    status          VARCHAR(16) DEFAULT 'ok',   -- ok | error | rate_limited
    error           TEXT DEFAULT '',
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_gemini_calls_user_id    ON gemini_calls (user_id);
CREATE INDEX IF NOT EXISTS ix_gemini_calls_job_id     ON gemini_calls (job_id);
CREATE INDEX IF NOT EXISTS ix_gemini_calls_clip_id    ON gemini_calls (clip_id);
CREATE INDEX IF NOT EXISTS ix_gemini_calls_created_at ON gemini_calls (created_at);
```

### Rollback

Only required if the admin panel is being ripped out — the table is
strictly additive.  A drop is safe because no other table has FKs into
`gemini_calls`:

```sql
DROP TABLE IF EXISTS gemini_calls;
```

### Notes

* `users.is_admin` already exists (Boolean, default false) — no migration
  required for the admin gate.
* OAuth access / refresh tokens are already stored Fernet-encrypted in
  `oauth_tokens.refresh_token_enc` + `access_token_enc`.  The admin
  endpoints never serialize these columns — see `routers/admin.py`
  `_MASK_FIELDS` + `_mask_oauth()` for the redaction helper.

---

## Phase 13 — Pipeline V2 per-step progress (Step 10)

Adds one nullable VARCHAR column to `jobs` so the V2 Inngest orchestrator
can write per-step progress that the UI surfaces while a 10-minute V2
render is running.  The legacy V1 subprocess path (the four pre-V2
platforms) leaves this column NULL.

```sql
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS current_stage VARCHAR(40);
```

### Values

Written by the V2 orchestrator at the start of each Inngest step.  Reset
to NULL when the job finalizes (success or failure).  Permitted values:

* `stage_0_ingest`
* `stage_1_transcribe`
* `stage_2_continuity`
* `stage_2_5_entities`
* `stage_3_fanout`
* `stage_4_render`
* `finalize`

The orchestrator writes this synchronously (not as an Inngest sub-step)
because the write is fire-and-forget — UI freshness matters more than
durability.  Reads from the column treat NULL as "no V2 step in flight".

### Rollback

```sql
ALTER TABLE jobs DROP COLUMN IF EXISTS current_stage;
```

Safe — column is purely additive, no FKs, no indexes.

---

## Phase 14 — V2 Beta: Job naming (D-13.11)

Adds one nullable VARCHAR column to `jobs` so users can give each job
a human-readable label that surfaces in JobsList + JobDetail.  Empty
on rows created before Phase 14; the create endpoint defaults to the
first 80 chars of `video_name` when the form field is left blank.
Editable mid-flight via `PATCH /api/jobs/{id}/rename`.

```sql
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS name VARCHAR(120);
```

### Rollback

```sql
ALTER TABLE jobs DROP COLUMN IF EXISTS name;
```

Safe — additive, no FK, no index.  UI falls back to `video_name`
when `name` is NULL so a column drop is non-breaking at the data
layer.

---

## Phase 15 — V2 Beta: Job feedback (D-13.8 + D-13.13)

Adds one new table: `job_feedback`.  Captures 0–100 rating + optional
free-text comment after a V2 job reaches `status='done'`.  Aggregated
by `/api/v2/stats` (per-user) and `/api/admin/v2-stats` (global), and
listed paginated for ops via `/api/admin/v2-feedback`.

```sql
CREATE TABLE IF NOT EXISTS job_feedback (
    id            SERIAL PRIMARY KEY,
    job_id        INTEGER NOT NULL REFERENCES jobs(id)  ON DELETE CASCADE,
    user_id       INTEGER          REFERENCES users(id) ON DELETE SET NULL,
    rating        INTEGER NOT NULL,
    comment       TEXT DEFAULT '',
    submitted_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_job_feedback_user UNIQUE (job_id, user_id),
    CONSTRAINT ck_job_feedback_rating CHECK (rating >= 0 AND rating <= 100)
);

CREATE INDEX IF NOT EXISTS ix_job_feedback_job_id        ON job_feedback (job_id);
CREATE INDEX IF NOT EXISTS ix_job_feedback_user_id       ON job_feedback (user_id);
CREATE INDEX IF NOT EXISTS ix_job_feedback_submitted_at  ON job_feedback (submitted_at);
```

### Cascade behaviour

* `job_id ON DELETE CASCADE` — feedback without the parent job is
  meaningless; deleting the job removes its feedback rows.
* `user_id ON DELETE SET NULL` — admin aggregate stats (average rating,
  rating distribution) survive a user-account deletion.

### Endpoints

* `POST /api/jobs/{job_id}/feedback` — auth required; status must be
  `done`; returns 409 if the calling user already submitted feedback
  for the job.
* `GET  /api/v2/stats` — user-scoped aggregates (own jobs only).
* `GET  /api/admin/v2-feedback` — admin-only paginated list.
* `GET  /api/admin/v2-stats` — admin-only global aggregates +
  rating distribution + failure-by-slug breakdown.

### Rollback

```sql
DROP TABLE IF EXISTS job_feedback;
```

Safe — no other table has FKs into `job_feedback`.  Drop alone is
sufficient; no need to clean up dependent objects.
