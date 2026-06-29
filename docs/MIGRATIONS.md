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

---

## Phase 1 — Upload Rewrite v2 (8 new tables + users.plan_tier_id)

Schema agent's deliverable for the publish/upload system rewrite. See
`docs/upload-rewrite/CONTRACTS.md` §3 for the spec source of truth and
`docs/upload-rewrite/DECISIONS.md` Decisions 3, 4, 6, 8 for the plan-tier
seed values + the existing-user backfill rule.

Eight new tables (PlanTier → BrandProfile → MasterVideo → PublishTask →
UploadJobV2 → PublishAttempt → CreditLedger → QuotaBurnLog) plus one
ADD COLUMN on `users` (`plan_tier_id`). Strictly additive — the legacy
`upload_jobs` table is preserved untouched during cutover so rollback is
trivial.

Run the sections IN ORDER (the FK chain is enforced):

```sql
-- ── 1. plan_tiers (no FK dependencies) ─────────────────────────────
CREATE TABLE IF NOT EXISTS plan_tiers (
    id                          SERIAL PRIMARY KEY,
    name                        VARCHAR(16) NOT NULL UNIQUE,
    monthly_credit_allotment    INTEGER NOT NULL,
    slot_cap_active_uploads     INTEGER NOT NULL,
    direct_path_allowed         BOOLEAN NOT NULL DEFAULT TRUE,
    max_channels                INTEGER NOT NULL,
    max_publishes_per_day       INTEGER NOT NULL,
    created_at                  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Seed the three tiers (idempotent via ON CONFLICT on the UNIQUE name).
-- Values are LOCKED by DECISIONS.md Decisions 3 + 4 + 6.
-- -1 sentinel = unlimited.
INSERT INTO plan_tiers
  (name, monthly_credit_allotment, slot_cap_active_uploads,
   direct_path_allowed, max_channels, max_publishes_per_day)
VALUES
  ('free',         300,   5, FALSE,  1,  1),
  ('pro',         2000,  20, TRUE,  -1, -1),
  ('enterprise', 15000, 100, TRUE,  -1, -1)
ON CONFLICT (name) DO NOTHING;

-- ── 2. master_videos (depends on jobs) ─────────────────────────────
CREATE TABLE IF NOT EXISTS master_videos (
    id                 SERIAL PRIMARY KEY,
    source_upload_id   INTEGER NOT NULL UNIQUE REFERENCES jobs(id) ON DELETE CASCADE,
    r2_key             TEXT NOT NULL,
    duration_seconds   DOUBLE PRECISION NOT NULL,
    bytes              BIGINT NOT NULL,
    width              INTEGER NOT NULL,
    height             INTEGER NOT NULL,
    status             VARCHAR(16) NOT NULL DEFAULT 'staging',
    pipeline_version   VARCHAR(16) NOT NULL,
    clean_master       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at         TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_master_videos_source_upload_id
    ON master_videos (source_upload_id);

-- ── 3. brand_profiles (depends on user_assets; soft FK on owner) ───
CREATE TABLE IF NOT EXISTS brand_profiles (
    id                 SERIAL PRIMARY KEY,
    owner_kind         VARCHAR(16) NOT NULL,
    owner_id           INTEGER NOT NULL,
    name               VARCHAR(120) NOT NULL,
    version            VARCHAR(40) NOT NULL,
    logo_asset_id      INTEGER REFERENCES user_assets(id) ON DELETE SET NULL,
    watermark_text     VARCHAR(64),
    watermark_opacity  DOUBLE PRECISION DEFAULT 0.35,
    watermark_position VARCHAR(16) DEFAULT 'lower-center',
    colors_json        TEXT,
    fonts_json         TEXT,
    intro_asset_id     INTEGER REFERENCES user_assets(id) ON DELETE SET NULL,
    outro_asset_id     INTEGER REFERENCES user_assets(id) ON DELETE SET NULL,
    socials_json       TEXT,
    created_at         TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_brand_profiles_owner
    ON brand_profiles (owner_kind, owner_id);

-- ── 4. users.plan_tier_id ADD COLUMN + backfill ────────────────────
-- Decision 8: every existing user gets the 'pro' tier id so today's
-- behaviour is preserved (no silent slot-cap or paywall mid-flight).
-- The FK is added as NOT VALID first then VALIDATEd so it doesn't
-- block writers during the migration on a large `users` table.
ALTER TABLE users ADD COLUMN IF NOT EXISTS plan_tier_id INTEGER;
ALTER TABLE users
    ADD CONSTRAINT fk_users_plan_tier
    FOREIGN KEY (plan_tier_id) REFERENCES plan_tiers(id) ON DELETE SET NULL
    NOT VALID;
ALTER TABLE users VALIDATE CONSTRAINT fk_users_plan_tier;
CREATE INDEX IF NOT EXISTS ix_users_plan_tier_id ON users (plan_tier_id);

UPDATE users
   SET plan_tier_id = (SELECT id FROM plan_tiers WHERE name = 'pro' LIMIT 1)
 WHERE plan_tier_id IS NULL;

-- ── 5. publish_tasks (depends on users, master_videos) ─────────────
CREATE TABLE IF NOT EXISTS publish_tasks (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    master_video_id INTEGER NOT NULL REFERENCES master_videos(id) ON DELETE CASCADE,
    priority        VARCHAR(16) NOT NULL DEFAULT 'normal',
    status          VARCHAR(20) NOT NULL DEFAULT 'queued',
    target_count    INTEGER NOT NULL DEFAULT 0,
    completed_count INTEGER NOT NULL DEFAULT 0,
    failed_count    INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_publish_tasks_user_id
    ON publish_tasks (user_id);
CREATE INDEX IF NOT EXISTS ix_publish_tasks_master_video_id
    ON publish_tasks (master_video_id);
CREATE INDEX IF NOT EXISTS ix_publish_tasks_created_at
    ON publish_tasks (created_at);
CREATE INDEX IF NOT EXISTS ix_publish_tasks_user_status_priority
    ON publish_tasks (user_id, status, priority);

-- ── 6. upload_jobs_v2 (depends on publish_tasks, channels,
--                       oauth_tokens, brand_profiles, clips) ────────
CREATE TABLE IF NOT EXISTS upload_jobs_v2 (
    id                       SERIAL PRIMARY KEY,
    publish_task_id          INTEGER NOT NULL REFERENCES publish_tasks(id) ON DELETE CASCADE,
    clip_id                  INTEGER REFERENCES clips(id) ON DELETE SET NULL,
    channel_id               INTEGER NOT NULL REFERENCES channels(id) ON DELETE SET NULL,
    oauth_token_id           INTEGER REFERENCES oauth_tokens(id) ON DELETE SET NULL,
    brand_profile_id         INTEGER REFERENCES brand_profiles(id) ON DELETE SET NULL,
    upload_path              VARCHAR(8) NOT NULL,
    publish_kind             VARCHAR(8) NOT NULL,
    thumbnail_source         VARCHAR(24),
    thumbnail_r2_key         TEXT,
    branded_artifact_r2_key  TEXT,
    status                   VARCHAR(20) NOT NULL DEFAULT 'queued',
    attempts                 INTEGER NOT NULL DEFAULT 0,
    last_error               TEXT,
    idempotency_key          VARCHAR(64) NOT NULL UNIQUE,
    publish_version          VARCHAR(40) NOT NULL,
    youtube_video_id         VARCHAR(32),
    predicted_quota_units    INTEGER NOT NULL,
    predicted_credit_cost    INTEGER NOT NULL,
    actual_quota_units       INTEGER,
    bytes_uploaded           BIGINT NOT NULL DEFAULT 0,
    upload_uri               TEXT,
    priority_at_dispatch     VARCHAR(16),
    dispatched_at            TIMESTAMP WITH TIME ZONE,
    started_at               TIMESTAMP WITH TIME ZONE,
    finished_at              TIMESTAMP WITH TIME ZONE,
    created_at               TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    -- Brief §2: Shorts cannot carry a thumbnail (YouTube API restriction).
    CONSTRAINT ck_thumbnail_video_only
        CHECK ( (publish_kind = 'short'
                  AND thumbnail_source IS NULL
                  AND thumbnail_r2_key IS NULL)
                OR publish_kind = 'video' )
);
CREATE INDEX IF NOT EXISTS ix_upload_jobs_v2_publish_task_id
    ON upload_jobs_v2 (publish_task_id);
CREATE INDEX IF NOT EXISTS ix_upload_jobs_v2_channel_id
    ON upload_jobs_v2 (channel_id);
CREATE INDEX IF NOT EXISTS ix_upload_jobs_v2_oauth_token_id
    ON upload_jobs_v2 (oauth_token_id);
CREATE INDEX IF NOT EXISTS ix_upload_jobs_v2_brand_profile_id
    ON upload_jobs_v2 (brand_profile_id);
CREATE INDEX IF NOT EXISTS ix_upload_jobs_v2_upload_path
    ON upload_jobs_v2 (upload_path);
CREATE INDEX IF NOT EXISTS ix_upload_jobs_v2_status
    ON upload_jobs_v2 (status);
CREATE INDEX IF NOT EXISTS ix_upload_jobs_v2_youtube_video_id
    ON upload_jobs_v2 (youtube_video_id);

-- ── 7. publish_attempts (depends on upload_jobs_v2) ────────────────
CREATE TABLE IF NOT EXISTS publish_attempts (
    id               SERIAL PRIMARY KEY,
    upload_job_id    INTEGER NOT NULL REFERENCES upload_jobs_v2(id) ON DELETE CASCADE,
    idempotency_key  VARCHAR(64) NOT NULL UNIQUE,
    youtube_video_id VARCHAR(32),
    status           VARCHAR(16) NOT NULL,
    attempt_no       INTEGER NOT NULL,
    worker_id        VARCHAR(64) NOT NULL,
    created_at       TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_publish_attempts_upload_job_id
    ON publish_attempts (upload_job_id);
CREATE INDEX IF NOT EXISTS ix_publish_attempts_created_at
    ON publish_attempts (created_at);

-- ── 8. credit_ledger (depends on users, upload_jobs_v2) ────────────
CREATE TABLE IF NOT EXISTS credit_ledger (
    id                    SERIAL PRIMARY KEY,
    user_id               INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    delta                 INTEGER NOT NULL,
    reason                VARCHAR(32) NOT NULL,
    upload_job_id         INTEGER REFERENCES upload_jobs_v2(id) ON DELETE SET NULL,
    path                  VARCHAR(8),
    publish_kind          VARCHAR(8),
    predicted_quota_units INTEGER,
    balance_after         INTEGER NOT NULL,
    created_at            TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_credit_ledger_user_id
    ON credit_ledger (user_id);
CREATE INDEX IF NOT EXISTS ix_credit_ledger_upload_job_id
    ON credit_ledger (upload_job_id);
CREATE INDEX IF NOT EXISTS ix_credit_ledger_created_at
    ON credit_ledger (created_at);

-- ── 9. quota_burn_log (depends on upload_jobs_v2) ──────────────────
CREATE TABLE IF NOT EXISTS quota_burn_log (
    id                     SERIAL PRIMARY KEY,
    upload_job_id          INTEGER REFERENCES upload_jobs_v2(id) ON DELETE SET NULL,
    operation              VARCHAR(40) NOT NULL,
    predicted_cost         INTEGER NOT NULL,
    observed_outcome       VARCHAR(20) NOT NULL,
    http_status            INTEGER,
    was_quota_exceeded     BOOLEAN NOT NULL DEFAULT FALSE,
    reconciled_actual_cost INTEGER,
    reconciled_at          TIMESTAMP WITH TIME ZONE,
    created_at             TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_quota_burn_log_upload_job_id
    ON quota_burn_log (upload_job_id);
CREATE INDEX IF NOT EXISTS ix_quota_burn_log_operation
    ON quota_burn_log (operation);
CREATE INDEX IF NOT EXISTS ix_quota_burn_log_created_at
    ON quota_burn_log (created_at);
```

### Notes

* All eight tables are STRICTLY additive — the legacy `upload_jobs`
  table is left untouched. Phase 3 ships a separate drop migration
  after the new path is the only path.
* `brand_profiles.owner_id` is intentionally a soft FK (no `REFERENCES`
  clause) because Postgres cannot enforce a single FK across three
  parent tables (`users` / `channels` / `oauth_tokens`).  The D-agent's
  resolver validates the reference at write time.
* `upload_jobs_v2.ck_thumbnail_video_only` is the database-level
  enforcement of the "Shorts cannot carry a thumbnail" rule from the
  brief.  The F-agent re-asserts this in application code as
  defense-in-depth.
* `users.plan_tier_id` stays nullable (column-level + FK SET NULL) so
  a deleted PlanTier row doesn't cascade-kill user accounts.  Should
  never happen in practice — the three seed rows live forever — but
  the SET NULL keeps the constraint honest.
* The Pro backfill is idempotent: re-running the `UPDATE` after Phase 1
  is a no-op because every row already has `plan_tier_id IS NOT NULL`.

### Rollback

Drop in REVERSE FK order.  This is **destructive** for any data already
written via the new path — only run if Phase 1 itself is being reverted
and no production user has placed a PublishTask yet.

```sql
DROP TABLE IF EXISTS quota_burn_log;
DROP TABLE IF EXISTS credit_ledger;
DROP TABLE IF EXISTS publish_attempts;
DROP TABLE IF EXISTS upload_jobs_v2;
DROP TABLE IF EXISTS publish_tasks;
DROP TABLE IF EXISTS brand_profiles;
DROP TABLE IF EXISTS master_videos;

ALTER TABLE users DROP CONSTRAINT IF EXISTS fk_users_plan_tier;
DROP INDEX IF EXISTS ix_users_plan_tier_id;
ALTER TABLE users DROP COLUMN IF EXISTS plan_tier_id;

DROP TABLE IF EXISTS plan_tiers;
```

Safe rollback path during the Phase 1 → Phase 2 window: the legacy
`upload_jobs` table is still alive and the legacy publish router
(`POST /api/clips/{clip_id}/publish`) is still always-on, so dropping
the new tables loses no production data.

---

## Insights / Trend Finder — channel analysis module (Phase 1: schema)

Adds four tables for the isolated `insights/` module (models in
`insights/models.py`, frozen by `scripts/test_insights_schema.py`). Read-only over
YouTube data; stores ingestion snapshots, per-video metrics, analysis runs, and
versioned reports. No change to any existing table.

```sql
CREATE TABLE IF NOT EXISTS insights_channel_snapshots (
    id                SERIAL PRIMARY KEY,
    user_id           INTEGER NOT NULL REFERENCES users(id)    ON DELETE CASCADE,
    channel_id        INTEGER REFERENCES channels(id)          ON DELETE SET NULL,
    google_channel_id VARCHAR(50)  NOT NULL,
    channel_title     VARCHAR(255) DEFAULT '',
    timezone          VARCHAR(64)  DEFAULT 'UTC',
    access_mode       VARCHAR(16)  NOT NULL DEFAULT 'public',   -- full | public
    subscriber_count  BIGINT  DEFAULT 0,
    video_count       INTEGER DEFAULT 0,
    view_count        BIGINT  DEFAULT 0,
    videos_ingested   INTEGER DEFAULT 0,
    channel_age_days  INTEGER,
    raw_cache_key     TEXT,
    raw_cached_at     TIMESTAMP WITH TIME ZONE,
    quota_predicted   INTEGER DEFAULT 0,
    quota_actual      INTEGER DEFAULT 0,
    status            VARCHAR(16) NOT NULL DEFAULT 'ok',        -- ok | partial | failed
    error             TEXT,
    created_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_insights_snap_user_id    ON insights_channel_snapshots (user_id);
CREATE INDEX IF NOT EXISTS ix_insights_snap_channel_id ON insights_channel_snapshots (channel_id);
CREATE INDEX IF NOT EXISTS ix_insights_snap_gcid       ON insights_channel_snapshots (google_channel_id);
CREATE INDEX IF NOT EXISTS ix_insights_snap_created_at ON insights_channel_snapshots (created_at);
CREATE INDEX IF NOT EXISTS ix_insights_snap_user_gcid  ON insights_channel_snapshots (user_id, google_channel_id);

CREATE TABLE IF NOT EXISTS insights_video_metrics (
    id                  SERIAL PRIMARY KEY,
    snapshot_id         INTEGER NOT NULL REFERENCES insights_channel_snapshots(id) ON DELETE CASCADE,
    video_id            VARCHAR(32) NOT NULL,
    title               TEXT DEFAULT '',
    description_short   TEXT DEFAULT '',
    tags                JSONB,
    published_at_utc    TIMESTAMP WITH TIME ZONE,
    published_at_local  TIMESTAMP WITH TIME ZONE,
    category_id         VARCHAR(16) DEFAULT '',
    thumbnail_url       TEXT DEFAULT '',
    duration_seconds    INTEGER DEFAULT 0,
    default_language    VARCHAR(16) DEFAULT '',
    is_short            BOOLEAN DEFAULT FALSE,
    view_count          BIGINT DEFAULT 0,
    like_count          BIGINT DEFAULT 0,
    comment_count       BIGINT DEFAULT 0,
    impressions               BIGINT,
    impressions_ctr           DOUBLE PRECISION,
    avg_view_seconds          DOUBLE PRECISION,
    avg_view_percentage       DOUBLE PRECISION,
    subscribers_gained        INTEGER,
    estimated_minutes_watched BIGINT,
    views_first_24h           BIGINT,
    views_first_48h           BIGINT,
    views_first_7d            BIGINT,
    traffic_sources           JSONB,
    views_per_hour_48h        DOUBLE PRECISION,
    reach_ratio               DOUBLE PRECISION,
    publish_dow               INTEGER,
    publish_hour_local        INTEGER,
    topic_cluster             VARCHAR(80),
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_insights_vm_snapshot_id ON insights_video_metrics (snapshot_id);
CREATE INDEX IF NOT EXISTS ix_insights_vm_video_id    ON insights_video_metrics (video_id);
CREATE INDEX IF NOT EXISTS ix_insights_vm_snap_video  ON insights_video_metrics (snapshot_id, video_id);

CREATE TABLE IF NOT EXISTS insights_analysis_runs (
    id             SERIAL PRIMARY KEY,
    snapshot_id    INTEGER NOT NULL REFERENCES insights_channel_snapshots(id) ON DELETE CASCADE,
    user_id        INTEGER NOT NULL REFERENCES users(id)                      ON DELETE CASCADE,
    maturity       VARCHAR(16) NOT NULL DEFAULT 'deep',     -- deep | starter | blend
    status         VARCHAR(16) NOT NULL DEFAULT 'ok',       -- ok | running | failed
    engine_version VARCHAR(16) DEFAULT 'v1',
    params         JSONB,
    results        JSONB,
    error          TEXT,
    created_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    finished_at    TIMESTAMP WITH TIME ZONE
);
CREATE INDEX IF NOT EXISTS ix_insights_runs_snapshot_id ON insights_analysis_runs (snapshot_id);
CREATE INDEX IF NOT EXISTS ix_insights_runs_user_id     ON insights_analysis_runs (user_id);
CREATE INDEX IF NOT EXISTS ix_insights_runs_created_at  ON insights_analysis_runs (created_at);

CREATE TABLE IF NOT EXISTS insights_report_versions (
    id              SERIAL PRIMARY KEY,
    analysis_run_id INTEGER NOT NULL REFERENCES insights_analysis_runs(id) ON DELETE CASCADE,
    user_id         INTEGER NOT NULL REFERENCES users(id)                  ON DELETE CASCADE,
    version         INTEGER DEFAULT 1,
    provider        VARCHAR(16) DEFAULT '',                 -- claude | gemini
    exec_summary    TEXT DEFAULT '',
    report_md       TEXT DEFAULT '',
    report_json     JSONB,
    tokens_in       INTEGER DEFAULT 0,
    tokens_out      INTEGER DEFAULT 0,
    cost_usd        DOUBLE PRECISION DEFAULT 0.0,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_insights_report_run_id  ON insights_report_versions (analysis_run_id);
CREATE INDEX IF NOT EXISTS ix_insights_report_user_id ON insights_report_versions (user_id);
CREATE INDEX IF NOT EXISTS ix_insights_report_created ON insights_report_versions (created_at);
CREATE INDEX IF NOT EXISTS ix_insights_report_run_ver ON insights_report_versions (analysis_run_id, version);
```

Dev/SQLite: created automatically by `Base.metadata.create_all()` once `main.py`
imports `insights.models` (wired in Phase 2 / ingestion). Rollback: `DROP TABLE` the
four `insights_*` tables — no existing table is touched, so nothing else is affected.
