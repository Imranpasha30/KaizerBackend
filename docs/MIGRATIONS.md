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
