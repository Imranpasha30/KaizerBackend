# Pipeline V2 — Beta operations runbook

> Phase 14 / Step 13 — written 2026-05-19.
> Operator: solo founder; on-call = same human.
> Audience = future-me + any hire who picks this up.

This is the **first-response playbook** for V2 Beta. If V2 is on
fire, start here.

---

## 1. Quick reference

| Question | Answer |
|---|---|
| **How do I disable V2 right now?** | `KAIZER_V2_ENABLED=0` in prod `.env`, restart backend. ~2 min. See §3. |
| **Where do I see V2 runs?** | Inngest Cloud dashboard → kaizer app → fn `process_video_v2`. |
| **Where do I check spend?** | Deepgram dashboard + GCP billing console (item 62: internal ledger lies). |
| **Where do failures show up?** | `Job.status='failed'` rows + log lines starting with `permanent:`. |
| **What's the daily Gemini cap?** | $50 (set in GCP console — physically blocks runaway). |
| **How do I cancel a stuck V2 run?** | Inngest Cloud dashboard → run → "Cancel" button. |

---

## 2. Pre-launch checklist (run once per environment)

Run the preflight script as the production user:

```bash
python pipeline_v2/scripts/preflight_v2_launch.py
```

Required env vars (D-13.2 locked list):

| Var                              | Required? | Value                                                        |
|----------------------------------|-----------|--------------------------------------------------------------|
| `KAIZER_V2_ENABLED`              | YES       | `1`                                                          |
| `INNGEST_EVENT_KEY`              | YES       | real prod key from Inngest Cloud                             |
| `INNGEST_SIGNING_KEY`            | YES       | real prod key from Inngest Cloud                             |
| `INNGEST_DEV`                    | YES       | `0` or unset                                                 |
| `KAIZER_STT_DEFAULT_PROVIDER`    | YES       | `deepgram` (backlog item 59 — Whisper-Groq broken on Telugu) |
| `DEEPGRAM_API_KEY`               | YES       | Stage 1 transcription                                        |
| `GEMINI_API_KEY`                 | YES       | Stages 2/2.5/3a/3b/3c                                        |
| `OPENAI_API_KEY`                 | warn      | Stage 4 image fallback                                       |
| `CSE_API_KEY` + `CSE_CX`         | warn      | image search                                                 |

GCP / external setup:

1. **Set a hard Gemini API spend cap in Google Cloud console** —
   `$50/day` initial limit. This is the only mechanism that
   *physically prevents* runaway. Path: GCP console → Billing →
   Budgets & alerts → New budget → Scope: Gemini API → Threshold $50
   → Action: Disable billing on threshold. **Do this BEFORE flipping
   `KAIZER_V2_ENABLED=1`.**
2. **Bookmark these for daily checks:**
   - Deepgram dashboard (spend + minutes processed)
   - GCP billing console (Gemini spend; the internal ledger lies — item 62)
   - Inngest Cloud dashboard (run history; cancel-all button lives here)

---

## 3. Rollback procedure (D-13.5)

**When to use:** any V2 production issue serious enough to want it
off — Stage 2 prompt regression, mid-Stage-4 hang, runaway retry
loop, suspicious billing alert.

```
1. SSH/access prod host running KaizerBackend.
2. Edit .env (or environment config):
     KAIZER_V2_ENABLED=0
3. Restart the backend service:
     systemctl restart kaizer-backend     # or docker compose restart
4. Verify the V2 platform is gone from /api/platforms:
     curl https://<prod-host>/api/platforms/ | jq '. | keys'
     -> must return exactly:
        ["instagram_reel", "youtube_full",
         "youtube_full_plus_shorts", "youtube_short"]
     -> "full_video_shorts_v2" MUST be absent.
5. The /api/inngest webhook endpoint also unmounts on restart with
   KAIZER_V2_ENABLED=0; new Inngest events for fn
   process_video_v2 will return 404, which Inngest treats as a
   transient outage (not a function-removed signal). Existing
   in-flight runs continue until they complete.
6. To kill in-flight runs too, open Inngest Cloud → Runs →
   filter "fn:kaizer-v2-process-video-v2" + state:running →
   Cancel each. Inngest's cancel is cooperative + SIGKILL (Step
   12.3 verified), so ffmpeg children die within ~14s.
```

**Time to rollback: <2 minutes.**

**What rollback DOES NOT do:**
- Refund API spend on in-flight runs (use Inngest Cloud cancel for that).
- Migrate existing V2 Job rows out of the DB (they stay as `status='failed'` or whatever they were).
- Affect any of the 4 V1 platforms — V1 code path was never touched.

---

## 4. Daily monitoring (the 5-minute pass)

Run these checks once per day for the first 2 weeks of Beta.

### 4.1 SQL — Jobs status counts (last 24h)

```sql
SELECT
  status,
  COUNT(*) AS n,
  SUM(CASE WHEN cancel_requested THEN 1 ELSE 0 END) AS cancel_requested_count
FROM jobs
WHERE platform = 'full_video_shorts_v2'
  AND created_at > NOW() - INTERVAL '24 hours'
GROUP BY status
ORDER BY n DESC;
```

**What to look for:**
- `failed` count > `done` count → something's broken.
- `cancel_requested_count` high relative to `cancelled` count → cancellations slow to propagate.
- `running` rows older than 30 min → check Inngest dashboard for stuck steps.

### 4.2 SQL — Failure-by-slug

```sql
SELECT
  SUBSTRING(error FROM 'permanent:([a-z_]+)') AS slug,
  COUNT(*) AS n
FROM jobs
WHERE platform = 'full_video_shorts_v2'
  AND status = 'failed'
  AND created_at > NOW() - INTERVAL '24 hours'
GROUP BY slug
ORDER BY n DESC;
```

**What to look for:** any single slug spiking (>3 occurrences/day at
Beta scale) means a class of failure is recurring. Cross-reference
with the corresponding stage's prompt or code path.

### 4.3 Admin dashboard

Open `/admin/v2` in the browser. The page surfaces:
- Total V2 jobs all-users
- Status breakdown
- Failure breakdown by slug
- Average rating + rating distribution
- Cancellation rate %
- Recent feedback (paginated, click to open the job)

### 4.4 Spend checks

| Provider  | Where to look                                                                          | Daily expected (Beta)  |
|-----------|----------------------------------------------------------------------------------------|------------------------|
| Deepgram  | dashboard.deepgram.com → Usage                                                         | $0.01–0.30             |
| Gemini    | console.cloud.google.com → Billing → Reports → filter by service "Generative Language" | $1–5                   |
| OpenAI    | platform.openai.com → Usage                                                             | $0.50–2.00 (image gen) |

**Red flags:**
- Gemini > $20 in one day → check Inngest for retry storms.
- OpenAI > $5 in one day → check Stage 4 ImageSourcer for runaway fallbacks.
- Deepgram > $5 in one day → unusually high job volume OR very long sources.

---

## 5. Common failure modes

### 5.1 `permanent:empty_file`

**Cause:** uploaded video was 0 bytes (often partial upload mid-network drop).
**Action:** user re-uploads. No backend fix needed.

### 5.2 `permanent:ffmpeg_not_found`

**Cause:** ffmpeg missing from PATH on the prod host.
**Action:** `apt install ffmpeg` (or whatever the host's package manager is). Hard requirement.

### 5.3 `permanent:stt_failed` (Deepgram)

**Cause:** Deepgram returned a non-2xx, OR the transcript is empty.
**Action:** check Deepgram dashboard for outage. If transcript is empty, the audio track is genuinely silent — user issue.

### 5.4 `permanent:json_invalid` (Stage 2/2.5/3 Gemini)

**Cause:** Gemini returned malformed JSON despite the `response_mime_type='application/json'` directive.
**Action:** retry the job (Inngest retries are usually enough). If 3 retries all fail with this slug, the prompt may have drifted past Gemini's structured-output capacity — bisect with shorter input.

### 5.5 Stuck in `running` state past 30 min

**Cause:** Inngest step hung or worker crashed mid-step.
**Action:**
1. Open Inngest Cloud → find the run.
2. If the run state shows "running" → the step is still progressing (long Gemini Pro call). Wait.
3. If the run state shows "completed" but DB shows `running` → orchestrator's terminal write didn't fire. Manually update: `UPDATE jobs SET status='failed', error='Inngest completed but no terminal write' WHERE id=<id>`.
4. If the run state shows "failed" → check the function-error tab for the slug; it should already be in `Job.error`.

### 5.6 V2 dashboard shows feedback you can't trace

**Cause:** `JobFeedback.user_id` is SET-NULL when the user account is
deleted (Phase 15 migration). The feedback row survives for aggregate
stats but the email column shows "deleted".
**Action:** none — by design. The numbers are still load-bearing.

---

## 6. Cutover decision (D-13.9)

**Gating criteria — ALL three must hold:**

1. ≥ 2 weeks of Beta uptime.
2. ≥ 50 production V2 jobs (count visible at `/admin/v2`).
3. ≥ 0 P1 incidents (defined as: any event that required rollback) in the trailing 7 days.

When all three are met, Step 14 (default cutover) is a separate
decision — re-open this discussion deliberately rather than acting on
the gate alone.

---

## 7. Escalation tree

Solo founder Beta → escalation is self-contained.

If/when a hire joins, the escalation expectation becomes:

1. **First responder** (whoever is awake): check §4 (daily monitor).
2. **If unable to diagnose within 30 min**: trigger rollback (§3) and notify the operator.
3. **Operator**: drive root-cause analysis with the codebase + backlog.

Never let a V2 issue burn for more than 30 min without rollback —
the V1 code path is unaffected; rollback is cheap.

---

## 8. Where everything lives

| Thing                          | Location                                                                                       |
|--------------------------------|------------------------------------------------------------------------------------------------|
| V2 orchestrator                | `pipeline_v2/pipeline_v2/orchestrator.py`                                                      |
| V2 stages (1–4)                | `pipeline_v2/pipeline_v2/stages/`                                                              |
| V2 prompts                     | `pipeline_v2/pipeline_v2/prompts/`                                                             |
| V2→V1 adapter                  | `pipeline_v2/pipeline_v2/editor_meta_adapter.py`                                               |
| Job feedback model + endpoints | `models.py`, `main.py` (POST /feedback, /v2/stats), `routers/admin.py` (v2-feedback, v2-stats) |
| Migration ledger               | `docs/MIGRATIONS.md` Phases 13, 14, 15                                                         |
| Backlog (active + fixed)       | `pipeline_v2/post_v2_backlog.md`                                                               |
| Step 12 validation summary     | `pipeline_v2/step12_completion_summary.md`                                                     |
| This runbook                   | `pipeline_v2/RUNBOOK.md`                                                                       |
| Preflight script               | `pipeline_v2/scripts/preflight_v2_launch.py`                                                   |
| Admin dashboard                | `/admin/v2` in the UI                                                                          |
| User-facing stats              | `/v2-stats` in the UI                                                                          |
