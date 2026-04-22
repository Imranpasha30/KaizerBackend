# Kaizer News — Phase 4 Roadmap

> **Status**: Design doc + interface stubs only. No Phase 4 subsystem is
> implemented in the v1 codebase committed to date. Every import path in
> `pipeline_core/phase4/` exists so upstream code can be written against
> stable contracts, but real implementations raise `NotImplementedError`
> at call time.

## Context

Phases 1–3.5 shipped the **table-stakes platform + the narrative moat**
that competitors don't have:

| Commit | Phase | What shipped |
|---|---|---|
| dd09141 | 1 | Broadcast-quality encode, input/output QA gates |
| 7905f35 | 2A | Indic-script caption renderer, 3-candidate thumbnails |
| eb8c066 | 2B | Beat-aware B-roll, Kalman face-track, progress API |
| 0686c08 | 3.1 | Narrative Engine (ASR + shot + Gemini + snapping) |
| 62376ad | 3.2 | Six render modes (A–F incl. Full-Narrative + Trailer + Series) |
| d40dea2 | 3.3 | Cross-platform variants (YT/IG/TikTok) + Reels loop scoring |
| c79059f | 3.4 | Originality + safety guardrails (watermark/dup/cadence) |
| 278cd44 | 3.5 | Post-publish feedback loop + `/api/uploads/{id}/feedback` |

295 fast tests passing, 12 slow tests gated, zero regressions across all
eight commits.

Phase 4 is the **billion-dollar moat** — compounding-data assets,
high-margin revenue streams, and B2B expansion. Each subsystem is 4–12
weeks of real work. This document sets the implementation order and
scope boundaries so nobody re-designs mid-build.

---

## Implementation order (recommended)

```
┌─────────────────────────────────────────────────────────────┐
│ Tier 1 — data & revenue foundation (weeks 1–4)              │
│   1. training_flywheel   (data collection only, no retrain) │
│   2. creator_graph       (clip_edges table + insertion hooks) │
│   3. trial_reels         (decide_promotion wired to cron)   │
│                                                             │
│ Tier 2 — revenue expansion (weeks 5–10)                     │
│   4. agency_mode         ($199 tier unlock)                 │
│   5. music_marketplace   (high-margin attach)               │
│                                                             │
│ Tier 3 — power users + B2B (weeks 11–16)                    │
│   6. pro_export          (FCPX / Premiere XML)              │
│   7. vertical_packs      (podcast, gaming, finance)         │
│   8. regional_api        (Telugu newsroom plugin)           │
│                                                             │
│ Tier 4 — the actual retrain (weeks 17+)                     │
│   9. training_flywheel.retrain_narrative_scorer             │
│  10. training_flywheel.deploy_model                         │
└─────────────────────────────────────────────────────────────┘
```

The training flywheel's *collection* must ship in Tier 1 even though
*retraining* is Tier 4 — data accumulation starts on day 1 but
retraining needs ≥500 records in a niche before it pays off.

---

## Tier 1 details

### 1. Training Flywheel — collection

**Stub**: `pipeline_core/phase4/training_flywheel.py`

**DB changes required**:
- `training_records` table: upload_job_id, clip_id, niche, narrative_role,
  hook_score, completion_score, composite_score, views_48h,
  retention_curve JSONB, shares_per_reach, video_hash, collected_at.

**Triggers**:
- 48h after `UploadJob.status` transitions to `done`, a cron pulls
  `ClipPerformance` + `FeedbackReport` and calls
  `collect_training_record()`.
- 7d re-collection for longer-tail signals.

**Cost**: cheap. Just DB rows.

### 2. Creator Graph — edge insertion

**Stub**: `pipeline_core/phase4/creator_graph.py`

**DB changes**:
- `clip_edges(edge_type, src_clip_id, dst_clip_id, meta JSONB, created_at)`
  with a unique index on `(edge_type, src, dst)`.

**Insertion hooks**:
- `render_series.chain_parts` → insert `series_part_of` edges between
  consecutive Parts.
- `variants.generate_variants` → insert `variant_of` edges linking each
  platform variant to its master.
- `render_modes.render_mode_clip` in mode='trailer' → insert `trailer_for`
  edge when the caller passes `long_form_video_id`.
- `narrative.extract_narrative_clips` → insert `narrative_beat_of` edges
  linking ClipCandidates to their source Job.

**Product wins**:
- "Series completion report" UI: list Parts 1–N with per-part retention.
- "Trailer ROI": long-form watch-time uplift vs trailer publish time.

### 3. Trial Reels — the smallest high-leverage feature

**Stub**: `pipeline_core/phase4/trial_reels.py`

`decide_promotion` is pure and already tested in principle. The Phase 4
work is:
1. **OAuth**: Meta Graph API token flow (currently Kaizer is YT-first;
   add IG business-account connect via the existing `youtube/oauth.py`
   helper extended for Meta).
2. **Publish path**: when the user picks `publish_mode='trial_reel'`,
   the IG uploader sends `is_trial_reel=true` in the container-create
   call.
3. **24h cron**: scan UploadJob rows with `publish_kind='trial_reel'` +
   `status='done'` + `created_at <= now-24h`; fetch insights; run
   `decide_promotion`; call `promote_trial` on promote-verdicts.

Meta's thresholds from the Tester's research doc (Hootsuite 2026,
Buffer 2026): `shares_per_reach >= 1.5%` + `completion_pct >= 50%`.

---

## Tier 2 details

### 4. Agency Mode — $199 tier unlock

**Stub**: `pipeline_core/phase4/agency_mode.py`

**DB changes**: three tables, one role enum.
- `agency_teams(id, owner_user_id, name, branding JSONB, monthly_cap)`
- `agency_members(agency_id, user_id, role)`  (roles: owner/admin/creator/viewer)
- `agency_audit_log(agency_id, actor_user_id, action, target_kind, target_id, ts, details JSONB)`

**Middleware**: `check_permission(user, agency, action)` guards every
route. Actions: `clip.create, clip.publish, asset.upload, team.invite,
billing.view, ...`

**Billing hook**: monthly caps roll up to the agency; overages bill the
agency owner's Stripe customer.

### 5. Music Marketplace — high-margin attach

**Stub**: `pipeline_core/phase4/music_marketplace.py`

Partners to approach (pick 1–2 to start):
- **Epidemic Sound** — strongest catalogue + existing API for creators
- **Lickd** — popular-music licensing, premium tier
- **Uppbeat** — cheapest, freemium friendly

**Revenue math** at 10k daily publishes × $0.10 effective fee ≈ $30k/mo
recurring on top of SaaS base.

**Technical**:
- Catalogue cache refreshed nightly
- Fingerprint pre-check against YouTube Content ID (avoid double-flag)
- Attribution string auto-injected into UploadJob.description

---

## Tier 3 details

### 6. Pro Export — FCPX / Premiere XML

**Stub**: `pipeline_core/phase4/pro_export.py`

Generates editor-ready project files. FCPX XML spec:
<https://developer.apple.com/fcpxml/>. Premiere uses the Final Cut Pro 7
XML interchange format.

**Tracks to emit**:
- V1: rendered master (with optional source slices instead)
- V2: B-roll track from `pipeline_core.broll.BRollResult.insertions`
- V3: compound clip for CTA overlay (cta_overlay.apply_cta output)
- A1: main audio
- Captions: sidecar SRT or title track with styling

**Markers**: narrative turning points from `ClipCandidate.meta`.

### 7. Vertical Packs — podcast / gaming / finance

**Stub**: `pipeline_core/phase4/vertical_packs.py`

v1 ships the `news` pack inline (hook openers "breaking / just in /
exclusive / confirmed"). Phase 4 adds:

- **podcast**: longer clips (60–90s standalone), less burned-in caption
  density, "chapter moment" hook detection.
- **gaming**: frame-burst hook scoring (clutches, eliminations); sync
  captions to SFX peaks not speech.
- **finance**: number-in-hook bonus, "context box" graphic overlay in
  quote_card thumbnail.
- **fitness**: rep-counter-style pacing, music-BPM sync for B-roll.
- **parenting / beauty**: TBD based on demand signal.

Pack loading: `resources/vertical_packs/<niche>.yaml`.

### 8. Regional API — B2B newsroom plugin

**Stub**: `pipeline_core/phase4/regional_api.py`

Target customers: Telugu/Hindi/Tamil news desks (TV9, V6, Sakshi, NTV,
RTV, Mojo Story). B2B subscription ₹20k–30k/mo.

**Routes**: `/api/regional/ingest`, `/api/regional/jobs/{id}`,
`/api/regional/catalog`, `/api/regional/publish`.

**Auth**: Partner API keys + HMAC request signing on POST.
**Rate limit**: per-org RPM + monthly clip cap.

---

## Tier 4 details

### 9–10. Training Flywheel — retrain + deploy

Preconditions: ≥500 TrainingRecords per niche, rolling 90-day window.

**Architecture** (indicative):
- Fine-tune a small (~50M param) text+audio bi-encoder from the
  heuristic narrative scores toward observed 48h retention +
  shares-per-reach.
- Per-niche adapter via LoRA when a niche has >5k records; otherwise
  use the global model.
- Model registry: `s3://kaizer-models/narrative-scorer/v{N}/`. Atomic
  swap via a `narrative.py` registry lookup at compose time.
- Rollback: if the next 48h of new clips shows a retention drop ≥10%
  relative to the prior snapshot, auto-revert.

**Ops**: new weekly retrain schedule; observability dashboard showing
score-vs-retention calibration curves.

---

## Non-goals in Phase 4

Explicitly out of scope:
- Rebuilding Phase 3.1's narrative engine from scratch. The heuristic
  engine ships with enough headroom; Phase 4 refines its scoring, not
  its algorithm.
- Multi-modal foundation model training. Fine-tuning existing OSS
  encoders (SigLIP, BEATs) is in-scope; full pre-training is not.
- Social graph features (creator-to-creator messaging). Out of scope
  — focus on creator-to-audience tooling.
- International expansion beyond India + diaspora for regional API.
  English/Spanish packs can be added in Phase 5.

---

## Metrics to watch

| Subsystem | Early signal | Kill criterion |
|---|---|---|
| Training flywheel | Record volume hitting 500/niche in 30d | <100 records in 60d → reconsider niches |
| Creator graph | Edge insertion lag <100ms p99 | Lag >500ms → move to async insert |
| Trial reels | Promotion decisions / day | Decisions <5/day → defer to Tier 3 |
| Agency mode | Paid Agency seats >10 in 90d | <3 in 90d → kill tier, refund |
| Music marketplace | Licenses granted / day × avg fee | <$100/day gross by month 3 → drop partner |
| Pro export | DAU of export feature | <1% of paid base → keep as nice-to-have, don't iterate |
| Vertical packs | Retention uplift per niche | <5% uplift vs global pack → remove pack |
| Regional API | Partner LoIs signed in 90d | 0 partners in 90d → defocus |

---

## How this doc is maintained

- Every Phase 4 subsystem shipping gets a new entry in the commit table
  at the top + a status note in its section.
- `NotImplementedError` messages in `pipeline_core/phase4/*.py` all
  point back to the specific section of this doc.
- UI / routers depending on Phase 4 functions handle `NotImplementedError`
  and surface a "Coming soon" state rather than 500ing.

Last updated when Phase 4 stubs were committed. This file is the source
of truth for scope; keep it in sync.
