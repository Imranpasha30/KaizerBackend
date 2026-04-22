# Kaizer News — User Manual

A complete guide for operating, running, and extending the Kaizer News
video pipeline. Written for a new team member who has never touched the
codebase before.

> **Last updated**: Phase 5 ship day. 15 commits on main, 446 tests
> passing. Current state: local storage mode; R2 credentials are
> pre-wired in `.env` and can be activated by flipping one env var.

---

## Table of Contents

1. [What Kaizer News Is](#1-what-kaizer-news-is)
2. [Quick Start (run it locally in 5 minutes)](#2-quick-start)
3. [Core Concepts](#3-core-concepts)
4. [Step-by-Step Usage](#4-step-by-step-usage)
5. [The Beta Editor (animated effects)](#5-the-beta-editor)
6. [Feedback Loop + Explainability](#6-feedback-loop)
7. [Phase 4 Features](#7-phase-4-features)
8. [Configuration Reference (`.env`)](#8-configuration-reference)
9. [Troubleshooting](#9-troubleshooting)
10. [API Reference](#10-api-reference)
11. [Architecture Overview](#11-architecture-overview)
12. [Project Structure](#12-project-structure)

---

## 1. What Kaizer News Is

A professional video-pipeline SaaS that turns long-form Telugu / Hindi /
Tamil news broadcasts (or any long-form content) into publishable YouTube
Shorts, Instagram Reels, and TikTok videos — with narrative-aware
cutting, Indic-script captions, per-destination branding, and automatic
cross-platform variants.

**The moat**: every competitor (OpusClip, Submagic, Klap) picks clips by
keyword density and ends them at timestamps. Kaizer uses a narrative
engine that picks **payoffs, not setups**, and ends clips at **story
beats**, not timestamps. Plus the only pro-editor beta mode in the
category with animated effects, and the only originality-guardrail
system that warns creators before platform de-dup flags hit them.

Six render modes cover every creator workflow:
- **Standalone** — one self-contained 45-60s clip
- **Trailer** — 30-50s with a "watch full video" CTA + YT Related Video attach
- **Series** — chained Part 1/Part 2 with pinned-comment automation
- **Promo** — 15-25s ad-style
- **Highlight** — single viral moment (OpusClip parity)
- **Full Narrative** — one 60-180s clip that narrates the whole source

---

## 2. Quick Start

### 2.1 Prerequisites (install once)

- **Python 3.10+** (the venv at `e:/kaizer new data training/venv/` has 3.10.11)
- **Node 18+** and npm
- **FFmpeg 6+** on PATH (run `ffmpeg -version` to check)
- **Postgres** OR SQLite (SQLite is default in dev)

### 2.2 Clone + install

The repo lives at `e:/kaizer new data training/`. Inside it:

```bash
# Backend
cd "e:/kaizer new data training/kaizer/KaizerBackend"
"e:/kaizer new data training/venv/Scripts/python.exe" -m pip install -r requirements.txt

# Frontend
cd "e:/kaizer new data training/kaizer/kaizerFrontned"
npm install
```

### 2.3 Start the stack

Two terminals, two commands.

**Terminal 1 — backend**:
```bash
cd "e:/kaizer new data training/kaizer/KaizerBackend"
"e:/kaizer new data training/venv/Scripts/python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000
```
Backend comes up in ~10 seconds. API root at `http://127.0.0.1:8000`.

**Terminal 2 — frontend**:
```bash
cd "e:/kaizer new data training/kaizer/kaizerFrontned"
npm run dev
```
Frontend comes up at `http://localhost:3000`.

Or use the convenience scripts at `e:/kaizer new data training/kaizer/start_backend.bat` and `start_frontend.bat`.

### 2.4 Verify it's alive

Open `http://localhost:3000` — log in or register. You should see the
Kaizer dashboard with a nav for Home, New Job, Assets, Channels, etc.

Test the new editor API is reachable:
```bash
curl http://127.0.0.1:8000/api/editor/styles
```
Expect a JSON array of 5 style packs (`minimal`, `cinematic`,
`news_flash`, `vibrant`, `calm`).

---

## 3. Core Concepts

### 3.1 The Pipeline

```
Source video
    → Validator (codec/resolution/duration/fps gate)
    → ASR (Whisper — Telugu-fine-tuned for te, whisper-small otherwise)
    → Shot detection (FFmpeg scdet + cut_point.onnx)
    → Audio RMS valleys (natural B-roll insertion points)
    → Narrative engine (Gemini turning-point labeling + heuristic scoring)
    → Clip boundary snapping (to shot + sentence + valley)
    → Render mode template (one of 6)
    → Compose (torn_card / follow_bar / split_frame layouts)
    → CTA overlay
    → Per-platform variants (YT Shorts, IG Reels, TikTok)
    → Guardrails (watermark, duplicate, repetition, cadence)
    → QA gate (bitrate, loudness, pix_fmt, aspect)
    → Storage upload (local today, R2 when env flipped)
    → YouTube upload worker (picks up, optionally downloads from R2, publishes)
```

### 3.2 Narrative Engine

Unlike competitors' keyword scoring, the engine scores clips by:
- **Hook strength** (first 3s: audio energy delta, face-on-frame, question words, leading digit)
- **Completion** (clip ends on terminal punctuation + no dangling discourse markers)
- **Importance** (Gemini's narrative role label: setup / turn / climax / coda, or 0.5 neutral on fallback)

Composite score = weighted blend, weights differ per render mode (trailer mode privileges hook 0.5; standalone balances 0.4/0.3/0.3).

### 3.3 Cross-Platform Variants

One rendered master → three platform MP4s:

| Platform | Bitrate | Safe zone (1080×1920) | CTA default | Loop engineered |
|---|---|---|---|---|
| YouTube Shorts | 8 Mbps | (60, 120, 900, 1520) | "Watch full video on my channel →" | No |
| Instagram Reels | 7.5 Mbps @ 30fps strict | (60, 250, 840, 1350) | "Send this to someone who…" | **Yes** (pHash+audio+motion) |
| TikTok | 6 Mbps | (60, 150, 860, 1420) | "Follow for more ⇢" | No |

### 3.4 Guardrails (pre-publish safety checks)

Five checks each return `info` / `warn` / `block` alerts:
- **Watermark detector** — OpenCV template match against TikTok/CapCut/Snap/YT Shorts
- **Self-duplicate** — pHash of keyframes vs user's prior uploads (≥70% similarity = warn)
- **Template repetition** — last 5 titles' 3-gram overlap (≥80% = warn)
- **Cadence governor** — <6h between Reels OR >4/week = warn
- **Music rights** — stub today; lights up when a fingerprint DB is wired

Run all at once via `guardrails.run_all_guardrails(video, user_id=..., platform=..., db=db)` → `GuardrailsReport`.

---

## 4. Step-by-Step Usage

### 4.1 Connecting a YouTube channel

1. Navigate to `/channels` in the frontend.
2. Click **Add Channel** → fill the style-profile form (name, title formula, desc style, footer, hashtags).
3. Click **Connect YouTube** → OAuth flow redirects to Google, you pick which YT channel to authorize, returns.
4. You'll land back on `/channels` with the channel marked connected. The OAuth token, refresh token, channel metadata (subs, views, avatar URL) are cached in `oauth_tokens` table.

### 4.2 Uploading a source video

1. Navigate to `/new` (New Job).
2. Choose platform (Shorts / Reels / TikTok / Long).
3. Choose frame layout (torn_card / follow_bar / split_frame).
4. Upload the source file. Max 2 hours, 4K max resolution. The validator rejects unsupported codecs with HTTP 400.
5. The pipeline starts. Watch progress at `/jobs/{job_id}` — stages tick through (Transcribing 15% → Detecting clips 25% → Composing 65% → QA 92% → done 100%).

### 4.3 Reviewing clips

Go to `/jobs/{job_id}`. You'll see generated clips with:
- Video preview
- Auto-generated SEO (title, description, tags, hashtags)
- 3 thumbnail candidates (face_lock / quote_card / punch_frame)
- Narrative scoring badges (hook / completion / composite)

Click any clip to open the **Editor** at `/jobs/{job_id}/edit/{clipId}`.

### 4.4 Editing a clip

In the editor you can:
- Trim start/end timecodes
- Edit title / description / tags / hashtags
- Pick a different thumbnail candidate
- Set a target platform

Click **Save** to persist, **Export** to generate the final MP4.

### 4.5 Publishing

Click **Publish** → a modal opens. You pick:
- **Preset**: Global (all connected YT accounts) / Individual / a Channel Group
- **Privacy**: public / unlisted / private
- **Schedule** (optional): pick a time; queued in `upload_jobs`

The upload worker (running automatically in the backend's asyncio loop)
picks each `queued` row every 3 seconds, applies per-destination logo
overlay via FFmpeg, mints fresh OAuth credentials, does a resumable
YouTube upload, applies the thumbnail, marks `done`.

Watch progress at `/uploads`.

---

## 5. The Beta Editor

A pro-editor mode with animated effects. Access via the **Beta** button
(pulsing NEW badge) next to the Export button in the regular editor.

### 5.1 The 5 style packs

| Pack | What it looks like |
|---|---|
| **Minimal** | Clean cuts, no color grade. Good when content carries itself. |
| **Cinematic** | Warm color grade, gentle Ken Burns, fade transitions, bounce-in text. |
| **News Flash** | Urgent red/warm push, whip-pan transitions, typewriter captions. |
| **Vibrant** | High-saturation, zoom-punch cuts, word-pop captions. |
| **Calm** | Cool-blue grade, slow dissolves, sliding captions. |

### 5.2 How it feels

- Five large gradient cards in a horizontal scroll row (Instagram-Reel-picker vibe)
- Hover a card → it lifts and glows
- Click a card → it scales up + accent border
- Optional **hook text** field (e.g. "BREAKING") that gets animated onto the clip
- Pick platform → safe-zone for captions adjusts automatically
- Click **Render Beta** → spinner → result slides in
- **Side-by-side synced video players** show current (no effects) vs beta (with effects). Play/pause/seek sync across both.
- **Effect chips** below the beta player list exactly what was applied: `color_grade:cinematic_warm`, `motion:ken_burns_in`, `text_animation:bounce_in`, etc.
- **QA pill** (green ✓ or amber warning) next to render time.

### 5.3 Under the hood

- Effects live in `pipeline_core/effects/` (transitions, text_animations, color_grade, motion, style_packs)
- Orchestrator: `pipeline_core/editor_pro.py::render_beta(master, *, style_pack, hook_text, …)`
- API: `POST /api/editor/render-beta`, response carries `current_url`, `beta_url`, `effects_applied`, `qa_ok`, `warnings`, `render_time_s`
- Result cached at `output/beta_renders/clip_{id}/latest.json` so `GET /api/editor/render-beta/{clip_id}` survives restarts

---

## 6. Feedback Loop

Seven days after publish, Kaizer can pull retention + shares data and
produce actionable next-render advice.

### 6.1 Endpoint

```
GET /api/uploads/{upload_job_id}/feedback
```

Response:
```json
{
  "status": "ready" | "no_analytics" | "upload_not_found",
  "retention_curve": [{"t_pct": 0, "retention_pct": 100}, ...],
  "dropoffs": [{"t_pct": 25, "drop_pct": 18, "severity": "major",
                "likely_causes": ["weak hook — first-3s failed to retain",
                                  "face not in frame by ~1.5s"]}],
  "recommendations": [
    {"kind": "hook", "message": "Hook score was 0.43; 30% dropped by 5%. Try face-forward first frame...",
     "actionable": true}
  ],
  "explainability": {"narrative_role": "climax", "hook_score": 0.43,
                     "completion_score": 0.71, "composite_score": 0.68}
}
```

### 6.2 How drop-off causes are inferred

- Drops in the **first 15%** of the clip → weak hook
- Drops in the **middle 15-60%** → pacing dip
- Drops in the **last 40%** → CTA too early OR payoff mismatch

Combined with the clip's saved narrative score, recommendations become
specific: e.g. "completion_score 0.3 + late drop = clip ended mid-
sentence — rely more on sentence-boundary snap next time."

### 6.3 Current limitation

The `fetch_retention_from_youtube` function is a v1 stub — it returns
`[]` even with a key. Real YT Analytics API integration is a Phase 4
Tier 3 task. For now, you can test the loop by calling
`generate_feedback_report(upload_job_id, db, retention_override=curve)`
with a hand-constructed retention curve.

---

## 7. Phase 4 Features

Six subsystems that live in `pipeline_core/phase4/`. All either fully work today or have stable interface stubs awaiting external credentials / accumulated data.

| Subsystem | Status | Notes |
|---|---|---|
| `training_flywheel` — data collection | ✅ live | `collect_training_record(upload_job_id, db)` writes real rows. Feeds future narrative-scorer retraining. |
| `training_flywheel` — retrain + deploy | Stub | Needs ≥500 TrainingRecords/niche first. |
| `creator_graph` — storage + link + traverse | ✅ live | `link_clips(src, dst, edge_type=...)`, `traverse(clip_id, edge_type=..., direction=...)`. 5 edge types: series_part_of, trailer_for, variant_of, reusable_source, narrative_beat_of. |
| `agency_mode` — RBAC | ✅ live | `create_team`, `add_member`, `check_permission`, `record_audit_entry`. Role hierarchy: owner > admin > creator > viewer. `.read` suffix always allows viewers. |
| `pro_export` — FCPX / Premiere XML | ✅ live | `export_project(source_path, markers=, broll_tracks=, caption_srt_path=, output_path=, format='fcpxml' | 'prproj_xml')`. Emits valid XML parseable by Apple FCP / Adobe Premiere. |
| `music_marketplace` | Stub | Needs Epidemic / Lickd / Uppbeat partner credentials. |
| `trial_reels` — `decide_promotion` | ✅ live | Pure function — promotes if shares/reach ≥ 1.5% + completion ≥ 50% + 24h elapsed. |
| `trial_reels` — live API | Stub | Needs Meta dev account + IG business-account token flow. |
| `regional_api` — `authenticate` | ✅ live | SHA-256 key lookup against `regional_api_keys` table. |
| `regional_api` — `submit_ingest` | Stub | Render-pipeline hookup is Tier 3. |
| `vertical_packs` | ✅ news pack inline | `load_pack('news')` returns the Indian-news vertical pack with opener words (breaking / just in / exclusive / confirmed) + CTA templates. |

---

## 8. Configuration Reference

All config lives in `e:/kaizer new data training/kaizer/KaizerBackend/.env` (gitignored).

### Required
```ini
DATABASE_URL=postgresql://...   # Railway Postgres in prod, or sqlite:///kaizer.db in dev
GEMINI_API_KEY=...              # Narrative engine falls back to heuristic mode without it
OPENAI_API_KEY=...              # Optional: SEO fallback when Gemini is rate-limited
```

### YouTube OAuth (required for publishing)
```ini
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
KAIZER_OAUTH_REDIRECT_URI=http://localhost:8000/api/yt/oauth/callback
```

### Storage
```ini
STORAGE_BACKEND=local    # or 'r2' to activate Cloudflare R2
R2_BUCKET=kaizernews
R2_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_ENDPOINT=https://<account>.r2.cloudflarestorage.com
R2_PUBLIC_BASE_URL=https://pub-XXX.r2.dev   # Enable "Public Development URL" on the bucket
```

### Pipeline options
```ini
KAIZER_OUTPUT_ROOT=          # Defaults to <backend>/output/api_pipeline
KAIZER_DEFAULT_LOGO=         # Path to a default overlay logo; empty = no overlay
KAIZER_BAKE_LOGO_AT_RENDER=0 # 1 = bake logo into the master at render (legacy);
                             # 0 = per-destination overlay at upload time (preferred)
PEXELS_API_KEY=              # Optional: used for stock image fetching
```

### Optional (Phase 4)
```ini
YOUTUBE_ANALYTICS_API_KEY=   # Activates real retention fetch in feedback loop (stub without it)
KAIZER_AUTH_REQUIRED=1       # Enforce JWT on all authenticated routes; 0 = dev mode (legacy user)
```

---

## 9. Troubleshooting

### Backend won't start — import error

Most often `.env` is missing or a required key isn't set. Check:
```bash
cd "e:/kaizer new data training/kaizer/KaizerBackend"
ls -la .env   # should exist, ~3KB
```
If missing, copy the template from a teammate — the file is gitignored so it doesn't live in git.

### Frontend shows "Network Error" on API calls

The frontend expects the backend on port 8000. Check:
1. Backend terminal shows `Uvicorn running on http://127.0.0.1:8000`
2. `curl http://127.0.0.1:8000/api/editor/styles` returns JSON
3. Browser console — is the request failing with CORS? If so, check `main.py`'s CORS middleware allows your frontend origin.

### "No faces detected — emitting centre-crop fallback" in logs

Normal for footage without faces (B-roll heavy news clips). The
thumbnail's face_lock candidate falls back to a rule-of-thirds centre
crop with `score=0.05` so the user always sees 3 thumbnail options.

### Pipeline fails with `ffprobe: command not found`

FFmpeg/FFprobe must be on PATH. On Windows download the full build
from https://www.gyan.dev/ffmpeg/builds/ and add the `bin/` folder to
your user PATH. Restart the backend after.

### R2 uploads fail with `InvalidAccessKeyId`

The R2 API token was revoked or wrong. Regenerate in Cloudflare → R2 →
Manage API Tokens; paste new credentials into `.env`; restart backend.

### Gemini returns 429 "quota exceeded"

The SEO engine has a built-in fallback chain: gemini-2.5-flash →
gemini-1.5-flash → heuristic-only. Quota reset happens daily. For
production scale, use a paid Gemini key and raise the quota.

### Tests fail with `FutureWarning: google.generativeai deprecated`

Harmless warning; the package still works. Will migrate to `google.genai`
when we hit actual breakage. Does not affect test passes.

---

## 10. API Reference

### Editor Beta
- `GET /api/editor/styles` — list the 5 style packs
- `POST /api/editor/render-beta` — render a clip with effects applied
- `GET /api/editor/render-beta/{clip_id}` — most-recent cached render

### Jobs + Progress
- `POST /api/jobs` — create a new pipeline job
- `GET /api/jobs` — list jobs
- `GET /api/jobs/{job_id}` — job detail (includes clips)
- `GET /api/jobs/{job_id}/progress` — real-time progress bar data

### Feedback (post-publish analytics)
- `GET /api/uploads/{upload_job_id}/feedback` — retention + dropoff + recommendations

### Channels + YouTube OAuth
- `GET /api/channels` — list the user's style-profile channels
- `POST /api/channels` — create a new style profile
- `POST /api/channel-groups` — create a publish preset
- `POST /api/yt/oauth/init` — start OAuth flow
- `GET /api/yt/oauth/callback` — OAuth callback (YT redirects here)
- `GET /api/yt/accounts` — connected YT accounts + cached metadata
- `POST /api/yt/accounts/{channel_id}/refresh` — refresh cached YT stats
- `POST /api/yt/accounts/{channel_id}/logo` — set per-account overlay logo

### Uploads
- `POST /api/uploads` — schedule a clip for publish
- `GET /api/uploads` — list queued + running + done upload jobs
- `POST /api/uploads/{id}/cancel` — cancel a queued upload

### Billing
- `GET /api/billing/plans` — 4 tiers: Free, Creator $19, Pro $49, Agency $199
- `GET /api/billing/me` — current user's plan + usage
- `POST /api/billing/dev/set-plan` — dev-only plan override

### Assets
- `GET /api/assets` — list uploaded images/logos
- `POST /api/assets` — upload an image
- `POST /api/assets/folders` — virtual folder
- `POST /api/assets/{id}/move` — move asset to folder

### Trending + Performance
- `GET /api/trending` — latest competitor-channel videos for topic intel
- `GET /api/performance` — clip performance trajectories

---

## 11. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  Frontend (React + Vite)     http://localhost:3000                  │
│    src/App.jsx routes → src/pages/*.jsx                             │
│    src/api/client.js — JWT-authenticated fetch wrapper              │
│    src/components/ — StylePackCard, SyncedVideoPair, LogoPicker,    │
│                      PublishModal, YouTubeAccountsPanel, etc.       │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ HTTP (fetch + JWT Bearer)
┌──────────────────────────────────▼──────────────────────────────────┐
│  FastAPI Backend             http://127.0.0.1:8000                  │
│    main.py → app.include_router(...)                                │
│    routers/auth, channels, seo, youtube_oauth, youtube_upload,      │
│            campaigns, performance, translation, trending, assets,   │
│            veo, channel_groups, billing, job_progress, feedback,    │
│            editor                                                    │
│                                                                      │
│    pipeline_core/ — the engine                                       │
│      pipeline.py        — top-level run_pipeline                     │
│      validator.py, qa.py — input/output gates                        │
│      asr.py, shot_detect.py, narrative.py, clip_boundaries.py       │
│      render_modes.py, render_series.py, cta_overlay.py              │
│      variants.py, loop_score.py                                      │
│      captions.py, thumbnails.py, broll.py, face_track.py            │
│      guardrails.py, feedback_loop.py                                 │
│      editor_pro.py + effects/ (transitions, text anim, colour, etc.) │
│      storage.py (R2 + Local)                                         │
│      phase4/ (training_flywheel, creator_graph, agency_mode,         │
│               pro_export, vertical_packs, regional_api, trial_reels) │
│                                                                      │
│    youtube/worker.py — async upload worker, resumable session URI    │
│    runner.py — subprocess runner for pipeline jobs                   │
└───────────┬───────────────────────┬─────────────────────────────────┘
            │                       │
            │                       │
┌───────────▼──────────┐  ┌─────────▼─────────────────────────────┐
│  Postgres / SQLite   │  │  Storage                              │
│  models.py (18+ tbls)│  │    STORAGE_BACKEND=local: output/*    │
│                      │  │    STORAGE_BACKEND=r2:    R2 bucket   │
│  Clip, Job,          │  │       kaizernews/clips/*              │
│  UploadJob, Channel, │  │       + beta_renders/*                │
│  Campaign,           │  │                                       │
│  ClipPerformance,    │  │  /media FastAPI mount serves local    │
│  TrainingRecord,     │  │       files to the frontend           │
│  ClipEdge,           │  │                                       │
│  AgencyTeam, …       │  │                                       │
└──────────────────────┘  └───────────────────────────────────────┘
```

**External services called by the backend:**
- Gemini (narrative labeling + SEO generation)
- YouTube Data API v3 (upload, metadata refresh)
- Google OAuth 2.0 (channel connect)
- Pexels (stock images, optional)
- Cloudflare R2 (storage, when enabled)

---

## 12. Project Structure

```
e:/kaizer new data training/
├── kaizer/
│   ├── README.md
│   ├── start_backend.bat
│   ├── start_frontend.bat
│   ├── KaizerBackend/              ← backend Python app (own git repo)
│   │   ├── main.py                  ← FastAPI entry + schema migrations
│   │   ├── models.py                ← SQLAlchemy tables (~20 models)
│   │   ├── database.py              ← engine + SessionLocal
│   │   ├── auth.py                  ← JWT middleware
│   │   ├── runner.py                ← subprocess wrapper for pipeline jobs
│   │   ├── requirements.txt
│   │   ├── requirements-railway.txt ← lean deploy deps (no torch heavy)
│   │   ├── .env                     ← gitignored — all secrets live here
│   │   ├── pipeline_core/           ← all rendering/AI modules
│   │   ├── routers/                 ← FastAPI routers, one per API area
│   │   ├── seo/                     ← SEO generator + composer
│   │   ├── youtube/                 ← OAuth, upload worker, logo_overlay
│   │   ├── learning/                ← corpus scheduler for channel learning
│   │   ├── billing/                 ← plan definitions + usage tracking
│   │   ├── analytics/               ← trending + performance utilities
│   │   ├── resources/               ← fonts, watermark_templates
│   │   ├── output/                  ← pipeline outputs (gitignored)
│   │   ├── tests/                   ← 446+ pytest cases
│   │   └── docs/
│   │       ├── PHASE4_ROADMAP.md
│   │       └── USER_MANUAL.md       ← this file
│   └── kaizerFrontned/              ← React frontend (own git repo)
│       ├── src/
│       │   ├── App.jsx
│       │   ├── pages/               ← Home, Editor, EditorBeta, Channels, …
│       │   ├── components/          ← StylePackCard, SyncedVideoPair, …
│       │   ├── api/client.js
│       │   └── index.css
│       ├── package.json
│       └── vite.config.js
├── venv/                            ← Python virtualenv (not in git)
└── models/                          ← pretrained models — whisper-telugu,
                                       cut_point.onnx, moment_detector.pt, …
```

---

## Support + Extending

- **Adding a render mode**: append to `render_modes.RENDER_MODE_CONFIGS` + the mode-dispatch in `render_mode_from_narrative`.
- **Adding a style pack**: append to `effects/style_packs.STYLE_PACKS`.
- **Adding a guardrail check**: write a new `check_*` function in `guardrails.py` + wire into `run_all_guardrails`.
- **Adding a vertical pack**: append to `phase4/vertical_packs._BUILTIN_PACKS` or drop a YAML under `resources/vertical_packs/<niche>.yaml` (YAML loader coming in Phase 4 Tier 3).
- **Changing encode quality**: edit `pipeline_core/pipeline.ENCODE_ARGS_SHORT_FORM` (final) or `ENCODE_ARGS_INTERMEDIATE` (raw cuts — no loudnorm).

Happy shipping.
