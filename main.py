import os
import json
import time
import shutil
import mimetypes
import asyncio
import sys
from pathlib import Path
from typing import Optional

# Windows asyncio subprocess support requires ProactorEventLoop.
# uvicorn (>= 0.18) sets SelectorEventLoop on Windows when running with
# --workers > 1 (or even single-worker in some configurations), which
# does NOT implement subprocess_exec -- ``run_ffmpeg`` dies with
# ``NotImplementedError`` inside Stage 0's first transcode call.
# Force the policy explicitly BEFORE any code touches asyncio. Safe
# no-op on Linux (the attribute only exists on Windows builds).
if sys.platform == "win32" and hasattr(
    asyncio, "WindowsProactorEventLoopPolicy"
):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from dotenv import load_dotenv

# override=True so a `.env` edit + uvicorn restart actually replaces
# any stale value already present in the parent shell's environment.
# Without override, load_dotenv silently keeps the OS-level value when
# both exist, which led to "key still empty" surprises during dev.
load_dotenv(override=True)

from database import engine, SessionLocal, Base, get_db
import models
import runner
import auth

from routers.auth import router as auth_router
from routers.channels import router as channels_router
from routers.seo import router as seo_router
from routers.youtube_oauth import router as youtube_oauth_router
from routers.youtube_upload import router as youtube_upload_router
from routers.campaigns import router as campaigns_router
from routers.performance import router as performance_router
from routers.translation import router as translation_router
from routers.trending import router as trending_router
from routers.assets import router as assets_router
from routers.channel_groups import router as channel_groups_router
from routers.billing import router as billing_router
from routers.job_progress import router as job_progress_router
from routers.feedback import router as feedback_router
from routers.editor import router as editor_router
from routers.live_director import router as live_director_router
from routers.admin import router as admin_router
from routers.work_monitor import router as work_monitor_router
from routers.postiz import router as postiz_router
from routers.yt_lookup import router as yt_lookup_router
from routers.bulletin_images import router as bulletin_images_router
from routers.express_mode import router as express_mode_router
from routers.heygen import router as heygen_router
from routers.live_studio import router as live_studio_router
from seo.default_channels import seed_channels
from youtube import worker as upload_worker
from learning import scheduler as corpus_scheduler


def _migrate_schema():
    """Add missing columns to existing tables — safe to run on every startup."""
    from sqlalchemy import text, inspect
    with engine.connect() as conn:
        inspector = inspect(engine)

        # ── jobs table columns ──────────────────────────────────────────
        existing_jobs = {c["name"] for c in inspector.get_columns("jobs")}
        # NOTE on cancel_requested: SQLAlchemy declares the ORM-level
        # default=False but the DB column itself needs a DEFAULT clause
        # (PG rejects ADDing a NOT NULL column to a populated table
        # without one). Dialect-aware default:
        #   Postgres: BOOLEAN NOT NULL DEFAULT FALSE
        #   SQLite:   BOOLEAN NOT NULL DEFAULT 0
        _bool_false = "BOOLEAN NOT NULL DEFAULT FALSE" if engine.dialect.name == "postgresql" \
                                                       else "BOOLEAN NOT NULL DEFAULT 0"
        job_additions = {
            "frame_layout":     "VARCHAR(50)",
            "video_name":       "VARCHAR(255)",
            "error":            "TEXT",
            "language":         "VARCHAR(10) DEFAULT 'te'",
            "user_id":          "INTEGER",
            "started_at":       "DATETIME",
            "finished_at":      "DATETIME",
            "cancel_requested": _bool_false,
            # Step 10 (V2 Inngest orchestrator): per-step progress
            # for V2 jobs. NULL on V1 jobs (legacy subprocess path
            # doesn't write progress). NULL at finalize.
            "current_stage":    "VARCHAR(40)",
            # Phase 14 / V2 Beta (D-13.11): user-supplied human label.
            # NULL on pre-Phase-14 rows; create endpoint defaults to
            # first 80 chars of video_name when caller omits the field.
            "name":             "VARCHAR(120)",
            # Item 104 (Transition library): per-job transition choice.
            # Catalog lives in pipeline_v2.transitions. NULL on pre-
            # item-104 rows; renderer falls back to "smart_cut".
            "transition_style": "VARCHAR(20) DEFAULT 'smart_cut'",
        }
        for col, dtype in job_additions.items():
            if col not in existing_jobs:
                conn.execute(text(f"ALTER TABLE jobs ADD COLUMN {col} {dtype}"))

        # ── user_id on multi-tenant tables ──────────────────────────────
        for tbl in ("channels", "upload_jobs", "campaigns", "competitor_channels"):
            if inspector.has_table(tbl):
                cols = {c["name"] for c in inspector.get_columns(tbl)}
                if "user_id" not in cols:
                    conn.execute(text(f"ALTER TABLE {tbl} ADD COLUMN user_id INTEGER"))

        # ── users.heygen_avatar_id / heygen_voice_id (Trending → HeyGen) ──
        if inspector.has_table("users"):
            user_cols = {c["name"] for c in inspector.get_columns("users")}
            if "heygen_avatar_id" not in user_cols:
                conn.execute(text("ALTER TABLE users ADD COLUMN heygen_avatar_id VARCHAR(64)"))
            if "heygen_voice_id" not in user_cols:
                conn.execute(text("ALTER TABLE users ADD COLUMN heygen_voice_id VARCHAR(64)"))

        # ── live_streams.backup_expires_at (Live Studio 48 h preview) ──
        # Added late; live_streams table itself is created by
        # Base.metadata.create_all on first run. Use the cross-dialect
        # type name — Postgres needs ``TIMESTAMP WITH TIME ZONE`` (not
        # ``DATETIME``, which PG doesn't define).
        if inspector.has_table("live_streams"):
            ls_cols = {c["name"] for c in inspector.get_columns("live_streams")}
            if "backup_expires_at" not in ls_cols:
                dialect = engine.dialect.name
                ts_type = "TIMESTAMP WITH TIME ZONE" if dialect == "postgresql" else "DATETIME"
                conn.execute(text(
                    f"ALTER TABLE live_streams ADD COLUMN backup_expires_at {ts_type}"
                ))

        # ── channels.logo_asset_id (per-channel video overlay logo) ─────
        if inspector.has_table("channels"):
            cols = {c["name"] for c in inspector.get_columns("channels")}
            if "logo_asset_id" not in cols:
                conn.execute(text("ALTER TABLE channels ADD COLUMN logo_asset_id INTEGER"))
            # Per-channel upload route override.  Null = use system default.
            if "upload_provider" not in cols:
                conn.execute(text("ALTER TABLE channels ADD COLUMN upload_provider VARCHAR(20)"))

        # ── upload_jobs.upload_provider (per-publish override) ──────────
        if inspector.has_table("upload_jobs"):
            ujcols = {c["name"] for c in inspector.get_columns("upload_jobs")}
            if "upload_provider" not in ujcols:
                conn.execute(text("ALTER TABLE upload_jobs ADD COLUMN upload_provider VARCHAR(20)"))

        # ── user_assets.folder_path (virtual folders for organization) ──
        if inspector.has_table("user_assets"):
            cols = {c["name"] for c in inspector.get_columns("user_assets")}
            if "folder_path" not in cols:
                conn.execute(text("ALTER TABLE user_assets ADD COLUMN folder_path VARCHAR(255) DEFAULT ''"))
            # Fingerprint of the source video the asset was generated
            # from — lets the "you've used this video before, reuse
            # its images?" prompt work after a re-upload.
            if "source_video_hash" not in cols:
                conn.execute(text(
                    "ALTER TABLE user_assets ADD COLUMN source_video_hash VARCHAR(64) DEFAULT ''"
                ))

        # ── users.socials (cross-promo links for SEO footer) ────────────
        if inspector.has_table("users"):
            ucols = {c["name"] for c in inspector.get_columns("users")}
            if "socials" not in ucols:
                conn.execute(text("ALTER TABLE users ADD COLUMN socials TEXT DEFAULT '{}'"))

            # ── users: billing / subscription columns ────────────────────
            user_billing_additions = {
                "plan":                   "VARCHAR(20) DEFAULT 'free'",
                "plan_cycle":             "VARCHAR(10) DEFAULT 'monthly'",
                "plan_renews_at":         "TIMESTAMP NULL" if engine.dialect.name == "postgresql" else "DATETIME",
                "stripe_customer_id":     "VARCHAR(64)",
                "stripe_subscription_id": "VARCHAR(64)",
                "monthly_clip_count":     "INTEGER DEFAULT 0",
                "usage_reset_at":         "TIMESTAMP NULL" if engine.dialect.name == "postgresql" else "DATETIME",
            }
            for col, dtype in user_billing_additions.items():
                if col not in ucols:
                    conn.execute(text(f"ALTER TABLE users ADD COLUMN {col} {dtype}"))

        # ── profile_destinations: cached YouTube metadata for the
        # multi-channel picker (lets the UI show channel-title /
        # avatar / sub-count for every Brand Account without re-calling
        # channels.list on every render).
        if inspector.has_table("profile_destinations"):
            pdcols = {c["name"] for c in inspector.get_columns("profile_destinations")}
            pd_additions = {
                "channel_title":         "VARCHAR(255) DEFAULT ''",
                "channel_thumbnail_url": "VARCHAR(500) DEFAULT ''",
                "channel_custom_url":    "VARCHAR(100) DEFAULT ''",
                "subscriber_count":      "INTEGER DEFAULT 0",
                "video_count":           "INTEGER DEFAULT 0",
                "enabled":               ("BOOLEAN DEFAULT TRUE NOT NULL"
                                          if engine.dialect.name == "postgresql"
                                          else "BOOLEAN DEFAULT 1 NOT NULL"),
            }
            for col, dtype in pd_additions.items():
                if col not in pdcols:
                    conn.execute(text(
                        f"ALTER TABLE profile_destinations ADD COLUMN {col} {dtype}"
                    ))

        # ── profile_destinations — seed from existing OAuthTokens so
        # pre-existing 1:1 profile↔destination links become rows in the
        # new many-to-many table without the user having to re-link them.
        # Dialect-aware: SQLite uses INSERT OR IGNORE, Postgres uses
        # ON CONFLICT DO NOTHING on the (profile_id, google_channel_id) UK.
        if inspector.has_table("profile_destinations") and inspector.has_table("oauth_tokens"):
            dialect = engine.dialect.name
            # ``enabled`` is NOT NULL with no DB-level default — the
            # SQLAlchemy ``default=True`` only applies during ORM
            # inserts. This raw-SQL backfill has to set the value
            # explicitly or Postgres rejects the row.
            if dialect == "postgresql":
                conn.execute(text("""
                    INSERT INTO profile_destinations (profile_id, google_channel_id, enabled)
                    SELECT channel_id, google_channel_id, TRUE
                    FROM oauth_tokens
                    WHERE google_channel_id IS NOT NULL AND google_channel_id != ''
                    ON CONFLICT DO NOTHING
                """))
            else:
                conn.execute(text("""
                    INSERT OR IGNORE INTO profile_destinations (profile_id, google_channel_id, enabled)
                    SELECT channel_id, google_channel_id, 1
                    FROM oauth_tokens
                    WHERE google_channel_id IS NOT NULL AND google_channel_id != ''
                """))

        # Migrate old 'frame' → 'frame_layout' if needed
        if "frame" in existing_jobs and "frame_layout" in existing_jobs:
            conn.execute(text(
                "UPDATE jobs SET frame_layout = frame WHERE frame_layout IS NULL"
            ))

        # ── clips table columns ─────────────────────────────────────────
        existing_clips = {c["name"] for c in inspector.get_columns("clips")}
        clip_additions = {
            "thumb_path":    "VARCHAR(500)",
            "image_path":    "VARCHAR(500)",
            "frame_type":    "VARCHAR(50)",
            "text":          "TEXT",
            "card_params":   "TEXT",
            "section_pct":   "TEXT",
            "follow_params": "TEXT",
            "seo":           "TEXT",
            "seo_variants":  "TEXT DEFAULT '{}'",
        }
        for col, dtype in clip_additions.items():
            if col not in existing_clips:
                conn.execute(text(f"ALTER TABLE clips ADD COLUMN {col} {dtype}"))

        # ── clips: Phase 5 storage columns ──────────────────────────────
        clip_storage_additions = {
            "storage_url":     "VARCHAR(500) DEFAULT ''",
            "storage_key":     "VARCHAR(500) DEFAULT ''",
            "storage_backend": "VARCHAR(20) DEFAULT ''",
        }
        for col, dtype in clip_storage_additions.items():
            if col not in existing_clips:
                conn.execute(text(f"ALTER TABLE clips ADD COLUMN {col} {dtype}"))

        # ── user_assets: Phase 5 storage columns ────────────────────────
        if inspector.has_table("user_assets"):
            existing_assets = {c["name"] for c in inspector.get_columns("user_assets")}
            asset_storage_additions = {
                "storage_url":     "VARCHAR(500) DEFAULT ''",
                "storage_key":     "VARCHAR(500) DEFAULT ''",
                "storage_backend": "VARCHAR(20) DEFAULT ''",
            }
            for col, dtype in asset_storage_additions.items():
                if col not in existing_assets:
                    conn.execute(text(f"ALTER TABLE user_assets ADD COLUMN {col} {dtype}"))

        # ── upload_jobs table columns (new: publish_kind) ────────────────
        if inspector.has_table("upload_jobs"):
            existing_uploads = {c["name"] for c in inspector.get_columns("upload_jobs")}
            upload_additions = {
                "publish_kind": "VARCHAR(10) DEFAULT 'video'",
            }
            for col, dtype in upload_additions.items():
                if col not in existing_uploads:
                    conn.execute(text(f"ALTER TABLE upload_jobs ADD COLUMN {col} {dtype}"))

        # ── oauth_tokens: cached YouTube-channel metadata ───────────────
        # Populated on OAuth connect + manual refresh.  Eliminates repeated
        # YT Data API calls for display-only fields (thumbnail, subs, etc.)
        if inspector.has_table("oauth_tokens"):
            existing_tokens = {c["name"] for c in inspector.get_columns("oauth_tokens")}
            token_additions = {
                "channel_description":   "TEXT DEFAULT ''",
                "channel_thumbnail_url": "VARCHAR(500) DEFAULT ''",
                "channel_custom_url":    "VARCHAR(120) DEFAULT ''",
                "channel_country":       "VARCHAR(10) DEFAULT ''",
                "subscriber_count":      "INTEGER DEFAULT 0",
                "video_count":           "INTEGER DEFAULT 0",
                "view_count":            "BIGINT DEFAULT 0",
                "metadata_cached_at":    "TIMESTAMP NULL" if engine.dialect.name == "postgresql" else "DATETIME",
                # Per-YT-account logo (lives here, NOT on Channel which is a
                # style template).  NULL = no overlay on rendered videos.
                "logo_asset_id":         "INTEGER",
                # Per-YT-account upload route ("postiz" | "kaizer" | null).
                # NULL = inherit Channel.upload_provider → system default.
                "upload_provider":       "VARCHAR(20)",
            }
            for col, dtype in token_additions.items():
                if col not in existing_tokens:
                    conn.execute(text(f"ALTER TABLE oauth_tokens ADD COLUMN {col} {dtype}"))

        # ── Phase 4 / Wave 3 new tables ─────────────────────────────────────────
        for tbl in ("training_records", "clip_edges", "agency_teams",
                    "agency_members", "agency_audit_log", "regional_api_keys"):
            if not inspector.has_table(tbl):
                Base.metadata.tables[tbl].create(conn)

        # ── Performance phase: full-channel video catalogue ─────────────
        # Caches YouTube Data API responses so the Performance page can
        # show percentiles, top videos, and per-video comparisons
        # without re-hitting the API on every render.
        if not inspector.has_table("channel_videos"):
            Base.metadata.tables["channel_videos"].create(conn)

        # ── Phase 6 — Autonomous Live Director tables ────────────────────────
        for tbl in ("live_events", "live_cameras", "director_log"):
            if not inspector.has_table(tbl):
                Base.metadata.tables[tbl].create(conn)

        # ── Usage telemetry: OpenAI + YouTube API call logs ────────────
        # Drives the admin Usage / AI Costs dashboard.  Mirrors the
        # existing ``gemini_calls`` design so the dashboard can show
        # all three providers side-by-side.
        for tbl in ("openai_calls", "youtube_api_calls"):
            if not inspector.has_table(tbl):
                Base.metadata.tables[tbl].create(conn)

        # ── Capacity planning: persistent system utilisation samples ───
        if not inspector.has_table("system_metrics"):
            Base.metadata.tables["system_metrics"].create(conn)
        else:
            # Additive migration: enrich existing table with the "Kaizer-only"
            # rollup columns. This is the per-process footprint of the Kaizer
            # stack (uvicorn + children + ffmpeg + vite + cloudflared), which
            # is what you actually size the cloud server against — the
            # whole-machine numbers above include Chrome, VS Code, etc.
            existing_sm = {c["name"] for c in inspector.get_columns("system_metrics")}
            sm_additions = {
                "kaizer_cpu_percent":  "FLOAT",
                "kaizer_rss_gb":       "FLOAT",
                "kaizer_proc_count":   "INTEGER",
                "kaizer_ffmpeg_count": "INTEGER",
                "kaizer_gpu_util":     "FLOAT",
            }
            for col, ddl in sm_additions.items():
                if col not in existing_sm:
                    conn.execute(text(f"ALTER TABLE system_metrics ADD COLUMN {col} {ddl}"))

        # ── Password reset tokens (forgot-password flow) ──────────────
        if not inspector.has_table("password_reset_tokens"):
            Base.metadata.tables["password_reset_tokens"].create(conn)

        # ── Phase 14 — V2 Beta launch: job feedback ──────────────────
        # CheckConstraint + UniqueConstraint + Index are baked into the
        # ORM definition; create() emits them with the table.
        if not inspector.has_table("job_feedback"):
            Base.metadata.tables["job_feedback"].create(conn)

        conn.commit()


def _seed_defaults():
    """Populate new DB with default channel profiles + ensure legacy user exists,
    and backfill any pre-auth rows with NULL user_id onto that legacy user.
    Idempotent — safe to run on every startup.
    """
    from sqlalchemy import text as _text
    from auth import ensure_legacy_user

    db = SessionLocal()
    try:
        # Legacy user — owns all pre-existing data from the single-tenant era
        legacy = ensure_legacy_user(db)

        # Backfill — any row with NULL user_id belongs to legacy until a real
        # user claims it.  Cheap UPDATE; noop when there's nothing to backfill.
        for tbl in ("jobs", "channels", "upload_jobs", "campaigns", "competitor_channels"):
            try:
                db.execute(_text(f"UPDATE {tbl} SET user_id = :uid WHERE user_id IS NULL"),
                           {"uid": legacy.id})
            except Exception as e:
                print(f"[startup] backfill {tbl} skipped: {e}")
        db.commit()

        added = seed_channels(db)
        if added:
            # Freshly seeded channels also belong to legacy user
            db.execute(_text("UPDATE channels SET user_id = :uid WHERE user_id IS NULL"),
                       {"uid": legacy.id})
            db.commit()
            print(f"[startup] Seeded {added} default channel(s) for legacy user")
    finally:
        db.close()


# Create tables if they don't exist, then safely add any missing columns
Base.metadata.create_all(bind=engine)
_migrate_schema()
_seed_defaults()

# ── Capacity planning + live log tail ──────────────────────────────────────
# Install the stdout/stderr tee BEFORE the FastAPI banner so the admin Logs
# tab also captures the boot banner. The sampler thread starts here too so
# its first row reflects "right after migrate" rather than "first request".
import system_observer
system_observer.install_log_capture()
system_observer.start_metric_sampler()

BASE_DIR    = Path(__file__).parent
MEDIA_ROOT  = BASE_DIR / "media"
OUTPUT_ROOT = Path(os.getenv("KAIZER_OUTPUT_ROOT", "/tmp/kaizer_output"))
MEDIA_ROOT.mkdir(exist_ok=True)
OUTPUT_ROOT.mkdir(exist_ok=True)

app = FastAPI(title="Kaizer Pipeline API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── OpenTelemetry — opt-in (KAIZER_OTEL_ENABLED=true) ───────────────────────
# Initialise the SDK and attach FastAPI auto-instrumentation BEFORE any
# router gets registered against `app`, so every endpoint is covered.
# When OTEL is disabled, init_tracing returns False and the helpers in
# tracing.span() are no-ops (zero overhead).
try:
    import tracing as _tracing
    _tracing.init_tracing(service_name="kaizer-backend",
                          service_version="2.0.0")
    _tracing.instrument_fastapi(app)
except Exception as _exc:
    print(f"[startup] tracing init failed (non-fatal): {_exc}")

# ── Routers ──────────────────────────────────────────────────────────────────
# Auth (register / login / Google sign-in / me).
app.include_router(auth_router)
# Phases 1–7: channels, SEO, OAuth, uploads.
app.include_router(channels_router)
app.include_router(seo_router)
app.include_router(youtube_oauth_router)
app.include_router(youtube_upload_router)
# Billion-dollar phases A–E: campaigns, performance, translation, trending radar.
app.include_router(campaigns_router)
app.include_router(performance_router)
app.include_router(translation_router)
app.include_router(trending_router)
app.include_router(assets_router)
app.include_router(channel_groups_router)
app.include_router(billing_router)
app.include_router(job_progress_router)   # Phase 2B — job progress endpoint
app.include_router(feedback_router)       # Phase 3.5 — post-publish feedback endpoint
app.include_router(editor_router)         # Wave 2 — editor beta endpoints
app.include_router(live_director_router)  # Phase 6 — Autonomous Live Director
app.include_router(admin_router)           # Phase 12 — admin panel REST surface
app.include_router(work_monitor_router)     # Live work-monitor dashboard (Claude/agents progress)
app.include_router(postiz_router)           # Cross-platform scheduling via Postiz (admin-only)
app.include_router(yt_lookup_router)        # YouTube channel lookup for Style References (auth'd)
app.include_router(bulletin_images_router)  # Per-image bulletin carousel mgmt (list/replace/recompose)
app.include_router(express_mode_router)     # Express Mode — one-click auto-publish (Whisper+Claude+Postiz)
app.include_router(heygen_router)           # HeyGen avatar generation for Trending (replaces Veo 3)
app.include_router(live_studio_router)      # Live Studio — bulk RTMP-live publishing (multi-video × multi-channel)

# ── Static files: /media → BASE_DIR/output  ──────────────────────────────────
# Serves beta-rendered MP4s (and any other output files) to the frontend
# <video> player via the /media/<relative-path> URL scheme.
# Added for Wave 2 editor beta; safe to have even when the output dir is empty.
from fastapi.staticfiles import StaticFiles as _StaticFiles

_output_dir = str(BASE_DIR / "output")
os.makedirs(_output_dir, exist_ok=True)
app.mount("/media", _StaticFiles(directory=_output_dir), name="media")


# ── Upload worker lifecycle ──────────────────────────────────────────────────
@app.on_event("startup")
async def _start_upload_worker():
    try:
        await upload_worker.start()
        print("[startup] upload worker running")
    except Exception as e:
        print(f"[startup] WARN: upload worker failed to start: {e}")


@app.on_event("shutdown")
async def _stop_upload_worker():
    await upload_worker.stop()


# ── Channel learning cron (Phase 7) ──────────────────────────────────────────
@app.on_event("startup")
async def _start_corpus_scheduler():
    try:
        corpus_scheduler.start()
    except Exception as e:
        print(f"[startup] WARN: corpus scheduler failed to start: {e}")


# ── RTMP live agent — orphan broadcast reconciler ────────────────────────────
# If the previous process died mid-stream, some UploadJob rows are stuck
# in status='uploading' with a video_id (broadcast id) already set on
# YouTube's side. This sweep finds them, checks the actual broadcast
# state on YT, and either marks them 'done' (auto-stop fired naturally)
# or 'failed' (broadcast unrecoverable). Cheap (~1 quota unit per orphan,
# free for healthy startups with no orphans).
@app.on_event("startup")
async def _reconcile_rtmp_orphans():
    try:
        from youtube.rtmp_agent import reconcile_orphan_broadcasts
        # Run in a thread so we never block FastAPI startup if the API call hangs.
        import threading
        threading.Thread(
            target=reconcile_orphan_broadcasts,
            name="kaizer-rtmp-reconciler",
            daemon=True,
        ).start()
    except Exception as e:
        print(f"[startup] WARN: rtmp reconciler failed to start: {e}")


@app.on_event("startup")
async def _live_studio_recovery():
    """After a backend crash mid-broadcast, scan for stuck streams + try
    to resume from their R2 preview backup. Daemon thread so a slow R2
    download doesn't block startup."""
    try:
        from live_studio import r2_backup
        import threading
        threading.Thread(
            target=r2_backup.recover_pending_streams,
            name="kaizer-live-studio-recovery",
            daemon=True,
        ).start()
    except Exception as e:
        print(f"[startup] WARN: live studio recovery failed to start: {e}")


@app.on_event("startup")
async def _live_studio_expiry_sweeper():
    """Daily-ish sweep that deletes R2 preview backups whose 48 h
    window has elapsed. Runs once on boot + every 6 h thereafter."""
    try:
        from live_studio import r2_backup
        import threading, time as _t

        def _loop():
            while True:
                try:
                    r2_backup.run_expiry_sweep()
                except Exception as exc:
                    print(f"[live-studio] expiry sweep error: {exc}")
                _t.sleep(6 * 3600)

        threading.Thread(target=_loop, name="kaizer-live-studio-expiry",
                         daemon=True).start()
    except Exception as e:
        print(f"[startup] WARN: live studio expiry sweeper failed to start: {e}")


@app.on_event("shutdown")
async def _stop_corpus_scheduler():
    corpus_scheduler.stop()

# ── Static config ────────────────────────────────────────────────────────────

PLATFORMS = {
    "instagram_reel":          {"label": "Instagram Reel", "width": 1080, "height": 1920},
    "youtube_short":           {"label": "YouTube Short",  "width": 1080, "height": 1920},
    "youtube_full":            {"label": "YouTube Full",   "width": 1920, "height": 1080},
    # ── Compound platform ─────────────────────────────────────────
    # Produces BOTH a long-form bulletin (1920x1080, fixed TV9 layout)
    # AND a set of vertical Shorts (1080x1920, user-selected frame
    # layout) from one upload. ``create_job`` detects this key and
    # fans out into TWO sibling Job rows so the pipeline code itself
    # stays single-platform — no special-case branches in pipeline.py.
    # The ``compound`` + ``expands_to`` fields are what create_job
    # reads; the frontend uses ``label`` + ``width``/``height`` to
    # render the picker (we show the Shorts dimensions because they
    # drive the frame-layout step the user sees next).
    "youtube_full_plus_shorts": {
        "label":      "Full Video + Shorts",
        "width":      1080,
        "height":     1920,
        "compound":   True,
        "expands_to": ["youtube_full", "youtube_short"],
    },
    # ── V2 platform (Step 11 — Inngest-orchestrated pipeline v2) ──
    # Produces both bulletin + shorts via the V2 multi-stage pipeline
    # (Deepgram STT -> Gemini Pro continuity -> Gemini Flash fan-out
    # -> render). Unlike youtube_full_plus_shorts this produces ONE
    # Job row -- the V2 Inngest function generates both output sets
    # internally and runner._import_clips reads the editor_meta.json
    # the V2 adapter writes.
    #
    # CRITICAL: do NOT add ``compound`` or ``expands_to`` markers here.
    # create_job uses those to fan out into sibling Job rows; for V2
    # that fan-out would create two pending jobs (one would never be
    # picked up by the Inngest worker -> permanent stuck "pending").
    # Step 11.1 has a regression test that pins this absence.
    "full_video_shorts_v2": {
        "label":  "Full Video + Shorts (V2 Beta)",
        "width":  1080,
        "height": 1920,
    },
}


# Feature-flag gate (Step 11 D-11.12). When KAIZER_V2_ENABLED is "0"
# / "false" / "no" the V2 platform is filtered out of the picker so
# the 4 existing platforms ship unaffected. Default ON for Beta.
def _v2_enabled() -> bool:
    raw = os.environ.get("KAIZER_V2_ENABLED", "1").strip().lower()
    return raw not in ("0", "false", "no", "off", "")


def resolve_job_name(name_input: Optional[str], video_filename: Optional[str]) -> str:
    """Phase 14 / V2 Beta (D-13.11) name resolution rule.

    User-supplied value is capped at 120 chars (DB column width). Blank
    or whitespace-only falls back to the first 80 chars of the upload
    filename so the jobs list always shows something readable.
    """
    cleaned = (name_input or "").strip()[:120]
    if cleaned:
        return cleaned
    return (video_filename or "")[:80]

# Frame layouts — single source of truth lives in pipeline_core.pipeline
# (the CLI's --frame argparse list is built from it). We keep human-friendly
# labels here so the React UI doesn't have to parse the CLI-style ones, but
# the KEYS are pinned to pipeline.FRAME_LAYOUTS so a layout that doesn't
# exist in the renderer can never be offered to the frontend.
from pipeline_core.pipeline import FRAME_LAYOUTS as _PIPELINE_FRAME_LAYOUTS

_FRAME_LABELS = {
    "torn_card":   "Torn Card — Classic torn-edge news card layout",
    "clean_card":  "Clean Card — Straight-edge layout with framed bottom image",
    "split_frame": "Split Frame — Thumbnail on top + Video on colored background",
    "follow_bar":  "Follow Bar — News card with follow bar at bottom",
}
FRAME_LAYOUTS = {
    k: _FRAME_LABELS.get(k, _PIPELINE_FRAME_LAYOUTS[k])
    for k in _PIPELINE_FRAME_LAYOUTS.keys()
}


from asset_resolver import materialize_asset_locally as _materialize_asset_locally


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"app": "Kaizer Pipeline API", "docs": "/docs", "health": "/api/health/"}

@app.get("/api/health/")
def health():
    return {"status": "ok"}

# ── Config ───────────────────────────────────────────────────────────────────

@app.get("/api/platforms/")
def get_platforms():
    # D-11.12: gate the V2 entry behind KAIZER_V2_ENABLED so the
    # rest of the picker is unaffected if Beta surfaces issues.
    if _v2_enabled():
        return PLATFORMS
    return {
        k: v for k, v in PLATFORMS.items()
        if k != "full_video_shorts_v2"
    }


# Static catalog for the V2 STT picker (Step 11.2). The ``configured``
# field is derived at request time from the corresponding API-key env
# var; everything else is static metadata the UI shows in tooltips +
# tier badges.
_V2_STT_PROVIDER_CATALOG = [
    {
        "id":              "whisper-groq",
        "display_name":    "Whisper (Groq)",
        "tier":            "free",
        "cost_per_min_usd": 0.0,
        "_api_key_env":    "GROQ_API_KEY",
        "description":     (
            "Free tier (rate-limited). Good multilingual accuracy. "
            "100 MB file cap on dev tier."
        ),
        # Step 12.5 / backlog item 59: surfaces in the wizard so a
        # user picking Whisper-Groq for Telugu/Hindi sees the
        # known-issue warning before submitting (the empirical
        # finding from Step 12.2a Path 2 investigation, also
        # backlog item 57).
        "warnings": [
            "Known timestamp issues with Telugu, Hindi, and other "
            "Indian-language audio. Use Deepgram for Indian-language "
            "content."
        ],
    },
    {
        "id":              "deepgram",
        "display_name":    "Deepgram Nova-3",
        "tier":            "premium",
        "cost_per_min_usd": 0.0097,
        "_api_key_env":    "DEEPGRAM_API_KEY",
        "description":     (
            "Premium tier. Per-word confidence + diarization. "
            "Telugu single-language mode for V2 Telugu workloads."
        ),
        "warnings": [],
    },
    {
        "id":              "assemblyai",
        "display_name":    "AssemblyAI Universal-2",
        "tier":            "mid",
        "cost_per_min_usd": 0.0070,
        "_api_key_env":    "ASSEMBLYAI_API_KEY",
        "description":     (
            "Mid-tier. Strong English accuracy; weaker on Indian "
            "languages. Includes word-level timestamps."
        ),
        "warnings": [],
    },
]


@app.get("/api/v2/stt/providers/")
def get_v2_stt_providers():
    """V2 STT provider catalog (Step 11.2).

    Returns the full 3-provider list regardless of which API keys are
    set, so the UI can show "Not configured" tooltips against the
    disabled options instead of mysteriously hiding them. The
    ``configured`` field is computed at request time so an operator
    can rotate keys without restarting the API.

    The internal ``_api_key_env`` field is stripped before returning.
    """
    out: list[dict] = []
    for p in _V2_STT_PROVIDER_CATALOG:
        api_key = os.environ.get(p["_api_key_env"], "").strip()
        out.append({
            "id":               p["id"],
            "display_name":     p["display_name"],
            "tier":             p["tier"],
            "cost_per_min_usd": p["cost_per_min_usd"],
            "configured":       bool(api_key),
            "description":      p["description"],
            # Step 12.5 / backlog 59: per-provider warning strings
            # (empty list = no warnings). Frontend renders these
            # as a tooltip / inline hint when surfacing the option.
            "warnings":         list(p.get("warnings") or []),
        })
    return out

@app.get("/api/frame-layouts/")
def get_frame_layouts():
    return FRAME_LAYOUTS

@app.get("/api/fonts/{filename}")
def serve_font(filename: str):
    from fastapi.responses import FileResponse
    font_path = BASE_DIR / "resources" / "fonts" / filename
    if not font_path.exists():
        raise HTTPException(status_code=404, detail="Font not found")
    return FileResponse(font_path, media_type="font/ttf", headers={
        "Cache-Control": "public, max-age=86400",
        "Access-Control-Allow-Origin": "*",
    })

@app.get("/api/frames/")          # legacy alias
def get_frames():
    return FRAME_LAYOUTS

# ── Jobs ─────────────────────────────────────────────────────────────────────

@app.get("/api/jobs/")
def list_jobs(db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    jobs = (
        db.query(models.Job)
          .filter(models.Job.user_id == user.id)
          .order_by(models.Job.created_at.desc())
          .all()
    )
    return [
        {
            "id": j.id,
            "status": j.status,
            "platform": j.platform,
            "frame_layout": j.frame_layout,
            "video_name": j.video_name,
            "name": j.name,
            "language": j.language or "te",
            "created_at": j.created_at,
            "clip_count": len(j.clips),
            # Item 104: surface the transition choice so the UI can
            # show a chip on the job card.
            "transition_style": (j.transition_style or "smart_cut"),
        }
        for j in jobs
    ]


@app.post("/api/jobs/create/")
async def create_job(
    video: UploadFile = File(...),
    platform: str = Form(...),
    frame_layout: str = Form(...),
    language: str = Form("te"),
    use_default_image: bool = Form(False),
    # Optional: pick which style-profile's logo to overlay.  Resolved to the
    # UserAsset.file_path at render time.  Absent / invalid = no logo.
    logo_channel_id: Optional[int] = Form(None),
    # V2 only (Step 11.3): STT provider key from the wizard's
    # "Choose STT" step. Ignored by V1 platforms; passed through to
    # runner.run_pipeline which threads it into the Inngest event.
    stt_provider: str = Form(""),
    # Bulletin pre-selected images. Comma-separated UserAsset IDs.
    # When present, the bulletin pass cycles through these instead of
    # calling OpenAI gpt-image-1. Only meaningful for platforms that
    # render a bulletin (youtube_full / youtube_full_plus_shorts).
    bulletin_image_ids: str = Form(""),
    # Phase 14 / V2 Beta (D-13.11): optional human-readable name. Caps at
    # 120 chars (DB column width); blank/missing defaults to first 80
    # chars of the source filename so the list never shows "(unnamed)".
    name: str = Form(""),
    # Item 104 (Transition library): operator's chosen inter-clip
    # transition for the V2 bulletin pass. One of the catalog names
    # in pipeline_v2.transitions (smart_cut / crossfade / fade_to_black
    # / dip_to_white / slide_left / wipe_right / dissolve). Blank or
    # unknown -> "smart_cut" (the default + only one implemented at
    # ship time). Ignored by V1 platforms.
    transition_style: str = Form("smart_cut"),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    import languages as _langs
    lang_cfg = _langs.get(language)  # falls back to default if invalid

    upload_dir = MEDIA_ROOT / "uploads"
    upload_dir.mkdir(exist_ok=True)
    video_path = upload_dir / video.filename

    with open(video_path, "wb") as f:
        f.write(await video.read())

    # Mirror the source video to R2 right after the local write. The
    # pipeline subprocess still reads from the local copy (faster than
    # a download round-trip), but R2 is the source of truth — if the
    # container restarts mid-render, retry can pull the source back.
    # Predictable key so retry/recovery doesn't need a DB lookup.
    _src_timestamp = time.strftime("%Y%m%d_%H%M%S")
    try:
        from pipeline_core.storage import get_storage_provider
        # Honours STORAGE_BACKEND — local mode mirrors to ``output/`` for
        # parity with prod's R2 mirror. Failure is non-fatal: the user's
        # uploaded file is still on disk; this branch only guards
        # post-restart recovery on a horizontal-scale prod deploy.
        get_storage_provider().upload(
            str(video_path),
            f"sources/{user.id}/{_src_timestamp}_{video.filename}",
            content_type=(video.content_type or "video/mp4"),
        )
    except Exception as _src_exc:
        print(f"[create_job] source video storage mirror failed (non-fatal): {_src_exc}")

    # ── Input validation gate (Phase 1) ──────────────────────────────────────
    # Validate the uploaded file before creating a job row or starting the
    # pipeline.  Hard errors (wrong codec, corrupt file, etc.) return HTTP 400
    # immediately so the user gets actionable feedback without wasting a job
    # slot.  Soft warnings are attached to the job's log field so they surface
    # in the UI after the pipeline completes.
    _validation_warnings: list[str] = []
    try:
        import sys as _sys
        _sys.path.insert(0, str(BASE_DIR / "pipeline_core"))
        from pipeline_core.validator import validate_input as _validate_input  # type: ignore
        _val_result = _validate_input(str(video_path))
        if not _val_result.ok:
            # Clean up uploaded file so it doesn't accumulate on disk
            try:
                video_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise HTTPException(
                status_code=400,
                detail={
                    "errors": _val_result.errors,
                    "warnings": _val_result.warnings,
                },
            )
        _validation_warnings = _val_result.warnings
    except HTTPException:
        raise
    except Exception as _ve:
        # Validator import failure or unexpected error — log and continue so a
        # broken validator never blocks all uploads.  The pipeline itself will
        # fail if the file is truly bad.
        import logging as _logging
        _logging.getLogger("kaizer.pipeline.validator").warning(
            "create_job: validator error (non-fatal): %s", _ve
        )

    _warning_prefix = ""
    if _validation_warnings:
        _warning_prefix = "[input warnings] " + "; ".join(_validation_warnings) + "\n"

    # Phase 14 / V2 Beta (D-13.11): see resolve_job_name() below.
    _name_clean = resolve_job_name(name, video.filename)

    # Single Job row regardless of platform. The compound platform
    # ("youtube_full_plus_shorts") stores its original key here so the
    # runner can detect it and run TWO pipeline passes internally —
    # one bulletin pass + one shorts pass — both importing clips into
    # THIS same job. UI shows it as one job with mixed-aspect clips.
    # Item 104: validate transition_style against the catalog; unknown
    # values get coerced to "smart_cut" rather than rejected so a stale
    # frontend can't 400 the user. The log line surfaces the coercion.
    try:
        from pipeline_v2.transitions import (
            is_valid_transition as _is_valid_transition,
            DEFAULT_TRANSITION_NAME as _DEFAULT_TRANSITION,
        )
        _ts = (transition_style or "").strip()
        if not _ts or not _is_valid_transition(_ts):
            if _ts and _ts != _DEFAULT_TRANSITION:
                _warning_prefix = (_warning_prefix or "") + (
                    f"[transition_style] unknown value {_ts!r} coerced "
                    f"to {_DEFAULT_TRANSITION!r}.\n"
                )
            _ts = _DEFAULT_TRANSITION
    except Exception as _trans_exc:
        # Import failure or unexpected error: fall back to the literal
        # default so create-job never breaks on a missing catalog.
        import logging as _logging
        _logging.getLogger("kaizer.transitions").warning(
            "create_job: transition catalog lookup failed (non-fatal): %s",
            _trans_exc,
        )
        _ts = "smart_cut"

    job = models.Job(
        user_id=user.id,
        platform=platform,
        frame_layout=frame_layout,
        video_name=video.filename,
        name=_name_clean,
        language=lang_cfg.code,
        status="pending",
        log=_warning_prefix,
        output_dir=str(OUTPUT_ROOT),
        transition_style=_ts,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # If the user opted in, look up their default asset and pass the absolute
    # path to the pipeline so every clip uses it instead of stock photos.
    # Surface what happened in job.log so silent fallbacks are visible.
    default_img_path = ""
    if use_default_image:
        asset = (
            db.query(models.UserAsset)
              .filter(
                  models.UserAsset.user_id == user.id,
                  models.UserAsset.is_default_ad == True,  # noqa: E712
              )
              .first()
        )
        if asset is None:
            job.log = (job.log or "") + (
                "[default-image] Use default image was on but no asset is "
                "marked as default — using stock photos.\n"
            )
        else:
            default_img_path = _materialize_asset_locally(asset)
            if default_img_path:
                job.log = (job.log or "") + (
                    f"[default-image] Using {asset.filename!r} (id={asset.id}) "
                    f"for every clip.\n"
                )
            else:
                job.log = (job.log or "") + (
                    f"[default-image] Failed to materialise asset {asset.id} "
                    f"({asset.filename!r}) — neither local file nor R2 copy "
                    f"could be opened. Falling back to stock photos.\n"
                )
        db.commit()

    # Per-destination logos: the pipeline now renders a CLEAN MASTER with no
    # logo overlay.  The upload worker applies each destination's logo at
    # publish time (youtube/logo_overlay.py) so Auto Wala videos get Auto
    # Wala's logo, Cyber Sphere videos get Cyber Sphere's.  Set
    # KAIZER_BAKE_LOGO_AT_RENDER=true in .env to restore the old single-logo
    # behavior (faster renders, but same logo across all destinations).
    _BAKE_AT_RENDER = (os.environ.get("KAIZER_BAKE_LOGO_AT_RENDER", "") or "").lower() == "true"
    default_logo_path = ""

    def _resolve_asset(asset_id):
        if not asset_id:
            return ""
        a = db.query(models.UserAsset).filter(
            models.UserAsset.id == asset_id,
            models.UserAsset.user_id == user.id,
        ).first()
        return _materialize_asset_locally(a) if a else ""

    # Only bake a logo at render time when the explicit legacy flag is on.
    # Default path: render clean master, let upload worker overlay per-
    # destination (handled in youtube/logo_overlay.py + worker.py).
    if _BAKE_AT_RENDER and logo_channel_id:
        ch = (
            db.query(models.Channel)
              .filter(
                  models.Channel.id == logo_channel_id,
                  models.Channel.user_id == user.id,
              )
              .first()
        )
        tok = ch.oauth_token if ch else None
        asset_id = (tok.logo_asset_id if tok and tok.logo_asset_id
                    else (ch.logo_asset_id if ch else None))
        default_logo_path = _resolve_asset(asset_id)

    # ── Pre-selected bulletin images ───────────────────────────────
    # Resolve each ID to an absolute on-disk path. Skip silently if a
    # row is missing or the file can't be materialised (R2 fetch fail).
    # The pipeline gets a single pipe-separated string; that
    # separator was picked because Windows paths contain ':' so a
    # colon-separator would break on the dev box.
    bulletin_image_paths: list[str] = []
    if bulletin_image_ids.strip():
        for _raw_id in bulletin_image_ids.split(","):
            _raw_id = _raw_id.strip()
            if not _raw_id.isdigit():
                continue
            _asset = (db.query(models.UserAsset)
                        .filter(models.UserAsset.id == int(_raw_id),
                                models.UserAsset.user_id == user.id)
                        .first())
            if not _asset:
                job.log = (job.log or "") + (
                    f"[bulletin-images] asset id={_raw_id} not found / not owned by user — skipped\n"
                )
                continue
            _path = _materialize_asset_locally(_asset)
            if _path and os.path.exists(_path):
                bulletin_image_paths.append(_path)
            else:
                job.log = (job.log or "") + (
                    f"[bulletin-images] asset {_asset.id} ({_asset.filename!r}) "
                    f"could not be materialised locally — skipped\n"
                )
        if bulletin_image_paths:
            job.log = (job.log or "") + (
                f"[bulletin-images] Pre-selected {len(bulletin_image_paths)} image(s) "
                f"for bulletin carousel; OpenAI generation will be skipped.\n"
            )
            db.commit()

    # ONE runner call, ONE Job row. When platform is the compound key
    # ("youtube_full_plus_shorts"), runner.run_pipeline detects it and
    # internally spawns two pipeline subprocesses in sequence — one
    # bulletin pass + one shorts pass — both importing clips into THIS
    # job. From the frontend's perspective it's a single job with
    # mixed-aspect-ratio clips.
    runner.run_pipeline(
        job_id=job.id,
        video_path=str(video_path),
        platform=platform,
        frame=frame_layout,
        language=lang_cfg.code,
        default_image=default_img_path,
        default_logo=default_logo_path,
        bulletin_images=bulletin_image_paths,
        # V2 only (Step 11.4): ignored unless platform=full_video_shorts_v2.
        stt_provider=stt_provider,
        # Item 104: V2 only. V1 paths ignore.
        transition_style=_ts,
        db_session_factory=SessionLocal,
    )

    return {"id": job.id, "status": job.status, "language": lang_cfg.code}


@app.get("/api/languages/")
def list_languages():
    """Language picker payload for the frontend New Job form."""
    import languages as _langs
    return _langs.list_options()


@app.post("/api/clips/raw-upload/")
async def raw_upload(
    video: UploadFile = File(...),
    title:    str = Form(""),
    language: str = Form("te"),
    platform: str = Form("youtube_full"),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Upload an already-edited video directly as a publishable Clip.

    Skips the whole Gemini-cut / compose pipeline. Creates a tiny Job with
    status='done' so all existing Editor / SEO / Publish / Uploads flows
    work against the resulting Clip unchanged.
    """
    import subprocess, sys
    import languages as _langs
    lang_cfg = _langs.get(language)

    # Save the file under a predictable path so /api/file/ allowlist accepts it
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    upload_dir = BASE_DIR / "output" / "raw_uploads" / timestamp
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(video.filename or "upload.mp4").name
    video_path = upload_dir / safe_name
    with open(video_path, "wb") as f:
        f.write(await video.read())

    # Mirror the source video to R2. We need the local file briefly for
    # the ffprobe-duration + ffmpeg-thumbnail steps below, so we keep
    # video_path on disk through THIS request — but we delete it at the
    # end before returning. R2 is then the source of truth.
    video_storage_url = ""
    video_storage_key = ""
    video_storage_backend = ""
    try:
        from pipeline_core.storage import get_storage_provider
        # Honours STORAGE_BACKEND — local writes to ``output/raw_uploads/``,
        # prod ships to R2. Either way the backend name is captured on
        # the Clip row so the worker's _ensure_local_clip can resolve
        # the file later.
        _stor = get_storage_provider()
        video_storage_key = f"raw_uploads/{user.id}/{timestamp}/{safe_name}"
        _obj = _stor.upload(
            str(video_path),
            video_storage_key,
            content_type=(video.content_type or "video/mp4"),
        )
        video_storage_url = _obj.url
        video_storage_backend = _stor.name
    except Exception as _src_exc:
        print(f"[raw-upload] storage mirror failed (non-fatal): {_src_exc}")

    # ffprobe for duration — silently defaults to 0 if the binary is missing
    duration = 0.0
    try:
        sys.path.insert(0, str(BASE_DIR / "pipeline_core"))
        from pipeline import FFMPEG_BIN, get_video_info  # type: ignore
        info = get_video_info(str(video_path))
        if info:
            duration = float(info.get("duration") or 0)
    except Exception:
        FFMPEG_BIN = "ffmpeg"  # Fallback — hope it's on PATH

    # Auto-generate a thumbnail from the first frame
    thumb_path = upload_dir / f"thumb_{safe_name}.jpg"
    try:
        subprocess.run(
            [FFMPEG_BIN, "-y", "-i", str(video_path),
             "-vframes", "1", "-q:v", "2", str(thumb_path)],
            capture_output=True, check=True, timeout=30,
        )
    except Exception as e:
        print(f"[raw-upload] thumb gen failed: {e}")
        thumb_path = None  # type: ignore

    # Mirror the freshly-generated thumbnail to storage. Honours
    # STORAGE_BACKEND — failures don't block; _clip_dict falls back
    # to /api/file/.
    thumb_storage_url = ""
    if thumb_path and thumb_path.exists():
        try:
            from pipeline_core.storage import get_storage_provider
            storage = get_storage_provider()
            obj = storage.upload(
                str(thumb_path),
                f"raw_uploads/{user.id}/{thumb_path.name}",
                content_type="image/jpeg",
            )
            thumb_storage_url = obj.url
        except Exception as exc:
            print(f"[raw-upload] thumb storage upload failed: {exc}")

    # Minimal Job acting as a container for the standalone Clip
    job = models.Job(
        user_id=user.id,
        platform=platform,
        frame_layout="raw_upload",
        video_name=safe_name,
        language=lang_cfg.code,
        status="done",
        log="[raw-upload] no pipeline run — user-edited video",
        output_dir=str(upload_dir),
    )
    db.add(job); db.commit(); db.refresh(job)

    display_title = (title or "").strip() or Path(safe_name).stem

    clip = models.Clip(
        job_id=job.id,
        clip_index=0,
        filename=safe_name,
        file_path=str(video_path),
        thumb_path=str(thumb_path) if thumb_path and thumb_path.exists() else "",
        image_path="",
        thumb_storage_url=thumb_storage_url,
        # Phase 5 storage fields — populated by the R2 mirror above so
        # _clip_dict.video_url returns the R2 URL on production.
        storage_url=video_storage_url,
        storage_key=video_storage_key,
        storage_backend=video_storage_backend,
        duration=duration,
        frame_type="raw_upload",
        text=display_title,
        sentiment="",
        entities=json.dumps([]),
        card_params=json.dumps({}),
        section_pct=json.dumps({}),
        follow_params=json.dumps({}),
        meta=json.dumps({
            "raw_upload": True,
            "platform": platform,
            "language": lang_cfg.code,
            "original_filename": safe_name,
        }),
    )
    db.add(clip); db.commit(); db.refresh(clip)

    # Both video + thumb are now in R2. Wipe the local copies — Railway's
    # ephemeral disk fills up otherwise (we hit "Container exceeding
    # maximum ephemeral storage"). asset_resolver.materialize_asset_locally
    # pulls bytes from R2 on demand for downstream operations that need
    # a real path (publish-time logo overlay, etc.).
    if video_storage_url:
        try:
            video_path.unlink(missing_ok=True)
            if thumb_path and Path(thumb_path).exists():
                Path(thumb_path).unlink(missing_ok=True)
            # Drop the now-empty timestamped directory
            try:
                upload_dir.rmdir()
            except OSError:
                pass
        except Exception as cleanup_exc:
            print(f"[raw-upload] cleanup warning: {cleanup_exc}")

    return {
        "job_id":   job.id,
        "clip_id":  clip.id,
        "duration": duration,
        "language": lang_cfg.code,
    }


@app.get("/api/jobs/{job_id}/")
def get_job(job_id: int, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    job = db.query(models.Job).filter(
        models.Job.id == job_id, models.Job.user_id == user.id,
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job.id,
        "status": job.status,
        "platform": job.platform,
        "frame_layout": job.frame_layout,
        "video_name": job.video_name,
        "name": job.name,
        "language": job.language or "te",
        "log": job.log,
        "created_at":  job.created_at,
        "started_at":  job.started_at.isoformat()  if job.started_at  else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "elapsed_seconds": _elapsed_seconds(job),
        # Item 104: surface the operator's transition selection.
        "transition_style": (job.transition_style or "smart_cut"),
        # Bulletin clips render first (16:9 long-form takes the lead
        # tile in the JobDetail grid), shorts follow in DB-insert
        # order. Backlog item 91 follow-up.
        "clips": [
            _clip_dict(c) for c in sorted(
                job.clips,
                key=lambda c: (0 if c.frame_type == "bulletin" else 1, c.id),
            )
        ],
    }


@app.get("/api/jobs/{job_id}/status/")
def get_job_status(job_id: int, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    job = db.query(models.Job).filter(
        models.Job.id == job_id, models.Job.user_id == user.id,
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    log_lines = (job.log or "").split("\n") if job.log else []
    progress_pct = _estimate_progress(log_lines, job.status)
    return {
        "status": job.status,
        "progress_pct": progress_pct,
        "log_lines": log_lines,
        "error": job.error or "",
        "started_at":  job.started_at.isoformat()  if job.started_at  else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "elapsed_seconds": _elapsed_seconds(job),
        # V2 per-step progress (Step 10.7 / Step 11.5). NULL for V1
        # jobs + V2 jobs at start/end. UI shows
        # "Stage X of 7: <human label>" only when this is non-null.
        "current_stage": job.current_stage,
        "platform": job.platform,
    }


# Phase 14 / V2 Beta (D-13.14): rename a job mid-flight.
@app.patch("/api/jobs/{job_id}/rename/")
def rename_job(
    job_id: int,
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    new_name = (payload.get("name") or "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="name must be non-empty")
    if len(new_name) > 120:
        raise HTTPException(status_code=400, detail="name must be <= 120 chars")

    job = db.query(models.Job).filter(
        models.Job.id == job_id, models.Job.user_id == user.id,
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job.name = new_name
    db.commit()
    db.refresh(job)
    return {"id": job.id, "name": job.name}


# Phase 14 / V2 Beta (D-13.13): submit 0-100 rating + optional comment.
@app.post("/api/jobs/{job_id}/feedback/")
def submit_job_feedback(
    job_id: int,
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    # Validate rating range up-front so we never INSERT a row that the
    # DB CHECK constraint would reject (CHECK rejection wraps in
    # IntegrityError which is harder to translate to a clear 400).
    raw_rating = payload.get("rating")
    if not isinstance(raw_rating, int) or isinstance(raw_rating, bool):
        raise HTTPException(status_code=400, detail="rating must be an integer")
    if raw_rating < 0 or raw_rating > 100:
        raise HTTPException(status_code=400, detail="rating must be in [0, 100]")
    comment = (payload.get("comment") or "").strip()

    job = db.query(models.Job).filter(
        models.Job.id == job_id, models.Job.user_id == user.id,
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done":
        raise HTTPException(
            status_code=400,
            detail="feedback can only be submitted on completed jobs",
        )

    # Dup check — one feedback per (job, user). Returns 409 on second
    # submission. UI may switch to an "Update" affordance later; for
    # now the first vote is locked in.
    existing = db.query(models.JobFeedback).filter(
        models.JobFeedback.job_id == job_id,
        models.JobFeedback.user_id == user.id,
    ).first()
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail="feedback already submitted for this job",
        )

    fb = models.JobFeedback(
        job_id=job_id,
        user_id=user.id,
        rating=raw_rating,
        comment=comment,
    )
    db.add(fb)
    db.commit()
    db.refresh(fb)
    return {
        "id": fb.id,
        "job_id": fb.job_id,
        "rating": fb.rating,
        "comment": fb.comment,
        "submitted_at": fb.submitted_at.isoformat() if fb.submitted_at else None,
    }


# Phase 14 / V2 Beta (D-13.12, user-facing): aggregate stats for the
# calling user's V2 jobs only. Cheap query — drives the JobsStats page
# header and the optional dashboard card.
@app.get("/api/v2/stats/")
def v2_user_stats(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    from sqlalchemy import func as _f
    V2 = "full_video_shorts_v2"

    rows = (
        db.query(models.Job.status, _f.count(models.Job.id))
          .filter(models.Job.user_id == user.id, models.Job.platform == V2)
          .group_by(models.Job.status)
          .all()
    )
    status_counts = {s: int(c) for s, c in rows}
    total       = sum(status_counts.values())
    completed   = status_counts.get("done", 0)
    failed      = status_counts.get("failed", 0)
    cancelled   = status_counts.get("cancelled", 0)
    success_rate = round((completed / total * 100), 1) if total else 0.0

    fb_row = (
        db.query(_f.avg(models.JobFeedback.rating), _f.count(models.JobFeedback.id))
          .join(models.Job, models.JobFeedback.job_id == models.Job.id)
          .filter(models.JobFeedback.user_id == user.id, models.Job.platform == V2)
          .one()
    )
    avg_rating   = round(float(fb_row[0]), 1) if fb_row[0] is not None else None
    rating_count = int(fb_row[1] or 0)

    return {
        "total_v2_jobs":   total,
        "completed_count": completed,
        "failed_count":    failed,
        "cancelled_count": cancelled,
        "success_rate_pct": success_rate,
        "average_rating":  avg_rating,
        "rating_count":    rating_count,
    }


def _elapsed_seconds(job: "models.Job") -> int | None:
    """Wall-clock runtime of the pipeline.  Live-counting while running."""
    start = job.started_at
    if not start:
        return None
    from datetime import datetime, timezone
    # SQLite strips tzinfo — treat stored naive datetimes as UTC.
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    end = job.finished_at
    if end is not None and end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    if end is None:
        end = datetime.now(timezone.utc)
    try:
        return max(0, int((end - start).total_seconds()))
    except Exception:
        return None


def _meta_path_from_log(log: str) -> Path | None:
    """Mine the stored job log for the exact editor_meta.json path.

    Handles three cases, in priority order:
      1. `[kaizer:meta] <abs_path>`   — emitted by pipeline.py post-write
      2. `Output: <dir>`              — legacy breadcrumb before the marker existed
    """
    if not log:
        return None
    import re
    # Priority 1: explicit marker
    for m in re.finditer(r"^\s*\[kaizer:meta\]\s+(.+?)\s*$", log, re.MULTILINE):
        p = Path(m.group(1).strip())
        if p.exists():
            return p
    # Priority 2: "Output: <dir>" printed by run_pipeline banner
    for m in re.finditer(r"^\s*Output:\s+(.+?)\s*$", log, re.MULTILINE):
        candidate = Path(m.group(1).strip()) / "editor_meta.json"
        if not candidate.is_absolute():
            candidate = (BASE_DIR / candidate).resolve()
        if candidate.exists():
            return candidate
    return None


@app.post("/api/jobs/{job_id}/reimport/")
def reimport_clips(job_id: int, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    """Re-run clip import for a job whose pipeline already finished.

    Used when `_import_clips` failed silently (e.g. the earlier cp1252 bug) or
    when a fresh editor_meta.json has been written to disk and the DB is stale.
    Clears any existing clip rows first so the import is idempotent.
    """
    job = db.query(models.Job).filter(
        models.Job.id == job_id, models.Job.user_id == user.id,
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Drop any stale Clip rows for this job so we don't accumulate duplicates
    db.query(models.Clip).filter(models.Clip.job_id == job_id).delete()
    db.commit()

    # Prefer the exact meta path mined from the job's log output — avoids
    # picking up an unrelated editor_meta.json from an older run via rglob.
    meta_override = _meta_path_from_log(job.log or "")

    try:
        runner._import_clips(job, db, meta_override=meta_override)
    except Exception as e:
        job.status = "failed"
        job.error = f"Reimport failed: {e}"
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))

    db.refresh(job)
    if not job.clips:
        job.status = "failed"
        job.error = "Reimport found 0 clips on disk."
        db.commit()
        raise HTTPException(
            status_code=422,
            detail="No clips found on disk for this job. Check the output directory.",
        )

    job.status = "done"
    job.error = ""
    db.commit()
    return {"imported": len(job.clips), "output_dir": job.output_dir}


@app.post("/api/jobs/{job_id}/export/")
def export_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    export_dir = BASE_DIR / "output" / "exports" / str(job_id)
    export_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for clip in job.clips:
        # Prefer local file (fast); fall back to R2 download for clips
        # whose local copy was wiped by a Railway redeploy.
        src_path = ""
        if clip.file_path and Path(clip.file_path).exists():
            src_path = clip.file_path
        elif clip.storage_key and clip.storage_backend:
            try:
                import tempfile as _tf
                from pipeline_core.storage import get_storage_provider
                provider = get_storage_provider(clip.storage_backend)
                tmp_dir = _tf.mkdtemp(prefix="kaizer_export_")
                src_path = str(Path(tmp_dir) / (Path(clip.storage_key).name or f"clip_{clip.id}.mp4"))
                provider.download(clip.storage_key, src_path)
            except Exception as _exc:
                print(f"[export] R2 fetch failed for clip {clip.id}: {_exc}")
                src_path = ""

        if src_path and Path(src_path).exists():
            dest = export_dir / (clip.filename or Path(src_path).name)
            shutil.copy2(src_path, dest)
            count += 1

    return {"count": count, "export_dir": str(export_dir)}


@app.delete("/api/jobs/{job_id}/delete/")
def delete_job(job_id: int, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    job = db.query(models.Job).filter(
        models.Job.id == job_id, models.Job.user_id == user.id,
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    db.delete(job)
    db.commit()
    return {"deleted": job_id}


@app.post("/api/jobs/{job_id}/cancel/")
def cancel_job_endpoint(
    job_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Stop a running (or queued) job.

    Behaviour:
      - 404 if the job doesn't exist or belongs to a different user.
      - No-op if the job is already in a terminal state (done / failed /
        cancelled) — returns the current status so the UI can refresh.
      - Otherwise: sets ``cancel_requested=True``, asks the runner to
        tree-kill the live subprocess (if any), and stamps the job as
        ``cancelled`` with ``finished_at=now``. The runner's exit
        handler will see ``cancel_requested`` and avoid re-marking the
        job as failed.
    """
    job = db.query(models.Job).filter(
        models.Job.id == job_id, models.Job.user_id == user.id,
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status in ("done", "failed", "cancelled"):
        return {
            "job_id":         job_id,
            "status":         job.status,
            "already_final":  True,
            "killed_pids":    [],
        }

    # Set the flag FIRST so the runner's exit handler can read it.
    job.cancel_requested = True
    db.commit()

    # Walk the subprocess tree + SIGKILL ffmpeg children. Returns even
    # when no live process is tracked (queued jobs, or a job whose
    # runner thread already exited).
    try:
        import runner as _runner
        kill_result = _runner.cancel_job(job_id)
    except Exception as exc:
        kill_result = {"job_id": job_id, "found_running": False,
                       "killed_pids": [], "error": str(exc)}

    # Stamp final state. If the runner thread is still running it will
    # see cancel_requested and skip the failed-status branch; this
    # final write is idempotent either way because both code paths
    # land on status="cancelled".
    job.status = "cancelled"
    if job.error and not job.error.startswith("Cancelled"):
        job.error = "Cancelled by user. " + (job.error[:500] if job.error else "")
    else:
        job.error = "Cancelled by user."
    from datetime import datetime as _dt, timezone as _tz
    job.finished_at = _dt.now(_tz.utc)
    db.commit()

    return {
        "job_id":         job_id,
        "status":         "cancelled",
        "already_final":  False,
        "found_running":  kill_result.get("found_running", False),
        "killed_pids":    kill_result.get("killed_pids", []),
    }


@app.get("/api/jobs/{job_id}/log/")
def stream_log(job_id: int):
    """SSE: stream log until job completes."""
    def event_stream():
        last_len = 0
        for _ in range(600):
            db = SessionLocal()
            job = db.query(models.Job).filter(models.Job.id == job_id).first()
            db.close()
            if not job:
                break
            log = job.log or ""
            if len(log) > last_len:
                yield f"data: {json.dumps({'log': log[last_len:], 'status': job.status})}\n\n"
                last_len = len(log)
            if job.status in ("done", "failed"):
                yield f"data: {json.dumps({'log': '', 'status': job.status, 'done': True})}\n\n"
                break
            time.sleep(1)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ── Clips ────────────────────────────────────────────────────────────────────

@app.get("/api/clips/{clip_id}/")
def get_clip(clip_id: int, db: Session = Depends(get_db)):
    clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    return _clip_dict(clip)


@app.post("/api/clips/{clip_id}/rerender/")
async def rerender_clip(clip_id: int, request: Request, db: Session = Depends(get_db)):
    clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    edits = await request.json()

    # Persist new params to DB
    if "text" in edits:
        clip.text = edits["text"]
    if "frame_type" in edits:
        clip.frame_type = edits["frame_type"]
    if "section_pct" in edits:
        clip.section_pct = json.dumps(edits["section_pct"])
    if "follow_params" in edits:
        clip.follow_params = json.dumps(edits["follow_params"])

    card_params = json.loads(clip.card_params or "{}")
    for key in ("font_size", "text_color", "font_file", "card_style"):
        if key in edits:
            card_params[key] = edits[key]
    clip.card_params = json.dumps(card_params)

    db.commit()

    # Actually re-compose the clip using pipeline functions
    meta = json.loads(clip.meta or "{}")
    raw_path = meta.get("raw_path", "")
    if not raw_path:
        raise HTTPException(status_code=422, detail="No source video path in clip metadata — cannot rerender")
    if not Path(raw_path).exists():
        raise HTTPException(status_code=410, detail="Source video has expired (server was redeployed) — please re-run the pipeline job to regenerate clips")

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _recompose_clip, clip, meta, edits, db)
    except Exception as e:
        import traceback
        print(f"[rerender] compose error:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Rerender failed: {e}")

    return _clip_dict(clip)


def _recompose_clip(clip, meta, edits, db):
    """Re-compose a clip using the pipeline's compose functions with updated params.

    Fast-path: we reuse ``meta.raw_path`` — the already-cut source clip
    that the original pipeline produced. That means no scene-detection,
    no re-encode of the source, no AI calls. The compose functions only
    redraw the overlay (text, card, follow-bar, image), which is typically
    a few seconds with NVENC vs ~minutes for a full re-pipeline.
    """
    import subprocess, sys, time

    t0 = time.time()
    raw_path = meta.get("raw_path", "")
    out_path = clip.file_path
    preset = meta.get("preset", {"width": 1080, "height": 1920})
    frame_type = clip.frame_type or meta.get("frame_type", "follow_bar")
    title_text = clip.text or meta.get("text", "KAIZER NEWS")
    image_path = clip.image_path or meta.get("image_path", "")

    card_params = json.loads(clip.card_params or "{}")
    follow_params = json.loads(clip.follow_params or "{}")
    section_pct = json.loads(clip.section_pct or "{}")

    print(f"[rerender] clip={clip.id} fast-path: reusing cached cut at {raw_path} "
          f"(frame_type={frame_type})")
    changed_keys = sorted(edits.keys()) if isinstance(edits, dict) else []
    if changed_keys:
        print(f"[rerender] clip={clip.id} changes: {changed_keys}")

    # Import pipeline compose functions
    sys.path.insert(0, str(BASE_DIR / "pipeline_core"))
    from pipeline import compose_clip, compose_follow_bar, compose_split_frame, FFMPEG_BIN

    if frame_type == "follow_bar":
        compose_follow_bar(
            raw_path, out_path, preset,
            title_text=title_text,
            font_file=card_params.get("font_file", "NotoSansTelugu-Bold.ttf"),
            text_color=follow_params.get("text_color", card_params.get("text_color", "#ffff00")),
            text_size=int(card_params.get("font_size", 60)),
            bg_color=follow_params.get("bg_color", "#1a0a2e"),
            follow_text=follow_params.get("follow_text", "FOLLOW KAIZER NEWS TELUGU"),
            follow_text_color=follow_params.get("follow_text_color", "#ffffff"),
            velvet_style=follow_params.get("velvet_style"),
        )
    elif frame_type == "split_frame":
        compose_split_frame(raw_path, image_path, out_path, preset)
    else:
        # torn_card
        # card_style may be stored nested (after editor rerender) or flat in card_params
        # (original pipeline format uses card_c0/card_c1/edge/jag/... at top level).
        # Support both by falling back to the flat card_params keys.
        cs = card_params.get("card_style") or {
            k: v for k, v in card_params.items()
            if k not in ("font_size", "font_file", "text_color")
        }
        compose_clip(
            raw_path, image_path, title_text, out_path, preset,
            font_size=card_params.get("font_size", 80),
            text_color=card_params.get("text_color", "#ffffff"),
            font_file=card_params.get("font_file", "NotoSansTelugu-Bold.ttf"),
            section_pct=section_pct or None,
            card_style=cs or None,
        )

    # Regenerate thumbnail
    thumb_path = clip.thumb_path
    if thumb_path and out_path:
        try:
            subprocess.run(
                [FFMPEG_BIN, "-y", "-i", out_path, "-vframes", "1", "-q:v", "2", thumb_path],
                capture_output=True, check=True, timeout=30,
            )
        except Exception:
            pass

    elapsed = time.time() - t0
    print(f"[rerender] clip={clip.id} done in {elapsed:.1f}s (fast-path)")


@app.post("/api/clips/{clip_id}/upload-image/")
async def upload_image(clip_id: int, image: UploadFile = File(...), db: Session = Depends(get_db)):
    clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    img_dir = MEDIA_ROOT / "images"
    img_dir.mkdir(exist_ok=True)
    img_path = img_dir / f"clip_{clip_id}_{image.filename}"

    with open(img_path, "wb") as f:
        f.write(await image.read())

    clip.image_path = str(img_path)

    # Upload to storage (honours STORAGE_BACKEND). In prod (R2) this
    # is followed by deleting the local copy because Railway's
    # ephemeral disk has a small cap. In local-dev mode the storage
    # IS the local disk, so the post-upload delete is conditional on
    # the backend name below.
    image_storage_url = ""
    try:
        from pipeline_core.storage import get_storage_provider
        storage = get_storage_provider()
        obj = storage.upload(
            str(img_path),
            f"clips/{clip_id}/{img_path.name}",
            content_type=(image.content_type or "image/jpeg"),
        )
        image_storage_url = obj.url
        clip.image_storage_url = image_storage_url
        # Only drop the temp copy when the remote backend OWNS the
        # bytes now (R2 / S3). In local-storage mode the "storage"
        # IS the local disk and deleting would orphan the file. The
        # asset_resolver downloads on demand for the R2 path.
        if storage.name != "local":
            try:
                img_path.unlink(missing_ok=True)
            except Exception as cleanup_exc:
                print(f"[clip-image] cleanup warning: {cleanup_exc}")
    except Exception as exc:
        print(f"[clip-image] storage upload failed for clip {clip_id}: {exc}")

    db.commit()

    return {
        "image_path": str(img_path),
        "image_url": image_storage_url or f"/api/file/?path={img_path}",
    }

# ── Branded download (clip + channel logo overlay) ──────────────────────────

@app.post("/api/clips/{clip_id}/download-with-logo/")
async def download_with_logo(
    clip_id: int,
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Render clip with the requested channel's logo overlaid and stream
    the result as a file download.

    Body: {"channel_id": int}.

    Reuses youtube/logo_overlay.overlay_logo() — same machinery the upload
    worker uses, so the burned-in look matches what publishing produces.
    Falls back to the unbranded clip if the channel has no logo configured.
    Cleans up temp files after the response is fully streamed.
    """
    from fastapi.responses import FileResponse
    from fastapi import BackgroundTasks
    import shutil as _shutil
    import tempfile as _tempfile
    from pathlib import Path as _Path

    channel_id = int((payload or {}).get("channel_id") or 0)
    if not channel_id:
        raise HTTPException(422, "channel_id is required")

    clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(404, "Clip not found")

    # Authorisation: the clip's job must belong to the requesting user.
    job = db.query(models.Job).filter(models.Job.id == clip.job_id).first()
    if not job or job.user_id != user.id:
        raise HTTPException(404, "Clip not found")

    # Resolve channel + logo (Channel has its own logo_asset_id; OAuthToken
    # has another. Worker prefers OAuthToken first; we mirror that.)
    channel = db.query(models.Channel).filter(
        models.Channel.id == channel_id,
        models.Channel.user_id == user.id,
    ).first()
    if not channel:
        raise HTTPException(404, "Channel not found")

    logo_path = ""
    tok = getattr(channel, "oauth_token", None)
    if tok and getattr(tok, "logo_asset_id", None):
        la = db.query(models.UserAsset).filter(
            models.UserAsset.id == tok.logo_asset_id,
            models.UserAsset.user_id == user.id,
        ).first()
        logo_path = _materialize_asset_locally(la)
    if not logo_path and getattr(channel, "logo_asset_id", None):
        la = db.query(models.UserAsset).filter(
            models.UserAsset.id == channel.logo_asset_id,
            models.UserAsset.user_id == user.id,
        ).first()
        logo_path = _materialize_asset_locally(la)

    # Get clip on local disk — download from R2 if not already there.
    cleanup_dirs: list[str] = []
    clip_local = clip.file_path or ""
    if not (clip_local and _Path(clip_local).exists()):
        if not (clip.storage_key and clip.storage_backend):
            raise HTTPException(404, "Clip video file not available")
        try:
            from pipeline_core.storage import get_storage_provider
            provider = get_storage_provider(clip.storage_backend)
            tmp_dir = _tempfile.mkdtemp(prefix="kaizer_dl_")
            tmp_path = str(_Path(tmp_dir) / (_Path(clip.storage_key).name or f"clip_{clip.id}.mp4"))
            provider.download(clip.storage_key, tmp_path)
            clip_local = tmp_path
            cleanup_dirs.append(tmp_dir)
        except Exception as exc:
            raise HTTPException(500, f"Failed to fetch clip from storage: {exc}")

    # Prefer the Pro Editor's beta render when one exists and is newer
    # than the original pipeline output.  The editor writes its result
    # to output/beta_renders/clip_<id>/<style>_beta.mp4 and records a
    # latest.json alongside.  Without this lookup, downloads always
    # served the original render — so any font / colour / style edit
    # the user made in the editor was invisible in the downloaded file.
    try:
        import json as _json
        beta_meta = BASE_DIR / "output" / "beta_renders" / f"clip_{clip_id}" / "latest.json"
        if beta_meta.exists():
            meta = _json.loads(beta_meta.read_text(encoding="utf-8"))
            beta_path = (meta.get("beta_path") or "").strip()
            if beta_path and _Path(beta_path).exists():
                beta_mtime = _Path(beta_path).stat().st_mtime
                orig_mtime = (
                    _Path(clip_local).stat().st_mtime
                    if clip_local and _Path(clip_local).exists()
                    else 0
                )
                if beta_mtime > orig_mtime:
                    print(
                        f"[download-with-logo] using beta render for clip "
                        f"{clip_id}: {beta_path} (style={meta.get('style_pack')})"
                    )
                    clip_local = beta_path
    except Exception as exc:
        # Beta-render lookup is best-effort.  Any failure falls back to
        # the original pipeline output so the user still gets a file.
        print(f"[download-with-logo] beta cache read failed for clip {clip_id}: {exc}")

    # Apply logo overlay (no-op when logo_path is empty).
    branded_path = clip_local
    if logo_path:
        try:
            from youtube import logo_overlay
            branded_path = logo_overlay.overlay_logo(clip_local, logo_path)
            if branded_path and branded_path != clip_local:
                cleanup_dirs.append(str(_Path(branded_path).parent))
        except Exception as exc:
            # Overlay failure → fall back to the clean master so the user
            # still gets a usable file.
            print(f"[download-with-logo] overlay failed for clip {clip_id}: {exc}")
            branded_path = clip_local

    # Build a friendly filename: <channel_slug>_<clip_filename>
    safe_ch = "".join(
        c for c in (channel.name or f"ch{channel.id}")
        if c.isalnum() or c in "-_"
    ) or f"ch{channel.id}"
    clip_basename = clip.filename or f"clip_{clip.id}.mp4"
    download_name = f"{safe_ch}_{clip_basename}"

    # Schedule cleanup after the response has been fully sent.
    bg = BackgroundTasks()
    for d in cleanup_dirs:
        bg.add_task(_shutil.rmtree, d, ignore_errors=True)

    return FileResponse(
        branded_path,
        media_type="video/mp4",
        filename=download_name,
        background=bg,
    )


# ── SEO text download (companion to the video download) ─────────────────────

@app.get("/api/clips/{clip_id}/download-seo/")
async def download_clip_seo(
    clip_id: int,
    channel_id: Optional[int] = None,
    fmt: str = "txt",
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Download a clip's SEO (title / description / tags / hashtags) as a
    text or JSON file the user can paste into YouTube / save alongside
    the downloaded video.

    ``channel_id`` — optional. When supplied, runs the same composer
    used at publish time so the user sees the EXACT title + description
    that would land on that channel (including the brand footer, fixed
    tags, and mandatory hashtags from Channel settings).  When omitted
    we return the clip's generic SEO straight from ``clip.seo``.

    ``fmt`` — ``txt`` (default, copy-paste friendly) or ``json`` (the
    raw structured payload for tooling).
    """
    from fastapi.responses import PlainTextResponse, JSONResponse

    clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(404, "Clip not found")

    # Authorisation — same rule as the video download.
    job = db.query(models.Job).filter(models.Job.id == clip.job_id).first()
    if not job or job.user_id != user.id:
        raise HTTPException(404, "Clip not found")

    # Resolve SEO with a multi-source fallback chain so we never 404 a
    # clip that has SEO somewhere, just not in the column we looked at
    # first.  Order:
    #   1. clip.seo (generic JSON, new path) + composer for the channel
    #   2. clip.seo_variants[channel_id] (legacy per-channel variant)
    #   3. clip.seo (generic, no channel scoping) — when channel lookup fails
    #   4. clip.seo_variants[any] — first available variant, marked as such
    import json as _json
    seo: dict = {}
    composed_for = ""
    notes: list[str] = []

    # Parse both columns up-front so we can fall back between them.
    try:
        generic_seo = _json.loads(clip.seo or "{}") if clip.seo else {}
        if not isinstance(generic_seo, dict):
            generic_seo = {}
    except Exception:
        generic_seo = {}
    try:
        variants = _json.loads(clip.seo_variants or "{}") if clip.seo_variants else {}
        if not isinstance(variants, dict):
            variants = {}
    except Exception:
        variants = {}

    if channel_id:
        channel = db.query(models.Channel).filter(
            models.Channel.id == int(channel_id),
            models.Channel.user_id == user.id,
        ).first()
        if not channel:
            raise HTTPException(404, "Channel not found")
        composed_for = channel.name or f"channel #{channel.id}"

        # Path 1 — generic SEO + composer overlay.  Best result.
        if generic_seo.get("title"):
            try:
                from seo.composer import compose
                seo = compose(generic_seo, channel, publish_kind="video")
            except Exception as exc:
                print(f"[download-seo] composer failed for clip {clip_id}: {exc}")
                seo = generic_seo  # raw fallback

        # Path 2 — legacy per-channel variant.  Older clips that were
        # generated before the generic+overlay refactor live here.
        if not seo:
            variant = (
                variants.get(str(channel_id))
                or variants.get(int(channel_id))
                if isinstance(variants, dict) else None
            )
            if isinstance(variant, dict) and variant.get("title"):
                seo = variant
                notes.append("legacy per-channel variant")
    else:
        # No channel_id — return the generic SEO as-is.
        if generic_seo.get("title"):
            seo = generic_seo

    # Final fallback — any populated variant the clip has at all.  Tag
    # it so the user knows it's not channel-specific.
    if not seo and variants:
        for key, variant in variants.items():
            if isinstance(variant, dict) and variant.get("title"):
                seo = variant
                notes.append(f"using variant for channel {key} (no exact match)")
                break

    if not seo:
        raise HTTPException(
            404,
            "No SEO generated for this clip yet. Open the clip in the Editor "
            "and click 'Generate SEO' first.",
        )

    # JSON format — return the dict as a downloadable .json.
    if (fmt or "").lower() == "json":
        download_name = f"clip_{clip_id}_seo"
        if composed_for:
            slug = "".join(c for c in composed_for if c.isalnum() or c in "-_") or "channel"
            download_name = f"{slug}_clip_{clip_id}_seo"
        return JSONResponse(
            content=seo,
            headers={
                "Content-Disposition": f'attachment; filename="{download_name}.json"',
            },
        )

    # TXT format — human-friendly with section labels. Mirrors the
    # exact field order YouTube's "Add details" page expects, so the
    # user can copy each section straight into the upload form.
    title       = (seo.get("title") or "").strip()
    description = (seo.get("description") or "").strip()
    keywords    = seo.get("keywords") or seo.get("tags") or []
    hashtags    = seo.get("hashtags") or []
    seo_score   = seo.get("seo_score")
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]
    if isinstance(hashtags, str):
        hashtags = [h.strip() for h in hashtags.split() if h.strip()]

    lines: list[str] = []
    if composed_for:
        lines.append(f"# Composed for channel: {composed_for}")
        lines.append("")
    lines.append("=" * 60)
    lines.append("TITLE")
    lines.append("=" * 60)
    lines.append(title or "(no title)")
    lines.append("")
    lines.append("=" * 60)
    lines.append("DESCRIPTION")
    lines.append("=" * 60)
    lines.append(description or "(no description)")
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"TAGS / KEYWORDS ({len(keywords)})")
    lines.append("=" * 60)
    lines.append(", ".join(keywords) if keywords else "(none)")
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"HASHTAGS ({len(hashtags)})")
    lines.append("=" * 60)
    lines.append(" ".join(hashtags) if hashtags else "(none)")
    if seo_score is not None:
        lines.append("")
        lines.append(f"SEO score: {seo_score}/100")

    download_name = f"clip_{clip_id}_seo"
    if composed_for:
        slug = "".join(c for c in composed_for if c.isalnum() or c in "-_") or "channel"
        download_name = f"{slug}_clip_{clip_id}_seo"
    return PlainTextResponse(
        content="\n".join(lines),
        headers={
            "Content-Disposition": f'attachment; filename="{download_name}.txt"',
        },
    )


# ── File serving (path-restricted, with Range support) ──────────────────────

def _allowed_file_roots() -> list[Path]:
    """Absolute paths the /api/file/ endpoint is permitted to read from."""
    roots = [
        BASE_DIR / "output",
        BASE_DIR / "media",
        MEDIA_ROOT,
        OUTPUT_ROOT,
    ]
    return [r.resolve() for r in roots if r.exists() or True]


def _is_safe_path(p: Path) -> bool:
    """Block path traversal: requested file must resolve under an allowed root."""
    try:
        resolved = p.resolve()
    except Exception:
        return False
    for root in _allowed_file_roots():
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


@app.get("/api/file/")
async def serve_file(path: str, request: Request):
    file_path = Path(path)
    if not _is_safe_path(file_path):
        raise HTTPException(status_code=403, detail="Path is not under an allowed root")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found — clips are ephemeral and expire on redeploy")

    mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    file_size = file_path.stat().st_size
    filename = file_path.name
    range_header = request.headers.get("range", "")

    # Common headers for all responses
    base_headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Access-Control-Expose-Headers": "Content-Disposition, Content-Length, Accept-Ranges",
    }

    if range_header.startswith("bytes="):
        start_str, _, end_str = range_header[6:].partition("-")
        start = int(start_str)
        end = int(end_str) if end_str else file_size - 1
        length = end - start + 1

        def _iter():
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            _iter(), status_code=206, media_type=mime,
            headers={
                **base_headers,
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Content-Length": str(length),
            },
        )

    return StreamingResponse(
        open(file_path, "rb"), media_type=mime,
        headers={**base_headers, "Content-Length": str(file_size)},
    )

# ── Helpers ──────────────────────────────────────────────────────────────────

def _clip_dict(c):
    def _furl(path):
        """Build /api/file/ URL — no exists() check; let endpoint handle 404."""
        return f"/api/file/?path={path}" if path else ""

    meta = json.loads(c.meta or "{}")
    raw_path = meta.get("raw_path", "")

    seo = None
    if c.seo:
        try:
            seo = json.loads(c.seo)
        except (ValueError, TypeError):
            seo = None

    # Prefer R2 URLs when populated; fall back to /api/file/ for legacy rows
    # so old uploads still render until the migration script runs.
    thumb_resolved = (
        getattr(c, "thumb_storage_url", "") or _furl(c.thumb_path)
    )
    image_resolved = (
        getattr(c, "image_storage_url", "") or _furl(c.image_path)
    )
    video_resolved = (
        (c.storage_url if (c.storage_backend or "") == "r2" else "")
        or _furl(c.file_path)
    )

    return {
        "id":           c.id,
        "job_id":       c.job_id,
        "clip_index":   c.clip_index,
        "filename":     c.filename,
        "file_path":    c.file_path,
        "thumb_path":   c.thumb_path or "",
        "thumb_url":    thumb_resolved,
        "image_path":   c.image_path or "",
        "image_url":    image_resolved,
        "raw_url":      _furl(raw_path),
        "duration":     c.duration,
        "frame_type":   c.frame_type,
        "text":         c.text,
        "sentiment":    c.sentiment,
        "entities":     json.loads(c.entities or "[]"),
        "card_params":  json.loads(c.card_params or "{}"),
        "section_pct":  json.loads(c.section_pct or "{}"),
        "follow_params":json.loads(c.follow_params or "{}"),
        "meta":         meta,
        "video_url":    video_resolved,
        "seo":          seo,
        "seo_variants": (lambda raw: (json.loads(raw) if raw else {}) or {})(getattr(c, "seo_variants", "") or "{}"),
    }


def _estimate_progress(log_lines: list, status: str) -> int:
    if status == "done":
        return 100
    if status == "failed":
        return 0
    # Estimate from pipeline step markers in log
    steps_found = sum(1 for l in log_lines if "STEP" in l.upper())
    return min(90, steps_found * 9)


# ── V2 Inngest serve mount (Step 12.2b) ──────────────────────────────────────
# Mounts the V2 Inngest webhook at /api/inngest so the Inngest Dev Server
# (and production Inngest Cloud) can discover process_video_v2 and drive
# its step execution. Guarded by KAIZER_V2_ENABLED so V1-only deployments
# get a byte-identical route table.
#
# Placed at the END of main.py so all V1 routes are declared first; the
# inngest serve registers via @app.get/@app.post decorators which would
# otherwise be shadowed if any V1 route shared the /api/inngest path
# (none do today, but this guarantees the property forward).
if _v2_enabled():
    import sys as _sys
    import os as _os
    _pipeline_v2_dir = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)),
        "pipeline_v2",
    )
    if _pipeline_v2_dir not in _sys.path:
        _sys.path.insert(0, _pipeline_v2_dir)
    from pipeline_v2.inngest_app import register_v2_inngest
    register_v2_inngest(app)
