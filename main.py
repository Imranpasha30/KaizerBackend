import os
import json
import time
import shutil
import mimetypes
import asyncio
import sys
import threading as _threading
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

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Request, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse
from sqlalchemy.orm import Session, selectinload
from dotenv import load_dotenv

# ── Desktop mode (KAIZER_DESKTOP=1) ──────────────────────────────────────────
# Set by desktop_entry.py BEFORE this module imports. One backend codebase,
# two shapes: SaaS (default — everything on) and desktop (localhost-only,
# single user, SQLite in userData, publish/scheduler/admin surfaces off).
# _DESKTOP is defined HERE (before dotenv/CORS/routers) because every gate
# below needs it; the router-registration section references it heavily.
_DESKTOP = (os.environ.get("KAIZER_DESKTOP", "") or "").strip() == "1"

if _DESKTOP:
    # Desktop: desktop_entry already set the authoritative env (DATABASE_URL
    # → userData SQLite, KAIZER_OUTPUT_ROOT, KAIZER_ENV_DIR). The SaaS-style
    # load_dotenv(override=True) below would let an install-dir .env STOMP
    # those (the documented ".env beats process env" gotcha) and silently
    # point the desktop at a server database. Instead load ONLY the userData
    # .env (saved API keys), never overriding what desktop_entry set.
    _kx_env_dir = (os.environ.get("KAIZER_ENV_DIR", "") or "").strip()
    if _kx_env_dir:
        load_dotenv(Path(_kx_env_dir) / ".env", override=False)
else:
    # override=True so a `.env` edit + uvicorn restart actually replaces
    # any stale value already present in the parent shell's environment.
    # Without override, load_dotenv silently keeps the OS-level value when
    # both exist, which led to "key still empty" surprises during dev.
    load_dotenv(override=True)

from database import engine, SessionLocal, Base, get_db
import models
import insights.models  # noqa: F401 — registers insights_* tables on Base before create_all
import runner
import auth
from rate_limit import rate_limited  # Wave 2 — per-tenant token buckets on heavy POSTs

from routers.auth import router as auth_router
from routers.channels import router as channels_router
from routers.seo import router as seo_router
from routers.youtube_oauth import router as youtube_oauth_router
from routers.youtube_upload import router as youtube_upload_router
from routers.publish_tasks import router as publish_tasks_router  # Phase 1.B — new publish path (KAIZER_NEW_PUBLISH_PATH gated)
from insights.router import router as insights_router  # Insights / Trend Finder (isolated module)
from routers.meta_oauth import router as meta_oauth_router
from routers.linkedin_oauth import router as linkedin_oauth_router
from routers.youtube_quota import router as youtube_quota_router
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
from routers.onboarding import router as onboarding_router
from routers.help import router as help_router
from routers.analytics_ai import router as analytics_ai_router
from routers.bulletin_images import router as bulletin_images_router
from routers.express_mode import router as express_mode_router
from routers.heygen import router as heygen_router
from routers.live_studio import router as live_studio_router
from routers.v4_editor import router as v4_editor_router
from routers.v4_defaults import router as v4_defaults_router
from routers.library    import router as library_router
from routers.profile    import router as profile_router
from routers.metrics    import router as metrics_router            # Phase 3.G — Prometheus /metrics
from routers.admin_upload_v2 import router as admin_upload_v2_router  # Phase 3.G — admin observability page
from routers.ws_progress import router as ws_progress_router       # Wave 3 — WebSocket live progress
from routers.quick_publish import router as quick_publish_router   # Quick Publish — SEO + thumbnail glue
from routers.custom_templates import router as custom_templates_router  # developer-uploaded HTML/CSS templates
from routers.avatar import router as avatar_router      # News Anchor — provider-agnostic AI presenter (ported kaizer-platform@d5fd482)
from routers.podcast import router as podcast_router    # Podcast editor — cutlist/punch-in/promo, ffmpeg or Remotion (ported)
from routers.desktop import router as desktop_router    # Desktop app licensing — activate/heartbeat/revoke (ported, Phase 4a)
from seo.default_channels import seed_channels
from youtube import worker as upload_worker
from learning import scheduler as corpus_scheduler


def _migrate_schema():
    """Add missing columns to existing tables — safe to run on every startup."""
    from sqlalchemy import text, inspect
    # Postiz connect-sessions: drop the legacy per-provider UNIQUE so a team's
    # members can connect the same platform concurrently (team-shared model).
    # The table is transient (short-lived rows) → drop+recreate loses nothing.
    # Idempotent: only rebuilds when the old unique constraint is present.
    try:
        _insp0 = inspect(engine)
        if "postiz_connect_sessions" in _insp0.get_table_names():
            _uc = _insp0.get_unique_constraints("postiz_connect_sessions")
            if any("provider" in (u.get("column_names") or []) for u in _uc):
                with engine.begin() as _c0:
                    _c0.execute(text("DROP TABLE postiz_connect_sessions"))
        if "postiz_connect_sessions" not in inspect(engine).get_table_names():
            _t = Base.metadata.tables.get("postiz_connect_sessions")
            if _t is not None:
                _t.create(bind=engine)
    except Exception as _e:
        print(f"[startup] postiz_connect_sessions migration skipped: {_e}")
    # master_videos.source_upload_id was UNIQUE (one master per job). The model
    # dropped that — V4 produces one master PER CLIP (the full video + each
    # short share a job) — but an existing DB still carries the UNIQUE index,
    # so the 2nd clip of a job fails to materialise with a UniqueViolation
    # ("Unable to materialise MasterVideo …"). Rebuild the index non-unique.
    try:
        _insp1 = inspect(engine)
        if "master_videos" in _insp1.get_table_names():
            for _ix in _insp1.get_indexes("master_videos"):
                if _ix.get("unique") and _ix.get("column_names") == ["source_upload_id"]:
                    with engine.begin() as _c1:
                        _c1.execute(text(f'DROP INDEX IF EXISTS {_ix["name"]}'))
                        _c1.execute(text(
                            f'CREATE INDEX IF NOT EXISTS {_ix["name"]} '
                            f'ON master_videos (source_upload_id)'
                        ))
                    print(f"[startup] master_videos.source_upload_id UNIQUE → non-unique ({_ix['name']})")
    except Exception as _e:
        print(f"[startup] master_videos source_upload_id index fix skipped: {_e}")
    # training_samples.impressions — real thumbnail impressions from the
    # YouTube Reporting API (learning/seo_learning.py real-CTR ingest).
    # Nullable add: existing rows read NULL until the next learn pass.
    try:
        _inspTS = inspect(engine)
        if "training_samples" in _inspTS.get_table_names():
            _tscols = {c["name"] for c in _inspTS.get_columns("training_samples")}
            if "impressions" not in _tscols:
                with engine.begin() as _cTS:
                    _cTS.execute(text(
                        "ALTER TABLE training_samples ADD COLUMN impressions BIGINT"))
                print("[startup] training_samples.impressions added")
    except Exception as _e:
        print(f"[startup] training_samples.impressions migration skipped: {_e}")
    # SEO competitor-intelligence wave (all nullable/default adds):
    #   channel_videos.tags            — public tags for harvest learning
    #   seo_learning_snapshots.kind    — 'own' | 'competitor' id namespaces
    #   training_samples.explored_hook — the A/B exploration ledger
    #   channels.use_competitor_intel  — the per-channel opt-in toggle
    try:
        _inspCI = inspect(engine)
        _adds = [
            ("users", "seo_engine",
             "ALTER TABLE users ADD COLUMN seo_engine VARCHAR(10) DEFAULT 'gemini'"),
            ("channel_videos", "tags", "ALTER TABLE channel_videos ADD COLUMN tags JSON"),
            ("seo_learning_snapshots", "kind",
             "ALTER TABLE seo_learning_snapshots ADD COLUMN kind VARCHAR(12) NOT NULL DEFAULT 'own'"),
            ("training_samples", "explored_hook",
             "ALTER TABLE training_samples ADD COLUMN explored_hook VARCHAR(12)"),
            ("channels", "use_competitor_intel",
             "ALTER TABLE channels ADD COLUMN use_competitor_intel BOOLEAN DEFAULT FALSE"),
            # Template remix: creator opt-in for others to fork+edit a template.
            ("custom_templates", "allow_remix",
             "ALTER TABLE custom_templates ADD COLUMN allow_remix BOOLEAN NOT NULL DEFAULT TRUE"),
            ("channels", "content_type_default",
             "ALTER TABLE channels ADD COLUMN content_type_default VARCHAR(20)"),
            # Live Studio branding conveyor: per-stream opt-in to stamp the
            # channel's logo before going live, plus the stamped temp file.
            #
            # Without these two, EVERY attempt to go live fails: creating a
            # batch inserts a LiveStream row, SQLAlchemy names all the model's
            # columns, and Postgres rejects the statement -- a 500 with
            # nothing useful on the client. create_all does not help, because
            # it only creates missing TABLES and never alters existing ones.
            ("live_streams", "apply_branding",
             "ALTER TABLE live_streams ADD COLUMN apply_branding BOOLEAN NOT NULL DEFAULT FALSE"),
            ("live_streams", "branded_path",
             "ALTER TABLE live_streams ADD COLUMN branded_path VARCHAR(512)"),
            ("onboarding_profiles", "channel_id",
             "ALTER TABLE onboarding_profiles ADD COLUMN channel_id VARCHAR(64)"),
            ("onboarding_profiles", "channel_title",
             "ALTER TABLE onboarding_profiles ADD COLUMN channel_title VARCHAR(200)"),
        ]
        for _tbl, _col, _sql in _adds:
            if _tbl in _inspCI.get_table_names():
                _cols = {c["name"] for c in _inspCI.get_columns(_tbl)}
                if _col not in _cols:
                    with engine.begin() as _cCI:
                        _cCI.execute(text(_sql))
                    print(f"[startup] {_tbl}.{_col} added")
    except Exception as _e:
        print(f"[startup] competitor-intel migrations skipped: {_e}")
    # Per-user Google key minting: mark any mint left 'pending' by a
    # mid-mint restart as failed so the admin sees it's retryable. New
    # tables (google_minted_keys, billing_rates) are auto-created by
    # create_all — no ALTER needed.
    try:
        from services.google_key_minter import sweep_stale_pending
        _swept = sweep_stale_pending()
        if _swept:
            print(f"[startup] {_swept} stale pending key-mints marked failed")
    except Exception as _e:
        print(f"[startup] key-mint sweep skipped: {_e}")
    # upload_jobs_v2 gained privacy_status + publish_at so the user's
    # public/unlisted/private choice (and scheduled publish time) carried by
    # the PublishRequest survives fan-out to the upload worker. Without them
    # every V2 upload hard-defaulted to 'private' regardless of selection.
    try:
        _insp2 = inspect(engine)
        if "upload_jobs_v2" in _insp2.get_table_names():
            _ujcols = {c["name"] for c in _insp2.get_columns("upload_jobs_v2")}
            with engine.begin() as _c2:
                if "privacy_status" not in _ujcols:
                    _c2.execute(text(
                        "ALTER TABLE upload_jobs_v2 ADD COLUMN privacy_status "
                        "VARCHAR(20) NOT NULL DEFAULT 'private'"))
                    print("[startup] upload_jobs_v2.privacy_status added")
                if "publish_at" not in _ujcols:
                    _c2.execute(text(
                        "ALTER TABLE upload_jobs_v2 ADD COLUMN publish_at "
                        "TIMESTAMP WITH TIME ZONE"))
                    print("[startup] upload_jobs_v2.publish_at added")
                # brand_mode: 'per_channel' (overlay this channel's branding)
                # | 'as_is' (source already branded — upload verbatim).
                if "brand_mode" not in _ujcols:
                    _c2.execute(text(
                        "ALTER TABLE upload_jobs_v2 ADD COLUMN brand_mode "
                        "VARCHAR(16) NOT NULL DEFAULT 'per_channel'"))
                    print("[startup] upload_jobs_v2.brand_mode added")
                # brand_placement: 'template' (use the template's logo/watermark
                # slot) | 'channel' (use this channel's own position instead).
                if "brand_placement" not in _ujcols:
                    _c2.execute(text(
                        "ALTER TABLE upload_jobs_v2 ADD COLUMN brand_placement "
                        "VARCHAR(16) NOT NULL DEFAULT 'template'"))
                    print("[startup] upload_jobs_v2.brand_placement added")
                # Per-publish YouTube setting OVERRIDES (win over the channel's
                # yt_* defaults at upload). NULL = no override → use channel default.
                if "yt_category_id" not in _ujcols:
                    _c2.execute(text("ALTER TABLE upload_jobs_v2 ADD COLUMN yt_category_id VARCHAR(10)"))
                    print("[startup] upload_jobs_v2.yt_category_id added")
                if "yt_default_language" not in _ujcols:
                    _c2.execute(text("ALTER TABLE upload_jobs_v2 ADD COLUMN yt_default_language VARCHAR(10)"))
                    print("[startup] upload_jobs_v2.yt_default_language added")
                if "yt_playlist_id" not in _ujcols:
                    _c2.execute(text("ALTER TABLE upload_jobs_v2 ADD COLUMN yt_playlist_id VARCHAR(64)"))
                    print("[startup] upload_jobs_v2.yt_playlist_id added")
                if "yt_license" not in _ujcols:
                    _c2.execute(text("ALTER TABLE upload_jobs_v2 ADD COLUMN yt_license VARCHAR(20)"))
                    print("[startup] upload_jobs_v2.yt_license added")
                if "yt_made_for_kids" not in _ujcols:
                    _c2.execute(text("ALTER TABLE upload_jobs_v2 ADD COLUMN yt_made_for_kids BOOLEAN"))
                    print("[startup] upload_jobs_v2.yt_made_for_kids added")
    except Exception as _e:
        print(f"[startup] upload_jobs_v2 privacy columns migration skipped: {_e}")
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
            # Item 114 (Stage 2 provider catalog): per-job LLM choice
            # for Stage 2 editorial decisions. One of {"gemini",
            # "claude"}. NULL on pre-item-114 rows; dispatcher falls
            # back to "gemini".
            "stage_2_provider":  "VARCHAR(20) DEFAULT 'gemini'",
            # V4 only: which model decided Step 1's KEEP/CUT plan
            # ("claude" or "gemini"). NULL on pre-rollout rows;
            # trim_engine._select_planner falls back to "claude".
            "v4_trim_planner":   "VARCHAR(20) DEFAULT 'claude'",
            # V4 only: which provider generated per-story images
            # ("auto", "gemini", or "openai"). NULL on pre-rollout
            # rows; image_provider._selected_image_provider falls
            # back to "auto" (V1 multi-source chain).
            "v4_image_provider": "VARCHAR(20) DEFAULT 'auto'",
            # V4 only: operator-supplied bulletin description.
            # When NON-NULL the orchestrator switches to
            # "source-preserved" mode (no Claude trim, no AI desc).
            "v4_predefined_description": "TEXT",
            # V4 only: render-output choice ("both" / "full-only" /
            # "shorts-only"). NULL on pre-rollout rows; orchestrator +
            # runner fall back to "both".
            "v4_output_format": "VARCHAR(20) DEFAULT 'both'",
            # V4 only: full-form effects mode ("auto"/"rich"/"off").
            # NULL = legacy job -> byte-identical render, warm cache.
            "v4_effects_mode": "VARCHAR(12)",
            "v4_theme": "VARCHAR(24)",
            # V4 only: user-directed effect picks ("edit using THESE") — a JSON
            # object of per-category id lists. NULL = full AI-Director autonomy.
            "v4_style_directives": "TEXT",
            # V4 only: which AI Director engine plans direction —
            # "v4" (ours, default) | "platform" (ported 3-layer engine).
            "v4_director_engine": "VARCHAR(12) DEFAULT 'v4'",
            # V4 only: original publish target ("instagram"/"youtube"/
            # "facebook"). Editor leads with this platform's SEO.
            "v4_target_platform": "VARCHAR(20) DEFAULT 'youtube'",
            # V4 Stage 2: defer the up-front MP4 render (edit-first; export on demand).
            "v4_defer_render": "BOOLEAN DEFAULT FALSE",
            # V4 audio-first mode (narration master + optional muted b-roll).
            "v4_audio_first": "BOOLEAN DEFAULT FALSE",
            # V4 only: channels chosen at generate time (JSON list of Channel
            # ids). Recorded by the New Job "Choose channels" step.
            "target_channel_ids": "TEXT",
            # Per-job INTRO override (UserAsset id). NULL = each channel uses
            # its own assigned intro. Set at job creation (assigned/demo/upload).
            "intro_asset_id": "INTEGER",
            # PER-CHANNEL intro overrides: JSON map {"<channel_id>": <asset_id>}.
            # A channel present uses that intro for this job; absent = its own.
            "intro_overrides": "TEXT",
            # Custom-template per-slot media: JSON {slot_key: asset_id} + the
            # slot whose video is the AI-trim "main".
            "template_media":  "TEXT",
            "main_media_slot": "VARCHAR(64)",
            # Per-slot text overrides set in the custom-template editor (JSON {slot:text}).
            "template_overrides": "TEXT",
            # Per-job custom-template HTML overrides: JSON {"<target>:<index>": "<html>"}.
            # Set when the operator visually edits the template for THIS job in the inline
            # builder. Renderer uses it verbatim; parent template untouched. Sanitized on save.
            "custom_html_overrides": "TEXT",
            "fullform_layout": "VARCHAR(50)",
            # Wave 2 (API scale): cached jobs-list cover image. Lazily
            # backfilled by list_jobs after its first stat()-walk so
            # subsequent listings stop hammering the filesystem. NULL =
            # not yet resolved (running jobs, pre-Wave-2 rows).
            "thumb_url":    "VARCHAR(500)",
            "thumb_aspect": "VARCHAR(8)",
            # V4 only: which BRAIN writes the Director's plan ("gemini"
            # default | "claude" | "openai"). Missing here meant creating a
            # job failed the same way going live did.
            "v4_director_provider": "VARCHAR(20) DEFAULT 'gemini'",
        }
        for col, dtype in job_additions.items():
            if col not in existing_jobs:
                conn.execute(text(f"ALTER TABLE jobs ADD COLUMN {col} {dtype}"))

        # ── custom_templates: preview-modal metadata + community rating ──
        if inspector.has_table("custom_templates"):
            _ct_cols = {c["name"] for c in inspector.get_columns("custom_templates")}
            for col, dtype in (("when_to_use", "TEXT"), ("how_to_use", "TEXT"),
                               ("rating_sum", "INTEGER"), ("rating_count", "INTEGER"),
                               ("derived_from", "INTEGER"),
                               ("is_builtin", "BOOLEAN"),
                               ("format", "VARCHAR(8) DEFAULT 'html'")):
                if col not in _ct_cols:
                    conn.execute(text(f"ALTER TABLE custom_templates ADD COLUMN {col} {dtype}"))

        # ── user_id on multi-tenant tables ──────────────────────────────
        for tbl in ("channels", "upload_jobs", "campaigns", "competitor_channels"):
            if inspector.has_table(tbl):
                cols = {c["name"] for c in inspector.get_columns(tbl)}
                if "user_id" not in cols:
                    conn.execute(text(f"ALTER TABLE {tbl} ADD COLUMN user_id INTEGER"))

        # ── channels.kind: 'account' (own channel, publishes) vs 'style'
        #    (competitor / style reference). Persistent so a DISCONNECTED
        #    account (token removed) no longer looks like a style reference
        #    and pollutes the SEO Settings tab. One-time backfill on the rows
        #    that exist before this column: connected OR no-competitor-signal
        #    -> 'account'; a competitor signal (handle / learned corpus /
        #    title formula) -> 'style'.
        if inspector.has_table("channels"):
            ch_cols = {c["name"] for c in inspector.get_columns("channels")}
            if "kind" not in ch_cols:
                conn.execute(text("ALTER TABLE channels ADD COLUMN kind VARCHAR(10)"))
                conn.execute(text("""
                    UPDATE channels SET kind = 'style'
                    WHERE kind IS NULL
                      AND id NOT IN (
                          SELECT channel_id FROM oauth_tokens
                          WHERE refresh_token_enc IS NOT NULL AND refresh_token_enc <> ''
                      )
                      AND (
                          (handle IS NOT NULL AND handle <> '')
                          OR (title_formula IS NOT NULL AND title_formula <> '')
                          OR id IN (SELECT channel_id FROM channel_corpus)
                      )
                """))
                conn.execute(text("UPDATE channels SET kind = 'account' WHERE kind IS NULL"))

        # ── users.heygen_avatar_id / heygen_voice_id (Trending → HeyGen) ──
        if inspector.has_table("users"):
            user_cols = {c["name"] for c in inspector.get_columns("users")}
            if "heygen_avatar_id" not in user_cols:
                conn.execute(text("ALTER TABLE users ADD COLUMN heygen_avatar_id VARCHAR(64)"))
            if "heygen_voice_id" not in user_cols:
                conn.execute(text("ALTER TABLE users ADD COLUMN heygen_voice_id VARCHAR(64)"))
            # Library role: creative users can upload to the shared
            # company library. Defaults to FALSE for existing rows.
            if "is_creative" not in user_cols:
                conn.execute(text(
                    "ALTER TABLE users ADD COLUMN is_creative BOOLEAN NOT NULL DEFAULT FALSE"
                ))
            # Profile picture (image or GIF) R2 key, plus the cached
            # creator rating aggregate (sum + count). All additive,
            # all default to safe zero/empty values.
            if "avatar_key" not in user_cols:
                conn.execute(text(
                    "ALTER TABLE users ADD COLUMN avatar_key VARCHAR(500) NOT NULL DEFAULT ''"
                ))
            if "creator_rating_sum" not in user_cols:
                conn.execute(text(
                    "ALTER TABLE users ADD COLUMN creator_rating_sum INTEGER NOT NULL DEFAULT 0"
                ))
            if "creator_rating_count" not in user_cols:
                conn.execute(text(
                    "ALTER TABLE users ADD COLUMN creator_rating_count INTEGER NOT NULL DEFAULT 0"
                ))

        # ── library_items: category + cached rating aggregate ──────────
        # Tables themselves are created by Base.metadata.create_all on
        # first run; this guard only handles upgrades on existing rows.
        if inspector.has_table("library_items"):
            li_cols = {c["name"] for c in inspector.get_columns("library_items")}
            if "category_id" not in li_cols:
                conn.execute(text(
                    "ALTER TABLE library_items ADD COLUMN category_id INTEGER"
                ))
            if "rating_sum" not in li_cols:
                conn.execute(text(
                    "ALTER TABLE library_items ADD COLUMN rating_sum INTEGER NOT NULL DEFAULT 0"
                ))
            if "rating_count" not in li_cols:
                conn.execute(text(
                    "ALTER TABLE library_items ADD COLUMN rating_count INTEGER NOT NULL DEFAULT 0"
                ))
            # Lazy-rendered watermarked copy for free-tier downloads.
            if "watermark_key" not in li_cols:
                conn.execute(text(
                    "ALTER TABLE library_items ADD COLUMN watermark_key VARCHAR(500) DEFAULT ''"
                ))
            # Watch counter — bumped on every play-ticket request.
            if "watch_count" not in li_cols:
                conn.execute(text(
                    "ALTER TABLE library_items ADD COLUMN watch_count INTEGER NOT NULL DEFAULT 0"
                ))

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
            # Per-video custom thumbnail (Live Studio thumbnail upload).
            # Nullable string; one image path replicated across all
            # streams that share a (batch_id, video_slot).
            if "thumbnail_path" not in ls_cols:
                conn.execute(text(
                    "ALTER TABLE live_streams ADD COLUMN thumbnail_path VARCHAR(512)"
                ))
            # Origin URL when the stream was ingested from a YouTube
            # link via yt-dlp instead of a direct chunked upload.
            if "source_url" not in ls_cols:
                conn.execute(text(
                    "ALTER TABLE live_streams ADD COLUMN source_url VARCHAR(1024)"
                ))

        # ── channels.logo_asset_id (per-channel video overlay logo) ─────
        if inspector.has_table("channels"):
            cols = {c["name"] for c in inspector.get_columns("channels")}
            if "logo_asset_id" not in cols:
                conn.execute(text("ALTER TABLE channels ADD COLUMN logo_asset_id INTEGER"))
            # Per-channel INTRO video (FK user_assets) — concatenated at the
            # head of the branded clip in the branding pass (anti-duplicate).
            if "intro_asset_id" not in cols:
                conn.execute(text("ALTER TABLE channels ADD COLUMN intro_asset_id INTEGER"))
            # Per-channel YouTube publish defaults (set from Kaizer).
            if "yt_category_id" not in cols:
                conn.execute(text("ALTER TABLE channels ADD COLUMN yt_category_id VARCHAR(10)"))
            if "yt_playlist_id" not in cols:
                conn.execute(text("ALTER TABLE channels ADD COLUMN yt_playlist_id VARCHAR(64)"))
            if "yt_default_language" not in cols:
                conn.execute(text("ALTER TABLE channels ADD COLUMN yt_default_language VARCHAR(10)"))
            if "yt_made_for_kids" not in cols:
                _bt = "BOOLEAN" if engine.dialect.name == "postgresql" else "INTEGER"
                conn.execute(text(f"ALTER TABLE channels ADD COLUMN yt_made_for_kids {_bt}"))
            if "yt_license" not in cols:
                conn.execute(text("ALTER TABLE channels ADD COLUMN yt_license VARCHAR(20)"))
            # Per-channel upload route override.  Null = use system default.
            if "upload_provider" not in cols:
                conn.execute(text("ALTER TABLE channels ADD COLUMN upload_provider VARCHAR(20)"))
            # Per-channel watermark — applied at upload time so each
            # destination gets its own brand stamp on the same render.
            if "watermark_text" not in cols:
                conn.execute(text("ALTER TABLE channels ADD COLUMN watermark_text VARCHAR(64) DEFAULT ''"))
            if "watermark_opacity" not in cols:
                conn.execute(text("ALTER TABLE channels ADD COLUMN watermark_opacity FLOAT DEFAULT 0.35"))
            if "watermark_position" not in cols:
                conn.execute(text("ALTER TABLE channels ADD COLUMN watermark_position VARCHAR(16) DEFAULT 'top-right'"))
            # Per-channel social links (JSON) — appended to the SEO
            # description footer at publish time so each channel's own
            # @handles get the reach boost instead of the user's.
            if "socials" not in cols:
                col_type = "JSONB" if engine.dialect.name == "postgresql" else "TEXT"
                conn.execute(text(f"ALTER TABLE channels ADD COLUMN socials {col_type}"))
            # Postiz delivery: the integration id this channel maps to in
            # Postiz (GET /public/v1/integrations). NULL = not bound.
            if "postiz_integration_id" not in cols:
                conn.execute(text("ALTER TABLE channels ADD COLUMN postiz_integration_id VARCHAR(64)"))

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
            # Subject label ("name-tag contract") — what/who the image
            # shows; the image↔speech timing AI matches by this label.
            if "description" not in cols:
                conn.execute(text(
                    "ALTER TABLE user_assets ADD COLUMN description TEXT DEFAULT ''"
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

            # V4 automation: per-user JSON blob with auto-pipeline defaults
            # so the wizard can be skipped end-to-end.
            if "v4_defaults" not in ucols:
                conn.execute(text("ALTER TABLE users ADD COLUMN v4_defaults TEXT"))

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

        # ── Phase 1 — Upload rewrite v2 schema (additive only) ─────────────
        # Adds: plan_tiers, brand_profiles, master_videos, publish_tasks,
        # upload_jobs_v2, publish_attempts, credit_ledger, quota_burn_log,
        # and users.plan_tier_id (with Pro backfill).
        # See docs/upload-rewrite/CONTRACTS.md §3 for column specs.
        # See docs/upload-rewrite/DECISIONS.md Decisions 3, 4, 6, 8 for the
        # plan_tier seed values + the Pro-default backfill rule.
        try:
            # 1) Create the 8 new tables. We pass the explicit table list
            #    (rather than calling create_all() on full metadata) for two
            #    reasons: (a) it stays surgical even if a future agent adds
            #    a half-written table to models.py, and (b) on Postgres
            #    this avoids the flaky "create_all sees all tables every
            #    boot" behaviour the existing channel_videos/live_events
            #    block above already works around.
            upload_v2_tables = [
                "plan_tiers",       # no FK deps
                "master_videos",    # depends on jobs
                "brand_profiles",   # depends on user_assets
                "publish_tasks",    # depends on users, master_videos
                "upload_jobs_v2",   # depends on publish_tasks, channels, oauth_tokens, brand_profiles, clips
                "publish_attempts", # depends on upload_jobs_v2
                "credit_ledger",    # depends on users, upload_jobs_v2
                "quota_burn_log",   # depends on upload_jobs_v2
            ]
            for tbl in upload_v2_tables:
                if not inspector.has_table(tbl):
                    Base.metadata.tables[tbl].create(conn)
            # Re-inspect so the seed + ADD COLUMN block below sees the
            # freshly-created tables on a brand-new DB.
            inspector = inspect(engine)

            # 2) Seed the three PlanTier rows. Values are LOCKED by
            #    DECISIONS.md Decisions 3 (slot caps), 4 (monthly credit
            #    allotments), and 6 (Free is RTMP-only + 1 channel + 1
            #    publish/day). -1 = unlimited.
            #    Idempotent: insert one tier at a time, skip if name exists.
            if inspector.has_table("plan_tiers"):
                existing_tiers = {
                    row[0] for row in conn.execute(
                        text("SELECT name FROM plan_tiers")
                    ).fetchall()
                }
                # (name, monthly_credit_allotment, slot_cap_active_uploads,
                #  direct_path_allowed, max_channels, max_publishes_per_day)
                _seed_bool_true = "TRUE" if engine.dialect.name == "postgresql" else "1"
                _seed_bool_false = "FALSE" if engine.dialect.name == "postgresql" else "0"
                tier_seeds = [
                    ("free",       300,    5, _seed_bool_false,  1,  1),
                    ("pro",       2000,   20, _seed_bool_true,  -1, -1),
                    ("enterprise", 15000, 100, _seed_bool_true,  -1, -1),
                ]
                for (name, alloc, cap, dpa, mc, mppd) in tier_seeds:
                    if name not in existing_tiers:
                        conn.execute(text(
                            "INSERT INTO plan_tiers "
                            "(name, monthly_credit_allotment, slot_cap_active_uploads, "
                            " direct_path_allowed, max_channels, max_publishes_per_day) "
                            f"VALUES (:n, :a, :c, {dpa}, :mc, :mppd)"
                        ), {"n": name, "a": alloc, "c": cap, "mc": mc, "mppd": mppd})

            # 3) ADD COLUMN users.plan_tier_id — nullable INTEGER for now.
            #    Matches the existing reflection-guarded ALTER TABLE pattern.
            if inspector.has_table("users"):
                ucols = {c["name"] for c in inspector.get_columns("users")}
                if "plan_tier_id" not in ucols:
                    conn.execute(text(
                        "ALTER TABLE users ADD COLUMN plan_tier_id INTEGER"
                    ))

            # 4) Backfill users.plan_tier_id → Pro tier id for any row
            #    where it is still NULL. Locked by DECISIONS.md Decision 8:
            #    "preserves today's behaviour exactly — no existing user is
            #    silently slot-capped or paywalled mid-flight."
            #    Idempotent: rows already pointing at any tier are skipped.
            if (inspector.has_table("users")
                    and inspector.has_table("plan_tiers")):
                conn.execute(text(
                    "UPDATE users SET plan_tier_id = ("
                    "  SELECT id FROM plan_tiers WHERE name = 'pro' LIMIT 1"
                    ") WHERE plan_tier_id IS NULL"
                ))
        except Exception as e:  # pragma: no cover — log + continue
            # Match the existing convention: migration failures log to
            # stdout but do not crash startup. The dev backend stays up
            # so other routes keep working while the operator fixes the
            # migration. Production Postgres operators run docs/MIGRATIONS.md
            # by hand, so a stray exception here is dev-only noise.
            print(f"[startup] upload-rewrite v2 migration skipped: {e}")

        # ── Per-clip MasterVideo (double-post fix) ──────────────────────
        # A V4 job produces 1 Full Video + N shorts. The old schema had
        # ONE master per job (source_upload_id UNIQUE), so publishing the
        # bulletin AND a short collapsed both onto one master → the same
        # file was uploaded twice. Drop the per-job uniqueness, add a
        # clip_id, and backfill it from the legacy r2_key shape so
        # existing masters become per-clip.
        try:
            if inspector.has_table("master_videos"):
                _is_pg = engine.dialect.name == "postgresql"
                mcols = {c["name"] for c in inspector.get_columns("master_videos")}
                if "clip_id" not in mcols:
                    conn.execute(text(
                        "ALTER TABLE master_videos ADD COLUMN clip_id INTEGER"
                    ))
                if _is_pg:
                    # Drop the auto-named UNIQUE constraint on
                    # source_upload_id (idempotent).
                    conn.execute(text(
                        "ALTER TABLE master_videos "
                        "DROP CONSTRAINT IF EXISTS master_videos_source_upload_id_key"
                    ))
                    conn.execute(text(
                        "CREATE INDEX IF NOT EXISTS ix_master_videos_clip_id "
                        "ON master_videos (clip_id)"
                    ))
                    # Backfill clip_id from the legacy key shapes:
                    #   legacy/clip/{id}/master.mp4
                    #   raw_uploads/{user}/clip_{id}/master.mp4
                    conn.execute(text(
                        r"UPDATE master_videos SET clip_id = "
                        r"CAST(substring(r2_key from 'legacy/clip/(\d+)/') AS INTEGER) "
                        r"WHERE clip_id IS NULL "
                        r"AND r2_key ~ 'legacy/clip/\d+/'"
                    ))
                    conn.execute(text(
                        r"UPDATE master_videos SET clip_id = "
                        r"CAST(substring(r2_key from 'clip_(\d+)/') AS INTEGER) "
                        r"WHERE clip_id IS NULL "
                        r"AND r2_key ~ 'clip_\d+/'"
                    ))
        except Exception as e:  # pragma: no cover
            print(f"[startup] per-clip master migration skipped: {e}")

        # ── Durable queue (Wave 1) — upload_jobs_v2 claim/lease columns ──
        # Additive + idempotent. Harmless with KAIZER_DURABLE_QUEUE=0:
        # next_attempt_at defaults to now(), the legacy scheduler ignores
        # every new column. See plan "Kaizer → Enterprise Grade" Wave 1.1.
        try:
            if inspector.has_table("upload_jobs_v2"):
                _is_pg = engine.dialect.name == "postgresql"
                _ts = "TIMESTAMPTZ" if _is_pg else "TIMESTAMP"
                _now = "now()" if _is_pg else "CURRENT_TIMESTAMP"
                jcols = {c["name"] for c in inspector.get_columns("upload_jobs_v2")}
                if "next_attempt_at" not in jcols:
                    if _is_pg:
                        conn.execute(text(
                            f"ALTER TABLE upload_jobs_v2 ADD COLUMN "
                            f"next_attempt_at {_ts} NOT NULL DEFAULT {_now}"
                        ))
                    else:
                        # SQLite can't ADD COLUMN with a non-constant
                        # default — add nullable, backfill below.
                        conn.execute(text(
                            f"ALTER TABLE upload_jobs_v2 ADD COLUMN next_attempt_at {_ts}"
                        ))
                if "lease_expires_at" not in jcols:
                    conn.execute(text(
                        f"ALTER TABLE upload_jobs_v2 ADD COLUMN lease_expires_at {_ts}"
                    ))
                if "claimed_by" not in jcols:
                    conn.execute(text(
                        "ALTER TABLE upload_jobs_v2 ADD COLUMN claimed_by VARCHAR(64)"
                    ))
                if "user_id" not in jcols:
                    conn.execute(text(
                        "ALTER TABLE upload_jobs_v2 ADD COLUMN user_id INTEGER"
                    ))
                if "priority" not in jcols:
                    conn.execute(text(
                        "ALTER TABLE upload_jobs_v2 ADD COLUMN "
                        "priority VARCHAR(16) DEFAULT 'normal'"
                    ))
                # Postiz delivery (upload_path='postiz'): the target Postiz
                # integration id, copied from Channel at fanout. NULL for
                # direct/rtmp jobs.
                if "postiz_integration_id" not in jcols:
                    conn.execute(text(
                        "ALTER TABLE upload_jobs_v2 ADD COLUMN "
                        "postiz_integration_id VARCHAR(64)"
                    ))

                # Backfill denormalized user_id / priority from publish_tasks
                # (idempotent — only touches rows still NULL/empty).
                conn.execute(text(
                    "UPDATE upload_jobs_v2 SET user_id = ("
                    "  SELECT user_id FROM publish_tasks"
                    "  WHERE publish_tasks.id = upload_jobs_v2.publish_task_id"
                    ") WHERE user_id IS NULL"
                ))
                conn.execute(text(
                    "UPDATE upload_jobs_v2 SET priority = COALESCE(("
                    "  SELECT priority FROM publish_tasks"
                    "  WHERE publish_tasks.id = upload_jobs_v2.publish_task_id"
                    "), 'normal') WHERE priority IS NULL OR priority = ''"
                ))
                conn.execute(text(
                    f"UPDATE upload_jobs_v2 SET next_attempt_at = {_now} "
                    "WHERE next_attempt_at IS NULL"
                ))

                # Partial indexes — supported by both Postgres and SQLite.
                _active = "('claimed','branding','ready_to_upload','uploading')"
                for stmt in (
                    "CREATE INDEX IF NOT EXISTS ix_ujv2_claim "
                    "ON upload_jobs_v2 (next_attempt_at, created_at) "
                    "WHERE status = 'queued'",
                    f"CREATE INDEX IF NOT EXISTS ix_ujv2_user_active "
                    f"ON upload_jobs_v2 (user_id) WHERE status IN {_active}",
                    f"CREATE INDEX IF NOT EXISTS ix_ujv2_lease "
                    f"ON upload_jobs_v2 (lease_expires_at) WHERE status IN {_active}",
                    "CREATE INDEX IF NOT EXISTS ix_ujv2_parked "
                    "ON upload_jobs_v2 (created_at) WHERE status = 'parked_quota'",
                ):
                    conn.execute(text(stmt))
        except Exception as e:  # pragma: no cover — log + continue
            print(f"[startup] durable-queue migration skipped: {e}")

        conn.commit()


def _seed_defaults():
    """Populate new DB with default channel profiles + ensure legacy user exists,
    and backfill any pre-auth rows with NULL user_id onto that legacy user.
    Idempotent — safe to run on every startup.
    """
    from sqlalchemy import text as _text
    from sqlalchemy import inspect as _sa_inspect
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

        # Library categories — operator-managed but seeded on first run so
        # uploaders have something to pick from. Admins can rename/add/
        # delete via /api/library/categories. Idempotent — only inserts
        # if the table is empty.
        try:
            existing = db.query(models.LibraryCategory).count()
        except Exception:
            existing = 0
        if existing == 0:
            seeds = [
                ("News",          "#c0392b", 10),
                ("Sports",        "#27ae60", 20),
                ("Entertainment", "#9b59b6", 30),
                ("Politics",      "#34495e", 40),
                ("Business",      "#f39c12", 50),
                ("Technology",    "#2980b9", 60),
                ("Lifestyle",     "#e67e22", 70),
                ("Education",     "#16a085", 80),
                ("Music",         "#e84393", 90),
                ("Other",         "#7f8c8d", 99),
            ]
            for name, color, sort_order in seeds:
                db.add(models.LibraryCategory(
                    name=name, color=color, sort_order=sort_order,
                ))
            db.commit()
            print(f"[startup] Seeded {len(seeds)} default library categories")
        # ── Onboarding: exempt the accounts that predate this feature ─────
        # The first-sign-in details form is gated on "does this user have an
        # onboarding row", so giving a row to everyone who already existed is
        # what makes the form show to NEW sign-ups only.
        #
        # ONE SHOT, EVER -- and that is the whole point. Re-running it would
        # make "pre-existing" mean "exists right now", so a new account that
        # opened the form and closed the tab without submitting would be handed
        # a legacy row by the next restart and never asked again, leaving a
        # profile with no mobile, company, languages or channel link that looks
        # filled in. The gate would be defeated by waiting for a restart, and
        # this machine restarts routinely.
        #
        # The marker and the rows commit together: a crash cannot leave the
        # marker set with the rows missing.
        try:
            # Imported here, not at module scope: this file's top-level
            # imports run AFTER _seed_defaults() is called, so a module-level
            # name would not exist yet. The local sqlalchemy imports above do
            # the same thing for the same reason.
            import datetime as _dtm
            _MARK = "onboarding_backfill_done"
            _insp = _sa_inspect(engine)
            if (_insp.has_table("onboarding_profiles")
                    and _insp.has_table("system_settings")
                    and db.query(models.SystemSetting)
                         .filter(models.SystemSetting.key == _MARK).first() is None):
                # Skip anyone who already has a row. The marker makes this
                # one-shot, but a deployment can still carry rows from a
                # partially-completed earlier run -- inserting blindly would
                # raise UniqueViolation, roll back, never set the marker, and
                # retry forever on every boot.
                _have = {r[0] for r in db.query(models.OnboardingProfile.user_id).all()}
                _added = 0
                for _u in db.query(models.User).all():
                    if _u.id in _have:
                        continue
                    db.add(models.OnboardingProfile(
                        user_id=_u.id, source="legacy",
                        full_name=(_u.name or ""), email=(_u.email or ""),
                    ))
                    _added += 1
                db.add(models.SystemSetting(
                    key=_MARK,
                    value=_dtm.datetime.now(_dtm.timezone.utc).isoformat()))
                db.commit()
                print(f"[startup] onboarding: {_added} pre-existing account(s) marked "
                      f"legacy. Accounts created from now on must fill the form.")
        except Exception as _e:
            # Never let this stop the app booting. Worst case an existing
            # user sees the form once, which they can fill; a failed start
            # is not recoverable.
            db.rollback()
            print(f"[startup] onboarding backfill skipped: {_e}")
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
# KAIZER_MEDIA_ROOT: uploads/media land here when set (desktop_entry points
# it at <userData>/media — the frozen install dir may be read-only). Unset →
# byte-identical to the historical BASE_DIR/media.
MEDIA_ROOT  = Path((os.getenv("KAIZER_MEDIA_ROOT", "") or "").strip()
                   or (BASE_DIR / "media"))
OUTPUT_ROOT = Path(os.getenv("KAIZER_OUTPUT_ROOT", "/tmp/kaizer_output"))
MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(exist_ok=True)

app = FastAPI(title="Kaizer Pipeline API", version="2.0.0")
if _DESKTOP:
    # Desktop: the only legitimate caller is the local shell/SPA on
    # 127.0.0.1 (any port — Electron/dev pick ephemeral ones), so pin CORS
    # to that origin family instead of "*".
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^http://127\.0\.0\.1(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # DNS-rebinding belt: the desktop API is unauthenticated-by-transport
    # (it trusts "whoever can reach localhost"). A malicious website can
    # point its OWN domain's DNS at 127.0.0.1 and make the victim's browser
    # call this API cross-origin with a foreign Host header (CORS doesn't
    # stop simple requests). Reject anything whose Host isn't literally
    # this machine.
    from fastapi.responses import JSONResponse as _JSONResponse

    _ALLOWED_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}

    @app.middleware("http")
    async def _desktop_host_guard(request: Request, call_next):
        raw = (request.headers.get("host") or "").strip().lower()
        if raw.startswith("["):          # IPv6 literal, e.g. [::1]:8765
            hostname = raw.split("]", 1)[0].lstrip("[")
        else:
            hostname = raw.split(":", 1)[0]
        if hostname not in _ALLOWED_LOCAL_HOSTS:
            return _JSONResponse(
                status_code=403,
                content={"detail": "For your security, Kaizer X only "
                                   "answers requests made from this "
                                   "computer."},
            )
        return await call_next(request)
else:
    # KAIZER_CORS_ORIGINS: comma-separated explicit allowlist. Falls back to
    # the production web app only -- narrower than the old "*" so the API
    # only answers browser-origin requests from domains we control.
    _cors_env = os.getenv("KAIZER_CORS_ORIGINS", "").strip()
    _cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()] or [
        "https://kaizerx.com",
        "https://www.kaizerx.com",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Wave 2 (API scale): gzip JSON responses — the jobs list / status
# payloads shrink ~10x over the wire. minimum_size skips tiny payloads
# where the gzip header would be pure overhead. Media playback is
# unaffected in practice: browsers send Accept-Encoding: identity for
# <video> range requests, so /api/file/ byte-ranges stay uncompressed.
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1024)


# ── Sync-route threadpool capacity (Wave 6 load-test finding) ───────────────
# Most routes are sync `def` handlers, which FastAPI runs on anyio's
# default 40-thread limiter. At 1,000+ concurrent clients that queue —
# not the DB, not the handlers — was the p95 bottleneck. Raise it; the
# DB pool (KAIZER_DB_POOL_SIZE/_MAX_OVERFLOW) must be sized to match.
@app.on_event("startup")
async def _raise_threadpool_capacity():
    try:
        import anyio.to_thread
        # Default 48: load-tested sweet spot on the dev box — 100
        # threads triggered GIL convoying (p95 12s); 48 gave p95 141ms
        # at 200 concurrent pollers + 200 WebSockets.
        tokens = int(os.environ.get("KAIZER_THREADPOOL_TOKENS", "48"))
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = max(40, tokens)
        print(f"[startup] sync-route threadpool tokens = {limiter.total_tokens}")
    except Exception as e:
        print(f"[startup] WARN: could not raise threadpool tokens: {e}")

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
# SaaS-only surfaces — SKIPPED in desktop mode (_DESKTOP): publishing/OAuth/
# quota/scheduling/analytics/admin all assume the multi-tenant server (its
# publish workers, its YouTube app credentials, its billing). The desktop app
# is render+edit only; publishing happens through the user's own channels on
# the SaaS. Local render/edit surfaces (avatar, podcast, templates,
# v4_editor, jobs, assets...) stay mounted below.
if not _DESKTOP:
    app.include_router(youtube_oauth_router)
    app.include_router(youtube_upload_router)
    app.include_router(publish_tasks_router)    # Phase 1.B — new publish path (KAIZER_NEW_PUBLISH_PATH gated; legacy route above stays on)
    app.include_router(insights_router)         # Insights / Trend Finder — channel root-cause analyzer
    app.include_router(meta_oauth_router)
    app.include_router(linkedin_oauth_router)
    app.include_router(youtube_quota_router)
    # Billion-dollar phases A–E: campaigns, performance, translation, trending radar.
    app.include_router(campaigns_router)
    app.include_router(performance_router)
    app.include_router(trending_router)
    app.include_router(billing_router)
    app.include_router(live_director_router)  # Phase 6 — Autonomous Live Director
    app.include_router(admin_router)           # Phase 12 — admin panel REST surface
    app.include_router(work_monitor_router)     # Live work-monitor dashboard (Claude/agents progress)
    app.include_router(postiz_router)           # Cross-platform scheduling via Postiz (admin-only)
    app.include_router(yt_lookup_router)
    app.include_router(onboarding_router)        # one-time details form on a new account's first sign-in        # YouTube channel lookup for Style References (auth'd)
    app.include_router(help_router)        # Help Centre: guides streamed from R2, not bundled
    app.include_router(analytics_ai_router)     # AI-powered Insights — coach reports + any-channel compare (auth'd + rate-limited)
    app.include_router(express_mode_router)     # Express Mode — one-click auto-publish (Whisper+Claude+Postiz)
    app.include_router(heygen_router)           # HeyGen avatar generation for Trending (replaces Veo 3)
app.include_router(translation_router)
app.include_router(assets_router)
app.include_router(channel_groups_router)
app.include_router(job_progress_router)   # Phase 2B — job progress endpoint
app.include_router(feedback_router)       # Phase 3.5 — post-publish feedback endpoint
app.include_router(editor_router)         # Wave 2 — editor beta endpoints
app.include_router(bulletin_images_router)  # Per-image bulletin carousel mgmt (list/replace/recompose)
from routers.editing_features import router as editing_features_router  # noqa: E402
app.include_router(editing_features_router)  # Admin: full editing-engine catalog + dummy previews
from routers.editing_features import user_router as style_catalog_router  # noqa: E402
app.include_router(style_catalog_router)  # User: style picker (direct the AI Director per category)
from routers.style_packs import router as style_packs_router  # noqa: E402
app.include_router(style_packs_router)  # user-composed style packs + admin user-creations
app.include_router(avatar_router)           # News Anchor — AI presenter (avatar_studio | echomimic | heygen)
app.include_router(podcast_router)          # Podcast editor — single-cam cutdown + promos
app.include_router(desktop_router)          # Desktop app — device activation/licensing (3-device limit)
from routers.account_requests import router as account_requests_router  # noqa: E402
app.include_router(account_requests_router)  # Desktop account requests + admin approval + managed key bundle
from routers.admin_billing import router as admin_billing_router  # noqa: E402
app.include_router(admin_billing_router)     # Per-user Google key minting + usage/billing view
app.include_router(live_studio_router)      # Live Studio — bulk RTMP-live publishing (multi-video × multi-channel)
app.include_router(v4_editor_router)        # V4 — canvas editor (read/write canvas.json, re-render)
app.include_router(v4_defaults_router)      # V4 — user auto-pipeline defaults
app.include_router(library_router)          # Shared company Library — creative uploads + "Use" picker
app.include_router(profile_router)          # User profile — avatar + creator rating
app.include_router(metrics_router)          # Phase 3.G — Prometheus /metrics (NOT auth-gated; scraped over internal network)
app.include_router(admin_upload_v2_router)  # Phase 3.G — admin observability page at /admin/upload-v2 (admin_required)
# Wave 3 — WebSocket live progress under /api so the Vite dev proxy
# ("/api": { ws: true }) and the production VITE_API_URL origin both
# forward the upgrade. Routes: /api/ws/jobs/{id}, /api/ws/uploads.
app.include_router(ws_progress_router, prefix="/api")
app.include_router(quick_publish_router)    # Quick Publish — /api/clips/{id}/quick-*
app.include_router(custom_templates_router) # /api/templates — developer-uploaded HTML/CSS templates
if _DESKTOP:
    # Desktop-ONLY local control surface (API keys + per-feature preflight).
    # No auth — the backend binds 127.0.0.1 for a single local user; the
    # strict _DESKTOP gate is what keeps the SaaS from ever exposing it.
    from routers.desktop_local import router as desktop_local_router  # noqa: E402
    app.include_router(desktop_local_router)

# ── Static files: /media → BASE_DIR/output  ──────────────────────────────────
# Serves beta-rendered MP4s (and any other output files) to the frontend
# <video> player via the /media/<relative-path> URL scheme.
# Added for Wave 2 editor beta; safe to have even when the output dir is empty.
from fastapi.staticfiles import StaticFiles as _StaticFiles

# Honor KAIZER_OUTPUT_ROOT so /media serves from the SAME local storage root
# the renderer + storage provider write to (e.g. a dedicated D:/ test folder).
# Falls back to BASE_DIR/output when the env var is unset.
_output_dir = os.path.abspath(
    os.environ.get("KAIZER_OUTPUT_ROOT") or str(BASE_DIR / "output")
)
os.makedirs(_output_dir, exist_ok=True)
app.mount("/media", _StaticFiles(directory=_output_dir), name="media")

# ── Static files: /releases → desktop installer downloads (SaaS) ─────────────
# When KAIZER_RELEASES_DIR points at an existing folder, serve it so the
# site's /desktop page can offer the desktop installer for download.
# Deliberately NOT desktop-gated (env-gated instead): unset → exact
# historical behavior, no mount.
_releases_dir = (os.environ.get("KAIZER_RELEASES_DIR", "") or "").strip()
if _releases_dir and os.path.isdir(_releases_dir):
    app.mount("/releases", _StaticFiles(directory=_releases_dir),
              name="releases")
    print(f"[startup] /releases served from {_releases_dir}")

# Desktop SPA note: the desktop build ALSO serves the built frontend at "/",
# but that mount lives at the very BOTTOM of this file — a "/" mount matches
# every path by prefix, so registering it here would shadow all the @app.*
# routes defined further down. Search for "Desktop SPA" below.


# ── Upload worker lifecycle ──────────────────────────────────────────────────
@app.on_event("startup")
async def _start_upload_worker():
    if _DESKTOP:
        return  # desktop: no publish path → no upload worker
    try:
        await upload_worker.start()
        print("[startup] upload worker running")
    except Exception as e:
        print(f"[startup] WARN: upload worker failed to start: {e}")


@app.on_event("shutdown")
async def _stop_upload_worker():
    await upload_worker.stop()


# ── Publish/Upload Rewrite v2 — Scheduler (Phase 1.C) ────────────────────────
# In-process weighted-fair scheduler that dispatches UploadJobV2 rows the
# Fanout service (services/fanout.py) creates. Phase 1 is a skeleton: it
# transitions jobs through the new state machine with synthetic sleeps but
# does NOT call YouTube. Phase 2's Upload agent replaces _run_job's body.
# Lives behind the KAIZER_NEW_PUBLISH_PATH=0 flag at the router layer so
# this is harmless to leave running until cutover.
def _durable_queue_enabled() -> bool:
    """Wave 1 flag — KAIZER_DURABLE_QUEUE=1 swaps the in-memory
    scheduler heap + boot-only recovery + crons.py for the Postgres
    SKIP LOCKED worker + leader-elected cron runner. Read per call so
    a .env flip + restart is the full rollback."""
    return (os.environ.get("KAIZER_DURABLE_QUEUE", "0") or "0").strip() == "1"


@app.on_event("startup")
async def _start_publish_v2_scheduler():
    if _DESKTOP:
        return  # desktop: no publish path → no v2 scheduler / durable worker
    if _durable_queue_enabled():
        try:
            from services import publish_worker as _pw
            await _pw.start()
            print(f"[startup] durable publish worker running "
                  f"(worker_id={_pw.WORKER_ID})")
        except Exception as e:
            print(f"[startup] WARN: durable publish worker failed to start: {e}")
        return
    try:
        from services import scheduler as _publish_scheduler
        await _publish_scheduler.start()
        print("[startup] publish v2 scheduler running")
    except Exception as e:
        print(f"[startup] WARN: publish v2 scheduler failed to start: {e}")


@app.on_event("startup")
async def _start_pipeline_factory():
    """Boot the staged 'factory' station pools when
    KAIZER_PIPELINE_FACTORY_WORKERS=1. Fully additive: the existing publish
    path is untouched, and stage-event instrumentation (the admin Pipeline
    Flow view) works whether or not this is enabled."""
    if _DESKTOP:
        return  # desktop: no publish factory
    try:
        from services import pipeline_factory as _pf
        if _pf.workers_enabled():
            _pf.start_workers()
            print(f"[startup] pipeline factory workers running depths={_pf.depths()}")
    except Exception as e:
        print(f"[startup] WARN: pipeline factory failed to start: {e}")


@app.on_event("shutdown")
async def _stop_pipeline_factory():
    try:
        from services import pipeline_factory as _pf
        _pf.stop_workers()
    except Exception:
        pass


@app.on_event("shutdown")
async def _stop_publish_v2_scheduler():
    if _DESKTOP:
        return  # desktop: scheduler was never started
    if _durable_queue_enabled():
        try:
            from services import publish_worker as _pw
            await _pw.shutdown()
        except Exception as e:
            print(f"[shutdown] WARN: durable publish worker shutdown failed: {e}")
        return
    try:
        from services import scheduler as _publish_scheduler
        await _publish_scheduler.shutdown()
    except Exception as e:
        print(f"[shutdown] WARN: publish v2 scheduler shutdown failed: {e}")


# ── Publish/Upload Rewrite v2 — Phase 3 recovery + background crons ─────────
# On boot, sweep ``publish_attempts`` rows whose status is 'in_flight'
# AND older than 600 s (assume a previous worker died before us). The
# F-agent's ``services.idempotency.recover_orphans`` does the DB-level
# work (flipping rows to 'recovered' + parent UploadJobV2 back to
# 'queued'); we re-enqueue them on the new scheduler.
#
# Plus: three background asyncio crons (branding cleanup hourly, credit
# allotment daily, burn-log reconcile daily). The crons module's
# ``shutdown()`` waits up to 10s for them to drain.
#
# Both hooks are wrapped in try/except so a corrupted row or missing
# table CANNOT prevent FastAPI from coming up.
@app.on_event("startup")
async def _v2_recovery_and_crons():
    if _DESKTOP:
        return  # desktop: no publish path → no recovery sweep / crons
    if _durable_queue_enabled():
        # Durable mode: boot-time recovery + crons.py are superseded.
        # The leader-elected cron runner's lease reaper does recovery
        # CONTINUOUSLY at runtime (60s) and the same registry carries
        # branding cleanup / credit allotment / burn reconcile plus the
        # new un-parker, exhausted sweep, and counter reconciler.
        try:
            from services import cron_runner as _cr
            await _cr.start()
            print("[startup] durable cron runner racing for leadership "
                  "(reaper+unpark+exhausted+counters+branding+credits+burn)")
        except Exception as e:
            print(f"[startup] WARN: durable cron runner failed to start: {e}")
        return

    try:
        from services import recovery as _v2_recovery
        n = _v2_recovery.recover_on_startup()
        if n:
            print(f"[startup] v2 recovery re-queued {n} orphan(s)")
        else:
            print("[startup] v2 recovery: no orphans")
    except Exception as e:
        print(f"[startup] WARN: v2 recovery failed (non-fatal): {e}")

    try:
        from services import crons as _v2_crons
        await _v2_crons.start()
        print("[startup] v2 background crons running (branding+credits+burn)")
    except Exception as e:
        print(f"[startup] WARN: v2 crons failed to start: {e}")


@app.on_event("shutdown")
async def _stop_v2_crons():
    if _DESKTOP:
        return  # desktop: crons were never started
    if _durable_queue_enabled():
        try:
            from services import cron_runner as _cr
            await _cr.shutdown()
        except Exception as e:
            print(f"[shutdown] WARN: durable cron runner shutdown failed: {e}")
        return
    try:
        from services import crons as _v2_crons
        await _v2_crons.shutdown()
    except Exception as e:
        print(f"[shutdown] WARN: v2 crons shutdown failed: {e}")


# ── Channel learning cron (Phase 7) ──────────────────────────────────────────
@app.on_event("startup")
async def _start_corpus_scheduler():
    if _DESKTOP:
        return  # desktop: no channel-learning corpus crawler
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
    if _DESKTOP:
        return  # desktop: no RTMP publishing → nothing to reconcile on YT
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
    if _DESKTOP:
        return  # desktop: no Live Studio broadcasting → no R2 recovery
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
    if _DESKTOP:
        return  # desktop: no R2 preview backups to expire
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


@app.on_event("startup")
async def _live_studio_orphan_sweeper():
    """At backend startup, sweep LiveStream rows that were mid-broadcast
    when the previous process died (status in
    starting/provisioning/streaming/queued/downloading) — their ffmpeg
    subprocess is gone, so they need to be marked failed honestly rather
    than appearing stuck-on-streaming forever in the UI.

    Runs synchronously on boot before serving requests.
    """
    if _DESKTOP:
        return  # desktop: no Live Studio broadcasting → no live_streams rows
    try:
        from sqlalchemy import text as _text
        with engine.begin() as conn:
            res = conn.execute(_text(
                """
                UPDATE live_streams
                   SET status      = 'failed',
                       error       = COALESCE(error, '')
                                     || ' | backend restarted while broadcast was in flight',
                       message     = 'backend restarted; broadcast interrupted',
                       finished_at = CURRENT_TIMESTAMP
                 WHERE status IN ('starting','provisioning','streaming','queued','downloading','uploaded')
                """
            ))
            n = res.rowcount or 0
            if n:
                print(f"[startup] marked {n} orphaned live_streams as failed (backend was restarted mid-broadcast)")
    except Exception as e:
        print(f"[startup] WARN: live studio orphan sweeper failed: {e}")


@app.on_event("startup")
async def _reconcile_v4_render_orphans():
    """After a backend restart, V4 render subprocesses (spawned detached) can be
    orphaned and jobs left stuck in 'running' — and a re-run of a still-orphaned
    job spawns a DUPLICATE orchestrator that collides on the same output and
    hangs the job (the exact stuck-job bug). On boot: (1) kill any orphan
    ``pipeline_v4.orchestrator`` process whose --output-dir is under THIS
    backend's OUTPUT_ROOT (so a LIVE restart never touches DEV renders), then
    (2) mark any full_video_shorts_v4 Job still 'running' as failed so it is
    never stuck in the UI and can be cleanly re-run. Daemon thread so a slow
    process scan never blocks startup."""
    def _run():
        import time as _t
        _t.sleep(2)  # let the app settle before scanning
        killed = 0
        try:
            import psutil  # already a dependency (runner/admin use it)
            root = str(OUTPUT_ROOT).replace("\\", "/").lower()
            me = os.getpid()
            for p in psutil.process_iter(["pid", "cmdline"]):
                try:
                    if p.info["pid"] == me:
                        continue
                    cmd = " ".join(p.info.get("cmdline") or []).replace("\\", "/").lower()
                except Exception:
                    continue
                # Two spawn shapes: dev/SaaS = "python -m pipeline_v4.orchestrator
                # --job-id N ..."; frozen desktop = "<exe> render --job-id N ..."
                # (runner.build_v4_spawn_cmd). Match BOTH or orphan reaping
                # silently dies on desktop. The OUTPUT_ROOT check below still
                # applies to both (the frozen argv carries --output-dir too).
                _is_render = ("pipeline_v4.orchestrator" in cmd
                              or " render --job-id" in cmd)
                if _is_render and root and root in cmd:
                    try:
                        p.kill(); killed += 1
                    except Exception:
                        pass
        except Exception as e:
            print(f"[startup] v4 orphan-render kill skipped: {e}")
        try:
            from database import SessionLocal as _S
            import models as _m
            from datetime import datetime as _dt, timezone as _tz
            db = _S()
            try:
                stuck = db.query(_m.Job).filter(
                    _m.Job.platform == "full_video_shorts_v4",
                    _m.Job.status == "running",
                ).all()
                for j in stuck:
                    j.status = "failed"
                    try:
                        j.finished_at = _dt.now(_tz.utc)
                    except Exception:
                        pass
                    try:
                        j.log = (j.log or "") + "\n[recovered] backend restarted mid-render — marked failed; re-run to render cleanly."
                    except Exception:
                        pass
                if stuck:
                    db.commit()
                print(f"[startup] v4 orphan reconcile: killed {killed} orphan render(s), "
                      f"marked {len(stuck)} stuck job(s) failed")
            finally:
                db.close()
        except Exception as e:
            print(f"[startup] WARN: v4 orphan reconcile failed: {e}")
    import threading as _threading
    _threading.Thread(target=_run, name="kaizer-v4-orphan-reconcile", daemon=True).start()


@app.on_event("shutdown")
async def _stop_corpus_scheduler():
    corpus_scheduler.stop()

# ── Static config ────────────────────────────────────────────────────────────

PLATFORMS = {
    "instagram_reel":          {"label": "Instagram Reel", "width": 1080, "height": 1920},
    "youtube_short":           {"label": "YouTube Short",  "width": 1080, "height": 1920},
    # Facebook Reel — vertical short, same 9:16 render as IG Reel / YT Short.
    # The frontend remaps it to a V4 shorts-only job; the per-platform SEO
    # gives it a Facebook-native caption at publish time.
    "facebook_reel":           {"label": "Facebook Reel",  "width": 1080, "height": 1920},
    "youtube_full":            {"label": "YouTube Full",   "width": 1920, "height": 1080},
    # NOTE: the old compound "youtube_full_plus_shorts" tile was RETIRED
    # 2026-06-17 — it was redundant with the V4 "Full Video + Shorts" tile
    # below (V4 produces both bulletin + shorts in one job via output_format
    # "both"). runner/create_job normalise any stale submission onto V4.
    # Also retired: legacy render platforms "full_video_shorts_v2"/"v3".
    # Do not re-add any of them — V4 is the single render path.
    # ── V4 platform — trim + canvas architecture ────────────────────
    # V4 (2026-06-03): two atomic passes — Step 1 builds a clean
    # trimmed.mp4 (Deepgram + Claude KEEP/CUT + single filter_complex
    # trim+concat with inline audio normalization), Step 2 overlays
    # the canvas (text panels + timed images + brand chrome) onto
    # that trimmed video with audio passthrough (-c:a copy). Lipsync
    # cannot drift because Step 2 never touches the audio. The canvas
    # JSON is the source of truth — editor reads/writes it; image
    # swaps, duration changes, and reordering only re-run Step 2
    # (~5-15 s). Same engine produces bulletin (16:9) and shorts (9:16).
    "full_video_shorts_v4": {
        "label":  "Full Video + Shorts",
        "width":  1080,
        "height": 1920,
    },
}


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

# Desktop: NO "/" banner — routes registered here beat the bottom-of-file
# SPA mount (Starlette matches in registration order), so this JSON banner
# would shadow the desktop frontend's index.html. The SaaS keeps it.
if not _DESKTOP:
    @app.get("/")
    def root():
        return {"app": "Kaizer Pipeline API", "docs": "/docs", "health": "/api/health/"}

@app.get("/api/health/")
def health():
    return {"status": "ok"}

# ── Config ───────────────────────────────────────────────────────────────────

@app.get("/api/platforms/")
def get_platforms():
    return PLATFORMS


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

def _published_videos_for_jobs(db, job_ids):
    """Map job_id -> [{channel, video_id, watch_url, status, privacy_status}] for every YouTube
    upload that produced a video id. Chain: Job (=MasterVideo.source_upload_id) -> MasterVideo ->
    PublishTask -> UploadJobV2.youtube_video_id. ONE batched query for the whole list (no N+1).
    Empty for jobs never published (running/failed/raw upload not yet posted). Used to surface the
    watch link on Quick Publish cards + the job detail page."""
    out: dict[int, list] = {}
    if not job_ids:
        return out
    try:
        rows = (
            db.query(
                models.MasterVideo.source_upload_id,
                models.UploadJobV2.youtube_video_id,
                models.UploadJobV2.status,
                models.UploadJobV2.privacy_status,
                models.Channel.name,
            )
            .join(models.PublishTask, models.PublishTask.master_video_id == models.MasterVideo.id)
            .join(models.UploadJobV2, models.UploadJobV2.publish_task_id == models.PublishTask.id)
            .outerjoin(models.Channel, models.Channel.id == models.UploadJobV2.channel_id)
            .filter(models.MasterVideo.source_upload_id.in_(job_ids),
                    models.UploadJobV2.youtube_video_id.isnot(None))
            .all()
        )
    except Exception as exc:
        print(f"[jobs] published-videos lookup failed: {exc}", flush=True)
        return out
    seen = set()
    for jid, vid, st, priv, chname in rows:
        if not vid:
            continue
        # Skip junk ids (a few legacy rows stored a UUID) — real YouTube ids are 11 url-safe chars.
        if not (len(vid) == 11 and all(c.isalnum() or c in "-_" for c in vid)):
            continue
        key = (jid, vid, chname or "")
        if key in seen:
            continue
        seen.add(key)
        out.setdefault(jid, []).append({
            "channel": chname or "",
            "video_id": vid,
            "watch_url": f"https://youtu.be/{vid}",
            "status": st or "",
            "privacy_status": priv or "",
        })
    return out


@app.get("/api/jobs/")
def list_jobs(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
    # Wave 2 (API scale): pagination. The response stays a bare array so
    # existing frontends that .map() over it keep working; the default
    # limit=50 caps the damage from a 1000-job tenant until the UI pages.
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    # selectinload kills the per-job lazy-load on j.clips (1 + N queries
    # collapse to 2 queries regardless of job count).
    jobs = (
        db.query(models.Job)
          .options(selectinload(models.Job.clips))
          .filter(models.Job.user_id == user.id)
          .order_by(models.Job.created_at.desc())
          .offset(offset)
          .limit(limit)
          .all()
    )

    # Use the SAME root the /media mount + renderer use (KAIZER_OUTPUT_ROOT),
    # not a hardcoded BASE_DIR/output — otherwise a render written to a custom
    # output root (e.g. a dedicated D:/ folder) fails relative_to() here and
    # the UI wrongly reports "not rendered yet".
    output_root = OUTPUT_ROOT.resolve()

    def _resolve_thumb(j) -> tuple[Optional[str], Optional[str]]:
        """Return (url, aspect) for the best available cover image.

        Lookup order:
          1. V4 editor's bulletin AI thumb       (16:9)
          2. V4 first short AI thumb             (9:16)
          3. First Clip.thumb_path that exists   (aspect from platform)
        Returns (None, None) when nothing's available.
        """
        if not j.output_dir:
            return None, None
        out = Path(j.output_dir)
        if not out.is_absolute():
            out = (BASE_DIR / out).resolve()

        # 1) V4 bulletin AI thumb (16:9)
        bulletin = out / "bulletin_thumb_ai.jpg"
        if bulletin.is_file():
            try:
                rel = bulletin.resolve().relative_to(output_root)
                return f"/media/{rel.as_posix()}?t={int(bulletin.stat().st_mtime)}", "16:9"
            except ValueError:
                pass

        # 2) First short AI thumb (9:16)
        short1 = out / "short_01_thumb_ai.jpg"
        if short1.is_file():
            try:
                rel = short1.resolve().relative_to(output_root)
                return f"/media/{rel.as_posix()}?t={int(short1.stat().st_mtime)}", "9:16"
            except ValueError:
                pass

        # 3) Per-clip thumb_path fallback. Walk only the first few clips
        # to keep the per-job stat work bounded.
        plat = (j.platform or "").lower()
        aspect = "9:16" if ("short" in plat or "reel" in plat) else "16:9"
        clips = sorted(j.clips, key=lambda c: c.clip_index or 0)[:3]
        for clip in clips:
            tp = (clip.thumb_path or "").strip()
            if not tp:
                continue
            p = Path(tp) if Path(tp).is_absolute() else (BASE_DIR / tp).resolve()
            if p.is_file():
                try:
                    rel = p.resolve().relative_to(output_root)
                    return f"/media/{rel.as_posix()}?t={int(p.stat().st_mtime)}", aspect
                except ValueError:
                    continue
        return None, None

    # One batched lookup of YouTube watch links — ONLY for Quick Publish (raw_upload) jobs, the
    # only cards that surface them; keeps the list payload lean for pipeline jobs (which can fan
    # out to 30+ channels).
    pub_by_job = _published_videos_for_jobs(
        db, [j.id for j in jobs if (j.frame_layout or "") == "raw_upload"])

    out: list[dict] = []
    backfilled = False
    for j in jobs:
        # Wave 2 (API scale): prefer the cached columns; only rows that
        # haven't resolved yet pay for the stat() walk, and the result
        # is written back (lazy backfill — converges after one listing).
        thumb_url, thumb_aspect = j.thumb_url, j.thumb_aspect
        if not thumb_url:
            thumb_url, thumb_aspect = _resolve_thumb(j)
            if thumb_url:
                j.thumb_url, j.thumb_aspect = thumb_url, thumb_aspect
                backfilled = True
        out.append({
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
            # Item 114: surface the Stage 2 provider choice so the
            # UI can show a chip on the job card.
            "stage_2_provider": (j.stage_2_provider or "gemini"),
            # Grid-redesign: cover image for the jobs list. NULL when no
            # thumb yet (pre-V4-editor, running jobs, very old rows).
            "thumbnail_url":    thumb_url,
            "thumbnail_aspect": thumb_aspect,
            # Published YouTube videos for this job (Quick Publish cards link straight to them).
            "published_videos": pub_by_job.get(j.id, []),
        })
    if backfilled:
        # One commit for every row backfilled above. Failure is non-fatal
        # — the next listing simply re-resolves from disk.
        try:
            db.commit()
        except Exception:
            db.rollback()
    return out


@app.post("/api/jobs/create/")
async def create_job(
    video: Optional[UploadFile] = File(None),
    # Shared Library picker: when set, the source video is fetched from
    # R2 instead of being uploaded as a body. Mutually exclusive with
    # ``video`` — exactly one MUST be present. The rest of the form is
    # unchanged so the wizard's downstream steps look identical.
    library_item_id: Optional[int] = Form(None),
    platform: str = Form(...),
    # Optional: a full-form-only job (kind-first wizard) has NO shorts frame, so this is
    # legitimately empty. Defaulting to "" (instead of required) avoids a 422 when the
    # client omits/empties it; the orchestrator falls back to torn_card if shorts render.
    frame_layout: str = Form(""),
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
    # Item 114 (Stage 2 provider catalog): operator's chosen LLM for
    # the editorial-decision stage. One of {"gemini", "claude"}. Blank
    # or unknown -> "gemini" (the default). Ignored by V1 platforms.
    stage_2_provider: str = Form("gemini"),
    # V4 only: studio-background video the user picked in the NewJob
    # wizard. Stamped onto canvas.layout.bg_video_path before the very
    # first render so the bulletin opens with the user-chosen backdrop
    # instead of flat black. Blank = legacy flat-colour behaviour.
    v4_bg_video_path:   str = Form(""),
    v4_bg_video_volume: float = Form(0.0),
    v4_bg_intro_seconds: float = Form(0.0),
    # V4 only: which model decides Step 1's KEEP/CUT plan.
    # "claude" (default — Opus 4.7) or "gemini" (2.5 Flash on Vertex).
    # Operator picks per-job in the wizard so we can A/B quality.
    # Persisted on Job.platform_meta JSON for later display.
    # Ignored by V1/V2 platforms.
    v4_trim_planner:    str = Form("claude"),
    # V4 only: content type / edit profile. "auto" (default) = the
    # system classifies the transcript and picks the right editing
    # persona (news/podcast/interview/vlog/generic); an explicit value
    # is the operator answering the "what type of video is this?"
    # question up front. Runner forwards as KAIZER_V4_CONTENT_TYPE.
    v4_content_type:    str = Form("auto"),
    # V4 only: which provider generates per-story images.
    # "auto" (multi-source V1 chain), "gemini" (Nano Banana), or
    # "openai" (gpt-image-1). Persisted on Job.v4_image_provider.
    v4_image_provider:  str = Form("auto"),
    # V4 only: operator-supplied description text. When non-empty,
    # the orchestrator preserves the source video AS-IS (no Claude
    # KEEP/CUT trim), still carves shorts, and uses this text as
    # the bulletin SEO description verbatim. Any language.
    v4_predefined_description: str = Form(""),
    # V4 only: which outputs to render. "both" (default — full video +
    # shorts), "full-only" (skip shorts), or "shorts-only" (skip the
    # bulletin/full video). Validated below; runner forwards as
    # KAIZER_V4_OUTPUT_FORMAT and the orchestrator gates rendering on it.
    v4_output_format:   str = Form("both"),
    # V4 only: full-form EFFECTS on the bulletin stories — "rich" (AI Director:
    # per-story mood/color/fx/transitions/overlays + sound design, the DEFAULT
    # for new V4 jobs), "auto" (broadcast-polish grade only) or "off". Runner
    # forwards as KAIZER_V4_EFFECTS_MODE and turns "rich" into KAIZER_V4_DIRECTOR=1;
    # the per-story cache hashes the resolved chain (legacy NULL rows stay byte-identical).
    v4_effects_mode:    str = Form("rich"),
    v4_theme:           str = Form(""),
    # V4 only: the user's per-category effect picks ("edit using THESE") — a
    # JSON object {style_packs,transitions,frame_fx,overlays,typography,
    # story_category:[...]}. Empty (the default) leaves every category to the
    # AI Director; any picked category constrains the Director to those ids
    # (and forces it on). Persisted on the Job + forwarded via the runner as
    # KAIZER_V4_STYLE_DIRECTIVES.
    v4_style_directives: str = Form(""),
    # V4 only: which AI Director ENGINE plans per-story direction — "v4"
    # (ours, full arsenal, DEFAULT) | "platform" (ported kaizer-platform
    # 3-layer engine: 5-mood signal formula + LLM confirm, adapted onto the
    # V4 vocabulary). Selects WHICH director, never WHETHER (effects mode).
    v4_director_engine: str = Form("v4"),
    # V4 Stage 2: defer the up-front MP4 render (edit-first; export on demand). Default off so
    # existing/auto-publish behaviour is unchanged. Persisted on the Job + forwarded to the runner.
    v4_defer_render:    bool = Form(False),
    # V4 only: max shorts per job. Default 8; the operator can opt into more
    # at job start. Runner forwards as KAIZER_V4_MAX_SHORTS; the orchestrator
    # caps the candidate list (a CEILING — content still decides the real count).
    v4_max_shorts:      int = Form(8),
    # V4 audio-first mode: the narration AUDIO is the master track. When true,
    # `audio` (below) is the required narration and `video` above becomes the
    # OPTIONAL muted reference b-roll (not the source). Images generated/uploaded
    # as usual. Persisted on Job.v4_audio_first; forwarded to the orchestrator.
    v4_audio_first:     bool = Form(False),
    # The narration audio upload (required when v4_audio_first).
    audio: Optional[UploadFile] = File(None),
    # V4 only: original publish target the operator picked ("instagram" /
    # "youtube" / "facebook"). Editor metadata only — not used for rendering.
    v4_target_platform: str = Form("youtube"),
    # V4 only: channels chosen AT GENERATE TIME (comma-separated Channel ids)
    # from the New Job "Choose channels" step. Persisted on the Job so the
    # editor + Publish flow know the intended targets. Empty = not chosen here
    # (channels picked later at publish — the legacy flow).
    channel_ids: str = Form(""),
    # PER-CHANNEL intro overrides — a JSON map {"<channel_id>": <asset_id>}.
    # A channel present uses that intro for this job; absent = its own assigned
    # intro. Set by the New Job inline per-channel intro picker.
    intro_overrides: str = Form(""),
    # Custom-template per-slot media: JSON {"<slot_key>": <asset_id>} for the slots
    # the user filled in the wizard's Media step; main_media_slot names the slot whose
    # video is the AI-trim "main". Only meaningful when frame_layout is "custom:<id>".
    template_media: str = Form(""),
    main_media_slot: str = Form(""),
    # "Full form video" (16:9) custom template, e.g. "custom:<id>".
    fullform_layout: str = Form(""),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
    # Wave 2 (API scale): plan-aware token bucket — each create kicks off
    # Gemini calls + FFmpeg renders, so this is the endpoint to cap.
    _rl=Depends(rate_limited("create")),
):
    import languages as _langs
    lang_cfg = _langs.get(language)  # falls back to default if invalid

    # Legacy render pipelines v2 (Inngest) and v3 retired 2026-06-17 — the
    # V4 trim+canvas engine is the single render path now. Normalise any
    # stale submission (cached frontend, in-flight share link) onto V4 so it
    # renders cleanly instead of 400-ing or hitting now-deleted dispatch code.
    if platform in ("full_video_shorts_v2", "full_video_shorts_v3"):
        platform = "full_video_shorts_v4"

    upload_dir = MEDIA_ROOT / "uploads"
    upload_dir.mkdir(exist_ok=True)

    # Audio-first: the narration AUDIO is the master/source; the optional
    # `video` upload becomes a MUTED reference b-roll (forwarded separately),
    # not the pipeline source. Stays None for normal video-first jobs.
    _ref_video_path: Optional[str] = None

    if v4_audio_first:
        # ── AUDIO-FIRST: narration audio is the source (video_path) ──
        if audio is None or not getattr(audio, "filename", ""):
            raise HTTPException(
                status_code=400,
                detail="Audio-first job requires an 'audio' narration upload.",
            )
        video_path = upload_dir / audio.filename
        with open(video_path, "wb") as f:
            while True:
                chunk = await audio.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        _src_filename     = audio.filename
        _src_content_type = audio.content_type or "audio/mpeg"
        # Optional muted reference b-roll (the `video` field, if supplied).
        if video is not None and getattr(video, "filename", ""):
            _rv_path = upload_dir / video.filename
            with open(_rv_path, "wb") as f:
                while True:
                    chunk = await video.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            _ref_video_path = str(_rv_path)
    else:
        # Exactly one of (video upload, library_item_id) must be present.
        if (video is None or not getattr(video, "filename", "")) and not library_item_id:
            raise HTTPException(
                status_code=400,
                detail="Either 'video' (upload) or 'library_item_id' must be provided.",
            )
        if video is not None and getattr(video, "filename", "") and library_item_id:
            raise HTTPException(
                status_code=400,
                detail="Pass 'video' OR 'library_item_id' — not both.",
            )

        # Resolve the source video to a local path + name + mime regardless
        # of which branch we came through; everything downstream reads these.
        if library_item_id:
            from routers.library import fetch_library_video_to as _fetch_lib
            video_path, _lib_item = _fetch_lib(library_item_id, upload_dir, db)
            _src_filename     = _lib_item.original_name or video_path.name
            _src_content_type = "video/mp4"
        else:
            video_path = upload_dir / video.filename
            # Wave 2 (API scale): stream the upload to disk in 1 MiB chunks
            # instead of buffering the whole file in RAM (a 2 GB upload used
            # to hold 2 GB resident per concurrent request).
            with open(video_path, "wb") as f:
                while True:
                    chunk = await video.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            _src_filename     = video.filename
            _src_content_type = video.content_type or "video/mp4"

    # Mirror the source video to R2 right after the local write. The
    # pipeline subprocess still reads from the local copy (faster than
    # a download round-trip), but R2 is the source of truth — if the
    # container restarts mid-render, retry can pull the source back.
    # Predictable key so retry/recovery doesn't need a DB lookup.
    # Library-sourced jobs skip this — the source already lives in R2
    # under library/<id>/video.<ext>.
    _src_timestamp = time.strftime("%Y%m%d_%H%M%S")
    if not library_item_id:
        try:
            from pipeline_core.storage import get_storage_provider
            # Honours STORAGE_BACKEND — local mode mirrors to ``output/`` for
            # parity with prod's R2 mirror. Failure is non-fatal: the user's
            # uploaded file is still on disk; this branch only guards
            # post-restart recovery on a horizontal-scale prod deploy.
            get_storage_provider().upload(
                str(video_path),
                f"sources/{user.id}/{_src_timestamp}_{_src_filename}",
                content_type=_src_content_type,
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
        # Audio-first: the upload is narration AUDIO, not a video — skip the
        # video-codec/resolution validator (Deepgram fails loud later if bad).
        _val_result = None if v4_audio_first else _validate_input(str(video_path))
        if _val_result is not None and not _val_result.ok:
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
        _validation_warnings = _val_result.warnings if _val_result is not None else []
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
    _name_clean = resolve_job_name(name, _src_filename)

    # Single Job row regardless of platform. The compound platform
    # ("youtube_full_plus_shorts") stores its original key here so the
    # runner can detect it and run TWO pipeline passes internally —
    # one bulletin pass + one shorts pass — both importing clips into
    # THIS same job. UI shows it as one job with mixed-aspect clips.
    # transition_style + stage_2_provider were knobs for the retired v2/v3
    # render pipelines (2026-06-17). The fields are still ACCEPTED so a stale
    # frontend never 400s, but they're vestigial — V4 ignores both. Coerce
    # blank/unknown to the historical defaults; no pipeline_v2 catalog lookup.
    _ts = (transition_style or "").strip() or "smart_cut"
    _s2p = (stage_2_provider or "").strip() or "gemini"

    # V4 only: validate the planner choice before persisting so a
    # stale frontend can never set garbage. Non-V4 platforms get
    # NULL to keep the column honest about "this row didn't use V4".
    _trim_planner_clean = (v4_trim_planner or "claude").strip().lower()
    if _trim_planner_clean not in {"claude", "gemini"}:
        _trim_planner_clean = "claude"
    _trim_planner_for_row = _trim_planner_clean if platform == "full_video_shorts_v4" else None

    # Same shape for the image provider pick.
    _image_provider_clean = (v4_image_provider or "auto").strip().lower()
    if _image_provider_clean not in {"auto", "gemini", "openai"}:
        _image_provider_clean = "auto"
    _image_provider_for_row = _image_provider_clean if platform == "full_video_shorts_v4" else None

    # Predefined description — V4-only, NULL stored elsewhere.
    # 8 kB ceiling so a runaway paste can't blow up the DB row.
    _predef_desc_clean = (v4_predefined_description or "").strip()
    if len(_predef_desc_clean) > 8000:
        _predef_desc_clean = _predef_desc_clean[:8000]
    # Audio-first is mutually exclusive with source-preserved (predefined
    # description) — the orchestrator prefers audio-first, so never carry a
    # predef under an audio-first job (it would be silently ignored anyway).
    _predef_desc_for_row = _predef_desc_clean if (platform == "full_video_shorts_v4" and _predef_desc_clean and not v4_audio_first) else None

    # Output-format pick — V4 only. Same validate-before-persist shape.
    _output_format_clean = (v4_output_format or "both").strip().lower()
    if _output_format_clean not in {"both", "full-only", "shorts-only", "trailer-only"}:
        _output_format_clean = "both"
    _output_format_for_row = _output_format_clean if platform == "full_video_shorts_v4" else None

    # Effects-mode pick — V4 only. Same validate-before-persist shape.
    _effects_mode_clean = (v4_effects_mode or "rich").strip().lower()
    if _effects_mode_clean not in {"auto", "rich", "off"}:
        _effects_mode_clean = "rich"
    _effects_mode_for_row = _effects_mode_clean if platform == "full_video_shorts_v4" else None
    # Director ENGINE pick — V4 only. Garbage coerces to "v4" (our engine).
    _director_engine_clean = (v4_director_engine or "v4").strip().lower()
    if _director_engine_clean not in {"v4", "platform"}:
        _director_engine_clean = "v4"
    _director_engine_for_row = _director_engine_clean if platform == "full_video_shorts_v4" else None
    # THEME PACK — validated against the real registry; unknown → classic.
    _theme_clean = (v4_theme or "").strip().lower()
    if _theme_clean:
        try:
            from pipeline_v4.theme_packs import get_theme as _gt_theme
            if _gt_theme(_theme_clean) is None:
                _theme_clean = ""
        except Exception:
            _theme_clean = ""
    _theme_for_row = _theme_clean if platform == "full_video_shorts_v4" else None

    # User-directed effect picks ("edit using THESE") — V4 only. Parse the JSON,
    # keep only recognized catalog categories, coerce each to a de-duped list of
    # non-empty string ids (story_category kept as a single string). Empty /
    # invalid → None (full AI-Director autonomy). Re-serialized canonically so a
    # stale/garbage payload never reaches the DB or the render env.
    _style_directives_for_row = None
    if platform == "full_video_shorts_v4":
        _sd_raw = (v4_style_directives or "").strip()
        if _sd_raw:
            try:
                _sd_in = json.loads(_sd_raw)
            except Exception:
                _sd_in = None
            if isinstance(_sd_in, dict):
                _sd_out = {}
                _list_keys = ("style_packs", "transitions", "frame_fx",
                              "overlays", "typography", "color_grades",
                              "sound", "layouts", "pip")
                for _k in _list_keys:
                    _v = _sd_in.get(_k)
                    if isinstance(_v, str):
                        _v = [_v]
                    if isinstance(_v, (list, tuple)):
                        _clean = []
                        for _x in _v:
                            _xs = str(_x).strip()[:80]
                            if _xs and _xs not in _clean:
                                _clean.append(_xs)
                        if _clean:
                            _sd_out[_k] = _clean[:64]
                _cat = _sd_in.get("story_category") or _sd_in.get("category")
                if isinstance(_cat, (list, tuple)):
                    _cat = _cat[0] if _cat else None
                if _cat and str(_cat).strip():
                    _sd_out["story_category"] = str(_cat).strip()[:60]
                if _sd_out:
                    _style_directives_for_row = json.dumps(_sd_out)

    # Target-platform pick — V4 only. Editor leads with this platform's SEO.
    _target_platform_clean = (v4_target_platform or "youtube").strip().lower()
    if _target_platform_clean not in {"youtube", "instagram", "facebook"}:
        _target_platform_clean = "youtube"
    _target_platform_for_row = _target_platform_clean if platform == "full_video_shorts_v4" else None

    # Channels chosen at generate time (New Job "Choose channels" step).
    # Parse the comma-separated ids → a clean de-duped JSON list, or None.
    _chan_ids_for_row = None
    try:
        _ids = [int(x) for x in (channel_ids or "").split(",") if x.strip().isdigit()]
        _seen: set = set()
        _ids = [i for i in _ids if not (i in _seen or _seen.add(i))]
        if _ids and platform == "full_video_shorts_v4":
            import json as _json
            _chan_ids_for_row = _json.dumps(_ids)
    except Exception:
        _chan_ids_for_row = None

    # PER-CHANNEL intro overrides — parse the {channel_id: asset_id} map, keep
    # only entries where BOTH the asset and the channel belong to this user
    # (defends against spoofed ids). None = no overrides (each channel's own).
    _intro_overrides_for_row = None
    try:
        if intro_overrides and platform == "full_video_shorts_v4":
            import json as _json
            _raw = _json.loads(intro_overrides)
            if isinstance(_raw, dict) and _raw:
                _want_assets = {int(v) for v in _raw.values() if str(v).strip().lstrip("-").isdigit() and int(v) > 0}
                _own_assets = {
                    a.id for a in db.query(models.UserAsset.id).filter(
                        models.UserAsset.id.in_(_want_assets or {0}),
                        models.UserAsset.user_id == user.id,
                    ).all()
                } if _want_assets else set()
                _own_channels = {
                    c.id for c in db.query(models.Channel.id).filter(
                        models.Channel.user_id == user.id,
                    ).all()
                }
                _clean = {}
                for k, v in _raw.items():
                    try:
                        cid, aid = int(k), int(v)
                    except (TypeError, ValueError):
                        continue
                    if cid in _own_channels and aid in _own_assets:
                        _clean[str(cid)] = aid
                if _clean:
                    _intro_overrides_for_row = _json.dumps(_clean)
    except Exception:
        _intro_overrides_for_row = None

    # Custom-template per-slot media (only for custom:<id> templates). JSON
    # {slot_key: asset_id}; main_media_slot names the slot that gets AI-trimmed.
    _template_media_for_row = {}
    _main_media_slot_for_row = ""
    try:
        if template_media and (str(frame_layout).startswith("custom:")
                               or str(fullform_layout).startswith("custom:")):
            _tm = _json.loads(template_media)
            _raw = _tm if isinstance(_tm, dict) else {}
            # A slot value is EITHER a scalar asset id (single image/video) OR a CAROUSEL struct
            # {carousel:[{id,duration_s,effect,effect_duration}], fit} — a story-driven slideshow.
            # Gather EVERY referenced id (scalars + carousel frames) for ONE ownership query, then
            # keep only owned assets (cross-user IDOR guard). Mirrors v4_editor.save_custom_template.
            _ref_ids = set()
            for _v in _raw.values():
                if isinstance(_v, dict) and _v.get("carousel"):
                    for _fr in (_v.get("carousel") or []):
                        try:
                            _ref_ids.add(int(_fr.get("id")))
                        except Exception:
                            pass
                else:
                    try:
                        _ref_ids.add(int(_v))
                    except Exception:
                        pass
            _own_ids = set()
            if _ref_ids:
                _own_ids = {r[0] for r in db.query(models.UserAsset.id).filter(
                    models.UserAsset.id.in_(_ref_ids),
                    models.UserAsset.user_id == user.id).all()}
            for _k, _v in _raw.items():
                if isinstance(_v, dict) and _v.get("carousel"):
                    _frames = []
                    for _fr in (_v.get("carousel") or []):
                        try:
                            _fid = int(_fr.get("id"))
                        except Exception:
                            continue
                        if _fid not in _own_ids:
                            continue
                        _frames.append({
                            "id": _fid,
                            "duration_s": max(0.5, min(float(_fr.get("duration_s") or 3.0), 30.0)),
                            "effect": str(_fr.get("effect") or "fade")[:16],
                            "effect_duration": max(0.0, min(float(_fr.get("effect_duration") or 0.4), 2.0)),
                        })
                    # Cost/DoS guard: a forged payload with thousands of frames would explode the
                    # ffmpeg overlay chain + GCP spend at render. 50 is far above any real slideshow.
                    _frames = _frames[:50]
                    if _frames:
                        _template_media_for_row[str(_k)] = {"carousel": _frames,
                                                            "fit": str(_v.get("fit") or "cover")[:10]}
                else:
                    try:
                        _iv = int(_v)
                    except Exception:
                        continue
                    if _iv in _own_ids:
                        _template_media_for_row[str(_k)] = _iv
            _main_media_slot_for_row = (main_media_slot or "").strip()[:64]
    except Exception:
        _template_media_for_row = {}

    # SECURITY (IDOR): a custom:<id> template must be the user's OWN or PUBLIC — never
    # render someone else's PRIVATE template by guessing its id.
    def _custom_template(layout_val):
        """Return the CustomTemplate row for a 'custom:<id>' value if it's available to
        this user, or None (also None for non-custom/empty values)."""
        s = str(layout_val or "")
        if not s.startswith("custom:"):
            return None
        try:
            _tid = int(s.split(":", 1)[1])
        except Exception:
            return False  # malformed -> treat as not-ok
        _t = db.get(models.CustomTemplate, _tid)
        if _t and _t.status != "disabled" and (_t.owner_id == user.id or _t.visibility == "public"):
            return _t
        return False

    def _custom_template_ok(layout_val) -> bool:
        return _custom_template(layout_val) is not False

    if not _custom_template_ok(frame_layout) or (fullform_layout and not _custom_template_ok(fullform_layout)):
        raise HTTPException(status_code=403, detail="Selected template is not available to you.")

    # CRASH-GUARD: a custom template's output form is decided by its own canvas aspect
    # (services.custom_templates.contract.aspect_kind). A full-form (16:9) template must
    # never be rendered as a short (9:16) or vice-versa — that produces a wrong-aspect
    # master that breaks the pipeline downstream. frame_layout is the SHORTS slot;
    # fullform_layout is the FULL-FORM slot. Reject a mismatch up front with a clear
    # message instead of letting it crash mid-render. (Frontend already filters by kind;
    # this is the authoritative backstop against a forged/stale request.)
    from services.custom_templates import aspect_kind as _aspect_kind
    _ft = _custom_template(frame_layout)
    if _ft:
        if _aspect_kind(_ft.canvas_w, _ft.canvas_h) != "short":
            raise HTTPException(status_code=400, detail=(
                f"'{_ft.name}' is a full-form (landscape) template — it can't be used as a "
                f"Short. Pick a 9:16 short template, or choose Full video / Both."))
    _fft = _custom_template(fullform_layout)
    if _fft:
        if _aspect_kind(_fft.canvas_w, _fft.canvas_h) != "full":
            raise HTTPException(status_code=400, detail=(
                f"'{_fft.name}' is a short (portrait) template — it can't be used as a "
                f"Full-form video. Pick a 16:9 full-form template, or choose Short / Both."))

    job = models.Job(
        user_id=user.id,
        platform=platform,
        frame_layout=frame_layout,
        video_name=_src_filename,
        name=_name_clean,
        language=lang_cfg.code,
        status="pending",
        log=_warning_prefix,
        output_dir=str(OUTPUT_ROOT),
        transition_style=_ts,
        stage_2_provider=_s2p,
        v4_trim_planner=_trim_planner_for_row,
        v4_image_provider=_image_provider_for_row,
        v4_predefined_description=_predef_desc_for_row,
        v4_output_format=_output_format_for_row,
        v4_effects_mode=_effects_mode_for_row,
        v4_theme=_theme_for_row,
        v4_style_directives=_style_directives_for_row,
        v4_director_engine=_director_engine_for_row,
        v4_defer_render=bool(v4_defer_render),
        v4_audio_first=bool(v4_audio_first),
        v4_target_platform=_target_platform_for_row,
        target_channel_ids=_chan_ids_for_row,
        intro_overrides=_intro_overrides_for_row,
        template_media=_template_media_for_row,
        main_media_slot=_main_media_slot_for_row,
        fullform_layout=((fullform_layout or "").strip().lower()
                         if (fullform_layout or "").strip().lower().startswith("custom:") else ""),
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
    bulletin_image_labels: dict[str, str] = {}   # path → subject label (name-tag contract)
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
                _label = (getattr(_asset, "description", "") or "").strip()
                if _label:
                    bulletin_image_labels[_path] = _label[:120]
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
        bulletin_image_labels=bulletin_image_labels,
        # V2 only (Step 11.4): ignored unless platform=full_video_shorts_v2.
        stt_provider=stt_provider,
        # Item 104: V2 only. V1 paths ignore.
        transition_style=_ts,
        # Item 114: V2 only. V1 paths ignore.
        stage_2_provider=_s2p,
        # V4 only: studio bg the user picked in the new-job wizard.
        # runner forwards into the V4 orchestrator, which stamps it onto
        # the freshly-built canvas.json so the very first render uses it.
        v4_bg_video_path=(v4_bg_video_path or "").strip() or None,
        v4_bg_video_volume=max(0.0, min(1.0, v4_bg_video_volume or 0.0)),
        v4_bg_intro_seconds=max(0.0, min(30.0, v4_bg_intro_seconds or 0.0)),
        # V4 only: planner pick from the wizard — runner validates +
        # forwards as KAIZER_V4_TRIM_PLANNER env to the orchestrator.
        v4_trim_planner=(v4_trim_planner or "claude").strip().lower(),
        # V4 only: content type ("auto" = classify; explicit = the
        # operator's answer). Runner forwards as KAIZER_V4_CONTENT_TYPE.
        v4_content_type=(v4_content_type or "auto").strip().lower(),
        # V4 only: image-provider pick — runner forwards as
        # KAIZER_V4_IMAGE_PROVIDER to image_provider._selected_image_provider.
        v4_image_provider=(v4_image_provider or "auto").strip().lower(),
        # V4 only: predefined description text — runner forwards as
        # KAIZER_V4_PREDEFINED_DESCRIPTION env so the orchestrator
        # switches to source-preserved mode.
        v4_predefined_description=_predef_desc_for_row or "",
        # V4 only: render-output choice — runner validates + forwards as
        # KAIZER_V4_OUTPUT_FORMAT so the orchestrator skips the bulletin
        # or the shorts accordingly.
        v4_output_format=_output_format_for_row or "both",
        # V4 only: full-form effects mode — runner validates + forwards as
        # KAIZER_V4_EFFECTS_MODE; per-story compose applies the chain and
        # hashes it (legacy NULL rows render byte-identically).
        v4_effects_mode=_effects_mode_for_row or "off",
        v4_theme=_theme_for_row or "",
        # V4 only: the user's per-category effect picks — runner validates +
        # forwards as KAIZER_V4_STYLE_DIRECTIVES so the AI Director's vocab is
        # constrained to the picks (empty categories stay AI-decided).
        v4_style_directives=(getattr(job, "v4_style_directives", "") or ""),
        # V4 only: which AI Director engine — runner forwards as
        # KAIZER_V4_DIRECTOR_ENGINE ("v4" default | "platform").
        v4_director_engine=(getattr(job, "v4_director_engine", "") or "v4"),
        # V4 Stage 2: defer the up-front render (persisted on the job above).
        v4_defer_render=bool(getattr(job, "v4_defer_render", False)),
        # V4 only: shorts-per-job ceiling (default 8; operator opt-in for more).
        v4_max_shorts=max(1, min(50, int(v4_max_shorts or 8))),
        # "Full form video" (16:9) custom template, e.g. "custom:<id>".
        fullform_layout=(job.fullform_layout or ""),
        # V4 audio-first: narration is the source (video_path above); forward
        # the optional muted reference b-roll so the orchestrator muxes it.
        v4_audio_first=bool(v4_audio_first),
        v4_ref_video_path=_ref_video_path,
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
    # Wave 2 (API scale): same "create" bucket as /api/jobs/create/ —
    # this endpoint also mints a Job + Clip and hits R2/ffmpeg.
    _rl=Depends(rate_limited("create")),
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
    # Wave 2 (API scale): chunked streaming write — never buffer the
    # whole upload in RAM (these are full edited videos, often GBs).
    with open(video_path, "wb") as f:
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

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
        # Full-form (16:9) custom template, e.g. "custom:18". The V4 editor reads this so
        # "Edit layout" opens the TEMPLATE editor (not the old region editor) for the bulletin.
        "fullform_layout": (job.fullform_layout or ""),
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
        # Item 114: surface the Stage 2 provider selection.
        "stage_2_provider": (job.stage_2_provider or "gemini"),
        # V4 KEEP/CUT planner — NULL for non-V4 platforms. Editor reads
        # this to render the "Planned by Claude / Gemini" badge.
        "v4_trim_planner": job.v4_trim_planner,
        # V4 image-provider pick — NULL for non-V4 platforms. Editor
        # uses this for the "Images: gemini/openai/auto" badge.
        "v4_image_provider": job.v4_image_provider,
        # V4 source-preserved mode — non-empty when the operator
        # supplied a verbatim bulletin description at submit time.
        # Editor renders a "Source preserved" badge when present.
        "v4_predefined_description": job.v4_predefined_description,
        # V4 render-output choice ("both"/"full-only"/"shorts-only").
        # NULL for non-V4 platforms; editor shows a "Shorts only" / etc. badge.
        "v4_output_format": job.v4_output_format,
        # V4 original publish target — editor leads with this platform's SEO.
        "v4_target_platform": job.v4_target_platform,
        # Channels chosen at generate time ("Choose channels" step) — a list
        # of Channel ids, or [] if not chosen here. Editor/Publish default to these.
        "target_channel_ids": (
            (lambda raw: (json.loads(raw) if raw else []) or [])(
                getattr(job, "target_channel_ids", None)
            )
        ),
        # Per-channel intro overrides: {channel_id: asset_id} or {} = each
        # channel uses its own assigned intro. Editor/UI read these.
        "intro_overrides": (
            (lambda raw: (json.loads(raw) if raw else {}) or {})(
                getattr(job, "intro_overrides", None)
            )
        ),
        # Bulletin clips render first (16:9 long-form takes the lead
        # tile in the JobDetail grid), shorts follow in DB-insert
        # order. Backlog item 91 follow-up.
        "clips": [
            _clip_dict(c) for c in sorted(
                job.clips,
                key=lambda c: (0 if c.frame_type == "bulletin" else 1, c.id),
            )
        ],
        # Published YouTube videos — surfaced for Quick Publish / raw-upload jobs (which have no
        # canvas to edit, so the detail page links to YouTube instead). Empty for pipeline jobs.
        "published_videos": (
            _published_videos_for_jobs(db, [job.id]).get(job.id, [])
            if (job.frame_layout or "") == "raw_upload" else []
        ),
    }


# Wave 2 (API scale): in-process micro-cache for the status poll. The
# frontend polls every visible job card every few seconds; at scale
# that's a query storm against rows that change at most once per
# pipeline step. A 2s TTL keeps the UI feeling live while collapsing
# the storm to ≤1 query per job per 2s per process. The cache stores
# the FULL payload plus the owner's user_id — ownership is re-checked
# on every hit; a mismatch falls through to the DB query (and its 404).
_STATUS_CACHE_TTL_S = 2.0
# Serve-stale ceiling: while ONE request refreshes an expired entry,
# the herd may be served a copy up to this old (load-test finding: at
# 300 pollers/job the synchronized 2s expiry caused multi-second p95
# stampede spikes — single-flight + bounded staleness flattens them).
_STATUS_CACHE_STALE_MAX_S = 10.0
_STATUS_CACHE_MAX   = 1000
_status_cache: dict[int, tuple[float, int, dict]] = {}  # job_id -> (mono_ts, owner_user_id, payload)
_status_cache_locks: dict[int, _threading.Lock] = {}    # job_id -> refresh lock
_status_cache_locks_guard = _threading.Lock()


def _status_cache_get(
    job_id: int, user_id: int, *, allow_stale: bool = False,
) -> Optional[dict]:
    entry = _status_cache.get(job_id)
    if not entry:
        return None
    ts, owner_id, payload = entry
    if owner_id != user_id:
        return None
    age = time.monotonic() - ts
    limit = _STATUS_CACHE_STALE_MAX_S if allow_stale else _STATUS_CACHE_TTL_S
    if age >= limit:
        return None
    return payload


def _status_cache_put(job_id: int, user_id: int, payload: dict) -> None:
    if len(_status_cache) > _STATUS_CACHE_MAX:
        # Bound the dicts: drop anything older than 60s (long-dead polls).
        cutoff = time.monotonic() - 60.0
        for k in [k for k, (ts, _, _) in list(_status_cache.items()) if ts < cutoff]:
            _status_cache.pop(k, None)
            with _status_cache_locks_guard:
                _status_cache_locks.pop(k, None)
    _status_cache[job_id] = (time.monotonic(), user_id, payload)


def _slice_status_log(payload: dict, since: int) -> dict:
    """Apply the ``since`` incremental-log window per request — the cache
    always holds the FULL payload. since=0 keeps the response shape
    byte-identical for existing pollers (log_offset is purely additive)."""
    if since <= 0:
        return payload
    out = dict(payload)
    out["log_lines"] = (payload.get("log_lines") or [])[since:]
    return out


@app.get("/api/jobs/{job_id}/status/")
def get_job_status(
    job_id: int,
    # Wave 2 (API scale): incremental log polling. since=N returns only
    # log_lines[N:]; clients track the returned log_offset (total line
    # count) and pass it back so each poll ships only new lines.
    since: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    # auth.current_user already ran (dependency) — the cache only
    # short-circuits the job-row query, never the auth itself.
    cached = _status_cache_get(job_id, user.id)
    if cached is not None:
        return _slice_status_log(cached, since)

    # Single-flight refresh: exactly ONE request re-queries an expired
    # entry; the rest of the herd is served the stale copy (≤10s old)
    # or waits for the refresher. Kills the synchronized-expiry
    # stampede the load test exposed at 300 pollers/job.
    with _status_cache_locks_guard:
        refresh_lock = _status_cache_locks.setdefault(job_id, _threading.Lock())
    if not refresh_lock.acquire(blocking=False):
        stale = _status_cache_get(job_id, user.id, allow_stale=True)
        if stale is not None:
            return _slice_status_log(stale, since)
        refresh_lock.acquire()  # nothing stale to serve — wait for refresh
    try:
        # Double-check: the refresher may have repopulated while we
        # waited on the lock.
        fresh = _status_cache_get(job_id, user.id)
        if fresh is not None:
            return _slice_status_log(fresh, since)

        job = db.query(models.Job).filter(
            models.Job.id == job_id, models.Job.user_id == user.id,
        ).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        log_lines = (job.log or "").split("\n") if job.log else []
        progress_pct = _estimate_progress(log_lines, job.status)
        payload = {
            "status": job.status,
            "progress_pct": progress_pct,
            "log_lines": log_lines,
            # Total line count — clients pass this back as ?since= to poll
            # incrementally. Additive field; old pollers ignore it.
            "log_offset": len(log_lines),
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
        _status_cache_put(job_id, user.id, payload)
        return _slice_status_log(payload, since)
    finally:
        refresh_lock.release()


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

    # V4 doesn't produce editor_meta.json — it produces canvas.json,
    # and the canvas editor reads it directly without going through
    # the Clip table. Reimport is a no-op for V4 jobs and must not
    # touch the job's status / error fields.
    if (job.platform or "") == "full_video_shorts_v4":
        return {"ok": True, "platform": job.platform, "note": "V4 jobs use canvas.json, not editor_meta.json. Nothing to reimport."}

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


@app.post("/api/jobs/{job_id}/pause/")
def pause_job_endpoint(
    job_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Hold a QUEUED job so it doesn't run when its turn comes.

    Pause only applies to a job still waiting in the queue (``pending`` /
    ``queued``). It sets ``status='paused'``; the render worker's post-slot
    gate then skips it, letting the next job proceed. A ``running`` job can't
    be suspended mid-encode — cancel it (and retry) instead. Resume with
    ``/resume/``.
    """
    job = db.query(models.Job).filter(
        models.Job.id == job_id, models.Job.user_id == user.id,
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == "running":
        raise HTTPException(status_code=409, detail=(
            "This job is already rendering — you can't pause it mid-render. "
            "Cancel it and retry later instead."))
    if job.status not in ("pending", "queued"):
        raise HTTPException(status_code=409, detail=(
            f"Only a queued job can be paused (this one is '{job.status}')."))
    job.status = "paused"
    db.commit()
    return {"job_id": job_id, "status": "paused"}


@app.post("/api/jobs/{job_id}/resume/")
def resume_job_endpoint(
    job_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Put a paused job back in the queue (re-launches it via the runner)."""
    job = db.query(models.Job).filter(
        models.Job.id == job_id, models.Job.user_id == user.id,
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "paused":
        raise HTTPException(status_code=409, detail=(
            f"Only a paused job can be resumed (this one is '{job.status}')."))
    try:
        import runner as _runner
        result = _runner.relaunch_job(job_id, SessionLocal)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Resume failed: {exc}")
    if not result.get("relaunched"):
        raise HTTPException(status_code=409,
                            detail=result.get("error", "Could not resume this job."))
    return {"job_id": job_id, "status": "pending"}


@app.post("/api/jobs/{job_id}/retry/")
def retry_job_endpoint(
    job_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Re-run a finished/failed/cancelled/paused job WITHOUT re-uploading.

    Reconstructs the render from the job's saved settings (source upload + all
    V4 render params) and re-queues it. Creation-time-only extras that aren't
    stored on the row (custom logo/default image, pre-picked bulletin images, a
    studio bg clip) are not re-applied.
    """
    job = db.query(models.Job).filter(
        models.Job.id == job_id, models.Job.user_id == user.id,
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status in ("running", "queued"):
        raise HTTPException(status_code=409, detail=(
            f"This job is currently {job.status} — cancel it before retrying."))
    try:
        import runner as _runner
        result = _runner.relaunch_job(job_id, SessionLocal)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retry failed: {exc}")
    if not result.get("relaunched"):
        raise HTTPException(status_code=409,
                            detail=result.get("error", "Could not retry this job."))
    return {"job_id": job_id, "status": "pending"}


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
    title_text = clip.text or meta.get("text", "KAIZER X")
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
            follow_text=follow_params.get("follow_text", "FOLLOW KAIZER X TELUGU"),
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

    # Wave 2 (API scale): chunked streaming write (same pattern as the
    # video uploads — images are small but the pattern costs nothing).
    with open(img_path, "wb") as f:
        while True:
            chunk = await image.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

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

    # Wave 2 (API scale): when there's no logo to burn and the clip only
    # exists in R2 (prod's ephemeral disk), redirect to a signed URL
    # instead of downloading the file just to re-stream it through
    # Python. Logo overlays and local/beta renders keep the existing
    # path — those genuinely need local bytes. (302 on POST → browsers
    # re-issue as GET, which is exactly what the signed URL expects.)
    if (
        not logo_path
        and (os.getenv("STORAGE_BACKEND", "local") or "local").strip().lower() == "r2"
        and (clip.storage_key or "").strip()
        and not (clip.file_path and _Path(clip.file_path).exists())
        and not (BASE_DIR / "output" / "beta_renders" / f"clip_{clip_id}" / "latest.json").exists()
    ):
        try:
            from pipeline_core.storage import get_storage_provider
            url = get_storage_provider(clip.storage_backend or None).get_url(
                clip.storage_key, signed=True,
            )
            return RedirectResponse(url, status_code=302)
        except Exception as exc:
            # Fall through to the download + stream path on any failure.
            print(f"[download-with-logo] R2 redirect failed (falling back): {exc}")

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


def _r2_redirect_for_path(db: Session, raw_path: str) -> Optional[RedirectResponse]:
    """Wave 2 (API scale): map an /api/file/?path=… request onto a Clip
    row's R2 object and 302 to it instead of proxying bytes through
    Python (one uvicorn worker can only stream so many 100 MB clips at
    once). Matches the exact path strings _clip_dict._furl builds URLs
    from. Returns None when nothing matches — caller falls through to
    the local streaming path unchanged.
    """
    try:
        clip = (
            db.query(models.Clip)
              .filter(
                  (models.Clip.file_path == raw_path)
                  | (models.Clip.thumb_path == raw_path)
                  | (models.Clip.image_path == raw_path)
              )
              .first()
        )
        if clip is None:
            return None
        # Rendered video → mint a fresh signed URL from the key (URLs
        # stored at upload time may be expired signatures when
        # R2_PUBLIC_BASE_URL is unset).
        if raw_path == (clip.file_path or "") and (clip.storage_key or "").strip():
            from pipeline_core.storage import get_storage_provider
            url = get_storage_provider(clip.storage_backend or None).get_url(
                clip.storage_key, signed=True,
            )
            return RedirectResponse(url, status_code=302)
        # Thumb / editorial image only persist a URL (no key column) —
        # redirect when one was captured at upload time.
        if raw_path == (clip.thumb_path or "") and (clip.thumb_storage_url or "").strip():
            return RedirectResponse(clip.thumb_storage_url, status_code=302)
        if raw_path == (clip.image_path or "") and (clip.image_storage_url or "").strip():
            return RedirectResponse(clip.image_storage_url, status_code=302)
    except Exception as exc:
        # Redirect resolution is best-effort — fall back to streaming.
        print(f"[serve_file] R2 redirect lookup failed (non-fatal): {exc}")
    return None


@app.get("/api/file/")
async def serve_file(path: str, request: Request, db: Session = Depends(get_db)):
    file_path = Path(path)
    if not _is_safe_path(file_path):
        raise HTTPException(status_code=403, detail="Path is not under an allowed root")

    # Wave 2 (API scale): on R2 deployments, hand the byte-shovelling to
    # Cloudflare via a signed-URL redirect. Local dev (STORAGE_BACKEND
    # unset/local) keeps the streaming path below byte-identical.
    if (os.getenv("STORAGE_BACKEND", "local") or "local").strip().lower() == "r2":
        redirect = _r2_redirect_for_path(db, path)
        if redirect is not None:
            return redirect

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

    # Wrap the file handle in a generator so it's closed when streaming
    # finishes (or the client disconnects). Passing a raw `open(...)` to
    # StreamingResponse leaks descriptors on every request — the source
    # of the "unclosed file" ResourceWarning spam in backend.err.log.
    def _whole_file_iter():
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(
        _whole_file_iter(), media_type=mime,
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
        # Per-short selection label (V4): why this segment was auto-picked +
        # its priority rank (1 = highest). Null on bulletin/legacy clips.
        "short_priority": meta.get("short_priority"),
        "short_why":      meta.get("short_why", ""),
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


# ── V2 Inngest serve mount — REMOVED 2026-06-17 ──────────────────────────────
# The Inngest-orchestrated pipeline v2 was retired; its /api/inngest webhook
# mount and the pipeline_v2 package are gone. V4 is the single render path.


# ── Desktop SPA ──────────────────────────────────────────────────────────────
# Desktop mode serves the built frontend from the backend itself so the app
# is one local origin (http://127.0.0.1:<port>) — no separate web server.
# MUST be the LAST route registered in this module: a "/" mount matches every
# path by prefix, so anything registered after it would be unreachable
# (that's why it is NOT next to the /media mount up top). API routes above
# win because Starlette matches routes in registration order.
if _DESKTOP:
    _spa_dir = (os.environ.get("KAIZER_SPA_DIR", "") or "").strip() \
        or str(BASE_DIR / "spa_dist")
    if os.path.isdir(_spa_dir):
        # SPA-aware static server: 404s outside api/ and media/ fall back
        # to index.html so client-side routes (/app, /jobs/5 ...) survive a
        # hard reload / deep link; api/media 404s stay real JSON 404s.
        from routers.desktop_local import SPAStaticFiles as _SPAStaticFiles  # noqa: E402
        app.mount("/", _SPAStaticFiles(directory=_spa_dir, html=True),
                  name="spa")
        print(f"[startup] desktop SPA served from {_spa_dir}")
    else:
        print(f"[startup] desktop SPA dir not found ({_spa_dir}) — API only")
