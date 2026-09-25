from sqlalchemy import (
    Column, Integer, BigInteger, String, Text, DateTime, ForeignKey, Float, Boolean,
    JSON, UniqueConstraint, Index, CheckConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base


class User(Base):
    """App-level user account — distinct from a connected YouTube account.

    `password_hash` is nullable so users who sign in exclusively with Google
    don't need to set a password.  `google_sub` is the OpenID Connect
    subject from Google Sign-In (NOT the YouTube channel id).
    """
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    email         = Column(String(255), unique=True, nullable=False, index=True)
    name          = Column(String(255), default="")
    password_hash = Column(String(255), nullable=True)   # argon2 or bcrypt
    google_sub    = Column(String(64),  nullable=True, unique=True, index=True)
    is_active     = Column(Boolean, default=True)
    is_admin      = Column(Boolean, default=False)
    # Creative role: same permissions as a normal user PLUS the ability
    # to upload videos to the shared company Library (see LibraryItem
    # below). Admins implicitly have this. Toggled by an admin via the
    # admin UI or a one-shot SQL update.
    is_creative   = Column(Boolean, default=False, nullable=False)
    # Which LLM writes this user's SEO: 'gemini' (default) | 'claude'.
    # Same prompts, same verifier, same learning either way — only the
    # writer is swapped.
    seo_engine    = Column(String(10), default="gemini")
    # Profile picture (image or GIF). Stored on R2 at
    # ``library/users/<id>/avatar.<ext>``. URL minted on demand by the
    # storage provider; empty = show initials fallback. Used everywhere
    # a creator is rendered.
    avatar_key    = Column(String(500), default="", nullable=False)
    # Cached aggregate of the creator's per-user ratings. Sum and count
    # bumped by CreatorRating writes so list endpoints never have to
    # GROUP BY at query time.
    creator_rating_sum   = Column(Integer, default=0, nullable=False)
    creator_rating_count = Column(Integer, default=0, nullable=False)
    # Cross-promo links used by SEO to populate "follow me" footer in descriptions.
    # Free-form dict: {"twitter":"@...", "instagram":"...", "whatsapp_community":"...", "website":"...", ...}
    socials       = Column(JSON, default=dict)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())
    last_login_at = Column(DateTime(timezone=True), nullable=True)

    # ── Billing / subscription ──────────────────────────────────────────
    # `plan` matches keys in billing/plans.py.  New accounts default to "free".
    # Stripe fields are populated later when the actual billing pipe is wired;
    # kept nullable so development works without Stripe configured.
    plan               = Column(String(20),  default="free", index=True)
    plan_cycle         = Column(String(10),  default="monthly")   # monthly | yearly
    plan_renews_at     = Column(DateTime(timezone=True), nullable=True)
    # Stripe bookkeeping — set by webhooks once live.  Null on free tier.
    stripe_customer_id     = Column(String(64), nullable=True, index=True)
    stripe_subscription_id = Column(String(64), nullable=True, index=True)
    # Monthly usage counter.  Reset to 0 on the first of every month by a
    # cron (or lazily on first check after `usage_reset_at`).
    monthly_clip_count = Column(Integer,     default=0)
    usage_reset_at     = Column(DateTime(timezone=True), nullable=True)

    # ── HeyGen avatar defaults ──────────────────────────────────────────
    # Per-user "remember my last pick" for the Trending → HeyGen flow.
    # Either can be null; the picker UI falls back to the server-wide
    # HEYGEN_DEFAULT_AVATAR_ID / HEYGEN_DEFAULT_VOICE_ID env when both
    # are blank. Stored as plain ids (HeyGen returns ~32-char strings).
    heygen_avatar_id = Column(String(64), nullable=True)
    heygen_voice_id  = Column(String(64), nullable=True)

    # ── V4 automation defaults ─────────────────────────────────────────
    # JSON blob the V4 pipeline applies when the user runs in auto mode
    # ("upload + publish to YouTube — no clicks in between"). See
    # routers/v4_defaults.py for the shape. Stored as text so we can
    # extend the schema without an Alembic migration on every iteration.
    v4_defaults = Column(Text, nullable=True)

    # ── Upload Rewrite v2: plan-tier link ─────────────────────────────
    # FK to plan_tiers row (free | pro | enterprise). Drives slot caps,
    # monthly credit allotment, direct-path gating. Nullable for now to
    # keep the migration safe; _migrate_schema() backfills existing rows
    # to the 'pro' tier id immediately after the table is created and
    # seeded. See docs/upload-rewrite/CONTRACTS.md §3.8 + DECISIONS.md
    # Decision 8. The legacy ``plan`` string column above stays during
    # the transition and is dropped in Phase 3.
    plan_tier_id  = Column(Integer, ForeignKey("plan_tiers.id", ondelete="SET NULL"), nullable=True, index=True)
    plan_tier     = relationship("PlanTier", foreign_keys=[plan_tier_id])


class Job(Base):
    __tablename__ = "jobs"

    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    status       = Column(String(20), default="pending")   # pending | running | done | failed | cancelled
    platform     = Column(String(50))
    frame_layout = Column(String(50))
    # Custom-template per-slot media (services/custom_templates). JSON map
    # {slot_key: asset_id} for the slots the user filled; the MAIN one
    # (main_media_slot) is the clip Kaizer AI-trims — the rest are used as-is.
    template_media  = Column(JSON, default=dict)
    main_media_slot = Column(String(64), default="")
    # Per-slot TEXT overrides set in the custom-template editor. JSON {slot_key: text} —
    # these REPLACE the AI/filler-generated text for that slot at render (operator retypes
    # the headline/ticker/etc.); absent slots fall back to the filler. Read by BOTH the
    # orchestrator render and the editor re-render.
    template_overrides = Column(JSON, default=dict)
    # "Full form video" (16:9) custom template, e.g. "custom:<id>" — separate from
    # frame_layout (which is the 9:16 shorts template). "" = built-in bulletin.
    fullform_layout = Column(String(50), default="")
    video_name   = Column(String(255))
    # V2 Beta (Phase 14 / D-13.11): user-supplied human label for the
    # job. Shown in JobsList + JobDetail. Defaults to first 80 chars
    # of video_name when the form field is left blank. Editable mid-
    # flight via PATCH /api/jobs/{id}/rename. NULL on pre-Phase-14 jobs.
    name         = Column(String(120), nullable=True, default=None)
    language     = Column(String(10), default="te")        # ISO 639-1: te|hi|ta|kn|ml|bn|mr|gu|en
    output_dir   = Column(String(500), default="")
    log          = Column(Text, default="")
    error        = Column(Text, default="")
    created_at   = Column(DateTime(timezone=True), server_default=func.now())
    # Wall-clock timing of the pipeline subprocess.
    started_at   = Column(DateTime(timezone=True), nullable=True)
    finished_at  = Column(DateTime(timezone=True), nullable=True)
    # User-initiated cancellation flag. The cancel endpoint sets this AND
    # calls runner.cancel_job() to walk the subprocess tree + kill ffmpeg
    # children. The flag persists on the DB row so a runner restart can
    # detect "this job was cancelled while we were down" and refuse to
    # resume it.
    cancel_requested = Column(Boolean, default=False, nullable=False)

    # V2-only field (per Step 10 D-10.7): the orchestrator writes the
    # current Inngest step name here at the start of each step so the
    # UI can show "Stage 4 of 6: Rendering bulletin..." progress on
    # long-running V2 jobs. Set to NULL at finalize. Stays NULL for
    # the legacy V1 subprocess path (4 platforms before "Full Video +
    # Shorts (V2)") which doesn't write progress mid-pipeline. Values:
    # stage_0_ingest | stage_1_transcribe | stage_2_continuity |
    # stage_2_5_entities | stage_3_fanout | stage_4_render | finalize
    current_stage = Column(String(40), default=None, nullable=True)

    # Item 104 (Transition library): operator's chosen inter-clip
    # transition for the bulletin pass. Catalog defined in
    # pipeline_v2.transitions. NULL on pre-item-104 rows; the renderer
    # falls back to "smart_cut" (the default + only implemented entry
    # at item-104 ship time) when this field is NULL or names a
    # not-yet-implemented entry. Stays NULL for V1 platforms.
    transition_style = Column(String(20), default="smart_cut", nullable=True)

    # Item 114 (Stage 2 provider catalog): operator's chosen LLM
    # for Stage 2 editorial decisions. One of {"gemini", "claude"}.
    # NULL on pre-item-114 rows; the dispatcher falls back to
    # "gemini" (the default) when this field is NULL or names an
    # unknown provider. Stays NULL for V1 platforms.
    stage_2_provider = Column(String(20), default="gemini", nullable=True)

    # V4 only: which model decided Step 1's KEEP/CUT plan.
    # "claude" (Opus 4.7) or "gemini" (2.5 Flash on Vertex).
    # Picked per-job in the new-job wizard so the operator can A/B
    # quality. NULL on pre-rollout rows; trim_engine._select_planner
    # falls back to "claude" when this is NULL or unknown. Stays
    # NULL for non-V4 platforms.
    v4_trim_planner  = Column(String(20), default="claude", nullable=True)

    # V4 only: which provider generated per-story images.
    # "auto" (V1 multi-source chain — Google CSE / DDG / Pexels /
    # OpenAI), "gemini" (Nano Banana — gemini-2.5-flash-image), or
    # "openai" (gpt-image-1). NULL on pre-rollout rows; the dispatcher
    # falls back to "auto". Stays NULL for non-V4 platforms.
    v4_image_provider = Column(String(20), default="auto", nullable=True)

    # V4 only: operator-supplied description text (Telugu / English /
    # any language). When NON-EMPTY, the orchestrator switches to
    # "source-preserved" mode: the bulletin is kept AS-IS (no Claude
    # KEEP/CUT), shorts are still carved, and the SEO description is
    # the operator's verbatim text. Used when the operator already has
    # a polished video they don't want re-edited. NULL = legacy flow
    # (Claude trim plan + AI-generated SEO description). Stays NULL
    # for non-V4 platforms.
    v4_predefined_description = Column(Text, default=None, nullable=True)

    # V4 only: which outputs the operator chose to render —
    #   "both" (full video + shorts), "full-only" (bulletin, no shorts),
    #   or "shorts-only" (shorts/reels, no bulletin). The orchestrator
    #   gates rendering on this and also stamps it onto canvas.json so
    #   editor re-renders honour the original choice. NULL/"both" for
    #   legacy + non-V4 rows.
    v4_output_format = Column(String(20), default="both", nullable=True)

    # V4 only: full-form EFFECTS mode the user picked in the wizard —
    #   "auto" (tasteful broadcast polish, the default for new jobs),
    #   "rich" (the job's content-type style-pack look on every story) or
    #   "off". NULL = legacy job → renders byte-identically (no effects,
    #   no cache invalidation). Forwarded as KAIZER_V4_EFFECTS_MODE.
    v4_effects_mode = Column(String(12), nullable=True)
    # V4 THEME PACK key ("" / NULL = classic look). Forwarded as
    # KAIZER_V4_THEME; the canvas persists it so re-renders wear it too.
    v4_theme = Column(String(24), nullable=True)

    # V4 only: the user's per-category effect picks ("edit using THESE") — a
    #   JSON object {style_packs,transitions,frame_fx,overlays,typography,
    #   story_category:[...]}. Empty/NULL = leave every category to the AI
    #   Director; any picked category constrains the Director to those ids
    #   (and forces it on). Forwarded as KAIZER_V4_STYLE_DIRECTIVES.
    v4_style_directives = Column(Text, nullable=True)

    # V4 only: WHICH AI Director engine plans the per-story direction.
    #   "v4" (default/NULL) = our full-arsenal per-story Director;
    #   "platform" = the ported kaizer-platform 3-layer engine (5 moods,
    #   signal formula + LLM confirm) adapted onto the V4 vocabulary.
    #   Forwarded as KAIZER_V4_DIRECTOR_ENGINE. Selects WHICH director
    #   runs, never WHETHER (that stays with v4_effects_mode).
    v4_director_engine = Column(String(12), default="v4", nullable=True)

    # V4 only: which BRAIN writes the Director's per-story plan.
    #   "gemini" (default/NULL) — also runs the tone/vision sensors
    #   (multimodal; Claude/ChatGPT plans are text-only by design);
    #   "claude" | "openai" (BYO key). Forwarded as
    #   KAIZER_V4_DIRECTOR_PROVIDER after key-aware resolution in the
    #   runner. Orthogonal to v4_director_engine (which LOGIC plans).
    v4_director_provider = Column(String(20), default="gemini", nullable=True)

    # V4 only: the original publish target the operator picked in the wizard
    #   ("instagram" / "youtube" / "facebook"). Reel/Short/Full all run on the
    #   same V4 engine, so this is the ONLY record of which platform was
    #   intended — the editor uses it to lead with that platform's SEO
    #   (Instagram caption+hashtags vs YouTube title+tags). NULL/"youtube" for
    #   legacy + non-V4 rows.
    v4_target_platform = Column(String(20), default="youtube", nullable=True)

    # V4 only (Stage 2): DEFER the heavy compose. When true, the pipeline produces the raw
    # cut video + canvas.json scene + Clip rows but skips the up-front MP4 render; the render
    # then happens on demand ("export") in the editor. Edit-first jobs finish fast. Default off
    # so existing/auto-publish jobs are unaffected.
    v4_defer_render = Column(Boolean, default=False, nullable=True)

    # V4 only: AUDIO-FIRST mode. True when the job was created from an uploaded
    # AUDIO narration (the master track) + optional MUTED reference video +
    # generated/uploaded images. NULL/False = normal video-first V4.
    v4_audio_first = Column(Boolean, default=False, nullable=True)

    # V4 only: the channels the operator chose AT GENERATE TIME to publish
    # this job to (JSON list of Channel ids, e.g. "[3,7,12]"). Recorded by
    # the New Job wizard's "Choose channels" step so the editor + Publish
    # flow know the intended targets up front. NULL = not chosen at generate
    # (channels picked later at publish, the legacy flow). The master render
    # is channel-agnostic; this only scopes per-channel SEO/branding/publish.
    target_channel_ids = Column(Text, default=None, nullable=True)

    # PER-CHANNEL intro overrides for this job: JSON map {"<channel_id>":
    # <UserAsset id>, ...}. A channel present here uses THAT intro for this
    # job; a channel ABSENT uses its own assigned intro (the default). Chosen
    # inline per channel at job creation (its own / platform-demo / upload).
    # The branding resolver folds the looked-up asset into the version hash so
    # each channel's override forks its brand cache cleanly.
    intro_overrides = Column(Text, default=None, nullable=True)

    # PER-JOB custom-template HTML overrides (custom:<id> renders only). JSON map
    # {"<target>:<index>": "<html>", ...} where target is "bulletin"|"short" and index
    # is the output index. When present for an output, the renderer uses THIS HTML
    # verbatim (the operator visually edited the template for this job in the inline
    # builder — moved/resized/recoloured/retyped) instead of the parent template +
    # filler. The PARENT template is never modified; only this job is affected. NULL =
    # no overrides (normal path). HTML is sanitized (scripts/external URLs stripped)
    # on save, same as an uploaded template. Column(JSON) (migrated as TEXT, the same
    # pattern as template_overrides/template_media) — read via _jdict, assign a fresh dict.
    custom_html_overrides = Column(JSON, default=dict)

    # (Legacy/unused) single per-job intro override — superseded by the
    # per-channel ``intro_overrides`` map above; kept so the column isn't
    # dropped. Always NULL on new jobs.
    intro_asset_id = Column(
        Integer, ForeignKey("user_assets.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )

    # Wave 2 (API scale): cached cover image for the jobs list. The
    # list endpoint's _resolve_thumb walk does 5+ filesystem stat()
    # calls per job; once a thumb resolves we write it back here so
    # subsequent listings read two columns instead of hitting disk.
    # NULL until the first listing after the job's thumbs exist
    # (running jobs, pre-Wave-2 rows). thumb_aspect is "16:9"/"9:16".
    thumb_url    = Column(String(500), nullable=True)
    thumb_aspect = Column(String(8), nullable=True)

    clips = relationship("Clip", back_populates="job", cascade="all, delete")


class Clip(Base):
    __tablename__ = "clips"

    id           = Column(Integer, primary_key=True, index=True)
    job_id       = Column(Integer, ForeignKey("jobs.id"))
    clip_index   = Column(Integer, default=0)
    filename     = Column(String(255), default="")
    file_path    = Column(String(500), default="")
    thumb_path   = Column(String(500), default="")
    image_path   = Column(String(500), default="")
    duration     = Column(Float, default=0)
    frame_type   = Column(String(50), default="")
    text         = Column(Text, default="")
    sentiment    = Column(String(50), default="")
    entities     = Column(Text, default="[]")    # JSON array
    card_params  = Column(Text, default="{}")    # JSON: font_size, text_color, font_file, card_style
    section_pct  = Column(Text, default="{}")    # JSON: {video, text, image}
    follow_params= Column(Text, default="{}")    # JSON: follow_text, bg_color, etc.
    meta         = Column(Text, default="{}")    # raw pipeline meta JSON
    seo          = Column(Text, default="")      # most-recently-generated SEO (JSON str, empty until generated) — kept for back-compat + "current" display
    seo_variants = Column(Text, default="{}")    # JSON dict {channel_id: enforced_seo_payload} — one variant per style profile

    # ── Storage (Phase 5) ──────────────────────────────────────────
    # When storage_backend='local' the URL is /media/<key>; when 'r2' the URL
    # is a CDN/public link or a signed URL (frontend re-fetches if expired).
    # file_path remains populated for backwards compatibility.
    storage_url     = Column(String(500), default="")
    storage_key     = Column(String(500), default="")
    storage_backend = Column(String(20),  default="")
    # Separate R2 URLs for the thumbnail and editorial image. storage_url
    # is reserved for the rendered video; these mirror it for the JPGs.
    thumb_storage_url = Column(String(500), default="")
    image_storage_url = Column(String(500), default="")

    job = relationship("Job", back_populates="clips")
    upload_jobs = relationship("UploadJob", back_populates="clip", cascade="all, delete")


# ─────────────────────────────────────────────────────────────────────────────
# Channels — YouTube channel profiles driving SEO + upload targeting
# ─────────────────────────────────────────────────────────────────────────────

class Channel(Base):
    __tablename__ = "channels"
    # Profile names are unique PER USER, not globally — two users can both
    # have a "Kaizer X Telugu" style profile.
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_channel_user_name"),)

    id                 = Column(Integer, primary_key=True, index=True)
    user_id            = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    name               = Column(String(255), nullable=False, index=True)
    handle             = Column(String(100), default="")
    # What this channel IS — the durable distinction the UI's two tabs rely on:
    #   "account" = a YouTube channel the user OWNS and publishes to (carries
    #               branding/logo/watermark/socials).
    #   "style"   = a competitor / style reference, used ONLY to mine SEO voice;
    #               never published to.
    # Set explicitly at creation. A DISCONNECTED account KEEPS kind="account"
    # (only its token is removed) so it never falls into the style list when
    # its token disappears — the bug this column fixes.
    kind               = Column(String(10), default="account", nullable=True, index=True)
    language           = Column(String(10), default="te")
    # OPT-IN (operator requirement): when True, SEO generation for this
    # channel receives the tracked competitors' topic-matched intelligence
    # (their winning titles to differentiate from + tags to harvest).
    use_competitor_intel = Column(Boolean, default=False)
    title_formula      = Column(Text, default="")
    desc_style         = Column(String(50), default="hook_first")
    footer             = Column(Text, default="")
    fixed_tags         = Column(JSON, default=list)
    hashtags           = Column(JSON, default=list)
    # Optional overlay logo for videos rendered under this channel.  FK to
    # a UserAsset the user picked from their Assets library.  Null = no
    # logo (the SaaS default — users opt in via the Style Profiles page).
    logo_asset_id      = Column(Integer, ForeignKey("user_assets.id", ondelete="SET NULL"), nullable=True, index=True)
    # Optional per-channel INTRO video, concatenated at the HEAD of the
    # branded clip in the branding pass (anti-duplicate lever: a unique
    # intro per channel changes the opening frames + adds distinct head
    # audio). FK to a UserAsset the user uploaded/picked. Null = no intro
    # (the default). BrandProfile.intro_asset_id mirrors this for the
    # profile-based resolve path; this column serves the common
    # synthesise-from-legacy-columns path (most channels have no BrandProfile).
    intro_asset_id     = Column(Integer, ForeignKey("user_assets.id", ondelete="SET NULL"), nullable=True, index=True)
    # ── Per-channel YouTube publish defaults (set from Kaizer so the
    # operator never opens YouTube Studio). Applied to every upload to this
    # channel at publish time. Only fields the YouTube Data API can actually
    # set live here — monetization/comments/remixing/AI-disclosure are
    # Studio-only / channel-default and are NOT modelled. ──
    yt_category_id     = Column(String(10),  nullable=True)   # e.g. "25" News&Politics; null=default
    yt_playlist_id     = Column(String(64),  nullable=True)   # YouTube playlist id; null=none
    yt_default_language= Column(String(10),  nullable=True)   # e.g. "te"; null=default "te"
    yt_made_for_kids   = Column(Boolean,     nullable=True)   # null=inherit default (False)
    yt_license         = Column(String(20),  nullable=True)   # "youtube" | "creativeCommon"; null=youtube
    mandatory_hashtags = Column(JSON, default=list)
    is_priority        = Column(Boolean, default=False)
    # Per-channel upload route override.  Null = use the system-wide
    # default from system_settings.UPLOAD_PROVIDER.  Lets the admin
    # send one channel through Postiz and another through the native
    # YouTube uploader simultaneously — useful while comparing the two
    # paths side-by-side, or when one channel's google project has
    # exhausted its daily quota.
    upload_provider    = Column(String(20), nullable=True)  # "postiz" | "kaizer" | "native_rtmp" | null
    # Postiz integration id this channel maps to (from Postiz's
    # GET /public/v1/integrations). Set when the user picks Postiz as the
    # delivery for this channel; the v2 dispatch routes upload_path='postiz'
    # jobs to this integration. NULL = not bound (postiz delivery inert).
    postiz_integration_id = Column(String(64), nullable=True)
    # Per-channel watermark — applied at upload time (before videos.insert)
    # so the same rendered file can ship to multiple destinations with
    # each channel's own brand stamp. Empty text = logo-only watermark
    # using the channel's logo_asset_id image.
    watermark_text     = Column(String(64), default="", nullable=True)
    watermark_opacity  = Column(Float,      default=0.35, nullable=True)
    watermark_position = Column(String(16), default="top-right", nullable=True)
    # Per-channel social links — injected into the SEO description
    # footer at publish time. Same shape as the per-user socials JSON
    # so the existing UI inputs (YouTube/Twitter/Instagram/...) port
    # over directly. None / {} = nothing added; the user-level socials
    # are still used as a fallback by compose() when this is empty.
    socials            = Column(JSON, default=dict, nullable=True)
    created_at         = Column(DateTime(timezone=True), server_default=func.now())
    updated_at         = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    oauth_token = relationship(
        "OAuthToken", back_populates="channel", uselist=False,
        cascade="all, delete-orphan",
    )
    upload_jobs = relationship(
        "UploadJob", back_populates="channel", cascade="all, delete-orphan",
    )
    corpus = relationship(
        "ChannelCorpus", back_populates="channel", uselist=False,
        cascade="all, delete-orphan",
    )


class ProfileDestination(Base):
    """Many-to-many: which style profiles are allowed to publish to which
    YouTube destinations (for a given user).

    `google_channel_id` identifies the real YouTube channel.  A profile is
    publishable to a destination iff a row exists here AND some profile
    owned by the same user has an OAuthToken matching that google_channel_id
    (so we can actually upload).
    """
    __tablename__ = "profile_destinations"
    __table_args__ = (
        UniqueConstraint("profile_id", "google_channel_id", name="uq_profdest"),
    )

    id                = Column(Integer, primary_key=True, index=True)
    profile_id        = Column(Integer, ForeignKey("channels.id"), nullable=False, index=True)
    google_channel_id = Column(String(50), nullable=False, index=True)
    created_at        = Column(DateTime(timezone=True), server_default=func.now())
    # ── Cached YouTube metadata (multi-channel picker UI) ─────────────────
    # Populated at OAuth callback time for every Brand Account on the
    # signed-in Google identity. Lets the frontend show channel name +
    # avatar without re-calling channels.list per render.
    channel_title         = Column(String(255), default="")
    channel_thumbnail_url = Column(String(500), default="")
    channel_custom_url    = Column(String(100), default="")
    subscriber_count      = Column(Integer, default=0)
    video_count           = Column(Integer, default=0)
    enabled               = Column(Boolean, default=True, nullable=False)


class OAuthToken(Base):
    __tablename__ = "oauth_tokens"

    id                   = Column(Integer, primary_key=True, index=True)
    channel_id           = Column(Integer, ForeignKey("channels.id"), unique=True, nullable=False)
    google_channel_id    = Column(String(50), default="")
    google_channel_title = Column(String(255), default="")
    refresh_token_enc    = Column(Text, default="")      # Fernet-encrypted base64
    access_token_enc     = Column(Text, default="")      # Fernet-encrypted base64
    token_expiry         = Column(DateTime(timezone=True), nullable=True)
    scopes               = Column(Text, default="")
    connected_at         = Column(DateTime(timezone=True), server_default=func.now())
    last_refreshed_at    = Column(DateTime(timezone=True), nullable=True)

    # Cached YouTube-channel metadata — populated on OAuth connect + on manual
    # refresh.  Eliminates repeat YT Data API calls for display-only fields.
    # Stale until the user clicks "refresh" on the Channels page.
    channel_description  = Column(Text,        default="")
    channel_thumbnail_url = Column(String(500), default="")
    channel_custom_url   = Column(String(120), default="")
    channel_country      = Column(String(10),  default="")
    subscriber_count     = Column(Integer,     default=0)
    video_count          = Column(Integer,     default=0)
    view_count           = Column(BigInteger,  default=0)
    metadata_cached_at   = Column(DateTime(timezone=True), nullable=True)

    # Per-YouTube-account logo overlay.  The logo is a property of the REAL
    # YT account (the user owns Auto Wala; "TV9 Telugu" is just a writing-
    # style template and never gets a logo).  Resolved at render time via
    # KAIZER_DEFAULT_LOGO env; empty = no overlay.
    logo_asset_id        = Column(Integer, ForeignKey("user_assets.id", ondelete="SET NULL"), nullable=True, index=True)

    # Per-YouTube-account upload route — same shape as the channel/job-
    # level column.  This is the LEVEL users naturally configure on
    # the "My YouTube Accounts" cards (one knob per real destination,
    # not per style profile).  Worker precedence: job → oauth_token →
    # channel → system default.  Null = inherit from channel/system.
    upload_provider      = Column(String(20), nullable=True)

    channel = relationship("Channel", back_populates="oauth_token")


class MetaAccount(Base):
    """A connected Meta destination — a Facebook Page and optionally
    a linked Instagram Business/Creator account that the operator owns.

    One row per Page (Meta's model). The ig_user_id is only set when
    the Page has an IG account linked AND the operator has granted
    the instagram_basic + instagram_content_publish permissions during
    OAuth.

    Tokens here are LONG-LIVED Page tokens (≈60 days). The OAuth
    refresh worker swaps them for fresh long-lived tokens before they
    expire — Meta tokens don't auto-refresh like Google's, but a
    long-lived token can mint another long-lived token at any point.

    Encryption: same Fernet pattern as OAuthToken — never store the
    plaintext on disk."""
    __tablename__ = "meta_accounts"

    id                   = Column(Integer, primary_key=True, index=True)
    user_id              = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"),
                                  nullable=False, index=True)

    # Facebook side — every row has these.
    fb_user_id           = Column(String(50), default="")   # the human Meta user that authorised
    fb_page_id           = Column(String(50), default="", index=True)
    fb_page_name         = Column(String(255), default="")
    fb_page_category     = Column(String(120), default="")
    fb_page_picture_url  = Column(String(500), default="")
    fb_page_url          = Column(String(500), default="")

    # Instagram side — present only when the Page has a linked IG
    # Business/Creator account AND the operator granted IG perms.
    ig_user_id           = Column(String(50), default="", index=True)
    ig_username          = Column(String(120), default="")
    ig_profile_pic_url   = Column(String(500), default="")
    ig_account_type      = Column(String(40),  default="")  # "BUSINESS" | "CREATOR" | ""

    # Auth — long-lived Page access token (~60 days). The user
    # access token isn't stored after the initial exchange; we
    # derive Page tokens from the user token at connect time and
    # store ONLY those.
    page_access_token_enc = Column(Text, default="")
    page_token_expiry     = Column(DateTime(timezone=True), nullable=True)
    granted_scopes        = Column(Text, default="")  # comma-separated

    connected_at          = Column(DateTime(timezone=True), server_default=func.now())
    last_refreshed_at     = Column(DateTime(timezone=True), nullable=True)
    last_publish_at       = Column(DateTime(timezone=True), nullable=True)

    # Per-account upload routing knob — matches the OAuthToken column
    # so the worker's 4-tier precedence works the same way for Meta
    # destinations. Null = inherit from system default.
    upload_provider       = Column(String(20), nullable=True)

    # Per-account quota counters. Meta's published rate limit is 200
    # calls/hour/user but the practical content-publish ceiling is
    # much lower (a few dozen video posts per day per Page). Tracked
    # for observability + soft-cap in the worker.
    publishes_today       = Column(Integer, default=0)
    publishes_today_at    = Column(DateTime(timezone=True), nullable=True)


class LinkedInAccount(Base):
    """One connected LinkedIn destination — either a personal profile
    or a Company Page the operator is an admin of.

    LinkedIn tokens are simpler than Meta's: a single short-lived
    access token (60 days) plus a refresh token (365 days) you can
    swap for a fresh access token at any time. Both are Fernet-
    encrypted at rest.

    The `linkedin_urn` is what we POST against. For a personal
    profile it's "urn:li:person:<id>"; for a Company Page it's
    "urn:li:organization:<id>". Single column covers both."""
    __tablename__ = "linkedin_accounts"

    id                = Column(Integer, primary_key=True, index=True)
    user_id           = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"),
                               nullable=False, index=True)

    linkedin_urn      = Column(String(120), default="", index=True)  # urn:li:person:... or urn:li:organization:...
    linkedin_id       = Column(String(64),  default="")              # raw id without urn prefix
    profile_name      = Column(String(255), default="")
    profile_headline  = Column(String(255), default="")
    profile_pic_url   = Column(String(500), default="")
    profile_url       = Column(String(500), default="")
    # "person" | "organization" — picks the right Posts API field
    # (author=urn vs. author=urn but with different payload shape).
    account_type      = Column(String(20),  default="person")

    access_token_enc  = Column(Text, default="")
    refresh_token_enc = Column(Text, default="")
    token_expiry      = Column(DateTime(timezone=True), nullable=True)
    granted_scopes    = Column(Text, default="")

    connected_at      = Column(DateTime(timezone=True), server_default=func.now())
    last_refreshed_at = Column(DateTime(timezone=True), nullable=True)
    last_publish_at   = Column(DateTime(timezone=True), nullable=True)

    upload_provider   = Column(String(20), nullable=True)
    publishes_today    = Column(Integer, default=0)
    publishes_today_at = Column(DateTime(timezone=True), nullable=True)


class ChannelGroup(Base):
    """User-defined group of YouTube destinations for one-click fan-out.

    Stores a list of `google_channel_id`s.  At publish time the user picks
    a group (e.g. "English", "Telugu") and every destination in the group is
    auto-selected — no manual checking on every upload.

    `is_default_all` is a virtual marker for the implicit "Global" group
    ("publish to every connected YT account").  The UI renders it but it's
    never persisted — this row stays user-managed only.
    """
    __tablename__ = "channel_groups"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_group_user_name"),)

    id                 = Column(Integer, primary_key=True, index=True)
    user_id            = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name               = Column(String(100), nullable=False)
    description        = Column(Text, default="")
    google_channel_ids = Column(JSON, default=list)  # ["UC_abc...", "UC_xyz..."]
    created_at         = Column(DateTime(timezone=True), server_default=func.now())
    updated_at         = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class OAuthState(Base):
    """Short-lived random state tokens to validate OAuth callbacks against CSRF."""
    __tablename__ = "oauth_states"

    id         = Column(Integer, primary_key=True, index=True)
    state      = Column(String(64), unique=True, nullable=False, index=True)
    channel_id = Column(Integer, ForeignKey("channels.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class UploadJob(Base):
    __tablename__ = "upload_jobs"

    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    clip_id        = Column(Integer, ForeignKey("clips.id"), nullable=False, index=True)
    channel_id     = Column(Integer, ForeignKey("channels.id"), nullable=False, index=True)
    status         = Column(String(20), default="queued", index=True)
    # queued | uploading | processing | done | failed | cancelled
    privacy_status = Column(String(20), default="private")  # public | unlisted | private
    publish_kind   = Column(String(10), default="video")    # "video" | "short"
    publish_at     = Column(DateTime(timezone=True), nullable=True)
    title          = Column(Text, default="")
    description    = Column(Text, default="")
    tags           = Column(JSON, default=list)
    category_id    = Column(String(10), default="25")   # 25 = News & Politics
    made_for_kids  = Column(Boolean, default=False)
    # Per-publish upload route override.  Wins over Channel.upload_provider
    # which wins over system_settings.UPLOAD_PROVIDER.  Lets a user pick
    # "send this one through Postiz / send this one through native" at
    # publish time — primarily a comparison tool while we validate the
    # two paths produce identical YouTube metadata.
    upload_provider = Column(String(20), nullable=True)  # "postiz" | "kaizer" | null
    video_id       = Column(String(50), default="")     # YouTube video ID after insert
    upload_uri     = Column(Text, default="")           # resumable session URI
    bytes_uploaded = Column(Integer, default=0)
    bytes_total    = Column(Integer, default=0)
    attempts       = Column(Integer, default=0)
    last_error     = Column(Text, default="")
    log            = Column(Text, default="")
    created_at     = Column(DateTime(timezone=True), server_default=func.now())
    updated_at     = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    clip    = relationship("Clip", back_populates="upload_jobs")
    channel = relationship("Channel", back_populates="upload_jobs")


class ApiQuota(Base):
    __tablename__ = "api_quota"

    id           = Column(Integer, primary_key=True, index=True)
    date         = Column(String(10), nullable=False)     # YYYY-MM-DD (IST)
    api_key_hash = Column(String(16), nullable=False)
    units_used   = Column(Integer, default=0)

    __table_args__ = (
        UniqueConstraint("date", "api_key_hash", name="uq_api_quota_date_key"),
        Index("ix_api_quota_date", "date"),
    )


class ChannelCorpus(Base):
    __tablename__ = "channel_corpus"

    id           = Column(Integer, primary_key=True, index=True)
    channel_id   = Column(Integer, ForeignKey("channels.id"), unique=True, nullable=False)
    payload      = Column(JSON, default=dict)    # {top_titles, hook_patterns, emotional_triggers, power_words}
    refreshed_at = Column(DateTime(timezone=True), server_default=func.now())

    channel = relationship("Channel", back_populates="corpus")


# ─────────────────────────────────────────────────────────────────────────────
# Phase A — Auto-Publish Campaigns
# ─────────────────────────────────────────────────────────────────────────────

class Campaign(Base):
    __tablename__ = "campaigns"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_campaign_user_name"),)

    id                 = Column(Integer, primary_key=True, index=True)
    user_id            = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    name               = Column(String(255), nullable=False, index=True)
    channel_ids        = Column(JSON, default=list)         # [int, ...] — fan out across these
    spacing_minutes    = Column(Integer, default=120)       # gap between scheduled slots
    privacy_status     = Column(String(20), default="private")  # public | unlisted | private
    auto_seo           = Column(Boolean, default=True)
    auto_translate_to  = Column(JSON, default=list)         # ["hi", "ta", "en"] — Phase D fan-out
    daily_cap          = Column(Integer, default=0)         # 0 = unlimited
    quiet_hours_start  = Column(Integer, default=0)         # 0-23, skip slots in this window
    quiet_hours_end    = Column(Integer, default=0)
    thumbnail_ab       = Column(Boolean, default=False)     # Phase C
    active             = Column(Boolean, default=True)
    created_at         = Column(DateTime(timezone=True), server_default=func.now())
    updated_at         = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class JobCampaign(Base):
    """Link a pipeline job to a campaign — when the job finishes all its clips
    are auto-enqueued according to the campaign's rules."""
    __tablename__ = "job_campaigns"

    id          = Column(Integer, primary_key=True, index=True)
    job_id      = Column(Integer, ForeignKey("jobs.id"), nullable=False, index=True)
    campaign_id = Column(Integer, ForeignKey("campaigns.id"), nullable=False, index=True)
    status      = Column(String(20), default="pending")  # pending | seo | scheduled | done | failed
    last_error  = Column(Text, default="")
    created_at  = Column(DateTime(timezone=True), server_default=func.now())
    updated_at  = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (UniqueConstraint("job_id", "campaign_id", name="uq_job_campaign"),)


# ─────────────────────────────────────────────────────────────────────────────
# Phase B — Analytics feedback loop
# ─────────────────────────────────────────────────────────────────────────────

class ClipPerformance(Base):
    """Periodic snapshots of live video stats. Multiple rows per upload over
    time — we keep the history so we can plot trajectories."""
    __tablename__ = "clip_performance"

    id             = Column(Integer, primary_key=True, index=True)
    upload_job_id  = Column(Integer, ForeignKey("upload_jobs.id"), nullable=False, index=True)
    clip_id        = Column(Integer, ForeignKey("clips.id"), nullable=True, index=True)
    channel_id     = Column(Integer, ForeignKey("channels.id"), nullable=True, index=True)
    video_id       = Column(String(50), default="", index=True)
    views          = Column(Integer, default=0)
    likes          = Column(Integer, default=0)
    comments       = Column(Integer, default=0)
    seo_score      = Column(Integer, default=0)     # captured at upload time
    hours_since_publish = Column(Float, default=0)  # uploaded_at → sampled_at
    sampled_at     = Column(DateTime(timezone=True), server_default=func.now())


class TrainingSample(Base):
    """ONE training-ready row per published video: the content + the exact SEO
    that shipped (FEATURES), joined to the measured outcome (LABELS), plus meta.

    Snapshotted at poll time (and backfillable) so it's self-contained — the
    features reflect what actually published and survive later edits/deletes,
    and the export needs ZERO cleaning: feed it straight to a model. One row
    per ``video_id`` (upserted to the LATEST outcome)."""
    __tablename__ = "training_samples"

    id              = Column(Integer, primary_key=True, index=True)
    # ── identity / meta ──
    video_id        = Column(String(50), unique=True, index=True)
    user_id         = Column(Integer, index=True, nullable=True)
    channel_id      = Column(Integer, index=True, nullable=True)
    clip_id         = Column(Integer, nullable=True)
    upload_job_id   = Column(Integer, nullable=True)
    channel_name    = Column(String(255), default="")
    platform        = Column(String(20), default="youtube")  # youtube | instagram | facebook
    language        = Column(String(10), default="")
    kind            = Column(String(20), default="")         # bulletin | short layout
    # ── FEATURES (input) — snapshot of what was published ──
    content_text    = Column(Text, default="")               # story headline / clip text
    seo_title       = Column(Text, default="")
    seo_description = Column(Text, default="")
    seo_keywords    = Column(JSON, default=list)
    seo_hashtags    = Column(JSON, default=list)
    style_source_id = Column(Integer, nullable=True)         # competitor style used (or NULL)
    # derived numeric features — ready for ML, no parsing at train time
    title_len       = Column(Integer, default=0)
    desc_len        = Column(Integer, default=0)
    keyword_count   = Column(Integer, default=0)
    hashtag_count   = Column(Integer, default=0)
    seo_score       = Column(Integer, default=0)             # gen-time SEO score
    # ── LABELS (outcome) ──
    views           = Column(Integer, default=0)
    likes           = Column(Integer, default=0)
    comments        = Column(Integer, default=0)
    hours_since_publish = Column(Float, default=0.0)
    views_per_hour  = Column(Float, default=0.0)             # derived label
    ctr             = Column(Float, nullable=True)           # REAL thumbnail CTR (Reporting API v1)
    impressions     = Column(BigInteger, nullable=True)      # thumbnail impressions (Reporting API v1)
    explored_hook   = Column(String(12), nullable=True)      # A/B ledger: hook form probed (or NULL)
    # ── bookkeeping ──
    first_seen_at   = Column(DateTime(timezone=True), server_default=func.now())
    captured_at     = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class SeoLearningSnapshot(Base):
    """Point-in-time per-channel SEO learning state (learning/seo_learning.py).

    One row per (channel, window) per refresh: measured hook/script/length
    performance buckets + top keywords + the derived POLICY the generator
    consumes. Append-only so the Insights UI can graph how learning evolves
    week over week — a real curve, not a mock."""
    __tablename__ = "seo_learning_snapshots"

    id           = Column(Integer, primary_key=True)
    channel_id   = Column(Integer, index=True, nullable=False)
    # 'own' = channel_id is a models.Channel id (the user's channel);
    # 'competitor' = channel_id is a models.CompetitorChannel id. Keeps
    # both intelligences in one store without id collisions.
    kind         = Column(String(12), nullable=False, default="own")
    window_days  = Column(Integer, nullable=False, default=30)
    samples      = Column(Integer, nullable=False, default=0)
    payload      = Column(JSON, nullable=False, default=dict)
    computed_at  = Column(DateTime(timezone=True), server_default=func.now(), index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Phase C — Thumbnail A/B variants
# ─────────────────────────────────────────────────────────────────────────────

class ThumbnailVariant(Base):
    __tablename__ = "thumbnail_variants"

    id             = Column(Integer, primary_key=True, index=True)
    upload_job_id  = Column(Integer, ForeignKey("upload_jobs.id"), nullable=False, index=True)
    variant_idx    = Column(Integer, default=0)         # 0 = primary, 1+ = alternates
    image_path     = Column(String(500), default="")
    hook_text      = Column(String(255), default="")
    status         = Column(String(20), default="pending")
    # pending | served (applied to YT) | swapped_in | swapped_out | skipped
    served_at      = Column(DateTime(timezone=True), nullable=True)
    swapped_at     = Column(DateTime(timezone=True), nullable=True)
    views_at_swap  = Column(Integer, default=0)
    created_at     = Column(DateTime(timezone=True), server_default=func.now())


# ─────────────────────────────────────────────────────────────────────────────
# Phase D — Multi-language rebroadcast (SEO-level translation v1)
# ─────────────────────────────────────────────────────────────────────────────

class ClipTranslation(Base):
    """Translated SEO payload for a clip in a target language. Mirrors the
    shape of Clip.seo but in another language."""
    __tablename__ = "clip_translations"

    id          = Column(Integer, primary_key=True, index=True)
    clip_id     = Column(Integer, ForeignKey("clips.id"), nullable=False, index=True)
    language    = Column(String(10), nullable=False)       # ISO 639-1
    payload     = Column(JSON, default=dict)               # {title, description, tags, hashtags, hook}
    created_at  = Column(DateTime(timezone=True), server_default=func.now())
    updated_at  = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (UniqueConstraint("clip_id", "language", name="uq_clip_lang"),)


# ─────────────────────────────────────────────────────────────────────────────
# Phase E — Trending-topic radar
# ─────────────────────────────────────────────────────────────────────────────

class ChannelVideo(Base):
    """Cached catalogue of every video on a connected YouTube channel.

    The poller in analytics/poller.py only samples uploads Kaizer
    itself published. This table holds the channel's ENTIRE upload
    history (fetched from YouTube Data API and cached) so the
    Performance page can compute percentiles, surface "compare to
    your other videos" views, and let the user pick any of their
    own videos — not just the ones Kaizer created.

    Keyed by (user_id, google_channel_id, video_id) because the same
    YouTube video id is globally unique but per-user ownership
    matters for tenant isolation.
    """
    __tablename__ = "channel_videos"
    __table_args__ = (
        UniqueConstraint("user_id", "google_channel_id", "video_id",
                          name="uq_channel_video"),
        Index("ix_channel_videos_gcid", "user_id", "google_channel_id"),
    )

    id                 = Column(Integer, primary_key=True, index=True)
    user_id            = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    google_channel_id  = Column(String(50), nullable=False)
    video_id           = Column(String(50), nullable=False)
    title              = Column(Text, default="")
    # Snippet description trimmed to ~500 chars — full text isn't
    # useful for the dashboard and wastes row size on long copy.
    description_short  = Column(Text, default="")
    published_at       = Column(DateTime(timezone=True), nullable=True, index=True)
    duration_seconds   = Column(Integer, default=0)
    view_count         = Column(BigInteger, default=0, index=True)
    like_count         = Column(Integer, default=0)
    comment_count      = Column(Integer, default=0)
    thumbnail_url      = Column(String(500), default="")
    # Public video tags from the Data API — the raw material for
    # competitor tag harvesting (and our own tag-performance learning).
    tags               = Column(JSON, nullable=True)
    last_synced_at     = Column(DateTime(timezone=True), server_default=func.now(),
                                 onupdate=func.now())


class AnalyticsAiReport(Base):
    """Cached AI-generated analytics report (the "AI Coach").

    One row per (user, scope) generation — ``scope`` is a
    google_channel_id for a single-channel report or the literal
    ``"all"`` for the cross-channel overview. The payload is the
    SCHEMA-VALIDATED JSON the LLM returned (never raw model output),
    so re-rendering a cached report can never inject anything the
    validator didn't approve. Reports are reused while fresh to keep
    LLM spend and abuse surface low.
    """
    __tablename__ = "analytics_ai_reports"
    __table_args__ = (
        Index("ix_ai_reports_user_scope", "user_id", "scope"),
    )

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    scope      = Column(String(64), nullable=False, default="all")
    provider   = Column(String(16), default="")          # "gemini" | "claude"
    payload    = Column(JSON, default=dict)              # validated AiReport JSON
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PostizIntegration(Base):
    """Per-user ownership of a social channel connected through the
    business's SINGLE Postiz org (the env POSTIZ_API_KEY).

    Postiz itself has no notion of our users — every channel a user
    connects lands in that one org. This table is how Kaizer enforces
    per-user isolation: a user may only see / bind / publish-to /
    disconnect the integrations THEY connected. ``integration_id`` is
    Postiz's GET /public/v1/integrations id and is GLOBALLY UNIQUE here
    (one Postiz channel belongs to exactly one Kaizer user), so two users
    can never claim the same channel.

    Binding a Kaizer channel to one of these for delivery is OPT-IN — by
    default every channel uploads natively (our own YouTube quota). In
    'production' delivery mode it also serves as the auto-fallback target
    when the YouTube quota is exhausted.
    """
    __tablename__ = "postiz_integrations"
    __table_args__ = (
        UniqueConstraint("integration_id", name="uq_postiz_integration_id"),
        Index("ix_postiz_int_user", "user_id"),
    )

    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    integration_id = Column(String(64), nullable=False)
    provider       = Column(String(40), default="")     # instagram | facebook | youtube | x | …
    name           = Column(String(255), default="")    # Postiz display label
    identifier     = Column(String(255), default="")    # @handle / platform id
    picture        = Column(String(500), default="")    # avatar URL (optional)
    created_at     = Column(DateTime(timezone=True), server_default=func.now())


class PostizConnectSession(Base):
    """Short-lived pre-connect snapshot for the Postiz connect→finalize
    handshake — DB-backed (NOT process memory) so it survives across uvicorn
    workers. Connects are serialized only ACROSS teams (the router blocks a
    DIFFERENT team mid-connecting the same provider), so teammates can connect
    the same platform concurrently while a different team is asked to retry —
    keeping cross-team attribution unambiguous. Rows are pruned by TTL on the
    next connect.
    """
    __tablename__ = "postiz_connect_sessions"
    __table_args__ = (
        Index("ix_postiz_connect_provider", "provider", "created_ts"),
    )

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    provider   = Column(String(40), nullable=False)
    before_ids = Column(JSON, default=list)   # org integration ids before this connect
    created_ts = Column(Float, nullable=False)  # time.time() at connect (TTL clock)


class UserAsset(Base):
    """Images / logos a user uploads to reuse across clips.

    Marking one as `is_default_ad` makes the pipeline use that image instead
    of fetching a fresh stock photo (Pexels) or Gemini-generated card when
    the user enables "Use my default image" on a job.
    """
    __tablename__ = "user_assets"

    id            = Column(Integer, primary_key=True, index=True)
    user_id       = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    filename      = Column(String(255), default="")
    file_path     = Column(String(500), nullable=False)     # absolute path on disk
    thumb_path    = Column(String(500), default="")         # optional downscaled jpg
    kind          = Column(String(32), default="image")     # image | logo | background
    mime          = Column(String(64), default="image/jpeg")
    size_bytes    = Column(Integer, default=0)
    width         = Column(Integer, default=0)
    height        = Column(Integer, default=0)
    is_default_ad = Column(Boolean, default=False, index=True)
    tags          = Column(JSON, default=list)              # ["news", "telugu", …]
    # Subject label ("name-tag contract"): one line describing WHAT/WHO the
    # image shows (e.g. "PM addressing parliament"). Stamped at upload
    # (user-typed or vision auto-caption) or at AI generation (from the
    # image prompt's beat subject). The image↔speech timing AI matches
    # images to spoken words by THIS label — never by pixels.
    description   = Column(Text, default="")
    # Virtual folder path — slash-separated like "logos/english/".  Empty
    # = Assets root.  No physical folders on disk; this is purely a UI
    # organization string the frontend groups by.
    folder_path   = Column(String(255), default="", index=True)
    # Optional fingerprint of the SOURCE VIDEO this asset was generated
    # from (same hash function gemini_cache.hash_file_prefix uses — first
    # 4 MiB + size, mtime EXCLUDED).  Set on bulletin OpenAI images by
    # the runner so re-uploads of the same source can offer "reuse the
    # images we already generated for this video" instead of paying for
    # gpt-image-1 again.  Empty = "no source-video link" (e.g. user-
    # uploaded logos, manually-uploaded backgrounds).
    source_video_hash = Column(String(64), default="", index=True)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())

    # ── Storage (Phase 5) ──────────────────────────────────────────
    # When storage_backend='local' the URL is /media/<key>; when 'r2' the URL
    # is a CDN/public link or a signed URL (frontend re-fetches if expired).
    # file_path remains populated for backwards compatibility.
    storage_url       = Column(String(500), default="")
    storage_key       = Column(String(500), default="")
    storage_backend   = Column(String(20),  default="")
    # Optional: separate R2 URL for the generated thumbnail. If empty,
    # _to_dict falls back to storage_url so legacy rows still render.
    thumb_storage_url = Column(String(500), default="")


class LibraryItem(Base):
    """Shared "company library" of source videos that any logged-in user
    can pick from when starting a new job. Uploaded exclusively by
    ``is_creative`` or ``is_admin`` users — normal users can browse +
    "Use" only.

    Single source of truth = Cloudflare R2. Both local-dev and prod
    instances read/write the same bucket so a creative uploads once
    and every user (regardless of which deployment they're on) sees it.
    The video + thumbnail live at:

        library/<id>/video.<ext>
        library/<id>/thumb.jpg

    ``deleted_at`` is soft-delete only — the row stays for audit and
    the R2 object stays for 30 days before a separate sweep deletes it.
    """
    __tablename__ = "library_items"

    id            = Column(Integer, primary_key=True, index=True)
    # Creative user who uploaded it. ForeignKey is nullable so deleting
    # a creator account doesn't cascade-wipe the library.
    uploader_id   = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"),
                            nullable=True, index=True)
    # User-supplied title. Falls back to original filename when blank.
    title         = Column(String(200), nullable=False, default="")
    description   = Column(Text, default="")
    # R2 object keys (relative to bucket root; the prefix env var is
    # applied transparently by R2Storage._k). Empty until upload finishes.
    video_key     = Column(String(500), nullable=False, default="")
    thumb_key     = Column(String(500), default="")
    # Watermarked copy for free-tier downloads. Generated lazily on the
    # first free-user download request and cached on R2 — paid users
    # never trigger this render. Empty = not yet generated.
    watermark_key = Column(String(500), default="")
    # Cached metadata so the listing endpoint doesn't probe ffprobe on
    # every fetch. Populated at upload time.
    duration_secs = Column(Float, default=0)
    file_size     = Column(Integer, default=0)
    width         = Column(Integer, default=0)
    height        = Column(Integer, default=0)
    # Original filename — surfaced to the user when title is blank.
    original_name = Column(String(255), default="")
    # Tracks how many jobs have been kicked off from this item — useful
    # signal for creatives to see what's getting used.
    use_count     = Column(Integer, default=0, nullable=False)
    # Watch count — incremented on every play-ticket request (the call
    # the <video> tag makes before streaming). Shown on the card so
    # creators see what's popular.
    watch_count   = Column(Integer, default=0, nullable=False)
    # Optional FK to a LibraryCategory. ondelete=SET NULL so deleting a
    # category never touches the videos — they just become uncategorized.
    # category_id is what the row is bound to (NOT the name) so renames
    # are free.
    category_id   = Column(Integer,
                            ForeignKey("library_categories.id", ondelete="SET NULL"),
                            nullable=True, index=True)
    # Cached aggregate of per-video ratings. Sum + count denormalized
    # here so the list endpoint never has to GROUP BY ratings.
    rating_sum    = Column(Integer, default=0, nullable=False)
    rating_count  = Column(Integer, default=0, nullable=False)
    created_at    = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    deleted_at    = Column(DateTime(timezone=True), nullable=True)


class LibraryCategory(Base):
    """Operator-managed category taxonomy for the company library.

    Videos reference categories by id (NEVER name) so admins can rename
    a category and every existing video keeps its binding. Soft-delete
    means a deleted category just stops appearing in pickers; videos
    that used it have their ``category_id`` set NULL via the FK's
    ``ondelete='SET NULL'`` rule — no orphans, no data loss.
    """
    __tablename__ = "library_categories"

    id          = Column(Integer, primary_key=True, index=True)
    name        = Column(String(80), nullable=False, default="")
    # Optional accent colour for chips. Free-form CSS string ("#c0392b"
    # or named token).
    color       = Column(String(20), default="")
    sort_order  = Column(Integer, default=0, nullable=False)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())
    deleted_at  = Column(DateTime(timezone=True), nullable=True)


class LibraryItemRating(Base):
    """One rating per (item, user). Upsert pattern — re-rating updates
    the same row. The library item's cached ``rating_sum`` /
    ``rating_count`` get adjusted by the delta at upsert time."""
    __tablename__ = "library_item_ratings"
    __table_args__ = (
        UniqueConstraint("item_id", "user_id", name="uq_libitem_rating_user"),
    )

    id         = Column(Integer, primary_key=True, index=True)
    item_id    = Column(Integer,
                          ForeignKey("library_items.id", ondelete="CASCADE"),
                          nullable=False, index=True)
    user_id    = Column(Integer,
                          ForeignKey("users.id", ondelete="CASCADE"),
                          nullable=False, index=True)
    stars      = Column(Integer, nullable=False, default=5)   # 1..5
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(),
                          onupdate=func.now())


class CreatorRating(Base):
    """One rating per (creator, rater) — same upsert pattern as
    LibraryItemRating. Updates the User row's cached creator
    aggregate."""
    __tablename__ = "creator_ratings"
    __table_args__ = (
        UniqueConstraint("creator_id", "rater_id", name="uq_creator_rating_pair"),
    )

    id         = Column(Integer, primary_key=True, index=True)
    creator_id = Column(Integer,
                          ForeignKey("users.id", ondelete="CASCADE"),
                          nullable=False, index=True)
    rater_id   = Column(Integer,
                          ForeignKey("users.id", ondelete="CASCADE"),
                          nullable=False, index=True)
    stars      = Column(Integer, nullable=False, default=5)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(),
                          onupdate=func.now())


class CompetitorChannel(Base):
    """YouTube channels we monitor for topic intelligence."""
    __tablename__ = "competitor_channels"
    __table_args__ = (
        UniqueConstraint("user_id", "youtube_channel_id", name="uq_competitor_user_ytid"),
    )

    id                 = Column(Integer, primary_key=True, index=True)
    user_id            = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    name               = Column(String(255), nullable=False)
    handle             = Column(String(100), default="")
    # Global uniqueness dropped — two users can both track @TV9TeluguLive
    youtube_channel_id = Column(String(50), nullable=False, index=True)
    language           = Column(String(10), default="te")
    active             = Column(Boolean, default=True)
    created_at         = Column(DateTime(timezone=True), server_default=func.now())


class TrendingTopic(Base):
    __tablename__ = "trending_topics"

    id                 = Column(Integer, primary_key=True, index=True)
    source_channel_id  = Column(Integer, ForeignKey("competitor_channels.id"), nullable=False, index=True)
    video_id           = Column(String(50), nullable=False, index=True)
    video_title        = Column(Text, default="")
    video_url          = Column(Text, default="")
    published_at       = Column(DateTime(timezone=True), nullable=True)
    view_count         = Column(Integer, default=0)
    topic_summary      = Column(Text, default="")
    keywords           = Column(JSON, default=list)
    urgency            = Column(String(20), default="normal")   # hot | normal | low
    used_for_job_id    = Column(Integer, ForeignKey("jobs.id"), nullable=True)
    fetched_at         = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (UniqueConstraint("source_channel_id", "video_id", name="uq_competitor_video"),)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 / Wave 3 — Training Flywheel, Creator Graph, Agency Mode,
#                    Regional API
# ─────────────────────────────────────────────────────────────────────────────

class TrainingRecord(Base):
    """One labeled data point for the narrative scorer flywheel."""
    __tablename__ = "training_records"
    __table_args__ = (
        UniqueConstraint("upload_job_id", name="uq_training_upload"),
        Index("ix_training_niche", "niche"),
        Index("ix_training_collected_at", "collected_at"),
    )
    id                 = Column(Integer, primary_key=True, index=True)
    upload_job_id      = Column(Integer, ForeignKey("upload_jobs.id", ondelete="CASCADE"), nullable=False)
    clip_id            = Column(Integer, ForeignKey("clips.id", ondelete="CASCADE"), nullable=False, index=True)
    niche              = Column(String(50), default="")
    narrative_role     = Column(String(32), default="")
    hook_score         = Column(Float,   default=0.0)
    completion_score   = Column(Float,   default=0.0)
    composite_score    = Column(Float,   default=0.0)
    views_48h          = Column(Integer, default=0)
    retention_curve    = Column(JSON,    default=list)
    shares_per_reach   = Column(Float,   default=0.0)
    video_hash         = Column(String(64), default="")
    collected_at       = Column(DateTime(timezone=True), server_default=func.now())


class ClipEdge(Base):
    """Typed edges between Clips. Enables series/variant/trailer graph queries."""
    __tablename__ = "clip_edges"
    __table_args__ = (
        UniqueConstraint("edge_type", "src_clip_id", "dst_clip_id", name="uq_clip_edge"),
        Index("ix_clip_edge_src", "src_clip_id"),
        Index("ix_clip_edge_dst", "dst_clip_id"),
        Index("ix_clip_edge_type", "edge_type"),
    )
    id            = Column(Integer, primary_key=True, index=True)
    edge_type     = Column(String(32), nullable=False)    # series_part_of | trailer_for | variant_of | reusable_source | narrative_beat_of
    src_clip_id   = Column(Integer, ForeignKey("clips.id", ondelete="CASCADE"), nullable=False)
    dst_clip_id   = Column(Integer, ForeignKey("clips.id", ondelete="CASCADE"), nullable=False)
    edge_metadata = Column(JSON, default=dict)   # NOTE: named edge_metadata not 'meta' to avoid SQLAlchemy reserved-word clash
    created_at    = Column(DateTime(timezone=True), server_default=func.now())


class AgencyTeam(Base):
    """Agency workspace — multi-account billing + RBAC."""
    __tablename__ = "agency_teams"
    __table_args__ = (UniqueConstraint("owner_user_id", "name", name="uq_agency_owner_name"),)
    id                 = Column(Integer, primary_key=True, index=True)
    owner_user_id      = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name               = Column(String(255), nullable=False)
    branding           = Column(JSON, default=dict)           # {logo_url, accent_color, domain, ...}
    monthly_clip_cap   = Column(Integer, default=0)
    created_at         = Column(DateTime(timezone=True), server_default=func.now())
    updated_at         = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class AgencyMember(Base):
    """User↔Agency membership with a role (owner/admin/creator/viewer)."""
    __tablename__ = "agency_members"
    __table_args__ = (UniqueConstraint("agency_id", "user_id", name="uq_agency_member"),)
    id         = Column(Integer, primary_key=True, index=True)
    agency_id  = Column(Integer, ForeignKey("agency_teams.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    role       = Column(String(16), nullable=False, default="creator")   # owner | admin | creator | viewer
    added_at   = Column(DateTime(timezone=True), server_default=func.now())


class AgencyAuditLog(Base):
    """Who-did-what log inside an agency (immutable, insert-only)."""
    __tablename__ = "agency_audit_log"
    __table_args__ = (
        Index("ix_agency_audit_agency", "agency_id"),
        Index("ix_agency_audit_ts", "timestamp"),
    )
    id              = Column(Integer, primary_key=True, index=True)
    agency_id       = Column(Integer, ForeignKey("agency_teams.id", ondelete="CASCADE"), nullable=False)
    actor_user_id   = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action          = Column(String(64), nullable=False)      # clip.create / billing.view / team.invite / ...
    target_kind     = Column(String(32), default="")          # clip | user | asset | upload_job | ...
    target_id       = Column(Integer, default=0)
    timestamp       = Column(DateTime(timezone=True), server_default=func.now())
    details         = Column(JSON, default=dict)


class RegionalApiKey(Base):
    """Partner API keys for the B2B newsroom /api/regional/* endpoints."""
    __tablename__ = "regional_api_keys"
    __table_args__ = (UniqueConstraint("org_id", "api_key_hash", name="uq_regional_org_key"),)
    id              = Column(Integer, primary_key=True, index=True)
    org_id          = Column(String(64), nullable=False, index=True)
    api_key_hash    = Column(String(64), nullable=False, index=True)    # SHA-256 hex of the raw key
    label           = Column(String(120), default="")
    rate_limit_rpm  = Column(Integer, default=60)
    monthly_cap     = Column(Integer, default=1000)
    active          = Column(Boolean, default=True)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 — Autonomous Live Director
# ─────────────────────────────────────────────────────────────────────────────


class LiveEvent(Base):
    """A live production event managed by the Autonomous Live Director.

    ``status``     : scheduled | live | ended | failed
    ``config_json``: per-event director config (min_shot_s, max_shot_s,
                     reaction_threshold, speaker_vad_hold_ms, …).
    ``rtmp_key_hash``: SHA-256 of the ingest auth key (never stored in plain text).
    ``program_url``  : Public URL of the live program output (HLS / RTMP).
    """
    __tablename__ = "live_events"

    id            = Column(Integer, primary_key=True, index=True)
    user_id       = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name          = Column(String(255), nullable=False)
    venue         = Column(String(255), default="")
    starts_at     = Column(DateTime(timezone=True), server_default=func.now())
    ends_at       = Column(DateTime(timezone=True), nullable=True)
    status        = Column(String(20), default="scheduled", index=True)   # scheduled | live | ended | failed
    config_json   = Column(JSON, default=dict)
    rtmp_key_hash = Column(String(64), default="")
    program_url   = Column(String(500), default="")
    created_at    = Column(DateTime(timezone=True), server_default=func.now())
    updated_at    = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ─────────────────────────────────────────────────────────────────────────────
# Phase 12 — Admin panel: Gemini call accounting
#
# Raw prompts + responses are NOT stored on this row (privacy + DB bloat).
# Only the metadata needed for per-user quota tracking, cost estimates and
# the admin analytics dashboard.  In dev (SQLite) Base.metadata.create_all()
# in main.py creates this table automatically on startup.  In prod (Postgres)
# ops runs the DDL in docs/MIGRATIONS.md manually.
# ─────────────────────────────────────────────────────────────────────────────


class GeminiCall(Base):
    """One row per call to a Gemini model — summed by the admin panel for
    cost + quota reporting."""
    __tablename__ = "gemini_calls"

    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    job_id         = Column(Integer, ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True, index=True)
    clip_id        = Column(Integer, ForeignKey("clips.id", ondelete="SET NULL"), nullable=True, index=True)
    model          = Column(String(64), nullable=False)         # e.g. "gemini-2.0-flash-exp"
    purpose        = Column(String(64), default="")             # "seo" | "script" | "style-classify" | "thumbnail" | ...
    prompt_tokens  = Column(Integer, default=0)
    output_tokens  = Column(Integer, default=0)
    total_tokens   = Column(Integer, default=0)
    # Video accounting — Gemini bills video uploads per-second in addition to
    # tokens, so both file size AND duration need to be tracked to estimate
    # real cost per clip and price subscription tiers fairly.
    file_bytes       = Column(Integer, default=0)                 # total bytes uploaded (all file parts)
    video_duration_s = Column(Float,   default=0.0)               # total seconds of video uploaded
    cost_usd       = Column(Float,   default=0.0)               # estimated from COST_PER_1K_TOKENS
    latency_ms     = Column(Integer, default=0)
    status         = Column(String(16), default="ok")           # ok | error | rate_limited
    error          = Column(Text,    default="")
    created_at     = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class OpenAiCall(Base):
    """One row per OpenAI API call (currently only gpt-image-1 for bulletin
    images — extend ``purpose`` when we add chat-completion or whisper).

    Mirrors ``gemini_calls``' design so the admin Usage page can show
    Gemini + OpenAI spend side-by-side without bespoke aggregation.
    """
    __tablename__ = "openai_calls"

    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    job_id         = Column(Integer, ForeignKey("jobs.id", ondelete="SET NULL"),  nullable=True, index=True)
    clip_id        = Column(Integer, ForeignKey("clips.id", ondelete="SET NULL"), nullable=True, index=True)
    model          = Column(String(64),  nullable=False)   # gpt-image-1 / gpt-4o / whisper-1 / ...
    purpose        = Column(String(64),  default="")       # bulletin-image / thumbnail / chat / transcribe
    # Image-gen-specific (NULL for non-image calls).
    image_size     = Column(String(20),  default="")       # 1024x1024 / 1024x1536 / 1536x1024
    image_quality  = Column(String(10),  default="")       # low / medium / high / auto
    image_count    = Column(Integer,     default=0)        # how many images returned in this single call
    # Text/audio-call-specific (NULL for image calls).
    prompt_tokens  = Column(Integer,     default=0)
    output_tokens  = Column(Integer,     default=0)
    total_tokens   = Column(Integer,     default=0)
    cost_usd       = Column(Float,       default=0.0)      # best-effort from a hardcoded price table
    latency_ms     = Column(Integer,     default=0)
    status         = Column(String(16),  default="ok")     # ok / error / rate_limited
    error          = Column(Text,        default="")
    created_at     = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class AnthropicCall(Base):
    """One row per Claude/Anthropic API call (cut-planning, SEO, thumbnail
    direction, Express). Mirrors ``gemini_calls`` / ``openai_calls`` so the
    admin Usage page sums Gemini + OpenAI + Claude spend side-by-side.

    Token columns use the SAME names as the other two tables
    (``prompt_tokens``/``output_tokens``/``total_tokens``) so the admin
    aggregation stays uniform; Anthropic's ``input_tokens`` maps to
    ``prompt_tokens``. Cache columns are Anthropic-specific (prompt caching)
    and default to 0 for non-cached calls.
    """
    __tablename__ = "anthropic_calls"

    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    job_id         = Column(Integer, ForeignKey("jobs.id", ondelete="SET NULL"),  nullable=True, index=True)
    clip_id        = Column(Integer, ForeignKey("clips.id", ondelete="SET NULL"), nullable=True, index=True)
    model          = Column(String(64),  nullable=False)   # claude-opus-4-7 / claude-sonnet-4-6 / ...
    purpose        = Column(String(64),  default="")       # cut-plan / seo / thumbnail / express / ...
    prompt_tokens  = Column(Integer,     default=0)        # = Anthropic input_tokens
    output_tokens  = Column(Integer,     default=0)
    total_tokens   = Column(Integer,     default=0)
    # Anthropic prompt-caching accounting (NULL/0 when caching not used).
    cache_read_tokens  = Column(Integer, default=0)
    cache_write_tokens = Column(Integer, default=0)
    cost_usd       = Column(Float,       default=0.0)      # best-effort from a hardcoded price table
    latency_ms     = Column(Integer,     default=0)
    status         = Column(String(16),  default="ok")     # ok / error / rate_limited
    error          = Column(Text,        default="")
    created_at     = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class YouTubeApiCall(Base):
    """One row per YouTube Data API call (mostly uploads + thumbnail sets +
    metadata fetches).

    YouTube doesn't charge dollars — they bill in *quota units* against a
    daily cap (10,000 / day per Google Cloud project by default).  We log
    each call so the admin Usage page can answer:
      • "Which videos burned the most quota?"  → upload kind correlation
      • "What % of today's cap have we used?"  → operational alarm
      • "Which user is driving us toward the cap?"
    """
    __tablename__ = "youtube_api_calls"

    id            = Column(Integer, primary_key=True, index=True)
    user_id       = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    job_id        = Column(Integer, ForeignKey("jobs.id", ondelete="SET NULL"),  nullable=True, index=True)
    clip_id       = Column(Integer, ForeignKey("clips.id", ondelete="SET NULL"), nullable=True, index=True)
    upload_job_id = Column(Integer, ForeignKey("upload_jobs.id", ondelete="SET NULL"), nullable=True, index=True)
    channel_id    = Column(Integer, ForeignKey("channels.id", ondelete="SET NULL"), nullable=True, index=True)
    # The actual YouTube channel id (UCxxxxx) that the call targeted.
    # Stored alongside the local FK so we can group by destination even
    # after the Channel row is deleted.
    google_channel_id = Column(String(50), default="")
    video_id      = Column(String(50), default="")        # YT video id (set after upload)
    operation     = Column(String(64), nullable=False)    # videos.insert / thumbnails.set / videos.list / channels.list / search.list
    quota_cost    = Column(Integer, default=0)            # cost in YouTube quota units for THIS call
    publish_kind  = Column(String(10), default="")        # short / video — only for uploads
    file_bytes    = Column(BigInteger, default=0)         # only for videos.insert
    duration_seconds = Column(Float,   default=0.0)       # only for videos.insert
    success       = Column(Boolean, default=True)
    http_status   = Column(Integer, default=0)            # 200 / 403 (quotaExceeded) / 401 / 5xx
    error         = Column(Text,    default="")
    created_at    = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class LiveCamera(Base):
    """One camera feed registered for a live event.

    ``cam_id``    : Short machine-readable identifier matching the RTMP path.
    ``mic_id``    : Optional linked microphone feed ID.
    ``role_hints``: List of tags that bias director rules.
    ``iso_url``   : R2 URL of the ISO recording for post-event edit.
    """
    __tablename__ = "live_cameras"
    __table_args__ = (UniqueConstraint("event_id", "cam_id", name="uq_live_camera"),)

    id          = Column(Integer, primary_key=True, index=True)
    event_id    = Column(Integer, ForeignKey("live_events.id", ondelete="CASCADE"), nullable=False, index=True)
    cam_id      = Column(String(64), nullable=False)
    label       = Column(String(255), default="")
    mic_id      = Column(String(64), default="")
    role_hints  = Column(JSON, default=list)
    iso_url     = Column(String(500), default="")
    created_at  = Column(DateTime(timezone=True), server_default=func.now())


class DirectorLogEntry(Base):
    """Immutable log of every director decision during a live event.

    Queryable by ``(event_id, t)`` for the control-surface log view and for
    post-event cue-sheet generation.

    ``kind``      : selection | override | camera_lost | health
    ``cam_id``    : Camera involved (may be empty for health events).
    ``confidence``: 0.0–1.0 score from the director rule that fired.
    ``reason``    : Human-readable explanation string.
    ``payload``   : Full structured payload (JSON).
    """
    __tablename__ = "director_log"
    __table_args__ = (Index("ix_director_log_event_t", "event_id", "t"),)

    id          = Column(Integer, primary_key=True, index=True)
    event_id    = Column(Integer, ForeignKey("live_events.id", ondelete="CASCADE"), nullable=False)
    t           = Column(Float, nullable=False)
    kind        = Column(String(32), default="selection")
    cam_id      = Column(String(64), default="")
    confidence  = Column(Float, default=0.0)
    reason      = Column(Text, default="")
    payload     = Column(JSON, default=dict)


class SystemSetting(Base):
    """Singleton-style system-wide config persisted in DB.

    Used for admin-controlled toggles that apply to ALL users — e.g.
    `upload_provider` ('postiz' vs 'kaizer'), feature flags, etc. Each
    row is one key/value pair so we can add new settings without
    migrations: just call `set_system_setting(db, key, value)`.
    Read-side helper `get_system_setting(db, key, default)` is in
    routers/admin.py (or wherever it lives) — never read directly so
    the default is consistent.
    """
    __tablename__ = "system_settings"

    key        = Column(String(64), primary_key=True)
    value      = Column(String(512), nullable=False, default="")
    updated_at = Column(DateTime(timezone=True),
                        server_default=func.now(),
                        onupdate=func.now())


class SystemMetric(Base):
    """Persistent CPU / RAM / GPU / disk sample.

    A daemon thread in main.py writes one row every ~30 s; old rows are pruned
    after 14 days. The admin Capacity tab reads this to size the future
    cloud deployment ("p95 CPU at peak", "max GPU VRAM during compose", etc.).

    All fields are nullable so a missing GPU or a transient psutil error
    doesn't blow up the whole sample.
    """
    __tablename__ = "system_metrics"

    id              = Column(Integer, primary_key=True, index=True)
    ts              = Column(DateTime(timezone=True), server_default=func.now(),
                             nullable=False, index=True)
    cpu_percent     = Column(Float,   nullable=True)
    cpu_count       = Column(Integer, nullable=True)
    ram_percent     = Column(Float,   nullable=True)
    ram_used_gb     = Column(Float,   nullable=True)
    ram_total_gb    = Column(Float,   nullable=True)
    disk_percent    = Column(Float,   nullable=True)
    disk_used_gb    = Column(Float,   nullable=True)
    disk_total_gb   = Column(Float,   nullable=True)
    gpu_util        = Column(Float,   nullable=True)   # 0..100, null if no GPU
    gpu_mem_used_mb = Column(Integer, nullable=True)
    gpu_mem_total_mb= Column(Integer, nullable=True)
    gpu_temp_c      = Column(Integer, nullable=True)
    proc_rss_gb     = Column(Float,   nullable=True)   # uvicorn worker RSS
    proc_threads    = Column(Integer, nullable=True)
    live_events     = Column(Integer, nullable=True)   # in-process live sessions
    # Network deltas — bytes since previous sample. Null on the first sample.
    net_rx_bps      = Column(BigInteger, nullable=True)
    net_tx_bps      = Column(BigInteger, nullable=True)

    # ── "Kaizer-only" rollup ────────────────────────────────────────
    # Sum across the whole Kaizer process family: uvicorn + all its
    # descendants (pipeline subprocesses, ffmpeg) + vite + cloudflared
    # + the Redis container's docker-proxy process. This is the number
    # to size the future cloud server against — the rest of the machine
    # load belongs to Chrome / VS Code / etc and is irrelevant.
    kaizer_cpu_percent  = Column(Float,   nullable=True)
    kaizer_rss_gb       = Column(Float,   nullable=True)
    kaizer_proc_count   = Column(Integer, nullable=True)
    kaizer_ffmpeg_count = Column(Integer, nullable=True)   # how many ffmpeg's running = active encode stages
    kaizer_gpu_util     = Column(Float,   nullable=True)   # best-effort, via nvidia-smi pmon


class LoginCode(Base):
    """A one-shot six-digit code emailed to sign somebody in.

    Shaped after ``PasswordResetToken`` and for the same reasons: only the
    HASH is stored, so a leaked snapshot cannot be replayed; ``used_at``
    makes it one-shot; a separate table keeps every request auditable.

    ``attempts`` is the one addition. A reset link is 32 random bytes and
    nobody guesses it, but six digits is a million possibilities -- ample
    until you notice nothing stopped an attacker making a million guesses.
    Five wrong tries and the row is dead.

    ``email`` is stored beside ``user_id`` because the account may not exist
    yet: the first successful code CREATES it, so there is no separate
    sign-up to get wrong.
    """
    __tablename__ = "login_codes"

    id          = Column(Integer, primary_key=True, index=True)
    email       = Column(String(320), nullable=False, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    code_hash   = Column(String(64), nullable=False, index=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())
    expires_at  = Column(DateTime(timezone=True), nullable=False)
    used_at     = Column(DateTime(timezone=True), nullable=True)
    attempts    = Column(Integer, nullable=False, default=0)
    requested_ip = Column(String(64), nullable=True)


class OnboardingProfile(Base):
    """What we ask for once, on a new account's first sign-in.

    Its own table rather than columns on ``users``: create_all brings a
    missing TABLE into being on any deployment, whereas a missing COLUMN
    needs a migration somebody has to remember to write.

    Every field except ``user_id`` is nullable at the database level and
    required by the API instead. That is deliberate -- it lets the startup
    backfill write a sparse ``legacy`` row for each pre-existing account
    without inventing values for them, while new submissions are still
    validated in full. Whether the form was filled is "does a row exist",
    which is a question with no edge cases.
    """
    __tablename__ = "onboarding_profiles"

    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, ForeignKey("users.id"), nullable=False,
                          unique=True, index=True)
    full_name    = Column(String(160), nullable=True)
    mobile       = Column(String(32),  nullable=True)
    company_name = Column(String(200), nullable=True)   # company OR channel name
    email        = Column(String(320), nullable=True)
    website      = Column(String(500), nullable=True)   # the one optional field
    languages    = Column(String(200), nullable=True)   # comma-separated codes
    channel_link = Column(String(500), nullable=True)
    #: Resolved from YouTube when the channel could be confirmed. NULL means
    #: "not verified" -- a missing API key, a spent quota or a legacy /c/ URL
    #: with no cheap resolver -- never "the channel is fake"; a channel proven
    #: not to exist is refused at the door and no row is written.
    channel_id    = Column(String(64),  nullable=True)
    channel_title = Column(String(200), nullable=True)
    # "form"   filled in by the person
    # "legacy" written by the backfill for an account that predates this
    source       = Column(String(12), nullable=False, default="form")
    created_at   = Column(DateTime(timezone=True), server_default=func.now())
    updated_at   = Column(DateTime(timezone=True), server_default=func.now(),
                          onupdate=func.now())


class PasswordResetToken(Base):
    """One-shot, time-limited reset token issued by /auth/forgot.

    The token string is what we email / log; only its sha256 lives in the
    DB so a leaked DB snapshot can't be used to impersonate. ``used_at``
    is set the moment the token is consumed — re-use is a 400.

    Why a separate table (not a column on ``users``): a user can request
    multiple resets back-to-back (mistyped email, link lost), and we want
    each request audit-able. Tokens expire 30 min after issue; the
    cleanup runs lazily on the next /auth/forgot call.
    """
    __tablename__ = "password_reset_tokens"

    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    token_hash  = Column(String(64), nullable=False, unique=True, index=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())
    expires_at  = Column(DateTime(timezone=True), nullable=False)
    used_at     = Column(DateTime(timezone=True), nullable=True)
    requested_ip = Column(String(64), nullable=True)


class ExpressJob(Base):
    """Persistent record of an Express Mode autopub job.

    Mirrors the in-memory ``express/state.py`` map but survives backend
    restarts so the History panel can show jobs older than uptime.
    State is written transactionally on every step transition (queued
    → running → done | failed). Logs are kept short (last 500 lines)
    so a misbehaving 30-min pipeline doesn't bloat the DB.

    No FK to a "jobs" table — this is a separate flow from the main
    Kaizer pipeline. Only the user_id link matters for tenancy.
    """
    __tablename__ = "express_jobs"

    id           = Column(String(32), primary_key=True)   # secrets.token_urlsafe id
    user_id      = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    status       = Column(String(16), nullable=False, default="queued")   # queued|running|done|failed
    mode         = Column(String(32), nullable=True)      # publish-as-is|ai-trim|shorts
    step         = Column(String(32), nullable=True)
    progress     = Column(Integer, nullable=False, default=0)
    message      = Column(String(512), nullable=True)
    title        = Column(String(255), nullable=True)     # populated from results when done
    log_json     = Column(Text, nullable=True)            # JSON-encoded list[str] of recent lines
    results_json = Column(Text, nullable=True)            # JSON-encoded results dict (videos, postiz refs)
    error        = Column(Text, nullable=True)
    created_at   = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at   = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class LiveBatch(Base):
    """One submission of the Live Studio form.

    A user uploads N videos and picks M channels per video; we expand
    into N*M LiveStream rows (one ffmpeg push each). The batch holds
    the shared metadata (SEO source flag, AI-generated SEO, etc.) so
    individual streams don't duplicate it.

    The actual video file lives in the system temp dir during upload +
    broadcast, then optionally promoted to R2 for the backup/recovery
    path. We never store the full file in DB.
    """
    __tablename__ = "live_batches"

    id            = Column(Integer, primary_key=True, index=True)
    user_id       = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    # Public id used in URLs / WS topics — separate from PK so we never
    # leak the int counter to clients.
    public_id     = Column(String(32), nullable=False, unique=True, index=True)
    status        = Column(String(16), nullable=False, default="queued")
    #   queued | uploading | streaming | done | failed | canceled
    message       = Column(String(512), nullable=True)
    total_streams = Column(Integer, nullable=False, default=0)   # N*M
    streams_done  = Column(Integer, nullable=False, default=0)
    streams_failed= Column(Integer, nullable=False, default=0)
    created_at    = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at    = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class LiveStream(Base):
    """One (video × channel) broadcast.

    The granular unit. Each row corresponds to a single ffmpeg
    subprocess pushing one video file (possibly looped) to one
    YouTube channel for a configured duration in hours.

    File path lifecycle:
      - while uploading: ``upload_path`` is the growing temp file
      - while streaming: same path, ffmpeg reads from it
      - after broadcast: file is deleted from temp; R2 copy survives
        in ``backup_url`` if backup_enabled.

    Status flow:
      queued -> uploading -> streaming -> done | failed
      Special: ``canceled`` if the user kills it mid-broadcast.
    """
    __tablename__ = "live_streams"

    id            = Column(Integer, primary_key=True, index=True)
    batch_id      = Column(Integer, ForeignKey("live_batches.id"), nullable=False, index=True)
    user_id       = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    channel_id    = Column(Integer, ForeignKey("channels.id"), nullable=True, index=True)

    # Source video grouping — multiple rows can share the same video
    # (when one upload broadcasts to many channels). ``video_slot``
    # uniquely identifies the video within the batch.
    video_slot    = Column(Integer, nullable=False, default=0)

    status        = Column(String(16), nullable=False, default="queued")
    progress_pct  = Column(Integer, nullable=False, default=0)
    message       = Column(String(512), nullable=True)
    error         = Column(Text, nullable=True)

    # Upload tracking
    upload_path   = Column(String(512), nullable=True)        # local temp file
    upload_bytes  = Column(BigInteger, nullable=False, default=0)
    upload_total  = Column(BigInteger, nullable=True)         # known content-length, may be null
    upload_done   = Column(Boolean, nullable=False, default=False)

    # Broadcast config
    target_hours  = Column(Float, nullable=False, default=1.0)
    started_at    = Column(DateTime(timezone=True), nullable=True)
    finished_at   = Column(DateTime(timezone=True), nullable=True)

    # YouTube broadcast (minted via youtube/rtmp_provider.py)
    yt_broadcast_id = Column(String(64), nullable=True)
    yt_stream_id    = Column(String(64), nullable=True)
    yt_ingest_url   = Column(String(255), nullable=True)
    yt_stream_key   = Column(String(255), nullable=True)
    yt_video_id     = Column(String(64), nullable=True)

    # SEO — source flag distinguishes user-typed (trusted) from AI
    seo_source    = Column(String(16), nullable=False, default="user")  # user | ai
    title         = Column(String(255), nullable=True)
    description   = Column(Text, nullable=True)
    tags_json     = Column(Text, nullable=True)
    privacy       = Column(String(16), nullable=False, default="unlisted")
    made_for_kids = Column(Boolean, nullable=False, default=False)

    # R2 preview backup (48 h post-broadcast, then auto-deleted; UI
    # shows "Will be removed within 48 hrs"). YouTube is the durable
    # copy — this R2 file is just a quick-preview convenience. The
    # DB row itself survives forever for audit / history.
    backup_enabled    = Column(Boolean, nullable=False, default=True)
    backup_url        = Column(String(512), nullable=True)
    backup_key        = Column(String(255), nullable=True)
    backup_expires_at = Column(DateTime(timezone=True), nullable=True, index=True)

    # Optional per-video custom thumbnail. One image per video_slot —
    # all streams that share a slot point at the same file. The
    # orchestrator hands this to youtube.uploader.set_thumbnail() once
    # the broadcast is minted. NULL = let YouTube auto-pick a frame.
    thumbnail_path    = Column(String(512), nullable=True)

    # When the stream's source is a YouTube URL (yt-dlp ingested) rather
    # than a user-uploaded file, we record the original URL here for
    # history / audit. NULL = file-upload source.
    source_url        = Column(String(1024), nullable=True)

    # Quick Live branding conveyor: when True the orchestrator stamps the
    # channel's logo/watermark onto the source (pipeline_v4.watermark.
    # stamp_for_channel) BEFORE going live — status passes through
    # "branding" and the stamped temp file lands in branded_path (deleted
    # after the broadcast). False = stream the source as-is, instantly.
    apply_branding    = Column(Boolean, nullable=False, default=False)
    branded_path      = Column(String(512), nullable=True)

    created_at    = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at    = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ─────────────────────────────────────────────────────────────────────────────
# Phase 14 — V2 Beta launch: job feedback (D-13.8 + D-13.13)
#
# Captures 0–100 rating + optional free-text comment after a V2 job
# reaches status='done'. Surfaced to the admin via /api/admin/v2-feedback
# and aggregated into /api/v2/stats + /api/admin/v2-stats.
#
# One row per (job, user) — the endpoint enforces this and returns 409
# on duplicate submission. user_id ON DELETE SET NULL so aggregate
# admin stats survive a user account deletion; job_id ON DELETE CASCADE
# because feedback without the parent job is meaningless.
# ─────────────────────────────────────────────────────────────────────────────


class JobFeedback(Base):
    __tablename__ = "job_feedback"
    __table_args__ = (
        UniqueConstraint("job_id", "user_id", name="uq_job_feedback_user"),
        CheckConstraint("rating >= 0 AND rating <= 100", name="ck_job_feedback_rating"),
        Index("ix_job_feedback_submitted_at", "submitted_at"),
    )

    id           = Column(Integer, primary_key=True, index=True)
    job_id       = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"),
                          nullable=False, index=True)
    user_id      = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"),
                          nullable=True, index=True)
    rating       = Column(Integer, nullable=False)
    comment      = Column(Text, default="")
    submitted_at = Column(DateTime(timezone=True), server_default=func.now())


# ──────────────────────────────────────────────────────────────────────────
# Upload Rewrite v2 — new tables for the PublishTask → Fanout → Scheduler
# → BrandingWorker → UploadWorker pipeline. See docs/upload-rewrite/.
# Additive ONLY. Old `upload_jobs` table is preserved during cutover.
#
# Spec sources of truth:
#   - docs/upload-rewrite/CONTRACTS.md §3 (schema)
#   - docs/upload-rewrite/DECISIONS.md Decisions 3, 4, 6, 8 (plan-tier seed)
#
# All foreign keys here reference existing tables (jobs, users, channels,
# oauth_tokens, clips, user_assets) but use SET NULL / CASCADE semantics
# matching the brief: a deleted user wipes their entire publish history;
# a deleted asset just NULL-outs the brand profile column.
# ──────────────────────────────────────────────────────────────────────────


class PlanTier(Base):
    """Subscription tier definition — drives slot caps + monthly credit
    allotments + direct-path gating. Three rows are seeded by
    main._migrate_schema() at startup: ``free``, ``pro``, ``enterprise``.
    Values from DECISIONS.md Decisions 3 + 4 + 6.

    Convention: ``-1`` means *unlimited* for the cap columns so the
    column stays INT-typed without NULL juggling at the call sites.
    """
    __tablename__ = "plan_tiers"

    id                       = Column(Integer, primary_key=True, index=True)
    # Tier name — one of {'free','pro','enterprise'}. Unique because the
    # F-agent's credit allotment lookup is keyed by name.
    name                     = Column(String(16), unique=True, nullable=False, index=True)
    monthly_credit_allotment = Column(Integer, nullable=False)
    slot_cap_active_uploads  = Column(Integer, nullable=False)
    # Decision 6: Free is RTMP-only. Pro+ may pick Direct.
    direct_path_allowed      = Column(Boolean, nullable=False, default=True)
    # -1 sentinel = unlimited.
    max_channels             = Column(Integer, nullable=False)
    max_publishes_per_day    = Column(Integer, nullable=False)
    created_at               = Column(DateTime(timezone=True), server_default=func.now())
    updated_at               = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class BrandProfile(Base):
    """Per-owner branding template (logo + watermark + colors + fonts +
    intro/outro + socials). Owner is polymorphic — ``owner_kind`` is one
    of {'user','channel','oauth_token'} and ``owner_id`` is the matching
    row id in that table. The D-agent's resolver walks oauth_token →
    channel → user in that precedence order, matching the existing
    ``pipeline_v4/watermark.py:_resolve_channel_logo`` behaviour.

    ``owner_id`` is intentionally a soft FK (no REFERENCES clause)
    because Postgres cannot enforce a single FK across three different
    parent tables. The resolver validates at write time.

    ``version`` is a short hash recomputed by the D-agent on every save;
    feeds into ``upload_jobs_v2.publish_version``.
    """
    __tablename__ = "brand_profiles"
    __table_args__ = (
        Index("ix_brand_profiles_owner", "owner_kind", "owner_id"),
    )

    id                 = Column(Integer, primary_key=True, index=True)
    owner_kind         = Column(String(16), nullable=False)
    owner_id           = Column(Integer, nullable=False)
    name               = Column(String(120), nullable=False)
    # Short content hash; recomputed on save by the D-agent resolver.
    version            = Column(String(40), nullable=False)
    logo_asset_id      = Column(Integer, ForeignKey("user_assets.id", ondelete="SET NULL"), nullable=True)
    watermark_text     = Column(String(64), nullable=True)
    watermark_opacity  = Column(Float, default=0.35)
    watermark_position = Column(String(16), default="lower-center")
    # JSON-as-text for SQLite/Postgres compatibility — Postgres can read
    # a TEXT column as JSON via ::json cast when needed.
    colors_json        = Column(Text, nullable=True)
    fonts_json         = Column(Text, nullable=True)
    intro_asset_id     = Column(Integer, ForeignKey("user_assets.id", ondelete="SET NULL"), nullable=True)
    outro_asset_id     = Column(Integer, ForeignKey("user_assets.id", ondelete="SET NULL"), nullable=True)
    socials_json       = Column(Text, nullable=True)
    created_at         = Column(DateTime(timezone=True), server_default=func.now())
    updated_at         = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class MasterVideo(Base):
    """Clean, unbranded rendered output of the V4 pipeline. Distinct
    from the raw ``SourceUpload`` (= a ``Job`` row) and from per-channel
    ``Clip`` rows. One MasterVideo per rendered Job — a single MasterVideo
    fans out to N channels via PublishTask, each with its own branding
    pass cached in R2.

    ``source_upload_id`` is unique so the same Job cannot produce two
    MasterVideo rows. ``clean_master=False`` rows exist only during the
    KAIZER_CLEAN_MASTER=0 → 1 transition window; once Phase 2 cutover
    completes, the Branding Worker refuses to process False rows.
    """
    __tablename__ = "master_videos"

    id               = Column(Integer, primary_key=True, index=True)
    # NOTE: source_upload_id is NO LONGER unique. A V4 job produces
    # MULTIPLE distinct output files (1 Full Video + N shorts); the
    # legacy→v2 bridge needs ONE MasterVideo PER CLIP (each its own
    # file), so several masters can share one job. Per-clip identity is
    # ``clip_id`` below; the old per-job uniqueness caused the bulletin
    # and every short to collapse onto one master → the same video got
    # posted multiple times.
    source_upload_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"),
                              nullable=False, index=True)
    # The specific Clip (rendered output) this master is the clean
    # source for. Nullable for the native v2 path (one master per job,
    # no Clip); set for every legacy-bridge master.
    clip_id          = Column(Integer, ForeignKey("clips.id", ondelete="CASCADE"),
                              nullable=True, index=True)
    r2_key           = Column(Text, nullable=False)
    duration_seconds = Column(Float, nullable=False)
    bytes            = Column(BigInteger, nullable=False)
    width            = Column(Integer, nullable=False)
    height           = Column(Integer, nullable=False)
    # staging | ready | failed
    status           = Column(String(16), nullable=False, default="staging")
    # 'v4_clean' (KAIZER_CLEAN_MASTER=1) or 'v4_legacy_branded' (KAIZER_CLEAN_MASTER=0).
    pipeline_version = Column(String(16), nullable=False)
    clean_master     = Column(Boolean, nullable=False, default=True)
    created_at       = Column(DateTime(timezone=True), server_default=func.now())
    updated_at       = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class PublishTask(Base):
    """One user's intent to publish one MasterVideo to N channels. The
    Fanout service (B-agent) creates exactly one PublishTask per
    ``POST /api/publish-tasks`` call, then explodes into N
    ``UploadJobV2`` rows in the same transaction. ``target_count`` is
    frozen at creation; ``completed_count``/``failed_count`` are bumped
    atomically by the F-agent on every terminal UploadJob transition.

    Status terminal transitions when ``completed_count + failed_count
    == target_count``: all-success → 'completed', all-fail → 'failed',
    mixed → 'partial_failed'.
    """
    __tablename__ = "publish_tasks"
    __table_args__ = (
        Index("ix_publish_tasks_user_status_priority", "user_id", "status", "priority"),
    )

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"),
                             nullable=False, index=True)
    master_video_id = Column(Integer, ForeignKey("master_videos.id", ondelete="CASCADE"),
                             nullable=False, index=True)
    # critical | high | normal | low — scheduler dispatches by this rank.
    priority        = Column(String(16), nullable=False, default="normal")
    # queued | fanning_out | dispatched | completed | partial_failed | failed | cancelled
    status          = Column(String(20), nullable=False, default="queued")
    target_count    = Column(Integer, nullable=False, default=0)
    completed_count = Column(Integer, nullable=False, default=0)
    failed_count    = Column(Integer, nullable=False, default=0)
    created_at      = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at      = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class UploadJobV2(Base):
    """Refactored per-channel upload job — the v2 of the legacy
    ``upload_jobs`` table. Lives alongside the old one during cutover
    so rollback is trivial (drop the new table, legacy code is
    untouched).

    Key differences from legacy ``UploadJob``:
      - parented by ``publish_task_id`` (not by ``clip_id`` alone)
      - explicit ``upload_path`` enum ('direct' | 'rtmp') replaces the
        old ``upload_provider`` polymorphism
      - first-class ``publish_kind`` ('video' | 'short') — Shorts skip
        the thumbnail step entirely (brief §2)
      - ``idempotency_key`` UNIQUE — dedupe before + after at the DB layer
      - ``predicted_quota_units`` + ``predicted_credit_cost`` reserved
        up front; ``actual_quota_units`` filled by the F-agent's
        reconciliation pass against Google's reported usage
      - ``priority_at_dispatch`` is frozen by the scheduler when the
        job is released so per-tier accounting stays honest even when
        aging promoted the effective priority later

    ``thumbnail_source`` MUST be NULL when ``publish_kind='short'`` —
    YouTube API does not support custom thumbnails on Shorts. The
    F-agent enforces this in code (defense in depth on top of the
    Postgres-side CHECK constraint we add via raw SQL in Phase 3).
    """
    __tablename__ = "upload_jobs_v2"

    id                      = Column(Integer, primary_key=True, index=True)
    publish_task_id         = Column(Integer, ForeignKey("publish_tasks.id", ondelete="CASCADE"),
                                     nullable=False, index=True)
    # Legacy bridge: lets the V2 path still reference the originating Clip
    # row during the transition. Phase 3 may drop this column.
    clip_id                 = Column(Integer, ForeignKey("clips.id", ondelete="SET NULL"),
                                     nullable=True)
    channel_id              = Column(Integer, ForeignKey("channels.id", ondelete="SET NULL"),
                                     nullable=False, index=True)
    # Resolved at dispatch by the scheduler from the channel; NULL until
    # then so a deleted token doesn't cascade-kill in-flight rows.
    oauth_token_id          = Column(Integer, ForeignKey("oauth_tokens.id", ondelete="SET NULL"),
                                     nullable=True, index=True)
    brand_profile_id        = Column(Integer, ForeignKey("brand_profiles.id", ondelete="SET NULL"),
                                     nullable=True, index=True)
    # direct (videos.insert, 1600u) | rtmp (liveBroadcasts*, ~150u)
    # | postiz (hand the branded artifact to Postiz, 0 YouTube quota)
    upload_path             = Column(String(8), nullable=False, index=True)
    # Postiz integration id (copied from Channel.postiz_integration_id at
    # fanout) — the target integration for upload_path='postiz' jobs.
    # NULL for direct/rtmp.
    postiz_integration_id   = Column(String(64), nullable=True)
    # video | short
    publish_kind            = Column(String(8), nullable=False)
    # pipeline_generated | user_override | user_uploaded — NULL when
    # publish_kind='short' (Shorts skip the thumbnail step entirely).
    thumbnail_source        = Column(String(24), nullable=True)
    thumbnail_r2_key        = Column(Text, nullable=True)
    branded_artifact_r2_key = Column(Text, nullable=True)
    # queued | claimed | branding | ready_to_upload | uploading |
    # completed | failed | cancelled | parked_quota
    # ('claimed' = durable-queue worker holds the lease; retry is
    #  expressed as status='queued' with next_attempt_at in the future)
    status                  = Column(String(20), nullable=False, default="queued", index=True)
    # Branding mode at upload: 'per_channel' (apply this channel's logo +
    # watermark + socials at upload, the default) | 'as_is' (the source video
    # is ALREADY branded — upload it verbatim to every channel, no overlay).
    brand_mode              = Column(String(16), nullable=False, default="per_channel",
                                     server_default="per_channel")
    # Logo/watermark PLACEMENT when overlaying (brand_mode='per_channel'):
    # 'template' (use the template's marked slot if it has one — designer intent,
    # the default) | 'channel' (ignore the slot, use this channel's own position
    # setting / default corner). Content (which logo, the text, opacity) always
    # comes from the channel either way; this only chooses WHERE.
    brand_placement         = Column(String(16), nullable=False, default="template",
                                     server_default="template")
    # Per-publish YouTube setting OVERRIDES carried from the publish modal.
    # NULL = no override for this publish → fall back to the channel's yt_*
    # default at upload. These WIN over the channel default when set.
    yt_category_id          = Column(String(10), nullable=True)
    yt_default_language     = Column(String(10), nullable=True)
    yt_playlist_id          = Column(String(64), nullable=True)
    yt_license              = Column(String(20), nullable=True)
    yt_made_for_kids        = Column(Boolean, nullable=True)
    attempts                = Column(Integer, nullable=False, default=0)
    last_error              = Column(Text, nullable=True)
    # SHA256(master_video_id|channel_id|publish_version) as hex.
    idempotency_key         = Column(String(64), unique=True, nullable=False)
    publish_version         = Column(String(40), nullable=False)
    youtube_video_id        = Column(String(32), nullable=True, index=True)
    # Publish intent carried from the PublishRequest through fan-out so
    # dispatch honours the user's choice instead of hard-defaulting to
    # private. 'public' | 'unlisted' | 'private'. publish_at (UTC, when
    # set) schedules a private→public flip at YouTube.
    privacy_status          = Column(String(20), nullable=False, default="private",
                                     server_default="private")
    publish_at              = Column(DateTime(timezone=True), nullable=True)
    predicted_quota_units   = Column(Integer, nullable=False)
    predicted_credit_cost   = Column(Integer, nullable=False)
    # Filled by F-agent reconciliation pass.
    actual_quota_units      = Column(Integer, nullable=True)
    bytes_uploaded          = Column(BigInteger, nullable=False, default=0)
    # Resumable session URI — gap in DISCOVERY §2 the E-agent closes by
    # persisting after request.execute() returns the URI.
    upload_uri              = Column(Text, nullable=True)
    # ── Durable queue (Postgres SKIP LOCKED) ────────────────────────
    # Claimable when status='queued' AND next_attempt_at <= now().
    # Retry backoff = future next_attempt_at; no extra status needed.
    next_attempt_at         = Column(DateTime(timezone=True), nullable=False,
                                     server_default=func.now())
    # Crash detection: a worker renews this while it holds the job.
    # The reaper requeues any active-status row whose lease expired.
    lease_expires_at        = Column(DateTime(timezone=True), nullable=True)
    # worker_id (host:pid:uuid) fencing token — every status/progress
    # write is guarded with "AND claimed_by = :worker_id" so a zombie
    # thread can never clobber a reclaimed job.
    claimed_by              = Column(String(64), nullable=True)
    # Denormalized from publish_tasks at fanout INSERT time so the hot
    # claim query + per-user active-cap aggregate need no joins.
    # Soft reference (no FK) — publish_tasks.user_id is the truth.
    user_id                 = Column(Integer, nullable=True, index=True)
    priority                = Column(String(16), nullable=False,
                                     default="normal", server_default="normal")
    # Frozen at scheduler release; aging may change effective priority
    # in the live queue but the books use the snapshot here.
    priority_at_dispatch    = Column(String(16), nullable=True)
    dispatched_at           = Column(DateTime(timezone=True), nullable=True)
    started_at              = Column(DateTime(timezone=True), nullable=True)
    finished_at             = Column(DateTime(timezone=True), nullable=True)
    created_at              = Column(DateTime(timezone=True), server_default=func.now())
    updated_at              = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class PublishAttempt(Base):
    """Idempotency log — one row per upload attempt against a given
    ``idempotency_key``. The UNIQUE constraint is the database-level
    dedupe guarantee: two workers picking up the same key simultaneously
    cannot both ``INSERT … ON CONFLICT DO NOTHING`` and proceed to
    ``videos.insert``.

    Status:
      - in_flight: worker registered the attempt and is uploading now
      - completed: upload succeeded, ``youtube_video_id`` is populated
      - failed: permanent failure; caller may retry (creates new row)
      - recovered: F-agent's ``recover_orphans`` confirmed via
        ``videos.list`` (1 unit; NEVER ``search.list``, brief §2 + §9)
        that the upload landed on YouTube even though the worker died
        before recording success
    """
    __tablename__ = "publish_attempts"

    id               = Column(Integer, primary_key=True, index=True)
    upload_job_id    = Column(Integer, ForeignKey("upload_jobs_v2.id", ondelete="CASCADE"),
                              nullable=False, index=True)
    # Same value as upload_jobs_v2.idempotency_key for the parent.
    idempotency_key  = Column(String(64), unique=True, nullable=False)
    youtube_video_id = Column(String(32), nullable=True)
    # in_flight | completed | failed | recovered
    status           = Column(String(16), nullable=False)
    # 1-based.
    attempt_no       = Column(Integer, nullable=False)
    worker_id        = Column(String(64), nullable=False)
    created_at       = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at       = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class CreditLedger(Base):
    """Append-only signed credit ledger. ``balance_after`` is
    denormalised (last-row-wins per user) so a user's current balance
    is a single index lookup, not a sum over history.

    When ``reason`` is ``upload_direct`` or ``upload_rtmp`` both
    ``path`` and ``publish_kind`` MUST be populated (the F-agent
    enforces this in code). For ``monthly_allotment``, ``refund``, and
    ``admin_adjustment`` they stay NULL.
    """
    __tablename__ = "credit_ledger"

    id                    = Column(Integer, primary_key=True, index=True)
    user_id               = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"),
                                   nullable=False, index=True)
    # Signed: negative for upload spend, positive for allotment / refund.
    delta                 = Column(Integer, nullable=False)
    # monthly_allotment | upload_direct | upload_rtmp | refund | admin_adjustment
    reason                = Column(String(32), nullable=False)
    upload_job_id         = Column(Integer, ForeignKey("upload_jobs_v2.id", ondelete="SET NULL"),
                                   nullable=True, index=True)
    # direct | rtmp — only when reason is one of the upload_* values.
    path                  = Column(String(8), nullable=True)
    publish_kind          = Column(String(8), nullable=True)
    predicted_quota_units = Column(Integer, nullable=True)
    balance_after         = Column(Integer, nullable=False)
    created_at            = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class QuotaBurnLog(Base):
    """Predicted-vs-actual reconciliation log for every YouTube API
    call we make through the new upload path. Orthogonal to the
    existing ``youtube_api_calls`` forensic log (which stays unchanged,
    brief §0 + §4) — that one captures every API call ever made; this
    one specifically captures the *delta* between what ``reserve()``
    gated on and what Google actually charged.

    The F-agent's reconciliation pass (Phase 2) fills
    ``reconciled_actual_cost`` + ``reconciled_at`` from Google's
    reported usage. The brief is firm: never treat the published
    1,600-unit ``videos.insert`` cost as truth.
    """
    __tablename__ = "quota_burn_log"

    id                     = Column(Integer, primary_key=True, index=True)
    # NULL for non-upload calls (e.g. recovery probes via videos.list).
    upload_job_id          = Column(Integer, ForeignKey("upload_jobs_v2.id", ondelete="SET NULL"),
                                    nullable=True, index=True)
    # e.g. 'videos.insert', 'thumbnails.set', 'liveBroadcasts.insert'.
    operation              = Column(String(40), nullable=False)
    predicted_cost         = Column(Integer, nullable=False)
    # success | transient_error | quota_exceeded | permanent_error
    observed_outcome       = Column(String(20), nullable=False)
    http_status            = Column(Integer, nullable=True)
    was_quota_exceeded     = Column(Boolean, nullable=False, default=False)
    reconciled_actual_cost = Column(Integer, nullable=True)
    reconciled_at          = Column(DateTime(timezone=True), nullable=True)
    created_at             = Column(DateTime(timezone=True), server_default=func.now(),
                                    nullable=False, index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Custom (developer-uploaded) HTML/CSS video templates
# ─────────────────────────────────────────────────────────────────────────────
class CustomTemplate(Base):
    """A developer-uploaded HTML/CSS template (see services/custom_templates/).

    The uploaded bundle (HTML + CSS + assets) is extracted to ``dir_path``; the
    parsed contract (canvas + discovered video/image/text slots) is snapshotted in
    ``contract_json`` so the picker and render pipeline understand it without
    re-parsing. ``visibility`` is the uploader's choice — ``private`` (only the
    owner) or ``public`` (the shared library). Jobs reference a template as the
    string ``custom:<id>`` in the existing ``frame_layout`` field (no schema change
    to jobs/clips)."""
    __tablename__ = "custom_templates"

    id            = Column(Integer, primary_key=True, index=True)
    owner_id      = Column(Integer, index=True, nullable=True)      # users.id
    name          = Column(String(120), default="")
    slug          = Column(String(140), unique=True, index=True)
    visibility    = Column(String(10), default="private")          # private | public
    status        = Column(String(12), default="ready")            # ready | invalid | disabled
    # storage
    dir_path      = Column(String(500), default="")                # extracted bundle dir
    entry_rel     = Column(String(300), default="index.html")
    # Authoring format: "html" (the builder-editable HTML/CSS system) or
    # "svg" (an uploaded SVG layout auto-wrapped into an HTML entry at
    # upload; original kept as template.svg; builder editing blocked).
    format        = Column(String(8), default="html", nullable=True)
    preview_path  = Column(String(500), default="")                # generated preview image
    # contract (machine understanding of the template)
    canvas_w      = Column(Integer, default=1080)
    canvas_h      = Column(Integer, default=1920)
    contract_json = Column(JSON, default=dict)                     # {canvas, slots, warnings}
    # builder: id of the template this was forked/derived from (NULL = an original).
    # Drives the "Built on <name>" attribution shown on public forks.
    derived_from  = Column(Integer, nullable=True)
    # Remix: whether other users may fork+edit this template into their own copy.
    # Default True (open). SVG templates are never remixable (no editable HTML);
    # built-in/system templates are always remixable regardless of this flag.
    allow_remix   = Column(Boolean, nullable=False, default=True)
    # meta
    description   = Column(Text, default="")
    when_to_use   = Column(Text, default="")    # shown in the preview modal
    how_to_use    = Column(Text, default="")    # shown in the preview modal
    use_count     = Column(Integer, default=0)
    rating_sum    = Column(Integer, default=0)  # community rating (sum of 1..5 stars)
    rating_count  = Column(Integer, default=0)
    # First-party curated design: shown under "Built-in templates" in the
    # job-selection picker (above user/community creations). NULL/false =
    # a normal user creation. Set by operators, never by the upload API.
    is_builtin    = Column(Boolean, nullable=True)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())
    updated_at    = Column(DateTime(timezone=True), server_default=func.now(),
                           onupdate=func.now())


class UserStylePack(Base):
    """A user-COMPOSED style pack: pick the LOOK of one built-in pack,
    the MOTION (transitions+pace) of another, the SOUND identity of a
    third and the CARD design of a fourth — saved with a name and
    reusable on any future job (operator: 'creative users build new
    styles from existing ones'). PRIVATE to its owner; admins see every
    user creation in the admin tab."""
    __tablename__ = "user_style_packs"

    id         = Column(Integer, primary_key=True, index=True)
    owner_id   = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"),
                        index=True, nullable=False)
    name       = Column(String(80), nullable=False)
    # {"look": packKey, "motion": packKey, "sound": packKey, "cards": packKey}
    spec       = Column(JSON, default=dict)
    created_at = Column(DateTime, server_default=func.now())


# Ported from kaizer-platform@d5fd482 server/models.py (class DesktopLicense).
# Changes: columns verbatim; docstring trimmed of vendor-only cross-references
# (tenancy/seed_plans.py, Organization/OrganizationQuota — modules/tables that
# do not exist in this backend). Table is created by main.py's existing
# Base.metadata.create_all(bind=engine) — no migration entry needed.
class DesktopLicense(Base):
    """Desktop app machine activation.

    One row per machine fingerprint activated against a user's account.
    The desktop shell generates a fingerprint client-side (hash of
    hostname + a hardware identifier) and calls `POST /api/desktop/activate`
    on first launch; this table is the server-side record of which machines
    are currently allowed to run the app under that account.

    `ACTIVATION_LIMIT` (routers/desktop.py) is a placeholder constant —
    "configurable placeholder pending real economics" ("N systems per
    contract" has no locked number yet). Revoking sets `revoked=True`
    rather than deleting the row, so activation history stays auditable.

    Scoped to `user_id` only — a per-user limit is the correct scope for
    today's single-tenant-first reality, and adding a tenant FK later is
    an additive migration, not a breaking one.
    """
    __tablename__ = "desktop_licenses"
    __table_args__ = (
        UniqueConstraint("user_id", "machine_fingerprint", name="uq_desktop_license_user_machine"),
    )

    id                  = Column(Integer, primary_key=True, index=True)
    # index=True auto-generates "ix_desktop_licenses_user_id" — no separate
    # explicit Index() needed (that would collide on the same name).
    user_id             = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"),
                                  nullable=False, index=True)
    machine_fingerprint = Column(String(128), nullable=False, index=True)
    # Human-readable label only (e.g. hostname) — never used for identity,
    # only for display in the "your activated machines" list.
    machine_label       = Column(String(255), default="")
    activated_at        = Column(DateTime(timezone=True), server_default=func.now())
    last_seen_at        = Column(DateTime(timezone=True), nullable=True)
    revoked             = Column(Boolean, nullable=False, default=False)


class AccountRequest(Base):
    """A prospective user asking the admin for a Kaizer X account.

    The desktop login screen submits email + password + an optional note;
    the password is bcrypt-hashed IMMEDIATELY (never stored raw) so an
    admin approval later can mint the User row without a second password
    exchange. status: pending → approved | rejected.
    """
    __tablename__ = "account_requests"

    id            = Column(Integer, primary_key=True, index=True)
    email         = Column(String(255), nullable=False, index=True)
    name          = Column(String(255), default="")
    password_hash = Column(String(255), nullable=False)
    # Why they want access — free text shown to the admin.
    note          = Column(Text, default="")
    status        = Column(String(12), nullable=False, default="pending", index=True)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())
    decided_at    = Column(DateTime(timezone=True), nullable=True)
    decided_by    = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"),
                           nullable=True)


class ManagedApiKey(Base):
    """Server-managed API keys injected into desktop installs at sign-in.

    user_id NULL = the DEFAULT bundle every user receives; a row with a
    user_id overrides the default for that key name. Values are Fernet-
    encrypted with the same KAIZER_ENCRYPTION_KEY that protects YouTube
    OAuth tokens (crypto.encrypt/decrypt). Only names in the desktop
    injectable whitelist (routers/account_requests.INJECTABLE_KEYS) are
    ever accepted or served.
    """
    __tablename__ = "managed_api_keys"
    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_managed_key_user_name"),
    )

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"),
                        nullable=True, index=True)
    name       = Column(String(64), nullable=False, index=True)
    value_enc  = Column(Text, nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(),
                        onupdate=func.now())


class GoogleMintedKey(Base):
    """Metadata for a Google Cloud API key we minted FOR a specific user.

    The key STRING never lives here — it goes into ManagedApiKey (Fernet)
    so the existing keys-bundle injection delivers it to the desktop. This
    table maps the key's Cloud-Monitoring credential_id ("apikey:<uid>")
    back to the user for the per-user usage/billing view, and carries mint
    status so account approval can soft-fail (Google down → user stays on
    the shared bundle) without losing track of what to retry.

    status: pending | active | failed | revoked
    """
    __tablename__ = "google_minted_keys"
    __table_args__ = (
        UniqueConstraint("user_id", "env_name", name="uq_minted_key_user_env"),
    )

    id                = Column(Integer, primary_key=True, index=True)
    user_id           = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"),
                               nullable=False, index=True)
    env_name          = Column(String(64), nullable=False)   # YOUTUBE_DATA_API_KEY | GEMINI_API_KEY
    key_resource_name = Column(String(256), default="")      # projects/*/locations/global/keys/*
    key_uid           = Column(String(64), default="", index=True)  # credential_id = "apikey:"+uid
    display_name      = Column(String(128), default="")
    status            = Column(String(12), nullable=False, default="pending", index=True)
    error             = Column(Text, default="")
    created_at        = Column(DateTime(timezone=True), server_default=func.now())
    updated_at        = Column(DateTime(timezone=True), server_default=func.now(),
                               onupdate=func.now())
    revoked_at        = Column(DateTime(timezone=True), nullable=True)


class BillingRate(Base):
    """Operator-configured $/1000-request rates for the admin billing view.

    cost_per_1000   = operator's estimated underlying cost basis
    charge_per_1000 = what the operator suggests billing the user
    metric ∈ {'youtube_requests', 'gemini_requests'}.
    """
    __tablename__ = "billing_rates"

    id              = Column(Integer, primary_key=True)
    metric          = Column(String(32), unique=True, nullable=False)
    cost_per_1000   = Column(Float, nullable=False, default=0.0)
    charge_per_1000 = Column(Float, nullable=False, default=0.0)
    currency        = Column(String(8), nullable=False, default="USD")
    updated_at      = Column(DateTime(timezone=True), server_default=func.now(),
                             onupdate=func.now())
