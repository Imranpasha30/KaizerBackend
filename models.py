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


class Job(Base):
    __tablename__ = "jobs"

    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    status       = Column(String(20), default="pending")   # pending | running | done | failed | cancelled
    platform     = Column(String(50))
    frame_layout = Column(String(50))
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
    # have a "Kaizer News Telugu" style profile.
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_channel_user_name"),)

    id                 = Column(Integer, primary_key=True, index=True)
    user_id            = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    name               = Column(String(255), nullable=False, index=True)
    handle             = Column(String(100), default="")
    language           = Column(String(10), default="te")
    title_formula      = Column(Text, default="")
    desc_style         = Column(String(50), default="hook_first")
    footer             = Column(Text, default="")
    fixed_tags         = Column(JSON, default=list)
    hashtags           = Column(JSON, default=list)
    # Optional overlay logo for videos rendered under this channel.  FK to
    # a UserAsset the user picked from their Assets library.  Null = no
    # logo (the SaaS default — users opt in via the Style Profiles page).
    logo_asset_id      = Column(Integer, ForeignKey("user_assets.id", ondelete="SET NULL"), nullable=True, index=True)
    mandatory_hashtags = Column(JSON, default=list)
    is_priority        = Column(Boolean, default=False)
    # Per-channel upload route override.  Null = use the system-wide
    # default from system_settings.UPLOAD_PROVIDER.  Lets the admin
    # send one channel through Postiz and another through the native
    # YouTube uploader simultaneously — useful while comparing the two
    # paths side-by-side, or when one channel's google project has
    # exhausted its daily quota.
    upload_provider    = Column(String(20), nullable=True)  # "postiz" | "kaizer" | "native_rtmp" | null
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
    last_synced_at     = Column(DateTime(timezone=True), server_default=func.now(),
                                 onupdate=func.now())


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
