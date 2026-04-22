from sqlalchemy import (
    Column, Integer, BigInteger, String, Text, DateTime, ForeignKey, Float, Boolean,
    JSON, UniqueConstraint, Index,
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


class Job(Base):
    __tablename__ = "jobs"

    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    status       = Column(String(20), default="pending")   # pending | running | done | failed
    platform     = Column(String(50))
    frame_layout = Column(String(50))
    video_name   = Column(String(255))
    language     = Column(String(10), default="te")        # ISO 639-1: te|hi|ta|kn|ml|bn|mr|gu|en
    output_dir   = Column(String(500), default="")
    log          = Column(Text, default="")
    error        = Column(Text, default="")
    created_at   = Column(DateTime(timezone=True), server_default=func.now())
    # Wall-clock timing of the pipeline subprocess.
    started_at   = Column(DateTime(timezone=True), nullable=True)
    finished_at  = Column(DateTime(timezone=True), nullable=True)

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
    created_at    = Column(DateTime(timezone=True), server_default=func.now())

    # ── Storage (Phase 5) ──────────────────────────────────────────
    # When storage_backend='local' the URL is /media/<key>; when 'r2' the URL
    # is a CDN/public link or a signed URL (frontend re-fetches if expired).
    # file_path remains populated for backwards compatibility.
    storage_url     = Column(String(500), default="")
    storage_key     = Column(String(500), default="")
    storage_backend = Column(String(20),  default="")


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
