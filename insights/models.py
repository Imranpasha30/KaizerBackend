"""Insights / Trend Finder — FROZEN DB schema (Phase 1).

Four tables model: one user → many channels → many analysis runs → versioned reports.

  ChannelSnapshot   one ingestion run's point-in-time capture of a channel + ingest
                    metadata (analytics-access mode, quota cost, raw-pull cache key).
  VideoMetric       per-video metrics for a snapshot (the analytical grain). Data-API
                    fields always present; Analytics-API fields NULL in public mode.
  AnalysisRun       one execution of the analysis engine over a snapshot; `results`
                    holds the evidence-backed computations (each cites video_ids).
  ReportVersion     a versioned, human-readable strategy report generated from a run.

They register on the shared ``database.Base`` so main.py's
``Base.metadata.create_all()`` creates them in dev; production runs the manual SQL in
``docs/MIGRATIONS.md``. Future column changes go through the lazy ``_migrate_schema()``
pattern in main.py.

SCHEMA IS FROZEN: ``scripts/test_insights_schema.py`` asserts the exact column set of
every table. Changing a column requires updating that freeze in the same commit — on
purpose.

Kaizer X is a video PRODUCTION & analytics tool: these tables exist to understand and
improve editorial/content performance, never to operate channels for bulk monetization.
"""
from sqlalchemy import (
    Column, Integer, BigInteger, String, Text, DateTime, ForeignKey, Float, Boolean,
    JSON, Index,
)
from sqlalchemy.sql import func
from database import Base


class ChannelSnapshot(Base):
    """One ingestion run's capture of a channel + ingest metadata. A re-run within the
    cache TTL reuses the most recent snapshot instead of re-pulling (quota safety)."""
    __tablename__ = "insights_channel_snapshots"

    id                = Column(Integer, primary_key=True, index=True)
    user_id           = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    # The Kaizer Channel row this snapshot belongs to (one user → many channels). Nullable so
    # a public-data-only snapshot of a non-connected channel can still be stored.
    channel_id        = Column(Integer, ForeignKey("channels.id", ondelete="SET NULL"), nullable=True, index=True)
    google_channel_id = Column(String(50), nullable=False, index=True)
    channel_title     = Column(String(255), default="")
    # Channel-local timezone for publish day/hour math (IANA name, e.g. "Asia/Kolkata").
    timezone          = Column(String(64), default="UTC")
    # Analytics ACCESS mode for this pull:
    #   "full"   = owner authorized with yt-analytics.readonly → CTR/retention/traffic available
    #   "public" = Data-API-only (non-owned channel, or owner who hasn't granted analytics)
    access_mode       = Column(String(16), nullable=False, default="public")
    subscriber_count  = Column(BigInteger, default=0)
    video_count       = Column(Integer, default=0)        # total videos reported by the channel
    view_count        = Column(BigInteger, default=0)     # lifetime channel views
    videos_ingested   = Column(Integer, default=0)        # VideoMetric rows captured this snapshot
    channel_age_days  = Column(Integer, nullable=True)    # days since channel publishedAt (maturity input)
    # Raw API pull cached (R2, 24h TTL) so re-runs don't re-burn quota.
    raw_cache_key     = Column(Text, nullable=True)
    raw_cached_at     = Column(DateTime(timezone=True), nullable=True)
    # Quota accounting (predicted vs actual; mirrors quota_burn_log semantics).
    quota_predicted   = Column(Integer, default=0)
    quota_actual      = Column(Integer, default=0)
    status            = Column(String(16), nullable=False, default="ok")   # ok | partial | failed
    error             = Column(Text, nullable=True)
    created_at        = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    __table_args__ = (Index("ix_insights_snap_user_gcid", "user_id", "google_channel_id"),)


class VideoMetric(Base):
    """Per-video metrics for a snapshot. Data-API fields are always populated; the
    Analytics-API fields (impressions/CTR/retention/traffic/subs/early-window) are NULL
    in public mode or when the channel didn't grant yt-analytics.readonly."""
    __tablename__ = "insights_video_metrics"

    id                  = Column(Integer, primary_key=True, index=True)
    snapshot_id         = Column(Integer, ForeignKey("insights_channel_snapshots.id", ondelete="CASCADE"), nullable=False, index=True)
    video_id            = Column(String(32), nullable=False, index=True)
    # ── Data API v3 (always available) ──
    title               = Column(Text, default="")
    description_short   = Column(Text, default="")
    tags                = Column(JSON, default=list)
    published_at_utc    = Column(DateTime(timezone=True), nullable=True)
    published_at_local  = Column(DateTime(timezone=True), nullable=True)
    category_id         = Column(String(16), default="")
    thumbnail_url       = Column(Text, default="")
    duration_seconds    = Column(Integer, default=0)
    default_language    = Column(String(16), default="")
    is_short            = Column(Boolean, default=False)
    view_count          = Column(BigInteger, default=0)
    like_count          = Column(BigInteger, default=0)
    comment_count       = Column(BigInteger, default=0)
    # ── Analytics API v2 (full mode only; NULL otherwise) ──
    impressions         = Column(BigInteger, nullable=True)
    impressions_ctr     = Column(Float, nullable=True)        # 0..1 (normalized from API %)
    avg_view_seconds    = Column(Float, nullable=True)
    avg_view_percentage = Column(Float, nullable=True)        # 0..100
    subscribers_gained  = Column(Integer, nullable=True)
    estimated_minutes_watched = Column(BigInteger, nullable=True)
    views_first_24h     = Column(BigInteger, nullable=True)
    views_first_48h     = Column(BigInteger, nullable=True)
    views_first_7d      = Column(BigInteger, nullable=True)
    # Per-video view split by traffic source (insightTrafficSourceType), normalized to
    # Studio labels: {"browse":n,"suggested":n,"search":n,"external":n,"notifications":n,
    # "shorts":n,"channel":n,"playlist":n,"other":n}. NULL in public mode.
    traffic_sources     = Column(JSON, nullable=True)
    # ── Derived at ingest (channel-relative) ──
    views_per_hour_48h  = Column(Float, nullable=True)        # early velocity
    reach_ratio         = Column(Float, nullable=True)        # views ÷ subscriber base
    publish_dow         = Column(Integer, nullable=True)      # 0=Mon … 6=Sun (local)
    publish_hour_local  = Column(Integer, nullable=True)      # 0..23 (local)
    topic_cluster       = Column(String(80), nullable=True)   # keyword-cluster label
    created_at          = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_insights_vm_snap_video", "snapshot_id", "video_id"),)


class AnalysisRun(Base):
    """One execution of the analysis engine over a snapshot. `results` is the computed,
    evidence-backed analysis (the 10 diagnostic dimensions or the starter plan) as JSON;
    every finding cites the video_ids behind it. `maturity` records which path ran."""
    __tablename__ = "insights_analysis_runs"

    id             = Column(Integer, primary_key=True, index=True)
    snapshot_id    = Column(Integer, ForeignKey("insights_channel_snapshots.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id        = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    # Auto-detected analysis path (the user never picks): "deep" | "starter" | "blend".
    maturity       = Column(String(16), nullable=False, default="deep")
    status         = Column(String(16), nullable=False, default="ok")   # ok | running | failed
    engine_version = Column(String(16), default="v1")
    params         = Column(JSON, default=dict)        # target_views, tier thresholds, weights…
    results        = Column(JSON, default=dict)        # the diagnostic/starter computations, evidence-backed
    error          = Column(Text, nullable=True)
    created_at     = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    finished_at    = Column(DateTime(timezone=True), nullable=True)


class ReportVersion(Base):
    """A versioned, human-readable strategy report generated from an AnalysisRun. Rendered
    in the V4 canvas / insights panel and exportable. `report_md` is the narrative;
    `report_json` holds the structured sections (exec summary, dimensions, ranked fixes,
    targets, next-10 checklist)."""
    __tablename__ = "insights_report_versions"

    id              = Column(Integer, primary_key=True, index=True)
    analysis_run_id = Column(Integer, ForeignKey("insights_analysis_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id         = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    version         = Column(Integer, default=1)
    provider        = Column(String(16), default="")   # which model wrote it: claude | gemini
    exec_summary    = Column(Text, default="")
    report_md       = Column(Text, default="")
    report_json     = Column(JSON, default=dict)
    tokens_in       = Column(Integer, default=0)
    tokens_out      = Column(Integer, default=0)
    cost_usd        = Column(Float, default=0.0)
    created_at      = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    __table_args__ = (Index("ix_insights_report_run_ver", "analysis_run_id", "version"),)
