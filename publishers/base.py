"""Publisher abstraction — the contract every publish destination
implements.

This module defines the interface the worker uses to push a finished
clip to a destination platform. Adding a new platform (Meta, X,
LinkedIn, TikTok, Threads) means writing a new ``Publisher`` subclass
in this package and registering it — no changes to the worker.

Lifecycle every Publisher follows
─────────────────────────────────
1. ``prepare(job, clip, video_path)`` — pre-flight: validate file
   format (resolution / duration / size), refresh access tokens,
   reserve any per-platform quota. Returns a ``PrepareResult`` with
   the resolved upload metadata (caption, hashtags, scheduled time,
   thumbnail path).
2. ``publish(prepare_result)`` — the actual upload call. Returns a
   ``PublishResult`` with the platform's video/post id and viewer URL.
3. ``verify(publish_result)`` — optional post-publish check that the
   platform actually accepted the content (some APIs return 200 then
   silently reject on review). Returns ``VerifyResult``.

Each method MUST be idempotent enough that the worker can retry on
transient failure without double-posting. Concrete publishers either:
- Use the platform's `external_id` / `idempotency_key` mechanism, OR
- Persist their own "did we already start this upload" marker on the
  PublishJob row so retries can resume from the last step.

Error contract
──────────────
Publishers raise one of:
- ``TransientPublishError``  → worker retries with backoff.
- ``TerminalPublishError``   → worker flips status to "provider_failed"
                                immediately. Includes a remediation_url
                                so the editor's UI can deep-link the
                                operator to the fix.
- ``QuotaExceededError``     → worker reschedules for the next quota
                                window (per-platform; YouTube resets at
                                midnight Pacific, Meta is a rolling
                                hour, etc.).

Anything else escaping the Publisher gets wrapped as TransientPublishError
by the worker shim — but please don't rely on that; raise explicitly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ── Error hierarchy ─────────────────────────────────────────────────


class PublishError(Exception):
    """Base for every error a Publisher can raise. Carries the verbatim
    upstream error so the operator UI can show it without guessing."""
    def __init__(self, message: str, *, remediation_url: str = "",
                 upstream_status: Optional[int] = None,
                 upstream_body: Optional[str] = None):
        super().__init__(message)
        self.remediation_url = remediation_url
        self.upstream_status = upstream_status
        self.upstream_body = upstream_body


class TransientPublishError(PublishError):
    """Recoverable failure — retry with backoff. Examples: 5xx, network
    timeout, momentary rate-limit that respects Retry-After."""


class TerminalPublishError(PublishError):
    """Unrecoverable failure — flip job status to ``provider_failed``
    immediately. Examples: revoked OAuth, deleted destination, deleted
    integration, missing required Meta permission.

    The ``remediation_url`` should deep-link the editor's UI to the
    place the operator can fix the issue (e.g. /channels for a revoked
    OAuth token, /settings/meta for a missing permission)."""


class QuotaExceededError(PublishError):
    """Per-platform quota is dry. Worker reschedules for the next
    quota window — does NOT count against the retry ladder."""
    def __init__(self, message: str, *, next_window_at: Optional[datetime] = None,
                 **kwargs):
        super().__init__(message, **kwargs)
        # Datetime when the quota becomes available again. Worker uses
        # this to wake the job up at the right moment instead of
        # polling.
        self.next_window_at = next_window_at or _next_day_midnight_utc()


def _next_day_midnight_utc() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0).replace(
        day=now.day + 1
    )


# ── Lifecycle result dataclasses ────────────────────────────────────


@dataclass
class PrepareResult:
    """Output of ``Publisher.prepare()``. Carries everything the
    publish call needs so ``publish()`` is purely the network call."""
    title: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    hashtags: list[str] = field(default_factory=list)
    privacy: str = "private"
    scheduled_at: Optional[datetime] = None
    thumbnail_path: str = ""
    # Resolved file we'll upload (may be a re-muxed / re-encoded variant
    # of the source if the destination has format constraints).
    upload_path: str = ""
    # Per-platform extras that don't fit the common schema.
    extras: dict = field(default_factory=dict)


@dataclass
class PublishResult:
    """Output of ``Publisher.publish()``. Worker stores these fields on
    the PublishJob row for cross-referencing and analytics."""
    # Platform's identifier for the published item (videoId on YT,
    # post_id on Meta, tweet id on X, etc.).
    external_id: str = ""
    # User-facing URL.
    public_url: str = ""
    # When the platform first acknowledged the upload (NOT the moment
    # it goes live for scheduled posts — that's `published_at`).
    accepted_at: Optional[datetime] = None
    # When the platform reports it's actually visible to viewers (for
    # public posts this is `accepted_at`; for scheduled it's later).
    published_at: Optional[datetime] = None
    # Raw provider response body — kept for audit trail. Subclasses
    # decide whether to truncate or store full JSON.
    raw_response: str = ""


@dataclass
class VerifyResult:
    """Output of ``Publisher.verify()`` — post-publish status check."""
    is_live: bool = False
    view_count: Optional[int] = None
    failure_reason: str = ""  # set when is_live=False AND we know why


# ── Abstract Publisher ──────────────────────────────────────────────


class Publisher(ABC):
    """Concrete publishers subclass this. Each instance is bound to a
    destination (a YT channel, an FB page, an IG account, etc.) and
    knows how to talk to that platform on that destination's behalf.

    The worker calls ``prepare()`` → ``publish()`` → ``verify()`` in
    order. Publishers can override the default ``verify()`` no-op if
    the platform supports a post-publish status check."""

    # Class-level provider identifier — must match what the operator
    # sets in OAuthToken.upload_provider / Channel.upload_provider /
    # system_setting('upload_provider'). Used by the worker's
    # ``get_publisher_for(provider, job, db)`` lookup.
    provider_key: str = ""

    # Set by subclass to a short, human-friendly platform name
    # ("YouTube", "Facebook Page", "Instagram Reels"). Surfaces in the
    # publish modal + status pills.
    display_name: str = ""

    def __init__(self, *, destination, db_session):
        """``destination`` is the platform-specific model row (Channel,
        MetaAccount, etc.). ``db_session`` is a live SQLAlchemy session
        the publisher can use for status updates."""
        self.destination = destination
        self.db = db_session

    @abstractmethod
    def prepare(self, *, job, clip, source_video_path: str) -> PrepareResult:
        """Pre-flight: validate format, refresh tokens, build metadata.
        Raise TerminalPublishError if the destination is broken
        (revoked OAuth, format incompatible, etc.).
        """

    @abstractmethod
    def publish(self, *, job, prepared: PrepareResult) -> PublishResult:
        """Upload the file + metadata. Idempotency: callers may retry
        this method; concrete publishers must either use the
        platform's idempotency mechanism OR persist a "did we start
        this" marker on the job row so retries resume cleanly."""

    def verify(self, *, job, published: PublishResult) -> VerifyResult:
        """Optional post-publish check. Default: trust the publish
        response and report live=True. Override when the platform has
        a moderation queue / scheduled-publish gap / review step."""
        return VerifyResult(is_live=True)

    # ── Helpers a publisher might want to share ────────────────────

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)
