"""Postiz publisher — wraps the existing worker.py:_process_via_postiz
path in the Publisher abstraction.

We keep Postiz alive as a peer publisher (alongside YouTube direct
and Meta) for two reasons:
1. Some operators have it set up + paid; ripping it out would force a
   migration day. With it as a registered publisher, those rows keep
   working without changes.
2. Postiz supports surfaces we haven't built direct connectors for
   yet (TikTok, X, LinkedIn, Threads). Until we ship native publishers
   for those, Postiz is the fan-out path.

The actual upload happens through ``_process_via_postiz`` in
``youtube/worker.py`` for now — this Publisher just delegates. A
follow-up refactor will move the implementation into this file so
worker.py shrinks; for now we prefer "preserve existing tested code".
"""
from __future__ import annotations

from datetime import datetime, timezone

from . import register
from .base import (
    Publisher,
    PrepareResult,
    PublishResult,
    VerifyResult,
    TerminalPublishError,
    TransientPublishError,
)


@register
class PostizPublisher(Publisher):
    provider_key = "postiz"
    display_name = "Postiz (fan-out)"

    def prepare(self, *, job, clip, source_video_path: str) -> PrepareResult:
        # Postiz handles its own metadata composition (Postiz reads the
        # clip + channel and builds the post body server-side). We
        # just verify the API key is present so we fail fast if it's
        # missing instead of after the upload starts.
        from clients import postiz as postiz_client
        if not postiz_client.is_enabled():
            raise TerminalPublishError(
                "Postiz upload selected but POSTIZ_API_KEY is empty.",
                remediation_url="/settings",
            )
        return PrepareResult(
            title=(clip.seo_title_final or "")[:120],
            description=(clip.seo_description_final or "")[:5000],
            tags=getattr(clip, "seo_keywords", []) or [],
            hashtags=getattr(clip, "seo_hashtags", []) or [],
            privacy=(job.privacy_status or "private"),
            scheduled_at=getattr(job, "publish_at", None),
            thumbnail_path=getattr(clip, "thumb_path", "") or "",
            upload_path=source_video_path,
        )

    def publish(self, *, job, prepared: PrepareResult) -> PublishResult:
        # Delegate to the existing legacy implementation in
        # youtube/worker.py. The legacy function mutates the job row
        # in-place (sets video_id, status, last_error) so we don't try
        # to construct a synthetic PublishResult — we read the row
        # back after the call.
        from youtube.worker import _process_via_postiz
        # The clip object isn't on the worker's signature anymore; the
        # caller (worker shim) hands us the resolved clip path so we
        # pass it through. _process_via_postiz looks the clip back up
        # from the job row.
        _process_via_postiz(self.db, job, getattr(job, "clip", None), prepared.upload_path)
        # Re-read the row to surface what the legacy function set.
        self.db.refresh(job)
        if job.status == "failed" or job.status == "provider_failed":
            raise TerminalPublishError(
                job.last_error or "Postiz upload failed",
                remediation_url="/settings",
            )
        if job.status not in ("done", "scheduled"):
            raise TransientPublishError(
                job.last_error or f"Postiz left job in unexpected state {job.status!r}"
            )
        return PublishResult(
            external_id=getattr(job, "video_id", "") or "",
            public_url=getattr(job, "video_url", "") or "",
            accepted_at=datetime.now(timezone.utc),
            published_at=getattr(job, "publish_at", None) or datetime.now(timezone.utc),
        )

    def verify(self, *, job, published: PublishResult) -> VerifyResult:
        # Postiz doesn't expose a strong post-publish status check;
        # trust the publish response.
        return VerifyResult(is_live=True)
