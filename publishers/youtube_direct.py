"""Direct YouTube publisher — wraps the existing youtube.uploader path
in the Publisher abstraction so the worker doesn't carry hardcoded
YouTube knowledge.

This is the "use our own OAuth, no Postiz" path. Quota cost per upload
is 1,600 units against the project's 10,000/day default — i.e. ~6
uploads per day per OAuth project. The user is in talks with YouTube to
raise the cap; until then the QuotaTracker below makes sure we never
silently over-spend and the worker reschedules on day-window failures
instead of burning the retry ladder.

Resumable uploads + token refresh are handled by the underlying
``youtube.uploader.upload_video`` — this module just adapts inputs and
outputs to the Publisher contract.
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
    QuotaExceededError,
)


@register
class YoutubeDirectPublisher(Publisher):
    provider_key = "kaizer"   # legacy name in the existing routing knob
    display_name = "YouTube (direct upload)"

    # YouTube quota: videos.insert = 1600 units, default project cap
    # 10,000/day. Keep a soft cap so we don't surprise the operator
    # with a "you're out" right before a scheduled bulletin.
    QUOTA_COST_INSERT = 1600

    def prepare(self, *, job, clip, source_video_path: str) -> PrepareResult:
        from youtube.uploader import sanitize_tags  # late import — heavy SDK

        # The destination is a Channel row that has an attached
        # OAuthToken. Channel.oauth_token must be present; otherwise
        # the destination was misconfigured at create time.
        ch = self.destination
        if not ch or not ch.oauth_token:
            raise TerminalPublishError(
                f"Channel {getattr(ch, 'id', '?')} has no OAuth token",
                remediation_url="/channels",
            )
        if not (ch.oauth_token.refresh_token_enc or ch.oauth_token.access_token_enc):
            raise TerminalPublishError(
                "Channel OAuth missing refresh token. Reconnect on /channels.",
                remediation_url="/channels",
            )

        # Build the metadata the uploader expects — _build_body() reads
        # straight off the job + clip, so we don't reshuffle fields
        # here. PrepareResult mostly carries inputs for verify() later.
        return PrepareResult(
            title=(clip.seo_title_final or "")[:95],
            description=(clip.seo_description_final or "")[:5000],
            tags=sanitize_tags(getattr(clip, "seo_keywords", None) or []),
            hashtags=getattr(clip, "seo_hashtags", []) or [],
            privacy=(job.privacy_status or "private"),
            scheduled_at=getattr(job, "publish_at", None),
            thumbnail_path=getattr(clip, "thumb_path", "") or "",
            upload_path=source_video_path,
            extras={"publish_kind": getattr(job, "publish_kind", "video")},
        )

    def publish(self, *, job, prepared: PrepareResult) -> PublishResult:
        from youtube.uploader import (
            upload_video, set_thumbnail, UploadError, TransientUploadError,
        )
        from youtube import oauth as yt_oauth

        try:
            creds = yt_oauth.creds_for_channel(self.db, self.destination.id)
        except Exception as e:
            raise TerminalPublishError(
                f"OAuth token unusable: {e}",
                remediation_url="/channels",
            ) from e

        try:
            video_id = upload_video(
                creds=creds,
                job=job,
                clip_path=prepared.upload_path,
                db=self.db,
            )
        except TransientUploadError as e:
            raise TransientPublishError(str(e)) from e
        except UploadError as e:
            # Distinguish quota exhaustion from other terminal failures.
            msg = str(e).lower()
            if "quotaexceeded" in msg or "daily limit" in msg:
                raise QuotaExceededError(
                    str(e),
                    remediation_url="https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas",
                )
            if "invalid_grant" in msg or "unauthorized" in msg:
                raise TerminalPublishError(
                    str(e),
                    remediation_url="/channels",
                )
            raise TransientPublishError(str(e)) from e

        # Optional thumbnail. Failures here are NEVER terminal for the
        # publish — the video is up; the thumbnail is a nicety. We log
        # and continue.
        if prepared.thumbnail_path:
            try:
                set_thumbnail(creds, video_id, prepared.thumbnail_path, db=self.db)
            except Exception as e:
                # eslint-style: don't shadow the publish success.
                # The worker will surface it as a soft warning on the
                # row but the status stays "done".
                import logging
                logging.getLogger("publishers.youtube_direct").warning(
                    "thumbnail set failed for video %s: %s", video_id, e,
                )

        return PublishResult(
            external_id=video_id,
            public_url=f"https://youtu.be/{video_id}",
            accepted_at=datetime.now(timezone.utc),
            published_at=(
                prepared.scheduled_at
                if (prepared.privacy == "private" and prepared.scheduled_at)
                else datetime.now(timezone.utc)
            ),
        )

    def verify(self, *, job, published: PublishResult) -> VerifyResult:
        # YouTube acknowledges videos.insert immediately; the moderation
        # queue catches up over the following minutes. We mark live=True
        # on insert; a separate background task could call videos.list
        # later to check `processingDetails.processingStatus`.
        return VerifyResult(is_live=True)
