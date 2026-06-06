"""TikTok video publisher.

Posts a finished short clip (or "long-form" up to 10 min in late 2024)
to a TikTok account via the TikTok for Developers Content Posting API.

What you need from TikTok before this can post a real video
───────────────────────────────────────────────────────────
1. TikTok for Developers account: developers.tiktok.com → register.
2. App in a project with the **Content Posting API** product enabled.
3. Approved use case:
   - "Direct Post" lets you post immediately on the user's behalf.
   - "Upload" lets you publish content to a draft for the user to
     review in the TikTok app (no auto-publish). Direct Post requires
     review approval (1-3 weeks).
4. Scopes:
   - user.info.basic
   - video.upload          (Upload to drafts)
   - video.publish         (Direct Post — requires review)
5. .env:
   TIKTOK_CLIENT_KEY, TIKTOK_CLIENT_SECRET, TIKTOK_REDIRECT_URI

Format constraints
──────────────────
- Video: ≤10 min (was 60s historically), 9:16 strongly preferred.
- Size: ≤500 MB.
- Codec: H.264/AAC.

Upload flow (Direct Post)
─────────────────────────
1. POST /v2/post/publish/video/init/
     → upload_url + publish_id
2. PUT  <upload_url>  (raw bytes, no auth header)
     → upload completes
3. POST /v2/post/publish/status/fetch/
     → poll until status == PUBLISH_COMPLETE
"""
from __future__ import annotations

import os
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
class TikTokPublisher(Publisher):
    provider_key = "tiktok"
    display_name = "TikTok"

    def prepare(self, *, job, clip, source_video_path: str) -> PrepareResult:
        if not (
            os.environ.get("TIKTOK_CLIENT_KEY")
            and os.environ.get("TIKTOK_CLIENT_SECRET")
        ):
            raise TerminalPublishError(
                "TikTok publisher not configured — set TIKTOK_CLIENT_KEY + TIKTOK_CLIENT_SECRET in .env.",
                remediation_url="/settings",
            )
        if not getattr(self.destination, "tiktok_open_id", None):
            raise TerminalPublishError(
                "TikTok account row missing tiktok_open_id — connect via OAuth first.",
                remediation_url="/settings",
            )
        publish_kind = getattr(job, "publish_kind", "video")
        if publish_kind != "short":
            # TikTok accepts long-form up to 10 min, but its CTR sweet
            # spot is firmly short-form. Worth refusing long-form by
            # default to surface the trade-off; can flip this when the
            # operator explicitly opts in.
            raise TerminalPublishError(
                "TikTok publisher refuses long-form by default — "
                "post the per-story shorts instead.",
                remediation_url="/settings",
            )
        return PrepareResult(
            title=(clip.seo_title_final or "")[:150],
            description=(clip.seo_description_final or "")[:2200],
            tags=getattr(clip, "seo_keywords", []) or [],
            hashtags=getattr(clip, "seo_hashtags", []) or [],
            privacy=(job.privacy_status or "private"),
            upload_path=source_video_path,
        )

    def publish(self, *, job, prepared: PrepareResult) -> PublishResult:
        raise TerminalPublishError(
            "TikTokPublisher is not yet wired. "
            "Pending: TikTokAccount model, TikTok OAuth router, "
            "Content Posting API helper. Note: Direct Post requires "
            "App Review (1-3 weeks); Upload-to-drafts is available "
            "immediately.",
            remediation_url="/settings",
        )

    def verify(self, *, job, published: PublishResult) -> VerifyResult:
        return VerifyResult(is_live=True)
