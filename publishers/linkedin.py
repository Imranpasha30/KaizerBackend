"""LinkedIn video publisher.

Posts a finished long-form bulletin to a LinkedIn personal profile or
Company Page via the LinkedIn API (Posts + Media Upload).

What you need from LinkedIn before this can post a real video
─────────────────────────────────────────────────────────────
1. Developer account: developer.linkedin.com → create an App.
2. Request "Marketing Developer Platform" access (for video) or
   "Share on LinkedIn" (basic share). Marketing Developer Platform
   gives you access to Company Pages.
3. Scopes:
   - w_member_social      (post to personal profile)
   - w_organization_social (post to Company Pages — requires verification)
   - r_organization_social (read responses)
4. .env:
   LINKEDIN_CLIENT_ID, LINKEDIN_CLIENT_SECRET, LINKEDIN_REDIRECT_URI

Format constraints
──────────────────
- Video: ≤10 min for personal, ≤30 min for Company Pages.
- Aspect: 1:1 / 16:9 / 9:16 all accepted.
- Size: ≤5 GB (very generous — bulletins always fit).

Upload flow
───────────
1. POST /rest/assets?action=registerUpload  → upload URL + asset URN.
2. PUT  <upload_url>                         → upload bytes.
3. POST /rest/posts                          → create the post with
                                                the asset URN.

LinkedIn is the simplest of the bunch — single PUT for the bytes,
no chunking required for our size range.
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
class LinkedInPublisher(Publisher):
    provider_key = "linkedin"
    display_name = "LinkedIn"

    def prepare(self, *, job, clip, source_video_path: str) -> PrepareResult:
        if not (
            os.environ.get("LINKEDIN_CLIENT_ID")
            and os.environ.get("LINKEDIN_CLIENT_SECRET")
        ):
            raise TerminalPublishError(
                "LinkedIn publisher not configured — set LINKEDIN_CLIENT_ID + LINKEDIN_CLIENT_SECRET in .env.",
                remediation_url="/settings",
            )
        if not getattr(self.destination, "linkedin_urn", None):
            raise TerminalPublishError(
                "LinkedIn account row missing linkedin_urn — connect via OAuth first.",
                remediation_url="/settings",
            )
        return PrepareResult(
            title=(clip.seo_title_final or "")[:200],
            description=(clip.seo_description_final or "")[:3000],
            tags=getattr(clip, "seo_keywords", []) or [],
            hashtags=getattr(clip, "seo_hashtags", []) or [],
            privacy=(job.privacy_status or "private"),
            upload_path=source_video_path,
        )

    def publish(self, *, job, prepared: PrepareResult) -> PublishResult:
        raise TerminalPublishError(
            "LinkedInPublisher is not yet wired. "
            "Pending: LinkedInAccount model, LinkedIn OAuth router, asset registration helper.",
            remediation_url="/settings",
        )

    def verify(self, *, job, published: PublishResult) -> VerifyResult:
        return VerifyResult(is_live=True)
