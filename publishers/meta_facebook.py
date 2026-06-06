"""Facebook Page publisher — posts a finished bulletin (or short) to a
connected FB Page via the Meta Graph API.

This is a scaffold module. The OAuth side (registering a Meta App,
exchanging short-lived tokens for long-lived Page tokens, storing them
in a ``MetaAccount`` table) is wired up separately by
``routers/meta_oauth.py``; this Publisher just does the actual posting
on top of an already-stored access token.

What you need from Meta before this can post a real video
─────────────────────────────────────────────────────────
1. Meta for Developers account → register an App.
2. Add the "Pages API" + "Instagram Graph API" products.
3. Production permissions (App Review takes 1–2 weeks):
     - ``pages_show_list``        — list which Pages the user owns
     - ``pages_read_engagement``  — read Page metadata
     - ``pages_manage_posts``     — post videos to the Page
     - ``pages_manage_metadata``  — required by Graph for video posts
     - ``business_management``    — multi-account
4. Test access tokens for development (the App stays in "dev mode"
   until review completes — only the App admin's own Pages can post).
5. Store the env vars:
     META_APP_ID, META_APP_SECRET, META_REDIRECT_URI

Format constraints handled here
───────────────────────────────
- Page video (long form): ≤240min, ≤4GB, ≥720p, 16:9 or 9:16. Our
  bulletin output fits.
- Page Reel (short form): ≤90s, 9:16, ≤4GB. Our per-story shorts fit.
The Publisher uses ``publish_kind`` on the job to pick the right
endpoint (``/videos`` vs ``/video_reels``).

Graph API endpoints
───────────────────
- Long video: POST /{page-id}/videos (multipart, resumable for >1GB)
- Reel:       POST /{page-id}/video_reels (3-step: start → upload → finish)
- Both:       returns {id} which is the FB video post id
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


META_GRAPH_API_VERSION = os.environ.get("META_GRAPH_API_VERSION", "v21.0")
META_GRAPH_BASE = f"https://graph.facebook.com/{META_GRAPH_API_VERSION}"


@register
class FacebookPagePublisher(Publisher):
    provider_key = "meta_fb"
    display_name = "Facebook Page"

    def prepare(self, *, job, clip, source_video_path: str) -> PrepareResult:
        dest = self.destination
        # destination is expected to be a MetaAccount row (defined when
        # we run the model migration). It carries a long-lived Page
        # token + Page id. If either is missing, fail terminally so the
        # operator can fix in /settings/meta.
        if not dest or not getattr(dest, "page_access_token_enc", ""):
            raise TerminalPublishError(
                "Facebook Page destination has no Page access token. "
                "Reconnect on /settings/meta.",
                remediation_url="/settings/meta",
            )
        if not getattr(dest, "fb_page_id", ""):
            raise TerminalPublishError(
                "Facebook Page destination missing fb_page_id.",
                remediation_url="/settings/meta",
            )

        # Format: bulletin → /videos (long form), short → /video_reels.
        publish_kind = getattr(job, "publish_kind", "video")
        endpoint_path = (
            "video_reels" if publish_kind == "short" else "videos"
        )

        return PrepareResult(
            title=(clip.seo_title_final or "")[:250],
            description=(clip.seo_description_final or "")[:5000],
            tags=getattr(clip, "seo_keywords", []) or [],
            hashtags=getattr(clip, "seo_hashtags", []) or [],
            privacy=(job.privacy_status or "private"),
            scheduled_at=getattr(job, "publish_at", None),
            thumbnail_path=getattr(clip, "thumb_path", "") or "",
            upload_path=source_video_path,
            extras={
                "endpoint_path": endpoint_path,
                "fb_page_id": dest.fb_page_id,
                "publish_kind": publish_kind,
            },
        )

    def publish(self, *, job, prepared: PrepareResult) -> PublishResult:
        """Skeleton — wires the contract; actual HTTP call lives in a
        ``_post_fb_video`` helper that we'll wire when MetaAccount
        model + OAuth router are in place.

        Until then this raises a clear, actionable terminal error so
        operator UI shows the right remediation link instead of a
        generic failure."""
        raise TerminalPublishError(
            "FacebookPagePublisher is not yet wired to the Graph API. "
            "Pending: register Meta App, run OAuth flow, populate "
            "MetaAccount row. See publishers/meta_facebook.py header.",
            remediation_url="/settings/meta",
        )

    def verify(self, *, job, published: PublishResult) -> VerifyResult:
        # Graph API returns a post id immediately; for Reels there's a
        # ~30s processing window. A future improvement: poll
        # GET /{video-id}?fields=status until status.video_status ==
        # "ready". For now we trust the publish response.
        return VerifyResult(is_live=True)
