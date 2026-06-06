"""X (Twitter) video publisher.

Posts a finished bulletin (or short clip) as a video tweet using X API
v2 + the legacy chunked-upload endpoint on api.twitter.com/2/media/upload.

What you need from X before this can post a real video
──────────────────────────────────────────────────────
1. Developer account: developer.twitter.com → sign up.
2. App in a Project with these access levels:
   - "Basic" tier ($100/month) — minimum for /tweets v2 POST.
   - OAuth 2.0 with PKCE OR OAuth 1.0a User Context (we use OAuth 2.0).
3. Scopes:
   - tweet.read
   - tweet.write
   - users.read
   - offline.access   (for refresh tokens)
4. .env:
   X_CLIENT_ID, X_CLIENT_SECRET, X_REDIRECT_URI

Format constraints
──────────────────
- Video: ≤140s for the free 1080p, ≤512 MB, MP4 H.264/AAC.
- Aspect: 16:9 best for Tweets, 9:16 for in-stream X video / Vertical.

Upload flow
───────────
1. POST /2/media/upload  command=INIT     → media_id_string
2. POST /2/media/upload  command=APPEND   → chunks (segment_index 0..n)
3. POST /2/media/upload  command=FINALIZE → start async processing
4. GET  /2/media/upload  command=STATUS   → poll until processing.state == "succeeded"
5. POST /2/tweets  body { text, media: { media_ids: [media_id_string] } }
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
class XTwitterPublisher(Publisher):
    provider_key = "x"
    display_name = "X (Twitter)"

    def prepare(self, *, job, clip, source_video_path: str) -> PrepareResult:
        if not (os.environ.get("X_CLIENT_ID") and os.environ.get("X_CLIENT_SECRET")):
            raise TerminalPublishError(
                "X publisher not configured — set X_CLIENT_ID + X_CLIENT_SECRET in .env.",
                remediation_url="/settings",
            )
        # destination expected to be an XAccount row (model TBD). For
        # now: clean refusal so the operator sees what's missing.
        if not getattr(self.destination, "x_user_id", None):
            raise TerminalPublishError(
                "X account row missing x_user_id — connect via OAuth first.",
                remediation_url="/settings",
            )
        return PrepareResult(
            title=(clip.seo_title_final or "")[:280],
            # Tweet body: hook + 280-char ceiling. We use the SEO hook
            # if available, fall back to title.
            description=(
                getattr(clip, "seo_hook", "")
                or clip.seo_title_final
                or ""
            )[:280],
            tags=getattr(clip, "seo_keywords", []) or [],
            hashtags=getattr(clip, "seo_hashtags", []) or [],
            privacy=(job.privacy_status or "private"),
            thumbnail_path=getattr(clip, "thumb_path", "") or "",
            upload_path=source_video_path,
        )

    def publish(self, *, job, prepared: PrepareResult) -> PublishResult:
        # Scaffold — wired the moment XAccount + OAuth router land.
        raise TerminalPublishError(
            "XTwitterPublisher is not yet wired. "
            "Pending: XAccount model, X OAuth router, chunked media upload helper.",
            remediation_url="/settings",
        )

    def verify(self, *, job, published: PublishResult) -> VerifyResult:
        return VerifyResult(is_live=True)
