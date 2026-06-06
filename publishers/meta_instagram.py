"""Instagram publisher — posts Reels (9:16, ≤90s) to a connected IG
Business or Creator account via the Meta Graph API.

IG publishing is gated through the Page the IG account is linked to.
You CANNOT post to an IG account that isn't linked to a Page — that's
a hard Meta restriction. The MetaAccount model will carry both
``fb_page_id`` and ``ig_user_id`` for that reason.

What you need from Meta beyond the Facebook side
────────────────────────────────────────────────
- ``instagram_basic``           — discover the linked IG account
- ``instagram_content_publish`` — post to it (gated behind App Review)
- The IG account MUST be a Business or Creator account (personal
  accounts cannot be posted to via the API).

Publishing flow (Reels are the only video type the public API allows
posting via /media_publish today; Feed posts and Stories are limited):
1. POST /{ig-user-id}/media
     media_type=REELS, video_url=<public URL or upload session id>,
     caption=...
   → returns {id} of a "container"
2. Poll GET /{container-id}?fields=status_code until FINISHED
3. POST /{ig-user-id}/media_publish?creation_id={container-id}
   → returns {id} which is the IG media id

Constraint: the video_url must be publicly fetchable by Meta's
crawler. For client-side uploads, Meta exposes a Resumable Upload API
that returns an upload session id you can pass as media_url. Either
way, the file must be on the public internet briefly — our backend
serves /media/... so the bulletin URL works if the deployment is
public.
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
class InstagramReelsPublisher(Publisher):
    provider_key = "meta_ig"
    display_name = "Instagram Reels"

    def prepare(self, *, job, clip, source_video_path: str) -> PrepareResult:
        dest = self.destination
        if not dest or not getattr(dest, "page_access_token_enc", ""):
            raise TerminalPublishError(
                "Instagram destination has no Page access token.",
                remediation_url="/settings/meta",
            )
        if not getattr(dest, "ig_user_id", ""):
            raise TerminalPublishError(
                "Instagram destination missing ig_user_id — the IG "
                "account must be a Business/Creator account linked to "
                "a Page that the operator owns.",
                remediation_url="/settings/meta",
            )

        # Instagram public API only allows posting REELS via
        # /media_publish today. Force publish_kind=short and let the
        # operator see a clear refusal if they try to push a long-form
        # bulletin to IG (which is the right answer — IG isn't a
        # long-form surface).
        publish_kind = getattr(job, "publish_kind", "video")
        if publish_kind != "short":
            raise TerminalPublishError(
                "Instagram destinations only accept short-form "
                "(publish_kind=short). The bulletin can't be pushed "
                "to IG; only the individual shorts can.",
                remediation_url="/settings/meta",
            )

        return PrepareResult(
            title=(clip.seo_title_final or "")[:80],
            description=(clip.seo_description_final or "")[:2200],
            tags=getattr(clip, "seo_keywords", []) or [],
            hashtags=getattr(clip, "seo_hashtags", []) or [],
            privacy=(job.privacy_status or "private"),
            scheduled_at=getattr(job, "publish_at", None),
            thumbnail_path=getattr(clip, "thumb_path", "") or "",
            upload_path=source_video_path,
            extras={
                "ig_user_id": dest.ig_user_id,
                "publish_kind": publish_kind,
            },
        )

    def publish(self, *, job, prepared: PrepareResult) -> PublishResult:
        # Scaffold — see header for the 3-step flow. Wired when
        # MetaAccount model + Resumable Upload helper are in place.
        raise TerminalPublishError(
            "InstagramReelsPublisher is not yet wired to the Graph API. "
            "Pending: MetaAccount migration, OAuth flow, Resumable "
            "Upload helper. See publishers/meta_instagram.py header.",
            remediation_url="/settings/meta",
        )

    def verify(self, *, job, published: PublishResult) -> VerifyResult:
        return VerifyResult(is_live=True)
