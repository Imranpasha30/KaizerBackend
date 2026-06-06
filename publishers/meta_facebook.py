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
        """Two-phase upload:
        1. ``POST /{page-id}/videos`` (multipart) or
           ``POST /{page-id}/video_reels`` (3-step Reels flow).
        2. On accepted response, store the returned video/post id.

        The choice between /videos and /video_reels was made in
        prepare() — we just dispatch to the right helper here.
        """
        import os
        import httpx

        page_id = prepared.extras["fb_page_id"]
        endpoint_path = prepared.extras["endpoint_path"]
        from routers.meta_oauth import page_access_token_for, GRAPH_BASE
        token = page_access_token_for(self.db, self.destination.id)
        if not token:
            raise TerminalPublishError(
                "Could not decrypt Page access token — reconnect on /settings/meta",
                remediation_url="/settings/meta",
            )

        size = os.path.getsize(prepared.upload_path)
        if size > 4 * 1024 * 1024 * 1024:
            raise TerminalPublishError(
                f"File is {size / 1e9:.1f} GB — Meta's hard cap is 4 GB",
                remediation_url="/channels",
            )

        url = f"{GRAPH_BASE}/{page_id}/{endpoint_path}"
        # Caption: title + description + hashtag block. Matches what the
        # SEO contract already produces.
        caption_pieces = []
        if prepared.title:       caption_pieces.append(prepared.title)
        if prepared.description: caption_pieces.append(prepared.description)
        caption = "\n\n".join(caption_pieces)[:5000]

        try:
            with open(prepared.upload_path, "rb") as fh:
                with httpx.Client(timeout=300) as cx:
                    # For Reels there's a 3-step (start → upload → finish)
                    # flow Meta documents; for now we use the simpler
                    # one-shot multipart for both /videos and Reels and
                    # rely on Graph to switch under the hood. If Reels
                    # rejects the one-shot we'll swap to the 3-step path
                    # in a follow-up.
                    r = cx.post(
                        url,
                        files={"source": (os.path.basename(prepared.upload_path), fh, "video/mp4")},
                        data={
                            "access_token": token,
                            "description": caption,
                            "published": "true",
                        },
                    )
        except httpx.RequestError as e:
            raise TransientPublishError(f"Network error talking to Meta: {e}") from e

        if r.status_code >= 400:
            try:
                body = r.json()
            except Exception:
                body = {"raw": r.text[:500]}
            # Meta returns 400 + error.code = 200 for "Permission denied"
            # which is terminal until the operator reconnects with
            # broader scopes / passes App Review.
            err = (body.get("error") or {})
            meta_code = err.get("code")
            meta_subcode = err.get("error_subcode")
            msg = err.get("message") or str(body)
            if meta_code in (190, 102):   # OAuth invalid / session expired
                raise TerminalPublishError(
                    f"Meta auth invalid (code {meta_code}): {msg}",
                    remediation_url="/settings/meta",
                    upstream_status=r.status_code,
                    upstream_body=str(body)[:500],
                )
            if meta_code == 200:   # Permission denied
                raise TerminalPublishError(
                    f"Meta permission missing (code 200 subcode {meta_subcode}): {msg}",
                    remediation_url="/settings/meta",
                    upstream_status=r.status_code,
                    upstream_body=str(body)[:500],
                )
            if 500 <= r.status_code < 600:
                raise TransientPublishError(
                    f"Meta {r.status_code}: {msg}",
                    upstream_status=r.status_code,
                    upstream_body=str(body)[:500],
                )
            raise TransientPublishError(
                f"Meta {r.status_code}: {msg}",
                upstream_status=r.status_code,
                upstream_body=str(body)[:500],
            )

        body = r.json()
        video_id = body.get("id") or body.get("post_id") or ""
        if not video_id:
            raise TransientPublishError(
                f"Meta returned 200 but no id: {body}",
            )

        # Bump the Page's publish counter for the dashboard.
        self.destination.last_publish_at = datetime.now(timezone.utc)
        today = datetime.now(timezone.utc).date()
        last = (self.destination.publishes_today_at or
                datetime(2000, 1, 1, tzinfo=timezone.utc)).date()
        if today != last:
            self.destination.publishes_today = 0
            self.destination.publishes_today_at = datetime.now(timezone.utc)
        self.destination.publishes_today = (self.destination.publishes_today or 0) + 1
        self.db.commit()

        public_url = f"https://www.facebook.com/{video_id}"
        return PublishResult(
            external_id=video_id,
            public_url=public_url,
            accepted_at=datetime.now(timezone.utc),
            published_at=datetime.now(timezone.utc),
            raw_response=str(body)[:1000],
        )

    def verify(self, *, job, published: PublishResult) -> VerifyResult:
        # Graph API returns a post id immediately; for Reels there's a
        # ~30s processing window. A future improvement: poll
        # GET /{video-id}?fields=status until status.video_status ==
        # "ready". For now we trust the publish response.
        return VerifyResult(is_live=True)
