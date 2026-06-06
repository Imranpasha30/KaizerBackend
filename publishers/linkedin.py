"""LinkedIn video publisher — production implementation.

Posts a finished bulletin to a LinkedIn personal profile via the
LinkedIn Posts + Assets APIs. Personal profile posts use
w_member_social; Company Page posts use w_organization_social
(requires Marketing Developer Platform access — left as future work).

Upload flow (3 steps)
─────────────────────
1. POST /v2/assets?action=registerUpload
     - registers an upload, returns an uploadUrl + asset URN
2. POST <uploadUrl> with the raw video bytes
     - single PUT for files up to ~5GB (LinkedIn handles chunking
       server-side for the simple endpoint we use)
3. POST /rest/posts
     - creates the public post referencing the asset URN

Reference: https://learn.microsoft.com/en-us/linkedin/marketing/integrations/community-management/shares/videos-api
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import httpx

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

    API_BASE = "https://api.linkedin.com/v2"
    REST_BASE = "https://api.linkedin.com/rest"

    def prepare(self, *, job, clip, source_video_path: str) -> PrepareResult:
        if not (
            os.environ.get("LINKEDIN_CLIENT_ID")
            and os.environ.get("LINKEDIN_CLIENT_SECRET")
        ):
            raise TerminalPublishError(
                "LinkedIn publisher not configured — set LINKEDIN_CLIENT_ID + LINKEDIN_CLIENT_SECRET in .env.",
                remediation_url="/settings/social",
            )
        dest = self.destination
        if not dest or not getattr(dest, "linkedin_urn", None):
            raise TerminalPublishError(
                "LinkedIn account row missing linkedin_urn — connect via OAuth first.",
                remediation_url="/settings/social",
            )

        # Compose post text — title + description, capped at LinkedIn's
        # 3000-char post limit.
        chunks = []
        if clip.seo_title_final:       chunks.append(clip.seo_title_final)
        if clip.seo_description_final: chunks.append(clip.seo_description_final)
        body = "\n\n".join(chunks)[:3000]

        return PrepareResult(
            title=(clip.seo_title_final or "")[:200],
            description=body,
            tags=getattr(clip, "seo_keywords", []) or [],
            hashtags=getattr(clip, "seo_hashtags", []) or [],
            privacy=(job.privacy_status or "public"),
            upload_path=source_video_path,
            extras={"author_urn": dest.linkedin_urn},
        )

    def publish(self, *, job, prepared: PrepareResult) -> PublishResult:
        from routers.linkedin_oauth import access_token_for
        token = access_token_for(self.db, self.destination.id)
        if not token:
            raise TerminalPublishError(
                "Could not decrypt access token — reconnect on /settings/social",
                remediation_url="/settings/social",
            )

        author_urn = prepared.extras["author_urn"]
        size = os.path.getsize(prepared.upload_path)
        if size > 5 * 1024 * 1024 * 1024:
            raise TerminalPublishError(
                f"File is {size / 1e9:.1f} GB — LinkedIn cap is 5 GB",
                remediation_url="/channels",
            )

        # ── Step 1: register the upload ──
        register_body = {
            "registerUploadRequest": {
                "recipes": ["urn:li:digitalmediaRecipe:feedshare-video"],
                "owner": author_urn,
                "serviceRelationships": [{
                    "relationshipType": "OWNER",
                    "identifier": "urn:li:userGeneratedContent",
                }],
            }
        }
        try:
            with httpx.Client(timeout=60) as cx:
                r1 = cx.post(
                    f"{self.API_BASE}/assets?action=registerUpload",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                        "X-Restli-Protocol-Version": "2.0.0",
                    },
                    json=register_body,
                )
        except httpx.RequestError as e:
            raise TransientPublishError(f"Network error to LinkedIn: {e}") from e

        if r1.status_code >= 400:
            self._raise_for_linkedin(r1, "registerUpload")
        rdata = r1.json()
        upload_url = (
            rdata.get("value", {})
                 .get("uploadMechanism", {})
                 .get("com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest", {})
                 .get("uploadUrl")
        )
        asset_urn = rdata.get("value", {}).get("asset")
        if not (upload_url and asset_urn):
            raise TransientPublishError(
                f"registerUpload returned unexpected shape: {rdata}"
            )

        # ── Step 2: upload bytes ──
        try:
            with open(prepared.upload_path, "rb") as fh:
                with httpx.Client(timeout=600) as cx:
                    r2 = cx.put(
                        upload_url,
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Content-Type": "application/octet-stream",
                        },
                        content=fh.read(),
                    )
        except httpx.RequestError as e:
            raise TransientPublishError(f"Network error during upload: {e}") from e

        if r2.status_code >= 400:
            self._raise_for_linkedin(r2, "asset upload")

        # ── Step 3: create the post ──
        # Posts API requires the LinkedIn-Version header pinned to a
        # specific YYYYMM. May 2024 is the version we built against.
        post_body = {
            "author": author_urn,
            "commentary": prepared.description,
            "visibility": "PUBLIC" if prepared.privacy == "public" else "CONNECTIONS",
            "distribution": {
                "feedDistribution": "MAIN_FEED",
                "targetEntities": [],
                "thirdPartyDistributionChannels": [],
            },
            "content": {
                "media": {
                    "id": asset_urn,
                    "title": prepared.title,
                },
            },
            "lifecycleState": "PUBLISHED",
            "isReshareDisabledByAuthor": False,
        }
        try:
            with httpx.Client(timeout=120) as cx:
                r3 = cx.post(
                    f"{self.REST_BASE}/posts",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                        "LinkedIn-Version": "202405",
                        "X-Restli-Protocol-Version": "2.0.0",
                    },
                    json=post_body,
                )
        except httpx.RequestError as e:
            raise TransientPublishError(f"Network error creating post: {e}") from e

        if r3.status_code >= 400:
            self._raise_for_linkedin(r3, "posts create")

        # LinkedIn returns the post URN in the X-RestLi-Id header for
        # POST /rest/posts (no body on success).
        post_urn = r3.headers.get("X-RestLi-Id") or r3.headers.get("x-restli-id") or ""
        if not post_urn:
            # Some flows return JSON with {id: "..."}.
            try:
                post_urn = r3.json().get("id") or ""
            except Exception:
                post_urn = ""

        # Bump per-account counter for the dashboard.
        self.destination.last_publish_at = datetime.now(timezone.utc)
        today = datetime.now(timezone.utc).date()
        last = (self.destination.publishes_today_at or
                datetime(2000, 1, 1, tzinfo=timezone.utc)).date()
        if today != last:
            self.destination.publishes_today = 0
            self.destination.publishes_today_at = datetime.now(timezone.utc)
        self.destination.publishes_today = (self.destination.publishes_today or 0) + 1
        self.db.commit()

        # Construct the viewer URL — for a post urn:li:ugcPost:1234567,
        # the URL is https://www.linkedin.com/feed/update/<urn>/
        return PublishResult(
            external_id=post_urn or asset_urn,
            public_url=(
                f"https://www.linkedin.com/feed/update/{post_urn}/"
                if post_urn else ""
            ),
            accepted_at=datetime.now(timezone.utc),
            published_at=datetime.now(timezone.utc),
            raw_response=str(post_urn)[:500],
        )

    def verify(self, *, job, published: PublishResult) -> VerifyResult:
        return VerifyResult(is_live=True)

    def _raise_for_linkedin(self, r, what):
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text[:500]}
        msg = body.get("message") or body.get("error") or str(body)
        # 401 + invalid token = terminal
        if r.status_code == 401:
            raise TerminalPublishError(
                f"LinkedIn auth invalid during {what}: {msg}",
                remediation_url="/settings/social",
                upstream_status=r.status_code,
                upstream_body=str(body)[:500],
            )
        if r.status_code == 403:
            raise TerminalPublishError(
                f"LinkedIn permission missing during {what}: {msg}",
                remediation_url="/settings/social",
                upstream_status=r.status_code,
                upstream_body=str(body)[:500],
            )
        if 500 <= r.status_code < 600:
            raise TransientPublishError(
                f"LinkedIn {r.status_code} during {what}: {msg}",
                upstream_status=r.status_code,
                upstream_body=str(body)[:500],
            )
        raise TransientPublishError(
            f"LinkedIn {r.status_code} during {what}: {msg}",
            upstream_status=r.status_code,
            upstream_body=str(body)[:500],
        )
