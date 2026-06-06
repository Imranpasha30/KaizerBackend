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
        """3-step Reels flow:
        1. POST /{ig-user-id}/media — creates a container, takes
           video_url + caption + media_type=REELS.
        2. Poll GET /{container-id}?fields=status_code until FINISHED.
        3. POST /{ig-user-id}/media_publish?creation_id=<container>.

        The video_url has to be publicly fetchable by Meta's crawler.
        For dev-local this won't work; for production deploys behind
        the cloudflared tunnel (your test.kaizerx.com) it does.
        """
        import time
        import httpx
        from routers.meta_oauth import page_access_token_for, GRAPH_BASE

        ig_user_id = prepared.extras["ig_user_id"]
        token = page_access_token_for(self.db, self.destination.id)
        if not token:
            raise TerminalPublishError(
                "Could not decrypt Page access token — reconnect on /settings/meta",
                remediation_url="/settings/meta",
            )

        # We need a public URL for the video. The worker hands us a
        # local path in source_video_path, so we resolve it to the
        # /media/... URL the FastAPI app already serves. Operator
        # deployment MUST be reachable from the public internet for
        # Meta's crawler — Meta will reject loopback / private IPs.
        public_base = (
            os.environ.get("KAIZER_PUBLIC_BASE_URL")
            or os.environ.get("PUBLIC_BASE_URL")
            or ""
        ).rstrip("/")
        if not public_base:
            raise TerminalPublishError(
                "KAIZER_PUBLIC_BASE_URL is not set — IG Reels can't be "
                "posted without a public video_url for Meta to fetch.",
                remediation_url="/settings/meta",
            )
        # The upload_path is an absolute filesystem path inside the
        # backend's output/ tree; the /media/ static mount serves
        # everything under output/. Derive the relative URL.
        from pathlib import Path
        from main import BASE_DIR
        try:
            rel = Path(prepared.upload_path).resolve().relative_to(
                (BASE_DIR / "output").resolve()
            )
        except (ValueError, OSError):
            raise TerminalPublishError(
                f"Upload path {prepared.upload_path!r} not under /output — "
                "can't build a public URL for Meta.",
                remediation_url="/settings/meta",
            )
        video_url = f"{public_base}/media/{rel.as_posix()}"

        # Caption is what shows under the Reel — title + description.
        caption_pieces = []
        if prepared.title:       caption_pieces.append(prepared.title)
        if prepared.description: caption_pieces.append(prepared.description)
        caption = "\n\n".join(caption_pieces)[:2200]

        # ── Step 1: create container ──
        try:
            with httpx.Client(timeout=120) as cx:
                r1 = cx.post(
                    f"{GRAPH_BASE}/{ig_user_id}/media",
                    data={
                        "access_token": token,
                        "media_type": "REELS",
                        "video_url": video_url,
                        "caption": caption,
                    },
                )
        except httpx.RequestError as e:
            raise TransientPublishError(f"Network error to Meta: {e}") from e
        if r1.status_code >= 400:
            self._raise_for_meta(r1, "create container")
        container_id = (r1.json().get("id") or "")
        if not container_id:
            raise TransientPublishError(f"Meta returned 200 but no id: {r1.json()}")

        # ── Step 2: poll until FINISHED ──
        # Meta says Reels containers typically finish in <30s; we poll
        # every 5s for up to 3 minutes. ERROR / EXPIRED is terminal.
        deadline = time.time() + 180
        last_status = ""
        while time.time() < deadline:
            try:
                with httpx.Client(timeout=30) as cx:
                    rs = cx.get(
                        f"{GRAPH_BASE}/{container_id}",
                        params={
                            "access_token": token,
                            "fields": "status_code,status",
                        },
                    )
            except httpx.RequestError as e:
                raise TransientPublishError(f"Network error polling: {e}") from e
            if rs.status_code >= 400:
                self._raise_for_meta(rs, "poll container")
            body = rs.json()
            last_status = body.get("status_code", "") or body.get("status", "")
            if last_status == "FINISHED":
                break
            if last_status in ("ERROR", "EXPIRED"):
                raise TerminalPublishError(
                    f"Meta container failed: {last_status} — {body}",
                    remediation_url="/settings/meta",
                )
            time.sleep(5)
        else:
            raise TransientPublishError(
                f"Meta container did not finish in 3 min (last status: {last_status!r})"
            )

        # ── Step 3: publish ──
        try:
            with httpx.Client(timeout=120) as cx:
                r3 = cx.post(
                    f"{GRAPH_BASE}/{ig_user_id}/media_publish",
                    data={
                        "access_token": token,
                        "creation_id": container_id,
                    },
                )
        except httpx.RequestError as e:
            raise TransientPublishError(f"Network error publishing: {e}") from e
        if r3.status_code >= 400:
            self._raise_for_meta(r3, "media_publish")
        body3 = r3.json()
        media_id = body3.get("id") or ""
        if not media_id:
            raise TransientPublishError(f"Meta returned 200 but no media id: {body3}")

        # Counter bump (same shape as the FB publisher).
        self.destination.last_publish_at = datetime.now(timezone.utc)
        today = datetime.now(timezone.utc).date()
        last = (self.destination.publishes_today_at or
                datetime(2000, 1, 1, tzinfo=timezone.utc)).date()
        if today != last:
            self.destination.publishes_today = 0
            self.destination.publishes_today_at = datetime.now(timezone.utc)
        self.destination.publishes_today = (self.destination.publishes_today or 0) + 1
        self.db.commit()

        return PublishResult(
            external_id=media_id,
            public_url=f"https://www.instagram.com/reel/{media_id}/",
            accepted_at=datetime.now(timezone.utc),
            published_at=datetime.now(timezone.utc),
            raw_response=str(body3)[:1000],
        )

    def _raise_for_meta(self, r, what):
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text[:500]}
        err = (body.get("error") or {})
        meta_code = err.get("code")
        meta_subcode = err.get("error_subcode")
        msg = err.get("message") or str(body)
        if meta_code in (190, 102):
            raise TerminalPublishError(
                f"Meta auth invalid during {what} (code {meta_code}): {msg}",
                remediation_url="/settings/meta",
                upstream_status=r.status_code,
                upstream_body=str(body)[:500],
            )
        if meta_code == 200:
            raise TerminalPublishError(
                f"Meta permission missing during {what} "
                f"(code 200 subcode {meta_subcode}): {msg}",
                remediation_url="/settings/meta",
                upstream_status=r.status_code,
                upstream_body=str(body)[:500],
            )
        if 500 <= r.status_code < 600:
            raise TransientPublishError(
                f"Meta {r.status_code} during {what}: {msg}",
                upstream_status=r.status_code,
                upstream_body=str(body)[:500],
            )
        raise TransientPublishError(
            f"Meta {r.status_code} during {what}: {msg}",
            upstream_status=r.status_code,
            upstream_body=str(body)[:500],
        )

    def verify(self, *, job, published: PublishResult) -> VerifyResult:
        return VerifyResult(is_live=True)
