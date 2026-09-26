"""The Help Centre: guides that live in R2, not in the applications.

WHY THE FILES ARE NOT IN THE REPO. A guide bundled into the desktop app is a
guide you can only correct by shipping a new 4 GB build -- fifty minutes, and
the installer grows for ever. In object storage the same file is one upload,
both products read one URL, and adding next month's guide costs no build at
all. The three here are 15.5 MB; a help centre that earns its name is an order
of magnitude more within a year.

WHY THE ASSETS ARE SERVED THROUGH THIS ROUTER RATHER THAN A PUBLIC BUCKET.
Cloudflare will hand out a public ``*.r2.dev`` URL for a bucket, and
R2_PUBLIC_BASE_URL is set to one -- but it answers 403 because public access is
off, and it should STAY off: this is the same bucket that holds customers'
uploaded video. Turning it public to publish three help files would make every
object in it readable by anyone who can guess a key. So the bucket stays
private and this router mints a short-lived signed URL per request and
redirects to it, which is the pattern main.py already uses for rendered clips.
Cloudflare still does the byte-shovelling; the redirect carries range requests,
so a video seeks normally.

WHY THE ASSET URLS ARE ABSOLUTE. The desktop runs its own local engine, which
has no R2 credentials -- it could serve this catalogue but could never sign an
asset URL. So the catalogue points assets at the HOSTED api instead
(KAIZER_HELP_ASSET_BASE), and the same JSON is correct in a browser and in the
app. A <video src> and an <a href> are ordinary navigations, so no CORS
preflight and no IPC bridge is involved.

ADDING A GUIDE is one entry in GUIDES below plus an upload to the matching key.
No frontend change, and nothing to rebuild on the desktop.
"""
from __future__ import annotations

import os
import pathlib
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/help", tags=["help"])


# ── the catalogue ─────────────────────────────────────────────────────
# `key` is the object in R2. `id` is what the URL exposes, so the storage
# layout can be reorganised without breaking links people have open.
GUIDES: List[dict] = [
    {
        "id": "live-studio",
        "title": "Connect a channel and go live",
        "summary": (
            "Link a YouTube channel to Kaizer X, set the stream up, and take it "
            "live — the whole path, start to finish."
        ),
        "topic": "Live Studio",
        "order": 10,
        "assets": [
            {
                "id": "live-studio-video-te",
                "kind": "video",
                "language": "te",
                "language_label": "తెలుగు",
                "label": "Watch the walkthrough",
                "key": "help/live-studio/connect-and-go-live-te.mp4",
                "content_type": "video/mp4",
                # Sizes are display hints ("10.1 MB" next to the link).
                # MEASURED from the object in R2, not from the local file:
                # the two differ, and the browser fetches the former.
                # scratchpad/probe_help_routes.py re-checks them.
                "bytes": 10_637_215,
            },
            {
                "id": "live-studio-pdf-en",
                "kind": "pdf",
                "language": "en",
                "language_label": "English",
                "label": "Read the guide",
                "key": "help/live-studio/live-studio-guide-en.pdf",
                "content_type": "application/pdf",
                "bytes": 1_304_858,
            },
            {
                "id": "live-studio-pdf-te",
                "kind": "pdf",
                "language": "te",
                "language_label": "తెలుగు",
                "label": "గైడ్ చదవండి",
                "key": "help/live-studio/live-studio-guide-te.pdf",
                "content_type": "application/pdf",
                "bytes": 4_266_272,
            },
        ],
    },
]


def _asset_base() -> str:
    """Where a browser should fetch help assets from.

    The desktop's local engine has no R2 credentials, so it can serve this
    catalogue but cannot sign a URL. Pointing assets at the hosted API makes
    one catalogue correct in both products. Overridable for a self-hosted
    deployment; trailing slash trimmed so joins never double up.
    """
    # One name, not two. KAIZER_API_URL is the DESKTOP SHELL's variable --
    # main.js reads it to find the cloud -- and honouring it here would mean a
    # backend silently changing where it points because of a setting that
    # belongs to a different process. Unset is the correct behaviour: the
    # hosted API is where these assets are, which is why this needs no writer.
    base = os.getenv("KAIZER_HELP_ASSET_BASE") or "https://api.kaizerx.com"
    return base.strip().rstrip("/")


def _find_asset(asset_id: str) -> Optional[dict]:
    for guide in GUIDES:
        for asset in guide["assets"]:
            if asset["id"] == asset_id:
                return asset
    return None


class HelpAsset(BaseModel):
    id: str
    kind: str
    language: str
    language_label: str
    label: str
    url: str
    content_type: str
    bytes: int


class HelpGuide(BaseModel):
    id: str
    title: str
    summary: str
    topic: str
    assets: List[HelpAsset]


@router.get("/guides", response_model=List[HelpGuide])
def list_guides():
    """Every guide, with a ready-to-use URL per asset.

    Deliberately unauthenticated: this is documentation, the same material a
    prospective customer is shown, and requiring a session would keep it out of
    the sign-in screen and the landing page where it is most useful.
    """
    base = _asset_base()
    out = []
    for guide in sorted(GUIDES, key=lambda g: g.get("order", 0)):
        out.append({
            "id": guide["id"],
            "title": guide["title"],
            "summary": guide["summary"],
            "topic": guide["topic"],
            "assets": [
                {
                    "id": a["id"],
                    "kind": a["kind"],
                    "language": a["language"],
                    "language_label": a["language_label"],
                    "label": a["label"],
                    "url": f"{base}/api/help/asset/{a['id']}",
                    "content_type": a["content_type"],
                    "bytes": a["bytes"],
                }
                for a in guide["assets"]
            ],
        })
    return out


@router.get("/asset/{asset_id}")
def get_asset(asset_id: str, request: Request, download: bool = False):
    """Stream one help asset from storage, through this server.

    Deliberately NOT a redirect. A 302 would hand the reader a signed URL
    carrying the storage host and an AWS signature, and following it replaces
    the address bar -- so a person reading a guide ends up looking at
    infrastructure, and a PDF opens outside the application instead of in it.
    Streaming keeps the URL ours, which is also what makes the in-app viewer
    possible: an <iframe> at /api/help/asset/<id> is same-origin, an <iframe>
    at an opaque storage host is not.

    Range requests are forwarded to storage and answered 206, because without
    them a video cannot be seeked and has to arrive whole before it will play.
    """
    asset = _find_asset(asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="No such help asset.")

    rng = request.headers.get("range", "")

    try:
        from pipeline_core.storage import get_storage_provider
        backend = (os.getenv("STORAGE_BACKEND", "local") or "local").strip().lower()
        # Help assets live in R2 even when this deployment keeps its renders
        # locally, so ask for that provider by name when credentials exist.
        if os.getenv("R2_BUCKET") and os.getenv("R2_ACCESS_KEY_ID"):
            backend = "r2"
        provider = get_storage_provider(backend)
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail="The help library is unavailable right now.",
        ) from exc

    # ── R2 / S3: stream the object, forwarding any byte range ──────────
    client = getattr(provider, "_get_client", None)
    if callable(client):
        try:
            extra = {"Range": rng} if rng else {}
            obj = provider._get_client().get_object(
                Bucket=provider.bucket, Key=provider._k(asset["key"]), **extra)
        except Exception as exc:                               # noqa: BLE001
            # A key that is missing, or a range past the end.
            raise HTTPException(
                status_code=404,
                detail="That guide is not in the library.") from exc

        body = obj["Body"]
        headers = {
            "Accept-Ranges": "bytes",
            # Guides change rarely and are not secret; letting the browser keep
            # one means re-opening the help page does not re-download 10 MB.
            "Cache-Control": "public, max-age=3600",
        }
        if obj.get("ContentRange"):
            headers["Content-Range"] = obj["ContentRange"]
        if obj.get("ContentLength") is not None:
            headers["Content-Length"] = str(obj["ContentLength"])
        if download:
            name = asset["key"].rsplit("/", 1)[-1]
            headers["Content-Disposition"] = f'attachment; filename="{name}"'
        else:
            # inline, so the browser's own PDF viewer renders it in the frame
            headers["Content-Disposition"] = "inline"

        def _chunks(stream, size=256 * 1024):
            try:
                while True:
                    part = stream.read(size)
                    if not part:
                        break
                    yield part
            finally:
                try:
                    stream.close()
                except Exception:                              # noqa: BLE001
                    pass

        return StreamingResponse(
            _chunks(body),
            status_code=206 if obj.get("ContentRange") else 200,
            media_type=asset["content_type"],
            headers=headers,
        )

    # ── local provider: serve the file if this install happens to have it ──
    try:
        local = provider.ensure_local(asset["key"])
        data = pathlib.Path(local).read_bytes()
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(
            status_code=404,
            detail="That guide is not in this installation's library.") from exc
    return Response(content=data, media_type=asset["content_type"],
                    headers={"Accept-Ranges": "bytes",
                             "Content-Disposition": "inline"})
