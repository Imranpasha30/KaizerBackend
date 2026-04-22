"""
kaizer.routers.editor
======================
Wave 2A — Pro Editor Beta HTTP router.

Exposes three endpoints:

  GET  /api/editor/styles
       Public catalogue of all StylePack objects. No auth required.

  POST /api/editor/render-beta
       Apply a StylePack to a Clip, producing a polished "beta" MP4.
       Runs synchronously (v1). Caches the result in latest.json on disk.
       TODO (Phase 4+): move render to a background task / job queue.

  GET  /api/editor/render-beta/{clip_id}
       Return the most recently cached BetaRenderResult for a clip (from
       latest.json), or 404 when no render has been run yet.

URL convention
--------------
  _url fields are built from the beta/current absolute paths by stripping
  the ``<BASE_DIR>/output/`` prefix and prepending ``/media/``.
  The frontend can serve those via the ``/media`` static-files mount added
  to main.py (served from ``<BASE_DIR>/output``).

Authentication
--------------
  POST /api/editor/render-beta requires authentication via the standard
  JWT Bearer pattern (auth.current_user).  The GET catalogue and cache
  lookup endpoints are intentionally unauthenticated.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

import auth
import models
from database import get_db
from pipeline_core.effects.style_packs import (
    STYLE_PACKS,
    get_style_pack,
    list_style_packs,
)
from pipeline_core.editor_pro import BetaRenderResult, render_beta
from pipeline_core.storage import get_storage_provider

logger = logging.getLogger("kaizer.routers.editor")

router = APIRouter(prefix="/api/editor", tags=["editor"])

# ── Directory constants ───────────────────────────────────────────────────────

# Backend root: parent of routers/
BASE_DIR: Path = Path(__file__).resolve().parent.parent
OUTPUT_DIR: Path = BASE_DIR / "output"
BETA_RENDERS_ROOT: Path = OUTPUT_DIR / "beta_renders"


# ── Pydantic schemas ──────────────────────────────────────────────────────────


class StyleSchema(BaseModel):
    """Serialisable representation of a StylePack for the API catalogue."""

    name: str
    label: str
    description: str
    transition: str
    color_preset: str
    motion: Optional[str]
    text_animation: str
    caption_animation: str


class RenderBetaRequest(BaseModel):
    """Request body for POST /api/editor/render-beta."""

    clip_id: int
    style_pack: str
    hook_text: Optional[str] = None
    platform: str = "youtube_short"


class RenderBetaResponse(BaseModel):
    """Response payload for a successful beta render."""

    clip_id: int
    current_path: str
    current_url: str
    beta_path: str
    beta_url: str
    style_pack: str
    effects_applied: list[str]
    render_time_s: float
    qa_ok: bool
    warnings: list[str]


# ── Internal helpers ──────────────────────────────────────────────────────────


def _path_to_url(abs_path: str) -> str:
    """Convert an absolute path under BASE_DIR/output/ to a /media/ URL.

    Example:
        .../KaizerBackend/output/beta_renders/clip_7/foo_beta_cinematic.mp4
        → /media/beta_renders/clip_7/foo_beta_cinematic.mp4
    """
    try:
        rel = Path(abs_path).resolve().relative_to(OUTPUT_DIR.resolve())
        return "/media/" + rel.as_posix()
    except ValueError:
        # abs_path is not under OUTPUT_DIR — return a fallback API-file URL
        logger.warning(
            "editor: path %r is outside OUTPUT_DIR; using /api/file/ fallback",
            abs_path,
        )
        return f"/api/file/?path={abs_path}"


def _clip_output_dir(clip_id: int) -> Path:
    """Return (and ensure) the per-clip beta-render output directory."""
    d = BETA_RENDERS_ROOT / f"clip_{clip_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _latest_json_path(clip_id: int) -> Path:
    """Path to the on-disk latest.json cache for a given clip."""
    return BETA_RENDERS_ROOT / f"clip_{clip_id}" / "latest.json"


def _write_latest_cache(
    clip_id: int,
    result: BetaRenderResult,
    *,
    storage_url: str = "",
    storage_key: str = "",
    storage_backend: str = "",
) -> None:
    """Persist the render result to output/beta_renders/clip_{id}/latest.json.

    When storage_backend is 'r2', *storage_url* and *storage_key* are written
    alongside the local paths so that the GET handler can reconstruct the
    correct URL without re-uploading.
    """
    cache_path = _latest_json_path(clip_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "style_pack": result.style_pack,
        "beta_path": result.beta_path,
        "current_path": result.current_path,
        "effects_applied": result.effects_applied,
        "qa_ok": result.qa_ok,
        "warnings": result.warnings,
        "rendered_at": datetime.now(timezone.utc).isoformat(),
        # Phase 5 storage fields — present so the GET handler knows which
        # backend produced the URL and can return it directly.
        "storage_url": storage_url,
        "storage_key": storage_key,
        "storage_backend": storage_backend,
    }
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _read_latest_cache(clip_id: int) -> Optional[dict]:
    """Read the on-disk latest.json for *clip_id*. Returns None if absent."""
    cache_path = _latest_json_path(clip_id)
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("editor: failed to read latest.json for clip %d: %s", clip_id, exc)
        return None


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get("/styles", response_model=list[StyleSchema])
def get_styles() -> list[StyleSchema]:
    """Return the full StylePack catalogue.

    200 OK — always. No authentication required (static data).
    """
    return [
        StyleSchema(
            name=pack.name,
            label=pack.label,
            description=pack.description,
            transition=pack.transition,
            color_preset=pack.color_preset,
            motion=pack.motion,
            text_animation=pack.text_animation,
            caption_animation=pack.caption_animation,
        )
        for pack in list_style_packs()
    ]


@router.post("/render-beta", response_model=RenderBetaResponse)
def post_render_beta(
    body: RenderBetaRequest,
    db: Session = Depends(get_db),
    _user: models.User = Depends(auth.current_user),
) -> RenderBetaResponse:
    """Apply a StylePack to a Clip and return the beta-rendered MP4.

    Validates that *clip_id* exists in the DB and that *style_pack* is one of
    the known keys in STYLE_PACKS before invoking the render pipeline.

    The render runs **synchronously** in the request handler. For a 30-second
    clip this blocks the connection for up to ~30 s — acceptable for a single
    creator workstation in v1.
    TODO (Phase 4+): dispatch to a Celery/arq background worker and return a
    job_id so the frontend can poll for completion.

    Parameters
    ----------
    body : RenderBetaRequest
        clip_id, style_pack, optional hook_text, platform.

    Returns
    -------
    RenderBetaResponse
        Full render result including URLs for both the original and beta clips.

    Raises
    ------
    HTTPException(404)
        If *clip_id* does not exist in the clips table.
    HTTPException(400)
        If *style_pack* is not a recognised key in STYLE_PACKS.
    HTTPException(500)
        If the render pipeline raises an unexpected exception.
    """
    # ── Validate style_pack early so the error is 400, not 500 ───────────────
    if body.style_pack not in STYLE_PACKS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown style pack {body.style_pack!r}. "
                f"Valid options: {sorted(STYLE_PACKS.keys())}"
            ),
        )

    # ── Fetch the Clip row ────────────────────────────────────────────────────
    clip: Optional[models.Clip] = (
        db.query(models.Clip).filter(models.Clip.id == body.clip_id).first()
    )
    if clip is None:
        raise HTTPException(status_code=404, detail=f"Clip {body.clip_id} not found")

    master_path = (clip.file_path or "").replace("\\", "/")

    # ── Prepare output directory ──────────────────────────────────────────────
    output_dir = _clip_output_dir(body.clip_id)

    # ── Invoke render_beta ────────────────────────────────────────────────────
    try:
        result: BetaRenderResult = render_beta(
            master_path,
            style_pack=body.style_pack,
            hook_text=body.hook_text,
            output_dir=str(output_dir),
            platform=body.platform,
        )
    except Exception as exc:
        logger.exception(
            "editor: render_beta failed for clip %d with pack %r: %s",
            body.clip_id, body.style_pack, exc,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Render failed for clip {body.clip_id}: {exc}",
        )

    # ── Phase 5: upload to cloud storage when STORAGE_BACKEND != 'local' ────────
    storage_url: str = ""
    storage_key: str = ""
    storage_backend_name: str = ""
    beta_url_for_response: str = _path_to_url(result.beta_path)

    try:
        provider = get_storage_provider()
        if provider.name != "local" and result.beta_path and os.path.isfile(result.beta_path):
            storage_key = f"beta_renders/clip_{body.clip_id}/{body.style_pack}.mp4"
            stored = provider.upload(
                result.beta_path,
                storage_key,
                content_type="video/mp4",
            )
            storage_url = stored.url
            storage_backend_name = stored.backend
            beta_url_for_response = stored.url
            logger.info(
                "editor: uploaded beta render clip=%d key=%r url=%r",
                body.clip_id, storage_key, storage_url,
            )
            # Delete the local temp file — cloud is the source of truth now.
            try:
                os.remove(result.beta_path)
            except Exception as rm_exc:
                logger.warning(
                    "editor: could not remove local beta file %r: %s",
                    result.beta_path, rm_exc,
                )
    except Exception as upload_exc:
        logger.error(
            "editor: storage upload failed for clip %d: %s — returning local URL",
            body.clip_id, upload_exc,
        )
        # Do NOT silently swallow when R2 is configured — re-raise so the
        # caller knows the upload failed.  For local backend, this branch
        # is never reached.
        storage_backend_env = os.environ.get("STORAGE_BACKEND", "local").lower()
        if storage_backend_env != "local":
            raise HTTPException(
                status_code=500,
                detail=f"Storage upload failed for clip {body.clip_id}: {upload_exc}",
            )

    # ── Cache result to disk ──────────────────────────────────────────────────
    try:
        _write_latest_cache(
            body.clip_id,
            result,
            storage_url=storage_url,
            storage_key=storage_key,
            storage_backend=storage_backend_name,
        )
    except Exception as exc:
        logger.warning(
            "editor: could not write latest.json for clip %d: %s",
            body.clip_id, exc,
        )

    # ── Build response ────────────────────────────────────────────────────────
    return RenderBetaResponse(
        clip_id=body.clip_id,
        current_path=result.current_path,
        current_url=_path_to_url(result.current_path),
        beta_path=result.beta_path,
        beta_url=beta_url_for_response,
        style_pack=result.style_pack,
        effects_applied=result.effects_applied,
        render_time_s=result.render_time_s,
        qa_ok=result.qa_ok,
        warnings=result.warnings,
    )


@router.get("/render-beta/{clip_id}", response_model=RenderBetaResponse)
def get_latest_render(clip_id: int) -> RenderBetaResponse:
    """Return the most recently cached beta render result for *clip_id*.

    Reads from ``output/beta_renders/clip_{clip_id}/latest.json`` written by
    the POST endpoint.  The render_time_s field is set to 0.0 for cached
    responses (no timing information is stored).

    Returns
    -------
    RenderBetaResponse
        Cached render payload.

    Raises
    ------
    HTTPException(404)
        When no render has been run for this clip yet.
    """
    cache = _read_latest_cache(clip_id)
    if cache is None:
        raise HTTPException(
            status_code=404,
            detail=f"No beta render found for clip {clip_id}. Run POST /api/editor/render-beta first.",
        )

    current_path = cache.get("current_path", "")
    beta_path = cache.get("beta_path", "")

    # Phase 5: when the render was uploaded to cloud storage the cached
    # storage_url supersedes the local /media/ URL.
    cached_storage_url: str = cache.get("storage_url", "")
    cached_storage_backend: str = cache.get("storage_backend", "")
    if cached_storage_url and cached_storage_backend and cached_storage_backend != "local":
        beta_url_for_response = cached_storage_url
    else:
        beta_url_for_response = _path_to_url(beta_path)

    return RenderBetaResponse(
        clip_id=clip_id,
        current_path=current_path,
        current_url=_path_to_url(current_path),
        beta_path=beta_path,
        beta_url=beta_url_for_response,
        style_pack=cache.get("style_pack", ""),
        effects_applied=cache.get("effects_applied", []),
        render_time_s=0.0,
        qa_ok=cache.get("qa_ok", True),
        warnings=cache.get("warnings", []),
    )
