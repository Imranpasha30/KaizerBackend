"""V4 editor endpoints — read/write canvas.json, upload images,
re-render on demand.

The frontend's Editor tab for V4 jobs talks to these routes. Two key
design notes:

  * Every WRITE validates the body against the V4JobCanvas Pydantic
    model. A malformed canvas can never land on disk.
  * Re-render runs Step 2 only — the trimmed video from Step 1 is
    sunk cost. Bulletin re-renders are ~5-15 s.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

import auth
import models
from database import SessionLocal, get_db
from pipeline_v4 import canvas_engine
from pipeline_v4 import v1_bridge
from pipeline_v4 import image_provider
from pipeline_v4 import seo_provider
from pipeline_v4.canvas_schema import Canvas, CanvasSEO, V4JobCanvas


router = APIRouter(prefix="/api/v4", tags=["v4-editor"])


CANVAS_JSON_NAME = "canvas.json"


# ─── Helpers ────────────────────────────────────────────────────────

def _backend_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _media_root() -> Path:
    """Root the /media static mount serves from — KAIZER_OUTPUT_ROOT when set
    (e.g. a dedicated D:/ render folder), else <backend>/output. MUST match
    main.py's mount or the /media URLs we build here 404 and the editor wrongly
    shows 'not rendered yet'."""
    import os
    return Path(os.environ.get("KAIZER_OUTPUT_ROOT") or (_backend_root() / "output"))


def _v4_dir_for(job: models.Job) -> Path:
    """Resolve a V4 job's output dir. We accept either an absolute
    path or a project-relative one; same shape main.py uses."""
    if not job.output_dir:
        raise HTTPException(404, "Job has no output_dir yet — pipeline may still be running")
    out = Path(job.output_dir)
    if not out.is_absolute():
        out = (_backend_root() / out).resolve()
    if not out.is_dir():
        raise HTTPException(404, f"V4 output dir not found: {out}")
    return out


def _read_canvas(out_dir: Path) -> V4JobCanvas:
    p = out_dir / CANVAS_JSON_NAME
    if not p.is_file():
        raise HTTPException(404, "canvas.json not found — V4 pipeline may not have finished step 1 yet")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(500, f"canvas.json is malformed: {exc}")
    try:
        return V4JobCanvas.model_validate(raw)
    except Exception as exc:
        raise HTTPException(500, f"canvas.json failed schema validation: {exc}")


def _write_canvas(out_dir: Path, jc: V4JobCanvas) -> Path:
    p = out_dir / CANVAS_JSON_NAME
    body = json.loads(jc.model_dump_json())
    p.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def _owned_job(job_id: int, user: models.User, db: Session) -> models.Job:
    j = db.query(models.Job).filter(
        models.Job.id == job_id,
        models.Job.user_id == user.id,
    ).first()
    if not j:
        raise HTTPException(404, "Job not found")
    if (j.platform or "") != "full_video_shorts_v4":
        raise HTTPException(
            400, f"Job {job_id} is not a V4 job (platform={j.platform!r})"
        )
    return j


# ─── In-flight render tracking ──────────────────────────────────────
# We rerender canvases in a daemon thread so the request returns
# immediately. The UI polls /render/state.
_RENDERING: dict[int, dict] = {}
_RENDER_LOCK = threading.Lock()


def _set_state(job_id: int, **kv) -> None:
    with _RENDER_LOCK:
        cur = _RENDERING.get(job_id) or {}
        cur.update(kv)
        _RENDERING[job_id] = cur


# ─── "Render all channels" batch state ──────────────────────────────
# Pre-renders every selected channel's FULL branded clip (logo + watermark +
# zoom + nudge + that channel's intro) in the editor so the operator sees a
# ready video per channel. Runs in a daemon thread; the UI polls
# /prepare-channel-videos/state. _PREPARE_RUN_LOCK serializes the whole
# cascade GLOBALLY (one job's batch at a time) so it never fans out N ffmpegs
# at once and starves live re-renders.
_PREPARE: dict[int, dict] = {}
_PREPARE_LOCK = threading.Lock()
_PREPARE_RUN_LOCK = threading.Lock()


# ─── Read ───────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}/canvas")
def get_canvas(
    job_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    job = _owned_job(job_id, user, db)
    out_dir = _v4_dir_for(job)
    jc = _read_canvas(out_dir)
    state = _RENDERING.get(job_id) or {}

    # Compute web URLs for the rendered videos so the editor's
    # player can load them directly. /media/<rel> is the static
    # mount over the backend's output/ dir; we derive the relative
    # path from out_dir.
    backend_root = _backend_root()
    def _media_url(filename: str) -> Optional[str]:
        p = out_dir / filename
        if not p.is_file():
            return None
        try:
            rel = p.resolve().relative_to(_media_root().resolve())
            # forward-slash for URL safety; cache-bust by mtime so the
            # browser picks up freshly re-rendered files.
            mt = int(p.stat().st_mtime)
            return f"/media/{rel.as_posix()}?t={mt}"
        except ValueError:
            return None

    bulletin_url = _media_url(jc.bulletin.output_filename)
    trimmed_url = _media_url(Path(jc.trimmed_bulletin_path).name)
    shorts_urls = [_media_url(s.output_filename) for s in jc.shorts]

    # RAW trimmed short clips (for the live compositor — plays before any render). Prefer each
    # short's own trimmed_video_path, fall back to the job-level trimmed_shorts_paths[i].
    def _short_trimmed_url(i, sc):
        p = (getattr(sc, "trimmed_video_path", "") or "")
        _tsp = getattr(jc, "trimmed_shorts_paths", []) or []
        if not p and i < len(_tsp):
            p = _tsp[i]
        return _media_url(Path(p).name) if p else None
    trimmed_shorts_urls = [_short_trimmed_url(i, s) for i, s in enumerate(jc.shorts)]

    # Materialised Clip IDs so the V4Editor can call the existing
    # /api/clips/{id}/publish endpoint directly. clip_index 0 is the
    # bulletin, 1..N are the shorts in order.
    bulletin_clip_id: Optional[int] = None
    shorts_clip_ids: list[Optional[int]] = [None] * len(jc.shorts)
    # Side index: clip_index -> Clip row, for the SEO merge below.
    clips_by_index: dict[int, models.Clip] = {}
    try:
        clips = db.query(models.Clip).filter(
            models.Clip.job_id == job_id
        ).order_by(models.Clip.clip_index).all()
        for c in clips:
            clips_by_index[c.clip_index] = c
            if c.clip_index == 0:
                bulletin_clip_id = c.id
            elif 1 <= c.clip_index <= len(shorts_clip_ids):
                shorts_clip_ids[c.clip_index - 1] = c.id
    except Exception:
        pass

    # ─── Merge Clip.seo into canvas.seo if canvas.seo is empty ─────
    # The V2-style SEO regen path (POST /api/clips/{id}/seo/generate)
    # writes to the Clip row — not to canvas.json — so a job whose SEO
    # was regenerated via that path has the fresh SEO ONLY in the DB.
    # Without this merge, a page refresh shows empty SEO in the editor
    # because canvas.json's seo block is still the empty initial state.
    # Server-side merge is the right home for this: any future SEO
    # consumer (publisher, exporter, analytics) reads from the canvas
    # response and gets the same merged view.
    def _canvas_seo_is_empty(seo) -> bool:
        if not seo:
            return True
        title = getattr(seo, "title", "") or ""
        desc = getattr(seo, "description", "") or ""
        return not (title.strip() or desc.strip())

    def _hydrate_canvas_seo_from_clip(canv, clip) -> bool:
        """Returns True if a hydration happened (caller decides whether
        to also write it back to canvas.json so the merge is sticky)."""
        if not clip or not (clip.seo or "").strip():
            return False
        if not _canvas_seo_is_empty(canv.seo):
            return False
        try:
            raw = json.loads(clip.seo)
        except (ValueError, TypeError):
            return False
        if not (raw.get("title") or raw.get("description")):
            return False
        # Filter to fields CanvasSEO knows about; extras would fail
        # Pydantic validation on PUT canvas.
        from pipeline_v4.canvas_schema import CanvasSEO as _CanvasSEO
        allowed = set(_CanvasSEO.model_fields.keys())
        filtered = {k: v for k, v in raw.items() if k in allowed}
        try:
            canv.seo = _CanvasSEO(**filtered)
        except Exception:
            return False
        return True

    hydrated_any = False
    if _hydrate_canvas_seo_from_clip(jc.bulletin, clips_by_index.get(0)):
        hydrated_any = True
    for i, sc in enumerate(jc.shorts):
        if _hydrate_canvas_seo_from_clip(sc, clips_by_index.get(i + 1)):
            hydrated_any = True
    # Persist the merge so subsequent GETs don't need to re-hydrate
    # AND so the publish path (which reads canvas.json) sees the SEO.
    if hydrated_any:
        try:
            _write_canvas(out_dir, jc)
        except Exception as exc:
            print(f"[v4/canvas] hydration write failed: {exc}", flush=True)

    return {
        "canvas": json.loads(jc.model_dump_json()),
        "pool_listing": _list_pool(out_dir, job_id),
        "render_state": state.get("state", "idle"),
        "render_msg":   state.get("msg", ""),
        "render_target": state.get("target", ""),
        # Stage 2/3: "render_deferred" => the pipeline produced the scene but skipped the
        # up-front compose; the editor shows an Export action to render on demand.
        "current_stage": getattr(job, "current_stage", "") or "",
        "bulletin_url": bulletin_url,
        "trimmed_url":  trimmed_url,
        "shorts_urls":  shorts_urls,
        "trimmed_shorts_urls": trimmed_shorts_urls,
        "bulletin_clip_id": bulletin_clip_id,
        "shorts_clip_ids":  shorts_clip_ids,
    }


def _list_pool(out_dir: Path, job_id: Optional[int] = None) -> list[dict]:
    """Enumerate pool images. The ``url`` must match the
    ``serve_pool_file`` route exactly:  /api/v4/jobs/{int}/pool/{name}
    (numeric job id, path component ``pool`` not ``_pool``). The old
    URL pointed at ``{job_dir.name}/_pool/...`` which the route never
    matched — every thumbnail in the editor came back broken."""
    pool_dir = out_dir / "_pool"
    if not pool_dir.is_dir():
        return []
    out = []
    for f in sorted(pool_dir.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        # Cache-bust by mtime so a replaced image at the SAME filename
        # (overwrite) never shows the browser's cached old copy in the panel.
        try:
            _mt = int(f.stat().st_mtime)
        except OSError:
            _mt = 0
        if job_id is not None:
            url = f"/api/v4/jobs/{job_id}/pool/{f.name}?t={_mt}"
        else:
            # Back-compat path when the caller can't supply a job id.
            url = f"/api/v4/jobs/{out_dir.name}/pool/{f.name}?t={_mt}"
        out.append({
            "filename": f.name,
            "size_bytes": f.stat().st_size,
            "url": url,
        })
    return out


@router.get("/jobs/{job_id}/pool/{filename}")
def serve_pool_file(
    job_id: int,
    filename: str,
    token: Optional[str] = Query(default=None),
    authorization: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
):
    """Serve a pool image directly so the editor can preview without
    going through /api/file/ allowlist friction.

    Accepts the JWT via ``Authorization: Bearer`` header OR ``?token=``
    query param. The query path exists because plain ``<img src>`` tags
    in the browser can't attach Authorization headers — without this the
    editor's pool thumbnails come back 401 when ``KAIZER_AUTH_REQUIRED``
    is on, and the user sees broken-image icons."""
    from fastapi.responses import FileResponse

    raw = ""
    if authorization and authorization.lower().startswith("bearer "):
        raw = authorization.split(" ", 1)[1].strip()
    elif token:
        raw = token.strip()

    user: Optional[models.User] = None
    if raw:
        payload = auth.decode_token(raw)
        if payload:
            uid = int(payload.get("sub") or 0)
            if uid:
                u = db.query(models.User).filter(models.User.id == uid).first()
                if u and u.is_active:
                    user = u

    if user is None:
        if os.getenv("KAIZER_AUTH_REQUIRED", "false").lower() in ("1", "true", "yes", "on"):
            raise HTTPException(401, "Authentication required")
        user = auth.ensure_legacy_user(db)

    job = _owned_job(job_id, user, db)
    out_dir = _v4_dir_for(job)
    p = out_dir / "_pool" / Path(filename).name
    if not p.is_file():
        raise HTTPException(404, "pool image not found")
    ext = p.suffix.lower().lstrip(".")
    mt = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
          "png": "image/png", "webp": "image/webp"}.get(ext, "application/octet-stream")
    return FileResponse(p, media_type=mt)


# ─── Write ──────────────────────────────────────────────────────────

class CanvasUpdateIn(BaseModel):
    canvas: dict     # Raw JSON; we re-validate against V4JobCanvas


@router.put("/jobs/{job_id}/canvas")
def put_canvas(
    job_id: int,
    payload: CanvasUpdateIn,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    job = _owned_job(job_id, user, db)
    out_dir = _v4_dir_for(job)
    # Validate the payload before writing — refuses malformed canvases
    try:
        jc = V4JobCanvas.model_validate(payload.canvas)
    except Exception as exc:
        raise HTTPException(400, f"canvas.json failed schema validation: {exc}")
    _write_canvas(out_dir, jc)
    # Sync Clip rows so SEO / text / image edits propagate to the
    # publish flow without waiting for a fresh job run.
    try:
        from pipeline_v4 import orchestrator as v4_orch
        bulletin_path = out_dir / jc.bulletin.output_filename
        n = v4_orch._materialise_clips(
            job_id=job_id,
            out_dir=out_dir,
            bulletin_canvas=jc.bulletin,
            shorts_canvases=jc.shorts,
            bulletin_path=str(bulletin_path) if bulletin_path.is_file() else "",
        )
    except Exception as exc:
        print(f"[v4/editor] clip sync failed (soft-skip): {exc}", flush=True)
        n = -1
    return {"ok": True, "saved": True, "clips_synced": n}


# ─── Pool: upload a new image ────────────────────────────────────────

@router.post("/jobs/{job_id}/pool/upload")
async def upload_pool_image(
    job_id: int,
    image: UploadFile = File(...),
    label: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    job = _owned_job(job_id, user, db)
    out_dir = _v4_dir_for(job)
    pool_dir = out_dir / "_pool"
    pool_dir.mkdir(parents=True, exist_ok=True)

    raw = await image.read()
    if not raw:
        raise HTTPException(400, "empty upload")
    if len(raw) > 8 * 1024 * 1024:
        raise HTTPException(413, "image > 8 MB")

    base = Path(image.filename or "image.jpg").name
    safe_stem = "".join(c if (c.isalnum() or c in "._-") else "_" for c in base)
    target = pool_dir / safe_stem
    # de-collide
    i = 0
    while target.exists():
        i += 1
        stem, ext = os.path.splitext(safe_stem)
        target = pool_dir / f"{stem}_{i}{ext}"
    target.write_bytes(raw)
    return {
        "ok": True,
        "filename": target.name,
        "label": label or target.stem,
        "size_bytes": target.stat().st_size,
        "url": f"/api/v4/jobs/{job_id}/pool/{target.name}",
    }


# ─── Pool: fetch authentic image for a story ────────────────────────

class FetchImageIn(BaseModel):
    """One-off image fetch driven by the editor's 'Auto-fetch' button."""
    title: str = ""
    title_english: str = ""
    summary: str = ""
    story_index: int = 0
    prefer_real_photo: bool = True   # default: real-photo > AI render


# ─── AI image generation (Nano Banana via Vertex) ───────────────────
# Same Vertex client + GCP credits the thumbnail flow uses, but with a
# B-roll-tuned prompt template. Result lands in the job's _pool/ so it
# can be picked from the editor's image dropdowns just like an
# uploaded or auto-fetched image.

class GenerateImageIn(BaseModel):
    """Generate a new sidebar/carousel image for a story.

    First call: omit ``tweak`` → Gemini writes a fresh prompt from the
    story title + summary. Subsequent calls: pass a ``tweak`` string →
    Gemini reuses the previous prompt (stored next to the JPG) and
    applies the tweak. Cheaper than starting from scratch."""
    title: str = ""
    title_english: str = ""
    summary: str = ""
    story_index: int = 0
    tweak: str = ""
    label: str = ""


@router.post("/jobs/{job_id}/pool/ai-generate")
def ai_generate_pool_image(
    job_id: int,
    payload: GenerateImageIn,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Two-step Vertex call: prompt-writer then Nano Banana. Saves the
    JPG into ``<job>/_pool/`` so the editor's image picker treats it
    the same as any uploaded or auto-fetched image. The user can then
    point a specific story or short at it via the existing image
    selector. Persists the prompt alongside the JPG for a future tweak."""
    from pipeline_v4 import image_ai as _iai
    job = _owned_job(job_id, user, db)
    out_dir = _v4_dir_for(job)
    pool_dir = out_dir / "_pool"
    pool_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic naming so re-generation per story overwrites the
    # previous attempt rather than littering the pool with duplicates.
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in
                   (payload.label or f"story{payload.story_index:02d}_ai"))[:60]
    out_path = pool_dir / f"{safe}.jpg"
    prompt_path = out_path.with_suffix(out_path.suffix + ".prompt.txt")

    tweak = (payload.tweak or "").strip()
    previous_prompt = ""
    if tweak and prompt_path.is_file():
        try:
            previous_prompt = prompt_path.read_text(encoding="utf-8")
        except Exception:
            previous_prompt = ""

    saved, final_prompt = _iai.make_image_for_story(
        title_native=payload.title or "",
        title_english=payload.title_english or "",
        summary=payload.summary or "",
        language=(job.language or "te"),
        out_path=str(out_path),
        previous_prompt=previous_prompt,
        tweak=tweak,
    )
    if not saved or not out_path.is_file():
        raw = getattr(_iai.generate_image, "last_error", "") or ""
        hint = ""
        if "RESOURCE_EXHAUSTED" in raw or " 429" in raw:
            hint = ("Gemini quota/credits exhausted. Top up at "
                    "https://ai.studio/projects or check GCP billing.")
        elif "PERMISSION_DENIED" in raw or " 403" in raw:
            hint = ("Vertex AI permission denied — check the SA has "
                    "the 'Vertex AI User' role on the project.")
        detail = hint or (raw[:300] if raw else "Nano Banana returned no image.")
        raise HTTPException(502, detail)

    try:
        prompt_path.write_text(final_prompt, encoding="utf-8")
    except Exception as exc:
        print(f"[v4/image-ai] prompt persist soft-fail: {exc}", flush=True)

    return {
        "ok": True,
        "filename": out_path.name,
        "size_bytes": out_path.stat().st_size,
        "url": f"/api/v4/jobs/{job_id}/pool/{out_path.name}",
        "label": payload.label or (payload.title or "")[:60],
        "prompt": final_prompt,
        "iterated": bool(previous_prompt and tweak),
    }


@router.post("/jobs/{job_id}/pool/fetch")
def fetch_pool_image(
    job_id: int,
    payload: FetchImageIn,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Run V1's CSE / DDG / Pexels / OpenAI chain for one story and
    drop the winner into the job's _pool/ folder. Also copies to the
    operator's user_assets/ so the same image is reusable later."""
    job = _owned_job(job_id, user, db)
    out_dir = _v4_dir_for(job)
    user_assets = image_provider._user_assets_dir_for(user.id)
    q = image_provider.StoryImageQuery(
        story_index=payload.story_index,
        title=payload.title or "",
        title_en=payload.title_english or "",
        summary=payload.summary or "",
        prefer_real_photo=payload.prefer_real_photo,
    )
    fn = image_provider.fetch_story_image(
        query=q,
        language=getattr(job, "language", None) or "te",
        pool_dir=out_dir / "_pool",
        user_assets_dir=user_assets,
    )
    if not fn:
        raise HTTPException(502, "no authentic image found")
    pool_dir = out_dir / "_pool"
    p = pool_dir / fn
    return {
        "ok": True,
        "filename": fn,
        "size_bytes": p.stat().st_size,
        "url": f"/api/v4/jobs/{job_id}/pool/{fn}",
        "label": (payload.title or payload.title_english or "")[:60],
    }


# ─── Pool: delete an image ───────────────────────────────────────────

@router.delete("/jobs/{job_id}/pool/{filename}")
def delete_pool_image(
    job_id: int,
    filename: str,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    job = _owned_job(job_id, user, db)
    out_dir = _v4_dir_for(job)
    p = out_dir / "_pool" / Path(filename).name
    if not p.is_file():
        raise HTTPException(404, "pool image not found")
    p.unlink()
    return {"ok": True, "deleted": filename}


# ─── Background video — list samples + serve bytes ──────────────────
# Bg videos are referenced from canvas.layout.bg_video_path as
# "sample:<name>" (bundled demo) or "asset:<id>" (user upload). User
# uploads reuse the existing /api/assets/upload endpoint with
# folder_path="v4_bg_videos" — no new upload route needed. These two
# endpoints just expose the sample list and serve the sample bytes
# behind a JWT-aware route so the editor can preview them.

@router.get("/bg-samples")
def list_bg_video_samples(
    user: models.User = Depends(auth.current_user),
) -> list[dict]:
    """List bundled background-video demos shipped with the frontend.
    The canvas references these by ``sample:<filename>``."""
    from pipeline_v4.canvas_engine import bg_video_samples_dir
    d = bg_video_samples_dir()
    if not d.is_dir():
        return []
    out = []
    for f in sorted(d.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() not in (".mp4", ".webm", ".mov", ".m4v"):
            continue
        out.append({
            "filename": f.name,
            "size_bytes": f.stat().st_size,
            "url": f"/api/v4/bg-samples/{f.name}",
            "ref": f"sample:{f.name}",
        })
    return out


@router.get("/bg-samples/{filename}")
def serve_bg_video_sample(
    filename: str,
    token: Optional[str] = Query(default=None),
    authorization: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
):
    """Serve a bundled bg-video sample. Accepts JWT via header OR query
    so HTML5 <video> tags work (they can't attach Authorization)."""
    from fastapi.responses import FileResponse
    from pipeline_v4.canvas_engine import bg_video_samples_dir

    raw = ""
    if authorization and authorization.lower().startswith("bearer "):
        raw = authorization.split(" ", 1)[1].strip()
    elif token:
        raw = token.strip()
    if raw and not auth.decode_token(raw):
        if os.getenv("KAIZER_AUTH_REQUIRED", "false").lower() in ("1","true","yes","on"):
            raise HTTPException(401, "invalid token")
    elif not raw and os.getenv("KAIZER_AUTH_REQUIRED", "false").lower() in ("1","true","yes","on"):
        raise HTTPException(401, "auth required")

    d = bg_video_samples_dir()
    p = d / Path(filename).name
    if not p.is_file():
        raise HTTPException(404, "bg sample not found")
    return FileResponse(p, media_type="video/mp4")


# ─── Auto-distribute images across a story timeline ──────────────────

class AutoDistributeIn(BaseModel):
    """Spread N images evenly across a bulletin story's duration and
    stamp the chosen transition. Used by the editor's one-click
    "Auto distribute" button so the operator doesn't have to drag
    durations by hand for every image."""
    story_index: int
    add_filenames: list[str] = []         # optional: append these before distributing
    effect: str = "fade"                  # cut | fade | slide_left | slide_right | zoom_in
    effect_duration: float = 0.4
    keep_existing_order: bool = True      # False = sort by current t_start, True = use list order


def _spread_images_evenly(
    *,
    sources: list[str],
    story_dur: float,
    effect: str,
    effect_duration: float,
) -> list[dict]:
    """Return a list of CanvasImage-shaped dicts spread across [0, story_dur].

    Math: with N images and overlap ed (the fade duration), each image
    is visible for L = (D + (N-1)*ed) / N seconds, and image i starts at
    i * (L - ed). With effect="cut" or ed=0, there's no overlap and each
    image gets D/N exclusively. Clamps ed so very short stories don't
    end up always-faded."""
    n = len(sources)
    if n <= 0:
        return []
    if story_dur <= 0.5:
        # Edge case — nothing meaningful we can do; everyone gets the
        # tiny window and the editor warns elsewhere.
        story_dur = max(0.5, story_dur)
    if n == 1:
        return [{
            "src": sources[0],
            "t_start": 0.0,
            "t_end": round(story_dur, 3),
            "effect": effect,
            "effect_duration": round(min(effect_duration, story_dur / 2.0), 3),
            "source": "manual",
        }]

    ed = max(0.0, effect_duration) if effect != "cut" else 0.0
    # Clamp ed so the smallest hold window is positive — otherwise the
    # animation runs the entire image duration.
    max_ed_total = story_dur / (3 * n)   # very conservative
    ed = min(ed, max_ed_total)

    L = (story_dur + (n - 1) * ed) / n
    out: list[dict] = []
    for i, src in enumerate(sources):
        t_start = i * (L - ed)
        t_end = t_start + L
        # Snap last image's end to the exact story end so we don't leave
        # a tiny gap from float drift.
        if i == n - 1:
            t_end = story_dur
        out.append({
            "src": src,
            "t_start": round(max(0.0, t_start), 3),
            "t_end": round(min(story_dur, t_end), 3),
            "effect": effect,
            "effect_duration": round(ed, 3),
            "source": "manual",
        })
    return out


@router.post("/jobs/{job_id}/bulletin/auto-distribute")
def auto_distribute_bulletin(
    job_id: int,
    payload: AutoDistributeIn,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    """One-click: distribute the bulletin story's current images (plus
    any ``add_filenames`` to append) evenly across its duration and stamp
    the chosen transition. Returns the freshly-written canvas so the
    editor can refresh without a separate GET."""
    job = _owned_job(job_id, user, db)
    out_dir = _v4_dir_for(job)
    cpath = out_dir / "canvas.json"
    if not cpath.is_file():
        raise HTTPException(404, "canvas.json not found for this job")
    raw = json.loads(cpath.read_text(encoding="utf-8"))
    jc = V4JobCanvas.model_validate(raw)

    if payload.story_index < 0 or payload.story_index >= len(jc.bulletin.stories):
        raise HTTPException(400, f"story_index {payload.story_index} out of range")
    story = jc.bulletin.stories[payload.story_index]
    story_dur = story.video_t_end - story.video_t_start
    if story_dur <= 0:
        raise HTTPException(400, "story has zero or negative duration; cannot distribute")

    # Build the source list. Order = existing images (in current array
    # order if keep_existing_order, otherwise sorted by t_start) followed
    # by any newly-added filenames.
    existing = list(story.images)
    if not payload.keep_existing_order:
        existing.sort(key=lambda im: im.t_start)
    sources = [im.src for im in existing]
    for fn in payload.add_filenames:
        if fn and fn not in sources:
            sources.append(fn)

    new_images = _spread_images_evenly(
        sources=sources,
        story_dur=story_dur,
        effect=payload.effect,
        effect_duration=payload.effect_duration,
    )
    # Mutate the canvas in place; Pydantic re-validation on write catches
    # any malformed timings before they land on disk.
    raw_bulletin = raw.get("bulletin") or {}
    raw_stories = raw_bulletin.get("stories") or []
    raw_stories[payload.story_index]["images"] = new_images
    jc2 = V4JobCanvas.model_validate(raw)
    _write_canvas(out_dir, jc2)
    return {
        "ok": True,
        "story_index": payload.story_index,
        "count": len(new_images),
        "story_duration": round(story_dur, 3),
        "images": new_images,
    }


# ─── Re-render (Step 2 only) ─────────────────────────────────────────

class RenderIn(BaseModel):
    target: str = "bulletin"      # "bulletin" | "short"
    index: int = 0                # for target=short


@router.post("/jobs/{job_id}/render")
def trigger_render(
    job_id: int,
    payload: RenderIn,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    job = _owned_job(job_id, user, db)
    out_dir = _v4_dir_for(job)
    jc = _read_canvas(out_dir)

    if payload.target == "bulletin":
        target_canvas = jc.bulletin
        tgt_label = "bulletin"
    elif payload.target == "short":
        if not (0 <= payload.index < len(jc.shorts)):
            raise HTTPException(400, f"short index {payload.index} out of range")
        target_canvas = jc.shorts[payload.index]
        tgt_label = f"short_{payload.index + 1:02d}"
    else:
        raise HTTPException(400, f"target must be 'bulletin' or 'short' (got {payload.target!r})")

    with _RENDER_LOCK:
        cur = _RENDERING.get(job_id) or {}
        if cur.get("state") == "running":
            raise HTTPException(409, f"render already running for job {job_id}: {cur.get('target')}")
        _RENDERING[job_id] = {"state": "queued", "target": tgt_label, "msg": "spawning worker"}

    def _worker():
        from services import stage_events as _se
        _rr_env = _se.Envelope(
            tenant_id=getattr(user, "id", None), user_id=getattr(user, "id", None),
            job_id=job_id, clip_id=None, channel_id=None, label=tgt_label,
        )
        try:
            _set_state(job_id, state="running", msg=f"compositing {tgt_label}")
            # RE-RENDER belt telemetry (admin Pipeline Flow). Slice is the
            # already-trimmed source (instant here), then compose drives the
            # canvas overlay; encode is the final mux.
            _se.emit_lane(_rr_env, "rerender", "slice", _se.ENTERED)
            _se.emit_lane(_rr_env, "rerender", "slice", _se.EXITED)
            _se.emit_lane(_rr_env, "rerender", "compose", _se.ENTERED)
            # Resolve pool images so V1's layouts have sidebar/short
            # image inputs to work with.
            pool_dir = out_dir / "_pool"
            pool_paths = [
                str(pool_dir / p["filename"])
                for p in _list_pool(out_dir)
                if (pool_dir / p["filename"]).is_file()
            ]
            # Pull the user's brand suffix + watermark defaults so the
            # editor's re-renders match what a fresh pipeline run would
            # produce (no "KAIZER X" leak, same watermark stamp).
            from pipeline_v4 import orchestrator as _v4_orch
            user_d = _v4_orch._load_user_defaults(job_id) or {}
            channel_name = (user_d.get("brand_suffix") or "").lstrip(" |·-_•").strip()
            wm_text = user_d.get("watermark_text", "") or ""
            wm_op = float(user_d.get("watermark_opacity", 0.35) or 0.35)
            wm_pos = user_d.get("watermark_position", "top-right") or "top-right"

            # Custom-template editor state (per-slot media + text overrides). Read fresh so
            # the editor's edits apply on re-render. Empty for built-in jobs.
            _tmedia, _tmain = _v4_orch._load_template_media(job_id)
            _tover = _v4_orch._load_template_overrides(job_id)
            _ff_layout = (getattr(job, "fullform_layout", "") or "").strip().lower()
            _is_ff_custom = _ff_layout.startswith("custom:") and _ff_layout.split(":", 1)[1].isdigit()

            if payload.target == "bulletin" and _is_ff_custom:
                # Custom FULL-FORM template: render through the custom engine (NOT the built-in
                # bulletin), mirroring the orchestrator's full-form branch — with per-slot
                # media + the editor's text overrides. This is what makes editing a custom
                # full-form job actually re-render the custom template.
                _ff_stories = [{
                    "headline": (s.title_native or s.title_english or ""),
                    "headline_alt": (s.title_english or ""), "subtitle": "",
                    "body": (s.summary or ""), "kicker": "", "images": [],
                } for s in jc.bulletin.stories]
                _ff_first = _ff_stories[0] if _ff_stories else {}
                v1_bridge.render_short(v1_bridge.ShortRenderInputs(
                    trimmed_short_path=jc.bulletin.trimmed_video_path,
                    title_text=(channel_name or ""),
                    output_path=str(out_dir / jc.bulletin.output_filename),
                    work_dir=out_dir, layout=_ff_layout, language=jc.language,
                    brand_logo=jc.bulletin.layout.brand_logo_path,
                    watermark_text=wm_text, watermark_opacity=wm_op, watermark_position=wm_pos,
                    template_media=_tmedia, main_media_slot=_tmain,
                    template_text_overrides=_tover, expected_kind="full",
                    headline=_ff_first.get("headline") or "",
                    headline_alt=_ff_first.get("headline_alt") or "",
                    body=_ff_first.get("body") or "",
                    support_images=pool_paths, stories=_ff_stories,
                    # per-job visual edit (inline builder) -> render verbatim if present
                    template_html_override=_v4_orch._load_html_override(job_id, "bulletin", 0),
                ))
            elif payload.target == "bulletin":
                # Re-render the whole bulletin via V1 broadcast layout.
                # Build a list of trim_engine.TrimmedStory-shaped objects
                # from the canvas's stories (any dict with the same
                # attributes works — v1_bridge only reads .title_native /
                # .video_t_start / .video_t_end).
                from types import SimpleNamespace
                stories = [
                    SimpleNamespace(
                        title_native=s.title_native,
                        title_english=s.title_english,
                        summary=s.summary,
                        video_t_start=s.video_t_start,
                        video_t_end=s.video_t_end,
                        # Per-story image carousel — the renderer turns these
                        # into a timed sidebar so replace/reorder/timing/fade
                        # actually show (previously the carousel was ignored).
                        images=list(s.images or []),
                    )
                    for s in jc.bulletin.stories
                ]
                _t_spd, _t_col = v1_bridge.extract_ticker_overrides(jc.bulletin.stories)
                v1_bridge.render_bulletin(v1_bridge.BulletinRenderInputs(
                    trimmed_bulletin_path=jc.bulletin.trimmed_video_path,
                    stories=stories,
                    output_path=str(out_dir / jc.bulletin.output_filename),
                    work_dir=out_dir,
                    language=jc.language,
                    brand_logo=jc.bulletin.layout.brand_logo_path,
                    sidebar_images=pool_paths,
                    layout=jc.bulletin.layout,
                    channel_name=channel_name,
                    watermark_text=wm_text,
                    watermark_opacity=wm_op,
                    watermark_position=wm_pos,
                    ticker_speed_s=_t_spd,
                    ticker_bg_color=_t_col,
                ))
            else:
                # Single short — full V1-parity editor knobs from
                # canvas.short_config drive the V1 composer.
                sc = jc.shorts[payload.index]
                cfg = sc.short_config
                story = sc.stories[0] if sc.stories else None
                story_title = ""
                if story:
                    story_title = (story.title_native or story.title_english or "").strip()
                title = (cfg.text if cfg and cfg.text else story_title) or "KAIZER X"
                _short_layout = (cfg.layout if cfg else v1_bridge.DEFAULT_SHORTS_LAYOUT)
                _is_short_custom = str(_short_layout or "").lower().startswith("custom:")

                if cfg and cfg.image_filename:
                    cand = pool_dir / cfg.image_filename
                    short_image = str(cand) if cand.is_file() else None
                else:
                    short_image = (
                        pool_paths[payload.index % len(pool_paths)]
                        if pool_paths else None
                    )

                v1_bridge.render_short(v1_bridge.ShortRenderInputs(
                    trimmed_short_path=sc.trimmed_video_path,
                    title_text=title,
                    output_path=str(out_dir / sc.output_filename),
                    work_dir=out_dir,
                    layout=_short_layout,
                    language=jc.language,
                    image_path=short_image,
                    brand_logo=sc.layout.brand_logo_path,
                    thumbnail_path=short_image,
                    font_file=(cfg.font_file if cfg else None),
                    font_size=(cfg.font_size if cfg else None),
                    text_color=(cfg.text_color if cfg else None),
                    section_pct=(cfg.section_pct.model_dump() if cfg else None),
                    card_style=(cfg.card_style.model_dump() if cfg else None),
                    follow_params=(cfg.follow_params.model_dump() if cfg else None),
                    watermark_text=wm_text,
                    watermark_opacity=wm_op,
                    watermark_position=wm_pos,
                    # Custom-template short: feed per-slot media + content + editor overrides
                    # (ignored by the built-in V1 composers, so harmless for non-custom).
                    template_media=(_tmedia if _is_short_custom else None),
                    main_media_slot=(_tmain if _is_short_custom else ""),
                    template_text_overrides=(_tover if _is_short_custom else None),
                    expected_kind=("short" if _is_short_custom else ""),
                    headline=(title if _is_short_custom else None),
                    headline_alt=((story.title_english or "") if (story and _is_short_custom) else None),
                    body=((story.summary or "") if (story and _is_short_custom) else None),
                    support_images=([short_image] if (short_image and _is_short_custom) else None),
                    # per-job visual edit (inline builder) -> render verbatim if present
                    template_html_override=(_v4_orch._load_html_override(job_id, "short", payload.index)
                                            if _is_short_custom else None),
                ))
            _set_state(job_id, state="done", msg="render complete")
            _se.emit_lane(_rr_env, "rerender", "compose", _se.EXITED)
            _se.emit_lane(_rr_env, "rerender", "encode", _se.ENTERED)
            _se.emit_lane(_rr_env, "rerender", "encode", _se.EXITED)
            print(f"[v4/render] job={job_id} target={tgt_label} done")
        except Exception as exc:
            _set_state(job_id, state="failed", msg=str(exc)[:300])
            for _st in ("slice", "compose", "encode"):
                _se.emit_lane(_rr_env, "rerender", _st, _se.FAILED)
            print(f"[v4/render] job={job_id} target={tgt_label} failed: {exc}")

    threading.Thread(target=_worker, daemon=True,
                     name=f"v4-render-{job_id}").start()
    return {"ok": True, "state": "queued", "target": tgt_label}


@router.get("/jobs/{job_id}/render/state")
def render_state(
    job_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    _owned_job(job_id, user, db)
    return _RENDERING.get(job_id) or {"state": "idle", "msg": "", "target": ""}


# ─── Watermarked download ───────────────────────────────────────────
# Works for both bulletin and shorts — same code path, just a different
# source file. Per-channel override via ?channel_id=N, falls back to
# the user's V4 defaults when no channel is named so the downloaded
# file always carries SOME branding instead of shipping clean.

@router.get("/jobs/{job_id}/download")
def download_with_watermark(
    job_id: int,
    target: str = "bulletin",         # "bulletin" | "short"
    index: int = 0,
    channel_id: Optional[int] = None, # null -> stamp with user defaults
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    from fastapi.responses import FileResponse
    from pipeline_v4 import watermark as _wm

    job = _owned_job(job_id, user, db)
    out_dir = _v4_dir_for(job)
    jc = _read_canvas(out_dir)

    if target == "bulletin":
        rel = jc.bulletin.output_filename
        label = "bulletin"
    elif target == "short":
        if not (0 <= index < len(jc.shorts)):
            raise HTTPException(400, f"short index {index} out of range")
        rel = jc.shorts[index].output_filename
        label = f"short_{index + 1:02d}"
    else:
        raise HTTPException(400, f"target must be 'bulletin' or 'short' (got {target!r})")

    source = out_dir / rel
    if not source.is_file():
        raise HTTPException(404, f"{label} render not on disk — run a render first")

    # Resolve optional channel scope for the stamp.
    channel = None
    if channel_id:
        channel = db.query(models.Channel).filter(
            models.Channel.id == channel_id,
            models.Channel.user_id == user.id,
        ).first()
        if not channel:
            raise HTTPException(404, "channel not found")

    stamped = _wm.stamp_for_channel(
        source_path=str(source),
        channel=channel,
        user=user,
        db=db,
    )

    download_name = f"{label}{'_' + channel.name.replace(' ', '_') if channel else ''}.mp4"
    return FileResponse(
        stamped, media_type="video/mp4", filename=download_name,
    )


@router.get("/jobs/{job_id}/preview/per-channel-video")
def preview_per_channel_video(
    job_id: int,
    target: str = "bulletin",          # "bulletin" | "short"
    index: int = 0,
    channel_id: Optional[int] = None,  # null -> user-default stamp
    token: Optional[str] = Query(default=None),
    authorization: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
):
    """Inline-playable per-channel branded PREVIEW of a rendered clip.

    Stamps the chosen channel's logo + watermark onto the rendered master
    (the SAME ``stamp_for_channel`` the download + publish paths use) and
    returns it INLINE (no attachment filename) so a plain ``<video src>`` can
    play it. Accepts ``?token=`` like ``serve_pool_file`` because a ``<video>``
    tag can't attach an Authorization header.

    NOTE: this previews the BRANDING layer (logo + watermark). The deeper
    anti-duplicate layer — the per-channel intro, zoom and audio nudge — is
    added by the publish-time branding pass, so the published file differs a
    touch more than this preview shows.
    """
    from fastapi.responses import FileResponse
    from pipeline_v4 import watermark as _wm

    # Auth: Bearer header OR ?token= query (mirrors serve_pool_file).
    raw = ""
    if authorization and authorization.lower().startswith("bearer "):
        raw = authorization.split(" ", 1)[1].strip()
    elif token:
        raw = token.strip()
    user: Optional[models.User] = None
    if raw:
        payload = auth.decode_token(raw)
        if payload:
            uid = int(payload.get("sub") or 0)
            if uid:
                u = db.query(models.User).filter(models.User.id == uid).first()
                if u and u.is_active:
                    user = u
    if user is None:
        if os.getenv("KAIZER_AUTH_REQUIRED", "false").lower() in ("1", "true", "yes", "on"):
            raise HTTPException(401, "Authentication required")
        user = auth.ensure_legacy_user(db)

    job = _owned_job(job_id, user, db)
    out_dir = _v4_dir_for(job)
    jc = _read_canvas(out_dir)
    if target == "bulletin":
        rel = jc.bulletin.output_filename
    elif target == "short":
        if not (0 <= index < len(jc.shorts)):
            raise HTTPException(400, f"short index {index} out of range")
        rel = jc.shorts[index].output_filename
    else:
        raise HTTPException(400, f"target must be 'bulletin' or 'short' (got {target!r})")

    source = out_dir / rel
    if not source.is_file():
        raise HTTPException(404, "render not on disk — run a render first")

    channel = None
    if channel_id:
        channel = db.query(models.Channel).filter(
            models.Channel.id == channel_id,
            models.Channel.user_id == user.id,
        ).first()
        if not channel:
            raise HTTPException(404, "channel not found")

    # FULL per-channel branded preview — the exact transforms publish applies
    # (logo + watermark + per-channel zoom + audio nudge + intro concat) so the
    # operator sees precisely what each channel will receive. Cached per
    # (clip, channel, brand-version) under the job dir; a re-render (newer
    # source) or a brand change (new version) busts it. Read-only w.r.t. the
    # publish pipeline.
    if channel is not None:
        from services import branding as _branding
        from services.brand_resolver import resolve_brand_profile
        try:
            # Per-CHANNEL intro override (this job's inline picker) wins over the
            # channel's own intro — so the preview matches publish.
            _ov = {}
            try:
                _ov = json.loads(getattr(job, "intro_overrides", None) or "{}") or {}
            except Exception:
                _ov = {}
            resolved = resolve_brand_profile(
                db, int(channel.id),
                job_intro_asset_id=_ov.get(str(channel.id)),
            )
        except Exception as exc:
            raise HTTPException(500, f"brand resolve failed: {exc}")
        preview_dir = out_dir / "_preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        cache_path = preview_dir / f"{target}_{index}_{channel.id}_{resolved.version}.mp4"
        fresh = (
            cache_path.is_file()
            and cache_path.stat().st_size > 0
            and cache_path.stat().st_mtime >= source.stat().st_mtime
        )
        if not fresh:
            try:
                _branding.brand_local_for_preview(str(source), str(cache_path), resolved)
            except Exception as exc:
                raise HTTPException(500, f"preview render failed: {str(exc)[:200]}")
        return FileResponse(str(cache_path), media_type="video/mp4")

    # No channel named → light user-default stamp (logo + watermark only).
    stamped = _wm.stamp_for_channel(
        source_path=str(source), channel=None, user=user, db=db,
    )
    return FileResponse(stamped, media_type="video/mp4")


# ─── Render ALL channel videos (batch, each with its own intro) ──────

class PrepareChannelsIn(BaseModel):
    target: str = "bulletin"   # "bulletin" | "short"
    index: int = 0


@router.post("/jobs/{job_id}/prepare-channel-videos")
def prepare_channel_videos(
    job_id: int,
    payload: PrepareChannelsIn,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Pre-render the FULL branded clip for EVERY selected channel (each with
    its own logo/watermark/zoom/nudge + INTRO) so the editor shows a ready
    video per channel. Runs in a background daemon thread, serialized GLOBALLY
    (one batch at a time) so it never fans out N ffmpegs at once and starves a
    live re-render. Poll ``/prepare-channel-videos/state``. Reuses
    ``brand_local_for_preview`` (the same transforms publish applies); does NOT
    touch the publish/upload flow."""
    job = _owned_job(job_id, user, db)
    out_dir = _v4_dir_for(job)
    jc = _read_canvas(out_dir)
    target, index = payload.target, payload.index
    if target == "bulletin":
        rel = jc.bulletin.output_filename
    elif target == "short":
        if not (0 <= index < len(jc.shorts)):
            raise HTTPException(400, f"short index {index} out of range")
        rel = jc.shorts[index].output_filename
    else:
        raise HTTPException(400, "target must be 'bulletin' or 'short'")
    source = out_dir / rel
    if not source.is_file():
        raise HTTPException(404, "render not on disk — run a render first")

    # Connected channels, scoped to the job's chosen channels when set.
    channels = [c for c in db.query(models.Channel).filter(models.Channel.user_id == user.id).all()
                if c.oauth_token is not None and bool(c.oauth_token.refresh_token_enc)]
    try:
        _tids = json.loads(job.target_channel_ids) if getattr(job, "target_channel_ids", None) else []
    except Exception:
        _tids = []
    _target_ids = {int(t) for t in _tids} if isinstance(_tids, list) else set()
    if _target_ids:
        scoped = [c for c in channels if c.id in _target_ids]
        if scoped:
            channels = scoped
    if not channels:
        raise HTTPException(400, "no connected channels to prepare")

    with _PREPARE_LOCK:
        cur = _PREPARE.get(job_id)
        # Reject when a batch is queued OR running — we set "queued" right here
        # under the same lock, so a second concurrent request sees it and 409s
        # before it can spawn a duplicate worker (closes the check→spawn race).
        if cur and cur.get("state") in ("queued", "running"):
            raise HTTPException(409, "a channel-video batch is already running for this job")
        _PREPARE[job_id] = {
            "state": "queued", "done": 0, "total": len(channels),
            "target": target, "index": index,
            "channels": [{"channel_id": c.id, "name": c.name or f"Channel #{c.id}",
                          "ready": False, "error": ""} for c in channels],
        }

    src_str = str(source)
    # SNAPSHOT at spawn time — the worker renders THIS channel list; a later
    # edit to job.target_channel_ids won't retro-change an in-flight batch
    # (re-run "Render all" to pick up a new selection).
    ch_ids = [(c.id, c.name or f"Channel #{c.id}") for c in channels]
    preview_dir = out_dir / "_preview"
    _uid = getattr(user, "id", None)
    # Per-CHANNEL intro overrides for this job (channel_id → asset_id). Each
    # channel's batch render uses its OWN override so it matches publish. Empty
    # map = each channel uses its own assigned intro.
    _job_intro_map = {}
    try:
        _job_intro_map = json.loads(getattr(job, "intro_overrides", None) or "{}") or {}
    except Exception:
        _job_intro_map = {}

    # All _PREPARE[job_id] mutations go through these so the state endpoint
    # never reads a half-written dict (it copies under the same lock).
    def _prep_set(**kv):
        with _PREPARE_LOCK:
            st = _PREPARE.get(job_id)
            if st is not None:
                st.update(kv)

    def _prep_mark(idx, **kv):
        with _PREPARE_LOCK:
            st = _PREPARE.get(job_id)
            if st is not None and 0 <= idx < len(st.get("channels", [])):
                st["channels"][idx].update(kv)

    def _worker():
        from services import branding as _b
        from services.brand_resolver import resolve_brand_profile
        from database import SessionLocal as _SL
        from services import stage_events as _se
        # One conveyor vehicle per channel on the "channel_render" lane
        # (queued → rendering → ready). channel_id makes each its own unit.
        envs = {
            cid: _se.Envelope(tenant_id=_uid, user_id=_uid, job_id=job_id,
                              clip_id=None, channel_id=cid, label=cname)
            for (cid, cname) in ch_ids
        }
        # Global serialize: one batch cascade at a time across ALL jobs so we
        # never fan out N ffmpegs. Bounded — each brand_local_for_preview runs
        # ffmpeg with an encode timeout, and this `with` releases on any exit.
        with _PREPARE_RUN_LOCK:
            db2 = _SL()
            try:
                _prep_set(state="running")
                preview_dir.mkdir(parents=True, exist_ok=True)
                try:
                    src_mtime = os.path.getmtime(src_str) if os.path.isfile(src_str) else 0
                except OSError:
                    src_mtime = 0
                # All channels enter the conveyor "queued" up-front; each then
                # drives queued → rendering → ready as the serial loop reaches it.
                for cid in envs:
                    _se.emit_lane(envs[cid], "channel_render", "queued", _se.ENTERED)
                for i, (cid, cname) in enumerate(ch_ids):
                    env = envs[cid]
                    _se.emit_lane(env, "channel_render", "queued", _se.EXITED)
                    _se.emit_lane(env, "channel_render", "rendering", _se.ENTERED)
                    try:
                        resolved = resolve_brand_profile(db2, cid, job_intro_asset_id=_job_intro_map.get(str(cid)))
                        cache_path = preview_dir / f"{target}_{index}_{cid}_{resolved.version}.mp4"
                        fresh = (cache_path.is_file() and cache_path.stat().st_size > 0
                                 and cache_path.stat().st_mtime >= src_mtime)
                        if not fresh:
                            _b.brand_local_for_preview(src_str, str(cache_path), resolved)
                        _se.emit_lane(env, "channel_render", "rendering", _se.EXITED)
                        _se.emit_lane(env, "channel_render", "ready", _se.ENTERED)
                        _prep_mark(i, ready=True)
                    except Exception as exc:
                        # Clear any half-open tx so the reused session stays
                        # usable for the next channel.
                        try:
                            db2.rollback()
                        except Exception:
                            pass
                        _se.emit_lane(env, "channel_render", "rendering", _se.FAILED)
                        _prep_mark(i, error=str(exc)[:160])
                    _prep_set(done=i + 1)
                _prep_set(state="done")
            except Exception as exc:
                _prep_set(state="failed", msg=str(exc)[:200])
            finally:
                db2.close()

    threading.Thread(target=_worker, daemon=True, name=f"v4-prepare-{job_id}").start()
    return {"ok": True, "state": "queued", "total": len(ch_ids)}


@router.get("/jobs/{job_id}/prepare-channel-videos/state")
def prepare_channel_videos_state(
    job_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    _owned_job(job_id, user, db)
    # Copy under the lock so we never hand back a dict the worker is mid-write on.
    import copy as _copy
    with _PREPARE_LOCK:
        st = _PREPARE.get(job_id)
        if st is None:
            return {"state": "idle", "done": 0, "total": 0, "channels": []}
        return _copy.deepcopy(st)


# ─── SEO text download (plain, or account-specific with socials) ────

def _format_seo_txt(title: str, description: str,
                    keywords: list, hashtags: list) -> str:
    """Lay the SEO out as a copy-paste-friendly .txt for manual upload."""
    parts = [
        "TITLE",
        (title or "").strip(),
        "",
        "DESCRIPTION",
        (description or "").strip(),
    ]
    ht = [h for h in (hashtags or []) if str(h).strip()]
    if ht:
        parts += ["", "HASHTAGS", " ".join(str(h).strip() for h in ht)]
    kw = [k for k in (keywords or []) if str(k).strip()]
    if kw:
        parts += ["", "TAGS", ", ".join(str(k).strip() for k in kw)]
    return "\n".join(parts).strip() + "\n"


@router.get("/jobs/{job_id}/seo/download")
def download_seo_text(
    job_id: int,
    target: str = "bulletin",          # "bulletin" | "short"
    index: int = 0,
    channel_id: Optional[int] = None,  # null -> plain (no socials)
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Download a canvas's SEO as a plain-text file for manual upload.

    With ``channel_id`` set to one of the user's CONNECTED accounts, the
    SEO is run through the SAME publish-time composer used by the upload
    worker, so that account's social links (and brand footer / mandatory
    tags) are injected into the description — "account-specific". Without
    it, the raw generated SEO is returned — "plain"."""
    from fastapi.responses import PlainTextResponse

    job = _owned_job(job_id, user, db)
    out_dir = _v4_dir_for(job)
    jc = _read_canvas(out_dir)

    if target == "bulletin":
        canv = jc.bulletin
        label = "bulletin"
        publish_kind = "video"
    elif target == "short":
        if not (0 <= index < len(jc.shorts)):
            raise HTTPException(400, f"short index {index} out of range")
        canv = jc.shorts[index]
        label = f"short_{index + 1:02d}"
        publish_kind = "short"
    else:
        raise HTTPException(400, f"target must be 'bulletin' or 'short' (got {target!r})")

    if canv.seo is None:
        raise HTTPException(404, "no SEO generated for this target yet — regenerate first")
    generic = json.loads(canv.seo.model_dump_json())

    suffix = ""
    if channel_id:
        channel = db.query(models.Channel).filter(
            models.Channel.id == channel_id,
            models.Channel.user_id == user.id,
        ).first()
        if not channel:
            raise HTTPException(404, "channel not found")
        try:
            from seo.composer import compose as _seo_compose
            composed = _seo_compose(generic, channel, publish_kind=publish_kind)
        except Exception as exc:
            raise HTTPException(500, f"SEO compose failed: {exc}")
        title       = composed.get("title", "") or ""
        description = composed.get("description", "") or ""
        keywords    = composed.get("keywords") or []
        hashtags    = composed.get("hashtags") or []
        suffix = "_" + (channel.name or f"ch{channel_id}").replace(" ", "_")
    else:
        title       = generic.get("title", "") or ""
        description = generic.get("description", "") or ""
        keywords    = generic.get("keywords") or []
        hashtags    = generic.get("hashtags") or []

    body = _format_seo_txt(title, description, keywords, hashtags)
    fname = f"{label}{suffix}_seo.txt"
    return PlainTextResponse(
        body,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


# ─── AI thumbnail (Nano Banana via Gemini) ──────────────────────────

class ThumbnailGenIn(BaseModel):
    """Trigger AI thumbnail generation for one canvas.

    First call: omit ``tweak`` -> Gemini writes a full prompt from SEO.
    Subsequent calls: pass a ``tweak`` string -> Gemini reuses the
    previous prompt (stored on disk) and applies the tweak. Cheaper
    than starting from scratch every time.

    ``style`` selects one of pipeline_v4.thumbnail_styles.STYLES.
    ``reference_asset_ids`` is the list of UserAsset rows (typically in
    folder_path='thumbnail_refs') whose images become Nano Banana
    multi-image input — preserves the operator's reporter / subject
    face. Pass [] for styles that don't need a reference.

    ``provider`` chooses the image generator. "gemini" (default,
    Nano Banana — supports reference images for face preservation)
    or "openai" (gpt-image-1 — no reference support, but stronger
    photorealistic results for symbolic stories). Reference-style
    requests with provider=openai are coerced to gemini server-side
    because OpenAI's API doesn't accept image input on /images."""
    target: str = "bulletin"      # "bulletin" | "short"
    index: int = 0                 # for target="short"
    tweak: str = ""                # natural-language nudge for iteration
    style: str = "symbolic"
    reference_asset_ids: list[int] = []
    provider: str = "gemini"       # "gemini" | "openai"
    # Director engine preset — picks which brain runs Pass 1 + Pass 2.
    # "gemini" (default — cheap), "hybrid" (Claude plans, Gemini prompts),
    # or "claude" (premium — Claude does both). See thumbnail_director
    # __init__.py DIRECTOR_PRESETS for the cost / quality tradeoffs.
    engine: str = "gemini"         # "gemini" | "hybrid" | "claude"
    # Shout-text rendering mode.
    # "ai"      — image model renders the text inside the plate (more
    #             dramatic / integrated; Indic ligatures may garble).
    # "overlay" — Pillow paints native-script text after generation
    #             (Indic typography guaranteed correct; looks like a
    #             sticker on top of the plate).
    # Default "ai" matches operator preference (2026-06-09).
    text_mode: str = "ai"          # "ai" | "overlay"
    # Per-thumbnail language override. Empty / unknown = use the
    # canvas's language (the most common case). Set to an ISO code
    # (te / hi / ta / kn / ml / bn / mr / gu / en) to force the
    # shout text in that language's native script for this generation
    # only — useful for cross-language A/B or one-off English fallback.
    language: str = ""             # "" = use canvas language


@router.get("/thumbnail/styles")
def list_thumbnail_styles(
    user: models.User = Depends(auth.current_user),
) -> list[dict]:
    """Return the thumbnail-style catalog as JSON. Powers the editor's
    style picker — the frontend never hard-codes style keys."""
    from pipeline_v4.thumbnail_styles import list_styles
    return list_styles()


@router.get("/jobs/{job_id}/thumbnail")
def get_existing_thumbnail(
    job_id: int,
    target: str = "bulletin",
    index: int = 0,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Return the existing AI thumbnail's URL + prompt if the file is
    on disk. Used by the editor on mount so a page refresh doesn't
    wipe the preview the operator just generated. Returns
    ``{"url": "", "prompt": ""}`` (no 404) when the thumbnail hasn't
    been generated yet — the frontend treats empty as "show Generate
    button"."""
    job = _owned_job(job_id, user, db)
    out_dir = _v4_dir_for(job)

    if target == "bulletin":
        name = "bulletin_thumb_ai.jpg"
    elif target == "short":
        # Same naming convention as generate_thumbnail() below.
        name = f"short_{index + 1:02d}_thumb_ai.jpg"
    else:
        raise HTTPException(400, f"target must be 'bulletin' or 'short' (got {target!r})")

    jpg = out_dir / name
    if not jpg.is_file():
        return {"url": "", "prompt": ""}

    backend_root = Path(__file__).resolve().parent.parent
    try:
        rel = jpg.resolve().relative_to(_media_root().resolve())
        url = f"/media/{rel.as_posix()}?t={int(jpg.stat().st_mtime)}"
    except ValueError:
        # File exists but not under /output — unusual; fall back to no URL.
        url = ""

    prompt_path = jpg.with_suffix(jpg.suffix + ".prompt.txt")
    prompt = ""
    if prompt_path.is_file():
        try:
            prompt = prompt_path.read_text(encoding="utf-8")
        except OSError:
            prompt = ""

    # Bare-plate URL — used by the manual drag editor as the canvas
    # background. Convention: ``<name>.plate.<ext>``. Empty when the
    # thumbnail was produced in text_mode=ai (no clean plate to keep).
    from pipeline_v4.thumbnail_ai import _plate_path_for
    plate_url = ""
    plate_file = Path(_plate_path_for(str(jpg)))
    if plate_file.is_file():
        try:
            plate_rel = plate_file.resolve().relative_to(_media_root().resolve())
            plate_url = f"/media/{plate_rel.as_posix()}?t={int(plate_file.stat().st_mtime)}"
        except ValueError:
            plate_url = ""

    return {"url": url, "prompt": prompt, "plate_url": plate_url}


# ─── Manual drag-to-position text overlay editor ─────────────────────


class ThumbnailReoverlayIn(BaseModel):
    """Re-paint the text overlay on an existing AI-generated plate.

    Lets the operator drag-position the text box, edit the text
    content, change colour preset, and adjust font sizing without
    burning another AI call. Server loads the saved bare plate,
    re-runs Pillow with the operator's geometry, and overwrites
    the final thumbnail.

    ``x_pct``, ``y_pct`` are the TOP-LEFT corner of the text box
    as a percentage of the canvas. ``w_pct``, ``h_pct`` are the
    box dimensions, also as percentages. All four are 0-100.
    """
    target: str = "bulletin"          # "bulletin" | "short"
    index: int = 0
    text: str = ""                     # native-script shout text
    x_pct: float = 0.0
    y_pct: float = 72.0
    w_pct: float = 100.0
    h_pct: float = 28.0
    color_preset: str = "white_on_red"
    language: str = ""                 # "" = use canvas language


@router.post("/jobs/{job_id}/thumbnail/reoverlay")
def reoverlay_thumbnail(
    job_id: int,
    payload: ThumbnailReoverlayIn,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Re-paint the text overlay on a previously-generated thumbnail.

    Cheap (~200ms Pillow pass, zero AI calls). Requires that the
    original generation ran in ``text_mode=overlay`` so a bare plate
    was saved alongside the final JPG.
    """
    from pipeline_v4 import thumbnail_ai as _tai
    job = _owned_job(job_id, user, db)
    out_dir = _v4_dir_for(job)
    jc = _read_canvas(out_dir)

    if payload.target == "bulletin":
        out_name = "bulletin_thumb_ai.jpg"
    elif payload.target == "short":
        if not (0 <= payload.index < len(jc.shorts)):
            raise HTTPException(400, f"short index {payload.index} out of range")
        out_name = f"short_{payload.index + 1:02d}_thumb_ai.jpg"
    else:
        raise HTTPException(400, f"target must be 'bulletin' or 'short' (got {payload.target!r})")

    out_path = out_dir / out_name
    plate_path = Path(_tai._plate_path_for(str(out_path)))
    if not plate_path.is_file():
        raise HTTPException(
            409,
            "No bare plate saved for this thumbnail. Re-generate it with "
            "Text=Overlay enabled so a clean plate is kept for editing.",
        )
    if not (payload.text or "").strip():
        raise HTTPException(400, "text is required")

    # Resolve effective language — operator override > canvas default.
    eff_lang = (payload.language or "").strip().lower()
    if eff_lang:
        try:
            import languages as _langs
            eff_lang = _langs.get(eff_lang).code
        except Exception:
            eff_lang = ""
    if not eff_lang:
        eff_lang = jc.language or "te"

    ok = _tai.overlay_native_text(
        image_path=str(out_path),
        text=payload.text.strip(),
        language=eff_lang,
        style="symbolic",  # symbolic geometry is overridden anyway
        geometry_override={
            "x_pct": payload.x_pct,
            "y_pct": payload.y_pct,
            "w_pct": payload.w_pct,
            "h_pct": payload.h_pct,
        },
        color_preset=(payload.color_preset or "white_on_red").strip().lower(),
        source_plate_path=str(plate_path),
    )
    if not ok:
        raise HTTPException(502, "overlay re-paint failed — see server log")

    backend_root = _backend_root()
    rel = out_path.resolve().relative_to(_media_root().resolve())
    mt = int(out_path.stat().st_mtime)
    return {
        "ok": True,
        "url": f"/media/{rel.as_posix()}?t={mt}",
        "target": payload.target,
        "index": payload.index,
    }


@router.post("/jobs/{job_id}/thumbnail/generate")
def generate_thumbnail(
    job_id: int,
    payload: ThumbnailGenIn,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Run prompt-writer Gemini → Nano Banana image gen. Returns the
    public URL of the saved JPG so the editor can preview it. Also
    updates ``Clip.thumb_path`` if a matching Clip row exists.

    Stores the produced prompt next to the JPG as ``<name>.prompt.txt``
    so a follow-up tweak call can pick it up and iterate without
    re-deriving from SEO context."""
    from pipeline_v4 import thumbnail_ai as _tai
    job = _owned_job(job_id, user, db)
    out_dir = _v4_dir_for(job)
    jc = _read_canvas(out_dir)

    if payload.target == "bulletin":
        canv = jc.bulletin
        out_name = "bulletin_thumb_ai.jpg"
        clip_index = 0
    elif payload.target == "short":
        if not (0 <= payload.index < len(jc.shorts)):
            raise HTTPException(400, f"short index {payload.index} out of range")
        canv = jc.shorts[payload.index]
        out_name = f"short_{payload.index + 1:02d}_thumb_ai.jpg"
        clip_index = payload.index + 1
    else:
        raise HTTPException(400, f"target must be 'bulletin' or 'short' (got {payload.target!r})")

    out_path = out_dir / out_name
    prompt_path = out_path.with_suffix(out_path.suffix + ".prompt.txt")

    previous_prompt = ""
    tweak = (payload.tweak or "").strip()
    if tweak and prompt_path.is_file():
        try:
            previous_prompt = prompt_path.read_text(encoding="utf-8")
        except Exception:
            previous_prompt = ""

    # Resolve reference-asset rows to absolute file paths. Filter to
    # this operator's own assets so the route can't be used to read
    # someone else's uploads. Unknown / non-image assets are skipped
    # silently (the prompt-writer is told how many references actually
    # made it through, so it doesn't promise face preservation for a
    # reference that didn't load).
    reference_paths: list[str] = []
    if payload.reference_asset_ids:
        rows = (
            db.query(models.UserAsset)
              .filter(
                  models.UserAsset.id.in_(payload.reference_asset_ids),
                  models.UserAsset.user_id == user.id,
              )
              .all()
        )
        # Preserve operator-specified order; SQLAlchemy result order is
        # implementation-defined, but the visual difference between
        # "left subject" and "right subject" for split-screen matters.
        by_id = {a.id: a for a in rows}
        for aid in payload.reference_asset_ids:
            a = by_id.get(aid)
            if not a:
                continue
            p = a.file_path or ""
            if p and os.path.isfile(p):
                reference_paths.append(p)

    # Provider dispatch. OpenAI doesn't accept image inputs on its
    # /images endpoint, so reference-image styles (reporter_led /
    # split_screen / etc.) get auto-coerced to Gemini regardless of
    # the picker. Symbolic / text-dominant styles can use either.
    chosen_provider = (payload.provider or "gemini").strip().lower()
    if chosen_provider not in {"gemini", "openai"}:
        chosen_provider = "gemini"
    if chosen_provider == "openai" and reference_paths:
        print(f"[v4/thumb] provider=openai requested but {len(reference_paths)} "
              f"reference image(s) supplied -- coercing to gemini for face preservation",
              flush=True)
        chosen_provider = "gemini"

    final_prompt = ""
    saved: Optional[str] = None
    if chosen_provider == "openai":
        # Build a richer prompt from canvas SEO + transcript so the
        # OpenAI thumbnail isn't generic. Title is the strongest
        # signal; description provides grounding. Names hint helps
        # OpenAI render specific subjects when applicable.
        seo = getattr(canv, "seo", None)
        title_str = ""
        brief_str = ""
        if seo is not None:
            title_str = (getattr(seo, "title", "") or "").strip()
            brief_str = (getattr(seo, "description", "") or "").strip()[:600]
        if not title_str:
            # Fall back to the first story's headline.
            stories_list = getattr(canv, "stories", []) or []
            if stories_list:
                s0 = stories_list[0]
                title_str = (getattr(s0, "title_native", "")
                             or getattr(s0, "title_english", "") or "").strip()
        try:
            from express.ai_image import generate_thumbnail as _openai_gen
            saved = _openai_gen(
                api_key="",  # falls back to OPENAI_API_KEY env
                title=title_str or "Telugu news headline",
                brief=brief_str,
                names_hint="",
                output_path=str(out_path),
                size="1536x1024",
                quality="medium",
                timeout_s=90,
            )
            # Persist a readable prompt so a follow-up tweak iterates
            # cleanly. OpenAI's styled wrapper isn't exposed, so we
            # reconstruct what we sent.
            final_prompt = f"[openai gpt-image-1]\nTitle: {title_str}\n\n{brief_str}"
        except Exception as exc:
            print(f"[v4/thumb] openai thumbnail failed: {exc}", flush=True)
            saved = None
    else:
        # Per-thumbnail language override. The frontend can send
        # ``payload.language`` to force a different shout-text script
        # without touching the canvas. Validate via languages.get so a
        # stale frontend can't pass garbage — unknown codes fall back
        # to the canvas language. Empty = use canvas language.
        _eff_lang = (payload.language or "").strip().lower()
        if _eff_lang:
            try:
                import languages as _langs
                _eff_lang = _langs.get(_eff_lang).code
            except Exception:
                _eff_lang = ""
        if not _eff_lang:
            _eff_lang = jc.language or "te"
        saved, final_prompt = _tai.make_thumbnail_for_canvas(
            canvas=canv, out_path=str(out_path),
            language=_eff_lang,
            previous_prompt=previous_prompt,
            tweak=tweak,
            style=(payload.style or "symbolic"),
            reference_paths=reference_paths,
            engine=(payload.engine or "gemini").strip().lower(),
            text_mode=(payload.text_mode or "ai").strip().lower(),
        )

    if not saved or not out_path.is_file():
        # Pull the verbatim Gemini error so the user can see what
        # actually went wrong (most commonly: 429 prepayment depleted
        # or 403 model not enabled on the project).
        raw = getattr(_tai.generate_thumbnail_image, "last_error", "")
        hint = ""
        if chosen_provider == "openai":
            hint = (
                "OpenAI gpt-image-1 didn't return an image. Check "
                "OPENAI_API_KEY in .env (must have image-gen access) or "
                "switch the provider toggle back to Gemini."
            )
        elif "RESOURCE_EXHAUSTED" in raw or "quota" in raw.lower() or " 429" in raw:
            hint = (
                "Gemini quota/credits exhausted. Either top up at "
                "https://ai.studio/projects or switch GEMINI_API_KEY in "
                ".env to a free-tier key from https://aistudio.google.com/apikey."
            )
        elif "PERMISSION_DENIED" in raw or " 403" in raw:
            hint = (
                "Image-gen model not enabled for this project. Enable "
                "'gemini-2.5-flash-image' or use a different GEMINI_API_KEY."
            )
        elif "API key" in raw or "401" in raw:
            hint = "GEMINI_API_KEY appears to be missing or invalid — check .env."
        detail = hint or (raw[:300] if raw else f"{chosen_provider} didn't return an image.")
        raise HTTPException(502, detail)

    # Persist the prompt for the next iteration. Best-effort — a failed
    # write only means the next tweak falls back to a from-scratch call.
    try:
        prompt_path.write_text(final_prompt, encoding="utf-8")
    except Exception as exc:
        print(f"[v4/thumb] prompt persist soft-fail: {exc}", flush=True)

    try:
        clip = db.query(models.Clip).filter(
            models.Clip.job_id == job_id,
            models.Clip.clip_index == clip_index,
        ).first()
        if clip:
            clip.thumb_path = str(out_path)
            db.commit()
    except Exception as exc:
        print(f"[v4/thumb] Clip.thumb_path update soft-fail: {exc}", flush=True)

    backend_root = _backend_root()
    rel = out_path.resolve().relative_to(_media_root().resolve())
    mt = int(out_path.stat().st_mtime)
    return {
        "ok": True,
        "thumb_path": str(out_path),
        "url": f"/media/{rel.as_posix()}?t={mt}",
        "target": payload.target,
        "index": payload.index,
        "prompt": final_prompt,
        "iterated": bool(previous_prompt and tweak),
    }


# ─── SEO regenerate ─────────────────────────────────────────────────

class SeoRegenerateIn(BaseModel):
    """Trigger one Gemini call for either the bulletin or a single short."""
    target: str = "bulletin"      # "bulletin" | "short"
    index: int = 0                 # for target="short"
    style_source_id: Optional[int] = None   # competitor style ref (SEO Settings)


def _resolve_style_source(db: Session, user: models.User, style_source_id):
    """Load a user-owned style-reference Channel for SEO emulation.

    Returns the Channel (its ``corpus``/``title_formula``/``desc_style``
    drive the writing voice) or ``None`` when no id was supplied. Raises
    404 when the id is unknown / not owned, and 409 when it points at a
    connected publish account (those are NOT style references)."""
    if not style_source_id:
        return None
    ch = (db.query(models.Channel)
          .filter(models.Channel.id == style_source_id,
                  models.Channel.user_id == user.id)
          .first())
    if ch is None:
        raise HTTPException(404, "style_source not found")
    tok = ch.oauth_token
    if tok is not None and (tok.refresh_token_enc or "").strip():
        raise HTTPException(409, "style_source points at a connected account, "
                                 "not a style reference")
    return ch


def _seo_belt(job_id, user_id, stage, status):
    """Emit a SEO-lane transition for the admin Pipeline Flow belt. Best-effort
    — telemetry must never break the SEO endpoints."""
    try:
        from services import stage_events as _se
        env = _se.Envelope(tenant_id=user_id, user_id=user_id, job_id=job_id,
                           clip_id=None, channel_id=None, label=f"job {job_id}")
        _se.emit_lane(env, "seo", stage, status)
    except Exception:
        pass


@router.post("/jobs/{job_id}/seo/regenerate")
def regenerate_seo(
    job_id: int,
    payload: SeoRegenerateIn,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Re-run V4's Gemini SEO for one target. Writes the result back
    into canvas.json so the editor sees it on next load. Skips when
    the user has already edited it (``edited_by_user=True``) unless
    they pass that as a fresh request via the UI."""
    job = _owned_job(job_id, user, db)
    out_dir = _v4_dir_for(job)
    jc = _read_canvas(out_dir)
    style_source = _resolve_style_source(db, user, payload.style_source_id)
    _seo_belt(job_id, user.id, "generate", "entered")

    if payload.target == "bulletin":
        canv = jc.bulletin
        story0 = canv.stories[0] if canv.stories else None
        body = "\n".join(
            f"- {s.title_native or s.title_english}"
            for s in canv.stories if s.title_native or s.title_english
        )
        seo_dict = seo_provider.generate_seo(seo_provider.SeoInput(
            kind="bulletin",
            language=jc.language,
            title_native=(story0.title_native if story0 else "") or "",
            title_english=(story0.title_english if story0 else "") or "",
            summary=(story0.summary if story0 else "") or "",
            body=body,
            style_source_id=payload.style_source_id,
        ), style_source=style_source)
        canv.seo = CanvasSEO(**seo_dict)
        canv.seo.edited_by_user = False
    elif payload.target == "short":
        if not (0 <= payload.index < len(jc.shorts)):
            raise HTTPException(400, f"short index {payload.index} out of range")
        canv = jc.shorts[payload.index]
        story0 = canv.stories[0] if canv.stories else None
        if not story0:
            raise HTTPException(400, "short has no story metadata")
        seo_dict = seo_provider.generate_seo(seo_provider.SeoInput(
            kind="short",
            language=jc.language,
            title_native=story0.title_native or "",
            title_english=story0.title_english or "",
            summary=story0.summary or "",
            style_source_id=payload.style_source_id,
        ), style_source=style_source)
        canv.seo = CanvasSEO(**seo_dict)
        canv.seo.edited_by_user = False
    else:
        _seo_belt(job_id, user.id, "generate", "failed")
        raise HTTPException(400, f"target must be 'bulletin' or 'short' (got {payload.target!r})")

    _write_canvas(out_dir, jc)
    # generate done → score (the checker runs inside generate_seo) → done.
    _seo_belt(job_id, user.id, "generate", "exited")
    _seo_belt(job_id, user.id, "score", "entered")
    _seo_belt(job_id, user.id, "score", "exited")
    return {"ok": True, "seo": json.loads(canv.seo.model_dump_json())}


@router.get("/jobs/{job_id}/seo/per-channel-preview")
def per_channel_seo_preview(
    job_id: int,
    mode: str = "per_channel",
    target: str = "bulletin",
    index: int = 0,
    generate: bool = False,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Preview the EXACT SEO that would be sent to each connected channel + WHY.

    mode='shared'      -> every channel shows the identical base SEO (free).
    mode='per_channel' -> per channel. ``generate=False`` (default) = the cheap
                          tags-only adapt (instant, no AI; same headline).
                          ``generate=True`` = FULL per-channel generation — a
                          distinct title/description/tags written by the real
                          SEO engine (channel voice + its winning keywords) and
                          scored for SEO/CTR. One AI call per channel.
    """
    job = _owned_job(job_id, user, db)
    jc = _read_canvas(_v4_dir_for(job))
    if target == "bulletin":
        canv = jc.bulletin
    elif target == "short" and 0 <= index < len(jc.shorts):
        canv = jc.shorts[index]
    else:
        raise HTTPException(400, "invalid target/index")
    base_seo = json.loads(canv.seo.model_dump_json()) if getattr(canv, "seo", None) else {}

    # Content keywords from the clip's own stories (advisory; safe if unavailable).
    stories = getattr(canv, "stories", []) or []
    content = "\n".join(((getattr(s, "title_native", "") or "") + " " +
                         (getattr(s, "summary", "") or "")) for s in stories)
    try:
        from seo.score_checker import _content_keywords
        ckw = _content_keywords(content, top=10)
    except Exception:
        ckw = []

    channels = [c for c in db.query(models.Channel).filter(models.Channel.user_id == user.id).all()
                if c.oauth_token is not None and bool(c.oauth_token.refresh_token_enc)]  # our publish channels only
    # Scope to the channels chosen for THIS job ("Choose channels" step) when
    # present — that selection is exactly why we capture it. Fall back to ALL
    # connected channels when the job pinned none (or none of them are linked).
    try:
        _tids = json.loads(job.target_channel_ids) if getattr(job, "target_channel_ids", None) else []
    except Exception:
        _tids = []
    _target_ids = {int(t) for t in _tids} if isinstance(_tids, list) else set()
    if _target_ids:
        _scoped = [c for c in channels if c.id in _target_ids]
        if _scoped:
            channels = _scoped
    fb: dict = {}
    if mode == "per_channel":
        from seo.performance_profile import build_channel_profile
        for ch in channels:
            try:
                p = build_channel_profile(db, ch.id)
                if p.get("ready"):
                    fb[ch.id] = {"winning_keywords": p.get("winning_keywords", [])}
            except Exception:
                pass

    from seo.per_channel import build_channel_previews
    previews = build_channel_previews(
        base_seo=base_seo, channels=channels,
        feedback_by_channel=fb, content_keywords=ckw, mode=mode,
        generate=bool(generate), content_text=content, language=(jc.language or "te"),
    )
    return {"mode": mode, "base_seo": base_seo, "channels": previews, "generated": bool(generate)}


class PerChannelApplyIn(BaseModel):
    clip_id: int
    mode: str = "per_channel"     # "per_channel" (paid) | "shared" (free)
    target: str = "bulletin"
    index: int = 0


@router.post("/jobs/{job_id}/seo/apply-per-channel")
def apply_per_channel_seo(
    job_id: int,
    payload: PerChannelApplyIn,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Persist per-channel SEO so each channel PUBLISHES with its own title/tags.

    Writes the adapted SEO into the clip's ``seo_variants`` keyed by channel id,
    each marked ``_per_channel`` so the publish composer (upload_dispatch.
    _compose_metadata) uses it for that channel. mode='shared' removes our
    markers (reverts that clip to the single shared SEO — the free default).
    NOTE: per-channel is the PAID tier; bill/gate at the billing layer (the UI
    labels it 'paid'). Idempotent; never touches non-marked legacy variants.
    """
    import json as _json
    job = _owned_job(job_id, user, db)
    clip = db.query(models.Clip).filter(models.Clip.id == payload.clip_id).first()
    if clip is None or clip.job_id != job.id:
        raise HTTPException(404, "clip not found for this job")
    _seo_belt(job_id, user.id, "per_channel", "entered")

    try:
        variants = _json.loads(clip.seo_variants or "{}") if clip.seo_variants else {}
        if not isinstance(variants, dict):
            variants = {}
    except Exception:
        variants = {}
    # Clear any prior per-channel-marked entries (clean slate); keep legacy ones.
    variants = {k: v for k, v in variants.items()
                if not (isinstance(v, dict) and v.get("_per_channel"))}

    if payload.mode == "shared":
        clip.seo_variants = _json.dumps(variants)
        db.add(clip); db.commit()
        _seo_belt(job_id, user.id, "per_channel", "exited")
        return {"ok": True, "mode": "shared", "applied": 0}

    # per_channel — compute adapted SEO per channel (same as the preview).
    jc = _read_canvas(_v4_dir_for(job))
    if payload.target == "bulletin":
        canv = jc.bulletin
    elif payload.target == "short" and 0 <= payload.index < len(jc.shorts):
        canv = jc.shorts[payload.index]
    else:
        raise HTTPException(400, "invalid target/index")
    base_seo = json.loads(canv.seo.model_dump_json()) if getattr(canv, "seo", None) else {}
    stories = getattr(canv, "stories", []) or []
    content = "\n".join(((getattr(s, "title_native", "") or "") + " " +
                         (getattr(s, "summary", "") or "")) for s in stories)
    try:
        from seo.score_checker import _content_keywords
        ckw = _content_keywords(content, top=10)
    except Exception:
        ckw = []
    channels = [c for c in db.query(models.Channel).filter(models.Channel.user_id == user.id).all()
                if c.oauth_token is not None and bool(c.oauth_token.refresh_token_enc)]  # our publish channels only
    # Scope to the channels chosen for THIS job ("Choose channels" step) — SAME
    # rule as the preview, so Apply writes variants for exactly the channels the
    # operator sees/edits (not all ~40 connected channels — that was a cost bomb
    # AND meant the preview and the saved set could differ).
    try:
        _tids = json.loads(job.target_channel_ids) if getattr(job, "target_channel_ids", None) else []
    except Exception:
        _tids = []
    _target_ids = {int(t) for t in _tids} if isinstance(_tids, list) else set()
    if _target_ids:
        _scoped = [c for c in channels if c.id in _target_ids]
        if _scoped:
            channels = _scoped
    from seo.performance_profile import build_channel_profile
    from seo.per_channel import build_channel_previews
    try:
        from analytics.channel_catalog import ensure_synced as _ensure_synced
    except Exception:
        _ensure_synced = None
    fb: dict = {}
    for ch in channels:
        # Pull the channel's REAL YouTube catalogue (best-effort) so its native
        # videos feed the winning-keyword profile. Cached + re-synced weekly.
        try:
            gcid = (ch.oauth_token.google_channel_id or "") if ch.oauth_token else ""
            if _ensure_synced is not None and gcid:
                _ensure_synced(db, user.id, gcid)
        except Exception:
            pass
        try:
            p = build_channel_profile(db, ch.id)
            if p.get("ready"):
                fb[ch.id] = {"winning_keywords": p.get("winning_keywords", [])}
        except Exception:
            pass
    previews = build_channel_previews(
        base_seo=base_seo, channels=channels,
        feedback_by_channel=fb, content_keywords=ckw, mode="per_channel",
        generate=True, content_text=content, language=(jc.language or "te"),
    )
    applied = 0
    for p in previews:
        cid = p.get("channel_id")
        if cid is None:
            continue
        _v = {
            "title": p.get("title", ""),
            "description": p.get("description", ""),
            "tags": p.get("tags", []),
            "_per_channel": True,
        }
        if p.get("hashtags"):
            _v["hashtags"] = p.get("hashtags")
        if p.get("seo_score") is not None:
            _v["seo_score"] = p.get("seo_score")
        variants[str(cid)] = _v
        applied += 1
    clip.seo_variants = _json.dumps(variants)
    db.add(clip); db.commit()
    _seo_belt(job_id, user.id, "per_channel", "exited")
    # Return the generated previews too, so the UI can SHOW the distinct titles
    # it just SAVED (one click = write + apply + display) instead of forcing a
    # separate preview step that didn't persist.
    return {"ok": True, "mode": "per_channel", "applied": applied,
            "channels": previews, "generated": True}


# ─── Per-platform SEO preview (YouTube / Instagram / Facebook) ─────

class PlatformPreviewIn(BaseModel):
    clip_id: int
    platform: str = "youtube"            # youtube | instagram | facebook
    channel_id: Optional[int] = None     # connected account -> overlay socials
    force: bool = False                  # regenerate even if cached


@router.post("/jobs/{job_id}/seo/platform-preview")
def platform_seo_preview(
    job_id: int,
    payload: PlatformPreviewIn,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Preview (and cache) the EXACT per-platform copy a clip will publish with.

    youtube           -> the proven title + description + tags (unchanged path).
    instagram/facebook-> a native caption + hashtags from a dedicated Gemini
                         call (``seo.platform_seo``), CACHED onto the clip so
                         the publish dispatch reuses the identical text. When a
                         connected ``channel_id`` is given, the channel's own
                         socials / footer / mandatory hashtags are overlaid via
                         the same composer the upload worker uses — so what you
                         preview is what you publish.
    """
    job = _owned_job(job_id, user, db)
    clip = db.query(models.Clip).filter(models.Clip.id == payload.clip_id).first()
    if clip is None or clip.job_id != job.id:
        raise HTTPException(404, "clip not found for this job")

    platform = (payload.platform or "youtube").strip().lower()
    if platform not in ("youtube", "instagram", "facebook"):
        raise HTTPException(400, "platform must be youtube, instagram or facebook")
    _seo_belt(job_id, user.id, "platform", "entered")

    # Base generic SEO from the clip (publish source of truth).
    try:
        generic = json.loads(clip.seo) if clip.seo else {}
        if not isinstance(generic, dict):
            generic = {}
    except Exception:
        generic = {}

    publish_kind = "short" if (clip.frame_type or "") != "bulletin" else "video"

    variant: dict = {}
    if platform in ("instagram", "facebook"):
        from services.platform_variants import ensure_platform_variant
        variant = ensure_platform_variant(
            db, clip, platform, force=bool(payload.force),
        ) or {}
        if variant.get("caption"):
            generic.setdefault("platform_variants", {})[platform] = variant

    composed = None
    if payload.channel_id:
        channel = db.query(models.Channel).filter(
            models.Channel.id == payload.channel_id,
            models.Channel.user_id == user.id,
        ).first()
        if not channel:
            raise HTTPException(404, "channel not found")
        try:
            from seo.composer import compose as _seo_compose
            composed = _seo_compose(
                generic, channel, publish_kind=publish_kind, platform=platform,
            )
        except Exception as exc:
            raise HTTPException(500, f"compose failed: {exc}")

    _seo_belt(job_id, user.id, "platform", "exited")
    return {
        "platform":  platform,
        "clip_id":   clip.id,
        # Raw per-platform variant (IG/FB) — caption + hashtags + advisory score.
        "variant":   variant,
        # Channel-composed output (socials/footer overlaid) when channel given.
        "composed":  composed,
        # For YouTube with no channel, surface the plain generic for the UI.
        "generic":   {
            "title": generic.get("title", ""),
            "description": generic.get("description", ""),
            "hashtags": generic.get("hashtags", []),
            "keywords": generic.get("keywords", []),
        } if platform == "youtube" else None,
    }


# ─── SEO comparison: pasted YouTube video vs our generated SEO ─────

import re as _re

_YT_URL_PATTERNS = (
    # Standard watch URL: youtube.com/watch?v=ID
    _re.compile(r"(?:youtube\.com|youtube-nocookie\.com)/watch\?(?:.*&)?v=([A-Za-z0-9_-]{11})"),
    # Short link: youtu.be/ID
    _re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})"),
    # Embed: youtube.com/embed/ID
    _re.compile(r"youtube\.com/embed/([A-Za-z0-9_-]{11})"),
    # Shorts: youtube.com/shorts/ID
    _re.compile(r"youtube\.com/shorts/([A-Za-z0-9_-]{11})"),
    # Naked video ID — last-resort
    _re.compile(r"^([A-Za-z0-9_-]{11})$"),
)


def _parse_youtube_video_id(url: str) -> str:
    """Extract the 11-char video ID from any common YouTube URL shape.
    Returns "" when the string doesn't look like a YouTube reference at
    all — caller raises 400 in that case. Handles the four shapes the
    operator is likely to paste: watch?v=, youtu.be, /embed/, /shorts/.
    """
    s = (url or "").strip()
    if not s:
        return ""
    for pat in _YT_URL_PATTERNS:
        m = pat.search(s)
        if m:
            return m.group(1)
    return ""


def _fetch_yt_video_meta(video_id: str) -> dict:
    """Hit ``videos.list(part=snippet,statistics)`` for one video.
    Uses ``settings.yt_data_api_key`` because the comparison call
    targets PUBLIC videos and shouldn't burn OAuth credit. Returns
    a normalised dict. Raises HTTPException on auth / not-found
    so the route can surface a clean error.
    """
    try:
        from googleapiclient.discovery import build as _yt_build
        from config import settings as _settings
    except Exception as exc:
        raise HTTPException(500, f"YouTube API client unavailable: {exc}")
    api_key = (getattr(_settings, "yt_data_api_key", "") or "").strip()
    if not api_key:
        raise HTTPException(
            500,
            "YOUTUBE_DATA_API_KEY is not configured on the server — "
            "comparison needs a public API key to read the target video.",
        )
    try:
        yt = _yt_build("youtube", "v3", developerKey=api_key, cache_discovery=False)
        resp = yt.videos().list(
            part="snippet,statistics",
            id=video_id,
            maxResults=1,
        ).execute()
    except Exception as exc:
        raise HTTPException(502, f"YouTube videos.list failed: {exc}")
    items = resp.get("items") or []
    if not items:
        raise HTTPException(
            404,
            f"Video {video_id!r} not found, private, or deleted. "
            f"Comparison needs a public video.",
        )
    item = items[0]
    snippet = item.get("snippet") or {}
    stats = item.get("statistics") or {}
    return {
        "video_id":         video_id,
        "title":            (snippet.get("title") or "").strip(),
        "description":      (snippet.get("description") or "").strip(),
        "tags":             list(snippet.get("tags") or []),
        "channel_title":    (snippet.get("channelTitle") or "").strip(),
        "published_at":     snippet.get("publishedAt") or "",
        "view_count":       int(stats.get("viewCount") or 0),
        "like_count":       int(stats.get("likeCount") or 0),
        "comment_count":    int(stats.get("commentCount") or 0),
    }


def _compare_seo_via_gemini(*, ours: dict, theirs: dict, language: str) -> dict:
    """One Vertex Gemini call that scores both SEOs across 5 axes.

    Returns a strict JSON shape:
      {
        axes: [
          {name, ours_score, theirs_score, winner, reason},
          ...
        ],
        overall: {ours_total, theirs_total, winner, summary},
      }

    Axes are picked to mirror what a YouTube growth coach checks:
    title strength, description hook, keyword/tag relevance, hashtag
    discoverability, CTR potential. All on 0-100; integer-only so
    the UI table reads cleanly.
    """
    try:
        from seo.generator import _gemini_client
        from google.genai import types as genai_types
    except Exception as exc:
        raise HTTPException(500, f"Vertex Gemini SDK unavailable: {exc}")
    try:
        client = _gemini_client()
    except Exception as exc:
        raise HTTPException(
            500,
            f"Gemini client init failed: {exc}. "
            f"Check KAIZER_GCP_PROJECT / KAIZER_VERTEX_CREDENTIALS.",
        )

    sys_prompt = (
        "You are a YouTube growth strategist comparing two videos' SEO "
        "metadata head-to-head. Score each video on 5 axes (0-100 integers) "
        "and explain the winner per axis in ONE SHORT sentence (under 25 words). "
        "Be honest — if one side is clearly weaker, say so. No diplomatic ties "
        "unless they're genuinely identical.\n\n"
        "Axes:\n"
        "  1. title_strength       — hook power, clarity, length fit (<=70 chars), curiosity gap\n"
        "  2. description_hook     — first 100 chars grab attention + state value\n"
        "  3. keyword_relevance    — tags/keywords match the actual topic + cover search intent\n"
        "  4. hashtag_discovery    — hashtags drive feed discovery in the target language\n"
        "  5. ctr_potential        — overall headline + thumbnail-language synergy\n\n"
        "Output ONE JSON object, no prose, no markdown fences.\n"
        "KEEP REASONS UNDER 25 WORDS EACH. KEEP SUMMARY UNDER 3 SENTENCES.\n"
        "Schema:\n"
        "{\n"
        '  "axes": [\n'
        '    {"name": "title_strength", "ours_score": int, "theirs_score": int, '
        '"winner": "ours"|"theirs"|"tie", "reason": "1 sentence why"},\n'
        '    ... one entry per axis above ...\n'
        "  ],\n"
        '  "overall": {\n'
        '    "ours_total": int (sum of our axis scores),\n'
        '    "theirs_total": int (sum of their axis scores),\n'
        '    "winner": "ours"|"theirs"|"tie",\n'
        '    "summary": "2-3 sentences explaining the verdict + 1 concrete fix to close the gap"\n'
        "  }\n"
        "}"
    )

    user_payload = {
        "language": language or "te",
        "ours": {
            "title":        (ours.get("title") or "")[:300],
            "description":  (ours.get("description") or "")[:2000],
            "keywords":     list(ours.get("keywords") or [])[:40],
            "hashtags":     list(ours.get("hashtags") or [])[:20],
        },
        "theirs": {
            "title":        (theirs.get("title") or "")[:300],
            "description":  (theirs.get("description") or "")[:2000],
            "tags":         list(theirs.get("tags") or [])[:40],
            "channel":      theirs.get("channel_title") or "",
            "view_count":   theirs.get("view_count") or 0,
            "like_count":   theirs.get("like_count") or 0,
        },
    }
    # Structured-output schema. Vertex Gemini honours this when
    # response_mime_type=application/json — eliminates the
    # "model invented extra fields / forgot a comma" failure mode that
    # made the original 2048-token cap truncate mid-stream.
    compare_schema = {
        "type": "object",
        "properties": {
            "axes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name":         {"type": "string"},
                        "ours_score":   {"type": "integer"},
                        "theirs_score": {"type": "integer"},
                        "winner":       {"type": "string", "enum": ["ours", "theirs", "tie"]},
                        "reason":       {"type": "string"},
                    },
                    "required": ["name", "ours_score", "theirs_score", "winner", "reason"],
                },
            },
            "overall": {
                "type": "object",
                "properties": {
                    "ours_total":   {"type": "integer"},
                    "theirs_total": {"type": "integer"},
                    "winner":       {"type": "string", "enum": ["ours", "theirs", "tie"]},
                    "summary":      {"type": "string"},
                },
                "required": ["ours_total", "theirs_total", "winner", "summary"],
            },
        },
        "required": ["axes", "overall"],
    }

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=json.dumps(user_payload, ensure_ascii=False),
            config=genai_types.GenerateContentConfig(
                system_instruction=sys_prompt,
                response_mime_type="application/json",
                response_schema=compare_schema,
                temperature=0.2,
                # 8192 is comfortable headroom for 5 axes × short reasons
                # + a 3-sentence summary. The original 2048 cap truncated
                # responses mid-JSON (observed in prod 2026-06-08).
                max_output_tokens=8192,
            ),
        )
    except Exception as exc:
        raise HTTPException(502, f"Gemini compare call failed: {exc}")

    raw = (resp.text or "").strip()
    if not raw:
        # If we hit the token cap, surface that clearly so the operator
        # knows to retry rather than seeing "empty body".
        try:
            finish_reason = (resp.candidates[0].finish_reason or "").lower() if resp.candidates else ""
        except Exception:
            finish_reason = ""
        if "max_token" in finish_reason or "length" in finish_reason:
            raise HTTPException(502,
                "Gemini hit its output token cap. Retry — the schema enforces "
                "valid JSON so the next attempt should fit.")
        raise HTTPException(502, "Gemini returned empty body for the comparison")
    # Strip markdown fences just in case the model decorated.
    cleaned = _re.sub(r"^```(?:json)?\s*|\s*```$", "", raw,
                       flags=_re.IGNORECASE | _re.DOTALL).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = _re.search(r"\{.*\}", cleaned, flags=_re.DOTALL)
        if not m:
            raise HTTPException(502, f"Gemini returned non-JSON: {cleaned[:300]}")
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            # Truncated payload — outer brace match but inner JSON
            # incomplete. Tell the operator to retry rather than
            # silently dropping; the next call will fit under 8K.
            raise HTTPException(502,
                f"Gemini response was truncated mid-JSON. Retry. "
                f"(first 200 chars: {cleaned[:200]})")
    # Light validation — the UI table assumes these fields.
    if "axes" not in data or "overall" not in data:
        raise HTTPException(502, f"Gemini response missing axes/overall: {data}")
    return data


class SeoCompareIn(BaseModel):
    """Compare this canvas's SEO with the SEO of a paste-in YouTube video."""
    target: str = "bulletin"      # "bulletin" | "short"
    index: int = 0
    youtube_url: str = ""


@router.post("/jobs/{job_id}/seo/compare-with-youtube")
def compare_seo_with_youtube(
    job_id: int,
    payload: SeoCompareIn,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Score this canvas's SEO head-to-head against a paste-in YouTube
    video. Returns per-axis scores + overall winner + a 2-3 sentence
    coach summary explaining how to close the gap.

    Cost: 1 YouTube Data API unit (videos.list) + 1 Gemini 2.5 Flash
    call (~$0.0005). Cheap enough to run on every paste.
    """
    video_id = _parse_youtube_video_id(payload.youtube_url)
    if not video_id:
        raise HTTPException(
            400,
            "Couldn't extract a video ID from the URL. Paste a full "
            "youtube.com/watch?v=..., youtu.be/..., or /shorts/... link.",
        )

    # Resolve our canvas SEO.
    job = _owned_job(job_id, user, db)
    out_dir = _v4_dir_for(job)
    jc = _read_canvas(out_dir)
    if payload.target == "bulletin":
        canv = jc.bulletin
    elif payload.target == "short":
        if not (0 <= payload.index < len(jc.shorts)):
            raise HTTPException(400, f"short index {payload.index} out of range")
        canv = jc.shorts[payload.index]
    else:
        raise HTTPException(400, f"target must be 'bulletin' or 'short' (got {payload.target!r})")

    seo = canv.seo
    if not seo or not (getattr(seo, "title", "") or "").strip():
        raise HTTPException(
            409,
            "This canvas has no SEO yet — generate or regenerate it first, "
            "then come back to compare.",
        )

    ours = {
        "title":       (getattr(seo, "title", "") or ""),
        "description": (getattr(seo, "description", "") or ""),
        "keywords":    list(getattr(seo, "keywords", None) or []),
        "hashtags":    list(getattr(seo, "hashtags", None) or []),
    }

    _seo_belt(job_id, user.id, "compare", "entered")
    try:
        # 1 YouTube API unit.
        theirs = _fetch_yt_video_meta(video_id)
        # 1 Gemini call.
        comparison = _compare_seo_via_gemini(
            ours=ours, theirs=theirs, language=jc.language or "te",
        )
    except Exception:
        _seo_belt(job_id, user.id, "compare", "failed")
        raise
    _seo_belt(job_id, user.id, "compare", "exited")

    return {
        "ok": True,
        "video_id": video_id,
        "ours": ours,
        "theirs": theirs,
        "comparison": comparison,
    }


# ─── Regenerate SEO using a previous comparison as guidance ──────────


class SeoRegenFromCompareIn(BaseModel):
    """Rewrite this canvas's SEO using a prior head-to-head comparison
    as guidance. The frontend captures the response from
    ``compare-with-youtube`` and feeds the relevant pieces here so
    Gemini sees exactly which axes to improve on and why.

    ``theirs`` is the metadata of the benchmark video (used as a
    quality target). ``comparison`` is the per-axis scoring + coach
    summary. The endpoint refuses to run when the canvas has no SEO
    yet (regenerate-fresh is the right call in that case)."""
    target: str = "bulletin"      # "bulletin" | "short"
    index: int = 0
    theirs: dict
    comparison: dict


@router.post("/jobs/{job_id}/seo/regenerate-from-comparison")
def regenerate_seo_from_comparison(
    job_id: int,
    payload: SeoRegenFromCompareIn,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Rewrite this canvas's SEO using the comparison's verdict as the
    rewriting target. Gemini receives our current SEO + the benchmark
    video's SEO + the per-axis verdict, and is asked to produce a new
    SEO that closes the gap on every losing axis while preserving the
    operator's voice and the channel's native language.

    Cost: 1 Gemini 2.5 Flash call. Cheap enough to run every time the
    operator clicks "Regenerate to close the gap".
    """
    job = _owned_job(job_id, user, db)
    out_dir = _v4_dir_for(job)
    jc = _read_canvas(out_dir)

    if payload.target == "bulletin":
        canv = jc.bulletin
    elif payload.target == "short":
        if not (0 <= payload.index < len(jc.shorts)):
            raise HTTPException(400, f"short index {payload.index} out of range")
        canv = jc.shorts[payload.index]
    else:
        raise HTTPException(400, f"target must be 'bulletin' or 'short' (got {payload.target!r})")

    seo = canv.seo
    if not seo or not (getattr(seo, "title", "") or "").strip():
        raise HTTPException(
            409,
            "This canvas has no SEO yet — run a fresh Regenerate first, "
            "then come back to refine using a comparison.",
        )

    ours = {
        "title":       (getattr(seo, "title", "") or ""),
        "description": (getattr(seo, "description", "") or ""),
        "keywords":    list(getattr(seo, "keywords", None) or []),
        "hashtags":    list(getattr(seo, "hashtags", None) or []),
    }

    _seo_belt(job_id, user.id, "generate", "entered")

    # Build the Gemini prompt with a strict JSON schema. The schema
    # mirrors what seo_provider.generate_seo returns so the result
    # plugs straight back into CanvasSEO without translation.
    try:
        from seo.generator import _gemini_client
        from google.genai import types as genai_types
    except Exception as exc:
        _seo_belt(job_id, user.id, "generate", "failed")
        raise HTTPException(500, f"Vertex Gemini SDK unavailable: {exc}")
    try:
        client = _gemini_client()
    except Exception as exc:
        _seo_belt(job_id, user.id, "generate", "failed")
        raise HTTPException(500, f"Gemini client init failed: {exc}")

    sys_prompt = (
        "You are a YouTube growth strategist rewriting a video's SEO "
        "metadata to beat a benchmark competitor. You have the head-to-"
        "head comparison from a previous Gemini scoring pass and you "
        "MUST close the gap on every axis where the benchmark won, "
        "while keeping the strengths where we already won.\n\n"
        "Rules:\n"
        "  1. Title: under 70 chars, native script if the language is "
        "non-Latin, must carry the curiosity hook of the SEO description.\n"
        "  2. Description: 700-1800 chars. Hook in line 1, hashtags "
        "inlined into the body, ends with the brand suffix line.\n"
        "  3. Keywords: 15-25 relevant tags, mix of native + English.\n"
        "  4. Hashtags: 6-10 entries, each starting with #.\n"
        "  5. Preserve the operator's brand voice — same language, same "
        "register. Don't reinvent the editorial angle, just sharpen it.\n"
        "  6. Specifically address every losing-axis 'reason' from the "
        "comparison — if our description hook scored 30 vs their 60 "
        "because 'description is mismatched with the title', fix that.\n\n"
        "Output ONE JSON object, no prose, no markdown fences.\n"
        "Schema:\n"
        "{\n"
        '  "title": str,\n'
        '  "description": str,\n'
        '  "keywords": [str, ...],\n'
        '  "hashtags": [str, ...],\n'
        '  "hook": str (first-line attention-grabber),\n'
        '  "thumbnail_text": str (2-4 native-script words for the thumbnail)\n'
        "}"
    )

    user_payload = {
        "language": jc.language or "te",
        "ours_current_seo": ours,
        "benchmark_video": {
            "title":        (payload.theirs.get("title") or "")[:300],
            "description":  (payload.theirs.get("description") or "")[:2000],
            "tags":         list(payload.theirs.get("tags") or [])[:40],
            "channel":      payload.theirs.get("channel_title") or "",
            "view_count":   payload.theirs.get("view_count") or 0,
            "like_count":   payload.theirs.get("like_count") or 0,
        },
        "head_to_head_verdict": payload.comparison,
    }

    response_schema = {
        "type": "object",
        "properties": {
            "title":          {"type": "string"},
            "description":    {"type": "string"},
            "keywords":       {"type": "array", "items": {"type": "string"}},
            "hashtags":       {"type": "array", "items": {"type": "string"}},
            "hook":           {"type": "string"},
            "thumbnail_text": {"type": "string"},
        },
        "required": ["title", "description", "keywords", "hashtags"],
    }

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=json.dumps(user_payload, ensure_ascii=False),
            config=genai_types.GenerateContentConfig(
                system_instruction=sys_prompt,
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.4,
                max_output_tokens=8192,
            ),
        )
    except Exception as exc:
        _seo_belt(job_id, user.id, "generate", "failed")
        raise HTTPException(502, f"Gemini regen call failed: {exc}")

    raw = (resp.text or "").strip()
    if not raw:
        _seo_belt(job_id, user.id, "generate", "failed")
        raise HTTPException(502, "Gemini returned empty body for the rewrite")
    cleaned = _re.sub(r"^```(?:json)?\s*|\s*```$", "", raw,
                       flags=_re.IGNORECASE | _re.DOTALL).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = _re.search(r"\{.*\}", cleaned, flags=_re.DOTALL)
        if not m:
            _seo_belt(job_id, user.id, "generate", "failed")
            raise HTTPException(502, f"Gemini returned non-JSON: {cleaned[:300]}")
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            _seo_belt(job_id, user.id, "generate", "failed")
            raise HTTPException(502, f"Gemini response truncated mid-JSON. Retry. ({cleaned[:200]})")

    # Apply to the canvas. Use the same hashtag-inlining helper the
    # fresh-regenerate path uses so the description always carries
    # its hashtags. ``edited_by_user`` flips to True so the next bulk
    # regen doesn't overwrite this comparison-tuned version.
    new_keywords = [str(k).strip() for k in (data.get("keywords") or []) if str(k).strip()][:30]
    new_hashtags = [str(h).strip() for h in (data.get("hashtags") or []) if str(h).strip()][:12]
    try:
        from pipeline_v4.seo_provider import inline_hashtags_into_description
        new_description = inline_hashtags_into_description(
            description=(data.get("description") or "").strip(),
            hashtags=new_hashtags,
        )
    except Exception:
        new_description = (data.get("description") or "").strip()

    # Replace the SEO in place.
    canv.seo = CanvasSEO(
        title=(data.get("title") or "").strip(),
        description=new_description,
        keywords=new_keywords,
        hashtags=new_hashtags,
        hook=(data.get("hook") or "").strip(),
        thumbnail_text=(data.get("thumbnail_text") or "").strip(),
        edited_by_user=True,
    )
    _write_canvas(out_dir, jc)
    _seo_belt(job_id, user.id, "generate", "exited")
    return {"ok": True, "seo": json.loads(canv.seo.model_dump_json())}


# ─── Custom-template editor ──────────────────────────────────────────────────
# For a custom-template job, the bulletin canvas editor doesn't apply (the custom
# render bypasses canvas.json layout). These endpoints power a dedicated editor panel:
# edit per-slot TEXT, swap per-slot MEDIA, and SWITCH the template — persisted on the Job
# (template_overrides / template_media / fullform_layout|frame_layout) so BOTH the
# orchestrator render and the editor re-render honour them.

def _custom_layout_for(job: models.Job, target: str) -> str:
    v = ((job.fullform_layout if target == "bulletin" else job.frame_layout) or "").strip()
    return v if (v.lower().startswith("custom:") and v.split(":", 1)[1].isdigit()) else ""


def _jdict(v) -> dict:
    """A JSON column declared Column(JSON) but stored as TEXT (this codebase's pattern)
    comes back as a STRING — parse it. Returns {} for anything non-dict."""
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.strip():
        try:
            d = json.loads(v)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}
    return {}


@router.get("/jobs/{job_id}/custom-template")
def get_custom_template(job_id: int, target: str = "bulletin", index: int = 0,
                        db: Session = Depends(get_db),
                        user: models.User = Depends(auth.current_user)) -> dict:
    """Editor context for a custom-template job: per-slot text (current value = saved
    override else the offline filler's default), image slots + media, switchable templates,
    and the live preview URL. ``is_custom`` False -> the job isn't a custom-template job."""
    job = _owned_job(job_id, user, db)
    layout = _custom_layout_for(job, target)
    if not layout:
        return {"is_custom": False}
    from services import custom_templates as ct
    t = db.get(models.CustomTemplate, int(layout.split(":", 1)[1]))
    if not t or not t.dir_path:
        return {"is_custom": False}
    try:
        bundle = ct.Bundle(root_dir=t.dir_path, entry_rel=t.entry_rel or "index.html", files=[])
        with open(bundle.entry_path, encoding="utf-8", errors="replace") as fh:
            _norm, contract = ct.normalize_and_discover(fh.read())
    except Exception:
        return {"is_custom": False}
    kind = contract.kind
    # canvas stories drive the default (filler) text shown in each field
    try:
        jc = _read_canvas(_v4_dir_for(job))
        stories = (list(jc.bulletin.stories) if target == "bulletin"
                   else (list(jc.shorts[index].stories) if 0 <= index < len(jc.shorts) else []))
    except Exception:
        stories = []
    first = stories[0] if stories else None
    content = ct.ContentBundle(
        headline=((first.title_native or first.title_english) if first else ""),
        headline_alt=((first.title_english or "") if first else ""),
        body=((first.summary or "") if first else ""),
        stories=[{"headline": (s.title_native or s.title_english or "")} for s in stories],
    )
    try:
        defaults = dict(ct.build_slot_fill(contract, content, kind=kind).texts)
    except Exception:
        defaults = {}
    overrides = _jdict(job.template_overrides)
    text_slots = [{
        "key": s.key,
        "value": (overrides.get(s.key) if s.key in overrides else defaults.get(s.key, "")),
        "is_override": s.key in overrides,
        "placeholder": (s.placeholder or "")[:80],
    } for s in contract.text_slots]
    image_slots = [{"key": s.key} for s in contract.image_slots]
    rows = (db.query(models.CustomTemplate)
            .filter(models.CustomTemplate.status != "disabled")
            .filter((models.CustomTemplate.owner_id == user.id)
                    | (models.CustomTemplate.visibility == "public"))
            .order_by(models.CustomTemplate.created_at.desc()).all())
    options = [{"id": r.id, "name": r.name or "Untitled", "key": f"custom:{r.id}",
                "preview_url": f"/api/templates/{r.id}/preview"}
               for r in rows if ct.aspect_kind(r.canvas_w, r.canvas_h) == kind]
    return {
        "is_custom": True, "kind": kind, "target": target, "index": index, "layout": layout,
        "template": {"id": t.id, "name": t.name or "Untitled",
                     "preview_url": f"/api/templates/{t.id}/preview"},
        "text_slots": text_slots, "image_slots": image_slots,
        "template_media": _jdict(job.template_media),
        "main_media_slot": job.main_media_slot or "",
        "options": options,
        # True when the operator has a saved per-job visual design override for this output
        # (inline builder). The editor shows an "active" badge + a reset-to-template action.
        "has_html_override": bool(_jdict(job.custom_html_overrides).get(_override_key(target, index))),
    }


class CustomTemplateSaveIn(BaseModel):
    target: str = "bulletin"
    overrides: Optional[dict] = None        # {slot_key: text}
    layout: Optional[str] = None            # switch template: "custom:<id>" or "" to revert
    template_media: Optional[dict] = None   # {slot_key: asset_id}
    main_media_slot: Optional[str] = None


@router.post("/jobs/{job_id}/custom-template")
def save_custom_template(job_id: int, body: CustomTemplateSaveIn,
                         db: Session = Depends(get_db),
                         user: models.User = Depends(auth.current_user)) -> dict:
    """Persist custom-template editor edits onto the Job. Re-render afterwards (POST
    /jobs/{id}/render) to apply. Validates a template SWITCH against the target's kind."""
    job = _owned_job(job_id, user, db)
    from services import custom_templates as ct
    if body.overrides is not None:
        job.template_overrides = {str(k): ("" if v is None else str(v))[:600]
                                  for k, v in body.overrides.items()}
    if body.layout is not None:
        lv = (body.layout or "").strip()
        if lv == "":
            if body.target == "bulletin":
                job.fullform_layout = ""
            else:
                job.frame_layout = "torn_card"
        elif lv.lower().startswith("custom:") and lv.split(":", 1)[1].isdigit():
            t = db.get(models.CustomTemplate, int(lv.split(":", 1)[1]))
            if not t or t.status == "disabled" or not (t.owner_id == user.id or t.visibility == "public"):
                raise HTTPException(403, "Selected template is not available to you.")
            want = "full" if body.target == "bulletin" else "short"
            if ct.aspect_kind(t.canvas_w, t.canvas_h) != want:
                raise HTTPException(400, f"'{t.name}' is a {ct.aspect_kind(t.canvas_w, t.canvas_h)}-form "
                                         f"template — it can't be used for a {want}-form output.")
            if body.target == "bulletin":
                job.fullform_layout = lv
            else:
                job.frame_layout = lv
                # shorts re-render reads the layout from canvas short_config — keep in sync.
                try:
                    out_dir = _v4_dir_for(job)
                    jc = _read_canvas(out_dir)
                    idx = 0
                    if 0 <= idx < len(jc.shorts) and jc.shorts[idx].short_config:
                        jc.shorts[idx].short_config.layout = lv
                        _write_canvas(out_dir, jc)
                except Exception:
                    pass
    if body.template_media is not None:
        raw = body.template_media if isinstance(body.template_media, dict) else {}
        # A slot value is EITHER a scalar asset id (single image/video) OR a CAROUSEL struct
        # {carousel:[{id,duration_s,effect,effect_duration}], fit}. Gather every referenced id
        # (scalars + carousel frames) for ONE ownership query, then keep only owned assets.
        ref_ids = set()
        for v in raw.values():
            if isinstance(v, dict) and v.get("carousel"):
                for fr in (v.get("carousel") or []):
                    try:
                        ref_ids.add(int(fr.get("id")))
                    except Exception:
                        pass
            else:
                try:
                    ref_ids.add(int(v))
                except Exception:
                    pass
        owned = set()
        if ref_ids:
            owned = {r[0] for r in db.query(models.UserAsset.id).filter(
                models.UserAsset.id.in_(ref_ids),
                models.UserAsset.user_id == user.id).all()}
        clean_tm = {}
        for k, v in raw.items():
            if isinstance(v, dict) and v.get("carousel"):
                frames = []
                for fr in (v.get("carousel") or []):
                    try:
                        fid = int(fr.get("id"))
                    except Exception:
                        continue
                    if fid not in owned:
                        continue
                    frames.append({
                        "id": fid,
                        "duration_s": max(0.5, min(float(fr.get("duration_s") or 3.0), 30.0)),
                        "effect": str(fr.get("effect") or "fade")[:16],
                        "effect_duration": max(0.0, min(float(fr.get("effect_duration") or 0.4), 2.0)),
                    })
                frames = frames[:50]   # cost/DoS guard — cap carousel length (mirrors create_job)
                if frames:
                    clean_tm[str(k)] = {"carousel": frames, "fit": str(v.get("fit") or "cover")[:10]}
            else:
                try:
                    iv = int(v)
                except Exception:
                    continue
                if iv in owned:
                    clean_tm[str(k)] = iv
        job.template_media = clean_tm
    if body.main_media_slot is not None:
        job.main_media_slot = (body.main_media_slot or "").strip()[:64]
    db.commit()
    return {"ok": True}


# ── PER-JOB visual design override (inline builder) ────────────────────────────────────
# The operator opens THIS job's template in the inline visual builder, moves/resizes/
# recolours/retypes over the job's real content, and saves. The result is stored as a
# per-job HTML override (Job.custom_html_overrides) — the PARENT template is never touched.
# The renderer uses the override verbatim (literal mode) for this job only.
_OVERRIDE_MAX_BYTES = 5_000_000   # HTML + inline CSS + a few baked (downscaled) media data-URIs.


def _override_key(target: str, index: int) -> str:
    t = "bulletin" if (target or "").strip().lower() == "bulletin" else "short"
    return f"{t}:{max(0, int(index or 0))}"


@router.get("/jobs/{job_id}/custom-html")
def get_custom_html(job_id: int, target: str = "bulletin", index: int = 0,
                    db: Session = Depends(get_db),
                    user: models.User = Depends(auth.current_user)) -> dict:
    """HTML to load into the inline builder for THIS job+output. Returns the saved per-job
    override if one exists; otherwise the PARENT template HTML pre-filled with this job's
    real text (WYSIWYG) so the operator edits over what they'll actually ship. ``is_custom``
    False -> not a custom-template job."""
    job = _owned_job(job_id, user, db)
    layout = _custom_layout_for(job, target)
    if not layout:
        return {"is_custom": False}
    from services import custom_templates as ct
    from services.custom_templates.fill import fill_html
    t = db.get(models.CustomTemplate, int(layout.split(":", 1)[1]))
    if not t or not t.dir_path:
        return {"is_custom": False}
    try:
        bundle = ct.Bundle(root_dir=t.dir_path, entry_rel=t.entry_rel or "index.html", files=[])
        with open(bundle.entry_path, encoding="utf-8", errors="replace") as fh:
            _norm, contract = ct.normalize_and_discover(fh.read())
        parent_html = _norm or ""
        if not parent_html:
            with open(bundle.entry_path, encoding="utf-8", errors="replace") as fh:
                parent_html = fh.read()
    except Exception:
        return {"is_custom": False}

    overrides = _jdict(job.custom_html_overrides)
    key = _override_key(target, index)
    saved = overrides.get(key)
    if isinstance(saved, str) and saved.strip():
        html = saved
        is_override = True
    else:
        # Bake this job's real per-slot TEXT into the parent template (same computation as
        # get_custom_template) so the builder is WYSIWYG. Images stay as the template shows
        # them (the story photos drop in at render); the operator edits layout/colours/text.
        try:
            jc = _read_canvas(_v4_dir_for(job))
            stories = (list(jc.bulletin.stories) if target == "bulletin"
                       else (list(jc.shorts[index].stories) if 0 <= index < len(jc.shorts) else []))
        except Exception:
            stories = []
        first = stories[0] if stories else None
        content = ct.ContentBundle(
            headline=((first.title_native or first.title_english) if first else ""),
            headline_alt=((first.title_english or "") if first else ""),
            body=((first.summary or "") if first else ""),
            stories=[{"headline": (s.title_native or s.title_english or "")} for s in stories],
        )
        try:
            texts = dict(ct.build_slot_fill(contract, content, kind=contract.kind).texts)
        except Exception:
            texts = {}
        texts.update({str(k): ("" if v is None else str(v))
                      for k, v in _jdict(job.template_overrides).items()})
        try:
            html = fill_html(parent_html, texts=texts)
        except Exception:
            html = parent_html
        is_override = False

    return {
        "is_custom": True, "target": target, "index": index, "layout": layout,
        "is_override": is_override,
        "canvas": [contract.canvas_w, contract.canvas_h],
        "html": html,
        "template": {"id": t.id, "name": t.name or "Untitled"},
    }


class CustomHtmlSaveIn(BaseModel):
    target: str = "bulletin"
    index: int = 0
    html: str = ""                  # the edited HTML; "" / whitespace REVERTS to the template


@router.post("/jobs/{job_id}/custom-html")
def save_custom_html(job_id: int, body: CustomHtmlSaveIn,
                     db: Session = Depends(get_db),
                     user: models.User = Depends(auth.current_user)) -> dict:
    """Persist (or revert) the per-job visual design override. HTML is sanitized exactly like
    an uploaded template (scripts/external URLs stripped) and its canvas aspect must match the
    output kind. Re-render afterwards to apply. The parent template is never modified."""
    job = _owned_job(job_id, user, db)
    layout = _custom_layout_for(job, body.target)
    if not layout:
        raise HTTPException(400, "This output doesn't use a custom template.")
    overrides = dict(_jdict(job.custom_html_overrides))   # fresh dict -> JSON column sees the change
    key = _override_key(body.target, body.index)

    raw = body.html or ""
    if not raw.strip():
        overrides.pop(key, None)                          # revert to the parent template
        job.custom_html_overrides = overrides
        db.commit()
        return {"ok": True, "is_override": False}

    if len(raw.encode("utf-8", "ignore")) > _OVERRIDE_MAX_BYTES:
        raise HTTPException(413, "Edited design is too large.")

    from services import custom_templates as ct
    from services.custom_templates.bundle import neutralize_external_html
    try:
        clean, removed = neutralize_external_html(raw)
    except Exception:
        clean, removed = raw, 0
    # The edit must keep the output's aspect (a full-form output can't become 9:16, etc.).
    try:
        _norm, contract = ct.normalize_and_discover(clean)
        want = "full" if (body.target or "").strip().lower() == "bulletin" else "short"
        got = ct.aspect_kind(contract.canvas_w, contract.canvas_h)
        if got != want:
            raise HTTPException(400, f"This edited design is {got}-form ({contract.canvas_w}x"
                                     f"{contract.canvas_h}) but the output is {want}-form.")
    except HTTPException:
        raise
    except Exception:
        pass

    overrides[key] = clean
    job.custom_html_overrides = overrides
    db.commit()
    return {"ok": True, "is_override": True, "sanitized_removed": removed}
