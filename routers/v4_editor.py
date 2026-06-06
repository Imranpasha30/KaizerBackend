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
            rel = p.resolve().relative_to(backend_root.resolve() / "output")
            # forward-slash for URL safety; cache-bust by mtime so the
            # browser picks up freshly re-rendered files.
            mt = int(p.stat().st_mtime)
            return f"/media/{rel.as_posix()}?t={mt}"
        except ValueError:
            return None

    bulletin_url = _media_url(jc.bulletin.output_filename)
    trimmed_url = _media_url(Path(jc.trimmed_bulletin_path).name)
    shorts_urls = [_media_url(s.output_filename) for s in jc.shorts]

    # Materialised Clip IDs so the V4Editor can call the existing
    # /api/clips/{id}/publish endpoint directly. clip_index 0 is the
    # bulletin, 1..N are the shorts in order.
    bulletin_clip_id: Optional[int] = None
    shorts_clip_ids: list[Optional[int]] = [None] * len(jc.shorts)
    try:
        clips = db.query(models.Clip).filter(
            models.Clip.job_id == job_id
        ).order_by(models.Clip.clip_index).all()
        for c in clips:
            if c.clip_index == 0:
                bulletin_clip_id = c.id
            elif 1 <= c.clip_index <= len(shorts_clip_ids):
                shorts_clip_ids[c.clip_index - 1] = c.id
    except Exception:
        pass

    return {
        "canvas": json.loads(jc.model_dump_json()),
        "pool_listing": _list_pool(out_dir, job_id),
        "render_state": state.get("state", "idle"),
        "render_msg":   state.get("msg", ""),
        "render_target": state.get("target", ""),
        "bulletin_url": bulletin_url,
        "trimmed_url":  trimmed_url,
        "shorts_urls":  shorts_urls,
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
        if job_id is not None:
            url = f"/api/v4/jobs/{job_id}/pool/{f.name}"
        else:
            # Back-compat path when the caller can't supply a job id.
            url = f"/api/v4/jobs/{out_dir.name}/pool/{f.name}"
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
        try:
            _set_state(job_id, state="running", msg=f"compositing {tgt_label}")
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
            # produce (no "KAIZER NEWS" leak, same watermark stamp).
            from pipeline_v4 import orchestrator as _v4_orch
            user_d = _v4_orch._load_user_defaults(job_id) or {}
            channel_name = (user_d.get("brand_suffix") or "").lstrip(" |·-_•").strip()
            wm_text = user_d.get("watermark_text", "") or ""
            wm_op = float(user_d.get("watermark_opacity", 0.35) or 0.35)
            wm_pos = user_d.get("watermark_position", "top-right") or "top-right"

            if payload.target == "bulletin":
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
                    )
                    for s in jc.bulletin.stories
                ]
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
                title = (cfg.text if cfg and cfg.text else story_title) or "KAIZER NEWS"

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
                    layout=(cfg.layout if cfg else v1_bridge.DEFAULT_SHORTS_LAYOUT),
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
                ))
            _set_state(job_id, state="done", msg="render complete")
            print(f"[v4/render] job={job_id} target={tgt_label} done")
        except Exception as exc:
            _set_state(job_id, state="failed", msg=str(exc)[:300])
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


# ─── AI thumbnail (Nano Banana via Gemini) ──────────────────────────

class ThumbnailGenIn(BaseModel):
    """Trigger AI thumbnail generation for one canvas.

    First call: omit ``tweak`` -> Gemini writes a full prompt from SEO.
    Subsequent calls: pass a ``tweak`` string -> Gemini reuses the
    previous prompt (stored on disk) and applies the tweak. Cheaper
    than starting from scratch every time."""
    target: str = "bulletin"      # "bulletin" | "short"
    index: int = 0                 # for target="short"
    tweak: str = ""                # natural-language nudge for iteration


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

    saved, final_prompt = _tai.make_thumbnail_for_canvas(
        canvas=canv, out_path=str(out_path),
        language=jc.language or "te",
        previous_prompt=previous_prompt,
        tweak=tweak,
    )
    if not saved or not out_path.is_file():
        # Pull the verbatim Gemini error so the user can see what
        # actually went wrong (most commonly: 429 prepayment depleted
        # or 403 model not enabled on the project).
        raw = getattr(_tai.generate_thumbnail_image, "last_error", "")
        hint = ""
        if "RESOURCE_EXHAUSTED" in raw or "quota" in raw.lower() or " 429" in raw:
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
        detail = hint or (raw[:300] if raw else "Nano Banana didn't return an image.")
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
    rel = out_path.resolve().relative_to(backend_root.resolve() / "output")
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
        ))
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
        ))
        canv.seo = CanvasSEO(**seo_dict)
        canv.seo.edited_by_user = False
    else:
        raise HTTPException(400, f"target must be 'bulletin' or 'short' (got {payload.target!r})")

    _write_canvas(out_dir, jc)
    return {"ok": True, "seo": json.loads(canv.seo.model_dump_json())}
