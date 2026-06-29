"""Custom-template router — developer-uploaded HTML/CSS video templates.

Upload a ZIP (HTML/CSS + assets) or a single .html, validate + parse it, auto-generate a
preview thumbnail, and expose it in a library (the uploader's private templates + everyone's
public ones). Jobs select a template via the string ``custom:<id>`` in ``frame_layout``.
See services/custom_templates/TEMPLATE_CONTRACT.md for the developer rules.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import threading
from html import escape
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
from sqlalchemy import or_
from sqlalchemy.orm import Session

import auth
import models
from database import get_db
from services import custom_templates as ct
from services.custom_templates import bundle as _ctb   # neutralize_external_html (sanitiser)
from services.custom_templates import infer as _cti     # normalize_and_discover

router = APIRouter(prefix="/api/templates", tags=["templates"])

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_ROOT = BASE_DIR / "output" / "custom_templates"
TEMPLATES_ROOT.mkdir(parents=True, exist_ok=True)
CONTRACT_MD = Path(ct.__file__).resolve().parent / "TEMPLATE_CONTRACT.md"
MAX_UPLOAD = 35 * 1024 * 1024
MAX_TEMPLATES_PER_USER = 60


def _slugify(name: str) -> str:
    s = re.sub(r"[^\w\- ]", "", (name or "").strip().lower())
    s = re.sub(r"[\s_]+", "-", s).strip("-")
    return s or "template"


def _to_dict(t: models.CustomTemplate, *, uid: Optional[int]) -> dict:
    contract = t.contract_json or {}
    slots = contract.get("slots", []) if isinstance(contract, dict) else []
    return {
        "id": t.id,
        "key": f"custom:{t.id}",
        "name": t.name or "Untitled template",
        "slug": t.slug,
        "visibility": t.visibility,
        "status": t.status,
        "mine": uid is not None and t.owner_id == uid,
        "canvas": [t.canvas_w, t.canvas_h],
        # Output form, derived from the template's own canvas aspect (the code itself):
        # landscape -> "full" (16:9 full-form video), portrait/square -> "short" (9:16).
        # The wizard filters on this so a Short job never even sees full-form templates.
        "kind": ct.aspect_kind(t.canvas_w, t.canvas_h),
        "video_slots": sum(1 for s in slots if s.get("kind") == "video"),
        "image_slots": sum(1 for s in slots if s.get("kind") == "image"),
        "text_slots": sum(1 for s in slots if s.get("kind") == "text"),
        # Per-slot detail so the New Job wizard can render one media picker per slot
        # (video/background/intro/image) + let the user mark the AI-trim "main" video.
        "slots": [{"kind": s.get("kind"), "key": (s.get("name") or s.get("kind")),
                   "default": bool(s.get("default")),
                   # Hint: author marked this image slot data-kaizer-carousel="1". The media
                   # picker uses it as the DEFAULT (slideshow vs single) but lets the operator
                   # turn ANY image slot into a story-driven slideshow regardless.
                   "carousel": bool(s.get("carousel"))} for s in slots],
        # Per-slot fill projection (what each slot will be filled with at render time).
        "slot_audit": contract.get("slot_audit", []) if isinstance(contract, dict) else [],
        "preview_url": f"/api/templates/{t.id}/preview",  # endpoint self-heals if missing
        "description": t.description or "",
        "when_to_use": getattr(t, "when_to_use", "") or "",
        "how_to_use": getattr(t, "how_to_use", "") or "",
        "use_count": t.use_count or 0,
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "rating": (round((t.rating_sum or 0) / (t.rating_count or 1), 1) if (t.rating_count or 0) else None),
        "rating_count": t.rating_count or 0,
        # builder: the owner can open this in the visual editor + save back.
        "editable": (uid is not None and t.owner_id == uid),
        "derived_from": getattr(t, "derived_from", None),
    }


def _read_entry(t: models.CustomTemplate) -> str:
    """Read a template's current entry HTML (the editable source) from disk. Validates the
    resolved path stays INSIDE the template's bundle dir (defends against a tampered
    dir_path/entry_rel ever pointing at an arbitrary file)."""
    try:
        d = os.path.realpath(t.dir_path or "")
        if not d:
            return ""
        p = os.path.realpath(os.path.join(d, t.entry_rel or "index.html"))
        if p != d and not p.startswith(d + os.sep):
            return ""
        with open(p, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except Exception:
        return ""


def _attach_built_on(items: list, db: Session) -> list:
    """Resolve the 'Built on <name>' attribution for any forked templates (one query)."""
    ids = {d.get("derived_from") for d in items if d.get("derived_from")}
    names: dict = {}
    if ids:
        for rid, rname in (db.query(models.CustomTemplate.id, models.CustomTemplate.name)
                           .filter(models.CustomTemplate.id.in_(ids)).all()):
            names[rid] = rname
    for d in items:
        df = d.get("derived_from")
        d["built_on"] = (names.get(df) or None) if df else None
    return items


def _ignore_symlinks(dirpath, names):
    """copytree ignore-fn: skip symlinks so a fork can't follow a link out of the bundle."""
    return [n for n in names if os.path.islink(os.path.join(dirpath, n))]


def _dir_size(path: str) -> int:
    """Total bytes of regular (non-symlink) files under ``path``."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    return total


_BLANK_BODY = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="kaizer:canvas" content="{w}x{h}">
<title>{title}</title>
<style>
  :root{{--kaizer-brand:#ff5a3c;--kaizer-accent:#3ad1c8;--kaizer-text:#ffffff;
        --kaizer-bg:#0b0d12;--kaizer-font:'Segoe UI',Roboto,Arial,sans-serif;}}
  *{{margin:0;padding:0;box-sizing:border-box;}}
  html,body{{width:{w}px;height:{h}px;overflow:hidden;}}
  body{{position:relative;background:var(--kaizer-bg);color:var(--kaizer-text);
       font-family:var(--kaizer-font);}}
  #kx-stage{{position:absolute;inset:0;width:{w}px;height:{h}px;}}
  .kx-el{{position:absolute;box-sizing:border-box;}}
  #kx-video{{left:0;top:0;width:100%;height:100%;background:transparent;}}
</style></head>
<body><div id="kx-stage">
  <div class="kx-el" id="kx-video" data-kaizer="video"></div>
</div></body></html>"""


def _blank_html(w: int, h: int, name: str) -> str:
    return _BLANK_BODY.format(w=int(w), h=int(h), title=escape((name or "Template")[:80]))


def _apply_html(row: models.CustomTemplate, raw_html: str, db: Session,
                *, regen_preview: bool = True):
    """Sanitize edited HTML, write it as the template's bundle entry, re-discover its
    contract, refresh canvas/status, and regenerate the preview. Returns the contract.
    Shared by save-back, blank-create, and fork — reuses the upload pipeline's primitives,
    so a builder template renders through the exact same path as an uploaded one."""
    clean, _n = _ctb.neutralize_external_html(raw_html or "")
    clean = clean.replace(' id="kx-ed-sel"', "")   # drop the editor's selection marker if it leaked in
    norm, contract = _cti.normalize_and_discover(clean)
    final_html = norm or clean
    dest = TEMPLATES_ROOT / str(row.id)
    bdir = dest / "bundle"
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "index.html").write_text(final_html, encoding="utf-8")
    try:  # keep the sanitized source (pre-inference) as the sidecar for re-inference/fork
        (bdir / ".kaizer_src.html").write_text(clean, encoding="utf-8")
    except Exception:
        pass
    cj = contract.to_json()
    cj["slot_audit"] = ct.audit_template(contract)
    row.dir_path = str(bdir)
    row.entry_rel = "index.html"
    row.canvas_w, row.canvas_h = contract.canvas_w, contract.canvas_h
    row.contract_json = cj
    row.status = "ready" if (contract.video_slots or contract.background_slots) else "invalid"
    db.commit()
    db.refresh(row)
    # Preview render launches Chromium synchronously. Gate it behind the same bounded
    # semaphore the lazy GET /preview uses (non-blocking) so save/fork can't spawn unbounded
    # browsers or pile up request threads — if the slots are busy we skip and the thumbnail
    # self-heals on the next GET /preview.
    if regen_preview and _PREVIEW_SEM.acquire(blocking=False):
        try:
            bundle = ct.Bundle(root_dir=str(bdir), entry_rel="index.html", files=[])
            pv = str(dest / "preview.png")
            ct.render_preview(bundle, contract, pv)
            row.preview_path = pv
            db.commit()
        except Exception as exc:
            print(f"[templates] preview regen failed for {row.id}: {exc!r}", flush=True)
        finally:
            _PREVIEW_SEM.release()
    return contract


@router.get("/contract", response_class=PlainTextResponse)
def get_contract():
    try:
        return PlainTextResponse(CONTRACT_MD.read_text(encoding="utf-8"),
                                 media_type="text/markdown")
    except Exception:
        raise HTTPException(404, "contract not found")


@router.get("")
@router.get("/")
def list_templates(kind: Optional[str] = None,
                   db: Session = Depends(get_db),
                   user: models.User = Depends(auth.current_user)):
    """List the user's own + public templates. ``kind=short|full`` filters to that
    output form (derived from each template's canvas aspect) so the New Job wizard can
    show only the templates that fit the chosen output kind."""
    rows = (db.query(models.CustomTemplate)
            .filter(models.CustomTemplate.status != "disabled")
            .filter(or_(models.CustomTemplate.owner_id == user.id,
                        models.CustomTemplate.visibility == "public"))
            .order_by(models.CustomTemplate.created_at.desc())
            .all())
    out = [_to_dict(t, uid=user.id) for t in rows]
    _kind = (kind or "").strip().lower()
    if _kind in ("short", "full"):
        out = [d for d in out if d["kind"] == _kind]
    _attach_built_on(out, db)
    return out


@router.post("")
@router.post("/")
async def upload_template(
    file: Optional[UploadFile] = File(None),
    code: str = Form(""),
    name: str = Form(""),
    visibility: str = Form("private"),
    description: str = Form(""),
    when_to_use: str = Form(""),
    how_to_use: str = Form(""),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    visibility = "public" if visibility == "public" else "private"
    # Two ways in: upload a .zip/.html FILE, or PASTE the template HTML directly (for users
    # who don't have a file). Pasted code is treated as a single .html template — it goes
    # through the exact same sanitize/infer/AI-understand path as an uploaded .html.
    has_file = bool(file is not None and (file.filename or "").strip())
    code = code or ""
    if not has_file and not code.strip():
        raise HTTPException(400, "Provide a .zip/.html file or paste your template HTML.")
    if has_file:
        fname = file.filename or "template.zip"
        ext = os.path.splitext(fname)[1].lower()
        if ext not in (".zip", ".html", ".htm"):
            raise HTTPException(400, "Upload a .zip bundle or a single .html file.")
    else:
        fname, ext = "pasted-template.html", ".html"
        if len(code.encode("utf-8", "ignore")) > MAX_UPLOAD:
            raise HTTPException(400, "Pasted code too large (max 35 MB).")
    # SECURITY (DoS / disk): cap templates per user.
    _existing = (db.query(models.CustomTemplate)
                 .filter(models.CustomTemplate.owner_id == user.id).count())
    if _existing >= MAX_TEMPLATES_PER_USER:
        raise HTTPException(400, f"Template limit reached ({MAX_TEMPLATES_PER_USER}). Delete one first.")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        if has_file:
            size = 0
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(400, "Upload too large (max 35 MB).")
                tmp.write(chunk)
        else:
            tmp.write(code.encode("utf-8", "ignore"))
        tmp.close()

        row = models.CustomTemplate(
            owner_id=user.id,
            name=(name.strip() or os.path.splitext(fname)[0])[:120],
            visibility=visibility, status="processing",
            description=(description or "")[:2000],
            when_to_use=(when_to_use or "")[:2000],
            how_to_use=(how_to_use or "")[:4000],
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        row.slug = f"{_slugify(row.name)}-{row.id}"

        dest = TEMPLATES_ROOT / str(row.id)
        try:
            bundle, contract = ct.prepare_bundle(tmp.name, str(dest / "bundle"))
        except ct.BundleError as exc:
            db.delete(row)
            db.commit()
            shutil.rmtree(dest, ignore_errors=True)
            raise HTTPException(400, str(exc))
        except HTTPException:
            raise
        except Exception as exc:
            # Robustness: no uploaded template — however broken/hostile — may 500 the
            # server or strand a "processing" row. Fail clean with a 400.
            db.delete(row)
            db.commit()
            shutil.rmtree(dest, ignore_errors=True)
            print(f"[templates] prepare_bundle failed: {exc!r}", flush=True)
            raise HTTPException(400, f"Could not process this template: {exc}")

        # AI UNDERSTANDING (pure-intelligent): an LLM reasons over the rendered layout to
        # mark regions even when the offline rules can't (e.g. machine exports with no
        # naming hints). AI-primary; on any failure the offline contract above stands.
        # Runs in a thread (sync Playwright + LLM call). Gated by KAIZER_TEMPLATE_AI_UNDERSTAND.
        try:
            _ai_contract = await run_in_threadpool(ct.understand_and_mark, bundle)
            if _ai_contract is not None:
                contract = _ai_contract
                row.canvas_w, row.canvas_h = contract.canvas_w, contract.canvas_h
        except Exception as exc:
            print(f"[templates] AI understand skipped for {row.id}: {exc!r}", flush=True)

        # Generate the preview in a thread — Playwright's SYNC api cannot run inside
        # this async handler's event loop (it raises), which silently skipped previews.
        preview_path = ""
        try:
            await run_in_threadpool(ct.render_preview, bundle, contract, str(dest / "preview.png"))
            preview_path = str(dest / "preview.png")
        except Exception as exc:
            print(f"[templates] preview gen failed for {row.id}: {exc!r}", flush=True)
            preview_path = ""

        # Upload-time slot audit: what each slot WILL be filled with at render (the offline
        # filler's projection) + whether its authored inner text looks like a placeholder.
        # Lets the dev see "headline ← story headline" and warns about anything that would
        # render blank — so a leaking placeholder is caught at upload, not in the output.
        audit = ct.audit_template(contract)
        for a in audit:
            if a["kind"] == "text" and a["is_placeholder"]:
                if a.get("blank_in_final"):
                    contract.warnings.append(
                        f"Text slot '{a['key']}' (\"{a['placeholder'][:40]}\") renders BLANK in "
                        f"the master — it's stamped per-channel at publish.")
                else:
                    contract.warnings.append(
                        f"Placeholder in '{a['key']}' (\"{a['placeholder'][:40]}\") will be "
                        f"auto-filled from {a['fill_source']}.")
            elif a["kind"] == "image" and a["is_placeholder"]:
                contract.warnings.append(
                    f"Image slot '{a['key']}' placeholder will be replaced by its "
                    f"{a['fill_source']} (or hidden if none is available).")

        row.dir_path = bundle.root_dir
        row.entry_rel = bundle.entry_rel
        row.canvas_w, row.canvas_h = contract.canvas_w, contract.canvas_h
        _cj = contract.to_json()          # captures the audit-derived warnings above
        _cj["slot_audit"] = audit
        row.contract_json = _cj
        row.preview_path = preview_path
        # A template is renderable if it has somewhere to put the clip — a video OR a
        # background slot (inference may have found one from a <video> tag / id / class).
        row.status = "ready" if (contract.video_slots or contract.background_slots) else "invalid"
        db.commit()
        db.refresh(row)

        out = _to_dict(row, uid=user.id)
        out["warnings"] = contract.warnings
        return out
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


# SECURITY (DoS): cap concurrent on-demand preview renders so GET /preview can't be
# hammered into spawning unbounded Chromium. Uploads generate the preview up front, so
# this lazy path is a rare fallback — if the cap is busy we 404 rather than queue.
_PREVIEW_SEM = threading.BoundedSemaphore(2)


def _ensure_preview(t: models.CustomTemplate, db: Session) -> str:
    """Generate the preview PNG from the stored bundle if missing. Returns the path."""
    if not t.dir_path or not os.path.isdir(t.dir_path):
        raise RuntimeError("template bundle missing")
    if not _PREVIEW_SEM.acquire(blocking=False):
        raise RuntimeError("preview render busy")
    try:
        bundle = ct.Bundle(root_dir=t.dir_path, entry_rel=t.entry_rel or "index.html", files=[])
        with open(bundle.entry_path, encoding="utf-8", errors="replace") as fh:
            contract = ct.discover(fh.read())
        dest = TEMPLATES_ROOT / str(t.id)
        dest.mkdir(parents=True, exist_ok=True)
        out = str(dest / "preview.png")
        ct.render_preview(bundle, contract, out)
        t.preview_path = out
        db.commit()
        return out
    finally:
        _PREVIEW_SEM.release()


@router.get("/{tid}/preview")
def template_preview(tid: int, db: Session = Depends(get_db)):
    t = db.get(models.CustomTemplate, tid)
    if not t:
        raise HTTPException(404, "not found")
    if not t.preview_path or not os.path.isfile(t.preview_path):
        try:
            _ensure_preview(t, db)   # self-heal (sync def -> runs in threadpool, Playwright OK)
        except Exception as exc:
            raise HTTPException(404, f"no preview ({exc})")
    if not t.preview_path or not os.path.isfile(t.preview_path):
        raise HTTPException(404, "no preview")
    return FileResponse(t.preview_path, media_type="image/png")


class PatchBody(BaseModel):
    visibility: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    when_to_use: Optional[str] = None
    how_to_use: Optional[str] = None


@router.patch("/{tid}")
def patch_template(tid: int, body: PatchBody, db: Session = Depends(get_db),
                   user: models.User = Depends(auth.current_user)):
    t = db.get(models.CustomTemplate, tid)
    if not t:
        raise HTTPException(404, "not found")
    if t.owner_id != user.id:
        raise HTTPException(403, "not your template")
    if body.visibility in ("public", "private"):
        t.visibility = body.visibility
    if body.name is not None:
        t.name = body.name.strip()[:120]
    if body.description is not None:
        t.description = body.description[:2000]
    if body.when_to_use is not None:
        t.when_to_use = body.when_to_use[:2000]
    if body.how_to_use is not None:
        t.how_to_use = body.how_to_use[:4000]
    db.commit()
    db.refresh(t)
    return _to_dict(t, uid=user.id)


class RateBody(BaseModel):
    stars: int


@router.post("/{tid}/rate")
def rate_template(tid: int, body: RateBody, db: Session = Depends(get_db),
                  user: models.User = Depends(auth.current_user)):
    """Add a 1–5 star community rating. Visible to the rater only if the template is
    public or theirs (mirrors the access rule)."""
    t = db.get(models.CustomTemplate, tid)
    if not t or t.status == "disabled":
        raise HTTPException(404, "not found")
    if not (t.owner_id == user.id or t.visibility == "public"):
        raise HTTPException(403, "not available")
    stars = max(1, min(5, int(body.stars)))
    t.rating_sum = (t.rating_sum or 0) + stars
    t.rating_count = (t.rating_count or 0) + 1
    db.commit()
    db.refresh(t)
    return _to_dict(t, uid=user.id)


@router.delete("/{tid}")
def delete_template(tid: int, db: Session = Depends(get_db),
                    user: models.User = Depends(auth.current_user)):
    t = db.get(models.CustomTemplate, tid)
    if not t:
        raise HTTPException(404, "not found")
    if t.owner_id != user.id:
        raise HTTPException(403, "not your template")
    shutil.rmtree(TEMPLATES_ROOT / str(t.id), ignore_errors=True)
    db.delete(t)
    db.commit()
    return {"ok": True}


# ── Visual builder: blank-create, full-fetch (with HTML), save-back, fork ──────────────

class BlankBody(BaseModel):
    name: str = "Untitled template"
    width: int = 1080
    height: int = 1920
    visibility: str = "private"


@router.post("/blank")
def create_blank(body: BlankBody, db: Session = Depends(get_db),
                 user: models.User = Depends(auth.current_user)):
    """Create a NEW blank template at the chosen canvas size — the starting point for the
    visual builder. Returns the template + its editable HTML."""
    _existing = (db.query(models.CustomTemplate)
                 .filter(models.CustomTemplate.owner_id == user.id).count())
    if _existing >= MAX_TEMPLATES_PER_USER:
        raise HTTPException(400, f"Template limit reached ({MAX_TEMPLATES_PER_USER}). Delete one first.")
    w = max(64, min(int(body.width or 1080), 4096))
    h = max(64, min(int(body.height or 1920), 4096))
    vis = "public" if body.visibility == "public" else "private"
    row = models.CustomTemplate(owner_id=user.id,
                                name=(body.name.strip() or "Untitled template")[:120],
                                visibility=vis, status="processing")
    db.add(row); db.commit(); db.refresh(row)
    row.slug = f"{_slugify(row.name)}-{row.id}"
    try:
        contract = _apply_html(row, _blank_html(w, h, row.name), db)
    except Exception as exc:
        db.delete(row); db.commit()
        shutil.rmtree(TEMPLATES_ROOT / str(row.id), ignore_errors=True)
        raise HTTPException(400, f"Could not create template: {exc}")
    out = _to_dict(row, uid=user.id)
    out["html"] = _read_entry(row)
    out["built_on"] = None
    out["warnings"] = list(contract.warnings)
    return out


@router.get("/{tid}")
def get_template(tid: int, db: Session = Depends(get_db),
                 user: models.User = Depends(auth.current_user)):
    """Full template incl. its editable HTML source. Owner OR a public template (so it can
    be opened, previewed, and forked)."""
    t = db.get(models.CustomTemplate, tid)
    # 404 (not 403) for both missing AND no-access so private template IDs can't be enumerated.
    if not t or t.status == "disabled" or not (t.owner_id == user.id or t.visibility == "public"):
        raise HTTPException(404, "not found")
    out = _to_dict(t, uid=user.id)
    out["html"] = _read_entry(t)
    _attach_built_on([out], db)
    return out


class SaveHtmlBody(BaseModel):
    html: str
    name: Optional[str] = None
    visibility: Optional[str] = None


@router.put("/{tid}")
def save_template_html(tid: int, body: SaveHtmlBody, db: Session = Depends(get_db),
                       user: models.User = Depends(auth.current_user)):
    """Save edited HTML from the visual builder back to the template (owner only) — the
    'edit the code via UI, save as HTML' core. Re-sanitizes, re-discovers the contract,
    regenerates the preview, so the saved template always renders."""
    t = db.get(models.CustomTemplate, tid)
    if not t:
        raise HTTPException(404, "not found")
    if t.owner_id != user.id:
        raise HTTPException(403, "not your template")
    html = body.html or ""
    if not html.strip():
        raise HTTPException(400, "Empty template HTML.")
    if len(html.encode("utf-8", "ignore")) > MAX_UPLOAD:
        raise HTTPException(400, "Template too large (max 35 MB).")
    if body.name is not None:
        t.name = (body.name.strip()[:120] or t.name)
    if body.visibility in ("public", "private"):
        t.visibility = body.visibility
    contract = _apply_html(t, html, db)
    out = _to_dict(t, uid=user.id)
    out["html"] = _read_entry(t)
    out["warnings"] = list(contract.warnings)
    _attach_built_on([out], db)
    return out


class ForkBody(BaseModel):
    html: Optional[str] = None


@router.post("/{tid}/fork")
def fork_template(tid: int, body: ForkBody | None = None, db: Session = Depends(get_db),
                  user: models.User = Depends(auth.current_user)):
    """Fork a template (yours OR a public one) into a NEW private copy you own, recording
    'Built on <original>'. If the builder passes its current ``html``, the fork is created
    with those edits in ONE call — no separate save, so a save failure can't leave an orphan
    copy with stale HTML."""
    src = db.get(models.CustomTemplate, tid)
    # 404 (not 403) for missing AND no-access (no private-id enumeration).
    if not src or src.status == "disabled" or not (src.owner_id == user.id or src.visibility == "public"):
        raise HTTPException(404, "not found")
    _existing = (db.query(models.CustomTemplate)
                 .filter(models.CustomTemplate.owner_id == user.id).count())
    if _existing >= MAX_TEMPLATES_PER_USER:
        raise HTTPException(400, f"Template limit reached ({MAX_TEMPLATES_PER_USER}). Delete one first.")
    # Use the builder's current edits if supplied, else the source's saved HTML.
    edited = (body.html if (body and body.html and body.html.strip()) else "")
    if edited and len(edited.encode("utf-8", "ignore")) > MAX_UPLOAD:
        raise HTTPException(400, "Template too large (max 35 MB).")
    html = edited or _read_entry(src)
    if not html.strip():
        raise HTTPException(400, "Source template has no HTML to fork.")
    # DoS guard: don't duplicate an oversized asset bundle (60 forks x 120 MB would exhaust disk).
    if src.dir_path and os.path.isdir(src.dir_path) and _dir_size(src.dir_path) > MAX_UPLOAD:
        raise HTTPException(400, "Source template's assets are too large to fork (max 35 MB).")
    row = models.CustomTemplate(
        owner_id=user.id, name=(f"{src.name} (copy)")[:120], visibility="private",
        status="processing", derived_from=src.id, description=src.description or "",
        when_to_use=getattr(src, "when_to_use", "") or "",
        how_to_use=getattr(src, "how_to_use", "") or "")
    db.add(row); db.commit(); db.refresh(row)
    row.slug = f"{_slugify(row.name)}-{row.id}"
    dest = TEMPLATES_ROOT / str(row.id)
    try:
        # copy the source bundle assets (fonts/images), SKIPPING symlinks so the copy can't
        # follow a link out of the bundle, then apply the HTML.
        if src.dir_path and os.path.isdir(src.dir_path):
            shutil.copytree(src.dir_path, str(dest / "bundle"), dirs_exist_ok=True,
                            ignore=_ignore_symlinks)
        contract = _apply_html(row, html, db)
    except Exception as exc:
        db.delete(row); db.commit()
        shutil.rmtree(dest, ignore_errors=True)
        raise HTTPException(400, f"Could not fork template: {exc}")
    out = _to_dict(row, uid=user.id)
    out["html"] = _read_entry(row)
    out["built_on"] = src.name
    out["warnings"] = list(contract.warnings)
    return out
