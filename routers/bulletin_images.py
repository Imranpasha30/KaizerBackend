"""Bulletin per-image management — list, replace, recompose.

The bulletin renderer emits a side-carousel of 4–6 images per story.
That gives a 5-story bulletin 25-ish images on disk, and the user
sometimes wants to swap one specific image after the fact without
re-running Gemini analysis + OpenAI generation (which takes ~12 min
on Tier 1 alone).

Three endpoints here cover the full UX:

  GET  /api/jobs/{jid}/bulletin-images
       → returns the per-story image grid the editor displays.

  POST /api/jobs/{jid}/bulletin-images/replace  (multipart)
       → writes the uploaded image to a specific
         story_NN_assets/images/news_XX.jpg slot, overwriting whatever
         was there.

  POST /api/jobs/{jid}/bulletin-images/recompose
       → kicks off a compose-only re-render. The pipeline subprocess
         detects the cached Gemini analysis + cached per-story images
         and skips straight to FFmpeg compose, saving ~15 minutes.

All three are admin / owner gated through ``auth.current_user`` —
they read/write disk files in the job's own output directory so
cross-tenant access is impossible by construction.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter, Depends, File, Form, HTTPException, UploadFile,
)
from sqlalchemy.orm import Session

import auth
import models
from database import get_db


router = APIRouter(prefix="/api/jobs", tags=["bulletin-images"])


# ─── Path resolution ─────────────────────────────────────────────────

def _bulletin_dir_for(job: models.Job) -> Path:
    """Absolute path to ``<job.output_dir>/bulletin/``. ``Job.output_dir``
    is stored as a project-relative POSIX-ish path (e.g.
    ``output\\youtube_full\\20260513_144338``) — we resolve it against
    the backend root so the same code works on Windows + Linux."""
    if not job.output_dir:
        raise HTTPException(404, "Job has no output directory yet — pipeline may still be running")
    backend_root = Path(__file__).resolve().parent.parent
    out = (backend_root / job.output_dir).resolve()
    bdir = out / "bulletin"
    if not bdir.is_dir():
        raise HTTPException(404, f"Bulletin folder not found at {bdir}")
    return bdir


def _safe_join(base: Path, *parts: str) -> Path:
    """Path-join with anti-traversal check. Raises 400 if the joined
    path escapes ``base`` (defends against ``..`` or absolute-path
    injection from the request payload)."""
    p = (base.joinpath(*parts)).resolve()
    try:
        p.relative_to(base.resolve())
    except ValueError:
        raise HTTPException(400, "Invalid image slot — path traversal blocked")
    return p


# ─── Helpers ────────────────────────────────────────────────────────

def _media_url_for(abs_path: Path) -> str:
    """Frontend-reachable URL for an absolute disk path. Uses the
    existing ``/api/file/`` passthrough so we don't have to mount yet
    another StaticFiles handler."""
    return f"/api/file/?path={str(abs_path)}"


def _row_for_image(story_idx: int, slot_idx: int, abs_path: Path) -> dict:
    """Standard image record shape used by every code path below."""
    try:
        size = abs_path.stat().st_size
    except OSError:
        size = 0
    return {
        "story_index": int(story_idx),
        "slot_index":  int(slot_idx),
        "filename":    abs_path.name,
        "abs_path":    str(abs_path),
        "url":         _media_url_for(abs_path),
        "size_bytes":  int(size),
    }


def _slot_from_filename(name: str) -> int:
    """``news_03.jpg`` → 3; non-matching names → 0."""
    try:
        return int(Path(name).stem.split("_")[1])
    except (IndexError, ValueError):
        return 0


def _resolve_image_paths(bdir: Path) -> list[dict]:
    """Find every bulletin carousel image for this job, handling all
    three storage variants the pipeline has used over time:

      1. ``_generated_images.json`` manifest — newest, authoritative
         when present. May point cross-job (cache reuse) — we follow.
      2. ``story_NN_assets/images/news_XX.jpg`` — pre-per-job-pool layout
      3. ``_job_pool/{real,generated}/news_XX.jpg`` — per-job pool layout
         (added 2026-05) when ``KAIZER_BULLETIN_IMAGE_MODE=per_job``

    Always returns a list (empty if nothing on disk anywhere)."""
    rows: list[dict] = []

    # ── 1) Manifest ────────────────────────────────────────────────
    manifest_path = bdir / "_generated_images.json"
    if manifest_path.is_file():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            seen: set[tuple[int, str]] = set()
            for entry in data:
                p = Path(entry.get("path", "") or "")
                if not p.is_file():
                    continue
                story_idx = int(entry.get("story_index", 0) or 0)
                slot_idx  = _slot_from_filename(entry.get("filename", "") or p.name)
                # dedupe same (story, filename) pairs
                key = (story_idx, p.name.lower())
                if key in seen:
                    continue
                seen.add(key)
                rows.append(_row_for_image(story_idx, slot_idx, p))
            if rows:
                return rows
        except (OSError, json.JSONDecodeError):
            pass

    # ── 2) Per-story dirs ──────────────────────────────────────────
    for story_dir in sorted(bdir.glob("story_*_assets")):
        try:
            story_idx = int(story_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        img_dir = story_dir / "images"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if not img.is_file():
                continue
            if not img.name.lower().startswith("news_"):
                continue
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            rows.append(_row_for_image(story_idx, _slot_from_filename(img.name), img))

    if rows:
        return rows

    # ── 3) Per-job pool fallback ───────────────────────────────────
    # When the per-job-pool mode (default since 2026-05) is in use,
    # all images live under bulletin/_job_pool/{real,generated}/ and
    # the per-story dirs are empty. The pool is shared across stories
    # so we surface them as story_index=0 with sequential slots.
    pool_root = bdir / "_job_pool"
    slot = 1
    for sub in ("real", "generated"):
        d = pool_root / sub
        if not d.is_dir():
            continue
        for img in sorted(d.iterdir()):
            if not img.is_file():
                continue
            if not img.name.lower().startswith("news_"):
                continue
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            rows.append(_row_for_image(0, slot, img))
            slot += 1

    return rows


# Backwards-compat alias — earlier code references this name.
def _scan_story_images(bdir: Path) -> list[dict]:
    return _resolve_image_paths(bdir)
    return rows


# ─── In-flight recompose tracking ────────────────────────────────────
# Prevents two parallel recomposes for the same job (which would race
# on the bulletin.mp4 output). One per job at a time; status visible
# via the list endpoint.
_RECOMPOSING: dict[int, dict] = {}
_RECOMPOSE_LOCK = threading.Lock()


# ─── Endpoints ──────────────────────────────────────────────────────

@router.get("/{job_id}/bulletin-images")
def list_bulletin_images(
    job_id: int,
    db: Session = Depends(get_db),
    _: models.User = Depends(auth.current_user),
) -> dict:
    """List every per-story image in this job's bulletin folder,
    grouped by story for the editor grid."""
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    bdir = _bulletin_dir_for(job)

    # Prefer the manifest when it exists — it carries the AI vs real
    # provenance which the UI can show as a chip. Fallback to filesystem
    # scan if missing.
    manifest_path = bdir / "_generated_images.json"
    manifest: Optional[list[dict]] = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = None

    rows = _scan_story_images(bdir)

    # ── Pool view (PRIMARY, since the pipeline now uses one shared
    # pool of 5–6 images cycled across all stories).  Dedupe by
    # ``abs_path`` so duplicates from the cycling don't repeat the
    # same slot 4-5 times in the UI.  Track which story_indexes each
    # unique image appears in, so the UI can show it as metadata.
    pool: list[dict] = []
    seen: dict[str, dict] = {}    # abs_path → pool entry
    for r in rows:
        key = r["abs_path"]
        if key in seen:
            ex = seen[key]
            if r["story_index"] not in ex["story_indexes"]:
                ex["story_indexes"].append(int(r["story_index"]))
            continue
        entry = {
            **r,
            "story_indexes": [int(r["story_index"])],
            # Stable pool slot for the UI / replace-by-pool-slot calls.
            "pool_slot":    len(pool) + 1,
        }
        seen[key] = entry
        pool.append(entry)
    # Sort pool by slot_index for predictable ordering (1, 2, 3, ...).
    pool.sort(key=lambda x: (x["slot_index"], x["story_indexes"][0]))

    # Group by story for the legacy / detail view.
    grouped: dict[int, list[dict]] = {}
    for r in rows:
        grouped.setdefault(r["story_index"], []).append(r)
    stories = [
        {
            "story_index": sidx,
            "images":      sorted(grouped[sidx], key=lambda x: x["slot_index"]),
        }
        for sidx in sorted(grouped)
    ]

    recompose = _RECOMPOSING.get(job_id) or {}

    # Detect the rendering mode for the UI badge — pool mode kicks in
    # when there's clear cycling (fewer unique paths than total slots).
    is_pool_mode = len(pool) < len(rows)

    return {
        "job_id":          job.id,
        "bulletin_path":   str((bdir / "bulletin.mp4")),
        "pool":            pool,                # ← primary view
        "stories":         stories,             # legacy / detail
        "total_images":    len(rows),
        "unique_images":   len(pool),
        "is_pool_mode":    is_pool_mode,
        "manifest_seen":   manifest_path.is_file(),
        "recompose_state":   recompose.get("state", "idle"),
        "recompose_msg":     recompose.get("msg", ""),
        "recompose_scope":   recompose.get("scope", ""),
        "recompose_verify":  bool(recompose.get("verify", False)),
        "recompose_psnr_db": recompose.get("psnr_db"),
    }


@router.post("/{job_id}/bulletin-images/replace")
def replace_bulletin_image(
    job_id: int,
    story_index: int = Form(...),
    slot_index:  int = Form(...),
    image:       UploadFile = File(...),
    db: Session = Depends(get_db),
    _: models.User = Depends(auth.current_user),
) -> dict:
    """Overwrite a single ``news_NN.jpg`` slot with the uploaded image.

    The slot must already exist (we don't create new slots — story
    image counts are decided at generation time). Idempotent: if the
    image content is identical to what's there, the file is rewritten
    but no recompose is needed.
    """
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    bdir = _bulletin_dir_for(job)

    # Resolve the target slot. We allow .jpg only on the output side —
    # whatever the user uploaded gets re-encoded to JPG to keep the
    # carousel renderer's expectations stable.
    story_dir = _safe_join(bdir, f"story_{story_index:02d}_assets")
    img_dir   = _safe_join(story_dir, "images")
    img_dir.mkdir(parents=True, exist_ok=True)
    target    = _safe_join(img_dir, f"news_{slot_index:02d}.jpg")

    # If the original existed, back it up next to the new file as
    # ``news_NN.jpg.prev`` so the user can revert manually if needed.
    if target.exists():
        try:
            backup = target.with_suffix(target.suffix + ".prev")
            shutil.copy2(target, backup)
        except OSError:
            # Backup is best-effort; don't block the actual replace.
            pass

    # Stream-write the uploaded file. We re-encode through Pillow when
    # the extension differs OR the file is large (>2 MB) to keep the
    # carousel render consistent on JPG.
    raw_bytes = image.file.read()
    if not raw_bytes:
        raise HTTPException(400, "Empty upload")

    needs_reencode = (
        not (image.filename or "").lower().endswith((".jpg", ".jpeg"))
        or len(raw_bytes) > 2 * 1024 * 1024
    )
    if needs_reencode:
        try:
            import io
            from PIL import Image
            im = Image.open(io.BytesIO(raw_bytes))
            if im.mode != "RGB":
                im = im.convert("RGB")
            # Cap to 1920×1080 to match the broadcast layout's max sidebar size.
            im.thumbnail((1920, 1080), Image.LANCZOS)
            im.save(target, "JPEG", quality=88, optimize=True)
        except Exception as exc:
            raise HTTPException(400, f"Could not re-encode uploaded image: {exc}")
    else:
        target.write_bytes(raw_bytes)

    # ── Manifest update ────────────────────────────────────────
    # The manifest may have pointed at a cross-job path (cache reuse
    # by per-job pool). Repoint the matching (story_index, slot_index)
    # entry to OUR local file so recompose reads from this job's own
    # bulletin dir — no more cross-job mutation, change is fully
    # scoped to ``job_id``.
    manifest_path = bdir / "_generated_images.json"
    if manifest_path.is_file():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            target_filename = f"news_{slot_index:02d}.jpg"
            updated_any = False
            for entry in data:
                if (
                    int(entry.get("story_index", -1)) == int(story_index)
                    and Path(entry.get("filename", "") or "").stem
                        == Path(target_filename).stem
                ):
                    entry["path"]     = str(target)
                    entry["filename"] = target_filename
                    entry["user_replaced_at"] = datetime.now(timezone.utc).isoformat()
                    updated_any = True
                    break
            if not updated_any:
                # No matching slot in manifest — append a new entry so
                # recompose still picks up the user's edit.
                data.append({
                    "story_index": int(story_index),
                    "path":        str(target),
                    "filename":    target_filename,
                    "user_replaced_at": datetime.now(timezone.utc).isoformat(),
                })
            manifest_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[bulletin-images] manifest update failed: {exc}")

    return {
        "ok":           True,
        "story_index":  story_index,
        "slot_index":   slot_index,
        "filename":     target.name,
        "abs_path":     str(target),
        "url":          _media_url_for(target),
        "size_bytes":   target.stat().st_size,
        "needs_recompose": True,
        "message":      "Image replaced. Click 'Re-compose bulletin' to apply the change to bulletin.mp4.",
    }


_VALID_SCOPES = ("auto", "text-only", "images", "full")


def _apply_scope_invalidation(bdir: Path, scope: str) -> dict[str, int]:
    """Pre-delete intermediate outputs (and their ``.deps.json``
    sidecars) before the pipeline runs, based on which class of change
    the caller knows about. ``auto`` does nothing — the content-hash
    deps tracking in ``compose_deps`` figures out what's stale on its
    own. The other scopes are *optimisations*: they tell the pipeline
    "skip the hash check, just rebuild these" which saves a stat call
    per intermediate but more importantly forces a rebuild when the
    user's change wasn't a file replacement (e.g. they edited the
    ``StoryMeta.title`` directly via the editor)."""
    from pipeline_core import compose_deps

    removed: dict[str, int] = {}
    if scope == "text-only":
        # Text lives in the lower-third strap drawn by the per-story
        # compose. Sidebar + takeovers don't render text so they're
        # safe to keep.
        removed["composed_story"] = compose_deps.invalidate_glob(bdir, "composed_story_*.mp4")
    elif scope == "images":
        # Images flow through sidebar + takeover, and the composed
        # story embeds the sidebar visually — so all three classes
        # of intermediates need to rebuild.
        removed["sidebar"]        = compose_deps.invalidate_glob(bdir, "_sidebar_*.mp4")
        removed["sidebar_static"] = compose_deps.invalidate_glob(bdir, "_sidebar_*.png")
        removed["takeover"]       = compose_deps.invalidate_glob(bdir, "takeover_*.mp4")
        removed["composed_story"] = compose_deps.invalidate_glob(bdir, "composed_story_*.mp4")
    elif scope == "full":
        # Nuke everything bulletin/ produced. Pool + Gemini cache are
        # at a different level (output/_gemini_cache, bulletin/_job_pool)
        # so they survive — full == "rebuild from cached images".
        for pattern in (
            "composed_story_*.mp4", "_sidebar_*.mp4", "_sidebar_*.png",
            "takeover_*.mp4", "_ticker.png", "_bug.png", "bulletin.mp4",
        ):
            key = pattern.split("*")[0].rstrip("._")
            removed[key] = compose_deps.invalidate_glob(bdir, pattern)
    # scope == "auto" — fall through, no pre-invalidation.
    return removed


def _shadow_psnr_compare(fast_path: Path, full_path: Path) -> Optional[float]:
    """Run FFmpeg's psnr filter to compare two encodes. Returns the
    average PSNR in dB, or None on failure. >50 dB ≈ visually
    identical; <40 dB ≈ user-visible difference. Used only by the
    verification path under ``KAIZER_FAST_RECOMPOSE_VERIFY=true``."""
    import re
    import subprocess
    if not (fast_path.is_file() and full_path.is_file()):
        return None
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-i", str(fast_path), "-i", str(full_path),
                "-lavfi", "psnr", "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[bulletin-recompose] psnr ffmpeg failed: {exc}")
        return None
    # FFmpeg writes psnr stats to stderr like:
    #   [Parsed_psnr_0 @ ...] PSNR y:43.123456 u:... v:... average:42.987...
    m = re.search(r"average:([0-9.]+)", proc.stderr or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


@router.post("/{job_id}/bulletin-images/recompose")
def recompose_bulletin(
    job_id: int,
    scope: str = Form("auto"),
    verify: bool = Form(False),
    db: Session = Depends(get_db),
    _: models.User = Depends(auth.current_user),
) -> dict:
    """Kick off a compose-only re-render in a background thread.

    Reuses the existing pipeline subprocess (via ``runner.run_pipeline``)
    — the per-story image cache + Gemini cache short-circuit make this
    a compose-only path. With the ``compose_deps`` content-hash check
    in place, stages whose inputs didn't change are skipped entirely,
    so a single-image swap typically rebuilds only one sidebar + one
    composed story (~30s vs ~7 min for a full bulletin).

    Parameters
    ----------
    scope : one of ``auto`` (default — let the deps tracker decide),
            ``text-only`` (force composed_story rebuild),
            ``images`` (force sidebar + takeover + composed_story),
            ``full`` (rebuild every intermediate).
    verify : when true OR when ``KAIZER_FAST_RECOMPOSE_VERIFY=true`` is
            set in the env, also run a parallel full rebuild and PSNR-
            compare the two outputs to validate the fast path. The
            slow path's bulletin.mp4 wins (canonical) regardless of
            the result — verification just logs whether the fast path
            was equivalent.

    Returns immediately with state="queued"; clients poll the list
    endpoint to watch ``recompose_state`` flip queued → running → done.
    """
    if scope not in _VALID_SCOPES:
        raise HTTPException(400, f"Invalid scope {scope!r}. Allowed: {_VALID_SCOPES}")

    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    bdir = _bulletin_dir_for(job)   # validates the dir exists

    env_verify = (os.environ.get("KAIZER_FAST_RECOMPOSE_VERIFY") or "").strip().lower() in ("1", "true", "yes", "on")
    do_verify = bool(verify or env_verify)

    # Concurrency guard — only one recompose per job at a time.
    with _RECOMPOSE_LOCK:
        cur = _RECOMPOSING.get(job_id) or {}
        if cur.get("state") in ("queued", "running"):
            raise HTTPException(409, f"Bulletin recompose already {cur['state']} for this job")
        _RECOMPOSING[job_id] = {
            "state": "queued",
            "msg":   f"spawning worker (scope={scope}, verify={do_verify})",
            "scope": scope,
            "verify": do_verify,
        }

    def _run():
        try:
            with _RECOMPOSE_LOCK:
                _RECOMPOSING[job_id] = {
                    "state": "running",
                    "msg":   f"fast pass (scope={scope})",
                    "scope": scope,
                    "verify": do_verify,
                }
            # Step 1: pre-invalidate per scope so the pipeline rebuilds
            # what the caller knows changed. ``auto`` = no-op; compose_deps
            # decides per stage.
            removed = _apply_scope_invalidation(bdir, scope)
            if removed:
                summary = ", ".join(f"{k}:{v}" for k, v in removed.items() if v)
                print(f"[bulletin-recompose] job={job_id} pre-invalidate ({scope}): {summary}")

            try:
                import runner
                runner.run_pipeline(job.id)
            except Exception as exc:
                with _RECOMPOSE_LOCK:
                    _RECOMPOSING[job_id] = {"state": "failed", "msg": str(exc)[:300]}
                print(f"[bulletin-recompose] job={job_id} failed (fast pass): {exc}")
                return

            psnr_db: Optional[float] = None
            if do_verify:
                # Step 2 (verify only): keep fast output aside, force a
                # full rebuild, and PSNR-compare. The full rebuild is
                # canonical — that's what the user keeps.
                fast_out = bdir / "bulletin.mp4"
                fast_keep = bdir / "bulletin.fast.mp4"
                if fast_out.is_file():
                    try:
                        if fast_keep.exists():
                            fast_keep.unlink()
                        fast_out.rename(fast_keep)
                    except OSError as exc:
                        print(f"[bulletin-recompose] verify: could not stash fast output: {exc}")
                with _RECOMPOSE_LOCK:
                    _RECOMPOSING[job_id] = {
                        "state": "running",
                        "msg":   "verify: running full rebuild for PSNR baseline",
                        "scope": scope,
                        "verify": do_verify,
                    }
                _apply_scope_invalidation(bdir, "full")
                full_out = bdir / "bulletin.mp4"
                try:
                    runner.run_pipeline(job.id)
                except Exception as exc:
                    # Verify pass crashed mid-render. The fast output is
                    # still safe at fast_keep — restore it as the canonical
                    # bulletin.mp4 so the user isn't left with a missing /
                    # half-written file.
                    print(f"[bulletin-recompose] job={job_id} verify full pass failed: {exc}")
                    try:
                        if full_out.exists():
                            full_out.unlink()
                        if fast_keep.is_file():
                            fast_keep.rename(full_out)
                            print(f"[bulletin-recompose] job={job_id} verify: restored fast output as bulletin.mp4")
                    except OSError as restore_exc:
                        print(f"[bulletin-recompose] verify: restore failed: {restore_exc}")
                else:
                    psnr_db = _shadow_psnr_compare(fast_keep, full_out)
                    if psnr_db is None:
                        verdict = "psnr unavailable"
                    elif psnr_db >= 50.0:
                        verdict = f"fast-path equivalent (psnr={psnr_db:.2f} dB)"
                    else:
                        verdict = f"WARN fast-path drift (psnr={psnr_db:.2f} dB < 50)"
                    print(f"[bulletin-recompose] job={job_id} verify: {verdict}")
                    # Clean up the fast snapshot; full is canonical now.
                    try:
                        if fast_keep.is_file():
                            fast_keep.unlink()
                    except OSError:
                        pass

            done_msg = "bulletin.mp4 refreshed"
            if do_verify and psnr_db is not None:
                done_msg += f" — verify psnr={psnr_db:.2f} dB"
            with _RECOMPOSE_LOCK:
                _RECOMPOSING[job_id] = {
                    "state": "done",
                    "msg":   done_msg,
                    "scope": scope,
                    "verify": do_verify,
                    "psnr_db": psnr_db,
                }
        finally:
            print(f"[bulletin-recompose] job={job_id} state={_RECOMPOSING.get(job_id, {}).get('state')}")

    threading.Thread(target=_run, name=f"bulletin-recompose-{job_id}", daemon=True).start()

    return {
        "ok":      True,
        "job_id":  job_id,
        "state":   "queued",
        "scope":   scope,
        "verify":  do_verify,
        "message": f"Re-compose started (scope={scope}) — poll /bulletin-images for state changes.",
    }
