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
from database import SessionLocal, get_db


router = APIRouter(prefix="/api/jobs", tags=["bulletin-images"])


# ─── Path resolution ─────────────────────────────────────────────────

def _bulletin_dir_for(job: models.Job) -> Path:
    """Absolute path to ``<job.output_dir>/bulletin/``. ``Job.output_dir``
    is stored as a project-relative POSIX-ish path (e.g.
    ``output\\youtube_full\\20260513_144338``) — we resolve it against
    the backend root so the same code works on Windows + Linux.

    Fallback: for compound jobs (``youtube_full_plus_shorts``) the
    pipeline writes BOTH a ``youtube_short/<ts>/`` directory AND a
    ``youtube_full/<ts>/`` directory, and the import loop captures
    whichever ``[kaizer:meta]`` marker the subprocess emitted last
    as the job's ``output_dir``. If the short-side path won the
    race the stored ``output_dir`` won't contain a ``bulletin/``
    subfolder — but the bulletin still exists, just under a sibling
    ``youtube_full/`` directory. We find it by mtime proximity to
    the job's ``created_at``.
    """
    if not job.output_dir:
        raise HTTPException(404, "Job has no output directory yet — pipeline may still be running")
    backend_root = Path(__file__).resolve().parent.parent
    out = (backend_root / job.output_dir).resolve()
    bdir = out / "bulletin"
    if bdir.is_dir():
        return bdir

    # Fallback search — look for the closest-in-time youtube_full
    # sibling that has a bulletin/ subfolder.
    full_root = backend_root / "output" / "youtube_full"
    if full_root.is_dir() and job.created_at:
        try:
            target_ts = job.created_at.timestamp()
        except Exception:
            target_ts = None
        best: Optional[Path] = None
        best_dt: float = float("inf")
        for sub in full_root.iterdir():
            if not sub.is_dir():
                continue
            cand = sub / "bulletin"
            if not cand.is_dir():
                continue
            try:
                dir_ts = sub.stat().st_mtime
            except OSError:
                continue
            if target_ts is None:
                # No created_at → return the newest sibling.
                if dir_ts > -best_dt:
                    best_dt = -dir_ts
                    best = cand
            else:
                # Must have been created AFTER the job started — the
                # bulletin can't predate the job's submission. Allow
                # a small clock-skew margin (5 minutes early).
                if dir_ts < target_ts - 300:
                    continue
                gap = abs(dir_ts - target_ts)
                if gap < best_dt:
                    best_dt = gap
                    best = cand
        if best is not None:
            return best

    raise HTTPException(404, f"Bulletin folder not found at {bdir} (no sibling youtube_full match either)")


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
            # BUG-FIX 2026-05-26: dedup key was (story_idx, filename) which
            # collapsed `real/images/news_01.jpg` and `generated/news_01.jpg`
            # (different FILES at different paths, but same basename) into
            # one row -- losing the generated variant entirely. The editor
            # then showed only 3 pool images for a job that actually has
            # 5-7. Use the resolved absolute path so distinct files in
            # real/ vs generated/ subdirs are both preserved.
            seen: set[tuple[int, str]] = set()
            for entry in data:
                p = Path(entry.get("path", "") or "")
                if not p.is_file():
                    continue
                story_idx = int(entry.get("story_index", 0) or 0)
                slot_idx  = _slot_from_filename(entry.get("filename", "") or p.name)
                # dedupe same (story, ABSOLUTE PATH) pairs -- pure
                # filename was too aggressive (see bug-fix note above)
                key = (story_idx, str(p.resolve()).lower())
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
    #
    # Pool-mode short-circuit: when the manifest exists AND the
    # (story_index, slot) entry points at a shared ``_job_pool/...``
    # file, OVERWRITE THAT POOL FILE directly. Every other story that
    # references the same pool entry then sees the new image without
    # us having to fan-out the replace across N story_dir copies.
    # Without this branch, the previous behavior added a fresh file
    # at ``story_NN_assets/images/news_XX.jpg`` and only ONE of the
    # N stories visually updated — the pool count drifted upward
    # every replace.
    target: Optional[Path] = None
    manifest_path = bdir / "_generated_images.json"
    if manifest_path.is_file():
        try:
            _data_pre = json.loads(manifest_path.read_text(encoding="utf-8"))
            target_filename = f"news_{slot_index:02d}.jpg"
            for entry in _data_pre:
                if (
                    int(entry.get("story_index", -1)) == int(story_index)
                    and Path(entry.get("filename", "") or "").stem
                        == Path(target_filename).stem
                ):
                    src_path = Path(entry.get("path", "") or "")
                    # Only overwrite-in-place when the entry points at
                    # a sane existing pool file inside THIS bulletin
                    # dir — never write outside the dir for safety.
                    try:
                        if (src_path.is_file()
                                and bdir.resolve() in src_path.resolve().parents):
                            target = src_path
                    except OSError:
                        pass
                    break
        except (OSError, json.JSONDecodeError):
            pass

    if target is None:
        # Fallback: legacy per-story-dir slot. Used when no manifest
        # exists (older job layouts) or the manifest entry points
        # somewhere we don't trust.
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

    # Resolve a viable source video path. Two-tier lookup:
    #   1) The original upload at output/raw_uploads/<ts>/<video_name>
    #      (kept across pipeline runs in dev; cleaned up in prod).
    #   2) Fallback: any cached raw_clip_*.mp4 in the same output dir.
    #      The pipeline subprocess only needs a real MP4 so its initial
    #      ffprobe call succeeds. The cut-clips stage caches per-clip
    #      outputs and skips re-cutting when raw_clip_NN.mp4 already
    #      exists, so providing a clip as "the video" is functionally
    #      equivalent for a recompose-only run.
    out_root = bdir.parent       # bdir is <output_dir>/bulletin

    def _resolve_source_video() -> str:
        try:
            backend_root = Path(__file__).resolve().parent.parent
            uploads_root = backend_root / "output" / "raw_uploads"
            vname = (job.video_name or "").strip()
            if vname and uploads_root.is_dir():
                hits = sorted(
                    uploads_root.glob(f"*/{vname}"),
                    key=lambda p: p.stat().st_mtime if p.exists() else 0,
                    reverse=True,
                )
                for h in hits:
                    if h.is_file():
                        return str(h)
        except Exception:
            pass
        # Fallback — any cached clip is enough for ffprobe to succeed.
        for clip in sorted(out_root.glob("raw_clip_*.mp4")):
            if clip.is_file() and clip.stat().st_size > 100_000:
                return str(clip)
        return ""

    video_path = _resolve_source_video()
    if not video_path:
        with _RECOMPOSE_LOCK:
            _RECOMPOSING[job_id] = {
                "state": "failed",
                "msg":   ("recompose requires either the original upload or a cached "
                          "raw_clip_*.mp4 — neither was found in the job's output dir"),
            }
        raise HTTPException(
            409,
            "Cannot recompose: source video and cached clips are both missing. "
            "Re-submit the job from scratch.",
        )

    # Cached Gemini analysis at the job's output root. When present,
    # the pipeline subprocess will short-circuit the analysis call
    # entirely (no Gemini quota, no re-upload). For a recompose this
    # is the difference between "writes a new bulletin" and "fails on
    # the missing source video".
    analysis_path = out_root / "gemini_analysis.json"
    reuse_analysis = str(analysis_path) if analysis_path.is_file() else ""

    platform = job.platform or "youtube_full"
    frame = job.frame_layout or "torn_card"
    lang = job.language or "te"
    s2p = getattr(job, "stage_2_provider", None) or "gemini"
    tstyle = getattr(job, "transition_style", None) or "smart_cut"

    # Captured at recompose start so we can clean up the OLD paired
    # directories (youtube_full + youtube_short) AFTER the new render
    # succeeds. Filled inside _run_pipeline_sync.
    new_meta_paths: list[str] = []

    # Snapshot the current image pool BEFORE spawning the subprocess.
    # The pipeline accepts a pipe-separated list of pre-selected images
    # via ``KAIZER_BULLETIN_IMAGES`` — we hand it the user's existing
    # pool (including any swaps they made via /replace) so the new
    # render uses THOSE instead of asking OpenAI for fresh generations
    # from scratch. Without this, every recompose would shrink the
    # pool back to whatever Gemini's image_plan generates by default
    # (typically 3 images), losing all the user's curation work.
    try:
        existing_rows = _resolve_image_paths(bdir)
        # Deduplicate by absolute path — many rows can point at the
        # same pool file (story_indexes list).
        seen_pool: set[str] = set()
        existing_pool: list[str] = []
        for r in existing_rows:
            p = str(Path(r.get("abs_path", "")).resolve())
            if p and p not in seen_pool and Path(p).is_file():
                seen_pool.add(p)
                existing_pool.append(p)
        if existing_pool:
            print(f"[bulletin-recompose] job={job_id} preserving {len(existing_pool)} existing pool images for the new render")
    except Exception as exc:
        print(f"[bulletin-recompose] job={job_id} could not snapshot existing pool: {exc}")
        existing_pool = []

    def _run_pipeline_sync() -> int:
        """Spawn the V1 pipeline subprocess SYNCHRONOUSLY (block until
        it exits). ``runner.run_pipeline`` is fire-and-forget — it
        spawns a daemon thread and returns immediately, which leaves
        the recompose worker thinking the job finished before it
        actually started. For a recompose we need the opposite: the
        ``state=done`` UI signal must mean the bulletin file is
        physically on disk.

        Returns the subprocess return code. Output is streamed into
        backend stdout with a ``[bulletin-recompose:pipe]`` prefix so
        the operator can tail the same backend.log they'd already be
        watching.

        Also captures ``[kaizer:meta] <path>`` markers — the pipeline
        emits one per output (youtube_full + youtube_short for compound
        jobs). The recompose worker uses these to swap ``job.output_dir``
        to the new render and delete the old paired directories.
        """
        import sys as _sys
        import subprocess as _sub
        backend_root = Path(__file__).resolve().parent.parent
        pipeline_script = backend_root / "pipeline_core" / "pipeline.py"
        if not pipeline_script.is_file():
            raise RuntimeError(f"pipeline.py not found at {pipeline_script}")
        # Pipeline.py accepts EITHER --platform <single_platform_name>
        # OR --compound (runs the both-pass compound flow). The compound
        # job's platform field is ``youtube_full_plus_shorts`` which is
        # NOT a valid value for --platform; argparse rejects it with
        # rc=2 — see runner.py's pass-list construction for the same
        # split. For a recompose we want the same shape as the
        # original render.
        if platform == "youtube_full_plus_shorts":
            cmd = [
                _sys.executable, "-u", str(pipeline_script),
                video_path,
                "--compound",
                "--frame", frame,
                "--language", lang,
            ]
        else:
            cmd = [
                _sys.executable, "-u", str(pipeline_script),
                video_path,
                "--platform", platform,
                "--frame", frame,
                "--language", lang,
            ]
        env = {
            **os.environ,
            "KAIZER_JOB_ID":  str(job.id),
            "KAIZER_USER_ID": str(job.user_id or 0),
            "PYTHONUNBUFFERED":    "1",
            "PYTHONIOENCODING":    "utf-8",
        }
        if reuse_analysis:
            env["KAIZER_REUSE_ANALYSIS_FROM"] = reuse_analysis
        # Hand the existing pool to the bulletin pass so it cycles
        # through the user's curated images instead of asking OpenAI
        # for fresh ones from scratch. Pipe-separated because Windows
        # paths contain ':'.
        if existing_pool:
            env["KAIZER_BULLETIN_IMAGES"] = "|".join(existing_pool)
        proc = _sub.Popen(
            cmd, cwd=str(backend_root),
            stdout=_sub.PIPE, stderr=_sub.STDOUT,
            text=True, bufsize=1,
            encoding="utf-8", errors="replace",
            env=env,
        )
        for line in proc.stdout:
            ln = line.rstrip()
            if not ln:
                continue
            print(f"[bulletin-recompose:pipe job={job.id}] {ln[-400:]}")
            # Capture the [kaizer:meta] markers — one per render path
            # (compound jobs emit two: full + shorts). The orchestrator
            # uses these to repoint the job at the new output dir.
            if "[kaizer:meta]" in ln:
                payload = ln.split("[kaizer:meta]", 1)[1].strip()
                if payload:
                    new_meta_paths.append(payload)
        return proc.wait()

    def _run():
        # ``bdir`` is rewritten by the swap-and-cleanup step below once
        # the new render lands at a fresh ``output/youtube_full/<ts>/``
        # directory. Declaring it nonlocal here keeps Python from
        # treating that assignment as creating a fresh local that
        # shadows the enclosing scope (which would make this very
        # first ``_apply_scope_invalidation(bdir, ...)`` call crash
        # with UnboundLocalError).
        nonlocal bdir
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
                rc = _run_pipeline_sync()
            except Exception as exc:
                with _RECOMPOSE_LOCK:
                    _RECOMPOSING[job_id] = {"state": "failed", "msg": str(exc)[:300]}
                print(f"[bulletin-recompose] job={job_id} failed (fast pass): {exc}")
                return
            if rc != 0:
                with _RECOMPOSE_LOCK:
                    _RECOMPOSING[job_id] = {
                        "state": "failed",
                        "msg":   f"pipeline subprocess exited rc={rc}",
                    }
                print(f"[bulletin-recompose] job={job_id} fast pass returned rc={rc}")
                return

            # ── Render-new / repoint swap ──────────────────────────────
            # The pipeline always writes to a fresh ``<output>/youtube_full/<ts>/``
            # directory. To make the editor's player pick up the new
            # bulletin without breaking anything else, we:
            #   1. Resolve the NEW youtube_full output dir from the
            #      [kaizer:meta] markers captured during the subprocess.
            #   2. Repoint ``job.output_dir`` at it.
            #   3. RENAME the OLD paired directories with a ``.old``
            #      suffix instead of deleting them. We lost a user's
            #      replaced images by deleting outright once — never
            #      again. A disk-janitor script can prune ``*.old``
            #      dirs older than N days; that decision is reversible.
            try:
                new_yt_full_dir: Optional[Path] = None
                new_yt_short_dir: Optional[Path] = None
                for mp in new_meta_paths:
                    pp = Path(mp.strip())
                    parent = pp.parent
                    if "youtube_full" in str(parent).replace("\\", "/").lower():
                        new_yt_full_dir = parent
                    elif "youtube_short" in str(parent).replace("\\", "/").lower():
                        new_yt_short_dir = parent

                if new_yt_full_dir and new_yt_full_dir.is_dir():
                    old_yt_full = bdir.parent
                    backend_root = Path(__file__).resolve().parent.parent
                    raw_out = (job.output_dir or "").strip()
                    if raw_out:
                        candidate = Path(raw_out)
                        if not candidate.is_absolute():
                            candidate = (backend_root / candidate).resolve()
                        if (candidate != old_yt_full
                                and candidate.is_dir()
                                and "youtube_short" in str(candidate).replace("\\", "/").lower()):
                            old_yt_short = candidate
                        else:
                            old_yt_short = None
                    else:
                        old_yt_short = None

                    new_rel = str(new_yt_full_dir.resolve())
                    # Build path-substitution map for Clip rows.
                    # The editor's player URL is built from Clip.file_path,
                    # not from job.output_dir — so repointing the job
                    # alone leaves the player fetching the OLD path
                    # (which we just demoted). Rewrite every Clip row
                    # of this job: old yt_full prefix → new yt_full,
                    # old yt_short prefix → new yt_short, in
                    # file_path / thumb_path / image_path / meta JSON.
                    subs: list[tuple[str, str]] = []
                    backend_root = Path(__file__).resolve().parent.parent
                    if old_yt_full and new_yt_full_dir:
                        # Absolute + project-relative variants — Clip
                        # rows have been observed with both shapes.
                        abs_old = str(old_yt_full.resolve())
                        abs_new = str(new_yt_full_dir.resolve())
                        subs.append((abs_old, abs_new))
                        try:
                            rel_old = str(old_yt_full.resolve().relative_to(backend_root))
                            rel_new = str(new_yt_full_dir.resolve().relative_to(backend_root))
                            subs.append((rel_old, rel_new))
                        except ValueError:
                            pass
                    if old_yt_short and new_yt_short_dir:
                        abs_old = str(old_yt_short.resolve())
                        abs_new = str(new_yt_short_dir.resolve())
                        subs.append((abs_old, abs_new))
                        try:
                            rel_old = str(old_yt_short.resolve().relative_to(backend_root))
                            rel_new = str(new_yt_short_dir.resolve().relative_to(backend_root))
                            subs.append((rel_old, rel_new))
                        except ValueError:
                            pass

                    def _sub_all(s: str) -> str:
                        if not s:
                            return s
                        out = s
                        for a, b in subs:
                            if not a or not b:
                                continue
                            out = out.replace(a, b)
                            # JSON-escaped variant — meta column stores
                            # paths inside JSON where every backslash is
                            # doubled. Replace both shapes.
                            out = out.replace(a.replace("\\", "\\\\"), b.replace("\\", "\\\\"))
                        return out

                    job_db = SessionLocal()
                    try:
                        j2 = job_db.query(models.Job).filter(models.Job.id == job.id).first()
                        if j2:
                            j2.output_dir = new_rel
                        # Rewrite every Clip row's path fields.
                        clip_rows = job_db.query(models.Clip).filter(models.Clip.job_id == job.id).all()
                        rewritten = 0
                        for cr in clip_rows:
                            changed = False
                            for attr in ("file_path", "thumb_path", "image_path"):
                                v = getattr(cr, attr, None)
                                if isinstance(v, str) and v:
                                    nv = _sub_all(v)
                                    if nv != v:
                                        setattr(cr, attr, nv)
                                        changed = True
                            if cr.meta:
                                nm = _sub_all(cr.meta)
                                if nm != cr.meta:
                                    cr.meta = nm
                                    changed = True
                            if changed:
                                rewritten += 1
                        job_db.commit()
                        print(f"[bulletin-recompose] job={job_id} repointed output_dir -> {new_rel}, rewrote {rewritten}/{len(clip_rows)} clip path fields")
                    finally:
                        job_db.close()

                    # Safe demotion — rename OLD dirs to ``<name>.old.<ts>``.
                    # If a .old already exists from a previous recompose
                    # we just leave the current name in place rather
                    # than risk clobbering it.
                    import time as _time
                    ts_suffix = _time.strftime("%Y%m%d_%H%M%S")
                    for old_dir in (old_yt_full, old_yt_short):
                        if old_dir and old_dir.is_dir() and old_dir not in (new_yt_full_dir, new_yt_short_dir):
                            demoted = old_dir.with_name(f"{old_dir.name}.old.{ts_suffix}")
                            try:
                                old_dir.rename(demoted)
                                print(f"[bulletin-recompose] job={job_id} demoted OLD dir: {old_dir} -> {demoted.name}")
                            except OSError as exc:
                                print(f"[bulletin-recompose] could not demote {old_dir}: {exc}")

                    # Switch ``bdir`` to the new path for downstream
                    # steps (verify pass + done message).
                    bdir = new_yt_full_dir / "bulletin"
                else:
                    print(f"[bulletin-recompose] job={job_id} could not find new yt_full dir in "
                          f"{len(new_meta_paths)} meta markers — old paths kept")
            except Exception as exc:
                # Swap failed but the new render is on disk — log and
                # leave the old paths in place so the user can still
                # find the result by walking output/.
                print(f"[bulletin-recompose] job={job_id} swap-and-cleanup soft-fail: {exc}")

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
                    rc2 = _run_pipeline_sync()
                    if rc2 != 0:
                        raise RuntimeError(f"pipeline subprocess exited rc={rc2}")
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
