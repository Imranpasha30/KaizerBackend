"""V4 orchestrator — end-to-end job runner.

  source video  →  Step 1 (trim)  →  trimmed.mp4
                      ↓
                build canvas.json
                      ↓
                Step 2 (composite)  →  bulletin.mp4

Sub-process entry: ``python -m pipeline_v4.orchestrator --job-id N
--source <path> --output-dir <dir> --language te``.

The orchestrator also:
  - Builds the initial pool of images from the user's bulletin_images
    env (KAIZER_BULLETIN_IMAGES) or from cached _job_pool, if any
  - Asks Claude for per-story image timings (only if pool is non-empty)
  - Materialises canvas.json on disk so the editor can read/edit later
  - Updates the Job row's status + log via a thin DB adapter
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from pipeline_v4 import prompts as v4_prompts
from pipeline_v4 import trim_engine
from pipeline_v4 import canvas_engine
from pipeline_v4 import v1_bridge
from pipeline_v4 import image_provider
from pipeline_v4 import seo_provider
from pipeline_v4.canvas_schema import (
    Canvas, CanvasImage, CanvasLayout, CanvasStory, CanvasTextBlock, V4JobCanvas,
    ShortConfig, CanvasSEO,
)


CANVAS_JSON_NAME = "canvas.json"


# ─── Image-pool ingest ──────────────────────────────────────────────

def _ingest_image_pool(output_dir: Path, env_pool: str) -> list[dict]:
    """Copy/link user-provided images (via KAIZER_BULLETIN_IMAGES env)
    into ``<output_dir>/_pool/`` and return a list of
    ``{filename, label, kind}`` describing the pool. Returns [] if
    no images are available — Step 2 still renders, just without
    carousel images."""
    pool_dir = output_dir / "_pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    if not env_pool:
        return out
    seen: set[str] = set()
    for raw in (p.strip() for p in env_pool.split("|")):
        if not raw:
            continue
        src = Path(raw)
        if not src.is_file():
            continue
        filename = src.name
        # Avoid name collisions in the pool dir
        i = 0
        candidate = filename
        while candidate in seen or (pool_dir / candidate).exists():
            i += 1
            stem, suf = os.path.splitext(filename)
            candidate = f"{stem}_{i}{suf}"
        seen.add(candidate)
        try:
            shutil.copy2(src, pool_dir / candidate)
        except OSError as exc:
            print(f"[v4/pool] skip {src}: {exc}")
            continue
        out.append({"filename": candidate, "label": src.stem, "kind": "photo"})
    return out


# ─── Initial canvas builder (called once after Step 1) ──────────────

def _bulletin_layout(width: int = 1920, height: int = 1080,
                     brand_logo: Optional[str] = None) -> CanvasLayout:
    """News-channel bulletin layout (1920x1080, edge-to-edge, no gaps):

      ┌─────────────────────────────────────┬───────────────────┐
      │  VIDEO (left 65%)                   │  IMAGE (right 35%)│  top 79%
      │  1248 x 853 at (0,0)                │  672 x 853        │
      ├─────────────────────────────────────┴───────────────────┤
      │  RED full-width strap: BREAKING + main headline         │  11%
      ├─────────────────────────────────────────────────────────┤
      │  YELLOW full-width ticker: scrolling summary            │  10%
      └─────────────────────────────────────────────────────────┘

    Proportions match V1 longform_compose (which the user already
    knows is the right look). Both panels touch each other — 65 + 35
    = 100% width, no centre gap. Cover-crop scaling fills each panel
    completely (no letterbox padding).
    """
    # Defaults: studio-background-friendly geometry the operator dialled
    # in via Copy-layout (2026-06-05). Leaves visible breathing room on
    # all four sides so the bg video reads as a real news set instead of
    # peeking through a 1px frame. To revert to edge-to-edge, set
    # video_x/y/w/h_pct = 1.5625/4.63/65.625/74.07 and picture to
    # 68.75/4.63/29.6875/74.07.
    return CanvasLayout(
        width=width, height=height, bg_color="#000000",
        video_x_pct=6.756, video_y_pct=15.311,
        video_w_pct=56.664, video_h_pct=60.673,
        picture_x_pct=70.888, picture_y_pct=16.578,
        picture_w_pct=24.698, picture_h_pct=58.863,
        brand_logo_path=brand_logo,
        brand_logo_x_pct=92.0, brand_logo_y_pct=2.0, brand_logo_w_pct=6.0,
    )


def _short_layout(width: int = 1080, height: int = 1920,
                  brand_logo: Optional[str] = None) -> CanvasLayout:
    """9:16 vertical Short (1080x1920):

      ┌─────────────────────┐
      │ VIDEO  (100% x 55%) │   talking-head, cover-crop
      ├─────────────────────┤
      │ IMAGE  (100% x 24%) │   contextual image
      ├─────────────────────┤
      │ RED strap (11%)     │   headline
      ├─────────────────────┤
      │ YELLOW ticker (10%) │   summary
      └─────────────────────┘

    Stacks vertically so every pixel is filled — no black gutters."""
    return CanvasLayout(
        width=width, height=height, bg_color="#000000",
        video_x_pct=0.0, video_y_pct=0.0, video_w_pct=100.0, video_h_pct=55.0,
        # Image directly under the video, full-width, 24% tall.
        picture_x_pct=0.0, picture_y_pct=55.0,
        picture_w_pct=100.0, picture_h_pct=24.0,
        brand_logo_path=brand_logo,
        brand_logo_x_pct=82.0, brand_logo_y_pct=2.5, brand_logo_w_pct=14.0,
    )


def _default_image_timings(story_duration: float, pool_count: int) -> list[tuple[int, float, float]]:
    """Heuristic fallback when Claude doesn't decide. Cycles the pool
    in ~4s windows. Returns [(pool_index, t_start, t_end), ...]."""
    if pool_count == 0 or story_duration <= 0:
        return []
    window = 4.0
    out: list[tuple[int, float, float]] = []
    t = 0.0
    idx = 0
    while t < story_duration:
        end = min(story_duration, t + window)
        out.append((idx % pool_count, t, end))
        t = end
        idx += 1
    return out


# ISO -> default bold font basename. Matches resources/fonts/.
_LANG_TO_FONT = {
    "te": "NotoSansTelugu-Bold.ttf",
    "hi": "NotoSansDevanagari-Bold.ttf",
    "ta": "NotoSansTamil-Bold.ttf",
    "kn": "NotoSansKannada-Bold.ttf",
    "ml": "NotoSansMalayalam-Bold.ttf",
    "bn": "NotoSansBengali-Bold.ttf",
    "mr": "NotoSansDevanagari-Bold.ttf",
    "gu": "NotoSansGujarati-Bold.ttf",
}


def _build_initial_canvas(
    *,
    trim_result: trim_engine.TrimResult,
    pool: list[dict],
    kind: str,
    output_filename: str,
    brand_logo: Optional[str] = None,
    language: str = "te",
) -> Canvas:
    if kind == "bulletin":
        layout = _bulletin_layout(brand_logo=brand_logo)
        # Studio-background video the operator picked in the new-job
        # wizard (forwarded via env so the orchestrator subprocess can
        # see it without a DB lookup). Only stamped on the bulletin —
        # shorts use V1's compose path which doesn't read this field.
        _bg_path = (os.environ.get("KAIZER_V4_BG_VIDEO_PATH") or "").strip()
        if _bg_path:
            layout.bg_video_path = _bg_path
            try:
                layout.bg_video_volume = float(os.environ.get("KAIZER_V4_BG_VIDEO_VOLUME", "0.0"))
            except ValueError:
                layout.bg_video_volume = 0.0
            try:
                layout.bg_intro_seconds = float(os.environ.get("KAIZER_V4_BG_INTRO_SECONDS", "0.0"))
            except ValueError:
                layout.bg_intro_seconds = 0.0
    else:
        layout = _short_layout(brand_logo=brand_logo)

    stories: list[CanvasStory] = []
    for s in trim_result.stories:
        # Image timings — default heuristic; the editor can let Claude
        # re-decide later by toggling claude_decided_timings.
        timings = _default_image_timings(s.duration, len(pool))
        images = [
            CanvasImage(
                src=pool[idx]["filename"],
                t_start=ts, t_end=te,
                source="manual" if pool[idx].get("kind") == "user" else "claude",
                label=pool[idx].get("label"),
            )
            for (idx, ts, te) in timings
        ]
        text_blocks = []
        if s.title_native:
            # BREAKING headline — red strap below the video+image row.
            text_blocks.append(CanvasTextBlock(
                kind="lower_third",
                text=s.title_native,
                t_start=0.0, t_end=None,
            ))
        # Yellow ticker at the very bottom — uses the summary if Claude
        # provided one; otherwise repeats the headline so the bottom
        # 10% strip isn't blank.
        ticker_text = (s.summary or s.title_english or s.title_native or "").strip()
        if ticker_text:
            text_blocks.append(CanvasTextBlock(
                kind="ticker",
                text=ticker_text,
                t_start=0.0, t_end=None,
            ))
        stories.append(CanvasStory(
            story_index=s.story_index,
            video_t_start=s.video_t_start,
            video_t_end=s.video_t_end,
            title_native=s.title_native,
            title_english=s.title_english,
            summary=s.summary,
            images=images,
            text_blocks=text_blocks,
            claude_decided_timings=True,
        ))

    # Default short_config — V1's torn_card with the right script font
    # and the first pool image as hero. Editor can replace any of it.
    short_config: Optional[ShortConfig] = None
    if kind == "short":
        first_image = pool[0]["filename"] if pool else None
        primary_story = stories[0] if stories else None
        short_config = ShortConfig(
            layout="torn_card",
            text=(primary_story.title_native or primary_story.title_english
                  if primary_story else None),
            font_file=_LANG_TO_FONT.get(language, "NotoSansTelugu-Bold.ttf"),
            text_color="#FFFFFF",
            image_filename=first_image,
        )

    return Canvas(
        kind=kind,                     # "bulletin" or "short"
        output_filename=output_filename,
        layout=layout,
        stories=stories,
        trimmed_video_path=trim_result.trimmed_path,
        short_config=short_config,
    )


# ─── Persistence + DB hooks ─────────────────────────────────────────

def _save_canvas_json(out_dir: Path, job_canvas: V4JobCanvas) -> Path:
    p = out_dir / CANVAS_JSON_NAME
    p.write_text(
        json.dumps(json.loads(job_canvas.model_dump_json()),
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return p


def _update_job_status(job_id: int, *, status: str, current_stage: str = "",
                       error: str = "") -> None:
    """Best-effort DB update. Imports happen lazily so a CLI-only run
    without the FastAPI app loaded still works for testing.

    Also stamps started_at on first 'running' transition (in case the
    runner didn't set it) and finished_at on any terminal status, so
    the JobDetail elapsed badge has both endpoints."""
    try:
        from datetime import datetime as _dt, timezone as _tz
        from database import SessionLocal
        import models
        sess = SessionLocal()
        try:
            j = sess.query(models.Job).filter(models.Job.id == job_id).first()
            if not j:
                return
            j.status = status
            now = _dt.now(_tz.utc)
            if status == "running" and j.started_at is None:
                j.started_at = now
            if status in ("done", "failed", "cancelled") and j.finished_at is None:
                j.finished_at = now
            if hasattr(j, "current_stage"):
                j.current_stage = current_stage
            if error:
                j.error = (j.error or "") + ("\n" if j.error else "") + error[:2000]
            sess.commit()
        finally:
            sess.close()
    except Exception as exc:
        print(f"[v4/db] status update soft-fail: {exc}")


def _extract_thumbnail(video_path: Path, out_path: Path, *,
                       seek_sec: float = 1.5) -> bool:
    """Snap a single JPEG at ``seek_sec`` into ``out_path``. Returns
    True on success. Soft-fails (returns False) so the materialise
    step never blocks on a missing thumbnail."""
    try:
        import shutil as _sh, subprocess as _sp
        ffmpeg = _sh.which("ffmpeg") or "ffmpeg"
        cmd = [
            ffmpeg, "-y", "-v", "error",
            "-ss", f"{max(0.0, seek_sec):.2f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "3",
            str(out_path),
        ]
        r = _sp.run(cmd, capture_output=True, text=True, timeout=20)
        return r.returncode == 0 and out_path.is_file()
    except Exception:
        return False


def _materialise_clips(
    *,
    job_id: int,
    out_dir: Path,
    bulletin_canvas: Canvas,
    shorts_canvases: list,
    bulletin_path: str,
) -> int:
    """Create one Clip row per rendered V4 output so the existing
    publish/upload flow can pick them up. Idempotent — clears any
    prior V4 clips for this job before writing the fresh set."""
    from database import SessionLocal
    import models

    sess = SessionLocal()
    try:
        # Drop any prior V4 clips for this job (re-render scenario).
        sess.query(models.Clip).filter(models.Clip.job_id == job_id).delete()

        n = 0
        # Bulletin (one Clip, frame_type="bulletin").
        if bulletin_path and Path(bulletin_path).is_file():
            bc = bulletin_canvas
            story0 = bc.stories[0] if bc.stories else None
            seo_str = ""
            if bc.seo:
                try:
                    seo_str = bc.seo.model_dump_json()
                except Exception:
                    seo_str = ""
            bp = Path(bulletin_path)
            thumb = bp.with_name(bp.stem + "_thumb.jpg")
            _extract_thumbnail(bp, thumb, seek_sec=2.0)
            sess.add(models.Clip(
                job_id=job_id,
                clip_index=0,
                filename=bp.name,
                file_path=str(bp),
                thumb_path=str(thumb) if thumb.is_file() else "",
                image_path="",
                duration=float(sum(
                    (s.video_t_end - s.video_t_start) for s in bc.stories
                ) or 0.0),
                frame_type="bulletin",
                text=(story0.title_native if story0 else "") or "",
                seo=seo_str,
            ))
            n += 1

        # Shorts (one Clip each, frame_type = chosen V1 layout).
        for i, sc in enumerate(shorts_canvases):
            short_path = out_dir / sc.output_filename
            if not short_path.is_file():
                continue
            story0 = sc.stories[0] if sc.stories else None
            seo_str = ""
            if sc.seo:
                try:
                    seo_str = sc.seo.model_dump_json()
                except Exception:
                    seo_str = ""
            cfg = sc.short_config
            frame_type = (cfg.layout if cfg else "torn_card") or "torn_card"
            image_filename = (cfg.image_filename if cfg and cfg.image_filename else "")
            image_path = str(out_dir / "_pool" / image_filename) if image_filename else ""
            text = (cfg.text if cfg and cfg.text
                    else (story0.title_native if story0 else "")) or ""
            duration = (story0.video_t_end - story0.video_t_start) if story0 else 0.0
            thumb = short_path.with_name(short_path.stem + "_thumb.jpg")
            _extract_thumbnail(short_path, thumb, seek_sec=1.5)
            sess.add(models.Clip(
                job_id=job_id,
                clip_index=i + 1,
                filename=short_path.name,
                file_path=str(short_path),
                thumb_path=str(thumb) if thumb.is_file() else "",
                image_path=image_path,
                duration=float(max(0.0, duration)),
                frame_type=frame_type,
                text=text,
                seo=seo_str,
            ))
            n += 1

        sess.commit()
        return n
    finally:
        sess.close()


def _maybe_auto_publish(job_id: int, log: callable) -> None:
    """If the job's owner has ``auto_publish=True`` AND
    ``require_consent=False`` in their V4 defaults, queue an UploadJob
    for every materialised Clip to every default channel.

    When ``require_consent=True`` (the safer default) the editor shows
    a "Ready to publish to N channels — confirm?" banner instead and
    nothing leaves the box without a human click."""
    from database import SessionLocal
    import models
    sess = SessionLocal()
    try:
        job = sess.query(models.Job).filter(models.Job.id == job_id).first()
        if not job or not job.user_id:
            return
        user = sess.query(models.User).filter(models.User.id == job.user_id).first()
        if not user or not user.v4_defaults:
            return
        try:
            d = json.loads(user.v4_defaults)
        except Exception:
            return
        if not d.get("auto_publish") or d.get("require_consent", True):
            log("[v4/publish] auto-publish gated -- waiting on user consent")
            return
        channel_ids = [int(c) for c in (d.get("channel_ids") or []) if c]
        if not channel_ids:
            log("[v4/publish] auto-publish skipped -- no channels in defaults")
            return
        clips = sess.query(models.Clip).filter(
            models.Clip.job_id == job_id
        ).order_by(models.Clip.clip_index).all()
        if not clips:
            return
        privacy = d.get("privacy", "public")
        from datetime import datetime as _dt, timezone as _tz
        n = 0
        for c in clips:
            kind = "video" if (c.frame_type == "bulletin") else "short"
            for ch_id in channel_ids:
                sess.add(models.UploadJob(
                    user_id=user.id,
                    clip_id=c.id,
                    channel_id=ch_id,
                    status="queued",
                    publish_kind=kind,
                    privacy_status=privacy,
                    use_seo=True,
                    created_at=_dt.now(_tz.utc),
                ))
                n += 1
        sess.commit()
        log(f"[v4/publish] auto-publish queued {n} UploadJob(s)")
    finally:
        sess.close()


def _load_user_defaults(job_id: int) -> dict:
    """Return the owning user's V4 defaults dict (or {} on any miss).
    Used by the renderer to apply brand suffix / watermark text without
    each call site having to import the user model itself."""
    try:
        from database import SessionLocal
        import models
        sess = SessionLocal()
        try:
            j = sess.query(models.Job).filter(models.Job.id == job_id).first()
            if not j or not j.user_id:
                return {}
            u = sess.query(models.User).filter(models.User.id == j.user_id).first()
            if not u or not u.v4_defaults:
                return {}
            return json.loads(u.v4_defaults) or {}
        finally:
            sess.close()
    except Exception:
        return {}


def _get_job_user_id(job_id: int) -> Optional[int]:
    """Best-effort lookup of the job's owner so generated assets can
    land in user_assets/<uid>/. Returns None when the DB isn't reachable
    (e.g. CLI test runs). The image provider then skips the assets
    copy without erroring."""
    try:
        from database import SessionLocal
        import models
        sess = SessionLocal()
        try:
            j = sess.query(models.Job).filter(models.Job.id == job_id).first()
            return getattr(j, "user_id", None) if j else None
        finally:
            sess.close()
    except Exception:
        return None


def _append_job_log(job_id: int, line: str) -> None:
    try:
        from database import SessionLocal
        import models
        sess = SessionLocal()
        try:
            j = sess.query(models.Job).filter(models.Job.id == job_id).first()
            if j:
                j.log = (j.log or "") + line + "\n"
                sess.commit()
        finally:
            sess.close()
    except Exception:
        pass


# ─── Public end-to-end entry ────────────────────────────────────────

def run_job(
    *,
    job_id: int,
    source_video: str,
    output_dir: str,
    language: str = "te",
    brand_logo: Optional[str] = None,
    bulletin_images_env: str = "",
) -> str:
    """Run the full V4 pipeline. Returns the canvas.json path."""
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        print(msg, flush=True)
        _append_job_log(job_id, msg)

    # Job-global stopwatch + per-stage timer. Same pattern as V2/V3 so
    # the user can read off the per-stage and total elapsed in one glance.
    job_t0 = time.time()
    def _fmt_elapsed(secs: float) -> str:
        return f"{secs:.1f}s" if secs < 60 else f"{int(secs // 60)}m{int(secs % 60):02d}s"

    _log(f"[v4] job {job_id} starting -- source={source_video}")
    _log(f"[v4] output_dir={out_dir}")
    _update_job_status(job_id, status="running", current_stage="step1_trim")

    try:
        # ─── Stage 1/3 — atomic trim+concat ────────────────────────
        stage_t0 = time.time()
        trim_result = trim_engine.run_step1(
            source_video=source_video,
            output_dir=str(out_dir),
            language=language,
            output_filename="trimmed_bulletin.mp4",
        )
        _log(f"[v4] Stage 1/3 DONE in {_fmt_elapsed(time.time() - stage_t0)} "
             f"-- {trim_result.trimmed_duration_sec:.1f}s output, "
             f"{len(trim_result.stories)} stories  (total {_fmt_elapsed(time.time() - job_t0)})")

        # ─── Ingest user-supplied image pool ────────────────────────
        pool = _ingest_image_pool(out_dir, bulletin_images_env)
        _log(f"[v4] pool ingested -- {len(pool)} images")

        # ─── Auto-fetch authentic images if user gave none ─────────
        # User-uploaded images always win. Only when the pool is empty
        # do we run V1's CSE/DDG/Pexels/OpenAI chain to land one image
        # per story. Generated images also land in the operator's
        # user_assets/ folder so they're reusable across future jobs.
        try:
            user_assets = image_provider._user_assets_dir_for(_get_job_user_id(job_id))
            new_imgs = image_provider.auto_populate_pool(
                stories=list(trim_result.stories),
                pool_dir=out_dir / "_pool",
                language=language,
                user_assets_dir=user_assets,
                only_if_empty=True,
            )
            if new_imgs:
                pool.extend(new_imgs)
                _log(f"[v4] auto-fetched {len(new_imgs)} authentic image(s)")
        except Exception as exc:
            _log(f"[v4] image auto-fetch failed (soft-skip): {exc}")

        # ─── Stage 2/3 — build canvas + per-story short trims ──────
        stage_t0 = time.time()
        _update_job_status(job_id, status="running", current_stage="step2_canvas")
        bulletin_canvas = _build_initial_canvas(
            trim_result=trim_result,
            pool=pool,
            kind="bulletin",
            output_filename="bulletin.mp4",
            brand_logo=brand_logo,
            language=language,
        )

        # ─── Per-story shorts ───────────────────────────────────────
        # Goal: at LEAST 4 shorts per job (target 4-5+). Long stories
        # (>90s) get chunked into ~60s sub-windows; medium stories
        # become one short each; stories <8s are skipped unless we'd
        # otherwise fall under the minimum (then the floor drops to 5s).
        SHORT_MIN_SEC      = 8.0
        SHORT_MIN_FALLBACK = 5.0     # used only when total candidates < TARGET_MIN
        SHORT_MAX_SEC      = 90.0
        SHORT_TARGET_SEC   = 60.0
        TARGET_MIN_SHORTS  = 4

        def _split_spans(spans: list, target_sec: float) -> list[list]:
            """Greedy partition: walk spans in order, start a new chunk
            whenever the running duration would exceed ``target_sec``.
            Returns a list of chunks (each a list of KeptSpan)."""
            chunks: list[list] = []
            cur: list = []
            cur_dur = 0.0
            for sp in spans:
                d = max(0.0, sp.duration)
                if cur and cur_dur + d > target_sec:
                    chunks.append(cur)
                    cur = []
                    cur_dur = 0.0
                cur.append(sp)
                cur_dur += d
            if cur:
                chunks.append(cur)
            return chunks

        # Build candidate list first (no I/O), then trim. Each candidate
        # is (display_idx, title_native, title_english, summary, spans, est_dur).
        candidates: list[tuple] = []
        for s_idx, s in enumerate(trim_result.stories):
            if s.duration < SHORT_MIN_SEC:
                continue
            if s.duration <= SHORT_MAX_SEC:
                candidates.append((
                    s_idx, s.title_native, s.title_english, s.summary,
                    list(s.source_spans), s.duration, None,
                ))
            else:
                # Split into ~60s sub-windows.
                chunks = _split_spans(list(s.source_spans), SHORT_TARGET_SEC)
                for sub_i, chunk in enumerate(chunks):
                    chunk_dur = sum(sp.duration for sp in chunk)
                    if chunk_dur < SHORT_MIN_SEC:
                        continue
                    candidates.append((
                        s_idx, s.title_native, s.title_english, s.summary,
                        chunk, chunk_dur, sub_i + 1,
                    ))

        # Fallback: still under the minimum? accept stories 5-8s too.
        if len(candidates) < TARGET_MIN_SHORTS:
            existing_keys = {(c[0], c[6]) for c in candidates}
            for s_idx, s in enumerate(trim_result.stories):
                if SHORT_MIN_FALLBACK <= s.duration < SHORT_MIN_SEC and (s_idx, None) not in existing_keys:
                    candidates.append((
                        s_idx, s.title_native, s.title_english, s.summary,
                        list(s.source_spans), s.duration, None,
                    ))
                    if len(candidates) >= TARGET_MIN_SHORTS:
                        break

        _log(f"[v4] shorts plan -- {len(candidates)} candidates "
             f"(target >= {TARGET_MIN_SHORTS})")

        shorts_canvases: list[Canvas] = []
        trimmed_shorts_paths: list[str] = []
        for idx, (s_idx, title_n, title_e, summary, spans, dur, sub_i) in enumerate(candidates):
            label = (
                f"short_{idx + 1:02d}"
                if sub_i is None
                else f"short_{idx + 1:02d}_s{s_idx + 1:02d}p{sub_i}"
            )
            short_trimmed_path = str(out_dir / f"trimmed_{label}.mp4")
            try:
                trim_engine.atomic_trim_concat(
                    source_video=source_video,
                    spans=spans,
                    output_path=short_trimmed_path,
                )
            except Exception as exc:
                _log(f"[v4] {label}: trim failed: {exc}")
                continue
            trimmed_shorts_paths.append(short_trimmed_path)
            _log(f"[v4] {label}: trimmed {dur:.1f}s -> {Path(short_trimmed_path).name}")

            short_trim = trim_engine.TrimResult(
                trimmed_path=short_trimmed_path,
                trimmed_duration_sec=dur,
                stories=[
                    trim_engine.TrimmedStory(
                        story_index=0,
                        title_native=title_n,
                        title_english=title_e,
                        summary=summary,
                        video_t_start=0.0,
                        video_t_end=dur,
                        source_spans=spans,
                    )
                ],
                source_duration_sec=trim_result.source_duration_sec,
                removed_sec_total=trim_result.removed_sec_total,
            )
            short_canvas = _build_initial_canvas(
                trim_result=short_trim,
                pool=pool,
                kind="short",
                output_filename=f"{label}.mp4",
                brand_logo=brand_logo,
                language=language,
            )
            shorts_canvases.append(short_canvas)

        # ─── SEO — Gemini 2.5 Flash per short + one for the bulletin ──
        # Soft-fails per canvas so a Gemini outage never blocks render.
        seo_t0 = time.time()
        try:
            bull_story = bulletin_canvas.stories[0] if bulletin_canvas.stories else None
            bull_body = "\n".join(
                f"- {s.title_native or s.title_english}"
                for s in bulletin_canvas.stories if s.title_native or s.title_english
            )
            bull_seo = seo_provider.generate_seo(seo_provider.SeoInput(
                kind="bulletin",
                language=language,
                title_native=(bull_story.title_native if bull_story else "") or "",
                title_english=(bull_story.title_english if bull_story else "") or "",
                summary=(bull_story.summary if bull_story else "") or "",
                body=bull_body,
            ))
            bulletin_canvas.seo = CanvasSEO(**bull_seo)
            for sc in shorts_canvases:
                st = sc.stories[0] if sc.stories else None
                if not st:
                    continue
                sc.seo = CanvasSEO(**seo_provider.generate_seo(seo_provider.SeoInput(
                    kind="short",
                    language=language,
                    title_native=st.title_native or "",
                    title_english=st.title_english or "",
                    summary=st.summary or "",
                )))
            _log(f"[v4] SEO generated for bulletin + {len(shorts_canvases)} "
                 f"shorts in {_fmt_elapsed(time.time() - seo_t0)}")
        except Exception as exc:
            _log(f"[v4] SEO generation failed (soft-skip): {exc}")

        job_canvas = V4JobCanvas(
            job_id=job_id,
            language=language,
            bulletin=bulletin_canvas,
            shorts=shorts_canvases,
            trimmed_bulletin_path=trim_result.trimmed_path,
            trimmed_shorts_paths=trimmed_shorts_paths,
        )
        canvas_json_path = _save_canvas_json(out_dir, job_canvas)
        _log(f"[v4] Stage 2/3 DONE in {_fmt_elapsed(time.time() - stage_t0)} "
             f"-- canvas.json written, {len(shorts_canvases)} shorts queued  "
             f"(total {_fmt_elapsed(time.time() - job_t0)})")

        # ─── Stage 3/3 — render bulletin + shorts via V1 layouts ───
        stage_t0 = time.time()
        pool_dir = out_dir / "_pool"
        # Absolute paths in the pool — V1 composers expect file paths.
        pool_paths = [str(pool_dir / p["filename"]) for p in pool
                      if (pool_dir / p["filename"]).is_file()]

        # Bulletin: V1 broadcast layout (left video / right sidebar /
        # animated lower-third / scrolling ticker).
        bull_t0 = time.time()
        try:
            user_d = _load_user_defaults(job_id)
            channel_name = (
                (user_d.get("brand_suffix") or "")
                .lstrip(" |·-_•").strip()
            )  # strip any " | " / " - " prefix the user typed in defaults
            # Watermark is NOT baked in here — render produces a clean
            # file. The upload worker (pipeline_v4.watermark.stamp_for_
            # channel) stamps each destination's own logo + text at
            # publish time so the same source can ship to N channels
            # with N different brand stamps.
            bulletin_out = v1_bridge.render_bulletin(v1_bridge.BulletinRenderInputs(
                trimmed_bulletin_path=trim_result.trimmed_path,
                stories=list(trim_result.stories),
                output_path=str(out_dir / bulletin_canvas.output_filename),
                work_dir=out_dir,
                language=language,
                brand_logo=brand_logo,
                sidebar_images=pool_paths,
                layout=bulletin_canvas.layout,
                channel_name=channel_name,
                watermark_text="",
                watermark_opacity=0.0,
                watermark_position="top-right",
            ))
            _log(f"[v4] bulletin rendered in {_fmt_elapsed(time.time() - bull_t0)} -> {bulletin_out}")
        except Exception as exc:
            _log(f"[v4] bulletin render failed after "
                 f"{_fmt_elapsed(time.time() - bull_t0)}: {exc}")
            bulletin_out = ""

        # Shorts: pull every V1-parity knob from sc.short_config so the
        # editor's controls drive the render. Title falls back to the
        # parent story when the user hasn't typed an override.
        for s_idx, sc in enumerate(shorts_canvases):
            short_t0 = time.time()
            cfg = sc.short_config
            story0 = sc.stories[0] if sc.stories else None
            story_title = ""
            if story0:
                story_title = (story0.title_native or story0.title_english or "").strip()
            title = (cfg.text if cfg and cfg.text else story_title) or "KAIZER NEWS"

            # Image: editor's choice first, then round-robin fallback.
            if cfg and cfg.image_filename:
                cand = pool_dir / cfg.image_filename
                short_image = str(cand) if cand.is_file() else None
            else:
                short_image = pool_paths[s_idx % len(pool_paths)] if pool_paths else None

            try:
                short_out = v1_bridge.render_short(v1_bridge.ShortRenderInputs(
                    trimmed_short_path=sc.trimmed_video_path,
                    title_text=title,
                    output_path=str(out_dir / sc.output_filename),
                    work_dir=out_dir,
                    layout=(cfg.layout if cfg else v1_bridge.DEFAULT_SHORTS_LAYOUT),
                    language=language,
                    image_path=short_image,
                    brand_logo=brand_logo,
                    thumbnail_path=short_image,
                    font_file=(cfg.font_file if cfg else None),
                    font_size=(cfg.font_size if cfg else None),
                    text_color=(cfg.text_color if cfg else None),
                    section_pct=(cfg.section_pct.model_dump() if cfg else None),
                    card_style=(cfg.card_style.model_dump() if cfg else None),
                    follow_params=(cfg.follow_params.model_dump() if cfg else None),
                    # Clean render — per-channel stamping happens in the
                    # upload worker. See orchestrator.py bulletin block.
                    watermark_text="",
                    watermark_opacity=0.0,
                    watermark_position="top-right",
                ))
                _log(f"[v4] short {s_idx + 1} rendered in "
                     f"{_fmt_elapsed(time.time() - short_t0)} -> {short_out}")
            except Exception as exc:
                _log(f"[v4] short {s_idx + 1}: render failed after "
                     f"{_fmt_elapsed(time.time() - short_t0)}: {exc}")

        _log(f"[v4] Stage 3/3 DONE in {_fmt_elapsed(time.time() - stage_t0)}  "
             f"(total {_fmt_elapsed(time.time() - job_t0)})")

        # ─── Materialise each rendered output as a Clip row ────────
        # The existing publish / YouTube-upload pipeline reads Clip rows.
        # Without this V4 outputs would be invisible to the publish
        # modal even though they're sitting on disk. SEO from canvas.json
        # rides into clip.seo so _compose_metadata picks it up unchanged.
        try:
            n_clips = _materialise_clips(
                job_id=job_id,
                out_dir=out_dir,
                bulletin_canvas=bulletin_canvas,
                shorts_canvases=shorts_canvases,
                bulletin_path=bulletin_out,
            )
            _log(f"[v4] materialised {n_clips} Clip row(s) for publish")
        except Exception as exc:
            _log(f"[v4] clip materialisation failed (soft-skip): {exc}")

        # ─── Optional auto-publish ────────────────────────────────
        # If the user enabled auto_publish (and didn't require an
        # explicit consent click), enqueue uploads now. With consent
        # required, we leave the publish for the user to confirm in
        # the editor's "Ready to publish" banner.
        try:
            _maybe_auto_publish(job_id, _log)
        except Exception as exc:
            _log(f"[v4] auto-publish soft-skip: {exc}")

        _update_job_status(job_id, status="done", current_stage="done")
        _log(f"[v4] job {job_id} DONE -- total {_fmt_elapsed(time.time() - job_t0)}")

        # Machine-readable markers the runner picks up
        print(f"[kaizer:v4:canvas] {canvas_json_path}", flush=True)
        print(f"[kaizer:v4:bulletin] {bulletin_out}", flush=True)
        return str(canvas_json_path)

    except Exception as exc:
        tb = traceback.format_exc()
        _log(f"[v4] FAILED after {_fmt_elapsed(time.time() - job_t0)}: {exc}")
        _log(tb)
        _update_job_status(job_id, status="failed", error=str(exc)[:1500])
        raise


# ─── CLI ────────────────────────────────────────────────────────────

def _cli() -> int:
    ap = argparse.ArgumentParser(description="V4 pipeline runner")
    ap.add_argument("--job-id", type=int, required=True)
    ap.add_argument("--source", required=True, help="path to source video")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--language", default="te")
    ap.add_argument("--brand-logo", default=None)
    args = ap.parse_args()

    bulletin_images_env = os.environ.get("KAIZER_BULLETIN_IMAGES", "")

    try:
        run_job(
            job_id=args.job_id,
            source_video=args.source,
            output_dir=args.output_dir,
            language=args.language,
            brand_logo=args.brand_logo,
            bulletin_images_env=bulletin_images_env,
        )
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(_cli())
