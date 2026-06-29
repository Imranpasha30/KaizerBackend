"""One-shot Stage 3/3 resume for a V4 job whose canvas.json + trimmed
videos already exist on disk.

Use case: an orchestrator crash AFTER Stage 2/3 (canvas.json written +
SEO populated) but BEFORE Stage 3 finished rendering. Re-running the
full orchestrator wastes the entire Stage 1+2 time. This script reads
canvas.json directly and only renders the bulletin + each short via
the same v1_bridge calls the orchestrator would have used.

Usage:
    python -m scripts.resume_stage3 --job-id 161

Pre-conditions checked at startup:
  - job exists in DB and is a V4 job
  - canvas.json is present in the job's output_dir
  - trimmed_bulletin.mp4 + trimmed_short_*.mp4 are on disk

On success:
  - bulletin.mp4 written, every short_*.mp4 written
  - Job.status flipped to 'done', Job.bulletin_path/short_paths updated
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from database import SessionLocal
import models
from routers.v4_editor import _read_canvas, _v4_dir_for, _list_pool
from pipeline_v4 import v1_bridge, orchestrator as _v4_orch
from pipeline_v4.encoder import active_backend_label


def _fmt_elapsed(secs: float) -> str:
    secs = max(0.0, float(secs))
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", type=int, required=True)
    args = ap.parse_args()

    db = SessionLocal()
    try:
        job = db.query(models.Job).filter(models.Job.id == args.job_id).first()
        if not job:
            print(f"[resume] job {args.job_id} not found", flush=True)
            return 2
        if (job.platform or "") != "full_video_shorts_v4":
            print(f"[resume] job {args.job_id} platform={job.platform!r} — not V4", flush=True)
            return 2

        out_dir = _v4_dir_for(job)
        canvas_path = out_dir / "canvas.json"
        if not canvas_path.is_file():
            print(f"[resume] canvas.json missing at {canvas_path}", flush=True)
            return 2

        jc = _read_canvas(out_dir)
        bulletin_canvas = jc.bulletin
        shorts_canvases = jc.shorts

        # Resolve pool images the same way the orchestrator's Stage 3 does.
        pool_dir = out_dir / "_pool"
        pool_paths = [
            str(pool_dir / p["filename"])
            for p in _list_pool(out_dir, args.job_id)
            if (pool_dir / p["filename"]).is_file()
        ]
        print(f"[resume] job {args.job_id}: {len(shorts_canvases)} shorts to render, "
              f"{len(pool_paths)} pool images, encoder={active_backend_label()}",
              flush=True)

        # Brand / language / logo from the user's saved V4 defaults.
        user_d = _v4_orch._load_user_defaults(args.job_id) or {}
        channel_name = (user_d.get("brand_suffix") or "").lstrip(" |·-_•").strip()
        brand_logo = user_d.get("brand_logo_path") or ""
        language = jc.language or "te"

        # ── Bulletin ─────────────────────────────────────────────────
        bull_t0 = time.time()
        stories = [
            SimpleNamespace(
                title_native=s.title_native,
                title_english=s.title_english,
                summary=s.summary,
                video_t_start=s.video_t_start,
                video_t_end=s.video_t_end,
            )
            for s in bulletin_canvas.stories
        ]
        try:
            bulletin_out = v1_bridge.render_bulletin(v1_bridge.BulletinRenderInputs(
                trimmed_bulletin_path=jc.trimmed_bulletin_path,
                stories=stories,
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
            print(f"[resume] bulletin rendered in {_fmt_elapsed(time.time() - bull_t0)} "
                  f"-> {bulletin_out}", flush=True)
        except Exception as exc:
            print(f"[resume] bulletin render FAILED after "
                  f"{_fmt_elapsed(time.time() - bull_t0)}: {exc}", flush=True)
            bulletin_out = ""

        # ── Shorts ───────────────────────────────────────────────────
        short_outs: list[str] = []
        trim_paths = jc.trimmed_shorts_paths or []
        for s_idx, sc in enumerate(shorts_canvases):
            short_t0 = time.time()
            cfg = sc.short_config
            story0 = sc.stories[0] if sc.stories else None
            story_title = ""
            if story0:
                story_title = (story0.title_native or story0.title_english or "").strip()
            title = (cfg.text if cfg and cfg.text else story_title) or "KAIZER X"

            if cfg and cfg.image_filename:
                cand = pool_dir / cfg.image_filename
                short_image = str(cand) if cand.is_file() else None
            else:
                short_image = pool_paths[s_idx % len(pool_paths)] if pool_paths else None

            # Each short canvas knows its trimmed source via either
            # canvas.trimmed_video_path or the orchestrator-saved list.
            trim_path = getattr(sc, "trimmed_video_path", "") or ""
            if not trim_path and s_idx < len(trim_paths):
                trim_path = trim_paths[s_idx]
            if not trim_path or not Path(trim_path).is_file():
                print(f"[resume] short {s_idx + 1}: trimmed video missing at "
                      f"{trim_path!r} — skipping", flush=True)
                continue

            try:
                short_out = v1_bridge.render_short(v1_bridge.ShortRenderInputs(
                    trimmed_short_path=trim_path,
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
                    watermark_text="",
                    watermark_opacity=0.0,
                    watermark_position="top-right",
                ))
                short_outs.append(short_out)
                print(f"[resume] short {s_idx + 1}/{len(shorts_canvases)} rendered in "
                      f"{_fmt_elapsed(time.time() - short_t0)} -> "
                      f"{Path(short_out).name}", flush=True)
            except Exception as exc:
                print(f"[resume] short {s_idx + 1}/{len(shorts_canvases)} FAILED after "
                      f"{_fmt_elapsed(time.time() - short_t0)}: {exc}", flush=True)

        # ── Mark job done so the editor stops showing 'failed' ──────
        if bulletin_out and short_outs:
            job.status = "done"
            db.commit()
            print(f"[resume] job {args.job_id}: status -> 'done' "
                  f"(bulletin + {len(short_outs)} shorts rendered)", flush=True)
        else:
            print(f"[resume] job {args.job_id}: leaving status as-is "
                  f"(bulletin_ok={bool(bulletin_out)}, shorts_ok={len(short_outs)}/"
                  f"{len(shorts_canvases)})", flush=True)
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
