"""Phase-2 verification: render ONE short twice — plain (effects_vf="") vs
graded (effects_vf = a real Director mood grade) — to confirm the short
effects pass works and the short takes on its story's look. Throwaway."""
from __future__ import annotations
import os, sys
from pathlib import Path

os.environ["KAIZER_V4_DIRECTOR"] = "1"
os.environ["KAIZER_V4_EFFECTS_MODE"] = "rich"
os.environ.setdefault("KAIZER_V4_CONTENT_TYPE", "news")
os.environ.setdefault("KAIZER_V4_RENDER_CONCURRENCY", "1")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from database import SessionLocal
import models
from routers.v4_editor import _read_canvas, _v4_dir_for, _list_pool
from pipeline_v4 import v1_bridge, director

JOB = int(sys.argv[1]) if len(sys.argv) > 1 else 579
MOOD = sys.argv[2] if len(sys.argv) > 2 else "crime"

db = SessionLocal()
try:
    job = db.query(models.Job).filter(models.Job.id == JOB).first()
    out_dir = _v4_dir_for(job)
    jc = _read_canvas(out_dir)
    pool_dir = out_dir / "_pool"
    pool_paths = [str(pool_dir / p["filename"]) for p in _list_pool(out_dir, JOB)
                  if (pool_dir / p["filename"]).is_file()]
    shorts = jc.shorts or []
    if not shorts:
        print("no shorts in canvas"); raise SystemExit(1)
    sc = shorts[0]
    cfg = sc.short_config
    story0 = sc.stories[0] if sc.stories else None
    title = (cfg.text if cfg and cfg.text else
             ((story0.title_native or story0.title_english) if story0 else "")) or "KAIZER X"
    img = None
    if cfg and cfg.image_filename and (pool_dir / cfg.image_filename).is_file():
        img = str(pool_dir / cfg.image_filename)
    elif pool_paths:
        img = pool_paths[0]

    # Build a real Director mood grade via the exact orchestrator path.
    plan = director.formula_plan(list(jc.bulletin.stories), MOOD)
    d = next(iter(plan.values()))
    efx = director.story_fx_chain(d)
    print(f"[shortfx] job {JOB} short0 '{sc.output_filename}' layout={cfg.layout if cfg else '?'} "
          f"mood={MOOD} grade_vf={efx!r}", flush=True)

    def _render(tag, effects_vf):
        outp = str(out_dir / f"_verify/short_{tag}.mp4")
        v1_bridge.render_short(v1_bridge.ShortRenderInputs(
            trimmed_short_path=sc.trimmed_video_path,
            title_text=title, output_path=outp, work_dir=out_dir,
            layout=(cfg.layout if cfg else v1_bridge.DEFAULT_SHORTS_LAYOUT),
            language=jc.language or "te", image_path=img, thumbnail_path=img,
            font_size=(cfg.font_size if cfg else None),
            text_color=(cfg.text_color if cfg else None),
            section_pct=(cfg.section_pct.model_dump() if cfg else None),
            card_style=(cfg.card_style.model_dump() if cfg else None),
            follow_params=(cfg.follow_params.model_dump() if cfg else None),
            watermark_text="", watermark_opacity=0.0,
            effects_vf=effects_vf,
        ))
        print(f"[shortfx] rendered {tag} -> {outp}", flush=True)

    _render("plain", "")
    _render("graded", efx)
    print("[shortfx] DONE", flush=True)
finally:
    db.close()
