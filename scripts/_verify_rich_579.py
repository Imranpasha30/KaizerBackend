"""Phase-1 verification: render a job's bulletin with the AI Director ON
(effects_mode='rich' + directives), exactly like the orchestrator does, to
judge the real output quality before wiring it as a default. Outputs
bulletin_RICH.mp4 and prints the Director's per-story plan. Throwaway."""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
from types import SimpleNamespace as NS

# Force the Director ON for this render (orchestrator normally gates this).
os.environ["KAIZER_V4_DIRECTOR"] = "1"
os.environ["KAIZER_V4_EFFECTS_MODE"] = "rich"
os.environ.setdefault("KAIZER_V4_CONTENT_TYPE", "news")
# Serialize per-story compose to 1 ffmpeg at a time — these are repeated
# verification renders on a host that has hard-reset under concurrent NVENC.
# Slower, but only one GPU encode runs at once. (Prod default is 3.)
os.environ.setdefault("KAIZER_V4_RENDER_CONCURRENCY", "1")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from database import SessionLocal
import models
from routers.v4_editor import _read_canvas, _v4_dir_for, _list_pool
from pipeline_v4 import v1_bridge, orchestrator as _v4_orch
from pipeline_v4.encoder import active_backend_label
from pipeline_v4 import director as _director

JOB = int(sys.argv[1]) if len(sys.argv) > 1 else 579

db = SessionLocal()
try:
    job = db.query(models.Job).filter(models.Job.id == JOB).first()
    out_dir = _v4_dir_for(job)
    jc = _read_canvas(out_dir)
    bc = jc.bulletin
    pool_dir = out_dir / "_pool"
    pool_paths = [str(pool_dir / p["filename"])
                  for p in _list_pool(out_dir, JOB)
                  if (pool_dir / p["filename"]).is_file()]
    ud = _v4_orch._load_user_defaults(JOB) or {}
    channel_name = (ud.get("brand_suffix") or "").lstrip(" |·-_•").strip()
    brand_logo = ud.get("brand_logo_path") or ""
    language = jc.language or "te"

    render_stories = [NS(
        title_native=cs.title_native, title_english=cs.title_english,
        summary=cs.summary, video_t_start=cs.video_t_start,
        video_t_end=cs.video_t_end, images=list(cs.images or []),
        text_blocks=list(cs.text_blocks or []),
        name_strap=getattr(cs, "name_strap", False),
        story_index=cs.story_index) for cs in bc.stories]

    # words_by_story sidecar (optional; Director degrades to text-only)
    wbs = {}
    sw = out_dir / "story_words.json"
    if sw.is_file():
        try:
            wbs = json.loads(sw.read_text(encoding="utf-8"))
        except Exception:
            wbs = {}

    print(f"[rich] job {JOB}: {len(render_stories)} stories, director_enabled="
          f"{_director.directives_enabled()}, encoder={active_backend_label()}", flush=True)

    t0 = time.time()
    plan = None
    try:
        plan = _director.plan_direction(
            stories=render_stories, words_by_story=wbs, category="news",
            source_path=jc.trimmed_bulletin_path, language=language)
    except Exception as exc:
        print(f"[rich] plan_direction FAILED (soft): {exc}", flush=True)
    print(f"[rich] director planned in {int(time.time()-t0)}s", flush=True)
    if isinstance(plan, dict):
        for k in sorted(plan.keys(), key=lambda x: str(x)):
            d = plan[k]
            print(f"[rich]  story {k}: mood={getattr(d,'mood',None)!r} "
                  f"transition_in={getattr(d,'transition_in',None)!r} "
                  f"fx={getattr(d,'fx',None)} "
                  f"overlays={[getattr(o,'id',o) for o in (getattr(d,'overlays',None) or [])]} "
                  f"captions={bool(getattr(d,'captions',None))}", flush=True)
    else:
        print(f"[rich]  plan is {type(plan).__name__} (no per-story directives)", flush=True)

    t1 = time.time()
    out = v1_bridge.render_bulletin(v1_bridge.BulletinRenderInputs(
        trimmed_bulletin_path=jc.trimmed_bulletin_path,
        stories=render_stories,
        output_path=str(out_dir / "bulletin_RICH.mp4"),
        work_dir=out_dir,
        language=language,
        brand_logo=brand_logo,
        sidebar_images=pool_paths,
        layout=bc.layout,
        channel_name=channel_name,
        story_transition=getattr(bc, "story_transition", None),
        effects_mode="rich",
        directives=plan,
        watermark_text="", watermark_opacity=0.0, watermark_position="top-right",
    ))
    print(f"[rich] DONE render in {int(time.time()-t1)}s -> {out}", flush=True)
finally:
    db.close()
