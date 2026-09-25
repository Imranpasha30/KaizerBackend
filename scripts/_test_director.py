"""Diagnostic: isolate the Director LLM plan from the sensors. Runs
plan_direction with source_path='' (skips scene/energy/tone sensors) so we
learn whether the LLM call itself works and produces VARIED per-story
decisions. Throwaway."""
from __future__ import annotations
import os, sys, json
from pathlib import Path
from types import SimpleNamespace as NS

os.environ["KAIZER_V4_DIRECTOR"] = "1"
os.environ.setdefault("KAIZER_V4_CONTENT_TYPE", "news")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from database import SessionLocal
import models
from routers.v4_editor import _read_canvas, _v4_dir_for
from pipeline_v4 import director as _director

JOB = int(sys.argv[1]) if len(sys.argv) > 1 else 579
SKIP_SENSORS = "--sensors" not in sys.argv  # default: skip sensors (fast LLM isolate)

db = SessionLocal()
try:
    job = db.query(models.Job).filter(models.Job.id == JOB).first()
    out_dir = _v4_dir_for(job)
    jc = _read_canvas(out_dir)
    bc = jc.bulletin
    stories = [NS(
        title_native=cs.title_native, title_english=cs.title_english,
        summary=cs.summary, video_t_start=cs.video_t_start,
        video_t_end=cs.video_t_end, story_index=cs.story_index)
        for cs in bc.stories]
    wbs = {}
    sw = out_dir / "story_words.json"
    if sw.is_file():
        try:
            wbs = json.loads(sw.read_text(encoding="utf-8"))
        except Exception:
            wbs = {}
    src = "" if SKIP_SENSORS else jc.trimmed_bulletin_path
    print(f"[test] job {JOB}: {len(stories)} stories, skip_sensors={SKIP_SENSORS}, "
          f"source={'<none>' if not src else 'set'}", flush=True)
    plan = _director.plan_direction(
        stories=stories, words_by_story=wbs, category="news",
        source_path=src, language=jc.language or "te")
    print(f"[test] plan type={type(plan).__name__}, n={len(plan) if hasattr(plan,'__len__') else '?'}", flush=True)
    moods = set()
    for k in sorted(plan.keys(), key=lambda x: str(x)):
        d = plan[k]
        moods.add(getattr(d, "mood", None))
        print(f"[test]  story {k}: mood={getattr(d,'mood',None)!r} "
              f"transition_in={getattr(d,'transition_in',None)!r} "
              f"fx={getattr(d,'fx',None)} "
              f"overlays={[getattr(o,'id',o) for o in (getattr(d,'overlays',None) or [])]}", flush=True)
    print(f"[test] DISTINCT moods across stories: {sorted(str(m) for m in moods)} "
          f"({len(moods)} unique) — variety={'YES' if len(moods) > 1 else 'NO (uniform → LLM likely failed)'}", flush=True)
finally:
    db.close()
