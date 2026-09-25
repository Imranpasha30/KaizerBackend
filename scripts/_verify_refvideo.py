"""Phase-3a verification: inject a full-screen reference-VIDEO cutaway into a
story and render, to confirm the compositor plays B-roll full-screen for its
window (hard cut, ticker/lower-third on top). Throwaway."""
from __future__ import annotations
import os, sys
from pathlib import Path

os.environ["KAIZER_V4_EFFECTS_MODE"] = "off"        # isolate the cutaway
os.environ.setdefault("KAIZER_V4_RENDER_CONCURRENCY", "1")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from database import SessionLocal
import models
from routers.v4_editor import _read_canvas, _v4_dir_for, _list_pool
from pipeline_v4 import v1_bridge, orchestrator as _v4_orch
from pipeline_v4.canvas_schema import CanvasImage

JOB = 579
db = SessionLocal()
try:
    job = db.query(models.Job).filter(models.Job.id == JOB).first()
    out_dir = _v4_dir_for(job)
    jc = _read_canvas(out_dir)
    bc = jc.bulletin
    pool_dir = out_dir / "_pool"
    pool_paths = [str(pool_dir / p["filename"]) for p in _list_pool(out_dir, JOB)
                  if (pool_dir / p["filename"]).is_file()]
    ud = _v4_orch._load_user_defaults(JOB) or {}
    channel_name = (ud.get("brand_suffix") or "").lstrip(" |·-_•").strip()

    MODE = sys.argv[1] if len(sys.argv) > 1 else "duck"   # duck | mute
    # Inject a full-screen B-roll cutaway into story 0 at 12-17s (a clean slot).
    cut = CanvasImage(src="_ref_broll_test.mp4", t_start=12.0, t_end=17.0,
                      source="user", media_kind="video", spotlight="fullscreen",
                      audio_mode=MODE, label="TEST B-ROLL")
    bc.stories[0].images = list(bc.stories[0].images or []) + [cut]
    print(f"[refvid] audio_mode={MODE}; injected video cutaway into story 0 [12-17s]; "
          f"story0 now has {len(bc.stories[0].images)} media entries", flush=True)

    out = v1_bridge.render_bulletin(v1_bridge.BulletinRenderInputs(
        trimmed_bulletin_path=jc.trimmed_bulletin_path,
        stories=bc.stories[:1],   # story 0 only — fast duck-audio check
        output_path=str(out_dir / f"bulletin_{MODE.upper()}.mp4"),
        work_dir=out_dir, language=jc.language or "te",
        brand_logo=ud.get("brand_logo_path") or "",
        sidebar_images=pool_paths, layout=bc.layout, channel_name=channel_name,
        story_transition=getattr(bc, "story_transition", None),
        watermark_text="", watermark_opacity=0.0,
    ))
    print(f"[refvid] DONE -> {out}", flush=True)
finally:
    db.close()
