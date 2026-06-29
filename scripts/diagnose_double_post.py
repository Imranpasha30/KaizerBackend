"""Diagnose the double-post + per-channel branding report.

Shows the most recent publish tasks, their child upload jobs, the clip
each REALLY is, the MasterVideo each resolves to (+ its r2_key), and
the per-channel logo/watermark config — so we can see (1) whether two
different clips collapsed onto one MasterVideo (the double-post) and
(2) which channels have branding configured.
"""
import sys
sys.path.insert(0, ".")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")

from sqlalchemy import text
from database import SessionLocal

db = SessionLocal()
try:
    rows = db.execute(text("""
        SELECT pt.id AS task, uj.id AS ujob, uj.clip_id, c.frame_type,
               uj.channel_id, ch.name AS channel, uj.status,
               pt.master_video_id AS mv, mv.r2_key, mv.clean_master,
               mv.source_upload_id AS job
        FROM upload_jobs_v2 uj
        JOIN publish_tasks pt ON pt.id = uj.publish_task_id
        LEFT JOIN clips c ON c.id = uj.clip_id
        LEFT JOIN channels ch ON ch.id = uj.channel_id
        LEFT JOIN master_videos mv ON mv.id = pt.master_video_id
        ORDER BY uj.id DESC LIMIT 30
    """)).fetchall()

    print("=== recent upload jobs (newest first) ===")
    print(f"{'ujob':>5} {'task':>4} {'clip':>5} {'frame':>10} "
          f"{'chan':>5} {'channel':<16} {'mv':>4} {'status':<12} r2_key")
    seen_mv_clips = {}  # master_video_id -> set of clip frame_types
    for r in rows:
        (task, ujob, clip_id, frame, chan, channel, status,
         mv, r2_key, clean, job) = r
        print(f"{ujob:>5} {task:>4} {str(clip_id):>5} {str(frame):>10} "
              f"{str(chan):>5} {str(channel or '')[:16]:<16} {str(mv):>4} "
              f"{str(status):<12} {r2_key}")
        if mv is not None:
            seen_mv_clips.setdefault(mv, set()).add((clip_id, frame))

    print("\n=== MasterVideo → distinct clips pointing at it ===")
    print("(more than one clip on a master = THE DOUBLE-POST BUG)")
    for mv, clips in seen_mv_clips.items():
        flag = "  <-- BUG: 1 master, many clips" if len(clips) > 1 else ""
        print(f"  master {mv}: clips={sorted(str(c) for c in clips)}{flag}")

    # Per-channel branding config for the channels above.
    chan_ids = sorted({r[4] for r in rows if r[4] is not None})
    if chan_ids:
        print("\n=== per-channel branding config ===")
        for cid in chan_ids:
            ch = db.execute(text(
                "SELECT name, logo_asset_id, watermark_text, watermark_opacity "
                "FROM channels WHERE id = :c"), {"c": cid}).fetchone()
            tok = db.execute(text(
                "SELECT logo_asset_id FROM oauth_tokens WHERE channel_id = :c"),
                {"c": cid}).fetchone()
            bp = db.execute(text(
                "SELECT count(*) FROM brand_profiles WHERE "
                "(owner_kind='channel' AND owner_id=:c)"), {"c": cid}).scalar()
            tok_logo = tok[0] if tok else None
            eff_logo = tok_logo or (ch[1] if ch else None)
            wm = (ch[2] if ch else "") or ""
            has_brand = bool(eff_logo) or bool(wm.strip())
            verdict = "BRANDED" if has_brand else "NO BRANDING (unconfigured)"
            print(f"  channel {cid} ({(ch[0] if ch else '?')}): "
                  f"channel.logo={ch[1] if ch else None} "
                  f"token.logo={tok_logo} -> effective_logo={eff_logo} "
                  f"watermark_text={wm!r} brand_profiles={bp} => {verdict}")

    # Are the recently-published masters from jobs that have MULTIPLE
    # clips (= the long+short-share-one-job collapse)?
    print("\n=== master -> job -> sibling clips (the collapse check) ===")
    for mv in sorted({r[7] for r in rows if r[7] is not None}, reverse=True)[:6]:
        row = db.execute(text(
            "SELECT source_upload_id, r2_key FROM master_videos WHERE id = :m"),
            {"m": mv}).fetchone()
        if not row:
            continue
        job = row[0]
        clips = db.execute(text(
            "SELECT id, frame_type, filename FROM clips WHERE job_id = :j "
            "ORDER BY clip_index"), {"j": job}).fetchall()
        print(f"  master {mv} (job {job}, key {row[1]}): "
              f"job has {len(clips)} clip(s):")
        for c in clips:
            print(f"      clip {c[0]} frame={c[1]!r} file={c[2]}")
        if len(clips) > 1:
            print("      ^^ MULTIPLE clips share this job -> one master "
                  "can't represent them all (the double-post)")
finally:
    db.close()
