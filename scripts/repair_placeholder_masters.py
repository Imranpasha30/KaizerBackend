"""One-time repair: heal MasterVideo rows stuck on the placeholder
``legacy/clip/{id}/master.mp4`` key that was never uploaded to storage.

Symptom (the bug this fixes): every dispatch of an UploadJobV2 whose
master carries the placeholder dies in branding with
``StorageWriteError: download failed ... No such file or directory``
and burns its retries.

For each broken master:
  1. parse the clip id from the key → load the Clip
  2. resolve real bytes: clip.storage_key if set, else UPLOAD the
     clip's local rendered file under the SAME key shape (so the row's
     key becomes real)
  3. update the row: real r2_key + clean_master per KAIZER_CLEAN_MASTER
     (post-cutover V4 renders are clean → branding must apply the logo)
  4. report the affected upload_jobs_v2 so failed ones can be retried
     from the publish card (retry re-reserves credits correctly).

Run from KaizerBackend/:  python scripts/repair_placeholder_masters.py
Add --dry-run to preview.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

from sqlalchemy import text  # noqa: E402

import models  # noqa: E402
from database import SessionLocal  # noqa: E402
from pipeline_core.storage import get_storage_provider  # noqa: E402


def main() -> int:
    dry = "--dry-run" in sys.argv
    provider = get_storage_provider()
    clean_flag = (os.environ.get("KAIZER_CLEAN_MASTER", "0") or "0").strip() == "1"

    db = SessionLocal()
    repaired = skipped = failed = 0
    try:
        masters = (
            db.query(models.MasterVideo)
            .filter(models.MasterVideo.r2_key.like("legacy/clip/%"))
            .all()
        )
        print(f"placeholder-key masters found: {len(masters)} "
              f"(KAIZER_CLEAN_MASTER={'1' if clean_flag else '0'})")
        for m in masters:
            key = m.r2_key or ""
            try:
                if provider.exists(key):
                    print(f"  master {m.id}: key already real ({key}) — skip")
                    skipped += 1
                    continue
            except Exception:
                pass  # treat as missing

            mt = re.match(r"legacy/clip/(\d+)/master\.mp4", key)
            clip = (
                db.query(models.Clip)
                .filter(models.Clip.id == int(mt.group(1)))
                .first()
            ) if mt else None
            if clip is None:
                print(f"  master {m.id}: no clip behind {key!r} — cannot repair")
                failed += 1
                continue

            new_key = (clip.storage_key or "").strip()
            if not new_key:
                fp = (clip.file_path or "").strip()
                if not (fp and os.path.isfile(fp)):
                    print(f"  master {m.id} (clip {clip.id}): no storage key "
                          f"and no local file at {fp!r} — cannot repair "
                          f"(re-render needed)")
                    failed += 1
                    continue
                if dry:
                    print(f"  master {m.id} (clip {clip.id}): [dry-run] would "
                          f"upload {fp} -> {key}")
                    repaired += 1
                    continue
                stored = provider.upload(fp, key, content_type="video/mp4")
                new_key = stored.key
                clip.storage_key = stored.key
                clip.storage_url = stored.url
                db.add(clip)

            if dry:
                print(f"  master {m.id} (clip {clip.id}): [dry-run] would set "
                      f"r2_key={new_key!r} clean_master={clean_flag}")
                repaired += 1
                continue

            m.r2_key = new_key
            m.clean_master = bool(clean_flag)
            m.pipeline_version = "v4_clean" if clean_flag else m.pipeline_version
            m.status = "ready"
            db.add(m)
            db.commit()
            print(f"  master {m.id} (clip {clip.id}): repaired -> {new_key} "
                  f"(clean_master={clean_flag})")
            repaired += 1

            # Surface the jobs that were burning retries on this master.
            rows = db.execute(text(
                "SELECT uj.id, uj.status, uj.attempts "
                "FROM upload_jobs_v2 uj "
                "JOIN publish_tasks pt ON pt.id = uj.publish_task_id "
                "WHERE pt.master_video_id = :mid"
            ), {"mid": m.id}).fetchall()
            for r in rows:
                hint = ("will succeed on its next retry"
                        if r[1] == "queued"
                        else "hit Retry on the publish card"
                        if r[1] in ("failed", "cancelled")
                        else r[1])
                print(f"      upload_job {r[0]}: status={r[1]} "
                      f"attempts={r[2]} -> {hint}")

        print(f"\nrepaired={repaired} skipped={skipped} unrepairable={failed}")
        return 0 if failed == 0 else 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
