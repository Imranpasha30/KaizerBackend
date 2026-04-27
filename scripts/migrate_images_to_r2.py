"""One-shot backfill: upload every local image referenced in the DB to R2.

Run from the host where the historical files actually exist on disk
(typically the user's local Windows box — Railway's container is
ephemeral and lost the originals already).

Usage::

    cd kaizer/KaizerBackend
    "<venv>/Scripts/python.exe" scripts/migrate_images_to_r2.py

Idempotent: rows whose ``storage_backend`` is already ``"r2"`` are
skipped.  Re-running picks up any rows that were missed (e.g. files
that were temporarily offline).

Behaviour
---------
1. Pre-step DDL — ``ALTER TABLE … ADD COLUMN IF NOT EXISTS …`` for the
   four new columns (``user_assets.thumb_storage_url``,
   ``clips.thumb_storage_url``, ``clips.image_storage_url``).  Required
   when running against a Postgres install that already has the
   tables; SQLAlchemy's ``Base.metadata.create_all`` adds tables but
   never adds columns.

2. Loop ``UserAsset`` rows where ``storage_backend != 'r2'``: upload
   ``file_path`` (and ``thumb_path`` if present), populate columns,
   commit per row.

3. Loop ``Clip`` rows where ``thumb_path`` or ``image_path`` is set
   and the corresponding ``*_storage_url`` is empty.  Upload, populate,
   commit.

4. Print ``(uploaded, skipped_already_r2, skipped_missing, errors)``
   summary.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the backend package importable when run as a script
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from sqlalchemy import text  # noqa: E402

import models  # noqa: E402
from database import SessionLocal, engine  # noqa: E402
from pipeline_core.storage import get_storage_provider  # noqa: E402


def _alter_add_columns(session) -> None:
    """Idempotent DDL — adds the four storage columns if missing.

    Postgres ``ADD COLUMN IF NOT EXISTS`` is supported on 9.6+.  SQLite
    doesn't support it, but SQLite installs are dev-only and we control
    the schema there via metadata.create_all so this DDL is a no-op
    target.
    """
    dialect = engine.dialect.name
    if dialect == "sqlite":
        print("[migrate] SQLite detected — skipping ALTER (create_all handles it).")
        return

    stmts = [
        "ALTER TABLE user_assets ADD COLUMN IF NOT EXISTS thumb_storage_url VARCHAR(500) DEFAULT ''",
        "ALTER TABLE clips       ADD COLUMN IF NOT EXISTS thumb_storage_url VARCHAR(500) DEFAULT ''",
        "ALTER TABLE clips       ADD COLUMN IF NOT EXISTS image_storage_url VARCHAR(500) DEFAULT ''",
    ]
    for stmt in stmts:
        session.execute(text(stmt))
    session.commit()
    print("[migrate] DDL applied (4 ADD COLUMN IF NOT EXISTS).")


def _upload(storage, local_path: str, key: str, content_type: str) -> str:
    """Wrap storage.upload — returns the public URL.  Raises on failure."""
    obj = storage.upload(local_path, key, content_type=content_type)
    return obj.url


def _migrate_user_assets(session, storage) -> dict:
    counts = {"uploaded": 0, "skipped_already_r2": 0, "skipped_missing": 0, "errors": 0}
    rows = (
        session.query(models.UserAsset)
        .filter(models.UserAsset.kind != "folder_marker")
        .all()
    )
    for i, a in enumerate(rows, start=1):
        try:
            already = (a.storage_backend or "") == "r2"
            if already and a.storage_url:
                counts["skipped_already_r2"] += 1
                continue

            file_exists = a.file_path and Path(a.file_path).exists()
            if not file_exists:
                print(f"  [skip] asset id={a.id} file missing: {a.file_path!r}")
                counts["skipped_missing"] += 1
                continue

            rel_dir = f"user_assets/{a.user_id}/"
            mime = a.mime or "application/octet-stream"
            url = _upload(storage, a.file_path, rel_dir + Path(a.file_path).name, mime)
            a.storage_backend = "r2"
            a.storage_url = url
            a.storage_key = rel_dir + Path(a.file_path).name

            if a.thumb_path and Path(a.thumb_path).exists() and not a.thumb_storage_url:
                a.thumb_storage_url = _upload(
                    storage, a.thumb_path, rel_dir + Path(a.thumb_path).name, "image/jpeg",
                )

            session.commit()
            counts["uploaded"] += 1
            if i % 25 == 0:
                print(f"  …{i}/{len(rows)} user_assets processed")
        except Exception as exc:
            session.rollback()
            counts["errors"] += 1
            print(f"  [error] asset id={a.id}: {exc}")
    return counts


def _migrate_clips(session, storage) -> dict:
    counts = {"uploaded_thumb": 0, "uploaded_image": 0,
              "skipped_already_r2": 0, "skipped_missing": 0, "errors": 0}
    rows = session.query(models.Clip).all()
    for i, c in enumerate(rows, start=1):
        try:
            need_thumb = bool(c.thumb_path) and not c.thumb_storage_url
            need_image = bool(c.image_path) and not c.image_storage_url
            if not need_thumb and not need_image:
                counts["skipped_already_r2"] += 1
                continue

            if need_thumb:
                if c.thumb_path and Path(c.thumb_path).exists():
                    c.thumb_storage_url = _upload(
                        storage,
                        c.thumb_path,
                        f"clips/{c.id}/{Path(c.thumb_path).name}",
                        "image/jpeg",
                    )
                    counts["uploaded_thumb"] += 1
                else:
                    counts["skipped_missing"] += 1
                    print(f"  [skip] clip id={c.id} thumb missing: {c.thumb_path!r}")

            if need_image:
                if c.image_path and Path(c.image_path).exists():
                    # Best-effort content-type from extension
                    ext = Path(c.image_path).suffix.lower()
                    ct = {
                        ".png": "image/png", ".webp": "image/webp",
                        ".gif": "image/gif", ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                    }.get(ext, "image/jpeg")
                    c.image_storage_url = _upload(
                        storage,
                        c.image_path,
                        f"clips/{c.id}/{Path(c.image_path).name}",
                        ct,
                    )
                    counts["uploaded_image"] += 1
                else:
                    counts["skipped_missing"] += 1
                    print(f"  [skip] clip id={c.id} image missing: {c.image_path!r}")

            session.commit()
            if i % 25 == 0:
                print(f"  …{i}/{len(rows)} clips processed")
        except Exception as exc:
            session.rollback()
            counts["errors"] += 1
            print(f"  [error] clip id={c.id}: {exc}")
    return counts


def main() -> None:
    if os.environ.get("STORAGE_BACKEND", "").lower() != "r2" and \
       not os.environ.get("R2_BUCKET"):
        print("WARNING: STORAGE_BACKEND is not 'r2' and R2_BUCKET is unset.")
        print("         Make sure R2_* env vars are configured before running.")

    storage = get_storage_provider("r2")
    print(f"[migrate] using R2 bucket={getattr(storage, 'bucket', '?')!r}")

    session = SessionLocal()
    try:
        _alter_add_columns(session)

        print("\n[migrate] === user_assets ===")
        a_counts = _migrate_user_assets(session, storage)
        print(f"  user_assets: {a_counts}")

        print("\n[migrate] === clips ===")
        c_counts = _migrate_clips(session, storage)
        print(f"  clips: {c_counts}")

        print("\n[migrate] DONE.")
        print(f"  user_assets summary: {a_counts}")
        print(f"  clips summary:       {c_counts}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
