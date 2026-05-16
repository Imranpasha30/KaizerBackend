"""One-shot storage promotion: local ->R2.

When dev work is done locally with ``STORAGE_BACKEND=local``, every
clip / thumbnail / image / user-asset lands on disk and the DB row
points at a ``/media/<key>`` URL served by FastAPI's static mount.
On launch day (or any time you flip ``STORAGE_BACKEND=r2``), the
NEW rows go to R2, but OLD rows still point at dead local URLs.

This module promotes those old rows:

  1. Walk every Clip + UserAsset row that has either
     ``storage_backend='local'`` OR a ``/media/`` URL on one of the
     auxiliary fields (``thumb_storage_url`` / ``image_storage_url``).
  2. For each row, upload the local file to R2 under the same key.
  3. Update the row to point at the R2 URL with ``storage_backend='r2'``.

Safety:
  - Idempotent. Skips rows already on R2.
  - Doesn't delete the local file. Re-running is harmless; rolling back
    is "flip STORAGE_BACKEND=local in env and the files are still there."
  - Per-row try/except. One broken row doesn't abort the migration.
  - ``dry_run=True`` (default) reports what WOULD happen without writing
    to either R2 or the DB. Always run dry-run first.

Where it's called from:
  - ``scripts/promote_storage_to_r2.py`` — CLI wrapper for ops.
  - ``POST /api/admin/storage/promote-to-r2`` — admin-gated HTTP
    endpoint backing the "Promote to R2" admin button.

Both call :func:`promote_local_to_r2` with a SQLAlchemy session.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger("kaizer.storage_migration")


# ── Result dataclasses ───────────────────────────────────────────
@dataclass
class TableResult:
    table: str
    scanned: int = 0
    migrated: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list = field(default_factory=list)   # [{row_id, field, error}]

    def to_dict(self) -> dict:
        return {
            "table": self.table,
            "scanned":  self.scanned,
            "migrated": self.migrated,
            "skipped":  self.skipped,
            "failed":   self.failed,
            "errors":   self.errors[:50],   # cap so HTTP response stays sane
        }


@dataclass
class MigrationReport:
    dry_run: bool
    tables: dict = field(default_factory=dict)   # {table_name: TableResult}
    totals: dict = field(default_factory=lambda: {
        "scanned": 0, "migrated": 0, "skipped": 0, "failed": 0,
    })

    def add(self, tr: TableResult) -> None:
        self.tables[tr.table] = tr
        self.totals["scanned"]  += tr.scanned
        self.totals["migrated"] += tr.migrated
        self.totals["skipped"]  += tr.skipped
        self.totals["failed"]   += tr.failed

    def to_dict(self) -> dict:
        return {
            "dry_run": self.dry_run,
            "tables":  {k: v.to_dict() for k, v in self.tables.items()},
            "totals":  self.totals,
        }


# ── Path resolution ─────────────────────────────────────────────
def _output_root() -> Path:
    """Where LocalStorage writes files. Mirrors the resolution rule in
    ``pipeline_core.storage.get_storage_provider``."""
    env_root = os.environ.get("KAIZER_OUTPUT_ROOT")
    if env_root:
        return Path(env_root)
    base_dir = Path(__file__).resolve().parent   # KaizerBackend/
    return base_dir / "output"


def _resolve_local_path(*, storage_url: str, storage_key: str,
                        file_path: str) -> Optional[Path]:
    """Find the actual file on disk for a row.

    Resolution order:
      1. ``file_path`` if it's absolute AND exists (legacy rows).
      2. ``storage_key`` joined with KAIZER_OUTPUT_ROOT.
      3. ``storage_url`` if it begins with ``/media/`` — strip prefix
         then join.

    Returns the Path if a real file is found, else None.
    """
    if file_path:
        p = Path(file_path)
        if p.is_absolute() and p.is_file():
            return p
    root = _output_root()
    if storage_key:
        candidate = root / storage_key.lstrip("/\\")
        if candidate.is_file():
            return candidate
    if storage_url and storage_url.startswith("/media/"):
        rel = storage_url[len("/media/"):]
        candidate = root / rel
        if candidate.is_file():
            return candidate
    return None


def _key_from_url(url: str) -> str:
    """Pull the storage key out of a ``/media/<key>`` URL."""
    if url.startswith("/media/"):
        return url[len("/media/"):]
    return ""


def _is_local_url(url: str) -> bool:
    return bool(url) and url.startswith("/media/")


# ── Per-row migration ───────────────────────────────────────────
def _upload_to_r2(local_path: Path, key: str, content_type: str = "video/mp4"):
    """Upload one local file to R2, return the stored URL."""
    from pipeline_core.storage import get_storage_provider
    r2 = get_storage_provider("r2")   # explicit, ignores STORAGE_BACKEND
    safe_key = key.replace("\\", "/").lstrip("/")
    # Idempotency — if the same key is already there, don't re-upload.
    # The URL still needs to be computed to update the DB row.
    if r2.exists(safe_key):
        return r2.get_url(safe_key, signed=False), safe_key, True   # already_there
    stored = r2.upload(str(local_path), safe_key, content_type=content_type)
    return stored.url, safe_key, False


def _guess_content_type(path: Path) -> str:
    import mimetypes
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


# ── Clip table migration ────────────────────────────────────────
def _migrate_clips(db: Session, *, dry_run: bool,
                   progress: Callable[[str], None]) -> TableResult:
    """Migrate every Clip row's primary video + thumb + image fields
    that still point at local storage."""
    import models
    tr = TableResult(table="clips")

    rows = (db.query(models.Clip)
              .filter((models.Clip.storage_backend == "local") |
                      (models.Clip.thumb_storage_url.like("/media/%")) |
                      (models.Clip.image_storage_url.like("/media/%")))
              .all())
    tr.scanned = len(rows)
    if tr.scanned == 0:
        progress("  [clips] no local rows to migrate")
        return tr

    for clip in rows:
        row_dirty = False
        # ── Primary video ───────────────────────────────────
        if (clip.storage_backend or "").lower() == "local":
            local = _resolve_local_path(
                storage_url=clip.storage_url or "",
                storage_key=clip.storage_key or "",
                file_path=clip.file_path or "",
            )
            if local is None:
                tr.failed += 1
                tr.errors.append({"row_id": clip.id, "field": "storage_url",
                                  "error": "local file not found"})
            else:
                key = clip.storage_key or _key_from_url(clip.storage_url or "") \
                      or f"clips/{clip.id}/{local.name}"
                try:
                    if not dry_run:
                        new_url, new_key, already = _upload_to_r2(
                            local, key, content_type="video/mp4")
                        clip.storage_url     = new_url
                        clip.storage_key     = new_key
                        clip.storage_backend = "r2"
                        row_dirty = True
                    tr.migrated += 1
                    progress(f"  [clips#{clip.id}] video ->r2 ({key})")
                except Exception as exc:
                    tr.failed += 1
                    tr.errors.append({"row_id": clip.id, "field": "storage_url",
                                      "error": str(exc)})
                    logger.exception("clip %d video upload failed", clip.id)
        # ── Thumbnail (auxiliary URL, no separate backend col) ─────
        if _is_local_url(clip.thumb_storage_url or ""):
            local = _resolve_local_path(
                storage_url=clip.thumb_storage_url,
                storage_key="",
                file_path=clip.thumb_path or "",
            )
            if local is None:
                tr.failed += 1
                tr.errors.append({"row_id": clip.id, "field": "thumb_storage_url",
                                  "error": "local file not found"})
            else:
                key = _key_from_url(clip.thumb_storage_url) \
                      or f"clips/{clip.id}/{local.name}"
                try:
                    if not dry_run:
                        new_url, _k, _a = _upload_to_r2(local, key,
                                                       content_type="image/jpeg")
                        clip.thumb_storage_url = new_url
                        row_dirty = True
                    tr.migrated += 1
                    progress(f"  [clips#{clip.id}] thumb ->r2 ({key})")
                except Exception as exc:
                    tr.failed += 1
                    tr.errors.append({"row_id": clip.id, "field": "thumb_storage_url",
                                      "error": str(exc)})
        # ── Editorial image ─────────────────────────────────
        if _is_local_url(clip.image_storage_url or ""):
            local = _resolve_local_path(
                storage_url=clip.image_storage_url,
                storage_key="",
                file_path=clip.image_path or "",
            )
            if local is None:
                tr.failed += 1
                tr.errors.append({"row_id": clip.id, "field": "image_storage_url",
                                  "error": "local file not found"})
            else:
                key = _key_from_url(clip.image_storage_url) \
                      or f"clips/{clip.id}/{local.name}"
                try:
                    if not dry_run:
                        new_url, _k, _a = _upload_to_r2(local, key,
                                                       content_type=_guess_content_type(local))
                        clip.image_storage_url = new_url
                        row_dirty = True
                    tr.migrated += 1
                    progress(f"  [clips#{clip.id}] image ->r2 ({key})")
                except Exception as exc:
                    tr.failed += 1
                    tr.errors.append({"row_id": clip.id, "field": "image_storage_url",
                                      "error": str(exc)})

        if row_dirty and not dry_run:
            try:
                db.commit()
            except Exception as commit_exc:
                db.rollback()
                tr.failed += 1
                tr.errors.append({"row_id": clip.id, "field": "commit",
                                  "error": str(commit_exc)})

    # If we never had a dirty row, no commit happens — that's fine.
    return tr


# ── UserAsset migration ─────────────────────────────────────────
def _migrate_user_assets(db: Session, *, dry_run: bool,
                         progress: Callable[[str], None]) -> TableResult:
    import models
    tr = TableResult(table="user_assets")
    rows = (db.query(models.UserAsset)
              .filter((models.UserAsset.storage_backend == "local") |
                      (models.UserAsset.thumb_storage_url.like("/media/%")))
              .all())
    tr.scanned = len(rows)
    if tr.scanned == 0:
        progress("  [user_assets] no local rows to migrate")
        return tr

    for asset in rows:
        row_dirty = False
        # ── Primary ────────────────────────────────────────
        if (asset.storage_backend or "").lower() == "local":
            local = _resolve_local_path(
                storage_url=asset.storage_url or "",
                storage_key=asset.storage_key or "",
                file_path=asset.file_path or "",
            )
            if local is None:
                tr.failed += 1
                tr.errors.append({"row_id": asset.id, "field": "storage_url",
                                  "error": "local file not found"})
            else:
                key = asset.storage_key or _key_from_url(asset.storage_url or "") \
                      or f"user_assets/{asset.user_id}/{local.name}"
                try:
                    if not dry_run:
                        new_url, new_key, _a = _upload_to_r2(
                            local, key, content_type=_guess_content_type(local))
                        asset.storage_url     = new_url
                        asset.storage_key     = new_key
                        asset.storage_backend = "r2"
                        row_dirty = True
                    tr.migrated += 1
                    progress(f"  [user_assets#{asset.id}] primary ->r2 ({key})")
                except Exception as exc:
                    tr.failed += 1
                    tr.errors.append({"row_id": asset.id, "field": "storage_url",
                                      "error": str(exc)})
                    logger.exception("user_asset %d upload failed", asset.id)
        # ── Thumbnail ─────────────────────────────────────
        if _is_local_url(asset.thumb_storage_url or ""):
            local = _resolve_local_path(
                storage_url=asset.thumb_storage_url,
                storage_key="",
                file_path=asset.thumb_path or "",
            )
            if local is None:
                tr.failed += 1
                tr.errors.append({"row_id": asset.id, "field": "thumb_storage_url",
                                  "error": "local file not found"})
            else:
                key = _key_from_url(asset.thumb_storage_url) \
                      or f"user_assets/{asset.user_id}/{local.name}"
                try:
                    if not dry_run:
                        new_url, _k, _a = _upload_to_r2(local, key,
                                                       content_type="image/jpeg")
                        asset.thumb_storage_url = new_url
                        row_dirty = True
                    tr.migrated += 1
                    progress(f"  [user_assets#{asset.id}] thumb ->r2 ({key})")
                except Exception as exc:
                    tr.failed += 1
                    tr.errors.append({"row_id": asset.id, "field": "thumb_storage_url",
                                      "error": str(exc)})

        if row_dirty and not dry_run:
            try:
                db.commit()
            except Exception as commit_exc:
                db.rollback()
                tr.failed += 1
                tr.errors.append({"row_id": asset.id, "field": "commit",
                                  "error": str(commit_exc)})

    return tr


# ── Public entrypoint ───────────────────────────────────────────
def promote_local_to_r2(
    db: Session,
    *,
    dry_run: bool = True,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Promote every row that still points at local storage to R2.

    Pre-conditions:
      - R2 environment variables are set (``R2_BUCKET``, ``R2_ACCOUNT_ID``,
        ``R2_ACCESS_KEY_ID``, ``R2_SECRET_ACCESS_KEY``). If they're not
        the first upload throws and the whole report comes back with
        every row in ``failed``.
      - The local files referenced by the DB rows must still exist at
        ``KAIZER_OUTPUT_ROOT/<key>`` or wherever ``file_path`` points.
        Missing files are counted under ``failed`` with a clear error.

    Args:
        db: SQLAlchemy session — caller owns lifetime / close().
        dry_run: When True (default) NOTHING is uploaded or written;
            the report reflects what WOULD happen.
        progress_cb: Optional callback for per-row progress lines
            (stream to admin UI, CLI stdout, etc.).

    Returns:
        Structured report — see :class:`MigrationReport`.
    """
    progress = progress_cb or (lambda _msg: None)
    progress(f"=== Local ->R2 storage promotion ({'DRY-RUN' if dry_run else 'LIVE'}) ===")

    report = MigrationReport(dry_run=dry_run)
    try:
        # Pre-flight: try to instantiate the R2 provider — fails fast with
        # a clear message when R2 env vars are missing.
        from pipeline_core.storage import get_storage_provider
        get_storage_provider("r2")   # raises if not configured
    except Exception as exc:
        progress(f"PREFLIGHT FAILED: {exc}")
        report.tables["preflight"] = TableResult(
            table="preflight", failed=1,
            errors=[{"row_id": 0, "field": "r2_config", "error": str(exc)}],
        )
        report.totals["failed"] += 1
        return report.to_dict()

    report.add(_migrate_clips(db,       dry_run=dry_run, progress=progress))
    report.add(_migrate_user_assets(db, dry_run=dry_run, progress=progress))

    progress(f"=== Done. scanned={report.totals['scanned']} "
             f"migrated={report.totals['migrated']} "
             f"skipped={report.totals['skipped']} "
             f"failed={report.totals['failed']} ===")
    return report.to_dict()
