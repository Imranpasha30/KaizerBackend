"""Fair render queue + disk/temp hygiene (Wave 4, item I).

Adds three nullable claim columns to the existing ``jobs`` table
(idempotent ALTER, see :func:`ensure_schema`) and a per-user
round-robin claim primitive so one user queueing 10 jobs can't starve
every other user's first job:

    rank = row_number() OVER (PARTITION BY user_id ORDER BY created_at)
    next = ORDER BY rank, created_at  -- every user's 1st queued job
                                      -- beats any user's 2nd

Claiming sets ``render_claimed_by`` + ``render_lease_expires_at``
(+10 min). It does NOT flip ``status`` — runner.run_pipeline owns the
status lifecycle exactly as before. An expired lease makes the row
claimable again (crash recovery).

The whole queue is opt-in: runner.py only consults it when
``KAIZER_FAIR_RENDER_QUEUE=1``. With the flag unset, nothing here is
on the render path (ensure_schema/sweep still run at startup — both
are idempotent and harmless).

Per-user cap: env ``KAIZER_RENDER_CAP_PER_USER`` (default 2).
TODO(plan-tiers): derive the cap from ``plan_tiers`` (e.g. a dedicated
render-cap column next to ``slot_cap_active_uploads``) once product
defines per-tier values; env-only for now per Wave 4 scope.

Lanes: ``render_lane`` is 'batch' (default) or 'interactive'.
NOTE (verified during Wave 4): the V4 editor re-render
(routers/v4_editor.trigger_render) does NOT go through the jobs queue —
it spawns an in-process daemon thread calling v1_bridge directly, so
nothing sets 'interactive' yet. :func:`set_lane` + the runner's
reserved-slot logic are in place for when that path migrates.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from sqlalchemy import text

LEASE_MINUTES = 10

_schema_lock = threading.Lock()
_schema_done = False

# SQLite has no SKIP LOCKED — serialize claims process-locally (dev is a
# single process; production runs Postgres).
_sqlite_claim_lock = threading.Lock()


def _engine():
    from database import engine
    return engine


def _is_postgres() -> bool:
    try:
        return _engine().dialect.name == "postgresql"
    except Exception:
        return False


def render_cap_per_user() -> int:
    """Max simultaneously-running jobs per user (claim eligibility gate).

    TODO(plan-tiers): wire to plan_tiers once per-tier render caps are
    defined; env-only for now.
    """
    try:
        return max(1, int(os.environ.get("KAIZER_RENDER_CAP_PER_USER", "2")))
    except ValueError:
        return 2


# ─── Schema ──────────────────────────────────────────────────────────

def ensure_schema() -> None:
    """Idempotent ALTERs adding the claim columns + a partial index on
    pending jobs. Called once from runner startup (NOT main.py). Safe to
    call repeatedly; safe on both Postgres and SQLite."""
    global _schema_done
    with _schema_lock:
        if _schema_done:
            return
        eng = _engine()
        if _is_postgres():
            stmts = [
                "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS "
                "render_lane VARCHAR(16) DEFAULT 'batch'",
                "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS "
                "render_claimed_by VARCHAR(64)",
                "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS "
                "render_lease_expires_at TIMESTAMPTZ",
                "CREATE INDEX IF NOT EXISTS ix_jobs_pending_render "
                "ON jobs (created_at) WHERE status = 'pending'",
            ]
            with eng.begin() as conn:
                for s in stmts:
                    conn.execute(text(s))
        else:
            with eng.begin() as conn:
                cols = {
                    row[1]
                    for row in conn.exec_driver_sql(
                        "PRAGMA table_info(jobs)"
                    ).fetchall()
                }
                if "render_lane" not in cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE jobs ADD COLUMN "
                        "render_lane VARCHAR(16) DEFAULT 'batch'"
                    )
                if "render_claimed_by" not in cols:
                    conn.exec_driver_sql(
                        "ALTER TABLE jobs ADD COLUMN render_claimed_by VARCHAR(64)"
                    )
                if "render_lease_expires_at" not in cols:
                    # SQLite: TIMESTAMP (no TZ type).
                    conn.exec_driver_sql(
                        "ALTER TABLE jobs ADD COLUMN render_lease_expires_at TIMESTAMP"
                    )
                conn.exec_driver_sql(
                    "CREATE INDEX IF NOT EXISTS ix_jobs_pending_render "
                    "ON jobs (created_at) WHERE status = 'pending'"
                )
        _schema_done = True


# ─── Claim primitive ─────────────────────────────────────────────────

def _head_sql(cand_clause: str, *, pg: bool) -> str:
    """Fair-order head query. Eligible rows: pending/queued, unclaimed
    (or lease expired), matching lane filter + candidate scope, owner
    under the running-jobs cap.

    NOTE: 'queued' is included because the legacy V1 path flips a job
    to 'queued' while it waits on the concurrency gate — those rows
    must stay visible to the fair ordering.
    """
    if pg:
        now = "now()"
        null_safe_eq = "r.user_id IS NOT DISTINCT FROM j.user_id"
    else:
        now = "datetime('now')"
        null_safe_eq = "r.user_id IS j.user_id"
    return f"""
        SELECT j.id
          FROM jobs j
          JOIN (
              SELECT id,
                     row_number() OVER (PARTITION BY user_id
                                        ORDER BY created_at, id) AS rk
                FROM jobs
               WHERE status IN ('pending', 'queued')
                 AND (render_claimed_by IS NULL
                      OR render_lease_expires_at IS NULL
                      OR render_lease_expires_at < {now})
                 AND (CAST(:lane AS VARCHAR) IS NULL
                      OR COALESCE(render_lane, 'batch') = CAST(:lane AS VARCHAR))
                 {cand_clause}
          ) t ON t.id = j.id
         WHERE (SELECT count(*) FROM jobs r
                 WHERE r.status = 'running'
                   AND {null_safe_eq}) < :cap
         ORDER BY t.rk, j.created_at, j.id
         LIMIT 1
    """


def claim_next_render(
    worker_id: str,
    lane_filter: Optional[str] = None,
    *,
    candidate_ids: Optional[list[int]] = None,
    only_job_id: Optional[int] = None,
    cap: Optional[int] = None,
) -> Optional[int]:
    """Claim the next renderable job in per-user round-robin order.

    Returns the claimed job id, or None when nothing is claimable (or
    when ``only_job_id`` is given and that job is not the fair-order
    head yet — the runner's per-job worker threads use this to wait
    their turn).

    ``candidate_ids`` scopes the ordering to jobs this process is
    actually able to run (per-job worker threads), so a zombie
    'pending' row left by a dead process can't block the queue.

    Claim = set ``render_claimed_by`` + ``render_lease_expires_at``
    (+10 min). Status is NOT touched.
    """
    cap = cap if cap is not None else render_cap_per_user()
    cand_clause = ""
    if candidate_ids is not None:
        ids = ",".join(str(int(i)) for i in candidate_ids)
        if not ids:
            return None
        cand_clause = f"AND id IN ({ids})"

    eng = _engine()
    params = {"lane": lane_filter, "cap": int(cap)}

    if _is_postgres():
        head_sql = _head_sql(cand_clause, pg=True)
        claim_sql = """
            WITH pick AS (
                SELECT id FROM jobs
                 WHERE id = :head
                   AND status IN ('pending', 'queued')
                   AND (render_claimed_by IS NULL
                        OR render_lease_expires_at IS NULL
                        OR render_lease_expires_at < now())
                   FOR UPDATE SKIP LOCKED
            )
            UPDATE jobs
               SET render_claimed_by = :wid,
                   render_lease_expires_at = now() + (:lease * interval '1 minute')
             WHERE id IN (SELECT id FROM pick)
            RETURNING id
        """
        with eng.begin() as conn:
            row = conn.execute(text(head_sql), params).first()
            if row is None:
                return None
            head = int(row[0])
            if only_job_id is not None and head != int(only_job_id):
                return None
            claimed = conn.execute(
                text(claim_sql),
                {"head": head, "wid": (worker_id or "")[:64],
                 "lease": LEASE_MINUTES},
            ).first()
            return int(claimed[0]) if claimed else None

    # SQLite (dev): no SKIP LOCKED — serialize within the process.
    with _sqlite_claim_lock:
        head_sql = _head_sql(cand_clause, pg=False)
        with eng.begin() as conn:
            row = conn.execute(text(head_sql), params).first()
            if row is None:
                return None
            head = int(row[0])
            if only_job_id is not None and head != int(only_job_id):
                return None
            res = conn.execute(
                text(
                    "UPDATE jobs SET render_claimed_by = :wid, "
                    "render_lease_expires_at = datetime('now', '+' || :lease || ' minutes') "
                    "WHERE id = :head AND status IN ('pending', 'queued') "
                    "AND (render_claimed_by IS NULL "
                    "     OR render_lease_expires_at IS NULL "
                    "     OR render_lease_expires_at < datetime('now'))"
                ),
                {"wid": (worker_id or "")[:64], "lease": LEASE_MINUTES,
                 "head": head},
            )
            return head if (res.rowcount or 0) > 0 else None


def release_claim(job_id: int) -> None:
    """Clear the claim columns (called when a render finishes or the
    worker abandons the job). Best-effort — an expired lease covers the
    failure case anyway."""
    try:
        with _engine().begin() as conn:
            conn.execute(
                text(
                    "UPDATE jobs SET render_claimed_by = NULL, "
                    "render_lease_expires_at = NULL WHERE id = :jid"
                ),
                {"jid": int(job_id)},
            )
    except Exception:
        pass


def set_lane(job_id: int, lane: str) -> None:
    """Stamp a job's render lane ('batch' | 'interactive').

    Reserved for the future: the V4 editor re-render currently bypasses
    the jobs queue entirely (in-process thread), so nothing calls this
    with 'interactive' yet. See module docstring."""
    lane = (lane or "batch").strip().lower()
    if lane not in ("batch", "interactive"):
        lane = "batch"
    try:
        with _engine().begin() as conn:
            conn.execute(
                text("UPDATE jobs SET render_lane = :lane WHERE id = :jid"),
                {"lane": lane, "jid": int(job_id)},
            )
    except Exception:
        pass


def queue_position(job_id: int) -> Optional[int]:
    """1-based position of ``job_id`` among pending jobs under the same
    per-user round-robin ordering the claimer uses. None when the job
    isn't pending (or on any query failure)."""
    sql = """
        SELECT pos FROM (
            SELECT j.id AS id,
                   row_number() OVER (ORDER BY t.rk, j.created_at, j.id) AS pos
              FROM jobs j
              JOIN (
                  SELECT id,
                         row_number() OVER (PARTITION BY user_id
                                            ORDER BY created_at, id) AS rk
                    FROM jobs
                   WHERE status IN ('pending', 'queued')
              ) t ON t.id = j.id
        ) z
        WHERE z.id = :jid
    """
    try:
        with _engine().connect() as conn:
            row = conn.execute(text(sql), {"jid": int(job_id)}).first()
        return int(row[0]) if row else None
    except Exception:
        return None


# ─── Disk guard + temp-dir sweeper ───────────────────────────────────

def min_free_disk_gb() -> float:
    try:
        return float(os.environ.get("KAIZER_MIN_FREE_DISK_GB", "10"))
    except ValueError:
        return 10.0


def free_disk_gb(path) -> float:
    """Free gigabytes on the volume holding ``path`` (walks up to the
    nearest existing ancestor so a not-yet-created output root works)."""
    p = Path(path)
    probe = p
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    usage = shutil.disk_usage(str(probe))
    return usage.free / (1024 ** 3)


def disk_guard_ok(output_root) -> tuple[bool, str]:
    """Returns ``(ok, message)``. ok=False means the render must not
    start — message is suitable for Job.error. A failed probe NEVER
    blocks a render (fail-open)."""
    try:
        free = free_disk_gb(output_root)
    except Exception as exc:
        return True, f"disk check skipped ({exc})"
    floor = min_free_disk_gb()
    if free < floor:
        return False, (
            f"Insufficient disk space to start render: {free:.1f} GB free "
            f"at {output_root} (minimum KAIZER_MIN_FREE_DISK_GB={floor:g}). "
            f"Free up disk space and re-submit the job."
        )
    return True, ""


# Storage dirs under the output root that hold DB-referenced media (masters,
# clips, logos, raw inputs) and must NEVER be swept as render scratch.
_RENDER_SCRATCH_PROTECT = {
    "legacy", "clips", "user_assets", "raw_uploads", "branded", "_gemini_cache",
}


def render_keep_last() -> int:
    """How many recent per-job render output dirs to retain per category.
    Env: ``KAIZER_RENDER_KEEP_LAST`` (default 10 — must exceed render
    concurrency so an in-flight job is never swept)."""
    try:
        return max(1, int(os.environ.get("KAIZER_RENDER_KEEP_LAST", "10")))
    except Exception:
        return 10


def sweep_old_render_outputs(output_root, keep: int | None = None) -> int:
    """Bound local render-scratch growth: within each render category dir
    under ``output_root`` (e.g. ``full_video_shorts_v4``), keep the most
    recent ``keep`` per-job subdirs (by mtime) and delete older ones.

    With STORAGE_BACKEND=r2 the finished master is already in R2, so the
    local per-job working dir is disposable once it ages out. DB-referenced
    storage dirs (``legacy``/``clips``/``user_assets``/``raw_uploads``/
    ``branded``) are protected and never touched. Keeping the most-recent K
    guarantees an in-progress job (newest mtime) is never swept. Best-effort;
    never raises. Returns the number of dirs removed."""
    keep = keep if keep is not None else render_keep_last()
    root = Path(output_root)
    if not root.is_dir():
        return 0
    removed = 0
    try:
        cats = [d for d in root.iterdir()
                if d.is_dir() and d.name not in _RENDER_SCRATCH_PROTECT]
    except OSError:
        return 0
    for cat in cats:
        try:
            jobs = [d for d in cat.iterdir() if d.is_dir()]
        except OSError:
            continue
        if len(jobs) <= keep:
            continue
        try:
            jobs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        except OSError:
            continue
        for d in jobs[keep:]:
            shutil.rmtree(d, ignore_errors=True)
            if not d.exists():
                removed += 1
    return removed


def sweep_orphaned_tempdirs(max_age_hours: int = 24) -> int:
    """Delete ``kaizer_*`` directories under the system temp dir that
    are older than ``max_age_hours`` (crashed renders leave these
    behind). Returns the number of directories removed. Best-effort —
    in-use files on Windows simply survive the sweep."""
    removed = 0
    cutoff = time.time() - float(max_age_hours) * 3600.0
    tmp = Path(tempfile.gettempdir())
    try:
        entries = list(tmp.glob("kaizer_*"))
    except OSError:
        return 0
    for d in entries:
        try:
            if not d.is_dir():
                continue
            if d.stat().st_mtime >= cutoff:
                continue
            shutil.rmtree(d, ignore_errors=True)
            if not d.exists():
                removed += 1
        except OSError:
            continue
    return removed
