"""
kaizer.routers.job_progress
===========================
GET /api/jobs/{job_id}/progress — real-time pipeline progress endpoint.

Parses the job's log text for known stage-marker strings and maps them to a
progress percentage.  No additional DB columns are needed; the pipeline already
prints stage-marker lines to job.log.

Usage
-----
    GET /api/jobs/42/progress
    → {
        "job_id": 42,
        "status": "running",
        "stage": "Transcribing",
        "percent": 15.0,
        "elapsed_s": 38,
        "eta_s": 215,
        "message": "Transcribing audio with Whisper..."
      }

ProgressResponse fields
-----------------------
  job_id    : int          — The job identifier.
  status    : str          — pending | running | done | failed
  stage     : str          — Human-readable stage name (parsed from log).
  percent   : float        — 0.0 – 100.0  estimated pipeline progress.
  elapsed_s : int          — Wall-clock seconds since job.started_at.
  eta_s     : int | None   — Estimated remaining seconds, or None if unknown.
  message   : str          — Last meaningful log line.

Stage heuristics
----------------
Known stage markers and their approximate progress weights are listed in
STAGE_PROGRESS below.  The log is scanned line-by-line; the LAST line that
contains any marker string (case-insensitive substring) wins.

ETA calculation
---------------
  eta_s = (100 - percent) / percent * elapsed_s   when percent > 5
  eta_s = None                                     otherwise

Authentication note
-------------------
This endpoint intentionally does NOT require authentication so that a
lightweight polling client (e.g. a job-status widget) can hit it without
a token.  Job IDs are opaque integers — knowing an ID gives no useful
information beyond progress.  Add auth.current_user dependency here if the
project's security requirements change.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db
import models

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


# ── Stage progress map ────────────────────────────────────────────────────────

# Each entry: (marker_substring, progress_pct).  The LAST matching line in
# job.log determines the current stage.  Markers are matched case-insensitively.
STAGE_PROGRESS: list[tuple[str, float]] = [
    ("Validating input",      2.0),
    ("Extracting audio",      5.0),
    ("Transcribing",         15.0),
    ("Detecting clips",      25.0),
    ("Cutting clip",         35.0),   # may appear multiple times — use last
    ("Generating SEO",       55.0),
    ("Composing",            65.0),   # compose_clip / compose_follow_bar / compose_split_frame
    ("Rendering thumbnail",  85.0),
    ("Running QA",           92.0),
    ("done",                100.0),
]

# Stage name to return when a marker is matched (parallel list — same index)
_STAGE_NAMES: list[str] = [
    "Validating input",
    "Extracting audio",
    "Transcribing",
    "Detecting clips",
    "Cutting clip",
    "Generating SEO",
    "Composing",
    "Rendering thumbnail",
    "Running QA",
    "Done",
]


# ── Response schema ───────────────────────────────────────────────────────────

class ProgressResponse(BaseModel):
    """Progress snapshot for a pipeline job.

    Attributes
    ----------
    job_id : int
        Unique identifier for the job.
    status : str
        Current lifecycle status: pending | running | done | failed.
    stage : str
        Human-readable stage name inferred from the job log.
    percent : float
        Estimated pipeline progress in the range [0.0, 100.0].
    elapsed_s : int
        Seconds elapsed since the job started (0 if not yet started).
    eta_s : int | None
        Estimated remaining seconds, or None when the estimate is unreliable
        (e.g. when percent ≤ 5, or when the job has not started).
    message : str
        The last non-empty log line that does not look like a debug trace.
    """

    job_id: int
    status: str
    stage: str
    percent: float
    elapsed_s: int
    eta_s: Optional[int]
    message: str


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_stage(log: str, status: str) -> tuple[str, float]:
    """Scan log for the last stage marker and return (stage_name, percent).

    Parameters
    ----------
    log : str
        Full text of job.log.
    status : str
        Current job status from the DB.

    Returns
    -------
    (stage_name, percent)
    """
    if status == "done":
        return ("Done", 100.0)

    if status == "failed":
        # Return the last known progress before failure.
        # Fall through to the scan so we get the most recent marker.
        pass

    best_stage = "Starting"
    best_pct = 1.0

    if not log:
        return (best_stage, best_pct)

    lines = log.splitlines()
    for line in lines:
        line_lower = line.lower()
        for idx, (marker, pct) in enumerate(STAGE_PROGRESS):
            if marker.lower() in line_lower:
                best_stage = _STAGE_NAMES[idx]
                best_pct = pct
                # Do NOT break — we want the LAST match

    if status == "failed":
        return ("Failed", best_pct)

    return (best_stage, best_pct)


def _last_message(log: str) -> str:
    """Return the last non-empty, non-debug line from the log.

    Heuristic: skip lines that start with common debug prefixes like
    'DEBUG', '[debug]', or are just whitespace.
    """
    if not log:
        return ""
    lines = log.splitlines()
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith("debug") or lower.startswith("[debug]"):
            continue
        return stripped[:200]  # cap length for API response
    return ""


def _elapsed_seconds(job: "models.Job") -> int:
    """Compute wall-clock elapsed seconds since job.started_at.

    Returns 0 if the job has not started yet.
    """
    start = job.started_at
    if start is None:
        return 0
    # SQLite stores naive UTC datetimes; treat as UTC.
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    end = job.finished_at
    if end is not None:
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
    else:
        end = datetime.now(timezone.utc)
    try:
        return max(0, int((end - start).total_seconds()))
    except Exception:
        return 0


def _compute_eta(percent: float, elapsed_s: int) -> Optional[int]:
    """Estimate remaining seconds.

    Formula: (100 - percent) / percent * elapsed_s
    Returns None when percent ≤ 5 (too early to be reliable) or elapsed_s == 0.
    """
    if percent <= 5.0 or elapsed_s <= 0:
        return None
    try:
        remaining = (100.0 - percent) / percent * elapsed_s
        return max(0, int(remaining))
    except ZeroDivisionError:
        return None


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.get("/{job_id}/progress", response_model=ProgressResponse)
def get_job_progress(
    job_id: int,
    db: Session = Depends(get_db),
) -> ProgressResponse:
    """Return real-time progress for a pipeline job.

    Parses job.log for stage-marker strings to estimate percentage completion
    without requiring any additional DB columns.

    Parameters
    ----------
    job_id : int
        Primary key of the job row.

    Returns
    -------
    ProgressResponse

    Raises
    ------
    HTTPException(404)
        If no job with the given id exists.
    """
    job: Optional[models.Job] = (
        db.query(models.Job).filter(models.Job.id == job_id).first()
    )
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    status = job.status or "pending"
    log = job.log or ""

    stage_name, percent = _parse_stage(log, status)
    elapsed_s = _elapsed_seconds(job)
    eta_s = _compute_eta(percent, elapsed_s)
    message = _last_message(log)

    return ProgressResponse(
        job_id=job_id,
        status=status,
        stage=stage_name,
        percent=round(percent, 1),
        elapsed_s=elapsed_s,
        eta_s=eta_s,
        message=message,
    )
