"""
kaizer.routers.feedback
========================
GET /api/uploads/{upload_job_id}/feedback — post-publish feedback report.

Returns a `FeedbackReport`-shaped JSON payload produced by
`pipeline_core.feedback_loop.generate_feedback_report`. This is the
compounding-data moat endpoint: creators get a concrete next-render
plan based on how their live video is actually retaining.

Usage
-----
    GET /api/uploads/42/feedback
    → {
        "upload_job_id": 42,
        "status": "ready" | "no_analytics" | "upload_not_found",
        "retention_curve": [{"t_pct": 0, "retention_pct": 100}, ...],
        "dropoffs": [{"t_pct": 25, "drop_pct": 18, "severity": "major",
                       "likely_causes": [...]}, ...],
        "recommendations": [{"kind": "hook", "message": "...", "actionable": true}, ...],
        "explainability": {"hook_score": 0.72, "completion_score": 0.41, ...},
        "warnings": [...]
      }

Query params
------------
  youtube_api_key  : str — override server-side key for debugging (optional).

Authentication
--------------
Intentionally unauthenticated for v1; upload job IDs are opaque integers.
Add auth.current_user as a Depends when the security posture changes.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import SessionLocal
from pipeline_core import feedback_loop

router = APIRouter(prefix="/api/uploads", tags=["feedback"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Pydantic response schemas ─────────────────────────────────────────────────


class RetentionSampleSchema(BaseModel):
    t_pct: float
    retention_pct: float


class DropoffEventSchema(BaseModel):
    t_pct: float
    drop_pct: float
    severity: str
    likely_causes: list[str]


class RecommendationSchema(BaseModel):
    kind: str
    message: str
    actionable: bool


class FeedbackResponse(BaseModel):
    upload_job_id: int
    status: str
    retention_curve: list[RetentionSampleSchema]
    dropoffs: list[DropoffEventSchema]
    recommendations: list[RecommendationSchema]
    explainability: dict
    warnings: list[str]


# ── Route ─────────────────────────────────────────────────────────────────────


@router.get("/{upload_job_id}/feedback", response_model=FeedbackResponse)
def get_upload_feedback(
    upload_job_id: int,
    db: Session = Depends(get_db),
    youtube_api_key: Optional[str] = Query(
        default=None,
        description="Optional override. Defaults to YOUTUBE_ANALYTICS_API_KEY env var.",
    ),
):
    """Produce a feedback report for the upload. 404 when upload doesn't exist."""
    key = youtube_api_key or os.environ.get("YOUTUBE_ANALYTICS_API_KEY") or None

    report = feedback_loop.generate_feedback_report(
        upload_job_id=upload_job_id,
        db=db,
        youtube_api_key=key,
    )
    if report.status == "upload_not_found":
        raise HTTPException(404, f"UploadJob {upload_job_id} not found")

    return FeedbackResponse(
        upload_job_id=report.upload_job_id,
        status=report.status,
        retention_curve=[
            RetentionSampleSchema(t_pct=s.t_pct, retention_pct=s.retention_pct)
            for s in report.retention_curve
        ],
        dropoffs=[
            DropoffEventSchema(
                t_pct=d.t_pct, drop_pct=d.drop_pct, severity=d.severity,
                likely_causes=list(d.likely_causes),
            )
            for d in report.dropoffs
        ],
        recommendations=[
            RecommendationSchema(kind=r.kind, message=r.message, actionable=r.actionable)
            for r in report.recommendations
        ],
        explainability=report.explainability,
        warnings=report.warnings,
    )
