"""
kaizer.pipeline.feedback_loop
==============================
Post-publish feedback loop — the compounding-data moat.

Seven days after a clip is published, this module pulls its retention
curve (YouTube Analytics / Instagram Insights when wired; empty stub
otherwise), detects where viewers dropped off, cross-references the
saved Narrative Engine metadata, and produces concrete recommendations
for the NEXT render — not just an analytics dashboard.

Every clip that ships through this loop improves the next one's
selection. Nobody else in the SaaS clipping category closes this loop
— they stop at "here are your analytics".

Usage
-----
    from pipeline_core.feedback_loop import generate_feedback_report

    report = generate_feedback_report(
        upload_job_id=42,
        db=db,
        youtube_api_key=None,   # when None, retention_curve will be empty + stub warning
    )
    for rec in report.recommendations:
        print(rec.kind, "→", rec.message)

FeedbackReport fields
---------------------
    upload_job_id     : int
    status            : str            — 'ready' | 'no_analytics' | 'upload_not_found'
    retention_curve   : list[RetentionSample]
    dropoffs          : list[DropoffEvent]
    recommendations   : list[Recommendation]
    explainability    : dict           — narrative scoring breakdown for the clip
    warnings          : list[str]

Intentional v1 stubs
---------------------
  * fetch_retention_from_youtube: when no API key → returns [] + warning;
    real implementation (OAuth + Analytics v2 call) is a Phase 4 task.
  * Instagram Insights: not implemented — same stub shape; any caller
    wanting IG analytics should pass the curve directly.
  * No background scheduler here — the trigger (7-day-after-publish) is
    orchestrated elsewhere. This module is pure: given an upload_job_id
    and a db session, produce a report NOW.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger("kaizer.pipeline.feedback_loop")


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class RetentionSample:
    """One point on a retention curve.

    t_pct is the clip-timeline percentage (0.0-100.0), retention_pct is
    the fraction of viewers still watching at that point (0.0-100.0).
    """
    t_pct: float
    retention_pct: float


@dataclass
class DropoffEvent:
    """A notable retention drop between two consecutive RetentionSamples.

    severity:
      'minor'    — 5-15 pct drop
      'major'    — 15-35 pct drop
      'critical' — >35 pct drop (audience abandoned)
    """
    t_pct: float
    drop_pct: float
    severity: str
    likely_causes: list[str] = field(default_factory=list)


@dataclass
class Recommendation:
    """A concrete fix to apply to the NEXT render."""
    kind: str          # 'hook' | 'pacing' | 'completion' | 'cta' | 'originality' | 'general'
    message: str
    actionable: bool = True


@dataclass
class FeedbackReport:
    upload_job_id: int
    status: str
    retention_curve: list[RetentionSample]
    dropoffs: list[DropoffEvent]
    recommendations: list[Recommendation]
    explainability: dict
    warnings: list[str] = field(default_factory=list)


# ── Thresholds (tests monkey-patch these) ────────────────────────────────────

_DROP_MINOR = 5.0
_DROP_MAJOR = 15.0
_DROP_CRITICAL = 35.0

# Early = first 15% of the clip (i.e., the hook window for a 45-60s clip).
_EARLY_PCT = 15.0
_MID_PCT = 60.0


# ── Stub: YouTube / Instagram analytics fetchers ─────────────────────────────

def fetch_retention_from_youtube(
    video_id: str,
    *,
    api_key: Optional[str] = None,
) -> list[RetentionSample]:
    """Fetch per-percentile audience retention for a YouTube video.

    v1 stub — always returns [] when `api_key` is None/empty. A real
    implementation requires OAuth2 (YouTube Analytics API v2) + the
    channel's owner credentials, which we'll wire via the existing
    youtube/oauth.py helper in Phase 4.

    Never raises.
    """
    if not api_key:
        logger.info("fetch_retention_from_youtube: no api_key (stub mode) for %s", video_id)
        return []
    # Real implementation would go here. For v1 we keep the interface so
    # tests and callers don't branch.
    logger.warning(
        "fetch_retention_from_youtube: api_key supplied but live fetch is a "
        "Phase 4 task — returning empty curve for %s", video_id,
    )
    return []


def fetch_retention_from_instagram(
    media_id: str,
    *,
    access_token: Optional[str] = None,
) -> list[RetentionSample]:
    """Instagram Reels insights — same stub shape.

    v1 returns []. Meta Graph API's video retention insights require the
    business-account token flow (Phase 4).
    """
    if not access_token:
        logger.info(
            "fetch_retention_from_instagram: no access_token (stub mode) for %s",
            media_id,
        )
        return []
    logger.warning(
        "fetch_retention_from_instagram: live fetch is a Phase 4 task — empty for %s",
        media_id,
    )
    return []


# ── Drop-off analysis ────────────────────────────────────────────────────────

def analyze_dropoff(
    retention_curve: list[RetentionSample],
    *,
    minor_threshold: float = _DROP_MINOR,
    major_threshold: float = _DROP_MAJOR,
    critical_threshold: float = _DROP_CRITICAL,
) -> list[DropoffEvent]:
    """Detect notable drops between consecutive retention samples.

    A drop is the delta (prev.retention_pct - curr.retention_pct). We
    tag severity by magnitude and attach likely-causes based on where on
    the clip timeline the drop happened.

    Returns drops sorted by t_pct ascending. Empty input → [].
    """
    if len(retention_curve) < 2:
        return []

    # Ensure sorted by t_pct
    curve = sorted(retention_curve, key=lambda s: s.t_pct)

    events: list[DropoffEvent] = []
    for prev, curr in zip(curve[:-1], curve[1:]):
        drop = prev.retention_pct - curr.retention_pct
        if drop < minor_threshold:
            continue
        if drop >= critical_threshold:
            severity = "critical"
        elif drop >= major_threshold:
            severity = "major"
        else:
            severity = "minor"
        causes = _likely_causes_for(curr.t_pct, severity)
        events.append(DropoffEvent(
            t_pct=curr.t_pct, drop_pct=drop, severity=severity, likely_causes=causes,
        ))
    return events


def _likely_causes_for(t_pct: float, severity: str) -> list[str]:
    """Heuristic causes keyed on when the drop happened on the timeline."""
    causes: list[str] = []
    if t_pct <= _EARLY_PCT:
        causes.append("weak hook — first-3s failed to retain")
        if severity in ("major", "critical"):
            causes.append("face not in frame by ~1.5s")
            causes.append("caption invisible on mute (IG default)")
    elif t_pct <= _MID_PCT:
        causes.append("pacing dip — content lost momentum mid-clip")
        if severity in ("major", "critical"):
            causes.append("b-roll or cutaway broke narrative flow")
    else:  # late
        causes.append("CTA appeared too early or too long")
        if severity == "critical":
            causes.append("payoff underwhelming vs setup — expectation mismatch")
    return causes


# ── Explainability: pull narrative scoring from DB ───────────────────────────

def explain_clip_scoring(clip_id: int, db: Session) -> dict:
    """Load the Clip row's saved narrative/SEO metadata and return a
    human-readable scoring breakdown:

        {
          "narrative_role": str,        # from ClipCandidate.meta if saved
          "hook_score":  float,
          "completion_score": float,
          "importance_score": float,
          "composite_score": float,
          "seo_score": int,             # if captured at upload time
          "duration": float,
        }

    Missing fields degrade to None rather than raising.
    """
    try:
        import models  # lazy
    except ImportError:
        logger.warning("explain_clip_scoring: models module unavailable")
        return {}

    clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
    if clip is None:
        logger.info("explain_clip_scoring: clip %d not found", clip_id)
        return {}

    payload: dict = {
        "clip_id": clip.id,
        "duration": float(clip.duration or 0.0),
    }

    # Try to parse the narrative-engine-produced JSON from clip.meta
    try:
        meta_raw = clip.meta or "{}"
        parsed = json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
    except Exception:
        parsed = {}

    for key in (
        "narrative_role", "hook_score", "completion_score",
        "importance_score", "composite_score",
    ):
        payload[key] = parsed.get(key)

    # SEO score may live in clip.seo (JSON string) or clip_performance
    try:
        seo_raw = clip.seo or ""
        if seo_raw:
            seo = json.loads(seo_raw) if isinstance(seo_raw, str) else seo_raw
            payload["seo_score"] = seo.get("score") if isinstance(seo, dict) else None
    except Exception:
        payload["seo_score"] = None

    return payload


# ── Recommendation generation ────────────────────────────────────────────────

def generate_recommendations(
    dropoffs: list[DropoffEvent],
    explainability: dict,
    *,
    platform: str = "youtube_short",
) -> list[Recommendation]:
    """Turn drop-offs + saved scoring into concrete next-render advice.

    Always returns at least one general recommendation even when no
    drops detected (tells the creator what the data does and doesn't
    show).
    """
    recs: list[Recommendation] = []

    # Hook feedback
    early_drops = [d for d in dropoffs if d.t_pct <= _EARLY_PCT]
    if early_drops:
        worst = max(early_drops, key=lambda d: d.drop_pct)
        hook_score = explainability.get("hook_score")
        if hook_score is not None and hook_score < 0.6:
            msg = (f"Hook score was {hook_score:.2f}; {worst.drop_pct:.0f}% of viewers "
                   f"dropped by {worst.t_pct:.0f}%. Try: face-forward first frame, "
                   f"caption visible on mute by t=0.5s, question or number in opening line.")
        else:
            msg = (f"{worst.drop_pct:.0f}% dropped in the first {worst.t_pct:.0f}% "
                   "even though the hook scored OK. Consider reshooting the opener.")
        recs.append(Recommendation(kind="hook", message=msg, actionable=True))

    # Pacing feedback (mid-clip drops)
    mid_drops = [d for d in dropoffs if _EARLY_PCT < d.t_pct <= _MID_PCT]
    if mid_drops:
        worst = max(mid_drops, key=lambda d: d.drop_pct)
        recs.append(Recommendation(
            kind="pacing",
            message=(f"Mid-clip drop of {worst.drop_pct:.0f}% at {worst.t_pct:.0f}%. "
                     "Add a visual beat or on-screen question at this point in the "
                     "next render to re-hook attention."),
            actionable=True,
        ))

    # Completion feedback (late drops)
    late_drops = [d for d in dropoffs if d.t_pct > _MID_PCT]
    if late_drops:
        completion = explainability.get("completion_score")
        worst = max(late_drops, key=lambda d: d.drop_pct)
        if completion is not None and completion < 0.5:
            msg = (f"Completion score {completion:.2f} + {worst.drop_pct:.0f}% late drop. "
                   "The clip probably ended mid-sentence — rely more on the sentence-"
                   "boundary snap in clip_boundaries next time.")
            recs.append(Recommendation(kind="completion", message=msg, actionable=True))
        else:
            recs.append(Recommendation(
                kind="cta",
                message=(f"Late drop of {worst.drop_pct:.0f}% at {worst.t_pct:.0f}%. "
                         "Your CTA may have appeared too long or pointed viewers "
                         "off-platform too early."),
                actionable=True,
            ))

    # Platform-specific originality nudge (IG-only)
    if platform == "instagram_reel" and not dropoffs:
        recs.append(Recommendation(
            kind="originality",
            message=("Retention held, but Reels rewards seamless loops. Check the "
                     "next render's loop_score and aim for overall ≥60."),
            actionable=True,
        ))

    # Always include a general note when nothing specific fired
    if not recs:
        if dropoffs:
            recs.append(Recommendation(
                kind="general",
                message="Retention shape is healthy across the clip. Keep doing what you did.",
                actionable=False,
            ))
        else:
            recs.append(Recommendation(
                kind="general",
                message=("No retention data available yet — the feedback loop needs a "
                         "YouTube/Instagram API key to pull retention curves. Wire it "
                         "via settings to enable drop-off diagnostics."),
                actionable=False,
            ))

    return recs


# ── Orchestrator ─────────────────────────────────────────────────────────────

def generate_feedback_report(
    upload_job_id: int,
    db: Session,
    *,
    youtube_api_key: Optional[str] = None,
    instagram_access_token: Optional[str] = None,
    retention_override: Optional[list[RetentionSample]] = None,
) -> FeedbackReport:
    """Build a FeedbackReport for the given upload_job_id.

    Flow:
      1. Load UploadJob; if missing → status='upload_not_found'.
      2. If retention_override is provided, use that (bypass fetchers).
      3. Else fetch retention from YT or IG based on publish_kind.
         If both return [] → status='no_analytics'.
      4. analyze_dropoff + explain_clip_scoring + generate_recommendations.
      5. Return a FeedbackReport; never raise.
    """
    warnings: list[str] = []
    try:
        import models  # lazy
    except ImportError:
        logger.warning("generate_feedback_report: models import failed")
        return FeedbackReport(
            upload_job_id=upload_job_id, status="upload_not_found",
            retention_curve=[], dropoffs=[], recommendations=[],
            explainability={}, warnings=["models module unavailable"],
        )

    up = db.query(models.UploadJob).filter(models.UploadJob.id == upload_job_id).first()
    if up is None:
        return FeedbackReport(
            upload_job_id=upload_job_id, status="upload_not_found",
            retention_curve=[], dropoffs=[], recommendations=[],
            explainability={}, warnings=[f"upload_job {upload_job_id} not found"],
        )

    # Retention curve
    if retention_override is not None:
        curve = list(retention_override)
    else:
        if up.publish_kind == "short":
            curve = fetch_retention_from_youtube(up.video_id or "", api_key=youtube_api_key)
            if not curve:
                curve = fetch_retention_from_instagram(up.video_id or "",
                                                       access_token=instagram_access_token)
        else:
            curve = fetch_retention_from_youtube(up.video_id or "", api_key=youtube_api_key)

    status = "ready" if curve else "no_analytics"
    if not curve:
        warnings.append("No retention data available — recommendations will be generic.")

    # Drop-off + explainability + recs
    dropoffs = analyze_dropoff(curve)
    explain = explain_clip_scoring(up.clip_id, db) if up.clip_id else {}
    platform = "instagram_reel" if (up.publish_kind == "short"
                                    and (up.channel_id and False)) else "youtube_short"
    # NOTE: we cannot reliably derive IG vs YT from publish_kind alone; callers
    # that know they're on IG should pass retention_override + check platform in the
    # recommendation post-processing. For v1 default to youtube_short.
    recs = generate_recommendations(dropoffs, explain, platform=platform)

    return FeedbackReport(
        upload_job_id=upload_job_id,
        status=status,
        retention_curve=curve,
        dropoffs=dropoffs,
        recommendations=recs,
        explainability=explain,
        warnings=warnings,
    )
