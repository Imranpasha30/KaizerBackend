"""Tests for pipeline_core.feedback_loop — post-publish feedback loop."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import models
from database import Base
from pipeline_core import feedback_loop as fl


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


def _seed_upload_job(db, *, upload_id=1, user_id=1, channel_id=1, clip_id=1,
                     publish_kind="short", narrative_meta=None, seo_score=None):
    """Seed user/channel/job/clip/upload with optional narrative + SEO meta."""
    u = models.User(id=user_id, email=f"u{user_id}@x.com", name=f"user{user_id}")
    db.add(u)
    c = models.Channel(id=channel_id, user_id=user_id, name=f"ch{channel_id}")
    db.add(c)
    job = models.Job(id=upload_id + 100, user_id=user_id, status="done",
                     platform="youtube_short", video_name="test")
    db.add(job)
    db.flush()
    clip = models.Clip(
        id=clip_id, job_id=job.id, clip_index=0,
        duration=45.0,
        meta=json.dumps(narrative_meta) if narrative_meta else "{}",
        seo=json.dumps({"score": seo_score}) if seo_score is not None else "",
    )
    db.add(clip)
    db.flush()
    up = models.UploadJob(
        id=upload_id, user_id=user_id, clip_id=clip_id, channel_id=channel_id,
        status="done", publish_kind=publish_kind, title="t",
        video_id="vid123",
    )
    db.add(up)
    db.commit()
    return up


# ══════════════════════════════════════════════════════════════════════════════
# 1. Dataclass shapes
# ══════════════════════════════════════════════════════════════════════════════


def test_retention_sample_dataclass():
    s = fl.RetentionSample(t_pct=10.0, retention_pct=85.0)
    assert s.t_pct == 10.0 and s.retention_pct == 85.0


def test_dropoff_event_dataclass():
    d = fl.DropoffEvent(t_pct=20, drop_pct=12, severity="minor")
    assert d.likely_causes == []


def test_recommendation_dataclass():
    r = fl.Recommendation(kind="hook", message="fix")
    assert r.actionable is True


def test_feedback_report_dataclass_shape():
    r = fl.FeedbackReport(
        upload_job_id=1, status="no_analytics",
        retention_curve=[], dropoffs=[], recommendations=[],
        explainability={},
    )
    assert r.warnings == []


# ══════════════════════════════════════════════════════════════════════════════
# 2. fetch_retention stubs
# ══════════════════════════════════════════════════════════════════════════════


def test_fetch_youtube_no_key_returns_empty_list():
    assert fl.fetch_retention_from_youtube("vid123", api_key=None) == []
    assert fl.fetch_retention_from_youtube("vid123", api_key="") == []


def test_fetch_youtube_with_key_returns_empty_v1_stub():
    """Spec: v1 returns [] even with a key + a warning log. Phase 4 wires real API."""
    result = fl.fetch_retention_from_youtube("vid123", api_key="fake-key")
    assert result == []


def test_fetch_instagram_no_token_returns_empty():
    assert fl.fetch_retention_from_instagram("media_1", access_token=None) == []


def test_fetch_instagram_with_token_returns_empty_v1_stub():
    assert fl.fetch_retention_from_instagram("media_1", access_token="fake") == []


# ══════════════════════════════════════════════════════════════════════════════
# 3. analyze_dropoff
# ══════════════════════════════════════════════════════════════════════════════


def test_analyze_dropoff_empty_curve_returns_empty():
    assert fl.analyze_dropoff([]) == []


def test_analyze_dropoff_single_sample_returns_empty():
    assert fl.analyze_dropoff([fl.RetentionSample(0, 100)]) == []


def test_analyze_dropoff_no_notable_drops():
    curve = [
        fl.RetentionSample(0, 100),
        fl.RetentionSample(25, 97),
        fl.RetentionSample(50, 95),
        fl.RetentionSample(75, 93),
        fl.RetentionSample(100, 90),
    ]
    # All drops are < 5% minor threshold
    assert fl.analyze_dropoff(curve) == []


def test_analyze_dropoff_minor_major_critical_severities():
    curve = [
        fl.RetentionSample(0, 100),
        fl.RetentionSample(10, 91),    # 9% drop → minor
        fl.RetentionSample(30, 70),    # 21% drop → major
        fl.RetentionSample(60, 30),    # 40% drop → critical
    ]
    events = fl.analyze_dropoff(curve)
    severities = [e.severity for e in events]
    assert severities == ["minor", "major", "critical"]


def test_analyze_dropoff_events_sorted_by_t_pct():
    curve = [
        fl.RetentionSample(0, 100),
        fl.RetentionSample(60, 60),   # 40 drop at 60%
        fl.RetentionSample(30, 90),   # 10 drop at 30% (between 0 and 60)
    ]
    events = fl.analyze_dropoff(curve)
    # Sorted by t_pct ascending
    t_pcts = [e.t_pct for e in events]
    assert t_pcts == sorted(t_pcts)


def test_analyze_dropoff_early_drop_tags_hook_cause():
    curve = [fl.RetentionSample(0, 100), fl.RetentionSample(5, 70)]
    events = fl.analyze_dropoff(curve)
    assert len(events) == 1
    causes_joined = " ".join(events[0].likely_causes)
    assert "hook" in causes_joined.lower()


def test_analyze_dropoff_late_drop_tags_cta_cause():
    curve = [fl.RetentionSample(0, 100), fl.RetentionSample(80, 60)]
    events = fl.analyze_dropoff(curve)
    assert len(events) == 1
    causes_joined = " ".join(events[0].likely_causes)
    assert "cta" in causes_joined.lower() or "payoff" in causes_joined.lower()


# ══════════════════════════════════════════════════════════════════════════════
# 4. explain_clip_scoring
# ══════════════════════════════════════════════════════════════════════════════


def test_explain_missing_clip_returns_empty_dict(db_session):
    assert fl.explain_clip_scoring(clip_id=999, db=db_session) == {}


def test_explain_pulls_narrative_meta_from_clip(db_session):
    _seed_upload_job(
        db_session, clip_id=7,
        narrative_meta={
            "narrative_role": "climax",
            "hook_score": 0.82,
            "completion_score": 0.71,
            "importance_score": 0.9,
            "composite_score": 0.81,
        },
        seo_score=92,
    )
    result = fl.explain_clip_scoring(clip_id=7, db=db_session)
    assert result["narrative_role"] == "climax"
    assert result["hook_score"] == 0.82
    assert result["completion_score"] == 0.71
    assert result["composite_score"] == 0.81
    assert result["seo_score"] == 92
    assert result["duration"] == 45.0


def test_explain_handles_missing_meta_fields(db_session):
    _seed_upload_job(db_session, clip_id=7, narrative_meta={})
    result = fl.explain_clip_scoring(clip_id=7, db=db_session)
    assert result["narrative_role"] is None
    assert result["hook_score"] is None


# ══════════════════════════════════════════════════════════════════════════════
# 5. generate_recommendations
# ══════════════════════════════════════════════════════════════════════════════


def test_recommendations_empty_dropoffs_returns_general_note():
    recs = fl.generate_recommendations(dropoffs=[], explainability={})
    assert len(recs) >= 1
    assert any(r.kind == "general" for r in recs)


def test_recommendations_early_drop_fires_hook_advice():
    drops = [fl.DropoffEvent(t_pct=5, drop_pct=30, severity="major",
                             likely_causes=["weak hook"])]
    recs = fl.generate_recommendations(dropoffs=drops,
                                        explainability={"hook_score": 0.4})
    kinds = [r.kind for r in recs]
    assert "hook" in kinds


def test_recommendations_mid_drop_fires_pacing_advice():
    drops = [fl.DropoffEvent(t_pct=35, drop_pct=20, severity="major",
                             likely_causes=["pacing"])]
    recs = fl.generate_recommendations(dropoffs=drops, explainability={})
    kinds = [r.kind for r in recs]
    assert "pacing" in kinds


def test_recommendations_late_drop_with_low_completion_fires_completion_advice():
    drops = [fl.DropoffEvent(t_pct=80, drop_pct=25, severity="major",
                             likely_causes=["cta"])]
    recs = fl.generate_recommendations(
        dropoffs=drops, explainability={"completion_score": 0.3},
    )
    kinds = [r.kind for r in recs]
    assert "completion" in kinds


def test_recommendations_late_drop_with_high_completion_fires_cta_advice():
    drops = [fl.DropoffEvent(t_pct=80, drop_pct=25, severity="major",
                             likely_causes=["cta"])]
    recs = fl.generate_recommendations(
        dropoffs=drops, explainability={"completion_score": 0.9},
    )
    kinds = [r.kind for r in recs]
    assert "cta" in kinds


# ══════════════════════════════════════════════════════════════════════════════
# 6. generate_feedback_report orchestrator
# ══════════════════════════════════════════════════════════════════════════════


def test_report_upload_not_found(db_session):
    r = fl.generate_feedback_report(upload_job_id=999, db=db_session)
    assert r.status == "upload_not_found"
    assert r.recommendations == []


def test_report_no_analytics_when_both_fetchers_return_empty(db_session):
    _seed_upload_job(db_session, upload_id=1, clip_id=1)
    r = fl.generate_feedback_report(upload_job_id=1, db=db_session)
    assert r.status == "no_analytics"
    assert r.retention_curve == []
    assert len(r.recommendations) >= 1


def test_report_ready_when_retention_override_supplied(db_session):
    _seed_upload_job(db_session, upload_id=1, clip_id=1)
    curve = [fl.RetentionSample(0, 100), fl.RetentionSample(50, 60)]
    r = fl.generate_feedback_report(
        upload_job_id=1, db=db_session, retention_override=curve,
    )
    assert r.status == "ready"
    assert len(r.retention_curve) == 2
    assert len(r.dropoffs) >= 1


def test_report_explainability_populated_from_narrative_meta(db_session):
    _seed_upload_job(
        db_session, upload_id=1, clip_id=1,
        narrative_meta={
            "narrative_role": "turn",
            "hook_score": 0.65,
            "completion_score": 0.55,
            "composite_score": 0.62,
        },
    )
    r = fl.generate_feedback_report(upload_job_id=1, db=db_session)
    assert r.explainability.get("narrative_role") == "turn"
    assert r.explainability.get("hook_score") == 0.65


def test_report_never_raises_on_broken_meta(db_session):
    """Garbage JSON in clip.meta must NOT propagate an exception."""
    u = models.User(id=1, email="u@x.com", name="u")
    db_session.add(u)
    c = models.Channel(id=1, user_id=1, name="c")
    db_session.add(c)
    job = models.Job(id=100, user_id=1, status="done",
                     platform="youtube_short", video_name="t")
    db_session.add(job)
    db_session.flush()
    clip = models.Clip(id=1, job_id=100, clip_index=0, meta="<<NOT JSON>>")
    db_session.add(clip)
    up = models.UploadJob(id=1, user_id=1, clip_id=1, channel_id=1,
                          status="done", publish_kind="short", title="t",
                          video_id="v")
    db_session.add(up)
    db_session.commit()

    # Does not raise
    r = fl.generate_feedback_report(upload_job_id=1, db=db_session)
    assert r.status == "no_analytics"
