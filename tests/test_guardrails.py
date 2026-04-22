"""Tests for pipeline_core.guardrails — pre-publish originality + safety."""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import models
from database import Base
from pipeline_core import guardrails as g


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def db_session():
    """Isolated in-memory SQLite DB session with all models created."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def empty_templates_dir(tmp_path):
    """An empty directory to pass as templates_dir."""
    d = tmp_path / "empty_templates"
    d.mkdir()
    return str(d).replace("\\", "/")


@pytest.fixture
def tiny_fake_mp4(tmp_path):
    """A 4-byte file named .mp4 — used when we only care about path existence
    and mock out the frame extraction."""
    p = tmp_path / "tiny.mp4"
    p.write_bytes(b"fake")
    return str(p).replace("\\", "/")


def _insert_user(db, user_id: int = 1):
    """Insert a minimal User row and return it."""
    u = models.User(id=user_id, email=f"u{user_id}@x.com", name=f"user{user_id}")
    db.add(u)
    db.commit()
    return u


def _insert_channel(db, user_id: int = 1, channel_id: int = 1):
    """Insert a minimal Channel row and return it."""
    c = models.Channel(id=channel_id, user_id=user_id, name=f"ch{channel_id}")
    db.add(c)
    db.commit()
    return c


def _insert_upload(
    db,
    *,
    upload_id: int,
    user_id: int,
    channel_id: int,
    clip_id: int,
    title: str,
    publish_kind: str = "short",
    status: str = "done",
    age_hours: float = 1.0,
    thumb_path: str = "",
):
    """Insert Job, Clip, UploadJob trio with controllable age."""
    now = datetime.now(timezone.utc)
    when = now - timedelta(hours=age_hours)

    job = models.Job(
        id=upload_id + 1000, user_id=user_id, status="done",
        platform="youtube_short", video_name="test",
    )
    db.add(job)
    db.flush()

    clip = models.Clip(
        id=clip_id, job_id=job.id, clip_index=0,
        file_path="", thumb_path=thumb_path,
    )
    db.add(clip)
    db.flush()

    up = models.UploadJob(
        id=upload_id, user_id=user_id, clip_id=clip_id, channel_id=channel_id,
        status=status, publish_kind=publish_kind,
        title=title, created_at=when, updated_at=when,
    )
    db.add(up)
    db.commit()
    return up


# ══════════════════════════════════════════════════════════════════════════════
# 1. Dataclass + report.ok logic
# ══════════════════════════════════════════════════════════════════════════════


def test_alert_dataclass_shape():
    a = g.GuardrailAlert(severity="warn", code="x.y", message="msg")
    assert a.severity == "warn"
    assert a.code == "x.y"
    assert a.message == "msg"
    assert a.details == {}


def test_report_ok_true_when_no_block_alerts(db_session):
    """GuardrailsReport.ok is True iff no alert has severity='block'."""
    _insert_user(db_session, user_id=1)
    _insert_channel(db_session, user_id=1, channel_id=1)
    # skip everything so no block alerts appear
    report = g.run_all_guardrails(
        "nonexistent.mp4", user_id=1, platform="youtube_short", db=db_session,
        skip=["watermark", "duplicate", "repetition", "cadence", "music_rights"],
    )
    assert report.ok is True
    block_alerts = [a for a in report.all_alerts if a.severity == "block"]
    assert block_alerts == []


def test_report_ok_false_when_any_block_alert(db_session, mocker):
    """Inject a block alert through watermark → report.ok=False."""
    _insert_user(db_session, user_id=1)
    blocked = g.WatermarkResult(
        alerts=[g.GuardrailAlert(severity="block", code="watermark.x", message="blocked")],
        frames_sampled=1, templates_checked=[],
    )
    mocker.patch("pipeline_core.guardrails.detect_watermarks", return_value=blocked)
    mocker.patch(
        "pipeline_core.guardrails.check_self_duplicate",
        return_value=g.DuplicateResult(alerts=[], top_matches=[]),
    )
    mocker.patch(
        "pipeline_core.guardrails.check_template_repetition",
        return_value=g.RepetitionResult(alerts=[], recent_uploads_examined=0, detected_pattern=None),
    )
    mocker.patch(
        "pipeline_core.guardrails.check_cadence",
        return_value=g.CadenceResult(
            alerts=[], hours_since_last=None, weekly_count=0, platform="youtube_short",
        ),
    )
    mocker.patch(
        "pipeline_core.guardrails.check_music_rights",
        return_value=g.MusicRightsResult(alerts=[], fingerprint_checked=False, status="unknown"),
    )
    r = g.run_all_guardrails("x.mp4", user_id=1, platform="youtube_short", db=db_session)
    assert r.ok is False


def test_report_all_alerts_flattens_from_subresults(db_session, mocker):
    mocker.patch(
        "pipeline_core.guardrails.detect_watermarks",
        return_value=g.WatermarkResult(
            alerts=[g.GuardrailAlert(severity="info", code="w.x", message="w")],
            frames_sampled=0, templates_checked=[],
        ),
    )
    mocker.patch(
        "pipeline_core.guardrails.check_self_duplicate",
        return_value=g.DuplicateResult(
            alerts=[g.GuardrailAlert(severity="info", code="d.x", message="d")],
            top_matches=[],
        ),
    )
    mocker.patch(
        "pipeline_core.guardrails.check_template_repetition",
        return_value=g.RepetitionResult(
            alerts=[g.GuardrailAlert(severity="info", code="r.x", message="r")],
            recent_uploads_examined=0, detected_pattern=None,
        ),
    )
    mocker.patch(
        "pipeline_core.guardrails.check_cadence",
        return_value=g.CadenceResult(
            alerts=[g.GuardrailAlert(severity="info", code="c.x", message="c")],
            hours_since_last=None, weekly_count=0, platform="youtube_short",
        ),
    )
    mocker.patch(
        "pipeline_core.guardrails.check_music_rights",
        return_value=g.MusicRightsResult(
            alerts=[g.GuardrailAlert(severity="info", code="m.x", message="m")],
            fingerprint_checked=False, status="unknown",
        ),
    )
    r = g.run_all_guardrails("x.mp4", user_id=1, platform="youtube_short", db=db_session)
    codes = {a.code for a in r.all_alerts}
    assert {"w.x", "d.x", "r.x", "c.x", "m.x"}.issubset(codes)


# ══════════════════════════════════════════════════════════════════════════════
# 2. detect_watermarks
# ══════════════════════════════════════════════════════════════════════════════


def test_watermarks_empty_templates_dir_emits_info_alert(empty_templates_dir, tiny_fake_mp4):
    r = g.detect_watermarks(tiny_fake_mp4, templates_dir=empty_templates_dir, sample_frames=2)
    assert isinstance(r, g.WatermarkResult)
    # With no templates, function must not crash and must emit a non-'block' alert.
    block_alerts = [a for a in r.alerts if a.severity == "block"]
    assert block_alerts == [], "No templates → no block alerts"
    info_codes = {a.code for a in r.alerts}
    # Either no_templates_available or no_templates_loaded or no_frames_extracted —
    # all acceptable info outcomes for this "nothing to match" case.
    assert any(c.startswith("watermark.") for c in info_codes)


def test_watermarks_returns_watermarkresult_dataclass(tiny_fake_mp4):
    r = g.detect_watermarks(tiny_fake_mp4, templates_dir=None, sample_frames=2)
    assert isinstance(r, g.WatermarkResult)
    assert isinstance(r.alerts, list)
    assert isinstance(r.frames_sampled, int)
    assert isinstance(r.templates_checked, list)


def test_watermarks_missing_video_returns_info_alert_not_crash(empty_templates_dir):
    r = g.detect_watermarks("/definitely/does/not/exist.mp4",
                            templates_dir=empty_templates_dir, sample_frames=2)
    assert isinstance(r, g.WatermarkResult)
    block_alerts = [a for a in r.alerts if a.severity == "block"]
    assert block_alerts == []


# ══════════════════════════════════════════════════════════════════════════════
# 3. check_self_duplicate
# ══════════════════════════════════════════════════════════════════════════════


def test_self_duplicate_no_prior_uploads_empty_matches(db_session, tiny_fake_mp4):
    _insert_user(db_session, user_id=1)
    r = g.check_self_duplicate(tiny_fake_mp4, user_id=1, db=db_session)
    assert isinstance(r, g.DuplicateResult)
    assert r.top_matches == []
    warn_alerts = [a for a in r.alerts if a.severity == "warn"]
    assert warn_alerts == []


def test_self_duplicate_skips_pruned_file_paths(db_session, tiny_fake_mp4):
    """Prior upload's clip.file_path doesn't exist on disk → skipped gracefully."""
    _insert_user(db_session, user_id=1)
    _insert_channel(db_session, user_id=1, channel_id=1)
    _insert_upload(
        db_session, upload_id=1, user_id=1, channel_id=1, clip_id=1,
        title="test", publish_kind="short", status="done", age_hours=1.0,
    )
    # clip.file_path is the empty string by default → pruned file
    r = g.check_self_duplicate(tiny_fake_mp4, user_id=1, db=db_session)
    # No crash. May or may not have alerts depending on whether pHash could even run.
    assert isinstance(r, g.DuplicateResult)


def test_self_duplicate_respects_lookback_days(db_session, tiny_fake_mp4):
    """A prior upload older than lookback_days must not produce a match."""
    _insert_user(db_session, user_id=1)
    _insert_channel(db_session, user_id=1, channel_id=1)
    # 60 days old
    _insert_upload(
        db_session, upload_id=1, user_id=1, channel_id=1, clip_id=1,
        title="ancient", publish_kind="short", status="done", age_hours=60 * 24,
    )
    r = g.check_self_duplicate(tiny_fake_mp4, user_id=1, db=db_session, lookback_days=30)
    assert r.top_matches == []


# ══════════════════════════════════════════════════════════════════════════════
# 4. check_template_repetition
# ══════════════════════════════════════════════════════════════════════════════


def test_repetition_empty_history_no_warn(db_session):
    _insert_user(db_session, user_id=1)
    r = g.check_template_repetition(user_id=1, db=db_session, lookback_count=5)
    assert isinstance(r, g.RepetitionResult)
    warn_alerts = [a for a in r.alerts if a.severity == "warn"]
    assert warn_alerts == []


def test_repetition_identical_titles_across_last_n_fires_warn(db_session):
    _insert_user(db_session, user_id=1)
    _insert_channel(db_session, user_id=1, channel_id=1)
    # 5 identical titles
    for i in range(5):
        _insert_upload(
            db_session, upload_id=i + 1, user_id=1, channel_id=1, clip_id=i + 1,
            title="Telugu news today breaking update", publish_kind="short",
            status="done", age_hours=i + 1,
        )
    r = g.check_template_repetition(user_id=1, db=db_session, lookback_count=5)
    warn_codes = {a.code for a in r.alerts if a.severity == "warn"}
    assert "repetition.template_overused" in warn_codes, (
        f"Expected template_overused warn, got alerts: {[a.code for a in r.alerts]}"
    )
    assert r.detected_pattern is not None


def test_repetition_varied_titles_does_not_fire(db_session):
    _insert_user(db_session, user_id=1)
    _insert_channel(db_session, user_id=1, channel_id=1)
    varied = [
        "Stock market crash explained",
        "Monsoon arrives in Hyderabad",
        "New policy from state government",
        "Cricket match highlights today",
        "Technology startup raises 50 crore",
    ]
    for i, title in enumerate(varied):
        _insert_upload(
            db_session, upload_id=i + 1, user_id=1, channel_id=1, clip_id=i + 1,
            title=title, publish_kind="short", status="done", age_hours=i + 1,
        )
    r = g.check_template_repetition(user_id=1, db=db_session, lookback_count=5)
    warn_alerts = [a for a in r.alerts if a.severity == "warn"]
    assert warn_alerts == [], (
        f"Varied titles should not fire; got {[a.code for a in warn_alerts]}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. check_cadence
# ══════════════════════════════════════════════════════════════════════════════


def test_cadence_no_uploads_emits_first_post_info(db_session):
    _insert_user(db_session, user_id=1)
    r = g.check_cadence(user_id=1, db=db_session, platform="instagram_reel")
    codes = {a.code for a in r.alerts}
    assert "cadence.first_post" in codes
    assert r.weekly_count == 0
    assert r.hours_since_last is None


def test_cadence_too_soon_fires_warn(db_session):
    _insert_user(db_session, user_id=1)
    _insert_channel(db_session, user_id=1, channel_id=1)
    _insert_upload(
        db_session, upload_id=1, user_id=1, channel_id=1, clip_id=1,
        title="recent", publish_kind="short", status="done", age_hours=2.0,
    )
    r = g.check_cadence(user_id=1, db=db_session, platform="instagram_reel",
                        min_hours_between_ms=6.0)
    warn_codes = {a.code for a in r.alerts if a.severity == "warn"}
    assert "cadence.too_soon" in warn_codes
    assert r.hours_since_last is not None and r.hours_since_last < 6.0


def test_cadence_weekly_cap_exceeded_reel(db_session):
    _insert_user(db_session, user_id=1)
    _insert_channel(db_session, user_id=1, channel_id=1)
    # 5 reel uploads in last week, cap=4
    for i in range(5):
        _insert_upload(
            db_session, upload_id=i + 1, user_id=1, channel_id=1, clip_id=i + 1,
            title=f"Title {i}", publish_kind="short",
            status="done", age_hours=6 + i * 24,  # spaced across last week
        )
    r = g.check_cadence(user_id=1, db=db_session, platform="instagram_reel",
                        min_hours_between_ms=6.0, weekly_cap_reel=4)
    warn_codes = {a.code for a in r.alerts if a.severity == "warn"}
    assert "cadence.weekly_cap_exceeded" in warn_codes


def test_cadence_weekly_cap_not_exceeded_youtube_short(db_session):
    _insert_user(db_session, user_id=1)
    _insert_channel(db_session, user_id=1, channel_id=1)
    for i in range(15):
        _insert_upload(
            db_session, upload_id=i + 1, user_id=1, channel_id=1, clip_id=i + 1,
            title=f"Title {i}", publish_kind="short",
            status="done", age_hours=6 + i * 10,
        )
    r = g.check_cadence(user_id=1, db=db_session, platform="youtube_short",
                        min_hours_between_ms=6.0, weekly_cap_short=21)
    warn_codes = {a.code for a in r.alerts if a.severity == "warn"
                  and a.code == "cadence.weekly_cap_exceeded"}
    assert warn_codes == set(), "15 < 21 → no cap alert"


def test_cadence_status_filter_excludes_failed(db_session):
    """failed/cancelled UploadJobs should NOT count toward cadence."""
    _insert_user(db_session, user_id=1)
    _insert_channel(db_session, user_id=1, channel_id=1)
    _insert_upload(
        db_session, upload_id=1, user_id=1, channel_id=1, clip_id=1,
        title="failed upload", publish_kind="short", status="failed", age_hours=2.0,
    )
    r = g.check_cadence(user_id=1, db=db_session, platform="instagram_reel")
    # No completed uploads → first_post
    codes = {a.code for a in r.alerts}
    assert "cadence.first_post" in codes


# ══════════════════════════════════════════════════════════════════════════════
# 6. check_music_rights (stub)
# ══════════════════════════════════════════════════════════════════════════════


def test_music_rights_no_db_returns_unknown_status():
    r = g.check_music_rights("any.mp4", fingerprint_db_path=None)
    assert r.status == "unknown"
    codes = {a.code for a in r.alerts}
    assert "music_rights.fingerprint_unavailable" in codes
    assert r.fingerprint_checked is False


def test_music_rights_missing_db_path_returns_unknown(tmp_path):
    r = g.check_music_rights("any.mp4",
                             fingerprint_db_path=str(tmp_path / "doesnt_exist.db"))
    assert r.status == "unknown"


def test_music_rights_returns_musicrightsresult_dataclass():
    r = g.check_music_rights("any.mp4")
    assert isinstance(r, g.MusicRightsResult)
    assert isinstance(r.alerts, list)
    assert isinstance(r.fingerprint_checked, bool)
    assert r.status in {"unknown", "clean", "flagged"}


# ══════════════════════════════════════════════════════════════════════════════
# 7. run_all_guardrails orchestration
# ══════════════════════════════════════════════════════════════════════════════


def test_run_all_calls_every_check_by_default(db_session, mocker):
    _insert_user(db_session, user_id=1)
    m_wm = mocker.patch("pipeline_core.guardrails.detect_watermarks",
                        return_value=g.WatermarkResult(alerts=[], frames_sampled=0, templates_checked=[]))
    m_dup = mocker.patch("pipeline_core.guardrails.check_self_duplicate",
                         return_value=g.DuplicateResult(alerts=[], top_matches=[]))
    m_rep = mocker.patch("pipeline_core.guardrails.check_template_repetition",
                         return_value=g.RepetitionResult(alerts=[], recent_uploads_examined=0, detected_pattern=None))
    m_cad = mocker.patch("pipeline_core.guardrails.check_cadence",
                         return_value=g.CadenceResult(alerts=[], hours_since_last=None,
                                                       weekly_count=0, platform="instagram_reel"))
    m_mr = mocker.patch("pipeline_core.guardrails.check_music_rights",
                        return_value=g.MusicRightsResult(alerts=[], fingerprint_checked=False, status="unknown"))

    g.run_all_guardrails("x.mp4", user_id=1, platform="instagram_reel", db=db_session)

    assert m_wm.call_count == 1
    assert m_dup.call_count == 1
    assert m_rep.call_count == 1
    assert m_cad.call_count == 1
    assert m_mr.call_count == 1


def test_run_all_skip_param_skips_checks(db_session, mocker):
    _insert_user(db_session, user_id=1)
    m_cad = mocker.patch("pipeline_core.guardrails.check_cadence")
    mocker.patch("pipeline_core.guardrails.detect_watermarks",
                 return_value=g.WatermarkResult(alerts=[], frames_sampled=0, templates_checked=[]))
    mocker.patch("pipeline_core.guardrails.check_self_duplicate",
                 return_value=g.DuplicateResult(alerts=[], top_matches=[]))
    mocker.patch("pipeline_core.guardrails.check_template_repetition",
                 return_value=g.RepetitionResult(alerts=[], recent_uploads_examined=0, detected_pattern=None))
    mocker.patch("pipeline_core.guardrails.check_music_rights",
                 return_value=g.MusicRightsResult(alerts=[], fingerprint_checked=False, status="unknown"))

    r = g.run_all_guardrails("x.mp4", user_id=1, platform="instagram_reel", db=db_session,
                             skip=["cadence"])
    assert m_cad.call_count == 0
    skip_codes = {a.code for a in r.all_alerts}
    assert any(c.endswith("skipped") for c in skip_codes)


def test_run_all_exception_in_check_recorded_as_info_not_raised(db_session, mocker):
    _insert_user(db_session, user_id=1)

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    mocker.patch("pipeline_core.guardrails.detect_watermarks", side_effect=boom)
    mocker.patch("pipeline_core.guardrails.check_self_duplicate",
                 return_value=g.DuplicateResult(alerts=[], top_matches=[]))
    mocker.patch("pipeline_core.guardrails.check_template_repetition",
                 return_value=g.RepetitionResult(alerts=[], recent_uploads_examined=0, detected_pattern=None))
    mocker.patch("pipeline_core.guardrails.check_cadence",
                 return_value=g.CadenceResult(alerts=[], hours_since_last=None,
                                                weekly_count=0, platform="instagram_reel"))
    mocker.patch("pipeline_core.guardrails.check_music_rights",
                 return_value=g.MusicRightsResult(alerts=[], fingerprint_checked=False, status="unknown"))

    r = g.run_all_guardrails("x.mp4", user_id=1, platform="instagram_reel", db=db_session)
    # Does NOT raise. An info alert captures the failure.
    codes = {a.code for a in r.all_alerts}
    assert any("watermark" in c for c in codes)


def test_run_all_guardrails_returns_report_dataclass(db_session, mocker):
    _insert_user(db_session, user_id=1)
    mocker.patch("pipeline_core.guardrails.detect_watermarks",
                 return_value=g.WatermarkResult(alerts=[], frames_sampled=0, templates_checked=[]))
    mocker.patch("pipeline_core.guardrails.check_self_duplicate",
                 return_value=g.DuplicateResult(alerts=[], top_matches=[]))
    mocker.patch("pipeline_core.guardrails.check_template_repetition",
                 return_value=g.RepetitionResult(alerts=[], recent_uploads_examined=0, detected_pattern=None))
    mocker.patch("pipeline_core.guardrails.check_cadence",
                 return_value=g.CadenceResult(alerts=[], hours_since_last=None,
                                                weekly_count=0, platform="instagram_reel"))
    mocker.patch("pipeline_core.guardrails.check_music_rights",
                 return_value=g.MusicRightsResult(alerts=[], fingerprint_checked=False, status="unknown"))
    r = g.run_all_guardrails("x.mp4", user_id=1, platform="instagram_reel", db=db_session)
    assert isinstance(r, g.GuardrailsReport)
    assert isinstance(r.all_alerts, list)
    assert isinstance(r.warnings, list)
    assert isinstance(r.ok, bool)
