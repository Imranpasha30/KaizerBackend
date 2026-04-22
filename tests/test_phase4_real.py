"""
tests/test_phase4_real.py
==========================
Real DB-backed tests for Wave 3A Phase 4 implementations.

Uses in-memory SQLite (same pattern as test_guardrails.py).
All 18+ cases are independent — each gets its own db_session fixture.
"""
from __future__ import annotations

import hashlib

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import models
from database import Base


# ══════════════════════════════════════════════════════════════════════════════
# Shared fixture
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def db_session():
    """Isolated in-memory SQLite DB with all models created."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_user(db, user_id: int = 1) -> models.User:
    u = models.User(id=user_id, email=f"u{user_id}@test.com", name=f"user{user_id}")
    db.add(u); db.flush()
    return u


def _make_channel(db, user_id: int = 1, channel_id: int = 1,
                  language: str = "te") -> models.Channel:
    c = models.Channel(id=channel_id, user_id=user_id,
                       name=f"ch{channel_id}", language=language)
    db.add(c); db.flush()
    return c


def _make_job(db, user_id: int = 1, job_id: int = 1) -> models.Job:
    j = models.Job(id=job_id, user_id=user_id, status="done",
                   platform="youtube_short", video_name="test.mp4")
    db.add(j); db.flush()
    return j


def _make_clip(db, job_id: int = 1, clip_id: int = 1,
               meta: str = "{}") -> models.Clip:
    c = models.Clip(id=clip_id, job_id=job_id, clip_index=0,
                    filename="clip.mp4", file_path="", meta=meta)
    db.add(c); db.flush()
    return c


def _make_upload_job(db, upload_id: int = 1, user_id: int = 1,
                     clip_id: int = 1, channel_id: int = 1) -> models.UploadJob:
    uj = models.UploadJob(
        id=upload_id, user_id=user_id, clip_id=clip_id,
        channel_id=channel_id, status="done",
    )
    db.add(uj); db.commit()
    return uj


def _seed_full(db, upload_id: int = 1, user_id: int = 1,
               language: str = "te") -> models.UploadJob:
    """Create User + Channel + Job + Clip + UploadJob in one call."""
    _make_user(db, user_id)
    _make_channel(db, user_id=user_id, channel_id=upload_id, language=language)
    _make_job(db, user_id=user_id, job_id=upload_id)
    _make_clip(db, job_id=upload_id, clip_id=upload_id)
    return _make_upload_job(db, upload_id=upload_id, user_id=user_id,
                            clip_id=upload_id, channel_id=upload_id)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Training Flywheel
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainingFlywheel:

    def test_collect_training_record_creates_row(self, db_session):
        from pipeline_core.phase4.training_flywheel import collect_training_record
        _seed_full(db_session, upload_id=1)

        result = collect_training_record(upload_job_id=1, db=db_session)

        assert result is not None
        assert result.upload_job_id == 1
        assert result.clip_id == 1
        assert result.niche == "news_te"
        # Row exists in DB
        row = db_session.query(models.TrainingRecord).filter_by(upload_job_id=1).first()
        assert row is not None
        assert row.niche == "news_te"

    def test_collect_training_record_reuses_row_on_re_collect(self, db_session):
        """UniqueConstraint on upload_job_id: second call updates, not duplicates."""
        from pipeline_core.phase4.training_flywheel import collect_training_record
        _seed_full(db_session, upload_id=1)

        r1 = collect_training_record(upload_job_id=1, db=db_session)
        r2 = collect_training_record(upload_job_id=1, db=db_session)

        assert r1 is not None
        assert r2 is not None
        count = db_session.query(models.TrainingRecord).filter_by(upload_job_id=1).count()
        assert count == 1, "Must not create duplicate rows"

    def test_collect_training_record_missing_upload_returns_none(self, db_session):
        from pipeline_core.phase4.training_flywheel import collect_training_record
        result = collect_training_record(upload_job_id=9999, db=db_session)
        assert result is None

    def test_collect_training_record_parses_narrative_meta(self, db_session):
        """hook_score / completion_score / composite_score come from clip.meta JSON."""
        import json
        from pipeline_core.phase4.training_flywheel import collect_training_record

        _make_user(db_session, 1)
        _make_channel(db_session, user_id=1, channel_id=1, language="hi")
        _make_job(db_session, user_id=1, job_id=1)
        meta_json = json.dumps({
            "hook_score": 0.85,
            "completion_score": 0.72,
            "composite_score": 0.79,
            "narrative_role": "turning_point",
        })
        _make_clip(db_session, job_id=1, clip_id=1, meta=meta_json)
        _make_upload_job(db_session, upload_id=1, user_id=1, clip_id=1, channel_id=1)

        result = collect_training_record(upload_job_id=1, db=db_session)

        assert result is not None
        assert result.hook_score == pytest.approx(0.85)
        assert result.completion_score == pytest.approx(0.72)
        assert result.composite_score == pytest.approx(0.79)
        assert result.narrative_role == "turning_point"
        assert result.niche == "news_hi"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Creator Graph
# ══════════════════════════════════════════════════════════════════════════════

class TestCreatorGraph:

    def _seed_clips(self, db, count: int = 3):
        u = _make_user(db, 1)
        j = _make_job(db, user_id=1, job_id=1)
        clips = []
        for i in range(1, count + 1):
            c = _make_clip(db, job_id=1, clip_id=i)
            clips.append(c)
        db.commit()
        return clips

    def test_link_clips_inserts_edge(self, db_session):
        from pipeline_core.phase4.creator_graph import link_clips, ClipEdge
        self._seed_clips(db_session, 2)

        edge = link_clips(1, 2, edge_type="variant_of", db=db_session)

        assert isinstance(edge, ClipEdge)
        assert edge.src_clip_id == 1
        assert edge.dst_clip_id == 2
        assert edge.edge_type == "variant_of"
        row = db_session.query(models.ClipEdge).filter_by(
            src_clip_id=1, dst_clip_id=2, edge_type="variant_of"
        ).first()
        assert row is not None

    def test_link_clips_duplicate_edge_no_error(self, db_session):
        """Inserting the same edge twice must not raise and returns existing."""
        from pipeline_core.phase4.creator_graph import link_clips
        self._seed_clips(db_session, 2)

        e1 = link_clips(1, 2, edge_type="series_part_of", db=db_session)
        e2 = link_clips(1, 2, edge_type="series_part_of", db=db_session)

        assert e1.edge_type == e2.edge_type == "series_part_of"
        count = db_session.query(models.ClipEdge).filter_by(
            src_clip_id=1, dst_clip_id=2, edge_type="series_part_of"
        ).count()
        assert count == 1

    def test_traverse_out_direction_returns_dst_ids(self, db_session):
        from pipeline_core.phase4.creator_graph import link_clips, traverse
        self._seed_clips(db_session, 3)

        link_clips(1, 2, edge_type="variant_of", db=db_session)
        link_clips(1, 3, edge_type="variant_of", db=db_session)

        result = traverse(1, edge_type="variant_of", direction="out", db=db_session)

        assert sorted(result) == [2, 3]

    def test_traverse_in_direction_returns_src_ids(self, db_session):
        from pipeline_core.phase4.creator_graph import link_clips, traverse
        self._seed_clips(db_session, 3)

        link_clips(1, 3, edge_type="trailer_for", db=db_session)
        link_clips(2, 3, edge_type="trailer_for", db=db_session)

        result = traverse(3, edge_type="trailer_for", direction="in", db=db_session)

        assert sorted(result) == [1, 2]

    def test_traverse_unknown_direction_raises(self, db_session):
        from pipeline_core.phase4.creator_graph import traverse
        with pytest.raises(ValueError, match="direction"):
            traverse(1, edge_type="variant_of", direction="sideways", db=db_session)

    def test_traverse_filters_by_edge_type(self, db_session):
        from pipeline_core.phase4.creator_graph import link_clips, traverse
        self._seed_clips(db_session, 3)

        link_clips(1, 2, edge_type="variant_of", db=db_session)
        link_clips(1, 3, edge_type="series_part_of", db=db_session)

        # Only variant_of edges from clip 1
        result = traverse(1, edge_type="variant_of", direction="out", db=db_session)
        assert result == [2]

        # Only series_part_of edges from clip 1
        result2 = traverse(1, edge_type="series_part_of", direction="out", db=db_session)
        assert result2 == [3]

    def test_traverse_empty_result_returns_empty_list(self, db_session):
        from pipeline_core.phase4.creator_graph import traverse
        self._seed_clips(db_session, 1)

        result = traverse(1, edge_type="variant_of", direction="out", db=db_session)
        assert result == []


# ══════════════════════════════════════════════════════════════════════════════
# 3. Agency Mode
# ══════════════════════════════════════════════════════════════════════════════

class TestAgencyMode:

    def test_create_team_creates_owner_member(self, db_session):
        from pipeline_core.phase4.agency_mode import create_team, AgencyTeam
        _make_user(db_session, 1)

        team = create_team(owner_user_id=1, name="NewsRoom Alpha", db=db_session)

        assert isinstance(team, AgencyTeam)
        assert team.owner_user_id == 1
        assert team.name == "NewsRoom Alpha"
        assert team.member_count == 1

        # DB has 1 team row
        assert db_session.query(models.AgencyTeam).count() == 1
        # DB has 1 member row with role='owner'
        member = db_session.query(models.AgencyMember).filter_by(
            agency_id=team.id, user_id=1
        ).first()
        assert member is not None
        assert member.role == "owner"

    def test_add_member_duplicate_returns_existing(self, db_session):
        from pipeline_core.phase4.agency_mode import create_team, add_member
        _make_user(db_session, 1)
        _make_user(db_session, 2)

        team = create_team(owner_user_id=1, name="Team B", db=db_session)
        m1 = add_member(team.id, user_id=2, role="creator", db=db_session)
        m2 = add_member(team.id, user_id=2, role="creator", db=db_session)

        assert m1.user_id == m2.user_id == 2
        count = db_session.query(models.AgencyMember).filter_by(
            agency_id=team.id, user_id=2
        ).count()
        assert count == 1

    def test_add_member_invalid_role_raises(self, db_session):
        from pipeline_core.phase4.agency_mode import add_member
        with pytest.raises(ValueError, match="role must be one of"):
            add_member(agency_id=1, user_id=1, role="godmode", db=db_session)

    def test_check_permission_owner_can_do_anything(self, db_session):
        from pipeline_core.phase4.agency_mode import create_team, check_permission
        _make_user(db_session, 1)
        team = create_team(owner_user_id=1, name="Team C", db=db_session)

        assert check_permission(1, team.id, "agency.delete", db=db_session) is True
        assert check_permission(1, team.id, "billing.view", db=db_session) is True
        assert check_permission(1, team.id, "clip.create", db=db_session) is True
        assert check_permission(1, team.id, "view", db=db_session) is True

    def test_check_permission_viewer_can_only_read(self, db_session):
        from pipeline_core.phase4.agency_mode import create_team, add_member, check_permission
        _make_user(db_session, 1)
        _make_user(db_session, 2)
        team = create_team(owner_user_id=1, name="Team D", db=db_session)
        add_member(team.id, user_id=2, role="viewer", db=db_session)

        assert check_permission(2, team.id, "view", db=db_session) is True
        assert check_permission(2, team.id, "clip.read", db=db_session) is True
        assert check_permission(2, team.id, "clip.create", db=db_session) is False
        assert check_permission(2, team.id, "billing.view", db=db_session) is False
        assert check_permission(2, team.id, "agency.delete", db=db_session) is False

    def test_check_permission_non_member_returns_false(self, db_session):
        from pipeline_core.phase4.agency_mode import create_team, check_permission
        _make_user(db_session, 1)
        _make_user(db_session, 99)
        team = create_team(owner_user_id=1, name="Team E", db=db_session)

        assert check_permission(99, team.id, "clip.create", db=db_session) is False

    def test_check_permission_admin_can_invite_but_not_agency(self, db_session):
        from pipeline_core.phase4.agency_mode import create_team, add_member, check_permission
        _make_user(db_session, 1)
        _make_user(db_session, 2)
        team = create_team(owner_user_id=1, name="Team F", db=db_session)
        add_member(team.id, user_id=2, role="admin", db=db_session)

        assert check_permission(2, team.id, "team.invite", db=db_session) is True
        assert check_permission(2, team.id, "billing.view", db=db_session) is True
        assert check_permission(2, team.id, "agency.delete", db=db_session) is False

    def test_record_audit_entry_inserts_row(self, db_session):
        from pipeline_core.phase4.agency_mode import (
            create_team, record_audit_entry, AuditLogEntry,
        )
        _make_user(db_session, 1)
        team = create_team(owner_user_id=1, name="Team G", db=db_session)

        entry = AuditLogEntry(
            agency_id=team.id,
            actor_user_id=1,
            action="clip.create",
            target_kind="clip",
            target_id=42,
            details={"note": "test"},
        )
        record_audit_entry(entry, db=db_session)

        row = db_session.query(models.AgencyAuditLog).filter_by(
            agency_id=team.id, action="clip.create"
        ).first()
        assert row is not None
        assert row.target_id == 42
        assert row.details == {"note": "test"}


# ══════════════════════════════════════════════════════════════════════════════
# 4. Regional API
# ══════════════════════════════════════════════════════════════════════════════

class TestRegionalApi:

    def _seed_key(self, db, raw_key: str = "secret123",
                  org_id: str = "tv9_telugu", active: bool = True) -> models.RegionalApiKey:
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        row = models.RegionalApiKey(
            org_id=org_id,
            api_key_hash=key_hash,
            label="TV9 Telugu",
            rate_limit_rpm=120,
            monthly_cap=5000,
            active=active,
        )
        db.add(row)
        db.commit()
        return row

    def test_authenticate_bad_key_raises(self, db_session):
        from pipeline_core.phase4.regional_api import authenticate
        self._seed_key(db_session, raw_key="correct_key")
        with pytest.raises(ValueError, match="Invalid API key"):
            authenticate("wrong_key", db=db_session)

    def test_authenticate_good_key_returns_credentials(self, db_session):
        from pipeline_core.phase4.regional_api import authenticate, PartnerCredentials
        self._seed_key(db_session, raw_key="my_secret_key", org_id="v6_news")

        creds = authenticate("my_secret_key", db=db_session)

        assert isinstance(creds, PartnerCredentials)
        assert creds.org_id == "v6_news"
        assert creds.rate_limit_rpm == 120
        assert creds.monthly_cap == 5000

    def test_authenticate_inactive_key_raises(self, db_session):
        from pipeline_core.phase4.regional_api import authenticate
        self._seed_key(db_session, raw_key="inactive_key", active=False)
        with pytest.raises(ValueError, match="Invalid API key"):
            authenticate("inactive_key", db=db_session)

    def test_authenticate_no_db_raises(self):
        from pipeline_core.phase4.regional_api import authenticate
        with pytest.raises(ValueError, match="Invalid API key"):
            authenticate("any_key", db=None)

    def test_authenticate_hash_stored_not_plaintext(self, db_session):
        """Verify the stored hash is SHA-256, not the raw key."""
        from pipeline_core.phase4.regional_api import authenticate
        raw = "plaintext_key"
        self._seed_key(db_session, raw_key=raw, org_id="sakshi_tv")

        creds = authenticate(raw, db=db_session)
        expected_hash = hashlib.sha256(raw.encode()).hexdigest()
        assert creds.api_key_hash == expected_hash
        # Ensure raw key is not stored
        row = db_session.query(models.RegionalApiKey).filter_by(org_id="sakshi_tv").first()
        assert row.api_key_hash != raw
        assert len(row.api_key_hash) == 64  # SHA-256 hex = 64 chars
