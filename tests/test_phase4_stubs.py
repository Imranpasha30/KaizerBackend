"""Tests for Phase 4 interface stubs.

These tests verify:
  1. All Phase 4 stub modules import cleanly.
  2. Dataclasses have the expected shape.
  3. Pure helper functions (no I/O required) work.
  4. Unimplemented operations raise NotImplementedError with a clear
     reference to the roadmap document — NOT AttributeError or silent
     passes.
"""
from __future__ import annotations

import pytest


# ══════════════════════════════════════════════════════════════════════════════
# Import-safety
# ══════════════════════════════════════════════════════════════════════════════


def test_phase4_package_imports():
    from pipeline_core import phase4
    assert hasattr(phase4, "training_flywheel")
    assert hasattr(phase4, "creator_graph")
    assert hasattr(phase4, "vertical_packs")
    assert hasattr(phase4, "agency_mode")
    assert hasattr(phase4, "pro_export")
    assert hasattr(phase4, "music_marketplace")
    assert hasattr(phase4, "trial_reels")
    assert hasattr(phase4, "regional_api")


def test_all_phase4_submodules_importable():
    """Each stub must import without raising."""
    from pipeline_core.phase4 import (
        training_flywheel, creator_graph, vertical_packs, agency_mode,
        pro_export, music_marketplace, trial_reels, regional_api,
    )
    assert training_flywheel is not None
    assert creator_graph is not None
    assert vertical_packs is not None
    assert agency_mode is not None
    assert pro_export is not None
    assert music_marketplace is not None
    assert trial_reels is not None
    assert regional_api is not None


# ══════════════════════════════════════════════════════════════════════════════
# training_flywheel
# ══════════════════════════════════════════════════════════════════════════════


def test_training_record_dataclass():
    from pipeline_core.phase4.training_flywheel import TrainingRecord
    r = TrainingRecord(upload_job_id=1, clip_id=2)
    assert r.upload_job_id == 1
    assert r.retention_curve == []


def test_collect_training_record_returns_none_in_stub():
    from pipeline_core.phase4.training_flywheel import collect_training_record
    assert collect_training_record(upload_job_id=1, db=None) is None


def test_retrain_narrative_scorer_raises_not_implemented():
    from pipeline_core.phase4.training_flywheel import retrain_narrative_scorer
    with pytest.raises(NotImplementedError, match="Phase 4"):
        retrain_narrative_scorer()


def test_deploy_model_raises_not_implemented():
    from pipeline_core.phase4.training_flywheel import deploy_model
    with pytest.raises(NotImplementedError, match="Phase 4"):
        deploy_model("whatever.bin")


# ══════════════════════════════════════════════════════════════════════════════
# creator_graph
# ══════════════════════════════════════════════════════════════════════════════


def test_clip_edge_dataclass():
    from pipeline_core.phase4.creator_graph import ClipEdge
    e = ClipEdge(edge_type="variant_of", src_clip_id=1, dst_clip_id=2, meta={"k": "v"})
    assert e.edge_type == "variant_of"


def test_link_clips_valid_edge_type_returns_clipedge():
    from pipeline_core.phase4.creator_graph import link_clips, ClipEdge
    result = link_clips(1, 2, edge_type="series_part_of")
    assert isinstance(result, ClipEdge)
    assert result.edge_type == "series_part_of"


def test_link_clips_invalid_edge_type_raises():
    from pipeline_core.phase4.creator_graph import link_clips
    with pytest.raises(ValueError, match="Unknown edge_type"):
        link_clips(1, 2, edge_type="fake_edge")


def test_traverse_raises_not_implemented():
    from pipeline_core.phase4.creator_graph import traverse
    with pytest.raises(NotImplementedError, match="Phase 4"):
        traverse(1, edge_type="variant_of")


# ══════════════════════════════════════════════════════════════════════════════
# vertical_packs
# ══════════════════════════════════════════════════════════════════════════════


def test_news_pack_shipped_inline():
    from pipeline_core.phase4.vertical_packs import load_pack
    pack = load_pack("news")
    assert pack is not None
    assert pack.niche == "news"
    assert "breaking" in pack.hook_opener_words


def test_unknown_pack_returns_none():
    from pipeline_core.phase4.vertical_packs import load_pack
    assert load_pack("doesnotexist") is None


def test_list_available_includes_news():
    from pipeline_core.phase4.vertical_packs import list_available
    assert "news" in list_available()


def test_apply_pack_to_narrative_adds_niche_tag():
    from pipeline_core.phase4.vertical_packs import load_pack, apply_pack_to_narrative
    pack = load_pack("news")
    result = apply_pack_to_narrative(pack, {"hook_score": 0.7})
    assert result["vertical_pack"] == "news"
    assert result["hook_score"] == 0.7


# ══════════════════════════════════════════════════════════════════════════════
# agency_mode
# ══════════════════════════════════════════════════════════════════════════════


def test_agency_roles_tuple():
    from pipeline_core.phase4.agency_mode import AGENCY_ROLES
    assert "owner" in AGENCY_ROLES
    assert "viewer" in AGENCY_ROLES


def test_agency_dataclasses():
    from pipeline_core.phase4.agency_mode import AgencyTeam, AgencyMember, AuditLogEntry
    t = AgencyTeam(id=1, owner_user_id=2, name="X")
    m = AgencyMember(user_id=2, agency_id=1, role="admin")
    e = AuditLogEntry(agency_id=1, actor_user_id=2, action="x", target_kind="clip", target_id=1)
    assert t.member_count == 0
    assert m.role == "admin"
    assert e.details == {}


def test_add_member_invalid_role_raises():
    from pipeline_core.phase4.agency_mode import add_member
    with pytest.raises(ValueError, match="role must be one of"):
        add_member(agency_id=1, user_id=1, role="godmode")


def test_create_team_not_implemented():
    from pipeline_core.phase4.agency_mode import create_team
    with pytest.raises(NotImplementedError):
        create_team(owner_user_id=1, name="X")


# ══════════════════════════════════════════════════════════════════════════════
# pro_export
# ══════════════════════════════════════════════════════════════════════════════


def test_pro_export_invalid_format_raises_valueerror():
    from pipeline_core.phase4.pro_export import export_project
    with pytest.raises(ValueError, match="format must be one of"):
        export_project("src.mp4", output_path="out.xml", format="avid_aaf")


def test_pro_export_valid_format_raises_not_implemented():
    from pipeline_core.phase4.pro_export import export_project
    with pytest.raises(NotImplementedError, match="Phase 4"):
        export_project("src.mp4", output_path="out.fcpxml", format="fcpxml")


# ══════════════════════════════════════════════════════════════════════════════
# music_marketplace
# ══════════════════════════════════════════════════════════════════════════════


def test_search_tracks_stub_returns_empty_list():
    from pipeline_core.phase4.music_marketplace import search_tracks
    assert search_tracks("calm") == []


def test_grant_license_not_implemented():
    from pipeline_core.phase4.music_marketplace import grant_license
    with pytest.raises(NotImplementedError):
        grant_license(track_id="x", user_id=1, upload_job_id=1)


# ══════════════════════════════════════════════════════════════════════════════
# trial_reels (decide_promotion is pure — test it properly)
# ══════════════════════════════════════════════════════════════════════════════


def test_decide_promotion_promotes_on_good_metrics():
    from pipeline_core.phase4.trial_reels import decide_promotion, TrialMetrics
    m = TrialMetrics(media_id="x", shares=30, reach=1000,
                     completion_pct=60.0, saves=10, elapsed_hours=24.5)
    d = decide_promotion(m)
    assert d.action == "promote"


def test_decide_promotion_keeps_trial_when_too_early():
    from pipeline_core.phase4.trial_reels import decide_promotion, TrialMetrics
    m = TrialMetrics(media_id="x", shares=30, reach=1000,
                     completion_pct=60.0, saves=10, elapsed_hours=12.0)
    d = decide_promotion(m)
    assert d.action == "keep_trial"


def test_decide_promotion_keeps_trial_on_low_shares():
    from pipeline_core.phase4.trial_reels import decide_promotion, TrialMetrics
    m = TrialMetrics(media_id="x", shares=5, reach=1000,      # 0.5%
                     completion_pct=60.0, saves=10, elapsed_hours=24.5)
    d = decide_promotion(m)
    assert d.action == "keep_trial"


def test_decide_promotion_keeps_trial_on_low_completion():
    from pipeline_core.phase4.trial_reels import decide_promotion, TrialMetrics
    m = TrialMetrics(media_id="x", shares=30, reach=1000,
                     completion_pct=40.0, saves=10, elapsed_hours=24.5)
    d = decide_promotion(m)
    assert d.action == "keep_trial"


def test_publish_as_trial_not_implemented():
    from pipeline_core.phase4.trial_reels import publish_as_trial
    with pytest.raises(NotImplementedError):
        publish_as_trial("x.mp4", access_token="t", ig_user_id="u")


# ══════════════════════════════════════════════════════════════════════════════
# regional_api
# ══════════════════════════════════════════════════════════════════════════════


def test_regional_ingest_request_dataclass():
    from pipeline_core.phase4.regional_api import RegionalIngestRequest
    r = RegionalIngestRequest(source_url="http://x", language="te", mode="standalone")
    assert r.language == "te"
    assert r.platforms == []


def test_authenticate_not_implemented():
    from pipeline_core.phase4.regional_api import authenticate
    with pytest.raises(NotImplementedError):
        authenticate("key")


def test_submit_ingest_not_implemented():
    from pipeline_core.phase4.regional_api import submit_ingest, RegionalIngestRequest
    req = RegionalIngestRequest(source_url=None, language="hi", mode="trailer")
    with pytest.raises(NotImplementedError):
        submit_ingest(req)
