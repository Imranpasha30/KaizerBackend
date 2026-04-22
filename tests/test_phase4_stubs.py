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


def test_collect_training_record_returns_none_when_no_db_or_missing():
    from pipeline_core.phase4.training_flywheel import collect_training_record
    # upload_job_id=1 with db=None → no DB lookup possible → returns None
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


def test_traverse_returns_empty_list_without_db():
    from pipeline_core.phase4.creator_graph import traverse
    # No db supplied → empty list, no crash
    result = traverse(1, edge_type="variant_of", db=None)
    assert result == []


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


def test_create_team_requires_db():
    from pipeline_core.phase4.agency_mode import create_team
    # Real implementation — no db supplied → ValueError (not NIE)
    with pytest.raises(ValueError):
        create_team(owner_user_id=1, name="X", db=None)


# ══════════════════════════════════════════════════════════════════════════════
# pro_export
# ══════════════════════════════════════════════════════════════════════════════


def test_export_unsupported_format_raises_valueerror():
    """format='avid' (or any unknown) must raise ValueError immediately."""
    from pipeline_core.phase4.pro_export import export_project
    with pytest.raises(ValueError, match="format must be one of"):
        export_project("src.mp4", output_path="out.xml", format="avid")


def test_export_missing_source_raises_valueerror(tmp_path):
    """source_path that does not exist must raise ValueError."""
    from pipeline_core.phase4.pro_export import export_project
    missing = str(tmp_path / "no_such_file_kaizer_test.mp4")
    with pytest.raises(ValueError):
        export_project(missing, output_path=str(tmp_path / "out.fcpxml"), format="fcpxml")


# ── Tests that need a real video file — skip if ffmpeg/ffprobe unavailable ──

def _make_test_mp4(tmp_path) -> str:
    """Create a minimal valid mp4 for export tests. Skip if ffmpeg missing."""
    import subprocess
    out = str(tmp_path / "test_source.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc2=duration=5:size=1080x1920:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-t", "5",
        out,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
    except FileNotFoundError:
        pytest.skip("ffmpeg not available — skipping pro_export integration test")
    if result.returncode != 0:
        pytest.skip("ffmpeg failed to create test mp4")
    return out


def test_export_fcpxml_produces_valid_xml(tmp_path):
    """export_project with format='fcpxml' writes parseable XML with root tag fcpxml v1.11."""
    import xml.etree.ElementTree as ET
    from pipeline_core.phase4.pro_export import export_project

    src = _make_test_mp4(tmp_path)
    out = str(tmp_path / "out.fcpxml")
    result = export_project(src, output_path=out, format="fcpxml")

    assert result.output_path == out
    assert result.format == "fcpxml"

    tree = ET.parse(out)
    root = tree.getroot()
    assert root.tag == "fcpxml"
    assert root.attrib.get("version") == "1.11"


def test_export_prproj_xml_produces_valid_xml(tmp_path):
    """export_project with format='prproj_xml' writes parseable XML with root tag xmeml v5."""
    import xml.etree.ElementTree as ET
    from pipeline_core.phase4.pro_export import export_project

    src = _make_test_mp4(tmp_path)
    out = str(tmp_path / "out.xml")
    result = export_project(src, output_path=out, format="prproj_xml")

    assert result.format == "prproj_xml"

    tree = ET.parse(out)
    root = tree.getroot()
    assert root.tag == "xmeml"
    assert root.attrib.get("version") == "5"


def test_export_marker_count_reflected_in_result(tmp_path):
    """Passing 3 markers gives result.markers == 3."""
    from pipeline_core.phase4.pro_export import export_project

    src = _make_test_mp4(tmp_path)
    out = str(tmp_path / "out_markers.fcpxml")
    mkrs = [
        {"t": 1.0, "label": "hook",   "color": "red"},
        {"t": 2.5, "label": "mid",    "color": "blue"},
        {"t": 4.0, "label": "cta",    "color": "green"},
    ]
    result = export_project(src, markers=mkrs, output_path=out, format="fcpxml")
    assert result.markers == 3


def test_export_broll_adds_secondary_track_result(tmp_path):
    """broll_tracks=[1 entry] -> result.tracks == 2; no broll -> 1."""
    from pipeline_core.phase4.pro_export import export_project

    src = _make_test_mp4(tmp_path)

    out_no_broll = str(tmp_path / "no_broll.fcpxml")
    res_no = export_project(src, output_path=out_no_broll, format="fcpxml")
    assert res_no.tracks == 1

    # Use the source as a stand-in b-roll (it exists on disk)
    out_broll = str(tmp_path / "with_broll.fcpxml")
    res_br = export_project(
        src,
        broll_tracks=[{"path": src, "t_start": 1.0, "duration": 2.0}],
        output_path=out_broll,
        format="fcpxml",
    )
    assert res_br.tracks == 2


def test_export_fcpxml_contains_chapter_markers(tmp_path):
    """2 markers -> 2 <chapter-marker> elements in parsed FCPXML."""
    import xml.etree.ElementTree as ET
    from pipeline_core.phase4.pro_export import export_project

    src = _make_test_mp4(tmp_path)
    out = str(tmp_path / "markers.fcpxml")
    mkrs = [
        {"t": 1.0, "label": "A", "color": "red"},
        {"t": 3.0, "label": "B", "color": "orange"},
    ]
    export_project(src, markers=mkrs, output_path=out, format="fcpxml")

    tree = ET.parse(out)
    chapter_markers = tree.getroot().findall(".//chapter-marker")
    assert len(chapter_markers) == 2


def test_export_prproj_contains_marker_elements(tmp_path):
    """2 markers -> 2 <marker> elements in parsed xmeml."""
    import xml.etree.ElementTree as ET
    from pipeline_core.phase4.pro_export import export_project

    src = _make_test_mp4(tmp_path)
    out = str(tmp_path / "markers.xml")
    mkrs = [
        {"t": 1.0, "label": "A", "color": "red"},
        {"t": 3.0, "label": "B", "color": "yellow"},
    ]
    export_project(src, markers=mkrs, output_path=out, format="prproj_xml")

    tree = ET.parse(out)
    marker_elems = tree.getroot().findall(".//marker")
    assert len(marker_elems) == 2


def test_export_fcpxml_file_uri_format(tmp_path):
    """Asset src attribute in FCPXML uses file:/// URI with forward slashes."""
    import xml.etree.ElementTree as ET
    from pipeline_core.phase4.pro_export import export_project

    src = _make_test_mp4(tmp_path)
    out = str(tmp_path / "uri_test.fcpxml")
    export_project(src, output_path=out, format="fcpxml")

    tree = ET.parse(out)
    assets = tree.getroot().findall(".//asset")
    assert assets, "No <asset> elements found in FCPXML"
    main_asset = assets[0]
    src_uri = main_asset.attrib.get("src", "")
    assert src_uri.startswith("file:///"), f"Expected file:/// URI, got: {src_uri!r}"
    assert "\\" not in src_uri, f"Back-slashes in URI: {src_uri!r}"


def test_export_missing_broll_file_path_emits_warning_not_raise(tmp_path):
    """A b-roll entry referencing a missing file emits a warning, does not raise."""
    from pipeline_core.phase4.pro_export import export_project

    src = _make_test_mp4(tmp_path)
    out = str(tmp_path / "warn_broll.fcpxml")
    missing_broll = str(tmp_path / "nonexistent_broll.mp4")

    result = export_project(
        src,
        broll_tracks=[{"path": missing_broll, "t_start": 1.0, "duration": 2.0}],
        output_path=out,
        format="fcpxml",
    )
    # Must not raise; must have at least one warning mentioning the missing path
    assert any("broll" in w.lower() or "nonexistent" in w.lower() or missing_broll in w
               for w in result.warnings), f"No broll warning emitted: {result.warnings}"


def test_export_srt_parse_failure_emits_warning(tmp_path):
    """A non-existent SRT path emits a warning and render proceeds."""
    from pipeline_core.phase4.pro_export import export_project

    src = _make_test_mp4(tmp_path)
    out = str(tmp_path / "warn_srt.fcpxml")
    missing_srt = str(tmp_path / "nonexistent.srt")

    result = export_project(
        src,
        caption_srt_path=missing_srt,
        output_path=out,
        format="fcpxml",
    )
    # Must not raise; must have at least one warning about the SRT
    assert any("srt" in w.lower() or "caption" in w.lower() or missing_srt in w
               for w in result.warnings), f"No SRT warning emitted: {result.warnings}"


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


def test_authenticate_bad_key_raises_valueerror():
    from pipeline_core.phase4.regional_api import authenticate
    # Real implementation — invalid key (no db) raises ValueError, not NIE
    with pytest.raises(ValueError, match="Invalid API key"):
        authenticate("bad_key", db=None)


def test_submit_ingest_not_implemented():
    from pipeline_core.phase4.regional_api import submit_ingest, RegionalIngestRequest
    req = RegionalIngestRequest(source_url=None, language="hi", mode="trailer")
    with pytest.raises(NotImplementedError):
        submit_ingest(req)
