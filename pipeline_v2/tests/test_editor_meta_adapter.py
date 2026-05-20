"""Unit tests for editor_meta_adapter (Step 8.2).

One test cluster per D-8.x decision so a future failure traces
directly to the locked behavior. Plus comprehensive coverage of:

  - Top-level shorts shape (all 21 keys present)
  - Top-level bulletin shape (all 7 keys present)
  - Per-clip dicts (all 24 shorts-clip keys, 14 bulletin-clip keys)
  - Helpers: _format_mmss_mmm, _strip_lang_suffix, _validate_shorts_indices_contiguous

No I/O, no SDK, no Gemini. Pure data transformation tests.
"""

from __future__ import annotations

import pytest

from pipeline_v2.editor_meta_adapter import (
    ClipRenderArtifacts,
    _format_mmss_mmm,
    _resolve_lang_cfg,
    _strip_lang_suffix,
    _validate_shorts_indices_contiguous,
    build_v1_bulletin_editor_meta,
    build_v1_shorts_editor_meta,
)
from pipeline_v2.models import (
    CleanTranscript,
    Entity,
    FullVideoCut,
    ImagePlan,
    ImagePlanEntry,
    JobOutput,
    Metadata,
    ShortsCut,
    SkippedSegment,
    StageTwoOutput,
    Word,
)


# ====================================================================== #
# Helpers                                                                 #
# ====================================================================== #


def _word(text: str, s: float, e: float) -> Word:
    return Word(w=text, s=s, e=e)


def _clean(n: int = 30) -> CleanTranscript:
    words = [_word(f"w{i}", i * 1.0, i * 1.0 + 0.8) for i in range(n)]
    return CleanTranscript(
        words=words,
        clip_boundaries={0: (0, n - 1)},
        source_word_map=list(range(n)),
    )


def _shorts_cut(index: int = 0, start: float = 10.0, end: float = 28.0,
                hook: str = "A test hook", importance: int = 7) -> ShortsCut:
    return ShortsCut(
        index=index,
        start_sec=start,
        end_sec=end,
        hook=hook,
        importance=importance,
    )


def _full_video_cut(index: int = 0, start: float = 0.0, end: float = 100.0,
                    importance: int = 8) -> FullVideoCut:
    return FullVideoCut(
        index=index,
        start_word_idx=0,
        end_word_idx=int(end),
        start_sec=start,
        end_sec=end,
        importance=importance,
    )


def _skipped(start_word: int = 0, end_word: int = 5,
             category: str = "hesitation") -> SkippedSegment:
    return SkippedSegment(
        start_word_idx=start_word, end_word_idx=end_word,
        start_sec=start_word * 1.0, end_sec=end_word * 1.0,
        category=category, reason="test",
    )


def _metadata(
    video_type: str = "SOLO",
    language: str = "te-en",
    headline: str = "బండి భగీరథ్‌ కేసులో కీలక మలుపులు",
    summary: str = "English summary text.",
    summary_native: str = "తెలుగు సారాంశం.",
    key_people: list[str] = None,
    key_people_native: list[str] = None,
    key_topics: list[str] = None,
    key_locations: list[str] = None,
    image_queries: list[str] = None,
    marquee: list[str] = None,
) -> Metadata:
    # `is None` check (not `or`) so callers can explicitly pass `[]`
    # to test empty-list scenarios. Falsy fallback would silently
    # substitute defaults for [].
    return Metadata(
        video_type=video_type,
        language=language,
        total_speakers=1,
        overall_summary=summary,
        overall_summary_native=summary_native,
        shorts_headline_native=headline,
        bulletin_marquee_points=marquee if marquee is not None else ["pt1", "pt2", "pt3"],
        image_search_queries=image_queries if image_queries is not None else ["q1", "q2"],
        key_people=key_people if key_people is not None else ["Bandi Bhagirath", "Revanth Reddy"],
        key_people_native=key_people_native if key_people_native is not None else ["బండి భగీరథ్", "రేవంత్ రెడ్డి"],
        key_topics=key_topics if key_topics is not None else ["case", "verdict"],
        key_locations=key_locations if key_locations is not None else ["Hyderabad", "Telangana"],
    )


def _job_output(
    shorts_cuts: list[ShortsCut] = None,
    full_video_cuts: list[FullVideoCut] = None,
    skipped: list[SkippedSegment] = None,
    metadata: Metadata = None,
    image_plan_entries: list[ImagePlanEntry] = None,
) -> JobOutput:
    """Construct a synthetic V2 JobOutput for testing."""
    clean = _clean(30)
    return JobOutput(
        stage_two=StageTwoOutput(
            full_video_cuts=full_video_cuts or [_full_video_cut()],
            skipped_segments=skipped or [_skipped()],
            clean_transcript=clean,
            retake_audit="Skipped 1 hesitation at 0-5s.",
        ),
        canonical_entities=[
            Entity(
                canonical_name="Bandi Bhagirath",
                native_name="బండి భగీరథ్",
                first_mention_word_idx=0,
                type="PERSON",
                mentions=[0],
            ),
        ],
        shorts_cuts=shorts_cuts or [_shorts_cut()],
        metadata=metadata or _metadata(),
        image_plan=ImagePlan(entries=image_plan_entries or []),
    )


def _artifact(idx: int = 0, **overrides) -> ClipRenderArtifacts:
    defaults = {
        "clip_path": f"/abs/clip_{idx:02d}.mp4",
        "raw_path": f"/abs/raw_{idx:02d}.mp4",
        "thumb_path": f"/abs/thumb_{idx:02d}.jpg",
        "image_path": f"/abs/img_{idx:02d}.jpg",
    }
    defaults.update(overrides)
    return ClipRenderArtifacts(**defaults)


def _preset() -> dict:
    return {
        "label": "YouTube Short", "width": 1080, "height": 1920,
        "min_dur": 15, "max_dur": 60, "ideal_dur": 45, "vertical": True,
    }


# ====================================================================== #
# Helpers under test                                                      #
# ====================================================================== #


class TestStripLangSuffix:
    @pytest.mark.parametrize("raw,expected", [
        ("te", "te"),
        ("te-en", "te"),
        ("hi", "hi"),
        ("hi-en", "hi"),
        ("en", "en"),
        ("te-IN", "te"),
        ("zh-Hans-CN", "zh"),
    ])
    def test_strip(self, raw, expected):
        assert _strip_lang_suffix(raw) == expected


class TestResolveLangCfg:
    def test_telugu(self):
        cfg = _resolve_lang_cfg("te")
        assert cfg.code == "te"
        assert cfg.name_english == "Telugu"
        assert cfg.name_native == "తెలుగు"
        assert cfg.script == "Telugu"

    def test_telugu_with_codemix_suffix(self):
        cfg = _resolve_lang_cfg("te-en")
        assert cfg.code == "te"
        assert cfg.name_english == "Telugu"

    def test_hindi(self):
        cfg = _resolve_lang_cfg("hi")
        assert cfg.name_english == "Hindi"
        assert cfg.script == "Devanagari"

    def test_unknown_falls_back_to_telugu(self):
        # V1's languages.get() falls back to Telugu on unknown codes.
        cfg = _resolve_lang_cfg("xyz")
        assert cfg.code == "te"


class TestFormatMmssMmm:
    """D-8.5: MM:SS.mmm always."""

    @pytest.mark.parametrize("seconds,expected", [
        (0.0,     "00:00.000"),
        (1.0,     "00:01.000"),
        (53.3,    "00:53.300"),
        (110.59,  "01:50.590"),
        (491.2,   "08:11.200"),
        (149.83,  "02:29.830"),
        (60.0,    "01:00.000"),
        (3600.0,  "60:00.000"),  # 1hr -- minutes can exceed 60
        (4530.5,  "75:30.500"),  # 75 min 30.5s
    ])
    def test_known_values(self, seconds, expected):
        assert _format_mmss_mmm(seconds) == expected

    def test_floating_point_carry(self):
        # 59.9999 should NOT produce invalid "00:60.000". Rounding to
        # milliseconds first then carrying.
        assert _format_mmss_mmm(59.9999) == "01:00.000"


# ====================================================================== #
# D-8.12: ShortsCut.index contiguity guardrail                            #
# ====================================================================== #


class TestValidateShortsIndicesContiguous:
    def test_in_order_passes(self):
        cuts = [_shorts_cut(0, 0, 20), _shorts_cut(1, 30, 50),
                _shorts_cut(2, 60, 80)]
        # Should not raise
        _validate_shorts_indices_contiguous(cuts)

    def test_out_of_order_passes(self):
        # Out-of-order is FINE -- the adapter sorts.
        cuts = [_shorts_cut(2, 60, 80), _shorts_cut(0, 0, 20),
                _shorts_cut(1, 30, 50)]
        _validate_shorts_indices_contiguous(cuts)

    def test_gap_raises(self):
        # Per user spec: index=[0, 1, 3] -> ValueError mentioning
        # missing index 2.
        cuts = [_shorts_cut(0, 0, 20), _shorts_cut(1, 30, 50),
                _shorts_cut(3, 60, 80)]
        with pytest.raises(ValueError, match=r"missing index 2"):
            _validate_shorts_indices_contiguous(cuts)

    def test_not_starting_at_zero_raises(self):
        # Per user spec: index=[1, 2, 3] -> ValueError mentioning
        # expected 0-start.
        cuts = [_shorts_cut(1, 0, 20), _shorts_cut(2, 30, 50),
                _shorts_cut(3, 60, 80)]
        with pytest.raises(ValueError, match=r"start at 0"):
            _validate_shorts_indices_contiguous(cuts)

    def test_empty_list_passes(self):
        # 0 cuts -> nothing to validate.
        _validate_shorts_indices_contiguous([])

    def test_single_cut_at_zero_passes(self):
        _validate_shorts_indices_contiguous([_shorts_cut(0, 0, 20)])

    def test_single_cut_at_nonzero_raises(self):
        with pytest.raises(ValueError, match=r"start at 0"):
            _validate_shorts_indices_contiguous([_shorts_cut(5, 0, 20)])


# ====================================================================== #
# Shorts adapter -- top-level structure                                   #
# ====================================================================== #


class TestShortsAdapterTopLevelStructure:
    """All 21 top-level keys present, correct types, correct sources."""

    def test_all_keys_present(self):
        out = build_v1_shorts_editor_meta(
            _job_output(),
            video_path="/abs/source.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )
        assert set(out.keys()) == {
            "video_path", "platform", "frame_layout", "language",
            "language_english", "language_native", "script", "preset",
            "video_type", "title_native", "title_telugu", "title_english",
            "summary", "summary_native", "summary_telugu",
            "people", "topics", "keywords", "clips", "created",
        }

    def test_video_path_passthrough(self):
        out = build_v1_shorts_editor_meta(
            _job_output(),
            video_path="/abs/myvideo.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )
        assert out["video_path"] == "/abs/myvideo.mp4"

    def test_platform_passthrough_d82(self):
        # D-8.2: platform string is caller-supplied.
        for plat in ["youtube_short", "instagram_reel",
                     "full_video_shorts_v2"]:
            out = build_v1_shorts_editor_meta(
                _job_output(),
                video_path="/abs/v.mp4",
                platform=plat,
                frame_layout="torn_card",
                preset=_preset(),
                timestamp="20260518_140000",
                clip_artifacts=[_artifact(0)],
            )
            assert out["platform"] == plat

    def test_language_metadata_d84(self):
        # D-8.4: import languages, strip code-mix suffix.
        out = build_v1_shorts_editor_meta(
            _job_output(metadata=_metadata(language="te-en")),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )
        assert out["language"] == "te"
        assert out["language_english"] == "Telugu"
        assert out["language_native"] == "తెలుగు"
        assert out["script"] == "Telugu"

    def test_title_native_from_shorts_headline_native(self):
        meta = _metadata(headline="HEADLINE TEXT")
        out = build_v1_shorts_editor_meta(
            _job_output(metadata=meta),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )
        assert out["title_native"] == "HEADLINE TEXT"
        assert out["title_telugu"] == "HEADLINE TEXT"  # legacy alias

    def test_title_english_default_empty_d83(self):
        # D-8.3: title_english defaults to "".
        out = build_v1_shorts_editor_meta(
            _job_output(),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )
        assert out["title_english"] == ""

    def test_title_english_caller_override_d83(self):
        # D-8.3: caller can pass title_english explicitly.
        out = build_v1_shorts_editor_meta(
            _job_output(),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
            title_english="ENGLISH HEADLINE",
        )
        assert out["title_english"] == "ENGLISH HEADLINE"

    def test_summary_telugu_legacy_alias(self):
        # Legacy alias: summary_telugu == summary_native.
        meta = _metadata(summary_native="తెలుగు సారాంశం")
        out = build_v1_shorts_editor_meta(
            _job_output(metadata=meta),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )
        assert out["summary_telugu"] == "తెలుగు సారాంశం"
        assert out["summary_native"] == "తెలుగు సారాంశం"

    def test_people_only_english_d88(self):
        # D-8.8: drop key_people_native at boundary.
        meta = _metadata(
            key_people=["Modi", "Reddy"],
            key_people_native=["మోదీ", "రెడ్డి"],
        )
        out = build_v1_shorts_editor_meta(
            _job_output(metadata=meta),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )
        assert out["people"] == ["Modi", "Reddy"]
        # No key_people_native field anywhere in the output dict
        assert "key_people_native" not in out

    def test_keywords_always_empty_d87(self):
        # D-8.7: keywords is always []. key_locations gets dropped.
        meta = _metadata(key_locations=["Hyderabad", "Mumbai"])
        out = build_v1_shorts_editor_meta(
            _job_output(metadata=meta),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )
        assert out["keywords"] == []
        # key_locations also dropped
        assert "key_locations" not in out

    def test_topics_pass_through(self):
        meta = _metadata(key_topics=["politics", "verdict"])
        out = build_v1_shorts_editor_meta(
            _job_output(metadata=meta),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )
        assert out["topics"] == ["politics", "verdict"]

    def test_video_type(self):
        meta = _metadata(video_type="PRESS_CONFERENCE")
        out = build_v1_shorts_editor_meta(
            _job_output(metadata=meta),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )
        assert out["video_type"] == "PRESS_CONFERENCE"

    def test_created_timestamp_passthrough(self):
        out = build_v1_shorts_editor_meta(
            _job_output(),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )
        assert out["created"] == "20260518_140000"

    def test_preset_passthrough(self):
        custom_preset = {"label": "Custom", "width": 720, "height": 1280}
        out = build_v1_shorts_editor_meta(
            _job_output(),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=custom_preset,
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )
        assert out["preset"] == custom_preset


# ====================================================================== #
# Shorts adapter -- per-clip dict structure                               #
# ====================================================================== #


class TestShortsAdapterPerClipStructure:
    def test_clip_has_all_24_keys(self):
        out = build_v1_shorts_editor_meta(
            _job_output(),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )
        assert len(out["clips"]) == 1
        clip = out["clips"][0]
        assert set(clip.keys()) == {
            "clip_path", "raw_path", "thumb_path", "image_path",
            "text", "language",
            "title_native", "title_telugu", "title_english",
            "start", "end", "duration",
            "summary", "mood", "importance",
            "video_type", "frame_type",
            "card_params", "split_params", "follow_params", "preset",
            "storage_url", "storage_key", "storage_backend",
        }

    def test_clip_paths_from_artifact(self):
        art = _artifact(
            0,
            clip_path="/x/clip.mp4",
            raw_path="/x/raw.mp4",
            thumb_path="/x/thumb.jpg",
            image_path="/x/img.jpg",
        )
        out = build_v1_shorts_editor_meta(
            _job_output(),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[art],
        )
        clip = out["clips"][0]
        assert clip["clip_path"] == "/x/clip.mp4"
        assert clip["raw_path"] == "/x/raw.mp4"
        assert clip["thumb_path"] == "/x/thumb.jpg"
        assert clip["image_path"] == "/x/img.jpg"

    def test_clip_text_is_per_short_hook(self):
        # Backlog item 99 superseded the original "every clip's text =
        # global headline" contract. Per-short hook is used now.
        meta = _metadata(headline="GLOBAL HEADLINE")
        cuts = [_shorts_cut(0, 10.0, 28.0, hook="unique per-short hook")]
        out = build_v1_shorts_editor_meta(
            _job_output(shorts_cuts=cuts, metadata=meta),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )
        assert out["clips"][0]["text"] == "unique per-short hook"
        # Top-level title_native still has the global headline (job-
        # level field; not per-clip).
        assert out["title_native"] == "GLOBAL HEADLINE"

    def test_clip_start_end_formatted_d85(self):
        # D-8.5: MM:SS.mmm format.
        cuts = [_shorts_cut(0, start=53.3, end=80.5)]
        out = build_v1_shorts_editor_meta(
            _job_output(shorts_cuts=cuts),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )
        clip = out["clips"][0]
        assert clip["start"] == "00:53.300"
        assert clip["end"] == "01:20.500"

    def test_clip_duration_rounded_d86(self):
        # D-8.6: round(end - start, 2).
        cuts = [_shorts_cut(0, start=53.3, end=70.4123456)]
        out = build_v1_shorts_editor_meta(
            _job_output(shorts_cuts=cuts),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )
        # 70.4123456 - 53.3 = 17.1123456 -> round 2 = 17.11
        assert out["clips"][0]["duration"] == 17.11

    def test_clip_summary_is_hook_d89(self):
        # D-8.9: per-clip summary = ShortsCut.hook.
        cuts = [_shorts_cut(0, hook="A punchy hook")]
        out = build_v1_shorts_editor_meta(
            _job_output(shorts_cuts=cuts),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )
        assert out["clips"][0]["summary"] == "A punchy hook"

    def test_clip_mood_always_empty(self):
        out = build_v1_shorts_editor_meta(
            _job_output(),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )
        assert out["clips"][0]["mood"] == ""

    def test_clip_importance_passthrough(self):
        cuts = [_shorts_cut(0, importance=9)]
        out = build_v1_shorts_editor_meta(
            _job_output(shorts_cuts=cuts),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )
        assert out["clips"][0]["importance"] == 9

    def test_clip_storage_fields_d810(self):
        # D-8.10: storage fields from ClipRenderArtifacts.
        art = _artifact(
            0,
            storage_url="https://r2.example/clip.mp4",
            storage_key="clips/20260518/01.mp4",
            storage_backend="r2",
        )
        out = build_v1_shorts_editor_meta(
            _job_output(),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[art],
        )
        clip = out["clips"][0]
        assert clip["storage_url"] == "https://r2.example/clip.mp4"
        assert clip["storage_key"] == "clips/20260518/01.mp4"
        assert clip["storage_backend"] == "r2"

    def test_clip_storage_fields_default_empty(self):
        # ClipRenderArtifacts defaults to empty strings.
        out = build_v1_shorts_editor_meta(
            _job_output(),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )
        clip = out["clips"][0]
        assert clip["storage_url"] == ""
        assert clip["storage_key"] == ""
        assert clip["storage_backend"] == ""

    def test_card_params_default_empty(self):
        out = build_v1_shorts_editor_meta(
            _job_output(),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )
        assert out["clips"][0]["card_params"] == {}
        assert out["clips"][0]["split_params"] == {}
        assert out["clips"][0]["follow_params"] == {}

    def test_card_params_caller_supplied(self):
        cp = {"font_size": 80, "card_c0": "#c10000"}
        out = build_v1_shorts_editor_meta(
            _job_output(),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
            card_params=cp,
        )
        assert out["clips"][0]["card_params"] == cp


# ====================================================================== #
# D-8.12 GUARDRAIL: sorting + contiguous index validation                 #
# ====================================================================== #


class TestShortsAdapterIndexGuardrail:
    """D-8.12: indices must form contiguous 0-based sequence; sort
    by index ASC; raise on gaps or non-zero start.
    """

    def test_indices_out_of_order_get_sorted(self):
        # User-required test name: index=[2, 0, 1] -> adapter produces
        # clips in [0, 1, 2] order
        cuts = [
            _shorts_cut(2, start=60, end=80, hook="C"),
            _shorts_cut(0, start=0,  end=20, hook="A"),
            _shorts_cut(1, start=30, end=50, hook="B"),
        ]
        # Artifacts aligned to SORTED order per caller contract
        arts = [_artifact(0), _artifact(1), _artifact(2)]
        out = build_v1_shorts_editor_meta(
            _job_output(shorts_cuts=cuts),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=arts,
        )
        # Clips emerge in sorted order: hook "A" first, "B" second, "C" third
        hooks_in_order = [c["summary"] for c in out["clips"]]
        assert hooks_in_order == ["A", "B", "C"]

    def test_indices_with_gap_raises(self):
        # User-required test name: index=[0, 1, 3] -> ValueError
        # mentioning missing index 2
        cuts = [_shorts_cut(0, 0, 20), _shorts_cut(1, 30, 50),
                _shorts_cut(3, 60, 80)]
        with pytest.raises(ValueError, match=r"missing index 2"):
            build_v1_shorts_editor_meta(
                _job_output(shorts_cuts=cuts),
                video_path="/abs/v.mp4",
                platform="youtube_short",
                frame_layout="torn_card",
                preset=_preset(),
                timestamp="20260518_140000",
                clip_artifacts=[_artifact(0), _artifact(1), _artifact(2)],
            )

    def test_indices_not_starting_at_zero_raises(self):
        # User-required test name: index=[1, 2, 3] -> ValueError
        # mentioning expected 0-start
        cuts = [_shorts_cut(1, 0, 20), _shorts_cut(2, 30, 50),
                _shorts_cut(3, 60, 80)]
        with pytest.raises(ValueError, match=r"start at 0"):
            build_v1_shorts_editor_meta(
                _job_output(shorts_cuts=cuts),
                video_path="/abs/v.mp4",
                platform="youtube_short",
                frame_layout="torn_card",
                preset=_preset(),
                timestamp="20260518_140000",
                clip_artifacts=[_artifact(0), _artifact(1), _artifact(2)],
            )

    def test_artifact_count_mismatch_raises(self):
        cuts = [_shorts_cut(0, 0, 20), _shorts_cut(1, 30, 50)]
        with pytest.raises(ValueError, match=r"clip_artifacts length"):
            build_v1_shorts_editor_meta(
                _job_output(shorts_cuts=cuts),
                video_path="/abs/v.mp4",
                platform="youtube_short",
                frame_layout="torn_card",
                preset=_preset(),
                timestamp="20260518_140000",
                clip_artifacts=[_artifact(0)],  # only 1 art for 2 cuts
            )


# ====================================================================== #
# Bulletin adapter -- top-level structure                                 #
# ====================================================================== #


class TestBulletinAdapterTopLevelStructure:
    """All 7 top-level keys present, correct types, correct sources."""

    def test_all_keys_present(self):
        out = build_v1_bulletin_editor_meta(
            _job_output(),
            platform="youtube_full",
            bulletin_artifacts=_artifact(0, clip_path="/abs/bulletin.mp4"),
            bulletin_duration_s=735.5,
        )
        assert set(out.keys()) == {
            "render_mode", "platform", "language",
            "stories", "skipped", "duration_s", "clips",
        }

    def test_render_mode_is_bulletin(self):
        out = build_v1_bulletin_editor_meta(
            _job_output(),
            platform="youtube_full",
            bulletin_artifacts=_artifact(0),
            bulletin_duration_s=100.0,
        )
        assert out["render_mode"] == "bulletin"

    def test_platform_passthrough(self):
        out = build_v1_bulletin_editor_meta(
            _job_output(),
            platform="full_video_shorts_v2",
            bulletin_artifacts=_artifact(0),
            bulletin_duration_s=100.0,
        )
        assert out["platform"] == "full_video_shorts_v2"

    def test_language_stripped(self):
        meta = _metadata(language="hi-en")
        out = build_v1_bulletin_editor_meta(
            _job_output(metadata=meta),
            platform="youtube_full",
            bulletin_artifacts=_artifact(0),
            bulletin_duration_s=100.0,
        )
        assert out["language"] == "hi"

    def test_stories_counts_full_video_cuts(self):
        cuts = [_full_video_cut(0), _full_video_cut(1), _full_video_cut(2)]
        out = build_v1_bulletin_editor_meta(
            _job_output(full_video_cuts=cuts),
            platform="youtube_full",
            bulletin_artifacts=_artifact(0),
            bulletin_duration_s=100.0,
        )
        assert out["stories"] == 3

    def test_skipped_counts_skipped_segments(self):
        skips = [_skipped(0, 2), _skipped(5, 7), _skipped(10, 12)]
        out = build_v1_bulletin_editor_meta(
            _job_output(skipped=skips),
            platform="youtube_full",
            bulletin_artifacts=_artifact(0),
            bulletin_duration_s=100.0,
        )
        assert out["skipped"] == 3

    def test_duration_s_passthrough(self):
        out = build_v1_bulletin_editor_meta(
            _job_output(),
            platform="youtube_full",
            bulletin_artifacts=_artifact(0),
            bulletin_duration_s=735.508333,
        )
        assert out["duration_s"] == 735.508333


# ====================================================================== #
# Bulletin adapter -- per-clip dict structure                             #
# ====================================================================== #


class TestBulletinAdapterClipStructure:
    """Bulletin shape has ONE clip with 14 keys."""

    def test_exactly_one_clip(self):
        out = build_v1_bulletin_editor_meta(
            _job_output(),
            platform="youtube_full",
            bulletin_artifacts=_artifact(0),
            bulletin_duration_s=100.0,
        )
        assert len(out["clips"]) == 1

    def test_bulletin_clip_has_all_keys(self):
        out = build_v1_bulletin_editor_meta(
            _job_output(),
            platform="youtube_full",
            bulletin_artifacts=_artifact(0),
            bulletin_duration_s=100.0,
        )
        clip = out["clips"][0]
        assert set(clip.keys()) == {
            "clip_path", "thumb_path", "image_path", "duration",
            "frame_type", "text", "sentiment", "entities",
            "card_params", "section_pct", "follow_params",
            "storage_url", "storage_key", "storage_backend",
            "clip_path_overlay", "clip_path_carousel_only",
        }

    def test_bulletin_clip_text_is_truncated_500(self):
        # V1's truncation point is 500 chars (pipeline.py:4362).
        long_native = "abcd" * 200   # 800 chars
        meta = _metadata(summary_native=long_native)
        out = build_v1_bulletin_editor_meta(
            _job_output(metadata=meta),
            platform="youtube_full",
            bulletin_artifacts=_artifact(0),
            bulletin_duration_s=100.0,
        )
        assert out["clips"][0]["text"] == "abcd" * 125   # exactly 500 chars

    def test_bulletin_clip_entities_from_key_people(self):
        # V1's bulletin shape calls it "entities" but it's people.
        meta = _metadata(key_people=["A", "B", "C"])
        out = build_v1_bulletin_editor_meta(
            _job_output(metadata=meta),
            platform="youtube_full",
            bulletin_artifacts=_artifact(0),
            bulletin_duration_s=100.0,
        )
        assert out["clips"][0]["entities"] == ["A", "B", "C"]

    def test_bulletin_clip_frame_type_is_bulletin(self):
        out = build_v1_bulletin_editor_meta(
            _job_output(),
            platform="youtube_full",
            bulletin_artifacts=_artifact(0),
            bulletin_duration_s=100.0,
        )
        assert out["clips"][0]["frame_type"] == "bulletin"

    def test_bulletin_clip_overlay_paths_passthrough(self):
        art = ClipRenderArtifacts(
            clip_path="/abs/bulletin_overlay.mp4",
            clip_path_overlay="/abs/bulletin_overlay.mp4",
            clip_path_carousel_only="/abs/bulletin_carousel.mp4",
        )
        out = build_v1_bulletin_editor_meta(
            _job_output(),
            platform="youtube_full",
            bulletin_artifacts=art,
            bulletin_duration_s=100.0,
        )
        clip = out["clips"][0]
        assert clip["clip_path_overlay"] == "/abs/bulletin_overlay.mp4"
        assert clip["clip_path_carousel_only"] == "/abs/bulletin_carousel.mp4"

    def test_bulletin_clip_overlay_paths_default_empty(self):
        out = build_v1_bulletin_editor_meta(
            _job_output(),
            platform="youtube_full",
            bulletin_artifacts=_artifact(0),
            bulletin_duration_s=100.0,
        )
        clip = out["clips"][0]
        assert clip["clip_path_overlay"] == ""
        assert clip["clip_path_carousel_only"] == ""


# ====================================================================== #
# D-8.11: V2-internal fields NOT surfaced in editor_meta.json             #
# ====================================================================== #


class TestV2InternalFieldsNotSurfaced:
    """Pin the dropped-fields contract from D-8.11. If a future
    maintainer accidentally adds any of these to editor_meta_adapter,
    these tests catch it.
    """

    def _build_shorts(self):
        return build_v1_shorts_editor_meta(
            _job_output(),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )

    def _build_bulletin(self):
        return build_v1_bulletin_editor_meta(
            _job_output(),
            platform="youtube_full",
            bulletin_artifacts=_artifact(0),
            bulletin_duration_s=100.0,
        )

    def test_shorts_no_image_plan(self):
        assert "image_plan" not in self._build_shorts()

    def test_bulletin_no_image_plan(self):
        assert "image_plan" not in self._build_bulletin()

    def test_shorts_no_clean_transcript(self):
        out = self._build_shorts()
        assert "clean_transcript" not in out
        assert "source_word_map" not in out
        assert "clip_boundaries" not in out

    def test_shorts_no_skipped_segments(self):
        assert "skipped_segments" not in self._build_shorts()

    def test_shorts_no_retake_audit(self):
        assert "retake_audit" not in self._build_shorts()

    def test_shorts_no_canonical_entities(self):
        # The full Entity objects are not surfaced (only English names
        # via "people").
        out = self._build_shorts()
        assert "canonical_entities" not in out
        assert "entities" not in out  # shorts shape doesn't have this

    def test_shorts_no_full_video_cuts(self):
        # The shorts adapter ignores full_video_cuts entirely
        # (bulletin adapter uses them for the "stories" count only).
        assert "full_video_cuts" not in self._build_shorts()

    def test_shorts_no_image_search_queries(self):
        assert "image_search_queries" not in self._build_shorts()

    def test_shorts_no_bulletin_marquee_points(self):
        assert "bulletin_marquee_points" not in self._build_shorts()

    def test_shorts_no_total_speakers(self):
        assert "total_speakers" not in self._build_shorts()

    def test_bulletin_no_clean_transcript(self):
        assert "clean_transcript" not in self._build_bulletin()


# ====================================================================== #
# D-8.13: empty-list emission                                             #
# ====================================================================== #


class TestEmptyListEmission:
    """D-8.13: empty lists are always [], never null, never missing."""

    def test_empty_key_people_emits_empty_list(self):
        meta = _metadata(key_people=[], key_people_native=[])
        out = build_v1_shorts_editor_meta(
            _job_output(metadata=meta),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )
        assert out["people"] == []   # not None, not missing

    def test_empty_key_topics_emits_empty_list(self):
        meta = _metadata(key_topics=[])
        out = build_v1_shorts_editor_meta(
            _job_output(metadata=meta),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )
        assert out["topics"] == []

    def test_keywords_always_empty(self):
        out = build_v1_shorts_editor_meta(
            _job_output(),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )
        # Always [], even when V2 has key_locations.
        assert out["keywords"] == []

    def test_bulletin_empty_entities_emits_empty_list(self):
        meta = _metadata(key_people=[])
        out = build_v1_bulletin_editor_meta(
            _job_output(metadata=meta),
            platform="youtube_full",
            bulletin_artifacts=_artifact(0),
            bulletin_duration_s=100.0,
        )
        assert out["clips"][0]["entities"] == []


# ======================================================================
# Backlog item 99: per-short card text comes from cut.hook
# ======================================================================


class TestPerShortText:
    """Each short's burned-in card text must come from its own
    ShortsCut.hook (Stage 3a produces a distinct 3-10 word hook for
    every cut). The previous adapter burned the GLOBAL
    Metadata.shorts_headline_native onto every short, making all 8
    cards identical (job 40 surfaced the issue empirically).
    """

    def test_each_short_uses_its_own_hook_for_text(self):
        # 4 shorts each with a distinct hook.
        cuts = [
            _shorts_cut(0, 10.0, 28.0, hook="First moment hook"),
            _shorts_cut(1, 40.0, 58.0, hook="Second moment hook"),
            _shorts_cut(2, 70.0, 88.0, hook="Third moment hook"),
            _shorts_cut(3, 100.0, 118.0, hook="Fourth moment hook"),
        ]
        meta = _metadata(headline="GLOBAL HEADLINE — must not appear on per-clip text")
        out = build_v1_shorts_editor_meta(
            _job_output(shorts_cuts=cuts, metadata=meta),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(i) for i in range(4)],
        )
        texts = [c["text"] for c in out["clips"]]
        # Each clip's text matches its own hook -- not the global headline.
        assert texts == [
            "First moment hook",
            "Second moment hook",
            "Third moment hook",
            "Fourth moment hook",
        ]
        # Top-level title_native is still the GLOBAL headline (job-level
        # V1 backwards-compat — see backlog 99).
        assert out["title_native"] == "GLOBAL HEADLINE — must not appear on per-clip text"

    def test_title_native_and_title_telugu_also_use_hook(self):
        cuts = [
            _shorts_cut(0, 10.0, 28.0, hook="HOOK ONE"),
            _shorts_cut(1, 40.0, 58.0, hook="HOOK TWO"),
        ]
        out = build_v1_shorts_editor_meta(
            _job_output(shorts_cuts=cuts),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0), _artifact(1)],
        )
        # title_native + title_telugu (legacy alias) -- both derived from
        # cut.hook so the V1 editor surfaces per-clip headlines wherever
        # it reads either field.
        assert out["clips"][0]["title_native"] == "HOOK ONE"
        assert out["clips"][0]["title_telugu"] == "HOOK ONE"
        assert out["clips"][1]["title_native"] == "HOOK TWO"
        assert out["clips"][1]["title_telugu"] == "HOOK TWO"

    def test_empty_hook_falls_back_to_global_headline(self):
        # Defensive: if Stage 3a ever ships a cut with an empty hook
        # (shouldn't happen on valid output -- Pydantic doesn't enforce
        # min_length on hook), fall back to the global title so the
        # card isn't blank.
        cuts = [
            _shorts_cut(0, 10.0, 28.0, hook=""),
            _shorts_cut(1, 40.0, 58.0, hook="real hook"),
        ]
        meta = _metadata(headline="FALLBACK HEADLINE")
        out = build_v1_shorts_editor_meta(
            _job_output(shorts_cuts=cuts, metadata=meta),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0), _artifact(1)],
        )
        assert out["clips"][0]["text"] == "FALLBACK HEADLINE"
        assert out["clips"][1]["text"] == "real hook"

    def test_summary_still_carries_hook_too(self):
        # The 'summary' field has been hook-based since D-8.9; ensure
        # the item-99 text change didn't disturb it.
        cuts = [_shorts_cut(0, 10.0, 28.0, hook="THE HOOK")]
        out = build_v1_shorts_editor_meta(
            _job_output(shorts_cuts=cuts),
            video_path="/abs/v.mp4",
            platform="youtube_short",
            frame_layout="torn_card",
            preset=_preset(),
            timestamp="20260518_140000",
            clip_artifacts=[_artifact(0)],
        )
        assert out["clips"][0]["summary"] == "THE HOOK"
        # text == summary == hook -- they're allowed to coincide.
        assert out["clips"][0]["text"] == "THE HOOK"
