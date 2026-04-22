"""
tests/test_variants.py
=======================
pytest coverage for pipeline_core/variants.py

Fast tests mock all heavy I/O (ffmpeg subprocess, validate_input, apply_cta,
validate_output, score_loop, _originality_delta) so they never touch real disk.

Slow test (gated by @pytest.mark.slow) exercises the full pipeline end-to-end
on the valid_short_mp4 fixture.
"""
from __future__ import annotations

import dataclasses
import os
from subprocess import CompletedProcess
from unittest.mock import MagicMock

import pytest

from pipeline_core.variants import (
    PLATFORM_VARIANT_SPECS,
    PlatformVariant,
    _originality_delta,
    generate_variants,
)
from pipeline_core.loop_score import LoopScore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok_validation_result():
    """Minimal mock of ValidationResult(ok=True)."""
    m = MagicMock()
    m.ok = True
    m.errors = []
    m.warnings = []
    return m


def _ok_qa_result():
    """Minimal mock of QAResult(ok=True)."""
    m = MagicMock()
    m.ok = True
    m.errors = []
    m.warnings = []
    return m


def _ok_cta_result(style: str = "soft_follow"):
    """Minimal mock of CTAResult."""
    m = MagicMock()
    m.cta_style = style
    m.warnings = []
    return m


def _ok_loop_score() -> LoopScore:
    return LoopScore(
        overall=75.0,
        visual_phash_distance=5,
        audio_xcorr=0.7,
        motion_continuity=0.8,
        suggestions=[],
    )


def _completed_process(returncode: int = 0) -> CompletedProcess:
    return CompletedProcess(args=[], returncode=returncode, stdout="", stderr="")


def _patch_full_pipeline(mocker, *, cta_style: str = "soft_follow",
                         loop_score_result: LoopScore | None = None,
                         originality: float = 0.5):
    """Patch every external dependency used by generate_variants.
    Returns a namespace of mock objects for assertions.
    """
    import shutil
    mock_validate_input = mocker.patch(
        "pipeline_core.variants.validate_input",
        return_value=_ok_validation_result(),
    )
    mock_subprocess = mocker.patch(
        "pipeline_core.variants.subprocess.run",
        return_value=_completed_process(0),
    )
    # _apply_caption_overlay calls subprocess internally — but it's called via
    # the module's internal _apply_caption_overlay function which in turn calls
    # subprocess.run (already patched above) and render_caption.
    mock_render_caption = mocker.patch(
        "pipeline_core.variants.render_caption",
    )
    mock_caption_result = MagicMock()
    mock_caption_result.image = MagicMock()
    mock_caption_result.image.save = MagicMock()
    mock_caption_result.width = 200
    mock_caption_result.height = 60
    mock_caption_result.warnings = []
    mock_render_caption.return_value = mock_caption_result

    mock_apply_cta = mocker.patch(
        "pipeline_core.variants.apply_cta",
        return_value=_ok_cta_result(cta_style),
    )
    mock_score_loop = mocker.patch(
        "pipeline_core.variants.score_loop",
        return_value=loop_score_result or _ok_loop_score(),
    )
    mock_validate_output = mocker.patch(
        "pipeline_core.variants.validate_output",
        return_value=_ok_qa_result(),
    )
    mock_originality = mocker.patch(
        "pipeline_core.variants._originality_delta",
        return_value=originality,
    )
    # Patch shutil.copy2 so we never need actual files
    mocker.patch("pipeline_core.variants._shutil" if hasattr(
        __import__("pipeline_core.variants", fromlist=["_shutil"]), "_shutil"
    ) else "shutil.copy2", create=True)

    # Also patch the internal shutil usage inside generate_variants (imported as _shutil)
    mocker.patch("shutil.copy2", return_value=None)
    mocker.patch("shutil.rmtree", return_value=None)

    # Patch _probe_fps and _probe_bitrate_kbps so they don't call ffprobe
    mocker.patch("pipeline_core.variants._probe_fps", return_value=30.0)
    mocker.patch("pipeline_core.variants._probe_bitrate_kbps", return_value=7500.0)

    class _NS:
        validate_input = mock_validate_input
        subprocess_run = mock_subprocess
        apply_cta = mock_apply_cta
        score_loop = mock_score_loop
        validate_output = mock_validate_output
        originality_delta = mock_originality

    return _NS()


# ===========================================================================
# SECTION A — PLATFORM_VARIANT_SPECS structure tests
# ===========================================================================

class TestPlatformVariantSpecs:
    def test_platform_variant_specs_has_three_platforms(self):
        """Exactly three platforms must be registered: youtube_short,
        instagram_reel, tiktok."""
        keys = set(PLATFORM_VARIANT_SPECS.keys())
        assert keys == {"youtube_short", "instagram_reel", "tiktok"}, (
            f"Expected exactly {{youtube_short, instagram_reel, tiktok}}, got {keys}"
        )

    def test_instagram_reel_requires_loop(self):
        """instagram_reel must have requires_loop=True (Reels loop rewards)."""
        spec = PLATFORM_VARIANT_SPECS["instagram_reel"]
        assert spec.requires_loop is True, (
            f"instagram_reel must require loop, got requires_loop={spec.requires_loop}"
        )

    def test_youtube_and_tiktok_do_not_require_loop(self):
        """youtube_short and tiktok must have requires_loop=False."""
        for platform in ("youtube_short", "tiktok"):
            spec = PLATFORM_VARIANT_SPECS[platform]
            assert spec.requires_loop is False, (
                f"{platform} must not require loop, got requires_loop={spec.requires_loop}"
            )

    def test_youtube_and_instagram_safe_zones_differ(self):
        """YouTube Short and Instagram Reel must have different safe_zone tuples."""
        yt = PLATFORM_VARIANT_SPECS["youtube_short"].safe_zone
        ig = PLATFORM_VARIANT_SPECS["instagram_reel"].safe_zone
        assert yt != ig, (
            f"youtube_short and instagram_reel safe_zones must differ. "
            f"YT={yt}, IG={ig}"
        )

    def test_ig_right_safe_boundary_narrower_than_yt(self):
        """IG safe_zone[2] < YT safe_zone[2]: Instagram right boundary is tighter
        (more conservative), leaving wider margins for UI chrome."""
        yt_right = PLATFORM_VARIANT_SPECS["youtube_short"].safe_zone[2]
        ig_right = PLATFORM_VARIANT_SPECS["instagram_reel"].safe_zone[2]
        assert ig_right < yt_right, (
            f"IG right safe boundary ({ig_right}) must be < YT right boundary ({yt_right})"
        )

    def test_cta_text_defaults_are_distinct(self):
        """All three platforms must have distinct CTA text defaults."""
        defaults = [
            PLATFORM_VARIANT_SPECS[p].cta_text_default
            for p in ("youtube_short", "instagram_reel", "tiktok")
        ]
        assert len(set(defaults)) == 3, (
            f"All three CTA text defaults must be distinct. Got: {defaults}"
        )

    def test_bitrate_specs_within_platform_envelopes(self):
        """Platform bitrates must fall within documented acceptable ranges."""
        yt_bitrate = PLATFORM_VARIANT_SPECS["youtube_short"].bitrate_kbps
        ig_bitrate = PLATFORM_VARIANT_SPECS["instagram_reel"].bitrate_kbps
        tt_bitrate = PLATFORM_VARIANT_SPECS["tiktok"].bitrate_kbps

        assert 4000 <= yt_bitrate <= 20000, (
            f"youtube_short bitrate {yt_bitrate} must be in [4000, 20000] kbps"
        )
        assert 5000 <= ig_bitrate <= 10000, (
            f"instagram_reel bitrate {ig_bitrate} must be in [5000, 10000] kbps"
        )
        assert 2000 <= tt_bitrate <= 10000, (
            f"tiktok bitrate {tt_bitrate} must be in [2000, 10000] kbps"
        )


# ===========================================================================
# SECTION B — generate_variants() behaviour tests
# ===========================================================================

class TestGenerateVariantsBehaviour:
    def test_empty_platforms_returns_empty_list(self, mocker, tmp_path):
        """platforms=[] must return [] without calling any pipeline step."""
        result = generate_variants(
            str(tmp_path / "master.mp4"),
            platforms=[],
            output_dir=str(tmp_path / "out"),
        )
        assert result == [], f"Empty platforms must return [], got {result}"

    def test_invalid_platform_raises_valueerror(self, tmp_path):
        """An unrecognised platform name must raise ValueError mentioning 'supported'."""
        with pytest.raises(ValueError, match=r"(?i)supported"):
            generate_variants(
                str(tmp_path / "master.mp4"),
                platforms=["snapchat"],
                output_dir=str(tmp_path / "out"),
            )

    def test_returns_in_input_order(self, mocker, tmp_path):
        """Results must be returned in the same order as the input platforms list."""
        ns = _patch_full_pipeline(mocker)
        os.makedirs(str(tmp_path / "out"), exist_ok=True)

        result = generate_variants(
            str(tmp_path / "master.mp4"),
            platforms=["tiktok", "youtube_short"],
            output_dir=str(tmp_path / "out"),
        )

        assert len(result) == 2, f"Expected 2 results, got {len(result)}"
        assert result[0].platform == "tiktok", (
            f"First result must be 'tiktok', got {result[0].platform}"
        )
        assert result[1].platform == "youtube_short", (
            f"Second result must be 'youtube_short', got {result[1].platform}"
        )

    def test_cta_text_override_reaches_apply_cta(self, mocker, tmp_path):
        """When cta_text_override provides a platform key, apply_cta must be called
        with that text for the overridden platform and the default text for others."""
        ns = _patch_full_pipeline(mocker)
        os.makedirs(str(tmp_path / "out"), exist_ok=True)

        custom_ig_text = "CUSTOM_IG_CTA_X"

        generate_variants(
            str(tmp_path / "master.mp4"),
            platforms=["instagram_reel", "youtube_short"],
            output_dir=str(tmp_path / "out"),
            cta_text_override={"instagram_reel": custom_ig_text},
        )

        assert ns.apply_cta.call_count == 2, (
            f"apply_cta must be called once per platform, got {ns.apply_cta.call_count}"
        )

        # Check the instagram_reel call used the override text
        ig_calls = [
            call for call in ns.apply_cta.call_args_list
            if call.kwargs.get("platform") == "instagram_reel"
               or (call.args and "instagram_reel" in str(call.args))
               or custom_ig_text in str(call)
        ]
        all_texts = [str(call) for call in ns.apply_cta.call_args_list]
        assert any(custom_ig_text in t for t in all_texts), (
            f"Custom IG CTA text '{custom_ig_text}' not found in apply_cta calls. "
            f"Calls: {all_texts}"
        )

        # The youtube_short call must NOT use the custom IG text
        yt_default = PLATFORM_VARIANT_SPECS["youtube_short"].cta_text_default
        yt_calls_str = [
            str(call) for call in ns.apply_cta.call_args_list
            if yt_default in str(call)
        ]
        assert yt_calls_str, (
            f"youtube_short must use its default CTA '{yt_default}'. "
            f"Calls: {all_texts}"
        )

    def test_ig_variant_gets_loop_score(self, mocker, tmp_path):
        """For platforms=['instagram_reel'], result[0].loop_score must be the
        LoopScore returned by the mocked score_loop."""
        expected_ls = _ok_loop_score()
        ns = _patch_full_pipeline(mocker, loop_score_result=expected_ls)
        os.makedirs(str(tmp_path / "out"), exist_ok=True)

        result = generate_variants(
            str(tmp_path / "master.mp4"),
            platforms=["instagram_reel"],
            output_dir=str(tmp_path / "out"),
        )

        assert len(result) == 1
        ls = result[0].loop_score
        assert ls is not None, "instagram_reel variant must have a loop_score (not None)"
        assert isinstance(ls, LoopScore), (
            f"loop_score must be a LoopScore instance, got {type(ls)}"
        )
        assert ls.overall == expected_ls.overall, (
            f"loop_score.overall must match mock return: "
            f"expected {expected_ls.overall}, got {ls.overall}"
        )

    def test_non_ig_variant_loop_score_is_none(self, mocker, tmp_path):
        """For platforms=['youtube_short'], loop_score must be None (no loop required)."""
        ns = _patch_full_pipeline(mocker)
        os.makedirs(str(tmp_path / "out"), exist_ok=True)

        result = generate_variants(
            str(tmp_path / "master.mp4"),
            platforms=["youtube_short"],
            output_dir=str(tmp_path / "out"),
        )

        assert len(result) == 1
        assert result[0].loop_score is None, (
            f"youtube_short must have loop_score=None, got {result[0].loop_score}"
        )
        ns.score_loop.assert_not_called()

    def test_low_originality_adds_warning(self, mocker, tmp_path):
        """When _originality_delta returns 0.10 (< 0.15 threshold), a warning
        containing 'originality' must appear in the variant's warnings."""
        _patch_full_pipeline(mocker, originality=0.10)
        os.makedirs(str(tmp_path / "out"), exist_ok=True)

        result = generate_variants(
            str(tmp_path / "master.mp4"),
            platforms=["youtube_short"],
            output_dir=str(tmp_path / "out"),
        )

        assert len(result) == 1
        originality_warnings = [
            w for w in result[0].warnings
            if "originality" in w.lower()
        ]
        assert originality_warnings, (
            f"Expected an 'originality' warning when score=0.10 < 0.15. "
            f"Got warnings: {result[0].warnings}"
        )


# ===========================================================================
# SECTION C — _originality_delta() unit tests
# ===========================================================================

class TestOriginalityDelta:
    def test_originality_delta_returns_float(self, mocker, tmp_path):
        """_originality_delta must return a float even when cv2 is mocked."""
        import cv2
        import numpy as np

        fake_frame = np.zeros((256, 256, 3), dtype=np.float32)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30
        mock_cap.read.return_value = (True, np.zeros((256, 256, 3), dtype=np.uint8))
        mocker.patch("pipeline_core.variants.cv2.VideoCapture", return_value=mock_cap)

        result = _originality_delta(str(tmp_path / "v.mp4"), str(tmp_path / "m.mp4"))

        assert isinstance(result, float), (
            f"_originality_delta must return float, got {type(result)}"
        )
        assert 0.0 <= result <= 1.0, (
            f"_originality_delta result must be in [0,1], got {result}"
        )

    def test_originality_delta_identical_files_near_zero(self, mocker, tmp_path):
        """Comparing a file to itself should yield a very low originality score
        (near zero: identical frames → mean abs diff = 0)."""
        import numpy as np

        # Both variant and master return identical frames
        identical_frame = np.full((256, 256, 3), 128, dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100  # 100 total frames
        mock_cap.read.return_value = (True, identical_frame)

        mocker.patch("pipeline_core.variants.cv2.VideoCapture", return_value=mock_cap)

        result = _originality_delta(
            str(tmp_path / "same.mp4"),
            str(tmp_path / "same.mp4"),
        )

        assert isinstance(result, float), f"Must return float, got {type(result)}"
        assert result < 0.05, (
            f"Identical frames should yield near-zero originality, got {result}"
        )


# ===========================================================================
# SECTION D — Slow end-to-end test
# ===========================================================================

@pytest.mark.slow
def test_generate_three_variants_end_to_end(valid_short_mp4, tmp_path):
    """Run generate_variants on the real 15s fixture for all 3 platforms.
    Assert 3 MP4 output files exist.
    """
    from pipeline_core.qa import validate_output

    output_dir = str(tmp_path / "variants_out")
    os.makedirs(output_dir, exist_ok=True)

    results = generate_variants(
        valid_short_mp4,
        platforms=["youtube_short", "instagram_reel", "tiktok"],
        output_dir=output_dir,
    )

    assert len(results) == 3, (
        f"Expected 3 PlatformVariant results, got {len(results)}"
    )

    for variant in results:
        assert isinstance(variant, PlatformVariant), (
            f"Each result must be PlatformVariant, got {type(variant)}"
        )
        assert os.path.isfile(variant.output_path), (
            f"Output MP4 must exist for platform {variant.platform!r}: "
            f"{variant.output_path}"
        )
        # Verify the file is readable by QA
        qa_result = validate_output(variant.output_path, platform=variant.platform)
        assert isinstance(qa_result.ok, bool), (
            f"validate_output must return a QAResult with ok field "
            f"for platform {variant.platform!r}"
        )
