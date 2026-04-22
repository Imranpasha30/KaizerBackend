"""
test_encode_args.py — Smoke tests for pipeline_core.pipeline.ENCODE_ARGS_SHORT_FORM

The Builder is replacing the scattered `-c:v libx264 -preset fast -crf 20`
blocks with a single ENCODE_ARGS_SHORT_FORM constant. These tests verify its
content and structure.

Required to contain:
  -b:v 8M  -maxrate 10M  -bufsize 16M
  -pix_fmt yuv420p
  -color_primaries bt709
  -af loudnorm=I=-14:TP=-1.5:LRA=11
  -movflags +faststart
  AAC codec at 48 kHz
"""
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Import guard — blocked on Builder
# ---------------------------------------------------------------------------
try:
    from pipeline_core.pipeline import ENCODE_ARGS_SHORT_FORM
    _CONSTANT_AVAILABLE = True
except ImportError:
    _CONSTANT_AVAILABLE = False
    ENCODE_ARGS_SHORT_FORM = None

pytestmark = pytest.mark.skipif(
    not _CONSTANT_AVAILABLE,
    reason="pipeline_core.pipeline.ENCODE_ARGS_SHORT_FORM not yet added (blocked on Builder)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _args_as_string(args: list) -> str:
    """Join list to a single string for substring searching."""
    return " ".join(str(a) for a in args)


def _consecutive_pair(args: list, key: str, value: str) -> bool:
    """Return True if key immediately precedes value in args (as separate elements)."""
    for i, arg in enumerate(args):
        if str(arg) == key and i + 1 < len(args) and str(args[i + 1]) == value:
            return True
    return False


# ===========================================================================
# 1. test_encode_args_constant_exists
# ===========================================================================

def test_encode_args_constant_exists():
    """ENCODE_ARGS_SHORT_FORM must be importable from pipeline_core.pipeline."""
    assert ENCODE_ARGS_SHORT_FORM is not None, (
        "ENCODE_ARGS_SHORT_FORM must exist and not be None"
    )


# ===========================================================================
# 2. test_encode_args_is_flat_list_of_strings
# ===========================================================================

def test_encode_args_is_flat_list_of_strings():
    """ENCODE_ARGS_SHORT_FORM must be a flat list (or tuple) of strings."""
    assert isinstance(ENCODE_ARGS_SHORT_FORM, (list, tuple)), (
        f"ENCODE_ARGS_SHORT_FORM must be a list or tuple, got {type(ENCODE_ARGS_SHORT_FORM)}"
    )
    for i, item in enumerate(ENCODE_ARGS_SHORT_FORM):
        assert isinstance(item, str), (
            f"Element [{i}] must be a string, got {type(item)!r}: {item!r}"
        )


# ===========================================================================
# 3. test_encode_args_includes_bitrate_cap
# ===========================================================================

def test_encode_args_includes_bitrate_cap():
    """Must contain -b:v, -maxrate, and -bufsize flags."""
    args = list(ENCODE_ARGS_SHORT_FORM)
    flat = _args_as_string(args)

    assert "-b:v" in args, f"Missing '-b:v' in ENCODE_ARGS_SHORT_FORM. Got: {flat}"
    assert "-maxrate" in args, f"Missing '-maxrate' in ENCODE_ARGS_SHORT_FORM. Got: {flat}"
    assert "-bufsize" in args, f"Missing '-bufsize' in ENCODE_ARGS_SHORT_FORM. Got: {flat}"


def test_encode_args_bitrate_values():
    """The bitrate values must be 8M, maxrate 10M, bufsize 16M per spec."""
    args = list(ENCODE_ARGS_SHORT_FORM)
    assert _consecutive_pair(args, "-b:v", "8M"), (
        f"Expected '-b:v 8M'. Args: {args}"
    )
    assert _consecutive_pair(args, "-maxrate", "10M"), (
        f"Expected '-maxrate 10M'. Args: {args}"
    )
    assert _consecutive_pair(args, "-bufsize", "16M"), (
        f"Expected '-bufsize 16M'. Args: {args}"
    )


# ===========================================================================
# 4. test_encode_args_includes_color_norm
# ===========================================================================

def test_encode_args_includes_pix_fmt_yuv420p():
    """Must contain -pix_fmt yuv420p."""
    args = list(ENCODE_ARGS_SHORT_FORM)
    assert _consecutive_pair(args, "-pix_fmt", "yuv420p"), (
        f"Expected '-pix_fmt yuv420p'. Args: {args}"
    )


def test_encode_args_includes_color_primaries_bt709():
    """Must contain -color_primaries bt709."""
    args = list(ENCODE_ARGS_SHORT_FORM)
    assert _consecutive_pair(args, "-color_primaries", "bt709"), (
        f"Expected '-color_primaries bt709'. Args: {args}"
    )


# ===========================================================================
# 5. test_encode_args_includes_loudnorm
# ===========================================================================

def test_encode_args_includes_loudnorm():
    """Must contain an -af flag with loudnorm=I=-14:TP=-1.5:LRA=11."""
    args = list(ENCODE_ARGS_SHORT_FORM)
    # -af may appear as a separate flag with the filter string as the next element
    flat = _args_as_string(args)

    assert "-af" in args, f"Missing '-af' flag in ENCODE_ARGS_SHORT_FORM. Got: {flat}"

    af_idx = args.index("-af")
    af_value = args[af_idx + 1] if af_idx + 1 < len(args) else ""

    assert "loudnorm" in af_value, (
        f"'-af' value must contain 'loudnorm'. Got: {af_value!r}"
    )
    assert "I=-14" in af_value, (
        f"loudnorm must set I=-14. Got af value: {af_value!r}"
    )
    assert "TP=-1.5" in af_value, (
        f"loudnorm must set TP=-1.5. Got af value: {af_value!r}"
    )
    assert "LRA=11" in af_value, (
        f"loudnorm must set LRA=11. Got af value: {af_value!r}"
    )


# ===========================================================================
# 6. test_encode_args_includes_faststart
# ===========================================================================

def test_encode_args_includes_faststart():
    """Must contain -movflags +faststart for web streaming."""
    args = list(ENCODE_ARGS_SHORT_FORM)
    flat = _args_as_string(args)

    assert "-movflags" in args, (
        f"Missing '-movflags' in ENCODE_ARGS_SHORT_FORM. Got: {flat}"
    )

    movflags_idx = args.index("-movflags")
    movflags_value = args[movflags_idx + 1] if movflags_idx + 1 < len(args) else ""

    assert "faststart" in movflags_value, (
        f"'-movflags' value must contain 'faststart'. Got: {movflags_value!r}"
    )


# ===========================================================================
# 7. test_encode_args_includes_aac_48khz
# ===========================================================================

def test_encode_args_includes_aac_48khz():
    """Must contain AAC audio codec and 48000 Hz sample rate."""
    args = list(ENCODE_ARGS_SHORT_FORM)
    flat = _args_as_string(args)

    # Audio codec: -c:a aac
    assert _consecutive_pair(args, "-c:a", "aac"), (
        f"Expected '-c:a aac'. Args: {args}"
    )

    # Sample rate: -ar 48000
    assert _consecutive_pair(args, "-ar", "48000"), (
        f"Expected '-ar 48000'. Args: {args}"
    )


# ===========================================================================
# 8. test_encode_args_no_crf_or_preset_fast_regression
# ===========================================================================

def test_encode_args_no_crf_or_preset_fast_regression():
    """
    The old ad-hoc pattern '-crf 20' / '-preset fast' should NOT be the primary
    quality control in ENCODE_ARGS_SHORT_FORM; bitrate caps replace them.
    This is a regression guard: if -b:v is present, -crf should not contradict it
    (they can coexist but -b:v must be there).
    """
    args = list(ENCODE_ARGS_SHORT_FORM)
    assert "-b:v" in args, (
        "ENCODE_ARGS_SHORT_FORM must use explicit bitrate (-b:v) not just CRF"
    )
