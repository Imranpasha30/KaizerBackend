"""
test_integration_smoke.py — End-to-end smoke test for Phase 1 pipeline.

Marked @pytest.mark.slow — excluded from default fast runs:
    pytest -m "not slow"

What this test does:
  1. Generates a 5-second 1080x1920 source with FFmpeg.
  2. Runs validator → expects OK.
  3. Re-encodes using ENCODE_ARGS_SHORT_FORM via raw subprocess FFmpeg call.
  4. Runs QA on output → expects OK with measurements populated.
  5. Asserts duration ≈ 5s, lufs ≈ -14±1, pix_fmt == 'yuv420p'.

This test deliberately does NOT mock anything — it proves the full stack
from validator → encode → QA works with real files.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Slow marker + import guards
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.slow

_VALIDATOR_AVAILABLE = False
_QA_AVAILABLE = False
_ENCODE_ARGS_AVAILABLE = False

try:
    from pipeline_core.validator import validate_input
    _VALIDATOR_AVAILABLE = True
except ImportError:
    pass

try:
    from pipeline_core.qa import validate_output
    _QA_AVAILABLE = True
except ImportError:
    pass

try:
    from pipeline_core.pipeline import ENCODE_ARGS_SHORT_FORM
    _ENCODE_ARGS_AVAILABLE = True
except ImportError:
    pass

_ALL_AVAILABLE = _VALIDATOR_AVAILABLE and _QA_AVAILABLE and _ENCODE_ARGS_AVAILABLE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_ffmpeg():
    """Skip if ffmpeg is not available."""
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except FileNotFoundError:
        pytest.skip("ffmpeg not available on PATH")


def _ffmpeg_generate_source(out_path: str, duration: int = 5) -> str:
    """Generate a test source video at 1080x1920, 30fps, with audio."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc2=duration={duration}:size=1080x1920:rate=30",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-t", str(duration),
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def _ffmpeg_reencode(source_path: str, out_path: str, encode_args: list[str]) -> str:
    """Re-encode source using the provided encode_args list."""
    cmd = [
        "ffmpeg", "-y",
        "-i", source_path,
        *encode_args,
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


# ===========================================================================
# The integration test
# ===========================================================================

@pytest.mark.slow
@pytest.mark.skipif(
    not _ALL_AVAILABLE,
    reason=(
        "Integration smoke requires validator, qa, and ENCODE_ARGS_SHORT_FORM — "
        "blocked on Builder. Missing: "
        + (", ".join(
            name for name, avail in [
                ("pipeline_core.validator", _VALIDATOR_AVAILABLE),
                ("pipeline_core.qa",        _QA_AVAILABLE),
                ("ENCODE_ARGS_SHORT_FORM",  _ENCODE_ARGS_AVAILABLE),
            ] if not avail
        ))
    ),
)
def test_full_pipeline_5s_youtube_short():
    """
    End-to-end smoke: source → validator → encode (ENCODE_ARGS_SHORT_FORM) → QA.

    Asserts:
      - validator returns ok=True for the source
      - re-encode completes without error
      - QA returns ok=True for youtube_short
      - duration ≈ 5s (within 1s)
      - pix_fmt == 'yuv420p' in measurements
      - lufs within -14 ± 1 in measurements (if lufs key present)
    """
    _require_ffmpeg()

    with tempfile.TemporaryDirectory(prefix="kaizer_smoke_") as tmp:
        source = os.path.join(tmp, "source.mp4")
        output = os.path.join(tmp, "output.mp4")

        # ── Step 1: generate source ──────────────────────────────────────────
        _ffmpeg_generate_source(source, duration=5)
        assert os.path.exists(source), "Source file not created"

        # ── Step 2: validate source ──────────────────────────────────────────
        val_result = validate_input(source)
        assert val_result.ok, (
            f"Validator rejected source file. "
            f"errors={val_result.errors}  warnings={val_result.warnings}"
        )

        # ── Step 3: re-encode with ENCODE_ARGS_SHORT_FORM ────────────────────
        try:
            _ffmpeg_reencode(source, output, list(ENCODE_ARGS_SHORT_FORM))
        except subprocess.CalledProcessError as exc:
            pytest.fail(
                f"FFmpeg re-encode failed with ENCODE_ARGS_SHORT_FORM.\n"
                f"stderr: {exc.stderr.decode(errors='replace')[:500]}"
            )
        assert os.path.exists(output), "Output file not created by re-encode"

        # ── Step 4: QA on output ─────────────────────────────────────────────
        qa_result = validate_output(output, platform="youtube_short", expected_duration_s=5.0)
        assert qa_result.ok, (
            f"QA failed on re-encoded output. "
            f"errors={qa_result.errors}  warnings={qa_result.warnings}"
        )

        # ── Step 5: measurement assertions ───────────────────────────────────
        assert qa_result.measurements, "QA measurements dict must be populated"

        # Duration ≈ 5s
        dur_key = next(
            (k for k in qa_result.measurements if "dur" in k.lower()), None
        )
        if dur_key:
            actual_duration = qa_result.measurements[dur_key]
            assert abs(actual_duration - 5.0) <= 1.0, (
                f"Duration should be ≈5s, got {actual_duration}s"
            )

        # pix_fmt == 'yuv420p'
        pix_key = next(
            (k for k in qa_result.measurements if "pix" in k.lower() or "fmt" in k.lower()), None
        )
        if pix_key:
            assert qa_result.measurements[pix_key] == "yuv420p", (
                f"pix_fmt must be yuv420p, got {qa_result.measurements[pix_key]!r}"
            )

        # LUFS within -14 ± 1
        lufs_key = next(
            (k for k in qa_result.measurements if "lufs" in k.lower() or "loudness" in k.lower()), None
        )
        if lufs_key:
            lufs_val = qa_result.measurements[lufs_key]
            if isinstance(lufs_val, (int, float)):
                assert -15.0 <= lufs_val <= -13.0, (
                    f"LUFS must be -14 ±1, got {lufs_val}"
                )
