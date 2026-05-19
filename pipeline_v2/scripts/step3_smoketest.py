"""Step 3 -- Stage 0 ingest smoketest.

Generates a small synthetic test clip (5s, black video + 100ms tone
burst at t=2.000s), runs ``run_stage_0`` on it, and asserts:

  - ``mezzanine.mp4`` exists, is CFR 30fps, codec h264.
  - ``source.mp3`` exists, is 48kHz mp3.
  - Click is preserved at t=2.000s ± 33ms (1 frame at 30fps).
  - Encoder choice is logged.

Exit 0 == green.

Run from KaizerBackend or project root; the script resolves paths
itself. Cleans up its temp output dir on exit.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# --- Path bootstrap (so v2 + v1 modules import cleanly) ----------------

HERE = Path(__file__).resolve().parent                          # .../scripts/
PIPELINE_V2_ROOT = HERE.parent                                   # .../pipeline_v2/
KAIZER_BACKEND = PIPELINE_V2_ROOT.parent                         # .../KaizerBackend/
for p in (PIPELINE_V2_ROOT, KAIZER_BACKEND):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from pipeline_v2.stages.stage_0_ingest import run_stage_0       # noqa: E402
from pipeline_v2.utils import ffmpeg_runner, ffprobe            # noqa: E402


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def info(msg: str) -> None:
    print(f"  ...    {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


# --- Helpers (duplicated from test file -- this script must stand alone
# so an operator can run it without installing pytest) ----------------


def make_fixture(out_path: Path, *, duration_s: float = 5.0,
                 click_at_s: float = 2.0) -> None:
    pre = click_at_s
    burst = 0.1
    post = max(0.0, duration_s - pre - burst)
    filter_complex = (
        f"color=c=black:s=320x240:r=30:d={duration_s}[v];"
        f"aevalsrc=0:d={pre}[a1];"
        f"sine=frequency=1000:duration={burst}[a2];"
        f"aevalsrc=0:d={post}[a3];"
        f"[a1][a2][a3]concat=n=3:v=0:a=1[a]"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-y", "-v", "error",
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac", "-ar", "48000", "-ac", "1",
        "-t", str(duration_s),
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"fixture build failed: {proc.stderr[-500:]}")


def detect_click_time(audio_path: Path) -> float:
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(audio_path),
        "-af", "silencedetect=noise=-40dB:d=0.05",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    for line in proc.stderr.splitlines():
        if "silence_end:" in line:
            for token in line.split():
                if token.endswith("|"):
                    token = token[:-1]
                try:
                    t = float(token)
                    if 0.5 < t < 100:
                        return t
                except ValueError:
                    continue
    raise RuntimeError(f"no silence_end found in {audio_path}")


# --- Main ----------------------------------------------------------------


async def main() -> int:
    failures: list[str] = []

    banner("Step 3 -- Stage 0 Ingest Smoketest")
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        fail("ffmpeg / ffprobe not on PATH -- nothing to smoketest")
        return 1
    info(f"ffmpeg: {shutil.which('ffmpeg')}")
    info(f"ffprobe: {shutil.which('ffprobe')}")

    # ------------------------------------------------------------------
    # 1. Encoder detection
    # ------------------------------------------------------------------
    banner("1. Encoder detection")
    encoders = ffmpeg_runner.list_encoders()
    info(f"h264_nvenc compiled in: {('h264_nvenc' in encoders)}")
    if "h264_nvenc" in encoders:
        info(f"NVENC runtime probe: {ffmpeg_runner.nvenc_runtime_ok()}")
    chosen = ffmpeg_runner.detect_encoder()
    ok(f"chosen encoder: {chosen}")

    # ------------------------------------------------------------------
    # 2. Build fixture + run Stage 0
    # ------------------------------------------------------------------
    banner("2. Build 5s click-at-2.0s fixture and run Stage 0")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        src = td_path / "fixture.mp4"
        out_dir = td_path / "stage0_out"

        info(f"generating fixture at {src}")
        make_fixture(src, duration_s=5.0, click_at_s=2.0)
        src_click = detect_click_time(src)
        info(f"fixture click time: {src_click:.4f}s (expected 2.0)")
        if abs(src_click - 2.0) > 0.05:
            failures.append(f"fixture click off by {abs(src_click-2.0)*1000:.1f}ms")
            fail(failures[-1])

        info("running Stage 0...")
        result = await run_stage_0(str(src), str(out_dir))
        ok(f"encoder_used: {result.encoder_used}")
        ok(f"duration: {result.duration_sec:.2f}s")
        ok(f"width x height: {result.width} x {result.height}")
        ok(f"source_was_vfr: {result.source_was_vfr}")
        ok(f"transcode: {result.transcode_seconds:.2f}s, "
           f"audio: {result.audio_extract_seconds:.2f}s, "
           f"wall: {result.wall_seconds:.2f}s")

        # --------------------------------------------------------------
        # 3. Verify mezzanine
        # --------------------------------------------------------------
        banner("3. Verify mezzanine.mp4")
        mezz = Path(result.mezzanine_path)
        if not mezz.is_file():
            failures.append(f"mezzanine missing: {mezz}")
            fail(failures[-1])
            return 1
        ok(f"file exists: {mezz} ({mezz.stat().st_size} bytes)")

        p = ffprobe.probe(str(mezz))
        info(f"mezzanine probe: codec={p.video_codec} fps={p.fps} "
             f"nominal={p.nominal_fps} vfr={p.is_vfr} "
             f"audio={p.audio_codec}/{p.sample_rate}Hz/{p.channels}ch")

        if p.video_codec != "h264":
            failures.append(f"mezzanine video codec {p.video_codec!r}, expected h264")
            fail(failures[-1])
        else:
            ok("mezzanine codec = h264")

        if not (p.fps and 29.5 < p.fps < 30.5):
            failures.append(f"mezzanine fps {p.fps!r}, expected ~30")
            fail(failures[-1])
        else:
            ok(f"mezzanine fps {p.fps:.3f} (~30)")

        if p.is_vfr:
            failures.append("mezzanine detected as VFR -- expected CFR")
            fail(failures[-1])
        else:
            ok("mezzanine is CFR (avg_fps == r_fps)")

        if p.audio_codec != "aac":
            failures.append(f"mezzanine audio {p.audio_codec!r}, expected aac")
            fail(failures[-1])
        else:
            ok("mezzanine audio = aac 48kHz stereo")

        # --------------------------------------------------------------
        # 4. Verify source.mp3
        # --------------------------------------------------------------
        banner("4. Verify source.mp3 (for Deepgram)")
        audio = Path(result.audio_path)
        if not audio.is_file():
            failures.append(f"audio missing: {audio}")
            fail(failures[-1])
            return 1
        ok(f"file exists: {audio} ({audio.stat().st_size} bytes)")

        ap = ffprobe.probe(str(audio))
        info(f"audio probe: codec={ap.audio_codec} sample_rate={ap.sample_rate}")

        if ap.audio_codec != "mp3":
            failures.append(f"audio codec {ap.audio_codec!r}, expected mp3")
            fail(failures[-1])
        else:
            ok("audio codec = mp3")

        if ap.sample_rate != 48000:
            failures.append(f"audio sample_rate {ap.sample_rate}, expected 48000")
            fail(failures[-1])
        else:
            ok("audio sample_rate = 48000")

        # --------------------------------------------------------------
        # 5. Timestamp preservation
        # --------------------------------------------------------------
        banner("5. Timestamp preservation (click at t=2.000s)")
        mezz_click = detect_click_time(mezz)
        drift_ms = abs(mezz_click - 2.0) * 1000
        info(f"mezzanine click at {mezz_click:.4f}s (drift {drift_ms:.1f}ms)")
        if drift_ms > 33:
            failures.append(
                f"click drift {drift_ms:.1f}ms exceeds 33ms (1 frame @ 30fps)"
            )
            fail(failures[-1])
        else:
            ok(f"click preserved within tolerance (drift {drift_ms:.1f}ms < 33ms)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    banner("Summary")
    if failures:
        print(f"  FAILURES ({len(failures)}):")
        for f in failures:
            print(f"   - {f}")
        return 1
    print("  ALL CHECKS PASSED")
    print(f"  - Stage 0 produced valid mezzanine + audio")
    print(f"  - Click at 2.000s preserved within 33ms tolerance")
    print(f"  - Encoder: {chosen} ({'NVENC' if chosen == 'h264_nvenc' else 'software fallback'})")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
