"""Step 4 -- Stage 1 (faster-whisper) smoketest.

Usage:
    # Validate the install, download / warm the model, run a tiny
    # dummy audio through end-to-end (no language check):
    python pipeline_v2/scripts/step4_smoketest.py --warmup

    # Real acceptance run on a Telugu / Hindi / English / code-mixed
    # fixture you provide:
    python pipeline_v2/scripts/step4_smoketest.py --fixture path/to/audio.mp3
    python pipeline_v2/scripts/step4_smoketest.py --fixture path/to/audio.mp3 --language te
    python pipeline_v2/scripts/step4_smoketest.py --fixture path/to/audio.mp3 --language hi
    python pipeline_v2/scripts/step4_smoketest.py --fixture path/to/audio.mp3 --no-language-hint

What it prints:
  - Model load latency (cold-start cost the worker pays once)
  - Transcribe wall time + realtime factor
  - Peak VRAM during transcription (GPU only)
  - Detected language + confidence
  - Total word count
  - 5 randomly-sampled words with their {start, end, text} so you can
    spot-check timestamp accuracy by playing the source audio at those
    points
  - First 80 words inline (a usable preview without dumping the full
    transcript)
  - Path to the full transcript JSON for offline inspection

Exit 0 = ran cleanly. Word-level accuracy + timestamp drift are
**your** call to judge from the printed sample.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# --- Path bootstrap ----------------------------------------------------

HERE = Path(__file__).resolve().parent
PIPELINE_V2_ROOT = HERE.parent
KAIZER_BACKEND = PIPELINE_V2_ROOT.parent
for p in (PIPELINE_V2_ROOT, KAIZER_BACKEND):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


def info(msg: str) -> None:
    print(f"  ...    {msg}")


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


# --- Synthetic fixture (for --warmup) ----------------------------------


def _build_warmup_fixture(out_path: Path) -> None:
    """Generate 2s of low-amplitude pink noise so faster-whisper has
    something to chew through that won't actually transcribe to text
    (but will exercise the full pipeline)."""
    if not shutil.which("ffmpeg"):
        sys.exit("FAIL: ffmpeg required for --warmup fixture generation")
    cmd = [
        "ffmpeg", "-hide_banner", "-y", "-v", "error",
        "-f", "lavfi", "-i", "anoisesrc=color=pink:duration=2:amplitude=0.05",
        "-ar", "48000", "-ac", "1",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        sys.exit(f"FAIL: warmup fixture build failed: {proc.stderr[-500:]}")


# --- Main --------------------------------------------------------------


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", type=str, help="Path to audio file to transcribe")
    ap.add_argument("--language", type=str, default=None,
                    help="ISO-639-1 hint (te, hi, en, ...). Omit for auto-detect.")
    ap.add_argument("--no-language-hint", action="store_true",
                    help="Force auto-detect even if --language was passed.")
    ap.add_argument("--warmup", action="store_true",
                    help="Generate a 2s noise fixture and run it end-to-end "
                         "to validate the install / measure cold-start cost.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output dir for transcript.json (default: tempdir)")
    ap.add_argument("--sample-words", type=int, default=5,
                    help="Number of random words to spot-check (default 5)")
    ap.add_argument("--preview-words", type=int, default=80,
                    help="Words to print inline as a preview (default 80)")
    args = ap.parse_args()

    if not args.warmup and not args.fixture:
        ap.error("Provide --fixture <path> OR --warmup")

    banner("Step 4 -- Stage 1 (faster-whisper large-v3) smoketest")

    # ------------------------------------------------------------------
    # GPU / device probe
    # ------------------------------------------------------------------
    banner("0. Hardware probe")
    try:
        import torch
        info(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                info(f"  GPU {i}: {p.name}  VRAM total: {p.total_memory/1024**3:.2f} GB  cap: {p.major}.{p.minor}")
    except ImportError:
        info("torch not installed -- will run on CPU (slow)")

    # ------------------------------------------------------------------
    # Resolve fixture
    # ------------------------------------------------------------------
    if args.warmup:
        tmpdir_ctx = tempfile.TemporaryDirectory()
        tmpdir = Path(tmpdir_ctx.__enter__())
        fixture = tmpdir / "warmup.wav"
        info(f"generating 2s warmup fixture at {fixture}")
        _build_warmup_fixture(fixture)
        language_hint = None
    else:
        tmpdir_ctx = None
        fixture = Path(args.fixture).resolve()
        if not fixture.is_file():
            fail(f"fixture not found: {fixture}")
            return 1
        language_hint = None if args.no_language_hint else args.language

    info(f"fixture: {fixture}")
    info(f"language_hint: {language_hint!r}")

    # ------------------------------------------------------------------
    # Model load (cold-start)
    # ------------------------------------------------------------------
    banner("1. Load model (cold-start cost)")
    import time

    from pipeline_v2.stages import stage_1_transcribe as s1

    t0 = time.perf_counter()
    s1.prewarm()
    cold_start = time.perf_counter() - t0
    cfg = s1._MODEL_CONFIG
    ok(f"model loaded in {cold_start:.1f}s")
    ok(f"   model_name:   {cfg.get('model_name')}")
    ok(f"   compute_type: {cfg.get('compute_type')}")
    ok(f"   device:       {cfg.get('device')}")

    # ------------------------------------------------------------------
    # Transcribe
    # ------------------------------------------------------------------
    banner("2. Transcribe")
    out_dir = Path(args.out) if args.out else (
        Path(tempfile.mkdtemp(prefix="step4_smoketest_"))
    )
    info(f"out_dir: {out_dir}")

    result = await s1.run_stage_1(
        str(fixture),
        language_hint=language_hint,
        out_dir=str(out_dir),
    )

    ok(f"audio_seconds:        {result.audio_seconds:.2f}s")
    ok(f"transcribe_seconds:   {result.transcribe_seconds:.2f}s")
    ok(f"realtime_factor:      {result.realtime_factor:.1f}x")
    ok(f"detected_language:    {result.detected_language} (p={result.detection_probability:.3f})")
    ok(f"language_hint:        {result.language_hint}")
    ok(f"detected_languages:   {result.transcript.detected_languages}")
    if result.peak_vram_mb is not None:
        ok(f"peak_vram_mb:         {result.peak_vram_mb:.0f} MB (absolute peak during transcribe)")
    else:
        ok("peak_vram_mb:         n/a (CPU or pynvml unavailable)")
    ok(f"word count:           {len(result.transcript.words)}")
    ok(f"transcript JSON:      {result.transcript_json_path}")

    # ------------------------------------------------------------------
    # Preview + random word spot-check
    # ------------------------------------------------------------------
    words = result.transcript.words
    if not words:
        info("(no words produced -- silent or noise-only audio)")
    else:
        banner(f"3. First {min(args.preview_words, len(words))} words (preview)")
        preview = words[: args.preview_words]
        print("   ", " ".join(w.w for w in preview))

        banner(f"4. {args.sample_words} random words for audio spot-check")
        info("Play the source audio at each [start..end] timecode and")
        info("confirm the spoken word matches the printed text within ~200ms.")
        print()
        n = min(args.sample_words, len(words))
        picks = random.sample(range(len(words)), n)
        picks.sort()
        for i, idx in enumerate(picks, 1):
            w = words[idx]
            print(f"     {i}.  [#{idx:5d}]  {w.s:7.3f}s -- {w.e:7.3f}s   '{w.w}'")

    # ------------------------------------------------------------------
    # Plan budget comparison
    # ------------------------------------------------------------------
    banner("5. Latency vs plan budget")
    # Plan: 30-min audio -> Deepgram is ~30s (realtime ~60x). Whisper
    # large-v3 on T4 GPU does ~10x realtime, so 30min audio = ~3min.
    # The full v2 pipeline budget is 5min wall time.
    #
    # First-call kernel compilation dominates on short audio, so the
    # realtime factor is meaningless for <30s clips. Don't extrapolate.
    if result.audio_seconds < 30:
        info(f"audio_seconds={result.audio_seconds:.1f}s, "
             f"transcribe={result.transcribe_seconds:.1f}s, "
             f"realtime={result.realtime_factor:.1f}x")
        info("(audio < 30s: extrapolation suppressed -- first-call CUDA")
        info(" kernel compilation dominates short-clip timings. Run on a")
        info(" 1+ min real fixture for representative latency.)")
    elif result.audio_seconds > 0:
        projected_30min = (30 * 60) / max(result.realtime_factor, 0.01)
        info(f"audio_seconds={result.audio_seconds:.1f}s, "
             f"transcribe={result.transcribe_seconds:.1f}s, "
             f"realtime={result.realtime_factor:.1f}x")
        info(f"projected 30-min audio -> {projected_30min:.0f}s "
             f"({projected_30min/60:.1f} min)")
        budget = 300                                  # 5 min total v2 budget
        if projected_30min > 0.6 * budget:
            info(f"WARNING: Stage 1 alone projects to >60% of 5-min total budget. "
                 f"Other stages (Gemini Pro + Flash x4 + render) need >2min combined.")
        else:
            ok(f"Stage 1 projects within budget ({projected_30min:.0f}s of {budget}s total)")

    # ------------------------------------------------------------------
    # Cleanup warmup tmpdir (if any)
    # ------------------------------------------------------------------
    if tmpdir_ctx is not None:
        tmpdir_ctx.__exit__(None, None, None)

    banner("Summary")
    print("  Smoketest ran end-to-end. Word-level accuracy + timestamp")
    print("  drift on language switches are YOUR call -- inspect the 5")
    print("  spot-check picks above against the source audio.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
