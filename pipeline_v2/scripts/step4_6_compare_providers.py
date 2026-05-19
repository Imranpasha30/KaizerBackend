"""Step 4.6 -- Cross-provider accuracy comparison.

Runs the same audio fixture through every STT provider whose API key
is present in ``.env``, then produces a side-by-side markdown report.
Providers with missing keys are skipped gracefully with a clear log
line -- the script never crashes because of a missing key.

This script always goes through ``run_stage_1`` (the dispatcher), NOT
direct provider classes. That way it exercises the full abstraction
layer and validates the contract enforcement (word-level timestamps,
provider field, etc.).

Usage::

    python pipeline_v2/scripts/step4_6_compare_providers.py \\
        --fixture E:/kaizer\\ new\\ data\\ training/videos/test.mp3 \\
        --language te \\
        --brief "Telugu political news podcast about ..." \\
        --names "Modi,Revanth Reddy,Telangana"

Required env (any subset of the 3; missing ones are skipped):
  GROQ_API_KEY          (whisper-groq)
  DEEPGRAM_API_KEY      (deepgram)
  ASSEMBLYAI_API_KEY    (assemblyai)

Outputs (under ``tests/fixtures/step4_6_compare/``):
  <provider>/transcript.json
  <provider>/stt_metadata.json
  REPORT.md                  -- the side-by-side comparison

The script doesn't accept API keys on the command line on purpose --
they live in .env, and the script just runs whatever is wired up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

# UTF-8 stdout so Telugu glyphs print cleanly on Windows cp1252 console.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# --- Path bootstrap ----------------------------------------------------

HERE = Path(__file__).resolve().parent
PIPELINE_V2_ROOT = HERE.parent
KAIZER_BACKEND = PIPELINE_V2_ROOT.parent
for p in (PIPELINE_V2_ROOT, KAIZER_BACKEND):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


# --- Provider registry (mirrors stages/stt/__init__.py PROVIDERS) ------

# (provider_registry_key, required env var, human label for the report)
PROVIDERS_TO_TEST = [
    ("whisper-groq", "GROQ_API_KEY",       "Whisper-Groq (whisper-large-v3)"),
    ("deepgram",     "DEEPGRAM_API_KEY",   "Deepgram Nova-3"),
    ("assemblyai",   "ASSEMBLYAI_API_KEY", "AssemblyAI Universal-2"),
]


# --- Script-bleed regexes ---------------------------------------------

# Telugu Unicode block U+0C00-U+0C7F.
RE_TELUGU = re.compile(r"[ఀ-౿]")

# All OTHER common Indic blocks (Devanagari for Hindi/Sanskrit, Bengali,
# Gurmukhi/Punjabi, Gujarati, Oriya, Tamil, Kannada, Malayalam). Used
# to detect cross-script bleed when the audio was supposed to be Telugu.
RE_OTHER_INDIC = re.compile(
    r"[ऀ-ॿ"   # Devanagari
    r"ঀ-৿"    # Bengali (the specific bleed the user called out)
    r"਀-੿"    # Gurmukhi
    r"઀-૿"    # Gujarati
    r"଀-୿"    # Oriya
    r"஀-௿"    # Tamil
    r"ಀ-೿"    # Kannada
    r"ഀ-ൿ"    # Malayalam
    r"]"
)


# --- .env loader -------------------------------------------------------


def _load_env() -> None:
    env_file = KAIZER_BACKEND / ".env"
    if not env_file.is_file():
        sys.exit(f"FAIL: .env not found at {env_file}")
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if k not in os.environ and v:
            os.environ[k] = v


def _key_present(name: str) -> bool:
    v = os.environ.get(name, "").strip()
    if not v:
        return False
    # Catch obvious placeholders.
    if v.lower() in ("<key>", "<your-api-key>", "your_key_here", "placeholder"):
        return False
    return True


# --- Output helpers ----------------------------------------------------


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f" {title}")
    print("=" * 72)


def info(msg: str) -> None:
    print(f"  ...    {msg}")


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def skip(msg: str) -> None:
    print(f"  [SKIP] {msg}")


# --- Per-provider stats ------------------------------------------------


def _confidence_stats(words) -> Optional[dict]:
    """Mean / min / max of populated word.confidence values. None if
    no word has a confidence (vendor gap, e.g. Whisper-Groq)."""
    vals = [w.confidence for w in words if w.confidence is not None]
    if not vals:
        return None
    return {
        "mean": statistics.mean(vals),
        "min": min(vals),
        "max": max(vals),
        "n":   len(vals),
    }


def _speaker_count(words) -> int:
    return len({w.speaker for w in words if w.speaker is not None})


def _script_bleed(transcript_text: str) -> dict:
    """Count Telugu vs other-Indic glyphs in the transcript text.

    Returns ``{telugu_chars, other_indic_chars, bleed_pct}``.
    ``bleed_pct = other / (telugu + other) * 100``.
    """
    telugu = len(RE_TELUGU.findall(transcript_text))
    other = len(RE_OTHER_INDIC.findall(transcript_text))
    total = telugu + other
    bleed_pct = (other / total * 100.0) if total > 0 else 0.0
    return {
        "telugu_chars": telugu,
        "other_indic_chars": other,
        "bleed_pct": bleed_pct,
    }


# --- Report builder ----------------------------------------------------


def _md_table_header(cols: list[str]) -> list[str]:
    return [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]


def build_report(
    *,
    fixture: Path,
    args: argparse.Namespace,
    results: dict,   # provider_key -> (Stage1Output | Exception)
    skipped: list[tuple[str, str]],
) -> str:
    lines: list[str] = []
    lines.append(f"# Step 4.6 — Cross-provider accuracy comparison")
    lines.append("")
    lines.append(f"- **Fixture**: `{fixture}`")
    lines.append(f"- **Fixture size**: {fixture.stat().st_size / (1024*1024):.1f} MB")
    lines.append(f"- **Language hint**: `{args.language}`")
    if args.brief:
        lines.append(f"- **Brief**: {args.brief!r}")
    names_list = [n.strip() for n in args.names.split(",") if n.strip()]
    if names_list:
        lines.append(f"- **Names** ({len(names_list)}): {names_list}")
    lines.append("")

    # ---- Skipped providers ----
    if skipped:
        lines.append("## Skipped providers")
        lines.append("")
        for p, k in skipped:
            lines.append(f"- `{p}` — env var `{k}` not set in `.env`")
        lines.append("")

    # ---- Per-provider headline metrics ----
    lines.append("## Headline metrics")
    lines.append("")
    lines.extend(_md_table_header([
        "Provider", "Words", "Audio (s)", "Wall (s)", "Realtime",
        "Cost (USD)", "Detected lang", "Speakers", "Bleed %", "Status",
    ]))

    for provider_key, env_key, label in PROVIDERS_TO_TEST:
        if provider_key in {p for p, _ in skipped}:
            lines.append(
                f"| {label} | — | — | — | — | — | — | — | — | SKIPPED |"
            )
            continue
        result = results.get(provider_key)
        if isinstance(result, Exception):
            lines.append(
                f"| {label} | — | — | — | — | — | — | — | — | "
                f"FAILED: {type(result).__name__}: {str(result)[:80]} |"
            )
            continue
        if result is None:
            lines.append(
                f"| {label} | — | — | — | — | — | — | — | — | NOT RUN |"
            )
            continue

        words = result.transcript.words
        bleed = _script_bleed(" ".join(w.w for w in words))
        speaker_n = _speaker_count(words)
        lines.append(
            f"| {label} "
            f"| {result.stt_word_count} "
            f"| {result.stt_audio_duration_sec:.1f} "
            f"| {result.stt_wall_seconds:.1f} "
            f"| {result.realtime_factor:.1f}x "
            f"| ${result.stt_cost_usd:.4f} "
            f"| {result.stt_language_detected} "
            f"| {speaker_n} "
            f"| {bleed['bleed_pct']:.1f}% "
            f"| OK |"
        )
    lines.append("")

    # ---- Per-word confidence stats ----
    lines.append("## Per-word confidence")
    lines.append("")
    lines.extend(_md_table_header(["Provider", "Coverage", "Mean", "Min", "Max"]))
    for provider_key, env_key, label in PROVIDERS_TO_TEST:
        result = results.get(provider_key)
        if not isinstance(result, type(None)) and not isinstance(result, Exception) and result is not None:
            stats = _confidence_stats(result.transcript.words)
            if stats is None:
                lines.append(
                    f"| {label} | 0/{result.stt_word_count} "
                    f"| n/a | n/a | n/a — *vendor gap (None for all words)* |"
                )
            else:
                lines.append(
                    f"| {label} | {stats['n']}/{result.stt_word_count} "
                    f"| {stats['mean']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} |"
                )
        else:
            lines.append(f"| {label} | — | — | — | — |")
    lines.append("")

    # ---- First 80 words preview per provider ----
    lines.append("## First 80 words (preview)")
    lines.append("")
    for provider_key, env_key, label in PROVIDERS_TO_TEST:
        result = results.get(provider_key)
        if isinstance(result, Exception) or result is None:
            continue
        words = result.transcript.words
        preview = " ".join(w.w for w in words[:80])
        lines.append(f"### {label}")
        lines.append("")
        lines.append("> " + preview)
        lines.append("")

    # ---- 5 random shared word indices (apples-to-apples) ----
    lines.append("## Cross-provider word-index spot-check")
    lines.append("")
    runnable = [
        r for r in results.values()
        if r is not None and not isinstance(r, Exception)
    ]
    if len(runnable) < 1:
        lines.append("_No successful providers to compare._")
        lines.append("")
    else:
        # Index domain = shortest transcript's length; ensures every
        # provider has a word at the picked index.
        min_words = min(len(r.transcript.words) for r in runnable)
        if min_words == 0:
            lines.append("_Shortest transcript has 0 words; nothing to pick._")
            lines.append("")
        else:
            random.seed(42)
            n_pick = min(5, min_words)
            picks = sorted(random.sample(range(min_words), n_pick))
            lines.append(
                f"Indices picked uniformly from the shortest transcript "
                f"({min_words} words). At each index, every provider's "
                f"word at that position is shown side-by-side with its "
                f"timestamp -- play the audio at those timestamps and "
                f"verify the spoken word matches within ±500ms."
            )
            lines.append("")
            header = ["Index"] + [
                label for _, _, label in PROVIDERS_TO_TEST
                if not isinstance(results.get(_pkey := _), Exception)
                and results.get(_pkey) is not None
            ]
            # Simpler: header is fixed across all 3 providers,
            # filling "—" for skipped/failed cells per row.
            header = ["Index"] + [label for _, _, label in PROVIDERS_TO_TEST]
            lines.extend(_md_table_header(header))
            for idx in picks:
                row = [f"#{idx}"]
                for provider_key, env_key, label in PROVIDERS_TO_TEST:
                    result = results.get(provider_key)
                    if isinstance(result, Exception) or result is None:
                        row.append("—")
                    else:
                        w = result.transcript.words[idx] if idx < len(result.transcript.words) else None
                        if w is None:
                            row.append("(beyond end)")
                        else:
                            row.append(
                                f"{w.s:.2f}s–{w.e:.2f}s `{w.w}`"
                            )
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    # ---- Script-bleed detail ----
    lines.append("## Script-bleed detail")
    lines.append("")
    lines.append(
        "Bleed = non-Telugu Indic characters as a percentage of all "
        "Indic characters in the transcript. Detects when a provider "
        "occasionally pulls glyphs from Bengali / Devanagari / etc. "
        "for Telugu phonemes."
    )
    lines.append("")
    lines.extend(_md_table_header([
        "Provider", "Telugu chars", "Other Indic", "Bleed %",
    ]))
    for provider_key, env_key, label in PROVIDERS_TO_TEST:
        result = results.get(provider_key)
        if isinstance(result, Exception) or result is None:
            continue
        words = result.transcript.words
        bleed = _script_bleed(" ".join(w.w for w in words))
        lines.append(
            f"| {label} | {bleed['telugu_chars']} "
            f"| {bleed['other_indic_chars']} "
            f"| {bleed['bleed_pct']:.1f}% |"
        )
    lines.append("")

    # ---- Footer ----
    lines.append("---")
    lines.append("")
    lines.append(
        f"Generated by `pipeline_v2/scripts/step4_6_compare_providers.py` "
        f"at {time.strftime('%Y-%m-%d %H:%M:%S')}."
    )
    return "\n".join(lines)


# --- Main --------------------------------------------------------------


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True, type=str,
                    help="Local path to the audio file to transcribe")
    ap.add_argument("--language", default="te", type=str,
                    help="ISO 639-1 code (default te). All 3 providers "
                         "accept 'te' for Telugu.")
    ap.add_argument("--brief", default="", type=str,
                    help="Free-text content description -- passed to each "
                         "provider's adaptation API (Whisper-Groq prompt / "
                         "Deepgram keyterm / AssemblyAI keyterms_prompt).")
    ap.add_argument("--names", default="", type=str,
                    help="Comma-separated proper nouns -- same as --brief, "
                         "boosted for proper-noun accuracy.")
    args = ap.parse_args()

    _load_env()

    banner("Step 4.6 -- Cross-provider STT comparison")
    fixture = Path(args.fixture).resolve()
    if not fixture.is_file():
        sys.exit(f"FAIL: fixture not found at {fixture}")
    info(f"fixture: {fixture}  ({fixture.stat().st_size / (1024*1024):.1f} MB)")
    info(f"language: {args.language!r}")

    # ---- Discover available providers --------------------------------
    banner("Available providers")
    available: list[tuple[str, str, str]] = []
    skipped: list[tuple[str, str]] = []
    for provider_key, env_key, label in PROVIDERS_TO_TEST:
        if _key_present(env_key):
            ok(f"{provider_key}: {env_key} is set")
            available.append((provider_key, env_key, label))
        else:
            skip(f"{env_key} not set in .env -- skipping {provider_key!r}")
            skipped.append((provider_key, env_key))

    if not available:
        sys.exit(
            "FAIL: no provider keys present in .env. Set at least one "
            "of GROQ_API_KEY / DEEPGRAM_API_KEY / ASSEMBLYAI_API_KEY."
        )

    # ---- Output dir --------------------------------------------------
    compare_root = (
        PIPELINE_V2_ROOT / "tests" / "fixtures" / "step4_6_compare"
    )
    compare_root.mkdir(parents=True, exist_ok=True)
    info(f"output root: {compare_root}")

    # Lazy import to keep --help fast.
    from pipeline_v2.stages.stt import run_stage_1

    # ---- Run each available provider --------------------------------
    results: dict = {}
    names_list = [n.strip() for n in args.names.split(",") if n.strip()]

    for provider_key, env_key, label in available:
        banner(f"RUN: {provider_key}  ({label})")
        out_dir = compare_root / provider_key
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            t0 = time.perf_counter()
            stage1_output = await run_stage_1(
                str(fixture),
                provider=provider_key,
                language_hint=args.language,
                brief=args.brief,
                names=names_list,
                out_dir=str(out_dir),
            )
            elapsed = time.perf_counter() - t0
            ok(f"completed in {elapsed:.1f}s -- "
               f"{stage1_output.stt_word_count} words, "
               f"${stage1_output.stt_cost_usd:.4f}")
            results[provider_key] = stage1_output
        except Exception as exc:
            fail(f"{provider_key} failed: {type(exc).__name__}: {exc}")
            results[provider_key] = exc

    # ---- Build markdown report --------------------------------------
    banner("Build comparison report")
    report = build_report(
        fixture=fixture, args=args, results=results, skipped=skipped,
    )
    report_path = compare_root / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    ok(f"report written: {report_path}")

    # ---- Console summary --------------------------------------------
    banner("Summary")
    n_ok = sum(
        1 for r in results.values()
        if r is not None and not isinstance(r, Exception)
    )
    n_fail = sum(1 for r in results.values() if isinstance(r, Exception))
    n_skip = len(skipped)
    info(f"providers run OK:    {n_ok}")
    info(f"providers failed:    {n_fail}")
    info(f"providers skipped:   {n_skip}")
    print()
    print(f"  Inspect {report_path} for the side-by-side comparison.")
    print(f"  Per-provider transcripts in {compare_root}/<provider>/.")
    print()
    print("  Next: spot-check 5 random word indices against the source")
    print("  audio. Pick the best provider visually (or pick multiple")
    print("  for the Step 11 dropdown -- accuracy tiers).")

    # Exit 0 unless EVERY available provider failed.
    return 0 if (n_ok > 0 or not available) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
