"""Step 4.2 Whisper-Groq probe.

Single-shot validation that Groq's hosted whisper-large-v3 returns
broadcast-grade word-level timestamps on a real Telugu fixture.
Bypasses the ``pipeline_v2.stages.stt`` abstraction entirely -- the
probe is a disposable diagnostic, not a registered provider.

If this probe passes acceptance, the next step is to wrap this logic in
``pipeline_v2/stages/stt/whisper_groq.py`` as a
``@register("whisper-groq")`` provider class.

Why Groq instead of Chirp 3?
  - Chirp 3 returned ``INTERNAL`` error on Telugu (Preview status, no
    SLA). See backlog: re-evaluate quarterly.
  - Teammate's V1 production code uses Groq-hosted Whisper successfully
    on identical content (Telugu + Hindi + Hindi-English code-mixed).
  - OpenAI-compatible API, no GCP setup, no Preview-status risk.

Probe is much simpler than the Chirp 3 one:
  - No GCS upload      (multipart POST direct to Groq API)
  - No polling loop    (synchronous response)
  - No cleanup logic   (nothing to clean up)

Usage::

    python pipeline_v2/scripts/step4_2_whisper_groq_probe.py \\
        --fixture E:/kaizer\\ new\\ data\\ training/videos/test.mp3 \\
        --language te \\
        --brief "Telugu political news podcast about ..." \\
        --names "Modi,Revanth Reddy,Telangana" \\
        [--model whisper-large-v3]

Required env (loaded from KaizerBackend/.env):
  GROQ_API_KEY=<key from console.groq.com>

Tier handling:
  - Free tier:  25 MB per-file cap; 480 audio-min/day
  - Dev tier:   100 MB per-file cap; higher quota
  - The probe enforces a 100 MB hard limit. Files between 25-100 MB get
    a clear warning (free tier will fail, dev tier OK).
  - For oversized files: Stage 0 (ingest) should extract audio at
    64kbps mono for the Groq path -- 30 min at that bitrate is <16 MB.
    Chunking (split + stitch) is deferred indefinitely; only build if
    64kbps mono is still insufficient.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

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


# --- Constants (locked; provider docstring will repeat) ----------------

# Per-file size caps published at https://console.groq.com/docs/speech-to-text
# Verified 2026-05-18.
GROQ_FREE_TIER_MB = 25
GROQ_DEV_TIER_MB = 100

# Pricing. Free tier is free (records audio_duration for ledger consistency).
# Dev tier is per audio-minute. Verified 2026-05-18; re-verify quarterly.
GROQ_DEV_USD_PER_HOUR = 0.04
GROQ_DEV_USD_PER_SECOND = GROQ_DEV_USD_PER_HOUR / 3600.0


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


def _require_env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        sys.exit(f"FAIL: env var {name!r} is empty / unset. "
                 f"Add it to KaizerBackend/.env.")
    return v


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


# --- Main --------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True, type=str,
                    help="Local path to the audio file to transcribe")
    ap.add_argument("--language", default="te", type=str,
                    help="ISO 639-1 code (default te). NOT BCP-47 -- Whisper "
                         "wants 'te' not 'te-IN'. Pass empty string for "
                         "Whisper's auto-detect.")
    ap.add_argument("--brief", default="", type=str,
                    help="Free-text content description -- joins with names "
                         "into Whisper's initial-prompt biasing string "
                         "(224-token cap).")
    ap.add_argument("--names", default="", type=str,
                    help="Comma-separated proper nouns -- joins with brief "
                         "into the initial-prompt string.")
    ap.add_argument("--model", default="whisper-large-v3", type=str,
                    help="Groq model id. Default whisper-large-v3 (matches "
                         "teammate's V1 production code). For A/B testing, "
                         "try whisper-large-v3-turbo.")
    args = ap.parse_args()

    _load_env()

    banner("Step 4.2 Whisper-Groq probe -- standalone, not registered as a provider")

    # ---- Env --------------------------------------------------------
    api_key = _require_env("GROQ_API_KEY")
    info(f"GROQ_API_KEY: {api_key[:6]}...{api_key[-4:]} ({len(api_key)} chars)")

    # ---- Fixture ----------------------------------------------------
    fixture = Path(args.fixture).resolve()
    if not fixture.is_file():
        sys.exit(f"FAIL: fixture not found at {fixture}")
    size_mb = fixture.stat().st_size / (1024 * 1024)
    info(f"fixture: {fixture}  ({size_mb:.1f} MB)")
    info(f"language: {args.language!r}  (empty = Whisper auto-detect)")
    info(f"model: {args.model}")

    # Tier-aware size check.
    if size_mb > GROQ_DEV_TIER_MB:
        sys.exit(
            f"FAIL: fixture is {size_mb:.1f} MB which exceeds Groq's "
            f"largest documented per-file cap ({GROQ_DEV_TIER_MB} MB on "
            f"dev tier). Mitigation: have Stage 0 (ingest) extract audio "
            f"at 64kbps mono -- 30 min at that bitrate is <16 MB."
        )
    elif size_mb > GROQ_FREE_TIER_MB:
        info(f"WARNING: file is {size_mb:.1f} MB which exceeds free-tier "
             f"cap ({GROQ_FREE_TIER_MB} MB). If you're on free tier this "
             f"request will fail. Dev tier accepts up to "
             f"{GROQ_DEV_TIER_MB} MB.")

    # ---- Build initial_prompt ---------------------------------------
    names_list = [n.strip() for n in args.names.split(",") if n.strip()]
    info(f"names ({len(names_list)}): {names_list}")

    # Same pattern as teammate's V1 production code: brief + ", ".join(names)
    prompt_parts: list[str] = []
    if args.brief.strip():
        prompt_parts.append(args.brief.strip())
    if names_list:
        prompt_parts.append(", ".join(names_list))
    initial_prompt = ". ".join(prompt_parts) if prompt_parts else None
    if initial_prompt:
        # Whisper's initial_prompt has a 224-token cap. Rough estimate at
        # 3 chars/token (conservative for multi-script content) gives
        # ~672 chars. We warn at 600 to leave headroom.
        if len(initial_prompt) > 600:
            info(f"WARNING: initial_prompt is {len(initial_prompt)} chars; "
                 f"may exceed Groq's 224-token cap. If the API returns "
                 f"400 about prompt length, trim the brief / names.")
        info(f"initial_prompt ({len(initial_prompt)} chars): "
             f"{initial_prompt[:120]!r}" +
             ("..." if len(initial_prompt) > 120 else ""))
    else:
        info("initial_prompt: (none -- no brief or names provided)")
        info("WARNING: running without prompt; results won't reflect "
             "production biasing. Pass --brief / --names for a "
             "representative test.")

    # ---- Lazy import after env load --------------------------------
    from groq import Groq

    client = Groq(api_key=api_key)

    # ---- Transcribe -------------------------------------------------
    banner("1. POST /audio/transcriptions")
    info(f"endpoint: api.groq.com/openai/v1/audio/transcriptions (via groq SDK)")
    info("response_format=verbose_json, timestamp_granularities=['word']")

    # Build the request kwargs. The groq SDK passes empty-string language
    # through; we want to omit it for auto-detect.
    kwargs: dict = {
        "model": args.model,
        "response_format": "verbose_json",
        "timestamp_granularities": ["word"],
    }
    if args.language:
        kwargs["language"] = args.language
    if initial_prompt:
        kwargs["prompt"] = initial_prompt

    t_submit = time.perf_counter()
    with open(fixture, "rb") as f:
        kwargs["file"] = (fixture.name, f.read())
    try:
        transcription = client.audio.transcriptions.create(**kwargs)
    except Exception as exc:
        fail(f"Groq API call failed: {type(exc).__name__}: {exc}")
        raise
    wall_secs = time.perf_counter() - t_submit
    ok(f"response received in {wall_secs:.1f}s wall")

    # ---- Parse ------------------------------------------------------
    banner("2. Parse response")

    audio_duration = float(getattr(transcription, "duration", 0.0) or 0.0)
    detected_language = getattr(transcription, "language", "") or ""
    raw_text = getattr(transcription, "text", "") or ""
    words_attr = getattr(transcription, "words", None) or []

    info(f"detected language: {detected_language!r}")
    info(f"audio duration:    {audio_duration:.1f}s ({audio_duration/60:.2f} min)")
    info(f"text length:       {len(raw_text)} chars")
    info(f"word objects:      {len(words_attr)}")

    if not words_attr:
        fail("ZERO word objects in response. response_format and "
             "timestamp_granularities may not have been honoured.")
        return 5

    # Normalise the response into the dicts we care about. Groq returns
    # Pydantic objects in 1.x; the field names follow the OpenAI shape.
    all_words: list[dict] = []
    for w in words_attr:
        all_words.append({
            "word": getattr(w, "word", "") or "",
            "start": float(getattr(w, "start", 0.0) or 0.0),
            "end": float(getattr(w, "end", 0.0) or 0.0),
        })

    # ---- Contract checks -------------------------------------------
    banner("3. Word-level contract checks")
    words_with_start = sum(1 for w in all_words if w["start"] > 0)
    words_with_end = sum(1 for w in all_words if w["end"] > 0)

    # Word[0] may legitimately have start=0.0. So check end>0 only.
    timestamped = sum(1 for w in all_words if w["end"] > 0)
    all_timestamped = timestamped == len(all_words)

    info(f"words with start > 0:  {words_with_start}/{len(all_words)}")
    info(f"words with end   > 0:  {words_with_end}/{len(all_words)}")
    if all_timestamped:
        ok("CONTRACT: every word has populated end timestamp")
    else:
        fail(f"CONTRACT VIOLATION: {len(all_words) - timestamped} words "
             f"with end == 0 -- segment-level fallback likely")

    # Groq doesn't return per-word confidence on the verbose_json path.
    # This is a vendor gap, not a contract violation -- Word.confidence
    # will be None for transcripts from this provider, same as Chirp 3.
    info("per-word confidence: not returned by Groq's verbose_json "
         "(vendor limitation). Stage 2 will treat None as 'unknown'.")

    # ---- Preview + spot-check -------------------------------------
    banner("4. First 80 words (preview)")
    print("   ", " ".join(w["word"].strip() for w in all_words[:80]))

    banner("5. 5 random word picks for audio spot-check")
    info("Play the source audio at each [start..end] timecode and")
    info("confirm the word matches what's spoken within +/-500ms.")
    print()
    import random
    random.seed(42)
    n = min(5, len(all_words))
    picks = sorted(random.sample(range(len(all_words)), n))
    for i, idx in enumerate(picks, 1):
        w = all_words[idx]
        print(f"     {i}.  [#{idx:5d}]  "
              f"{w['start']:7.3f}s -- {w['end']:7.3f}s   "
              f"{w['word']!r}")

    # ---- Cost estimate --------------------------------------------
    banner("6. Cost estimate")
    free_cost = 0.0
    dev_cost = audio_duration * GROQ_DEV_USD_PER_SECOND
    info(f"billed audio duration: {audio_duration:.1f}s "
         f"({audio_duration/60:.2f} min)")
    ok(f"free tier:    ${free_cost:.4f}  (recorded for ledger consistency)")
    info(f"if dev tier:  ${dev_cost:.4f}  (@ ${GROQ_DEV_USD_PER_HOUR}/hr)")

    # ---- Raw response dump for debugging --------------------------
    banner("7. Raw response (debugging)")
    out_dir = (PIPELINE_V2_ROOT / "tests" / "fixtures" / "step4_diag")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "whisper_groq_probe_raw.json"
    try:
        raw_dict = transcription.model_dump(mode="json")
    except AttributeError:
        # Older Pydantic shim; fall back to JSON serialise via .json()
        raw_dict = json.loads(transcription.model_dump_json())
    raw_path.write_text(json.dumps(raw_dict, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    info(f"raw response dumped: {raw_path}")

    # ---- Summary --------------------------------------------------
    banner("Summary")
    if all_timestamped:
        wpm = len(all_words) / max(audio_duration / 60, 0.01)
        print("  CONTRACT PASS: word-level timestamps present on every word.")
        print(f"  - {len(all_words)} words for {audio_duration:.1f}s audio "
              f"({wpm:.0f} wpm)")
        print(f"  - wall time: {wall_secs:.1f}s "
              f"(realtime factor {audio_duration / max(wall_secs, 0.01):.1f}x)")
        print("  - per-word confidence: not returned by Groq (vendor gap;")
        print("    Stage 2 treats None as 'unknown', not 'zero').")
        print("  Next: user spot-check on the 5 picks above before the")
        print("  full WhisperGroqProvider class is written.")
        return 0
    else:
        print("  CONTRACT VIOLATION -- word-level timestamps missing on")
        print("  one or more words. See section 3 for which check failed.")
        return 6


if __name__ == "__main__":
    sys.exit(main())
