"""Step 4.1 Chirp 3 probe.

Single-shot validation that Chirp 3 returns broadcast-grade word-level
timestamps on a real Telugu fixture. Bypasses the
``pipeline_v2.stages.stt`` abstraction entirely -- the probe is a
disposable diagnostic, not a registered provider.

If this probe passes acceptance, the next step is to wrap this logic in
``pipeline_v2/stages/stt/chirp3.py`` as a ``@register("chirp3")``
provider class. If it fails (word offsets missing / 0-valued / segment-
aligned, or text quality regresses with timestamps enabled), Chirp 3
becomes a no-go and we move to Step 4.2 (Whisper-Groq).

Usage:
    python pipeline_v2/scripts/step4_1_chirp3_probe.py \\
        --fixture E:/kaizer\\ new\\ data\\ training/videos/test.mp3 \\
        --language te-IN \\
        --brief "Telugu political news podcast about ..." \\
        --names "Modi,Revanth Reddy,Telangana"

Requires env vars (loaded from KaizerBackend/.env):
    GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
    GOOGLE_CLOUD_PROJECT=<project id>
    GOOGLE_CLOUD_REGION=us              # Chirp 3 GA only in us / eu multi-region
    KAIZER_GCS_STT_BUCKET=<bucket name in same multi-region>

Boost values (locked, document in provider when the full class is written):
  - names    -> PhraseSet entries at boost=15.0   (strict proper-noun biasing)
  - brief    -> word-tokenised + boost=5.0         (looser content-domain hint)

Vendor feature gaps to document in the future Chirp3Provider:

  - **No per-word confidence**: Chirp 3 does NOT support
    ``enable_word_confidence``. Word.confidence will always be None
    for transcripts from this provider. Stage 2 logic that uses
    confidence-weighted cut boundaries should treat None as
    "unknown confidence", not "zero confidence".
  - **Region**: Chirp 3 is GA only in ``us`` and ``eu`` multi-regions
    (as of 2026-05-18). The probe enforces ``GOOGLE_CLOUD_REGION`` to
    be one of those; setting ``asia-south1`` will fail loudly.
  - **Telugu Preview**: ``te-IN`` is Preview on Chirp 3 -- model
    behaviour and pricing may change without notice.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# UTF-8 stdout so Telugu glyphs print cleanly on Windows cp1252 console.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# --- Path bootstrap --------------------------------------------------------

HERE = Path(__file__).resolve().parent
PIPELINE_V2_ROOT = HERE.parent
KAIZER_BACKEND = PIPELINE_V2_ROOT.parent
for p in (PIPELINE_V2_ROOT, KAIZER_BACKEND):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


# --- Boost values (locked; provider docstring will repeat) -----------------

BOOST_NAMES = 15.0
BOOST_BRIEF = 5.0

# Chirp 3 price per minute. Verified 2026-05-18 against Google Cloud
# pricing page. Re-verify quarterly.
CHIRP3_USD_PER_MINUTE = 0.024


# --- .env loader -----------------------------------------------------------


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
        sys.exit(f"FAIL: env var {name!r} is empty / unset")
    return v


# --- Output helpers --------------------------------------------------------


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


# --- Helpers ---------------------------------------------------------------


def _to_seconds(d) -> float:
    """protobuf.Duration -> float seconds. None-safe."""
    if d is None:
        return 0.0
    seconds = getattr(d, "seconds", 0) or 0
    nanos = getattr(d, "nanos", 0) or 0
    return float(seconds) + float(nanos) / 1e9


def _tokenise_brief(brief: str) -> list[str]:
    """Pick boost-worthy tokens out of free-text brief.

    Strategy: split on whitespace + punctuation, drop tokens shorter
    than 4 chars (common English stop-words, articles), dedupe while
    preserving order. Cap at 32 tokens so the PhraseSet doesn't bloat.
    """
    if not brief.strip():
        return []
    raw = re.findall(r"[\wऀ-ॿఀ-౿]+", brief, flags=re.UNICODE)
    seen: set[str] = set()
    out: list[str] = []
    for t in raw:
        if len(t) < 4:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= 32:
            break
    return out


def _build_adaptation(names: list[str], brief: str):
    """Construct an inline SpeechAdaptation from names + brief.

    Returns None if both inputs are empty (don't send an empty
    adaptation -- the API may complain).
    """
    from google.cloud.speech_v2.types import PhraseSet, SpeechAdaptation

    phrases: list[PhraseSet.Phrase] = []
    for n in names:
        n = n.strip()
        if n:
            phrases.append(PhraseSet.Phrase(value=n, boost=BOOST_NAMES))
    for w in _tokenise_brief(brief):
        phrases.append(PhraseSet.Phrase(value=w, boost=BOOST_BRIEF))

    if not phrases:
        return None

    return SpeechAdaptation(
        phrase_sets=[
            SpeechAdaptation.AdaptationPhraseSet(
                inline_phrase_set=PhraseSet(phrases=phrases),
            )
        ]
    )


# --- Main -----------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True, type=str,
                    help="Local path to the audio file to transcribe")
    ap.add_argument("--language", default="te-IN", type=str,
                    help="BCP-47 code (default te-IN)")
    ap.add_argument("--brief", default="", type=str,
                    help="Free-text content description -- maps to "
                         "PhraseSet entries at boost=5.0")
    ap.add_argument("--names", default="", type=str,
                    help="Comma-separated proper nouns -- map to "
                         "PhraseSet entries at boost=15.0")
    ap.add_argument("--timeout-sec", default=600, type=int,
                    help="Max wait for BatchRecognize to complete. Extended "
                         "from 300s to 600s after observing Chirp 3 +"
                         "adaptation taking 4-5min on cold/fresh projects.")
    ap.add_argument("--poll-interval-sec", default=15, type=int,
                    help="Polling interval while waiting. Raised from 5s to "
                         "15s after a fresh project tripped "
                         "RESOURCE_EXHAUSTED on operations.get quotas.")
    args = ap.parse_args()

    _load_env()

    banner("Step 4.1 Chirp 3 probe -- standalone, not registered as a provider")

    # ---- Env probe ------------------------------------------------------
    sa_path = _require_env("GOOGLE_APPLICATION_CREDENTIALS")
    project = _require_env("GOOGLE_CLOUD_PROJECT")
    region = _require_env("GOOGLE_CLOUD_REGION")
    bucket_name = _require_env("KAIZER_GCS_STT_BUCKET")
    if not Path(sa_path).is_file():
        sys.exit(f"FAIL: service account JSON not found at {sa_path}")
    # Chirp 3 is GA only in 'us' and 'eu' multi-regions as of 2026-05-18.
    # Fail loudly here so a stale .env value (e.g. asia-south1 from
    # before the region migration) doesn't waste a transcribe attempt.
    if region not in ("us", "eu"):
        sys.exit(
            f"FAIL: GOOGLE_CLOUD_REGION={region!r} is not a Chirp 3 region. "
            f"Chirp 3 is GA only in 'us' and 'eu' multi-regions. "
            f"Update .env to GOOGLE_CLOUD_REGION=us (or eu) and ensure "
            f"KAIZER_GCS_STT_BUCKET is in the same multi-region."
        )
    info(f"service account: {sa_path}")
    info(f"project: {project}")
    info(f"region: {region}")
    info(f"bucket: {bucket_name}")

    # ---- Fixture --------------------------------------------------------
    fixture = Path(args.fixture).resolve()
    if not fixture.is_file():
        sys.exit(f"FAIL: fixture not found at {fixture}")
    info(f"fixture: {fixture}  ({fixture.stat().st_size / 1024 / 1024:.1f} MB)")
    info(f"language: {args.language}")
    info(f"brief: {args.brief[:80]!r}" + ("..." if len(args.brief) > 80 else ""))

    names = [n.strip() for n in args.names.split(",") if n.strip()]
    info(f"names ({len(names)}): {names}")

    if not args.brief.strip() and not names:
        info("WARNING: both brief and names are empty -- adaptation will "
             "be skipped. Run with --brief and --names for a representative "
             "test of biasing-on-Telugu accuracy.")

    # ---- Lazy SDK imports (after env load so any import-time creds
    #      resolution sees the right vars) ---------------------------------
    from google.cloud import speech_v2, storage
    from google.cloud.speech_v2.types import (
        BatchRecognizeFileMetadata,
        BatchRecognizeRequest,
        InlineOutputConfig,
        RecognitionConfig,
        RecognitionFeatures,
        RecognitionOutputConfig,
    )

    # ---- GCS upload -----------------------------------------------------
    banner("1. Upload fixture to GCS")
    blob_name = f"step4_1_probe/{uuid.uuid4().hex}{fixture.suffix}"
    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    info(f"target: {gcs_uri}")

    storage_client = storage.Client(project=project)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Cleanup-safety flags. See the finally block at the bottom of this
    # try for the full decision matrix. Short version:
    #   - operation_submitted=False -> never sent to server; safe to delete
    #   - operation_terminal=True   -> server reached done=true; safe to delete
    #   - otherwise (in-flight)     -> DO NOT delete; 24h lifecycle handles it
    operation_submitted = False
    operation_terminal = False
    operation = None  # bound after batch_recognize() to enable the error msg below

    try:
        t0 = time.perf_counter()
        blob.upload_from_filename(str(fixture), content_type="audio/mpeg")
        upload_secs = time.perf_counter() - t0
        ok(f"uploaded in {upload_secs:.1f}s")

        # ---- BatchRecognize ---------------------------------------------
        banner("2. Submit BatchRecognize")
        api_endpoint = f"{region}-speech.googleapis.com"
        info(f"api_endpoint: {api_endpoint}")
        client = speech_v2.SpeechClient(
            client_options={"api_endpoint": api_endpoint},
        )

        recognizer = f"projects/{project}/locations/{region}/recognizers/_"
        info(f"recognizer: {recognizer}")

        # enable_word_confidence is NOT set: Chirp 3 doesn't support
        # per-word confidence (vendor gap). Setting it True causes the
        # API to reject the request. Word.confidence will always be
        # None on Chirp 3 transcripts; Stage 2 should treat None as
        # "unknown confidence", not "zero confidence".
        features = RecognitionFeatures(
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
        )
        adaptation = _build_adaptation(names, args.brief)
        if adaptation is None:
            info("adaptation: (skipped -- no names or brief)")
        else:
            n_phrases = sum(len(p.inline_phrase_set.phrases)
                            for p in adaptation.phrase_sets)
            info(f"adaptation: 1 inline phrase set, {n_phrases} phrases")

        config = RecognitionConfig(
            auto_decoding_config=speech_v2.types.AutoDetectDecodingConfig(),
            model="chirp_3",
            language_codes=[args.language],
            features=features,
            adaptation=adaptation,
        )

        request = BatchRecognizeRequest(
            recognizer=recognizer,
            config=config,
            files=[BatchRecognizeFileMetadata(uri=gcs_uri)],
            recognition_output_config=RecognitionOutputConfig(
                inline_response_config=InlineOutputConfig(),
            ),
        )

        t_submit = time.perf_counter()
        operation = client.batch_recognize(request=request)
        operation_submitted = True   # cleanup-safety flag (see finally)
        info(f"operation submitted: {operation.operation.name}")

        # ---- Poll -------------------------------------------------------
        # On fresh GCP projects, ``operations.get()`` can trip
        # RESOURCE_EXHAUSTED at the 12 req/min cadence the original 5s
        # poll produced. We now:
        #   - Poll every 15s by default (lower API pressure)
        #   - Catch ResourceExhausted specifically and back off 30s
        #   - Cap consecutive quota errors at 3 (then raise so the
        #     operator can check the operation state out-of-band)
        # The operation may still be running server-side when we raise;
        # the GCS source file must NOT be deleted in that case (see the
        # cleanup logic at the bottom of this try block).
        from google.api_core.exceptions import ResourceExhausted

        banner("3. Poll operation")
        QUOTA_BACKOFF_SEC = 30
        MAX_CONSECUTIVE_QUOTA_ERRORS = 3

        deadline = time.perf_counter() + args.timeout_sec
        consecutive_quota = 0
        quota_retries = 0

        while True:
            try:
                if operation.done():
                    break
                consecutive_quota = 0      # reset on any successful poll
            except ResourceExhausted as exc:
                consecutive_quota += 1
                quota_retries += 1
                if consecutive_quota >= MAX_CONSECUTIVE_QUOTA_ERRORS:
                    op_name = operation.operation.name
                    op_id = op_name.rsplit("/", 1)[-1]
                    raise RuntimeError(
                        f"{MAX_CONSECUTIVE_QUOTA_ERRORS} consecutive "
                        f"RESOURCE_EXHAUSTED on operations.get(). The "
                        f"operation may still be running server-side -- "
                        f"check via:\n\n"
                        f"  gcloud speech operations describe {op_id} "
                        f"--region={region} --project={project}\n\n"
                        f"OR:\n\n"
                        f"  curl -H \"Authorization: Bearer "
                        f"$(gcloud auth print-access-token)\" \\\n"
                        f"       https://{api_endpoint}/v2/{op_name}\n\n"
                        f"GCS source file at {gcs_uri} was NOT deleted "
                        f"(in-flight operation may still be reading it). "
                        f"The bucket's 24h lifecycle rule will clean up."
                    ) from exc
                info(f"  ...quota throttle "
                     f"({consecutive_quota}/{MAX_CONSECUTIVE_QUOTA_ERRORS}), "
                     f"backoff {QUOTA_BACKOFF_SEC}s ...")
                time.sleep(QUOTA_BACKOFF_SEC)
                continue

            elapsed = time.perf_counter() - t_submit
            if time.perf_counter() > deadline:
                raise TimeoutError(
                    f"BatchRecognize did not finish within "
                    f"{args.timeout_sec}s. Operation name: "
                    f"{operation.operation.name}. GCS source file at "
                    f"{gcs_uri} was NOT deleted (in-flight op may still "
                    f"be reading it); 24h lifecycle will clean up."
                )
            info(f"  ...polling (elapsed {elapsed:.0f}s) ...")
            time.sleep(args.poll_interval_sec)

        # operation.done() returned True -- server reached terminal state.
        # From this point, the GCS source file is safe to delete.
        operation_terminal = True

        wall_secs = time.perf_counter() - t_submit
        response = operation.result()
        ok(f"batch op finished in {wall_secs:.1f}s wall "
           f"(quota retries: {quota_retries})")

        # ---- Parse ------------------------------------------------------
        banner("4. Parse response")
        if not response.results:
            fail("response.results is empty")
            return 3

        file_result = next(iter(response.results.values()))
        # file_result.transcript is a BatchRecognizeResults wrapping the
        # SpeechRecognitionResult list. The shape is the same as a
        # streaming RecognizeResponse but flattened.
        transcript_container = getattr(file_result, "transcript", None)
        if transcript_container is None or not transcript_container.results:
            fail("no transcript or empty results")
            print(file_result)
            return 4

        billed = getattr(transcript_container.metadata,
                         "total_billed_duration", None)
        billed_secs = _to_seconds(billed)
        ok(f"total_billed_duration: {billed_secs:.1f}s")

        # Collect all words across all SpeechRecognitionResult items
        all_words = []
        seen_languages: list[str] = []
        for sr in transcript_container.results:
            if not sr.alternatives:
                continue
            alt = sr.alternatives[0]
            if sr.language_code and sr.language_code not in seen_languages:
                seen_languages.append(sr.language_code)
            for w in alt.words:
                all_words.append({
                    "word": w.word,
                    "start": _to_seconds(w.start_offset),
                    "end": _to_seconds(w.end_offset),
                    "confidence": getattr(w, "confidence", None),
                    "speaker_label": getattr(w, "speaker_label", "") or None,
                })

        ok(f"detected languages in response: {seen_languages}")
        ok(f"total words: {len(all_words)}")
        if not all_words:
            fail("zero words returned -- Chirp 3 produced no word-level data")
            return 5

        # ---- Contract checks --------------------------------------------
        banner("5. Word-level contract checks")
        words_with_start = sum(1 for w in all_words if w["start"] > 0)
        words_with_end = sum(1 for w in all_words if w["end"] > 0)
        words_with_conf = sum(1 for w in all_words if w["confidence"] is not None)

        # "Every word has populated start AND end" -- strictly, word[0]
        # may legitimately have start=0.0. So check the OR semantics: a
        # word is "timestamped" if it has end > 0.
        timestamped = sum(1 for w in all_words if w["end"] > 0)
        all_timestamped = timestamped == len(all_words)
        # Confidence: known Chirp 3 vendor gap, not a contract check.
        # words_with_conf is reported as a data point below but does not
        # gate the pass/fail decision.

        info(f"words with start_offset > 0: {words_with_start}/{len(all_words)}")
        info(f"words with end_offset   > 0: {words_with_end}/{len(all_words)}")
        if all_timestamped:
            ok("CONTRACT: every word has populated end_offset")
        else:
            fail(f"CONTRACT VIOLATION: {len(all_words) - timestamped} words "
                 f"with end_offset == 0 -- segment-level fallback likely")
        # Per-word confidence is a known Chirp 3 vendor gap. Not a
        # contract violation; just an honest data note.
        ok(f"confidence not supported by Chirp 3 (vendor limitation, not our "
           f"error). {words_with_conf}/{len(all_words)} words had any "
           f"confidence value -- expected 0.")

        # ---- Preview + spot-check ---------------------------------------
        banner("6. First 80 words (preview)")
        print("   ", " ".join(w["word"].strip() for w in all_words[:80]))

        banner("7. 5 random word picks for audio spot-check")
        info("Play the source audio at each [start..end] timecode and")
        info("confirm the word matches what's spoken within +/-500ms.")
        print()
        import random
        random.seed(42)
        n = min(5, len(all_words))
        picks = sorted(random.sample(range(len(all_words)), n))
        for i, idx in enumerate(picks, 1):
            w = all_words[idx]
            conf = (f"  conf={w['confidence']:.3f}"
                    if w["confidence"] is not None else "")
            print(f"     {i}.  [#{idx:5d}]  "
                  f"{w['start']:7.3f}s -- {w['end']:7.3f}s   "
                  f"{w['word']!r}{conf}")

        # ---- Cost estimate ---------------------------------------------
        banner("8. Cost estimate")
        cost = billed_secs / 60.0 * CHIRP3_USD_PER_MINUTE
        info(f"billed audio duration: {billed_secs:.1f}s ({billed_secs/60:.2f} min)")
        info(f"price per minute:      ${CHIRP3_USD_PER_MINUTE:.4f}")
        ok(f"estimated cost:        ${cost:.4f}")

        # ---- Raw response dump for debugging ----------------------------
        banner("9. Raw response (debugging)")
        # MessageToDict is convenient for printing, but it can be huge.
        # We dump to a JSON file alongside step4_diag/ for offline inspection.
        from google.protobuf.json_format import MessageToDict
        raw_dict = MessageToDict(response._pb, preserving_proto_field_name=True)
        out_dir = (PIPELINE_V2_ROOT / "tests" / "fixtures" / "step4_diag")
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = out_dir / "chirp3_probe_raw.json"
        raw_path.write_text(json.dumps(raw_dict, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        info(f"raw response dumped: {raw_path}")
        # Print a small structural summary inline
        print(json.dumps({
            "results_count": len(response.results),
            "files": list(response.results.keys()),
            "first_file_result_keys": list(raw_dict.get("results", {}).keys())[:1],
        }, indent=2))

        banner("Summary")
        if all_timestamped:
            print("  CONTRACT PASS: word-level timestamps present on every word.")
            print(f"  - {len(all_words)} words for {billed_secs:.1f}s audio "
                  f"({len(all_words) / max(billed_secs/60, 0.01):.0f} wpm)")
            print("  - per-word confidence: not supported by Chirp 3 (vendor")
            print("    limitation, documented; Stage 2 will treat None as")
            print("    'unknown', not 'zero').")
            print("  Next: user spot-check on the 5 picks above before the")
            print("  full Chirp3Provider class is written.")
            return 0
        else:
            print("  CONTRACT VIOLATION -- word-level timestamps missing on")
            print("  one or more words. Chirp 3 is a no-go without further")
            print("  investigation. See section 5 for which check failed.")
            return 6

    finally:
        # CLEANUP DECISION MATRIX (do NOT relax without rereading the
        # earlier bug report about a deleted GCS file corrupting an
        # in-flight Chirp 3 read):
        #
        #   operation_submitted=False
        #     -> server never saw the file. Safe to delete now.
        #   operation_terminal=True
        #     -> server reached done=true. Safe to delete now.
        #   otherwise (submitted but not terminal)
        #     -> operation may STILL be running server-side even though
        #        our Python side hit an exception (e.g. ResourceExhausted
        #        on polling). Deleting the file now races the server's
        #        read and causes the operation to fail with NOT_FOUND.
        #        Leave the file alone; the bucket's 24h lifecycle rule
        #        will collect it.
        if not operation_submitted:
            try:
                if blob.exists():
                    blob.delete()
                    ok(f"cleanup: deleted {gcs_uri} (no operation submitted)")
            except Exception as exc:
                fail(f"cleanup: failed to delete {gcs_uri}: {exc}")
        elif operation_terminal:
            try:
                if blob.exists():
                    blob.delete()
                    ok(f"cleanup: deleted {gcs_uri} (operation reached terminal state)")
            except Exception as exc:
                fail(f"cleanup: failed to delete {gcs_uri}: {exc}")
        else:
            op_name = operation.operation.name if operation is not None else "<unknown>"
            info(
                f"cleanup: SKIPPED -- operation {op_name} was IN-FLIGHT when "
                f"this code path ran. The GCS source file at {gcs_uri} is "
                f"left in place so the running Chirp 3 read doesn't fail "
                f"with NOT_FOUND. The bucket's 24h lifecycle rule will "
                f"clean it up automatically."
            )


if __name__ == "__main__":
    sys.exit(main())
