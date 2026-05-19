"""Step 4.5 Sarvam STT probe.

Empirical verification of Sarvam's Batch API ``timestamps`` granularity.
The docs describe ``words[]`` as "chunk-level", but the wording is
ambiguous enough that documented evidence alone isn't proof. This
probe makes the API call ourselves and inspects the actual response
JSON for two cases:

  - WORD-LEVEL: each ``words[i]`` is a single Telugu token, no
    embedded whitespace -> avg subwords-per-item ~1.0. Sarvam joins
    the existing 3-provider lineup; we build the full provider.

  - PHRASE-LEVEL: each ``words[i]`` is a multi-word phrase, embedded
    spaces -> avg subwords-per-item > 1.5. Sarvam fails the
    architectural contract; we keep it deferred to the backlog.

Bypasses ``pipeline_v2.stages.stt`` entirely. Sarvam is NOT
registered as a provider until this probe's verdict comes back
word-level.

Usage::

    python pipeline_v2/scripts/step4_5_sarvam_probe.py \\
        --fixture E:/kaizer\\ new\\ data\\ training/videos/test.mp3 \\
        --language te-IN \\
        [--brief "..." --names "x,y,z"]

Required env (loaded from KaizerBackend/.env):
  SARVAM_API_KEY=<key from dashboard.sarvam.ai>

Notes on Sarvam's API shape (verified from sarvamai SDK 0.1.28):
  - Auth header (set by SDK): ``api-subscription-key: <key>``
    (NOT ``Authorization: Bearer ...`` -- non-standard convention)
  - Multi-step batch workflow:
        initialise(parameters) -> job_id
        get_upload_links(job_id, files) -> signed PUT URLs
        upload audio via HTTP PUT
        start(job_id)
        poll get_status(job_id) until Completed/Failed
        get_download_links(job_id, files) -> signed GET URLs
        download output JSON
  - Storage backend (Azure / Google / Local) is Sarvam-managed. We
    don't have a GCS-mid-flight risk like Chirp 3. No cleanup needed
    on our side -- Sarvam's lifecycle handles the input audio.
  - **No documented model adaptation API**. ``brief`` and ``names``
    have no native home in Sarvam's params. The probe logs this gap
    explicitly but proceeds without biasing -- the timestamps
    question is independent of adaptation quality.
"""

from __future__ import annotations

import argparse
import json
import os
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


# --- Constants ---------------------------------------------------------

# Word-vs-phrase heuristic: average whitespace-separated subwords per
# items[i] entry. Set the boundary between WORD-LEVEL and PHRASE-LEVEL.
WORD_LEVEL_AVG_THRESHOLD = 1.5

# Sarvam pricing band (last verified 2026-05-18). Used only for the
# probe's "if word-level confirmed, here's projected cost" report.
# Re-verify before building the full provider class.
SARVAM_USD_PER_MIN_LOW = 0.006     # tier floor
SARVAM_USD_PER_MIN_HIGH = 0.009    # tier ceiling

# Polling (Sarvam batch jobs typically complete in <60s for short audio
# but we follow the Chirp 3 cadence with longer headroom in case the
# fresh account / new region quota is slow).
POLL_INTERVAL_SEC = 10
TIMEOUT_SEC = 600


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
    # Common placeholder strings -- catch before the API call to give a
    # clearer error than Sarvam's 401.
    if v.lower() in ("<key>", "<your-api-key>", "your_key_here",
                     "your-sarvam-key", "placeholder"):
        sys.exit(f"FAIL: env var {name!r} looks like a placeholder ({v!r}). "
                 f"Replace with a real key from dashboard.sarvam.ai.")
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
    ap.add_argument("--language", default="te-IN", type=str,
                    help="BCP-47 code expected by Sarvam (default te-IN). "
                         "Sarvam's saaras:v3 supports 23 Indian languages.")
    ap.add_argument("--brief", default="", type=str,
                    help="Free-text content description -- LOGGED for "
                         "context but NOT sent (Sarvam has no documented "
                         "adaptation API).")
    ap.add_argument("--names", default="", type=str,
                    help="Comma-separated proper nouns -- LOGGED for "
                         "context but NOT sent (same as --brief).")
    ap.add_argument("--model", default="saaras:v3", type=str,
                    choices=["saaras:v3", "saarika:v2.5"],
                    help="Sarvam model (saaras:v3 covers 23 languages and "
                         "supports modes; saarika:v2.5 is the smaller "
                         "12-language tier and ignores --mode).")
    ap.add_argument("--mode", default="transcribe", type=str,
                    choices=["transcribe", "translate", "verbatim",
                             "translit", "codemix"],
                    help="Sarvam mode -- only honored by saaras:v3, "
                         "ignored by saarika:v2.5. Verbatim preserves "
                         "filler words / spoken numbers; useful for the "
                         "word-vs-phrase config sweep.")
    ap.add_argument("--use-realtime-endpoint", action="store_true",
                    help="Bypass the batch flow and use client."
                         "speech_to_text.transcribe() sync endpoint. "
                         "Caps at ~30s audio per call; not production-"
                         "usable for our 5-30min content but lets us "
                         "probe whether the real-time API returns "
                         "different timestamp granularity than batch.")
    ap.add_argument("--timeout-sec", default=TIMEOUT_SEC, type=int)
    ap.add_argument("--poll-interval-sec", default=POLL_INTERVAL_SEC, type=int)
    ap.add_argument("--reattach-job-id", default=None, type=str,
                    help="Skip initialise/upload/start/poll; jump straight "
                         "to download for an existing job_id (saves time "
                         "and the cost of re-transcribing during retry "
                         "investigation). The job must already be in "
                         "Completed state. The --fixture arg is still "
                         "required (used for the input filename mapping "
                         "and for the size/preview context).")
    args = ap.parse_args()

    _load_env()

    banner("Step 4.5 Sarvam probe -- empirical word-vs-phrase verification")

    api_key = _require_env("SARVAM_API_KEY")
    info(f"SARVAM_API_KEY: {api_key[:6]}...{api_key[-4:]} ({len(api_key)} chars)")

    fixture = Path(args.fixture).resolve()
    if not fixture.is_file():
        sys.exit(f"FAIL: fixture not found at {fixture}")
    size_mb = fixture.stat().st_size / (1024 * 1024)
    info(f"fixture: {fixture}  ({size_mb:.1f} MB)")
    info(f"language: {args.language}")
    info(f"model: {args.model}")

    names_list = [n.strip() for n in args.names.split(",") if n.strip()]
    info(f"names ({len(names_list)}): {names_list}")
    info(f"brief: {args.brief[:80]!r}" +
         ("..." if len(args.brief) > 80 else ""))

    if args.brief or names_list:
        info("WARNING: Sarvam has no documented model-adaptation API "
             "(no equivalent of Whisper's prompt / Deepgram's keyterm). "
             "The brief and names will NOT be applied -- they're logged "
             "here for record-keeping only. If Sarvam adds adaptation in "
             "a future SDK release, wire it in then.")

    # Lazy imports after env load so any import-time client config
    # reads the right values.
    from sarvamai import SarvamAI
    from sarvamai.requests import SpeechToTextJobParametersParams
    import httpx

    client = SarvamAI(api_subscription_key=api_key)

    # ------------------------------------------------------------------
    # Real-time endpoint path: bypass the batch flow entirely. Useful
    # for the word-vs-phrase config sweep -- if the sync endpoint
    # returns different timestamp granularity than batch, that's a
    # signal the batch API is doing phrase-chunking server-side and
    # is potentially fixable by a configuration flag we haven't found
    # yet. ~30s cap on the audio; this is research-only, NOT a
    # production transcription path for our 5-30min content.
    # ------------------------------------------------------------------
    if args.use_realtime_endpoint:
        banner("REALTIME: bypass batch flow, use sync endpoint")
        info(f"model={args.model}  mode={args.mode}  "
             f"language={args.language}")
        info("note: sync REST endpoint caps at ~30s audio per call")
        if args.model == "saarika:v2.5" and args.mode != "transcribe":
            info(f"WARNING: --mode={args.mode!r} is ignored when "
                 f"--model=saarika:v2.5 (mode only applies to saaras:v3)")

        t_call_start = time.perf_counter()
        with open(fixture, "rb") as f:
            sync_response = client.speech_to_text.transcribe(
                file=f,
                model=args.model,
                mode=args.mode,
                language_code=args.language,
            )
        elapsed = time.perf_counter() - t_call_start
        ok(f"sync endpoint returned in {elapsed:.1f}s")

        # SpeechToTextResponse is a Pydantic model with the same
        # TimestampsModel shape as the batch output. Convert to dict so
        # the rest of the flow (steps 6+) works unchanged.
        response_json = sync_response.model_dump(mode="json")

        # Persist raw output.
        out_dir = (PIPELINE_V2_ROOT / "tests" / "fixtures" / "step4_diag")
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = out_dir / "sarvam_probe_raw.json"
        raw_path.write_text(
            json.dumps(response_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        info(f"raw response dumped: {raw_path}")

        # Fall through to step 6 (word-vs-phrase check). The branch
        # below for reattach / fresh submission is skipped entirely.
        filename = fixture.name              # used by later steps if any
        job_id = sync_response.request_id or "realtime-no-id"

    # ------------------------------------------------------------------
    # Reattach path: skip submission, jump straight to result inspection
    # for an existing completed job. Useful when the upload + transcribe
    # succeeded on a prior run but the download step failed -- we
    # don't want to re-spend the API cost just to retry the parsing.
    # ------------------------------------------------------------------
    elif args.reattach_job_id:
        banner("REATTACH: skipping submission, jumping to step 5")
        job_id = args.reattach_job_id
        info(f"reattaching to job_id: {job_id}")
        existing = client.speech_to_text_job.get_job(job_id)
        if not existing.is_complete():
            fail(f"job {job_id} is not in a terminal state. Cannot "
                 f"download results yet. Wait for it to finish.")
            return 10
        if not existing.is_successful():
            fail(f"job {job_id} did NOT succeed (is_failed={existing.is_failed()}). "
                 f"Nothing to download.")
            return 11
        ok("reattached to existing completed job")
        # Skip directly to step 5 (download via SDK helper).
        # The helper handles filename discovery for us, so the old
        # output-filename guessing logic isn't needed.
        # We'll reach step 5 via the shared download block below.
        # ------------------------------------------------------------
        # Variables that the rest of the flow needs but normally come
        # from steps 1-4 in fresh-submission mode:
        # ------------------------------------------------------------
        filename = fixture.name              # for input -> output mapping
        # Skip the polling-time tracking; we don't have it on reattach.
    else:
        # ------------------------------------------------------------------
        # 1. Initialise job
        # ------------------------------------------------------------------
        banner("1. Initialise batch job")
        params: SpeechToTextJobParametersParams = {
            "model": args.model,
            "language_code": args.language,
            "mode": args.mode,                  # only honored by saaras:v3
            "with_timestamps": True,            # CRITICAL -- the whole point
            "with_diarization": True,           # collect speaker labels too
        }
        info(f"job parameters: {dict(params)}")
        init_resp = client.speech_to_text_job.initialise(job_parameters=params)
        job_id = init_resp.job_id
        ok(f"job_id: {job_id}")
        info(f"storage_container_type: {init_resp.storage_container_type}")
        info(f"initial state: {init_resp.job_state}")

        # ------------------------------------------------------------------
        # 2. Get upload links + PUT the audio
        # ------------------------------------------------------------------
        banner("2. Upload audio (signed-URL PUT)")
        filename = fixture.name
        upload_resp = client.speech_to_text_job.get_upload_links(
            job_id=job_id, files=[filename],
        )
        file_url_details = upload_resp.upload_urls.get(filename)
        if file_url_details is None:
            fail(f"no upload URL returned for filename {filename!r}; "
                 f"got keys: {list(upload_resp.upload_urls)}")
            return 2
        signed_url = file_url_details.file_url
        info(f"signed PUT URL host: {signed_url.split('/')[2] if '://' in signed_url else '?'}")

        audio_bytes = fixture.read_bytes()

        # Build the PUT headers based on the storage backend Sarvam picked
        # for this job. The upload-link response echoes
        # ``storage_container_type`` (Azure / Azure_V1 / Google / Local).
        #
        # Azure Blob Storage REJECTS block-blob PUTs that lack the
        # ``x-ms-blob-type: BlockBlob`` header (MissingRequiredHeader 400),
        # which is what tripped the first run. Google Cloud Storage signed
        # URLs sign the request based on the canonical header set; adding
        # an extra ``x-ms-blob-type`` would mismatch the signature, so we
        # only emit it for Azure backends. "Local" is Sarvam's dev-mode
        # passthrough; no Azure headers either.
        #
        # If Sarvam's response includes a recommended Content-Type in
        # ``file_metadata``, honor it. Otherwise default to audio/mpeg
        # (matches our mp3 fixture).
        put_headers: dict[str, str] = {}
        sct = str(getattr(upload_resp, "storage_container_type", "") or "")
        if sct in ("Azure", "Azure_V1"):
            put_headers["x-ms-blob-type"] = "BlockBlob"
            info(f"storage backend = {sct}; sending Azure required header "
                 f"x-ms-blob-type=BlockBlob")
        else:
            info(f"storage backend = {sct or '<unknown>'}; "
                 f"no Azure-specific headers added")

        file_metadata = getattr(file_url_details, "file_metadata", None) or {}
        recommended_ct = (
            file_metadata.get("Content-Type")
            or file_metadata.get("content_type")
            or file_metadata.get("contentType")
        )
        if recommended_ct:
            put_headers["Content-Type"] = recommended_ct
            info(f"using Content-Type from file_metadata: {recommended_ct}")
        else:
            put_headers["Content-Type"] = "audio/mpeg"

        t_upload_start = time.perf_counter()
        put_resp = httpx.put(
            signed_url,
            content=audio_bytes,
            headers=put_headers,
            timeout=300.0,
        )
        upload_secs = time.perf_counter() - t_upload_start
        if put_resp.status_code not in (200, 201):
            fail(f"signed PUT failed (rc={put_resp.status_code}): "
                 f"{put_resp.text[:500]}")
            return 3
        ok(f"uploaded {len(audio_bytes) / 1024 / 1024:.1f} MB in {upload_secs:.1f}s")

        # ------------------------------------------------------------------
        # 3. Start the job
        # ------------------------------------------------------------------
        banner("3. Start batch job")
        start_resp = client.speech_to_text_job.start(job_id=job_id)
        info(f"state after start: {start_resp.job_state}")

        # ------------------------------------------------------------------
        # 4. Poll until terminal
        # ------------------------------------------------------------------
        banner("4. Poll job status")
        t_poll_start = time.perf_counter()
        deadline = t_poll_start + args.timeout_sec
        terminal_states = {"Completed", "Failed"}
        final_status = None
        while True:
            status = client.speech_to_text_job.get_status(job_id=job_id)
            elapsed = time.perf_counter() - t_poll_start
            info(f"  state={status.job_state}  elapsed={elapsed:.0f}s  "
                 f"(success={status.successful_files_count} "
                 f"fail={status.failed_files_count})")
            if str(status.job_state) in terminal_states:
                final_status = status
                break
            if time.perf_counter() > deadline:
                fail(f"timeout after {args.timeout_sec}s; "
                     f"last state was {status.job_state}. "
                     f"job_id={job_id} (check dashboard.sarvam.ai for status)")
                return 4
            time.sleep(args.poll_interval_sec)

        if str(final_status.job_state) != "Completed":
            fail(f"job ended in state {final_status.job_state!r}; "
                 f"error_message={final_status.error_message!r}; "
                 f"job_id={job_id}")
            return 5
        ok(f"job completed in {time.perf_counter() - t_poll_start:.1f}s")

    # ------------------------------------------------------------------
    # 5a. Discover the output filename via per-file metadata
    # ------------------------------------------------------------------
    # SKIPPED in --use-realtime-endpoint mode: the sync endpoint returned
    # the transcript directly into response_json already; no batch
    # outputs to fetch.
    #
    # Earlier iterations got the batch download wrong twice:
    #   - First: guessed "test.json" (stem + .json), 400 BadRequest
    #     because Sarvam doesn't use the input-filename naming.
    #   - Second: used get_file_results() and treated its output as
    #     the transcript. WRONG -- get_file_results() returns
    #     PROCESSING METADATA per file:
    #         {"file_name": "test.mp3", "status": "Success",
    #          "error_message": "", "output_file": "0.json"}
    #     The actual transcript lives at the path named by
    #     metadata["output_file"]. Sarvam uses a sequential
    #     "<index>.json" convention for batch outputs (the first
    #     file is "0.json", the second "1.json", etc.) -- document
    #     this in the future SarvamProvider class.
    if args.use_realtime_endpoint:
        banner("5. SKIPPED -- realtime endpoint already populated response_json")
    else:
        banner("5a. Discover output filename from per-file metadata")
        job_handle = client.speech_to_text_job.get_job(job_id)

        # For transparency, log the input -> output mapping.
        try:
            mappings = job_handle.get_output_mappings()
            info(f"input -> output mapping: {mappings}")
        except Exception as exc:
            info(f"(get_output_mappings unavailable: "
                 f"{type(exc).__name__}: {exc})")

        file_results = job_handle.get_file_results()
        # file_results shape: {input_filename: [metadata_dict, ...]}
        file_meta = None
        if filename in file_results and file_results[filename]:
            file_meta = file_results[filename][0]
        elif file_results:
            # Fall back to first available -- Sarvam may have re-keyed.
            first_key = next(iter(file_results))
            info(f"input filename {filename!r} not in file_results; "
                 f"falling back to first key {first_key!r}")
            file_meta = file_results[first_key][0]
        if not file_meta:
            fail("get_file_results() returned an empty dict; "
                 "no transcript available")
            return 6
        info(f"file metadata: {file_meta}")

        if str(file_meta.get("status", "")).lower() != "success":
            fail(f"file processing did NOT succeed: "
                 f"status={file_meta.get('status')!r}, "
                 f"error={file_meta.get('error_message')!r}")
            return 7

        output_filename = file_meta.get("output_file") or ""
        if not output_filename:
            fail(f"metadata has no 'output_file' field: {file_meta}")
            return 8
        ok(f"output filename per metadata: {output_filename!r}")

        # --------------------------------------------------------------
        # 5b. Fetch the actual transcript content (two-tier)
        # --------------------------------------------------------------
        banner("5b. Fetch transcript content")
        sarvam_download_dir = (
            PIPELINE_V2_ROOT / "tests" / "fixtures" / "step4_diag"
            / "sarvam_download"
        )
        sarvam_download_dir.mkdir(parents=True, exist_ok=True)
        for stale in sarvam_download_dir.glob("*"):
            try:
                stale.unlink()
            except OSError:
                pass

        response_json = None
        transcript_source = ""

        # --- Tier 1: SDK helper ---
        try:
            info("attempting SDK download_outputs()")
            success = job_handle.download_outputs(
                output_dir=str(sarvam_download_dir),
            )
            if not success:
                info("download_outputs returned False; falling through")
            else:
                downloaded = sorted(sarvam_download_dir.glob("*.json"))
                if not downloaded:
                    info(f"download_outputs returned True but no *.json found "
                         f"in {sarvam_download_dir}; falling through")
                else:
                    info(f"download_outputs wrote {len(downloaded)} file(s): "
                         f"{[f.name for f in downloaded]}")
                    content_path = next(
                        (f for f in downloaded if f.name == output_filename),
                        downloaded[0],
                    )
                    response_json = json.loads(
                        content_path.read_text(encoding="utf-8")
                    )
                    transcript_source = f"download_outputs -> {content_path.name}"
                    ok(f"loaded {content_path.name} "
                       f"({content_path.stat().st_size} bytes)")
        except Exception as exc:
            info(f"download_outputs raised {type(exc).__name__}: {exc}; "
                 f"falling through to manual download")

        # --- Tier 2: manual get_download_links() + httpx.get() ---
        if response_json is None:
            info(f"manual fallback: get_download_links(files=[{output_filename!r}])")
            download_resp = client.speech_to_text_job.get_download_links(
                job_id=job_id, files=[output_filename],
            )
            dl = download_resp.download_urls.get(output_filename)
            if dl is None and download_resp.download_urls:
                actual_key = next(iter(download_resp.download_urls))
                info(f"{output_filename!r} not in download_urls; "
                     f"using first available {actual_key!r}")
                dl = download_resp.download_urls[actual_key]
            if dl is None:
                fail(f"no download URL returned by Sarvam for "
                     f"{output_filename!r}; cannot inspect transcript")
                return 9
            get_resp = httpx.get(dl.file_url, timeout=120.0)
            if get_resp.status_code != 200:
                fail(f"manual download failed (rc={get_resp.status_code}): "
                     f"{get_resp.text[:500]}")
                return 10
            response_json = get_resp.json()
            transcript_source = f"manual httpx.get -> {output_filename}"
            ok(f"manual download succeeded ({len(get_resp.content)} bytes)")

        info(f"transcript source: {transcript_source}")

        # Persist raw output for offline inspection.
        out_dir = (PIPELINE_V2_ROOT / "tests" / "fixtures" / "step4_diag")
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = out_dir / "sarvam_probe_raw.json"
        raw_path.write_text(
            json.dumps(response_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        info(f"raw response dumped: {raw_path}")

    # ------------------------------------------------------------------
    # 6. THE CRITICAL CHECK: word-vs-phrase granularity
    # ------------------------------------------------------------------
    banner("6. Word-vs-phrase granularity check (THE GATE)")

    timestamps = response_json.get("timestamps") or {}
    items_words = timestamps.get("words") or []
    items_start = timestamps.get("start_time_seconds") or []
    items_end = timestamps.get("end_time_seconds") or []

    info(f"items in words[]:              {len(items_words)}")
    info(f"items in start_time_seconds[]: {len(items_start)}")
    info(f"items in end_time_seconds[]:   {len(items_end)}")

    if not items_words:
        fail("words[] is EMPTY -- no timestamps returned at all. "
             "Sarvam failed to produce timestamps for this audio. "
             "Cannot draw word-vs-phrase conclusion.")
        return 8

    if not (len(items_words) == len(items_start) == len(items_end)):
        fail("words[] / start_time_seconds[] / end_time_seconds[] are "
             "DIFFERENT lengths. Response shape is broken; can't pair "
             "tokens with timestamps cleanly.")
        return 9

    # Compute the heuristic: average whitespace-separated tokens per
    # items[i]. ~1.0 means each item is one word; >1.5 means
    # multi-word phrases.
    subword_counts: list[int] = []
    for item in items_words:
        text = (item or "").strip()
        n = len(re.split(r"\s+", text)) if text else 0
        subword_counts.append(n)

    avg_subwords = statistics.mean(subword_counts)
    max_subwords = max(subword_counts)
    items_with_2plus = sum(1 for c in subword_counts if c >= 2)
    pct_phrases = 100.0 * items_with_2plus / len(subword_counts)

    info(f"average subwords per item: {avg_subwords:.2f}")
    info(f"max subwords in any item:  {max_subwords}")
    info(f"items with 2+ subwords:    {items_with_2plus}/{len(items_words)} "
         f"({pct_phrases:.1f}%)")
    info(f"threshold: avg > {WORD_LEVEL_AVG_THRESHOLD} -> PHRASE-LEVEL")

    print()
    print("  First 10 items in words[] for visual eyeballing:")
    print("  " + "-" * 64)
    for i in range(min(10, len(items_words))):
        item_text = items_words[i]
        start = items_start[i]
        end = items_end[i]
        subw = subword_counts[i]
        print(f"  [{i:3d}] {start:7.3f}s -> {end:7.3f}s   "
              f"subwords={subw}   {item_text!r}")

    word_level = avg_subwords <= WORD_LEVEL_AVG_THRESHOLD

    # ------------------------------------------------------------------
    # 7. Preview + 5 random picks (for context, regardless of verdict)
    # ------------------------------------------------------------------
    banner("7. First 80 items + 5 random spot-checks (for context)")
    preview_n = min(80, len(items_words))
    print("   ",
          " ".join((items_words[i] or "").strip() for i in range(preview_n)))

    print()
    import random
    random.seed(42)
    picks = sorted(random.sample(range(len(items_words)), min(5, len(items_words))))
    print("   5 random items:")
    for i, idx in enumerate(picks, 1):
        print(f"     {i}.  [#{idx:5d}]  "
              f"{items_start[idx]:7.3f}s -- {items_end[idx]:7.3f}s   "
              f"{items_words[idx]!r}")

    # ------------------------------------------------------------------
    # 8. Other context (transcript, billed duration, cost estimate)
    # ------------------------------------------------------------------
    banner("8. Other response context")
    transcript_text = response_json.get("transcript", "") or ""
    detected_language = response_json.get("language_code", "") or ""
    diarized = response_json.get("diarized_transcript") or {}
    diar_entries = diarized.get("entries") if isinstance(diarized, dict) else None

    audio_dur = float(items_end[-1]) if items_end else 0.0
    info(f"detected language: {detected_language}")
    info(f"transcript length: {len(transcript_text)} chars")
    info(f"approx audio duration (from last end_time): {audio_dur:.1f}s")
    info(f"diarized speaker entries: "
         f"{len(diar_entries) if diar_entries else 0}")

    est_low = audio_dur / 60.0 * SARVAM_USD_PER_MIN_LOW
    est_high = audio_dur / 60.0 * SARVAM_USD_PER_MIN_HIGH
    info(f"cost estimate (Sarvam tier band): "
         f"${est_low:.4f} -- ${est_high:.4f}  ({audio_dur/60:.2f} min)")

    # ------------------------------------------------------------------
    # 9. VERDICT
    # ------------------------------------------------------------------
    banner("9. VERDICT")
    if word_level:
        print()
        print("  *** VERDICT: WORD-LEVEL ✓ ***")
        print()
        print("  Sarvam returns true per-word timestamps "
              f"(avg subwords/item = {avg_subwords:.2f}, threshold "
              f"{WORD_LEVEL_AVG_THRESHOLD}).")
        print("  Architectural contract IS satisfied. Sarvam joins the")
        print("  V2 launch lineup as the 4th provider. Next step: build")
        print("  pipeline_v2/stages/stt/sarvam.py mirroring the Whisper-")
        print("  Groq / Deepgram / AssemblyAI provider pattern.")
        print()
        print(f"  Caveats to bake into the provider docstring:")
        print(f"   - No documented adaptation API; brief / names are no-ops.")
        print(f"   - Multi-step batch workflow (initialise -> upload -> ")
        print(f"     start -> poll -> download) makes the sync wall time")
        print(f"     longer than single-call providers.")
        print(f"   - Telugu code is BCP-47 'te-IN' (NOT 'te' like Whisper).")
        print(f"   - Sarvam job_id for this probe run: {job_id}")
        return 0
    else:
        print()
        print("  *** VERDICT: PHRASE-LEVEL ✗ ***")
        print()
        print("  Sarvam's words[] contains multi-token phrases, not")
        print("  individual spoken words.")
        print(f"  Average subwords per item = {avg_subwords:.2f} "
              f"(threshold {WORD_LEVEL_AVG_THRESHOLD}).")
        print(f"  {pct_phrases:.1f}% of items have 2+ subwords.")
        print()
        print("  Architectural contract NOT satisfied. Stage 2 cuts on")
        print("  word boundaries; phrase-level data would land cuts")
        print("  mid-phrase, corrupt retake-skip logic, and break")
        print("  image-overlay timing.")
        print()
        print("  CONFIRMED DROP: Sarvam stays deferred to backlog.")
        print("  V2 ships with 3 providers (Whisper-Groq / Deepgram /")
        print("  AssemblyAI). Quarterly re-check via this same probe.")
        print(f"  Sarvam job_id for this probe run: {job_id}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
