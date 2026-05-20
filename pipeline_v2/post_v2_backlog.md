# Pipeline V2 — Post-Launch Backlog

Items deferred from the V2 build, kept here so future maintainers (or
future-me) can find them.

---

## Schema evolution log

### Step 0 (Plan A prompt swap) — 2026-05-16

The Step 0 prompt swap intentionally evolved the v1 Gemini JSON schema.
Future readers of v1's `pipeline_core/pipeline.py` will see a schema
that diverges from any pre-Step-0 documentation. Below is the diff for
context.

**`clips[*]` (per-clip object)**
| Before | After |
|---|---|
| `hook` (string) | renamed → `summary` |
| — | added `summary_native` (Telugu/native rendering of summary) |
| — | added `mood` (tone/emotion tag) |
| — | added `speakers` (list of speaker labels) |
| `importance` | unchanged |
| `index`, `start`, `end` | unchanged |

**`shorts_cuts[*]` (per-short object)**
| Before | After |
|---|---|
| `index`, `hook`, `importance` | **removed** |
| — | added `summary` |
| `shorts_headline_native` (was top-level) | moved per-item |
| `bulletin_marquee_points` (was top-level) | moved per-item |
| `start`, `end` | unchanged |

Net effect: each short can carry its own headline + marquee. Less
duplication when multiple shorts have different framings.

**`image_plan[*]` (per-image object)**
| Before | After |
|---|---|
| `id`, `topic_clue`, `search_query`, `search_query_native`, `reason` | **removed** |
| `entity_name`, `description`, `clip_index`, `show_at`, `duration` | kept |

Net effect: trimmed unused metadata. Smaller per-image payload.

**`skipped_segments[*]`** — unchanged (start, end, reason, category).
**`retake_audit`** — unchanged (string, mandatory, non-empty, not "SKIPPED").
**Top-level keys** — `shorts_headline_native` and `bulletin_marquee_points`
are no longer top-level (moved into shorts_cuts items as above).

**V2 implications**:
- Stage 2 in V2 produces a DIFFERENT output (continuity decisions only:
  `full_video_cuts`, `skipped_segments`, `retake_audit`). The V1 schema
  changes documented here only matter for:
  1. Understanding what changed in Step 0
  2. The V2 Stage 4 render adapter (`editor_meta_adapter.py`, Step 8)
     when it maps V2's `StageTwoOutput` into v1-compatible
     `editor_meta.json` for the existing editor tab
- The Step 0 regression baseline at
  `pipeline_v2/tests/fixtures/step5_baseline/v1_step0_solo.json` is the
  canonical reference for the current shape (SOLO mode).

---

## Re-evaluation triggers

### Chirp 3 (Google Cloud Speech-to-Text v2)
- **Deferred**: 2026-05-18. Telugu support listed but in Preview
  status; returned INTERNAL error on real Telugu audio (verified via
  `scripts/step4_1_chirp3_probe.py`).
- **Re-check cadence**: quarterly.
- **Signal sources**: Speech-to-Text v2 release notes, Google Cloud
  regions page for Chirp 3 GA expansion, `te-IN` moving from Preview
  → GA.
- **Verification path**: re-run `step4_1_chirp3_probe.py` against
  `test.mp3`. If Telugu transcribes without INTERNAL → write the full
  `Chirp3Provider` class, add to `pipeline_v2/stages/stt/__init__.py`
  auto-loader tuple.

### Sarvam AI
- **Deferred**: 2026-05-18. 4-config empirical sweep (verbatim mode,
  saarika model, real-time endpoint, all on test.mp3 / test_28s.mp3)
  all returned phrase-level timestamps OR no timestamps. Architectural
  contract requires word-level.
- **Re-check cadence**: quarterly.
- **Signal sources**: sarvamai PyPI release notes, docs page updates
  to `speech-to-text-batch-api` regarding word-level support.
- **Verification path**: re-run `scripts/step4_5_sarvam_probe.py`
  against `test.mp3`. If any of the 4 configurations now reports
  `WORD-LEVEL ✓` → build the full `SarvamProvider` class. Probe
  artifacts: `pipeline_v2/tests/fixtures/step4_diag/sarvam_*.log` +
  `sarvam_probe_raw.json`.

---

## Provider deferred enhancements

### Whisper-Groq audio chunking
- **Current state**: provider raises `ValueError` if audio file
  exceeds Groq's per-file cap (25 MB free tier, 100 MB dev tier).
- **Mitigation in place**: Stage 0 (ingest) extracts audio at 64kbps
  mono for the Groq path — 30 min at that bitrate is <16 MB.
- **Build only if needed**: if 64kbps mono extraction is insufficient
  (e.g. workload moves to 90+ min recordings), implement audio
  chunking with silence-boundary splits + word-timestamp stitching at
  chunk seams. Estimated ~150 LOC.

---

## Cosmetic cleanups

### Rename `Stage2Output` → `GeminiStage2Response`
- **Why**: `Stage2Output` (LLM contract) vs `StageTwoOutput` (full stage
  return) differ only in snake_case-vs-CamelCase. Future readers will
  read them as the same thing.
- **What**: rename `Stage2Output` everywhere → `GeminiStage2Response`.
  Update imports in `stages/stage_2_continuity.py`, tests, and
  `models.py`.
- **When**: convenient cleanup pass, not urgent. Defer until Step 5.5
  or after.

---

## Pre-existing bugs

### Instagram Reel emits image_plan with "?" IDs
- **Surfaced**: 2026-05-18 during Step 0 verification.
- **Root cause**: standalone-mode jobs occasionally emit `image_plan`
  entries with `id = "?"` (literal question mark). Gemini fills a
  field that arguably shouldn't be in the standalone schema.
- **Workaround**: ignored — downstream renderer handles `"?"` IDs
  gracefully today.
- **Fix when**: AFTER Step 12 (V2 E2E). Not blocking V2 launch.
- **Resolution path**: either remove `id` from the base schema, or
  prompt explicitly forbid `"?"` placeholders, or post-validate and
  drop invalid entries with structured log.

### Item 57: Groq whisper-large-v3 returns ZERO word timestamps for Telugu audio

- **Surfaced**: 2026-05-19 during Step 12.2a Path 2 investigation.
- **Evidence**: probe `scripts/step4_2_whisper_groq_probe.py` against
  the canonical Bandi Bhagirath test.mp4 returned HTTP 200 in 21.6s
  with **976 word objects** all having `start=0` / `end=0`. Probe
  exit code 6 (contract violation, NOT network failure).
- **Audio metadata**: 10.89 MB mp3, 48 kHz stereo, 128 kbps, 713.92s.
  Well under Groq's 25 MB free-tier limit. Not size-related.
- **Severity**: production blocker for any Indian-language workload on
  Groq. STT silently completes "successfully" but emits unusable data
  (downstream stages get no timing information).
- **Mitigation in place**: V2 Beta defaults to Deepgram for
  Indian-language jobs (see Item 59).
- **Recommended action**:
  1. File a Groq bug report citing model=`whisper-large-v3`,
     `response_format="verbose_json"`,
     `timestamp_granularities=["word"]`, `language="te"`.
  2. Re-test quarterly via the probe script.
- **Diagnostic artifact**: `pipeline_v2/tests/fixtures/step12_diag/groq_500_investigation/`.
- **Fix when**: Groq side; we can't patch it. Defaults to Deepgram
  until Groq fixes Telugu word-level timestamps.

### Item 58: Groq `with_raw_response.create()` 500s on 10MB+ audio files

- **Surfaced**: 2026-05-19 during Step 12.2a Path 2 investigation
  (same probe as Item 57).
- **Evidence**: V2's `whisper_groq.py:251` uses
  `client.audio.transcriptions.with_raw_response.create(**kwargs)`
  to capture `request_id` for diagnostics. The probe used plain
  `client.audio.transcriptions.create(**kwargs)` with identical
  kwargs — plain returned HTTP 200 (degenerate; see Item 57);
  `with_raw_response` consistently 500s for the same payload.
- **Severity**: V2-specific. Stacks with Item 57 — even if the plain
  API call returns 200, the `with_raw_response` wrapper still fails
  before the response payload is parsed.
- **Defer reason**: Beta uses Deepgram per Item 59; whisper-groq path
  isn't on the launch critical path.
- **When Item 57 is fixed**: also retest `with_raw_response` shape.
  If still failing, either switch V2 to plain `create()` and
  synthesise a request_id, OR file a second Groq bug report.

### Item 59: Beta launches with Deepgram as default STT for Indian-language content

- **Surfaced**: 2026-05-19 after Step 12.2a Path 2 investigation
  established whisper-groq is unusable for Telugu (Items 57+58).
- **Decision**: `KAIZER_STT_PROVIDER=deepgram` is the Beta default
  for any job whose detected language is Telugu, Hindi, or other
  Indian-language tier-1 codes. English-language jobs may use
  whisper-groq once Groq's Telugu issue (Item 57) is independently
  verified to NOT affect English transcription.
- **Cost impact**: Deepgram is ~$0.12 per 12-min job vs Groq's free
  tier. Acceptable at Beta-scale volumes.
- **UI implication**: STT provider hint in the wizard should mark
  Deepgram as "recommended for Telugu/Hindi" when language is one
  of those. Tracked in `/api/v2/stt/providers` endpoint TODO.

### Item 60: V1 `generate_image_pool_from_plan` violates the locked
policy — generates AI images of real public figures (PERSON entities)

- **Surfaced**: 2026-05-19 during Step 12.2a re-run #2 investigation.
- **File:line**: `pipeline_core/pipeline.py:1459-1579` (the function);
  called at `pipeline_core/pipeline.py:3574` (bulletin image_plan
  overlay flow, triggered whenever Gemini emits a non-empty
  `image_plan`).
- **Root cause**: V1's image_plan timeline-overlay flow is PURE
  GENERATION via OpenAI + Imagen. Even for `PERSON` entities,
  V1 calls `_oi.generate_news_image(entities=[entity_name])` and
  `_im.generate_news_image(entities=[entity_name])`. The only
  guardrail is the OpenAI/Imagen prompt template's "no real public
  figures" instruction — which works for A-list celebrities but
  not reliably for lesser-known regional figures. Empirical
  evidence: `output/youtube_full/20260516_155438/pool/openai/img_01__bandi_bhagirath.jpg`
  exists as a 169 KB generated image despite Bandi Bhagirath being
  a real Telangana MLA.
- **Why this is V1's bug, not V2's**: V2 deliberately does NOT
  inherit this flow (see Step 12.2a re-run #2 scope decision). V2's
  `pipeline_v2/pipeline_v2/stages/stage_4_image_source.py` is the
  policy-compliant replacement: PERSON entities go through
  search-only (CSE -> DDG -> Pexels), and search failures produce
  `image_unavailable` status rather than generated fakes.
- **Recommended V1 patch**: replace V1's `generate_image_pool_from_plan`
  with a flow that:
  1. Walks unique entity_names in image_plan
  2. For each, calls `search_news_images(skip_openai=True)` first
  3. On search miss, ONLY for non-PERSON entities, falls back to
     `_oi.generate_news_image(entities=[])` (matching V1's TRACK-2
     pattern at pipeline.py:3956 — never pass real-person names to gen)
  4. Drops PERSON entries that fail search with `image_unavailable`
- **Fix when**: separate decision after Step 12 ships. V2's beta
  launch is policy-compliant via its own ImageSourcer, so this
  doesn't block beta. V1 production keeps shipping the violation
  until patched.

### Item 61: V1 image-search chain ordering may need revisiting for
Indian-language coverage

- **Surfaced**: 2026-05-19 during Step 12.2a re-run #2 scope review.
- **Current order**: CSE -> DDG -> Pexels (V1's existing chain;
  preserved verbatim in V2's `ImageSourcer._search_chain`).
- **Rationale for ordering**: DDG returns news-context images; Pexels
  is stock-only. For Telugu/Hindi political-figure searches, DDG
  before Pexels is preferred. Operator noted this is the right call
  for Indian-language news today.
- **Re-evaluation trigger**: post-beta telemetry. If
  `person_image_unavailable` slug rates exceed 20% for Telugu/Hindi
  jobs, consider:
  - Adding more search sources (Bing Image Search, Yandex)
  - Re-ordering Pexels above DDG when DDG rate-limits us frequently
  - Adding a per-language CSE configuration (different CSE IDs for
    English-news vs Telugu-news image queries)
- **Fix when**: data-driven post-launch decision, NOT a planned task.

### Item 62: Gemini cost tracker not populating envelope (budget guards underestimate ~5x)

- **Surfaced**: 2026-05-19 during Step 12.2a re-run #3 manifest review.
- **Evidence**: `manifest.json.cost_usd` shows
  ``stage_2_continuity = 0.0``, ``stage_2_5_entities = 0.0``,
  ``stage_3_fanout = 0.0`` across every run, despite Gemini Pro
  Stage 2 + Gemini Flash Stages 2.5 / 3a / 3b / 3c making real
  paid API calls. Only Stage 1 Deepgram cost ($0.1154) lands
  in the ledger.
- **Estimated real spend per run**: ~$0.30-0.50 Gemini + $0.12
  Deepgram = ~$0.42-0.62 total. Ledger underreports by ~4-5x.
- **Risk**: the $5-7 Step 12 budget guard (D-12.9) reads from the
  envelope's `stage_costs` accumulator, so it'll never trigger
  the single-run-exceeds-$2 warning even if Gemini token usage
  blows up.
- **Likely root cause**: the per-stage Gemini call wrappers
  compute token costs but don't push them into the envelope dict
  (probably a missing assignment in one of `stages/stage_2_*.py`,
  `stages/stage_3*.py`). The Deepgram path does it correctly.
- **Fix scope**: thread a `cost_callback` into each Gemini call or
  add a return-tuple `(output, cost_usd)` so the orchestrator can
  accumulate into `envelope["stage_costs"]`. ~30-50 LOC.
- **Fix when**: BEFORE production launch (Beta is fine because we're
  inside cost ceilings manually). Track for Step 13.5.

### Item 63: Bulletin truncation via V1 idempotency cache + V2 two-pass pattern (FIXED in 12.2a re-run #5)

- **Surfaced**: 2026-05-19 during Step 12.2a re-run #3 quality review.
- **Symptom**: bulletin.mp4 rendered at 21s instead of the expected
  ~715s for a 12-min source.
- **Root cause**: V1's `cut_video_clips` at
  `pipeline_core/pipeline.py:1235` skips the cut when
  `<output_dir>/raw_clip_NN.mp4` already exists at >100KB. V2's
  shorts pass and bulletin pass both wrote to the same
  `output_dir` → bulletin pass found shorts' 21s raw_clip_01.mp4
  and reused it instead of cutting the 715s full_video_cut.
  Compose then truncated to 21s via FFmpeg `-shortest`.
- **Fix**: `stage_4_render.py:cut_raw_bulletin_stories` now writes
  to `self.bulletin_dir` (a dedicated subdir) instead of
  `self.output_dir`. V1's cache check no longer collides.
- **Regression tests landed**:
  - `test_writes_raw_clips_to_bulletin_dir_not_output_dir`
  - `test_bulletin_raw_paths_differ_from_shorts_raw_paths`
- **Lesson**: V1 was designed for single-pass orchestration; any V2
  multi-pass pattern that reuses V1 helpers must give each pass a
  unique output namespace.

### Item 64: clip_image_map mis-keyed for shorts pass (FIXED in 12.2a re-run #5)

- **Surfaced**: 2026-05-19 during Step 12.2a re-run #3 quality review.
- **Symptom**: all 9 shorts rendered with the same image
  (`telangana__pexels_00.jpg`) despite 6 distinct images being
  sourced into `self.image_pool`.
- **Root cause**: `stage_4_render.py:compose_shorts` built
  `clip_image_map` keyed by `r["clip_index"]` from resolved image_plan
  entries. But `ImagePlanEntry.clip_index` references the
  BULLETIN's `full_video_cuts[idx]`, NOT the shorts cut index.
  With 1 full_video_cut every image_plan entry mapped to
  `clip_index=0`. Only shorts[0] hit the map; the rest hit the
  `next(iter(image_pool.values()))` fallback — same Telangana image.
- **Fix**: replaced the `clip_index` lookup with a list of
  `(source_show_at_sec, image_path)` overlays. Per shorts cut at
  `[start_sec, end_sec]`, pick the first overlay whose
  `show_at_sec` falls inside; if no overlap match, round-robin
  through the resolved image_pool by short index.
- **Regression tests landed**: 4 in
  `TestComposeShortsPerClipImageSelection`:
  - `test_compose_shorts_uses_show_at_overlap_when_available`
  - `test_compose_shorts_round_robin_when_no_overlap`
  - `test_compose_shorts_distinct_images_per_clip` (locks the
    "no two consecutive shorts share the same image" assertion)
  - `test_compose_shorts_handles_missing_source_show_at_sec`
- **Lesson**: when a field name like `clip_index` is reused across
  data models (image_plan's clip_index = bulletin cut; shorts cut's
  index = shorts cut), grep every consumer before assuming they
  reference the same index space.

### Item 65: Lenient parser invariant cascade (lesson learned)

- **Surfaced**: 2026-05-19 during Step 12.2a re-run #4 (the
  re-run that introduced this bug and the re-run #5 fix).
- **Pattern**: any parser that DROPS entries from a list-shaped LLM
  response (Option E style "tolerant" parsing) MUST re-establish
  every positional invariant before passing the data forward.
  Examples of invariants that downstream code may silently
  depend on:
    - Sequential 0-based `index` field (broken Stage 3a -> Stage 4)
    - Contiguous time-window coverage (e.g. show_at_sec sequences)
    - Top-level count thresholds (e.g. `min_length` Pydantic
      constraints — handled, but worth listing)
    - Cross-reference integrity (e.g. `clip_index` referring to
      a still-present array position)
- **What broke**: Stage 3a's Option E lenient parser kept the
  Gemini-emitted `index` field on each surviving short. Downstream
  `build_v1_shorts_editor_meta`'s D-8.12 contiguity guardrail
  asserts indices are 0-based contiguous. Gaps from dropped
  entries triggered ValueError mid-Stage-4.
- **Fix**: renumber surviving entries via `model_copy(update=...)`
  AT the parser boundary, before constructing the output object.
  Defensive `assert` post-condition so renumber regressions fail
  loudly at Stage 3a rather than cascading.
- **Where to apply this pattern in future stage refactors**:
  - Any Stage that converts an LLM response list to a Pydantic
    model AND has a min_length / contiguity / cross-ref invariant
    upstream consumers depend on.
  - Specifically: future "tolerant parse" patches for Stage 2.5
    (entities), Stage 3b (metadata.bulletin_marquee_points list),
    Stage 3c (image_plan entries) MUST audit downstream consumers
    for any positional assumption before relaxing parser strictness.
- **Test coverage**: 3 regression tests landed for the Stage 3a fix
  (`test_parse_renumbers_*` covers contiguity, first-position drop,
  consecutive drops).

### Item 66: Bulletin editorial trimming — V1 has NO trimming concept beyond the 60-min floor warning

- **Surfaced**: 2026-05-19 during Step 12.2a re-run #5 manifest
  review. Confirmed by reading `pipeline_core/bulletin_stitcher.py`.
- **Current state**: V1's `stitch_bulletin` emits one soft warning
  if `total_min < DEFAULT_MIN_TOTAL_MIN` (60 min). That's the only
  duration-related logic. There is NO editorial trim, NO target
  ceiling, NO segment-selection algorithm — V2 receives the full
  source duration as bulletin output.
- **Beta impact**: a 12-min source produces a 12-min bulletin. A
  90-min source would produce a 90-min bulletin. Users get
  bulletin.mp4 at source-length minus skipped_segments.
- **Production target (per operator)**: ~30-50% editorial
  compression. A 60-min recording should yield a ~20-30 min
  bulletin selecting the most share-worthy segments.
- **Fix design (rough)**: add an "editorial selector" Stage 3.5
  between Stage 2.5 (entities) and Stage 3 (fan-out). The selector
  picks 5-15 "story segments" from `full_video_cuts` by importance
  + entity density + topical diversity, and emits a
  `BulletinSegments` model. Stage 4's bulletin pass consumes
  `BulletinSegments` instead of using all `full_video_cuts`.
- **Fix when**: post-Beta polish. Beta users see full-duration
  bulletins with image overlays; the editorial compression is a
  quality lift, not a launch blocker.

### Item 67: CSE/DDG returning 0 results for Telugu queries — Pexels-only coverage is fragile

- **Surfaced**: 2026-05-19 across all Step 12.2a runs. Empirical
  signal: every successful image came from Pexels; CSE and DDG
  never returned a usable URL for any of the 6 Telugu/Indian
  entities tested.
- **Hypotheses**:
  1. `GOOGLE_CSE_ID` + `GOOGLE_API_KEY` are missing from
     production `.env` (silently returns `[]` per
     `pipeline.py:1366`).
  2. The configured Google CSE engine doesn't include Indian
     news sites in its custom search corpus.
  3. DDG rate-limits or returns zero for non-English image
     searches.
- **Risk**: Pexels is currently 100% of V2's image coverage.
  If Pexels rate-limits us (5,000 req/hr limit) or removes the
  generic stock photos we're keying on (which happens for
  specific people-name queries), all `person_image_unavailable`
  slugs spike to 100%.
- **Recommended action sequence**:
  1. Verify `GOOGLE_CSE_ID` + `GOOGLE_API_KEY` in production `.env`.
  2. Inspect the configured CSE engine at
     <https://programmablesearchengine.google.com/> — confirm it
     includes Indian news domains (NDTV, TheHindu, Eenadu, etc.)
     in its corpus, not just global English news.
  3. If CSE still returns 0 after fixing #1 and #2, switch the
     chain to query in English (transliterated) — `"Bandi
     Bhagirath"` may not return photos via CSE because the
     entity is indexed under native script. The query
     constructor at
     `pipeline_v2/pipeline_v2/stages/stage_4_image_source.py:_build_person_query`
     already uses the canonical Latin-script name; verify the
     CSE engine indexes pages where that Latin form appears.
  4. Add a 4th source (Bing Image Search) as a redundancy if CSE
     remains weak post-launch.
- **Fix when**: BEFORE production launch. Beta is fine on
  Pexels-only for the canonical test fixture; production must
  not depend on a single source.

### Item 70: Inngest SDK 0.5.18 exposes both `send` (async) and `send_sync` (sync)

- **Surfaced**: 2026-05-19 during Step 12.2b test authoring.
- **Pattern**: ``Inngest.send(events) -> list[str]`` is async;
  ``Inngest.send_sync(events) -> list[str]`` is the sync twin.
  Both return the same shape (a list of event ID strings).
- **Production caller**: ``runner.py``'s V2 dispatcher is a plain
  sync function, so it uses ``send_sync``. Tests authored in plain
  sync ``def test_*`` must also use ``send_sync`` or they hit
  ``RuntimeWarning: coroutine 'Inngest.send' was never awaited``
  + ``TypeError: 'coroutine' object is not iterable``.
- **Guidance**: pick by caller context, NOT preference. If V2 ever
  exposes a public async API path that dispatches Inngest events,
  use ``send`` there. Everywhere else, ``send_sync``.

### Item 71: Inngest dev log doesn't emit a "deduplicated" signal

- **Surfaced**: 2026-05-19 during Step 12.2b re-run #3 root-cause
  investigation.
- **Behaviour**: when an event with a previously-seen ``Event.id``
  is sent inside the idempotency window (default 24h), Inngest
  dev logs ``"publishing event"`` + ``"received event"`` +
  ``"initializing fn"`` -- exactly the same trace as a fresh
  event -- but no ``FunctionRun`` record is created. The dedup
  happens internally without a distinct log line.
- **Operational implication**: when our production monitoring shows
  ``"event accepted but no run created"`` for a V2 job, the
  idempotency window is the first thing to check. Confirm by
  querying Inngest's runs API with the event's external_id /
  Event.id; if no run exists despite a successful ``send``, the
  job's Event.id was previously consumed within 24h.
- **Recommended action**: when this surfaces in prod, either
  (a) wait out the 24h window or (b) use a per-attempt-unique
  Event.id (timestamp-suffix pattern) for the retry. Production
  ``runner.py`` doesn't hit this because each Job.id is unique
  per DB auto-increment, but any code path that synthesises a
  retry event MUST consider the dedup window.
- **Fix when**: if Inngest releases a server version with an
  explicit dedup log, drop this item. Until then, document in
  ops runbook.

### Item 72: Orchestrator function signature must be `(ctx)` for Inngest SDK 0.5.18, not `(ctx, step)`

- **Surfaced**: 2026-05-19 during Step 12.2b run #4.
- **Bug**: ``orchestrator.py`` declared
  ``async def process_video_v2(ctx: Context, step: Step)``.
  Inngest SDK 0.5.18's executor calls handlers with a single
  positional argument (the ``Context``); ``step`` is accessed
  via ``ctx.step``. Verified at SDK source level in
  ``inngest/_internal/execution_lib/v0.py:141`` (``await
  handler(ctx)``).
- **Symptom**: ``TypeError: process_video_v2() missing 1 required
  positional argument: 'step'`` thrown on every Inngest
  invocation. Manifests as run.status=FAILED at Inngest, never
  reaching any V2 stage code.
- **Fix landed in Step 12.2b run #5**: signature reduced to
  ``async def process_video_v2(ctx: Context)`` with
  ``step = ctx.step`` bound locally at the top of the function.
- **Regression test landed**:
  ``tests/test_main_v2.py::TestV2InngestServeMount::
  test_process_video_v2_signature_compatible_with_inngest_sdk``
  inspects ``process_video_v2._handler``'s signature and asserts
  parameters == ['ctx']. Catches any future SDK convention drift
  at unit-test time rather than after 76s of Inngest retry burn.
- **Why 12.2a missed this**: 12.2a directly drives each stage
  handler in-process; it never invokes ``process_video_v2``
  through Inngest's calling convention. Step 12.2b was
  specifically designed (D-12.2 pushback) to exercise the real
  webhook path -- and it worked.

### Item 73: Earlier setup-verification "expected 500" explanation was incorrect

- **Surfaced**: 2026-05-19 retroactively during Step 12.2b run #4
  failure analysis.
- **Correction**: during 12.2b setup-verification (the first
  stub-event probe to test the Inngest dev wiring), I reported
  the HTTP 500 errors as "expected because the dummy event
  has job_id=-1 + nonexistent video, so Stage 0 ffprobe fails".
  That was wrong. The 500s were actually the orchestrator
  signature bug (Item 72) firing on every Inngest invocation,
  not data-driven Stage 0 failure. Stage 0 never ran.
- **Ops-runbook implication**: when production monitoring shows
  Inngest run status=FAILED with HTTP 500 from the SDK, the
  first place to look is the function signature compatibility
  with the deployed SDK version (Item 72) -- NOT the input
  payload. This is the opposite of the natural "bad event" hunch.

### Item 74: V2 orchestrator's terminal-failure catch must use `except Exception`, NOT `except BaseException`

- **Surfaced**: 2026-05-19 during Step 12.2b run #5 (the run that
  reached COMPLETED at Inngest layer but FAILed at finalize_db
  check).
- **Bug**: ``orchestrator.py`` wrapped the 7-step sequence in
  ``except BaseException as exc: _mark_job_failed(job_id, exc);
  raise``. Inngest SDK 0.5.18 uses three flow-control exception
  classes (``ResponseInterrupt``, ``SkipInterrupt``,
  ``NestedStepInterrupt``) ALL inheriting from ``BaseException``
  specifically to bypass user code's ``except Exception``
  blocks. Our ``except BaseException`` over-caught them and
  fired ``_mark_job_failed`` after every step yield --
  including the final success yield.
- **Symptom (silent inconsistency)**:
  - Inngest dashboard: run.status=COMPLETED (sees the
    ResponseInterrupt as the expected success signal)
  - V2 envelope: ``finalize.status="done"`` (the finalize step
    ran successfully + wrote 8 Clip rows)
  - DB: ``Job.status='failed'`` with ``Job.error`` containing
    the entire envelope dict serialized as a fake "error
    message"
- **Fix landed in Step 12.2b run #6**: changed catch to
  ``except Exception``. Added 8-line comment block explaining
  why. PASS confirmed -- Job.status='done' stays committed.
- **Regression tests landed (2 in test_main_v2.py)**:
  1. ``test_orchestrator_terminal_catch_does_not_intercept_baseexception``
     -- AST-level grep of ``orchestrator.py`` asserting
     ``except BaseException`` is absent from the process_video_v2
     body. Catches refactor regressions at unit-test time.
  2. ``test_inngest_sdk_flow_control_exceptions_subclass_baseexception``
     -- runtime check that ResponseInterrupt/SkipInterrupt/
     NestedStepInterrupt still subclass BaseException. If
     Inngest ever flips to Exception-based flow control, this
     test fails and signals we can re-evaluate the constraint.
- **Why 12.2a missed this**: 12.2a directly drives handlers --
  ``ResponseInterrupt`` only fires under real Inngest
  ``step.run()`` machinery. Like Item 72, this is the precise
  class of bug Step 12.2b was designed to extract.
- **Production implication**: any V2 job dispatched via Inngest
  pre-fix would silently land in Job.status='failed' even when
  the actual pipeline succeeded. The Kaizer UI would show
  "failed" jobs that have full output on disk + clips in the
  DB. Beta launch absolutely requires this fix.

### Item 75: UI surface for "cancelling, finishing current step" state

- **Surfaced**: 2026-05-19 during Step 12.3 Test 1 PASS analysis.
- **Behaviour**: V2's Layer 1 cooperative cancel-check fires at
  Inngest step boundaries, NOT mid-step. So a user-initiated
  cancel during Stage 1 Deepgram (30-60s) or Stage 2 Gemini Pro
  (60-90s) shows latency before Job.status flips to 'failed'.
  Test 1 measured this empirically at 14s for Stage 1; Stage 2
  could be 60-90s in the worst case.
- **UX implication**: clicking Cancel and seeing the job stay in
  "running" state for up to 90 seconds will feel broken to
  users. The UI should distinguish:
    - normal "running" state
    - "cancelling, finishing current step" state (Job.cancel_requested=True + status='running')
    - "cancelled/failed" terminal state
- **Fix scope**: UI work in JobDetail.jsx + a small admin-panel
  hint. ~15 LOC frontend + ~0 backend (Job.cancel_requested
  already exposed by /api/jobs/{id}).
- **Fix when**: Step 12.5 (UI hints / polish pass).

### Item 76: Stage 4 internal sub-phase cancel-check (FIXED in 12.3 Test 2)

- **Surfaced**: 2026-05-19 during Step 12.3 Test 2 attempt #1.
- **Bug**: ``Stage4Render._render_impl`` had no internal cancel
  check. A user cancel mid-Stage-4 sat idle until the next
  Inngest step boundary (``_finalize_handler``'s
  ``_check_cancelled``) fired -- which only happens AFTER Stage 4
  fully completes (~5 min worst case). Meanwhile FFmpeg kept
  spawning new subprocesses, ImageSourcer kept hitting external
  APIs, etc.
- **Symptom**: V1's ``cancel_job`` (registered via
  ``_V2WorkerProxy`` in ``_ACTIVE_PROCS``) walks the worker's
  descendants ONCE and SIGKILLs FFmpegs currently running. But
  new FFmpegs spawned by Stage 4 Python code AFTER that walk
  weren't intercepted. Net effect: user clicked Cancel,
  ``cancel_job`` killed maybe one in-flight ffmpeg, Stage 4
  spawned the next one and kept going for ~5 more minutes.
- **Fix landed in Step 12.3 Test 2 #3 (PASS at 14.2s cancel
  propagation)**:
  - Added ``cancel_check: Optional[callable]`` param to
    ``Stage4Render.render()`` + ``_render_impl()``.
  - Orchestrator's ``_stage_4_render_handler`` builds a
    closure ``def _stage_4_cancel_check(): _check_cancelled(job_id)``
    and passes it to ``render(cancel_check=...)``.
  - ``_render_impl`` calls ``cancel_check()`` before each
    sub-phase: cut_raw_shorts, resolve_images, compose_shorts,
    render_bulletin (4 invocations on the full-pipeline path).
  - When ``cancel_check`` raises NonRetriableError, propagation
    exits Stage 4 via render()'s classifier and into the
    orchestrator's terminal catch (Item 74) -> Job.status='failed'.
- **Regression tests landed (3 in test_stage_4_render.py)**:
  - ``test_render_calls_cancel_check_between_sub_phases``
  - ``test_render_propagates_cancel_check_exception``
  - ``test_render_without_cancel_check_runs_normally`` (backward
    compat for V1-only call paths + existing unit tests)
- **Empirical result (Step 12.3 Test 2 PASS)**:
  - FFmpeg observed within 0.1s of Stage 4 entry (active phase)
  - Cancel triggered at t+266.23s
  - All FFmpeg descendants gone at t+266.23s + **14.19s**
    (psutil-confirmed)
  - Inngest run terminal status (FAILED) at t+266.23s + **14.20s**
  - Total cancel propagation: ~14 seconds (vs ~5 min pre-fix)
- **Production implication**: V2 now has production-grade
  cancellation across all stages. User clicks Cancel ->
  Job.status='failed' within seconds regardless of which stage
  was running. Beta-ready.

### Item 77: D-10.10 idempotency dedup verified (Step 12.4 PASS)

- **Surfaced**: 2026-05-19 during Step 12.4 PASS.
- **Behaviour verified**: sending the same ``Event`` (same
  ``Event.id``) twice within Inngest's 24h idempotency window
  creates **exactly ONE** run. The second send returns its own
  ULID event_id (Inngest assigns a new internal id) but the
  same external_id, and Inngest's dedup suppresses the second
  run creation.
- **Empirical numbers (Step 12.4 PASS)**:
  - send #1 at t+0s -> event_id=01KS0A7VT82Z5E3XPBST11JSPE
  - send #2 at t+5s -> event_id=01KS0A80PRHTT9RDR849W51099
    (different internal id; same external Event.id="job-991200400")
  - 10s after both sends: runs() filtered by appIDs+from
    returned **exactly 1 run** (id=01KS0A7VZ22V8KDJK4DGDYN6MJ)
  - Post-30s status check: RUNNING (the dedup'd run is real, not stalled)
  - Cancel triggered for cost guard at t+45s; terminal FAILED at t+119s
- **Why this matters for production**: ``runner.py``'s V2
  dispatcher relies on this contract -- if a user re-submits
  the same Job (form double-click, browser refresh, etc.), the
  V2 path uses ``Event.id=f"job-{job_id}"`` so the second
  dispatch silently dedupes against the first. Without this
  guarantee we'd double-bill API spend per duplicate submission.
- **Regression test landed**:
  ``pipeline_v2/tests/test_e2e_v2_idempotency_12_4.py`` --
  marks ``@pytest.mark.integration``, sends two identical events,
  asserts exactly 1 run. Cost-conscious (cancels mid-pipeline
  for ~$0.30 total vs ~$0.50 if let to run).
- **Operational note**: future code paths that synthesise V2
  Inngest events MUST consider whether their Event.id should
  be unique-per-attempt (e.g. retry logic that wants a fresh
  run) or stable-per-job (default, dedupes duplicates). See
  item 71 for the diagnostic-signal angle when dedup is
  unexpectedly hit in production.

### Item 78: Pre-existing V1 test failures triage (out of Step 12 scope)

- **Surfaced**: 2026-05-19 during Step 12.6 broader-suite sweep.
- **Failures (7 total, all V1-side, all pre-Step-12)**:
  - ``tests/test_encode_args.py::test_encode_args_includes_loudnorm``
  - ``tests/test_live_director_webrtc_ingest.py::test_start_spawns_two_ffmpeg_subprocesses``
  - ``tests/test_live_director_webrtc_ingest.py::test_stop_closes_stdins_and_waits``
  - ``tests/test_live_director_webrtc_ingest.py::test_chunk_pump_forwards_to_both_stdins``
  - ``tests/test_live_director_webrtc_ingest.py::test_audio_reader_pushes_pcm_to_ring``
  - ``tests/test_narrative.py::test_extract_without_gemini_key_end_to_end``
  - ``tests/test_render_modes.py::TestRenderModeConfigs::test_all_six_modes_in_config``
- **Evidence these are pre-existing, not V2 regressions**: ``git log`` on
  each path shows last touch at or before ``b3861cb`` (Gemini SDK
  migration commit, pre-Step-12). No V2 commit (``b3685e1``, ``cb8c9a9``)
  touches these tests or the V1 code paths they exercise
  (``pipeline_core/``, ``live_director/``, ``narrative.py``, render-mode
  config).
- **Why deferred from Step 12**: Step 12's scope was V2 production
  validation. V1 unit-test fixes are out of scope and should not block
  Beta launch.
- **Recommended action**: dedicated mini-pass post-Step-13 to triage
  each failure (likely candidates: stale fixtures, mocked-API drift,
  loudnorm filter graph signature change). Time budget: ~1-2 hours
  total. No production impact (these are unit tests, not behaviour
  checks).

### Item 79: Clip.job_id ON DELETE CASCADE only at the ORM layer

- **Surfaced**: 2026-05-19 during Step 13 / Part 2.9 cascade verification.
- **Behaviour observed**: ``clips.job_id`` ForeignKey is declared as
  ``Column(Integer, ForeignKey("jobs.id"))`` -- no ``ondelete`` directive.
  ``Job.clips`` relationship has ``cascade="all, delete"`` so ORM
  deletes propagate; raw SQL deletes (or Postgres-side cascade in
  multi-tenant cleanup scripts) leak orphan Clip rows.
- **Why this is NOT a Step 13 fix**: the implementation plan
  explicitly listed ``models.Clip database schema`` as a "DO NOT
  TOUCH" item -- it's V1-shared and the migration would need
  Postgres-side ``ALTER TABLE`` planning + a backfill audit.
- **Recommended action**: post-Beta clean-up pass. Update the FK to
  ``ForeignKey("jobs.id", ondelete="CASCADE")`` and ship a Phase-N
  migration with a one-shot orphan-cleanup ``DELETE FROM clips
  WHERE job_id NOT IN (SELECT id FROM jobs)``.

### Item 80: Job naming + feedback shipped (Phase 14 / D-13.11 + D-13.13 + D-13.14)

- **Status**: LANDED.
- **Scope**:
  - ``Job.name VARCHAR(120) NULL`` column.
  - ``job_feedback`` table with CHECK [0,100] + unique (job_id, user_id).
  - ``POST /api/jobs/{id}/feedback/`` + ``PATCH /api/jobs/{id}/rename/``
    + name field on ``POST /api/jobs/create/``.
  - Pure helper ``resolve_job_name(name_input, video_filename)`` in
    main.py (8-line cap-and-fallback rule).
- **Pure-helper rationale**: extracted so the cap-and-fallback rule
  is unit-testable without TestClient + filesystem mocking. 7 tests
  exercise both branches + boundary conditions.
- **UI**: ``RenamableTitle`` + ``FeedbackPanel`` components live in
  ``JobDetail.jsx``. Home.jsx prefers ``job.name`` with
  ``video_name`` fallback so pre-Phase-14 jobs render unchanged.

### Item 81: V2 user-facing stats + admin dashboard shipped (D-13.12)

- **Status**: LANDED.
- **User-facing**: ``GET /api/v2/stats/`` aggregates the calling
  user's V2 jobs only. Surface = ``/v2-stats`` React route with a
  KPI grid + success-rate card + average-rating card. Read-only.
- **Admin-facing**: ``GET /api/admin/v2-feedback`` (paginated, 50/
  page default) + ``GET /api/admin/v2-stats`` (failure breakdown,
  rating distribution, cancellation rate). Mounted at ``/admin/v2``
  via a TABS entry in Admin.jsx.
- **Failure-slug classification**: ``_classify_failure_slug()`` maps
  free-form Job.error into a stable set keyed off the
  ``permanent:*`` prefix (e.g., ``empty_file``,
  ``ffmpeg_not_found``, ``stt_failed``). Falls through to ``other``
  for unrecognised errors so the bucket never crashes.

### Item 82: V2 Beta launch infrastructure landed (D-13.5 + D-13.6)

- **Status**: LANDED (script + docs).
- **Preflight script** ``pipeline_v2/scripts/preflight_v2_launch.py``:
  5 checks (env vars, ``INNGEST_DEV=0``, ``KAIZER_V2_ENABLED=1``,
  DB reachable, Inngest Cloud reachable). 20 unit tests cover all
  five branches. ``KAIZER_PREFLIGHT_SKIP_NETWORK=1`` lets unit tests
  skip the live TCP probe.
- **Runbook** ``pipeline_v2/RUNBOOK.md``: rollback procedure (D-13.5),
  daily monitoring SQL (D-13.4), spend dashboards, common
  ``permanent:*`` failure modes, cutover gating criteria (D-13.9).
- **Changelog** ``CHANGELOG.md``: first formal entry, ships with
  the V2 Beta launch. Prior changes remain in git log only.

### Item 83: GCP Gemini spend cap (D-13.6) is operator-managed, not code

- **Surfaced**: 2026-05-19 during Step 13 / D-13.6 lock-in.
- **Status**: DOCUMENTED, action required by operator.
- **What needs doing**: in Google Cloud console -> Billing -> Budgets
  & alerts -> create a $50/day budget scoped to "Generative
  Language" service with action "Disable billing on threshold".
- **Why this is critical**: the internal cost-tracking ledger is
  broken (backlog item 62) -- a Stage-2 retry loop on Gemini 2.5
  Pro could burn $50-100/hr before manual detection. The GCP cap is
  the only mechanism that physically prevents runaway.
- **Tracked here** because the preflight script can't verify GCP-
  console state from the host, and the rollback procedure assumes
  the cap exists. Confirm the cap is in place BEFORE flipping
  ``KAIZER_V2_ENABLED=1`` in production.

### Item 84: Phased Beta rollout cohort (D-13.3)

- **Surfaced**: 2026-05-19 during Step 13 / D-13.3 lock-in.
- **Locked plan**:
  1. **First 24h**: operator-only (~2-3 self-submitted jobs).
  2. **First week**: operator + 3-5 trusted users.
  3. **Week 2+**: broader release gated by telemetry (see item 80
     admin dashboard + RUNBOOK section 4 daily checks).
- **Mechanism**: no per-user gating in code -- ``KAIZER_V2_ENABLED``
  toggles the platform card for every signed-in user. Cohort
  control is "don't tell users" for now. If the cohort expands past
  trusted-users, consider a per-user flag (e.g.
  ``users.v2_beta_enabled``) but defer until the demand arrives.

### Item 85: Cutover gate (D-13.9) -- 3 criteria, all must hold

- **Surfaced**: 2026-05-19 during Step 13 / D-13.9 lock-in.
- **Gate definition**:
  1. >= 2 weeks of Beta uptime since launch.
  2. >= 50 production V2 jobs (``platform='full_video_shorts_v2'``).
  3. ZERO P1 incidents (defined as: any event that required
     ``KAIZER_V2_ENABLED=0`` rollback) in the trailing 7 days.
- **Step 14 (default cutover) is NOT in scope** of any current
  session -- when the gate clears, re-open the discussion
  deliberately rather than acting on the gate alone. Beta -> default
  is a one-way door for V1 users; the V1 paths can stay alive
  side-by-side indefinitely if the data supports it.

### Item 86: ``test_main_v2.py`` cross-test fixture pollution (latent risk)

- **Surfaced**: 2026-05-19 during Step 13 / Part 2 test design.
- **Concern**: the ``v2_beta_app`` fixture installs FastAPI
  ``dependency_overrides`` on the shared ``main.app`` object and
  drops them in a ``finally``. If a test in a sibling file imports
  ``main.app`` directly and runs in the same pytest session, it'll
  see the wrong DB binding briefly while the override is active.
- **Empirical status**: not currently a problem -- the shared-engine
  in-memory SQLite is named uniquely
  (``kaizer_v2_beta_test``) so a sibling test that creates its own
  ``kaizer_admin_test`` engine doesn't collide. ``app.dependency_
  overrides.clear()`` runs in the fixture teardown.
- **Recommended hardening (deferred)**: switch the fixtures to use a
  copy of the app instead of the singleton, or move to
  ``starlette.testclient.TestClient`` with a per-test app factory.
  ~30 min refactor. No active production impact.

### Item 87: Post-testing secret rotation pass

- **Surfaced**: 2026-05-20 during office-LAN bring-up inventory.
- **Context**: ``kaizer/KaizerBackend/.env.testrun`` and
  ``.env.testrun_12_2b`` (both gitignored, never reached git) carry
  real Gemini / Groq / Deepgram / R2 / Postgres / Inngest Cloud
  prod credentials from Step 12 test runs. They surfaced in tool
  output during inventory because the IDE had ``.env.testrun``
  open.
- **Why deferred**: office-LAN access is gated by the Cloudflare
  Tunnel so the live blast radius today is limited; the operator
  chose to rotate post-stabilization rather than mid-bring-up.
- **Recommended action (when V2 office-LAN phase is stable)**:
  1. Rotate every key visible in ``.env.testrun*``: Gemini,
     Groq, Deepgram, R2 access key + secret, Inngest Cloud
     EVENT_KEY + SIGNING_KEY, Postgres password.
  2. Replace the values in ``kaizer/KaizerBackend/.env`` (the
     active one).
  3. Delete the test-run env files OR scrub them to placeholder
     values.
  4. Confirm ``.env*`` remains in ``.gitignore`` (already true).
- **No production impact today** -- keys still work; only rotation
  hygiene is at stake. Treat as background work.
