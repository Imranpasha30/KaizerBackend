# TRACK 3B — Failure Mode Inventory + Backlog Review + Stage 2 Determinism Analysis

Author: Track 3B autonomous agent
Date: 2026-05-21
Mode: Code-only / static analysis. NO production code changes. NO live LLM calls.

Every claim follows the format: `[CLAIM]. [EVIDENCE]. [CONFIDENCE].`

---

## 3B.1 — STAGE 2 DETERMINISM (CODE-ONLY ANALYSIS)

### Critical observation framing
Job 53 produced 1 bulletin cut; Jobs 47/49/51/52 produced 25-33 cuts on similar Telugu monologue sources. Backlog item 118 frames this as the user-facing reason the production lip-sync whack-a-mole sequence keeps surfacing on new videos — when Stage 2 collapses the source into 1 cut, every downstream "fix per cut" investment is bypassed and the legacy compose-step segmentation re-introduces drift inside a single composite range.

[CLAIM: The 1-cut-vs-28-cut variance is documented in item 118. EVIDENCE: e:\kaizer new data training\kaizer\KaizerBackend\pipeline_v2\post_v2_backlog.md:1837-1841 ("Stage 2 (Gemini) produced ONLY 1 bulletin cut covering 587.5s"). CONFIDENCE: HIGH.]

[CLAIM: Item 107 explicitly admits Gemini Pro at temperature 0.2 is "meaningfully NON-DETERMINISTIC on this task — the same iter-1 prompt produced 7/7 in one session and 5/7 in another." EVIDENCE: post_v2_backlog.md:1015-1019. CONFIDENCE: HIGH.]

### Q1 — What temperature is Stage 2 called with?

[CLAIM: The Stage 2 default temperature for Gemini is 0.2. EVIDENCE: stage_2_continuity.py:76 `DEFAULT_TEMPERATURE = 0.2`; stage_2_providers.py:222 `temperature: float = 0.2`. CONFIDENCE: HIGH.]

[CLAIM: The Stage 2 default temperature for Claude is 0.0. EVIDENCE: stage_2_continuity.py:128-133 (per-provider default branch), stage_2_providers.py:349 `temperature: float = 0.0`. CONFIDENCE: HIGH.]

[CLAIM: Temperature is wired all the way down to the SDK kwargs. EVIDENCE: stage_2_providers.py:273 (`temperature=self.temperature`) for Gemini's `GenerateContentConfig`; stage_2_providers.py:414 (`"temperature": self.temperature`) for Claude's `client.messages.parse(...)` kwargs. CONFIDENCE: HIGH.]

[CLAIM: Temperature 0 does NOT guarantee deterministic output on either provider. EVIDENCE: stage_2_providers.py:333-335 explicitly documents this in the ClaudeStage2Provider docstring — "Note that temperature=0 does NOT guarantee identical outputs." CONFIDENCE: HIGH.]

[CLAIM: Production today defaults to Gemini at T=0.2 (the higher-variance combination). EVIDENCE: stage_2_providers.py:74 `DEFAULT_PROVIDER: str = PROVIDER_GEMINI`; item 114 design note at post_v2_backlog.md:1332-1333 says "Gemini remains the default — no behaviour change for existing users". CONFIDENCE: HIGH.]

### Q2 — What guards are in place for output schema validation?

Two layers exist:

[CLAIM: Gemini provider passes `response_schema=Stage2Output` to the SDK so JSON-shape is enforced API-side. EVIDENCE: stage_2_providers.py:272 (`response_schema=Stage2Output`). CONFIDENCE: HIGH.]

[CLAIM: Gemini's `response.parsed` is deliberately NOT trusted because it silently swallows JSONDecodeError + ValidationError, returning None. The provider re-parses via `response.text` + `json.loads` + `Stage2Output.model_validate` so failures surface as exceptions for the corrective-retry layer. EVIDENCE: stage_2_continuity.py:21-24 (docstring) and stage_2_providers.py:301-314 (the explicit re-parse path). CONFIDENCE: HIGH.]

[CLAIM: Claude provider uses `client.messages.parse(output_format=Stage2Output)` for native Pydantic validation, with a documented fallback if `parsed_output` is None on the text block. EVIDENCE: stage_2_providers.py:421-479. CONFIDENCE: HIGH.]

[CLAIM: A markdown-fence stripper defends against both providers occasionally wrapping their JSON in ```json … ``` fences. EVIDENCE: stage_2_providers.py:84-99 `_strip_markdown_fences`. CONFIDENCE: HIGH.]

[CLAIM: Semantic guards (out-of-bounds indices, overlapping SkippedSegments, missing retake_audit) live in `build_clean_transcript` / `assemble_stage_two_output` and RAISE rather than catch — those are NOT covered by the corrective retry. EVIDENCE: stage_2_continuity.py:316-329 (out-of-bounds + overlap raises ValueError), stage_2_continuity.py:415-421 (empty retake_audit raises ValueError). CONFIDENCE: HIGH.]

[CLAIM: One semantic guardrail is MISSING — there is no check that "cut count is plausible" or "skipped_segments density is plausible". A 1-cut-587s output passes every existing validator. EVIDENCE: review of stage_2_continuity.py + models.py:69-90; no min-cuts / cut-density check exists. CONFIDENCE: HIGH.]

[CLAIM: A second missing guardrail — no detection that `full_video_cuts` are essentially equivalent to "no cuts" (e.g. one cut spanning >90% of input). EVIDENCE: same code review; the schema only enforces `start_word_idx <= end_word_idx < n`. CONFIDENCE: HIGH.]

### Q3 — Prompt instructions that push toward many vs few cuts

Reading stage_2_prompt.md line-by-line for cut-count guidance:

[CLAIM: The prompt strongly pushes for MANY skipped_segments via the "Phrase-level retake detection" section. Quote: "A 5-minute uncut journalist recording typically has 5-15 phrase-level retakes. If you find only 0-2 retakes in a multi-minute transcript, you are almost certainly under-detecting. Re-sweep the array looking for repeated phrase openings." EVIDENCE: stage_2_prompt.md:174-179. CONFIDENCE: HIGH.]

[CLAIM: The prompt pushes for FEW full_video_cuts, not many. Quote: "A 5-minute monologue with one off-camera interruption typically yields one FullVideoCut and one SkippedSegment." EVIDENCE: stage_2_prompt.md:61-64. CONFIDENCE: HIGH.]

[CLAIM: There is a TENSION here that explains the 1-vs-28 variance. The prompt says ideal output is "few full_video_cuts, many skipped_segments" — but the downstream pipeline historically segmented further on its own (silence-trim, micro-fragment-split), so when an earlier prompt iteration encouraged many cuts it implicitly relied on the LLM picking story-level boundaries. With the current prompt that bias is gone — and Gemini at T=0.2 stochastically lands on either "1 huge cut" (literal compliance) or "20+ cuts" (story-level interpretation). EVIDENCE: comparing stage_2_prompt.md:61-64 against item 118 questions at post_v2_backlog.md:1872-1890. CONFIDENCE: MED — the architectural tension is real, the exact stochastic landing point is not provable from code alone.]

[CLAIM: The prompt's "Reasoning order" section says "A new FullVideoCut starts where the anchor's semantic thread begins (after warm_up). A FullVideoCut continues across skipped spans if the surrounding content is part of the same thought." That description is consistent with EITHER 1 cut for a coherent monologue OR 25 cuts for a multi-story bulletin — the model has no quantitative anchor. EVIDENCE: stage_2_prompt.md:233-236. CONFIDENCE: HIGH.]

[CLAIM: The few-shot examples all have a single FullVideoCut per snippet. Eight of the nine examples emit `"full_video_cuts": [{"index": 0, ...}]`. Only the warm_up example does too. This anchors the model toward "produce few cuts." EVIDENCE: stage_2_prompt.md:286-296, 326-336, 363-373, 405-415, 449-459, 491-501, 533-543, 575-585, 615-625 (every Output example shows exactly one cut). CONFIDENCE: HIGH.]

[CLAIM: HARD RULE #6 explicitly LICENSES the 1-cut behaviour. Quote: "full_video_cuts may span over skipped regions. A single FullVideoCut with start_word_idx=0, end_word_idx=100 plus a SkippedSegment covering word_idx=40-60 means: render words 0-39 then 61-100 as one continuous clip." EVIDENCE: stage_2_prompt.md:213-217. CONFIDENCE: HIGH.]

**Summary: the prompt PERMITS the 1-cut output the user is complaining about. Job 53's 1-cut output is a literal prompt-compliant response, not a model error.**

### Q4 — Non-determinism sources beyond LLM temperature

[CLAIM: No `random.seed` / `numpy.random.seed` is set anywhere in the Stage 2 path. EVIDENCE: grep over stage_2_continuity.py + stage_2_providers.py + models.py — no random module imports. CONFIDENCE: HIGH.]

[CLAIM: No parallel/concurrent calls inside Stage 2. The provider's `decide()` is a single awaited call per attempt. EVIDENCE: stage_2_providers.py:261-314 (Gemini); 395-479 (Claude) — both make exactly one `await client.…` invocation per call. CONFIDENCE: HIGH.]

[CLAIM: The corrective retry IS a second LLM call with different input (the original payload + a correction note), which is itself a source of non-determinism in the FAILURE PATH but does not affect successful first-attempt jobs. EVIDENCE: stage_2_continuity.py:233-247. CONFIDENCE: HIGH.]

[CLAIM: There is NO per-job seed / nonce passed to either SDK. Gemini's `generate_content` and Claude's `messages.parse` are not configured with any reproducibility token. EVIDENCE: stage_2_providers.py:269-283 (Gemini config has only temperature / thinking_config / max_output_tokens / response_schema / response_mime_type); 409-416 (Claude kwargs are model / max_tokens / system / messages / temperature / thinking only). CONFIDENCE: HIGH.]

[CLAIM: Inngest's exponential-backoff outer retry is a non-determinism source at the job level — a Stage 2 step that errors will retry with a fresh LLM call, potentially producing different decisions. EVIDENCE: stage_2_continuity.py:30-32 ("No SDK-level retry. We do 1 corrective retry … second failure raises so Inngest's exponential backoff is the outer retry layer"). CONFIDENCE: HIGH.]

[CLAIM: Token-budget exhaustion is a HIDDEN non-determinism source. If `max_output_tokens=16384` is too small for a long bulletin's response, the model truncates and the corrective-retry kicks in. The retry's response is independently sampled. EVIDENCE: stage_2_continuity.py:78 `DEFAULT_MAX_OUTPUT_TOKENS = 16384`; stage_2_providers.py:304-311 (empty response.text path — fires JSONDecodeError when finish_reason=MAX_TOKENS). CONFIDENCE: HIGH.]

[CLAIM: Gemini's thinking budget shares the output-token budget; if the model "thinks" hard about a complex transcript it may leave too few tokens for the response payload. EVIDENCE: stage_2_continuity.py:25-29 (the docstring says "Thinking mode shares the max_output_tokens budget. The Gemini provider caps thinking at 2048; the Claude provider disables thinking entirely for first ship."). CONFIDENCE: HIGH.]

[CLAIM: There is NO retry counter / per-attempt jitter for Inngest's outer retry — meaning a Stage 2 that succeeds with `1 cut` will simply commit that, and there is no "we already retried twice, escalate" log. EVIDENCE: stage_2_continuity.py:202-256 (`transcribe_to_decisions` raises after 1 corrective retry; the outer Inngest layer is opaque from here). CONFIDENCE: MED — the Inngest envelope handling is in orchestrator.py which I did not need to read fully, but the unit-of-retry visible in stage_2 is fixed at 1 corrective + N (Inngest default) outer.]

[CLAIM: Word-array order/content is the ONLY input that influences the LLM output; if Deepgram returns slightly different word boundaries on identical re-ingest, that's a second-order non-determinism. EVIDENCE: stage_2_providers.py:120-143 `_build_user_payload` — the only dynamic content is the word array + audio metadata. CONFIDENCE: HIGH (about the wiring); LOW about whether Deepgram itself is bit-identical across calls.]

### Q5 — Gemini provider vs Claude provider stability characteristics

| Parameter | Gemini provider | Claude provider | File:line |
|---|---|---|---|
| Default model | `gemini-2.5-pro` | `claude-sonnet-4-6` | stage_2_providers.py:77-78 |
| Default temperature | 0.2 | 0.0 | stage_2_providers.py:222, 349 |
| Thinking | enabled, budget 2048 | DISABLED entirely | stage_2_providers.py:222, 415 |
| Structured output | `response_schema=Stage2Output` via SDK config | `client.messages.parse(output_format=Stage2Output)` | stage_2_providers.py:272 vs 421-424 |
| Prompt caching | Not configured | Ephemeral cache on system block | stage_2_providers.py:382-384 |
| Max output tokens | 16384 | 16384 | both providers |
| Response parse path | response.text → json.loads → model_validate | content[].parsed_output → fallback to text → model_validate | stage_2_providers.py:304-314 vs 452-479 |

[CLAIM: Claude has TWO structural advantages for determinism: (a) lower default temperature (0 vs 0.2), (b) thinking disabled by default (no stochastic chain-of-thought routing). EVIDENCE: row 2 + row 3 of the table above. CONFIDENCE: HIGH.]

[CLAIM: Gemini's thinking config can swing output in ways temperature alone doesn't. The model spends 0-2048 tokens "thinking" before responding; the thinking content is not in the response but the routing of those tokens through different reasoning paths is itself a stochastic process. EVIDENCE: stage_2_providers.py:275-277 + the broader behaviour of Gemini thinking-config (documented at the Google docs, MED confidence on the routing claim). CONFIDENCE: MED.]

[CLAIM: Both providers have IDENTICAL retry policies — 1 corrective retry then propagate. Provider choice doesn't change recovery semantics. EVIDENCE: stage_2_continuity.py:202-256 — the retry policy lives in the dispatcher, not the providers. CONFIDENCE: HIGH.]

[CLAIM: Cost characteristics differ in ways that may matter for variance experiments. Gemini in: $1.25/M, out: $10/M. Claude in: $3/M, out: $15/M, with cache writes at $3.75/M and cache reads at $0.30/M. To run 5 trials per source, Claude is ~2.5x more expensive on a cold cache but ~12.5x cheaper on the static prompt portion after the first call. EVIDENCE: stage_2_providers.py:58-66. CONFIDENCE: HIGH.]

[CLAIM: Empirical observation from item 114 follow-up — Claude correctly identified a phrase-level partial-restart pattern that Gemini repeatedly fails on. This suggests Claude may be BOTH more deterministic AND more accurate for the Telugu retake task. EVIDENCE: post_v2_backlog.md:1472-1476 ("Claude caught a phrase-level partial-restart pattern that Gemini fails on"). CONFIDENCE: MED — one anecdote, not a controlled trial.]

### Q6 — Malformed-response handling / retry semantics

[CLAIM: Pipeline does exactly ONE in-step corrective retry. EVIDENCE: stage_2_continuity.py:202-256, explicit comment at line 213-217 ("On second failure: raise RuntimeError wrapping both errors. Inngest's exponential backoff is the outer retry layer — we deliberately do NOT retry beyond 1 in-step."). CONFIDENCE: HIGH.]

[CLAIM: The corrective retry appends the validation error string AND a tight prescription of valid `category` strings to the prompt. EVIDENCE: stage_2_continuity.py:234-243 (the `correction_note` template). CONFIDENCE: HIGH.]

[CLAIM: Only `json.JSONDecodeError` and `pydantic.ValidationError` trigger the corrective retry. All other exceptions (auth, rate-limit, network, internal SDK errors) propagate immediately. EVIDENCE: stage_2_continuity.py:225-226, 248-255 (the `except (json.JSONDecodeError, ValidationError)` is narrow on purpose). CONFIDENCE: HIGH.]

[CLAIM: A SEMANTICALLY valid but EDITORIALLY anomalous output (e.g. 1 cut spanning the whole video) passes the corrective-retry check trivially — Pydantic validation succeeds, no retry fires, the bad decision goes to the renderer. EVIDENCE: stage_2_continuity.py:225 (only Pydantic ValidationError + JSONDecodeError trigger retry); the model is happily-conformant, just under-segmenting. CONFIDENCE: HIGH.]

**Implication: Job 53's "1 cut" was not caught by ANY guard. The corrective retry only catches schema failures, not editorial under-detection.**

[CLAIM: The orchestrator records `last_cost_usd` and `last_usage` after EITHER a successful first attempt or successful retry, so the cost ledger correctly accumulates both calls when a retry fires. EVIDENCE: stage_2_continuity.py:166-174 (the properties delegate to provider state); stage_2_providers.py:285-299, 426-443 (each successful call updates state). CONFIDENCE: HIGH.]

### 3B.1 Conclusions / recommendations

1. **The 1-cut-vs-28-cut variance is partially explained by temperature 0.2 + a prompt that permits both readings.** Switching the Gemini default to T=0.0 would tighten determinism somewhat but not eliminate it (the prompt itself is ambiguous on cut count).

2. **A pre-render semantic guardrail is missing.** Recommendation: after `assemble_stage_two_output`, check `len(full_video_cuts) == 1 and (end_sec - start_sec) > 0.9 * source_duration` — if true, log a WARN and either fail the step (forcing Inngest retry) or apply a deterministic "split by silence > 1.5s" post-processor to fan out into reasonable story-level cuts.

3. **Stage 2's prompt should add quantitative cut-count guidance.** Today the few-shot examples ALL show one cut; that anchors the model toward few cuts even when the input is multi-story. Recommended addition: "For a multi-minute monologue covering ≥ 3 distinct news stories, emit one FullVideoCut PER STORY (typically 5-30 cuts)."

4. **Claude provider should be considered the production default for editorial determinism.** T=0, thinking-disabled, prompt-cache-warmed — its structural variance is materially lower than Gemini at T=0.2. Cost penalty is ~2.5x on cold-cache, ~negligible after.

5. **A determinism harness should land in the test suite.** Same input, 5 invocations, assert `cut_count` and `len(skipped_segments)` standard-deviation stay within tolerance. Without this, variance regressions don't show up until a user complains.

---

## 3B.2 — FAILURE MODE INVENTORY (items 111-118)

For each of today's documented failures, structured root-cause + band-aid-vs-root-fix verdict.

### Item 111 — Bulletin video freeze (crossfade stitcher xfade chain collapse)

[CLAIM: Symptom: Job 46 bulletin video froze at 1:48 while audio played to 7:54; ffprobe video=104.53s/3136 frames vs audio=474.09s. EVIDENCE: post_v2_backlog.md:1148-1152. CONFIDENCE: HIGH.]

[CLAIM: Root cause: ffmpeg's `xfade` filter does NOT chain reliably for 20+ video transitions with cumulative offsets in the hundreds of seconds. Job 46 had 24 chained xfade nodes; video output collapsed to the longest single segment's duration. EVIDENCE: post_v2_backlog.md:1153-1162. CONFIDENCE: HIGH (verified by the parallel acrossfade chain working with the same math).]

[CLAIM: Fix: 3-pass stitcher in bulletin_crossfade_stitcher.py (concat-demuxer for video, acrossfade chain for audio, mux with -shortest). EVIDENCE: post_v2_backlog.md:1163-1176; the file exists today at pipeline_v2/pipeline_v2/bulletin_crossfade_stitcher.py. CONFIDENCE: HIGH.]

[CLAIM: Fix verification: live re-run on Job 46's 25 composed_story files produced video=473.96s (14209 frames) vs audio=474.09s, mux to 474.09s via -shortest. EVIDENCE: post_v2_backlog.md:1204-1208. CONFIDENCE: HIGH.]

[CLAIM: Class of bug: ffmpeg filter graph behavioural limit (not arithmetic error). This is a class of bug where the library has scaling characteristics our test fixtures didn't exercise. Industry pattern: scale-load test before shipping any filter-graph that chains N>10 nodes. EVIDENCE: nature of the root cause (post_v2_backlog.md:1158-1162). CONFIDENCE: HIGH.]

[CLAIM: Band-aid OR root-cause? ROOT CAUSE for the xfade collapse, but the symptom (lip-sync drift) had OTHER contributing causes that re-surfaced (items 112/115/116). So this fix was correct but INSUFFICIENT in isolation. EVIDENCE: items 112-117 are subsequent same-class fixes. CONFIDENCE: HIGH.]

[CLAIM: Generalisation: "filter-graph operations that scale linearly with segment count" need explicit scale-tests at the upper bound the production data hits. The 22 existing tests passed because they were N<=5. EVIDENCE: post_v2_backlog.md:1210-1228 (test summary mentions a "25-segment Job-46 regression check" added during the fix). CONFIDENCE: HIGH.]

### Item 112 — Frame-aligned per-clip cut step (lip-sync drift)

[CLAIM: Symptom: Jobs 47+48 showed cumulative intra-segment a/v drift of +299ms / +469ms even though bulletin file's global delta was 24-40ms. Drift accumulated 0-32ms per composed_story segment. EVIDENCE: post_v2_backlog.md:1236-1240. CONFIDENCE: HIGH.]

[CLAIM: Root cause: AAC frames are 21.33ms, 30fps video frames are 33.33ms; grids don't align, so when V1's `cut_video_clips` sliced a source-time range, audio rounded to its sample grid and video to its frame grid, leaving 0-32ms of audio-longer-than-video per slice. EVIDENCE: post_v2_backlog.md:1241-1247. CONFIDENCE: HIGH.]

[CLAIM: Fix: `cut_clips_frame_aligned()` snaps boundaries to 30fps grid before ffmpeg, uses `-r 30 -fps_mode cfr -async 1`, verifies post-cut with ffprobe and retries on a 100ms grid for failures. EVIDENCE: post_v2_backlog.md:1248-1266. CONFIDENCE: HIGH.]

[CLAIM: Did the fix solve the issue? Per-clip drift dropped from +7.3ms to -0.008ms on the test slice (913x improvement) — but item 115 was reported on a SUBSEQUENT job, indicating the drift had a SECOND source the fix didn't address. EVIDENCE: post_v2_backlog.md:1293-1297 (verification) vs 1480-1505 (item 115 user report "still lip sync issue is there"). CONFIDENCE: HIGH.]

[CLAIM: Band-aid OR root-cause? ROOT-CAUSE for the cut-step drift specifically. Band-aid for the user-perceived lip-sync issue because two other sources remained (item 115 compose-step AAC residue; item 116 -to vs -t boundary bug). EVIDENCE: chain of items 112 → 115 → 116. CONFIDENCE: HIGH.]

[CLAIM: Class of bug: "Two grid systems must align but don't" — same family as item 111 (xfade scaling) and item 115/116. The architectural diagnosis emerges retrospectively in item 117: too many points where audio and video can disagree. EVIDENCE: post_v2_backlog.md:1709-1722. CONFIDENCE: HIGH.]

### Item 113 — drift_measure_v2.py methodology

[CLAIM: NO backlog item explicitly labelled "113" exists in post_v2_backlog.md. The numbering jumps from item 112 (line 1234) directly to item 114 (line 1321). EVIDENCE: grep of "Item 113" returns no matches in the backlog file. CONFIDENCE: HIGH.]

[CLAIM: Item 113 (per the Track 3B task description) refers to "drift_measure_v2.py methodology" — but no in-line description survives in the backlog. The user's instruction conflicts with backlog state. EVIDENCE: search of the entire backlog for "113" returns nothing. CONFIDENCE: HIGH.]

[CLAIM: drift_measure_v2 IS referenced by items 112, 115, 115-followup, and 116 as the empirical measurement tool. EVIDENCE: post_v2_backlog.md:1236, 1484, 1571-1573, 1635-1639. CONFIDENCE: HIGH.]

[CLAIM: From context, drift_measure_v2.py methodology likely lives at pipeline_v2/scripts/drift_measure_v2.py (not read here) and emits per-segment + global a/v delta plus a cumulative warn. EVIDENCE: indirect citations at post_v2_backlog.md:1638-1640 ("item 112's own [cut summary] diagnostic fired its WARN at -695.8ms cumulative a-v delta across 28 clips"). CONFIDENCE: MED — the script's existence is implied but its exact behaviour not read.]

[CLAIM: Item 113's missing backlog entry is itself a process gap. The user remembers it as an item; the backlog doesn't reflect it. Class of bug: documentation drift — fixes shipped without backlog updates erode the source of truth. EVIDENCE: gap between user description and backlog content. CONFIDENCE: HIGH.]

### Item 114 — Claude Sonnet 4.6 provider + SDK API bugs

[CLAIM: Symptom: Need for empirical A/B against Gemini for stage 2 editorial decisions. Gemini Pro non-determinism at T=0.2 motivated provider abstraction. EVIDENCE: post_v2_backlog.md:1326-1329. CONFIDENCE: HIGH.]

[CLAIM: Root cause: Architectural — Stage 2 was hard-coded to Gemini; testing alternatives required SDK boundary refactor. EVIDENCE: post_v2_backlog.md:1330-1332. CONFIDENCE: HIGH.]

[CLAIM: Fix: `Stage2Provider` ABC + Gemini + Claude subclasses + `create_provider` factory. Retry policy stays in dispatcher; providers are stateless. EVIDENCE: stage_2_providers.py:149-205 (ABC), 209-314 (Gemini subclass), 320-479 (Claude subclass), 491-526 (factory). CONFIDENCE: HIGH.]

[CLAIM: SDK BUG 1 caught post-commit: `response.parsed_output` does not exist on the top-level `ParsedMessage` object — it lives on each `ParsedTextBlock`. The original code did `getattr(response, "parsed_output", None)` → always None → silent fallback to raw-JSON path → SDK's native validation guarantees were defeated. EVIDENCE: post_v2_backlog.md:1425-1438; the fix is now visible in stage_2_providers.py:452-463 (walk `response.content` for the first text block with non-None `parsed_output`). CONFIDENCE: HIGH.]

[CLAIM: SDK BUG 2 caught post-commit: `_strip_unsupported_constraints` was 30 lines of dead code — the Anthropic SDK already runs `transform_schema` over the Pydantic JSON schema, folding `minimum`/`maximum` into description text. The dead method would have stripped constraints the SDK actually wanted. EVIDENCE: post_v2_backlog.md:1440-1448. CONFIDENCE: HIGH.]

[CLAIM: Both bugs were silent failures the original test suite missed because the mocks happened to be wrong in the same way as the production code. EVIDENCE: post_v2_backlog.md:1422-1423. CONFIDENCE: HIGH.]

[CLAIM: Band-aid OR root-cause? ROOT-CAUSE fixes for both SDK bugs. The mock-aligned-with-buggy-prod-code pattern is a deeper testing-process bug. EVIDENCE: SDK fix logic in the file as of today's HEAD. CONFIDENCE: HIGH.]

[CLAIM: Class of bug: "SDK contract violation that the SDK silently tolerated" (BUG 1) + "premature defensive code that contradicts SDK contract" (BUG 2). Both belong to a meta-class: testing SDK boundary without actually exercising the SDK. EVIDENCE: post_v2_backlog.md:1421-1448 (bug descriptions) + 1459-1467 (real API smoke test that caught them). CONFIDENCE: HIGH.]

[CLAIM: Generalisation: every NEW SDK integration MUST land with one real-API smoke test in CI (or a documented manual probe) — not just mocked unit tests. EVIDENCE: post_v2_backlog.md:1459-1468 (one real API call costing $0.046 caught BOTH bugs). CONFIDENCE: HIGH.]

[CLAIM: Anecdotal evidence that Claude solves a Stage 2 problem Gemini couldn't: Claude correctly identified a phrase-level partial-restart pattern (the same class of pattern item 107 was DEFERRED on for Gemini). EVIDENCE: post_v2_backlog.md:1472-1476. CONFIDENCE: MED — one transcript, not a controlled A/B.]

### Item 115 — AAC priming leak / compose-step residue (TWO sub-items)

#### Item 115 (initial)

[CLAIM: Symptom: User reported "still lip sync issue is there" on Job 49 after item 112 shipped. drift_measure_v2 showed per-clip a/v drift was 0.0-0.3ms (item 112 worked) but two downstream sources of AAC frame-quantization residue remained. EVIDENCE: post_v2_backlog.md:1481-1486. CONFIDENCE: HIGH.]

[CLAIM: Root cause A (compose step): every composed_story_NN.mp4 came out with audio 7-18ms LONGER than video because `-c:a aac -ar 48000 -shortest` re-encodes audio, and AAC rounds the encoded length UP to the next 1024-sample frame (21.33ms quantum). Across 26 segments this accumulated to +256ms. EVIDENCE: post_v2_backlog.md:1488-1494. CONFIDENCE: HIGH.]

[CLAIM: Root cause B (stitcher Pass 3 mux): `-c copy -shortest` cannot split an AAC packet at video EOF, so `-shortest` leaks up to one AAC frame past the last video frame. Bulletin-level global delta: -92.7ms. EVIDENCE: post_v2_backlog.md:1496-1501. CONFIDENCE: HIGH.]

[CLAIM: Fix A: `_align_composed_audio_to_video(composed_path)` re-muxes with `-c:v copy` + audio re-encode `-af "atrim=end=V,apad=whole_dur=V,asetpts=PTS-STARTPTS"`. Called after every composed_story_NN.mp4 + takeover_NN.mp4. EVIDENCE: post_v2_backlog.md:1508-1519. CONFIDENCE: HIGH.]

[CLAIM: Fix B: Pass 3 mux changed from `-c copy -shortest` to `-c:v copy -c:a aac -b:a 192k -ar 48000 -shortest`. Audio is now re-encoded so -shortest truncates sample-accurately. EVIDENCE: post_v2_backlog.md:1522-1526. CONFIDENCE: HIGH.]

#### Item 115 follow-up

[CLAIM: Symptom: Job 50 (post-item-115) — "after 1:23 sec lipsync issue". drift_measure_v2 confirmed per-segment near-zero (-7.3ms cumul / 33 segments) but stitcher Pass 2 audio output ran ~+350ms longer than the formula predicted. EVIDENCE: post_v2_backlog.md:1571-1577. CONFIDENCE: HIGH.]

[CLAIM: Root cause: Each composed_story AAC stream carries encoder-priming samples (first packet pts=-1024 / -21.33ms / 1024 samples) AND tail padding. ffmpeg's MP4 demuxer respects the edit list for direct stream copy, but a filter graph referencing `[N:a]` pulls decoded PCM INCLUDING the priming + tail. Across 33 inputs at ~10ms / segment of leaked padding, the acrossfade chain emitted 350ms of phantom audio. EVIDENCE: post_v2_backlog.md:1580-1593. CONFIDENCE: HIGH.]

[CLAIM: Fix: Per-input atrim normaliser inserted BEFORE each acrossfade node — `[N:a]atrim=0:d_N,asetpts=PTS-STARTPTS[n###]`. Then chain acrossfade across the [n###] labels. atrim=0:d clamps decoded PCM to declared duration (drops priming + tail); asetpts resets timestamps. EVIDENCE: post_v2_backlog.md:1595-1607. CONFIDENCE: HIGH.]

[CLAIM: Empirical verification: Pass 2 PCM duration dropped from 470.016s (+350ms) to 469.666s (delta 0.0ms) on Job 50. EVIDENCE: post_v2_backlog.md:1609-1611. CONFIDENCE: HIGH.]

[CLAIM: Did items 115 + 115-followup solve the issue? Item 116 was reported on Job 51 — drift_measure said the bulletin file was OK but item 112's cut diagnostic fired at -695.8ms cumulative. So item 115 fixed compose+stitcher residue; item 116 found a THIRD upstream source. EVIDENCE: post_v2_backlog.md:1633-1640. CONFIDENCE: HIGH.]

[CLAIM: Band-aid OR root-cause? Both sub-items are root-cause fixes for their specific ffmpeg behaviours. Together they belong to the broader pattern "audio encoding does not preserve sample-accuracy under chain operations without explicit normalisation." EVIDENCE: pattern across items 111/112/115/116. CONFIDENCE: HIGH.]

[CLAIM: Class of bug: "Implicit assumption that ffmpeg operations preserve sample/frame accuracy — they don't." Sub-types: AAC priming/padding (115), grid mismatch (112), -to vs -t (116), xfade chain collapse (111). EVIDENCE: collection of items. CONFIDENCE: HIGH.]

### Item 116 — cut step `-to` → `-t` (Path-2 lip-sync root cause)

[CLAIM: Symptom: Job 51 — "Path 2 (item 112) verification: FAILED user perception — lip-sync drift still visible". drift_measure_v2 said the bulletin file was OK (-25.3ms global, "within one frame") but item 112's own diagnostic fired at -695.8ms cumulative across 28 clips. EVIDENCE: post_v2_backlog.md:1635-1640. CONFIDENCE: HIGH.]

[CLAIM: Root cause: ffmpeg's input-side `-to` (before `-i`) is video-INCLUSIVE of the end frame: when `end_snap * 30` lands on an integer (roughly half of 1/30s-snapped boundaries), ffmpeg pulls one extra video frame past the cutoff while audio cuts cleanly. Per-clip: video 33ms LONGER than audio. EVIDENCE: post_v2_backlog.md:1641-1654. CONFIDENCE: HIGH.]

[CLAIM: Item 115's apad=whole_dur=V then padded the 33ms gap with SILENCE to make composed_story durations match, hiding the bug at the file-duration level. drift_measure_v2 passed but each segment ended with ~33ms of "mouth-moving + silent audio" — the user-visible artifact. Last word's final phoneme truncated to silence. EVIDENCE: post_v2_backlog.md:1655-1662. CONFIDENCE: HIGH.]

[CLAIM: Fix: A/B/C bake-off on Job 51 mezzanine found variant C (`-ss X -i FILE -t (Y-X)`) gave av_delta=-0.02ms vs variant A's -33.0ms. Production fix: drop `-to`, add `-t` after `-i`, bump precision to .6f. EVIDENCE: post_v2_backlog.md:1664-1679. CONFIDENCE: HIGH.]

[CLAIM: Why drift_measure_v2 didn't catch this earlier — because the symptom was per-segment "silent-audio-under-moving-lips", not a duration delta. The bulletin file's V/A durations matched (thanks to item 115's apad pad), so a duration-delta check could not see it. EVIDENCE: post_v2_backlog.md:1655-1662 (item 115's apad masking item 116). CONFIDENCE: HIGH.]

[CLAIM: Empirical: 5-second cut [61.8, 66.8] pre-fix produced 151 video frames + 5.000s audio (1 frame OVER); post-fix 150 frames + 4.998s (EXACT). EVIDENCE: post_v2_backlog.md:1685-1688. CONFIDENCE: HIGH.]

[CLAIM: Band-aid OR root-cause? ROOT-CAUSE fix. Also REVEALED that item 115's apad was a band-aid masking item 116 — they shipped together but item 115 alone would have left the artifact unobserved on drift-measure but visible to the user. EVIDENCE: post_v2_backlog.md:1681-1683 (item 115 apad now downgraded to safety net + WARN > 5ms). CONFIDENCE: HIGH.]

[CLAIM: Class of bug: "ffmpeg flag semantics differ between input-side and output-side". Input-side -to / -ss have different semantics than output-side -t / -ss. Tutorial-level subtlety that a code reviewer wouldn't catch without explicit knowledge. EVIDENCE: post_v2_backlog.md:1641-1654 (the bug description). CONFIDENCE: HIGH.]

[CLAIM: Generalisation: any ffmpeg invocation that takes precise duration-bounded inputs MUST use output-side timing (`-i FILE -t duration` or `-i FILE -ss start -to end` AFTER -i) when sample-accuracy matters. Pre-input `-to` is "fast seek but slack frame boundary". EVIDENCE: same item description; cross-reference with ffmpeg documentation (not fetched). CONFIDENCE: HIGH for the recommendation, MED for the exhaustiveness.]

### Item 117 — Unified raw timeline extract architecture (single-pass multi-output)

[CLAIM: Symptom: After items 111-116 each fixed a specific lip-sync regression source, the user reported the same drift class kept appearing on different source videos. Diagnosis: cut-then-recombine architecture has too many A/V misalignment risk points; each new video tickles a different seam. Whack-a-mole engineering doesn't scale. EVIDENCE: post_v2_backlog.md:1711-1717. CONFIDENCE: HIGH.]

[CLAIM: Root cause: ARCHITECTURAL — the pipeline has multiple seams where A and V can drift independently (cut, compose, stitcher Pass 2, stitcher Pass 3). Each fix patched one seam; new videos exposed new seams. EVIDENCE: post_v2_backlog.md:1709-1717 + the meta-summary in item 118 at 1830-1859. CONFIDENCE: HIGH.]

[CLAIM: Fix design: decode the mezzanine ONCE per render, produce all raw timeline outputs (bulletin + per-short) in a SINGLE ffmpeg invocation via filter_complex trim+atrim+concat. Then apply overlays in separate passes with `-c:a copy` so audio is byte-identical past the slice step. EVIDENCE: post_v2_backlog.md:1718-1723. CONFIDENCE: HIGH.]

[CLAIM: Implementation: 5 phases / 5 commits. New modules: edl_builder.py (Phase 1), stage_4_raw_extract.py (Phase 2), stage_4_bulletin_overlay.py (Phase 3), stage_4_shorts_overlay.py (Phase 4), stage_4_render.py wire-through with `KAIZER_USE_V2_RAW_EXTRACT` feature flag (Phase 5). EVIDENCE: post_v2_backlog.md:1738-1786. CONFIDENCE: HIGH.]

[CLAIM: Diagnostic phase pre-flight validated the architecture on 4 cross-video tests before any production code was touched. EVIDENCE: post_v2_backlog.md:1725-1736. CONFIDENCE: HIGH.]

[CLAIM: Did the fix solve the issue? PARTIALLY — Job 53 timed out at the configured 1800s and fell back to legacy cut step. Architecture is sound (verified standalone) but production hit a transient system state. Bulletin drift_measure was -69.3ms global (legacy fallback signature). EVIDENCE: post_v2_backlog.md:1841-1849. CONFIDENCE: HIGH.]

[CLAIM: Item 117 declares which earlier items are SUPERSEDED when the flag is on: 111, 115, 116 are functionally redundant; item 112 still used as cache check + fallback. EVIDENCE: post_v2_backlog.md:1802-1816. CONFIDENCE: HIGH.]

[CLAIM: Band-aid OR root-cause? ROOT-CAUSE attempt at the ARCHITECTURE level (not the symptom level). The earlier fixes (111-116) were root-cause at the symptom level. This is a higher-order intervention. EVIDENCE: post_v2_backlog.md:1709-1717 (explicit "whack-a-mole engineering doesn't scale" framing). CONFIDENCE: HIGH.]

[CLAIM: Class of bug: "Architecture has too many independent failure points; symptom-level fixes are unbounded." This is the highest-order bug class — code debt rolls up into architecture debt. EVIDENCE: post_v2_backlog.md:1830-1859 (item 118's diagnosis). CONFIDENCE: HIGH.]

[CLAIM: Job 53's failure is NOT an item 117 invalidation — even if item 117 had succeeded, the 1-cut Stage 2 output would have routed through the same downstream silence-trim + micro-fragment chain, re-introducing drift inside the legacy compose+stitcher path. The architectural fix is in the wrong PLACE. EVIDENCE: post_v2_backlog.md:1852-1859. CONFIDENCE: HIGH.]

### Item 118 — Research checkpoint (current)

[CLAIM: This item declares a HALT on reactive fixes until research is done. Halt criteria explicit: research above OR user delegates it explicitly. EVIDENCE: post_v2_backlog.md:1908-1912. CONFIDENCE: HIGH.]

[CLAIM: Four open architectural questions framed: (1) silence-trim + micro-fragment placement (Stage 2 semantic vs Stage 4 mechanical); (2) eliminate downstream segmentation; (3) Stage 2 determinism; (4) what does a professional Telugu news bulletin actually look like? EVIDENCE: post_v2_backlog.md:1872-1890. CONFIDENCE: HIGH.]

[CLAIM: Item 118 is the META-classification: the whack-a-mole pattern is the actual bug. The user has correctly identified that V2 has been through 17+ items and lip-sync is still broken in production. EVIDENCE: post_v2_backlog.md:1831-1835. CONFIDENCE: HIGH.]

[CLAIM: Generalisation: When N consecutive fixes target the same symptom class on the same product surface, STOP coding and research the architecture. Item 118 demonstrates this pattern explicitly. EVIDENCE: post_v2_backlog.md:1908-1916. CONFIDENCE: HIGH.]

### 3B.2 Cross-cutting patterns

| Item | Class of bug | Was fix root-cause? | Survived next job? |
|---|---|---|---|
| 111 | ffmpeg filter scaling limit | Yes (for xfade) | Drift class survived (112) |
| 112 | AAC/video grid mismatch | Yes (for cut step) | Drift class survived (115) |
| 114 | SDK contract + dead code | Yes | N/A (provider feature) |
| 114-fu | SDK contract violation | Yes | N/A (caught pre-prod) |
| 115 | AAC encoding residue (compose) | Yes (for compose) | Drift class survived (115fu) |
| 115fu | AAC priming leak (stitcher) | Yes (for stitcher) | Drift class survived (116) |
| 116 | ffmpeg -to inclusivity | Yes (for cut step) | Drift class survived (117) |
| 117 | Architecture | Attempted root-cause | Job 53 timeout + 1-cut variance |
| 118 | Process — too many band-aids | Meta-fix (halt) | TBD |

[CLAIM: Across items 111-117, EVERY individual fix is root-cause at its level, but the SYMPTOM (lip-sync drift) survives because the architecture has more drift sources than fixes. EVIDENCE: pattern from the table above + item 118 framing. CONFIDENCE: HIGH.]

[CLAIM: The team's debugging methodology is good (every item has empirical verification + tests). The DESIGN-LEVEL decision to layer the V2 pipeline on top of V1's compose+stitch chain is the deeper issue. EVIDENCE: post_v2_backlog.md:1856-1859 ("The architectural fix is in the wrong PLACE"); item 117's whole-pipeline overhaul scope. CONFIDENCE: HIGH.]

[CLAIM: Stage 2 (LLM editorial) and Stage 4 (render mechanics) currently have an IMPLICIT contract that Stage 2's cut list is what the renderer uses. In practice Stage 4's compose chain re-segments single Stage-2 cuts via silence-trim + micro-fragment-split. This violates the contract silently. EVIDENCE: post_v2_backlog.md:1853-1859 + 1872-1882. CONFIDENCE: HIGH.]

---

## 3B.3 — BACKLOG OPEN-ITEMS REVIEW (items <= 110)

### Item 57 — Groq whisper-large-v3 returns ZERO word timestamps for Telugu audio

[CLAIM: Status: OPEN. Recommendation: KEEP, no action. EVIDENCE: post_v2_backlog.md:139-160; item 59 documents Deepgram as default for Telugu, mitigating production impact. CONFIDENCE: HIGH.]

[CLAIM: Still relevant: YES, but mitigated by item 59. Re-test quarterly per the documented cadence. Cost to fix: bug is in Groq's product — we can't fix. Impact if not fixed: LOW (Deepgram covers). EVIDENCE: post_v2_backlog.md:151-160. CONFIDENCE: HIGH.]

### Item 58 — Groq `with_raw_response.create()` 500s on 10MB+ audio

[CLAIM: Status: OPEN. Stacked behind item 57. Cost: ~2h to switch to plain `create()` + synth request_id, IF item 57 ever resolves. Impact: LOW (Deepgram path covers Telugu). EVIDENCE: post_v2_backlog.md:162-179. CONFIDENCE: HIGH.]

### Item 59 — Deepgram as default Beta STT for Indian-language

[CLAIM: Status: DOCUMENTED + LANDED as decision. KEEP for reference but not actionable. EVIDENCE: post_v2_backlog.md:181-194. CONFIDENCE: HIGH.]

### Item 60 — V1 image generation policy violation (PERSON entities)

[CLAIM: Status: OPEN, V1-only. V2 already policy-compliant. Cost: ~4h to patch V1 image_plan flow. Impact: MED (legal/policy risk on V1 path until cutover). EVIDENCE: post_v2_backlog.md:196-231. CONFIDENCE: HIGH.]

[CLAIM: Recommendation: KEEP — patch when V1 path is touched next. If V1 is being deprecated post-cutover, this becomes moot. EVIDENCE: item 85's cutover gate definition at post_v2_backlog.md:783-795. CONFIDENCE: HIGH.]

### Item 61 — V1 image-search chain ordering for Indian-language

[CLAIM: Status: OPEN, data-driven post-launch decision. Cost: investigation-bound, not implementation-bound. Impact: LOW unless Pexels rate-limits. EVIDENCE: post_v2_backlog.md:233-250. CONFIDENCE: HIGH.]

### Item 62 — Gemini cost tracker not populating envelope (5x underestimate)

[CLAIM: Status: OPEN. Recommended FIX BEFORE production launch, currently NOT addressed. Cost: ~2-4h (thread cost_callback or return-tuple through stage_2/2.5/3). Impact: HIGH (budget guards underestimate by 4-5x — could enable runaway Gemini cost). EVIDENCE: post_v2_backlog.md:252-275. CONFIDENCE: HIGH.]

[CLAIM: Item 114 partially closed this for Stage 2 (the Claude provider records last_cost_usd correctly, and orchestrator was updated to read it). Stages 2.5, 3a, 3b, 3c likely still under-report. EVIDENCE: stage_2_providers.py:285-299 + 426-443; post_v2_backlog.md:1376-1381 ("Cost ledger now records the provider's actual last_cost_usd instead of the placeholder $0"). CONFIDENCE: MED — Stage 2 confirmed fixed; Stage 2.5/3 status not verified in this Track.]

[CLAIM: Recommendation: PROMOTE to active milestone — this is a financial-risk item that hasn't been resolved. EVIDENCE: same item description. CONFIDENCE: HIGH.]

### Item 63 — Bulletin truncation via V1 idempotency cache (FIXED)

[CLAIM: Status: FIXED in 12.2a re-run #5. CLOSE. EVIDENCE: post_v2_backlog.md:277-297 (FIX described, regression tests landed). CONFIDENCE: HIGH.]

### Item 64 — clip_image_map mis-keyed for shorts (FIXED)

[CLAIM: Status: FIXED in 12.2a re-run #5. CLOSE. EVIDENCE: post_v2_backlog.md:299-327. CONFIDENCE: HIGH.]

### Item 65 — Lenient parser invariant cascade (lesson learned)

[CLAIM: Status: LESSON LEARNED, not actionable. KEEP as design guidance. EVIDENCE: post_v2_backlog.md:329-363. CONFIDENCE: HIGH.]

### Item 66 — Bulletin editorial trimming (no V1 concept)

[CLAIM: Status: OPEN, post-Beta polish. Cost: ~1-2 days (new Stage 3.5 selector). Impact: MED (user-facing quality — 60-min source → 60-min bulletin today; target is 20-30 min compressed). EVIDENCE: post_v2_backlog.md:365-388. CONFIDENCE: HIGH.]

[CLAIM: This item is RELATED to item 118 question #1 ("silence-trim + micro-fragment Stage 2 vs Stage 4"). Both touch on "what does Stage 2 own vs what does the renderer own." Recommendation: MERGE conceptually with item 118 architectural work. EVIDENCE: post_v2_backlog.md:1872-1879. CONFIDENCE: MED.]

### Item 67 — CSE/DDG zero results for Telugu queries

[CLAIM: Status: OPEN, recommended BEFORE production launch. Cost: ~1h investigation + config; if real fix needed, ~4h to add Bing source. Impact: MED-HIGH (single point of failure on Pexels). EVIDENCE: post_v2_backlog.md:390-427. CONFIDENCE: HIGH.]

[CLAIM: Recommendation: PROMOTE — this is launch-blocking per its own description. EVIDENCE: post_v2_backlog.md:425-427 ("Fix when: BEFORE production launch"). CONFIDENCE: HIGH.]

### Items 70-74 — Inngest SDK behavioural notes (LANDED)

[CLAIM: Item 70 (send vs send_sync) — documentation only, KEEP. EVIDENCE: post_v2_backlog.md:429-442. CONFIDENCE: HIGH.]

[CLAIM: Item 71 (no dedup signal) — runbook entry, KEEP until Inngest releases explicit log. EVIDENCE: post_v2_backlog.md:444-468. CONFIDENCE: HIGH.]

[CLAIM: Item 72 (orchestrator signature) — FIXED + regression test. CLOSE. EVIDENCE: post_v2_backlog.md:470-498. CONFIDENCE: HIGH.]

[CLAIM: Item 73 (setup-verification explanation correction) — retroactive note only. CLOSE. EVIDENCE: post_v2_backlog.md:499-515. CONFIDENCE: HIGH.]

[CLAIM: Item 74 (`except BaseException` bug) — FIXED + regression test. CLOSE. EVIDENCE: post_v2_backlog.md:516-560. CONFIDENCE: HIGH.]

### Item 75 — UI surface for "cancelling, finishing current step"

[CLAIM: Status: OPEN, UI polish. Cost: ~15 LOC frontend. Impact: LOW (UX, not functional). EVIDENCE: post_v2_backlog.md:561-580. CONFIDENCE: HIGH.]

### Items 76-77 — Stage 4 cancel-check + idempotency dedup (FIXED)

[CLAIM: Both FIXED + landed with regression tests. CLOSE. EVIDENCE: post_v2_backlog.md:581-662. CONFIDENCE: HIGH.]

### Item 78 — Pre-existing V1 test failures (out of scope)

[CLAIM: Status: OPEN, ~1-2h triage when convenient. KEEP. EVIDENCE: post_v2_backlog.md:663-688. CONFIDENCE: HIGH.]

### Item 79 — Clip.job_id ON DELETE CASCADE at ORM only

[CLAIM: Status: OPEN, post-Beta DB cleanup. Cost: 1-2h migration. Impact: LOW (only matters if raw SQL deletes happen — not a current ops pattern). EVIDENCE: post_v2_backlog.md:689-704. CONFIDENCE: HIGH.]

### Items 80-86 — V2 Beta launch infrastructure (LANDED)

[CLAIM: Items 80 (rename/feedback), 81 (admin dashboard), 82 (preflight + runbook) — all LANDED. CLOSE. EVIDENCE: post_v2_backlog.md:706-738. CONFIDENCE: HIGH.]

[CLAIM: Item 83 (GCP cost cap) — DOCUMENTED, operator-action required. KEEP as ops checklist item. EVIDENCE: post_v2_backlog.md:753-768. CONFIDENCE: HIGH.]

[CLAIM: Item 84 (phased rollout cohort) — operational decision document. KEEP. EVIDENCE: post_v2_backlog.md:769-781. CONFIDENCE: HIGH.]

[CLAIM: Item 85 (cutover gate) — operational. KEEP — gate criteria are unmet today given item 118's halt state. EVIDENCE: post_v2_backlog.md:783-795. CONFIDENCE: HIGH.]

[CLAIM: Item 86 (test fixture pollution) — latent risk, ~30 min refactor. KEEP. EVIDENCE: post_v2_backlog.md:797-813. CONFIDENCE: HIGH.]

### Item 87 — Post-testing secret rotation

[CLAIM: Status: OPEN. Cost: ~2h to rotate keys + scrub files. Impact: MED (security hygiene, blast radius gated by Cloudflare Tunnel today). EVIDENCE: post_v2_backlog.md:815-837. CONFIDENCE: HIGH.]

[CLAIM: Recommendation: PROMOTE to active milestone — keys remained in non-rotated state per backlog; production stabilisation pending. EVIDENCE: post_v2_backlog.md:825-826 ("operator chose to rotate post-stabilization rather than mid-bring-up"). CONFIDENCE: HIGH.]

### Item 88 — Stage 4 blocks uvicorn worker (P1 UX)

[CLAIM: Status: CONFIG MITIGATION LANDED (`--workers 4`) but REAL FIX recommended is `await asyncio.to_thread(subprocess.run, ...)`. Item 89 mentions this fix landed when --workers was reverted to 1 ("Stage 4's asyncio.to_thread wrap (item 88)"). EVIDENCE: post_v2_backlog.md:862-867, 921-923. CONFIDENCE: HIGH.]

[CLAIM: Recommendation: VERIFY closure by checking stage_4_render.py for the asyncio.to_thread wrap. If present, CLOSE. EVIDENCE: cross-reference item 89 mention. CONFIDENCE: MED — not verified in code in this Track.]

### Item 89 — Windows asyncio loop + StepError + on_failure (FIXED)

[CLAIM: Status: FIXED with regression tests. CLOSE. EVIDENCE: post_v2_backlog.md:905-975. CONFIDENCE: HIGH.]

### Item 107 — Partial-restart detection patterns (DEFERRED)

[CLAIM: Status: DEFERRED — iter-2 prompt attempt REGRESSED iter-1 baseline. Captured implementation plan in subsequent item. EVIDENCE: post_v2_backlog.md:976-1035. CONFIDENCE: HIGH.]

[CLAIM: This item's "lesson learned" — Gemini Pro at T=0.2 is meaningfully non-deterministic — directly motivates item 114 (Claude provider) and feeds item 118's determinism question. EVIDENCE: post_v2_backlog.md:1015-1019. CONFIDENCE: HIGH.]

[CLAIM: Item 107 may be SUPERSEDED by item 114's Claude provider — backlog explicitly notes "Claude provider may solve item 107 out of the box — worth measuring in the A/B bake-off." EVIDENCE: post_v2_backlog.md:1472-1476. CONFIDENCE: HIGH.]

[CLAIM: Recommendation: KEEP open, MERGE outcome with item 114 A/B results. The deterministic post-processor approach is documented in "Item 107 implementation plan" at post_v2_backlog.md:1100-1144 — defer until Claude bake-off is decided. EVIDENCE: same. CONFIDENCE: HIGH.]

### Item 108 — Real smart_cut crossfade (LANDED then SUPERSEDED)

[CLAIM: Status: LANDED 2026-05-20 commit 5fb77c0. SUPERSEDED architecturally by item 111's 3-pass rewrite (the xfade chain it shipped was the bug item 111 fixed). EVIDENCE: post_v2_backlog.md:1036-1054 + 1153-1162. CONFIDENCE: HIGH.]

[CLAIM: Recommendation: CLOSE — superseded by item 111. EVIDENCE: same. CONFIDENCE: HIGH.]

### Item 109 — End-frame trim after last spoken word (LANDED)

[CLAIM: Status: LANDED in commit bc76d27 (`v2-iter2-ship` tag). CLOSE. EVIDENCE: post_v2_backlog.md:1056-1071. CONFIDENCE: HIGH.]

### Item 110 — V2 iteration-2 verification (LANDED)

[CLAIM: Status: LANDED; 8.1/10 grade. CLOSE. But note: ChatGPT-as-judge methodology + the user's claim "ChatGPT 8.1/10 measurement" today suggests this grade may need re-verification post items 111-117. EVIDENCE: post_v2_backlog.md:1073-1098. CONFIDENCE: HIGH.]

### 3B.3 Summary recommendations

**PROMOTE to active milestone (production-launch blockers, not yet resolved):**
- Item 62 — Cost tracker (financial risk)
- Item 67 — CSE/DDG image search fragility (launch-recommended)
- Item 87 — Secret rotation (security hygiene)
- Item 118 — Architectural research checkpoint (the meta-halt)

**KEEP open, no immediate action:**
- Items 57, 58, 60, 61, 66, 75, 78, 79, 84, 85, 86, 107

**CLOSE (resolved/superseded/landed):**
- Items 59, 63, 64, 65 (lesson), 70, 71 (runbook), 72, 73, 74, 76, 77, 80, 81, 82, 83 (ops docs), 89, 108 (superseded), 109, 110, 111, 112, 114, 115, 115-fu, 116, 117 (provisionally)

**MERGE conceptually:**
- Item 66 + Item 118 question #1 (editorial selector + Stage 2 vs Stage 4 ownership)
- Item 107 + Item 114 bake-off (partial-restart via Claude)

---

## 3B.4 — DEPENDENCY VERSION + DEPRECATION AUDIT

Reading pyproject.toml (e:\kaizer new data training\kaizer\KaizerBackend\pipeline_v2\pyproject.toml):

### Pinned versions in pyproject.toml

| Dependency | Pin | Latest (as of 2026-05) | Verdict |
|---|---|---|---|
| `pydantic` | `>=2.10,<3` | 2.x (compatible) | OK |
| `google-genai` | `>=1.75,<3` | **2.5.0** | UPGRADE GATED — see below |
| `groq` | `==1.2.0` | newer available | KEEP — Telugu blocked anyway (item 57) |
| `deepgram-sdk` | `==7.1.1` | **7.2.0** | Minor upgrade available |
| `assemblyai` | `==0.64.2` | unknown | LOW priority — not on default path |
| `inngest` | `==0.5.18` | 0.5.18 (latest) | Current |
| `structlog` | `>=24.0` | 24.x current | OK |
| `boto3` | `>=1.40` | current | OK |
| `psutil` | `>=7` | current | OK |
| `anthropic` (in requirements.txt per item 114) | `>=0.103.0` | **0.103.1** (May 19, 2026) | Patch upgrade available |

### google-genai (was 2.3.0; latest is 2.5.0)

[CLAIM: pyproject pinned `google-genai>=1.75,<3` with comment "Phase 0 target: 2.3.0 — venv has 1.75.0; upgrade gated on v1 compat check". EVIDENCE: pyproject.toml:29. CONFIDENCE: HIGH.]

[CLAIM: Latest google-genai is 2.5.0 per PyPI metadata. EVIDENCE: WebFetch on pypi.org/pypi/google-genai/json — "The latest version shown in the release history is 2.5.0". CONFIDENCE: HIGH.]

[CLAIM: Breaking changes between 1.75 and 2.5.0 (per WebFetch of the GitHub changelog) impact mainly the Interactions API (v2.0.0 overhaul). GenerateContent API — which is what Stage 2 uses (stage_2_providers.py:279-283 `client.aio.models.generate_content`) — remained unaffected. EVIDENCE: WebFetch on github.com/googleapis/python-genai/blob/main/CHANGELOG.md. CONFIDENCE: MED — single source for breaking-change summary, not deeply verified.]

[CLAIM: Other relevant breaking changes (v1.68.0): TextContent annotations refactored; ContentDelta unions renamed. These would only affect code that consumes streaming responses; Stage 2 uses non-streaming generate_content so unlikely to be affected. EVIDENCE: same WebFetch. CONFIDENCE: MED.]

[CLAIM: Recommendation: schedule google-genai upgrade to 2.5.0 once the V1 pipeline's compat check passes. Stage 2's code (response_schema + temperature + thinking_config + max_output_tokens) is on stable API surface. EVIDENCE: stage_2_providers.py:270-283 (all standard GenerateContentConfig fields). CONFIDENCE: MED.]

### anthropic SDK (current 0.103.0; latest 0.103.1)

[CLAIM: requirements.txt has `anthropic>=0.103.0` per item 114 description (post_v2_backlog.md:1391). The version Stage 2 was written against is 0.103.x (mentioned at post_v2_backlog.md:1426 "anthropic-sdk-python 0.103.x"). CONFIDENCE: HIGH.]

[CLAIM: Latest is 0.103.1, released 2026-05-19. EVIDENCE: WebFetch on PyPI / GitHub. CONFIDENCE: HIGH.]

[CLAIM: 0.103.1 is a single bug fix in SessionToolRunner (skip tool calls it does not own). No impact on Stage 2's `messages.parse(output_format=...)` path. EVIDENCE: WebFetch on github.com/anthropics/anthropic-sdk-python/releases. CONFIDENCE: HIGH.]

[CLAIM: No deprecations in `messages.parse` between 0.103.0 and 0.103.1. The Stage 2 BUG 1 (item 114 follow-up) noted that `ParsedTextBlock.parsed_output` is the canonical access path — still valid in 0.103.1. EVIDENCE: same WebFetch + post_v2_backlog.md:1425-1438. CONFIDENCE: HIGH.]

[CLAIM: Recommendation: bump to anthropic 0.103.1. Trivial — patch release only. EVIDENCE: same. CONFIDENCE: HIGH.]

### deepgram-sdk (pinned 7.1.1; latest 7.2.0)

[CLAIM: Pinned `deepgram-sdk==7.1.1`. Latest is 7.2.0. EVIDENCE: pyproject.toml:31 + WebFetch on PyPI. CONFIDENCE: HIGH.]

[CLAIM: Specific deprecations between 7.1.1 and 7.2.0 not documented in the WebFetch result (PyPI metadata doesn't carry release notes). Would need to fetch GitHub releases page for the SDK directly to enumerate. EVIDENCE: WebFetch result said "For specific deprecation details between these minor versions, you would need to consult the project's changelog or release notes on the GitHub repository". CONFIDENCE: MED.]

[CLAIM: Stage 1 STT (Deepgram) is the default for Telugu (item 59) — risk of regression on upgrade is non-trivial. Recommendation: stay on 7.1.1 until Stage 1 work is opened. EVIDENCE: post_v2_backlog.md:181-194 (item 59); pyproject.toml:31 ("Step 4.3 — Deepgram Nova-3 STT provider"). CONFIDENCE: HIGH.]

### inngest (pinned 0.5.18; latest 0.5.18)

[CLAIM: Pinned 0.5.18 — current latest. No upgrade needed. EVIDENCE: pyproject.toml:33 + WebFetch on PyPI. CONFIDENCE: HIGH.]

[CLAIM: Per Items 70, 72, 74, 89, the V2 orchestrator is deeply coupled to 0.5.18's specific behaviour (single-arg ctx, BaseException flow-control, on_failure hook). Upgrade would require revisiting all four. EVIDENCE: post_v2_backlog.md:429-498, 516-560, 905-975. CONFIDENCE: HIGH.]

[CLAIM: Recommendation: PIN STRICTLY (== 0.5.18, not >=). If Inngest releases 0.6.x, treat as a planned migration not a patch upgrade. EVIDENCE: behavioural coupling above. CONFIDENCE: HIGH.]

### Summary of upgrade recommendations

1. `anthropic`: bump to 0.103.1 — safe patch upgrade.
2. `google-genai`: KEEP on current pin; schedule a 2.5.0 audit when V1 compat check is done. Generate-content API stable.
3. `deepgram-sdk`: KEEP at 7.1.1 until a Stage 1 work-cycle opens.
4. `inngest`: KEEP at 0.5.18 strict pin.
5. `groq`: KEEP — Telugu blocked at provider level; upgrade is moot until item 57 resolves.

---

## 3B.5 — EXTRA: BACKLOG ORGANISATION vs INDUSTRY TRACKER FORMATS

(Time allowed — completing the optional task.)

### Observations on current backlog format

[CLAIM: Backlog uses ordered numeric items (57-118) with no taxonomy, no labels, no status field beyond title-embedded markers like "(FIXED)", "(COMPLETED)", "(DEFERRED)". EVIDENCE: post_v2_backlog.md throughout. CONFIDENCE: HIGH.]

[CLAIM: Some items appear at logical numeric jumps (57 → 58 → 59 → 60 then 70 — items 68-69 don't exist). EVIDENCE: grep "Item 6[0-9]" + "Item 7[0-9]" gaps. CONFIDENCE: HIGH.]

[CLAIM: Item 113 referenced by Track 3B brief is missing from the backlog file — a documentation drift symptom. EVIDENCE: grep "Item 113" returns nothing. CONFIDENCE: HIGH.]

### Comparison to industry tracker conventions

- **GitHub Issues** — adds labels (`bug`, `architecture`, `wontfix`), assignees, milestones, linked PRs.
- **Linear** — adds priority (P0-P3), workflow state (Triage / Backlog / In Progress / In Review / Done / Cancelled), cycle membership.
- **JIRA** — story points, sprint, fixVersion, epic linkage.
- **Notion-style table** — columns for owner, due date, related-PR, status.

[CLAIM: Backlog's current format optimises for solo-founder context-recovery ("future-me reading this"), not for team coordination. EVIDENCE: post_v2_backlog.md:2-4 explicit statement of audience. CONFIDENCE: HIGH.]

[CLAIM: The format works at current scale (~30 items, single maintainer) but would slow down at >50 items or with >1 contributor. EVIDENCE: heuristic from industry patterns; no internal evidence. CONFIDENCE: MED.]

### Recommendations IF the backlog grows

1. Add a YAML-frontmatter status field to each item (state + last-updated + priority). Keep the prose body unchanged.
2. Add a top-of-file index that auto-summarises status counts (open / closed / deferred).
3. Move closed items to an archive section to keep the active surface scannable.
4. Add explicit "supersedes" / "superseded-by" cross-refs (item 108 lacks this even though item 111 effectively supersedes it).

[CLAIM: Item 117 already does the supersedes pattern explicitly ("ITEMS SUPERSEDED BY ITEM 117"). Adopting this consistently would be a low-effort hygiene win. EVIDENCE: post_v2_backlog.md:1802-1816. CONFIDENCE: HIGH.]

### Code debt vs architecture debt classification

| Type | Items |
|---|---|
| Architecture debt | 66 (editorial trimming), 88 (worker blocking — partial), 118 (whole-pipeline halt), implicit Stage-2-vs-Stage-4 contract |
| Code debt | 57/58 (Groq), 60/61 (V1 image), 62 (cost tracker), 67 (image search), 75 (UI), 78 (V1 tests), 79 (CASCADE), 86 (fixture), 87 (secrets) |
| Lesson / runbook | 65, 70, 71, 73, 83, 84, 85 |
| Resolved / superseded | 59, 63, 64, 72, 74, 76, 77, 80, 81, 82, 89, 108, 109, 110, 111, 112, 114, 115, 115-fu, 116, 117 |

[CLAIM: Architecture-debt items (66, 88, 118) outweigh code-debt items in long-term cost; item 118 alone subsumes the whack-a-mole drift loop documented across 111-117. EVIDENCE: post_v2_backlog.md:1830-1916 (item 118's framing). CONFIDENCE: HIGH.]

[CLAIM: Code debt is healthily catalogued but not currently scheduled. Items 62 and 67 are flagged as "BEFORE production launch" in their own descriptions but no commit / branch / PR addresses them. EVIDENCE: post_v2_backlog.md:274-275 (item 62) + 425-427 (item 67). CONFIDENCE: HIGH.]

---

## Sources cited

- `e:\kaizer new data training\kaizer\KaizerBackend\pipeline_v2\pipeline_v2\stages\stage_2_continuity.py` (lines cited inline)
- `e:\kaizer new data training\kaizer\KaizerBackend\pipeline_v2\pipeline_v2\stages\stage_2_providers.py` (lines cited inline)
- `e:\kaizer new data training\kaizer\KaizerBackend\pipeline_v2\pipeline_v2\stages\stage_2_prompt.md` (lines cited inline)
- `e:\kaizer new data training\kaizer\KaizerBackend\pipeline_v2\pipeline_v2\models.py` (Stage2Output schema fields)
- `e:\kaizer new data training\kaizer\KaizerBackend\pipeline_v2\post_v2_backlog.md` (items cited by line)
- `e:\kaizer new data training\kaizer\KaizerBackend\pipeline_v2\pyproject.toml`
- WebFetch: pypi.org/pypi/google-genai/json
- WebFetch: pypi.org/pypi/anthropic/json
- WebFetch: pypi.org/pypi/deepgram-sdk/json
- WebFetch: pypi.org/pypi/inngest/json
- WebFetch: github.com/googleapis/python-genai/blob/main/CHANGELOG.md
- WebFetch: github.com/anthropics/anthropic-sdk-python/releases
- WebFetch: github.com/inngest/inngest-py/releases

END OF TRACK 3B REPORT.
