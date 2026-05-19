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
