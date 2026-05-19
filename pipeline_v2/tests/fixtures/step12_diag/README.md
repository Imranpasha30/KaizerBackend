# Step 12 — V2 Pipeline E2E Diagnostic Artifacts

Each subdirectory here is one E2E test run's complete pipeline output,
captured by `test_e2e_v2_pipeline.py` (the most-recent run is the
canonical "Step 12 PASS" certification artifact).

## Directory layout (per run)

```
step12_diag/
└── <YYYYMMDD_HHMMSS>__<mode>/
    ├── manifest.json              -- summary: pass/fail per check, stage costs, timing
    ├── stage_0_output.json        -- Stage0Output Pydantic dump
    ├── stage_1_transcript.json    -- Stage1Output WordLevelTranscript dump
    ├── stage_2_decisions.json     -- StageTwoOutput Pydantic dump
    ├── stage_2_5_entities.json    -- Stage2_5Output Pydantic dump
    ├── stage_3_output.json        -- Stage3Output (shorts + metadata + image_plan)
    ├── stage_4_result.json        -- RenderResult (paths + bulletin meta)
    ├── editor_meta_shorts.json    -- V1-shape shorts editor_meta (Step 8 adapter output)
    ├── editor_meta_bulletin.json  -- V1-shape bulletin editor_meta
    ├── cost_ledger.json           -- per-stage cost breakdown + total_usd
    └── bulletin_meta.json         -- bulletin.mp4 size + ffprobe duration + framerate
                                     (the actual .mp4 stays in `output/`; too large to
                                     commit and noisy in diffs)
```

`<mode>` is one of:

* `direct` — Step 12.2a direct-orchestrator-drive run (Inngest stubbed)
* `inngest` — Step 12.2b real Inngest Dev Server run (full event delivery path)
* `cancel` — Step 12.3 cancellation test (subset of artifacts; ends mid-pipeline)
* `idempotency` — Step 12.4 idempotency fixture comparison

## Reading a manifest

```jsonc
{
  "timestamp":  "20260519_142533",
  "mode":       "direct",
  "test_video": "/abs/path/test.mp4",
  "stt_provider": "whisper-groq",
  "cost_usd":   {
    "stage_0_ingest":     0.0,
    "stage_1_transcribe": 0.0,        // Free tier
    "stage_2_continuity": 0.4830,     // Gemini Pro
    "stage_2_5_entities": 0.0250,
    "stage_3_fanout":     0.1100,
    "stage_4_render":     0.0,
    "finalize":           0.0,
    "TOTAL":              0.6180
  },
  "checks": {
    "stage_0_mezzanine":      "PASS",
    "stage_1_word_array":     "PASS",
    "stage_2_decisions":      "PASS",
    "stage_2_5_entities":     "PASS",
    "stage_3_fanout":         "PASS",
    "stage_4_render":         "PASS",
    "finalize_db":            "PASS"
  },
  "wall_seconds": 287.5
}
```

## Cost-ledger comparison across runs

The `cost_usd.TOTAL` field lets us track API spend drift over time. A
single-run total exceeding $2 should trigger an investigation
(Gemini token explosion, retry storm, etc.) per the D-12.9 budget
warning threshold.

## Inngest Dev Server install (for 12.2b)

The Inngest CLI is a Go binary that hosts a local event router +
dashboard. Required only for the `inngest` mode test; the `direct`
mode test never invokes it.

### Install

**Windows / scoop:**
```
scoop bucket add extras
scoop install inngest
```

**macOS / brew:**
```
brew install inngest/tap/inngest
```

**Manual (any platform):**
Download the latest release from
<https://github.com/inngest/inngest/releases> and put `inngest` on PATH.

### Run

```
inngest dev -u http://localhost:8288   # default dashboard URL
```

The 12.2b test detects whether the CLI is on PATH at the start of
collection and skips with a clear `pytest.skip()` if missing. No
network calls happen during collection.
