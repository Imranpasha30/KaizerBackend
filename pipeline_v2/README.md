# Pipeline V2

Kaizer News V2 multi-stage pipeline. Lives side-by-side with the v1
single-call pipeline (`KaizerBackend/pipeline_core/pipeline.py`); the
existing 4 platform paths (Instagram Reel / YouTube Short / YouTube
Full / Full Video + Shorts) continue to use v1. Only the new
**"Full Video + Shorts (V2)"** platform option routes here.

## Architecture

```
Stage 0  Ingest         ffprobe + NVENC transcode + parallel mp3
Stage 1  Transcribe     Deepgram Nova-3 (multilingual, diarized)
Stage 2  Continuity     Gemini 2.5 Pro    — full_video_cuts + skipped_segments + clean_transcript
Stage 2.5 Entities      Gemini 2.5 Flash  — canonical_entities (cap 6)
Stage 3  Fan-out (asyncio.gather, one Inngest step)
   3a   Shorts          Gemini 2.5 Flash  — shorts_cuts
   3b   Metadata        Gemini 2.5 Flash  — headlines, marquee, summaries (native)
   3c   Image plan      Gemini 2.5 Flash  — image_plan with id reuse + boundary check
Stage 4  Render         FFmpeg/Pillow port of v1 rendering
```

Orchestrated by Inngest. Output passes through
`editor_meta_adapter.py` so the existing editor tab opens v2 clips
identically to v1 ones (byte-for-byte same `editor_meta.json` shape).

## Status

| Step | What                                | Status |
|------|--------------------------------------|--------|
| 1    | Scaffold + models                    | done   |
| 2    | Storage (R2 client)                  | pending |
| 3    | Stage 0 (ingest)                     | pending |
| 4    | Stage 1 (Deepgram)                   | pending |
| 5    | Stage 2 (continuity)                 | pending |
| 6    | Stage 2.5 (entities)                 | pending |
| 7    | Stage 3a/3b/3c (parallel)            | pending |
| 8    | editor_meta_adapter                  | pending |
| 9    | Stage 4 (render)                     | pending |
| 10   | Orchestrator (Inngest)               | pending |
| 11   | UI 5th card + runner.py routing      | pending |
| 12   | End-to-end test                      | pending |
| 13   | Soft launch (Beta badge)             | pending |
| 14   | Cutover (separate decision, weeks later) | pending |

## Run tests

```
cd kaizer/KaizerBackend/pipeline_v2
"e:/kaizer new data training/venv/Scripts/pytest.exe" tests/
```

## Scope rules

Do not touch:
- `KaizerBackend/pipeline_core/pipeline.py` (anything beyond the Step 0
  `GEMINI_PROMPT` swap)
- The 4 existing platform paths
- `models.Clip` database schema
- `editor_meta.json` format (we MATCH it via the adapter, byte-for-byte)
- v1 R2 storage keys (`jobs/{id}/*` without `/v2/`)
