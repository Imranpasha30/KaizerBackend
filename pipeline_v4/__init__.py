"""V4 pipeline — Trim + Canvas composite architecture.

Two-pass design that eliminates lipsync drift entirely:

  Step 1 (trim_engine):
    Source video → Deepgram word-level STT → Claude KEEP/CUT plan →
    ONE atomic ffmpeg pass (trim each KEEP block + concat + audio fade
    + libx264 + aac, all in one filter_complex) → trimmed.mp4

  Step 2 (canvas_engine):
    trimmed.mp4 + canvas JSON (text panels, image plan, brand chrome)
    → ONE ffmpeg pass that OVERLAYS layers on the video without
    re-encoding audio (-c:a copy) → bulletin.mp4 / short_NN.mp4

The canvas JSON is the source of truth for layout + timing. The
editor reads it, the user edits image swaps / durations / titles /
ordering, the editor writes it back, and Step 2 alone re-renders.
Step 1 output is sunk cost — image edits never trigger a re-trim.

Files:
  canvas_schema.py   Pydantic models for the canvas JSON
  text_renderer.py   PIL-based Telugu text panel PNG generator
  trim_engine.py     Step 1: transcribe + plan + atomic trim/concat
  canvas_engine.py   Step 2: composite trimmed video + overlays
  prompts.py         Claude prompts for KEEP/CUT + image durations
  orchestrator.py    Job entry point; calls Step 1 then Step 2
  cli.py             Subprocess entry point (invoked by runner.py)
"""
