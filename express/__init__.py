"""Express Mode — the "lazy-user" auto-publish pipeline.

Mirrors the teammate's ``postiz-yt-dashboard`` autopub flow:
  upload video → Whisper transcribe → Claude SEO + shorts + trim plan
  → ffmpeg renders → Postiz upload → multi-platform publish.

The user fills two short fields (brief, names hint), picks a few
strategy toggles, and hits one button. No editor, no clip review.

Module layout
-------------
- ``state``         in-memory job tracking + TTL cleanup
- ``whisper``       Groq + OpenAI Whisper transcription
- ``claude``        Anthropic Claude for SEO / shorts / trim
- ``telugu_title``  rsvg-convert (preferred) + Pillow fallback
- ``pipeline``      the orchestrator (port of runAutopubPipeline)

External deps (env)
-------------------
- ``ANTHROPIC_API_KEY``  required
- ``GROQ_API_KEY``       preferred (free tier, Telugu accurate)
- ``OPENAI_API_KEY``     fallback for Whisper + thumbnail/inset
- ``POSTIZ_API_KEY``     reused via clients/postiz.py
- ``KAIZER_RSVG_PATH``   optional override for rsvg-convert binary
"""
