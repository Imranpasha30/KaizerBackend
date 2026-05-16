"""HeyGen avatar video generation — replaces Veo 3 in Trending.

Pipeline per generation:
  1. ``heygen.transcript.fetch_transcript(video_url)``
       → tries YouTube caption track first (free, ~5 s)
       → falls back to audio + ``express.whisper.transcribe`` (Groq
         Whisper Large v3, ~10-30 s)
  2. ``heygen.script_builder.build_script(topic, transcript)``
       → single Gemini call ("news anchor" tone)
       → output ≤ 700 chars (~60-90 s spoken in Telugu)
  3. ``heygen.client.generate_video(api_key, avatar, voice, script)``
       → POST /v2/video/generate, returns ``video_id``
  4. Poll ``heygen.client.get_status(...)`` every 6 s
       → terminal states ``completed`` (with ``video_url``) or ``failed``
  5. Download the MP4 + auto-generate a thumbnail (ffmpeg frame 0)
  6. Create a Job + Clip row so the asset flows through the existing
     SEO / Publish / Uploads pipeline

External deps (env):
  - ``HEYGEN_API_KEY``      required
  - ``GEMINI_API_KEY``      required (script compression)
  - ``GROQ_API_KEY``        recommended (transcript fallback)
  - ``HEYGEN_DEFAULT_AVATAR_ID``  optional fallback
  - ``HEYGEN_DEFAULT_VOICE_ID``   optional fallback
"""
