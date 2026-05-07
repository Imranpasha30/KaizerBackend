# Video-Editing Enhancement Plan for Kaizer News

> **Goal**: enhance Kaizer's long-video → vertical-clip editing pipeline.
> **Method**: research three reference projects, audit Kaizer's current
> capabilities, identify gaps, propose a phased roadmap.

Last updated: 2026-04-30.

---

## TL;DR

| Reference repo | What it does | Useful to Kaizer? |
|---|---|---|
| **darkzOGx/youtube-automation-agent** | Node.js. Generates **original** YouTube videos from trending topics (script + thumbnail + SEO + auto-upload). No video editing. | ❌ Not useful — different problem |
| **RayVentura/ShortGPT** | Python. Generates **faceless** AI-narrated shorts from topics. MoviePy + ElevenLabs/EdgeTTS + Pexels. Has a translation/dubbing engine. | 🟡 Steal: TTS narration, dubbing engine, declarative EML editing markup |
| **rushindrasinha/youtube-shorts-pipeline** *(actually "Verticals v3")* | Python. Generates **niche** shorts (90s) from a topic + research. Word-level animated captions (yellow pop), Ken Burns, voice ducking. | 🟢 Steal: word-level ASS captions, Ken Burns, voice ducking, niche profiles, resumable pipeline state |

**Bottom line**: building on top of these 3 projects doesn't help our core
clip-from-long-video flow — they all generate from scratch. But the
**video-editing techniques they use (animated captions, Ken Burns motion,
voice ducking, declarative editing markup, dubbing) are all real wins**
we can layer onto Kaizer's existing pipeline. This plan extracts those
techniques into 6 concrete enhancements ranked by impact.

---

## Part 1 — What each reference project does

### 1.1 darkzOGx/youtube-automation-agent

- **Stack**: Node.js 18+, OpenAI / Gemini, YouTube Data API
- **Pipeline**: research trending topics → write script → design
  thumbnail → optimize SEO → upload + schedule → monitor analytics
- **Architecture**: agent-based — Content Strategy, Script Writer,
  Thumbnail Designer, SEO Optimizer, Publishing, Analytics agents
- **Video editing**: ❌ NONE. It only generates metadata (titles,
  scripts, thumbnails). The actual video file is assumed to exist or
  be sourced elsewhere.
- **Verdict**: Doesn't address Kaizer's problem at all. Skip.

### 1.2 RayVentura/ShortGPT

- **Stack**: Python, MoviePy, OpenAI, ElevenLabs, EdgeTTS, Pexels,
  TinyDB
- **Pipeline**: script generation → voiceover synthesis → asset
  sourcing → captions → MoviePy assembly → YouTube upload
- **Architecture**:
  - `ContentShortEngine` — short-form generator
  - `ContentVideoEngine` — long-form generator
  - **`ContentTranslationEngine` — dubs/translates videos across
    languages** *(interesting for Kaizer)*
  - **`EditingEngine` — uses EML/JSON markup language for declarative
    AI-driven edits** *(useful pattern)*
- **Editing techniques**: voiceover sync, captions, b-roll insertion,
  music — all via MoviePy
- **Verdict**: Generation tool, not a clipper. But two ideas are
  directly applicable:
  - **Translation/dubbing engine** — Kaizer renders Telugu but a
    creator publishing to global audiences needs Hindi/English variants.
  - **EML markup** — declarative spec of "what the editor should
    render," consumed by an AI-driven editor.

### 1.3 rushindrasinha/youtube-shorts-pipeline (Verticals v3)

- **Stack**: Python, ffmpeg, Whisper, Edge TTS / Kokoro / ElevenLabs,
  Claude / Gemini / GPT, Replicate / Pexels / ComfyUI
- **Pipeline**:
  1. **Research**  — DuckDuckGo + URL scraping for fact-checking
  2. **Script**    — LLM with niche-specific tone
  3. **Visuals**   — 3-5 b-roll frames (AI image gen or stock)
  4. **Voice**     — TTS with niche-suggested pacing
  5. **Captions**  — Whisper → word-level ASS (burned-in) + SRT
  6. **Assemble**  — ffmpeg + Ken Burns + voice ducking
  7. **Upload**    — YouTube with metadata + thumbnail + SRT
- **Niche profiles**: 15 built-in YAML configs (fitness, finance, tech,
  etc.) that shape tone, visuals, voice, captions, music
- **Editing techniques**:
  - **Word-level ASS captions** — each word highlighted in yellow as
    it's spoken. Major engagement boost.
  - **Ken Burns** — animated zoom/pan on still b-roll frames
  - **Voice ducking** — auto-reduce music volume during narration
  - **Vertical 9:16 auto-crop**
- **Verdict**: Most aligned with Kaizer's editing concerns. Several
  techniques transferable.

---

## Part 2 — Kaizer's current state (what we already have)

`pipeline_core/pipeline.py` is the heart of the system. Audit:

| Feature | Status | Code path |
|---|---|---|
| Long-video → multi-clip extraction (Gemini-driven scene selection) | ✅ Done | `cut_video_clips`, `analyze_video_with_gemini` |
| Vertical 9:16 reframing | ✅ Done | `compose_clip` w/ torn_card / follow_bar / split_frame layouts |
| Telugu / multi-language captions on cards | ✅ Done | `compose_clip` with NotoSansTelugu-Bold default |
| Per-channel logo overlay | ✅ Done | `youtube/logo_overlay.py` (applied at upload time) |
| SEO generation (title, description, tags) | ✅ Done | Gemini SEO via `seo/` |
| Editorial image (news photo alongside video) | ✅ Done | Now OpenAI `gpt-image-1` (`pipeline_core/openai_images.py`) → Pexels → Google CSE → DDG fallback |
| Thumbnail extraction | ✅ Done | `pipeline_core/thumbnails.py` (face_lock, quote_card, punch_frame) |
| OAuth + multi-channel publish | ✅ Done | `routers/youtube_oauth.py`, `youtube/uploader.py` |
| Bilingual SEO variants | ✅ Done | per-channel SEO variants |
| **Word-level animated captions** | ❌ Missing | — |
| **Ken Burns motion on editorial image** | ❌ Missing | — |
| **AI voice-over narration** | ❌ Missing | — |
| **Background music with voice ducking** | ❌ Missing | — |
| **Cross-language dubbing** | ❌ Missing | — |
| **Hook generation (first 3-sec attention grab)** | 🟡 Partial — Gemini picks the moment but no special editing on the hook | — |
| **Smart transitions between clips** | ❌ Missing — uses hard cuts | — |
| **B-roll insertion for visual variety mid-clip** | 🟡 Partial — single still image overlay, not mid-clip cuts | — |
| **Silence / hesitation trimming** | ❌ Missing | — |

**The gap is squarely on the visual-polish + audio-polish side.**
Kaizer's intelligence (Gemini-driven scene selection, AI captions,
SEO) is already strong. What's missing is the **production-quality
editing techniques** that make a clip feel professional rather than
"AI-generated."

---

## Part 3 — Six concrete enhancements (ranked by impact)

### 🥇 Enhancement 1 — Word-level animated captions *(highest engagement ROI)*

**What**: every word lights up yellow as it's spoken — Verticals-style.
Modern viral-clip standard.

**Why**: word-level captions are the single biggest engagement
multiplier for shorts. Studies show 80%+ retention vs 50% for static
captions. Kaizer currently burns Telugu text as a static card.

**How**:
1. Add Whisper (large-v3) transcription pass per clip after compose.
   Returns word-level timestamps.
2. Generate ASS subtitle file with `\k` karaoke timing or per-word
   `\an5\fad` highlighting.
3. Burn into clip via ffmpeg `subtitles=clip.ass:force_style=...`.

**Cost**: Whisper local CPU (free) — ~5-10s per minute of audio.
GPU much faster. Or use OpenAI Whisper API at $0.006/minute.

**Effort**: 2-3 days. New module `pipeline_core/captions.py`,
modification to compose step.

**File targets**: `pipeline_core/pipeline.py:compose_clip`,
`pipeline_core/captions.py` (new).

### 🥈 Enhancement 2 — Ken Burns motion on editorial image

**What**: instead of a static image overlay, slowly pan + zoom the
editorial image (3-5% per second) so it feels alive.

**Why**: matches modern news-broadcast aesthetic. Static images on
9:16 video look dated. Verticals does this on every b-roll frame.

**How**:
- ffmpeg `zoompan` filter applied to the image input track before
  overlay. Single line in the filter graph.
- Direction (zoom-in / zoom-out / pan-left / pan-right) randomly
  picked per clip so the same image doesn't always feel identical.

**Cost**: free (no extra API).

**Effort**: 1 day. Filter-graph modification in `compose_clip`.

**File targets**: `pipeline_core/pipeline.py:compose_clip` —
extend the `-filter_complex` chain.

### 🥉 Enhancement 3 — AI voice-over narration *(optional per channel)*

**What**: optional per-channel toggle: instead of using the original
speaker's voice, generate a Telugu/Hindi/English voice-over from the
clip transcript via ElevenLabs or Edge TTS, then ducked-music underneath.
Useful for channels without a presenter.

**Why**: opens the platform to creators who don't have on-camera
talent. Important for the SaaS user base. ShortGPT and Verticals
both default to TTS narration.

**How**:
- New module `pipeline_core/narration.py`
- Channel setting: `narration_mode = none | replace_audio | overlay`
- Edge TTS (free, 30+ languages, including Telugu) as default
- ElevenLabs as paid premium option for natural voices
- ffmpeg audio chain: orig audio (volume=0 or ducked) + narration
  + bgm (ducked when narration plays)

**Cost**: Edge TTS free. ElevenLabs ~$0.30/1000 chars.

**Effort**: 4-5 days. Audio mixing is finicky but well-trodden.

**File targets**: `pipeline_core/narration.py` (new),
`pipeline_core/pipeline.py:compose_clip` (audio chain modification),
`models.py:Channel` (add narration_mode column).

### 🏅 Enhancement 4 — Cross-language dubbing engine

**What**: take a Telugu clip → produce Hindi + English dubbed versions
automatically. Each version published to the corresponding language
channel.

**Why**: a single creator with one Telugu source video reaches 3× the
audience. Kaizer's Style Profiles already support per-channel
language; this closes the loop.

**How** (mirror ShortGPT's `ContentTranslationEngine`):
1. Extract original audio → Whisper transcribe → original-language SRT
2. Translate SRT via Gemini → target-language SRT
3. TTS in target language (Edge TTS / ElevenLabs)
4. Re-mix: target audio over original video with original audio
   ducked to silence
5. Burn target-language captions

**Cost**: Whisper free + Gemini ~$0.001/clip + TTS minimal.

**Effort**: 5-7 days. Need to handle pacing — translated text is
often longer/shorter, so audio length differs from video length.
Solutions: (a) speed-adjust target audio to fit, (b) trim/extend
silence frames in source video.

**File targets**: `pipeline_core/translation.py` (new),
new endpoint `POST /api/clips/{id}/dub`, frontend UI to pick target
languages.

### 🏆 Enhancement 5 — Hook editing (first 3 seconds)

**What**: identify the "hook" moment within each clip, apply
heightened production values to the first 3 seconds: faster cuts,
zoom-in animation, on-screen "🚨 BREAKING" or numbered list teaser.

**Why**: 80% of viewers drop in the first 3 seconds. The strongest
hook = the strongest retention metric.

**How**:
- Gemini already returns the most-engaging line per clip — use that
  to identify the hook timestamp
- Add a `hook_overlay` mode to `compose_clip`:
  - 1.5×–2× zoom-in on the speaker face for the first 3s
  - Optional banner overlay: "🚨 BREAKING" in red, fades after 2s
  - Optional teaser: extracted noun phrase from the hook line (e.g.
    "$2 BILLION SCANDAL?")
- Per-clip toggle so creators can A/B test

**Cost**: free (computed from existing Gemini output).

**Effort**: 3-4 days.

**File targets**: `pipeline_core/pipeline.py:compose_clip`,
`pipeline_core/hook.py` (new).

### 🎖️ Enhancement 6 — EML-style declarative editing markup

**What**: borrow ShortGPT's idea — every clip's render spec is a
JSON document the AI editor writes, which the renderer consumes.
Today the spec is implicit (frame_layout + card_params + section_pct
+ follow_params split across 4 columns). Refactoring it into a
single `edit_spec` JSON would:
- Make the editor's UI a single JSON-form view
- Make A/B testing trivial — change one field, re-render
- Make AI-driven re-edits feasible — Gemini outputs a JSON,
  pipeline renders it
- Allow versioned edit specs so past edits are recoverable

**Why**: enables future agentic edit flows ("AI, make this clip
30% punchier") that are otherwise impossible to express.

**How**: schema migration. Existing `card_params` / `section_pct`
become fields in a single `edit_spec` JSON column. compose_clip
reads from the new column with a back-compat shim for old rows.

**Cost**: free.

**Effort**: 1 week. Mostly data-modelling + UI rewrite. High
indirect value, low immediate user-visible value.

**File targets**: `models.py` (Clip model), `routers/editor.py`
(new), `pipeline_core/pipeline.py:compose_clip`.

---

## Part 4 — Recommended phasing

I recommend the order **1 → 2 → 3 → 4 → 5 → 6**:

| Phase | Effort | User-visible impact |
|---|---|---|
| Phase 1: word-level captions | 2-3 days | 🔥 Massive — engagement multiplier |
| Phase 2: Ken Burns motion | 1 day | 🔥 Big — clips immediately look professional |
| Phase 3: AI voice-over | 4-5 days | 🔥 Opens platform to faceless creators |
| Phase 4: dubbing engine | 5-7 days | 🟢 Big — 3× audience reach per source video |
| Phase 5: hook editing | 3-4 days | 🟢 Real engagement bump for early viewer retention |
| Phase 6: EML spec | 1 week | 🟡 Indirect — unlocks future agentic edits |

**Total: ~3-4 weeks of dev work for all six.**

If we have to pick ONE: **Phase 1 (word-level captions)**. Single
biggest user-visible improvement to clip quality. Everything else can
follow.

---

## Part 5 — Things we are explicitly NOT doing

- Adopting any of the 3 reference projects wholesale. None of them
  solve our core clip-from-long-video problem.
- Switching from ffmpeg to MoviePy (ShortGPT's choice). MoviePy is
  Python-pure but ~10× slower than ffmpeg for the same operations.
  Kaizer's pipeline already runs on the user's local Windows machine
  with NVENC GPU acceleration; MoviePy can't tap into NVENC.
- Migrating to Node.js (darkzOGx's stack). Our Python pipeline +
  Postgres + R2 architecture is mature and matches the team's skills.
- Building niche YAML profiles like Verticals. Kaizer's Style Profiles
  already serve this need at the channel level.
- Generating original videos from scratch. That's a different product
  category (faceless reels). Kaizer's value is creator-driven content
  with a real human source.

---

## Part 6 — Verification approach (when we ship a phase)

Each phase ships with:

1. **Local test**: run pipeline on a sample 10-min Telugu news video.
   Compare before/after clips side-by-side.
2. **Engagement A/B**: enable the new feature on one channel only,
   leave another channel as control. After 2 weeks compare retention
   curves and CTR. If new feature underperforms, kill switch via env
   var.
3. **Cost telemetry**: every external API call logged via
   `learning/gemini_log.py`-style accounting so we know real per-clip
   cost before opening to public creators.

---

## Appendix: file-level changes needed for Phase 1 (word-level captions)

If approved as the next priority, here's the concrete delta:

1. **New file** `pipeline_core/captions.py` (~150 lines):
   - `transcribe_clip(path, language) → list[Word]` using Whisper
   - `build_ass(words, style, output_path)` writing karaoke-timing ASS
2. **Edit** `pipeline_core/pipeline.py:compose_clip`:
   - After clip is composed but before final mux, run Whisper
   - Burn ASS via `ffmpeg -vf subtitles=clip.ass:force_style=...`
3. **Edit** `pipeline_core/hw_accel.py`: ensure subtitle filter is
   ordered after NVENC-friendly filters
4. **New env vars**: `KAIZER_WORD_CAPTIONS=1` (kill switch),
   `KAIZER_WHISPER_MODEL=large-v3`
5. **Channel toggle**: `Channel.word_captions_enabled` (boolean,
   default True) so creators can opt out
6. **Verification**: render one clip with vs without, eyeball
   readability + timing, ship behind kill-switch

---

*Document drafted from analysis of darkzOGx/youtube-automation-agent,
RayVentura/ShortGPT, and rushindrasinha/youtube-shorts-pipeline
(Verticals v3) READMEs against the current Kaizer News pipeline at
`pipeline_core/pipeline.py`.*
