# Express Mode — operator setup

Express Mode is Kaizer's "one-click auto-publish" tab. It ports the
teammate's `postiz-yt-dashboard` pipeline (Whisper → Claude → ffmpeg
→ Postiz) into Kaizer's Python/FastAPI core. Three modes:

| Mode | What it does | Time per 5-min source |
|---|---|---|
| `publish-as-is` | Whisper transcribe → Claude SEO → Postiz upload of the original video. No re-render. | ~30-60 s |
| `ai-trim` | Whisper → Claude SEO + trim plan → ffmpeg trim + grade + (optional) cinematic xfade + Ken Burns + grain → optional AI thumbnail → Postiz. | ~3-6 min |
| `shorts` | Whisper → Claude SEO + shorts plan (3-5 moments) → loop: optional AI inset (gpt-image-1) + cutClip with TV news split panel → Postiz post per short. | ~5-10 min |

---

## Required setup

### 1. API keys (per-user, stored in browser localStorage)

| Provider | Use | Get it from |
|---|---|---|
| **Anthropic** | Claude Sonnet 4.6 — SEO, trim plan, shorts plan | console.anthropic.com |
| **Groq** *(recommended)* | Whisper Large v3 — best Indic-language accuracy | console.groq.com |
| **OpenAI** *(alternative)* | Whisper-1 (Indic less accurate) **and** gpt-image-1 for AI inset/thumbnail | platform.openai.com |
| **Postiz** | YouTube publishing | Already in Kaizer's `.env` (`POSTIZ_API_KEY` + `POSTIZ_BASE_URL`) |

Keys are typed into the **Step 1: API keys** section of `/express`
and persisted in browser localStorage only. The backend never sees
them on disk — they're forwarded per-request as multipart form
fields to the upstream API.

### 2. Telugu rendering quality — install `rsvg-convert` *(optional, recommended)*

Without `rsvg-convert`, Express Mode falls back to Pillow + FreeType
for the Shorts title PNG. That works but doesn't run Pango's full
Indic shaping engine — complex conjunct stacks (e.g. ష్ట్ర, ద్రు)
may render less cleanly than they would on the teammate's macOS
setup.

**Install on Windows — pick ONE path:**

#### Option A: Scoop *(cleanest)*
```powershell
# Install Scoop if missing
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex

# Then install librsvg
scoop install librsvg
```
After this, `rsvg-convert.exe` is on PATH; Kaizer auto-detects it on
the next compose.

#### Option B: MSYS2
```powershell
# Install MSYS2 from https://www.msys2.org/, then in its shell:
pacman -S mingw-w64-x86_64-librsvg

# Add to system PATH:
#   C:\msys64\mingw64\bin
```

#### Option C: explicit env override
If you have an `rsvg-convert.exe` at a custom path (e.g. from a GIMP
install at `C:\Program Files\GIMP 2\bin\`), point Kaizer at it
directly:
```
KAIZER_RSVG_PATH=C:\Program Files\GIMP 2\bin\rsvg-convert.exe
```
Then restart the backend.

**Verify**: in `/express`, render a Shorts cut with a Telugu hook
containing a complex conjunct (e.g. `*షాకింగ్* నిజం`). With
`rsvg-convert` active, the conjunct stacks render correctly. Without,
the Pillow fallback may show simpler shaping.

---

## How the three modes consume keys

```
publish-as-is:  ANTHROPIC + (GROQ or OPENAI) + POSTIZ
ai-trim:        ANTHROPIC + (GROQ or OPENAI) + POSTIZ
                  + OPENAI (only if thumbnail_strategy = "ai")
shorts:         ANTHROPIC + (GROQ or OPENAI) + POSTIZ
                  + OPENAI (only if inset_strategy = "ai")
```

If the OpenAI key is blank in the UI, the AI Trim "AI thumbnail" and
Shorts "AI per short" dropdown options are disabled — falls back to
no thumbnail and to video-frame insets respectively.

---

## Backend module layout

```
KaizerBackend/express/
  state.py          — in-memory job map (6h TTL, multi-tenant by user_id)
  whisper.py        — Groq / OpenAI / custom Whisper transcription
  claude.py         — write_seo, plan_longform_trim, plan_shorts,
                      target_short_count, target_trim_window,
                      JSON salvage helpers
  color_grade.py    — 6 ffmpeg filter-chain presets
  render_longform.py — renderLongformTrim port (trim + grade + xfade)
  telugu_title.py   — title PNG: rsvg-convert preferred, Pillow fallback
  news_panel.py     — buildNewsLayoutFilter port (TV split layout)
  cut_clip.py       — cutClip port (orchestrator per Short)
  ai_image.py       — gpt-image-1 wrapper + styled prompts
  dot_pattern.py    — panel texture overlay (rsvg or Pillow)
  pipeline.py       — run_publish_as_is, run_ai_trim, run_shorts
```

Router: [routers/express_mode.py](../routers/express_mode.py)
exposes 3 endpoints:
- `POST /api/express/start` — kick off a job (multipart form)
- `GET  /api/express/status/{job_id}` — poll progress
- `GET  /api/express/integrations` — list Postiz channels

---

## Cost estimates per video

For a 5-minute source through full autopub (shorts mode, AI inset):

| Stage | Cost |
|---|---|
| Groq Whisper Large v3 | ~$0.001 (free tier covers it) |
| Anthropic Claude Sonnet 4.6 (2 calls: SEO + shorts plan) | ~$0.03 |
| OpenAI gpt-image-1 (medium quality, 3-5 insets) | ~$0.15-0.30 |
| Postiz | included in your plan |
| **Total** | **~$0.18-0.34 per shorts run** |

AI Trim mode skips per-short images, so it's just $0.03 + optional
$0.05 for the thumbnail = ~$0.08 max.

---

## Troubleshooting

**"Claude response was not valid JSON"**
The trim/shorts planners include salvage logic for truncated
responses. If you still see this, the transcript came back garbled —
try Groq Whisper (better Indic accuracy) or set the language hint to
`te` in Step 3.

**"Whisper says: audio file could not be decoded"**
The audio extract step uses `ffmpeg -i video.mp4 -vn -ac 1 -ar 16000
-b:a 64k -f mp3`. If your source has a malformed `moov` atom (common
on iPhone exports), this still works — we use a temp file, not stdin.

**"Image generation took too long"**
The Shorts inset has a 60s hard timeout, the long-form thumbnail 90s.
Both soft-fail: shorts fall back to a video frame, the thumbnail is
just skipped (YouTube picks a frame automatically).

**Long-form posted to Postiz but not on YouTube**
Postiz returns a `mediaId` as soon as upload + post-creation succeeds
on its side. After that, YouTube's queue can take 5-30 min — check
YouTube Studio → Content.

**Telugu glyphs render incorrectly without rsvg**
Install rsvg-convert (see "Required setup" above). The Pillow fallback
uses FreeType which has decent but not Pango-quality Indic shaping.
