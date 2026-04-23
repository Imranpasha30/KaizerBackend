# Phase 6 — Autonomous Live Director (ALD)

> **Status**: architecture + ingest layer in active build (agent dispatch). Not yet shipped.
> The creator SaaS (Phases 1–5) and the Live Director are two distinct surfaces sharing
> the Kaizer DB, R2 storage, and Gemini integration — but no single code path.

---

## Product thesis

Kaizer is **not** a video-editing tool. Kaizer is **two autonomous AI systems** that
remove the need for a human operator:

1. **Creator SaaS** (Phases 1-5): one input video → finished multi-platform content.
2. **Live Director** (Phase 6, this document): N live camera feeds → a single auto-
   switched program feed during a live concert/event, zero human director.

Phase 6 is the second product surface. Runs alongside the SaaS, shares the same
user/auth/storage infrastructure, but introduces a new long-lived-process model
(live events can run for hours) the pipeline-job model doesn't cover.

---

## Target experience

1. An event crew plugs N cameras (and mics) into whatever switcher/bridge they
   already own — OBS Studio, vMix, BMD ATEM, Wirecast — and configures each
   camera to **push RTMP** to Kaizer's ingest endpoint:
   `rtmp://<kaizer-host>:1935/live/<event-id>/<camera-id>`.
   (v2 adds SRT, v3 adds NDI, v4 adds WebRTC.)
2. Crew hits **Go Live** in the Kaizer frontend. From that moment:
   - Every feed is ingested + analyzed in real-time (≤2 s end-to-end latency).
   - A rule-driven AI director picks the "right" camera each moment.
   - A program encoder cuts between cameras, adds lower-thirds/overlays, pushes
     the program feed to YouTube Live / Twitch / file.
   - All ISO cameras record in parallel to R2 so a post-event polished edit is
     always possible.
3. Humans can **override** at any moment through the control surface — pin a cam,
   blacklist a cam, force a cut. Overrides are logged.
4. On end-of-event, Kaizer produces: the streamed program MP4, N ISO MP4s, a
   director decision log (JSON), and a cue sheet for post-production.

---

## 6.1 — Ingestion layer (`live_director/ingest.py`)

### Responsibilities

- Accept **RTMP pushes** from N cameras on one TCP port (1935) at paths
  `/live/<event_id>/<camera_id>`.
- Spawn one persistent `FFmpeg` subprocess per camera:
  ```
  ffmpeg -i rtmp://… -f rawvideo -pix_fmt bgr24 -s 1080x1920 -r 30 pipe:1
  ```
  → raw BGR frames into an asyncio `StreamReader`.
- Parallel audio extraction: `-f s16le -ar 16000 -ac 1 pipe:2` for mono 16-kHz PCM.
- Push frames + samples into a per-camera `RingBuffer` (see 6.1.2).
- Auto-restart on subprocess crash (exponential backoff capped at 30 s).
- Expose health stats: lag_ms, fps, dropped_frames.

### 6.1.1 — RTMP server

For v1: use **MediaMTX** (single Go binary, downloadable per-platform,
production-proven). Launched as a child process by the backend. Config:
```yaml
paths:
  ~^live/.*$:
    source: publisher
    readUser: kaizer
    readPass: <env R2_SECRET_ACCESS_KEY or generate>
```
Pushers authenticate with the event's unique key; readers authenticate with a
token issued by the Kaizer backend (so only the analyzers and the UI preview
can pull feeds).

### 6.1.2 — Ring buffer (`live_director/ring_buffer.py`)

Per-camera circular buffer holding the last ~5 seconds of:
- Decoded BGR frames, keyed by monotonic timestamp
- Raw PCM audio samples (16-bit, 16 kHz mono)

Analyzers pull from the ring buffer rather than hooking into the stream
directly — this decouples decode pace from analysis pace and lets multiple
analyzers share the same frames without re-decoding.

Implemented with `collections.deque(maxlen=…)` behind an asyncio `Lock`, with
separate video and audio deques. Latest-N retrieval + full-range scan.

### 6.1.3 — Signal types (`live_director/signals.py`)

Stable dataclasses passed between every layer. Defined once here so the
ingest, analyzers, director, and composer all speak the same grammar.

```python
@dataclass
class CameraConfig:
    id: str                  # e.g. "cam_stage_left"
    label: str
    mic_id: str | None       # "ambient" or a specific mic feed
    role_hints: list[str]    # ["stage", "wide", "closeup_artist_1"]

@dataclass
class SignalFrame:
    cam_id: str
    t: float                 # monotonic s since event start
    audio_rms: float         # 0-1 normalised
    vad_speaking: bool
    face_present: bool
    face_size_norm: float    # largest face bbox area / frame area
    face_identity: str | None
    scene: str               # 'stage' | 'crowd' | 'wide' | 'closeup' | 'graphic' | 'unknown'
    motion_mag: float        # 0-1
    reaction: str | None     # 'laugh' | 'cheer' | 'clap' | 'boo' | None
    beat_phase: float | None # 0-1 phase within the current bar

@dataclass
class CameraSelection:
    t: float
    cam_id: str
    transition: str          # 'cut' | 'dissolve' (only 'cut' in v1)
    confidence: float
    reason: str              # human-readable ("artist speaking + face in frame")

@dataclass
class DirectorEvent:
    t: float
    kind: str                # 'selection' | 'override' | 'camera_lost' | 'health'
    payload: dict
```

---

## 6.2 — Real-time analyzers (`live_director/analyzers/`)

Each analyzer is an `asyncio.Task` loop that pulls from one camera's ring buffer
at its own rate, emits `SignalFrame` partials into a `SignalBus` (asyncio Queue).

### 6.2.1 — `audio.py`
- RMS energy from latest 300 ms of PCM.
- Voice activity via `webrtcvad` (aggressiveness 2, 20 ms frames).
- Loudness trend over last 3 s (to distinguish speaking vs sustained tone).

### 6.2.2 — `face.py`
- OpenCV DNN face detector (res10 SSD, 300×300 inference, ~30 ms on CPU).
- Tracks largest face bbox, face_size_norm.
- Face identity: optional `face_recognition` library with a per-event
  embedding registry populated from uploaded artist headshots.

### 6.2.3 — `scene.py`
- Small CNN (MobileNetV2 fine-tuned) classifying each keyframe as
  stage / crowd / wide / closeup / graphic / unknown.
- Inference every 500 ms (not every frame) to keep it cheap.

### 6.2.4 — `motion.py`
- Optical flow magnitude over consecutive frames, normalised.

### 6.2.5 — `reaction.py`
- Audio-event classifier on the program audio (or per-camera mic).
- v1: simple energy-spike + spectral centroid heuristic to detect
  laugh / cheer / clap vs silence.
- v2: YAMNet or BEATs-small model for proper audio-event tagging.

### 6.2.6 — `beat.py`
- `librosa.beat.beat_track` on a rolling 8-second window of program audio.
- Emits a beat-phase signal (0.0 at downbeat, linear to 1.0 at next beat).

### 6.2.7 — SignalBus

A single asyncio fan-in: all analyzer outputs for all cameras merged,
sorted by timestamp, delivered to the director engine.

---

## 6.3 — Director decision engine (`live_director/director.py`)

Single asyncio consumer of the SignalBus. Maintains "which camera is currently
on program" state plus last-cut timestamp for min/max shot enforcement.

### Rule priority (highest → lowest; first match wins)

1. **Manual override in effect** → selected camera is `override.cam_id`.
2. **Critical reaction** (cheer/laugh > threshold, crowd cam available) →
   cut to the crowd cam with strongest audio energy.
3. **Designated speaker active** (VAD=True + face present + face_identity
   matches a tagged artist) → cut to the camera with the largest face bbox
   of that identity.
4. **Beat cut during music** (beat_phase near 0 AND last_cut > min_shot) →
   cut to next-in-rotation camera ordered by energy/motion.
5. **Min-shot-duration floor** — never cut within `min_shot_s` (default 2.5 s).
6. **Max-shot-duration ceiling** — if last_cut > max_shot (default 12 s),
   force a cut to the next highest-scoring camera.
7. **Default** — stay on current camera.

### Config (per event)

```yaml
min_shot_s: 2.5
max_shot_s: 12
reaction_threshold: 0.7
speaker_vad_hold_ms: 400
beat_cut_every_nth_bar: 2
crossfade_on_scene_change: true
```

Config stored in the `live_events` table; editable via the control surface.

### LLM-augmented context (v2)

Once a minute, send the last N seconds of SignalFrames to Gemini with a short
prompt: *"What's happening in this performance right now? Update the director
intent: calm song, energetic song, comedy set, speech, dancing."* The returned
tag biases rule weights (e.g. "comedy set" → reaction_threshold lowers because
laugh-reactions are expected/desirable).

---

## 6.4 — Program composer (`live_director/composer.py`)

Consumes `CameraSelection` decisions and produces the live program feed.

### 6.4.1 — Switcher subprocess

One long-running FFmpeg `filter_complex` with:
- N camera inputs (same as ingest, piped in)
- `streamselect` + `astreamselect` for the video + audio
- `sendcmd` channel to update the active stream on each CameraSelection
- Output to two sinks in parallel:
  - MP4 file on local disk (rotating 10-min chunks, uploaded to R2 on rollover)
  - Optional RTMP push (to YT Live / Twitch)

```
ffmpeg \
  -i rtmp://ingest/live/event123/cam1 \
  -i rtmp://ingest/live/event123/cam2 \
  -i rtmp://ingest/live/event123/cam3 \
  -filter_complex "[0:v][1:v][2:v]streamselect=inputs=3:map=0[vout]; \
                    [0:a][1:a][2:a]astreamselect=inputs=3:map=0[aout]" \
  -map "[vout]" -map "[aout]" \
  -c:v libx264 -preset veryfast -tune zerolatency -b:v 6M \
  -c:a aac -b:a 192k \
  -f flv rtmp://a.rtmp.youtube.com/live2/<stream_key> \
  -f segment -segment_time 600 -reset_timestamps 1 \
    -c copy "event123_program_%03d.mp4"
```

`sendcmd` updates the `streamselect` and `astreamselect` `map=X` param on each
CameraSelection (hard cut). Cross-dissolve requires a `tpad` + `xfade` trick
documented in 6.4.2.

### 6.4.2 — Cross-dissolve transitions (v2)

`xfade` doesn't operate on live streams directly — it requires known clip
durations. For dissolves between live sources, use a small side-process that
merges 300 ms of `tpad`-stretched cam A + 300 ms of cam B with `blend=all_mode=normal`
just around the cut boundary. Hard cut in v1 to ship.

### 6.4.3 — Overlays

Lower-third compositing via `overlay` filter on top of the program output.
Text rendered via the same `captions.render_caption` path used by the SaaS,
guaranteeing Indic-script support out of the box. Template library:

- `lower_third_name` — artist name + role
- `event_title` — top-left logo + event name
- `location_tag` — venue + city
- `countdown` — large centred numerals
- `song_title` — bottom-centred track name during instrumentals

Each overlay is a PNG generated on-demand + scheduled via `enable='between(t,start,end)'`.

---

## 6.5 — Output streaming (`live_director/output.py`)

Abstraction over program sinks:

- **RTMPSink** — pushes FLV to rtmp://a.rtmp.youtube.com/live2/<key>,
  live.twitch.tv/app/<key>, or any custom endpoint.
- **HLSSink** — emits .m3u8 + .ts chunks into R2 for direct browser playback
  via HLS.js on the frontend (low-latency HLS mode).
- **FileSink** — rotating 10-min MP4 chunks on disk, uploaded to R2 on each
  rollover under `events/<event_id>/program/%03d.mp4`.
- **ISORecorder** — N additional FFmpeg subprocesses (one per camera) writing
  its own untouched MP4 for post-event editing. Same rotation + R2 upload.

Each sink runs independently; any one failing doesn't break the others.

---

## 6.6 — Frontend control surface (`src/pages/LiveDirector.jsx`)

Tab order: Home, Jobs, **Live**, QuickPublish, Studio, …

### Layout

```
┌──────────────────────────────────────────────────────────────────────────┐
│  LIVE · event_123_spring_tour_hyderabad       00:14:32   ⬤ LIVE          │
├──────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────┐  ┌───────────────────────┐                    │
│  │  cam_stage   ⬤ ACTIVE │  │  cam_crowd            │                    │
│  │  (thumbnail, 5 fps)   │  │  (thumbnail)          │     ... more       │
│  │  audio_rms: 0.74       │  │  audio_rms: 0.12      │                    │
│  │  face: artist_1        │  │  face: (none)         │                    │
│  │  [pin] [blacklist]     │  │  [pin] [blacklist]    │                    │
│  └───────────────────────┘  └───────────────────────┘                    │
├──────────────────────────────────────────────────────────────────────────┤
│  PROGRAM OUTPUT                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │   (large HLS.js player — live program preview, ~3s behind)         │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  Director: auto ●      Override: [none ▾]      Shot: 4.2 s / 12 s       │
├──────────────────────────────────────────────────────────────────────────┤
│  LOG                                                                     │
│  00:14:28  auto → cam_stage   (artist speaking + face in frame)          │
│  00:14:12  beat-cut → cam_crowd                                          │
│  00:14:08  auto → cam_stage   (vad_hold 400ms)                           │
│  ...                                                                     │
└──────────────────────────────────────────────────────────────────────────┘
```

WebSocket: `ws://backend/api/live/<event_id>/stream` pushes:
- Thumbnail JPEGs (2-3 fps) per camera
- DirectorEvent stream (JSON)
- SignalFrame summaries (for the per-camera overlays)

### Override controls

- **Pin camera** — force a single cam until unpinned.
- **Blacklist camera** — never cut to this cam (e.g. a cam that went fuzzy).
- **Force cut to…** — instant override cut, logged.

---

## Data model additions

New tables in `models.py`:

```python
class LiveEvent(Base):
    __tablename__ = "live_events"
    id            = Column(Integer, primary_key=True)
    user_id       = Column(Integer, ForeignKey("users.id"))
    name          = Column(String(255), nullable=False)
    venue         = Column(String(255), default="")
    starts_at     = Column(DateTime(timezone=True))
    ends_at       = Column(DateTime(timezone=True), nullable=True)
    status        = Column(String(20), default="scheduled")   # scheduled | live | ended | failed
    config_json   = Column(JSON, default=dict)   # min_shot_s, max_shot_s, thresholds, overrides
    rtmp_key_hash = Column(String(64), default="")  # partner ingestion auth
    program_url   = Column(String(500), default="")
    created_at    = Column(DateTime(timezone=True), server_default=func.now())


class LiveCamera(Base):
    __tablename__ = "live_cameras"
    id          = Column(Integer, primary_key=True)
    event_id    = Column(Integer, ForeignKey("live_events.id"), nullable=False)
    cam_id      = Column(String(64), nullable=False)
    label       = Column(String(255))
    mic_id      = Column(String(64), default="")
    role_hints  = Column(JSON, default=list)
    iso_url     = Column(String(500), default="")   # R2 ISO recording URL
    __table_args__ = (UniqueConstraint("event_id", "cam_id"),)


class DirectorLogEntry(Base):
    __tablename__ = "director_log"
    id          = Column(Integer, primary_key=True)
    event_id    = Column(Integer, ForeignKey("live_events.id"), nullable=False, index=True)
    t           = Column(Float, nullable=False)
    kind        = Column(String(32), default="selection")
    cam_id      = Column(String(64), default="")
    confidence  = Column(Float, default=0.0)
    reason      = Column(Text, default="")
    payload     = Column(JSON, default=dict)
```

---

## Non-goals (explicitly out of scope for Phase 6)

- Full professional vision-mixer features (DSK, CG graphics engines). Overlays
  in 6.4.3 are text/PNG templates, not full NLE titles.
- Server-side audio mixing across cameras (we pick ONE camera's audio per
  switch; merging multi-mic audio is a post-event task).
- Video effects during program output (no color grading, no slow-mo).
- Storage-tier archival policies (handled separately by R2 lifecycle rules).
- Multi-region redundancy / failover (single-region for now).

---

## Build order & milestones

| Wave | Deliverable | Notes |
|------|-------------|-------|
| Foundation | `live_director/` package scaffolding + `signals.py` + `ring_buffer.py` + `ingest.py` + DB models + startup wiring | This session |
| Analyzers | `analyzers/` (audio, face, motion, scene stub); SignalBus asyncio Queue | Next session |
| Director | `director.py` rule engine + config loading + override handling | Next session |
| Composer | `composer.py` program FFmpeg subprocess + hard-cut switching | Next session |
| Output | `output.py` RTMP sink + HLS sink + MP4 file sink + ISO recorder | Next session |
| Frontend | `pages/LiveDirector.jsx` + WebSocket streams + override controls | Next session |
| Smoke test | 2 simulated RTMP feeds (ffmpeg lavfi + test mp4) → full pipeline → MP4 artefact in R2 | Next session |

Each wave ends with a committed changeset, test coverage ≥ 20 cases per new
module, zero regressions against the 457 baseline fast tests.

---

## Dependencies to add

```
webrtcvad==2.0.10
# aiortc + deepspeech can wait until v2 WebRTC / full reaction model
```

MediaMTX (RTMP server binary) downloaded to `resources/mediamtx/` on first run
via a small helper; not vendored into the repo (multi-platform binaries).

---

## Resumption notes

If a future session needs to pick this up cold:

1. Read this doc.
2. Check `git log --oneline` for the most-recent `live(` prefixed commits.
3. Run `pytest tests/test_live_director*.py -q` to confirm the current layer.
4. The next wave in the build-order table is the one to tackle.
5. `docs/USER_MANUAL.md` has teammate-facing overview content — update its
   "Live Director" section after each wave ships.
