# TRACK 3A — DRIFT INTRODUCTION POINT MAP

Exhaustive catalogue of every place in the Kaizer V2 pipeline where audio/video timing can drift apart, with file:line citations, drift magnitude, current mitigation, and applicability scope.

Sources read (all paths absolute):
- `e:\kaizer new data training\kaizer\KaizerBackend\pipeline_v2\post_v2_backlog.md` (items 111-118)
- `e:\kaizer new data training\kaizer\KaizerBackend\pipeline_v2\pipeline_v2\stages\stage_0_ingest.py`
- `e:\kaizer new data training\kaizer\KaizerBackend\pipeline_v2\pipeline_v2\utils\ffmpeg_runner.py`
- `e:\kaizer new data training\kaizer\KaizerBackend\pipeline_v2\pipeline_v2\utils\ffprobe.py`
- `e:\kaizer new data training\kaizer\KaizerBackend\pipeline_v2\pipeline_v2\stages\stage_1_transcribe.py`
- `e:\kaizer new data training\kaizer\KaizerBackend\pipeline_v2\pipeline_v2\stages\stage_2_continuity.py`
- `e:\kaizer new data training\kaizer\KaizerBackend\pipeline_v2\pipeline_v2\stages\stage_2_providers.py`
- `e:\kaizer new data training\kaizer\KaizerBackend\pipeline_v2\pipeline_v2\stages\stage_4_render.py`
- `e:\kaizer new data training\kaizer\KaizerBackend\pipeline_v2\pipeline_v2\stages\stage_4_raw_extract.py`
- `e:\kaizer new data training\kaizer\KaizerBackend\pipeline_v2\pipeline_v2\stages\stage_4_bulletin_overlay.py`
- `e:\kaizer new data training\kaizer\KaizerBackend\pipeline_v2\pipeline_v2\stages\stage_4_shorts_overlay.py`
- `e:\kaizer new data training\kaizer\KaizerBackend\pipeline_v2\pipeline_v2\bulletin_crossfade_stitcher.py`
- `e:\kaizer new data training\kaizer\KaizerBackend\pipeline_v2\pipeline_v2\render\edl_builder.py`
- `e:\kaizer new data training\kaizer\KaizerBackend\pipeline_core\pipeline.py` (V1 cut + overlay_image_plan)
- `e:\kaizer new data training\kaizer\KaizerBackend\pipeline_core\longform_compose.py` (V1 compose)
- `e:\kaizer new data training\kaizer\KaizerBackend\pipeline_core\bulletin_stitcher.py` (V1 stitcher)
- `e:\kaizer new data training\kaizer\KaizerBackend\pipeline_core\image_carousel.py`

Citation format: `[CLAIM]. [EVIDENCE: file:line + excerpt OR backlog item N]. [CONFIDENCE: HIGH|MED|LOW|UNVERIFIED]`.

---

## STAGE 0 — INGEST (mezzanine encode + parallel audio extract)

### D0.1 — VFR→CFR force re-encode of source video

CLAIM: Stage 0 unconditionally re-encodes the source to 30fps CFR, which can re-time frames if the source had a different nominal framerate (e.g. 25fps PAL, 24fps cinema, 29.97 NTSC, 60fps), discarding or duplicating frames to hit 30. Source-time vs mezzanine-time will *not* match if downstream code (Stage 2) reads source-time from the original word array but operates against the mezzanine.

EVIDENCE: `pipeline_v2/utils/ffmpeg_runner.py:207`
```
"-vsync", "cfr", "-r", "30",
"-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
```
Also `stage_0_ingest.py:13-17`: "The mezzanine is CFR-locked even when the source is VFR because every downstream stage assumes source-time == mezzanine-time."

DRIFT MAGNITUDE:
- Best case (source already 30fps CFR): 0ms.
- Worst case (source 60fps VFR with sustained drift): tens of frames re-timed across a 90-min file. Hard to bound; typically <1s of total drift on smartphone VFR sources, but the *per-segment* effect compounds when Stage 2 cuts at word boundaries derived from the un-re-timed source. UNVERIFIED for high-FPS VFR sources.

MITIGATION: `-vsync cfr -r 30` is set; audio is re-encoded to AAC 48k. There is NO explicit drop-frame / duplicate-frame logging or auditing.

APPLICABILITY: Affects ALL source video types (CFR-locking is unconditional). Risk highest for VFR phone uploads. CONFIDENCE: MED — the code is unambiguous; the actual residual drift on VFR inputs is not quantified anywhere in the backlog.

### D0.2 — Audio extracted from SOURCE (not mezzanine) for STT

CLAIM: Stage 0 extracts source audio in parallel from the *original* source file at 48kHz/128k mp3, not from the (CFR-locked) mezzanine. Stage 1 transcribes that mp3, producing word.s / word.e timestamps in *source* time. If the mezzanine's CFR re-time shifts content, Stage 2 / Stage 4 reads cuts in source-time and applies them to the *mezzanine* (CFR-shifted). Two coordinate systems that *should* coincide but may not.

EVIDENCE: `pipeline_v2/stages/stage_0_ingest.py:164-173`:
```
transcode_secs, audio_secs = await asyncio.gather(
    _timed(transcode_to_mezzanine(str(src), str(mezz), encoder=chosen), "transcode"),
    _timed(extract_audio(str(src), str(audio)), "audio extract"),
)
```
`ffmpeg_runner.py:216` extracts `src`, not `dst`. Stage 4's video_path is the mezzanine (e.g. `stage_4_render.py:1577` `video_path: Path`).

DRIFT MAGNITUDE:
- Best case (CFR source): 0ms.
- Worst case (true VFR source where -r 30 dropped 17 of 1800 frames over a minute): ~600ms accumulated mismatch between word.s timestamps and the mezzanine timeline. UNVERIFIED — no backlog measurement.

MITIGATION: None. The architecture assumes `is_vfr=False` (or that CFR-fixup is lossless), but `VideoProbe.is_vfr` is informational only (`ffprobe.py:37`).

APPLICABILITY: Affects source videos where ffprobe `avg_frame_rate != r_frame_rate` (`ffprobe.py:37-45`). For tripod-locked broadcast cameras this is a non-issue. For smartphone uploads it's a real risk. CONFIDENCE: HIGH (architecture description), MED (magnitude).

### D0.3 — NVENC vs libx264 produce different VFR→CFR behaviour

CLAIM: The two encoder code paths share `-vsync cfr -r 30` but differ in encoder-internal frame-decision: NVENC's `-rc vbr -cq 23` and libx264's `-crf 23` apply different rate-control state machines. Empirically NVENC and libx264 produce sample-accurate audio (since AAC encode is the same), but mid-stream PTS handling on VFR sources can differ by a frame at boundaries.

EVIDENCE: `ffmpeg_runner.py:186-201`. NVENC: `-preset p4 -tune hq -rc vbr -cq 23`. libx264: `-preset medium -crf 23 -pix_fmt yuv420p`.

DRIFT MAGNITUDE: ≤33ms per CFR adjustment boundary (one 30fps frame). Backlog has no measurement.

MITIGATION: None at this layer.

APPLICABILITY: Cross-host (NVENC on prod / libx264 in CI), reproducibility issue. CONFIDENCE: LOW — speculative.

### D0.4 — Stream timebase not pinned

CLAIM: There is NO explicit `-video_track_timescale` / `-time_base` argument in either the mezzanine encode (`ffmpeg_runner.py:203-213`) or any subsequent re-encode. ffmpeg defaults to the codec's natural timebase (1/15360 for H.264/MOV). Different ffmpeg builds (Windows native vs Linux nixpacks) may pick different defaults, surfacing as off-by-one-tick PTS differences when concatenated.

EVIDENCE: Search across `ffmpeg_runner.py`, `bulletin_crossfade_stitcher.py`, `stage_4_raw_extract.py`, `pipeline_core/pipeline.py:1241-1247` shows zero occurrences of `-video_track_timescale`.

DRIFT MAGNITUDE: Sub-frame, ≤1ms per segment, but compounds. UNVERIFIED magnitude.

MITIGATION: None.

APPLICABILITY: ALL source video types when multiple ffmpeg versions process the same pipeline. CONFIDENCE: LOW — theoretical.

---

## STAGE 1 — TRANSCRIBE (no AV drift; included for completeness)

### D1.1 — STT word timestamps from compressed mp3, not original PCM

CLAIM: Stage 1 transcribes the 128k mp3 produced by Stage 0. mp3 encoding shifts perceived word boundaries by up to one mp3 frame (≈26ms at 48k). Downstream cuts at word-boundaries inherit this jitter.

EVIDENCE: `stage_0_ingest.py:14` (`source.mp3`), `ffmpeg_runner.py:216-228` (`libmp3lame, -ar 48000, -b:a 128k`).

DRIFT MAGNITUDE: ≤26ms per word boundary, but it is content-positioning jitter, not A/V drift. Affects *editorial* cut quality, not lip-sync.

MITIGATION: N/A (this is not an A/V drift source).

APPLICABILITY: All. CONFIDENCE: HIGH (mechanism) but does not introduce lip-sync drift.

---

## STAGE 2 — CONTINUITY (non-determinism, downstream timing impact)

### D2.1 — LLM-driven cut decisions are non-deterministic

CLAIM: Stage 2 is non-deterministic even at `temperature=0` per the provider note ("temperature=0 does NOT guarantee identical outputs"). Different runs on the same input produce different `(start_sec, end_sec)` cut boundaries → different splice→silence→micro-fragment expansion → different downstream segment counts. This is not A/V *drift* but it determines how much per-segment drift accumulates.

EVIDENCE: `stage_2_providers.py:333-335`:
> "Determinism: temperature defaults to 0 -- Sonnet 4.6 still / Note that temperature=0 does NOT guarantee identical outputs."

Item 118 (backlog `:1837-1841`): "Stage 2 (Gemini) produced ONLY 1 bulletin cut covering 587.5s -- not the 28-cut decomposition of earlier jobs."

DRIFT MAGNITUDE: Indirect; controls the multiplier on every per-segment drift source below.

MITIGATION: None.

APPLICABILITY: All V2 jobs. CONFIDENCE: HIGH.

### D2.2 — Stage 2 cut boundaries are NOT frame-snapped at emit

CLAIM: Stage 2 emits `start_sec` / `end_sec` as floats with arbitrary precision (Gemini regularly emits 3-4 decimal places). Frame-snapping happens later in `_snap_to_frame_grid` (33.33ms grid). If Stage 2 emits a boundary at 12.3456s, snap rounds to 12.3667s (frame 370 at 30fps) — moving a phoneme by ~21ms inside the cut. Not a *drift* but a positioning offset.

EVIDENCE: `stage_4_render.py:606-608`:
```python
def _snap_to_frame_grid(t: float, frame_s: float = VIDEO_FRAME_S) -> float:
    return round(t / frame_s) * frame_s
```

DRIFT MAGNITUDE: ≤16.7ms (half frame) per boundary. Per-segment, not cumulative across the bulletin.

MITIGATION: Applied symmetrically to both audio and video by snapping the cut request BEFORE the ffmpeg call, so A/V remain in lockstep.

APPLICABILITY: All. CONFIDENCE: HIGH.

---

## STAGE 4 — CUT STEP (legacy + item-112 + item-117 paths)

The cut step has *three* code paths visible in the codebase:
1. V1 `_v1_cut_video_clips` (legacy) — `pipeline_core/pipeline.py:1213-1254`
2. V2 `_cut_one_clip_strict` (item 112+116) — `stage_4_render.py:649-745`
3. Item 117 unified extract `extract_raw_timeline` — `stage_4_raw_extract.py:206-403` via `edl_builder.build_extraction_edl` — `render/edl_builder.py:151-302`

### D4.CUT.1 — V1 legacy cut uses output-side `-t` + re-encode

CLAIM: V1's `cut_video_clips` invokes ffmpeg with `-ss <start> -t <dur> -i <src> + ENCODE_ARGS_INTERMEDIATE`. This re-encodes the segment with `_VIDEO_ARGS` (NVENC or libx264) + AAC. AAC frames quantize to 1024 samples / 21.33ms at 48k; video frames quantize to 33.33ms at 30fps. The two grids don't align; output audio_dur is typically 0-32ms LONGER than video_dur.

EVIDENCE: `pipeline_core/pipeline.py:1241-1247`:
```
cmd = (
    [FFMPEG_BIN, "-y",
     "-ss", str(round(start, 3)), "-t", str(round(dur, 3)),
     "-i", video_path]
    + ENCODE_ARGS_INTERMEDIATE
    + [out_path]
)
```
Backlog item 112 (`:1240-1247`): "audio_dur that runs 0-32ms LONGER than video_dur. AAC frames are 21.33ms; 30fps video frames are 33.33ms."

DRIFT MAGNITUDE: 0-32ms per slice. Best case 0ms, worst case ~+32ms audio-over-video per slice. Across 23-29 segments: ~300-500ms cumulative (item 112 measurement, backlog `:1246-1247`).

MITIGATION: Item 112 replaces this with `cut_clips_frame_aligned` when `Stage4Render.use_frame_aligned_cut=True` (default).

APPLICABILITY: All source video types. Falls back here when `use_frame_aligned_cut=False` or item 117 unified extract fails.

CONFIDENCE: HIGH (backlog item 112 has empirical measurements).

### D4.CUT.2 — V2 frame-aligned cut: pre-item-116 `-to` placement bug (HISTORICAL)

CLAIM: Before item 116, `_cut_one_clip_strict` used `-ss X -to Y -i FILE`. Input-side `-to` is video-INCLUSIVE of the end frame: when `end_snap * 30` lands on an integer, ffmpeg pulls one EXTRA video frame past the cutoff while audio cuts cleanly. Result: video 33ms LONGER than audio. Item 115's `apad=whole_dur=V` then padded the gap with SILENCE.

EVIDENCE: Backlog item 116 (`:1633-1700`). Current code at `stage_4_render.py:702-714` shows the FIXED form (`-ss X -i FILE -t (Y-X)`).

DRIFT MAGNITUDE: -33ms per slice (audio shorter by one frame on half of all boundaries). Cumulative -695.8ms across 28 clips on job 51 (backlog `:1639`).

MITIGATION: Item 116 — `-to` removed, `-t` placed AFTER `-i`. Current code at `stage_4_render.py:703-707`:
```python
cmd = [
    ffmpeg_bin, "-y",
    "-ss", f"{start_snap:.6f}",
    "-i", video_path,
    "-t", f"{duration_s:.6f}",
```

APPLICABILITY: Closed regression. Will NOT recur unless someone reverts the flag order.

CONFIDENCE: HIGH.

### D4.CUT.3 — V2 frame-aligned cut: residual AAC/30fps grid misalignment

CLAIM: Even with item 116's `-t (Y-X)`, AAC quantization can still leave ≤21ms residue per slice because the encoder rounds the encoded audio length up to a 1024-sample frame boundary.

EVIDENCE: `stage_4_render.py:594` (`PER_CLIP_AV_TOLERANCE_S = 0.035`), `:740-744`:
```python
"ok": abs(delta_s) <= PER_CLIP_AV_TOLERANCE_S,
```

DRIFT MAGNITUDE: ≤21ms per slice. Per-clip *audio-over-video* residue.

MITIGATION:
- `_cut_one_clip_strict` does `_probe_av_durations` post-cut and re-cuts with the 100ms fallback grid (`stage_4_render.py:836-841`) if drift > 35ms.
- Item 115's `_align_composed_audio_to_video` runs LATER (per composed segment) to atrim+apad audio to exact video duration (`stage_4_render.py:326-429`).

APPLICABILITY: All source video types when the legacy cut path runs (item 117 unified extract bypasses this).

CONFIDENCE: HIGH.

### D4.CUT.4 — Idempotency cache reuses files across runs without re-verifying flags

CLAIM: `cut_clips_frame_aligned` and V1 `cut_video_clips` reuse a >100KB cached file at `raw_clip_NN.mp4` without re-running the strict flags check. If a prior run produced a file with item 116's buggy `-to` semantics, that cached file persists into a post-fix run.

EVIDENCE: `stage_4_render.py:812-822`:
```python
if _os.path.exists(out_path) and _os.path.getsize(out_path) > 100_000:
    v_dur, a_dur = _probe_av_durations(out_path, ffprobe_bin)
    delta_ms = (a_dur - v_dur) * 1000.0
    cumulative_delta_ms += delta_ms
    clip["raw_path"] = out_path
```

DRIFT MAGNITUDE: Reuses pre-fix bug (-33ms per slice from D4.CUT.2). Job-level: ≤695ms (item 116 measurement).

MITIGATION: The probe DOES update the cumulative delta sum, so the [cut WARN] line fires if accumulated drift > 100ms. But the cached file itself is not re-cut.

APPLICABILITY: Any job where `output_dir` was created before a relevant fix shipped.

CONFIDENCE: HIGH.

### D4.CUT.5 — Item 117 unified extract: filter-graph trim semantics

CLAIM: Item 117's `build_extraction_edl` uses `trim=start=S:end=E` and `atrim=start=S:end=E` (filter-graph operators on decoded frames), not container-seek (`-ss`/`-to`). Filter-graph `trim` cuts on decoded frame PTS without inclusive/exclusive ambiguity. This is the structural fix that eliminates the entire `-to`/`-t` family of bugs.

EVIDENCE: `render/edl_builder.py:218-225`:
```python
parts.append(
    f"[0:v]trim=start={_fmt(s)}:end={_fmt(e)},"
    f"setpts=PTS-STARTPTS[bv{i:02d}]"
)
parts.append(
    f"[0:a]atrim=start={_fmt(s)}:end={_fmt(e)},"
    f"asetpts=PTS-STARTPTS[ba{i:02d}]"
)
```

DRIFT MAGNITUDE: Backlog item 117 measurement (`:1726-1734`): bulletin A/V delta -0.01ms vs legacy -695.8ms. Per-shorts max |A/V| 0.67ms.

MITIGATION: This IS the mitigation for D4.CUT.1, D4.CUT.2, D4.CUT.3. Gated by `KAIZER_USE_V2_RAW_EXTRACT=1` (`stage_4_render.py:1672-1673`).

APPLICABILITY: ALL source video types per the Test 4 cross-video verification (backlog `:1732-1734`, HEVC 1080p 25fps). But: failure modes still exist (D4.CUT.6 below).

CONFIDENCE: HIGH for the architectural claim; MED on whether it covers EVERY edge case (only tested on 2 source videos).

### D4.CUT.6 — Item 117 unified extract: silent fallback to legacy path on failure

CLAIM: When the unified extract throws (NVENC failure, filter-graph too large, timeout, post-extract verification failure), `_render_impl` catches it and falls through to the legacy `cut_raw_*` path WITHOUT re-running with libx264. The job then ships with the legacy -33ms-per-segment drift signature, and the operator sees only a WARN line.

EVIDENCE: `stage_4_render.py:2881-2896`:
```python
try:
    self._run_unified_raw_extract(...)
except Exception as exc:
    logger.warning(
        "stage_4: unified raw-extract failed (falling back ...): %s", exc,
    )
    _p(f"  [v2-extract WARN] unified extract failed; falling back to legacy cut step: {exc}")
```

Backlog item 118 (`:1841-1849`): "Item 117 unified extract started, timed out at the configured 1800s, fell back to legacy cut step. ... Bulletin drift_measure on Job 53 output: -69.3ms global, -1733 ms video lost -- legacy drift signature, confirming the fallback path produced the broken output."

DRIFT MAGNITUDE: Reverts to the full legacy drift envelope (up to -695ms cumulative cut drift, per item 116).

MITIGATION: None. The fallback is by design ("don't take the render down").

APPLICABILITY: Any time NVENC enters a degraded state (driver instability per item 118), or the filter graph string exceeds some build's limit, or the 1800s timeout fires on a very long source.

CONFIDENCE: HIGH.

### D4.CUT.7 — Item 117 post-extract tolerance is 5ms; legacy gate is 35ms

CLAIM: Item 117 has stricter per-output verification (5ms) than item 112's per-clip verify (35ms). If a job runs legacy cut and produces 20ms per-clip drift, it passes the [cut summary] guardrail (cumulative ≤100ms across N≤5 segments), but if item 117 had run on the same input the same output would have failed verification and triggered the fallback. The verification is asymmetric.

EVIDENCE:
- Item 117: `stage_4_raw_extract.py:68` (`POST_EXTRACT_AV_TOLERANCE_MS: float = 5.0`).
- Legacy: `stage_4_render.py:594` (`PER_CLIP_AV_TOLERANCE_S: float = 0.035`) and `:603` (`CUMULATIVE_AV_WARN_MS: float = 100.0`).

DRIFT MAGNITUDE: Threshold-only; no drift introduction.

MITIGATION: N/A.

APPLICABILITY: Architectural observation; explains why legacy-fallback jobs can SHIP with drift the item 117 path would have rejected.

CONFIDENCE: HIGH.

---

## STAGE 4 — COMPOSE STEP (per-clip overlay)

### D4.COMP.1 — V1 `compose_bulletin_story` re-encodes audio (AAC frame rounding)

CLAIM: V1's `compose_bulletin_story` runs a filter_complex with multiple PNG inputs (sidebar, ticker, lower-third, bug) and re-encodes both video AND audio with `-r 30 -fps_mode cfr -async 1 -c:a aac -shortest`. AAC re-encode rounds the audio output UP to the next 1024-sample frame. The `-shortest` flag truncates to the shorter stream — but with looped PNG inputs producing infinite video and AAC rounding audio up, the audio reliably ends 7-18ms LATER than video, segment-by-segment.

EVIDENCE: `pipeline_core/longform_compose.py:530-551`:
```
"-c:v", "libx264", "-preset", "medium", "-crf", "20",
"-pix_fmt", "yuv420p",
"-r", "30", "-fps_mode", "cfr", "-async", "1",
"-c:a", "aac", "-b:a", "192k", "-ar", "48000",
"-shortest",
```
Backlog item 115 (`:1488-1494`): "every composed_story_NN.mp4 came out with audio 7-18ms LONGER than video. ... Over 26 segments this accumulated to **+256ms**."

DRIFT MAGNITUDE: +7-18ms per segment. CUMULATIVE: +256ms over 26 segments (job 49). Best case: 0ms (lucky boundary alignment). Worst case: +21ms (full AAC frame).

MITIGATION: `_align_composed_audio_to_video` runs after each compose (`stage_4_render.py:2585`, `:326-429`):
```python
"-af", f"atrim=end={v_dur_str},apad=whole_dur={v_dur_str},asetpts=PTS-STARTPTS"
```
This re-encodes audio with the trim+pad applied. Empirically reduces +7-18ms to <1ms per segment (backlog `:1538-1540`).

APPLICABILITY: ALL source video types when the compose path runs (which is ALWAYS for the bulletin — the unified extract only replaces the CUT step, compose still runs on raw_clip_NN.mp4 outputs).

CONFIDENCE: HIGH.

### D4.COMP.2 — `compose_pip_story` PiP source has its own `-ss/-t` cut

CLAIM: `compose_pip_story` adds a SECOND `-ss X -t Y -i pip_clip` input for the inset video (`longform_compose.py:633-637`). This is a third inner cut that can drift its own A/V — but since the PiP is *visual-only* and its audio is discarded (the main story's `0:a?` is the only audio mapped), PiP drift is V-only.

EVIDENCE: `longform_compose.py:633-637`:
```python
cmd += [
    "-ss", f"{pip_start_s:.3f}",
    "-t", f"{pip_duration_s:.3f}",
    "-i", pip_clip_path,
]
```
Main mapping: `:701` `"-map", "0:a?"`.

DRIFT MAGNITUDE: PiP-internal A/V drift is irrelevant to the main story's lip-sync. The PiP visual is timed by the main story's PTS via `overlay=...enable='lt(t,{pip_duration_s:.3f})'` (`:676`). Risk: 0ms for main lip-sync.

MITIGATION: N/A (drift impossible by structure).

APPLICABILITY: All.

CONFIDENCE: HIGH.

### D4.COMP.3 — `build_sidebar_carousel` and `build_fullscreen_takeover` carousels

CLAIM: Sidebar and takeover carousels are stitched from Ken-Burns segments via `_crossfade_chain` and concat. These produce *silent* video. Takeovers are then inserted between stories — the bulletin stitcher must pad audio with silence to match the takeover video length, which it does via the V1 stitcher's concat-demux behavior (audio gap is implicit silence in the concat'd output).

EVIDENCE: `pipeline_core/image_carousel.py:253-332` (`_build_carousel`). Takeover branch at `stage_4_render.py:2616-2644` then `_align_composed_audio_to_video(takeover_path)` at `:2642`.

DRIFT MAGNITUDE: Same AAC rounding as D4.COMP.1, mitigated by the same `_align_composed_audio_to_video` call.

MITIGATION: Item 115 aligned. APPLICABILITY: only when `effective_takeovers=True` AND `inter_story_boundary=True` (different parent_v2_index).

CONFIDENCE: HIGH.

### D4.COMP.4 — `_align_composed_audio_to_video` SAFETY-NET behaviour pre-pads silence

CLAIM: When upstream has produced a composed segment whose audio is *shorter* than video (which happened pre-item-116 due to the `-to` bug), `_align_composed_audio_to_video` uses `apad=whole_dur=V` to *pad with silence*. Post-item-116 the gap should be <1ms (AAC tail residue only), but the safety net can mask real lost audio from a future cut-step regression.

EVIDENCE: `stage_4_render.py:363-373`:
```python
MAX_EXPECTED_PAD_MS = 5.0
pre_pad_gap_ms = (v_dur - a_dur_before) * 1000.0
if pre_pad_gap_ms > MAX_EXPECTED_PAD_MS:
    logger.warning("stage_4: a/v alignment will pad %s with %.1fms ...")
```

DRIFT MAGNITUDE: Pad ≤33ms per pre-item-116 segment. Per item 116 (`:1656-1661`): "Item 115's apad=whole_dur=V then padded the 33ms gap with SILENCE to make composed_story durations match."

MITIGATION: WARN log if pad > 5ms. The pad itself is the *workaround*, not a fix — the upstream cut step must produce audio == video for this to be a no-op.

APPLICABILITY: All bulletin path. Failure mode IS the user-visible bug ("mouth-moving + silent audio" tail per segment).

CONFIDENCE: HIGH.

---

## STAGE 4 — STITCHER (V2 crossfade + V1 concat-demux)

The bulletin stitcher has two flavours:
1. V2 `bulletin_crossfade_stitcher.stitch_bulletin_with_crossfade` (3-pass, audio-crossfade) — `bulletin_crossfade_stitcher.py:266-527`
2. V1 `bulletin_stitcher.stitch_bulletin` (concat-demux, hard cuts) — `pipeline_core/bulletin_stitcher.py:136-308`

Selection: `stage_4_render.py:2682` — V2 used when `audio_overlap_s > 0 or video_overlap_s > 0`, i.e. when `transition_style ∈ {smart_cut, crossfade}`. Default is `smart_cut` (`:1619`), so V2 is the production path.

### D4.STITCH.1 — V2 Pass 1: video concat-demux can fail on codec drift

CLAIM: Pass 1 stream-copies video via concat demux (`-c:v copy -an`). If any one composed_story has slightly different codec params (e.g. a re-encode at a different `-cq`/`-crf`, mismatched SAR, different bit-depth), the demux silently produces a corrupted output. The V1 fallback re-encodes with `-r 30 -fps_mode cfr` (`bulletin_stitcher.py:266-275`) which itself can drift.

EVIDENCE: `bulletin_crossfade_stitcher.py:414-435`:
```python
cmd1 = [
    ffmpeg_bin, "-y",
    "-f", "concat", "-safe", "0",
    "-i", manifest_path,
    "-c:v", "copy", "-an",
    "-movflags", "+faststart",
    video_only_path,
]
```
NO fallback to re-encode on V2 Pass 1 (unlike V1 which does at `bulletin_stitcher.py:260-288`).

DRIFT MAGNITUDE: Variable. Concat-demux failure is usually full or zero — not partial drift. But silent corruption of frame timing can drop sub-second portions.

MITIGATION: None inside the V2 stitcher; V1 stitcher has a re-encode fallback.

APPLICABILITY: When `_align_composed_audio_to_video` re-encodes audio but not video — the video remains stream-copyable. Risk if a future change re-encodes one segment differently.

CONFIDENCE: MED.

### D4.STITCH.2 — V2 Pass 2: acrossfade chain pre-item-115-followup leaked AAC priming

CLAIM (HISTORICAL, FIXED): When acrossfade pulls from `[N:a]` directly, ffmpeg's AAC decoder emits encoder-priming samples (PTS -1024 / -21.33ms) + tail padding. Across 33 inputs on job 50 this leaked 350ms of phantom audio.

EVIDENCE: Backlog item 115 follow-up (`:1569-1631`). FIX at `bulletin_crossfade_stitcher.py:185-194`:
```python
for k, d in enumerate(durations):
    lab = f"n{k:03d}"
    parts.append(
        f"[{k}:a]atrim=0:{d:.6f},asetpts=PTS-STARTPTS[{lab}]"
    )
    norm_labels.append(lab)
```

DRIFT MAGNITUDE (HISTORICAL): +350ms over 33 segments = ~10ms/segment cumulative.

MITIGATION: atrim normalisers before every acrossfade input. APPLICABILITY: closed regression unless someone bypasses the normalisers.

CONFIDENCE: HIGH.

### D4.STITCH.3 — V2 Pass 2: acrossfade `d=overlap` consumes from BOTH neighbouring segments

CLAIM: acrossfade with `d=0.08` consumes 80ms from the END of segment K and 80ms from the START of segment K+1, blending them. The output total: `sum(durs) - (N-1) * overlap`. If audio_overlap_s > min(durations), segment K's content is fully consumed in the fade. Code guards against this at `bulletin_crossfade_stitcher.py:381-408` (V1 fallback when `any(d <= audio_overlap_s)`).

EVIDENCE: `bulletin_crossfade_stitcher.py:110-117`:
```python
for i, d in enumerate(durations):
    if d <= overlap_s:
        raise ValueError(
            f"segment[{i}] duration {d:.3f}s is <= overlap_s ..."
        )
```

DRIFT MAGNITUDE: When the V1 fallback triggers, the bulletin uses hard cuts (no crossfade) — that itself is fine for A/V sync. But it disables the audio click suppression.

MITIGATION: V1 fallback is automatic.

APPLICABILITY: Source videos with very short Stage-2 sub-cuts (<80ms). Unusual; usually <1.5s micro-fragments are dropped by `collapse_micro_fragments` first.

CONFIDENCE: HIGH (code path is explicit).

### D4.STITCH.4 — V2 Pass 3: `-c:v copy + -c:a aac -shortest` boundary

CLAIM: Pass 3 muxes the Pass 1 video and Pass 2 audio. Video is stream-copied; audio is re-encoded to AAC; `-shortest` truncates to the shorter stream. The audio re-encode allows `-shortest` to be sample-accurate. Per item 115 (`:1567`), this changed from `-c copy -shortest` (which could leak ≤21ms of audio past video EOF due to AAC packet atomicity) to the current re-encode form.

EVIDENCE: `bulletin_crossfade_stitcher.py:478-488`:
```python
cmd3 = [
    ffmpeg_bin, "-y",
    "-i", video_only_path,
    "-i", audio_only_path,
    "-map", "0:v", "-map", "1:a",
    "-c:v", "copy",
    "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
    "-shortest",
    "-movflags", "+faststart",
    output_path,
]
```

DRIFT MAGNITUDE: Residual ≤21ms at the END of the bulletin (one AAC frame). Down from pre-item-115's unbounded `-c copy` behaviour.

MITIGATION: Done. APPLICABILITY: All V2 jobs through this path.

CONFIDENCE: HIGH.

### D4.STITCH.5 — V1 fallback stitcher re-encode resets frame timing

CLAIM: When V1's concat-demux fails (stream-copy rc != 0), it retries with `-c:v libx264 -r 30 -fps_mode cfr -async 1 -c:a aac` over the *concat manifest*. This re-encode is line-by-line equivalent to a fresh cut step, with all the same AAC-frame rounding risks.

EVIDENCE: `pipeline_core/bulletin_stitcher.py:260-287`.

DRIFT MAGNITUDE: New AAC re-encode = +0 to +21ms at the file-end. Per-segment boundaries are concatenated by ffmpeg without further re-cuts, so the *inside-stitch* boundaries are clean.

MITIGATION: Only fires on codec drift. The warning `"concat fell back to re-encode — upstream slice codecs may have drifted"` (`:288`) surfaces.

APPLICABILITY: Unlikely on V2 (composed_story files all come from the same compose code path).

CONFIDENCE: HIGH.

---

## STAGE 4 — BULLETIN OVERLAY (item 117 phase 3)

### D4.OVL.1 — `apply_bulletin_overlays` enforces `-c:a copy` + sha256 verify

CLAIM: Item 117 phase 3 applies overlays with `-c:a copy` and verifies audio sha256 input==output. This is the architectural guarantee that audio is byte-identical past the overlay step.

EVIDENCE: `stage_4_bulletin_overlay.py:349-357`:
```python
cmd += [
    "-filter_complex", video_filter_complex,
    "-map", f"[{video_out_label}]",
    "-map", "0:a",
    *video_args,
    "-c:a", "copy",   # <- THE INVARIANT
```
And `:391-398` sha256 enforcement.

DRIFT MAGNITUDE: 0ms by sha256 verification.

MITIGATION: This IS the mitigation. Note: only fires when `KAIZER_USE_V2_RAW_EXTRACT=1` *and* the overlay step is invoked (item 117 phase 3+ wiring is opt-in).

APPLICABILITY: Item 117 path only. The legacy bulletin path uses `_v1_overlay_image_plan` (`pipeline_core/pipeline.py:1679-1741`) which re-encodes audio via `ENCODE_ARGS_INTERMEDIATE`.

CONFIDENCE: HIGH.

### D4.OVL.2 — V1 `overlay_image_plan` re-encodes audio via ENCODE_ARGS_INTERMEDIATE

CLAIM: V1's `overlay_image_plan` (used by default in `render_bulletin` step 6 at `stage_4_render.py:2733`) re-encodes audio with the full ENCODE_ARGS_INTERMEDIATE chain — meaning the final bulletin's audio is AAC-rounded one more time at the END.

EVIDENCE: `pipeline_core/pipeline.py:1730-1738`:
```python
cmd = (
    [FFMPEG_BIN, "-y"]
    + inputs
    + ["-filter_complex", fg, "-map", prev, "-map", "0:a?"]
    + ENCODE_ARGS_INTERMEDIATE
    + [out_path]
)
```

DRIFT MAGNITUDE: One more AAC frame rounding (≤21ms) at the END of the bulletin. Not cumulative per segment.

MITIGATION: None at this step in the legacy path. The trailing residue is bounded.

APPLICABILITY: Bulletin with image_plan entries. Item 117 phase 3 would supersede this; today the legacy path is the production path.

CONFIDENCE: HIGH.

---

## STAGE 4 — SHORTS OVERLAY (item 117 phase 4)

### D4.SHRT.1 — `apply_short_overlays` enforces `-c:a copy` + sha256

CLAIM: Symmetric with D4.OVL.1.

EVIDENCE: `stage_4_shorts_overlay.py:233-241`.

DRIFT MAGNITUDE: 0ms.

MITIGATION: Built-in. APPLICABILITY: Item 117 path only.

CONFIDENCE: HIGH.

### D4.SHRT.2 — V1-path `compose_shorts` re-encodes audio (no equivalent of item 115 align)

CLAIM: `compose_shorts` (`stage_4_render.py:2125-2282`) calls `_dispatch_compose` which invokes V1's `compose_clip` / `compose_clean_card` / `compose_follow_bar`. These re-encode audio with `ENCODE_ARGS_INTERMEDIATE`. There is NO `_align_composed_audio_to_video` call in the shorts path (only the bulletin path at `:2585`).

EVIDENCE: `stage_4_render.py:2239-2261`. Comparison to bulletin path at `:2585`.

DRIFT MAGNITUDE: +7-18ms per short. Shorts are typically standalone (no concat), so per-shorts drift is at-EOF only.

MITIGATION: None for shorts. The per-shorts drift surfaces only at the END of each 15-60s short — the user usually doesn't notice 18ms at the end of a viral clip.

APPLICABILITY: All V2 shorts production.

CONFIDENCE: HIGH.

---

## SILENCE-TRIM AND MICRO-FRAGMENT EXPANSION

### D4.SPLIT.1 — `apply_silence_trims_to_cuts` splits N cuts into M (M >> N)

CLAIM: `apply_silence_trims_to_cuts` walks each cut, finds silences > 1.5s inside, and SPLITS each cut into multiple sub-cuts AT EVERY SILENCE BOUNDARY. A single bulletin cut with 5 internal silences becomes 6 sub-cuts.

EVIDENCE: `stage_4_render.py:1354-1434`. Detection at `:1328-1351`. Threshold `SILENCE_TRIM_THRESHOLD_S = 1.5` (`:1325`). Default ON via `Stage4Render.silence_trim_threshold_s = SILENCE_TRIM_THRESHOLD_S` (`:1625`).

DRIFT MAGNITUDE: This is the multiplier for D4.COMP.1 / D4.CUT.3. Item 118 (`:1854-1858`): "the downstream silence-trim + micro-fragment-split path in the V1 chain split that 1 cut into 22 composed segments and re-introduced drift inside the legacy compose+stitcher chain."

If Stage 2 emits 1 bulletin cut covering 587.5s and silence-trim splits it into 22 sub-cuts → 22 composed_story files → 22 AAC-rounding events of +7-18ms = +154 to +396ms cumulative drift even with item 116's fix.

MITIGATION:
- Item 115's `_align_composed_audio_to_video` runs PER segment, so each segment's AAC residue is bounded.
- BUT item 117 unified extract operates ON THE STAGE-2 CUTS (`stage_4_render.py:1697-1699`), so when N=1 Stage-2 cut gets silence-split into M=22, the unified extract only sees 1 input range. The 22 sub-segments are produced DOWNSTREAM via the silence trim, AFTER unified extract.

Critical insight (backlog `:1852-1859`): "even if item 117 had succeeded on Job 53, it would NOT have fixed the lip-sync drift ... With only 1 bulletin cut, item 117's multi-cut alignment win doesn't apply."

Looking at `stage_4_render.py:3022-3098` (the splice + silence-trim + micro-fragment chain): silence-trim runs AFTER `splice_cuts_minus_skipped` and BEFORE `render_bulletin`. Then `render_bulletin` calls `cut_raw_bulletin_stories(full_video_cuts=spliced_cuts, ...)`. So the cut step receives the EXPANDED M sub-cuts, not the original N. Each sub-cut goes through `cut_clips_frame_aligned` → its own `raw_clip_NN.mp4` → its own compose → its own AAC rounding event.

APPLICABILITY: ALL bulletins where the source has any ≥1.5s inter-word gap (i.e. ~all real-world news content).

CONFIDENCE: HIGH (architectural).

### D4.SPLIT.2 — Item 117 unified extract uses ORIGINAL full_video_cuts, NOT silence-split sub-cuts

CLAIM: `_run_unified_raw_extract` reads `bulletin_ranges = [(c.start_sec, c.end_sec) for c in full_video_cuts]` (`stage_4_render.py:1697-1699`). The `full_video_cuts` here is the ORIGINAL Stage 2 output (`:2847`), before splice/silence/micro-fragment expansion. So the unified extract produces one raw_clip_NN.mp4 per Stage-2 cut — but later, `_render_impl` runs the SAME splice/silence/micro-fragment chain that the legacy path runs, producing M > N sub-cuts that DON'T have matching raw files in the unified-extract output.

EVIDENCE:
- `stage_4_render.py:2846-2847`:
```python
shorts_cuts = job_output.shorts_cuts
full_video_cuts = job_output.stage_two.full_video_cuts
```
- Unified extract at `:2876-2884` uses these RAW values.
- Splice + silence-trim + micro-fragment chain runs at `:3023-3118` and produces `spliced_cuts`.
- `render_bulletin(full_video_cuts=spliced_cuts, ...)` at `:3165` then cuts the *expanded* list.

This means item 117's unified extract is, at present, DEAD CODE for the bulletin path when Stage 2 emits cuts that span across silence regions. The unified-extract bulletin file would have N stories; the actual bulletin compose chain operates on M sub-cuts cut FROM THE MEZZANINE.

ACTUALLY — let me re-read `cut_raw_bulletin_stories`. `stage_4_render.py:1778-1834` shows it always calls `cut_clips_frame_aligned` (or V1 cut) against `self.video_path` (mezzanine). The unified-extract output (per_story mode, `raw_clip_NN.mp4` files) IS in `self.bulletin_dir`. The cache check at `:812` `if _os.path.exists(out_path) and _os.path.getsize(out_path) > 100_000` would *reuse* the unified-extract files IF the M sub-cuts' filenames `raw_clip_NN.mp4` align with the unified-extract's NN. They will NOT align when M ≠ N.

DRIFT MAGNITUDE: When silence-trim expands N→M, the unified extract's raw_clip_NN files (indexed 1..N) get OVERWRITTEN by sub-cut files (indexed 1..M) cut from mezzanine via legacy `cut_clips_frame_aligned`. The unified extract was essentially wasted work. The actual bulletin is built from legacy cut-step outputs WITH all the legacy drift sources (mitigated to ≤35ms per slice + ≤21ms AAC by items 112+116, but not perfect like item 117 promised).

MITIGATION: None. The unified-extract architecture is incomplete: it does not flow through the silence-trim chain.

APPLICABILITY: All source videos with internal silences ≥1.5s. Per item 118, this is essentially universal.

CONFIDENCE: HIGH for the file-flow analysis. MED on whether the cache-hit collision actually overwrites — depends on exact path equality between unified-extract output names and legacy cut output names. Both use `raw_clip_{i:02d}.mp4` in `bulletin_dir` (`stage_4_render.py:808` and `stage_4_raw_extract.py:169`), so the cache HIT path overwrites only when M == N.

### D4.SPLIT.3 — `collapse_micro_fragments` drops sub-cuts but renumbers indexes

CLAIM: After silence-trim splits N→M, `collapse_micro_fragments` drops sub-cuts <1.5s. Renumbering shifts the cut index counter. Downstream filename collisions are avoided by sequential numbering, but bookkeeping (parent_v2_index) needs to stay parallel.

EVIDENCE: `stage_4_render.py:1115-1194`.

DRIFT MAGNITUDE: None directly. Indirectly affects how many compose events fire.

MITIGATION: Per-parent guard at `:1170-1177` (keep longest if all drop).

APPLICABILITY: All. CONFIDENCE: HIGH.

---

## CROSS-CUTTING CONCERNS

### DC.1 — `-shortest` semantics on N-input filter graphs

CLAIM: `-shortest` truncates output to the shortest INPUT stream's EOF, but with `-loop 1 -i img.png` inputs the loop is infinite, so the shortest is whichever of (main story video, main story audio, encoder-rounded audio) ends first. For `compose_bulletin_story` the main video and audio enter at the same PTS=0 from the same input file, so `-shortest` lands within one AAC frame of the shorter stream.

EVIDENCE: `longform_compose.py:548` (`"-shortest"`). PNG inputs at `:476-481` are `-loop 1`.

DRIFT MAGNITUDE: ≤21ms (one AAC frame) at file EOF.

MITIGATION: D4.COMP.1's `_align_composed_audio_to_video` handles this.

APPLICABILITY: All compose calls. CONFIDENCE: HIGH.

### DC.2 — `-async 1` resync drifts audio to PTS

CLAIM: `-async 1` instructs ffmpeg to resync audio to the first video frame's PTS, then play through. If the video has CFR holes (which it shouldn't after `-vsync cfr -r 30`), `-async 1` would silently shift audio samples to compensate. Combined with `-vsync cfr` it should be a no-op on a clean mezzanine.

EVIDENCE: Multiple sites: `stage_4_render.py:712-713`, `longform_compose.py:545-546`, `bulletin_stitcher.py:272`.

DRIFT MAGNITUDE: Sub-frame on a clean CFR mezzanine. Up to a frame on a damaged input.

MITIGATION: Implicit (the CFR force is the precondition).

APPLICABILITY: All re-encodes. CONFIDENCE: MED.

### DC.3 — `setpts=PTS-STARTPTS` and `asetpts=PTS-STARTPTS` in filter graphs

CLAIM: Resets timestamps to start-from-zero after trim. Necessary when chaining trim → concat → encode; without it, ffmpeg sees gaps and either silently fills or refuses to mux.

EVIDENCE: `edl_builder.py:220`, `:224`, `:274`, `:278`. `bulletin_crossfade_stitcher.py:192`.

DRIFT MAGNITUDE: Architectural pre-requisite, no drift introduced when correctly used. MISUSE (forgetting `asetpts` on one branch) silently shifts audio.

MITIGATION: All current usages pair `trim`/`atrim` with the appropriate `setpts`. CONFIDENCE: HIGH.

### DC.4 — End-frame trim `-c copy` re-mux

CLAIM: `apply_end_frame_trim` (`stage_4_render.py:939-973`) trims the bulletin trailing slack with `ffmpeg -t T -c copy`. `-c copy` cannot split mid-packet, so audio may end ≤21ms (one AAC frame) shy of the requested target. The A/V invariant guard subtracts `tail_trim_s` from the expected sum (`:3334`), so a small residual is absorbed.

EVIDENCE: `stage_4_render.py:959-966`. Invariant at `:3332-3346`.

DRIFT MAGNITUDE: ≤21ms at the trimmed boundary.

MITIGATION: A/V invariant tolerance `AV_INVARIANT_TOLERANCE_S = 0.2` (`:467`).

APPLICABILITY: All bulletins where the trim-target is computed.

CONFIDENCE: HIGH.

### DC.5 — Pipeline-wide `+faststart` post-process

CLAIM: `-movflags +faststart` does not introduce A/V drift (it re-arranges atoms post-encode, preserving sample timing).

EVIDENCE: All ffmpeg sites cited above include `+faststart`.

DRIFT MAGNITUDE: 0ms. CONFIDENCE: HIGH.

### DC.6 — Compose cache is fingerprint-based but does NOT re-probe outputs

CLAIM: `_v1_compose_deps.is_fresh` decides reuse based on input file mtimes + a `composed_extra` dict (`stage_4_render.py:2516-2537`). It does NOT re-probe the cached output's A/V drift. A pre-fix composed file with the buggy `_align_composed_audio_to_video` outcome would be reused.

EVIDENCE: `stage_4_render.py:2540-2542`:
```python
if _v1_compose_deps.is_fresh(composed_path, composed_inputs, composed_extra):
    logger.info("stage_4: composed_story_%02d.mp4 cached", i)
    composed_ok = True
```
Cache-buster `"av_align_v": 1` (`:2536`) was added in item 115 to invalidate pre-fix outputs.

DRIFT MAGNITUDE: Inherits prior buggy state. Bounded by the cache-buster's monotonic version field.

MITIGATION: Bump `av_align_v` when the alignment algorithm changes. CONFIDENCE: HIGH.

---

## ANSWERS TO ADDITIONAL ANALYSIS QUESTIONS

### Q1: Which drift sources are CUMULATIVE vs CONSTANT?

CUMULATIVE (compound across N segments):
- D0.1 (VFR→CFR shifts compound where source has long VFR sections) — UNVERIFIED magnitude.
- D4.CUT.1 (V1 legacy AAC rounding: 0-32ms × N segments).
- D4.CUT.2 (HISTORICAL pre-item-116 `-to` bug: -33ms × N).
- D4.CUT.3 (residual AAC quant: ≤21ms × N).
- D4.COMP.1 (compose AAC rounding: 7-18ms × N segments).
- D4.SPLIT.1 (silence-trim multiplies N→M, then each of the above × M).

CONSTANT (fixed offset regardless of N):
- D4.STITCH.4 (Pass 3 `-shortest`: ≤21ms at file EOF only).
- D4.OVL.2 / D4.SHRT.2 (overlay re-encode: ≤21ms at file EOF only).
- DC.4 (end-frame trim: ≤21ms at trim boundary).

EVIDENCE: backlog item 112 measurements show 300-500ms cumulative for D4.CUT.1; item 115 shows +256ms for D4.COMP.1 across 26 segments. Both unambiguously cumulative.

CONFIDENCE: HIGH.

### Q2: Addressable via incremental fix vs require architectural change?

INCREMENTAL (already addressed or trivially patchable):
- D4.CUT.2 — fixed (item 116).
- D4.STITCH.2 — fixed (item 115 follow-up).
- D4.STITCH.4 — fixed (item 115).
- D4.OVL.1 / D4.SHRT.1 — built-in invariants.

ARCHITECTURAL (require structural change):
- D4.SPLIT.2 — item 117 unified extract operates on Stage-2 cuts, but bulletin compose runs on POST-silence-trim sub-cuts. The two paths don't compose. Fix requires moving silence-trim/micro-fragment expansion UPSTREAM of cut, OR running unified extract on the *post-expansion* sub-cuts, OR eliminating compose entirely.
- D4.SPLIT.1 — silence-trim's N→M expansion is the root multiplier. Architectural choice (per item 118): should silence-trim happen at Stage 2 (semantic) or Stage 4 (mechanical)?
- D4.SHRT.2 — shorts have no `_align_composed_audio_to_video`. Adding one would close the gap but architecturally shorts should follow item 117 phase 4 (`apply_short_overlays` with `-c:a copy`) instead.
- D4.CUT.6 — fallback path silently masks unified-extract failures with the broken legacy path. Architecturally the fallback should re-run with libx264 or fail loudly, not silently revert.

CONFIDENCE: HIGH.

### Q3: What does item 117's filter graph GUARANTEE that legacy cut does not?

The `build_extraction_edl` function at `render/edl_builder.py:151-302` guarantees:

1. **Frame-snapped boundaries before ffmpeg** — `_normalize_cuts` at `:121-142` snaps every (start, end) to the 1/30s grid, then drops degenerate entries. Legacy cut snaps too (`_snap_to_frame_grid` at `stage_4_render.py:606`) but the snapping is per-clip in a loop; the EDL builder normalizes the entire batch atomically and surfaces dropped cuts (`DroppedCut` at `edl_builder.py:88-95`).

2. **Single decode → single PTS clock** — all outputs come from `[0:v]` and `[0:a]` of ONE `-i mezzanine.mp4` (`stage_4_raw_extract.py:188`). The decoder produces frames once with one PTS sequence; trim then operates on those frames. Legacy cut re-decodes the source for each cut, and each cut's `-ss` seek can land on a different keyframe (potentially producing slightly different first-frame PTS).

3. **Filter-graph `trim`/`atrim` operates on decoded frame indices, not container packet boundaries** — there is no `-to` inclusive/exclusive ambiguity. The cut is performed AFTER decode, so the boundary is on raw PCM samples for audio and on decoded RGB frames for video. Legacy cut uses `-ss/-t` at container level, which is keyframe-bounded for video and sample-bounded for audio (different grids).

4. **Concat at filter-graph level** (`concat=n=N:v=1:a=0` and `concat=n=N:v=0:a=1` at `edl_builder.py:251-254`) — concatenation is in the decoded-frame domain, not via concat-demux at container level. No risk of codec-parameter drift between segments.

5. **Post-extract verification** — every output is ffprobed and asserted within 5ms (`stage_4_raw_extract.py:369-374`). Legacy verification is 35ms per clip and only warns at 100ms cumulative.

EVIDENCE: `edl_builder.py:151-302`, `stage_4_raw_extract.py:206-403`. Backlog item 117 (`:1726-1734`).

CONFIDENCE: HIGH.

### Q4: Silence-trim N→M split — where does it happen, and is it drift-neutral?

WHERE: `stage_4_render.py:3066-3098` (in `_render_impl`):
```python
if (self.silence_trim_threshold_s > 0
    and self.original_words
    and spliced_cuts):
    silence_trims = detect_silence_trims(
        list(self.original_words),
        threshold_s=self.silence_trim_threshold_s,
    )
    if silence_trims:
        ...
        spliced_cuts, parent_v2_indexes = apply_silence_trims_to_cuts(
            spliced_cuts, parent_v2_indexes, silence_trims,
        )
```

Helpers at `stage_4_render.py:1328-1351` (`detect_silence_trims`) and `:1354-1434` (`apply_silence_trims_to_cuts`).

IS IT DRIFT-NEUTRAL? NO — at the *semantic* (Stage-2-cuts) level it is drift-neutral (it operates on float seconds, no encode/decode). BUT downstream:

- Each of the M sub-cuts becomes a separate `raw_clip_NN.mp4` via `cut_clips_frame_aligned`, accumulating M instances of the cut-step residue (D4.CUT.3, ≤21ms each).
- Each becomes a separate `composed_story_NN.mp4` via `compose_bulletin_story`, accumulating M instances of the compose AAC rounding (D4.COMP.1, 7-18ms each — partly mitigated by `_align_composed_audio_to_video`).
- Each becomes a separate acrossfade input in the stitcher Pass 2, accumulating M-1 crossfade boundaries (each consumes 80ms of audio — bounded by the `compute_total_duration` formula, not drift-inducing by itself).

So silence-trim is a DRIFT MULTIPLIER: it does not introduce drift, but it scales the per-segment drift sources by M/N where M = total sub-cuts after split and N = original Stage-2 cuts.

EVIDENCE: Backlog item 118 (`:1854-1858`) makes this exact point.

CONFIDENCE: HIGH.

---

## EXECUTIVE SUMMARY OF FINDINGS

The pipeline has 17+ distinct drift introduction points across 8 stages. The single most important architectural finding is **D4.SPLIT.2**: item 117's unified-extract bypasses the cut-step drift class, but its outputs are *not* aligned to the downstream silence-trim + micro-fragment expansion. The result is that item 117 only "works" when Stage-2's cuts already match the final spliced-sub-cut set, which is rare in real-world news content with frequent inter-word silences. Item 118's observation that "even if item 117 had succeeded on Job 53, it would NOT have fixed the lip-sync drift" is precisely confirmed by reading the wiring.

The legacy path has ~5 cumulative drift sources (per cut, per compose, per stitch, per overlay, per end-trim). Items 112, 115, 115-followup, 116, and 117 closed 4 of them. The remaining structural source is **silence-trim acting downstream of cut decisions**: it inflates the segment count without informing the LLM, so item 117's promise of "one decode = one cut = no drift" never composes with the splice/silence expansion that produces the actual rendered segments. Either silence-trim must move to Stage 2 (semantic), the compose chain must be eliminated (one-pass overlay everything), or unified-extract must operate on POST-expansion sub-cuts (which loses item 117's audio bit-identity for the bulletin path because each sub-cut would then go through legacy compose).

Constant-offset sources (≤21ms at file EOF only) are bounded by AAC frame size and not a perception issue. Cumulative sources are what produce the user-visible 100ms+ lip-sync drift; addressing the multiplier (M) is the highest leverage architectural change.
