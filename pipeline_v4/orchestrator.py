"""V4 orchestrator — end-to-end job runner.

  source video  →  Step 1 (trim)  →  trimmed.mp4
                      ↓
                build canvas.json
                      ↓
                Step 2 (composite)  →  bulletin.mp4

Sub-process entry: ``python -m pipeline_v4.orchestrator --job-id N
--source <path> --output-dir <dir> --language te``.

The orchestrator also:
  - Builds the initial pool of images from the user's bulletin_images
    env (KAIZER_BULLETIN_IMAGES) or from cached _job_pool, if any
  - Asks Claude for per-story image timings (only if pool is non-empty)
  - Materialises canvas.json on disk so the editor can read/edit later
  - Updates the Job row's status + log via a thin DB adapter
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from pipeline_v4 import prompts as v4_prompts
from pipeline_v4 import trim_engine
from pipeline_v4 import canvas_engine
from pipeline_v4 import v1_bridge
from pipeline_v4 import image_provider
from pipeline_v4 import seo_provider
from pipeline_v4 import qc as v4_qc
from pipeline_v4.canvas_schema import (
    Canvas, CanvasImage, CanvasLayout, CanvasStory, CanvasTextBlock, V4JobCanvas,
    ShortConfig, CanvasSEO,
)


CANVAS_JSON_NAME = "canvas.json"


# ─── Image-pool ingest ──────────────────────────────────────────────

def _decode_pool_labels() -> dict[str, str]:
    """Decode ``KAIZER_BULLETIN_IMAGE_LABELS_B64`` — a base64 utf-8 JSON map
    ``{absolute_source_path: subject_label}`` the runner builds from each
    picked asset's ``UserAsset.description`` (name-tag contract). Base64 for
    the same reason as KAIZER_V4_PREDEFINED_DESCRIPTION: Telugu/Hindi text
    and Windows paths survive env passage without escaping shenanigans.
    Returns {} when unset or undecodable — labels then fall back to the
    filename stem."""
    raw = (os.environ.get("KAIZER_BULLETIN_IMAGE_LABELS_B64") or "").strip()
    if not raw:
        return {}
    try:
        import base64 as _b64
        data = json.loads(_b64.b64decode(raw).decode("utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if v}
    except Exception as exc:
        print(f"[v4/pool] failed to decode image labels (using filename stems): {exc}",
              flush=True)
    return {}


def _ingest_image_pool(output_dir: Path, env_pool: str) -> list[dict]:
    """Copy/link user-provided images (via KAIZER_BULLETIN_IMAGES env)
    into ``<output_dir>/_pool/`` and return a list of
    ``{filename, label, kind}`` describing the pool. Returns [] if
    no images are available — Step 2 still renders, just without
    carousel images."""
    pool_dir = output_dir / "_pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    if not env_pool:
        return out
    labels = _decode_pool_labels()
    seen: set[str] = set()
    for raw in (p.strip() for p in env_pool.split("|")):
        if not raw:
            continue
        src = Path(raw)
        if not src.is_file():
            continue
        filename = src.name
        # Avoid name collisions in the pool dir
        i = 0
        candidate = filename
        while candidate in seen or (pool_dir / candidate).exists():
            i += 1
            stem, suf = os.path.splitext(filename)
            candidate = f"{stem}_{i}{suf}"
        seen.add(candidate)
        try:
            shutil.copy2(src, pool_dir / candidate)
        except OSError as exc:
            print(f"[v4/pool] skip {src}: {exc}")
            continue
        label = (labels.get(raw) or labels.get(str(src)) or src.stem)[:120]
        out.append({"filename": candidate, "label": label, "kind": "photo"})
    return out


def _identity_words(words: list[dict], t0: float, t1: float) -> list[dict]:
    """Remap Deepgram words to STORY-RELATIVE time for timelines where
    output == source (source-preserved / audio-first modes — no cuts):
    keep words inside [t0, t1], subtract t0. Same {"w","s","e"} shape
    trim_engine stamps onto TrimmedStory.words."""
    out: list[dict] = []
    for w in (words or []):
        try:
            ws = float(w.get("s") or w.get("start") or 0.0)
            we = float(w.get("e") or w.get("end") or 0.0)
        except (TypeError, ValueError):
            continue
        if we < t0 or ws > t1:
            continue
        tok = (w.get("w") or w.get("word") or "").strip()
        if not tok:
            continue
        obj = {
            "w": tok,
            "s": round(max(ws, t0) - t0, 2),
            "e": round(max(min(we, t1) - t0, max(ws, t0) - t0), 2),
        }
        if w.get("spk") is not None:
            obj["spk"] = w["spk"]
        out.append(obj)
    return out


def _write_story_words_sidecar(output_dir: Path, trim_result, language: str = "") -> None:
    """Persist per-story word timestamps to ``<out_dir>/story_words.json`` so
    the editor's "Sync images to speech" endpoint can re-run the timing AI
    later without re-transcribing. Fail-soft: the sidecar is a convenience,
    never a render blocker."""
    try:
        payload = {
            "schema": 1,
            "language": language or os.environ.get("KAIZER_V4_LANGUAGE", ""),
            "stories": {
                str(s.story_index): list(getattr(s, "words", []) or [])
                for s in (trim_result.stories or [])
            },
        }
        (Path(output_dir) / "story_words.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8",
        )
        n = sum(len(v) for v in payload["stories"].values())
        print(f"[v4/step1] story_words.json written ({n} words, "
              f"{len(payload['stories'])} stories)", flush=True)
    except Exception as exc:
        print(f"[v4/step1] story_words.json write failed (soft-skip): {exc}",
              flush=True)


# ─── Initial canvas builder (called once after Step 1) ──────────────

def _decode_predefined_description() -> str:
    """Decode the operator-supplied bulletin description that the
    runner passed via ``KAIZER_V4_PREDEFINED_DESCRIPTION`` (base64 utf8).

    Returns "" when the env var isn't set or decoding fails — the
    orchestrator then falls back to the legacy AI flow. Base64 is used
    so newlines / quotes / Telugu / Hindi characters survive Windows
    env var passage without escaping shenanigans.
    """
    raw = (os.environ.get("KAIZER_V4_PREDEFINED_DESCRIPTION") or "").strip()
    if not raw:
        return ""
    try:
        import base64
        return base64.b64decode(raw).decode("utf-8").strip()
    except Exception as exc:
        print(f"[v4] failed to decode predefined description (skipping): {exc}",
              flush=True)
        return ""


def _ffprobe_duration(path: str) -> float:
    """Return the source video's duration in seconds via ffprobe.
    Falls back to 0.0 on any failure so the caller can detect the
    missing measurement and skip the source-preserved shortcut
    rather than producing a bogus all-spanning story."""
    try:
        import subprocess as _sp
        ff = shutil.which("ffprobe") or "ffprobe"
        proc = _sp.run(
            [ff, "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=10,
        )
        return float((proc.stdout or "0").strip() or 0.0)
    except Exception:
        return 0.0


def _build_source_preserved_trim_result(*, source_video: str, output_dir: Path,
                                         output_filename: str = "trimmed_bulletin.mp4"
                                         ) -> trim_engine.TrimResult:
    """Source-preserved mode entry point. Replaces ``trim_engine.run_step1``
    when the operator supplied a predefined description — keeps the
    bulletin AS-IS without re-encoding or running Claude.

    Strategy:
      1. Stream-copy the source into ``output_dir/output_filename``
         so downstream code that expects ``trimmed_bulletin.mp4`` at
         that path keeps working. Stream-copy (`-c copy`) avoids
         re-encoding so the operator's pixel-exact video is preserved.
      2. Still run Deepgram + the planner so shorts get carved at
         sensible story boundaries. The full transcript is mapped to
         ONE TrimmedStory spanning the entire video — no spans get
         cut, but story_index boundaries still come back for the
         shorts loop to split on.
    """
    dur = _ffprobe_duration(source_video)
    if dur <= 0.5:
        raise RuntimeError(
            f"source-preserved mode: ffprobe couldn't read a positive "
            f"duration from {source_video!r} -- aborting before the "
            f"shortcut produces an empty bulletin"
        )

    # 1) Stream-copy source -> trimmed_bulletin.mp4. ``+faststart``
    #    so playback starts immediately when served over HTTP.
    import subprocess as _sp
    out_path = output_dir / output_filename
    output_dir.mkdir(parents=True, exist_ok=True)
    ff = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ff, "-y", "-v", "error",
        "-i", source_video,
        "-c", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]
    proc = _sp.run(cmd, capture_output=True, text=True, timeout=60 * 10)
    if proc.returncode != 0:
        raise RuntimeError(
            f"source-preserved stream-copy failed (rc={proc.returncode}): "
            f"{(proc.stderr or '')[-800:]}"
        )

    # 2) Still get Deepgram words so the shorts loop can carve on
    #    story boundaries. Reuse trim_engine's internal helpers.
    audio_mp3 = str(output_dir / "_step1_audio.mp3")
    trim_engine._extract_audio_mp3(source_video, audio_mp3)
    try:
        # ``language="multi"`` matches the legacy default Deepgram
        # invocation — the orchestrator hands its language through.
        words, src_duration = trim_engine._deepgram_words(
            audio_mp3, language=os.environ.get("KAIZER_V4_LANGUAGE", "multi")
        )
    finally:
        try:
            os.unlink(audio_mp3)
        except OSError:
            pass

    # 3) Run the planner JUST for story boundaries — we'll ignore
    #    the CUT spans and treat every KEEP span as in-bulletin.
    from pipeline_v4 import content_type as _ct
    _ct.resolve_and_record(words=words, duration=src_duration,
                           out_dir=output_dir,
                           language=os.environ.get("KAIZER_V4_LANGUAGE", ""))
    planner_label, planner_fn = trim_engine._select_planner()
    print(f"[v4/step1] source-preserved mode -- planner={planner_label} "
          f"will be used ONLY for short story boundaries", flush=True)
    try:
        planner_stories, _ = planner_fn(
            words=words,
            language=os.environ.get("KAIZER_V4_LANGUAGE", "multi"),
            duration_sec=src_duration,
        )
    except Exception as exc:
        print(f"[v4/step1] planner soft-fail in source-preserved mode "
              f"(falling back to single full-duration story): {exc}",
              flush=True)
        planner_stories = []

    # 4) Convert planner output into TrimmedStory rows. Times are
    #    PASSED THROUGH unchanged because the trimmed video IS the
    #    source — output timeline = source timeline. If the planner
    #    returned nothing usable, fall back to one full-duration
    #    story so shorts get carved on duration alone.
    stories_out: list[trim_engine.TrimmedStory] = []
    if planner_stories:
        for s_idx, s in enumerate(planner_stories):
            story_spans: list[trim_engine.KeptSpan] = []
            for sp in (s.get("kept_spans") or []):
                try:
                    ss = float(sp.get("start_sec") or 0.0)
                    ee = float(sp.get("end_sec") or 0.0)
                except (TypeError, ValueError):
                    continue
                if ee <= ss + 0.05:
                    continue
                story_spans.append(trim_engine.KeptSpan(
                    start_sec=ss, end_sec=ee,
                    reason=str(sp.get("reason") or "")[:40],
                ))
            if not story_spans:
                continue
            # Collect verbatim transcript for the story so image
            # generation still gets the rich grounding.
            story_txt_words: list[str] = []
            for sp_obj in story_spans:
                for w in words:
                    ws = float(w.get("s") or w.get("start") or 0.0)
                    we = float(w.get("e") or w.get("end") or 0.0)
                    if we < sp_obj.start_sec or ws > sp_obj.end_sec:
                        continue
                    tok = (w.get("w") or w.get("word") or "").strip()
                    if tok:
                        story_txt_words.append(tok)
            transcript_text = " ".join(story_txt_words)[:1200]
            _t0 = story_spans[0].start_sec
            _t1 = story_spans[-1].end_sec
            stories_out.append(trim_engine.TrimmedStory(
                story_index=s_idx,
                title_native=str(s.get("title_native") or "")[:200],
                title_english=str(s.get("title_english") or "")[:200],
                summary=str(s.get("summary") or "")[:500],
                video_t_start=_t0,
                video_t_end=_t1,
                source_spans=story_spans,
                transcript_text=transcript_text,
                # Nothing is cut in source-preserved mode → identity remap.
                # Include EVERY word inside the story window (not just the
                # planner's spans — between-span speech still plays).
                words=_identity_words(words, _t0, _t1),
            ))
    if not stories_out:
        # Last-resort fallback: ONE story spanning the whole video.
        stories_out.append(trim_engine.TrimmedStory(
            story_index=0,
            title_native="",
            title_english="",
            summary="",
            video_t_start=0.0,
            video_t_end=src_duration,
            source_spans=[trim_engine.KeptSpan(
                start_sec=0.0, end_sec=src_duration, reason="full",
            )],
            transcript_text=" ".join(
                (w.get("w") or w.get("word") or "").strip() for w in words
            ).strip()[:1200],
            words=_identity_words(words, 0.0, src_duration),
        ))

    print(f"[v4/step1] source-preserved DONE -- "
          f"{src_duration:.1f}s preserved, {len(stories_out)} stories "
          f"identified for shorts (no CUTs applied)", flush=True)

    return trim_engine.TrimResult(
        trimmed_path=str(out_path),
        trimmed_duration_sec=src_duration,
        stories=stories_out,
        source_duration_sec=src_duration,
        removed_sec_total=0.0,
    )


def _mux_audio_master(*, narration: str, ref_video: Optional[str], audio_dur: float,
                      out_path: Path, layout_w: int = 1920, layout_h: int = 1080) -> None:
    """Build the audio-first ``trimmed_bulletin.mp4`` — a real A/V file whose
    AUDIO is the uploaded narration (the master track) and whose VIDEO is
    either the optional muted reference b-roll (looped/padded to the narration
    length) or a black filler (hidden behind the full-screen image layout).

    The whole audio-first design hinges on this: downstream Stage 2/3, the
    editor re-render, QC and shorts all read this single file as [0:v]+[0:a],
    so once it carries the narration as its audio, the narration is the master
    everywhere with ZERO changes to the composers. Length-locked with ``-t`` so
    the video track can't drift from the audio (a drift would blank the
    bulletin at the QC duration gate).
    """
    from pipeline_v4 import encoder as _encoder
    from pipeline_v4.ffmpeg_exec import run_ffmpeg as _run_ffmpeg

    ff = shutil.which("ffmpeg") or "ffmpeg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # veryfast: this is a plain mux/loop, not a quality-critical compose.
    venc = _encoder.video_encoder_args(crf=20, preset_hint="veryfast")
    dur = f"{max(0.1, audio_dur):.3f}"

    if ref_video:
        # Loop the muted reference b-roll to cover the narration. Keep its
        # NATIVE aspect (only round to even dims) so the bulletin panel sizing
        # (built from src_aspect = the ref's aspect) matches — same contract as
        # a normal V4 source video. No crop, no pad.
        vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=30,setsar=1"
        cmd = [
            ff, "-y", "-v", "error",
            "-stream_loop", "-1", "-i", ref_video,
            "-i", narration,
            "-map", "0:v:0", "-map", "1:a:0",
            "-t", dur,
            "-vf", vf,
            *venc, "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart",
            str(out_path),
        ]
    else:
        # No reference video → black filler (the full-screen image layout
        # covers it entirely; it exists only so the composers have a [0:v]).
        cmd = [
            ff, "-y", "-v", "error",
            "-f", "lavfi", "-i", f"color=c=black:s={layout_w}x{layout_h}:r=30",
            "-i", narration,
            "-map", "0:v:0", "-map", "1:a:0",
            "-t", dur,
            *venc, "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart",
            str(out_path),
        ]
    _run_ffmpeg(cmd, timeout=max(180, int(audio_dur) + 180),
                log_label="audio_first_mux", retry=1)


def _build_audio_driven_trim_result(*, narration_audio: str, ref_video: Optional[str],
                                    output_dir: Path, language: str = "multi",
                                    output_filename: str = "trimmed_bulletin.mp4",
                                    layout_w: int = 1920, layout_h: int = 1080
                                    ) -> trim_engine.TrimResult:
    """Audio-first Stage 1. The uploaded AUDIO is the master: transcribe it
    with Deepgram, run the SAME KEEP/CUT planner purely for story boundaries +
    titles/summaries (nothing is cut), then mux the narration onto the muted
    reference b-roll (or a black filler) to produce a normal
    ``trimmed_bulletin.mp4`` that the rest of V4 consumes unchanged.

    Returns the same ``trim_engine.TrimResult`` shape as ``run_step1`` /
    ``_build_source_preserved_trim_result`` so run_job needs no structural
    change beyond the Stage-1 branch selection.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lang = (language or "multi").strip() or "multi"

    # 1) Transcribe the NARRATION (not any video). Normalise to mp3 first so
    #    Deepgram gets a small payload; _extract_audio_mp3's "-vn" is a
    #    harmless no-op on an audio-only input.
    audio_mp3 = str(output_dir / "_step1_narration.mp3")
    trim_engine._extract_audio_mp3(narration_audio, audio_mp3)
    try:
        words, audio_dur = trim_engine._deepgram_words(audio_mp3, language=lang)
    finally:
        try:
            os.unlink(audio_mp3)
        except OSError:
            pass
    # Deepgram metadata.duration can come back 0 — fall back to a direct
    # ffprobe of the narration so the timeline is never zero.
    if audio_dur <= 0.5:
        audio_dur = _ffprobe_duration(narration_audio)
    if audio_dur <= 0.5:
        raise RuntimeError(
            f"audio-first: could not read a positive duration from the "
            f"narration {narration_audio!r} — aborting before an empty render"
        )

    # 2) Story boundaries via the SAME planner (Claude/Gemini) — used ONLY for
    #    titles/summaries/segmentation; no spans are cut.
    from pipeline_v4 import content_type as _ct
    _ct.resolve_and_record(words=words, duration=audio_dur,
                           out_dir=output_dir, language=lang)
    planner_label, planner_fn = trim_engine._select_planner()
    print(f"[v4/step1] AUDIO-FIRST mode -- planner={planner_label} for story "
          f"boundaries over {audio_dur:.1f}s of narration", flush=True)
    try:
        planner_stories, _ = planner_fn(
            words=words, language=lang, duration_sec=audio_dur,
        )
    except Exception as exc:
        print(f"[v4/step1] planner soft-fail in audio-first mode "
              f"(falling back to single full-duration story): {exc}", flush=True)
        planner_stories = []

    # Length-lock the mux to the TRUE narration length. Deepgram's
    # metadata.duration is measured on the mp3 transcode and can drift by a
    # fraction of a second (LAME padding / VBR), which would truncate or pad the
    # MASTER track. ffprobe the original narration directly; keep the Deepgram
    # value only for the planner's word timeline.
    timeline_dur = _ffprobe_duration(narration_audio)
    if timeline_dur <= 0.5:
        timeline_dur = audio_dur

    # 3) Build CONTIGUOUS stories that TILE [0, timeline_dur] with NO gaps.
    #    CRITICAL: the bulletin composer (v1_bridge.render_bulletin) slices each
    #    story's [video_t_start, video_t_end] out of trimmed_bulletin.mp4 and
    #    concatenates them. If stories had gaps (the KEEP/CUT planner leaves
    #    them by design), the narration in those gaps would be silently DROPPED
    #    — but audio-first must keep the FULL narration. So we use the planner
    #    ONLY for topic boundaries + titles/summaries, and make each story a
    #    contiguous segment from its start to the NEXT story's start. Every
    #    second of narration then belongs to exactly one story; sum of ranges
    #    == timeline_dur, so nothing is cut and QC's duration gate stays honest.
    planned: list[dict] = []
    for s in (planner_stories or []):
        starts = []
        for sp in (s.get("kept_spans") or []):
            try:
                starts.append(float(sp.get("start_sec") or 0.0))
            except (TypeError, ValueError):
                pass
        if not starts:
            continue
        seg_start = min(starts)
        if seg_start < 0.0 or seg_start >= timeline_dur:
            continue
        planned.append({
            "start": seg_start,
            "title_native": str(s.get("title_native") or "")[:200],
            "title_english": str(s.get("title_english") or "")[:200],
            "summary": str(s.get("summary") or "")[:500],
        })
    planned.sort(key=lambda p: p["start"])
    # Drop near-duplicate starts (avoid <1s slivers that would break slicing).
    deduped: list[dict] = []
    for p in planned:
        if deduped and (p["start"] - deduped[-1]["start"]) < 1.0:
            continue
        deduped.append(p)
    planned = deduped

    def _seg_transcript(a: float, b: float) -> str:
        toks = []
        for w in words:
            ws = float(w.get("s") or w.get("start") or 0.0)
            if a <= ws < b:
                t = (w.get("w") or w.get("word") or "").strip()
                if t:
                    toks.append(t)
        return " ".join(toks)[:1200]

    stories_out: list[trim_engine.TrimmedStory] = []
    if planned:
        planned[0]["start"] = 0.0   # first segment always covers any pre-roll
        for i, p in enumerate(planned):
            seg_start = p["start"]
            seg_end = planned[i + 1]["start"] if (i + 1 < len(planned)) else timeline_dur
            if seg_end <= seg_start + 0.05:
                continue
            stories_out.append(trim_engine.TrimmedStory(
                story_index=len(stories_out),
                title_native=p["title_native"],
                title_english=p["title_english"],
                summary=p["summary"],
                video_t_start=seg_start,
                video_t_end=seg_end,
                source_spans=[trim_engine.KeptSpan(
                    start_sec=seg_start, end_sec=seg_end, reason="audio")],
                transcript_text=_seg_transcript(seg_start, seg_end),
                # Narration timeline == output timeline (nothing cut).
                words=_identity_words(words, seg_start, seg_end),
            ))
    if not stories_out:
        # Fallback: ONE story spanning the whole narration.
        stories_out.append(trim_engine.TrimmedStory(
            story_index=0,
            title_native="", title_english="", summary="",
            video_t_start=0.0, video_t_end=timeline_dur,
            source_spans=[trim_engine.KeptSpan(
                start_sec=0.0, end_sec=timeline_dur, reason="full")],
            transcript_text=_seg_transcript(0.0, timeline_dur),
            words=_identity_words(words, 0.0, timeline_dur),
        ))

    # 4) Produce the muxed A/V trimmed_bulletin.mp4 (narration = master audio),
    #    length-locked to the TRUE narration length.
    out_path = output_dir / output_filename
    _mux_audio_master(
        narration=narration_audio, ref_video=ref_video, audio_dur=timeline_dur,
        out_path=out_path, layout_w=layout_w, layout_h=layout_h,
    )

    print(f"[v4/step1] AUDIO-FIRST DONE -- {timeline_dur:.1f}s narration, "
          f"{len(stories_out)} contiguous stories, "
          f"ref_video={'yes' if ref_video else 'no (fullscreen images)'}", flush=True)

    return trim_engine.TrimResult(
        trimmed_path=str(out_path),
        trimmed_duration_sec=timeline_dur,
        stories=stories_out,
        source_duration_sec=timeline_dur,
        removed_sec_total=0.0,
    )


def _probe_source_aspect(source_video: str) -> float:
    """Return the source video's width/height aspect ratio.

    Falls back to 16:9 (≈1.78) when ffprobe is missing or the file
    can't be parsed — keeps Stage 1 alive on a probe failure rather
    than crashing the whole job. The aspect drives _bulletin_layout's
    inner video panel sizing so a vertical phone-shot source doesn't
    get cover-crop-zoomed into 16:9 chrome (the original bug: outer
    layout stays 1920×1080, but the inner video frame adapts to
    whatever shape the operator uploaded).
    """
    try:
        import subprocess as _sp
        ff = shutil.which("ffprobe") or "ffprobe"
        proc = _sp.run(
            [ff, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=s=x:p=0", source_video],
            capture_output=True, text=True, timeout=10,
        )
        out = (proc.stdout or "").strip()
        if "x" in out:
            w_s, h_s = out.split("x", 1)
            w, h = int(w_s.strip()), int(h_s.strip())
            if w > 0 and h > 0:
                return float(w) / float(h)
    except Exception as exc:
        print(f"[v4] ffprobe source aspect failed (defaulting to 16:9): {exc}", flush=True)
    return 16.0 / 9.0


def _bulletin_layout(width: int = 1920, height: int = 1080,
                     brand_logo: Optional[str] = None,
                     source_aspect: float = 16.0 / 9.0) -> CanvasLayout:
    """News-channel bulletin layout. Outer canvas stays a standard 1920×
    1080 broadcast frame; INNER video panel is sized to match the
    source video's aspect ratio so vertical / square sources don't
    get cover-crop-zoomed into 16:9 chrome.

    Two regimes (decided by ``source_aspect = source_w / source_h``):

    1. WIDE source (>= 1.2, i.e. landscape) → the V1-parity broadcast
       layout: wide video left, narrow image sidebar right.

         ┌───────────────────────────┬───────────┐
         │  VIDEO (left ~57%)        │  IMAGE 25%│  top 76%
         ├───────────────────────────┴───────────┤
         │  RED strap: BREAKING + headline       │  11%
         ├───────────────────────────────────────┤
         │  YELLOW ticker: scrolling summary     │  10%
         └───────────────────────────────────────┘

    2. NARROW source (< 1.2, i.e. vertical / square — phone shots) →
       narrow video panel centered-left, wide image panel right so
       the image carousel + chyrons get more pixels:

         ┌─────────┬─────────────────────────────┐
         │ VIDEO   │  IMAGE (wide sidebar)       │
         │ (~22%)  │                             │
         ├─────────┴─────────────────────────────┤
         │  RED strap                            │
         ├───────────────────────────────────────┤
         │  YELLOW ticker                        │
         └───────────────────────────────────────┘

    The video panel's height stays at ~60% of canvas height in both
    regimes; only its WIDTH adapts to source aspect. That means a
    vertical 9:16 source renders crisp at native aspect (no crop, no
    zoom, no letterbox) inside the broadcast frame.
    """
    # Wide source → use the V1-tuned defaults the operator already
    # likes (matches studio-bg geometry from 2026-06-05).
    if source_aspect >= 1.2:
        return CanvasLayout(
            width=width, height=height, bg_color="#000000",
            video_x_pct=6.756, video_y_pct=15.311,
            video_w_pct=56.664, video_h_pct=60.673,
            picture_x_pct=70.888, picture_y_pct=16.578,
            picture_w_pct=24.698, picture_h_pct=58.863,
            brand_logo_path=brand_logo,
            brand_logo_x_pct=92.0, brand_logo_y_pct=2.0, brand_logo_w_pct=6.0,
            lt_ease="out_cubic",
        )

    # Narrow source — compute the video panel's width from the actual
    # source aspect so the inner video is rendered at its NATIVE
    # ratio (no cover-crop, no letterbox). Math:
    #   panel_height_px = video_h_pct * canvas_h / 100
    #   panel_width_px  = panel_height_px * source_aspect
    #   video_w_pct     = panel_width_px / canvas_w * 100
    video_h_pct = 60.673
    video_x_pct = 6.756
    video_y_pct = 15.311
    panel_h_px  = video_h_pct * height / 100.0
    panel_w_px  = max(80.0, panel_h_px * source_aspect)
    video_w_pct = min(40.0, panel_w_px / width * 100.0)  # cap at 40% to leave room

    # Picture panel takes the rest of the right side with a small gap.
    gap_pct = 3.0
    picture_x_pct = video_x_pct + video_w_pct + gap_pct
    right_margin_pct = 4.5
    picture_w_pct = max(20.0, 100.0 - picture_x_pct - right_margin_pct)
    return CanvasLayout(
        width=width, height=height, bg_color="#000000",
        video_x_pct=video_x_pct, video_y_pct=video_y_pct,
        video_w_pct=video_w_pct, video_h_pct=video_h_pct,
        picture_x_pct=picture_x_pct, picture_y_pct=16.578,
        picture_w_pct=picture_w_pct, picture_h_pct=58.863,
        brand_logo_path=brand_logo,
        brand_logo_x_pct=92.0, brand_logo_y_pct=2.0, brand_logo_w_pct=6.0,
        lt_ease="out_cubic",
    )


def _short_layout(width: int = 1080, height: int = 1920,
                  brand_logo: Optional[str] = None) -> CanvasLayout:
    """9:16 vertical Short (1080x1920):

      ┌─────────────────────┐
      │ VIDEO  (100% x 55%) │   talking-head, cover-crop
      ├─────────────────────┤
      │ IMAGE  (100% x 24%) │   contextual image
      ├─────────────────────┤
      │ RED strap (11%)     │   headline
      ├─────────────────────┤
      │ YELLOW ticker (10%) │   summary
      └─────────────────────┘

    Stacks vertically so every pixel is filled — no black gutters."""
    return CanvasLayout(
        width=width, height=height, bg_color="#000000",
        video_x_pct=0.0, video_y_pct=0.0, video_w_pct=100.0, video_h_pct=55.0,
        # Image directly under the video, full-width, 24% tall.
        picture_x_pct=0.0, picture_y_pct=55.0,
        picture_w_pct=100.0, picture_h_pct=24.0,
        brand_logo_path=brand_logo,
        brand_logo_x_pct=82.0, brand_logo_y_pct=2.5, brand_logo_w_pct=14.0,
    )


def _audio_fullscreen_layout(width: int = 1920, height: int = 1080,
                             brand_logo: Optional[str] = None) -> CanvasLayout:
    """Audio-first, NO reference video: the generated image fills the ENTIRE
    frame (a 16:9 image over the 16:9 canvas = perfect cover) while the
    narration plays. The muted black filler [0:v] sits fully behind the image.
    The red headline strap + yellow ticker still overlay at the bottom (drawn
    after the image), so it reads as a full-screen news explainer.
    """
    return CanvasLayout(
        width=width, height=height, bg_color="#000000",
        # Filler video fills the frame (hidden behind the image below).
        video_x_pct=0.0, video_y_pct=0.0, video_w_pct=100.0, video_h_pct=100.0,
        # Image full-frame on top.
        picture_x_pct=0.0, picture_y_pct=0.0, picture_w_pct=100.0, picture_h_pct=100.0,
        # Black (not the default white) tile border → the 3px frame is invisible
        # on the black canvas, so the image reads as true full-bleed.
        tile_border_color="#000000",
        brand_logo_path=brand_logo,
        brand_logo_x_pct=90.0, brand_logo_y_pct=2.5, brand_logo_w_pct=8.0,
        lt_ease="out_cubic",
    )


def _default_image_timings(story_duration: float, pool_count: int) -> list[tuple[int, float, float]]:
    """Heuristic fallback when the timing AI doesn't decide. Returns
    [(pool_index, t_start, t_end), ...].

    The OLD version cut a fixed 4s window in a while-loop, so ONE image on
    a 96s story became 24 identical windows → a 24-LAYER carousel rendered
    frame-by-frame (job 608: ~43min for 4 stories). Now the window count
    scales with the POOL: one image = one window; more images cycle at a
    readable ~10s dwell, each shown at most ~twice. No frantic slideshow,
    no redundant layers. KAIZER_V4_IMG_DWELL_S tunes the dwell."""
    if pool_count == 0 or story_duration <= 0:
        return []
    if pool_count == 1:
        # One image = one window covering the story (the prior behaviour
        # was 24 contiguous windows of it — same look, 24× the render).
        return [(0, 0.0, round(float(story_duration), 3))]
    try:
        dwell = max(4.0, float(os.environ.get("KAIZER_V4_IMG_DWELL_S", "10") or 10))
    except (TypeError, ValueError):
        dwell = 10.0
    # each image shown once..twice; never fewer than the pool (show them
    # all), never more than 2 full cycles.
    n = max(pool_count, min(int(round(story_duration / dwell)), pool_count * 2))
    window = story_duration / n
    out: list[tuple[int, float, float]] = []
    for i in range(n):
        t = i * window
        end = float(story_duration) if i == n - 1 else (i + 1) * window
        out.append((i % pool_count, round(t, 3), round(end, 3)))
    return out


# ISO -> default bold font basename. Matches resources/fonts/.
_LANG_TO_FONT = {
    "te": "NotoSansTelugu-Bold.ttf",
    "hi": "NotoSansDevanagari-Bold.ttf",
    "ta": "NotoSansTamil-Bold.ttf",
    "kn": "NotoSansKannada-Bold.ttf",
    "ml": "NotoSansMalayalam-Bold.ttf",
    "bn": "NotoSansBengali-Bold.ttf",
    "mr": "NotoSansDevanagari-Bold.ttf",
    "gu": "NotoSansGujarati-Bold.ttf",
}


def _build_initial_canvas(
    *,
    trim_result: trim_engine.TrimResult,
    pool: list[dict],
    kind: str,
    output_filename: str,
    brand_logo: Optional[str] = None,
    language: str = "te",
    # Width/height ratio of the source video. Used to size the inner
    # video panel so vertical / square phone-shot sources don't get
    # cover-crop-zoomed into 16:9 chrome. Defaults to 16:9 for back-
    # compat with callers that haven't been updated to probe yet.
    source_aspect: float = 16.0 / 9.0,
    # Audio-first with NO reference video → the image fills the whole frame
    # (the muted filler [0:v] sits hidden behind it).
    fullscreen_image: bool = False,
) -> Canvas:
    if kind == "bulletin":
        if fullscreen_image:
            layout = _audio_fullscreen_layout(brand_logo=brand_logo)
        else:
            layout = _bulletin_layout(brand_logo=brand_logo, source_aspect=source_aspect)
        # Designed layout from the library ("lib:<key>" in the wizard's
        # full-form picker): override the tile geometry with the design's
        # zone percentages. Unknown/non-renderable key → default layout
        # (fail-soft; the job still renders).
        _ffl = (os.environ.get("KAIZER_V4_FULLFORM_LAYOUT", "") or "").strip().lower()
        if _ffl.startswith("lib:") and not fullscreen_image:
            try:
                from pipeline_v4.layout_library import to_canvas_pcts
                _pcts = to_canvas_pcts(_ffl[4:])
                if _pcts:
                    for _k, _v in _pcts.items():
                        setattr(layout, _k, float(_v))
                    print(f"[v4] designed layout applied: {_ffl[4:]}",
                          flush=True)
            except Exception as _lx:
                print(f"[v4] designed layout skipped ({_lx})", flush=True)
        # THEME PACK: stamp the job's theme onto the canvas so the render
        # AND every editor re-render wear it ("" = classic, byte-identical).
        _thm_env = (os.environ.get("KAIZER_V4_THEME") or "").strip().lower()
        if _thm_env:
            layout.theme = _thm_env
        # Studio-background video the operator picked in the new-job
        # wizard (forwarded via env so the orchestrator subprocess can
        # see it without a DB lookup). Only stamped on the bulletin —
        # shorts use V1's compose path which doesn't read this field.
        _bg_path = (os.environ.get("KAIZER_V4_BG_VIDEO_PATH") or "").strip()
        if _bg_path:
            layout.bg_video_path = _bg_path
            try:
                layout.bg_video_volume = float(os.environ.get("KAIZER_V4_BG_VIDEO_VOLUME", "0.0"))
            except ValueError:
                layout.bg_video_volume = 0.0
            try:
                layout.bg_intro_seconds = float(os.environ.get("KAIZER_V4_BG_INTRO_SECONDS", "0.0"))
            except ValueError:
                layout.bg_intro_seconds = 0.0
    else:
        layout = _short_layout(brand_logo=brand_logo)

    # Layout safety (spec 3.13): generated bulletin layouts are clamped so
    # media can never start life overlapping the headline/ticker strips.
    if kind == "bulletin":
        try:
            from pipeline_v4 import layout_safety
            if layout_safety.clamp_media(layout):
                print("[v4] layout-safety: clamped media tiles out of the "
                      "text strip", flush=True)
        except Exception as _ls_exc:
            print(f"[v4] layout-safety clamp skipped: {_ls_exc}", flush=True)

    stories: list[CanvasStory] = []
    for s in trim_result.stories:
        # Image timings — default heuristic; the editor can let Claude
        # re-decide later by toggling claude_decided_timings.
        #
        # Per-story scoping: an AI-generated image carries a
        # ``story_index`` field bound at generation time. We FILTER
        # the pool to images whose story_index matches THIS story —
        # without this, every story cycled through every image
        # ("topic 1 image appeared in topic 3", etc.). User-uploaded
        # images don't carry a story_index — they're treated as
        # shared B-roll and appear in every story (back-compat with
        # the prior behaviour where a hand-curated upload set was
        # expected to be a generic B-roll bank).
        own_pool = [
            (i, p) for i, p in enumerate(pool)
            if p.get("story_index") == s.story_index
        ]
        shared_pool = [
            (i, p) for i, p in enumerate(pool)
            if p.get("story_index") is None
        ]
        # Prefer own_pool when this story produced its own image;
        # fall back to shared B-roll only when the story has nothing
        # of its own. Keeps the "topic 1 image stays in topic 1" rule
        # while still rendering something useful if the AI fetch
        # failed for this story specifically.
        story_pool = own_pool if own_pool else shared_pool

        # Transcript-grounded timing (Phase-1 engine): the timing AI
        # places each image on the exact spoken words, matching by the
        # pool labels (name-tag contract). Gaps are expected — the
        # renderer cuts back to the main video. ANY failure/empty plan
        # → the legacy 4s heuristic below, so a paid job never fails
        # because of timing.
        images: list[CanvasImage] = []
        plan = None
        if story_pool:
            try:
                from pipeline_v4 import image_timing
                plan = image_timing.decide_story_timings(
                    title_native=s.title_native,
                    title_english=s.title_english,
                    summary=s.summary,
                    duration=s.duration,
                    words=list(getattr(s, "words", None) or []),
                    pool=[{"label": p.get("label"), "kind": p.get("kind")}
                          for _, p in story_pool],
                    language=language,
                )
            except Exception as exc:
                print(f"[v4] image timing failed for story {s.story_index + 1} "
                      f"({exc}) — heuristic fallback", flush=True)
                plan = None
        if plan:
            for e in plan:
                _, p = story_pool[e["pool_index"]]
                dwell = float(e["t_end"]) - float(e["t_start"])
                # A VIDEO pool entry (.mp4/.mov/…) is a full-screen
                # reference-video CUTAWAY (B-roll), placed by the same
                # label↔speech engine as images, then always shown
                # full-screen for its window (never a sidebar still).
                _vid = str(p.get("filename") or "").lower().endswith(
                    (".mp4", ".mov", ".webm", ".m4v", ".mkv"))
                images.append(CanvasImage(
                    src=p["filename"],
                    t_start=e["t_start"], t_end=e["t_end"],
                    source="manual" if p.get("kind") == "user" else "claude",
                    label=p.get("label"),
                    # Polish C motion: long dwells get a gentle Ken-Burns
                    # zoom instead of sitting static; short ones keep the
                    # crossfade.
                    effect="zoom_in" if dwell >= 4.0 else "fade",
                    timing_mode="content",
                    confidence=e.get("confidence"),
                    importance=e.get("importance"),
                    spotlight=("fullscreen" if _vid else e.get("spotlight")),
                    matched_text=e.get("matched_text") or None,
                    media_kind=("video" if _vid else "image"),
                    # Face-aware framing: the pool ingest stamped a detected
                    # focal point; absent → schema default 50/50 (centered).
                    offset_x_pct=float(p.get("offset_x_pct") or 50.0),
                    offset_y_pct=float(p.get("offset_y_pct") or 50.0),
                ))
        else:
            timings = _default_image_timings(s.duration, len(story_pool))
            def _mk_img(local_idx, ts, te):
                _pe = story_pool[local_idx][1]
                _fn = _pe["filename"]
                _vid = str(_fn or "").lower().endswith(
                    (".mp4", ".mov", ".webm", ".m4v", ".mkv"))
                return CanvasImage(
                    src=_fn, t_start=ts, t_end=te,
                    source="manual" if _pe.get("kind") == "user" else "claude",
                    label=_pe.get("label"),
                    spotlight=("fullscreen" if _vid else None),
                    media_kind=("video" if _vid else "image"),
                    # Face-aware framing from the pool ingest (see the
                    # timing-plan branch above) — same fill-when-present rule.
                    offset_x_pct=float(_pe.get("offset_x_pct") or 50.0),
                    offset_y_pct=float(_pe.get("offset_y_pct") or 50.0),
                )
            images = [_mk_img(local_idx, ts, te) for (local_idx, ts, te) in timings]
        text_blocks = []
        if s.title_native:
            # BREAKING headline — red strap below the video+image row.
            text_blocks.append(CanvasTextBlock(
                kind="lower_third",
                text=s.title_native,
                t_start=0.0, t_end=None,
            ))
        # Yellow ticker at the very bottom — uses the summary if Claude
        # provided one; otherwise repeats the headline so the bottom
        # 10% strip isn't blank.
        ticker_text = (s.summary or s.title_english or s.title_native or "").strip()
        if ticker_text:
            text_blocks.append(CanvasTextBlock(
                kind="ticker",
                text=ticker_text,
                t_start=0.0, t_end=None,
            ))
        stories.append(CanvasStory(
            story_index=s.story_index,
            video_t_start=s.video_t_start,
            video_t_end=s.video_t_end,
            title_native=s.title_native,
            title_english=s.title_english,
            summary=s.summary,
            images=images,
            text_blocks=text_blocks,
            claude_decided_timings=True,
            # Polish A: the name-strap only makes sense when the timing
            # engine placed the images (labels = real subjects). Heuristic
            # fallbacks keep it off — labels may be filename stems.
            name_strap=bool(plan),
        ))

    # Default short_config — V1's torn_card with the right script font
    # and the first pool image as hero. Editor can replace any of it.
    short_config: Optional[ShortConfig] = None
    if kind == "short":
        first_image = pool[0]["filename"] if pool else None
        primary_story = stories[0] if stories else None
        # Honour the operator's selected short template instead of always
        # torn_card. Forwarded from runner via KAIZER_V4_SHORT_LAYOUT; falls
        # back to torn_card for a missing/unknown value.
        _supported = ("torn_card", "clean_card", "split_frame", "follow_bar")
        _sel = (os.environ.get("KAIZER_V4_SHORT_LAYOUT", "") or "").strip().lower()
        # Also allow a developer-uploaded template ("custom:<id>").
        _is_custom = _sel.startswith("custom:") and _sel.split(":", 1)[1].isdigit()
        _short_lay = _sel if (_sel in _supported or _is_custom) else "torn_card"
        short_config = ShortConfig(
            layout=_short_lay,
            text=(primary_story.title_native or primary_story.title_english
                  if primary_story else None),
            font_file=_LANG_TO_FONT.get(language, "NotoSansTelugu-Bold.ttf"),
            text_color="#FFFFFF",
            image_filename=first_image,
        )

    return Canvas(
        kind=kind,                     # "bulletin" or "short"
        output_filename=output_filename,
        layout=layout,
        stories=stories,
        trimmed_video_path=trim_result.trimmed_path,
        short_config=short_config,
        # NEW multi-story bulletins crossfade between stories (native
        # motion); single-story and legacy canvases keep the hard cut.
        story_transition=("fade" if (kind == "bulletin" and len(stories) > 1)
                          else None),
    )


# ─── Persistence + DB hooks ─────────────────────────────────────────

def _save_canvas_json(out_dir: Path, job_canvas: V4JobCanvas) -> Path:
    p = out_dir / CANVAS_JSON_NAME
    p.write_text(
        json.dumps(json.loads(job_canvas.model_dump_json()),
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return p


# ── Factory live-belt telemetry (best-effort; never breaks a render) ─────
try:
    from services import stage_events as _se
except Exception:
    _se = None

try:
    from services import stage_gate as _stgate
except Exception:
    _stgate = None


def _gate_acquire():
    """Acquire the cross-process/machine NVENC gate for the compose stage."""
    if _stgate is not None:
        try:
            return _stgate.acquire("encode")
        except Exception:
            return None
    return None


def _gate_release(tok) -> None:
    if _stgate is not None and tok is not None:
        try:
            _stgate.release(tok)
        except Exception:
            pass


def _belt(stage: str, status: str) -> None:
    """Emit a station transition using the thread-bound render envelope."""
    if _se is not None:
        try:
            _se.emit_here(stage, status)
        except Exception:
            pass


def _belt_bind(job_id: int) -> None:
    """Bind this render job's identity envelope to the worker thread so the
    deep stages (trim/transcribe/cut-plan) emit attributable belt events."""
    if _se is None:
        return
    try:
        from database import SessionLocal
        import models
        s = SessionLocal()
        try:
            j = s.query(models.Job).filter(models.Job.id == job_id).first()
            uid = getattr(j, "user_id", None) if j else None
            vname = (getattr(j, "video_name", "") or "") if j else ""
        finally:
            s.close()
        env = _se.Envelope(
            tenant_id=uid, user_id=uid, job_id=int(job_id), clip_id=None,
            channel_id=None, upload_job_id=None,
            label=(vname or f"job {job_id}")[:40])
        _se.bind(env)
    except Exception:
        pass


def _belt_finish(status: str) -> None:
    """Terminal: clear this unit from all stations + unbind the thread."""
    if _se is not None:
        try:
            _se.finish(_se.current(), status)
            _se.unbind()
        except Exception:
            pass


def _update_job_status(job_id: int, *, status: str, current_stage: str = "",
                       error: str = "") -> None:
    """Best-effort DB update. Imports happen lazily so a CLI-only run
    without the FastAPI app loaded still works for testing.

    Also stamps started_at on first 'running' transition (in case the
    runner didn't set it) and finished_at on any terminal status, so
    the JobDetail elapsed badge has both endpoints."""
    try:
        from datetime import datetime as _dt, timezone as _tz
        from database import SessionLocal
        import models
        sess = SessionLocal()
        try:
            j = sess.query(models.Job).filter(models.Job.id == job_id).first()
            if not j:
                return
            j.status = status
            now = _dt.now(_tz.utc)
            if status == "running" and j.started_at is None:
                j.started_at = now
            if status in ("done", "failed", "cancelled") and j.finished_at is None:
                j.finished_at = now
            if hasattr(j, "current_stage"):
                j.current_stage = current_stage
            if error:
                j.error = (j.error or "") + ("\n" if j.error else "") + error[:2000]
            sess.commit()
        finally:
            sess.close()
    except Exception as exc:
        print(f"[v4/db] status update soft-fail: {exc}")
    # Belt terminal cleanup — clear this unit from every station so the live
    # conveyor never shows a ghost card after a job ends.
    if status in ("done", "failed", "cancelled"):
        _belt_finish("failed" if status == "failed" else "exited")


def _extract_thumbnail(video_path: Path, out_path: Path, *,
                       seek_sec: float = 1.5) -> bool:
    """Snap a single JPEG at ``seek_sec`` into ``out_path``. Returns
    True on success. Soft-fails (returns False) so the materialise
    step never blocks on a missing thumbnail."""
    try:
        import shutil as _sh, subprocess as _sp
        ffmpeg = _sh.which("ffmpeg") or "ffmpeg"
        cmd = [
            ffmpeg, "-y", "-v", "error",
            "-ss", f"{max(0.0, seek_sec):.2f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "3",
            str(out_path),
        ]
        r = _sp.run(cmd, capture_output=True, text=True, timeout=20)
        return r.returncode == 0 and out_path.is_file()
    except Exception:
        return False


def _materialise_clips(
    *,
    job_id: int,
    out_dir: Path,
    bulletin_canvas: Canvas,
    shorts_canvases: list,
    bulletin_path: str,
    defer: bool = False,
) -> int:
    """Create one Clip row per V4 output so the existing publish/upload flow can pick them up.
    Idempotent — clears any prior V4 clips for this job before writing the fresh set.
    ``defer=True`` (Stage 2 deferred render): create the Clip rows pointing at their FUTURE
    output paths even though the MP4s don't exist yet, so the editor can open the job and the
    on-demand export writes the files at those exact paths."""
    from database import SessionLocal
    import models
    import json as _json

    sess = SessionLocal()
    try:
        # Drop any prior V4 clips for this job (re-render scenario).
        sess.query(models.Clip).filter(models.Clip.job_id == job_id).delete()

        n = 0
        # Bulletin (one Clip, frame_type="bulletin").
        if bulletin_path and (defer or Path(bulletin_path).is_file()):
            bc = bulletin_canvas
            story0 = bc.stories[0] if bc.stories else None
            seo_str = ""
            if bc.seo:
                try:
                    seo_str = bc.seo.model_dump_json()
                except Exception:
                    seo_str = ""
            bp = Path(bulletin_path)
            thumb_path = ""
            if bp.is_file():
                thumb = bp.with_name(bp.stem + "_thumb.jpg")
                _extract_thumbnail(bp, thumb, seek_sec=2.0)
                thumb_path = str(thumb) if thumb.is_file() else ""
            sess.add(models.Clip(
                job_id=job_id,
                clip_index=0,
                filename=bp.name,
                file_path=str(bp),
                thumb_path=thumb_path,
                image_path="",
                duration=float(sum(
                    (s.video_t_end - s.video_t_start) for s in bc.stories
                ) or 0.0),
                frame_type="bulletin",
                text=(story0.title_native if story0 else "") or "",
                seo=seo_str,
            ))
            n += 1

        # Shorts (one Clip each, frame_type = chosen V1 layout).
        for i, sc in enumerate(shorts_canvases):
            short_path = out_dir / sc.output_filename
            if not short_path.is_file() and not defer:
                continue
            story0 = sc.stories[0] if sc.stories else None
            seo_str = ""
            if sc.seo:
                try:
                    seo_str = sc.seo.model_dump_json()
                except Exception:
                    seo_str = ""
            cfg = sc.short_config
            frame_type = (cfg.layout if cfg else "torn_card") or "torn_card"
            image_filename = (cfg.image_filename if cfg and cfg.image_filename else "")
            image_path = str(out_dir / "_pool" / image_filename) if image_filename else ""
            text = (cfg.text if cfg and cfg.text
                    else (story0.title_native if story0 else "")) or ""
            duration = (story0.video_t_end - story0.video_t_start) if story0 else 0.0
            thumb_path = ""
            if short_path.is_file():
                thumb = short_path.with_name(short_path.stem + "_thumb.jpg")
                _extract_thumbnail(short_path, thumb, seek_sec=1.5)
                thumb_path = str(thumb) if thumb.is_file() else ""
            # Per-short selection label → clip.meta (already exposed by the
            # _clip_dict serializer). Additive + defensive: a missing label
            # just means no badge, never a failed materialisation.
            _short_meta: dict = {}
            try:
                if cfg is not None:
                    if getattr(cfg, "priority", None) is not None:
                        _short_meta["short_priority"] = int(cfg.priority)
                    if getattr(cfg, "why_selected", None):
                        _short_meta["short_why"] = str(cfg.why_selected)
            except Exception:
                _short_meta = {}
            sess.add(models.Clip(
                job_id=job_id,
                clip_index=i + 1,
                filename=short_path.name,
                file_path=str(short_path),
                thumb_path=thumb_path,
                image_path=image_path,
                duration=float(max(0.0, duration)),
                frame_type=frame_type,
                text=text,
                seo=seo_str,
                meta=_json.dumps(_short_meta) if _short_meta else "{}",
            ))
            n += 1

        sess.commit()
        return n
    finally:
        sess.close()


def _maybe_auto_publish(job_id: int, log: callable) -> None:
    """If the job's owner has ``auto_publish=True`` AND
    ``require_consent=False`` in their V4 defaults, queue an UploadJob
    for every materialised Clip to every default channel.

    When ``require_consent=True`` (the safer default) the editor shows
    a "Ready to publish to N channels — confirm?" banner instead and
    nothing leaves the box without a human click."""
    from database import SessionLocal
    import models
    sess = SessionLocal()
    try:
        job = sess.query(models.Job).filter(models.Job.id == job_id).first()
        if not job or not job.user_id:
            return
        user = sess.query(models.User).filter(models.User.id == job.user_id).first()
        if not user or not user.v4_defaults:
            return
        try:
            d = json.loads(user.v4_defaults)
        except Exception:
            return
        if not d.get("auto_publish") or d.get("require_consent", True):
            log("[v4/publish] auto-publish gated -- waiting on user consent")
            return
        channel_ids = [int(c) for c in (d.get("channel_ids") or []) if c]
        if not channel_ids:
            log("[v4/publish] auto-publish skipped -- no channels in defaults")
            return
        clips = sess.query(models.Clip).filter(
            models.Clip.job_id == job_id
        ).order_by(models.Clip.clip_index).all()
        if not clips:
            return
        privacy = d.get("privacy", "public")
        from datetime import datetime as _dt, timezone as _tz
        n = 0
        for c in clips:
            kind = "video" if (c.frame_type == "bulletin") else "short"
            for ch_id in channel_ids:
                sess.add(models.UploadJob(
                    user_id=user.id,
                    clip_id=c.id,
                    channel_id=ch_id,
                    status="queued",
                    publish_kind=kind,
                    privacy_status=privacy,
                    use_seo=True,
                    created_at=_dt.now(_tz.utc),
                ))
                n += 1
        sess.commit()
        log(f"[v4/publish] auto-publish queued {n} UploadJob(s)")
    finally:
        sess.close()


def _load_user_defaults(job_id: int) -> dict:
    """Return the owning user's V4 defaults dict (or {} on any miss).
    Used by the renderer to apply brand suffix / watermark text without
    each call site having to import the user model itself."""
    try:
        from database import SessionLocal
        import models
        sess = SessionLocal()
        try:
            j = sess.query(models.Job).filter(models.Job.id == job_id).first()
            if not j or not j.user_id:
                return {}
            u = sess.query(models.User).filter(models.User.id == j.user_id).first()
            if not u or not u.v4_defaults:
                return {}
            return json.loads(u.v4_defaults) or {}
        finally:
            sess.close()
    except Exception:
        return {}


def _get_job_user_id(job_id: int) -> Optional[int]:
    """Best-effort lookup of the job's owner so generated assets can
    land in user_assets/<uid>/. Returns None when the DB isn't reachable
    (e.g. CLI test runs). The image provider then skips the assets
    copy without erroring."""
    try:
        from database import SessionLocal
        import models
        sess = SessionLocal()
        try:
            j = sess.query(models.Job).filter(models.Job.id == job_id).first()
            return getattr(j, "user_id", None) if j else None
        finally:
            sess.close()
    except Exception:
        return None


def _append_job_log(job_id: int, line: str) -> None:
    try:
        from database import SessionLocal
        import models
        sess = SessionLocal()
        try:
            j = sess.query(models.Job).filter(models.Job.id == job_id).first()
            if j:
                j.log = (j.log or "") + line + "\n"
                sess.commit()
        finally:
            sess.close()
    except Exception:
        pass


def _load_template_media(job_id: int) -> tuple[dict, str]:
    """Resolve a custom-template job's per-slot media to {slot_key: file_path} + the
    name of the AI-trim 'main' slot. Returns ({}, "") when none / on any error."""
    out: dict = {}
    main = ""
    try:
        from database import SessionLocal
        import models
        sess = SessionLocal()
        try:
            j = sess.query(models.Job).filter(models.Job.id == job_id).first()
            if not j:
                return {}, ""
            main = (getattr(j, "main_media_slot", "") or "").strip()
            tm = getattr(j, "template_media", None) or {}
            if isinstance(tm, str):
                import json as _json
                try:
                    tm = _json.loads(tm)
                except Exception:
                    tm = {}
            if isinstance(tm, dict):
                def _resolve_asset(aid):
                    # SECURITY: only resolve assets owned by THIS job's user (defense-in-depth
                    # against a poisoned/legacy template_media row) — applied to EVERY id, incl.
                    # every carousel frame.
                    try:
                        a = (sess.query(models.UserAsset)
                             .filter(models.UserAsset.id == int(aid),
                                     models.UserAsset.user_id == j.user_id).first())
                        if a and a.file_path and os.path.isfile(a.file_path):
                            return a.file_path
                    except Exception:
                        pass
                    return None
                for slot, val in tm.items():
                    # CAROUSEL: {carousel:[{id,duration_s,effect,effect_duration}], fit, offsets}
                    if isinstance(val, dict) and val.get("carousel"):
                        frames = []
                        for fr in (val.get("carousel") or []):
                            if not isinstance(fr, dict):
                                continue
                            p = _resolve_asset(fr.get("id"))
                            if not p:
                                continue
                            frames.append({
                                "path": p,
                                "duration_s": float(fr.get("duration_s") or 3.0),
                                "effect": str(fr.get("effect") or "fade"),
                                "effect_duration": float(fr.get("effect_duration") or 0.4),
                            })
                        frames = frames[:50]   # defense-in-depth: cap even a poisoned/legacy Job row
                        if frames:
                            out[str(slot)] = {"carousel": frames, "fit": val.get("fit") or "cover",
                                              "offset_x_pct": val.get("offset_x_pct"),
                                              "offset_y_pct": val.get("offset_y_pct")}
                        continue
                    # scalar asset id -> single image/video path
                    p = _resolve_asset(val)
                    if p:
                        out[str(slot)] = p
        finally:
            sess.close()
    except Exception:
        return {}, ""
    return out, main


def _template_autoimages_on() -> bool:
    """KAIZER_V4_TEMPLATE_AUTOIMAGES gate — default ON."""
    return (os.environ.get("KAIZER_V4_TEMPLATE_AUTOIMAGES", "1") or "1").strip().lower() \
        not in ("0", "off", "false", "no")


def _active_custom_template_ids(output_format: str) -> list[int]:
    """CustomTemplate ids actually used by THIS render (fullform + shorts envs)."""
    ids: list[int] = []
    _ff = (os.environ.get("KAIZER_V4_FULLFORM_LAYOUT", "") or "").strip().lower()
    if (output_format not in ("shorts-only", "trailer-only")
            and _ff.startswith("custom:") and _ff.split(":", 1)[1].isdigit()):
        ids.append(int(_ff.split(":", 1)[1]))
    _sl = (os.environ.get("KAIZER_V4_SHORT_LAYOUT", "") or "").strip().lower()
    if (output_format not in ("full-only", "trailer-only")
            and _sl.startswith("custom:") and _sl.split(":", 1)[1].isdigit()):
        _tid = int(_sl.split(":", 1)[1])
        if _tid not in ids:
            ids.append(_tid)
    return ids


def _empty_image_slots(contract_jsons: list, template_media: dict,
                       max_slots: int) -> list[tuple[str, bool]]:
    """Pure slot-fill decision: which image slots have NO user media yet.
    Dedupes by slot key across templates (first template wins — ONE shared
    template_media map serves both the fullform and short custom templates)
    and caps at max_slots so a slot-heavy template can't explode Gemini
    spend. Returns [(slot_key, is_carousel), ...]."""
    empty: list[tuple[str, bool]] = []
    seen: set[str] = set()
    tm = template_media if isinstance(template_media, dict) else {}
    for cj in (contract_jsons or []):
        if not isinstance(cj, dict):
            continue
        for s in (cj.get("slots") or []):
            if not isinstance(s, dict) or s.get("kind") != "image":
                continue
            key = str(s.get("name") or s.get("kind") or "").strip()
            if not key or key in seen or key in tm:
                continue
            seen.add(key)
            empty.append((key, bool(s.get("carousel"))))
    return empty[:max(1, int(max_slots or 1))]


def _slot_fill_value(is_carousel: bool, asset_ids: list):
    """Media-map value for a slot from the generated asset ids — single slot
    -> the first id (scalar, the shape _load_template_media resolves), carousel
    slot -> the carousel struct the create_job/editor save paths write. None
    when no asset survived generation (slot stays empty, so the positional
    support_images pool fill still applies at render)."""
    ids = [int(a) for a in (asset_ids or []) if a]
    if not ids:
        return None
    if not is_carousel:
        return ids[0]
    return {"carousel": [{"id": i, "duration_s": 3.0, "effect": "fade",
                          "effect_duration": 0.4}
                         # defense-in-depth: same 50-frame cap as _load_template_media
                         for i in ids[:50]],
            "fit": "cover"}


def _merge_template_media(existing: dict, filled: dict) -> dict:
    """Fresh-dict merge (JSON column — never mutate in place, see the
    models.py:249 pattern or the commit silently no-ops). User-filled slot
    keys always win over auto-generated ones."""
    merged = dict(existing or {})
    for k, v in (filled or {}).items():
        merged.setdefault(k, v)
    return merged


def _persist_generated_slot_asset(sess, *, user_id: int, src_path: str,
                                  label: str, job_id: int) -> Optional[int]:
    """UserAsset row for a pipeline-generated slot image. Unlike routers.assets.
    _persist_ai_image this NEVER deletes the local file — _load_template_media
    resolves by local file_path (isfile check). Storage upload + thumb are
    best-effort. Returns the asset id or None."""
    import models
    p = Path(src_path)
    if not p.is_file():
        return None
    thumb_path, w, h = "", 0, 0
    try:
        from routers.assets import _thumb_for           # lazy: pulls FastAPI once
        thumb_path, w, h = _thumb_for(p)
    except Exception:
        pass
    storage_backend = storage_key = storage_url = thumb_storage_url = ""
    try:
        from pipeline_core.storage import get_storage_provider
        storage = get_storage_provider()
        rel_dir = f"user_assets/{user_id}/"
        obj = storage.upload(str(p), rel_dir + p.name, content_type="image/jpeg")
        storage_backend, storage_key, storage_url = storage.name, obj.key, obj.url
        if thumb_path:
            thumb_storage_url = storage.upload(
                thumb_path, rel_dir + Path(thumb_path).name,
                content_type="image/jpeg").url
        # NOTE: local file intentionally KEPT even on remote storage.
    except Exception as exc:
        print(f"[v4] slot-asset storage upload failed (soft): {exc}", flush=True)
    row = models.UserAsset(
        user_id=user_id, filename=p.name, file_path=str(p.resolve()),
        thumb_path=thumb_path, kind="image", mime="image/jpeg",
        size_bytes=p.stat().st_size, width=w, height=h, is_default_ad=False,
        tags=["ai-generated", f"job:{job_id}"], folder_path="generated",
        description=(label or "").strip()[:500],
        storage_backend=storage_backend, storage_key=storage_key,
        storage_url=storage_url, thumb_storage_url=thumb_storage_url,
    )
    sess.add(row); sess.commit(); sess.refresh(row)
    return row.id


def _autofill_template_images(*, job_id: int, trim_result, language: str,
                              output_format: str) -> int:
    """PIPELINE AUTO-FILL: for a custom-template job whose image slots the user
    left EMPTY, plan imagery FROM THE STORY CONTENT and generate + attach it —
    single slot -> 1 image, carousel-marked slot -> story-driven N — persisting
    to Job.template_media so THIS render (Stage 3 re-reads the row via
    _load_template_media) AND the editor both see the images. Never overwrites
    a user-filled slot. Returns the number of slots filled; 0 on any skip.
    Caller wraps in try/except — any failure -> render proceeds with template
    defaults (fail-soft)."""
    if not _template_autoimages_on():
        return 0
    tids = _active_custom_template_ids(output_format)
    if not tids:
        return 0
    from database import SessionLocal
    import models
    sess = SessionLocal()
    try:
        j = sess.query(models.Job).filter(models.Job.id == job_id).first()
        if not j or not j.user_id:
            return 0
        tm = getattr(j, "template_media", None) or {}
        if isinstance(tm, str):
            try:
                tm = json.loads(tm)
            except Exception:
                tm = {}
        if not isinstance(tm, dict):
            tm = {}
        # Discover EMPTY image slots across the active template(s).
        contract_jsons = []
        for tid in tids:
            t = sess.get(models.CustomTemplate, tid)
            contract_jsons.append((getattr(t, "contract_json", None) or {}) if t else {})
        # Cost guard: don't let a slot-heavy template explode Gemini spend.
        _max_slots = max(1, int(os.environ.get(
            "KAIZER_V4_TEMPLATE_AUTOIMAGES_MAX_SLOTS", "4") or 4))
        empty = _empty_image_slots(contract_jsons, tm, _max_slots)
        if not empty:
            return 0

        stories = list(getattr(trim_result, "stories", None) or [])
        s0 = stories[0] if stories else None
        t_nat = (getattr(s0, "title_native", "") or "") if s0 else ""
        t_eng = (getattr(s0, "title_english", "") or "") if s0 else ""
        # Single slots: primary story (summary enriched with its transcript, the
        # image_provider._generate_via_gemini precedent). Carousels: the WHOLE
        # narrative so the planner's beats span every story.
        s0_sum = (getattr(s0, "summary", "") or "") if s0 else ""
        s0_tr = (getattr(s0, "transcript_text", "") or "") if s0 else ""
        single_summary = (f"{s0_sum}\n\nSpoken transcript: {s0_tr[:600]}"
                          if s0_tr else s0_sum)[:1200]
        combined = "\n".join(
            f"- {(getattr(s, 'title_native', '') or getattr(s, 'title_english', '') or '').strip()}: "
            f"{(getattr(s, 'summary', '') or '').strip()}"
            for s in stories)[:1200] or single_summary

        out_root = image_provider._user_assets_dir_for(j.user_id)
        if out_root is None:
            return 0
        from pipeline_v4 import image_ai as _iai
        import uuid as _uuid
        # Honor the job's image-engine pick (wizard → Job.v4_image_provider →
        # KAIZER_V4_IMAGE_PROVIDER env, set per-job by the runner). "auto"
        # keeps Gemini here — the web-photo chain has no slot-fill notion.
        _engine = ("openai" if image_provider._selected_image_provider() == "openai"
                   else "gemini")
        filled: dict = {}
        for key, is_car in empty:
            safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in key)[:32]
            if is_car:
                triples = _iai.make_images_for_story(
                    title_native=t_nat, title_english=t_eng, summary=combined,
                    language=language, count=None, out_dir=str(out_root),
                    base=f"tmpl_{job_id}_{safe}_{_uuid.uuid4().hex[:6]}",
                    max_images=max(1, min(int(os.environ.get(
                        "KAIZER_AI_IMAGE_MAX", "8") or 8), 12)),
                    engine=_engine)
                frame_ids = []
                _first_frame_path = None
                for path, _prompt, label in triples:
                    aid = _persist_generated_slot_asset(
                        sess, user_id=j.user_id, src_path=path,
                        label=label, job_id=job_id)
                    if aid:
                        frame_ids.append(aid)
                        if _first_frame_path is None:
                            _first_frame_path = path
                val = _slot_fill_value(True, frame_ids)
                if val:
                    # Face-aware framing for the slot's cover-crop: the
                    # SLOT-level offsets are the only channel the renderer
                    # reads (_load_template_media; per-frame offsets aren't
                    # plumbed), so the first surviving frame's face decides.
                    # Centered/failed → omit (legacy struct shape). Scalar
                    # slots have NO offset channel (bare asset id) — a known
                    # limitation, not widened in this pass.
                    try:
                        if _first_frame_path:
                            from pipeline_v4 import face_focus
                            _fox, _foy = face_focus.focal_point(_first_frame_path)
                            if (_fox, _foy) != (50.0, 50.0):
                                val["offset_x_pct"] = _fox
                                val["offset_y_pct"] = _foy
                    except Exception:
                        pass
                    filled[key] = val
            else:
                out_path = out_root / f"tmpl_{job_id}_{safe}_{_uuid.uuid4().hex[:6]}.jpg"
                saved, final_prompt = _iai.make_image_for_story(
                    title_native=t_nat, title_english=t_eng,
                    summary=single_summary or combined, language=language,
                    out_path=str(out_path), engine=_engine)
                if saved and out_path.is_file():
                    aid = _persist_generated_slot_asset(
                        sess, user_id=j.user_id, src_path=str(out_path),
                        label=_iai.derive_label(final_prompt), job_id=job_id)
                    val = _slot_fill_value(False, [aid] if aid else [])
                    if val:
                        filled[key] = val
        if not filled:
            return 0
        j.template_media = _merge_template_media(tm, filled)
        sess.commit()
        return len(filled)
    finally:
        sess.close()


def _load_template_overrides(job_id: int) -> dict:
    """Per-slot TEXT overrides set in the custom-template editor (Job.template_overrides).
    Returns {slot_key: text}. {} on any error / when none."""
    try:
        from database import SessionLocal
        import models
        sess = SessionLocal()
        try:
            j = sess.query(models.Job).filter(models.Job.id == job_id).first()
            ov = (getattr(j, "template_overrides", None) or {}) if j else {}
            if isinstance(ov, str):
                import json as _json
                try:
                    ov = _json.loads(ov)
                except Exception:
                    ov = {}
            return {str(k): ("" if v is None else str(v)) for k, v in ov.items()} if isinstance(ov, dict) else {}
        finally:
            sess.close()
    except Exception:
        return {}


def _load_html_override(job_id: int, target: str, index: int) -> str:
    """Per-job VISUAL design override (Job.custom_html_overrides). When the operator edited
    the template for THIS job+output in the inline builder, returns that HTML so the render
    uses it verbatim (literal mode). "" when none / on any error -> normal template path.
    Keyed "<target>:<index>" with target in {"bulletin","short"}."""
    try:
        from database import SessionLocal
        import models
        sess = SessionLocal()
        try:
            j = sess.query(models.Job).filter(models.Job.id == job_id).first()
            ov = (getattr(j, "custom_html_overrides", None) or {}) if j else {}
            if isinstance(ov, str):
                import json as _json
                try:
                    ov = _json.loads(ov)
                except Exception:
                    ov = {}
            if not isinstance(ov, dict):
                return ""
            t = "bulletin" if (target or "").strip().lower() == "bulletin" else "short"
            v = ov.get(f"{t}:{max(0, int(index or 0))}")
            return v if (isinstance(v, str) and v.strip()) else ""
        finally:
            sess.close()
    except Exception:
        return ""


# ─── Public end-to-end entry ────────────────────────────────────────

def run_job(
    *,
    job_id: int,
    source_video: str,
    output_dir: str,
    language: str = "te",
    brand_logo: Optional[str] = None,
    bulletin_images_env: str = "",
) -> str:
    """Run the full V4 pipeline. Returns the canvas.json path."""
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        print(msg, flush=True)
        _append_job_log(job_id, msg)

    # Job-global stopwatch + per-stage timer. Same pattern as V2/V3 so
    # the user can read off the per-stage and total elapsed in one glance.
    job_t0 = time.time()
    def _fmt_elapsed(secs: float) -> str:
        return f"{secs:.1f}s" if secs < 60 else f"{int(secs // 60)}m{int(secs % 60):02d}s"

    _log(f"[v4] job {job_id} starting -- source={source_video}")
    _log(f"[v4] output_dir={out_dir}")
    # Bind this job's identity to the worker thread so every render stage
    # emits attributable events to the live conveyor, then open the belt.
    _belt_bind(job_id)
    _belt("ingest", "entered")
    _compose_tok = None  # NVENC gate token, held only during the compose stage
    # Probe the source video's aspect ratio NOW so Stage 2 can size
    # the inner video panel to match. Without this, a vertical (9:16)
    # phone-shot source gets cover-crop-zoomed into a 16:9 chrome
    # panel and the operator's face ends up massively cropped.
    # Audio-first mode: the narration AUDIO is the master track (it arrives as
    # --source); the optional MUTED reference b-roll comes via env. When there
    # is no reference video the generated image fills the frame (operator's
    # choice). Read the flags up front so Stage 1 + the canvas can branch.
    audio_first = (os.environ.get("KAIZER_V4_AUDIO_FIRST", "0") or "0").strip().lower() in ("1", "true", "yes")
    ref_video = (os.environ.get("KAIZER_V4_REF_VIDEO_PATH", "") or "").strip() or None
    if audio_first:
        _log(f"[v4] AUDIO-FIRST MODE active (narration={source_video}; "
             f"ref_video={ref_video or 'none -> fullscreen images'})")

    if audio_first:
        # No video to probe for a no-ref job; a ref b-roll drives panel sizing.
        src_aspect = _probe_source_aspect(ref_video) if ref_video else (16.0 / 9.0)
    else:
        src_aspect = _probe_source_aspect(source_video)
    src_aspect_label = (
        "wide (>=1.2)" if src_aspect >= 1.2 else f"narrow ({src_aspect:.2f})"
    )
    _log(f"[v4] source aspect: {src_aspect:.3f} -> {src_aspect_label} bulletin layout")
    # Source-preserved mode: when the operator supplied a description
    # the orchestrator skips Claude KEEP/CUT entirely and keeps the
    # source video AS-IS as the bulletin. Shorts are still carved.
    predef_description = _decode_predefined_description()
    if predef_description:
        _log(f"[v4] SOURCE-PRESERVED MODE active "
             f"(predef description supplied — {len(predef_description)} chars)")
    # Output-format choice — which outputs the operator asked us to render.
    #   "both" (default) renders the full video (bulletin) + shorts;
    #   "full-only" skips ALL shorts work (no candidates, trims, SEO, render);
    #   "shorts-only" skips the bulletin/full-video render + its SEO.
    # Forwarded from runner.py via KAIZER_V4_OUTPUT_FORMAT. Validated here
    # again so a stray value never reaches the render gates.
    output_format = (os.environ.get("KAIZER_V4_OUTPUT_FORMAT", "both") or "both").strip().lower()
    if output_format not in {"both", "full-only", "shorts-only", "trailer-only"}:
        output_format = "both"
    if output_format != "both":
        _log(f"[v4] output_format={output_format} -- "
             f"{'shorts only (no full video)' if output_format == 'shorts-only' else 'full video only (no shorts)'}")
    # DEFER RENDER (Stage 2, default OFF): when set, the pipeline produces the raw cut video +
    # canvas.json scene + Clip rows but SKIPS the up-front compose — the heavy MP4 render then
    # happens on demand ("export") in the editor. Edit-first jobs finish fast; auto-publish is
    # skipped for a deferred job (export first, then publish). Forwarded via KAIZER_V4_DEFER_RENDER.
    defer_render = (os.environ.get("KAIZER_V4_DEFER_RENDER", "0") or "0").strip().lower() in ("1", "true", "yes")
    if defer_render:
        _log("[v4] KAIZER_V4_DEFER_RENDER=1 -- render deferred (scene-only; export in the editor)")
    _belt("ingest", "exited")
    _update_job_status(job_id, status="running", current_stage="step1_trim")

    try:
        # ─── Stage 1/3 — atomic trim+concat (or source-preserved) ──
        stage_t0 = time.time()
        if audio_first:
            # AUDIO-FIRST wins over source-preserved (they are mutually
            # exclusive). Narration is the master; the muted ref b-roll (if
            # any) is muxed under it, else a black filler for fullscreen images.
            trim_result = _build_audio_driven_trim_result(
                narration_audio=source_video,
                ref_video=ref_video,
                output_dir=out_dir,
                language=language,
                output_filename="trimmed_bulletin.mp4",
            )
        elif predef_description:
            trim_result = _build_source_preserved_trim_result(
                source_video=source_video,
                output_dir=out_dir,
                output_filename="trimmed_bulletin.mp4",
            )
        else:
            trim_result = trim_engine.run_step1(
                source_video=source_video,
                output_dir=str(out_dir),
                language=language,
                output_filename="trimmed_bulletin.mp4",
            )
        _log(f"[v4] Stage 1/3 DONE in {_fmt_elapsed(time.time() - stage_t0)} "
             f"-- {trim_result.trimmed_duration_sec:.1f}s output, "
             f"{len(trim_result.stories)} stories  (total {_fmt_elapsed(time.time() - job_t0)})")

        # Persist per-story word timestamps for the editor's
        # "Sync images to speech" (fail-soft, never blocks the render).
        _write_story_words_sidecar(out_dir, trim_result, language=language)

        # ─── Stage 1.5 — TV-style profanity bleep ───────────────────
        # Mute + 1kHz tone over censorable words on the trimmed master
        # (audio-only rewrite, video stream copied). Fail-soft; report
        # lands in bleep_report.json for the editor. KAIZER_V4_BLEEP=0
        # turns the channel censor off.
        from pipeline_v4 import bleep as v4_bleep
        v4_bleep.run_bleep_pass(trim_result=trim_result, out_dir=out_dir)

        # ─── Ingest user-supplied image pool ────────────────────────
        pool = _ingest_image_pool(out_dir, bulletin_images_env)
        _log(f"[v4] pool ingested -- {len(pool)} images")

        # ─── Auto-fetch authentic images if user gave none ─────────
        # User-uploaded images always win. Only when the pool is empty
        # do we run V1's CSE/DDG/Pexels/OpenAI chain to land one image
        # per story. Generated images also land in the operator's
        # user_assets/ folder so they're reusable across future jobs.
        if output_format == "trailer-only":
            # Trailer-only: no bulletin compose → sidebar images are never
            # shown. Skip the whole fetch/generate spend.
            _log("[v4] output_format=trailer-only -- skipping image auto-fetch")
        else:
            try:
                user_assets = image_provider._user_assets_dir_for(_get_job_user_id(job_id))
                new_imgs = image_provider.auto_populate_pool(
                    stories=list(trim_result.stories),
                    pool_dir=out_dir / "_pool",
                    language=language,
                    user_assets_dir=user_assets,
                    only_if_empty=True,
                )
                if new_imgs:
                    pool.extend(new_imgs)
                    _log(f"[v4] auto-fetched {len(new_imgs)} authentic image(s)")
            except Exception as exc:
                _log(f"[v4] image auto-fetch failed (soft-skip): {exc}")

        # ─── FACE-AWARE FRAMING: stamp a focal point on every pool image ──
        # ONE choke point covers env-pool ingest + auto-fetch + AI-generate:
        # each image entry carries the detected face center as
        # offset_x/y_pct, which _build_initial_canvas copies onto the
        # CanvasImage so every cover-crop keeps heads in frame. Stamped
        # ONLY when non-default (offsets already hash unconditionally in
        # v1_bridge._carousel_blob) so face-less images keep the schema
        # default and old canvases stay hash-identical. The editor's
        # 9-point grid overwrites the fields later — user framing wins.
        # Fail-soft: detection can never block a paid render.
        try:
            from pipeline_v4 import face_focus
            for _pe in pool:
                _fn = str(_pe.get("filename") or "")
                if (_fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                        and "offset_x_pct" not in _pe):
                    _ox, _oy = face_focus.focal_point(out_dir / "_pool" / _fn)
                    if (_ox, _oy) != (50.0, 50.0):
                        _pe["offset_x_pct"], _pe["offset_y_pct"] = _ox, _oy
        except Exception as _ffx:
            _log(f"[v4] face-focus skipped (soft): {_ffx}")

        # ─── Reference-video AUTO-SOURCE fallback (opt-in, OFF by default) ──
        # For each story with NO uploaded reference clip, best-effort fetch one
        # short related clip (yt-dlp) so the AI has B-roll to cut to. Gated by
        # KAIZER_V4_AUTOSOURCE_VIDEO; capped by KAIZER_V4_AUTOSOURCE_MAX.
        # COPYRIGHT is the operator's responsibility (see ref_video_source.py);
        # the uploaded-clip path is the safe primary source. Fully fail-soft.
        try:
            from pipeline_v4 import ref_video_source as _rvs
            _VEXT = (".mp4", ".mov", ".webm", ".m4v", ".mkv")
            if _rvs.autosource_enabled() and output_format != "trailer-only":
                _have_vid = {p.get("story_index") for p in pool
                             if str(p.get("filename") or "").lower().endswith(_VEXT)}
                _cap = int(os.environ.get("KAIZER_V4_AUTOSOURCE_MAX", "3") or "3")
                _got = 0
                for s in trim_result.stories:
                    if _got >= _cap:
                        break
                    if s.story_index in _have_vid:
                        continue
                    _q = (getattr(s, "title_english", "")
                          or getattr(s, "title_native", "") or "").strip()
                    if not _q:
                        continue
                    _clip = _rvs.autosource_clip(_q, out_dir / "_pool", label=_q)
                    if _clip:
                        _clip["story_index"] = s.story_index
                        _clip["kind"] = "autosource"
                        pool.append(_clip)
                        _got += 1
                if _got:
                    _log(f"[v4] auto-sourced {_got} reference video(s)")
        except Exception as _asx:
            _log(f"[v4] reference-video auto-source skipped (soft): {_asx}")

        # ─── Custom-template AUTO-IMAGES (story-driven slot fill) ───────
        # Image slots the user left EMPTY get AI imagery planned FROM THE
        # STORY CONTENT (single slot -> 1 image; carousel slot -> story-
        # driven N) and persisted to Job.template_media BEFORE Stage 3 —
        # both the full-form custom branch and the shorts custom path
        # re-read the row via _load_template_media, and the editor sees
        # the same asset ids. Fully fail-soft; KAIZER_V4_TEMPLATE_
        # AUTOIMAGES=0 turns it off.
        try:
            if output_format != "trailer-only":
                _n_auto = _autofill_template_images(
                    job_id=job_id, trim_result=trim_result,
                    language=language, output_format=output_format)
                if _n_auto:
                    _log(f"[v4] template auto-images: filled {_n_auto} empty image slot(s)")
        except Exception as _tai_exc:
            _log(f"[v4] template auto-images skipped (soft): {_tai_exc}")

        # ─── Stage 2/3 — build canvas + per-story short trims ──────
        stage_t0 = time.time()
        _update_job_status(job_id, status="running", current_stage="step2_canvas")
        _belt("trim", "exited")
        # Acquire the NVENC gate before the heavy compose stage (canvas +
        # carousel + materialise). Blocks under back-pressure so only N jobs
        # compose at once across the whole fleet; released in the finally.
        _compose_tok = _gate_acquire()
        _belt("compose", "entered")
        bulletin_canvas = _build_initial_canvas(
            trim_result=trim_result,
            pool=pool,
            kind="bulletin",
            output_filename="bulletin.mp4",
            brand_logo=brand_logo,
            language=language,
            source_aspect=src_aspect,
            # Audio-first with no reference video → image fills the frame.
            fullscreen_image=(audio_first and not ref_video),
        )

        # ─── Per-story shorts ───────────────────────────────────────
        # Goal: at LEAST 4 shorts per job (target 4-5+). Long stories
        # (>90s) get chunked into ~60s sub-windows; medium stories
        # become one short each; stories <8s are skipped unless we'd
        # otherwise fall under the minimum (then the floor drops to 5s).
        SHORT_MIN_SEC      = 6.0
        SHORT_MIN_FALLBACK = 4.0     # used only when total candidates < TARGET_MIN
        SHORT_MAX_SEC      = 35.0    # a "short" stays SHORT (retained for the fallback bound)
        SHORT_TARGET_SEC   = 25.0    # split EVERY story into ~25s windows → multiple short clips
        TARGET_MIN_SHORTS  = 4

        def _split_spans(spans: list, target_sec: float) -> list[list]:
            """Greedy partition: walk spans in order, start a new chunk
            whenever the running duration would exceed ``target_sec``.
            Returns a list of chunks (each a list of KeptSpan)."""
            chunks: list[list] = []
            cur: list = []
            cur_dur = 0.0
            for sp in spans:
                d = max(0.0, sp.duration)
                if cur and cur_dur + d > target_sec:
                    chunks.append(cur)
                    cur = []
                    cur_dur = 0.0
                cur.append(sp)
                cur_dur += d
            if cur:
                chunks.append(cur)
            return chunks

        def _subdivide_long_spans(spans: list, target_sec: float) -> list:
            """Split any single span longer than ~target_sec into ~equal
            sub-spans. Audio-first stories are ONE contiguous span each (the
            full segment), so without this a long story would yield one
            over-long 'short'. Time-based split (contiguous audio, no gaps)."""
            out: list = []
            for sp in spans:
                d = max(0.0, sp.duration)
                if d <= target_sec * 1.35:
                    out.append(sp)
                    continue
                n = max(2, int(d // target_sec) + (1 if (d % target_sec) > 1.0 else 0))
                step = d / n
                for k in range(n):
                    a = sp.start_sec + k * step
                    b = sp.end_sec if k == n - 1 else sp.start_sec + (k + 1) * step
                    out.append(trim_engine.KeptSpan(start_sec=a, end_sec=b, reason=sp.reason))
            return out

        # Build candidate list first (no I/O), then trim. Each candidate
        # is (display_idx, title_native, title_english, summary, spans, est_dur).
        candidates: list[tuple] = []
        for s_idx, s in enumerate(trim_result.stories):
            if s.duration < SHORT_MIN_SEC:
                continue
            # ALWAYS split a story into short ~SHORT_TARGET_SEC windows so a long
            # story yields MULTIPLE short clips (not one long "short"). A short story
            # (≤ target) stays a single window. Split is at kept-span boundaries, so
            # windows never cut mid-sentence.
            # Audio-first stories are single contiguous spans → subdivide long
            # ones by time first so they become multiple ~25s short windows.
            _short_spans = (_subdivide_long_spans(list(s.source_spans), SHORT_TARGET_SEC)
                            if audio_first else list(s.source_spans))
            chunks = _split_spans(_short_spans, SHORT_TARGET_SEC)
            multi = len(chunks) > 1
            for sub_i, chunk in enumerate(chunks):
                chunk_dur = sum(sp.duration for sp in chunk)
                if chunk_dur < SHORT_MIN_SEC:
                    continue
                candidates.append((
                    s_idx, s.title_native, s.title_english, s.summary,
                    chunk, chunk_dur, (sub_i + 1) if multi else None,
                ))

        # Fallback: still under the minimum? accept stories 5-8s too.
        if len(candidates) < TARGET_MIN_SHORTS:
            existing_keys = {(c[0], c[6]) for c in candidates}
            for s_idx, s in enumerate(trim_result.stories):
                if SHORT_MIN_FALLBACK <= s.duration < SHORT_MIN_SEC and (s_idx, None) not in existing_keys:
                    candidates.append((
                        s_idx, s.title_native, s.title_english, s.summary,
                        list(s.source_spans), s.duration, None,
                    ))
                    if len(candidates) >= TARGET_MIN_SHORTS:
                        break

        # ─── Cap the number of shorts ───────────────────────────────
        # Default 8 per job. The operator can opt into more at job start
        # (KAIZER_V4_MAX_SHORTS, forwarded by runner). This is a CEILING,
        # not a target — the real count still depends on how many qualifying
        # segments the video yields, so a short source can't reach the cap.
        # Candidates are ordered earliest-story-first, so capping keeps the
        # lead segments (the highest-priority ones).
        try:
            _max_shorts = int(os.environ.get("KAIZER_V4_MAX_SHORTS", "8") or "8")
        except ValueError:
            _max_shorts = 8
        _max_shorts = max(1, min(50, _max_shorts))
        if len(candidates) > _max_shorts:
            _log(f"[v4] shorts cap -- {len(candidates)} candidates -> keeping {_max_shorts}")
            candidates = candidates[:_max_shorts]

        # Output-format gate (full-only): drop every shorts candidate so
        # no trims, no shorts canvases, no shorts SEO, and no shorts render
        # happen downstream. The bulletin path is untouched. Setting
        # candidates=[] cascades cleanly: the trim pool (`if candidates`),
        # the canvas loop (zip over empty), the SEO loop and the shorts
        # render (`if shorts_canvases`) all no-op.
        if output_format in ("full-only", "trailer-only") and candidates:
            _log(f"[v4] output_format=full-only -- skipping all {len(candidates)} shorts")
            candidates = []

        _log(f"[v4] shorts plan -- {len(candidates)} candidates "
             f"(target >= {TARGET_MIN_SHORTS})")

        shorts_canvases: list[Canvas] = []
        trimmed_shorts_paths: list[str] = []

        # Render/trim concurrency knob — shared by the parallel trims
        # here (Wave 4 item D) and the parallel shorts renders in
        # Stage 3 (item C). Default 3; set to 1 for the old sequential
        # behaviour.
        try:
            _render_workers = max(1, int(os.environ.get(
                "KAIZER_V4_RENDER_CONCURRENCY", "3") or "3"))
        except ValueError:
            _render_workers = 3

        def _short_label(idx: int, cand: tuple) -> str:
            c_s_idx, _tn, _te, _su, _sp, _du, c_sub_i = cand
            return (
                f"short_{idx + 1:02d}"
                if c_sub_i is None
                else f"short_{idx + 1:02d}_s{c_s_idx + 1:02d}p{c_sub_i}"
            )

        def _short_why(idx: int, cand: tuple) -> str:
            """Human reason this segment was auto-picked as a short, for the
            operator-facing selection label. Heuristic from the signals the
            candidate already carries (position, duration, sub-chunk)."""
            _si, _tn, _te, _su, _sp, c_dur, c_sub = cand
            d = float(c_dur or 0.0)
            if idx == 0:
                return "Top pick — lead story, highest reach potential"
            if c_sub is not None:
                return "Standout moment pulled from a longer story"
            if 20.0 <= d <= 55.0:
                return "Punchy length — ideal for Shorts/Reels retention"
            if d < 8.0:
                return "Quick highlight clip"
            return "Complete story — strong standalone short"

        def _trim_one_candidate(idx: int, cand: tuple) -> Optional[str]:
            """One per-short atomic trim. Returns the trimmed path or
            None on failure (a failed trim must not kill the others —
            same per-candidate isolation as the old sequential loop)."""
            _c_s_idx, _tn, _te, _su, c_spans, c_dur, _c_sub_i = cand
            label = _short_label(idx, cand)
            short_trimmed_path = str(out_dir / f"trimmed_{label}.mp4")
            try:
                trim_engine.atomic_trim_concat(
                    # Audio-first: carve shorts from the muxed A/V master (it has
                    # real video + the narration audio). source_video is the raw
                    # narration audio, which has NO [0:v] stream → the trim would
                    # fail and the short would be silently dropped.
                    source_video=(trim_result.trimmed_path if audio_first else source_video),
                    spans=c_spans,
                    output_path=short_trimmed_path,
                )
            except Exception as exc:
                _log(f"[v4] {label}: trim failed: {exc}")
                return None
            _log(f"[v4] {label}: trimmed {c_dur:.1f}s -> {Path(short_trimmed_path).name}")
            return short_trimmed_path

        # Bounded parallel trims: each candidate is an independent
        # single-pass ffmpeg job over the source, so they parallelise
        # cleanly. Results are gathered in candidate order so labels,
        # canvases and clip indices stay deterministic.
        trim_results: list[Optional[str]] = []
        if candidates:
            from concurrent.futures import ThreadPoolExecutor as _TrimPool
            with _TrimPool(
                max_workers=min(_render_workers, len(candidates)),
                thread_name_prefix="v4-trim",
            ) as _trim_pool:
                trim_results = list(_trim_pool.map(
                    _trim_one_candidate, range(len(candidates)), candidates,
                ))

        for idx, (cand, short_trimmed_path) in enumerate(zip(candidates, trim_results)):
            if not short_trimmed_path:
                continue
            s_idx, title_n, title_e, summary, spans, dur, sub_i = cand
            label = _short_label(idx, cand)
            trimmed_shorts_paths.append(short_trimmed_path)

            short_trim = trim_engine.TrimResult(
                trimmed_path=short_trimmed_path,
                trimmed_duration_sec=dur,
                stories=[
                    trim_engine.TrimmedStory(
                        story_index=0,
                        title_native=title_n,
                        title_english=title_e,
                        summary=summary,
                        video_t_start=0.0,
                        video_t_end=dur,
                        source_spans=spans,
                    )
                ],
                source_duration_sec=trim_result.source_duration_sec,
                removed_sec_total=trim_result.removed_sec_total,
            )
            short_canvas = _build_initial_canvas(
                trim_result=short_trim,
                pool=pool,
                kind="short",
                output_filename=f"{label}.mp4",
                brand_logo=brand_logo,
                language=language,
            )
            # Per-short selection label (priority + why). Display-only +
            # ADDITIVE — wrapped so a label hiccup can NEVER break the render.
            try:
                if short_canvas.short_config is not None:
                    short_canvas.short_config.priority = idx + 1
                    short_canvas.short_config.why_selected = _short_why(idx, cand)
            except Exception:
                pass
            shorts_canvases.append(short_canvas)

        # ─── SEO — Gemini 2.5 Flash per short + one for the bulletin ──
        # Run all SEO calls (bulletin + N shorts) in parallel — they're
        # independent network round-trips so a ThreadPoolExecutor with
        # one slot per call cuts wall time from ~N+1 × per_call_seconds
        # down to ~per_call_seconds. Each call is wrapped individually
        # so a single Gemini hiccup on one short never blocks the
        # others (the old single-try-around-everything aborted the
        # whole batch on the first exception).
        from concurrent.futures import ThreadPoolExecutor

        bull_story = bulletin_canvas.stories[0] if bulletin_canvas.stories else None
        bull_body = "\n".join(
            f"- {s.title_native or s.title_english}"
            for s in bulletin_canvas.stories if s.title_native or s.title_english
        )
        # Build the work list as (label, canvas_target, SeoInput). The
        # label is only for log messages on per-call failure. We index
        # by canvas object so the result writes back to the right slot.
        seo_jobs: list[tuple[str, object, seo_provider.SeoInput]] = []
        # Output-format gate (shorts-only): no bulletin render -> no need to
        # spend a Gemini call on bulletin SEO. Skipping the append leaves
        # bulletin_canvas.seo as None, which is fine (no bulletin Clip).
        if output_format != "shorts-only":
            seo_jobs.append((
                "bulletin",
                bulletin_canvas,
                seo_provider.SeoInput(
                    kind="bulletin",
                    language=language,
                    title_native=(bull_story.title_native if bull_story else "") or "",
                    title_english=(bull_story.title_english if bull_story else "") or "",
                    summary=(bull_story.summary if bull_story else "") or "",
                    body=bull_body,
                ),
            ))
        for idx, sc in enumerate(shorts_canvases):
            st = sc.stories[0] if sc.stories else None
            if not st:
                continue
            seo_jobs.append((
                f"short#{idx+1}",
                sc,
                seo_provider.SeoInput(
                    kind="short",
                    language=language,
                    title_native=st.title_native or "",
                    title_english=st.title_english or "",
                    summary=st.summary or "",
                ),
            ))

        # Wave 4 item F: SEO calls are SUBMITTED here but COLLECTED only
        # after Stage 3 — the Gemini round-trips overlap with the render
        # instead of serialising in front of it. SEO results aren't an
        # input to the render (they ride on canvas.json + Clip.seo), so
        # the overlap is free wall-time. canvas.json is saved once now
        # (without SEO, so the editor can read it during the render) and
        # RE-saved after SEO lands — its FINAL content is identical to
        # the pre-overlap behaviour.
        seo_executor = None
        seo_futures: list[tuple[str, object, object]] = []   # (label, canvas, future)
        seo_max_workers = 0
        # Render-time SEO gate (operator decision 2026-07-14): SEO is
        # regenerated on demand later anyway (editor Regenerate button,
        # per-channel "Write distinct SEO", POST /clips/{id}/seo/generate,
        # quick-seo), so the render-time Gemini calls are pure spend — and
        # the parallel burst drained the Vertex per-minute quota that the
        # Director needs (429s starved the layout/effects plan mid-render).
        # Default "0" = SKIP. Set KAIZER_V4_SEO_AT_RENDER=1 to restore the
        # old overlap-with-Stage-3 generation. Skipping leaves
        # canvas.seo=None and Clip.seo="" — every consumer handles that:
        # the editor panel shows empty fields + Regenerate, publish falls
        # back to clip.text title + channel footer, and per-channel
        # variants in clip.seo_variants still take precedence once applied.
        if seo_jobs and os.environ.get(
                "KAIZER_V4_SEO_AT_RENDER", "0").strip() != "1":
            _log(f"[v4] SEO at render OFF (KAIZER_V4_SEO_AT_RENDER!=1) -- "
                 f"skipping {len(seo_jobs)} SEO call(s); generate on-demand "
                 f"in the editor / at publish")
            seo_jobs = []
        if seo_jobs:
            # Cap concurrency so a 12-short batch doesn't fan out to 12
            # Gemini calls at once and trip rate limits. 4 is the sweet
            # spot — fast enough to win 3-4x on common batches, slow
            # enough that Vertex AI's per-minute quota stays happy.
            seo_max_workers = min(len(seo_jobs), 4)
            # NOTE: name this `seo_executor`, NOT `pool` — there's an
            # outer `pool` variable (the image-asset list) that Stage 3
            # iterates over later. Naming the executor `pool` shadowed
            # it and Stage 3 then tried `for p in pool` against a
            # closed ThreadPoolExecutor (job 161 crashed this way).
            seo_executor = ThreadPoolExecutor(
                max_workers=seo_max_workers, thread_name_prefix="kaizer-seo",
            )
            for label, target, inp in seo_jobs:
                seo_futures.append(
                    (label, target, seo_executor.submit(seo_provider.generate_seo, inp))
                )
            _log(f"[v4] SEO submitted for {len(seo_jobs)} canvases "
                 f"(parallel x{seo_max_workers}) -- overlapping with Stage 3 render")
            # SEO belt (admin Pipeline Flow): this render's SEO is generating.
            try:
                from services import stage_events as _se
                _se.emit_lane_here("seo", "generate", _se.ENTERED)
            except Exception:
                pass

        job_canvas = V4JobCanvas(
            job_id=job_id,
            language=language,
            bulletin=bulletin_canvas,
            shorts=shorts_canvases,
            output_format=output_format,
            effects_mode=(os.environ.get("KAIZER_V4_EFFECTS_MODE") or "").strip().lower(),
            trimmed_bulletin_path=trim_result.trimmed_path,
            trimmed_shorts_paths=trimmed_shorts_paths,
        )
        canvas_json_path = _save_canvas_json(out_dir, job_canvas)
        _log(f"[v4] Stage 2/3 DONE in {_fmt_elapsed(time.time() - stage_t0)} "
             f"-- canvas.json written, {len(shorts_canvases)} shorts queued  "
             f"(total {_fmt_elapsed(time.time() - job_t0)})")

        # ── Pre-compose creative-quality gate (ADVISORY; adopted from
        # OpenMontage's slideshow/variation scorers, rebuilt on Kaizer's
        # canvas stories). Flag-gated + log-only + fully wrapped: it flags
        # slideshow-feel / monotony (visual starvation, pacing, image reuse...)
        # BEFORE we burn NVENC slots, but NEVER blocks the render — thresholds
        # are still being tuned on real Telugu-news distributions.
        if os.environ.get("KAIZER_PRECOMPOSE_QC", "0").strip() == "1":
            # Layout safety (spec 3.13): media must never cover the
            # headline/ticker strips. Fail LOUD in the log — the clamp at
            # canvas build should make this unreachable, so a hit here
            # means an operator-forced layout or a builder bug.
            try:
                from pipeline_v4 import layout_safety
                _viol = layout_safety.check(
                    job_canvas.bulletin.layout, job_canvas.bulletin.stories)
                for _v in _viol:
                    _log(f"[v4][qc] LAYOUT-SAFETY VIOLATION: {_v}")
            except Exception as _ls_exc:
                _log(f"[v4][qc] layout-safety check skipped: {_ls_exc}")
            try:
                from pipeline_v4.quality import score_canvas_quality
                _q = score_canvas_quality(json.loads(job_canvas.model_dump_json()))
                _b = _q.get("bulletin") or {}
                _log(f"[v4][qc] pre-compose quality: worst={_q.get('worst_verdict')} "
                     f"bulletin={_b.get('verdict')}({_b.get('average')}) "
                     f"shorts=[{', '.join(s.get('verdict', '?') for s in _q.get('shorts', []))}]")
                for _name, _r in ([("bulletin", _b)]
                                  + [(s.get("kind"), s) for s in _q.get("shorts", [])]):
                    if _r and _r.get("verdict") in ("revise", "fail"):
                        _w = max(_r["dimensions"].items(), key=lambda kv: kv[1]["score"])
                        _log(f"[v4][qc] {_name} {_r['verdict'].upper()} "
                             f"(avg {_r['average']}): {_w[0]} -> {_w[1]['reason']}")
            except Exception as _qc_exc:
                _log(f"[v4][qc] pre-compose quality scoring skipped: {_qc_exc}")

        # ─── Stage 3/3 — render bulletin + shorts via V1 layouts ───
        stage_t0 = time.time()
        try:
            from pipeline_v4 import encoder as _enc
            _est = _enc.resolve_status()
            _log(f"[v4] Stage 3/3 START -- video encoder: {_est['backend']} "
                 f"(intent={_est['intent']}, "
                 f"KAIZER_VIDEO_ENCODER={os.environ.get('KAIZER_VIDEO_ENCODER', 'auto')!r}, "
                 f"KAIZER_FORCE_ENCODER={os.environ.get('KAIZER_FORCE_ENCODER', '')!r})")
            if _est["level"] == "error":
                # Forced NVENC but no session — FAIL LOUD here rather than let
                # ffmpeg silently CPU-crawl for minutes. Propagates to the job
                # try/except below → status=failed with this clear message.
                _log(f"[v4] ENCODER ERROR -- {_est['message']}")
                raise RuntimeError(f"encoder unavailable: {_est['message']}")
            if _est["level"] == "warn":
                _log(f"[v4] !! ENCODER WARNING -- {_est['message']}")
        except RuntimeError:
            raise
        except Exception as _enc_exc:
            _log(f"[v4] encoder status check skipped (soft): {_enc_exc}")
        pool_dir = out_dir / "_pool"
        # Absolute paths in the pool — V1 composers expect file paths.
        pool_paths = [str(pool_dir / p["filename"]) for p in pool
                      if (pool_dir / p["filename"]).is_file()]

        # Post-render QC gate (Wave 4 item H). Skippable via KAIZER_V4_QC=0.
        _qc_on = v4_qc.qc_enabled()
        _bull_w = int(getattr(bulletin_canvas.layout, "width", 1920) or 1920)
        _bull_h = int(getattr(bulletin_canvas.layout, "height", 1080) or 1080)
        # Expected bulletin duration = sum of story durations. When an
        # intro reel is configured the final file is intro+crossfade
        # longer — skip the duration check for that case (stream/size/
        # resolution checks still apply).
        _bull_expected_dur: Optional[float] = float(sum(
            (s.video_t_end - s.video_t_start) for s in trim_result.stories
        ) or 0.0) or None
        if ((getattr(bulletin_canvas.layout, "bg_video_path", None) or "")
                and float(getattr(bulletin_canvas.layout, "bg_intro_seconds", 0.0) or 0.0) > 0.0):
            _bull_expected_dur = None
        # Tolerances widen with story count: every stitched segment can
        # carry up to ~1 AAC frame (~23ms) of mux rounding, so a strict
        # 0.5s/0.1s gate would false-positive on 15+ story bulletins.
        _n_stories = max(1, len(trim_result.stories))
        _bull_dur_tol = max(0.5, 0.06 * _n_stories)
        _bull_av_tol = max(0.1, 0.05 * _n_stories)

        # AI DIRECTOR plan — computed ONCE here so the bulletin AND the
        # shorts share the same per-story mood decisions (a crime short
        # gets the same cold grade as its bulletin story). Gated by
        # KAIZER_V4_DIRECTOR (set when effects_mode='rich'); any failure →
        # None → renders proceed exactly as without the Director.
        _stage3_director_plan = None
        try:
            from pipeline_v4.director import (directives_enabled as _dir_on,
                                              parse_style_directives as _parse_dirs,
                                              plan_direction as _plan_dir)
            # The user's per-category effect picks ("edit using THESE").
            # When present they force the Director on even if the render's
            # effects mode wouldn't otherwise (an explicit user direction),
            # and constrain its vocabulary per category.
            _user_picks, _cat_override = _parse_dirs(
                os.environ.get("KAIZER_V4_STYLE_DIRECTIVES"))
            if _dir_on() or _user_picks or _cat_override:
                _wbs_s3 = {}
                try:
                    _sw_s3 = Path(out_dir) / "story_words.json"
                    if _sw_s3.is_file():
                        _wbs_s3 = json.loads(_sw_s3.read_text(encoding="utf-8"))
                        # The sidecar is envelope-wrapped ({"schema":1,
                        # "language":..., "stories":{idx:[words]}}) but the
                        # sensors expect the bare {idx:[words]} map. Passing
                        # the wrapper made sense_pacing hit int("schema") and
                        # skip EVERYTHING — the Director ran with "pacing 0
                        # stories" on every job (job 610 trace).
                        if isinstance(_wbs_s3, dict) and "stories" in _wbs_s3:
                            _wbs_s3 = _wbs_s3.get("stories") or {}
                except Exception:
                    _wbs_s3 = {}
                _dir_kwargs = dict(
                    stories=list(bulletin_canvas.stories),
                    words_by_story=_wbs_s3,
                    category=(_cat_override
                              or os.environ.get("KAIZER_V4_CONTENT_TYPE") or "news"),
                    source_path=trim_result.trimmed_path, language=language,
                    user_picks=(_user_picks or None))
                # DIRECTOR ENGINE pick (Job.v4_director_engine → env):
                # "platform" runs the ported kaizer-platform 3-layer engine
                # (adapter validates onto our vocabulary); any failure or
                # empty plan falls back to OUR V4 engine — a paid job always
                # gets a direction. Default/"v4" = unchanged behavior.
                _deng = (os.environ.get("KAIZER_V4_DIRECTOR_ENGINE", "v4")
                         or "v4").strip().lower()
                if _deng == "platform":
                    try:
                        from pipeline_v4.director_platform import (
                            plan_direction_platform as _plan_platform)
                        _stage3_director_plan = _plan_platform(**_dir_kwargs)
                        print(f"[v4] director engine=platform planned "
                              f"{len(_stage3_director_plan or {})} stories",
                              flush=True)
                    except Exception as _px:
                        print(f"[v4] platform director failed (soft) — "
                              f"falling back to v4 engine: {_px}", flush=True)
                        _stage3_director_plan = None
                if not _stage3_director_plan:
                    if _deng == "platform":
                        print("[v4] platform plan empty — using v4 engine",
                              flush=True)
                    _stage3_director_plan = _plan_dir(**_dir_kwargs)
        except Exception as _dx_s3:
            print(f"[v4] stage-3 director plan skipped (soft): {_dx_s3}", flush=True)
            _stage3_director_plan = None
        # LIVE LAYOUTS gate: when the user PINNED a whole-job layout
        # (an explicit lib:/custom: full-form pick) or the kill switch is
        # set, strip the plan's per-story layout decisions so neither the
        # render nor the canvas seeding ever sees them — the user's
        # explicit choice always wins over the Director.
        if _stage3_director_plan:
            _ffl_gate = (os.environ.get("KAIZER_V4_FULLFORM_LAYOUT", "")
                         or "").strip().lower()
            _ll_off = ((os.environ.get("KAIZER_V4_LIVE_LAYOUTS", "1")
                        or "1").strip().lower() in ("0", "off", "false"))
            if (_ll_off or _ffl_gate.startswith("lib:")
                    or _ffl_gate.startswith("custom:")):
                for _dstrip in _stage3_director_plan.values():
                    _dstrip.layout = ""
                    _dstrip.layout_moments = []
                print("[v4] live layouts disabled for this job "
                      f"(gate={'kill-switch' if _ll_off else _ffl_gate})",
                      flush=True)
        # Per-short effects -vf resolver: the short's story mood grade (+
        # garnish) when the Director planned it, else the effects-mode base
        # grade. '' when effects are off → the short renders byte-identically.
        _efx_mode_s3 = (os.environ.get("KAIZER_V4_EFFECTS_MODE") or "").strip().lower()
        def _short_effects_vf(story_index) -> str:
            if _efx_mode_s3 in ("", "off", "none", "legacy"):
                return ""
            try:
                from pipeline_v4.director import story_fx_chain as _story_fx
                d = _stage3_director_plan.get(int(story_index)) if _stage3_director_plan else None
                if d is not None:
                    return _story_fx(d)
                return v1_bridge._effects_vf_from_env()
            except Exception:
                return ""

        # Bulletin: V1 broadcast layout (left video / right sidebar /
        # animated lower-third / scrolling ticker).
        bull_t0 = time.time()
        # Set on ANY full-video failure so the job can finish "done" for
        # the shorts while still TELLING the user the full video is
        # missing (job 600 rendered 8 shorts, silently dropped the full
        # video, and the UI claimed everything was fine).
        _bulletin_fail_note = ""
        try:
            user_d = _load_user_defaults(job_id)
            channel_name = (
                (user_d.get("brand_suffix") or "")
                .lstrip(" |·-_•").strip()
            )  # strip any " | " / " - " prefix the user typed in defaults

            # Watermark is NOT baked in here — render produces a clean
            # file. The upload worker (pipeline_v4.watermark.stamp_for_
            # channel) stamps each destination's own logo + text at
            # publish time so the same source can ship to N channels
            # with N different brand stamps.
            def _render_bulletin_once() -> str:
                _out = str(out_dir / bulletin_canvas.output_filename)
                # TRAILER-ONLY jobs: the "full video" IS the trailer. The
                # 16:9 cut lands as bulletin.mp4 so every downstream step
                # (clips, preview, SEO, publish, branding) works unchanged;
                # a 9:16 companion lands beside it for Shorts/Reels.
                if output_format == "trailer-only":
                    from pipeline_v4 import trailer as _tr
                    _wbs = {str(s.story_index): list(getattr(s, "words", []) or [])
                            for s in trim_result.stories}
                    _style = (os.environ.get("KAIZER_V4_TRAILER_STYLE") or "auto")
                    _plan = _tr.plan_trailer_moments(
                        stories=bulletin_canvas.stories, words_by_story=_wbs,
                        duration=trim_result.trimmed_duration_sec,
                        language=language)
                    _tr.render_trailer(
                        source_path=trim_result.trimmed_path,
                        stories=bulletin_canvas.stories, words_by_story=_wbs,
                        out_path=_out, work_dir=out_dir / "_trailer",
                        aspect="16:9", language=language,
                        channel_name=channel_name, plan=_plan, style=_style)
                    try:
                        _tr.render_trailer(
                            source_path=trim_result.trimmed_path,
                            stories=bulletin_canvas.stories, words_by_story=_wbs,
                            out_path=str(out_dir / "trailer_9x16.mp4"),
                            work_dir=out_dir / "_trailer", aspect="9:16",
                            language=language, channel_name=channel_name,
                            plan=_plan, style=_style)
                    except Exception as _t9exc:
                        _log(f"[v4] 9:16 trailer companion failed (soft): {_t9exc}")
                    try:
                        (out_dir / "render_report.json").write_text(
                            json.dumps({"bulletin_method": "trailer_engine",
                                        "trailer_style": _style}), encoding="utf-8")
                    except Exception:
                        pass
                    return _out
                # "Full form video" custom template: render the FULL trimmed video
                # through the verified custom engine at the template's own (16:9) canvas,
                # instead of the built-in multi-story bulletin composer.
                _ff = (os.environ.get("KAIZER_V4_FULLFORM_LAYOUT", "") or "").strip().lower()
                if _ff.startswith("custom:") and _ff.split(":", 1)[1].isdigit():
                    _log(f"[v4] full-form custom template {_ff}")
                    _ffm, _ffmain = _load_template_media(job_id)
                    _ffov = _load_template_overrides(job_id)
                    # Feed the template's text/image slots the REAL story content (not just
                    # the channel name) so headline/ticker/image slots fill instead of
                    # leaking the author's placeholders. Mirrors what render_bulletin gets.
                    _ff_stories = [{
                        "headline": (s.title_native or s.title_english or ""),
                        "headline_alt": (s.title_english or ""),
                        "subtitle": "",
                        "body": (s.summary or ""),
                        "kicker": "",
                        "images": [],
                    } for s in trim_result.stories]
                    _ff_first = _ff_stories[0] if _ff_stories else {}
                    v1_bridge.render_short(v1_bridge.ShortRenderInputs(
                        trimmed_short_path=trim_result.trimmed_path,
                        title_text=(channel_name or ""),
                        output_path=_out,
                        work_dir=out_dir,
                        layout=_ff,
                        language=language,
                        brand_logo=brand_logo,
                        watermark_text="",
                        watermark_opacity=0.0,
                        watermark_position="top-right",
                        template_media=_ffm,
                        main_media_slot=_ffmain,
                        template_text_overrides=_ffov,
                        expected_kind="full",   # crash-guard: reject a short template here
                        headline=_ff_first.get("headline") or "",
                        headline_alt=_ff_first.get("headline_alt") or "",
                        body=_ff_first.get("body") or "",
                        support_images=list(pool_paths or []),
                        stories=_ff_stories,
                        # per-job visual edit (inline builder) -> render verbatim if present
                        template_html_override=_load_html_override(job_id, "bulletin", 0),
                        # Director/effects-mode grade for the template FOOTAGE only (the
                        # design overlay stays crisp): whole-video mode-based chain — the
                        # KAIZER_V4_EFFECTS_MODE env is set in this orchestrator subprocess.
                        # "" (mode off) keeps the render byte-identical to legacy.
                        effects_vf=v1_bridge._effects_vf_from_env(),
                    ))
                    try:
                        (out_dir / "render_report.json").write_text(
                            json.dumps({"bulletin_method": "custom_template",
                                        "fullform_layout": _ff}), encoding="utf-8")
                    except Exception:
                        pass
                    return _out
                try:
                    (out_dir / "render_report.json").write_text(
                        json.dumps({"bulletin_method": "builtin_composer",
                                    "fullform_layout": _ff or ""}), encoding="utf-8")
                except Exception:
                    pass
                _t_spd, _t_col = v1_bridge.extract_ticker_overrides(bulletin_canvas.stories)
                # Render from the CANVAS stories (same shape the editor's
                # re-render uses) — the initial output already carries the
                # engine-decided image windows, gaps, spotlight/PiP and
                # name-straps instead of a static sidebar. Previously the
                # raw TrimmedStory list (no .images) was passed here, so
                # the first render silently ignored the canvas timings.
                from types import SimpleNamespace as _NS
                _render_stories = [
                    _NS(
                        title_native=cs.title_native,
                        title_english=cs.title_english,
                        summary=cs.summary,
                        video_t_start=cs.video_t_start,
                        video_t_end=cs.video_t_end,
                        images=list(cs.images or []),
                        text_blocks=list(cs.text_blocks or []),
                        name_strap=getattr(cs, "name_strap", False),
                        story_index=cs.story_index,
                    )
                    for cs in bulletin_canvas.stories
                ]
                # AI DIRECTOR (gated): computed once at Stage-3 scope and
                # shared with the shorts loop so both get the same per-story
                # moods. See _stage3_director_plan above.
                _director_plan = _stage3_director_plan
                return v1_bridge.render_bulletin(v1_bridge.BulletinRenderInputs(
                    trimmed_bulletin_path=trim_result.trimmed_path,
                    stories=_render_stories,
                    output_path=_out,
                    work_dir=out_dir,
                    language=language,
                    brand_logo=brand_logo,
                    sidebar_images=pool_paths,
                    layout=bulletin_canvas.layout,
                    channel_name=channel_name,
                    watermark_text="",
                    watermark_opacity=0.0,
                    watermark_position="top-right",
                    ticker_speed_s=_t_spd,
                    ticker_bg_color=_t_col,
                    story_transition=getattr(bulletin_canvas, "story_transition", None),
                    effects_mode=(os.environ.get("KAIZER_V4_EFFECTS_MODE") or "").strip().lower(),
                    directives=_director_plan,
                ))

            # Output-format gate (shorts-only): skip the full-video render
            # entirely. bulletin_out="" -> _materialise_clips emits no
            # bulletin Clip, so the publish modal shows only the shorts.
            if output_format == "shorts-only":
                _log("[v4] output_format=shorts-only -- skipped bulletin/full-video render")
                bulletin_out = ""
            elif defer_render:
                _log("[v4] deferred -- skipping up-front full-video render (export in the editor)")
                bulletin_out = ""
            else:
                bulletin_out = _render_bulletin_once()
            if _qc_on and bulletin_out:
                _viols = v4_qc.verify_render(
                    bulletin_out,
                    expected_duration=_bull_expected_dur,
                    expected_w=_bull_w, expected_h=_bull_h,
                    tolerance_s=_bull_dur_tol,
                    av_sync_tolerance_s=_bull_av_tol,
                )
                if _viols:
                    _log(f"[v4] QC violations on bulletin "
                         f"({'; '.join(_viols)}) -- retrying render once")
                    bulletin_out = _render_bulletin_once()
                    _viols = v4_qc.verify_render(
                        bulletin_out,
                        expected_duration=_bull_expected_dur,
                        expected_w=_bull_w, expected_h=_bull_h,
                        tolerance_s=_bull_dur_tol,
                        av_sync_tolerance_s=_bull_av_tol,
                    )
                    if _viols:
                        _log(f"[v4] QC FAILED bulletin: {'; '.join(_viols)}")
                        bulletin_out = ""
                        _bulletin_fail_note = "QC: " + "; ".join(_viols)[:200]
            if bulletin_out:
                _log(f"[v4] bulletin rendered in {_fmt_elapsed(time.time() - bull_t0)} -> {bulletin_out}")
        except Exception as exc:
            _log(f"[v4] bulletin render failed after "
                 f"{_fmt_elapsed(time.time() - bull_t0)}: {exc}")
            bulletin_out = ""
            _bulletin_fail_note = str(exc)[:300]

        # Shorts: pull every V1-parity knob from sc.short_config so the
        # editor's controls drive the render. Title falls back to the
        # parent story when the user hasn't typed an override.
        #
        # Wave 4 item C: shorts render through a bounded ThreadPool
        # (KAIZER_V4_RENDER_CONCURRENCY, default 3) instead of
        # sequentially. Each short keeps its own try/except so one
        # failure never kills the others, and the log line format is
        # UNCHANGED ("[v4] short N rendered in ...") because the
        # frontend parses it.
        # Custom-template per-slot media (resolved once): {slot: path} + main slot.
        _custom_media, _custom_main_slot = _load_template_media(job_id)
        _custom_overrides = _load_template_overrides(job_id)
        def _render_one_short(s_idx: int, sc) -> None:
            short_t0 = time.time()
            cfg = sc.short_config
            story0 = sc.stories[0] if sc.stories else None
            story_title = ""
            if story0:
                story_title = (story0.title_native or story0.title_english or "").strip()
            title = (cfg.text if cfg and cfg.text else story_title) or "KAIZER X"

            # Image: editor's choice first, then round-robin fallback.
            if cfg and cfg.image_filename:
                cand = pool_dir / cfg.image_filename
                short_image = str(cand) if cand.is_file() else None
            else:
                short_image = pool_paths[s_idx % len(pool_paths)] if pool_paths else None

            def _do_render() -> str:
                return v1_bridge.render_short(v1_bridge.ShortRenderInputs(
                    trimmed_short_path=sc.trimmed_video_path,
                    title_text=title,
                    output_path=str(out_dir / sc.output_filename),
                    work_dir=out_dir,
                    layout=(cfg.layout if cfg else v1_bridge.DEFAULT_SHORTS_LAYOUT),
                    language=language,
                    image_path=short_image,
                    brand_logo=brand_logo,
                    thumbnail_path=short_image,
                    font_file=(cfg.font_file if cfg else None),
                    font_size=(cfg.font_size if cfg else None),
                    text_color=(cfg.text_color if cfg else None),
                    section_pct=(cfg.section_pct.model_dump() if cfg else None),
                    card_style=(cfg.card_style.model_dump() if cfg else None),
                    follow_params=(cfg.follow_params.model_dump() if cfg else None),
                    # dual_video layout: the audio-first reference b-roll (if
                    # any) becomes the second cam; absent → torn_card fallback.
                    second_video_path=(os.environ.get("KAIZER_V4_REF_VIDEO_PATH") or None),
                    # Clean render — per-channel stamping happens in the
                    # upload worker. See orchestrator.py bulletin block.
                    watermark_text="",
                    watermark_opacity=0.0,
                    watermark_position="top-right",
                    template_media=_custom_media,
                    main_media_slot=_custom_main_slot,
                    template_text_overrides=_custom_overrides,
                    expected_kind="short",   # crash-guard: reject a full-form template here
                    # Real story content for custom-template slots (filler maps these so a
                    # headline/ticker/image slot never shows the author's placeholder).
                    headline=title,
                    headline_alt=((story0.title_english or "") if story0 else None),
                    body=((story0.summary or "") if story0 else None),
                    support_images=([short_image] if short_image else None),
                    # per-job visual edit (inline builder) -> render verbatim if present
                    template_html_override=_load_html_override(job_id, "short", s_idx),
                    # Phase 2: per-story mood grade so the short matches its
                    # bulletin story's look ('' when effects are off).
                    effects_vf=_short_effects_vf(getattr(story0, "story_index", s_idx)),
                ))

            try:
                short_out = _do_render()
                if _qc_on:
                    _exp = (story0.video_t_end - story0.video_t_start) if story0 else None
                    _sviols = v4_qc.verify_render(
                        short_out, expected_duration=_exp,
                        expected_w=1080, expected_h=1920,
                    )
                    if _sviols:
                        _log(f"[v4] QC violations on short {s_idx + 1} "
                             f"({'; '.join(_sviols)}) -- retrying render once")
                        short_out = _do_render()
                        _sviols = v4_qc.verify_render(
                            short_out, expected_duration=_exp,
                            expected_w=1080, expected_h=1920,
                        )
                        if _sviols:
                            _log(f"[v4] QC FAILED short {s_idx + 1}: {'; '.join(_sviols)}")
                            # Quarantine the artefact so _materialise_clips
                            # (which scans by canvas output_filename) skips it.
                            try:
                                _p = Path(short_out)
                                if _p.is_file():
                                    _p.replace(_p.with_name(_p.stem + "_qcfailed.mp4"))
                            except OSError:
                                pass
                            return
                _log(f"[v4] short {s_idx + 1} rendered in "
                     f"{_fmt_elapsed(time.time() - short_t0)} -> {short_out}")
            except Exception as exc:
                _log(f"[v4] short {s_idx + 1}: render failed after "
                     f"{_fmt_elapsed(time.time() - short_t0)}: {exc}")

        if shorts_canvases and defer_render:
            _log(f"[v4] deferred -- skipping up-front render of {len(shorts_canvases)} short(s) "
                 f"(export in the editor)")
        elif shorts_canvases:
            _short_workers = min(_render_workers, len(shorts_canvases))
            if _short_workers > 1:
                with ThreadPoolExecutor(
                    max_workers=_short_workers, thread_name_prefix="v4-short",
                ) as _short_pool:
                    # map() drains every task; exceptions are already
                    # swallowed inside _render_one_short.
                    list(_short_pool.map(
                        _render_one_short,
                        range(len(shorts_canvases)),
                        shorts_canvases,
                    ))
            else:
                for s_idx, sc in enumerate(shorts_canvases):
                    _render_one_short(s_idx, sc)

        _log(f"[v4] Stage 3/3 DONE in {_fmt_elapsed(time.time() - stage_t0)}  "
             f"(total {_fmt_elapsed(time.time() - job_t0)})")

        # ─── Collect the overlapped SEO results (Wave 4 item F) ────
        # The futures were submitted before Stage 3; most (usually all)
        # are already done by now, so this join costs ~0s. Per-call
        # soft-fail semantics are identical to the old serial block.
        if seo_futures:
            seo_t0 = time.time()
            succeeded = 0
            failed = 0
            for label, target, fut in seo_futures:
                try:
                    result = fut.result(timeout=600)
                except Exception as exc:  # noqa: BLE001 — soft-fail per call
                    failed += 1
                    _log(f"[v4] SEO failed for {label} (soft-skip): {exc}")
                    continue
                try:
                    target.seo = CanvasSEO(**result)
                    succeeded += 1
                except Exception as schema_exc:  # noqa: BLE001
                    failed += 1
                    _log(f"[v4] SEO schema rejected for {label}: {schema_exc}")
            if seo_executor is not None:
                seo_executor.shutdown(wait=False)
            _log(f"[v4] SEO generated for {succeeded}/{len(seo_futures)} canvases "
                 f"(parallel x{seo_max_workers}, {failed} failed) "
                 f"in {_fmt_elapsed(time.time() - seo_t0)}")
            # SEO belt: generate done → score (the checker ran inside) → done.
            try:
                from services import stage_events as _se
                _se.emit_lane_here("seo", "generate", _se.EXITED)
                _se.emit_lane_here("seo", "score", _se.ENTERED)
                _se.emit_lane_here("seo", "score", _se.EXITED)
            except Exception:
                pass

        # ── Source-preserved override: operator's description WINS ──
        # When the operator supplied a verbatim description at submit
        # time, swap it into the bulletin SEO so the YouTube
        # description is the operator's exact words. Title + tags +
        # hashtags are left as Gemini drafted them — operator can
        # re-edit those in the editor if needed. Shorts SEO stays
        # AI-generated because the operator only supplied one
        # bulletin-level description, not per-short text.
        # SEO-at-render skipped: still carry the operator's verbatim
        # description into canvas.json + clip.seo via an empty SEO shell
        # (title/tags stay empty until an on-demand generate fills them).
        if predef_description and bulletin_canvas.seo is None:
            bulletin_canvas.seo = CanvasSEO()
            # A TITLE-LESS seo dict is discarded wholesale at publish
            # (upload_dispatch._compose_metadata gates on title), which
            # would silently drop the operator's verbatim description.
            # Seed the title from the first story so the shell survives
            # to YouTube; the on-demand generate can still improve it.
            try:
                _s0 = (bulletin_canvas.stories or [None])[0]
                bulletin_canvas.seo.title = ((getattr(_s0, "title_english", "")
                                              or getattr(_s0, "title_native", "")
                                              or "").strip()[:95])
            except Exception:
                pass
        if predef_description and bulletin_canvas.seo:
            try:
                bulletin_canvas.seo.description = predef_description
                bulletin_canvas.seo.edited_by_user = True
                _log(f"[v4] bulletin SEO description overridden with "
                     f"operator-supplied text ({len(predef_description)} chars)")
            except Exception as exc:
                _log(f"[v4] failed to apply predef description override (soft): {exc}")

        # Persist the AI-Director's broadcast graphics (overlays / HUD) onto
        # each story so the EDITOR can change their text / timing / which
        # graphic and re-render. Seed once with each graphic's default text;
        # never clobber a story that already carries overlays (an editor edit
        # on a re-run). Empty plan (effects off / defer without director) →
        # no-op, renderer keeps its live-Director fallback (byte-identical).
        try:
            from pipeline_v4.overlays import overlay_text_fields as _otf
            from pipeline_v4.canvas_schema import CanvasOverlay as _CO
            if _stage3_director_plan:
                for _st in bulletin_canvas.stories:
                    if getattr(_st, "overlays", None):
                        continue
                    _dvp = _stage3_director_plan.get(int(_st.story_index))
                    if not _dvp or not getattr(_dvp, "overlays", None):
                        continue
                    _seeded = []
                    for _o in (_dvp.overlays or [])[:3]:
                        _oid = str(_o.get("id", "") or "")
                        if not _oid:
                            continue
                        _seeded.append(_CO(id=_oid,
                                           t=float(_o.get("t", 0.8) or 0.8),
                                           fields=dict(_otf(_oid))))
                    if _seeded:
                        _st.overlays = _seeded
        except Exception as _oxe:
            _log(f"[v4] overlay seed into canvas skipped (soft): {_oxe}")

        # Persist the AI-Director's LIVE-LAYOUT decisions onto each story
        # (same discipline as overlays: seed once, never clobber an
        # editor's persisted value; empty plan → no-op, byte-identical).
        try:
            from pipeline_v4.canvas_schema import CanvasLayoutMoment as _CLM
            if _stage3_director_plan:
                for _st in bulletin_canvas.stories:
                    _dvp = _stage3_director_plan.get(int(_st.story_index))
                    if not _dvp:
                        continue
                    if (not getattr(_st, "layout_key", "")
                            and getattr(_dvp, "layout", "")):
                        _st.layout_key = _dvp.layout
                    if (not getattr(_st, "layout_moments", None)
                            and getattr(_dvp, "layout_moments", None)):
                        _st.layout_moments = [
                            _CLM(layout=str(_m.get("layout", "") or ""),
                                 t=float(_m.get("t", 0.0) or 0.0),
                                 dur=float(_m.get("dur", 4.0) or 4.0),
                                 transition=(_m.get("transition", "push")
                                             or "push"))
                            for _m in (_dvp.layout_moments or [])
                            if isinstance(_m, dict) and _m.get("layout")]
        except Exception as _lxe:
            _log(f"[v4] layout seed into canvas skipped (soft): {_lxe}")

        # Re-save canvas.json now that SEO (+ the predef override) has
        # landed. Rebuilding V4JobCanvas from the same canvases makes
        # the final file content identical to the pre-overlap flow,
        # regardless of whether pydantic copied the sub-models at the
        # first construction.
        job_canvas = V4JobCanvas(
            job_id=job_id,
            language=language,
            bulletin=bulletin_canvas,
            shorts=shorts_canvases,
            output_format=output_format,
            effects_mode=(os.environ.get("KAIZER_V4_EFFECTS_MODE") or "").strip().lower(),
            trimmed_bulletin_path=trim_result.trimmed_path,
            trimmed_shorts_paths=trimmed_shorts_paths,
        )
        canvas_json_path = _save_canvas_json(out_dir, job_canvas)

        # ─── Materialise each rendered output as a Clip row ────────
        # The existing publish / YouTube-upload pipeline reads Clip rows.
        # Without this V4 outputs would be invisible to the publish
        # modal even though they're sitting on disk. SEO from canvas.json
        # rides into clip.seo so _compose_metadata picks it up unchanged.
        try:
            n_clips = _materialise_clips(
                job_id=job_id,
                out_dir=out_dir,
                bulletin_canvas=bulletin_canvas,
                shorts_canvases=shorts_canvases,
                # Deferred: point the bulletin clip at its FUTURE output path so the editor's
                # export writes there and publish finds it (the file just doesn't exist yet).
                bulletin_path=(str(out_dir / bulletin_canvas.output_filename)
                               if (defer_render and output_format != "shorts-only")
                               else bulletin_out),
                defer=defer_render,
            )
            _log(f"[v4] materialised {n_clips} Clip row(s) for publish")
        except Exception as exc:
            _log(f"[v4] clip materialisation failed (soft-skip): {exc}")

        # ─── Auto per-channel SEO (YouTube-duplication bypass) ─────
        # For a multi-channel job, write a DISTINCT title/description/tags per
        # chosen channel onto each clip NOW, so every channel publishes with its
        # own SEO without the operator clicking "Apply per-channel". Idempotent,
        # fail-soft, and a no-op for single-channel / shared-mode jobs. Runs
        # BEFORE auto-publish so an auto-published job already carries the
        # per-channel variants. (KAIZER_V4_AUTO_CHANNEL_SEO=0 to disable.)
        try:
            from seo.auto_channel_seo import auto_generate_channel_seo_for_job
            auto_generate_channel_seo_for_job(job_id, language=language, log=_log)
        except Exception as exc:
            _log(f"[v4] auto-channel-seo soft-skip: {exc}")

        # ─── Optional auto-publish ────────────────────────────────
        # If the user enabled auto_publish (and didn't require an
        # explicit consent click), enqueue uploads now. With consent
        # required, we leave the publish for the user to confirm in
        # the editor's "Ready to publish" banner.
        if defer_render:
            _log("[v4] deferred -- skipping auto-publish (export in the editor, then publish)")
        else:
            try:
                _maybe_auto_publish(job_id, _log)
            except Exception as exc:
                _log(f"[v4] auto-publish soft-skip: {exc}")

        _belt("compose", "exited")
        _update_job_status(
            job_id, status="done",
            current_stage=("render_deferred" if defer_render else "done"),
            error=(("Full video render FAILED: " + _bulletin_fail_note
                    + " — the shorts are ready; re-render the full video "
                      "from the canvas editor.")
                   if _bulletin_fail_note else ""))
        _log(f"[v4] job {job_id} DONE -- total {_fmt_elapsed(time.time() - job_t0)}")

        # Machine-readable markers the runner picks up
        print(f"[kaizer:v4:canvas] {canvas_json_path}", flush=True)
        print(f"[kaizer:v4:bulletin] {bulletin_out}", flush=True)
        return str(canvas_json_path)

    except Exception as exc:
        tb = traceback.format_exc()
        _log(f"[v4] FAILED after {_fmt_elapsed(time.time() - job_t0)}: {exc}")
        _log(tb)
        _update_job_status(job_id, status="failed", error=str(exc)[:1500])
        raise
    finally:
        # Always release the NVENC gate, whether the job succeeded, failed,
        # or was cancelled mid-compose — never leak an encoder slot.
        _gate_release(_compose_tok)


# ─── CLI ────────────────────────────────────────────────────────────

def _cli() -> int:
    ap = argparse.ArgumentParser(description="V4 pipeline runner")
    ap.add_argument("--job-id", type=int, required=True)
    ap.add_argument("--source", required=True, help="path to source video")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--language", default="te")
    ap.add_argument("--brand-logo", default=None)
    args = ap.parse_args()

    bulletin_images_env = os.environ.get("KAIZER_BULLETIN_IMAGES", "")

    try:
        run_job(
            job_id=args.job_id,
            source_video=args.source,
            output_dir=args.output_dir,
            language=args.language,
            brand_logo=args.brand_logo,
            bulletin_images_env=bulletin_images_env,
        )
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(_cli())
