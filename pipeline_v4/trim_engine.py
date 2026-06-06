"""V4 Step 1 — atomic trim+concat producing a drift-free trimmed video.

Pipeline:
  1. Extract MP3 audio from source video
  2. Deepgram nova-3 word-level transcription
  3. Claude marks each word as KEEP/CUT and groups into stories
  4. ONE ffmpeg filter_complex pass:
       - trim each KEEP span from BOTH video and audio
       - audio fade-in/out at boundaries to mask the joins
       - concat all spans into one video + one audio stream
       - libx264 + aac re-encode in lockstep → A/V locked at sample-level
     Output: trimmed_bulletin.mp4 (or trimmed_short_NN.mp4)
  5. Compute video_t_start / video_t_end PER STORY in the OUTPUT
     timeline — the canvas engine uses these to know where each
     story's footage sits in the trimmed video.

Lipsync contract: Step 2 NEVER re-cuts the trimmed video. It only
overlays. Audio passes through with -c:a copy. As long as Step 1
emits a properly-muxed file, lipsync cannot drift.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Add the backend root to sys.path so we can import sibling modules
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from pipeline_v4 import prompts as v4_prompts


@dataclass
class KeptSpan:
    start_sec: float
    end_sec: float
    reason: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class TrimmedStory:
    """One story in the OUTPUT (trimmed) timeline."""
    story_index: int
    title_native: str
    title_english: str
    summary: str
    video_t_start: float    # where in trimmed.mp4 this story begins
    video_t_end: float      # where in trimmed.mp4 this story ends
    source_spans: list[KeptSpan] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.video_t_end - self.video_t_start)


@dataclass
class TrimResult:
    trimmed_path: str
    trimmed_duration_sec: float
    stories: list[TrimmedStory]
    source_duration_sec: float
    removed_sec_total: float


# ─── Step 1.A — audio extract ───────────────────────────────────────

def _extract_audio_mp3(video_path: str, out_mp3: str, *, ffmpeg_bin: str = "ffmpeg") -> str:
    """Pull mono 22kHz mp3 from the source video. Deepgram doesn't need
    high-bitrate; this keeps the upload payload small."""
    cmd = [
        ffmpeg_bin, "-y", "-v", "error",
        "-i", video_path,
        "-vn", "-ac", "1", "-ar", "22050",
        "-c:a", "libmp3lame", "-q:a", "5",
        out_mp3,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"audio extract failed: {proc.stderr[-400:]}")
    return out_mp3


# ─── Step 1.B — Deepgram words ──────────────────────────────────────

def _deepgram_words(audio_path: str, language: str = "multi") -> tuple[list[dict], float]:
    """Run Deepgram nova-3. Returns (word_array, duration_sec).

    Each word: {"i": int, "w": str, "s": float, "e": float}.
    Same shape V3 uses — reused so the prompt format is consistent.
    """
    from deepgram import DeepgramClient

    api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY not set in env")
    dg = DeepgramClient(api_key=api_key)
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    kwargs = {
        "request":     audio_bytes,
        "model":       "nova-3",
        "punctuate":   True,
        "diarize":     False,
        "smart_format": True,
    }
    if language:
        kwargs["language"] = language
    response = dg.listen.v1.media.transcribe_file(**kwargs)

    metadata = getattr(response, "metadata", None)
    audio_dur = float(getattr(metadata, "duration", 0.0) or 0.0)
    results = getattr(response, "results", None)
    channels = getattr(results, "channels", None) or []
    if not channels:
        return [], audio_dur
    alts = getattr(channels[0], "alternatives", None) or []
    if not alts:
        return [], audio_dur
    raw_words = getattr(alts[0], "words", None) or []
    words: list[dict] = []
    for w in raw_words:
        text = (
            getattr(w, "punctuated_word", None)
            or getattr(w, "word", "")
            or ""
        ).strip()
        if not text:
            continue
        words.append({
            "i": len(words),
            "w": text,
            "s": float(getattr(w, "start", 0.0) or 0.0),
            "e": float(getattr(w, "end", 0.0) or 0.0),
        })
    return words, audio_dur


# ─── Step 1.C — Claude KEEP/CUT plan ────────────────────────────────

def _target_duration_window(src_sec: float) -> tuple[float, float]:
    """Same heuristic the teammate uses — duration-tiered trim windows."""
    m = src_sec / 60.0
    if m < 3:    return (src_sec * 0.7, src_sec * 0.9)
    if m <= 4:   return (110, 130)
    if m <= 6:   return (120, 180)
    if m <= 10:  return (180, 210)
    if m <= 20:  return (240, 480)
    return (240, 600)


def _strip_code_fences(text: str) -> str:
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _claude_keep_cut_plan(
    *,
    words: list[dict],
    language: str,
    duration_sec: float,
) -> tuple[list[dict], float]:
    """Call Claude with the KEEP/CUT prompt. Returns (stories, removed_sec)."""
    from anthropic import Anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in env")
    client = Anthropic(api_key=api_key)

    target_min, target_max = _target_duration_window(duration_sec)
    user_prompt = v4_prompts.build_keep_cut_user_prompt(
        words=words,
        target_min_sec=target_min,
        target_max_sec=target_max,
        language=language,
        duration_sec=duration_sec,
    )

    msg = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=8192,
        system=v4_prompts.KEEP_CUT_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = msg.content[0].text if msg.content else ""
    raw = _strip_code_fences(raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        # Try to recover: find the outer { ... } block
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            raise RuntimeError(f"Claude returned non-JSON: {raw[:400]}") from exc
        data = json.loads(m.group(0))

    stories = data.get("stories") or []
    removed = float(data.get("removed_sec_total") or 0.0)
    return stories, removed


# ─── Step 1.D — atomic ffmpeg trim+concat ───────────────────────────

def _build_filter_complex(spans: list[KeptSpan], *, audio_fade_sec: float = 0.05) -> str:
    """Build the ffmpeg filter_complex string that trims each span
    from both video and audio, applies a brief audio fade at every
    boundary, and concats everything into [vout] + [aout].
    """
    parts: list[str] = []
    for i, sp in enumerate(spans):
        # Video trim — setpts resets the clock so concat doesn't see
        # the source timestamps.
        parts.append(
            f"[0:v]trim=start={sp.start_sec:.3f}:end={sp.end_sec:.3f},"
            f"setpts=PTS-STARTPTS[v{i}]"
        )
        dur = sp.duration
        fade_in = f"afade=t=in:st=0:d={audio_fade_sec:.3f}"
        fade_out_at = max(0.0, dur - audio_fade_sec)
        fade_out = f"afade=t=out:st={fade_out_at:.3f}:d={audio_fade_sec:.3f}"
        parts.append(
            f"[0:a]atrim=start={sp.start_sec:.3f}:end={sp.end_sec:.3f},"
            f"asetpts=PTS-STARTPTS,{fade_in},{fade_out}[a{i}]"
        )
    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(spans)))
    parts.append(f"{concat_inputs}concat=n={len(spans)}:v=1:a=1[vout][aout]")
    return ";".join(parts)


def atomic_trim_concat(
    *,
    source_video: str,
    spans: list[KeptSpan],
    output_path: str,
    ffmpeg_bin: str = "ffmpeg",
) -> None:
    """Run the single ffmpeg pass that produces a trimmed video.

    Public entry — used by both Step 1 (the bulletin trim) and the
    orchestrator's per-short trim (one short = one story's spans).
    Always emits a single, drift-free mp4 via one filter_complex pass.
    """
    if not spans:
        raise RuntimeError("no kept spans -- nothing to trim")

    filter_complex = _build_filter_complex(spans)

    cmd = [
        ffmpeg_bin, "-y", "-v", "error",
        "-i", source_video,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "21",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-movflags", "+faststart",
        output_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60 * 30)
    if proc.returncode != 0:
        raise RuntimeError(f"atomic trim+concat failed: {proc.stderr[-800:]}")


# Back-compat alias for any internal call sites that still use the
# underscore-prefixed name.
_atomic_trim_concat = atomic_trim_concat


# ─── Public entry ──────────────────────────────────────────────────

def run_step1(
    *,
    source_video: str,
    output_dir: str,
    language: str = "multi",
    output_filename: str = "trimmed_bulletin.mp4",
    audio_fade_sec: float = 0.05,
) -> TrimResult:
    """End-to-end Step 1. Returns paths + the trimmed-timeline story
    layout that Step 2 will consume.
    """
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    # 1) Extract audio
    audio_mp3 = str(output_dir_p / "_step1_audio.mp3")
    print(f"[v4/step1] extracting audio ...", flush=True)
    _extract_audio_mp3(source_video, audio_mp3)

    # 2) Deepgram words
    print(f"[v4/step1] Deepgram nova-3 transcribing ({language}) ...", flush=True)
    words, src_duration = _deepgram_words(audio_mp3, language=language)
    print(f"[v4/step1]   got {len(words)} words across {src_duration:.1f}s source", flush=True)
    try:
        os.unlink(audio_mp3)
    except OSError:
        pass

    # 3) Claude KEEP/CUT plan
    print(f"[v4/step1] Claude opus-4-7 planning KEEP/CUT ...", flush=True)
    claude_stories, removed_sec = _claude_keep_cut_plan(
        words=words,
        language=language,
        duration_sec=src_duration,
    )
    print(f"[v4/step1]   {len(claude_stories)} stories, ~{removed_sec:.1f}s removed", flush=True)

    # Flatten kept_spans into one ordered list (preserving story
    # grouping for the output timeline).
    all_spans: list[KeptSpan] = []
    stories_out: list[TrimmedStory] = []
    cursor = 0.0          # current position in the OUTPUT timeline
    for s_idx, s in enumerate(claude_stories):
        s_start_in_output = cursor
        story_spans: list[KeptSpan] = []
        for sp in (s.get("kept_spans") or []):
            try:
                ss = float(sp.get("start_sec") or 0.0)
                ee = float(sp.get("end_sec") or 0.0)
            except (TypeError, ValueError):
                continue
            if ee <= ss + 0.05:
                continue
            sp_obj = KeptSpan(start_sec=ss, end_sec=ee, reason=str(sp.get("reason") or "")[:40])
            story_spans.append(sp_obj)
            all_spans.append(sp_obj)
            cursor += sp_obj.duration
        if not story_spans:
            continue
        stories_out.append(TrimmedStory(
            story_index=s_idx,
            title_native=str(s.get("title_native") or "")[:200],
            title_english=str(s.get("title_english") or "")[:200],
            summary=str(s.get("summary") or "")[:500],
            video_t_start=s_start_in_output,
            video_t_end=cursor,
            source_spans=story_spans,
        ))

    if not all_spans:
        raise RuntimeError("Claude returned no kept spans -- nothing to render")

    # 4) Atomic ffmpeg pass
    output_path = str(output_dir_p / output_filename)
    print(f"[v4/step1] atomic ffmpeg trim+concat ({len(all_spans)} spans) -> {output_filename}", flush=True)
    _atomic_trim_concat(
        source_video=source_video,
        spans=all_spans,
        output_path=output_path,
    )

    trimmed_dur = cursor
    print(f"[v4/step1]   done -- {trimmed_dur:.1f}s trimmed output", flush=True)

    return TrimResult(
        trimmed_path=output_path,
        trimmed_duration_sec=trimmed_dur,
        stories=stories_out,
        source_duration_sec=src_duration,
        removed_sec_total=removed_sec,
    )
