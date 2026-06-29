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
from pipeline_v4.encoder import video_encoder_args as _enc_args
from pipeline_v4.encoder import video_decoder_args as _dec_args

# Factory live-belt telemetry — best-effort, never breaks a render. Uses the
# thread-bound envelope set by the orchestrator (services.stage_events.bind),
# so we emit station transitions without threading job_id through signatures.
try:
    from services import stage_events as _se
except Exception:  # CLI-only runs without the app loaded
    _se = None

try:
    from services import stage_gate as _gate
except Exception:
    _gate = None

try:
    from learning.claude_log import log_anthropic_call as _log_anthropic
except Exception:
    _log_anthropic = None


from contextlib import contextmanager as _contextmanager


class _NoopCall:
    def record(self, *a, **k): pass
    def record_tokens(self, *a, **k): pass


@_contextmanager
def _anthropic_log(model: str, purpose: str):
    """Claude usage-logging context manager (no-op if the logger is
    unavailable). Yields an object with .record(resp). A failure of the
    Claude call itself propagates normally — the logger records the error and
    re-raises; we never swallow the caller's exception."""
    if _log_anthropic is None:
        yield _NoopCall()
        return
    with _log_anthropic(db=None, model=model, purpose=purpose) as c:
        yield c


def _belt(stage: str, status: str) -> None:
    if _se is not None:
        try:
            _se.emit_here(stage, status)
        except Exception:
            pass


def _encode_gate():
    """Cross-process/machine NVENC gate context manager (no-op if disabled)."""
    if _gate is not None:
        try:
            return _gate.gate("encode")
        except Exception:
            pass
    from contextlib import nullcontext
    return nullcontext()


from pipeline_v4.ffmpeg_exec import run_ffmpeg as _run_ffmpeg


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
    # Verbatim spoken text in the story's KEEP spans. Built by
    # walking the Deepgram word array and joining words that fall
    # inside the story's source_spans. Carried forward so image
    # generation prompts can reference the actual incident words
    # ("today PM Modi launched ... in Hyderabad") instead of just
    # Claude's headline / summary — much higher chance of getting
    # an event-specific photo or render.
    transcript_text: str = ""

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


def _loads_lenient(raw: str):
    """Parse model JSON tolerantly: strict → outer ``{ ... }`` → trailing-comma
    repair. Repairs are CONSERVATIVE (only ever-invalid constructs, so valid
    content is never corrupted). Raises ``json.JSONDecodeError`` if none parse —
    genuine syntax errors (e.g. a missing comma) are left for the caller's
    model-retry, which is the only safe fix for those.
    """
    s = _strip_code_fences(raw or "")
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    cand = m.group(0) if m else s
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        pass
    # Trailing commas before } or ] are ALWAYS invalid JSON → safe to strip.
    repaired = re.sub(r",(\s*[}\]])", r"\1", cand)
    return json.loads(repaired)   # raises if still malformed


def _gemini_repair_json(client, genai_types, model: str, broken: str) -> str:
    """Ask Gemini to return a strictly-valid version of a JSON blob it produced
    with a syntax error. Best-effort; the caller still parses the result
    tolerantly. This rescues the job#500-class failure where the cut-plan JSON
    had a stray ``Expecting ',' delimiter`` that no regex can safely repair."""
    fix_prompt = (
        "The text below was meant to be ONE valid JSON object but has a syntax "
        "error (often a missing comma). Return ONLY the corrected JSON — the "
        "SAME data and keys, no commentary, no code fences, strictly parseable:\n\n"
        + (broken or "")[:120000]
    )
    resp = client.models.generate_content(
        model=model,
        contents=fix_prompt,
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=16384,
        ),
    )
    return (resp.text or "").strip()


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

    _model = "claude-opus-4-7"
    with _anthropic_log(model=_model, purpose="cut-plan") as _acall:
        msg = client.messages.create(
            model=_model,
            max_tokens=8192,
            system=v4_prompts.KEEP_CUT_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )
        _acall.record(msg)
    raw = msg.content[0].text if msg.content else ""
    try:
        data = _loads_lenient(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Claude returned non-JSON: {raw[:400]}") from exc

    stories = data.get("stories") or []
    removed = float(data.get("removed_sec_total") or 0.0)
    return stories, removed


def _gemini_keep_cut_plan(
    *,
    words: list[dict],
    language: str,
    duration_sec: float,
) -> tuple[list[dict], float]:
    """Call Gemini 2.5 Flash on Vertex AI with the SAME KEEP/CUT prompt
    Claude uses, so an A/B comparison is apples-to-apples.

    Why Vertex Gemini for an alternate planner:
      - The Vertex client is already wired (seo_provider uses it for
        SEO generation), so no extra auth or new env vars.
      - gemini-2.5-flash is ~10× cheaper than Claude Opus 4.7 for the
        same prompt — meaningful at SaaS scale if quality holds.
      - 1M context window handles every realistic news source (the
        current 2530-word transcripts land around 30-40K tokens).
      - response_mime_type=application/json enforces the schema so we
        don't need a recovery regex like the Claude path does.

    Returns the SAME (stories, removed_sec) tuple shape Claude returns,
    so downstream code in ``run_step1`` doesn't care which planner ran.
    """
    try:
        from seo.generator import _gemini_client
        from google.genai import types as genai_types
    except Exception as exc:
        raise RuntimeError(
            f"Vertex Gemini SDK unavailable: {exc}. "
            f"Switch planner to claude or fix the google-genai install."
        )

    try:
        client = _gemini_client()
    except Exception as exc:
        raise RuntimeError(
            f"Gemini client init failed: {exc}. "
            f"Check KAIZER_GCP_PROJECT / KAIZER_VERTEX_CREDENTIALS or "
            f"switch planner to claude."
        )

    target_min, target_max = _target_duration_window(duration_sec)
    user_prompt = v4_prompts.build_keep_cut_user_prompt(
        words=words,
        target_min_sec=target_min,
        target_max_sec=target_max,
        language=language,
        duration_sec=duration_sec,
    )

    model = os.environ.get("KAIZER_V4_TRIM_GEMINI_MODEL", "gemini-2.5-flash")
    try:
        resp = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=v4_prompts.KEEP_CUT_SYSTEM,
                response_mime_type="application/json",
                # Editorial decisions should be reproducible; low temp
                # mirrors Claude's typical default behaviour.
                temperature=0.2,
                # Big enough headroom for 20+ stories × N kept_spans each
                # without truncation; matches what we saw in real jobs.
                max_output_tokens=16384,
            ),
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini KEEP/CUT call failed: {exc}")

    raw = ""
    try:
        raw = (resp.text or "").strip()
    except Exception:
        raw = ""
    if not raw:
        raise RuntimeError("Gemini KEEP/CUT returned empty body")

    try:
        data = _loads_lenient(raw)
    except json.JSONDecodeError as exc:
        # Malformed JSON (e.g. a missing comma — the job#500 failure). No regex
        # can safely fix that, so ask Gemini ONCE to repair its own output,
        # then parse tolerantly again. Loud-fail only if the repair also fails.
        pos = str(exc)
        try:
            fixed = _gemini_repair_json(client, genai_types, model, raw)
            data = _loads_lenient(fixed)
            print(f"[v4/step1] gemini cut-plan JSON repaired on retry ({pos})", flush=True)
        except Exception as exc2:
            raise RuntimeError(
                f"Gemini KEEP/CUT JSON unparseable even after repair ({pos}): {raw[:400]}"
            ) from exc2

    stories = data.get("stories") or []
    removed = float(data.get("removed_sec_total") or 0.0)
    return stories, removed


def _select_planner() -> tuple[str, callable]:
    """Resolve which KEEP/CUT planner to use this run.

    Reads ``KAIZER_V4_TRIM_PLANNER`` (set per-job by the runner from
    the Job.meta.trim_planner choice). Unknown values fall back to
    ``claude`` so an env typo doesn't crash a paid job.

    Returns ``(planner_label, planner_function)`` so the orchestrator
    can log which planner ran without re-reading the env var.
    """
    choice = (os.environ.get("KAIZER_V4_TRIM_PLANNER") or "claude").strip().lower()
    if choice == "gemini":
        return ("gemini", _gemini_keep_cut_plan)
    if choice != "claude":
        print(f"[v4/step1] unknown KAIZER_V4_TRIM_PLANNER={choice!r}, "
              f"falling back to claude", flush=True)
    return ("claude", _claude_keep_cut_plan)


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
        # GPU decode when NVENC is active (input option — must precede -i).
        *_dec_args(),
        "-i", source_video,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        *_enc_args(crf=21, preset_hint="medium"),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-movflags", "+faststart",
        output_path,
    ]
    # Centralised runner: one retry, NVENC→libx264 fallback on GPU
    # session/CUDA failures, stderr-tail logging. Raises RuntimeError
    # on final failure (same contract as the old inline check).
    _run_ffmpeg(cmd, timeout=60 * 30, log_label="atomic_trim_concat")


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
    _belt("transcribe", "entered")
    print(f"[v4/step1] Deepgram nova-3 transcribing ({language}) ...", flush=True)
    words, src_duration = _deepgram_words(audio_mp3, language=language)
    _belt("transcribe", "exited")
    print(f"[v4/step1]   got {len(words)} words across {src_duration:.1f}s source", flush=True)
    try:
        os.unlink(audio_mp3)
    except OSError:
        pass

    # 3) KEEP/CUT plan — Claude OR Gemini, picked via env var so each
    #    job records which engine it used (operator A/B comparison).
    planner_label, planner_fn = _select_planner()
    pretty_model = (
        "Claude opus-4-7" if planner_label == "claude"
        else f"Gemini {os.environ.get('KAIZER_V4_TRIM_GEMINI_MODEL', 'gemini-2.5-flash')}"
    )
    _belt("cut_plan", "entered")
    print(f"[v4/step1] {pretty_model} planning KEEP/CUT "
          f"(KAIZER_V4_TRIM_PLANNER={planner_label}) ...", flush=True)
    claude_stories, removed_sec = planner_fn(
        words=words,
        language=language,
        duration_sec=src_duration,
    )
    _belt("cut_plan", "exited")
    print(f"[v4/step1]   {len(claude_stories)} stories, ~{removed_sec:.1f}s removed "
          f"(planner={planner_label})", flush=True)

    # Flatten kept_spans into one ordered list (preserving story grouping for
    # the output timeline). Wrapped in a helper so we can re-run it verbatim
    # after a planner retry without duplicating the logic.
    def _flatten(stories_in):
        spans: list[KeptSpan] = []
        out: list[TrimmedStory] = []
        cur = 0.0          # current position in the OUTPUT timeline
        for s_idx, s in enumerate(stories_in):
            s_start_in_output = cur
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
                spans.append(sp_obj)
                cur += sp_obj.duration
            if not story_spans:
                continue
            # Collect the spoken transcript for this story by walking the
            # Deepgram word array and joining every word that falls inside
            # any of its KEEP spans. This is the actual speech content the
            # anchor delivered, NOT Claude's headline / summary — the
            # extra grounding lets image generation match the specific
            # incident (place names, people, dates).
            story_words: list[str] = []
            for sp_obj in story_spans:
                for w in words:
                    ws = float(w.get("s") or w.get("start") or 0.0)
                    we = float(w.get("e") or w.get("end") or 0.0)
                    if we < sp_obj.start_sec or ws > sp_obj.end_sec:
                        continue
                    tok = (w.get("w") or w.get("word") or "").strip()
                    if tok:
                        story_words.append(tok)
            # Trim to a reasonable size — image prompts don't need 30s of
            # filler. 1200 chars ≈ 200 words ≈ the most salient sentences.
            transcript_text = " ".join(story_words)[:1200]
            out.append(TrimmedStory(
                story_index=s_idx,
                title_native=str(s.get("title_native") or "")[:200],
                title_english=str(s.get("title_english") or "")[:200],
                summary=str(s.get("summary") or "")[:500],
                video_t_start=s_start_in_output,
                video_t_end=cur,
                source_spans=story_spans,
                transcript_text=transcript_text,
            ))
        return spans, out, cur

    all_spans, stories_out, cursor = _flatten(claude_stories)

    # The planner occasionally returns ZERO usable spans (a transient model
    # blip, or a source with little clean speech). Don't hard-fail the whole
    # job: retry the planner once, then fall back to keeping the FULL source
    # so SOMETHING renders and the operator can refine the cut in the canvas.
    if not all_spans:
        print(f"[v4/step1] planner={planner_label} returned no kept spans — retrying once", flush=True)
        try:
            claude_stories, removed_sec = planner_fn(
                words=words, language=language, duration_sec=src_duration,
            )
            all_spans, stories_out, cursor = _flatten(claude_stories)
        except Exception as _retry_exc:
            print(f"[v4/step1] planner retry failed: {_retry_exc}", flush=True)

    if not all_spans:
        if (src_duration or 0.0) <= 0.1:
            raise RuntimeError(
                "trim planner kept nothing and the source has no usable "
                "audio/duration — cannot render"
            )
        print("[v4/step1] still no kept spans — keeping FULL source as fallback "
              "(refine the cut in the canvas editor)", flush=True)
        _full = KeptSpan(start_sec=0.0, end_sec=float(src_duration), reason="full-source fallback")
        _full_words = " ".join(
            (w.get("w") or w.get("word") or "").strip() for w in words
        ).strip()[:1200]
        all_spans = [_full]
        cursor = float(src_duration)
        stories_out = [TrimmedStory(
            story_index=0,
            title_native="",
            title_english="",
            summary="",
            video_t_start=0.0,
            video_t_end=cursor,
            source_spans=[_full],
            transcript_text=_full_words,
        )]
        removed_sec = 0.0

    # 4) Atomic ffmpeg pass — gated: bounds simultaneous NVENC encodes across
    #    ALL render jobs (this box + any other) so admission can be raised
    #    without thrashing the single encoder.
    _belt("trim", "entered")
    output_path = str(output_dir_p / output_filename)
    print(f"[v4/step1] atomic ffmpeg trim+concat ({len(all_spans)} spans) -> {output_filename}", flush=True)
    with _encode_gate():
        _atomic_trim_concat(
            source_video=source_video,
            spans=all_spans,
            output_path=output_path,
        )

    trimmed_dur = cursor
    _belt("trim", "exited")
    print(f"[v4/step1]   done -- {trimmed_dur:.1f}s trimmed output", flush=True)

    return TrimResult(
        trimmed_path=output_path,
        trimmed_duration_sec=trimmed_dur,
        stories=stories_out,
        source_duration_sec=src_duration,
        removed_sec_total=removed_sec,
    )
