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
import time
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
    # Per-word timestamps in STORY-RELATIVE OUTPUT time (seconds from
    # this story's start in the trimmed video): [{"w","s","e"}, …].
    # Source-time Deepgram words are remapped across the concatenated
    # KEEP spans (words in cut regions are dropped; straddlers are
    # clamped to the span edge). This is what lets the image↔speech
    # timing AI place each image on the exact spoken words.
    words: list[dict] = field(default_factory=list)

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
    """Pull mono 16kHz ~48kbps mp3 for Deepgram. nova-3 transcribes 16kHz
    speech perfectly, and the small payload is what matters here: the
    Deepgram UPLOAD (not its processing) is the slow step on a busy uplink
    (jobs 596/599/607), so a ~9MB file (vs ~25MB at 22kHz/q5) uploads
    ~3x faster and times out far less."""
    cmd = [
        ffmpeg_bin, "-y", "-v", "error",
        "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "libmp3lame", "-b:a", "48k",
        out_mp3,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                          errors="replace")   # non-UTF-8 ffmpeg output must not crash the reader
    if proc.returncode != 0:
        raise RuntimeError(f"audio extract failed: {proc.stderr[-400:]}")
    return out_mp3


# ─── Step 1.B — Deepgram words ──────────────────────────────────────

def _deepgram_words(audio_path: str, language: str = "multi",
                    *, diarize: bool | None = None) -> tuple[list[dict], float]:
    """Run Deepgram nova-3. Returns (word_array, duration_sec).

    Each word: {"i": int, "w": str, "s": float, "e": float}.
    Same shape V3 uses — reused so the prompt format is consistent.

    ``diarize``: None (default) keeps the existing ``KAIZER_V4_DIARIZE``
    env-gated behavior — every V4 call site passes nothing and is byte-for-
    byte unchanged. True/False overrides the env explicitly (the podcast
    shim passes True: its camera plan is dead without speaker labels).
    """
    from deepgram import DeepgramClient

    api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY not set in env")
    dg = DeepgramClient(api_key=api_key)
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    # Diarization (speaker labels per word) — needed by the podcast /
    # interview profiles for speaker-aware grids and chaptering.
    # Env-gated (when diarize is None) so news jobs keep the exact request
    # they always sent; an explicit diarize=True/False wins over the env.
    if diarize is None:
        _diarize = (os.environ.get("KAIZER_V4_DIARIZE", "0") or "0").strip().lower() \
            in ("1", "true", "yes", "on")
    else:
        _diarize = bool(diarize)
    kwargs = {
        "request":     audio_bytes,
        "model":       "nova-3",
        "punctuate":   True,
        "diarize":     _diarize,
        "smart_format": True,
    }
    if language:
        kwargs["language"] = language
    # The Deepgram UPLOAD (not its processing) times out on a busy uplink
    # (jobs 596/599/607). A 600s timeout meant each failed attempt hung TEN
    # MINUTES before retrying — two timeouts turned Stage 1 into 30min.
    # Fail-FAST with escalating timeouts instead: a hung upload is detected
    # in ~2-3min and retried, so a transient stall costs minutes not
    # tens-of-minutes, while a genuinely slow link still gets a long final
    # attempt. Base tunable via KAIZER_V4_DEEPGRAM_TIMEOUT (default 150s).
    try:
        _base_to = max(60, int(os.environ.get("KAIZER_V4_DEEPGRAM_TIMEOUT", "150") or "150"))
    except ValueError:
        _base_to = 150
    _timeouts = [_base_to, _base_to * 2, _base_to * 3, _base_to * 4]  # 150,300,450,600
    response = None
    _last = None
    for _attempt, _to in enumerate(_timeouts, start=1):
        kwargs["request_options"] = {"timeout_in_seconds": _to}
        try:
            response = dg.listen.v1.media.transcribe_file(**kwargs)
            break
        except Exception as exc:
            _last = exc
            _msg = str(exc).lower()
            if _attempt < len(_timeouts) and ("timed out" in _msg or "timeout" in _msg):
                print(f"[v4/step1] Deepgram upload timeout after {_to}s "
                      f"(attempt {_attempt}/{len(_timeouts)}) — retrying...",
                      flush=True)
                time.sleep(5)
                continue
            raise
    if response is None:
        raise RuntimeError(f"Deepgram transcription failed: {_last}")

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
        entry = {
            "i": len(words),
            "w": text,
            "s": float(getattr(w, "start", 0.0) or 0.0),
            "e": float(getattr(w, "end", 0.0) or 0.0),
        }
        # Speaker id — present only when diarization ran, so the word
        # shape (and every downstream sidecar) is unchanged otherwise.
        spk = getattr(w, "speaker", None)
        if spk is not None:
            try:
                entry["spk"] = int(spk)
            except (TypeError, ValueError):
                pass
        # Per-word confidence — same conditional-key pattern as "spk"
        # (sidecar consumers access keys by name and ignore extras; the
        # podcast shim maps it to the planners' `confidence`).
        conf = getattr(w, "confidence", None)
        if conf is not None:
            try:
                entry["conf"] = float(conf)
            except (TypeError, ValueError):
                pass
        words.append(entry)
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
    repair → missing-comma-at-newline repair. Repairs are CONSERVATIVE (only
    ever-invalid constructs, so valid content is never corrupted). Raises
    ``json.JSONDecodeError`` if none parse — the caller's model-retry is the
    last resort.
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
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    # Missing comma at a STRUCTURAL newline (job#599: Gemini dropped a comma
    # between two members). Safe because a literal newline can never occur
    # INSIDE a valid JSON string — so token-end + newline + token-start is
    # always a member boundary that must carry a comma:
    #   "value"\n"key":   →   "value",\n"key":
    #   }\n{  /  ]\n"     →   },\n{  /  ],\n"
    #   123\n"  /  true\n" (number/bool/null then next key)
    repaired2 = re.sub(r'"(\s*\n\s*)"', r'",\1"', repaired)
    repaired2 = re.sub(r"([}\]])(\s*\n\s*)([{\[\"])", r"\1,\2\3", repaired2)
    repaired2 = re.sub(r'(\d|true|false|null)(\s*\n\s*)"', r'\1,\2"', repaired2)
    try:
        return json.loads(repaired2)
    except json.JSONDecodeError:
        pass
    # GENERAL position-guided comma repair — the robust catch-all. Python's
    # decoder reports the EXACT offset where it expected a ',' ("Expecting
    # ',' delimiter"); insert one there and retry, bounded. This fixes every
    # missing-comma shape the regex tiers above don't (arbitrary indentation,
    # ]-on-its-own-line before a key, same-line members) — the Director's
    # 16-story arc plans routinely hit these. Only ever acts on the comma
    # delimiter error and each insert strictly advances the parse point, so
    # it converges; any OTHER error (truncation, bad colon) re-raises for the
    # caller's model-repair retry.
    cand3 = repaired2
    for _ in range(500):
        try:
            return json.loads(cand3)
        except json.JSONDecodeError as exc:
            if "',' delimiter" in (exc.msg or "") and 0 < exc.pos <= len(cand3):
                cand3 = cand3[:exc.pos] + "," + cand3[exc.pos:]
                continue
            raise
    return json.loads(cand3)   # bound exhausted → raise the final error


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

    from pipeline_v4.edit_profiles import active_profile_key, keep_cut_system_for
    _model = "claude-opus-4-7"
    with _anthropic_log(model=_model, purpose="cut-plan") as _acall:
        msg = client.messages.create(
            model=_model,
            max_tokens=8192,
            # Profile-aware persona (news == legacy prompt byte-for-byte).
            system=keep_cut_system_for(active_profile_key()),
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

    from pipeline_v4.edit_profiles import active_profile_key, keep_cut_system_for
    model = os.environ.get("KAIZER_V4_TRIM_GEMINI_MODEL", "gemini-2.5-flash")
    # Vertex 429 = PER-MINUTE quota, not credits (job 599: three quick
    # planner runs tripped it) — ride it out with the same backoff the
    # Director/image paths use instead of hard-failing the job.
    resp = None
    _last_exc = None
    # Ladder long enough to ride out SUSTAINED Vertex pressure (job 599:
    # three 30/60s attempts all 429'd with LIVE idle — regional/project
    # contention can outlast a per-minute window).
    for _attempt, _pause in ((1, 30), (2, 90), (3, 240), (4, 0)):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    # Profile-aware persona (news == legacy prompt byte-for-byte).
                    system_instruction=keep_cut_system_for(active_profile_key()),
                    response_mime_type="application/json",
                    # Editorial decisions should be reproducible; low temp
                    # mirrors Claude's typical default behaviour.
                    temperature=0.2,
                    # Big enough headroom for 20+ stories × N kept_spans each
                    # without truncation; matches what we saw in real jobs.
                    max_output_tokens=16384,
                ),
            )
            break
        except Exception as exc:
            _last_exc = exc
            _m = str(exc)
            if _pause and ("RESOURCE_EXHAUSTED" in _m or " 429" in _m or "429 " in _m):
                print(f"[v4/step1] Gemini KEEP/CUT 429 (attempt {_attempt}/4) "
                      f"— waiting {_pause}s for the quota window...", flush=True)
                time.sleep(_pause)
                continue
            raise RuntimeError(f"Gemini KEEP/CUT call failed: {exc}")
    if resp is None:
        raise RuntimeError(f"Gemini KEEP/CUT call failed: {_last_exc}")

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


def tighten_spans_to_speech(
    stories_in: list[dict],
    words: list[dict],
    *,
    max_gap: float = 1.2,
    pad: float = 0.25,
    min_span: float = 0.4,
) -> list[dict]:
    """Deterministic silence guarantee (the LLM is *asked* to cut dead
    air; this pass *ensures* it). For every kept span: snap the edges to
    the actual speech inside it and SPLIT it wherever consecutive words
    are more than ``max_gap`` seconds apart, keeping ``pad`` of air on
    each side so cuts never clip a breath.

    Safety rails: a span with no words inside is kept verbatim (the
    planner may keep intentional non-speech footage); fragments shorter
    than ``min_span`` are dropped; a story that would lose every span
    keeps its originals. Pure function — returns a new list."""
    tw = []
    for w in (words or []):
        try:
            ws = float(w.get("s") or w.get("start") or 0.0)
            we = float(w.get("e") or w.get("end") or 0.0)
        except (TypeError, ValueError):
            continue
        if we > ws:
            tw.append((ws, we))
    tw.sort()

    out_stories: list[dict] = []
    for s in (stories_in or []):
        new_spans: list[dict] = []
        orig_spans = list(s.get("kept_spans") or [])
        for sp in orig_spans:
            try:
                a = float(sp.get("start_sec") or 0.0)
                b = float(sp.get("end_sec") or 0.0)
            except (TypeError, ValueError):
                new_spans.append(dict(sp))
                continue
            inside = [(ws, we) for (ws, we) in tw if we > a and ws < b]
            if not inside:
                new_spans.append(dict(sp))       # non-speech footage → verbatim
                continue
            # Split runs of speech at silences > max_gap.
            runs: list[list[tuple[float, float]]] = [[inside[0]]]
            for ws, we in inside[1:]:
                if ws - runs[-1][-1][1] > max_gap:
                    runs.append([])
                runs[-1].append((ws, we))
            for run in runs:
                ra = max(a, run[0][0] - pad)
                rb = min(b, run[-1][1] + pad)
                if rb - ra >= min_span:
                    new_spans.append({
                        "start_sec": round(ra, 3),
                        "end_sec": round(rb, 3),
                        "reason": str(sp.get("reason") or "")[:40],
                    })
        ns = dict(s)
        ns["kept_spans"] = new_spans if new_spans else orig_spans
        out_stories.append(ns)
    return out_stories


def _maybe_tighten(stories_in: list[dict], words: list[dict]) -> list[dict]:
    """Apply the silence guarantee unless KAIZER_V4_TIGHTEN_SILENCE=0.
    Parameters come from the active edit profile. Fail-soft."""
    flag = (os.environ.get("KAIZER_V4_TIGHTEN_SILENCE", "1") or "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return stories_in
    try:
        from pipeline_v4.edit_profiles import active_profile_key, get_profile
        p = get_profile(active_profile_key())
        before = sum(
            (float(sp.get("end_sec") or 0) - float(sp.get("start_sec") or 0))
            for s in stories_in for sp in (s.get("kept_spans") or []))
        out = tighten_spans_to_speech(
            stories_in, words, max_gap=p.tighten_max_gap, pad=p.tighten_pad)
        after = sum(
            (float(sp.get("end_sec") or 0) - float(sp.get("start_sec") or 0))
            for s in out for sp in (s.get("kept_spans") or []))
        if before - after > 0.05:
            print(f"[v4/step1] silence tighten: -{before - after:.1f}s of dead air "
                  f"(profile={p.key}, gap>{p.tighten_max_gap}s)", flush=True)
        return out
    except Exception as exc:
        print(f"[v4/step1] silence tighten skipped (soft): {exc}", flush=True)
        return stories_in


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

def _bad_color_trc(source_video: str) -> bool:
    """True when the source's transfer characteristic is INVALID
    ("reserved"/"unknown") — browser-encoded uploads (the in-app
    compressor pre-fix) carry trc:reserved, and ffmpeg 8's swscaler
    refuses the nv12→yuv420p conversion with it (Error -129, jobs
    601/602). Fail-soft False: probing may never block a trim."""
    try:
        proc = subprocess.run(
            [os.environ.get("KAIZER_FFPROBE_BIN", "ffprobe"), "-v", "error",
             "-select_streams", "v:0", "-show_entries",
             "stream=color_transfer", "-of", "csv=p=0", source_video],
            capture_output=True, text=True, timeout=30, errors="replace")
        trc = (proc.stdout or "").strip().lower()
        return trc in ("reserved", "unknown")
    except Exception:
        return False


def _build_filter_complex(spans: list[KeptSpan], *, audio_fade_sec: float = 0.05,
                          normalize_trc: bool = False) -> str:
    """Build the ffmpeg filter_complex string that trims each span
    from both video and audio, applies a brief audio fade at every
    boundary, and concats everything into [vout] + [aout].

    ``normalize_trc`` heads each video chain with a setparams that
    rewrites an invalid transfer tag to bt709 (metadata only, pixels
    untouched) — the swscaler otherwise dies with -129. Conditional, so
    valid sources keep the exact legacy command.
    """
    _fix = "setparams=color_trc=bt709," if normalize_trc else ""
    parts: list[str] = []
    for i, sp in enumerate(spans):
        # Video trim — setpts resets the clock so concat doesn't see
        # the source timestamps.
        parts.append(
            f"[0:v]{_fix}trim=start={sp.start_sec:.3f}:end={sp.end_sec:.3f},"
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


def _split_spans_balanced(spans: list[KeptSpan]) -> tuple[list[KeptSpan],
                                                          list[KeptSpan]]:
    """Split an ORDERED span list into two contiguous halves of roughly
    equal total duration (greedy prefix walk). Order is preserved so the
    two encoded chunks concat back into exactly the single-pass output."""
    total = sum(s.duration for s in spans)
    acc = 0.0
    cut = 1
    for i, s in enumerate(spans):
        acc += s.duration
        if acc >= total / 2.0:
            cut = min(max(1, i + 1), len(spans) - 1)
            break
    return spans[:cut], spans[cut:]


def atomic_trim_concat(
    *,
    source_video: str,
    spans: list[KeptSpan],
    output_path: str,
    ffmpeg_bin: str = "ffmpeg",
) -> None:
    """Produce the trimmed master.

    Public entry — used by both Step 1 (the bulletin trim) and the
    orchestrator's per-short trim (one short = one story's spans).
    Emits a single, drift-free mp4.

    LONG sources (kept duration >= KAIZER_V4_TRIM_SPLIT_MIN_S, default
    600s) are encoded as TWO span-halves in PARALLEL (2 NVENC sessions —
    the box's proven-safe ceiling; Stage 1 has the GPU to itself) and
    joined losslessly with the concat demuxer. Job 611 measured the
    single-pass encode at 6m48s for 28.5 min of footage while the
    preset/tune levers moved nothing — the pass is decode/filter-bound,
    so halving the footage per process is the lever that actually cuts
    wall-clock (~2x). Content is identical to the single pass: same
    per-span trims, same encoder settings, and the join lands on a span
    boundary where the per-span 50ms audio fade already takes the level
    to zero (so AAC priming at the join is inaudible by construction).
    Shorts and short sources keep the byte-identical single pass. Any
    chunk failure falls back to the single pass. Kill switch:
    KAIZER_V4_TRIM_PARALLEL=0.
    """
    if not spans:
        raise RuntimeError("no kept spans -- nothing to trim")

    _norm = _bad_color_trc(source_video)
    if _norm:
        print("[v4/step1] source carries an invalid color transfer tag "
              "(trc:reserved) — normalizing to bt709 in the trim pass",
              flush=True)

    try:
        _split_min = float(os.environ.get(
            "KAIZER_V4_TRIM_SPLIT_MIN_S", "600") or "600")
    except (TypeError, ValueError):
        _split_min = 600.0
    _total_kept = sum(s.duration for s in spans)
    # DEFAULT OFF. Job 613 measured chunking at 11m32s Stage 1 — WORSE
    # than single-pass (10m09s) — because both chunks shared one NVDEC
    # decoder (hwaccel era) and, post-decoder-fix, both share the one
    # NVENC engine anyway: on single-encoder GPUs splitting a shared
    # bottleneck adds contention instead of halving time. With software
    # decode (5658b14) the single pass is already encoder-bound (~3min
    # for a 29-min source). Opt-in for hosts with multiple NVENC engines.
    _par_on = (os.environ.get("KAIZER_V4_TRIM_PARALLEL", "0") or "0") \
        .strip().lower() in ("1", "true", "on", "yes")
    if _par_on and len(spans) >= 2 and _total_kept >= _split_min:
        try:
            _trim_parallel_two_chunks(
                source_video=source_video, spans=spans,
                output_path=output_path, ffmpeg_bin=ffmpeg_bin, norm=_norm)
            return
        except Exception as exc:
            print(f"[v4/step1] parallel trim failed ({exc}) — falling back "
                  f"to the single-pass trim", flush=True)

    _single_pass_trim(source_video=source_video, spans=spans,
                      output_path=output_path, ffmpeg_bin=ffmpeg_bin,
                      norm=_norm)


def _trim_cmd(*, source_video: str, spans: list[KeptSpan], output_path: str,
              ffmpeg_bin: str, norm: bool) -> list[str]:
    filter_complex = _build_filter_complex(spans, normalize_trc=norm)
    return [
        ffmpeg_bin, "-y", "-v", "error",
        # GPU decode when NVENC is active (input option — must precede -i).
        *_dec_args(),
        "-i", source_video,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        # The trimmed master is an INTERMEDIATE — Stage 3 recomposites and
        # RE-ENCODES every story from it (bulletin via _slice_video crf 20;
        # shorts via compose_clip*), so the trim's encode SPEED can be traded
        # up freely without touching FINAL quality. Levers, all quality-neutral
        # here because a downstream re-encode owns the deliverable:
        #   - preset veryfast → NVENC p1 (fastest; less per-frame analysis).
        #   - tune ll → skip NVENC's HQ rate-control lookahead (faster still).
        # At the fixed near-lossless cq (crf 19) both keep the same quality
        # TARGET — they cost slightly more bits, not less quality. All env-
        # tunable (KAIZER_V4_TRIM_PRESET / _TRIM_TUNE / _TRIM_CRF).
        # INVARIANT: keep any lean/maxrate setting INSIDE this atomic_trim_concat
        # cmd only — never in _slice_video or the final composite.
        *_enc_args(
            crf=int(os.environ.get("KAIZER_V4_TRIM_CRF", "19") or "19"),
            preset_hint=(os.environ.get("KAIZER_V4_TRIM_PRESET", "veryfast")
                         or "veryfast"),
            tune=(os.environ.get("KAIZER_V4_TRIM_TUNE", "ll") or "ll")),
        # Optional peak-bitrate ceiling on the throwaway intermediate. The
        # near-lossless cq otherwise inflated it to ~2.4x the source (a 27-min
        # podcast became 3.15GB), which slows the write + every Stage-3 slice
        # read. Quality-neutral (every consumer re-encodes) for ≤1080p sources.
        # DEFAULT OFF (byte-identical to before); set KAIZER_V4_TRIM_MAXRATE=8M.
        *(["-maxrate", (os.environ.get("KAIZER_V4_TRIM_MAXRATE") or ""),
           "-bufsize", (os.environ.get("KAIZER_V4_TRIM_BUFSIZE")
                        or os.environ.get("KAIZER_V4_TRIM_MAXRATE") or "")]
          if (os.environ.get("KAIZER_V4_TRIM_MAXRATE") or "").strip() else []),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-movflags", "+faststart",
        output_path,
    ]


def _single_pass_trim(*, source_video: str, spans: list[KeptSpan],
                      output_path: str, ffmpeg_bin: str, norm: bool) -> None:
    cmd = _trim_cmd(source_video=source_video, spans=spans,
                    output_path=output_path, ffmpeg_bin=ffmpeg_bin, norm=norm)
    # Centralised runner: one retry, NVENC→libx264 fallback on GPU
    # session/CUDA failures, stderr-tail logging. Raises RuntimeError
    # on final failure (same contract as the old inline check).
    _run_ffmpeg(cmd, timeout=60 * 30, log_label="atomic_trim_concat")


def _trim_parallel_two_chunks(*, source_video: str, spans: list[KeptSpan],
                              output_path: str, ffmpeg_bin: str,
                              norm: bool) -> None:
    """Encode two contiguous span-halves CONCURRENTLY (2 NVENC sessions),
    then join losslessly with the concat demuxer. Chunk temp files live
    beside the output and are always cleaned up. Raises on any failure —
    atomic_trim_concat falls back to the single pass."""
    from concurrent.futures import ThreadPoolExecutor
    a, b = _split_spans_balanced(spans)
    if not a or not b:
        raise RuntimeError("split produced an empty chunk")
    paths = [output_path + ".chunk0.mp4", output_path + ".chunk1.mp4"]
    lst = output_path + ".concat.txt"
    print(f"[v4/step1] parallel trim: 2 chunks "
          f"({sum(s.duration for s in a):.0f}s + "
          f"{sum(s.duration for s in b):.0f}s)", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=2,
                                thread_name_prefix="v4-trimchunk") as pool:
            futs = [
                pool.submit(
                    _run_ffmpeg,
                    _trim_cmd(source_video=source_video, spans=part,
                              output_path=p, ffmpeg_bin=ffmpeg_bin,
                              norm=norm),
                    timeout=60 * 30, log_label=f"trim_chunk_{i}")
                for i, (part, p) in enumerate(zip((a, b), paths))
            ]
            for f in futs:
                f.result()
        for p in paths:
            if not os.path.isfile(p) or os.path.getsize(p) < 1024:
                raise RuntimeError(f"trim chunk missing/empty: {p}")
        with open(lst, "w", encoding="utf-8") as fh:
            for p in paths:
                fh.write("file '" + p.replace("'", "'\\''") + "'\n")
        # Lossless join: same codec/params in both chunks; the join point
        # is a span boundary whose 50ms audio fade already sits at zero.
        _run_ffmpeg(
            [ffmpeg_bin, "-y", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", lst, "-c", "copy", "-movflags", "+faststart",
             output_path],
            timeout=60 * 10, log_label="trim_chunk_join")
        if not os.path.isfile(output_path) \
                or os.path.getsize(output_path) < 1024:
            raise RuntimeError("concat join produced no output")
    finally:
        for p in paths + [lst]:
            try:
                os.remove(p)
            except OSError:
                pass


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

    # 2b) Content type → edit profile. One cheap LLM call over the
    #     transcript head decides HOW to edit (news persona vs podcast
    #     chapters vs interview Q&A…); the operator's explicit wizard
    #     pick wins; unsure → news (legacy behavior) + flagged.
    from pipeline_v4 import content_type as _ct
    _ct.resolve_and_record(words=words, duration=src_duration,
                           out_dir=output_dir_p, language=language)

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
    claude_stories = _maybe_tighten(claude_stories, words)
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
            span_out_starts: list[float] = []   # OUTPUT-timeline start of each span
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
                span_out_starts.append(cur)
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
            #
            # While we're pairing (span, word) anyway, also remap each
            # word's SOURCE time into STORY-RELATIVE OUTPUT time: the
            # trimmed video is the KEEP spans concatenated, so a word at
            # source ws inside a span whose output copy starts at
            # span_out_start lands at  span_out_start + (ws - span.start)
            # in the output — minus the story's own output start to make
            # it story-relative. Straddlers clamp to the span edge.
            story_words: list[str] = []
            story_word_objs: list[dict] = []
            for sp_obj, sp_out_start in zip(story_spans, span_out_starts):
                for w in words:
                    ws = float(w.get("s") or w.get("start") or 0.0)
                    we = float(w.get("e") or w.get("end") or 0.0)
                    if we < sp_obj.start_sec or ws > sp_obj.end_sec:
                        continue
                    tok = (w.get("w") or w.get("word") or "").strip()
                    if not tok:
                        continue
                    story_words.append(tok)
                    rel_s = sp_out_start + (max(ws, sp_obj.start_sec) - sp_obj.start_sec) - s_start_in_output
                    rel_e = sp_out_start + (min(we, sp_obj.end_sec) - sp_obj.start_sec) - s_start_in_output
                    obj = {
                        "w": tok,
                        "s": round(rel_s, 2),
                        "e": round(max(rel_e, rel_s), 2),
                    }
                    if w.get("spk") is not None:
                        obj["spk"] = w["spk"]
                    story_word_objs.append(obj)
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
                words=story_word_objs,
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
            claude_stories = _maybe_tighten(claude_stories, words)
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
            # No cuts → output timeline == source timeline (identity remap).
            words=[
                {"w": tok, "s": round(float(w.get("s") or 0.0), 2),
                 "e": round(float(w.get("e") or 0.0), 2)}
                for w in words
                if (tok := (w.get("w") or w.get("word") or "").strip())
            ],
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
