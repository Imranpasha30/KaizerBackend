"""
kaizer.pipeline.narrative
==========================
Narrative Engine — top-level orchestrator for the Kaizer News video pipeline.

Picks PAYOFFS not setups, ends clips at story beats not timestamps, and
supports Telugu/Hindi out of the box.  Runs ASR → shot detection → audio RMS
valley detection → Gemini narrative labeling → snap-to-beat boundary snapping
→ composite scoring.

Usage
-----
    from pipeline_core.narrative import extract_narrative_clips, NarrativeResult

    result = extract_narrative_clips(
        "/path/to/video.mp4",
        mode="standalone",
        target_clips=5,
        language="te",
        gemini_api_key="...",   # optional; falls back to GEMINI_API_KEY env var
    )
    for clip in result.candidates:
        print(
            f"[{clip.start:.1f}s – {clip.end:.1f}s]  "
            f"role={clip.narrative_role}  score={clip.composite_score:.3f}"
        )

NarrativeResult fields
----------------------
  candidates      : list[ClipCandidate]  — Sorted by composite_score descending.
  source_duration : float                — Total video duration in seconds.
  language        : str                  — Detected or provided ISO 639-1 code.
  warnings        : list[str]            — Non-fatal issues.

ClipCandidate fields
--------------------
  start, end, duration  : float
  narrative_role        : str   — setup/opportunity/turn/setback/climax/coda/unlabeled
  hook_score            : float — [0,1]; strength of the first 3 s of transcript
  completion_score      : float — [0,1]; how well the clip ends on a beat
  importance_score      : float — [0,1]; Gemini's overall importance
  composite_score       : float — weighted blend used for sorting
  transcript_slice      : str
  start_sources         : list[str]
  end_sources           : list[str]
  meta                  : dict
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("kaizer.pipeline.narrative")

# ── Constants ──────────────────────────────────────────────────────────────────

# Base dir for model path resolution (same convention as other modules)
_BASE_DIR: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

# Narrative roles recognised by the engine
_VALID_ROLES: frozenset[str] = frozenset({
    "setup", "opportunity", "turn", "setback", "climax", "coda", "unlabeled",
})

# Number of RMS valleys to extract for boundary snapping
_TOP_VALLEYS: int = 10

# Gemini model preference list (newest first; fall back on quota/404 errors)
_GEMINI_MODELS: list[str] = ["gemini-2.5-flash", "gemini-1.5-flash"]

# Mode → (min_s, max_s) clip window
_MODE_DURATION: dict[str, tuple[float, float]] = {
    "standalone":     (45.0, 60.0),
    "trailer":        (30.0, 50.0),
    "series":         (45.0, 90.0),
    "promo":          (15.0, 25.0),
    "highlight":      (20.0, 60.0),
    "full_narrative": (60.0, 180.0),
}

_DEFAULT_MODE_DURATION: tuple[float, float] = (30.0, 90.0)

# Hooks are scored on the first N seconds of the clip's transcript
_HOOK_WINDOW_S: float = 3.0

# RMS valley window length
_VALLEY_WINDOW_S: float = 0.5

# Gemini prompt template
_GEMINI_PROMPT_TEMPLATE = """\
You are a professional video editor for a news channel. Analyse the following \
video transcript (with sentence timestamps) and identify the {n} most \
narratively important turning points that would make compelling clip segments.

Video details:
  - Total duration: {duration:.1f} seconds
  - Language: {language}
  - Shot count: {shot_count}

Transcript (format: [start_s – end_s] index: text):
{transcript_block}

For each turning point, return a JSON object with:
  - "anchor_index": integer index of the pivotal sentence (0-based)
  - "narrative_role": one of setup/opportunity/turn/setback/climax/coda
  - "importance_score": float 0.0 to 1.0 (1.0 = most important)
  - "reason": brief one-sentence explanation

Return ONLY a JSON array of {n} objects, no other text. Example:
[
  {{"anchor_index": 4, "narrative_role": "climax", "importance_score": 0.95,
    "reason": "The revelation that changes everything."}},
  ...
]
"""


# ── Public dataclasses ─────────────────────────────────────────────────────────

@dataclass
class ClipCandidate:
    """A single clip candidate produced by the narrative engine.

    Attributes
    ----------
    start : float
        Clip start in seconds.
    end : float
        Clip end in seconds.
    duration : float
        ``end - start`` in seconds.
    narrative_role : str
        Semantic role in the story arc.
    hook_score : float
        Strength of the clip's opening 3 seconds [0, 1].
    completion_score : float
        How cleanly the clip ends on a sentence beat [0, 1].
    importance_score : float
        Gemini's assessed importance [0, 1]; defaults to 0.5 when unavailable.
    composite_score : float
        Weighted blend of the above three scores used for sorting.
    transcript_slice : str
        Transcript text that falls within this clip window.
    start_sources : list[str]
        Signals that drove the snapped start edge.
    end_sources : list[str]
        Signals that drove the snapped end edge.
    meta : dict
        Arbitrary extra metadata (Gemini reason, anchor sentence, etc.).
    """

    start: float
    end: float
    duration: float
    narrative_role: str
    hook_score: float
    completion_score: float
    importance_score: float
    composite_score: float
    transcript_slice: str
    start_sources: list[str] = field(default_factory=list)
    end_sources: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class NarrativeResult:
    """Result of :func:`extract_narrative_clips`.

    Attributes
    ----------
    candidates : list[ClipCandidate]
        Clip candidates sorted by ``composite_score`` descending.
    source_duration : float
        Total duration of the source video in seconds.
    language : str
        Detected or provided ISO 639-1 language code.
    warnings : list[str]
        Non-fatal issues collected during the pipeline run.
    """

    candidates: list[ClipCandidate]
    source_duration: float
    language: str
    warnings: list[str] = field(default_factory=list)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _probe_duration(video_path: str) -> float:
    """Return video duration in seconds via ffprobe. Returns 0.0 on failure."""
    try:
        from pipeline_core.qa import FFPROBE_BIN  # type: ignore
        ffprobe = FFPROBE_BIN
    except Exception:
        import shutil as _sh
        ffprobe = _sh.which("ffprobe") or "ffprobe"

    cmd = [
        ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            return float(data.get("format", {}).get("duration", 0.0))
    except Exception as exc:
        logger.warning("narrative: ffprobe duration probe failed: %s", exc)
    return 0.0


def _compute_rms_valleys(video_path: str, top_k: int = _TOP_VALLEYS) -> list[float]:
    """Extract the timestamps of the *top_k* quietest 0.5-second windows.

    Uses librosa on the extracted 16 kHz mono audio.  Returns an empty list
    if the audio cannot be read.
    """
    try:
        import librosa  # type: ignore
    except ImportError:
        logger.warning("narrative: librosa not available; RMS valleys skipped")
        return []

    try:
        from pipeline_core.pipeline import FFMPEG_BIN  # type: ignore
        ffmpeg = FFMPEG_BIN
    except Exception:
        import shutil as _sh
        ffmpeg = _sh.which("ffmpeg") or "ffmpeg"

    tmp_wav: Optional[str] = None
    try:
        fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="kaizer_rms_")
        os.close(fd)
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-y", tmp_wav,
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
        if proc.returncode != 0:
            logger.warning("narrative: audio extraction for RMS failed")
            return []

        y, sr = librosa.load(tmp_wav, sr=16000, mono=True)
    except Exception as exc:
        logger.warning("narrative: RMS valley extraction failed: %s", exc)
        return []
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except Exception:
                pass

    if len(y) == 0:
        return []

    hop = int(sr * _VALLEY_WINDOW_S)
    frame_length = hop * 2

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
    n_frames = len(rms)

    if n_frames == 0:
        return []

    # Pick the top_k frames with lowest RMS
    n_pick = min(top_k, n_frames)
    valley_indices = np.argsort(rms)[:n_pick]

    # Convert frame index → timestamp (centre of the window)
    valley_times = [(int(i) * _VALLEY_WINDOW_S + _VALLEY_WINDOW_S / 2) for i in valley_indices]
    valley_times.sort()
    logger.debug("narrative: RMS valleys at %s", [f"{t:.2f}" for t in valley_times])
    return valley_times


def _build_transcript_block(sentences: list) -> str:
    """Build a compact numbered block from Sentence objects for the Gemini prompt."""
    lines: list[str] = []
    for i, sent in enumerate(sentences):
        start = getattr(sent, "start", 0.0)
        end = getattr(sent, "end", 0.0)
        text = getattr(sent, "text", "")
        lines.append(f"[{start:.1f}s – {end:.1f}s] {i}: {text}")
    return "\n".join(lines)


def _call_gemini(
    prompt: str,
    api_key: str,
) -> Optional[list[dict]]:
    """Call Gemini API with *prompt* and return the parsed list of turning points.

    Tries models in _GEMINI_MODELS order.  Returns None on any failure
    (network, quota, bad key, JSON parse error).
    """
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        logger.warning("narrative: google-generativeai not installed; Gemini skipped")
        return None

    genai.configure(api_key=api_key)

    for model_name in _GEMINI_MODELS:
        try:
            logger.info("narrative: calling Gemini model %s", model_name)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            raw_text = response.text or ""

            # Extract JSON array from the response text
            json_match = re.search(r"\[.*\]", raw_text, re.DOTALL)
            if not json_match:
                logger.warning(
                    "narrative: Gemini (%s) returned no JSON array; raw=%r",
                    model_name, raw_text[:200],
                )
                continue

            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                logger.info(
                    "narrative: Gemini (%s) returned %d turning points",
                    model_name, len(parsed),
                )
                return parsed

        except Exception as exc:
            logger.warning(
                "narrative: Gemini model %s failed: %s; trying next", model_name, exc
            )
            continue

    logger.warning("narrative: all Gemini models failed; falling back to heuristic-only mode")
    return None


def _sentences_in_window(sentences: list, start: float, end: float) -> list:
    """Return sentences whose midpoint falls within [start, end]."""
    result = []
    for s in sentences:
        s_start = getattr(s, "start", 0.0)
        s_end = getattr(s, "end", 0.0)
        mid = (s_start + s_end) / 2.0
        if start <= mid <= end:
            result.append(s)
    return result


def _transcript_slice(sentences: list, start: float, end: float) -> str:
    """Return the concatenated text of sentences within [start, end]."""
    return " ".join(
        getattr(s, "text", "") for s in _sentences_in_window(sentences, start, end)
    ).strip()


def _hook_score(sentences: list, clip_start: float, clip_end: float) -> float:
    """Score the hook strength of the first _HOOK_WINDOW_S of the clip.

    Heuristics:
      - Has content in the first window (not silent).
      - First sentence starts with a question or exclamation (+0.2).
      - First sentence is short and punchy (≤ 10 words, +0.2).
      - First word is a strong trigger word (verb/imperative/number, +0.2).
      - First sentence has no unresolved pronoun reference (+0.1).

    Score range: [0, 1].
    """
    hook_end = clip_start + _HOOK_WINDOW_S
    hook_sentences = _sentences_in_window(sentences, clip_start, min(hook_end, clip_end))

    if not hook_sentences:
        return 0.1  # clip has content but no mapped transcript in hook window

    score = 0.3  # base score for having any hook content
    first_sent_text = (getattr(hook_sentences[0], "text", "") or "").strip()

    # Question or exclamation
    if re.search(r"[?!]", first_sent_text):
        score += 0.2

    # Short punchy sentence
    word_count = len(first_sent_text.split())
    if 0 < word_count <= 10:
        score += 0.2

    # Starts with a number, verb-like word, or common news opener
    first_word = re.split(r"\s+", first_sent_text)[0].lower() if first_sent_text else ""
    _strong_openers = {
        "breaking", "watch", "listen", "revealed", "exclusive", "alert",
        "warning", "new", "just", "now", "today", "urgent",
    }
    if first_word in _strong_openers or re.match(r"^\d", first_word):
        score += 0.2

    # No unresolved pronoun
    from pipeline_core.clip_boundaries import _DANGLING_PRONOUN_RE  # type: ignore
    if not _DANGLING_PRONOUN_RE.match(first_sent_text):
        score += 0.1

    return min(1.0, score)


def _completion_score(sentences: list, clip_start: float, clip_end: float) -> float:
    """Score how cleanly the clip ends on a narrative beat.

    Uses :func:`pipeline_core.clip_boundaries.detect_completion` on the last
    sentence in the clip window.

    Returns a float in [0, 1]:
      - 1.0 → last sentence is complete (≥2 heuristics pass)
      - 0.5 → only 1 heuristic passes
      - 0.2 → no sentence found in window
    """
    from pipeline_core.clip_boundaries import detect_completion  # type: ignore

    in_window = _sentences_in_window(sentences, clip_start, clip_end)
    if not in_window:
        return 0.2

    last_sent = in_window[-1]
    is_complete, reasons = detect_completion(last_sent)

    n_reasons = len(reasons)
    if is_complete:
        return 1.0
    elif n_reasons == 1:
        return 0.5
    else:
        return 0.2


def _composite_score(
    importance: float,
    hook: float,
    completion: float,
    mode: str,
) -> float:
    """Compute the composite score using mode-specific weights."""
    if mode == "trailer":
        # Trailers live or die on the hook
        return 0.5 * hook + 0.25 * importance + 0.25 * completion
    else:
        # Default blend for standalone, series, promo, highlight, full_narrative
        return 0.4 * importance + 0.3 * hook + 0.3 * completion


def _heuristic_turning_points(
    sentences: list,
    target_clips: int,
    mode_min_s: float,
    mode_max_s: float,
    source_duration: float,
) -> list[dict]:
    """Generate turning points heuristically when Gemini is unavailable.

    Distributes *target_clips* evenly across the video, anchored to sentence
    midpoints.  Returns a list of dicts compatible with the Gemini output format
    (minus 'reason').
    """
    if not sentences:
        return []

    n = len(sentences)
    result: list[dict] = []

    if target_clips >= n:
        # Fewer sentences than requested clips; anchor on each sentence
        for i, sent in enumerate(sentences):
            result.append({
                "anchor_index": i,
                "narrative_role": "unlabeled",
                "importance_score": 0.5,
                "reason": "heuristic fallback",
            })
        return result

    # Evenly space anchors across the sentence list
    indices = [int(round(i * (n - 1) / (target_clips - 1))) for i in range(target_clips)] \
        if target_clips > 1 else [n // 2]

    # Deduplicate
    seen: set[int] = set()
    for idx in indices:
        idx = max(0, min(idx, n - 1))
        if idx not in seen:
            seen.add(idx)
            result.append({
                "anchor_index": idx,
                "narrative_role": "unlabeled",
                "importance_score": 0.5,
                "reason": "heuristic fallback",
            })

    return result


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_narrative_clips(
    video_path: str,
    *,
    mode: str = "standalone",
    target_clips: int = 5,
    min_clip_s: float = 15.0,
    max_clip_s: float = 90.0,
    language: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
) -> NarrativeResult:
    """Run the full Narrative Engine pipeline on a video file.

    Parameters
    ----------
    video_path : str
        Absolute path to the input video.
    mode : str
        Clip mode — controls target duration and composite score weights.
        One of: ``standalone``, ``trailer``, ``series``, ``promo``,
        ``highlight``, ``full_narrative``.
    target_clips : int
        Number of clip candidates to return.
    min_clip_s, max_clip_s : float
        Global clip duration bounds (override mode defaults if tighter).
    language : str | None
        ISO 639-1 code for ASR.  ``None`` = auto-detect.
    gemini_api_key : str | None
        Gemini API key.  Falls back to ``GEMINI_API_KEY`` env var.
        When absent, the engine runs in heuristic-only mode.

    Returns
    -------
    NarrativeResult
        Never raises; all failures are captured in ``warnings``.
    """
    warnings: list[str] = []

    # ── Step 1: probe duration ────────────────────────────────────────────────
    if not os.path.isfile(video_path):
        warnings.append(f"Video file not found: {video_path!r}")
        return NarrativeResult(
            candidates=[], source_duration=0.0,
            language=language or "unknown", warnings=warnings,
        )

    source_duration = _probe_duration(video_path)
    if source_duration <= 0.0:
        warnings.append("Could not determine video duration; pipeline may produce poor results.")
        source_duration = 0.0

    logger.info("narrative: video=%s duration=%.1fs mode=%s", video_path, source_duration, mode)

    # ── Step 2: Gemini key resolution ─────────────────────────────────────────
    api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    use_gemini = bool(api_key)
    if not use_gemini:
        warnings.append(
            "GEMINI_API_KEY is not set. Running in heuristic-only mode — "
            "narrative_role will be 'unlabeled' and importance_score defaults to 0.5."
        )

    # ── Step 3: ASR transcription ─────────────────────────────────────────────
    try:
        from pipeline_core.asr import transcribe  # type: ignore
        asr_result = transcribe(video_path, language=language)
        warnings.extend(asr_result.warnings)
        sentences = asr_result.sentences
        detected_language = asr_result.language
        logger.info(
            "narrative: ASR produced %d sentences, language=%s",
            len(sentences), detected_language,
        )
    except Exception as exc:
        logger.exception("narrative: ASR failed: %s", exc)
        warnings.append(f"ASR failed: {exc}; continuing without transcript.")
        sentences = []
        detected_language = language or "unknown"

    # ── Step 4: Shot detection ────────────────────────────────────────────────
    try:
        from pipeline_core.shot_detect import detect_shots  # type: ignore
        shot_boundaries = detect_shots(video_path, method="scdet", threshold=0.4)
        shot_times = [b.t for b in shot_boundaries]
        logger.info("narrative: %d shot boundaries detected", len(shot_times))
    except Exception as exc:
        logger.warning("narrative: shot detection failed: %s", exc)
        warnings.append(f"Shot detection failed: {exc}")
        shot_times = []

    # ── Step 5: Audio RMS valleys ─────────────────────────────────────────────
    try:
        valley_times = _compute_rms_valleys(video_path, top_k=_TOP_VALLEYS)
        logger.info("narrative: %d RMS valleys found", len(valley_times))
    except Exception as exc:
        logger.warning("narrative: RMS valley extraction failed: %s", exc)
        warnings.append(f"RMS valley extraction failed: {exc}")
        valley_times = []

    # ── Step 6: Gemini narrative labeling ─────────────────────────────────────
    # full_narrative → force target_clips=1
    if mode == "full_narrative":
        target_clips = 1

    gemini_points: Optional[list[dict]] = None
    if use_gemini and sentences:
        try:
            transcript_block = _build_transcript_block(sentences)
            prompt = _GEMINI_PROMPT_TEMPLATE.format(
                n=target_clips,
                duration=source_duration,
                language=detected_language,
                shot_count=len(shot_times),
                transcript_block=transcript_block,
            )
            gemini_points = _call_gemini(prompt, api_key)
        except Exception as exc:
            logger.warning("narrative: Gemini call preparation failed: %s", exc)
            warnings.append(f"Gemini labeling failed: {exc}")
            gemini_points = None

    if gemini_points is None:
        if use_gemini:
            warnings.append("Gemini returned no usable output; using heuristic fallback.")
    # Ensure gemini_points is set even if Gemini was not attempted
    # (heuristic fallback applied below)

    # ── Build turning points list ──────────────────────────────────────────────
    # Determine mode clip duration window
    mode_bounds = _MODE_DURATION.get(mode, _DEFAULT_MODE_DURATION)
    mode_min_s = max(min_clip_s, mode_bounds[0])
    mode_max_s = min(max_clip_s, mode_bounds[1])

    # Fallback to heuristic if Gemini not available or returned nothing
    if not gemini_points:
        turning_points = _heuristic_turning_points(
            sentences, target_clips, mode_min_s, mode_max_s, source_duration
        )
    else:
        turning_points = gemini_points

    if not turning_points:
        warnings.append("No turning points found; returning empty candidates.")
        return NarrativeResult(
            candidates=[],
            source_duration=source_duration,
            language=detected_language,
            warnings=warnings,
        )

    # ── Steps 7–9: Build, snap, and score clip candidates ─────────────────────
    from pipeline_core.clip_boundaries import snap_boundaries  # type: ignore

    candidates: list[ClipCandidate] = []

    for tp in turning_points:
        anchor_idx = int(tp.get("anchor_index", 0))
        narrative_role = str(tp.get("narrative_role", "unlabeled")).lower()
        if narrative_role not in _VALID_ROLES:
            narrative_role = "unlabeled"
        importance = float(tp.get("importance_score", 0.5))
        importance = max(0.0, min(1.0, importance))
        gemini_reason = tp.get("reason", "")

        # Anchor to a sentence if available
        if sentences and 0 <= anchor_idx < len(sentences):
            anchor_sent = sentences[anchor_idx]
            anchor_mid = (anchor_sent.start + anchor_sent.end) / 2.0
        elif sentences:
            anchor_sent = sentences[len(sentences) // 2]
            anchor_mid = (anchor_sent.start + anchor_sent.end) / 2.0
        else:
            # No ASR — distribute evenly
            if target_clips > 1:
                step = source_duration / (target_clips + 1)
                anchor_mid = step * (turning_points.index(tp) + 1)
            else:
                anchor_mid = source_duration / 2.0

        # Step 7: build proposed window centred on anchor
        half = (mode_min_s + mode_max_s) / 4.0  # half of average target duration
        proposed_start = max(0.0, anchor_mid - half)
        proposed_end = min(source_duration or (anchor_mid + half * 2), anchor_mid + half)

        # Clamp proposed duration to mode bounds
        proposed_dur = proposed_end - proposed_start
        if proposed_dur < mode_min_s and source_duration > mode_min_s:
            proposed_end = min(
                source_duration, proposed_start + mode_min_s
            )
        if proposed_dur > mode_max_s:
            proposed_end = proposed_start + mode_max_s

        # Step 8: snap to signals
        try:
            snap = snap_boundaries(
                proposed_start,
                proposed_end,
                shots=shot_times,
                sentences=sentences if sentences else None,
                valleys=valley_times if valley_times else None,
                search_window_s=2.0,
                min_clip_duration_s=mode_min_s,
                max_clip_duration_s=mode_max_s,
            )
            clipped_start = snap.start
            clipped_end = snap.end
            start_sources = snap.start_sources
            end_sources = snap.end_sources
            warnings.extend(snap.warnings)
        except Exception as exc:
            logger.warning("narrative: snap_boundaries failed: %s", exc)
            warnings.append(f"snap_boundaries error: {exc}")
            clipped_start = proposed_start
            clipped_end = proposed_end
            start_sources = []
            end_sources = []

        # Step 9: score
        hook = _hook_score(sentences, clipped_start, clipped_end)
        completion = _completion_score(sentences, clipped_start, clipped_end)
        composite = _composite_score(importance, hook, completion, mode)

        transcript_text = _transcript_slice(sentences, clipped_start, clipped_end)
        duration = clipped_end - clipped_start

        meta: dict = {
            "gemini_reason": gemini_reason,
            "anchor_index": anchor_idx,
            "proposed_start": proposed_start,
            "proposed_end": proposed_end,
        }
        if mode == "series":
            meta["series_chaining"] = True  # TODO: implement cross-clip continuity in v2

        candidates.append(ClipCandidate(
            start=clipped_start,
            end=clipped_end,
            duration=duration,
            narrative_role=narrative_role,
            hook_score=hook,
            completion_score=completion,
            importance_score=importance,
            composite_score=composite,
            transcript_slice=transcript_text,
            start_sources=start_sources,
            end_sources=end_sources,
            meta=meta,
        ))

    # ── Step 10: sort and trim ─────────────────────────────────────────────────
    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    top_candidates = candidates[:target_clips]

    logger.info(
        "narrative: returning %d/%d candidates (mode=%s, gemini=%s)",
        len(top_candidates), len(candidates), mode, use_gemini and gemini_points is not None,
    )

    return NarrativeResult(
        candidates=top_candidates,
        source_duration=source_duration,
        language=detected_language,
        warnings=warnings,
    )
