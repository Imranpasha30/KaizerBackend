"""Step 4 diagnostic: raw faster-whisper output, untouched.

Bypasses pipeline_v2.stages.stage_1_transcribe entirely. Goes straight
to ``faster_whisper.WhisperModel.transcribe()`` so we see exactly what
the library returns BEFORE any of our code transforms it.

Runs the SAME fixture twice:

  Config A: word_timestamps=True, vad_filter=True   (our current behaviour)
  Config B: word_timestamps=True, vad_filter=False  (isolates VAD's role)

For each: prints info object + first 10 segments with full word arrays.

Usage:
    python pipeline_v2/scripts/step4_diagnose.py --fixture <path> [--language te]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

# Force UTF-8 on stdout/stderr so Telugu glyphs (and the model's
# multilingual punctuation strings, which contain CJK chars) don't blow
# up Windows' default cp1252 console.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# --- Path bootstrap (so the script can be invoked from any cwd) --------

HERE = Path(__file__).resolve().parent
PIPELINE_V2_ROOT = HERE.parent
KAIZER_BACKEND = PIPELINE_V2_ROOT.parent
for p in (PIPELINE_V2_ROOT, KAIZER_BACKEND):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f" {title}")
    print("=" * 72)


def kv(k: str, v) -> None:
    print(f"  {k:32s} {v}")


def _ns_to_dict(obj):
    """Convert a faster-whisper Segment/Word (named tuple-ish) to a dict
    suitable for JSON pretty-printing. Falls back to repr() for fields
    we can't serialise (e.g. token arrays)."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_ns_to_dict(x) for x in obj]
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _ns_to_dict(v) for k, v in asdict(obj).items()}
    if hasattr(obj, "_asdict"):       # named tuple
        return {k: _ns_to_dict(v) for k, v in obj._asdict().items()}
    if hasattr(obj, "__dict__"):
        return {k: _ns_to_dict(v) for k, v in vars(obj).items()
                if not k.startswith("_")}
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return repr(obj)


def run_config(model, audio_path: str, *, label: str, kwargs: dict) -> dict:
    banner(f"{label}: model.transcribe() kwargs as called")
    for k, v in sorted(kwargs.items()):
        kv(k, repr(v))

    # Run the transcribe. We materialise segments here (the iterator
    # streams them) so we can re-inspect later.
    t0 = time.perf_counter()
    segments_iter, info = model.transcribe(audio_path, **kwargs)
    segments = list(segments_iter)
    elapsed = time.perf_counter() - t0

    banner(f"{label}: TranscriptionInfo")
    kv("language", info.language)
    kv("language_probability", f"{info.language_probability:.4f}")
    kv("duration", f"{info.duration:.3f}s")
    duration_after_vad = getattr(info, "duration_after_vad", None)
    kv("duration_after_vad", f"{duration_after_vad:.3f}s" if duration_after_vad is not None else "n/a")
    kv("transcription_options", getattr(info, "transcription_options", "n/a"))
    kv("vad_options", getattr(info, "vad_options", "n/a"))
    kv("(wall time)", f"{elapsed:.1f}s")
    kv("(segments returned)", len(segments))

    banner(f"{label}: first 10 segments (raw)")
    for i, seg in enumerate(segments[:10]):
        # Each Segment has .id .seek .start .end .text .tokens .words
        # .avg_logprob .compression_ratio .no_speech_prob etc.
        print(f"  -- segment[{i}] id={getattr(seg, 'id', '?')} seek={getattr(seg, 'seek', '?')} --")
        kv("start", f"{seg.start:.3f}s")
        kv("end", f"{seg.end:.3f}s")
        kv("duration", f"{seg.end - seg.start:.3f}s")
        text_preview = seg.text[:140] + ("..." if len(seg.text) > 140 else "")
        kv("text", repr(text_preview))
        kv("avg_logprob", f"{getattr(seg, 'avg_logprob', float('nan')):.3f}")
        kv("compression_ratio", f"{getattr(seg, 'compression_ratio', float('nan')):.3f}")
        kv("no_speech_prob", f"{getattr(seg, 'no_speech_prob', float('nan')):.3f}")
        kv("temperature", f"{getattr(seg, 'temperature', float('nan'))}")

        words = seg.words
        if words is None:
            print("    seg.words: None    <-- word_timestamps failed for this segment")
        elif len(words) == 0:
            print("    seg.words: []      <-- empty")
        else:
            print(f"    seg.words ({len(words)}):")
            for j, w in enumerate(words):
                print(f"      [{j:3d}]  {w.start:7.3f}s -> {w.end:7.3f}s   '{w.word}'   p={w.probability:.3f}")

    # Persist full raw output for offline diff
    out_dir = Path("e:/kaizer new data training/kaizer/KaizerBackend/pipeline_v2/tests/fixtures/step4_diag")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"raw_{label.lower()}.json"
    full = {
        "label": label,
        "kwargs": {k: (v if isinstance(v, (str, int, float, bool, type(None))) else repr(v))
                   for k, v in kwargs.items()},
        "wall_seconds": elapsed,
        "info": {
            "language": info.language,
            "language_probability": float(info.language_probability),
            "duration": float(info.duration),
            "duration_after_vad": float(duration_after_vad) if duration_after_vad is not None else None,
        },
        "segments": [_ns_to_dict(s) for s in segments],
    }
    out_file.write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    kv("(full raw dump)", str(out_file))
    return {"info": info, "segments": segments, "wall_seconds": elapsed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True, type=str)
    ap.add_argument("--language", default="te", type=str,
                    help="Language hint to pin (default: te). Use 'auto' to omit.")
    args = ap.parse_args()

    fixture = Path(args.fixture).resolve()
    if not fixture.is_file():
        sys.exit(f"fixture not found: {fixture}")

    language_hint = None if args.language == "auto" else args.language

    banner("Diagnostic harness -- bypassing pipeline_v2.stages.stage_1_transcribe")
    kv("fixture", str(fixture))
    kv("fixture size MB", f"{fixture.stat().st_size / (1024*1024):.1f}")
    kv("language_hint", repr(language_hint))

    # Load the model with the SAME settings as the wrapper (so we
    # reproduce the production environment exactly).
    from faster_whisper import WhisperModel
    banner("Loading WhisperModel(large-v3, cuda, int8_float16)")
    t0 = time.perf_counter()
    model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
    kv("(model load wall)", f"{time.perf_counter() - t0:.1f}s")

    # Config A: pre-fix defaults (the buggy config we started with).
    # Kept here so future regressions are diffable against the original
    # failure mode.
    config_a = dict(
        language=language_hint,
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
    )
    # Config B: same as A minus VAD, to isolate whether VAD is the
    # source of any drift.
    config_b = dict(
        language=language_hint,
        word_timestamps=True,
        vad_filter=False,
        beam_size=5,
    )
    # Config C: the POST-FIX production config. Mirrors the kwargs in
    # pipeline_v2.stages.stage_1_transcribe._transcribe_sync exactly.
    # If you change one place, change the other.
    config_c = dict(
        language=language_hint,
        word_timestamps=True,
        condition_on_previous_text=False,
        no_repeat_ngram_size=3,
        compression_ratio_threshold=1.8,
        hallucination_silence_threshold=2.0,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
    )

    a = run_config(model, str(fixture), label="A_VAD_ON", kwargs=config_a)
    b = run_config(model, str(fixture), label="B_VAD_OFF", kwargs=config_b)
    c = run_config(model, str(fixture), label="C_POSTFIX", kwargs=config_c)

    # Side-by-side summary of first 5 words (the user's specific ask).
    banner("Side-by-side: first 5 WORDS in each config")
    print(f"  {'idx':<4} {'A (PRE-FIX)':<50} {'B (VAD OFF)':<50} {'C (POST-FIX)':<50}")

    def first_n_words(segments, n):
        out = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    out.append(w)
                    if len(out) >= n:
                        return out
        return out

    wa = first_n_words(a["segments"], 5)
    wb = first_n_words(b["segments"], 5)
    wc = first_n_words(c["segments"], 5)
    for i in range(max(len(wa), len(wb), len(wc))):
        la = f"{wa[i].start:6.2f}-{wa[i].end:6.2f}  '{wa[i].word}'" if i < len(wa) else "(no more)"
        lb = f"{wb[i].start:6.2f}-{wb[i].end:6.2f}  '{wb[i].word}'" if i < len(wb) else "(no more)"
        lc = f"{wc[i].start:6.2f}-{wc[i].end:6.2f}  '{wc[i].word}'" if i < len(wc) else "(no more)"
        print(f"  {i:<4} {la:<50} {lb:<50} {lc:<50}")

    # Word-count comparison — the headline number for "did the fix work?"
    def count_words(segments):
        return sum(len(s.words) for s in segments if s.words)
    banner("Word-count summary")
    kv("A (PRE-FIX)   total words", count_words(a["segments"]))
    kv("A (PRE-FIX)   segments",    len(a["segments"]))
    kv("B (VAD OFF)   total words", count_words(b["segments"]))
    kv("B (VAD OFF)   segments",    len(b["segments"]))
    kv("C (POST-FIX)  total words", count_words(c["segments"]))
    kv("C (POST-FIX)  segments",    len(c["segments"]))
    kv("C (POST-FIX)  wall time",   f"{c['wall_seconds']:.1f}s")

    banner("Done -- raw JSON dumps in tests/fixtures/step4_diag/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
