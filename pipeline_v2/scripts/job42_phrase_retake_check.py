"""Manual Stage 2 verification: phrase-level retake detection on Job 42.

CONTEXT
=======
Gemini Pro audited Job 42's bulletin and graded it 4/10 "amateur
patchwork" because Stage 2 missed phrase-level false starts. Gemini
identified 7+ retake spots in Job 42's source audio that the
production Stage 2 prompt did not catch:

  - 00:24  first-attempt restart
  - 01:27
  - 01:44
  - 02:08
  - 04:10
  - 04:27
  - 05:25 / 05:34

The Stage 2 prompt has now been updated with an explicit
"Phrase-level retake detection" section + 3 new few-shot examples
(stage_2_prompt.md). This script re-runs Stage 2 on Job 42's
transcript with the updated prompt and reports which of the 7
Gemini-identified spots are now caught.

Acceptance bar (per user): >=5 of 7 detected.

USAGE
=====
    cd e:/kaizer\ new\ data\ training/kaizer/KaizerBackend
    python pipeline_v2/scripts/job42_phrase_retake_check.py

Output: JSON + console table written to
    pipeline_v2/scripts/job42_phrase_retake_result.json
    pipeline_v2/scripts/job42_phrase_retake_result.txt

COST
====
~$0.10-0.20 (one Gemini 2.5 Pro call on a 1167-word transcript).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Windows: force UTF-8 on stdout so Telugu/Hindi reason strings don't
# crash the console (cp1252 cannot encode Devanagari/Telugu).
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# --- Make the package importable when run as a script ---------------
HERE = Path(__file__).resolve().parent
BACKEND_ROOT = HERE.parent.parent     # .../KaizerBackend
PROJECT_ROOT = BACKEND_ROOT.parent     # .../kaizer
PIPELINE_V2_ROOT = BACKEND_ROOT / "pipeline_v2"
sys.path.insert(0, str(PIPELINE_V2_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

# Load env BEFORE importing Stage 2 (it reads GEMINI_API_KEY on call).
from dotenv import load_dotenv  # noqa: E402
load_dotenv(BACKEND_ROOT / ".env", override=True)

from pipeline_v2.models import (  # noqa: E402
    Stage1Output,
    Word,
    WordLevelTranscript,
)
from pipeline_v2.stages.stage_2_continuity import (  # noqa: E402
    Stage2ContinuityEditor,
)


JOB_42_DIR = (
    BACKEND_ROOT / "output" / "full_video_shorts_v2" / "job_42"
)

# Gemini-Pro-identified phrase-retake spots (seconds, MM:SS in the source).
# A detection counts as "caught" if any skipped_segment of category
# `retake` overlaps within +/- TOLERANCE_SEC of the spot.
GEMINI_SPOTS_SEC = [
    ("00:24", 24.0),
    ("01:27", 87.0),
    ("01:44", 104.0),
    ("02:08", 128.0),
    ("04:10", 250.0),
    ("04:27", 267.0),
    ("05:25", 325.0),
    ("05:34", 334.0),
]
TOLERANCE_SEC = 5.0   # accept detection within +/- 5s of the marker


def _load_stage1_output() -> Stage1Output:
    transcript_path = JOB_42_DIR / "transcript.json"
    meta_path = JOB_42_DIR / "stt_metadata.json"

    transcript_raw = json.loads(transcript_path.read_text(encoding="utf-8"))
    meta_raw = json.loads(meta_path.read_text(encoding="utf-8"))

    words = [
        Word(
            w=w["w"],
            s=float(w["s"]),
            e=float(w["e"]),
            speaker=w.get("speaker"),
            confidence=w.get("confidence"),
        )
        for w in transcript_raw["words"]
    ]

    transcript = WordLevelTranscript(
        words=words,
        duration_sec=float(meta_raw["stt_audio_duration_sec"]),
        detected_languages=[meta_raw.get("stt_language_detected", "te")],
        provider=meta_raw["stt_provider"],
    )

    return Stage1Output(
        transcript=transcript,
        transcript_json_path=str(transcript_path),
        metadata_json_path=str(meta_path),
        stt_provider=meta_raw["stt_provider"],
        stt_audio_duration_sec=float(meta_raw["stt_audio_duration_sec"]),
        stt_wall_seconds=float(meta_raw["stt_wall_seconds"]),
        stt_cost_usd=float(meta_raw["stt_cost_usd"]),
        stt_word_count=int(meta_raw["stt_word_count"]),
        stt_avg_confidence=meta_raw.get("stt_avg_confidence"),
        stt_language_detected=meta_raw.get("stt_language_detected", "te"),
        stt_request_id=meta_raw.get("stt_request_id", "manual-job42"),
        stt_language_hint=meta_raw.get("stt_language_hint", "te"),
        stt_brief=meta_raw.get("stt_brief", ""),
        stt_names=meta_raw.get("stt_names", []),
    )


def _format_mmss(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"


async def main() -> int:
    print("=" * 70)
    print("Stage 2 phrase-retake verification on Job 42")
    print("=" * 70)

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY is not set in environment.", file=sys.stderr)
        return 1

    print(f"[1/4] Loading Job 42 transcript from {JOB_42_DIR} ...")
    stage1 = _load_stage1_output()
    n_words = len(stage1.transcript.words)
    duration = stage1.stt_audio_duration_sec
    print(f"      {n_words} words, {duration:.1f}s audio")

    print("[2/4] Calling Stage 2 with updated prompt (gemini-2.5-pro)...")
    editor = Stage2ContinuityEditor()
    decisions = await editor.transcribe_to_decisions(stage1)
    n_cuts = len(decisions.full_video_cuts)
    n_skipped = len(decisions.skipped_segments)
    n_retakes = sum(
        1 for s in decisions.skipped_segments if s.category.value == "retake"
    )
    print(
        f"      Stage 2 returned: {n_cuts} cuts, {n_skipped} skipped "
        f"segments ({n_retakes} of category=retake)"
    )

    print("[3/4] Matching against Gemini-Pro-identified spots...")
    matches: list[dict] = []
    for label, spot_sec in GEMINI_SPOTS_SEC:
        hit = None
        for seg in decisions.skipped_segments:
            if seg.category.value != "retake":
                continue
            if (
                seg.start_sec - TOLERANCE_SEC
                <= spot_sec
                <= seg.end_sec + TOLERANCE_SEC
            ):
                hit = seg
                break
        matches.append({
            "label": label,
            "spot_sec": spot_sec,
            "caught": hit is not None,
            "matched_span": (
                f"{hit.start_sec:.2f}-{hit.end_sec:.2f}s"
                if hit
                else None
            ),
            "matched_reason": hit.reason if hit else None,
        })

    caught_count = sum(1 for m in matches if m["caught"])
    target = 5
    verdict = "PASS" if caught_count >= target else "FAIL"

    # Write JSON FIRST so the report data is captured even if a
    # console-encoding error later crashes the print path on Windows.
    out_json = HERE / "job42_phrase_retake_result.json"
    out_txt = HERE / "job42_phrase_retake_result.txt"
    payload = {
        "verdict": verdict,
        "caught_count": caught_count,
        "target": target,
        "tolerance_sec": TOLERANCE_SEC,
        "matches": matches,
        "n_cuts": n_cuts,
        "n_skipped": n_skipped,
        "n_retakes": n_retakes,
        "retake_audit": decisions.retake_audit,
        "skipped_segments": [
            {
                "category": s.category.value,
                "start_sec": s.start_sec,
                "end_sec": s.end_sec,
                "start_word_idx": s.start_word_idx,
                "end_word_idx": s.end_word_idx,
                "reason": s.reason,
            }
            for s in decisions.skipped_segments
        ],
        "full_video_cuts": [
            {
                "index": c.index,
                "start_sec": c.start_sec,
                "end_sec": c.end_sec,
                "start_word_idx": c.start_word_idx,
                "end_word_idx": c.end_word_idx,
                "importance": c.importance,
            }
            for c in decisions.full_video_cuts
        ],
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_json}")

    print()
    print("[4/4] Report")
    print()
    print(f"{'spot':>8}  {'sec':>7}  status   matched_span")
    print("-" * 70)
    for m in matches:
        status = "CAUGHT " if m["caught"] else "MISSED "
        span = m["matched_span"] or "-"
        print(f"{m['label']:>8}  {m['spot_sec']:>7.1f}  {status}  {span}")
    print()
    print(f"Total caught: {caught_count}/{len(matches)} "
          f"(target >= {target}). Verdict: {verdict}")
    print()
    print("All skipped_segments emitted by Stage 2:")
    print(
        f"{'idx':>3}  {'category':<15}  "
        f"{'start':>8}  {'end':>8}  reason"
    )
    print("-" * 70)
    for i, seg in enumerate(decisions.skipped_segments):
        try:
            reason_snippet = seg.reason[:80]
        except Exception:
            reason_snippet = "<reason unprintable>"
        try:
            print(
                f"{i:>3}  {seg.category.value:<15}  "
                f"{_format_mmss(seg.start_sec):>8}  "
                f"{_format_mmss(seg.end_sec):>8}  "
                f"{reason_snippet}"
            )
        except UnicodeEncodeError:
            print(f"{i:>3}  {seg.category.value:<15}  "
                  f"{_format_mmss(seg.start_sec):>8}  "
                  f"{_format_mmss(seg.end_sec):>8}  "
                  f"<unicode reason; see json>")
    print()
    try:
        print(f"retake_audit: {decisions.retake_audit}")
    except UnicodeEncodeError:
        print("retake_audit: <unicode; see json>")

    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
