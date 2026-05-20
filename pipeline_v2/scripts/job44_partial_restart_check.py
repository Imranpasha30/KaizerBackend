"""Iteration 2 -- Stage 2 partial-restart verification on Job 44.

CONTEXT
=======
Gemini Pro's V2 audit (graded 7.5/10) flagged 4 partial-restart
spots in Job 44 (the C7040.mp4 re-run with the iteration-1 Stage 2
prompt):

  07:11  rapid single-word repeat ('దాంట్లో' 'దాంట్లో' within 0.34s)
  07:33  partial short-phrase restart with divergent completion
  08:02  rapid back-to-back repeat of subject phrase ('వాళ్ళకి')
  08:12  rapid back-to-back repeat ('నాకు బండి సంజయ్' twice)

These are the 4 NEW patterns added to stage_2_prompt.md's "PARTIAL
RESTART DETECTION (Phase 2 patterns)" section in iteration 2.

The 7 spots from iteration 1 (full phrase-level retakes) are also
re-checked to confirm no regression.

Total spot list: 11 (7 iteration-1 + 4 iteration-2).

Acceptance bar (per user): >=9 of 11 caught (any skipped category)
to declare iteration 2 prompt update a success without further
iteration. <8 of 11 = STOP for prompt iteration.

USAGE
=====
    python pipeline_v2/scripts/job44_partial_restart_check.py
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

HERE = Path(__file__).resolve().parent
BACKEND_ROOT = HERE.parent.parent
PIPELINE_V2_ROOT = BACKEND_ROOT / "pipeline_v2"
sys.path.insert(0, str(PIPELINE_V2_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

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


JOB_DIR = BACKEND_ROOT / "output" / "full_video_shorts_v2" / "job_44"

# All 11 spots (iteration-1's 7 + iteration-2's 4). The 05:25/05:34
# combined marker counts as one logical spot (user's iteration-1
# brief grouped them).
ALL_SPOTS = [
    # Iteration 1 spots (full phrase-level retakes):
    ("i1: 00:24", [24.0]),
    ("i1: 01:27", [87.0]),
    ("i1: 01:44", [104.0]),
    ("i1: 02:08", [128.0]),
    ("i1: 04:10", [250.0]),
    ("i1: 04:27", [267.0]),
    ("i1: 05:25/34", [325.0, 334.0]),
    # Iteration 2 spots (partial restart patterns):
    ("i2: 07:11", [431.0]),    # Pattern A: single content-word repeat
    ("i2: 07:33", [453.0]),    # Pattern D: partial short-phrase restart
    ("i2: 08:02", [482.0]),    # Pattern C: back-to-back subject repeat
    ("i2: 08:12", [492.0]),    # Pattern C: 'నాకు బండి సంజయ్' twice
]
TOLERANCE_SEC = 5.0


def _load_stage1_output() -> Stage1Output:
    transcript_path = JOB_DIR / "transcript.json"
    meta_path = JOB_DIR / "stt_metadata.json"
    transcript_raw = json.loads(transcript_path.read_text(encoding="utf-8"))
    meta_raw = json.loads(meta_path.read_text(encoding="utf-8"))
    words = [
        Word(
            w=w["w"], s=float(w["s"]), e=float(w["e"]),
            speaker=w.get("speaker"), confidence=w.get("confidence"),
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
        stt_request_id=meta_raw.get("stt_request_id", "manual-job44"),
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
    print("Iteration 2: Stage 2 partial-restart verification on Job 44")
    print("=" * 70)
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY unset.", file=sys.stderr)
        return 1

    print(f"[1/4] Loading Job 44 transcript from {JOB_DIR} ...")
    stage1 = _load_stage1_output()
    print(f"      {len(stage1.transcript.words)} words, "
          f"{stage1.stt_audio_duration_sec:.1f}s audio")

    print("[2/4] Calling Stage 2 with updated prompt (gemini-2.5-pro)...")
    editor = Stage2ContinuityEditor()
    decisions = await editor.transcribe_to_decisions(stage1)
    n_cuts = len(decisions.full_video_cuts)
    n_skipped = len(decisions.skipped_segments)
    by_cat: dict[str, int] = {}
    for s in decisions.skipped_segments:
        by_cat[s.category.value] = by_cat.get(s.category.value, 0) + 1
    print(f"      {n_cuts} cuts, {n_skipped} skipped segments")
    print(f"      Categories: {by_cat}")

    print("[3/4] Matching against the 11 spots...")
    matches: list[dict] = []
    for label, sec_candidates in ALL_SPOTS:
        hit_any = None
        hit_retake = None
        for cand in sec_candidates:
            for seg in decisions.skipped_segments:
                if not (seg.start_sec - TOLERANCE_SEC <= cand <= seg.end_sec + TOLERANCE_SEC):
                    continue
                if hit_any is None:
                    hit_any = seg
                if seg.category.value == "retake" and hit_retake is None:
                    hit_retake = seg
        matches.append({
            "label": label,
            "spot_sec": sec_candidates,
            "caught_any": hit_any is not None,
            "caught_retake": hit_retake is not None,
            "matched_any_span": (
                f"{hit_any.start_sec:.2f}-{hit_any.end_sec:.2f}s"
                if hit_any else None
            ),
            "matched_any_category": (hit_any.category.value if hit_any else None),
        })

    caught_any = sum(1 for m in matches if m["caught_any"])
    caught_retake = sum(1 for m in matches if m["caught_retake"])
    target_any = 9
    target_retake = 7
    verdict_any = "PASS" if caught_any >= target_any else "FAIL"
    verdict_retake = "PASS" if caught_retake >= target_retake else "FAIL"

    # Write JSON FIRST
    out_json = HERE / "job44_partial_restart_result.json"
    payload = {
        "verdict_any": verdict_any,
        "verdict_retake": verdict_retake,
        "caught_any": caught_any,
        "caught_retake": caught_retake,
        "target_any": target_any,
        "target_retake": target_retake,
        "tolerance_sec": TOLERANCE_SEC,
        "matches": matches,
        "n_cuts": n_cuts,
        "n_skipped": n_skipped,
        "by_category": by_cat,
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
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_json}")

    print()
    print("[4/4] Report")
    print()
    print(f"{'spot':>14}  status (any)   status (retake)   matched span (any-category)")
    print("-" * 90)
    for m in matches:
        any_s = "CAUGHT" if m["caught_any"] else "MISSED"
        ret_s = "CAUGHT" if m["caught_retake"] else "MISSED"
        span = m["matched_any_span"] or "-"
        cat = m["matched_any_category"] or "-"
        print(f"{m['label']:>14}  {any_s:>13}  {ret_s:>15}  {span} ({cat})")
    print()
    print(f"Caught (any category):  {caught_any}/{len(matches)}  "
          f"(target >= {target_any}). Verdict: {verdict_any}")
    print(f"Caught (retake-only):   {caught_retake}/{len(matches)}  "
          f"(target >= {target_retake}). Verdict: {verdict_retake}")
    print()

    return 0 if verdict_any == "PASS" else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
