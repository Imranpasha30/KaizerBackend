"""Step 5.0 -- V1 pipeline regression test on the upgraded SDK.

PURPOSE
=======
The surface-layer check already confirmed:
  - google-genai 2.4.0 imports cleanly
  - v1 pipeline.py imports under the new SDK
  - client.models.generate_content() still works with v1's call signature

This script extends that coverage to the parts the surface check
skipped:

  - client.files.upload(file=..., config=UploadFileConfig(...))
    [v1 calls this with the new SDK pattern at pipeline_core/pipeline.py L906]
  - client.files.get(name=...) polling until ACTIVE
  - client.models.generate_content(model, contents=[video_file, prompt])
    with a REAL video reference (not just a text-only prompt)
  - client.files.delete(name=...) cleanup
  - JSON response parses + matches v1's expected schema shape

It does this by directly calling v1's ``analyze_video_with_gemini()``
function, which IS the production code path -- not a re-implementation.

If this script reports PASS, the SDK upgrade is regression-safe across
the full Gemini surface v1 uses. If it reports FAIL, roll back via:
    pip install google-genai==1.75.0

COST + TIME
===========
~$0.50 in Gemini API charges (video upload + analysis on ~325 MB mp4).
5-10 minutes wall time (upload + polling + Gemini analysis).

OUTPUTS
=======
- pipeline_v2/tests/fixtures/step5_diag/v1_regression_output.json
  (raw response_json from this run)
- Console PASS / FAIL report with field-by-field structural comparison
  against a baseline (job 20260516_161342, persisted gemini_analysis.json).

USAGE
=====
    python pipeline_v2/scripts/step5_0_v1_regression.py
    # or override fixture / baseline:
    python pipeline_v2/scripts/step5_0_v1_regression.py \\
        --fixture "E:/kaizer new data training/videos/test.mp4" \\
        --baseline path/to/gemini_analysis.json \\
        --platform youtube_short \\
        --language te
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# UTF-8 stdout for Telugu glyphs.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# --- Path bootstrap ----------------------------------------------------

HERE = Path(__file__).resolve().parent
PIPELINE_V2_ROOT = HERE.parent
KAIZER_BACKEND = PIPELINE_V2_ROOT.parent
# v1 imports require KaizerBackend on sys.path.
for p in (PIPELINE_V2_ROOT, KAIZER_BACKEND):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


# --- Default fixture + baseline ---------------------------------------

DEFAULT_FIXTURE = r"E:/kaizer new data training/videos/test.mp4"
# Step-0-accurate baseline captured during the 2026-05-18 SOLO regression
# (mode=single, platform=youtube_short, te). The OLD baseline at
# output/youtube_short/20260516_161342/gemini_analysis.json was PRE-Step-0
# and produced false-positive failures on the (correctly-evolved) new
# schema -- do not reuse it.
DEFAULT_BASELINE = (
    PIPELINE_V2_ROOT / "tests" / "fixtures" / "step5_baseline"
    / "v1_step0_solo.json"
)
DEFAULT_PLATFORM = "youtube_short"
DEFAULT_LANGUAGE = "te"


# --- Schema expectations (Step-0 actual, locked 2026-05-18) ----------

# BASE_KEYS are emitted in BOTH single (SOLO) and compound modes.
# Step 0's _BASE_SCHEMA_BLOCK lists them; _COMPOUND_SCHEMA_BLOCK is
# base ∪ compound. Verified from the 2026-05-18 SOLO regression run.
BASE_KEYS = [
    "video_type",
    "language",
    "total_speakers",
    "clips",
    "overall_summary",
    "overall_summary_native",
    "image_search_queries",
    "key_people",
    "key_people_native",
    "key_topics",
    "key_locations",
    "shorts_cuts",
    "image_plan",
    "skipped_segments",
    "retake_audit",
]

# COMPOUND_KEYS are checked ONLY when video_type=="COMPOUND". In SOLO
# mode the schema doesn't require these (Step 0's _BASE_SCHEMA_BLOCK
# is used for single mode, which omits the bulletin/marquee bundle).
COMPOUND_KEYS = [
    "full_video_cuts",   # the bulletin cut bundle, compound-only
]

# Per-list-item required keys: ACTUAL shape after Step 0's prompt swap
# evolved the schema. Key changes from pre-Step-0:
#   clips:            'hook' renamed to 'summary'; +summary_native, mood, speakers
#   shorts_cuts:      'index'/'hook'/'importance' removed;
#                     +summary, shorts_headline_native, bulletin_marquee_points
#                     (last two were top-level in v1; moved per-item in Step 0)
#   image_plan:       id, topic_clue, search_query, search_query_native, reason REMOVED
#                     kept: clip_index, entity_name, show_at, duration, description
#   full_video_cuts:  unchanged shape (compound-only)
#   skipped_segments: unchanged shape
REQUIRED_LIST_ITEM_KEYS = {
    "clips":            ["index", "start", "end", "summary",
                         "summary_native", "mood", "speakers", "importance"],
    "full_video_cuts":  ["index", "start", "end", "summary",
                         "summary_native", "importance"],
    "shorts_cuts":      ["start", "end", "summary",
                         "shorts_headline_native", "bulletin_marquee_points"],
    "image_plan":       ["clip_index", "entity_name", "show_at",
                         "duration", "description"],
    "skipped_segments": ["start", "end", "reason", "category"],
}

# Categories the Step 0 prompt-swap locked in. Any value outside this
# set is a regression of the forbid-invention block.
VALID_SKIPPED_CATEGORIES = {
    "warm_up", "retake", "crew_talk", "hesitation",
    "aside", "self_correction",
}


# --- .env loader ------------------------------------------------------


def _load_env() -> None:
    env_file = KAIZER_BACKEND / ".env"
    if not env_file.is_file():
        sys.exit(f"FAIL: .env not found at {env_file}")
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if k not in os.environ and v:
            os.environ[k] = v


# --- Output helpers ---------------------------------------------------


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f" {title}")
    print("=" * 72)


def info(msg: str) -> None:
    print(f"  ...    {msg}")


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


# --- Main -------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=DEFAULT_FIXTURE, type=str)
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE), type=str,
                    help="Path to baseline gemini_analysis.json from a "
                         "prior PASS job. Used for structural comparison "
                         "(field names + per-item shapes), NOT value "
                         "equality -- different audio produces different "
                         "cuts, but the JSON shape should match.")
    ap.add_argument("--platform", default=DEFAULT_PLATFORM, type=str,
                    choices=["instagram_reel", "youtube_short", "youtube_full"])
    ap.add_argument("--language", default=DEFAULT_LANGUAGE, type=str)
    args = ap.parse_args()

    _load_env()

    banner("Step 5.0 v1 regression -- full Gemini surface under google-genai 2.4.0")

    # ---- Sanity: SDK version ----------------------------------------
    import google.genai as gen
    sdk_ver = getattr(gen, "__version__", "<unknown>")
    info(f"google-genai version: {sdk_ver}")
    if not sdk_ver.startswith("2."):
        warn(f"expected 2.x; got {sdk_ver}. Did the upgrade revert?")

    # ---- Fixture --------------------------------------------------
    fixture = Path(args.fixture).resolve()
    if not fixture.is_file():
        sys.exit(f"FAIL: fixture not found at {fixture}")
    size_mb = fixture.stat().st_size / (1024 * 1024)
    info(f"fixture: {fixture}  ({size_mb:.1f} MB)")

    # ---- Baseline --------------------------------------------------
    baseline_path = Path(args.baseline).resolve()
    baseline = None
    if baseline_path.is_file():
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            info(f"baseline: {baseline_path}  ({len(baseline)} top-level keys)")
        except Exception as exc:
            warn(f"baseline at {baseline_path} unreadable: {exc}")
            baseline = None
    else:
        warn(f"baseline not found at {baseline_path} -- "
             f"structural comparison will use BASE_KEYS / COMPOUND_KEYS only")

    # ---- Key check ------------------------------------------------
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        sys.exit("FAIL: GEMINI_API_KEY missing in .env")

    # ---- Import v1's actual function ------------------------------
    info("importing v1 pipeline_core.pipeline ...")
    import pipeline_core.pipeline as v1pipe
    preset = v1pipe.PLATFORM_PRESETS[args.platform]
    info(f"using preset: {args.platform} "
         f"(min_dur={preset.get('min_dur')} max_dur={preset.get('max_dur')} "
         f"ideal_dur={preset.get('ideal_dur')})")

    # ---- THE BIG ONE: call v1's analyze_video_with_gemini ---------
    # This exercises:
    #   1. client.files.upload(file=, config=UploadFileConfig)
    #   2. client.files.get(name=) polling until ACTIVE
    #   3. client.models.generate_content(model=, contents=[video, prompt])
    #   4. client.files.delete(name=)
    # All four surfaces that surface check skipped.
    banner("Invoking v1.analyze_video_with_gemini()  (~5-10 min, ~$0.50 cost)")
    t0 = time.perf_counter()
    try:
        result = v1pipe.analyze_video_with_gemini(
            video_path=str(fixture),
            preset=preset,
            language=args.language,
            mode="single",
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        fail(f"v1 analyze_video_with_gemini FAILED after {elapsed:.1f}s: "
             f"{type(exc).__name__}: {exc}")
        return 1
    elapsed = time.perf_counter() - t0
    ok(f"v1 call returned in {elapsed:.1f}s")

    # ---- Persist raw output ---------------------------------------
    out_dir = (PIPELINE_V2_ROOT / "tests" / "fixtures" / "step5_diag")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v1_regression_output.json"
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    info(f"raw output saved: {out_path}")

    # ---- Structural validation ------------------------------------
    failures: list[str] = []
    warnings: list[str] = []

    banner("Structural validation")

    # 1. Top-level keys present? Mode-aware:
    #    - BASE_KEYS required in both SOLO and COMPOUND.
    #    - COMPOUND_KEYS required only when video_type == "COMPOUND".
    video_type = (result.get("video_type") or "").upper().strip()
    info(f"video_type: {video_type or '<missing>'}")

    missing_base = [k for k in BASE_KEYS if k not in result]
    if missing_base:
        failures.append(f"missing BASE keys: {missing_base}")
        fail(f"BASE keys: missing {missing_base}")
    else:
        ok(f"BASE keys: all {len(BASE_KEYS)} required present")

    if video_type == "COMPOUND":
        missing_compound = [k for k in COMPOUND_KEYS if k not in result]
        if missing_compound:
            failures.append(f"missing COMPOUND keys: {missing_compound}")
            fail(f"COMPOUND keys: missing {missing_compound}")
        else:
            ok(f"COMPOUND keys: all {len(COMPOUND_KEYS)} required present "
               f"(video_type=COMPOUND)")
    else:
        info(f"COMPOUND keys: skipped (video_type={video_type or '?'}, "
             f"not COMPOUND)")

    # 2. Per-list-item shapes
    for list_key, required_item_keys in REQUIRED_LIST_ITEM_KEYS.items():
        # full_video_cuts is COMPOUND-only. Skip the per-item shape
        # check entirely in SOLO mode (the key may be absent or empty).
        if list_key == "full_video_cuts" and video_type != "COMPOUND":
            info(f"{list_key}: skipped (COMPOUND-only)")
            continue
        items = result.get(list_key)
        if not isinstance(items, list):
            if items is None and list_key not in BASE_KEYS:
                # COMPOUND-only key missing in SOLO is OK
                continue
            failures.append(f"{list_key!r} is not a list (got {type(items).__name__})")
            fail(f"{list_key}: not a list")
            continue
        if not items:
            warnings.append(f"{list_key!r} is empty list (audio-dependent; not a hard failure)")
            warn(f"{list_key}: empty (audio-dependent)")
            continue
        first = items[0]
        if not isinstance(first, dict):
            failures.append(f"{list_key}[0] is not a dict (got {type(first).__name__})")
            fail(f"{list_key}[0]: not a dict")
            continue
        missing_item_keys = [k for k in required_item_keys if k not in first]
        if missing_item_keys:
            failures.append(f"{list_key}[0] missing keys: {missing_item_keys}")
            fail(f"{list_key}[0]: missing {missing_item_keys}")
        else:
            ok(f"{list_key}[0]: all {len(required_item_keys)} required item keys present  "
               f"(list length: {len(items)})")

    # 3. skipped_segments categories must be in the locked enum
    skipped = result.get("skipped_segments") or []
    if skipped:
        observed_cats = set()
        invented = set()
        for seg in skipped:
            if not isinstance(seg, dict):
                continue
            cat = seg.get("category", "")
            observed_cats.add(cat)
            if cat not in VALID_SKIPPED_CATEGORIES:
                invented.add(cat)
        if invented:
            failures.append(
                f"skipped_segments invented categories: {invented} "
                f"(allowed: {VALID_SKIPPED_CATEGORIES}). Step 0's "
                f"forbid-invention block may have regressed."
            )
            fail(f"skipped_segments: invented categories {invented}")
        else:
            ok(f"skipped_segments: all categories in locked enum "
               f"({observed_cats})")

    # 4. retake_audit present + non-empty (Step 0 locked this as mandatory)
    retake_audit = result.get("retake_audit", "")
    if not isinstance(retake_audit, str) or not retake_audit.strip():
        failures.append(
            f"retake_audit is empty/missing (got {retake_audit!r}). "
            f"Step 0's HARD RULES make this mandatory."
        )
        fail("retake_audit: empty (Step 0 regression)")
    elif retake_audit.upper() == "SKIPPED":
        failures.append(
            "retake_audit == 'SKIPPED' (Step 0 forbid 'SKIPPED' value)"
        )
        fail("retake_audit: forbidden SKIPPED value")
    else:
        ok(f"retake_audit: non-empty ({len(retake_audit)} chars)")

    # 5. Baseline comparison (structure only, not values)
    if baseline:
        banner("Baseline shape comparison (job 20260516_161342)")
        baseline_keys = set(baseline.keys())
        result_keys = set(result.keys())
        only_in_baseline = baseline_keys - result_keys
        only_in_new = result_keys - baseline_keys
        if only_in_baseline:
            failures.append(
                f"keys missing in new run vs baseline: {only_in_baseline}"
            )
            fail(f"missing vs baseline: {only_in_baseline}")
        if only_in_new:
            warn(f"new run has extra keys (not in baseline): {only_in_new} "
                 f"-- not a hard failure, possibly a prompt evolution")
        if not only_in_baseline and not only_in_new:
            ok("top-level key set matches baseline byte-for-byte")

        # Compare per-list-item key sets
        for list_key in REQUIRED_LIST_ITEM_KEYS:
            b_items = baseline.get(list_key) or []
            r_items = result.get(list_key) or []
            if not b_items or not r_items:
                continue
            b_keys = set(b_items[0].keys()) if isinstance(b_items[0], dict) else set()
            r_keys = set(r_items[0].keys()) if isinstance(r_items[0], dict) else set()
            only_b = b_keys - r_keys
            only_r = r_keys - b_keys
            if only_b:
                failures.append(
                    f"{list_key}[0] missing keys vs baseline: {only_b}"
                )
                fail(f"{list_key}[0] vs baseline: missing {only_b}")
            elif only_r:
                warn(f"{list_key}[0] has extra keys vs baseline: {only_r}")
            else:
                ok(f"{list_key}[0]: per-item keys match baseline")

    # ---- Summary --------------------------------------------------
    banner("Summary")
    print(f"  cost (Gemini API):   ~$0.50 (1 video upload + 1 generate_content)")
    print(f"  wall time:           {elapsed:.1f}s")
    print(f"  output saved:        {out_path}")
    print(f"  warnings:            {len(warnings)}")
    print(f"  failures:            {len(failures)}")
    print()
    if warnings:
        for w in warnings:
            print(f"   [WARN] {w}")
        print()
    if failures:
        print("  ====== REGRESSION FAILED ======")
        for f in failures:
            print(f"   [FAIL] {f}")
        print()
        print("  Recommendation: roll back to google-genai==1.75.0")
        print("    pip install google-genai==1.75.0")
        return 2

    print("  ====== REGRESSION PASS ======")
    print()
    print("  The full Gemini surface v1 uses (files.upload + polling +")
    print("  generate_content + files.delete) works under google-genai==2.4.0.")
    print("  Stage 2 build is regression-safe to proceed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
