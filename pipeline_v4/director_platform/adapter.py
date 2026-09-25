# The BRIDGE between the ported platform AI Director (sensors→formula→LLM,
# 5 moods) and OUR V4 render vocabulary. Not upstream code — new for the
# dual-director merge (kaizer-platform@d5fd482 provided the engine; this file
# makes its decisions renderable by the V4 compose layer).
"""plan_direction_platform() — drop-in alternative to
pipeline_v4.director.plan_direction, selected per job via
Job.v4_director_engine == "platform" (env KAIZER_V4_DIRECTOR_ENGINE).

Design:
  * Sensors run ONCE on the whole trimmed master (the expensive part);
    per story we override speech pace (from that story's words) and
    duration, then run the deterministic formula — free.
  * Layer 3 (LLM confirm/override) runs per story on its title/summary
    text; env KAIZER_V4_PLATFORM_LLM=0 turns it off (formula-only mode).
  * Each story's final STYLE PACK maps onto a V4 StoryDirective: we start
    from OUR category formula_plan baseline (so layouts/overlays/captions
    stay category-correct) and overlay pack-specific transition/fx/grade —
    every id validated against the SAME _vocab(user_picks) the V4 engine
    uses, so user pins are honored and a stale id can never leak. The pack
    (not the mood) is the key because it is the ONLY field llm_refine
    validates — exactly how the vendor consumes its own refine output
    (their director.py renders refined.style_pack; mood is trail data).
  * Fail-soft: any error returns {} and the orchestrator falls back to the
    V4 engine (then formula). A paid job always renders.
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Optional

from pipeline_v4.director_platform.formula import apply_formula
from pipeline_v4.director_platform.llm_refine import refine_with_llm
from pipeline_v4.director_platform.sensors import (
    SensorReadings, _speech_pace_wps, extract_sensors,
)

# Platform style pack → V4 vocabulary overrides. Applied ON TOP of the
# category baseline; every id is validated against _vocab(user_picks) at run
# time and silently dropped (baseline kept) when not available/pinned-away.
# "minimal" deliberately maps to NO overrides = pure category baseline.
# Packs map 1:1 to the formula moods (vibrant↔energetic, news_flash↔urgent,
# calm↔somber, cinematic↔cinematic, minimal↔neutral).
PACK_TO_DIRECTIVE: dict = {
    "vibrant":    {"transition": "zoom_punch_out", "fx": ("vibrance_pop",),
                   "grade": ""},
    "news_flash": {"transition": "whip_left", "fx": ("micro_shake",),
                   "grade": "breaking_red_alert"},
    "calm":       {"transition": "dissolve", "fx": ("film_grain",),
                   "grade": "bleach_bypass", "bed_on": False},
    "cinematic":  {"transition": "fade", "fx": ("soft_glow",),
                   "grade": "cinematic_teal_orange"},
    "minimal":    {},
}


def _llm_enabled() -> bool:
    return (os.environ.get("KAIZER_V4_PLATFORM_LLM", "1") or "1").strip() \
        not in ("0", "false", "off")


def _story_text(story) -> str:
    # CanvasStory's real fields FIRST (title_native/title_english/summary —
    # canvas_schema.py:149); generic names kept last for duck-typed callers.
    parts = []
    for attr in ("title_native", "title_english", "summary",
                 "title", "script", "text"):
        val = getattr(story, attr, None)
        if val:
            parts.append(str(val))
    return "\n".join(parts).strip()


def plan_direction_platform(*, stories, words_by_story, category: str,
                            source_path: str, language: str = "",
                            user_picks: Optional[dict] = None):
    """Platform-engine plan: dict[int, StoryDirective]. {} on total failure
    (caller falls back to the V4 engine). Mirrors plan_direction's kwargs."""
    try:
        return _plan_impl(stories=stories, words_by_story=words_by_story,
                          category=category, source_path=source_path,
                          language=language, user_picks=user_picks)
    except Exception as exc:  # noqa: BLE001 — a paid job must always render
        print(f"[v4/director-platform] plan failed (soft): {exc}", flush=True)
        return {}


def _plan_impl(*, stories, words_by_story, category: str,
               source_path: str, language: str = "",
               user_picks: Optional[dict] = None):
    from pipeline_v4.director import (
        StoryDirective, _directive_dict, _enforce_variety, _vocab,
        _write_trace, formula_plan,
    )

    stories = list(stories or [])
    if not stories:
        return {}

    # Category baseline from OUR engine — layouts, overlays, captions,
    # layout_moments and user pins all handled exactly like the V4 path.
    base_plan = formula_plan(stories, category, user_picks)
    v = _vocab(user_picks)

    # Layer 1 once per video (expensive); per-story fields overridden below.
    global_sensors = extract_sensors(source_path, words=None)
    if global_sensors.warnings:
        print("[v4/director-platform] sensor warnings: "
              + " | ".join(global_sensors.warnings), flush=True)

    trace: dict = {
        "mode": "platform",
        "category": category or "news",
        "language": language or "",
        "user_picks": {k: sorted(vv) for k, vv in (user_picks or {}).items()},
        "sensors": {
            "duration_s": round(global_sensors.duration_s, 1),
            "audio_rms_mean": round(global_sensors.audio_rms_mean, 4),
            "audio_rms_peak": round(global_sensors.audio_rms_peak, 4),
            "integrated_lufs": global_sensors.integrated_lufs,
            "scene_change_rate_per_min":
                round(global_sensors.scene_change_rate_per_min, 1),
            "avg_brightness": round(global_sensors.avg_brightness, 3),
            "avg_saturation": round(global_sensors.avg_saturation, 3),
            "warnings": list(global_sensors.warnings),
        },
        "stories": {},
        "steps": ["platform engine: global sensors measured once; "
                  "per-story pace/duration + 5-rule formula"
                  + ("; per-story LLM tone review" if _llm_enabled()
                     else "; LLM layer OFF (KAIZER_V4_PLATFORM_LLM=0)")],
    }

    llm_on = _llm_enabled()
    out = {}
    for s in stories:
        idx = int(getattr(s, "story_index", 0) or 0)
        directive = base_plan.get(idx)
        if directive is None:
            continue

        # Per-story sensor view: this story's pace + duration. The sidecar
        # JSON carries STRING keys ("0") — accept both (the int-only lookup
        # was exactly the silent-miss bug our own director hit on job 610).
        _wbs = words_by_story or {}
        words = _wbs.get(idx) or _wbs.get(str(idx)) or []
        try:
            dur = max(0.5, float(getattr(s, "video_t_end", 0.0))
                      - float(getattr(s, "video_t_start", 0.0)))
        except (TypeError, ValueError):
            dur = global_sensors.duration_s or 30.0
        story_sensors: SensorReadings = replace(
            global_sensors,
            duration_s=dur,
            speech_pace_wps=_speech_pace_wps(words),
        )

        candidate = apply_formula(story_sensors)
        if llm_on:
            refined = refine_with_llm(candidate, story_sensors,
                                      _story_text(s))
            pack, mood, reason = (refined.style_pack, refined.mood,
                                  refined.reason)
            trail = (f"platform:{candidate.rule_id}"
                     + (f"; llm[{refined.provider_used}]:{reason}"
                        if refined.provider_used else f"; {reason}"))
            provider_used = refined.provider_used
            overridden = refined.overridden
        else:
            pack, mood, reason = (candidate.style_pack, candidate.mood,
                                  candidate.reason)
            trail = f"platform:{candidate.rule_id}; {reason}"
            provider_used, overridden = None, False

        # Key the treatment off the PACK — the only LLM-validated field
        # (guaranteed one of the 5 PLATFORM_PACKS); mood is free text.
        over = PACK_TO_DIRECTIVE.get(pack, {})
        # Validate every override id against the V4 vocabulary; invalid or
        # pinned-away ids keep the category baseline (never crash, never leak).
        trans = over.get("transition", "")
        if trans and trans in v["transitions"]:
            directive.transition_in = trans
        fx_over = [f for f in over.get("fx", ()) if f in v["fx"]]
        if fx_over:
            directive.fx = fx_over[:2]
        grade = over.get("grade", "")
        if grade and grade in v["grades"]:
            directive.grade = grade
        if over.get("bed_on") is False:
            directive.bed_on = False
        directive.why = trail[:300]
        out[idx] = directive

        trace["stories"][str(idx)] = {
            "duration_s": round(dur, 1),
            "title": _story_text(s)[:120],
            "rule_id": candidate.rule_id,
            "formula_mood": candidate.mood,
            "mood": mood,
            "style_pack": pack,
            "llm_provider": provider_used,
            "llm_overridden": overridden,
            "reason": reason,
            "speech_pace_wps": round(story_sensors.speech_pace_wps, 2),
        }

    # Anti-monotony: same invariant the V4 engine enforces — adjacent
    # stories never share a transition_in (director._enforce_variety).
    rotated = _enforce_variety(out, v)
    if rotated:
        trace["steps"].append(
            f"variety pass rotated {rotated} repeated transition(s)")

    trace["decisions"] = {str(i): _directive_dict(d)
                          for i, d in sorted(out.items())}
    trace["summary"] = (
        f"{len(out)} stories via platform engine; "
        f"{sum(1 for st in trace['stories'].values() if st['llm_overridden'])}"
        f" LLM override(s)")
    # Same file + shape the job UI's Director-decisions view reads
    # (mode='platform' tells the operator WHICH engine produced it; also
    # overwrites a stale v4 trace when a retry switches engines). Only
    # next to a real source file — never into the CWD on synthetic paths.
    if source_path and os.path.isfile(source_path):
        _write_trace(source_path, trace)

    return out
