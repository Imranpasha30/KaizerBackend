"""Pre-compose creative-quality scorer (Kaizer-native).

ADOPTED from OpenMontage's slideshow_risk.py + variation_checker.py — but
rebuilt on Kaizer's ACTUAL plan (canvas.json `stories[]`), not OpenMontage's
scene vocabulary (shot_size / camera_movement / shot_intent / hero_moment),
which Kaizer's news cut-plan does not have. Same idea: score the plan for
"slideshow-feel" / monotony BEFORE we burn an NVENC slot, so a technically-
valid-but-visually-monotonous clip can be flagged (advisory) or regenerated.

A Kaizer clip (bulletin or a short) is a list of STORIES. Each story has:
  video_t_start / video_t_end   -> duration
  images[]                      -> visual richness (B-roll / picture-in-picture)
  text_blocks[]                 -> on-screen text density
  title_native / summary        -> content

Dimensions (each 0-5, LOWER is better):
  visual_starvation : stories with no images = animated-text slideshow risk
  pacing_monotony   : story durations all near-identical = no editorial rhythm
  text_overload     : lots of text_blocks but few images = caption wall
  image_reuse       : same image reused across stories = repetitive visuals
  shot_density      : too few stories over a long runtime = static long-takes

Verdict (mirrors OpenMontage): <2 strong, <3 acceptable, <4 revise, >=4 fail.
ADVISORY by design — callers log the verdict (and may trigger ONE regenerate);
it must never hard-block, since thresholds need tuning on real Telugu-news
distributions. Pure-Python, dependency-free.
"""
from __future__ import annotations

from collections import Counter
from typing import Any


def _verdict(avg: float) -> str:
    if avg < 2.0:
        return "strong"
    if avg < 3.0:
        return "acceptable"
    if avg < 4.0:
        return "revise"
    return "fail"


def _story_duration(s: dict) -> float:
    try:
        return max(0.0, float(s.get("video_t_end", 0)) - float(s.get("video_t_start", 0)))
    except (TypeError, ValueError):
        return 0.0


def _img_keys(s: dict) -> list[str]:
    """Normalize a story's images[] to comparable keys (filename/url/str)."""
    out = []
    for im in (s.get("images") or []):
        if isinstance(im, dict):
            out.append(str(im.get("filename") or im.get("path") or im.get("url") or im))
        else:
            out.append(str(im))
    return out


def _dim_visual_starvation(stories: list[dict]) -> dict[str, Any]:
    n = len(stories)
    starved = sum(1 for s in stories if not _img_keys(s))
    ratio = starved / n
    score = min(5.0, ratio * 5.0)
    if ratio > 0.6:
        reason = f"{starved}/{n} stories have NO images — plays like animated text"
    elif ratio > 0.3:
        reason = f"{starved}/{n} stories have no images — thin visuals"
    else:
        reason = "Most stories carry visuals"
    return {"score": round(score, 1), "reason": reason}


def _dim_pacing_monotony(stories: list[dict], total_dur: float) -> dict[str, Any]:
    durs = [_story_duration(s) for s in stories]
    durs = [d for d in durs if d > 0]
    if len(durs) < 3:
        return {"score": 0.0, "reason": "Too few timed stories to assess pacing"}
    mean = sum(durs) / len(durs)
    if mean <= 0:
        return {"score": 0.0, "reason": "No story durations"}
    var = sum((d - mean) ** 2 for d in durs) / len(durs)
    cv = (var ** 0.5) / mean  # coefficient of variation
    # Low CV = every story the same length = metronomic. High CV = varied rhythm.
    if cv < 0.15:
        score, reason = 3.5, f"Story lengths nearly identical (CV={cv:.2f}) — metronomic pacing"
    elif cv < 0.30:
        score, reason = 1.5, f"Modest pacing variety (CV={cv:.2f})"
    else:
        score, reason = 0.0, f"Good pacing variety (CV={cv:.2f})"
    return {"score": score, "reason": reason}


def _dim_text_overload(stories: list[dict]) -> dict[str, Any]:
    n = len(stories)
    tb = sum(len(s.get("text_blocks") or []) for s in stories)
    imgs = sum(len(_img_keys(s)) for s in stories)
    per_story = tb / n
    # Lots of text + few images = caption wall.
    if per_story >= 3 and imgs <= n:
        score, reason = 3.5, f"{tb} text blocks vs {imgs} images over {n} stories — text-dominant"
    elif per_story >= 2 and imgs < n:
        score, reason = 2.0, f"Text-leaning ({tb} blocks / {imgs} images)"
    else:
        score, reason = 0.5, "Text and visuals balanced"
    return {"score": score, "reason": reason}


def _dim_image_reuse(stories: list[dict]) -> dict[str, Any]:
    keys = [k for s in stories for k in _img_keys(s)]
    if len(keys) < 3:
        return {"score": 0.0, "reason": "Too few images to assess reuse"}
    uniq_ratio = len(set(keys)) / len(keys)
    if uniq_ratio < 0.5:
        score, reason = 3.0, f"Only {uniq_ratio:.0%} of images are unique — heavy reuse"
    elif uniq_ratio < 0.75:
        score, reason = 1.5, f"{uniq_ratio:.0%} unique images — some reuse"
    else:
        score, reason = 0.0, "Varied imagery"
    most = Counter(keys).most_common(1)[0]
    if most[1] >= max(3, len(keys) * 0.5):
        score = min(5.0, score + 1.0)
        reason += f"; one image used {most[1]}x"
    return {"score": round(score, 1), "reason": reason}


def _dim_shot_density(stories: list[dict], total_dur: float) -> dict[str, Any]:
    if total_dur <= 0:
        return {"score": 0.0, "reason": "Unknown runtime"}
    spm = len(stories) / (total_dur / 60.0)  # stories per minute
    # News tolerates long talking segments, so this is lenient: only flag very
    # sparse cutting on longer pieces.
    if total_dur > 90 and spm < 0.8:
        score, reason = 2.5, f"{len(stories)} stories over {total_dur:.0f}s ({spm:.1f}/min) — long static takes"
    elif total_dur > 60 and spm < 0.5:
        score, reason = 1.5, f"Sparse cutting ({spm:.1f} stories/min)"
    else:
        score, reason = 0.0, f"Adequate cut density ({spm:.1f} stories/min)"
    return {"score": score, "reason": reason}


def score_clip_quality(clip: dict, kind: str = "") -> dict[str, Any]:
    """Score one Kaizer clip (a bulletin dict or a short dict with stories[])."""
    stories = clip.get("stories") or []
    kind = kind or clip.get("kind") or "clip"
    if not stories:
        return {"average": 5.0, "verdict": "fail", "kind": kind,
                "dimensions": {"empty": {"score": 5.0, "reason": "No stories in plan"}}}
    total_dur = sum(_story_duration(s) for s in stories)
    dims = {
        "visual_starvation": _dim_visual_starvation(stories),
        "pacing_monotony": _dim_pacing_monotony(stories, total_dur),
        "text_overload": _dim_text_overload(stories),
        "image_reuse": _dim_image_reuse(stories),
        "shot_density": _dim_shot_density(stories, total_dur),
    }
    avg = sum(d["score"] for d in dims.values()) / len(dims)
    return {"average": round(avg, 2), "verdict": _verdict(avg), "kind": kind,
            "stories": len(stories), "duration": round(total_dur, 1), "dimensions": dims}


def score_canvas_quality(canvas: dict) -> dict[str, Any]:
    """Score a full V4 canvas.json (the bulletin + each short). Returns a
    structured report; callers decide what to do with low verdicts."""
    report: dict[str, Any] = {"bulletin": None, "shorts": [], "worst_verdict": "strong"}
    order = {"strong": 0, "acceptable": 1, "revise": 2, "fail": 3}
    worst = 0
    if isinstance(canvas.get("bulletin"), dict):
        report["bulletin"] = score_clip_quality(canvas["bulletin"], "bulletin")
        worst = max(worst, order[report["bulletin"]["verdict"]])
    for i, s in enumerate(canvas.get("shorts") or []):
        r = score_clip_quality(s, f"short_{i+1:02d}")
        report["shorts"].append(r)
        worst = max(worst, order[r["verdict"]])
    report["worst_verdict"] = [k for k, v in order.items() if v == worst][0]
    return report
