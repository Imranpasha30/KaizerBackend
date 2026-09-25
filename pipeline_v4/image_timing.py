"""V4 image↔speech timing engine — the Phase-1 "brain".

Replaces the blind 4-second image cycle: an LLM reads the story's
per-word timestamps (TrimmedStory.words) plus the image manifest
(pool_index + subject label — the "name-tag contract") and places each
image on the exact spoken words. Gaps are expected — the renderer cuts
back to the main video (Unit 5).

Guarantees, in order of importance:
  1. A paid job NEVER fails because of timing: any error, empty plan or
     garbage JSON → the caller falls back to the legacy heuristic
     (``orchestrator._default_image_timings``).
  2. Raw model JSON is never trusted — ``sanitize_timing_plan`` enforces
     bounds, overlap-freedom, dwell limits (polish C), anchor-word
     verification + a confidence gate (polish D) and same-label entity
     consistency (polish B).
  3. Operator-pinned windows (``timing_mode="pinned"``) are reserved
     verbatim; the model and sanitizer route around them.

Model routing (DEV .env only — no code edit to swap):
  KAIZER_V4_IMAGE_TIMING          1|0     master switch (default 1)
  KAIZER_V4_IMAGE_TIMING_MODEL    provider:model, default
                                  "gemini:gemini-2.5-flash" (the same
                                  Vertex client every other V4 aux call
                                  uses); "claude:<model>" switches to the
                                  Anthropic path.
  KAIZER_V4_IMAGE_TIMING_MIN_CONF confidence gate, default 0.55
  KAIZER_V4_IMAGE_MIN_DWELL       seconds, default 2.5
  KAIZER_V4_IMAGE_MAX_DWELL       seconds, default 8.0
"""
from __future__ import annotations

import json
import os
from typing import Optional

from pipeline_v4 import prompts as v4_prompts
from pipeline_v4.trim_engine import _loads_lenient, _gemini_repair_json, _anthropic_log


# ─── Env knobs ──────────────────────────────────────────────────────

def _enabled() -> bool:
    return (os.environ.get("KAIZER_V4_IMAGE_TIMING", "1") or "1").strip().lower() \
        not in ("0", "false", "no", "off")


def _model_spec() -> tuple[str, str]:
    """Return (provider, model) from KAIZER_V4_IMAGE_TIMING_MODEL.
    Accepts "provider:model" or a bare model name (→ gemini). Unknown
    provider falls back to the default so an env typo never crashes."""
    raw = (os.environ.get("KAIZER_V4_IMAGE_TIMING_MODEL")
           or "gemini:gemini-2.5-flash").strip()
    if ":" in raw:
        prov, _, model = raw.partition(":")
        prov = prov.strip().lower()
        model = model.strip()
        if prov in ("gemini", "claude") and model:
            return prov, model
        print(f"[v4/img-timing] bad KAIZER_V4_IMAGE_TIMING_MODEL={raw!r}, "
              f"using default", flush=True)
        return "gemini", "gemini-2.5-flash"
    return "gemini", raw or "gemini-2.5-flash"


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def min_confidence() -> float:
    return _float_env("KAIZER_V4_IMAGE_TIMING_MIN_CONF", 0.55)


def min_dwell() -> float:
    return _float_env("KAIZER_V4_IMAGE_MIN_DWELL", 2.5)


def max_dwell() -> float:
    return _float_env("KAIZER_V4_IMAGE_MAX_DWELL", 8.0)


def _spotlight_enabled() -> bool:
    """Master switch for AUTO spotlight selection on built-in templates
    (spec 3.6: built-ins ON by default; custom templates opt in
    elsewhere). Operator overrides in the editor are always honored."""
    return (os.environ.get("KAIZER_V4_SPOTLIGHT", "1") or "1").strip().lower() \
        not in ("0", "false", "no", "off")


def auto_spotlight(plan: list[dict], *, min_dwell_s: float = 2.5,
                   threshold: float = 0.8, cap: int = 2) -> list[dict]:
    """Mark the story's KEY moments for a full-screen pop (spec 3.6/3.7):
    up to ``cap`` windows with importance ≥ ``threshold`` and a readable
    dwell get ``spotlight="fullscreen"`` stamped on the plan entry. The
    decision is materialized HERE (canvas-build/resync time) so the
    editor shows it and the render cache hashes it — never a hidden
    render-time choice. No-op when KAIZER_V4_SPOTLIGHT=0."""
    if not plan or not _spotlight_enabled():
        return plan
    candidates = sorted(
        (e for e in plan
         if float(e.get("importance") or 0.0) >= threshold
         and (float(e["t_end"]) - float(e["t_start"])) >= min_dwell_s),
        key=lambda e: float(e.get("importance") or 0.0),
        reverse=True,
    )
    for e in candidates[:max(0, cap)]:
        e["spotlight"] = "fullscreen"
    return plan


# ─── Model calls ────────────────────────────────────────────────────

def _call_gemini(system: str, user: str, model: str) -> str:
    """Same Vertex client + JSON-mime + self-repair pattern as the
    KEEP/CUT alternate planner (trim_engine._gemini_keep_cut_plan)."""
    from seo.generator import _gemini_client
    from google.genai import types as genai_types
    # Space request starts across the whole process: a 16-story job fires
    # 16 timing calls at once and Vertex's PER-MINUTE quota 429s nearly
    # all of them into 30s sleeps + heuristic fallbacks (quality loss).
    # Paced, the same calls all succeed (job 611: ~13min of collisions).
    from pipeline_v4.api_pace import pace
    pace()

    client = _gemini_client()
    resp = client.models.generate_content(
        model=model,
        contents=user,
        config=genai_types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            temperature=0.1,
            max_output_tokens=8192,
        ),
    )
    raw = (resp.text or "").strip()
    if not raw:
        raise RuntimeError("gemini image-timing returned empty body")
    try:
        _loads_lenient(raw)   # parse test only — caller parses for real
        return raw
    except json.JSONDecodeError:
        fixed = _gemini_repair_json(client, genai_types, model, raw)
        _loads_lenient(fixed)
        print("[v4/img-timing] gemini JSON repaired on retry", flush=True)
        return fixed


def _call_claude(system: str, user: str, model: str) -> str:
    """Anthropic path, mirroring trim_engine._claude_keep_cut_plan."""
    from anthropic import Anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in env")
    client = Anthropic(api_key=api_key)
    with _anthropic_log(model=model, purpose="image-timing") as _acall:
        msg = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        _acall.record(msg)
    return msg.content[0].text if msg.content else ""


def _call_model(system: str, user: str) -> str:
    provider, model = _model_spec()
    if provider == "claude":
        return _call_claude(system, user, model)
    return _call_gemini(system, user, model)


# ─── Sanitizer (pure — never trust raw model JSON) ──────────────────

def _normalize_label(label: str) -> str:
    return " ".join((label or "").casefold().split())


def sanitize_timing_plan(
    entries: list,
    *,
    duration: float,
    pool_count: int,
    pool_labels: Optional[list[str]] = None,
    words: Optional[list[dict]] = None,
    pinned: tuple = (),
    min_dwell_s: float = 2.5,
    max_dwell_s: float = 8.0,
    min_conf: float = 0.55,
) -> Optional[list[dict]]:
    """Turn raw model output into a safe, non-overlapping plan.

    Returns a sorted list of
      {pool_index, t_start, t_end, confidence, importance, matched_text}
    or None when nothing survives (caller falls back to the heuristic).

    Steps (each documented at the site):
      validate → clamp/snap → route around pinned → de-overlap →
      dwell limits (polish C) → anchor-verify + confidence gate
      (polish D) → entity consistency (polish B).
    """
    if duration <= 0 or pool_count <= 0:
        return None
    words = words or []
    pool_labels = pool_labels or []

    # 1) Validate raw entries.
    plan: list[dict] = []
    for e in (entries or []):
        if not isinstance(e, dict):
            continue
        try:
            idx = int(e.get("pool_index"))
            ts = float(e.get("t_start"))
            te = float(e.get("t_end"))
        except (TypeError, ValueError):
            continue
        if not (0 <= idx < pool_count):
            continue
        if te <= ts + 0.05:
            continue
        try:
            conf = float(e.get("confidence", 0.7) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        try:
            imp = float(e.get("importance", 0.0) or 0.0)
        except (TypeError, ValueError):
            imp = 0.0
        anchors = e.get("anchor_word_indexes")
        anchors = [a for a in anchors if isinstance(a, int)] if isinstance(anchors, list) else []
        plan.append({
            "pool_index": idx,
            "t_start": ts, "t_end": te,
            "confidence": max(0.0, min(conf, 1.0)),
            "importance": max(0.0, min(imp, 1.0)),
            "anchors": anchors,
        })
    if not plan:
        return None

    # 2) Clamp to [0, duration]; snap edges within 0.3s of the bounds so
    #    near-misses don't leave sliver gaps at the very start/end.
    for e in plan:
        if e["t_start"] < 0.3:
            e["t_start"] = 0.0
        if e["t_end"] > duration - 0.3:
            e["t_end"] = duration
        e["t_start"] = max(0.0, min(e["t_start"], duration))
        e["t_end"] = max(0.0, min(e["t_end"], duration))
    plan = [e for e in plan if e["t_end"] > e["t_start"] + 0.05]

    # 3) Pinned windows are reserved verbatim: clip model windows that
    #    overlap them (drop when fully covered).
    for (pa, pb) in (pinned or ()):
        nxt: list[dict] = []
        for e in plan:
            if e["t_end"] <= pa or e["t_start"] >= pb:
                nxt.append(e)                       # no overlap
            elif e["t_start"] < pa and e["t_end"] <= pb:
                e["t_end"] = pa; nxt.append(e)      # clip tail
            elif e["t_start"] >= pa and e["t_end"] > pb:
                e["t_start"] = pb; nxt.append(e)    # clip head
            elif e["t_start"] < pa and e["t_end"] > pb:
                e["t_end"] = pa; nxt.append(e)      # keep the front piece
            # else: fully inside the reserved interval → dropped
        plan = [e for e in nxt if e["t_end"] > e["t_start"] + 0.05]
    if not plan:
        return None

    # 4) Sort; resolve remaining overlaps by trimming the LATER window.
    plan.sort(key=lambda e: (e["t_start"], e["t_end"]))
    deoverlapped: list[dict] = []
    for e in plan:
        if deoverlapped and e["t_start"] < deoverlapped[-1]["t_end"]:
            e["t_start"] = deoverlapped[-1]["t_end"]
        if e["t_end"] > e["t_start"] + 0.05:
            deoverlapped.append(e)
    plan = deoverlapped

    # 5) Dwell limits (polish C): too short → extend into free space up
    #    to min dwell, else drop (a flash-frame is worse than no image);
    #    too long → truncate (stale image).
    kept: list[dict] = []
    for i, e in enumerate(plan):
        dwell = e["t_end"] - e["t_start"]
        if dwell > max_dwell_s:
            e["t_end"] = e["t_start"] + max_dwell_s
        elif dwell < min_dwell_s:
            nxt_start = plan[i + 1]["t_start"] if i + 1 < len(plan) else duration
            e["t_end"] = min(e["t_start"] + min_dwell_s, nxt_start, duration)
            if e["t_end"] - e["t_start"] < min_dwell_s - 0.01:
                continue   # can't reach a readable dwell — drop
        kept.append(e)
    plan = kept
    if not plan:
        return None

    # 6) Anchor verification + confidence gate (polish D): the claimed
    #    anchor words must exist and their midpoint must fall inside the
    #    window — otherwise halve the confidence. Windows below the gate
    #    become gaps (renderer cuts to the main video).
    for e in plan:
        ok = False
        for a in e["anchors"]:
            if 0 <= a < len(words):
                try:
                    mid = (float(words[a].get("s", 0.0)) + float(words[a].get("e", 0.0))) / 2.0
                except (TypeError, ValueError):
                    continue
                if e["t_start"] - 0.5 <= mid <= e["t_end"] + 0.5:
                    ok = True
                    break
        if not ok:
            e["confidence"] *= 0.5
        e["matched_text"] = " ".join(
            str(words[a].get("w", "")) for a in e["anchors"] if 0 <= a < len(words)
        )[:200]
    plan = [e for e in plan if e["confidence"] >= min_conf]
    if not plan:
        return None

    # 7) Entity consistency (polish B): two pool images with the same
    #    normalized label are the same ENTITY — later windows reuse the
    #    first-used image so a person never changes face mid-story.
    label_to_idx: dict[str, int] = {}
    for e in plan:
        idx = e["pool_index"]
        lab = _normalize_label(pool_labels[idx]) if idx < len(pool_labels) else ""
        if not lab:
            continue
        if lab in label_to_idx:
            e["pool_index"] = label_to_idx[lab]
        else:
            label_to_idx[lab] = idx

    for e in plan:
        e.pop("anchors", None)
        e["t_start"] = round(e["t_start"], 3)
        e["t_end"] = round(e["t_end"], 3)
        e["confidence"] = round(e["confidence"], 3)
        e["importance"] = round(e["importance"], 3)
    return plan


# ─── Entry point ────────────────────────────────────────────────────

def decide_story_timings(
    *,
    title_native: str = "",
    title_english: str = "",
    summary: str = "",
    duration: float,
    words: list[dict],
    pool: list[dict],
    pinned_windows: tuple = (),
    language: str = "",
) -> Optional[list[dict]]:
    """Transcript-grounded timing plan for ONE story.

    ``pool`` — [{label, kind}, …]; list index == pool_index in the result.
    ``words`` — story-relative [{"w","s","e"}, …] (TrimmedStory.words).

    Returns sanitized [{pool_index, t_start, t_end, confidence,
    importance, matched_text}, …] or None → caller uses the legacy
    heuristic. Never raises."""
    try:
        if not _enabled():
            return None
        if duration <= 0 or not pool:
            return None
        if not words:
            # No transcript grounding → the model would just be guessing;
            # the 4s heuristic is more honest here.
            print("[v4/img-timing] no word timestamps for this story — "
                  "using heuristic timings", flush=True)
            return None

        system = v4_prompts.IMAGE_TIMING_SYSTEM_V2.format(
            min_dwell=min_dwell(), max_dwell=max_dwell(),
        )
        user = v4_prompts.build_image_timing_user_prompt_v2(
            story_title=title_native or title_english,
            story_title_english=title_english,
            story_summary=summary,
            duration_sec=duration,
            words=words,
            image_pool=pool,
            pinned_windows=list(pinned_windows or ()),
            language=language,
        )
        # Vertex 429 = PER-MINUTE quota (the render's other AI calls can
        # drain it); one 30s retry rides into the next window instead of
        # silently downgrading 50+ images to the blind 4s heuristic
        # (operator-hit on job 594).
        try:
            raw = _call_model(system, user)
        except Exception as _tx:
            if "429" not in str(_tx):
                raise
            print("[v4/img-timing] 429 per-minute quota — retrying in 30s",
                  flush=True)
            import time as _t
            _t.sleep(30)
            raw = _call_model(system, user)
        data = _loads_lenient(raw)
        entries = data.get("images") if isinstance(data, dict) else None
        plan = sanitize_timing_plan(
            entries or [],
            duration=duration,
            pool_count=len(pool),
            pool_labels=[str(p.get("label") or "") for p in pool],
            words=words,
            pinned=pinned_windows,
            min_dwell_s=min_dwell(),
            max_dwell_s=max_dwell(),
            min_conf=min_confidence(),
        )
        if plan:
            plan = auto_spotlight(plan, min_dwell_s=min_dwell())
            provider, model = _model_spec()
            n_spot = sum(1 for e in plan if e.get("spotlight"))
            print(f"[v4/img-timing] {provider}:{model} placed {len(plan)} "
                  f"window(s) over {duration:.1f}s"
                  f"{f' ({n_spot} spotlight)' if n_spot else ''}", flush=True)
        else:
            print("[v4/img-timing] no windows survived sanitize — "
                  "using heuristic timings", flush=True)
        return plan
    except Exception as exc:
        print(f"[v4/img-timing] timing call failed ({exc}) — "
              f"using heuristic timings", flush=True)
        return None
