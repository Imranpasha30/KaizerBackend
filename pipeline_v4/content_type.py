"""Content-type detection — the system identifies what it is editing.

"Feed ANY raw video" needs the pipeline to know whether it is looking at
a news bulletin, a podcast, an interview or a vlog BEFORE the cut
planner runs — the edit profile (persona, grouping, silence tightening)
depends on it. Detection is one cheap LLM call over the transcript head
Stage 1 already produced for free.

Resolution order (first hit wins):
  1. Operator's explicit wizard pick — ``KAIZER_V4_CONTENT_TYPE`` set to
     a concrete type (the "ask the user" option; "auto"/"" means detect).
  2. LLM classification with confidence ≥ KAIZER_V4_TYPE_MIN_CONF (0.6).
  3. Fallback: **news** — today's fleet is news content, so an unsure
     classifier must land on the behavior every existing job already
     gets. The sidecar records ``needs_confirmation: true`` so the UI
     can ask the operator next time.

The decision is recorded to ``content_type.json`` in the job dir and
exported as ``KAIZER_V4_EDIT_PROFILE`` for the planner. Fail-soft
everywhere — classification can never break a render.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from pipeline_v4.edit_profiles import PROFILES

SIDECAR_NAME = "content_type.json"
VALID_TYPES = tuple(PROFILES.keys())          # news / podcast / interview / vlog / generic
_FALLBACK = "news"

_CLASSIFY_SYSTEM = """You classify a spoken-word video from its transcript. Output ONE JSON object, no prose.

Types (pick exactly one):
- "news"      — anchor/reporter reading news items; formal register; multiple distinct stories
- "podcast"   — multi-person conversation; casual register; long turns; hosts/guests
- "interview" — clear question-and-answer structure between an interviewer and a guest
- "vlog"      — single creator speaking to camera about their day/topic; informal, first person
- "generic"   — anything else (lecture, tutorial, speech, sermon, product demo…)

Schema: {"type": "<one of the five>", "confidence": 0.0-1.0, "reasons": "<one line>"}
Confidence is YOUR certainty; if the excerpt is ambiguous, say so with a low number."""


def _min_conf() -> float:
    try:
        return float(os.environ.get("KAIZER_V4_TYPE_MIN_CONF", "") or 0.6)
    except (TypeError, ValueError):
        return 0.6


def _classify_llm(words: list[dict], duration: float,
                  language: str = "") -> Optional[dict]:
    """One Gemini-flash call over the transcript head/tail. None on any
    failure — caller falls back."""
    try:
        from seo.generator import _gemini_client
        from google.genai import types as genai_types
        from pipeline_v4.trim_engine import _loads_lenient

        toks = [str(w.get("w", "")).strip() for w in (words or []) if w.get("w")]
        if len(toks) < 12:
            return None                      # too little speech to judge
        head = " ".join(toks[:180])
        tail = " ".join(toks[-60:]) if len(toks) > 240 else ""
        user = (f"Duration: {duration:.0f}s. Words: {len(toks)}. "
                f"Language hint: {language or 'unknown'}.\n\n"
                f"Transcript start:\n{head}\n"
                + (f"\nTranscript end:\n{tail}\n" if tail else ""))
        model = os.environ.get("KAIZER_V4_TYPE_MODEL", "gemini-2.5-flash")
        client = _gemini_client()
        # 512 tokens TRUNCATED the JSON mid-string on thinking models
        # ("Unterminated string…") so EVERY job fell back to the generic
        # "news" formula (operator-hit on jobs 575/592/594). 2048 gives
        # headroom; one retry rides out per-minute-quota blips and
        # transient truncations.
        data = None
        for _at in range(2):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=user,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=_CLASSIFY_SYSTEM,
                        response_mime_type="application/json",
                        temperature=0.1,
                        max_output_tokens=2048,
                    ),
                )
                data = _loads_lenient((resp.text or "").strip())
                break
            except Exception as _cx:
                if _at == 0:
                    if "429" in str(_cx):
                        import time as _t
                        _t.sleep(30)
                    continue
                raise
        t = str(data.get("type", "")).strip().lower()
        conf = float(data.get("confidence", 0.0) or 0.0)
        if t not in VALID_TYPES:
            return None
        return {"type": t, "confidence": max(0.0, min(conf, 1.0)),
                "reasons": str(data.get("reasons", ""))[:300]}
    except Exception as exc:
        print(f"[v4/type] classification failed ({exc}) — falling back", flush=True)
        return None


def resolve_and_record(*, words: list[dict], duration: float,
                       out_dir=None, language: str = "") -> str:
    """Decide the job's content type, export KAIZER_V4_EDIT_PROFILE for
    the planner, persist the decision sidecar. Returns the profile key.
    Never raises."""
    try:
        explicit = (os.environ.get("KAIZER_V4_CONTENT_TYPE") or "auto").strip().lower()
        record: dict
        if explicit in VALID_TYPES:
            key = explicit
            record = {"schema": 1, "type": key, "source": "user",
                      "confidence": 1.0, "needs_confirmation": False}
            print(f"[v4/type] operator pick: {key}", flush=True)
        else:
            det = _classify_llm(words, duration, language)
            if det and det["confidence"] >= _min_conf():
                key = det["type"]
                record = {"schema": 1, "type": key, "source": "detected",
                          "confidence": det["confidence"],
                          "reasons": det.get("reasons", ""),
                          "needs_confirmation": False}
                print(f"[v4/type] detected: {key} "
                      f"(confidence {det['confidence']:.2f}) — {det.get('reasons', '')}",
                      flush=True)
            else:
                key = _FALLBACK
                record = {"schema": 1, "type": key,
                          "source": "fallback",
                          "confidence": (det or {}).get("confidence", 0.0),
                          "detected_type": (det or {}).get("type"),
                          "needs_confirmation": True}
                print(f"[v4/type] unsure "
                      f"({(det or {}).get('type')}@{(det or {}).get('confidence', 0)}) "
                      f"— using '{key}' profile; flagged for operator confirmation",
                      flush=True)
        os.environ["KAIZER_V4_EDIT_PROFILE"] = key
        if out_dir is not None:
            try:
                (Path(out_dir) / SIDECAR_NAME).write_text(
                    json.dumps(record, ensure_ascii=False, indent=1),
                    encoding="utf-8")
            except OSError:
                pass
        return key
    except Exception as exc:
        print(f"[v4/type] resolve failed (soft): {exc}", flush=True)
        os.environ["KAIZER_V4_EDIT_PROFILE"] = _FALLBACK
        return _FALLBACK
