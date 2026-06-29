"""AI-generate-to-fit: rewrite a template slot's text so it FITS the slot's capacity.

The offline filler (filler.build_slot_fill) maps real story content onto each slot, and the
renderer auto-shrinks the font so it never clips. But a long summary in a small box (e.g. a
"hook"/kicker) shrinks to an unreadable size. This module asks Gemini to write a CONCISE
version sized to each small slot's capacity (preserving the facts + language), so the box
reads well and big slots (headline) stay prominent.

Design for cost/safety:
  - Role-based capacity budgets approximate each slot's size (a kicker is small, a headline
    medium). Only slots whose mapped text EXCEEDS its budget are sent to Gemini.
  - ONE batched Gemini call per render; results cached on disk by content hash so re-renders
    / per-channel renders don't re-spend quota.
  - Any failure (quota, parse, disabled) falls back to the offline text + the renderer's
    pixel-exact auto-fit. The render never breaks.
  - Gated by KAIZER_TEMPLATE_AI_FIT (default on); set to 0 to disable AI and use offline only.
"""
from __future__ import annotations

import hashlib
import json
import os

from .filler import build_slot_fill

# "Nice fit" length per slot role (chars). Above this we ask Gemini to shorten. Roughly
# tracks box size: kicker/cta tiny, hook small, headline/subtitle medium, body large.
# ticker/marquee are intentionally absent — they scroll, so length is fine.
_ROLE_BUDGET = {
    "kicker": 28, "cta": 28, "watermark": 40, "hook": 70,
    "headline": 130, "title": 130, "subtitle": 170, "caption": 170,
    "body": 700,
}

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output",
                          "custom_templates", "_aifit_cache")


def _enabled() -> bool:
    return os.environ.get("KAIZER_TEMPLATE_AI_FIT", "1").strip().lower() not in ("0", "false", "no")


def _cache_key(over: dict) -> str:
    payload = json.dumps({k: [t, c] for k, (t, c) in sorted(over.items())}, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _cache_get(key: str):
    try:
        with open(os.path.join(_CACHE_DIR, key + ".json"), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _cache_put(key: str, val: dict) -> None:
    if not val:
        return  # only cache successful (non-empty) results so a transient failure can retry
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(os.path.join(_CACHE_DIR, key + ".json"), "w", encoding="utf-8") as fh:
            json.dump(val, fh, ensure_ascii=False)
    except Exception:
        pass


def _gemini_shorten(over: dict, *, db=None) -> dict:
    """over = {slot: (text, max_chars)} -> {slot: shortened_text}. {} on any failure."""
    try:
        from seo.generator import (GEMINI_MODEL, _SEO_MODEL_CHAIN, _gemini_client,  # noqa
                                   genai_types)
    except Exception:
        return {}
    sys_p = (
        "You are a multilingual news caption editor. For each field, rewrite the text so it is "
        "AT MOST max_chars characters long, preserving the key facts and keeping the SAME "
        "language and script as the input (e.g. Telugu stays Telugu). Make it punchy and "
        "complete — not cut off mid-word. Return ONLY a JSON object mapping each field name to "
        "its shortened string, nothing else."
    )
    user_p = json.dumps({k: {"text": t, "max_chars": int(c)} for k, (t, c) in over.items()},
                        ensure_ascii=False)
    models = _SEO_MODEL_CHAIN or [GEMINI_MODEL]
    for m in models:
        try:
            client = _gemini_client()
            cfg = genai_types.GenerateContentConfig(
                system_instruction=sys_p, response_mime_type="application/json",
                # 2.5-flash thinking tokens can eat a small cap and truncate JSON; 2048 is
                # plenty for the short shortened-text payload + thinking headroom.
                temperature=0.4, max_output_tokens=2048)
            resp = client.models.generate_content(model=m, contents=user_p, config=cfg)
            data = json.loads((resp.text or "").strip())
            if isinstance(data, dict):
                return {k: str(v).strip() for k, v in data.items()
                        if isinstance(v, str) and v.strip()}
        except Exception as exc:
            low = str(exc).lower()
            if any(t in low for t in ("429", "quota", "exhausted", "resourceexhausted", "404", "not found")):
                continue  # quota/model gone -> try the next model in the chain
            break        # other error -> give up, caller falls back to offline + auto-fit
    return {}


def fit_texts(contract, content, *, kind: str = "short", db=None):
    """Like build_slot_fill but with AI-shortened text for slots whose mapped content
    overflows the slot's capacity. Returns (texts: dict, sf: SlotFill). Never raises."""
    sf = build_slot_fill(contract, content, kind=kind)
    texts = dict(sf.texts)
    if not _enabled():
        return texts, sf
    over = {}
    for key, val in texts.items():
        budget = _ROLE_BUDGET.get((key or "").lower())
        if budget and len(val) > budget:
            over[key] = (val, budget)
    if not over:
        return texts, sf
    ck = _cache_key(over)
    fitted = _cache_get(ck)
    if fitted is None:
        fitted = _gemini_shorten(over, db=db)
        _cache_put(ck, fitted)
    for k, v in (fitted or {}).items():
        if k in texts and v and v.strip():
            texts[k] = v.strip()
    return texts, sf
