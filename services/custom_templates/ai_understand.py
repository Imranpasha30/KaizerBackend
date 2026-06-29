"""AI template understanding — the "pure intelligent" layer.

The offline `infer.py` bot understands templates via expert heuristics (id/class/tags/
structure). This module adds a REAL AI pass: it renders the template, collects every
candidate element's geometry + text + style cues, and asks an LLM to reason about the
layout — "this large empty box is the video, this biggest text is the headline, this wide
bottom bar is the ticker, this small corner mark is the logo" — then marks the template
from that understanding. It handles templates with ZERO naming hints (machine exports)
that defeat the rules.

AI-primary, rules-fallback: when the AI is off / out of quota / fails, callers fall back
to the offline `infer.normalize_and_discover`. Result cached per template (one LLM call
per upload). Gated by KAIZER_TEMPLATE_AI_UNDERSTAND (default on; 0 = offline rules only).
"""
from __future__ import annotations

import hashlib
import json
import os

from lxml import html as lxml_html

from . import infer as _infer
from . import renderer as _renderer
from .bundle import Bundle
from .contract import DEFAULT_CANVAS, SLOT_ATTR, _classify, discover, parse_safe

# Tags that can be a fillable region. We tag each with data-ai-idx, render to get geometry,
# and let the LLM pick. (Page chrome is filtered by size/visibility downstream.)
_CANDIDATE_TAGS = {"div", "video", "img", "section", "figure", "p", "span", "a", "marquee",
                   "h1", "h2", "h3", "h4", "h5", "h6"}
_MAX_TAG = 400          # hard cap on elements tagged (pathological templates)
_MAX_SEND = 60          # most candidates sent to the LLM
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output",
                          "custom_templates", "_aiunderstand_cache")

# role -> the data-kaizer value to stamp
_ROLE_MARKER = {
    "video": "video", "headline": "headline", "subtitle": "subtitle", "ticker": "ticker",
    "kicker": "kicker", "image": "image", "image2": "image:2", "image3": "image:3",
    "logo": "logo", "watermark": "watermark", "cta": "cta", "hook": "hook",
}


def _enabled() -> bool:
    return os.environ.get("KAIZER_TEMPLATE_AI_UNDERSTAND", "1").strip().lower() not in ("0", "false", "no")


def _cache_paths(html: str):
    h = hashlib.sha1((html or "").encode("utf-8")).hexdigest()
    return os.path.join(_CACHE_DIR, h + ".html")


def _llm_assign(cands: list, cw: int, ch: int, *, taken=None, db=None) -> dict:
    """Ask the LLM to map each role to a candidate index. {} on any failure.
    ``taken`` = roles the author already placed explicitly (the LLM must not output them)."""
    try:
        from seo.generator import (GEMINI_MODEL, _SEO_MODEL_CHAIN, _gemini_client,  # noqa
                                   genai_types)
    except Exception:
        return {}
    sys_p = (
        "You are a senior motion-graphics designer mapping a NEWS VIDEO template's regions. "
        "You are given the canvas size and a list of candidate elements (index, tag, id, "
        "class, own-text, x, y, w, h, font px, has-background-image, is-empty). Decide which "
        "element best plays each role, by REASONING about the layout like a human:\n"
        "- video: the MAIN footage area — a large box, usually empty or a <video>, often the "
        "biggest empty region.\n"
        "- headline: the largest / most prominent TEXT (pick the leaf element that holds the "
        "text, not a wrapper).\n"
        "- subtitle: secondary text near the headline.\n"
        "- kicker: a small short label/eyebrow (e.g. a chip) near the headline.\n"
        "- ticker: a WIDE, short element near the BOTTOM (a strap/bar).\n"
        "- image / image2 / image3: photo regions — <img> or a box with a background image.\n"
        "- logo: a small ~square mark, often in a corner.\n"
        "- watermark: a small faint text mark.\n"
        "Use null when no element fits a role. Never use one index for two roles. "
        "Any role listed in 'already_assigned' is already placed by the author — return null "
        "for those and only assign the REMAINING roles. "
        "Return ONLY a JSON object: {\"video\":idx|null,\"headline\":..,\"subtitle\":..,"
        "\"kicker\":..,\"ticker\":..,\"image\":..,\"image2\":..,\"image3\":..,\"logo\":..,"
        "\"watermark\":..}."
    )
    user_p = json.dumps({"canvas": [cw, ch], "elements": cands,
                         "already_assigned": sorted(taken or [])}, ensure_ascii=False)
    for m in (_SEO_MODEL_CHAIN or [GEMINI_MODEL]):
        try:
            client = _gemini_client()
            cfg = genai_types.GenerateContentConfig(
                system_instruction=sys_p, response_mime_type="application/json",
                # gemini-2.5 spends "thinking" tokens before the answer; a small cap gets
                # eaten by thinking and truncates the JSON (finish=MAX_TOKENS). 4096 leaves
                # room for both (the JSON output itself is tiny).
                temperature=0.2, max_output_tokens=4096)
            resp = client.models.generate_content(model=m, contents=user_p, config=cfg)
            data = json.loads((resp.text or "").strip())
            if isinstance(data, dict):
                out = {}
                for role, v in data.items():
                    if role not in _ROLE_MARKER or isinstance(v, bool):
                        continue
                    # the model may return the index as an int OR a digit-string ("3")
                    if isinstance(v, int):
                        out[role] = v
                    elif isinstance(v, str) and v.strip().lstrip("-").isdigit():
                        out[role] = int(v)
                return out
        except Exception as exc:
            low = str(exc).lower()
            if any(t in low for t in ("429", "quota", "exhausted", "resourceexhausted", "404", "not found")):
                continue
            break
    return {}


def understand_and_mark(bundle, *, db=None):
    """Render + LLM-understand the template in ``bundle`` and write AI-chosen data-kaizer
    markers onto its entry HTML. Returns the discovered TemplateContract on success, or
    None (caller keeps the offline result). Never raises. SAFE TO CALL IN A THREADPOOL
    (uses sync Playwright)."""
    if not _enabled():
        return None
    # Work on the raw sanitized source (author markers intact, no offline markers) so the
    # AI is primary; fall back to the live entry if the sidecar isn't there.
    src = os.path.join(bundle.root_dir, ".kaizer_src.html")
    try:
        path = src if os.path.isfile(src) else bundle.entry_path
        with open(path, encoding="utf-8", errors="replace") as fh:
            html = fh.read()
    except Exception:
        return None

    # cache hit -> reuse the marked HTML
    cache_path = _cache_paths(html)
    try:
        if os.path.isfile(cache_path):
            with open(cache_path, encoding="utf-8") as fh:
                marked = fh.read()
            with open(bundle.entry_path, "w", encoding="utf-8") as fh:
                fh.write(marked)
            return discover(marked)
    except Exception:
        pass

    doc, _w = parse_safe(html)
    if doc is None:
        return None
    cw, ch = (_infer.infer_canvas(doc, html) or DEFAULT_CANVAS)

    # AUTHOR markers win: if the author explicitly marked a slot, keep it and don't let the
    # AI reassign that role (operator rule: "template says where -> there"). The AI fills
    # only what the author left unmarked.
    author_roles = set()
    for el in doc.xpath(f"//*[@{SLOT_ATTR}]"):
        cls = _classify(el.get(SLOT_ATTR, ""))
        if cls:
            author_roles.add(cls[1] or cls[0])

    # tag candidate elements with data-ai-idx (document order)
    i = 0
    for el in doc.iter():
        tag = getattr(el, "tag", None)
        if isinstance(tag, str) and tag in _CANDIDATE_TAGS:
            if i >= _MAX_TAG:
                break
            el.set("data-ai-idx", str(i))
            i += 1
    if i == 0:
        return None

    probe_rel = "_ai_probe.html"
    probe_path = os.path.join(bundle.root_dir, probe_rel)
    try:
        with open(probe_path, "w", encoding="utf-8") as fh:
            fh.write(lxml_html.tostring(doc, encoding="unicode"))
        cands = _renderer.probe_candidates(
            Bundle(root_dir=bundle.root_dir, entry_rel=probe_rel, files=[]), (cw, ch))
    except Exception:
        cands = []
    finally:
        try:
            os.remove(probe_path)
        except Exception:
            pass
    if not cands:
        return None

    # keep meaningful candidates (visible, non-trivial), cap the count sent
    vis = [c for c in cands if c.get("w", 0) >= 24 and c.get("h", 0) >= 14]
    vis.sort(key=lambda c: c.get("w", 0) * c.get("h", 0), reverse=True)
    send = vis[:_MAX_SEND]
    if not send:
        return None

    mapping = _llm_assign(send, cw, ch, taken=author_roles, db=db)
    if not mapping:
        return None

    # apply markers by data-ai-idx, one element per role, no idx reused
    used = set()
    applied = 0
    for role, idx in mapping.items():
        if idx is None or idx in used:
            continue
        if role in author_roles:                 # author already placed this role
            continue
        marker = _ROLE_MARKER.get(role)
        if not marker:
            continue
        els = doc.xpath(f"//*[@data-ai-idx='{int(idx)}']")
        if not els or els[0].get(SLOT_ATTR):     # element already carries an author marker
            continue
        els[0].set(SLOT_ATTR, marker)
        used.add(idx)
        applied += 1
    # strip the temp probe attrs
    for el in doc.xpath("//*[@data-ai-idx]"):
        try:
            del el.attrib["data-ai-idx"]
        except Exception:
            pass
    if applied == 0:
        return None

    # Drop any media slot the AI placed on a container that wraps a same-kind slot (phantom
    # slot guard, shared with the offline inferer).
    try:
        _infer.dedupe_nested_media_slots(doc)
    except Exception:
        pass

    try:
        marked = lxml_html.tostring(doc, encoding="unicode")
    except Exception:
        return None
    try:
        with open(bundle.entry_path, "w", encoding="utf-8") as fh:
            fh.write(marked)
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            fh.write(marked)
    except Exception:
        pass
    return discover(marked, canvas=(cw, ch))
