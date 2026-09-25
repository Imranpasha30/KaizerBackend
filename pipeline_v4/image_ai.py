"""V4 story-image generator — Nano Banana via Vertex AI.

Sister module to ``thumbnail_ai.py``. Where the thumbnail is a single
high-CTR click-bait composition for the YouTube thumbnail, the story
images here are the B-roll / sidebar / carousel imagery shown next to
the talking head while the news plays.

Priority chain for "give me a story image":
    1. User-uploaded image in the job's _pool/ (always wins)
    2. AI-generated via Nano Banana (this module — new)
    3. Web search (V1's CSE -> DDG -> Pexels chain in image_provider)
    4. Generated news-card fallback (V1's PIL renderer)

Why this module instead of always falling back to web search:
  - Free reign over composition / aspect ratio
  - No third-party rate limits
  - Bills to your GCP credits (same Vertex path as thumbnails)
  - Iterative tweak path: rewrite the previous prompt with a small
    change instead of starting from scratch on every retry
"""
from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple


DEFAULT_IMAGE_MODEL  = os.environ.get("KAIZER_NANO_BANANA_MODEL",       "gemini-2.5-flash-image")
DEFAULT_PROMPT_MODEL = os.environ.get("KAIZER_NANO_BANANA_PROMPT_MODEL", "gemini-2.5-flash")


# Sidebar images are NOT thumbnails — no big shout text, no garish red
# CTA palette. They're cinematic B-roll the viewer sees next to the
# anchor while the story plays. Documentary / news-magazine vibe.
_PROMPT_WRITER_SYSTEM = """\
You are a news-broadcast art director writing image prompts for the
B-roll / sidebar slot that plays next to the news anchor while a story
is read. Given a news clip's headline + summary, write ONE image
prompt for a single sidebar image.

Rules:
1. Output a single prompt, no preamble, no bullet points, no JSON.
2. 16:9 cinematic news B-roll aesthetic. Documentary photography vibe.
3. NO large overlay text. NO shout headlines. The sidebar is contextual
   imagery — text comes from the news strap/ticker, not the picture.
4. Faces of REAL public figures should be replaced with generic
   silhouettes, back-of-head shots, or environmental scene-setters.
   Never name a real person.
5. For crime / scam / fraud stories: prefer evocative symbols
   (handcuffs, evidence bags, court house, vault, gold bars) over
   trying to depict the criminal or victim directly.
6. For political / policy stories: prefer parliament buildings, voting
   booths, podium silhouettes, papers being signed.
7. For economic stories: market charts, currency stacks, factory
   floors, shipping ports.
8. Sharp focus, natural lighting, slightly desaturated like a real
   news photo — NOT a movie poster, NOT a vibrant social-media post.
9. No watermarks, no channel branding, no logos.

Output: ONE prompt string under 300 words. Nothing else.
"""

_PROMPT_TWEAKER_SYSTEM = """\
You are revising a news B-roll image prompt. The user is happy with
most of the previous prompt but wants ONE change. Rewrite the prompt
so that change is applied, keeping every other element intact.

Rules:
1. Output a SINGLE revised prompt, no preamble, no JSON.
2. Keep the cinematic news B-roll aesthetic and 16:9 format.
3. Apply the user's tweak verbatim wherever it overrides the original
   (e.g. "darker mood" -> dial down brightness; "courtroom instead of
   prison" -> swap the scene).
4. Never name real public figures; keep silhouettes/symbolic imagery
   if the previous prompt used them.
5. No watermarks, no channel branding.

Output: ONE prompt string under 300 words.
"""


# Multi-image planner: break ONE story into several DISTINCT visual beats, one
# image per beat — for a carousel/slideshow whose length is driven by the story,
# not a fixed template count. Each beat is a different moment/angle so the
# slideshow tells the story instead of showing the same picture N times.
_MULTI_PROMPT_WRITER_SYSTEM = """\
You are a news-broadcast art director planning a SEQUENCE of B-roll / sidebar
images for ONE news story. The images play one after another (a slideshow)
beside the anchor while the story is read, so each image must show a DIFFERENT
beat of the story — establishing scene, the key subject/symbol, the
consequence/aftermath, the reaction, the wider context — never the same shot
twice.

Decide how many images the story naturally needs (more beats = more images),
between {min_n} and {max_n}. If the user gives an explicit count, output EXACTLY
that many.

Each prompt follows these rules:
1. 16:9 cinematic news B-roll aesthetic, documentary photography vibe.
2. NO large overlay text, NO shout headlines, NO watermarks, NO channel logos.
3. Replace real public figures with generic silhouettes / back-of-head /
   environmental scene-setters. Never name a real person.
4. Crime/scam: evocative symbols (handcuffs, evidence, courthouse, vault).
   Politics: parliament, podium silhouettes, signed papers. Economy: charts,
   currency, factory floors, ports. Disaster/weather: flooded streets, rescue
   crews, storm skies — match the actual story.
5. Sharp focus, natural lighting, slightly desaturated like a real news photo.

Output ONLY a JSON array of objects, one per beat:
[{"prompt": "full image prompt for beat 1", "label": "short subject line"},
 {"prompt": "full image prompt for beat 2", "label": "short subject line"}]
`label` is a ≤8-word subject line saying WHO/WHAT the image shows (e.g.
"parliament building at dusk", "flooded street rescue"). When the story names
a person/place, put the NAME in the label (native script + English is best) —
the label is used to sync the image to the exact spoken words, it is NOT
rendered into the picture.
No preamble, no markdown fences, nothing but the JSON array.
"""


def _gemini_client():
    """Reuse the Vertex client builder from thumbnail_ai so both image
    generators share the same auth + region setup. If the user later
    flips to AI Studio mode, both surfaces switch together."""
    from pipeline_v4.thumbnail_ai import _gemini_client as build
    return build()


_CAPTION_SYSTEM = """\
You label news images for a broadcast archive. Given ONE image, output a
single short subject line (max 8 words) saying WHO or WHAT it shows —
e.g. "politician addressing parliament", "flooded street with rescue boats",
"stock market chart falling". If a well-known public figure or landmark is
clearly identifiable, name them. No preamble, no punctuation-heavy prose,
no quotes — just the label text.
"""


def caption_image(path: str, *, mime: str = "image/jpeg") -> str:
    """One-line subject label for an image ("name-tag contract").

    Used to auto-label user uploads that arrive without a description so
    the image↔speech timing AI can match them to spoken words. Returns ""
    on any failure — callers treat the label as optional."""
    try:
        data = Path(path).read_bytes()
    except Exception:
        return ""
    if not data:
        return ""
    try:
        from google.genai import types as genai_types
        client = _gemini_client()
        resp = client.models.generate_content(
            model=DEFAULT_PROMPT_MODEL,
            contents=[
                genai_types.Part.from_bytes(data=data, mime_type=mime or "image/jpeg"),
                "Label this image.",
            ],
            config=genai_types.GenerateContentConfig(
                system_instruction=_CAPTION_SYSTEM,
                temperature=0.2,
            ),
        )
        text = (resp.text or "").strip().strip('"').strip()
        # One line only; defensive length cap
        return text.splitlines()[0][:120] if text else ""
    except Exception as exc:
        print(f"[v4/image-ai] caption failed for {path}: {exc}", flush=True)
        return ""


def write_image_prompt(
    *,
    title_native: str = "",
    title_english: str = "",
    summary: str = "",
    language: str = "te",
    note: str = "",
) -> str:
    """Ask Gemini for a sidebar-image brief from the story content.
    Falls back to a deterministic stock prompt on failure. ``note`` is an
    optional per-slot operator direction (e.g. "night-time, blue tones")
    merged into the brief so the image honours it."""
    facts = []
    if title_native:  facts.append(f"native-script headline: {title_native}")
    if title_english: facts.append(f"English headline: {title_english}")
    if summary:       facts.append(f"summary: {summary[:600]}")
    facts.append(f"language: {language}")
    if note:
        facts.append(f"operator note — MUST be reflected in the image: {note[:300]}")
    user_msg = (
        "Write the sidebar-image prompt for this news story.\n\n"
        + "\n".join(facts)
    )
    try:
        from google.genai import types as genai_types
        # Space request starts: a 16-story job fires 16 prompt-writer
        # calls at once and Vertex's per-minute quota 429s most into the
        # generic fallback prompt (weaker images). Paced, they succeed.
        from pipeline_v4.api_pace import pace
        pace()
        client = _gemini_client()
        resp = client.models.generate_content(
            model=DEFAULT_PROMPT_MODEL,
            contents=user_msg,
            config=genai_types.GenerateContentConfig(
                system_instruction=_PROMPT_WRITER_SYSTEM,
                temperature=0.8,
            ),
        )
        text = (resp.text or "").strip()
        if text:
            return text
    except Exception as exc:
        print(f"[v4/image-ai] prompt writer failed: {exc}", flush=True)

    headline = (title_native or title_english or "BREAKING").strip()
    return (
        f"Cinematic 16:9 news B-roll photograph about: {headline}. "
        f"Documentary style, natural lighting, slightly desaturated colour. "
        f"Generic silhouettes only (no recognisable real people). "
        f"Sharp focus, evocative composition. No overlay text, no logo."
        + (f" Operator direction: {note[:200]}." if note else "")
    )


def tweak_image_prompt(*, previous_prompt: str, tweak: str) -> str:
    """One Gemini text call that rewrites the last prompt with a small
    change. Cheaper than writing a new prompt from scratch."""
    if not previous_prompt:
        return tweak or ""
    if not tweak:
        return previous_prompt
    try:
        from google.genai import types as genai_types
        client = _gemini_client()
        user_msg = (
            f"Previous prompt:\n{previous_prompt}\n\n"
            f"User tweak: {tweak}\n\n"
            f"Output the revised prompt."
        )
        resp = client.models.generate_content(
            model=DEFAULT_PROMPT_MODEL,
            contents=user_msg,
            config=genai_types.GenerateContentConfig(
                system_instruction=_PROMPT_TWEAKER_SYSTEM,
                temperature=0.6,
            ),
        )
        text = (resp.text or "").strip()
        if text:
            return text
    except Exception as exc:
        print(f"[v4/image-ai] tweak failed: {exc}", flush=True)
    return f"{previous_prompt}\n\nAdditional direction: {tweak}"


def derive_label(prompt: str) -> str:
    """Deterministic ≤8-word subject line from an image prompt — the
    fallback when the planner didn't supply a ``label``. Strips the
    prompt-writer's stock preamble and keeps the first clause."""
    if not prompt:
        return ""
    t = prompt.strip().splitlines()[0]
    # Stock preambles from write_image_prompt / the deterministic fallback:
    # "Cinematic 16:9 news B-roll photograph about: X." / "... photo of X, ..."
    t = re.sub(
        r"(?i)^(a|an|the)?\s*(cinematic|documentary|photorealistic|wide|close-up)?"
        r"[\w\s:,'-]{0,40}?(photograph|photo|image|shot|scene|b-roll)\s+"
        r"(of|about|showing)[:\s]+",
        "", t, count=1,
    )
    t = re.split(r"[.;:]", t)[0].strip().strip(",")
    words = t.split()
    return " ".join(words[:8])[:120]


def _parse_beat_list(text: str) -> List[dict]:
    """Pull a JSON array of image beats out of a model response, tolerating
    code fences / stray prose / plain-string items / `[{"prompt": …}]` shapes.
    Returns ``[{"prompt": str, "label": str}, …]`` ([] when nothing usable);
    a missing label is derived from the prompt."""
    if not text:
        return []
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t).strip()
    m = re.search(r"\[.*\]", t, re.DOTALL)
    blob = m.group(0) if m else t
    try:
        data = json.loads(blob)
    except Exception:
        return []
    out: List[dict] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str) and item.strip():
                p = item.strip()
                out.append({"prompt": p, "label": derive_label(p)})
            elif isinstance(item, dict):
                p = item.get("prompt") or item.get("text") or ""
                if isinstance(p, str) and p.strip():
                    p = p.strip()
                    lab = item.get("label") or item.get("subject") or ""
                    lab = lab.strip() if isinstance(lab, str) else ""
                    out.append({"prompt": p, "label": (lab or derive_label(p))[:120]})
    return out


def _parse_prompt_list(text: str) -> List[str]:
    """Back-compat shim: prompt strings only (see ``_parse_beat_list``)."""
    return [b["prompt"] for b in _parse_beat_list(text)]


def plan_story_beats(
    *,
    title_native: str = "",
    title_english: str = "",
    summary: str = "",
    language: str = "te",
    count: Optional[int] = None,
    max_images: int = 8,
    note: str = "",
) -> List[dict]:
    """Break ONE story into a SEQUENCE of distinct B-roll image beats
    ``[{"prompt", "label"}, …]`` — one per visual beat, so a carousel's length is
    driven by the story, not a fixed template count. ``label`` is the beat's
    ≤8-word subject line (the "name-tag contract": the image↔speech sync
    matches images to spoken words by this label). ``count=None`` → the model
    decides how many beats (clamped 1..max_images); explicit ``count`` →
    exactly that many. Always returns ≥1 beat (falls back to a single prompt
    + angle variations)."""
    max_images = max(1, min(int(max_images or 8), 12))
    want = max(1, min(int(count), max_images)) if count is not None else None
    min_n = want or 2
    max_n = want or max_images

    facts = []
    if title_native:  facts.append(f"native-script headline: {title_native}")
    if title_english: facts.append(f"English headline: {title_english}")
    if summary:       facts.append(f"summary: {summary[:800]}")
    facts.append(f"language: {language}")
    if note:
        facts.append(f"operator note — MUST be reflected in EVERY image: {note[:300]}")
    if want:
        facts.append(f"REQUIRED image count: exactly {want}")
    user_msg = "Plan the sidebar-image sequence for this news story.\n\n" + "\n".join(facts)
    system = _MULTI_PROMPT_WRITER_SYSTEM.format(min_n=min_n, max_n=max_n)
    try:
        from google.genai import types as genai_types
        client = _gemini_client()
        resp = client.models.generate_content(
            model=DEFAULT_PROMPT_MODEL,
            contents=user_msg,
            config=genai_types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.85,
                max_output_tokens=4096,   # flash spends thinking tokens — keep headroom for the JSON
            ),
        )
        beats = _parse_beat_list(resp.text or "")
        if beats:
            return beats[:(want or max_images)]
    except Exception as exc:
        print(f"[v4/image-ai] multi-prompt planner failed: {exc}", flush=True)

    # Fallback: one solid prompt, padded with distinct angles to the wanted count.
    base = write_image_prompt(title_native=title_native, title_english=title_english,
                              summary=summary, language=language, note=note)
    base_label = derive_label(base) or (title_english or title_native or "")[:120]
    n = want or 1
    if n <= 1:
        return [{"prompt": base, "label": base_label}]
    angles = ["establishing wide shot", "close detail of the key subject",
              "the aftermath / consequence", "the wider location / context",
              "a symbolic object from the story", "reaction of bystanders",
              "an overhead / aerial perspective", "a quiet human moment"]
    out = [{"prompt": base, "label": base_label}]
    for i in range(1, n):
        angle = angles[(i - 1) % len(angles)]
        out.append({"prompt": f"{base} Variation: {angle}.",
                    "label": f"{base_label} — {angle}"[:120]})
    return out


def plan_story_images(
    *,
    title_native: str = "",
    title_english: str = "",
    summary: str = "",
    language: str = "te",
    count: Optional[int] = None,
    max_images: int = 8,
    note: str = "",
) -> List[str]:
    """Back-compat shim over ``plan_story_beats`` — prompt strings only."""
    return [b["prompt"] for b in plan_story_beats(
        title_native=title_native, title_english=title_english, summary=summary,
        language=language, count=count, max_images=max_images, note=note,
    )]


def generate_image(
    *,
    prompt: str,
    out_path: str,
    width: int = 1280,
    height: int = 720,
) -> Optional[str]:
    """Call Nano Banana with ``prompt`` and save the first image to
    ``out_path``. Stamps ``generate_image.last_error`` on failure so
    the route can surface the verbatim error to the UI."""
    generate_image.last_error = ""
    if not prompt:
        return None
    try:
        from google.genai import types as genai_types
    except Exception as exc:
        generate_image.last_error = f"SDK missing: {exc}"
        print(f"[v4/image-ai] google-genai missing: {exc}", flush=True)
        return None
    try:
        client = _gemini_client()
    except Exception as exc:
        generate_image.last_error = f"auth: {exc}"
        print(f"[v4/image-ai] auth: {exc}", flush=True)
        return None
    try:
        resp = client.models.generate_content(
            model=DEFAULT_IMAGE_MODEL,
            contents=[prompt],
            config=genai_types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                temperature=0.9,
            ),
        )
    except Exception as exc:
        msg = str(exc)
        generate_image.last_error = msg
        print(f"[v4/image-ai] image generate failed: {exc}", flush=True)
        return None

    try:
        for cand in (resp.candidates or []):
            content = getattr(cand, "content", None)
            for part in (getattr(content, "parts", None) or []):
                inline = getattr(part, "inline_data", None)
                if not inline:
                    continue
                raw = getattr(inline, "data", None)
                if raw is None:
                    continue
                if isinstance(raw, str):
                    raw = base64.b64decode(raw)
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                with open(out_path, "wb") as fh:
                    fh.write(raw)
                _ensure_aspect(out_path, width, height)
                return out_path
    except Exception as exc:
        generate_image.last_error = f"parse: {exc}"
        print(f"[v4/image-ai] response parse failed: {exc}", flush=True)
    return None


generate_image.last_error = ""


# gpt-image-1 renders from the SAME refined art-director prompts as the
# Gemini path — documentary B-roll, NOT the garish BIG-TV thumbnail
# template in express/ai_image (that style is for thumbnails). This
# short suffix restates the technical constraints image models weigh
# most when they close out a generation.
_OPENAI_BROLL_SUFFIX = (
    " Photorealistic 16:9 documentary news photograph, sharp focus, "
    "natural lighting, slightly desaturated. No text, no watermarks, no logos."
)


def _generate_image_openai(
    *,
    prompt: str,
    out_path: str,
    width: int = 1280,
    height: int = 720,
) -> Optional[str]:
    """OpenAI gpt-image-1 renderer behind the same contract as
    ``generate_image`` (saves to out_path, stamps
    ``generate_image.last_error`` on failure). gpt-image-1 only takes
    fixed canvases — pick by orientation, then cover-crop to the exact
    target like the Gemini path does."""
    generate_image.last_error = ""
    if not prompt:
        return None
    try:
        from express.ai_image import _generate as _openai_raw
    except Exception as exc:
        generate_image.last_error = f"openai wrapper missing: {exc}"
        print(f"[v4/image-ai] openai wrapper missing: {exc}", flush=True)
        return None
    size = "1536x1024" if width >= height else "1024x1536"
    quality = (os.environ.get("KAIZER_OPENAI_IMAGE_QUALITY") or "high").strip().lower()
    if quality not in {"low", "medium", "high", "auto"}:
        quality = "high"
    try:
        buf = _openai_raw(
            api_key="",  # falls back to OPENAI_API_KEY env
            prompt=(prompt + _OPENAI_BROLL_SUFFIX)[:4000],
            size=size,
            quality=quality,
            timeout_s=120,
        )
    except Exception as exc:
        generate_image.last_error = str(exc)
        print(f"[v4/image-ai] openai generate failed: {exc}", flush=True)
        return None
    try:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "wb") as fh:
            fh.write(buf)
        _ensure_aspect(out_path, width, height)
        return out_path
    except Exception as exc:
        generate_image.last_error = f"save: {exc}"
        print(f"[v4/image-ai] openai save failed: {exc}", flush=True)
        return None


def render_image(
    *,
    prompt: str,
    out_path: str,
    width: int = 1280,
    height: int = 720,
    engine: str = "gemini",
) -> Optional[str]:
    """Engine-routed renderer: the SAME refined prompt, the user's chosen
    image engine. ``engine`` ∈ {"gemini", "openai"} — anything else falls
    back to Gemini so a typo never breaks a render."""
    eng = (engine or "gemini").strip().lower()
    if eng == "openai":
        return _generate_image_openai(prompt=prompt, out_path=out_path,
                                      width=width, height=height)
    return generate_image(prompt=prompt, out_path=out_path,
                          width=width, height=height)


def _ensure_aspect(path: str, width: int, height: int) -> None:
    """Cover-crop the saved JPG to exactly width x height."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            img = img.convert("RGB")
            src_ratio = img.width / max(1, img.height)
            tgt_ratio = width / height
            if src_ratio > tgt_ratio:
                new_h = height
                new_w = int(img.width * (height / img.height))
                img = img.resize((new_w, new_h), Image.LANCZOS)
                x = (new_w - width) // 2
                img = img.crop((x, 0, x + width, height))
            else:
                new_w = width
                new_h = int(img.height * (width / img.width))
                img = img.resize((new_w, new_h), Image.LANCZOS)
                y = (new_h - height) // 2
                img = img.crop((0, y, width, y + height))
            img.save(path, "JPEG", quality=92)
    except Exception as exc:
        print(f"[v4/image-ai] aspect fix failed: {exc}", flush=True)


def make_image_for_story(
    *,
    title_native: str = "",
    title_english: str = "",
    summary: str = "",
    language: str = "te",
    out_path: str,
    previous_prompt: str = "",
    tweak: str = "",
    width: int = 1280,
    height: int = 720,
    note: str = "",
    engine: str = "gemini",
) -> Tuple[Optional[str], str]:
    """Two-step: derive prompt (tweak path when previous_prompt+tweak
    supplied, otherwise fresh from SEO) -> render image -> save.
    Returns ``(saved_path or None, final_prompt)`` so the caller can
    persist the prompt for a future tweak iteration. ``note`` = optional
    per-slot operator direction (the tweak path ignores it — that path
    already carries an explicit user direction). ``engine`` picks the
    RENDERER (gemini Nano Banana / openai gpt-image-1); the art-director
    prompt-writing step is shared by both."""
    if previous_prompt and tweak:
        prompt = tweak_image_prompt(previous_prompt=previous_prompt, tweak=tweak)
    else:
        prompt = write_image_prompt(
            title_native=title_native,
            title_english=title_english,
            summary=summary,
            language=language,
            note=note,
        )
    print(f"[v4/image-ai] engine={engine} prompt: {prompt[:160]}…", flush=True)
    saved = render_image(prompt=prompt, out_path=out_path, width=width, height=height,
                         engine=engine)
    return saved, prompt


def make_images_for_story(
    *,
    title_native: str = "",
    title_english: str = "",
    summary: str = "",
    language: str = "te",
    count: Optional[int] = None,
    out_dir: str,
    base: str = "ai",
    width: int = 1280,
    height: int = 720,
    max_images: int = 8,
    note: str = "",
    engine: str = "gemini",
) -> List[Tuple[str, str, str]]:
    """Plan a story-driven image SEQUENCE and render each beat to
    ``out_dir/<base>_<i>.jpg``. Returns ``[(saved_path, prompt, label), …]``
    for every image that rendered (failed beats are skipped so the caller
    still gets the successes); ``label`` is the beat's subject line for the
    name-tag contract. ``count`` forces an exact number; else the planner
    picks 1..max_images from the story. Use this for carousels / multi-image
    slots; for a single sidebar image use ``make_image_for_story``."""
    beats = plan_story_beats(
        title_native=title_native, title_english=title_english,
        summary=summary, language=language, count=count, max_images=max_images,
        note=note,
    )
    os.makedirs(out_dir, exist_ok=True)
    out: List[Tuple[str, str, str]] = []
    for i, beat in enumerate(beats):
        prompt = beat["prompt"]
        path = os.path.join(out_dir, f"{base}_{i + 1}.jpg")
        saved = render_image(prompt=prompt, out_path=path, width=width, height=height,
                             engine=engine)
        if saved and os.path.isfile(path):
            out.append((path, prompt, beat.get("label") or derive_label(prompt)))
    return out
