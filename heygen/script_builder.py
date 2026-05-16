"""Compress a YouTube transcript + topic context into an avatar script.

Why a builder (and not just feed the raw transcript to HeyGen):
  - Raw YouTube transcripts are 5-20 min of speech. HeyGen caps per
    generation around 1500 chars (~60-90 s in Telugu, faster in
    English). Sending the whole thing 413s.
  - The original speaker's filler, asides, ad reads, channel intro
    rhetoric, "subscribe + like" tags etc. don't belong in the
    Kaizer avatar's narration. They distract from the news.
  - We want the avatar to sound like a Kaizer News anchor — same
    diction, same tonality the channel's existing content uses.

Strategy
--------
Single Gemini call (``gemini-2.5-flash``, low cost) with a strict
prompt: produce ONE continuous narration in the topic's language,
≤ 700 chars, no bullets / no scene-marks. Topic title + summary feed
in as authoritative context so Gemini doesn't drift if the transcript
is garbled. Falls back to a hard-truncated transcript when Gemini is
unavailable (no key, no quota) — degraded but never crashes.
"""
from __future__ import annotations

import json
import os
from typing import Optional


_SCRIPT_MODEL = os.environ.get("KAIZER_TOPIC_MODEL", "gemini-2.5-flash")

# Character budget — Telugu reads slower than English so we leave
# headroom below HeyGen's 1500 hard cap. ~700 chars ≈ 60-90 s spoken
# in Telugu, 30-50 s in English.
_SCRIPT_MAX_CHARS = 700


_PROMPT_TEMPLATE = """You are a Telugu news scriptwriter for Kaizer News.

Rewrite the source material below into a SINGLE continuous narration
spoken by a Kaizer News anchor on camera. The narration must:

  - Be in **pure {language_name}** ({language_code}). This is the
    HARDEST constraint. The source transcript may be English, Hindi,
    or a code-mix. TRANSLATE everything into {language_name} — every
    word, every name, every number. Do NOT write English words in
    Telugu script (e.g. write "ఎన్నికలు" not "ఎలెక్షన్", write
    "ముఖ్యమంత్రి" not "చీఫ్ మినిస్టర్"). The avatar will pronounce
    whatever you write verbatim, so non-Telugu words sound jarring.
  - Stay under {max_chars} characters (this is a HARD cap — HeyGen
    rejects longer scripts).
  - Read as one flowing piece. NO bullet points. NO scene marks.
    NO "subscribe / like / hit the bell". NO sponsor reads.
  - Lead with the strongest hook in the first sentence.
  - Cover the 2-3 most important factual beats. Drop everything else.
  - Stay grounded in the AUTHORITATIVE CONTEXT below when the
    transcript is noisy / off-topic / has Whisper transcription
    errors. Trust the title + summary OVER the transcript on conflict.
  - End with a concrete, news-style sign-off (e.g. "ఇదే వార్త, మీ
    Kaizer News నుండి"). Do NOT promote subscribing.

LANGUAGE PURITY RULE (zero exceptions):
- If the source has English news terms like "BJP", "Congress",
  "election", "minister", "rally", "press conference", you MUST
  translate them: BJP -> బీజేపీ, Congress -> కాంగ్రెస్, election ->
  ఎన్నికలు, minister -> మంత్రి, rally -> ర్యాలీ, press conference ->
  పత్రికా సమావేశం.
- Proper nouns (people, places) stay in their native form
  (Pawan Kalyan -> పవన్ కల్యాణ్, Hyderabad -> హైదరాబాద్) — but
  always Telugu script.
- Numbers: spell them in Telugu words for natural delivery
  (2025 -> రెండు వేల ఇరవై ఐదు OR 2025 in digit form is OK if
   the rest of the sentence flows).

OUTPUT FORMAT (zero deviation):
Return STRICT JSON: {{"script": "<the narration>"}} — nothing else.
No prose around it, no markdown fences, no "Here is the JSON".
Every visible character in "script" must be Telugu script, Telugu
punctuation (। , . ! ?), digits, or spaces. NOTHING in Latin script.

AUTHORITATIVE CONTEXT:
- Title:   {title}
- Summary: {summary}
- Keywords: {keywords}

SOURCE TRANSCRIPT (from {transcript_source}, may contain ASR errors
or English filler — TRANSLATE + paraphrase, do not copy verbatim):
\"\"\"
{transcript}
\"\"\"
"""


_LANG_NAMES = {
    "te": "Telugu",
    "hi": "Hindi",
    "en": "English",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
    "mr": "Marathi",
}


_SCRIPT_SCHEMA = {
    "type": "object",
    "required": ["script"],
    "properties": {"script": {"type": "string"}},
}

_TRANSLATE_SCHEMA = {
    "type": "object",
    "required": ["text"],
    "properties": {"text": {"type": "string"}},
}


# ─── Telugu-purity detector ────────────────────────────────────────
# Telugu Unicode block: U+0C00..U+0C7F. We count the fraction of
# "alphabetic" chars (excluding spaces/punctuation/digits) that fall
# inside that block. A pure-Telugu script reads ~95-99%.

def _telugu_purity(text: str) -> float:
    """Return the fraction of alphabetic characters that are Telugu.
    Spaces / digits / punctuation are ignored from both numerator
    and denominator. Returns 1.0 for empty input (caller checks
    length separately)."""
    if not text:
        return 1.0
    total = 0
    telugu = 0
    for ch in text:
        if ch.isalpha():
            total += 1
            if "ఀ" <= ch <= "౿":
                telugu += 1
    if total == 0:
        return 1.0
    return telugu / total


# ─── Pre-stage: explicit Telugu translation ────────────────────────
# When the source transcript is English / Hindi / heavy code-mix,
# we run a dedicated Gemini "translator" pass BEFORE the script
# compression. Keeps the compression prompt focused and gives Gemini
# the easier subtask of translation (it's not also juggling length
# caps and narrative structure).

_TRANSLATE_PROMPT = """You are a professional Telugu translator
working for a Telugu news channel.

Translate the source text below into PURE TELUGU. Output rules:

  - Every alphabetic character MUST be in the Telugu Unicode script
    (U+0C00-U+0C7F). NO Latin script (English) characters anywhere.
  - Translate English news terms idiomatically (BJP -> బీజేపీ,
    Congress -> కాంగ్రెస్, minister -> మంత్రి, election ->
    ఎన్నికలు, rally -> ర్యాలీ, etc.).
  - Proper nouns (people, places) get phonetic Telugu transliteration
    (Pawan Kalyan -> పవన్ కల్యాణ్, Hyderabad -> హైదరాబాద్).
  - Keep the meaning faithful — paraphrase only when the source has
    transcription errors or filler ("um", "uh"). Do NOT summarize or
    compress here — that's a separate step downstream.
  - Numbers may stay as digits (2025) or be spelled in Telugu words —
    your choice based on natural flow.
  - Keep paragraph breaks if present in the source.

OUTPUT FORMAT (zero deviation):
Return STRICT JSON: {"text": "<translated text>"} — nothing else.

SOURCE:
\"\"\"
{source}
\"\"\"
"""


def _translate_to_telugu(text: str, *, api_key: str, model: str,
                        timeout_s: int = 120) -> Optional[str]:
    """Single Gemini call: translate ``text`` to pure Telugu. Returns
    the translated string, or None on failure."""
    if not text or not api_key:
        return None
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        return None
    try:
        client = genai.Client(api_key=api_key)
        cfg = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_TRANSLATE_SCHEMA,
            temperature=0.2,
            max_output_tokens=4096,
        )
        # Cap input at 15k chars to keep the prompt budget reasonable.
        resp = client.models.generate_content(
            model=model,
            contents=_TRANSLATE_PROMPT.replace("{source}", text[:15000]),
            config=cfg,
        )
        data = json.loads((resp.text or "").strip())
        translated = (data.get("text") or "").strip()
        return translated or None
    except Exception as exc:
        print(f"[heygen/script_builder] translation failed: {exc}")
        return None


def _fallback_truncate(transcript: str, max_chars: int) -> str:
    """When Gemini isn't reachable, return a hard-truncated transcript.
    Tries to land on a sentence boundary so the avatar doesn't stop
    mid-word."""
    text = (transcript or "").strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    for boundary in (". ", "! ", "? ", "। ", "; ", ", "):
        idx = cut.rfind(boundary)
        if idx > max_chars * 0.6:
            return cut[: idx + 1].strip()
    return cut.strip()


def build_script(
    *,
    topic_title: str,
    topic_summary: str,
    topic_keywords: list[str],
    transcript: str,
    transcript_source: str = "captions",
    language: str = "te",
    max_chars: int = _SCRIPT_MAX_CHARS,
) -> dict:
    """Produce the avatar-ready script.

    Returns ``{script, source: "gemini"|"fallback", model: str}``.
    The pipeline only consumes ``script``; the other fields land in
    the Clip's meta for traceability.
    """
    transcript = (transcript or "").strip()
    if not transcript:
        # Nothing to compress — return summary + title as the script.
        # Still useful: HeyGen will narrate "Topic title. Summary."
        fallback = (
            (topic_title or "").strip() + ". " + (topic_summary or "").strip()
        ).strip()
        return {
            "script": _fallback_truncate(fallback, max_chars) or topic_title or "",
            "source": "fallback",
            "model":  "",
        }

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        # Gemini not configured — degrade to truncated transcript.
        print("[heygen/script_builder] no GEMINI_API_KEY — returning truncated transcript")
        return {
            "script": _fallback_truncate(transcript, max_chars),
            "source": "fallback",
            "model":  "",
        }

    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        print("[heygen/script_builder] google-genai SDK missing — fallback")
        return {
            "script": _fallback_truncate(transcript, max_chars),
            "source": "fallback",
            "model":  "",
        }

    lang_code = (language or "te").lower().strip()[:2]
    lang_name = _LANG_NAMES.get(lang_code, "Telugu")

    # ── Pre-stage: explicit translation when source is non-Telugu ──
    # Skip for non-Telugu target languages (only kicks in for te).
    source_purity = _telugu_purity(transcript)
    translation_applied = False
    if lang_code == "te" and source_purity < 0.60:
        print(f"[heygen/script_builder] source Telugu purity = {source_purity:.0%}, "
              f"running explicit translation pass first")
        translated = _translate_to_telugu(
            transcript[:15000], api_key=api_key, model=_SCRIPT_MODEL,
        )
        if translated:
            transcript = translated
            translation_applied = True
            new_purity = _telugu_purity(transcript)
            print(f"[heygen/script_builder] post-translation purity = {new_purity:.0%} "
                  f"({len(translated)} chars)")

    prompt = _PROMPT_TEMPLATE.format(
        language_name=lang_name,
        language_code=lang_code,
        max_chars=max_chars,
        title=(topic_title or "")[:200],
        summary=(topic_summary or "")[:600],
        keywords=", ".join(topic_keywords or [])[:200],
        transcript_source=transcript_source + (" + translated" if translation_applied else ""),
        # Cap the input transcript at 12k chars to keep the Gemini
        # context budget reasonable; the longest news clip transcripts
        # we've seen come in around 8-10k.
        transcript=transcript[:12000],
    )

    def _run_compress(prompt_text: str) -> Optional[str]:
        """One Gemini call. Returns the script or None on failure."""
        try:
            client = genai.Client(api_key=api_key)
            cfg = genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_SCRIPT_SCHEMA,
                temperature=0.4,
                max_output_tokens=1024,
            )
            resp = client.models.generate_content(
                model=_SCRIPT_MODEL,
                contents=prompt_text,
                config=cfg,
            )
            data = json.loads((resp.text or "").strip())
            return ((data.get("script") or "")).strip() or None
        except Exception as exc:
            print(f"[heygen/script_builder] compress Gemini call failed: {exc}")
            return None

    try:
        script = _run_compress(prompt)
        if not script:
            raise RuntimeError("empty script in Gemini response")

        # ── Post-validation: was the OUTPUT pure Telugu? ──
        # If purity is < 80%, retry ONCE with a stricter "translate
        # ALL English glyphs you see" instruction prepended.
        if lang_code == "te":
            out_purity = _telugu_purity(script)
            if out_purity < 0.80:
                print(f"[heygen/script_builder] output Telugu purity = {out_purity:.0%}, "
                      f"retrying with stricter constraint")
                retry_prompt = (
                    "PREVIOUS ATTEMPT FAILED — your last output contained "
                    "Latin (English) script characters. Re-output the script "
                    "with ZERO English characters. Every alphabetic char "
                    "MUST be Telugu Unicode (U+0C00-U+0C7F).\n\n"
                    + prompt
                )
                retry_script = _run_compress(retry_prompt)
                if retry_script and _telugu_purity(retry_script) >= _telugu_purity(script):
                    script = retry_script
                    out_purity = _telugu_purity(script)
                    print(f"[heygen/script_builder] retry purity = {out_purity:.0%}")

        # Defensive cap — Gemini sometimes overshoots its own max_chars.
        if len(script) > max_chars:
            script = _fallback_truncate(script, max_chars)
        return {
            "script":           script,
            "source":           "gemini",
            "model":            _SCRIPT_MODEL,
            "translated":       translation_applied,
            "output_purity":    round(_telugu_purity(script), 3) if lang_code == "te" else None,
            "input_purity":     round(source_purity, 3) if lang_code == "te" else None,
        }
    except Exception as exc:
        print(f"[heygen/script_builder] Gemini compression failed: {exc} — fallback")
        return {
            "script": _fallback_truncate(transcript, max_chars),
            "source": "fallback",
            "model":  "",
        }
