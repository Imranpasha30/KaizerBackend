"""TV-style profanity bleep — mute the bad word, play the beep.

What a broadcast channel does per video, automated: after Stage 1 we have
EVERY word with its exact [start, end] in the trimmed timeline
(TrimmedStory.words — the same substrate the image↔speech sync rides).
This module (1) detects words to censor against per-language wordlists
(+ optional per-job extra words), (2) rewrites the trimmed master's AUDIO
in place — the word window is muted and a 1 kHz tone plays instead —
video stream copied, and (3) writes ``bleep_report.json`` next to the
canvas so the editor can show exactly what was bleeped and why.

Design rules:
  * Exact normalized-token matching ONLY (casefold + punctuation strip).
    No substring matching — "assessment" must never trigger "ass".
    Inflections are enumerated in the list instead.
  * Fail-soft: any error leaves the master untouched and the render
    continues unbleeped (a paid job never fails on censoring).
  * Cache-safe: render_bulletin hashes the spans overlapping each story
    (conditional ingredient, like gaps/straps) so changing the bleep set
    re-renders exactly the affected stories.

Env (DEV .env):
  KAIZER_V4_BLEEP          1|0  master switch (default 1 — the channel bleeps)
  KAIZER_BLEEP_EXTRA_WORDS comma-separated additional words (any script)
  KAIZER_BLEEP_FREQ        beep frequency Hz (default 1000)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import unicodedata
from pathlib import Path
from typing import Optional

BLEEP_REPORT_NAME = "bleep_report.json"

# ── Wordlists — unambiguous profanity/slurs only ─────────────────────
# Conservative by design: every entry is offensive in isolation, so exact
# token matching cannot false-positive on normal news/podcast speech.
# Extend per-job/per-tenant via KAIZER_BLEEP_EXTRA_WORDS.

_EN = {
    "fuck", "fucking", "fucked", "fucker", "motherfucker", "motherfucking",
    "shit", "bullshit", "shitty",
    "bitch", "bitches",
    "asshole", "assholes",
    "bastard", "bastards",
    "cunt", "dick", "dickhead", "pussy", "slut", "whore",
}

# Hindi — Devanagari + common Roman transliterations (incl. frequent
# inflections; exact-match only, so variants are enumerated).
_HI = {
    "मादरचोद", "भोसड़ी", "भोसड़ीके", "चूतिया", "चुतिया", "गांड", "गाण्ड",
    "रंडी", "हरामी", "हरामज़ादा", "कमीना", "भड़वा", "लौड़ा", "लौड़े",
    "madarchod", "maderchod", "behenchod", "bhenchod", "bhosdike",
    "bhosdi", "chutiya", "chutiye", "chutia", "gaand", "gand",
    "randi", "haramzada", "haraami", "bhadwa", "lauda", "laude", "lode",
}

# Telugu — script + common transliterations.
_TE = {
    "లంజ", "లంజా", "పూకు", "మొడ్డ", "దెంగ", "దెంగు", "దెంగెయ్",
    "ఎర్రిపూకు", "సచ్చినోడ",
    "lanja", "lanjaa", "pooku", "puku", "modda", "denga", "dengu",
    "dengey", "erripooku", "erripuka",
}

_BASE_WORDS: frozenset[str] = frozenset(_EN | _HI | _TE)

# Strip everything that is not a letter/digit in ANY script — keeps
# Telugu/Devanagari intact while dropping punctuation like "…", "?", "'".
_PUNCT_RE = re.compile(r"[^\wऀ-ॿఀ-౿]+", re.UNICODE)


def normalize_token(tok: str) -> str:
    t = unicodedata.normalize("NFC", (tok or "").strip().casefold())
    return _PUNCT_RE.sub("", t)


def active_wordlist(extra: tuple = ()) -> frozenset[str]:
    words = set(_BASE_WORDS)
    env_extra = (os.environ.get("KAIZER_BLEEP_EXTRA_WORDS") or "").strip()
    for src in (env_extra.split(",") if env_extra else []):
        n = normalize_token(src)
        if n:
            words.add(n)
    for src in (extra or ()):
        n = normalize_token(str(src))
        if n:
            words.add(n)
    return frozenset(words)


def _enabled() -> bool:
    return (os.environ.get("KAIZER_V4_BLEEP", "1") or "1").strip().lower() \
        not in ("0", "false", "no", "off")


# ── Detection ────────────────────────────────────────────────────────

def detect_bleep_spans(
    words: list[dict],
    *,
    extra_words: tuple = (),
    pad: float = 0.04,
    merge_gap: float = 0.15,
) -> list[dict]:
    """Scan an absolute-timeline word array ``[{"w","s","e"}, …]`` and
    return merged censor spans ``[{"start","end","words":[…]}, …]``.
    ``pad`` widens each hit slightly (ASR boundaries clip consonants);
    hits closer than ``merge_gap`` merge into one beep."""
    wl = active_wordlist(extra_words)
    hits: list[tuple[float, float, str]] = []
    for w in (words or []):
        tok = normalize_token(str(w.get("w", "")))
        if not tok or tok not in wl:
            continue
        try:
            s = float(w.get("s", 0.0))
            e = float(w.get("e", 0.0))
        except (TypeError, ValueError):
            continue
        if e <= s:
            continue
        hits.append((max(0.0, s - pad), e + pad, str(w.get("w", "")).strip()))
    if not hits:
        return []
    hits.sort()
    spans: list[dict] = []
    for s, e, tok in hits:
        if spans and s <= spans[-1]["end"] + merge_gap:
            spans[-1]["end"] = max(spans[-1]["end"], e)
            spans[-1]["words"].append(tok)
        else:
            spans.append({"start": round(s, 3), "end": round(e, 3), "words": [tok]})
    for sp in spans:
        sp["start"] = round(sp["start"], 3)
        sp["end"] = round(sp["end"], 3)
    return spans


# ── Application (in-place, audio-only) ───────────────────────────────

def apply_bleep(video_path: str, spans: list[dict], *, timeout: int = 900) -> bool:
    """Mute each span and play a beep instead — audio re-encoded, video
    stream copied. In place. Returns True when the file was rewritten;
    False (never raises) on any failure or empty spans."""
    try:
        if not spans:
            return False
        p = Path(video_path)
        if not p.is_file():
            return False
        freq = int(float(os.environ.get("KAIZER_BLEEP_FREQ", "1000") or 1000))
        enable = "+".join(
            f"between(t,{float(sp['start']):.3f},{float(sp['end']):.3f})"
            for sp in spans
        )
        # Original audio muted during the spans; a tone gated to ONLY the
        # spans is mixed on top. normalize=0 keeps the programme level
        # (the -14 LUFS conform finishes the job at finalization anyway).
        fc = (
            f"[0:a]volume=0:enable='{enable}'[prog];"
            f"sine=frequency={freq}:sample_rate=48000[tone];"
            f"[tone]volume=0.30,volume=0:enable='not({enable})'[beep];"
            f"[prog][beep]amix=inputs=2:duration=first:"
            f"dropout_transition=0:normalize=0[aout]"
        )
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=p.suffix, dir=str(p.parent))
        os.close(tmp_fd)
        try:
            ffmpeg = os.environ.get("KAIZER_FFMPEG_BIN", "ffmpeg")
            proc = subprocess.run(
                [ffmpeg, "-y", "-v", "error", "-i", str(p),
                 "-filter_complex", fc,
                 "-map", "0:v?", "-map", "[aout]",
                 "-c:v", "copy",
                 "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                 "-movflags", "+faststart",
                 tmp_path],
                capture_output=True, text=True, timeout=timeout,
            )
            if proc.returncode != 0 or not (os.path.isfile(tmp_path)
                                            and os.path.getsize(tmp_path) > 0):
                print(f"[v4/bleep] apply failed: {(proc.stderr or '')[-300:]}",
                      flush=True)
                return False
            os.replace(tmp_path, str(p))
        finally:
            try:
                if os.path.isfile(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
        return True
    except Exception as exc:
        print(f"[v4/bleep] apply failed (soft-skip): {exc}", flush=True)
        return False


# ── Orchestrator entry point ─────────────────────────────────────────

def run_bleep_pass(*, trim_result, out_dir: Path) -> Optional[dict]:
    """Stage 1.5: detect + censor profanity on the trimmed master.

    Words come per-story in STORY-RELATIVE time (TrimmedStory.words);
    this remaps them to the absolute trimmed timeline, detects, applies,
    and writes ``bleep_report.json``. Returns the report dict, or None
    when disabled / nothing found / failed. Never raises."""
    try:
        if not _enabled():
            return None
        abs_words: list[dict] = []
        for s in (trim_result.stories or []):
            base = float(getattr(s, "video_t_start", 0.0) or 0.0)
            for w in (getattr(s, "words", None) or []):
                try:
                    abs_words.append({
                        "w": w.get("w", ""),
                        "s": base + float(w.get("s", 0.0)),
                        "e": base + float(w.get("e", 0.0)),
                    })
                except (TypeError, ValueError):
                    continue
        spans = detect_bleep_spans(abs_words)
        report = {
            "schema": 1,
            "enabled": True,
            "spans": spans,
            "applied": False,
            "word_count_scanned": len(abs_words),
        }
        if spans:
            report["applied"] = apply_bleep(trim_result.trimmed_path, spans)
            flat = [w for sp in spans for w in sp["words"]]
            state = "BLEEPED" if report["applied"] else "detected but NOT applied"
            print(f"[v4/bleep] {len(spans)} span(s), {len(flat)} word(s) {state}: "
                  f"{', '.join(flat[:10])}", flush=True)
        else:
            print(f"[v4/bleep] clean — no censorable words in "
                  f"{len(abs_words)} scanned", flush=True)
        try:
            (Path(out_dir) / BLEEP_REPORT_NAME).write_text(
                json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError as exc:
            print(f"[v4/bleep] report write failed (soft): {exc}", flush=True)
        return report
    except Exception as exc:
        print(f"[v4/bleep] pass failed (soft-skip, render continues): {exc}",
              flush=True)
        return None


def load_report_spans(work_dir) -> list[tuple[float, float]]:
    """[(start, end), …] from a job dir's bleep report ([] when absent).
    Used by render_bulletin to hash the spans overlapping each story."""
    try:
        p = Path(work_dir) / BLEEP_REPORT_NAME
        if not p.is_file():
            return []
        data = json.loads(p.read_text(encoding="utf-8"))
        if not data.get("applied"):
            return []
        return [(float(sp["start"]), float(sp["end"]))
                for sp in (data.get("spans") or [])]
    except Exception:
        return []
