"""Trailer engine — a movie-style teaser cut from any finished job.

"Like how a movie has a trailer": the system picks the most dramatic
micro-moments from the word-timestamped transcript, cuts them into a
fast, escalating sequence with punchy color, varied transitions, hook /
end title cards in the language font, and SYNTHESIZED sound design
(impact hits, whooshes, a riser into the finale — generated with ffmpeg,
so there is nothing to license), in 16:9 or 9:16. Finished file is
loudness-conformed like every other final.

Pipeline (all on the existing substrate):
  1. plan_trailer_moments — Gemini picks 5-8 quotable/dramatic windows
     (+ a hook line + per-moment overlay text) from the per-story words
     the Phase-1 engine already produces; a strict sanitizer bounds
     everything; deterministic fallback = each story's opening beat.
  2. render_trailer — per-moment graded slices (contrast/saturation
     punch, vignette, cinematic bars on 16:9, PIL text overlays), hook
     card → moments → end card stitched with per-joint xfade transitions
     (native motion), then a sound-design pass (SFX at every joint +
     riser under the finale) and the -14 LUFS conform.

Drop premium SFX into ``<backend>/assets/sfx/{hit,whoosh,riser}.wav`` to
replace the synthesized ones — the engine prefers files when present.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

# ── Planner ──────────────────────────────────────────────────────────

_TRAILER_SYSTEM = """You are a movie-trailer editor cutting a teaser for a video (news bulletin, podcast or any spoken-word piece). From the word-level transcript you pick the moments that make people NEED to watch.

Pick 10-15 MOMENTS, each 2-5 seconds (a proper teaser runs 45-90 seconds total), choosing:
- the most dramatic reveals, numbers, names, accusations, promises
- the most quotable single lines (a moment should be ONE punchy beat, never a full explanation)
- spread across the WHOLE video (beginning, middle, end) so the teaser samples everything
Also write:
- "hook_text": one short opening line for the title card (native script of the content, max 8 words) that sets the stakes
- per moment an optional "overlay" (max 6 words, native script) shown on screen — only when a caption genuinely adds punch, else ""
- "category": the content's mood/genre, EXACTLY one of:
  news, breaking_news, politics, crime, horror, thriller, action, sports,
  cricket, tech, business, finance, health, education, devotional,
  mythology, history, documentary, travel, food, comedy, romance, drama,
  music, festival, motivational, celebrity, movie_review, gaming, kids,
  weather
  (this drives the trailer's whole look and sound design — a crime story
  gets a dark tense cut, a festival story gets a bright celebratory one)

- per moment an optional "mood" from the SAME category list — set it ONLY when that moment's mood clearly differs from the overall category (a funny beat inside a crime story → "comedy"; a tragic beat inside a comedy → "drama"). Most moments should omit it. The editor then treats each moment in its own mood (look + sound) while the overall category carries the cut.

OUTPUT one JSON object, no prose:
{"hook_text": "...", "category": "crime", "moments": [
  {"t_start": 12.4, "t_end": 15.1, "overlay": "...", "punch": 0.9, "mood": ""}
]}
Times come from the transcript timestamps. "punch" 0-1 = how hard the moment hits (the finale should be your highest). Moments must not overlap and must be sorted."""


def _target_trailer_sec() -> float:
    """Operator-tunable teaser length (a real trailer is 45-90 s, not a
    10 s blip). KAIZER_V4_TRAILER_TARGET_SEC, default 60."""
    try:
        return max(20.0, min(150.0, float(
            os.environ.get("KAIZER_V4_TRAILER_TARGET_SEC", "60"))))
    except ValueError:
        return 60.0


def sanitize_moments(
    raw: list, *, duration: float, min_len: float = 1.6, max_len: float = 6.0,
    cap: int = 16, total_cap: float = None,
) -> list[dict]:
    """Bound the model's picks: clamp to the timeline, enforce per-moment
    and total length, drop overlaps (earlier wins), cap the count."""
    out: list[dict] = []
    for m in (raw or []):
        if not isinstance(m, dict):
            continue
        try:
            a = max(0.0, float(m.get("t_start")))
            b = min(float(duration), float(m.get("t_end")))
        except (TypeError, ValueError):
            continue
        if b - a < min_len * 0.5:
            continue
        if b - a < min_len:
            b = min(duration, a + min_len)
        if b - a > max_len:
            b = a + max_len
        if b - a < min_len * 0.5:
            continue
        try:
            punch = max(0.0, min(float(m.get("punch", 0.5) or 0.5), 1.0))
        except (TypeError, ValueError):
            punch = 0.5
        mood = ""
        try:
            from pipeline_v4.trailer_styles import (SUBCATEGORY_ALIASES,
                                                    STYLES, resolve_category)
            mk = str(m.get("mood", "") or "").strip().lower()
            if mk and (mk in STYLES or mk in SUBCATEGORY_ALIASES):
                mood = resolve_category(mk)   # invalid mood → "" → pack of the cut
        except Exception:
            mood = ""
        out.append({"t_start": round(a, 3), "t_end": round(b, 3),
                    "overlay": str(m.get("overlay", "") or "")[:80],
                    "punch": punch, "mood": mood})
    out.sort(key=lambda m: m["t_start"])
    if total_cap is None:
        total_cap = _target_trailer_sec()
    kept: list[dict] = []
    for m in out:
        if kept and m["t_start"] < kept[-1]["t_end"] + 0.1:
            continue
        kept.append(m)
        if len(kept) >= cap:
            break
    while kept and sum(m["t_end"] - m["t_start"] for m in kept) > total_cap:
        kept.pop()
    return kept


def ensure_min_content(moments: list[dict], *, duration: float,
                       target: float = None) -> list[dict]:
    """GUARANTEE a real teaser length. The caps fix alone wasn't enough:
    a stingy plan (short source / one story / lazy LLM) still yielded a
    ~6s cut that was mostly cards. This expander (pure, deterministic):

      1. computes the ACHIEVABLE content target — the env target, but
         never more than ~55% of the source (you can't tease longer
         than the film);
      2. WIDENS existing moments toward 6s (never overlapping the next);
      3. if still short, ADDS sampler moments in the largest unused
         stretches (a real trailer samples the whole timeline anyway);
    Sorted, non-overlapping, inside [0, duration]. Never raises."""
    ms = sorted((dict(m) for m in (moments or [])),
                key=lambda m: float(m["t_start"]))
    if not ms or duration <= 4.0:
        return ms
    if target is None:
        target = _target_trailer_sec()
    eff = min(float(target), max(8.0, duration * 0.55))

    def total() -> float:
        return sum(float(m["t_end"]) - float(m["t_start"]) for m in ms)

    # 1) widen each moment up to 6s toward the next moment / the end
    for i, m in enumerate(ms):
        if total() >= eff:
            break
        limit = (float(ms[i + 1]["t_start"]) - 0.15 if i + 1 < len(ms)
                 else duration)
        new_end = min(float(m["t_start"]) + 6.0, limit,
                      float(m["t_end"]) + (eff - total()))
        if new_end > float(m["t_end"]):
            m["t_end"] = round(new_end, 3)

    # 2) fill the largest unused stretches with 3.2s sampler moments
    guard = 0
    while total() < eff and guard < 24:
        guard += 1
        gaps = []
        prev = 0.0
        for m in ms:
            if float(m["t_start"]) - prev >= 4.0:
                gaps.append((prev, float(m["t_start"])))
            prev = max(prev, float(m["t_end"]))
        if duration - prev >= 4.0:
            gaps.append((prev, duration))
        if not gaps:
            break
        a, b = max(gaps, key=lambda g: g[1] - g[0])
        mid = (a + b) / 2.0
        take = min(3.2, b - a - 0.8, eff - total())
        if take < 1.2:
            break
        ms.append({"t_start": round(mid - take / 2, 3),
                   "t_end": round(mid + take / 2, 3),
                   "overlay": "", "punch": 0.55, "mood": ""})
        ms.sort(key=lambda m: float(m["t_start"]))
    return ms


def fallback_moments(stories, *, duration: float, per: float = 3.4,
                     cap: int = 12) -> list[dict]:
    """Deterministic plan when the LLM is unavailable: each story's
    opening beat (headlines are read first — natural trailer material)."""
    out = []
    for s in (stories or [])[:cap]:
        try:
            a = float(getattr(s, "video_t_start", None) if not isinstance(s, dict)
                      else s.get("video_t_start", 0.0)) + 0.3
        except (TypeError, ValueError):
            continue
        b = min(float(duration), a + per)
        if b - a >= 1.2:
            out.append({"t_start": round(a, 3), "t_end": round(b, 3),
                        "overlay": "", "punch": 0.6})
    return out


def plan_trailer_moments(
    *, stories, words_by_story: dict, duration: float, language: str = "",
) -> dict:
    """LLM trailer plan over the ABSOLUTE-timeline words; sanitized; the
    deterministic fallback on any failure. Returns
    ``{"hook_text": str, "moments": [...]}`` (moments possibly from
    fallback, never empty when stories exist). Never raises."""
    hook = ""
    category = ""
    moments: list[dict] = []
    try:
        lines = []
        n = 0
        for s in (stories or []):
            base = float(getattr(s, "video_t_start", None) if not isinstance(s, dict)
                         else s.get("video_t_start", 0.0) or 0.0)
            sw = words_by_story.get(str(
                getattr(s, "story_index", None) if not isinstance(s, dict)
                else s.get("story_index", 0))) or []
            title = (getattr(s, "title_native", None) if not isinstance(s, dict)
                     else s.get("title_native", "")) or ""
            lines.append(f"--- story: {title}")
            for w in sw:
                if n >= 1200:
                    break
                try:
                    lines.append(f"[{base + float(w['s']):.2f}] {w['w']}")
                    n += 1
                except (TypeError, ValueError, KeyError):
                    continue
        user = (f"Video duration: {duration:.1f}s. Language: {language or 'unknown'}.\n"
                f"Word transcript (absolute seconds):\n" + "\n".join(lines))
        from seo.generator import _gemini_client
        from google.genai import types as genai_types
        from pipeline_v4.trim_engine import _loads_lenient
        model = os.environ.get("KAIZER_V4_TRAILER_MODEL", "gemini-2.5-flash")
        # Bind the client to a name for the whole call — an inline
        # `_gemini_client().models.generate_content(...)` gets GC'd mid-
        # request in google-genai 2.4.0 ("client has been closed"). Same
        # bug (and fix) as pipeline_v4/director.py:plan_direction.
        _client = _gemini_client()
        resp = _client.models.generate_content(
            model=model, contents=user,
            config=genai_types.GenerateContentConfig(
                system_instruction=_TRAILER_SYSTEM,
                response_mime_type="application/json",
                temperature=0.4, max_output_tokens=2048,
            ),
        )
        data = _loads_lenient((resp.text or "").strip())
        hook = str(data.get("hook_text", "") or "")[:80]
        from pipeline_v4.trailer_styles import STYLES
        cat = str(data.get("category", "") or "").strip().lower()
        category = cat if cat in STYLES else ""
        moments = sanitize_moments(data.get("moments") or [], duration=duration)
        if moments:
            print(f"[v4/trailer] planner picked {len(moments)} moments "
                  f"({sum(m['t_end'] - m['t_start'] for m in moments):.1f}s)"
                  f"{f', category={category}' if category else ''}",
                  flush=True)
    except Exception as exc:
        print(f"[v4/trailer] planner failed ({exc}) — fallback moments", flush=True)
    if not moments:
        moments = fallback_moments(stories, duration=duration)
        print(f"[v4/trailer] fallback: {len(moments)} story-opening moments",
              flush=True)
    # Length guarantee: widen + sample until the teaser has real content
    # (a stingy plan on a short source produced 6s cuts that were mostly
    # cards — operator-reported). Applies to LLM and fallback paths.
    before = sum(m["t_end"] - m["t_start"] for m in moments)
    moments = ensure_min_content(moments, duration=duration)
    after = sum(m["t_end"] - m["t_start"] for m in moments)
    if after - before > 0.5:
        print(f"[v4/trailer] length guarantee: content {before:.1f}s -> "
              f"{after:.1f}s ({len(moments)} moments)", flush=True)
    if not hook:
        s0 = (stories or [None])[0]
        hook = ((getattr(s0, "title_native", None) if s0 is not None and not isinstance(s0, dict)
                 else (s0 or {}).get("title_native", "")) or "")[:80]
    return {"hook_text": hook, "category": category, "moments": moments}


# ── Sound design (synthesized — nothing to license) ─────────────────

def _sfx_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "assets" / "sfx"


def ensure_sfx(work_dir: Path, style=None) -> dict[str, str]:
    """{kind: wav_path} for hit / whoosh / riser, SYNTHESIZED WITH THE
    STYLE PACK'S SOUND-DESIGN PARAMETERS — horror gets a 45 Hz long-decay
    boom, comedy a bright pop, crime a dark band-limited whoosh (spec:
    each category has its own sound effects). Operator-supplied files in
    ``assets/sfx/{key}_{kind}.wav`` (per-style) or ``assets/sfx/{kind}.wav``
    (global) win over synthesis. Cached per style in ``work_dir``.
    Missing pieces are simply omitted (fail-soft)."""
    from pipeline_v4.trailer_styles import get_style
    st = style or get_style("news")
    ffmpeg = os.environ.get("KAIZER_FFMPEG_BIN", "ffmpeg")
    hp, lp = st.whoosh_band
    hit_dur = st.hit_decay + 0.1
    recipes = {
        # The category "boom" — fundamental + decay from the pack.
        "hit": (f"sine=frequency={st.hit_freq}:duration={hit_dur:.2f}",
                f"afade=t=out:st=0.05:d={st.hit_decay:.2f},volume=2.2"),
        # The cut "whoosh" — band-limited pink noise, pack-colored.
        "whoosh": ("anoisesrc=color=pink:duration=0.55:amplitude=0.8",
                   f"highpass=f={hp},lowpass=f={lp},"
                   f"afade=t=in:st=0:d=0.22,afade=t=out:st=0.22:d=0.33,volume=1.1"),
        # Tension riser into the finale — darkness from the pack.
        "riser": ("anoisesrc=color=pink:duration=2.6:amplitude=0.6",
                  f"lowpass=f={st.riser_lp},afade=t=in:st=0:d=2.2,"
                  f"afade=t=out:st=2.35:d=0.25,volume=1.0"),
    }
    out: dict[str, str] = {}
    for kind, (src, af) in recipes.items():
        for custom in (_sfx_dir() / f"{st.key}_{kind}.wav",
                       _sfx_dir() / f"{kind}.wav"):
            if custom.is_file():
                out[kind] = str(custom)
                break
        if kind in out:
            continue
        dst = Path(work_dir) / f"_sfx_{st.key}_{kind}.wav"
        if dst.is_file():
            out[kind] = str(dst)
            continue
        try:
            subprocess.run(
                [ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-i", src,
                 "-af", af, "-ar", "48000", "-ac", "2", str(dst)],
                check=True, capture_output=True, timeout=60)
            out[kind] = str(dst)
        except Exception as exc:
            print(f"[v4/trailer] sfx '{kind}' synth failed (skipping): {exc}",
                  flush=True)
    return out


def synth_bed(work_dir: Path, style, duration: float) -> Optional[str]:
    """The pack's mood bed (dark drone for horror, warm pad for
    devotional, …) synthesized at the trailer's exact length. None when
    the pack has no bed or synthesis fails."""
    if not getattr(style, "bed", None) or duration <= 1.0:
        return None
    src_tpl, af = style.bed
    src = src_tpl.format(d=f"{duration:.2f}")
    dst = Path(work_dir) / f"_bed_{style.key}.wav"
    try:
        # A multi-source bed ("sine=..,sine=..") becomes an amix.
        parts = src.split(",")
        cmd = [os.environ.get("KAIZER_FFMPEG_BIN", "ffmpeg"), "-y", "-v", "error"]
        for p in parts:
            cmd += ["-f", "lavfi", "-i", p.strip()]
        if len(parts) > 1:
            cmd += ["-filter_complex",
                    "".join(f"[{i}:a]" for i in range(len(parts)))
                    + f"amix=inputs={len(parts)}:duration=first:"
                      f"dropout_transition=0[m];[m]{af},"
                      f"afade=t=in:st=0:d=1.0,"
                      f"afade=t=out:st={max(0.0, duration - 1.2):.2f}:d=1.2[out]",
                    "-map", "[out]"]
        else:
            cmd += ["-af", f"{af},afade=t=in:st=0:d=1.0,"
                           f"afade=t=out:st={max(0.0, duration - 1.2):.2f}:d=1.2"]
        cmd += ["-t", f"{duration:.2f}", "-ar", "48000", "-ac", "2", str(dst)]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return str(dst) if dst.is_file() and dst.stat().st_size > 0 else None
    except Exception as exc:
        print(f"[v4/trailer] bed synth failed (skipping): {exc}", flush=True)
        return None


# ── Cards + treatment ────────────────────────────────────────────────

def _render_card_png(*, text: str, sub: str, w: int, h: int,
                     font_path: Optional[str], out_path: str,
                     bg: tuple = (8, 8, 10),
                     accent: tuple = (193, 18, 18)) -> Optional[str]:
    """Cinematic title card in the style pack's colors: dark field,
    accent rule, big centered text."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (w, h), tuple(bg))
        d = ImageDraw.Draw(img)
        d.rectangle([w // 2 - 90, int(h * 0.38), w // 2 + 90, int(h * 0.38) + 6],
                    fill=tuple(accent))

        def _font(size):
            try:
                return (ImageFont.truetype(font_path, size)
                        if font_path and os.path.isfile(font_path)
                        else ImageFont.load_default())
            except Exception:
                return ImageFont.load_default()

        main = " ".join((text or "").split())[:90]
        size = 72 if w >= 1900 else 56
        f = _font(size)
        while main and size > 28:
            box = d.textbbox((0, 0), main, font=f)
            if box[2] - box[0] <= w - 160:
                break
            size -= 6
            f = _font(size)
        box = d.textbbox((0, 0), main, font=f)
        d.text(((w - (box[2] - box[0])) // 2,
                int(h * 0.44)), main, font=f, fill=(245, 245, 247))
        if sub:
            fs = _font(30)
            sb = d.textbbox((0, 0), sub, font=fs)
            d.text(((w - (sb[2] - sb[0])) // 2, int(h * 0.44) + size + 40),
                   sub, font=fs, fill=(200, 200, 208))
        img.save(out_path, "PNG")
        return out_path
    except Exception as exc:
        print(f"[v4/trailer] card render failed: {exc}", flush=True)
        return None


def _overlay_png(*, text: str, w: int, font_path: Optional[str],
                 out_path: str) -> Optional[str]:
    """Per-moment caption strip (PIL — Indic-safe, unlike drawtext)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        H = 110
        img = Image.new("RGBA", (w, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        try:
            f = (ImageFont.truetype(font_path, 44)
                 if font_path and os.path.isfile(font_path)
                 else ImageFont.load_default())
        except Exception:
            f = ImageFont.load_default()
        t = " ".join((text or "").split())[:60]
        box = d.textbbox((0, 0), t, font=f)
        tw = box[2] - box[0]
        x0 = (w - tw) // 2
        d.rectangle([x0 - 24, 8, x0 + tw + 24, H - 8], fill=(8, 8, 10, 205))
        d.rectangle([x0 - 24, 8, x0 - 18, H - 8], fill=(193, 18, 18, 255))
        d.text((x0, (H - (box[3] - box[1])) // 2 - box[1]), t, font=f,
               fill=(255, 255, 255, 255))
        img.save(out_path, "PNG")
        return out_path
    except Exception:
        return None


# ── Structures — assembly architectures over the same ingredients ────

def apply_structure(structure: str, moments: list[dict]) -> dict:
    """Turn a TRAILER_STRUCTURES entry + the sanitized moments into a
    concrete assembly plan. Pure function (unit-tested). Unknown
    structure → classic. Never raises."""
    from pipeline_v4.trailer_styles import TRAILER_STRUCTURES
    key = (structure or "classic").strip().lower()
    if key not in TRAILER_STRUCTURES:
        key = "classic"
    spec = TRAILER_STRUCTURES[key]
    ms = list(moments or [])
    if spec.get("order") == "punch_asc":
        ms = sorted(ms, key=lambda m: float(m.get("punch", 0.5) or 0.5))
    cold = None
    if spec.get("cold_open") and len(ms) > 1:
        if spec.get("cold_pick") == "overlay":
            cands = [m for m in ms if m.get("overlay")] or ms
        else:
            cands = ms
        cold = max(cands, key=lambda m: float(m.get("punch", 0.5) or 0.5))
        ms = [cold] + [m for m in ms if m is not cold]
    n = len(ms)
    number_before: dict[int, str] = {}
    for j, val in enumerate(range(min(3, spec.get("number_cards", 0) or 0), 0, -1)):
        idx = n - min(3, spec.get("number_cards", 0) or 0) + j
        if 0 <= idx < n:
            number_before[idx] = str(val)
    stinger: list[tuple[float, float]] = []
    if spec.get("stinger"):
        tops = sorted(ms, key=lambda m: -float(m.get("punch", 0.5) or 0.5))
        for m in tops[:int(spec["stinger"])]:
            stinger.append((float(m["t_start"]), 0.9))
    return {
        "key": key,
        "moments": ms,
        "hook_first": spec.get("hook_first", True),   # True/False/None(no card)
        "cold_open": bool(spec.get("cold_open")),
        "stinger": stinger,                            # [(src_t, dur)] pre-title
        "number_before": number_before,                # {moment_idx: "3"}
        "mid_card": (spec.get("mid_card") if n >= 4 else None),
        "mid_index": (n // 2 if (spec.get("mid_card") and n >= 4) else None),
        "reprise": (dict(ms[0]) if spec.get("reprise") and n >= 2 else None),
        "pace_mult": float(spec.get("pace_mult", 1.0) or 1.0),
        "hook_hold": float(spec.get("hook_hold", 2.0) or 2.0),
    }


def render_trailer(
    *,
    source_path: str,
    stories,
    words_by_story: dict,
    out_path: str,
    work_dir,
    aspect: str = "16:9",
    language: str = "te",
    channel_name: str = "",
    plan: Optional[dict] = None,
    style: str = "auto",
    structure: str = "",
    style_override=None,   # composed TrailerStyle (user-mixed pack) wins
) -> str:
    """Cut the trailer. Raises on hard failure (caller reports); every
    cosmetic layer (cards, SFX, overlays) is individually fail-soft."""
    from pipeline_v4.v1_bridge import (_enc_args, _ffmpeg_bin, _lang_cfg,
                                       _probe_clip_duration,
                                       build_xfade_stitch_graph)
    from pipeline_v4.audio_conform import conform_loudness

    W, H = (1080, 1920) if aspect == "9:16" else (1920, 1080)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg_bin()
    lang_cfg = _lang_cfg(language)
    duration = _probe_clip_duration(source_path)
    if not duration:
        raise RuntimeError(f"trailer: cannot probe source {source_path}")

    if plan is None:
        plan = plan_trailer_moments(stories=stories, words_by_story=words_by_story,
                                    duration=duration, language=language)
    moments = plan.get("moments") or []
    if not moments:
        raise RuntimeError("trailer: no usable moments")
    hook_text = plan.get("hook_text") or (channel_name or "COMING UP")

    # Assembly structure: explicit param > env > classic. Reorders the
    # moments and dictates stingers / number cards / mid card / reprise.
    sp = apply_structure(
        structure or os.environ.get("KAIZER_V4_TRAILER_STRUCTURE", ""),
        moments)
    moments = sp["moments"]
    print(f"[v4/trailer] structure: {sp['key']}", flush=True)

    # Style pack: explicit key wins; "auto" = the planner's category;
    # unknown/absent → news. Drives grade, transitions, cards and SFX.
    from pipeline_v4.trailer_styles import STYLES, get_style
    if style_override is not None:
        st = style_override           # user-composed pack (already resolved)
    else:
        sk = (style or "auto").strip().lower()
        st = STYLES[sk] if sk in STYLES else get_style(plan.get("category") or "")
    print(f"[v4/trailer] style pack: {st.key} ({st.label})", flush=True)

    # 1) Graded moment slices — the pack's grade + frame effects +
    #    vignette; cinematic bars per pack (16:9 only); escalating
    #    micro-heat; optional flash-cut entrance; ANIMATED text overlays
    #    (alpha fade + eased slide-up).
    bars = int(H * 0.11)
    use_bars = st.bars and aspect != "9:16"
    clip_paths: list[str] = []
    clip_packs: list = []      # pack per moment clip (mixed-mood support)
    for i, m in enumerate(moments):
        # a moment may carry its OWN mood (funny beat in a crime story…):
        # that moment gets that pack's look + joint sound, while the cut
        # overall keeps the dominant category's pace/cards/bed
        mst = STYLES.get(m.get("mood") or "", st)
        clip_packs.append(mst)
        heat = 1.0 + 0.04 * (i / max(1, len(moments) - 1))
        vf = (f"scale={W}:{H}:force_original_aspect_ratio=increase,"
              f"crop={W}:{H},setsar=1,"
              f"{mst.grade}"
              f"{',' + mst.extra_vf if mst.extra_vf else ''},"
              f"eq=contrast={heat:.3f},"
              f"vignette=PI/4.5")
        if use_bars:
            vf += (f",drawbox=x=0:y=0:w={W}:h={bars}:color=black:t=fill"
                   f",drawbox=x=0:y={H - bars}:w={W}:h={bars}:color=black:t=fill")
        if mst.flash_in:
            vf += f",fade=t=in:st=0:d=0.12:color={mst.flash_in}"
        # punch-driven frame FX (duration-neutral garnish from the 75-FX
        # registry): hard moments get handheld urgency, the finale gets a
        # flash-bang hit under its boom
        from pipeline_v4.frame_fx import get_fx_vf as _fx
        try:
            if float(m.get("punch", 0.5) or 0.5) >= 0.85:
                # shake crops a few px — scale back or xfade rejects the
                # size mismatch at stitch time (-22 Invalid argument)
                vf += "," + _fx("micro_shake") + f",scale={W}:{H}"
            if i == len(moments) - 1:
                vf += "," + _fx("flash_bang")
        except Exception:
            pass
        clip = str(work / f"_tr_m{i:02d}.mp4")
        cmd = [ffmpeg, "-y", "-v", "error",
               "-ss", f"{m['t_start']:.3f}", "-to", f"{m['t_end']:.3f}",
               "-i", source_path]
        ov = None
        ov_seq = None
        if m.get("overlay"):
            # PROPER text animation (operator): hard moments get the
            # typography engine's physics entrances as PNG SEQUENCES —
            # spring crash on bright packs, whip on dark packs, fade-
            # scale on soft ones. Softer moments keep the eased slide.
            if float(m.get("punch", 0.5) or 0.5) >= 0.7:
                try:
                    from pipeline_v4 import typography as _ty
                    _motion = ("crash" if mst.flash_in == "white"
                               else "whip_left" if mst.hit_freq <= 60
                               else "fade_scale")
                    _wins = _ty.headline_crash(
                        m["overlay"], out_dir=str(work), frames=12, fps=30,
                        canvas_w=max(400, W - 200),
                        font_size=(64 if aspect == "9:16" else 78),
                        color=tuple(mst.card_accent),
                        prefix=f"ovsq{i:02d}", motion=_motion)
                    if _wins:
                        ov_seq = str(work / f"_ovsq{i:02d}_%03d.png")
                except Exception as _tx:
                    print(f"[v4/trailer] animated overlay failed (soft): {_tx}",
                          flush=True)
            if not ov_seq:
                ov = _overlay_png(text=m["overlay"], w=W,
                                  font_path=lang_cfg.font_primary,
                                  out_path=str(work / f"_tr_ov{i:02d}.png"))
        if ov_seq:
            # finite image2 sequence: the 12 animation frames play at
            # 30fps then eof_action=repeat HOLDS the settled frame to
            # the end of the clip — physics entrance + steady hold.
            cmd += ["-framerate", "30", "-i", ov_seq,
                    "-filter_complex",
                    f"[1:v]format=rgba[ovf];"
                    f"[0:v]{vf}[v0];[v0][ovf]overlay="
                    f"x='(W-w)/2':y='(H-h)*0.70':eof_action=repeat[outv]",
                    "-map", "[outv]", "-map", "0:a?"]
        elif ov:
            # -shortest is REQUIRED here: the looped overlay PNG is an
            # infinite input — without it ffmpeg encodes forever (until
            # the watchdog timeout) instead of stopping at the clip end.
            # Text animation: alpha fade-in + eased slide-up (native
            # motion out-cubic baked into the overlay y expression).
            oy = H - (bars if use_bars else 0) - 130
            cmd += ["-loop", "1", "-i", ov,
                    "-filter_complex",
                    f"[1:v]format=rgba,fade=t=in:st=0:d=0.3:alpha=1[ovf];"
                    f"[0:v]{vf}[v0];[v0][ovf]overlay=x=0:"
                    f"y='{oy}+60*pow(max(0\\,1-t/0.35)\\,3)'"
                    f":shortest=1[outv]",
                    "-map", "[outv]", "-map", "0:a?", "-shortest"]
        else:
            cmd += ["-vf", vf, "-map", "0:v", "-map", "0:a?"]
        cmd += [*_enc_args(crf=19, preset_hint="medium"), "-pix_fmt", "yuv420p",
                "-r", "30", "-fps_mode", "cfr",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                "-movflags", "+faststart", clip]
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        clip_paths.append(clip)

    # 2) Hook + end cards — pack colors, slow PUSH-IN (zoompan on an
    #    oversized still: static text feels dead in a trailer), fades,
    #    silent audio bed.
    def _card_clip(png: Optional[str], dur: float, name: str) -> Optional[str]:
        if not png:
            return None
        out = str(work / name)
        frames = max(2, int(round(dur * 30)))
        try:
            subprocess.run(
                [ffmpeg, "-y", "-v", "error",
                 "-i", png,
                 "-f", "lavfi", "-t", f"{dur:.2f}",
                 "-i", "anullsrc=r=48000:cl=stereo",
                 "-vf", f"zoompan=z='min(1+0.0012*on,1.06)':"
                        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                        f"d={frames}:s={W}x{H}:fps=30,setsar=1,"
                        f"fade=t=in:st=0:d=0.3,fade=t=out:st={dur - 0.35:.2f}:d=0.35",
                 *_enc_args(crf=19, preset_hint="medium"), "-pix_fmt", "yuv420p",
                 "-r", "30", "-fps_mode", "cfr",
                 "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                 "-shortest", "-movflags", "+faststart", out],
                check=True, capture_output=True, timeout=300)
            return out
        except Exception as exc:
            print(f"[v4/trailer] card clip failed (skipping): {exc}", flush=True)
            return None

    # Cards rendered 25% oversized so the push-in never softens the text.
    cw, ch = int(W * 1.25), int(H * 1.25)
    hook_png = _render_card_png(text=hook_text, sub="", w=cw, h=ch,
                                font_path=lang_cfg.font_primary,
                                bg=st.card_bg, accent=st.card_accent,
                                out_path=str(work / "_tr_hook.png"))
    end_png = _render_card_png(text=channel_name or "WATCH NOW",
                               sub=st.card_sub, w=cw, h=ch,
                               font_path=lang_cfg.font_primary,
                               bg=st.card_bg, accent=st.card_accent,
                               out_path=str(work / "_tr_end.png"))
    # Structure-aware assembly: stingers → title (per placement) →
    # moments interleaved with number/mid cards → reprise → end card.
    def _stinger_clip(k: int, src_t: float, dur: float) -> Optional[str]:
        """Blink-fast graded glimpse (no overlays) for flash-forward."""
        out = str(work / f"_tr_sting{k}.mp4")
        vf = (f"scale={W}:{H}:force_original_aspect_ratio=increase,"
              f"crop={W}:{H},setsar=1,{st.grade},eq=contrast=1.06")
        try:
            subprocess.run(
                [ffmpeg, "-y", "-v", "error",
                 "-ss", f"{src_t:.3f}", "-t", f"{dur:.2f}",
                 "-i", source_path, "-vf", vf,
                 "-map", "0:v", "-map", "0:a?",
                 *_enc_args(crf=19, preset_hint="medium"),
                 "-pix_fmt", "yuv420p", "-r", "30", "-fps_mode", "cfr",
                 "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                 "-movflags", "+faststart", out],
                check=True, capture_output=True, timeout=300)
            return out
        except Exception:
            return None

    def _number_card(val: str, k: int) -> Optional[str]:
        png = _render_card_png(text=val, sub="", w=cw, h=ch,
                               font_path=lang_cfg.font_primary,
                               bg=st.card_bg, accent=st.card_accent,
                               out_path=str(work / f"_tr_num{k}.png"))
        return _card_clip(png, 1.0, f"_tr_card_num{k}.mp4")

    hook_clip = (None if sp["hook_first"] is None
                 else _card_clip(hook_png, sp["hook_hold"], "_tr_card_hook.mp4"))
    mid_clip = None
    if sp["mid_card"]:
        mid_png = _render_card_png(text=sp["mid_card"], sub="", w=cw, h=ch,
                                   font_path=lang_cfg.font_primary,
                                   bg=st.card_bg, accent=st.card_accent,
                                   out_path=str(work / "_tr_mid.png"))
        mid_clip = _card_clip(mid_png, 1.4, "_tr_card_mid.mp4")

    assembly: list[tuple[Optional[str], object]] = []
    for k, (s_t, s_d) in enumerate(sp["stinger"]):
        assembly.append((_stinger_clip(k, s_t, s_d), st))
    if sp["hook_first"] is True:
        assembly.append((hook_clip, st))
    for i2, (c, pk) in enumerate(zip(clip_paths, clip_packs)):
        if i2 in sp["number_before"]:
            assembly.append((_number_card(sp["number_before"][i2], i2), st))
        if sp["mid_index"] is not None and i2 == sp["mid_index"]:
            assembly.append((mid_clip, st))
        assembly.append((c, pk))
        if i2 == 0 and sp["hook_first"] is False:
            assembly.append((hook_clip, st))     # cold open: title after
    if sp["reprise"]:
        rm = sp["reprise"]
        r_end = min(float(rm["t_end"]), float(rm["t_start"]) + 1.2)
        assembly.append((_stinger_clip(99, float(rm["t_start"]),
                                       max(0.6, r_end - float(rm["t_start"]))),
                         st))
    assembly.append((_card_clip(end_png, 1.8, "_tr_card_end.mp4"), st))

    seq, seq_packs = [], []
    for c, pk in assembly:
        if c:
            seq.append(c)
            seq_packs.append(pk)

    # 3) Stitch with the PACK's transitions at the PACK's pace. If the
    #    host ffmpeg rejects a fancy xfade name, retry once with plain
    #    fades — the trailer never dies on a transition.
    durations = [(_probe_clip_duration(p) or 1.0) for p in seq]
    stitched = str(work / "_tr_stitched.mp4")
    joint_times: list[float] = []
    if len(seq) == 1:
        stitched = seq[0]
    else:
        from pipeline_v4.trailer_styles import (builtin_equivalent,
                                                xfade_arg as _xfade_arg)
        # A transition must be SEEN (operator: 'looks like hard cuts').
        # Pack pace now scales a 0.45-0.9s window instead of being the
        # raw duration (0.12-0.25s = 4-8 frames = perceptually a cut),
        # capped so short clips (stingers) aren't eaten by the fade.
        fade_d = max(0.45, min(0.9, st.pace * sp["pace_mult"] * 2.4))
        fade_d = max(0.30, min(fade_d, min(durations) / 2.5))

        def _stitch(trans_list: Optional[tuple]) -> None:
            nonlocal joint_times
            joint_times = []
            fc_parts: list[str] = []
            v_cur, a_cur = "[0:v]", "[0:a]"
            total = durations[0]
            for i in range(1, len(seq)):
                off = max(0.0, total - fade_d)
                joint_times.append(off)
                if trans_list is None:
                    # mixed-mood: the INCOMING clip announces itself with
                    # its own pack's transition
                    tp = seq_packs[i].transitions
                    trans = tp[(i - 1) % len(tp)]
                else:
                    trans = trans_list[(i - 1) % len(trans_list)]
                v_out = f"[v{i}]" if i < len(seq) - 1 else "[vout]"
                a_out = f"[a{i}]" if i < len(seq) - 1 else "[aout]"
                fc_parts.append(f"{v_cur}[{i}:v]xfade={_xfade_arg(trans)}:"
                                f"duration={fade_d:.3f}:offset={off:.3f}{v_out}")
                fc_parts.append(f"{a_cur}[{i}:a]acrossfade=d={fade_d:.3f}{a_out}")
                v_cur, a_cur = v_out, a_out
                total = off + fade_d + (durations[i] - fade_d)
            cmd = [ffmpeg, "-y", "-v", "error"]
            for p in seq:
                cmd += ["-i", p]
            cmd += ["-filter_complex", ";".join(fc_parts),
                    "-map", "[vout]", "-map", "[aout]",
                    *_enc_args(crf=19, preset_hint="medium"), "-pix_fmt", "yuv420p",
                    "-r", "30", "-fps_mode", "cfr",
                    "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                    "-movflags", "+faststart", stitched]
            subprocess.run(cmd, check=True, capture_output=True, timeout=1200)

        try:
            _stitch(None)      # per-joint packs (falls back to st for cards)
        except subprocess.CalledProcessError as exc:
            print(f"[v4/trailer] pack transitions failed "
                  f"({(exc.stderr or b'').decode(errors='replace')[-200:]}) — "
                  f"retrying with builtin equivalents", flush=True)
            try:
                # same MOTION per joint, builtin implementation — still a
                # real visible transition, never an invisible micro-fade
                _stitch(tuple(
                    builtin_equivalent(
                        seq_packs[i].transitions[(i - 1)
                                                 % len(seq_packs[i].transitions)])
                    for i in range(1, len(seq))))
            except subprocess.CalledProcessError:
                print("[v4/trailer] builtin retry failed — fadeblack",
                      flush=True)
                _stitch(("fadeblack",))

    # 4) Sound design — the PACK's synth SFX: whoosh at every joint, the
    #    category boom on the cards, riser under the finale, and (for
    #    moody packs) the mood BED under the whole cut. Audio-only pass
    #    (video stream copied).
    final_src = stitched
    sfx = ensure_sfx(work, st)
    total_dur = _probe_clip_duration(stitched) or sum(durations)
    events: list[tuple[str, float, float]] = []      # (path, at_sec, gain)
    for j, jt in enumerate(joint_times):
        # whoosh of the INCOMING clip's pack (mixed-mood: a comedy beat
        # inside a crime cut arrives with a bright pop, not a dark hiss)
        jp = seq_packs[j + 1] if j + 1 < len(seq_packs) else st
        jsfx = sfx if jp.key == st.key else ensure_sfx(work, jp)
        if "whoosh" in jsfx:
            events.append((jsfx["whoosh"], max(0.0, jt - 0.15), 0.9))
    if "hit" in sfx:
        events.append((sfx["hit"], 0.0, 0.9))                        # hook boom
        if joint_times:
            events.append((sfx["hit"], joint_times[-1] + 0.1, 0.9))  # end boom
    if "riser" in sfx and total_dur > 4.0:
        events.append((sfx["riser"], max(0.0, total_dur - 3.2), 0.9))
    bed_path = synth_bed(work, st, total_dur)
    if bed_path:
        events.append((bed_path, 0.0, st.bed_gain))
    if events:
        try:
            sfx_out = str(work / "_tr_sfx.mp4")
            cmd = [ffmpeg, "-y", "-v", "error", "-i", stitched]
            for path, _t, _g in events:
                cmd += ["-i", path]
            parts = []
            mix_ins = "[0:a]"
            for k, (_path, t, g) in enumerate(events, start=1):
                ms = int(round(t * 1000))
                parts.append(f"[{k}:a]adelay={ms}|{ms},volume={g:.2f}[e{k}]")
                mix_ins += f"[e{k}]"
            parts.append(f"{mix_ins}amix=inputs={len(events) + 1}:"
                         f"duration=first:dropout_transition=0:normalize=0[aout]")
            cmd += ["-filter_complex", ";".join(parts),
                    "-map", "0:v", "-map", "[aout]",
                    "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                    "-movflags", "+faststart", sfx_out]
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            final_src = sfx_out
        except Exception as exc:
            print(f"[v4/trailer] sound-design pass failed (keeping plain mix): {exc}",
                  flush=True)

    if os.path.abspath(final_src) != os.path.abspath(out_path):
        os.replace(final_src, out_path)
    conform_loudness(out_path)
    print(f"[v4/trailer] {aspect} trailer done — {len(moments)} moments, "
          f"{_probe_clip_duration(out_path) or 0:.1f}s → {Path(out_path).name}",
          flush=True)
    return out_path
