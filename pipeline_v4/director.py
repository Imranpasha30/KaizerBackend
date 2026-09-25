"""AI Director — decides WHAT to use WHEN, per story, like a TV editor.

The 3-layer brain (operator-approved design, chat 2026-07-06):

  1. SENSORS (facts, deterministic, fail-soft): scene cuts (ffmpeg
     scdet), audio energy peaks/silences (ebur128), plus the word
     timestamps + diarization the pipeline already has.
  2. FORMULA (the grammar): every parent category has a baseline recipe
     — pack, caption style, signature overlays, transition, bed — so a
     paid job gets a professional edit even if the LLM layer fails.
  3. LLM DIRECTOR (the judgment): Gemini (same client as trim planning)
     reads the per-story text + sensor facts + the registry VOCABULARY
     (ids only) and writes a timed decision plan per story: mood/pack,
     transition in, frame FX, overlays with timestamps, caption style,
     emphasis moments. A SANITIZER validates every id against the real
     registries and clamps every time — a hallucinated effect can never
     reach ffmpeg.

Output: {"stories": {index: StoryDirective}} consumed by the bulletin
compose (per-story effects chain + per-joint transition). Gated by
KAIZER_V4_DIRECTOR=1; any failure → formula baseline; formula failure →
empty directives (legacy render). A paid job never fails because of
the Director.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Optional

# ── Registry vocabularies (ids the LLM may use; sanitizer enforces) ──


def _vocab(override=None):
    """The ids the Director may choose from. ``override`` (the user's
    "edit using THESE" picks: {packs,transitions,fx,overlays,captions})
    CONSTRAINS a category to only the user's ids when they picked any —
    intersected with the real registry so a stale pick can never leak.
    Categories the user left empty keep the full vocabulary, so the AI
    still decides those. This is how a user directs the AI per-category."""
    from pipeline_v4.frame_fx import FX
    from pipeline_v4.overlays import REGISTRY
    from pipeline_v4.trailer_styles import STYLES, TRANSITION_CATALOG
    from pipeline_v4.typography import VARIANTS
    try:
        # LIVE LAYOUTS: only layouts the composer can truly draw today.
        from pipeline_v4.layout_library import RENDERABLE_V2
        _layouts = set(RENDERABLE_V2)
    except Exception:
        _layouts = set()
    try:
        # Color-grade overrides: the 23 broadcast looks. Fail-soft — a
        # missing/broken registry must not take the whole Director down.
        from pipeline_v4.color_grades import GRADES
        _grades = set(GRADES)
    except Exception:
        _grades = set()
    try:
        # Sound vocabulary from the owned synthesized library: the key
        # PREFIX is the family contract (sting_* opens a story, ui_*
        # ticks under graphics) — so the buckets stay honest even as
        # sounds are added.
        from pipeline_v4.sound_library import LIBRARY
        _stings = {k for k in LIBRARY if k.startswith("sting_")}
        _ui_sounds = {k for k in LIBRARY if k.startswith("ui_")}
    except Exception:
        _stings, _ui_sounds = set(), set()
    full = {
        "packs": set(STYLES),
        "transitions": set(TRANSITION_CATALOG),
        "fx": set(FX),
        "overlays": {r["id"] for r in REGISTRY},
        "captions": {v["id"] for v in VARIANTS if v["engine"] == "karaoke"}
                    | {"none"},
        "layouts": _layouts,
        "grades": _grades,
        "stings": _stings,
        "ui_sounds": _ui_sounds,
    }
    if override:
        for cat in ("packs", "transitions", "fx", "overlays", "captions",
                    "layouts", "grades", "stings", "ui_sounds"):
            picks = override.get(cat)
            if picks:
                inter = full[cat] & set(picks)
                if inter:
                    full[cat] = inter
    return full


# ── Layer 2: FORMULAS — per-parent-category baseline recipes ─────────

@dataclass(frozen=True)
class Formula:
    pack: str
    transition: str
    fx: tuple = ()                  # subtle, duration-neutral only
    overlays: tuple = ()            # overlay ids placed at story start
    captions: str = "none"          # karaoke variant id or "none"


FORMULAS: dict[str, Formula] = {
    "news":        Formula("news", "diag_soft_tl", (),
                           ("bug_live_updates",), "none"),
    "breaking_news": Formula("breaking_news", "whip_left", ("micro_shake",),
                             ("banner_flash_update",), "none"),
    "politics":    Formula("politics", "barn_door_open", (),
                           ("tag_bigstory",), "none"),
    "crime":       Formula("crime", "glitch_slices", ("vignette_soft",),
                           ("tag_investigation",), "none"),
    "sports":      Formula("sports", "whip_right", ("vibrance_pop",),
                           ("bug_sports_desk",), "none"),
    "finance":     Formula("finance", "blinds_v", (),
                           ("mkt_sensex_up",), "none"),
    "tech":        Formula("tech", "pixel_dissolve", (),
                           ("tag_tech",), "none"),
    "health":      Formula("health", "fade", (),
                           ("tag_explainer",), "none"),
    "devotional":  Formula("devotional", "clock_sweep", ("soft_glow",),
                           (), "none"),
    "comedy":      Formula("comedy", "zoom_punch_out", ("vibrance_pop",),
                           (), "none"),
    "weather":     Formula("weather", "ripple_wave", (),
                           ("banner_weather_warn",), "none"),
    "education":   Formula("education", "blinds_h", (),
                           ("tag_explainer",), "none"),
    "entertainment": Formula("celebrity", "stretch_h", (),
                             ("tag_entertainment",), "none"),
    "documentary": Formula("documentary", "dissolve", ("film_grain",),
                           (), "none"),
    "generic":     Formula("news", "fade", (), (), "none"),
}


def formula_for(category: str) -> Formula:
    from pipeline_v4.trailer_styles import resolve_category
    key = (category or "").strip().lower()
    if key in FORMULAS:
        return FORMULAS[key]
    pack = resolve_category(key)
    return FORMULAS.get(pack, FORMULAS["generic"])


# ── Layer 1: SENSORS ─────────────────────────────────────────────────

def _parse_scene_cuts(stderr: str) -> list[float]:
    cuts = []
    for m in re.finditer(r"pts_time:(\d+\.?\d*)", stderr or ""):
        cuts.append(round(float(m.group(1)), 2))
    return cuts[:400]


def sense_scene_cuts(source_path: str, *, threshold: float = 0.4,
                     timeout: int = 240) -> list[float]:
    """Scene-change timestamps via scdet. [] on any failure.

    Decodes DOWNSCALED (320px, no audio): scene detection compares frame-
    to-frame content difference, so 320px finds the same cut timestamps as
    native 1080p at a fraction of the decode cost. The old native-res pass
    exceeded the 240s timeout on long sources (a 27-min podcast), burning
    the FULL timeout and returning [] — job 610's trace showed
    "scene_cuts 0": 4 minutes spent, zero signal. On timeout the partial
    stderr is salvaged — cuts found before the deadline still count."""
    cmd = [os.environ.get("KAIZER_FFMPEG_BIN", "ffmpeg"), "-v", "info",
           "-an", "-i", source_path,
           "-vf", f"scale=320:-2,select='gt(scene,{threshold})',metadata=print",
           "-f", "null", "-"]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            errors="replace")   # ffmpeg stderr may carry non-UTF-8 metadata bytes
        return _parse_scene_cuts(proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        se = exc.stderr
        if isinstance(se, bytes):
            se = se.decode("utf-8", "replace")
        cuts = _parse_scene_cuts(se or "")
        print(f"[v4/director] scene-cut sense timed out at {timeout}s — "
              f"salvaged {len(cuts)} partial cut(s)", flush=True)
        return cuts
    except Exception:
        return []


def sense_energy_peaks(source_path: str, *, timeout: int = 240) -> dict:
    """Loudness envelope highlights via ebur128 momentary readings:
    {"peaks": [t...], "silences": [[a,b]...]}. Empty dict on failure.

    ``-vn``: ebur128 is an audio filter, but without -vn ffmpeg ALSO
    decoded the full video into the null muxer — on a 27-min 1080p source
    that blew the 240s timeout and returned {} (4 minutes burned, zero
    signal — job 610). Audio-only decode finishes a 27-min source in
    seconds. Partial stderr is salvaged on timeout."""
    _stderr = ""
    try:
        proc = subprocess.run(
            [os.environ.get("KAIZER_FFMPEG_BIN", "ffmpeg"), "-v", "info",
             "-vn", "-i", source_path, "-filter_complex", "ebur128=metadata=1",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=timeout,
            errors="replace")   # ffmpeg stderr may carry non-UTF-8 metadata bytes
        _stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        se = exc.stderr
        if isinstance(se, bytes):
            se = se.decode("utf-8", "replace")
        _stderr = se or ""
        print(f"[v4/director] energy sense timed out at {timeout}s — "
              f"salvaging partial readings", flush=True)
    except Exception:
        return {}
    try:
        readings: list[tuple[float, float]] = []
        for m in re.finditer(r"t:\s*(\d+\.?\d*)\s+.*?M:\s*(-?\d+\.?\d*)",
                             _stderr):
            readings.append((float(m.group(1)), float(m.group(2))))
        if not readings:
            return {}
        loud = sorted(readings, key=lambda r: -r[1])[:8]
        peaks = sorted(round(t, 2) for t, _ in loud)
        silences, run = [], None
        for t, m_val in readings:
            if m_val < -40.0:
                run = run or [t, t]
                run[1] = t
            elif run:
                if run[1] - run[0] >= 1.0:
                    silences.append([round(run[0], 2), round(run[1], 2)])
                run = None
        return {"peaks": peaks, "silences": silences[:20]}
    except Exception:
        return {}


_TONES = ("calm", "serious", "urgent", "angry", "fearful", "sad",
          "excited", "joyful", "tense", "neutral")


def _slice_story_audio(source_path: str, *, start: float, end: float,
                       out_dir: str, idx: int,
                       max_sec: float = 50.0) -> Optional[str]:
    """Mono 16 kHz WAV slice of one story (center-windowed when long) —
    small enough to inline to Gemini. None on any failure."""
    try:
        dur = max(0.0, float(end) - float(start))
        if dur < 1.0:
            return None
        if dur > max_sec:
            pad = (dur - max_sec) / 2.0
            start, end = float(start) + pad, float(start) + pad + max_sec
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"_tone_{idx:02d}.wav")
        ok = subprocess.run(
            [os.environ.get("KAIZER_FFMPEG_BIN", "ffmpeg"), "-y", "-v", "error",
             "-ss", f"{float(start):.2f}", "-to", f"{float(end):.2f}",
             "-i", source_path, "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", out],
            capture_output=True, timeout=180).returncode == 0
        if ok and os.path.isfile(out) and 0 < os.path.getsize(out) < 18_000_000:
            return out
        return None
    except Exception:
        return None


def _san_tone(raw: dict) -> Optional[dict]:
    """Clamp one tone reading to the fixed schema. None if unusable."""
    if not isinstance(raw, dict):
        return None
    tone = str(raw.get("tone", "") or "").strip().lower()
    if tone not in _TONES:
        return None
    try:
        arousal = max(0.0, min(1.0, float(raw.get("arousal", 0.5))))
        valence = max(-1.0, min(1.0, float(raw.get("valence", 0.0))))
    except (TypeError, ValueError):
        arousal, valence = 0.5, 0.0
    return {"tone": tone, "arousal": round(arousal, 2),
            "valence": round(valence, 2),
            "note": str(raw.get("note", "") or "")[:80]}


def sense_tone(source_path: str, stories, *, work_dir: str = "",
               language: str = "", cap: int = 10) -> dict[int, dict]:
    """LISTEN to each story's audio (Gemini multimodal): emotional tone,
    arousal, valence — acoustic evidence the words can hide (fear in a
    calm sentence). Gated KAIZER_V4_TONE_SENSE (default OFF — opt-in).
    {} on any failure — the Director then decides from words alone."""
    # Default OFF for LATENCY only: one Gemini audio call per story adds
    # ~60-90s to planning. The JSON path itself is FIXED (the old 200-token
    # cap truncated thinking-model output mid-string — "Unterminated
    # string" — so every call failed soft; now 2048 tokens + one
    # 429/parse retry, the content_type.py pattern). Set
    # KAIZER_V4_TONE_SENSE=1 to opt in.
    if (os.environ.get("KAIZER_V4_TONE_SENSE") or "0").strip() not in ("1", "on", "true"):
        return {}
    out: dict[int, dict] = {}
    try:
        import tempfile
        from seo.generator import _gemini_client
        from google.genai import types as genai_types
        from pipeline_v4.trim_engine import _loads_lenient
        client = _gemini_client()
        model = os.environ.get("KAIZER_V4_DIRECTOR_MODEL", "gemini-2.5-flash")
        wdir = work_dir or tempfile.mkdtemp(prefix="kx_tone_")
        for s in (stories or [])[:cap]:
            idx = int(getattr(s, "story_index", 0) or 0)
            wav = _slice_story_audio(
                source_path, start=float(getattr(s, "video_t_start", 0.0)),
                end=float(getattr(s, "video_t_end", 0.0)),
                out_dir=wdir, idx=idx)
            if not wav:
                continue
            try:
                with open(wav, "rb") as fh:
                    audio = genai_types.Part.from_bytes(
                        data=fh.read(), mime_type="audio/wav")
                # 200 tokens truncated thinking-model JSON mid-string
                # ("Unterminated string"), failing EVERY reading; 2048
                # gives headroom, and one retry rides out per-minute-429
                # blips + transient truncations (content_type.py pattern).
                t = None
                for _at in range(2):
                    try:
                        resp = client.models.generate_content(
                            model=model,
                            contents=[audio,
                                      "Listen to the SOUND of this speech (tone of "
                                      "voice, pace, tension), not just the words"
                                      + (f" (language: {language})" if language else "")
                                      + ". Report ONLY JSON: {\"tone\": one of "
                                      + str(list(_TONES))
                                      + ", \"arousal\": 0..1 (energy), \"valence\": "
                                        "-1..1 (negative..positive), \"note\": "
                                        "<=12 words on what you heard}."],
                            config=genai_types.GenerateContentConfig(
                                response_mime_type="application/json",
                                temperature=0.2, max_output_tokens=2048),
                        )
                        t = _san_tone(_loads_lenient((resp.text or "").strip()))
                        if t or _at:      # got a reading, or retry spent
                            break
                    except Exception as _tx:
                        if _at == 0:
                            if "429" in str(_tx):
                                import time as _time
                                _time.sleep(30)
                            continue
                        raise
                if t:
                    out[idx] = t
            except Exception as exc:
                print(f"[v4/director] tone story {idx} failed (soft): {exc}",
                      flush=True)
            finally:
                try:
                    os.remove(wav)
                except OSError:
                    pass
        if out:
            print(f"[v4/director] tone sensed on {len(out)} stories: "
                  + ", ".join(f"{i}:{t['tone']}" for i, t in sorted(out.items())),
                  flush=True)
    except Exception as exc:
        print(f"[v4/director] tone sensing unavailable (soft): {exc}",
              flush=True)
    return out


def sense_ser(wav_by_story: dict[int, str]) -> dict[int, dict]:
    """OPTIONAL second acoustic signal: a local speech-emotion classifier
    (transformers pipeline) — gated KAIZER_V4_SER=1 and only if the
    `transformers` stack is installed. {} otherwise; never raises."""
    if (os.environ.get("KAIZER_V4_SER") or "").strip() not in ("1", "true", "on"):
        return {}
    try:
        from transformers import pipeline as _hf_pipeline
        clf = _hf_pipeline("audio-classification",
                           model="superb/wav2vec2-base-superb-er")
        out = {}
        for idx, wav in (wav_by_story or {}).items():
            try:
                top = clf(wav, top_k=1)
                if top:
                    out[int(idx)] = {"label": str(top[0]["label"]).lower(),
                                     "score": round(float(top[0]["score"]), 2)}
            except Exception:
                continue
        return out
    except Exception as exc:
        print(f"[v4/director] SER unavailable (soft): {exc}", flush=True)
        return {}


def sense_pacing(words_by_story: dict, durs: dict[int, float]) -> dict[int, dict]:
    """Speech-rhythm facts from the word timestamps the pipeline already
    has (zero extra cost): words-per-minute, long pauses, the fastest
    stretch, and "hot moments" — the word right after a long pause and
    the start of the fastest run, i.e. where a human editor would cut or
    punch. Pure Python, never raises, {} on no data."""
    out: dict[int, dict] = {}
    try:
        for key, words in (words_by_story or {}).items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            dur = float(durs.get(idx, 0.0) or 0.0)
            ts = []
            for w in (words or []):
                try:
                    ts.append(float(w.get("s", 0.0)))
                except (TypeError, ValueError, AttributeError):
                    continue
            if len(ts) < 8 or dur < 5.0:
                continue
            ts.sort()
            wpm = round(len(ts) / dur * 60.0)
            pauses = []
            hot = []
            for a, b in zip(ts, ts[1:]):
                gap = b - a
                if gap >= 1.1 and len(pauses) < 6:
                    pauses.append([round(a, 2), round(gap, 2)])
                    if 1.0 <= b <= dur - 2.0:
                        hot.append(round(b, 2))
            # fastest stretch: densest 8-second window of word starts
            best_n, best_t = 0, 0.0
            j = 0
            for i, t0 in enumerate(ts):
                while ts[j] < t0 - 8.0:
                    j += 1
                if i - j + 1 > best_n:
                    best_n, best_t = i - j + 1, ts[j]
            if best_n >= 10 and 0.5 <= best_t <= dur - 3.0:
                hot.append(round(best_t, 2))
            out[idx] = {
                "wpm": wpm,
                "pauses": pauses,
                "hot_moments": sorted(set(hot))[:6],
            }
    except Exception:
        return {}
    return out


def sense_frames(source_path: str, stories, *, work_dir: str = "",
                 language: str = "", cap: int = 8) -> dict[int, dict]:
    """EYES: sample 2 frames per story and let Gemini DESCRIBE what is on
    screen (shot type, people count, setting, motion, on-screen text) in
    ONE batched call — so layout/grade decisions are grounded in the real
    picture, not just the words. Gated KAIZER_V4_VISION_SENSE (default
    ON). {} on any failure — the Director then decides from text alone."""
    if (os.environ.get("KAIZER_V4_VISION_SENSE") or "1").strip().lower() \
            in ("0", "false", "off", "no"):
        return {}
    if not (source_path and os.path.isfile(source_path)):
        return {}
    try:
        import tempfile
        from seo.generator import _gemini_client
        from google.genai import types as genai_types
        from pipeline_v4.trim_engine import _loads_lenient
        wdir = work_dir or tempfile.mkdtemp(prefix="kx_vision_")
        os.makedirs(wdir, exist_ok=True)
        ffmpeg = os.environ.get("KAIZER_FFMPEG_BIN", "ffmpeg")
        parts: list = []
        grabbed: list[int] = []
        for s in (stories or [])[:cap]:
            idx = int(getattr(s, "story_index", 0) or 0)
            try:
                t0 = float(getattr(s, "video_t_start", 0.0) or 0.0)
                t1 = float(getattr(s, "video_t_end", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            span = t1 - t0
            if span < 2.0:
                continue
            story_parts: list = []
            for k, frac in ((0, 0.25), (1, 0.7)):
                tt = t0 + span * frac
                fp = os.path.join(wdir, f"_vis_{idx:02d}_{k}.jpg")
                try:
                    if not (os.path.isfile(fp) and os.path.getsize(fp) > 0):
                        r = subprocess.run(
                            [ffmpeg, "-y", "-v", "error", "-ss", f"{tt:.2f}",
                             "-i", source_path, "-frames:v", "1",
                             "-vf", "scale=640:-2", "-q:v", "5", fp],
                            capture_output=True, timeout=60)
                        if r.returncode != 0:
                            continue
                    with open(fp, "rb") as fh:
                        story_parts.append(genai_types.Part.from_bytes(
                            data=fh.read(), mime_type="image/jpeg"))
                except Exception:
                    continue
            if story_parts:
                parts.append(f"story {idx} frames:")
                parts.extend(story_parts)
                grabbed.append(idx)
        if not parts:
            return {}
        parts.append(
            "These are sampled frames from a news bulletin's stories"
            + (f" (language: {language})" if language else "")
            + ". For EACH story report what is ON SCREEN. ONLY JSON: "
              '{"stories": [{"index": 0, "shot": "one of '
              'closeup|medium|wide|two_shot|group|graphic|scenery", '
              '"people": <int>, "setting": "<=6 words", '
              '"motion": "static|moderate|busy", '
              '"dark": true/false, "on_screen_text": true/false}]}')
        client = _gemini_client()
        model = os.environ.get("KAIZER_V4_DIRECTOR_MODEL", "gemini-2.5-flash")
        data = None
        for _at in range(2):
            try:
                resp = client.models.generate_content(
                    model=model, contents=parts,
                    config=genai_types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2, max_output_tokens=2048),
                )
                data = _loads_lenient((resp.text or "").strip())
                break
            except Exception as _vx:
                if _at == 0 and "429" in str(_vx):
                    import time as _time
                    _time.sleep(30)
                    continue
                raise
        out: dict[int, dict] = {}
        _shots = {"closeup", "medium", "wide", "two_shot", "group",
                  "graphic", "scenery"}
        for row in (data or {}).get("stories") or []:
            if not isinstance(row, dict):
                continue
            try:
                idx = int(row.get("index"))
            except (TypeError, ValueError):
                continue
            if idx not in grabbed:
                continue
            shot = str(row.get("shot", "") or "").strip().lower()
            try:
                people = max(0, min(20, int(row.get("people", 0) or 0)))
            except (TypeError, ValueError):
                people = 0
            out[idx] = {
                "shot": shot if shot in _shots else "medium",
                "people": people,
                "setting": str(row.get("setting", "") or "")[:60],
                "motion": str(row.get("motion", "") or "")[:10],
                "dark": bool(row.get("dark", False)),
                "on_screen_text": bool(row.get("on_screen_text", False)),
            }
        if out:
            print(f"[v4/director] vision sensed {len(out)} stories: "
                  + ", ".join(f"{i}:{v['shot']}/{v['people']}p"
                              for i, v in sorted(out.items())), flush=True)
        return out
    except Exception as exc:
        print(f"[v4/director] vision sensing unavailable (soft): {exc}",
              flush=True)
        return {}


# ── Layer 3: the LLM decision plan + sanitizer ───────────────────────

_DIRECTOR_SYSTEM = """You are the senior editor of a TV channel finishing a multi-story bulletin. For EVERY story you decide the treatment, like a human editor in a grading suite.

You receive: per-story title/summary/duration + word timestamps, sensor facts (scene cuts, loud moments), and the EXACT vocabulary of ids you may use (packs, transitions, frame fx, overlays, caption styles, color grades, stings, ui sounds). NEVER invent an id.

Decide per story:
- "mood": pack id — the story's OWN mood (a funny story inside a news bulletin gets comedy). Stories can differ; that's the point.
- "transition_in": transition id for the cut INTO this story.
- "fx": 0-2 SUBTLE frame-fx ids (garnish, not noise; duration-neutral only — never slow_motion/speed_up/freeze).
- "overlays": 0-3 of [{"id": overlay-id, "t": seconds-into-story}] — a banner/tag/chip only where it genuinely informs.
- "captions": karaoke caption variant id, or "none" (news bulletins usually "none"; shorts-style content benefits).
- "emphasis": the 1-2 word-timestamps (seconds into the story) whose WORDS deserve a visual punch — the name, the number, the verdict (drives micro-zoom/strap timing). Pick the words a human editor would hit, never random beats; omit when nothing earns it.
- "grade": color-grade id ONLY when this story deserves a DIFFERENT look than its mood pack already gives (a memorial story inside a pop bulletin → monochrome); omit to keep the pack's own grade.
- "sting": opening sting id played on the cut INTO this story (omit for the mood default).
- "ui_sound": ui tick id played under overlay/graphic pop-ins (omit for none).
- "bed_on": false to SILENCE the music bed on solemn/tragic stories (default true).
- "layout": layout id — this story's SCREEN ARRANGEMENT, like a director cutting the studio. Match the content (talking-head → fullscreen anchor or over-the-shoulder; picture-rich → a bulletin/split with the image rail; big visual story → an inset PiP look) and VARY across stories so the bulletin feels alive. Use image-rail/inset layouts ONLY for stories that have images (each story line shows its image count). "" = the channel's default.
- "layout_moments": [{"layout": video-only layout id, "t": seconds-into-story, "dur": seconds}] — MID-STORY switches (pull the video fullscreen for a dramatic stretch, pull back to a floating card, then return). Direct a presentation ARC: on stories over ~60s plan a switch roughly every 20-40s at the moments the content turns (a 120s story deserves 3-5). When the bulletin has ONE story, these moments are your ONLY direction tool — never leave a long story static. Each moment >=3s, inside the story, never in the first/last second. Short stories (<20s): usually [].
- "why": <=25 words explaining THIS story's treatment — which sensor fact or content point drove the mood/layout/emphasis choices (shown to the human editor reviewing your work).

DIRECT THE WHOLE SHOW, not isolated stories:
- ARC: the bulletin is ONE programme. Open strong (bold mood + sting), vary the middle so no two adjacent stories feel identical, and give the final story a closing punch. Never use the same transition_in twice in a row; never let three consecutive stories share a mood unless the content truly demands it.
- PACING (sensor "pacing_per_story"): "hot_moments" are the natural hit-points — the word after a long pause, the fastest-spoken stretch. Put emphasis there and start layout_moments there, not at arbitrary times. High wpm = cut faster (shorter moments, snappier transitions); slow, deliberate speech = let shots breathe.
- EYES (sensor "visual_per_story"): you can SEE each story's frames. Match the screen: a two-person interview → fullscreen/anchor looks, never side panels over people; a busy action frame → calmer graphics; a dark/dim scene → a lifting or moody grade that fits; text already on screen → fewer overlays.
- EARS (sensor "voice_tone_per_story"): arousal >= 0.7 → punchy treatment (use the hot moments for emphasis, energetic transition, bolder grade). Negative valence + low arousal → solemn: bed_on false, calm grade, gentle transition, no comedy mood.
- Any story >= 45s MUST carry at least one layout_moment — a long static frame is a dead broadcast.

OUTPUT one JSON object, no prose:
{"stories": [{"index": 0, "mood": "crime", "transition_in": "glitch_slices",
  "fx": ["vignette_soft"], "overlays": [{"id": "tag_investigation", "t": 1.5}],
  "captions": "none", "emphasis": [12.4], "layout": "news_ots_right",
  "layout_moments": [{"layout": "news_fullscreen_anchor", "t": 8.0, "dur": 5.0}],
  "grade": "bleach_bypass", "sting": "sting_negative", "ui_sound": "ui_tick",
  "bed_on": true,
  "why": "crime story, tense voice at 12s pause; investigation tag + punch on the verdict word"}]}"""


@dataclass
class StoryDirective:
    mood: str = ""
    transition_in: str = ""
    fx: list = field(default_factory=list)
    overlays: list = field(default_factory=list)     # [{"id","t"}]
    captions: str = "none"
    emphasis: list = field(default_factory=list)
    # LIVE LAYOUTS: this story's screen arrangement ("" = job default)
    # + mid-story layout switches [{"layout","t","dur","transition"}].
    layout: str = ""
    layout_moments: list = field(default_factory=list)
    # LOOK + SOUND vocabulary (rendered by the compose side; "" = keep
    # the mood pack's own default, so absent fields are byte-compatible):
    grade: str = ""       # color-grade id overriding the pack's grade
    sting: str = ""       # opening sting id ("" = the mood default)
    ui_sound: str = ""    # ui tick id under overlay/graphic pop-ins
    bed_on: bool = True   # False silences the music bed (solemn stories)
    # The Director's own one-line rationale (free text, never rendered —
    # surfaced in the Director-decisions debug view).
    why: str = ""


def sanitize_plan(raw: dict, *, stories, category: str,
                  user_picks: dict | None = None) -> dict[int, StoryDirective]:
    """Validate every id against the REAL registries, clamp every time,
    cap counts; anything invalid falls back to the formula. Pure-ish
    (reads registries), never raises. ``user_picks`` constrains the
    allowed ids per-category (see ``_vocab``)."""
    v = _vocab(user_picks)
    base = formula_for(category)
    up = user_picks or {}

    # When the user PINNED a single-value category, an out-of-vocab LLM choice
    # must snap to THEIR pick — not the generic formula default (otherwise a
    # stray LLM id silently ignores the user's direction).
    def _fb(cat, formula_default):
        for x in (up.get(cat) or []):
            if x in v[cat]:
                return x
        return formula_default

    _fb_pack = _fb("packs", base.pack)
    _fb_trans = _fb("transitions", base.transition)
    _fb_cap = _fb("captions", "none")
    # Look/sound formula default is "" (= the mood pack's own choice), so
    # these only bite when the USER pinned a grade/sting/ui sound.
    _fb_grade = _fb("grades", "")
    _fb_sting = _fb("stings", "")
    _fb_ui = _fb("ui_sounds", "")
    out: dict[int, StoryDirective] = {}
    durs: dict[int, float] = {}
    has_imgs: dict[int, bool] = {}
    for s in (stories or []):
        idx = int(getattr(s, "story_index", 0) or 0)
        try:
            durs[idx] = max(0.5, float(getattr(s, "video_t_end", 0.0))
                            - float(getattr(s, "video_t_start", 0.0)))
        except (TypeError, ValueError):
            durs[idx] = 30.0
        has_imgs[idx] = bool(getattr(s, "images", None))

    def _layout_ok(key: str, idx: int) -> bool:
        """A layout id is usable for THIS story: renderable, and its
        picture surfaces / image wash have images to show."""
        if key not in v["layouts"]:
            return False
        try:
            from pipeline_v4.layout_library import story_geometry
            g = story_geometry(key)
        except Exception:
            return False
        if g is None:
            return False
        if not has_imgs.get(idx) and (g.picture or g.pips
                                      or g.bg == "bg_image"):
            return False
        return True

    def _moment_ok(key: str) -> bool:
        """Moment layouts are VIDEO-centric (no picture surface)."""
        if key not in v["layouts"]:
            return False
        try:
            from pipeline_v4.layout_library import story_geometry
            g = story_geometry(key)
        except Exception:
            return False
        return bool(g is not None and not g.picture and not g.pips)
    banned_fx = {"slow_motion", "speed_up", "freeze_intro", "stutter_frames",
                 "mirror", "kaleidoscope_quad"}   # duration/size changers
    for row in (raw or {}).get("stories") or []:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("index"))
        except (TypeError, ValueError):
            continue
        if idx not in durs:
            continue
        d = StoryDirective()
        mood = str(row.get("mood", "") or "").strip().lower()
        d.mood = mood if mood in v["packs"] else _fb_pack
        tr = str(row.get("transition_in", "") or "").strip().lower()
        d.transition_in = tr if tr in v["transitions"] else _fb_trans
        d.fx = [f for f in (row.get("fx") or [])
                if isinstance(f, str) and f in v["fx"]
                and f not in banned_fx][:2]
        ovl = []
        for o in (row.get("overlays") or [])[:3]:
            if not isinstance(o, dict):
                continue
            oid = str(o.get("id", "") or "")
            if oid not in v["overlays"]:
                continue
            try:
                t = max(0.0, min(float(o.get("t", 0.0) or 0.0),
                                 durs[idx] - 2.0))
            except (TypeError, ValueError):
                t = 0.5
            ovl.append({"id": oid, "t": round(t, 2)})
        d.overlays = ovl
        cap = str(row.get("captions", "none") or "none").strip().lower()
        d.captions = cap if cap in v["captions"] else _fb_cap
        # LOOK + SOUND: every id must exist in its registry bucket — a
        # hallucinated grade/sting can never reach ffmpeg; invalid falls
        # back to the user pin (if any) else "" (the pack's own default).
        gr = str(row.get("grade", "") or "").strip().lower()
        d.grade = gr if gr in v["grades"] else _fb_grade
        st = str(row.get("sting", "") or "").strip().lower()
        d.sting = st if st in v["stings"] else _fb_sting
        ui = str(row.get("ui_sound", "") or "").strip().lower()
        d.ui_sound = ui if ui in v["ui_sounds"] else _fb_ui
        # bed_on must be a real JSON boolean; anything else keeps the bed
        # (silencing music on a paid render needs an explicit decision).
        _bed = row.get("bed_on", True)
        d.bed_on = _bed if isinstance(_bed, bool) else True
        # Free-text rationale for the decisions view — never rendered.
        d.why = str(row.get("why", "") or "")[:220]
        emph = []
        for e in (row.get("emphasis") or [])[:2]:
            try:
                emph.append(round(max(0.0, min(float(e), durs[idx])), 2))
            except (TypeError, ValueError):
                continue
        d.emphasis = emph
        # LIVE LAYOUTS: the story's screen arrangement — invalid or
        # asset-starved picks fall back to "" (the job default), never
        # to a blank surface.
        lay = str(row.get("layout", "") or "").strip().lower()
        d.layout = lay if _layout_ok(lay, idx) else ""
        moms = []
        # Moment budget: ~one per 45s, capped at 3 (env KAIZER_V4_MAX_MOMENTS).
        # Each non-fullscreen moment adds TWO full-canvas branches (tile +
        # blurred echo) to the story's filtergraph; 5-8 of them made the
        # heaviest stories OOM even composing solo (job 606 story 12). Three
        # well-placed switches keep it dynamic and premium — a real director
        # uses restraint, not a cut every 20s — while every story renders
        # reliably. Dial up on a bigger-RAM box.
        try:
            _mmax = max(1, int(os.environ.get("KAIZER_V4_MAX_MOMENTS", "3") or "3"))
        except ValueError:
            _mmax = 3
        _mcap = max(1, min(_mmax, int(durs[idx] // 45) + 1))
        if durs[idx] >= 8.0:      # a mid-story switch needs room to breathe
            for m in (row.get("layout_moments") or [])[:_mcap]:
                if not isinstance(m, dict):
                    continue
                mk = str(m.get("layout", "") or "").strip().lower()
                if not _moment_ok(mk):
                    continue
                try:
                    mt = max(0.0, float(m.get("t", 0.0) or 0.0))
                    md = float(m.get("dur", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if md < 2.5:
                    continue
                moms.append({"layout": mk, "t": round(mt, 2),
                             "dur": round(md, 2), "transition": "push"})
        d.layout_moments = moms
        out[idx] = d
    # stories the model skipped get the formula (+ the user's pinned
    # look/sound picks, so a pin still applies to a skipped story)
    for idx in durs:
        if idx not in out:
            out[idx] = StoryDirective(mood=base.pack,
                                      transition_in=base.transition,
                                      fx=list(base.fx),
                                      overlays=[{"id": o, "t": 0.8}
                                                for o in base.overlays],
                                      captions=base.captions,
                                      grade=_fb_grade,
                                      sting=_fb_sting,
                                      ui_sound=_fb_ui)
    return out


def formula_plan(stories, category: str,
                 user_picks: dict | None = None) -> dict[int, StoryDirective]:
    """Layer-2-only plan (no LLM): every story gets the category recipe,
    overridden by any explicit ``user_picks`` — the deterministic
    'edit using THESE' path (used when the LLM is off/failed or the user
    pinned choices). Each pick is validated against the real registry."""
    base = formula_for(category)
    up = user_picks or {}
    v = _vocab(user_picks)

    def _one(cat, fallback):
        for x in (up.get(cat) or []):
            if x in v[cat]:
                return x
        return fallback

    _pack = _one("packs", base.pack)
    _trans = _one("transitions", base.transition)
    _cap = _one("captions", base.captions)
    # Look/sound defaults are "" / bed on (= each pack's own choice);
    # only an explicit user pin changes the deterministic baseline.
    _grade = _one("grades", "")
    _sting = _one("stings", "")
    _ui = _one("ui_sounds", "")
    _fx = [x for x in (up.get("fx") or []) if x in v["fx"]] or list(base.fx)
    _ovl = [x for x in (up.get("overlays") or []) if x in v["overlays"]] or list(base.overlays)
    # LIVE LAYOUTS baseline: a deterministic screen rotation so even the
    # no-LLM path varies like a real broadcast ("" = the default look).
    _rot_rich = ["", "news_ots_right", "news_bulletin_left",
                 "news_fullscreen_anchor"]
    _rot_lean = ["", "news_fullscreen_anchor"]
    _user_lay = [x for x in (up.get("layouts") or []) if x in v["layouts"]]
    out = {}
    for s in (stories or []):
        idx = int(getattr(s, "story_index", 0) or 0)
        _rot = (_rot_rich if getattr(s, "images", None) else _rot_lean)
        _lay = (_user_lay[idx % len(_user_lay)] if _user_lay
                else _rot[idx % len(_rot)])
        if _lay and _lay not in v["layouts"]:
            _lay = ""
        # Deterministic presentation arc for LONG stories (the no-LLM
        # baseline must move too): a switch every ~30s, alternating the
        # fullscreen pull-in and the card-over-echo pull-back.
        _moms = []
        try:
            _dur = float(getattr(s, "video_t_end", 0.0)) - float(
                getattr(s, "video_t_start", 0.0))
        except (TypeError, ValueError):
            _dur = 0.0
        if _dur >= 50.0:
            _alt = ["news_fullscreen_anchor", "studio_bg_center"]
            # Same cap as the LLM path (KAIZER_V4_MAX_MOMENTS, default 3) —
            # keeps the no-LLM baseline's graphs within the memory budget.
            try:
                _fmax = max(1, int(os.environ.get("KAIZER_V4_MAX_MOMENTS", "3") or "3"))
            except ValueError:
                _fmax = 3
            _t, _k = 22.0, 0
            while _t + 8.0 < _dur - 2.0 and len(_moms) < _fmax:
                _moms.append({"layout": _alt[_k % 2], "t": round(_t, 2),
                              "dur": 6.0, "transition": "push"})
                _t += 40.0
                _k += 1
        out[idx] = StoryDirective(mood=_pack,
                                  transition_in=_trans,
                                  fx=list(_fx),
                                  overlays=[{"id": o, "t": 0.8} for o in _ovl],
                                  captions=_cap,
                                  layout=_lay,
                                  layout_moments=_moms,
                                  grade=_grade,
                                  sting=_sting,
                                  ui_sound=_ui,
                                  bed_on=True)
    return out


def _directive_dict(d: StoryDirective) -> dict:
    """Full JSON shape of one directive — the trace/debug contract."""
    return {"mood": d.mood, "transition_in": d.transition_in, "fx": d.fx,
            "overlays": d.overlays, "captions": d.captions,
            "emphasis": d.emphasis, "layout": d.layout,
            "layout_moments": d.layout_moments, "grade": d.grade,
            "sting": d.sting, "ui_sound": d.ui_sound, "bed_on": d.bed_on,
            "why": d.why}


def _write_trace(source_path: str, trace: dict) -> None:
    """Persist the Director's full decision trail next to the job's media
    (``director_trace.json``) — the Director-decisions view in the job UI
    reads it, so the operator can see every sensor fact, every choice and
    the model's own WHY. Fail-soft: tracing may never hurt a render."""
    try:
        if not source_path:
            return
        import datetime as _dt
        trace["written_at"] = _dt.datetime.now(
            _dt.timezone.utc).isoformat(timespec="seconds")
        p = os.path.join(os.path.dirname(source_path), "director_trace.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(trace, fh, ensure_ascii=False, indent=1)
    except Exception as exc:
        print(f"[v4/director] trace write soft-fail: {exc}", flush=True)


def _enforce_variety(plan: dict[int, StoryDirective], vocab) -> int:
    """Deterministic anti-monotony pass: no two ADJACENT stories may share
    a transition_in (a human editor never cuts the same way twice in a
    row). Returns how many transitions were rotated. Mood is left to the
    model — content legitimately repeats moods; transitions never need to."""
    if len(plan) < 2 or len(vocab.get("transitions") or ()) < 2:
        return 0
    ordered = sorted(plan)
    options = sorted(vocab["transitions"])
    fixed = 0
    for prev_i, cur_i in zip(ordered, ordered[1:]):
        prev_t = plan[prev_i].transition_in
        if plan[cur_i].transition_in == prev_t and prev_t:
            k = options.index(prev_t) if prev_t in options else 0
            plan[cur_i].transition_in = options[(k + 1) % len(options)]
            fixed += 1
    return fixed


def _plan_quality_issues(plan: dict[int, StoryDirective],
                         durs: dict[int, float],
                         words_by_story: dict | None = None) -> list[str]:
    """Heuristic review of a sanitized plan — the concrete complaints a
    supervising editor would raise. Pure, never raises."""
    issues: list[str] = []
    try:
        for idx in sorted(plan):
            d = plan[idx]
            dur = float(durs.get(idx, 0.0) or 0.0)
            if dur >= 45.0 and not d.layout_moments:
                issues.append(
                    f"story {idx} runs {dur:.0f}s with ZERO layout_moments — "
                    f"a static frame that long is a dead broadcast")
        moods = [plan[i].mood for i in sorted(plan)]
        if len(moods) >= 3 and len(set(moods)) == 1:
            issues.append("every story shares one mood — the show is monotone")
        n_emph = sum(len(plan[i].emphasis) for i in plan)
        has_words = any((words_by_story or {}).get(str(i)) for i in plan)
        if has_words and len(plan) >= 2 and n_emph == 0:
            issues.append("no emphasis anywhere — nothing is punched, "
                          "nothing lands")
        n_ovl = sum(len(plan[i].overlays) for i in plan)
        if len(plan) >= 3 and n_ovl == 0:
            issues.append("no overlays anywhere — the bulletin has no "
                          "broadcast graphics at all")
    except Exception:
        return issues
    return issues


def _plan_token_budget(n_stories) -> int:
    """Output-token ceiling for a whole-bulletin plan. A fixed 8192 silently
    TRUNCATED large bulletins (25+ stories) mid-JSON: the tail stories never
    reached the model's output, so parse/repair recovered only a short plan
    and every missing story fell back to the undirected formula baseline — a
    "thin plan" on exactly the big jobs that most need direction. Each story's
    directive (mood/look/sound + up to ~5 layout_moments + overlays +
    emphasis) costs ~350-500 output tokens, so scale headroom with the story
    count. Small bulletins (<=10 stories) keep the original 8192 unchanged, so
    typical jobs render byte-for-byte as before. Capped below gemini-2.5-
    flash's 65536 ceiling. Override with KAIZER_V4_DIRECTOR_MAX_TOKENS."""
    env = os.environ.get("KAIZER_V4_DIRECTOR_MAX_TOKENS")
    if env:
        try:
            return max(2048, int(env))
        except (TypeError, ValueError):
            pass
    try:
        n = max(1, int(n_stories))
    except (TypeError, ValueError):
        n = 1
    return max(8192, min(60000, n * 500 + 3072))


def _review_plan(*, client, genai_types, loads_lenient, model: str,
                 user_prompt: str, plan: dict[int, StoryDirective],
                 issues: list[str], stories, category: str,
                 user_picks, durs, words_by_story) -> dict[int, StoryDirective]:
    """SECOND PASS: show the model its own sanitized plan plus the
    reviewer's concrete complaints; take the revision only if it actually
    reviews better. One extra call, gated by the caller. Never raises."""
    try:
        prev = {"stories": [
            {"index": i, **_directive_dict(d)}
            for i, d in sorted(plan.items())]}
        revise = (
            user_prompt
            + "\n\nYOUR PREVIOUS PLAN:\n" + json.dumps(prev, ensure_ascii=False)
            + "\n\nA supervising editor reviewed it and found:\n- "
            + "\n- ".join(issues)
            + "\n\nReturn the SAME JSON shape with these problems FIXED. "
              "Keep every decision that wasn't criticised.")
        resp = None
        for _at in range(2):
            try:
                resp = client.models.generate_content(
                    model=model, contents=revise,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=_DIRECTOR_SYSTEM,
                        response_mime_type="application/json",
                        temperature=0.3,
                        max_output_tokens=_plan_token_budget(len(durs))),
                )
                break
            except Exception as _rx:
                if _at == 0 and "429" in str(_rx):
                    import time as _time
                    _time.sleep(30)
                    continue
                raise
        revised = sanitize_plan(loads_lenient((resp.text or "").strip()),
                                stories=stories, category=category,
                                user_picks=user_picks)
        if not revised:
            return plan, False
        before = len(issues)
        after = len(_plan_quality_issues(revised, durs, words_by_story))
        if after < before:
            print(f"[v4/director] self-review: {before} issue(s) → {after} "
                  f"after revision — using the revised plan", flush=True)
            return revised, True
        print(f"[v4/director] self-review: revision did not improve "
              f"({before} → {after}) — keeping the first plan", flush=True)
        return plan, False
    except Exception as exc:
        print(f"[v4/director] self-review failed (soft): {exc}", flush=True)
        return plan, False


def plan_direction(*, stories, words_by_story: dict, category: str,
                   source_path: str = "", language: str = "",
                   user_picks: dict | None = None) -> dict[int, StoryDirective]:
    """The full Director: sensors → LLM plan → sanitize. Any failure →
    formula plan. Never raises, never returns None. ``user_picks`` (the
    operator's "edit using THESE" selections) constrains the vocabulary
    per-category; empty categories are left to the AI."""
    try:
        durs: dict[int, float] = {}
        total_dur = 0.0
        for s in (stories or []):
            _i = int(getattr(s, "story_index", 0) or 0)
            try:
                _d = max(0.0, float(getattr(s, "video_t_end", 0.0))
                         - float(getattr(s, "video_t_start", 0.0)))
            except (TypeError, ValueError):
                _d = 0.0
            durs[_i] = _d
            total_dur += _d
        # DECISION TRACE: everything the Director saw and chose, persisted
        # as director_trace.json for the job UI's Director-decisions view.
        _trace: dict = {
            "mode": "llm",
            "category": category or "news",
            "language": language or "",
            "stories": {
                str(int(getattr(s, "story_index", 0) or 0)): {
                    "duration_s": round(durs.get(
                        int(getattr(s, "story_index", 0) or 0), 0.0), 1),
                    "title": (getattr(s, "title_native", "")
                              or getattr(s, "title_english", "") or "")[:120],
                } for s in (stories or [])
            },
            "user_picks": {k: sorted(vv) for k, vv in (user_picks or {}).items()},
            "steps": [],
        }
        sensors = {}
        # RHYTHM: free facts from the word timestamps already in hand.
        pacing = sense_pacing(words_by_story or {}, durs)
        if pacing:
            sensors["pacing_per_story"] = {str(i): p for i, p in pacing.items()}
        if source_path and os.path.isfile(source_path):
            # The four media sensors are INDEPENDENT (two local ffmpeg
            # passes, two Gemini round-trips), so run them concurrently —
            # sequential they cost sum(), parallel they cost max(). On a
            # long source this turns minutes of Stage-3 lead-in into the
            # cost of the slowest single sensor. Results and their order
            # in the prompt are identical to the sequential form.
            from concurrent.futures import ThreadPoolExecutor as _SensePool
            with _SensePool(max_workers=4,
                            thread_name_prefix="v4-sense") as _sp:
                _f_cuts = _sp.submit(sense_scene_cuts, source_path)
                _f_energy = _sp.submit(sense_energy_peaks, source_path)
                # EYES: Gemini describes sampled frames per story so the
                # layout/grade choices match what is ON SCREEN.
                _f_vision = _sp.submit(sense_frames, source_path, stories,
                                       language=language)
                # EARS: per-story acoustic tone (Gemini listens to the
                # audio). Words can hide fear; the voice doesn't.
                _f_tone = _sp.submit(sense_tone, source_path, stories,
                                     language=language)
                sensors["scene_cuts"] = _f_cuts.result()[:100]
                sensors["energy"] = _f_energy.result()
                vision = _f_vision.result()
                tone = _f_tone.result()
            if vision:
                sensors["visual_per_story"] = {
                    str(i): f for i, f in vision.items()}
            if tone:
                sensors["voice_tone_per_story"] = {
                    str(i): t for i, t in tone.items()}
        _trace["sensors"] = sensors
        _trace["steps"].append(
            f"sensors gathered: pacing {len(pacing)} stories, "
            f"scene_cuts {len(sensors.get('scene_cuts') or [])}, "
            f"vision {len(sensors.get('visual_per_story') or {})} stories, "
            f"tone {len(sensors.get('voice_tone_per_story') or {})} stories")
        v = _vocab(user_picks)
        lines = []
        for s in (stories or []):
            idx = int(getattr(s, "story_index", 0) or 0)
            dur = float(getattr(s, "video_t_end", 0.0)) - float(
                getattr(s, "video_t_start", 0.0))
            _n_imgs = len(getattr(s, "images", None) or [])
            lines.append(f"--- story {idx} ({dur:.1f}s, {_n_imgs} images): "
                         f"{getattr(s, 'title_native', '') or ''} | "
                         f"{(getattr(s, 'summary', '') or '')[:220]}")
            sw = (words_by_story or {}).get(str(idx)) or []
            if sw:
                lines.append("words: " + " ".join(
                    f"[{float(w.get('s', 0)):.1f}]{w.get('w', '')}"
                    for w in sw[:80]))
        user = (
            f"Bulletin category: {category or 'news'}. Language: {language or '?'}.\n"
            f"The show: {len(durs)} stories, {total_dur:.0f}s total — direct it "
            f"as ONE programme with an arc, not isolated clips.\n"
            f"Sensors: {json.dumps(sensors, ensure_ascii=False)[:6000]}\n"
            f"VOCABULARY (use ONLY these ids):\n"
            f"packs: {sorted(v['packs'])}\n"
            f"transitions: {sorted(v['transitions'])}\n"
            f"fx: {sorted(v['fx'])}\n"
            f"overlays: {sorted(v['overlays'])[:120]}\n"
            f"captions: {sorted(v['captions'])}\n"
            f"layouts: {sorted(v['layouts'])}\n"
            f"grades: {sorted(v['grades'])}\n"
            f"stings: {sorted(v['stings'])}\n"
            f"ui_sounds: {sorted(v['ui_sounds'])}\n"
            + "\n".join(lines))
        from seo.generator import _gemini_client
        from google.genai import types as genai_types
        from pipeline_v4.trim_engine import _gemini_repair_json, _loads_lenient
        model = os.environ.get("KAIZER_V4_DIRECTOR_MODEL", "gemini-2.5-flash")
        # Keep an explicit reference to the client for the whole call. In
        # google-genai 2.4.0 an INLINE `_gemini_client().models.generate_
        # content(...)` lets the temporary Client get GC'd mid-request,
        # closing its httpx transport → "Cannot send a request, as the
        # client has been closed." This silently sent every Director plan
        # to the formula fallback. Binding it to a name fixes it.
        _client = _gemini_client()
        # Vertex 429 = PER-MINUTE quota (the parallel SEO burst right
        # before Stage 3 routinely drains it, which silently sent every
        # plan to the formula fallback — no content-aware layouts/looks).
        # Waiting into the next quota window almost always succeeds.
        _tries = max(1, int(os.environ.get(
            "KAIZER_V4_DIRECTOR_RETRIES", "3") or 3))
        resp = None
        for _at in range(_tries):
            try:
                resp = _client.models.generate_content(
                    model=model, contents=user,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=_DIRECTOR_SYSTEM,
                        response_mime_type="application/json",
                        # Scale the output ceiling with the story count. A
                        # fixed 8192 overflowed 4096 (why it was raised) but
                        # still TRUNCATED 25+-story bulletins mid-JSON → the
                        # tail stories dropped to the formula baseline.
                        temperature=0.35,
                        max_output_tokens=_plan_token_budget(len(durs)),
                    ),
                )
                break
            except Exception as _rexc:
                if "429" in str(_rexc) and _at < _tries - 1:
                    _wait = 30 * (_at + 1)
                    print(f"[v4/director] 429 per-minute quota — retry "
                          f"{_at + 2}/{_tries} in {_wait}s", flush=True)
                    import time as _time
                    _time.sleep(_wait)
                    continue
                raise
        _raw_plan = (resp.text or "").strip()
        try:
            data = _loads_lenient(_raw_plan)
        except Exception as _pe:
            # The deterministic tiers (incl. position-guided comma repair)
            # didn't cover it — ask the model to fix its own JSON ONCE, then
            # parse again (the trim planner's proven job#500 recovery). Only
            # if THAT also fails do we drop to the formula baseline.
            print(f"[v4/director] plan JSON malformed ({_pe}) — model self-repair",
                  flush=True)
            _fixed = _gemini_repair_json(_client, genai_types, model, _raw_plan)
            data = _loads_lenient(_fixed)
            _trace["steps"].append(
                "plan JSON was malformed → repaired on a second model call")
        _trace["model"] = model
        _trace["steps"].append(
            f"model {model} returned "
            f"{len((data or {}).get('stories') or [])} story rows")
        plan = sanitize_plan(data, stories=stories, category=category, user_picks=user_picks)
        _trace["steps"].append("sanitizer validated every id against the "
                              "real registries (invalid → formula fallback)")
        # Anti-monotony: adjacent stories never share a transition.
        _rot = _enforce_variety(plan, v)
        if _rot:
            print(f"[v4/director] variety pass rotated {_rot} repeated "
                  f"transition(s)", flush=True)
            _trace["steps"].append(
                f"variety pass rotated {_rot} repeated transition(s)")
        # SELF-REVIEW: a supervising-editor pass over the sanitized plan;
        # concrete complaints go back to the model ONCE for a revision.
        # Gated KAIZER_V4_DIRECTOR_REVIEW (default ON).
        if (os.environ.get("KAIZER_V4_DIRECTOR_REVIEW") or "1").strip().lower() \
                not in ("0", "false", "off", "no"):
            _issues = _plan_quality_issues(plan, durs, words_by_story)
            _trace["review_issues"] = list(_issues)
            if _issues:
                print(f"[v4/director] self-review found {len(_issues)} "
                      f"issue(s): " + " | ".join(_issues), flush=True)
                plan, _adopted = _review_plan(
                    client=_client, genai_types=genai_types,
                    loads_lenient=_loads_lenient, model=model,
                    user_prompt=user, plan=plan, issues=_issues,
                    stories=stories, category=category,
                    user_picks=user_picks, durs=durs,
                    words_by_story=words_by_story)
                _trace["review_revision_adopted"] = _adopted
                _trace["steps"].append(
                    f"self-review: {len(_issues)} issue(s) → revision "
                    + ("ADOPTED" if _adopted else "kept the first plan"))
                _enforce_variety(plan, v)
            else:
                _trace["steps"].append("self-review: no issues found")
        n_moods = len({d.mood for d in plan.values()})
        n_moms = sum(len(d.layout_moments) for d in plan.values())
        n_emph = sum(len(d.emphasis) for d in plan.values())
        print(f"[v4/director] plan: {len(plan)} stories, {n_moods} moods, "
              f"{n_moms} moments, {n_emph} emphasis", flush=True)
        _trace["decisions"] = {str(i): _directive_dict(d)
                               for i, d in sorted(plan.items())}
        _trace["summary"] = (f"{len(plan)} stories, {n_moods} moods, "
                             f"{n_moms} moments, {n_emph} emphasis")
        _write_trace(source_path, _trace)
        return plan
    except Exception as exc:
        print(f"[v4/director] LLM plan failed ({exc}) — formula baseline",
              flush=True)
        plan = formula_plan(stories, category, user_picks=user_picks)
        _write_trace(source_path, {
            "mode": "formula",
            "category": category or "news",
            "error": str(exc)[:400],
            "steps": ["LLM Director failed — the deterministic formula "
                      "baseline directed every story"],
            "decisions": {str(i): _directive_dict(d)
                          for i, d in sorted(plan.items())},
        })
        return plan


def directives_enabled() -> bool:
    return (os.environ.get("KAIZER_V4_DIRECTOR") or "").strip() in ("1", "true", "on")


# Catalog section key (as the admin/user effects catalog exposes it) → the
# Director vocabulary bucket it constrains. Color grades + sound (stings)
# are Director-consumed now (per-story grade/sting/ui_sound directives);
# categories outside these buckets (PiP…) are still ignored here rather
# than faked.
_DIRECTIVE_KEY_MAP = {
    "style_packs": "packs",
    "packs": "packs",
    "transitions": "transitions",
    "frame_fx": "fx",
    "fx": "fx",
    "overlays": "overlays",
    "typography": "captions",
    "captions": "captions",
    "layouts": "layouts",
    "screen_layouts": "layouts",
    "color_grades": "grades",
    "grades": "grades",
    "sound": "stings",
    "sounds": "stings",
}


def parse_style_directives(raw):
    """Parse the user's per-category effect picks (from
    ``KAIZER_V4_STYLE_DIRECTIVES`` — a JSON string, or an already-decoded
    dict) into ``(user_picks, category)``.

    ``user_picks`` = {packs,transitions,fx,overlays,captions,layouts,
    grades,stings}, holding only the buckets the user actually picked
    (empty buckets stay AI-decided).
    ``category`` = an optional story-category override (from a
    ``story_category``/``category`` pick) or ``None``. Returns ``({}, None)``
    on anything unparseable, so the caller behaves exactly as if no
    directives were given."""
    if not raw:
        return {}, None
    data = raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except Exception:
            return {}, None
    if not isinstance(data, dict):
        return {}, None
    picks: dict[str, list] = {}
    for src, bucket in _DIRECTIVE_KEY_MAP.items():
        vals = data.get(src)
        if isinstance(vals, str):
            vals = [vals]
        if isinstance(vals, (list, tuple)):
            clean = [str(x).strip() for x in vals if str(x).strip()]
            if clean:
                dest = picks.setdefault(bucket, [])
                for c in clean:
                    if c not in dest:
                        dest.append(c)
    cat = data.get("story_category") or data.get("category")
    if isinstance(cat, (list, tuple)):
        cat = cat[0] if cat else None
    cat = str(cat).strip() if cat else ""
    return picks, (cat or None)


def story_fx_chain(d: StoryDirective, *, base_chain: str = "") -> str:
    """Compose the story's effects -vf: its mood pack's grade (or the
    Director's per-story grade OVERRIDE when set) + the Director's
    garnish fx. Linear chains only (graph fx are filtered by the
    sanitizer's ban list)."""
    try:
        from pipeline_v4.frame_fx import get_fx_vf
        from pipeline_v4.trailer_styles import STYLES
        parts = []
        p = STYLES.get(d.mood)
        # Grade override: the sanitizer already validated the id, and
        # get_grade_vf is LUT-aware + falls back to a neutral look on an
        # unknown id — a wrong grade can never kill the render. The
        # default d.grade == "" keeps the pack grade CHARACTER-IDENTICAL
        # (byte-compat golden requirement).
        _ovr = ""
        if getattr(d, "grade", ""):
            try:
                from pipeline_v4.color_grades import get_grade_vf
                _ovr = get_grade_vf(d.grade) or ""
            except Exception:
                _ovr = ""
        if p is not None:
            _g = _ovr or p.grade
            chain = _g + ("," + p.extra_vf if p.extra_vf else "")
            parts.append(chain if ";" not in chain else _g)
        elif _ovr:
            parts.append(_ovr)
        elif base_chain:
            parts.append(base_chain)
        for f in d.fx:
            frag = get_fx_vf(f)
            if frag and ";" not in frag:
                parts.append(frag)
        return ",".join(x for x in parts if x)
    except Exception:
        return base_chain
