"""Sound library — 50+ genuinely useful editing sounds, zero copyright.

Every element is SYNTHESIZED on this machine with ffmpeg (sine/noise
sources + filter envelopes) — we own every sample outright, which is
stronger than "open source": there is no license at all to comply with.

Families: impacts, whooshes, risers, stingers, UI, tension/heartbeat,
ambiences, transition audio, musical, misc. The trailer engine keeps
its per-pack parameterized boom/whoosh/riser/bed (trailer.ensure_sfx);
this library is the NAMED vocabulary formulas and the Director pick
from, and what the admin tab lists.

Operator preferences (chat 2026-07-06): 50+ useful sounds, and sound
previews long enough to actually hear — one-shots preview as a DOUBLE
play (effect · gap · effect, ≥3 s), ambiences/pads run 6 s.

Drop-in override: assets/sfx/<key>.wav next to the backend beats the
synthesized version, so real recorded SFX can replace any recipe
without a code change.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Sound:
    key: str
    label: str
    used_for: str
    family: str
    src: str          # lavfi graph (may contain amix of several sources)
    af: str           # audio filter chain shaping the envelope/tone
    dur: float        # seconds
    one_shot: bool = True   # False = ambience/pad (previewed as-is)


def _s(key, label, used_for, family, src, af, dur, one_shot=True):
    return Sound(key, label, used_for, family, src, af, dur, one_shot)


def _tone(freqs: list, d: float) -> str:
    """lavfi graph mixing N sine tones."""
    if len(freqs) == 1:
        return f"sine=frequency={freqs[0]}:duration={d}"
    parts = [f"sine=frequency={f}:duration={d}[s{i}]"
             for i, f in enumerate(freqs)]
    lbls = "".join(f"[s{i}]" for i in range(len(freqs)))
    return ";".join(parts) + f";{lbls}amix=inputs={len(freqs)}:normalize=1"


_NOISE = "anoisesrc=color=white:seed=7:duration={d}:amplitude=0.7"
_PINK = "anoisesrc=color=pink:seed=7:duration={d}:amplitude=0.8"
_BROWN = "anoisesrc=color=brown:seed=7:duration={d}:amplitude=0.9"


_LIB: tuple[Sound, ...] = (
    # ── IMPACTS ───────────────────────────────────────────────────────
    _s("impact_deep", "Deep impact boom", "Big reveals, dramatic hits",
       "impacts", _tone([48, 96], 1.6),
       "volume='exp(-3.2*t)':eval=frame,lowpass=f=220,volume=12dB", 1.6),
    _s("impact_punch", "Punchy hit", "Headline stabs, hard cuts",
       "impacts", _tone([85, 170], 0.9),
       "volume='exp(-6*t)':eval=frame,lowpass=f=500,volume=10dB", 0.9),
    _s("impact_metal", "Metallic clang", "Industrial/impact accents",
       "impacts", _tone([220, 331, 553], 1.4),
       "volume='exp(-4*t)':eval=frame,highpass=f=180,"
       "aecho=0.7:0.5:40:0.35,volume=6dB", 1.4),
    _s("impact_sub_drop", "Sub drop", "Beat drops, trailer moments",
       "impacts", "sine=frequency=120:duration=1.2",
       "asetrate=48000*0.999,vibrato=f=2:d=0.9,"
       "volume='exp(-2.2*t)':eval=frame,lowpass=f=150,volume=12dB", 1.2),
    _s("impact_braam", "Cinematic braam", "Trailer signature blast",
       "impacts", _tone([65, 82, 98, 131], 2.4),
       "volume='min(t*9,1)*exp(-1.4*t)':eval=frame,"
       "aecho=0.8:0.6:60|110:0.4|0.25,lowpass=f=900,volume=9dB", 2.4),
    _s("impact_glass", "Glass shatter", "Break-in, crash accents",
       "impacts", _NOISE.format(d=0.7),
       "highpass=f=3500,volume='exp(-9*t)':eval=frame,"
       "aecho=0.6:0.4:25:0.3,volume=6dB", 0.7),
    _s("impact_thud", "Body thud", "Falls, physical beats",
       "impacts", _NOISE.format(d=0.5),
       "lowpass=f=180,volume='exp(-10*t)':eval=frame,volume=11dB", 0.5),

    # ── WHOOSHES ──────────────────────────────────────────────────────
    _s("whoosh_air", "Air whoosh", "Any cut, graphic slide-in",
       "whooshes", _NOISE.format(d=0.7),
       "bandpass=f=1200:w=900,volume='sin(PI*t/0.7)':eval=frame,volume=8dB",
       0.7),
    _s("whoosh_dark", "Dark whoosh", "Tense/crime cuts",
       "whooshes", _NOISE.format(d=0.8),
       "lowpass=f=700,volume='sin(PI*t/0.8)':eval=frame,volume=9dB", 0.8),
    _s("whoosh_bright", "Bright whoosh", "Light/pop cuts",
       "whooshes", _NOISE.format(d=0.55),
       "highpass=f=1800,volume='sin(PI*t/0.55)':eval=frame,volume=8dB", 0.55),
    _s("whoosh_double", "Double whoosh", "Two-step reveals",
       "whooshes", _NOISE.format(d=1.1),
       "bandpass=f=1000:w=800,"
       "volume='sin(PI*min(t,0.5)/0.5)*lt(t,0.5)+sin(PI*(t-0.55)/0.5)*gte(t,0.55)':eval=frame,"
       "volume=8dB", 1.1),
    _s("whoosh_reverse", "Reverse whoosh", "Suck-in before a drop",
       "whooshes", _NOISE.format(d=0.9),
       "bandpass=f=1400:w=1000,volume='pow(t/0.9,2)':eval=frame,volume=9dB",
       0.9),
    _s("whoosh_long", "Cinematic long whoosh", "Slow majestic transitions",
       "whooshes", _NOISE.format(d=1.6),
       "bandpass=f=900:w=700,volume='sin(PI*t/1.6)':eval=frame,"
       "aecho=0.6:0.4:70:0.3,volume=14dB", 1.6),

    # ── RISERS ────────────────────────────────────────────────────────
    _s("riser_dark", "Dark riser", "Build into a reveal",
       "risers", _NOISE.format(d=2.6),
       "lowpass=f=1800,volume='pow(t/2.6,1.6)':eval=frame,volume=9dB", 2.6,
       one_shot=False),
    _s("riser_bright", "Bright riser", "Energetic build-ups",
       "risers", _NOISE.format(d=2.2),
       "highpass=f=900,volume='pow(t/2.2,1.8)':eval=frame,volume=8dB", 2.2,
       one_shot=False),
    _s("riser_tonal", "Tonal riser", "Pitch climbing tension",
       "risers", "sine=frequency=200:duration=2.4",
       "vibrato=f=0.45:d=1,volume='pow(t/2.4,1.4)':eval=frame,volume=8dB",
       2.4, one_shot=False),
    _s("riser_pulse", "Pulsing riser", "Rhythmic build",
       "risers", _tone([110, 165], 2.4),
       "tremolo=f=7:d=0.8,volume='pow(t/2.4,1.5)':eval=frame,volume=8dB",
       2.4, one_shot=False),

    # ── STINGERS ──────────────────────────────────────────────────────
    _s("sting_news", "News sting", "Segment open/close",
       "stingers", _tone([392, 523, 659], 1.6),
       "volume='exp(-2.2*t)':eval=frame,aecho=0.6:0.4:120:0.3,volume=5dB",
       1.6),
    _s("sting_breaking", "Breaking sting", "Urgent triple hit",
       "stingers", _tone([440, 466], 1.5),
       "tremolo=f=6:d=0.95,volume='exp(-1.8*t)':eval=frame,volume=7dB", 1.5),
    _s("sting_positive", "Positive sting", "Good-news beat, wins",
       "stingers", _tone([523, 659, 784], 1.4),
       "volume='min(t*12,1)*exp(-2*t)':eval=frame,volume=5dB", 1.4),
    _s("sting_negative", "Negative sting", "Bad-news beat, losses",
       "stingers", _tone([196, 155], 1.8),
       "volume='exp(-1.6*t)':eval=frame,lowpass=f=800,volume=7dB", 1.8),
    _s("sting_question", "Question sting", "Cliffhangers, mysteries",
       "stingers", "sine=frequency=330:duration=1.2",
       "vibrato=f=1.2:d=0.85,volume='exp(-1.4*t)':eval=frame,volume=6dB",
       1.2),

    # ── UI / INTERFACE ────────────────────────────────────────────────
    _s("ui_tick", "Tick", "Overlay appears, list items",
       "ui", "sine=frequency=1800:duration=0.09",
       "volume='exp(-40*t)':eval=frame,volume=4dB", 0.09),
    _s("ui_pop", "Pop", "Bubble/graphic pop-in",
       "ui", "sine=frequency=600:duration=0.16",
       "asetrate=48000*1.001,volume='exp(-18*t)':eval=frame,volume=6dB",
       0.16),
    _s("ui_click", "Click", "Button/beat accents",
       "ui", _NOISE.format(d=0.06),
       "highpass=f=2500,volume='exp(-50*t)':eval=frame,volume=3dB", 0.06),
    _s("ui_notify", "Notification", "Incoming info, social moments",
       "ui", _tone([880, 1174], 0.5),
       "volume='exp(-6*t)':eval=frame,volume=4dB", 0.5),
    _s("ui_swipe", "Swipe", "Panel slide, page turn",
       "ui", _NOISE.format(d=0.3),
       "bandpass=f=2000:w=1500,volume='sin(PI*t/0.3)':eval=frame,volume=5dB",
       0.3),
    _s("ui_error", "Error buzz", "Wrong answers, fails (memes too)",
       "ui", "sine=frequency=110:duration=0.6",
       "tremolo=f=30:d=0.9,volume='exp(-4*t)':eval=frame,volume=7dB", 0.6),
    _s("ui_success", "Success ding", "Checkmarks, achievements",
       "ui", _tone([1320, 1760], 0.9),
       "volume='exp(-4.5*t)':eval=frame,volume=4dB", 0.9),

    # ── TENSION / HEARTBEAT ───────────────────────────────────────────
    _s("heartbeat_slow", "Heartbeat (slow)", "Dread, waiting, suspense",
       "tension", "sine=frequency=58:duration=2.2",
       "volume='exp(-14*mod(t,1.1))+0.7*exp(-14*mod(t-0.25,1.1))*gte(t,0.25)':eval=frame,"
       "lowpass=f=140,volume=11dB", 2.2),
    _s("heartbeat_fast", "Heartbeat (fast)", "Panic, chase, countdown",
       "tension", "sine=frequency=62:duration=2.0",
       "volume='exp(-16*mod(t,0.55))+0.7*exp(-16*mod(t-0.14,0.55))*gte(t,0.14)':eval=frame,"
       "lowpass=f=150,volume=11dB", 2.0),
    _s("clock_tick", "Clock tick-tock", "Deadlines, countdowns",
       "tension", _NOISE.format(d=2.4),
       "highpass=f=1200,volume='exp(-60*mod(t,0.6))':eval=frame,volume=6dB",
       2.4),
    _s("tension_pulse", "Tension pulse", "Underscore for revelations",
       "tension", _tone([55, 82], 3.0),
       "tremolo=f=2.4:d=0.85,volume=6dB", 3.0, one_shot=False),
    _s("drone_dark", "Dark drone", "Horror/crime floor",
       "tension", _tone([55, 58, 110], 6.0),
       "lowpass=f=300,volume=5dB", 6.0, one_shot=False),

    # ── AMBIENCES (6 s floors) ────────────────────────────────────────
    _s("amb_rain", "Rain", "Weather, moody exteriors",
       "ambience", _PINK.format(d=6.0),
       "highpass=f=400,lowpass=f=7000,tremolo=f=11:d=0.12,volume=2dB",
       6.0, one_shot=False),
    _s("amb_wind", "Wind", "Storms, desolate scenes",
       "ambience", _PINK.format(d=6.0),
       "lowpass=f=650,tremolo=f=0.35:d=0.55,volume=4dB", 6.0,
       one_shot=False),
    _s("amb_city", "City hum", "Urban stories, street footage",
       "ambience", _BROWN.format(d=6.0) + "[n];sine=frequency=120:duration=6[h];"
       "[n][h]amix=inputs=2:weights='1 0.15'",
       "lowpass=f=1500,volume=3dB", 6.0, one_shot=False),
    _s("amb_crowd", "Crowd murmur", "Events, protests, stadiums",
       "ambience", _PINK.format(d=6.0),
       "bandpass=f=700:w=800,tremolo=f=5.5:d=0.4,volume=4dB", 6.0,
       one_shot=False),
    _s("amb_room", "Room tone", "Interview beds, silence filler",
       "ambience", _BROWN.format(d=6.0),
       "lowpass=f=900,volume=-6dB", 6.0, one_shot=False),
    _s("amb_night", "Night crickets", "Rural night scenes",
       "ambience", "sine=frequency=4200:duration=6[c];" + _BROWN.format(d=6.0)
       + "[f];[c][f]amix=inputs=2:weights='0.5 1'",
       "tremolo=f=13:d=0.9,lowpass=f=6000,volume=1dB", 6.0, one_shot=False),
    _s("amb_ocean", "Ocean waves", "Coastal, calm sequences",
       "ambience", _PINK.format(d=6.0),
       "lowpass=f=800,tremolo=f=0.14:d=0.8,volume=5dB", 6.0, one_shot=False),
    _s("amb_thunder", "Thunder rumble", "Storm drama",
       "ambience", _BROWN.format(d=6.0),
       "lowpass=f=120,volume='0.4+0.6*exp(-2*mod(t,2.7))':eval=frame,"
       "volume=8dB", 6.0, one_shot=False),

    # ── TRANSITION AUDIO ──────────────────────────────────────────────
    _s("trans_static", "Static burst", "Channel-change cuts, glitch",
       "transition", _NOISE.format(d=0.4),
       "volume='exp(-6*t)':eval=frame,volume=6dB", 0.4),
    _s("trans_glitch", "Glitch stutter", "Digital tears",
       "transition", _NOISE.format(d=0.6),
       "tremolo=f=28:d=1,bandpass=f=2500:w=2000,"
       "volume='exp(-4*t)':eval=frame,volume=17dB", 0.6),
    _s("trans_tape_stop", "Tape stop", "Comedy brakes, rewinds",
       "transition", "sine=frequency=440:duration=0.9",
       "vibrato=f=6:d=1,volume='exp(-2.5*t)':eval=frame,"
       "lowpass=f=1200,volume=6dB", 0.9),
    _s("trans_rewind", "Rewind chirp", "Replay/flashback intros",
       "transition", "sine=frequency=880:duration=0.7",
       "tremolo=f=16:d=0.9,volume='pow(1-t/0.7,0.6)':eval=frame,volume=5dB",
       0.7),
    _s("trans_shutter", "Camera shutter", "Photo-freeze moments",
       "transition", _NOISE.format(d=0.18),
       "highpass=f=1500,volume='exp(-28*t)':eval=frame,volume=5dB", 0.18),

    # ── MUSICAL ELEMENTS ──────────────────────────────────────────────
    _s("note_pluck", "Pluck", "Bullet-point reveals",
       "musical", "sine=frequency=880:duration=0.8",
       "volume='exp(-7*t)':eval=frame,aecho=0.5:0.3:90:0.25,volume=4dB",
       0.8),
    _s("pad_warm", "Warm pad", "Hopeful/devotional floor",
       "musical", _tone([220, 277, 330], 6.0),
       "lowpass=f=1200,tremolo=f=0.3:d=0.25,volume=1dB", 6.0,
       one_shot=False),
    _s("pad_dark", "Dark pad", "Somber/serious floor",
       "musical", _tone([110, 131, 165], 6.0),
       "lowpass=f=800,volume=2dB", 6.0, one_shot=False),
    _s("arp_rise", "Arpeggio rise", "Positive reveals, stats going up",
       "musical", _tone([262, 330, 392, 523], 1.6),
       "tremolo=f=8:d=0.6,volume='min(t*4,1)':eval=frame,volume=4dB", 1.6),
    _s("bass_pulse", "Bass pulse", "Rhythmic underscore",
       "musical", "sine=frequency=55:duration=2.4",
       "tremolo=f=4:d=0.95,volume=9dB", 2.4, one_shot=False),
    _s("chime_bell", "Bell chime", "Time passing, calm notes",
       "musical", _tone([1760, 2637], 2.2),
       "volume='exp(-2.2*t)':eval=frame,aecho=0.6:0.4:200:0.3,volume=2dB",
       2.2),

    # ── MISC ──────────────────────────────────────────────────────────
    _s("applause_burst", "Applause", "Wins, celebrations",
       "misc", _NOISE.format(d=2.2),
       "bandpass=f=2200:w=2400,tremolo=f=12:d=0.55,"
       "volume='min(t*8,1)*exp(-1.1*t)':eval=frame,volume=13dB", 2.2),
    _s("camera_flash", "Camera flash", "Paparazzi/press moments",
       "misc", _tone([2200], 0.25) + "[t];" + _NOISE.format(d=0.25)
       + "[n];[t][n]amix=inputs=2:weights='0.4 1'",
       "highpass=f=1800,volume='exp(-22*t)':eval=frame,volume=5dB", 0.25),
    _s("countdown_beep", "Countdown beeps", "3-2-1 moments",
       "misc", "sine=frequency=990:duration=2.3",
       "volume='exp(-22*mod(t,1))':eval=frame,volume=5dB", 2.3),
)

LIBRARY: dict[str, Sound] = {s.key: s for s in _LIB}


def _ffmpeg() -> str:
    return os.environ.get("KAIZER_FFMPEG_BIN", "ffmpeg")


def _override_path(key: str) -> Optional[Path]:
    base = Path(__file__).resolve().parent.parent / "assets" / "sfx"
    p = base / f"{key}.wav"
    return p if p.is_file() else None


def ensure_sound(key: str, out_dir: Path) -> Optional[str]:
    """Synthesize (once) one library sound → wav path. Drop-in override
    in assets/sfx/<key>.wav wins. Unknown/failed → None (fail-soft)."""
    snd = LIBRARY.get((key or "").strip().lower())
    if not snd:
        return None
    ov = _override_path(snd.key)
    if ov:
        return str(ov)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"snd_{snd.key}.wav"
    if dst.is_file():
        return str(dst)
    cmd = [_ffmpeg(), "-y", "-v", "error",
           "-f", "lavfi", "-i", snd.src,
           "-af", snd.af, "-t", f"{snd.dur}",
           "-ar", "48000", "-ac", "1", str(dst)]
    try:
        ok = subprocess.run(cmd, capture_output=True,
                            timeout=120).returncode == 0
    except Exception:
        ok = False
    return str(dst) if (ok and dst.is_file()) else None


def preview_sound(key: str, out_dir: Path) -> Optional[str]:
    """A HEARABLE preview: one-shots play twice with a gap (≥3 s total);
    ambiences/pads/risers return their full-length file."""
    snd = LIBRARY.get((key or "").strip().lower())
    if not snd:
        return None
    src = ensure_sound(key, out_dir)
    if not src:
        return None
    if not snd.one_shot:
        return src
    dst = Path(out_dir) / f"prev_{snd.key}.wav"
    if dst.is_file():
        return str(dst)
    gap = max(0.5, 1.6 - snd.dur)          # short blips get a longer gap
    tail = max(0.0, 3.2 - 2 * snd.dur - gap)
    fc = (f"[0:a]apad=pad_dur={gap:.2f}[p];"
          f"[p][1:a]concat=n=2:v=0:a=1,apad=pad_dur={tail:.2f}[a]")
    cmd = [_ffmpeg(), "-y", "-v", "error", "-i", src, "-i", src,
           "-filter_complex", fc, "-map", "[a]",
           "-ar", "48000", "-ac", "1", str(dst)]
    try:
        ok = subprocess.run(cmd, capture_output=True,
                            timeout=120).returncode == 0
    except Exception:
        ok = False
    return str(dst) if (ok and dst.is_file()) else src


def sound_catalog() -> list[dict]:
    """[{key,label,used_for,family,duration}] for the admin features tab."""
    return [{"key": s.key, "label": s.label, "used_for": s.used_for,
             "family": s.family, "duration": s.dur} for s in _LIB]
