"""Frame FX registry — texture/motion/glitch treatments as vetted chains.

Sister registry to color_grades: every effect is a parameterized ffmpeg
-vf fragment builder using ONLY standard filters, validated by rendering
a real frame in tests (the "Option not found" bug class dies here, not
in a paid render). Color LOOKS live in color_grades; this module is the
spec's §5 VFX vocabulary: grain, chromatic aberration, scanlines, VHS,
shake, dutch angle, strobe, slow-mo/speed, blur, pixelate, flash cuts.

Style packs and formulas reference effects BY ID; the Admin Features tab
lists + previews from FX (each row renders against a test pattern).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Fx:
    key: str
    label: str
    used_for: str
    build: Callable[..., str]          # (**params) -> vf fragment
    defaults: dict = field(default_factory=dict)
    note: str = ""                     # constraints (audio handling etc.)


def _fx(key, label, used_for, build, defaults=None, note=""):
    return Fx(key, label, used_for, build, defaults or {}, note)


_F: tuple[Fx, ...] = (
    _fx("film_grain", "Film grain",
        "Documentary authenticity, archive feel",
        lambda strength=7: f"noise=alls={int(strength)}:allf=t",
        {"strength": 7}),
    _fx("heavy_grain", "Heavy grain",
        "Horror, found footage, raw urgency",
        lambda strength=14: f"noise=alls={int(strength)}:allf=t",
        {"strength": 14}),
    _fx("chromatic_aberration", "Chromatic aberration (RGB split)",
        "Glitch tension, cyber, tech breakage",
        lambda px=4: f"rgbashift=rh={int(px)}:bv=-{int(px)}",
        {"px": 4}),
    _fx("vhs", "VHS distortion",
        "Found footage, caught-on-tape, retro",
        lambda: "noise=alls=12:allf=t,rgbashift=rh=2:bv=2,"
                "drawgrid=w=iw:h=3:t=1:c=black@0.25"),
    _fx("crt_scanlines", "CRT scanlines",
        "Old monitor / surveillance aesthetic",
        lambda gap=4: f"drawgrid=w=iw:h={int(gap)}:t=1:c=black@0.35",
        {"gap": 4}),
    _fx("gaussian_blur", "Gaussian blur",
        "Background defocus layers, dreamy inserts",
        lambda sigma=12: f"gblur=sigma={float(sigma)}",
        {"sigma": 12}),
    _fx("pixelate", "Pixelate / mosaic",
        "Censorship, identity protection, gaming",
        lambda w=1920, h=1080, block=16:
            f"scale=trunc(iw/{int(block)}):trunc(ih/{int(block)}),"
            f"scale={int(w)}:{int(h)}:flags=neighbor",
        {"w": 1920, "h": 1080, "block": 16}),
    _fx("vignette_dark", "Dark vignette",
        "Focus attention, serious tone",
        lambda angle=4.5: f"vignette=PI/{float(angle)}",
        {"angle": 4.5}),
    _fx("handheld_shake", "Handheld shake",
        "Urgency, breaking news raw feel",
        lambda amp=6: (f"crop=iw-{2 * int(amp) + 4}:ih-{2 * int(amp) + 4}:"
                       f"x='{int(amp) + 2}+{int(amp)}*sin(t*37)':"
                       f"y='{int(amp) + 2}+{int(amp) - 1}*cos(t*29)'"),
        {"amp": 6},
        note="deterministic pseudo-shake; crops a few px off the edges"),
    _fx("dutch_angle", "Dutch angle",
        "Unease, tension, action",
        lambda deg=3.0: f"rotate={float(deg)}*PI/180:fillcolor=black",
        {"deg": 3.0}),
    _fx("strobe", "Strobe flash",
        "Emergency, club, disorientation (use sparingly)",
        lambda period=6: (f"eq=brightness='if(lt(mod(n\\,{int(period)})\\,1)"
                          f"\\,0.4\\,0)':eval=frame"),
        {"period": 6}),
    _fx("flash_white_in", "Flash-cut entrance (white)",
        "High-energy clip entrances (sports, breaking)",
        lambda d=0.12: f"fade=t=in:st=0:d={float(d)}:color=white",
        {"d": 0.12}),
    _fx("flash_black_in", "Flash-cut entrance (black)",
        "Dark/moody clip entrances (crime, horror)",
        lambda d=0.12: f"fade=t=in:st=0:d={float(d)}:color=black",
        {"d": 0.12}),
    _fx("slow_motion", "Slow motion",
        "Emotional beats, sports impacts, tragedy",
        lambda factor=0.5: f"setpts={1.0 / max(0.1, float(factor)):.4f}*PTS",
        {"factor": 0.5},
        note="video-only: caller must handle audio (atempo or mute)"),
    _fx("speed_up", "Speed up",
        "Timelapse feel, montage energy",
        lambda factor=2.0: f"setpts={1.0 / max(0.2, float(factor)):.4f}*PTS",
        {"factor": 2.0},
        note="video-only: caller must handle audio"),
    _fx("mirror", "Mirror split",
        "Creative/artistic symmetry",
        lambda: "crop=iw/2:ih:0:0,split[l][r];[r]hflip[rf];[l][rf]hstack",
        note="filter GRAPH (contains ;) — use in -filter_complex, not -vf"),

    # ── camera moves (digital zoom/pan; w,h = output canvas) ──────────
    # NOTE: crop evaluates w/h ONCE (t is NaN there) — only x/y animate.
    # Time-varying zoom on video therefore uses zoompan (z per frame).
    _fx("slow_zoom_in", "Slow zoom in",
        "Build intensity on a subject, interview emphasis",
        lambda w=1920, h=1080: (
            "zoompan=z='1+0.06*in_time':x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':d=1:s={int(w)}x{int(h)}:fps=25"),
        {"w": 1920, "h": 1080}),
    _fx("slow_zoom_out", "Slow zoom out",
        "Reveal context, closing shots",
        lambda w=1920, h=1080: (
            "zoompan=z='max(1.30-0.06*in_time,1.001)':"
            "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d=1:s={int(w)}x{int(h)}:fps=25"),
        {"w": 1920, "h": 1080}),
    _fx("crash_zoom_in", "Crash zoom in",
        "Shock reveals, comedy punch-ins",
        lambda w=1920, h=1080: (
            "zoompan=z='1+2.5*min(in_time,0.35)':x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':d=1:s={int(w)}x{int(h)}:fps=25"),
        {"w": 1920, "h": 1080}),
    _fx("crash_zoom_out", "Crash zoom out",
        "Whip-back reveals",
        lambda w=1920, h=1080: (
            "zoompan=z='max(1.9-2.5*in_time,1.001)':x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':d=1:s={int(w)}x{int(h)}:fps=25"),
        {"w": 1920, "h": 1080}),
    _fx("zoom_pulse", "Zoom pulse (heartbeat)",
        "Beat-synced emphasis, music segments",
        lambda w=1920, h=1080: (
            "zoompan=z='1+0.035*abs(sin(in_time*3.6))':"
            "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d=1:s={int(w)}x{int(h)}:fps=25"),
        {"w": 1920, "h": 1080}),
    _fx("bounce_zoom", "Bounce zoom",
        "Playful entrances (comedy, kids, promos)",
        lambda w=1920, h=1080: (
            "zoompan=z='1+0.15*exp(-1.5*in_time)*abs(cos(in_time*9))':"
            "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d=1:s={int(w)}x{int(h)}:fps=25"),
        {"w": 1920, "h": 1080}),
    _fx("pan_left", "Pan left",
        "Scanning a scene, landscape sweeps",
        lambda w=1920, h=1080: (
            "crop=w='iw*0.86':h='ih*0.86':"
            "x='(iw-ow)*(1-min(t/4,1))':y='(ih-oh)/2',"
            f"scale={int(w)}:{int(h)}"),
        {"w": 1920, "h": 1080}),
    _fx("pan_right", "Pan right",
        "Scanning a scene the other way",
        lambda w=1920, h=1080: (
            "crop=w='iw*0.86':h='ih*0.86':"
            "x='(iw-ow)*min(t/4,1)':y='(ih-oh)/2',"
            f"scale={int(w)}:{int(h)}"),
        {"w": 1920, "h": 1080}),
    _fx("tilt_up", "Tilt up",
        "Reveal from ground to subject (buildings, reveals)",
        lambda w=1920, h=1080: (
            "crop=w='iw*0.86':h='ih*0.86':x='(iw-ow)/2':"
            "y='(ih-oh)*(1-min(t/4,1))',"
            f"scale={int(w)}:{int(h)}"),
        {"w": 1920, "h": 1080}),
    _fx("tilt_down", "Tilt down",
        "Sky-to-subject establishing move",
        lambda w=1920, h=1080: (
            "crop=w='iw*0.86':h='ih*0.86':x='(iw-ow)/2':"
            "y='(ih-oh)*min(t/4,1)',"
            f"scale={int(w)}:{int(h)}"),
        {"w": 1920, "h": 1080}),
    _fx("ken_burns_diag", "Ken Burns drift",
        "Documentary photo/footage life",
        lambda w=1920, h=1080: (
            "zoompan=z='1+0.05*in_time':"
            "x='(iw-iw/zoom)*min(in_time/5,1)':"
            "y='(ih-ih/zoom)*min(in_time/5,1)':"
            f"d=1:s={int(w)}x{int(h)}:fps=25"),
        {"w": 1920, "h": 1080}),
    _fx("vertigo_zoom", "Vertigo zoom",
        "Unease + slow rotation (thriller/psych)",
        lambda w=1920, h=1080: (
            "zoompan=z='1+0.07*in_time':x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':d=1:s={int(w)}x{int(h)}:fps=25,"
            "rotate='0.02*sin(t*1.4)':fillcolor=black"),
        {"w": 1920, "h": 1080}),

    # ── shake & sway ──────────────────────────────────────────────────
    _fx("earthquake", "Earthquake shake",
        "Disaster/impact moments",
        lambda: ("crop=iw-32:ih-32:x='16+14*sin(t*41)':"
                 "y='16+12*cos(t*33)'"),
        note="crops 16px borders"),
    _fx("micro_shake", "Micro shake",
        "Subtle handheld life on static shots",
        lambda: "crop=iw-8:ih-8:x='4+2*sin(t*13)':y='4+2*cos(t*11)'"),
    _fx("whip_shake", "Whip shake",
        "Violent horizontal judder (impacts, bass hits)",
        lambda: "crop=iw-24:ih-24:x='12+11*sin(t*55)':y=12"),
    _fx("pendulum_sway", "Pendulum sway",
        "Dreamy/drunk/unstable POV",
        lambda: "rotate='0.035*sin(t*1.8)':fillcolor=black"),
    _fx("spin_slow", "Slow spin",
        "Stylized montage moments",
        lambda: "rotate='t*0.35':fillcolor=black"),

    # ── time & motion trails ──────────────────────────────────────────
    _fx("freeze_intro", "Freeze-frame intro",
        "Hold the first frame ~1s then play (character intros)",
        lambda: "loop=loop=30:size=1:start=0,setpts=N/FRAME_RATE/TB",
        note="adds ~1.2s at start; video-only"),
    _fx("stutter_frames", "Frame stutter",
        "Glitchy repeated-frame tension",
        lambda: "shuffleframes=mapping='0 0 0',setpts=N/FRAME_RATE/TB",
        note="video-only"),
    _fx("motion_smear", "Motion smear",
        "Speed feel on fast footage (sports, chases)",
        lambda frames=6: f"tmix=frames={int(frames)}",
        {"frames": 6}),
    _fx("echo_trails", "Echo trails",
        "Ghosting trails (horror, dream, drug sequences)",
        lambda decay=0.96: f"lagfun=decay={float(decay)}",
        {"decay": 0.96}),
    _fx("datamosh_smear", "Datamosh smear",
        "Digital decay aesthetic",
        lambda: "tmix=frames=10,rgbashift=rh=6:bv=-6"),

    # ── composite looks (filter graphs) ───────────────────────────────
    _fx("soft_glow", "Soft glow",
        "Beauty/dream lift on faces and lights",
        lambda: ("split[gA][gB];[gB]gblur=sigma=12[gBb];"
                 "[gA][gBb]blend=all_mode=screen:all_opacity=0.45"),
        note="filter GRAPH — use in -filter_complex"),
    _fx("halation_glow", "Halation bloom",
        "Filmic red-tinted highlight bloom",
        lambda: ("split[hA][hB];[hB]gblur=sigma=16,"
                 "colorbalance=rs=0.3:rm=0.2[hBb];"
                 "[hA][hBb]blend=all_mode=lighten:all_opacity=0.5"),
        note="filter GRAPH — use in -filter_complex"),
    _fx("double_exposure", "Double exposure",
        "Artistic mirrored ghost blend (music, intros)",
        lambda: ("split[dA][dB];[dB]hflip[dBf];"
                 "[dA][dBf]blend=all_mode=average"),
        note="filter GRAPH — use in -filter_complex"),
    _fx("kaleidoscope_quad", "Kaleidoscope",
        "Four-way mirrored pattern (music/abstract)",
        lambda: ("crop=iw/2:ih/2:0:0,split=4[k0][k1][k2][k3];"
                 "[k1]hflip[k1f];[k2]vflip[k2f];[k3]hflip,vflip[k3f];"
                 "[k0][k1f]hstack[ktop];[k2f][k3f]hstack[kbot];"
                 "[ktop][kbot]vstack"),
        note="filter GRAPH — use in -filter_complex"),

    # ── era / environment looks ───────────────────────────────────────
    _fx("old_film", "Old film",
        "Archive/history segments",
        lambda: ("hue=s=0.35,noise=alls=11:allf=t,vignette=PI/4.5,"
                 "eq=contrast=1.12:brightness=-0.02")),
    _fx("security_cam", "Security camera",
        "Crime reconstructions, surveillance framing",
        lambda: ("hue=s=0.2,drawgrid=w=iw:h=3:t=1:c=black@0.3,"
                 "vignette=PI/4,eq=contrast=1.15")),
    _fx("underwater_haze", "Underwater haze",
        "Submerged/flood footage mood",
        lambda: ("gblur=sigma=3,colorbalance=bs=0.22:gm=0.08,"
                 "eq=brightness=0.02")),
    _fx("dream_haze", "Dream haze",
        "Flashbacks, memories, soft reveries",
        lambda: ("gblur=sigma=2,curves=all='0/0.06 1/0.97',"
                 "eq=saturation=0.85")),
    _fx("tape_damage", "Damaged tape",
        "Corrupted-recording horror beats",
        lambda: ("noise=alls=16:allf=t+u,hue=s=0.55,"
                 "drawgrid=w=iw:h=5:t=1:c=black@0.2")),
    _fx("film_dirt", "Film dirt",
        "Projector-dust texture",
        lambda: "noise=c0s=14:c0f=t+u"),

    # ── color treatments ──────────────────────────────────────────────
    _fx("sepia_tone", "Sepia",
        "Vintage/memory framing",
        lambda: ("colorchannelmixer="
                 ".393:.769:.189:0:.349:.686:.168:0:.272:.534:.131")),
    _fx("mono_contrast", "Mono high-contrast",
        "Dramatic B&W statements",
        lambda: "hue=s=0,eq=contrast=1.5"),
    _fx("duotone_cool", "Duotone cool",
        "Stylized tech/crime posters",
        lambda: ("hue=s=0,eq=contrast=1.25,"
                 "colorbalance=bs=0.25:bm=0.12:rs=-0.08")),
    _fx("duotone_warm", "Duotone warm",
        "Stylized lifestyle/gold looks",
        lambda: ("hue=s=0,eq=contrast=1.2,"
                 "colorbalance=rs=0.22:rm=0.12:bs=-0.10")),
    _fx("posterize_bands", "Posterize",
        "Graphic poster reduction",
        lambda: "lutyuv=y='trunc(val/48)*48'"),
    _fx("solarize", "Solarize",
        "Psychedelic inversion of highlights",
        lambda: "curves=all='0/0 0.5/1 1/0'"),
    _fx("negative_invert", "Negative",
        "Full color inversion (flash frames, horror)",
        lambda: "negate"),
    _fx("hue_shift_90", "Hue shift 90°",
        "Alien/off-world recolor",
        lambda deg=90: f"hue=h={int(deg)}",
        {"deg": 90}),
    _fx("color_cycle", "Color cycle",
        "Party/EDM animated hue sweep",
        lambda: "hue=h='mod(t*40,360)'"),
    _fx("saturation_pulse", "Saturation pulse",
        "Beat-reactive color breathing",
        lambda: "eq=saturation='1+0.45*sin(t*4)':eval=frame"),
    _fx("vibrance_pop", "Vibrance pop",
        "Food/travel color lift without skin blowout",
        lambda strength=0.4: f"vibrance=intensity={float(strength)}",
        {"strength": 0.4}),
    _fx("contrast_punch", "Contrast punch",
        "Instant punchy grade for promos",
        lambda: "eq=contrast=1.45:saturation=1.1"),
    _fx("washed_out", "Washed out",
        "Bleak/aftermath desaturated lift",
        lambda: "curves=all='0/0.15 1/1',eq=saturation=0.6"),
    _fx("sketch_edges", "Pencil sketch",
        "Illustrated inserts/explainers",
        lambda: "edgedetect,negate"),
    _fx("emboss_relief", "Emboss relief",
        "Metallic stamped texture",
        lambda: ("convolution='-2 -1 0 -1 1 1 0 1 2':"
                 "'-2 -1 0 -1 1 1 0 1 2':'-2 -1 0 -1 1 1 0 1 2'")),
    _fx("neon_edges", "Neon edges",
        "Cyber outline glow look",
        lambda: ("edgedetect=mode=colormix:high=0.2,"
                 "eq=saturation=1.8:contrast=1.3")),

    # ── light & frame devices ─────────────────────────────────────────
    _fx("letterbox_cinema", "Cinema letterbox",
        "Instant film-look bars on 16:9",
        lambda: ("drawbox=x=0:y=0:w=iw:h=ih*0.12:c=black:t=fill,"
                 "drawbox=x=0:y=ih*0.88:w=iw:h=ih*0.12:c=black:t=fill")),
    _fx("side_bars", "Side pillars",
        "Focus-center pillarbox device",
        lambda: ("drawbox=x=0:y=0:w=iw*0.10:h=ih:c=black:t=fill,"
                 "drawbox=x=iw*0.90:y=0:w=iw*0.10:h=ih:c=black:t=fill")),
    _fx("vignette_soft", "Soft vignette",
        "Gentle viewer focus",
        lambda: "vignette=PI/6"),
    _fx("vignette_pulse", "Breathing vignette",
        "Slow darkness breathing (tension)",
        lambda: "vignette=angle='PI/5+PI/20*sin(t*2)':eval=frame"),
    _fx("light_bloom_edges", "Edge light bloom",
        "Dreamlike bright-edge wash",
        lambda: "vignette=angle=PI/4:mode=backward"),
    _fx("film_burn_in", "Film burn entrance",
        "Warm burn-in from orange (retro openers)",
        lambda d=0.5: f"fade=t=in:d={float(d)}:color=orange",
        {"d": 0.5}),
    _fx("white_burn_out", "White burn exit",
        "Burn-to-white ending",
        lambda st=3.5, d=0.5: f"fade=t=out:st={float(st)}:d={float(d)}:color=white",
        {"st": 3.5, "d": 0.5},
        note="st must be clip_duration - d"),
    _fx("flash_bang", "Flash bang",
        "Explosive bright hit decaying (impacts)",
        lambda: "eq=brightness='0.6*exp(-3*t)':eval=frame"),

    # ── flicker & glitch garnish ──────────────────────────────────────
    _fx("exposure_pulse", "Exposure pulse",
        "Slow light breathing",
        lambda: "eq=brightness='0.05*sin(t*3)':eval=frame"),
    _fx("crt_flicker", "CRT flicker",
        "Old-monitor strobe-less flicker",
        lambda: ("eq=brightness='0.05*sin(t*72)':eval=frame,"
                 "drawgrid=w=iw:h=3:t=1:c=black@0.28")),
    _fx("glitch_pulse", "Glitch pulse",
        "Heavy RGB tear + brightness jitter",
        lambda: ("rgbashift=rh=8:bv=-8:gh=-4,"
                 "eq=brightness='0.06*sin(t*47)':eval=frame")),
)

FX: dict[str, Fx] = {f.key: f for f in _F}


def get_fx_vf(key: str, **params) -> str:
    """The -vf fragment for an effect id with optional param overrides.
    Unknown id → '' (a missing garnish never kills a render)."""
    f = FX.get((key or "").strip().lower())
    if not f:
        return ""
    kw = dict(f.defaults)
    kw.update(params or {})
    return f.build(**kw)


def fx_catalog() -> list[dict]:
    """[{key,label,used_for,params,note}] for the admin features tab."""
    return [{"key": f.key, "label": f.label, "used_for": f.used_for,
             "params": dict(f.defaults), "note": f.note} for f in _F]
