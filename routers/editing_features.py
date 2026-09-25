"""Admin "Editing Features" — the living catalog of the editing engine.

Operator requirement (email 2026-07-06): a separate admin section that
lists EVERY effect, frame, transition, sound and template the system
uses, each with a rendered DUMMY so the admin can SEE/HEAR what it is.
Operator follow-up (chat 2026-07-06): previews must be CapCut-style —
rendered on REAL footage from this installation, not test patterns.

GET /api/admin/editing-features            → the full catalog, grouped
GET /api/admin/editing-features/preview    → renders (and caches) a
    dummy for one item ON A REAL CLIP: MP4 through the exact filter
    chain for grades/FX/style packs, a real-footage xfade MP4 for
    transitions, overlay/typography composited onto a real frame,
    WAV for sounds. Returns {"url": "/api/file/?path=...", "media":
    "video"|"image"|"audio"} which the SPA loads directly.

Source clip: KAIZER_FEATURE_PREVIEW_CLIP env override, else the newest
raw story footage found under the render roots, else (clipless hosts /
CI) synthetic test patterns — previews always work.

Previews are generated ON DEMAND and cached under
<media_root>/_feature_previews/, keyed by a fingerprint of the source
clip so dropping in a new clip regenerates everything.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

import auth
import models

router = APIRouter(prefix="/api/admin/editing-features",
                   tags=["admin-editing-features"])

_PV_W, _PV_H = 480, 270            # mp4 preview canvas
_FR_W, _FR_H = 960, 540            # png composite canvas


def _media_root() -> Path:
    base = Path(__file__).resolve().parent.parent
    return Path(os.environ.get("KAIZER_OUTPUT_ROOT") or (base / "output"))


def _cache_dir() -> Path:
    d = _media_root() / "_feature_previews"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resp(p: str, media: str) -> dict:
    return {"url": f"/api/file/?path={p}", "media": media}


def _ffmpeg() -> str:
    return os.environ.get("KAIZER_FFMPEG_BIN", "ffmpeg")


def _run(cmd: list, timeout: int = 120) -> bool:
    try:
        return subprocess.run(cmd, capture_output=True,
                              timeout=timeout).returncode == 0
    except Exception:
        return False


# ─── Real source clip ───────────────────────────────────────────────

_CLIP_ROOTS = (
    "D:/kaizer-dev-renders",
    "D:/kaizer-renders",
)


def _find_real_clip() -> Path | None:
    """A real piece of footage from this installation. Prefers raw
    (pre-compose) story clips — clean video with no overlays baked in."""
    env = os.environ.get("KAIZER_FEATURE_PREVIEW_CLIP", "").strip()
    if env and Path(env).is_file():
        return Path(env)
    best, best_mtime = None, 0.0
    roots = [Path(r) for r in _CLIP_ROOTS] + [_media_root()]
    for root in roots:
        if not root.is_dir():
            continue
        for pat in ("**/_bulletin/raw_story_*.mp4", "**/raw_story_*.mp4"):
            try:
                for p in root.glob(pat):
                    try:
                        st = p.stat()
                    except OSError:
                        continue
                    if st.st_size > 2_000_000 and st.st_mtime > best_mtime:
                        best, best_mtime = p, st.st_mtime
            except OSError:
                continue
        if best:
            break
    return best


def _clip_duration(path: Path) -> float:
    try:
        out = subprocess.run(
            [os.environ.get("KAIZER_FFPROBE_BIN", "ffprobe"), "-v", "error",
             "-show_entries", "format=duration", "-of", "json", str(path)],
            capture_output=True, timeout=30)
        return float(json.loads(out.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def _ensure_base() -> dict | None:
    """Extract (once per source clip) the tiny normalized excerpts every
    preview builds from: base.mp4 (4 s), a/b.mp4 (1.4 s, two different
    scenes, for transitions) and frame.png (one real frame)."""
    clip = _find_real_clip()
    if not clip:
        return None
    st = clip.stat()
    fp = hashlib.md5(f"{clip}|{st.st_size}|{int(st.st_mtime)}"
                     .encode()).hexdigest()[:8]
    cache = _cache_dir()
    base = {"fp": fp,
            "base": cache / f"src_{fp}_base.mp4",
            "a": cache / f"src_{fp}_a.mp4",
            "b": cache / f"src_{fp}_b.mp4",
            "frame": cache / f"src_{fp}_frame.png"}
    if all(p.is_file() for k, p in base.items() if k != "fp"):
        return base
    dur = _clip_duration(clip)
    if dur < 4.0:
        return None
    scale_pv = (f"scale={_PV_W}:{_PV_H}:force_original_aspect_ratio=increase,"
                f"crop={_PV_W}:{_PV_H},fps=25,format=yuv420p")

    def _tag_png(text: str) -> Path | None:
        # both transition excerpts come from the SAME clip — a corner tag
        # makes the A→B changeover readable even on a static shot
        try:
            from PIL import Image, ImageDraw
            from pipeline_v4.overlays import _font
            img = Image.new("RGBA", (150, 40), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle([0, 0, 149, 39], radius=8, fill=(10, 10, 14, 200))
            d.text((14, 8), text, font=_font(None, 22), fill=(255, 255, 255, 255))
            p = cache / f"_tag_{text.replace(' ', '')}.png"
            img.save(p, "PNG")
            return p
        except Exception:
            return None

    def cut(dst: Path, at: float, secs: float, tag: Path | None = None) -> bool:
        cmd = [_ffmpeg(), "-y", "-v", "error",
               "-ss", f"{max(0.0, at):.2f}", "-t", f"{secs:.2f}",
               "-i", str(clip)]
        if tag:
            cmd += ["-i", str(tag), "-filter_complex",
                    f"[0:v]{scale_pv}[v0];[v0][1:v]overlay=12:12[v]",
                    "-map", "[v]"]
        else:
            cmd += ["-vf", scale_pv]
        cmd += ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "27",
                "-movflags", "+faststart", str(dst)]
        return _run(cmd)

    ok = (cut(base["base"], dur * 0.30, 4.0)
          and cut(base["a"], dur * 0.12, 1.4, _tag_png("CLIP A"))
          and cut(base["b"], dur * 0.62, 1.4, _tag_png("CLIP B"))
          and _run([_ffmpeg(), "-y", "-v", "error",
                    "-ss", f"{dur * 0.40:.2f}", "-i", str(clip),
                    "-vf", f"scale={_FR_W}:{_FR_H}:force_original_aspect_ratio=increase,"
                           f"crop={_FR_W}:{_FR_H}",
                    "-frames:v", "1", str(base["frame"])]))
    return base if ok else None


# ─── Catalog ────────────────────────────────────────────────────────

# The catalog categories a user pick can actually DIRECT the AI Director with
# today (a pick here constrains the render). Other sections are shown for
# context but stay engine-decided. Keys here map to render vocab in
# pipeline_v4.director._DIRECTIVE_KEY_MAP (+ "taxonomy" → story category).
DIRECTABLE_SECTIONS = {"style_packs", "transitions", "frame_fx",
                       "overlays", "typography", "taxonomy",
                       "color_grades", "sound"}


def build_catalog_sections() -> list:
    """The full editing-engine catalog, grouped into sections. Shared by the
    admin catalog and the user-facing style picker (no auth here — callers
    gate access)."""
    from pipeline_v4.color_grades import grade_catalog
    from pipeline_v4.frame_fx import fx_catalog
    from pipeline_v4.overlays import REGISTRY as OVERLAYS
    from pipeline_v4.trailer_styles import (STYLES, SUBCATEGORY_ALIASES,
                                            TRANSITION_CATALOG)
    from pipeline_v4.typography import CATALOG as TYPO

    sections = []
    sections.append({
        "key": "style_packs",
        "label": "Style packs (look + sound identity per category)",
        "preview_kind": "style_pack",
        "items": [{"id": p.key, "label": p.label,
                   "used_for": f"pace {p.pace}s · boom {p.hit_freq}Hz · "
                               f"transitions: {', '.join(p.transitions[:3])}…"}
                  for p in STYLES.values()],
    })
    sections.append({
        "key": "transitions",
        "label": "Transitions (story-to-story / trailer cuts)",
        "preview_kind": "transition",
        "items": [{"id": t, "label": t, "used_for": "xfade transition"}
                  for t in TRANSITION_CATALOG],
    })
    sections.append({
        "key": "color_grades",
        "label": "Color grades / LUT looks",
        "preview_kind": "grade",
        "items": [{"id": g["key"], "label": g["label"],
                   "used_for": g["used_for"] + (" [LUT file]" if g["source"] == "lut" else "")}
                  for g in grade_catalog()],
    })
    sections.append({
        "key": "frame_fx",
        "label": "Frame effects (texture / motion / glitch)",
        "preview_kind": "fx",
        "items": [{"id": f["key"], "label": f["label"], "used_for": f["used_for"]}
                  for f in fx_catalog()],
    })
    sections.append({
        "key": "overlays",
        "label": "Overlays + HUD frames (banners, bugs, stamps, CCTV/viewfinder…)",
        "preview_kind": "overlay",
        "items": [{"id": r["id"], "label": r["label"], "used_for": r["used_for"]}
                  for r in OVERLAYS],
    })
    sections.append({
        "key": "typography",
        "label": "Typography / text animation",
        "preview_kind": "typography",
        "items": [{"id": t["id"], "label": t["label"], "used_for": t["used_for"]}
                  for t in TYPO],
    })
    from pipeline_v4.sound_library import sound_catalog
    sections.append({
        "key": "sound",
        "label": "Sound design (synthesized — we own every sample; drop-in overrides in assets/sfx)",
        "preview_kind": "sound",
        "items": [
            {"id": "hit@news", "label": "Impact boom (news)", "used_for": "Card reveals, headline hits"},
            {"id": "hit@horror", "label": "Impact boom (horror 45Hz)", "used_for": "Horror/crime deep boom"},
            {"id": "hit@comedy", "label": "Impact pop (comedy)", "used_for": "Light bright pop"},
            {"id": "whoosh@news", "label": "Whoosh (news pack)", "used_for": "Every cut / graphic slide-in"},
            {"id": "whoosh@crime", "label": "Whoosh (crime pack)", "used_for": "Tense cuts"},
            {"id": "riser@news", "label": "Tension riser (news pack)", "used_for": "Build into the finale"},
            {"id": "bed@horror", "label": "Mood bed — dark drone", "used_for": "Under a whole horror cut"},
            {"id": "bed@devotional", "label": "Mood bed — warm pad", "used_for": "Under devotional cuts"},
            {"id": "bleep", "label": "Censor bleep 1kHz", "used_for": "Profanity censoring (auto, word-exact)"},
        ] + [{"id": s["key"], "label": s["label"],
              "used_for": f"[{s['family']}] {s['used_for']}"}
             for s in sound_catalog()],
    })
    from pipeline_v4.layout_library import layout_catalog
    _lib = layout_catalog()
    sections.append({
        "key": "layouts",
        "label": "Layouts / screen designs (bg video · carousels · every zone a video needs)",
        "preview_kind": "layout",
        "items": [{"id": r["key"], "label": r["label"], "used_for": r["used_for"]}
                  for r in _lib if r["family"] != "pip"],
    })
    sections.append({
        "key": "pip_variants",
        "label": "Picture-in-picture variants",
        "preview_kind": "layout",
        "items": [{"id": r["key"], "label": r["label"], "used_for": r["used_for"]}
                  for r in _lib if r["family"] == "pip"],
    })
    from pipeline_v4.trailer_styles import structure_catalog
    sections.append({
        "key": "trailer_structures",
        "label": "Trailer structures (how the teaser is assembled)",
        "preview_kind": None,
        "items": [{"id": s["key"], "label": s["label"],
                   "used_for": s["used_for"]} for s in structure_catalog()],
    })
    sections.append({
        "key": "taxonomy",
        "label": "Story categories (~90 editorial subcategories → packs)",
        "preview_kind": None,
        "items": [{"id": sub, "label": sub.upper(), "used_for": f"→ {pack} pack"}
                  for sub, pack in sorted(SUBCATEGORY_ALIASES.items())],
    })
    return sections


@router.get("")
def catalog(_: models.User = Depends(auth.admin_required)) -> dict:
    sections = build_catalog_sections()
    return {"sections": sections,
            "totals": {s["key"]: len(s["items"]) for s in sections}}


# ─── Preview renderers ──────────────────────────────────────────────

def _chain_mp4(chain: str, out: Path, base: dict) -> bool:
    """The exact filter chain applied to real footage → short MP4."""
    graph = ";" in chain
    cmd = [_ffmpeg(), "-y", "-v", "error", "-i", str(base["base"])]
    cmd += (["-filter_complex", chain] if graph else ["-vf", chain])
    cmd += ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "27",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    return _run(cmd) and out.is_file()


def _testsrc_png(chain: str, out: Path) -> bool:
    """Clipless fallback: one frame of a test pattern through the chain."""
    graph = ";" in chain
    cmd = [_ffmpeg(), "-y", "-v", "error",
           "-f", "lavfi", "-i", f"testsrc2=s={_PV_W}x{_PV_H}:d=0.5:r=25"]
    cmd += (["-filter_complex", chain] if graph else ["-vf", chain])
    cmd += ["-frames:v", "1", str(out)]
    return _run(cmd, 60) and out.is_file()


def _chain_preview(chain: str, safe_id: str, prefix: str) -> dict:
    cache = _cache_dir()
    base = _ensure_base()
    if base:
        out = cache / f"{prefix}_{safe_id}_{base['fp']}.mp4"
        if out.is_file() or _chain_mp4(chain, out, base):
            return _resp(str(out), "video")
    out = cache / f"{prefix}_{safe_id}_tsrc.png"
    if out.is_file() or _testsrc_png(chain, out):
        return _resp(str(out), "image")
    raise HTTPException(500, f"{prefix} preview render failed")


def _transition_mp4(name: str, out: Path, base: dict | None) -> bool:
    from pipeline_v4.trailer_styles import xfade_arg
    if base:
        inputs = ["-i", str(base["a"]), "-i", str(base["b"])]
    else:
        inputs = ["-f", "lavfi", "-i", f"smptebars=s={_PV_W}x{_PV_H}:d=1.4:r=25",
                  "-f", "lavfi", "-i", f"testsrc2=s={_PV_W}x{_PV_H}:d=1.4:r=25"]
    fc = f"[0:v][1:v]xfade={xfade_arg(name)}:duration=0.6:offset=0.7[v]"
    cmd = [_ffmpeg(), "-y", "-v", "error", *inputs,
           "-filter_complex", fc, "-map", "[v]",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "27",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    return _run(cmd) and out.is_file()


def _pack_demo_mp4(pack, out: Path, base: dict) -> bool:
    """A style pack is a full editing identity — look + cutting rhythm +
    its own transitions + its own SOUND (boom/whoosh/bed) + card design.
    So its preview is a ~3.5s mini-trailer WITH AUDIO, not a color swatch:
    clip A (graded) → pack transition + whoosh → clip B (graded) →
    fade + boom → title card in the pack's colors, mood bed underneath."""
    from pipeline_v4.trailer import _render_card_png, ensure_sfx, synth_bed

    cache = _cache_dir()
    chain = pack.grade + ("," + pack.extra_vf if pack.extra_vf else "")
    if ";" in chain:               # demo graph needs linear per-clip chains
        chain = pack.grade

    card = cache / f"_card_{pack.key}.png"
    if not card.is_file() and not _render_card_png(
            text=pack.label.upper(), sub=pack.card_sub, w=_PV_W, h=_PV_H,
            font_path=None, out_path=str(card),
            bg=pack.card_bg, accent=pack.card_accent):
        return False
    sfx = ensure_sfx(cache, pack)
    whoosh, hit = sfx.get("whoosh"), sfx.get("hit")
    bed = synth_bed(cache, pack, 3.8)

    A, B, CARD = 1.4, 1.4, 1.5
    d1 = min(0.5, max(0.3, pack.pace * 1.5))
    o1 = A - d1                       # first transition start
    d2 = 0.35
    o2 = A + B - d1 - d2              # card reveal start
    trans = pack.transitions[0] if pack.transitions else "fadeblack"

    cmd = [_ffmpeg(), "-y", "-v", "error",
           "-i", str(base["a"]), "-i", str(base["b"]),
           "-loop", "1", "-t", f"{CARD}", "-i", str(card)]
    aud_inputs, aud_labels = [], []
    for pth, at_ms, vol in ((whoosh, int(o1 * 1000), "1.0"),
                            (hit, int(o2 * 1000), "1.0"),
                            (bed, 0, f"{pack.bed_gain:.2f}")):
        if pth and Path(pth).is_file():
            idx = 3 + len(aud_labels)          # inputs 0/1/2 are video
            aud_inputs += ["-i", str(pth)]
            lbl = f"au{idx}"
            aud_labels.append(
                (lbl, f"[{idx}:a]adelay={at_ms}:all=1,volume={vol}[{lbl}]"))
    # settb on every branch: the card comes from a looped PNG whose
    # timebase differs from the mp4 excerpts — xfade rejects mismatches
    fc = (f"[0:v]{chain},format=yuv420p,setsar=1,settb=AVTB[va];"
          f"[1:v]{chain},format=yuv420p,setsar=1,settb=AVTB[vb];"
          f"[2:v]scale={_PV_W}:{_PV_H},setsar=1,fps=25,format=yuv420p,settb=AVTB[vc];"
          f"[va][vb]xfade=transition={trans}:duration={d1}:offset={o1:.2f}[vab];"
          f"[vab][vc]xfade=transition=fadeblack:duration={d2}:offset={o2:.2f}[v]")
    maps = ["-map", "[v]"]
    if aud_labels:
        fc += ";" + ";".join(expr for _, expr in aud_labels)
        # apad then trim: without a mood bed the mix ends at the last SFX
        # and -shortest would chop the title card off the video
        fc += (";" + "".join(f"[{lbl}]" for lbl, _ in aud_labels)
               + f"amix=inputs={len(aud_labels)}:normalize=0,apad,"
                 f"atrim=0:{A + B + CARD - d1 - d2:.2f}[a]")
        maps += ["-map", "[a]"]
    cmd += [*aud_inputs, "-filter_complex", fc, *maps,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "27",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", "-shortest", str(out)]
    return _run(cmd, 180) and out.is_file()


# where each overlay family sits when composited on a real frame
# (fractions of frame W/H for the element's top-left; None = centered)
_OVERLAY_POS = {
    "banner": ("full_width", 0.78),
    "bug": (0.80, 0.06),
    "stamp": (None, 0.28),
    "locator": (0.03, 0.05),
    "attribution": (0.03, 0.90),
    "progress": (0.33, 0.05),
    "countdown": ("cover", None),
    "frame_hud": ("cover", None),
    "info_panel": (0.55, 0.22),
    "score_bug": (0.03, 0.06),
    "weather_chip": (0.03, 0.06),
    "market_chip": (0.03, 0.06),
    "poll_bar": (0.12, 0.60),
    "social_card": (0.06, 0.30),
    "headline_tag": (0.05, 0.66),
}


def _composite_on_frame(png_in: Path, family: str, out: Path,
                        base: dict) -> bool:
    from PIL import Image
    try:
        frame = Image.open(base["frame"]).convert("RGBA")
        el = Image.open(png_in).convert("RGBA")
        fw, fh = frame.size
        rule = _OVERLAY_POS.get(family, (None, None))
        if rule[0] == "cover":
            el = el.resize((fw, fh))
            frame.alpha_composite(el)
        elif rule[0] == "full_width":
            r = fw / el.width
            el = el.resize((fw, max(1, int(el.height * r))))
            frame.alpha_composite(el, (0, int(fh * rule[1])))
        else:
            # element at ~native scale relative to a 1920-wide canvas
            r = fw / 1920.0
            el = el.resize((max(1, int(el.width * r)),
                            max(1, int(el.height * r))))
            x = int(fw * rule[0]) if rule[0] is not None else (fw - el.width) // 2
            y = int(fh * rule[1]) if rule[1] is not None else (fh - el.height) // 2
            frame.alpha_composite(el, (max(0, x), max(0, y)))
        frame.convert("RGB").save(out, "PNG")
        return True
    except Exception:
        return False


def _layout_png(key: str, out: Path) -> bool:
    from PIL import Image, ImageDraw
    from pipeline_v4.overlays import _font
    vertical = key in ("torn_card", "clean_card", "split_frame",
                       "follow_bar", "dual_video")
    W, H = (152, 270) if vertical else (480, 270)
    img = Image.new("RGB", (W, H), (12, 12, 16))
    d = ImageDraw.Draw(img)
    f = _font(None, 13)

    def box(x, y, w, h, label, color=(60, 130, 220)):
        d.rectangle([x, y, x + w, y + h], outline=color, width=2)
        d.text((x + 5, y + 4), label, font=f, fill=(230, 230, 235))

    if key == "bulletin":
        box(12, 22, 300, 180, "VIDEO")
        box(324, 22, 140, 105, "IMAGES", (220, 160, 40))
        box(0, 212, W, 30, "HEADLINE STRAP", (200, 40, 40))
        box(0, 246, W, 22, "TICKER", (220, 200, 40))
    elif key == "audio_fullscreen":
        box(0, 0, W - 1, 208, "FULL-FRAME IMAGE", (220, 160, 40))
        box(0, 212, W, 30, "HEADLINE STRAP", (200, 40, 40))
        box(0, 246, W, 22, "TICKER", (220, 200, 40))
    elif key in ("torn_card", "clean_card"):
        box(0, 0, W - 1, 120, "VIDEO")
        box(0, 124, W - 1, 50, "HEADLINE", (200, 40, 40))
        box(0, 178, W - 1, 90, "IMAGE", (220, 160, 40))
    elif key == "split_frame":
        box(0, 0, W - 1, 132, "VIDEO")
        box(0, 136, W - 1, 132, "HERO IMAGE", (220, 160, 40))
    elif key == "follow_bar":
        box(0, 0, W - 1, 150, "VIDEO")
        box(0, 154, W - 1, 60, "HEADLINE", (200, 40, 40))
        box(0, 218, W - 1, 50, "FOLLOW CTA", (160, 60, 220))
    elif key == "dual_video":
        box(0, 0, W - 1, 120, "CAM A")
        box(0, 124, W - 1, 22, "TITLE", (200, 40, 40))
        box(0, 150, W - 1, 118, "CAM B")
    elif key.startswith("grid_"):
        from pipeline_v4.grid_engine import plan_grid
        n = int(key.split("_")[1])
        pages = plan_grid(n, duration=10.0, canvas_w=1920, canvas_h=1080)
        sx = W / 1920.0
        sy = H / 1080.0
        for i, (x, y, w, h) in enumerate(pages[0].cells):
            box(int(x * sx), int(y * sy), int(w * sx), int(h * sy),
                f"SPK {i + 1}")
    elif key == "pip_corner":
        box(0, 0, W - 1, H - 1, "MAIN VIDEO")
        box(W - 130, 16, 110, 62, "INSET", (220, 160, 40))
    elif key == "spotlight":
        box(0, 0, W - 1, H - 1, "IMAGE FULL-SCREEN", (220, 160, 40))
        box(0, 240, W, 28, "TEXT STAYS ON TOP", (200, 40, 40))
    elif key == "trailer":
        box(0, 0, 92, H - 1, "HOOK", (200, 40, 40))
        box(96, 0, 130, H - 1, "MOMENTS")
        box(230, 0, 130, H - 1, "MOMENTS")
        box(364, 0, 114, H - 1, "END CARD", (200, 40, 40))
    elif key == "custom_html":
        box(0, 0, W - 1, H - 1, "ANY LAYOUT (HTML/CSS)", (160, 60, 220))
    else:
        return False
    img.save(out, "PNG")
    return True


def _typography_png(item_id: str, out_dir: Path, out: Path) -> bool:
    from pipeline_v4 import typography as ty
    tmp = out_dir / "_ty_variants"
    tmp.mkdir(parents=True, exist_ok=True)
    pick = ty.render_variant(item_id, tmp)
    if not pick:
        return False
    base = _ensure_base()
    if base and _composite_on_frame(Path(pick), "banner", out, base):
        return True
    import shutil
    shutil.copyfile(pick, out)
    return True


def _double_play(src: str, out: Path) -> str:
    """effect · gap · effect — one-shots too short to judge get played
    twice (operator: 'in 1s preview I am not able to hear it')."""
    if out.is_file():
        return str(out)
    fc = ("[0:a]apad=pad_dur=0.6[p];[p][1:a]concat=n=2:v=0:a=1,"
          "apad=pad_dur=0.4[a]")
    ok = _run([_ffmpeg(), "-y", "-v", "error", "-i", src, "-i", src,
               "-filter_complex", fc, "-map", "[a]",
               "-ar", "48000", "-ac", "1", str(out)], 60)
    return str(out) if (ok and out.is_file()) else src


def _sound_wav(item_id: str, out_dir: Path) -> str | None:
    from pipeline_v4.sound_library import LIBRARY, preview_sound
    from pipeline_v4.trailer import ensure_sfx, synth_bed
    from pipeline_v4.trailer_styles import get_style
    if item_id in LIBRARY:
        return preview_sound(item_id, out_dir)   # already hearable-length
    if item_id == "bleep":
        dst = out_dir / "_preview_bleep.wav"
        if not dst.is_file():
            ok = _run([_ffmpeg(), "-y", "-v", "error", "-f", "lavfi",
                       "-i", "sine=frequency=1000:duration=1.0",
                       "-af", "volume=0.5", "-ar", "48000", str(dst)], 60)
            if not ok:
                return None
        return _double_play(str(dst), out_dir / "_prev2_bleep.wav")
    kind, _, pack = item_id.partition("@")
    style = get_style(pack or "news")
    if kind == "bed":
        return synth_bed(out_dir, style, 6.0)
    sfx = ensure_sfx(out_dir, style)
    p = sfx.get(kind)
    if p and kind in ("hit", "whoosh"):
        return _double_play(p, out_dir / f"_prev2_{style.key}_{kind}.wav")
    return p


def render_item_preview(kind: str, id: str) -> dict:
    """Render (once) and return the dummy for one catalog item. Read-only +
    cached; shared by the admin catalog and the user style picker."""
    from pipeline_v4.color_grades import GRADES, get_grade_vf
    from pipeline_v4.frame_fx import FX, get_fx_vf
    from pipeline_v4.overlays import REGISTRY, render_registry_item
    from pipeline_v4.trailer_styles import STYLES, TRANSITION_CATALOG

    cache = _cache_dir()
    safe_id = "".join(c if (c.isalnum() or c in "@_-") else "_" for c in id)[:60]

    if kind == "grade":
        if id not in GRADES:
            raise HTTPException(404, f"unknown grade {id!r}")
        return _chain_preview(get_grade_vf(id), safe_id, "grade")
    if kind == "fx":
        if id not in FX:
            raise HTTPException(404, f"unknown effect {id!r}")
        # preview-sized params where the fragment hardcodes output dims
        params = ({"w": _PV_W, "h": _PV_H}
                  if "w" in FX[id].defaults else {})
        return _chain_preview(get_fx_vf(id, **params), safe_id, "fx")
    if kind == "style_pack":
        if id not in STYLES:
            raise HTTPException(404, f"unknown pack {id!r}")
        p = STYLES[id]
        base = _ensure_base()
        if base:
            out = cache / f"packdemo_{safe_id}_{base['fp']}.mp4"
            if out.is_file() or _pack_demo_mp4(p, out, base):
                return _resp(str(out), "video_audio")
        # fallback: at least show the look
        chain = p.grade + ("," + p.extra_vf if p.extra_vf else "")
        return _chain_preview(chain, safe_id, "pack")
    if kind == "transition":
        if id not in TRANSITION_CATALOG:
            raise HTTPException(404, f"unknown transition {id!r}")
        base = _ensure_base()
        fp = base["fp"] if base else "tsrc"
        out = cache / f"trans_{safe_id}_{fp}.mp4"
        if not out.is_file() and not _transition_mp4(id, out, base):
            raise HTTPException(500, "transition preview render failed")
        return _resp(str(out), "video")
    if kind == "overlay":
        row = next((r for r in REGISTRY if r["id"] == id), None)
        if not row:
            raise HTTPException(404, f"unknown overlay {id!r}")
        base = _ensure_base()
        fp = base["fp"] if base else "raw"
        out = cache / f"overlay_{safe_id}_{fp}.png"
        if not out.is_file():
            el = cache / f"overlay_{safe_id}_el.png"
            if not el.is_file() and not render_registry_item(id, str(el)):
                raise HTTPException(500, "overlay render failed")
            if not (base and _composite_on_frame(el, row["family"], out, base)):
                out = el  # clipless fallback: the element itself
        return _resp(str(out), "image")
    if kind == "layout":
        from pipeline_v4.layout_library import render_layout_preview
        out = cache / f"layout_{safe_id}.png"
        if not out.is_file():
            ok = render_layout_preview(id, str(out)) or _layout_png(id, out)
            if not ok:
                raise HTTPException(404, f"unknown layout {id!r}")
        return _resp(str(out), "image")
    if kind == "typography":
        base = _ensure_base()
        fp = base["fp"] if base else "raw"
        out = cache / f"typo_{safe_id}_{fp}.png"
        if not out.is_file() and not _typography_png(id, cache, out):
            raise HTTPException(404, f"unknown typography {id!r}")
        return _resp(str(out), "image")
    if kind == "sound":
        p = _sound_wav(id, cache)
        if not p:
            raise HTTPException(404, f"unknown sound {id!r}")
        return _resp(p, "audio")
    raise HTTPException(400, f"unknown preview kind {kind!r}")


@router.get("/preview")
def preview(kind: str, id: str,
            _: models.User = Depends(auth.admin_required)) -> dict:
    return render_item_preview(kind, id)


# ─── User-facing style picker ───────────────────────────────────────
# The same catalog, but any signed-in user can browse it to DIRECT the AI
# ("edit using THESE") at job creation. Only the directable sections are
# offered for selection; the rest ride along read-only for context. Wired to
# Job.v4_style_directives → KAIZER_V4_STYLE_DIRECTIVES → the Director's vocab.

user_router = APIRouter(prefix="/api/v4/style-catalog",
                        tags=["v4-style-catalog"])


# Directable section → the Director vocab bucket a pick maps to. Items are
# intersected with the real vocab so the picker NEVER shows an id that would
# silently no-op (e.g. text-animation typography that isn't a caption style).
# "taxonomy" has no bucket — every story-category resolves, so it's kept whole.
_SECTION_BUCKET = {"style_packs": "packs", "transitions": "transitions",
                   "frame_fx": "fx", "overlays": "overlays",
                   "typography": "captions",
                   # Newly directable: a color-grade pick constrains the
                   # Director's "grades" bucket; a sound pick constrains its
                   # "stings" bucket (both via _DIRECTIVE_KEY_MAP entries
                   # "color_grades"/"sound" — the section keys MUST match).
                   "color_grades": "grades", "sound": "stings"}
# User-facing label overrides where the admin label reads oddly after the
# directable-only filter (typography → only caption styles remain).
_USER_LABEL = {
    "typography": "Caption style (word-by-word, for shorts/reels)",
    "taxonomy": "Story category (what kind of story — sets the base look)",
    # Admin label talks about drop-in override paths — dev-speak; keep the
    # user picker plain.
    "sound": "Sound design (impact hits, whooshes, risers)",
}


@user_router.get("")
def style_catalog(directable_only: bool = False,
                  _: models.User = Depends(auth.current_user)) -> dict:
    """The catalog for the user picker. Each section is tagged ``directable``
    (a user pick actually constrains the render) so the UI can offer those for
    selection and show the rest as engine-decided. ``directable_only=1`` drops
    the context-only sections entirely. Directable sections are filtered to the
    ids the Director can truly consume, so every shown option really directs
    the render."""
    from pipeline_v4.director import _vocab
    vocab = _vocab()
    sections = []
    for s in build_catalog_sections():
        d = s["key"] in DIRECTABLE_SECTIONS
        if directable_only and not d:
            continue
        s = {**s, "directable": d}
        bucket = _SECTION_BUCKET.get(s["key"])
        if bucket:
            # Fail-soft: intersect only when the Director's vocab actually
            # exposes this bucket. If it doesn't (yet), show the raw catalog
            # instead of an empty section — picks then simply stay
            # engine-decided rather than the picker looking broken.
            allow = vocab.get(bucket)
            if allow is not None:
                s["items"] = [it for it in s["items"] if it["id"] in allow]
        if s["key"] in _USER_LABEL:
            s["label"] = _USER_LABEL[s["key"]]
        sections.append(s)
    return {"sections": sections,
            "totals": {s["key"]: len(s["items"]) for s in sections},
            "directable_keys": sorted(DIRECTABLE_SECTIONS)}


@user_router.get("/preview")
def style_preview(kind: str, id: str,
                  _: models.User = Depends(auth.current_user)) -> dict:
    """CapCut-style preview of one catalog item for the user picker (read-only,
    cached — same renderer the admin catalog uses)."""
    return render_item_preview(kind, id)
