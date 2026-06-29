"""High-level orchestration for custom templates: prepare a bundle, render a video.

Thin glue over bundle / contract / renderer / compose so callers (the API, the render
pipeline, the standalone test) have one entry point.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field


def _atomic_write(path: str, text: str) -> None:
    """Write text to path atomically (temp file + os.replace) so a concurrent reader/render
    never sees a half-written normalized template."""
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass

from . import bundle as _bundle
from . import compose as _compose
from . import contract as _contract
from . import infer as _infer
from . import renderer as _renderer


def _safe_bundle_path(root: str, rel: str) -> str | None:
    """Resolve a template-declared default (data-kaizer-default) to an absolute path that
    is provably UNDER the bundle root + an allowed media type. Returns None if it is
    absolute, escapes via '..'/symlink, or has a disallowed extension — preventing a
    malicious default like '../../etc/passwd' from being fed to ffmpeg."""
    try:
        if not rel or os.path.isabs(rel) or rel.startswith(("/", "\\")):
            return None
        if os.path.splitext(rel)[1].lower() not in _bundle.ALLOWED_EXT:
            return None
        root_r = os.path.realpath(root)
        cand = os.path.realpath(os.path.join(root, rel))
        if cand == root_r or cand.startswith(root_r + os.sep):
            return cand
    except Exception:
        pass
    return None


def _neutralize_css_files(b: "_bundle.Bundle") -> int:
    """Blank external url()/@import in every .css file in the bundle (defense in depth)."""
    removed = 0
    for rel in b.files:
        if rel.lower().endswith(".css"):
            p = os.path.join(b.root_dir, rel)
            try:
                with open(p, encoding="utf-8", errors="replace") as fh:
                    css = fh.read()
                clean, n = _bundle.neutralize_external_css(css)
                if n:
                    with open(p, "w", encoding="utf-8") as fh:
                        fh.write(clean)
                    removed += n
            except Exception:
                continue
    return removed


def prepare_bundle(src_path: str, dest_dir: str) -> tuple["_bundle.Bundle", "_contract.TemplateContract"]:
    """Extract + sanitize an uploaded template and discover its contract.
    Writes the sanitized entry HTML back to disk so the renderer loads local-only."""
    b = _bundle.extract_bundle(src_path, dest_dir)
    # Sanitize EVERY html file in the bundle (not just the entry) — a sibling page the
    # entry could navigate to must also be script/external-free.
    entry_norm = (b.entry_rel or "").replace("\\", "/").lower()
    entry_clean = ""
    for rel in b.files:
        if not rel.lower().endswith((".html", ".htm")):
            continue
        p = os.path.join(b.root_dir, rel)
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                raw_html = fh.read()
            clean_html, _ = _bundle.neutralize_external_html(raw_html)
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(clean_html)
            if rel.replace("\\", "/").lower() == entry_norm:
                entry_clean = clean_html
        except Exception:
            continue
    if not entry_clean:
        with open(b.entry_path, encoding="utf-8", errors="replace") as fh:
            entry_clean = fh.read()
    _neutralize_css_files(b)
    # Keep the raw sanitized entry (no injected markers) as a sidecar so the AI-understanding
    # pass can reason about the author's ORIGINAL layout. Best-effort.
    try:
        with open(os.path.join(b.root_dir, ".kaizer_src.html"), "w", encoding="utf-8") as _sf:
            _sf.write(entry_clean)
    except Exception:
        pass
    # Tolerant inference: if the author used id/class/semantic tags / other attributes
    # instead of data-kaizer (or no markers at all), inject synthetic markers + infer the
    # canvas, and write the normalized entry back so the renderer + filler understand it.
    try:
        norm_html, contract = _infer.normalize_and_discover(entry_clean)
        if norm_html and norm_html != entry_clean:
            _atomic_write(b.entry_path, norm_html)
    except Exception:
        contract = _contract.discover(entry_clean)   # never fail the upload on inference
    return b, contract


@dataclass
class RenderRequest:
    """Inputs to fill a prepared template."""
    videos: dict[str, str] = field(default_factory=dict)    # slot key -> clip path
    texts: dict[str, str] = field(default_factory=dict)     # slot key -> text
    images: dict[str, str] = field(default_factory=dict)    # slot key -> image path
    logo_path: str | None = None
    brand: dict[str, str] = field(default_factory=dict)     # css var -> value
    fps: int = 30
    duration: float | None = None
    intro_path: str | None = None      # cold-open clip prepended before the content
    main_slot: str = ""                # which video slot drives audio (the AI-trim main)
    # LITERAL mode (per-job HTML override): the HTML already has the operator's final text/
    # images baked into the DOM (edited in the visual builder for THIS job). Render it AS-IS
    # — do NOT clear/fill text or image slots — and only composite the video slot. Leaves
    # logo/watermark slots empty so the per-channel publish stamp still drops each channel's
    # own brand at the marked spot. clear_unfilled flips OFF so baked content is preserved.
    literal: bool = False
    # SCROLLING TICKER (still renders): when the template has a ticker/marquee slot, a marquee
    # strip is composited over the finished MP4 at that slot. text=None -> use the slot's own
    # text; speed_s = seconds per loop (None -> 200 px/s); bg_color/font_px/lang override the
    # strip styling (None -> match the slot's bg + auto-size the font to the slot height).
    ticker_text: str | None = None
    ticker_speed_s: float | None = None
    ticker_bg_color: str | None = None
    ticker_font_px: int | None = None
    ticker_lang: str | None = None


def _map_videos(video_rects: list[dict], videos: dict[str, str],
                primary: str | None = None) -> list["_compose.Placement"]:
    """Attach a clip path to each measured video/background rect. Falls back to the
    primary clip when a slot has no explicit media (so an unfilled background reuses the
    main clip — 'both the same')."""
    if primary is None and videos:
        primary = videos.get("video") or next(iter(videos.values()))
    placements: list[_compose.Placement] = []
    for r in video_rects:
        path = videos.get(r["key"]) or videos.get("video") or primary
        if not path:
            continue
        placements.append(_compose.Placement(
            x=r["x"], y=r["y"], w=r["w"], h=r["h"], video=path,
            background=(r.get("kind") == "background")))
    return placements


def render_template(bundle: "_bundle.Bundle", contract: "_contract.TemplateContract",
                    req: RenderRequest, work_dir: str, out_path: str) -> dict:
    """Render the prepared template with ``req`` to ``out_path``. Returns a small report."""
    os.makedirs(work_dir, exist_ok=True)
    canvas = (contract.canvas_w, contract.canvas_h)

    # Merge in the template's BUNDLED DEFAULTS for any slot the caller didn't fill
    # (so an uploaded template's default bg/intro/image is used unless swapped).
    videos = dict(req.videos or {})
    images = dict(req.images or {})
    intro = req.intro_path
    for s in contract.slots:
        if s.default:
            p = _safe_bundle_path(bundle.root_dir, s.default)
            if p and os.path.isfile(p):
                if s.kind in ("video", "background"):
                    videos.setdefault(s.key, p)
                elif s.kind == "image":
                    images.setdefault(s.key, p)
                elif s.kind == "intro" and not intro:
                    intro = p

    # Custom renders are STILL by default: a 1080p ANIMATED capture (CDP virtual-time +
    # hundreds of screenshots) peaks ~900 MB and is slow/unstable — over the operator's
    # 500 MB budget. A still overlay (~140 MB, seconds) is the stable default. Opt into
    # motion with KAIZER_TEMPLATE_ANIMATE=1 (CSS animations/ticker scroll, higher RAM).
    _animate = os.environ.get("KAIZER_TEMPLATE_ANIMATE", "0").strip().lower() in ("1", "true", "yes")
    _still = not _animate

    # Carousel image slots: their value in `images` is a {carousel:[...]} spec, NOT a path. Keep
    # them out of the still-fill (the slideshow is composited post-render) and tell the renderer
    # to punch a transparent hole at each so the slideshow shows through.
    _carousel_keys = [k for k, v in images.items() if isinstance(v, dict) and v.get("carousel")]
    _images_for_fill = {k: v for k, v in images.items() if k not in _carousel_keys}

    data = _renderer.RenderData(
        texts=req.texts, images=_images_for_fill, logo_path=req.logo_path, brand=req.brand,
        # final render: never leak an author placeholder — EXCEPT in literal/override mode,
        # where the HTML already carries the operator's baked text/images for this job and
        # must render verbatim (clearing would wipe what they typed in the builder).
        clear_unfilled=not req.literal,
        # still mode: hide the ticker text — a SCROLLING strip is composited post-render so
        # the crawl actually moves (a still frame can't scroll). Animated mode scrolls natively.
        hide_ticker=_still,
        carousel_keys=_carousel_keys,
    )

    # The "main" clip drives audio + capture duration.
    primary = (videos.get(req.main_slot) or videos.get("video")
               or (next(iter(videos.values())) if videos else None))
    cap_dur = req.duration or (_compose.probe_duration(primary) if primary else 0.0) or 6.0
    frames_dir = os.path.join(work_dir, "frames")
    ar = _renderer.render_animated(bundle, data, frames_dir, canvas,
                                   fps=req.fps, duration=cap_dur, force_still=_still)

    placements = _map_videos(ar.video_rects, videos, primary=primary)
    # Main clip first (non-background) so compose takes its audio.
    if placements and primary:
        placements.sort(key=lambda pl: (1 if pl.background else 0, 0 if pl.video == primary else 1))

    design = ({"frames_dir": ar.frames_dir, "pattern": ar.frame_pattern, "fps": ar.fps}
              if ar.animated and ar.frame_count > 1
              else os.path.join(ar.frames_dir, ar.frame_pattern % 0))

    # ELEMENT ENTRANCES (still mode): crop each animated element out of the still design + punch
    # its hole, so it can be re-composited with its entrance motion (no static copy underneath).
    _elem_overlays: list[dict] = []
    if _still and getattr(ar, "anim_rects", None) and isinstance(design, str) and os.path.isfile(design):
        try:
            from PIL import Image as _PILImage
            _img = _PILImage.open(design).convert("RGBA")
            for _i, _a in enumerate(ar.anim_rects):
                _x, _y, _w, _h = int(_a["x"]), int(_a["y"]), int(_a["w"]), int(_a["h"])
                _crop = _img.crop((_x, _y, _x + _w, _y + _h))
                _cp = os.path.join(work_dir, f"_anim_{_i}.png")
                _crop.save(_cp)
                _img.paste((0, 0, 0, 0), (_x, _y, _x + _w, _y + _h))   # punch the element's hole
                _elem_overlays.append({"path": _cp, "x": _x, "y": _y, "w": _w, "h": _h,
                                       "anim": _a.get("anim") or "fade",
                                       "dur": float(_a.get("dur") or 0.5)})
            _img.save(design)
        except Exception:
            _elem_overlays = []   # fall back: leave the elements baked (no entrance)

    _compose.compose(design, placements, out_path, canvas=canvas,
                     fps=req.fps, duration=req.duration)

    # SCROLLING TICKER: composite a marquee strip over the finished still at the ticker slot's
    # rect — efficient ffmpeg overlay (one re-encode pass, NO per-frame screenshots, so it
    # works on a 12-min bulletin). Applied BEFORE the intro prepend so the crawl never plays
    # over the cold-open. Only in still mode (animated scrolls natively); never fails the render.
    _ticker_done = False
    if _still:
        tr = getattr(ar, "ticker_rect", None)
        t_text = (req.ticker_text or (tr or {}).get("text") or "").strip()
        if tr and t_text:
            try:
                _overlay_ticker_on_rect(
                    out_path, rect=tr, text=t_text, work_dir=work_dir, fps=req.fps,
                    speed_s=req.ticker_speed_s,
                    bg_color=(req.ticker_bg_color or tr.get("bg")),
                    font_px=req.ticker_font_px, lang=req.ticker_lang,
                )
                _ticker_done = True
            except Exception as _te:
                try:
                    ar.warnings.append(f"ticker overlay skipped: {str(_te)[:200]}")
                except Exception:
                    pass

    # IMAGE CAROUSEL: composite a timed slideshow at each carousel slot's rect — looped stills +
    # crossfade/effects via ONE ffmpeg pass (no per-frame screenshots, works on long videos).
    # Applied before the intro prepend; never fails the whole render.
    _carousel_done = 0
    for _cr in (getattr(ar, "carousel_rects", None) or []):
        _spec = images.get(_cr.get("key"))
        if not (isinstance(_spec, dict) and _spec.get("carousel")):
            continue
        try:
            _overlay_carousel_on_rect(out_path, rect=_cr, spec=_spec, work_dir=work_dir,
                                      fps=req.fps, total_dur=cap_dur)
            _carousel_done += 1
        except Exception as _ce:
            try:
                ar.warnings.append(f"carousel overlay skipped ({_cr.get('key')}): {str(_ce)[:200]}")
            except Exception:
                pass

    # ELEMENT ENTRANCES: re-composite each cropped element with its entrance motion (fade/slide
    # in over the first `dur`, then static) — fills the punched hole. One ffmpeg pass; never fails.
    _anim_done = 0
    if _elem_overlays:
        try:
            _overlay_element_entrances(out_path, elems=_elem_overlays, work_dir=work_dir, fps=req.fps)
            _anim_done = len(_elem_overlays)
        except Exception as _ae:
            try:
                ar.warnings.append(f"element entrances skipped: {str(_ae)[:200]}")
            except Exception:
                pass

    # Intro cold-open: prepend the intro clip (override or bundled default).
    if intro and os.path.isfile(intro):
        tmp_main = out_path + ".main.mp4"
        try:
            os.replace(out_path, tmp_main)
            _compose.prepend_intro(intro, tmp_main, out_path, canvas, fps=req.fps)
        finally:
            try:
                os.remove(tmp_main)
            except Exception:
                pass

    return {
        "out": out_path,
        "canvas": [contract.canvas_w, contract.canvas_h],
        "video_slots": len(ar.video_rects),
        "placements": len(placements),
        "animated": ar.animated,
        "frames": ar.frame_count,
        "intro": bool(intro and os.path.isfile(out_path)),
        "ticker": bool(_ticker_done),
        "carousels": int(_carousel_done),
        "anim_entrances": int(_anim_done),
        "warnings": list(contract.warnings) + list(ar.warnings),
        # logo/watermark slot rects (in canvas px) so the per-channel publish stamp can
        # place each channel's own logo/watermark AT the template's marked spot.
        "brand_rects": dict(getattr(ar, "brand_rects", {}) or {}),
    }


def _overlay_carousel_on_rect(out_path: str, *, rect: dict, spec: dict, work_dir: str,
                              fps: int = 30, total_dur=None) -> None:
    """Composite a timed image SLIDESHOW at ``rect`` over the finished MP4, in place.

    ``spec`` = {carousel:[{path,duration_s,effect,effect_duration}], ...}. Each frame is a looped
    still shown from its start time onward (newest painted on top), so a fade-in on the next
    frame crossfades over the previous one; the last frame holds to the end. ONE ffmpeg pass —
    looped still inputs + overlay enable='gte(t,start)', no per-frame screenshots, so it runs on
    a 12-minute video. Stage 3 supports fade (crossfade) + cut; slide/zoom are added in Stage 4.
    Raises on failure (the caller swallows it so a carousel glitch never loses the whole render)."""
    import os as _os
    import shutil as _sh
    import subprocess as _sp
    rx, ry = int(rect["x"]), int(rect["y"])
    rw, rh = int(rect["w"]), int(rect["h"])
    frames = [f for f in (spec.get("carousel") or [])
              if isinstance(f, dict) and f.get("path") and _os.path.isfile(str(f.get("path")))]
    if rw < 8 or rh < 8 or not frames:
        return
    cap = float(total_dur) if total_dur else None
    starts: list[float] = []
    t = 0.0
    for f in frames:
        if cap is not None and t > cap - 0.4:      # never push a frame past the video end
            t = max(0.0, cap - 0.5)
        starts.append(round(t, 3))
        t += max(0.5, float(f.get("duration_s") or 3.0))

    ff = _sh.which("ffmpeg") or "ffmpeg"
    _dur = (_compose.probe_duration(out_path) or (cap or 6.0))
    tmp = out_path + ".carousel.mp4"

    def _build_fc(effects_on: bool):
        """filter_complex for the slideshow. effects_on=False -> hard cuts (the fail-soft path).
        \\, escapes commas inside ffmpeg expressions (gte/max/min), same as the ticker overlay."""
        _fc: list[str] = []
        for i, f in enumerate(frames):
            eff = ((f.get("effect") or "fade").strip().lower()) if effects_on else "cut"
            ed = max(0.0, min(float(f.get("effect_duration") or 0.4), 2.0))
            dur_i = max(0.5, float(f.get("duration_s") or 3.0))
            if eff == "zoom_in":
                zf = max(2, int(round((dur_i + 1.0) * fps)))
                chain = (f"[{i+1}:v]scale={rw*2}:{rh*2}:force_original_aspect_ratio=increase,"
                         f"crop={rw*2}:{rh*2},zoompan=z='min(zoom+0.0010\\,1.4)':d={zf}:"
                         f"s={rw}x{rh}:fps={fps},setsar=1,format=yuva420p")
                if ed > 0.05:
                    chain += f",fade=t=in:st={starts[i]:.3f}:d={ed:.3f}:alpha=1"
            else:
                chain = (f"[{i+1}:v]scale={rw}:{rh}:force_original_aspect_ratio=increase,"
                         f"crop={rw}:{rh},setsar=1,format=yuva420p")
                if eff == "fade" and ed > 0.05:
                    chain += f",fade=t=in:st={starts[i]:.3f}:d={ed:.3f}:alpha=1"
            _fc.append(chain + f"[c{i}]")
        _prev = "0:v"
        for i, f in enumerate(frames):
            eff = ((f.get("effect") or "fade").strip().lower()) if effects_on else "cut"
            ed = max(0.1, min(float(f.get("effect_duration") or 0.4), 2.0))
            st = starts[i]
            nxt = f"o{i}"
            if eff == "slide_left":     # enters from the slot's right edge, settles at rx
                xe = f"'max({rx}\\,{rx+rw}-{rw}*(t-{st:.3f})/{ed:.3f})'"
                _fc.append(f"[{_prev}][c{i}]overlay=x={xe}:y={ry}:enable='gte(t\\,{st:.3f})'[{nxt}]")
            elif eff == "slide_right":  # enters from the slot's left edge, settles at rx
                xe = f"'min({rx}\\,{rx-rw}+{rw}*(t-{st:.3f})/{ed:.3f})'"
                _fc.append(f"[{_prev}][c{i}]overlay=x={xe}:y={ry}:enable='gte(t\\,{st:.3f})'[{nxt}]")
            else:
                _fc.append(f"[{_prev}][c{i}]overlay={rx}:{ry}:enable='gte(t\\,{st:.3f})'[{nxt}]")
            _prev = nxt
        return _fc, _prev

    def _run(effects_on: bool):
        _fc, _prev = _build_fc(effects_on)
        _cmd = [ff, "-y", "-i", out_path]
        for f in frames:
            _cmd += ["-loop", "1", "-i", str(f["path"])]
        _cmd += ["-filter_complex", ";".join(_fc), "-map", f"[{_prev}]", "-map", "0:a?",
                 "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                 "-c:a", "copy", "-movflags", "+faststart", "-t", f"{_dur:.3f}", tmp]
        return _sp.run(_cmd, capture_output=True, timeout=1800)

    r = _run(True)
    if r.returncode != 0 or not _os.path.isfile(tmp) or _os.path.getsize(tmp) < 1024:
        # fail-soft: a transition glitch must NEVER lose the whole render — retry with hard cuts.
        r = _run(False)
        if r.returncode != 0 or not _os.path.isfile(tmp) or _os.path.getsize(tmp) < 1024:
            err = (r.stderr[-500:].decode("utf-8", "ignore") if r.stderr else "carousel ffmpeg failed")
            raise RuntimeError(err)
    _os.replace(tmp, out_path)


def _overlay_element_entrances(out_path: str, *, elems: list, work_dir: str, fps: int = 30) -> None:
    """Re-composite each cropped element (already its own w×h PNG) with an ENTRANCE — fade or
    slide in over the first ``dur`` seconds, then static — filling the hole punched in the design.
    ONE ffmpeg pass. Fail-soft: on error, retries with plain static overlays (elements still
    appear, just without motion) so a punched hole is never left empty. Raises only if even that
    fails. \\, escapes commas inside the slide max()/min() expressions for the filtergraph parser."""
    import os as _os
    import shutil as _sh
    import subprocess as _sp
    elems = [e for e in (elems or []) if e and e.get("path") and _os.path.isfile(str(e.get("path")))]
    if not elems:
        return
    ff = _sh.which("ffmpeg") or "ffmpeg"
    _dur = _compose.probe_duration(out_path) or 6.0
    tmp = out_path + ".anim.mp4"

    def _build(effects_on: bool):
        _fc = []
        for i, e in enumerate(elems):
            anim = ((e.get("anim") or "fade").strip().lower()) if effects_on else "cut"
            d = max(0.1, min(float(e.get("dur") or 0.5), 3.0))
            chain = f"[{i+1}:v]setsar=1,format=yuva420p"
            if anim in ("fade", "zoom_in"):
                chain += f",fade=t=in:st=0:d={d:.3f}:alpha=1"
            _fc.append(chain + f"[e{i}]")
        _prev = "0:v"
        for i, e in enumerate(elems):
            anim = ((e.get("anim") or "fade").strip().lower()) if effects_on else "cut"
            d = max(0.1, min(float(e.get("dur") or 0.5), 3.0))
            x, y, w = int(e["x"]), int(e["y"]), int(e["w"])
            nxt = f"p{i}"
            if anim == "slide_left":     # slides in from the right of its spot, settles at x
                xe = f"'max({x}\\,{x+w}-{w}*t/{d:.3f})'"
                _fc.append(f"[{_prev}][e{i}]overlay=x={xe}:y={y}[{nxt}]")
            elif anim == "slide_right":  # slides in from the left of its spot, settles at x
                xe = f"'min({x}\\,{x-w}+{w}*t/{d:.3f})'"
                _fc.append(f"[{_prev}][e{i}]overlay=x={xe}:y={y}[{nxt}]")
            else:
                _fc.append(f"[{_prev}][e{i}]overlay={x}:{y}[{nxt}]")
            _prev = nxt
        return _fc, _prev

    def _run(effects_on: bool):
        _fc, _prev = _build(effects_on)
        _cmd = [ff, "-y", "-i", out_path]
        for e in elems:
            _cmd += ["-loop", "1", "-i", str(e["path"])]
        _cmd += ["-filter_complex", ";".join(_fc), "-map", f"[{_prev}]", "-map", "0:a?",
                 "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                 "-c:a", "copy", "-movflags", "+faststart", "-t", f"{_dur:.3f}", tmp]
        return _sp.run(_cmd, capture_output=True, timeout=1800)

    r = _run(True)
    if r.returncode != 0 or not _os.path.isfile(tmp) or _os.path.getsize(tmp) < 1024:
        r = _run(False)
        if r.returncode != 0 or not _os.path.isfile(tmp) or _os.path.getsize(tmp) < 1024:
            err = (r.stderr[-500:].decode("utf-8", "ignore") if r.stderr else "element entrance ffmpeg failed")
            raise RuntimeError(err)
    _os.replace(tmp, out_path)


def _normalize_css_color(c: str | None) -> str | None:
    """Convert a CSS colour (``rgb(...)`` / ``rgba(...)`` / ``#hex``) to ``#rrggbb``.
    Returns None for empty/transparent (so the ticker strip falls back to its default bar)."""
    s = (c or "").strip()
    if not s:
        return None
    if s.startswith("#"):
        return s
    import re
    m = re.match(r"rgba?\(([^)]+)\)", s, re.I)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split(",")]
    try:
        r, g, b = int(float(parts[0])), int(float(parts[1])), int(float(parts[2]))
        a = float(parts[3]) if len(parts) > 3 else 1.0
        if a < 0.05:
            return None
        clamp = lambda v: max(0, min(int(v), 255))
        return "#%02x%02x%02x" % (clamp(r), clamp(g), clamp(b))
    except Exception:
        return None


def _overlay_ticker_on_rect(out_path: str, *, rect: dict, text: str, work_dir: str,
                            fps: int = 30, speed_s=None, bg_color=None,
                            font_px=None, lang=None) -> None:
    """Composite a SCROLLING ticker strip over ``out_path`` at ``rect`` (canvas px), in place.

    Reuses the built-in bulletin crawl: render a wide transparent PNG (sized to the slot
    height) once, then drive the scroll with an ffmpeg overlay x-expression over the finished
    MP4 — a single re-encode pass, NO per-frame screenshots (works on a 12-minute video). The
    strip is confined to the slot via a crop-restore filtergraph so it never bleeds across the
    whole frame. Raises on failure (the caller swallows it so a ticker issue never fails the
    render). The author's static ticker text was already hidden in the still capture."""
    import os as _os
    import shutil as _sh
    import subprocess as _sp
    from pipeline_core.longform_compose import render_ticker, estimate_ticker_width

    rx, ry = int(rect["x"]), int(rect["y"])
    rw, rh = int(rect["w"]), int(rect["h"])
    text = (text or "").strip()
    if rw < 8 or rh < 6 or not text:
        return
    # Font: explicit override, else proportional to the slot height (fixes "ticker text is too
    # small") — a taller ticker bar automatically gets bigger text.
    fp = int(font_px) if font_px else max(18, min(int(rh * 0.55), 80))
    bg = _normalize_css_color(bg_color)
    png = _os.path.join(work_dir, "ticker_strip.png")
    render_ticker([text], (lang or "en"), None, png, height=rh, bg_color=bg, font_px=fp)

    # Scroll speed (px/s): seconds-per-loop if given, else the broadcast default. Loop distance
    # uses the SLOT width (rw), not the full canvas, because the strip loops within the slot.
    if speed_s and float(speed_s) > 0:
        sw = estimate_ticker_width([text], font_px=fp)
        speed = max(20.0, min(1000.0, (sw + rw) / float(speed_s)))
    else:
        speed = 200.0

    ff = _sh.which("ffmpeg") or "ffmpeg"
    tmp = out_path + ".ticker.mp4"
    # crop the slot region, scroll the strip inside it (x loops over the slot width rw), then
    # overlay the mixed slot back at (rx, ry). \\, escapes the comma inside mod() for ffmpeg.
    fc = (
        f"[0:v]split[base][crp];"
        f"[crp]crop={rw}:{rh}:{rx}:{ry}[slot];"
        f"[1:v]scale=-1:{rh}[tk];"
        f"[slot][tk]overlay=x='{rw}-mod(t*{speed:.1f}\\,w+{rw})':y=0:format=auto:shortest=1[mix];"
        f"[base][mix]overlay={rx}:{ry}:format=auto[v]"
    )
    cmd = [ff, "-y", "-i", out_path, "-loop", "1", "-i", png,
           "-filter_complex", fc, "-map", "[v]", "-map", "0:a?",
           "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-c:a", "copy", "-movflags", "+faststart", tmp]
    r = _sp.run(cmd, capture_output=True, timeout=1800)
    if r.returncode != 0 or not _os.path.isfile(tmp) or _os.path.getsize(tmp) < 1024:
        err = (r.stderr[-400:].decode("utf-8", "ignore") if r.stderr else "ffmpeg failed")
        raise RuntimeError(err)
    _os.replace(tmp, out_path)


_PLACEHOLDER = {
    "headline": "Your headline goes here",
    "hook": "Breaking: top story today",
    "subtitle": "A supporting subtitle",
    "title": "Title text",
    "caption": "Caption text",
    "cta": "Tap to watch",
    "kicker": "NEWS",
    "body": "Body copy preview.",
}


def render_preview(bundle: "_bundle.Bundle", contract: "_contract.TemplateContract",
                   out_png: str, *, brand: dict | None = None) -> str:
    """Render a still PREVIEW of the template (placeholder text; video slots shown as
    grey placeholders) for the picker thumbnail. Returns out_png."""
    import shutil
    import tempfile

    from PIL import Image

    texts = {s.key: _PLACEHOLDER.get(s.key, s.key.replace("_", " ").title())
             for s in contract.text_slots}
    work = tempfile.mkdtemp(prefix="kx_prev_")
    try:
        design = os.path.join(work, "design.png")
        # opaque=True: a thumbnail just needs to LOOK like the design. A transparent
        # (omit_background) capture can come back fully blank for templates with full-frame
        # semi-transparent layers (Chromium zeroes the alpha), so render the preview opaque.
        _renderer.render(bundle, _renderer.RenderData(texts=texts, brand=brand or {}),
                         design, canvas=(contract.canvas_w, contract.canvas_h), opaque=True)
        img = Image.open(design).convert("RGBA")
        bg = Image.new("RGBA", img.size, (28, 34, 48, 255))      # neutral slate (only shows if design has transparency)
        Image.alpha_composite(bg, img).convert("RGB").save(out_png, quality=88)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out_png
