"""Sandboxed renderer: a custom template's HTML/CSS -> a design PNG + video-slot rects.

Renders the (already-sanitized, local-only) template bundle in headless Chromium with a
hard network block, fills its text/image/logo slots, forces the video slots transparent
(so they become holes the real clips show through), measures each video slot's pixel
rectangle, and screenshots the canvas to a transparent PNG.

Safety: every network request except local ``file:`` / ``data:`` is aborted (no SSRF,
no remote fetch, no tracking); a fresh browser is launched per render; rendering is
time-boxed. The composer (compose.py) then drops the real video(s) into the rects.
"""
from __future__ import annotations

import base64
import contextlib
import json
import mimetypes
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .contract import SLOT_ATTR

_NAV_TIMEOUT_MS = 12000
_DONE_TIMEOUT_MS = 2500


def _quirks_mode() -> bool:
    """Escape hatch: KAIZER_TEMPLATE_QUIRKS=1 restores the pre-text-safety behavior —
    no DOCTYPE injection (quirks-mode render), the legacy auto-fit predicate, and no
    overlap guard. Read per render so it can be flipped without a restart."""
    return os.environ.get("KAIZER_TEMPLATE_QUIRKS", "").strip().lower() in ("1", "true", "yes")

# Memory-lean headless Chromium. --single-process + --no-zygote collapse the usual
# browser/renderer/gpu/zygote process tree into ONE process (roughly halves RSS), and the
# rest trims background services — keeping a render well under the operator's 500 MB budget.
# (Templates are already sanitized + network-blocked + JS-stripped, so the reduced process
# isolation is an acceptable trade for the RAM win.)
_CHROMIUM_ARGS = [
    # NOTE: deliberately NOT --single-process — it collapses compositing into one process
    # and makes multi-frame ANIMATED screenshots stall past the timeout. We hit the RAM
    # budget instead via the concurrency cap (one browser at a time) + these light flags.
    "--disable-gpu", "--disable-dev-shm-usage", "--disable-extensions",
    "--disable-background-networking", "--disable-default-apps", "--disable-sync",
    "--mute-audio", "--no-first-run", "--js-flags=--max-old-space-size=256",
]
# Serialise Chromium to ONE instance at a time: a render is ~250-350 MB, so a single
# browser keeps the whole engine well under the operator's 500 MB budget even when many
# shorts want to render at once (they queue on this semaphore).
_BROWSER_SEM = threading.BoundedSemaphore(1)


@contextlib.contextmanager
def _capped():
    """Hold a browser slot for the whole Chromium lifetime so concurrent renders/probes
    can't exceed the cap (bounding peak RAM)."""
    _BROWSER_SEM.acquire()
    try:
        yield
    finally:
        _BROWSER_SEM.release()


@dataclass
class RenderData:
    """Content to pour into the template's slots."""
    texts: dict[str, str] = field(default_factory=dict)     # slot key -> text
    images: dict[str, str] = field(default_factory=dict)    # slot key -> local image path
    logo_path: str | None = None
    brand: dict[str, str] = field(default_factory=dict)     # css var name -> value
    # Final render = True: any text/image/logo slot with NO supplied content has its author
    # placeholder wiped + hidden, so nothing leaks. Preview = False: placeholders kept so
    # the picker thumbnail looks populated.
    clear_unfilled: bool = False
    # STILL renders only: hide the ticker/marquee slot's TEXT (keep its layout box) so the
    # frozen author text doesn't sit under the scrolling-ticker overlay composited later.
    # The original text is stashed on data-kx-ticker so the rect measurement can recover it.
    hide_ticker: bool = False
    # Image slot keys opted into a carousel/slideshow — punched transparent + measured (NOT
    # baked) so a timed image overlay is composited at their rect after the still render.
    carousel_keys: list = field(default_factory=list)


@dataclass
class RenderResult:
    png_path: str
    canvas_w: int
    canvas_h: int
    video_rects: list[dict]                                  # [{key,x,y,w,h}]
    warnings: list[str] = field(default_factory=list)
    brand_rects: dict = field(default_factory=dict)          # {"logo": rect|None, "watermark": rect|None}


# JS: measure the logo + watermark slot rects so the per-channel publish stamp can place
# each channel's own logo/watermark AT the template's marked spot (else its default corner).
_BRAND_RECTS_JS = r"""
(attr) => {
  const pick = (el) => { const r = el.getBoundingClientRect();
    return { x: Math.round(r.left), y: Math.round(r.top),
             w: Math.round(r.width), h: Math.round(r.height) }; };
  const out = { logo: null, watermark: null };
  document.querySelectorAll(`[${attr}]`).forEach((el) => {
    const raw = (el.getAttribute(attr) || "").trim().toLowerCase();
    const base = raw.split(":")[0].trim();
    const kind = base === "logo" ? "logo" : (base === "watermark" ? "watermark" : null);
    if (!kind || out[kind]) return;          // first occurrence wins
    const r = pick(el);
    if (r.w >= 4 && r.h >= 4) out[kind] = r;  // ignore zero-size
  });
  return out;
}
"""


# JS: measure the FIRST ticker/marquee slot's rect (canvas px) + its computed bg colour and
# font size + its text, so a scrolling-ticker strip can be composited at exactly that spot.
# Reads the stashed data-kx-ticker (set when the still render hid the text) else textContent.
_TICKER_RECT_JS = r"""
(attr) => {
  const els = [...document.querySelectorAll(`[${attr}]`)];
  const el = els.find((e) => {
    const base = (e.getAttribute(attr) || "").trim().toLowerCase().split(":")[0].trim();
    return base === "ticker" || base === "marquee";
  });
  if (!el) return null;
  const r = el.getBoundingClientRect();
  const cs = getComputedStyle(el);
  const stash = el.getAttribute("data-kx-ticker");
  return {
    x: Math.round(r.left), y: Math.round(r.top),
    w: Math.round(r.width), h: Math.round(r.height),
    text: ((stash != null ? stash : (el.textContent || "")) || "").trim(),
    bg: cs.backgroundColor || "",
    font_px: Math.round(parseFloat(cs.fontSize) || 0),
  };
}
"""


# JS: measure every element carrying a data-kx-anim ENTRANCE (fade/slide_left/slide_right/zoom),
# with its rect (canvas px) + anim type + duration. These are NOT punched — the engine crops each
# from the still design, punches its hole, and re-composites it with the entrance motion.
_ANIM_RECTS_JS = r"""
() => {
  const out = [];
  document.querySelectorAll("[data-kx-anim]").forEach((el) => {
    const a = (el.getAttribute("data-kx-anim") || "").trim().toLowerCase();
    if (!a || a === "none" || a === "cut") return;
    const r = el.getBoundingClientRect();
    if (r.width < 6 || r.height < 6) return;
    out.push({ anim: a, dur: parseFloat(el.getAttribute("data-kx-anim-dur")) || 0.5,
               x: Math.round(r.left), y: Math.round(r.top),
               w: Math.round(r.width), h: Math.round(r.height) });
  });
  return out;
}
"""


def _data_uri(path: str) -> str | None:
    try:
        mime, _ = mimetypes.guess_type(path)
        mime = mime or "application/octet-stream"
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


# Bundled Noto fonts → guarantee Indic glyphs render in Chromium even when the template's
# own font lacks them. Injected as data-URI @font-face and appended as a pure FALLBACK
# (only resolves glyphs the template font is missing — never overrides a deliberate design).
_FONTS_DIR = Path(__file__).resolve().parents[2] / "resources" / "fonts"
_INDIC_FONTS = [
    # (block_lo, block_hi, css family, bundled filename)
    (0x0900, 0x097F, "KaizerNotoDeva", "NotoSansDevanagari-Regular.ttf"),
    (0x0980, 0x09FF, "KaizerNotoBeng", "NotoSansBengali-Regular.ttf"),
    (0x0A80, 0x0AFF, "KaizerNotoGujr", "NotoSansGujarati-Regular.ttf"),
    (0x0B80, 0x0BFF, "KaizerNotoTaml", "NotoSansTamil-Regular.ttf"),
    (0x0C00, 0x0C7F, "KaizerNotoTelu", "NotoSansTelugu-Regular.ttf"),
    (0x0C80, 0x0CFF, "KaizerNotoKnda", "NotoSansKannada-Regular.ttf"),
    (0x0D00, 0x0D7F, "KaizerNotoMlym", "NotoSansMalayalam-Regular.ttf"),
]


def _indic_fonts_for_texts(texts: dict) -> list:
    """[{family, src}] of bundled Indic fonts whose script actually appears in the slot
    text. Empty for Latin-only content (zero overhead for English templates)."""
    blob = " ".join(str(v) for v in (texts or {}).values())
    if not blob:
        return []
    out = []
    for lo, hi, family, fname in _INDIC_FONTS:
        if any(lo <= ord(ch) <= hi for ch in blob):
            uri = _data_uri(str(_FONTS_DIR / fname))
            if uri:
                out.append({"family": family, "src": uri})
    return out


# JS run in the page: set brand vars, fill slots, auto-fit + overlap-guard the text,
# blank video slots, return {rects, fit_warnings}. Async so Playwright awaits it natively.
_FILL_JS = r"""
async (payload) => {
  const ATTR = payload.attr;
  // KAIZER_TEMPLATE_QUIRKS=1 -> legacy fit predicate, no Indic floor, no overlap guard.
  const LEGACY = !!payload.legacyFit;
  const _warns = [];
  const root = document.documentElement;
  for (const [k, v] of Object.entries(payload.brand || {})) {
    try { root.style.setProperty(k, v); } catch (e) {}
  }
  // Indic fallback fonts: register the bundled @font-face faces, then append them to every
  // element's font-family so any glyph the template's font lacks resolves to Noto (no tofu).
  // Appended LAST = pure fallback; never changes a glyph the template font already renders.
  if (payload.fonts && payload.fonts.length) {
    try {
      const fams = payload.fonts.map((f) => "'" + f.family + "'").join(", ");
      const css = payload.fonts.map((f) =>
        "@font-face{font-family:'" + f.family + "';src:url('" + f.src +
        "') format('truetype');font-weight:400 900;font-display:swap;}").join("\n");
      const st = document.createElement("style"); st.textContent = css;
      (document.head || root).appendChild(st);
      const first = payload.fonts[0].family;
      document.querySelectorAll("*").forEach((el) => {
        try {
          const cur = getComputedStyle(el).fontFamily || "";
          if (cur.indexOf(first) === -1) el.style.fontFamily = (cur ? cur + ", " : "") + fams;
        } catch (e) {}
      });
    } catch (e) {}
  }
  // Wait for the injected @font-face faces (and the template's own fonts) BEFORE any
  // measuring, so the fit loop sees the REAL Indic glyph metrics — not the pre-swap
  // fallback (the legacy post-fit wait measured the wrong font, then the swap reflowed).
  if (!LEGACY) {
    try { if (document.fonts && document.fonts.ready) { await document.fonts.ready; } } catch (e) {}
  }
  const TEXT_ALIASES = new Set(["headline","hook","subtitle","title","caption","cta","kicker","body","ticker","marquee","watermark"]);
  const classify = (raw) => {
    raw = (raw || "").trim(); if (!raw) return null;
    let [base, name] = raw.split(":"); base = (base||"").trim().toLowerCase(); name = (name||"").trim();
    if (base === "video" || base === "clip") return ["video", name];
    if (base === "background" || base === "bg") return ["background", name || "background"];
    if (base === "intro") return ["intro", name || "intro"];
    if (base === "image" || base === "img" || base === "photo") return ["image", name];
    if (base === "logo") return ["logo", name || "logo"];
    if (base === "text") return ["text", name];
    if (TEXT_ALIASES.has(base)) return ["text", base];
    return null;
  };
  const setImg = (el, src) => {
    if (!src) return;
    if (el.tagName === "IMG") el.src = src;
    else el.style.backgroundImage = `url('${src}')`;
  };
  // FINAL render: hide a slot we couldn't fill so the author's placeholder never shows.
  // We keep the layout box (visibility:hidden, not display:none) so measured video rects
  // and surrounding flex/grid geometry don't shift after measurement.
  const hideEmpty = (el) => {
    el.style.visibility = "hidden";
    el.setAttribute("data-kaizer-empty", "1");
  };
  const clearUnfilled = !!payload.clearUnfilled;
  const rects = [];
  // L3 BASELINE (captured BEFORE any fill): slot pairs that already overlap in the
  // author's placeholder design are INTENTIONAL layers (caption chips over photos,
  // straps over art) — the overlap guard below must never fire on those pairs.
  const _baseBox = new Map();
  if (!LEGACY) {
    try {
      document.querySelectorAll(`[${ATTR}]`).forEach((el) => {
        if (!classify(el.getAttribute(ATTR))) return;
        const r = el.getBoundingClientRect();
        _baseBox.set(el, { left: r.left, top: r.top, right: r.right, bottom: r.bottom });
      });
    } catch (e) {}
  }
  const els = document.querySelectorAll(`[${ATTR}]`);
  els.forEach((el) => {
    const cls = classify(el.getAttribute(ATTR));
    if (!cls) return;
    const [kind, name] = cls;
    const key = name || kind;
    // "Use this media as-is" (operator uploaded media in the builder): keep the baked
    // content VERBATIM — don't fill it, don't clear it, don't punch it transparent.
    if (el.getAttribute("data-kaizer-fixed") === "1") return;
    if (kind === "text") {
      const t = payload.texts[key] ?? payload.texts[name] ?? null;
      // STILL render: hide the ticker text (a scrolling overlay is composited later). Stash
      // the resolved text on data-kx-ticker so the rect measurement can recover it.
      if (payload.hideTicker && (key === "ticker" || key === "marquee")) {
        const keep = (t !== "" && t != null) ? t : (el.textContent || "");
        el.setAttribute("data-kx-ticker", keep);
        el.textContent = ""; hideEmpty(el);
        return;
      }
      if (t !== "" && t != null) { el.textContent = t; }
      else if (clearUnfilled) { el.textContent = ""; hideEmpty(el); }
    } else if (kind === "image") {
      // Carousel slot: leave a transparent hole; the timed slideshow is composited here later.
      if (payload.carouselKeys && payload.carouselKeys.indexOf(key) !== -1) {
        el.style.background = "transparent"; el.style.backgroundColor = "transparent";
        if (el.tagName === "IMG") el.removeAttribute("src"); else el.innerHTML = "";
        return;
      }
      const src = payload.images[key] ?? payload.images[name] ?? "";
      if (src) { setImg(el, src); }
      else if (clearUnfilled) {
        if (el.tagName === "IMG") el.removeAttribute("src");
        else el.style.backgroundImage = "none";
        hideEmpty(el);
      }
    } else if (kind === "logo") {
      if (payload.logo) { setImg(el, payload.logo); }
      else if (clearUnfilled) {
        if (el.tagName === "IMG") el.removeAttribute("src");
        hideEmpty(el);
      }
    } else if (kind === "video" || kind === "background") {
      el.style.background = "transparent";
      el.style.backgroundColor = "transparent";
      if (el.tagName !== "VIDEO") el.innerHTML = "";
    } else if (kind === "intro") {
      el.style.display = "none";   // intro is a prepended clip, not part of the design
    }
  });
  // AUTO-FIT text: shrink each text slot's font so its FULL content fits its box (and the
  // canvas) — no clipping, no "..." cutoff, sized to the template. Tickers/marquees scroll,
  // so leave them at the author's size.
  const _textEls = [];
  document.querySelectorAll(`[${ATTR}]`).forEach((el) => {
    const c = classify(el.getAttribute(ATTR));
    if (!c || c[0] !== "text") return;
    const key = c[1] || c[0];
    if (key === "ticker" || key === "marquee") return;
    if (el.getAttribute("data-kaizer-empty") === "1") return;   // cleared/hidden slot
    _textEls.push(el);
  });
  // Fit predicate. LEGACY = scroll-vs-client only: on a HEIGHT-LESS slot clientHeight
  // grows WITH the content (scrollHeight always == clientHeight), so it could never
  // detect overflow there — the job-591 bug. The hardened predicate adds a canvas-bottom
  // bound (getBoundingClientRect), which finally gives height-less slots a limit;
  // max-height'd slots keep working because clientHeight caps at the bound.
  const _fitsSelf = (el) => (el.scrollHeight <= el.clientHeight + 1 && el.scrollWidth <= el.clientWidth + 1);
  const _fits = LEGACY ? _fitsSelf
    : (el) => (_fitsSelf(el) && el.getBoundingClientRect().bottom <= window.innerHeight + 2);
  const _INDIC_RE = /[ऀ-ൿ]/;
  _textEls.forEach((el) => {
    const cs = getComputedStyle(el);
    let f = parseFloat(cs.fontSize) || 0;
    if (!f) return;
    if (!LEGACY) {
      // Indic line-height floor: at tight line-heights (1.02) Telugu/Devanagari ink paints
      // OUTSIDE the line boxes (measured 269px ink vs 244.8px boxes) and lines collide.
      // Render-time inline style only — the stored template is never mutated.
      try {
        const lh = parseFloat(cs.lineHeight);
        if (_INDIC_RE.test(el.textContent || "") && isFinite(lh) && lh / f < 1.15) {
          el.style.lineHeight = "1.15";
        }
      } catch (e) {}
    }
    const minF = Math.max(11, f * 0.35);
    let g = 0;
    while (f > minF && !_fits(el) && g++ < 100) { f -= 1; el.style.fontSize = f + "px"; }
    if (!LEGACY) {
      // Still overflowing at the floor: clip a BOUNDED box (overflow:hidden clips nothing
      // without a height bound, so height-less boxes — scrollHeight==clientHeight — skip).
      try {
        const cs2 = getComputedStyle(el);
        const bounded = (cs2.maxHeight && cs2.maxHeight !== "none") ||
                        (el.scrollHeight > el.clientHeight + 1);
        if (cs2.overflow === "visible" && bounded) el.style.overflow = "hidden";
      } catch (e) {}
    }
  });
  // Global pass: shrink all text together while any of it still spills past the canvas.
  // LEGACY keyed on body.scrollHeight, which NEVER grows when the spill lives inside a
  // position:absolute stage on a fixed-height body (measured stuck at the canvas height)
  // — so the hardened pass checks the text elements' own bottoms instead.
  const _pastCanvas = () => _textEls.some((el) => {
    try { return el.getBoundingClientRect().bottom > window.innerHeight + 2; }
    catch (e) { return false; }
  });
  const _overflowing = LEGACY
    ? () => (document.body.scrollHeight > window.innerHeight + 2)
    : _pastCanvas;
  let _g = 0;
  while (_overflowing() && _g++ < 80) {
    let _shrunk = false;
    _textEls.forEach((el) => {
      const f = parseFloat(getComputedStyle(el).fontSize) || 0;
      if (f > 12) { el.style.fontSize = (f - 1) + "px"; _shrunk = true; }
    });
    if (!_shrunk) break;
  }

  // ── OVERLAP GUARD (L3): no text may paint over another slot ────────────────────
  // Runs AFTER fitting, BEFORE rect measurement. For every visible slot pair with a
  // text member: if their boxes (border rect ∪ painted INK for text — Indic ink paints
  // above the border box) intersect > 4px on BOTH axes, shrink the offending text 1px
  // at a time to a 12px floor, then hard-clip if still intersecting. Exempt: pairs the
  // author ALREADY layered (baseline overlap), background slots, and full-canvas zones.
  // Fail-soft: any error logs a warning and the render proceeds untouched.
  if (!LEGACY) {
    try {
      const TOL = 4;
      const W = window.innerWidth, H = window.innerHeight;
      const guardSlots = [];
      document.querySelectorAll(`[${ATTR}]`).forEach((el) => {
        const c = classify(el.getAttribute(ATTR));
        if (!c) return;
        const kind = c[0], key = c[1] || c[0];
        if (kind === "background" || kind === "intro") return;  // intentional layers
        if (key === "ticker" || key === "marquee") return;      // scrolls / hidden in stills
        if (el.getAttribute("data-kaizer-empty") === "1") return;
        const st = getComputedStyle(el);
        if (st.display === "none" || st.visibility === "hidden") return;
        guardSlots.push({ el: el, key: key, text: kind === "text" });
      });
      const box = (s) => {
        const r = s.el.getBoundingClientRect();
        let b = { left: r.left, top: r.top, right: r.right, bottom: r.bottom };
        if (s.text && getComputedStyle(s.el).overflow === "visible") {
          try {  // union with the ink rect — Telugu ink measured 13px ABOVE the border box
            const rng = document.createRange(); rng.selectNodeContents(s.el);
            const ir = rng.getBoundingClientRect();
            if (ir && ir.width > 0 && ir.height > 0) {
              b = { left: Math.min(b.left, ir.left), top: Math.min(b.top, ir.top),
                    right: Math.max(b.right, ir.right), bottom: Math.max(b.bottom, ir.bottom) };
            }
          } catch (e) {}
          // ink an overflow-clipping ancestor already crops can't paint -> clamp to it
          let p = s.el.parentElement, hops = 0;
          while (p && p !== document.body && hops++ < 4) {
            const ps = getComputedStyle(p);
            if (ps.overflow !== "visible") {
              const pr = p.getBoundingClientRect();
              b = { left: Math.max(b.left, pr.left), top: Math.max(b.top, pr.top),
                    right: Math.min(b.right, pr.right), bottom: Math.min(b.bottom, pr.bottom) };
              break;
            }
            p = p.parentElement;
          }
        }
        return b;
      };
      const inter = (a, b) => {
        const ix = Math.min(a.right, b.right) - Math.max(a.left, b.left);
        const iy = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
        return (ix > TOL && iy > TOL);
      };
      const isFull = (b) => ((b.right - b.left) >= W * 0.9 && (b.bottom - b.top) >= H * 0.9);
      const exempt = (a, b) => {
        const ba = _baseBox.get(a.el), bb = _baseBox.get(b.el);
        return !!(ba && bb && inter(ba, bb));
      };
      for (let i = 0; i < guardSlots.length; i++) {
        for (let j = i + 1; j < guardSlots.length; j++) {
          const A = guardSlots[i], B = guardSlots[j];
          if (!A.text && !B.text) continue;              // only text-vs-text / text-vs-media
          if (exempt(A, B)) continue;                    // author layered these on purpose
          let bA = box(A), bB = box(B);
          if (isFull(bA) || isFull(bB)) continue;        // full-canvas zones are exempt
          if (!inter(bA, bB)) continue;
          // Pick the text to shrink: text-vs-media always shrinks the text; text-vs-text
          // shrinks the DOWN-SPILLER (whose ink bottom crosses the other's top; tiebreak
          // = taller ink — the auto-grown box is the invader, not the victim).
          let T;
          if (A.text && B.text) {
            const aSpills = (bA.top < bB.top && bA.bottom > bB.top);
            const bSpills = (bB.top < bA.top && bB.bottom > bA.top);
            if (aSpills && !bSpills) T = A;
            else if (bSpills && !aSpills) T = B;
            else T = ((bA.bottom - bA.top) >= (bB.bottom - bB.top)) ? A : B;
          } else { T = A.text ? A : B; }
          const O = (T === A) ? B : A;
          const f0 = parseFloat(getComputedStyle(T.el).fontSize) || 0;
          let hit = true, g = 0;
          while (hit && g++ < 200) {
            const f = parseFloat(getComputedStyle(T.el).fontSize) || 0;
            if (f <= 12) break;                          // readability floor
            T.el.style.fontSize = (f - 1) + "px";
            bA = box(A); bB = box(B);
            hit = inter(bA, bB);
          }
          if (hit) {
            // Floor reached, still intersecting: hard-clip. overflow:hidden clips NOTHING
            // without a height bound, so height-less boxes also get a maxHeight to O's top.
            try {
              T.el.style.overflow = "hidden";
              const tb = T.el.getBoundingClientRect();
              const ob = box(O);
              const heightless = (T.el.scrollHeight <= T.el.clientHeight + 1);
              const room = Math.floor(ob.top - tb.top - TOL);
              if (heightless && tb.top < ob.top && room > 8) {
                T.el.style.maxHeight = room + "px";
              }
            } catch (e) {}
            _warns.push("overlap guard: CLIPPED text slot '" + T.key + "' against '" + O.key + "'");
          } else if (g > 1) {
            const f1 = parseFloat(getComputedStyle(T.el).fontSize) || 0;
            _warns.push("overlap guard: shrank text slot '" + T.key + "' " +
                        Math.round(f0) + "->" + Math.round(f1) + "px to clear '" + O.key + "'");
          }
        }
      }
    } catch (e) {
      try { _warns.push("overlap guard skipped (measurement error): " + (e && e.message ? e.message : e)); } catch (e2) {}
    }
  }

  // Measure video rects AFTER fills (layout settled). Read border-radius (from the
  // element, else its overflow-clipping ancestor) so rounded video frames stay rounded.
  document.querySelectorAll(`[${ATTR}]`).forEach((el) => {
    const cls = classify(el.getAttribute(ATTR));
    if (!cls) return;
    const _k = cls[1] || cls[0];
    // Carousel image slots are measured too (kind "image_carousel") so the slideshow can be
    // composited at their rect; plain image slots are baked, not measured.
    const isCar = cls[0] === "image" && payload.carouselKeys && payload.carouselKeys.indexOf(_k) !== -1;
    if (cls[0] !== "video" && cls[0] !== "background" && !isCar) return;
    if (el.getAttribute("data-kaizer-fixed") === "1") return;  // baked media -> don't composite here
    const r = el.getBoundingClientRect();
    let rad = parseFloat(getComputedStyle(el).borderTopLeftRadius) || 0;
    let p = el.parentElement, hops = 0;
    while (p && hops < 3 && rad < 1) {
      const ps = getComputedStyle(p);
      if (ps.overflow !== "visible") rad = parseFloat(ps.borderTopLeftRadius) || 0;
      p = p.parentElement; hops++;
    }
    rects.push({ key: _k, kind: isCar ? "image_carousel" : cls[0],
                 x: Math.round(r.left), y: Math.round(r.top),
                 w: Math.round(r.width), h: Math.round(r.height),
                 radius: Math.round(rad) });
  });
  return {rects: rects, fit_warnings: _warns};
}
"""


def _unpack_fill(res):
    """Unpack _FILL_JS's return: {rects, fit_warnings}. Tolerates the legacy bare-list
    shape (a stale cached page/older JS) so a shape mismatch can never fail a render."""
    if isinstance(res, dict):
        rects = res.get("rects") or []
        warns = [str(w) for w in (res.get("fit_warnings") or [])]
        return rects, warns
    return (res or []), []


# ── Overlap-guard math (pure-Python MIRROR of the in-page JS above) ──────────────
# The guard itself must run inside _FILL_JS (fonts have to shrink BEFORE the
# screenshot), so its pair-selection/tolerance/exemption math is mirrored here as
# pure functions to keep the logic unit-testable without a browser. KEEP IN SYNC.
OVERLAP_TOL_PX = 4          # boxes must intersect > this on BOTH axes to count
FULL_CANVAS_FRAC = 0.9      # a box covering >= this fraction of both dims is a "zone"


def _boxes_intersect(a: dict, b: dict, tol: int = OVERLAP_TOL_PX) -> bool:
    """True when boxes ({left,top,right,bottom}) overlap MORE than tol px on both axes."""
    try:
        ix = min(a["right"], b["right"]) - max(a["left"], b["left"])
        iy = min(a["bottom"], b["bottom"]) - max(a["top"], b["top"])
        return ix > tol and iy > tol
    except Exception:
        return False


def _is_full_canvas(box: dict, canvas: tuple, frac: float = FULL_CANVAS_FRAC) -> bool:
    """A near-full-canvas box is a background/mood ZONE — text over it is intentional."""
    try:
        cw, ch = float(canvas[0]), float(canvas[1])
        return ((box["right"] - box["left"]) >= cw * frac
                and (box["bottom"] - box["top"]) >= ch * frac)
    except Exception:
        return False


def overlap_guard_pairs(slots: list, canvas: tuple, tol: int = OVERLAP_TOL_PX,
                        baseline: dict | None = None) -> list:
    """Which slot pairs would the overlap guard act on?

    ``slots``   : [{key, kind, box:{left,top,right,bottom}}]
    ``baseline``: {key: box} measured BEFORE the fill (author-placeholder state) —
                  pairs already intersecting there are intentional layers, exempt.
    Returns [(text_key, other_key)] — only TEXT-vs-TEXT / TEXT-vs-media pairs beyond
    ``tol``; background/intro/ticker slots and full-canvas zones never participate.
    Fail-soft: a malformed slot is skipped, never raises."""
    out = []
    cand = []
    for s in (slots or []):
        try:
            kind = s.get("kind") or ""
            if kind in ("background", "intro"):
                continue
            if (s.get("key") or "") in ("ticker", "marquee"):
                continue
            if _is_full_canvas(s.get("box") or {}, canvas):
                continue
            cand.append(s)
        except Exception:
            continue
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            a, b = cand[i], cand[j]
            try:
                a_text = (a.get("kind") == "text")
                b_text = (b.get("kind") == "text")
                if not (a_text or b_text):
                    continue                      # media-vs-media: not our problem
                if baseline is not None:
                    ba = baseline.get(a.get("key")); bb = baseline.get(b.get("key"))
                    if ba and bb and _boxes_intersect(ba, bb, tol):
                        continue                  # author layered these on purpose
                if _boxes_intersect(a.get("box") or {}, b.get("box") or {}, tol):
                    t, o = (a, b) if a_text else (b, a)
                    out.append((t.get("key"), o.get("key")))
            except Exception:
                continue
    return out


def _build_payload(data: RenderData) -> dict:
    return {
        "attr": SLOT_ATTR,
        "texts": dict(data.texts or {}),
        "images": {k: (_data_uri(v) or "") for k, v in (data.images or {}).items()},
        "logo": _data_uri(data.logo_path) if data.logo_path else None,
        "brand": dict(data.brand or {}),
        "clearUnfilled": bool(getattr(data, "clear_unfilled", False)),
        "hideTicker": bool(getattr(data, "hide_ticker", False)),
        "carouselKeys": [str(k) for k in (getattr(data, "carousel_keys", []) or [])],
        "fonts": _indic_fonts_for_texts(data.texts),
        # KAIZER_TEMPLATE_QUIRKS=1 -> _FILL_JS uses the legacy fit predicate and skips
        # the Indic line-height floor + overlap guard (full old-behavior rollback).
        "legacyFit": _quirks_mode(),
    }


def _make_network_guard(bundle_root: str, entry_path: str | None = None):
    """Build a Playwright route handler that allows ONLY: data:/blob: URIs and file://
    URLs *under the template's own folder*. Everything else (http/https/ws, file:// to
    any other path like /etc/passwd) is aborted — closing SSRF, remote fetch, and local
    file-read (incl. JS-driven fetch('file:///...')) for untrusted template code.

    When ``entry_path`` is given, the entry document is served in STANDARDS mode: the
    upload-time sanitizer historically stripped the ``<!DOCTYPE html>`` (lxml round-trip),
    so every already-stored bundle rendered in quirks mode. Fulfilling the entry request
    with the doctype prepended fixes ALL stored bundles at render time — no re-ingest.
    (Verified pixel-identical on the built-ins; standards is the mode they were authored
    in.) KAIZER_TEMPLATE_QUIRKS=1 disables the prepend (old behavior)."""
    try:
        allowed = Path(bundle_root).resolve().as_uri().rstrip("/").lower() + "/"
    except Exception:
        allowed = ""
    entry_uri = ""
    if entry_path and not _quirks_mode():
        try:
            entry_uri = Path(entry_path).resolve().as_uri().lower()
        except Exception:
            entry_uri = ""

    def _guard(route):
        try:
            url = (route.request.url or "")
            low = url.lower()
            if low.startswith("data:") or low.startswith("blob:"):
                route.continue_()
                return
            if low.startswith("file:"):
                if allowed and low.startswith(allowed):
                    if entry_uri and low == entry_uri:
                        # fail-soft: any hiccup falls through to the normal file serve —
                        # a doctype problem must never abort the entry navigation.
                        try:
                            raw = Path(entry_path).read_text(encoding="utf-8",
                                                             errors="replace")
                            if not raw.lstrip("\ufeff \t\r\n").lower().startswith("<!doctype"):
                                route.fulfill(status=200,
                                              content_type="text/html; charset=utf-8",
                                              body="<!DOCTYPE html>\n" + raw)
                                return
                        except Exception:
                            pass
                    route.continue_()
                else:
                    route.abort()
                return
            route.abort()                       # http/https/ws/everything else
        except Exception:
            try:
                route.abort()
            except Exception:
                pass

    return _guard


def render(bundle, data: RenderData, out_png: str, canvas: tuple[int, int],
           *, opaque: bool = False) -> RenderResult:
    """Render ``bundle`` (a bundle.Bundle) with ``data`` to ``out_png``.
    ``canvas`` = (w, h). Raises RuntimeError on render failure.

    ``opaque=True`` screenshots WITHOUT a transparent background — use it for previews/
    thumbnails. (A transparent capture, ``omit_background``, is needed so video slots become
    see-through holes for the real composite, but it can render the WHOLE frame transparent
    for templates with full-frame semi-transparent layers — Chromium/SwiftShader zeroes the
    alpha — which makes a preview look blank. An opaque capture is reliable for a thumbnail.)"""
    from playwright.sync_api import sync_playwright

    cw, ch = int(canvas[0]), int(canvas[1])
    warnings: list[str] = []
    payload = _build_payload(data)
    entry_uri = Path(bundle.entry_path).resolve().as_uri()

    with _capped(), sync_playwright() as p:
        browser = p.chromium.launch(args=_CHROMIUM_ARGS)
        try:
            context = browser.new_context(
                viewport={"width": cw, "height": ch},
                device_scale_factor=1,
                java_script_enabled=True,
                accept_downloads=False,
            )
            context.set_default_timeout(_NAV_TIMEOUT_MS)
            context.route("**/*", _make_network_guard(bundle.root_dir,
                                                      entry_path=bundle.entry_path))

            page = context.new_page()
            context.on("page", lambda pp: (pp is not page) and pp.close())  # kill popups
            page.set_default_navigation_timeout(_NAV_TIMEOUT_MS)
            page.goto(entry_uri, wait_until="load")
            try:
                page.wait_for_function("window._done === true", timeout=_DONE_TIMEOUT_MS)
            except Exception:
                pass  # templates need not signal _done
            rects, _fit_warns = _unpack_fill(page.evaluate(_FILL_JS, payload))
            for _w in _fit_warns:
                # surface the guard's decisions in the render report AND the server log
                warnings.append(_w)
                try:
                    print(f"[templates] {_w}")
                except Exception:
                    pass
            if payload.get("fonts"):
                # belt-and-braces: _FILL_JS already awaits document.fonts.ready pre-fit,
                # but keep the post-fill wait for the quirks/legacy path.
                try:
                    page.evaluate("async () => { if (document.fonts) { await document.fonts.ready; } return true; }")
                except Exception:
                    pass
            try:
                brand = page.evaluate(_BRAND_RECTS_JS, SLOT_ATTR)
            except Exception:
                brand = {}
            page.screenshot(path=out_png, omit_background=not opaque, timeout=_NAV_TIMEOUT_MS,
                            clip={"x": 0, "y": 0, "width": cw, "height": ch})
        finally:
            browser.close()

    # Clamp rects into the canvas.
    clean_rects = []
    for r in (rects or []):
        x = max(0, min(int(r["x"]), cw)); y = max(0, min(int(r["y"]), ch))
        w = max(0, min(int(r["w"]), cw - x)); h = max(0, min(int(r["h"]), ch - y))
        if w >= 2 and h >= 2:
            clean_rects.append({"key": r["key"], "kind": r.get("kind", "video"),
                                "x": x, "y": y, "w": w, "h": h, "radius": int(r.get("radius") or 0)})
        else:
            warnings.append(f"Video slot '{r.get('key')}' has zero size — skipped.")

    # Punch each video slot transparent so the real clip shows through (a template's
    # opaque body/background would otherwise cover the composited video). Honors the
    # slot's border-radius for rounded video frames.
    # SKIP for opaque previews: a thumbnail wants the WHOLE design visible, and a
    # full-frame `background` slot would otherwise punch the entire frame transparent
    # (the cause of blank thumbnails for full-screen-background templates).
    if not opaque:
        _punch_holes(out_png, clean_rects, (cw, ch))

    return RenderResult(png_path=out_png, canvas_w=cw, canvas_h=ch,
                        video_rects=clean_rects, warnings=warnings,
                        brand_rects=_clamp_brand(brand, cw, ch))


def _clamp_brand(brand: dict, cw: int, ch: int) -> dict:
    """Clamp the logo/watermark slot rects into the canvas. Returns
    {"logo": rect|None, "watermark": rect|None}."""
    out: dict = {}
    for k in ("logo", "watermark"):
        r = (brand or {}).get(k)
        if not r:
            out[k] = None
            continue
        try:
            x = max(0, min(int(r["x"]), cw)); y = max(0, min(int(r["y"]), ch))
            w = max(0, min(int(r["w"]), cw - x)); h = max(0, min(int(r["h"]), ch - y))
            out[k] = {"x": x, "y": y, "w": w, "h": h} if (w >= 4 and h >= 4) else None
        except Exception:
            out[k] = None
    return out


def _clamp_anim(items, cw: int, ch: int) -> list[dict]:
    """Clamp entrance-animation element rects into the canvas; keep anim type + duration."""
    out = []
    for r in (items or []):
        try:
            x = max(0, min(int(r["x"]), cw)); y = max(0, min(int(r["y"]), ch))
            w = max(0, min(int(r["w"]), cw - x)); h = max(0, min(int(r["h"]), ch - y))
            if w >= 6 and h >= 6:
                out.append({"x": x, "y": y, "w": w, "h": h,
                            "anim": str(r.get("anim") or "fade"),
                            "dur": max(0.1, min(float(r.get("dur") or 0.5), 3.0))})
        except Exception:
            continue
    return out


def _clamp_ticker(t: dict | None, cw: int, ch: int) -> dict | None:
    """Clamp the ticker slot rect into the canvas; keep text/bg/font_px. None if absent/tiny."""
    if not t:
        return None
    try:
        x = max(0, min(int(t["x"]), cw)); y = max(0, min(int(t["y"]), ch))
        w = max(0, min(int(t["w"]), cw - x)); h = max(0, min(int(t["h"]), ch - y))
        if w < 8 or h < 6:
            return None
        return {"x": x, "y": y, "w": w, "h": h,
                "text": str(t.get("text") or ""), "bg": str(t.get("bg") or ""),
                "font_px": int(t.get("font_px") or 0)}
    except Exception:
        return None


def _punch_holes(png_path: str, rects: list[dict], canvas: tuple[int, int]) -> None:
    """Set alpha=0 inside each video rect (rounded if radius given) on the design PNG."""
    if not rects:
        return
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return
    img = Image.open(png_path).convert("RGBA")
    alpha = img.getchannel("A")
    draw = ImageDraw.Draw(alpha)
    for r in rects:
        x, y, w, h, rad = r["x"], r["y"], r["w"], r["h"], int(r.get("radius") or 0)
        box = [x, y, x + w - 1, y + h - 1]
        if rad > 0:
            draw.rounded_rectangle(box, radius=min(rad, w // 2, h // 2), fill=0)
        else:
            draw.rectangle(box, fill=0)
    img.putalpha(alpha)
    img.save(png_path)


# JS for the AI-understanding probe: read every element pre-tagged with data-ai-idx and
# return its geometry + text + style cues so an LLM can reason about which region is which.
_CANDIDATES_JS = r"""
() => {
  const out = [];
  document.querySelectorAll("[data-ai-idx]").forEach((el) => {
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    const own = (el.childElementCount === 0 ? (el.textContent || "") : "").trim();
    let cls = "";
    try { cls = (el.getAttribute("class") || "").slice(0, 60); } catch (e) {}
    out.push({
      idx: parseInt(el.getAttribute("data-ai-idx"), 10),
      tag: el.tagName.toLowerCase(),
      id: (el.id || "").slice(0, 40),
      cls: cls,
      text: own.slice(0, 90),
      x: Math.round(r.left), y: Math.round(r.top),
      w: Math.round(r.width), h: Math.round(r.height),
      font: Math.round(parseFloat(cs.fontSize) || 0),
      bgimg: (cs.backgroundImage && cs.backgroundImage !== "none") ? 1 : 0,
      empty: (el.childElementCount === 0 && !own) ? 1 : 0
    });
  });
  return out;
}
"""


def probe_candidates(bundle, canvas: tuple[int, int]) -> list:
    """Render ``bundle`` (whose entry HTML has elements pre-tagged with data-ai-idx) and
    return each tagged element's geometry + text + style cues. Used by the AI-understanding
    pass to let an LLM reason about the layout. Never raises -> [] on failure."""
    from playwright.sync_api import sync_playwright
    cw, ch = int(canvas[0]), int(canvas[1])
    entry_uri = Path(bundle.entry_path).resolve().as_uri()
    try:
        with _capped(), sync_playwright() as p:
            browser = p.chromium.launch(args=_CHROMIUM_ARGS)
            try:
                ctx = browser.new_context(viewport={"width": cw, "height": ch},
                                          device_scale_factor=1, java_script_enabled=True,
                                          accept_downloads=False)
                ctx.set_default_timeout(_NAV_TIMEOUT_MS)
                ctx.route("**/*", _make_network_guard(bundle.root_dir,
                                                      entry_path=bundle.entry_path))
                page = ctx.new_page()
                ctx.on("page", lambda pp: (pp is not page) and pp.close())
                page.set_default_navigation_timeout(_NAV_TIMEOUT_MS)
                page.goto(entry_uri, wait_until="load")
                cands = page.evaluate(_CANDIDATES_JS)
            finally:
                browser.close()
        return cands or []
    except Exception:
        return []


def _clamp_rects(rects, cw: int, ch: int) -> list[dict]:
    out = []
    for r in (rects or []):
        x = max(0, min(int(r["x"]), cw)); y = max(0, min(int(r["y"]), ch))
        w = max(0, min(int(r["w"]), cw - x)); h = max(0, min(int(r["h"]), ch - y))
        if w >= 2 and h >= 2:
            out.append({"key": r["key"], "kind": r.get("kind", "video"),
                        "x": x, "y": y, "w": w, "h": h,
                        "radius": int(r.get("radius") or 0)})
    return out


@dataclass
class AnimatedResult:
    frames_dir: str
    frame_pattern: str          # printf pattern, e.g. "f_%05d.png"
    frame_count: int
    fps: int
    canvas_w: int
    canvas_h: int
    video_rects: list[dict]
    animated: bool              # False => effectively a still (1 frame captured)
    warnings: list[str] = field(default_factory=list)
    brand_rects: dict = field(default_factory=dict)   # {"logo": rect|None, "watermark": rect|None}
    # First ticker/marquee slot rect (canvas px) + its text/bg/font_px, for the scrolling
    # ticker overlay. None when the template has no ticker slot. {x,y,w,h,text,bg,font_px}.
    ticker_rect: dict | None = None
    # Carousel image slot rects (canvas px) — {key,x,y,w,h} per slot — where a timed image
    # slideshow is composited after the still render. Empty when no carousel slots.
    carousel_rects: list[dict] = field(default_factory=list)
    # Elements with an entrance animation (data-kx-anim) — {x,y,w,h,anim,dur}. NOT punched here;
    # the engine crops each from the still, punches its hole, then re-composites with the motion.
    anim_rects: list[dict] = field(default_factory=list)


# Hard cap on captured animation frames. CSS animations LOOP, so a few seconds is enough —
# the composer tiles/loops the design over the full clip. Capping keeps the long-lived
# capture browser's RSS down (it accumulates over screenshots) + the render fast.
_MAX_ANIM_FRAMES = 192


# When a template has a full-frame ``background`` slot, the real clip is meant to FILL the
# frame with the design's graphics layered ON TOP. Rectangular hole-punching can't express
# "chrome over a full-frame video" (it would erase the chrome), and ``omit_background`` can
# zero the whole frame's alpha for such templates. So for these we recover TRUE per-pixel
# alpha with a two-pass capture (render over black + white) — opaque chrome stays opaque,
# empty areas go transparent, and semi-transparent scrims keep partial alpha (they darken
# the clip). First the template's own opaque body/html background is neutralised so the clip
# isn't covered (a background slot means "put the video here").
_NEUTRALIZE_BG_JS = r"""
() => {
  for (const el of [document.documentElement, document.body]) {
    el.style.setProperty('background', 'transparent', 'important');
    el.style.setProperty('background-color', 'transparent', 'important');
    el.style.setProperty('background-image', 'none', 'important');
  }
}
"""


def _screenshot_two_pass(page, cdp, clip, out_path) -> bool:
    """Capture ``page`` over a black AND a white backdrop and recover straight-alpha RGBA.
    Returns True on success, False if numpy is unavailable (caller falls back)."""
    try:
        import io

        import numpy as np
        from PIL import Image
    except Exception:
        return False

    def _shot(rgb):
        cdp.send("Emulation.setDefaultBackgroundColorOverride",
                 {"color": {"r": rgb[0], "g": rgb[1], "b": rgb[2], "a": 255}})
        raw = page.screenshot(omit_background=False, clip=clip, timeout=_NAV_TIMEOUT_MS)
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float32)

    black = _shot((0, 0, 0))
    white = _shot((255, 255, 255))
    # over-white minus over-black == (1-alpha); so alpha = 255 - that. Premultiplied colour
    # is the over-black render; straighten it by dividing out alpha.
    diff = np.clip(white - black, 0, 255)
    alpha = np.clip(255.0 - diff.max(axis=2), 0, 255)
    af = np.maximum(alpha, 1.0)[..., None] / 255.0
    rgb = np.clip(black / af, 0, 255).astype("uint8")
    Image.fromarray(np.dstack([rgb, alpha.astype("uint8")]), "RGBA").save(out_path)
    return True


def render_animated(bundle, data: RenderData, frames_dir: str, canvas: tuple[int, int],
                    *, fps: int = 24, duration: float = 6.0,
                    cap_seconds: float = 8.0, force_still: bool = False) -> AnimatedResult:
    """Capture the template as a transparent ANIMATED frame sequence using Chromium
    virtual time, so CSS animations / marquee / blink / gif advance deterministically.
    Every frame has the video slots punched transparent. If the template has no running
    animations it captures a single still frame. Returns :class:`AnimatedResult`."""
    from playwright.sync_api import sync_playwright

    cw, ch = int(canvas[0]), int(canvas[1])
    os.makedirs(frames_dir, exist_ok=True)
    fps = max(1, min(int(fps), 24))
    dur = max(0.2, min(float(duration), float(cap_seconds)))
    payload = _build_payload(data)
    entry_uri = Path(bundle.entry_path).resolve().as_uri()
    pattern = "f_%05d.png"
    warnings: list[str] = []
    rects, frame_count, animated = [], 0, False

    with _capped(), sync_playwright() as p:
        browser = p.chromium.launch(args=_CHROMIUM_ARGS)
        try:
            context = browser.new_context(viewport={"width": cw, "height": ch},
                                          device_scale_factor=1, java_script_enabled=True,
                                          accept_downloads=False)
            context.set_default_timeout(_NAV_TIMEOUT_MS)
            context.route("**/*", _make_network_guard(bundle.root_dir,
                                                      entry_path=bundle.entry_path))
            page = context.new_page()
            context.on("page", lambda pp: (pp is not page) and pp.close())  # kill popups
            page.set_default_navigation_timeout(_NAV_TIMEOUT_MS)
            page.goto(entry_uri, wait_until="load")
            rects, _fit_warns = _unpack_fill(page.evaluate(_FILL_JS, payload))
            for _w in _fit_warns:
                # flows into AnimatedResult.warnings -> engine.render_template's report
                warnings.append(_w)
                try:
                    print(f"[templates] {_w}")
                except Exception:
                    pass
            if payload.get("fonts"):
                # belt-and-braces for the quirks/legacy path (see render() above)
                try:
                    page.evaluate("async () => { if (document.fonts) { await document.fonts.ready; } return true; }")
                except Exception:
                    pass
            # A full-frame ``background`` slot means the clip fills the frame with the design
            # layered on top -> recover true per-pixel alpha (chrome stays, clip shows through)
            # instead of rectangular hole-punching. Neutralise the template's own opaque
            # body/html background so it doesn't cover the clip.
            has_bg = any((r.get("kind") == "background") for r in (rects or []))
            if has_bg:
                try:
                    page.evaluate(_NEUTRALIZE_BG_JS)
                except Exception:
                    has_bg = False
            try:
                brand = page.evaluate(_BRAND_RECTS_JS, SLOT_ATTR)
            except Exception:
                brand = {}
            try:
                ticker_rect = page.evaluate(_TICKER_RECT_JS, SLOT_ATTR)
            except Exception:
                ticker_rect = None
            try:
                anim_rects = page.evaluate(_ANIM_RECTS_JS)
            except Exception:
                anim_rects = []
            try:
                animated = int(page.evaluate("() => document.getAnimations().length")) > 0
            except Exception:
                animated = False
            if force_still:
                animated = False   # capture ONE still frame — light + stable (no CDP loop)

            cdp = context.new_cdp_session(page)
            expired = {"flag": True}
            cdp.on("Emulation.virtualTimeBudgetExpired", lambda *_a: expired.__setitem__("flag", True))

            def _advance(budget_ms: float):
                expired["flag"] = False
                try:
                    cdp.send("Emulation.setVirtualTimePolicy",
                             {"policy": "advance", "budget": budget_ms})
                except Exception:
                    return
                for _ in range(60):            # ~180ms safety cap per frame
                    if expired["flag"]:
                        break
                    page.wait_for_timeout(3)

            try:
                cdp.send("Emulation.setVirtualTimePolicy", {"policy": "pause"})
            except Exception:
                pass

            n = max(1, int(round(dur * fps))) if animated else 1
            n = min(n, _MAX_ANIM_FRAMES)   # hard cap -> bounded RAM + render time
            clip = {"x": 0, "y": 0, "width": cw, "height": ch}
            frame_ms = 1000.0 / fps
            two_pass = False
            for i in range(n):
                if animated and i > 0:
                    _advance(frame_ms)
                fpath = os.path.join(frames_dir, pattern % i)
                if has_bg and _screenshot_two_pass(page, cdp, clip, fpath):
                    two_pass = True
                else:
                    has_bg = False   # numpy missing -> use the hole-punch path for all frames
                    page.screenshot(path=fpath, omit_background=True, clip=clip,
                                    timeout=_NAV_TIMEOUT_MS)
            frame_count = n
        finally:
            browser.close()

    clean_all = _clamp_rects(rects, cw, ch)
    # Sub-box templates: punch each clip/carousel rect transparent so the clip / slideshow shows
    # through. Skipped for full-frame-background templates — the two-pass capture has correct alpha.
    if not two_pass:
        for i in range(frame_count):
            _punch_holes(os.path.join(frames_dir, pattern % i), clean_all, (cw, ch))
    # Carousel rects are punched (hole) but NOT video placements — a timed image overlay fills
    # them later; the main clip must NOT be composited into a carousel slot.
    carousel_rects = [r for r in clean_all if r.get("kind") == "image_carousel"]
    video_rects = [r for r in clean_all if r.get("kind") != "image_carousel"]
    return AnimatedResult(frames_dir=frames_dir, frame_pattern=pattern, frame_count=frame_count,
                          fps=fps, canvas_w=cw, canvas_h=ch, video_rects=video_rects,
                          animated=animated, warnings=warnings,
                          brand_rects=_clamp_brand(brand, cw, ch),
                          ticker_rect=_clamp_ticker(ticker_rect, cw, ch),
                          carousel_rects=carousel_rects,
                          anim_rects=_clamp_anim(anim_rects, cw, ch))
