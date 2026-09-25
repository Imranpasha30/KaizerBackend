"""Template TEXT SAFETY: standards-mode render + Indic-aware auto-fit + overlap guard.

Locks the systemic fix for the job-591 Mangli overlap (3-line Telugu headline painted
over the photo chip + subtitle on Minimal Light short, stored template 34):

  L1  the sanitizer's lxml round-trips preserve the author's <!DOCTYPE html>, and the
      renderer's network guard serves the ENTRY in standards mode by prepending the
      doctype at render time — fixing every ALREADY-STORED bundle without re-ingest.
  L2  the auto-fit predicate gained a canvas-bottom bound (height-less slots used to
      defeat scrollHeight==clientHeight), an Indic line-height floor (1.15), and a
      fonts.ready wait BEFORE measuring.
  L3  an overlap guard shrinks (then clips) any text that would paint over another
      slot — with exemptions for author-intended layers (baseline overlap), background
      slots, and full-canvas zones. Its pair/tolerance math is mirrored as pure Python
      (renderer.overlap_guard_pairs) so it is unit-testable without a browser.

KAIZER_TEMPLATE_QUIRKS=1 restores the old behavior end to end (escape hatch).

Playwright-marked tests launch a real headless Chromium through the REAL fill path
(_build_payload + _make_network_guard + _FILL_JS — the exact sequence render() and
render_animated() use) and skip cleanly when Chromium is unavailable.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from services.custom_templates import bundle as ct_bundle
from services.custom_templates import fill as ct_fill
from services.custom_templates import infer as ct_infer
from services.custom_templates import renderer as R
from services.custom_templates.contract import preserve_doctype

# The exact headline from the operator's job 587/591 renders (3 lines at 80px).
MANGLI = "సింగర్ మంగ్లీపై తప్పుడు ప్రచారం: గిరిజన సంఘాల ఆందోళన"
LONG_SUB = ("Tribal associations protest against the false propaganda targeting "
            "singer Mangli across three districts today")

BUNDLE_34 = Path(__file__).resolve().parents[1] / "output" / "custom_templates" / "34" / "bundle"


# ── L1: doctype preservation through every lxml round-trip ────────────────────────

def test_preserve_doctype_restores_only_authored_doctype():
    # authored doctype restored
    out = preserve_doctype("<!DOCTYPE html>\n<html><body/></html>", "<html><body></body></html>")
    assert out.startswith("<!DOCTYPE html>")
    # no doctype in the source -> nothing invented (lxml fabricates an HTML-4.0 default
    # in docinfo; we must never trust it)
    assert preserve_doctype("<html><body/></html>", "<html></html>") == "<html></html>"
    # already present in the output -> untouched (no double doctype)
    same = "<!DOCTYPE html><html></html>"
    assert preserve_doctype(same, same) == same
    # BOM + leading whitespace tolerated
    assert preserve_doctype("﻿  <!doctype html><x/>", "<x/>").lower().startswith("<!doctype html>")


def test_sanitizer_keeps_doctype():
    src = "<!DOCTYPE html>\n<html><head><title>t</title></head><body><p>hi</p></body></html>"
    clean, _n = ct_bundle.neutralize_external_html(src)
    assert clean.lstrip().lower().startswith("<!doctype html>")
    # and never invents one for doctype-less uploads
    clean2, _n2 = ct_bundle.neutralize_external_html("<html><body><p>x</p></body></html>")
    assert not clean2.lstrip().lower().startswith("<!doctype")


def test_normalize_keeps_doctype():
    src = ("<!DOCTYPE html><html><head><meta name=\"kaizer:canvas\" content=\"1080x1920\">"
           "</head><body><div data-kaizer=\"video\"></div></body></html>")
    out, _canvas, _warns = ct_infer.normalize_template(src)
    assert out.lstrip().lower().startswith("<!doctype html>")


def test_fill_keeps_doctype():
    src = ("<!DOCTYPE html><html><body><div data-kaizer=\"headline\">x</div>"
           "<div data-kaizer=\"video\"></div></body></html>")
    out = ct_fill.fill_html(src, texts={"headline": "hello"})
    assert out.lstrip().lower().startswith("<!doctype html>")
    assert ">hello<" in out


# ── L1: the render-time network-guard doctype prepend ─────────────────────────────

class _FakeRoute:
    def __init__(self, url):
        self.request = SimpleNamespace(url=url)
        self.action = None
        self.kwargs = None

    def continue_(self):
        self.action = "continue"

    def abort(self):
        self.action = "abort"

    def fulfill(self, **kw):
        self.action = "fulfill"
        self.kwargs = kw


def _entry_bundle(tmp_path, html):
    root = tmp_path / "bundle"
    root.mkdir(parents=True, exist_ok=True)
    entry = root / "index.html"
    entry.write_text(html, encoding="utf-8")
    return root, entry


def test_guard_fulfills_entry_with_doctype(tmp_path, monkeypatch):
    monkeypatch.delenv("KAIZER_TEMPLATE_QUIRKS", raising=False)
    root, entry = _entry_bundle(tmp_path, "<html lang=\"en\"><body>x</body></html>")
    guard = R._make_network_guard(str(root), entry_path=str(entry))
    route = _FakeRoute(Path(entry).resolve().as_uri())
    guard(route)
    assert route.action == "fulfill"
    assert route.kwargs["body"].startswith("<!DOCTYPE html>\n<html")
    assert "text/html" in route.kwargs["content_type"]


def test_guard_leaves_entry_that_already_has_doctype(tmp_path, monkeypatch):
    monkeypatch.delenv("KAIZER_TEMPLATE_QUIRKS", raising=False)
    root, entry = _entry_bundle(tmp_path, "<!DOCTYPE html>\n<html><body>x</body></html>")
    guard = R._make_network_guard(str(root), entry_path=str(entry))
    route = _FakeRoute(Path(entry).resolve().as_uri())
    guard(route)
    assert route.action == "continue"           # normal file serve, no rewrite needed


def test_guard_security_behavior_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("KAIZER_TEMPLATE_QUIRKS", raising=False)
    root, entry = _entry_bundle(tmp_path, "<html></html>")
    (root / "a.css").write_text("body{}", encoding="utf-8")
    guard = R._make_network_guard(str(root), entry_path=str(entry))
    # sibling asset under the bundle root -> allowed, untouched
    r1 = _FakeRoute((root / "a.css").resolve().as_uri())
    guard(r1); assert r1.action == "continue"
    # file OUTSIDE the bundle -> aborted (SSRF / local-read block intact)
    r2 = _FakeRoute(Path(tmp_path / "secret.txt").resolve().as_uri())
    guard(r2); assert r2.action == "abort"
    # remote -> aborted
    r3 = _FakeRoute("https://evil.example/x.js")
    guard(r3); assert r3.action == "abort"
    # data: -> allowed
    r4 = _FakeRoute("data:image/png;base64,AAAA")
    guard(r4); assert r4.action == "continue"


def test_guard_quirks_escape_hatch(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIZER_TEMPLATE_QUIRKS", "1")
    root, entry = _entry_bundle(tmp_path, "<html lang=\"en\"><body>x</body></html>")
    guard = R._make_network_guard(str(root), entry_path=str(entry))
    route = _FakeRoute(Path(entry).resolve().as_uri())
    guard(route)
    assert route.action == "continue"           # old behavior: no doctype injection


def test_payload_carries_quirks_flag(monkeypatch):
    monkeypatch.delenv("KAIZER_TEMPLATE_QUIRKS", raising=False)
    assert R._build_payload(R.RenderData(texts={"headline": "x"}))["legacyFit"] is False
    monkeypatch.setenv("KAIZER_TEMPLATE_QUIRKS", "1")
    assert R._build_payload(R.RenderData(texts={"headline": "x"}))["legacyFit"] is True


# ── L2/L3: the injected JS carries the hardened logic + new return shape ──────────

def test_fill_js_hardening_markers():
    js = R._FILL_JS
    assert "async (payload)" in js                    # Playwright awaits it natively
    assert "document.fonts.ready" in js               # fonts BEFORE measuring
    assert "window.innerHeight + 2" in js             # canvas-bottom fit bound
    assert 'el.style.lineHeight = "1.15"' in js       # Indic line-height floor
    assert "fit_warnings" in js and "rects: rects" in js
    assert "overlap guard" in js.lower()


def test_unpack_fill_both_shapes():
    rects, warns = R._unpack_fill({"rects": [{"key": "video"}], "fit_warnings": ["w1"]})
    assert rects == [{"key": "video"}] and warns == ["w1"]
    # legacy bare-list shape tolerated (fail-soft)
    rects2, warns2 = R._unpack_fill([{"key": "video"}])
    assert rects2 == [{"key": "video"}] and warns2 == []
    assert R._unpack_fill(None) == ([], [])


# ── L3: overlap-guard math as pure functions ──────────────────────────────────────

def _box(l, t, r, b):
    return {"left": l, "top": t, "right": r, "bottom": b}


def test_boxes_intersect_tolerance():
    a = _box(0, 0, 100, 100)
    assert R._boxes_intersect(a, _box(50, 50, 150, 150))          # clear overlap
    assert not R._boxes_intersect(a, _box(100, 100, 200, 200))    # touching only
    assert not R._boxes_intersect(a, _box(96, 0, 200, 100))       # 4px x-overlap == tol
    assert R._boxes_intersect(a, _box(95, 0, 200, 100))           # 5px > tol
    # one-axis-only overlap never counts
    assert not R._boxes_intersect(a, _box(0, 200, 100, 300))
    # malformed input is fail-soft, never raises
    assert R._boxes_intersect({}, a) is False


def test_full_canvas_zone_detection():
    canvas = (1080, 1920)
    assert R._is_full_canvas(_box(0, 0, 1080, 1920), canvas)
    assert R._is_full_canvas(_box(10, 20, 1060, 1880), canvas)    # >=90% both dims
    assert not R._is_full_canvas(_box(0, 0, 1080, 400), canvas)   # wide strip: not a zone


def test_guard_pairs_text_vs_media_and_text():
    canvas = (1080, 1920)
    slots = [
        {"key": "headline", "kind": "text", "box": _box(64, 1439, 1016, 1712)},
        {"key": "image", "kind": "image", "box": _box(64, 1632, 212, 1780)},
        {"key": "subtitle", "kind": "text", "box": _box(236, 1650, 1016, 1763)},
        {"key": "logo", "kind": "logo", "box": _box(924, 70, 1016, 162)},
    ]
    pairs = R.overlap_guard_pairs(slots, canvas)
    assert ("headline", "image") in pairs             # the job-591 photo overlap
    assert ("headline", "subtitle") in pairs          # the job-591 subtitle overlap
    # sub starts at x=236, photo ends at x=212 — no x-overlap, so no pair (job-591
    # measured geometry: only the HEADLINE invaded both).
    assert ("subtitle", "image") not in pairs
    assert all("logo" not in p for p in pairs)        # logo is clear of everything


def test_guard_pairs_exemptions():
    canvas = (1920, 1080)
    text = {"key": "cap", "kind": "text", "box": _box(1400, 702, 1800, 726)}
    photo = {"key": "image", "kind": "image", "box": _box(1426, 206, 1824, 766)}
    # 1) baseline exemption: the author's placeholder ALREADY overlapped -> intentional
    baseline = {"cap": _box(1400, 702, 1700, 726), "image": _box(1426, 206, 1824, 766)}
    assert R.overlap_guard_pairs([text, photo], canvas, baseline=baseline) == []
    # without the baseline the same pair WOULD fire (proves the exemption is what saves it)
    assert R.overlap_guard_pairs([text, photo], canvas) == [("cap", "image")]
    # 2) background slots never participate (text over full-frame video is a design)
    bg = {"key": "background", "kind": "background", "box": _box(0, 0, 1920, 1080)}
    assert R.overlap_guard_pairs([text, bg], canvas) == []
    # 3) full-canvas zones exempt even when kind is media
    zone = {"key": "image", "kind": "image", "box": _box(5, 5, 1915, 1075)}
    assert R.overlap_guard_pairs([text, zone], canvas) == []
    # 4) media-vs-media is not the guard's business
    m2 = {"key": "video", "kind": "video", "box": _box(1400, 300, 1800, 800)}
    assert R.overlap_guard_pairs([photo, m2], canvas) == []
    # 5) ticker/marquee excluded (scrolls; hidden in stills)
    tick = {"key": "ticker", "kind": "text", "box": _box(0, 700, 1920, 780)}
    assert R.overlap_guard_pairs([tick, photo], canvas) == []


# ── Playwright: the REAL fill path in headless Chromium ───────────────────────────

def _launch_page(p, root, entry, canvas):
    browser = p.chromium.launch(args=R._CHROMIUM_ARGS)
    ctx = browser.new_context(viewport={"width": canvas[0], "height": canvas[1]},
                              device_scale_factor=1, java_script_enabled=True,
                              accept_downloads=False)
    ctx.route("**/*", R._make_network_guard(str(root), entry_path=str(entry)))
    page = ctx.new_page()
    page.goto(Path(entry).resolve().as_uri(), wait_until="load")
    return browser, page


def _run_fill(tmp_or_root, entry, canvas, texts, *, hide_ticker=False, clear=False):
    """Drive _FILL_JS through the exact still-render sequence, then measure slots."""
    pw = pytest.importorskip("playwright.sync_api")
    sync_playwright = pw.sync_playwright
    payload = R._build_payload(R.RenderData(texts=texts, clear_unfilled=clear,
                                            hide_ticker=hide_ticker))
    measure = r"""
    () => {
      const out = { compat: document.compatMode, slots: [] };
      document.querySelectorAll("[data-kaizer]").forEach((el) => {
        const raw = (el.getAttribute("data-kaizer") || "").trim().toLowerCase();
        const key = raw.indexOf(":") !== -1 ? raw.split(":")[1] : raw;
        const cs = getComputedStyle(el);
        const r = el.getBoundingClientRect();
        let b = { left: r.left, top: r.top, right: r.right, bottom: r.bottom };
        try {
          const rng = document.createRange(); rng.selectNodeContents(el);
          const ir = rng.getBoundingClientRect();
          if (ir && ir.width > 0 && ir.height > 0 && cs.overflow === "visible") {
            b = { left: Math.min(b.left, ir.left), top: Math.min(b.top, ir.top),
                  right: Math.max(b.right, ir.right), bottom: Math.max(b.bottom, ir.bottom) };
          }
        } catch (e) {}
        out.slots.push({ key: key, kind: raw.split(":")[0], box: b,
                         font: parseFloat(cs.fontSize) || 0,
                         lineHeight: cs.lineHeight,
                         visible: cs.visibility !== "hidden" && cs.display !== "none",
                         fits: el.scrollHeight <= el.clientHeight + 1 });
      });
      return out;
    }
    """
    with sync_playwright() as p:
        try:
            browser, page = _launch_page(p, tmp_or_root, entry, canvas)
        except Exception as exc:                      # no Chromium on this host
            pytest.skip(f"Chromium unavailable: {exc}")
        try:
            res = page.evaluate(R._FILL_JS, payload)
            probe = page.evaluate(measure)
        finally:
            browser.close()
    return res, probe


def _slot(probe, key):
    for s in probe["slots"]:
        if s["key"] == key:
            return s
    raise AssertionError(f"slot {key!r} not found in {[s['key'] for s in probe['slots']]}")


@pytest.mark.playwright
def test_autofit_shrinks_telugu_headline_in_explicit_height_slot(tmp_path, monkeypatch):
    monkeypatch.delenv("KAIZER_TEMPLATE_QUIRKS", raising=False)
    # no doctype on purpose: also exercises the render-time standards-mode prepend
    html = """<html><head><meta charset="utf-8">
    <meta name="kaizer:canvas" content="1080x1920"><style>
      *{margin:0;padding:0;box-sizing:border-box}
      html,body{width:1080px;height:1920px;overflow:hidden;font-family:Arial}
      .h{position:absolute;left:40px;top:100px;width:1000px;height:120px;
         font-size:80px;line-height:1.02;overflow:hidden}
    </style></head><body>
      <div class="h" data-kaizer="headline">placeholder</div>
      <div data-kaizer="video" style="position:absolute;left:40px;top:400px;width:1000px;height:600px"></div>
    </body></html>"""
    root, entry = _entry_bundle(tmp_path, html)
    res, probe = _run_fill(root, entry, (1080, 1920), {"headline": MANGLI})
    assert probe["compat"] == "CSS1Compat"            # standards mode ACTIVE (L1 marker)
    h = _slot(probe, "headline")
    assert h["font"] < 80                             # auto-fit actually fired
    assert h["fits"]                                  # content fits the explicit-height box
    # Indic line-height floor applied (author wrote 1.02)
    lh = float(h["lineHeight"].replace("px", ""))
    assert lh / h["font"] >= 1.14


@pytest.mark.playwright
def test_quirks_env_restores_old_behavior(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIZER_TEMPLATE_QUIRKS", "1")
    html = """<html><head><meta charset="utf-8"><style>
      *{margin:0;padding:0;box-sizing:border-box}
      html,body{width:1080px;height:1920px;overflow:hidden}
      .stage{position:absolute;inset:0}
      .wrap{position:absolute;left:64px;right:64px;top:1392px}
      .h{font-size:80px;line-height:1.02;font-weight:900}
    </style></head><body><div class="stage"><div class="wrap">
      <div class="h" data-kaizer="headline">placeholder</div></div>
      <div data-kaizer="image" style="position:absolute;left:64px;top:1632px;width:148px;height:148px;background:#eee"></div>
    </div></body></html>"""
    root, entry = _entry_bundle(tmp_path, html)
    res, probe = _run_fill(root, entry, (1080, 1920), {"headline": MANGLI})
    assert probe["compat"] == "BackCompat"            # no doctype injection (old mode)
    h = _slot(probe, "headline")
    # Indic line-height floor is OFF: the author's tight 1.02 survives
    lh = float(h["lineHeight"].replace("px", ""))
    assert lh / h["font"] < 1.1
    # the overlap guard is OFF: it never speaks (no shrink/clip decisions surfaced)
    assert isinstance(res, dict) and res.get("fit_warnings") == []


@pytest.mark.playwright
def test_bundle_34_mangli_regression_no_overlap():
    """THE job-591 reproduction: stored Minimal Light short (template 34) + the exact
    Mangli headline + a long English subtitle must render with the headline SHRUNK
    below the author's 80px and NO text slot painting over any other slot."""
    if not (BUNDLE_34 / "index.html").is_file():
        pytest.skip("stored bundle 34 not present on this host")
    entry = BUNDLE_34 / "index.html"
    res, probe = _run_fill(BUNDLE_34, entry, (1080, 1920),
                           {"headline": MANGLI, "subtitle": LONG_SUB},
                           hide_ticker=True)
    assert probe["compat"] == "CSS1Compat"            # stored bundle renders in STANDARDS
    h = _slot(probe, "headline")
    assert 0 < h["font"] < 80                         # shrank below the authored size
    # No text slot box may intersect any other slot's box beyond the guard tolerance
    # (ticker excluded: it is hidden for stills and composited as a scroller later).
    vis = [s for s in probe["slots"] if s["visible"] and s["key"] not in ("ticker", "marquee")]
    for a in vis:
        for b in vis:
            if a is b or a["kind"] != "text":
                continue
            assert not R._boxes_intersect(a["box"], b["box"]), (
                f"text slot {a['key']!r} still overlaps {b['key']!r}: {a['box']} vs {b['box']}")
    # the guard/fit surfaced its decisions in the report channel
    assert isinstance(res, dict) and isinstance(res.get("fit_warnings"), list)
