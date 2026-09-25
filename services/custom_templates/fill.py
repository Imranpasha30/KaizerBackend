"""Server-side slot fill — bake job content into a template's HTML *without* a browser.

The visual builder loads a template for in-place editing of a SPECIFIC job. To make that
WYSIWYG, we pre-fill the text (and, when a usable URL is given, image) slots with this job's
real content so the operator edits over what they'll actually ship — not author placeholders.

This mirrors the render-time ``_FILL_JS`` (renderer.py) but runs in Python via lxml, the same
parser the contract uses. It is deliberately conservative:

  * TEXT slots  -> textContent replaced (children cleared) when a non-empty value is supplied.
  * IMAGE slots -> ``src`` / ``background-image`` set ONLY for http(s)/data URLs the browser
                   can actually load (local server paths are skipped — they'd 404 in the iframe).
  * video / background / intro / logo / watermark -> LEFT ALONE. The clip composites at render;
    logo + watermark are stamped per-channel at publish; intro is a prepended clip.

Returns the filled HTML string; never raises (falls back to the input on any parse failure).
"""
from __future__ import annotations

from lxml import html as lxml_html

from .contract import SLOT_ATTR, _classify, parse_safe, preserve_doctype


def _is_loadable_url(src: str) -> bool:
    s = (src or "").strip().lower()
    return s.startswith(("http://", "https://", "data:", "//", "/"))


def fill_html(html_str: str, texts: dict | None = None,
              images: dict | None = None) -> str:
    """Return ``html_str`` with text/image slots filled from ``texts``/``images``.

    ``texts``  : {slot_key: text}     — empty/missing keys keep the author placeholder.
    ``images`` : {slot_key: url}      — only browser-loadable URLs are applied.
    """
    texts = texts or {}
    images = images or {}
    doc, _w = parse_safe(html_str)
    if doc is None:
        return html_str

    for el in doc.xpath(f"//*[@{SLOT_ATTR}]"):
        cls = _classify(el.get(SLOT_ATTR, ""))
        if not cls:
            continue
        kind, name = cls
        key = name or kind
        if kind == "text":
            val = texts.get(key)
            if val is None and name:
                val = texts.get(name)
            if val is None or str(val).strip() == "":
                continue                      # keep the author placeholder (operator sees the box)
            for child in list(el):            # textContent = val (clear element children)
                el.remove(child)
            el.text = str(val)
        elif kind == "image":
            src = images.get(key) or (images.get(name) if name else None)
            if not src or not _is_loadable_url(str(src)):
                continue
            src = str(src)
            if el.tag == "img":
                el.set("src", src)
            else:
                style = (el.get("style") or "").rstrip()
                if style and not style.endswith(";"):
                    style += ";"
                el.set("style", f"{style}background-image:url('{src}');"
                                f"background-size:cover;background-position:center;")
        # video / background / intro / logo / watermark: intentionally untouched.

    try:
        # Keep the author's doctype (lxml drops it on parse) so the builder iframe
        # renders in the SAME mode (standards) as the final render — WYSIWYG parity.
        return preserve_doctype(html_str, lxml_html.tostring(doc, encoding="unicode"))
    except Exception:
        return html_str
