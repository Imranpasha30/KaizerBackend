"""Bundle handling for uploaded custom templates: safe extraction + sanitization.

An upload is either a ``.zip`` (index.html + css + assets/) or a single ``.html`` file.
This module:
  * extracts a zip safely (zip-slip, size-bomb, and file-type guards),
  * locates the entry HTML,
  * neutralizes EXTERNAL references (http/https/protocol-relative URLs, remote
    scripts/links, iframes) so the network-isolated renderer only ever touches local,
    bundled assets.

No rendering here — see renderer.py.
"""
from __future__ import annotations

import os
import re
import shutil
import zipfile
from dataclasses import dataclass

from lxml import html as lxml_html

MAX_BUNDLE_BYTES = 120 * 1024 * 1024    # 120 MB uncompressed (templates may bundle default videos)
MAX_FILES = 300
ALLOWED_EXT = {
    ".html", ".htm", ".css", ".js",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".avif", ".bmp",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".json", ".map",
    # bundled default media (background / intro / image defaults the user can swap)
    ".mp4", ".webm", ".mov", ".m4v",
}
_EXTERNAL_RE = re.compile(r"^\s*(?:[a-zA-Z][a-zA-Z0-9+.-]*:)?//", re.I)   # http://, https://, //cdn
_CSS_EXTERNAL_URL_RE = re.compile(r"""url\(\s*['"]?\s*((?:[a-z][a-z0-9+.-]*:)?//[^)'"]+)['"]?\s*\)""", re.I)
_CSS_IMPORT_RE = re.compile(r"""@import\s+(?:url\()?\s*['"]?((?:[a-z][a-z0-9+.-]*:)?//[^)'";]+)""", re.I)
# url() carrying a code-execution scheme (javascript:/vbscript:/data:) — but ALLOW safe raster
# data images (png/jpeg/gif/webp/avif/bmp). Blocks data:image/svg+xml (SVG can run onload), data:
# text/*, data:application/*, etc., in any CSS property (background, mask-image, filter, …).
_CSS_DANGER_URL_RE = re.compile(
    r"""url\(\s*['"]?\s*(?:javascript:|vbscript:|data:(?!image/(?:png|jpe?g|gif|webp|avif|bmp)))[^)]*\)""",
    re.I)


class BundleError(ValueError):
    """Raised when an uploaded bundle is unsafe or malformed."""


@dataclass
class Bundle:
    root_dir: str            # absolute dir holding the extracted/normalized template
    entry_rel: str           # entry html relative to root_dir (e.g. "index.html")
    files: list[str]         # relative paths of all bundled files

    @property
    def entry_path(self) -> str:
        return os.path.join(self.root_dir, self.entry_rel)


def _is_safe_member(name: str) -> bool:
    if not name or name.startswith("/") or name.startswith("\\"):
        return False
    norm = os.path.normpath(name).replace("\\", "/")
    if norm.startswith("../") or "/../" in norm or norm == ".." or os.path.isabs(norm):
        return False
    return True


def _find_entry(root_dir: str) -> str:
    """Pick the entry HTML: prefer index.html (shallowest), else the only/first .html."""
    candidates: list[str] = []
    for dirpath, _dirs, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith((".html", ".htm")):
                rel = os.path.relpath(os.path.join(dirpath, f), root_dir).replace("\\", "/")
                candidates.append(rel)
    if not candidates:
        raise BundleError("No HTML file found in the bundle — include an index.html.")
    candidates.sort(key=lambda r: (r.count("/"), 0 if os.path.basename(r).lower() == "index.html" else 1, r))
    return candidates[0]


def extract_bundle(src_path: str, dest_dir: str) -> Bundle:
    """Extract/copy an uploaded template into ``dest_dir`` safely. ``src_path`` is a
    ``.zip`` or a single ``.html``. Returns a :class:`Bundle`. Raises BundleError."""
    os.makedirs(dest_dir, exist_ok=True)
    lower = src_path.lower()

    if lower.endswith((".html", ".htm")):
        shutil.copyfile(src_path, os.path.join(dest_dir, "index.html"))
        return Bundle(root_dir=dest_dir, entry_rel="index.html", files=["index.html"])

    if not zipfile.is_zipfile(src_path):
        raise BundleError("Upload must be a .zip bundle or a single .html file.")

    files: list[str] = []
    total = 0
    with zipfile.ZipFile(src_path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if len(members) > MAX_FILES:
            raise BundleError(f"Too many files in bundle ({len(members)} > {MAX_FILES}).")
        _mb = MAX_BUNDLE_BYTES // (1024 * 1024)
        for m in members:
            if not _is_safe_member(m.filename):
                raise BundleError(f"Unsafe path in zip: {m.filename!r}")
            ext = os.path.splitext(m.filename)[1].lower()
            if ext not in ALLOWED_EXT:
                raise BundleError(f"Disallowed file type {ext or '(none)'} in bundle: {m.filename!r}")
            # Early reject on the DECLARED size (cheap), but the real enforcement is the
            # streaming cap below — a zip-bomb can forge file_size in the header.
            total += m.file_size
            if total > MAX_BUNDLE_BYTES:
                raise BundleError(f"Bundle too large (over {_mb} MB uncompressed).")
        written = 0
        for m in members:
            target = os.path.join(dest_dir, os.path.normpath(m.filename))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            # Stream with a HARD running cap on actual bytes written (defeats a forged
            # central-directory file_size / decompression bomb on the real read).
            with zf.open(m) as srcf, open(target, "wb") as dstf:
                while True:
                    chunk = srcf.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_BUNDLE_BYTES:
                        try:
                            dstf.close()
                            os.remove(target)
                        except Exception:
                            pass
                        raise BundleError(f"Bundle too large (over {_mb} MB uncompressed).")
                    dstf.write(chunk)
            files.append(os.path.relpath(target, dest_dir).replace("\\", "/"))

    entry_rel = _find_entry(dest_dir)
    return Bundle(root_dir=dest_dir, entry_rel=entry_rel, files=files)


# Code-execution URL schemes blocked in URL-bearing attributes. data:text/* (html/js),
# data:image/svg* (SVG can carry onload/script), and data:application/* (xml/xhtml) are all
# script-capable; safe raster data images (data:image/png, jpeg, …) are still allowed.
_DANGER_SCHEMES = ("javascript:", "vbscript:", "data:text/", "data:image/svg",
                   "data:application/")


def neutralize_external_html(html_str: str) -> tuple[str, int]:
    """Harden untrusted template HTML so rendering is local-only and script-free:
      * drop ALL <script> (inline + remote), <iframe>/<object>/<embed>/<base>, and
        <meta http-equiv=refresh>;
      * strip every inline event handler (on*) attribute;
      * blank any URL-bearing attribute (src/href/srcset/xlink:href/poster/…) whose value
        is external (http(s)://, //) or a dangerous scheme (javascript:/vbscript:/data:text/html);
      * blank external/js url() in inline style.
    CSS animations still work (no JS needed). Returns (clean_html, removed_count).
    NOTE: this is the upload-time layer; the renderer's network guard is the runtime layer."""
    if not html_str.strip():
        return html_str, 0
    doc = lxml_html.fromstring(html_str)   # NOTE: lxml drops the doctype here
    removed = 0

    # Drop elements that execute code or navigate, regardless of attributes.
    for tag in ("script", "iframe", "object", "embed", "base"):
        for el in doc.xpath(f"//{tag}"):
            el.drop_tree()
            removed += 1
    for el in doc.xpath("//meta[@http-equiv]"):
        if (el.get("http-equiv") or "").strip().lower() == "refresh":
            el.drop_tree()
            removed += 1

    for el in doc.iter():
        for attr in list(el.attrib.keys()):
            la = attr.lower()
            val = (el.get(attr) or "")
            v = val.strip().lower()
            if la.startswith("on"):                       # inline event handler → kill
                el.attrib.pop(attr, None); removed += 1; continue
            url_attr = (la in ("src", "href", "srcset", "data", "poster", "action",
                               "formaction", "background") or la.endswith("href"))
            if url_attr and (_EXTERNAL_RE.match(val) or v.startswith(_DANGER_SCHEMES)):
                el.attrib.pop(attr, None); removed += 1; continue
            if (not url_attr) and v.startswith(_DANGER_SCHEMES):
                el.attrib.pop(attr, None); removed += 1
        style = el.get("style")
        if style:
            new = _CSS_DANGER_URL_RE.sub("url()", style)   # url(javascript:/data:svg/…) → url()
            new = _CSS_EXTERNAL_URL_RE.sub("url()", new)    # url(//cdn) / url(https:) → url()
            if new != style:
                el.set("style", new)
                removed += 1

    clean = lxml_html.tostring(doc, encoding="unicode")
    # Re-attach the author's doctype: without it stored bundles render in QUIRKS mode
    # (lxml drops it at fromstring above) — a standards-correctness hazard at render time.
    from .contract import preserve_doctype
    return preserve_doctype(html_str, clean), removed


def neutralize_external_css(css_str: str) -> tuple[str, int]:
    """Blank external url()/@import in a CSS string. Returns (clean_css, removed)."""
    removed = 0
    def _u(m):
        nonlocal removed
        removed += 1
        return "url()"
    out = _CSS_DANGER_URL_RE.sub(_u, css_str)      # url(javascript:/data:svg/…) → url()
    out = _CSS_EXTERNAL_URL_RE.sub(_u, out)         # url(//cdn) / url(https:) → url()
    out2 = _CSS_IMPORT_RE.sub("/* external @import removed */", out)
    if out2 != out:
        removed += 1
    return out2, removed
