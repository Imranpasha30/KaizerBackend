"""Custom (developer-uploaded) HTML/CSS video templates for Kaizer X.

Pipeline: upload (.zip/.html) -> bundle.extract + sanitize -> contract.discover (slots
+ canvas) -> renderer.render (design PNG + video-slot rects, sandboxed) -> compose
(ffmpeg drops real clips into the slots + overlays the design) -> mp4.

See TEMPLATE_CONTRACT.md for the developer-facing rules.
"""
from .bundle import Bundle, BundleError, extract_bundle  # noqa: F401
from .contract import Slot, TemplateContract, aspect_kind, discover, parse_safe  # noqa: F401
from .infer import normalize_and_discover, normalize_template  # noqa: F401
from .filler import (  # noqa: F401
    ContentBundle, SlotFill, audit_template, build_slot_fill,
    derive_ticker, first_sentence, looks_like_placeholder,
)
from .ai_fit import fit_texts  # noqa: F401
from .ai_understand import understand_and_mark  # noqa: F401
from .engine import RenderRequest, prepare_bundle, render_preview, render_template  # noqa: F401
