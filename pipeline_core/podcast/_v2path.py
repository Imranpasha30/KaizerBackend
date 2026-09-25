# Ported from kaizer-platform@d5fd482 server/pipeline_core/podcast/_v2path.py.
# Changes from upstream: upstream bootstrapped server/pipeline_v2 onto sys.path and re-exported
# pipeline_v2.render.edl_builder.build_extraction_edl; pipeline_v2 was removed on this side, so
# this is now a thin compatibility re-export of the local pipeline_core/podcast/edl_builder.py.
"""Compatibility shim: re-export ``build_extraction_edl`` from the local copy.

Upstream, the podcast renderer reused ``pipeline_v2.render.edl_builder`` via a
one-line sys.path bootstrap here. ``pipeline_v2`` does not exist in this tree —
the pure-stdlib ``edl_builder`` module was ported into this package instead
(see ``pipeline_core/podcast/edl_builder.py``). This shim keeps any upstream
import path working without a sys.path hack.
"""

from __future__ import annotations

from pipeline_core.podcast.edl_builder import build_extraction_edl

__all__ = ["build_extraction_edl"]
