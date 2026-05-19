"""V2 image sourcing for Stage 4 (Step 9.2 pre-step).

Policy-compliant entity-image resolver. V2 owns the policy; V1 owns
the primitives (CSE/DDG/Pexels search, downloader, OpenAI/Imagen
generation).

POLICY (locked by operator; see Step 12.2a re-run #2 decision):

  * For ``Entity.type == PERSON``:
      - SEARCH-ONLY chain: Google CSE -> DuckDuckGo -> Pexels.
      - On miss across all three sources: status "image_unavailable".
      - NEVER generate. Hard rule. AI-rendered faces of real public
        figures are a misinformation vector we do not ship.
      - Logged with structured slug ``person_image_unavailable``.

  * For non-PERSON entities (PLACE / ORG / EVENT / OTHER):
      - Search-first (same CSE -> DDG -> Pexels chain).
      - On search miss: fall back to OpenAI gpt-image-1 generation.
      - On OpenAI refusal / miss: fall back to Google Imagen generation.
      - On all backends miss: status "image_missing".
      - Logged with structured slug ``non_person_image_missing``.

This module is the ONLY place V2 calls V1's image-sourcing primitives.
Any change to V1's signatures must flow through here, not through
``stage_4_render.py`` directly. Keeps the V2 -> V1 boundary contained.

Backlog item 61: V1 production's ``generate_image_pool_from_plan``
violates this same policy (pure-generation for every image_plan
entity, including PERSON). V2 deliberately does NOT inherit that
behaviour -- this module is V2's policy-compliant replacement.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional


def _ensure_kaizer_backend_on_path() -> None:
    """Mirror of the path-setup in stage_4_render.py.

    Adds ``KaizerBackend/`` to ``sys.path`` so the V1 imports below
    resolve. Defensive; the production dispatcher + ``conftest.py``
    set this up already.
    """
    here = Path(__file__).resolve()
    kaizer_backend = here.parents[3]
    s = str(kaizer_backend)
    if s not in sys.path:
        sys.path.insert(0, s)


_ensure_kaizer_backend_on_path()


# V1 primitives. Mirror of the naming convention used in stage_4_render.py.
from pipeline_core.pipeline import (   # noqa: E402
    _search_google_cse as _v1_search_google_cse,
    _search_duckduckgo as _v1_search_duckduckgo,
    _search_pexels as _v1_search_pexels,
    _download_image as _v1_download_image,
)
from pipeline_core import openai_images as _v1_openai_images   # noqa: E402
from pipeline_core import imagen as _v1_imagen                 # noqa: E402

from pipeline_v2.models import Entity, EntityType   # noqa: E402


logger = logging.getLogger("pipeline_v2.image_source")


# ---- Structured log slugs -------------------------------------------------

PERSON_IMAGE_UNAVAILABLE = "person_image_unavailable"
NON_PERSON_IMAGE_MISSING = "non_person_image_missing"


# ---- Helpers --------------------------------------------------------------


_SAFE_NAME_RE = re.compile(r"[^a-z0-9]+")


def _safe_filename(name: str) -> str:
    """``Bandi Bhagirath`` -> ``bandi_bhagirath``. Filename-safe slug."""
    s = _SAFE_NAME_RE.sub("_", name.strip().lower()).strip("_")
    return s or "image"


def _build_person_query(entity: Entity, brief: str) -> str:
    """Search query for PERSON entities.

    Mirrors V1's ``search_news_images`` people-query pattern
    (pipeline.py:1779): ``"{person} news photo"``. Brief context is
    intentionally NOT appended -- for politicians/celebrities, the
    "news photo" suffix yields cleaner search results than a long
    descriptive query.
    """
    return f"{entity.canonical_name} news photo"


def _build_non_person_query(entity: Entity, brief: str) -> str:
    """Search/generation query for non-PERSON entities.

    Combines entity name + a short brief snippet so the result has
    topical relevance (a Hyderabad story about a court ruling should
    return a courtroom/government photo, not a tourist skyline).
    """
    brief_snippet = (brief or "").strip()
    if brief_snippet:
        # Truncate brief to ~120 chars so the query stays focused.
        brief_snippet = brief_snippet[:120]
        return f"{entity.canonical_name} {brief_snippet}".strip()
    return entity.canonical_name


# ---- ImageSourcer ---------------------------------------------------------


class ImageSourcer:
    """Resolve one V2 ``Entity`` to a real on-disk image, per policy.

    Construct with a language code (drives OpenAI/Imagen location
    hints). The class is intentionally stateless beyond the language
    setting -- caller manages the output directory + the cross-entity
    pool (``Stage4Render.image_pool``).

    Public surface: ``source_for_entity()``. The class can be
    monkey-patched in tests by mocking the V1 search/generation
    primitives at the module level (e.g.
    ``patch("pipeline_v2.stages.stage_4_image_source._v1_search_google_cse",
    return_value=["http://..."])``).
    """

    def __init__(self, language: str = "en") -> None:
        self.language = (language or "en").split("-", 1)[0]

    # ---- Public surface ---------------------------------------------------

    def source_for_entity(
        self,
        entity: Entity,
        brief: str,
        out_dir: Path,
    ) -> Optional[Path]:
        """Resolve ``entity`` to an on-disk image, applying the locked
        policy. Returns the absolute Path on success; ``None`` on the
        terminal miss path (caller treats as ``image_unavailable``
        for PERSON / ``image_missing`` for others).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if entity.type == EntityType.PERSON:
            return self._source_person(entity, brief, out_dir)
        return self._source_non_person(entity, brief, out_dir)

    # ---- PERSON path: search-only -----------------------------------------

    def _source_person(
        self,
        entity: Entity,
        brief: str,
        out_dir: Path,
    ) -> Optional[Path]:
        query = _build_person_query(entity, brief)
        path = self._search_chain(query, entity, out_dir)
        if path is not None:
            return path

        # All three sources missed. Log structured warning + return
        # None. Hard rule: no generation fallback for PERSON.
        logger.warning(
            "image_source: PERSON entity has no available real image; "
            "skipping generation per policy (entity=%s)",
            entity.canonical_name,
            extra={
                "event":         PERSON_IMAGE_UNAVAILABLE,
                "entity":        entity.canonical_name,
                "entity_type":   "PERSON",
                "queries_tried": [query],
                "sources_tried": ["cse", "ddg", "pexels"],
            },
        )
        return None

    # ---- Non-PERSON path: search-first, generate as last resort -----------

    def _source_non_person(
        self,
        entity: Entity,
        brief: str,
        out_dir: Path,
    ) -> Optional[Path]:
        query = _build_non_person_query(entity, brief)

        path = self._search_chain(query, entity, out_dir)
        if path is not None:
            return path

        # Search miss across CSE/DDG/Pexels. Try generation.
        path = self._generate_openai(entity, query, brief, out_dir)
        if path is not None:
            return path

        path = self._generate_imagen(entity, query, brief, out_dir)
        if path is not None:
            return path

        logger.warning(
            "image_source: non-PERSON entity has no available image "
            "across search + generation (entity=%s type=%s)",
            entity.canonical_name, entity.type.value,
            extra={
                "event":         NON_PERSON_IMAGE_MISSING,
                "entity":        entity.canonical_name,
                "entity_type":   entity.type.value,
                "queries_tried": [query],
                "sources_tried": ["cse", "ddg", "pexels", "openai", "imagen"],
            },
        )
        return None

    # ---- Search chain (shared by PERSON + non-PERSON) ---------------------

    def _search_chain(
        self,
        query: str,
        entity: Entity,
        out_dir: Path,
    ) -> Optional[Path]:
        """CSE -> DDG -> Pexels. First successful download wins.

        Each searcher returns a list of URLs; we attempt download in
        order and return on the first viable file (>2 KB per V1's
        ``_download_image`` threshold). On any individual download
        failure we move to the next URL within the same source; on
        empty URL list we move to the next source.
        """
        safe = _safe_filename(entity.canonical_name)
        # V1's order: Google CSE -> DuckDuckGo -> Pexels. Kept per
        # operator decision (see Step 12.2a re-run #2 spec). DDG
        # placed before Pexels because DDG returns news-context
        # images while Pexels is stock-only.
        for source_name, search_fn, count in (
            ("cse",    _v1_search_google_cse, 3),
            ("ddg",    _v1_search_duckduckgo, 4),
            ("pexels", _v1_search_pexels,     3),
        ):
            try:
                urls = search_fn(query, count=count) or []
            except Exception as exc:
                logger.warning(
                    "image_source: %s search error for %r: %s",
                    source_name, query, exc,
                )
                continue

            for url_idx, url in enumerate(urls):
                dest = out_dir / f"{safe}__{source_name}_{url_idx:02d}.jpg"
                try:
                    ok = _v1_download_image(url, str(dest))
                except Exception as exc:
                    logger.warning(
                        "image_source: %s download error for %s: %s",
                        source_name, url, exc,
                    )
                    continue
                if ok and dest.exists() and dest.stat().st_size > 2_000:
                    logger.info(
                        "image_source: resolved %s via %s (%s)",
                        entity.canonical_name, source_name, dest.name,
                    )
                    return dest

        return None

    # ---- Generation backends (non-PERSON only) ----------------------------

    def _generate_openai(
        self,
        entity: Entity,
        query: str,
        brief: str,
        out_dir: Path,
    ) -> Optional[Path]:
        """OpenAI gpt-image-1. Returns Path on success, None on
        refusal/failure/disabled.

        Per V1's TRACK-2 pattern (pipeline.py:3958), we pass
        ``entities=[]`` so the prompt template doesn't try to render
        a specific name (the V1 prompt explicitly forbids real public
        figures and Imagen has similar guards). ``topics`` carries
        the entity's canonical_name + brief.
        """
        if not _v1_openai_images.is_enabled():
            return None
        safe = _safe_filename(entity.canonical_name)
        out_path = out_dir / f"{safe}__openai.jpg"
        try:
            result = _v1_openai_images.generate_news_image(
                query=query,
                entities=[],
                topics=[entity.canonical_name],
                language=self.language,
                out_path=str(out_path),
            )
        except Exception as exc:
            logger.warning(
                "image_source: openai generation exception for %s: %s",
                entity.canonical_name, exc,
            )
            return None
        if result and out_path.exists() and out_path.stat().st_size > 10_000:
            logger.info(
                "image_source: resolved %s via openai (%s)",
                entity.canonical_name, out_path.name,
            )
            return out_path
        return None

    def _generate_imagen(
        self,
        entity: Entity,
        query: str,
        brief: str,
        out_dir: Path,
    ) -> Optional[Path]:
        """Google Imagen. Same conventions as OpenAI path."""
        if not _v1_imagen.is_enabled():
            return None
        safe = _safe_filename(entity.canonical_name)
        out_path = out_dir / f"{safe}__imagen.jpg"
        try:
            result = _v1_imagen.generate_news_image(
                query=query,
                description=brief or None,
                entities=[],
                topics=[entity.canonical_name],
                language=self.language,
                out_path=str(out_path),
            )
        except Exception as exc:
            logger.warning(
                "image_source: imagen generation exception for %s: %s",
                entity.canonical_name, exc,
            )
            return None
        if result and out_path.exists() and out_path.stat().st_size > 10_000:
            logger.info(
                "image_source: resolved %s via imagen (%s)",
                entity.canonical_name, out_path.name,
            )
            return out_path
        return None
