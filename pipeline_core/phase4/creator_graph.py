"""
kaizer.pipeline.phase4.creator_graph
=====================================
Clip graph — promotes clips to first-class objects with typed edges.

Edges:
  series_part_of     — ClipA is part N of series owned by ClipZ
  trailer_for        — ClipA is a trailer pointing at long-form ClipZ
  variant_of         — ClipA is a platform variant of rendered master ClipZ
  reusable_source    — ClipA was used as B-roll inside ClipZ
  narrative_beat_of  — ClipA is turning-point N inside video Z

Why this is a moat
------------------
Competitors treat clips as flat rows. With typed edges:
  - "Which series had highest completion on Part 3?" is a single query.
  - "Find all trailers whose long-form got a retention lift" is trivial.
  - Per-edge benchmarks (trailer-to-long conversion %, series part-to-
    part drop-off) become first-class product features.

Storage
-------
v2 can use Postgres recursive CTEs on an (edge_type, src_clip_id,
dst_clip_id, meta JSONB) table. v3 may switch to Neo4j / Memgraph.

Phase 4 scope
-------------
  - Add `clip_edges` table migration
  - Edge insertion hooks in render_series.chain_parts and variants.generate_variants
  - GraphQL-style traversal API
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("kaizer.pipeline.phase4.creator_graph")


EDGE_TYPES = (
    "series_part_of",
    "trailer_for",
    "variant_of",
    "reusable_source",
    "narrative_beat_of",
)


@dataclass
class ClipEdge:
    edge_type: str
    src_clip_id: int
    dst_clip_id: int
    meta: dict


def link_clips(
    src_clip_id: int,
    dst_clip_id: int,
    *,
    edge_type: str,
    meta: dict | None = None,
    db=None,
) -> ClipEdge:
    """Record a typed edge between two clips.

    Phase 4 implementation: insert into `clip_edges` table with uniqueness
    on (edge_type, src, dst).
    """
    if edge_type not in EDGE_TYPES:
        raise ValueError(f"Unknown edge_type: {edge_type}. Must be one of {EDGE_TYPES}")
    logger.info(
        "link_clips(%s -> %s, edge_type=%s): Phase 4 stub", src_clip_id, dst_clip_id, edge_type,
    )
    return ClipEdge(edge_type=edge_type, src_clip_id=src_clip_id,
                    dst_clip_id=dst_clip_id, meta=meta or {})


def traverse(clip_id: int, *, edge_type: str, direction: str = "out", db=None) -> list[int]:
    """Return clip IDs connected to `clip_id` by `edge_type`.

    direction='out' → clips where src = clip_id.
    direction='in'  → clips where dst = clip_id.
    """
    raise NotImplementedError(
        "creator_graph.traverse is a Phase 4 task. "
        "See docs/PHASE4_ROADMAP.md § Creator Graph."
    )
