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
v2 uses Postgres / SQLite on a (edge_type, src_clip_id, dst_clip_id,
edge_metadata JSONB) table with a unique constraint.
v3 may switch to Neo4j / Memgraph.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

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
    meta: Optional[dict] = None,
    db: Optional[Session] = None,
) -> ClipEdge:
    """Record a typed edge between two clips.

    INSERTs into `clip_edges` table with uniqueness on
    (edge_type, src_clip_id, dst_clip_id). On conflict returns the
    existing edge without raising. When db is None, returns an in-memory
    dataclass only (useful for testing without a DB).
    """
    if edge_type not in EDGE_TYPES:
        raise ValueError(f"Unknown edge_type: {edge_type}. Must be one of {EDGE_TYPES}")

    edge_meta = meta or {}

    if db is None:
        logger.debug(
            "link_clips(%s -> %s, edge_type=%s): no db — in-memory only",
            src_clip_id, dst_clip_id, edge_type,
        )
        return ClipEdge(
            edge_type=edge_type,
            src_clip_id=src_clip_id,
            dst_clip_id=dst_clip_id,
            meta=edge_meta,
        )

    import models  # type: ignore

    # Check for existing edge first (avoids try/except on every call)
    existing = (
        db.query(models.ClipEdge)
        .filter(
            models.ClipEdge.edge_type == edge_type,
            models.ClipEdge.src_clip_id == src_clip_id,
            models.ClipEdge.dst_clip_id == dst_clip_id,
        )
        .first()
    )
    if existing is not None:
        logger.debug(
            "link_clips: edge already exists (id=%s)", existing.id
        )
        return ClipEdge(
            edge_type=existing.edge_type,
            src_clip_id=existing.src_clip_id,
            dst_clip_id=existing.dst_clip_id,
            meta=existing.edge_metadata or {},
        )

    row = models.ClipEdge(
        edge_type=edge_type,
        src_clip_id=src_clip_id,
        dst_clip_id=dst_clip_id,
        edge_metadata=edge_meta,
    )
    db.add(row)
    try:
        db.commit()
        db.refresh(row)
        logger.info(
            "link_clips: inserted edge id=%s (%s -> %s, type=%s)",
            row.id, src_clip_id, dst_clip_id, edge_type,
        )
    except IntegrityError:
        db.rollback()
        # Race condition — another process inserted concurrently; load it
        existing = (
            db.query(models.ClipEdge)
            .filter(
                models.ClipEdge.edge_type == edge_type,
                models.ClipEdge.src_clip_id == src_clip_id,
                models.ClipEdge.dst_clip_id == dst_clip_id,
            )
            .first()
        )
        if existing is not None:
            return ClipEdge(
                edge_type=existing.edge_type,
                src_clip_id=existing.src_clip_id,
                dst_clip_id=existing.dst_clip_id,
                meta=existing.edge_metadata or {},
            )
        # Shouldn't happen, but don't crash
        logger.error("link_clips: IntegrityError but no existing row found")
        return ClipEdge(
            edge_type=edge_type,
            src_clip_id=src_clip_id,
            dst_clip_id=dst_clip_id,
            meta=edge_meta,
        )

    return ClipEdge(
        edge_type=row.edge_type,
        src_clip_id=row.src_clip_id,
        dst_clip_id=row.dst_clip_id,
        meta=row.edge_metadata or {},
    )


def traverse(
    clip_id: int,
    *,
    edge_type: str,
    direction: str = "out",
    db: Optional[Session] = None,
) -> list[int]:
    """Return clip IDs connected to `clip_id` by `edge_type`.

    direction='out' → clips where src_clip_id = clip_id  (clip_id → ?)
    direction='in'  → clips where dst_clip_id = clip_id  (? → clip_id)
    Unknown direction raises ValueError.
    Returns [] when db is None or no edges exist.
    """
    if direction not in ("out", "in"):
        raise ValueError(
            f"Unknown direction: {direction!r}. Must be 'out' or 'in'."
        )

    if db is None:
        logger.debug(
            "traverse(clip_id=%s, edge_type=%s, direction=%s): no db",
            clip_id, edge_type, direction,
        )
        return []

    import models  # type: ignore

    try:
        if direction == "out":
            rows = (
                db.query(models.ClipEdge.dst_clip_id)
                .filter(
                    models.ClipEdge.src_clip_id == clip_id,
                    models.ClipEdge.edge_type == edge_type,
                )
                .all()
            )
            return [r[0] for r in rows]
        else:  # direction == "in"
            rows = (
                db.query(models.ClipEdge.src_clip_id)
                .filter(
                    models.ClipEdge.dst_clip_id == clip_id,
                    models.ClipEdge.edge_type == edge_type,
                )
                .all()
            )
            return [r[0] for r in rows]
    except Exception as exc:
        logger.error(
            "traverse(clip_id=%s, edge_type=%s, direction=%s) error: %s",
            clip_id, edge_type, direction, exc, exc_info=True,
        )
        return []
