"""Prometheus ``/metrics`` endpoint — Phase 3.G.

Standard Prometheus text-exposition format. Intentionally **not**
auth-gated (typical pattern: Prometheus scrapes over the internal
network or behind a reverse-proxy basic-auth). The OBSERVABILITY.md
operator doc explains how to put auth in front if needed.

The endpoint computes ``collect_snapshot(db)`` on every scrape — gauge
values and counter deltas are read from the DB + scheduler snapshot at
call time, not pushed from inside the (read-only-to-us) services.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from sqlalchemy.orm import Session

from database import get_db
from services.metrics_collector import (
    collect_snapshot,
    render_latest,
    CONTENT_TYPE_LATEST,
)

log = logging.getLogger("kaizer.routers.metrics")

router = APIRouter(tags=["observability"])


@router.get("/metrics", response_class=Response)
def metrics(db: Session = Depends(get_db)) -> Response:
    """Prometheus scrape endpoint.

    Returns the standard text-exposition format with content-type
    ``text/plain; version=0.0.4; charset=utf-8``.

    Not auth-gated by design — Prometheus scrapes typically happen over
    the internal cluster network. For exposure to the public internet,
    wrap with reverse-proxy basic-auth or restrict by source IP.
    """
    try:
        collect_snapshot(db)
    except Exception as exc:
        # A snapshot failure must NOT 500 the /metrics endpoint —
        # Prometheus would alert on the scrape failure and the operator
        # wouldn't know which sub-collector broke. The collector
        # internally logs sub-failures; we still return whatever values
        # the registry has (last-known-good).
        log.exception("metrics: collect_snapshot raised: %s", exc)

    body = render_latest()
    return Response(content=body, media_type=CONTENT_TYPE_LATEST)
