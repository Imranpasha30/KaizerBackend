"""
kaizer.pipeline.phase4
=======================
Interface stubs for Phase 4 — the billion-dollar moat.

These modules establish the API shape for long-horizon features so
upstream callers (routers, workers, UI) can be written against stable
contracts today, even though real implementations span weeks to months
each.

Every public function in this package either:
  - Returns a well-formed empty/default result with a warning log
    noting Phase 4 status, OR
  - Raises NotImplementedError with a clear reference to the GSD
    roadmap document (docs/PHASE4_ROADMAP.md).

Subsystems
----------
  training_flywheel  — label-from-telemetry + per-niche model retrain.
  creator_graph      — clips as first-class objects with Part/Trailer edges.
  vertical_packs     — per-niche narrative prompt + scoring tuning.
  agency_mode        — multi-account RBAC + bulk asset management.
  pro_export         — FCPX / Premiere XML project export.
  music_marketplace  — licensed track catalogue + royalty split.
  trial_reels        — Meta Trial Reels API adapter + auto-promote.
  regional_api       — B2B newsroom plugin (Telugu / Hindi / Tamil).

Importing this package never performs I/O, database queries, or network
calls. Submodules may raise NotImplementedError at call time for
as-yet-unimplemented operations.
"""
from pipeline_core.phase4 import (
    agency_mode,
    creator_graph,
    music_marketplace,
    pro_export,
    regional_api,
    training_flywheel,
    trial_reels,
    vertical_packs,
)

__all__ = [
    "agency_mode",
    "creator_graph",
    "music_marketplace",
    "pro_export",
    "regional_api",
    "training_flywheel",
    "trial_reels",
    "vertical_packs",
]
