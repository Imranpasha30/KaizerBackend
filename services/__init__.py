"""Kaizer Publish/Upload v2 services package.

Owned by multiple agents per CONTRACTS.md §1:
  - B (Fanout):     ``fanout.py``
  - C (Scheduler):  ``scheduler.py`` (sibling package, owned by C-agent)
  - D (Branding):   ``branding/`` (owned by D-agent)
  - E (Upload):     ``upload/`` (owned by E-agent)
  - F (Idempotency+Quota+Credits): ``credits.py``, ``idempotency.py``,
                    ``burn/`` (owned by F-agent, Phase 2 full impl)

Phase 1.B ships the minimum needed for ``POST /api/publish-tasks``:
  - ``fanout.create_publish_task(...)`` — the fan-out service
  - ``credits.reserve / refund / get_balance`` — minimal stub the
    F-agent will replace with the full row-locking implementation
    in Phase 2
  - ``idempotency.compute_key(...)`` — the only F-agent function the
    Fanout layer needs in Phase 1

All other F-agent surfaces (``check_or_register``, ``record_success``,
``recover_orphans``, ``allot_monthly``, ``reconcile_window``) are
explicitly NOT in this commit — they show up in Phase 2 when the
upload workers exist to call them.
"""

# Re-export scheduler entry points for cleaner imports + so the
# Fanout service's ``from services.scheduler import scheduler_enqueue``
# fallback chain hits the real module on the first try.
try:
    from services.scheduler import (  # noqa: F401
        scheduler_enqueue,
        start as scheduler_start,
        shutdown as scheduler_shutdown,
        snapshot as scheduler_snapshot,
    )
except Exception:  # pragma: no cover — degrades to fallback at import sites
    pass

# Re-export Branding-agent (D) public surface so callers don't have to
# know whether the impl lives in a module or a package. Lazy/guarded so
# branding's optional deps (boto3, PIL) don't break imports of other
# services when branding can't load.
try:
    from services.branding import (  # noqa: F401
        process_upload_job as branding_process_upload_job,
        brand_artifact_cache_key,
        cleanup_expired as branding_cleanup_expired,
        snapshot as branding_snapshot,
    )
    from services.brand_resolver import (  # noqa: F401
        resolve_brand_profile,
        ResolvedBrand,
    )
except Exception:  # pragma: no cover
    pass

# Re-export Upload-agent (E) entry point so the orchestrator can call
# ``services.upload_dispatch_process(id)`` without knowing the module
# layout. The actual implementation lives in ``services/upload_dispatch.py``.
try:
    from services.upload_dispatch import (  # noqa: F401
        process as upload_dispatch_process,
        PlanTierViolation as UploadPlanTierViolation,
    )
except Exception:  # pragma: no cover
    pass

# ─── Phase 2.F surface re-exports ───────────────────────────────────
# The F-agent owns these four modules; expose the full surface so
# callers can `from services import idempotency, credits, burn_log`.
try:
    from services import idempotency  # noqa: F401
    from services import credits  # noqa: F401
    from services import burn_log  # noqa: F401
except Exception:  # pragma: no cover — degrade gracefully in partial installs
    pass
