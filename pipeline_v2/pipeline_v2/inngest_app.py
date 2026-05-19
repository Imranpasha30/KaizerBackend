"""V2 Inngest serve handler (Step 12.2b setup).

Exposes ``process_video_v2`` (and any future V2 Inngest functions)
on the host FastAPI app via ``inngest.fast_api.serve``. The Inngest
Dev Server (and production Inngest Cloud) discover the function by
polling the SDK's well-known serve endpoint at ``/api/inngest``
(``inngest._internal.const.DEFAULT_SERVE_PATH``).

Why this module exists
----------------------
Pre-Step-12.2b, V2 only used the Inngest client for ``client.send()``
(triggered from ``runner.py`` when a job is submitted) -- the function
itself was defined via ``@_inngest.create_function(...)`` in
``orchestrator.py`` but never exposed over HTTP. Inngest's worker
model requires the function to be reachable so the dev server can
fetch the function spec + drive step execution. This module wires
that up at process startup.

The hosting FastAPI app is ``main.py``'s singleton ``app``. The
registration happens AFTER all V1 routes are declared so the Inngest
route (mounted at ``/api/inngest``) can't be accidentally shadowed
by a V1 route. Registration is feature-flagged behind
``_v2_enabled()`` -- when V2 is off, the route doesn't mount and the
V1 production path is byte-identical to pre-V2.

Idempotency
-----------
``register_v2_inngest`` is safe to call multiple times -- a
module-level ``_REGISTERED`` flag short-circuits subsequent calls.
This matters for test pytest fixtures that import main.py multiple
times across cases, and for any future hot-reload setup.
"""

from __future__ import annotations

import logging

import inngest.fast_api
from fastapi import FastAPI

from pipeline_v2.inngest_client import get_client
from pipeline_v2.orchestrator import process_video_v2


logger = logging.getLogger("pipeline_v2.inngest_app")


# Module-level idempotency flag. Reset for tests via
# ``_reset_for_tests`` so re-imports don't false-trip the guard.
_REGISTERED = False


def register_v2_inngest(app: FastAPI) -> None:
    """Mount the V2 Inngest serve endpoint on ``app``.

    Wraps ``inngest.fast_api.serve(app, client, [process_video_v2])``.
    The default serve path is ``/api/inngest`` -- the canonical
    Inngest convention (``inngest._internal.const.DEFAULT_SERVE_PATH``).

    Safe to call multiple times: subsequent calls are no-ops.
    Callers (notably ``main.py``) should only call this after all
    V1 routes have been declared so the Inngest route's path can't
    collide with anything V1.
    """
    global _REGISTERED
    if _REGISTERED:
        logger.debug(
            "register_v2_inngest: already registered, skipping"
        )
        return
    inngest.fast_api.serve(
        app,
        get_client(),
        [process_video_v2],
    )
    _REGISTERED = True
    logger.info(
        "register_v2_inngest: mounted process_video_v2 at /api/inngest"
    )


def _reset_for_tests() -> None:
    """Drop the idempotency guard so tests can re-register.

    Production code MUST NOT call this. Tests that reload main.py
    with different env vars across cases use this in setup/teardown.
    """
    global _REGISTERED
    _REGISTERED = False
