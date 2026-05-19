"""Singleton Inngest client for the V2 pipeline.

The client is constructed once on first import and reused by every
orchestrator + every dispatcher call site (runner.py's V2 branch
fires events through it).

Env vars (per the Beta launch contract -- Step 11 will set these in
the runner's startup):

  INNGEST_APP_ID            default "kaizer-v2"
  INNGEST_EVENT_KEY         signed event production key (prod only)
  INNGEST_SIGNING_KEY       signed signing key (prod only)
  INNGEST_DEV               "1" / "true" for Dev Server mode (no keys
                            required); the Inngest SDK auto-detects
                            but we surface the flag for clarity

In Dev Server mode the keys are optional and the client connects to
http://127.0.0.1:8288 by default. In production the keys MUST be set
in the .env / deployment env; the client raises at first send() if
they are missing.
"""

from __future__ import annotations

import os
from threading import Lock
from typing import Optional

from inngest import Inngest


_CLIENT: Optional[Inngest] = None
_CLIENT_LOCK = Lock()


def _is_dev_mode() -> bool:
    """Detect Inngest Dev Server mode.

    True if explicitly set OR if no event-key + signing-key are set
    (defensive fallback so a missing prod config doesn't accidentally
    talk to the production event store).
    """
    explicit = os.environ.get("INNGEST_DEV", "").strip().lower()
    if explicit in ("1", "true", "yes"):
        return True
    has_keys = bool(
        os.environ.get("INNGEST_EVENT_KEY", "").strip()
        and os.environ.get("INNGEST_SIGNING_KEY", "").strip()
    )
    return not has_keys


def get_client() -> Inngest:
    """Return the singleton Inngest client.

    Constructed lazily on first call so a process that imports this
    module but never sends events (e.g. tests that import the
    orchestrator module to inspect handlers) doesn't pay the SDK
    init cost or fail on missing env vars.
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            return _CLIENT
        app_id = os.environ.get("INNGEST_APP_ID", "kaizer-v2").strip() or "kaizer-v2"
        _CLIENT = Inngest(
            app_id=app_id,
            is_production=not _is_dev_mode(),
        )
        return _CLIENT


def reset_client_for_tests() -> None:
    """Drop the singleton so tests can re-init with patched env.

    Production code MUST NOT call this. Tests that mutate INNGEST_*
    env vars between cases should call this in setup/teardown.
    """
    global _CLIENT
    with _CLIENT_LOCK:
        _CLIENT = None
