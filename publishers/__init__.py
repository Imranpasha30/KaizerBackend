"""Publisher registry — every concrete platform Publisher is registered
here under its ``provider_key`` so the worker can look up the right
one for a given destination.

Adding a new platform = drop a new file in this package + register the
class below. The worker code never needs to know what platforms exist.
"""
from .base import (
    Publisher,
    PublishError,
    TransientPublishError,
    TerminalPublishError,
    QuotaExceededError,
    PrepareResult,
    PublishResult,
    VerifyResult,
)


# ── Registry ────────────────────────────────────────────────────────

# provider_key -> Publisher subclass
_REGISTRY: dict[str, type[Publisher]] = {}


def register(cls: type[Publisher]) -> type[Publisher]:
    """Decorator: registers a Publisher subclass under its provider_key.

    Usage:
        @register
        class YoutubeDirectPublisher(Publisher):
            provider_key = "youtube_direct"
            ...
    """
    if not cls.provider_key:
        raise ValueError(f"{cls.__name__}: provider_key must be set")
    if cls.provider_key in _REGISTRY:
        raise ValueError(
            f"provider_key {cls.provider_key!r} already registered to "
            f"{_REGISTRY[cls.provider_key].__name__}"
        )
    _REGISTRY[cls.provider_key] = cls
    return cls


def get_publisher_class(provider_key: str) -> type[Publisher]:
    """Look up a Publisher class by provider key. Raises KeyError if
    no such publisher is registered — the caller (typically the worker)
    can decide whether to fall back to a legacy path or flip the job to
    provider_failed."""
    try:
        return _REGISTRY[provider_key]
    except KeyError:
        raise KeyError(
            f"no Publisher registered for provider_key {provider_key!r}. "
            f"Known: {sorted(_REGISTRY.keys())}"
        )


def list_providers() -> list[dict[str, str]]:
    """Payload for the frontend's destination-picker dropdown."""
    return [
        {"key": cls.provider_key, "name": cls.display_name}
        for cls in _REGISTRY.values()
    ]


# ── Eager-import concrete publishers so their @register decorators fire ──

# Each module is wrapped in try/except so a missing dependency on one
# platform (e.g. Meta SDK) doesn't break import of the whole package.
def _safe_import(modname: str) -> None:
    try:
        __import__(f"publishers.{modname}", fromlist=["*"])
    except Exception as exc:  # noqa: BLE001 — diagnostic logging is intentional
        # Logged at module-import time so a misconfigured publisher is
        # visible at worker boot, not silently silent for an hour.
        import sys
        print(
            f"[publishers] failed to import {modname!r}: {exc}",
            file=sys.stderr,
            flush=True,
        )


_safe_import("youtube_direct")
_safe_import("postiz")
_safe_import("meta_facebook")
_safe_import("meta_instagram")
_safe_import("x_twitter")
_safe_import("linkedin")
_safe_import("tiktok")

# Background workers (idempotent — only start when their platform is
# actually configured via env vars).
try:
    from .meta_token_refresh import start_meta_refresh_loop
    start_meta_refresh_loop()
except Exception as exc:  # noqa: BLE001
    import sys
    print(f"[publishers] meta_token_refresh boot failed: {exc}",
          file=sys.stderr, flush=True)
