# Ported from kaizer-platform@d5fd482 server/avatar/__init__.py.
# Changes from upstream: the default provider is now SMART — "avatar_studio"
# when its studio checkout exists, else "heygen" (avatar-studio is a private
# repo not yet cloned here; without this, bare /generate calls die with
# "avatar-studio not found"). Explicit name arg and AVATAR_PROVIDER env still
# override, unchanged.
"""Avatar engine registry — env-selected provider behind one contract.

    from avatar import get_provider
    provider = get_provider()            # AVATAR_PROVIDER env, smart default
    provider = get_provider("echomimic") # explicit (e.g. premium tier)

Providers:
    avatar_studio  local MuseTalk farm (default when its checkout exists)
    echomimic      vast.ai GPU burst (premium/hero clips)
    heygen         remote API (OFF unless AVATAR_HEYGEN_ENABLED=true;
                   the fallback default while avatar-studio is absent)

Provider modules are imported lazily: a broken/missing dependency in one
engine (e.g. httpx for heygen) must never take down the others.
"""
from __future__ import annotations

import importlib
import os

from .base import (
    AvatarInfo,
    AvatarProvider,
    AvatarProviderError,
    GenerationRequest,
    GenerationResult,
    VoiceInfo,
    gender_counts,
)

# The PREFERRED default when the local studio checkout is present.
DEFAULT_PROVIDER = "avatar_studio"


def default_provider() -> str:
    """Smart default: "avatar_studio" when its studio dir exists, else "heygen".

    PORT CHANGE vs upstream (which hardcoded DEFAULT_PROVIDER): avatar-studio
    is a private repo that may not be cloned next to this backend yet — until
    it is, defaulting to it would make every bare get_provider() call fail
    with "avatar-studio not found". Explicit names and AVATAR_PROVIDER env
    still win (see get_provider); this only decides the unconfigured default.
    """
    try:
        from .studio import studio_dir
        if (studio_dir() / "generate.py").exists():
            return DEFAULT_PROVIDER
    except Exception:  # noqa: BLE001 — a broken studio probe must not kill the registry
        pass
    return "heygen"


# name -> (module, class). Imported on first get_provider(name).
_REGISTRY: dict[str, tuple[str, str]] = {
    "avatar_studio": ("avatar.studio", "AvatarStudioProvider"),
    "echomimic": ("avatar.echomimic", "EchoMimicBurstProvider"),
    "heygen": ("avatar.heygen_provider", "HeyGenProvider"),
}

# Instantiated lazily and cached — providers are stateless beyond config.
_instances: dict[str, AvatarProvider] = {}


def provider_names() -> list[str]:
    return list(_REGISTRY)


def get_provider(name: str | None = None) -> AvatarProvider:
    """Return the requested provider (or the env-configured default).

    Resolution order: explicit ``name`` > ``AVATAR_PROVIDER`` env >
    smart default (see :func:`default_provider`).

    Raises AvatarProviderError for unknown names or engines whose module
    can't import — callers surface the message instead of 500ing.
    """
    key = (name or os.environ.get("AVATAR_PROVIDER") or default_provider()).strip().lower()
    if key not in _REGISTRY:
        raise AvatarProviderError(
            f"unknown avatar provider '{key}' — choose from {provider_names()}")
    if key not in _instances:
        mod_name, cls_name = _REGISTRY[key]
        try:
            module = importlib.import_module(mod_name)
        except ImportError as exc:
            raise AvatarProviderError(
                f"provider '{key}' failed to import: {exc}") from exc
        _instances[key] = getattr(module, cls_name)()
    return _instances[key]


__all__ = [
    "AvatarInfo",
    "AvatarProvider",
    "AvatarProviderError",
    "GenerationRequest",
    "GenerationResult",
    "VoiceInfo",
    "DEFAULT_PROVIDER",
    "default_provider",
    "gender_counts",
    "get_provider",
    "provider_names",
]
