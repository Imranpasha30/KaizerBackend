# Ported from kaizer-platform@d5fd482 server/avatar/base.py.
# Changes from upstream: VoiceInfo grew an optional ``preview_url`` field
# (appended LAST so positional construction is unaffected) — remote
# providers (HeyGen) publish a hosted preview clip instead of a local
# ref.wav; otherwise verbatim port; stdlib-only.
"""Avatar provider interface — the seam that replaced the hard HeyGen wiring.

Every avatar engine (local Avatar Studio / MuseTalk, vast.ai EchoMimic
burst, legacy HeyGen) implements the same small surface so routers and
pipeline code never import an engine directly:

    provider = avatar.get_provider()          # env-selected
    result   = provider.generate(request, on_progress=cb)   # blocking

``generate`` is intentionally BLOCKING — every caller in this codebase
already runs generations on a background daemon thread (see
routers/heygen.py's pattern), so an async submit/poll split would only
duplicate state machines. Providers that are remote (HeyGen) hide their
own submit/poll/download loop behind the blocking call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol, runtime_checkable


class AvatarProviderError(RuntimeError):
    """Generation failed, provider misconfigured, or engine unavailable."""


@dataclass(frozen=True)
class AvatarInfo:
    """One selectable presenter."""
    id: str                      # e.g. "avatar1", "avatar2", heygen avatar_id
    name: str
    kind: str                    # "video_loop" | "photo" | "remote"
    preview_path: str = ""       # local file or URL for the picker
    engine: str = ""             # e.g. "musetalk", "heygen"


@dataclass(frozen=True)
class VoiceInfo:
    """One cloned/synthetic voice."""
    id: str                      # e.g. "te_f_studio", heygen voice_id
    language: str                # ISO 639-1 where known ("te", "hi", ...)
    name: str = ""
    gender: str = ""             # "f" | "m" | ""
    preview_url: str = ""        # hosted sample clip (remote providers)


@dataclass(frozen=True)
class GenerationRequest:
    script: str
    avatar_id: str
    voice_id: str
    out_dir: Path                # provider writes clip (+ scratch) here
    width: int = 1080
    height: int = 1920
    language: str = "te"
    engine: str = ""             # provider-specific override ("musetalk", ...)
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    video_path: Path
    duration_s: float
    provider: str                # provider name that produced it
    engine: str = ""             # actual engine used
    thumbnail_path: Optional[Path] = None
    meta: dict = field(default_factory=dict)


def gender_counts(voices: list[VoiceInfo]) -> dict[str, int]:
    """{"f": n, "m": n, "": n} for a voice list.

    "" counts entries with no gender tag (legacy ``anchor_*`` ids that
    predate the <lang>_<gender>_<variant> naming) — surfaced rather than
    folded into f/m so a skewed pack ratio can't hide behind untagged
    voices.
    """
    counts = {"f": 0, "m": 0, "": 0}
    for v in voices:
        counts[v.gender if v.gender in ("f", "m") else ""] += 1
    return counts


# Progress callback: (percent 0-100, human message)
ProgressFn = Callable[[int, str], None]


def _noop_progress(_pct: int, _msg: str) -> None:
    return None


@runtime_checkable
class AvatarProvider(Protocol):
    """The contract every avatar engine implements."""

    name: str

    def available(self) -> tuple[bool, str]:
        """(ready, reason). ``reason`` explains *why not* when not ready —
        surfaced verbatim in the UI so misconfiguration is self-diagnosing."""
        ...

    def list_avatars(self) -> list[AvatarInfo]:
        ...

    def list_voices(self) -> list[VoiceInfo]:
        ...

    def generate(
        self,
        request: GenerationRequest,
        on_progress: ProgressFn = _noop_progress,
    ) -> GenerationResult:
        """Produce a talking-head clip. Blocking; raises AvatarProviderError."""
        ...
