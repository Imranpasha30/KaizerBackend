# Ported from kaizer-platform@d5fd482 server/avatar/heygen_provider.py.
# Changes from upstream: (1) list_avatars/list_voices now check available()
# first and wrap heygen.client errors in AvatarProviderError — the raw
# HeyGenAuthError (a RuntimeError, not AvatarProviderError) used to escape
# to the router and 500 the catalog endpoints when the key/flag was
# missing. (2) list_voices populates VoiceInfo.preview_url from the API's
# ``preview_audio`` field. Otherwise verbatim port; imports OUR
# heygen.client, which is content-identical to the vendor's — not re-copied.
"""HeyGenProvider — legacy remote engine, OFF by default.

Kept for reference and as an emergency fallback, per PLATFORM_PLAN.md:
"HeyGen is dead" for the fleet (per-video API fees vs ₹≈0 local
MuseTalk), but the wrapper survives so flipping ``AVATAR_HEYGEN_ENABLED
=true`` (plus a valid ``HEYGEN_API_KEY``) restores it without touching
call sites.

Adapts the existing ``heygen.client`` submit/poll/download loop into the
blocking ``AvatarProvider.generate`` contract.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from heygen import client as heygen_client

from .base import (
    AvatarInfo,
    AvatarProviderError,
    GenerationRequest,
    GenerationResult,
    ProgressFn,
    VoiceInfo,
    _noop_progress,
)

_POLL_INTERVAL_S = 6
_RENDER_TIMEOUT_S = 15 * 60


def _enabled() -> bool:
    return os.environ.get("AVATAR_HEYGEN_ENABLED", "").strip().lower() in (
        "1", "true", "yes",
    )


class HeyGenProvider:
    """Legacy HeyGen v2 API behind the provider contract."""

    name = "heygen"

    def available(self) -> tuple[bool, str]:
        if not _enabled():
            return False, "HeyGen is disabled (set AVATAR_HEYGEN_ENABLED=true)"
        if not os.environ.get("HEYGEN_API_KEY", "").strip():
            return False, "HEYGEN_API_KEY missing"
        return True, ""

    def list_avatars(self) -> list[AvatarInfo]:
        ready, reason = self.available()
        if not ready:
            raise AvatarProviderError(reason)
        try:
            data = heygen_client.list_avatars()
        except heygen_client.HeyGenError as exc:
            raise AvatarProviderError(str(exc)) from exc
        return [
            AvatarInfo(
                id=a.get("avatar_id", ""),
                name=a.get("avatar_name", ""),
                kind="remote",
                preview_path=a.get("preview_image_url", "") or "",
                engine="heygen",
            )
            for a in data.get("avatars", [])
            if a.get("avatar_id")
        ]

    def list_voices(self) -> list[VoiceInfo]:
        ready, reason = self.available()
        if not ready:
            raise AvatarProviderError(reason)
        try:
            raw = heygen_client.list_voices()
        except heygen_client.HeyGenError as exc:
            raise AvatarProviderError(str(exc)) from exc
        return [
            VoiceInfo(
                id=v.get("voice_id", ""),
                language=(v.get("language") or "")[:32],
                name=v.get("name", ""),
                gender=(v.get("gender") or "")[:1].lower(),
                preview_url=v.get("preview_audio") or "",
            )
            for v in raw
            if v.get("voice_id")
        ]

    def generate(
        self,
        request: GenerationRequest,
        on_progress: ProgressFn = _noop_progress,
    ) -> GenerationResult:
        ready, reason = self.available()
        if not ready:
            raise AvatarProviderError(reason)

        try:
            on_progress(10, "submitting to HeyGen")
            video_id = heygen_client.generate_video(
                avatar_id=request.avatar_id,
                voice_id=request.voice_id,
                script=request.script,
                width=request.width,
                height=request.height,
            )

            deadline = time.time() + _RENDER_TIMEOUT_S
            status: dict = {}
            while time.time() < deadline:
                status = heygen_client.get_status(video_id=video_id)
                state = status.get("status")
                if state == "completed":
                    break
                if state == "failed":
                    err = status.get("error")
                    msg = err.get("message") if isinstance(err, dict) else str(err)
                    raise AvatarProviderError(f"HeyGen failed: {msg or 'unknown'}")
                elapsed = _RENDER_TIMEOUT_S - max(0.0, deadline - time.time())
                pct = min(88, 15 + int((elapsed / _RENDER_TIMEOUT_S) * 73))
                on_progress(pct, f"HeyGen status: {state}")
                time.sleep(_POLL_INTERVAL_S)
            else:
                raise AvatarProviderError("HeyGen render exceeded 15 min timeout")

            url = status.get("video_url")
            if not url:
                raise AvatarProviderError("HeyGen returned no video_url on completed")

            out_dir = Path(request.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            video_path = out_dir / "clip_avatar.mp4"
            on_progress(92, "downloading rendered video")
            heygen_client.download_to_file(url, str(video_path))
        except heygen_client.HeyGenError as exc:
            raise AvatarProviderError(str(exc)) from exc

        return GenerationResult(
            video_path=video_path,
            duration_s=float(status.get("duration") or 0.0),
            provider=self.name,
            engine="heygen",
            meta={"heygen_video_id": video_id,
                  "avatar_id": request.avatar_id,
                  "voice_id": request.voice_id},
        )
