# Ported from kaizer-platform@d5fd482 server/avatar/echomimic.py.
# Changes from upstream: none (verbatim port; inherits the studio_dir()
# fallback fix from our avatar/studio.py port).
"""EchoMimicBurstProvider — vast.ai GPU burst for premium/hero clips.

Routes a generation to a rented vast.ai instance through avatar-studio's
``farm/vast.py`` helper (which handles ssh, remote venv, and pulling the
rendered file back):

    <studio-venv-python> farm/vast.py run <instance> \
        --script-file <f> --voice <v> --avatar <a> --mode video \
        --local-out <clip.mp4>

Economics (measured 2026-07-06, PLATFORM_PLAN.md §5): ~$0.31/GPU-hr;
weights download is the real cost → keep a persistent network volume
(~$3.50/mo), after which per-clip ≈ $0.10-0.30. Reserve for hero
content; the local MuseTalk farm is the daily driver.

Config: ``AVATAR_VAST_INSTANCE`` env holds the rented instance id.
Unset ⇒ provider reports unavailable (never silently spends money).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .base import (
    AvatarInfo,
    AvatarProviderError,
    GenerationRequest,
    GenerationResult,
    ProgressFn,
    VoiceInfo,
    _noop_progress,
)
from .studio import AvatarStudioProvider, _probe_duration, _studio_python, studio_dir


class EchoMimicBurstProvider:
    """Cloud burst engine. Catalog mirrors the local studio (same voices
    and avatar assets — the remote box renders with the same repo)."""

    name = "echomimic"

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or studio_dir()
        self._studio = AvatarStudioProvider(self.root)

    def _instance(self) -> str:
        return os.environ.get("AVATAR_VAST_INSTANCE", "").strip()

    def available(self) -> tuple[bool, str]:
        vast_py = self.root / "farm" / "vast.py"
        if not vast_py.exists():
            return False, f"no {vast_py} — avatar-studio farm missing"
        if not self._instance():
            return False, ("AVATAR_VAST_INSTANCE not set — rent a box "
                           "(farm/vast.py up) and export its id")
        return True, ""

    def list_avatars(self) -> list[AvatarInfo]:
        return self._studio.list_avatars()

    def list_voices(self) -> list[VoiceInfo]:
        return self._studio.list_voices()

    def generate(
        self,
        request: GenerationRequest,
        on_progress: ProgressFn = _noop_progress,
    ) -> GenerationResult:
        ready, reason = self.available()
        if not ready:
            raise AvatarProviderError(reason)
        script = (request.script or "").strip()
        if not script:
            raise AvatarProviderError("script is empty")

        out_dir = Path(request.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        video_path = out_dir / "clip_avatar.mp4"
        script_file = out_dir / "avatar_script.txt"
        script_file.write_text(script, encoding="utf-8")

        py = _studio_python(self.root)
        instance = self._instance()
        cmd = [str(py), str(self.root / "farm" / "vast.py"), "run", instance,
               "--script-file", str(script_file),
               "--voice", request.voice_id,
               "--avatar", request.avatar_id or "avatar1",
               "--mode", "video",
               "--local-out", str(video_path)]

        on_progress(10, f"vast.ai burst: instance={instance}")
        proc = subprocess.run(
            cmd, cwd=str(self.root), capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        if proc.returncode != 0:
            tail = (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-5:]
            raise AvatarProviderError(
                f"vast render failed (exit {proc.returncode}): "
                f"{' | '.join(tail)[:600]} — check farm/vast.py status")
        if not video_path.exists() or video_path.stat().st_size == 0:
            raise AvatarProviderError(
                f"vast render exited 0 but produced no file at {video_path}")

        on_progress(95, "probing clip")
        return GenerationResult(
            video_path=video_path,
            duration_s=_probe_duration(video_path),
            provider=self.name,
            engine="echomimic",
            meta={"voice_id": request.voice_id, "avatar_id": request.avatar_id,
                  "vast_instance": instance},
        )
