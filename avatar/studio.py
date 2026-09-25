# Ported from kaizer-platform@d5fd482 server/avatar/studio.py.
# Changes from upstream: studio_dir()'s sibling fallback is now
# here.parents[1].parent / "avatar-studio" (upstream used parents[2].parent,
# which assumed the deeper <repo>/server/avatar/ tree; from
# KaizerBackend/avatar/studio.py this resolves to e:\kaizer-dev\avatar-studio,
# the sibling of the KaizerBackend repo root). $AVATAR_STUDIO_DIR still wins,
# ~/avatar-studio is still the last candidate.
"""AvatarStudioProvider — local MuseTalk farm (the default engine).

Shells the sibling avatar-studio repo's ``generate.py`` inside ITS OWN
venv (IndicF5 + MuseTalk live there; their dependency tree must never
mix with this backend's). Same invocation contract as bulletin-gen's
``pipeline/anchor.py`` bridge, which is the proven production path:

    <studio>/venv/Scripts/python -u generate.py --mode video \
        --voice <voice> --text-file <script.txt> --out <clip.mp4> \
        [--photo assets/<avatarN>.png] [--engine musetalk]

avatar1 = the default video-loop anchor (no --photo); every other
avatar id animates ``assets/<id>.png`` (base-loop + MuseTalk redub).

Marginal cost ≈ ₹0; a clip takes ~3-4 min on the RTX 4060 box.
"""
from __future__ import annotations

import os
import shutil
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

_LANG_NAMES = {
    "as": "Assamese", "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "or": "Odia",
    "pa": "Punjabi", "ta": "Tamil", "te": "Telugu", "en": "English",
}


def studio_dir() -> Path:
    """Resolve the avatar-studio checkout.

    $AVATAR_STUDIO_DIR wins; otherwise the canonical home INSIDE the
    backend (<KaizerBackend>/engines/avatar-studio — one folder holds
    everything; gitignored + excluded from promote's robocopy), then the
    legacy repo-sibling, then ~/avatar-studio.
    """
    env = os.environ.get("AVATAR_STUDIO_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "engines" / "avatar-studio",
        here.parents[1].parent / "avatar-studio",   # legacy repo-sibling
        Path.home() / "avatar-studio",
    ]
    for c in candidates:
        if (c / "generate.py").exists():
            return c
    return candidates[0]


def _studio_python(root: Path) -> Path:
    win = root / "venv" / "Scripts" / "python.exe"
    posix = root / "venv" / "bin" / "python"
    return win if win.exists() else posix


def _ffprobe_bin() -> str:
    return os.environ.get("KAIZER_FFPROBE", "").strip() or shutil.which("ffprobe") or ""


def _probe_duration(path: Path) -> float:
    """Clip duration via ffprobe; 0.0 when ffprobe is unavailable."""
    probe = _ffprobe_bin()
    if not probe:
        return 0.0
    try:
        out = subprocess.run(
            [probe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=30, check=True,
        )
        return float(out.stdout.strip() or 0.0)
    except (subprocess.SubprocessError, ValueError, OSError):
        return 0.0


class AvatarStudioProvider:
    """Local avatar-studio (MuseTalk) engine behind the provider contract."""

    name = "avatar_studio"

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or studio_dir()

    # ── availability ────────────────────────────────────────────────
    def available(self) -> tuple[bool, str]:
        if not (self.root / "generate.py").exists():
            return False, (f"avatar-studio not found at {self.root} — "
                           f"set AVATAR_STUDIO_DIR")
        py = _studio_python(self.root)
        if not py.exists():
            return False, f"avatar-studio venv missing: {py}"
        return True, ""

    # ── catalog ─────────────────────────────────────────────────────
    def list_avatars(self) -> list[AvatarInfo]:
        assets = self.root / "assets"
        avatars = [AvatarInfo(
            id="avatar1", name="Studio Anchor (video)", kind="video_loop",
            preview_path=str(assets / "avatar.mp4"), engine="musetalk",
        )]
        if assets.is_dir():
            for png in sorted(assets.glob("avatar*.png")):
                stem = png.stem
                if stem == "avatar1" or stem.endswith("_crop"):
                    continue
                avatars.append(AvatarInfo(
                    id=stem, name=f"Photo Anchor {stem.removeprefix('avatar')}",
                    kind="photo", preview_path=str(png), engine="musetalk",
                ))
        return avatars

    def list_voices(self) -> list[VoiceInfo]:
        voices_dir = self.root / "voices"
        if not voices_dir.is_dir():
            return []
        out: list[VoiceInfo] = []
        for d in sorted(voices_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            vid = d.name
            # Naming: <lang>_<gender>_<variant> (te_f_studio) or the
            # legacy anchor_* voices, which are Telugu anchors.
            parts = vid.split("_")
            lang = parts[0] if parts[0] in _LANG_NAMES else "te"
            gender = parts[1] if len(parts) > 1 and parts[1] in ("f", "m") else ""
            out.append(VoiceInfo(
                id=vid, language=lang,
                name=f"{_LANG_NAMES.get(lang, lang)} {vid}", gender=gender,
            ))
        return out

    def voice_sample_path(self, voice_id: str) -> Path | None:
        """Reference clip for the voice's clone source, if one exists.

        Every voice dir has ``ref.wav`` (the recording MuseTalk's TTS
        clones from) — real audio, not a generated sample, but a true
        representative of what the voice sounds like. Resolve-and-check
        against ``voices_dir`` since ``voice_id`` is caller-controlled
        (came off a URL path param) and must not escape via ``../``.
        """
        voices_dir = (self.root / "voices").resolve()
        candidate = (voices_dir / voice_id / "ref.wav").resolve()
        if voices_dir not in candidate.parents:
            return None
        return candidate if candidate.is_file() else None

    # ── generation ──────────────────────────────────────────────────
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

        # Script goes through a UTF-8 file — Telugu/Devanagari text on a
        # Windows command line is a mojibake minefield.
        script_file = out_dir / "avatar_script.txt"
        script_file.write_text(script, encoding="utf-8")

        py = _studio_python(self.root)
        cmd = [str(py), "-u", "generate.py", "--mode", "video",
               "--voice", request.voice_id,
               "--text-file", str(script_file),
               "--out", str(video_path)]
        if request.avatar_id and request.avatar_id != "avatar1":
            photo = self.root / "assets" / f"{request.avatar_id}.png"
            if not photo.exists():
                raise AvatarProviderError(
                    f"unknown avatar '{request.avatar_id}': no {photo}")
            cmd += ["--photo", str(photo)]
        engine = request.engine or os.environ.get("AVATAR_STUDIO_ENGINE", "").strip()
        if engine:
            cmd += ["--engine", engine]

        on_progress(5, f"avatar-studio: voice={request.voice_id} "
                       f"avatar={request.avatar_id}")
        # PYTHONUTF8 propagates to generate.py AND its MuseTalk child —
        # MuseTalk's inference prints CJK brackets, which explode with
        # UnicodeEncodeError on Windows' default cp1252 when piped.
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        # cwd = studio root so generate.py's own relative subprocess calls
        # (MuseTalk inference, ffmpeg composite) resolve as they do there.
        proc = subprocess.Popen(
            cmd, cwd=str(self.root), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1, env=env,
        )
        lines: list[str] = []
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip("\n")
            if not line:
                continue
            lines.append(line)
            # Coarse progress from the engine's known phase markers.
            low = line.lower()
            if "voice" in low or "tts" in low:
                on_progress(25, line[:160])
            elif "musetalk" in low or "lip" in low:
                on_progress(60, line[:160])
            elif "->" in line:
                on_progress(85, line[:160])
        proc.stdout.close()
        code = proc.wait()
        if code != 0:
            tail = " | ".join(lines[-5:])
            raise AvatarProviderError(
                f"generate.py exited {code}: {tail[:600]}")
        if not video_path.exists() or video_path.stat().st_size == 0:
            raise AvatarProviderError(
                f"generate.py exited 0 but produced no file at {video_path}")

        on_progress(95, "probing clip")
        return GenerationResult(
            video_path=video_path,
            duration_s=_probe_duration(video_path),
            provider=self.name,
            engine=engine or "auto(musetalk)",
            meta={"voice_id": request.voice_id, "avatar_id": request.avatar_id,
                  "studio_dir": str(self.root)},
        )
