# Ported from kaizer-platform@d5fd482 server/pipeline_core/podcast/remotion_bridge.py.
# Changes from upstream: remotion dir resolution changed from a parents[3] repo-root walk to
# env KAIZER_REMOTION_DIR (default = REPO-RELATIVE <KaizerBackend's parent>/remotion, computed
# from __file__ — never a hardcoded DEV path, so a promote renders from LIVE's own tree),
# read at call time; node invocation verbatim.
"""Remotion renderer bridge — opt-in alternative to the ffmpeg path.

This module is the Python seam onto the self-contained ``remotion/`` Node
package (repo root, NOT inside ``web/``). It:

  1. Serialises the Phase-1 plans (keep_ranges + punch_ins + captions, plus an
     optional promo block) into the EDL JSON contract that
     ``remotion/src/edl.ts`` validates.
  2. Subprocess-runs ``node remotion/render_podcast.mjs --composition ...
     --props <edl.json> --out <mp4>`` — MIRRORING the existing ffmpeg
     subprocess pattern in ``render.py`` (``subprocess.run`` + timeout + stderr
     tail capture).
  3. Fails with a *typed* exception (:class:`RemotionUnavailable` /
     :class:`RemotionRenderError`) so the caller can honestly fall back to the
     ffmpeg path and log the reason, rather than silently swallowing.

Opt-in only: nothing here runs unless the caller explicitly requests
``renderer="remotion"`` (env ``KAIZER_PODCAST_RENDERER=remotion``). The default
ffmpeg behaviour in ``render.py`` is untouched.

The EDL emitted here is the SAME contract a hand-authored EDL uses, so the
node CLI is exercised identically whether driven by Python or by hand.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger("pipeline_core.podcast.remotion_bridge")

# Locate the remotion/ Node package. Upstream walked parents[3] to the repo
# root; here it lives INSIDE the backend at <KaizerBackend>/engines/remotion
# (one folder holds everything; engines/ is gitignored + excluded from
# promote's robocopy so each tree carries its own checkout). Computed
# repo-relative from __file__ (this file is
# <KaizerBackend>/pipeline_core/podcast/…, so parents[2] is KaizerBackend).
# NEVER a hardcoded absolute DEV path: a promote to LIVE would otherwise
# keep rendering from the DEV tree. Overridable via env KAIZER_REMOTION_DIR,
# which always wins. Resolved at CALL time so env changes are honored
# without a re-import.
_DEFAULT_REMOTION_DIR = Path(__file__).resolve().parents[2] / "engines" / "remotion"


def _remotion_dir() -> Path:
    override = os.environ.get("KAIZER_REMOTION_DIR", "").strip()
    return Path(override) if override else _DEFAULT_REMOTION_DIR


def _render_cli() -> Path:
    return _remotion_dir() / "render_podcast.mjs"

# Composition ids registered in remotion/src/Root.tsx.
COMPOSITION_EDIT = "PodcastEdit"
COMPOSITION_PROMO_169 = "PodcastPromo"
COMPOSITION_PROMO_916 = "PodcastPromoVertical"


class RemotionUnavailable(RuntimeError):
    """node / the remotion package / the CLI is not available on this box.

    The caller should fall back to the ffmpeg renderer and log this.
    """


class RemotionRenderError(RuntimeError):
    """The remotion CLI ran but exited non-zero (render failed)."""


def _node_exe() -> str | None:
    return shutil.which("node")


def remotion_available() -> bool:
    """True iff node is on PATH, the CLI exists, and node_modules is installed."""
    if _node_exe() is None:
        return False
    if not _render_cli().is_file():
        return False
    if not (_remotion_dir() / "node_modules").is_dir():
        return False
    return True


def _as_range_list(ranges: Sequence[tuple[float, float]]) -> list[list[float]]:
    return [[float(s), float(e)] for s, e in ranges]


def _punch_to_dict(p: Any) -> dict:
    """Serialise a PunchIn (dataclass) or already-a-dict into EDL shape."""
    if is_dataclass(p) and not isinstance(p, type):
        d = asdict(p)
    elif isinstance(p, dict):
        d = dict(p)
    else:  # structural fallback
        d = {
            "start_s": p.start_s,
            "end_s": p.end_s,
            "zoom": p.zoom,
            "mode": p.mode,
            "trigger": getattr(p, "trigger", None),
        }
    return {
        "start_s": float(d["start_s"]),
        "end_s": float(d["end_s"]),
        "zoom": float(d["zoom"]),
        "mode": str(d["mode"]),
        "trigger": d.get("trigger"),
    }


def build_edl(
    *,
    source_path: str,
    source_width: int,
    source_height: int,
    source_fps: float,
    source_duration: float | None,
    keep_ranges: Sequence[tuple[float, float]],
    punch_ins: Sequence[Any] = (),
    captions: Sequence[dict] = (),
    promo_keep_ranges: Sequence[tuple[float, float]] = (),
    promo_captions: Sequence[dict] = (),
    promo_end_card_sec: float = 0.0,
    theme: str | None = None,
    language: str | None = None,
    name_plate: str | None = None,
    promo_end_card_text: str | None = None,
) -> dict:
    """Build the EDL dict matching ``remotion/src/edl.ts`` exactly.

    Captions keep their native ``{w, start_s, end_s}`` shape — the TS contract
    accepts ``w`` (Python-native) as well as ``word``.

    ``language`` and ``theme`` drive the Remotion design wave's per-language
    theming (``theme`` is the theme KEY in ``EdlSchema``; ``language`` is the
    raw code, threaded too so whichever field the design wave keys on is
    populated). ``name_plate`` (speaker lower-third) and ``promo_end_card_text``
    (localized CTA) populate the optional graphics slots when the data is
    cheaply available; omitted otherwise (never fabricated).
    """
    edl: dict = {
        "source": {
            "path": source_path,
            "width": int(source_width),
            "height": int(source_height),
            "fps": float(source_fps),
        },
        "keep_ranges": _as_range_list(keep_ranges),
        "punch_ins": [_punch_to_dict(p) for p in punch_ins],
        "captions": [dict(c) for c in captions],
    }
    if source_duration is not None:
        edl["source"]["duration"] = float(source_duration)
    if promo_keep_ranges:
        promo: dict = {
            "keep_ranges": _as_range_list(promo_keep_ranges),
            "captions": [dict(c) for c in promo_captions],
            "end_card_sec": float(promo_end_card_sec),
        }
        if promo_end_card_text:
            promo["end_card"] = str(promo_end_card_text)
        edl["promo"] = promo
    if theme:
        edl["theme"] = theme
    if language:
        edl["language"] = language
    if name_plate:
        edl["name_plate"] = str(name_plate)
    return edl


def _preserve_edl(edl_path: str, out_path: str) -> None:
    """Move the temp EDL next to the intended output (``<out_path>.edl.json``)
    so a failed render's props survive for debugging/replay-by-hand.
    ``shutil.move`` (not ``os.replace``) because the temp dir is usually on a
    different drive than the output dir on this box."""
    try:
        shutil.move(edl_path, f"{out_path}.edl.json")
    except OSError as exc:
        logger.warning("could not preserve EDL %s -> %s.edl.json: %s",
                       edl_path, out_path, exc)


def render_with_remotion(
    *,
    composition: str,
    edl: dict,
    out_path: str,
    timeout: int = 1800,
) -> dict:
    """Write the EDL to a temp file and run the node render CLI.

    Returns the parsed JSON summary the CLI prints on its last stdout line
    (includes ``render_ms`` etc.). Raises :class:`RemotionUnavailable` if the
    toolchain is missing, or :class:`RemotionRenderError` on non-zero exit or
    timeout. On a render failure/timeout the EDL props are KEPT at
    ``<out_path>.edl.json`` for debugging; on success the temp file is
    deleted.
    """
    if not remotion_available():
        raise RemotionUnavailable(
            f"remotion renderer unavailable: node={_node_exe()!r}, "
            f"cli_exists={_render_cli().is_file()}, "
            f"node_modules={(_remotion_dir() / 'node_modules').is_dir()}"
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Temp EDL file (deleted on success; moved to <out_path>.edl.json on
    # render failure — see _preserve_edl).
    fd, edl_path = tempfile.mkstemp(prefix="podcast_edl_", suffix=".json")
    os.close(fd)
    Path(edl_path).write_text(json.dumps(edl), encoding="utf-8")

    cmd = [
        _node_exe(),
        str(_render_cli()),
        "--composition", composition,
        "--props", edl_path,
        "--out", out_path,
    ]
    logger.info("podcast remotion render: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            cwd=str(_remotion_dir()),
        )
    except FileNotFoundError as exc:  # node vanished between check and run
        try:
            os.unlink(edl_path)  # availability failure, not a render failure
        except OSError:
            pass
        raise RemotionUnavailable(f"node not executable: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        _preserve_edl(edl_path, out_path)
        raise RemotionRenderError(
            f"remotion render timed out after {timeout}s "
            f"(composition={composition}); EDL kept at {out_path}.edl.json"
        ) from exc

    if proc.returncode != 0:
        _preserve_edl(edl_path, out_path)
        raise RemotionRenderError(
            "remotion render failed (exit %d):\n%s"
            % (proc.returncode, proc.stderr.decode(errors="replace")[-4000:])
        )

    # Success — the temp EDL is spent.
    try:
        os.unlink(edl_path)
    except OSError:
        pass

    # The CLI prints a JSON summary on the last stdout line.
    stdout = proc.stdout.decode(errors="replace").strip()
    last = stdout.splitlines()[-1] if stdout else "{}"
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        # Render succeeded but summary unparseable — return a minimal record.
        return {"ok": True, "out": out_path, "summary_raw": last[:500]}


__all__ = [
    "RemotionUnavailable",
    "RemotionRenderError",
    "remotion_available",
    "build_edl",
    "render_with_remotion",
    "COMPOSITION_EDIT",
    "COMPOSITION_PROMO_169",
    "COMPOSITION_PROMO_916",
]
