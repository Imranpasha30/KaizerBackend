"""Content-aware dependency tracking for bulletin compose stages.

Why this exists
---------------
The bulletin pipeline runs several FFmpeg compose stages — sidebar
carousel, per-story compose, between-story takeover, final stitch.
Each is ~20-60s of GPU encode time, so re-running all of them when a
user only replaces one image is wasteful.

The previous cache check was naive::

    if os.path.exists(out) and os.path.getsize(out) > 100_000:
        skip rebuild

That misses a critical case: the *output* still exists from the last
run, but the user replaced an *input* image since then. The naive
check skips the rebuild and the user sees the OLD bulletin even
after clicking "Re-compose".

This module records, for every compose output, a sidecar
``<output>.deps.json`` that captures:

  - Every input file's (path, size, mtime_ns) fingerprint
  - Any non-file params (text strings, durations, colors) the stage
    consumed, as ``extra={...}``

On the next run, ``is_fresh(out, inputs, extra)`` returns True only
when every input fingerprint matches and ``extra`` is byte-equal.
Otherwise the output is considered stale and the caller rebuilds.

Design notes
------------
- File fingerprint is (size, mtime_ns), not a content hash. This
  matches GNU make's behavior — fast, no read I/O, and robust as
  long as the writer (``write_bytes``, ``im.save``, ``ffmpeg -y``)
  updates mtime. Both Pillow ``save`` and ``write_bytes`` do.
- The sidecar lives next to the output, so removing the output also
  removes the deps file's reference (it stays as an orphan, no harm).
- The deps file is small (~1 KB), JSON, human-readable for
  debugging.
- Missing deps file → output is treated as stale (forces rebuild).
  That makes the first run after deploying this module act exactly
  like a full rebuild, which is the safe default.

Used by
-------
``pipeline_core/pipeline.py`` wraps each of these stages:

  - ticker.png         (inputs: headlines, font, lang)
  - _bug.png           (inputs: channel name, logo file)
  - _sidebar_NN.mp4    (inputs: imgs[:5], story_dur)
  - composed_story_NN.mp4 (inputs: raw_path, sidebar, ticker, bug, font, meta)
  - takeover_NN.mp4    (inputs: imgs[:4], takeover_dur)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional


_DEPS_SUFFIX = ".deps.json"
_SCHEMA_VERSION = 1


def _fingerprint_file(path: str | os.PathLike) -> Optional[dict]:
    """Return ``{"size": N, "mtime_ns": M}`` for ``path``, or None
    if the file is missing. Cheap — one ``stat()`` call."""
    try:
        st = os.stat(path)
    except (OSError, TypeError):
        return None
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _fingerprint_inputs(inputs: Iterable[str | os.PathLike]) -> dict[str, dict]:
    """Map each input path → its fingerprint. Inputs that don't exist
    are recorded as ``None`` — the freshness check treats a missing
    input the same as a changed one (forces rebuild)."""
    out: dict[str, dict] = {}
    for p in inputs:
        if not p:
            continue
        key = os.path.abspath(str(p))
        out[key] = _fingerprint_file(key)
    return out


def _deps_path_for(output: str | os.PathLike) -> Path:
    return Path(str(output) + _DEPS_SUFFIX)


def is_fresh(
    output: str | os.PathLike,
    inputs: Iterable[str | os.PathLike],
    extra: Optional[dict[str, Any]] = None,
    *,
    min_size: int = 100_000,
) -> bool:
    """Return True iff ``output`` exists, has size ≥ ``min_size``, AND
    every input fingerprint matches the recorded deps. False otherwise,
    which the caller should treat as "rebuild required".

    Parameters
    ----------
    output : path to the file we previously built
    inputs : iterable of paths that fed into the build
    extra  : non-file params (text, ints, dict). Stored as-is; must
             be JSON-serializable. Compared by JSON round-trip so
             dict key ordering doesn't matter.
    min_size : sanity floor — sub-100KB outputs are treated as stale
               (catches half-written / failed previous runs).
    """
    out_path = str(output)
    if not os.path.exists(out_path):
        return False
    try:
        if os.path.getsize(out_path) < min_size:
            return False
    except OSError:
        return False

    deps_path = _deps_path_for(output)
    if not deps_path.is_file():
        return False

    try:
        recorded = json.loads(deps_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    # Schema-mismatch deps → treat as stale (next write upgrades it).
    if recorded.get("schema") != _SCHEMA_VERSION:
        return False

    # Compare extras (non-file params). JSON round-trip canonicalises
    # dict key order so {"a":1,"b":2} == {"b":2,"a":1}.
    expected_extra = json.loads(json.dumps(extra or {}, sort_keys=True))
    saved_extra = recorded.get("extra") or {}
    saved_extra = json.loads(json.dumps(saved_extra, sort_keys=True))
    if expected_extra != saved_extra:
        return False

    # Compare per-input fingerprints. Any missing/changed input → stale.
    saved_inputs: dict[str, Optional[dict]] = recorded.get("inputs") or {}
    current = _fingerprint_inputs(inputs)
    if set(saved_inputs.keys()) != set(current.keys()):
        return False
    for key, cur_fp in current.items():
        saved_fp = saved_inputs.get(key)
        if cur_fp is None or saved_fp is None:
            return False
        if cur_fp.get("size") != saved_fp.get("size"):
            return False
        if cur_fp.get("mtime_ns") != saved_fp.get("mtime_ns"):
            return False

    return True


def mark_built(
    output: str | os.PathLike,
    inputs: Iterable[str | os.PathLike],
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Write the sidecar deps file for ``output``. Call AFTER the
    output has been written successfully — if writing the deps fails
    we log but don't raise (the only consequence is a free rebuild
    next time)."""
    deps_path = _deps_path_for(output)
    payload = {
        "schema":  _SCHEMA_VERSION,
        "output":  os.path.abspath(str(output)),
        "inputs":  _fingerprint_inputs(inputs),
        "extra":   extra or {},
    }
    try:
        deps_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"[compose_deps] warn: could not write {deps_path}: {exc}")


def invalidate(output: str | os.PathLike) -> None:
    """Remove both the output and its deps sidecar. Used by scoped
    recompose (``scope="text-only"`` etc.) to force a rebuild without
    touching other stages."""
    out_path = Path(str(output))
    try:
        out_path.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        _deps_path_for(output).unlink(missing_ok=True)
    except OSError:
        pass


def invalidate_glob(directory: str | os.PathLike, pattern: str) -> int:
    """Invalidate every output matching ``directory/pattern``. Returns
    the count removed. ``pattern`` is a glob like ``composed_story_*.mp4``.
    Used by scope-based recompose to wipe a class of intermediates."""
    d = Path(str(directory))
    if not d.is_dir():
        return 0
    n = 0
    for p in d.glob(pattern):
        invalidate(p)
        n += 1
    return n
