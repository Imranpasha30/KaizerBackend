"""Shared Gemini-response cache backed by Redis.

Why Redis (not the existing on-disk ``OUTPUT_ROOT/_gemini_cache``)?
  - **Cross-pod**: every worker shares one cache, so a hit by pod A
    benefits pod B. The disk cache only ever helps the same machine.
  - **TTL eviction**: stale entries (model retired, prompt rewritten)
    age out without manual cleanup of an ever-growing folder.
  - **Atomic stat counters**: each tier of cache (video / translation /
    seo / thumbnails) has its own hits + misses + bytes-saved counter
    so the admin dashboard can show $ saved per tier per day.

Design:
  - One Redis hash KEY per cached entry: ``kaizer:cache:gemini:{kind}:{hash}``
    — value is the JSON response payload (UTF-8 string).
  - Counters at ``kaizer:cache:gemini:{kind}:stats`` hash holding
    ``{hits, misses, bytes_in, bytes_out, last_hit_ts, last_miss_ts}``.
  - All entries TTL'd. Default 7 days for video analysis (heavy +
    rarely re-uploaded); 30 days for translation / SEO / thumbnails
    (deterministic + much smaller).

Failure model:
  - Redis unreachable → `cache_get` returns None, `cache_set` swallows
    the error. Caller falls through to a normal Gemini call. Caching
    is a perf optimisation, never a correctness path.

The on-disk cache in ``pipeline_core/pipeline.py:_cache_dir`` stays as
a SECOND tier — Redis-miss but disk-hit still avoids the Gemini call,
and on a worker reboot we don't pay for cold-cache traffic until the
disk warms back into Redis. ``cache_get`` lifts disk hits into Redis
automatically when ``warm_from_disk`` is passed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger("kaizer.gemini_cache")


# ── Cache "kind" namespaces ─────────────────────────────────────
# Each kind has its own TTL + stat counter. Adding a new kind is
# just appending to this dict — callers reference by the string key.
KINDS = {
    "video":       {"ttl_s": 7 * 86400,   "purpose": "Video → cut-list analysis"},
    "translation": {"ttl_s": 30 * 86400,  "purpose": "SEO JSON → other-language SEO JSON"},
    "seo":         {"ttl_s": 30 * 86400,  "purpose": "Cut + transcript → SEO JSON"},
    "thumbnail":   {"ttl_s": 30 * 86400,  "purpose": "Thumbnail concept generation"},
    "trending":    {"ttl_s": 6 * 3600,    "purpose": "Trending topic radar"},
}

KEY_PREFIX = "kaizer:cache:gemini"


def _redis():
    """Lazy import — avoids a hard dep on the queue module at import time."""
    from redis_queue import get_client, is_enabled
    if not is_enabled():
        return None
    return get_client()


def _kind_key(kind: str, h: str) -> str:
    return f"{KEY_PREFIX}:{kind}:{h}"


def _stats_key(kind: str) -> str:
    return f"{KEY_PREFIX}:{kind}:stats"


def make_key(*parts: Any) -> str:
    """Build a stable cache hash from heterogeneous inputs.

    Each part is JSON-stringified (sort_keys=True for dicts) before
    hashing so reordering kwargs doesn't bust the cache. Bytes parts
    are hashed directly. ``None`` parts are skipped.
    """
    h = hashlib.sha256()
    for p in parts:
        if p is None:
            continue
        if isinstance(p, bytes):
            h.update(p)
        elif isinstance(p, str):
            h.update(p.encode("utf-8"))
        else:
            try:
                h.update(json.dumps(p, sort_keys=True, default=str).encode("utf-8"))
            except Exception:
                h.update(str(p).encode("utf-8"))
        h.update(b"\x1f")   # field separator so part boundaries matter
    return h.hexdigest()[:32]


def hash_file_prefix(path: str, *, prefix_bytes: int = 4 * 1024 * 1024) -> str:
    """Content hash from the first ``prefix_bytes`` of a file + size.

    4 MiB is enough to distinguish any two real videos and keeps the
    hash cheap on multi-hundred-MB sources.

    ``mtime`` is INTENTIONALLY excluded — re-uploading the same video
    creates a new file with a fresh mtime, which busts the cache even
    though the bytes are identical. The whole point of this cache is
    to skip Gemini when the user uploads the same source twice.
    """
    h = hashlib.sha256()
    try:
        st = os.stat(path)
        h.update(f"{st.st_size}".encode())
        with open(path, "rb") as f:
            h.update(f.read(prefix_bytes))
    except Exception:
        h.update(path.encode("utf-8"))
    return h.hexdigest()[:32]


def cache_get(kind: str, key_hash: str) -> Optional[dict]:
    """Return the cached response dict for (kind, key_hash) or None.

    Increments hits/misses on the stats hash so the admin dashboard
    can chart them. A Redis outage returns None silently — the caller
    will treat it as a miss and call Gemini.
    """
    cli = _redis()
    if cli is None:
        return None
    k = _kind_key(kind, key_hash)
    try:
        raw = cli.get(k)
    except Exception as exc:
        logger.warning("cache_get redis err: %s", exc)
        return None
    s = _stats_key(kind)
    try:
        if raw is None:
            cli.hincrby(s, "misses", 1)
            cli.hset(s, "last_miss_ts", int(time.time()))
            return None
        cli.hincrby(s, "hits", 1)
        cli.hincrby(s, "bytes_out", len(raw))
        cli.hset(s, "last_hit_ts", int(time.time()))
    except Exception:
        # Don't let stats failures hide the hit.
        pass
    try:
        return json.loads(raw)
    except Exception as exc:
        logger.warning("cache_get JSON decode failed (kind=%s key=%s): %s",
                       kind, key_hash, exc)
        return None


def cache_set(kind: str, key_hash: str, value: dict, *, ttl_s: Optional[int] = None) -> None:
    """Store a Gemini response dict under (kind, key_hash) with TTL.

    Best-effort: if Redis is unreachable or the value isn't JSON-
    serialisable, we log and return — caller already has the value
    in hand from the live API call.
    """
    cli = _redis()
    if cli is None:
        return
    if ttl_s is None:
        ttl_s = int(KINDS.get(kind, {}).get("ttl_s") or (7 * 86400))
    try:
        payload = json.dumps(value, default=str)
    except Exception as exc:
        logger.warning("cache_set serialise failed (kind=%s): %s", kind, exc)
        return
    k = _kind_key(kind, key_hash)
    s = _stats_key(kind)
    try:
        cli.setex(k, ttl_s, payload)
        cli.hincrby(s, "bytes_in", len(payload))
        cli.hincrby(s, "writes", 1)
    except Exception as exc:
        logger.warning("cache_set redis err (kind=%s): %s", kind, exc)


def stats(kind: Optional[str] = None) -> dict:
    """Stats snapshot for a single kind or all kinds.

    Shape::

        {
          "video":       {"hits": 12, "misses": 3, "writes": 3, ...},
          "translation": {...},
          "_total":      {"hits": ..., "misses": ..., "hit_rate": 0.86},
        }

    All counters are 0 when Redis is unreachable so the admin UI
    doesn't crash.
    """
    cli = _redis()
    out: dict = {}
    total_hits = 0
    total_misses = 0
    targets = [kind] if kind else list(KINDS.keys())
    for k in targets:
        row: dict = {"hits": 0, "misses": 0, "writes": 0,
                     "bytes_in": 0, "bytes_out": 0,
                     "last_hit_ts": 0, "last_miss_ts": 0,
                     "ttl_s": KINDS.get(k, {}).get("ttl_s")}
        if cli is not None:
            try:
                raw = cli.hgetall(_stats_key(k))
                for fld, val in (raw or {}).items():
                    try:
                        row[fld] = int(val)
                    except (TypeError, ValueError):
                        row[fld] = val
            except Exception:
                pass
        total_hits += int(row.get("hits") or 0)
        total_misses += int(row.get("misses") or 0)
        out[k] = row

    seen = total_hits + total_misses
    out["_total"] = {
        "hits":     total_hits,
        "misses":   total_misses,
        "hit_rate": (total_hits / seen) if seen else 0.0,
    }
    return out


def reset(kind: Optional[str] = None) -> int:
    """Clear cache entries (and stats) for a kind, or for ALL kinds.

    Returns the number of cache keys deleted. Admin-only — exposed at
    ``DELETE /api/admin/cache/gemini[?kind=video]``.
    """
    cli = _redis()
    if cli is None:
        return 0
    n = 0
    targets = [kind] if kind else list(KINDS.keys())
    for k in targets:
        # SCAN+DEL — safer than KEYS on a busy Redis. Pattern includes
        # the stats key so we wipe counters too.
        for pattern in (f"{_kind_key(k, '*')}", _stats_key(k)):
            cursor = 0
            while True:
                cursor, batch = cli.scan(cursor=cursor, match=pattern, count=500)
                if batch:
                    n += cli.delete(*batch)
                if cursor == 0:
                    break
    return n
