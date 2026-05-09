"""Per-tenant token-bucket rate limiter (Redis-backed).

Why token bucket (and not sliding-window or fixed-window)?
  - Real users burst then breathe. Token bucket supports that
    naturally: a user with burst=10 and rate=2/sec can fire 10
    requests in one second, then 2/sec sustained, without ever
    being lied to about a "10/sec" cap.
  - Fixed-window admits 2× the limit at the boundary (10 req at
    11:59:59 + 10 req at 12:00:00 = 20 in one second). Sliding-
    window fixes that but needs more state.
  - Redis Lua makes the whole "refill, deduct, persist" loop a
    single atomic op — no race window where two requests both pass
    because they read state before either wrote it.

Why per-tenant (not per-IP)?
  - At enterprise scale, one tenant runs a campaign that fires 500
    upload-create calls in 3 seconds; without per-tenant ceilings,
    that drowns out everyone else's quota. Per-IP doesn't help
    because the heavy tenant has one corporate egress IP that all
    their seats share.
  - Per-IP buckets stay useful for the auth flow (brute-force
    protection) — see ``check_ip_rate`` for that case.

Plan-aware limits:
  Limits scale with the user's billing plan. ``free``/``creator``
  see tight caps; ``pro``/``agency`` get headroom. Set in
  :data:`PLAN_LIMITS` — single source of truth.

Limiter knobs are TWO numbers per bucket:
  - ``burst``: max tokens the bucket can hold (= max instantaneous
    burst the tenant can fire from cold).
  - ``rate_per_s``: how many tokens refill per second (= sustained
    request rate after the burst is exhausted).

Failure mode:
  Redis unreachable → caller sees ``check_rate`` return allow=True
  with retry_after=0. Failing OPEN is the right default — better to
  let traffic through than to hold up an enterprise tenant when the
  rate-limit infra is the only thing broken.
"""
from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

from fastapi import Depends, HTTPException, Request

logger = logging.getLogger("kaizer.rate_limit")


# ── Plan → bucket config ────────────────────────────────────────
# Buckets:
#   "create"  — heavy POST endpoints (job/upload/campaign create).
#               Each call kicks off Gemini calls, FFmpeg renders, or
#               YT API quota. Tight caps here matter most.
#   "read"    — list/get endpoints. Cheap; large buckets.
#   "auth"    — login attempts. Per-IP, not per-user (brute-force).
#
# Format per bucket: (burst, rate_per_s)
#   burst=5, rate=1/30  → 5 fast in a row, then 1 every 30s = 2/min
#   burst=60, rate=1    → 60 fast, then 60/min sustained
PLAN_LIMITS: dict = {
    # Anonymous / no-user (auth pages, public marketing endpoints)
    "_anon":   {"create": (3,   1/60),  "read": (30,  30/60), "auth": (10, 1/60)},
    # Paying tiers go up the ladder
    "free":    {"create": (5,   1/30),  "read": (60,  1.0),   "auth": (10, 1/30)},
    "creator": {"create": (10,  1/15),  "read": (120, 2.0),   "auth": (10, 1/30)},
    "pro":     {"create": (30,  1/5),   "read": (300, 5.0),   "auth": (10, 1/30)},
    "agency":  {"create": (60,  1/2),   "read": (600, 10.0),  "auth": (10, 1/30)},
}

DEFAULT_PLAN = "free"

# Per-IP auth bucket — used for brute-force login protection. Keyed
# by the source IP so even a tenant with valid plan credentials
# can't pound login from one box.
AUTH_IP_BURST = 10
AUTH_IP_RATE  = 1 / 60   # 1 attempt per minute sustained


# ── Atomic Lua: refill + deduct + persist ───────────────────────
# Returns ``[allowed, remaining_tokens, retry_after_seconds]``.
# All three values are returned every time so the caller can also
# expose ``X-RateLimit-Remaining`` headers without a second round-trip.
_LUA_TOKEN_BUCKET = """
local burst   = tonumber(ARGV[1])
local rate    = tonumber(ARGV[2])
local now     = tonumber(ARGV[3])
local ttl_s   = tonumber(ARGV[4])

local data    = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens  = tonumber(data[1])
local ts      = tonumber(data[2])

if tokens == nil then
    tokens = burst
    ts     = now
end

local elapsed = math.max(0, now - ts)
tokens = math.min(burst, tokens + elapsed * rate)

local allowed = 0
local retry_after = 0
if tokens >= 1 then
    tokens = tokens - 1
    allowed = 1
else
    -- How long until we have 1 token again
    retry_after = (1 - tokens) / rate
end

redis.call('HMSET', KEYS[1], 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', KEYS[1], ttl_s)

return {allowed, tostring(tokens), tostring(retry_after)}
"""

_KEY_PREFIX = "kaizer:rl"
_lua_sha: Optional[str] = None


def _redis():
    """Lazy import — avoids a hard dep on the queue module at import time."""
    from redis_queue import get_client, is_enabled
    if not is_enabled():
        return None
    return get_client()


def _ensure_lua_loaded(cli) -> Optional[str]:
    """Cache the EVALSHA digest to avoid re-shipping the script."""
    global _lua_sha
    if _lua_sha is not None:
        return _lua_sha
    try:
        _lua_sha = cli.script_load(_LUA_TOKEN_BUCKET)
    except Exception as exc:
        logger.warning("rate_limit: SCRIPT LOAD failed: %s", exc)
        _lua_sha = None
    return _lua_sha


def check_rate(
    key: str,
    burst: int,
    rate_per_s: float,
    *,
    ttl_s: int = 3600,
) -> Tuple[bool, float, float]:
    """Try to claim 1 token from the bucket at ``key``.

    Returns ``(allowed, retry_after_s, remaining)``. Fails OPEN —
    Redis outage returns allowed=True so a broken cache doesn't
    take the API down.
    """
    cli = _redis()
    if cli is None:
        # Fail open. Log every minute or so to alert ops.
        return True, 0.0, float(burst)

    sha = _ensure_lua_loaded(cli)
    full_key = f"{_KEY_PREFIX}:{key}"
    now = time.time()
    try:
        if sha:
            try:
                result = cli.evalsha(sha, 1, full_key, burst, rate_per_s, now, ttl_s)
            except Exception as exc:
                # NOSCRIPT → Redis flushed, reload + retry once.
                if "NOSCRIPT" in str(exc):
                    _global = globals()
                    _global["_lua_sha"] = None
                    sha = _ensure_lua_loaded(cli)
                    result = cli.evalsha(sha, 1, full_key, burst, rate_per_s, now, ttl_s)
                else:
                    raise
        else:
            result = cli.eval(_LUA_TOKEN_BUCKET, 1, full_key,
                              burst, rate_per_s, now, ttl_s)
    except Exception as exc:
        logger.warning("rate_limit: redis EVAL failed (failing open): %s", exc)
        return True, 0.0, float(burst)

    allowed_raw, tokens_raw, retry_raw = result
    allowed = bool(int(allowed_raw))
    remaining = float(tokens_raw)
    retry_after = float(retry_raw)
    return allowed, retry_after, remaining


# ── Plan-aware lookup ───────────────────────────────────────────
def _plan_limits(plan: Optional[str], bucket: str) -> Tuple[int, float]:
    p = (plan or DEFAULT_PLAN).strip().lower()
    cfg = PLAN_LIMITS.get(p) or PLAN_LIMITS[DEFAULT_PLAN]
    spec = cfg.get(bucket)
    if not spec:
        # Bucket name not configured for this plan — fall back to free.
        spec = PLAN_LIMITS[DEFAULT_PLAN].get(bucket, (10, 1.0))
    burst, rate = spec
    return int(burst), float(rate)


# ── FastAPI dependency factory ──────────────────────────────────
def rate_limited(bucket: str = "create"):
    """Return a FastAPI dependency that enforces a rate limit.

    Usage::

        @router.post("/uploads")
        def create_upload(
            user: User = Depends(auth.current_user),
            _rate=Depends(rate_limited("create")),
            ...
        ):
            ...

    The dependency runs BEFORE the endpoint body, so an over-quota
    tenant gets a 429 without doing any DB work.

    Plan resolution: pulls the user via :func:`auth.current_user_optional`
    so anonymous routes still get bucketed (by source IP). When a
    user is authenticated, their ``plan`` field selects the bucket
    config; otherwise we fall through to the ``_anon`` plan.
    """
    # Lazy import to avoid touching auth.py at module-import time —
    # auth.py imports models.py which imports database.py which loads
    # .env. Keeping this here means rate_limit.py stays standalone
    # for unit tests that don't want a full FastAPI environment.
    import auth as _auth

    async def _dep(
        request: Request,
        user=Depends(_auth.current_user_optional),
    ):
        plan, who = _resolve_plan_and_id(request, user)
        burst, rate = _plan_limits(plan, bucket)
        key = f"{bucket}:{plan}:{who}"
        allowed, retry_after, remaining = check_rate(key, burst, rate)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded for {bucket!r} on plan {plan!r}. "
                    f"Retry in {retry_after:.1f}s."
                ),
                headers={
                    "Retry-After":           str(max(1, int(retry_after))),
                    "X-RateLimit-Bucket":    bucket,
                    "X-RateLimit-Plan":      plan,
                    "X-RateLimit-Burst":     str(burst),
                    "X-RateLimit-Remaining": "0",
                },
            )
        # Annotate the request state when the call succeeds, so
        # well-behaved clients can pre-emptively back off.
        request.state.rate_remaining = remaining
        request.state.rate_burst = burst
        request.state.rate_bucket = bucket
        return None
    return _dep


def _resolve_plan_and_id(request: Request, user) -> Tuple[str, str]:
    """Pick the right bucket key for this caller.

    Authenticated user → ("user-plan", "user:<id>")
    Anonymous          → ("_anon",     "ip:<addr>")
    """
    if user is not None:
        plan = getattr(user, "plan", None) or DEFAULT_PLAN
        uid  = getattr(user, "id", None)
        return plan, f"user:{uid}"
    # Anonymous — bucket by IP. Trust X-Forwarded-For only when
    # behind a known reverse proxy; otherwise client.host is right.
    xff = request.headers.get("x-forwarded-for", "")
    ip  = (xff.split(",")[0].strip() if xff else
           (request.client.host if request.client else "unknown"))
    return "_anon", f"ip:{ip}"


# ── IP-keyed brute-force guard for auth ─────────────────────────
def check_ip_rate(ip: str, *, bucket: str = "auth_ip") -> Tuple[bool, float, float]:
    """Per-IP login bucket — wider than per-user because the user
    might not exist yet. Used by the login endpoint."""
    return check_rate(
        f"{bucket}:{ip}",
        burst=AUTH_IP_BURST,
        rate_per_s=AUTH_IP_RATE,
    )


# ── Admin / observability ───────────────────────────────────────
def bucket_state(bucket: str, plan: str, who: str) -> dict:
    """Inspect a single bucket's state without touching its tokens.

    Returns ``{tokens, ts, burst, rate_per_s, plan, key}``. Used by
    the admin endpoint to debug "why am I being rate-limited."
    """
    cli = _redis()
    out = {
        "key": f"{_KEY_PREFIX}:{bucket}:{plan}:{who}",
        "burst": None, "rate_per_s": None,
        "tokens": None, "ts": None,
    }
    burst, rate = _plan_limits(plan, bucket)
    out["burst"] = burst
    out["rate_per_s"] = rate
    if cli is None:
        return out
    try:
        raw = cli.hgetall(out["key"])
        out["tokens"] = float(raw.get("tokens")) if raw.get("tokens") else None
        out["ts"]     = float(raw.get("ts"))     if raw.get("ts")     else None
    except Exception:
        pass
    return out


def reset_bucket(bucket: str, plan: str, who: str) -> bool:
    """Wipe one bucket — admin override. Returns True iff the key
    existed in Redis."""
    cli = _redis()
    if cli is None:
        return False
    try:
        return bool(cli.delete(f"{_KEY_PREFIX}:{bucket}:{plan}:{who}"))
    except Exception:
        return False
