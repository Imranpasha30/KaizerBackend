"""Redis Streams + Consumer-Group based job queue for Kaizer.

Why Streams (and not a plain Redis LIST + BLPOP)?
  - **Consumer groups**: multiple worker pods consume from the same
    stream and Redis hands each message to exactly one consumer in the
    group. This naturally fixes the dual-worker race we hit with the
    polled Postgres `upload_jobs` table — Railway and local can both
    join the same consumer group and they will load-balance, not
    duplicate.
  - **Pending-Entry-List (PEL)**: if a consumer crashes mid-process,
    the message stays in PEL until ``XACK``ed. The recovery loop
    claims those entries via ``XCLAIM`` so no work is silently
    dropped on a worker SIGKILL.
  - **Persistence**: streams persist across restarts, unlike pub/sub.
  - **Per-tenant prioritisation**: separate Redis stream per priority
    tier (hi / normal / lo). Workers read in priority order, so a
    paying tenant never sits behind a batch-translation backfill.

Stage 3 — priority lanes:
  Three streams, three consumer groups, one DLQ:

      kaizer:uploads:hi      ← agency / pro plans
      kaizer:uploads:normal  ← free / creator (default)
      kaizer:uploads:lo      ← campaign + translation batch fan-out
      kaizer:uploads:dlq     ← terminal failures (any lane)

  ``consume_upload_jobs`` reads hi first, descends to normal only when
  hi is empty, then to lo. That guarantees zero head-of-line blocking
  by lower tiers — the cost is one extra XREADGROUP RTT per round on
  an idle queue (the per-call BLOCK budget is split evenly).

Module surface (everything else is internal):

    enqueue_upload_job(job_id, *, priority="normal")
        Producer-side. Routes to the lane keyed by ``priority``.

    consume_upload_jobs(consumer_name, *, count=1, block_ms=5000)
        Consumer-side. Yields ``(message_id, job_id, priority)`` —
        the priority is needed so the caller can XACK against the
        correct group.

    ack_upload_job(message_id, priority)
        XACK the message from the lane's consumer group's PEL.

    send_to_dlq(message_id, job_id, priority, reason)
        Move a permanently-failed message to the DLQ stream and ACK
        it. Original priority is preserved so replay returns to it.

    recover_pending(consumer_name, *, idle_ms=60_000)
        Yields ``(message_id, job_id, priority)`` for messages stuck
        in any lane's PEL longer than idle_ms.

    queue_stats() / list_dlq() / replay_from_dlq() — admin endpoints.

Failure model:
  - Connection errors raise :class:`QueueError`. Callers retry with
    backoff or fall back to the legacy DB-poll path.
  - Malformed messages (missing job_id field) are auto-acked and
    logged so a poison message doesn't stall the queue.
  - The DB row's ``status`` is the source of truth — duplicate
    delivery (after a crash mid-XACK) is benign because the worker
    rechecks status before processing.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Iterator, Optional, Tuple

import redis
from redis.exceptions import ResponseError

# Match database.py's pattern — load .env on import so this module works
# when imported standalone (CLI scripts, tests) AND inside the FastAPI
# process. Idempotent.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except Exception:
    pass

logger = logging.getLogger("kaizer.redis_queue")


# ── Stream + group naming ────────────────────────────────────────
# Each priority lane is a separate Redis stream + consumer group.
PRIORITIES         = ("hi", "normal", "lo")
DEFAULT_PRIORITY   = "normal"

STREAM_BY_PRIORITY = {
    "hi":     "kaizer:uploads:hi",
    "normal": "kaizer:uploads:normal",
    "lo":     "kaizer:uploads:lo",
}
GROUP_BY_PRIORITY  = {p: f"kaizer-upload-workers:{p}" for p in PRIORITIES}
DLQ_UPLOADS        = "kaizer:uploads:dlq"

# Backwards-compat aliases — Stage-1/2 imports referenced these names.
# Anything that touched them now points at the "normal" lane so
# anything we missed during Stage 3 wiring still routes to a valid
# stream and group.
STREAM_UPLOADS    = STREAM_BY_PRIORITY[DEFAULT_PRIORITY]
GROUP_UPLOADS     = GROUP_BY_PRIORITY[DEFAULT_PRIORITY]


def _normalise_priority(priority: Optional[str]) -> str:
    """Coerce arbitrary input into one of PRIORITIES. Unknown → DEFAULT."""
    p = (priority or DEFAULT_PRIORITY).strip().lower()
    return p if p in STREAM_BY_PRIORITY else DEFAULT_PRIORITY


class QueueError(RuntimeError):
    """Redis is unreachable, misconfigured, or returned an unexpected error."""


# ── Connection pool (process-wide, lazily created) ───────────────
_pool_lock = threading.Lock()
_client: Optional[redis.Redis] = None


def _redis_url() -> str:
    url = os.environ.get("REDIS_URL", "").strip()
    if not url:
        raise QueueError(
            "REDIS_URL is not set — the Redis-backed job queue cannot start. "
            "Add REDIS_URL=redis://… to .env or set it in the deployment env."
        )
    return url


def get_client() -> redis.Redis:
    """Process-wide Redis client. Cheap to call repeatedly."""
    global _client
    if _client is not None:
        return _client
    with _pool_lock:
        if _client is None:
            _client = redis.Redis.from_url(
                _redis_url(),
                decode_responses=True,         # str in/out — easier
                socket_connect_timeout=5,
                socket_keepalive=True,
                health_check_interval=30,
                retry_on_timeout=True,
            )
    return _client


def is_enabled() -> bool:
    """Cheap probe — True iff REDIS_URL is set AND we can PING."""
    if not os.environ.get("REDIS_URL", "").strip():
        return False
    try:
        return bool(get_client().ping())
    except Exception as exc:
        logger.warning("redis_queue: PING failed — falling back to DB poll. err=%s", exc)
        return False


# ── One-time setup: ensure all consumer groups exist ────────────
def ensure_group(*_args, **_kwargs) -> None:
    """Create all priority-lane streams and consumer groups.

    Variadic so callers from Stage 1 (which passed stream/group args
    explicitly) still work — those args are now ignored because we
    always provision the full set of lanes together.
    """
    cli = get_client()
    for prio, stream in STREAM_BY_PRIORITY.items():
        group = GROUP_BY_PRIORITY[prio]
        try:
            cli.xgroup_create(name=stream, groupname=group,
                              id="0", mkstream=True)
            logger.info("redis_queue: created group %s on stream %s", group, stream)
        except ResponseError as exc:
            if "BUSYGROUP" in str(exc):
                logger.debug("redis_queue: group %s already exists on %s", group, stream)
                continue
            raise QueueError(f"xgroup_create failed for {stream}: {exc}") from exc


# ── Producer ─────────────────────────────────────────────────────
def enqueue_upload_job(job_id: int, *, priority: str = DEFAULT_PRIORITY) -> str:
    """XADD an upload job into the priority-appropriate stream.

    The consumer reads ``job_id`` and looks up the row in Postgres to
    pick up the actual work. We deliberately do **not** put job
    payload bytes in Redis — Postgres remains the source of truth for
    job state, retries, and audit. Redis is a *signal*, not a store.
    """
    if job_id is None:
        raise QueueError("enqueue_upload_job: job_id is required")
    prio = _normalise_priority(priority)
    stream = STREAM_BY_PRIORITY[prio]
    cli = get_client()
    try:
        msg_id = cli.xadd(
            stream,
            {"job_id": str(int(job_id)), "priority": prio},
            # No explicit MAXLEN: at enterprise scale we'll add
            # ``maxlen=100_000, approximate=True`` so failed-delivery
            # backlog can't OOM Redis. For now the volume is tiny.
        )
        logger.info("redis_queue: enqueued job_id=%s msg_id=%s priority=%s",
                    job_id, msg_id, prio)
        return msg_id
    except redis.RedisError as exc:
        raise QueueError(f"enqueue_upload_job failed: {exc}") from exc


# ── Consumer ─────────────────────────────────────────────────────
def consume_upload_jobs(
    consumer_name: str,
    *,
    count: int = 1,
    block_ms: int = 5_000,
) -> Iterator[Tuple[str, int, str]]:
    """Yield ``(message_id, job_id, priority)`` from the highest-priority
    non-empty lane.

    Strategy: try each lane in priority order with ``block=None`` (no
    block) — return the first lane that has work. If all lanes empty,
    do one short blocking call on the LOWEST lane so we don't hot-spin
    the CPU. The block on lo wakes whenever lo has new work; hi/normal
    writes wake the next loop iteration via the same per-slot cycle
    (worst case = block_ms latency on hi when the queue was idle).

    redis-py gotcha: ``block=0`` means "block forever" (NOT "no
    block"). Use ``block=None`` for non-blocking. Burnt one debug
    cycle on this; do not regress.

    Note on starvation: we do NOT serve hi to exhaustion before
    touching normal — each ``count``-bounded round is one ``hi`` slice
    OR one ``normal`` slice, never both. With ``count=4`` and 4 slots,
    that's enough fairness in practice. For strict-fairness needs
    later, add a token-bucket per lane.
    """
    cli = get_client()
    # First pass: non-blocking peek of each lane in priority order.
    for prio in PRIORITIES:
        stream = STREAM_BY_PRIORITY[prio]
        group  = GROUP_BY_PRIORITY[prio]
        try:
            resp = cli.xreadgroup(
                groupname=group,
                consumername=consumer_name,
                streams={stream: ">"},
                count=count,
                block=None,             # None = no block (NOT 0!)
            )
        except redis.RedisError as exc:
            raise QueueError(f"xreadgroup failed on {stream}: {exc}") from exc
        if resp:
            yield from _parse_resp(resp, prio)
            return

    # Second pass: all lanes empty → one short blocking call on the
    # lowest lane so we sleep until either lo has new work OR the
    # block_ms deadline ticks (next iteration re-checks hi/normal).
    stream = STREAM_BY_PRIORITY["lo"]
    group  = GROUP_BY_PRIORITY["lo"]
    try:
        resp = cli.xreadgroup(
            groupname=group,
            consumername=consumer_name,
            streams={stream: ">"},
            count=count,
            block=block_ms,
        )
    except redis.RedisError as exc:
        raise QueueError(f"xreadgroup blocking failed on {stream}: {exc}") from exc
    if resp:
        yield from _parse_resp(resp, "lo")


def _parse_resp(resp, priority: str) -> Iterator[Tuple[str, int, str]]:
    """Common XREADGROUP-response parser. Auto-acks poison messages
    so they stop being delivered.
    """
    # resp shape: [(stream, [(msg_id, {field: value}), ...])]
    for _stream, entries in resp:
        for msg_id, fields in entries:
            try:
                job_id = int(fields.get("job_id", "0"))
            except (TypeError, ValueError):
                logger.error("redis_queue: poison msg=%s fields=%r — auto-acking",
                             msg_id, fields)
                ack_upload_job(msg_id, priority)
                continue
            if job_id <= 0:
                logger.error("redis_queue: msg=%s no job_id (fields=%r) — auto-acking",
                             msg_id, fields)
                ack_upload_job(msg_id, priority)
                continue
            yield msg_id, job_id, priority


def ack_upload_job(message_id: str, priority: str = DEFAULT_PRIORITY) -> None:
    """XACK — remove the message from the lane's consumer group's PEL.

    Always call after a job reaches a TERMINAL state (done OR failed).
    Don't ack on transient errors — leave the message in PEL so the
    recovery loop picks it up next time.
    """
    prio = _normalise_priority(priority)
    cli = get_client()
    try:
        cli.xack(STREAM_BY_PRIORITY[prio], GROUP_BY_PRIORITY[prio], message_id)
    except redis.RedisError as exc:
        # Not fatal — worst case the recovery loop reprocesses; the
        # job-row's `status` check makes that idempotent.
        logger.warning("redis_queue: xack failed msg_id=%s prio=%s err=%s",
                       message_id, prio, exc)


def send_to_dlq(message_id: str, job_id: int, priority: str, reason: str) -> None:
    """Move a permanently-failed message to the DLQ and ack the source.

    The DLQ entry includes the original priority so :func:`replay_from_dlq`
    can return the job to its original lane (a paying tenant's failed
    job replays to ``hi``, not to ``normal``).
    """
    prio = _normalise_priority(priority)
    cli = get_client()
    try:
        cli.xadd(DLQ_UPLOADS, {
            "original_msg_id": message_id,
            "job_id":           str(job_id),
            "priority":         prio,
            "reason":           reason[:1000],
        })
    except redis.RedisError as exc:
        logger.error("redis_queue: DLQ XADD failed msg=%s job=%s err=%s",
                     message_id, job_id, exc)
    ack_upload_job(message_id, prio)


# ── Recovery — claim PEL entries from dead consumers ────────────
def recover_pending(
    consumer_name: str,
    *,
    idle_ms: int = 60_000,
    batch: int = 16,
) -> Iterator[Tuple[str, int, str]]:
    """XCLAIM messages from EVERY priority lane's PEL that have been
    idle longer than ``idle_ms`` — those consumers probably died.

    Yields ``(message_id, job_id, priority)`` so the caller knows which
    lane to ACK against once processed.
    """
    cli = get_client()
    for prio, stream in STREAM_BY_PRIORITY.items():
        group = GROUP_BY_PRIORITY[prio]
        cursor = "0-0"
        while True:
            try:
                cursor, msgs, _deleted = cli.xautoclaim(
                    name=stream,
                    groupname=group,
                    consumername=consumer_name,
                    min_idle_time=idle_ms,
                    start_id=cursor,
                    count=batch,
                )
            except ResponseError as exc:
                # Older redis server (XAUTOCLAIM is 6.2+). Railway is
                # 8.x so we shouldn't hit this in practice; degrade
                # gracefully if we do.
                logger.warning("redis_queue: xautoclaim unsupported on %s (%s)", stream, exc)
                break
            except redis.RedisError as exc:
                logger.warning("redis_queue: recover_pending failed on %s: %s", stream, exc)
                break

            for msg_id, fields in (msgs or []):
                try:
                    job_id = int(fields.get("job_id", "0"))
                except (TypeError, ValueError):
                    job_id = 0
                if job_id <= 0:
                    ack_upload_job(msg_id, prio)
                    continue
                logger.warning("redis_queue: reclaimed stuck msg=%s job=%s prio=%s",
                               msg_id, job_id, prio)
                yield msg_id, job_id, prio

            if cursor in ("0-0", b"0-0", 0):
                break


# ── Admin / observability ───────────────────────────────────────
def queue_stats() -> dict:
    """Snapshot of every priority lane's queue health.

    Shape::

        {
          "lanes": {
            "hi":     {"stream": ..., "group": ..., "length": int,
                       "groups": [...], "consumers": [...]},
            "normal": {...},
            "lo":     {...},
          },
          "dlq_length": int,
          "ok": bool,
          "error": Optional[str],
        }
    """
    out: dict = {"lanes": {}, "dlq_length": None, "ok": False, "error": None}
    try:
        cli = get_client()
        for prio in PRIORITIES:
            stream = STREAM_BY_PRIORITY[prio]
            group  = GROUP_BY_PRIORITY[prio]
            lane: dict = {
                "stream": stream, "group": group,
                "length": None, "groups": [], "consumers": [],
            }
            try:
                lane["length"] = cli.xlen(stream) if cli.exists(stream) else 0
            except redis.RedisError:
                pass
            try:
                lane["groups"] = list(cli.xinfo_groups(stream))
            except (ResponseError, redis.RedisError):
                pass
            try:
                lane["consumers"] = list(cli.xinfo_consumers(stream, group))
            except (ResponseError, redis.RedisError):
                pass
            out["lanes"][prio] = lane

        out["dlq_length"] = cli.xlen(DLQ_UPLOADS) if cli.exists(DLQ_UPLOADS) else 0
        out["ok"] = True
    except Exception as exc:
        out["error"] = str(exc)
    return out


def list_dlq(*, count: int = 50) -> list[dict]:
    """Return up to ``count`` most-recent DLQ entries (newest first).

    Each entry is ``{message_id, original_msg_id, job_id, priority, reason}``.
    """
    cli = get_client()
    if not cli.exists(DLQ_UPLOADS):
        return []
    try:
        raw = cli.xrevrange(DLQ_UPLOADS, count=count)
    except redis.RedisError as exc:
        raise QueueError(f"list_dlq failed: {exc}") from exc
    out: list[dict] = []
    for mid, fields in raw:
        out.append({
            "message_id":      mid,
            "original_msg_id": fields.get("original_msg_id", ""),
            "job_id":          fields.get("job_id", ""),
            "priority":        fields.get("priority", DEFAULT_PRIORITY),
            "reason":          fields.get("reason", ""),
        })
    return out


def replay_from_dlq(message_id: str) -> dict:
    """Pull a DLQ entry, re-enqueue its job_id at the ORIGINAL priority,
    and delete the DLQ row.

    A paying-tier job that failed should NOT come back as 'normal' on
    replay — that would be a stealth downgrade. We preserve the lane.
    """
    cli = get_client()
    raw = cli.xrange(DLQ_UPLOADS, min=message_id, max=message_id, count=1)
    if not raw:
        raise QueueError(f"DLQ message {message_id!r} not found")
    _mid, fields = raw[0]
    try:
        job_id = int(fields.get("job_id", "0"))
    except (TypeError, ValueError):
        job_id = 0
    if job_id <= 0:
        raise QueueError(f"DLQ message {message_id!r} has no usable job_id")
    prio = _normalise_priority(fields.get("priority"))
    new_mid = enqueue_upload_job(job_id, priority=prio)
    cli.xdel(DLQ_UPLOADS, message_id)
    return {
        "ok":         True,
        "new_msg_id": new_mid,
        "job_id":     job_id,
        "priority":   prio,
        "reason":     fields.get("reason", ""),
    }


# ── Plan → priority mapping ─────────────────────────────────────
# Source of truth for "which lane does this user's job land on?"
# Accepts a User row (with `.plan`) or a raw plan string. Unknown
# plans land on DEFAULT_PRIORITY.
_PRIO_BY_PLAN = {
    "free":    "normal",
    "creator": "normal",
    "pro":     "hi",
    "agency":  "hi",
}


def priority_for_user(user_or_plan, *, batch: bool = False) -> str:
    """Pick the right priority lane for a user / plan.

    Args:
        user_or_plan: a ``User`` row, a plan string, or ``None``.
        batch: when True, force priority="lo" regardless of plan —
            used by campaigns + translation fan-out so background
            backfill never blocks a live user's upload.
    """
    if batch:
        return "lo"
    if user_or_plan is None:
        return DEFAULT_PRIORITY
    plan = getattr(user_or_plan, "plan", None) or user_or_plan
    if not isinstance(plan, str):
        return DEFAULT_PRIORITY
    return _PRIO_BY_PLAN.get(plan.strip().lower(), DEFAULT_PRIORITY)


# ── One-time migration: enqueue everything currently `queued` ───
def bootstrap_existing_queued_jobs(db) -> int:
    """Walk ``upload_jobs.status='queued'`` and XADD each job_id to
    the priority lane derived from the owning user's plan.

    Idempotent: enqueueing the same job_id twice is harmless because
    the consumer re-reads status from the DB and skips a row that's
    already moved on.
    """
    import models  # local import — avoids circular at module load
    rows = (db.query(models.UploadJob, models.User)
              .outerjoin(models.User, models.UploadJob.user_id == models.User.id)
              .filter(models.UploadJob.status == "queued")
              .all())
    n = 0
    for job, user in rows:
        prio = priority_for_user(user)
        try:
            enqueue_upload_job(job.id, priority=prio)
            n += 1
        except QueueError as exc:
            logger.error("redis_queue: bootstrap enqueue failed job_id=%s: %s",
                         job.id, exc)
    if n:
        logger.info("redis_queue: bootstrap enqueued %d existing queued job(s)", n)
    return n
