"""Admin endpoints for the Live Map tab.

`GET /api/admin/live/map` is one read that returns every stage of live streaming
as it is right now — the data behind the node graph. The panel polls it every two
seconds, which is why it is one endpoint and not nine: nine round trips at that
rate is a self-inflicted load problem, and the parts genuinely depend on each
other (a channel's health only means something if the worker that wrote it is
alive).

The two actions here exist because the map offers them and an action with no
endpoint behind it is worse than no action. Both are admin-only and both operate
across users, which is exactly why they could not reuse the customer-facing
routes: those check ownership, so an admin clearing a blocked channel for a
customer would get a 403.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import auth
import models
from database import get_db

router = APIRouter(prefix="/api/admin/live", tags=["admin", "live"])


def _service():
    """The live engine, or a 503 that says how to turn it on.

    A 404 would be wrong: the tab exists, the admin is allowed to see it, and
    the engine simply is not running here. 503 with the reason is what lets
    someone fix it instead of wondering whether the page is broken.
    """
    from live_integration import get_live_service
    svc = get_live_service()
    if svc is None:
        raise HTTPException(
            503,
            "The live engine (v2) is not running in this deployment, so there is "
            "nothing to map. It needs KAIZER_LIVE_ENGINE=v2, a Fernet key, a "
            "runnable ffmpeg/ffprobe, and Redis reachable at KAIZER_LIVE_REDIS. "
            "The backend log says which of those is missing.")
    return svc


@router.get("/map")
def live_map(events: int = 40, db: Session = Depends(get_db),
             _=Depends(auth.admin_required)):
    """Every stage of the live pipeline, from real state. No secrets."""
    from live_map import build_map
    return build_map(_service(), db, events=max(1, min(events, 200)))


@router.post("/encodes/retry")
def retry_failed_encodes(_=Depends(auth.admin_required)):
    """Put today's failed encode jobs back on the queue.

    Most encode failures are transient in a way a retry fixes — the box was out
    of disk, ffmpeg was killed by a restart — and the alternative was asking the
    customer to upload the same file again. A job that fails for a real reason
    fails again and keeps its error, so retrying is safe; it is not a way to make
    an unfixable file streamable.
    """
    from kaizer_live import encode as enc
    svc = _service()
    r, prefix = svc.r, svc.s.prefix
    requeued, skipped = [], []
    for key in r.scan_iter(f"{prefix}:encode:job:*"):
        rec = r.hgetall(key)
        if rec.get("state") != "failed":
            continue
        job_id = rec.get("job_id") or key.rsplit(":", 1)[-1]
        # Three attempts is where a transient failure stops being transient.
        if int(rec.get("attempts") or 0) >= 3:
            skipped.append({"job_id": job_id, "attempts": int(rec.get("attempts") or 0),
                            "error": (rec.get("error") or "")[:200]})
            continue
        r.hset(key, mapping={"state": "queued", "error": ""})
        r.rpush(enc.queue_key(prefix), job_id)
        requeued.append(job_id)
    return {"requeued": requeued, "skipped_after_3_attempts": skipped,
            "queue_depth": enc.queue_depth(r, prefix)}


@router.delete("/channels/{channel_id}/block")
def admin_unblock_channel(channel_id: int, db: Session = Depends(get_db),
                          _=Depends(auth.admin_required)):
    """Clear a channel's block, for any customer.

    A channel is blocked after a failure that will repeat until somebody changes
    something on YouTube — live streaming not enabled, a revoked token — so that
    the next try is refused for free instead of spending 103 units to fail the
    same way. Clearing it is the "I have fixed it, try again" button.

    The customer-facing route cannot serve this: it checks ownership, so an admin
    helping a customer would be refused their own 403.
    """
    ch = db.query(models.Channel).get(channel_id)
    if ch is None:
        raise HTTPException(404, f"channel {channel_id} not found")
    svc = _service()
    was = svc.blocked(str(channel_id))
    svc.unblock_channel(str(channel_id))
    return {"channel_id": channel_id, "channel": ch.name, "blocked": False,
            "was_blocked_because": was or ""}
