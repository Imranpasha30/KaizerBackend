"""The admin Live Pipeline Map: every stage of live streaming, from real state.

This is the data behind the admin "Live Map" tab. It is modelled on the planning
rig the founder drew (`live-rig.html`) — the same nodes, the same wires — with
one difference that is the entire point: the rig simulated a day, and this
reports the one actually happening. Every number here comes from Redis, from the
relays' own counters, or from the database. Nothing is estimated, and where a
figure cannot be known it is null rather than a plausible guess.

WHERE EACH NUMBER COMES FROM, because a panel nobody trusts is worse than none:

  uploads / checker   counters bumped where the checker runs (nothing else
                      records a file that passed and went straight to R2)
  encode queue        the queue's own length, and its workers' heartbeats
  redis               INFO from the live stack's own instance
  control             the sweeper's heartbeat; STALE IS THE INTERESTING CASE,
                      because a missing sweeper breaks nothing visibly — videos
                      go live and simply never end
  workers             worker heartbeats, which expire, so a dead box disappears
                      rather than lingering as "ok"
  relays              the relay registry plus each relay's own reported state
  channels            per-channel health the worker writes, including a bitrate
                      it measures over a 5 s window
  youtube             what the single confirmation actually returned
  credit              the ledger, including the reserve held back so open
                      broadcasts can always be ended

NO SECRETS LEAVE HERE. Not the presigned R2 source (it carries credentials in its
query string), not a destination URL, not a stream key. `_scrub` is applied to
every dict on the way out and the tests assert on it, because this endpoint is
the one place where everything about a live stream is gathered in one object —
which makes it exactly the wrong place to be careless.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import models

#: Fields that must never reach a browser, whatever they are attached to.
#: dst_enc is the sealed destination; source/upload_path are presigned URLs or
#: server paths; the rest are belt and braces for anything added later.
SECRET_FIELDS = frozenset({
    "dst_enc", "stream_key", "ingest_url", "source", "source_url", "upload_path",
    "url", "fernet_key", "refresh_token", "refresh_token_enc",
})

#: Health states a channel can be in, and whether an operator should act.
# "starved" is the link failing to carry the stream: YouTube's
# videoIngestionStarved, where viewers buffer. It reads healthy from every
# angle except the only one that matters, so it belongs with the broken ones.
_BAD = ("stalled", "failed", "starved")
_WARN = ("reconnecting", "restarting", "connecting", "starting", "queued")


def _scrub(d: Any) -> Any:
    """Drop secret fields from anything on its way to the browser.

    Applied to the whole payload rather than at each call site: this function is
    the one place to look to be sure, and a field added to a node next year is
    covered without anyone remembering to think about it.
    """
    if isinstance(d, dict):
        return {k: _scrub(v) for k, v in d.items() if k not in SECRET_FIELDS}
    if isinstance(d, list):
        return [_scrub(x) for x in d]
    return d


def _worst(*states: str) -> str:
    """The state a node should show when its parts disagree: the worst one.

    A node that is half broken is broken. Showing "ok" because most of its
    channels are fine is how a panel teaches people to ignore it.
    """
    for s in ("bad", "warn", "idle"):
        if s in states:
            return s
    return "ok"


def _node(state: str, name: str, sub: str = "", *, metrics=None, detail: str = "",
          actions=None, items=None) -> dict:
    return {"state": state, "name": name, "sub": sub, "metrics": metrics or [],
            "detail": detail, "actions": actions or [], "items": items or []}


def _metric(node: dict, label: str):
    """One metric's value by name. By name and not by position, because reading
    metrics[0] means adding a metric silently relabels a wire."""
    for m in node.get("metrics", []):
        if m["label"] == label:
            return m["value"]
    return None


def _m(label: str, value, unit: str = "") -> dict:
    """One metric. `value` may be None, which the panel renders as a dash —
    never as 0, because "not known" and "none" are different facts."""
    return {"label": label, "value": value, "unit": unit}


# ── the pieces ───────────────────────────────────────────────────────

def _redis_node(r, prefix: str) -> dict:
    """The control plane. If this is unhealthy nothing else on the map is true."""
    try:
        info = r.info()
    except Exception as exc:
        return _node("bad", "Redis", "control plane",
                     detail=f"unreachable: {exc}. Nothing on this map can be trusted "
                            f"while this is down, and no stream can be started or stopped.")
    used_mb = (info.get("used_memory") or 0) / 1048576
    max_mb = (info.get("maxmemory") or 0) / 1048576
    policy = info.get("maxmemory_policy", "?")
    aof = int(info.get("aof_enabled") or 0) == 1
    keys = sum(v.get("keys", 0) for k, v in info.items()
               if k.startswith("db") and isinstance(v, dict))

    state, detail = "ok", ""
    # An eviction policy other than noeviction is a live-stream outage waiting
    # to happen: the evicted key is a channel's desired state, the worker's next
    # reconcile sees nothing wanted, and the stream stops with nothing logged.
    if policy != "noeviction":
        state = "bad"
        detail = (f"maxmemory-policy is {policy}, not noeviction. Under memory pressure "
                  f"Redis will delete a live video's state and the stream will stop "
                  f"with no error anywhere.")
    elif not aof:
        state = "warn"
        detail = ("append-only file is off: restarting Redis would lose the state of "
                  "streams that are still publishing, leaving their broadcasts open "
                  "on YouTube with nothing able to end them.")
    elif max_mb and used_mb / max_mb > 0.85:
        state = "warn"
        detail = f"using {used_mb:.0f} of {max_mb:.0f} MB; writes fail outright at the limit."
    return _node(state, "Redis", f"{info.get('redis_version', '?')} on {prefix}",
                 metrics=[_m("keys", keys),
                          _m("memory", round(used_mb, 1), "MB"),
                          _m("policy", policy),
                          _m("persistence", "AOF" if aof else "none"),
                          _m("clients", info.get("connected_clients"))],
                 detail=detail)


def _control_node(r, prefix: str) -> dict:
    """The sweeper. Its absence is the quietest serious fault in the system."""
    from kaizer_live.control import sweeper_status
    st = sweeper_status(r, prefix)
    if st.get("alive"):
        return _node("warn" if st.get("last_error") else "ok", "Control", "the sweeper",
                     metrics=[_m("sweeps", st.get("sweeps")),
                              _m("last seen", st.get("age_s"), "s"),
                              _m("on", st.get("host") or "?")],
                     detail=st.get("last_error") or "")
    return _node("bad", "Control", "the sweeper",
                 metrics=[_m("sweeps", None), _m("last seen", st.get("age_s"), "s")],
                 detail="No sweeper is running. Broadcasts will start and will then "
                        "never be confirmed or ended, and the credit held back to end "
                        "them stays reserved. Start it with: python -m kaizer_live.control "
                        "--service-factory live_integration:get_live_service")


def _workers_node(service) -> dict:
    """The boxes that move bytes. A heartbeat expires, so a dead box vanishes
    rather than sitting here looking healthy."""
    ws = service.workers()
    if not ws:
        return _node("bad", "Stream workers", "none running",
                     detail="No stream worker is alive, so nothing can go live: go_live "
                            "refuses with 503 rather than accepting a broadcast it cannot "
                            "carry. Start one with: python -m kaizer_live.worker --id <name>")
    items, states = [], []
    for w in sorted(ws, key=lambda x: x.get("id", "")):
        slots = int(w.get("slots") or 0)
        used = int(w.get("reserved") or 0)
        share = used / slots if slots else 0
        st = "bad" if share >= 1 else "warn" if share > 0.85 else "ok"
        states.append(st)
        items.append({"id": w.get("id"), "host": w.get("host"), "state": st,
                      "relays": int(w.get("relays") or 0),
                      "max_relays": int(w.get("max_relays") or 0) or None,
                      "channels": used, "slots": slots,
                      "health_addr": w.get("health_addr") or ""})
    return _node(_worst(*states), "Stream workers", f"{len(ws)} box{'es' if len(ws) != 1 else ''}",
                 metrics=[_m("channels", sum(i["channels"] for i in items)),
                          _m("capacity", sum(i["slots"] for i in items)),
                          _m("relays", sum(i["relays"] for i in items))],
                 items=items)


def _relays_node(r, prefix: str, live: list[dict]) -> dict:
    """One relay per live video: its single ffmpeg, and its connections."""
    regs = {}
    for key in r.scan_iter(f"{prefix}:relay:*"):
        h = r.hgetall(key)
        if h.get("video_id"):
            regs[h["video_id"]] = h
    if not live:
        return _node("idle", "Relays", "nothing live",
                     metrics=[_m("relays", 0), _m("ffmpeg", 0)])
    states = []
    for v in live:
        rd = v.get("reader") or {}
        rs = rd.get("state") or ""
        states.append("bad" if rs in ("failed", "stalled") else
                      "warn" if rs in ("starting", "restarting") else "ok")
    # A live video with no relay recorded is the interesting failure: the desired
    # state says it should be on air and nothing is carrying it.
    missing = [v["video_id"] for v in live if v["video_id"] not in regs]
    if missing:
        states.append("bad")
    restarts = sum(int((v.get("reader") or {}).get("relay_restarts") or 0) for v in live)
    return _node(_worst(*states), "Relays", f"{len(regs)} running, one per video",
                 metrics=[_m("relays", len(regs)),
                          _m("ffmpeg", len(regs)),          # exactly one per relay, by design
                          _m("restarts", restarts),
                          _m("adopted", sum(1 for h in regs.values() if h.get("worker_id")))],
                 detail=("no relay is recorded for " + ", ".join(missing)) if missing else "")


def _credit_node(service) -> dict:
    snap = service.ledger.snapshot()
    limit = snap["limit"] or 1
    spent = (snap["used"] + snap["reserve"]) / limit
    state = "bad" if snap["broadcasts_left_today"] == 0 else "warn" if spent > 0.8 else "ok"
    return _node(state, "Daily credit", "YouTube Data API units",
                 metrics=[_m("used", snap["used"]),
                          _m("reserved", snap["reserve"]),
                          _m("limit", limit),
                          _m("per broadcast", snap["broadcast_cost"]),
                          _m("left today", snap["broadcasts_left_today"], "broadcasts")],
                 detail=("Credit is exhausted; channels queue until the reset. "
                         if snap["broadcasts_left_today"] == 0 else "")
                        + f"{snap['reserve']} units are held back so the "
                          f"{snap['open_broadcasts']} open broadcast(s) can always be ended.",
                 actions=[{"kind": "set_user_cap", "label": "edit a user's cap"}])


def _encode_node(r, prefix: str, counts: dict) -> dict:
    from kaizer_live import encode as enc
    depth = enc.queue_depth(r, prefix)
    workers = enc.encode_workers(r, prefix)
    state = "ok"
    detail = ""
    if depth and not workers:
        state, detail = "bad", (f"{depth} file(s) waiting and no encode worker running: "
                                f"they will never become streamable. Start one with: "
                                f"python -m kaizer_live.encode --id <name>")
    elif depth > 10:
        state, detail = "warn", f"{depth} files waiting; each is minutes of CPU."
    elif counts.get("encodes_failed"):
        state = "warn"
        detail = f"{counts['encodes_failed']} encode(s) failed today."
    elif not depth and not workers:
        state = "idle"
    return _node(state, "Encode queue", "repairs files the checker rejected",
                 metrics=[_m("waiting", depth),
                          _m("workers", len(workers)),
                          _m("done today", counts.get("encodes_done")),
                          _m("failed today", counts.get("encodes_failed"))],
                 detail=detail,
                 items=[{"id": w.get("id"), "host": w.get("host"),
                         "current": w.get("current") or "", "done": int(w.get("done") or 0),
                         "failed": int(w.get("failed") or 0)} for w in workers],
                 actions=[{"kind": "retry_encodes", "label": "retry failed jobs"}])


def _checker_node(counts: dict) -> dict:
    checked = counts.get("uploads_checked") or 0
    passed = counts.get("uploads_passed") or 0
    queued = counts.get("encodes_queued") or 0
    state = "idle" if not checked else "ok"
    detail = ""
    # Files arriving and none passing is a real fault with no other symptom: the
    # customer sees uploads that never become streamable and no error anywhere.
    if checked >= 3 and passed == 0:
        state = "warn"
        detail = (f"{checked} file(s) checked today and none went live as they were; "
                  f"{queued} went to the encode queue. If that is every upload, the "
                  f"encoder writing them needs its keyframe interval fixed.")
    return _node(state, "Checker", "ffprobe, before any credit is spent",
                 metrics=[_m("checked today", checked),
                          _m("stream-ready", passed),
                          _m("sent to encode", queued)],
                 detail=detail)


#: How long a channel may be pushed to without YouTube confirming the broadcast
#: live before that counts as a fault rather than a startup. Ours confirm in
#: about 18 seconds; the one that prompted this took several minutes, and the
#: panel called it broken the whole time -- a false alarm is the expensive kind,
#: because it teaches an operator that red means nothing.
CONFIRM_GRACE_S = 180.0


def _youtube_node(live: list[dict]) -> dict:
    """Is YouTube getting what it needs? Not "is a process running".

    STARVED CHANNELS BELONG HERE. A broadcast ran for half an hour with YouTube
    reporting videoIngestionStarved -- viewers buffering -- while this node read
    `ok`, because it only counted whether the broadcast was confirmed live. The
    channel table showed the fault and the GRAPH did not, which defeats the
    point of drawing one: the whole reason for a map is that live streaming
    fails at the joins, and this is the join between the relay and YouTube.
    """
    on = confirmed = not_live = starting = starved = 0
    need = sent = 0.0
    now = time.time()
    for v in live:
        for c in v.get("channels", []):
            if c.get("state") != "on":
                continue
            on += 1
            yt = c.get("youtube")
            confirmed += yt == "live"
            # THREE DIFFERENT THINGS, not one.
            #
            #   pending      still inside the first checks
            #   unconfirmed  we stopped asking; the engine re-asks on a backoff
            #   not_live     YouTube itself said complete/revoked
            #
            # Only the last is a fault. `unconfirmed` means our check budget ran
            # out, which says nothing about the broadcast -- reporting "YouTube
            # does not report them live" about a stream that is live and healthy
            # is the false alarm this exists to remove.
            if yt == "not_live":
                not_live += 1
            elif yt == "unconfirmed":
                starting += 1
            elif yt == "pending":
                try:
                    joined = float(c.get("joined_at") or 0)
                except (TypeError, ValueError):
                    joined = 0.0
                # No joined_at is judged, not excused: missing timing must never
                # become a way to never report a fault.
                if joined and (now - joined) < CONFIRM_GRACE_S:
                    starting += 1
                else:
                    not_live += 1
            h = c.get("health") or {}
            if h.get("state") == "starved":
                starved += 1
                try:
                    need += float(h.get("source_mbps") or 0)
                    sent += float(h.get("kbps") or 0) / 1000.0
                except (TypeError, ValueError):
                    pass
    if not on:
        return _node("idle", "YouTube", "nothing on air",
                     metrics=[_m("on air", 0)])

    state = ("bad" if (not_live or starved)
             else "warn" if (starting or confirmed < on) else "ok")
    detail = ""
    if starved:
        detail = (f"{starved} channel(s) cannot be fed fast enough: about "
                  f"{sent:.2f} Mbps is getting through of the {need:.2f} Mbps the "
                  f"video needs. YouTube calls this videoIngestionStarved and "
                  f"viewers buffer. The upload is the limit — re-encode smaller "
                  f"(KAIZER_LIVE_ENCODE_MBPS) or use a faster connection.")
    elif not_live:
        detail = (f"{not_live} channel(s) are being pushed to and YouTube does not "
                  f"report them live. Check the broadcast still exists and is bound "
                  f"to the stream being pushed to, and that the file has an audio "
                  f"track — YouTube Live will not start ingest without one.")
    elif starting:
        detail = (f"{starting} channel(s) are not confirmed live yet: video is going "
                  f"out and YouTube has not flipped the broadcast over. Ours usually "
                  f"take about 20 seconds; YouTube sometimes takes minutes, and the "
                  f"engine keeps re-checking. Not a fault on its own.")
    elif confirmed < on:
        detail = "waiting for the single confirmation, a few seconds after ffmpeg starts"
    return _node(state, "YouTube", f"{on} channel{'s' if on != 1 else ''} on air",
                 metrics=[_m("on air", on), _m("confirmed live", confirmed),
                          _m("starting", starting), _m("not live", not_live),
                          _m("starved", starved)],
                 detail=detail)


def _uploads_node(db, counts: dict) -> dict:
    """What arrived today, from the database rather than a counter: a row is
    harder to lose than an increment."""
    try:
        from datetime import datetime, timedelta, timezone
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        started = (db.query(models.LiveStream)
                     .filter(models.LiveStream.created_at >= since).count())
    except Exception:
        started = None
    return _node("ok" if started else "idle", "Uploads", "to R2, direct from the browser",
                 metrics=[_m("live rows, 24 h", started),
                          _m("checked", counts.get("uploads_checked"))])


# ── usage: the questions the founder asked to be able to answer ──────

def _usage(db, live: list[dict], events: list[dict]) -> dict:
    """Who is using Live Studio, and how far one video reaches.

    `users_live_now` and `users_today` are different questions and both get
    asked. Someone who streamed this morning and stopped is using the product;
    they are not using it right now.
    """
    users_now, channels_now, fanout = set(), 0, []
    for v in live:
        on = [c for c in v.get("channels", []) if c.get("state") == "on"]
        queued = [c for c in v.get("channels", []) if c.get("state") == "queued"]
        if v.get("user_id"):
            users_now.add(str(v["user_id"]))
        channels_now += len(on)
        fanout.append({
            "video_id": v.get("video_id"),
            "user_id": v.get("user_id"),
            "channels_on": len(on),
            "channels_queued": len(queued),
            "confirmed_live": sum(c.get("youtube") == "live" for c in on),
            "worker": v.get("worker"),
            "started_at": v.get("started_at"),
        })
    fanout.sort(key=lambda f: (-f["channels_on"], str(f["video_id"])))

    users_today, starts_today = set(), 0
    day_ago = time.time() - 86400
    for e in events:
        if float(e.get("ts") or 0) < day_ago:
            continue
        if e.get("type") in ("channel_started", "channel_added"):
            starts_today += 1
            if e.get("user_id"):
                users_today.add(str(e["user_id"]))

    # Names, so the panel reads as people rather than as integers.
    names = {}
    if users_now or users_today:
        try:
            ids = [int(u) for u in (users_now | users_today) if str(u).isdigit()]
            for u in db.query(models.User).filter(models.User.id.in_(ids)).all():
                names[str(u.id)] = u.name or u.email or f"user {u.id}"
        except Exception:
            names = {}

    widest = fanout[0] if fanout else None
    return {
        "users_live_now": len(users_now),
        "users_today": len(users_today),
        "channels_live_now": channels_now,
        "videos_live_now": len(live),
        "broadcast_starts_24h": starts_today,
        "widest_fanout": widest["channels_on"] if widest else 0,
        "widest_fanout_video": widest["video_id"] if widest else None,
        "per_video": fanout,
        "user_names": names,
        "live_users": sorted(
            ({"user_id": u, "name": names.get(u, f"user {u}"),
              "videos": sum(1 for f in fanout if str(f["user_id"]) == u),
              "channels": sum(f["channels_on"] for f in fanout if str(f["user_id"]) == u)}
             for u in users_now),
            key=lambda x: -x["channels"]),
    }


def _channel_rows(db, live: list[dict]) -> list[dict]:
    """Every channel on air, with its own health. The map's detail table."""
    ids = {int(c["channel_id"]) for v in live for c in v.get("channels", [])
           if str(c.get("channel_id", "")).isdigit()}
    names = {}
    if ids:
        try:
            for ch in db.query(models.Channel).filter(models.Channel.id.in_(ids)).all():
                names[str(ch.id)] = ch.name or f"channel {ch.id}"
        except Exception:
            names = {}
    rows = []
    for v in live:
        for c in v.get("channels", []):
            h = c.get("health") or {}
            hs = h.get("state") or ""
            state = ("bad" if hs in _BAD or c.get("youtube") == "not_live"
                     else "warn" if hs in _WARN or c.get("state") == "queued"
                     else "ok" if c.get("state") == "on" else "idle")
            rows.append({
                "video_id": v.get("video_id"),
                "channel_id": c.get("channel_id"),
                "channel": names.get(str(c.get("channel_id")), f"channel {c.get('channel_id')}"),
                "state": state,
                "spec_state": c.get("state"),
                "health": hs,
                "youtube": c.get("youtube"),
                "kbps": float(h["kbps"]) if h.get("kbps") not in (None, "") else None,
                # What the SOURCE is producing, so a starved channel can be read
                # as the comparison it actually is rather than a bare number.
                "source_mbps": float(h["source_mbps"]) if h.get("source_mbps") not in (None, "") else None,
                "reconnects": int(h.get("reconnects") or 0),
                "credit": int(c.get("credit") or 0),
                "watch_url": c.get("watch_url") or "",
                "reason": c.get("reason") or "",
                "error": c.get("error") or "",
                # masked already, and kept because recognising which key is in
                # use is genuinely useful when a channel misbehaves
                "destination": c.get("dst_masked") or "",
            })
    rows.sort(key=lambda x: ({"bad": 0, "warn": 1, "ok": 2, "idle": 3}[x["state"]],
                             str(x["video_id"]), str(x["channel_id"])))
    return rows


# ── the whole map ────────────────────────────────────────────────────

def build_map(service, db, *, events: int = 40) -> dict:
    """Everything the Live Map tab draws, in one read.

    One call rather than nine, because the panel polls every two seconds and
    nine round trips at that rate is a self-inflicted load problem — and because
    parts of it read from each other: a channel's state depends on the health
    the worker wrote, which is only meaningful if the worker is alive.
    """
    from kaizer_live import counters

    r, prefix = service.r, service.s.prefix
    over = service.overview(events=events)
    counts = counters.today(r, prefix)

    # overview() gives a summary per live video; the map needs each one's
    # channels, so read them properly.
    live = []
    for row in over.get("live_videos", []):
        try:
            live.append(service.status(row["video_id"]))
        except Exception:
            continue

    nodes = {
        "uploads": _uploads_node(db, counts),
        "checker": _checker_node(counts),
        "encode": _encode_node(r, prefix, counts),
        "redis": _redis_node(r, prefix),
        "control": _control_node(r, prefix),
        "workers": _workers_node(service),
        "relays": _relays_node(r, prefix, live),
        "youtube": _youtube_node(live),
        "credit": _credit_node(service),
    }

    def ws(*names: str) -> str:
        return _worst(*(nodes[n]["state"] for n in names))

    # Wires carry the state of the weaker end: a link is only as good as what it
    # connects. "ctl" marks a control path rather than a path video travels, so
    # the drawing can dash it.
    wires = [
        {"from": "uploads", "to": "checker", "state": ws("uploads", "checker"),
         "label": f"{counts.get('uploads_checked') or 0} today"},
        {"from": "checker", "to": "encode", "state": ws("checker", "encode"),
         "label": f"{counts.get('encodes_queued') or 0} to fix"},
        {"from": "checker", "to": "redis", "state": ws("checker", "redis"),
         "label": "stream-ready"},
        {"from": "encode", "to": "redis", "state": ws("encode", "redis"), "label": "repaired"},
        {"from": "redis", "to": "control", "state": ws("redis", "control"), "kind": "ctl"},
        {"from": "redis", "to": "workers", "state": ws("redis", "workers"),
         "kind": "ctl", "label": "desired state"},
        {"from": "workers", "to": "relays", "state": ws("workers", "relays"),
         "label": f"{len(live)} video{'s' if len(live) != 1 else ''}"},
        {"from": "relays", "to": "youtube", "state": ws("relays", "youtube"),
         "label": f"{_metric(nodes['youtube'], 'on air') or 0} channels"},
        {"from": "control", "to": "youtube", "state": ws("control", "youtube"),
         "kind": "ctl", "label": "insert / bind / transition"},
        {"from": "credit", "to": "control", "state": ws("credit", "control"), "kind": "ctl"},
    ]

    payload = {
        "generated_at": time.time(),
        "prefix": prefix,
        "engine": "v2",
        "nodes": nodes,
        "wires": wires,
        "usage": _usage(db, live, over.get("events", [])),
        "channels": _channel_rows(db, live),
        "blocked_channels": over.get("blocked_channels") or {},
        "user_caps": over.get("user_caps") or {},
        "events": over.get("events") or [],
        "credit": over.get("credit") or {},
        "counters": counts,
        "worst": _worst(*(n["state"] for n in nodes.values())),
    }
    return _scrub(payload)
