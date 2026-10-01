"""Where kaizer_live meets KaizerBackend.

kaizer_live is deliberately ignorant of this application: it knows Redis, ffmpeg
and its own Go relay, and it takes its YouTube behaviour from three callables
handed in at construction. This module supplies them, so the engine stays
portable and every KaizerBackend-specific decision lives in one readable file.

The engine itself now lives in its own repository — E:/kaizer-dev/kaizer-live-stack,
installed with `pip install -e .` — so that the worker, the sweeper and the encode
queue can be deployed and restarted without touching this application. There is
exactly one copy of it; the vendored copy that used to sit in this directory was
removed, because two copies of a streaming engine drift and the drift is only
found on air.

THERE IS NO MANUAL STREAM-KEY MODE. Every channel goes live as an API broadcast
on its own reused stream key. Pushing to a pasted Studio key costs no quota, so
it looked like the answer, and it was tested properly: 90 seconds to a real
channel's persistent key, stream `active` throughout, and no broadcast ever
appeared. YouTube stopped auto-creating a broadcast for a persistent key on
1 September 2020, so a key now carries video that nobody can watch unless a
broadcast exists — and the only way to make one is insert (50) + bind (50).
100 units is the floor, and no arrangement of keys avoids it.
"""
from __future__ import annotations

from typing import Optional

import models


# ─── Which copy of the engine this deployment runs ───────────────────

def resolve_stack_dir() -> str:
    """The kaizer-live-stack directory this deployment should import from.

    DEV AND LIVE SHARE ONE VIRTUALENV ON THE BUILD MACHINE, and the engine is
    installed into it as an editable package — which is a pointer, and a venv
    holds exactly one per package name. Without this, `import kaizer_live` from
    the LIVE backend resolves to whichever folder was installed last, so a
    half-finished edit on DEV would be what a paying customer's broadcast runs
    on: no deploy, no restart, no promote, and nothing to see, because both
    sides import the same module name and neither says where from.

    Derived rather than merely read, for the same reason the Redis prefix is: an
    explicit KAIZER_LIVE_STACK wins, but when it is absent the answer comes from
    where THIS backend lives, so forgetting the variable cannot reach across.
    Returns "" when no local copy exists, and the installed package is then used
    — which is the right answer on a standalone deployment that has only one.
    """
    import os

    explicit = (os.getenv("KAIZER_LIVE_STACK") or "").strip()
    if explicit:
        return explicit if os.path.isdir(explicit) else ""

    here = os.path.dirname(os.path.abspath(__file__))          # .../KaizerBackend
    project = os.path.dirname(here)                            # the project root
    for cand in (os.path.join(project, "kaizer-live-stack"),
                 os.path.join(os.path.dirname(project), "kaizer-live-stack")):
        if os.path.isdir(os.path.join(cand, "kaizer_live")):
            return cand
    return ""


def _use_local_stack() -> str:
    """Put this deployment's own engine first on sys.path, and say which it is.

    Announced once at import, because a wrong answer here cannot be detected at
    runtime — the module imports cleanly either way — so the only place it can
    be caught is a line somebody reads after a deploy.
    """
    import sys

    d = resolve_stack_dir()
    if d and d not in sys.path:
        sys.path.insert(0, d)
    return d


_STACK_DIR = _use_local_stack()

# ─── The connected provider ──────────────────────────────────────────
#
# kaizer_live's ConnectedProvider takes two callables and knows nothing about
# this application. These are them.
#
# TWO THINGS HERE ARE LOAD-BEARING AND MUST NOT BE "TIDIED UP":
#
#   enable_auto_stop=False. Looping a copy-stream leaves a micro-gap at each
#   loop boundary, and YouTube's auto-stop reads those gaps as the end of the
#   broadcast — an operator reported a 9-hour stream closing early before this
#   was set. We own the duration instead, and close the broadcast ourselves.
#
#   Because auto-stop is off, finalize_broadcast's transition(complete) is the
#   ONLY thing that ever ends a Live Studio broadcast. It is not redundant, and
#   with a reused per-channel stream key an un-closed broadcast holds that key
#   and blocks the channel's next broadcast.

def _yt_timeout() -> float:
    """Seconds any single YouTube call may take before it is abandoned.

    These calls happen inside an HTTP request that has already spent the start
    credit and claimed the channel, so a hung Google connection is a customer
    watching a spinner while their quota is gone. The transport's own default is
    120 s and is shared with the uploader, where minutes are legitimate; this is
    the live path's own, much shorter, limit.
    """
    try:
        return float(build_settings().yt_timeout_s)
    except Exception:
        return 10.0


def make_connected_provider(session_factory):
    """Build kaizer_live's ConnectedProvider over KaizerBackend's YouTube code."""
    from kaizer_live.providers import ConnectedProvider, ProviderError, is_permanent

    def _load(db, channel_id: str):
        ch = db.query(models.Channel).get(int(channel_id))
        if ch is None:
            raise RuntimeError(f"channel {channel_id} not found")
        from youtube import oauth as yt_oauth
        return ch, yt_oauth.get_credentials(db, ch.id)

    def start_fn(channel_id: str, meta: dict) -> dict:
        from youtube import rtmp_provider as yt_rtmp
        # A PERMANENT FAILURE MUST BE RAISED AS ONE -- see the except clause at
        # the end. Live streaming not enabled on the channel, a revoked token, a
        # suspended channel: each fails identically every time it is tried, and
        # every attempt spends the start credit before finding out. Marked
        # permanent, the engine blocks the channel after the first failure and
        # refuses the second for free. Left unmarked, the same customer pays 103
        # units per retry for something that cannot change until they go and fix
        # it in YouTube.
        db = session_factory()
        try:
            ch, creds = _load(db, channel_id)

            # REUSE THE EXISTING ROW. Live Studio already created one per
            # (video, channel) -- it is the durable history the admin tab
            # reads -- and start_through_engine was called with it. Creating a
            # second one here made a duplicate AND crashed: batch_id is an
            # INTEGER foreign key to live_batches.id, and the engine's video
            # id is "<batch>-<slot>", so Postgres refused "48-0".
            #
            # The engine's video id carries the batch and slot, so the row can
            # be found from what the provider is given.
            row = None
            vid = str(meta.get("video_id") or "")
            if "-" in vid:
                b, _, slot = vid.rpartition("-")
                try:
                    row = (db.query(models.LiveStream)
                             .filter(models.LiveStream.batch_id == int(b),
                                     models.LiveStream.video_slot == int(slot),
                                     models.LiveStream.channel_id == ch.id)
                             .order_by(models.LiveStream.id.desc()).first())
                except (TypeError, ValueError):
                    row = None

            if row is None:
                # The engine was driven directly through its own API rather
                # than by Live Studio, so there is no row yet. batch_id stays
                # NULL: it is a foreign key, and inventing a value would point
                # at a batch that does not exist.
                row = models.LiveStream(
                    user_id=meta.get("user_id") or ch.user_id,
                    channel_id=ch.id,
                    video_slot=0,
                    status="provisioning",
                    title=(meta.get("title") or "Live broadcast")[:200],
                    description=(meta.get("description") or "")[:5000],
                    privacy=(meta.get("privacy") or "unlisted"),
                    target_hours=float(meta.get("hours") or 1.0),
                )
                db.add(row)
                db.commit()

            target = yt_rtmp.obtain_rtmp_target(
                creds=creds,
                job=row,
                channel=ch,
                title=row.title,
                description=row.description or "",
                privacy_status=row.privacy or "unlisted",
                enable_auto_stop=False,     # see the note above — do not change
                timeout_s=_yt_timeout(),
                db=db,
            )

            # Remember the channel's reusable stream so the next broadcast
            # costs 1 unit to look up instead of 50 to mint.
            sid = target.get("stream_id") or ""
            if sid and getattr(ch, "yt_stream_id", None) != sid:
                ch.yt_stream_id = sid

            row.yt_broadcast_id = target.get("broadcast_id") or ""
            row.yt_stream_id = sid
            row.yt_video_id = target.get("video_id") or target.get("broadcast_id") or ""
            row.status = "streaming"
            db.commit()

            vid = row.yt_video_id
            return {
                "ingest_url": target.get("ingest_url") or "",
                "stream_key": target.get("stream_key") or "",
                "broadcast_id": target.get("broadcast_id") or "",
                "watch_url": f"https://www.youtube.com/watch?v={vid}" if vid else "",
                "live_stream_row_id": row.id,
            }
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"YouTube broadcast start failed: {exc}",
                                permanent=is_permanent(exc)) from exc
        finally:
            db.close()

    def _record_delivery(row, channel_id: str) -> None:
        """Copy the engine's byte counters onto the row, once, at the end.

        `reader.in_bytes` is what ffmpeg produced; an output's `bytes` is what
        was actually written to YouTube. Both are cumulative and both are
        already in Redis -- and both are deleted by the sweep moments after
        this. Kept here, a starved broadcast says so on its own row forever
        instead of only to whoever happened to be polling at the time.

        The engine's video id comes from the ROW (batch_id + video_slot -- the
        "75-0" in the logs), not from the target: the sweep builds its target as
        Target("", "", broadcast_id, watch_url) and it carries no video id.

        Best-effort by construction: a broadcast that ended correctly must never
        be recorded as failed because its statistics could not be read. That
        also means a mistake in here is SILENT, so every name it touches is one
        the test below proves exists.
        """
        import json as _json
        try:
            vid = engine_video_id(str(row.batch_id or ""), int(row.video_slot or 0))
            svc = get_live_service()
            if svc is None or not vid:
                return
            # svc.k is the service's own Keys instance, already built on its
            # prefix -- so this cannot read the wrong deployment's keys.
            r, k = svc.r, svc.k
            rd = r.hgetall(k.health_reader(vid)) or {}
            ch = r.hgetall(k.health(vid, str(channel_id))) or {}
            if not rd and not ch:
                return                      # nothing to keep; NULL says so

            def _i(d, f):
                try:
                    return int(float(d.get(f) or 0))
                except (TypeError, ValueError):
                    return 0

            produced = _i(rd, "in_bytes")
            delivered = _i(ch, "bytes")
            rec = {
                "source_bytes": produced,
                "delivered_bytes": delivered,
                # None, not 0: with no source reading there is no ratio to
                # report, and 0 would read as "nothing got through".
                "delivered_pct": (round(delivered / produced * 100, 1)
                                  if produced > 0 else None),
                "dropped": _i(ch, "dropped"),
                "reconnects": _i(ch, "reconnects"),
                "last_kbps": _i(ch, "kbps"),
                "source_mbps": (float(ch.get("source_mbps") or 0) or None),
                "engine_video_id": vid,
            }
            row.delivery_json = _json.dumps(rec)
        except Exception:
            # Never let statistics fail a finished broadcast.
            pass

    def end_fn(channel_id: str, target) -> None:
        from youtube import rtmp_provider as yt_rtmp
        db = session_factory()
        try:
            ch, creds = _load(db, channel_id)
            bid = getattr(target, "broadcast_id", "") or ""
            if not bid:
                return
            row = (db.query(models.LiveStream)
                     .filter(models.LiveStream.yt_broadcast_id == bid)
                     .order_by(models.LiveStream.id.desc()).first())
            # confirm_active=False: the engine already confirmed this
            # broadcast went live, once, moments after ffmpeg started flowing.
            # Polling again here would pay up to 15 more units for an answer it
            # has -- and it would run just after the push was stopped, where the
            # not-active branch DELETES the broadcast. For a stream that ran for
            # hours, that deletes the recording.
            yt_rtmp.finalize_broadcast(
                creds=creds, job=row, channel=ch, broadcast_id=bid,
                confirm_active=False, timeout_s=_yt_timeout(), db=db)
            if row is not None and row.status not in ("failed", "canceled"):
                from datetime import datetime, timezone
                row.status = "done"
                row.finished_at = datetime.now(timezone.utc)
                # Keep the delivery counters BEFORE the sweep deletes them.
                # This is the only moment they exist and the row is in hand.
                _record_delivery(row, channel_id)
                db.commit()
        finally:
            db.close()

    def check_fn(channel_id: str, broadcast_id: str) -> str:
        """YouTube's lifeCycleStatus for one broadcast. ONE unit.

        This is the whole of the engine's confirmation, and the reason the start
        path no longer polls: instead of asking every two seconds for thirty
        seconds while starting (15 units), it waits until ffmpeg has actually
        been flowing and asks once. Two are prepaid in the start cost.

        Returns "" if it could not be read. The engine treats that as
        "unchecked" rather than "not live" -- a network blip must not be
        recorded as a broadcast that never started.
        """
        from youtube import rtmp_provider as yt_rtmp
        db = session_factory()
        try:
            ch, creds = _load(db, channel_id)
            row = (db.query(models.LiveStream)
                     .filter(models.LiveStream.yt_broadcast_id == broadcast_id)
                     .order_by(models.LiveStream.id.desc()).first())
            return yt_rtmp.broadcast_lifecycle(
                creds, broadcast_id, job=row, channel=ch,
                timeout_s=_yt_timeout(), db=db)
        except Exception as exc:
            print(f"[kaizer_live] could not read broadcast {broadcast_id}: {exc}", flush=True)
            return ""
        finally:
            db.close()

    return ConnectedProvider(start_fn, end_fn, check_fn)


# ─── Settings, and the assembled service ─────────────────────────────

def derive_prefix(explicit: Optional[str], database_url: Optional[str]) -> str:
    """Which Redis key prefix this deployment owns.

    DEV and LIVE run against the same Redis server, so the prefix is the only
    thing keeping them apart: sharing one would let a DEV sweep end a paying
    customer's broadcast.

    DERIVED, not merely read. An explicit KAIZER_LIVE_PREFIX wins, because a
    deployment may legitimately want its own. But when it is absent the database
    name decides, so FORGETTING the variable on DEV cannot reach LIVE -- the
    failure mode of a plain `os.getenv(..., "kl")` is that every misconfigured
    box defaults to the production prefix.

    A pure function on purpose. The rule is worth testing, and testing it
    through the environment is unreliable: importing the application's config
    runs load_dotenv(), which puts .env's values back into os.environ, so a test
    cannot express "pretend this is unset" by deleting it.
    """
    explicit = (explicit or "").strip()
    if explicit:
        return explicit
    db = (database_url or "").lower()
    return "kldev" if ("kaizer_dev" in db or "_dev" in db) else "kl"


def build_settings():
    """kaizer_live Settings, filled from KaizerBackend's own configuration.

    THE PREFIX IS DERIVED, NOT JUST READ. DEV and LIVE run against the same
    Redis server, and sharing a key prefix would let a DEV sweep end a LIVE
    customer's broadcast. KAIZER_LIVE_PREFIX wins when set; otherwise the
    database name decides, so forgetting the variable on DEV cannot reach LIVE.
    """
    import os

    from kaizer_live.config import Settings

    # LOAD THE APPLICATION'S CONFIGURATION FIRST, before reading a single
    # variable. Importing `config` runs load_dotenv() at its module level, so
    # doing it later — as the Fernet lookup below used to — meant the first call
    # to this function read a half-loaded environment and every later call read
    # a different one. The prefix is what keeps DEV and LIVE apart in a shared
    # Redis, so a prefix that depends on call order is a DEV sweep that can end
    # a paying customer's broadcast, once, unreproducibly.
    try:
        from config import settings as _app_settings
    except Exception:
        _app_settings = None

    prefix = derive_prefix(os.getenv("KAIZER_LIVE_PREFIX"), os.getenv("DATABASE_URL"))

    redis_url = (os.getenv("KAIZER_LIVE_REDIS")
                 or os.getenv("REDIS_URL")
                 or "redis://127.0.0.1:6379/0")

    # The same Fernet key crypto.py uses, so destinations encrypted in Redis
    # open with the secret this deployment already manages.
    fernet = (os.getenv("KAIZER_LIVE_FERNET_KEY") or "").strip()
    if not fernet and _app_settings is not None:
        # The same key crypto.py uses. A second secret would be a second thing
        # to lose, and the worker could not open what the API sealed.
        try:
            fernet = _app_settings.encryption_key or ""
        except Exception:
            fernet = ""

    # Resolve ffmpeg rather than trusting the child's PATH. Production learned
    # this the hard way with yt-dlp: a bare name resolves only if the venv's
    # bin directory reaches the subprocess, which on a buildpack deploy it
    # does not.
    # AN EXPLICIT SETTING WINS. This used to take live_studio.streamer's
    # _FFMPEG_BIN first and consult the environment only if that import FAILED.
    # The import does not fail -- it succeeds and returns the bare name
    # "ffmpeg" -- so KAIZER_LIVE_FFMPEG was read by nothing, and the engine
    # refused to start telling the operator to set the variable it was ignoring.
    import shutil
    _ff = (os.getenv("KAIZER_LIVE_FFMPEG") or "").strip()
    if not _ff:
        try:
            from live_studio.streamer import _FFMPEG_BIN as _ff
        except Exception:
            _ff = "ffmpeg"
    # A bare name is only useful where it resolves, and on a buildpack deploy
    # the venv's bin directory does not reach a subprocess. Resolve it now so a
    # missing ffmpeg is a startup message rather than a dead broadcast.
    if _ff and not os.path.isfile(_ff):
        _ff = shutil.which(_ff) or _ff

    ffprobe = (os.getenv("KAIZER_LIVE_FFPROBE") or "").strip()
    if not ffprobe:
        # Prefer the ffprobe beside the ffmpeg we settled on: a mismatched pair
        # from two builds can disagree about what a file contains.
        cand = ""
        low = _ff.lower()
        if low.endswith("ffmpeg.exe"):
            cand = _ff[: -len("ffmpeg.exe")] + "ffprobe.exe"
        elif low.endswith("ffmpeg"):
            cand = _ff[: -len("ffmpeg")] + "ffprobe"
        ffprobe = cand if (cand and os.path.isfile(cand)) else (shutil.which("ffprobe") or "ffprobe")

    return Settings(
        redis_url=redis_url,
        prefix=prefix,
        daily_credit=int(os.getenv("KAIZER_LIVE_DAILY_CREDIT", "10000")),
        ffmpeg=_ff,
        ffprobe=ffprobe,
        fernet_key=fernet,
    )


_SERVICE = None


def get_live_service():
    """The one LiveService for this process, built on first use.

    Returns None when the engine is switched off or its dependencies are
    missing, so main.py can mount nothing and the old Live Studio path keeps
    working untouched.
    """
    global _SERVICE
    if _SERVICE is not None:
        return _SERVICE
    import os
    if (os.getenv("KAIZER_LIVE_ENGINE") or "").strip().lower() != "v2":
        return None
    try:
        import redis as _redis
        from database import SessionLocal
        from kaizer_live.service import LiveService

        s = build_settings()
        if not s.fernet_key:
            print("[kaizer_live] no Fernet key available; engine not started", flush=True)
            return None
        # ffmpeg must be RUNNABLE, not merely named. A bare "ffmpeg" resolves
        # only if the venv/bin directory reaches the subprocess, which on a
        # buildpack deploy it does not -- production lost a day to exactly this
        # with yt-dlp. Fail at startup, where the message is read, rather than
        # at the first broadcast, where it is not.
        import subprocess as _sp
        for _name, _bin in (("ffmpeg", s.ffmpeg), ("ffprobe", s.ffprobe)):
            try:
                _sp.run([_bin, "-version"], stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
                        timeout=10, check=True)
            except Exception as _exc:
                print(f"[kaizer_live] {_name} is not runnable as {_bin!r} "
                      f"({type(_exc).__name__}); engine not started. Put it on PATH "
                      f"or set KAIZER_LIVE_FFMPEG / KAIZER_LIVE_FFPROBE.", flush=True)
                return None

        r = _redis.Redis.from_url(s.redis_url, decode_responses=True)
        r.ping()
        # One provider, because there is one way to go live: an API broadcast on
        # the channel's reused stream key. No key store reaches the service --
        # the only stream key in the system is the one YouTube issues, and it
        # goes straight into the Fernet-sealed destination in Redis.
        _SERVICE = LiveService(r, s, make_connected_provider(SessionLocal))
        # ONLY A STREAM-READY FILE MAY GO LIVE. Wrapped here rather than
        # inside the vendored engine, which stays portable and knows nothing
        # about R2 or this application. Going live spends quota BEFORE any
        # video moves, so a file YouTube will reject costs real units and puts
        # a broken stream in front of a customer's audience. One ffprobe on
        # the URL is cheap; the broadcast that fails is not.
        _inner_go_live = _SERVICE.go_live

        def _checked_go_live(video_id, user_id, source, channels, **kw):
            assert_stream_ready(source)
            return _inner_go_live(video_id, user_id, source, channels, **kw)

        _SERVICE.go_live = _checked_go_live

        import kaizer_live as _pkg
        _from = os.path.dirname(os.path.dirname(os.path.abspath(_pkg.__file__)))
        print(f"[kaizer_live] engine v2 ready (prefix={s.prefix}, "
              f"credit={s.daily_credit}, relay={s.relay_bin}, "
              f"{s.costs.broadcast_total()} units/broadcast)", flush=True)
        # WHICH COPY. Sharing a venv between deployments makes this the one
        # fact that decides whose code a broadcast runs, and the only place it
        # can be checked is here.
        print(f"[kaizer_live] engine loaded from {_from}", flush=True)
        return _SERVICE
    except Exception as exc:
        # Never take the API down because the live engine cannot start; the
        # old path still serves every existing customer.
        print(f"[kaizer_live] engine unavailable: {type(exc).__name__}: {exc}", flush=True)
        return None


def owns_channel(user, channel_id: str) -> bool:
    """Ownership check handed to kaizer_live's router."""
    from database import SessionLocal
    db = SessionLocal()
    try:
        ch = db.query(models.Channel).get(int(channel_id))
        return bool(ch and ch.user_id == getattr(user, "id", None))
    except (TypeError, ValueError):
        return False
    finally:
        db.close()


# ─── The source the reader pulls from ────────────────────────────────

# SigV4 caps a presigned URL at seven days. The reader ffmpeg re-opens its
# source on every reconnect and on every loop pass, so a URL that expires
# mid-broadcast stops the video dead — with no way to recover, because the
# worker has only the URL, not the object key.
#
# So: sign for the full seven days, and refuse to accept a duration that could
# outlive it. Six days leaves a day of margin for a broadcast that starts late
# or a worker that takes the job over from a dead box.
#
# THE PROPER FIX, when this matters: give the worker the object key and let it
# presign for itself, so a long broadcast re-signs as it goes instead of
# carrying one URL to its grave. That needs R2 credentials on the worker box,
# which is a deployment decision, not a code one.
PRESIGN_TTL_S = 7 * 24 * 3600
MAX_LIVE_HOURS = 6 * 24          # 144h — one day inside the signature's life


def presign_live_source(key: str, *, expires_s: int = PRESIGN_TTL_S) -> str:
    """A presigned GET URL for a stream-ready object, for kaizer_live's reader."""
    # get_storage_provider, not get_provider -- the latter does not exist.
    from pipeline_core.storage import get_storage_provider
    return get_storage_provider().get_url(key, signed=True, expires_s=expires_s)


def clamp_live_hours(hours: float) -> tuple[float, str]:
    """Hold the requested duration inside the signed URL's lifetime.

    Returns (hours, note). The note is empty when nothing was clamped, and
    otherwise says plainly what happened — a broadcast that quietly runs for
    less time than asked is a support ticket nobody can explain.
    """
    try:
        h = float(hours or 0)
    except (TypeError, ValueError):
        h = 0.0
    if h <= 0:
        return 1.0, ""
    if h > MAX_LIVE_HOURS:
        return (float(MAX_LIVE_HOURS),
                f"requested {h:.0f}h, capped at {MAX_LIVE_HOURS}h: the source URL "
                f"is signed for 7 days and the reader re-opens it on every loop")
    return h, ""


# ─── Making a file safe to stream ────────────────────────────────────
#
# The live path copies bytes; it never re-encodes. That is what makes one
# broadcast cost 13% of a CPU core instead of a whole one. The price is that
# the file has to be right BEFORE it goes live: wrong codec, a moov atom at
# the end, a variable frame rate or keyframes too far apart and YouTube either
# refuses the stream or plays it badly, in front of the customer's audience.
#
# So the check runs once, when the upload lands, and any fix runs once, in an
# encode job. Nothing here may ever be called from the live path.

def check_source(source: str, *, count: bool = True) -> dict:
    """ffprobe verdict for a file or URL: none / remux / audio / reencode.

    Counted, because nothing else records a file that passed. A file that is
    already stream-ready goes straight to R2 and leaves no trace anywhere, so
    without this the admin map's Checker node could only ever show zero -- and
    "uploads are arriving and none of them pass" is a real failure with no other
    symptom.

    `count=False` for a re-check of something this code just produced: counting
    those would report two files checked for every one a customer sent.
    """
    from kaizer_live import checker, counters
    s = build_settings()
    verdict = checker.check(s.ffprobe, source,
                            link_kbps=s.deliverable_kbps).as_dict()
    if count:
        try:
            r = get_live_redis()
            if r is not None:
                counters.bump(r, s.prefix, "uploads_checked")
                if verdict.get("stream_ready"):
                    counters.bump(r, s.prefix, "uploads_passed")
                # "encodes_queued" is NOT bumped here. encode.enqueue() owns it,
                # and a counter with two writers counts a thing twice: a verdict
                # of "needs repair" is not the same event as a job being queued,
                # and a file checked twice (a retry, a re-check) would inflate it
                # further. One event, one writer.
        except Exception:
            pass          # a counter is never worth failing a broadcast for
    return verdict


def get_live_redis():
    """The live stack's Redis, or None. Never raises: callers are on the path of
    a broadcast, and a metric must not be able to stop one."""
    try:
        import redis as _redis
        r = _redis.Redis.from_url(build_settings().redis_url, decode_responses=True)
        r.ping()
        return r
    except Exception:
        return None


# normalize_for_live() and ensure_stream_ready() used to live here.
#
# They repaired a file in whatever process called them, which meant minutes of
# ffmpeg competing with that process's real work, and they needed storage
# credentials wherever they ran. Nothing called ensure_stream_ready in the end,
# and normalize_for_live existed only to be called from it.
#
# The encode queue replaced both: prepare_source() enqueues the repair and waits
# for it, and kaizer_live.encode does the work in its own process -- one job at
# a time, at below-normal priority, on a machine that can be a different one,
# holding no credentials because its job carries a presigned GET and a presigned
# PUT. Two implementations of "how to repair a video for live" would drift, and
# the drift would only be discovered on air.


class NotStreamReady(RuntimeError):
    """Raised instead of going live with a file YouTube would reject."""


def _is_local_path(source: str) -> bool:
    """A path on the worker's own disk, rather than something fetched."""
    s = (source or "").strip()
    return bool(s) and not s.startswith(("http://", "https://", "rtmp://", "rtmps://"))


def assert_stream_ready(source: str) -> dict:
    """Gate the live path. Cheap: one ffprobe, no transfer of the whole file.

    Deliberately a hard stop rather than a warning. Starting a broadcast spends
    quota BEFORE any video moves, so letting a doomed file through costs real
    units and puts a broken stream on a customer's channel.
    """
    v = check_source(source)
    if v.get("stream_ready"):
        return v

    # A LOCAL FILE MAY KEEP ITS INDEX AT THE END. The checker flags that as
    # needing a remux, which is correct for an HTTP source -- ffmpeg would have
    # to pull the whole object before the first frame -- and wrong for a file on
    # the worker's own disk, which it simply seeks. Most MP4s written by any
    # ordinary encoder put moov last, so treating this as fatal would refuse
    # almost every real upload.
    reasons = list(v.get("reasons") or [])
    if _is_local_path(source) and v.get("action") == "remux":
        moov_only = all("moov" in r or "faststart" in r or "index" in r
                        for r in reasons) and bool(reasons)
        if moov_only:
            out = dict(v)
            out["stream_ready"] = True
            out["warnings"] = list(out.get("warnings") or []) + [
                "index (moov) is at the end; fine for a local file, would need "
                "a remux before this could be streamed from a URL"]
            return out

    raise NotStreamReady(
        f"this file needs {v.get('action')} before it can be streamed: "
        + "; ".join(reasons or ["no reason given"]))


# ─── Starting a broadcast through the engine ─────────────────────────

def engine_video_id(batch_id: str, video_slot: int) -> str:
    """One engine video per (batch, slot).

    It becomes a Redis key segment AND an RTMP path (rtmp://…/v/<id>), and is
    truncated to 64 chars when stored as LiveStream.batch_id -- so it stays
    short and has nothing in it that needs escaping.
    """
    safe = "".join(c for c in str(batch_id or "") if c.isalnum() or c in "-_")[:40]
    return f"{safe}-{int(video_slot)}"


def live_broadcasts_on_channel(db, channel_id: int) -> list:
    """Non-terminal LiveStream rows for a channel: what is on air right now.

    Used to refuse anything that would take away what a running broadcast needs
    to be ENDED -- its OAuth credentials, or the channel row itself. Ending is
    the part that matters: with enable_auto_stop deliberately off,
    transition(complete) is the only thing that ever closes a broadcast, and it
    needs those credentials. Revoke them mid-broadcast and it can never be
    closed, so the channel's reused stream key is held for ever and its 50-unit
    close reserve is re-reserved every day.
    """
    import models as _m
    TERMINAL = ("done", "failed", "canceled")
    return (db.query(_m.LiveStream)
              .filter(_m.LiveStream.channel_id == channel_id,
                      ~_m.LiveStream.status.in_(TERMINAL))
              .order_by(_m.LiveStream.id.desc())
              .all())


def engine_owns_row(row) -> str:
    """The engine's state for this row's channel, or "" if the engine has none.

    ASK THIS BEFORE MARKING ANY LIVE ROW DEAD. The engine's relays run in their
    own process, so a broadcast surviving an API restart is not an orphan -- it
    is the entire point of the split. Code that assumes "the backend restarted,
    therefore the ffmpeg is gone" is right about the classic path and wrong
    about this one, and acting on it marks a streaming broadcast failed while
    it goes on reaching the customer's audience. A terminal row then disarms the
    Stop button, so nobody can end it either.

    Returns one of the engine's channel states -- "on", "queued", "stopping",
    "end_pending", "ended", "failed" -- or "" when this row is not the
    engine's (a classic broadcast, the engine switched off, or never started).

    Never raises: every caller is in a cleanup path, and a cleanup that dies
    because Redis blinked is worse than one that is slightly conservative. On
    any error it answers "unknown", which callers must treat as "leave it
    alone" -- the safe direction is always to leave a possibly-live broadcast
    running rather than to abandon a definitely-live one.
    """
    try:
        svc = get_live_service()
        if svc is None or not getattr(row, "channel_id", None):
            return ""
        vid = engine_video_id(row.batch_id, row.video_slot)
        st = svc.status(vid)
    except Exception as exc:
        from kaizer_live.service import LiveError
        if isinstance(exc, LiveError):
            return ""                      # the engine genuinely has no such video
        print(f"[live_studio] could not ask the engine about stream "
              f"{getattr(row, 'id', '?')}: {exc}", flush=True)
        return "unknown"
    for c in st.get("channels", []):
        if str(c.get("channel_id")) == str(row.channel_id):
            return str(c.get("state") or "")
    return ""


#: States in which the engine is still carrying, or about to carry, this
#: broadcast. "unknown" is included deliberately: when we could not ask, the
#: safe answer is to leave it running.
ENGINE_LIVE_STATES = ("on", "queued", "stopping", "end_pending", "unknown")


def engine_is_carrying(row) -> bool:
    """True when this row must NOT be marked dead or restarted by cleanup code."""
    return engine_owns_row(row) in ENGINE_LIVE_STATES


def stop_row_on_engine(row, *, reason: str = "stopped") -> str:
    """Stop this row's channel on the engine. Returns a note, "" if not ours.

    Raises if the engine is running and could not be told -- callers must not
    report a stop that did not happen.
    """
    svc = get_live_service()
    if svc is None or not getattr(row, "channel_id", None):
        return ""
    from kaizer_live.service import LiveError
    vid = engine_video_id(row.batch_id, row.video_slot)
    try:
        out = svc.stop_channel(vid, str(row.channel_id), reason=reason)
    except LiveError:
        return ""                          # classic, or never started
    return f"engine stopped {vid}/{row.channel_id} ({out.get('state')})"


def start_through_engine(db, row) -> dict:
    """Start (or join) an engine broadcast for one LiveStream row.

    The classic path is one row per (video, channel); the engine is one video
    with many channels. So the first channel of a slot opens the video and the
    rest are added to it — which is also what makes "add a channel while it is
    live" work at all.

    The source is the absolute upload path. The browser never sees it and must
    not: the frontend has no business knowing where the worker's disk is.
    """
    from kaizer_live.service import ChannelRequest, LiveError

    svc = get_live_service()
    if svc is None:
        raise RuntimeError("live engine is not running")

    vid = engine_video_id(row.batch_id, row.video_slot)
    src = (row.source_url or "").strip() or (row.upload_path or "").strip()
    if not src:
        raise RuntimeError("nothing to stream: neither an upload nor a source url")

    hours, note = clamp_live_hours(row.target_hours)
    # HAS THIS CHANNEL GOT A STREAM TO REUSE? The engine cannot know -- it has
    # no database -- and the answer is worth 49 units: liveStreams.list costs 1,
    # liveStreams.insert costs 50. Measured on the first production broadcast,
    # which Google billed 151 while the ledger recorded 103.
    #
    # The daily limit is enforced against the ledger, so under-counting lets the
    # engine keep admitting starts after the real quota is gone -- and that
    # failure arrives as a 403 from Google with the panel still showing credit.
    _ch = db.query(models.Channel).get(int(row.channel_id)) if row.channel_id else None
    _reuse = bool(getattr(_ch, "yt_stream_id", "") or "")

    # No mode argument: every channel is an API broadcast on its reused key.
    req = ChannelRequest(
        str(row.channel_id),
        (row.title or "Live broadcast")[:100],
        (row.description or "")[:5000],
        (row.privacy or "unlisted"),
        bool((row.thumbnail_path or "").strip()),
        {"reuse_key": _reuse},
    )

    try:
        svc.status(vid)                      # already running: join it
        out = svc.add_channel(vid, req)
    except LiveError:
        out = svc.go_live(vid, str(row.user_id or ""), src, [req],
                          duration_s=hours * 3600.0, loop=True)
    if note:
        out = dict(out)
        out["note"] = note
    return out


# ─── Preparing a file that cannot be streamed as it is ───────────────

def ready_path_for(src: str) -> str:
    """Where the stream-ready copy of a source lives.

    Deterministic and beside the original, so a retry finds the copy made last
    time instead of spending minutes re-encoding the same file again.
    """
    import os
    base, _ext = os.path.splitext(src)
    return base + ".ready.mp4"


#: How long to wait for the encode queue before giving up on one file. A long
#: video is genuinely tens of minutes of CPU, and the alternative to waiting is
#: telling a customer their upload failed when it was only slow.
ENCODE_WAIT_S = 3 * 3600


def prepare_source(src: str, on_progress=None, *, stream_id: int = 0) -> tuple:
    """Return (path_to_stream, note). Repairs the file only if it has to.

    THE REPAIR HAPPENS IN THE ENCODE QUEUE, NOT HERE. A re-encode is minutes at
    full tilt; doing it in this process would compete with every request it
    serves, and doing it beside the relays would compete with the streams --
    which need very little CPU but need it on time, or every viewer on every
    channel sees a stutter. The queue exists so that work can be moved to a
    different machine.

    Still blocking, and still called on a background thread: what it blocks on
    now is another process finishing, not ffmpeg running here.
    """
    import os
    import time
    import uuid as _uuid

    from kaizer_live import encode as _encode

    v = check_source(src)
    if v.get("stream_ready"):
        return src, ""

    ready = ready_path_for(src)
    if os.path.isfile(ready) and os.path.getsize(ready) > 0:
        # count=False: this is a file this code produced, not one a customer sent.
        after = check_source(ready, count=False)
        if after.get("stream_ready"):
            return ready, "used the prepared copy from an earlier attempt"

    action = v.get("action") or "reencode"
    why = "; ".join(v.get("reasons") or [])

    r = get_live_redis()
    if r is None:
        raise NotStreamReady(
            f"this file cannot be streamed as it is ({why}) and the encode queue "
            f"is unreachable, so it cannot be repaired. Check that the live "
            f"stack's Redis is running.")

    s = build_settings()
    job_id = f"live-{stream_id or 'x'}-{_uuid.uuid4().hex[:8]}"
    _encode.enqueue(r, s.prefix, job_id, source=src, action=action,
                    out_path=ready, reasons=why, video_id=str(stream_id or ""))
    if on_progress:
        on_progress(f"queued for repair ({action}): {why}"[:512])

    deadline = time.time() + ENCODE_WAIT_S
    last = ""
    while time.time() < deadline:
        rec = _encode.job(r, s.prefix, job_id)
        state = rec.get("state", "")
        if state == "done":
            out = rec.get("ready_path") or ready
            return out, f"repaired for live ({action}): {why}"
        if state == "failed":
            raise NotStreamReady(f"this file could not be repaired: {rec.get('error', 'no reason given')}")
        if state != last and on_progress:
            depth = _encode.queue_depth(r, s.prefix)
            workers = len(_encode.encode_workers(r, s.prefix))
            if state == "queued" and not workers:
                # Say so rather than sitting silently: nothing is wrong with the
                # file, there is simply nobody to repair it.
                on_progress(f"waiting for an encode worker ({depth} file(s) queued, none running)")
            else:
                on_progress(f"repairing ({state}; {depth} queued)"[:512])
            last = state
        time.sleep(2.0)

    raise NotStreamReady(
        f"the file was still being repaired after {ENCODE_WAIT_S // 3600} hours. "
        f"Check that an encode worker is running: python -m kaizer_live.encode --id <name>")


def prepare_and_start(stream_id: int) -> None:
    """Prepare the file if needed, then start the broadcast. Runs on a thread.

    Owns its own session: the request that spawned it has long since returned,
    and its session is closed.
    """
    from datetime import datetime, timezone
    from database import SessionLocal

    db = SessionLocal()
    try:
        row = db.query(models.LiveStream).get(int(stream_id))
        if row is None:
            return

        def _note(msg: str) -> None:
            row.message = msg
            db.commit()

        src = (row.source_url or "").strip() or (row.upload_path or "").strip()
        try:
            ready, note = prepare_source(src, on_progress=_note, stream_id=row.id)
        except Exception as exc:
            row.status = "failed"
            row.error = f"could not prepare the file: {exc}"[:2000]
            row.message = f"failed: could not prepare the file: {exc}"[:512]
            row.finished_at = datetime.now(timezone.utc)
            db.commit()
            print(f"[live_studio] STREAM {row.id} -> failed | preparing "
                  f"| err={' '.join(str(exc).split())[:2000]}", flush=True)
            return

        # DID THE OPERATOR CANCEL WHILE WE WERE PREPARING? Repairing a long
        # video is minutes, and this runs on a background thread for all of
        # them. Without this re-read, pressing Stop during that window marked
        # the row cancelled and then the broadcast started anyway -- the same
        # defect as the Stop button that did not reach the engine, one step
        # earlier, and with quota spent on a broadcast nobody asked for.
        db.refresh(row)
        if row.status in ("canceled", "failed", "done"):
            print(f"[live_studio] STREAM {row.id} -> {row.status} before it went live; "
                  f"not starting", flush=True)
            return

        # Stream the prepared copy, not the original.
        if ready != src and not (row.source_url or "").strip():
            row.upload_path = ready
            db.commit()

        try:
            out = start_through_engine(db, row)
            row.status = "streaming"
            row.message = (note or f"live engine: {out.get('state') or 'started'}")[:512]
            db.commit()
        except Exception as exc:
            # DID IT ACTUALLY FAIL? start_through_engine is not atomic: it tries
            # add_channel and falls back to go_live, and things can throw AFTER
            # the engine has taken the channel -- a 409 because it is already
            # attached, or a commit failing once go_live has returned.
            #
            # Writing `failed` then is the worst of both: the broadcast streams
            # on, and the terminal row disarms the customer's Stop button,
            # because cancel_stream returns early on a terminal status. So ask
            # the engine before believing the exception.
            state = engine_owns_row(row)
            if state in ("on", "queued", "stopping"):
                row.status = "streaming"
                row.message = (f"live engine: {state} (the start reported an error "
                               f"but the channel is on: {exc})")[:512]
                db.commit()
                print(f"[live_studio] STREAM {row.id} -> streaming | via engine "
                      f"| start raised but the channel is {state}: "
                      f"{' '.join(str(exc).split())[:500]}", flush=True)
                return

            row.status = "failed"
            row.error = f"live engine refused: {exc}"[:2000]
            row.message = f"failed: live engine refused: {exc}"[:512]
            row.finished_at = datetime.now(timezone.utc)
            db.commit()
            print(f"[live_studio] STREAM {row.id} -> failed | via engine "
                  f"| err={' '.join(str(exc).split())[:2000]}", flush=True)
    finally:
        db.close()
