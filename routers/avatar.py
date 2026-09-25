# Ported from kaizer-platform@d5fd482 server/routers/avatar.py.
# Changes from upstream: (1) TENANCY STRIPPED — the vendor's tenant_context
# seam + _record_tenant_usage (tenancy.avatar_enforcement) are removed; this
# backend has no tenancy/Organization model, only the plain user path remains.
# (2) GPU-CONTENTION GUARD added — the SHARED services.gpu_gate.GPU_GATE
# serializes provider.generate() for every LOCAL provider (this host
# HARD-RESETS under concurrent GPU load; operator rule: one render at a
# time); "heygen" is exempt (remote API, no local GPU touched).
"""Provider-agnostic avatar generation router.

Supersedes ``routers/heygen.py`` (kept, gated off) — same job model,
but the engine is selected through the ``avatar`` provider registry:
local MuseTalk (default), vast.ai EchoMimic burst, or legacy HeyGen.

Endpoints
---------
  GET  /api/avatar/providers                    availability of each engine
  GET  /api/avatar/avatars?provider=            presenter catalog
  GET  /api/avatar/voices?provider=             voice catalog
  POST /api/avatar/generate                     script -> clip (direct)
  POST /api/avatar/generate-from-topic/{id}     trending topic -> clip
  GET  /api/avatar/status/{key}                 poll a generation

Both generate endpoints return immediately with a status ``key``; the
render runs on a daemon thread (same pattern as heygen/veo). On success
a Job + Clip row is created so the asset flows through the existing
editor / SEO / publish pipeline unchanged.
"""
from __future__ import annotations

import json
import re
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

import auth
import models
from avatar import (
    AvatarProviderError,
    GenerationRequest,
    gender_counts,
    get_provider,
    provider_names,
)
from database import SessionLocal, get_db
from heygen import script_builder
from heygen import transcript as topic_transcript
from services.gpu_gate import GPU_GATE

# Belt for the catalog endpoints: a raw heygen.client error that escapes a
# provider must map to 502, not 500. Guarded import — heygen's deps (httpx)
# are optional and must never take this router down (see avatar/__init__).
try:
    from heygen.client import HeyGenError
except Exception:  # noqa: BLE001 — placeholder; never raised when deps absent
    class HeyGenError(RuntimeError):  # type: ignore[no-redef]
        pass

router = APIRouter(prefix="/api/avatar", tags=["avatar"])


# ── GPU-contention guard (PORT ADDITION, not in upstream) ─────────
# This host HARD-RESETS under concurrent GPU render load (see memory note
# project_machine_stability_server_mode — 9 concurrent NVENC ffmpegs took
# the box down). Every LOCAL avatar engine (avatar_studio → MuseTalk on
# the local card, echomimic → shells avatar-studio's farm helper from this
# box) must therefore render ONE AT A TIME: a second request WAITS on the
# SHARED ``services.gpu_gate.GPU_GATE`` (imported above — one gate for every
# ad-hoc GPU render across routers, so avatar and podcast renders serialize
# against EACH OTHER, not just among themselves). "heygen" is exempt — it's
# a remote HTTP API and touches no local GPU, so serializing it would only
# add latency for nothing. Process-local, matching the process-local
# _status dict (v1 scope).
_GPU_EXEMPT_PROVIDERS = ("heygen",)


# avatar_id / voice_id reach filesystem paths + CLI args downstream —
# allowlist them before any thread is spawned.
_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def _require_safe_id(value: str, field: str) -> str:
    if not _ID_RE.fullmatch(value or ""):
        raise HTTPException(422, f"{field} must match [A-Za-z0-9_.-]{{1,64}}")
    return value


# ── Status tracker (string-keyed; topics use "topic:<id>") ────────
_status: dict[str, dict] = {}
_lock = threading.Lock()
_STATUS_TTL_S = 24 * 3600


def _set_status(key: str, **kw) -> None:
    with _lock:
        cur = _status.get(key) or {}
        cur.update(kw)
        cur["updated_at"] = time.time()
        _status[key] = cur
        # Hygiene: evict entries stale for >24h so the process-local dict
        # can't grow unbounded across weeks of uptime.
        cutoff = time.time() - _STATUS_TTL_S
        for stale in [k for k, v in _status.items()
                      if k != key and v.get("updated_at", 0.0) < cutoff]:
            del _status[stale]


def _get_status(key: str) -> dict:
    with _lock:
        return dict(_status.get(key) or {"state": "idle"})


_RUNNING_STATES = ("queued", "transcribing", "scripting", "rendering",
                   "downloading")


def _is_running(key: str) -> bool:
    return _get_status(key).get("state") in _RUNNING_STATES


def _fail_job(job_id: Optional[int], error: str) -> None:
    """Mark an up-front avatar Job row failed (thread-safe, own session).

    Jobs are created BEFORE the render thread starts (operator: a
    generation must appear in the jobs list immediately, and survive the
    user navigating away) — so every error path must settle the row."""
    if not job_id:
        return
    db = SessionLocal()
    try:
        j = db.get(models.Job, job_id)
        if j is not None and j.status not in ("done", "failed"):
            j.status = "failed"
            j.error = (error or "avatar generation failed")[:2000]
            db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def _sweep_orphaned_avatar_jobs() -> None:
    """A backend restart kills the daemon render threads — settle any
    Job rows they left 'running' so the jobs list never shows a zombie.
    Import-time, fully guarded (tests may have no DB)."""
    try:
        db = SessionLocal()
        try:
            rows = (db.query(models.Job)
                    .filter(models.Job.frame_layout == "avatar_generated",
                            models.Job.status == "running").all())
            for j in rows:
                j.status = "failed"
                j.error = "backend restarted mid-generation — generate again"
            if rows:
                db.commit()
                print(f"[avatar] swept {len(rows)} orphaned running job(s)",
                      flush=True)
        finally:
            db.close()
    except Exception:
        pass


_sweep_orphaned_avatar_jobs()


# ── Shared render-and-persist path ────────────────────────────────

def _render_and_persist(
    key: str,
    *,
    user_id: int,
    provider_name: Optional[str],
    script: str,
    avatar_id: str,
    voice_id: str,
    platform: str,
    language: str,
    topic: Optional[dict] = None,
    job_id: Optional[int] = None,
) -> None:
    """Blocking render + Job/Clip rows. Runs on a daemon thread with its
    own DB session.

    Local providers are serialized through the shared ``GPU_GATE`` (see
    the guard comment above) — a second concurrent request blocks here
    until the first render releases the GPU.
    """
    db = SessionLocal()
    try:
        provider = get_provider(provider_name)
        ready, reason = provider.available()
        if not ready:
            _set_status(key, state="error", error=reason)
            _fail_job(job_id, reason)
            return

        is_vertical = platform.endswith("short") or platform.endswith("reel")
        from runner import OUTPUT_ROOT
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(OUTPUT_ROOT) / "avatar" / f"{key.replace(':', '_')}_{ts}"

        def on_progress(pct: int, msg: str) -> None:
            # Provider progress maps onto the 10-90 band of the job.
            _set_status(key, state="rendering",
                        progress=10 + int(pct * 0.8), message=msg)

        gpu_bound = provider.name not in _GPU_EXEMPT_PROVIDERS
        if gpu_bound:
            _set_status(key, state="rendering", progress=8,
                        message="Waiting for the GPU render slot…")
            GPU_GATE.acquire()
        try:
            _set_status(key, state="rendering", progress=10,
                        message=f"{provider.name} rendering…")
            result = provider.generate(
                GenerationRequest(
                    script=script,
                    avatar_id=avatar_id,
                    voice_id=voice_id,
                    out_dir=out_dir,
                    width=1080 if is_vertical else 1920,
                    height=1920 if is_vertical else 1080,
                    language=language,
                ),
                on_progress=on_progress,
            )
        finally:
            if gpu_bound:
                GPU_GATE.release()

        # Thumbnail — frame 0, best-effort (mirrors heygen/veo helper).
        thumb_path = out_dir / "thumb_avatar.jpg"
        try:
            import subprocess
            from pipeline_core.pipeline import FFMPEG_BIN  # type: ignore
            subprocess.run(
                [FFMPEG_BIN, "-y", "-i", str(result.video_path),
                 "-vframes", "1", "-q:v", "2", str(thumb_path)],
                capture_output=True, check=True, timeout=30,
            )
        except Exception as exc:  # noqa: BLE001 — thumbnail is cosmetic
            print(f"[avatar] thumb gen failed: {exc}")

        topic_info = topic or {}
        # The Job row was created UP FRONT (status='running') so the jobs
        # list shows the generation immediately — settle it now. Fallback
        # create keeps old callers working if job_id is absent.
        job = db.get(models.Job, job_id) if job_id else None
        if job is None:
            job = models.Job(
                user_id=user_id,
                platform=platform,
                frame_layout="avatar_generated",
                video_name=f"avatar_{key.replace(':', '_')}.mp4",
                language=language,
                status="running",
            )
            db.add(job)
        job.status = "done"
        job.log = (
            f"[avatar] provider={result.provider} engine={result.engine}\n"
            f"avatar={avatar_id}, voice={voice_id}\n"
            f"script[:300]: {script[:300]}"
        )
        job.output_dir = str(out_dir)
        db.commit(); db.refresh(job)

        clip = models.Clip(
            job_id=job.id,
            clip_index=0,
            filename=result.video_path.name,
            file_path=str(result.video_path),
            thumb_path=str(thumb_path) if thumb_path.exists() else "",
            image_path="",
            duration=result.duration_s,
            frame_type="avatar_generated",
            text=(topic_info.get("summary") or script)[:300],
            sentiment="",
            entities=json.dumps(topic_info.get("keywords") or []),
            card_params=json.dumps({}),
            section_pct=json.dumps({}),
            follow_params=json.dumps({}),
            meta=json.dumps({
                "source": result.provider,
                "engine": result.engine,
                "avatar_id": avatar_id,
                "voice_id": voice_id,
                "script": script,
                "language": language,
                "platform": platform,
                **({"topic_id": topic_info["id"]} if topic_info.get("id") else {}),
                **result.meta,
            }),
        )
        db.add(clip); db.commit(); db.refresh(clip)

        if topic_info.get("id"):
            row = db.query(models.TrendingTopic).get(topic_info["id"])
            if row is not None:
                row.used_for_job_id = job.id
                db.commit()

        _set_status(key, state="done", progress=100,
                    message="Done — clip ready in editor",
                    job_id=job.id, clip_id=clip.id)
    except AvatarProviderError as exc:
        _set_status(key, state="error", error=str(exc)[:400])
        _fail_job(job_id, str(exc))
    except Exception as exc:  # noqa: BLE001 — thread boundary; surface via status
        traceback.print_exc()
        _set_status(key, state="error", error=str(exc)[:400])
        _fail_job(job_id, str(exc))
    finally:
        db.close()


# ── Catalog endpoints ─────────────────────────────────────────────

@router.get("/providers")
def list_providers(user: models.User = Depends(auth.current_user)) -> dict:
    out = []
    for name in provider_names():
        try:
            ready, reason = get_provider(name).available()
        except AvatarProviderError as exc:
            ready, reason = False, str(exc)
        out.append({"name": name, "available": ready, "reason": reason})
    try:
        return {"providers": out, "default": get_provider().name}
    except AvatarProviderError as exc:
        # A misconfigured default must not 500 the whole catalog.
        return {"providers": out, "default": None, "default_error": str(exc)}


@router.get("/avatars")
def list_avatars(
    provider: Optional[str] = None,
    user: models.User = Depends(auth.current_user),
) -> dict:
    try:
        p = get_provider(provider)
        return {"provider": p.name,
                "avatars": [vars(a) for a in p.list_avatars()]}
    except (AvatarProviderError, HeyGenError) as exc:
        # HeyGenError = belt: a raw client error escaping a provider.
        raise HTTPException(502, str(exc))


@router.get("/voices")
def list_voices(
    provider: Optional[str] = None,
    language: Optional[str] = None,
    user: models.User = Depends(auth.current_user),
) -> dict:
    try:
        p = get_provider(provider)
        voices = p.list_voices()
    except (AvatarProviderError, HeyGenError) as exc:
        # HeyGenError = belt: a raw client error escaping a provider.
        raise HTTPException(502, str(exc))
    if language:
        voices = [v for v in voices if v.language == language]
    # gender_counts covers the filtered list — the UI's 70F/30M pack
    # gauge would be meaningless computed across all languages.
    return {"provider": p.name, "voices": [vars(v) for v in voices],
            "count": len(voices), "gender_counts": gender_counts(voices)}


@router.get("/voices/{voice_id}/sample")
def voice_sample(
    voice_id: str,
    provider: Optional[str] = None,
    user: models.User = Depends(auth.current_user),
):
    """Reference audio for one voice, when the active provider has one.

    The local studio provider ships a clone-source ``ref.wav`` per voice
    (served as a file); remote providers (HeyGen) publish a hosted preview
    clip instead — redirect to it when the resolved voice has one; 404
    otherwise rather than pretending to have a sample.
    """
    try:
        p = get_provider(provider)
    except (AvatarProviderError, HeyGenError) as exc:
        raise HTTPException(502, str(exc))
    getter = getattr(p, "voice_sample_path", None)
    path = getter(voice_id) if getter else None
    if path is not None:
        return FileResponse(path, media_type="audio/wav")
    # Remote providers: redirect to the API's hosted preview audio.
    try:
        match = next((v for v in p.list_voices() if v.id == voice_id), None)
    except (AvatarProviderError, HeyGenError) as exc:
        raise HTTPException(502, str(exc))
    if match is not None and getattr(match, "preview_url", ""):
        return RedirectResponse(match.preview_url)
    raise HTTPException(404, f"no sample audio for voice '{voice_id}'")


@router.get("/avatars/{avatar_id}/preview")
def avatar_preview(
    avatar_id: str,
    provider: Optional[str] = None,
    user: models.User = Depends(auth.current_user),
):
    """Thumbnail for one presenter card. Local studio avatars serve their
    photo (or a cached poster frame extracted once from the video-loop
    anchor); remote providers redirect to the API's hosted preview."""
    _require_safe_id(avatar_id, "avatar_id")
    try:
        p = get_provider(provider)
        match = next((a for a in p.list_avatars() if a.id == avatar_id), None)
    except (AvatarProviderError, HeyGenError) as exc:
        raise HTTPException(502, str(exc))
    pv = getattr(match, "preview_path", "") if match else ""
    if not pv:
        raise HTTPException(404, f"no preview for avatar '{avatar_id}'")
    if pv.startswith(("http://", "https://")):
        return RedirectResponse(pv)
    path = Path(pv)
    if not path.is_file():
        raise HTTPException(404, "preview file missing")
    if path.suffix.lower() in (".mp4", ".mov", ".webm"):
        poster = path.with_name(f".{path.stem}_poster.jpg")
        if not poster.is_file():
            try:
                import subprocess
                from pipeline_core.pipeline import FFMPEG_BIN  # type: ignore
                subprocess.run(
                    [FFMPEG_BIN, "-y", "-i", str(path), "-vframes", "1",
                     "-q:v", "3", str(poster)],
                    capture_output=True, timeout=30, check=True)
            except Exception as exc:
                raise HTTPException(404, f"poster extraction failed: {exc}")
        return FileResponse(str(poster), media_type="image/jpeg")
    mt = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(str(path), media_type=mt)


# ── Direct generation (script in, clip out) ───────────────────────

class GenerateDirect(BaseModel):
    script: str
    avatar_id: str = "avatar1"
    voice_id: str = "anchor_f1"
    platform: str = "youtube_short"     # youtube_short | youtube_full | instagram_reel
    language: str = "te"
    provider: Optional[str] = None      # None -> env default


@router.post("/generate")
def generate_direct(
    payload: GenerateDirect,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    script = payload.script.strip()
    if not script:
        raise HTTPException(400, "script required")
    avatar_id = _require_safe_id(payload.avatar_id.strip(), "avatar_id")
    voice_id = _require_safe_id(payload.voice_id.strip(), "voice_id")
    key = f"direct:{uuid.uuid4().hex[:12]}"
    # Job row UP FRONT: the generation shows in the jobs list immediately
    # and survives the user navigating away from /anchor (operator ask).
    job = models.Job(
        user_id=user.id,
        platform=payload.platform,
        frame_layout="avatar_generated",
        video_name=("Anchor: " + script)[:80],
        language=payload.language,
        status="running",
        log=f"[avatar] queued key={key} "
            f"provider={payload.provider or 'default'} "
            f"avatar={avatar_id} voice={voice_id}",
    )
    db.add(job); db.commit(); db.refresh(job)
    _set_status(key, state="queued", progress=0, message="Queued",
                job_id=job.id, clip_id=None, error="", user_id=user.id)
    threading.Thread(
        target=_render_and_persist,
        kwargs=dict(
            key=key, user_id=user.id, provider_name=payload.provider,
            script=script, avatar_id=avatar_id,
            voice_id=voice_id, platform=payload.platform,
            language=payload.language, job_id=job.id,
        ),
        daemon=True,
    ).start()
    return {"status": "queued", "key": key, "job_id": job.id}


# ── Topic-driven generation (trending -> transcript -> script) ────

class GenFromTopic(BaseModel):
    platform: str = "youtube_short"
    language: str = "te"
    avatar_id: str = "avatar1"
    voice_id: str = "anchor_f1"
    provider: Optional[str] = None


def _run_topic_job(key: str, topic_id: int, user_id: int,
                   payload: GenFromTopic,
                   job_id: Optional[int] = None) -> None:
    """Transcript + script phases, then the shared render path.

    Catch-all mirrors ``_render_and_persist``'s handler — this runs on a
    daemon thread, so an unexpected exception must land in the status
    entry, not die silently and leave the job stuck on "scripting".
    """
    try:
        db = SessionLocal()
        try:
            topic = (
                db.query(models.TrendingTopic)
                  .join(models.CompetitorChannel,
                        models.CompetitorChannel.id == models.TrendingTopic.source_channel_id)
                  .filter(models.TrendingTopic.id == topic_id,
                          models.CompetitorChannel.user_id == user_id)
                  .first()
            )
            if not topic:
                _set_status(key, state="error", error="Topic not found or not yours")
                _fail_job(job_id, "Topic not found or not yours")
                return
            if not topic.video_url:
                _set_status(key, state="error",
                            error="Topic has no video_url to transcribe")
                _fail_job(job_id, "Topic has no video_url to transcribe")
                return

            _set_status(key, state="transcribing", progress=5,
                        message="Pulling video transcript…")
            try:
                tr = topic_transcript.fetch_transcript(
                    topic.video_url, language=payload.language)
            except topic_transcript.TranscriptError as exc:
                _set_status(key, state="error", error=str(exc)[:300])
                _fail_job(job_id, str(exc))
                return

            _set_status(key, state="scripting", progress=8,
                        message="Building avatar script…")
            keywords: list = []
            try:
                if isinstance(topic.keywords, str):
                    keywords = json.loads(topic.keywords)
                elif isinstance(topic.keywords, list):
                    keywords = topic.keywords
            except (ValueError, TypeError):
                keywords = []
            pack = script_builder.build_script(
                topic_title=topic.video_title or "",
                topic_summary=topic.topic_summary or "",
                topic_keywords=keywords,
                transcript=tr["transcript"],
                transcript_source=tr.get("source", "captions"),
                language=payload.language or tr.get("language") or "te",
            )
            if not pack["script"]:
                _set_status(key, state="error",
                            error="Script builder produced empty output")
                _fail_job(job_id, "Script builder produced empty output")
                return

            topic_info = {"id": topic_id,
                          "summary": topic.topic_summary or topic.video_title,
                          "keywords": keywords}
        finally:
            db.close()

        _render_and_persist(
            key, user_id=user_id, provider_name=payload.provider,
            script=pack["script"], avatar_id=payload.avatar_id,
            voice_id=payload.voice_id, platform=payload.platform,
            language=payload.language, topic=topic_info, job_id=job_id,
        )
    except Exception as exc:  # noqa: BLE001 — thread boundary; surface via status
        traceback.print_exc()
        _set_status(key, state="error", error=str(exc)[:400])
        _fail_job(job_id, str(exc))


@router.post("/generate-from-topic/{topic_id}")
def generate_from_topic(
    topic_id: int,
    payload: GenFromTopic,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    payload.avatar_id = _require_safe_id(
        payload.avatar_id.strip(), "avatar_id")
    payload.voice_id = _require_safe_id(
        payload.voice_id.strip(), "voice_id")
    key = f"topic:{topic_id}"
    if _is_running(key):
        return {"status": "already_running", "current": _get_status(key),
                "key": key}
    exists = (
        db.query(models.TrendingTopic)
          .join(models.CompetitorChannel,
                models.CompetitorChannel.id == models.TrendingTopic.source_channel_id)
          .filter(models.TrendingTopic.id == topic_id,
                  models.CompetitorChannel.user_id == user.id)
          .first()
    )
    if not exists:
        raise HTTPException(404, "Topic not found")

    # Job row UP FRONT (see generate_direct) — visible in the jobs list
    # from the first second, settled by _fail_job on every error path.
    job = models.Job(
        user_id=user.id,
        platform=payload.platform,
        frame_layout="avatar_generated",
        video_name=("Anchor: " + (exists.video_title or f"topic {topic_id}"))[:80],
        language=payload.language,
        status="running",
        log=f"[avatar] queued key={key} topic={topic_id} "
            f"provider={payload.provider or 'default'}",
    )
    db.add(job); db.commit(); db.refresh(job)
    _set_status(key, state="queued", progress=0, message="Queued",
                job_id=job.id, clip_id=None, error="", user_id=user.id)
    threading.Thread(
        target=_run_topic_job, args=(key, topic_id, user.id, payload, job.id),
        daemon=True,
    ).start()
    return {"status": "queued", "key": key, "job_id": job.id}


@router.get("/status/{key}")
def get_status(
    key: str,
    user: models.User = Depends(auth.current_user),  # auth gate
) -> dict:
    st = _get_status(key)
    owner = st.get("user_id")
    if (owner is not None and owner != user.id
            and not getattr(user, "is_admin", False)):
        # 404 (not 403) — don't confirm another user's key exists.
        raise HTTPException(404, "status key not found")
    return st


@router.get("/active")
def active_generation(
    user: models.User = Depends(auth.current_user),
) -> dict:
    """The caller's newest IN-FLIGHT generation, so /anchor can resume
    live progress after the user navigated away and came back (the Job
    row is in the jobs list either way; this restores the live view)."""
    with _lock:
        mine = [(k, dict(v)) for k, v in _status.items()
                if v.get("user_id") == user.id
                and v.get("state") in _RUNNING_STATES]
    if not mine:
        return {"active": False}
    key, st = max(mine, key=lambda kv: kv[1].get("updated_at", 0.0))
    return {"active": True, "key": key,
            "state": st.get("state"), "progress": st.get("progress"),
            "message": st.get("message"), "job_id": st.get("job_id")}
