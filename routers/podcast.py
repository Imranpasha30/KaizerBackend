# Ported from kaizer-platform@d5fd482 server/routers/podcast.py.
# Changes from upstream: (1) auto-STT rewired from the removed pipeline_v2 STT dispatcher to
# this tree's Deepgram helper (pipeline_v4.trim_engine._extract_audio_mp3/_deepgram_words) via
# a thin word-shape adapter — supplied transcript_json path kept verbatim; (2) the render call
# is serialized behind the PROCESS-WIDE services.gpu_gate.GPU_GATE (single shared GPU on this
# host — one gate shared with every other ad-hoc GPU router, not a module-local semaphore);
# (3) imports/endpoints/Job-row handling otherwise verbatim (field-compatible with our models.Job).
"""Podcast mode router (Phase 1 — single-cam MVP).

Endpoints
---------
  POST /api/podcast/jobs        create a podcast job (multipart video OR
                                asset_id; optional pre-supplied transcript
                                JSON for boxes without STT keys)
  GET  /api/podcast/jobs/{key}  poll job status
  GET  /api/podcast/jobs/{key}/results   edit + promos + cutlist JSON

The pipeline runs on a daemon thread. Job rows REUSE the existing ``Job``
model with ``platform='podcast'`` — no schema change (the cut-list + result
payload live in the in-memory store and a results.json on disk;
``Job.output_dir`` points at them).

Transcription: uses this tree's Deepgram nova-3 helper (the same one
pipeline_v4's trim engine uses) when ``DEEPGRAM_API_KEY`` is set. Word-level
speaker labels (diarization) are requested EXPLICITLY (``diarize=True``) —
the podcast camera plan is dead without speaker labels, so this shim does
not depend on the V4 ``KAIZER_V4_DIARIZE`` env gate (news jobs keep their
env-gated behavior untouched). If no key is available, the create endpoint
400s upfront and the caller MUST supply a ``transcript_json`` param (the
word-level ``{"words": [...]}`` shape) — this is the honest limitation on a
box without STT keys.

GPU rule (this host): renders are serialized ONE at a time behind the
process-wide ``services.gpu_gate.GPU_GATE`` — the box hard-resets under
concurrent GPU render load, and per-module private gates cannot serialize
against each other.

Restart resilience: the in-memory ``_status`` store dies with the process,
so (a) at import a sweep marks podcast Job rows stranded ``running`` as
failed, and (b) the status/results endpoints fall back to the Job row (and
``results.json`` on disk) when a key is missing from the store.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

import auth
import models
from asset_resolver import materialize_asset_locally
from config import settings
from database import SessionLocal, get_db

from pipeline_core.podcast.cutlist import CutlistConfig, build_keep_ranges
from pipeline_core.podcast.punchin import PunchInConfig, build_punch_in_plan
from pipeline_core.podcast.promo import PromoConfig, build_promo_plan

# ONE render at a time on this host — the machine hard-resets under
# concurrent GPU (NVENC) render load, and DEV+LIVE share the single GPU.
# The gate is the PROCESS-WIDE shared one (services/gpu_gate.py), so this
# router serializes against every other ad-hoc GPU consumer too.
from services.gpu_gate import GPU_GATE

router = APIRouter(prefix="/api/podcast", tags=["podcast"])

logger = logging.getLogger("routers.podcast")


def _media_url_for(abs_path: Optional[str]) -> Optional[str]:
    """HTTP-reachable URL for a rendered output path. Outputs live under
    ``settings.output_root``, which main.py mounts at ``/media`` — so a file
    at ``<output_root>/podcast/<key>/edit.mp4`` is served at
    ``/media/podcast/<key>/edit.mp4``. Returns None for a path outside the
    output root (frontend then shows its honest 'not servable' fallback)."""
    if not abs_path:
        return None
    try:
        rel = Path(abs_path).resolve().relative_to(Path(settings.output_root).resolve())
    except (ValueError, OSError):
        return None
    return "/media/" + rel.as_posix()


# ── In-memory status store ────────────────────────────────────────────────
_status: dict[str, dict] = {}
_lock = threading.Lock()

_RUNNING_STATES = ("queued", "transcribing", "planning", "rendering")


def _set_status(key: str, **kw) -> None:
    with _lock:
        cur = _status.get(key) or {}
        cur.update(kw)
        cur["updated_at"] = time.time()
        _status[key] = cur


def _get_status(key: str) -> dict:
    with _lock:
        return dict(_status.get(key) or {"state": "idle"})


# ── Upload filename sanitizer ─────────────────────────────────────────────

# Extension whitelist for direct uploads (video/audio containers only).
_ALLOWED_UPLOAD_EXTS = (".mp4", ".mov", ".mkv", ".m4a", ".wav", ".webm")


def _safe_upload_name(filename: Optional[str]) -> str:
    """Sanitize a CLIENT-CONTROLLED upload filename into a bare, safe name.

    ``UploadFile.filename`` is attacker-controlled: joining it into out_dir
    allows path traversal (``..\\..\\x.mp4``) and absolute-path override
    (``C:\\anything``) = arbitrary file write. Defense: basename only, then
    flatten every char outside ``[A-Za-z0-9._-]`` to ``_``, then enforce the
    extension whitelist. Raises ``ValueError`` on a disallowed extension
    (endpoint maps it to HTTP 422).
    """
    name = os.path.basename(filename or "source.mp4")
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name) or "source.mp4"
    ext = os.path.splitext(name)[1].lower()
    if ext not in _ALLOWED_UPLOAD_EXTS:
        raise ValueError(
            f"unsupported upload extension {ext or '(none)'!r}; "
            f"allowed: {', '.join(_ALLOWED_UPLOAD_EXTS)}"
        )
    return name


# ── Restart resilience ────────────────────────────────────────────────────


def _sweep_orphaned_jobs() -> None:
    """Mark podcast Job rows stranded ``running`` by a backend restart as
    failed. The pipeline runs on an in-process daemon thread, so a restart
    kills it silently — without this sweep those rows sit ``running``
    forever. Guarded: the DB may be unavailable at import (tests / partial
    boots) and that must never block the module import."""
    try:
        db = SessionLocal()
        try:
            rows = (
                db.query(models.Job)
                .filter(models.Job.platform == "podcast",
                        models.Job.status == "running")
                .all()
            )
            for row in rows:
                row.status = "failed"
                row.error = "backend restarted mid-job"
            if rows:
                db.commit()
                logger.warning(
                    "podcast: marked %d orphaned running job(s) as failed "
                    "after restart", len(rows))
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001 — import must survive a dead DB
        logger.warning("podcast: restart sweep skipped (DB unavailable?): %s", exc)


def _status_from_db(key: str, db: Session) -> Optional[dict]:
    """Rebuild a status dict for a key the in-memory store lost (restart).

    Finds the Job row whose ``output_dir`` ends with the key's on-disk dir
    form (``pod:<hex>`` → ``pod_<hex>``; the hex makes the suffix unique) and
    serves state from the row plus ``results.json`` from disk when present.
    Returns None when no such row exists (a genuinely unknown key)."""
    dir_form = key.replace(":", "_")
    try:
        row = (
            db.query(models.Job)
            .filter(models.Job.platform == "podcast",
                    models.Job.output_dir.endswith(dir_form, autoescape=True))
            .order_by(models.Job.id.desc())
            .first()
        )
    except Exception as exc:  # noqa: BLE001 — DB fallback is best-effort
        logger.warning("podcast: status DB fallback failed for %s: %s", key, exc)
        return None
    if row is None:
        return None
    st: dict = {"job_id": row.id, "recovered": True}
    results_path = os.path.join(row.output_dir or "", "results.json")
    if row.status == "done":
        st.update(state="done", progress=100, message="Complete", error="")
        if row.output_dir and os.path.isfile(results_path):
            try:
                st["results"] = json.loads(
                    Path(results_path).read_text(encoding="utf-8"))
                st["results_path"] = results_path
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("podcast: results.json unreadable for %s: %s",
                               key, exc)
    else:
        # failed — or any non-done state (a 'running' row here means the
        # sweep itself could not run; the thread is gone either way).
        st.update(state="error",
                  error=(row.error or "backend restarted mid-job")[:500])
    _set_status(key, **st)  # cache so subsequent polls skip the DB
    return _get_status(key)


# Import-time sweep (module import == process start for this router).
_sweep_orphaned_jobs()


# ── Word double for pre-supplied transcript JSON ──────────────────────────

class _W:
    """Duck-typed Word (matches the planners' ``WordT`` attribute shape)."""

    __slots__ = ("w", "s", "e", "speaker", "confidence")

    def __init__(self, w: str, s: float, e: float,
                 speaker: Optional[int] = None, confidence: Optional[float] = None):
        self.w, self.s, self.e = w, float(s), float(e)
        self.speaker, self.confidence = speaker, confidence


def _words_from_transcript_json(raw: str) -> list[_W]:
    """Parse a word-level-transcript-shaped JSON string into Word doubles.

    Accepts either ``{"words": [...]}`` or a bare ``[...]`` list. Each word:
    ``{"w": str, "s": float, "e": float, "speaker": int?, "confidence": float?}``.

    ``raw`` arrives already decoded to ``str`` by FastAPI's ``Form(...)``
    dependency (Starlette's multipart parser defaults the field charset to
    UTF-8), so ``json.loads`` here never touches raw bytes/codepage decoding.
    """
    data = json.loads(raw)
    words = data["words"] if isinstance(data, dict) else data
    out: list[_W] = []
    for i, wd in enumerate(words):
        try:
            out.append(_W(
                w=str(wd["w"]), s=wd["s"], e=wd["e"],
                speaker=wd.get("speaker"), confidence=wd.get("confidence"),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"transcript word {i} invalid: {exc}") from exc
    if not out:
        raise ValueError("transcript_json contained no words")
    return out


# ── Pipeline (runs on a daemon thread) ────────────────────────────────────


def _run_podcast_job(
    key: str,
    *,
    user_id: int,
    source_path: str,
    out_dir: str,
    transcript_json: Optional[str],
    stt_provider: Optional[str],
    language: Optional[str],
    silence_threshold_ms: float,
    renderer: Optional[str] = None,
) -> None:
    db = SessionLocal()
    job: Optional[models.Job] = None
    try:
        job = models.Job(
            user_id=user_id, status="running", platform="podcast",
            video_name=os.path.basename(source_path), output_dir=out_dir,
            language=language or "en",
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        _set_status(key, job_id=job.id)

        # 1) Transcript.
        if transcript_json:
            _set_status(key, state="planning", progress=20,
                        message="Using supplied transcript")
            words = _words_from_transcript_json(transcript_json)
        else:
            _set_status(key, state="transcribing", progress=10,
                        message="Transcribing (STT)")
            words = _transcribe_via_stt(source_path, stt_provider, language, out_dir)

        # 1.5) Editorial pass — the LLM "Podcast Director" (Phase A). Makes
        #      SEMANTIC decisions the mechanical cutlist can't: rambling/
        #      aside/retake drops (labeled + audited), meaning-based emphasis,
        #      and promo picks. Skipped honestly when no LLM key is set; a
        #      failure falls back to the mechanical path and is RECORDED in
        #      results.editorial.error (never silently swallowed).
        from pipeline_core.podcast.editorial import (
            editorial_provider, run_editorial, filter_dropped_words,
            captions_for_ranges, promo_plan_from_picks,
        )
        editorial = None
        editorial_error: Optional[str] = None
        prov = editorial_provider()
        if prov:
            _set_status(key, state="planning", progress=30,
                        message=f"AI editorial pass ({prov})")
            try:
                editorial = run_editorial(words, language=language)
            except Exception as exc:  # noqa: BLE001 — degrade, don't fail the job
                editorial_error = str(exc)[:300]
                logger.warning("editorial pass failed; falling back to "
                               "mechanical cutlist: %s", exc)
        cut_words = filter_dropped_words(words, editorial) if editorial else words

        # 2) Cut-list (mechanical silence/filler pass on the kept words).
        _set_status(key, state="planning", progress=40, message="Building cut-list")
        cut = build_keep_ranges(
            cut_words, CutlistConfig(silence_cut_threshold_ms=silence_threshold_ms))
        if not cut.keep_ranges:
            raise RuntimeError("cut-list produced no keep_ranges (empty/too-aggressive)")

        # 2.5) Vision quality gate (Phase B) — carve out out-of-focus/bad-
        #      exposure spans the transcript can't see (audio can be fine
        #      while the picture is unusable). Honest opt-out via env; a
        #      failure degrades to the mechanical keep_ranges, never silently.
        keep_ranges: tuple[tuple[float, float], ...] = cut.keep_ranges
        quality_spans: list[dict] = []
        quality_error: Optional[str] = None
        if os.environ.get("KAIZER_PODCAST_VISION_GATE", "auto").strip().lower() != "off":
            _set_status(key, state="planning", progress=45, message="Checking video quality")
            try:
                from pipeline_core.podcast.vision_quality import (
                    analyze_video_quality, subtract_spans,
                )
                bad = analyze_video_quality(source_path)
                if bad:
                    keep_ranges = subtract_spans(keep_ranges, bad)
                    quality_spans = [
                        {"start_s": b.start_s, "end_s": b.end_s,
                         "category": b.category, "reason": b.reason}
                        for b in bad
                    ]
            except Exception as exc:  # noqa: BLE001 — degrade, don't fail the job
                quality_error = str(exc)[:300]
                logger.warning("vision quality gate failed; keeping mechanical "
                               "ranges unchanged: %s", exc)
        if not keep_ranges:
            raise RuntimeError(
                "vision quality gate removed every kept range (source may be "
                "entirely out of focus/misexposed)")

        # 3) Punch-in plan.
        punch = build_punch_in_plan(keep_ranges, cut_words, PunchInConfig())

        # 3.5) Camera plan (Phase C) — virtual multi-cam reframe. Only
        #      engages with two confidently-mapped speaker face-slots; a
        #      disengaged plan renders the original framing (see
        #      camera_plan.py for the honesty rule). A failure degrades the
        #      same way — no reframe, never a failed job.
        camera_plan = None
        camera_info: dict = {"engaged": False, "reason": "skipped"}
        if os.environ.get("KAIZER_PODCAST_CAMERA_PLAN", "auto").strip().lower() != "off":
            _set_status(key, state="planning", progress=55, message="Planning camera framing")
            try:
                from pipeline_core.podcast.camera_plan import (
                    build_camera_plan, CameraPlanConfig,
                )
                camera_plan = build_camera_plan(
                    cut_words, keep_ranges, source_path, CameraPlanConfig())
                camera_info = {
                    "engaged": camera_plan.engaged, "reason": camera_plan.reason,
                    "windows": len(camera_plan.windows),
                }
            except Exception as exc:  # noqa: BLE001 — degrade, don't fail the job
                logger.warning("camera plan failed; rendering without reframe: %s", exc)
                camera_info = {"engaged": False, "reason": f"error: {exc}"[:300]}

        # 4) Word-pop captions for the main edit (edited-timeline coords).
        #    Editorial emphasis (meaning-based) wins when available; otherwise
        #    the deterministic heuristic (numbers + longest content word).
        from pipeline_core.podcast.emphasis import mark_emphasis
        if editorial and editorial.emphasis_starts:
            edit_captions = captions_for_ranges(
                cut_words, keep_ranges, editorial.emphasis_starts)
        else:
            edit_captions = mark_emphasis(_edit_captions(cut_words, keep_ranges))

        # 5) Promo plan — director's picks when present, heuristic otherwise.
        #    Note: promo picks are not (yet) checked against vision-quality
        #    bad spans — a known Phase-D gap, not silently glossed over.
        promo = None
        promo_source = "heuristic"
        if editorial and editorial.promo_picks:
            promo = promo_plan_from_picks(words, editorial, PromoConfig())
            promo_captions = list(promo.captions)  # emphasis already tagged
            promo_source = "editorial"
        if promo is None or not promo.keep_ranges:
            promo = build_promo_plan(cut_words, keep_ranges, PromoConfig())
            promo_captions = mark_emphasis(promo.captions)
            promo_source = "heuristic"

        # 6) Render — serialized behind the single-GPU gate (see module
        #    docstring; concurrent GPU renders hard-reset this host).
        _set_status(key, state="rendering", progress=65, message="Rendering edit + promos")
        from pipeline_core.podcast.render import render_podcast
        with GPU_GATE:
            result = render_podcast(
                source_path=source_path,
                out_dir=out_dir,
                edit_keep_ranges=keep_ranges,
                punch_ins=punch,
                edit_captions=edit_captions,
                camera_plan=camera_plan,
                promo_keep_ranges=promo.keep_ranges,
                promo_captions=promo_captions,
                renderer=renderer,
                language=language,
                promo_end_card_sec=promo.end_card_sec,
            )

        # 7) Persist a transparency results.json.
        #    Outputs live under settings.output_root, which is the same dir
        #    main.py mounts at /media — so expose HTTP-reachable URLs the
        #    frontend <video> players can fetch, alongside the raw FS paths.
        results = {
            "edit": result.edit_path,
            "promo_169": result.promo_169_path,
            "promo_916": result.promo_916_path,
            "edit_url": _media_url_for(result.edit_path),
            "promo_169_url": _media_url_for(result.promo_169_path),
            "promo_916_url": _media_url_for(result.promo_916_path),
            "renderer_used": result.renderer_used,
            "theme": result.theme,
            "cutlist": {
                # keep_ranges/kept/removed reflect the FINAL ranges actually
                # rendered (post vision-quality gate), not just the mechanical
                # cutlist's own output — `dropped` stays the mechanical
                # silence/filler/stutter annotations for that transparency.
                "keep_ranges": [list(r) for r in keep_ranges],
                "dropped": list(cut.dropped),
                "source_duration_s": cut.source_duration_s,
                "kept_seconds": sum(e - s for s, e in keep_ranges),
                "removed_seconds": max(
                    0.0, cut.source_duration_s - sum(e - s for s, e in keep_ranges)),
            },
            "vision_quality": {
                "spans": quality_spans,
                "error": quality_error,
            },
            "camera": camera_info,
            "punch_ins": [asdict(p) for p in punch],
            "promo": {
                "keep_ranges": [list(r) for r in promo.keep_ranges],
                "segments": [asdict(s) for s in promo.segments],
                "target_duration_s": promo.target_duration_s,
                "total_duration_s": promo.total_duration_s,
                "end_card_sec": promo.end_card_sec,
                "source": promo_source,
            },
            # Editorial transparency — what the AI director decided and why.
            # engine=None + error=None means no LLM key was configured (the
            # honest mechanical-only fallback); error records a failed pass.
            "editorial": {
                "engine": editorial.engine if editorial else None,
                "drops": editorial.drops if editorial else [],
                "audit": editorial.audit if editorial else "",
                "error": editorial_error,
                "promo_source": promo_source,
            },
        }
        results_path = os.path.join(out_dir, "results.json")
        Path(results_path).write_text(json.dumps(results, indent=2), encoding="utf-8")

        job.status = "done"
        db.commit()
        _set_status(key, state="done", progress=100, message="Complete",
                    results=results, results_path=results_path,
                    renderer=result.renderer_used, theme=result.theme)
    except Exception as exc:  # noqa: BLE001 — surface any pipeline failure
        tb = traceback.format_exc()
        # In-memory status FIRST — pollers must see 'error' even if the DB
        # write below fails (a dirty session must never strand the key at
        # state='rendering' with Job.status='running').
        _set_status(key, state="error", error=str(exc)[:500], traceback=tb[-2000:])
        if job is not None:
            try:
                db.rollback()  # the session may be dirty/invalid from the failure
                job.status = "failed"
                job.error = str(exc)[:2000]
                db.commit()
            except Exception as db_exc:  # noqa: BLE001 — status already 'error'
                logger.warning(
                    "podcast job %s: could not persist failure to Job row: %s",
                    key, db_exc)
    finally:
        db.close()


def _transcribe_via_stt(
    source_path: str, provider: Optional[str], language: Optional[str], out_dir: str
) -> list:
    """Transcribe via this tree's Deepgram nova-3 helper (word-level).

    PORT SHIM (upstream used the removed pipeline_v2 STT dispatcher): reuses
    ``pipeline_v4.trim_engine._extract_audio_mp3`` + ``_deepgram_words`` — the
    exact path the V4 trim engine runs — and adapts the returned
    ``{"i","w","s","e","spk"?}`` dicts to the planners' ``WordT`` attribute
    shape (``w``/``s``/``e``/``speaker``/``confidence``):

      * ``speaker`` maps from ``spk`` — this shim requests diarization
        EXPLICITLY (``diarize=True``; the vendor pipeline always did — the
        camera plan is dead without speaker labels). V4 news jobs are
        untouched: their call sites omit the parameter and keep the
        ``KAIZER_V4_DIARIZE`` env-gated behavior.
      * ``confidence`` maps from ``conf`` (per-word Deepgram confidence,
        emitted by ``_deepgram_words`` when the response carries it); the
        cutlist's soft-filler pass uses it, falling back to its repeat-guard
        when absent.
      * language: passes the V4 convention (``KAIZER_V4_LANGUAGE``, default
        ``"multi"`` — nova-3's multilingual mode) rather than the raw job
        language hint, which nova-3 may not accept as a monolingual code.
        The job ``language`` still drives editorial + theme.

    Deepgram is the only auto-STT provider on this box. Raises with a clear
    message if no key is available so the caller knows to supply a
    transcript instead.
    """
    prov = provider or _default_stt_provider()
    if prov is None or prov != "deepgram":
        raise RuntimeError(
            "no STT provider available on this box; supply transcript_json "
            "(available providers: ['deepgram'] — set DEEPGRAM_API_KEY)"
        )

    from pipeline_v4.trim_engine import _deepgram_words, _extract_audio_mp3

    audio_mp3 = os.path.join(out_dir, "podcast_stt.mp3")
    _extract_audio_mp3(source_path, audio_mp3)
    dg_language = os.environ.get("KAIZER_V4_LANGUAGE", "multi")
    raw_words, _dur = _deepgram_words(audio_mp3, language=dg_language, diarize=True)
    if not raw_words:
        raise RuntimeError(
            "Deepgram returned no words for this source; supply transcript_json"
        )
    return [
        _W(w=d["w"], s=d["s"], e=d["e"], speaker=d.get("spk"),
           confidence=d.get("conf"))
        for d in raw_words
    ]


def _default_stt_provider() -> Optional[str]:
    """Pick a provider whose key is present in the environment.

    PORT SHIM: Deepgram only (upstream also offered assemblyai/whisper-groq
    via the pipeline_v2 dispatcher, which this tree does not carry).
    """
    if os.environ.get("DEEPGRAM_API_KEY", "").strip():
        return "deepgram"
    return None


def _edit_captions(words, keep_ranges) -> list[dict]:
    """Per-word captions mapped into the MAIN edit's edited timeline."""
    caps: list[dict] = []
    acc = 0.0
    for s, e in keep_ranges:
        for w in words:
            if s <= float(w.s) <= e:
                caps.append({
                    "w": w.w,
                    "start_s": round(acc + (float(w.s) - s), 3),
                    "end_s": round(acc + (float(w.e) - s), 3),
                })
        acc += e - s
    return caps


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.post("/jobs")
async def create_podcast_job(
    video: Optional[UploadFile] = File(None),
    asset_id: Optional[int] = Form(None),
    transcript_json: Optional[str] = Form(None),
    stt_provider: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    silence_threshold_ms: float = Form(450.0),
    renderer: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Create a single-cam podcast job. Provide EITHER a multipart ``video``
    file OR an ``asset_id``. Optionally supply ``transcript_json`` to skip STT.
    """
    if not video and not asset_id:
        raise HTTPException(400, "provide either a video file or asset_id")

    if renderer is not None and renderer not in ("ffmpeg", "remotion"):
        raise HTTPException(400, "renderer must be 'ffmpeg' or 'remotion'")

    # Fail upfront (not minutes later on the worker thread) when there is no
    # transcript AND the STT path can't run: Deepgram is the only auto-STT
    # provider on this box (see _transcribe_via_stt / _default_stt_provider),
    # and it needs DEEPGRAM_API_KEY.
    if not transcript_json:
        prov = stt_provider or _default_stt_provider()
        if prov != "deepgram" or not os.environ.get("DEEPGRAM_API_KEY", "").strip():
            raise HTTPException(
                400,
                "no STT provider available on this box (set DEEPGRAM_API_KEY) "
                "— supply transcript_json instead",
            )

    # Resolve the effective renderer now (explicit arg > env > 'ffmpeg') so we
    # can echo the chosen backend back to the caller immediately.
    from pipeline_core.podcast.render import _select_renderer
    chosen_renderer = _select_renderer(renderer)

    key = f"pod:{uuid.uuid4().hex[:12]}"
    # The colon in `key` is the API/URL/status-store identity, but Windows
    # disallows ':' in a path component (WinError 267), so derive a
    # filesystem-safe directory name for the on-disk output. The /media URLs are
    # computed from this actual path (see _media_url_for), so they stay correct.
    out_dir = str(settings.output_root / "podcast" / key.replace(":", "_"))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Resolve the source path.
    if video is not None:
        # SANITIZED name only — video.filename is client-controlled (path
        # traversal / absolute-path override otherwise).
        try:
            name = _safe_upload_name(video.filename)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        src = os.path.join(out_dir, name)
        # Stream to disk in 8MB chunks — never buffer the whole upload
        # (podcast sources are multi-GB) in memory.
        written = 0
        with open(src, "wb") as fh:
            while chunk := await video.read(8 << 20):
                fh.write(chunk)
                written += len(chunk)
        if not written:
            try:
                os.unlink(src)
            except OSError:
                pass
            raise HTTPException(400, "uploaded video is empty")
    else:
        asset = (
            db.query(models.UserAsset)
            .filter(models.UserAsset.id == asset_id,
                    models.UserAsset.user_id == user.id)
            .first()
        )
        if not asset:
            raise HTTPException(404, "asset not found")
        src = materialize_asset_locally(asset)
        if not src or not os.path.isfile(src):
            raise HTTPException(400, "asset has no usable local bytes")

    _set_status(key, state="queued", progress=0, message="Queued",
                job_id=None, error="", renderer=chosen_renderer)
    threading.Thread(
        target=_run_podcast_job,
        kwargs=dict(
            key=key, user_id=user.id, source_path=src, out_dir=out_dir,
            transcript_json=transcript_json, stt_provider=stt_provider,
            language=language, silence_threshold_ms=silence_threshold_ms,
            renderer=renderer,
        ),
        daemon=True,
    ).start()
    return {"status": "queued", "key": key, "renderer": chosen_renderer}


@router.get("/jobs/{key}")
def get_podcast_status(
    key: str,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    st = _get_status(key)
    if st.get("state") == "idle":
        # In-memory store lost the key (backend restart) — fall back to the
        # Job row + on-disk results.json.
        st = _status_from_db(key, db)
        if st is None:
            raise HTTPException(404, "unknown job key")
    # Don't leak the traceback field in the light status view.
    light = {k: v for k, v in st.items() if k not in ("results", "traceback")}
    return {"key": key, **light}


@router.get("/jobs/{key}/results")
def get_podcast_results(
    key: str,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
) -> dict:
    st = _get_status(key)
    if st.get("state") == "idle":
        st = _status_from_db(key, db)
        if st is None:
            raise HTTPException(404, "unknown job key")
    if st.get("state") != "done":
        raise HTTPException(409, f"job not done (state={st.get('state')})")
    if st.get("results") is None and st.get("recovered"):
        raise HTTPException(
            409, "job finished before a restart but results.json is missing "
                 "on disk; re-run the job")
    return {"key": key, "results": st.get("results")}


__all__ = ["router"]
