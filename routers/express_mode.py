"""Express Mode router — the auto-publish ("lazy user") tab.

Two endpoints:

  POST /api/express/start
      Accepts a multipart upload + Postiz integration list + AI keys
      + strategy/tone fields. Returns ``{job_id}`` immediately and
      runs the pipeline in a background thread.

  GET  /api/express/status/{job_id}
      Returns the current job state for polling: step, progress %,
      message, log tail, results (when done) or error (when failed).

Auth: every endpoint requires a logged-in Kaizer user. Express jobs
are scoped to the owning user — cross-user reads return 404 to keep
multi-tenant tenancy intact (state.get returns None for foreign jobs).

Why we don't validate every AI key up front: the actual API calls
happen inside the pipeline, where errors are caught + surfaced via
``state.mark_failed``. Validating server-side would mean a probe call
per provider per submit — that's slower and burns quota.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

import auth
import models
from database import get_db
from express import pipeline as express_pipeline
from express import state as job_state


router = APIRouter(prefix="/api/express", tags=["express-mode"])


# Where uploaded videos land while the pipeline is in flight. Cleaned
# up by the pipeline's _cleanup() block when the job finishes (or
# fails). Putting them under the system temp dir keeps backup
# pressure off the project tree.
_UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "kaizer-express-uploads")
os.makedirs(_UPLOAD_DIR, exist_ok=True)


@router.post("/start")
def start_job(
    file:                   UploadFile  = File(...),
    integration_ids:        str         = Form(...),     # JSON array of Postiz integration ids
    mode:                   str         = Form("publish-as-is"),  # publish-as-is | ai-trim | shorts (shorts pending)
    anthropic_api_key:      str         = Form(""),
    transcription_provider: str         = Form("groq"),
    transcription_api_key:  str         = Form(""),
    transcription_base_url: str         = Form(""),
    transcription_model:    str         = Form(""),
    brief:                  str         = Form(""),
    names_hint:             str         = Form(""),
    style_guide:            str         = Form(""),
    language:               str         = Form(""),
    privacy:                str         = Form("private"),
    made_for_kids:          bool        = Form(False),
    title_override:         str         = Form(""),
    description_override:   str         = Form(""),
    tags_override:          str         = Form(""),       # comma-separated
    schedule_at_iso:        str         = Form(""),
    color_grade:            str         = Form("subtle"), # off | subtle | cinematic | news-vivid | warm | cool
    cinematic_edit:         bool        = Form(False),
    panel_color:            str         = Form("#dc2626"), # shorts panel color (red default)
    footer_text:            str         = Form("KAIZER NEWS NETWORK"),
    short_count_override:   str         = Form(""),        # blank = auto-pick from duration
    # Session 3 additions:
    openai_api_key:         str         = Form(""),        # for gpt-image-1
    inset_strategy:         str         = Form("frame"),   # frame | ai (Shorts)
    thumbnail_strategy:     str         = Form("none"),    # none | ai  (AI Trim)
    layout:                 str         = Form("news"),    # news | branded (Shorts)
    logo_corner:            str         = Form("top-right"), # branded layout only
    user:                   models.User = Depends(auth.current_user),
) -> dict:
    """Kick off an Express Mode autopub job.

    Validates inputs cheaply, persists the upload to disk, mints a
    job id, then spawns the pipeline in a daemon thread. Returns
    immediately so the browser doesn't time out — the client polls
    /status until ``status: done|failed``.
    """
    # ── Input validation ──────────────────────────────────────────
    import json
    try:
        iids = json.loads(integration_ids)
        if not isinstance(iids, list) or not all(isinstance(x, str) and x for x in iids):
            raise ValueError("integration_ids must be a non-empty list of strings")
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(400, f"integration_ids invalid: {exc}")

    if mode not in ("publish-as-is", "ai-trim", "shorts"):
        raise HTTPException(
            400,
            f"mode {mode!r} not supported — use publish-as-is, ai-trim, or shorts."
        )

    # Per-request keys are optional. Empty → the express/{claude,whisper,
    # ai_image} modules fall back to ANTHROPIC_API_KEY / GROQ_API_KEY /
    # OPENAI_API_KEY in env. If env is ALSO missing, the pipeline marks
    # the job failed with a clear "ANTHROPIC_API_KEY missing" message
    # so the user knows which key to provide.

    # ── Persist the upload ────────────────────────────────────────
    # Sanitise the filename — drop directory parts, keep extension.
    safe_name = os.path.basename(file.filename or "upload.mp4")
    suffix = os.path.splitext(safe_name)[1].lower() or ".mp4"
    fd, dst = tempfile.mkstemp(suffix=suffix, dir=_UPLOAD_DIR,
                                prefix=f"express-{user.id}-")
    os.close(fd)
    try:
        with open(dst, "wb") as out:
            shutil.copyfileobj(file.file, out)
    except OSError as exc:
        os.unlink(dst)
        raise HTTPException(500, f"could not save upload: {exc}")

    # Parse the comma-separated tag override if provided.
    parsed_tags: Optional[list[str]] = None
    if tags_override.strip():
        parsed_tags = [t.strip() for t in tags_override.split(",") if t.strip()]

    # ── Spin up the job ───────────────────────────────────────────
    jid = job_state.new_job(user_id=user.id)
    job_state.append_log(jid, f"[start] user={user.id} mode={mode} integrations={len(iids)}")

    common_kwargs = dict(
        jid=jid,
        video_path=dst,
        integration_ids=iids,
        brief=brief,
        names_hint=names_hint,
        style_guide=style_guide,
        language=language or None,
        privacy=privacy,
        made_for_kids=bool(made_for_kids),
        title_override=title_override or None,
        description_override=description_override or None,
        tags_override=parsed_tags,
        schedule_at_iso=schedule_at_iso or None,
        anthropic_api_key=anthropic_api_key,
        transcription_provider=transcription_provider,
        transcription_api_key=transcription_api_key,
        transcription_base_url=transcription_base_url or None,
        transcription_model=transcription_model or None,
    )

    # Parse the short-count override only when meaningful.
    shorts_n: Optional[int] = None
    if short_count_override.strip():
        try:
            shorts_n = max(1, min(8, int(short_count_override.strip())))
        except ValueError:
            shorts_n = None

    def _runner():
        if mode == "ai-trim":
            express_pipeline.run_ai_trim(
                **common_kwargs,
                color_grade_preset=color_grade,
                cinematic_edit=bool(cinematic_edit),
                openai_api_key=openai_api_key,
                thumbnail_strategy=thumbnail_strategy or "none",
            )
        elif mode == "shorts":
            express_pipeline.run_shorts(
                **common_kwargs,
                color_grade_preset=color_grade,
                cinematic_edit=bool(cinematic_edit),
                panel_color=panel_color or "#dc2626",
                footer_text=footer_text or "KAIZER NEWS NETWORK",
                short_count_override=shorts_n,
                openai_api_key=openai_api_key,
                inset_strategy=inset_strategy or "frame",
                layout=layout or "news",
                logo_corner=logo_corner or "top-right",
            )
        else:
            express_pipeline.run_publish_as_is(**common_kwargs)
        # Pipeline owns cleanup of audio + the video upload. If for
        # any reason it didn't, fall through to a defensive delete.
        try:
            if os.path.isfile(dst):
                os.unlink(dst)
        except OSError:
            pass

    threading.Thread(target=_runner, name=f"express-{jid}", daemon=True).start()

    return {
        "ok":      True,
        "job_id":  jid,
        "status":  "queued",
        "message": "Express Mode pipeline starting…",
    }


@router.get("/jobs")
def list_jobs(
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Return all Express Mode jobs owned by the current user (newest
    first, max 50). Each entry is a summary — only the fields the
    history UI needs. The full /status/{job_id} endpoint is still the
    source of truth for inspecting any one job.

    State is in-memory with a 6 h TTL, so this list survives page
    reloads but not backend restarts.
    """
    jobs = job_state.list_for_user(user.id, limit=50)
    summaries: list[dict] = []
    for j in jobs:
        results = j.get("results") or {}
        summaries.append({
            "id":         j["id"],
            "status":     j.get("status", "unknown"),
            "step":       j.get("step", ""),
            "progress":   int(j.get("progress", 0)),
            "message":    (j.get("message", "") or "")[:140],
            "mode":       results.get("mode") if isinstance(results, dict) else None,
            "title":      results.get("title") if isinstance(results, dict) else None,
            "created_at": j.get("created_at"),
            "updated_at": j.get("updated_at"),
        })
    return {"jobs": summaries, "count": len(summaries)}


@router.get("/key-status")
def key_status(
    _: models.User = Depends(auth.current_user),
) -> dict:
    """Report which API providers have a server-side env-fallback key
    set. The UI uses this to show a "server fallback active" badge so
    operators can submit jobs without re-typing keys when ``.env``
    already has them. Returns only booleans — never the keys.
    """
    return {
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY", "").strip()),
        "groq":      bool(os.environ.get("GROQ_API_KEY", "").strip()),
        "openai":    bool(os.environ.get("OPENAI_API_KEY", "").strip()),
        "postiz":    bool(os.environ.get("POSTIZ_API_KEY", "").strip()),
        "anthropic_model": os.environ.get("ANTHROPIC_MODEL", "").strip() or "claude-sonnet-4-6",
    }


@router.get("/status/{job_id}")
def get_status(
    job_id: str,
    user:   models.User = Depends(auth.current_user),
) -> dict:
    """Return the current state of one Express Mode job. The frontend
    polls this every 2-5 s while the bar isn't at 100%.

    Returns 404 if the job is unknown, expired (>6 h old), OR belongs
    to a different user — by design these collapse into one "not
    found" response so an attacker can't enumerate other users' job
    ids.
    """
    j = job_state.get(job_id, user_id=user.id)
    if not j:
        raise HTTPException(404, "job not found or expired")

    # Tail of recent log lines — last 50 is enough for the UI to show
    # what's happening without flooding the response payload.
    log = j.get("log") or []
    return {
        "id":         j["id"],
        "status":     j.get("status", "unknown"),
        "step":       j.get("step", ""),
        "progress":   int(j.get("progress", 0)),
        "message":    j.get("message", ""),
        "log_tail":   log[-50:],
        "results":    j.get("results"),
        "error":      j.get("error"),
        "created_at": j.get("created_at"),
        "updated_at": j.get("updated_at"),
    }


@router.get("/integrations")
def list_postiz_integrations(
    user: models.User = Depends(auth.current_user),
) -> dict:
    """Proxy to Postiz's /integrations so the UI can render a channel
    picker without the frontend needing to know the Postiz token. The
    token lives only in backend env (POSTIZ_API_KEY)."""
    from clients import postiz as postiz_client
    if not postiz_client.is_enabled():
        raise HTTPException(
            503,
            "Postiz isn't configured on this server. Set POSTIZ_API_KEY in .env."
        )
    try:
        items = postiz_client.list_integrations()
    except postiz_client.PostizAuthError as exc:
        raise HTTPException(401, f"Postiz auth failed: {exc}")
    except postiz_client.PostizError as exc:
        raise HTTPException(502, f"Postiz error: {exc}")
    return {"integrations": items}
