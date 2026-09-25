"""Upload endpoints — enqueue publish, list queue, detail, cancel, progress SSE."""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from database import SessionLocal, get_db
import models
import auth
from youtube import quota


router = APIRouter(prefix="/api", tags=["youtube-upload"])


# ─── Schemas ──────────────────────────────────────────────────────────────

class PublishRequest(BaseModel):
    # Legacy single-target support — kept so existing frontend still works.
    channel_id:     Optional[int] = None
    # New: fan-out to multiple style profiles in one call.  Each entry creates
    # its own UploadJob queued independently.  If set, `channel_id` is ignored.
    channel_ids:    Optional[List[int]] = None
    privacy_status: str = "private"
    publish_at:     Optional[datetime] = None
    use_seo:        bool = True
    # "short" → append #Shorts hashtag (YouTube auto-classifies vertical ≤60s
    # clips with that hashtag as a Short). "video" → standard upload.
    publish_kind:   str = "video"
    # When set, force every destination to use THIS channel's SEO variant
    # from clip.seo_variants (overrides the per-destination auto-match).
    seo_variant_override: Optional[int] = None
    # When set, read clip.seo / seo_variants from THIS sibling clip instead of
    # the clip being published. Used by the bulk-publish flow so a video's
    # 5 clips can all share the SEO generated for one of them. The clip's own
    # SEO still wins when present — this only fires when the publishing clip
    # has no SEO of its own. Source must belong to the same job.
    seo_source_clip_id: Optional[int] = None
    # Per-destination override map: { "<dest_channel_id>": <variant_channel_id> }.
    # Wins over `seo_variant_override` for destinations it specifies.  Lets a
    # user publish Auto Wala with "Suman TV Live" voice AND Cyber Sphere with
    # "Personal 2" voice in the same click.
    variant_by_channel: Optional[dict[str, int]] = None
    # Optional overrides if use_seo is False or the user wants to tweak at publish time
    title:          Optional[str] = Field(None, max_length=150)
    description:    Optional[str] = None
    tags:           Optional[List[str]] = None
    category_id:    Optional[str] = None
    made_for_kids:  bool = False
    # Per-publish upload route override.  "postiz" | "kaizer" |
    # "native_rtmp" | null.  When null we fall through to
    # Channel.upload_provider and then to the system-wide default —
    # so most users never set this; it's exposed primarily for one-shot
    # overrides and side-by-side comparison runs.
    upload_provider: Optional[str] = None

    # Per-CHANNEL override map: { "12": "native_rtmp", "13": "kaizer" }.
    # Wins over ``upload_provider`` (the batch-wide value) so a single
    # publish call can route different channels through different
    # providers.  Used by the bulk publish modal to honour each
    # channel's row-level selector.  Keys are stringified channel IDs
    # to match JSON conventions.  Missing keys fall through to the
    # batch value, then the per-channel default, then system default.
    upload_provider_by_channel: Optional[dict[str, Optional[str]]] = None
    # Per-CHANNEL explicit Postiz integration binding: { "17": "cmqc5..." }.
    # The publish modal lists Postiz integrations from the LIVE Postiz org
    # (postiz_client.list_integrations), which is a superset of what's mapped
    # in our PostizIntegration table. When the admin name-matches a channel to
    # one of those live integrations, the frontend sends the resolved id here
    # so dispatch can route upload_path='postiz' WITHOUT relying on the DB
    # table (which may have no row for a Postiz-only channel). Dispatch also
    # persists it onto Channel.postiz_integration_id so future publishes +
    # the production auto-fallback resolve it natively. Keys = stringified
    # channel IDs.
    postiz_integration_by_channel: Optional[dict[str, Optional[str]]] = None
    # Branding mode for this publish: 'per_channel' (overlay each channel's
    # logo+watermark+socials at upload — the default) | 'as_is' (the video is
    # already branded; upload it verbatim to every channel, no overlay).
    brand_mode: Optional[str] = "per_channel"
    # Logo/watermark placement when overlaying: 'template' (use the template's
    # marked slot — the default) | 'channel' (use this channel's own position).
    brand_placement: Optional[str] = "template"

    # ── Thumbnail selection (videos only; shorts never get a thumbnail) ──
    # Global mode for every destination unless overridden per-channel below:
    #   None       → legacy behaviour (rendered thumbnail, or a clip.meta
    #                 publish_thumbnail_key if Quick Publish stored one)
    #   "rendered" → use the pipeline-rendered thumbnail
    #   "custom"   → use custom_thumbnail_r2_key (staged via /thumbnail/stage)
    #   "none"     → upload with no custom thumbnail
    thumbnail_mode: Optional[str] = None
    # R2 key of a staged custom thumbnail applied to ALL destinations when
    # thumbnail_mode == "custom". Produced by POST /clips/{id}/thumbnail/stage.
    custom_thumbnail_r2_key: Optional[str] = None
    # Per-CHANNEL thumbnail override: { "12": "rendered" | "none" |
    # "custom:<r2_key>" | "<r2_key>" }. Wins over thumbnail_mode for the
    # channels it names. Lets each channel publish with its own thumbnail,
    # mirroring per-channel SEO. Keys are stringified channel IDs.
    thumbnail_by_channel: Optional[dict[str, str]] = None
    # Per-CHANNEL YouTube publish-setting OVERRIDES for THIS upload:
    #   { "12": {"category_id": "25", "language": "te", "playlist_id": "PL..",
    #            "license": "youtube"|"creativeCommon", "made_for_kids": false} }
    # Any field omitted/null = no override for that channel → the channel's
    # saved yt_* default is used. Each present value WINS over the default for
    # this publish only (the saved Style-Profile default is not changed).
    publish_settings_by_channel: Optional[dict[str, dict]] = None

    @field_validator("brand_mode")
    @classmethod
    def _brand_mode(cls, v):
        v = (v or "per_channel").strip().lower()
        if v not in {"per_channel", "as_is"}:
            raise ValueError("brand_mode must be 'per_channel' or 'as_is'")
        return v

    @field_validator("brand_placement")
    @classmethod
    def _brand_placement(cls, v):
        v = (v or "template").strip().lower()
        if v not in {"template", "channel"}:
            raise ValueError("brand_placement must be 'template' or 'channel'")
        return v

    @field_validator("upload_provider")
    @classmethod
    def _upload_provider(cls, v):
        if v is None or v == "":
            return None
        v = str(v).strip().lower()
        if v not in {"postiz", "kaizer", "native_rtmp"}:
            raise ValueError("upload_provider must be 'postiz', 'kaizer', 'native_rtmp', or null")
        return v

    @field_validator("upload_provider_by_channel")
    @classmethod
    def _upload_provider_map(cls, v):
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("upload_provider_by_channel must be a dict")
        cleaned: dict[str, Optional[str]] = {}
        for k, vv in v.items():
            if vv is None or vv == "":
                cleaned[str(k)] = None
                continue
            vs = str(vv).strip().lower()
            if vs not in {"postiz", "kaizer", "native_rtmp"}:
                raise ValueError(
                    f"upload_provider_by_channel[{k!r}] must be 'postiz', "
                    f"'kaizer', 'native_rtmp', or null"
                )
            cleaned[str(k)] = vs
        return cleaned or None

    @field_validator("postiz_integration_by_channel")
    @classmethod
    def _postiz_iid_map(cls, v):
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("postiz_integration_by_channel must be a dict")
        cleaned: dict[str, Optional[str]] = {}
        for k, vv in v.items():
            sv = (str(vv).strip() if vv is not None else "")
            cleaned[str(k)] = sv or None
        return cleaned or None

    @field_validator("privacy_status")
    @classmethod
    def _privacy(cls, v: str) -> str:
        v = (v or "").lower().strip()
        if v not in ("public", "private", "unlisted"):
            raise ValueError("privacy_status must be public | private | unlisted")
        return v

    @field_validator("publish_kind")
    @classmethod
    def _publish_kind(cls, v: str) -> str:
        v = (v or "video").lower().strip()
        if v not in ("short", "video"):
            raise ValueError("publish_kind must be 'short' or 'video'")
        return v

    @field_validator("publish_at")
    @classmethod
    def _publish_at(cls, v: Optional[datetime]) -> Optional[datetime]:
        if v is None:
            return None
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        if v <= datetime.now(timezone.utc):
            raise ValueError("publish_at must be in the future")
        return v


# ─── Helpers ──────────────────────────────────────────────────────────────

def _to_dict(job: models.UploadJob) -> dict:
    clip = job.clip
    channel = job.channel
    return {
        "id":             job.id,
        "clip_id":        job.clip_id,
        "channel_id":     job.channel_id,
        "channel_name":   channel.name if channel else None,
        "clip_filename":  clip.filename if clip else None,
        "clip_thumb_url": (
            (getattr(clip, "thumb_storage_url", "") if clip else "")
            or (f"/api/file/?path={clip.thumb_path}" if clip and clip.thumb_path else "")
        ),
        "status":         job.status,
        "privacy_status": job.privacy_status,
        "publish_kind":   job.publish_kind or "video",
        "publish_at":     job.publish_at.isoformat() if job.publish_at else None,
        "title":          job.title,
        "description":    job.description,
        "tags":           list(job.tags or []),
        "category_id":    job.category_id,
        "made_for_kids":  bool(job.made_for_kids),
        "video_id":       job.video_id or None,
        "video_url":      f"https://youtu.be/{job.video_id}" if job.video_id else None,
        # Per-job override (null = resolved at worker time from
        # channel/system default).  Surfaced so the Uploads page
        # can display "via Postiz" / "via Native YouTube" on each row.
        "upload_provider": getattr(job, "upload_provider", None) or None,
        "bytes_uploaded": job.bytes_uploaded or 0,
        "bytes_total":    job.bytes_total or 0,
        "progress_pct":   round(100 * (job.bytes_uploaded or 0) / max(job.bytes_total or 1, 1), 1),
        "attempts":       job.attempts or 0,
        "last_error":     job.last_error or "",
        "log":            job.log or "",
        "created_at":     job.created_at.isoformat() if job.created_at else None,
        "updated_at":     job.updated_at.isoformat() if job.updated_at else None,
    }


def _compose_metadata(
    clip: models.Clip,
    channel: models.Channel,
    payload: PublishRequest,
) -> tuple[str, str, list[str]]:
    """Resolve title / description / tags for an upload to `channel`.

    Content + Brand Overlay model:
      - `clip.seo` holds the GENERIC (channel-agnostic) SEO produced by the
        generator.
      - `seo.composer.compose(generic, channel)` overlays the destination's
        name suffix, mandatory hashtags, fixed tags, and footer.
      - Result is the exact metadata uploaded to YouTube for this destination.

    Legacy `clip.seo_variants` is still read as a fallback for older clips
    that were generated before the refactor (they stored per-channel full
    SEO blobs instead of a single generic one).  When present, we apply the
    variant AS-IS for that destination (no composer re-pass — it's already
    channel-scoped from the old flow).

    Shorts handling is performed inside composer.compose().
    """
    # Manual override wins everything — user typed title/desc in PublishModal
    if not payload.use_seo:
        title = (payload.title or clip.text or f"Kaizer clip #{clip.id}")[:100]
        description = payload.description or ""
        tags = payload.tags or []
        if payload.publish_kind == "short":
            if "shorts" not in [t.lower() for t in tags]:
                tags = ["shorts", *tags]
        from youtube.uploader import sanitize_tags
        return title, description, sanitize_tags(tags)

    # ─── Try generic SEO + brand overlay (new path) ───
    generic: dict = {}
    if clip.seo:
        try:
            g = json.loads(clip.seo)
            if isinstance(g, dict) and g.get("title"):
                generic = g
        except (ValueError, TypeError):
            pass

    if generic:
        from seo.composer import compose
        composed = compose(generic, channel, publish_kind=payload.publish_kind)
        title = (payload.title or composed["title"])[:100]
        description = payload.description or composed["description"]
        tags = payload.tags if payload.tags is not None else composed["keywords"]
        from youtube.uploader import sanitize_tags
        return title, description, sanitize_tags(tags)

    # ─── Legacy fallback: per-channel variants from old generation runs ───
    try:
        variants = json.loads(clip.seo_variants or "{}")
    except (ValueError, TypeError):
        variants = {}
    if not isinstance(variants, dict):
        variants = {}

    per_dest = payload.variant_by_channel or {}
    dest_key = str(channel.id)
    legacy_seo: dict = {}
    if dest_key in per_dest:
        v_key = str(per_dest[dest_key])
        if v_key in variants:
            legacy_seo = variants[v_key] or {}
    if not legacy_seo and payload.seo_variant_override is not None:
        key = str(payload.seo_variant_override)
        if key in variants:
            legacy_seo = variants[key] or {}
    if not legacy_seo and dest_key in variants:
        legacy_seo = variants[dest_key] or {}

    title = (payload.title
             or legacy_seo.get("title")
             or clip.text
             or f"Kaizer clip #{clip.id}")[:100]
    description = payload.description or legacy_seo.get("description") or ""
    tags = payload.tags if payload.tags is not None else (legacy_seo.get("keywords") or [])

    if payload.publish_kind == "short":
        shorts_tag = "#Shorts"
        if shorts_tag.lower() not in title.lower():
            candidate = f"{title} {shorts_tag}"
            if len(candidate) <= 100:
                title = candidate
            elif shorts_tag.lower() not in description.lower():
                description = (shorts_tag + "\n\n" + description).strip()
        if "shorts" not in [t.lower() for t in tags]:
            tags = ["shorts", *tags]

    from youtube.uploader import sanitize_tags
    return title, description, sanitize_tags(tags)


# ─── Endpoints ────────────────────────────────────────────────────────────

from rate_limit import rate_limited as _rate_limited


# ─── Legacy → v2 redirect (Phase 3 cutover) ───────────────────────────────
#
# When ``KAIZER_NEW_PUBLISH_PATH=1``, every legacy
# ``POST /api/clips/:id/publish`` request is translated into a
# ``PublishTaskRequest`` and routed through ``services.fanout.create_publish_task``.
# Response shape stays the same as the legacy path (single dict OR
# ``{jobs: [...]}`` for fan-out) so existing frontend code keeps working
# without a deploy.
#
# Default (``=0``) is unchanged — legacy logic still runs unmodified.
#
# Mapping rules (Decision 10 — opaque caller-supplied version strings):
#   - clip → its Job → MasterVideo (lookup or synthesise transiently)
#   - channel_ids → targets[N]
#   - publish_kind, privacy_status, publish_at → carried per-target
#   - upload_provider:
#         'kaizer'      -> direct
#         'native_rtmp' -> rtmp
#         'postiz'      -> direct (Postiz uses the Direct path under the hood)
#         null/missing  -> channel.upload_provider then default 'direct'
#   - SEO controls: seo_version + metadata_version are stubbed from the
#     clip + channel ids (stable, deterministic) so re-publish dedupes.
#
# MasterVideo handling: if the clip's Job already has a MasterVideo row
# we reuse it. Otherwise we synthesise one with ``clean_master=False``
# (legacy render baked the logo). The Branding worker detects
# ``clean_master=False`` and SKIPS the logo overlay — only the text
# watermark gets applied — per services/branding.py docstring.
# Decision 12 covers this transitional behaviour.
def _legacy_publish_to_v2_enabled() -> bool:
    """Re-read every request so an ops flip doesn't require a restart."""
    import os as _os
    return (_os.environ.get("KAIZER_NEW_PUBLISH_PATH", "0") or "0").strip() == "1"


def _ensure_rendered_thumb_key(clip) -> Optional[str]:
    """Upload the clip's locally-rendered thumbnail to storage and return its
    key. The V4 render leaves the thumbnail on local disk (``clip.thumb_path``,
    e.g. ``…/job_<id>/bulletin_thumb.jpg``) but NOT in R2 — and the dispatch
    layer only pushes a thumbnail to YouTube when an ``thumbnail_r2_key`` is
    present. Without this the "rendered" thumbnail silently never reaches
    YouTube. Returns the storage key, or None if there's no usable file.
    """
    try:
        import os as _os
        tp = (getattr(clip, "thumb_path", "") or "").strip()
        if not tp or not _os.path.exists(tp):
            return None
        ext = "png" if tp.lower().endswith(".png") else "jpg"
        ct = "image/png" if ext == "png" else "image/jpeg"
        from pipeline_core.storage import get_storage_provider
        # Deterministic key (overwrites on re-publish — the rendered thumb for
        # a clip is stable, so we don't accumulate duplicates).
        key = f"publish_thumbs/rendered/clip_{int(clip.id)}.{ext}"
        stored = get_storage_provider().upload(tp, key, content_type=ct)
        return stored.key
    except Exception as exc:
        print(f"[uploads] rendered-thumb upload failed for clip {getattr(clip,'id','?')}: {exc}")
        return None


def _map_legacy_upload_provider(provider: Optional[str]) -> str:
    """Map the legacy ``upload_provider`` enum onto the v2 ``upload_path`` enum.

    Returns 'direct' or 'rtmp'.
    """
    if provider is None:
        return "direct"
    p = str(provider).strip().lower()
    if p == "native_rtmp":
        return "rtmp"
    # 'kaizer' (native Direct) and 'postiz' (proxy through Postiz, which
    # uses videos.insert under the hood) both map to direct on the new
    # path. Postiz parity is preserved at the application layer; v2
    # doesn't model the proxy as a distinct path.
    return "direct"


def _resolve_postiz_integration_for_channel(db, user, ch):
    """Best-effort link a Kaizer channel to a connected Postiz integration.

    When a publish selects Postiz for a channel that has no explicit binding,
    match it to one of the user's TEAM-owned Postiz integrations by name (or
    handle/identifier) — same name means the same channel, e.g. the Kaizer
    channel "Kaizer 5" maps to the Postiz integration named "Kaizer 5".
    Returns the integration_id or None when nothing matches.
    """
    try:
        from services.postiz_scope import team_user_ids
        team = team_user_ids(db, user.id)
    except Exception:
        team = {user.id}
    rows = (db.query(models.PostizIntegration)
              .filter(models.PostizIntegration.user_id.in_(team)).all())
    if not rows:
        return None
    nm = (getattr(ch, "name", "") or "").strip().lower()
    hd = (getattr(ch, "handle", "") or "").strip().lstrip("@").lower()
    for r in rows:
        if nm and (r.name or "").strip().lower() == nm:
            return r.integration_id
    for r in rows:
        rid = (r.identifier or "").strip().lstrip("@").lower()
        if hd and rid and rid == hd:
            return r.integration_id
    return None


def _legacy_to_v2_redirect(
    db: Session,
    user: "models.User",
    clip_id: int,
    payload: PublishRequest,
):
    """Translate a legacy PublishRequest into a v2 PublishTaskRequest
    and call ``services.fanout.create_publish_task``.

    Returns the same response shape the legacy ``publish_clip`` returns:
      * single UploadJob dict when one target was requested, OR
      * ``{"jobs": [...], "count": n}`` for fan-out.

    Raises HTTPException for the same validation conditions the legacy
    code raises (404 on missing clip/channel, 409 on un-linked YouTube
    token, 422 on no targets / no rendered file, etc.) — so the
    frontend sees identical error semantics.
    """
    import logging as _logging
    import os as _os
    from services import fanout as _fanout
    from services import credits as _credits
    from services import idempotency as _idempotency

    _log = _logging.getLogger("kaizer.publish.legacy_v2")

    # ── 1. Validate the clip (mirror legacy 404/422 surface) ─────────────
    clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    if not clip.file_path:
        raise HTTPException(
            status_code=422,
            detail="Clip has no rendered file — run the pipeline first",
        )

    # ── 2. Resolve target list (same dedupe-preserving order) ────────────
    raw_ids: list[int] = []
    if payload.channel_ids:
        raw_ids.extend(payload.channel_ids)
    if payload.channel_id is not None:
        raw_ids.append(payload.channel_id)
    seen: set[int] = set()
    target_ids = [i for i in raw_ids if not (i in seen or seen.add(i))]
    if not target_ids:
        raise HTTPException(
            status_code=422,
            detail="At least one destination must be selected.",
        )

    # ── 3. Load channels + validate ownership/OAuth (mirror legacy) ──────
    channels = (
        db.query(models.Channel).filter(models.Channel.id.in_(target_ids)).all()
    )
    by_id = {c.id: c for c in channels}
    missing = [i for i in target_ids if i not in by_id]
    if missing:
        raise HTTPException(
            status_code=404, detail=f"Profile(s) not found: {missing}"
        )
    for cid in target_ids:
        ch = by_id[cid]
        if not ch.oauth_token or not ch.oauth_token.refresh_token_enc:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Style profile '{ch.name}' is not linked to YouTube. "
                    "Open Style Profiles → Link my YT."
                ),
            )

    # ── 4. Resolve / synthesise MasterVideo (PER CLIP) ───────────────────
    # One MasterVideo per CLIP, not per job. A V4 job emits 1 Full Video
    # + N shorts — each a distinct file. Keying the master by clip means
    # publishing the bulletin and a short no longer collapse onto a
    # single master (which used to upload the SAME file twice).
    job_id = int(getattr(clip, "job_id", 0) or 0)
    master = (
        db.query(models.MasterVideo)
        .filter(models.MasterVideo.clip_id == int(clip.id))
        .first()
    )
    # Legacy adopt: a pre-fix per-job master whose r2_key already points
    # at THIS clip's file (the rows the repair script healed). Claim it
    # for this clip so we don't orphan a real uploaded artifact.
    if master is None and job_id:
        legacy = (
            db.query(models.MasterVideo)
            .filter(
                models.MasterVideo.source_upload_id == job_id,
                models.MasterVideo.clip_id.is_(None),
                models.MasterVideo.r2_key.like(f"%/clip/{int(clip.id)}/%"),
            )
            .first()
        )
        if legacy is None:
            legacy = (
                db.query(models.MasterVideo)
                .filter(
                    models.MasterVideo.source_upload_id == job_id,
                    models.MasterVideo.clip_id.is_(None),
                    models.MasterVideo.r2_key.like(f"%clip_{int(clip.id)}/%"),
                )
                .first()
            )
        if legacy is not None:
            legacy.clip_id = int(clip.id)
            db.add(legacy)
            db.commit()
            master = legacy

    # Raw uploads (Quick Publish) are CLEAN user videos — nothing was
    # baked at render time, so the Branding worker must apply the
    # per-channel logo AND watermark. Detect via frame_type or the
    # meta flag raw-upload sets.
    _is_raw_upload = (getattr(clip, "frame_type", "") or "") == "raw_upload"
    if not _is_raw_upload:
        try:
            import json as _json
            _meta = _json.loads(clip.meta) if clip.meta else {}
            _is_raw_upload = bool(
                isinstance(_meta, dict) and _meta.get("raw_upload")
            )
        except Exception:
            pass

    # An existing master whose key is the old placeholder shape is
    # BROKEN for the branding worker (it can't download bytes that were
    # never uploaded — every dispatch fails with StorageWriteError and
    # the job burns its retries). Repair it on this publish instead of
    # re-dispatching into the same wall.
    _needs_repair = bool(
        master is not None
        and (master.r2_key or "").startswith("legacy/clip/")
    )
    if _needs_repair:
        try:
            from pipeline_core.storage import get_storage_provider as _gsp
            _needs_repair = not _gsp().exists(master.r2_key)
        except Exception:
            _needs_repair = True  # can't verify → resolve a real key

    if master is None or _needs_repair:
        # Synthesise (or repair) the transitional MasterVideo row.
        # clean_master semantics:
        #   * Raw uploads (Quick Publish): True — the user's finished
        #     video has NO Kaizer branding; Branding applies the full
        #     per-channel logo + watermark pass.
        #   * Rendered clips with KAIZER_CLEAN_MASTER=1 (post-cutover
        #     V4): ALSO True — the render skipped the logo bake, so
        #     Branding owns the logo pass.
        #   * Pre-cutover legacy renders (flag off): False — logo is
        #     already baked; Branding applies text-only.
        try:
            import os
            file_path = (clip.file_path or "").strip()
            local_ok = bool(file_path and os.path.exists(file_path))
            storage_key = (getattr(clip, "storage_key", "") or "").strip()

            # The Branding worker downloads the master via the storage
            # provider, so a REAL key is required. The raw-upload
            # endpoint mirrors to storage and (on R2) deletes the local
            # file — so "local file missing" is NORMAL there. Only fail
            # when we have neither bytes source.
            if not local_ok and not storage_key:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Clip file unavailable (no local file at "
                        f"{file_path!r} and no storage key). "
                        "Re-render or re-upload before publishing."
                    ),
                )

            file_bytes = (
                int(os.path.getsize(file_path)) if local_ok else 1
            )
            duration = float(getattr(clip, "duration", None) or 0.0)
            # Best-effort dimensions: probe only when a local file
            # exists; the v2 path uses these for analytics only.
            width, height = 1920, 1080
            if local_ok:
                try:
                    import subprocess
                    r = subprocess.run(
                        [
                            "ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height",
                            "-of", "csv=p=0:s=x", file_path,
                        ],
                        capture_output=True, text=True, timeout=10,
                    )
                    if r.returncode == 0 and r.stdout.strip():
                        parts = r.stdout.strip().split("x")
                        if len(parts) == 2:
                            width = int(parts[0])
                            height = int(parts[1])
                except Exception:
                    pass

            # Resolve the master's storage key — for EVERY clip kind.
            # Precedence:
            #   1. clip.storage_key — an already-mirrored object.
            #   2. Upload the local rendered file NOW. This is the
            #      normal path for V4 clips on a local-storage dev box;
            #      the placeholder-key shape is reused as the real key
            #      so previously-synthesised rows heal automatically.
            r2_key = storage_key
            if not r2_key and local_ok:
                from pipeline_core.storage import get_storage_provider
                upload_key = f"legacy/clip/{clip.id}/master.mp4"
                if _is_raw_upload:
                    _user_id = int(getattr(user, "id", 0) or 0)
                    upload_key = (
                        f"raw_uploads/{_user_id}/clip_{clip.id}/master.mp4"
                    )
                stored = get_storage_provider().upload(
                    file_path, upload_key, content_type="video/mp4",
                )
                r2_key = stored.key
                clip.storage_key = stored.key
                clip.storage_url = stored.url
                db.add(clip)
                # Persist the template's logo/watermark slot sidecar next to the
                # master so the publish branding worker can drop each channel's
                # logo at the template's designed spot (else it defaults to a
                # corner, leaving the marked slot empty). No-op when absent.
                try:
                    _sc = file_path + ".slots.json"
                    if os.path.isfile(_sc):
                        get_storage_provider().upload(
                            _sc, upload_key + ".slots.json",
                            content_type="application/json",
                        )
                except Exception:
                    pass
            if not r2_key:
                # storage_key absent AND no local file was caught above;
                # absent AND upload raised lands in the except below.
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Could not place clip_id={clip.id} into storage "
                        "for publishing. Re-render or re-upload."
                    ),
                )

            _clean = bool(
                _is_raw_upload
                or (os.environ.get("KAIZER_CLEAN_MASTER", "0") or "0").strip() == "1"
            )
            _pipe_ver = (
                "quick_publish" if _is_raw_upload
                else ("v4_clean" if _clean else "v4_legacy_branded")
            )
            if master is None:
                master = models.MasterVideo(
                    source_upload_id=job_id,
                    clip_id=int(clip.id),
                    r2_key=r2_key,
                    duration_seconds=max(0.1, duration),
                    bytes=max(1, file_bytes),
                    width=width,
                    height=height,
                    status="ready",
                    pipeline_version=_pipe_ver,
                    clean_master=_clean,
                )
                db.add(master)
            else:
                # Repair-in-place: future dispatches of EXISTING jobs
                # read this row, so healing it un-bricks their retries.
                master.r2_key = r2_key
                master.bytes = max(1, file_bytes)
                master.duration_seconds = max(0.1, duration)
                master.clean_master = _clean
                master.pipeline_version = _pipe_ver
                master.status = "ready"
                db.add(master)
            db.commit()
            db.refresh(master)
            _log.info(
                "legacy_to_v2_redirect: %s MasterVideo id=%s for "
                "clip_id=%s (clean_master=%s, r2_key=%s)",
                ("repaired" if _needs_repair else "synthesised"),
                master.id, clip.id, master.clean_master, r2_key,
            )
        except HTTPException:
            raise
        except Exception as exc:
            db.rollback()
            _log.exception(
                "legacy_to_v2_redirect: failed to synthesise MasterVideo "
                "for clip_id=%s: %s", clip.id, exc,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Unable to materialise MasterVideo for clip_id={clip.id}: {exc}",
            )

    # ── 5. Build FanoutTargets ───────────────────────────────────────────
    # Stable version strings for idempotency:
    #   brand_profile_version = "legacy_clip_{clip_id}"
    #   seo_version = "clip_{clip_id}_v{seo_variant_override or 0}"
    #   metadata_version = sha256 of payload metadata signature
    import hashlib as _hashlib

    # Fold brand_mode into the brand version so the idempotency key differs
    # when the same clip is re-published to the same channel under a different
    # branding mode (per_channel vs as_is) — otherwise the second publish
    # would be deduped as a no-op.
    brand_v = (
        f"legacyclip{clip.id}_{payload.brand_mode or 'per_channel'}"
        f"_{payload.brand_placement or 'template'}"
    )[:64]
    seo_v_base = (
        f"{clip.id}_v{payload.seo_variant_override or 0}_"
        f"{payload.seo_source_clip_id or 0}_"
        f"{int(bool(payload.use_seo))}"
    )
    seo_v = _hashlib.sha256(seo_v_base.encode("utf-8")).hexdigest()[:24]
    metadata_signature = (
        f"{payload.title or ''}|"
        f"{payload.description or ''}|"
        f"{','.join(payload.tags or [])}|"
        f"{payload.category_id or '25'}|"
        f"{int(payload.made_for_kids)}|"
        f"{payload.privacy_status}|"
        f"{payload.publish_at.isoformat() if payload.publish_at else ''}|"
        f"{payload.publish_kind}"
    )
    metadata_v = _hashlib.sha256(metadata_signature.encode("utf-8")).hexdigest()[:24]

    targets: list[_fanout.FanoutTarget] = []
    for cid in target_ids:
        ch = by_id[cid]
        # Resolve effective upload_provider per-channel, mirroring legacy
        # precedence (per-channel override > batch override > channel default).
        effective_provider = payload.upload_provider
        if payload.upload_provider_by_channel:
            ch_key = str(ch.id)
            if ch_key in payload.upload_provider_by_channel:
                effective_provider = payload.upload_provider_by_channel[ch_key]
        if effective_provider is None:
            effective_provider = getattr(ch, "upload_provider", None)

        upload_path = _map_legacy_upload_provider(effective_provider)
        postiz_iid = None
        if (effective_provider or "").strip().lower() == "postiz":
            # Route to Postiz: resolve WHICH connected Postiz channel this maps
            # to. Precedence:
            #   1. explicit binding from the request (frontend name-matched the
            #      channel to a LIVE Postiz integration the DB may not know yet),
            #   2. the channel's persisted postiz_integration_id,
            #   3. auto-match by name/handle against team-owned DB integrations.
            # Whatever resolves is persisted onto the channel so future
            # publishes + the production auto-fallback resolve it natively.
            postiz_iid = None
            if payload.postiz_integration_by_channel:
                postiz_iid = (payload.postiz_integration_by_channel.get(str(ch.id)) or "").strip() or None
            if not postiz_iid:
                postiz_iid = (getattr(ch, "postiz_integration_id", "") or "").strip() or None
            if not postiz_iid:
                postiz_iid = _resolve_postiz_integration_for_channel(db, user, ch)
            if postiz_iid and (getattr(ch, "postiz_integration_id", "") or "").strip() != postiz_iid:
                ch.postiz_integration_id = postiz_iid
                db.add(ch)
            if postiz_iid:
                upload_path = "postiz"
            else:
                # FAIL-CLOSED: the user explicitly chose to deliver this channel
                # via Postiz. We must NOT silently fall back to a native YouTube
                # upload (that burns YT quota and posts somewhere the user didn't
                # intend). Error loudly so they re-pick the channel instead.
                _log.warning(
                    "publish: channel=%s (%r) requested Postiz but no matching "
                    "Postiz channel resolved — refusing to fall back to native.",
                    ch.id, getattr(ch, "name", ""),
                )
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Channel '{getattr(ch, 'name', '')}' is set to publish via "
                        f"Postiz, but no matching Postiz channel was found. Re-select "
                        f"it in the Postiz list (matched by name) and try again — "
                        f"nothing was uploaded."
                    ),
                )
        publish_kind = payload.publish_kind or "video"

        # Shorts MUST have thumbnail_source=None; videos require
        # 'pipeline_generated' as the legacy implicit default — unless
        # Quick Publish stored a user/AI thumbnail on the clip
        # (clip.meta.publish_thumbnail_key), in which case the dispatch
        # layer downloads it and runs thumbnails.set after the upload.
        if publish_kind == "short":
            thumb_src = None
            thumb_key = None
        else:
            # Resolve the thumbnail for THIS channel. Precedence:
            #   1. payload.thumbnail_by_channel[ch.id]  (per-channel override)
            #   2. payload.thumbnail_mode (+ custom_thumbnail_r2_key)  (batch)
            #   3. legacy default: rendered, or a clip.meta publish_thumbnail_key
            _spec = None
            if payload.thumbnail_by_channel:
                _spec = payload.thumbnail_by_channel.get(str(ch.id))
            if _spec is None and payload.thumbnail_mode:
                _m = (payload.thumbnail_mode or "").strip().lower()
                if _m == "custom" and (payload.custom_thumbnail_r2_key or "").strip():
                    _spec = "custom:" + payload.custom_thumbnail_r2_key.strip()
                elif _m in ("rendered", "none"):
                    _spec = _m
            if _spec is not None:
                _s = str(_spec).strip()
                if _s == "none":
                    thumb_src, thumb_key = None, None
                elif _s in ("rendered", ""):
                    thumb_src, thumb_key = "pipeline_generated", None
                elif _s.startswith("custom:"):
                    thumb_src, thumb_key = "user_uploaded", _s[len("custom:"):]
                else:
                    thumb_src, thumb_key = "user_uploaded", _s
            else:
                # Legacy default — rendered, unless Quick Publish stored a key.
                thumb_src = "pipeline_generated"
                thumb_key = None
                try:
                    import json as _json
                    _cm = _json.loads(clip.meta) if clip.meta else {}
                    _ptk = (
                        (_cm.get("publish_thumbnail_key") or "").strip()
                        if isinstance(_cm, dict) else ""
                    )
                    if _ptk:
                        thumb_src = "user_uploaded"
                        thumb_key = _ptk
                except Exception:
                    pass

        # Materialise a "rendered" thumbnail into storage so it actually
        # reaches YouTube — dispatch only calls thumbnails.set when an
        # thumbnail_r2_key is present, and the V4 render leaves the rendered
        # thumbnail on local disk only. (Shorts have thumb_src=None, so this
        # is a no-op for them.)
        if thumb_src == "pipeline_generated" and not thumb_key:
            _rk = _ensure_rendered_thumb_key(clip)
            if _rk:
                thumb_key = _rk

        # Per-channel YouTube publish-setting overrides for THIS upload.
        # Any field absent/blank ⇒ no override ⇒ channel yt_* default used.
        _ps = (payload.publish_settings_by_channel or {}).get(str(ch.id)) or {}
        _ov_lang = (_ps.get("language") or _ps.get("default_language") or None)
        _mfk_raw = _ps.get("made_for_kids")

        targets.append(
            _fanout.FanoutTarget(
                channel_id=int(ch.id),
                upload_path=upload_path,
                publish_kind=publish_kind,
                brand_profile_id=None,
                postiz_integration_id=postiz_iid,
                thumbnail_source=thumb_src,
                thumbnail_r2_key=thumb_key,
                scheduled_at=payload.publish_at,
                privacy_status=payload.privacy_status,
                brand_mode=(payload.brand_mode or "per_channel"),
                brand_placement=(payload.brand_placement or "template"),
                brand_profile_version=brand_v,
                seo_version=seo_v,
                metadata_version=metadata_v,
                yt_category_id=(_ps.get("category_id") or None),
                yt_default_language=_ov_lang,
                yt_playlist_id=(_ps.get("playlist_id") or None),
                yt_license=(_ps.get("license") or None),
                yt_made_for_kids=(bool(_mfk_raw) if _mfk_raw is not None else None),
            )
        )

    request = _fanout.PublishTaskRequest(
        master_video_id=int(master.id),
        targets=targets,
        priority="normal",
    )

    # ── 6. Dispatch through the v2 fanout service ────────────────────────
    try:
        result = _fanout.create_publish_task(db, user, request)
        db.commit()
    except _fanout.DuplicatePublishVersionError as exc:
        # Dedupe-by-design: return the same job rows the legacy frontend
        # would have seen if this were a no-op re-publish.
        #
        # ROLLBACK (not commit): the fanout flushed a PublishTask row as
        # 'fanning_out' before hitting the duplicate, with ZERO jobs
        # attached. Committing here persisted that as a PHANTOM publish
        # stuck at "Starting / 0%" forever (it has no jobs for the worker
        # to pick up). Rolling back discards the orphan; the SELECT below
        # re-reads the genuinely-existing task on a clean session.
        db.rollback()
        existing_jobs = (
            db.query(models.UploadJobV2)
            .filter(models.UploadJobV2.publish_task_id == exc.existing_publish_task_id)
            .order_by(models.UploadJobV2.id.asc())
            .all()
        )
        _log.info(
            "legacy_to_v2_redirect: dedupe-by-design — returning existing "
            "publish_task_id=%s (%d jobs)",
            exc.existing_publish_task_id, len(existing_jobs),
        )
        return _v2_jobs_to_legacy_shape(
            db, existing_jobs, clip, by_id, already_published=True,
        )
    except _credits.InsufficientCreditsError as exc:
        db.rollback()
        raise HTTPException(
            status_code=402,
            detail={
                "code": "insufficient_credits",
                "message": str(exc),
                "balance": int(getattr(exc, "balance", 0)),
                "needed": int(getattr(exc, "needed", 0)),
            },
        )
    except _fanout.PlanTierViolationError as exc:
        db.rollback()
        raise HTTPException(
            status_code=403,
            detail={"code": exc.code, "message": str(exc)},
        )
    except _fanout.MasterVideoNotReadyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
        )
    except _fanout.FanoutError as exc:
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail={"code": exc.code, "message": str(exc)},
        )
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        _log.exception(
            "legacy_to_v2_redirect: fanout failed for clip_id=%s: %s",
            clip.id, exc,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Publish failed: {exc}",
        )

    # ── 7. Serialise the v2 result into legacy response shape ────────────
    created_jobs = (
        db.query(models.UploadJobV2)
        .filter(models.UploadJobV2.id.in_(result.upload_job_ids))
        .order_by(models.UploadJobV2.id.asc())
        .all()
    )
    _log.info(
        "legacy_to_v2_redirect: created publish_task_id=%s upload_job_ids=%s "
        "(skipped %s already-published channel(s))",
        result.publish_task_id, result.upload_job_ids,
        getattr(result, "skipped_count", 0),
    )
    out = _v2_jobs_to_legacy_shape(db, created_jobs, clip, by_id)
    # Partial publish: some channels were brand-new (uploaded), others were
    # already published with this version (skipped). Surface the skip count
    # so the UI can say "published to the new channels; N already had it"
    # instead of looking like it ignored the rest.
    skipped = int(getattr(result, "skipped_count", 0) or 0)
    if skipped and isinstance(out, dict):
        out["skipped_already_published"] = skipped
    return out


def _v2_jobs_to_legacy_shape(
    db: Session,
    v2_jobs: list,
    clip: "models.Clip",
    by_id: dict,
    *,
    already_published: bool = False,
):
    """Render a list of ``UploadJobV2`` rows into the same response
    shape ``_to_dict`` produces for the legacy ``UploadJob``.

    Single target → flat dict; multiple → ``{jobs: [...], count: n}``.

    ``already_published=True`` marks a dedupe-by-design no-op (the clip
    was already published to the selected channel(s) with the same
    video / SEO / privacy). We attach ``already_published`` + a
    human-readable ``message`` so the UI can show a clear notice instead
    of silently redirecting — otherwise the user thinks publish broke.
    """
    def _to_legacy_shape(j) -> dict:
        ch = by_id.get(j.channel_id)
        return {
            "id": int(j.id),
            "clip_id": int(clip.id),
            "channel_id": int(j.channel_id) if j.channel_id is not None else None,
            "channel_name": ch.name if ch else None,
            "clip_filename": getattr(clip, "filename", None),
            "clip_thumb_url": (
                (getattr(clip, "thumb_storage_url", "") or "")
                or (f"/api/file/?path={clip.thumb_path}" if getattr(clip, "thumb_path", None) else "")
            ),
            "status": j.status,
            "privacy_status": getattr(j, "privacy_status", "private"),
            "publish_kind": j.publish_kind or "video",
            "publish_at": j.publish_at.isoformat() if getattr(j, "publish_at", None) else None,
            "title": getattr(j, "title", None),
            "description": getattr(j, "description", None),
            "tags": list(getattr(j, "tags", None) or []),
            "category_id": getattr(j, "category_id", None) or "25",
            "made_for_kids": bool(getattr(j, "made_for_kids", False)),
            "video_id": j.youtube_video_id or None,
            "video_url": (
                f"https://youtu.be/{j.youtube_video_id}" if j.youtube_video_id else None
            ),
            # Mirror legacy: v2's upload_path maps back to a legacy-ish hint.
            "upload_provider": (
                "native_rtmp" if (j.upload_path or "") == "rtmp" else "kaizer"
            ),
            "bytes_uploaded": int(getattr(j, "bytes_uploaded", 0) or 0),
            "bytes_total": 0,
            "progress_pct": 0.0,
            "attempts": int(getattr(j, "attempts", 0) or 0),
            "last_error": getattr(j, "last_error", "") or "",
            "log": "",
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "updated_at": j.updated_at.isoformat() if j.updated_at else None,
            # v2-only fields surfaced for diagnostics — harmless to legacy callers.
            "v2_publish_task_id": int(j.publish_task_id),
            "v2_idempotency_key": j.idempotency_key,
            # Job-wise Publishes UI deep-link (additive): the modal
            # navigates to /uploads/{publish_task_id} after publishing.
            "publish_task_id": int(j.publish_task_id),
        }

    _n_ch = len({j.channel_id for j in v2_jobs}) if v2_jobs else 0
    _dupe_msg = (
        f"This clip is already published to "
        f"{_n_ch} channel{'s' if _n_ch != 1 else ''} with the same video, "
        f"SEO and privacy — so no new upload was created. Change the SEO, "
        f"privacy, or thumbnail to publish a new version, or open the "
        f"existing publish below."
        if _n_ch else
        "This clip was already published with these exact settings — no "
        "new upload was created."
    )
    if len(v2_jobs) == 1:
        out = _to_legacy_shape(v2_jobs[0])
        if already_published:
            out["already_published"] = True
            out["message"] = _dupe_msg
        return out
    out = {
        "jobs": [_to_legacy_shape(j) for j in v2_jobs],
        "count": len(v2_jobs),
        "publish_task_id": (
            int(v2_jobs[0].publish_task_id) if v2_jobs else None
        ),
    }
    if already_published:
        out["already_published"] = True
        out["message"] = _dupe_msg
    return out


class _PublishedStatusRequest(BaseModel):
    clip_ids: list[int] = []


@router.post("/clips/published-status")
def clips_published_status(
    payload: _PublishedStatusRequest,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Which of the given clips are ALREADY published, broken down per
    channel. The bulk-publish modal calls this to badge + auto-deselect
    channels that already have the clip(s), so the user publishes only to
    the NEW channels instead of the whole batch quietly de-duping.

    Tenant-scoped to the caller's own clips/jobs. Returns:
        {"clip_count": N,
         "by_channel": {"<channel_id>": {"channel_id", "count",
                                         "clip_ids", "video_ids"}}}
    where ``count`` is how many of the requested clips already have a
    COMPLETED upload on that channel.
    """
    clip_ids = sorted({int(c) for c in (payload.clip_ids or []) if c is not None})
    if not clip_ids:
        return {"clip_count": 0, "by_channel": {}}
    # Tenant isolation: only the caller's own clips (clip → job → user).
    owned_ids = [
        r[0] for r in (
            db.query(models.Clip.id)
            .join(models.Job, models.Job.id == models.Clip.job_id)
            .filter(models.Clip.id.in_(clip_ids), models.Job.user_id == user.id)
            .all()
        )
    ]
    if not owned_ids:
        return {"clip_count": 0, "by_channel": {}}
    rows = (
        db.query(
            models.UploadJobV2.channel_id,
            models.MasterVideo.clip_id,
            models.UploadJobV2.youtube_video_id,
        )
        .join(models.PublishTask, models.PublishTask.id == models.UploadJobV2.publish_task_id)
        .join(models.MasterVideo, models.MasterVideo.id == models.PublishTask.master_video_id)
        .filter(
            models.MasterVideo.clip_id.in_(owned_ids),
            models.UploadJobV2.status == "completed",
            models.UploadJobV2.user_id == int(user.id),
        )
        .all()
    )
    by_channel: dict = {}
    for ch_id, clip_id, vid in rows:
        if ch_id is None:
            continue
        key = str(int(ch_id))
        e = by_channel.setdefault(
            key, {"channel_id": int(ch_id), "clip_ids": [], "video_ids": []},
        )
        if int(clip_id) not in e["clip_ids"]:
            e["clip_ids"].append(int(clip_id))
        if vid and vid not in e["video_ids"]:
            e["video_ids"].append(vid)
    for e in by_channel.values():
        e["count"] = len(e["clip_ids"])
    return {"clip_count": len(owned_ids), "by_channel": by_channel}


@router.post("/clips/{clip_id}/thumbnail/stage")
def stage_thumbnail(
    clip_id: int,
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Store a custom thumbnail for a clip and return its storage key WITHOUT
    mutating the clip. Lets the publish modals assign different thumbnails per
    channel: the returned ``key`` goes into PublishRequest.thumbnail_by_channel
    (as ``"custom:<key>"``) or ``custom_thumbnail_r2_key`` (shared across all).
    """
    import os as _os
    import shutil as _shutil
    import tempfile as _tempfile
    # Ownership check (mirrors quick_publish._owned_clip).
    clip = db.query(models.Clip).filter(models.Clip.id == int(clip_id)).first()
    if clip is None:
        raise HTTPException(status_code=404, detail="Clip not found")
    job = db.query(models.Job).filter(models.Job.id == clip.job_id).first()
    if job is not None and job.user_id is not None:
        if int(job.user_id) != int(user.id) and not bool(user.is_admin):
            raise HTTPException(status_code=403, detail="Not your clip")
    ct = (image.content_type or "").lower()
    if ct not in ("image/jpeg", "image/jpg", "image/png"):
        raise HTTPException(status_code=400, detail="Thumbnail must be JPG or PNG")
    _MAX = 4 * 1024 * 1024
    data = image.file.read(_MAX + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload")
    if len(data) > _MAX:
        raise HTTPException(status_code=400, detail="Thumbnail must be ≤ 4 MB")
    ext = "png" if "png" in ct else "jpg"
    tmp = _tempfile.mkdtemp(prefix="kaizer_stagethumb_")
    try:
        local = _os.path.join(tmp, f"thumb.{ext}")
        with open(local, "wb") as f:
            f.write(data)
        from pipeline_core.storage import get_storage_provider
        key = f"publish_thumbs/{int(user.id)}/clip_{int(clip.id)}/thumb_{int(time.time())}.{ext}"
        stored = get_storage_provider().upload(local, key, content_type=ct)
        return {"key": stored.key, "url": stored.url}
    finally:
        _shutil.rmtree(tmp, ignore_errors=True)


@router.post("/clips/{clip_id}/publish")
def publish_clip(
    clip_id: int,
    payload: PublishRequest,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
    _rl=Depends(_rate_limited("create")),
):
    """Publish a clip to one or more YouTube destinations in one call.

    Accepts either the legacy `channel_id` (single) or the newer `channel_ids`
    list (fan-out → one UploadJob per entry).  Returns a single dict when only
    one target was requested, or `{jobs: [...]}` for a fan-out.

    Phase 3 cutover: when ``KAIZER_NEW_PUBLISH_PATH=1``, this handler
    delegates to ``_legacy_to_v2_redirect`` which routes through
    ``services.fanout.create_publish_task``. The response shape stays
    identical so the existing frontend works unchanged. Default is
    unchanged — flip the flag to opt in to the new path.
    """
    if _legacy_publish_to_v2_enabled():
        return _legacy_to_v2_redirect(db, user, clip_id, payload)

    clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    if not clip.file_path:
        raise HTTPException(status_code=422, detail="Clip has no rendered file — run the pipeline first")

    # Resolve target list — dedupe but preserve order
    raw_ids: list[int] = []
    if payload.channel_ids:
        raw_ids.extend(payload.channel_ids)
    if payload.channel_id is not None:
        raw_ids.append(payload.channel_id)
    seen: set[int] = set()
    target_ids = [i for i in raw_ids if not (i in seen or seen.add(i))]
    if not target_ids:
        raise HTTPException(status_code=422, detail="At least one destination must be selected.")

    # Look up all targets up-front and validate them before creating any jobs.
    channels = db.query(models.Channel).filter(models.Channel.id.in_(target_ids)).all()
    by_id = {c.id: c for c in channels}
    missing = [i for i in target_ids if i not in by_id]
    if missing:
        raise HTTPException(status_code=404, detail=f"Profile(s) not found: {missing}")

    for cid in target_ids:
        ch = by_id[cid]
        if not ch.oauth_token or not ch.oauth_token.refresh_token_enc:
            raise HTTPException(
                status_code=409,
                detail=f"Style profile '{ch.name}' is not linked to YouTube. Open Style Profiles → Link my YT.",
            )

    # Resolve which clip's SEO to use:
    #   - default: this clip's own SEO
    #   - if `seo_source_clip_id` is set AND this clip lacks its own SEO:
    #     fall back to the source clip (must be in the same job, owned by
    #     the same user). Lets a video's 5 clips share one SEO without
    #     forcing the user to regenerate per clip.
    def _has_real_seo(c) -> bool:
        """True iff *c* has either a populated `seo` JSON string OR a
        non-empty `seo_variants` dict. The bare-truthiness check on
        `seo_variants` is wrong because the column DEFAULTS to the
        string '{}' (which is truthy) — every clip without variants
        looked like it 'had SEO' under that test, so the donor lookup
        below never fired and inheriting clips were rejected with
        'no SEO metadata'."""
        if c.seo:
            return True
        try:
            sv = json.loads(c.seo_variants or "{}")
            return bool(isinstance(sv, dict) and sv)
        except (ValueError, TypeError):
            return False

    seo_clip = clip
    if payload.use_seo and not _has_real_seo(clip) and payload.seo_source_clip_id:
        donor = db.query(models.Clip).filter(
            models.Clip.id == payload.seo_source_clip_id,
        ).first()
        if donor is None:
            raise HTTPException(
                status_code=404,
                detail=f"seo_source_clip_id={payload.seo_source_clip_id} not found.",
            )
        if donor.job_id != clip.job_id:
            raise HTTPException(
                status_code=400,
                detail="seo_source_clip_id must belong to the same job as the publishing clip.",
            )
        if _has_real_seo(donor):
            seo_clip = donor

    # Accept either a legacy clip.seo OR a per-channel variant on the
    # resolved seo_clip (which may be the donor).
    if payload.use_seo:
        try:
            _variants_check = json.loads(seo_clip.seo_variants or "{}")
        except (ValueError, TypeError):
            _variants_check = {}
        if not seo_clip.seo and not (isinstance(_variants_check, dict) and _variants_check):
            raise HTTPException(
                status_code=409,
                detail="Clip has no SEO metadata. Generate SEO first, or pass use_seo=false with manual title/description.",
            )

    # Scheduled uploads must be private-until-publish
    if payload.publish_at and payload.privacy_status != "private":
        raise HTTPException(
            status_code=422,
            detail="Scheduled uploads require privacy_status='private' (YouTube flips it to public at publish_at).",
        )

    # Fan-out: one UploadJob per target, each with independently composed metadata
    created: list[models.UploadJob] = []
    for cid in target_ids:
        channel = by_id[cid]
        # Use seo_clip (may be the donor sibling) for SEO content;
        # everything else still uses the actual clip being uploaded.
        title, description, tags = _compose_metadata(seo_clip, channel, payload)
        # Resolve the per-(clip × channel) upload route.
        # Precedence (most specific → least specific):
        #   1. payload.upload_provider_by_channel[str(channel.id)]
        #      — the per-channel override from the publish modal
        #   2. payload.upload_provider — the batch-wide override
        # Whatever lands in `effective_provider` is stamped on the
        # UploadJob row and wins over Channel.upload_provider and
        # system default in the worker's routing chain.
        # ``None`` here means "let the worker resolve" (no override).
        effective_provider = payload.upload_provider
        if payload.upload_provider_by_channel:
            ch_key = str(channel.id)
            if ch_key in payload.upload_provider_by_channel:
                # Explicit None / "" in the map = "remove override for
                # this channel only, even if a batch override is set".
                effective_provider = payload.upload_provider_by_channel[ch_key]
        job = models.UploadJob(
            user_id=user.id,
            clip_id=clip.id,
            channel_id=channel.id,
            status="queued",
            privacy_status=payload.privacy_status,
            publish_kind=payload.publish_kind,
            publish_at=payload.publish_at,
            title=title,
            description=description,
            tags=tags,
            category_id=payload.category_id or "25",
            made_for_kids=payload.made_for_kids,
            # null at this layer means "let the worker resolve via
            # Channel.upload_provider → system default".  Set on the
            # row so admin can audit what each comparison run used.
            upload_provider=effective_provider,
        )
        db.add(job)
        created.append(job)
    db.commit()
    for j in created:
        db.refresh(j)

    # Signal the upload worker via Redis Streams. Done AFTER commit so
    # the message can never reference a row that wasn't persisted.
    # Failure is non-fatal: if Redis is down, the legacy DB-poll path
    # in `youtube/worker.py` will still pick the row up (degraded
    # mode, single-host only).
    #
    # Priority lane is derived from the user's plan — pro/agency tenants
    # land on the ``hi`` stream so they jump ahead of free-tier batch
    # work. ``priority_for_user`` is the single source of truth for
    # this mapping (see redis_queue._PRIO_BY_PLAN).
    try:
        from redis_queue import enqueue_upload_job, is_enabled as _redis_on, priority_for_user
        if _redis_on():
            prio = priority_for_user(user)
            for j in created:
                try:
                    enqueue_upload_job(j.id, priority=prio)
                except Exception as exc:
                    # Log but don't fail the API call — the row exists
                    # and the DB-poll fallback will catch it.
                    print(f"[uploads] redis enqueue failed for job {j.id}: {exc}")
    except Exception as exc:
        print(f"[uploads] redis_queue import failed (skipping enqueue): {exc}")

    # Single-target → legacy shape; fan-out → list under `jobs`
    if len(created) == 1:
        return _to_dict(created[0])
    return {"jobs": [_to_dict(j) for j in created], "count": len(created)}


@router.get("/uploads")
def list_uploads(
    status_filter: Optional[str] = Query(None, alias="status"),
    channel_id:    Optional[int] = None,
    clip_id:       Optional[int] = None,
    limit:         int = 100,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    q = db.query(models.UploadJob).filter(models.UploadJob.user_id == user.id)
    if status_filter:
        q = q.filter(models.UploadJob.status == status_filter)
    if channel_id:
        q = q.filter(models.UploadJob.channel_id == channel_id)
    if clip_id:
        q = q.filter(models.UploadJob.clip_id == clip_id)
    rows = q.order_by(models.UploadJob.created_at.desc()).limit(limit).all()
    return [_to_dict(r) for r in rows]


@router.get("/uploads/{upload_id}")
def get_upload(upload_id: int, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    row = db.query(models.UploadJob).filter(
        models.UploadJob.id == upload_id, models.UploadJob.user_id == user.id,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Upload not found")
    return _to_dict(row)


@router.delete("/uploads/{upload_id}")
def cancel_upload(upload_id: int, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    row = db.query(models.UploadJob).filter(
        models.UploadJob.id == upload_id, models.UploadJob.user_id == user.id,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Upload not found")
    if row.status in ("done", "failed", "cancelled"):
        return {"upload_id": upload_id, "status": row.status, "cancelled": False,
                "note": f"already terminal ({row.status})"}
    if row.status == "uploading":
        # We can only mark the row; in-flight chunks will notice on next checkpoint
        row.status = "cancelled"
        row.last_error = (row.last_error or "") + "\n[user] cancelled"
        db.commit()
        return {"upload_id": upload_id, "status": "cancelled", "cancelled": True,
                "note": "in-flight chunks may continue briefly"}
    row.status = "cancelled"
    db.commit()
    return {"upload_id": upload_id, "status": "cancelled", "cancelled": True}


@router.post("/uploads/{upload_id}/retry")
def retry_upload(upload_id: int, db: Session = Depends(get_db), user: models.User = Depends(auth.current_user)):
    row = db.query(models.UploadJob).filter(
        models.UploadJob.id == upload_id, models.UploadJob.user_id == user.id,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Upload not found")
    if row.status not in ("failed", "cancelled"):
        raise HTTPException(status_code=409, detail=f"Cannot retry while status={row.status}")
    row.status = "queued"
    row.attempts = 0
    row.last_error = ""
    # Keep upload_uri + bytes_uploaded so we resume where we left off
    db.commit()
    # Re-signal the worker via Redis. If Redis is unavailable, the
    # DB-poll fallback will still see status='queued' on its next
    # cycle (slower, but works). Retry returns the job to the lane
    # matching the user's CURRENT plan — if they upgraded since the
    # original failure, they get the upgrade on the retry.
    try:
        from redis_queue import enqueue_upload_job, is_enabled as _redis_on, priority_for_user
        if _redis_on():
            enqueue_upload_job(row.id, priority=priority_for_user(user))
    except Exception as exc:
        print(f"[uploads/retry] redis enqueue failed for job {row.id}: {exc}")
    return _to_dict(row)


@router.get("/uploads/{upload_id}/log")
def stream_log(upload_id: int):
    """SSE stream — emits job dict every POLL_MS until status is terminal."""
    def event_stream():
        last_payload: Optional[str] = None
        # 15 minutes of polling max; UI should give up or reconnect
        for _ in range(15 * 60 * 2):
            db = SessionLocal()
            try:
                row = db.query(models.UploadJob).filter(models.UploadJob.id == upload_id).first()
                if not row:
                    yield f"data: {json.dumps({'error': 'not found'})}\n\n"
                    return
                payload = json.dumps(_to_dict(row), ensure_ascii=False, default=str)
                if payload != last_payload:
                    yield f"data: {payload}\n\n"
                    last_payload = payload
                if row.status in ("done", "failed", "cancelled"):
                    return
            finally:
                db.close()
            time.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/quota")
def get_quota(db: Session = Depends(get_db)):
    """Lightweight quota snapshot for the UI.

    Quota-truth fix: serve the SAME snapshot the v2 gate enforces —
    cap resolved via services/quota_sync (Google's real assigned limit,
    hourly-synced), usage from the production 'oauth' bucket.

    Returns BOTH buckets so the UI can show the right number:
      - flat ``{date, used, limit, remaining}`` = the 10,000 "Queries" pool
        (kept for backward-compat with existing consumers),
      - ``queries`` = same Queries-pool snapshot,
      - ``uploads`` = the SEPARATE videos.insert 100/day bucket
        ``{date, used, cap:100, remaining}`` — the real "X / 100 uploads
        today" the Uploads page should display (uploads do NOT consume the
        10,000 Queries pool).
    """
    from youtube import quota_v2 as _quota_v2
    _queries = _quota_v2.snapshot(db)
    return {
        **_queries,
        "queries": _queries,
        "uploads": _quota_v2.uploads_snapshot(db),
    }
