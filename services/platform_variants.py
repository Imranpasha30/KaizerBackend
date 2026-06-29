"""Per-platform SEO variant cache + publish-platform resolution.

Bridges the per-platform generator (``seo.platform_seo``) and the publish
path. Two jobs:

  * ``resolve_publish_platform(db, job)`` — figure out which social platform an
    UploadJobV2 actually targets: direct/rtmp = YouTube; postiz = the bound
    Postiz integration's provider (instagram / facebook / …), read from the
    cached ``PostizIntegration`` row (NO extra Postiz API call).

  * ``ensure_platform_variant(db, clip, platform)`` — lazily generate (one
    Gemini call) the platform's caption + hashtags and CACHE it inside the
    clip's SEO JSON at ``seo['platform_variants'][platform]``. Returns the
    cached variant on subsequent calls, so we spend at most one call per
    (clip, platform) actually used. Fails closed (returns {} / the base) so a
    publish is never blocked.

The YouTube path never touches this module — it keeps its proven
title+description+tags flow.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import models

log = logging.getLogger("kaizer.platform_variants")

# Platforms we generate native captions for. Anything else (youtube, x,
# linkedin, tiktok, unknown) falls back to the generic/YouTube shaping.
SOCIAL_PLATFORMS = ("instagram", "facebook")


def resolve_publish_platform(db, job: "models.UploadJobV2") -> str:
    """Return the lowercase platform an upload job targets.

    direct / rtmp -> "youtube". postiz -> the integration's provider
    (instagram / facebook / youtube / x / …). Safe default "youtube" when
    unresolved so the proven YouTube shaping is used.
    """
    up = (getattr(job, "upload_path", None) or "direct").strip().lower()
    if up in ("direct", "rtmp"):
        return "youtube"
    if up == "postiz":
        iid = (getattr(job, "postiz_integration_id", "") or "").strip()
        if iid:
            try:
                pi = (
                    db.query(models.PostizIntegration)
                    .filter(models.PostizIntegration.integration_id == iid)
                    .first()
                )
                if pi and (pi.provider or "").strip():
                    return pi.provider.strip().lower()
            except Exception as exc:  # noqa: BLE001 — never block publish
                log.warning("resolve_publish_platform: lookup failed (%s)", exc)
        return "youtube"
    return "youtube"


def _parse_seo(clip) -> Dict[str, Any]:
    if not clip or not getattr(clip, "seo", None):
        return {}
    try:
        parsed = json.loads(clip.seo)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def get_cached_variant(clip, platform: str) -> Dict[str, Any]:
    """Return the cached platform variant from clip.seo, or {} if none."""
    platform = (platform or "").strip().lower()
    seo = _parse_seo(clip)
    pv = seo.get("platform_variants")
    if isinstance(pv, dict):
        v = pv.get(platform)
        if isinstance(v, dict):
            return v
    return {}


def ensure_platform_variant(
    db,
    clip,
    platform: str,
    *,
    force: bool = False,
    style_source=None,
    persist: bool = True,
) -> Dict[str, Any]:
    """Return the platform's caption+hashtags variant, generating + caching
    it on first use.

    - Reads/writes ``seo['platform_variants'][platform]`` inside clip.seo.
    - ``force=True`` regenerates even if cached.
    - ``persist=False`` skips the DB write (used by read-only previews that
      still want a fresh generation without committing).
    Returns {} for unsupported platforms (youtube/x/…) — the caller then uses
    the generic shaping.
    """
    platform = (platform or "").strip().lower()
    if platform not in SOCIAL_PLATFORMS or clip is None:
        return {}

    seo = _parse_seo(clip)
    pv = seo.get("platform_variants")
    if not isinstance(pv, dict):
        pv = {}

    existing = pv.get(platform)
    if (not force) and isinstance(existing, dict) and existing.get("caption"):
        # Respect operator edits + cached generations.
        return existing

    try:
        from seo.platform_seo import generate_platform_seo
    except Exception as exc:  # noqa: BLE001
        log.warning("ensure_platform_variant: import failed (%s)", exc)
        return existing if isinstance(existing, dict) else {}

    variant = generate_platform_seo(
        platform=platform,
        language=(seo.get("language") or "te"),
        title_native=((getattr(clip, "text", "") or "") or seo.get("title") or ""),
        title_english="",
        summary="",
        base_seo=seo,
        style_source=style_source,
    )

    if not variant.get("caption"):
        # Generation failed — don't cache an empty shell; let the composer
        # fall back to deterministic shaping.
        return existing if isinstance(existing, dict) else variant

    pv[platform] = variant
    seo["platform_variants"] = pv
    if persist:
        # Persist the cache via a SEPARATE short-lived session so we never
        # commit/expire the caller's transaction. The publish dispatch shares
        # one session across stages and commits its own state explicitly;
        # committing it here (and SQLAlchemy's expire-on-commit) would yank its
        # in-flight job/channel objects out from under it. We re-read the clip
        # fresh and merge, so concurrent per-platform writes don't clobber.
        try:
            from database import SessionLocal
            s2 = SessionLocal()
            try:
                c2 = s2.query(models.Clip).filter(models.Clip.id == clip.id).first()
                if c2 is not None:
                    try:
                        cur = json.loads(c2.seo) if c2.seo else {}
                    except Exception:
                        cur = {}
                    if not isinstance(cur, dict):
                        cur = {}
                    cpv = cur.get("platform_variants")
                    if not isinstance(cpv, dict):
                        cpv = {}
                    cpv[platform] = variant
                    cur["platform_variants"] = cpv
                    c2.seo = json.dumps(cur, ensure_ascii=False)
                    s2.add(c2)
                    s2.commit()
            finally:
                s2.close()
        except Exception as exc:  # noqa: BLE001
            log.warning("ensure_platform_variant: persist failed (%s)", exc)
    return variant
