"""Auto per-channel SEO — the YouTube-duplication bypass, applied UP FRONT.

When a render job finishes, every clip should publish to each of its chosen
channels with a DISTINCT title/description/tags so YouTube doesn't flag the
same video across the operator's channels as duplicate. That used to require a
manual "Apply per-channel" click in the editor; this module does it
automatically right after the pipeline materialises the clips (operator
decision 2026-06-18: "auto at job confirm").

Design guarantees:
  * IDEMPOTENT — only fills channels that don't already have a marked
    ``_per_channel`` variant, so a manual edit (or a re-run) is never clobbered.
  * FAIL-SOFT — any error is swallowed; the job still completes. A channel that
    fails generation simply falls back to the clip's shared SEO at publish
    (current behaviour) — never worse than before.
  * SCOPED — only runs for jobs targeting 2+ connected channels. A single
    channel needs no dedup, and shared-mode/free single-channel jobs stay
    byte-for-byte unchanged (no AI spend, no output change).
  * REUSES the exact same generation + variant shape as the editor's
    apply-per-channel path (``build_channel_previews(generate=True)`` →
    ``_per_channel``-marked entry), so publish (`upload_dispatch._compose_metadata`)
    consumes it identically.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional


def _enabled() -> bool:
    """OFF by default (operator decision 2026-06-18): distinct per-channel SEO is
    MANUAL — it generates ONLY when the operator clicks "Write distinct SEO" in
    the editor, never automatically during render (which re-ran on every
    re-render and burned AI calls). Set KAIZER_V4_AUTO_CHANNEL_SEO=1 to opt back
    into auto-at-render generation. The editor button is unaffected by this flag."""
    return os.environ.get("KAIZER_V4_AUTO_CHANNEL_SEO", "0").strip() == "1"


def auto_generate_channel_seo_for_job(
    job_id: int,
    *,
    language: str = "te",
    log: Optional[Callable[[str], None]] = None,
    max_channels: int = 50,
) -> dict:
    """Ensure each clip of ``job_id`` has distinct per-channel SEO for the job's
    chosen channels. Returns a summary dict. Never raises."""
    def _log(msg: str) -> None:
        if log:
            try:
                log(msg)
            except Exception:
                pass

    if not _enabled():
        return {"skipped": "disabled"}

    # Lazy imports so importing this module never drags SEO/DB deps into a
    # light context, and the orchestrator import stays cheap.
    try:
        from database import SessionLocal
        import models
        from seo.per_channel import build_channel_previews
    except Exception as exc:
        _log(f"[v4] auto-channel-seo: deps unavailable, skipping ({exc})")
        return {"skipped": "deps", "error": str(exc)}

    db = SessionLocal()
    try:
        job = db.query(models.Job).filter(models.Job.id == job_id).first()
        if job is None:
            return {"skipped": "no_job"}

        # ── Resolve the job's chosen channels (target_channel_ids) ──────────
        try:
            tids = json.loads(job.target_channel_ids) if job.target_channel_ids else []
        except Exception:
            tids = []
        target_ids = {int(t) for t in tids} if isinstance(tids, list) else set()
        if not target_ids:
            return {"skipped": "no_target_channels"}

        # Only CONNECTED channels of this user that are in the target set can
        # actually publish — those are the ones that need distinct SEO.
        channels = [
            c for c in db.query(models.Channel)
                         .filter(models.Channel.user_id == job.user_id).all()
            if c.id in target_ids
            and c.oauth_token is not None and bool(c.oauth_token.refresh_token_enc)
        ]
        if len(channels) < 2:
            # 0/1 publishable channel → no duplication to avoid.
            return {"skipped": "lt2_channels", "channels": len(channels)}
        if len(channels) > max_channels:
            _log(f"[v4] auto-channel-seo: capping {len(channels)} channels to "
                 f"{max_channels} (rest fall back to shared SEO)")
            channels = channels[:max_channels]

        # ── Winning-keyword profile per channel (pure DB, shared across clips) ─
        fb: dict = {}
        try:
            from seo.performance_profile import build_channel_profile
        except Exception:
            build_channel_profile = None
        try:
            from analytics.channel_catalog import ensure_synced as _ensure_synced
        except Exception:
            _ensure_synced = None
        for ch in channels:
            try:
                gcid = (ch.oauth_token.google_channel_id or "") if ch.oauth_token else ""
                if _ensure_synced is not None and gcid:
                    _ensure_synced(db, job.user_id, gcid)
            except Exception:
                pass
            if build_channel_profile is not None:
                try:
                    p = build_channel_profile(db, ch.id)
                    if p.get("ready"):
                        fb[ch.id] = {"winning_keywords": p.get("winning_keywords", [])}
                except Exception:
                    pass

        try:
            from seo.score_checker import _content_keywords
        except Exception:
            _content_keywords = None

        clips = db.query(models.Clip).filter(models.Clip.job_id == job.id).all()
        total_generated = 0
        clips_touched = 0
        for clip in clips:
            # base SEO comes from the clip's own (shared) SEO produced by the
            # pipeline. No base title → nothing to differentiate from.
            try:
                base_seo = json.loads(clip.seo) if clip.seo else {}
            except Exception:
                base_seo = {}
            if not isinstance(base_seo, dict) or not (base_seo.get("title") or "").strip():
                continue

            try:
                variants = json.loads(clip.seo_variants or "{}") if clip.seo_variants else {}
                if not isinstance(variants, dict):
                    variants = {}
            except Exception:
                variants = {}

            # IDEMPOTENT: only channels missing a marked variant.
            missing = [
                ch for ch in channels
                if not (isinstance(variants.get(str(ch.id)), dict)
                        and variants[str(ch.id)].get("_per_channel"))
            ]
            if not missing:
                continue

            content_text = (
                (base_seo.get("title") or "") + "\n" + (base_seo.get("description") or "")
            ).strip()
            ckw = []
            if _content_keywords is not None:
                try:
                    ckw = _content_keywords(content_text, top=10)
                except Exception:
                    ckw = []

            try:
                previews = build_channel_previews(
                    base_seo=base_seo, channels=missing,
                    feedback_by_channel=fb, content_keywords=ckw, mode="per_channel",
                    generate=True, content_text=content_text, language=language or "te",
                )
            except Exception as exc:
                _log(f"[v4] auto-channel-seo: generation failed for clip "
                     f"{clip.id} (soft-skip): {exc}")
                continue

            applied_here = 0
            for p in previews:
                cid = p.get("channel_id")
                if cid is None:
                    continue
                title = (p.get("title") or "").strip()
                if not title:
                    continue   # generation fell back empty → leave channel on shared SEO
                _v = {
                    "title": title,
                    "description": p.get("description", ""),
                    "tags": p.get("tags", []),
                    "_per_channel": True,
                }
                if p.get("hashtags"):
                    _v["hashtags"] = p.get("hashtags")
                if p.get("seo_score") is not None:
                    _v["seo_score"] = p.get("seo_score")
                variants[str(cid)] = _v
                applied_here += 1

            if applied_here:
                clip.seo_variants = json.dumps(variants)
                db.add(clip)
                db.commit()
                total_generated += applied_here
                clips_touched += 1

        _log(f"[v4] auto-channel-seo: wrote {total_generated} per-channel "
             f"SEO variant(s) across {clips_touched} clip(s) for {len(channels)} channel(s)")
        return {
            "ok": True,
            "channels": len(channels),
            "clips_touched": clips_touched,
            "variants_written": total_generated,
        }
    except Exception as exc:
        _log(f"[v4] auto-channel-seo: unexpected error (soft-skip): {exc}")
        return {"skipped": "error", "error": str(exc)}
    finally:
        db.close()
