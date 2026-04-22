"""
kaizer.pipeline.phase4.training_flywheel
=========================================
Private training flywheel — the unmatchable moat.

Every clip creators publish (with its live analytics trajectory)
becomes labeled data. After 6 months of 1000+ users this dataset is
structurally unreachable by any competitor starting today.

Data pipeline (v2)
---------------------------------------------------------
  1. collect_training_record(upload_job_id, db)
     Pulls the Clip's narrative meta (turning-point label, hook_score,
     completion_score) + ClipPerformance snapshots (views, retention at
     T+48h and T+7d) + the uploaded MP4 hash. Emits one TrainingRecord
     row per clip.

  2. extract_features(record)
     Multimodal features: CLIP frame embeddings (keyframes + seam),
     BEATs audio, multilingual E5 text (title + transcript), plus
     hand-crafted: hook-window audio RMS delta, first-face-onscreen
     latency, caption-area score.

  3. retrain_narrative_scorer(min_records=500, niche=None)
     Fine-tunes a small (~50M param) bi-encoder from the narrative
     engine's current heuristic scores toward observed 48h retention
     + 7d shares-per-reach. Per-niche model when a niche accumulates
     enough data; otherwise a global fallback.

  4. deploy_model(model_path)
     Atomic swap via the narrative.py compose-time registry. Rollback
     via the prior snapshot on any QA regression.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

logger = logging.getLogger("kaizer.pipeline.phase4.training_flywheel")

# Language code → niche string used to bucket training data per dialect.
_LANG_NICHE: dict[str, str] = {
    "te": "news_te",
    "hi": "news_hi",
    "ta": "news_ta",
    "kn": "news_kn",
    "ml": "news_ml",
    "bn": "news_bn",
    "mr": "news_mr",
    "gu": "news_gu",
    "en": "news_en",
}


@dataclass
class TrainingRecord:
    upload_job_id: int
    clip_id: int
    niche: Optional[str] = None
    narrative_role: Optional[str] = None
    hook_score: float = 0.0
    completion_score: float = 0.0
    composite_score: float = 0.0
    views_48h: int = 0
    retention_curve: list = field(default_factory=list)
    shares_per_reach: float = 0.0
    video_hash: str = ""
    collected_at: Optional[str] = None  # ISO datetime when the row was written


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _compute_video_hash(file_path: str) -> str:
    """Return a pHash hex string for the first frame of *file_path*.

    Degrades silently to "" when ffmpeg / cv2 / imagehash are absent or the
    file is missing/unreadable.
    """
    if not file_path:
        return ""
    import os
    if not os.path.exists(file_path):
        return ""
    try:
        from pipeline_core.loop_score import _phash_64  # type: ignore
        import subprocess, tempfile, os as _os
        # Extract a single keyframe via ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            from pipeline_core.pipeline import FFMPEG_BIN  # type: ignore
        except Exception:
            FFMPEG_BIN = "ffmpeg"
        result = subprocess.run(
            [FFMPEG_BIN, "-y", "-i", file_path, "-vframes", "1", "-q:v", "2", tmp_path],
            capture_output=True, timeout=15,
        )
        if result.returncode != 0 or not _os.path.exists(tmp_path):
            return ""
        import cv2  # type: ignore
        frame = cv2.imread(tmp_path)
        _os.unlink(tmp_path)
        if frame is None:
            return ""
        hash_int = _phash_64(frame)
        return format(hash_int, "016x")
    except Exception as exc:
        logger.debug("_compute_video_hash failed for %s: %s", file_path, exc)
        return ""


def collect_training_record(upload_job_id: int, db: Session) -> Optional[TrainingRecord]:
    """Build + persist a TrainingRecord for the given upload.

    Returns a dataclass snapshot; also INSERTs (or UPDATEs on conflict) the
    models.TrainingRecord row. Idempotent via UniqueConstraint on upload_job_id.
    Missing fields degrade to defaults — never raises.
    """
    try:
        import models  # type: ignore

        # ── 1. Load UploadJob ──────────────────────────────────────────────
        upload_job = db.query(models.UploadJob).filter(
            models.UploadJob.id == upload_job_id
        ).first()
        if upload_job is None:
            logger.warning(
                "collect_training_record: UploadJob %s not found", upload_job_id
            )
            return None

        # ── 2. Load Clip ───────────────────────────────────────────────────
        clip = db.query(models.Clip).filter(
            models.Clip.id == upload_job.clip_id
        ).first()
        if clip is None:
            logger.warning(
                "collect_training_record: Clip %s not found for UploadJob %s",
                upload_job.clip_id, upload_job_id,
            )
            return None

        # ── 3. Parse narrative meta from clip.meta JSON ────────────────────
        try:
            meta = json.loads(clip.meta or "{}")
        except (ValueError, TypeError):
            meta = {}

        narrative_role  = str(meta.get("narrative_role") or meta.get("turning_point_label") or "")
        hook_score      = _safe_float(meta.get("hook_score"))
        completion_score = _safe_float(meta.get("completion_score"))
        composite_score = _safe_float(meta.get("composite_score"))

        # ── 4. Niche from Channel.language ─────────────────────────────────
        niche = "news"
        try:
            channel = db.query(models.Channel).filter(
                models.Channel.id == upload_job.channel_id
            ).first()
            if channel and channel.language:
                niche = _LANG_NICHE.get(channel.language.lower(), f"news_{channel.language.lower()}")
        except Exception as exc:
            logger.debug("collect_training_record: channel lookup failed: %s", exc)

        # ── 5. Latest ClipPerformance snapshot (≈T+48h) ────────────────────
        views_48h = 0
        retention_curve: list = []
        shares_per_reach = 0.0
        try:
            perf = (
                db.query(models.ClipPerformance)
                .filter(models.ClipPerformance.upload_job_id == upload_job_id)
                .order_by(models.ClipPerformance.sampled_at.desc())
                .first()
            )
            if perf:
                views_48h = _safe_int(perf.views)
                # shares_per_reach: not a direct column — derive from meta if present
                shares_per_reach = _safe_float(meta.get("shares_per_reach"))
        except Exception as exc:
            logger.debug("collect_training_record: performance lookup failed: %s", exc)

        # ── 6. Video hash ──────────────────────────────────────────────────
        video_hash = _compute_video_hash(clip.file_path or "")

        # ── 7. INSERT or UPDATE training_records row ───────────────────────
        existing = db.query(models.TrainingRecord).filter(
            models.TrainingRecord.upload_job_id == upload_job_id
        ).first()

        if existing is None:
            row = models.TrainingRecord(
                upload_job_id   = upload_job_id,
                clip_id         = clip.id,
                niche           = niche,
                narrative_role  = narrative_role,
                hook_score      = hook_score,
                completion_score = completion_score,
                composite_score = composite_score,
                views_48h       = views_48h,
                retention_curve = retention_curve,
                shares_per_reach = shares_per_reach,
                video_hash      = video_hash,
            )
            db.add(row)
            try:
                db.commit()
                db.refresh(row)
            except IntegrityError:
                db.rollback()
                # Another process may have inserted concurrently — reload
                existing = db.query(models.TrainingRecord).filter(
                    models.TrainingRecord.upload_job_id == upload_job_id
                ).first()
                row = existing
        else:
            # Update in-place (idempotent re-collect)
            existing.clip_id          = clip.id
            existing.niche            = niche
            existing.narrative_role   = narrative_role
            existing.hook_score       = hook_score
            existing.completion_score = completion_score
            existing.composite_score  = composite_score
            existing.views_48h        = views_48h
            existing.retention_curve  = retention_curve
            existing.shares_per_reach = shares_per_reach
            if video_hash:
                existing.video_hash   = video_hash
            db.commit()
            row = existing

        if row is None:
            return None

        collected_at_str = (
            row.collected_at.isoformat() if row.collected_at else None
        )

        return TrainingRecord(
            upload_job_id    = upload_job_id,
            clip_id          = clip.id,
            niche            = niche,
            narrative_role   = narrative_role,
            hook_score       = hook_score,
            completion_score = completion_score,
            composite_score  = composite_score,
            views_48h        = views_48h,
            retention_curve  = retention_curve,
            shares_per_reach = shares_per_reach,
            video_hash       = video_hash,
            collected_at     = collected_at_str,
        )

    except Exception as exc:
        logger.error(
            "collect_training_record(upload_job_id=%s) unexpected error: %s",
            upload_job_id, exc, exc_info=True,
        )
        try:
            db.rollback()
        except Exception:
            pass
        return None


def retrain_narrative_scorer(
    min_records: int = 500,
    niche: Optional[str] = None,
    *,
    out_path: Optional[str] = None,
) -> str:
    """Fine-tune the narrative scorer on collected TrainingRecords.

    Not implemented in v1. See docs/PHASE4_ROADMAP.md § Training Flywheel.
    """
    raise NotImplementedError(
        "training_flywheel.retrain_narrative_scorer is a Phase 4 task. "
        "See docs/PHASE4_ROADMAP.md for the implementation plan."
    )


def deploy_model(model_path: str) -> bool:
    """Atomic swap of the production narrative scorer.

    Not implemented in v1.
    """
    raise NotImplementedError(
        "training_flywheel.deploy_model is a Phase 4 task. "
        "See docs/PHASE4_ROADMAP.md for the implementation plan."
    )
