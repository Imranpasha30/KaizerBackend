"""
kaizer.pipeline.guardrails
===========================
Pre-publish originality and safety guardrails for Kaizer News.

Warns creators BEFORE a video goes live when its content is likely to trip
platform de-duplication, reach-suppression, or demonetisation triggers:

  • Watermark detection — TikTok/CapCut/Snap logos cause IG reach drops.
  • Self-duplicate detection — pHash similarity to user's own recent uploads.
  • Template-repetition detection — same title 3-grams / thumbnail palette.
  • Cadence check — posting too fast or exceeding weekly platform caps.
  • Music rights stub — placeholder for future Content-ID fingerprinting.

Usage
-----
    from pipeline_core.guardrails import run_all_guardrails

    db = SessionLocal()
    report = run_all_guardrails(
        "/path/to/clip.mp4",
        user_id=42,
        platform="instagram_reel",
        db=db,
    )
    if not report.ok:
        for alert in report.all_alerts:
            if alert.severity == "block":
                raise ValueError(alert.message)

GuardrailAlert fields
---------------------
    severity  : str   — 'info' | 'warn' | 'block'
    code      : str   — machine-readable, e.g. 'watermark.tiktok_detected'
    message   : str   — human-readable sentence
    details   : dict  — optional structured data

GuardrailsReport fields
-----------------------
    ok                   : bool                  — False if any alert is severity='block'
    watermark            : WatermarkResult
    duplicate            : DuplicateResult
    repetition           : RepetitionResult
    cadence              : CadenceResult
    music_rights         : MusicRightsResult
    all_alerts           : list[GuardrailAlert]  — all alerts from every sub-check
    warnings             : list[str]             — human-readable summary warnings

Decision flags / intentional stubs
-----------------------------------
  - music_rights is a v1 stub: no network calls, no fingerprint DB required.
    Set fingerprint_db_path to a local DB file path when wiring Phase 4.
  - detect_watermarks uses cv2.matchTemplate at multiple scales.  The template
    directory ships empty; the function emits 'info' (not 'block') when empty.
  - check_self_duplicate only reads files already on disk; pruned clips are
    skipped with a warning rather than hard-failing.
  - check_template_repetition title-shingle logic only; thumbnail dominant-
    colour heuristic runs when clip.thumb_path exists.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger("kaizer.pipeline.guardrails")

# ── Default paths ─────────────────────────────────────────────────────────────

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_TEMPLATES_DIR = os.path.join(
    _BACKEND_ROOT, "resources", "watermark_templates"
)

# ── Result dataclasses ────────────────────────────────────────────────────────


@dataclass
class GuardrailAlert:
    """A single alert produced by any guardrail check.

    Attributes
    ----------
    severity : str
        'info'  — informational; no action required.
        'warn'  — creator should review before publishing.
        'block' — hard stop; platform policy violation likely.
    code : str
        Machine-readable dot-separated identifier, e.g.
        'watermark.tiktok_detected'.
    message : str
        Human-readable sentence suitable for display in the UI.
    details : dict
        Optional structured payload (IDs, similarity scores, etc.).
    """

    severity: str
    code: str
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class WatermarkResult:
    """Result of detect_watermarks().

    Attributes
    ----------
    alerts            : list[GuardrailAlert]
    frames_sampled    : int   — number of video frames analysed
    templates_checked : list[str] — filenames of templates compared
    """

    alerts: list[GuardrailAlert]
    frames_sampled: int
    templates_checked: list[str]


@dataclass
class DuplicateResult:
    """Result of check_self_duplicate().

    Attributes
    ----------
    alerts       : list[GuardrailAlert]
    top_matches  : list[dict]
        Each entry: {upload_job_id, video_id, avg_phash_distance, similarity_pct}
    """

    alerts: list[GuardrailAlert]
    top_matches: list[dict]


@dataclass
class RepetitionResult:
    """Result of check_template_repetition().

    Attributes
    ----------
    alerts                   : list[GuardrailAlert]
    recent_uploads_examined  : int
    detected_pattern         : str | None
        Human-readable description of the detected pattern, or None.
    """

    alerts: list[GuardrailAlert]
    recent_uploads_examined: int
    detected_pattern: Optional[str]


@dataclass
class CadenceResult:
    """Result of check_cadence().

    Attributes
    ----------
    alerts            : list[GuardrailAlert]
    hours_since_last  : float | None   — None when no previous upload exists
    weekly_count      : int
    platform          : str
    """

    alerts: list[GuardrailAlert]
    hours_since_last: Optional[float]
    weekly_count: int
    platform: str


@dataclass
class MusicRightsResult:
    """Result of check_music_rights().

    Attributes
    ----------
    alerts               : list[GuardrailAlert]
    fingerprint_checked  : bool   — False when no fingerprint DB is available
    status               : str    — 'unknown' | 'clean' | 'flagged'
    """

    alerts: list[GuardrailAlert]
    fingerprint_checked: bool
    status: str


@dataclass
class GuardrailsReport:
    """Aggregated result from run_all_guardrails().

    Attributes
    ----------
    ok           : bool               — False if any alert has severity='block'
    watermark    : WatermarkResult
    duplicate    : DuplicateResult
    repetition   : RepetitionResult
    cadence      : CadenceResult
    music_rights : MusicRightsResult
    all_alerts   : list[GuardrailAlert]
    warnings     : list[str]          — prose summaries for logging / UI
    """

    ok: bool
    watermark: WatermarkResult
    duplicate: DuplicateResult
    repetition: RepetitionResult
    cadence: CadenceResult
    music_rights: MusicRightsResult
    all_alerts: list[GuardrailAlert]
    warnings: list[str]


# ── Internal helpers ──────────────────────────────────────────────────────────


def _find_ffmpeg() -> str:
    """Locate the ffmpeg binary: PATH first, then pipeline.py's resolved path."""
    import shutil as _sh

    p = _sh.which("ffmpeg")
    if p:
        return p
    try:
        from pipeline_core.pipeline import FFMPEG_BIN as _ff  # type: ignore
        return _ff
    except Exception:
        pass
    return "ffmpeg"


def _extract_frames_ffmpeg(
    video_path: str,
    n_frames: int,
    *,
    ffmpeg_bin: str,
) -> list["np.ndarray"]:  # type: ignore[name-defined]
    """Extract *n_frames* evenly-spaced frames from *video_path* via FFmpeg.

    Returns a list of BGR numpy arrays (may be fewer than n_frames if the
    video is short or extraction partially fails).  Returns [] on hard failure.
    Imports numpy and cv2 lazily.
    """
    import numpy as np  # lazy — kept out of module-level

    try:
        import cv2  # lazy
    except ImportError:
        logger.warning("guardrails: cv2 not available — frame extraction skipped")
        return []

    # Probe duration
    try:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=nokey=1:noprint_wrappers=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        duration_s = float((probe.stdout or "0").strip())
    except Exception as exc:
        logger.warning("guardrails: ffprobe duration failed for %s: %s", video_path, exc)
        return []

    if duration_s <= 0:
        return []

    # Pick evenly-spaced timestamps, avoiding 0 s and the final frame
    step = duration_s / (n_frames + 1)
    timestamps = [step * (i + 1) for i in range(n_frames)]

    frames: list[np.ndarray] = []
    with tempfile.TemporaryDirectory() as tmp:
        for idx, ts in enumerate(timestamps):
            out_path = os.path.join(tmp, f"frame_{idx:04d}.png")
            cmd = [
                ffmpeg_bin,
                "-hide_banner",
                "-ss", f"{ts:.4f}",
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "2",
                out_path,
            ]
            try:
                subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
                if os.path.exists(out_path):
                    frame = cv2.imread(out_path)
                    if frame is not None:
                        frames.append(frame)
            except Exception as exc:
                logger.debug(
                    "guardrails: frame extraction failed at ts=%.2fs for %s: %s",
                    ts, video_path, exc,
                )

    return frames


def _dominant_color_strip(
    img_bgr: "np.ndarray",  # type: ignore[name-defined]
    *,
    strip_height_frac: float = 0.1,
) -> Optional[tuple[int, int, int]]:
    """Return the (B, G, R) mean colour of the bottom strip of an image.

    Used as a cheap heuristic to detect identical caption-bar backgrounds
    across thumbnails.  Returns None if img_bgr is None or too small.
    """
    try:
        import numpy as np  # lazy

        if img_bgr is None or img_bgr.size == 0:
            return None
        h, w = img_bgr.shape[:2]
        strip_h = max(1, int(h * strip_height_frac))
        strip = img_bgr[h - strip_h:, :]
        mean_bgr = np.mean(strip.reshape(-1, 3), axis=0)
        return (int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2]))
    except Exception:
        return None


def _colors_similar(
    c1: tuple[int, int, int],
    c2: tuple[int, int, int],
    *,
    tolerance: int = 30,
) -> bool:
    """Return True when the Euclidean distance between two BGR colours is below
    *tolerance*."""
    import math

    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
    return dist < tolerance


def _title_3grams(title: str) -> set[str]:
    """Return the set of lowercase word 3-grams from *title*."""
    words = title.lower().split()
    if len(words) < 3:
        # Pad with a sentinel so we still get at least one n-gram
        return {" ".join(words)}
    return {" ".join(words[i: i + 3]) for i in range(len(words) - 2)}


# ── Individual checks ─────────────────────────────────────────────────────────


def detect_watermarks(
    video_path: str,
    *,
    templates_dir: Optional[str] = None,
    sample_frames: int = 8,
    match_threshold: float = 0.7,
) -> WatermarkResult:
    """Sample *sample_frames* frames and run cv2.matchTemplate against known
    watermark PNGs.

    Parameters
    ----------
    video_path : str
        Path to the video file to analyse.
    templates_dir : str | None
        Directory containing watermark PNG templates.  Defaults to
        ``resources/watermark_templates/`` relative to the backend root.
        Created (as empty) if it does not exist; emits 'info' alert when
        no PNGs are found.
    sample_frames : int
        Number of frames to sample from the video.
    match_threshold : float
        cv2.TM_CCOEFF_NORMED score at which a template match is accepted
        (0–1; default 0.7).

    Returns
    -------
    WatermarkResult
        Never raises — all failure paths degrade gracefully.

    Notes
    -----
    Matching runs at three scales (1.0×, 0.75×, 0.5×) to handle logo size
    variation across different device resolutions.
    """
    alerts: list[GuardrailAlert] = []
    templates_checked: list[str] = []
    frames_sampled: int = 0

    if templates_dir is None:
        templates_dir = _DEFAULT_TEMPLATES_DIR

    # Ensure the templates directory exists (create if absent — do not fail)
    try:
        os.makedirs(templates_dir, exist_ok=True)
    except OSError as exc:
        logger.warning("guardrails: cannot create templates_dir %s: %s", templates_dir, exc)

    # Collect PNG templates
    png_files: list[str] = []
    if os.path.isdir(templates_dir):
        for name in sorted(os.listdir(templates_dir)):
            if name.lower().endswith(".png"):
                png_files.append(name)

    if not png_files:
        alerts.append(
            GuardrailAlert(
                severity="info",
                code="watermark.no_templates_available",
                message=(
                    "No watermark templates found in templates_dir — "
                    "watermark detection is disabled. Add PNG templates to "
                    f"enable detection: {templates_dir}"
                ),
                details={"templates_dir": templates_dir},
            )
        )
        return WatermarkResult(
            alerts=alerts,
            frames_sampled=0,
            templates_checked=[],
        )

    # Lazy cv2 import
    try:
        import cv2  # type: ignore
    except ImportError:
        alerts.append(
            GuardrailAlert(
                severity="info",
                code="watermark.cv2_unavailable",
                message="OpenCV (cv2) is not installed — watermark detection skipped.",
            )
        )
        return WatermarkResult(
            alerts=alerts,
            frames_sampled=0,
            templates_checked=[],
        )

    ffmpeg_bin = _find_ffmpeg()
    frames = _extract_frames_ffmpeg(video_path, sample_frames, ffmpeg_bin=ffmpeg_bin)
    frames_sampled = len(frames)

    if frames_sampled == 0:
        alerts.append(
            GuardrailAlert(
                severity="info",
                code="watermark.no_frames_extracted",
                message=f"Could not extract frames from {video_path!r} — watermark detection skipped.",
                details={"video_path": video_path},
            )
        )
        return WatermarkResult(
            alerts=alerts,
            frames_sampled=0,
            templates_checked=[],
        )

    # Load templates
    loaded_templates: list[tuple[str, "np.ndarray"]] = []  # type: ignore[name-defined]
    for name in png_files:
        tpl_path = os.path.join(templates_dir, name)
        try:
            tpl = cv2.imread(tpl_path, cv2.IMREAD_COLOR)
            if tpl is not None and tpl.size > 0:
                loaded_templates.append((name, tpl))
                templates_checked.append(name)
            else:
                logger.warning("guardrails: failed to load template %s", tpl_path)
        except Exception as exc:
            logger.warning("guardrails: error loading template %s: %s", tpl_path, exc)

    if not loaded_templates:
        alerts.append(
            GuardrailAlert(
                severity="info",
                code="watermark.no_templates_loaded",
                message="Template PNGs were found but none could be loaded by cv2.",
            )
        )
        return WatermarkResult(
            alerts=alerts,
            frames_sampled=frames_sampled,
            templates_checked=[],
        )

    # Template matching at multiple scales
    _SCALES = (1.0, 0.75, 0.5)
    triggered_templates: set[str] = set()

    for frame in frames:
        frame_h, frame_w = frame.shape[:2]

        for tpl_name, tpl_bgr in loaded_templates:
            if tpl_name in triggered_templates:
                continue  # already flagged — no need to keep checking

            tpl_h, tpl_w = tpl_bgr.shape[:2]

            for scale in _SCALES:
                sw = int(tpl_w * scale)
                sh = int(tpl_h * scale)

                # Skip if scaled template is bigger than frame or too small
                if sw <= 0 or sh <= 0 or sw >= frame_w or sh >= frame_h:
                    continue

                try:
                    import cv2 as _cv2  # already imported; alias for clarity
                    tpl_scaled = _cv2.resize(tpl_bgr, (sw, sh), interpolation=_cv2.INTER_AREA)
                    result_map = _cv2.matchTemplate(
                        frame, tpl_scaled, _cv2.TM_CCOEFF_NORMED
                    )
                    _, max_val, _, _ = _cv2.minMaxLoc(result_map)
                    if max_val >= match_threshold:
                        triggered_templates.add(tpl_name)
                        logger.info(
                            "guardrails: watermark matched — template=%s scale=%.2f score=%.3f",
                            tpl_name, scale, max_val,
                        )
                        break  # scale loop
                except Exception as exc:
                    logger.debug(
                        "guardrails: matchTemplate error (template=%s scale=%.2f): %s",
                        tpl_name, scale, exc,
                    )

    # Emit block alerts for every matched template
    for tpl_name in sorted(triggered_templates):
        stem = os.path.splitext(tpl_name)[0].lower()
        code = f"watermark.{stem}_detected"
        alerts.append(
            GuardrailAlert(
                severity="block",
                code=code,
                message=(
                    f"Watermark '{stem}' was detected in this video. "
                    "Posting platform-branded watermarks on other platforms "
                    "suppresses reach (IG: up to 80% drop for TikTok-branded content)."
                ),
                details={
                    "template_file": tpl_name,
                    "match_threshold": match_threshold,
                },
            )
        )

    logger.info(
        "guardrails: detect_watermarks — frames=%d templates=%d triggered=%d alerts=%d",
        frames_sampled,
        len(loaded_templates),
        len(triggered_templates),
        len(alerts),
    )

    return WatermarkResult(
        alerts=alerts,
        frames_sampled=frames_sampled,
        templates_checked=templates_checked,
    )


def check_self_duplicate(
    video_path: str,
    *,
    user_id: int,
    db: "Session",  # type: ignore[name-defined]
    frames_sampled: int = 6,
    similarity_threshold_pct: float = 70.0,
    lookback_days: int = 30,
) -> DuplicateResult:
    """Compare *video_path* against the user's recent uploads via pHash.

    Only the most-recent 10 UploadJob rows where the linked Clip.file_path
    exists on disk are examined (skips pruned files with an info log).

    Parameters
    ----------
    video_path : str
        Path to the candidate video.
    user_id : int
        DB user ID to scope the query.
    db : sqlalchemy.orm.Session
        Active session — not closed by this function.
    frames_sampled : int
        Frames to sample per video.
    similarity_threshold_pct : float
        Similarity ≥ this value triggers a 'warn' alert (default: 70 %).
    lookback_days : int
        Only inspect uploads created within this many days.

    Returns
    -------
    DuplicateResult
    """
    alerts: list[GuardrailAlert] = []
    top_matches: list[dict] = []

    # Lazy import of pHash utilities from loop_score (same package)
    try:
        from pipeline_core.loop_score import _phash_64, _hamming  # type: ignore
    except ImportError as exc:
        alerts.append(
            GuardrailAlert(
                severity="info",
                code="duplicate.phash_unavailable",
                message=f"pHash utilities not available — duplicate check skipped: {exc}",
            )
        )
        return DuplicateResult(alerts=alerts, top_matches=[])

    try:
        import cv2  # type: ignore
    except ImportError:
        alerts.append(
            GuardrailAlert(
                severity="info",
                code="duplicate.cv2_unavailable",
                message="OpenCV (cv2) not installed — duplicate check skipped.",
            )
        )
        return DuplicateResult(alerts=alerts, top_matches=[])

    # ── Compute hashes of candidate video ─────────────────────────────────────
    ffmpeg_bin = _find_ffmpeg()
    candidate_frames = _extract_frames_ffmpeg(
        video_path, frames_sampled, ffmpeg_bin=ffmpeg_bin
    )

    if not candidate_frames:
        alerts.append(
            GuardrailAlert(
                severity="info",
                code="duplicate.no_frames_extracted",
                message=f"Could not extract frames from candidate video {video_path!r} — duplicate check skipped.",
                details={"video_path": video_path},
            )
        )
        return DuplicateResult(alerts=alerts, top_matches=[])

    candidate_hashes = [_phash_64(f) for f in candidate_frames]

    # ── Query recent UploadJobs for this user ──────────────────────────────────
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    # Import models lazily — avoids circular import and broken-DB startup issues
    try:
        from models import UploadJob, Clip  # type: ignore
        from sqlalchemy import select  # type: ignore
    except ImportError as exc:
        alerts.append(
            GuardrailAlert(
                severity="info",
                code="duplicate.models_unavailable",
                message=f"DB models not importable — duplicate check skipped: {exc}",
            )
        )
        return DuplicateResult(alerts=alerts, top_matches=[])

    try:
        rows = (
            db.query(
                UploadJob.id.label("upload_job_id"),
                UploadJob.video_id.label("video_id"),
                Clip.file_path.label("file_path"),
            )
            .join(Clip, UploadJob.clip_id == Clip.id)
            .filter(
                UploadJob.user_id == user_id,
                UploadJob.created_at >= cutoff,
                UploadJob.status.in_(["done", "processing"]),
            )
            .order_by(UploadJob.created_at.desc())
            .limit(10)
            .all()
        )
    except Exception as exc:
        alerts.append(
            GuardrailAlert(
                severity="info",
                code="duplicate.db_query_failed",
                message=f"DB query for prior uploads failed — duplicate check skipped: {exc}",
            )
        )
        return DuplicateResult(alerts=alerts, top_matches=[])

    skipped_pruned = 0

    for row in rows:
        prior_path: str = row.file_path or ""

        if not prior_path or not os.path.isfile(prior_path):
            skipped_pruned += 1
            continue

        # Avoid comparing the video against itself
        try:
            if os.path.abspath(prior_path) == os.path.abspath(video_path):
                continue
        except Exception:
            continue

        prior_frames = _extract_frames_ffmpeg(
            prior_path, frames_sampled, ffmpeg_bin=ffmpeg_bin
        )

        if not prior_frames:
            skipped_pruned += 1
            continue

        prior_hashes = [_phash_64(f) for f in prior_frames]

        # Cross-compare all candidate × prior hashes; take the mean distance
        distances = [
            _hamming(ch, ph)
            for ch in candidate_hashes
            for ph in prior_hashes
        ]
        avg_distance = sum(distances) / max(1, len(distances))
        similarity_pct = max(0.0, 100.0 - (avg_distance / 64.0) * 100.0)

        match_entry = {
            "upload_job_id": row.upload_job_id,
            "video_id": row.video_id or "",
            "avg_phash_distance": round(avg_distance, 2),
            "similarity_pct": round(similarity_pct, 1),
        }
        top_matches.append(match_entry)

        if similarity_pct >= similarity_threshold_pct:
            alerts.append(
                GuardrailAlert(
                    severity="warn",
                    code="duplicate.self_repost",
                    message=(
                        f"This video is {similarity_pct:.0f}% visually similar to a "
                        f"prior upload (job #{row.upload_job_id}). "
                        "Re-uploading near-identical content without meaningful changes "
                        "risks 'reused content' demonetisation on YouTube and reach "
                        "suppression on Instagram."
                    ),
                    details={
                        "prior_upload_job_id": row.upload_job_id,
                        "similarity_pct": similarity_pct,
                        "avg_phash_distance": round(avg_distance, 2),
                    },
                )
            )

    # Sort top_matches descending by similarity for UI convenience
    top_matches.sort(key=lambda m: m["similarity_pct"], reverse=True)

    if skipped_pruned:
        logger.info(
            "guardrails: check_self_duplicate — skipped %d pruned/missing prior clips",
            skipped_pruned,
        )

    logger.info(
        "guardrails: check_self_duplicate — compared=%d skipped=%d matches_above_threshold=%d",
        len(rows) - skipped_pruned,
        skipped_pruned,
        sum(1 for a in alerts if a.code == "duplicate.self_repost"),
    )

    return DuplicateResult(alerts=alerts, top_matches=top_matches)


def check_template_repetition(
    *,
    user_id: int,
    db: "Session",  # type: ignore[name-defined]
    lookback_count: int = 5,
) -> RepetitionResult:
    """Detect over-reuse of title templates and thumbnail caption bars.

    Inspects the most-recent *lookback_count* UploadJobs for this user.
    Flags severity='warn' when ≥ 80 % of title word 3-grams are shared
    across all examined uploads, OR when thumbnails share an identical
    dominant bottom-strip colour (same caption-bar background).

    Parameters
    ----------
    user_id : int
    db : Session
    lookback_count : int
        Number of recent uploads to compare (default: 5).

    Returns
    -------
    RepetitionResult
    """
    alerts: list[GuardrailAlert] = []
    detected_pattern: Optional[str] = None
    recent_uploads_examined: int = 0

    try:
        from models import UploadJob, Clip  # type: ignore
    except ImportError as exc:
        alerts.append(
            GuardrailAlert(
                severity="info",
                code="repetition.models_unavailable",
                message=f"DB models not importable — repetition check skipped: {exc}",
            )
        )
        return RepetitionResult(
            alerts=alerts,
            recent_uploads_examined=0,
            detected_pattern=None,
        )

    try:
        rows = (
            db.query(
                UploadJob.id.label("upload_job_id"),
                UploadJob.title.label("title"),
                UploadJob.channel_id.label("channel_id"),
                UploadJob.publish_kind.label("publish_kind"),
                Clip.thumb_path.label("thumb_path"),
            )
            .join(Clip, UploadJob.clip_id == Clip.id)
            .filter(
                UploadJob.user_id == user_id,
                UploadJob.status.in_(["done", "processing"]),
            )
            .order_by(UploadJob.created_at.desc())
            .limit(lookback_count)
            .all()
        )
    except Exception as exc:
        alerts.append(
            GuardrailAlert(
                severity="info",
                code="repetition.db_query_failed",
                message=f"DB query failed — repetition check skipped: {exc}",
            )
        )
        return RepetitionResult(
            alerts=alerts,
            recent_uploads_examined=0,
            detected_pattern=None,
        )

    recent_uploads_examined = len(rows)

    if recent_uploads_examined < 2:
        alerts.append(
            GuardrailAlert(
                severity="info",
                code="repetition.insufficient_history",
                message=(
                    f"Only {recent_uploads_examined} upload(s) found — need at least 2 "
                    "to detect template repetition."
                ),
            )
        )
        return RepetitionResult(
            alerts=alerts,
            recent_uploads_examined=recent_uploads_examined,
            detected_pattern=None,
        )

    # ── Title 3-gram analysis ─────────────────────────────────────────────────
    titles: list[str] = [row.title or "" for row in rows]
    all_ngram_sets: list[set[str]] = [_title_3grams(t) for t in titles if t.strip()]

    title_pattern: Optional[str] = None

    if len(all_ngram_sets) >= 2:
        # Intersection across all title 3-gram sets
        common_ngrams: set[str] = all_ngram_sets[0]
        for s in all_ngram_sets[1:]:
            common_ngrams = common_ngrams & s

        # Union for the coverage denominator
        union_ngrams: set[str] = set()
        for s in all_ngram_sets:
            union_ngrams |= s

        if union_ngrams:
            overlap_ratio = len(common_ngrams) / len(union_ngrams)
            if overlap_ratio >= 0.8 and common_ngrams:
                # Pick the most representative shared 3-gram for the message
                sample = sorted(common_ngrams)[:3]
                snippet = ", ".join(f"'{g}'" for g in sample)
                title_pattern = (
                    f"Last {recent_uploads_examined} titles share 3-gram(s): {snippet}"
                )

    # ── Thumbnail dominant-colour analysis ───────────────────────────────────
    thumb_pattern: Optional[str] = None

    try:
        import cv2  # type: ignore

        colors: list[tuple[int, int, int]] = []
        for row in rows:
            tp: str = row.thumb_path or ""
            if tp and os.path.isfile(tp):
                img = cv2.imread(tp)
                color = _dominant_color_strip(img)
                if color is not None:
                    colors.append(color)

        if len(colors) >= 2:
            # Check if all sampled thumbnail strip colours are within tolerance
            ref = colors[0]
            if all(_colors_similar(ref, c) for c in colors[1:]):
                thumb_pattern = (
                    f"Last {len(colors)} thumbnails share the same caption-bar "
                    f"background colour (BGR ~{ref}) — caption template may be "
                    "over-reused."
                )
    except Exception as exc:
        logger.debug("guardrails: thumbnail colour analysis failed: %s", exc)

    # ── Combine findings ──────────────────────────────────────────────────────
    combined_pattern: Optional[str] = None
    if title_pattern and thumb_pattern:
        combined_pattern = f"{title_pattern}. {thumb_pattern}"
    elif title_pattern:
        combined_pattern = title_pattern
    elif thumb_pattern:
        combined_pattern = thumb_pattern

    if combined_pattern:
        detected_pattern = combined_pattern
        alerts.append(
            GuardrailAlert(
                severity="warn",
                code="repetition.template_overused",
                message=(
                    "Your recent uploads appear to reuse the same title template "
                    "and/or thumbnail design. Platform algorithms may classify this "
                    "as low-originality content and reduce distribution. "
                    f"Pattern detected: {combined_pattern}"
                ),
                details={
                    "title_pattern": title_pattern,
                    "thumbnail_pattern": thumb_pattern,
                    "uploads_examined": recent_uploads_examined,
                },
            )
        )

    logger.info(
        "guardrails: check_template_repetition — examined=%d pattern_found=%s",
        recent_uploads_examined,
        detected_pattern is not None,
    )

    return RepetitionResult(
        alerts=alerts,
        recent_uploads_examined=recent_uploads_examined,
        detected_pattern=detected_pattern,
    )


def check_cadence(
    *,
    user_id: int,
    db: "Session",  # type: ignore[name-defined]
    platform: str,
    min_hours_between_ms: float = 6.0,
    weekly_cap_reel: int = 4,
    weekly_cap_short: int = 21,
) -> CadenceResult:
    """Inspect upload cadence for *user_id* on *platform*.

    Computes:
      - ``hours_since_last``: hours since the most-recent upload for this
        platform family (None when no prior upload exists).
      - ``weekly_count``: successful/in-flight uploads in the last 7 days.

    Platform → publish_kind mapping
    --------------------------------
      youtube_short / instagram_reel / tiktok → 'short'
      youtube_long → 'video'

    Parameters
    ----------
    user_id : int
    db : Session
    platform : str
        One of: 'youtube_short', 'instagram_reel', 'tiktok', 'youtube_long'.
    min_hours_between_ms : float
        Minimum hours between consecutive posts (default: 6).  Reels spam
        filters activate below this threshold.
    weekly_cap_reel : int
        Weekly cap for instagram_reel / tiktok (default: 4).
    weekly_cap_short : int
        Weekly cap for youtube_short (default: 21 — 3/day).

    Returns
    -------
    CadenceResult
    """
    alerts: list[GuardrailAlert] = []

    # Resolve publish_kind from platform
    _SHORT_PLATFORMS = {"youtube_short", "instagram_reel", "tiktok"}
    if platform in _SHORT_PLATFORMS:
        publish_kind_filter = "short"
    else:
        publish_kind_filter = "video"

    try:
        from models import UploadJob  # type: ignore
    except ImportError as exc:
        alerts.append(
            GuardrailAlert(
                severity="info",
                code="cadence.models_unavailable",
                message=f"DB models not importable — cadence check skipped: {exc}",
            )
        )
        return CadenceResult(
            alerts=alerts,
            hours_since_last=None,
            weekly_count=0,
            platform=platform,
        )

    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)

    try:
        # Weekly count
        weekly_count: int = (
            db.query(UploadJob.id)
            .filter(
                UploadJob.user_id == user_id,
                UploadJob.publish_kind == publish_kind_filter,
                UploadJob.created_at >= week_ago,
                UploadJob.status.in_(["done", "processing"]),
            )
            .count()
        )

        # Most-recent upload timestamp
        last_row = (
            db.query(UploadJob.created_at)
            .filter(
                UploadJob.user_id == user_id,
                UploadJob.publish_kind == publish_kind_filter,
                UploadJob.status.in_(["done", "processing"]),
            )
            .order_by(UploadJob.created_at.desc())
            .first()
        )
    except Exception as exc:
        alerts.append(
            GuardrailAlert(
                severity="info",
                code="cadence.db_query_failed",
                message=f"DB query failed — cadence check skipped: {exc}",
            )
        )
        return CadenceResult(
            alerts=alerts,
            hours_since_last=None,
            weekly_count=0,
            platform=platform,
        )

    # Compute hours_since_last (tz-aware arithmetic)
    hours_since_last: Optional[float] = None
    if last_row and last_row.created_at:
        last_ts = last_row.created_at
        # Ensure tz-aware for safe subtraction
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        delta_s = (now - last_ts).total_seconds()
        hours_since_last = delta_s / 3600.0

    # ── Alert logic ───────────────────────────────────────────────────────────

    if weekly_count == 0 and hours_since_last is None:
        alerts.append(
            GuardrailAlert(
                severity="info",
                code="cadence.first_post",
                message=(
                    f"This appears to be your first post on {platform}. "
                    "No cadence history to analyse — great time to start a consistent schedule!"
                ),
                details={"platform": platform},
            )
        )
    else:
        # Too soon check
        if hours_since_last is not None and hours_since_last < min_hours_between_ms:
            alerts.append(
                GuardrailAlert(
                    severity="warn",
                    code="cadence.too_soon",
                    message=(
                        f"Your last {platform} upload was only "
                        f"{hours_since_last:.1f} hours ago (minimum recommended gap: "
                        f"{min_hours_between_ms:.0f} hours). Posting too rapidly risks "
                        "triggering spam filters — especially Instagram Reels reach suppression."
                    ),
                    details={
                        "hours_since_last": round(hours_since_last, 2),
                        "min_hours_recommended": min_hours_between_ms,
                        "platform": platform,
                    },
                )
            )

        # Weekly cap check
        if platform == "instagram_reel" or platform == "tiktok":
            cap = weekly_cap_reel
        elif platform == "youtube_short":
            cap = weekly_cap_short
        else:
            cap = None  # youtube_long: no specific cap enforced here

        if cap is not None and weekly_count > cap:
            alerts.append(
                GuardrailAlert(
                    severity="warn",
                    code="cadence.weekly_cap_exceeded",
                    message=(
                        f"You have posted {weekly_count} times in the last 7 days on "
                        f"{platform} (recommended cap: {cap}). Exceeding platform cadence "
                        "limits can cause reach throttling or spam classification."
                    ),
                    details={
                        "weekly_count": weekly_count,
                        "weekly_cap": cap,
                        "platform": platform,
                    },
                )
            )

    logger.info(
        "guardrails: check_cadence — platform=%s publish_kind=%s "
        "hours_since_last=%s weekly_count=%d alerts=%d",
        platform,
        publish_kind_filter,
        f"{hours_since_last:.1f}" if hours_since_last is not None else "None",
        weekly_count,
        len(alerts),
    )

    return CadenceResult(
        alerts=alerts,
        hours_since_last=hours_since_last,
        weekly_count=weekly_count,
        platform=platform,
    )


def check_music_rights(
    video_path: str,
    *,
    fingerprint_db_path: Optional[str] = None,
) -> MusicRightsResult:
    """v1 stub — music rights fingerprinting is deferred to Phase 4.

    When *fingerprint_db_path* is None or does not exist on disk, returns
    status='unknown' with an 'info' alert.  No network calls are made.

    Parameters
    ----------
    video_path : str
        Path to the video file (accepted but not analysed in v1).
    fingerprint_db_path : str | None
        Path to a local Content-ID-style fingerprint database.  When provided
        and the file exists, a real lookup will be implemented in Phase 4.

    Returns
    -------
    MusicRightsResult
    """
    alerts: list[GuardrailAlert] = []

    db_missing = fingerprint_db_path is None or not os.path.exists(
        fingerprint_db_path or ""
    )

    if db_missing:
        alerts.append(
            GuardrailAlert(
                severity="info",
                code="music_rights.fingerprint_unavailable",
                message=(
                    "Music rights fingerprint database is not configured. "
                    "Audio Content-ID checking will be available in Phase 4 "
                    "when a fingerprint DB is wired. No action required for now."
                ),
                details={
                    "fingerprint_db_path": fingerprint_db_path,
                    "note": "v1 stub — Phase 4 will implement real fingerprint lookup",
                },
            )
        )
        logger.info(
            "guardrails: check_music_rights — stub (no fingerprint_db_path); status=unknown"
        )
        return MusicRightsResult(
            alerts=alerts,
            fingerprint_checked=False,
            status="unknown",
        )

    # Phase 4 placeholder: fingerprint_db_path exists — actual lookup not yet
    # implemented.  Emit info rather than silently returning 'clean'.
    alerts.append(
        GuardrailAlert(
            severity="info",
            code="music_rights.fingerprint_lookup_not_implemented",
            message=(
                "A fingerprint database was found but the lookup logic is not yet "
                "implemented (Phase 4). Status reported as 'unknown'."
            ),
            details={"fingerprint_db_path": fingerprint_db_path},
        )
    )
    logger.info(
        "guardrails: check_music_rights — fingerprint_db exists but lookup stub; status=unknown"
    )
    return MusicRightsResult(
        alerts=alerts,
        fingerprint_checked=False,
        status="unknown",
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────


def run_all_guardrails(
    video_path: str,
    *,
    user_id: int,
    platform: str,
    db: "Session",  # type: ignore[name-defined]
    skip: Optional[list[str]] = None,
    templates_dir: Optional[str] = None,
    fingerprint_db_path: Optional[str] = None,
) -> GuardrailsReport:
    """Run every guardrail check and assemble a :class:`GuardrailsReport`.

    Parameters
    ----------
    video_path : str
        Path to the video file to analyse.
    user_id : int
        DB user ID — used to scope duplicate / repetition / cadence queries.
    platform : str
        One of: 'youtube_short', 'instagram_reel', 'tiktok', 'youtube_long'.
    db : sqlalchemy.orm.Session
        Active SQLAlchemy session.  Not closed by this function.
    skip : list[str] | None
        Names of checks to skip entirely.  Valid names:
        'watermark', 'duplicate', 'repetition', 'cadence', 'music_rights'.
        Skipped checks receive a default empty result with an 'info' alert
        code='<check>.skipped'.
    templates_dir : str | None
        Passed to :func:`detect_watermarks`.
    fingerprint_db_path : str | None
        Passed to :func:`check_music_rights`.

    Returns
    -------
    GuardrailsReport
        ``report.ok`` is False if any alert across any check has
        severity='block'.

        Individual checks that raise unexpectedly are caught — the exception
        is recorded as an 'info' alert so the report is always returned.

    Notes
    -----
    Checks execute sequentially in the order:
    watermark → duplicate → repetition → cadence → music_rights.
    """
    skip_set: set[str] = set(skip or [])
    warnings_prose: list[str] = []

    def _skipped_result(check_name: str) -> GuardrailAlert:
        return GuardrailAlert(
            severity="info",
            code=f"{check_name}.skipped",
            message=f"Guardrail '{check_name}' was skipped by caller request.",
        )

    def _error_result(check_name: str, exc: BaseException) -> GuardrailAlert:
        msg = f"Guardrail '{check_name}' raised an unexpected exception: {exc!r}"
        logger.exception("guardrails: %s raised: %s", check_name, exc)
        warnings_prose.append(msg)
        return GuardrailAlert(
            severity="info",
            code=f"{check_name}.internal_error",
            message=msg,
        )

    # ── watermark ─────────────────────────────────────────────────────────────
    if "watermark" in skip_set:
        watermark_result = WatermarkResult(
            alerts=[_skipped_result("watermark")],
            frames_sampled=0,
            templates_checked=[],
        )
    else:
        try:
            watermark_result = detect_watermarks(
                video_path,
                templates_dir=templates_dir,
            )
        except Exception as exc:
            watermark_result = WatermarkResult(
                alerts=[_error_result("watermark", exc)],
                frames_sampled=0,
                templates_checked=[],
            )

    # ── duplicate ─────────────────────────────────────────────────────────────
    if "duplicate" in skip_set:
        duplicate_result = DuplicateResult(
            alerts=[_skipped_result("duplicate")],
            top_matches=[],
        )
    else:
        try:
            duplicate_result = check_self_duplicate(
                video_path,
                user_id=user_id,
                db=db,
            )
        except Exception as exc:
            duplicate_result = DuplicateResult(
                alerts=[_error_result("duplicate", exc)],
                top_matches=[],
            )

    # ── repetition ────────────────────────────────────────────────────────────
    if "repetition" in skip_set:
        repetition_result = RepetitionResult(
            alerts=[_skipped_result("repetition")],
            recent_uploads_examined=0,
            detected_pattern=None,
        )
    else:
        try:
            repetition_result = check_template_repetition(
                user_id=user_id,
                db=db,
            )
        except Exception as exc:
            repetition_result = RepetitionResult(
                alerts=[_error_result("repetition", exc)],
                recent_uploads_examined=0,
                detected_pattern=None,
            )

    # ── cadence ───────────────────────────────────────────────────────────────
    if "cadence" in skip_set:
        cadence_result = CadenceResult(
            alerts=[_skipped_result("cadence")],
            hours_since_last=None,
            weekly_count=0,
            platform=platform,
        )
    else:
        try:
            cadence_result = check_cadence(
                user_id=user_id,
                db=db,
                platform=platform,
            )
        except Exception as exc:
            cadence_result = CadenceResult(
                alerts=[_error_result("cadence", exc)],
                hours_since_last=None,
                weekly_count=0,
                platform=platform,
            )

    # ── music_rights ──────────────────────────────────────────────────────────
    if "music_rights" in skip_set:
        music_result = MusicRightsResult(
            alerts=[_skipped_result("music_rights")],
            fingerprint_checked=False,
            status="unknown",
        )
    else:
        try:
            music_result = check_music_rights(
                video_path,
                fingerprint_db_path=fingerprint_db_path,
            )
        except Exception as exc:
            music_result = MusicRightsResult(
                alerts=[_error_result("music_rights", exc)],
                fingerprint_checked=False,
                status="unknown",
            )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    all_alerts: list[GuardrailAlert] = (
        watermark_result.alerts
        + duplicate_result.alerts
        + repetition_result.alerts
        + cadence_result.alerts
        + music_result.alerts
    )

    ok = not any(a.severity == "block" for a in all_alerts)

    # Prose warnings for logging convenience
    for a in all_alerts:
        if a.severity in ("warn", "block"):
            warnings_prose.append(f"[{a.severity.upper()}] {a.code}: {a.message}")

    logger.info(
        "guardrails: run_all_guardrails — video=%s user=%d platform=%s "
        "ok=%s total_alerts=%d blocks=%d warns=%d",
        video_path,
        user_id,
        platform,
        ok,
        len(all_alerts),
        sum(1 for a in all_alerts if a.severity == "block"),
        sum(1 for a in all_alerts if a.severity == "warn"),
    )

    return GuardrailsReport(
        ok=ok,
        watermark=watermark_result,
        duplicate=duplicate_result,
        repetition=repetition_result,
        cadence=cadence_result,
        music_rights=music_result,
        all_alerts=all_alerts,
        warnings=warnings_prose,
    )
