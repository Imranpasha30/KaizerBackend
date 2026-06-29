"""Brand profile resolver — Phase 2.D Branding agent.

Resolves a fully-materialised brand for a given channel, walking the
precedence chain documented in CONTRACTS.md §4.3 + DECISIONS.md
Decision 11/12:

  1. BrandProfile (NEW world) — first try owner_kind='oauth_token',
     then 'channel', then 'user'. Whole row wins.
  2. Synthesised from legacy Channel/OAuthToken/User columns when no
     BrandProfile row exists (transition window before Phase 3 column
     drops).

Returns a transient ``ResolvedBrand`` dataclass — NOT a SQLAlchemy
object — so callers don't accidentally persist a synthesised brand.

The resolver does NOT mutate the database. The version hash is the
stable cache-key input for the Branding worker
(``brand_artifact_cache_key``).

Owned by: D-agent (CONTRACTS.md §4.3, Decision 13).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session

import models

log = logging.getLogger("kaizer.branding.resolver")


@dataclass
class ResolvedBrand:
    """A fully-materialised brand for a single upload destination.

    Either ``brand_profile_id`` is non-None (the resolver loaded a real
    BrandProfile row) or it is None (the resolver synthesised an
    ephemeral brand from the legacy Channel/OAuthToken/User columns).
    """

    # Real DB row id, or None for a synthesised legacy brand.
    brand_profile_id: Optional[int]
    # Stable short hash of the materialised inputs — feeds into the
    # branded-artifact cache key. See ``_version_hash`` below.
    version: str
    # FK into user_assets — what the renderer overlays as the top-right bug.
    logo_asset_id: Optional[int]
    # Materialised local filesystem path of the logo PNG/JPG. None when
    # there is no logo OR when the asset_resolver couldn't fetch it.
    logo_local_path: Optional[str]
    # The text watermark string (never None — empty means "no text overlay").
    watermark_text: str
    # 0.05 .. 1.0 — clamped before use.
    watermark_opacity: float
    # 'lower-center' | 'upper-center' | 'center-left' | 'center-right'
    # (legacy corner names map to lower-center; see pipeline_v4/watermark.py).
    watermark_position: str
    # JSON-encoded socials dict the SEO footer composer reads. Always a
    # JSON string (empty dict ``"{}"`` when nothing's configured) so the
    # version hash is stable across reruns.
    socials_json: str
    # Per-channel A/V pacing nudge factor (1.0 = no nudge). When > 1.0 the
    # branding pass speeds BOTH video (setpts=PTS/F) and audio (atempo=F) by
    # the SAME factor, so the same master renders to a distinct audio+video
    # fingerprint per channel WITHOUT lip-sync drift — defeating YouTube's
    # near-duplicate suppression. Derived deterministically from channel.id
    # and gated by KAIZER_BRAND_AV_NUDGE; folded into ``version`` so the cache
    # forks per channel. Default 1.0 keeps output byte-identical (cache-safe).
    nudge_factor: float = 1.0
    # FK into user_assets for the per-channel INTRO video, concatenated at the
    # HEAD of the branded clip (anti-duplicate). None = no intro. Folded into
    # ``version`` so a channel's intro forks its cache; absent = hash unchanged.
    intro_asset_id: Optional[int] = None
    # Materialised local path of the intro video (None when absent/unfetchable).
    intro_local_path: Optional[str] = None
    # Per-channel VISUAL zoom factor (1.0 = none). When > 1.0 the branding pass
    # zooms the PICTURE by this factor (scale-up + center-crop back to the exact
    # canvas) so the perceptual VIDEO hash differs per channel — the picture
    # half of the anti-dup, complementing the audio nudge. ~2-4% = nearly
    # invisible to viewers but a real macro change the perceptual hash keeps.
    # Derived from channel.id, gated by KAIZER_BRAND_VISUAL_ZOOM, folded into
    # ``version``. 1.0 = byte-identical (cache-safe).
    zoom_factor: float = 1.0


# ─── Internal helpers ─────────────────────────────────────────────────────


def _version_hash(
    logo_asset_id: Optional[int],
    wm_text: str,
    wm_opacity: float,
    wm_pos: str,
    socials_json: str,
    nudge_factor: float = 1.0,
    intro_asset_id: Optional[int] = None,
    zoom_factor: float = 1.0,
) -> str:
    """Stable short hash of the materialised brand inputs.

    Used as the cache-key suffix (CONTRACTS §4.3, Decision 11):

        branded/{master_video_id}/{brand_profile_version}.mp4

    16 hex chars = 64 bits, vastly more than enough for our population
    of distinct brands. We deliberately serialise opacity to 3 decimals
    so 0.350001 vs 0.3500009 don't fork the cache.
    """
    # Render-algorithm version. Bump this whenever the OVERLAY logic
    # changes (not the inputs) so previously-cached branded artifacts are
    # invalidated and re-rendered. r2 = watermark positioning fix
    # (corners honoured + Shorts lift watermark out of the news-card area).
    _RENDER_ALGO_VERSION = "r3"  # r3 = per-channel pan/framing offset + micro color-grade
    payload = (
        f"{_RENDER_ALGO_VERSION}|"
        f"{logo_asset_id if logo_asset_id is not None else 'none'}|"
        f"{wm_text}|"
        f"{wm_opacity:.3f}|"
        f"{wm_pos}|"
        f"{socials_json}"
    )
    # Append the A/V-nudge token ONLY when the nudge is active. A non-nudged
    # brand (factor 1.0) therefore hashes IDENTICALLY to before this feature
    # shipped — existing cached branded artifacts stay valid and nothing
    # re-renders unless the operator turns the nudge on for a channel. When
    # active, the distinct factor forks the cache so each channel gets its
    # own fingerprint-differentiated render. (No _RENDER_ALGO_VERSION bump
    # needed: the OFF path is byte-identical, the ON path forks via this token.)
    # Same conditional-append discipline for the per-channel intro: absent ⇒
    # hash unchanged (cache-safe); present ⇒ forks so the intro re-renders.
    if intro_asset_id:
        payload += f"|intro={int(intro_asset_id)}"
    if abs(float(nudge_factor) - 1.0) > 1e-6:
        payload += f"|nudge={float(nudge_factor):.4f}"
    # Same conditional-append for the per-channel visual zoom (picture-half
    # anti-dup): 1.0 ⇒ hash unchanged (cache-safe); >1.0 ⇒ forks per channel.
    if abs(float(zoom_factor) - 1.0) > 1e-6:
        payload += f"|zoom={float(zoom_factor):.4f}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _flag(name: str, default: bool) -> bool:
    """Tri-state env flag: '1' forces ON, '0' forces OFF, unset → default."""
    v = (os.environ.get(name) or "").strip()
    if v == "1":
        return True
    if v == "0":
        return False
    return default


def _antidup_on() -> bool:
    """Master anti-duplicate switch — ON by default. So the SAME master rendered
    to multiple channels gets a per-channel-distinct audio+video fingerprint
    (nudge + zoom) without anyone flipping a flag. Each lever can still be
    force-disabled individually via its own KAIZER_BRAND_* flag, and the whole
    system off via KAIZER_BRAND_ANTIDUP=0."""
    return _flag("KAIZER_BRAND_ANTIDUP", True)


def _derive_nudge_factor(channel_id: int) -> float:
    """Per-channel A/V pacing nudge, derived deterministically from channel.id.

    Returns 1.0 (no nudge) unless ``KAIZER_BRAND_AV_NUDGE=1``. When enabled,
    maps the channel id into a small, imperceptible speed-up factor
    (1.012 .. 1.028, i.e. +1.2% .. +2.8%) so the SAME master video renders to
    a distinct audio+video fingerprint per channel — defeating YouTube's
    near-duplicate suppression with NO lip-sync drift (the branding pass
    applies video setpts and audio atempo by this SAME factor).

    Deterministic ⇒ stable across re-renders ⇒ the branded-artifact cache key
    stays consistent for a given channel (no needless re-encodes). All factors
    are strictly > 1.0 so the output is always slightly SHORTER than the master
    (never longer), keeping the QC duration gate one-sided and simple.

    No DB column required: the value is a pure function of ``channel.id`` +
    the env flag, so it auto-applies to every existing and future channel.
    """
    # ON by default (master anti-dup); set KAIZER_BRAND_AV_NUDGE=0 to disable just this lever.
    if not _flag("KAIZER_BRAND_AV_NUDGE", _antidup_on()):
        return 1.0
    try:
        cid = int(channel_id)
    except (TypeError, ValueError):
        return 1.0
    h = int(hashlib.sha256(f"avnudge:{cid}".encode("utf-8")).hexdigest()[:8], 16)
    factor = 1.012 + (h % 17) * 0.001  # 17 stable buckets: 1.012 .. 1.028
    # Hard safety clamp — an env/code mistake can never ship a chipmunk render.
    factor = max(1.0, min(1.05, factor))
    return round(factor, 4)


def _derive_zoom_factor(channel_id: int) -> float:
    """Per-channel VISUAL zoom, deterministic from channel.id.

    Returns 1.0 (no zoom) unless ``KAIZER_BRAND_VISUAL_ZOOM=1``. When enabled,
    maps the channel id into a small, near-invisible zoom (1.02 .. 1.04, i.e.
    +2%..+4%): the branding pass scales the picture up by this factor and
    center-crops back to the exact canvas, so the perceptual VIDEO fingerprint
    differs per channel — the PICTURE half of the anti-dup, paired with the
    audio nudge (which breaks audio + timing). Barely noticeable to viewers but
    a real macro change the perceptual hash keeps (unlike invisible watermarks,
    which it discards). Deterministic ⇒ cache-stable; no DB column.
    """
    # ON by default (master anti-dup); set KAIZER_BRAND_VISUAL_ZOOM=0 to disable just this lever.
    if not _flag("KAIZER_BRAND_VISUAL_ZOOM", _antidup_on()):
        return 1.0
    try:
        cid = int(channel_id)
    except (TypeError, ValueError):
        return 1.0
    h = int(hashlib.sha256(f"vzoom:{cid}".encode("utf-8")).hexdigest()[:8], 16)
    factor = 1.02 + (h % 11) * 0.002  # 11 stable buckets: 1.020 .. 1.040
    # Hard clamp — never an extreme crop that loses edge text/lower-thirds.
    factor = max(1.0, min(1.06, factor))
    return round(factor, 4)


def _derive_pan_offset(channel_id: int) -> tuple:
    """Per-channel FRAMING offset (fractions 0..1) for the zoom crop window, so each channel's
    framing differs — not just its scale. (0.5, 0.5) = centered (no pan) when zoom is off."""
    if not _flag("KAIZER_BRAND_VISUAL_ZOOM", _antidup_on()):
        return (0.5, 0.5)
    try:
        cid = int(channel_id)
    except (TypeError, ValueError):
        return (0.5, 0.5)
    hx = int(hashlib.sha256(f"vpanx:{cid}".encode("utf-8")).hexdigest()[:8], 16)
    hy = int(hashlib.sha256(f"vpany:{cid}".encode("utf-8")).hexdigest()[:8], 16)
    return (round(0.30 + (hx % 9) * 0.05, 3), round(0.30 + (hy % 9) * 0.05, 3))   # 0.30 .. 0.70


def _derive_grade(channel_id: int) -> tuple:
    """Per-channel INVISIBLE colour micro-grade (brightness delta, saturation factor) so the
    picture fingerprint shifts a touch more. (0.0, 1.0) = no grade when anti-dup is off.
    Hard-clamped so it can never be a visible colour shift."""
    if not _antidup_on():
        return (0.0, 1.0)
    try:
        cid = int(channel_id)
    except (TypeError, ValueError):
        return (0.0, 1.0)
    hb = int(hashlib.sha256(f"vgradeb:{cid}".encode("utf-8")).hexdigest()[:8], 16)
    hs = int(hashlib.sha256(f"vgrades:{cid}".encode("utf-8")).hexdigest()[:8], 16)
    bright = max(-0.05, min(0.05, round(-0.03 + (hb % 7) * 0.01, 3)))   # -0.03 .. +0.03
    sat = max(0.90, min(1.10, round(0.96 + (hs % 9) * 0.01, 3)))        # 0.96 .. 1.04
    return (bright, sat)


def _materialise_asset(
    db: Session, asset_id: Optional[int], label: str = "asset",
) -> Optional[str]:
    """Pull a UserAsset row and resolve it to a local file path.

    Generic id→path resolver (logo, intro video, …). Returns None on any
    failure rather than raising — branding still works without the asset.
    """
    if not asset_id:
        return None
    try:
        asset = db.query(models.UserAsset).filter(
            models.UserAsset.id == int(asset_id)
        ).first()
        if asset is None:
            log.warning(
                "brand_resolver: UserAsset id=%d not found; skipping %s",
                asset_id, label,
            )
            return None
        # Match pipeline_v4/watermark.py's behaviour exactly.
        try:
            from asset_resolver import materialize_asset_locally
        except Exception:
            log.warning(
                "brand_resolver: asset_resolver not importable; using "
                "raw UserAsset.file_path fallback",
            )
            fp = (getattr(asset, "file_path", "") or "").strip()
            return fp if (fp and os.path.isfile(fp)) else None
        local = materialize_asset_locally(asset)
        if local and os.path.isfile(local):
            return local
        # Direct path fallback when materialise returned empty.
        fp = (getattr(asset, "file_path", "") or "").strip()
        if fp and os.path.isfile(fp):
            return fp
        log.warning(
            "brand_resolver: UserAsset id=%d (%s) has no usable local bytes",
            asset_id, label,
        )
        return None
    except Exception as exc:
        log.warning(
            "brand_resolver: failed to materialise %s asset id=%s: %s",
            label, asset_id, exc,
        )
        return None


def _materialise_logo(db: Session, logo_asset_id: Optional[int]) -> Optional[str]:
    """Logo asset_id → local path (thin wrapper over _materialise_asset)."""
    return _materialise_asset(db, logo_asset_id, label="logo overlay")


def _resolve_user_v4_defaults(user: Optional[models.User]) -> dict:
    """Read User.v4_defaults JSON-as-text (routers/v4_defaults.py shape)."""
    if user is None:
        return {}
    raw = getattr(user, "v4_defaults", None)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _resolve_socials(channel: models.Channel) -> dict:
    """Channel-wins precedence per pipeline_v4/watermark.py behaviour.

    The channel.socials column holds either a dict, a JSON string, or
    None. We always return a dict for downstream callers.
    """
    raw = getattr(channel, "socials", None)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _normalize_position(raw: Optional[str]) -> str:
    """Coerce legacy corner names to lower-center (matches
    pipeline_v4/watermark.py:_watermark_overlay_xy)."""
    if not raw:
        return "lower-center"
    p = raw.strip().lower().replace("_", "-")
    if p in ("top-left", "top-right", "bottom-left", "bottom-right"):
        return "lower-center"
    if p in ("upper-center", "lower-center", "center-left", "center-right"):
        return p
    return "lower-center"


def _clamp_opacity(raw) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.35
    if v < 0.05:
        return 0.05
    if v > 1.0:
        return 1.0
    return v


# ─── Public API ───────────────────────────────────────────────────────────


def _try_brand_profile(
    db: Session,
    owner_kind: str,
    owner_id: Optional[int],
) -> Optional[models.BrandProfile]:
    if not owner_id:
        return None
    return (
        db.query(models.BrandProfile)
        .filter(
            models.BrandProfile.owner_kind == owner_kind,
            models.BrandProfile.owner_id == int(owner_id),
        )
        .first()
    )


def _brand_is_empty(brand: ResolvedBrand) -> bool:
    """A brand with no logo, no watermark text, no socials, AND no intro —
    nothing for the worker to overlay or prepend. (A channel that only set an
    intro is NOT empty: it must keep its OWN intro rather than inheriting a
    sibling's brand.) The nudge is intentionally excluded — it's a derived
    per-channel knob, not author-set brand content."""
    return (
        not brand.logo_asset_id
        and not (brand.watermark_text or "").strip()
        and (brand.socials_json or "") in ("", "{}")
        and not brand.intro_asset_id
    )


def _find_branded_sibling(db: Session, channel_id: int) -> Optional[int]:
    """Find another connected channel for the SAME real YouTube account
    (same ``google_channel_id``, same user) that DOES carry branding.

    This is what makes branding account-scoped: the user connected the
    same channel as several profiles (e.g. Auto Wala = Personal 9 +
    Personal 10) and only set the logo on one of them. Publishing
    through the empty one should still wear the account's brand.
    Returns the sibling channel id, or None.
    """
    ch = db.query(models.Channel).filter(models.Channel.id == int(channel_id)).first()
    if ch is None:
        return None
    tok = getattr(ch, "oauth_token", None)
    gcid = (getattr(tok, "google_channel_id", "") or "").strip() if tok else ""
    if not gcid:
        return None
    siblings = (
        db.query(models.Channel)
        .join(models.OAuthToken, models.OAuthToken.channel_id == models.Channel.id)
        .filter(
            models.Channel.user_id == ch.user_id,
            models.Channel.id != ch.id,
            models.OAuthToken.google_channel_id == gcid,
            # Inherit branding only from a REAL connected account, never a
            # style reference that happens to share this google_channel_id.
            models.OAuthToken.refresh_token_enc.isnot(None),
            models.OAuthToken.refresh_token_enc != "",
        )
        .all()
    )
    # Deterministic: lowest id among siblings that actually has branding.
    for s in sorted(siblings, key=lambda c: c.id):
        s_tok = getattr(s, "oauth_token", None)
        has_logo = bool(
            (getattr(s_tok, "logo_asset_id", None) if s_tok else None)
            or getattr(s, "logo_asset_id", None)
        )
        has_wm = bool((getattr(s, "watermark_text", "") or "").strip())
        has_soc = bool(_resolve_socials(s))
        if has_logo or has_wm or has_soc:
            return int(s.id)
    return None


def resolve_brand_profile(
    db: Session, channel_id: int, job_intro_asset_id: Optional[int] = None,
) -> ResolvedBrand:
    """Return a fully-resolved brand for ``channel_id``, account-scoped.

    First resolves for the channel itself; if that yields an EMPTY brand
    (no logo/watermark/socials), inherits the brand from a sibling
    connection of the SAME real YouTube account — so every profile of an
    account wears that account's branding consistently, no matter which
    one the publish routed through.

    ``job_intro_asset_id`` (when > 0) is a PER-JOB intro override that wins
    over the channel's assigned intro for THIS job's renders/publishes.
    Default ``None`` = unchanged behaviour (each channel uses its own intro).
    It's folded into the version hash, so an override forks the brand cache
    cleanly (no collision with the non-override artifact).
    """
    # The nudge is keyed to the REQUESTED channel (the one actually
    # publishing), not the sibling we may borrow visual branding from — so
    # two distinct real accounts always diverge even when one inherits the
    # other's logo/watermark. Computed once and threaded into both resolves.
    nudge = _derive_nudge_factor(channel_id)
    zoom = _derive_zoom_factor(channel_id)
    brand = _resolve_for_channel(db, channel_id, nudge_factor=nudge, zoom_factor=zoom,
                                 job_intro_asset_id=job_intro_asset_id)
    if _brand_is_empty(brand):
        sib = _find_branded_sibling(db, channel_id)
        if sib is not None:
            sib_brand = _resolve_for_channel(db, sib, nudge_factor=nudge, zoom_factor=zoom,
                                             job_intro_asset_id=job_intro_asset_id)
            if not _brand_is_empty(sib_brand):
                log.info(
                    "brand_resolver: channel_id=%d has no brand of its own; "
                    "inheriting account-level branding from sibling channel "
                    "id=%d (same YouTube account)", channel_id, sib,
                )
                return sib_brand
    return brand


def _resolve_for_channel(
    db: Session, channel_id: int, nudge_factor: float = 1.0, zoom_factor: float = 1.0,
    job_intro_asset_id: Optional[int] = None,
) -> ResolvedBrand:
    """Resolve a brand for ONE channel row (no account-level fallback).

    Precedence (CONTRACTS §4.3, mirrors
    pipeline_v4/watermark.py:_resolve_channel_logo:39-91):

      1. BrandProfile WHERE owner_kind='oauth_token' AND owner_id=channel.oauth_token.id
      2. BrandProfile WHERE owner_kind='channel'     AND owner_id=channel.id
      3. BrandProfile WHERE owner_kind='user'        AND owner_id=channel.user_id
      4. Synthesise from legacy Channel/OAuthToken/User columns.

    Returns a ResolvedBrand. Never raises — degrades to a no-watermark
    no-logo brand when everything is empty (the branding worker
    short-circuits to a copy in that case).
    """
    channel = (
        db.query(models.Channel).filter(models.Channel.id == int(channel_id)).first()
    )
    if channel is None:
        raise ValueError(f"brand_resolver: channel_id={channel_id} not found")

    oauth_token = getattr(channel, "oauth_token", None)
    user = (
        db.query(models.User).filter(models.User.id == int(channel.user_id)).first()
        if channel.user_id is not None
        else None
    )

    # ── 1/2/3: real BrandProfile rows ──────────────────────────────────
    real_bp: Optional[models.BrandProfile] = None
    if oauth_token is not None:
        real_bp = _try_brand_profile(db, "oauth_token", oauth_token.id)
    if real_bp is None:
        real_bp = _try_brand_profile(db, "channel", channel.id)
    if real_bp is None and channel.user_id is not None:
        real_bp = _try_brand_profile(db, "user", channel.user_id)

    if real_bp is not None:
        wm_text = (real_bp.watermark_text or "").strip()
        wm_opacity = _clamp_opacity(real_bp.watermark_opacity)
        wm_pos = _normalize_position(real_bp.watermark_position)
        socials_dict = {}
        if real_bp.socials_json:
            try:
                parsed = json.loads(real_bp.socials_json)
                socials_dict = parsed if isinstance(parsed, dict) else {}
            except Exception:
                socials_dict = {}
        # Stable JSON serialisation (sorted keys) so two semantically
        # identical brands hash the same regardless of insertion order.
        socials_json = json.dumps(socials_dict, sort_keys=True, separators=(",", ":"))
        logo_path = _materialise_logo(db, real_bp.logo_asset_id)
        # Intro precedence: a PER-JOB override wins; else the BrandProfile's
        # own intro; else the channel-level column (legacy).
        if job_intro_asset_id and int(job_intro_asset_id) > 0:
            intro_asset_id = int(job_intro_asset_id)
        else:
            intro_asset_id = getattr(real_bp, "intro_asset_id", None) or getattr(
                channel, "intro_asset_id", None
            )
            intro_asset_id = int(intro_asset_id) if intro_asset_id else None
        intro_path = _materialise_asset(db, intro_asset_id, label="intro video")
        version = _version_hash(
            real_bp.logo_asset_id, wm_text, wm_opacity, wm_pos, socials_json,
            nudge_factor, intro_asset_id, zoom_factor,
        )
        log.info(
            "brand_resolver: channel_id=%d → BrandProfile id=%d (owner=%s/%d) "
            "version=%s nudge=%.4f intro=%s zoom=%.4f",
            channel_id, real_bp.id, real_bp.owner_kind, real_bp.owner_id,
            version, nudge_factor, intro_asset_id, zoom_factor,
        )
        return ResolvedBrand(
            brand_profile_id=int(real_bp.id),
            version=version,
            logo_asset_id=real_bp.logo_asset_id,
            logo_local_path=logo_path,
            watermark_text=wm_text,
            watermark_opacity=wm_opacity,
            watermark_position=wm_pos,
            socials_json=socials_json,
            nudge_factor=nudge_factor,
            intro_asset_id=intro_asset_id,
            intro_local_path=intro_path,
            zoom_factor=zoom_factor,
        )

    # ── 4: synthesise from legacy columns ──────────────────────────────
    # Logo precedence: oauth_token.logo_asset_id → channel.logo_asset_id
    logo_asset_id: Optional[int] = None
    if oauth_token is not None and getattr(oauth_token, "logo_asset_id", None):
        logo_asset_id = int(oauth_token.logo_asset_id)
    elif getattr(channel, "logo_asset_id", None):
        logo_asset_id = int(channel.logo_asset_id)

    user_defaults = _resolve_user_v4_defaults(user)

    chan_text = (getattr(channel, "watermark_text", "") or "").strip()
    wm_text = chan_text or (user_defaults.get("watermark_text") or "").strip()

    chan_op = getattr(channel, "watermark_opacity", None)
    if chan_op is not None:
        wm_opacity = _clamp_opacity(chan_op)
    else:
        wm_opacity = _clamp_opacity(user_defaults.get("watermark_opacity", 0.35))

    chan_pos = (getattr(channel, "watermark_position", "") or "").strip()
    wm_pos = _normalize_position(
        chan_pos or user_defaults.get("watermark_position") or "lower-center"
    )

    socials_dict = _resolve_socials(channel)
    socials_json = json.dumps(socials_dict, sort_keys=True, separators=(",", ":"))

    # Intro precedence: a PER-JOB override wins; else mirrors logo
    # (oauth_token.intro_asset_id → channel.intro_asset_id; oauth_token side is
    # future-proofing — only the channel column exists today).
    intro_asset_id: Optional[int] = None
    if job_intro_asset_id and int(job_intro_asset_id) > 0:
        intro_asset_id = int(job_intro_asset_id)
    elif oauth_token is not None and getattr(oauth_token, "intro_asset_id", None):
        intro_asset_id = int(oauth_token.intro_asset_id)
    elif getattr(channel, "intro_asset_id", None):
        intro_asset_id = int(channel.intro_asset_id)

    logo_path = _materialise_logo(db, logo_asset_id)
    intro_path = _materialise_asset(db, intro_asset_id, label="intro video")
    version = _version_hash(
        logo_asset_id, wm_text, wm_opacity, wm_pos, socials_json,
        nudge_factor, intro_asset_id, zoom_factor,
    )

    log.info(
        "brand_resolver: channel_id=%d → SYNTHESISED legacy brand "
        "(logo_asset_id=%s, wm_text=%r, version=%s, nudge=%.4f, intro=%s, zoom=%.4f)",
        channel_id, logo_asset_id, wm_text, version, nudge_factor, intro_asset_id, zoom_factor,
    )
    return ResolvedBrand(
        brand_profile_id=None,
        version=version,
        logo_asset_id=logo_asset_id,
        logo_local_path=logo_path,
        watermark_text=wm_text,
        watermark_opacity=wm_opacity,
        watermark_position=wm_pos,
        socials_json=socials_json,
        nudge_factor=nudge_factor,
        intro_asset_id=intro_asset_id,
        intro_local_path=intro_path,
        zoom_factor=zoom_factor,
    )


__all__ = ["ResolvedBrand", "resolve_brand_profile"]
