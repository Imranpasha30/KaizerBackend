"""Shared watermark stamping — used at upload time (per-channel) and
optionally at render time (per-user fallback).

Architecture:
  * Renders produce a CLEAN MP4 (no baked-in watermark).
  * At publish time the upload worker pulls the destination channel's
    `watermark_text` / `watermark_opacity` / `watermark_position` plus
    its `logo_asset_id` image and stamps them onto a temp copy.
  * If the channel has nothing set we fall back to the user's V4
    defaults so the user can configure one global watermark up front
    and still override per-channel later.

This decouples one render from N destinations — the same source file
can ship to many channels each with its own brand stamp, no re-render.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from pipeline_v4.encoder import video_encoder_args as _enc_args


def _ffmpeg_bin() -> str:
    try:
        from pipeline_core.pipeline import FFMPEG_BIN
        if FFMPEG_BIN:
            return FFMPEG_BIN
    except Exception:
        pass
    return shutil.which("ffmpeg") or "ffmpeg"


def _resolve_channel_logo(channel, db) -> Optional[str]:
    """Resolve the channel's overlay logo with the same precedence the
    rest of the platform uses:

      1. ``oauth_token.logo_asset_id`` — the per-YouTube-account logo
         the user sets via the "change" link on each My Accounts card.
         This is the canonical brand image for a real upload destination.
      2. ``channel.logo_asset_id`` — fallback on the Style Profile row
         (used when there's no OAuth token, e.g. style profiles).

    Returns the local filesystem path to a usable image, or None when
    no logo is configured anywhere. R2-only assets are materialised
    via the existing asset_resolver helper so the upload worker / V4
    download endpoint can both use them transparently.
    """
    if channel is None:
        return None

    # Collect candidate UserAsset IDs in priority order.
    candidate_ids = []
    tok = getattr(channel, "oauth_token", None)
    if tok and getattr(tok, "logo_asset_id", None):
        candidate_ids.append(tok.logo_asset_id)
    if getattr(channel, "logo_asset_id", None):
        candidate_ids.append(channel.logo_asset_id)
    if not candidate_ids:
        return None

    try:
        import models
        # The existing helper materialises cloud-stored assets (R2) into
        # a tempdir so callers always get a local file path back. Falls
        # back to the original file_path when the asset is on disk.
        from asset_resolver import materialize_asset_locally
    except Exception:
        materialize_asset_locally = None

    for aid in candidate_ids:
        try:
            asset = db.query(models.UserAsset).filter(models.UserAsset.id == aid).first()
            if not asset:
                continue
            if materialize_asset_locally is not None:
                local = materialize_asset_locally(asset)
                if local and os.path.isfile(local):
                    return local
            # Direct path fallback in case asset_resolver couldn't load.
            fp = (getattr(asset, "file_path", "") or "").strip()
            if fp and os.path.isfile(fp):
                return fp
        except Exception:
            continue
    return None


def _resolve_user_defaults(user) -> dict:
    if not user or not getattr(user, "v4_defaults", None):
        return {}
    try:
        return json.loads(user.v4_defaults) or {}
    except Exception:
        return {}


def _watermark_overlay_xy(position: str, margin: int = 40) -> tuple[str, str]:
    """Position semantics: corners are reserved for the channel logo
    (always top-right at full opacity). Watermark text uses the mid
    bands so the two never collide.

        upper-center → top third, horizontally centered
        lower-center → bottom third, horizontally centered
        center-left  → vertically centered, near the left edge
        center-right → vertically centered, near the right edge
    """
    p = (position or "lower-center").lower().replace("_", "-")
    if p == "upper-center": return f"(W-w)/2", f"H/4"
    if p == "lower-center": return f"(W-w)/2", f"H*3/4-h/2"
    if p == "center-left":  return f"{margin}", f"(H-h)/2"
    if p == "center-right": return f"W-w-{margin}", f"(H-h)/2"
    # Legacy / fallback — treat corner names as lower-center so older
    # saved settings don't collide with the logo.
    if p in ("top-left", "top-right", "bottom-left", "bottom-right"):
        return f"(W-w)/2", f"H*3/4-h/2"
    return f"(W-w)/2", f"H*3/4-h/2"


def _render_plate_png(
    *,
    text: str,
    logo_path: Optional[str],   # kept for signature compat; ignored — logo is the top-right bug, not part of the watermark
    canvas_w: int,
    canvas_h: int,
    opacity: float,
    out_path: str,
) -> str:
    """Text-only watermark plate. The channel logo lives separately as
    the top-right bug at full opacity (rendered by the existing channel-
    bug helper). This plate is just the user's brand text — sized to
    sit comfortably in the mid bands of the frame without competing
    with the logo or the lower-third strap."""
    _ = logo_path  # intentionally unused; logo is its own overlay
    from PIL import Image, ImageDraw, ImageFont
    # Auto-size to the text — narrow band centered around the text so
    # we don't paint a wide translucent box across half the frame.
    try:
        font_size = max(28, int(canvas_h * 0.045))
        font = ImageFont.load_default()
        try:
            from pipeline_core.pipeline import FONTS_DIR
            candidate = os.path.join(FONTS_DIR, "NotoSans-Bold.ttf")
            if os.path.isfile(candidate):
                font = ImageFont.truetype(candidate, font_size)
        except Exception:
            pass
        bbox = font.getbbox(text[:30])
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        font = ImageFont.load_default()
        text_w, text_h = 240, 40

    pad_x, pad_y = 16, 8
    plate_w = max(40, text_w + 2 * pad_x)
    plate_h = max(20, text_h + 2 * pad_y)
    alpha = max(0, min(255, int(round(opacity * 255))))

    img = Image.new("RGBA", (plate_w, plate_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if text:
        # 1-px shadow for legibility on bright frames; both fade with opacity.
        d.text((pad_x + 1, pad_y + 1), text[:30],
               font=font, fill=(0, 0, 0, alpha))
        d.text((pad_x, pad_y), text[:30],
               font=font, fill=(255, 255, 255, alpha))
    img.save(out_path, "PNG")
    return out_path


def _video_dimensions(path: str) -> tuple[int, int]:
    """Probe (width, height). Defaults to 1920x1080 on failure so the
    plate render still produces something sensible."""
    try:
        import shutil as _sh
        probe = _sh.which("ffprobe") or "ffprobe"
        r = subprocess.run([
            probe, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json", path,
        ], capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            data = json.loads(r.stdout or "{}")
            s = (data.get("streams") or [{}])[0]
            w = int(s.get("width") or 1920)
            h = int(s.get("height") or 1080)
            return w, h
    except Exception:
        pass
    return 1920, 1080


def _read_slot_rects(source_path: str, video_w: int, video_h: int) -> dict:
    """Read the logo/watermark slot rects a custom template recorded next to its master
    (``<master>.slots.json``), scaled from the template canvas to the actual video size.
    Returns {"logo": rect|None, "watermark": rect|None}. {} when there's no side-file (a
    built-in or slot-less template) -> caller uses the default corner positions."""
    try:
        with open((source_path or "") + ".slots.json", encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception:
        return {"logo": None, "watermark": None}
    cv = d.get("canvas") or [video_w, video_h]
    try:
        sx = float(video_w) / float(cv[0] or video_w)
        sy = float(video_h) / float(cv[1] or video_h)
    except Exception:
        sx = sy = 1.0

    def _scl(r):
        if not r:
            return None
        try:
            return {"x": int(round(r["x"] * sx)), "y": int(round(r["y"] * sy)),
                    "w": int(round(r["w"] * sx)), "h": int(round(r["h"] * sy))}
        except Exception:
            return None
    return {"logo": _scl(d.get("logo")), "watermark": _scl(d.get("watermark"))}


def stamp_for_channel(
    *,
    source_path: str,
    channel,
    user,
    db,
    work_dir: Optional[str] = None,
) -> str:
    """Produce a watermarked copy of ``source_path`` using the channel's
    watermark settings, falling back to the user's V4 defaults when the
    channel hasn't set its own. Returns the path to the new file (in a
    fresh tempdir when ``work_dir`` is None) or ``source_path`` itself
    when no watermark is configured anywhere.

    The watermark is applied with `-c:a copy` so audio is bit-identical
    to the source — zero lipsync drift, same trick V4's Step 2 uses.
    """
    if not source_path or not os.path.isfile(source_path):
        return source_path

    # Channel overrides everything; otherwise fall back to user defaults.
    # channel=None is allowed (used by the download path when no specific
    # destination has been picked) — in that case only user defaults apply.
    chan_text = (getattr(channel, "watermark_text", "") or "").strip() if channel else ""
    user_d = _resolve_user_defaults(user)
    text = chan_text or (user_d.get("watermark_text", "") or "")

    chan_op = getattr(channel, "watermark_opacity", None) if channel else None
    opacity = float(chan_op) if chan_op is not None else float(user_d.get("watermark_opacity", 0.35) or 0.35)
    opacity = max(0.05, min(1.0, opacity))

    chan_pos = (getattr(channel, "watermark_position", "") or "").strip() if channel else ""
    position = chan_pos or (user_d.get("watermark_position", "lower-center") or "lower-center")

    # Per-channel logo first; when no channel was named (the "User
    # defaults" download path) fall back to ANY logo this user has
    # configured on any of their YouTube accounts so the saved file
    # still carries a brand image instead of being unbranded.
    logo_path = _resolve_channel_logo(channel, db) if channel else None
    if not logo_path and user is not None:
        try:
            import models
            tok = (
                db.query(models.OAuthToken)
                  .join(models.Channel, models.OAuthToken.channel_id == models.Channel.id)
                  .filter(models.Channel.user_id == user.id,
                          models.OAuthToken.logo_asset_id.isnot(None))
                  .order_by(models.OAuthToken.connected_at.desc())
                  .first()
            )
            if tok and tok.channel:
                logo_path = _resolve_channel_logo(tok.channel, db)
        except Exception:
            pass
    if not (text or logo_path):
        # Nothing to stamp — return the original untouched.
        return source_path

    work_dir = work_dir or tempfile.mkdtemp(prefix="kaizer_wm_")
    os.makedirs(work_dir, exist_ok=True)
    canvas_w, canvas_h = _video_dimensions(source_path)

    # Build the ffmpeg filtergraph. Two independent overlays:
    #   1) Channel logo bug — ALWAYS top-right, full opacity.
    #      Width is ~10% of canvas so the bug stays small. Skipped when
    #      the channel has no logo configured.
    #   2) Text watermark — user-positioned mid-band overlay, alpha
    #      driven by the per-channel opacity slider.
    inputs = ["-i", source_path]
    chain_parts: list[str] = []
    last_label = "0:v"

    # Custom-template slot rects (logo/watermark) recorded next to the master. When the
    # template MARKS a spot, the per-channel logo/watermark goes THERE; otherwise the
    # default corner/band below. Operator's rule: "template says where -> there; else default."
    _slots = _read_slot_rects(source_path, canvas_w, canvas_h)
    _logo_rect = _slots.get("logo")
    _wm_rect = _slots.get("watermark")

    # KAIZER_CLEAN_MASTER (Decision 1): when "1", SKIP the default top-right logo bug (the
    # Phase 2 Branding Worker owns that pass — baking here would double-stamp). BUT a
    # template's EXPLICIT logo slot is the designated spot, so we still stamp into it.
    _clean_master = os.environ.get("KAIZER_CLEAN_MASTER", "0").strip() == "1"
    if logo_path and (_logo_rect or not _clean_master):
        from PIL import Image as _PI
        try:
            with _PI.open(logo_path) as _logo:
                _logo = _logo.convert("RGBA")
                if _logo_rect:
                    # FIT the logo inside the template's marked slot (contain), centered.
                    rw, rh = max(8, _logo_rect["w"]), max(8, _logo_rect["h"])
                    ratio = min(rw / max(1, _logo.width), rh / max(1, _logo.height))
                    bug_w = max(8, int(_logo.width * ratio))
                    bug_h = max(8, int(_logo.height * ratio))
                    ox = _logo_rect["x"] + (rw - bug_w) // 2
                    oy = _logo_rect["y"] + (rh - bug_h) // 2
                    xy = f"x={ox}:y={oy}"
                else:
                    bug_h = max(48, int(canvas_h * 0.10))
                    ratio = bug_h / max(1, _logo.height)
                    bug_w = max(48, int(_logo.width * ratio))
                    margin = max(20, int(canvas_w * 0.015))
                    xy = f"x=W-w-{margin}:y={margin}"
                bug_png = os.path.join(work_dir, "_wm_bug.png")
                _logo.resize((bug_w, bug_h), _PI.LANCZOS).save(bug_png, "PNG")
            inputs += ["-loop", "1", "-i", bug_png]
            chain_parts.append(f"[{last_label}][1:v]overlay={xy}:format=auto[bug]")
            last_label = "bug"
        except Exception as exc:
            print(f"[watermark] bug overlay skipped: {exc}", flush=True)

    # Text watermark — at the template's watermark slot if marked, else the chosen mid band.
    if text:
        plate_path = os.path.join(work_dir, "_wm_plate.png")
        _render_plate_png(
            text=text, logo_path=None,
            canvas_w=canvas_w, canvas_h=canvas_h,
            opacity=opacity, out_path=plate_path,
        )
        plate_idx = len(inputs) // 2  # number of -i pairs so far
        inputs += ["-loop", "1", "-i", plate_path]
        if _wm_rect:
            # Center the text plate inside the template's marked watermark slot (w/h are the
            # plate's own dims — ffmpeg overlay refs — so it centers regardless of plate size).
            rx, ry, rw, rh = _wm_rect["x"], _wm_rect["y"], _wm_rect["w"], _wm_rect["h"]
            xy = f"x={rx}+({rw}-w)/2:y={ry}+({rh}-h)/2"
        else:
            wx, wy = _watermark_overlay_xy(position)
            xy = f"x={wx}:y={wy}"
        chain_parts.append(
            f"[{last_label}][{plate_idx}:v]overlay={xy}:format=auto[outv]"
        )
        last_label = "outv"

    if not chain_parts:
        return source_path

    out_path = os.path.join(work_dir, "stamped_" + Path(source_path).name)
    cmd = [_ffmpeg_bin(), "-y", "-v", "error"] + inputs + [
        "-filter_complex", ";".join(chain_parts),
        "-map", f"[{last_label}]", "-map", "0:a?",
        *_enc_args(crf=20, preset_hint="veryfast"),
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-shortest", "-movflags", "+faststart",
        out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not os.path.isfile(out_path):
        print(f"[watermark] stamp failed (rc={r.returncode}): "
              f"{(r.stderr or '')[-300:]}", flush=True)
        return source_path
    return out_path


# ── SEO description footer builders ──────────────────────────────────

_SOCIAL_LABELS = {
    "youtube":   "YouTube",
    "instagram": "Instagram",
    "twitter":   "X / Twitter",
    "x":         "X / Twitter",
    "facebook":  "Facebook",
    "tiktok":    "TikTok",
    "threads":   "Threads",
    "linkedin":  "LinkedIn",
    "whatsapp":  "WhatsApp",
    "telegram":  "Telegram",
    "website":   "Website",
    "email":     "Contact",
}

# Canonical URL templates per platform ({h} = cleaned handle). We emit
# FULL https URLs because YouTube only auto-links full URLs in a
# description — that's what makes the social links actually clickable
# (the way YouTube renders channel links). Platforms NOT listed here
# (website/email/whatsapp/unknown) fall back to showing the raw value.
_SOCIAL_URL = {
    "youtube":   "https://www.youtube.com/@{h}",
    "instagram": "https://www.instagram.com/{h}",
    "twitter":   "https://x.com/{h}",
    "x":         "https://x.com/{h}",
    "facebook":  "https://www.facebook.com/{h}",
    "tiktok":    "https://www.tiktok.com/@{h}",
    "threads":   "https://www.threads.net/@{h}",
    "linkedin":  "https://www.linkedin.com/in/{h}",
    "telegram":  "https://t.me/{h}",
}


def _clean_handle(val: str) -> str:
    """Reduce any stored social value to a bare handle.

    Accepts a full URL (returned as-is), '@handle', 'handle', or the
    legacy 'platform@handle' junk we have in the DB (e.g.
    'isntagram@autowalla'). We always take the part AFTER the last '@'
    so a mistyped/prefixed platform name can't pollute the handle, then
    strip slashes/spaces. The platform is decided by the dict KEY, never
    by this value — so typos in the value can't mis-route the link.
    """
    v = (val or "").strip()
    if not v:
        return ""
    if v.lower().startswith(("http://", "https://")):
        return v  # already a clickable URL — pass through untouched
    if "@" in v:
        v = v.rsplit("@", 1)[-1]
    return v.strip().strip("/").strip()


def social_url(platform: str, val: str) -> str:
    """Build a clickable URL for one {platform: value} pair.

    Full URLs pass through. Known platforms become canonical https URLs
    from the bare handle. WhatsApp only links when the handle is a phone
    number (wa.me requires digits). Unknown/website/email show the raw
    cleaned value so nothing the user typed silently disappears.
    """
    p = (platform or "").strip().lower()
    handle = _clean_handle(val)
    if not handle:
        return ""
    if handle.lower().startswith(("http://", "https://")):
        return handle
    if p == "whatsapp":
        # WhatsApp has no username URL — a link needs a phone number
        # (wa.me/<digits>) or a full chat-invite URL. A bare username
        # (e.g. "kaizer30") can't be linked, so OMIT it rather than emit
        # a dead plain-text line. Full URLs already returned above.
        digits = "".join(c for c in handle if c.isdigit())
        return f"https://wa.me/{digits}" if len(digits) >= 8 else ""
    tmpl = _SOCIAL_URL.get(p)
    if tmpl:
        return tmpl.format(h=handle)
    # Unknown platform (website/email/custom): only keep it if the user
    # gave a real URL — otherwise omit so every emitted line is a link.
    return handle if handle.lower().startswith(("http://", "https://")) else ""


def build_socials_footer(socials: dict) -> str:
    """Format {platform: handle_or_url} into a multi-line description
    footer of CLICKABLE links. Each per-channel handle is normalized to
    its canonical https URL so YouTube renders it as a real link.
    Returns '' when nothing usable is set."""
    if not socials or not isinstance(socials, dict):
        return ""
    lines: list[str] = []
    for key, val in socials.items():
        url = social_url(key, val)
        if not url:
            continue
        label = _SOCIAL_LABELS.get((key or "").lower(), (key or "").title())
        lines.append(f"{label}: {url}")
    if not lines:
        return ""
    return "Follow us:\n" + "\n".join(lines)
